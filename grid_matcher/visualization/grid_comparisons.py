# grid_matcher/visualization/grid_comparisons.py
"""Grid comparison visualization functions."""

import pandas as pd
import numpy as np
import os
from pathlib import Path
import json
import plotly.graph_objects as go
import plotly.subplots as sp


def extract_buses_from_lines(gdf):
    """Extract bus information from line data."""
    # Try to find bus columns
    from_cols = ['from_node', 'from_bus', 'bus0', 'from']
    to_cols = ['to_node', 'to_bus', 'bus1', 'to']

    from_col = next((col for col in from_cols if col in gdf.columns), None)
    to_col = next((col for col in to_cols if col in gdf.columns), None)

    if not from_col or not to_col:
        print(f"Could not find bus columns in DataFrame. Available columns: {gdf.columns.tolist()}")
        return pd.DataFrame()

    # Extract all bus IDs from both columns and combine into a single Series
    all_buses = pd.concat([
        gdf[from_col].dropna(),
        gdf[to_col].dropna()
    ])

    # Get unique bus IDs
    unique_buses = all_buses.unique()
    print(f"Extracted {len(unique_buses)} unique buses")

    # Create DataFrame with bus IDs
    buses_df = pd.DataFrame({'bus_id': unique_buses})

    # Get voltage column name
    voltage_col = next((col for col in ['v_nom', 'voltage'] if col in gdf.columns), None)

    if voltage_col:
        # First approach: extract voltage from bus ID if it contains voltage information
        if buses_df['bus_id'].dtype == 'object':  # Only try string operations on string data
            buses_df['voltage_from_id'] = buses_df['bus_id'].apply(
                lambda x: x.split('-')[-1] if isinstance(x, str) and '-' in x else None
            )
        else:
            buses_df['voltage_from_id'] = None

        # Second approach: determine voltage from connected lines
        bus_voltages = {}
        for bus in unique_buses:
            # Find all lines connected to this bus
            connected_lines = gdf[(gdf[from_col] == bus) | (gdf[to_col] == bus)]
            if not connected_lines.empty:
                try:
                    # Get most common voltage value, safely handling empty results
                    voltage_values = connected_lines[voltage_col].dropna()
                    if not voltage_values.empty:
                        voltage = voltage_values.value_counts().index[0]
                        bus_voltages[bus] = voltage
                except Exception as e:
                    print(f"Error determining voltage for bus {bus}: {e}")
                    continue

        # Add voltage from connected lines
        buses_df['voltage_from_lines'] = buses_df['bus_id'].map(bus_voltages)

        # Use best available voltage (prefer from lines, fallback to ID)
        buses_df['voltage'] = buses_df['voltage_from_lines'].fillna(buses_df['voltage_from_id'])

        # Convert voltage to numeric where possible for proper filtering later
        try:
            buses_df['voltage'] = pd.to_numeric(buses_df['voltage'], errors='ignore')
        except:
            pass

        # Standardize 380kV to 400kV (works for both string and numeric values)
        buses_df['voltage'] = buses_df['voltage'].apply(
            lambda v: 400 if v == 380 or v == '380' or v == 380.0 else
            220 if v == 220 or v == '220' or v == 220.0 else v
        )

        # Handle other voltage values that should map to our standard voltage levels
        voltage_mapping = {
            '380.0': 400, 380.0: 400,
            '220.0': 220, 220.0: 220,
            '225.0': 220, 225.0: 220  # Sometimes 225kV is used instead of 220kV
        }

        buses_df['voltage'] = buses_df['voltage'].apply(
            lambda v: voltage_mapping.get(v, v) if v in voltage_mapping else v
        )

        # Filter to include only 220kV and 400kV buses for comparison purposes
        buses_df = buses_df[buses_df['voltage'].isin([220, 400, '220', '400'])]

        # Convert voltage to string for consistency in grouping and reporting
        buses_df['voltage'] = buses_df['voltage'].astype(str)

        # Count buses by voltage
        voltage_counts = buses_df.groupby('voltage').size()
        print(f"Bus counts by voltage: {voltage_counts.to_dict()}")

    return buses_df


def compare_line_length_by_voltage(pypsa_df, jao_df):
    """Compare line lengths between PyPSA and JAO datasets by voltage level."""
    # Get voltage column names based on what's available
    pypsa_voltage_col = next((col for col in ['v_nom', 'voltage'] if col in pypsa_df.columns), None)
    jao_voltage_col = next((col for col in ['v_nom', 'voltage'] if col in jao_df.columns), None)

    # Get length column names
    pypsa_length_col = next((col for col in ['length_km', 'length'] if col in pypsa_df.columns), None)
    jao_length_col = next((col for col in ['length_km', 'length'] if col in jao_df.columns), None)

    if not all([pypsa_voltage_col, jao_voltage_col, pypsa_length_col, jao_length_col]):
        print(f"Warning: Missing required columns for length comparison")
        print(f"PyPSA columns: {pypsa_df.columns.tolist()}")
        print(f"JAO columns: {jao_df.columns.tolist()}")
        return pd.DataFrame()

    # Handle voltage standardization - treat 380kV as 400kV
    pypsa_df = pypsa_df.copy()
    jao_df = jao_df.copy()

    # Convert voltage to string for consistency
    pypsa_df[pypsa_voltage_col] = pypsa_df[pypsa_voltage_col].astype(str)
    jao_df[jao_voltage_col] = jao_df[jao_voltage_col].astype(str)

    # Standardize 380kV to 400kV
    pypsa_df[pypsa_voltage_col] = pypsa_df[pypsa_voltage_col].apply(
        lambda v: "400" if v == "380" else v)
    jao_df[jao_voltage_col] = jao_df[jao_voltage_col].apply(
        lambda v: "400" if v == "380" else v)

    # Filter to only include AC lines with standard voltages (220kV and 400kV)
    std_voltages = ["220", "400"]
    pypsa_filtered = pypsa_df[pypsa_df[pypsa_voltage_col].isin(std_voltages)]
    jao_filtered = jao_df[jao_df[jao_voltage_col].isin(std_voltages)]

    print(f"PyPSA lines after filtering for standard voltages: {len(pypsa_filtered)} of {len(pypsa_df)}")
    print(f"JAO lines after filtering for standard voltages: {len(jao_filtered)} of {len(jao_df)}")

    # Group by voltage level and sum lengths
    pypsa_by_voltage = pypsa_filtered.groupby(pypsa_voltage_col)[pypsa_length_col].sum() / 1000  # Convert to km
    jao_by_voltage = jao_filtered.groupby(jao_voltage_col)[jao_length_col].sum() / 1000  # Convert to km

    # Create comparison dataframe
    voltage_levels = sorted(set(pypsa_by_voltage.index).union(jao_by_voltage.index))
    comparison = pd.DataFrame(index=voltage_levels,
                              columns=['PyPSA (km)', 'JAO (km)', 'Difference (km)', 'Difference (%)'])

    for v in voltage_levels:
        pypsa_len = pypsa_by_voltage.get(v, 0)
        jao_len = jao_by_voltage.get(v, 0)
        diff = jao_len - pypsa_len
        pct_diff = (diff / pypsa_len * 100) if pypsa_len != 0 else float('inf')

        comparison.loc[v] = [pypsa_len, jao_len, diff, f"{pct_diff:.2f}%"]

    # Add totals
    comparison.loc['Total'] = [
        pypsa_by_voltage.sum(),
        jao_by_voltage.sum(),
        jao_by_voltage.sum() - pypsa_by_voltage.sum(),
        f"{(jao_by_voltage.sum() - pypsa_by_voltage.sum()) / pypsa_by_voltage.sum() * 100:.2f}%"
    ]

    return comparison


def compare_substations_count(pypsa_df, jao_df):
    """Count and compare substations/nodes by voltage level."""
    # Get voltage column names based on what's available
    pypsa_voltage_col = next((col for col in ['v_nom', 'voltage'] if col in pypsa_df.columns), None)
    jao_voltage_col = next((col for col in ['v_nom', 'voltage'] if col in jao_df.columns), None)

    if not all([pypsa_voltage_col, jao_voltage_col]):
        print("Warning: Missing required voltage columns for substation comparison")
        return pd.DataFrame()

    # Handle voltage standardization - treat 380kV as 400kV
    pypsa_df = pypsa_df.copy()
    jao_df = jao_df.copy()

    # Convert voltage to string for consistency
    pypsa_df[pypsa_voltage_col] = pypsa_df[pypsa_voltage_col].astype(str)
    jao_df[jao_voltage_col] = jao_df[jao_voltage_col].astype(str)

    # Standardize 380kV to 400kV
    pypsa_df[pypsa_voltage_col] = pypsa_df[pypsa_voltage_col].apply(
        lambda v: "400" if v == "380" else v)
    jao_df[jao_voltage_col] = jao_df[jao_voltage_col].apply(
        lambda v: "400" if v == "380" else v)

    # Filter to only include standard voltages (220kV and 400kV)
    std_voltages = ["220", "400"]
    pypsa_filtered = pypsa_df[pypsa_df[pypsa_voltage_col].isin(std_voltages)]
    jao_filtered = jao_df[jao_df[jao_voltage_col].isin(std_voltages)]

    # Count substations/buses by voltage level
    pypsa_substations = pypsa_filtered.groupby(pypsa_voltage_col).size()
    jao_substations = jao_filtered.groupby(jao_voltage_col).size()

    # Create comparison dataframe
    voltage_levels = sorted(set(pypsa_substations.index).union(jao_substations.index))
    comparison = pd.DataFrame(index=voltage_levels,
                              columns=['PyPSA Count', 'JAO Count', 'Difference', 'Difference (%)'])

    for v in voltage_levels:
        pypsa_count = pypsa_substations.get(v, 0)
        jao_count = jao_substations.get(v, 0)
        diff = jao_count - pypsa_count
        pct_diff = (diff / pypsa_count * 100) if pypsa_count != 0 else float('inf')

        comparison.loc[v] = [pypsa_count, jao_count, diff, f"{pct_diff:.2f}%"]

    # Add totals
    comparison.loc['Total'] = [
        pypsa_substations.sum(),
        jao_substations.sum(),
        jao_substations.sum() - pypsa_substations.sum(),
        f"{(jao_substations.sum() - pypsa_substations.sum()) / pypsa_substations.sum() * 100:.2f}%"
    ]

    return comparison


def compare_circuits_count(pypsa_df, jao_df):
    """Count and compare circuits/lines by voltage level."""
    # Get voltage column names based on what's available
    pypsa_voltage_col = next((col for col in ['v_nom', 'voltage'] if col in pypsa_df.columns), None)
    jao_voltage_col = next((col for col in ['v_nom', 'voltage'] if col in jao_df.columns), None)

    if not all([pypsa_voltage_col, jao_voltage_col]):
        print("Warning: Missing required voltage columns for circuit comparison")
        return pd.DataFrame()

    # Handle voltage standardization - treat 380kV as 400kV
    pypsa_df = pypsa_df.copy()
    jao_df = jao_df.copy()

    # Convert voltage to string for consistency
    pypsa_df[pypsa_voltage_col] = pypsa_df[pypsa_voltage_col].astype(str)
    jao_df[jao_voltage_col] = jao_df[jao_voltage_col].astype(str)

    # Standardize 380kV to 400kV
    pypsa_df[pypsa_voltage_col] = pypsa_df[pypsa_voltage_col].apply(
        lambda v: "400" if v == "380" else v)
    jao_df[jao_voltage_col] = jao_df[jao_voltage_col].apply(
        lambda v: "400" if v == "380" else v)

    # Filter to only include standard voltages (220kV and 400kV)
    std_voltages = ["220", "400"]
    pypsa_filtered = pypsa_df[pypsa_df[pypsa_voltage_col].isin(std_voltages)]
    jao_filtered = jao_df[jao_df[jao_voltage_col].isin(std_voltages)]

    # Count lines by voltage level
    pypsa_lines = pypsa_filtered.groupby(pypsa_voltage_col).size()
    jao_lines = jao_filtered.groupby(jao_voltage_col).size()

    # Create comparison dataframe
    voltage_levels = sorted(set(pypsa_lines.index).union(jao_lines.index))
    comparison = pd.DataFrame(index=voltage_levels,
                              columns=['PyPSA Lines', 'JAO Lines', 'Difference', 'Difference (%)'])

    for v in voltage_levels:
        pypsa_count = pypsa_lines.get(v, 0)
        jao_count = jao_lines.get(v, 0)
        diff = jao_count - pypsa_count
        pct_diff = (diff / pypsa_count * 100) if pypsa_count != 0 else float('inf')

        comparison.loc[v] = [pypsa_count, jao_count, diff, f"{pct_diff:.2f}%"]

    # Add totals
    comparison.loc['Total'] = [
        pypsa_lines.sum(),
        jao_lines.sum(),
        jao_lines.sum() - pypsa_lines.sum(),
        f"{(jao_lines.sum() - pypsa_lines.sum()) / pypsa_lines.sum() * 100:.2f}%"
    ]

    return comparison


def create_comparison_html(line_df, substation_df, circuit_df, output_path):
    """Create HTML visualization with all three comparisons."""
    # Check if we have data
    if line_df.empty or substation_df.empty or circuit_df.empty:
        print("Warning: One or more comparison DataFrames are empty!")

    # Extract voltage levels (excluding the 'Total' row)
    line_df_filtered = line_df.drop('Total', errors='ignore')
    substation_df_filtered = substation_df.drop('Total', errors='ignore')
    circuit_df_filtered = circuit_df.drop('Total', errors='ignore')

    # Get totals for summary
    line_total_row = line_df.loc['Total'] if 'Total' in line_df.index else pd.Series({
        'JAO (km)': 0, 'PyPSA (km)': 0, 'Difference (km)': 0, 'Difference (%)': '0.00%'
    })

    substation_total_row = substation_df.loc['Total'] if 'Total' in substation_df.index else pd.Series({
        'JAO Count': 0, 'PyPSA Count': 0, 'Difference': 0, 'Difference (%)': '0.00%'
    })

    circuit_total_row = circuit_df.loc['Total'] if 'Total' in circuit_df.index else pd.Series({
        'JAO Lines': 0, 'PyPSA Lines': 0, 'Difference': 0, 'Difference (%)': '0.00%'
    })

    # Prepare data for Plotly charts - reset_index to make voltage a column
    line_data = line_df_filtered.reset_index().rename(columns={'index': 'voltage'})
    substation_data = substation_df_filtered.reset_index().rename(columns={'index': 'voltage'})
    circuit_data = circuit_df_filtered.reset_index().rename(columns={'index': 'voltage'})

    # Convert percentage strings to float values for plotting
    for df in [line_data, substation_data, circuit_data]:
        if not df.empty and 'Difference (%)' in df.columns:
            df['Difference (%)'] = df['Difference (%)'].str.rstrip('%').astype(float)

    # Extract values for KPI cards - safely handle conversion
    try:
        jao_total_length = float(line_total_row['JAO (km)'])
        pypsa_total_length = float(line_total_row['PyPSA (km)'])
        length_diff_pct = float(line_total_row['Difference (%)'].rstrip('%')) if isinstance(
            line_total_row['Difference (%)'], str) else line_total_row['Difference (%)']
    except (ValueError, AttributeError) as e:
        print(f"Error converting line lengths: {e}")
        jao_total_length = 0
        pypsa_total_length = 0
        length_diff_pct = 0

    try:
        jao_total_substations = int(substation_total_row['JAO Count'])
        pypsa_total_substations = int(substation_total_row['PyPSA Count'])
        substation_diff_pct = float(substation_total_row['Difference (%)'].rstrip('%')) if isinstance(
            substation_total_row['Difference (%)'], str) else substation_total_row['Difference (%)']
    except (ValueError, AttributeError) as e:
        print(f"Error converting substation counts: {e}")
        jao_total_substations = 0
        pypsa_total_substations = 0
        substation_diff_pct = 0

    try:
        jao_total_circuits = int(circuit_total_row['JAO Lines'])
        pypsa_total_circuits = int(circuit_total_row['PyPSA Lines'])
        circuit_diff_pct = float(circuit_total_row['Difference (%)'].rstrip('%')) if isinstance(
            circuit_total_row['Difference (%)'], str) else circuit_total_row['Difference (%)']
    except (ValueError, AttributeError) as e:
        print(f"Error converting circuit counts: {e}")
        jao_total_circuits = 0
        pypsa_total_circuits = 0
        circuit_diff_pct = 0

    # Create HTML content
    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Grid Comparison: JAO vs PyPSA</title>
        <script src="https://cdn.plot.ly/plotly-2.24.1.min.js"></script>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f8f9fa;
            }}
            h1 {{
                color: #2c3e50;
                text-align: center;
                margin-bottom: 30px;
                border-bottom: 2px solid #ddd;
                padding-bottom: 15px;
            }}
            .dashboard-container {{
                max-width: 1400px;
                margin: 0 auto;
                display: flex;
                flex-direction: column;
                gap: 30px;
            }}
            .kpi-container {{
                display: flex;
                flex-wrap: wrap;
                gap: 20px;
                justify-content: space-between;
                margin-bottom: 20px;
            }}
            .kpi-card {{
                flex: 1;
                min-width: 280px;
                background-color: white;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                padding: 20px;
                text-align: center;
            }}
            .kpi-title {{
                font-size: 16px;
                margin-bottom: 10px;
                color: #666;
            }}
            .kpi-value {{
                font-size: 24px;
                font-weight: bold;
                margin-bottom: 5px;
            }}
            .kpi-comparison {{
                font-size: 14px;
            }}
            .kpi-diff-positive {{
                color: #e74c3c;
            }}
            .kpi-diff-negative {{
                color: #2ecc71;
            }}
            .kpi-diff-neutral {{
                color: #3498db;
            }}
            .chart-container {{
                background-color: white;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                padding: 20px;
                margin-bottom: 20px;
            }}
            .chart {{
                height: 400px;
                width: 100%;
            }}
            h2 {{
                color: #2c3e50;
                border-bottom: 1px solid #ddd;
                padding-bottom: 10px;
                margin-top: 0;
            }}
            .comparison-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
                gap: 25px;
                margin-bottom: 30px;
            }}
            .data-source {{
                font-size: 12px;
                color: #666;
                text-align: right;
                margin-top: 10px;
            }}
            .summary-table {{
                width: 100%;
                border-collapse: collapse;
                margin-bottom: 20px;
            }}
            .summary-table th, .summary-table td {{
                border: 1px solid #ddd;
                padding: 8px;
                text-align: right;
            }}
            .summary-table th:first-child, .summary-table td:first-child {{
                text-align: left;
            }}
            .summary-table th {{
                background-color: #f2f2f2;
            }}
            .summary-table tr:nth-child(even) {{
                background-color: #f9f9f9;
            }}
            .notes {{
                font-size: 0.9em;
                font-style: italic;
                color: #666;
                margin-top: 5px;
                margin-bottom: 15px;
            }}
            .error-message {{
                color: red;
                padding: 20px;
                text-align: center;
                font-weight: bold;
            }}
        </style>
    </head>
    <body>
        <div class="dashboard-container">
            <h1>JAO vs PyPSA Grid Comparison</h1>

            <div class="notes">
                Notes: This comparison focuses on the standard voltage levels (220kV and 400kV) in both datasets. 
                380kV lines are considered equivalent to 400kV lines for comparison purposes.
            </div>

            <!-- KPI Section -->
            <div class="kpi-container">
                <!-- Line Length KPI -->
                <div class="kpi-card">
                    <div class="kpi-title">Total Line Length</div>
                    <div class="kpi-value">{pypsa_total_length:.0f} km</div>
                    <div class="kpi-comparison">
                        JAO: {jao_total_length:.0f} km 
                        (<span class="kpi-diff-{'positive' if length_diff_pct > 0 else 'negative' if length_diff_pct < 0 else 'neutral'}">{length_diff_pct:.2f}%</span>)
                    </div>
                </div>

                <!-- Substation Count KPI -->
                <div class="kpi-card">
                    <div class="kpi-title">Total Substations</div>
                    <div class="kpi-value">{pypsa_total_substations}</div>
                    <div class="kpi-comparison">
                        JAO: {jao_total_substations} 
                        (<span class="kpi-diff-{'positive' if substation_diff_pct > 0 else 'negative' if substation_diff_pct < 0 else 'neutral'}">{substation_diff_pct:.2f}%</span>)
                    </div>
                </div>

                <!-- Circuit Count KPI -->
                <div class="kpi-card">
                    <div class="kpi-title">Total Circuits</div>
                    <div class="kpi-value">{pypsa_total_circuits}</div>
                    <div class="kpi-comparison">
                        JAO: {jao_total_circuits} 
                        (<span class="kpi-diff-{'positive' if circuit_diff_pct > 0 else 'negative' if circuit_diff_pct < 0 else 'neutral'}">{circuit_diff_pct:.2f}%</span>)
                    </div>
                </div>
            </div>

            <!-- Summary Tables Section -->
            <div class="chart-container">
                <h2>Line Length Comparison by Voltage Level</h2>
                <table class="summary-table">
                    <thead>
                        <tr>
                            <th>Voltage Level</th>
                            <th>JAO (km)</th>
                            <th>PyPSA (km)</th>
                            <th>Difference (km)</th>
                            <th>Difference (%)</th>
                        </tr>
                    </thead>
                    <tbody>
                        {''.join([f"<tr><td>{idx} kV</td><td>{row['JAO (km)']:.1f}</td><td>{row['PyPSA (km)']:.1f}</td><td>{row['Difference (km)']:.1f}</td><td>{row['Difference (%)']}</td></tr>" for idx, row in line_df.iterrows()])}
                    </tbody>
                </table>
                <div id="lineChart" class="chart"></div>
            </div>

            <!-- Comparison Section -->
            <div class="comparison-grid">
                <!-- Substation Comparison -->
                <div class="chart-container">
                    <h2>Substation Count Comparison</h2>
                    <table class="summary-table">
                        <thead>
                            <tr>
                                <th>Voltage Level</th>
                                <th>JAO Count</th>
                                <th>PyPSA Count</th>
                                <th>Difference</th>
                                <th>Difference (%)</th>
                            </tr>
                        </thead>
                        <tbody>
                            {''.join([f"<tr><td>{idx} kV</td><td>{row['JAO Count']}</td><td>{row['PyPSA Count']}</td><td>{row['Difference']}</td><td>{row['Difference (%)']}</td></tr>" for idx, row in substation_df.iterrows()])}
                        </tbody>
                    </table>
                    <div id="substationChart" class="chart"></div>
                </div>

                <!-- Circuit Comparison -->
                <div class="chart-container">
                    <h2>Circuit Count Comparison</h2>
                    <table class="summary-table">
                        <thead>
                            <tr>
                                <th>Voltage Level</th>
                                <th>JAO Lines</th>
                                <th>PyPSA Lines</th>
                                <th>Difference</th>
                                <th>Difference (%)</th>
                            </tr>
                        </thead>
                        <tbody>
                            {''.join([f"<tr><td>{idx} kV</td><td>{row['JAO Lines']}</td><td>{row['PyPSA Lines']}</td><td>{row['Difference']}</td><td>{row['Difference (%)']}</td></tr>" for idx, row in circuit_df.iterrows()])}
                        </tbody>
                    </table>
                    <div id="circuitChart" class="chart"></div>
                </div>
            </div>

            <div class="data-source">Data source: JAO and PyPSA network datasets</div>
        </div>

        <script>
            // Line Length Chart
            const lineData = {json.dumps(line_data.to_dict('records'))};

            // Substation Count Chart
            const substationData = {json.dumps(substation_data.to_dict('records'))};

            // Circuit Count Chart
            const circuitData = {json.dumps(circuit_data.to_dict('records'))};

            try {{
                // Create Line Length Bar Chart
                if (lineData && lineData.length > 0) {{
                    const lineTraces = [
                        {{
                            x: lineData.map(d => d.voltage + ' kV'),
                            y: lineData.map(d => d['JAO (km)']),
                            name: 'JAO',
                            type: 'bar',
                            marker: {{ color: 'rgba(58, 71, 80, 0.7)' }}
                        }},
                        {{
                            x: lineData.map(d => d.voltage + ' kV'),
                            y: lineData.map(d => d['PyPSA (km)']),
                            name: 'PyPSA',
                            type: 'bar',
                            marker: {{ color: 'rgba(246, 78, 139, 0.7)' }}
                        }}
                    ];

                    const lineLayout = {{
                        barmode: 'group',
                        yaxis: {{ title: 'Length (km)' }},
                        legend: {{
                            x: 0.5,
                            y: 1.1,
                            orientation: 'h',
                            xanchor: 'center'
                        }},
                        margin: {{ l: 50, r: 20, b: 40, t: 30, pad: 4 }}
                    }};

                    Plotly.newPlot('lineChart', lineTraces, lineLayout);
                }} else {{
                    document.getElementById('lineChart').innerHTML = '<div class="error-message">No line length data available for chart</div>';
                }}

                // Create Substation Bar Chart
                if (substationData && substationData.length > 0) {{
                    const substationTraces = [
                        {{
                            x: substationData.map(d => d.voltage + ' kV'),
                            y: substationData.map(d => d['JAO Count']),
                            name: 'JAO',
                            type: 'bar',
                            marker: {{ color: 'rgba(58, 71, 80, 0.7)' }}
                        }},
                        {{
                            x: substationData.map(d => d.voltage + ' kV'),
                            y: substationData.map(d => d['PyPSA Count']),
                            name: 'PyPSA',
                            type: 'bar',
                            marker: {{ color: 'rgba(246, 78, 139, 0.7)' }}
                        }}
                    ];

                    const substationLayout = {{
                        barmode: 'group',
                        yaxis: {{ title: 'Number of Substations' }},
                        legend: {{
                            x: 0.5,
                            y: 1.1,
                            orientation: 'h',
                            xanchor: 'center'
                        }},
                        margin: {{ l: 50, r: 20, b: 40, t: 30, pad: 4 }}
                    }};

                    Plotly.newPlot('substationChart', substationTraces, substationLayout);
                }} else {{
                    document.getElementById('substationChart').innerHTML = '<div class="error-message">No substation data available for chart</div>';
                }}

                // Create Circuit Bar Chart
                if (circuitData && circuitData.length > 0) {{
                    const circuitTraces = [
                        {{
                            x: circuitData.map(d => d.voltage + ' kV'),
                            y: circuitData.map(d => d['JAO Lines']),
                            name: 'JAO',
                            type: 'bar',
                            marker: {{ color: 'rgba(58, 71, 80, 0.7)' }}
                        }},
                        {{
                            x: circuitData.map(d => d.voltage + ' kV'),
                            y: circuitData.map(d => d['PyPSA Lines']),
                            name: 'PyPSA',
                            type: 'bar',
                            marker: {{ color: 'rgba(246, 78, 139, 0.7)' }}
                        }}
                    ];

                    const circuitLayout = {{
                        barmode: 'group',
                        yaxis: {{ title: 'Number of Circuits' }},
                        legend: {{
                            x: 0.5,
                            y: 1.1,
                            orientation: 'h',
                            xanchor: 'center'
                        }},
                        margin: {{ l: 50, r: 20, b: 40, t: 30, pad: 4 }}
                    }};

                    Plotly.newPlot('circuitChart', circuitTraces, circuitLayout);
                }} else {{
                    document.getElementById('circuitChart').innerHTML = '<div class="error-message">No circuit data available for chart</div>';
                }}

            }} catch (e) {{
                console.error("Error rendering charts:", e);
                document.body.innerHTML += '<div class="error-message">Error rendering charts: ' + e.message + '</div>';
            }}
        </script>
    </body>
    </html>
    """

    # Write the HTML to file
    with open(output_path, 'w') as f:
        f.write(html)

    print(f"Comparison visualization saved to: {output_path}")
    return output_path


def generate_grid_comparisons(jao_gdf, pypsa_gdf, output_dir="output"):
    """
    Generate all grid comparison visualizations by extracting bus data from lines.

    Parameters:
    -----------
    jao_gdf : GeoDataFrame
        JAO lines data
    pypsa_gdf : GeoDataFrame
        PyPSA lines data
    output_dir : str
        Directory to save output files

    Returns:
    --------
    dict
        Dictionary with paths to generated visualizations
    """
    os.makedirs(output_dir, exist_ok=True)
    output_dir = Path(output_dir)

    print("Extracting buses from line data...")

    # Extract buses from lines
    pypsa_buses = extract_buses_from_lines(pypsa_gdf)
    jao_buses = extract_buses_from_lines(jao_gdf)

    print(f"Extracted buses: JAO={len(jao_buses)}, PyPSA={len(pypsa_buses)}")

    # Generate all three comparisons
    print("Generating line length comparison...")
    line_comparison = compare_line_length_by_voltage(pypsa_gdf, jao_gdf)

    print("Generating substation comparison...")
    substation_comparison = compare_substations_count(pypsa_buses, jao_buses)

    print("Generating circuit comparison...")
    circuit_comparison = compare_circuits_count(pypsa_gdf, jao_gdf)

    # Create combined visualization
    print("Creating combined visualization...")
    html_path = output_dir / "grid_comparisons.html"
    visualization_path = create_comparison_html(
        line_comparison,
        substation_comparison,
        circuit_comparison,
        html_path
    )

    # Also save the raw data
    line_comparison.to_csv(output_dir / "line_length_comparison.csv")
    substation_comparison.to_csv(output_dir / "substation_comparison.csv")
    circuit_comparison.to_csv(output_dir / "circuit_comparison.csv")

    return {
        'html': str(visualization_path),
        'line_comparison': line_comparison,
        'substation_comparison': substation_comparison,
        'circuit_comparison': circuit_comparison
    }