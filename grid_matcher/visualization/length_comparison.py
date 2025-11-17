import pandas as pd


# Assuming you have PyPSA and JAO dataframes with line data
# PyPSA typically has 'lines' or 'links' dataframes with 'length' and 'v_nom' columns
# JAO likely has similar data structure

def compare_line_length_by_voltage(pypsa_df, jao_df):
    # Group by voltage level and sum lengths
    pypsa_by_voltage = pypsa_df.groupby('v_nom')['length'].sum() / 1000  # Convert to km
    jao_by_voltage = jao_df.groupby('voltage')['length'].sum() / 1000  # Adjust column names as needed

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

# Example usage:
# voltage_comparison = compare_line_length_by_voltage(pypsa.lines_df, jao_lines)
# print(voltage_comparison)

def compare_substations_count(pypsa_df, jao_df):
    # Count substations/buses by voltage level
    pypsa_substations = pypsa_df.groupby('v_nom').size()
    jao_substations = jao_df.groupby('voltage').size()  # Adjust column name as needed

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


# Example usage:
# substation_comparison = compare_substations_count(pypsa.buses, jao_nodes)
# print(substation_comparison)

def compare_circuits_count(pypsa_df, jao_df):
    # Count lines by voltage level
    pypsa_lines = pypsa_df.groupby('v_nom').size()
    jao_lines = jao_df.groupby('voltage').size()  # Adjust column name as needed

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

    # Additionally compare circuits per voltage level (if you have parallel lines data)
    # This would require 'num_parallel' or similar column in your datasets

    return comparison


# Example usage:
# circuits_comparison = compare_circuits_count(pypsa.lines, jao_lines)
# print(circuits_comparison)

def calculate_difference_percentage(original, allocated):
    """
    Calculate the percentage difference between original and allocated values.

    Returns a percentage value between -100% and infinity where:
    - 0% means identical values
    - 100% means allocated is double the original
    - -50% means allocated is half the original
    """
    if original == 0:
        if allocated == 0:
            return 0  # Both zero means no difference
        else:
            return 100  # Original is zero, allocated is non-zero

    # Calculate percentage difference
    percent_diff = ((allocated - original) / original) * 100

    return percent_diff


def compare_line_lengths(jao_gdf, pypsa_gdf, matching_results=None, output_dir="output"):
    """
    Compare line lengths between TSO and PyPSA datasets.
    """
    import pandas as pd
    import numpy as np
    import os
    from pathlib import Path

    output_path = Path(output_dir)
    os.makedirs(output_path, exist_ok=True)

    print("Comparing line lengths between TSO and PyPSA datasets...")

    # Extract line lengths
    jao_length_col = next((col for col in ['length_km', 'length'] if col in jao_gdf.columns), None)
    pypsa_length_col = next((col for col in ['length_km', 'length'] if col in pypsa_gdf.columns), None)

    if not jao_length_col or not pypsa_length_col:
        print(f"Warning: Length columns not found. TSO columns: {jao_gdf.columns.tolist()}")
        print(f"PyPSA columns: {pypsa_gdf.columns.tolist()}")
        jao_length_col = 'length'  # Default
        pypsa_length_col = 'length'  # Default

    jao_total = jao_gdf[jao_length_col].sum()
    pypsa_total = pypsa_gdf[pypsa_length_col].sum()

    print(f"Total TSO line length: {jao_total:.2f} km")
    print(f"Total PyPSA line length: {pypsa_total:.2f} km")

    # Calculate percentage difference
    if jao_total > 0:
        diff_pct = (pypsa_total - jao_total) / jao_total * 100
    else:
        diff_pct = 0

    print(f"Difference in total length: {diff_pct:.2f}%")

    # Calculate breakdown by voltage level
    try:
        # Get voltage column names
        jao_voltage_col = next((col for col in ['v_nom', 'voltage'] if col in jao_gdf.columns), None)
        pypsa_voltage_col = next((col for col in ['v_nom', 'voltage'] if col in pypsa_gdf.columns), None)

        # Group by voltage level
        if jao_voltage_col and pypsa_voltage_col:
            # Create copies to avoid modifying originals
            jao_copy = jao_gdf.copy()
            pypsa_copy = pypsa_gdf.copy()

            # Standardize voltage levels - treat 380kV as 400kV
            if jao_voltage_col in jao_copy.columns:
                jao_copy[jao_voltage_col] = jao_copy[jao_voltage_col].apply(
                    lambda v: "400" if str(v) == "380" else str(v))

            if pypsa_voltage_col in pypsa_copy.columns:
                pypsa_copy[pypsa_voltage_col] = pypsa_copy[pypsa_voltage_col].apply(
                    lambda v: "400" if str(v) == "380" else str(v))

            # Group TSO data
            jao_by_voltage = jao_copy.groupby(jao_voltage_col)[jao_length_col].sum()
            jao_pct_by_voltage = jao_by_voltage / jao_total * 100 if jao_total > 0 else 0

            # Group PyPSA data
            pypsa_by_voltage = pypsa_copy.groupby(pypsa_voltage_col)[pypsa_length_col].sum()
            pypsa_pct_by_voltage = pypsa_by_voltage / pypsa_total * 100 if pypsa_total > 0 else 0

            # Combine into a single DataFrame
            voltage_levels = sorted(set(jao_by_voltage.index) | set(pypsa_by_voltage.index))
            comparison_data = []

            for voltage in voltage_levels:
                # Only include main voltage levels (220kV and 400kV)
                if str(voltage) in ['220', '400']:
                    jao_length = jao_by_voltage.get(voltage, 0)
                    pypsa_length = pypsa_by_voltage.get(voltage, 0)

                    if jao_length > 0:
                        diff_pct_voltage = (pypsa_length - jao_length) / jao_length * 100
                    else:
                        diff_pct_voltage = None

                    comparison_data.append({
                        'voltage': str(voltage),
                        'jao_length_km': jao_length,
                        'jao_percentage': jao_pct_by_voltage.get(voltage, 0),
                        'pypsa_length_km': pypsa_length,
                        'pypsa_percentage': pypsa_pct_by_voltage.get(voltage, 0),
                        'difference_percentage': diff_pct_voltage
                    })

            comparison_df = pd.DataFrame(comparison_data)

        else:
            # If voltage columns not found, create default DataFrame
            comparison_df = pd.DataFrame([
                {'voltage': '220', 'jao_length_km': 0, 'jao_percentage': 0, 'pypsa_length_km': 0, 'pypsa_percentage': 0,
                 'difference_percentage': None},
                {'voltage': '400', 'jao_length_km': 0, 'jao_percentage': 0, 'pypsa_length_km': 0, 'pypsa_percentage': 0,
                 'difference_percentage': None}
            ])

    except Exception as e:
        print(f"Error creating voltage breakdown: {e}")
        import traceback
        traceback.print_exc()
        comparison_df = pd.DataFrame([
            {'voltage': '220', 'jao_length_km': 0, 'jao_percentage': 0, 'pypsa_length_km': 0, 'pypsa_percentage': 0,
             'difference_percentage': None},
            {'voltage': '400', 'jao_length_km': 0, 'jao_percentage': 0, 'pypsa_length_km': 0, 'pypsa_percentage': 0,
             'difference_percentage': None}
        ])

    # Generate HTML report
    from grid_matcher.visualization.length_comparison import create_length_comparison_html

    html_report = create_length_comparison_html(
        comparison_df,
        jao_total,
        pypsa_total,
        diff_pct,
        output_path,
        jao_gdf=jao_gdf,
        jao_lines=jao_gdf,
        pypsa_lines=pypsa_gdf,
        treat_380kv_as_400kv=True  # Add this flag
    )

    return {
        'jao_total': jao_total,
        'pypsa_total': pypsa_total,
        'difference_percentage': diff_pct,
        'comparison_df': comparison_df,
        'html_report': html_report
    }


def create_length_comparison_html(
        comparison_df,
        jao_total,
        pypsa_total,
        diff_pct,
        output_dir,
        jao_gdf=None,
        jao_lines=None,
        pypsa_lines=None,
        treat_380kv_as_400kv=False
):
    """Create HTML report comparing line lengths."""
    import pandas as pd
    import json
    from pathlib import Path

    # Output file path
    output_path = Path(output_dir) / 'length_comparison.html'

    # Debug input data
    print(f"create_length_comparison_html() received:")
    print(f"  - comparison_df shape: {comparison_df.shape}")
    print(f"  - jao_total: {jao_total}")
    print(f"  - pypsa_total: {pypsa_total}")
    print(f"  - diff_pct: {diff_pct}")
    print(f"  - treat_380kv_as_400kv: {treat_380kv_as_400kv}")

    # Make a copy to avoid modifying the original
    df_for_charts = comparison_df.copy()

    # Print voltage values to debug
    if not df_for_charts.empty and 'voltage' in df_for_charts.columns:
        print(f"  - Unique voltage values: {df_for_charts['voltage'].unique()}")

    # Convert the DataFrame to a list of dictionaries for charts
    # Use original data without filtering to ensure we have data for charts
    data_records = df_for_charts.to_dict('records')

    # For debugging
    print(f"  - Data records for charts: {data_records}")

    # Create the summary table
    summary_table_rows = []

    # Add row for total
    total_row = f"""
    <tr>
        <td><strong>TOTAL</strong></td>
        <td>{jao_total:.2f} km</td>
        <td>100%</td>
        <td>{pypsa_total:.2f} km</td>
        <td>100%</td>
        <td>{diff_pct:.2f}%</td>
    </tr>
    """
    summary_table_rows.append(total_row)

    # Add rows for voltage levels if data exists
    if data_records:
        for record in data_records:
            voltage = record.get('voltage', 'N/A')
            jao_length = record.get('jao_length_km', 0)
            jao_pct = record.get('jao_percentage', 0)
            pypsa_length = record.get('pypsa_length_km', 0)
            pypsa_pct = record.get('pypsa_percentage', 0)
            diff_pct = record.get('difference_percentage', None)

            if diff_pct is not None:
                diff_pct_str = f"{diff_pct:.2f}%"
            else:
                diff_pct_str = "N/A"

            row = f"""
            <tr>
                <td>{voltage} kV</td>
                <td>{jao_length:.2f} km</td>
                <td>{jao_pct:.2f}%</td>
                <td>{pypsa_length:.2f} km</td>
                <td>{pypsa_pct:.2f}%</td>
                <td>{diff_pct_str}</td>
            </tr>
            """
            summary_table_rows.append(row)

    # Create the summary table HTML
    summary_table = f"""
    <table class="summary-table">
        <thead>
            <tr>
                <th>Voltage Level</th>
                <th>TSO Length</th>
                <th>TSO %</th>
                <th>PyPSA Length</th>
                <th>PyPSA %</th>
                <th>Difference</th>
            </tr>
        </thead>
        <tbody>
            {''.join(summary_table_rows)}
        </tbody>
    </table>
    """

    # Create the page HTML with the summary table and charts
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Line Length Comparison: TSO vs PyPSA</title>
        <script src="https://cdn.plot.ly/plotly-2.24.1.min.js"></script>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 20px;
                background-color: #f9f9f9;
            }}
            .chart-container {{
                background-color: white;
                border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                padding: 20px;
                margin-bottom: 30px;
            }}
            h2 {{
                color: #333;
                border-bottom: 1px solid #ddd;
                padding-bottom: 10px;
                margin-top: 0;
            }}
            .note {{
                font-size: 0.9em;
                font-style: italic;
                color: #666;
                margin-top: 5px;
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
            #barChart {{
                height: 500px;
                width: 100%;
            }}
            .pie-chart {{
                height: 400px;
                width: 100%;
            }}
            .error-message {{
                color: red;
                font-weight: bold;
                padding: 20px;
                text-align: center;
            }}
        </style>
    </head>
    <body>
        <div class="chart-container">
            <h2>Summary of Line Lengths: TSO vs PyPSA</h2>
            {f"<div class='note'>Note: 380kV lines in PyPSA data are shown as 400kV for comparison.</div>" if treat_380kv_as_400kv else ""}
            {summary_table}
        </div>

        <div class="chart-container">
            <h2>Line Length Comparison by Voltage Level</h2>
            <div id="barChart"></div>
        </div>

        <div class="chart-container">
            <h2>Line Length Distribution Comparison</h2>
            <div id="pieCharts" style="display: flex; flex-wrap: wrap;">
                <div id="jaoPieChart" class="pie-chart" style="width: 50%;"></div>
                <div id="pypsaPieChart" class="pie-chart" style="width: 50%;"></div>
            </div>
        </div>

        <script>
            // Check if Plotly is loaded
            if (typeof Plotly === 'undefined') {{
                document.body.innerHTML += '<div class="error-message">Error: Plotly library failed to load. Please check your internet connection or try a different browser.</div>';
            }}

            // Data for charts
            const data = {json.dumps(data_records)};

            // Check if data is available and log for debugging
            console.log("Chart data:", data);

            try {{
                // Bar chart for comparison
                if (data && data.length > 0) {{
                    const barChartData = [
                        {{
                            x: data.map(d => d.voltage + ' kV'),
                            y: data.map(d => d.jao_length_km),
                            name: 'TSO',
                            type: 'bar',
                            marker: {{ color: 'rgba(58, 71, 80, 0.6)' }}
                        }},
                        {{
                            x: data.map(d => d.voltage + ' kV'),
                            y: data.map(d => d.pypsa_length_km),
                            name: 'PyPSA',
                            type: 'bar',
                            marker: {{ color: 'rgba(246, 78, 139, 0.6)' }}
                        }}
                    ];

                    const barChartLayout = {{
                        title: 'Line Length by Voltage Level',
                        barmode: 'group',
                        yaxis: {{
                            title: 'Length (km)'
                        }},
                        legend: {{
                            x: 0.1,
                            y: 1.1,
                            orientation: 'h'
                        }},
                        annotations: [
                            {{
                                x: 0.5,
                                y: -0.15,
                                showarrow: false,
                                text: `Total TSO: {jao_total:.2f} km | Total PyPSA: {pypsa_total:.2f} km | Difference: {diff_pct:.2f}%`,
                                xref: 'paper',
                                yref: 'paper',
                                font: {{ size: 12 }}
                            }}
                        ],
                        height: 500,
                        autosize: true
                    }};

                    Plotly.newPlot('barChart', barChartData, barChartLayout);

                    // Pie charts for distribution
                    const jaoPieData = [
                        {{
                            labels: data.map(d => d.voltage + ' kV'),
                            values: data.map(d => d.jao_length_km),
                            type: 'pie',
                            textinfo: 'label+percent',
                            hoverinfo: 'label+value+percent',
                            marker: {{
                                colors: ['rgba(58, 71, 80, 0.8)', 'rgba(58, 71, 80, 0.5)']
                            }}
                        }}
                    ];

                    const jaoPieLayout = {{
                        title: 'TSO Line Length Distribution',
                        showlegend: false,
                        height: 400,
                        autosize: true,
                        margin: {{ t: 50, b: 20, l: 20, r: 20 }},
                        annotations: [
                            {{
                                x: 0.5,
                                y: -0.1,
                                showarrow: false,
                                text: `Total: {jao_total:.2f} km`,
                                xref: 'paper',
                                yref: 'paper',
                                font: {{ size: 12 }}
                            }}
                        ]
                    }};

                    const pypsaPieData = [
                        {{
                            labels: data.map(d => d.voltage + ' kV'),
                            values: data.map(d => d.pypsa_length_km),
                            type: 'pie',
                            textinfo: 'label+percent',
                            hoverinfo: 'label+value+percent',
                            marker: {{
                                colors: ['rgba(246, 78, 139, 0.8)', 'rgba(246, 78, 139, 0.5)']
                            }}
                        }}
                    ];

                    const pypsaPieLayout = {{
                        title: 'PyPSA Line Length Distribution',
                        showlegend: false,
                        height: 400,
                        autosize: true,
                        margin: {{ t: 50, b: 20, l: 20, r: 20 }},
                        annotations: [
                            {{
                                x: 0.5,
                                y: -0.1,
                                showarrow: false,
                                text: `Total: {pypsa_total:.2f} km`,
                                xref: 'paper',
                                yref: 'paper',
                                font: {{ size: 12 }}
                            }}
                        ]
                    }};

                    Plotly.newPlot('jaoPieChart', jaoPieData, jaoPieLayout);
                    Plotly.newPlot('pypsaPieChart', pypsaPieData, pypsaPieLayout);
                }} else {{
                    document.getElementById('barChart').innerHTML = '<div class="error-message">No data available for charts</div>';
                    document.getElementById('jaoPieChart').innerHTML = '<div class="error-message">No data available for charts</div>';
                    document.getElementById('pypsaPieChart').innerHTML = '<div class="error-message">No data available for charts</div>';
                }}
            }} catch (e) {{
                console.error("Error rendering charts:", e);
                document.getElementById('barChart').innerHTML = '<div class="error-message">Error rendering chart: ' + e.message + '</div>';
                document.getElementById('jaoPieChart').innerHTML = '<div class="error-message">Error rendering chart: ' + e.message + '</div>';
                document.getElementById('pypsaPieChart').innerHTML = '<div class="error-message">Error rendering chart: ' + e.message + '</div>';
            }}
        </script>
    </body>
    </html>
    """

    # Write the HTML to file
    with open(output_path, 'w') as f:
        f.write(html_content)

    print(f"HTML report saved to: {output_path}")
    return output_path

