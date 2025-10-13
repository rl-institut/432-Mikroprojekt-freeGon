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


def prepare_visualization_data(matching_results, pypsa_gdf, jao_gdf=None, use_demo_jao_values=False):
    """
    Transform matching results into the format expected by the parameter comparison visualization.
    """
    print("Preparing visualization data...")

    # Create a lookup for JAO parameters
    jao_lookup = {}
    if jao_gdf is not None:
        for _, row in jao_gdf.iterrows():
            jao_id = str(row['id'])
            jao_lookup[jao_id] = {
                'r': float(row.get('r', 0) or 0),
                'x': float(row.get('x', 0) or 0),
                'b': float(row.get('b', 0) or 0),
                'length_km': float(row.get('length_km', row.get('length', 0)) or 0)
            }
        print(f"Created JAO lookup table with {len(jao_lookup)} entries")

    # Create enhanced results with the expected structure
    enhanced_results = []

    # Count how many results have matched_lines_data with allocated parameters
    has_allocation_data = 0
    missing_allocation_data = 0

    for result in matching_results:
        if not result.get('matched', False):
            continue

        # IMPORTANT: Check if result already has matched_lines_data with allocated parameters
        # This happens for manual matches which already have allocated parameters
        segs = result.get('matched_lines_data', [])
        has_all_params = all('allocated_r' in seg for seg in segs) if segs else False

        if has_all_params:
            has_allocation_data += 1
            # Already has properly allocated parameters, use it directly
            enhanced_result = {
                'jao_id': result.get('jao_id', ''),
                'matched': True,
                'jao_r': result.get('jao_r', 0),
                'jao_x': result.get('jao_x', 0),
                'jao_b': result.get('jao_b', 0),
                'jao_length_km': result.get('jao_length_km', 0),
                'matched_lines_data': segs
            }
            enhanced_results.append(enhanced_result)
            continue

        missing_allocation_data += 1
        # Standard processing for results without allocated parameters
        # Get JAO ID and length
        jao_id = str(result.get('jao_id', ''))
        jao_length_km = float(result.get('jao_length_km', result.get('jao_length', 0)) or 0)

        # Get JAO parameters
        jao_r = float(result.get('jao_r', 0) or 0)
        jao_x = float(result.get('jao_x', 0) or 0)
        jao_b = float(result.get('jao_b', 0) or 0)

        # Try lookup if parameters are missing
        if (jao_r == 0 and jao_x == 0 and jao_b == 0) and jao_id in jao_lookup:
            jao_data = jao_lookup[jao_id]
            jao_r = jao_data['r']
            jao_x = jao_data['x']
            jao_b = jao_data['b']
            if jao_length_km <= 0:
                jao_length_km = jao_data['length_km']

        # Get matched PyPSA IDs
        pypsa_ids = result.get('pypsa_ids', [])
        if not isinstance(pypsa_ids, list):
            pypsa_ids = [pypsa_ids]

        if not pypsa_ids:
            continue

        # Create matched_lines_data array expected by visualization
        matched_lines_data = []
        total_pypsa_r = 0
        total_pypsa_x = 0
        total_pypsa_b = 0
        total_pypsa_length = 0

        for pypsa_id in pypsa_ids:
            # Find the pypsa line in the dataframe
            matching_rows = pypsa_gdf[pypsa_gdf['id'].astype(str) == str(pypsa_id)]
            if len(matching_rows) == 0 and 'line_id' in pypsa_gdf.columns:
                matching_rows = pypsa_gdf[pypsa_gdf['line_id'].astype(str) == str(pypsa_id)]

            if len(matching_rows) == 0:
                continue

            pypsa_row = matching_rows.iloc[0]

            # Extract and convert parameters
            try:
                r = float(pypsa_row.get('r', 0) or 0)
                x = float(pypsa_row.get('x', 0) or 0)
                b = float(pypsa_row.get('b', 0) or 0)
                length = float(pypsa_row.get('length', 0) or 0)
                circuits = int(pypsa_row.get('circuits', 1) or 1)

                total_pypsa_r += r
                total_pypsa_x += x
                total_pypsa_b += b
                total_pypsa_length += length

                # Create line data entry
                matched_line = {
                    'network_id': pypsa_id,
                    'length_km': length,
                    'num_parallel': circuits,
                    'allocation_status': 'Applied',
                    'original_r': r,
                    'original_x': x,
                    'original_b': b
                }
                matched_lines_data.append(matched_line)
            except Exception as e:
                print(f"Error processing PyPSA ID {pypsa_id}: {str(e)}")

        # Only process if we have lines with data and valid JAO parameters
        if matched_lines_data and (jao_r > 0 or jao_x > 0 or jao_b > 0 or use_demo_jao_values):
            # Use demo values if needed and allowed
            if (jao_r == 0 and jao_x == 0 and jao_b == 0) and use_demo_jao_values:
                import random
                variation_factor = lambda: random.uniform(0.8, 1.2)
                jao_r = total_pypsa_r * variation_factor()
                jao_x = total_pypsa_x * variation_factor()
                jao_b = total_pypsa_b * variation_factor()

                if jao_length_km <= 0:
                    jao_length_km = total_pypsa_length / 1000

                jao_id = f"{jao_id} (demo values)"

            # Skip if still no valid parameters
            if jao_r == 0 and jao_x == 0 and jao_b == 0:
                continue

            # Calculate per-km values
            jao_r_per_km = jao_r / jao_length_km if jao_length_km > 0 else 0
            jao_x_per_km = jao_x / jao_length_km if jao_length_km > 0 else 0
            jao_b_per_km = jao_b / jao_length_km if jao_length_km > 0 else 0

            # Allocate parameters to each line
            for line in matched_lines_data:
                circuits = line.get('num_parallel', 1)

                # Store allocated parameters
                line['allocated_r'] = jao_r
                line['allocated_x'] = jao_x
                line['allocated_b'] = jao_b

                # Store per-km values
                line['allocated_r_per_km'] = jao_r_per_km
                line['allocated_x_per_km'] = jao_x_per_km
                line['allocated_b_per_km'] = jao_b_per_km

            # Create the enhanced result
            enhanced_result = {
                'jao_id': jao_id,
                'matched': True,
                'jao_r': jao_r,
                'jao_x': jao_x,
                'jao_b': jao_b,
                'jao_length_km': jao_length_km,
                'matched_lines_data': matched_lines_data
            }
            enhanced_results.append(enhanced_result)

    print(f"Created {len(enhanced_results)} enhanced results with line data")
    print(f"Results with pre-allocated parameters: {has_allocation_data}")
    print(f"Results needing parameter allocation: {missing_allocation_data}")

    return enhanced_results

def visualize_parameter_comparison(matching_results, pypsa_gdf, output_dir="output"):
    """
    Create HTML visualization comparing original and allocated electrical parameters
    with divergence-based coloring and column filtering.
    Includes both total and per-km comparisons with circuit adjustments.
    """
    import os
    import json
    import numpy as np
    import pandas as pd
    from pathlib import Path
    from scipy.stats import pearsonr

    print("Creating parameter comparison visualizations...")

    # Define calculation function at the beginning
    def calc_diff_pct(a, b):
        """
        Calculate the percentage difference between a (allocated) and b (original).
        """
        if abs(b) < 1e-9:
            return None

        # Calculate as percentage change from original to allocated
        return ((a - b) / abs(b)) * 100.0

    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Create a lookup for PyPSA lines
    pypsa_lookup = {}
    for _, row in pypsa_gdf.iterrows():
        key = str(row.get('line_id', row.get('id', '')))
        pypsa_lookup[key] = row

    # Prepare data for analysis
    all_data = []

    for result in matching_results:
        if not result.get('matched', False):
            continue

        segments = result.get('matched_lines_data', [])
        if not segments:
            continue

        jao_id = result.get('jao_id', 'unknown')
        jao_length_km = float(result.get('jao_length_km', 0) or 0)
        jao_r_total = float(result.get('jao_r', 0) or 0)
        jao_x_total = float(result.get('jao_x', 0) or 0)
        jao_b_total = float(result.get('jao_b', 0) or 0)

        # Calculate per-km values
        jao_r_km = jao_r_total / jao_length_km if jao_length_km > 0 else 0
        jao_x_km = jao_x_total / jao_length_km if jao_length_km > 0 else 0
        jao_b_km = jao_b_total / jao_length_km if jao_length_km > 0 else 0

        for segment in segments:
            pid = segment.get('network_id', '')
            pypsa_row = pypsa_lookup.get(pid)

            if pypsa_row is None or segment.get('allocation_status') not in ['Applied', 'Parallel Circuit']:
                continue

            # Get segment data
            length_km = float(segment.get('length_km', 0) or 0)
            circuits = int(segment.get('num_parallel', 1) or 1)

            # Get allocated parameters from segment
            allocated_r = float(segment.get('allocated_r', 0) or 0)
            allocated_x = float(segment.get('allocated_x', 0) or 0)
            allocated_b = float(segment.get('allocated_b', 0) or 0)

            # Get original parameters directly from PyPSA dataframe
            original_r = float(pypsa_row.get('r', 0) or 0)
            original_x = float(pypsa_row.get('x', 0) or 0)
            original_b = float(pypsa_row.get('b', 0) or 0)

            # Convert PyPSA length from meters to kilometers
            length_km_converted = length_km / 1000 if length_km > 0 else 0

            # Calculate PyPSA per-km values with proper unit conversion
            original_r_per_km = original_r / length_km_converted if length_km_converted > 0 else 0
            original_x_per_km = original_x / length_km_converted if length_km_converted > 0 else 0
            original_b_per_km = original_b / length_km_converted if length_km_converted > 0 else 0

            # Apply circuit adjustment for per-km values
            jao_r_km_adjusted = jao_r_km / circuits
            jao_x_km_adjusted = jao_x_km / circuits
            jao_b_km_adjusted = jao_b_km * circuits

            # Calculate percentage differences for TOTAL values (original vs. allocated)
            diff_r_pct = calc_diff_pct(allocated_r, original_r)
            diff_x_pct = calc_diff_pct(allocated_x, original_x)
            diff_b_pct = calc_diff_pct(allocated_b, original_b)

            # Calculate percentage differences for PER-KM values (circuit-adjusted)
            diff_r_pct_km = calc_diff_pct(jao_r_km_adjusted, original_r_per_km)
            diff_x_pct_km = calc_diff_pct(jao_x_km_adjusted, original_x_per_km)
            diff_b_pct_km = calc_diff_pct(jao_b_km_adjusted, original_b_per_km)

            # Add to data (only once per segment)
            all_data.append({
                'jao_id': jao_id,
                'pypsa_id': pid,
                'length_km': length_km,
                'circuits': circuits,
                # Total values
                'original_r': original_r,
                'allocated_r': allocated_r,
                'diff_r_pct': diff_r_pct,
                'original_x': original_x,
                'allocated_x': allocated_x,
                'diff_x_pct': diff_x_pct,
                'original_b': original_b,
                'allocated_b': allocated_b,
                'diff_b_pct': diff_b_pct,
                # Per-km values
                'original_r_per_km': original_r_per_km,
                'original_x_per_km': original_x_per_km,
                'original_b_per_km': original_b_per_km,
                'jao_r_km_adjusted': jao_r_km_adjusted,
                'jao_x_km_adjusted': jao_x_km_adjusted,
                'jao_b_km_adjusted': jao_b_km_adjusted,
                'diff_r_pct_km': diff_r_pct_km,
                'diff_x_pct_km': diff_x_pct_km,
                'diff_b_pct_km': diff_b_pct_km
            })

    # Calculate correlation coefficients
    df = pd.DataFrame(all_data)
    r_values = {}
    for param in ['r', 'x', 'b']:
        if len(df) > 1:
            original_col = f'original_{param}'
            allocated_col = f'allocated_{param}'
            mask = (df[original_col] > 0) & (df[allocated_col] > 0)
            if mask.sum() > 1:
                r_val, p_val = pearsonr(df.loc[mask, original_col], df.loc[mask, allocated_col])
                r_values[param] = {'r': r_val, 'p': p_val}
            else:
                r_values[param] = {'r': None, 'p': None}
        else:
            r_values[param] = {'r': None, 'p': None}

    # Generate HTML with per-km comparison charts added
    html = """<!DOCTYPE html>
<html>
<head>
    <title>Electrical Parameter Comparison</title>
    <meta charset="utf-8">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-colorschemes"></script>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }
        h1, h2, h3 {
            color: #333;
        }
        .container {
            background-color: white;
            padding: 20px;
            border-radius: 5px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        .chart-container {
            width: 100%;
            height: 400px;
            margin-bottom: 30px;
        }
        table {
            border-collapse: collapse;
            width: 100%;
            margin-top: 20px;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }
        th {
            background-color: #4CAF50;
            color: white;
            position: sticky;
            top: 0;
        }
        .filters input {
            width: 90%;
            padding: 5px;
            margin: 2px 0;
            border: 1px solid #ddd;
        }
        .good-match {
            background-color: #c8e6c9;
        }
        .moderate-match {
            background-color: #fff9c4;
        }
        .poor-match {
            background-color: #ffccbc;
        }
        .data-table-container {
            max-height: 600px;
            overflow-y: auto;
        }
        .section-divider {
            border-top: 2px solid #4CAF50;
            margin: 30px 0;
            padding-top: 10px;
        }
        .color-legend {
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 15px 0;
            padding: 10px;
            background-color: #fff;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .color-gradient {
            width: 300px;
            height: 20px;
            background: linear-gradient(to right, #0047ab, #6495ED, #4CAF50, #FF6B6B, #dc143c);
            margin: 0 15px;
            border-radius: 3px;
        }
        .color-label {
            font-size: 12px;
            color: #555;
        }
        .tabs {
            display: flex;
            margin-bottom: 20px;
        }
        .tab {
            padding: 10px 20px;
            cursor: pointer;
            background: #ddd;
            margin-right: 5px;
            border-radius: 5px 5px 0 0;
        }
        .tab.active {
            background: #4CAF50;
            color: white;
        }
        .note {
            font-style: italic;
            color: #666;
            margin: 10px 0;
            padding: 10px;
            background-color: #f9f9f9;
            border-left: 4px solid #4CAF50;
        }
    </style>
</head>
<body>
    <h1>Electrical Parameter Comparison</h1>

    <div class="note">
        <strong>Note:</strong> This visualization shows both total and per-km values. The per-km comparisons use circuit-adjusted values:
        <ul>
            <li>For series elements (R, X): JAO value is divided by the number of circuits</li>
            <li>For parallel elements (B): JAO value is multiplied by the number of circuits</li>
        </ul>
        This adjustment matches the parameter allocation table's approach.
    </div>

    <div class="color-legend">
        <span class="color-label">-100%</span>
        <div class="color-gradient"></div>
        <span class="color-label">+100%</span>
        <span style="margin-left: 20px; font-style: italic; color: #666;">Color represents difference percentage</span>
    </div>

    <h2 class="section-divider">Total Parameter Comparison</h2>

    <div class="container">
        <h2>Resistance (R) Comparison - Total Values</h2>
        <div class="chart-container">
            <canvas id="r-scatter"></canvas>
        </div>
    </div>

    <div class="container">
        <h2>Reactance (X) Comparison - Total Values</h2>
        <div class="chart-container">
            <canvas id="x-scatter"></canvas>
        </div>
    </div>

    <div class="container">
        <h2>Susceptance (B) Comparison - Total Values</h2>
        <div class="chart-container">
            <canvas id="b-scatter"></canvas>
        </div>
    </div>

    <h2 class="section-divider">Per-km Parameter Comparison (Circuit-adjusted)</h2>

    <div class="container">
        <h2>PyPSA vs JAO Resistance (R) Per-km Comparison</h2>
        <div class="chart-container">
            <canvas id="r-per-km"></canvas>
        </div>
    </div>

    <div class="container">
        <h2>PyPSA vs JAO Reactance (X) Per-km Comparison</h2>
        <div class="chart-container">
            <canvas id="x-per-km"></canvas>
        </div>
    </div>

    <div class="container">
        <h2>PyPSA vs JAO Susceptance (B) Per-km Comparison</h2>
        <div class="chart-container">
            <canvas id="b-per-km"></canvas>
        </div>
    </div>

    <h2 class="section-divider">PyPSA vs JAO Parameter Comparison</h2>

    <div class="container">
        <h2>PyPSA vs JAO Resistance (R) Comparison</h2>
        <div class="chart-container">
            <canvas id="r-comparison"></canvas>
        </div>
    </div>

    <div class="container">
        <h2>PyPSA vs JAO Reactance (X) Comparison</h2>
        <div class="chart-container">
            <canvas id="x-comparison"></canvas>
        </div>
    </div>

    <div class="container">
        <h2>PyPSA vs JAO Susceptance (B) Comparison</h2>
        <div class="chart-container">
            <canvas id="b-comparison"></canvas>
        </div>
    </div>

    <div class="container">
        <h2>Data Table</h2>
        <div class="tabs">
            <div class="tab active" onclick="showTable('total-table')">Total Values</div>
            <div class="tab" onclick="showTable('per-km-table')">Per-km Values</div>
        </div>

        <div id="total-table" class="data-table-container">
            <table id="data-table">
                <thead>
                    <tr>
                        <th>JAO ID</th>
                        <th>PyPSA ID</th>
                        <th>Length (km)</th>
                        <th>Circuits</th>
                        <th>Original R (Ω)</th>
                        <th>Allocated R (Ω)</th>
                        <th>R Diff (%)</th>
                        <th>Original X (Ω)</th>
                        <th>Allocated X (Ω)</th>
                        <th>X Diff (%)</th>
                        <th>Original B (S)</th>
                        <th>Allocated B (S)</th>
                        <th>B Diff (%)</th>
                    </tr>
                    <tr class="filters">
                        <td><input type="text" data-col="0" placeholder="Filter JAO ID"></td>
                        <td><input type="text" data-col="1" placeholder="Filter PyPSA ID"></td>
                        <td><input type="text" data-col="2" placeholder="Filter Length"></td>
                        <td><input type="text" data-col="3" placeholder="Filter Circuits"></td>
                        <td><input type="text" data-col="4" placeholder="Filter Original R"></td>
                        <td><input type="text" data-col="5" placeholder="Filter Allocated R"></td>
                        <td><input type="text" data-col="6" placeholder="Filter R Diff"></td>
                        <td><input type="text" data-col="7" placeholder="Filter Original X"></td>
                        <td><input type="text" data-col="8" placeholder="Filter Allocated X"></td>
                        <td><input type="text" data-col="9" placeholder="Filter X Diff"></td>
                        <td><input type="text" data-col="10" placeholder="Filter Original B"></td>
                        <td><input type="text" data-col="11" placeholder="Filter Allocated B"></td>
                        <td><input type="text" data-col="12" placeholder="Filter B Diff"></td>
                    </tr>
                </thead>
                <tbody>
                    <!-- Data rows will be inserted here -->
                </tbody>
            </table>
        </div>

        <div id="per-km-table" class="data-table-container" style="display: none;">
            <table id="per-km-data-table">
                <thead>
                    <tr>
                        <th>JAO ID</th>
                        <th>PyPSA ID</th>
                        <th>Length (km)</th>
                        <th>Circuits</th>
                        <th>PyPSA R (Ω/km)</th>
                        <th>JAO R (Ω/km)</th>
                        <th>R Diff (%)</th>
                        <th>PyPSA X (Ω/km)</th>
                        <th>JAO X (Ω/km)</th>
                        <th>X Diff (%)</th>
                        <th>PyPSA B (S/km)</th>
                        <th>JAO B (S/km)</th>
                        <th>B Diff (%)</th>
                    </tr>
                    <tr class="filters">
                        <td><input type="text" data-col="0" placeholder="Filter JAO ID"></td>
                        <td><input type="text" data-col="1" placeholder="Filter PyPSA ID"></td>
                        <td><input type="text" data-col="2" placeholder="Filter Length"></td>
                        <td><input type="text" data-col="3" placeholder="Filter Circuits"></td>
                        <td><input type="text" data-col="4" placeholder="Filter PyPSA R"></td>
                        <td><input type="text" data-col="5" placeholder="Filter JAO R"></td>
                        <td><input type="text" data-col="6" placeholder="Filter R Diff"></td>
                        <td><input type="text" data-col="7" placeholder="Filter PyPSA X"></td>
                        <td><input type="text" data-col="8" placeholder="Filter JAO X"></td>
                        <td><input type="text" data-col="9" placeholder="Filter X Diff"></td>
                        <td><input type="text" data-col="10" placeholder="Filter PyPSA B"></td>
                        <td><input type="text" data-col="11" placeholder="Filter JAO B"></td>
                        <td><input type="text" data-col="12" placeholder="Filter B Diff"></td>
                    </tr>
                </thead>
                <tbody>
                    <!-- Per-km data rows will be inserted here -->
                </tbody>
            </table>
        </div>
    </div>

    <script>
        // Data from Python
        const data = DATA_PLACEHOLDER;

        // Function to get color based on actual difference percentage - MORE DISTINCT COLORS
        function getDifferenceColor(diffPct) {
            if (diffPct === null || isNaN(diffPct)) {
                return 'rgba(100, 100, 100, 0.8)';  // Dark gray for N/A
            }

            // Clamp to a reasonable range
            const clampedDiff = Math.max(-100, Math.min(100, diffPct));

            // Create a continuous color gradient
            if (clampedDiff < 0) {
                // Blend from blue to green as we approach zero from negative side
                const ratio = Math.min(1, Math.abs(clampedDiff) / 50);  // 50% as full intensity point
                const r = Math.round(76 + (0 - 76) * ratio);  // Blend from green's R to blue's R
                const g = Math.round(175 + (71 - 175) * ratio);  // Blend from green's G to blue's G
                const b = Math.round(80 + (171 - 80) * ratio);  // Blend from green's B to blue's B
                return `rgba(${r}, ${g}, ${b}, 0.8)`;
            } else {
                // Blend from green to red as we move positive
                const ratio = Math.min(1, clampedDiff / 50);  // 50% as full intensity point
                const r = Math.round(76 + (220 - 76) * ratio);  // Blend from green's R to red's R
                const g = Math.round(175 + (20 - 175) * ratio);  // Blend from green's G to red's G
                const b = Math.round(80 + (60 - 80) * ratio);  // Blend from green's B to red's B
                return `rgba(${r}, ${g}, ${b}, 0.8)`;
            }
        }

        // Create scatter plots for total values
        function createScatterPlot(param, canvasId) {
            const ctx = document.getElementById(canvasId).getContext('2d');

            // Prepare data - IMPORTANT: Use diff_param_pct (not diff_param_pct_km)
            const chartData = data.map(item => {
                const diffPct = item['diff_' + param + '_pct']; // For TOTAL values
                return {
                    x: item['original_' + param],
                    y: item['allocated_' + param],
                    diffPct: diffPct,
                    color: getDifferenceColor(diffPct)
                };
            }).filter(d => d.x > 0 && d.y > 0);

            // Calculate min/max for both axes
            const allX = chartData.map(d => d.x);
            const allY = chartData.map(d => d.y);
            const minX = Math.min(...allX) * 0.9;
            const maxX = Math.max(...allX) * 1.1;
            const minY = Math.min(...allY) * 0.9;
            const maxY = Math.max(...allY) * 1.1;

            // Calculate diagonal line points
            const min = Math.min(minX, minY);
            const max = Math.max(maxX, maxY);
            const diagonalLine = [{ x: min, y: min }, { x: max, y: max }];

            // Create chart
            new Chart(ctx, {
                type: 'scatter',
                data: {
                    datasets: [
                        {
                            label: 'Parameter Values',
                            data: chartData,
                            backgroundColor: chartData.map(d => d.color),
                            pointRadius: 6,
                            pointHoverRadius: 8,
                            pointBorderColor: 'rgba(0,0,0,0.2)'
                        },
                        {
                            label: 'y = x',
                            data: diagonalLine,
                            showLine: true,
                            fill: false,
                            pointRadius: 0,
                            borderColor: 'rgba(128, 128, 128, 0.5)',
                            borderDash: [5, 5],
                            borderWidth: 2
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        title: {
                            display: true,
                            text: 'Original vs Allocated ' + param.toUpperCase() + ' Values (R=' + R_VALUES[param] + ')',
                            font: { size: 16 }
                        },
                        tooltip: {
                            callbacks: {
                                label: function(context) {
                                    const point = chartData[context.dataIndex];
                                    return [
                                        'Original: ' + point.x.toFixed(6),
                                        'Allocated: ' + point.y.toFixed(6),
                                        'Difference: ' + (point.diffPct ? point.diffPct.toFixed(2) + '%' : 'N/A')
                                    ];
                                }
                            }
                        },
                        legend: {
                            display: false // Hide the default legend as we have custom gradient legend
                        }
                    },
                    scales: {
                        x: {
                            title: {
                                display: true,
                                text: 'Original ' + (param === 'r' ? 'Resistance (Ω)' : param === 'x' ? 'Reactance (Ω)' : 'Susceptance (S)')
                            }
                        },
                        y: {
                            title: {
                                display: true,
                                text: 'Allocated ' + (param === 'r' ? 'Resistance (Ω)' : param === 'x' ? 'Reactance (Ω)' : 'Susceptance (S)')
                            }
                        }
                    }
                }
            });
        }

        // Create PyPSA vs JAO comparison charts for total values
        function createComparisonChart(param, canvasId) {
            const ctx = document.getElementById(canvasId).getContext('2d');

            // Prepare data - IMPORTANT: Use diff_param_pct (not diff_param_pct_km)
            const chartData = data.map(item => {
                const diffPct = item['diff_' + param + '_pct']; // For TOTAL values
                return {
                    x: item['original_' + param],
                    y: item['allocated_' + param],
                    jao_id: item.jao_id,
                    pypsa_id: item.pypsa_id,
                    length_km: item.length_km,
                    circuits: item.circuits,
                    diffPct: diffPct,
                    color: getDifferenceColor(diffPct)
                };
            }).filter(d => d.x > 0 && d.y > 0);

            // Calculate min/max for both axes
            const allX = chartData.map(d => d.x);
            const allY = chartData.map(d => d.y);
            const minX = Math.min(...allX) * 0.9;
            const maxX = Math.max(...allX) * 1.1;
            const minY = Math.min(...allY) * 0.9;
            const maxY = Math.max(...allY) * 1.1;

            // Calculate diagonal line points
            const min = Math.min(minX, minY);
            const max = Math.max(maxX, maxY);
            const diagonalLine = [{ x: min, y: min }, { x: max, y: max }];

            // Create chart
            new Chart(ctx, {
                type: 'scatter',
                data: {
                    datasets: [
                        {
                            label: 'Parameter Values',
                            data: chartData,
                            backgroundColor: chartData.map(d => d.color),
                            pointRadius: 6,
                            pointHoverRadius: 8,
                            pointBorderColor: 'rgba(0,0,0,0.2)'
                        },
                        {
                            label: 'y = x',
                            data: diagonalLine,
                            showLine: true,
                            fill: false,
                            pointRadius: 0,
                            borderColor: 'rgba(128, 128, 128, 0.5)',
                            borderDash: [5, 5],
                            borderWidth: 2
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        title: {
                            display: true,
                            text: 'PyPSA vs JAO ' + param.toUpperCase() + ' Values (R=' + R_VALUES[param] + ')',
                            font: { size: 16 }
                        },
                        tooltip: {
                            callbacks: {
                                label: function(context) {
                                    const point = chartData[context.dataIndex];
                                    return [
                                        'JAO ID: ' + point.jao_id,
                                        'PyPSA ID: ' + point.pypsa_id,
                                        'Length: ' + point.length_km.toFixed(2) + ' km',
                                        'Circuits: ' + point.circuits,
                                        'PyPSA: ' + point.x.toFixed(6),
                                        'JAO: ' + point.y.toFixed(6),
                                        'Difference: ' + (point.diffPct ? point.diffPct.toFixed(2) + '%' : 'N/A')
                                    ];
                                }
                            }
                        },
                        legend: {
                            display: false // Hide the default legend as we have custom gradient legend
                        }
                    },
                    scales: {
                        x: {
                            title: {
                                display: true,
                                text: 'PyPSA ' + (param === 'r' ? 'Resistance (Ω)' : param === 'x' ? 'Reactance (Ω)' : 'Susceptance (S)')
                            }
                        },
                        y: {
                            title: {
                                display: true,
                                text: 'JAO ' + (param === 'r' ? 'Resistance (Ω)' : param === 'x' ? 'Reactance (Ω)' : 'Susceptance (S)')
                            }
                        }
                    }
                }
            });
        }

        // Create per-km comparison charts
        function createPerKmChart(param, canvasId) {
            const ctx = document.getElementById(canvasId).getContext('2d');

            // Prepare per-km data with circuit adjustments - IMPORTANT: Use diff_param_pct_km
            const chartData = data.map(item => {
                const diffPct = item['diff_' + param + '_pct_km']; // For PER-KM values
                return {
                    x: item['original_' + param + '_per_km'],
                    y: item['jao_' + param + '_km_adjusted'],
                    jao_id: item.jao_id,
                    pypsa_id: item.pypsa_id,
                    length_km: item.length_km,
                    circuits: item.circuits,
                    diffPct: diffPct,
                    color: getDifferenceColor(diffPct)
                };
            }).filter(d => d.x > 0 && d.y > 0);

            // Calculate min/max for both axes
            const allX = chartData.map(d => d.x);
            const allY = chartData.map(d => d.y);
            const minX = Math.min(...allX) * 0.9;
            const maxX = Math.max(...allX) * 1.1;
            const minY = Math.min(...allY) * 0.9;
            const maxY = Math.max(...allY) * 1.1;

            // Calculate diagonal line points
            const min = Math.min(minX, minY);
            const max = Math.max(maxX, maxY);
            const diagonalLine = [{ x: min, y: min }, { x: max, y: max }];

            // Create chart
            new Chart(ctx, {
                type: 'scatter',
                data: {
                    datasets: [
                        {
                            label: 'Parameter Values Per-km',
                            data: chartData,
                            backgroundColor: chartData.map(d => d.color),
                            pointRadius: 6,
                            pointHoverRadius: 8,
                            pointBorderColor: 'rgba(0,0,0,0.2)'
                        },
                        {
                            label: 'y = x',
                            data: diagonalLine,
                            showLine: true,
                            fill: false,
                            pointRadius: 0,
                            borderColor: 'rgba(128, 128, 128, 0.5)',
                            borderDash: [5, 5],
                            borderWidth: 2
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        title: {
                            display: true,
                            text: 'PyPSA vs JAO ' + param.toUpperCase() + ' Per-km Values (Circuit-adjusted, R=' + R_VALUES[param] + ')',
                            font: { size: 16 }
                        },
                        tooltip: {
                            callbacks: {
                                label: function(context) {
                                    const point = chartData[context.dataIndex];
                                    return [
                                        'JAO ID: ' + point.jao_id,
                                        'PyPSA ID: ' + point.pypsa_id,
                                        'Length: ' + point.length_km.toFixed(2) + ' km',
                                        'Circuits: ' + point.circuits,
                                        'PyPSA: ' + point.x.toFixed(6) + ' per km',
                                        'JAO: ' + point.y.toFixed(6) + ' per km (circuit-adjusted)',
                                        'Difference: ' + (point.diffPct ? point.diffPct.toFixed(2) + '%' : 'N/A')
                                    ];
                                }
                            }
                        },
                        legend: {
                            display: false // Hide the default legend as we have custom gradient legend
                        }
                    },
                    scales: {
                        x: {
                            title: {
                                display: true,
                                text: 'PyPSA ' + (param === 'r' ? 'Resistance (Ω/km)' : param === 'x' ? 'Reactance (Ω/km)' : 'Susceptance (S/km)')
                            }
                        },
                        y: {
                            title: {
                                display: true,
                                text: 'JAO ' + (param === 'r' ? 'Resistance (Ω/km)' : param === 'x' ? 'Reactance (Ω/km)' : 'Susceptance (S/km)') + ' (circuit-adjusted)'
                            }
                        }
                    }
                }
            });
        }

        // Populate the data table for total values
        function populateTable() {
            const tbody = document.querySelector('#data-table tbody');
            tbody.innerHTML = '';

            data.forEach(item => {
                const row = document.createElement('tr');

                // Add cells with TOTAL values and diff_*_pct
                [
                    item.jao_id,
                    item.pypsa_id,
                    item.length_km.toFixed(2),
                    item.circuits,
                    item.original_r.toFixed(6),
                    item.allocated_r.toFixed(6),
                    item.diff_r_pct ? (Math.abs(item.diff_r_pct) > 1000 ? (item.diff_r_pct > 0 ? '+' : '-') + '999.99%' : item.diff_r_pct.toFixed(2) + '%') : 'N/A',
                    item.original_x.toFixed(6),
                    item.allocated_x.toFixed(6),
                    item.diff_x_pct ? (Math.abs(item.diff_x_pct) > 1000 ? (item.diff_x_pct > 0 ? '+' : '-') + '999.99%' : item.diff_x_pct.toFixed(2) + '%') : 'N/A',
                    item.original_b.toExponential(6),
                    item.allocated_b.toExponential(6),
                    item.diff_b_pct ? (Math.abs(item.diff_b_pct) > 1000 ? (item.diff_b_pct > 0 ? '+' : '-') + '999.99%' : item.diff_b_pct.toFixed(2) + '%') : 'N/A'
                ].forEach((text, index) => {
                    const td = document.createElement('td');
                    td.textContent = text;

                    // Add color coding for difference percentages
                    if (index === 6 || index === 9 || index === 12) { // R, X, B diff columns
                        const pct = parseFloat(text);
                        if (!isNaN(pct)) {
                            if (Math.abs(pct) <= 20) {
                                td.className = 'good-match';
                            } else if (Math.abs(pct) <= 50) {
                                td.className = 'moderate-match';
                            } else {
                                td.className = 'poor-match';
                            }
                        }
                    }

                    row.appendChild(td);
                });

                tbody.appendChild(row);
            });
        }

        // Populate the per-km data table
        function populatePerKmTable() {
            const tbody = document.querySelector('#per-km-data-table tbody');
            tbody.innerHTML = '';

            data.forEach(item => {
                const row = document.createElement('tr');

                // Add cells with PER-KM values and diff_*_pct_km
                [
                    item.jao_id,
                    item.pypsa_id,
                    item.length_km.toFixed(2),
                    item.circuits,
                    item.original_r_per_km.toFixed(6),
                    item.jao_r_km_adjusted.toFixed(6),
                    item.diff_r_pct_km ? (Math.abs(item.diff_r_pct_km) > 1000 ? (item.diff_r_pct_km > 0 ? '+' : '-') + '999.99%' : item.diff_r_pct_km.toFixed(2) + '%') : 'N/A',
                    item.original_x_per_km.toFixed(6),
                    item.jao_x_km_adjusted.toFixed(6),
                    item.diff_x_pct_km ? (Math.abs(item.diff_x_pct_km) > 1000 ? (item.diff_x_pct_km > 0 ? '+' : '-') + '999.99%' : item.diff_x_pct_km.toFixed(2) + '%') : 'N/A',
                    item.original_b_per_km.toExponential(6),
                    item.jao_b_km_adjusted.toExponential(6),
                    item.diff_b_pct_km ? (Math.abs(item.diff_b_pct_km) > 1000 ? (item.diff_b_pct_km > 0 ? '+' : '-') + '999.99%' : item.diff_b_pct_km.toFixed(2) + '%') : 'N/A'
                ].forEach((text, index) => {
                    const td = document.createElement('td');
                    td.textContent = text;

                    // Add color coding for difference percentages
                    if (index === 6 || index === 9 || index === 12) { // R, X, B diff columns
                        const pct = parseFloat(text);
                        if (!isNaN(pct)) {
                            if (Math.abs(pct) <= 20) {
                                td.className = 'good-match';
                            } else if (Math.abs(pct) <= 50) {
                                td.className = 'moderate-match';
                            } else {
                                td.className = 'poor-match';
                            }
                        }
                    }

                    row.appendChild(td);
                });

                tbody.appendChild(row);
            });
        }

        // Column filtering
        function setupFilters() {
            const inputs = document.querySelectorAll('.filters input');

            inputs.forEach(input => {
                input.addEventListener('keyup', function() {
                    const column = parseInt(this.dataset.col);
                    const value = this.value.toLowerCase();
                    const tableId = this.closest('table').id;

                    const rows = document.querySelectorAll('#' + tableId + ' tbody tr');
                    rows.forEach(row => {
                        const cell = row.querySelectorAll('td')[column];
                        const text = cell.textContent.toLowerCase();

                        // Check if row is already hidden by another filter
                        const isHidden = row.style.display === 'none';

                        // If filter is empty, don't hide based on this column
                        if (value === '' && !isHidden) {
                            row.style.display = '';
                        } else if (text.includes(value)) {
                            // If this filter matches and row isn't hidden by another filter
                            if (!isHidden) {
                                row.style.display = '';
                            }
                        } else {
                            // This filter doesn't match, hide the row
                            row.style.display = 'none';
                        }
                    });
                });
            });
        }

        // Tab switching
        function showTable(tableId) {
            document.getElementById('total-table').style.display = 'none';
            document.getElementById('per-km-table').style.display = 'none';
            document.getElementById(tableId).style.display = 'block';

            // Update tab active state
            document.querySelectorAll('.tab').forEach(tab => {
                tab.classList.remove('active');
            });

            if (tableId === 'total-table') {
                document.querySelectorAll('.tab')[0].classList.add('active');
            } else {
                document.querySelectorAll('.tab')[1].classList.add('active');
            }
        }

        // Initialize when the page loads
        document.addEventListener('DOMContentLoaded', function() {
            // Create total value charts
            createScatterPlot('r', 'r-scatter');
            createScatterPlot('x', 'x-scatter');
            createScatterPlot('b', 'b-scatter');

            // Create PyPSA vs JAO comparison charts
            createComparisonChart('r', 'r-comparison');
            createComparisonChart('x', 'x-comparison');
            createComparisonChart('b', 'b-comparison');

            // Create per-km comparison charts
            createPerKmChart('r', 'r-per-km');
            createPerKmChart('x', 'x-per-km');
            createPerKmChart('b', 'b-per-km');

            // Initialize data tables
            populateTable();
            populatePerKmTable();
            setupFilters();
        });
    </script>
</body>
</html>
"""

    # Format R values for display
    r_value_text = {}
    for param in ['r', 'x', 'b']:
        if r_values[param]['r'] is not None:
            r_value_text[param] = f"{r_values[param]['r']:.4f}"
        else:
            r_value_text[param] = "N/A"

    # Replace placeholders with actual data
    html = html.replace("DATA_PLACEHOLDER", json.dumps(all_data))
    html = html.replace("R_VALUES", json.dumps(r_value_text))

    # Write HTML file
    output_file = output_path / "parameter_comparison.html"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Parameter comparison visualization saved to {output_file}")
    return str(output_file)