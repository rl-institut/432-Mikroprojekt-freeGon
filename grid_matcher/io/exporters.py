import pandas as pd
import geopandas as gpd


def create_results_csv(matches, output_path):
    """Save matching results to a CSV file."""
    print(f"Saving results to {output_path}...")

    # Expand pypsa_ids to semicolon-separated string
    for match in matches:
        if 'pypsa_ids' in match and isinstance(match['pypsa_ids'], list):
            match['pypsa_ids'] = ';'.join(match['pypsa_ids'])

    # Create a DataFrame from matches
    results_df = pd.DataFrame(matches)

    # Save to CSV
    results_df.to_csv(output_path, index=False)

    print(f"Results saved to {output_path}")
    return output_path

from pathlib import Path
import os

def generate_pypsa_with_eic(matches, jao_gdf, pypsa_gdf, output_dir):
    """
    Generate and save PyPSA lines with EIC codes from JAO matching.

    This function preserves all original columns from the PyPSA dataframe and
    adds only the EIC_Code column right after the line_id column.

    Parameters:
    -----------
    matches : list
        List of match dictionaries from the matching process
    jao_gdf : GeoDataFrame
        GeoDataFrame containing JAO lines with EIC codes
    pypsa_gdf : GeoDataFrame
        GeoDataFrame containing PyPSA lines
    output_dir : Path or str
        Directory to save output files
    """
    print("\n===== GENERATING PYPSA WITH EIC CODES =====")

    # Convert output_dir to Path object if it's a string
    output_dir = Path(output_dir) if not isinstance(output_dir, Path) else output_dir

    # Create a mapping from PyPSA ID to JAO EIC_Code
    pypsa_to_eic = {}

    # Get the correct ID column name (line_id)
    id_col = 'line_id'  # First column from the input file

    # First, make sure all pypsa_ids are correctly processed as strings
    for match in matches:
        if match.get('matched', False) and 'pypsa_ids' in match:
            jao_id = str(match['jao_id'])
            jao_rows = jao_gdf[jao_gdf['id'].astype(str) == jao_id]

            if not jao_rows.empty:
                eic_code = str(jao_rows.iloc[0].get('EIC_Code', ''))

                if eic_code:
                    # Get PyPSA IDs from match and ensure they're strings
                    pypsa_ids = match['pypsa_ids']
                    if isinstance(pypsa_ids, str):
                        pypsa_ids = [pid.strip() for pid in pypsa_ids.split(';')]

                    # Add to mapping with explicit string conversion
                    for pypsa_id in pypsa_ids:
                        pypsa_id = str(pypsa_id)
                        pypsa_to_eic[pypsa_id] = eic_code

    print(f"Created mapping for {len(pypsa_to_eic)} PyPSA IDs")

    # Create a copy of the original PyPSA dataframe to preserve all columns
    pypsa_with_eic = pypsa_gdf.copy()

    # Create the EIC_Code column
    pypsa_with_eic['EIC_Code'] = pypsa_with_eic[id_col].astype(str).map(pypsa_to_eic).fillna('')

    # Reorder columns to place EIC_Code right after line_id (second column)
    columns = list(pypsa_with_eic.columns)
    id_index = columns.index(id_col)
    columns.remove('EIC_Code')
    columns.insert(id_index + 1, 'EIC_Code')  # Place as second column
    pypsa_with_eic = pypsa_with_eic[columns]

    # Calculate match statistics for PyPSA (for reporting only)
    pypsa_match_count = sum(pypsa_with_eic['EIC_Code'] != '')
    pypsa_match_rate = pypsa_match_count / len(pypsa_gdf) if len(pypsa_gdf) > 0 else 0

    print(f"Assigned EIC codes to {pypsa_match_count} out of {len(pypsa_gdf)} PyPSA lines ({pypsa_match_rate:.1%})")

    # Save to CSV
    pypsa_eic_file = output_dir / 'pypsa_with_eic.csv'
    pypsa_with_eic.to_csv(pypsa_eic_file, index=False)
    print(f"Saved PyPSA with EIC codes to CSV: {pypsa_eic_file}")

    return pypsa_match_count, pypsa_with_eic, [pypsa_eic_file]

def generate_jao_with_pypsa(matches, jao_gdf, pypsa_gdf, output_dir):
    """
    Generate and save JAO lines with mapped PyPSA IDs.

    Parameters:
    -----------
    matches : list
        List of match dictionaries from the matching process
    jao_gdf : GeoDataFrame
        GeoDataFrame containing JAO lines
    pypsa_gdf : GeoDataFrame
        GeoDataFrame containing PyPSA lines
    output_dir : Path
        Directory to save output files

    Returns:
    --------
    tuple
        (jao_with_pypsa_df, output_files)
    """
    # Use same approach to avoid geometry issues
    jao_data = {
        'id': [],
        'EIC_Code': [],
        'NE_name': [],
        'v_nom': [],
        'length_km': [],
        'r_per_km': [],
        'x_per_km': [],
        'b_per_km_uS': [],
        'PyPSA_ids': [],
        'match_quality': [],
        'has_match': []
    }

    # Add any other columns from original dataframe that might be useful
    for col in jao_gdf.columns:
        if col not in jao_data and col not in ['geometry', 'start_point', 'end_point']:
            jao_data[col] = []

    # Populate the data dictionary from jao_gdf
    for _, row in jao_gdf.iterrows():
        jao_id = str(row['id'])

        # Find match for this JAO ID
        match_info = next((m for m in matches if str(m.get('jao_id', '')) == jao_id and m.get('matched', False)), None)

        pypsa_ids = []
        match_quality = ''
        has_match = False

        if match_info:
            has_match = True
            match_quality = match_info.get('match_quality', '')

            # Get PyPSA IDs
            if 'pypsa_ids' in match_info:
                if isinstance(match_info['pypsa_ids'], list):
                    pypsa_ids = [str(pid) for pid in match_info['pypsa_ids']]
                else:
                    pypsa_ids = [str(pid).strip() for pid in match_info['pypsa_ids'].split(';')]

        # Add row data to our dictionary
        for col in jao_data:
            if col == 'id':
                jao_data[col].append(jao_id)
            elif col == 'PyPSA_ids':
                jao_data[col].append(';'.join(pypsa_ids))
            elif col == 'match_quality':
                jao_data[col].append(match_quality)
            elif col == 'has_match':
                jao_data[col].append(has_match)
            elif col in row:
                jao_data[col].append(row[col])
            else:
                jao_data[col].append(None)

    # Create DataFrame from the dictionary
    jao_with_pypsa = pd.DataFrame(jao_data)

    # Save to CSV
    jao_pypsa_file = output_dir / 'jao_with_pypsa.csv'
    jao_with_pypsa.to_csv(jao_pypsa_file, index=False)
    print(f"Saved JAO with PyPSA IDs to CSV: {jao_pypsa_file}")

    output_files = [jao_pypsa_file]

    # Create a GeoDataFrame version for GIS formats
    try:
        # Create a clean copy of geometry data
        jao_geometry_data = []
        for _, row in jao_gdf.iterrows():
            jao_geometry_data.append(row.geometry)

        # Create a new GeoDataFrame with our data and the original geometries
        jao_with_pypsa_geo = gpd.GeoDataFrame(
            jao_with_pypsa,
            geometry=jao_geometry_data,
            crs=jao_gdf.crs
        )

        # Save as GeoJSON
        jao_pypsa_geojson = output_dir / 'jao_with_pypsa.geojson'
        jao_with_pypsa_geo.to_file(jao_pypsa_geojson, driver='GeoJSON')
        print(f"Saved JAO with PyPSA IDs to GeoJSON: {jao_pypsa_geojson}")
        output_files.append(jao_pypsa_geojson)

    except Exception as e:
        print(f"Error creating JAO GeoJSON file: {e}")
        print("Saving JAO WKT geometries to CSV instead...")

        # Add WKT geometry column to the CSV
        jao_wkt_geometries = []
        for _, row in jao_gdf.iterrows():
            if row.geometry is not None:
                jao_wkt_geometries.append(row.geometry.wkt)
            else:
                jao_wkt_geometries.append(None)

        # Add WKT column to the DataFrame
        jao_with_pypsa['geometry_wkt'] = jao_wkt_geometries

        # Save updated CSV
        jao_with_pypsa.to_csv(jao_pypsa_file, index=False)
        print(f"Saved JAO with PyPSA IDs and WKT geometries to CSV: {jao_pypsa_file}")

    return jao_with_pypsa, output_files


def generate_matching_statistics(matches, jao_gdf, pypsa_gdf, pypsa_match_count, output_dir):
    """
    Generate detailed matching statistics and save to CSV.

    Parameters:
    -----------
    matches : list
        List of match dictionaries from the matching process
    jao_gdf : GeoDataFrame
        GeoDataFrame containing JAO lines
    pypsa_gdf : GeoDataFrame
        GeoDataFrame containing PyPSA lines
    pypsa_match_count : int
        Number of matched PyPSA lines
    output_dir : Path
        Directory to save output files

    Returns:
    --------
    Path
        Path to the statistics CSV file
    """
    # Count matches
    matched_count = sum(1 for m in matches if m.get('matched', False))
    match_rate = matched_count / len(jao_gdf) if len(jao_gdf) > 0 else 0
    pypsa_match_rate = pypsa_match_count / len(pypsa_gdf) if len(pypsa_gdf) > 0 else 0

    # Quality breakdown
    quality_counts = {
        'Excellent': 0,
        'Good': 0,
        'Fair': 0,
        'Poor': 0,
        'Bus-Based': 0,
        'Parallel': 0,
        'Visual': 0,
        'Desperate': 0
    }

    for match in matches:
        if match.get('matched', False):
            quality = match.get('match_quality', '')
            if 'Bus-Based' in quality:
                quality_counts['Bus-Based'] += 1
            elif 'Excellent' in quality:
                quality_counts['Excellent'] += 1
            elif 'Good' in quality:
                quality_counts['Good'] += 1
            elif 'Fair' in quality:
                quality_counts['Fair'] += 1
            elif 'Visual' in quality or 'Enhanced Visual' in quality:
                quality_counts['Visual'] += 1
            elif 'Desperate' in quality:
                quality_counts['Desperate'] += 1
            else:
                quality_counts['Poor'] += 1

            if match.get('is_parallel_circuit', False) or 'Parallel Circuit' in quality:
                quality_counts['Parallel'] += 1

    # Create statistics data structure
    match_stats = []

    # Count by voltage level
    jao_voltage_groups = jao_gdf.groupby('v_nom').size().to_dict()
    jao_matched_by_voltage = {}

    for match in matches:
        if match.get('matched', False):
            jao_id = match['jao_id']
            jao_row = jao_gdf[jao_gdf['id'] == jao_id]
            if not jao_row.empty:
                voltage = jao_row.iloc[0].get('v_nom', 0)
                jao_matched_by_voltage[voltage] = jao_matched_by_voltage.get(voltage, 0) + 1

    for voltage, count in sorted(jao_voltage_groups.items()):
        matched = jao_matched_by_voltage.get(voltage, 0)
        match_stats.append({
            'Category': f'JAO {voltage}kV lines',
            'Total': count,
            'Matched': matched,
            'Unmatched': count - matched,
            'Match Rate': f"{matched / count:.1%}" if count > 0 else "N/A"
        })

    # Count by match quality
    for quality, count in quality_counts.items():
        if count > 0:
            match_stats.append({
                'Category': f'Match Quality: {quality}',
                'Total': count,
                'Matched': count,
                'Unmatched': 0,
                'Match Rate': "100.0%"
            })

    # Overall stats
    match_stats.append({
        'Category': 'JAO TOTAL',
        'Total': len(jao_gdf),
        'Matched': matched_count,
        'Unmatched': len(jao_gdf) - matched_count,
        'Match Rate': f"{match_rate:.1%}"
    })

    match_stats.append({
        'Category': 'PyPSA TOTAL',
        'Total': len(pypsa_gdf),
        'Matched': pypsa_match_count,
        'Unmatched': len(pypsa_gdf) - pypsa_match_count,
        'Match Rate': f"{pypsa_match_rate:.1%}"
    })

    # Save stats to CSV
    stats_file = output_dir / 'matching_statistics.csv'
    pd.DataFrame(match_stats).to_csv(stats_file, index=False)
    print(f"Matching statistics saved to: {stats_file}")

    # Print summary to console
    print(f"\nMatching Results Summary:")
    print(f"Total JAO Lines: {len(jao_gdf)}")
    print(f"Total PyPSA Lines: {len(pypsa_gdf)}")
    print(f"Matched JAO Lines: {matched_count}/{len(jao_gdf)} ({match_rate:.1%})")
    print(f"Matched PyPSA Lines: {pypsa_match_count}/{len(pypsa_gdf)} ({pypsa_match_rate:.1%})")

    print(f"\nMatch Quality Breakdown:")
    print(f"  Bus-Based Matches: {quality_counts['Bus-Based']}")
    print(f"  Excellent: {quality_counts['Excellent']}")
    print(f"  Good: {quality_counts['Good']}")
    print(f"  Fair: {quality_counts['Fair']}")
    print(f"  Visual Matches: {quality_counts['Visual']}")
    print(f"  Desperate Matches: {quality_counts['Desperate']}")
    print(f"  Poor: {quality_counts['Poor']}")
    print(f"  Parallel Circuits: {quality_counts['Parallel']}")

    return stats_file


def export_unmatched_pypsa_details(matches, jao_gdf, pypsa_gdf, output_dir):
    """
    Analyze and export details about unmatched PyPSA lines including likely reasons
    for not being matched.

    Parameters:
    -----------
    matches : list
        List of match dictionaries from the matching process
    jao_gdf : GeoDataFrame
        GeoDataFrame containing JAO lines
    pypsa_gdf : GeoDataFrame
        GeoDataFrame containing PyPSA lines
    output_dir : Path
        Directory to save output files

    Returns:
    --------
    Path
        Path to the output file
    """
    import pandas as pd
    from pathlib import Path
    import numpy as np
    from shapely.geometry import Point

    print("\n===== ANALYZING UNMATCHED PYPSA LINES =====")

    # Get all matched PyPSA IDs
    matched_pypsa_ids = set()
    for match in matches:
        if match.get('matched', False):
            pypsa_ids = match.get('pypsa_ids', [])
            if isinstance(pypsa_ids, str):
                pypsa_ids = [pid.strip() for pid in pypsa_ids.split(';') if pid.strip()]
            matched_pypsa_ids.update(pypsa_ids)

    print(f"Found {len(matched_pypsa_ids)} matched PyPSA lines")

    # Find unmatched PyPSA lines
    unmatched_pypsa = []
    for _, row in pypsa_gdf.iterrows():
        pypsa_id = str(row.get('line_id', row.get('id', '')))
        if pypsa_id not in matched_pypsa_ids:
            unmatched_pypsa.append(row)

    print(f"Found {len(unmatched_pypsa)} unmatched PyPSA lines")

    if not unmatched_pypsa:
        print("No unmatched PyPSA lines to analyze")
        return None

    # Function to safely get length in km
    def length_km(row):
        try:
            length = float(row.get('length', 0))
            return length / 1000.0 if length > 1000 else length
        except (TypeError, ValueError):
            return 0

    # Analyze unmatched lines
    results = []
    for row in unmatched_pypsa:
        pypsa_id = str(row.get('line_id', row.get('id', '')))
        voltage = row.get('voltage', row.get('v_nom', 0))
        bus0 = row.get('bus0', '')
        bus1 = row.get('bus1', '')
        length = length_km(row)
        geom = row.geometry

        # Determine likely reasons for not matching
        reasons = []

        # Reason 1: Voltage - non-standard voltage classes are harder to match
        if voltage not in [220, 380, 400]:
            reasons.append("Non-standard voltage class")

        # Reason 2: Length - very short lines often don't have matches
        if length < 1.0:
            reasons.append("Very short line (<1km)")

        # Reason 3: Isolated - not connected to main network
        if not (bus0 and bus1):
            reasons.append("Missing bus connections")

        # Reason 4: Geographic isolation - check if there are JAO lines nearby
        has_nearby_jao = False
        if geom is not None:
            buffer_dist = 0.01  # ~1km in degrees
            buffer = geom.buffer(buffer_dist)
            for _, jao_row in jao_gdf.iterrows():
                if jao_row.geometry is not None and buffer.intersects(jao_row.geometry):
                    has_nearby_jao = True
                    break

        if not has_nearby_jao:
            reasons.append("No nearby JAO lines")

        # Reason 5: Already at circuit capacity
        # (This would require tracking which PyPSA lines were considered but rejected)

        # If no specific reasons identified
        if not reasons:
            if length > 20:
                reasons.append("Long line, possible cross-border or in area not covered by JAO")
            else:
                reasons.append("Unknown (may be duplicate path or parallel circuit)")

        results.append({
            'pypsa_id': pypsa_id,
            'voltage': voltage,
            'bus0': bus0,
            'bus1': bus1,
            'length_km': length,
            'reasons_unmatched': '; '.join(reasons)
        })

    # Create DataFrame and save to CSV
    df = pd.DataFrame(results)

    # Sort by voltage then length
    df = df.sort_values(['voltage', 'length_km'], ascending=[False, False])

    # Save to CSV
    output_path = Path(output_dir) / 'unmatched_pypsa_analysis.csv'
    df.to_csv(output_path, index=False)

    # Generate HTML report
    html_path = Path(output_dir) / 'unmatched_pypsa_analysis.html'

    # Group by reason for summary
    reason_counts = {}
    for r in results:
        reasons = r['reasons_unmatched'].split('; ')
        for reason in reasons:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

    # Sort reasons by frequency
    sorted_reasons = sorted(reason_counts.items(), key=lambda x: x[1], reverse=True)

    # Create HTML content with advanced filtering and sorting
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Unmatched PyPSA Lines Analysis</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
            h1, h2 {{ color: #333; }}
            table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; position: sticky; top: 0; }}
            .summary {{ background-color: #f9f9f9; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
            .filters {{ margin-bottom: 20px; display: flex; flex-wrap: wrap; gap: 10px; }}
            .filter-group {{ display: flex; flex-direction: column; margin-right: 15px; }}
            .filter-group label {{ font-weight: bold; margin-bottom: 5px; }}
            input[type=text], select {{ padding: 8px; width: 180px; }}
            th.sortable {{ cursor: pointer; }}
            th.sortable::after {{ content: "⇅"; margin-left: 5px; color: #999; }}
            th.sort-asc::after {{ content: "↑"; color: #333; }}
            th.sort-desc::after {{ content: "↓"; color: #333; }}
            .table-container {{ max-height: 70vh; overflow-y: auto; }}
            .hidden {{ display: none; }}
        </style>
    </head>
    <body>
        <h1>Unmatched PyPSA Lines Analysis</h1>

        <div class="summary">
            <h2>Summary</h2>
            <p>Total unmatched PyPSA lines: {len(unmatched_pypsa)}</p>

            <h3>Reasons for Unmatched Status:</h3>
            <ul>
    """

    for reason, count in sorted_reasons:
        percentage = 100 * count / len(unmatched_pypsa)
        html_content += f"<li><strong>{reason}</strong>: {count} lines ({percentage:.1f}%)</li>\n"

    html_content += """
            </ul>
        </div>

        <div class="filters">
            <div class="filter-group">
                <label for="filterPyPSAID">PyPSA ID</label>
                <input type="text" id="filterPyPSAID" placeholder="Filter PyPSA ID">
            </div>
            <div class="filter-group">
                <label for="filterVoltage">Voltage (kV)</label>
                <input type="text" id="filterVoltage" placeholder="Filter voltage">
            </div>
            <div class="filter-group">
                <label for="filterBus0">Bus0</label>
                <input type="text" id="filterBus0" placeholder="Filter bus0">
            </div>
            <div class="filter-group">
                <label for="filterBus1">Bus1</label>
                <input type="text" id="filterBus1" placeholder="Filter bus1">
            </div>
            <div class="filter-group">
                <label for="filterLength">Length (km)</label>
                <input type="text" id="filterLength" placeholder="Filter length">
            </div>
            <div class="filter-group">
                <label for="filterReasons">Reasons</label>
                <input type="text" id="filterReasons" placeholder="Filter reasons">
            </div>
            <div class="filter-group">
                <label>&nbsp;</label>
                <button onclick="clearFilters()">Clear Filters</button>
            </div>
        </div>

        <div class="table-container">
            <table id="resultsTable">
                <thead>
                    <tr>
                        <th class="sortable" onclick="sortTable(0)">PyPSA ID</th>
                        <th class="sortable" onclick="sortTable(1)">Voltage (kV)</th>
                        <th class="sortable" onclick="sortTable(2)">Bus0</th>
                        <th class="sortable" onclick="sortTable(3)">Bus1</th>
                        <th class="sortable" onclick="sortTable(4)">Length (km)</th>
                        <th class="sortable" onclick="sortTable(5)">Reasons Unmatched</th>
                    </tr>
                </thead>
                <tbody>
    """

    for r in results:
        html_content += f"""
                    <tr>
                        <td>{r['pypsa_id']}</td>
                        <td>{r['voltage']}</td>
                        <td>{r['bus0']}</td>
                        <td>{r['bus1']}</td>
                        <td>{r['length_km']:.2f}</td>
                        <td>{r['reasons_unmatched']}</td>
                    </tr>
        """

    html_content += """
                </tbody>
            </table>
        </div>

        <script>
            // Column filtering functionality
            const filters = {
                'filterPyPSAID': 0,
                'filterVoltage': 1,
                'filterBus0': 2,
                'filterBus1': 3,
                'filterLength': 4,
                'filterReasons': 5
            };

            // Add event listeners to all filter inputs
            for (const [filterId, colIndex] of Object.entries(filters)) {
                document.getElementById(filterId).addEventListener('input', function() {
                    filterTable();
                });
            }

            function filterTable() {
                const table = document.getElementById('resultsTable');
                const rows = table.getElementsByTagName('tr');

                // Skip header row
                for (let i = 1; i < rows.length; i++) {
                    let showRow = true;
                    const row = rows[i];

                    // Check each filter
                    for (const [filterId, colIndex] of Object.entries(filters)) {
                        const filterValue = document.getElementById(filterId).value.toLowerCase();
                        if (filterValue) {
                            const cell = row.getElementsByTagName('td')[colIndex];
                            const cellText = cell.textContent.toLowerCase();

                            if (cellText.indexOf(filterValue) === -1) {
                                showRow = false;
                                break;
                            }
                        }
                    }

                    // Show or hide row
                    row.style.display = showRow ? '' : 'none';
                }
            }

            function clearFilters() {
                for (const filterId of Object.keys(filters)) {
                    document.getElementById(filterId).value = '';
                }
                filterTable();
            }

            // Sorting functionality
            let currentSortCol = -1;
            let sortAscending = true;

            function sortTable(colIndex) {
                const table = document.getElementById('resultsTable');
                const tbody = table.getElementsByTagName('tbody')[0];
                const rows = Array.from(tbody.getElementsByTagName('tr'));

                // Update sort direction
                if (currentSortCol === colIndex) {
                    sortAscending = !sortAscending;
                } else {
                    sortAscending = true;

                    // Reset all column headers
                    const headers = table.getElementsByTagName('th');
                    for (let i = 0; i < headers.length; i++) {
                        headers[i].classList.remove('sort-asc', 'sort-desc');
                    }
                }

                // Update sort indicator
                const header = table.getElementsByTagName('th')[colIndex];
                header.classList.remove('sort-asc', 'sort-desc');
                header.classList.add(sortAscending ? 'sort-asc' : 'sort-desc');

                currentSortCol = colIndex;

                // Sort rows
                rows.sort((a, b) => {
                    let valueA = a.getElementsByTagName('td')[colIndex].textContent;
                    let valueB = b.getElementsByTagName('td')[colIndex].textContent;

                    // Handle numeric values (voltage, length)
                    if (colIndex === 1 || colIndex === 4) {
                        valueA = parseFloat(valueA) || 0;
                        valueB = parseFloat(valueB) || 0;
                        return sortAscending ? valueA - valueB : valueB - valueA;
                    } 
                    // Text values
                    else {
                        return sortAscending 
                            ? valueA.localeCompare(valueB) 
                            : valueB.localeCompare(valueA);
                    }
                });

                // Reorder rows
                rows.forEach(row => tbody.appendChild(row));
            }

            // Initial sort by voltage (descending)
            document.addEventListener('DOMContentLoaded', function() {
                sortTable(1);
                sortTable(1); // Toggle to descending
            });
        </script>
    </body>
    </html>
    """

    with open(html_path, 'w') as f:
        f.write(html_content)

    print(f"Saved unmatched PyPSA analysis to {output_path} and {html_path}")
    return html_path



def export_enhanced_matching_results(matching_results, jao_gdf, pypsa_gdf, pypsa_110_gdf=None,
                                     output_path="output/enhanced_matching_results.csv"):
    """
    Export enhanced matching results to CSV, including 110kV PyPSA data if provided.

    Parameters:
    -----------
    matching_results : list
        List of dictionaries containing JAO-PyPSA matching results
    jao_gdf : GeoDataFrame
        GeoDataFrame containing the JAO lines data
    pypsa_gdf : GeoDataFrame
        GeoDataFrame containing the PyPSA lines data
    pypsa_110_gdf : GeoDataFrame, optional
        GeoDataFrame containing the 110kV PyPSA lines data
    output_path : str, default="output/enhanced_matching_results.csv"
        Path to save the CSV file
    """
    import pandas as pd
    import os
    from pathlib import Path

    print(f"Exporting enhanced matching results to {output_path}...")

    # Create a list to store the rows for the CSV
    rows = []

    # Add a field to indicate the data source
    for result in matching_results:
        # Basic information
        row = {
            'jao_id': result.get('jao_id', ''),
            'jao_name': result.get('jao_name', ''),
            'v_nom': result.get('v_nom', ''),
            'matched': result.get('matched', False),
            'data_source': result.get('source', 'main-matching-process'),
            'reason': result.get('reason', '')
        }

        # Add matched PyPSA IDs if available
        if result.get('matched', False):
            pypsa_ids = result.get('pypsa_ids', [])
            if isinstance(pypsa_ids, str):
                pypsa_ids = [p.strip() for p in pypsa_ids.split(";") if p.strip()]
            row['pypsa_ids'] = ';'.join(map(str, pypsa_ids))

            # Add length information
            row['jao_length_km'] = result.get('jao_length_km', 0)

            # Add electrical parameters
            for param in ['r', 'x', 'b']:
                jao_param = result.get(f'jao_{param}', None)
                if jao_param is not None:
                    row[f'jao_{param}'] = jao_param

        rows.append(row)

    # Create DataFrame and export to CSV
    df = pd.DataFrame(rows)

    # Ensure output directory exists
    Path(output_path).parent.mkdir(exist_ok=True, parents=True)

    # Export to CSV
    df.to_csv(output_path, index=False)
    print(f"Exported {len(df)} records to {output_path}")

    return output_path
