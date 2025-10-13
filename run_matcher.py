#!/usr/bin/env python
# run_matcher.py - Custom script for handling complex data formats

import os
import sys
import re
import pandas as pd
import geopandas as gpd
import argparse
from pathlib import Path
from shapely.geometry import LineString
from grid_matcher.matcher.original_matcher import run_original_matching, load_data
from grid_matcher.visualization.comparison import visualize_parameter_comparison
# Import functions from manual_matching.py
from grid_matcher.manual.manual_matching import (
    load_manual_matches_file, save_manual_matches_file,
    validate_jao_id, validate_pypsa_ids,
    apply_manual_matches,
    add_predefined_manual_matches, import_new_lines_from_csv
)
# Import directly from exporters and visualization for regenerating outputs
from grid_matcher.io.exporters import (
    create_results_csv, generate_pypsa_with_eic, generate_jao_with_pypsa
)
from grid_matcher.visualization.maps import create_jao_pypsa_visualization
from grid_matcher.visualization.reports import create_enhanced_summary_table


# Parse command-line arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description="Grid Matcher matcher Mode")

    # Matching inclusion options
    parser.add_argument("--include-dc-matching", action="store_true", help="Include DC links in matching")
    parser.add_argument("--include-110kv-matching", action="store_true", help="Include 110kV lines in matching")

    # Output inclusion options
    parser.add_argument("--no-dc-output", action="store_true", help="Exclude DC links from output")
    parser.add_argument("--no-110kv-output", action="store_true", help="Exclude 110kV lines from output")

    # Visualization options
    parser.add_argument("--no-viz", action="store_true", help="Skip parameter visualization")

    # Verbosity option
    parser.add_argument("--quiet", action="store_true", help="Disable verbose output")

    # Manual matching options
    parser.add_argument("--manual", action="store_true", default=None, help="Enable manual matching")
    parser.add_argument("--no-manual", action="store_true", help="Disable manual matching")
    parser.add_argument("--add-predefined", action="store_true", default=None, help="Add predefined manual matches")
    parser.add_argument("--no-predefined", action="store_true", help="Don't add predefined matches")
    parser.add_argument("--import-new-lines", action="store_true", help="Import new lines from CSV files")

    # Output directory
    parser.add_argument("--output", "-o", type=str, help="Output directory", default="output/matcher")

    return parser.parse_args()

# ===== CONFIGURATION =====
# Skip DC and 110kV in matching for speed
INCLUDE_DC_IN_MATCHING = False
INCLUDE_110KV_IN_MATCHING = False

# Control whether to include DC and 110kV in the enhanced output
INCLUDE_DC_IN_OUTPUT = True  # Set to False to exclude DC links from final output
INCLUDE_110KV_IN_OUTPUT = True  # Set to False to exclude 110kV lines from final output

# Control whether to generate parameter comparison visualization
GENERATE_PARAMETER_VISUALIZATION = True

# Control manual matching options
ENABLE_MANUAL_MATCHING = True
ADD_PREDEFINED_MATCHES = True
IMPORT_NEW_LINES = False

VERBOSE = True

# File paths
DATA_DIR = Path("grid_matcher/data")
MANUAL_MATCHES_FILE = DATA_DIR / "manual_matches.json"  # Path to manual matches file

JAO_PATH = DATA_DIR / "jao_lines.csv"
PYPSA_PATH = DATA_DIR / "pypsa_lines.csv"
PYPSA_110KV_PATH = DATA_DIR / "pypsa_lines_110kv.csv"
PYPSA_DC_PATH = DATA_DIR / "pypsa_dc_links.csv"


def parse_dc_links_direct(filepath):
    """Parse DC links data with direct manual method"""
    print(f"Parsing DC links from {filepath}")

    dc_links = []

    try:
        with open(filepath, 'r') as f:
            # Read header
            header_line = f.readline().strip()

            # Process data lines
            for line_num, line in enumerate(f, start=2):
                try:
                    if line.strip().startswith(('relation/', 'way/')):
                        # Extract data fields
                        fields = {}

                        # Find LINESTRING section
                        linestring_start = line.find('LINESTRING')
                        if linestring_start < 0:
                            continue

                        # Extract parts before LINESTRING
                        parts = line[:linestring_start].strip().split('\t')
                        if len(parts) < 3:
                            parts = line[:linestring_start].strip().split(',')

                        # Extract ID and other fields
                        fields['link_id'] = parts[0].strip() if len(parts) > 0 else f"dc-link-{line_num}"
                        fields['bus0'] = parts[1].strip() if len(parts) > 1 else None
                        fields['bus1'] = parts[2].strip() if len(parts) > 2 else None

                        # Extract voltage - look for pattern like "320-DC"
                        voltage_match = re.search(r'(\d+)[-\s]?DC', line)
                        if voltage_match:
                            fields['voltage'] = int(voltage_match.group(1))
                        else:
                            fields['voltage'] = 0

                        # Extract length
                        length_match = re.search(r'\d+\.\d+(?=\s+True|\s+False)', line)
                        if length_match:
                            fields['length'] = float(length_match.group(0))
                        else:
                            fields['length'] = 0

                        # Extract underground and under_construction
                        fields['underground'] = 'True' in line.split('\t')
                        fields['under_construction'] = 'False' in line.split('\t')

                        # Extract LINESTRING geometry
                        geom_str = line[linestring_start:].strip()

                        # Parse coordinates
                        match = re.search(r'LINESTRING\s*\((.*?)\)', geom_str)
                        if match:
                            coords_text = match.group(1)
                            coords = []

                            # Extract coordinate pairs
                            for coord_pair in re.findall(r'(-?\d+\.\d+)\s+(-?\d+\.\d+)', coords_text):
                                x, y = float(coord_pair[0]), float(coord_pair[1])
                                coords.append((x, y))

                            if len(coords) >= 2:
                                fields['geometry'] = LineString(coords)
                                dc_links.append(fields)
                                print(f"Extracted DC link {fields['link_id']} with {len(coords)} points")

                except Exception as e:
                    print(f"Error parsing line {line_num}: {str(e)}")

        print(f"Extracted {len(dc_links)} DC links")
        return dc_links

    except Exception as e:
        print(f"Error reading file: {str(e)}")
        return []


def parse_110kv_links_direct(filepath):
    """Parse 110kV links data with direct manual method"""
    print(f"Parsing 110kV links from {filepath}")

    kv110_links = []

    try:
        with open(filepath, 'r') as f:
            # Read header
            header_line = f.readline().strip()

            # Process data lines
            for line_num, line in enumerate(f, start=2):
                try:
                    if line.strip().startswith(('relation/', 'way/', 'merged')):
                        # Extract data fields
                        fields = {}

                        # Find LINESTRING section
                        linestring_start = line.find('LINESTRING')
                        if linestring_start < 0:
                            continue

                        # Split the line before LINESTRING by tab (or comma if needed)
                        parts = line[:linestring_start].strip().split('\t')
                        if len(parts) < 5:  # Not enough parts with tab separator
                            parts = line[:linestring_start].strip().split(',')

                        # Extract ID and type
                        fields['line_id'] = parts[0].strip() if len(parts) > 0 else f"110kv-line-{line_num}"
                        fields['bus0'] = parts[1].strip() if len(parts) > 1 else None
                        fields['bus1'] = parts[2].strip() if len(parts) > 2 else None

                        # Extract voltage
                        if len(parts) > 3 and parts[3].strip().isdigit():
                            fields['voltage'] = int(parts[3].strip())
                        else:
                            fields['voltage'] = 110  # Default to 110kV

                        # Extract other parameters with regex - match numbers between fields
                        current_match = re.search(r'(?<=\s)(\d+\.\d+)(?=\s+\d+\s)', line)
                        if current_match:
                            fields['i_nom'] = float(current_match.group(1))

                        # Extract circuits
                        circuits_match = re.search(r'(?<=\s)(\d+)(?=\s+\d+\.\d+\s)', line)
                        if circuits_match:
                            fields['circuits'] = int(circuits_match.group(1))

                        # Extract other electrical parameters (more complex regex)
                        params = re.findall(r'(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)', line)
                        if params and len(params[0]) >= 4:
                            fields['s_nom'] = float(params[0][0])
                            fields['r'] = float(params[0][1])
                            fields['x'] = float(params[0][2])
                            fields['b'] = float(params[0][3])

                        # Extract length
                        length_match = re.search(r'(\d+\.\d+)(?=\s+(?:True|False)\s+(?:True|False))', line)
                        if length_match:
                            fields['length'] = float(length_match.group(1))

                        # Extract underground and under_construction
                        fields['underground'] = 'True' in line.split('\t')
                        fields['under_construction'] = 'False' in line.split('\t')

                        # Extract type
                        type_match = re.search(r'(?:True|False)\s+(?:True|False)\s+([^L]+)(?=\s*LINESTRING)', line)
                        if type_match:
                            fields['type'] = type_match.group(1).strip()

                        # Extract LINESTRING geometry
                        geom_str = line[linestring_start:].strip()

                        # Parse coordinates
                        match = re.search(r'LINESTRING\s*\((.*?)\)', geom_str)
                        if match:
                            coords_text = match.group(1)
                            coords = []

                            # Extract coordinate pairs
                            for coord_pair in re.findall(r'(-?\d+\.\d+)\s+(-?\d+\.\d+)', coords_text):
                                x, y = float(coord_pair[0]), float(coord_pair[1])
                                coords.append((x, y))

                            if len(coords) >= 2:
                                fields['geometry'] = LineString(coords)
                                kv110_links.append(fields)

                except Exception as e:
                    print(f"Error parsing line {line_num}: {str(e)}")

        print(f"Extracted {len(kv110_links)} 110kV links")
        return kv110_links

    except Exception as e:
        print(f"Error reading file: {str(e)}")
        return []


def append_to_csv_export(output_dir, dc_links=None, links_110kv=None, include_dc=True, include_110kv=True):
    """Append DC and 110kV data to the output CSV file."""
    pypsa_csv_path = output_dir / "pypsa_with_eic.csv"
    if not os.path.exists(pypsa_csv_path):
        print(f"PyPSA output file not found: {pypsa_csv_path}")
        return

    try:
        print(f"Appending to output CSV file: {pypsa_csv_path}")
        pypsa_df = pd.read_csv(pypsa_csv_path)
        original_count = len(pypsa_df)

        additional_records = []

        # Process DC links
        if dc_links and include_dc:
            print(f"Adding {len(dc_links)} DC links to export")
            for link in dc_links:
                # Create record with all standard fields as None
                record = {col: None for col in pypsa_df.columns}

                # Set ID
                record['id'] = link.get('link_id', '')

                # Also preserve the original link_id in line_id column
                record['line_id'] = link.get('link_id', '')

                # Set voltage with DC marker
                voltage = link.get('voltage')
                record['voltage'] = f"{voltage}-DC" if voltage else "0-DC"

                # Set bus connections
                record['bus0'] = link.get('bus0')
                record['bus1'] = link.get('bus1')

                # Set length
                record['length'] = link.get('length')

                # Set underground/under_construction flags
                record['underground'] = link.get('underground', False)
                record['under_construction'] = link.get('under_construction', False)

                # Set source marker
                record['source'] = 'DC'

                # Set geometry
                record['geometry'] = link['geometry'].wkt if 'geometry' in link and link['geometry'] else None

                additional_records.append(record)
        elif dc_links and not include_dc:
            print(f"Skipping {len(dc_links)} DC links as include_dc is False")

        # Process 110kV links
        if links_110kv and include_110kv:
            print(f"Adding {len(links_110kv)} 110kV links to export")
            for link in links_110kv:
                # Create record with all standard fields as None
                record = {col: None for col in pypsa_df.columns}

                # Set ID (line_id in 110kV file maps to id in output)
                record['id'] = link.get('line_id', '')

                # Also preserve the original line_id in line_id column
                record['line_id'] = link.get('line_id', '')

                # Set voltage
                voltage = link.get('voltage')
                record['voltage'] = voltage if voltage else 110

                # Set bus connections
                record['bus0'] = link.get('bus0')
                record['bus1'] = link.get('bus1')

                # Set electrical parameters
                record['i_nom'] = link.get('i_nom')
                record['s_nom'] = link.get('s_nom')
                record['r'] = link.get('r')
                record['x'] = link.get('x')
                record['b'] = link.get('b')
                record['length'] = link.get('length')
                record['circuits'] = link.get('circuits')
                record['type'] = link.get('type')

                # Set underground/under_construction flags
                record['underground'] = link.get('underground', False)
                record['under_construction'] = link.get('under_construction', False)

                # Set source marker
                record['source'] = '110kV'

                # Set geometry
                record['geometry'] = link['geometry'].wkt if 'geometry' in link and link['geometry'] else None

                additional_records.append(record)
        elif links_110kv and not include_110kv:
            print(f"Skipping {len(links_110kv)} 110kV links as include_110kv is False")

        # Create DataFrame from additional records
        if additional_records:
            additional_df = pd.DataFrame(additional_records)

            # Add line_id column to original DataFrame if it doesn't exist
            if 'line_id' not in pypsa_df.columns:
                pypsa_df['line_id'] = None

            # Ensure all columns match between the two DataFrames
            for col in additional_df.columns:
                if col not in pypsa_df.columns:
                    pypsa_df[col] = None

            for col in pypsa_df.columns:
                if col not in additional_df.columns:
                    additional_df[col] = None

            # Merge with original and save
            merged_df = pd.concat([pypsa_df, additional_df], ignore_index=True)

            new_path = output_dir / "pypsa_with_eic_enhanced.csv"
            merged_df.to_csv(new_path, index=False)

            print(f"Added {len(additional_records)} rows to output CSV file")
            print(f"Original: {original_count}, new: {len(merged_df)}")
            print(f"Enhanced CSV saved to: {new_path}")

            return merged_df  # Return the merged DataFrame for visualization
        else:
            print("No additional data to append to CSV")
            # If no records added but we still want to save a copy of the original file
            if dc_links or links_110kv:
                new_path = output_dir / "pypsa_with_eic_enhanced.csv"
                pypsa_df.to_csv(new_path, index=False)
                print(f"Copied original data to: {new_path}")

            return pypsa_df  # Return the original DataFrame for visualization

    except Exception as e:
        print(f"Error appending to CSV export: {e}")
        import traceback
        traceback.print_exc()
        return None


def regenerate_outputs(results, jao_gdf, pypsa_gdf, output_dir):
    """
    Regenerate all output files to ensure manual matches are included.
    """
    print("\n===== REGENERATING OUTPUTS WITH MANUAL MATCHES =====")

    # 1. Export results to CSV
    csv_file = output_dir / 'jao_pypsa_matches.csv'
    create_results_csv(results, csv_file)
    print(f"Updated match results saved to {csv_file}")

    # 2. Generate PyPSA with EIC codes
    pypsa_match_count, pypsa_with_eic, pypsa_eic_files = generate_pypsa_with_eic(
        results, jao_gdf, pypsa_gdf, output_dir
    )
    print(f"Generated PyPSA with EIC codes - {pypsa_match_count} lines matched")

    # 3. Generate JAO with PyPSA electrical parameters
    jao_with_pypsa = generate_jao_with_pypsa(
        results, jao_gdf, pypsa_gdf, output_dir
    )
    print("Generated JAO lines with PyPSA electrical parameters")

    # 4. Create visualization map
    map_file = output_dir / 'jao_pypsa_matches.html'
    create_jao_pypsa_visualization(jao_gdf, pypsa_gdf, results, map_file)
    print(f"Updated visualization map saved to {map_file}")

    return pypsa_with_eic


def main():
    # Make configuration variables global so we can modify them
    global INCLUDE_DC_IN_MATCHING, INCLUDE_110KV_IN_MATCHING
    global INCLUDE_DC_IN_OUTPUT, INCLUDE_110KV_IN_OUTPUT
    global GENERATE_PARAMETER_VISUALIZATION, VERBOSE
    global ENABLE_MANUAL_MATCHING, ADD_PREDEFINED_MATCHES, IMPORT_NEW_LINES

    # Parse command-line arguments
    args = parse_arguments()

    # Set configuration from command-line arguments
    OUTPUT_DIR = Path(args.output)

    # Override matching inclusion if specified in arguments
    INCLUDE_DC_IN_MATCHING = args.include_dc_matching
    INCLUDE_110KV_IN_MATCHING = args.include_110kv_matching

    # Override output inclusion if specified in arguments
    INCLUDE_DC_IN_OUTPUT = not args.no_dc_output
    INCLUDE_110KV_IN_OUTPUT = not args.no_110kv_output

    # Override visualization option if specified
    GENERATE_PARAMETER_VISUALIZATION = not args.no_viz

    # Override verbosity option if specified
    VERBOSE = not args.quiet

    # Only override defaults if flags were explicitly set
    if args.manual is not None:
        ENABLE_MANUAL_MATCHING = args.manual
    if args.no_manual:
        ENABLE_MANUAL_MATCHING = False
    if args.add_predefined is not None:
        ADD_PREDEFINED_MATCHES = args.add_predefined
    if args.no_predefined:
        ADD_PREDEFINED_MATCHES = False

    # This flag doesn't have the None default treatment
    IMPORT_NEW_LINES = args.import_new_lines

    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("===== GRID MATCHER matcher MODE =====")
    print(f"Include DC links in matching: {INCLUDE_DC_IN_MATCHING}")
    print(f"Include 110kV lines in matching: {INCLUDE_110KV_IN_MATCHING}")
    print(f"Include DC links in output: {INCLUDE_DC_IN_OUTPUT}")
    print(f"Include 110kV lines in output: {INCLUDE_110KV_IN_OUTPUT}")
    print(f"Generate parameter visualization: {GENERATE_PARAMETER_VISUALIZATION}")
    print(f"Enable manual matching: {ENABLE_MANUAL_MATCHING}")
    print(f"Add predefined matches: {ADD_PREDEFINED_MATCHES}")
    print(f"Import new lines: {IMPORT_NEW_LINES}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Verbose output: {VERBOSE}")

    # Load main data
    print("\n===== LOADING MAIN DATA =====")
    jao_gdf, pypsa_gdf = load_data(JAO_PATH, PYPSA_PATH)

    # Import new lines if enabled
    if IMPORT_NEW_LINES:
        jao_gdf, pypsa_gdf = import_new_lines_from_csv(jao_gdf, pypsa_gdf, DATA_DIR)


    # Apply predefined manual matches if enabled
    manual_matches = []
    if ENABLE_MANUAL_MATCHING:
        # Load existing matches first
        if os.path.exists(MANUAL_MATCHES_FILE):
            manual_matches = load_manual_matches_file(MANUAL_MATCHES_FILE)
            print(f"Loaded {len(manual_matches)} manual matches from {MANUAL_MATCHES_FILE}")

        # Add predefined matches if enabled
        if ADD_PREDEFINED_MATCHES:
            manual_matches = add_predefined_manual_matches(jao_gdf, pypsa_gdf, manual_matches, interactive=False)
            save_manual_matches_file(manual_matches, MANUAL_MATCHES_FILE)




    # Load manual matches without interactive mode
    elif ENABLE_MANUAL_MATCHING:
        if os.path.exists(MANUAL_MATCHES_FILE):
            manual_matches = load_manual_matches_file(MANUAL_MATCHES_FILE)
            print(f"Loaded {len(manual_matches)} manual matches from {MANUAL_MATCHES_FILE}")

    # Run matching with configurable DC and 110kV inclusion
    results = run_original_matching(
        jao_gdf,
        pypsa_gdf,
        OUTPUT_DIR,
        include_110kv=INCLUDE_110KV_IN_MATCHING,
        include_dc_links=INCLUDE_DC_IN_MATCHING,
        pypsa_110kv_path=PYPSA_110KV_PATH if INCLUDE_110KV_IN_MATCHING else None,
        pypsa_dc_path=PYPSA_DC_PATH if INCLUDE_DC_IN_MATCHING else None,
        verbose=VERBOSE
    )

    print("\n===== MATCHING COMPLETE =====")

    # Apply manual matches if enabled and we have matches
    regenerate_needed = False
    if ENABLE_MANUAL_MATCHING and manual_matches:
        results = apply_manual_matches(results, jao_gdf, pypsa_gdf, manual_matches)
        regenerate_needed = True

    # If manual matches were applied, regenerate the output files
    pypsa_with_eic = None
    if regenerate_needed:
        pypsa_with_eic = regenerate_outputs(results, jao_gdf, pypsa_gdf, OUTPUT_DIR)

    # Extract DC and 110kV links using direct methods (if not included in matching)
    print("\n===== PROCESSING ADDITIONAL DATA =====")

    dc_links = []
    if os.path.exists(PYPSA_DC_PATH) and not INCLUDE_DC_IN_MATCHING:
        dc_links = parse_dc_links_direct(PYPSA_DC_PATH)
        print(f"Extracted {len(dc_links)} DC links")
    else:
        print("DC links already included in matching or file not found")

    links_110kv = []
    if os.path.exists(PYPSA_110KV_PATH) and not INCLUDE_110KV_IN_MATCHING:
        links_110kv = parse_110kv_links_direct(PYPSA_110KV_PATH)
        print(f"Extracted {len(links_110kv)} 110kV links")
    else:
        print("110kV links already included in matching or file not found")

    # Add to output CSV
    print("\n===== ADDING ADDITIONAL DATA TO OUTPUT =====")
    enhanced_pypsa_gdf = append_to_csv_export(
        OUTPUT_DIR,
        dc_links,
        links_110kv,
        include_dc=INCLUDE_DC_IN_OUTPUT,
        include_110kv=INCLUDE_110KV_IN_OUTPUT
    )

    # Generate parameter comparison visualization if requested
    if GENERATE_PARAMETER_VISUALIZATION and enhanced_pypsa_gdf is not None:
        print("\n===== GENERATING PARAMETER COMPARISON VISUALIZATION =====")
        try:
            # Use the results directly from run_original_matching function
            if results:
                print(f"Total matching results: {len(results)}")
                matched_count = sum(1 for r in results if r.get("matched", False))
                print(f"Matched results: {matched_count}")

                # Convert enhanced_pypsa_gdf to GeoDataFrame if needed
                if isinstance(enhanced_pypsa_gdf, pd.DataFrame) and not isinstance(enhanced_pypsa_gdf,
                                                                                   gpd.GeoDataFrame):
                    try:
                        if 'geometry' in enhanced_pypsa_gdf.columns:
                            from shapely import wkt
                            enhanced_pypsa_gdf['geometry'] = enhanced_pypsa_gdf['geometry'].apply(
                                lambda x: wkt.loads(x) if isinstance(x, str) else x
                            )
                            enhanced_pypsa_gdf = gpd.GeoDataFrame(
                                enhanced_pypsa_gdf,
                                geometry='geometry',
                                crs="EPSG:4326"
                            )
                    except Exception as e:
                        print(f"Error converting to GeoDataFrame: {e}")

                # Import the preparation function from comparison.py
                from grid_matcher.visualization.comparison import prepare_visualization_data

                # Transform the data into the expected format
                enhanced_results = prepare_visualization_data(
                    results,
                    enhanced_pypsa_gdf,
                    jao_gdf=jao_gdf
                )

                # Create parameter comparison visualization
                visualization_path = visualize_parameter_comparison(
                    enhanced_results,
                    enhanced_pypsa_gdf,
                    output_dir=OUTPUT_DIR
                )
                print(f"Parameter comparison visualization saved to: {visualization_path}")

                # ALSO create the parameter summary table
                from grid_matcher.visualization.reports import create_enhanced_summary_table
                summary_path = create_enhanced_summary_table(
                    jao_gdf,
                    enhanced_pypsa_gdf,
                    results,
                    output_dir=OUTPUT_DIR
                )
                print(f"Parameter summary table saved to: {summary_path}")

            else:
                print("No matching results available")
                print("Parameter comparison visualization could not be generated")
        except Exception as e:
            print(f"Error generating parameter comparison visualization: {e}")
            import traceback
            traceback.print_exc()

    print("\n===== PROCESS COMPLETE =====")


if __name__ == "__main__":
    main()