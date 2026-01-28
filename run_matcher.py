#!/usr/bin/env python
# run_matcher.py - Custom script for handling complex data formats

import os
import sys
import random
import re
import pandas as pd
import geopandas as gpd
import argparse
from pathlib import Path
from shapely.geometry import LineString
from shapely.geometry.point import Point

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
from grid_matcher.visualization.maps import create_jao_pypsa_visualization, save_dual_match_png
from grid_matcher.visualization.reports import create_enhanced_summary_table
from grid_matcher.visualization.length_comparison import compare_line_lengths
from grid_matcher.visualization.grid_comparisons import generate_grid_comparisons
from grid_matcher.matcher.match_transformers import run_transformer_matching_pipeline
from grid_matcher.matcher.transformer_utils import create_updated_pypsa_with_jao_params
from shapely import wkt
from grid_matcher.matcher.match_transformers import create_lines_transformers_matched_map, _safe_wkt_load




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
    parser.add_argument("--no-length-comparison", action="store_true", help="Skip line length comparison")
    parser.add_argument("--grid-comparison", action="store_true", help="Generate grid comparison visualizations")
    parser.add_argument("--no-grid-comparison", action="store_true", help="Skip grid comparison visualizations")

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

    # match transformers
    parser.add_argument("--include-transformers", action="store_true", help="Include transformer matching")
    parser.add_argument("--no-transformers", action="store_true", help="Skip transformer matching")
    parser.add_argument("--transformers-distance", type=float, default=5.0,
                        help="Distance threshold for transformer matching in km")

    return parser.parse_args()

# ===== CONFIGURATION =====
# Skip DC and 110kV in matching for speed
INCLUDE_DC_IN_MATCHING = False
INCLUDE_110KV_IN_MATCHING = False

# Control whether to include DC and 110kV in the enhanced output
INCLUDE_DC_IN_OUTPUT = False  # Set to False to exclude DC links from final output
INCLUDE_110KV_IN_OUTPUT = False  # Set to False to exclude 110kV lines from final output

# Control whether to generate parameter comparison visualization
GENERATE_PARAMETER_VISUALIZATION = True
GENERATE_LENGTH_COMPARISON = True
GENERATE_GRID_COMPARISON = True

# Control manual matching options
ENABLE_MANUAL_MATCHING = True
ADD_PREDEFINED_MATCHES = True
IMPORT_NEW_LINES = False

# Include transformers:
INCLUDE_TRANSFORMER_MATCHING = True

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


def update_pypsa_with_jao_data(pypsa_path, jao_path, output_dir):
    """
    Update PyPSA electrical parameters with values from JAO data.

    Parameters:
    -----------
    pypsa_path : Path or str
        Path to the PyPSA CSV file with EIC codes
    jao_path : Path or str
        Path to the JAO CSV file with PyPSA matches
    output_dir : Path or str
        Directory to save updated files

    Returns:
    --------
    Path
        Path to the updated PyPSA file
    """
    print("\n===== UPDATING PYPSA WITH JAO ELECTRICAL PARAMETERS =====")

    # Ensure output directory exists
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Print available files in output directory
    csv_files = list(output_dir.glob("*.csv"))
    print(f"Found {len(csv_files)} CSV files in {output_dir}")
    for file in csv_files:
        if "pypsa" in file.name.lower() and "eic" in file.name.lower():
            print(f"Found PyPSA file: {file.name}")
        if "jao" in file.name.lower() and "matches" in file.name.lower():
            print(f"Found JAO file: {file.name}")

    # Load data
    pypsa_df = pd.read_csv(pypsa_path)
    jao_df = pd.read_csv(jao_path)

    print(f"PyPSA data: {len(pypsa_df)} rows")
    print(f"JAO data: {len(jao_df)} rows")

    # Print column names for debugging
    print(f"PyPSA columns: {list(pypsa_df.columns)}")
    print(f"JAO columns: {list(jao_df.columns)}")

    # Track changed parameters for reporting
    changed_params = []

    # Update PyPSA parameters from JAO data
    for _, jao_row in jao_df.iterrows():
        if not pd.isna(jao_row.get('pypsa_ids')) and jao_row.get('matched', False):
            # Split in case there are multiple IDs separated by comma or similar
            pypsa_ids = [id.strip() for id in str(jao_row['pypsa_ids']).split(',')]

            for pypsa_id in pypsa_ids:
                # First try with line_id
                matching_rows = pypsa_df[pypsa_df['line_id'] == pypsa_id]

                # If no match, try with id column
                if matching_rows.empty:
                    matching_rows = pypsa_df[pypsa_df['id'] == pypsa_id]

                if not matching_rows.empty:
                    # Process each matching PyPSA row with JAO data
                    for idx, pypsa_row in matching_rows.iterrows():
                        changes = {}

                        # Map of PyPSA parameters to JAO parameters - ONLY include r, x, and b
                        param_map = {
                            'r': 'jao_r',
                            'x': 'jao_x',
                            'b': 'jao_b'
                        }

                        # Try to update each parameter
                        for pypsa_param, jao_param in param_map.items():
                            # Skip if JAO parameter doesn't exist or is NaN
                            if jao_param not in jao_row or pd.isna(jao_row[jao_param]):
                                continue

                            # Skip if PyPSA parameter doesn't exist
                            if pypsa_param not in pypsa_df.columns:
                                continue

                            # Get current and new values
                            original_val = pypsa_df.at[idx, pypsa_param]
                            updated_val = jao_row[jao_param]

                            # Check if we need to update (with tolerance for float comparison)
                            if pd.isna(original_val) or pd.isna(updated_val):
                                # Handle NaN values
                                if pd.isna(original_val) and not pd.isna(updated_val):
                                    # Update NaN to a value
                                    pypsa_df.at[idx, pypsa_param] = updated_val
                                    changes[pypsa_param] = {
                                        'original': 'NaN',
                                        'updated': updated_val
                                    }
                            elif isinstance(original_val, (int, float)) and isinstance(updated_val, (int, float)):
                                # Numeric comparison with tolerance
                                if abs(float(original_val) - float(updated_val)) > 1e-6:
                                    pypsa_df.at[idx, pypsa_param] = updated_val
                                    changes[pypsa_param] = {
                                        'original': original_val,
                                        'updated': updated_val
                                    }
                            else:
                                # String or other type comparison
                                if str(original_val) != str(updated_val):
                                    pypsa_df.at[idx, pypsa_param] = updated_val
                                    changes[pypsa_param] = {
                                        'original': original_val,
                                        'updated': updated_val
                                    }

                        # Record changes if any were made
                        if changes:
                            changed_params.append({
                                'id': pypsa_id,
                                'changes': changes
                            })

    # Save updated PyPSA data
    output_file = output_dir / "pypsa_with_jao_params.csv"
    pypsa_df.to_csv(output_file, index=False)
    print(f"Updated PyPSA data saved to: {output_file}")

    # Generate comparison report
    if changed_params:
        comparison_str = f"Updated {len(changed_params)} PyPSA lines with JAO parameters\n\n"

        # Print sample of changes
        sample_size = min(10, len(changed_params))
        import random
        samples = random.sample(changed_params, sample_size)

        comparison_str += f"Comparing {sample_size} randomly selected lines with updated parameters:\n"
        comparison_str += "=" * 80 + "\n"

        for change in samples:
            line_id = change['id']
            comparison_str += f"\nLine ID: {line_id}\n"
            comparison_str += f"{'Parameter':<12}| {'Original':<22} | {'Updated':<22} | {'Changed'}\n"
            comparison_str += f"{'-' * 12}|{'-' * 24}|{'-' * 24}|{'-' * 8}\n"

            for param, values in change['changes'].items():
                original = values['original']
                updated = values['updated']
                comparison_str += f"{param:<12}| {original:<22} | {updated:<22} | Yes\n"

        # Print to console
        print(comparison_str)

        # Save comparison to file
        comparison_file = output_dir / "parameter_comparison.txt"
        with open(comparison_file, 'w') as f:
            f.write(comparison_str)
        print(f"Parameter comparison saved to: {comparison_file}")

        # Also save the list of all changed lines
        changed_lines_file = output_dir / "changed_lines.csv"
        changed_ids = [item['id'] for item in changed_params]
        pd.DataFrame({'changed_line_id': changed_ids}).to_csv(changed_lines_file, index=False)
        print(f"List of all changed lines saved to: {changed_lines_file}")
    else:
        print("No parameters were updated.")

    return output_file


def append_to_csv_export(output_dir, dc_links=None, links_110kv=None, include_dc=True, include_110kv=True):
    """Append DC and 110kV data to the output CSV file."""
    # Try to use the enhanced file first, fall back to the original
    pypsa_enhanced_path = output_dir / "pypsa_with_eic_enhanced.csv"
    pypsa_original_path = output_dir / "pypsa_with_eic.csv"

    pypsa_csv_path = pypsa_enhanced_path if os.path.exists(pypsa_enhanced_path) else pypsa_original_path

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

            new_path = output_dir / "pypsa_with_eic_enhanced_complete.csv"
            merged_df.to_csv(new_path, index=False)

            print(f"Added {len(additional_records)} rows to output CSV file")
            print(f"Original: {original_count}, new: {len(merged_df)}")
            print(f"Enhanced CSV saved to: {new_path}")

            return merged_df  # Return the merged DataFrame for visualization
        else:
            print("No additional data to append to CSV")
            # If no records added but we still want to save a copy of the original file
            if dc_links or links_110kv:
                new_path = output_dir / "pypsa_with_eic_enhanced_complete.csv"
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

def _load_transformers_csv_as_gdf(path: Path) -> gpd.GeoDataFrame:
    """
    Read a transformer CSV with a WKT 'geometry' column (or lon/lat fallback)
    and return a GeoDataFrame in EPSG:4326.
    """
    df = pd.read_csv(path)

    # WKT geometry (preferred)
    if "geometry" in df.columns:
        df["geometry"] = df["geometry"].apply(
            lambda s: wkt.loads(s) if isinstance(s, str) and s.strip() else None
        )
    # fallback: try lon/lat style columns
    elif {"lon", "lat"}.issubset(df.columns):
        df["geometry"] = df.apply(
            lambda r: Point(float(r["lon"]), float(r["lat"])) if pd.notna(r["lon"]) and pd.notna(r["lat"]) else None,
            axis=1
        )
    else:
        # last resort: try x/y
        if {"x", "y"}.issubset(df.columns):
            df["geometry"] = df.apply(
                lambda r: Point(float(r["x"]), float(r["y"])) if pd.notna(r["x"]) and pd.notna(r["y"]) else None,
                axis=1
            )
        else:
            df["geometry"] = None

    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")
    return gdf



def main():
    # ===== Global flags we may override via CLI =====
    global INCLUDE_DC_IN_MATCHING, INCLUDE_110KV_IN_MATCHING
    global INCLUDE_DC_IN_OUTPUT, INCLUDE_110KV_IN_OUTPUT
    global GENERATE_PARAMETER_VISUALIZATION, GENERATE_LENGTH_COMPARISON, GENERATE_GRID_COMPARISON, VERBOSE
    global ENABLE_MANUAL_MATCHING, ADD_PREDEFINED_MATCHES, IMPORT_NEW_LINES
    global INCLUDE_TRANSFORMER_MATCHING

    # ----- Parse args & apply configuration -----
    args = parse_arguments()
    OUTPUT_DIR = Path(args.output)

    INCLUDE_DC_IN_MATCHING   = bool(args.include_dc_matching)
    INCLUDE_110KV_IN_MATCHING = bool(args.include_110kv_matching)

    INCLUDE_DC_IN_OUTPUT     = not args.no_dc_output
    INCLUDE_110KV_IN_OUTPUT  = not args.no_110kv_output

    GENERATE_PARAMETER_VISUALIZATION = not args.no_viz
    GENERATE_LENGTH_COMPARISON       = not args.no_length_comparison
    GENERATE_GRID_COMPARISON         = True if args.grid_comparison else (False if args.no_grid_comparison else GENERATE_GRID_COMPARISON)

    VERBOSE = not args.quiet

    if args.manual is not None:
        ENABLE_MANUAL_MATCHING = bool(args.manual)
    if args.no_manual:
        ENABLE_MANUAL_MATCHING = False
    if args.add_predefined is not None:
        ADD_PREDEFINED_MATCHES = bool(args.add_predefined)
    if args.no_predefined:
        ADD_PREDEFINED_MATCHES = False

    INCLUDE_TRANSFORMER_MATCHING = True if args.include_transformers else (False if args.no_transformers else INCLUDE_TRANSFORMER_MATCHING)

    IMPORT_NEW_LINES = bool(args.import_new_lines)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("===== GRID MATCHER matcher MODE =====")
    print(f"Include DC links in matching:      {INCLUDE_DC_IN_MATCHING}")
    print(f"Include 110kV lines in matching:   {INCLUDE_110KV_IN_MATCHING}")
    print(f"Include DC links in output:        {INCLUDE_DC_IN_OUTPUT}")
    print(f"Include 110kV lines in output:     {INCLUDE_110KV_IN_OUTPUT}")
    print(f"Generate parameter visualization:  {GENERATE_PARAMETER_VISUALIZATION}")
    print(f"Generate length comparison:        {GENERATE_LENGTH_COMPARISON}")
    print(f"Generate grid comparisons:         {GENERATE_GRID_COMPARISON}")
    print(f"Enable manual matching:            {ENABLE_MANUAL_MATCHING}")
    print(f"Add predefined matches:            {ADD_PREDEFINED_MATCHES}")
    print(f"Import new lines:                  {IMPORT_NEW_LINES}")
    print(f"Include transformer matching:      {INCLUDE_TRANSFORMER_MATCHING}")
    print(f"Output directory:                  {OUTPUT_DIR}")
    print(f"Verbose output:                    {VERBOSE}")

    # ----- Load main line data -----
    print("\n===== LOADING MAIN DATA =====")
    jao_gdf, pypsa_gdf = load_data(JAO_PATH, PYPSA_PATH)

    if IMPORT_NEW_LINES:
        try:
            jao_gdf, pypsa_gdf = import_new_lines_from_csv(jao_gdf, pypsa_gdf, DATA_DIR)
            print("Imported new lines from CSV.")
        except Exception as e:
            print(f"Warning: failed to import new lines: {e}")

    # ----- Manual matches: load and (optionally) add predefined -----
    manual_matches = []
    if ENABLE_MANUAL_MATCHING:
        try:
            if MANUAL_MATCHES_FILE.exists():
                manual_matches = load_manual_matches_file(MANUAL_MATCHES_FILE)
                print(f"Loaded {len(manual_matches)} manual matches from {MANUAL_MATCHES_FILE}")
            if ADD_PREDEFINED_MATCHES:
                manual_matches = add_predefined_manual_matches(jao_gdf, pypsa_gdf, manual_matches, interactive=False)
                save_manual_matches_file(manual_matches, MANUAL_MATCHES_FILE)
                print(f"Saved {len(manual_matches)} manual matches (after adding predefined).")
        except Exception as e:
            print(f"Warning: manual matching prep failed: {e}")

    # ----- Run automated matching (lines) -----
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

    # ----- Apply manual matches and regenerate outputs if needed -----
    if ENABLE_MANUAL_MATCHING and manual_matches:
        try:
            results = apply_manual_matches(results, jao_gdf, pypsa_gdf, manual_matches)
            _ = regenerate_outputs(results, jao_gdf, pypsa_gdf, OUTPUT_DIR)
        except Exception as e:
            print(f"Warning: applying/regenerating manual matches failed: {e}")

    # ----- DC / 110kV parsing (if not included in matching) -----
    print("\n===== PROCESSING ADDITIONAL DATA =====")
    dc_links = []
    links_110kv = []
    try:
        if (not INCLUDE_DC_IN_MATCHING) and PYPSA_DC_PATH.exists():
            dc_links = parse_dc_links_direct(PYPSA_DC_PATH)
            print(f"Extracted {len(dc_links)} DC links")
        else:
            print("DC links already included in matching or file not found")
    except Exception as e:
        print(f"Warning: DC parsing failed: {e}")

    try:
        if (not INCLUDE_110KV_IN_MATCHING) and PYPSA_110KV_PATH.exists():
            links_110kv = parse_110kv_links_direct(PYPSA_110KV_PATH)
            print(f"Extracted {len(links_110kv)} 110kV links")
        else:
            print("110kV links already included in matching or file not found")
    except Exception as e:
        print(f"Warning: 110kV parsing failed: {e}")

    # ----- Update PyPSA electrical params from JAO matches -----
    pypsa_eic_file    = OUTPUT_DIR / "pypsa_with_eic.csv"
    jao_matches_file  = OUTPUT_DIR / "jao_pypsa_matches.csv"
    enhanced_path     = OUTPUT_DIR / "pypsa_with_eic_enhanced.csv"

    try:
        updated_pypsa_file = update_pypsa_with_jao_data(pypsa_eic_file, jao_matches_file, OUTPUT_DIR)
        # keep downstream code expecting this filename happy
        import shutil
        shutil.copyfile(updated_pypsa_file, enhanced_path)
        print(f"Copied updated parameters to: {enhanced_path}")
    except Exception as e:
        print(f"Warning: updating PyPSA with JAO params failed: {e}")

    # ----- Append DC/110kV to export, get GeoDataFrame for visualizations -----
    enhanced_pypsa_gdf = append_to_csv_export(
        OUTPUT_DIR,
        dc_links=dc_links,
        links_110kv=links_110kv,
        include_dc=INCLUDE_DC_IN_OUTPUT,
        include_110kv=INCLUDE_110KV_IN_OUTPUT
    )

    png = save_dual_match_png(
        jao_gdf,
        enhanced_pypsa_gdf,  # must be a GeoDataFrame
        results,  # not "matching_results"
        out_path=OUTPUT_DIR / "jao_pypsa_map.png",
    )

    # ----- Visualizations (parameter comparison / length / grid) -----
    if GENERATE_PARAMETER_VISUALIZATION and enhanced_pypsa_gdf is not None:
        print("\n===== GENERATING PARAMETER COMPARISON VISUALIZATION =====")
        try:
            if results:
                # ensure GeoDataFrame
                if isinstance(enhanced_pypsa_gdf, pd.DataFrame) and not isinstance(enhanced_pypsa_gdf, gpd.GeoDataFrame):
                    if 'geometry' in enhanced_pypsa_gdf.columns:
                        from shapely import wkt as _wkt
                        enhanced_pypsa_gdf['geometry'] = enhanced_pypsa_gdf['geometry'].apply(
                            lambda x: _wkt.loads(x) if isinstance(x, str) else x
                        )
                    enhanced_pypsa_gdf = gpd.GeoDataFrame(enhanced_pypsa_gdf, geometry='geometry', crs="EPSG:4326")

                from grid_matcher.visualization.comparison import prepare_visualization_data
                enhanced_results = prepare_visualization_data(results, enhanced_pypsa_gdf, jao_gdf=jao_gdf)
                viz_path = visualize_parameter_comparison(enhanced_results, enhanced_pypsa_gdf, output_dir=OUTPUT_DIR)
                print(f"Parameter comparison visualization saved to: {viz_path}")

                from grid_matcher.visualization.reports import create_enhanced_summary_table
                summary_path = create_enhanced_summary_table(jao_gdf, enhanced_pypsa_gdf, results, output_dir=OUTPUT_DIR)
                print(f"Parameter summary table saved to: {summary_path}")
            else:
                print("No matching results available for visualization.")
        except Exception as e:
            print(f"Warning: parameter visualization failed: {e}")

    if GENERATE_LENGTH_COMPARISON and enhanced_pypsa_gdf is not None and jao_gdf is not None:
        print("\n===== GENERATING LINE LENGTH COMPARISON =====")
        try:
            length_comparison = compare_line_lengths(
                jao_gdf, enhanced_pypsa_gdf, matching_results=results, output_dir=OUTPUT_DIR
            )
            print(f"Line length comparison visualization saved to: {length_comparison['html_report']}")
        except Exception as e:
            print(f"Warning: line length comparison failed: {e}")

    if GENERATE_GRID_COMPARISON and enhanced_pypsa_gdf is not None and jao_gdf is not None:
        print("\n===== GENERATING GRID COMPARISON VISUALIZATIONS =====")
        try:
            grid_comparisons = generate_grid_comparisons(jao_gdf, enhanced_pypsa_gdf, output_dir=OUTPUT_DIR)
            print(f"Grid comparison visualizations saved to: {grid_comparisons['html']}")
        except Exception as e:
            print(f"Warning: grid comparisons failed: {e}")

    # ----- Transformer matching + update + map -----
    transformer_results = None
    if INCLUDE_TRANSFORMER_MATCHING:
        print("\n===== RUNNING TRANSFORMER MATCHING =====")
        JAO_TRANSFORMERS_PATH   = DATA_DIR / "jao_transformers.csv"
        PYPSA_TRANSFORMERS_PATH = DATA_DIR / "pypsa_transformers.csv"

        if JAO_TRANSFORMERS_PATH.exists() and PYPSA_TRANSFORMERS_PATH.exists():
            try:
                transformer_results = run_transformer_matching_pipeline(
                    JAO_TRANSFORMERS_PATH, PYPSA_TRANSFORMERS_PATH, OUTPUT_DIR, args.transformers_distance
                )
                print("Transformer matching completed.")
                print(f"Matched {transformer_results['statistics']['matched_count']} "
                      f"out of {transformer_results['statistics']['total_jao']} transformers.")
            except Exception as e:
                print(f"Warning: transformer matching failed: {e}")

            # Prefer cleaned transformer CSVs emitted by the pipeline (fallback to originals)
            cleaned_jao_tx   = OUTPUT_DIR / "jao_transformers_clean.csv"
            cleaned_pypsa_tx = OUTPUT_DIR / "pypsa_transformers_clean.csv"
            jao_tx_path_for_update   = cleaned_jao_tx   if cleaned_jao_tx.exists()   else JAO_TRANSFORMERS_PATH
            pypsa_tx_path_for_update = cleaned_pypsa_tx if cleaned_pypsa_tx.exists() else PYPSA_TRANSFORMERS_PATH

            # Update PyPSA transformer CSV with JAO electrical params
            try:
                updated_pypsa_tx = OUTPUT_DIR / "pypsa_transformers_updated.csv"
                create_updated_pypsa_with_jao_params(
                    pypsa_csv=str(pypsa_tx_path_for_update),
                    jao_csv=str(jao_tx_path_for_update),
                    matches=transformer_results['results_file'] if transformer_results else None,
                    out_csv=str(updated_pypsa_tx),
                    overwrite_s_nom=False,
                    add_eic=True
                )
                print(f"Updated PyPSA transformers saved to: {updated_pypsa_tx}")
            except Exception as e:
                print(f"Warning: updating PyPSA transformers failed: {e}")
        else:
            print("Transformer data files not found. Skipping transformer matching.")

    # Voltage-filtered lines + transformers map (only if we ran transformer matching)
    pypsa_lines_csv = DATA_DIR / "pypsa_clipped_MVHV_lines.csv"
    jao_lines_csv   = DATA_DIR / "jao_clipped_lines.csv"

    if transformer_results and pypsa_lines_csv.exists() and jao_lines_csv.exists():
        try:
            # Small robust CSV->GDF loader (uses _safe_wkt_load and glues split geometry columns)
            def _load_tx(path: Path, drop_dummies: bool = False) -> gpd.GeoDataFrame:
                # Always read as normal comma-separated CSV; cleaned files have quoted WKT
                df = pd.read_csv(path)  # <- no sep=None, no quotechar override

                if drop_dummies and "name" in df.columns:
                    s = df["name"].astype(str)
                    df = df[~s.str.startswith("T_", na=False)]
                    for c in ("Comment", "comment"):
                        if c in df.columns:
                            df = df[~df[c].astype(str).str.contains("Added for coherence", case=False, na=False)]

                # Parse WKT if present
                if "geometry" in df.columns:
                    df["geometry"] = df["geometry"].apply(_safe_wkt_load)
                else:
                    df["geometry"] = None

                gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")
                mask = gdf["geometry"].notna() & ~gdf["geometry"].apply(lambda g: getattr(g, "is_empty", True))
                dropped = int((~mask).sum())
                if dropped:
                    print(f"{path.name}: dropped {dropped} rows with invalid/missing geometry")
                return gdf.loc[mask].copy()

            # Prefer cleaned CSVs for mapping
            jao_tx_csv_for_map = OUTPUT_DIR / "jao_transformers_clean.csv"
            pypsa_tx_csv_for_map = OUTPUT_DIR / "pypsa_transformers_clean.csv"
            if not jao_tx_csv_for_map.exists():   jao_tx_csv_for_map = DATA_DIR / "jao_transformers.csv"
            if not pypsa_tx_csv_for_map.exists(): pypsa_tx_csv_for_map = DATA_DIR / "pypsa_transformers.csv"

            jao_tx_gdf = _load_tx(jao_tx_csv_for_map, drop_dummies=True)
            pypsa_tx_gdf = _load_tx(pypsa_tx_csv_for_map)

            lv_lines_csv = DATA_DIR / "pypsa_clipped_LV_lines.csv"

            lines_tx_map_out = OUTPUT_DIR / "lines_plus_transformers_matched.html"
            create_lines_transformers_matched_map(
                pypsa_lines_csv=pypsa_lines_csv,
                jao_lines_csv=jao_lines_csv,
                jao_transformers_gdf=jao_tx_gdf,
                pypsa_transformers_gdf=pypsa_tx_gdf,
                matches=transformer_results['results_file'],
                out_html=lines_tx_map_out,
                allowed_kv=(220, 225, 380, 400),
                germany_boundary=None,
                simplify_tolerance=0.0,
                lv_lines_csv = lv_lines_csv,
            )
            print(f"Voltage-filtered lines + transformers (matched/unmatched) map saved to: {lines_tx_map_out}")
        except Exception as e:
            print(f"Warning: creating lines+transformers map failed: {e}")
    elif transformer_results is None:
        print("Skipping lines+transformers map: transformer matching did not run.")
    else:
        missing = []
        if not pypsa_lines_csv.exists(): missing.append("pypsa_clipped_MVHV_lines.csv")
        if not jao_lines_csv.exists():   missing.append("jao_clipped_lines.csv")
        print("Skipping lines+transformers map: missing", ", ".join(missing) if missing else "(unknown)")



if __name__ == "__main__":
    main()