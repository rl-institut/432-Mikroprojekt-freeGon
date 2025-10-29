#!/usr/bin/env python
# run_matcher.py - Grid matcher utility for JAO/OSM to PyPSA matching

import os
import sys
import re
import pandas as pd
import geopandas as gpd
import argparse
import traceback
from pathlib import Path
from shapely.geometry import LineString
from shapely import wkt

# Import matcher modules
from grid_matcher.matcher.original_matcher import run_original_matching, load_data, get_start_point, get_end_point
from grid_matcher.matcher.osm_pypsa_matcher import run_osm_pypsa_matching, load_osm_data

# Import utility functions
from grid_matcher.manual.manual_matching import (
    load_manual_matches_file, save_manual_matches_file,
    add_predefined_manual_matches, import_new_lines_from_csv
)

# Import exporters and visualization functions
from grid_matcher.io.exporters import (
    create_results_csv, generate_pypsa_with_eic, generate_jao_with_pypsa
)
from grid_matcher.io.loaders import load_110kv_data
from grid_matcher.visualization.reports import create_enhanced_summary_table
from grid_matcher.visualization.comparison import prepare_visualization_data, visualize_parameter_comparison
from grid_matcher.visualization.maps import create_jao_pypsa_visualization, create_osm_pypsa_match_visualization



def debug_matching_failure(osm_gdf, pypsa_gdf):
    """Diagnose why matching is failing"""
    print("\n===== DEBUGGING MATCH FAILURE =====")

    if pypsa_gdf is None:
        print("CRITICAL: PyPSA GeoDataFrame is None!")
        return

    # Check coordinate ranges to identify CRS issues
    osm_x_range = (osm_gdf.bounds.minx.min(), osm_gdf.bounds.maxx.max())
    osm_y_range = (osm_gdf.bounds.miny.min(), osm_gdf.bounds.maxy.max())

    pypsa_x_range = (pypsa_gdf.bounds.minx.min(), pypsa_gdf.bounds.maxx.max())
    pypsa_y_range = (pypsa_gdf.bounds.miny.min(), pypsa_gdf.bounds.maxy.max())

    print(f"OSM X range: {osm_x_range}")
    print(f"OSM Y range: {osm_y_range}")
    print(f"PyPSA X range: {pypsa_x_range}")
    print(f"PyPSA Y range: {pypsa_y_range}")

    # Check if the datasets even overlap
    from shapely.geometry import box
    osm_box = box(osm_x_range[0], osm_y_range[0], osm_x_range[1], osm_y_range[1])
    pypsa_box = box(pypsa_x_range[0], pypsa_y_range[0], pypsa_x_range[1], pypsa_y_range[1])

    if osm_box.intersects(pypsa_box):
        print("Datasets have overlapping extents")
        overlap = osm_box.intersection(pypsa_box)
        overlap_area = overlap.area
        osm_area = osm_box.area
        pypsa_area = pypsa_box.area
        print(f"Overlap percentage of OSM: {overlap_area / osm_area * 100:.1f}%")
        print(f"Overlap percentage of PyPSA: {overlap_area / pypsa_area * 100:.1f}%")
    else:
        print("CRITICAL: Datasets do not overlap at all")

    # Check for PyPSA ID columns before attempting buffer test
    pypsa_id_cols = [col for col in pypsa_gdf.columns if 'id' in col.lower()]
    print(f"\nPyPSA potential ID columns: {pypsa_id_cols}")

    # Pick a reasonable ID column for PyPSA
    pypsa_id_col = None
    preferred_cols = ['link_id', 'id', 'line_id']
    for col in preferred_cols:
        if col in pypsa_gdf.columns:
            pypsa_id_col = col
            print(f"Using '{pypsa_id_col}' as PyPSA ID column")
            break

    if not pypsa_id_col and pypsa_id_cols:
        pypsa_id_col = pypsa_id_cols[0]
        print(f"Using '{pypsa_id_col}' as PyPSA ID column")

    if not pypsa_id_col:
        print("CRITICAL: No ID column found in PyPSA data!")
        # Create a temporary index-based ID
        pypsa_gdf['temp_id'] = pypsa_gdf.index.astype(str)
        pypsa_id_col = 'temp_id'
        print("Created temporary 'temp_id' column based on index")

    # Try a very simple buffer-based match to see if ANY matches can be found
    print("\nAttempting basic buffer overlap test...")
    match_count = 0
    large_buffer = 10000  # 10km buffer

    # Take a small sample to speed up debugging
    osm_sample = osm_gdf.head(min(100, len(osm_gdf)))

    for _, osm_line in osm_sample.iterrows():
        osm_id = osm_line.get('id', str(osm_line.name))
        buffered = osm_line.geometry.buffer(large_buffer)
        intersects = pypsa_gdf[pypsa_gdf.geometry.intersects(buffered)]
        if len(intersects) > 0:
            match_count += 1
            if match_count <= 3:  # Show a few examples
                print(f"Sample match: OSM ID {osm_id} matches with PyPSA IDs: {intersects[pypsa_id_col].tolist()}")

    print(f"Found {match_count}/{len(osm_sample)} possible matches with a {large_buffer}m buffer")

    # Check CRS
    print(f"\nOSM CRS: {osm_gdf.crs}")
    print(f"PyPSA CRS: {pypsa_gdf.crs}")

    # Check columns
    print(f"\nOSM columns: {osm_gdf.columns.tolist()}")
    print(f"PyPSA columns: {pypsa_gdf.columns.tolist()}")

    # Check for required columns for matching
    osm_req_cols = ['geometry', 'id', 'voltage']
    pypsa_req_cols = ['geometry', pypsa_id_col, 'voltage']

    missing_osm = [col for col in osm_req_cols if col not in osm_gdf.columns]
    missing_pypsa = [col for col in pypsa_req_cols if col not in pypsa_gdf.columns]

    if missing_osm:
        print(f"\nCRITICAL: Missing required columns in OSM data: {missing_osm}")

    if missing_pypsa:
        print(f"CRITICAL: Missing required columns in PyPSA data: {missing_pypsa}")

    # Print sample rows for inspection
    print("\nSample OSM row:")
    try:
        print(osm_gdf.iloc[0].to_dict())
    except:
        print("Could not print sample OSM row")

    print("\nSample PyPSA row:")
    try:
        print(pypsa_gdf.iloc[0].to_dict())
    except:
        print("Could not print sample PyPSA row")

# Parse command-line arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description="Grid Matcher Mode")

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

    parser.add_argument('--ignore-voltage-differences', action='store_true',
                        help='Ignore voltage differences when matching OSM and PyPSA lines')

    # Manual matching options
    parser.add_argument("--manual", action="store_true", default=None, help="Enable manual matching")
    parser.add_argument("--no-manual", action="store_true", help="Disable manual matching")
    parser.add_argument("--add-predefined", action="store_true", default=None, help="Add predefined manual matches")
    parser.add_argument("--no-predefined", action="store_true", help="Don't add predefined matches")
    parser.add_argument("--import-new-lines", action="store_true", help="Import new lines from CSV files")

    # OSM matching option
    parser.add_argument("--osm-matching", action="store_true", help="Use OSM lines instead of JAO lines for matching")

    # Output directory
    parser.add_argument("--output", "-o", type=str, help="Output directory", default="output/matcher")

    return parser.parse_args()


# ===== CONFIGURATION =====
# Default settings (can be overridden by command-line args)
INCLUDE_DC_IN_MATCHING = False
INCLUDE_110KV_IN_MATCHING = False
INCLUDE_DC_IN_OUTPUT = True
INCLUDE_110KV_IN_OUTPUT = True
GENERATE_PARAMETER_VISUALIZATION = True
ENABLE_MANUAL_MATCHING = True
ADD_PREDEFINED_MATCHES = True
IMPORT_NEW_LINES = False
VERBOSE = True

# File paths
DATA_DIR = Path("grid_matcher/data")
MANUAL_MATCHES_FILE = DATA_DIR / "manual_matches.json"
JAO_PATH = DATA_DIR / "jao_lines.csv"
PYPSA_PATH = DATA_DIR / "pypsa_lines.csv"
PYPSA_110KV_PATH = DATA_DIR / "pypsa_lines_110kv_fixed.csv"
PYPSA_DC_PATH = DATA_DIR / "pypsa_dc_links.csv"
OSM_PATH = DATA_DIR / "osm_lines_fixed.csv"


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
                                if VERBOSE:
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
        return None

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
        traceback.print_exc()
        return None


def regenerate_outputs(results, source_gdf, pypsa_gdf, output_dir, is_osm_mode=False):
    """
    Regenerate all output files to ensure manual matches are included.
    """
    print("\n===== REGENERATING OUTPUTS WITH MANUAL MATCHES =====")

    source_type = "osm" if is_osm_mode else "jao"

    # 1. Export results to CSV
    csv_file = output_dir / f'{source_type}_pypsa_matches.csv'
    create_results_csv(results, csv_file)
    print(f"Updated match results saved to {csv_file}")

    # 2. Generate PyPSA with source codes/parameters
    if is_osm_mode:
        # For OSM mode, use OSM parameter export
        from grid_matcher.matcher.osm_pypsa_matcher import generate_pypsa_with_osm_parameters
        pypsa_match_count, pypsa_with_source, pypsa_files = generate_pypsa_with_osm_parameters(
            results, source_gdf, pypsa_gdf, output_dir
        )
    else:
        # For JAO mode, use EIC export
        pypsa_match_count, pypsa_with_source, pypsa_files = generate_pypsa_with_eic(
            results, source_gdf, pypsa_gdf, output_dir
        )

        # Generate JAO with PyPSA electrical parameters (only for JAO mode)
        source_with_pypsa = generate_jao_with_pypsa(
            results, source_gdf, pypsa_gdf, output_dir
        )
        print("Generated JAO lines with PyPSA electrical parameters")

    print(f"Generated PyPSA with {source_type.upper()} data - {pypsa_match_count} lines matched")

    # 3. Create visualization map
    map_file = output_dir / f'{source_type}_pypsa_matches.html'

    create_jao_pypsa_visualization(source_gdf, pypsa_gdf, results, map_file)
    print(f"Updated visualization map saved to {map_file}")

    return pypsa_with_source


def apply_manual_matches_safely(results, source_gdf, pypsa_gdf, manual_matches):
    """Safe wrapper for apply_manual_matches with proper imports"""
    from grid_matcher.manual.manual_matching import apply_manual_matches
    return apply_manual_matches(results, source_gdf, pypsa_gdf, manual_matches)


def preprocess_osm_data(osm_gdf):
    """Preprocess OSM data to make it compatible with PyPSA matching"""

    print("\n===== PREPROCESSING OSM DATA =====")
    original_count = len(osm_gdf)

    # 1. Fix duplicate IDs by creating unique IDs
    if 'result_id' in osm_gdf.columns and osm_gdf['result_id'].nunique() < len(osm_gdf) * 0.9:
        print(f"WARNING: ID duplication detected - {osm_gdf['result_id'].nunique()} unique IDs for {len(osm_gdf)} rows")

        # Create new unique IDs based on combined attributes
        osm_gdf['original_id'] = osm_gdf['result_id']  # Store original ID

        # Combine branch_id and view_id to create unique identifier
        if 'branch_id' in osm_gdf.columns and 'view_id' in osm_gdf.columns:
            print("Creating unique IDs from branch_id and view_id")
            osm_gdf['result_id'] = osm_gdf['branch_id'].astype(str) + '-' + osm_gdf['view_id'].astype(str)
        else:
            print("Creating sequential unique IDs")
            osm_gdf['result_id'] = [f"osm-{i + 1}" for i in range(len(osm_gdf))]

        # Update 'id' column (used by the matcher)
        osm_gdf['id'] = osm_gdf['result_id']

        print(f"Created {osm_gdf['result_id'].nunique()} unique IDs")

    # 2. Convert MultiLineString geometries to LineString
    from shapely.geometry import LineString, MultiLineString
    multiline_count = sum(osm_gdf.geometry.apply(lambda geom: isinstance(geom, MultiLineString)))
    if multiline_count > 0:
        print(f"Converting {multiline_count} MultiLineString geometries to LineString")

        def convert_multiline_to_line(geometry):
            if isinstance(geometry, MultiLineString):
                # If it's a simple MultiLineString with one part, convert directly
                if len(geometry.geoms) == 1:
                    return LineString(geometry.geoms[0])

                # If multiple parts, merge them if possible or use the longest segment
                try:
                    # Try to merge all parts (works if they're connected)
                    all_coords = []
                    for line in geometry.geoms:
                        all_coords.extend(list(line.coords))
                    return LineString(all_coords)
                except:
                    # If merging fails, use the longest segment
                    longest_line = max(geometry.geoms, key=lambda line: line.length)
                    return LineString(longest_line)
            return geometry

        # Apply the conversion
        osm_gdf['geometry'] = osm_gdf['geometry'].apply(convert_multiline_to_line)

        # Verify the conversion
        new_multiline_count = sum(osm_gdf.geometry.apply(lambda geom: isinstance(geom, MultiLineString)))
        print(f"Conversion complete: {multiline_count} -> {new_multiline_count} MultiLineStrings remaining")

    # 3. Normalize voltage representation (convert from volts to kV if needed)
    if 'branch_voltage' in osm_gdf.columns:
        sample_voltage = osm_gdf['branch_voltage'].iloc[0] if len(osm_gdf) > 0 else 0
        if sample_voltage and sample_voltage > 1000:  # Likely in volts
            print("Converting voltage from volts to kV")
            osm_gdf['voltage'] = osm_gdf['branch_voltage'] / 1000
            print(f"Voltage conversion example: {sample_voltage} V -> {sample_voltage / 1000} kV")

    # 4. Ensure 'id' column exists and is used as the primary identifier
    if 'id' not in osm_gdf.columns:
        print("Adding 'id' column based on result_id")
        osm_gdf['id'] = osm_gdf['result_id']

    # 5. Add name column if missing (for display purposes)
    if 'name' not in osm_gdf.columns:
        print("Adding empty 'name' column")
        osm_gdf['name'] = ""

    # 6. Ensure circuit information is properly formatted
    if 'cables' in osm_gdf.columns and 'circuits' not in osm_gdf.columns:
        print("Adding 'circuits' column based on 'cables'")
        osm_gdf['circuits'] = osm_gdf['cables']

    # 7. Ensure consistent column names with PyPSA dataset
    if 'br_r' in osm_gdf.columns and 'r' not in osm_gdf.columns:
        print("Adding 'r' column based on br_r")
        osm_gdf['r'] = osm_gdf['br_r']

    if 'br_x' in osm_gdf.columns and 'x' not in osm_gdf.columns:
        print("Adding 'x' column based on br_x")
        osm_gdf['x'] = osm_gdf['br_x']

    if 'br_b' in osm_gdf.columns and 'b' not in osm_gdf.columns:
        print("Adding 'b' column based on br_b")
        osm_gdf['b'] = osm_gdf['br_b']

    # 8. Check for valid geometries
    invalid_count = osm_gdf.geometry.isna().sum() + sum(~osm_gdf.geometry.is_valid)
    if invalid_count > 0:
        print(f"WARNING: Found {invalid_count} invalid geometries in OSM data")
        # Fix invalid geometries if needed
        # osm_gdf = osm_gdf[~osm_gdf.geometry.isna() & osm_gdf.geometry.is_valid].copy()

    print(f"Preprocessing complete: {original_count} -> {len(osm_gdf)} OSM lines")

    return osm_gdf

def create_diagnostic_visualization(osm_gdf, pypsa_gdf, output_file, sample_size=10):
    """
    Create a diagnostic map showing sample unmatched lines to visually inspect why they're not matching.
    """
    import folium
    from folium.features import GeoJsonPopup, GeoJsonTooltip
    import geopandas as gpd
    import random

    # Create a copy to avoid modifying originals
    osm = osm_gdf.copy()
    pypsa = pypsa_gdf.copy()

    # Ensure geometries are in WGS84
    if osm.crs and osm.crs != 'EPSG:4326':
        osm = osm.to_crs('EPSG:4326')
    if pypsa.crs and pypsa.crs != 'EPSG:4326':
        pypsa = pypsa.to_crs('EPSG:4326')

    # Sample random unmatched lines for inspection
    osm_sample = osm.sample(min(sample_size, len(osm)))
    pypsa_sample = pypsa.sample(min(sample_size, len(pypsa)))

    # Create a map centered on the data
    bounds = osm.total_bounds
    center_lat = (bounds[1] + bounds[3]) / 2
    center_lon = (bounds[0] + bounds[2]) / 2
    m = folium.Map(location=[center_lat, center_lon], zoom_start=6, tiles='CartoDB positron')

    # Add OSM lines in red
    folium.GeoJson(
        osm_sample,
        name="OSM Sample",
        style_function=lambda x: {'color': 'red', 'weight': 2, 'opacity': 0.8},
        tooltip=GeoJsonTooltip(fields=['id', 'voltage', 'length_km'], aliases=['OSM ID', 'Voltage (kV)', 'Length (km)'])
    ).add_to(m)

    # Add PyPSA lines in blue
    folium.GeoJson(
        pypsa_sample,
        name="PyPSA Sample",
        style_function=lambda x: {'color': 'blue', 'weight': 2, 'opacity': 0.8},
        tooltip=GeoJsonTooltip(fields=['id', 'voltage', 'length_km'],
                               aliases=['PyPSA ID', 'Voltage (kV)', 'Length (km)'])
    ).add_to(m)

    # Add nearest neighbors analysis
    for _, osm_line in osm_sample.iterrows():
        osm_geom = osm_line.geometry

        # Find the 3 closest PyPSA lines
        pypsa['distance'] = pypsa.geometry.apply(lambda g: osm_geom.distance(g))
        closest = pypsa.sort_values('distance').head(3)

        for _, pypsa_line in closest.iterrows():
            # Draw a light line connecting centroids
            osm_centroid = osm_geom.centroid
            pypsa_centroid = pypsa_line.geometry.centroid

            folium.PolyLine(
                [(osm_centroid.y, osm_centroid.x), (pypsa_centroid.y, pypsa_centroid.x)],
                color='green',
                weight=1,
                opacity=0.5,
                dash_array='5,5',
                popup=f"Distance: {pypsa_line.distance:.2f}m"
            ).add_to(m)

    # Add legends and layer control
    folium.LayerControl().add_to(m)

    # Save the map
    m.save(output_file)
    print(f"Diagnostic visualization saved to {output_file}")

    # Return some statistics for further analysis
    return {
        'osm_sample_count': len(osm_sample),
        'pypsa_sample_count': len(pypsa_sample),
        'average_min_distance': pypsa.groupby(pypsa.index).distance.min().mean()
    }


def debug_endpoint_matching(osm_gdf, pypsa_gdf, output_dir):
    """Analyze why endpoint matching might be failing"""
    import pandas as pd
    import numpy as np
    import os
    from shapely.geometry import Point

    # Create empty dataframes to store results
    results = []

    # Sample some OSM lines to analyze
    sample_size = min(100, len(osm_gdf))
    osm_sample = osm_gdf.sample(sample_size)

    # For each OSM line, find the distances to all PyPSA endpoints
    for osm_idx, osm_row in osm_sample.iterrows():
        # Get the OSM line's endpoints
        if osm_row.geometry.geom_type == 'LineString':
            osm_start = Point(osm_row.geometry.coords[0])
            osm_end = Point(osm_row.geometry.coords[-1])
        elif osm_row.geometry.geom_type == 'MultiLineString':
            # Get first and last points of the multilinestring
            coords = [list(geom.coords) for geom in osm_row.geometry.geoms]
            all_coords = [coord for sublist in coords for coord in sublist]
            osm_start = Point(all_coords[0])
            osm_end = Point(all_coords[-1])
        else:
            # Skip non-line geometries
            continue

        # Find minimum distances to PyPSA endpoints
        min_start_dist = float('inf')
        min_end_dist = float('inf')

        for pypsa_idx, pypsa_row in pypsa_gdf.iterrows():
            if pypsa_row.geometry.geom_type == 'LineString':
                pypsa_start = Point(pypsa_row.geometry.coords[0])
                pypsa_end = Point(pypsa_row.geometry.coords[-1])

                # Calculate distances
                start_to_start = osm_start.distance(pypsa_start)
                start_to_end = osm_start.distance(pypsa_end)
                end_to_start = osm_end.distance(pypsa_start)
                end_to_end = osm_end.distance(pypsa_end)

                # Update minimum distances
                min_start_dist = min(min_start_dist, start_to_start, start_to_end)
                min_end_dist = min(min_end_dist, end_to_start, end_to_end)

        # Store results
        results.append({
            'osm_id': osm_row.get('id', str(osm_idx)),
            'osm_voltage': osm_row.get('voltage', None),
            'osm_length_km': osm_row.get('length_km', None),
            'min_start_dist': min_start_dist,
            'min_end_dist': min_end_dist,
            'both_endpoints_under_2km': min_start_dist < 2000 and min_end_dist < 2000
        })

    # Convert to DataFrame and save
    results_df = pd.DataFrame(results)
    csv_file = os.path.join(output_dir, 'endpoint_matching_debug.csv')
    results_df.to_csv(csv_file, index=False)

    # Print summary statistics
    print("\n===== ENDPOINT MATCHING DEBUG =====")
    print(f"Analyzed {len(results)} OSM lines for endpoint matching")
    print(f"Average minimum start point distance: {results_df.min_start_dist.mean():.1f} meters")
    print(f"Average minimum end point distance: {results_df.min_end_dist.mean():.1f} meters")
    print(
        f"Lines with both endpoints within 2km: {results_df.both_endpoints_under_2km.sum()} ({results_df.both_endpoints_under_2km.mean() * 100:.1f}%)")

    # Suggest improvements
    if results_df.both_endpoints_under_2km.mean() < 0.5:
        print("\nSuggestion: Increase endpoint matching distance threshold")

    return results_df


def analyze_geometry_compatibility(osm_gdf, pypsa_gdf):
    """Analyze if there are geometry type or structure issues"""

    print("\n===== GEOMETRY COMPATIBILITY ANALYSIS =====")

    # Check geometry types
    osm_types = osm_gdf.geometry.type.value_counts()
    pypsa_types = pypsa_gdf.geometry.type.value_counts()

    print("OSM geometry types:")
    for gtype, count in osm_types.items():
        print(f"  {gtype}: {count} ({count / len(osm_gdf) * 100:.1f}%)")

    print("\nPyPSA geometry types:")
    for gtype, count in pypsa_types.items():
        print(f"  {gtype}: {count} ({count / len(pypsa_gdf) * 100:.1f}%)")

    # Check for MultiLineStrings vs LineStrings mismatch
    if 'MultiLineString' in osm_types and 'LineString' in pypsa_types:
        print("\nWarning: OSM uses MultiLineStrings while PyPSA uses LineStrings")
        print("This may cause issues in endpoint detection and shape comparison")
        print("Suggestion: Convert MultiLineStrings to LineStrings or handle both types specifically")

    # Check geometry complexity (number of vertices)
    osm_vertices = osm_gdf.geometry.apply(lambda g:
                                          sum(len(list(geom.coords)) for geom in g.geoms) if hasattr(g, 'geoms')
                                          else len(list(g.coords)) if hasattr(g, 'coords') else 0
                                          )

    pypsa_vertices = pypsa_gdf.geometry.apply(lambda g:
                                              sum(len(list(geom.coords)) for geom in g.geoms) if hasattr(g, 'geoms')
                                              else len(list(g.coords)) if hasattr(g, 'coords') else 0
                                              )

    print(
        f"\nOSM vertices per line: min={osm_vertices.min()}, max={osm_vertices.max()}, mean={osm_vertices.mean():.1f}")
    print(
        f"PyPSA vertices per line: min={pypsa_vertices.min()}, max={pypsa_vertices.max()}, mean={pypsa_vertices.mean():.1f}")

    if osm_vertices.mean() > pypsa_vertices.mean() * 2:
        print("\nWarning: OSM lines have significantly more vertices than PyPSA lines")
        print("This may affect shape matching and Hausdorff distance calculations")
        print("Suggestion: Consider simplifying OSM geometries or adjusting similarity thresholds")


def test_increased_buffer_sizes(osm_gdf, pypsa_gdf, output_dir):
    """Test if increased buffer sizes would help matching"""
    import pandas as pd
    import os
    import numpy as np
    from shapely.geometry import LineString, MultiLineString

    # Sample size for testing
    sample_size = min(500, len(osm_gdf))
    osm_sample = osm_gdf.sample(sample_size)

    buffer_sizes = [50, 100, 200, 500, 1000, 2000]
    results = []

    for _, osm_row in osm_sample.iterrows():
        osm_id = osm_row.get('id', 'unknown')
        osm_geom = osm_row.geometry

        # Try different buffer sizes
        for buffer_size in buffer_sizes:
            # Create buffer around OSM line
            buffer = osm_geom.buffer(buffer_size)

            # Count PyPSA lines that intersect with this buffer
            intersecting = pypsa_gdf[pypsa_gdf.intersects(buffer)]
            count = len(intersecting)

            # If we have intersections, calculate more metrics
            if count > 0:
                # Find best match based on Hausdorff distance
                best_match = None
                best_distance = float('inf')

                for _, pypsa_row in intersecting.iterrows():
                    try:
                        # Convert to LineString if needed for Hausdorff calculation
                        osm_line = osm_geom
                        pypsa_line = pypsa_row.geometry

                        if isinstance(osm_line, MultiLineString):
                            # Concatenate all parts
                            all_coords = []
                            for part in osm_line.geoms:
                                all_coords.extend(list(part.coords))
                            osm_line = LineString(all_coords)

                        # Calculate Hausdorff distance
                        hausdorff_dist = osm_line.hausdorff_distance(pypsa_line)

                        if hausdorff_dist < best_distance:
                            best_distance = hausdorff_dist
                            best_match = pypsa_row.get('id', 'unknown')
                    except Exception as e:
                        continue

                results.append({
                    'osm_id': osm_id,
                    'buffer_size': buffer_size,
                    'intersecting_count': count,
                    'best_match': best_match,
                    'hausdorff_distance': best_distance
                })
            else:
                results.append({
                    'osm_id': osm_id,
                    'buffer_size': buffer_size,
                    'intersecting_count': 0,
                    'best_match': None,
                    'hausdorff_distance': None
                })

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Save to CSV
    csv_file = os.path.join(output_dir, 'buffer_testing_results.csv')
    results_df.to_csv(csv_file, index=False)

    # Print summary
    print("\n===== BUFFER SIZE TESTING =====")
    for buffer_size in buffer_sizes:
        subset = results_df[results_df.buffer_size == buffer_size]
        match_count = subset[subset.intersecting_count > 0].shape[0]
        match_percent = match_count / sample_size * 100
        avg_intersections = subset[subset.intersecting_count > 0].intersecting_count.mean()

        print(f"Buffer {buffer_size}m: {match_count}/{sample_size} lines ({match_percent:.1f}%) have matches")
        if not np.isnan(avg_intersections):
            print(f"  Avg. intersections per matched line: {avg_intersections:.1f}")

    # Analyze Hausdorff distances
    hausdorff_data = results_df.dropna(subset=['hausdorff_distance'])
    if not hausdorff_data.empty:
        print("\nHausdorff distance statistics:")
        print(f"  Min: {hausdorff_data.hausdorff_distance.min():.1f}")
        print(f"  Max: {hausdorff_data.hausdorff_distance.max():.1f}")
        print(f"  Mean: {hausdorff_data.hausdorff_distance.mean():.1f}")
        print(f"  Median: {hausdorff_data.hausdorff_distance.median():.1f}")

        # Suggest threshold
        suggested = np.percentile(hausdorff_data.hausdorff_distance, 75)
        print(f"\nSuggested Hausdorff distance threshold: {suggested:.1f}")

    return results_df


# Add this as a fallback method for lines that don't match with path-based approach
def direct_buffer_overlap_matching(osm_gdf, pypsa_gdf, existing_matches):
    """Match lines purely by buffer overlap"""
    new_matches = []
    # Get IDs of already matched OSM lines
    matched_ids = set(m['osm_id'] for m in existing_matches if m.get('matched', False))

    # For each unmatched OSM line
    for _, osm_row in osm_gdf.iterrows():
        osm_id = osm_row['id']
        if osm_id in matched_ids:
            continue

        # Create a larger buffer for OSM line
        osm_buffer = osm_row.geometry.buffer(0.02)  # 2km buffer

        # Find PyPSA lines that intersect this buffer
        intersecting = pypsa_gdf[pypsa_gdf.geometry.intersects(osm_buffer)]

        if len(intersecting) > 0:
            # Calculate overlap percentage for each
            intersecting['overlap'] = intersecting.apply(
                lambda row: osm_buffer.intersection(row.geometry.buffer(0.02)).area / osm_buffer.area,
                axis=1
            )

            # Sort by overlap
            best_matches = intersecting.sort_values('overlap', ascending=False).head(1)

            if len(best_matches) > 0 and best_matches.iloc[0]['overlap'] > 0.5:
                # Found a good match
                match = {
                    'osm_id': osm_id,
                    'matched': True,
                    'pypsa_ids': [best_matches.iloc[0]['id']],
                    'match_quality': 'Simplified Geometry Match',
                    'geometric_score': best_matches.iloc[0]['overlap']
                }
                new_matches.append(match)

    return new_matches


import folium
from folium.plugins import MarkerCluster
import pandas as pd
import geopandas as gpd


def create_simple_match_visualization(osm_gdf, pypsa_gdf, matches, output_path):
    """
    Creates a simple map showing OSM and PyPSA lines with matching visualization.
    """
    import folium
    from folium import plugins
    import json

    # Create a centered map
    map_center = [51.1657, 10.4515]  # Center of Germany
    m = folium.Map(location=map_center, zoom_start=6,
                   tiles='CartoDB positron')

    # Create feature groups
    fg_osm_matched = folium.FeatureGroup(name="OSM Matched Lines")
    fg_osm_unmatched = folium.FeatureGroup(name="OSM Unmatched Lines")
    fg_pypsa_matched = folium.FeatureGroup(name="PyPSA Matched Lines")
    fg_pypsa_unmatched = folium.FeatureGroup(name="PyPSA Unmatched Lines")

    # Create sets for faster lookup of matched IDs
    if isinstance(matches, list):
        matched_osm_ids = set()
        matched_pypsa_ids = set()
        for match in matches:
            if match.get('matched', False):
                if 'osm_id' in match:
                    matched_osm_ids.add(match['osm_id'])
                if 'pypsa_ids' in match:
                    if isinstance(match['pypsa_ids'], list):
                        matched_pypsa_ids.update(match['pypsa_ids'])
                    else:
                        matched_pypsa_ids.add(match['pypsa_ids'])
    else:  # DataFrame
        matched_osm_ids = set(matches[matches['matched']]['osm_id'].unique())
        matched_pypsa_ids = set()
        for _, row in matches[matches['matched']].iterrows():
            if 'pypsa_ids' in row:
                if isinstance(row['pypsa_ids'], list):
                    matched_pypsa_ids.update(row['pypsa_ids'])
                else:
                    matched_pypsa_ids.add(row['pypsa_ids'])

    # Create a GeoJSON layer for search functionality
    search_layer = folium.FeatureGroup(name="Search Layer")

    # Add OSM lines
    id_field = 'id' if 'id' in osm_gdf.columns else 'osm_id'
    for idx, row in osm_gdf.iterrows():
        osm_id = row[id_field]
        is_matched = osm_id in matched_osm_ids

        # Convert geometry to GeoJSON
        geojson = folium.GeoJson(
            row.geometry.__geo_interface__,
            style_function=lambda x: {
                'color': 'green' if is_matched else 'red',
                'weight': 3 if is_matched else 2,
                'opacity': 0.8 if is_matched else 0.6
            },
            tooltip=f"OSM ID: {osm_id} ({'Matched' if is_matched else 'Unmatched'})"
        )

        # Add to appropriate feature group
        if is_matched:
            geojson.add_to(fg_osm_matched)
        else:
            geojson.add_to(fg_osm_unmatched)

        # Add search point for this line
        folium.CircleMarker(
            location=[row.geometry.centroid.y, row.geometry.centroid.x],
            radius=0,  # Make it invisible
            popup=f"OSM ID: {osm_id}",
            tooltip=f"OSM ID: {osm_id}",
            fill=False,
            opacity=0,
            properties={'name': f"OSM: {osm_id}"}
        ).add_to(search_layer)

    # Add PyPSA lines
    for idx, row in pypsa_gdf.iterrows():
        pypsa_id = row.get('id', str(idx))
        is_matched = pypsa_id in matched_pypsa_ids

        # Convert geometry to GeoJSON
        geojson = folium.GeoJson(
            row.geometry.__geo_interface__,
            style_function=lambda x: {
                'color': 'blue' if is_matched else 'purple',
                'weight': 2 if is_matched else 1.5,
                'opacity': 0.7 if is_matched else 0.6,
                'dashArray': '5, 5' if is_matched else '3, 3'
            },
            tooltip=f"PyPSA ID: {pypsa_id} ({'Matched' if is_matched else 'Unmatched'})"
        )

        # Add to appropriate feature group
        if is_matched:
            geojson.add_to(fg_pypsa_matched)
        else:
            geojson.add_to(fg_pypsa_unmatched)

        # Add search point for this line
        folium.CircleMarker(
            location=[row.geometry.centroid.y, row.geometry.centroid.x],
            radius=0,  # Make it invisible
            popup=f"PyPSA ID: {pypsa_id}",
            tooltip=f"PyPSA ID: {pypsa_id}",
            fill=False,
            opacity=0,
            properties={'name': f"PyPSA: {pypsa_id}"}
        ).add_to(search_layer)

    # Add feature groups to map
    search_layer.add_to(m)
    fg_osm_matched.add_to(m)
    fg_osm_unmatched.add_to(m)
    fg_pypsa_matched.add_to(m)
    fg_pypsa_unmatched.add_to(m)

    # Add layer control
    folium.LayerControl().add_to(m)

    # Add search control - using the layer approach
    search_control = plugins.Search(
        layer=search_layer,
        geom_type='Point',
        placeholder='Search for OSM/PyPSA ID',
        collapsed=False,
        search_label='name',
        search_zoom=12,
        position='topright'
    ).add_to(m)

    # Add title and legend
    title_html = '''
    <div style="position: fixed; top: 10px; left: 50%; transform: translateX(-50%); z-index: 9999; background-color: white; 
                padding: 10px; border-radius: 5px; box-shadow: 0 0 15px rgba(0,0,0,0.2);">
        <h3 style="margin: 0; text-align: center;">OSM-PyPSA Line Matching</h3>
        <div style="display: flex; justify-content: center; margin-top: 5px;">
            <div style="margin: 0 5px;"><span style="color: green;">■</span> Matched OSM</div>
            <div style="margin: 0 5px;"><span style="color: red;">■</span> Unmatched OSM</div>
            <div style="margin: 0 5px;"><span style="color: blue;">- -</span> Matched PyPSA</div>
            <div style="margin: 0 5px;"><span style="color: purple;">- -</span> Unmatched PyPSA</div>
        </div>
    </div>
    '''

    m.get_root().html.add_child(folium.Element(title_html))

    # Save to file
    m.save(output_path)
    print(f"Visualization saved to: {output_path}")

    return m


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
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Override matching inclusion if specified in arguments
    INCLUDE_DC_IN_MATCHING = args.include_dc_matching
    INCLUDE_110KV_IN_MATCHING = args.include_110kv_matching or args.osm_matching  # Always include 110kV for OSM matching

    # Override output inclusion if specified in arguments
    INCLUDE_DC_IN_OUTPUT = not args.no_dc_output
    INCLUDE_110KV_IN_OUTPUT = not args.no_110kv_output or args.osm_matching  # Always include 110kV for OSM mode

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

    # Determine matching mode
    matching_mode = "OSM-PyPSA" if args.osm_matching else "JAO-PyPSA"

    print(f"===== GRID MATCHER {matching_mode} MODE =====")
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

    # Process according to matching mode
    if args.osm_matching:
        process_osm_matching(args, OUTPUT_DIR)
    else:
        process_jao_matching(args, OUTPUT_DIR)

    print("\n===== PROCESS COMPLETE =====")


def process_osm_matching(args, OUTPUT_DIR):
    """Handle OSM-PyPSA matching workflow"""
    print("\n===== LOADING DATA FOR OSM-PYPSA MATCHING =====")

    # Load OSM data
    osm_gdf = load_osm_data(OSM_PATH, verbose=VERBOSE)
    if osm_gdf is None:
        print("ERROR: Could not load OSM data from", OSM_PATH)
        return

    # Preprocess OSM data
    osm_gdf = preprocess_osm_data(osm_gdf)

    # Load PyPSA 110kV data - this is required for OSM matching
    print("\n===== LOADING PYPSA 110KV DATA =====")
    pypsa_110kv_gdf = None
    if os.path.exists(PYPSA_110KV_PATH):
        pypsa_110kv_gdf = load_110kv_data(PYPSA_110KV_PATH, verbose=VERBOSE)
        # Preprocess PyPSA data
        pypsa_110kv_gdf = preprocess_pypsa_110kv_data(pypsa_110kv_gdf)

    if pypsa_110kv_gdf is None:
        print("ERROR: Could not load PyPSA 110kV data from", PYPSA_110KV_PATH)
        return

    # Verify matching data compatibility
    verify_matching_data(osm_gdf, pypsa_110kv_gdf)

    # Process manual matches if enabled
    manual_matches = []
    if ENABLE_MANUAL_MATCHING:
        # Load existing matches
        if os.path.exists(MANUAL_MATCHES_FILE):
            manual_matches = load_manual_matches_file(MANUAL_MATCHES_FILE)
            print(f"Loaded {len(manual_matches)} manual matches from {MANUAL_MATCHES_FILE}")

        # Add predefined matches if enabled
        if ADD_PREDEFINED_MATCHES:
            manual_matches = add_predefined_manual_matches(osm_gdf, pypsa_110kv_gdf, manual_matches,
                                                           interactive=False)
            save_manual_matches_file(manual_matches, MANUAL_MATCHES_FILE)

    # New approach: Use multiple matching methods in order of accuracy
    print("\n===== MULTI-STAGE OSM-PYPSA MATCHING =====")

    # Stage 1: Standard matching
    print("STAGE 1: Standard matching")
    standard_results = run_osm_pypsa_matching(
        osm_gdf,
        pypsa_110kv_gdf,
        OUTPUT_DIR,
        include_dc_links=INCLUDE_DC_IN_MATCHING,
        verbose=VERBOSE,
        ignore_voltage_differences=args.ignore_voltage_differences
    )

    # Count matched items from standard matching
    standard_matched = sum(1 for r in standard_results if r.get("matched", False))
    standard_unmatched = len(osm_gdf) - standard_matched
    print(f"Standard matching results: {standard_matched} matched, {standard_unmatched} unmatched")

    # Stage 2: Buffer matching for remaining unmatched lines
    print("\nSTAGE 2: Buffer overlap matching for remaining lines")

    # Identify unmatched OSM lines after standard matching
    matched_ids_stage1 = set(m['osm_id'] for m in standard_results if m.get("matched", False))
    unmatched_osm_gdf = osm_gdf[~osm_gdf['id'].isin(matched_ids_stage1)]
    print(f"Attempting buffer matching on {len(unmatched_osm_gdf)} unmatched lines")

    # Perform buffer matching
    buffer_results = []
    try:
        # Implementation of the buffer overlap matching
        for _, osm_row in unmatched_osm_gdf.iterrows():
            osm_id = osm_row['id']
            # Create a buffer around OSM line - try with a relatively large buffer
            osm_buffer = osm_row.geometry.buffer(0.02)  # 2km buffer
            # Find PyPSA lines that intersect this buffer
            intersecting = pypsa_110kv_gdf[pypsa_110kv_gdf.geometry.intersects(osm_buffer)]

            if len(intersecting) > 0:
                # Calculate overlap percentage for each
                intersecting['overlap'] = intersecting.apply(
                    lambda row: osm_buffer.intersection(row.geometry.buffer(0.01)).area / osm_buffer.area,
                    axis=1
                )
                # Sort by overlap
                best_matches = intersecting.sort_values('overlap', ascending=False).head(1)
                if len(best_matches) > 0 and best_matches.iloc[0]['overlap'] > 0.5:
                    # Found a good match
                    match = {
                        'osm_id': osm_id,
                        'matched': True,
                        'pypsa_ids': [best_matches.iloc[0]['id']],
                        'match_quality': 'Buffer Match',
                        'is_geometric_match': True,
                        'geometric_score': best_matches.iloc[0]['overlap']
                    }
                    buffer_results.append(match)
        print(f"Buffer matching found {len(buffer_results)} additional matches")
    except Exception as e:
        print(f"ERROR in buffer matching: {e}")
        traceback.print_exc()

    # Stage 3: Distance-based matching as a last resort
    print("\nSTAGE 3: Distance-based matching for remaining lines")

    # Identify lines still unmatched after buffer matching
    matched_ids_stage2 = set(m['osm_id'] for m in buffer_results if m.get("matched", False))
    all_matched_ids_so_far = matched_ids_stage1.union(matched_ids_stage2)
    still_unmatched_osm_gdf = osm_gdf[~osm_gdf['id'].isin(all_matched_ids_so_far)]
    print(f"Attempting distance-based matching on {len(still_unmatched_osm_gdf)} unmatched lines")

    # Perform distance matching
    distance_results = []
    try:
        # Simple distance-based matching
        for _, osm_row in still_unmatched_osm_gdf.iterrows():
            osm_id = osm_row['id']
            osm_geom = osm_row.geometry
            # Calculate distances to all PyPSA lines
            pypsa_110kv_gdf['distance'] = pypsa_110kv_gdf.apply(
                lambda row: osm_geom.distance(row.geometry), axis=1
            )
            # Get closest PyPSA line
            closest = pypsa_110kv_gdf.sort_values('distance').iloc[0]
            # If within reasonable distance, consider it a match
            if closest['distance'] < 0.05:  # 5km threshold
                match = {
                    'osm_id': osm_id,
                    'matched': True,
                    'pypsa_ids': [closest['id']],
                    'match_quality': 'Distance Match',
                    'is_geometric_match': True,
                    'geometric_score': 1.0 - (closest['distance'] * 20)  # Scale to 0-1
                }
                distance_results.append(match)
        print(f"Distance matching found {len(distance_results)} additional matches")
    except Exception as e:
        print(f"ERROR in distance matching: {e}")
        traceback.print_exc()

    # Combine results from all three stages into a single consolidated list
    all_matching_results = []
    all_matching_results.extend(standard_results)
    all_matching_results.extend(buffer_results)
    all_matching_results.extend(distance_results)

    # Calculate final matching statistics
    final_matched = sum(1 for r in all_matching_results if r.get("matched", False))
    final_unmatched = len(osm_gdf) - final_matched

    print("\n===== OSM-PYPSA MATCHING SUMMARY =====")
    print(f"Total OSM lines: {len(osm_gdf)}")
    print(f"Standard matching: {standard_matched} lines ({standard_matched / len(osm_gdf) * 100:.1f}%)")
    print(f"Buffer matching: {len(buffer_results)} additional lines ({len(buffer_results) / len(osm_gdf) * 100:.1f}%)")
    print(
        f"Distance matching: {len(distance_results)} additional lines ({len(distance_results) / len(osm_gdf) * 100:.1f}%)")
    print(f"Final match rate: {final_matched}/{len(osm_gdf)} ({final_matched / len(osm_gdf) * 100:.1f}%)")

    # Apply manual matches if enabled and we have matches
    if ENABLE_MANUAL_MATCHING and manual_matches:
        print("\n===== APPLYING MANUAL MATCHES =====")
        all_matching_results = apply_manual_matches_safely(all_matching_results, osm_gdf, pypsa_110kv_gdf,
                                                           manual_matches)

    # Create CSV output of results
    csv_file = OUTPUT_DIR / 'osm_pypsa_matches.csv'
    create_results_csv(all_matching_results, csv_file)
    print(f"Match results saved to {csv_file}")

    # Generate PyPSA with OSM parameters
    print("\n===== GENERATING PYPSA WITH OSM PARAMETERS =====")
    from grid_matcher.matcher.osm_pypsa_matcher import generate_pypsa_with_osm_parameters
    pypsa_match_count, pypsa_with_osm, pypsa_files = generate_pypsa_with_osm_parameters(
        all_matching_results, osm_gdf, pypsa_110kv_gdf, OUTPUT_DIR
    )
    print(f"Generated PyPSA with OSM data - {pypsa_match_count} lines matched")

    # After the existing visualization code in process_osm_matching
    # Keep the existing visualization
    print("\n===== GENERATING VISUALIZATION MAP =====")
    map_file = OUTPUT_DIR / 'osm_pypsa_matches.html'

    # In your process_osm_matching function, after you've completed matching
    # Add this code after line ~2140 (after the existing visualization code)

    print("\n===== GENERATING SIMPLE MATCH VISUALIZATION =====")
    simple_match_file = OUTPUT_DIR / 'osm_pypsa_simple_matches.html'
    create_simple_match_visualization(osm_gdf, pypsa_110kv_gdf, all_matching_results, simple_match_file)
    print(f"Simple match visualization saved to {simple_match_file}")

    # Add these debugging lines right here
    print("Sample OSM IDs:", osm_gdf['id'].head().tolist())
    print("Sample match OSM IDs:", [r['osm_id'] for r in all_matching_results[:5] if r.get("matched", False)])

    create_jao_pypsa_visualization(osm_gdf, pypsa_110kv_gdf, all_matching_results, map_file)
    print(f"Visualization map saved to {map_file}")

    # Add new specialized OSM-PyPSA visualization
    print("\n===== GENERATING OSM-PYPSA MATCH VISUALIZATION =====")
    osm_match_map_file = OUTPUT_DIR / 'osm_pypsa_matches_detailed.html'
    create_osm_pypsa_match_visualization(osm_gdf, pypsa_110kv_gdf, all_matching_results, osm_match_map_file)
    print(f"OSM-PyPSA specialized match visualization saved to {osm_match_map_file}")


    # Generate parameter visualization if requested
    if GENERATE_PARAMETER_VISUALIZATION:
        print("\n===== GENERATING PARAMETER COMPARISON VISUALIZATION =====")
        try:
            # Prepare visualization data using all matching results
            print(f"Total matching results: {len(all_matching_results)}")
            matched_count = sum(1 for r in all_matching_results if r.get("matched", False))
            print(f"Matched results: {matched_count}")

            # Transform data for visualization
            enhanced_results = prepare_visualization_data(
                all_matching_results,
                pypsa_110kv_gdf,
                jao_gdf=osm_gdf
            )

            # Create parameter comparison visualization
            visualization_path = visualize_parameter_comparison(
                enhanced_results,
                pypsa_110kv_gdf,
                output_dir=OUTPUT_DIR
            )
            print(f"Parameter comparison visualization saved to: {visualization_path}")

            # Create enhanced summary table
            summary_path = create_enhanced_summary_table(
                osm_gdf,
                pypsa_110kv_gdf,
                all_matching_results,
                output_dir=OUTPUT_DIR
            )
            print(f"Parameter summary table saved to: {summary_path}")
        except Exception as e:
            print(f"Error generating parameter comparison visualization: {e}")
            traceback.print_exc()


def process_jao_matching(args, OUTPUT_DIR):
    """Handle JAO-PyPSA matching workflow"""
    # Load main data
    print("\n===== LOADING MAIN DATA =====")
    jao_gdf, pypsa_gdf = load_data(JAO_PATH, PYPSA_PATH)

    # Import new lines if enabled
    if IMPORT_NEW_LINES:
        jao_gdf, pypsa_gdf = import_new_lines_from_csv(jao_gdf, pypsa_gdf, DATA_DIR)

    # Process manual matches if enabled
    manual_matches = []
    if ENABLE_MANUAL_MATCHING:
        # Load existing matches
        if os.path.exists(MANUAL_MATCHES_FILE):
            manual_matches = load_manual_matches_file(MANUAL_MATCHES_FILE)
            print(f"Loaded {len(manual_matches)} manual matches from {MANUAL_MATCHES_FILE}")

        # Add predefined matches if enabled
        if ADD_PREDEFINED_MATCHES:
            manual_matches = add_predefined_manual_matches(jao_gdf, pypsa_gdf, manual_matches, interactive=False)
            save_manual_matches_file(manual_matches, MANUAL_MATCHES_FILE)

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
    if ENABLE_MANUAL_MATCHING and manual_matches:
        print("\n===== APPLYING MANUAL MATCHES =====")
        results = apply_manual_matches_safely(results, jao_gdf, pypsa_gdf, manual_matches)

    # Regenerate outputs with manual matches
    print("\n===== GENERATING OUTPUTS =====")

    # 1. Export results to CSV
    csv_file = OUTPUT_DIR / 'jao_pypsa_matches.csv'
    create_results_csv(results, csv_file)
    print(f"Match results saved to {csv_file}")

    # 2. Generate PyPSA with source codes/parameters
    pypsa_match_count, pypsa_with_source, pypsa_files = generate_pypsa_with_eic(
        results, jao_gdf, pypsa_gdf, OUTPUT_DIR
    )

    # 3. Generate JAO with PyPSA electrical parameters
    source_with_pypsa = generate_jao_with_pypsa(
        results, jao_gdf, pypsa_gdf, OUTPUT_DIR
    )
    print(f"Generated outputs with {pypsa_match_count} matched lines")

    # 4. Create visualization map with all results
    map_file = OUTPUT_DIR / 'jao_pypsa_matches.html'
    create_jao_pypsa_visualization(jao_gdf, pypsa_gdf, results, map_file)
    print(f"Visualization map saved to {map_file}")

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

                # Transform data for visualization
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

                # Create enhanced summary table
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
            traceback.print_exc()



def preprocess_pypsa_110kv_data(pypsa_gdf):
    """Preprocess PyPSA data to ensure compatibility"""
    print("\n===== PREPROCESSING PYPSA DATA =====")

    # 1. Ensure ID columns are properly set
    if 'id' not in pypsa_gdf.columns and 'line_id' in pypsa_gdf.columns:
        print("Adding 'id' column based on line_id")
        pypsa_gdf['id'] = pypsa_gdf['line_id']

    # 2. Ensure voltage is numeric
    if 'voltage' in pypsa_gdf.columns:
        try:
            # Extract numeric value from strings like '110-AC' if needed
            if pypsa_gdf['voltage'].dtype == 'object':
                print("Converting voltage strings to numeric values")
                pypsa_gdf['voltage'] = pypsa_gdf['voltage'].str.extract('(\d+)').astype(float)
        except Exception as e:
            print(f"Warning: Could not convert voltage to numeric: {e}")

    # 3. Make sure length is in km if it's in m
    if 'length' in pypsa_gdf.columns:
        sample_length = pypsa_gdf['length'].iloc[0] if len(pypsa_gdf) > 0 else 0
        if sample_length and sample_length > 1000:  # Likely in meters
            if 'length_km' not in pypsa_gdf.columns:
                print(f"Adding length_km column (converted from meters)")
                pypsa_gdf['length_km'] = pypsa_gdf['length'] / 1000.0

    return pypsa_gdf


def verify_matching_data(osm_gdf, pypsa_gdf):
    """Verify data is properly prepared for matching"""
    print("\n===== VERIFYING MATCHING DATA =====")

    # Check ID uniqueness
    osm_id_unique = osm_gdf['id'].nunique()
    pypsa_id_unique = pypsa_gdf['id'].nunique()

    print(f"OSM ID uniqueness: {osm_id_unique}/{len(osm_gdf)} ({osm_id_unique / len(osm_gdf) * 100:.1f}%)")
    print(f"PyPSA ID uniqueness: {pypsa_id_unique}/{len(pypsa_gdf)} ({pypsa_id_unique / len(pypsa_gdf) * 100:.1f}%)")

    if osm_id_unique < len(osm_gdf) * 0.9:
        print("WARNING: OSM IDs are not sufficiently unique!")

    # Check geometry types
    osm_geom_types = osm_gdf.geometry.type.value_counts().to_dict()
    pypsa_geom_types = pypsa_gdf.geometry.type.value_counts().to_dict()

    print(f"OSM geometry types: {osm_geom_types}")
    print(f"PyPSA geometry types: {pypsa_geom_types}")

    # Check voltage ranges
    if 'voltage' in osm_gdf.columns and 'voltage' in pypsa_gdf.columns:
        osm_volt_range = (osm_gdf['voltage'].min(), osm_gdf['voltage'].max())
        pypsa_volt_range = (pypsa_gdf['voltage'].min(), pypsa_gdf['voltage'].max())

        print(f"OSM voltage range: {osm_volt_range}")
        print(f"PyPSA voltage range: {pypsa_volt_range}")

        if abs(osm_volt_range[0] - pypsa_volt_range[0]) > 100:
            print("WARNING: Voltage ranges differ significantly between datasets!")

    # Try simplified matching on a small sample
    print("\nTesting simplified matching on sample...")
    buffer_size = 5000  # 5km buffer
    sample_size = min(10, len(osm_gdf))

    osm_sample = osm_gdf.sample(sample_size) if len(osm_gdf) > sample_size else osm_gdf.head(sample_size)

    for idx, osm_line in osm_sample.iterrows():
        buffered = osm_line.geometry.buffer(buffer_size)
        matches = pypsa_gdf[pypsa_gdf.geometry.intersects(buffered)]

        print(f"OSM line {osm_line['id']} has {len(matches)} potential matches within {buffer_size}m")

        if len(matches) > 0:
            first_match = matches.iloc[0]
            print(f"  Example match: {first_match['id']} (voltage={first_match.get('voltage', 'N/A')})")

            # Calculate distance between line endpoints
            try:
                from grid_matcher.matcher.original_matcher import get_start_point, get_end_point

                osm_start = get_start_point(osm_line.geometry)
                osm_end = get_end_point(osm_line.geometry)

                pypsa_start = get_start_point(first_match.geometry)
                pypsa_end = get_end_point(first_match.geometry)

                from shapely.geometry import Point

                start_dist = Point(osm_start).distance(Point(pypsa_start))
                end_dist = Point(osm_end).distance(Point(pypsa_end))

                print(f"  Start point distance: {start_dist:.0f}m, End point distance: {end_dist:.0f}m")
            except Exception as e:
                print(f"  Could not calculate endpoint distances: {e}")


if __name__ == "__main__":
    main()