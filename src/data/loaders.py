import os
import logging
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString, MultiLineString
from shapely import wkt
from typing import Tuple, Optional, Union
from src.data.processors import safe_calc_per_km
logger = logging.getLogger(__name__)




def parse_wkt_geometry(geom_str: Optional[str]) -> Optional[Union[LineString, MultiLineString]]:
    """
    Parse WKT geometry string into a Shapely geometry with improved error handling.
    """
    if not isinstance(geom_str, str):
        return None
    try:
        geometry = wkt.loads(geom_str)
        return geometry
    except Exception as e:
        logger.debug(f"Error parsing geometry: {str(e)}")
        return None


def load_data(dlr_file: str, network_file: str) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Load DLR and network line data.
    The function reads CSV files, parses the WKT geometries, ensures that
    CRS is set to WGS84, and adds default columns if necessary.
    """
    logger.info(f"Loading data from {dlr_file} and {network_file}")

    if not os.path.exists(dlr_file):
        logger.error(f"DLR file not found: {dlr_file}")
        raise FileNotFoundError(f"DLR file not found: {dlr_file}")
    if not os.path.exists(network_file):
        logger.error(f"Network file not found: {network_file}")
        raise FileNotFoundError(f"Network file not found: {network_file}")

    dlr_lines = pd.read_csv(dlr_file)
    network_lines = pd.read_csv(network_file)
    logger.info(f"Raw data: {len(dlr_lines)} DLR lines and {len(network_lines)} network lines")

    # If network file uses a different column name for geometry.
    if 'geometry' in dlr_lines.columns and 'geom' in network_lines.columns:
        network_lines['geometry'] = network_lines['geom']
    elif 'geom' in network_lines.columns:
        network_lines['geometry'] = network_lines['geom']

    dlr_lines['geometry'] = dlr_lines['geometry'].apply(parse_wkt_geometry)
    network_lines['geometry'] = network_lines['geometry'].apply(parse_wkt_geometry)

    valid_dlr = dlr_lines['geometry'].notna().sum()
    valid_network = network_lines['geometry'].notna().sum()
    logger.info(f"Valid geometries: {valid_dlr}/{len(dlr_lines)} DLR lines and {valid_network}/{len(network_lines)} network lines")

    dlr_lines = gpd.GeoDataFrame(dlr_lines, geometry='geometry')
    network_lines = gpd.GeoDataFrame(network_lines, geometry='geometry')
    dlr_lines.crs = "EPSG:4326"
    network_lines.crs = "EPSG:4326"
    dlr_lines = dlr_lines[dlr_lines['geometry'].notna()]
    network_lines = network_lines[network_lines['geometry'].notna()]

    for df, name in [(dlr_lines, 'DLR'), (network_lines, 'network')]:
        if 'length' not in df.columns:
            logger.warning(f"'length' column not found in {name} data. Adding it based on geometry.")
            df['length'] = df.geometry.length * 111  # rough conversion degrees to km
        for param in ['r', 'x', 'b']:
            if param not in df.columns:
                logger.warning(f"'{param}' column not found in {name} data. Adding it with default value 0.")
                df[param] = 0

    logger.info("Calculating per-km electrical parameters...")
    for df in [dlr_lines, network_lines]:
        df['r_per_km'] = df.apply(lambda row: safe_calc_per_km(row, 'r'), axis=1)
        df['x_per_km'] = df.apply(lambda row: safe_calc_per_km(row, 'x'), axis=1)
        df['b_per_km'] = df.apply(lambda row: safe_calc_per_km(row, 'b'), axis=1)

    return dlr_lines, network_lines

def load_germany_boundary(geojson_file: str) -> Optional[gpd.GeoDataFrame]:
    """
    Load a detailed Germany boundary from a GeoJSON file.
    The function dissolves multiple geometries into one and ensures the CRS is WGS84.
    """
    try:
        if not os.path.exists(geojson_file):
            logger.warning(f"GeoJSON file not found: {geojson_file}")
            return None

        logger.info(f"Loading Germany boundary from GeoJSON file: {geojson_file}")
        germany_gdf = gpd.read_file(geojson_file)
        if germany_gdf.empty:
            logger.warning("GeoJSON file loaded but contains no geometries")
            return None

        if len(germany_gdf) > 1:
            logger.info(f"Dissolving {len(germany_gdf)} features into a single boundary")
            germany_gdf = germany_gdf.dissolve().reset_index()
        if germany_gdf.crs is None:
            logger.warning("No CRS found in GeoJSON, assuming EPSG:4326")
            germany_gdf.crs = "EPSG:4326"
        elif germany_gdf.crs != "EPSG:4326":
            logger.info(f"Converting CRS from {germany_gdf.crs} to EPSG:4326")
            germany_gdf = germany_gdf.to_crs("EPSG:4326")
        logger.info("Successfully loaded detailed Germany boundary from GeoJSON")
        return germany_gdf

    except Exception as e:
        logger.error(f"Error loading Germany boundary from GeoJSON: {str(e)}")
        return None


def load_pypsa_eur_data(file_path: str) -> gpd.GeoDataFrame:
    """
    Load PyPSA-EUR line data from CSV file, properly handling geometries with embedded commas.

    Parameters:
    - file_path: Path to the pypsa-eur-lines.csv file

    Returns:
    - GeoDataFrame with line geometries and attributes
    """
    logger.info(f"Loading PyPSA-EUR data from {file_path}")
    if not os.path.exists(file_path):
        logger.error(f"PyPSA-EUR file not found: {file_path}")
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    try:
        # Read the file as text
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # Parse header
        header = lines[0].strip().split(',')

        # Find the index of the geometry column
        geometry_col_index = header.index('geometry')

        # Parse each line manually
        data = []
        for i, line in enumerate(lines[1:], 1):  # Skip header
            # Skip empty lines
            if not line.strip():
                continue

            # Find the start of the geometry string (it's enclosed in quotes)
            geom_start = line.find("'LINESTRING")
            if geom_start == -1:
                logger.warning(f"No LINESTRING found in line {i + 1}, skipping")
                continue

            # Split the line into fields (before geometry) and geometry
            fields_part = line[:geom_start].strip().rstrip(',')
            geom_part = line[geom_start:].strip()

            # Split the fields part by comma
            fields = fields_part.split(',')

            # Ensure we have the right number of fields before geometry
            if len(fields) < geometry_col_index:
                # Try to fix by adding empty fields
                fields.extend([''] * (geometry_col_index - len(fields)))

            # Create row with all fields including geometry at the end
            row = fields.copy()
            row.append(geom_part)

            # Add to data
            data.append(row)

        # Create DataFrame
        df = pd.DataFrame(data, columns=header)

        # Parse geometries
        geometries = []
        for geom_str in df['geometry']:
            try:
                # Clean up the string - remove quotes
                if geom_str.startswith("'") and geom_str.endswith("'"):
                    geom_str = geom_str[1:-1]

                # Parse the WKT
                geometry = wkt.loads(geom_str)
                geometries.append(geometry)
            except Exception as e:
                logger.warning(f"Error parsing geometry: {str(e)}")
                geometries.append(None)

        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame(df, geometry=geometries, crs="EPSG:4326")

        # Remove rows with invalid geometries
        gdf = gdf[gdf.geometry.notna()]

        # Set ID and voltage columns
        if 'id' not in gdf.columns:
            gdf['id'] = gdf['line_id']

        if 'v_nom' not in gdf.columns:
            gdf['v_nom'] = pd.to_numeric(gdf['voltage'], errors='coerce')

        # Convert numerical columns
        for col in ['r', 'x', 'b', 'length', 'i_nom', 's_nom']:
            if col in gdf.columns:
                gdf[col] = pd.to_numeric(gdf[col], errors='coerce')

        # Calculate per-km electrical parameters
        gdf['r_per_km'] = gdf.apply(lambda row: safe_calc_per_km(row, 'r'), axis=1)
        gdf['x_per_km'] = gdf.apply(lambda row: safe_calc_per_km(row, 'x'), axis=1)
        gdf['b_per_km'] = gdf.apply(lambda row: safe_calc_per_km(row, 'b'), axis=1)

        logger.info(f"Successfully loaded {len(gdf)} PyPSA-EUR lines with valid geometries")

        # Log the first few lines for debugging
        if len(gdf) > 0:
            sample = gdf.iloc[0]
            logger.info(f"Sample PyPSA-EUR line: ID={sample['line_id']}, Voltage={sample['voltage']}, "
                        f"Length={sample['length']}, Geometry type={sample.geometry.geom_type}")

        return gdf

    except Exception as e:
        logger.error(f"Error loading PyPSA-EUR data: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")


def load_tso_data(file_path, tso_name):
    """
    A simplified function to load TSO line data from CSV with direct column mapping.

    Parameters:
    - file_path: Path to the CSV file
    - tso_name: Name of the TSO ('50Hertz' or 'TenneT')

    Returns:
    - GeoDataFrame with line geometries
    """
    logger.info(f"Loading {tso_name} line data from {file_path} using simplified loader")

    if not os.path.exists(file_path):
        logger.error(f"{tso_name} file not found: {file_path}")
        raise FileNotFoundError(f"{tso_name} file not found: {file_path}")

    # Load the CSV data
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Loaded {len(df)} rows from {tso_name} data")
    except Exception as e:
        logger.error(f"Error loading {tso_name} data: {str(e)}")
        raise

    # Create direct column mappings based on the provided CSV structure
    if tso_name == '50Hertz':
        column_map = {
            'id_col': 'NE_name',
            'lon1_col': 'Longitude_Substation_1',
            'lat1_col': 'Latitude_Substation_1',
            'lon2_col': 'Longitude_Substation_2',
            'lat2_col': 'Latitude_Substation_2',
            'voltage_col': 'Voltage_level(kV)',
            'length_col': 'Length_(km)',
            'r_col': 'Resistance_R(Ω)',
            'x_col': 'Reactance_X(Ω)',
            'b_col': 'Susceptance_B(μS)'
        }
    elif tso_name == 'TenneT':
        # Updated column mapping based on your actual TenneT data structure
        column_map = {
            'id_col': 'NE_name',
            'lon1_col': 'Longitude',  # These are the correct column names
            'lat1_col': 'Latitude',
            'lon2_col': 'Longitude',  # These will be accessed with a row offset
            'lat2_col': 'Latitude',  # for Substation_2
            'voltage_col': 'Voltage_level(kV)',
            'length_col': 'Length_(km)',
            'r_col': 'Resistance_R(Ω)',
            'x_col': 'Reactance_X(Ω)',
            'b_col': 'Susceptance_B(μS)'
        }
    else:
        raise ValueError(f"Unknown TSO: {tso_name}")

    # Log column availability
    for name, col in column_map.items():
        logger.info(f"{name} ({col}): {'Available' if col in df.columns else 'MISSING'}")

    # Check if essential coordinate columns exist
    coord_cols = [column_map['lon1_col'], column_map['lat1_col']]
    missing_coords = [col for col in coord_cols if col not in df.columns]

    if missing_coords:
        # Print available columns for debugging
        logger.error(f"Missing coordinate columns: {missing_coords}")
        logger.error(f"Available columns: {df.columns.tolist()}")
        raise ValueError(f"Cannot find proper coordinate columns in {tso_name} data")

    # Create geometries - simple and direct approach
    geometries = []
    for idx, row in df.iterrows():
        try:
            if tso_name == 'TenneT':
                # Get coordinates directly from the row for TenneT data
                x1 = float(row[column_map['lon1_col']])  # Longitude for Substation_1
                y1 = float(row[column_map['lat1_col']])  # Latitude for Substation_1

                # Get the longitude and latitude for Substation_2 (same row, different columns)
                # The 10th and 11th columns contain the Substation_2 coordinates
                x2 = float(row['Longitude'])  # Assuming this is 10th column
                y2 = float(row['Latitude'])  # Assuming this is 11th column
            else:
                # Use standard approach for other TSOs
                x1 = float(row[column_map['lon1_col']])
                y1 = float(row[column_map['lat1_col']])
                x2 = float(row[column_map['lon2_col']])
                y2 = float(row[column_map['lat2_col']])

            # Create LineString
            line = LineString([(x1, y1), (x2, y2)])
            geometries.append(line)

            # Debug first few
            if idx < 3:
                logger.info(f"Line {idx}: ({x1}, {y1}) to ({x2}, {y2})")

        except Exception as e:
            logger.warning(f"Error creating geometry for row {idx}: {str(e)}")
            geometries.append(None)

    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(df, geometry=geometries, crs="EPSG:4326")

    # Add ID column
    id_col = column_map['id_col']
    if id_col in df.columns:
        gdf['id'] = df[id_col]
    else:
        gdf['id'] = [f"{tso_name}_{i}" for i in range(len(df))]

    # Add voltage column
    voltage_col = column_map['voltage_col']
    if voltage_col in df.columns:
        gdf['v_nom'] = df[voltage_col]
    else:
        gdf['v_nom'] = 380  # Default to 380 kV

    # Get length directly
    length_col = column_map['length_col']
    if length_col in df.columns:
        gdf['length'] = df[length_col]
    else:
        # Calculate length in kilometers (rough approximation)
        gdf['length'] = gdf.geometry.length * 111

    # Get electrical parameters directly
    r_col = column_map['r_col']
    x_col = column_map['x_col']
    b_col = column_map['b_col']

    if r_col in df.columns:
        gdf['r'] = df[r_col]
    else:
        gdf['r'] = 0

    if x_col in df.columns:
        gdf['x'] = df[x_col]
    else:
        gdf['x'] = 0

    if b_col in df.columns:
        # Convert μS to S
        gdf['b'] = df[b_col] * 1e-6
    else:
        gdf['b'] = 0

    # Calculate per-km electrical parameters
    gdf['r_per_km'] = gdf.apply(lambda row: safe_calc_per_km(row, 'r'), axis=1)
    gdf['x_per_km'] = gdf.apply(lambda row: safe_calc_per_km(row, 'x'), axis=1)
    gdf['b_per_km'] = gdf.apply(lambda row: safe_calc_per_km(row, 'b'), axis=1)

    # Remove rows with invalid geometries
    valid_geom_count = gdf.geometry.notna().sum()
    gdf = gdf[gdf.geometry.notna()]
    logger.info(f"Created {len(gdf)} valid line geometries from {tso_name} data (from {len(df)} rows)")

    return gdf


def load_tennet_data(file_path):
    """Load TenneT data with more flexible column structure handling"""
    logger.info(f"Loading TenneT data from {file_path}")

    if not os.path.exists(file_path):
        logger.error(f"TenneT file not found: {file_path}")
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    try:
        # Load the data
        df = pd.read_csv(file_path)
        logger.info(f"Loaded {len(df)} rows from TenneT data")
        logger.info(f"Available columns: {df.columns.tolist()}")

        # Identify coordinate columns - try different patterns
        coordinate_patterns = [
            # Standard .1 suffix for duplicates
            {'lon1': 'Longitude', 'lat1': 'Latitude', 'lon2': 'Longitude.1', 'lat2': 'Latitude.1'},
            # Substation numbering in column names
            {'lon1': 'Longitude_Substation_1', 'lat1': 'Latitude_Substation_1',
             'lon2': 'Longitude_Substation_2', 'lat2': 'Latitude_Substation_2'},
            # Simple column numbering
            {'lon1': 'Longitude', 'lat1': 'Latitude', 'lon2': 'Longitude_2', 'lat2': 'Latitude_2'},
            # Completely different column names
            {'lon1': 'lon1', 'lat1': 'lat1', 'lon2': 'lon2', 'lat2': 'lat2'},
            # For files with columns based on our renaming logic
            {'lon1': 'Longitude', 'lat1': 'Latitude', 'lon2': 'Longitude.0', 'lat2': 'Latitude.0'},
        ]

        # Find which pattern matches our columns
        matching_pattern = None
        for pattern in coordinate_patterns:
            if (pattern['lon1'] in df.columns and pattern['lat1'] in df.columns and
                    pattern['lon2'] in df.columns and pattern['lat2'] in df.columns):
                matching_pattern = pattern
                logger.info(f"Found matching coordinate pattern: {pattern}")
                break

        # If no standard pattern found, try to detect coordinate columns
        if matching_pattern is None:
            logger.info("No standard coordinate pattern found, detecting columns...")
            lon_cols = [col for col in df.columns if 'lon' in col.lower() or 'longitude' in col.lower()]
            lat_cols = [col for col in df.columns if 'lat' in col.lower() or 'latitude' in col.lower()]

            logger.info(f"Detected longitude columns: {lon_cols}")
            logger.info(f"Detected latitude columns: {lat_cols}")

            if len(lon_cols) >= 2 and len(lat_cols) >= 2:
                matching_pattern = {
                    'lon1': lon_cols[0], 'lat1': lat_cols[0],
                    'lon2': lon_cols[1], 'lat2': lat_cols[1]
                }
                logger.info(f"Created coordinate pattern from detected columns: {matching_pattern}")

        # Check if we have found the columns
        if matching_pattern is None:
            # Last resort - try a row-based approach where the same column name is used multiple times
            if 'Longitude' in df.columns and 'Latitude' in df.columns:
                logger.info("Trying a row-based approach with Full_name as identifier")

                # Check if we can group by Full_name or similar column
                if 'Full_name' in df.columns:
                    logger.info("Found 'Full_name' column for substation grouping")
                    # Special processing here
                    substations = df['Full_name'].unique()
                    # This doesn't work as-is but illustrates the approach
                else:
                    logger.error("Cannot identify coordinate column pattern and no Full_name column for grouping")
                    return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
            else:
                logger.error("Cannot identify coordinate column pattern")
                return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

        # Create geometries from the matched columns
        geometries = []
        successful_count = 0

        # Print some sample data for debugging
        if len(df) > 0:
            sample_row = df.iloc[0]
            logger.info(f"Sample row first few columns: {dict(list(sample_row.items())[:10])}")

        # Try row-based approach first if we have Full_name column
        if 'Full_name' in df.columns:
            logger.info("Using Full_name-based row grouping approach")
            # Keep track of processed rows to avoid duplicates
            processed_rows = set()

            for idx, row in df.iterrows():
                # Skip already processed rows
                if idx in processed_rows:
                    continue

                try:
                    # Get the Full_name for the current row (1st substation)
                    sub1_name = row['Full_name']
                    sub1_lon = float(row['Longitude'])
                    sub1_lat = float(row['Latitude'])

                    # Look for the next row with the same NE_name but different Full_name (2nd substation)
                    next_row = None
                    next_idx = None

                    if 'NE_name' in df.columns:
                        ne_name = row['NE_name']

                        # Look through the rows for a matching second substation
                        for i, r in df.iloc[idx + 1:].iterrows():
                            if r['NE_name'] == ne_name and r['Full_name'] != sub1_name:
                                next_row = r
                                next_idx = i
                                break

                        if next_row is not None:
                            # Found the matching second substation
                            sub2_lon = float(next_row['Longitude'])
                            sub2_lat = float(next_row['Latitude'])

                            # Make sure both coordinate sets are valid
                            if (-180 <= sub1_lon <= 180 and -90 <= sub1_lat <= 90 and
                                    -180 <= sub2_lon <= 180 and -90 <= sub2_lat <= 90):

                                # Create LineString geometry
                                line = LineString([(sub1_lon, sub1_lat), (sub2_lon, sub2_lat)])
                                geometries.append(line)
                                successful_count += 1

                                # Mark both rows as processed
                                processed_rows.add(idx)
                                processed_rows.add(next_idx)

                                if successful_count <= 3:  # Log first few
                                    logger.info(f"Created TenneT geometry from rows {idx} and {next_idx}: "
                                                f"({sub1_lon}, {sub1_lat}) to ({sub2_lon}, {sub2_lat}) "
                                                f"[{sub1_name} to {next_row['Full_name']}]")
                            else:
                                logger.warning(
                                    f"Invalid coordinates: ({sub1_lon}, {sub1_lat}) to ({sub2_lon}, {sub2_lat})")
                                geometries.append(None)
                        else:
                            # No matching second substation found
                            logger.warning(f"No matching second substation found for {ne_name} at row {idx}")
                            geometries.append(None)
                    else:
                        logger.warning("NE_name column not found, cannot match substations")
                        geometries.append(None)
                except Exception as e:
                    logger.warning(f"Error creating TenneT geometry for rows starting at {idx}: {str(e)}")
                    geometries.append(None)

            # Fill in gaps for any unprocessed rows
            for idx in range(len(df)):
                if idx not in processed_rows:
                    geometries.append(None)

        else:
            # Use standard column-based approach with the matching pattern
            for idx, row in df.iterrows():
                try:
                    # Extract coordinates using pattern
                    x1 = float(row[matching_pattern['lon1']])
                    y1 = float(row[matching_pattern['lat1']])
                    x2 = float(row[matching_pattern['lon2']])
                    y2 = float(row[matching_pattern['lat2']])

                    # Validate coordinates
                    if (-180 <= x1 <= 180 and -90 <= y1 <= 90 and
                            -180 <= x2 <= 180 and -90 <= y2 <= 90):
                        geom = LineString([(x1, y1), (x2, y2)])
                        geometries.append(geom)
                        successful_count += 1

                        if idx < 3:  # Log first few
                            logger.info(f"Created TenneT geometry for row {idx}: ({x1}, {y1}) to ({x2}, {y2})")
                    else:
                        logger.warning(f"TenneT row {idx} has coordinates out of range: {x1}, {y1}, {x2}, {y2}")
                        geometries.append(None)
                except Exception as e:
                    logger.warning(f"Error creating TenneT geometry for row {idx}: {str(e)}")
                    geometries.append(None)

        logger.info(f"Successfully created {successful_count} TenneT geometries out of {len(df)} rows")

        if successful_count == 0:
            logger.error("No valid geometries created from TenneT data")
            # Dump the first few rows to help diagnose the issue
            if len(df) > 0:
                logger.info(f"First row data: {df.iloc[0].to_dict()}")
            return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame(df, geometry=geometries, crs="EPSG:4326")

        # Set ID column
        if 'id' not in gdf.columns:
            if 'NE_name' in gdf.columns:
                gdf['id'] = gdf['NE_name']
            else:
                gdf['id'] = [f"TenneT_{i}" for i in range(len(gdf))]

        # Set voltage column
        if 'v_nom' not in gdf.columns:
            if 'Voltage_level(kV)' in gdf.columns:
                gdf['v_nom'] = gdf['Voltage_level(kV)']
            else:
                gdf['v_nom'] = 380  # Default

        # Drop invalid geometries
        gdf = gdf[gdf.geometry.notna()]
        logger.info(f"Created TenneT GeoDataFrame with {len(gdf)} valid lines")

        # Add electrical parameters
        if 'Resistance_R(Ω)' in gdf.columns:
            gdf['r'] = gdf['Resistance_R(Ω)']
        else:
            gdf['r'] = 0

        if 'Reactance_X(Ω)' in gdf.columns:
            gdf['x'] = gdf['Reactance_X(Ω)']
        else:
            gdf['x'] = 0

        if 'Susceptance_B(μS)' in gdf.columns:
            gdf['b'] = gdf['Susceptance_B(μS)'] * 1e-6  # Convert μS to S
        else:
            gdf['b'] = 0

        # Add length
        if 'length' not in gdf.columns:
            if 'Length_(km)' in gdf.columns:
                gdf['length'] = gdf['Length_(km)']
            else:
                gdf['length'] = gdf.geometry.length * 111

        # Calculate per-km values
        gdf['r_per_km'] = gdf.apply(lambda row: safe_calc_per_km(row, 'r'), axis=1)
        gdf['x_per_km'] = gdf.apply(lambda row: safe_calc_per_km(row, 'x'), axis=1)
        gdf['b_per_km'] = gdf.apply(lambda row: safe_calc_per_km(row, 'b'), axis=1)

        # Add TSO column
        gdf['TSO'] = 'TenneT'

        return gdf

    except Exception as e:
        logger.error(f"Error loading TenneT data: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")


def create_tennet_geojson(input_file, output_file='data/processed/tennet_lines.geojson'):
    """Create a simplified TenneT GeoJSON file from the CSV data"""
    logger.info(f"Creating TenneT GeoJSON from {input_file}")

    try:
        # Load the TenneT CSV
        df = pd.read_csv(input_file)
        logger.info(f"Loaded {len(df)} rows from TenneT CSV")

        # Create a list to store features
        features = []

        # Group by the line ID (NE_name)
        if 'NE_name' in df.columns:
            groups = df.groupby('NE_name')
            for name, group in groups.items():
                if len(group) >= 2:
                    # Extract data for first substation
                    sub1 = group.iloc[0]
                    # Extract data for second substation
                    sub2 = group.iloc[1]

                    try:
                        # Get coordinates
                        lon1 = float(sub1['Longitude'])
                        lat1 = float(sub1['Latitude'])
                        lon2 = float(sub2['Longitude'])
                        lat2 = float(sub2['Latitude'])

                        # Create geometry
                        geom = LineString([(lon1, lat1), (lon2, lat2)])

                        # Add feature
                        feature = {
                            'type': 'Feature',
                            'properties': {
                                'id': name,
                                'v_nom': float(sub1.get('Voltage_level(kV)', 380)),
                                'r': float(sub1.get('Resistance_R(Ω)', 0)),
                                'x': float(sub1.get('Reactance_X(Ω)', 0)),
                                'b': float(sub1.get('Susceptance_B(μS)', 0)) * 1e-6,
                                'length': float(sub1.get('Length_(km)', 0)),
                                'TSO': 'TenneT'
                            },
                            'geometry': geom.__geo_interface__
                        }
                        features.append(feature)

                        if len(features) <= 3:  # Log first few
                            logger.info(f"Created TenneT feature for {name}: ({lon1}, {lat1}) to ({lon2}, {lat2})")
                    except Exception as e:
                        logger.warning(f"Error creating feature for {name}: {str(e)}")
        else:
            logger.warning("NE_name column not found in TenneT data")

            # Fallback approach - try to pair rows based on column positions
            logger.info("Trying fallback approach - pairing rows")
            for i in range(0, len(df), 2):
                if i + 1 < len(df):
                    try:
                        # Get rows
                        sub1 = df.iloc[i]
                        sub2 = df.iloc[i + 1]

                        # Get coordinates
                        lon1 = float(sub1['Longitude'])
                        lat1 = float(sub1['Latitude'])
                        lon2 = float(sub2['Longitude'])
                        lat2 = float(sub2['Latitude'])

                        # Create geometry
                        geom = LineString([(lon1, lat1), (lon2, lat2)])

                        # Add feature
                        feature = {
                            'type': 'Feature',
                            'properties': {
                                'id': f"TenneT_{i // 2}",
                                'v_nom': 380,
                                'r': 0,
                                'x': 0,
                                'b': 0,
                                'length': 0,
                                'TSO': 'TenneT'
                            },
                            'geometry': geom.__geo_interface__
                        }
                        features.append(feature)
                    except Exception as e:
                        logger.warning(f"Error creating feature for rows {i},{i + 1}: {str(e)}")

        # Create GeoJSON
        geojson = {
            'type': 'FeatureCollection',
            'features': features
        }

        # Save to file
        import json
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(geojson, f)

        logger.info(f"Created TenneT GeoJSON with {len(features)} features at {output_file}")
        return output_file

    except Exception as e:
        logger.error(f"Error creating TenneT GeoJSON: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None