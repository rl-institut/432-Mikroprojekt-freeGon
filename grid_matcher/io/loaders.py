"""Data loading utilities for Grid Matcher."""

import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import LineString, Point, MultiLineString
from shapely import wkt
from collections import defaultdict

from ..utils.logger import get_logger
from ..utils.helpers import parse_linestring, get_start_point, get_end_point, calculate_length_meters, \
    extract_coordinates

logger = get_logger(__name__)


def load_data(jao_path, pypsa_path, verbose=False):
    """
    Load JAO and PyPSA data from CSV files with enhanced preprocessing.

    Parameters
    ----------
    jao_path : str
        Path to JAO CSV file
    pypsa_path : str
        Path to PyPSA CSV file
    verbose : bool, optional
        Whether to print verbose output

    Returns
    -------
    tuple
        (jao_gdf, pypsa_gdf) - GeoDataFrames containing the loaded data
    """
    if verbose:
        logger.info(f"Loading JAO lines from {jao_path}")
    jao_df = pd.read_csv(jao_path)
    jao_geometry = jao_df['geometry'].apply(parse_linestring)
    jao_gdf = gpd.GeoDataFrame(jao_df, geometry=jao_geometry)

    if verbose:
        logger.info(f"Loading PyPSA lines from {pypsa_path}")
    pypsa_df = pd.read_csv(pypsa_path)
    pypsa_geometry = pypsa_df['geometry'].apply(parse_linestring)
    pypsa_gdf = gpd.GeoDataFrame(pypsa_df, geometry=pypsa_geometry)

    # Ensure IDs are strings
    jao_gdf['id'] = jao_gdf['id'].astype(str)
    pypsa_gdf['id'] = pypsa_gdf['id'].astype(str)

    # Add endpoints for matching
    jao_gdf['start_point'] = jao_gdf.geometry.apply(get_start_point)
    jao_gdf['end_point'] = jao_gdf.geometry.apply(get_end_point)

    pypsa_gdf['start_point'] = pypsa_gdf.geometry.apply(get_start_point)
    pypsa_gdf['end_point'] = pypsa_gdf.geometry.apply(get_end_point)

    # Calculate lengths in km
    jao_gdf['length_km'] = jao_gdf.geometry.apply(
        lambda g: calculate_length_meters(g) / 1000 if g is not None else 0
    )

    pypsa_gdf['length_km'] = pypsa_gdf.geometry.apply(
        lambda g: calculate_length_meters(g) / 1000 if g is not None else 0
    )

    # Ensure circuits column exists (default to 1)
    if 'circuits' not in pypsa_gdf.columns:
        pypsa_gdf['circuits'] = 1
    else:
        pypsa_gdf['circuits'] = pd.to_numeric(pypsa_gdf['circuits'], errors='coerce').fillna(1)

    pypsa_gdf['circuits'] = pypsa_gdf['circuits'].apply(lambda x: max(1, int(x)))

    # Standardize voltage columns
    if 'v_nom' not in jao_gdf.columns and 'voltage' in jao_gdf.columns:
        jao_gdf['v_nom'] = jao_gdf['voltage']

    if 'voltage' not in pypsa_gdf.columns and 'v_nom' in pypsa_gdf.columns:
        pypsa_gdf['voltage'] = pypsa_gdf['v_nom']

    # Ensure electrical parameters are available for matching
    for df in [jao_gdf, pypsa_gdf]:
        # Standardize parameter names
        if 'r_ohm_per_km' in df.columns and 'r_per_km' not in df.columns:
            df['r_per_km'] = df['r_ohm_per_km']
        if 'x_ohm_per_km' in df.columns and 'x_per_km' not in df.columns:
            df['x_per_km'] = df['x_ohm_per_km']
        if 'b_mho_per_km' in df.columns and 'b_per_km' not in df.columns:
            df['b_per_km'] = df['b_mho_per_km']

    # Identify duplicate geometries in JAO data (parallel circuits)
    jao_geometry_groups = defaultdict(list)
    for idx, row in jao_gdf.iterrows():
        if row.geometry is not None:
            wkt_str = row.geometry.wkt
            jao_geometry_groups[wkt_str].append(row['id'])

    # Mark parallel circuits
    jao_gdf['is_parallel_circuit'] = False
    jao_gdf['parallel_group'] = None

    for wkt_str, ids in jao_geometry_groups.items():
        if len(ids) > 1:
            group_id = '_'.join(sorted(ids))
            jao_gdf.loc[jao_gdf['id'].isin(ids), 'is_parallel_circuit'] = True
            jao_gdf.loc[jao_gdf['id'].isin(ids), 'parallel_group'] = group_id

    if verbose:
        logger.info(f"Loaded {len(jao_gdf)} JAO lines and {len(pypsa_gdf)} PyPSA lines")
        logger.info(f"Found {sum(jao_gdf['is_parallel_circuit'])} JAO lines that are part of parallel circuits")

    return jao_gdf, pypsa_gdf


def load_dc_links(filepath, verbose=False):
    """
    Load DC links data from CSV file with robust error handling.

    Parameters
    ----------
    filepath : str
        Path to DC links CSV file
    verbose : bool, optional
        Whether to print verbose output

    Returns
    -------
    GeoDataFrame
        DC links data with geometries
    """
    if verbose:
        logger.info(f"Loading DC links from {filepath}")

    try:
        # Try different CSV parsing approaches for robustness
        try:
            # Standard approach
            df = pd.read_csv(filepath)
        except Exception as e:
            if verbose:
                logger.warning(f"Standard CSV parsing failed: {e}, trying Python engine")
            try:
                # Try with Python engine for more flexibility
                df = pd.read_csv(filepath, engine='python')
            except Exception as e2:
                if verbose:
                    logger.warning(f"Python engine failed too: {e2}, trying with error handling")
                # For pandas <1.3 vs >=1.3 compatibility
                if pd.__version__ < '1.3':
                    df = pd.read_csv(filepath, error_bad_lines=False)
                else:
                    df = pd.read_csv(filepath, on_bad_lines='skip')

        # Process the geometry column
        if 'geometry' in df.columns:
            # Clean and normalize geometry strings
            df['geometry'] = df['geometry'].astype(str).str.replace("'", "").str.strip()

            # Filter for valid LINESTRING or MULTILINESTRING WKT
            valid_mask = df['geometry'].apply(
                lambda x: x.upper().startswith('LINESTRING') or x.upper().startswith('MULTILINESTRING')
            )

            if valid_mask.sum() > 0:
                valid_df = df[valid_mask].copy()

                # Safe WKT parsing
                def safe_parse_wkt(wkt_str):
                    try:
                        return wkt.loads(wkt_str)
                    except Exception:
                        return None

                valid_df['geometry'] = valid_df['geometry'].apply(safe_parse_wkt)
                valid_df = valid_df[valid_df['geometry'].notnull()]

                if len(valid_df) > 0:
                    # Add DC specific fields
                    valid_df['is_dc'] = True
                    valid_df['v_nom'] = "DC"
                    valid_df['voltage'] = "DC"

                    # Create GeoDataFrame
                    gdf = gpd.GeoDataFrame(valid_df, geometry='geometry', crs="EPSG:4326")

                    if verbose:
                        logger.info(f"Successfully loaded {len(gdf)} DC links")

                    return gdf

        if verbose:
            logger.warning("No valid DC geometries found in file")
        return None

    except Exception as e:
        if verbose:
            logger.error(f"Error loading DC links: {e}")
        return None


def load_110kv_data(filepath, verbose=False):
    """
    Load 110kV grid data from CSV file.

    Parameters
    ----------
    filepath : str
        Path to 110kV grid data CSV file
    verbose : bool, optional
        Whether to print verbose output

    Returns
    -------
    GeoDataFrame
        110kV grid data with geometries
    """
    if verbose:
        logger.info(f"Loading 110kV data from {filepath}")

    try:
        # Read the CSV file
        df = pd.read_csv(filepath, low_memory=False)

        # Filter for valid WKT geometries
        valid_starts = ['POINT', 'LINESTRING', 'POLYGON', 'MULTI']
        valid_mask = df['geometry'].apply(
            lambda x: isinstance(x, str) and any(str(x).strip().upper().startswith(prefix) for prefix in valid_starts)
        )

        valid_df = df[valid_mask].copy()

        if verbose:
            logger.info(f"Filtered from {len(df)} to {len(valid_df)} rows with valid geometries")

        # Convert geometries to shapely objects
        valid_df['geometry'] = valid_df['geometry'].apply(lambda x: wkt.loads(x) if isinstance(x, str) else x)

        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame(valid_df, geometry='geometry', crs="EPSG:4326")

        # Ensure correct data types
        gdf['v_nom'] = 110
        gdf['voltage'] = 110  # Support both naming conventions

        # Add additional fields that may be needed
        if 'num_parallel' in gdf.columns and 'circuits' not in gdf.columns:
            gdf['circuits'] = gdf['num_parallel'].astype(int)
        elif 'circuits' not in gdf.columns:
            gdf['circuits'] = 1

        if 'line_id' not in gdf.columns and 'id' in gdf.columns:
            gdf['line_id'] = gdf['id']

        if verbose:
            logger.info(f"Successfully loaded {len(gdf)} 110kV lines")

        return gdf

    except Exception as e:
        if verbose:
            logger.error(f"Error loading 110kV data: {e}")
        return None


def load_dc_data(filepath, verbose=False):
    """
    Load DC links with custom geometry parser.

    Parameters
    ----------
    filepath : str
        Path to DC links data file
    verbose : bool, optional
        Whether to print verbose output

    Returns
    -------
    GeoDataFrame
        DC links data with geometries
    """
    if verbose:
        logger.info(f"Loading DC data from {filepath}")

    try:
        # Determine delimiter by inspecting the file
        with open(filepath, 'r') as f:
            first_line = f.readline().strip()
            delimiter = '\t' if '\t' in first_line else ','
            if verbose:
                logger.info(f"Using {delimiter} delimiter")

        # Load data
        df = pd.read_csv(filepath, delimiter=delimiter, low_memory=False)

        if verbose:
            logger.info(f"Loaded {len(df)} rows with {len(df.columns)} columns")

        # Identify geometry columns
        geometry_columns = [col for col in df.columns if 'geometry' in str(col).lower()]

        if verbose:
            logger.info(f"Found {len(geometry_columns)} geometry-related columns")

        # Handle geometry
        if 'geometry' not in df.columns and len(geometry_columns) == 0:
            if verbose:
                logger.warning("No geometry column found in data")
            return None

        # Combine split geometry columns if needed
        if len(geometry_columns) > 1:
            if verbose:
                logger.info("Combining split geometry columns...")
            df['combined_geometry'] = ''
            for col in geometry_columns:
                df['combined_geometry'] = df['combined_geometry'] + df[col].fillna('').astype(str)
            geometry_col = 'combined_geometry'
        else:
            geometry_col = 'geometry'

        # Parse geometries
        if verbose:
            logger.info("Parsing geometries using custom parser...")
        geometries = []
        valid_indices = []

        for idx, row in df.iterrows():
            geom_str = str(row[geometry_col]).replace("'", "").strip()
            geom = extract_coordinates(geom_str)

            if geom is not None:
                geometries.append(geom)
                valid_indices.append(idx)

        # Create GeoDataFrame with valid geometries
        if verbose:
            logger.info(f"Found {len(valid_indices)} valid geometries out of {len(df)} rows")

        if len(valid_indices) == 0:
            if verbose:
                logger.warning("No valid geometries could be parsed")
            return None

        # Create GeoDataFrame
        valid_df = df.loc[valid_indices].copy()
        valid_gdf = gpd.GeoDataFrame(
            valid_df,
            geometry=geometries,
            crs="EPSG:4326"
        )

        # Add DC-specific fields if not present
        if 'is_dc' not in valid_gdf.columns:
            valid_gdf['is_dc'] = True

        if 'voltage' not in valid_gdf.columns:
            valid_gdf['voltage'] = "DC"

        if verbose:
            logger.info(f"Successfully created GeoDataFrame with {len(valid_gdf)} DC links")

        return valid_gdf

    except Exception as e:
        if verbose:
            logger.error(f"Error loading DC data: {str(e)}")
        return None