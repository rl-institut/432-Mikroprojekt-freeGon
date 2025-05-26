import logging
import pandas as pd
import geopandas as gpd
from typing import List

logger = logging.getLogger(__name__)


def validate_required_columns(df: pd.DataFrame, required_columns: List[str]) -> bool:
    """
    Check if the DataFrame contains all required columns.

    Parameters:
    - df: DataFrame to validate
    - required_columns: List of column names that must be present

    Returns:
    - True if all columns are present, False otherwise
    """
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        logger.warning(f"Missing required columns: {missing_columns}")
        return False

    return True


def validate_geometry_column(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Validate and filter a GeoDataFrame to ensure all geometries are valid.

    Parameters:
    - gdf: GeoDataFrame to validate

    Returns:
    - Filtered GeoDataFrame with only valid geometries
    """
    if not isinstance(gdf, gpd.GeoDataFrame):
        logger.error("Input is not a GeoDataFrame")
        return gpd.GeoDataFrame(geometry=[])

    # Check for empty GeoDataFrame
    if gdf.empty:
        logger.warning("Empty GeoDataFrame")
        return gdf

    # Check if geometry column exists
    if 'geometry' not in gdf.columns:
        logger.error("No 'geometry' column found in GeoDataFrame")
        return gpd.GeoDataFrame(geometry=[])

    # Count initial geometries
    initial_count = len(gdf)

    # Check for None geometries
    none_geoms = gdf['geometry'].isna()
    if none_geoms.any():
        logger.warning(f"Found {none_geoms.sum()} None geometries")
        gdf = gdf[~none_geoms].copy()

    # Check for empty geometries
    empty_geoms = gdf['geometry'].apply(lambda g: g is not None and g.is_empty)
    if empty_geoms.any():
        logger.warning(f"Found {empty_geoms.sum()} empty geometries")
        gdf = gdf[~empty_geoms].copy()

    # Check for invalid geometries
    invalid_geoms = gdf['geometry'].apply(lambda g: g is not None and not g.is_valid)
    if invalid_geoms.any():
        logger.warning(f"Found {invalid_geoms.sum()} invalid geometries")

        # Try to fix invalid geometries
        def fix_geometry(geom):
            if geom is None or geom.is_empty:
                return None

            if not geom.is_valid:
                try:
                    try:
                        from shapely.validation import make_valid
                        return make_valid(geom)
                    except ImportError:
                        return geom.buffer(0)
                except Exception:
                    return None
            return geom

        gdf.loc[invalid_geoms, 'geometry'] = gdf.loc[invalid_geoms, 'geometry'].apply(fix_geometry)

        # Remove any that are still None or invalid
        gdf = gdf[gdf['geometry'].notna() &
                  gdf['geometry'].apply(lambda g: not g.is_empty and g.is_valid)].copy()

    # Count final geometries
    final_count = len(gdf)
    if final_count < initial_count:
        logger.warning(f"Removed {initial_count - final_count} invalid geometries")

    return gdf


def validate_line_geometries(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Validate and filter a GeoDataFrame to ensure all geometries are LineString or MultiLineString.

    Parameters:
    - gdf: GeoDataFrame to validate

    Returns:
    - Filtered GeoDataFrame with only line geometries
    """
    if gdf.empty:
        return gdf

    # First validate general geometry column
    gdf = validate_geometry_column(gdf)

    # Check for line geometries
    initial_count = len(gdf)
    is_line = gdf['geometry'].apply(
        lambda g: g is not None and (g.geom_type == 'LineString' or g.geom_type == 'MultiLineString')
    )

    if not is_line.all():
        non_lines = ~is_line
        logger.warning(f"Found {non_lines.sum()} non-line geometries (types: "
                       f"{set(gdf.loc[non_lines, 'geometry'].apply(lambda g: g.geom_type if g is not None else 'None'))})")
        gdf = gdf[is_line].copy()

    final_count = len(gdf)
    if final_count < initial_count:
        logger.warning(f"Removed {initial_count - final_count} non-line geometries")

    return gdf


def validate_numeric_columns(df: pd.DataFrame, numeric_columns: List[str]) -> pd.DataFrame:
    """
    Validate and convert columns to numeric types, with warnings for non-numeric values.

    Parameters:
    - df: DataFrame to validate
    - numeric_columns: List of column names that should be numeric

    Returns:
    - DataFrame with converted numeric columns
    """
    df = df.copy()

    for col in numeric_columns:
        if col in df.columns:
            # Count non-numeric values before conversion
            non_numeric_count = pd.to_numeric(df[col], errors='coerce').isna().sum()

            if non_numeric_count > 0:
                logger.warning(f"Column '{col}' contains {non_numeric_count} non-numeric values")

            # Convert to numeric, setting non-numeric values to NaN
            df[col] = pd.to_numeric(df[col], errors='coerce')

            # Replace NaN with 0
            na_count = df[col].isna().sum()
            if na_count > 0:
                logger.warning(f"Replacing {na_count} NaN values in column '{col}' with 0")
                df[col] = df[col].fillna(0)

    return df


def validate_coordinate_ranges(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Validate that all coordinates are within reasonable ranges for WGS84.

    Parameters:
    - gdf: GeoDataFrame to validate

    Returns:
    - Filtered GeoDataFrame with only valid coordinates
    """
    if gdf.empty:
        return gdf

    initial_count = len(gdf)

    def has_valid_coords(geom):
        if geom is None or geom.is_empty:
            return False

        def check_coords(coords):
            for x, y in coords:
                if not (-180 <= x <= 180 and -90 <= y <= 90):
                    return False
            return True

        if geom.geom_type == 'LineString':
            return check_coords(geom.coords)
        elif geom.geom_type == 'MultiLineString':
            return all(check_coords(line.coords) for line in geom.geoms)
        return False

    valid_coords = gdf['geometry'].apply(has_valid_coords)

    if not valid_coords.all():
        invalid_count = (~valid_coords).sum()
        logger.warning(f"Found {invalid_count} geometries with coordinates outside valid ranges")
        gdf = gdf[valid_coords].copy()

    final_count = len(gdf)
    if final_count < initial_count:
        logger.warning(f"Removed {initial_count - final_count} geometries with invalid coordinates")

    return gdf


def validate_line_lengths(gdf: gpd.GeoDataFrame, min_length_km: float = 0.1) -> gpd.GeoDataFrame:
    """
    Validate and filter lines to ensure they have a minimum length.

    Parameters:
    - gdf: GeoDataFrame to validate
    - min_length_km: Minimum length in kilometers

    Returns:
    - Filtered GeoDataFrame with only lines longer than min_length_km
    """
    if gdf.empty:
        return gdf

    initial_count = len(gdf)

    # Calculate length if not already present
    if 'length' not in gdf.columns:
        # Approximate conversion from degrees to km
        gdf['length'] = gdf.geometry.length * 111

    # Filter by length
    gdf = gdf[gdf['length'] >= min_length_km].copy()

    final_count = len(gdf)
    if final_count < initial_count:
        logger.info(f"Removed {initial_count - final_count} lines shorter than {min_length_km} km")

    return gdf


def validate_and_set_crs(gdf: gpd.GeoDataFrame, target_crs: str = "EPSG:4326") -> gpd.GeoDataFrame:
    """
    Validate CRS and convert to target CRS if necessary.

    Parameters:
    - gdf: GeoDataFrame to validate
    - target_crs: Target CRS

    Returns:
    - GeoDataFrame with the target CRS
    """
    if gdf.empty:
        # Create empty GeoDataFrame with correct CRS
        return gpd.GeoDataFrame(geometry=[], crs=target_crs)

    # Check if CRS is set
    if gdf.crs is None:
        logger.warning(f"CRS not set. Assuming {target_crs}")
        gdf.crs = target_crs

    # Convert to target CRS if different
    if gdf.crs != target_crs:
        logger.info(f"Converting from {gdf.crs} to {target_crs}")
        try:
            gdf = gdf.to_crs(target_crs)
        except Exception as e:
            logger.error(f"Error converting CRS: {e}")

    return gdf