import logging
import geopandas as gpd
from shapely.geometry import LineString, box

logger = logging.getLogger(__name__)


def clip_to_germany_strict(lines_gdf: gpd.GeoDataFrame,
                           germany_gdf: gpd.GeoDataFrame,
                           min_length_km: float = 1.0) -> gpd.GeoDataFrame:
    """
    Return only those parts of every source line that lie completely inside Germany.
    This function is very strict to guarantee that no segment extends beyond the borders.

    Parameters:
    - lines_gdf: GeoDataFrame containing line geometries to clip
    - germany_gdf: GeoDataFrame containing Germany boundary
    - min_length_km: Minimum length in kilometers to keep for clipped segments

    Returns:
    - GeoDataFrame with clipped lines inside Germany
    """
    if lines_gdf is None or lines_gdf.empty:
        logger.warning("Empty GeoDataFrame passed to clip_to_germany_strict")
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    # Make copies for safety
    lines = lines_gdf.copy()
    germany = germany_gdf.copy()

    # Make sure CRS is set
    if not lines.crs:
        lines.crs = "EPSG:4326"
    if not germany.crs:
        germany.crs = "EPSG:4326"

    common_crs = "EPSG:3035"  # Meter-based projection for Europe
    try:
        lines = lines.to_crs(common_crs)
        germany = germany.to_crs(common_crs)
    except Exception as e:
        logger.error(f"CRS conversion failed: {str(e)}. Trying with original CRSs.")

    # Optionally, fix source geometries if invalid
    def fix_geometry(geom):
        if not geom.is_valid:
            try:
                # Try to use make_valid if available
                try:
                    from shapely.validation import make_valid
                    geom = make_valid(geom)
                except ImportError:
                    # Fallback to buffer(0) if make_valid is not available
                    geom = geom.buffer(0)
            except Exception as e:
                logger.warning(f"Could not fix geometry: {e}")
        return geom

    lines['geometry'] = lines['geometry'].apply(fix_geometry)

    # Get Germany boundary as a single geometry
    germany_boundary = germany.geometry.unary_union

    # Apply a negative buffer of 500 m to shrink the boundary slightly
    buffer_distance = -500  # 500 m inward
    germany_boundary_shrunk = germany_boundary.buffer(buffer_distance)

    clipped_lines = []
    for idx, row in lines.iterrows():
        # Skip missing or empty geometries
        if row.geometry is None or row.geometry.is_empty:
            continue
        try:
            # Check if the geometry intersects the shrunk boundary
            if not row.geometry.intersects(germany_boundary_shrunk):
                continue
            clipped_geom = row.geometry.intersection(germany_boundary_shrunk)
            if clipped_geom.is_empty:
                continue

            # Depending on type, select only segments longer than min_length_km
            if clipped_geom.geom_type == 'LineString':
                if clipped_geom.length >= min_length_km * 1000:
                    new_row = row.copy()
                    new_row.geometry = clipped_geom
                    clipped_lines.append(new_row)
            elif clipped_geom.geom_type == 'MultiLineString':
                valid_parts = [part for part in clipped_geom.geoms if part.length >= min_length_km * 1000]
                if valid_parts:
                    from shapely.geometry import MultiLineString
                    new_geom = valid_parts[0] if len(valid_parts) == 1 else MultiLineString(valid_parts)
                    new_row = row.copy()
                    new_row.geometry = new_geom
                    clipped_lines.append(new_row)
            elif hasattr(clipped_geom, 'geoms'):
                line_parts = [part for part in clipped_geom.geoms if
                              part.geom_type == 'LineString' and part.length >= min_length_km * 1000]
                if line_parts:
                    from shapely.geometry import MultiLineString
                    new_geom = line_parts[0] if len(line_parts) == 1 else MultiLineString(line_parts)
                    new_row = row.copy()
                    new_row.geometry = new_geom
                    clipped_lines.append(new_row)
        except Exception as e:
            logger.error(f"Error processing line {idx}: {str(e)}")
            continue

    if not clipped_lines:
        return gpd.GeoDataFrame(geometry=[], crs=common_crs).to_crs("EPSG:4326")

    result_gdf = gpd.GeoDataFrame(clipped_lines, crs=common_crs)
    result_gdf = result_gdf[~result_gdf.geometry.is_empty & result_gdf.geometry.notna()]

    try:
        result_gdf = result_gdf.to_crs("EPSG:4326")
    except Exception as e:
        logger.error(f"Error converting clipped data back to EPSG:4326: {str(e)}")

    logger.info(f"Strict clipping: {len(result_gdf)} lines remain from original {len(lines_gdf)}")
    return result_gdf


def calculate_distance_km(lon1, lat1, lon2, lat2):
    """
    Calculate distance between two points in kilometers using the haversine formula.

    Parameters:
    - lon1, lat1: Longitude and latitude of first point in degrees
    - lon2, lat2: Longitude and latitude of second point in degrees

    Returns:
    - Distance in kilometers
    """
    from math import radians, sin, cos, sqrt, atan2

    # Earth's radius in kilometers
    R = 6371.0

    # Convert degrees to radians
    lon1_rad = radians(lon1)
    lat1_rad = radians(lat1)
    lon2_rad = radians(lon2)
    lat2_rad = radians(lat2)

    # Difference in coordinates
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad

    # Haversine formula
    a = sin(dlat / 2) ** 2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c

    return distance


def line_length_km(line_geometry):
    """
    Calculate the length of a line in kilometers.

    Parameters:
    - line_geometry: Shapely LineString or MultiLineString

    Returns:
    - Length in kilometers
    """
    if line_geometry is None or line_geometry.is_empty:
        return 0.0

    # For LineString, sum distances between consecutive points
    if line_geometry.geom_type == 'LineString':
        coords = list(line_geometry.coords)
        if len(coords) < 2:
            return 0.0

        length_km = 0.0
        for i in range(len(coords) - 1):
            lon1, lat1 = coords[i]
            lon2, lat2 = coords[i + 1]
            length_km += calculate_distance_km(lon1, lat1, lon2, lat2)
        return length_km

    # For MultiLineString, sum lengths of component lines
    elif line_geometry.geom_type == 'MultiLineString':
        return sum(line_length_km(line) for line in line_geometry.geoms)

    return 0.0


def simplify_geometry(geometry, tolerance=0.0001):
    """
    Simplify a geometry while preserving topology.

    Parameters:
    - geometry: Shapely geometry to simplify
    - tolerance: Tolerance for simplification in degrees

    Returns:
    - Simplified geometry
    """
    if geometry is None or geometry.is_empty:
        return geometry

    try:
        return geometry.simplify(tolerance, preserve_topology=True)
    except Exception as e:
        logger.error(f"Error simplifying geometry: {e}")
        return geometry


def create_bounding_box(gdf, buffer_km=50.0):
    """
    Create a bounding box around a GeoDataFrame with a buffer.

    Parameters:
    - gdf: GeoDataFrame
    - buffer_km: Buffer size in kilometers

    Returns:
    - Shapely Polygon representing the bounding box
    """
    if gdf is None or gdf.empty:
        return None

    # Get the total bounds of the GeoDataFrame
    minx, miny, maxx, maxy = gdf.total_bounds

    # Convert buffer from km to degrees (approximate)
    buffer_deg = buffer_km / 111.0

    # Create a bounding box with buffer
    return box(minx - buffer_deg, miny - buffer_deg,
               maxx + buffer_deg, maxy + buffer_deg)


def ensure_line_direction(line_geom, from_point, to_point):
    """
    Ensure a line geometry is oriented from one point towards another.
    Reverses geometry direction if needed.

    Parameters:
    - line_geom: Shapely LineString
    - from_point: (x, y) coordinates where the line should start
    - to_point: (x, y) coordinates where the line should end

    Returns:
    - LineString with correct orientation
    """
    if line_geom is None or line_geom.is_empty or line_geom.geom_type != 'LineString':
        return line_geom

    coords = list(line_geom.coords)
    if len(coords) < 2:
        return line_geom

    # Check start point
    start_point = coords[0]
    end_point = coords[-1]

    # Calculate which end is closer to the from_point
    start_dist = ((start_point[0] - from_point[0]) ** 2 +
                  (start_point[1] - from_point[1]) ** 2) ** 0.5
    end_dist = ((end_point[0] - from_point[0]) ** 2 +
                (end_point[1] - from_point[1]) ** 2) ** 0.5

    # If end is closer to from_point, reverse the line
    if end_dist < start_dist:
        return LineString(coords[::-1])

    return line_geom