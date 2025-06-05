import logging
import geopandas as gpd
from shapely.geometry import LineString, box
from shapely.geometry.multilinestring import MultiLineString

logger = logging.getLogger(__name__)


def clip_to_germany_strict(
        lines_gdf:   gpd.GeoDataFrame,
        germany_gdf: gpd.GeoDataFrame,
        min_length_km: float = 1.0,
        id_col: str = "id"
) -> gpd.GeoDataFrame:
    """
    Keep only the portions of each line that lie *inside* Germany
    (with a 500-m safety offset).  A segment shorter than `min_length_km`
    is *dropped* **only if** the *whole circuit* (all pieces that share
    `id_col`) is shorter than the threshold.  This preserves the little
    connector stubs that were causing visual gaps.

    Parameters
    ----------
    lines_gdf      : GeoDataFrame with LineString / MultiLineString geometries
    germany_gdf    : GeoDataFrame with the German boundary
    min_length_km  : Minimum length for an entire logical line to survive
                     (set 0 to disable length filtering completely)
    id_col         : Column that identifies one logical circuit
                     (defaults to "id"; if missing, the old behaviour is kept)
    """
    if lines_gdf is None or lines_gdf.empty:
        logger.warning("Empty GeoDataFrame passed to clip_to_germany_strict")
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    # --- copy & set CRS -----------------------------------------------------
    lines   = lines_gdf.copy()
    germany = germany_gdf.copy()

    if lines.crs is None:
        lines.crs = "EPSG:4326"
    if germany.crs is None:
        germany.crs = "EPSG:4326"

    metric_crs = "EPSG:3035"                       # metres
    lines   = lines.to_crs(metric_crs)
    germany = germany.to_crs(metric_crs)

    # --- shrink boundary by 500 m ------------------------------------------
    germany_boundary = germany.geometry.unary_union.buffer(-500)

    # --- intersection -------------------------------------------------------
    clipped_rows = []
    for idx, row in lines.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue

        if not geom.intersects(germany_boundary):
            continue

        try:
            part = geom.intersection(germany_boundary)
        except Exception as e:
            logger.error(f"Intersection failed for row {idx}: {e}")
            continue

        if part.is_empty:
            continue

        # keep only the LineString pieces (drop tiny polygons from overlaps)
        if part.geom_type == "LineString":
            pieces = [part]
        elif part.geom_type == "MultiLineString":
            pieces = list(part.geoms)
        elif hasattr(part, "geoms"):
            pieces = [g for g in part.geoms if g.geom_type == "LineString"]
        else:
            pieces = []

        if not pieces:
            continue

        # build new geometry – join pieces together if there are many
        new_geom = pieces[0] if len(pieces) == 1 else MultiLineString(pieces)

        new_row = row.copy()
        new_row.geometry = new_geom
        clipped_rows.append(new_row)

    if not clipped_rows:         # nothing survived
        return gpd.GeoDataFrame(geometry=[], crs=metric_crs).to_crs("EPSG:4326")

    clipped = gpd.GeoDataFrame(clipped_rows, crs=metric_crs)

    # --- length-based filtering  -------------------------------------------
    if min_length_km > 0:
        clipped["seg_len_km"] = clipped.geometry.length / 1000.0

        if id_col in clipped.columns:
            # total length of each logical circuit
            total_len = (
                clipped.groupby(id_col)["seg_len_km"]
                .transform("sum")
                .rename("total_len_km")
            )
            clipped["total_len_km"] = total_len
            keep_mask = (clipped["total_len_km"] >= min_length_km)
        else:
            # fall back to old behaviour (per segment)
            keep_mask = (clipped["seg_len_km"] >= min_length_km)

        before = len(clipped)
        clipped = clipped[keep_mask].drop(columns=["seg_len_km", "total_len_km"], errors="ignore")
        logger.info(
            f"Strict clipping: kept {len(clipped)} of {before} segments "
            f"(threshold {min_length_km} km, grouped by '{id_col}' "
            f"{'present' if id_col in lines_gdf.columns else 'absent'})."
        )
    else:
        logger.info("Strict clipping: length filter disabled – kept every intersecting segment.")

    # --- back to WGS-84 -----------------------------------------------------
    return clipped.to_crs("EPSG:4326")


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