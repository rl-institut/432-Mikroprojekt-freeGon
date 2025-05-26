import logging
from shapely.geometry import LineString, MultiLineString
logger = logging.getLogger(__name__)


def get_geometry_coords(geometry):
    """
    Safely get coordinates from any geometry type.

    Parameters:
    - geometry: Shapely geometry object (LineString or MultiLineString)

    Returns:
    - List of coordinate tuples
    """
    if geometry is None:
        return []

    # For LineString
    if geometry.geom_type == 'LineString':
        return list(geometry.coords)
    # For MultiLineString
    elif geometry.geom_type == 'MultiLineString':
        all_coords = []
        for line in geometry.geoms:
            all_coords.extend(list(line.coords))
        return all_coords
    return []


def calculate_line_direction(line):
    """
    Calculate normalized direction vector for a line.

    Parameters:
    - line: Shapely geometry object (LineString or MultiLineString)

    Returns:
    - Tuple (dx, dy) - normalized direction vector
    """
    if line is None:
        return 0, 0

    if isinstance(line, MultiLineString):
        # For MultiLineString, use the longest component
        if len(line.geoms) == 0:
            return 0, 0
        longest_line = max(line.geoms, key=lambda x: x.length)
        return calculate_line_direction(longest_line)

    if not isinstance(line, LineString):
        return 0, 0

    coords = list(line.coords)
    if len(coords) < 2:
        return 0, 0

    start, end = coords[0], coords[-1]
    dx, dy = end[0] - start[0], end[1] - start[1]
    length = (dx ** 2 + dy ** 2) ** 0.5
    if length > 0:
        return dx / length, dy / length
    return 0, 0


def direction_similarity(line1, line2):
    """
    Calculate direction similarity between lines (absolute value of dot product).

    Parameters:
    - line1: First Shapely geometry
    - line2: Second Shapely geometry

    Returns:
    - float: Similarity value between 0-1 where 1 means parallel lines
    """
    dir1_x, dir1_y = calculate_line_direction(line1)
    dir2_x, dir2_y = calculate_line_direction(line2)
    dot_product = dir1_x * dir2_x + dir1_y * dir2_y
    # Return absolute value - we're interested in parallel lines regardless of direction
    return abs(dot_product)


def create_geometry_hash(source_lines):
    """
    Create hash mapping for identifying parallel circuit geometries.

    Parameters:
    - source_lines: GeoDataFrame with source lines

    Returns:
    - dict: Mapping of geometry hash to list of line IDs
    """
    geom_hash_to_ids = {}

    for idx, row in source_lines.iterrows():
        if row.geometry is None or row.geometry.is_empty:
            continue

        # Create a simple hash of the geometry coordinates
        try:
            coords = get_geometry_coords(row.geometry)
            if not coords:  # Skip if no coordinates
                continue

            # Use start and end points for hashing to find parallel lines
            coord_str = f"{coords[0]}-{coords[-1]}"
            geom_hash = hash(coord_str)

            if geom_hash not in geom_hash_to_ids:
                geom_hash_to_ids[geom_hash] = []
            geom_hash_to_ids[geom_hash].append(str(row['id']))
        except Exception as e:
            logger.warning(f"Error creating hash for geometry: {e}")
            continue

    return geom_hash_to_ids