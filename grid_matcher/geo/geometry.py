from typing import Iterable, Tuple, List, Sequence, Optional, Dict, Union
from collections import Counter, defaultdict
import math
import re
import numpy as np

from shapely import wkt
from shapely.geometry import LineString, MultiLineString, Point, GeometryCollection
from shapely.geometry.base import BaseGeometry
from shapely.ops import linemerge, unary_union

# Type alias for shapely geometries
Geom = BaseGeometry

# Default buffer size for geometric operations
DEFAULT_BUFFER_M = 120.0

# Utility functions
def parse_linestring(wkt_str):
    """Return the exact geometry from WKT."""
    try:
        return wkt.loads(wkt_str)
    except Exception as exc:
        print(f"[parse_linestring] bad WKT → {exc}")
        return None

def calculate_length_meters(geometry):
    """Calculate length in meters for a geometry."""
    if geometry is None:
        return 0

    # Get centroid latitude for conversion
    centroid_lat = geometry.centroid.y
    # Approximate meters per degree at this latitude
    meters_per_deg = 111111 * np.cos(np.radians(abs(centroid_lat)))
    # Convert length from degrees to meters
    return float(geometry.length) * meters_per_deg

def get_start_point(geometry):
    """Safely extract the start point from a geometry."""
    if geometry is None:
        return None

    if isinstance(geometry, LineString):
        if len(geometry.coords) > 0:
            return Point(geometry.coords[0])
    elif isinstance(geometry, MultiLineString):
        if len(geometry.geoms) > 0 and len(geometry.geoms[0].coords) > 0:
            return Point(geometry.geoms[0].coords[0])

    return None

def get_end_point(geometry):
    """Safely extract the end point from a geometry."""
    if geometry is None:
        return None

    if isinstance(geometry, LineString):
        if len(geometry.coords) > 0:
            return Point(geometry.coords[-1])
    elif isinstance(geometry, MultiLineString):
        if len(geometry.geoms) > 0 and len(geometry.geoms[-1].coords) > 0:
            return Point(geometry.geoms[-1].coords[-1])

    return None


# ------------------------------ utilities ------------------------------
def _is_line(g: Geom) -> bool:
    """Check if geometry is a LineString or MultiLineString."""
    return isinstance(g, (LineString, MultiLineString))

def _iter_line_parts(g: Geom) -> Iterable[LineString]:
    """Yield only valid LineStrings from g (deeply), ignoring empties."""
    if g is None:
        return
    if isinstance(g, LineString):
        if not g.is_empty:
            yield g
    elif isinstance(g, MultiLineString):
        for ls in g.geoms:
            if ls is not None and not ls.is_empty:
                yield ls
    else:
        try:
            for sub in getattr(g, "geoms", []):
                yield from _iter_line_parts(sub)
        except Exception:
            return

def _snap_key(pt: Point, grid_m: float) -> Tuple[float, float]:
    """Round a point to a grid key."""
    return (round(pt.x / grid_m) * grid_m, round(pt.y / grid_m) * grid_m)

def _collect_row_endkeys(geom: Geom, grid_m: float) -> List[Tuple[float, float]]:
    """All snapped endpoint keys for a row geometry (deduplicated)."""
    keys: List[Tuple[float, float]] = []
    seen = set()
    for ls in _iter_line_parts(geom):
        p0 = Point(ls.coords[0])
        p1 = Point(ls.coords[-1])
        for p in (p0, p1):
            k = _snap_key(p, grid_m)
            if k not in seen:
                keys.append(k)
                seen.add(k)
    return keys

def _farthest_endpoint_pair(parts: Sequence[LineString], grid_m: float) -> Tuple[Optional[Point], Optional[Point]]:
    """Pick farthest pair among snapped endpoints, preferring degree-1 nodes."""
    if not parts:
        return None, None
    # degree by snapped key
    deg: Dict[Tuple[float, float], int] = defaultdict(int)
    keyed_pts: List[Tuple[Tuple[float, float], Point]] = []
    for ls in parts:
        p0 = Point(ls.coords[0]); p1 = Point(ls.coords[-1])
        k0 = _snap_key(p0, grid_m); k1 = _snap_key(p1, grid_m)
        deg[k0] += 1; deg[k1] += 1
        keyed_pts.append((k0, p0)); keyed_pts.append((k1, p1))
    leaf_keys = {k for k, c in deg.items() if c == 1}
    candidates = [p for (k, p) in keyed_pts if k in leaf_keys] or [p for (_, p) in keyed_pts]
    if len(candidates) < 2:
        return (candidates[0], candidates[0]) if candidates else (None, None)
    best_i, best_j = 0, 1
    best_d = candidates[0].distance(candidates[1])
    for i in range(len(candidates)):
        pi = candidates[i]
        for j in range(i + 1, len(candidates)):
            pj = candidates[j]
            d = pi.distance(pj)
            if d > best_d:
                best_d = d
                best_i, best_j = i, j
    return candidates[best_i], candidates[best_j]

def _safe_union(parts: Sequence[LineString]) -> Geom:
    """Union without raising, preserving multiple parts when needed."""
    if not parts:
        return None
    try:
        uu = unary_union(MultiLineString(parts))
        return uu
    except Exception:
        # fall back: keep as MultiLineString
        return MultiLineString(parts)

def _geom_len_m(geom: Geom) -> float:
    """Get the length of a geometry in meters."""
    return float(getattr(geom, "length", 0.0) or 0.0)

def _merge_lines_safely(geoms):
    """Merge a list of LineString/MultiLineString into a single (Multi)LineString safely."""
    parts = [g for g in geoms if g is not None and not g.is_empty]
    if not parts:
        return None

    u = unary_union(parts)
    # linemerge only accepts MultiLineString/GeometryCollection
    if isinstance(u, LineString):
        return u

    try:
        m = linemerge(u)
    except Exception:
        # Fall back: return union as-is if linemerge barfs
        m = u

    # Normalize empty collections
    if isinstance(m, GeometryCollection) and len(m) == 0:
        return None
    return m

def _merge_lines_safely_min(geoms):
    """Minimal version of safe line merging."""
    parts = [g for g in geoms if g is not None and not g.is_empty]
    if not parts:
        return None
    u = unary_union(parts)
    if isinstance(u, LineString):
        return u
    try:
        m = linemerge(u)
    except Exception:
        m = u
    if isinstance(m, GeometryCollection) and len(m) == 0:
        return None
    return m

def _as_int_or_none(v):
    """Convert to int or return None for NaN/None/invalid."""
    try:
        if v is None:
            return None
        if isinstance(v, bool):
            # Unlikely to be meaningful here, drop it
            return None
        if isinstance(v, float):
            if math.isnan(v):
                return None
            return int(round(v))
        return int(v)
    except Exception:
        return None

def _mode_int(values):
    """Mode of integer-like values; returns None if nothing valid."""
    ints = []
    for v in values:
        iv = _as_int_or_none(v)
        if iv is not None and iv > 0:
            ints.append(iv)
    if not ints:
        return None
    # Counter.most_common already breaks ties deterministically
    return Counter(ints).most_common(1)[0][0]

def _hausdorff_m(a, b):
    """Calculate Hausdorff distance in meters between two geometries."""
    try:
        return float(a.hausdorff_distance(b))
    except Exception:
        return float("inf")

def _buffered_iou(a, b, w=DEFAULT_BUFFER_M):
    """Calculate IoU (Intersection over Union) with buffers."""
    try:
        ab = a.buffer(w)
        bb = b.buffer(w)
        inter = ab.intersection(bb).area
        den = ab.union(bb).area
        if den <= 0:
            return 0.0
        return float(inter / den)
    except Exception:
        return 0.0

def _segment_fits_jao(seg, jao_line, buffer_m=DEFAULT_BUFFER_M):
    """Decide if a PyPSA segment lies along the JAO corridor."""
    try:
        dH = _hausdorff_m(seg, jao_line)  # in meters (same CRS)
        if dH <= buffer_m * 1.25:
            return True
        iou = _buffered_iou(seg, jao_line, w=buffer_m)
        return iou >= 0.20
    except Exception:
        return False

def extract_coordinates(geom_str):
    """Extract coordinates from potentially incomplete LINESTRING string."""
    try:
        if not isinstance(geom_str, str) or not geom_str.upper().startswith('LINESTRING'):
            return None

        coords_match = re.search(r'\((.*?)(\)|$)', geom_str)
        if not coords_match:
            return None

        coords_text = coords_match.group(1).strip()
        coords = []
        for coord_pair in re.findall(r'(-?\d+\.?\d*)\s+(-?\d+\.?\d*)', coords_text):
            try:
                x, y = float(coord_pair[0]), float(coord_pair[1])
                coords.append((x, y))
            except ValueError:
                continue

        if len(coords) < 2:
            return None

        return LineString(coords)
    except Exception as e:
        print(f"Error extracting coordinates: {str(e)}")
        return None

def _ensure_crs(gdf, epsg=4326):
    """Ensure a GeoDataFrame has the correct coordinate reference system."""
    try:
        if gdf.crs is None:
            gdf = gdf.set_crs(epsg, allow_override=True)
        return gdf
    except Exception:
        return gdf

def _to_meters(gdf):
    """Convert a GeoDataFrame to a meter-based projection."""
    try:
        return gdf.to_crs(3857)
    except Exception:
        return gdf

def _as_point(obj):
    """Return shapely Point from Point or (x, y); else None."""
    if obj is None:
        return None
    if hasattr(obj, "x") and hasattr(obj, "y"):
        return obj
    try:
        x, y = float(obj[0]), float(obj[1])
        return Point(x, y)
    except Exception:
        return None

def _ensure_linestring(geom):
    """Return a LineString for consistent endpoint operations."""
    if geom is None:
        return None
    if isinstance(geom, LineString):
        return geom
    try:
        # Try to get first part of MultiLineString
        if hasattr(geom, 'geoms'):
            parts = list(geom.geoms)
            return max(parts, key=lambda g: g.length) if parts else None
        # Last resort: try to build from coords
        return LineString(list(geom.coords))
    except Exception:
        return None

def get_geometry_coords(geom):
    """Get coordinates from a geometry safely."""
    if geom is None:
        return []

    if isinstance(geom, LineString):
        return list(geom.coords)
    elif isinstance(geom, MultiLineString):
        if len(geom.geoms) == 0:
            return []
        # Extract first and last coords from first and last parts
        first_coords = list(geom.geoms[0].coords)
        last_coords = list(geom.geoms[-1].coords)
        if not first_coords or not last_coords:
            return []
        return [first_coords[0], last_coords[-1]]
    elif hasattr(geom, 'coords'):
        return list(geom.coords)
    return []

def angle_between(g1, g2):
    """Calculate the angle between two line geometries."""
    try:
        coords1 = get_geometry_coords(g1)
        coords2 = get_geometry_coords(g2)

        if len(coords1) < 2 or len(coords2) < 2:
            return 90  # Default if not enough points

        # Get direction vectors
        v1 = np.array([coords1[-1][0] - coords1[0][0], coords1[-1][1] - coords1[0][1]])
        v2 = np.array([coords2[-1][0] - coords2[0][0], coords2[-1][1] - coords2[0][1]])

        # Normalize vectors
        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)

        if v1_norm == 0 or v2_norm == 0:
            return 90

        v1 = v1 / v1_norm
        v2 = v2 / v2_norm

        # Calculate angle using dot product
        dot_product = np.clip(np.dot(v1, v2), -1.0, 1.0)
        angle_rad = np.arccos(dot_product)
        angle_deg = math.degrees(angle_rad)

        return angle_deg
    except Exception as e:
        print(f"Error calculating angle: {e}")
        return 90  # Default value on error

def _meters_to_degrees(m, lat_deg=50.0):
    """Convert meters to degrees based on latitude."""
    # ~111.32 km per degree lon at equator; scale lon by cos(lat)
    if m is None:
        return 0.0
    lon_deg = m / (111320.0 * max(0.15, math.cos(math.radians(lat_deg))))
    lat_deg_per_m = 1.0 / 110574.0
    lat_deg_val = m * lat_deg_per_m
    # we use the *max* to avoid creating skinny buffers; OK for snapping/buffering
    return max(lon_deg, lat_deg_val)