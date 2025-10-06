import math
from collections import Counter, defaultdict
from email.mime import base
from typing import Iterable, Tuple, List, Sequence, Optional, Dict

import numpy as np
from prompt_toolkit.data_structures import Point
from shapely import wkt
from shapely.geometry.linestring import LineString
from shapely.geometry.multilinestring import MultiLineString
from shapely.lib import unary_union


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

def _num(val):
    """
    Safely convert a value to float.
    - Returns None if conversion isn't possible or val is NaN/None/''.
    - Handles strings with thousands separators and European decimals.
    """
    import math

    if val is None:
        return None

    # Fast-path for numerics
    if isinstance(val, (int, float)):
        if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
            return None
        return float(val)

    # String cleanup
    try:
        s = str(val).strip()
        if s == "":
            return None
        # normalize thousands/decimal separators
        # if there's a comma but no dot, treat comma as decimal
        if "," in s and "." not in s:
            s = s.replace(" ", "").replace("\u00a0", "").replace(",", ".")
        else:
            # otherwise, remove commas as thousands separators
            s = s.replace(",", "")
            s = s.replace(" ", "").replace("\u00a0", "")
        x = float(s)
        if math.isnan(x) or math.isinf(x):
            return None
        return x
    except Exception:
        return None


def _to_km(length_value):
    """
    Convert a length to kilometers using a simple heuristic:
    - If value >= 1000, assume meters and divide by 1000.
    - Otherwise assume kilometers already.
    Returns None if input is missing or non-numeric.
    """
    v = _num(length_value)
    if v is None:
        return None
    return v / 1000.0 if abs(v) >= 1000.0 else v



def _listify_ids(v):
        if not v:
            return []
        if isinstance(v, str):
            return [t.strip() for t in v.replace(";", ",").split(",") if t.strip()]
        if isinstance(v, (list, tuple, set)):
            return [str(x) for x in v if str(x)]
        return [str(v)]

def _safe_int(val):
    """Safely convert a value to int."""
    try:
        if val is None:
            return 0
        return int(float(val))
    except (ValueError, TypeError):
        return 0


def _merge_lines_safely(geoms):
    """Merge a list of LineString/MultiLineString into a single (Multi)LineString safely."""
    from shapely.geometry import LineString, MultiLineString, GeometryCollection
    from shapely.ops import linemerge, unary_union

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
    from shapely.geometry import LineString, MultiLineString, GeometryCollection
    from shapely.ops import linemerge, unary_union
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

from shapely.geometry.base import BaseGeometry
Geom = BaseGeometry

def _is_line(g: Geom) -> bool:
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
    return float(getattr(geom, "length", 0.0) or 0.0)


def extract_coordinates(geom_str):
    """Extract coordinates from potentially incomplete LINESTRING string"""
    try:
        if not isinstance(geom_str, str) or not geom_str.upper().startswith('LINESTRING'):
            return None

        import re

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

