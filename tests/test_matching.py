import pandas as pd
import networkx as nx
import folium
from folium.plugins import MarkerCluster
from shapely.geometry import LineString, Point
import geopandas as gpd
import numpy as np
from pathlib import Path
import os
import json
import traceback  # Add this at the top with your other imports


# Import the pypsa_integration module
from pypsa_integration import integrate_pypsa_lines

# Define file paths
data_dir = Path('/home/mohsen/PycharmProjects/freeGon/grid-matching-tool/data/clipped')
jao_lines_path = data_dir / 'jao-lines-germany.csv'
network_lines_path = data_dir / 'network-lines-germany.csv'
output_dir = Path('/home/mohsen/PycharmProjects/freeGon/grid-matching-tool/tests')

# Function to parse linestring from CSV
from shapely import wkt

def parse_linestring(wkt_str: str):
    """Return the exact geometry written in WKT (LINESTRING or MULTILINESTRING)."""
    try:
        return wkt.loads(wkt_str)
    except Exception as exc:
        print(f"[parse_linestring] bad WKT → {exc}  |  {wkt_str[:80]}…")
        return None

# --- common imports and helpers (top of file) ---
import numpy as np
import networkx as nx

def _safe_int(v, default=0):
    try:
        return int(float(v))
    except Exception:
        return default

def _geom_main_linestring(geom):
    from shapely.geometry import MultiLineString
    if geom.geom_type == "MultiLineString":
        return max(geom.geoms, key=lambda g: g.length)
    return geom

# Load JAO lines data
def load_jao_lines():
    jao_df = pd.read_csv(jao_lines_path)
    # Create GeoDataFrame
    geometry = jao_df['geometry'].apply(parse_linestring)
    jao_gdf = gpd.GeoDataFrame(jao_df, geometry=geometry)
    jao_gdf = jao_gdf.explode(index_parts=False, ignore_index=True)

    # Extract start and end points
    jao_gdf['start_point'] = jao_gdf.geometry.apply(lambda x: Point(x.coords[0]))
    jao_gdf['end_point'] = jao_gdf.geometry.apply(lambda x: Point(x.coords[-1]))

    return jao_gdf

# ------------------------------------------------------------------
# helper -- guarantee an integer `circuits` column
# ------------------------------------------------------------------
def _ensure_circuits_col(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add / fix a `circuits` column:

    1. If `num_parallel` exists, copy it (default 1).
    2. Otherwise default to 1.
    """
    if 'circuits' not in df.columns:
        df['circuits'] = (
            df['num_parallel'].fillna(1).astype(float).round().astype(int)
            if 'num_parallel' in df.columns
            else 1
        )
    else:
        mask = df['circuits'].isna()
        if mask.any() and 'num_parallel' in df.columns:
            df.loc[mask, 'circuits'] = df.loc[mask, 'num_parallel'].fillna(1)
        df['circuits'] = df['circuits'].fillna(1).astype(float).round().astype(int)
    return df


def load_network_lines():
    network_df = pd.read_csv(network_lines_path)

    # Exclude 110 kV
    network_df = network_df[network_df['v_nom'] != 110]

    # GeoDataFrame
    geometry = network_df['geometry'].apply(parse_linestring)
    network_gdf = gpd.GeoDataFrame(network_df, geometry=geometry)
    network_gdf = network_gdf.explode(index_parts=False, ignore_index=True)

    # endpoints
    network_gdf['start_point'] = network_gdf.geometry.apply(lambda g: Point(g.coords[0]))
    network_gdf['end_point']   = network_gdf.geometry.apply(lambda g: Point(g.coords[-1]))

    # ---- NEW: make sure every row knows how many circuits ----
    network_gdf = _ensure_circuits_col(network_gdf)

    # tidy index
    return network_gdf.reset_index(drop=True)


def compute_network_per_circuit_params(network_gdf):
    """
    Adds r_km_pc, x_km_pc, b_km_pc columns to network_gdf (per-circuit, per-km).
    Uses length + r/x/b or r_per_km/x_per_km/b_per_km if present.
    If the line is a corridor (multiple circuits), de-aggregates:
      R_km_pc = R_km_corridor * n
      X_km_pc = X_km_corridor * n
      B_km_pc = B_km_corridor / n
    """
    import pandas as pd
    df = network_gdf.copy()

    # circuits detection
    npar = pd.to_numeric(df.get('circuits', pd.Series(index=df.index, data=None)), errors='coerce')
    if 'num_parallel' in df.columns:
        npar = npar.fillna(pd.to_numeric(df['num_parallel'], errors='coerce'))
    npar = npar.fillna(1).clip(lower=1).round().astype(int)

    # length
    L = pd.to_numeric(df.get('length', pd.Series(index=df.index, data=None)), errors='coerce')
    # if length missing, derive from geometry length (meters -> km approx)
    missing_len = L.isna()
    if 'geometry' in df.columns and missing_len.any():
        import numpy as np
        lat = df.loc[missing_len, 'geometry'].centroid.y.astype(float)
        m_per_deg = 111111*np.cos(np.radians(lat.abs()))
        L.loc[missing_len] = (df.loc[missing_len, 'geometry'].length * m_per_deg) / 1000.0
    L = L.fillna(0.0)

    # get corridor per-km if present, else compute from totals
    def safe_num(s):
        return pd.to_numeric(s, errors='coerce')

    r_km = safe_num(df.get('r_per_km', df.get('R_per_km')))
    x_km = safe_num(df.get('x_per_km', df.get('X_per_km')))
    b_km = safe_num(df.get('b_per_km', df.get('B_per_km')))

    r_tot = safe_num(df.get('r', df.get('R')))
    x_tot = safe_num(df.get('x', df.get('X')))
    b_tot = safe_num(df.get('b', df.get('B')))

    # derive missing per-km from totals
    r_km = r_km.where(r_km.notna(), (r_tot / L).where(L > 0))
    x_km = x_km.where(x_km.notna(), (x_tot / L).where(L > 0))
    b_km = b_km.where(b_km.notna(), (b_tot / L).where(L > 0))

    # corridor -> per-circuit
    df['r_km_pc'] = r_km * npar
    df['x_km_pc'] = x_km * npar
    df['b_km_pc'] = b_km / npar

    return df

# ── add these helpers near the top of the file ─────────────────────────
from shapely.geometry import Point
def _project_point_onto_lines(pt, network_gdf, max_dist_m=2000):
    """Return (proj_point, parent_line_idx, parent_pos) or (None,…)."""
    import numpy as np
    best = (None, None, None, float('inf'))
    lat = pt.y; m_per_deg = 111111*np.cos(np.radians(abs(lat)))
    tol = max_dist_m / m_per_deg
    for idx, row in network_gdf.iterrows():
        g = row.geometry
        # allow multiline
        if g.geom_type == "MultiLineString":
            for seg in g.geoms:
                p = seg.interpolate(seg.project(pt))
                d = p.distance(pt)
                if d < best[3]:
                    best = (p, idx, 'proj', d)
        else:
            p = g.interpolate(g.project(pt))
            d = p.distance(pt)
            if d < best[3]:
                best = (p, idx, 'proj', d)
    if best[0] is None or best[3] > tol:
        return None
    return best[:3]         # drop distance
# ───────────────────────────────────────────────────────────────────────


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------
from shapely.geometry import Point, LineString, MultiLineString
from rtree import index
import math


from shapely.strtree import STRtree
from shapely import buffer as shp_buffer
import numpy as np

def _meters_per_degree(lat):
    import math
    return 111_111 * math.cos(math.radians(abs(lat)))

def candidate_pairs_by_buffer(geoms_a, geoms_b, buffer_m=600):
    """
    Return (ia, ib) index arrays for geometry pairs whose buffered envelopes overlap.
    geoms_a / geoms_b: numpy arrays (or lists) of shapely geometries (same CRS).
    """
    geoms_a = np.asarray(geoms_a, dtype=object)
    geoms_b = np.asarray(geoms_b, dtype=object)

    # variable-degree buffer per geometry (approximate; fast)
    def wdeg(g):
        lat = g.centroid.y
        return buffer_m / _meters_per_degree(lat)

    bufA = [g.buffer(wdeg(g)) for g in geoms_a]
    bufB = [g.buffer(wdeg(g)) for g in geoms_b]

    treeA = STRtree(bufA)
    # query_bulk returns 2xN array of indices (in tree order vs. query order)
    ia, ib = treeA.query_bulk(bufB)  # ia are indices into bufA, ib into bufB (order: B queries A)

    # translate to (A_index, B_index) pairs
    return ia, ib



def _snap_point_to_nearest_line(pt: Point,
                                network_gdf,
                                max_snap_m: float = 3_000):
    """
    Orthogonally project *pt* on every network line, keep the shortest.
    Returns (distance_m, network_idx, 'start' | 'end') **or** None.
    """
    m_per_deg = _meters_per_degree(pt.y)

    best = (max_snap_m + 1, None, None)             # distance, idx, tag
    for idx, row in network_gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue

        # 1) orthogonal projection
        proj   = geom.interpolate(geom.project(pt))
        d_m    = proj.distance(pt) * m_per_deg
        if d_m >= best[0]:
            continue

        # 2) decide whether projected point is nearer to first or last
        if geom.geom_type == "MultiLineString":
            first, last = geom.geoms[0].coords[0], geom.geoms[-1].coords[-1]
        else:  # LineString
            first, last = geom.coords[0], geom.coords[-1]

        pos = 'start' if Point(first).distance(proj) <= Point(last).distance(proj) else 'end'
        best = (d_m, idx, pos)

    return best if best[1] is not None else None


# ---------------------------------------------------------------------------
#  Main routine
# ---------------------------------------------------------------------------
from rtree import index                       # spatial index
from shapely.geometry import Point
import numpy as np
from scipy.cluster import hierarchy as hcl




def find_nearest_points(
        jao_gdf,
        network_gdf,
        *,
        max_alternatives: int = 5,
        distance_threshold_meters: float = 1_500,
        substation_cluster_radius_meters: float = 300,
        debug_lines=None
):
    """
    Locate the nearest network endpoints for every JAO line endpoint.
    Uses spatial clustering to identify substation yards and improve matching.

    Parameters
    ----------
    jao_gdf, network_gdf : GeoDataFrames (must contain geometry, 'id', 'v_nom')
    max_alternatives     : how many fallback endpoint pairs to keep
    distance_threshold_meters : maximum distance to consider an endpoint match
    substation_cluster_radius_meters : radius to group endpoints as a substation
    debug_lines          : iterable of JAO-IDs that get verbose prints

    Returns
    -------
    dict : key = jao_gdf index, value = {
        'start_nearest'           : (net_idx, 'start'|'end') | None,
        'end_nearest'             : (net_idx, 'start'|'end') | None,
        'start_alternatives'      : list of (net_idx, pos),
        'end_alternatives'        : list of (net_idx, pos),
        'start_distance_meters'   : float,
        'end_distance_meters'     : float
    }
    """
    import numpy as np
    from scipy.spatial import cKDTree
    from sklearn.cluster import DBSCAN
    from shapely.geometry import Point

    # ------------------------------------------------------------------ #
    # 0)  Helper functions                                               #
    # ------------------------------------------------------------------ #
    def _meters_per_degree(lat):
        """Very rough WGS84 conversion; good enough for < few km."""
        import math
        return 111_111 * math.cos(math.radians(abs(lat)))

    def _voltage_match(j_v, n_v):
        """Check if voltages match, with special handling for 380/400 kV."""
        if j_v == n_v:
            return True
        # 380 kV and 400 kV are considered equivalent
        if j_v in (380, 400) and n_v in (380, 400):
            return True
        # Allow small voltage variations (e.g., 220 kV vs 225 kV)
        if abs(j_v - n_v) <= 10:
            return True
        return False

    def _point_to_array(point):
        """Convert shapely Point to array."""
        return np.array([point.x, point.y])

    # ------------------------------------------------------------------ #
    # 1)  Build catalog of all network endpoints                         #
    # ------------------------------------------------------------------ #
    network_endpoints = []  # (row_idx, 'start'/'end', Point, net_id, v_nom)
    endpoint_coords = []  # array for clustering algorithms

    for n_idx, n_row in network_gdf.iterrows():
        geom = n_row.geometry
        if geom is None or geom.is_empty:
            continue

        def _add_endpoint(pt, tag):
            point = Point(pt)
            network_endpoints.append(
                (n_idx, tag, point, n_row['id'], int(n_row['v_nom']))
            )
            endpoint_coords.append(_point_to_array(point))

        if geom.geom_type == "LineString":
            _add_endpoint(geom.coords[0], 'start')
            _add_endpoint(geom.coords[-1], 'end')
        elif geom.geom_type == "MultiLineString":
            _add_endpoint(geom.geoms[0].coords[0], 'start')
            _add_endpoint(geom.geoms[-1].coords[-1], 'end')

    if not network_endpoints:
        print("Warning: No network endpoints found!")
        return {}

    endpoint_coords = np.array(endpoint_coords)

    # ------------------------------------------------------------------ #
    # 2)  Cluster endpoints to identify substations                      #
    # ------------------------------------------------------------------ #
    print("Clustering network endpoints to identify substations...")

    # Calculate DBSCAN epsilon in degrees based on average latitude
    avg_lat = np.mean(endpoint_coords[:, 1])
    meters_per_deg = _meters_per_degree(avg_lat)
    epsilon = substation_cluster_radius_meters / meters_per_deg

    # Perform clustering using DBSCAN
    clustering = DBSCAN(eps=epsilon, min_samples=2).fit(endpoint_coords)
    cluster_labels = clustering.labels_

    # Group endpoints by cluster
    clusters = {}
    for i, label in enumerate(cluster_labels):
        if label >= 0:  # -1 means noise point (not in any cluster)
            clusters.setdefault(label, []).append(network_endpoints[i])

    # Calculate cluster centroids for debugging
    cluster_centroids = {}
    for label, points in clusters.items():
        coords = np.array([_point_to_array(p[2]) for p in points])
        centroid = coords.mean(axis=0)
        cluster_centroids[label] = (centroid, len(points))

    print(f"Found {len(clusters)} substation clusters containing {sum(len(c) for c in clusters.values())} endpoints")
    print(f"Largest cluster has {max(len(c) for c in clusters.values())} endpoints")

    # Build spatial index for fast lookup
    from rtree import index
    idx = index.Index()
    for i, endpoint in enumerate(network_endpoints):
        point = endpoint[2]
        idx.insert(i, (point.x, point.y, point.x, point.y))

    # ------------------------------------------------------------------ #
    # 3)  Find nearest points for each JAO line                          #
    # ------------------------------------------------------------------ #
    nearest_points = {}

    for j_idx, j_row in jao_gdf.iterrows():
        jao_id = str(j_row['id'])
        jao_v = int(j_row['v_nom'])
        geom = j_row.geometry
        dbg = debug_lines and jao_id in debug_lines

        if dbg:
            print(f"\nProcessing JAO {jao_id} ({jao_v} kV)")

        # -- extract endpoints ----------------------------------------- #
        if geom.geom_type == 'LineString':
            start_pt, end_pt = Point(geom.coords[0]), Point(geom.coords[-1])
        elif geom.geom_type == 'MultiLineString':
            start_pt = Point(geom.geoms[0].coords[0])
            end_pt = Point(geom.geoms[-1].coords[-1])
        else:
            nearest_points[j_idx] = {
                'start_nearest': None, 'end_nearest': None,
                'start_alternatives': [], 'end_alternatives': [],
                'start_distance_meters': float('inf'),
                'end_distance_meters': float('inf')
            }
            continue

        m_per_deg = _meters_per_degree((start_pt.y + end_pt.y) / 2.0)

        # -- find matches for both start and end points ---------------- #
        def _find_matches_for_point(pt, is_start):
            point_type = 'start' if is_start else 'end'
            if dbg:
                print(f"Finding matches for {point_type} point")

            # Search spatial index for nearby points
            buffer_size = distance_threshold_meters / m_per_deg
            nearby_indices = list(idx.intersection((
                pt.x - buffer_size, pt.y - buffer_size,
                pt.x + buffer_size, pt.y + buffer_size
            )))

            candidates = []
            # Collect distance to each endpoint
            for i in nearby_indices:
                n_idx, n_pos, n_pt, n_id, n_v = network_endpoints[i]

                # Check if voltages match
                if not _voltage_match(jao_v, n_v):
                    continue

                # Calculate distance in meters
                dist_deg = pt.distance(n_pt)
                dist_m = dist_deg * m_per_deg

                # Only keep if within threshold
                if dist_m <= distance_threshold_meters:
                    candidates.append((dist_m, n_idx, n_pos, n_pt))

            # If no direct matches, try using cluster information
            if not candidates:
                # Find the nearest cluster centroid
                for label, (centroid, size) in cluster_centroids.items():
                    centroid_pt = Point(centroid)
                    dist_deg = pt.distance(centroid_pt)
                    dist_m = dist_deg * m_per_deg

                    # If point is near a cluster, add all cluster members as candidates
                    if dist_m <= distance_threshold_meters * 1.5:  # slightly larger radius
                        for n_idx, n_pos, n_pt, n_id, n_v in clusters[label]:
                            if _voltage_match(jao_v, n_v):
                                # Calculate actual distance to endpoint
                                exact_dist_m = pt.distance(n_pt) * m_per_deg
                                candidates.append((exact_dist_m, n_idx, n_pos, n_pt))

            # Sort by distance
            candidates.sort()

            # Select best match and alternatives
            best_match = candidates[0] if candidates else None
            alternatives = [(c[1], c[2]) for c in candidates[1:max_alternatives + 1]]

            if dbg:
                match_str = f"{best_match[1]}/{best_match[2]} at {best_match[0]:.1f}m" if best_match else "None"
                print(f"  Best match: {match_str}")
                print(f"  Alternatives: {alternatives}")

            return (
                (best_match[1], best_match[2]) if best_match else None,
                alternatives,
                best_match[0] if best_match else float('inf')
            )

        # -- find matches for both endpoints --------------------------- #
        s_match, s_alts, s_dist = _find_matches_for_point(start_pt, True)
        e_match, e_alts, e_dist = _find_matches_for_point(end_pt, False)

        # -- special case: if only one endpoint matched, try harder for the other -- #
        if (s_match is None) != (e_match is None):
            # Determine which endpoint is missing a match
            missing_end = 'start' if s_match is None else 'end'
            missing_pt = start_pt if missing_end == 'start' else end_pt

            if dbg:
                print(f"Only one endpoint matched, trying harder for {missing_end}...")

            # Increase search radius for the missing endpoint
            buffer_size = distance_threshold_meters * 2 / m_per_deg
            nearby_indices = list(idx.intersection((
                missing_pt.x - buffer_size, missing_pt.y - buffer_size,
                missing_pt.x + buffer_size, missing_pt.y + buffer_size
            )))

            candidates = []
            for i in nearby_indices:
                n_idx, n_pos, n_pt, n_id, n_v = network_endpoints[i]

                # More permissive voltage matching
                if abs(jao_v - n_v) <= 30:  # more tolerance
                    # Calculate distance in meters
                    dist_deg = missing_pt.distance(n_pt)
                    dist_m = dist_deg * m_per_deg

                    # Increased threshold
                    if dist_m <= distance_threshold_meters * 2:
                        candidates.append((dist_m, n_idx, n_pos, n_pt))

            # Sort by distance
            candidates.sort()

            # Update the missing match if found
            if candidates:
                if missing_end == 'start':
                    s_match = (candidates[0][1], candidates[0][2])
                    s_dist = candidates[0][0]
                    s_alts = [(c[1], c[2]) for c in candidates[1:max_alternatives + 1]]
                else:
                    e_match = (candidates[0][1], candidates[0][2])
                    e_dist = candidates[0][0]
                    e_alts = [(c[1], c[2]) for c in candidates[1:max_alternatives + 1]]

                if dbg:
                    print(f"  Found match for {missing_end}: {s_match if missing_end == 'start' else e_match}")

        # Store results
        nearest_points[j_idx] = {
            'start_nearest': s_match,
            'end_nearest': e_match,
            'start_alternatives': s_alts,
            'end_alternatives': e_alts,
            'start_distance_meters': s_dist,
            'end_distance_meters': e_dist
        }

    # ------------------------------------------------------------------ #
    # 4)  Report statistics                                              #
    # ------------------------------------------------------------------ #
    tot = len(jao_gdf)
    s_ok = sum(1 for d in nearest_points.values() if d['start_nearest'])
    e_ok = sum(1 for d in nearest_points.values() if d['end_nearest'])
    b_ok = sum(1 for d in nearest_points.values()
               if d['start_nearest'] and d['end_nearest'])

    print(f"Endpoint coverage ({distance_threshold_meters:.0f} m):  "
          f"start {s_ok}/{tot} ({s_ok / tot * 100:.1f}%)  "
          f"end {e_ok}/{tot} ({e_ok / tot * 100:.1f}%)  "
          f"both {b_ok}/{tot} ({b_ok / tot * 100:.1f}%)")

    return nearest_points



import math
import pandas as pd

def _to_km(value):
    """Best-effort: treat 'value' as km if it's already small, as meters if very large."""
    if value is None or pd.isna(value):
        return 0.0
    v = float(value)
    # Heuristic: lengths > 1000 are probably meters
    return v / 1000.0 if v > 1000 else v

def _get_first_existing(series, *candidates):
    for c in candidates:
        if c in series and pd.notna(series[c]):
            return series[c]
    return None

def _find_row_by_id(df, key):
    """Try common id/name columns to find a single row matching key."""
    if key is None:
        return None
    candidates = ['id', 'ID', 'jao_id', 'JAO_ID', 'network_id', 'name']
    for col in candidates:
        if col in df.columns:
            try:
                hit = df[df[col] == key]
                if len(hit) == 1:
                    return hit.iloc[0]
            except Exception:
                pass
    # Also try index name
    try:
        if key in df.index:
            return df.loc[key]
    except Exception:
        pass
    return None

import pandas as pd

def _num(x):
    """Parse numeric values; return None for None/''/'nan' etc."""
    if x is None:
        return None
    if isinstance(x, str) and x.strip() == "":
        return None
    try:
        v = float(x)
    except Exception:
        return None
    if pd.isna(v):
        return None
    return v

def _extract_jao_params(result, jao_gdf):
    """
    Returns a dict with:
      jao_length_km, jao_r_total, jao_x_total, jao_b_total,
      jao_r_per_km, jao_x_per_km, jao_b_per_km

    Robust to None/missing totals; prefers per‑km if available.
    """
    out = {
        'jao_length_km': 0.0,
        'jao_r_total': None, 'jao_x_total': None, 'jao_b_total': None,
        'jao_r_per_km': None, 'jao_x_per_km': None, 'jao_b_per_km': None
    }

    # --- length ---
    if _num(result.get('jao_length_km')) is not None:
        out['jao_length_km'] = float(result['jao_length_km'])
    elif _num(result.get('jao_length')) is not None:
        # your _to_km(...) helper converts m→km when needed
        out['jao_length_km'] = _to_km(result['jao_length'])
    else:
        row = _find_row_by_id(jao_gdf, result.get('jao_id'))
        if row is not None:
            Lcand = _num(row.get('length_km'))
            if Lcand is None:
                Lcand = _num(row.get('length'))
                Lcand = _to_km(Lcand) if Lcand is not None else None
            out['jao_length_km'] = Lcand or 0.0

    L = max(out['jao_length_km'], 1e-9)

    # Helper to compute totals from per-km
    def apply_per_km(rkm, xkm, bkm):
        if rkm is not None: out['jao_r_per_km'] = rkm; out['jao_r_total'] = rkm * L
        if xkm is not None: out['jao_x_per_km'] = xkm; out['jao_x_total'] = xkm * L
        if bkm is not None: out['jao_b_per_km'] = bkm; out['jao_b_total'] = bkm * L

    # Helper to compute per-km from totals
    def apply_totals(rt, xt, bt):
        if rt is not None: out['jao_r_total'] = rt; out['jao_r_per_km'] = rt / L
        if xt is not None: out['jao_x_total'] = xt; out['jao_x_per_km'] = xt / L
        if bt is not None: out['jao_b_total'] = bt; out['jao_b_per_km'] = bt / L

    # --- Priority 1: per‑km already on result ---
    rkm_res = _num(result.get('jao_r_per_km'))
    xkm_res = _num(result.get('jao_x_per_km'))
    bkm_res = _num(result.get('jao_b_per_km'))
    if any(v is not None for v in (rkm_res, xkm_res, bkm_res)):
        apply_per_km(rkm_res, xkm_res, bkm_res)

    # --- Priority 2: totals on result (only if not already filled) ---
    if out['jao_r_total'] is None or out['jao_x_total'] is None or out['jao_b_total'] is None:
        rt_res = _num(result.get('jao_r'))
        xt_res = _num(result.get('jao_x'))
        bt_res = _num(result.get('jao_b'))
        if any(v is not None for v in (rt_res, xt_res, bt_res)):
            apply_totals(rt_res, xt_res, bt_res)

    # --- Priority 3: fetch from jao_gdf (per‑km preferred, else totals) ---
    if (out['jao_r_total'] is None) or (out['jao_x_total'] is None) or (out['jao_b_total'] is None) \
       or (out['jao_r_per_km'] is None) or (out['jao_x_per_km'] is None) or (out['jao_b_per_km'] is None):
        row = _find_row_by_id(jao_gdf, result.get('jao_id'))
        if row is not None:
            # try per‑km first
            rkm = _num(row.get('R_per_km') or row.get('r_per_km'))
            xkm = _num(row.get('X_per_km') or row.get('x_per_km'))
            bkm = _num(row.get('B_per_km') or row.get('b_per_km'))
            if any(v is not None for v in (rkm, xkm, bkm)):
                # only fill what’s still missing
                if out['jao_r_per_km'] is None or out['jao_r_total'] is None:
                    apply_per_km(rkm, None, None)
                if out['jao_x_per_km'] is None or out['jao_x_total'] is None:
                    apply_per_km(None, xkm, None)
                if out['jao_b_per_km'] is None or out['jao_b_total'] is None:
                    apply_per_km(None, None, bkm)
            else:
                # try totals
                rt = _num(row.get('R_total') or row.get('R'))
                xt = _num(row.get('X_total') or row.get('X'))
                bt = _num(row.get('B_total') or row.get('B'))
                if any(v is not None for v in (rt, xt, bt)):
                    if out['jao_r_total'] is None or out['jao_r_per_km'] is None:
                        apply_totals(rt, None, None)
                    if out['jao_x_total'] is None or out['jao_x_per_km'] is None:
                        apply_totals(None, xt, None)
                    if out['jao_b_total'] is None or out['jao_b_per_km'] is None:
                        apply_totals(None, None, bt)

    return out


def find_best_path(G, start_node, end_node, voltage, max_attempts=20):
    """
    Find the best path between start and end nodes considering:
    1. Shortest path (by default)
    2. Path with minimal detours
    3. Path with consistent voltage
    """
    import networkx as nx

    paths = []

    # Try various shortest path algorithms with different weights
    try:
        # Standard shortest path
        path = nx.shortest_path(G, start_node, end_node, weight='weight')
        paths.append(('shortest', path))

        # Path with minimal number of segments
        path = nx.shortest_path(G, start_node, end_node, weight=1)
        paths.append(('minimal_segments', path))

        # Path with preference for same voltage
        def voltage_weight(u, v, data):
            edge_voltage = data.get('voltage', 0)
            if edge_voltage == voltage:
                return data.get('weight', 1) * 0.8  # Prefer same voltage
            else:
                return data.get('weight', 1) * 1.2  # Penalize different voltage

        path = nx.shortest_path(G, start_node, end_node, weight=voltage_weight)
        paths.append(('voltage_preference', path))

        # K shortest paths
        for i, path in enumerate(nx.shortest_simple_paths(G, start_node, end_node, weight='weight')):
            if i >= max_attempts:
                break
            paths.append((f'k_shortest_{i}', path))

    except nx.NetworkXNoPath:
        return None

    # Score and select the best path
    best_path = None
    best_score = float('inf')

    for label, path in paths:
        # Extract the network lines in this path
        segments = []
        for i in range(len(path) - 1):
            edge_data = G.get_edge_data(path[i], path[i + 1])
            if edge_data and 'id' in edge_data and not edge_data.get('connector', False):
                segments.append(edge_data)

        # Skip paths without real segments
        if not segments:
            continue

        # Calculate total length
        total_length = sum(s.get('weight', 0) for s in segments)

        # Calculate voltage consistency
        voltage_match = sum(1 for s in segments if s.get('voltage') == voltage) / len(segments)

        # Score = length * (2 - voltage_match)
        # This favors shorter paths with better voltage match
        score = total_length * (2 - voltage_match)

        if score < best_score:
            best_score = score
            best_path = path

    return best_path


def multi_stage_matching(jao_gdf, network_gdf, G):
    """
    Apply matching techniques in stages, with each stage handling a specific case:
    1. Endpoint-based path matching
    2. Geometric proximity matching
    3. Parallel circuit matching
    4. Shared corridor matching
    """
    # Stage 1: Endpoint matching
    print("Stage 1: Endpoint-based matching...")
    nearest_points_dict = find_nearest_points(jao_gdf, network_gdf, max_alternatives=10, distance_threshold_meters=1500)
    results = find_matching_network_lines_with_duplicates(jao_gdf, network_gdf, nearest_points_dict, G, ...)

    # Get unmatched JAO lines
    matched_jao_ids = {r['jao_id'] for r in results if r.get('matched')}
    unmatched_jao_gdf = jao_gdf[~jao_gdf['id'].astype(str).isin(matched_jao_ids)]

    # Stage 2: Geometric matching for remaining lines
    print(f"Stage 2: Geometric matching for {len(unmatched_jao_gdf)} unmatched lines...")
    geometric_results = match_remaining_lines_by_geometry(unmatched_jao_gdf, network_gdf, results)

    # Update matched JAO IDs
    matched_jao_ids = {r['jao_id'] for r in geometric_results if r.get('matched')}
    unmatched_jao_gdf = jao_gdf[~jao_gdf['id'].astype(str).isin(matched_jao_ids)]

    # Stage 3: Parallel circuit matching
    print(f"Stage 3: Parallel circuit matching for {len(unmatched_jao_gdf)} unmatched lines...")
    parallel_results = match_parallel_circuit_jao_with_network(geometric_results, unmatched_jao_gdf, network_gdf, G)

    # Stage 4: Apply corridor matching as a last resort
    print("Stage 4: Corridor-based matching...")
    final_results = corridor_parallel_match(parallel_results, jao_gdf, network_gdf)

    return final_results


def enhance_network_graph_connectivity(G, network_gdf, max_gap_meters=200):
    """
    Add connections between endpoints that are close to each other but not connected.
    This helps bridge small gaps in the network data.
    """
    import numpy as np
    from scipy.spatial import cKDTree

    # Extract all node positions
    positions = np.array([(data['x'], data['y']) for _, data in G.nodes(data=True)])
    nodes = list(G.nodes())

    # Build KD-tree for efficient nearest neighbor search
    tree = cKDTree(positions)

    # For each node, find nearby nodes within max_gap_meters
    avg_lat = np.mean(positions[:, 1])
    max_gap_degrees = max_gap_meters / (111111 * np.cos(np.radians(abs(avg_lat))))

    connections_added = 0

    for i, node in enumerate(nodes):
        # Query the tree for nodes within the distance threshold
        nearby_indices = tree.query_ball_point(positions[i], max_gap_degrees)

        for j in nearby_indices:
            if i == j:  # Skip self
                continue

            neighbor = nodes[j]

            # Skip if already connected
            if G.has_edge(node, neighbor):
                continue

            # Add new edge with connector flag
            G.add_edge(
                node, neighbor,
                weight=positions[i].distance(positions[j]),
                connector=True
            )
            connections_added += 1

    print(f"Added {connections_added} connections to enhance graph connectivity")
    return G


def identify_parallel_circuits(network_gdf, buffer_meters=50):
    """
    Identify sets of parallel circuits in the network by:
    1. Buffering each line
    2. Finding overlapping buffers with similar direction
    """
    import geopandas as gpd
    from shapely.geometry import LineString
    import numpy as np

    # Function to get the main direction vector of a line
    def get_direction_vector(line):
        if line.geom_type == 'MultiLineString':
            line = LineString([line.geoms[0].coords[0], line.geoms[-1].coords[-1]])

        coords = np.array(line.coords)
        vec = coords[-1] - coords[0]
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    # Create a buffer for each line
    avg_lat = network_gdf.geometry.centroid.y.mean()
    buffer_degrees = buffer_meters / (111111 * np.cos(np.radians(abs(avg_lat))))

    network_gdf = network_gdf.copy()
    network_gdf['buffer'] = network_gdf.geometry.buffer(buffer_degrees)
    network_gdf['direction'] = network_gdf.geometry.apply(get_direction_vector)

    # Find overlapping buffers
    parallel_groups = []
    processed = set()

    for idx, row in network_gdf.iterrows():
        if idx in processed:
            continue

        current_group = [idx]
        processed.add(idx)

        # Find lines with overlapping buffer and similar direction
        for other_idx, other_row in network_gdf.iterrows():
            if other_idx in processed:
                continue

            # Check voltage match
            if row['v_nom'] != other_row['v_nom']:
                continue

            # Check buffer overlap
            if not row['buffer'].intersects(other_row['buffer']):
                continue

            # Check direction alignment
            alignment = abs(np.dot(row['direction'], other_row['direction']))
            if alignment > 0.85:  # Roughly parallel (within ~30 degrees)
                current_group.append(other_idx)
                processed.add(other_idx)

        if len(current_group) > 1:
            parallel_groups.append(current_group)

    return parallel_groups


def _extract_network_segment_params(net_row):
    """
    Return length_km, original_totals, original_per_km
    The **totals/per-km are normalised to ONE circuit**.
    If the network row already represents multiple circuits
    (via 'circuits' or 'num_parallel'), we undo the aggregation:
       R, X  →  divide by n
       B     →  multiply by n
    """
    n_circ = _num(net_row.get('circuits')) \
             or _num(net_row.get('num_parallel')) \
             or 1.0                      # assume single circuit by default
    n_circ = max(float(n_circ), 1.0)

    length_km = _to_km(_get_first_existing(net_row, 'length_km', 'length'))
    L = max(length_km, 1e-9)

    # ---- read whatever is there ----
    r_km = _get_first_existing(net_row, 'R_per_km', 'r_per_km')
    x_km = _get_first_existing(net_row, 'X_per_km', 'x_per_km')
    b_km = _get_first_existing(net_row, 'B_per_km', 'b_per_km')

    r_tot = _get_first_existing(net_row, 'R_total', 'R')
    x_tot = _get_first_existing(net_row, 'X_total', 'X')
    b_tot = _get_first_existing(net_row, 'B_total', 'B')

    # ---- derive missing side ----
    if r_km is None and r_tot is not None:
        r_km = float(r_tot) / L
    if x_km is None and x_tot is not None:
        x_km = float(x_tot) / L
    if b_km is None and b_tot is not None:
        b_km = float(b_tot) / L

    if r_tot is None and r_km is not None:
        r_tot = float(r_km) * L
    if x_tot is None and x_km is not None:
        x_tot = float(x_km) * L
    if b_tot is None and b_km is not None:
        b_tot = float(b_km) * L

    n = _num(net_row.get('circuits')) or _num(net_row.get('num_parallel')) or 1
    n = max(1, int(n))

    # ---- **undo aggregation** ----
    # per-circuit values
    r_tot_1 = r_tot / n_circ if r_tot is not None else None
    x_tot_1 = x_tot / n_circ if x_tot is not None else None
    b_tot_1 = b_tot * n_circ if b_tot is not None else None  # susceptance adds
    r_km_1  = r_km  / n_circ if r_km  is not None else None
    x_km_1  = x_km  / n_circ if x_km  is not None else None
    b_km_1  = b_km  * n_circ if b_km  is not None else None

    return length_km, r_tot_1, x_tot_1, b_tot_1, r_km_1, x_km_1, b_km_1



# --- helper utilities inside create_enhanced_summary_table ---
def _is_num(x):
    try:
        return x is not None and not pd.isna(x) and isinstance(float(x), float)
    except Exception:
        return False

def _first_col(d, candidates):
    for c in candidates:
        if c in d.columns:
            return c
    return None

def _get_len_km_from_result_or_gdf(res, gdf, jao_id_val):
    # prefer length on result
    if _is_num(res.get('jao_length_km')):
        return float(res['jao_length_km'])
    if _is_num(res.get('jao_length')):  # meters
        return float(res['jao_length']) / 1000.0

    # fallback to gdf: common length column names
    len_col = _first_col(gdf, ['length_km', 'len_km', 'lengthkm', 'length'])
    if len_col is not None:
        row = gdf.loc[gdf[_first_col(gdf, ['jao_id','JAO_ID','id','ID'])] == jao_id_val]
        if not row.empty:
            v = row.iloc[0][len_col]
            # if looks like meters, convert
            if _is_num(v):
                v = float(v)
                if v > 1000:  # likely meters
                    return v / 1000.0
                return v

    # as a last resort: try geometry length if present (assumes meters)
    if 'geometry' in gdf.columns:
        id_col = _first_col(gdf, ['jao_id','JAO_ID','id','ID'])
        if id_col:
            row = gdf.loc[gdf[id_col] == jao_id_val]
            if not row.empty:
                try:
                    L = row.iloc[0].geometry.length
                    if _is_num(L):
                        L = float(L)
                        return L / 1000.0 if L > 1000 else L
                except Exception:
                    pass
    return None

def _get_jao_params(res, gdf):
    """
    Return dict with totals & per-km for (r,x,b) using:
    1) values already in `res`
    2) fallback to jao_gdf columns if needed
    Computes missing totals/per-km from length when possible.
    """
    out = {
        'length_km': None,
        'r_total': None, 'x_total': None, 'b_total': None,
        'r_km': None, 'x_km': None, 'b_km': None
    }

    jao_id_val = res.get('jao_id')
    L = _get_len_km_from_result_or_gdf(res, gdf, jao_id_val)
    if _is_num(L):
        out['length_km'] = float(L)

    # 1) take from result if present
    if _is_num(res.get('jao_r')): out['r_total'] = float(res['jao_r'])
    if _is_num(res.get('jao_x')): out['x_total'] = float(res['jao_x'])
    if _is_num(res.get('jao_b')): out['b_total'] = float(res['jao_b'])
    if _is_num(res.get('jao_r_per_km')): out['r_km'] = float(res['jao_r_per_km'])
    if _is_num(res.get('jao_x_per_km')): out['x_km'] = float(res['jao_x_per_km'])
    if _is_num(res.get('jao_b_per_km')): out['b_km'] = float(res['jao_b_per_km'])

    have_all = all(_is_num(out[k]) for k in ['r_total','x_total','b_total','r_km','x_km','b_km'])
    # 2) fallback to gdf only if anything missing
    if not have_all:
        id_col = _first_col(gdf, ['jao_id','JAO_ID','id','ID'])
        row = gdf.loc[gdf[id_col] == jao_id_val] if id_col else pd.DataFrame()
        if not row.empty:
            row = row.iloc[0]
            # candidate column sets (adjust if your column names differ)
            totals_cands = [('R','X','B'), ('r_total','x_total','b_total'), ('r','x','b')]
            perkm_cands  = [('R_per_km','X_per_km','B_per_km'), ('r_km','x_km','b_km'), ('r_per_km','x_per_km','b_per_km')]

            # try totals
            for cols in totals_cands:
                Rc, Xc, Bc = cols
                if Rc in row and Xc in row and Bc in row:
                    if _is_num(row[Rc]) and out['r_total'] is None: out['r_total'] = float(row[Rc])
                    if _is_num(row[Xc]) and out['x_total'] is None: out['x_total'] = float(row[Xc])
                    if _is_num(row[Bc]) and out['b_total'] is None: out['b_total'] = float(row[Bc])
                    break
            # try per-km
            for cols in perkm_cands:
                Rkc, Xkc, Bkc = cols
                if Rkc in row and Xkc in row and Bkc in row:
                    if _is_num(row[Rkc]) and out['r_km'] is None: out['r_km'] = float(row[Rkc])
                    if _is_num(row[Xkc]) and out['x_km'] is None: out['x_km'] = float(row[Xkc])
                    if _is_num(row[Bkc]) and out['b_km'] is None: out['b_km'] = float(row[Bkc])
                    break

    # 3) complete missing totals/per-km using length
    L = out['length_km']
    if _is_num(L) and L > 0:
        if _is_num(out['r_total']) and not _is_num(out['r_km']):
            out['r_km'] = out['r_total'] / L
        if _is_num(out['x_total']) and not _is_num(out['x_km']):
            out['x_km'] = out['x_total'] / L
        if _is_num(out['b_total']) and not _is_num(out['b_km']):
            out['b_km'] = out['b_total'] / L

        if _is_num(out['r_km']) and not _is_num(out['r_total']):
            out['r_total'] = out['r_km'] * L
        if _is_num(out['x_km']) and not _is_num(out['x_total']):
            out['x_total'] = out['x_km'] * L
        if _is_num(out['b_km']) and not _is_num(out['b_total']):
            out['b_total'] = out['b_km'] * L

    return out
# --- end helpers ---

def allocate_electrical_parameters(jao_gdf, network_gdf, matching_results):
    """
    For each matched JAO line:
      - Read JAO per-km parameters
      - For each network segment, check if it represents parallel circuits
      - Calculate per-circuit and aggregate-equivalent parameters
      - Ensure each network line is only allocated parameters once
      - Compute per-km diffs vs network per-km
    """
    # Track which network lines have already been allocated parameters
    allocated_network_ids = set()

    # Sort results to process non-duplicates first
    sorted_results = sorted(matching_results, key=lambda r: r.get('is_duplicate', False))

    print("\n=== ALLOCATING ELECTRICAL PARAMETERS WITH PARALLEL CIRCUIT HANDLING ===")

    net_by_id = {str(r['id']): r for _, r in network_gdf.iterrows()}

    for result in sorted_results:
        jao_id = result.get('jao_id', 'unknown')
        print(f"Processing JAO {jao_id}")

        # Skip if not matched or no network lines
        if not result.get('matched', False) or not result.get('network_ids'):
            continue

        # Flag for duplicate JAOs
        is_duplicate = result.get('is_duplicate', False)

        # Check if this was a pruned match - if so, we need to remove any network IDs
        # from the allocated_network_ids set to allow reallocating to the pruned set
        if result.get('pruned_branches', False):
            print(f"  JAO {jao_id} was pruned - resetting allocation status for its network lines")
            # Get the current network IDs
            current_network_ids = result.get('network_ids', [])
            # Remove them from the allocated set
            for nid in current_network_ids:
                if nid in allocated_network_ids:
                    allocated_network_ids.remove(nid)
            # Clear existing allocations
            if 'matched_lines_data' in result:
                result.pop('matched_lines_data')

        # Get JAO parameters
        jp = _extract_jao_params(result, jao_gdf)
        result['jao_length_km'] = jp['jao_length_km']
        result['jao_r'] = jp['jao_r_total']
        result['jao_x'] = jp['jao_x_total']
        result['jao_b'] = jp['jao_b_total']
        result['jao_r_per_km'] = jp['jao_r_per_km']
        result['jao_x_per_km'] = jp['jao_x_per_km']
        result['jao_b_per_km'] = jp['jao_b_per_km']

        print(f"  JAO {jao_id} per-km: R={jp['jao_r_per_km']} X={jp['jao_x_per_km']} B={jp['jao_b_per_km']}")

        # Skip if we don't have all JAO parameters
        if any(v is None for v in (jp['jao_r_per_km'], jp['jao_x_per_km'], jp['jao_b_per_km'])):
            print(f"  Missing JAO parameters for {jao_id}, skipping")
            continue

        # New arrays for segments and totals
        segments = []
        total_matched_km = 0.0
        sum_alloc_r = 0.0
        sum_alloc_x = 0.0
        sum_alloc_b = 0.0

        for net_id in result.get('network_ids', []):
            nid = str(net_id)
            row = net_by_id.get(nid)
            if row is None:
                print(f"  Network line {nid} not found in network_gdf")
                continue

            # circuits
            raw_circuits = _num(row.get('circuits'))
            raw_np = _num(row.get('num_parallel'))
            num_parallel = int(raw_circuits or raw_np or 1)
            print(f"  {nid}: circuits={num_parallel} (row[circuits]={raw_circuits}, row[num_parallel]={raw_np})")

            # accept several aliases; use 'circuits' if your normalization above ran
            def _first_existing_val(row, *names):
                for n in names:
                    if n in row and pd.notna(row[n]):
                        return _num(row[n])
                return None

            raw_circuits = _first_existing_val(row, 'circuits', 'num_parallel', 'parallel_num', 'n_circuits')
            num_parallel = int(raw_circuits or 1)

            print(f"  {net_id}: circuits={num_parallel} "
                  f"(row[circuits]={row.get('circuits', None)}, "
                  f"row[num_parallel]={row.get('num_parallel', None)}, "
                  f"row[parallel_num]={row.get('parallel_num', None)})")

            # (optional) inference only if still 1
            if num_parallel == 1:
                pass  # Placeholder for inference logic

            # 1. Direct attribute access (for pandas Series)
            if hasattr(row, 'num_parallel'):
                try:
                    np_value = row.num_parallel
                    print(f"  Direct attribute access found: {np_value}, type: {type(np_value)}")
                    if pd.notna(np_value) and np_value is not None:
                        num_parallel = int(float(np_value))
                        print(f"  Using num_parallel={num_parallel} from direct attribute access")
                except Exception as e:
                    print(f"  Error reading num_parallel via attribute: {e}")

            # 2. Dictionary-style access (for dict-like objects)
            elif hasattr(row, 'get') and callable(row.get):
                try:
                    np_value = row.get('num_parallel')
                    print(f"  Dictionary access found: {np_value}, type: {type(np_value)}")
                    if pd.notna(np_value) and np_value is not None:
                        num_parallel = int(float(np_value))
                        print(f"  Using num_parallel={num_parallel} from dictionary access")
                except Exception as e:
                    print(f"  Error reading num_parallel via dict: {e}")

            # 3. Try direct indexing (for pandas Series)
            elif hasattr(row, '__getitem__'):
                try:
                    np_value = row['num_parallel']
                    print(f"  Indexing found: {np_value}, type: {type(np_value)}")
                    if pd.notna(np_value) and np_value is not None:
                        num_parallel = int(float(np_value))
                        print(f"  Using num_parallel={num_parallel} from indexing")
                except Exception as e:
                    print(f"  Error reading num_parallel via indexing: {e}")

            # 4. Special case for Line_17046
            if net_id == 'Line_17046':
                print(f"  SPECIAL CASE: Setting num_parallel=2 for {net_id}")
                num_parallel = 2

            # (optional) Last resort inference if it's still 1
            if num_parallel == 1:
                try:
                    length_km = _to_km(_num(_get_first_existing(row, 'length_km', 'length')))
                    r_val = _num(_get_first_existing(row, 'r', 'R'))
                    x_val = _num(_get_first_existing(row, 'x', 'X'))

                    if r_val is not None and length_km and length_km > 0:
                        r_per_km = r_val / length_km
                        ratio = jp['jao_r_per_km'] / r_per_km if r_per_km > 0 else 0
                        if 1.8 <= ratio <= 2.2:
                            print(f"  Inferred num_parallel=2 based on R ratio {ratio:.2f}")
                            num_parallel = 2
                    elif x_val is not None and length_km and length_km > 0:
                        x_per_km = x_val / length_km
                        ratio = jp['jao_x_per_km'] / x_per_km if x_per_km > 0 else 0
                        if 1.8 <= ratio <= 2.2:
                            print(f"  Inferred num_parallel=2 based on X ratio {ratio:.2f}")
                            num_parallel = 2
                except Exception as e:
                    print(f"  Error inferring num_parallel: {e}")

            print(f"  Final num_parallel value for {net_id}: {num_parallel}")

            # --- STEP 2: Get network parameters ---
            # Get length in km
            length_km = _to_km(_get_first_existing(row, 'length_km', 'length'))
            total_matched_km += length_km

            # Get original aggregate values from network row
            orig_r_km = _num(_get_first_existing(row, 'r_per_km', 'R_per_km'))
            orig_x_km = _num(_get_first_existing(row, 'x_per_km', 'X_per_km'))
            orig_b_km = _num(_get_first_existing(row, 'b_per_km', 'B_per_km'))

            orig_r = _num(_get_first_existing(row, 'r', 'R'))
            orig_x = _num(_get_first_existing(row, 'x', 'X'))
            orig_b = _num(_get_first_existing(row, 'b', 'B'))

            # Calculate per-km values if not directly available
            if orig_r_km is None and orig_r is not None and length_km > 0:
                orig_r_km = orig_r / length_km
            if orig_x_km is None and orig_x is not None and length_km > 0:
                orig_x_km = orig_x / length_km
            if orig_b_km is None and orig_b is not None and length_km > 0:
                orig_b_km = orig_b / length_km

            # Calculate total values if not directly available
            if orig_r is None and orig_r_km is not None:
                orig_r = orig_r_km * length_km
            if orig_x is None and orig_x_km is not None:
                orig_x = orig_x_km * length_km
            if orig_b is None and orig_b_km is not None:
                orig_b = orig_b_km * length_km

            # --- STEP 3: Calculate per-circuit values ---
            # For network row (de-aggregate):
            orig_r_km_pc = orig_r_km * num_parallel if orig_r_km is not None else None
            orig_x_km_pc = orig_x_km * num_parallel if orig_x_km is not None else None
            orig_b_km_pc = orig_b_km / num_parallel if orig_b_km is not None else None

            orig_r_pc = orig_r * num_parallel if orig_r is not None else None
            orig_x_pc = orig_x * num_parallel if orig_x is not None else None
            orig_b_pc = orig_b / num_parallel if orig_b is not None else None

            # --- STEP 4: Calculate aggregate-equivalent values for allocation ---
            # JAO values are per-circuit, so adjust for num_parallel:
            agg_r_km = jp['jao_r_per_km'] / num_parallel
            agg_x_km = jp['jao_x_per_km'] / num_parallel
            agg_b_km = jp['jao_b_per_km'] * num_parallel

            agg_r = agg_r_km * length_km
            agg_x = agg_x_km * length_km
            agg_b = agg_b_km * length_km

            # --- STEP 5: Determine allocation status ---
            # Check if this line was already allocated parameters, but consider pruned matches
            # If the match was pruned, we want to allocate parameters even if previously allocated
            already_allocated = nid in allocated_network_ids and not result.get('pruned_branches', False)

            if is_duplicate:
                allocation_status = "Duplicate JAO (skip)"
                alloc_r = alloc_x = alloc_b = 0.0
            elif already_allocated:
                allocation_status = "Already allocated (skip)"
                alloc_r = alloc_x = alloc_b = 0.0
            else:
                allocation_status = "Applied"
                alloc_r = agg_r
                alloc_x = agg_x
                alloc_b = agg_b

                # Mark as allocated
                allocated_network_ids.add(nid)

                # Add to running sums
                sum_alloc_r += alloc_r
                sum_alloc_x += alloc_x
                sum_alloc_b += alloc_b

            # --- STEP 6: Calculate difference percentages ---
            def diff_pct(a, b):
                if a is None or b is None or abs(b) < 1e-9:
                    return float('inf')
                return 100.0 * (a - b) / abs(b)

            # Per-circuit differences
            r_diff_pc = diff_pct(jp['jao_r_per_km'], orig_r_km_pc)
            x_diff_pc = diff_pct(jp['jao_x_per_km'], orig_x_km_pc)
            b_diff_pc = diff_pct(jp['jao_b_per_km'], orig_b_km_pc)

            # Aggregate differences
            r_diff_agg = diff_pct(agg_r_km, orig_r_km)
            x_diff_agg = diff_pct(agg_x_km, orig_x_km)
            b_diff_agg = diff_pct(agg_b_km, orig_b_km)

            # --- STEP 7: Create segment data ---
            segment = {
                'network_id': nid,
                'length_km': length_km,
                'num_parallel': num_parallel,
                'segment_ratio': length_km / max(jp['jao_length_km'], 1e-9),

                # Allocated values (adjusted for num_parallel)
                'allocated_r': alloc_r,
                'allocated_x': alloc_x,
                'allocated_b': alloc_b,
                'allocated_r_per_km': agg_r_km,
                'allocated_x_per_km': agg_x_km,
                'allocated_b_per_km': agg_b_km,

                # Original values (as stored in network_gdf)
                'original_r': orig_r or 0.0,
                'original_x': orig_x or 0.0,
                'original_b': orig_b or 0.0,
                'original_r_per_km': orig_r_km or 0.0,
                'original_x_per_km': orig_x_km or 0.0,
                'original_b_per_km': orig_b_km or 0.0,

                # Per-circuit values for comparison
                'jao_r_per_km_pc': jp['jao_r_per_km'],
                'jao_x_per_km_pc': jp['jao_x_per_km'],
                'jao_b_per_km_pc': jp['jao_b_per_km'],
                'original_r_per_km_pc': orig_r_km_pc or 0.0,
                'original_x_per_km_pc': orig_x_km_pc or 0.0,
                'original_b_per_km_pc': orig_b_km_pc or 0.0,

                # Difference percentages
                'r_diff_percent_pc': r_diff_pc,
                'x_diff_percent_pc': x_diff_pc,
                'b_diff_percent_pc': b_diff_pc,
                'r_diff_percent': r_diff_agg,
                'x_diff_percent': x_diff_agg,
                'b_diff_percent': b_diff_agg,

                'allocation_status': allocation_status
            }

            segments.append(segment)

        # Update result with segment data and totals
        result['matched_lines_data'] = segments
        result['matched_km'] = total_matched_km
        result['coverage_ratio'] = total_matched_km / max(jp['jao_length_km'], 1e-9)
        result['allocated_r_sum'] = sum_alloc_r
        result['allocated_x_sum'] = sum_alloc_x
        result['allocated_b_sum'] = sum_alloc_b

        # Calculate residuals
        def residual_pct(total, alloc):
            if total is None or abs(total) < 1e-12:
                return float('inf')
            return 100.0 * (total - alloc) / abs(total)

        result['residual_r_percent'] = residual_pct(result['jao_r'], sum_alloc_r)
        result['residual_x_percent'] = residual_pct(result['jao_x'], sum_alloc_x)
        result['residual_b_percent'] = residual_pct(result['jao_b'], sum_alloc_b)

    return matching_results


def _get_circuits_from_row(row):
    # prefer a normalized 'circuits' col if present, else look for num_parallel
    raw = _num(row.get('circuits'))
    if raw is None:
        raw = _num(row.get('num_parallel'))
    return int(raw or 1)


def match_fragmented_lines(matching_results, jao_gdf, network_gdf, max_gap_meters=1000):
    """
    Connect network segments that should be part of the same JAO line but have small gaps.
    Also fixes matches with extreme length ratios to prevent incorrect allocations.
    """
    print("\nFixing fragmented line segments...")

    # First fix any matches with extreme length ratios
    fix_extreme_ratio_matches(matching_results, jao_gdf, network_gdf)

    # Track JAO lines we've already processed
    processed_jao_ids = set()
    extreme_ratio_jao_ids = set(r.get('jao_id') for r in matching_results if r.get('_extreme_ratio_fixed'))

    for result in matching_results:
        jao_id = result.get('jao_id')

        # Skip if not matched, already processed, or already fixed as extreme ratio
        if not result.get('matched') or jao_id in processed_jao_ids or jao_id in extreme_ratio_jao_ids:
            continue

        # Skip if no network IDs
        if not result.get('network_ids'):
            continue

        # Get JAO geometry
        jao_rows = jao_gdf[jao_gdf['id'].astype(str) == str(jao_id)]
        if jao_rows.empty:
            continue

        jao_geom = jao_rows.iloc[0].geometry
        jao_voltage = _safe_int(jao_rows.iloc[0]['v_nom'])

        # Create buffer around JAO line to constrain candidate network lines
        avg_lat = jao_geom.centroid.y
        m_per_deg = _meters_per_degree(avg_lat)
        buffer_width = max_gap_meters / m_per_deg
        jao_buffer = jao_geom.buffer(buffer_width)

        # Get existing network lines and their endpoints
        matched_segments = []
        for nid in result.get('network_ids'):
            net_rows = network_gdf[network_gdf['id'].astype(str) == str(nid)]
            if not net_rows.empty:
                matched_segments.append({
                    'id': nid,
                    'geometry': net_rows.iloc[0].geometry,
                    'voltage': _safe_int(net_rows.iloc[0]['v_nom'])
                })

        # If no matched segments, skip
        if not matched_segments:
            continue

        # Find candidate segments that could connect the matched segments
        # Only consider segments that are BOTH:
        # 1. Inside the JAO buffer (close to the JAO line)
        # 2. Close to an endpoint of an already matched segment
        candidates = []

        # Extract endpoints of matched segments
        matched_endpoints = []
        for seg in matched_segments:
            geom = seg['geometry']
            if geom.geom_type == "LineString":
                matched_endpoints.append(Point(geom.coords[0]))
                matched_endpoints.append(Point(geom.coords[-1]))
            elif geom.geom_type == "MultiLineString":
                matched_endpoints.append(Point(geom.geoms[0].coords[0]))
                matched_endpoints.append(Point(geom.geoms[-1].coords[-1]))

        # Find unmatched network segments that could connect to matched segments
        for _, row in network_gdf.iterrows():
            nid = str(row['id'])

            # Skip if already matched to this JAO
            if nid in result.get('network_ids', []):
                continue

            # Skip if different voltage
            if not _same_voltage(jao_voltage, _safe_int(row['v_nom'])):
                continue

            # Skip if not inside JAO buffer
            if not jao_buffer.intersects(row.geometry):
                continue

            # Check if this segment is close to any matched endpoint
            for endpoint in matched_endpoints:
                geom = row.geometry

                # Get segment endpoints
                seg_endpoints = []
                if geom.geom_type == "LineString":
                    seg_endpoints = [Point(geom.coords[0]), Point(geom.coords[-1])]
                elif geom.geom_type == "MultiLineString":
                    seg_endpoints = [Point(geom.geoms[0].coords[0]), Point(geom.geoms[-1].coords[-1])]

                # Check distance between endpoints
                for seg_ep in seg_endpoints:
                    dist_deg = endpoint.distance(seg_ep)
                    dist_m = dist_deg * m_per_deg

                    if dist_m <= max_gap_meters:
                        candidates.append({
                            'id': nid,
                            'distance': dist_m,
                            'geometry': row.geometry
                        })
                        break

        # If no candidates, nothing to do
        if not candidates:
            continue

        # Sort by distance
        candidates.sort(key=lambda x: x['distance'])

        # Add the candidates to the match
        added = 0
        for candidate in candidates:
            # Add this segment only if adding it doesn't make the match worse
            current_ratio = result.get('length_ratio', 1.0)

            # Calculate new ratio if we add this segment
            cand_length = calculate_length_meters(candidate['geometry'])
            new_total = result.get('path_length', 0) + cand_length
            new_ratio = new_total / result.get('jao_length', 1)

            # Only add if it doesn't make ratio much worse
            if new_ratio < current_ratio * 1.2:  # Allow 20% increase at most
                result.setdefault('network_ids', []).append(candidate['id'])
                result['path_length'] = float(new_total)
                if result.get('jao_length'):
                    result['length_ratio'] = float(new_ratio)
                added += 1
                print(f"  Added segment {candidate['id']} to JAO {jao_id} (gap: {candidate['distance']:.1f}m)")

        if added > 0:
            processed_jao_ids.add(jao_id)

    return matching_results


def fix_extreme_ratio_matches(matching_results, jao_gdf, network_gdf):
    """
    Fix matches with extreme length ratios to prevent incorrect allocations.
    This includes cases where network lines are much shorter or much longer than the JAO line.
    """
    print("Fixing matches with extreme length ratios...")

    # Define thresholds for extreme ratios
    MIN_RATIO_THRESHOLD = 0.05  # Network line is <= 5% of JAO length
    MAX_RATIO_THRESHOLD = 5.0  # Network line is >= 500% of JAO length

    # Helper function for percentage difference
    def _pct_diff(a, b):
        try:
            if a is None or b is None or abs(b) < 1e-9:
                return float('inf')
            return 100.0 * (a - b) / abs(b)
        except Exception:
            return float('inf')

    # Helper function for parsing numeric values
    def _num(x):
        try:
            if x is None:
                return None
            if isinstance(x, (int, float)):
                if isinstance(x, float) and math.isnan(x):
                    return None
                return float(x)
            s = str(x).strip().replace(",", ".")
            if s == "":
                return None
            return float(s)
        except Exception:
            return None

    # Helper function for getting km value
    def _to_km(value):
        if value is None or pd.isna(value):
            return 0.0
        v = float(value)
        # Heuristic: lengths > 1000 are probably meters
        return v / 1000.0 if v > 1000 else v

    # Helper function to get first existing value from a row
    def _get_first_existing(row, *names):
        for name in names:
            if name in row and pd.notna(row[name]):
                return row[name]
        return None

    fixed_count = 0

    for result in matching_results:
        # Skip if not matched
        if not result.get('matched'):
            continue

        jao_id = result.get('jao_id')

        # Check if length ratio is extreme
        ratio = result.get('length_ratio')
        if ratio is None:
            # Calculate ratio if missing
            if result.get('path_length') and result.get('jao_length') and result['jao_length'] > 0:
                ratio = result['path_length'] / result['jao_length']

        # Skip if ratio is within normal range
        if ratio is None or (MIN_RATIO_THRESHOLD <= ratio <= MAX_RATIO_THRESHOLD):
            continue

        print(f"  JAO {jao_id} has extreme length ratio: {ratio:.3f}")

        # Get JAO parameters
        jao_rows = jao_gdf[jao_gdf['id'].astype(str) == str(jao_id)]
        if jao_rows.empty:
            print(f"  Error: JAO {jao_id} not found in jao_gdf")
            continue

        jao_row = jao_rows.iloc[0]

        # Get JAO parameters
        jao_r = _num(_get_first_existing(jao_row, 'r', 'R'))
        jao_x = _num(_get_first_existing(jao_row, 'x', 'X'))
        jao_b = _num(_get_first_existing(jao_row, 'b', 'B'))

        jao_r_per_km = _num(_get_first_existing(jao_row, 'R_per_km', 'r_per_km'))
        jao_x_per_km = _num(_get_first_existing(jao_row, 'X_per_km', 'x_per_km'))
        jao_b_per_km = _num(_get_first_existing(jao_row, 'B_per_km', 'b_per_km'))

        # If per-km not available, calculate from totals
        jao_length_km = _to_km(_get_first_existing(jao_row, 'length', 'length_km'))
        if jao_r_per_km is None and jao_r is not None and jao_length_km > 0:
            jao_r_per_km = jao_r / jao_length_km
        if jao_x_per_km is None and jao_x is not None and jao_length_km > 0:
            jao_x_per_km = jao_x / jao_length_km
        if jao_b_per_km is None and jao_b is not None and jao_length_km > 0:
            jao_b_per_km = jao_b / jao_length_km

        # Process each network line individually
        network_ids = result.get('network_ids', [])
        segments = []
        total_length_km = 0

        for nid in network_ids:
            net_rows = network_gdf[network_gdf['id'].astype(str) == str(nid)]
            if net_rows.empty:
                continue

            net_row = net_rows.iloc[0]
            length_km = _to_km(_get_first_existing(net_row, 'length_km', 'length'))
            total_length_km += length_km

            # Determine number of parallel circuits
            num_parallel = 1
            if 'circuits' in net_row and pd.notna(net_row['circuits']):
                num_parallel = max(1, int(net_row['circuits']))
            elif 'num_parallel' in net_row and pd.notna(net_row['num_parallel']):
                num_parallel = max(1, int(net_row['num_parallel']))

            # Get network parameters
            orig_r_km = _num(_get_first_existing(net_row, 'r_per_km', 'R_per_km'))
            orig_x_km = _num(_get_first_existing(net_row, 'x_per_km', 'X_per_km'))
            orig_b_km = _num(_get_first_existing(net_row, 'b_per_km', 'B_per_km'))

            orig_r = _num(_get_first_existing(net_row, 'r', 'R'))
            orig_x = _num(_get_first_existing(net_row, 'x', 'X'))
            orig_b = _num(_get_first_existing(net_row, 'b', 'B'))

            # Ensure totals and per-km are consistent
            if orig_r_km is None and orig_r is not None and length_km > 0:
                orig_r_km = orig_r / length_km
            if orig_x_km is None and orig_x is not None and length_km > 0:
                orig_x_km = orig_x / length_km
            if orig_b_km is None and orig_b is not None and length_km > 0:
                orig_b_km = orig_b / length_km

            if orig_r is None and orig_r_km is not None:
                orig_r = orig_r_km * length_km
            if orig_x is None and orig_x_km is not None:
                orig_x = orig_x_km * length_km
            if orig_b is None and orig_b_km is not None:
                orig_b = orig_b_km * length_km

            # Calculate per-circuit values
            orig_r_km_pc = orig_r_km * num_parallel if orig_r_km is not None else None
            orig_x_km_pc = orig_x_km * num_parallel if orig_x_km is not None else None
            orig_b_km_pc = orig_b_km / num_parallel if orig_b_km is not None else None

            # Calculate aggregate-equivalent values for allocation
            segment_ratio = length_km / total_length_km if total_length_km > 0 else 0

            # Calculate allocated parameters
            alloc_r = jao_r * segment_ratio if jao_r is not None else None
            alloc_x = jao_x * segment_ratio if jao_x is not None else None
            alloc_b = jao_b * segment_ratio if jao_b is not None else None

            # Calculate per-km values
            alloc_r_per_km = alloc_r / length_km if alloc_r is not None and length_km > 0 else None
            alloc_x_per_km = alloc_x / length_km if alloc_x is not None and length_km > 0 else None
            alloc_b_per_km = alloc_b / length_km if alloc_b is not None and length_km > 0 else None

            # Create segment data
            segment = {
                'network_id': nid,
                'length_km': length_km,
                'num_parallel': num_parallel,
                'segment_ratio': segment_ratio,

                # Allocated values
                'allocated_r': alloc_r,
                'allocated_x': alloc_x,
                'allocated_b': alloc_b,
                'allocated_r_per_km': alloc_r_per_km,
                'allocated_x_per_km': alloc_x_per_km,
                'allocated_b_per_km': alloc_b_per_km,

                # Original values
                'original_r': orig_r or 0.0,
                'original_x': orig_x or 0.0,
                'original_b': orig_b or 0.0,
                'original_r_per_km': orig_r_km or 0.0,
                'original_x_per_km': orig_x_km or 0.0,
                'original_b_per_km': orig_b_km or 0.0,

                # Per-circuit values
                'jao_r_per_km_pc': jao_r_per_km,
                'jao_x_per_km_pc': jao_x_per_km,
                'jao_b_per_km_pc': jao_b_per_km,
                'original_r_per_km_pc': orig_r_km_pc or 0.0,
                'original_x_per_km_pc': orig_x_km_pc or 0.0,
                'original_b_per_km_pc': orig_b_km_pc or 0.0,

                # Difference percentages
                'r_diff_percent_pc': _pct_diff(jao_r_per_km, orig_r_km_pc),
                'x_diff_percent_pc': _pct_diff(jao_x_per_km, orig_x_km_pc),
                'b_diff_percent_pc': _pct_diff(jao_b_per_km, orig_b_km_pc),
                'r_diff_percent': _pct_diff(alloc_r, orig_r),
                'x_diff_percent': _pct_diff(alloc_x, orig_x),
                'b_diff_percent': _pct_diff(alloc_b, orig_b),

                'allocation_status': f'Fixed extreme ratio ({ratio:.3f})'
            }

            segments.append(segment)

        # Replace the segments with our fixed ones
        result['matched_lines_data'] = segments

        # Set the coverage using the actual matched network length
        result['matched_km'] = total_length_km
        if jao_length_km and jao_length_km > 0:
            # For extreme ratios, we set a "normalized" coverage to prevent backfilling
            result['coverage_ratio'] = 1.0

        # Calculate allocated sums
        alloc_r_sum = sum(seg.get('allocated_r', 0.0) or 0.0 for seg in segments)
        alloc_x_sum = sum(seg.get('allocated_x', 0.0) or 0.0 for seg in segments)
        alloc_b_sum = sum(seg.get('allocated_b', 0.0) or 0.0 for seg in segments)

        result['allocated_r_sum'] = alloc_r_sum
        result['allocated_x_sum'] = alloc_x_sum
        result['allocated_b_sum'] = alloc_b_sum

        # Calculate residuals
        def residual_pct(total, alloc):
            if total is None or abs(total) < 1e-12:
                return 0.0
            return 100.0 * (total - alloc) / abs(total)

        result['residual_r_percent'] = residual_pct(jao_r, alloc_r_sum)
        result['residual_x_percent'] = residual_pct(jao_x, alloc_x_sum)
        result['residual_b_percent'] = residual_pct(jao_b, alloc_b_sum)

        # Mark as fixed
        result['_extreme_ratio_fixed'] = True

        fixed_count += 1

    print(f"Fixed {fixed_count} matches with extreme length ratios")
    return matching_results


def handle_extreme_ratio_matches(matching_results, jao_gdf, network_gdf,
                                 min_ratio_threshold=0.05, max_ratio_threshold=5.0):
    """
    Fix matches with extreme length ratios (both very small and very large).

    Parameters:
    - matching_results: List of match results
    - jao_gdf: JAO GeoDataFrame
    - network_gdf: Network GeoDataFrame
    - min_ratio_threshold: Minimum acceptable ratio (default 0.05)
    - max_ratio_threshold: Maximum acceptable ratio (default 5.0)

    Returns:
    - Updated matching_results
    """
    import math
    import pandas as pd

    print("\nHandling extreme length ratio matches...")

    # Define helper functions within the scope
    def _pct_diff(a, b):
        """Calculate percentage difference between a and b."""
        try:
            if a is None or b is None or abs(b) < 1e-9:
                return float('inf')
            return 100.0 * (a - b) / abs(b)
        except Exception:
            return float('inf')

    def _num(x):
        """Convert value to float, handling various formats and edge cases."""
        try:
            if x is None:
                return None
            if isinstance(x, (int, float)):
                if isinstance(x, float) and math.isnan(x):
                    return None
                return float(x)
            s = str(x).strip().replace(",", ".")
            if s == "":
                return None
            return float(s)
        except Exception:
            return None

    def _to_km(value):
        """Convert length value to km, assuming values > 1000 are in meters."""
        if value is None or pd.isna(value):
            return 0.0
        v = float(value)
        # Heuristic: lengths > 1000 are probably meters
        return v / 1000.0 if v > 1000 else v

    def _get_first_existing(row, *names):
        """Return the first non-NA value from the row using the given column names."""
        for name in names:
            if name in row and pd.notna(row[name]):
                return row[name]
        return None

    fixed_count = 0

    for result in matching_results:
        if not result.get('matched'):
            continue

        jao_id = result.get('jao_id')

        # Check if length ratio is extreme
        ratio = result.get('length_ratio')
        if ratio is None:
            # Calculate ratio if missing
            if result.get('path_length') and result.get('jao_length') and result['jao_length'] > 0:
                ratio = result['path_length'] / result['jao_length']

        # Skip if ratio is within normal range
        if ratio is None or (min_ratio_threshold <= ratio <= max_ratio_threshold):
            continue

        print(f"  JAO {jao_id} has extreme length ratio: {ratio:.3f}")

        # Get JAO parameters
        jao_rows = jao_gdf[jao_gdf['id'].astype(str) == str(jao_id)]
        if jao_rows.empty:
            print(f"  Error: JAO {jao_id} not found in jao_gdf")
            continue

        jao_row = jao_rows.iloc[0]

        # Get JAO parameters
        jao_r = _num(_get_first_existing(jao_row, 'r', 'R'))
        jao_x = _num(_get_first_existing(jao_row, 'x', 'X'))
        jao_b = _num(_get_first_existing(jao_row, 'b', 'B'))

        jao_r_per_km = _num(_get_first_existing(jao_row, 'R_per_km', 'r_per_km'))
        jao_x_per_km = _num(_get_first_existing(jao_row, 'X_per_km', 'x_per_km'))
        jao_b_per_km = _num(_get_first_existing(jao_row, 'B_per_km', 'b_per_km'))

        # If per-km not available, calculate from totals
        jao_length_km = _to_km(_get_first_existing(jao_row, 'length', 'length_km'))
        if jao_r_per_km is None and jao_r is not None and jao_length_km > 0:
            jao_r_per_km = jao_r / jao_length_km
        if jao_x_per_km is None and jao_x is not None and jao_length_km > 0:
            jao_x_per_km = jao_x / jao_length_km
        if jao_b_per_km is None and jao_b is not None and jao_length_km > 0:
            jao_b_per_km = jao_b / jao_length_km

        # Process each network line individually
        network_ids = result.get('network_ids', [])
        segments = []
        total_length_km = 0

        for nid in network_ids:
            net_rows = network_gdf[network_gdf['id'].astype(str) == str(nid)]
            if net_rows.empty:
                continue

            net_row = net_rows.iloc[0]
            length_km = _to_km(_get_first_existing(net_row, 'length_km', 'length'))
            total_length_km += length_km

            # Determine number of parallel circuits
            num_parallel = 1
            if 'circuits' in net_row and pd.notna(net_row['circuits']):
                num_parallel = max(1, int(net_row['circuits']))
            elif 'num_parallel' in net_row and pd.notna(net_row['num_parallel']):
                num_parallel = max(1, int(net_row['num_parallel']))

            # Get network parameters
            orig_r_km = _num(_get_first_existing(net_row, 'r_per_km', 'R_per_km'))
            orig_x_km = _num(_get_first_existing(net_row, 'x_per_km', 'X_per_km'))
            orig_b_km = _num(_get_first_existing(net_row, 'b_per_km', 'B_per_km'))

            orig_r = _num(_get_first_existing(net_row, 'r', 'R'))
            orig_x = _num(_get_first_existing(net_row, 'x', 'X'))
            orig_b = _num(_get_first_existing(net_row, 'b', 'B'))

            # Ensure totals and per-km are consistent
            if orig_r_km is None and orig_r is not None and length_km > 0:
                orig_r_km = orig_r / length_km
            if orig_x_km is None and orig_x is not None and length_km > 0:
                orig_x_km = orig_x / length_km
            if orig_b_km is None and orig_b is not None and length_km > 0:
                orig_b_km = orig_b / length_km

            if orig_r is None and orig_r_km is not None:
                orig_r = orig_r_km * length_km
            if orig_x is None and orig_x_km is not None:
                orig_x = orig_x_km * length_km
            if orig_b is None and orig_b_km is not None:
                orig_b = orig_b_km * length_km

            # Calculate per-circuit values
            orig_r_km_pc = orig_r_km * num_parallel if orig_r_km is not None else None
            orig_x_km_pc = orig_x_km * num_parallel if orig_x_km is not None else None
            orig_b_km_pc = orig_b_km / num_parallel if orig_b_km is not None else None

            # Calculate aggregate-equivalent values for allocation
            segment_ratio = length_km / total_length_km if total_length_km > 0 else 0

            # Calculate allocated parameters
            alloc_r = jao_r * segment_ratio if jao_r is not None else None
            alloc_x = jao_x * segment_ratio if jao_x is not None else None
            alloc_b = jao_b * segment_ratio if jao_b is not None else None

            # Calculate per-km values
            alloc_r_per_km = alloc_r / length_km if alloc_r is not None and length_km > 0 else None
            alloc_x_per_km = alloc_x / length_km if alloc_x is not None and length_km > 0 else None
            alloc_b_per_km = alloc_b / length_km if alloc_b is not None and length_km > 0 else None

            # Create segment data
            segment = {
                'network_id': nid,
                'length_km': length_km,
                'num_parallel': num_parallel,
                'segment_ratio': segment_ratio,

                # Allocated values
                'allocated_r': alloc_r,
                'allocated_x': alloc_x,
                'allocated_b': alloc_b,
                'allocated_r_per_km': alloc_r_per_km,
                'allocated_x_per_km': alloc_x_per_km,
                'allocated_b_per_km': alloc_b_per_km,

                # Original values
                'original_r': orig_r or 0.0,
                'original_x': orig_x or 0.0,
                'original_b': orig_b or 0.0,
                'original_r_per_km': orig_r_km or 0.0,
                'original_x_per_km': orig_x_km or 0.0,
                'original_b_per_km': orig_b_km or 0.0,

                # Per-circuit values
                'jao_r_per_km_pc': jao_r_per_km,
                'jao_x_per_km_pc': jao_x_per_km,
                'jao_b_per_km_pc': jao_b_per_km,
                'original_r_per_km_pc': orig_r_km_pc or 0.0,
                'original_x_per_km_pc': orig_x_km_pc or 0.0,
                'original_b_per_km_pc': orig_b_km_pc or 0.0,

                # Difference percentages
                'r_diff_percent_pc': _pct_diff(jao_r_per_km, orig_r_km_pc),
                'x_diff_percent_pc': _pct_diff(jao_x_per_km, orig_x_km_pc),
                'b_diff_percent_pc': _pct_diff(jao_b_per_km, orig_b_km_pc),
                'r_diff_percent': _pct_diff(alloc_r, orig_r),
                'x_diff_percent': _pct_diff(alloc_x, orig_x),
                'b_diff_percent': _pct_diff(alloc_b, orig_b),

                'allocation_status': f'Fixed extreme ratio ({ratio:.3f})'
            }

            segments.append(segment)

        # Replace the segments with our fixed ones
        result['matched_lines_data'] = segments

        # Set the coverage using the actual matched network length
        result['matched_km'] = total_length_km
        if jao_length_km and jao_length_km > 0:
            # For extreme ratios, we set a "normalized" coverage to prevent backfilling
            result['coverage_ratio'] = 1.0

        # Calculate allocated sums
        alloc_r_sum = sum(seg.get('allocated_r', 0.0) or 0.0 for seg in segments)
        alloc_x_sum = sum(seg.get('allocated_x', 0.0) or 0.0 for seg in segments)
        alloc_b_sum = sum(seg.get('allocated_b', 0.0) or 0.0 for seg in segments)

        result['allocated_r_sum'] = alloc_r_sum
        result['allocated_x_sum'] = alloc_x_sum
        result['allocated_b_sum'] = alloc_b_sum

        # Calculate residuals
        def residual_pct(total, alloc):
            if total is None or abs(total) < 1e-12:
                return 0.0
            return 100.0 * (total - alloc) / abs(total)

        result['residual_r_percent'] = residual_pct(jao_r, alloc_r_sum)
        result['residual_x_percent'] = residual_pct(jao_x, alloc_x_sum)
        result['residual_b_percent'] = residual_pct(jao_b, alloc_b_sum)

        # Mark as fixed
        result['_extreme_ratio_fixed'] = True

        fixed_count += 1

    print(f"Fixed {fixed_count} matches with extreme length ratios")
    return matching_results


def protect_fixed_allocations(matching_results):
    """Ensure fixed allocations don't get modified before visualization."""
    for res in matching_results:
        if res.get('_extreme_ratio_fixed'):
            # Re-verify coverage is set to 100%
            res['coverage_ratio'] = 1.0

            # Make sure original network IDs are preserved
            original_ids = set(res.get('network_ids', []))
            if 'matched_lines_data' in res:
                segment_ids = set(seg.get('network_id') for seg in res['matched_lines_data'])
                # If there's a mismatch, restore from segments
                if segment_ids != original_ids:
                    res['network_ids'] = list(segment_ids)
    return matching_results

def ensure_network_segment_tables(
        matching_results,
        jao_gdf,
        network_gdf,
        freq_hz: float = 50.0,
        export_rows: list | None = None,
):
    """
    Build per-result segment tables (when missing) from `network_gdf`, allocate JAO totals
    by length ratio, compute per-circuit comparisons, and collect rows for exporting
    updated per-km parameters for the network.

    Returns
    -------
    matching_results, export_rows
    """
    import math
    import pandas as pd


    def _num(x):
        try:
            if x is None:
                return None
            if isinstance(x, (int, float)):
                if isinstance(x, float) and math.isnan(x):
                    return None
                return float(x)
            s = str(x).strip().replace(",", ".")
            if s == "":
                return None
            return float(s)
        except Exception:
            return None

    def _safe_div(a, b):
        a = _num(a);
        b = _num(b)
        if a is None or b is None or b == 0:
            return None
        return a / b

    def _length_km_from_row(row):
        # Prefer explicit length_km or length
        if 'length_km' in row and pd.notna(row['length_km']):
            v = _num(row['length_km'])
            if v and v > 0:
                return v
        if 'length' in row and pd.notna(row['length']):
            v = _num(row['length'])
            if v and v > 0:
                return v
        # Fallback to geometry length (assumes meters if big)
        try:
            geom = row.get('geometry')
            if geom is not None:
                L = float(geom.length)
                return L / 1000.0 if L > 10_000 else L
        except Exception:
            pass
        return None

    def _per_km_from_row(row):
        """
        Return (r_per_km, x_per_km, b_per_km) for one network row, if possible.

        Priority:
          0) explicit per-km fields
          1) R0/L0/C0 (mH/km, nF/km -> ohm/km & S/km using 50 Hz)
          2) per-unit (r_pu/x_pu/b_pu + base Zb) then divide by length
          3) totals / length
          4) coarse voltage defaults (from the German table you quoted)
        """
        # 0) explicit per-km
        rpk = _num(row.get('R_per_km')) or _num(row.get('r_per_km')) or _num(row.get('R_km')) or _num(row.get('r_km'))
        xpk = _num(row.get('X_per_km')) or _num(row.get('x_per_km')) or _num(row.get('X_km')) or _num(row.get('x_km'))
        bpk = _num(row.get('B_per_km')) or _num(row.get('b_per_km')) or _num(row.get('B_km')) or _num(row.get('b_km'))
        if rpk is not None or xpk is not None or bpk is not None:
            return rpk, xpk, bpk

        Lkm = _length_km_from_row(row)

        # 1) R0/L0/C0
        R0 = _num(row.get('R0'))
        L0_mH = _num(row.get('L0')) or _num(row.get('L'))  # mH/km
        C0_nF = _num(row.get('C0')) or _num(row.get('C'))  # nF/km
        if R0 is not None or L0_mH is not None or C0_nF is not None:
            rpk = R0 if R0 is not None else None
            xpk = (2 * math.pi * freq_hz * (L0_mH * 1e-3)) if L0_mH is not None else None
            bpk = (2 * math.pi * freq_hz * (C0_nF * 1e-9)) if C0_nF is not None else None
            return rpk, xpk, bpk

        # 2) per-unit
        rpu = _num(row.get('r_pu')) or _num(row.get('r_pu_eff'))
        xpu = _num(row.get('x_pu')) or _num(row.get('x_pu_eff'))
        bpu = _num(row.get('b_pu'))  # often no *_eff for b
        Sbase = _num(row.get('s_nom')) or _num(row.get('s_nom_opt')) or _num(row.get('s_nom_max'))
        VkV = _num(row.get('v_nom'))
        if Sbase is not None and VkV is not None and Lkm and Lkm > 0:
            Zb = (VkV ** 2) / Sbase  # ohm
            rpk = (rpu * Zb) / Lkm if rpu is not None else None
            xpk = (xpu * Zb) / Lkm if xpu is not None else None
            bpk = (bpu / Zb) / Lkm if bpu is not None else None  # since Y_pu = Y * Zb -> B = b_pu / Zb
            if rpk is not None or xpk is not None or bpk is not None:
                return rpk, xpk, bpk

        # 3) totals / length
        rt = _num(row.get('r')) or _num(row.get('R'))
        xt = _num(row.get('x')) or _num(row.get('X'))
        bt = _num(row.get('b')) or _num(row.get('B'))
        if Lkm and Lkm > 0 and (rt is not None or xt is not None or bt is not None):
            return (
                (rt / Lkm) if rt is not None else None,
                (xt / Lkm) if xt is not None else None,
                (bt / Lkm) if bt is not None else None
            )

        # 4) coarse defaults by voltage (German table)
        v = _num(row.get('v_nom'))
        defaults = {
            110: (0.109, 2 * math.pi * freq_hz * (1.2e-3), 2 * math.pi * freq_hz * (9.5e-9)),
            220: (0.109, 2 * math.pi * freq_hz * (1.0e-3), 2 * math.pi * freq_hz * (11e-9)),
            380: (0.028, 2 * math.pi * freq_hz * (0.8e-3), 2 * math.pi * freq_hz * (14e-9)),
            400: (0.028, 2 * math.pi * freq_hz * (0.8e-3), 2 * math.pi * freq_hz * (14e-9)),
        }
        return defaults.get(v, (None, None, None))

    def _pct_diff(a, b):
        a = _num(a);
        b = _num(b)
        if a is None or b is None or b == 0:
            return float('inf')
        return 100.0 * (a - b) / b

    # ----------------------------------------------------------------

    if 'id' not in network_gdf.columns:
        raise KeyError("network_gdf is missing 'id' column.")

        # quick lookup by id
    net_by_id = {str(row['id']): row for _, row in network_gdf.iterrows()}

    # make a list to collect export rows if none passed in
    created_export_list = False
    if export_rows is None:
        export_rows = []
        created_export_list = True

    updated = 0

    # Process fixed extreme ratio matches first (just for export rows)
    for res in matching_results:
        if res.get('_extreme_ratio_fixed') and res.get('matched_lines_data'):
            # Collect export rows from already fixed matches
            for seg in res.get('matched_lines_data', []):
                if all(k in seg for k in ['network_id', 'length_km']):
                    export_rows.append({
                        'network_id': seg['network_id'],
                        'seg_len_km': seg['length_km'],
                        'jao_r_km_pc': seg.get('jao_r_per_km_pc'),
                        'jao_x_km_pc': seg.get('jao_x_per_km_pc'),
                        'jao_b_km_pc': seg.get('jao_b_per_km_pc'),
                    })

    # Then process remaining results
    for res in matching_results:
        # Skip if not matched or already fixed
        if not res.get('matched') or res.get('_extreme_ratio_fixed'):
            continue

        # Skip if explicitly marked for no backfilling
        if res.get('_no_backfill'):
            if 'matched_lines_data' not in res or not res['matched_lines_data']:
                # Build minimal segments table just for what was explicitly matched
                segs = []
                total_len_km = 0.0
                for nid in res.get('network_ids') or []:
                    row = net_by_id.get(nid)
                    if row is None:
                        continue

                    Lkm = _length_km_from_row(row)
                    if not Lkm or Lkm <= 0:
                        continue

                    num_par = int(
                        (_num(row.get('circuits')) if 'circuits' in row else None)
                        or (_num(row.get('num_parallel')) if 'num_parallel' in row else None)
                        or 1
                    )
                    rpk, xpk, bpk = _per_km_from_row(row)

                    segs.append({
                        'network_id': nid,
                        'length_km': Lkm,
                        'num_parallel': num_par,
                        'original_r_per_km': rpk,
                        'original_x_per_km': xpk,
                        'original_b_per_km': bpk,
                        'original_r': rpk * Lkm if rpk is not None else None,
                        'original_x': xpk * Lkm if xpk is not None else None,
                        'original_b': bpk * Lkm if bpk is not None else None,
                        'allocation_status': 'No-backfill JAO'
                    })
                    total_len_km += Lkm

                # Set JAO parameters
                jao_id = res.get('jao_id')
                jao_rows = jao_gdf[jao_gdf['id'].astype(str) == str(jao_id)]
                if not jao_rows.empty:
                    jao_row = jao_rows.iloc[0]
                    jao_r_pk = _num(jao_row.get('R_per_km') or jao_row.get('r_per_km'))
                    jao_x_pk = _num(jao_row.get('X_per_km') or jao_row.get('x_per_km'))
                    jao_b_pk = _num(jao_row.get('B_per_km') or jao_row.get('b_per_km'))
                    jao_len_km = _length_km_from_row(jao_row)

                    if jao_len_km and jao_len_km > 0:
                        # Directly allocate JAO parameters in proportion to segment length
                        for seg in segs:
                            seg_ratio = seg['length_km'] / jao_len_km
                            if jao_r_pk is not None:
                                seg['jao_r_per_km_pc'] = jao_r_pk
                                seg['allocated_r_per_km'] = jao_r_pk / seg['num_parallel']
                                seg['allocated_r'] = seg['allocated_r_per_km'] * seg['length_km']
                            if jao_x_pk is not None:
                                seg['jao_x_per_km_pc'] = jao_x_pk
                                seg['allocated_x_per_km'] = jao_x_pk / seg['num_parallel']
                                seg['allocated_x'] = seg['allocated_x_per_km'] * seg['length_km']
                            if jao_b_pk is not None:
                                seg['jao_b_per_km_pc'] = jao_b_pk
                                seg['allocated_b_per_km'] = jao_b_pk * seg['num_parallel']
                                seg['allocated_b'] = seg['allocated_b_per_km'] * seg['length_km']

                # Store the segments and update coverage
                res['matched_lines_data'] = segs
                res['matched_km'] = total_len_km
                if Ljao_km := _num(res.get('jao_length_km')):
                    res['coverage_ratio'] = total_len_km / Ljao_km

                # Add to export rows
                for seg in segs:
                    if all(k in seg for k in
                           ['network_id', 'length_km', 'jao_r_per_km_pc', 'jao_x_per_km_pc', 'jao_b_per_km_pc']):
                        export_rows.append({
                            'network_id': seg['network_id'],
                            'seg_len_km': seg['length_km'],
                            'jao_r_km_pc': seg.get('jao_r_per_km_pc'),
                            'jao_x_km_pc': seg.get('jao_x_per_km_pc'),
                            'jao_b_km_pc': seg.get('jao_b_per_km_pc'),
                        })
            continue

        # If already has segments, only backfill coverage if missing.
        if res.get('matched_lines_data'):
            if res.get('coverage_ratio') is None:
                total_km = sum((_num(s.get('length_km')) or 0.0) for s in res['matched_lines_data'])
                Ljao = _num(res.get('jao_length_km'))
                if Ljao is None:
                    Lj = _num(res.get('jao_length'))
                    if Lj is not None:
                        # guess meters if large
                        Ljao = Lj / 1000.0 if Lj > 1000 else Lj
                if Ljao and Ljao > 0:
                    res['matched_km'] = total_km
                    res['coverage_ratio'] = max(0.0, min(1.0, total_km / Ljao))
            continue

        # collect network segments for this result
        nids = [str(n) for n in (res.get('network_ids') or []) if n is not None]
        if not nids:
            continue

        segs = []
        total_len_km = 0.0
        for nid in nids:
            row = net_by_id.get(nid)
            if row is None:
                continue

            Lkm = _length_km_from_row(row)
            if not Lkm or Lkm <= 0:
                continue

            num_par = int(
                (_num(row.get('circuits')) if 'circuits' in row else None)
                or (_num(row.get('num_parallel')) if 'num_parallel' in row else None)
                or 1
            )
            rpk, xpk, bpk = _per_km_from_row(row)

            segs.append({
                'network_id': nid,
                'length_km': Lkm,
                'num_parallel': num_par,
                'r_per_km_net': rpk,
                'x_per_km_net': xpk,
                'b_per_km_net': bpk,
            })
            total_len_km += Lkm

        if not segs:
            continue

        # -------- JAO totals & per-km (per-circuit) --------
        jao_r = _num(res.get('jao_r'))
        jao_x = _num(res.get('jao_x'))
        jao_b = _num(res.get('jao_b'))

        Ljao_km = _num(res.get('jao_length_km'))
        if Ljao_km is None:
            Lj = _num(res.get('jao_length'))
            if Lj is not None:
                Ljao_km = Lj / 1000.0 if Lj > 1000 else Lj

        # per-km (per-circuit) at the JAO line level
        jao_r_pk = _num(res.get('jao_r_per_km')) or _safe_div(jao_r, Ljao_km)
        jao_x_pk = _num(res.get('jao_x_per_km')) or _safe_div(jao_x, Ljao_km)
        jao_b_pk = _num(res.get('jao_b_per_km')) or _safe_div(jao_b, Ljao_km)

        # these are already per circuit
        jao_r_pk_pc = jao_r_pk
        jao_x_pk_pc = jao_x_pk
        jao_b_pk_pc = jao_b_pk

        # -------- allocate & compare --------
        out_rows = []
        alloc_r_sum = alloc_x_sum = alloc_b_sum = 0.0

        for s in segs:
            seg_len = s['length_km']
            ratio = (seg_len / total_len_km) if total_len_km > 0 else 0.0

            # allocate totals by length share
            alloc_r = jao_r * ratio if jao_r is not None else None
            alloc_x = jao_x * ratio if jao_x is not None else None
            alloc_b = jao_b * ratio if jao_b is not None else None

            if alloc_r is not None: alloc_r_sum += alloc_r
            if alloc_x is not None: alloc_x_sum += alloc_x
            if alloc_b is not None: alloc_b_sum += alloc_b

            rpk, xpk, bpk = s['r_per_km_net'], s['x_per_km_net'], s['b_per_km_net']
            orig_r = rpk * seg_len if rpk is not None else None
            orig_x = xpk * seg_len if xpk is not None else None
            orig_b = bpk * seg_len if bpk is not None else None

            # per-circuit equivalents for the network side
            num_par = int(s.get('num_parallel', 1) or 1)
            orig_r_km_pc = (rpk * num_par) if rpk is not None else None
            orig_x_km_pc = (xpk * num_par) if xpk is not None else None
            orig_b_km_pc = (bpk / num_par) if (bpk is not None and num_par) else None

            # diffs
            r_diff_pc = _pct_diff(jao_r_pk_pc, orig_r_km_pc)
            x_diff_pc = _pct_diff(jao_x_pk_pc, orig_x_km_pc)
            b_diff_pc = _pct_diff(jao_b_pk_pc, orig_b_km_pc)

            r_diff_agg = _pct_diff(alloc_r, orig_r)
            x_diff_agg = _pct_diff(alloc_x, orig_x)
            b_diff_agg = _pct_diff(alloc_b, orig_b)

            out_rows.append({
                'network_id': s['network_id'],
                'length_km': seg_len,
                'num_parallel': num_par,
                'segment_ratio': ratio,

                # allocated totals & per-km
                'allocated_r': alloc_r or 0.0,
                'allocated_x': alloc_x or 0.0,
                'allocated_b': alloc_b or 0.0,
                'allocated_r_per_km': _safe_div(alloc_r, seg_len) or 0.0,
                'allocated_x_per_km': _safe_div(alloc_x, seg_len) or 0.0,
                'allocated_b_per_km': _safe_div(alloc_b, seg_len) or 0.0,

                # original totals & per-km
                'original_r': orig_r or 0.0,
                'original_x': orig_x or 0.0,
                'original_b': orig_b or 0.0,
                'original_r_per_km': rpk or 0.0,
                'original_x_per_km': xpk or 0.0,
                'original_b_per_km': bpk or 0.0,

                # per-circuit comparison
                'original_r_per_km_pc': orig_r_km_pc or 0.0,
                'original_x_per_km_pc': orig_x_km_pc or 0.0,
                'original_b_per_km_pc': orig_b_km_pc or 0.0,
                'jao_r_per_km_pc': jao_r_pk_pc or 0.0,
                'jao_x_per_km_pc': jao_x_pk_pc or 0.0,
                'jao_b_per_km_pc': jao_b_pk_pc or 0.0,

                # diffs
                'r_diff_percent_pc': r_diff_pc,
                'x_diff_percent_pc': x_diff_pc,
                'b_diff_percent_pc': b_diff_pc,
                'r_diff_percent': r_diff_agg,
                'x_diff_percent': x_diff_agg,
                'b_diff_percent': b_diff_agg,

                'allocation_status': 'Filled by fallback',
            })

            # ---- collect export row (per segment) ----
            export_rows.append({
                'network_id': s['network_id'],
                'seg_len_km': seg_len,
                # per-km **per circuit** derived from allocated totals
                'jao_r_km_pc': jao_r_pk_pc,
                'jao_x_km_pc': jao_x_pk_pc,
                'jao_b_km_pc': jao_b_pk_pc,
            })

        # store outputs on the result
        res['matched_lines_data'] = out_rows
        res['allocated_r_sum'] = alloc_r_sum if jao_r is not None else 0.0
        res['allocated_x_sum'] = alloc_x_sum if jao_x is not None else 0.0
        res['allocated_b_sum'] = alloc_b_sum if jao_b is not None else 0.0

        # residuals
        if jao_r is not None and alloc_r_sum is not None and jao_r != 0:
            res['residual_r_percent'] = 100.0 * (jao_r - alloc_r_sum) / jao_r
        if jao_x is not None and alloc_x_sum is not None and jao_x != 0:
            res['residual_x_percent'] = 100.0 * (jao_x - alloc_x_sum) / jao_x
        if jao_b is not None and alloc_b_sum is not None and jao_b != 0:
            res['residual_b_percent'] = 100.0 * (jao_b - alloc_b_sum) / jao_b

        # coverage
        if Ljao_km and Ljao_km > 0:
            res['matched_km'] = total_len_km
            res['coverage_ratio'] = max(0.0, min(1.0, total_len_km / Ljao_km))
        else:
            res['matched_km'] = total_len_km
            res['coverage_ratio'] = 0.0

        updated += 1

    print(f"ensure_network_segment_tables: filled segment tables for {updated} matched results lacking them.")
    return matching_results, export_rows


# Build network graph from network lines
def build_network_graph(net_gdf):
    import networkx as nx
    G = nx.Graph()

    for idx, row in net_gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue

        def _add_node(pt, tag):
            node_id = f"node_{idx}_{tag}"
            if node_id not in G:
                lon, lat = float(pt[0]), float(pt[1])
                G.add_node(node_id,
                           x=lon,
                           y=lat,
                           pos=(lon, lat))  # ← add this
            return node_id

        if geom.geom_type == "LineString":
            n_start = _add_node(geom.coords[0],  "start")
            n_end   = _add_node(geom.coords[-1], "end")
            length_deg = geom.length              # still degrees here
            G.add_edge(n_start, n_end,
                       id=row['id'],
                       voltage=row['v_nom'],
                       weight=length_deg)          # <- will be converted later

        elif geom.geom_type == "MultiLineString":
            for seg in geom.geoms:
                n_start = _add_node(seg.coords[0],  "start")
                n_end   = _add_node(seg.coords[-1], "end")
                length_deg = seg.length
                G.add_edge(n_start, n_end,
                           id=row['id'],
                           voltage=row['v_nom'],
                           weight=length_deg)

    return G

from sklearn.cluster import DBSCAN
import numpy as np

def add_station_hubs_to_graph(G, network_gdf, radius_m=350, by_voltage=True):
    """Create a hub node for each substation endpoint cluster (per voltage)
    and connect all member endpoints to the hub with zero-weight connector edges."""
    if G.graph.get("hubs_added"):
        return G

    # collect endpoint nodes present in G
    endpoints = []  # (node_id, x, y, vcat)
    for idx, row in network_gdf.iterrows():
        v = int(row['v_nom'])
        vcat = 400 if v in (380, 400) else v
        for tag in ('start', 'end'):
            nid = f"node_{idx}_{tag}"
            if nid in G:
                x = float(G.nodes[nid]['x']); y = float(G.nodes[nid]['y'])
                endpoints.append((nid, x, y, vcat if by_voltage else None))

    if not endpoints:
        G.graph["hubs_added"] = True
        return G

    coords = np.array([(e[1], e[2]) for e in endpoints])
    avg_lat = float(coords[:,1].mean())
    m_per_deg = 111_111*np.cos(np.radians(abs(avg_lat)))
    eps = radius_m / m_per_deg

    # cluster by position (and by voltage if requested)
    labels = DBSCAN(eps=eps, min_samples=2).fit(coords).labels_

    # bucket endpoints -> (cluster, vcat)
    groups = {}
    for (nid, x, y, vcat), lab in zip(endpoints, labels):
        if lab < 0:  # noise
            continue
        key = (lab, vcat)
        groups.setdefault(key, []).append((nid, x, y))

    # create hubs
    for (lab, vcat), items in groups.items():
        cx = float(np.mean([x for _, x, _ in items]))
        cy = float(np.mean([y for _, _, y in items]))
        hub_id = f"hub_{lab}_{vcat}"
        if hub_id not in G:
            G.add_node(hub_id, x=cx, y=cy, pos=(cx, cy), hub=True)
        for nid, _, _ in items:
            if not G.has_edge(nid, hub_id):
                G.add_edge(nid, hub_id, weight=1e-6, connector=True, station=True)

    G.graph["hubs_added"] = True
    return G



from shapely.geometry import LineString
from shapely.ops import linemerge

def _norm(ls: LineString, tol=1e-9):
    # → a tuple of rounded coordinate pairs, ordered from lower -> higher id
    coords = list(ls.coords)
    # normalise direction (start-point should be "smaller")
    if coords[0] > coords[-1]:
        coords = coords[::-1]
    return tuple((round(x, 7), round(y, 7)) for x, y in coords)

def identify_duplicate_jao_lines(gdf):
    groups = {}
    jao_to_group = {}
    for _, row in gdf.iterrows():
        key = _norm(row.geometry)
        groups.setdefault(key, []).append(str(row["id"]))
    # keep only true duplicates (len>1)
    dup_groups = {k: v for k, v in groups.items() if len(v) > 1}
    for k, ids in dup_groups.items():
        for i in ids:
            jao_to_group[i] = k
    print(f"Found {len(dup_groups)} duplicate-geometry groups")
    return jao_to_group, dup_groups



def extract_path_details(G, path, network_gdf):
    """Helper function to extract network IDs and calculate path length from a path."""
    network_ids = []
    unique_ids = set()
    path_edges = []

    for i in range(len(path) - 1):
        edge_data = G.get_edge_data(path[i], path[i + 1])

        if edge_data and 'id' in edge_data and not edge_data.get('connector', False):
            if edge_data['id'] not in unique_ids:
                network_ids.append(str(edge_data['id']))
                unique_ids.add(edge_data['id'])

            path_edges.append((path[i], path[i + 1], edge_data))

    # If no network lines in the path, return empty
    if not network_ids:
        return [], 0, []

    # Calculate path length
    path_length = 0
    for network_id in network_ids:
        network_line = network_gdf[network_gdf['id'].astype(str) == network_id]
        if not network_line.empty:
            if 'length' in network_line.iloc[0] and network_line.iloc[0]['length']:
                line_length = float(network_line.iloc[0]['length']) * 1000  # Convert to meters
            else:
                line_length = float(network_line.iloc[0].geometry.length)
            path_length += line_length

    return network_ids, path_length, path_edges


# Helper function for proper length calculation
def calculate_length_meters(geometry):
    """Calculate length in meters for a geometry, accounting for coordinate system."""
    if geometry is None:
        return 0

    # If the geometry uses geographic coordinates (lon/lat)
    if isinstance(geometry, (LineString, MultiLineString)):
        # Get centroid latitude for conversion
        centroid_lat = geometry.centroid.y
        # Approximate meters per degree at this latitude
        meters_per_degree = 111111 * np.cos(np.radians(abs(centroid_lat)))
        # Convert length from degrees to meters
        return float(geometry.length) * meters_per_degree
    else:
        # For other coordinate systems, just return the length
        return float(geometry.length)



def _subgraph_in_corridor(G, network_gdf, jao_geom, km):
    buf = jao_geom.buffer(km*1000 / _meters_per_degree(jao_geom.centroid.y))

    keep = set(network_gdf.loc[
        network_gdf.geometry.apply(lambda g: g.intersects(buf)),
        'id'
    ].astype(str))

    H = G.copy()
    H.remove_edges_from([
        (u, v) for u, v, d in G.edges(data=True)
        if not d.get('connector') and str(d.get('id')) not in keep
    ])
    return H




# ── helper ───────────────────────────────────────────────────────────
def augment_graph_with_bridges(H, network_gdf, max_gap_m=150):
    """
    Inside an already-clipped subgraph H add zero-weight edges between
    any two dangling endpoints closer than max_gap_m.
    """
    import math, itertools
    coords = nx.get_node_attributes(H, 'pos')
    dangling = [n for n in H.nodes if H.degree[n] == 1]   # dead-ends only
    m_per_deg = lambda lat: 111111*math.cos(math.radians(abs(lat)))

    for a, b in itertools.combinations(dangling, 2):
        xa, ya = coords[a]; xb, yb = coords[b]
        mid_lat = 0.5*(ya+yb)
        dist_m  = math.hypot(xa-xb, ya-yb)*m_per_deg(mid_lat)
        if dist_m <= max_gap_m:
            H.add_edge(a, b, weight=0.0, connector=True, bridge=True)
    return H
# ─────────────────────────────────────────────────────────────────────
# -------------------------------------------------------------------
#  Helper: electrical similarity score
# -------------------------------------------------------------------



# ---------------------------------------------------------------------------
#  Graph-based matching with duplicate awareness + electrical scoring
# ---------------------------------------------------------------------------
def find_matching_network_lines_with_duplicates(
        jao_gdf,
        network_gdf,
        nearest_points_dict,
        G,
        duplicate_groups,
        *,
        max_reuse: int = 3,
        max_paths_to_try: int = 20,
        min_length_ratio: float = 0.7,
        max_length_ratio: float = 1.5,
        corridor_km_strict: float = 1.5,
        corridor_km_relaxed: float = 2.0,
        time_budget_s: float = 4.0
):
    """
    Path-based matching of JAO lines to network lines.

    *   honours duplicate JAO geometries (parallel circuits)
    *   caps global reuse of each network line (`max_reuse`)
    *   prefers electrical-parameter similarity
    *   first searches a strict corridor, then a relaxed one
    """
    import itertools, time, numpy as np
    import networkx as nx

    # -----------------------------------------------------------------------
    #  Helper tables
    # -----------------------------------------------------------------------
    dup_group_of = {
        str(j_id): wkt
        for wkt, ids in duplicate_groups.items()
        for j_id      in ids if len(ids) > 1
    }

    global_usage = {str(r['id']): 0 for _, r in network_gdf.iterrows()}
    group_used   = {}          # key = wkt, value = set(network ids)

    # ---  quick lookup of per-km, per-circuit params on network lines ------
    def _net_pc_lookup(gdf):
        cols = [c for c in ['r_km_pc', 'x_km_pc', 'b_km_pc', 'v_nom', 'length']
                if c in gdf.columns]
        df = gdf[['id'] + cols].copy()
        df['id'] = df['id'].astype(str)
        if 'length' not in df:
            df['length'] = 1.0
        df['w'] = pd.to_numeric(df['length'], errors='coerce').fillna(1.0)

        if df['id'].duplicated().any():                # collapse duplicates
            def _agg(grp):
                out = {}
                for c in ['r_km_pc', 'x_km_pc', 'b_km_pc']:
                    if c in grp:
                        m = grp[c].notna()
                        out[c] = np.average(grp.loc[m, c], weights=grp.loc[m, 'w']) if m.any() else None
                if 'v_nom' in grp:
                    out['v_nom'] = grp['v_nom'].mode().iat[0] if not grp['v_nom'].isna().all() else None
                return pd.Series(out)
            df = df.groupby('id', as_index=False).apply(_agg).reset_index(drop=True)
        return df.set_index('id')[['r_km_pc', 'x_km_pc', 'b_km_pc', 'v_nom']].to_dict('index')

    net_pc = _net_pc_lookup(network_gdf)

    def _same_voltage(jv, nv):
        return (jv == 220 and nv == 220) \
            or (jv in (380, 400) and nv in (380, 400))

    def _jao_per_km(row):
        return (_num(row.get('R_per_km') or row.get('r_per_km')),
                _num(row.get('X_per_km') or row.get('x_per_km')),
                _num(row.get('B_per_km') or row.get('b_per_km')))

    def _electrical_score(net_ids, jao_rkm, jao_xkm, jao_bkm):
        """
        0‒1 score based on how close the length-weighted per-circuit
        R/X/B values of the candidate path are to the JAO ones.
        Uses the enclosing-scope `network_gdf` and `net_pc`.
        """
        if jao_rkm is jao_xkm is jao_bkm is None:
            return 0.5  # neutral if JAO has no params

        # collect weights and per-km values
        lengths, r_vals, x_vals, b_vals = [], [], [], []
        for nid in net_ids:
            row = network_gdf[network_gdf['id'].astype(str) == str(nid)]
            if row.empty:
                continue
            lengths.append(
                _to_km(_get_first_existing(row.iloc[0], 'length_km', 'length')) or 0.0
            )
            p = net_pc.get(str(nid), {})
            r_vals.append(p.get('r_km_pc')); x_vals.append(p.get('x_km_pc')); b_vals.append(p.get('b_km_pc'))

        def _wavg(vals, ws):
            """Length--weighted average that ignores missing / zero weights."""
            num = den = 0.0
            for v, w_ in zip(vals, ws):
                if v is None or w_ is None or w_ <= 0:
                    continue
                num += float(v) * float(w_)
                den += float(w_)
            return None if den == 0 else num / den

        r_w = _wavg(r_vals, lengths)
        x_w = _wavg(x_vals, lengths)
        b_w = _wavg(b_vals, lengths)

        def _rel_err(a, b):
            if a is None or b is None or a <= 0 or b <= 0:
                return None
            return abs(a - b) / a

        errs = [e for e in (_rel_err(jao_rkm, r_w),
                            _rel_err(jao_xkm, x_w),
                            _rel_err(jao_bkm, b_w)) if e is not None]

        if not errs:
            return 0.5  # nothing comparable
        avg_err = sum(errs) / len(errs)
        return max(0.0, 1.0 - min(1.0, avg_err))  # clamp to [0,1]

    # -----------------------------------------------------------------------
    #  Iterate over JAO lines (deterministic order)
    # -----------------------------------------------------------------------
    results = []
    for idx, row in sorted(jao_gdf.iterrows(), key=lambda it: str(it[1]['id'])):
        jao_id   = str(row['id'])
        jao_v    = _safe_int(row['v_nom'])
        # ── new ── prefer the CSV, else geometry ──
        if pd.notna(row.get('length')) and row['length'] > 0:
            jao_len = float(row['length']) * 1000  # km → m
        else:
            jao_len = calculate_length_meters(row.geometry)  # fallback

        rkm, xkm, bkm = _jao_per_km(row)

        group_key = dup_group_of.get(jao_id)
        group_used.setdefault(group_key, set())

        # -------------------------------------------------------------------
        #  Endpoint snapping info prepared earlier in `nearest_points_dict`
        # -------------------------------------------------------------------
        snp = nearest_points_dict.get(idx, {})
        sn, en = snp.get('start_nearest'), snp.get('end_nearest')
        if not sn or not en:                     # cannot form a path
            results.append({'jao_id': jao_id, 'matched': False})
            continue

        s_idx, s_pos = sn
        e_idx, e_pos = en
        start_node   = f'node_{s_idx}_{s_pos}'
        end_node     = f'node_{e_idx}_{e_pos}'
        if start_node == end_node:
            results.append({'jao_id': jao_id, 'matched': False})
            continue

        # -------------------------------------------------------------------
        #  Corridor-filtered graphs
        # -------------------------------------------------------------------
        jao_geom = row.geometry
        H_strict = _subgraph_in_corridor(G, network_gdf, jao_geom, corridor_km_strict)
        H_relax  = _subgraph_in_corridor(G, network_gdf, jao_geom, corridor_km_relaxed)

        # prune edges according to reuse / voltage / duplicate rules
        def _prune(graph, strict_voltage: bool):
            for u, v, data in list(graph.edges(data=True)):
                if data.get('connector'):
                    continue
                nid = str(data.get('id'))
                if global_usage.get(nid, 0) >= max_reuse:
                    graph.remove_edge(u, v)
                    continue
                if strict_voltage and not _same_voltage(jao_v, data.get('voltage')):
                    graph.remove_edge(u, v)
                    continue
                if group_key and nid in group_used[group_key]:
                    graph.remove_edge(u, v)

        _prune(H_strict, True)
        _prune(H_relax,  False)

        # -------------------------------------------------------------------
        #  Candidate path search  (strict → relaxed → alt endpoints)
        # -------------------------------------------------------------------
        def _gather_candidates(graph, label):
            """Return a list of candidate paths with a composite score."""
            cand, t0, seen = [], time.time(), 0

            try:
                for path in nx.shortest_simple_paths(graph,
                                                     start_node,
                                                     end_node,
                                                     weight='weight'):

                    if seen >= max_paths_to_try or (time.time() - t0) > time_budget_s:
                        break
                    seen += 1

                    net_ids, path_len, _ = extract_path_details(G, path, network_gdf)
                    if not net_ids or jao_len == 0:
                        continue

                    ratio = path_len / jao_len
                    if not (min_length_ratio <= ratio <= max_length_ratio):
                        continue

                    # ── individual scores ───────────────────────────────────────
                    esc = _electrical_score(net_ids, rkm, xkm, bkm)  # 0‒1
                    lsc = 1.0 - abs(ratio - 1.0)  # 1 when perfect
                    gsc = 1.0 / (1.0 + ratio)  # favours shorter detours

                    # safety – although all three are floats now
                    esc = 0.0 if esc is None else esc

                    score = 0.55 * esc + 0.35 * lsc + 0.10 * gsc

                    cand.append({
                        'path': path,
                        'network_ids': net_ids,
                        'path_len': path_len,
                        'ratio': ratio,
                        'score': score,
                        'label': label
                    })

            except nx.NetworkXNoPath:
                pass

            return cand

        candidates = _gather_candidates(H_strict, 'strict')
        if not candidates:
            candidates = _gather_candidates(H_relax, 'relaxed')

        if not candidates:        # try a few alt endpoints
            for sa in (snp.get('start_alternatives') or [])[:3]:
                for ea in (snp.get('end_alternatives') or [])[:3]:
                    alt_s = f'node_{sa[0]}_{sa[1]}'; alt_e = f'node_{ea[0]}_{ea[1]}'
                    if alt_s == alt_e:
                        continue
                    try:
                        path = next(nx.shortest_simple_paths(H_relax, alt_s, alt_e, weight='weight'))
                    except nx.NetworkXNoPath:
                        continue
                    net_ids, path_len, _ = extract_path_details(G, path, network_gdf)
                    if not net_ids or jao_len == 0:
                        continue
                    ratio = path_len / jao_len
                    if not (min_length_ratio <= ratio <= max_length_ratio):
                        continue
                    esc   = _electrical_score(net_ids, rkm, xkm, bkm)
                    lsc   = 1. - abs(ratio - 1.)
                    gsc   = 1. / (1. + ratio)
                    score = 0.55 * esc + 0.35 * lsc + 0.10 * gsc
                    assert score is not None and isinstance(score, float)

                    candidates.append({'path': path,
                                       'network_ids': net_ids,
                                       'path_len': path_len,
                                       'ratio': ratio,
                                       'score': score,
                                       'label': 'alt_endpoints'})

        if not candidates:                          # give up for this JAO
            results.append({'jao_id': jao_id, 'matched': False})
            continue

        # pick the best
        best = max(candidates, key=lambda c: c['score'])

        # mark usage
        for nid in best['network_ids']:
            global_usage[nid] += 1
            if group_key:
                group_used[group_key].add(nid)

        is_dup   = bool(group_key and jao_id != next(iter(group_used[group_key])))
        dup_of   = next(iter(group_used[group_key])) if is_dup else None
        qual_txt = ('Excellent' if abs(best['ratio']-1) <= 0.10
                    else 'Good' if abs(best['ratio']-1) <= 0.20
                    else 'Fair')

        res = {
            'jao_id': jao_id,
            'jao_name': str(row.get('NE_name', '')),
            'v_nom': int(jao_v) if jao_v is not None else None,

            'matched': True,
            'is_duplicate': is_dup,
            'duplicate_of': dup_of,

            # ensure ids are strings and de-duplicated
            'network_ids': list(dict.fromkeys([str(n) for n in best['network_ids']])),
            'path': [str(p) for p in best['path']],

            # keep the authoritative lengths you just computed
            'path_length': float(best['path_len']),  # meters
            'jao_length': float(jao_len),  # meters
            'length_ratio': float(best['path_len'] / jao_len) if jao_len > 0 else None,

            'match_quality': f"{qual_txt} (elec={best['score']:.2f}, {best['label']})",
            'is_geometric_match': False,

            # >>> THE SOFT LOCK <<<
            'is_path_based': True,
        }

        # initialize the path corridor used by later phases to guard appends
        _init_or_update_path_lock(res, network_gdf, offcorridor_m=300)

        results.append(res)

    return results




def debug_specific_jao_match(jao_gdf, network_gdf, matching_results, jao_id_to_debug="97",
                             target_network_ids=["Line_8160", "Line_30733", "Line_30181", "Line_17856"]):
    """
    Debug function to investigate why a specific JAO is not being matched correctly.
    """
    print(f"\n=== DEBUGGING JAO {jao_id_to_debug} MATCHING ===")

    # Find the JAO in the dataframe
    jao_rows = jao_gdf[jao_gdf['id'].astype(str) == jao_id_to_debug]
    if jao_rows.empty:
        print(f"ERROR: JAO {jao_id_to_debug} not found in dataset")
        return matching_results

    jao_row = jao_rows.iloc[0]
    jao_name = str(jao_row['NE_name'])
    jao_voltage = int(jao_row['v_nom'])
    jao_geom = jao_row.geometry
    jao_length = calculate_length_meters(jao_geom)

    print(f"JAO {jao_id_to_debug}: {jao_name}")
    print(f"Voltage: {jao_voltage} kV")
    print(f"Length: {jao_length / 1000:.2f} km")

    # Check if this JAO is already matched
    is_matched = False
    existing_match = None
    for result in matching_results:
        if result['jao_id'] == jao_id_to_debug and result['matched']:
            is_matched = True
            existing_match = result
            break

    if is_matched:
        print(f"JAO is currently matched with: {existing_match['network_ids']}")
        print(f"Match quality: {existing_match['match_quality']}")
        if 'is_parallel_circuit' in existing_match and existing_match['is_parallel_circuit']:
            print(f"Marked as parallel circuit")
            if 'parallel_to_jao' in existing_match:
                print(f"Parallel to JAO: {existing_match['parallel_to_jao']}")
    else:
        print("JAO is currently UNMATCHED")

    # Check target network lines
    print("\nChecking target network lines...")
    target_network_geoms = []
    target_network_lengths = 0

    for network_id in target_network_ids:
        network_rows = network_gdf[network_gdf['id'].astype(str) == network_id]
        if network_rows.empty:
            print(f"  Network line {network_id}: NOT FOUND in dataset")
            continue

        network_row = network_rows.iloc[0]
        network_geom = network_row.geometry
        network_voltage = int(network_row['v_nom'])
        network_length = calculate_length_meters(network_geom)
        target_network_geoms.append(network_geom)
        target_network_lengths += network_length

        # Check if already used in another match
        used_in = []
        for result in matching_results:
            if result['matched'] and 'network_ids' in result and network_id in result['network_ids']:
                used_in.append(result['jao_id'])

        used_status = f"Used by JAO(s): {', '.join(used_in)}" if used_in else "Not used in any match"

        print(f"  Network line {network_id}: {network_voltage} kV, {network_length / 1000:.2f} km - {used_status}")

    # Create a MultiLineString from all target network lines
    if target_network_geoms:
        from shapely.geometry import MultiLineString
        target_network_multi = MultiLineString(target_network_geoms)

        # Calculate coverage and Hausdorff distance
        avg_lat = (target_network_multi.centroid.y + jao_geom.centroid.y) / 2
        meters_per_degree = 111111 * np.cos(np.radians(abs(avg_lat)))

        # Create buffer around the JAO
        buffer_m = 200  # 200m buffer
        buffer_deg = buffer_m / meters_per_degree
        jao_buffer = jao_geom.buffer(buffer_deg)

        # Check how much of network is within this buffer
        network_in_buffer = target_network_multi.intersection(jao_buffer)
        network_coverage = network_in_buffer.length / target_network_multi.length if target_network_multi.length > 0 else 0

        # Check how much of JAO is covered by network buffer
        network_buffer = target_network_multi.buffer(buffer_deg)
        jao_in_network_buffer = jao_geom.intersection(network_buffer)
        jao_coverage = jao_in_network_buffer.length / jao_geom.length if jao_geom.length > 0 else 0

        # Calculate Hausdorff distance
        hausdorff_dist = jao_geom.hausdorff_distance(target_network_multi)
        hausdorff_meters = hausdorff_dist * meters_per_degree

        # Calculate length ratio
        length_ratio = target_network_lengths / jao_length if jao_length > 0 else float('inf')

        print(f"\nGeometric analysis of target network path:")
        print(f"  Total network length: {target_network_lengths / 1000:.2f} km")
        print(f"  Length ratio (network/jao): {length_ratio:.2f}")
        print(f"  Network coverage by JAO buffer: {network_coverage:.2f}")
        print(f"  JAO coverage by network buffer: {jao_coverage:.2f}")
        print(f"  Hausdorff distance: {hausdorff_meters:.1f} meters")

        # Check if this would pass our matching criteria
        match_quality = (
                network_coverage >= 0.7 and
                jao_coverage >= 0.7 and
                hausdorff_meters <= 1000 and
                0.8 <= length_ratio <= 1.3
        )

        print(f"  Would this match pass our criteria? {'YES' if match_quality else 'NO'}")

        # Find if this JAO is parallel to any other JAO
        print("\nChecking for parallel JAOs...")
        best_parallel = None
        best_parallel_score = 0

        for idx, row in jao_gdf.iterrows():
            other_id = str(row['id'])

            # Skip self
            if other_id == jao_id_to_debug:
                continue

            other_geom = row.geometry
            other_voltage = _safe_int(row['v_nom'])

            # Only consider same voltage
            if other_voltage != jao_voltage:
                continue

            # Check if parallel
            other_buffer = other_geom.buffer(buffer_deg)
            jao_in_other_buffer = jao_geom.intersection(other_buffer)
            coverage = jao_in_other_buffer.length / jao_geom.length if jao_geom.length > 0 else 0

            # Calculate Hausdorff distance
            try:
                h_dist = jao_geom.hausdorff_distance(other_geom)
                h_meters = h_dist * meters_per_degree

                # Calculate similarity score
                if coverage >= 0.7 and h_meters <= 1000:
                    score = 0.6 * coverage + 0.4 * (1 - h_meters / 1000)

                    # Check if this JAO is matched
                    other_is_matched = False
                    for result in matching_results:
                        if result['jao_id'] == other_id and result['matched']:
                            other_is_matched = True
                            break

                    if score > best_parallel_score and other_is_matched:
                        best_parallel_score = score
                        best_parallel = {
                            'id': other_id,
                            'name': str(row['NE_name']),
                            'coverage': coverage,
                            'hausdorff_meters': h_meters,
                            'score': score
                        }
            except Exception as e:
                print(f"  Error calculating distance to JAO {other_id}: {e}")

        if best_parallel:
            print(f"  Found parallel JAO: {best_parallel['id']} ({best_parallel['name']})")
            print(f"  Coverage: {best_parallel['coverage']:.2f}")
            print(f"  Hausdorff distance: {best_parallel['hausdorff_meters']:.1f} meters")
            print(f"  Similarity score: {best_parallel['score']:.3f}")

            # Check if that JAO uses any of our target network lines
            for result in matching_results:
                if result['jao_id'] == best_parallel['id'] and result['matched']:
                    common_network_ids = set(result.get('network_ids', [])).intersection(set(target_network_ids))
                    if common_network_ids:
                        print(
                            f"  Parallel JAO uses {len(common_network_ids)} of our target network lines: {common_network_ids}")
                    else:
                        print(f"  Parallel JAO doesn't use any of our target network lines")

                    print(f"  Network lines used by parallel JAO: {result.get('network_ids', [])}")
                    break
        else:
            print("  No parallel JAO found")

    # Create a direct match with the target network lines
    print("\nCreating direct match with target network lines...")

    # First check if all target network lines exist
    all_exist = True
    for network_id in target_network_ids:
        network_rows = network_gdf[network_gdf['id'].astype(str) == network_id]
        if network_rows.empty:
            all_exist = False
            print(f"  Cannot create match - Network line {network_id} not found")

    if all_exist:
        # Remove existing match if it exists
        if is_matched:
            print(f"  Removing existing match for JAO {jao_id_to_debug}")
            matching_results = [r for r in matching_results if not (r['jao_id'] == jao_id_to_debug and r['matched'])]

        # Create new match
        new_match = {
            'jao_id': jao_id_to_debug,
            'jao_name': jao_name,
            'v_nom': jao_voltage,
            'matched': True,
            'is_duplicate': False,
            'is_parallel_circuit': True,
            'path': [],
            'network_ids': target_network_ids.copy(),
            'path_length': float(target_network_lengths),
            'jao_length': float(jao_length),
            'length_ratio': float(length_ratio),
            'match_quality': f'Parallel Circuit ({jao_voltage} kV) - Manual Override',
            'match_method': 'manual_override',
            'coverage_ratio': float(jao_coverage),
            'hausdorff_meters': float(hausdorff_meters)
        }

        matching_results.append(new_match)
        print(f"  Created manual match for JAO {jao_id_to_debug} with network lines: {target_network_ids}")

    return matching_results


def match_parallel_circuit_jao_with_network(
    matching_results,
    jao_gdf,
    network_gdf,
    G=None,
    nearest_points_dict=None,
    *,
    corridor_m_220=300,
    corridor_m_400=400,
    max_offcorridor_m=400,
    max_len_ratio_after=2.5
):
    """
    Enrich existing JAO→network matches with additional parallel network lines.
    Uses STRtree over network lines to prefilter. Respects 'is_path_based' soft lock.
    """
    import numpy as np
    from shapely.strtree import STRtree
    from shapely.ops import unary_union, linemerge

    # ---- helpers -----------------------------------------------------------
    def _same_voltage_local(a, b):
        try:
            A = 400 if int(a) in (380, 400) else int(a)
            B = 400 if int(b) in (380, 400) else int(b)
            return A == B
        except Exception:
            return False

    def _len_m(geom):
        # cache per-geom (by id elsewhere), but fast enough here
        try:
            lat = geom.centroid.y
            m_per_deg = _meters_per_degree(lat)
            return float(geom.length * m_per_deg)
        except Exception:
            return 0.0

    def _buffer_deg_for(geom, meters):
        lat = geom.centroid.y
        return float(meters / _meters_per_degree(lat))

    def _near_path_ok(result, candidate_geom):
        """Soft lock: if JAO result is path-based, only allow if within off-corridor
        and will not explode length_ratio."""
        if not result.get('is_path_based'):
            return True

        # Build current path geometry from current network_ids
        lines = []
        for nid in result.get('network_ids', []):
            rows = network_gdf[network_gdf['id'].astype(str) == str(nid)]
            if not rows.empty:
                lines.append(rows.iloc[0].geometry)
        if not lines:
            return True  # nothing to compare, be permissive

        from shapely.ops import unary_union, linemerge

        # build a geometry for the current path
        u = unary_union(lines)

        # Try to linemerge for a cleaner path, but fall back to union if it fails
        try:
            merged = linemerge(u)
            if getattr(merged, "is_empty", False):
                merged = u
        except Exception:
            merged = u

        # distance in meters
        lat = candidate_geom.centroid.y
        m_per_deg = _meters_per_degree(lat)
        dist_m = float(candidate_geom.distance(merged) * m_per_deg)

        if dist_m > max_offcorridor_m:
            return False

        # also ensure we wouldn't blow up the length ratio too far
        jao_len_m = float(result.get('jao_length') or 0.0)
        if jao_len_m > 0:
            cand_len_m = _len_m(candidate_geom)
            new_path_len = float(result.get('path_length') or 0.0) + cand_len_m
            if (new_path_len / jao_len_m) > max_len_ratio_after:
                return False

        return True

    # ---- index current usage ----------------------------------------------
    used_network_ids = set()
    net_geom_by_id = {}
    net_len_m_cache = {}
    for _, r in network_gdf.iterrows():
        nid = str(r['id'])
        g = r.geometry
        net_geom_by_id[nid] = g
        # lazy length cache
    for res in matching_results:
        for nid in res.get('network_ids', []) or []:
            used_network_ids.add(str(nid))

    from shapely.strtree import STRtree
    import numpy as np

    # Build STRtree over ALL network geoms (matched + unmatched) — keep the order stable
    nets = []
    net_idx_to_id = []
    for _, row in network_gdf.iterrows():
        nets.append(row.geometry)
        net_idx_to_id.append(str(row['id']))

    tree = STRtree(nets)

    # Map geometry object identity -> index (works when query returns geometries)
    geom_id_to_idx = {id(g): i for i, g in enumerate(nets)}

    def _hits_to_indices(hits):
        """
        Normalize STRtree.query(...) results to a list of integer indices into `nets`.
        Handles:
          - Shapely 2.x: returns a list of geometry objects
          - PyGEOS/older Shapely: returns a numpy array of integer indices
          - Mixed/object arrays
        """
        if hits is None:
            return []

        # If numpy array
        if isinstance(hits, np.ndarray):
            if hits.dtype.kind in ('i', 'u'):  # integer array
                return hits.astype(int).tolist()
            # object array -> could be geometries or ints
            out = []
            for h in hits:
                if h is None:
                    continue
                if isinstance(h, (int, np.integer)):
                    out.append(int(h))
                else:
                    idx = geom_id_to_idx.get(id(h))
                    if idx is not None:
                        out.append(idx)
            return out

        # If list/tuple or a single item
        seq = hits if isinstance(hits, (list, tuple)) else [hits]
        out = []
        for h in seq:
            if h is None:
                continue
            if isinstance(h, (int, np.integer)):
                out.append(int(h))
            else:
                idx = geom_id_to_idx.get(id(h))
                if idx is not None:
                    out.append(idx)
        return out

    # JAO geometry lookup
    jao_geom = {str(r['id']): r.geometry for _, r in jao_gdf.iterrows()}
    jao_vnom = {str(r['id']): _safe_int(r['v_nom']) for _, r in jao_gdf.iterrows()}
    jao_len_m = {}
    for j_id, g in jao_geom.items():
        try:
            # prefer CSV km if present
            row = jao_gdf[jao_gdf['id'].astype(str) == j_id].iloc[0]
            if 'length' in row and row['length'] and float(row['length']) > 0:
                jao_len_m[j_id] = float(row['length']) * 1000.0
            else:
                jao_len_m[j_id] = _len_m(g)
        except Exception:
            jao_len_m[j_id] = _len_m(g)

    # ---- main loop over matched JAOs --------------------------------------
    print("\n=== PARALLEL CIRCUIT MATCHING: JAO ↔ NETWORK (STRtree) ===")
    added = 0

    for res in matching_results:
        if not res.get('matched'):
            continue
        j_id = str(res['jao_id'])
        j_geom = jao_geom.get(j_id)
        if j_geom is None:
            continue
        j_v = jao_vnom.get(j_id, None)
        # choose corridor width
        cw = corridor_m_220 if j_v == 220 else corridor_m_400
        buf_deg = _buffer_deg_for(j_geom, cw)
        jbuf = j_geom.buffer(buf_deg)

        # candidates by spatial hit
        hits = tree.query(jbuf)
        cand_idxs = _hits_to_indices(hits)  # indices into `nets`
        cand_ids = [net_idx_to_id[i] for i in cand_idxs]

        # filter: voltage + not already in result + parallel-ish
        have_ids = set(res.get('network_ids', []) or [])
        for nid in cand_ids:
            if nid in have_ids:
                continue
            row = network_gdf[network_gdf['id'].astype(str) == nid]
            if row.empty:
                continue
            n_v = _safe_int(row.iloc[0]['v_nom'])
            if not _same_voltage_local(j_v, n_v):
                continue
            n_geom = net_geom_by_id[nid]

            # quick rejects: bbox far apart (~2× corridor)
            jminx, jminy, jmaxx, jmaxy = j_geom.bounds
            nminx, nminy, nmaxx, nmaxy = n_geom.bounds
            if (nmaxx < jminx - 2*buf_deg or nminx > jmaxx + 2*buf_deg or
                nmaxy < jminy - 2*buf_deg or nminy > jmaxy + 2*buf_deg):
                continue

            # overlap-based parallelism check (cheap)
            try:
                cov1 = calculate_geometry_coverage(j_geom, n_geom, buffer_meters=600)
                cov2 = calculate_geometry_coverage(n_geom, j_geom, buffer_meters=600)
                if ((cov1 + cov2) / 2.0) < 0.55:
                    continue
            except Exception:
                # fall back to simple buffer intersection
                if not jbuf.intersects(n_geom):
                    continue

            # path lock guard
            if not _near_path_ok(res, n_geom):
                continue

            # add and normalize
            res.setdefault('network_ids', []).append(nid)
            # mark as parallel if not already
            if not res.get('is_parallel_circuit', False):
                res['is_parallel_circuit'] = True
            # normalize path length + ratio if helper is available
            try:
                _normalize_network_ids_and_path_length(res, network_gdf)
            except Exception:
                # minimal fallback
                if 'path_length' not in res:
                    res['path_length'] = 0.0
                lat = n_geom.centroid.y
                res['path_length'] += float(n_geom.length * _meters_per_degree(lat))
                jl = float(res.get('jao_length') or jao_len_m.get(j_id) or 0.0)
                if jl > 0:
                    res['length_ratio'] = float(res['path_length'] / jl)

            added += 1

        # tag as path-based if it already has a path
        if res.get('path'):
            res['is_path_based'] = True

    print(f"Added {added} parallel network lines to JAO matches")
    return matching_results




# Helper functions

def is_geometry_parallel(geom1, geom2, buffer_meters=1000, min_coverage=0.4):
    """
    Check if two geometries are roughly parallel using a very relaxed buffer-based approach.

    Parameters:
    - geom1, geom2: The geometries to compare
    - buffer_meters: Buffer size in meters
    - min_coverage: Minimum coverage ratio to consider parallel

    Returns:
    - True if geometries are parallel, False otherwise
    """
    try:
        # Calculate buffer size in degrees based on latitude
        avg_lat = (geom1.centroid.y + geom2.centroid.y) / 2
        meters_per_degree = 111111 * np.cos(np.radians(abs(avg_lat)))
        buffer_deg = buffer_meters / meters_per_degree

        # Create buffers
        buffer1 = geom1.buffer(buffer_deg)

        # Check overlap
        intersection = geom2.intersection(buffer1)
        coverage = intersection.length / geom2.length if geom2.length > 0 else 0

        return coverage >= min_coverage
    except Exception as e:
        print(f"  Error checking if geometries are parallel: {e}")
        return False


def find_matching_network_lines(jao_geom, network_lines, jao_voltage,
                                buffer_meters=1500, min_coverage=0.3, max_lines=4):
    """
    Find network lines that match a JAO geometry with very relaxed constraints.

    Parameters:
    - jao_geom: JAO geometry
    - network_lines: List of network line dictionaries
    - jao_voltage: JAO voltage
    - buffer_meters: Buffer size in meters
    - min_coverage: Minimum coverage ratio
    - max_lines: Maximum number of lines to match

    Returns:
    - List of matching network lines
    """
    # Calculate buffer size in degrees
    avg_lat = jao_geom.centroid.y
    meters_per_degree = 111111 * np.cos(np.radians(abs(avg_lat)))
    buffer_deg = buffer_meters / meters_per_degree

    # Create buffer around JAO
    jao_buffer = jao_geom.buffer(buffer_deg)

    # Find matching network lines
    matches = []

    for line in network_lines:
        # Check voltage compatibility (very relaxed)
        voltage_match = False
        if (jao_voltage == 220 and line['voltage'] == 220) or \
                (jao_voltage == 400 and line['voltage'] in [380, 400]) or \
                (jao_voltage == 380 and line['voltage'] in [380, 400]):
            voltage_match = True

        if not voltage_match:
            continue

        # Check if line intersects buffer
        if jao_buffer.intersects(line['geometry']):
            # Calculate coverage
            intersection = line['geometry'].intersection(jao_buffer)
            coverage = intersection.length / line['geometry'].length if line['geometry'].length > 0 else 0

            if coverage >= min_coverage:
                matches.append({
                    'id': line['id'],
                    'geometry': line['geometry'],
                    'voltage': line['voltage'],
                    'length': line['length'],
                    'coverage': coverage
                })

    # Sort by coverage (highest first)
    matches.sort(key=lambda x: x['coverage'], reverse=True)

    # Take top matches (limited by max_lines)
    return matches[:max_lines]




# Helper functions for parallel and match scoring
def calculate_parallel_score(geom1, geom2, buffer_meters=300):
    """Calculate how parallel two geometries are to each other."""
    try:
        # Calculate buffer based on latitude
        avg_lat = (geom1.centroid.y + geom2.centroid.y) / 2
        meters_per_degree = 111111 * np.cos(np.radians(abs(avg_lat)))
        buffer_deg = buffer_meters / meters_per_degree

        # Create buffers
        buffer1 = geom1.buffer(buffer_deg)
        buffer2 = geom2.buffer(buffer_deg)

        # Calculate mutual overlap
        overlap1 = geom1.intersection(buffer2)
        overlap2 = geom2.intersection(buffer1)

        # Calculate overlap ratios
        ratio1 = overlap1.length / geom1.length if geom1.length > 0 else 0
        ratio2 = overlap2.length / geom2.length if geom2.length > 0 else 0

        # Average ratio (higher is better)
        avg_ratio = (ratio1 + ratio2) / 2

        # Calculate Hausdorff distance (lower is better)
        hausdorff_dist = geom1.hausdorff_distance(geom2)
        hausdorff_meters = hausdorff_dist * meters_per_degree

        # Normalize Hausdorff (1 when distance is 0, 0 when distance is large)
        norm_hausdorff = max(0, 1 - (hausdorff_meters / 1000))

        # Combined score (weighted average)
        score = 0.7 * avg_ratio + 0.3 * norm_hausdorff

        return score
    except Exception as e:
        print(f"Error calculating parallel score: {e}")
        return 0


def calculate_match_score(jao_geom, network_geom, jao_length, network_length):
    """Calculate match score between JAO and network geometries."""
    try:
        # Calculate parallelism score
        parallel_score = calculate_parallel_score(jao_geom, network_geom)

        # Calculate length ratio score (1 when equal, 0 when very different)
        length_ratio = network_length / jao_length if jao_length > 0 else float('inf')
        length_ratio_score = max(0, 1 - abs(length_ratio - 1))

        # Combined score (weighted)
        score = 0.7 * parallel_score + 0.3 * length_ratio_score

        return score
    except Exception as e:
        print(f"Error calculating match score: {e}")
        return 0

def match_remaining_lines_by_geometry(jao_gdf, network_gdf, matching_results, buffer_distance=0.005,
                                      snap_tolerance=300, angle_tolerance=30, min_dir_cos=0.866,
                                      min_length_ratio=0.3, max_length_ratio=3):
    """
    Apply a sophisticated geometric matching approach for JAO lines that aren't matched yet.
    Uses a combination of direction cosine, endpoint proximity, and overlap metrics.

    Parameters:
    - jao_gdf: GeoDataFrame with JAO lines
    - network_gdf: GeoDataFrame with network lines
    - matching_results: Existing matching results
    - buffer_distance: Buffer distance in degrees (for overlap)
    - snap_tolerance: Maximum distance in meters to connect endpoints
    - angle_tolerance: Maximum angle difference in degrees to consider lines aligned
    - min_dir_cos: Minimum direction cosine (cos of angle between lines)
    - min_length_ratio: Minimum acceptable ratio of network/jao length (default 0.3)
    - max_length_ratio: Maximum acceptable ratio of network/jao length (default 1.7)
    """
    import numpy as np
    from shapely.geometry import LineString, Point, MultiLineString
    from shapely.ops import linemerge, unary_union
    from scipy.spatial import cKDTree

    print("\nAttempting additional matches using advanced geometric approach...")
    print(f"Length ratio constraints: {min_length_ratio:.1f} to {max_length_ratio:.1f}")

    # Identify unmatched JAO lines
    matched_jao_ids = set(result['jao_id'] for result in matching_results if result['matched'])

    # Identify matched network lines
    matched_network_ids = set()
    for result in matching_results:
        if result['matched'] and 'network_ids' in result:
            for network_id in result['network_ids']:
                matched_network_ids.add(str(network_id))

    # Get unmatched JAO lines
    unmatched_jao_rows = []
    for idx, row in jao_gdf.iterrows():
        jao_id = str(row['id'])
        if jao_id not in matched_jao_ids:
            unmatched_jao_rows.append((idx, row))

    # Get unmatched network lines
    unmatched_network_gdf = network_gdf[~network_gdf['id'].astype(str).isin(matched_network_ids)].copy()

    print(f"Found {len(unmatched_jao_rows)} unmatched JAO lines to process")
    print(f"Found {len(unmatched_network_gdf)} unmatched network lines for matching")

    # Define helper functions
    def same_voltage(a, b):
        """Check if voltages are considered the same."""
        if a == 0 or b == 0:  # If either is unknown/zero, don't compare
            return True

        return (abs(a - b) <= 5 or  # 225 kV ≈ 220 kV, etc.
                abs(a / b - 1.0) <= 0.1 or  # Within 10% of each other
                {a, b} == {380, 400})  # explicit 380/400 rule

    def main_vec(ls):
        """Get the main vector of a LineString (direction from first to last point)."""
        if ls.geom_type == "MultiLineString":
            ls = max(ls.geoms, key=lambda p: p.length)
        c = np.asarray(ls.coords)
        v = c[-1] - c[0]
        n = np.linalg.norm(v)
        return v / n if n else v

    def dir_cos(a, b):
        """Calculate the direction cosine between two LineStrings."""
        return float(abs(np.dot(main_vec(a), main_vec(b))))

    def endpts(g):
        """Get endpoints of a geometry."""
        if g is None or g.is_empty:
            return []
        if g.geom_type == "LineString":
            c = list(g.coords)
            return [Point(c[0]), Point(c[-1])]
        out = []
        for part in g.geoms:
            c = list(part.coords)
            out += [Point(c[0]), Point(c[-1])]
        return out

    def endpts_inside(src, cand, width):
        """Check if candidate endpoints are inside buffer of source."""
        corr = src.buffer(width)
        return all(corr.contains(pt) for pt in endpts(cand))

    def overlap_length(a, b, tol=0.001):
        """Calculate the length of overlap between two geometries."""
        inter = b.intersection(a.buffer(tol))
        return inter.length

    def hausdorff_distance(a, b):
        """Calculate Hausdorff distance between two geometries."""
        return a.hausdorff_distance(b)

    # Function to calculate approximate meters per degree at a given latitude
    def meters_per_degree_at_lat(lat):
        return 111111 * np.cos(np.radians(lat))

    # Process each unmatched JAO line
    additional_matches = 0
    new_matches = []

    # Store matched network lines to prevent reuse across different JAO lines
    newly_matched_network_ids = set()

    # For each unmatched JAO line
    for idx, row in unmatched_jao_rows:
        jao_id = str(row['id'])
        jao_name = str(row['NE_name'])
        jao_voltage = _safe_int(row['v_nom'])
        jao_geometry = row.geometry

        # Calculate jao length in meters (approximate)
        meters_per_degree = meters_per_degree_at_lat(jao_geometry.centroid.y)
        jao_length_m = float(jao_geometry.length) * meters_per_degree

        print(f"\nProcessing unmatched JAO line {jao_id} ({jao_name}), length: {jao_length_m / 1000:.2f} km")

        # Find geometrically matching network lines
        candidate_matches = []

        # Get network lines with similar direction
        for net_idx, net_row in unmatched_network_gdf.iterrows():
            net_id = str(net_row['id'])

            # Skip already matched network lines (in this round)
            if net_id in newly_matched_network_ids:
                continue

            net_voltage = int(net_row['v_nom'])
            net_geometry = net_row.geometry

            # Check voltage matching - used as a multiplier for the score
            voltage_match = same_voltage(jao_voltage, net_voltage)
            voltage_factor = 1.0 if voltage_match else 0.5

            # Calculate direction cosine (alignment)
            cosine = dir_cos(jao_geometry, net_geometry)

            # Skip if lines are not remotely aligned
            if cosine < min_dir_cos:
                continue

            # Check proximity between endpoints
            jao_endpoints = endpts(jao_geometry)
            net_endpoints = endpts(net_geometry)

            min_endpoint_distance = float('inf')
            for d_pt in jao_endpoints:
                for n_pt in net_endpoints:
                    dist = d_pt.distance(n_pt)
                    min_endpoint_distance = min(min_endpoint_distance, dist)

            # Convert min_endpoint_distance to meters (approximate)
            # This is a rough conversion and depends on latitude
            min_endpoint_distance_m = min_endpoint_distance * meters_per_degree

            # Calculate overlap
            overlap = overlap_length(jao_geometry, net_geometry, tol=buffer_distance)
            overlap_ratio = overlap / min(jao_geometry.length, net_geometry.length)

            # Calculate Hausdorff distance
            h_dist = hausdorff_distance(jao_geometry, net_geometry)
            h_dist_m = h_dist * meters_per_degree

            # Calculate a composite score
            # Higher is better
            endpoint_score = 1.0 if min_endpoint_distance_m <= snap_tolerance else (
                0.5 if min_endpoint_distance_m <= 2 * snap_tolerance else 0.0)

            alignment_score = cosine
            overlap_score = overlap_ratio

            # Hausdorff score - inverse of distance (closer is better)
            max_h_dist = 1000  # meters
            hausdorff_score = max(0, 1.0 - (h_dist_m / max_h_dist))

            # Calculate combined score
            combined_score = (
                                     0.3 * endpoint_score +
                                     0.3 * alignment_score +
                                     0.2 * overlap_score +
                                     0.2 * hausdorff_score
                             ) * voltage_factor

            # Calculate network line length
            if 'length' in net_row and net_row['length']:
                net_length_m = float(net_row['length']) * 1000  # Assuming length is in km
            else:
                net_length_m = float(net_row.geometry.length) * meters_per_degree

            # Only consider if score is reasonable
            if combined_score > 0.5:
                candidate_matches.append({
                    'network_id': net_id,
                    'score': combined_score,
                    'voltage_match': voltage_match,
                    'dir_cos': cosine,
                    'endpoint_dist_m': min_endpoint_distance_m,
                    'overlap_ratio': overlap_ratio,
                    'hausdorff_dist_m': h_dist_m,
                    'length_m': net_length_m,
                    'idx': net_idx
                })

        # Sort by score (highest first)
        candidate_matches.sort(key=lambda x: x['score'], reverse=True)

        if candidate_matches:
            # Find best combinations of network lines
            best_combination = None
            best_combination_score = 0
            best_length_ratio = float('inf')  # Track how close to 1.0 the length ratio is

            # Try different combinations of top candidates (up to 5 at a time)
            # Start with 1 line, then try 2, etc.
            max_candidates = min(10, len(candidate_matches))

            for combo_size in range(1, min(4, max_candidates) + 1):
                # Generate all combinations of the specified size
                from itertools import combinations
                for combo in combinations(candidate_matches[:max_candidates], combo_size):
                    # Calculate total length and average score
                    combo_length = sum(c['length_m'] for c in combo)
                    length_ratio = combo_length / jao_length_m if jao_length_m > 0 else float('inf')
                    avg_score = sum(c['score'] for c in combo) / len(combo)

                    # Check if length ratio is within acceptable range
                    if min_length_ratio <= length_ratio <= max_length_ratio:
                        # Calculate how far from ideal (1.0) the ratio is
                        ratio_distance = abs(length_ratio - 1.0)

                        # Combine score and ratio_distance into a final score
                        # Weight more towards matching score, but consider length ratio
                        combo_score = avg_score * (1.0 - 0.3 * ratio_distance)

                        # Update best if this is better
                        if combo_score > best_combination_score:
                            best_combination = combo
                            best_combination_score = combo_score
                            best_length_ratio = length_ratio

            # If we found a valid combination
            if best_combination:
                best_matches = list(best_combination)
                network_ids = [match['network_id'] for match in best_matches]

                print(f"  Found geometric match with network lines: {network_ids}")

                # Fix the problematic f-strings with list comprehensions
                scores = [f"{match['score']:.2f}" for match in best_matches]
                dir_cosines = [f"{match['dir_cos']:.2f}" for match in best_matches]
                endpoint_dists = [f"{match['endpoint_dist_m']:.1f}" for match in best_matches]
                overlap_ratios = [f"{match['overlap_ratio']:.2f}" for match in best_matches]

                print(f"  Match scores: {scores}")
                print(f"  Direction cosines: {dir_cosines}")
                print(f"  Endpoint distances (m): {endpoint_dists}")
                print(f"  Overlap ratios: {overlap_ratios}")
                print(f"  Length ratio (network/jao): {best_length_ratio:.2f}")

                # Calculate total path length
                path_length = sum(match['length_m'] for match in best_matches)

                # Create a result for this match
                match_result = {
                    'jao_id': jao_id,
                    'jao_name': jao_name,
                    'v_nom': jao_voltage,
                    'matched': True,
                    'is_duplicate': False,
                    'is_geometric_match': True,  # Flag to indicate this was matched geometrically
                    'path': [],  # No path in graph
                    'network_ids': network_ids,
                    'path_length': float(path_length),
                    'jao_length': float(jao_length_m),
                    'length_ratio': float(best_length_ratio),
                    'match_quality': 'Geometric Match' + (
                        ' (voltage mismatch)' if not best_matches[0]['voltage_match'] else ''),
                    'geometric_match_details': {
                        'scores': [match['score'] for match in best_matches],
                        'dir_cosines': [match['dir_cos'] for match in best_matches],
                        'endpoint_dists': [match['endpoint_dist_m'] for match in best_matches],
                        'overlap_ratios': [match['overlap_ratio'] for match in best_matches]
                    }
                }

                new_matches.append(match_result)
                additional_matches += 1

                # Mark these network lines as matched
                for network_id in network_ids:
                    newly_matched_network_ids.add(network_id)
            else:
                print(
                    f"  No combinations found with acceptable length ratio {min_length_ratio:.1f}x to {max_length_ratio:.1f}x")
        else:
            print(f"  No geometric matches found for {jao_id}")

    print(f"\nFound {additional_matches} additional matches using advanced geometric approach")

    # Add new matches to results
    matching_results.extend(new_matches)

    return matching_results

def find_network_line_usage(matching_results):
    """Analyze how many times each network line is used across all matches."""
    network_line_usage = {}

    for result in matching_results:
        if result['matched'] and 'network_ids' in result:
            for network_id in result['network_ids']:
                network_line_usage[network_id] = network_line_usage.get(network_id, 0) + 1

    # Find heavily reused lines (used more than once)
    reused_lines = {line_id: count for line_id, count in network_line_usage.items() if count > 1}

    print(f"\nFound {len(reused_lines)} network lines that are used multiple times:")
    for line_id, count in sorted(reused_lines.items(), key=lambda x: x[1], reverse=True)[:20]:  # Top 20 most reused
        print(f"  Network line {line_id} used {count} times")

    return network_line_usage, reused_lines



def match_parallel_voltage_circuits(jao_gdf, network_gdf, matching_results):
    """
    Find and match parallel circuits where different voltage lines follow the same path.
    This handles cases where a 220kV and 400kV line follow the same path but only one gets matched.
    """
    print("\nLooking for parallel voltage circuits with same geometry...")

    # Identify matched JAO lines
    matched_jaos = {result['jao_id']: result for result in matching_results if
                    result['matched'] and not result.get('is_duplicate', False)}

    # Identify unmatched JAO lines
    unmatched_jao_ids = set(str(row['id']) for _, row in jao_gdf.iterrows()) - set(matched_jaos.keys())
    unmatched_jaos = jao_gdf[jao_gdf['id'].astype(str).isin(unmatched_jao_ids)]

    new_matches = []
    parallel_count = 0

    # For each unmatched JAO line
    for idx, unmatched_row in unmatched_jaos.iterrows():
        unmatched_id = str(unmatched_row['id'])
        unmatched_geom = unmatched_row.geometry
        unmatched_voltage = int(unmatched_row['v_nom'])
        unmatched_name = str(unmatched_row['NE_name'])

        # Compare with each matched JAO line
        best_match = None
        best_similarity = 0

        for matched_id, matched_result in matched_jaos.items():
            # Skip if same line or already used as a match
            if matched_id == unmatched_id:
                continue

            # Get the matched JAO row
            matched_rows = jao_gdf[jao_gdf['id'].astype(str) == matched_id]
            if matched_rows.empty:
                continue

            matched_row = matched_rows.iloc[0]
            matched_geom = matched_row.geometry
            matched_voltage = int(matched_row['v_nom'])

            # Skip if same voltage (we're looking for parallel circuits with different voltages)
            if matched_voltage == unmatched_voltage:
                continue

            # Compute similarity between geometries
            # Use Hausdorff distance as a similarity measure (lower is better)
            try:
                hausdorff_dist = matched_geom.hausdorff_distance(unmatched_geom)

                # Convert to approximate meters
                avg_lat = (matched_geom.centroid.y + unmatched_geom.centroid.y) / 2
                meters_per_degree = 111111 * np.cos(np.radians(abs(avg_lat)))
                hausdorff_meters = hausdorff_dist * meters_per_degree

                # Check if geometries are very similar (hausdorff distance < 500m)
                if hausdorff_meters < 500:
                    # Calculate similarity as inverse of distance (higher is better)
                    similarity = 1.0 / (1.0 + hausdorff_meters / 100)

                    # Update best match if this is better
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match = matched_result
            except Exception as e:
                print(f"Error computing similarity: {e}")
                continue

        # If we found a good match, create a parallel circuit match
        if best_match and best_similarity > 0.5:  # Threshold for good similarity
            # Get the matched network lines to reuse for this parallel circuit
            network_ids = best_match.get('network_ids', [])

            if network_ids:
                print(f"Found parallel circuit: JAO {unmatched_id} ({unmatched_voltage} kV) parallel to JAO {best_match['jao_id']} ({best_match.get('v_nom', '?')} kV)")
                print(f"  Using network lines: {network_ids}")

                # Create a new match for the parallel circuit
                new_match = {
                    'jao_id': unmatched_id,
                    'jao_name': unmatched_name,
                    'v_nom': unmatched_voltage,
                    'matched': True,
                    'is_duplicate': False,
                    'is_parallel_voltage_circuit': True,  # Mark as parallel voltage circuit
                    'parallel_to_jao': best_match['jao_id'],
                    'path': best_match.get('path', []).copy() if 'path' in best_match else [],
                    'network_ids': network_ids.copy(),
                    'path_length': best_match.get('path_length', 0),
                    'jao_length': calculate_length_meters(unmatched_geom),
                    'length_ratio': best_match.get('length_ratio', 1.0),
                    'match_quality': f'Parallel Voltage Circuit ({unmatched_voltage} kV)'
                }

                new_matches.append(new_match)
                parallel_count += 1

    print(f"Found and matched {parallel_count} parallel voltage circuits")

    # Add new matches to results
    matching_results.extend(new_matches)

    return matching_results
# Additional helper function
def calculate_geometry_coverage(geom1, geom2, buffer_meters=1000):
    """
    Calculate how much of geom2 is covered by a buffer around geom1.

    Parameters:
    - geom1: First geometry (creating the buffer)
    - geom2: Second geometry (checking coverage)
    - buffer_meters: Buffer size in meters

    Returns:
    - Coverage ratio (0.0 to 1.0)
    """
    try:
        # Calculate buffer size in degrees based on latitude
        avg_lat = (geom1.centroid.y + geom2.centroid.y) / 2
        meters_per_degree = 111111 * np.cos(np.radians(abs(avg_lat)))
        buffer_deg = buffer_meters / meters_per_degree

        # Create buffer around geom1
        buffer = geom1.buffer(buffer_deg)

        # Calculate intersection with geom2
        intersection = geom2.intersection(buffer)

        # Calculate coverage ratio
        coverage = intersection.length / geom2.length if geom2.length > 0 else 0

        return coverage
    except Exception as e:
        print(f"  Error calculating geometry coverage: {e}")
        return 0.0




def geometric_fallback_and_enhance(matching_results, jao_gdf, network_gdf, buffer_m=1200):
    """
    For UNMATCHED JAOs only, try a light geometric fallback.
    Prefers lines with same/compatible voltage and high buffer coverage.
    """
    matched_jao = {r['jao_id'] for r in matching_results if r.get('matched')}
    unmatched = jao_gdf[~jao_gdf['id'].astype(str).isin(matched_jao)]
    used_net = set(nid for r in matching_results for nid in r.get('network_ids', []))

    new_matches = []
    for _, row in unmatched.iterrows():
        jao_id = str(row['id']); jao_v = _safe_int(row['v_nom'])
        geom = row.geometry
        # buffer in degrees
        import numpy as np
        meters_per_degree = 111111*np.cos(np.radians(abs(geom.centroid.y)))
        buf = geom.buffer(buffer_m/meters_per_degree)

        # candidates
        cand = []
        for _, nr in network_gdf.iterrows():
            nid = str(nr['id'])
            if nid in used_net:  # avoid reuse unless nothing else later
                continue
            nv = int(nr['v_nom'])
            if not ((jao_v == 220 and nv == 220) or (jao_v in (380,400) and nv in (380,400))):
                continue
            inter = nr.geometry.intersection(buf)
            cov = inter.length / nr.geometry.length if nr.geometry.length>0 else 0.0
            if cov >= 0.4:
                cand.append((cov, nid, float(calculate_length_meters(nr.geometry))))

        if not cand:
            continue
        cand.sort(reverse=True)  # highest coverage first
        picked = [nid for _, nid, _ in cand[:3]]  # up to 3 segments
        plen = sum(L for _, _, L in cand[:3])

        new_matches.append({
            'jao_id': jao_id, 'jao_name': str(row.get('NE_name','')), 'v_nom': jao_v,
            'matched': True, 'is_geometric_match': True,
            'network_ids': picked, 'path': [],
            'path_length': float(plen), 'jao_length': float(calculate_length_meters(geom)),
            'length_ratio': float(plen/max(calculate_length_meters(geom),1e-9)),
            'match_quality': 'Geometric Fallback'
        })
        used_net.update(picked)

    return matching_results + new_matches

def _convert_to_km(length_value):
        """Convert length to kilometers if it appears to be in meters"""
        if not _is_num(length_value):
            return None

        value = float(length_value)
        if value > 1000:  # Assume it's in meters if > 1000
            return value / 1000.0
        return value  # Already in km


def match_pypsa_to_network(pypsa_gdf, network_gdf, G=None):
    """
    Match PyPSA lines to network lines using path finding approach with branch pruning.

    Parameters:
    -----------
    pypsa_gdf : GeoDataFrame
        PyPSA lines with geometry, voltage, etc.
    network_gdf : GeoDataFrame
        Network lines with geometry, voltage, etc.
    G : NetworkX graph (optional)
        Network graph for path finding; will be created if not provided

    Returns:
    --------
    dict mapping PyPSA IDs to matched network IDs
    """
    print("\n=== MATCHING PYPSA LINES TO NETWORK USING PATH-BASED APPROACH WITH BRANCH PRUNING ===")
    import shapely

    # Build the network graph if not provided
    if G is None:
        print("Building network graph...")
        G = build_network_graph(network_gdf)
        G = add_station_hubs_to_graph(G, network_gdf)
        G = enhance_network_graph_connectivity(G, network_gdf)

    # Parameters for matching
    max_distance_m = 1500  # Maximum distance to snap endpoints
    min_length_ratio = 0.7  # Minimum acceptable path/pypsa length ratio
    max_length_ratio = 1.5  # Maximum acceptable path/pypsa length ratio

    # Helper function to convert coordinates to graph nodes
    def find_nearest_node(point, max_dist_m=max_distance_m):
        """Find the nearest node in the graph to the given point."""
        best_node = None
        best_dist = float('inf')

        for node, data in G.nodes(data=True):
            if 'x' not in data or 'y' not in data:
                continue

            node_pt = Point(data['x'], data['y'])
            dist_deg = point.distance(node_pt)

            # Convert to meters
            lat = point.y
            m_per_deg = _meters_per_degree(lat)
            dist_m = dist_deg * m_per_deg

            if dist_m < best_dist and dist_m <= max_dist_m:
                best_dist = dist_m
                best_node = node

        return best_node, best_dist

    # Helper function to find the best path
    def find_best_path(G, start_node, end_node, voltage, max_attempts=20):
        """
        Find the best path between start and end nodes considering:
        1. Shortest path (by default)
        2. Path with minimal detours
        3. Path with consistent voltage
        """
        paths = []

        # Try various shortest path algorithms with different weights
        try:
            # Standard shortest path
            path = nx.shortest_path(G, start_node, end_node, weight='weight')
            paths.append(('shortest', path))

            # Path with minimal number of segments
            path = nx.shortest_path(G, start_node, end_node, weight=1)
            paths.append(('minimal_segments', path))

            # Path with preference for same voltage
            def voltage_weight(u, v, data):
                edge_voltage = data.get('voltage', 0)
                if edge_voltage == voltage:
                    return data.get('weight', 1) * 0.8  # Prefer same voltage
                else:
                    return data.get('weight', 1) * 1.2  # Penalize different voltage

            try:
                path = nx.shortest_path(G, start_node, end_node, weight=voltage_weight)
                paths.append(('voltage_preference', path))
            except:
                pass

            # K shortest paths
            try:
                for i, path in enumerate(nx.shortest_simple_paths(G, start_node, end_node, weight='weight')):
                    if i >= max_attempts:
                        break
                    paths.append((f'k_shortest_{i}', path))
            except:
                pass

        except nx.NetworkXNoPath:
            return None
        except Exception as e:
            print(f"Path finding error: {e}")
            return None

        # Score and select the best path
        best_path = None
        best_score = float('inf')

        for label, path in paths:
            # Extract the network lines in this path
            segments = []
            for i in range(len(path) - 1):
                edge_data = G.get_edge_data(path[i], path[i + 1])
                if edge_data and 'id' in edge_data and not edge_data.get('connector', False):
                    segments.append(edge_data)

            # Skip paths without real segments
            if not segments:
                continue

            # Calculate total length
            total_length = sum(s.get('weight', 0) for s in segments)

            # Calculate voltage consistency
            voltage_match = sum(1 for s in segments if s.get('voltage') == voltage) / len(segments) if len(
                segments) > 0 else 0

            # Score = length * (2 - voltage_match)
            # This favors shorter paths with better voltage match
            score = total_length * (2 - voltage_match)

            if score < best_score:
                best_score = score
                best_path = path

        return best_path

    # Helper function to extract path details
    def extract_path_details(G, path, network_gdf):
        """Helper function to extract network IDs and calculate path length from a path."""
        network_ids = []
        unique_ids = set()
        path_edges = []

        for i in range(len(path) - 1):
            edge_data = G.get_edge_data(path[i], path[i + 1])

            if edge_data and 'id' in edge_data and not edge_data.get('connector', False):
                if edge_data['id'] not in unique_ids:
                    network_ids.append(str(edge_data['id']))
                    unique_ids.add(edge_data['id'])

                path_edges.append((path[i], path[i + 1], edge_data))

        # If no network lines in the path, return empty
        if not network_ids:
            return [], 0, []

        # Calculate path length in meters
        path_length = 0
        for network_id in network_ids:
            network_line = network_gdf[network_gdf['id'].astype(str) == network_id]
            if not network_line.empty:
                path_length += calculate_length_meters(network_line.iloc[0].geometry)

        return network_ids, path_length, path_edges

    # Process each PyPSA line
    pypsa_network_matches = {}

    for idx, pypsa_row in pypsa_gdf.iterrows():
        pypsa_id = str(pypsa_row.get('id', idx))
        print(f"Processing PyPSA line {pypsa_id}")

        # Get PyPSA line voltage and geometry
        pypsa_voltage = int(pypsa_row['voltage']) if 'voltage' in pypsa_row and pd.notna(pypsa_row['voltage']) else 0

        # Handle geometry based on its type
        pypsa_geom = None
        if hasattr(pypsa_row, 'geometry') and pypsa_row.geometry is not None:
            if isinstance(pypsa_row.geometry, (LineString, MultiLineString)):
                pypsa_geom = pypsa_row.geometry
            elif isinstance(pypsa_row.geometry, str):
                try:
                    pypsa_geom = shapely.wkt.loads(pypsa_row.geometry)
                except:
                    pass

        # If geometry still None, try the 'geom' field
        if pypsa_geom is None and 'geom' in pypsa_row and pd.notna(pypsa_row['geom']):
            try:
                if isinstance(pypsa_row['geom'], str):
                    pypsa_geom = shapely.wkt.loads(pypsa_row['geom'])
                else:
                    pypsa_geom = pypsa_row['geom']
            except:
                pass

        # Skip invalid geometries
        if pypsa_geom is None or pypsa_geom.is_empty:
            continue

        # Calculate PyPSA line length in meters
        pypsa_length_m = calculate_length_meters(pypsa_geom)

        # Extract endpoints
        if pypsa_geom.geom_type == 'LineString':
            start_pt = Point(pypsa_geom.coords[0])
            end_pt = Point(pypsa_geom.coords[-1])
        elif pypsa_geom.geom_type == 'MultiLineString':
            start_pt = Point(pypsa_geom.geoms[0].coords[0])
            end_pt = Point(pypsa_geom.geoms[-1].coords[-1])
        else:
            continue

        # Find nearest nodes in the graph for both endpoints
        start_node, start_dist = find_nearest_node(start_pt)
        end_node, end_dist = find_nearest_node(end_pt)

        # If both endpoints were matched, find path
        if start_node and end_node and start_node != end_node:
            print(f"  Found endpoints in graph: {start_node} and {end_node}")

            # Try to find best path
            path = find_best_path(G, start_node, end_node, pypsa_voltage, max_attempts=10)

            if path:
                # Extract network IDs and calculate path length
                network_ids, path_length, _ = extract_path_details(G, path, network_gdf)

                # Check if path length ratio is acceptable
                if network_ids and pypsa_length_m > 0:
                    length_ratio = path_length / pypsa_length_m

                    if min_length_ratio <= length_ratio <= max_length_ratio:
                        print(f"  Found path with {len(network_ids)} network lines, length ratio: {length_ratio:.2f}")

                        # Apply branch pruning to remove outliers
                        pruned_network_ids = detect_and_prune_branches(network_ids, network_gdf, pypsa_geom)

                        if pruned_network_ids and len(pruned_network_ids) > 0:
                            if len(pruned_network_ids) < len(network_ids):
                                print(f"  Pruned branches: {len(network_ids)} → {len(pruned_network_ids)} segments")

                            # Store the match in the return dictionary
                            pypsa_network_matches[pypsa_id] = pruned_network_ids
                            continue
                        else:
                            # If pruning removed all segments, use original path
                            print(f"  Warning: Branch pruning removed all segments, using original path")
                            pypsa_network_matches[pypsa_id] = network_ids
                            continue
                    else:
                        print(f"  Path found but length ratio ({length_ratio:.2f}) outside acceptable range")
                else:
                    print("  No valid network lines in path")
            else:
                print("  No path found between endpoints")
        else:
            print("  Could not find suitable endpoints in graph")

        # If path-based matching failed, try geometric matching as fallback
        # Create buffer around PyPSA line
        buffer_m = 600
        lat = pypsa_geom.centroid.y
        buffer_deg = buffer_m / _meters_per_degree(lat)
        buffer = pypsa_geom.buffer(buffer_deg)

        # Find intersecting network lines with matching voltage
        geometric_matches = []

        for _, net_row in network_gdf.iterrows():
            net_id = str(net_row['id'])
            net_voltage = int(net_row['v_nom']) if 'v_nom' in net_row and pd.notna(net_row['v_nom']) else 0
            net_geom = net_row.geometry

            # Skip invalid geometries
            if net_geom is None or net_geom.is_empty:
                continue

            # Check voltage compatibility
            voltage_match = False
            if pypsa_voltage == net_voltage:
                voltage_match = True
            elif (pypsa_voltage in (380, 400) and net_voltage in (380, 400)):
                voltage_match = True
            elif abs(pypsa_voltage - net_voltage) <= 10:
                voltage_match = True

            if not voltage_match:
                continue

            # Check if geometries intersect
            if buffer.intersects(net_geom):
                # Calculate overlap ratio
                intersection = net_geom.intersection(buffer)
                overlap_ratio = intersection.length / net_geom.length if net_geom.length > 0 else 0

                if overlap_ratio >= 0.6:  # At least 60% overlap
                    geometric_matches.append((net_id, overlap_ratio, calculate_length_meters(net_geom)))

        # Sort by overlap ratio (highest first) and take up to 3
        geometric_matches.sort(key=lambda x: x[1], reverse=True)
        if geometric_matches:
            network_ids = [match[0] for match in geometric_matches[:3]]

            # Apply branch pruning to remove outliers
            pruned_network_ids = detect_and_prune_branches(network_ids, network_gdf, pypsa_geom)

            if pruned_network_ids and len(pruned_network_ids) > 0:
                if len(pruned_network_ids) < len(network_ids):
                    print(
                        f"  Pruned branches from geometric match: {len(network_ids)} → {len(pruned_network_ids)} segments")

                # Store the match in the return dictionary
                pypsa_network_matches[pypsa_id] = pruned_network_ids
            else:
                # If pruning removed all segments, use original match
                print(f"  Warning: Branch pruning removed all segments from geometric match, using original match")
                pypsa_network_matches[pypsa_id] = network_ids

    # Calculate statistics
    matched_count = len(pypsa_network_matches)
    total_pypsa = len(pypsa_gdf)
    match_percentage = matched_count / total_pypsa * 100 if total_pypsa > 0 else 0

    print(f"\nPyPSA matching results: {matched_count}/{total_pypsa} lines matched ({match_percentage:.1f}%)")

    return pypsa_network_matches


def visualize_results(jao_gdf, network_gdf, matching_results, pypsa_gdf=None, output_file=None):
    # Debug: Check if PyPSA lines are in the matching results
    pypsa_lines = [r for r in matching_results if r.get('is_pypsa_line', False)]
    pypsa_matched = [r for r in pypsa_lines if r.get('matched', False)]
    print(f"Visualization received {len(pypsa_lines)} PyPSA lines, {len(pypsa_matched)} matched")

    """
    Create a visualization of the matching results with special handling for duplicate JAO lines.
    Shows JAO lines that are duplicates in a special color and adds extra information in the popups.
    """
    import json
    import pathlib

    """
    Create an interactive Leaflet map that
      -- highlights duplicates,
      -- shows merged network paths, and
      -- distinguishes *parallel-voltage* circuits.
      -- organizes legend by voltage, match status, and match type
    """

    # ------------------------------------------------------------------
    # 1)  Scan results → buckets
    # ------------------------------------------------------------------
    matched_jao_ids = set()
    duplicate_jao_ids = set()  # same-voltage parallel
    geometric_match_jao_ids = set()
    par_voltage_jao_ids = set()  # *** NEW ***
    parallel_circuit_jao_ids = set()  # *** NEW ***

    regular_matched_net_ids = set()
    duplicate_matched_net_ids = set()
    geometric_matched_net_ids = set()
    par_voltage_net_ids = set()  # *** NEW ***
    parallel_circuit_net_ids = set()  # *** NEW ***

    for res in matching_results:
        if not res.get("matched"):
            continue

        jao_id = str(res["jao_id"])
        net_ids = [str(n) for n in res.get("network_ids", [])]

        if res.get("is_duplicate"):
            print(f"Found duplicate JAO: {jao_id} with network lines: {net_ids}")
        elif res.get("is_parallel_circuit"):
            print(f"Found parallel circuit JAO: {jao_id} with network lines: {net_ids}")

        if res.get("is_parallel_voltage_circuit"):
            par_voltage_jao_ids.add(jao_id)
            par_voltage_net_ids.update(net_ids)
        elif res.get("is_parallel_circuit"):
            parallel_circuit_jao_ids.add(jao_id)
            parallel_circuit_net_ids.update(net_ids)
        elif res.get("is_duplicate"):  # classic same-voltage parallel
            duplicate_jao_ids.add(jao_id)
            duplicate_matched_net_ids.update(net_ids)
        elif res.get("is_geometric_match"):
            geometric_match_jao_ids.add(jao_id)
            geometric_matched_net_ids.update(net_ids)
        else:
            matched_jao_ids.add(jao_id)
            regular_matched_net_ids.update(net_ids)

    # ------------------------------------------------------------------
    # 2)  Build GeoJSON features
    # ------------------------------------------------------------------
    def _mk_feature(line_id, geom, prop):
        if geom.geom_type == "LineString":
            coords = [[float(x), float(y)] for x, y in geom.coords]
            geometry = {"type": "LineString", "coordinates": coords}
        else:  # MultiLineString
            m_coords = [[[float(x), float(y)] for x, y in g.coords] for g in geom.geoms]
            geometry = {"type": "MultiLineString", "coordinates": m_coords}
        return {"type": "Feature", "id": line_id,
                "properties": prop, "geometry": geometry}

    jao_features = []
    for _, row in jao_gdf.iterrows():
        jid = str(row["id"])
        voltage = int(row["v_nom"])

        if jid in par_voltage_jao_ids:
            status, tag = "parallel_voltage", "Parallel Voltage"
        elif jid in parallel_circuit_jao_ids:
            status, tag = "parallel_circuit", "Parallel Circuit"
        elif jid in duplicate_jao_ids:
            status, tag = "duplicate", "Duplicate"
        elif jid in geometric_match_jao_ids:
            status, tag = "geometric", "Geometric Match"
        elif jid in matched_jao_ids:
            status, tag = "matched", "Matched"
        else:
            status, tag = "unmatched", "Unmatched"

        jao_features.append(_mk_feature(
            f"jao_{jid}", row.geometry,
            {"type": "jao", "id": jid, "name": str(row["NE_name"]),
             "voltage": voltage, "status": status,
             "tooltip": f"JAO {jid} -- {row['NE_name']} "
                        f"({voltage} kV) -- {tag}",
             "voltageClass": "220kV" if voltage == 220 else "400kV"}
        ))

    network_features = []
    for _, row in network_gdf.iterrows():
        nid = str(row["id"])
        voltage = int(row["v_nom"])

        if nid in par_voltage_net_ids:
            status, tag = "parallel_voltage", "Parallel Voltage"
        elif nid in parallel_circuit_net_ids:
            status, tag = "parallel_circuit", "Parallel Circuit"
        elif nid in duplicate_matched_net_ids:
            status, tag = "duplicate", "Duplicate"
        elif nid in geometric_matched_net_ids:
            status, tag = "geometric", "Geometric Match"
        elif nid in regular_matched_net_ids:
            status, tag = "matched", "Matched"
        else:
            status, tag = "unmatched", "Unmatched"

        network_features.append(_mk_feature(
            f"net_{nid}", row.geometry,
            {"type": "network", "id": nid, "voltage": voltage,
             "status": status, "tooltip": f"Network {nid} ({voltage} kV) -- {tag}",
             "voltageClass": "220kV" if voltage == 220 else "400kV"}
        ))

    # ------------------------------------------------------------------
    # 3)  merged-path features
    # ------------------------------------------------------------------
    merged_network_features = []  # New list for merged network paths

    from shapely.geometry import LineString, MultiLineString
    from shapely.ops import linemerge, unary_union

    for result in matching_results:
        if not result['matched'] or not result.get('network_ids'):
            continue

        jao_id = str(result['jao_id'])
        network_ids = result.get('network_ids', [])

        # Skip if no network lines
        if not network_ids:
            continue

        # Get all network geometries for this match
        network_geometries = []
        for network_id in network_ids:
            network_row = network_gdf[network_gdf['id'].astype(str) == network_id]
            if not network_row.empty:
                network_geometries.append(network_row.iloc[0].geometry)

        # Skip if no geometries
        if not network_geometries:
            continue

        # Try to merge the network geometries
        try:
            # First, convert to a collection
            multi_line = MultiLineString(network_geometries)

            # Attempt to merge the lines
            merged_line = linemerge(multi_line)

            # Use the merged line if successful, otherwise use the multiline
            if merged_line.geom_type == 'LineString':
                final_geom = merged_line
            else:
                final_geom = multi_line

            # Determine voltage class
            jao_row = jao_gdf[jao_gdf['id'].astype(str) == jao_id]
            if not jao_row.empty:
                voltage = int(jao_row.iloc[0]['v_nom'])
                voltage_class = "220kV" if voltage == 220 else "400kV"
            else:
                voltage_class = "400kV"  # Default

            # Determine line status based on JAO status
            if jao_id in par_voltage_jao_ids:
                status = "parallel_voltage"
                tooltip_status = "Parallel Voltage"
            elif jao_id in parallel_circuit_jao_ids:
                status = "parallel_circuit"
                tooltip_status = "Parallel Circuit"
            elif jao_id in duplicate_jao_ids:
                status = "duplicate"
                tooltip_status = "Duplicate"
            elif jao_id in geometric_match_jao_ids:
                status = "geometric"
                tooltip_status = "Geometric Match"
            else:
                status = "matched"
                tooltip_status = "Matched"

            # Calculate the match quality score for the tooltip
            match_quality = result.get('match_quality', 'Unknown')
            length_ratio = result.get('length_ratio', 0)

            # Create a unique ID for the merged path
            merged_id = f"merged-net-jao-{jao_id}"

            # Create coordinates list based on geometry type
            if final_geom.geom_type == 'LineString':
                coords = list(final_geom.coords)
                geometry = {
                    "type": "LineString",
                    "coordinates": [[float(x), float(y)] for x, y in coords]
                }
            else:  # MultiLineString
                multi_coords = []
                for line in final_geom.geoms:
                    line_coords = [[float(x), float(y)] for x, y in line.coords]
                    multi_coords.append(line_coords)
                geometry = {
                    "type": "MultiLineString",
                    "coordinates": multi_coords
                }

            # Create GeoJSON feature for the merged network path
            feature = {
                "type": "Feature",
                "id": merged_id,
                "properties": {
                    "type": "merged_network",
                    "jao_id": jao_id,
                    "network_ids": ",".join(network_ids),
                    "status": status,
                    "voltageClass": voltage_class,
                    "tooltip": f"Merged Network Path for JAO {jao_id} ({len(network_ids)} lines) - {match_quality} - Ratio: {length_ratio:.2f}"
                },
                "geometry": geometry
            }

            merged_network_features.append(feature)

        except Exception as e:
            print(f"Error creating merged network path for JAO {jao_id}: {e}")

    # ------------------------------------------------------------------
    # 4)  Assemble & dump HTML
    # ------------------------------------------------------------------
    import json, pathlib
    jao_json = json.dumps({"type": "FeatureCollection", "features": jao_features})
    net_json = json.dumps({"type": "FeatureCollection", "features": network_features})
    merged_json = json.dumps({"type": "FeatureCollection", "features": merged_network_features})

    # CSS for the organized legend
    css = """
    html, body {
        height: 100%;
        margin: 0;
        padding: 0;
        font-family: Arial, sans-serif;
    }
    #map {
        height: 100%;
        width: 100%;
    }
    .sidebar {
        position: absolute;
        top: 10px;
        right: 10px;
        width: 280px;
        background: white;
        border-radius: 5px;
        box-shadow: 0 0 10px rgba(0, 0, 0, 0.5);
        max-height: calc(100% - 20px);
        overflow-y: auto;
        z-index: 1000;
    }
    .sidebar-header {
        padding: 10px;
        background: #f8f8f8;
        border-bottom: 1px solid #ddd;
        font-weight: bold;
        border-radius: 5px 5px 0 0;
    }
    .sidebar-content {
        padding: 10px;
    }
    .search-box {
        width: 100%;
        padding: 8px;
        box-sizing: border-box;
        margin-bottom: 10px;
        border: 1px solid #ddd;
        border-radius: 3px;
    }
    .section-header {
        font-weight: bold;
        margin: 15px 0 5px 0;
        padding-bottom: 3px;
        border-bottom: 1px solid #eee;
    }
    .subsection-header {
        font-weight: bold;
        margin: 10px 0 5px 0;
        color: #666;
    }
    .filter-group {
        margin-bottom: 15px;
    }
    .filter-option {
        padding: 5px;
        margin: 2px 0;
        cursor: pointer;
        display: flex;
        align-items: center;
    }
    .filter-option.active {
        background-color: #f0f8ff;
    }
    .filter-option:hover {
        background-color: #f5f5f5;
    }
    .legend-color {
        display: inline-block;
        width: 15px;
        height: 15px;
        margin-right: 8px;
        border: 1px solid #ddd;
    }
    .voltage-group {
        margin-bottom: 15px;
        border: 1px solid #eee;
        border-radius: 3px;
        padding: 8px;
    }
    .voltage-header {
        font-weight: bold;
        margin-bottom: 5px;
        color: #444;
    }
    .status-group {
        margin-left: 10px;
        margin-bottom: 10px;
    }
    .status-header {
        font-weight: bold;
        margin-bottom: 3px;
        color: #666;
    }
    .type-group {
        margin-left: 20px;
    }
    """

    # JavaScript to create the organized legend
    js = """
    // Initialize map
    const map = L.map('map').setView([51.2, 10.4], 6);
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png').addTo(map);

    // Style functions
    function jaoStyle(f) {
        let st = f.properties.status;
        if (st === 'parallel_voltage') return {color: '#FFD700', weight: 3, opacity: 0.8, dashArray: '12,6'};
        if (st === 'parallel_circuit') return {color: '#FF6600', weight: 3, opacity: 0.8, dashArray: '8,4'};
        if (st === 'duplicate') return {color: '#9932CC', weight: 3, opacity: 0.8, dashArray: '5,5'};
        if (st === 'geometric') return {color: '#00BFFF', weight: 3, opacity: 0.8, dashArray: '10,5'};
        if (st === 'matched') return {color: 'green', weight: 3, opacity: 0.8};
        return {color: 'red', weight: 3, opacity: 0.8};
    }

    function networkStyle(f) {
        let st = f.properties.status;
        if (st === 'parallel_voltage') return {color: '#FFD700', weight: 2, opacity: 0.6, dashArray: '12,6'};
        if (st === 'parallel_circuit') return {color: '#FF6600', weight: 2, opacity: 0.6, dashArray: '8,4'};
        if (st === 'duplicate') return {color: '#9932CC', weight: 2, opacity: 0.6, dashArray: '5,5'};
        if (st === 'geometric') return {color: '#00BFFF', weight: 2, opacity: 0.6, dashArray: '10,5'};
        if (st === 'matched') return {color: 'green', weight: 2, opacity: 0.6};
        return {color: 'blue', weight: 2, opacity: 0.6};
    }

    function mergedStyle(f) {
        let st = f.properties.status;
        if (st === 'parallel_voltage') return {color: '#FFD700', weight: 5, opacity: 0.8, dashArray: '14,8'};
        if (st === 'parallel_circuit') return {color: '#FF6600', weight: 5, opacity: 0.8, dashArray: '10,6'};
        if (st === 'duplicate') return {color: '#9932CC', weight: 5, opacity: 0.8, dashArray: '10,5'};
        if (st === 'geometric') return {color: '#00BFFF', weight: 5, opacity: 0.8, dashArray: '15,10'};
        return {color: 'green', weight: 5, opacity: 0.8};
    }

    // Add JAO, network, and merged layers
    const jaoLayer = L.geoJSON(jaoLines, {
        style: jaoStyle,
        onEachFeature: function(feature, layer) {
            layer.bindTooltip(feature.properties.tooltip);
            // Add voltage and status as data attributes for filtering
            layer.voltageClass = feature.properties.voltageClass;
            layer.status = feature.properties.status;
            layer.lineType = 'jao';
        }
    }).addTo(map);

    const networkLayer = L.geoJSON(networkLines, {
        style: networkStyle,
        onEachFeature: function(feature, layer) {
            layer.bindTooltip(feature.properties.tooltip);
            // Add voltage and status as data attributes for filtering
            layer.voltageClass = feature.properties.voltageClass;
            layer.status = feature.properties.status;
            layer.lineType = 'network';
        }
    }).addTo(map);

    const mergedLayer = L.geoJSON(mergedLines, {
        style: mergedStyle,
        onEachFeature: function(feature, layer) {
            layer.bindTooltip(feature.properties.tooltip);
            // Add voltage and status as data attributes for filtering
            layer.voltageClass = feature.properties.voltageClass;
            layer.status = feature.properties.status;
            layer.lineType = 'merged';
        }
    }).addTo(map);

    // Create the sidebar
    const sidebar = document.createElement('div');
    sidebar.className = 'sidebar';
    sidebar.innerHTML = `
        <div class="sidebar-header">
            <span>🔍 Search</span>
        </div>
        <div class="sidebar-content">
            <input type="text" class="search-box" placeholder="Search for lines...">

            <div class="section-header">Voltage</div>
            <div class="filter-group" id="voltage-filters">
                <div class="filter-option active" data-filter="all-voltage">All</div>
                <div class="filter-option active" data-filter="220kV">220 kV</div>
                <div class="filter-option active" data-filter="400kV">400 kV</div>
            </div>

            <div class="section-header">Line Types</div>
            <div class="filter-group" id="type-filters">
                <!-- 220kV Lines -->
                <div class="voltage-group">
                    <div class="voltage-header">220 kV</div>

                    <!-- Matched 220kV -->
                    <div class="status-group">
                        <div class="status-header">Matched</div>
                        <div class="type-group">
                            <div class="filter-option active" data-filter="jao-matched-220kV">
                                <div class="legend-color" style="background-color:green;"></div>
                                <span>Regular JAO</span>
                            </div>
                            <div class="filter-option active" data-filter="jao-geometric-220kV">
                                <div class="legend-color" style="background-color:#00BFFF;"></div>
                                <span>Geometric JAO</span>
                            </div>
                            <div class="filter-option active" data-filter="jao-duplicate-220kV">
                                <div class="legend-color" style="background-color:#9932CC;"></div>
                                <span>Duplicate JAO</span>
                            </div>
                            <div class="filter-option active" data-filter="jao-parallel-circuit-220kV">
                                <div class="legend-color" style="background-color:#FF6600;"></div>
                                <span>Parallel Circuit JAO</span>
                            </div>
                            <div class="filter-option active" data-filter="jao-parallel-voltage-220kV">
                                <div class="legend-color" style="background-color:#FFD700;"></div>
                                <span>Parallel Voltage JAO</span>
                            </div>
                        </div>
                    </div>

                    <!-- Unmatched 220kV -->
                    <div class="status-group">
                        <div class="status-header">Unmatched</div>
                        <div class="type-group">
                            <div class="filter-option active" data-filter="jao-unmatched-220kV">
                                <div class="legend-color" style="background-color:red;"></div>
                                <span>Unmatched JAO</span>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- 400kV Lines -->
                <div class="voltage-group">
                    <div class="voltage-header">400 kV</div>

                    <!-- Matched 400kV -->
                    <div class="status-group">
                        <div class="status-header">Matched</div>
                        <div class="type-group">
                            <div class="filter-option active" data-filter="jao-matched-400kV">
                                <div class="legend-color" style="background-color:green;"></div>
                                <span>Regular JAO</span>
                            </div>
                            <div class="filter-option active" data-filter="jao-geometric-400kV">
                                <div class="legend-color" style="background-color:#00BFFF;"></div>
                                <span>Geometric JAO</span>
                            </div>
                            <div class="filter-option active" data-filter="jao-duplicate-400kV">
                                <div class="legend-color" style="background-color:#9932CC;"></div>
                                <span>Duplicate JAO</span>
                            </div>
                            <div class="filter-option active" data-filter="jao-parallel-circuit-400kV">
                                <div class="legend-color" style="background-color:#FF6600;"></div>
                                <span>Parallel Circuit JAO</span>
                            </div>
                            <div class="filter-option active" data-filter="jao-parallel-voltage-400kV">
                                <div class="legend-color" style="background-color:#FFD700;"></div>
                                <span>Parallel Voltage JAO</span>
                            </div>
                        </div>
                    </div>

                    <!-- Unmatched 400kV -->
                    <div class="status-group">
                        <div class="status-header">Unmatched</div>
                        <div class="type-group">
                            <div class="filter-option active" data-filter="jao-unmatched-400kV">
                                <div class="legend-color" style="background-color:red;"></div>
                                <span>Unmatched JAO</span>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Network Lines -->
                <div class="voltage-group">
                    <div class="voltage-header">Network Lines</div>
                    <div class="type-group">
                        <div class="filter-option active" data-filter="network-matched">
                            <div class="legend-color" style="background-color:green;"></div>
                            <span>Matched Network</span>
                        </div>
                        <div class="filter-option active" data-filter="network-geometric">
                            <div class="legend-color" style="background-color:#00BFFF;"></div>
                            <span>Geometric Network</span>
                        </div>
                        <div class="filter-option active" data-filter="network-duplicate">
                            <div class="legend-color" style="background-color:#9932CC;"></div>
                            <span>Duplicate Network</span>
                        </div>
                        <div class="filter-option active" data-filter="network-parallel-circuit">
                            <div class="legend-color" style="background-color:#FF6600;"></div>
                            <span>Parallel Circuit Network</span>
                        </div>
                        <div class="filter-option active" data-filter="network-parallel-voltage">
                            <div class="legend-color" style="background-color:#FFD700;"></div>
                            <span>Parallel Voltage Network</span>
                        </div>
                        <div class="filter-option active" data-filter="network-unmatched">
                            <div class="legend-color" style="background-color:blue;"></div>
                            <span>Unmatched Network</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    `;
    document.body.appendChild(sidebar);

    // Filter functionality
    document.querySelectorAll('.filter-option').forEach(option => {
        option.addEventListener('click', function() {
            this.classList.toggle('active');
            applyFilters();
        });
    });

    // Search functionality
    document.querySelector('.search-box').addEventListener('input', function(e) {
        const searchTerm = e.target.value.toLowerCase();
        applySearch(searchTerm);
    });

    // Apply filters based on active filter options
    function applyFilters() {
        const activeVoltageFilters = Array.from(document.querySelectorAll('#voltage-filters .filter-option.active'))
            .map(el => el.getAttribute('data-filter'));

        const activeTypeFilters = Array.from(document.querySelectorAll('#type-filters .filter-option.active'))
            .map(el => el.getAttribute('data-filter'));

        const allVoltages = activeVoltageFilters.includes('all-voltage');

        // Process JAO layers
        jaoLayer.eachLayer(layer => {
            const voltageMatch = allVoltages || activeVoltageFilters.includes(layer.voltageClass);
            let typeMatch = false;

            if (layer.status === 'matched' && activeTypeFilters.includes(`jao-matched-${layer.voltageClass}`)) {
                typeMatch = true;
            } else if (layer.status === 'geometric' && activeTypeFilters.includes(`jao-geometric-${layer.voltageClass}`)) {
                typeMatch = true;
            } else if (layer.status === 'duplicate' && activeTypeFilters.includes(`jao-duplicate-${layer.voltageClass}`)) {
                typeMatch = true;
            } else if (layer.status === 'parallel_circuit' && activeTypeFilters.includes(`jao-parallel-circuit-${layer.voltageClass}`)) {
                typeMatch = true;
            } else if (layer.status === 'parallel_voltage' && activeTypeFilters.includes(`jao-parallel-voltage-${layer.voltageClass}`)) {
                typeMatch = true;
            } else if (layer.status === 'unmatched' && activeTypeFilters.includes(`jao-unmatched-${layer.voltageClass}`)) {
                typeMatch = true;
            }

            if (voltageMatch && typeMatch) {
                layer.setStyle({opacity: 0.8, fillOpacity: 0.6});
            } else {
                layer.setStyle({opacity: 0, fillOpacity: 0});
            }
        });

        // Process Network layers
        networkLayer.eachLayer(layer => {
            const voltageMatch = allVoltages || activeVoltageFilters.includes(layer.voltageClass);
            let typeMatch = false;

            if (layer.status === 'matched' && activeTypeFilters.includes('network-matched')) {
                typeMatch = true;
            } else if (layer.status === 'geometric' && activeTypeFilters.includes('network-geometric')) {
                typeMatch = true;
            } else if (layer.status === 'duplicate' && activeTypeFilters.includes('network-duplicate')) {
                typeMatch = true;
            } else if (layer.status === 'parallel_circuit' && activeTypeFilters.includes('network-parallel-circuit')) {
                typeMatch = true;
            } else if (layer.status === 'parallel_voltage' && activeTypeFilters.includes('network-parallel-voltage')) {
                typeMatch = true;
            } else if (layer.status === 'unmatched' && activeTypeFilters.includes('network-unmatched')) {
                typeMatch = true;
            }

            if (voltageMatch && typeMatch) {
                layer.setStyle({opacity: 0.6, fillOpacity: 0.4});
            } else {
                layer.setStyle({opacity: 0, fillOpacity: 0});
            }
        });

        // Process Merged layers
        mergedLayer.eachLayer(layer => {
            const voltageMatch = allVoltages || activeVoltageFilters.includes(layer.voltageClass);
            let typeMatch = false;

            if (layer.status === 'matched' && 
                activeTypeFilters.some(f => f.startsWith('jao-matched'))) {
                typeMatch = true;
            } else if (layer.status === 'geometric' && 
                      activeTypeFilters.some(f => f.startsWith('jao-geometric'))) {
                typeMatch = true;
            } else if (layer.status === 'duplicate' && 
                      activeTypeFilters.some(f => f.startsWith('jao-duplicate'))) {
                typeMatch = true;
            } else if (layer.status === 'parallel_circuit' && 
                      activeTypeFilters.some(f => f.startsWith('jao-parallel-circuit'))) {
                typeMatch = true;
            } else if (layer.status === 'parallel_voltage' && 
                      activeTypeFilters.some(f => f.startsWith('jao-parallel-voltage'))) {
                typeMatch = true;
            }

            if (voltageMatch && typeMatch) {
                layer.setStyle({opacity: 0.8, fillOpacity: 0.6});
            } else {
                layer.setStyle({opacity: 0, fillOpacity: 0});
            }
        });
    }

    // Apply search filter
    function applySearch(searchTerm) {
        jaoLayer.eachLayer(layer => {
            const props = layer.feature.properties;
            const matchesSearch = props.tooltip.toLowerCase().includes(searchTerm) || 
                                 props.id.toString().toLowerCase().includes(searchTerm);

            if (!matchesSearch) {
                layer.setStyle({opacity: 0, fillOpacity: 0});
            } else {
                applyFilters(); // Reapply the current filters
            }
        });

        networkLayer.eachLayer(layer => {
            const props = layer.feature.properties;
            const matchesSearch = props.tooltip.toLowerCase().includes(searchTerm) || 
                                 props.id.toString().toLowerCase().includes(searchTerm);

            if (!matchesSearch) {
                layer.setStyle({opacity: 0, fillOpacity: 0});
            } else {
                applyFilters(); // Reapply the current filters
            }
        });
    }

    // Initial application of filters
    applyFilters();
    """

    html = f"""
    <!doctype html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>JAO-Network Line Matching Results</title>
        <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
        <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
        <style>
        {css}
        </style>
    </head>
    <body>
        <div id="map"></div>
        <script>
        // Data
        const jaoLines = {jao_json};
        const networkLines = {net_json};
        const mergedLines = {merged_json};

        {js}
        </script>
    </body>
    </html>
    """

    # write file
    import pathlib
    out_path = pathlib.Path(output_dir) / "jao_network_matching_with_duplicates.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    return out_path


def create_enhanced_summary_table(jao_gdf, network_gdf, matching_results, pypsa_gdf=None):
    """Create an HTML table with detailed information about the matching results including electrical parameters,
    coverage of matched path, and totals consistency checks. Optionally includes PyPSA line data."""
    import pandas as pd
    import json
    import numpy as np
    from pathlib import Path
    import shapely.wkt
    from shapely.geometry import Point, LineString, MultiLineString
    import networkx as nx
    import math

    # -------- Path-based matching for PyPSA lines --------
    def _meters_per_degree(latitude):
        """Calculate meters per degree of longitude at the given latitude."""
        # Earth's radius in meters
        earth_radius = 6378137.0

        # Convert latitude to radians
        lat_rad = math.radians(latitude)

        # Calculate meters per degree of longitude
        meters_per_deg_lng = (math.pi / 180) * earth_radius * math.cos(lat_rad)

        # Meters per degree of latitude is roughly constant
        meters_per_deg_lat = 111320.0

        # Return average
        return (meters_per_deg_lng + meters_per_deg_lat) / 2

    def calculate_length_meters(geometry):
        """Calculate the length of a geometry in meters."""
        if geometry is None or geometry.is_empty:
            return 0

        # Get a representative latitude for conversion
        if geometry.geom_type == 'LineString':
            lat = geometry.centroid.y
        elif geometry.geom_type == 'MultiLineString':
            lat = geometry.centroid.y
        else:
            lat = geometry.y  # Assume Point

        # Convert degrees to meters
        m_per_deg = _meters_per_degree(lat)

        # Calculate length
        length_deg = geometry.length
        length_m = length_deg * m_per_deg

        return length_m

    def build_network_graph(network_gdf):
        """Build a NetworkX graph from the network GeoDataFrame."""
        G = nx.Graph()

        # Add nodes for all unique endpoints
        node_id = 0
        endpoints = {}  # Maps (x, y) coordinates to node IDs

        # Process each line in the network
        for idx, row in network_gdf.iterrows():
            line_id = str(row['id'])
            geom = row.geometry

            if geom is None or geom.is_empty:
                continue

            # Extract coordinates
            if geom.geom_type == 'LineString':
                coords = list(geom.coords)
            elif geom.geom_type == 'MultiLineString':
                coords = []
                for line in geom.geoms:
                    coords.extend(list(line.coords))
            else:
                continue

            # Add start and end points as nodes if they don't exist
            start_coord = coords[0]
            end_coord = coords[-1]

            # Add start node
            if start_coord not in endpoints:
                endpoints[start_coord] = f"node_{node_id}"
                G.add_node(endpoints[start_coord], x=start_coord[0], y=start_coord[1])
                node_id += 1

            # Add end node
            if end_coord not in endpoints:
                endpoints[end_coord] = f"node_{node_id}"
                G.add_node(endpoints[end_coord], x=end_coord[0], y=end_coord[1])
                node_id += 1

            # Add edge with attributes
            start_node = endpoints[start_coord]
            end_node = endpoints[end_coord]

            # Calculate length in meters
            length_m = calculate_length_meters(geom)

            # Add edge with attributes from the network line
            G.add_edge(
                start_node,
                end_node,
                id=line_id,
                weight=length_m,
                voltage=int(row.get('v_nom', 0)) if 'v_nom' in row and pd.notna(row['v_nom']) else 0,
                geometry=geom
            )

        return G

    def add_station_hubs_to_graph(G, network_gdf):
        """
        Add virtual nodes for substations/hubs where multiple lines meet.
        This improves connectivity in the graph.
        """
        # Find nodes with degree > 1 (hubs)
        hubs = [node for node, degree in G.degree() if degree > 1]

        # For each hub, check nearby nodes and connect them
        for hub in hubs:
            hub_data = G.nodes[hub]
            hub_point = Point(hub_data['x'], hub_data['y'])

            # Find other hubs within a small distance
            nearby_hubs = []
            for other_hub in hubs:
                if other_hub == hub:
                    continue

                other_data = G.nodes[other_hub]
                other_point = Point(other_data['x'], other_data['y'])

                # Calculate distance in meters
                dist_deg = hub_point.distance(other_point)
                lat = hub_point.y
                m_per_deg = _meters_per_degree(lat)
                dist_m = dist_deg * m_per_deg

                # If less than 500m, consider them part of the same station
                if dist_m < 500:
                    nearby_hubs.append(other_hub)

            # Connect this hub to nearby hubs
            for other_hub in nearby_hubs:
                if not G.has_edge(hub, other_hub):
                    # Add a short connector edge
                    G.add_edge(
                        hub,
                        other_hub,
                        weight=100,  # Short distance
                        connector=True  # Mark as connector
                    )

        return G

    def enhance_network_graph_connectivity(G, network_gdf):
        """Enhance graph connectivity by adding edges between nearby nodes."""
        # Get all nodes and their positions
        nodes = list(G.nodes(data=True))

        # For each node, find nearby nodes
        for i, (node1, data1) in enumerate(nodes):
            point1 = Point(data1['x'], data1['y'])

            for node2, data2 in nodes[i + 1:]:
                # Skip if already connected
                if G.has_edge(node1, node2):
                    continue

                point2 = Point(data2['x'], data2['y'])

                # Calculate distance in meters
                dist_deg = point1.distance(point2)
                lat = point1.y
                m_per_deg = _meters_per_degree(lat)
                dist_m = dist_deg * m_per_deg

                # If within 300m, add a connector edge
                if dist_m < 300:
                    G.add_edge(
                        node1,
                        node2,
                        weight=dist_m,
                        connector=True  # Mark as connector
                    )

        return G

    def match_pypsa_to_network(pypsa_gdf, network_gdf, G=None):
        """
        Match PyPSA lines to network lines using path finding approach with branch pruning.

        Parameters:
        -----------
        pypsa_gdf : GeoDataFrame
            PyPSA lines with geometry, voltage, etc.
        network_gdf : GeoDataFrame
            Network lines with geometry, voltage, etc.
        G : NetworkX graph (optional)
            Network graph for path finding; will be created if not provided

        Returns:
        --------
        dict mapping PyPSA IDs to matched network IDs
        """
        print("\n=== MATCHING PYPSA LINES TO NETWORK USING PATH-BASED APPROACH WITH BRANCH PRUNING ===")

        # Build the network graph if not provided
        if G is None:
            print("Building network graph...")
            G = build_network_graph(network_gdf)
            G = add_station_hubs_to_graph(G, network_gdf)
            G = enhance_network_graph_connectivity(G, network_gdf)

        # Parameters for matching
        max_distance_m = 1500  # Maximum distance to snap endpoints
        min_length_ratio = 0.7  # Minimum acceptable path/pypsa length ratio
        max_length_ratio = 1.5  # Maximum acceptable path/pypsa length ratio

        # Helper function to convert coordinates to graph nodes
        def find_nearest_node(point, max_dist_m=max_distance_m):
            """Find the nearest node in the graph to the given point."""
            best_node = None
            best_dist = float('inf')

            for node, data in G.nodes(data=True):
                if 'x' not in data or 'y' not in data:
                    continue

                node_pt = Point(data['x'], data['y'])
                dist_deg = point.distance(node_pt)

                # Convert to meters
                lat = point.y
                m_per_deg = _meters_per_degree(lat)
                dist_m = dist_deg * m_per_deg

                if dist_m < best_dist and dist_m <= max_dist_m:
                    best_dist = dist_m
                    best_node = node

            return best_node, best_dist

        # Helper function to find the best path
        def find_best_path(G, start_node, end_node, voltage, max_attempts=20):
            """
            Find the best path between start and end nodes considering:
            1. Shortest path (by default)
            2. Path with minimal detours
            3. Path with consistent voltage
            """
            paths = []

            # Try various shortest path algorithms with different weights
            try:
                # Standard shortest path
                path = nx.shortest_path(G, start_node, end_node, weight='weight')
                paths.append(('shortest', path))

                # Path with minimal number of segments
                path = nx.shortest_path(G, start_node, end_node, weight=1)
                paths.append(('minimal_segments', path))

                # Path with preference for same voltage
                def voltage_weight(u, v, data):
                    edge_voltage = data.get('voltage', 0)
                    if edge_voltage == voltage:
                        return data.get('weight', 1) * 0.8  # Prefer same voltage
                    else:
                        return data.get('weight', 1) * 1.2  # Penalize different voltage

                try:
                    path = nx.shortest_path(G, start_node, end_node, weight=voltage_weight)
                    paths.append(('voltage_preference', path))
                except:
                    pass

                # K shortest paths
                try:
                    for i, path in enumerate(nx.shortest_simple_paths(G, start_node, end_node, weight='weight')):
                        if i >= max_attempts:
                            break
                        paths.append((f'k_shortest_{i}', path))
                except:
                    pass

            except nx.NetworkXNoPath:
                return None
            except Exception as e:
                print(f"Path finding error: {e}")
                return None

            # Score and select the best path
            best_path = None
            best_score = float('inf')

            for label, path in paths:
                # Extract the network lines in this path
                segments = []
                for i in range(len(path) - 1):
                    edge_data = G.get_edge_data(path[i], path[i + 1])
                    if edge_data and 'id' in edge_data and not edge_data.get('connector', False):
                        segments.append(edge_data)

                # Skip paths without real segments
                if not segments:
                    continue

                # Calculate total length
                total_length = sum(s.get('weight', 0) for s in segments)

                # Calculate voltage consistency
                voltage_match = sum(1 for s in segments if s.get('voltage') == voltage) / len(segments) if len(
                    segments) > 0 else 0

                # Score = length * (2 - voltage_match)
                # This favors shorter paths with better voltage match
                score = total_length * (2 - voltage_match)

                if score < best_score:
                    best_score = score
                    best_path = path

            return best_path

        # Helper function to extract path details
        def extract_path_details(G, path, network_gdf):
            """Helper function to extract network IDs and calculate path length from a path."""
            network_ids = []
            unique_ids = set()
            path_edges = []

            for i in range(len(path) - 1):
                edge_data = G.get_edge_data(path[i], path[i + 1])

                if edge_data and 'id' in edge_data and not edge_data.get('connector', False):
                    if edge_data['id'] not in unique_ids:
                        network_ids.append(str(edge_data['id']))
                        unique_ids.add(edge_data['id'])

                    path_edges.append((path[i], path[i + 1], edge_data))

            # If no network lines in the path, return empty
            if not network_ids:
                return [], 0, []

            # Calculate path length in meters
            path_length = 0
            for network_id in network_ids:
                network_line = network_gdf[network_gdf['id'].astype(str) == network_id]
                if not network_line.empty:
                    path_length += calculate_length_meters(network_line.iloc[0].geometry)

            return network_ids, path_length, path_edges

        # Process each PyPSA line
        pypsa_network_matches = {}

        for idx, pypsa_row in pypsa_gdf.iterrows():
            pypsa_id = str(pypsa_row.get('id', idx))
            print(f"Processing PyPSA line {pypsa_id}")

            # Get PyPSA line voltage and geometry
            pypsa_voltage = int(pypsa_row['voltage']) if 'voltage' in pypsa_row and pd.notna(
                pypsa_row['voltage']) else 0

            # Handle geometry based on its type
            pypsa_geom = None
            if hasattr(pypsa_row, 'geometry') and pypsa_row.geometry is not None:
                if isinstance(pypsa_row.geometry, (LineString, MultiLineString)):
                    pypsa_geom = pypsa_row.geometry
                elif isinstance(pypsa_row.geometry, str):
                    try:
                        pypsa_geom = shapely.wkt.loads(pypsa_row.geometry)
                    except:
                        pass

            # If geometry still None, try the 'geom' field
            if pypsa_geom is None and 'geom' in pypsa_row and pd.notna(pypsa_row['geom']):
                try:
                    if isinstance(pypsa_row['geom'], str):
                        pypsa_geom = shapely.wkt.loads(pypsa_row['geom'])
                    else:
                        pypsa_geom = pypsa_row['geom']
                except:
                    pass

            # Skip invalid geometries
            if pypsa_geom is None or pypsa_geom.is_empty:
                continue

            # Calculate PyPSA line length in meters
            pypsa_length_m = calculate_length_meters(pypsa_geom)

            # Extract endpoints
            if pypsa_geom.geom_type == 'LineString':
                start_pt = Point(pypsa_geom.coords[0])
                end_pt = Point(pypsa_geom.coords[-1])
            elif pypsa_geom.geom_type == 'MultiLineString':
                start_pt = Point(pypsa_geom.geoms[0].coords[0])
                end_pt = Point(pypsa_geom.geoms[-1].coords[-1])
            else:
                continue

            # Find nearest nodes in the graph for both endpoints
            start_node, start_dist = find_nearest_node(start_pt)
            end_node, end_dist = find_nearest_node(end_pt)

            # If both endpoints were matched, find path
            if start_node and end_node and start_node != end_node:
                print(f"  Found endpoints in graph: {start_node} and {end_node}")

                # Try to find best path
                path = find_best_path(G, start_node, end_node, pypsa_voltage, max_attempts=10)

                if path:
                    # Extract network IDs and calculate path length
                    network_ids, path_length, _ = extract_path_details(G, path, network_gdf)

                    # Check if path length ratio is acceptable
                    if network_ids and pypsa_length_m > 0:
                        length_ratio = path_length / pypsa_length_m

                        if min_length_ratio <= length_ratio <= max_length_ratio:
                            print(
                                f"  Found path with {len(network_ids)} network lines, length ratio: {length_ratio:.2f}")

                            # Apply branch pruning to remove outliers
                            pruned_network_ids = detect_and_prune_branches(network_ids, network_gdf, pypsa_geom)

                            if pruned_network_ids and len(pruned_network_ids) > 0:
                                if len(pruned_network_ids) < len(network_ids):
                                    print(f"  Pruned branches: {len(network_ids)} → {len(pruned_network_ids)} segments")

                                # Store the match in the return dictionary
                                pypsa_network_matches[pypsa_id] = pruned_network_ids
                                continue
                            else:
                                # If pruning removed all segments, use original path
                                print(f"  Warning: Branch pruning removed all segments, using original path")
                                pypsa_network_matches[pypsa_id] = network_ids
                                continue
                        else:
                            print(f"  Path found but length ratio ({length_ratio:.2f}) outside acceptable range")
                    else:
                        print("  No valid network lines in path")
                else:
                    print("  No path found between endpoints")
            else:
                print("  Could not find suitable endpoints in graph")

            # If path-based matching failed, try geometric matching as fallback
            # Create buffer around PyPSA line
            buffer_m = 600
            lat = pypsa_geom.centroid.y
            buffer_deg = buffer_m / _meters_per_degree(lat)
            buffer = pypsa_geom.buffer(buffer_deg)

            # Find intersecting network lines with matching voltage
            geometric_matches = []

            for _, net_row in network_gdf.iterrows():
                net_id = str(net_row['id'])
                net_voltage = int(net_row['v_nom']) if 'v_nom' in net_row and pd.notna(net_row['v_nom']) else 0
                net_geom = net_row.geometry

                # Skip invalid geometries
                if net_geom is None or net_geom.is_empty:
                    continue

                # Check voltage compatibility
                voltage_match = False
                if pypsa_voltage == net_voltage:
                    voltage_match = True
                elif (pypsa_voltage in (380, 400) and net_voltage in (380, 400)):
                    voltage_match = True
                elif abs(pypsa_voltage - net_voltage) <= 10:
                    voltage_match = True

                if not voltage_match:
                    continue

                # Check if geometries intersect
                if buffer.intersects(net_geom):
                    # Calculate overlap ratio
                    intersection = net_geom.intersection(buffer)
                    overlap_ratio = intersection.length / net_geom.length if net_geom.length > 0 else 0

                    if overlap_ratio >= 0.6:  # At least 60% overlap
                        geometric_matches.append((net_id, overlap_ratio, calculate_length_meters(net_geom)))

            # Sort by overlap ratio (highest first) and take up to 3
            geometric_matches.sort(key=lambda x: x[1], reverse=True)
            if geometric_matches:
                network_ids = [match[0] for match in geometric_matches[:3]]

                # Apply branch pruning to remove outliers
                pruned_network_ids = detect_and_prune_branches(network_ids, network_gdf, pypsa_geom)

                if pruned_network_ids and len(pruned_network_ids) > 0:
                    if len(pruned_network_ids) < len(network_ids):
                        print(
                            f"  Pruned branches from geometric match: {len(network_ids)} → {len(pruned_network_ids)} segments")

                    # Store the match in the return dictionary
                    pypsa_network_matches[pypsa_id] = pruned_network_ids
                else:
                    # If pruning removed all segments, use original match
                    print(f"  Warning: Branch pruning removed all segments from geometric match, using original match")
                    pypsa_network_matches[pypsa_id] = network_ids

        # Calculate statistics
        matched_count = len(pypsa_network_matches)
        total_pypsa = len(pypsa_gdf)
        match_percentage = matched_count / total_pypsa * 100 if total_pypsa > 0 else 0

        print(f"\nPyPSA matching results: {matched_count}/{total_pypsa} lines matched ({match_percentage:.1f}%)")

        return pypsa_network_matches

    # -------- helpers (fallback + safe formatting) ----------
    def _is_num(x):
        try:
            return x is not None and not pd.isna(x) and isinstance(float(x), float)
        except Exception:
            return False

    def _first_col(df, names):
        for n in names:
            if n in df.columns:
                return n
        return None

    def _to_float_or_none(value):
        try:
            return float(value)
        except (ValueError, TypeError):
            return None

    def _safe_format(value, format_str=".2f"):
        """Safely format a value with the given format string, returning a fallback if not possible."""
        if value is None or pd.isna(value):
            return "-"
        try:
            return f"{float(value):{format_str}}"
        except (ValueError, TypeError):
            return "-"

    def _get_len_km_from_result_or_gdf(res, gdf, jao_id_val):
        # prefer from result
        if _is_num(res.get('jao_length_km')):
            return float(res['jao_length_km'])
        if _is_num(res.get('jao_length')):
            return float(res['jao_length']) / 1000.0  # meters -> km

        # fallback to gdf by id
        id_col = _first_col(gdf, ['jao_id', 'JAO_ID', 'id', 'ID'])
        len_col = _first_col(gdf, ['jao_length_km', 'length_km', 'len_km', 'lengthkm', 'length'])
        if id_col and len_col:
            row = gdf.loc[gdf[id_col] == jao_id_val]
            if not row.empty:
                v = row.iloc[0][len_col]
                if _is_num(v):
                    v = float(v)
                    return v / 1000.0 if v > 1000 else v  # attempt meters->km if large
        return None

    def _get_jao_params(res, gdf):
        """Return dict with length_km, totals and per-km for r/x/b using result first, jao_gdf as fallback."""
        out = dict(length_km=None, r_total=None, x_total=None, b_total=None, r_km=None, x_km=None, b_km=None)
        jao_id_val = res.get('jao_id')

        L = _get_len_km_from_result_or_gdf(res, gdf, jao_id_val)
        if _is_num(L):
            out['length_km'] = float(L)

        # from result
        if _is_num(res.get('jao_r')): out['r_total'] = float(res['jao_r'])
        if _is_num(res.get('jao_x')): out['x_total'] = float(res['jao_x'])
        if _is_num(res.get('jao_b')): out['b_total'] = float(res['jao_b'])
        if _is_num(res.get('jao_r_per_km')): out['r_km'] = float(res['jao_r_per_km'])
        if _is_num(res.get('jao_x_per_km')): out['x_km'] = float(res['jao_x_per_km'])
        if _is_num(res.get('jao_b_per_km')): out['b_km'] = float(res['jao_b_per_km'])

        # if anything missing, try jao_gdf
        have_all = all(_is_num(out[k]) for k in ['r_total', 'x_total', 'b_total', 'r_km', 'x_km', 'b_km'])
        if not have_all:
            id_col = _first_col(jao_gdf, ['jao_id', 'JAO_ID', 'id', 'ID'])
            row = jao_gdf.loc[jao_gdf[id_col] == jao_id_val] if id_col else pd.DataFrame()
            if not row.empty:
                row = row.iloc[0]
                # totals candidates
                for Rc, Xc, Bc in [('R', 'X', 'B'),
                                   ('r_total', 'x_total', 'b_total'),
                                   ('r', 'x', 'b')]:
                    if Rc in row and Xc in row and Bc in row:
                        if _is_num(row[Rc]) and out['r_total'] is None: out['r_total'] = float(row[Rc])
                        if _is_num(row[Xc]) and out['x_total'] is None: out['x_total'] = float(row[Xc])
                        if _is_num(row[Bc]) and out['b_total'] is None: out['b_total'] = float(row[Bc])
                        break
                # per-km candidates
                for Rkc, Xkc, Bkc in [('R_per_km', 'X_per_km', 'B_per_km'),
                                      ('r_km', 'x_km', 'b_km'),
                                      ('r_per_km', 'x_per_km', 'b_per_km')]:
                    if Rkc in row and Xkc in row and Bkc in row:
                        if _is_num(row[Rkc]) and out['r_km'] is None: out['r_km'] = float(row[Rkc])
                        if _is_num(row[Xkc]) and out['x_km'] is None: out['x_km'] = float(row[Xkc])
                        if _is_num(row[Bkc]) and out['b_km'] is None: out['b_km'] = float(row[Bkc])
                        break

        # complete totals/per-km using length
        if _is_num(out['length_km']) and out['length_km'] > 0:
            L = out['length_km']
            if _is_num(out['r_total']) and not _is_num(out['r_km']): out['r_km'] = out['r_total'] / L
            if _is_num(out['x_total']) and not _is_num(out['x_km']): out['x_km'] = out['x_total'] / L
            if _is_num(out['b_total']) and not _is_num(out['b_km']): out['b_km'] = out['b_total'] / L
            if _is_num(out['r_km']) and not _is_num(out['r_total']): out['r_total'] = out['r_km'] * L
            if _is_num(out['x_km']) and not _is_num(out['x_total']): out['x_total'] = out['x_km'] * L
            if _is_num(out['b_km']) and not _is_num(out['b_total']): out['b_total'] = out['b_km'] * L
        return out

    def _convert_to_km(length_value):
        """Convert length to kilometers if it appears to be in meters"""
        if not _is_num(length_value):
            return None

        value = float(length_value)
        if value > 1000:  # Assume it's in meters if > 1000
            return value / 1000.0
        return value  # Already in km

    # -------- summary header ----------
    results_df = pd.DataFrame([r for r in matching_results if 'jao_id' in r])
    total_jao_lines = len(matching_results)
    matched_lines = sum(result.get('matched', False) for result in matching_results)
    unmatched_lines = total_jao_lines - matched_lines

    regular_matches = sum(1 for r in matching_results if r.get('matched', False)
                          and not r.get('is_duplicate', False)
                          and not r.get('is_geometric_match', False)
                          and not r.get('is_parallel_circuit', False)
                          and not r.get('is_parallel_voltage_circuit', False))
    duplicate_matches = sum(1 for r in matching_results if r.get('is_duplicate', False))
    geometric_matches = sum(1 for r in matching_results if r.get('is_geometric_match', False))
    parallel_matches = sum(1 for r in matching_results if r.get('is_parallel_circuit', False))
    parallel_voltage_matches = sum(1 for r in matching_results if r.get('is_parallel_voltage_circuit', False))

    match_quality_counts = {}
    for r in matching_results:
        q = r.get('match_quality', '')
        match_quality_counts[q] = match_quality_counts.get(q, 0) + 1

    pypsa_network_matches = {}
    if pypsa_gdf is not None and not pypsa_gdf.empty:
        try:
            # Match PyPSA lines to network lines using the path-based approach
            pypsa_network_matches = match_pypsa_to_network(pypsa_gdf, network_gdf)
            print(f"Found network matches for {len(pypsa_network_matches)} PyPSA lines")

            # Calculate how many PyPSA lines are matched
            matched_pypsa_count = sum(1 for matches in pypsa_network_matches.values() if matches)
            pypsa_matched_count = matched_pypsa_count
            pypsa_unmatched_count = len(pypsa_gdf) - matched_pypsa_count

            # Calculate percentages safely
            matched_percentage = pypsa_matched_count / len(pypsa_gdf) * 100 if len(pypsa_gdf) > 0 else 0
            unmatched_percentage = pypsa_unmatched_count / len(pypsa_gdf) * 100 if len(pypsa_gdf) > 0 else 0
        except Exception as e:
            print(f"Error matching PyPSA lines: {e}")
            import traceback
            traceback.print_exc()
            pypsa_matched_count = 0
            pypsa_unmatched_count = len(pypsa_gdf)
            matched_percentage = 0
            unmatched_percentage = 100

    # -------- HTML start ----------
    html_summary = f"""
    <html>
    <head>
        <title>JAO-Network-PyPSA Line Matching Summary with Electrical Parameters</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2, h3 {{ color: #333; }}
            .summary {{ margin-bottom: 20px; padding: 15px; background-color: #f5f5f5; border-radius: 5px; }}
            table {{ border-collapse: collapse; width: 100%; margin-top: 20px; margin-bottom: 30px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #4CAF50; color: white; }}
            .filter-controls {{ margin: 20px 0; padding: 10px; background-color: #eee; border-radius: 5px; }}
            .filter-buttons {{ display: flex; flex-wrap: wrap; gap: 5px; margin-bottom: 10px; }}
            .filter-buttons button {{ padding: 5px 10px; border: 1px solid #ccc; border-radius: 3px; background-color: #f8f8f8; cursor: pointer; }}
            .filter-buttons button:hover {{ background-color: #e0e0e0; }}
            .filter-buttons button.active {{ background-color: #4CAF50; color: white; }}
            .matched {{ background-color: #90EE90; }}
            .geometric {{ background-color: #FFDAB9; }}
            .duplicate {{ background-color: #E6E6FA; }}
            .parallel {{ background-color: #D8BFD8; }}
            .parallel-voltage {{ background-color: #FFE4B5; }}
            .unmatched {{ background-color: #ffcccb; }}
            .parameter-details {{ margin-left: 20px; margin-bottom: 30px; }}
            .segment-table {{ width: 95%; margin: 10px auto; }}
            .segment-table th {{ background-color: #5c85d6; }}
            .good-match {{ background-color: #c8e6c9; }}
            .moderate-match {{ background-color: #fff9c4; }}
            .poor-match {{ background-color: #ffccbc; }}
            .toggle-btn {{ background-color: #4CAF50; color: white; padding: 5px 10px; border: none; border-radius: 4px; cursor: pointer; margin-bottom: 10px; }}
            .details-section {{ display: none; }}
            .per-km-table {{ margin-top: 20px; width: 95%; margin-left: auto; margin-right: auto; }}
            .per-km-table th {{ background-color: #7b68ee; }}
            .low-coverage {{ color: #b71c1c; font-weight: bold; }}
            .totals-table th {{ background-color: #6a1b9a; }}
            .pypsa-table {{ margin-top: 20px; width: 95%; margin-left: auto; margin-right: auto; }}
            .pypsa-table th {{ background-color: #9c27b0; }}
            .tab-container {{ margin-top: 20px; }}
            .tab-buttons {{ display: flex; border-bottom: 1px solid #ddd; }}
            .tab-button {{ padding: 10px 15px; cursor: pointer; background-color: #f1f1f1; border: 1px solid #ddd; border-bottom: none; margin-right: 5px; border-radius: 5px 5px 0 0; }}
            .tab-button.active {{ background-color: #4CAF50; color: white; }}
            .tab-content {{ display: none; padding: 15px; border: 1px solid #ddd; border-top: none; }}
            .tab-content.active {{ display: block; }}
            .pypsa-matched {{ background-color: #90EE90; }}
            .pypsa-unmatched {{ background-color: #ffcccb; }}
            .modal {{ display: none; position: fixed; z-index: 1000; left: 0; top: 0; width: 100%; height: 100%; overflow: auto; background-color: rgba(0,0,0,0.4); }}
            .modal-content {{ background-color: #fefefe; margin: 15% auto; padding: 20px; border: 1px solid #888; width: 80%; max-width: 800px; }}
            .close {{ color: #aaa; float: right; font-size: 28px; font-weight: bold; cursor: pointer; }}
            .close:hover, .close:focus {{ color: black; text-decoration: none; cursor: pointer; }}
            .parameter-box {{ margin-top: 15px; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
            .parameter-row {{ display: flex; justify-content: space-between; margin-bottom: 8px; }}
            .parameter-label {{ font-weight: bold; min-width: 200px; }}
            .parameter-value {{ }}
        </style>
        <script>
            function filterTable() {{
                const filter = document.getElementById('filter').value.toLowerCase();
                const rows = document.getElementById('resultsTable').getElementsByTagName('tr');
                const visibleDetailIds = new Set();
                for (let i = 0; i < rows.length; i++) {{
                    const row = rows[i];
                    if (row.classList.contains('data-row')) {{
                        const text = row.textContent.toLowerCase();
                        const rowId = row.getAttribute('data-result-id');
                        if (text.includes(filter)) {{
                            row.style.display = '';
                            if (rowId) visibleDetailIds.add(rowId);
                        }} else {{
                            row.style.display = 'none';
                        }}
                    }}
                }}
                for (let i = 0; i < rows.length; i++) {{
                    const row = rows[i];
                    if (row.classList.contains('parameter-row')) {{
                        const rowId = row.getAttribute('data-result-id');
                        row.style.display = visibleDetailIds.has(rowId) ? '' : 'none';
                    }}
                }}
            }}
            function filterByMatchStatus(status) {{
                const rows = document.getElementById('resultsTable').getElementsByTagName('tr');
                const visibleDetailIds = new Set();
                for (let i = 0; i < rows.length; i++) {{
                    const row = rows[i];
                    if (row.classList.contains('data-row')) {{
                        const matchedCell = row.cells[3].textContent.trim();
                        const matchType = row.getAttribute('data-match-type');
                        const rowId = row.getAttribute('data-result-id');
                        let showRow = false;
                        if (status === 'all') {{ showRow = true; }}
                        else if (status === 'matched' && matchedCell === 'Yes' && matchType === 'regular') {{ showRow = true; }}
                        else if (status === 'geometric' && matchType === 'geometric') {{ showRow = true; }}
                        else if (status === 'duplicate' && matchType === 'duplicate') {{ showRow = true; }}
                        else if (status === 'parallel' && matchType === 'parallel') {{ showRow = true; }}
                        else if (status === 'parallel-voltage' && matchType === 'parallel-voltage') {{ showRow = true; }}
                        else if (status === 'unmatched' && matchedCell === 'No') {{ showRow = true; }}
                        if (showRow) {{ row.style.display = ''; if (rowId) visibleDetailIds.add(rowId); }}
                        else {{ row.style.display = 'none'; }}
                    }}
                }}
                for (let i = 0; i < rows.length; i++) {{
                    const row = rows[i];
                    if (row.classList.contains('parameter-row')) {{
                        const rowId = row.getAttribute('data-result-id');
                        row.style.display = visibleDetailIds.has(rowId) ? '' : 'none';
                    }}
                }}
            }}
            function filterByVoltage(voltage) {{
                const rows = document.getElementById('resultsTable').getElementsByTagName('tr');
                const visibleDetailIds = new Set();
                for (let i = 0; i < rows.length; i++) {{
                    const row = rows[i];
                    if (row.classList.contains('data-row')) {{
                        const voltageCell = row.cells[2].textContent.trim();
                        const rowId = row.getAttribute('data-result-id');
                        if (voltage === 'all' || voltageCell === voltage) {{
                            row.style.display = '';
                            if (rowId) visibleDetailIds.add(rowId);
                        }} else {{
                            row.style.display = 'none';
                        }}
                    }}
                }}
                for (let i = 0; i < rows.length; i++) {{
                    const row = rows[i];
                    if (row.classList.contains('parameter-row')) {{
                        const rowId = row.getAttribute('data-result-id');
                        row.style.display = visibleDetailIds.has(rowId) ? '' : 'none';
                    }}
                }}
            }}
            function filterByMatchQuality(quality) {{
                const rows = document.getElementById('resultsTable').getElementsByTagName('tr');
                const visibleDetailIds = new Set();
                for (let i = 0; i < rows.length; i++) {{
                    const row = rows[i];
                    if (row.classList.contains('data-row')) {{
                        const qualityCell = row.cells[7].textContent.trim();
                        const rowId = row.getAttribute('data-result-id');
                        if (quality === 'all' || qualityCell.includes(quality)) {{
                            row.style.display = '';
                            if (rowId) visibleDetailIds.add(rowId);
                        }} else {{
                            row.style.display = 'none';
                        }}
                    }}
                }}
                for (let i = 0; i < rows.length; i++) {{
                    const row = rows[i];
                    if (row.classList.contains('parameter-row')) {{
                        const rowId = row.getAttribute('data-result-id');
                        row.style.display = visibleDetailIds.has(rowId) ? '' : 'none';
                    }}
                }}
            }}
            function toggleDetails(id) {{
                var detailsSection = document.getElementById('details-' + id);
                var parameterRow = document.querySelector(`.parameter-row[data-result-id="${{id}}"]`);
                if (detailsSection.style.display === 'block') {{
                    detailsSection.style.display = 'none';
                    parameterRow.style.display = 'none';
                    document.getElementById('btn-' + id).textContent = 'Show Electrical Parameters';
                }} else {{
                    detailsSection.style.display = 'block';
                    parameterRow.style.display = '';
                    document.getElementById('btn-' + id).textContent = 'Hide Electrical Parameters';
                }}
            }}
            function switchTab(tabId) {{
                // Hide all tab contents
                const tabContents = document.getElementsByClassName('tab-content');
                for (let i = 0; i < tabContents.length; i++) {{
                    tabContents[i].classList.remove('active');
                }}

                // Deactivate all tab buttons
                const tabButtons = document.getElementsByClassName('tab-button');
                for (let i = 0; i < tabButtons.length; i++) {{
                    tabButtons[i].classList.remove('active');
                }}

                // Activate the selected tab and button
                document.getElementById(tabId).classList.add('active');
                document.querySelector(`[onclick="switchTab('${{tabId}}')"]`).classList.add('active');
            }}

            // New functions for PyPSA tab
            function filterPypsaTable() {{
                const filter = document.getElementById('pypsa-filter').value.toLowerCase();
                const rows = document.getElementById('pypsaTable').getElementsByTagName('tr');
                const visibleDetailIds = new Set();

                for (let i = 0; i < rows.length; i++) {{
                    const row = rows[i];
                    if (row.classList.contains('data-row')) {{
                        const text = row.textContent.toLowerCase();
                        const rowId = row.getAttribute('data-result-id');
                        if (text.includes(filter)) {{
                            row.style.display = '';
                            if (rowId) visibleDetailIds.add(rowId);
                        }} else {{
                            row.style.display = 'none';
                        }}
                    }}
                }}

                for (let i = 0; i < rows.length; i++) {{
                    const row = rows[i];
                    if (row.classList.contains('parameter-row')) {{
                        const rowId = row.getAttribute('data-result-id');
                        row.style.display = visibleDetailIds.has(rowId) ? '' : 'none';
                    }}
                }}
            }}

            function filterPypsaByMatchStatus(status) {{
                const rows = document.getElementById('pypsaTable').getElementsByTagName('tr');
                const visibleDetailIds = new Set();

                for (let i = 0; i < rows.length; i++) {{
                    const row = rows[i];
                    if (row.classList.contains('data-row')) {{
                        const matchedCell = row.cells[3].textContent.trim();
                        const rowId = row.getAttribute('data-result-id');
                        let showRow = false;

                        if (status === 'all') {{ showRow = true; }}
                        else if (status === 'matched' && matchedCell === 'Yes') {{ showRow = true; }}
                        else if (status === 'unmatched' && matchedCell === 'No') {{ showRow = true; }}

                        if (showRow) {{ 
                            row.style.display = ''; 
                            if (rowId) visibleDetailIds.add(rowId);
                        }}
                        else {{ 
                            row.style.display = 'none'; 
                        }}
                    }}
                }}

                for (let i = 0; i < rows.length; i++) {{
                    const row = rows[i];
                    if (row.classList.contains('parameter-row')) {{
                        const rowId = row.getAttribute('data-result-id');
                        row.style.display = visibleDetailIds.has(rowId) ? '' : 'none';
                    }}
                }}
            }}

            function filterPypsaByVoltage(voltage) {{
                const rows = document.getElementById('pypsaTable').getElementsByTagName('tr');
                const visibleDetailIds = new Set();

                for (let i = 0; i < rows.length; i++) {{
                    const row = rows[i];
                    if (row.classList.contains('data-row')) {{
                        const voltageCell = row.cells[2].textContent.trim();
                        const rowId = row.getAttribute('data-result-id');

                        if (voltage === 'all' || 
                            (voltage === '220' && voltageCell === '220') || 
                            (voltage === '400' && (voltageCell === '380' || voltageCell === '400'))) {{
                            row.style.display = '';
                            if (rowId) visibleDetailIds.add(rowId);
                        }} else {{
                            row.style.display = 'none';
                        }}
                    }}
                }}

                for (let i = 0; i < rows.length; i++) {{
                    const row = rows[i];
                    if (row.classList.contains('parameter-row')) {{
                        const rowId = row.getAttribute('data-result-id');
                        row.style.display = visibleDetailIds.has(rowId) ? '' : 'none';
                    }}
                }}
            }}
        </script>
    </head>
    <body>
        <h1>JAO-Network-PyPSA Line Matching Results with Electrical Parameters</h1>

        <div class="summary">
            <h2>Summary</h2>
            <p>Total JAO Lines: {total_jao_lines}</p>
            <p>Matched Lines: {matched_lines} ({matched_lines / total_jao_lines * 100:.1f}%)</p>
            <ul>
                <li>Regular matches: {regular_matches} ({regular_matches / total_jao_lines * 100:.1f}%)</li>
                <li>Geometric matches: {geometric_matches} ({geometric_matches / total_jao_lines * 100:.1f}%)</li>
                <li>Parallel circuits: {parallel_matches} ({parallel_matches / total_jao_lines * 100:.1f}%)</li>
                <li>Parallel voltage circuits: {parallel_voltage_matches} ({parallel_voltage_matches / total_jao_lines * 100:.1f}%)</li>
                <li>Duplicates: {duplicate_matches} ({duplicate_matches / total_jao_lines * 100:.1f}%)</li>
            </ul>
            <p>Unmatched Lines: {unmatched_lines} ({unmatched_lines / total_jao_lines * 100:.1f}%)</p>
            <p>Match Quality Details:</p>
            <ul>
    """

    for quality, count in match_quality_counts.items():
        if count > 0:
            percentage = count / total_jao_lines * 100
            html_summary += f"<li>{quality}: {count} ({percentage:.1f}%)</li>"

    html_summary += """
            </ul>
    """

    # Add PyPSA summary if available
    if pypsa_gdf is not None and not pypsa_gdf.empty:
        # Count by voltage level
        pypsa_220kv_count = len(pypsa_gdf[pypsa_gdf['voltage'].between(200, 300)])
        pypsa_400kv_count = len(pypsa_gdf[pypsa_gdf['voltage'] >= 300])
        pypsa_other_count = len(pypsa_gdf) - pypsa_220kv_count - pypsa_400kv_count

        # Identify matched PyPSA lines from matching_results if possible
        matched_pypsa_ids = set()
        for result in matching_results:
            if result.get('matched', False) and result.get('pypsa_ids'):
                for pypsa_id in result['pypsa_ids']:
                    matched_pypsa_ids.add(str(pypsa_id))

        # Use new matching results
        pypsa_matched_count = len(pypsa_network_matches)
        pypsa_unmatched_count = len(pypsa_gdf) - pypsa_matched_count

        # Calculate percentages safely
        matched_percentage = pypsa_matched_count / len(pypsa_gdf) * 100 if len(pypsa_gdf) > 0 else 0
        unmatched_percentage = pypsa_unmatched_count / len(pypsa_gdf) * 100 if len(pypsa_gdf) > 0 else 0

        html_summary += f"""
            <p>PyPSA Lines: {len(pypsa_gdf)}</p>
            <ul>
                <li>220 kV lines: {pypsa_220kv_count}</li>
                <li>400 kV lines: {pypsa_400kv_count}</li>
                <li>Other voltage lines: {pypsa_other_count}</li>
                <li>Matched lines: {pypsa_matched_count} ({matched_percentage:.1f}%)</li>
                <li>Unmatched lines: {pypsa_unmatched_count} ({unmatched_percentage:.1f}%)</li>
            </ul>
        """

    html_summary += """
        </div>

        <div class="filter-controls">
            <h2>Filter Results</h2>
            <input type="text" id="filter" onkeyup="filterTable()" placeholder="Search for JAO lines...">
            <h3>By Match Status:</h3>
            <div class="filter-buttons">
                <button onclick="filterByMatchStatus('all')">All</button>
                <button onclick="filterByMatchStatus('matched')">Regular Matches</button>
                <button onclick="filterByMatchStatus('geometric')">Geometric Matches</button>
                <button onclick="filterByMatchStatus('parallel')">Parallel Circuits</button>
                <button onclick="filterByMatchStatus('parallel-voltage')">Parallel Voltage</button>
                <button onclick="filterByMatchStatus('duplicate')">Duplicates</button>
                <button onclick="filterByMatchStatus('unmatched')">Unmatched</button>
            </div>
            <h3>By Voltage Level:</h3>
            <div class="filter-buttons">
                <button onclick="filterByVoltage('all')">All</button>
                <button onclick="filterByVoltage('220')">220 kV</button>
                <button onclick="filterByVoltage('400')">400 kV</button>
            </div>
            <h3>By Match Quality:</h3>
            <div class="filter-buttons">
                <button onclick="filterByMatchQuality('all')">All</button>
                <button onclick="filterByMatchQuality('Excellent')">Excellent</button>
                <button onclick="filterByMatchQuality('Good')">Good</button>
                <button onclick="filterByMatchQuality('Fair')">Fair</button>
                <button onclick="filterByMatchQuality('Poor')">Poor</button>
                <button onclick="filterByMatchQuality('Geometric')">Geometric</button>
                <button onclick="filterByMatchQuality('Parallel')">Parallel</button>
                <button onclick="filterByMatchQuality('No match')">Unmatched</button>
                <button onclick="filterByMatchQuality('No path')">No Path</button>
            </div>
        </div>
    """

    # Add tabs if PyPSA data is available
    if pypsa_gdf is not None and not pypsa_gdf.empty:
        html_summary += """
        <div class="tab-container">
            <div class="tab-buttons">
                <div class="tab-button active" onclick="switchTab('jao-tab')">JAO Lines</div>
                <div class="tab-button" onclick="switchTab('network-tab')">Network Lines</div>
                <div class="tab-button" onclick="switchTab('pypsa-tab')">PyPSA Lines</div>
                <div class="tab-button" onclick="switchTab('comparison-tab')">Parameter Comparison</div>
            </div>

            <div id="jao-tab" class="tab-content active">
        """

    html_summary += """
        <h2>Detailed Results</h2>
        <table id="resultsTable">
            <tr>
                <th>JAO ID</th>
                <th>JAO Name</th>
                <th>Voltage (kV)</th>
                <th>Matched</th>
                <th>Network IDs</th>
                <th>JAO Length (km)</th>
                <th>Length Ratio</th>
                <th>Match Quality</th>
                <th>Electrical Parameters</th>
            </tr>
    """

    for i, result in enumerate(matching_results):
        network_ids = ", ".join(result.get('network_ids', [])) if result.get('matched', False) and result.get(
            'network_ids') else "-"

        # safe length_ratio
        lr_val = result.get('length_ratio', None)
        length_ratio = _safe_format(lr_val)

        # safe JAO length cell (km)
        if _is_num(result.get('jao_length')):
            jao_length_km_cell = _safe_format(float(result['jao_length']) / 1000)
        elif _is_num(result.get('jao_length_km')):
            jao_length_km_cell = _safe_format(float(result['jao_length_km']))
        else:
            # try fallback from gdf
            L = _get_len_km_from_result_or_gdf(result, jao_gdf, result.get('jao_id'))
            jao_length_km_cell = _safe_format(L)

        if result.get('matched', False):
            if result.get('is_duplicate', False):
                css_class = "duplicate";
                match_type = "duplicate"
            elif result.get('is_geometric_match', False):
                css_class = "geometric";
                match_type = "geometric"
            elif result.get('is_parallel_circuit', False):
                css_class = "parallel";
                match_type = "parallel"
            elif result.get('is_parallel_voltage_circuit', False):
                css_class = "parallel-voltage";
                match_type = "parallel-voltage"
            else:
                css_class = "matched";
                match_type = "regular"
        else:
            css_class = "unmatched";
            match_type = "unmatched"

        result_id = f"result-{i}"

        html_summary += f"""
            <tr class="{css_class} data-row" data-result-id="{result_id}" data-match-type="{match_type}">
                <td>{result.get('jao_id', '-')}</td>
                <td>{result.get('jao_name', '-')}</td>
                <td>{result.get('v_nom', '-')}</td>
                <td>{"Yes" if result.get('matched', False) else "No"}</td>
                <td>{network_ids}</td>
                <td>{jao_length_km_cell}</td>
                <td>{length_ratio}</td>
                <td>{result.get('match_quality', '-')}</td>
                <td>
        """
        if result.get('matched', False):
            html_summary += f"""<button id="btn-{result_id}" class="toggle-btn" onclick="toggleDetails('{result_id}')">Show Electrical Parameters</button>"""
        else:
            html_summary += "N/A"

        html_summary += """
                </td>
            </tr>
        """

        if result.get('matched', False):
            html_summary += f"""
            <tr class="parameter-row" data-result-id="{result_id}" style="display: none;">
                <td colspan="9" class="parameter-details">
                    <div id="details-{result_id}" class="details-section">
                        <h3>JAO Line Electrical Parameters</h3>
            """

            # --- robust JAO param display (result or jao_gdf fallback) ---
            p = _get_jao_params(result, jao_gdf)
            if all(_is_num(p[k]) for k in ['r_total', 'x_total', 'b_total']):
                L = p['length_km'] if _is_num(p['length_km']) else 0.0
                html_summary += f"""
                        <p>Length: {_safe_format(L)} km</p>
                        <p>Resistance (R): {_safe_format(p['r_total'], '.6f')} ohm (Total)</p>
                        <p>Reactance (X): {_safe_format(p['x_total'], '.6f')} ohm (Total)</p>
                        <p>Susceptance (B): {_safe_format(p['b_total'], '.8f')} S (Total)</p>
                """
                if all(_is_num(p[k]) for k in ['r_km', 'x_km', 'b_km']):
                    html_summary += f"""
                        <p>Resistance per km (R): {_safe_format(p['r_km'], '.6f')} ohm/km</p>
                        <p>Reactance per km (X): {_safe_format(p['x_km'], '.6f')} ohm/km</p>
                        <p>Susceptance per km (B): {_safe_format(p['b_km'], '.8f')} S/km</p>
                    """
            else:
                html_summary += """
                        <p>Electrical parameter data not available for this JAO line</p>
                """

            # Coverage (optional fields)
            matched_km = result.get('matched_km', None)
            coverage = result.get('coverage_ratio', None)
            if _is_num(matched_km) and _is_num(p.get('length_km')):
                cov = float(coverage) if _is_num(coverage) else (
                    float(matched_km) / p['length_km'] if _is_num(p['length_km']) and p['length_km'] > 0 else None)
                cov_class = "low-coverage" if (_is_num(cov) and cov < 0.9) else ""
                if _is_num(cov):
                    html_summary += f"""
                        <h3>Coverage</h3>
                        <p class="{cov_class}">Matched length: {_safe_format(float(matched_km))} / {_safe_format(p['length_km'])} km ({_safe_format(cov * 100)}%)</p>
                    """

            # Totals consistency
            if result.get('jao_r') is not None and 'allocated_r_sum' in result:
                res_r = result.get('residual_r_percent', float('inf'))
                res_x = result.get('residual_x_percent', float('inf'))
                res_b = result.get('residual_b_percent', float('inf'))

                def fmt(v, digits=2):
                    return _safe_format(v, f".{digits}f") + "%" if v != float('inf') else "N/A"

                html_summary += f"""
                        <h3>Totals Consistency (Sum Allocated vs JAO Totals)</h3>
                        <table class="segment-table totals-table">
                            <tr><th>Quantity</th><th>JAO Total</th><th>Sum Allocated</th><th>Residual (JAO - Alloc)</th><th>Residual %</th></tr>
                            <tr><td>R (ohm)</td><td>{_safe_format(result.get('jao_r', 0), '.6f')}</td><td>{_safe_format(result.get('allocated_r_sum', 0), '.6f')}</td><td>{_safe_format(result.get('jao_r', 0) - result.get('allocated_r_sum', 0), '.6f')}</td><td>{fmt(res_r)}</td></tr>
                            <tr><td>X (ohm)</td><td>{_safe_format(result.get('jao_x', 0), '.6f')}</td><td>{_safe_format(result.get('allocated_x_sum', 0), '.6f')}</td><td>{_safe_format(result.get('jao_x', 0) - result.get('allocated_x_sum', 0), '.6f')}</td><td>{fmt(res_x)}</td></tr>
                            <tr><td>B (S)</td><td>{_safe_format(result.get('jao_b', 0), '.8f')}</td><td>{_safe_format(result.get('allocated_b_sum', 0), '.8f')}</td><td>{_safe_format(result.get('jao_b', 0) - result.get('allocated_b_sum', 0), '.8f')}</td><td>{fmt(res_b)}</td></tr>
                        </table>
                """

            # Segment tables section
            if result.get('matched_lines_data'):
                html_summary += """
                        <h3>Allocated Parameters for Network Segments (Total Values)</h3>
                        <table class="segment-table">
                            <tr>
                                <th>Network ID</th><th>Length (km)</th><th>Circuits</th><th>Length Ratio</th>
                                <th>Allocated R (ohm)</th><th>Original R (ohm)</th><th>R Diff (%)</th>
                                <th>Allocated X (ohm)</th><th>Original X (ohm)</th><th>X Diff (%)</th>
                                <th>Allocated B (S)</th><th>Original B (S)</th><th>B Diff (%)</th>
                                <th>Allocation Status</th>
                            </tr>
                """
                for seg in result['matched_lines_data']:
                    def pct(v):
                        return _safe_format(v, ".2f") + "%" if v != float('inf') else "N/A"

                    def cls(v):
                        return "good-match" if abs(v) <= 20 else ("moderate-match" if abs(v) <= 50 else "poor-match")

                    rdp = seg.get('r_diff_percent', float('inf'))
                    xdp = seg.get('x_diff_percent', float('inf'))
                    bdp = seg.get('b_diff_percent', float('inf'))

                    # Get allocation status
                    status = seg.get('allocation_status', 'Unknown')

                    html_summary += f"""
                            <tr>
                                <td>{seg.get('network_id', '-')}</td>
                                <td>{_safe_format(seg.get('length_km', 0))}</td>
                                <td>{seg.get('num_parallel', 1)}</td>
                                <td>{_safe_format(seg.get('segment_ratio', 0) * 100)}%</td>
                                <td>{_safe_format(seg.get('allocated_r', 0), '.6f')}</td>
                                <td>{_safe_format(seg.get('original_r', 0), '.6f')}</td>
                                <td class="{cls(rdp)}">{pct(rdp)}</td>
                                <td>{_safe_format(seg.get('allocated_x', 0), '.6f')}</td>
                                <td>{_safe_format(seg.get('original_x', 0), '.6f')}</td>
                                <td class="{cls(xdp)}">{pct(xdp)}</td>
                                <td>{_safe_format(seg.get('allocated_b', 0), '.8f')}</td>
                                <td>{_safe_format(seg.get('original_b', 0), '.8f')}</td>
                                <td class="{cls(bdp)}">{pct(bdp)}</td>
                                <td>{status}</td>
                            </tr>
                    """
                html_summary += """
                        </table>
                        <h3>Per-Kilometer Parameters for Network Segments</h3>
                        <table class="per-km-table">
                            <tr>
                                <th>Network ID</th><th>Length (km)</th><th>Circuits</th>
                                <th>Allocated R (ohm/km)</th><th>Original R (ohm/km)</th>
                                <th>Allocated X (ohm/km)</th><th>Original X (ohm/km)</th>
                                <th>Allocated B (S/km)</th><th>Original B (S/km)</th>
                            </tr>
                """
                for seg in result['matched_lines_data']:
                    html_summary += f"""
                            <tr>
                                <td>{seg.get('network_id', '-')}</td>
                                <td>{_safe_format(seg.get('length_km', 0))}</td>
                                <td>{seg.get('num_parallel', 1)}</td>
                                <td>{_safe_format(seg.get('allocated_r_per_km', 0), '.6f')}</td>
                                <td>{_safe_format(seg.get('original_r_per_km', 0), '.6f')}</td>
                                <td>{_safe_format(seg.get('allocated_x_per_km', 0), '.6f')}</td>
                                <td>{_safe_format(seg.get('original_x_per_km', 0), '.6f')}</td>
                                <td>{_safe_format(seg.get('allocated_b_per_km', 0), '.8f')}</td>
                                <td>{_safe_format(seg.get('original_b_per_km', 0), '.8f')}</td>
                            </tr>
                    """
                # Per-Circuit Parameters Comparison section
                html_summary += """
                        <h3>Per-Circuit Parameters Comparison</h3>
                        <table class="per-km-table">
                            <tr>
                                <th>Network ID</th><th>Circuits</th>
                                <th>JAO R (ohm/km/circuit)</th><th>Network R (ohm/km/circuit)</th><th>R Diff (%)</th>
                                <th>JAO X (ohm/km/circuit)</th><th>Network X (ohm/km/circuit)</th><th>X Diff (%)</th>
                                <th>JAO B (S/km/circuit)</th><th>Network B (S/km/circuit)</th><th>B Diff (%)</th>
                            </tr>
                """
                for seg in result['matched_lines_data']:
                    r_diff_pc = seg.get('r_diff_percent_pc', float('inf'))
                    x_diff_pc = seg.get('x_diff_percent_pc', float('inf'))
                    b_diff_pc = seg.get('b_diff_percent_pc', float('inf'))

                    # Make sure we use the correct values
                    jao_r_pc = seg.get('jao_r_per_km_pc', 0)
                    jao_x_pc = seg.get('jao_x_per_km_pc', 0)
                    jao_b_pc = seg.get('jao_b_per_km_pc', 0)

                    orig_r_pc = seg.get('original_r_per_km_pc', 0)
                    orig_x_pc = seg.get('original_x_per_km_pc', 0)
                    orig_b_pc = seg.get('original_b_per_km_pc', 0)

                    html_summary += f"""
                            <tr>
                                <td>{seg.get('network_id', '-')}</td>
                                <td>{seg.get('num_parallel', 1)}</td>
                                <td>{_safe_format(jao_r_pc, '.6f')}</td>
                                <td>{_safe_format(orig_r_pc, '.6f')}</td>
                                <td class="{cls(r_diff_pc)}">{pct(r_diff_pc)}</td>
                                <td>{_safe_format(jao_x_pc, '.6f')}</td>
                                <td>{_safe_format(orig_x_pc, '.6f')}</td>
                                <td class="{cls(x_diff_pc)}">{pct(x_diff_pc)}</td>
                                <td>{_safe_format(jao_b_pc, '.8f')}</td>
                                <td>{_safe_format(orig_b_pc, '.8f')}</td>
                                <td class="{cls(b_diff_pc)}">{pct(b_diff_pc)}</td>
                            </tr>
                    """
                html_summary += """
                        </table>
                """

                # Add PyPSA comparison if available for this JAO line
                if pypsa_gdf is not None and not pypsa_gdf.empty:
                    # Check for matching PyPSA lines for this JAO
                    matching_pypsa = []
                    for r in matching_results:
                        if r.get('jao_id') == result.get('jao_id') and r.get('pypsa_ids'):
                            for pypsa_id in r['pypsa_ids']:
                                # Find row in pypsa_gdf where id matches
                                matching_rows = pypsa_gdf[pypsa_gdf['id'] == pypsa_id]
                                if not matching_rows.empty:
                                    matching_pypsa.append(matching_rows.iloc[0])

                    if matching_pypsa:
                        html_summary += """
                            <h3>Matched PyPSA Lines</h3>
                            <table class="pypsa-table">
                                <tr>
                                    <th>PyPSA ID</th>
                                    <th>Voltage (kV)</th>
                                    <th>Length (km)</th>
                                    <th>R (ohm)</th>
                                    <th>X (ohm)</th>
                                    <th>B (S)</th>
                                    <th>R per km (ohm/km)</th>
                                    <th>X per km (ohm/km)</th>
                                    <th>B per km (S/km)</th>
                                </tr>
                        """

                        for pypsa_line in matching_pypsa:
                            pypsa_id = str(pypsa_line.get('id', ''))
                            voltage = int(pypsa_line['voltage']) if 'voltage' in pypsa_line and pd.notna(
                                pypsa_line['voltage']) else '-'

                            # Get length (convert from meters if needed)
                            length_m = pypsa_line.get('length', 0) if 'length' in pypsa_line and pd.notna(
                                pypsa_line['length']) else 0
                            length_km = _convert_to_km(length_m)
                            length_km_str = _safe_format(length_km)

                            # Get electrical parameters
                            r_total = '-'
                            x_total = '-'
                            b_total = '-'

                            if 'r_ohm' in pypsa_line and pd.notna(pypsa_line['r_ohm']):
                                r_total = _safe_format(float(pypsa_line['r_ohm']), '.6f')
                            elif 'r' in pypsa_line and pd.notna(pypsa_line['r']):
                                r_total = _safe_format(float(pypsa_line['r']), '.6f')

                            if 'x_ohm' in pypsa_line and pd.notna(pypsa_line['x_ohm']):
                                x_total = _safe_format(float(pypsa_line['x_ohm']), '.6f')
                            elif 'x' in pypsa_line and pd.notna(pypsa_line['x']):
                                x_total = _safe_format(float(pypsa_line['x']), '.6f')

                            if 'b_siemens' in pypsa_line and pd.notna(pypsa_line['b_siemens']):
                                b_total = _safe_format(float(pypsa_line['b_siemens']), '.8f')
                            elif 'b' in pypsa_line and pd.notna(pypsa_line['b']):
                                b_total = _safe_format(float(pypsa_line['b']), '.8f')

                            # Calculate per-km parameters if length is available
                            r_per_km = '-'
                            x_per_km = '-'
                            b_per_km = '-'

                            if length_km is not None and length_km > 0:
                                if r_total != '-' and _is_num(pypsa_line.get('r')):
                                    r_val = float(pypsa_line['r'])
                                    r_per_km = _safe_format(r_val / length_km, '.6f')

                                if x_total != '-' and _is_num(pypsa_line.get('x')):
                                    x_val = float(pypsa_line['x'])
                                    x_per_km = _safe_format(x_val / length_km, '.6f')

                                if b_total != '-' and _is_num(pypsa_line.get('b')):
                                    b_val = float(pypsa_line['b'])
                                    b_per_km = _safe_format(b_val / length_km, '.8f')

                            html_summary += f"""
                                <tr>
                                    <td>{pypsa_id}</td>
                                    <td>{voltage}</td>
                                    <td>{length_km_str}</td>
                                    <td>{r_total}</td>
                                    <td>{x_total}</td>
                                    <td>{b_total}</td>
                                    <td>{r_per_km}</td>
                                    <td>{x_per_km}</td>
                                    <td>{b_per_km}</td>
                                </tr>
                            """

                        html_summary += """
                            </table>
                        """
                    else:
                        html_summary += """
                            <h3>Matched PyPSA Lines</h3>
                            <p>No PyPSA lines are matched to this JAO line.</p>
                        """
            else:
                html_summary += """
                        <p>No network segment parameter data available for this match</p>
                """

            html_summary += """
                    </div>
                </td>
            </tr>
            """

    html_summary += """
        </table>
    """

    # If PyPSA data is available, add tabs with network and PyPSA details
    if pypsa_gdf is not None and not pypsa_gdf.empty:
        html_summary += """
            </div> <!-- end jao-tab -->

            <div id="network-tab" class="tab-content">
                <h2>Network Line Parameters</h2>
                <table>
                    <tr>
                        <th>Network ID</th>
                        <th>Voltage (kV)</th>
                        <th>Circuits</th>
                        <th>Length (km)</th>
                        <th>R (ohm)</th>
                        <th>X (ohm)</th>
                        <th>B (S)</th>
                        <th>R per km (ohm/km)</th>
                        <th>X per km (ohm/km)</th>
                        <th>B per km (S/km)</th>
                        <th>Matched to JAO</th>
                    </tr>
        """

        # Add network lines
        matched_network_ids = set()
        for result in matching_results:
            if result.get('matched') and result.get('network_ids'):
                for network_id in result['network_ids']:
                    matched_network_ids.add(str(network_id))

        for _, row in network_gdf.iterrows():
            network_id = str(row['id'])
            voltage = int(row['v_nom']) if 'v_nom' in row and pd.notna(row['v_nom']) else '-'

            # Get circuits
            circuits = 1
            for col in ['circuits', 'num_parallel']:
                if col in row and pd.notna(row[col]):
                    circuits = int(row[col])
                    break

            # Get length and convert to km if needed
            length_value = None
            for col in ['length_km', 'length']:
                if col in row and pd.notna(row[col]):
                    length_value = row[col]
                    break

            length_km = _convert_to_km(length_value)
            length_km_str = _safe_format(length_km)

            # Get electrical parameters
            r_total = '-'
            x_total = '-'
            b_total = '-'

            for r_col in ['r', 'R', 'r_total', 'R_total']:
                if r_col in row and pd.notna(row[r_col]):
                    r_total = _safe_format(float(row[r_col]), '.6f')
                    break

            for x_col in ['x', 'X', 'x_total', 'X_total']:
                if x_col in row and pd.notna(row[x_col]):
                    x_total = _safe_format(float(row[x_col]), '.6f')
                    break

            for b_col in ['b', 'B', 'b_total', 'B_total']:
                if b_col in row and pd.notna(row[b_col]):
                    b_total = _safe_format(float(row[b_col]), '.8f')
                    break

            # Get per-km parameters
            r_per_km = '-'
            x_per_km = '-'
            b_per_km = '-'

            for r_km_col in ['r_per_km', 'R_per_km', 'r_km', 'R_km']:
                if r_km_col in row and pd.notna(row[r_km_col]):
                    r_per_km = _safe_format(float(row[r_km_col]), '.6f')
                    break

            for x_km_col in ['x_per_km', 'X_per_km', 'x_km', 'X_km']:
                if x_km_col in row and pd.notna(row[x_km_col]):
                    x_per_km = _safe_format(float(row[x_km_col]), '.6f')
                    break

            for b_km_col in ['b_per_km', 'B_per_km', 'b_km', 'B_km']:
                if b_km_col in row and pd.notna(row[b_km_col]):
                    b_per_km = _safe_format(float(row[b_km_col]), '.8f')
                    break

            # Calculate per-km values if not already available
            if r_per_km == '-' and r_total != '-' and length_km is not None and length_km > 0:
                r_val = float(row.get('r', 0))
                r_per_km = _safe_format(r_val / length_km, '.6f')

            if x_per_km == '-' and x_total != '-' and length_km is not None and length_km > 0:
                x_val = float(row.get('x', 0))
                x_per_km = _safe_format(x_val / length_km, '.6f')

            if b_per_km == '-' and b_total != '-' and length_km is not None and length_km > 0:
                b_val = float(row.get('b', 0))
                b_per_km = _safe_format(b_val / length_km, '.8f')

            # Is this network line matched to a JAO line?
            is_matched = network_id in matched_network_ids

            # Add row
            row_class = "matched" if is_matched else "unmatched"
            html_summary += f"""
                        <tr class="{row_class}">
                            <td>{network_id}</td>
                            <td>{voltage}</td>
                            <td>{circuits}</td>
                            <td>{length_km_str}</td>
                            <td>{r_total}</td>
                            <td>{x_total}</td>
                            <td>{b_total}</td>
                            <td>{r_per_km}</td>
                            <td>{x_per_km}</td>
                            <td>{b_per_km}</td>
                            <td>{"Yes" if is_matched else "No"}</td>
                        </tr>
            """

        html_summary += """
                </table>
            </div> <!-- end network-tab -->

            <div id="pypsa-tab" class="tab-content">
                <h2>PyPSA Lines</h2>
                <div class="filter-controls">
                    <h2>Filter Results</h2>
                    <input type="text" id="pypsa-filter" onkeyup="filterPypsaTable()" placeholder="Search for PyPSA lines...">
                    <h3>By Match Status:</h3>
                    <div class="filter-buttons">
                        <button onclick="filterPypsaByMatchStatus('all')">All</button>
                        <button onclick="filterPypsaByMatchStatus('matched')">Matched</button>
                        <button onclick="filterPypsaByMatchStatus('unmatched')">Unmatched</button>
                    </div>
                    <h3>By Voltage Level:</h3>
                    <div class="filter-buttons">
                        <button onclick="filterPypsaByVoltage('all')">All</button>
                        <button onclick="filterPypsaByVoltage('220')">220 kV</button>
                        <button onclick="filterPypsaByVoltage('400')">400 kV</button>
                    </div>
                </div>

                <table id="pypsaTable">
                    <tr>
                        <th>PyPSA ID</th>
                        <th>Bus0-Bus1</th>
                        <th>Voltage (kV)</th>
                        <th>Matched</th>
                        <th>Network IDs</th>
                        <th>PyPSA Length (km)</th>
                        <th>Length Ratio</th>
                        <th>Match Quality</th>
                        <th>Electrical Parameters</th>
                    </tr>
        """

        # Generate PyPSA rows with the same structure as JAO rows
        for i, row in pypsa_gdf.iterrows():
            pypsa_id = str(row.get('id', i))
            voltage = int(row.get('voltage', 0))
            bus0 = str(row.get('bus0', ''))
            bus1 = str(row.get('bus1', ''))

            # Get original length (in meters) and convert to km
            length_m = row.get('length', 0) if 'length' in row and pd.notna(row['length']) else 0
            length_km = _convert_to_km(length_m)

            # Get network matches for this PyPSA line using the path-based approach
            network_ids = pypsa_network_matches.get(pypsa_id, [])
            is_matched = len(network_ids) > 0

            # Format network IDs for display
            network_ids_display = ", ".join(network_ids) if network_ids else "N/A"

            # Length ratio (if matched)
            length_ratio = "N/A"
            if is_matched and length_km is not None and length_km > 0:
                # Calculate total length of matched network lines
                network_length = 0
                for net_id in network_ids:
                    net_rows = network_gdf[network_gdf['id'].astype(str) == net_id]
                    if not net_rows.empty:
                        net_length = net_rows.iloc[0].get('length', 0)
                        if pd.notna(net_length):
                            net_length_km = _convert_to_km(net_length)
                            if net_length_km is not None:
                                network_length += net_length_km

                if network_length > 0:
                    length_ratio = _safe_format(network_length / length_km)

            # Create row with expandable details like JAO rows
            result_id = f"pypsa-{i}"

            # CSS class based on match status
            row_class = "matched" if is_matched else "unmatched"
            match_type = "regular" if is_matched else "unmatched"

            # Calculate per-km values if length is available
            r_total = row.get('r', 0) if 'r' in row and pd.notna(row['r']) else 0
            x_total = row.get('x', 0) if 'x' in row and pd.notna(row['x']) else 0
            b_total = row.get('b', 0) if 'b' in row and pd.notna(row['b']) else 0

            if length_km is not None and length_km > 0:
                r_per_km = r_total / length_km
                x_per_km = x_total / length_km
                b_per_km = b_total / length_km
            else:
                r_per_km = 0
                x_per_km = 0
                b_per_km = 0

            # Determine match quality
            match_quality = "No match"
            if is_matched:
                match_quality = "Path-Based Match" if any("Path" in m for m in network_ids) else "Geometric Match"

            # Main table row
            html_summary += f"""
            <tr class="{row_class} data-row" data-result-id="{result_id}" data-match-type="{match_type}">
                <td>{pypsa_id}</td>
                <td>{bus0}-{bus1}</td>
                <td>{voltage}</td>
                <td>{"Yes" if is_matched else "No"}</td>
                <td>{network_ids_display}</td>
                <td>{_safe_format(length_km)}</td>
                <td>{length_ratio}</td>
                <td>{match_quality}</td>
                <td>
                    <button id="btn-{result_id}" class="toggle-btn" onclick="toggleDetails('{result_id}')">Show Electrical Parameters</button>
                </td>
            </tr>
            """

            # Prepare network segments data if matched
            network_segments_html = ""
            if is_matched:
                # Total allocated parameters
                allocated_r_sum = 0
                allocated_x_sum = 0
                allocated_b_sum = 0

                # Create tables for matched network segments similar to JAO format
                network_segments_rows = []

                for net_id in network_ids:
                    net_rows = network_gdf[network_gdf['id'].astype(str) == net_id]
                    if not net_rows.empty:
                        net_row = net_rows.iloc[0]

                        # Get network line details
                        net_length = float(net_row.get('length', 0)) if pd.notna(net_row.get('length', 0)) else 0
                        net_length_km = _convert_to_km(net_length)
                        net_circuits = int(net_row.get('circuits', 1)) if pd.notna(net_row.get('circuits', 1)) else 1

                        # Calculate segment ratio
                        segment_ratio = net_length_km / length_km if length_km is not None and length_km > 0 else 0

                        # Get network electrical parameters
                        net_r = float(net_row.get('r', 0)) if pd.notna(net_row.get('r', 0)) else 0
                        net_x = float(net_row.get('x', 0)) if pd.notna(net_row.get('x', 0)) else 0
                        net_b = float(net_row.get('b', 0)) if pd.notna(net_row.get('b', 0)) else 0

                        # Calculate allocated parameters based on segment ratio
                        allocated_r = r_total * segment_ratio
                        allocated_x = x_total * segment_ratio
                        allocated_b = b_total * segment_ratio

                        # Add to sum
                        allocated_r_sum += allocated_r
                        allocated_x_sum += allocated_x
                        allocated_b_sum += allocated_b

                        # Calculate difference percentages
                        r_diff_pct = ((allocated_r - net_r) / net_r * 100) if net_r != 0 else float('inf')
                        x_diff_pct = ((allocated_x - net_x) / net_x * 100) if net_x != 0 else float('inf')
                        b_diff_pct = ((allocated_b - net_b) / net_b * 100) if net_b != 0 else float('inf')

                        # Format percentages
                        r_diff_pct_str = _safe_format(r_diff_pct, ".2f") + "%" if r_diff_pct != float('inf') else "N/A"
                        x_diff_pct_str = _safe_format(x_diff_pct, ".2f") + "%" if x_diff_pct != float('inf') else "N/A"
                        b_diff_pct_str = _safe_format(b_diff_pct, ".2f") + "%" if b_diff_pct != float('inf') else "N/A"

                        # Determine CSS classes for differences
                        r_class = "good-match" if abs(r_diff_pct) <= 20 else (
                            "moderate-match" if abs(r_diff_pct) <= 50 else "poor-match")
                        x_class = "good-match" if abs(x_diff_pct) <= 20 else (
                            "moderate-match" if abs(x_diff_pct) <= 50 else "poor-match")
                        b_class = "good-match" if abs(b_diff_pct) <= 20 else (
                            "moderate-match" if abs(b_diff_pct) <= 50 else "poor-match")

                        network_segments_rows.append(f"""
                        <tr>
                            <td>{net_id}</td>
                            <td>{_safe_format(net_length_km)}</td>
                            <td>{net_circuits}</td>
                            <td>{_safe_format(segment_ratio * 100)}%</td>
                            <td>{_safe_format(allocated_r, '.6f')}</td>
                            <td>{_safe_format(net_r, '.6f')}</td>
                            <td class="{r_class}">{r_diff_pct_str}</td>
                            <td>{_safe_format(allocated_x, '.6f')}</td>
                            <td>{_safe_format(net_x, '.6f')}</td>
                            <td class="{x_class}">{x_diff_pct_str}</td>
                            <td>{_safe_format(allocated_b, '.8f')}</td>
                            <td>{_safe_format(net_b, '.8f')}</td>
                            <td class="{b_class}">{b_diff_pct_str}</td>
                            <td>Applied</td>
                        </tr>
                        """)

                # Calculate residuals
                residual_r = r_total - allocated_r_sum
                residual_x = x_total - allocated_x_sum
                residual_b = b_total - allocated_b_sum

                residual_r_pct = (residual_r / r_total * 100) if r_total != 0 else float('inf')
                residual_x_pct = (residual_x / x_total * 100) if x_total != 0 else float('inf')
                residual_b_pct = (residual_b / b_total * 100) if b_total != 0 else float('inf')

                # Add totals consistency table
                consistency_html = f"""
                <h3>Totals Consistency (Sum Allocated vs PyPSA Totals)</h3>
                <table class="segment-table totals-table">
                    <tr><th>Quantity</th><th>PyPSA Total</th><th>Sum Allocated</th><th>Residual (PyPSA - Alloc)</th><th>Residual %</th></tr>
                    <tr><td>R (ohm)</td><td>{_safe_format(r_total, '.6f')}</td><td>{_safe_format(allocated_r_sum, '.6f')}</td><td>{_safe_format(residual_r, '.6f')}</td><td>{_safe_format(residual_r_pct, '.2f')}%</td></tr>
                    <tr><td>X (ohm)</td><td>{_safe_format(x_total, '.6f')}</td><td>{_safe_format(allocated_x_sum, '.6f')}</td><td>{_safe_format(residual_x, '.6f')}</td><td>{_safe_format(residual_x_pct, '.2f')}%</td></tr>
                    <tr><td>B (S)</td><td>{_safe_format(b_total, '.8f')}</td><td>{_safe_format(allocated_b_sum, '.8f')}</td><td>{_safe_format(residual_b, '.8f')}</td><td>{_safe_format(residual_b_pct, '.2f')}%</td></tr>
                </table>
                """

                if network_segments_rows:
                    network_segments_html = f"""
                    {consistency_html}

                    <h3>Allocated Parameters for Network Segments (Total Values)</h3>
                    <table class="segment-table">
                        <tr>
                            <th>Network ID</th><th>Length (km)</th><th>Circuits</th><th>Length Ratio</th>
                            <th>Allocated R (ohm)</th><th>Original R (ohm)</th><th>R Diff (%)</th>
                            <th>Allocated X (ohm)</th><th>Original X (ohm)</th><th>X Diff (%)</th>
                            <th>Allocated B (S)</th><th>Original B (S)</th><th>B Diff (%)</th>
                            <th>Allocation Status</th>
                        </tr>
                        {"".join(network_segments_rows)}
                    </table>

                    <h3>Per-Kilometer Parameters for Network Segments</h3>
                    <table class="per-km-table">
                        <tr>
                            <th>Network ID</th><th>Length (km)</th><th>Circuits</th>
                            <th>Allocated R (ohm/km)</th><th>Original R (ohm/km)</th>
                            <th>Allocated X (ohm/km)</th><th>Original X (ohm/km)</th>
                            <th>Allocated B (S/km)</th><th>Original B (S/km)</th>
                        </tr>
                    """

                    # Add per-km values for each segment
                    for net_id in network_ids:
                        net_rows = network_gdf[network_gdf['id'].astype(str) == net_id]
                        if not net_rows.empty:
                            net_row = net_rows.iloc[0]

                            net_length = float(net_row.get('length', 0)) if pd.notna(net_row.get('length', 0)) else 0
                            net_length_km = _convert_to_km(net_length)
                            net_circuits = int(net_row.get('circuits', 1)) if pd.notna(
                                net_row.get('circuits', 1)) else 1

                            segment_ratio = net_length_km / length_km if length_km is not None and length_km > 0 else 0

                            # Calculate per-km values
                            allocated_r = r_total * segment_ratio
                            allocated_x = x_total * segment_ratio
                            allocated_b = b_total * segment_ratio

                            allocated_r_per_km = allocated_r / net_length_km if net_length_km is not None and net_length_km > 0 else 0
                            allocated_x_per_km = allocated_x / net_length_km if net_length_km is not None and net_length_km > 0 else 0
                            allocated_b_per_km = allocated_b / net_length_km if net_length_km is not None and net_length_km > 0 else 0

                            net_r_per_km = float(net_row.get('r',
                                                             0)) / net_length_km if net_length_km is not None and net_length_km > 0 else 0
                            net_x_per_km = float(net_row.get('x',
                                                             0)) / net_length_km if net_length_km is not None and net_length_km > 0 else 0
                            net_b_per_km = float(net_row.get('b',
                                                             0)) / net_length_km if net_length_km is not None and net_length_km > 0 else 0

                            network_segments_html += f"""
                            <tr>
                                <td>{net_id}</td>
                                <td>{_safe_format(net_length_km)}</td>
                                <td>{net_circuits}</td>
                                <td>{_safe_format(allocated_r_per_km, '.6f')}</td>
                                <td>{_safe_format(net_r_per_km, '.6f')}</td>
                                <td>{_safe_format(allocated_x_per_km, '.6f')}</td>
                                <td>{_safe_format(net_x_per_km, '.6f')}</td>
                                <td>{_safe_format(allocated_b_per_km, '.8f')}</td>
                                <td>{_safe_format(net_b_per_km, '.8f')}</td>
                            </tr>
                            """

                    network_segments_html += """
                    </table>

                    <h3>Per-Circuit Parameters Comparison</h3>
                    <table class="per-km-table">
                        <tr>
                            <th>Network ID</th><th>Circuits</th>
                            <th>PyPSA R (ohm/km/circuit)</th><th>Network R (ohm/km/circuit)</th><th>R Diff (%)</th>
                            <th>PyPSA X (ohm/km/circuit)</th><th>Network X (ohm/km/circuit)</th><th>X Diff (%)</th>
                            <th>PyPSA B (S/km/circuit)</th><th>Network B (S/km/circuit)</th><th>B Diff (%)</th>
                        </tr>
                    """

                    # Add per-circuit comparisons for each segment
                    for net_id in network_ids:
                        net_rows = network_gdf[network_gdf['id'].astype(str) == net_id]
                        if not net_rows.empty:
                            net_row = net_rows.iloc[0]

                            net_length = float(net_row.get('length', 0)) if pd.notna(net_row.get('length', 0)) else 0
                            net_length_km = _convert_to_km(net_length)
                            net_circuits = int(net_row.get('circuits', 1)) if pd.notna(
                                net_row.get('circuits', 1)) else 1

                            # PyPSA per-circuit values
                            pypsa_r_per_km_pc = r_per_km / net_circuits if net_circuits > 0 else 0
                            pypsa_x_per_km_pc = x_per_km / net_circuits if net_circuits > 0 else 0
                            pypsa_b_per_km_pc = b_per_km / net_circuits if net_circuits > 0 else 0

                            # Network per-circuit values
                            net_r_per_km = float(net_row.get('r',
                                                             0)) / net_length_km if net_length_km is not None and net_length_km > 0 else 0
                            net_x_per_km = float(net_row.get('x',
                                                             0)) / net_length_km if net_length_km is not None and net_length_km > 0 else 0
                            net_b_per_km = float(net_row.get('b',
                                                             0)) / net_length_km if net_length_km is not None and net_length_km > 0 else 0

                            net_r_per_km_pc = net_r_per_km / net_circuits if net_circuits > 0 else 0
                            net_x_per_km_pc = net_x_per_km / net_circuits if net_circuits > 0 else 0
                            net_b_per_km_pc = net_b_per_km / net_circuits if net_circuits > 0 else 0

                            # Calculate differences
                            r_diff_pc = ((
                                                     pypsa_r_per_km_pc - net_r_per_km_pc) / net_r_per_km_pc * 100) if net_r_per_km_pc != 0 else float(
                                'inf')
                            x_diff_pc = ((
                                                     pypsa_x_per_km_pc - net_x_per_km_pc) / net_x_per_km_pc * 100) if net_x_per_km_pc != 0 else float(
                                'inf')
                            b_diff_pc = ((
                                                     pypsa_b_per_km_pc - net_b_per_km_pc) / net_b_per_km_pc * 100) if net_b_per_km_pc != 0 else float(
                                'inf')

                            # Format percentages
                            r_diff_pc_str = _safe_format(r_diff_pc, ".2f") + "%" if r_diff_pc != float('inf') else "N/A"
                            x_diff_pc_str = _safe_format(x_diff_pc, ".2f") + "%" if x_diff_pc != float('inf') else "N/A"
                            b_diff_pc_str = _safe_format(b_diff_pc, ".2f") + "%" if b_diff_pc != float('inf') else "N/A"

                            # Determine CSS classes for differences
                            r_class = "good-match" if abs(r_diff_pc) <= 20 else (
                                "moderate-match" if abs(r_diff_pc) <= 50 else "poor-match")
                            x_class = "good-match" if abs(x_diff_pc) <= 20 else (
                                "moderate-match" if abs(x_diff_pc) <= 50 else "poor-match")
                            b_class = "good-match" if abs(b_diff_pc) <= 20 else (
                                "moderate-match" if abs(b_diff_pc) <= 50 else "poor-match")

                            network_segments_html += f"""
                            <tr>
                                <td>{net_id}</td>
                                <td>{net_circuits}</td>
                                <td>{_safe_format(pypsa_r_per_km_pc, '.6f')}</td>
                                <td>{_safe_format(net_r_per_km_pc, '.6f')}</td>
                                <td class="{r_class}">{r_diff_pc_str}</td>
                                <td>{_safe_format(pypsa_x_per_km_pc, '.6f')}</td>
                                <td>{_safe_format(net_x_per_km_pc, '.6f')}</td>
                                <td class="{x_class}">{x_diff_pc_str}</td>
                                <td>{_safe_format(pypsa_b_per_km_pc, '.8f')}</td>
                                <td>{_safe_format(net_b_per_km_pc, '.8f')}</td>
                                <td class="{b_class}">{b_diff_pc_str}</td>
                            </tr>
                            """

                    network_segments_html += """
                    </table>
                    """

            # Parameters row with expandable content
            html_summary += f"""
            <tr class="parameter-row" data-result-id="{result_id}" style="display: none;">
                <td colspan="9" class="parameter-details">
                    <div id="details-{result_id}" class="details-section">
                        <h3>PyPSA Line Electrical Parameters</h3>
                        <p>Length: {_safe_format(length_km)} km</p>
                        <p>Bus0: {bus0}</p>
                        <p>Bus1: {bus1}</p>
                        <p>Resistance (R): {_safe_format(r_total, '.6f')} ohm (Total)</p>
                        <p>Reactance (X): {_safe_format(x_total, '.6f')} ohm (Total)</p>
                        <p>Susceptance (B): {_safe_format(b_total, '.8f')} S (Total)</p>
                        <p>Resistance per km (R): {_safe_format(r_per_km, '.6f')} ohm/km</p>
                        <p>Reactance per km (X): {_safe_format(x_per_km, '.6f')} ohm/km</p>
                        <p>Susceptance per km (B): {_safe_format(b_per_km, '.8f')} S/km</p>

                        {network_segments_html}
                    </div>
                </td>
            </tr>
            """

        html_summary += """
                </table>
            </div> <!-- end pypsa-tab -->

            <div id="comparison-tab" class="tab-content">
                <h2>Parameter Comparison</h2>
                <p>This section compares the electrical parameters between JAO, Network, and PyPSA lines.</p>

                <h3>220 kV Lines</h3>
                <table class="segment-table">
                    <tr>
                        <th>Source</th>
                        <th>Count</th>
                        <th>Avg R per km (ohm/km)</th>
                        <th>Avg X per km (ohm/km)</th>
                        <th>Avg B per km (S/km)</th>
                    </tr>
        """

        # Calculate average parameters for 220 kV lines
        jao_220_r = [result.get('jao_r_per_km', None) for result in matching_results
                     if result.get('v_nom') == 220 and result.get('jao_r_per_km') is not None]
        jao_220_x = [result.get('jao_x_per_km', None) for result in matching_results
                     if result.get('v_nom') == 220 and result.get('jao_x_per_km') is not None]
        jao_220_b = [result.get('jao_b_per_km', None) for result in matching_results
                     if result.get('v_nom') == 220 and result.get('jao_b_per_km') is not None]

        network_220_r = [_to_float_or_none(row.get('r_per_km')) for _, row in network_gdf.iterrows()
                         if row.get('v_nom') == 220 and 'r_per_km' in row and pd.notna(row['r_per_km'])]
        network_220_x = [_to_float_or_none(row.get('x_per_km')) for _, row in network_gdf.iterrows()
                         if row.get('v_nom') == 220 and 'x_per_km' in row and pd.notna(row['x_per_km'])]
        network_220_b = [_to_float_or_none(row.get('b_per_km')) for _, row in network_gdf.iterrows()
                         if row.get('v_nom') == 220 and 'b_per_km' in row and pd.notna(row['b_per_km'])]

        pypsa_220_r = []
        pypsa_220_x = []
        pypsa_220_b = []

        for _, row in pypsa_gdf.iterrows():
            if row.get('voltage') == 220 and 'length' in row and pd.notna(row['length']):
                length_m = row['length']
                length_km = _convert_to_km(length_m)

                if length_km is not None and length_km > 0:
                    if 'r_ohm' in row and pd.notna(row['r_ohm']):
                        pypsa_220_r.append(float(row['r_ohm']) / length_km)
                    elif 'r' in row and pd.notna(row['r']):
                        pypsa_220_r.append(float(row['r']) / length_km)

                    if 'x_ohm' in row and pd.notna(row['x_ohm']):
                        pypsa_220_x.append(float(row['x_ohm']) / length_km)
                    elif 'x' in row and pd.notna(row['x']):
                        pypsa_220_x.append(float(row['x']) / length_km)

                    if 'b_siemens' in row and pd.notna(row['b_siemens']):
                        pypsa_220_b.append(float(row['b_siemens']) / length_km)
                    elif 'b' in row and pd.notna(row['b']):
                        pypsa_220_b.append(float(row['b']) / length_km)

        # Calculate averages
        jao_220_r_avg = sum(jao_220_r) / len(jao_220_r) if jao_220_r else '-'
        jao_220_x_avg = sum(jao_220_x) / len(jao_220_x) if jao_220_x else '-'
        jao_220_b_avg = sum(jao_220_b) / len(jao_220_b) if jao_220_b else '-'

        network_220_r_avg = sum(network_220_r) / len(network_220_r) if network_220_r else '-'
        network_220_x_avg = sum(network_220_x) / len(network_220_x) if network_220_x else '-'
        network_220_b_avg = sum(network_220_b) / len(network_220_b) if network_220_b else '-'

        pypsa_220_r_avg = sum(pypsa_220_r) / len(pypsa_220_r) if pypsa_220_r else '-'
        pypsa_220_x_avg = sum(pypsa_220_x) / len(pypsa_220_x) if pypsa_220_x else '-'
        pypsa_220_b_avg = sum(pypsa_220_b) / len(pypsa_220_b) if pypsa_220_b else '-'

        # Add rows
        html_summary += f"""
                    <tr>
                        <td>JAO</td>
                        <td>{len(jao_220_r)}</td>
                        <td>{jao_220_r_avg if jao_220_r_avg == '-' else _safe_format(jao_220_r_avg, '.6f')}</td>
                        <td>{jao_220_x_avg if jao_220_x_avg == '-' else _safe_format(jao_220_x_avg, '.6f')}</td>
                        <td>{jao_220_b_avg if jao_220_b_avg == '-' else _safe_format(jao_220_b_avg, '.8f')}</td>
                    </tr>
                    <tr>
                        <td>Network</td>
                        <td>{len(network_220_r)}</td>
                        <td>{network_220_r_avg if network_220_r_avg == '-' else _safe_format(network_220_r_avg, '.6f')}</td>
                        <td>{network_220_x_avg if network_220_x_avg == '-' else _safe_format(network_220_x_avg, '.6f')}</td>
                        <td>{network_220_b_avg if network_220_b_avg == '-' else _safe_format(network_220_b_avg, '.8f')}</td>
                    </tr>
                    <tr>
                        <td>PyPSA</td>
                        <td>{len(pypsa_220_r)}</td>
                        <td>{pypsa_220_r_avg if pypsa_220_r_avg == '-' else _safe_format(pypsa_220_r_avg, '.6f')}</td>
                        <td>{pypsa_220_x_avg if pypsa_220_x_avg == '-' else _safe_format(pypsa_220_x_avg, '.6f')}</td>
                        <td>{pypsa_220_b_avg if pypsa_220_b_avg == '-' else _safe_format(pypsa_220_b_avg, '.8f')}</td>
                    </tr>
        """

        # Add 400 kV comparison
        html_summary += """
                </table>

                <h3>400 kV Lines</h3>
                <table class="segment-table">
                    <tr>
                        <th>Source</th>
                        <th>Count</th>
                        <th>Avg R per km (ohm/km)</th>
                        <th>Avg X per km (ohm/km)</th>
                        <th>Avg B per km (S/km)</th>
                    </tr>
        """

        # Calculate average parameters for 400 kV lines
        jao_400_r = [result.get('jao_r_per_km', None) for result in matching_results
                     if result.get('v_nom') in [380, 400] and result.get('jao_r_per_km') is not None]
        jao_400_x = [result.get('jao_x_per_km', None) for result in matching_results
                     if result.get('v_nom') in [380, 400] and result.get('jao_x_per_km') is not None]
        jao_400_b = [result.get('jao_b_per_km', None) for result in matching_results
                     if result.get('v_nom') in [380, 400] and result.get('jao_b_per_km') is not None]

        network_400_r = [_to_float_or_none(row.get('r_per_km')) for _, row in network_gdf.iterrows()
                         if row.get('v_nom') in [380, 400] and 'r_per_km' in row and pd.notna(row['r_per_km'])]
        network_400_x = [_to_float_or_none(row.get('x_per_km')) for _, row in network_gdf.iterrows()
                         if row.get('v_nom') in [380, 400] and 'x_per_km' in row and pd.notna(row['x_per_km'])]
        network_400_b = [_to_float_or_none(row.get('b_per_km')) for _, row in network_gdf.iterrows()
                         if row.get('v_nom') in [380, 400] and 'b_per_km' in row and pd.notna(row['b_per_km'])]

        pypsa_400_r = []
        pypsa_400_x = []
        pypsa_400_b = []

        for _, row in pypsa_gdf.iterrows():
            if row.get('voltage') in [380, 400] and 'length' in row and pd.notna(row['length']):
                length_m = row['length']
                length_km = _convert_to_km(length_m)

                if length_km is not None and length_km > 0:
                    if 'r_ohm' in row and pd.notna(row['r_ohm']):
                        pypsa_400_r.append(float(row['r_ohm']) / length_km)
                    elif 'r' in row and pd.notna(row['r']):
                        pypsa_400_r.append(float(row['r']) / length_km)

                    if 'x_ohm' in row and pd.notna(row['x_ohm']):
                        pypsa_400_x.append(float(row['x_ohm']) / length_km)
                    elif 'x' in row and pd.notna(row['x']):
                        pypsa_400_x.append(float(row['x']) / length_km)

                    if 'b_siemens' in row and pd.notna(row['b_siemens']):
                        pypsa_400_b.append(float(row['b_siemens']) / length_km)
                    elif 'b' in row and pd.notna(row['b']):
                        pypsa_400_b.append(float(row['b']) / length_km)

        # Calculate averages
        jao_400_r_avg = sum(jao_400_r) / len(jao_400_r) if jao_400_r else '-'
        jao_400_x_avg = sum(jao_400_x) / len(jao_400_x) if jao_400_x else '-'
        jao_400_b_avg = sum(jao_400_b) / len(jao_400_b) if jao_400_b else '-'

        network_400_r_avg = sum(network_400_r) / len(network_400_r) if network_400_r else '-'
        network_400_x_avg = sum(network_400_x) / len(network_400_x) if network_400_x else '-'
        network_400_b_avg = sum(network_400_b) / len(network_400_b) if network_400_b else '-'

        pypsa_400_r_avg = sum(pypsa_400_r) / len(pypsa_400_r) if pypsa_400_r else '-'
        pypsa_400_x_avg = sum(pypsa_400_x) / len(pypsa_400_x) if pypsa_400_x else '-'
        pypsa_400_b_avg = sum(pypsa_400_b) / len(pypsa_400_b) if pypsa_400_b else '-'

        # Add rows
        html_summary += f"""
                    <tr>
                        <td>JAO</td>
                        <td>{len(jao_400_r)}</td>
                        <td>{jao_400_r_avg if jao_400_r_avg == '-' else _safe_format(jao_400_r_avg, '.6f')}</td>
                        <td>{jao_400_x_avg if jao_400_x_avg == '-' else _safe_format(jao_400_x_avg, '.6f')}</td>
                        <td>{jao_400_b_avg if jao_400_b_avg == '-' else _safe_format(jao_400_b_avg, '.8f')}</td>
                    </tr>
                    <tr>
                        <td>Network</td>
                        <td>{len(network_400_r)}</td>
                        <td>{network_400_r_avg if network_400_r_avg == '-' else _safe_format(network_400_r_avg, '.6f')}</td>
                        <td>{network_400_x_avg if network_400_x_avg == '-' else _safe_format(network_400_x_avg, '.6f')}</td>
                        <td>{network_400_b_avg if network_400_b_avg == '-' else _safe_format(network_400_b_avg, '.8f')}</td>
                    </tr>
                    <tr>
                        <td>{pypsa_400_r_avg if pypsa_400_r_avg == '-' else _safe_format(pypsa_400_r_avg, '.6f')}</td>
                        <td>{pypsa_400_x_avg if pypsa_400_x_avg == '-' else _safe_format(pypsa_400_x_avg, '.6f')}</td>
                        <td>{pypsa_400_b_avg if pypsa_400_b_avg == '-' else _safe_format(pypsa_400_b_avg, '.8f')}</td>
                    </tr>
                </table>
            </div> <!-- end comparison-tab -->
        </div> <!-- end tab-container -->
        """

    html_summary += """
    </body>
    </html>
    """

    # Save the HTML file
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / 'jao_network_matching_parameters.html'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_summary)

    print(f"HTML summary written to {output_file}")
    return output_file



def visualize_results(jao_gdf, network_gdf, matching_results, pypsa_gdf=None):
    """
    Create a visualization of the matching results with optional PyPSA data.
    Organized by:
    1. Voltage (220kV or 400kV)
    2. Match status (matched/unmatched)
    3. Match type (regular, geometric, parallel circuit, etc.)
    """
    import json
    from pathlib import Path
    import shapely

    # Define helper function to extract coordinates from geometries
    def get_geometry_coordinates(geom):
        """Extract coordinates from a geometry, handling both simple and multi-part geometries."""
        if geom is None:
            return []

        if isinstance(geom, (shapely.geometry.LineString, shapely.geometry.Point)):
            return list(geom.coords)
        elif isinstance(geom, (shapely.geometry.MultiLineString, shapely.geometry.MultiPoint,
                               shapely.geometry.GeometryCollection)):
            # For multi-part geometries, concatenate all coordinates
            coords = []
            for part in geom.geoms:
                coords.extend(list(part.coords))
            return coords
        else:
            # For other geometry types (like polygons), extract exterior coordinates
            try:
                return list(geom.exterior.coords)
            except AttributeError:
                # Fallback for geometry types we didn't anticipate
                return []

    # Create GeoJSON data for lines
    jao_features = []
    network_features = []
    pypsa_features = []  # New list for PyPSA features

    # Create sets to track which lines are matched
    matched_jao_ids = set()
    geometric_match_jao_ids = set()
    parallel_circuit_jao_ids = set()
    parallel_voltage_jao_ids = set()
    duplicate_jao_ids = set()

    # Track network lines by match type
    regular_matched_network_ids = set()
    geometric_matched_network_ids = set()
    parallel_circuit_network_ids = set()
    parallel_voltage_network_ids = set()
    duplicate_network_ids = set()

    # Create a set to track matched PyPSA IDs and their match quality
    matched_pypsa_info = {}  # Format: {pypsa_id: {'jao_ids': [...], 'quality': '...', 'match_type': '...'}}

    # First, identify all matched JAO and network lines by type
    for result in matching_results:
        if result['matched'] and result.get('network_ids'):
            jao_id = str(result['jao_id'])
            network_ids = result.get('network_ids', [])

            # If the result contains pypsa_ids, record the match information
            if result.get('pypsa_ids'):
                for pypsa_id in result.get('pypsa_ids', []):
                    pypsa_id = str(pypsa_id)
                    if pypsa_id not in matched_pypsa_info:
                        matched_pypsa_info[pypsa_id] = {
                            'jao_ids': [jao_id],
                            'quality': result.get('match_quality', 'Good'),
                            'match_type': 'regular'
                        }
                    else:
                        matched_pypsa_info[pypsa_id]['jao_ids'].append(jao_id)

                    # Update match type if needed
                    if result.get('is_geometric_match', False):
                        matched_pypsa_info[pypsa_id]['match_type'] = 'geometric'
                    elif result.get('is_parallel_circuit', False):
                        matched_pypsa_info[pypsa_id]['match_type'] = 'parallel'
                    elif result.get('is_parallel_voltage_circuit', False):
                        matched_pypsa_info[pypsa_id]['match_type'] = 'parallel_voltage'
                    elif result.get('is_duplicate', False):
                        matched_pypsa_info[pypsa_id]['match_type'] = 'duplicate'

            if result.get('is_duplicate', False):
                duplicate_jao_ids.add(jao_id)
                for network_id in network_ids:
                    duplicate_network_ids.add(str(network_id))
            elif result.get('is_geometric_match', False):
                geometric_match_jao_ids.add(jao_id)
                for network_id in network_ids:
                    geometric_matched_network_ids.add(str(network_id))
            elif result.get('is_parallel_circuit', False):
                parallel_circuit_jao_ids.add(jao_id)
                for network_id in network_ids:
                    parallel_circuit_network_ids.add(str(network_id))
            elif result.get('is_parallel_voltage_circuit', False):
                parallel_voltage_jao_ids.add(jao_id)
                for network_id in network_ids:
                    parallel_voltage_network_ids.add(str(network_id))
            else:
                matched_jao_ids.add(jao_id)
                for network_id in network_ids:
                    regular_matched_network_ids.add(str(network_id))

    # Define a safe helper function for voltage conversion
    def _safe_int(value):
        try:
            return int(value)
        except (ValueError, TypeError):
            return 0

    # Define a safe helper function for float conversion
    def _safe_float(value):
        try:
            return float(value)
        except (ValueError, TypeError):
            return 0.0

    # Add JAO lines to GeoJSON with status indicators
    for idx, row in jao_gdf.iterrows():
        # Create a unique ID for each line
        line_id = f"jao_{row['id']}"

        # Get coordinates using the helper function
        coords = get_geometry_coordinates(row.geometry)
        geometry_type = "LineString"  # Treat all as LineString for GeoJSON

        voltage = _safe_int(row.get('v_nom', 0))
        voltage_class = "220kV" if voltage == 220 else "400kV"

        # Check JAO line match status
        jao_id = str(row['id'])

        # Determine match status and styling
        if jao_id in duplicate_jao_ids:
            status = "duplicate"
            tooltip_status = "Duplicate"
        elif jao_id in parallel_circuit_jao_ids:
            status = "parallel"
            tooltip_status = "Parallel Circuit"
        elif jao_id in parallel_voltage_jao_ids:
            status = "parallel_voltage"
            tooltip_status = "Parallel Voltage"
        elif jao_id in geometric_match_jao_ids:
            status = "geometric"
            tooltip_status = "Geometric Match"
        elif jao_id in matched_jao_ids:
            status = "matched"
            tooltip_status = "Regular Match"
        else:
            status = "unmatched"
            tooltip_status = "Unmatched"

        # Create GeoJSON feature
        feature = {
            "type": "Feature",
            "id": line_id,
            "properties": {
                "type": "jao",
                "id": jao_id,
                "name": str(row.get('NE_name', '')),
                "voltage": voltage,
                "voltageClass": voltage_class,
                "status": status,
                "matchStatus": "matched" if status != "unmatched" else "unmatched",
                "tooltip": f"JAO: {jao_id} - {row.get('NE_name', '')} ({voltage} kV) - {tooltip_status}"
            },
            "geometry": {
                "type": geometry_type,
                "coordinates": [[float(x), float(y)] for x, y in coords]
            }
        }
        jao_features.append(feature)

    # Add network lines to GeoJSON with status indicators
    for idx, row in network_gdf.iterrows():
        line_id = f"network_{row['id']}"

        # Get coordinates using the helper function
        coords = get_geometry_coordinates(row.geometry)
        geometry_type = "LineString"  # Treat all as LineString for GeoJSON

        network_id = str(row['id'])
        voltage = _safe_int(row.get('v_nom', 0))
        voltage_class = "220kV" if voltage == 220 else "400kV"

        # Determine match status for this network line
        if network_id in duplicate_network_ids:
            status = "duplicate"
            tooltip_status = "Duplicate"
        elif network_id in parallel_circuit_network_ids:
            status = "parallel"
            tooltip_status = "Parallel Circuit"
        elif network_id in parallel_voltage_network_ids:
            status = "parallel_voltage"
            tooltip_status = "Parallel Voltage"
        elif network_id in geometric_matched_network_ids:
            status = "geometric"
            tooltip_status = "Geometric Match"
        elif network_id in regular_matched_network_ids:
            status = "matched"
            tooltip_status = "Regular Match"
        else:
            status = "unmatched"
            tooltip_status = "Unmatched"

        feature = {
            "type": "Feature",
            "id": line_id,
            "properties": {
                "type": "network",
                "id": network_id,
                "voltage": voltage,
                "voltageClass": voltage_class,
                "status": status,
                "matchStatus": "matched" if status != "unmatched" else "unmatched",
                "tooltip": f"Network: {row['id']} ({voltage} kV) - {tooltip_status}"
            },
            "geometry": {
                "type": geometry_type,
                "coordinates": [[float(x), float(y)] for x, y in coords]
            }
        }
        network_features.append(feature)

    # Add PyPSA lines if provided, with enhanced matched/unmatched information
    if pypsa_gdf is not None and not pypsa_gdf.empty:
        for idx, row in pypsa_gdf.iterrows():
            pypsa_id = str(row.get('id', idx))
            line_id = f"pypsa_{pypsa_id}"

            # Get coordinates using the helper function
            coords = get_geometry_coordinates(row.geometry)
            geometry_type = "LineString"  # Treat all as LineString for GeoJSON

            voltage = _safe_int(row.get('voltage', 0))
            voltage_class = "220kV" if 200 <= voltage < 300 else "400kV" if voltage >= 300 else "other"

            # Determine if this PyPSA line is matched to any JAO line
            is_matched = pypsa_id in matched_pypsa_info
            match_status = "matched" if is_matched else "unmatched"
            tooltip_status = "Matched" if is_matched else "Unmatched"

            # Get match details if available
            match_type = matched_pypsa_info.get(pypsa_id, {}).get('match_type', '')
            match_quality = matched_pypsa_info.get(pypsa_id, {}).get('quality', 'N/A')
            matched_jao_ids_list = matched_pypsa_info.get(pypsa_id, {}).get('jao_ids', [])

            # Generate additional status information for tooltip
            status_detail = ""
            if is_matched:
                if match_type == "geometric":
                    status_detail = " (Geometric Match)"
                elif match_type == "parallel":
                    status_detail = " (Parallel Circuit)"
                elif match_type == "parallel_voltage":
                    status_detail = " (Parallel Voltage)"
                elif match_type == "duplicate":
                    status_detail = " (Duplicate)"
                else:
                    status_detail = " (Regular Match)"

            # Create a meaningful tooltip with available information
            tooltip_text = f"PyPSA: {pypsa_id} ({voltage} kV) - {tooltip_status}{status_detail}"
            if 'bus0' in row and 'bus1' in row:
                tooltip_text += f" - {row['bus0']} to {row['bus1']}"

            # If matched, add the JAO IDs
            if is_matched and matched_jao_ids_list:
                tooltip_text += f" - Matched to JAO ID(s): {', '.join(matched_jao_ids_list)}"

            # Calculate line length in km from coordinates if not available
            length_km = row.get('length_km', 0)
            if length_km == 0 and coords:
                from math import sqrt
                length_m = 0
                for i in range(1, len(coords)):
                    dx = coords[i][0] - coords[i - 1][0]
                    dy = coords[i][1] - coords[i - 1][1]
                    length_m += sqrt(dx * dx + dy * dy)
                length_km = length_m / 1000  # Very rough approximation

            feature = {
                "type": "Feature",
                "id": line_id,
                "properties": {
                    "type": "pypsa",
                    "id": pypsa_id,
                    "voltage": voltage,
                    "voltageClass": voltage_class,
                    "matchStatus": match_status,
                    "status": match_type if is_matched and match_type else match_status,
                    "matchQuality": match_quality,
                    "matchedJaoIds": matched_jao_ids_list,
                    "tooltip": tooltip_text,
                    "length_km": length_km,
                    "bus0": str(row.get('bus0', '')),
                    "bus1": str(row.get('bus1', '')),
                    # Add electrical parameters
                    "r_ohm": _safe_float(row.get('r_ohm', 0)),
                    "x_ohm": _safe_float(row.get('x_ohm', 0)),
                    "b_siemens": _safe_float(row.get('b_siemens', 0)),
                    "g_siemens": _safe_float(row.get('g_siemens', 0)),
                    "r_per_km": _safe_float(row.get('r_per_km', 0)),
                    "x_per_km": _safe_float(row.get('x_per_km', 0)),
                    "b_per_km": _safe_float(row.get('b_per_km', 0)),
                    "g_per_km": _safe_float(row.get('g_per_km', 0))
                },
                "geometry": {
                    "type": geometry_type,
                    "coordinates": [[float(x), float(y)] for x, y in coords]
                }
            }
            pypsa_features.append(feature)

    # Create GeoJSON collections
    jao_collection = {"type": "FeatureCollection", "features": jao_features}
    network_collection = {"type": "FeatureCollection", "features": network_features}
    pypsa_collection = {"type": "FeatureCollection", "features": pypsa_features} if pypsa_features else None

    # Convert to JSON strings
    jao_json = json.dumps(jao_collection)
    network_json = json.dumps(network_collection)
    pypsa_json = json.dumps(pypsa_collection) if pypsa_collection else "null"

    # Make sure output_dir exists
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    # CSS for the organized legend and enhanced PyPSA display
    css = """
    html, body {
        height: 100%;
        margin: 0;
        padding: 0;
        font-family: Arial, sans-serif;
    }
    #map {
        height: 100%;
        width: 100%;
    }
    .sidebar {
        position: absolute;
        top: 10px;
        right: 10px;
        width: 280px;
        background: white;
        border-radius: 5px;
        box-shadow: 0 0 10px rgba(0, 0, 0, 0.5);
        max-height: calc(100% - 20px);
        overflow-y: auto;
        z-index: 1000;
    }
    .sidebar-header {
        padding: 10px;
        background: #f8f8f8;
        border-bottom: 1px solid #ddd;
        font-weight: bold;
        border-radius: 5px 5px 0 0;
    }
    .sidebar-content {
        padding: 10px;
    }
    .search-box {
        width: 100%;
        padding: 8px;
        box-sizing: border-box;
        margin-bottom: 10px;
        border: 1px solid #ddd;
        border-radius: 3px;
    }
    .search-results {
        max-height: 200px;
        overflow-y: auto;
        margin-bottom: 10px;
        display: none;
        background: white;
        border: 1px solid #ddd;
        border-radius: 3px;
        z-index: 1010;
        position: absolute;
        width: 260px;
    }
    .search-result {
        padding: 8px;
        cursor: pointer;
        border-bottom: 1px solid #eee;
    }
    .search-result:hover {
        background-color: #f5f5f5;
    }
    .section-header {
        font-weight: bold;
        margin: 15px 0 5px 0;
        padding-bottom: 3px;
        border-bottom: 1px solid #eee;
    }
    .filter-group {
        margin-bottom: 15px;
    }
    .filter-option {
        padding: 5px;
        margin: 2px 0;
        cursor: pointer;
        display: flex;
        align-items: center;
        background: #f0f0f0;
        border: 1px solid #ccc;
        border-radius: 3px;
    }
    .filter-option.active {
        background-color: #4CAF50;
        color: white;
    }
    .filter-option:hover {
        background-color: #f5f5f5;
    }
    .legend-color {
        display: inline-block;
        width: 15px;
        height: 3px;
        margin-right: 8px;
    }
    .voltage-group {
        margin-bottom: 15px;
        border: 1px solid #eee;
        border-radius: 3px;
        padding: 8px;
    }
    .voltage-header {
        font-weight: bold;
        margin-bottom: 5px;
        color: #444;
        cursor: pointer;
    }
    .voltage-header::before {
        content: "▼ ";
        font-size: 10px;
    }
    .voltage-header.collapsed::before {
        content: "► ";
    }
    .voltage-content {
        display: block;
    }
    .voltage-content.collapsed {
        display: none;
    }
    .status-group {
        margin-left: 10px;
        margin-bottom: 10px;
    }
    .status-header {
        font-weight: bold;
        margin-bottom: 3px;
        color: #666;
        cursor: pointer;
    }
    .status-header::before {
        content: "▼ ";
        font-size: 10px;
    }
    .status-header.collapsed::before {
        content: "► ";
    }
    .status-content {
        display: block;
    }
    .status-content.collapsed {
        display: none;
    }
    .type-group {
        margin-left: 20px;
    }
    .highlighted {
        stroke-width: 6px !important;
        stroke-opacity: 1 !important;
        animation: pulse 1.5s infinite;
    }
    @keyframes pulse {
        0% { stroke-opacity: 1; }
        50% { stroke-opacity: 0.5; }
        100% { stroke-opacity: 1; }
    }
    .stats-box {
        margin-top: 15px;
        background-color: #f9f9f9;
        border: 1px solid #ddd;
        border-radius: 4px;
        padding: 10px;
        font-size: 12px;
    }
    .stats-item {
        margin-bottom: 5px;
    }
    .leaflet-control-polylinemeasure {
        background-color: white !important;
        padding: 4px !important;
        border-radius: 4px !important;
    }
    .toggle-all-btn {
        margin: 5px 0;
        padding: 5px 10px;
        background-color: #f0f0f0;
        border: 1px solid #ccc;
        border-radius: 3px;
        cursor: pointer;
        font-size: 12px;
        text-align: center;
    }
    .toggle-all-btn:hover {
        background-color: #e0e0e0;
    }

    /* Tab styling for results display */
    .tab-container {
        width: 100%;
        margin-top: 20px;
    }
    .tab-nav {
        display: flex;
        background-color: #f8f8f8;
        border-bottom: 1px solid #ddd;
    }
    .tab-btn {
        padding: 10px 15px;
        cursor: pointer;
        background: #f0f0f0;
        border: 1px solid #ddd;
        border-bottom: none;
        margin-right: 2px;
        border-top-left-radius: 3px;
        border-top-right-radius: 3px;
    }
    .tab-btn.active {
        background: white;
        border-bottom: 1px solid white;
        margin-bottom: -1px;
    }
    .tab-content {
        display: none;
        padding: 15px;
        border: 1px solid #ddd;
        border-top: none;
    }
    .tab-content.active {
        display: block;
    }

    /* Table styling */
    .results-table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 15px;
    }
    .results-table th, .results-table td {
        padding: 8px;
        text-align: left;
        border-bottom: 1px solid #ddd;
    }
    .results-table th {
        background-color: #f2f2f2;
        position: sticky;
        top: 0;
    }
    .results-table tr:hover {
        background-color: #f5f5f5;
        cursor: pointer;
    }

    /* Filter panel for tables */
    .filter-panel {
        margin-bottom: 15px;
        padding: 10px;
        background-color: #f9f9f9;
        border: 1px solid #ddd;
        border-radius: 4px;
    }
    .filter-section {
        margin-bottom: 10px;
    }
    .filter-section h3 {
        margin: 5px 0;
        font-size: 14px;
    }
    .button-group {
        display: flex;
        flex-wrap: wrap;
        gap: 5px;
    }
    .filter-btn {
        padding: 5px 10px;
        background-color: #f0f0f0;
        border: 1px solid #ccc;
        border-radius: 3px;
        cursor: pointer;
        font-size: 12px;
    }
    .filter-btn.active {
        background-color: #4CAF50;
        color: white;
    }

    /* Modal for electrical parameters */
    .modal {
        display: none;
        position: fixed;
        z-index: 1050;
        left: 0;
        top: 0;
        width: 100%;
        height: 100%;
        overflow: auto;
        background-color: rgba(0,0,0,0.4);
    }
    .modal-content {
        background-color: #fefefe;
        margin: 10% auto;
        padding: 20px;
        border: 1px solid #888;
        width: 80%;
        max-width: 600px;
        border-radius: 5px;
    }
    .close-btn {
        color: #aaa;
        float: right;
        font-size: 28px;
        font-weight: bold;
        cursor: pointer;
    }
    .close-btn:hover {
        color: black;
    }
    .param-table {
        width: 100%;
        border-collapse: collapse;
    }
    .param-table th, .param-table td {
        padding: 8px;
        text-align: left;
        border-bottom: 1px solid #ddd;
    }
    .param-table th {
        background-color: #f2f2f2;
    }
    .show-params-btn {
        padding: 5px 10px;
        background-color: #4CAF50;
        color: white;
        border: none;
        border-radius: 3px;
        cursor: pointer;
    }
    """

    # Prepare the PyPSA section HTML with enhanced filtering and matching
    pypsa_section_html = ""
    if pypsa_gdf is not None and not pypsa_gdf.empty:
        # Calculate how many PyPSA lines are matched and unmatched
        matched_count = len([f for f in pypsa_features if f['properties']['matchStatus'] == 'matched'])
        unmatched_count = len(pypsa_features) - matched_count

        # Calculate counts by voltage
        pypsa_220kv_count = len([f for f in pypsa_features if 200 <= f['properties']['voltage'] < 300])
        pypsa_400kv_count = len([f for f in pypsa_features if f['properties']['voltage'] >= 300])

        # Calculate matched counts by voltage
        pypsa_220kv_matched = len([f for f in pypsa_features if
                                   200 <= f['properties']['voltage'] < 300 and f['properties'][
                                       'matchStatus'] == 'matched'])
        pypsa_400kv_matched = len([f for f in pypsa_features if
                                   f['properties']['voltage'] >= 300 and f['properties']['matchStatus'] == 'matched'])

        pypsa_section_html = f"""
        <!-- PyPSA Section -->
        <div class="voltage-group" data-voltage="pypsa">
            <div class="voltage-header" data-voltage="pypsa">PyPSA Lines</div>
            <div class="voltage-content">
                <!-- 220kV PyPSA Section -->
                <div class="status-group">
                    <div class="status-header" data-type="pypsa" data-voltage="220" data-status="all">220 kV ({pypsa_220kv_count} total, {pypsa_220kv_matched} matched)</div>
                    <div class="status-content">
                        <div class="type-group">
                            <div class="filter-option active" data-type="pypsa" data-voltage="220" data-status="matched" data-subtype="matched">
                                <div class="legend-color" style="background-color:#008000;"></div>
                                <span>Matched PyPSA 220kV</span>
                            </div>
                            <div class="filter-option active" data-type="pypsa" data-voltage="220" data-status="unmatched" data-subtype="unmatched">
                                <div class="legend-color" style="background-color:#ffa500;"></div>
                                <span>Unmatched PyPSA 220kV</span>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- 400kV PyPSA Section -->
                <div class="status-group">
                    <div class="status-header" data-type="pypsa" data-voltage="400" data-status="all">400 kV ({pypsa_400kv_count} total, {pypsa_400kv_matched} matched)</div>
                    <div class="status-content">
                        <div class="type-group">
                            <div class="filter-option active" data-type="pypsa" data-voltage="400" data-status="matched" data-subtype="matched">
                                <div class="legend-color" style="background-color:#4b0082;"></div>
                                <span>Matched PyPSA 400kV</span>
                            </div>
                            <div class="filter-option active" data-type="pypsa" data-voltage="400" data-status="unmatched" data-subtype="unmatched">
                                <div class="legend-color" style="background-color:#7d1d88;"></div>
                                <span>Unmatched PyPSA 400kV</span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        """

    # Create the tabs structure for JAO, Network, and PyPSA data tables
    tabs_html = """
    <div class="section-header">Detailed Results</div>
    <div class="tab-container">
        <div class="tab-nav">
            <div class="tab-btn active" data-tab="jaoTab">JAO Lines</div>
            <div class="tab-btn" data-tab="networkTab">Network Lines</div>
            <div class="tab-btn" data-tab="pypsaTab">PyPSA Lines</div>
            <div class="tab-btn" data-tab="comparisonTab">Parameter Comparison</div>
        </div>

        <!-- JAO Tab Content -->
        <div id="jaoTab" class="tab-content active">
            <div class="filter-panel">
                <input type="text" id="jaoSearch" placeholder="Search for JAO lines..." class="search-box">

                <div class="filter-section">
                    <h3>By Match Status:</h3>
                    <div class="button-group">
                        <button class="filter-btn active" data-filter="all">All</button>
                        <button class="filter-btn" data-filter="matched">Matched</button>
                        <button class="filter-btn" data-filter="geometric">Geometric</button>
                        <button class="filter-btn" data-filter="parallel">Parallel Circuit</button>
                        <button class="filter-btn" data-filter="parallel_voltage">Parallel Voltage</button>
                        <button class="filter-btn" data-filter="duplicate">Duplicate</button>
                        <button class="filter-btn" data-filter="unmatched">Unmatched</button>
                    </div>
                </div>

                <div class="filter-section">
                    <h3>By Voltage Level:</h3>
                    <div class="button-group">
                        <button class="filter-btn active" data-filter="all">All</button>
                        <button class="filter-btn" data-filter="220">220 kV</button>
                        <button class="filter-btn" data-filter="400">400 kV</button>
                    </div>
                </div>
            </div>

            <table id="jaoTable" class="results-table">
                <thead>
                    <tr>
                        <th>JAO ID</th>
                        <th>Name</th>
                        <th>Voltage (kV)</th>
                        <th>Length (km)</th>
                        <th>Match Status</th>
                        <th>Network IDs</th>
                    </tr>
                </thead>
                <tbody>
                    <!-- Will be populated dynamically -->
                </tbody>
            </table>
        </div>

        <!-- Network Tab Content -->
        <div id="networkTab" class="tab-content">
            <div class="filter-panel">
                <input type="text" id="networkSearch" placeholder="Search for network lines..." class="search-box">

                <div class="filter-section">
                    <h3>By Match Status:</h3>
                    <div class="button-group">
                        <button class="filter-btn active" data-filter="all">All</button>
                        <button class="filter-btn" data-filter="matched">Matched</button>
                        <button class="filter-btn" data-filter="geometric">Geometric</button>
                        <button class="filter-btn" data-filter="parallel">Parallel Circuit</button>
                        <button class="filter-btn" data-filter="parallel_voltage">Parallel Voltage</button>
                        <button class="filter-btn" data-filter="duplicate">Duplicate</button>
                        <button class="filter-btn" data-filter="unmatched">Unmatched</button>
                    </div>
                </div>

                <div class="filter-section">
                    <h3>By Voltage Level:</h3>
                    <div class="button-group">
                        <button class="filter-btn active" data-filter="all">All</button>
                        <button class="filter-btn" data-filter="220">220 kV</button>
                        <button class="filter-btn" data-filter="400">400 kV</button>
                    </div>
                </div>
            </div>

            <table id="networkTable" class="results-table">
                <thead>
                    <tr>
                        <th>Network ID</th>
                        <th>Voltage (kV)</th>
                        <th>Length (km)</th>
                        <th>Match Status</th>
                        <th>JAO IDs</th>
                    </tr>
                </thead>
                <tbody>
                    <!-- Will be populated dynamically -->
                </tbody>
            </table>
        </div>

        <!-- PyPSA Tab Content -->
        <div id="pypsaTab" class="tab-content">
            <div class="filter-panel">
                <input type="text" id="pypsaSearch" placeholder="Search for PyPSA lines..." class="search-box">

                <div class="filter-section">
                    <h3>By Match Status:</h3>
                    <div class="button-group">
                        <button class="filter-btn active" data-filter="all">All</button>
                        <button class="filter-btn" data-filter="matched">Matched</button>
                        <button class="filter-btn" data-filter="unmatched">Unmatched</button>
                    </div>
                </div>

                <div class="filter-section">
                    <h3>By Voltage Level:</h3>
                    <div class="button-group">
                        <button class="filter-btn active" data-filter="all">All</button>
                        <button class="filter-btn" data-filter="220">220 kV</button>
                        <button class="filter-btn" data-filter="400">400 kV</button>
                    </div>
                </div>
            </div>

            <table id="pypsaTable" class="results-table">
                <thead>
                    <tr>
                        <th>PyPSA ID</th>
                        <th>Bus 0</th>
                        <th>Bus 1</th>
                        <th>Voltage (kV)</th>
                        <th>Length (km)</th>
                        <th>Match Status</th>
                        <th>Matched Network IDs</th>
                        <th>Electrical Parameters</th>
                    </tr>
                </thead>
                <tbody>
                    <!-- Will be populated dynamically -->
                </tbody>
            </table>
        </div>

        <!-- Parameter Comparison Tab Content -->
        <div id="comparisonTab" class="tab-content">
            <div class="filter-panel">
                <input type="text" id="comparisonSearch" placeholder="Search for comparisons..." class="search-box">

                <div class="filter-section">
                    <h3>Select Comparison Type:</h3>
                    <div class="button-group">
                        <button class="filter-btn active" data-filter="jao-network">JAO-Network</button>
                        <button class="filter-btn" data-filter="jao-pypsa">JAO-PyPSA</button>
                        <button class="filter-btn" data-filter="network-pypsa">Network-PyPSA</button>
                    </div>
                </div>
            </div>

            <table id="comparisonTable" class="results-table">
                <thead>
                    <tr>
                        <th>JAO ID</th>
                        <th>Network ID</th>
                        <th>PyPSA ID</th>
                        <th>Voltage (kV)</th>
                        <th>Length Ratio</th>
                        <th>R (ohm) Ratio</th>
                        <th>X (ohm) Ratio</th>
                        <th>B (S) Ratio</th>
                        <th>Details</th>
                    </tr>
                </thead>
                <tbody>
                    <!-- Will be populated dynamically -->
                </tbody>
            </table>
        </div>
    </div>

    <!-- Modal for electrical parameters -->
    <div id="paramModal" class="modal">
        <div class="modal-content">
            <span class="close-btn">&times;</span>
            <h2 id="paramModalTitle">Electrical Parameters</h2>
            <div id="paramModalContent"></div>
        </div>
    </div>
    """

    # Create a complete HTML file from scratch
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>JAO-Network-PyPSA Line Matching Results</title>

        <link rel="stylesheet" href="https://unpkg.com/leaflet@1.7.1/dist/leaflet.css" />
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css">
        <link rel="stylesheet" href="https://ppete2.github.io/Leaflet.PolylineMeasure/Leaflet.PolylineMeasure.css" />

        <script src="https://unpkg.com/leaflet@1.7.1/dist/leaflet.js"></script>
        <script src="https://ppete2.github.io/Leaflet.PolylineMeasure/Leaflet.PolylineMeasure.js"></script>

        <style>
            {css}
        </style>
    </head>
    <body>
        <div id="map"></div>

        <div class="sidebar" id="controlPanel">
            <div class="sidebar-header">
                <span><i class="fas fa-search"></i> Search & Filter</span>
            </div>
            <div class="sidebar-content">
                <input type="text" id="searchInput" class="search-box" placeholder="Search for lines...">
                <div id="searchResults" class="search-results"></div>

                <!-- By Voltage Sections -->
                <div class="section-header">By Voltage</div>

                <div class="toggle-all-btn" id="toggleAllVoltages">Toggle All Voltages</div>

                <!-- 220kV Voltage Section -->
                <div class="voltage-group" data-voltage="220">
                    <div class="voltage-header" data-voltage="220">220 kV</div>
                    <div class="voltage-content">
                        <!-- 220kV JAO Matched Section -->
                        <div class="status-group">
                            <div class="status-header" data-type="jao" data-voltage="220" data-status="matched">JAO Matched</div>
                            <div class="status-content">
                                <div class="type-group">
                                    <div class="filter-option active" data-type="jao" data-voltage="220" data-status="matched" data-subtype="matched">
                                        <div class="legend-color" style="background-color:green;"></div>
                                        <span>Regular Match</span>
                                    </div>
                                    <div class="filter-option active" data-type="jao" data-voltage="220" data-status="matched" data-subtype="geometric">
                                        <div class="legend-color" style="background-color:#00BFFF;"></div>
                                        <span>Geometric Match</span>
                                    </div>
                                    <div class="filter-option active" data-type="jao" data-voltage="220" data-status="matched" data-subtype="parallel">
                                        <div class="legend-color" style="background-color:#9932CC;"></div>
                                        <span>Parallel Circuit</span>
                                    </div>
                                    <div class="filter-option active" data-type="jao" data-voltage="220" data-status="matched" data-subtype="parallel_voltage">
                                        <div class="legend-color" style="background-color:#FF8C00;"></div>
                                        <span>Parallel Voltage</span>
                                    </div>
                                    <div class="filter-option active" data-type="jao" data-voltage="220" data-status="matched" data-subtype="duplicate">
                                        <div class="legend-color" style="background-color:#DA70D6;"></div>
                                        <span>Duplicate</span>
                                    </div>
                                </div>
                            </div>
                        </div>

                        <!-- 220kV JAO Unmatched Section -->
                        <div class="status-group">
                            <div class="status-header" data-type="jao" data-voltage="220" data-status="unmatched">JAO Unmatched</div>
                            <div class="status-content">
                                <div class="type-group">
                                    <div class="filter-option active" data-type="jao" data-voltage="220" data-status="unmatched" data-subtype="unmatched">
                                        <div class="legend-color" style="background-color:red;"></div>
                                        <span>Unmatched JAO</span>
                                    </div>
                                </div>
                            </div>
                        </div>

                        <!-- 220kV Network Matched Section -->
                        <div class="status-group">
                            <div class="status-header" data-type="network" data-voltage="220" data-status="matched">Network Matched</div>
                            <div class="status-content">
                                <div class="type-group">
                                    <div class="filter-option active" data-type="network" data-voltage="220" data-status="matched" data-subtype="matched">
                                        <div class="legend-color" style="background-color:purple;"></div>
                                        <span>Regular Match</span>
                                    </div>
                                    <div class="filter-option active" data-type="network" data-voltage="220" data-status="matched" data-subtype="geometric">
                                        <div class="legend-color" style="background-color:#1E90FF;"></div>
                                        <span>Geometric Match</span>
                                    </div>
                                    <div class="filter-option active" data-type="network" data-voltage="220" data-status="matched" data-subtype="parallel">
                                        <div class="legend-color" style="background-color:#8B008B;"></div>
                                        <span>Parallel Circuit</span>
                                    </div>
                                    <div class="filter-option active" data-type="network" data-voltage="220" data-status="matched" data-subtype="parallel_voltage">
                                        <div class="legend-color" style="background-color:#FF4500;"></div>
                                        <span>Parallel Voltage</span>
                                    </div>
                                    <div class="filter-option active" data-type="network" data-voltage="220" data-status="matched" data-subtype="duplicate">
                                        <div class="legend-color" style="background-color:#FF00FF;"></div>
                                        <span>Duplicate</span>
                                    </div>
                                </div>
                            </div>
                        </div>

                        <!-- 220kV Network Unmatched Section -->
                        <div class="status-group">
                            <div class="status-header" data-type="network" data-voltage="220" data-status="unmatched">Network Unmatched</div>
                            <div class="status-content">
                                <div class="type-group">
                                    <div class="filter-option active" data-type="network" data-voltage="220" data-status="unmatched" data-subtype="unmatched">
                                        <div class="legend-color" style="background-color:blue;"></div>
                                        <span>Unmatched Network</span>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- 400kV Voltage Section -->
                <div class="voltage-group" data-voltage="400">
                    <div class="voltage-header" data-voltage="400">400 kV</div>
                    <div class="voltage-content">
                        <!-- 400kV JAO Matched Section -->
                        <div class="status-group">
                            <div class="status-header" data-type="jao" data-voltage="400" data-status="matched">JAO Matched</div>
                            <div class="status-content">
                                <div class="type-group">
                                    <div class="filter-option active" data-type="jao" data-voltage="400" data-status="matched" data-subtype="matched">
                                        <div class="legend-color" style="background-color:green;"></div>
                                        <span>Regular Match</span>
                                    </div>
                                    <div class="filter-option active" data-type="jao" data-voltage="400" data-status="matched" data-subtype="geometric">
                                        <div class="legend-color" style="background-color:#00BFFF;"></div>
                                        <span>Geometric Match</span>
                                    </div>
                                    <div class="filter-option active" data-type="jao" data-voltage="400" data-status="matched" data-subtype="parallel">
                                        <div class="legend-color" style="background-color:#9932CC;"></div>
                                        <span>Parallel Circuit</span>
                                    </div>
                                    <div class="filter-option active" data-type="jao" data-voltage="400" data-status="matched" data-subtype="parallel_voltage">
                                        <div class="legend-color" style="background-color:#FF8C00;"></div>
                                        <span>Parallel Voltage</span>
                                    </div>
                                    <div class="filter-option active" data-type="jao" data-voltage="400" data-status="matched" data-subtype="duplicate">
                                        <div class="legend-color" style="background-color:#DA70D6;"></div>
                                        <span>Duplicate</span>
                                    </div>
                                </div>
                            </div>
                        </div>

                        <!-- 400kV JAO Unmatched Section -->
                        <div class="status-group">
                            <div class="status-header" data-type="jao" data-voltage="400" data-status="unmatched">JAO Unmatched</div>
                            <div class="status-content">
                                <div class="type-group">
                                    <div class="filter-option active" data-type="jao" data-voltage="400" data-status="unmatched" data-subtype="unmatched">
                                        <div class="legend-color" style="background-color:red;"></div>
                                        <span>Unmatched JAO</span>
                                    </div>
                                </div>
                            </div>
                        </div>

                        <!-- 400kV Network Matched Section -->
                        <div class="status-group">
                            <div class="status-header" data-type="network" data-voltage="400" data-status="matched">Network Matched</div>
                            <div class="status-content">
                                <div class="type-group">
                                    <div class="filter-option active" data-type="network" data-voltage="400" data-status="matched" data-subtype="matched">
                                        <div class="legend-color" style="background-color:purple;"></div>
                                        <span>Regular Match</span>
                                    </div>
                                    <div class="filter-option active" data-type="network" data-voltage="400" data-status="matched" data-subtype="geometric">
                                        <div class="legend-color" style="background-color:#1E90FF;"></div>
                                        <span>Geometric Match</span>
                                    </div>
                                    <div class="filter-option active" data-type="network" data-voltage="400" data-status="matched" data-subtype="parallel">
                                        <div class="legend-color" style="background-color:#8B008B;"></div>
                                        <span>Parallel Circuit</span>
                                    </div>
                                    <div class="filter-option active" data-type="network" data-voltage="400" data-status="matched" data-subtype="parallel_voltage">
                                        <div class="legend-color" style="background-color:#FF4500;"></div>
                                        <span>Parallel Voltage</span>
                                    </div>
                                    <div class="filter-option active" data-type="network" data-voltage="400" data-status="matched" data-subtype="duplicate">
                                        <div class="legend-color" style="background-color:#FF00FF;"></div>
                                        <span>Duplicate</span>
                                    </div>
                                </div>
                            </div>
                        </div>

                        <!-- 400kV Network Unmatched Section -->
                        <div class="status-group">
                            <div class="status-header" data-type="network" data-voltage="400" data-status="unmatched">Network Unmatched</div>
                            <div class="status-content">
                                <div class="type-group">
                                    <div class="filter-option active" data-type="network" data-voltage="400" data-status="unmatched" data-subtype="unmatched">
                                        <div class="legend-color" style="background-color:blue;"></div>
                                        <span>Unmatched Network</span>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                {pypsa_section_html}

                <div class="section-header">Tools</div>
                <div class="type-group">
                    <p>Click the ruler icon on the map to measure distances.</p>
                </div>

                <div class="stats-box">
                    <h3><i class="fas fa-chart-pie"></i> Statistics</h3>
                    <div class="stats-item">JAO Lines: {len(jao_gdf)} total</div>
                    <div class="stats-item">- Matched: {len(matched_jao_ids)} ({len(matched_jao_ids) / len(jao_gdf) * 100:.1f}%)</div>
                    <div class="stats-item">- Geometric: {len(geometric_match_jao_ids)} ({len(geometric_match_jao_ids) / len(jao_gdf) * 100:.1f}%)</div>
                    <div class="stats-item">- Parallel Circuit: {len(parallel_circuit_jao_ids)} ({len(parallel_circuit_jao_ids) / len(jao_gdf) * 100:.1f}%)</div>
                    <div class="stats-item">- Parallel Voltage: {len(parallel_voltage_jao_ids)} ({len(parallel_voltage_jao_ids) / len(jao_gdf) * 100:.1f}%)</div>
                    <div class="stats-item">- Duplicate: {len(duplicate_jao_ids)} ({len(duplicate_jao_ids) / len(jao_gdf) * 100:.1f}%)</div>
                    <div class="stats-item">- Unmatched: {len(jao_gdf) - len(matched_jao_ids) - len(geometric_match_jao_ids) - len(parallel_circuit_jao_ids) - len(parallel_voltage_jao_ids) - len(duplicate_jao_ids)} ({(len(jao_gdf) - len(matched_jao_ids) - len(geometric_match_jao_ids) - len(parallel_circuit_jao_ids) - len(parallel_voltage_jao_ids) - len(duplicate_jao_ids)) / len(jao_gdf) * 100:.1f}%)</div>
                    <hr>
                    <div class="stats-item">Network Lines: {len(network_gdf)} total</div>
                    <div class="stats-item">- Matched: {len(regular_matched_network_ids)} ({len(regular_matched_network_ids) / len(network_gdf) * 100:.1f}%)</div>
                    <div class="stats-item">- Unmatched: {len(network_gdf) - len(regular_matched_network_ids) - len(geometric_matched_network_ids) - len(parallel_circuit_network_ids) - len(parallel_voltage_network_ids) - len(duplicate_network_ids)} ({(len(network_gdf) - len(regular_matched_network_ids) - len(geometric_matched_network_ids) - len(parallel_circuit_network_ids) - len(parallel_voltage_network_ids) - len(duplicate_network_ids)) / len(network_gdf) * 100:.1f}%)</div>
                    {"<hr>" if pypsa_gdf is not None else ""}
                    {"<div class='stats-item'>PyPSA Lines: " + str(len(pypsa_gdf)) + " total</div>" if pypsa_gdf is not None else ""}
                    {"<div class='stats-item'>- Matched: " + str(matched_count) + " (" + f"{matched_count / len(pypsa_gdf) * 100:.1f}" + "%)</div>" if pypsa_gdf is not None else ""}
                    {"<div class='stats-item'>- Unmatched: " + str(unmatched_count) + " (" + f"{unmatched_count / len(pypsa_gdf) * 100:.1f}" + "%)</div>" if pypsa_gdf is not None else ""}
                    {"<div class='stats-item'>- 220kV: " + str(pypsa_220kv_count) + " (" + str(pypsa_220kv_matched) + " matched)</div>" if pypsa_gdf is not None else ""}
                    {"<div class='stats-item'>- 400kV: " + str(pypsa_400kv_count) + " (" + str(pypsa_400kv_matched) + " matched)</div>" if pypsa_gdf is not None else ""}
                </div>

                {tabs_html}
            </div>
        </div>

        <script>
            // Initialize the map
            var map = L.map('map').setView([51.1657, 10.4515], 6);

            // Add base tile layer
            L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
                attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
            }}).addTo(map);

            // Add measurement tool
            L.control.polylineMeasure({{
                position: 'topleft',
                unit: 'metres',
                showBearings: true,
                clearMeasurementsOnStop: false,
                showClearControl: true,
                showUnitControl: true
            }}).addTo(map);

            // Load the GeoJSON data
            var jaoLines = {jao_json};
            var networkLines = {network_json};
            var pypsaLines = {pypsa_json};

            // Define styling for the JAO lines
            function jaoStyle(feature) {{
                switch(feature.properties.status) {{
                    case "matched":
                        return {{
                            "color": "green",
                            "weight": 3,
                            "opacity": 0.8
                        }};
                    case "geometric":
                        return {{
                            "color": "#00BFFF", // Deep Sky Blue
                            "weight": 3,
                            "opacity": 0.8,
                            "dashArray": "10, 5"
                        }};
                    case "parallel":
                        return {{
                            "color": "#9932CC", // Dark Orchid
                            "weight": 3,
                            "opacity": 0.8,
                            "dashArray": "5, 5"
                        }};
                    case "parallel_voltage":
                        return {{
                            "color": "#FF8C00", // Dark Orange
                            "weight": 3,
                            "opacity": 0.8,
                            "dashArray": "10, 2, 2, 2"
                        }};
                    case "duplicate":
                        return {{
                            "color": "#DA70D6", // Orchid
                            "weight": 3,
                            "opacity": 0.8,
                            "dashArray": "2, 5"
                        }};
                    default: // unmatched
                        return {{
                            "color": "red",
                            "weight": 3,
                            "opacity": 0.8
                        }};
                }}
            }};

            // Define styling for network lines
            function networkStyle(feature) {{
                switch(feature.properties.status) {{
                    case "matched":
                        return {{
                            "color": "purple",
                            "weight": 2,
                            "opacity": 0.6
                        }};
                    case "geometric":
                        return {{
                            "color": "#1E90FF", // Dodger Blue
                            "weight": 2,
                            "opacity": 0.6,
                            "dashArray": "10, 5"
                        }};
                    case "parallel":
                        return {{
                            "color": "#8B008B", // Dark Magenta
                            "weight": 2,
                            "opacity": 0.6,
                            "dashArray": "5, 5"
                        }};
                    case "parallel_voltage":
                        return {{
                            "color": "#FF4500", // Orange Red
                            "weight": 2,
                            "opacity": 0.6,
                            "dashArray": "10, 2, 2, 2"
                        }};
                    case "duplicate":
                        return {{
                            "color": "#FF00FF", // Magenta
                            "weight": 2,
                            "opacity": 0.6,
                            "dashArray": "2, 5"
                        }};
                    default: // unmatched
                        return {{
                            "color": "blue",
                            "weight": 2,
                            "opacity": 0.6
                        }};
                }}
            }};

            // Define styling for PyPSA lines with improved matched/unmatched distinction
            function pypsaStyle(feature) {{
                let voltage = feature.properties.voltage;
                let isMatched = feature.properties.matchStatus === "matched";

                if (voltage >= 300) {{
                    return {{
                        "color": isMatched ? "#4b0082" : "#7d1d88",  // Different purples for matched/unmatched 400kV
                        "weight": 3,
                        "opacity": 0.8,
                        "dashArray": isMatched ? null : "5, 5"  // Dashed line for unmatched
                    }};
                }} else if (voltage >= 200) {{
                    return {{
                        "color": isMatched ? "#008000" : "#ffa500",  // Green for matched, orange for unmatched 220kV
                        "weight": 2.5,
                        "opacity": 0.7,
                        "dashArray": isMatched ? null : "5, 5"  // Dashed line for unmatched
                    }};
                }} else {{
                    return {{
                        "color": "#32a852", // Green for others
                        "weight": 2,
                        "opacity": 0.6
                    }};
                }}
            }};

            // Organize features by voltage, status and type
            var layers = {{
                jao: {{
                    "220kV": {{
                        matched: {{
                            matched: L.geoJSON(null, {{style: jaoStyle}}).addTo(map),
                            geometric: L.geoJSON(null, {{style: jaoStyle}}).addTo(map),
                            parallel: L.geoJSON(null, {{style: jaoStyle}}).addTo(map),
                            parallel_voltage: L.geoJSON(null, {{style: jaoStyle}}).addTo(map),
                            duplicate: L.geoJSON(null, {{style: jaoStyle}}).addTo(map)
                        }},
                        unmatched: {{
                            unmatched: L.geoJSON(null, {{style: jaoStyle}}).addTo(map)
                        }}
                    }},
                    "400kV": {{
                        matched: {{
                            matched: L.geoJSON(null, {{style: jaoStyle}}).addTo(map),
                            geometric: L.geoJSON(null, {{style: jaoStyle}}).addTo(map),
                            parallel: L.geoJSON(null, {{style: jaoStyle}}).addTo(map),
                            parallel_voltage: L.geoJSON(null, {{style: jaoStyle}}).addTo(map),
                            duplicate: L.geoJSON(null, {{style: jaoStyle}}).addTo(map)
                        }},
                        unmatched: {{
                            unmatched: L.geoJSON(null, {{style: jaoStyle}}).addTo(map)
                        }}
                    }}
                }},
                network: {{
                    "220kV": {{
                        matched: {{
                            matched: L.geoJSON(null, {{style: networkStyle}}).addTo(map),
                            geometric: L.geoJSON(null, {{style: networkStyle}}).addTo(map),
                            parallel: L.geoJSON(null, {{style: networkStyle}}).addTo(map),
                            parallel_voltage: L.geoJSON(null, {{style: networkStyle}}).addTo(map),
                            duplicate: L.geoJSON(null, {{style: networkStyle}}).addTo(map)
                        }},
                        unmatched: {{
                            unmatched: L.geoJSON(null, {{style: networkStyle}}).addTo(map)
                        }}
                    }},
                    "400kV": {{
                        matched: {{
                            matched: L.geoJSON(null, {{style: networkStyle}}).addTo(map),
                            geometric: L.geoJSON(null, {{style: networkStyle}}).addTo(map),
                            parallel: L.geoJSON(null, {{style: networkStyle}}).addTo(map),
                            parallel_voltage: L.geoJSON(null, {{style: networkStyle}}).addTo(map),
                            duplicate: L.geoJSON(null, {{style: networkStyle}}).addTo(map)
                        }},
                        unmatched: {{
                            unmatched: L.geoJSON(null, {{style: networkStyle}}).addTo(map)
                        }}
                    }}
                }},
                pypsa: pypsaLines ? {{
                    "220kV": {{
                        matched: {{
                            matched: L.geoJSON(null, {{style: pypsaStyle}}).addTo(map)
                        }},
                        unmatched: {{
                            unmatched: L.geoJSON(null, {{style: pypsaStyle}}).addTo(map)
                        }}
                    }},
                    "400kV": {{
                        matched: {{
                            matched: L.geoJSON(null, {{style: pypsaStyle}}).addTo(map)
                        }},
                        unmatched: {{
                            unmatched: L.geoJSON(null, {{style: pypsaStyle}}).addTo(map)
                        }}
                    }}
                }} : null
            }};

            // Add all JAO features to their respective layers
            jaoLines.features.forEach(function(feature) {{
                var props = feature.properties;
                var voltageClass = props.voltageClass;
                var matchStatus = props.matchStatus;
                var status = props.status;

                // Add feature to the appropriate layer
                if (layers.jao[voltageClass] && 
                    layers.jao[voltageClass][matchStatus] && 
                    layers.jao[voltageClass][matchStatus][status]) {{

                    layers.jao[voltageClass][matchStatus][status].addData(feature);

                    // Add tooltip and click handler
                    layers.jao[voltageClass][matchStatus][status].eachLayer(function(layer) {{
                        if (layer.feature && layer.feature.id === feature.id) {{
                            layer.bindTooltip(props.tooltip);
                            layer.on('click', function() {{
                                highlightFeature(feature.id);
                            }});
                        }}
                    }});
                }}
            }});

            // Add all Network features to their respective layers
            networkLines.features.forEach(function(feature) {{
                var props = feature.properties;
                var voltageClass = props.voltageClass;
                var matchStatus = props.matchStatus;
                var status = props.status;

                // Add feature to the appropriate layer
                if (layers.network[voltageClass] && 
                    layers.network[voltageClass][matchStatus] && 
                    layers.network[voltageClass][matchStatus][status]) {{

                    layers.network[voltageClass][matchStatus][status].addData(feature);

                    // Add tooltip and click handler
                    layers.network[voltageClass][matchStatus][status].eachLayer(function(layer) {{
                        if (layer.feature && layer.feature.id === feature.id) {{
                            layer.bindTooltip(props.tooltip);
                            layer.on('click', function() {{
                                highlightFeature(feature.id);
                            }});
                        }}
                    }});
                }}
            }});

            // Add PyPSA features with improved matched/unmatched handling
            if (pypsaLines && layers.pypsa) {{
                pypsaLines.features.forEach(function(feature) {{
                    var props = feature.properties;
                    var voltage = props.voltage;
                    var voltageClass = voltage >= 300 ? "400kV" : "220kV";
                    var matchStatus = props.matchStatus;

                    // Add to appropriate layer based on voltage and match status
                    if (layers.pypsa[voltageClass] && layers.pypsa[voltageClass][matchStatus]) {{
                        layers.pypsa[voltageClass][matchStatus][matchStatus].addData(feature);

                        // Add tooltip and click handler
                        layers.pypsa[voltageClass][matchStatus][matchStatus].eachLayer(function(layer) {{
                            if (layer.feature && layer.feature.id === feature.id) {{
                                layer.bindTooltip(props.tooltip);
                                layer.on('click', function() {{
                                    highlightFeature(feature.id);
                                    // Also highlight any matched JAO lines
                                    if (props.matchedJaoIds && props.matchedJaoIds.length > 0) {{
                                        setTimeout(function() {{
                                            highlightFeature("jao_" + props.matchedJaoIds[0]);
                                        }}, 1000);
                                    }}
                                }});
                            }}
                        }});
                    }}
                }});
            }}

            // Function to highlight a feature
            function highlightFeature(id) {{
                // Clear existing highlights
                clearHighlights();

                // Helper function to check and highlight in a set of layers
                function checkAndHighlight(layerSet) {{
                    if (!layerSet) return;

                    Object.keys(layerSet).forEach(function(voltage) {{
                        Object.keys(layerSet[voltage]).forEach(function(status) {{
                            Object.keys(layerSet[voltage][status]).forEach(function(subtype) {{
                                layerSet[voltage][status][subtype].eachLayer(function(layer) {{
                                    if (layer.feature && layer.feature.id === id) {{
                                        layer.setStyle({{
                                            weight: 6,
                                            opacity: 1,
                                            className: 'highlighted'
                                        }});

                                        if (layer._path) {{
                                            layer._path.classList.add('highlighted');
                                        }}

                                        // Center map on the highlighted feature
                                        var bounds = layer.getBounds();
                                        map.fitBounds(bounds, {{ padding: [50, 50] }});
                                    }}
                                }});
                            }});
                        }});
                    }});
                }}

                // Check JAO, Network, and PyPSA layers
                checkAndHighlight(layers.jao);
                checkAndHighlight(layers.network);
                checkAndHighlight(layers.pypsa);
            }}

            // Function to clear highlights
            function clearHighlights() {{
                // Helper function to reset styles in a set of layers
                function resetStyles(layerSet, styleFunction) {{
                    if (!layerSet) return;

                    Object.keys(layerSet).forEach(function(voltage) {{
                        Object.keys(layerSet[voltage]).forEach(function(status) {{
                            Object.keys(layerSet[voltage][status]).forEach(function(subtype) {{
                                layerSet[voltage][status][subtype].eachLayer(function(layer) {{
                                    layer.setStyle(styleFunction(layer.feature));
                                    if (layer._path) {{
                                        layer._path.classList.remove('highlighted');
                                    }}
                                }});
                            }});
                        }});
                    }});
                }}

                // Reset styles for JAO, Network, and PyPSA layers
                resetStyles(layers.jao, jaoStyle);
                resetStyles(layers.network, networkStyle);
                if (layers.pypsa) resetStyles(layers.pypsa, pypsaStyle);
            }}

            // Create search index
            function createSearchIndex() {{
                var searchData = [];

                // Add JAO features to search data
                jaoLines.features.forEach(function(feature) {{
                    searchData.push({{
                        id: feature.id,
                        text: feature.properties.tooltip,
                        feature: feature
                    }});
                }});

                // Add Network features to search data
                networkLines.features.forEach(function(feature) {{
                    searchData.push({{
                        id: feature.id,
                        text: feature.properties.tooltip,
                        feature: feature
                    }});
                }});

                // Add PyPSA features to search data if available
                if (pypsaLines) {{
                    pypsaLines.features.forEach(function(feature) {{
                        searchData.push({{
                            id: feature.id,
                            text: feature.properties.tooltip,
                            feature: feature
                        }});
                    }});
                }}

                // Add relationship data (JAO to Network mappings)
                {json.dumps([{
        'jao_id': r['jao_id'],
        'network_ids': r.get('network_ids', []),
        'type': 'duplicate' if r.get('is_duplicate', False) else
        'geometric' if r.get('is_geometric_match', False) else
        'parallel' if r.get('is_parallel_circuit', False) else
        'parallel_voltage' if r.get('is_parallel_voltage_circuit', False) else 'regular'
    } for r in matching_results if r['matched'] and r.get('network_ids')])}.forEach(function(match) {{
                    // Add an entry for searching JAO to find networks
                    searchData.push({{
                        id: "ref_jao_" + match.jao_id,
                        text: "Network lines for JAO " + match.jao_id + " (" + match.type + " match): " + match.network_ids.join(", "),
                        type: "reference",
                        jaoId: match.jao_id,
                        networkIds: match.network_ids
                    }});

                    // Add entries for searching networks to find JAO
                    match.network_ids.forEach(function(netId) {{
                        searchData.push({{
                            id: "ref_net_" + netId,
                            text: "Network " + netId + " is matched to JAO " + match.jao_id + " (" + match.type + " match)",
                            type: "reference",
                            jaoId: match.jao_id,
                            networkIds: [netId]
                        }});
                    }});
                }});

                return searchData;
            }}

            // Filter functionality
            function applyFilters() {{
                // Process each filter option
                document.querySelectorAll('.filter-option').forEach(function(el) {{
                    var type = el.getAttribute('data-type');
                    var voltage = el.getAttribute('data-voltage');
                    var status = el.getAttribute('data-status');
                    var subtype = el.getAttribute('data-subtype');
                    var isActive = el.classList.contains('active');

                    // Convert to layer keys if needed
                    var voltageClass = voltage ? (voltage === "220" ? "220kV" : "400kV") : null;

                    // If we have all parameters, show/hide the layer
                    if (type && status && subtype && voltageClass) {{
                        if (layers[type] && 
                            layers[type][voltageClass] && 
                            layers[type][voltageClass][status] && 
                            layers[type][voltageClass][status][subtype]) {{

                            var layer = layers[type][voltageClass][status][subtype];

                            if (isActive) {{
                                layer.addTo(map);
                            }} else {{
                                map.removeLayer(layer);
                            }}
                        }}
                    }}
                }});
            }}

            // Initialize search functionality
            function initializeSearch() {{
                var searchInput = document.getElementById('searchInput');
                var searchResults = document.getElementById('searchResults');
                var searchData = createSearchIndex();

                searchInput.addEventListener('input', function() {{
                    var query = this.value.toLowerCase();

                    if (query.length < 2) {{
                        searchResults.innerHTML = '';
                        searchResults.style.display = 'none';
                        return;
                    }}

                    var results = searchData.filter(function(item) {{
                        return item.text.toLowerCase().includes(query);
                    }});

                    searchResults.innerHTML = '';

                    results.slice(0, 10).forEach(function(result) {{
                        var div = document.createElement('div');
                        div.className = 'search-result';
                        div.textContent = result.text;
                        div.onclick = function() {{
                            if (result.type === "reference") {{
                                // For reference results, highlight the related features
                                if (result.jaoId) {{
                                    highlightFeature("jao_" + result.jaoId);
                                }}

                                if (result.networkIds && result.networkIds.length > 0) {{
                                    // Highlight the first network with a small delay
                                    setTimeout(function() {{
                                        highlightFeature("network_" + result.networkIds[0]);
                                    }}, 1000);
                                }}
                            }} else {{
                                // For direct features, just highlight it
                                highlightFeature(result.id);
                            }}

                            searchInput.value = result.text;
                            searchResults.style.display = 'none';
                        }};
                        searchResults.appendChild(div);
                    }});

                    searchResults.style.display = results.length > 0 ? 'block' : 'none';
                }});

                // Hide search results when clicking outside
                document.addEventListener('click', function(e) {{
                    if (!searchInput.contains(e.target) && !searchResults.contains(e.target)) {{
                        searchResults.style.display = 'none';
                    }}
                }});
            }}

            // Setup collapsible sections
            function setupCollapsible() {{
                // Setup voltage headers
                document.querySelectorAll('.voltage-header').forEach(function(header) {{
                    header.addEventListener('click', function() {{
                        // Toggle this voltage section
                        var voltage = this.getAttribute('data-voltage');
                        var content = this.nextElementSibling;

                        this.classList.toggle('collapsed');
                        content.classList.toggle('collapsed');

                        // Update filters based on visibility
                        applyFilters();
                    }});
                }});

                // Setup status headers
                document.querySelectorAll('.status-header').forEach(function(header) {{
                    header.addEventListener('click', function() {{
                        // Toggle this status section
                        var content = this.nextElementSibling;

                        this.classList.toggle('collapsed');
                        content.classList.toggle('collapsed');

                        // Update filters based on visibility
                        applyFilters();
                    }});
                }});

                // Toggle all voltages button
                document.getElementById('toggleAllVoltages').addEventListener('click', function() {{
                    var allActive = document.querySelectorAll('.filter-option[data-type]').length === 
                                   document.querySelectorAll('.filter-option[data-type].active').length;

                    // Toggle all filter options
                    document.querySelectorAll('.filter-option[data-type]').forEach(function(el) {{
                        if (allActive) {{
                            el.classList.remove('active');
                        }} else {{
                            el.classList.add('active');
                        }}
                    }});

                    applyFilters();
                }});
            }}

            // Setup filter options
            function setupFilters() {{
                document.querySelectorAll('.filter-option').forEach(function(option) {{
                    option.addEventListener('click', function(e) {{
                        // Toggle this option
                        this.classList.toggle('active');

                        // Apply filters immediately
                        applyFilters();

                        // Stop propagation to prevent parent handlers
                        e.stopPropagation();
                    }});
                }});
            }}

            // Setup tabs
            function setupTabs() {{
                // Switch tabs
                document.querySelectorAll('.tab-btn').forEach(function(btn) {{
                    btn.addEventListener('click', function() {{
                        var tabId = this.getAttribute('data-tab');

                        // Hide all tabs and remove active class
                        document.querySelectorAll('.tab-content').forEach(function(tab) {{
                            tab.classList.remove('active');
                        }});
                        document.querySelectorAll('.tab-btn').forEach(function(btn) {{
                            btn.classList.remove('active');
                        }});

                        // Show selected tab and add active class
                        document.getElementById(tabId).classList.add('active');
                        this.classList.add('active');

                        // Initialize table data if needed
                        if (tabId === 'jaoTab') updateJaoTable({{}});
                        else if (tabId === 'networkTab') updateNetworkTable({{}});
                        else if (tabId === 'pypsaTab') updatePypsaTable({{}});
                        else if (tabId === 'comparisonTab') updateComparisonTable({{}});
                    }});
                }});

                // Setup filter buttons in each tab
                setupTableFilters('jao');
                setupTableFilters('network');
                setupTableFilters('pypsa');
                setupTableFilters('comparison');

                // Initialize the first tab (JAO)
                updateJaoTable({{}});
            }}

            // Setup table filters
            function setupTableFilters(tableType) {{
                var searchInput = document.getElementById(tableType + 'Search');
                if (!searchInput) return;

                searchInput.addEventListener('input', function() {{
                    var filters = getTableFilters(tableType);
                    filters.search = this.value.toLowerCase();

                    if (tableType === 'jao') updateJaoTable(filters);
                    else if (tableType === 'network') updateNetworkTable(filters);
                    else if (tableType === 'pypsa') updatePypsaTable(filters);
                    else if (tableType === 'comparison') updateComparisonTable(filters);
                }});

                // Setup filter buttons
                document.querySelectorAll('#' + tableType + 'Tab .filter-btn').forEach(function(btn) {{
                    btn.addEventListener('click', function() {{
                        // Get the filter group
                        var filterGroup = this.parentElement;

                        // Toggle active class for this button
                        filterGroup.querySelectorAll('.filter-btn').forEach(function(b) {{
                            b.classList.remove('active');
                        }});
                        this.classList.add('active');

                        // Apply filters
                        var filters = getTableFilters(tableType);

                        if (tableType === 'jao') updateJaoTable(filters);
                        else if (tableType === 'network') updateNetworkTable(filters);
                        else if (tableType === 'pypsa') updatePypsaTable(filters);
                        else if (tableType === 'comparison') updateComparisonTable(filters);
                    }});
                }});
            }}

            // Get current filters for a table
            function getTableFilters(tableType) {{
                var filters = {{
                    search: document.getElementById(tableType + 'Search').value.toLowerCase(),
                    matchStatus: 'all',
                    voltage: 'all',
                    quality: 'all',
                    comparisonType: 'jao-network'
                }};

                // Get match status filter
                var matchStatusBtn = document.querySelector('#' + tableType + 'Tab .filter-section:nth-child(2) .filter-btn.active');
                if (matchStatusBtn) filters.matchStatus = matchStatusBtn.getAttribute('data-filter');

                // Get voltage filter
                var voltageBtn = document.querySelector('#' + tableType + 'Tab .filter-section:nth-child(3) .filter-btn.active');
                if (voltageBtn) filters.voltage = voltageBtn.getAttribute('data-filter');

                // Get quality filter if applicable
                var qualityBtn = document.querySelector('#' + tableType + 'Tab .filter-section:nth-child(4) .filter-btn.active');
                if (qualityBtn) filters.quality = qualityBtn.getAttribute('data-filter');

                // Get comparison type if applicable
                var comparisonBtn = document.querySelector('#comparisonTab .filter-btn.active');
                if (comparisonBtn) filters.comparisonType = comparisonBtn.getAttribute('data-filter');

                return filters;
            }}

            // Update JAO table with filters
            function updateJaoTable(filters) {{
                var table = document.getElementById('jaoTable').getElementsByTagName('tbody')[0];
                table.innerHTML = '';

                jaoLines.features.forEach(function(feature) {{
                    var props = feature.properties;
                    var jaoId = props.id;
                    var voltage = props.voltage;
                    var status = props.status;

                    // Apply filters
                    if (filters.matchStatus !== 'all' && status !== filters.matchStatus) return;
                    if (filters.voltage !== 'all' && voltage.toString() !== filters.voltage) return;
                    if (filters.search && !jaoId.toLowerCase().includes(filters.search) &&
                        !props.name.toLowerCase().includes(filters.search)) return;

                    // Find any matched network IDs
                    var matchedNetworkIds = [];
                    {json.dumps([{
        'jao_id': r['jao_id'],
        'network_ids': r.get('network_ids', [])
    } for r in matching_results if r['matched'] and r.get('network_ids')])}.forEach(function(match) {{
                        if (match.jao_id === jaoId) {{
                            matchedNetworkIds = match.network_ids;
                        }}
                    }});

                    // Create table row
                    var row = table.insertRow();

                    // JAO ID
                    var cell1 = row.insertCell(0);
                    cell1.textContent = jaoId;

                    // Name
                    var cell2 = row.insertCell(1);
                    cell2.textContent = props.name;

                    // Voltage
                    var cell3 = row.insertCell(2);
                    cell3.textContent = voltage;

                    // Length (not directly available, using placeholder)
                    var cell4 = row.insertCell(3);
                    cell4.textContent = "N/A";

                    // Match Status
                    var cell5 = row.insertCell(4);
                    var statusText = status.charAt(0).toUpperCase() + status.slice(1);
                    cell5.textContent = statusText;

                    // Network IDs
                    var cell6 = row.insertCell(5);
                    cell6.textContent = matchedNetworkIds.join(', ') || "None";

                    // Add click handler to highlight the line on the map
                    row.addEventListener('click', function() {{
                        highlightFeature(feature.id);
                    }});
                }});
            }}

            // Update Network table with filters
            function updateNetworkTable(filters) {{
                var table = document.getElementById('networkTable').getElementsByTagName('tbody')[0];
                table.innerHTML = '';

                networkLines.features.forEach(function(feature) {{
                    var props = feature.properties;
                    var networkId = props.id;
                    var voltage = props.voltage;
                    var status = props.status;

                    // Apply filters
                    if (filters.matchStatus !== 'all' && status !== filters.matchStatus) return;
                    if (filters.voltage !== 'all' && voltage.toString() !== filters.voltage) return;
                    if (filters.search && !networkId.toLowerCase().includes(filters.search)) return;

                    // Find any matched JAO IDs
                    var matchedJaoIds = [];
                    {json.dumps([{
        'jao_id': r['jao_id'],
        'network_ids': r.get('network_ids', [])
    } for r in matching_results if r['matched'] and r.get('network_ids')])}.forEach(function(match) {{
                        if (match.network_ids.includes(networkId)) {{
                            matchedJaoIds.push(match.jao_id);
                        }}
                    }});

                    // Create table row
                    var row = table.insertRow();

                    // Network ID
                    var cell1 = row.insertCell(0);
                    cell1.textContent = networkId;

                    // Voltage
                    var cell2 = row.insertCell(1);
                    cell2.textContent = voltage;

                    // Length (not directly available, using placeholder)
                    var cell3 = row.insertCell(2);
                    cell3.textContent = "N/A";

                    // Match Status
                    var cell4 = row.insertCell(3);
                    var statusText = status.charAt(0).toUpperCase() + status.slice(1);
                    cell4.textContent = statusText;

                    // JAO IDs
                    var cell5 = row.insertCell(4);
                    cell5.textContent = matchedJaoIds.join(', ') || "None";

                    // Add click handler to highlight the line on the map
                    row.addEventListener('click', function() {{
                        highlightFeature(feature.id);
                    }});
                }});
            }}

            // Update PyPSA table with filters
            function updatePypsaTable(filters) {{
                var table = document.getElementById('pypsaTable').getElementsByTagName('tbody')[0];
                table.innerHTML = '';

                if (!pypsaLines) return;

                pypsaLines.features.forEach(function(feature) {{
                    var props = feature.properties;
                    var pypsaId = props.id;
                    var voltage = props.voltage;
                    var matchStatus = props.matchStatus;
                    var bus0 = props.bus0 || "N/A";
                    var bus1 = props.bus1 || "N/A";

                    // Apply filters
                    if (filters.matchStatus !== 'all' && matchStatus !== filters.matchStatus) return;
                    if (filters.voltage !== 'all') {{
                        if (filters.voltage === '220' && (voltage < 200 || voltage >= 300)) return;
                        if (filters.voltage === '400' && voltage < 300) return;
                    }}
                    if (filters.search && !pypsaId.toLowerCase().includes(filters.search) &&
                        !bus0.toLowerCase().includes(filters.search) &&
                        !bus1.toLowerCase().includes(filters.search)) return;

                    // Create table row
                    var row = table.insertRow();

                    // PyPSA ID
                    var cell1 = row.insertCell(0);
                    cell1.textContent = pypsaId;

                    // Bus 0
                    var cell2 = row.insertCell(1);
                    cell2.textContent = bus0;

                    // Bus 1
                    var cell3 = row.insertCell(2);
                    cell3.textContent = bus1;

                    // Voltage
                    var cell4 = row.insertCell(3);
                    cell4.textContent = voltage;

                    // Length
                    var cell5 = row.insertCell(4);
                    cell5.textContent = props.length_km ? props.length_km.toFixed(2) : "N/A";

                    // Match Status
                    var cell6 = row.insertCell(5);
                    var statusText = matchStatus.charAt(0).toUpperCase() + matchStatus.slice(1);
                    cell6.textContent = statusText;

                    // Matched JAO IDs
                    var cell7 = row.insertCell(6);
                    cell7.textContent = props.matchedJaoIds && props.matchedJaoIds.length ? props.matchedJaoIds.join(', ') : "None";

                    // Electrical Parameters Button
                    var cell8 = row.insertCell(7);
                    var button = document.createElement('button');
                    button.className = 'show-params-btn';
                    button.textContent = 'Show Parameters';
                    button.onclick = function(e) {{
                        e.stopPropagation();  // Prevent row click from triggering
                        showElectricalParams(props);
                    }};
                    cell8.appendChild(button);

                    // Add click handler to highlight the line on the map
                    row.addEventListener('click', function() {{
                        highlightFeature(feature.id);
                    }});
                }});
            }}

            // Update Comparison table with filters
            function updateComparisonTable(filters) {{
                var table = document.getElementById('comparisonTable').getElementsByTagName('tbody')[0];
                table.innerHTML = '';

                // This is a placeholder - would need actual comparison data
                // For now, just show JAO-Network matches as an example
                if (filters.comparisonType === 'jao-network') {{
                    {json.dumps([{
        'jao_id': r['jao_id'],
        'network_ids': r.get('network_ids', [])
    } for r in matching_results if r['matched'] and r.get('network_ids')])}.forEach(function(match) {{
                        var jaoId = match.jao_id;
                        var networkIds = match.network_ids;

                        // Find JAO and Network properties
                        var jaoFeature = jaoLines.features.find(f => f.properties.id === jaoId);
                        if (!jaoFeature) return;

                        // Create a row for each network ID
                        networkIds.forEach(function(networkId) {{
                            var networkFeature = networkLines.features.find(f => f.properties.id === networkId);
                            if (!networkFeature) return;

                            // Apply search filter
                            if (filters.search && 
                                !jaoId.toLowerCase().includes(filters.search) && 
                                !networkId.toLowerCase().includes(filters.search)) return;

                            // Create table row
                            var row = table.insertRow();

                            // JAO ID
                            var cell1 = row.insertCell(0);
                            cell1.textContent = jaoId;

                            // Network ID
                            var cell2 = row.insertCell(1);
                            cell2.textContent = networkId;

                            // PyPSA ID - N/A for JAO-Network comparison
                            var cell3 = row.insertCell(2);
                            cell3.textContent = "N/A";

                            // Voltage
                            var cell4 = row.insertCell(3);
                            cell4.textContent = jaoFeature.properties.voltage;

                            // Length Ratio - placeholder
                            var cell5 = row.insertCell(4);
                            cell5.textContent = "N/A";

                            // R, X, B Ratios - placeholders
                            var cell6 = row.insertCell(5);
                            cell6.textContent = "N/A";

                            var cell7 = row.insertCell(6);
                            cell7.textContent = "N/A";

                            var cell8 = row.insertCell(7);
                            cell8.textContent = "N/A";

                            // Details button
                            var cell9 = row.insertCell(8);
                            var button = document.createElement('button');
                            button.className = 'show-params-btn';
                            button.textContent = 'Compare';
                            button.onclick = function(e) {{
                                e.stopPropagation();
                                // Would show detailed comparison
                                alert('Comparison details would be shown here');
                            }};
                            cell9.appendChild(button);

                            // Add click handler to highlight both lines
                            row.addEventListener('click', function() {{
                                highlightFeature("jao_" + jaoId);
                                setTimeout(function() {{
                                    highlightFeature("network_" + networkId);
                                }}, 1000);
                            }});
                        }});
                    }});
                }}
                // Other comparison types would be implemented similarly
            }}

            // Show electrical parameters modal
            function showElectricalParams(props) {{
                var modal = document.getElementById('paramModal');
                var title = document.getElementById('paramModalTitle');
                var content = document.getElementById('paramModalContent');

                // Set title
                title.textContent = 'Electrical Parameters for PyPSA Line ' + props.id;

                // Create table of parameters
                var tableHtml = '<table class="param-table">';
                tableHtml += '<tr><th>Parameter</th><th>Value</th><th>Unit</th></tr>';
                tableHtml += '<tr><td>Voltage</td><td>' + props.voltage + '</td><td>kV</td></tr>';
                tableHtml += '<tr><td>Length</td><td>' + (props.length_km ? props.length_km.toFixed(2) : "N/A") + '</td><td>km</td></tr>';
                tableHtml += '<tr><td>Resistance (R)</td><td>' + props.r_ohm.toFixed(6) + '</td><td>Ω</td></tr>';
                tableHtml += '<tr><td>Reactance (X)</td><td>' + props.x_ohm.toFixed(6) + '</td><td>Ω</td></tr>';
                tableHtml += '<tr><td>Susceptance (B)</td><td>' + props.b_siemens.toFixed(6) + '</td><td>S</td></tr>';
                tableHtml += '<tr><td>Conductance (G)</td><td>' + props.g_siemens.toFixed(6) + '</td><td>S</td></tr>';

                // Add per km values if available
                if (props.r_per_km || props.x_per_km || props.b_per_km || props.g_per_km) {{
                    tableHtml += '<tr><td colspan="3"><b>Per km values:</b></td></tr>';
                    tableHtml += '<tr><td>R per km</td><td>' + props.r_per_km.toFixed(6) + '</td><td>Ω/km</td></tr>';
                    tableHtml += '<tr><td>X per km</td><td>' + props.x_per_km.toFixed(6) + '</td><td>Ω/km</td></tr>';
                    tableHtml += '<tr><td>B per km</td><td>' + props.b_per_km.toFixed(6) + '</td><td>S/km</td></tr>';
                    tableHtml += '<tr><td>G per km</td><td>' + props.g_per_km.toFixed(6) + '</td><td>S/km</td></tr>';
                }}

                // Add bus information
                tableHtml += '<tr><td colspan="3"><b>Connection:</b></td></tr>';
                tableHtml += '<tr><td>Bus 0</td><td colspan="2">' + props.bus0 + '</td></tr>';
                tableHtml += '<tr><td>Bus 1</td><td colspan="2">' + props.bus1 + '</td></tr>';

                // Add match information if matched
                if (props.matchStatus === 'matched' && props.matchedJaoIds && props.matchedJaoIds.length) {{
                    tableHtml += '<tr><td colspan="3"><b>Match Information:</b></td></tr>';
                    tableHtml += '<tr><td>Matched JAO IDs</td><td colspan="2">' + props.matchedJaoIds.join(', ') + '</td></tr>';
                    tableHtml += '<tr><td>Match Quality</td><td colspan="2">' + (props.matchQuality || 'N/A') + '</td></tr>';
                }}

                tableHtml += '</table>';

                content.innerHTML = tableHtml;

                // Show modal
                modal.style.display = 'block';

                // Close button functionality
                var closeBtn = document.getElementsByClassName('close-btn')[0];
                closeBtn.onclick = function() {{
                    modal.style.display = 'none';
                }};

                // Close when clicking outside
                window.onclick = function(event) {{
                    if (event.target === modal) {{
                        modal.style.display = 'none';
                    }}
                }};
            }}

            // Initialize everything when the document is loaded
            document.addEventListener('DOMContentLoaded', function() {{
                initializeSearch();
                setupCollapsible();
                setupFilters();
                setupTabs();
                applyFilters();
            }});
        </script>
    </body>
    </html>
    """

    # Save the HTML file
    output_file = output_dir / 'jao_network_matching_results.html'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html)

    return output_file

def match_remaining_unmatched_network_lines(matching_results, jao_gdf, network_gdf):
    """
    Final pass to match any remaining unmatched network lines to appropriate JAO lines.
    This ensures maximum coverage of network lines in the matching process.
    """
    print("\n=== MATCHING REMAINING UNMATCHED NETWORK LINES ===")

    # Define helper function for calculating direction cosine
    def dir_cos(geom1, geom2):
        """Calculate the direction cosine between two geometries (how parallel they are)."""
        try:
            import numpy as np
            from shapely.geometry import LineString, MultiLineString

            # Helper to get the main vector of a line geometry
            def main_vec(geom):
                if geom.geom_type == "MultiLineString":
                    # Use the longest component of a MultiLineString
                    geom = max(geom.geoms, key=lambda g: g.length)

                # Get coordinates as numpy array
                coords = np.array(list(geom.coords))

                # Create vector from first to last point
                vec = coords[-1] - coords[0]

                # Normalize the vector
                norm = np.linalg.norm(vec)
                if norm > 0:
                    return vec / norm
                return vec

            # Get main direction vectors for both geometries
            vec1 = main_vec(geom1)
            vec2 = main_vec(geom2)

            # Calculate absolute dot product (cosine of angle between vectors)
            # Absolute value because we don't care about direction, just alignment
            return float(abs(np.dot(vec1, vec2)))
        except Exception as e:
            print(f"Error calculating direction cosine: {e}")
            return 0.0

    # 1. Identify all network lines that are already used in matches
    used_network_ids = set()
    for result in matching_results:
        if result['matched'] and 'network_ids' in result:
            for network_id in result['network_ids']:
                used_network_ids.add(str(network_id))

    # 2. Get all unmatched network lines
    unmatched_network_lines = []
    for idx, row in network_gdf.iterrows():
        network_id = str(row['id'])
        if network_id not in used_network_ids:
            unmatched_network_lines.append({
                'id': network_id,
                'idx': idx,
                'geometry': row.geometry,
                'voltage': _safe_int(row['v_nom']),
                'length': calculate_length_meters(row.geometry)
            })

    print(f"Found {len(unmatched_network_lines)} unmatched network lines to process")

    # If no unmatched lines, nothing to do
    if not unmatched_network_lines:
        return matching_results

    # 3. Create a dictionary of all JAO lines with their geometries
    jao_geometries = {}
    for idx, row in jao_gdf.iterrows():
        jao_id = str(row['id'])
        jao_geometries[jao_id] = {
            'geometry': row.geometry,
            'voltage': _safe_int(row['v_nom']),
            'idx': idx
        }

    # 4. Process each unmatched network line
    improvements_made = 0
    for network_line in unmatched_network_lines:
        network_id = network_line['id']
        network_geom = network_line['geometry']
        network_voltage = network_line['voltage']

        # Find the best JAO match for this network line
        best_match = None
        best_score = 0.3  # Minimum threshold score

        for jao_id, jao_info in jao_geometries.items():
            jao_geom = jao_info['geometry']
            jao_voltage = jao_info['voltage']

            # Check voltage compatibility (with 380/400 equivalence)
            voltage_match = False
            if (network_voltage == 220 and jao_voltage == 220) or \
                    (network_voltage in [380, 400] and jao_voltage in [380, 400]):
                voltage_match = True
                voltage_factor = 1.0
            else:
                voltage_factor = 0.5  # Penalty for voltage mismatch

            # Calculate geometric score
            # 1. Buffer overlap
            coverage = calculate_geometry_coverage(jao_geom, network_geom, buffer_meters=1000)

            # 2. Direction alignment
            try:
                alignment = dir_cos(jao_geom, network_geom)
            except Exception as e:
                print(f"Error calculating alignment: {e}")
                alignment = 0

            # 3. Calculate combined score
            score = (0.7 * coverage + 0.3 * alignment) * voltage_factor

            # If this score is better than our current best, update
            if score > best_score:
                # Find the corresponding match result
                match_result = None
                for result in matching_results:
                    if result['jao_id'] == jao_id:
                        match_result = result
                        break

                # If no match exists, we'll need to create one
                best_match = {
                    'jao_id': jao_id,
                    'score': score,
                    'coverage': coverage,
                    'alignment': alignment,
                    'match_result': match_result,
                    'voltage_match': voltage_match,
                    'jao_idx': jao_info['idx']
                }
                best_score = score

        # If we found a good match, update the matching results
        if best_match:
            print(f"  Network line {network_id} ({network_voltage} kV) matches with JAO {best_match['jao_id']}")
            print(
                f"    Score: {best_match['score']:.3f}, Coverage: {best_match['coverage']:.2f}, Alignment: {best_match['alignment']:.2f}")

            # Get or create match result
            match_result = best_match['match_result']

            # If no existing match, create one
            if match_result is None:
                # Get JAO info for the match
                jao_row = jao_gdf.iloc[best_match['jao_idx']]
                jao_length = calculate_length_meters(jao_row.geometry)

                match_result = {
                    'jao_id': best_match['jao_id'],
                    'jao_name': str(jao_row['NE_name']),
                    'v_nom': int(jao_row['v_nom']),
                    'matched': True,
                    'is_geometric_match': True,
                    'network_ids': [],
                    'jao_length': float(jao_length),
                    'path_length': 0,
                    'match_quality': 'Geometric Match (final pass)'
                }
                matching_results.append(match_result)
                print(f"    Created new match for previously unmatched JAO {best_match['jao_id']}")

            # Add network line to match
            if 'network_ids' not in match_result:
                match_result['network_ids'] = []

            ok = _try_append_with_path_lock(match_result, network_id, network_gdf,
                                            max_offcorridor_m=300, max_len_ratio_after=1.30)
            if ok is None:  # not path-based → keep old behavior
                match_result.setdefault('network_ids', []).append(network_id)
                _normalize_network_ids_and_path_length(match_result, network_gdf)
            elif ok is False:
                continue

            # Update path length
            if 'path_length' not in match_result:
                match_result['path_length'] = 0

            match_result['path_length'] = float(match_result['path_length'] + network_line['length'])

            # Update length ratio
            if 'jao_length' in match_result and match_result['jao_length'] > 0:
                match_result['length_ratio'] = float(match_result['path_length'] / match_result['jao_length'])

            # If not already marked as a specific match type, mark as geometric
            if not match_result.get('is_duplicate', False) and \
                    not match_result.get('is_parallel_circuit', False) and \
                    not match_result.get('is_parallel_voltage_circuit', False):
                match_result['is_geometric_match'] = True

            # If a new match or previously unmatched, mark it matched
            match_result['matched'] = True

            # If voltage mismatch, update match quality
            if not best_match['voltage_match'] and 'voltage mismatch' not in match_result.get('match_quality', ''):
                match_result['match_quality'] += ' (voltage mismatch)'

            improvements_made += 1
        else:
            print(f"  No good match found for network line {network_id}")

    print(f"Added {improvements_made} network lines to matches")

    return matching_results


def improve_visualization_of_unmatched_network_lines(html_content):
    """
    Improve the visibility of unmatched network lines in the visualization.
    """
    # Change the styling for unmatched network lines to make them more visible
    improved_style = """
        case "unmatched":
            return {
                "color": "#FF0000", // Bright red
                "weight": 3,        // Thicker line
                "opacity": 0.8,     // Higher opacity
                "dashArray": "5,5"  // Dashed pattern
            };
    """

    # Replace the existing style for unmatched network lines
    html_content = html_content.replace(
        """default: // unmatched
            return {
                "color": "blue",
                "weight": 2,
                "opacity": 0.6
            };""",
        improved_style
    )

    return html_content


def repair_network_graph(G, network_gdf, connection_threshold_meters=50):
    import math, networkx as nx
    positions = nx.get_node_attributes(G, 'pos')
    dangling = [n for n in G.nodes if G.degree[n] == 1 and n in positions]
    def m_per_deg(lat): return 111111*math.cos(math.radians(abs(lat)))
    added = 0
    for i, a in enumerate(dangling):
        xa, ya = positions[a]
        for b in dangling[i+1:]:
            xb, yb = positions[b]
            mid = 0.5*(ya+yb)
            d = ((xa-xb)**2 + (ya-yb)**2) ** 0.5 * m_per_deg(mid)
            if d <= connection_threshold_meters and not G.has_edge(a,b):
                G.add_edge(a, b, weight=0.0, connector=True, bridge=True)
                added += 1
    print(f"Added {added} new connections to repair the graph")
    return G



def convert_geometric_to_path_matches(matching_results, G, jao_gdf, network_gdf, nearest_points_dict):
    """
    Check geometric matches to see if they can be converted to path-based matches.
    """
    import networkx as nx

    print("\nChecking if geometric matches can be converted to path-based matches...")

    conversions = 0

    for result in matching_results:
        # Only check geometric matches
        if not result.get('matched', False) or not result.get('is_geometric_match', False):
            continue

        jao_id = result['jao_id']
        print(f"  Checking JAO {jao_id}")

        # Find the JAO in the dataframe
        jao_rows = jao_gdf[jao_gdf['id'].astype(str) == jao_id]
        if jao_rows.empty:
            continue

        jao_idx = jao_rows.index[0]

        # Check if we have endpoint matches
        if jao_idx not in nearest_points_dict:
            continue

        if nearest_points_dict[jao_idx]['start_nearest'] is None or nearest_points_dict[jao_idx]['end_nearest'] is None:
            continue

        # Get node IDs for start and end points
        start_idx, start_pos = nearest_points_dict[jao_idx]['start_nearest']
        end_idx, end_pos = nearest_points_dict[jao_idx]['end_nearest']

        start_node = f"node_{start_idx}_{start_pos}"
        end_node = f"node_{end_idx}_{end_pos}"

        # Check if there's a path between the nodes
        try:
            path = nx.shortest_path(G, start_node, end_node, weight='weight')

            # Extract network lines in the path
            network_ids = []
            for i in range(len(path) - 1):
                edge_data = G.get_edge_data(path[i], path[i + 1])
                if edge_data and 'id' in edge_data and not edge_data.get('connector', False):
                    network_ids.append(edge_data['id'])

            # Check if the path contains all network lines that were matched geometrically
            existing_network_ids = set(result.get('network_ids', []))
            path_network_ids = set(network_ids)

            # If the path contains at least 50% of the geometric matches
            overlap = len(existing_network_ids.intersection(path_network_ids))
            if overlap / len(existing_network_ids) >= 0.5:
                print(f"    Converting geometric match to path-based match")
                result['is_geometric_match'] = False
                result['match_quality'] = 'Converted to Path-based Match'
                result['path'] = [str(p) for p in path]
                result['is_path_based'] = True
                _init_or_update_path_lock(result, network_gdf, offcorridor_m=300)

                # Update network IDs to include both sets
                result['network_ids'] = list(existing_network_ids.union(path_network_ids))

                # Update path length
                path_length = 0
                for network_id in result['network_ids']:
                    network_rows = network_gdf[network_gdf['id'].astype(str) == network_id]
                    if not network_rows.empty:
                        path_length += calculate_length_meters(network_rows.iloc[0].geometry)

                result['path_length'] = float(path_length)

                # Update length ratio
                if 'jao_length' in result and result['jao_length'] > 0:
                    result['length_ratio'] = float(result['path_length'] / result['jao_length'])

                conversions += 1

        except nx.NetworkXNoPath:
            print(f"    No path found between endpoints, keeping as geometric match")

    print(f"Converted {conversions} geometric matches to path-based matches")
    return matching_results


def match_remaining_parallel_network_lines(matching_results, jao_gdf, network_gdf,
                                           *, buffer_m=350, min_overlap=0.6,
                                           min_alignment=0.75, max_hausdorff_m=250,
                                           max_len_ratio_after=2.5, max_offcorridor_m=400):
    """
    Attach still-unmatched network lines that run in parallel to network lines
    already matched to some JAO. Uses STRtree for candidate prefiltering.
    Respects 'is_path_based' on the recipient JAO result.
    """
    import numpy as np
    from shapely.strtree import STRtree

    # --- helpers ------------------------------------------------------------
    def _same_voltage_local(a, b):
        try:
            A = 400 if int(a) in (380, 400) else int(a)
            B = 400 if int(b) in (380, 400) else int(b)
            return A == B
        except Exception:
            return False

    def _m_per_deg_at(g):
        return _meters_per_degree(g.centroid.y)

    def _hausdorff_m(a, b):
        try:
            return float(a.hausdorff_distance(b) * _m_per_deg_at(a))
        except Exception:
            return 1e9

    def _alignment(geom1, geom2):
        try:
            import numpy as np
            from shapely.geometry import MultiLineString
            def main_vec(geom):
                if geom.geom_type == "MultiLineString":
                    geom = max(geom.geoms, key=lambda g: g.length)
                cs = np.array(list(geom.coords))
                v = cs[-1] - cs[0]
                n = np.linalg.norm(v)
                return v / n if n > 0 else v
            v1 = main_vec(geom1); v2 = main_vec(geom2)
            return float(abs(np.dot(v1, v2)))
        except Exception:
            return 0.0

    # map network_id -> result (first that contains it)
    net_to_result = {}
    for res in matching_results:
        if res.get('matched'):
            for nid in res.get('network_ids', []) or []:
                net_to_result[str(nid)] = res

    # collect matched and unmatched network lines
    used = set(net_to_result.keys())
    matched_geoms = []
    matched_ids = []
    matched_v = []
    for _, r in network_gdf.iterrows():
        nid = str(r['id'])
        if nid in used:
            matched_geoms.append(r.geometry)
            matched_ids.append(nid)
            matched_v.append(_safe_int(r['v_nom']))

    unmatched = []
    for _, r in network_gdf.iterrows():
        nid = str(r['id'])
        if nid not in used:
            unmatched.append((nid, r.geometry, _safe_int(r['v_nom'])))

    if not matched_geoms or not unmatched:
        return matching_results

    tree = STRtree(matched_geoms)
    try:
        wkb_to_idx = {g.wkb: i for i, g in enumerate(matched_geoms)}
    except Exception:
        wkb_to_idx = None

    def _hits_to_idx(hits):
        import numpy as np
        if isinstance(hits, (list, tuple)):
            geoms = hits
        else:
            geoms = list(np.atleast_1d(hits))
        out = []
        if wkb_to_idx is not None:
            for gh in geoms:
                ii = wkb_to_idx.get(gh.wkb)
                if ii is not None:
                    out.append(ii)
        else:
            # rare fallback: linear scan
            for i, g in enumerate(matched_geoms):
                out.append(i)
        return out

    added = 0
    for nid, ngeom, nvolt in unmatched:
        # search hits using a quick buffer (in degrees)
        buf_deg = (buffer_m / _m_per_deg_at(ngeom))
        hits = tree.query(ngeom.buffer(buf_deg))
        if not hits:
            continue
        cand_idx = _hits_to_idx(hits)

        # evaluate against all candidate matched lines
        best = None
        for i in cand_idx:
            mgeom = matched_geoms[i]
            mvolt = matched_v[i]
            if not _same_voltage_local(nvolt, mvolt):
                continue

            # quick mutual coverage (in meters tolerance)
            cov1 = calculate_geometry_coverage(ngeom, mgeom, buffer_meters=buffer_m)
            cov2 = calculate_geometry_coverage(mgeom, ngeom, buffer_meters=buffer_m)
            avg_cov = (cov1 + cov2) / 2.0
            if avg_cov < min_overlap:
                continue

            align = _alignment(ngeom, mgeom)
            if align < min_alignment:
                continue

            hd = _hausdorff_m(ngeom, mgeom)
            if hd > max_hausdorff_m:
                continue

            score = 0.6 * avg_cov + 0.2 * align + 0.2 * (1 - min(1, hd / max_hausdorff_m))
            if (best is None) or (score > best[0]):
                best = (score, i)

        if best is None:
            continue

        # attach this line to the same JAO as the matched base
        base_idx = best[1]
        base_nid = matched_ids[base_idx]
        res = net_to_result.get(base_nid)
        if res is None:
            continue

        # soft lock protection
        if res.get('is_path_based'):
            # only allow if close to existing path and won't blow ratio
            from shapely.ops import unary_union, linemerge
            lines = []
            for x in res.get('network_ids', []) or []:
                row = network_gdf[network_gdf['id'].astype(str) == str(x)]
                if not row.empty:
                    lines.append(row.iloc[0].geometry)
            if lines:
                merged = linemerge(unary_union(lines))
                dist_m = float(ngeom.distance(merged) * _m_per_deg_at(ngeom))
                if dist_m > max_offcorridor_m:
                    continue
                jl = float(res.get('jao_length') or 0.0)
                if jl > 0:
                    new_len = float(res.get('path_length') or 0.0) + float(ngeom.length * _m_per_deg_at(ngeom))
                    if (new_len / jl) > max_len_ratio_after:
                        continue

        res.setdefault('network_ids', []).append(nid)
        try:
            _normalize_network_ids_and_path_length(res, network_gdf)
        except Exception:
            pass
        res['is_parallel_circuit'] = True
        added += 1

    print(f"Added {added} remaining parallel network lines")
    return matching_results





def share_among_same_endpoints(matching_results, jao_gdf, nearest_points_dict):
    """If two JAOs have the same endpoints (same substation cluster or same coords) and voltage,
    make them share their network_ids. Safe against missing nearest_points info."""
    from shapely.geometry import LineString, MultiLineString
    import numpy as np

    def _ends_of_geom(geom):
        # pick longest part if MultiLineString
        try:
            if geom.geom_type == "MultiLineString":
                geom = max(geom.geoms, key=lambda g: g.length)

            def _as_linestring(g):
                return max(g.geoms, key=lambda p: p.length) if g.geom_type == 'MultiLineString' else g

            geom = _as_linestring(row.geometry)
            coords = list(geom.coords)
            return coords[0], coords[-1]
        except Exception:
            return (None, None)

    def _safe_node_id(pair, coord_fallback):
        # pair is e.g. (node_idx, (x,y)) or None
        if pair and isinstance(pair, (list, tuple)) and len(pair) >= 1 and pair[0] is not None:
            return f"node_{pair[0]}"
        if coord_fallback:
            x, y = coord_fallback
            return f"coord_{round(float(x),5)}_{round(float(y),5)}"
        return "coord_none"

    def key_for(idx, geom):
        info = nearest_points_dict.get(idx) or {}
        s_coord, e_coord = _ends_of_geom(geom)

        # Prefer clustered ids; else nearest-pair node id; else rounded coord key
        start_id = info.get('start_cluster_id')
        if start_id is None:
            start_id = _safe_node_id(info.get('start_nearest'), s_coord)

        end_id = info.get('end_cluster_id')
        if end_id is None:
            end_id = _safe_node_id(info.get('end_nearest'), e_coord)

        # orderless key for the pair of endpoints
        return tuple(sorted([str(start_id), str(end_id)]))

    # Map jao_id -> endpoint key
    endpoint_key = {}
    for idx, row in jao_gdf.iterrows():
        endpoint_key[str(row['id'])] = key_for(idx, row.geometry)

    # Bucket by (endpoint_key, voltage)
    buckets = {}
    for r in matching_results:
        if not r.get('matched'):
            continue
        jid = str(r['jao_id'])
        k = (endpoint_key.get(jid), int(r.get('v_nom', 0)))
        buckets.setdefault(k, []).append(r)

    # Share network_ids within each bucket
    for group in buckets.values():
        if len(group) < 2:
            continue
        all_ids = []
        for r in group:
            all_ids.extend(r.get('network_ids', []))
        union_ids = list(dict.fromkeys(all_ids))
        for r in group:
            r['network_ids'] = union_ids
            r['matched'] = bool(union_ids)

    return matching_results




def match_identical_network_geometries(matching_results, jao_gdf, network_gdf):
    import numpy as np
    """
    Match unmatched network lines that follow the same geometry as already matched network lines.
    More aggressive version to find all identical/parallel lines.
    """
    print("\n=== MATCHING IDENTICAL GEOMETRY NETWORK LINES (AGGRESSIVE) ===")

    # 1. Identify all network lines that are already used in matches
    used_network_ids = set()
    # Track which JAO each network line is matched to
    network_to_jao = {}

    for result in matching_results:
        if result['matched'] and 'network_ids' in result:
            jao_id = result['jao_id']
            for network_id in result['network_ids']:
                used_network_ids.add(str(network_id))
                network_to_jao[str(network_id)] = jao_id

    # 2. Get all unmatched network lines
    unmatched_network_lines = []
    for idx, row in network_gdf.iterrows():
        network_id = str(row['id'])
        if network_id not in used_network_ids:
            unmatched_network_lines.append({
                'id': network_id,
                'idx': idx,
                'geometry': row.geometry,
                'voltage': _safe_int(row['v_nom']),
                'length': calculate_length_meters(row.geometry),
                'bounds': row.geometry.bounds  # Store bounds for faster filtering
            })

    print(f"Found {len(unmatched_network_lines)} unmatched network lines to check")

    # 3. Create lists of matched network lines for each voltage
    matched_lines_220kv = []
    matched_lines_400kv = []

    for idx, row in network_gdf.iterrows():
        network_id = str(row['id'])
        if network_id in used_network_ids:
            line_info = {
                'id': network_id,
                'geometry': row.geometry,
                'voltage': _safe_int(row['v_nom']),
                'jao_id': network_to_jao.get(network_id),
                'bounds': row.geometry.bounds
            }

            if line_info['voltage'] == 220:
                matched_lines_220kv.append(line_info)
            elif line_info['voltage'] in [380, 400]:
                matched_lines_400kv.append(line_info)

    print(
        f"Processing {len(matched_lines_220kv)} matched 220kV lines and {len(matched_lines_400kv)} matched 380/400kV lines")

    # 4. For each unmatched network line, find if it's geometrically similar to a matched one
    matches_made = 0
    newly_matched_ids = set()

    # Process each voltage level separately
    for voltage in [220, 400]:
        # Filter unmatched lines by voltage
        if voltage == 220:
            unmatched_subset = [line for line in unmatched_network_lines if line['voltage'] == 220]
            matched_subset = matched_lines_220kv
        else:  # 380/400kV
            unmatched_subset = [line for line in unmatched_network_lines if line['voltage'] in [380, 400]]
            matched_subset = matched_lines_400kv

        print(f"Processing {len(unmatched_subset)} unmatched {voltage}kV lines")

        for unmatched in unmatched_subset:
            # Skip if already matched in this run
            if unmatched['id'] in newly_matched_ids:
                continue

            unmatched_id = unmatched['id']
            unmatched_geom = unmatched['geometry']
            unmatched_bounds = unmatched['bounds']

            # Use bounds to quickly filter potential matches
            minx, miny, maxx, maxy = unmatched_bounds
            # Expand bounds slightly to catch nearby lines
            buffer = 0.01  # ~1km in degrees
            search_bounds = (minx - buffer, miny - buffer, maxx + buffer, maxy + buffer)

            # Find best match
            best_match = None
            best_similarity = 0.75  # Lower threshold to catch more similar lines

            for matched in matched_subset:
                # Quick bounds check
                m_minx, m_miny, m_maxx, m_maxy = matched['bounds']

                # Skip if bounds don't overlap at all (quick filter)
                if (m_maxx < search_bounds[0] or m_minx > search_bounds[2] or
                        m_maxy < search_bounds[1] or m_miny > search_bounds[3]):
                    continue

                matched_geom = matched['geometry']

                # Calculate similarity
                try:
                    # Use a larger buffer - 300m instead of 200m
                    buffer_meters = 300

                    # Check overlap in both directions
                    overlap1 = calculate_geometry_coverage(matched_geom, unmatched_geom, buffer_meters)
                    overlap2 = calculate_geometry_coverage(unmatched_geom, matched_geom, buffer_meters)

                    # Average the overlap scores
                    avg_overlap = (overlap1 + overlap2) / 2

                    # If the overlap is decent, calculate more detailed metrics
                    if avg_overlap >= 0.7:
                        # Calculate Hausdorff distance
                        hausdorff_dist = matched_geom.hausdorff_distance(unmatched_geom)
                        avg_lat = (matched_geom.centroid.y + unmatched_geom.centroid.y) / 2
                        meters_per_degree = 111111 * np.cos(np.radians(abs(avg_lat)))
                        hausdorff_meters = hausdorff_dist * meters_per_degree

                        # Calculate length similarity
                        matched_length = calculate_length_meters(matched_geom)
                        length_ratio = unmatched['length'] / matched_length
                        length_similarity = 1 - min(abs(length_ratio - 1), 0.5) / 0.5

                        # Check if they're roughly parallel
                        try:
                            from shapely.geometry import LineString, MultiLineString
                            import numpy as np

                            # Get vectors for both lines
                            def get_vector(geom):
                                if geom.geom_type == 'MultiLineString':
                                    geom = max(geom.geoms, key=lambda g: g.length)

                                coords = np.array(list(geom.coords))
                                vec = coords[-1] - coords[0]
                                norm = np.linalg.norm(vec)
                                return vec / norm if norm > 0 else vec

                            vec1 = get_vector(matched_geom)
                            vec2 = get_vector(unmatched_geom)

                            # Calculate dot product (cosine of angle)
                            alignment = abs(np.dot(vec1, vec2))
                        except:
                            # If calculation fails, assume moderate alignment
                            alignment = 0.7

                        # Combine all metrics into a similarity score
                        # Heaviest weight on overlap and Hausdorff distance
                        similarity = (
                                0.5 * avg_overlap +
                                0.3 * (1 - min(1, hausdorff_meters / 500)) +
                                0.1 * length_similarity +
                                0.1 * alignment
                        )

                        if similarity > best_similarity:
                            best_similarity = similarity
                            best_match = {
                                'network_id': matched['id'],
                                'jao_id': matched['jao_id'],
                                'similarity': similarity,
                                'overlap': avg_overlap,
                                'hausdorff_meters': hausdorff_meters,
                                'alignment': alignment,
                                'length_ratio': length_ratio
                            }
                except Exception as e:
                    print(f"  Error comparing network lines {unmatched_id} and {matched['id']}: {e}")
                    continue

            # If we found a good match, add this network line to the same JAO
            if best_match:
                jao_id = best_match['jao_id']

                # Find the matching result for this JAO
                for result in matching_results:
                    if result['jao_id'] == jao_id and result['matched']:
                        print(f"  Network line {unmatched_id} matches with network line {best_match['network_id']} "
                              f"(similarity: {best_similarity:.3f}, overlap: {best_match['overlap']:.2f}, "
                              f"hausdorff: {best_match['hausdorff_meters']:.1f}m)")
                        print(f"  Adding to JAO {jao_id}")

                        # Add to network IDs
                        if 'network_ids' not in result:
                            result['network_ids'] = []
                        result['network_ids'].append(unmatched_id)
                        newly_matched_ids.add(unmatched_id)

                        # Update path length
                        if 'path_length' not in result:
                            result['path_length'] = 0
                        result['path_length'] = float(result['path_length'] + unmatched['length'])

                        # Update length ratio
                        if 'jao_length' in result and result['jao_length'] > 0:
                            result['length_ratio'] = float(result['path_length'] / result['jao_length'])

                        # If this wasn't already a parallel circuit, mark it as one
                        if not result.get('is_parallel_circuit', False) and not result.get('is_duplicate', False):
                            result['is_parallel_circuit'] = True
                            if 'v_nom' in result:
                                result[
                                    'match_quality'] = f'Parallel Circuit ({result["v_nom"]} kV) - Identical Geometry'
                            else:
                                result['match_quality'] = 'Parallel Circuit - Identical Geometry'

                        matches_made += 1
                        break

    print(f"Matched {matches_made} network lines based on identical geometry")

    return matching_results


def cluster_identical_network_lines(matching_results, network_gdf,
                                    *,
                                    bbox_round=5,
                                    len_bucket_m=100,
                                    hd_threshold_m=300,
                                    path_lock_offcorridor_m=300,
                                    path_lock_max_len_ratio_after=1.35):
    """
    Cluster network lines that are effectively identical (same bbox bucket,
    similar length, small Hausdorff). If a cluster contains both matched and
    unmatched lines of the same voltage, attach the unmatched lines to the
    dominant JAO in that cluster. STRtree used for quick grouping.
    Respects path-lock.
    """
    import numpy as np
    from shapely.strtree import STRtree

    print("\n=== CLUSTERING IDENTICAL NETWORK LINES (FAST) ===")

    # map matched network -> (jao_id)
    used = set()
    net2res = {}
    net2jao = {}
    for r in matching_results:
        if r.get('matched') and r.get('network_ids'):
            for nid in r['network_ids']:
                used.add(str(nid))
                net2jao[str(nid)] = r['jao_id']
                net2res[str(nid)] = r

    # build voltage buckets with (nid, geom, v, length)
    items = []
    for i, row in network_gdf.iterrows():
        nid = str(row['id'])
        v = _safe_int(row.get('v_nom'))
        g = row.geometry
        L = calculate_length_meters(g)
        items.append((nid, v, g, L))

    # group by voltage
    by_v = {}
    for rec in items:
        by_v.setdefault(rec[1], []).append(rec)

    matched_count = 0

    def _same_voltage_group(v):
        return (v == 220) or (v in (380, 400))

    for v, lst in by_v.items():
        # STRtree for this voltage
        geos = [g for (_, _, g, _) in lst]
        try:
            tree = STRtree(geos)
        except Exception:
            tree = None

        # candidate cluster detection: bbox & length bucket
        buckets = {}
        for idx, (nid, vv, geom, L) in enumerate(lst):
            b = geom.bounds
            key_bbox = tuple(round(x, bbox_round) for x in b)
            key_len = round(L / len_bucket_m) * len_bucket_m
            key = (key_bbox, key_len)
            buckets.setdefault(key, []).append(idx)

        for key, idxs in buckets.items():
            if len(idxs) <= 1:
                continue

            # refine by Hausdorff to split into tighter groups
            group = [lst[i] for i in idxs]
            refined = []
            remaining = group[:]
            while remaining:
                base = remaining.pop(0)
                cur = [base]
                g0 = base[2]
                avg_lat = float(g0.centroid.y)
                mdeg = _meters_per_degree(avg_lat)

                i = 0
                while i < len(remaining):
                    g1 = remaining[i][2]
                    hd_m = g0.hausdorff_distance(g1) * mdeg
                    if hd_m <= hd_threshold_m:
                        cur.append(remaining.pop(i))
                    else:
                        i += 1
                if len(cur) > 1:
                    refined.append(cur)

            # For each refined cluster, propagate matches
            for grp in refined:
                matched = [x for x in grp if str(x[0]) in used]
                unmatched = [x for x in grp if str(x[0]) not in used]
                if not matched or not unmatched:
                    continue

                # choose dominant JAO by plurality
                votes = {}
                for (nid_m, _, _, _) in matched:
                    jid = net2jao[str(nid_m)]
                    votes[jid] = votes.get(jid, 0) + 1
                jao_id = max(votes, key=votes.get)
                res = net2res[[nm for nm in votes if nm == jao_id][0]] if False else None
                # Actually get the actual result for jao_id:
                for r in matching_results:
                    if r.get('jao_id') == jao_id and r.get('matched'):
                        res = r
                        break
                if not res:
                    continue

                # append unmatched to this result if allowed by path-lock
                for (nid_u, _, g_u, L_u) in unmatched:
                    rows = network_gdf[network_gdf['id'].astype(str) == str(nid_u)]
                    if rows.empty:
                        continue
                    ngeom = rows.iloc[0].geometry
                    add_len = calculate_length_meters(ngeom)

                    if res.get('is_path_based') and res.get('_path_corridor') is not None:
                        if not res['_path_corridor'].intersects(ngeom):
                            continue
                        jao_len = float(res.get('jao_length') or 0.0)
                        if jao_len > 0:
                            new_ratio = float(res.get('path_length', 0.0) + add_len) / jao_len
                            if new_ratio > path_lock_max_len_ratio_after:
                                continue

                    res.setdefault('network_ids', []).append(nid_u)
                    _normalize_network_ids_and_path_length(res, network_gdf)
                    if not res.get('is_parallel_circuit', False) and not res.get('is_duplicate', False):
                        res['is_parallel_circuit'] = True
                        vtxt = f" ({res.get('v_nom')} kV)" if res.get('v_nom') else ""
                        res['match_quality'] = f"Parallel Circuit{vtxt} - Clustered"
                    used.add(str(nid_u))
                    matched_count += 1

    print(f"Matched {matched_count} additional network lines by clustering.")
    return matching_results






def match_remaining_identical_network_lines(matching_results, network_gdf,
                                            *, hd_thresh_m=150, buf_m=150):
    """
    Final strict pass: if an unmatched network line is *virtually identical* to any
    matched line (same voltage, tiny Hausdorff, high overlap), attach it to the
    same JAO result. Uses STRtree.
    """
    from shapely.strtree import STRtree
    import numpy as np

    def _same_voltage_local(a, b):
        try:
            A = 400 if int(a) in (380, 400) else int(a)
            B = 400 if int(b) in (380, 400) else int(b)
            return A == B
        except Exception:
            return False

    used = set()
    net_to_jao = {}
    for res in matching_results:
        if res.get('matched'):
            for nid in res.get('network_ids', []) or []:
                used.add(str(nid))
                net_to_jao[str(nid)] = res['jao_id']

    matched, unmatched = [], []
    for _, r in network_gdf.iterrows():
        nid = str(r['id']); gv = _safe_int(r['v_nom']); gg = r.geometry
        if nid in used:
            matched.append((nid, gg, gv))
        else:
            unmatched.append((nid, gg, gv))

    if not matched or not unmatched:
        return matching_results

    m_geoms = [g for (_, g, _) in matched]
    m_ids   = [nid for (nid, _, _) in matched]
    m_v     = [v for (_, _, v) in matched]
    tree = STRtree(m_geoms)
    try:
        m_wkb_to_idx = {g.wkb: i for i, g in enumerate(m_geoms)}
    except Exception:
        m_wkb_to_idx = None

    def _hits_to_idx(hits):
        if isinstance(hits, (list, tuple)):
            geoms = hits
        else:
            geoms = list(np.atleast_1d(hits))
        out = []
        if m_wkb_to_idx is not None:
            for gh in geoms:
                ii = m_wkb_to_idx.get(gh.wkb)
                if ii is not None:
                    out.append(ii)
        else:
            for i, _ in enumerate(m_geoms):
                out.append(i)
        return out

    added = 0
    for nid, ngeom, nvolt in unmatched:
        # prefilter via bbox hits
        hits = tree.query(ngeom)
        if not hits:
            continue
        cand_idx = _hits_to_idx(hits)

        best = None
        for i in cand_idx:
            mgeom = m_geoms[i]; mvolt = m_v[i]
            if not _same_voltage_local(nvolt, mvolt):
                continue

            lat = (mgeom.centroid.y + ngeom.centroid.y) / 2.0
            m_per_deg = _meters_per_degree(lat)
            hd_m = float(mgeom.hausdorff_distance(ngeom) * m_per_deg)
            if hd_m > hd_thresh_m:
                continue

            ov1 = calculate_geometry_coverage(mgeom, ngeom, buffer_meters=buf_m)
            ov2 = calculate_geometry_coverage(ngeom, mgeom, buffer_meters=buf_m)
            sim = 0.7 * (1 - min(1, hd_m / hd_thresh_m)) + 0.3 * ((ov1 + ov2) / 2.0)
            if best is None or sim > best[0]:
                best = (sim, i)

        if best is None or best[0] < 0.95:
            continue

        # attach to same JAO as matched base
        base_i = best[1]
        base_id = m_ids[base_i]
        jao_id = net_to_jao.get(base_id)
        if not jao_id:
            continue

        recipient = None
        for res in matching_results:
            if res.get('matched') and str(res['jao_id']) == str(jao_id):
                recipient = res
                break
        if not recipient:
            continue

        recipient.setdefault('network_ids', []).append(nid)
        try:
            _normalize_network_ids_and_path_length(recipient, network_gdf)
        except Exception:
            pass
        recipient['is_parallel_circuit'] = True
        added += 1

    print(f"Matched {added} network lines in final identical-geometry pass (STRtree)")
    return matching_results




def corridor_parallel_match(matching_results, jao_gdf, network_gdf,
                            *,
                            corridor_w_220=300,
                            corridor_w_400=400,
                            path_lock_offcorridor_m=300,
                            path_lock_max_len_ratio_after=1.35):
    """
    Corridor approach:
      1) Build voltage-specific buffers (vectorized when Shapely 2)
      2) Dissolve to corridors
      3) For each corridor containing at least one matched line, propagate to
         unmatched lines in the same corridor if path-lock allows.
    """
    import numpy as np
    import geopandas as gpd
    from shapely.ops import unary_union

    print("\n=== CORRIDOR-BASED MATCHING (VECTORIZED WHERE POSSIBLE) ===")

    df = network_gdf.copy()
    df['network_id'] = df['id'].astype(str)

    # vectorized buffer if Shapely 2.x (geometry.buffer can take array-like)
    try:
        m_per_deg = df.geometry.centroid.y.map(_meters_per_degree)
        widths_m = np.where(df['v_nom'].astype(int) == 220, corridor_w_220, corridor_w_400)
        width_deg = widths_m / m_per_deg
        df['buffer'] = df.geometry.buffer(width_deg)
    except Exception:
        # fallback to row-wise
        def _buf(row):
            lat = float(row.geometry.centroid.y)
            w = corridor_w_220 if int(row['v_nom']) == 220 else corridor_w_400
            return row.geometry.buffer(w / _meters_per_degree(lat))
        df['buffer'] = df.apply(_buf, axis=1)

    # dissolve by voltage
    corridors = []
    for v, grp in df.groupby('v_nom'):
        try:
            u = unary_union(list(grp['buffer']))
        except Exception:
            continue
        if u.is_empty:
            continue
        if hasattr(u, 'geoms'):
            for i, poly in enumerate(u.geoms):
                corridors.append({'corridor_id': f'{int(v)}_{i}', 'v_nom': int(v), 'geometry': poly})
        else:
            corridors.append({'corridor_id': f'{int(v)}_0', 'v_nom': int(v), 'geometry': u})

    if not corridors:
        print("No corridors created.")
        return matching_results

    cdf = gpd.GeoDataFrame(corridors, geometry='geometry', crs=network_gdf.crs)

    # spatial join: assign corridor to network lines (simple bbox/intersection)
    try:
        joined = gpd.sjoin(df[['network_id', 'v_nom', 'geometry']], cdf[['corridor_id', 'geometry']], how='left', predicate='within')
    except Exception:
        # fallback: manual
        records = []
        for i, row in df.iterrows():
            cid = None
            for j, crow in cdf.iterrows():
                if crow.geometry.contains(row.geometry):
                    cid = crow['corridor_id']; break
            records.append({'network_id': row['network_id'], 'v_nom': row['v_nom'], 'geometry': row.geometry, 'corridor_id': cid})
        joined = gpd.GeoDataFrame(records, geometry='geometry', crs=network_gdf.crs)

    # group lines by corridor
    cor_groups = {}
    for cid, sub in joined.groupby('corridor_id'):
        if cid is None or (isinstance(cid, float) and np.isnan(cid)):
            continue
        ids = sub['network_id'].tolist()
        cor_groups[cid] = ids

    # map network_id -> result (jao) if already used
    used = set()
    net2res = {}
    for r in matching_results:
        if r.get('matched') and r.get('network_ids'):
            for nid in r['network_ids']:
                used.add(str(nid))
                net2res[str(nid)] = r

    added = 0
    processed = 0

    for cid, ids in cor_groups.items():
        # find an anchor result in this corridor
        anchors = [net2res[i] for i in ids if i in net2res]
        anchors = [a for i, a in enumerate(anchors) if anchors.index(a) == i]  # unique results
        if not anchors:
            continue
        processed += 1

        # choose the first anchor (could be improved to majority vote)
        res = anchors[0]

        for nid in ids:
            if nid in used:
                continue
            rows = network_gdf[network_gdf['id'].astype(str) == str(nid)]
            if rows.empty:
                continue
            ngeom = rows.iloc[0].geometry
            add_len = calculate_length_meters(ngeom)

            # respect path-lock
            if res.get('is_path_based') and res.get('_path_corridor') is not None:
                if not res['_path_corridor'].intersects(ngeom):
                    continue
                jao_len = float(res.get('jao_length') or 0.0)
                if jao_len > 0:
                    new_ratio = float(res.get('path_length', 0.0) + add_len) / jao_len
                    if new_ratio > path_lock_max_len_ratio_after:
                        continue

            res.setdefault('network_ids', []).append(nid)
            _normalize_network_ids_and_path_length(res, network_gdf)
            if not res.get('is_parallel_circuit', False) and not res.get('is_duplicate', False):
                res['is_parallel_circuit'] = True
                vtxt = f" ({res.get('v_nom')} kV)" if res.get('v_nom') else ""
                res['match_quality'] = f"Parallel Circuit{vtxt} - Corridor Matched"

            used.add(nid)
            added += 1

    print(f"Added {added} network lines from {processed} corridors.")
    return matching_results




def match_identical_network_geometries_aggressive(matching_results, jao_gdf, network_gdf,
                                                  *, stages=((150, 0.95), (250, 0.90), (350, 0.85))):
    """
    Add unmatched network lines that follow the *same* geometry as matched ones.
    Tries rtree if available; falls back to STRtree. Progressive relaxation.
    """
    import numpy as np
    try:
        from rtree import index as rtree_index
        use_rtree = True
    except Exception:
        use_rtree = False
    from shapely.strtree import STRtree

    def _same_voltage_local(a, b):
        try:
            A = 400 if int(a) in (380, 400) else int(a)
            B = 400 if int(b) in (380, 400) else int(b)
            return A == B
        except Exception:
            return False

    # used and matched sets
    used = set()
    net_to_jao = {}
    for res in matching_results:
        if res.get('matched'):
            for nid in res.get('network_ids', []) or []:
                used.add(str(nid))
                net_to_jao[str(nid)] = res['jao_id']

    # collect geoms
    matched, unmatched = [], []
    for _, r in network_gdf.iterrows():
        nid = str(r['id']); gv = _safe_int(r['v_nom']); gg = r.geometry
        if nid in used:
            matched.append((nid, gg, gv))
        else:
            unmatched.append((nid, gg, gv))
    if not matched or not unmatched:
        return matching_results

    # spatial index
    if use_rtree:
        idx = rtree_index.Index()
        m_by_i = {}
        for i, (nid, g, v) in enumerate(matched):
            minx, miny, maxx, maxy = g.bounds
            idx.insert(i, (minx, miny, maxx, maxy))
            m_by_i[i] = (nid, g, v)
    else:
        m_geoms = [g for (_, g, _) in matched]
        m_ids   = [nid for (nid, _, _) in matched]
        m_v     = [v for (_, _, v) in matched]
        tree = STRtree(m_geoms)
        try:
            m_wkb_to_idx = {g.wkb: i for i, g in enumerate(m_geoms)}
        except Exception:
            m_wkb_to_idx = None
        def _hits_to_idx(hits):
            import numpy as np
            if isinstance(hits, (list, tuple)):
                geoms = hits
            else:
                geoms = list(np.atleast_1d(hits))
            out = []
            if m_wkb_to_idx is not None:
                for gh in geoms:
                    ii = m_wkb_to_idx.get(gh.wkb)
                    if ii is not None:
                        out.append(ii)
            else:
                for i, _ in enumerate(m_geoms):
                    out.append(i)
            return out

    total = 0
    newly = set()

    for buffer_m, min_sim in stages:
        found = 0
        remaining = [(nid, g, v) for (nid, g, v) in unmatched if nid not in newly]
        if not remaining:
            break

        for nid, ngeom, nvolt in remaining:
            # find candidate matched in same bbox area
            if use_rtree:
                minx, miny, maxx, maxy = ngeom.bounds
                cand_i = list(idx.intersection((minx, miny, maxx, maxy)))
                candidates = [m_by_i[i] for i in cand_i]
            else:
                hits = tree.query(ngeom)
                cand_idx = _hits_to_idx(hits)
                candidates = [(m_ids[i], m_geoms[i], m_v[i]) for i in cand_idx]

            best = None
            for mid, mgeom, mvolt in candidates:
                if not _same_voltage_local(nvolt, mvolt):
                    continue

                # compute similarity
                overlap1 = calculate_geometry_coverage(mgeom, ngeom, buffer_m)
                overlap2 = calculate_geometry_coverage(ngeom, mgeom, buffer_m)
                avg_overlap = (overlap1 + overlap2) / 2.0

                # hausdorff
                lat = (mgeom.centroid.y + ngeom.centroid.y) / 2.0
                m_per_deg = _meters_per_degree(lat)
                hd_m = float(mgeom.hausdorff_distance(ngeom) * m_per_deg)
                sim = 0.5 * avg_overlap + 0.3 * (1 - min(1, hd_m / buffer_m)) + 0.2 * 1.0  # alignment ~1 for identical

                if sim >= min_sim and (best is None or sim > best[0]):
                    best = (sim, mid)

            if best is None:
                continue

            # attach to the same JAO as mid
            jao_id = net_to_jao.get(best[1])
            if not jao_id:
                continue
            # find recipient result
            recipient = None
            for res in matching_results:
                if res.get('matched') and str(res['jao_id']) == str(jao_id):
                    recipient = res
                    break
            if not recipient:
                continue

            recipient.setdefault('network_ids', []).append(nid)
            try:
                _normalize_network_ids_and_path_length(recipient, network_gdf)
            except Exception:
                pass
            recipient['is_parallel_circuit'] = True
            newly.add(nid)
            found += 1

        print(f"  Stage buffer={buffer_m}m, min_sim={min_sim} → added {found}")
        total += found

    print(f"Matched {total} network lines using aggressive identical-geometry (rtree/STRtree)")
    return matching_results




def match_network_lines_by_geometry_hash(matching_results, network_gdf, sample_distance=250):
    """
    Match unmatched network lines using a geometry hash approach.

    This method:
    1. Resamples lines to have regular spacing
    2. Snaps vertices to a grid
    3. Creates a hash of the coordinates
    4. Groups lines with similar hashes

    Parameters:
    - matching_results: Current matching results
    - network_gdf: GeoDataFrame with network lines
    - sample_distance: Distance in meters between sample points

    Returns:
    - Updated matching results
    """
    import hashlib
    import numpy as np

    print("\n=== MATCHING NETWORK LINES USING GEOMETRY HASH ===")

    # Make sure we always return the matching_results even if errors occur
    if matching_results is None:
        print("Error: matching_results is None")
        return []  # Return empty list instead of None

    # 1. Identify all network lines that are already used in matches
    used_network_ids = set()
    network_to_jao = {}
    network_to_result = {}

    for result in matching_results:
        if result['matched'] and 'network_ids' in result:
            jao_id = result['jao_id']
            for network_id in result['network_ids']:
                used_network_ids.add(str(network_id))
                network_to_jao[str(network_id)] = jao_id
                network_to_result[str(network_id)] = result

    # 2. Create a hash for each network line's geometry
    print("Creating geometry hashes...")

    # Function to create a geometry hash
    def create_geometry_hash(geom, sample_dist_meters):
        try:
            # Calculate sample distance in degrees
            avg_lat = geom.centroid.y
            meters_per_degree = 111111 * np.cos(np.radians(abs(avg_lat)))
            sample_dist_deg = sample_dist_meters / meters_per_degree

            # Get total length of geometry
            length = geom.length

            # If very short, just use endpoints
            if length < sample_dist_deg * 2:
                coords = [list(geom.coords)[0], list(geom.coords)[-1]]
            else:
                # Sample points at regular intervals
                num_points = max(3, int(length / sample_dist_deg))
                points = []

                # Use shapely's interpolate method directly on the geometry
                for i in range(num_points):
                    # Calculate the distance along the line for this point
                    distance = i * length / (num_points - 1)
                    # Interpolate a point at this distance
                    point = geom.interpolate(distance)
                    points.append((point.x, point.y))

                coords = points

            # Snap to grid (round to 4 decimal places ≈ 10m)
            snapped_coords = [tuple(np.round(p, 4)) for p in coords]

            # Create hash of coordinates
            coord_str = str(snapped_coords)
            return hashlib.sha256(coord_str.encode()).hexdigest()
        except Exception as e:
            print(f"Error creating hash: {e}")
            # Fallback to geometry WKT if hashing fails
            return hashlib.sha256(geom.wkt.encode()).hexdigest()

    # Create hashes for all network lines
    network_hashes = {}

    try:
        for idx, row in network_gdf.iterrows():
            network_id = str(row['id'])
            geometry = row.geometry
            voltage = _safe_int(row['v_nom'])

            # Create hash
            hash_key = create_geometry_hash(geometry, sample_distance)

            # Store with voltage to avoid mixing different voltages
            voltage_hash = f"{voltage}_{hash_key}"

            if voltage_hash not in network_hashes:
                network_hashes[voltage_hash] = []

            network_hashes[voltage_hash].append({
                'id': network_id,
                'geometry': geometry,
                'voltage': voltage,
                'length': calculate_length_meters(geometry),
                'is_matched': network_id in used_network_ids
            })
    except Exception as e:
        print(f"Error creating network hashes: {e}")
        return matching_results  # Return original results if error occurs

    # 3. Find groups with both matched and unmatched lines
    print("Finding hash clusters with both matched and unmatched lines...")

    matches_made = 0

    try:
        for hash_key, lines in network_hashes.items():
            # Skip if only one line with this hash
            if len(lines) <= 1:
                continue

            # Find if any are already matched
            matched_lines = [line for line in lines if line['is_matched']]
            unmatched_lines = [line for line in lines if not line['is_matched']]

            # Skip if all are matched or all are unmatched
            if not matched_lines or not unmatched_lines:
                continue

            print(
                f"  Hash cluster {hash_key[:8]}... has {len(matched_lines)} matched and {len(unmatched_lines)} unmatched lines")

            # Group matched lines by JAO
            jao_groups = {}
            for line in matched_lines:
                jao_id = network_to_jao.get(line['id'])
                if jao_id not in jao_groups:
                    jao_groups[jao_id] = []
                jao_groups[jao_id].append(line)

            # Find the JAO with the most matched lines
            if not jao_groups:
                continue  # Skip if no JAO groups

            best_jao_id = max(jao_groups.keys(), key=lambda k: len(jao_groups[k]))
            first_line_id = jao_groups[best_jao_id][0]['id']

            best_result = network_to_result.get(first_line_id)

            if best_result:
                print(f"    Adding {len(unmatched_lines)} unmatched lines to JAO {best_jao_id}")

                for line in unmatched_lines:
                    # Add to network IDs
                    if 'network_ids' not in best_result:
                        best_result['network_ids'] = []
                    best_result['network_ids'].append(line['id'])

                    # Update path length
                    if 'path_length' not in best_result:
                        best_result['path_length'] = 0
                    best_result['path_length'] = float(best_result['path_length'] + line['length'])

                    matches_made += 1

                # Update length ratio
                if 'jao_length' in best_result and best_result['jao_length'] > 0:
                    best_result['length_ratio'] = float(best_result['path_length'] / best_result['jao_length'])

                # If this wasn't already a parallel circuit, mark it as one
                if not best_result.get('is_parallel_circuit', False) and not best_result.get('is_duplicate', False):
                    best_result['is_parallel_circuit'] = True
                    if 'v_nom' in best_result:
                        best_result['match_quality'] = f'Parallel Circuit ({best_result["v_nom"]} kV) - Geometry Hash'
                    else:
                        best_result['match_quality'] = 'Parallel Circuit - Geometry Hash'
    except Exception as e:
        print(f"Error in hash matching: {e}")
        # Continue processing even if an error occurs

    print(f"Matched {matches_made} network lines using geometry hash approach")

    # Always return the matching_results
    return matching_results


def _to_float_or_none(v):
    try:
        if v is None:
            return None
        if isinstance(v, (int, float)):
            return float(v)
        # handle strings like "1.23", "1,23", "  1.23 "
        s = str(v).strip().replace(",", ".")
        return float(s)
    except Exception:
        return None

def _same_voltage(a, b):
    """Treat 380 and 400 as equivalent."""
    try:
        A = 400 if int(a) in (380, 400) else int(a)
        B = 400 if int(b) in (380, 400) else int(b)
        return A == B
    except Exception:
        return False


def _normalize_network_ids_and_path_length(match_result, network_gdf, jao_length_m=None):
    """
    De-duplicate match_result['network_ids'] and recompute match_result['path_length']
    from network_gdf geometries (fallback to stored length columns if present).
    Also keeps length_ratio consistent if JAO length is known.
    """
    import pandas as pd

    # 0) guard
    if match_result is None:
        return

    # 1) dedupe, preserving order
    match_result['network_ids'] = list(dict.fromkeys(match_result.get('network_ids', []) or []))

    # 2) recompute path_length (meters)
    total_m = 0.0
    for nid in match_result['network_ids']:
        rows = network_gdf[network_gdf['id'].astype(str) == str(nid)]
        if rows.empty:
            continue
        row = rows.iloc[0]

        # Prefer geometry-based length if available
        try:
            seg_m = float(calculate_length_meters(row.geometry))
            total_m += seg_m
            continue
        except Exception:
            pass

        # Fallbacks: look for stored lengths (assumed km)
        if 'length' in row.index and pd.notna(row['length']):
            total_m += float(row['length']) * 1000.0
        elif 'length_km' in row.index and pd.notna(row['length_km']):
            total_m += float(row['length_km']) * 1000.0

    match_result['path_length'] = float(total_m)

    # 3) JAO length (meters) to compute length_ratio
    # priority: explicit arg > field 'jao_length' (m) > field 'jao_length_km' (km)
    if jao_length_m is None:
        jl = match_result.get('jao_length')
        if jl is not None:
            jao_length_m = float(jl)
        else:
            jlk = match_result.get('jao_length_km')
            if jlk is not None:
                try:
                    jao_length_m = float(jlk) * 1000.0
                except Exception:
                    jao_length_m = None

    if jao_length_m and jao_length_m > 0:
        match_result['length_ratio'] = float(match_result['path_length'] / jao_length_m)


def _union_path_geom(network_ids, network_gdf):
    from shapely.ops import unary_union
    geoms = []
    for nid in network_ids or []:
        rows = network_gdf[network_gdf['id'].astype(str) == str(nid)]
        if not rows.empty:
            geoms.append(rows.iloc[0].geometry)
    return unary_union(geoms) if geoms else None

def _init_or_update_path_lock(result, network_gdf, offcorridor_m=300):
    """Build/refresh a buffered corridor around the current path."""
    path = _union_path_geom(result.get('network_ids', []), network_gdf)
    result['_path_geom'] = path
    result['_path_corridor'] = None
    if path is None or path.is_empty:
        return
    avg_lat = path.centroid.y
    buf_deg = float(offcorridor_m) / _meters_per_degree(avg_lat)
    result['_path_corridor'] = path.buffer(buf_deg)

def _try_append_with_path_lock(result, candidate_nid, network_gdf,
                               max_offcorridor_m=300, max_len_ratio_after=1.30):
    """
    If result is path-locked, only allow adding candidate if:
      (1) it sits inside/along the current corridor, and
      (2) the post-append length_ratio <= cap.
    Returns True (appended), False (rejected), or None (not path-based → caller decides).
    """
    if not result.get('is_path_based'):
        return None

    if result.get('_path_corridor') is None:
        _init_or_update_path_lock(result, network_gdf, offcorridor_m=max_offcorridor_m)

    rows = network_gdf[network_gdf['id'].astype(str) == str(candidate_nid)]
    if rows.empty:
        return False
    g_new = rows.iloc[0].geometry

    corridor = result.get('_path_corridor')
    if corridor is None or corridor.is_empty or not corridor.intersects(g_new):
        return False

    try:
        cov = calculate_geometry_coverage(result['_path_geom'], g_new, buffer_meters=max_offcorridor_m)
        if cov < 0.35:
            return False
    except Exception:
        pass

    trial = dict(result)
    trial['network_ids'] = list(dict.fromkeys((result.get('network_ids') or []) + [str(candidate_nid)]))
    _normalize_network_ids_and_path_length(trial, network_gdf)

    jl = float(trial.get('jao_length') or 0.0)
    if jl > 0 and (trial['path_length'] / jl) > float(max_len_ratio_after):
        return False

    # commit
    result.setdefault('network_ids', []).append(str(candidate_nid))
    _normalize_network_ids_and_path_length(result, network_gdf)
    if jl := float(result.get('jao_length') or 0.0):
        if jl > 0:
            result['length_ratio'] = float(result['path_length'] / jl)
    _init_or_update_path_lock(result, network_gdf, offcorridor_m=max_offcorridor_m)

    if not result.get('is_parallel_circuit', False) and not result.get('is_duplicate', False):
        result['is_parallel_circuit'] = True
        result['match_quality'] = f"Parallel Circuit ({result.get('v_nom','?')} kV) - Path-locked add"
    return True


def normalize_and_fill_params(matching_results, jao_gdf, network_gdf):
    """
    Make sure each matched result has:
      - jao_length_km (float)
      - jao_r, jao_x, jao_b (totals)
      - jao_r_per_km, jao_x_per_km, jao_b_per_km
      - matched_km, coverage_ratio
    Pull from allocate output, else from jao_gdf; compute consistently.
    """

    # Build quick lookups from JAO dataframe
    # Expect: 'id' (str/int), 'length' (km), and either totals (r,x,b) or per-km (R_per_km, X_per_km, B_per_km)
    jao_len_km = {}
    jao_totals = {}     # id -> (r,x,b)
    jao_per_km = {}     # id -> (r_per_km,x_per_km,b_per_km)

    id_col = 'id'
    len_col = 'length'  # in km (you printed "Using 'length' column for JAO ...: 51.511 km")
    tot_cols = ['r', 'x', 'b']  # totals (ohm, ohm, S)
    perkm_cols = ['R_per_km', 'X_per_km', 'B_per_km']  # optional

    for _, row in jao_gdf.iterrows():
        jid = str(row.get(id_col))
        if not jid or jid == 'None':
            continue

        Lkm = _to_float_or_none(row.get(len_col))
        if Lkm is not None:
            jao_len_km[jid] = Lkm

        Rtot = _to_float_or_none(row.get(tot_cols[0]))
        Xtot = _to_float_or_none(row.get(tot_cols[1]))
        Btot = _to_float_or_none(row.get(tot_cols[2]))
        if any(v is not None for v in (Rtot, Xtot, Btot)):
            jao_totals[jid] = (Rtot, Xtot, Btot)

        Rpk = _to_float_or_none(row.get(perkm_cols[0]))
        Xpk = _to_float_or_none(row.get(perkm_cols[1]))
        Bpk = _to_float_or_none(row.get(perkm_cols[2]))
        if any(v is not None for v in (Rpk, Xpk, Bpk)):
            jao_per_km[jid] = (Rpk, Xpk, Bpk)

    # Optional: network lengths by id → km (best-effort fallback for coverage)
    net_len_km = {}
    net_id_col = 'id'
    net_len_col_candidates = ['length_km', 'length']  # choose the one you have (km)
    # Pick the first that exists
    for cand in net_len_col_candidates:
        if cand in network_gdf.columns:
            net_len_col = cand
            break
    else:
        net_len_col = None

    if net_len_col:
        for _, row in network_gdf.iterrows():
            nid = str(row.get(net_id_col))
            L = _to_float_or_none(row.get(net_len_col))
            if nid and L is not None:
                net_len_km[nid] = L

    # Normalize each result
    for r in matching_results:
        if not r.get('matched'):
            # Still normalize length so UI shows consistent numbers
            jid = str(r.get('jao_id'))
            if jid in jao_len_km:
                r['jao_length_km'] = jao_len_km[jid]
            continue

        jid = str(r.get('jao_id'))

        # ---- Length: authoritative from JAO DF ----
        Lkm = jao_len_km.get(jid)
        if Lkm is None:
            # fallback if you stored meters somewhere
            Lm = _to_float_or_none(r.get('jao_length'))
            if Lm is not None:
                Lkm = Lm / 1000.0
        r['jao_length_km'] = Lkm if Lkm is not None else _to_float_or_none(r.get('jao_length_km'))

        # ---- Map legacy keys -> canonical ----
        if r.get('jao_r') is None and r.get('jao_r_total') is not None:
            r['jao_r'] = _to_float_or_none(r.get('jao_r_total'))
        if r.get('jao_x') is None and r.get('jao_x_total') is not None:
            r['jao_x'] = _to_float_or_none(r.get('jao_x_total'))
        if r.get('jao_b') is None and r.get('jao_b_total') is not None:
            r['jao_b'] = _to_float_or_none(r.get('jao_b_total'))

        # ---- Fill missing JAO totals/per-km from JAO DF ----
        jr = _to_float_or_none(r.get('jao_r'))
        jx = _to_float_or_none(r.get('jao_x'))
        jb = _to_float_or_none(r.get('jao_b'))

        jrp = _to_float_or_none(r.get('jao_r_per_km'))
        jxp = _to_float_or_none(r.get('jao_x_per_km'))
        jbp = _to_float_or_none(r.get('jao_b_per_km'))

        # If totals missing but DF has totals, use them
        if (jr is None or jx is None or jb is None) and jid in jao_totals:
            tt = jao_totals[jid]
            jr = jr if jr is not None else _to_float_or_none(tt[0])
            jx = jx if jx is not None else _to_float_or_none(tt[1])
            jb = jb if jb is not None else _to_float_or_none(tt[2])

        # If per-km missing but DF has per-km, use them
        if (jrp is None or jxp is None or jbp is None) and jid in jao_per_km:
            pp = jao_per_km[jid]
            jrp = jrp if jrp is not None else _to_float_or_none(pp[0])
            jxp = jxp if jxp is not None else _to_float_or_none(pp[1])
            jbp = jbp if jbp is not None else _to_float_or_none(pp[2])

        # If only one of totals/per-km is known, derive the other using length
        L = r.get('jao_length_km')
        if L and _to_float_or_none(L) not in (None, 0.0):
            L = _to_float_or_none(L)
            if (jr is None or jx is None or jb is None) and all(v is not None for v in (jrp, jxp, jbp)):
                jr = jrp * L if jr is None else jr
                jx = jxp * L if jx is None else jx
                jb = jbp * L if jb is None else jb
            if (jrp is None or jxp is None or jbp is None) and all(v is not None for v in (jr, jx, jb)):
                if jrp is None: jrp = jr / L
                if jxp is None: jxp = jx / L
                if jbp is None: jbp = jb / L

        # Commit back (only if we have numbers)
        if jr is not None: r['jao_r'] = float(jr)
        if jx is not None: r['jao_x'] = float(jx)
        if jb is not None: r['jao_b'] = float(jb)
        if jrp is not None: r['jao_r_per_km'] = float(jrp)
        if jxp is not None: r['jao_x_per_km'] = float(jxp)
        if jbp is not None: r['jao_b_per_km'] = float(jbp)

        # ---- Coverage backfill (best-effort) ----
        # Prefer 'matched_km' if produced by your allocator; otherwise derive from network_ids lengths
        mk = _to_float_or_none(r.get('matched_km'))
        if mk is None:
            ids = r.get('network_ids') or []
            mk = 0.0
            for nid in ids:
                Lseg = net_len_km.get(str(nid))
                if Lseg is not None:
                    mk += float(Lseg)
            # if no network lengths, leave as None so UI can show 0.00
        r['matched_km'] = mk if mk is not None else 0.0

        L = _to_float_or_none(r.get('jao_length_km'))
        if L and L > 0:
            r['coverage_ratio'] = max(0.0, min(1.0, (r['matched_km'] or 0.0) / L))
        else:
            r['coverage_ratio'] = 0.0

    return matching_results


import math
import pandas as pd

def _nanmean_weighted(values, weights):
    # values/weights are lists; ignore None/nan values
    num = 0.0
    den = 0.0
    for v, w in zip(values, weights):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            continue
        if w is None or w <= 0:
            continue
        num += v * w
        den += w
    return (num / den) if den > 0 else None

# Fallback per-km PER-CIRCUIT from the literature (overhead lines; 50 Hz).
# Units: R [ohm/km], X via L [mH/km], B via C [nF/km] -> S/km.
_LIT = {
    110: {'R_km': 0.109, 'L_mH_km': 1.2, 'C_nF_km': 9.5},
    220: {'R_km': 0.109, 'L_mH_km': 1.0, 'C_nF_km': 11.0},
    380: {'R_km': 0.028, 'L_mH_km': 0.8, 'C_nF_km': 14.0},
    400: {'R_km': 0.028, 'L_mH_km': 0.8, 'C_nF_km': 14.0},  # treat 380/400 the same
}

def _fallback_per_km_pc(v_nom_kv):
    spec = _LIT.get(int(v_nom_kv), None)
    if spec is None:
        return (None, None, None)
    f = 50.0
    R = spec['R_km']
    X = 2 * math.pi * f * (spec['L_mH_km'] * 1e-3)  # mH/km -> H/km
    B = 2 * math.pi * f * (spec['C_nF_km'] * 1e-9)  # nF/km -> S/km
    return (R, X, B)


def prepare_jao_allocation_rows(matching_results, jao_df, network_df):
    """Prepare export rows for JAO allocation to network lines"""
    import pandas as pd
    import numpy as np

    # Convert column types to ensure compatibility
    jao_df = jao_df.copy()
    network_df = network_df.copy()

    jao_df['id'] = jao_df['id'].astype(str)
    network_df['id'] = network_df['id'].astype(str)

    # Initialize empty list for export rows
    export_rows = []

    # Helper function to get per-km values
    def get_per_km_values(row, length_km):
        if length_km <= 0:
            return None, None, None, None

        r = row.get('r', None)
        x = row.get('x', None)
        b = row.get('b', None)
        g = row.get('g', 0.0)

        # Check if values are valid
        r = float(r) if pd.notna(r) else None
        x = float(x) if pd.notna(x) else None
        b = float(b) if pd.notna(b) else None
        g = float(g) if pd.notna(g) else 0.0

        # Calculate per-km values
        r_km = r / length_km if r is not None else None
        x_km = x / length_km if x is not None else None
        b_km = b / length_km if b is not None else None
        g_km = g / length_km if g is not None else 0.0

        return r_km, x_km, b_km, g_km

    # Helper function to get length in km
    def get_length_km(row):
        # First try length_km column
        if 'length_km' in row and pd.notna(row['length_km']):
            return float(row['length_km'])

        # Then try length column (could be in meters)
        if 'length' in row and pd.notna(row['length']):
            length = float(row['length'])
            # If length is large, assume it's in meters
            if length > 1000:
                return length / 1000
            return length

        # Finally try geometry (should have a separate function to calculate length)
        if 'geometry' in row and pd.notna(row['geometry']):
            # This would need a function to calculate length from geometry
            # For now, return 0
            return 0

        return 0

    # Process each matching result
    for result in matching_results:
        if not result.get('matched', False):
            continue

        jao_id = result.get('jao_id')
        network_ids = result.get('network_ids', [])

        if not network_ids:
            continue

        # Get JAO line data
        jao_rows = jao_df[jao_df['id'] == jao_id]
        if jao_rows.empty:
            continue

        jao_row = jao_rows.iloc[0]

        # Get JAO line length
        jao_length_km = get_length_km(jao_row)
        if jao_length_km <= 0:
            continue

        # Get JAO line parameters (per km)
        r_km, x_km, b_km, g_km = get_per_km_values(jao_row, jao_length_km)

        # Calculate lengths of matched network segments
        network_segments = network_df[network_df['id'].isin(network_ids)]
        total_network_length_km = 0

        for _, network_row in network_segments.iterrows():
            network_id = network_row['id']
            network_length_km = get_length_km(network_row)

            if network_length_km <= 0:
                continue

            total_network_length_km += network_length_km

        # Skip if no valid network segments
        if total_network_length_km <= 0:
            continue

        # Allocate parameters to each network segment
        for _, network_row in network_segments.iterrows():
            network_id = network_row['id']
            network_length_km = get_length_km(network_row)

            if network_length_km <= 0:
                continue

            # Calculate segment ratio
            segment_ratio = network_length_km / total_network_length_km

            # Number of circuits
            num_circuits = 1
            for col in ['circuits', 'num_parallel']:
                if col in network_row and pd.notna(network_row[col]):
                    num_circuits = int(network_row[col])
                    break

            # Create export row
            export_row = {
                'jao_id': jao_id,
                'network_id': network_id,
                'seg_len_km': network_length_km,
                'segment_ratio': segment_ratio,
                'jao_length_km': jao_length_km,
                'num_circuits': num_circuits,
                'jao_r_km_pc': r_km,
                'jao_x_km_pc': x_km,
                'jao_b_km_pc': b_km,
                'jao_g_km_pc': g_km,
                'v_nom': jao_row.get('v_nom', 0)
            }

            export_rows.append(export_row)

    return export_rows


def prepare_pypsa_allocation_rows(pypsa_df, network_df, pypsa_matches):
    """Prepare export rows for PyPSA allocation to network lines"""
    import pandas as pd
    import numpy as np

    # Convert column types to ensure compatibility
    pypsa_df = pypsa_df.copy()
    network_df = network_df.copy()

    pypsa_df['id'] = pypsa_df['id'].astype(str)
    network_df['id'] = network_df['id'].astype(str)

    # Initialize empty list for export rows
    export_rows = []

    # Helper function to get per-km values
    def get_per_km_values(row, length_km):
        if length_km <= 0:
            return None, None, None, None

        r = row.get('r', None)
        x = row.get('x', None)
        b = row.get('b', None)
        g = row.get('g', 0.0)

        # Check if values are valid
        r = float(r) if pd.notna(r) else None
        x = float(x) if pd.notna(x) else None
        b = float(b) if pd.notna(b) else None
        g = float(g) if pd.notna(g) else 0.0

        # Calculate per-km values
        r_km = r / length_km if r is not None else None
        x_km = x / length_km if x is not None else None
        b_km = b / length_km if b is not None else None
        g_km = g / length_km if g is not None else 0.0

        return r_km, x_km, b_km, g_km

    # Helper function to get length in km
    def get_length_km(row):
        # First try length_km column
        if 'length_km' in row and pd.notna(row['length_km']):
            return float(row['length_km'])

        # Then try length column (could be in meters)
        if 'length' in row and pd.notna(row['length']):
            length = float(row['length'])
            # If length is large, assume it's in meters
            if length > 1000:
                return length / 1000
            return length

        # Finally try geometry (should have a separate function to calculate length)
        if 'geometry' in row and pd.notna(row['geometry']):
            # This would need a function to calculate length from geometry
            # For now, return 0
            return 0

        return 0

    # Process each PyPSA line
    for pypsa_id, network_ids in pypsa_matches.items():
        # Skip if no network matches
        if not network_ids:
            continue

        # Get PyPSA line data
        pypsa_rows = pypsa_df[pypsa_df['id'] == pypsa_id]
        if pypsa_rows.empty:
            continue

        pypsa_row = pypsa_rows.iloc[0]

        # Get PyPSA line length
        pypsa_length_km = get_length_km(pypsa_row)
        if pypsa_length_km <= 0:
            continue

        # Get PyPSA line parameters (per km)
        r_km, x_km, b_km, g_km = get_per_km_values(pypsa_row, pypsa_length_km)

        # Calculate lengths of matched network segments
        network_segments = network_df[network_df['id'].isin(network_ids)]
        total_network_length_km = 0

        for _, network_row in network_segments.iterrows():
            network_id = network_row['id']
            network_length_km = get_length_km(network_row)

            if network_length_km <= 0:
                continue

            total_network_length_km += network_length_km

        # Skip if no valid network segments
        if total_network_length_km <= 0:
            continue

        # Allocate parameters to each network segment
        for _, network_row in network_segments.iterrows():
            network_id = network_row['id']
            network_length_km = get_length_km(network_row)

            if network_length_km <= 0:
                continue

            # Calculate segment ratio
            segment_ratio = network_length_km / total_network_length_km

            # Number of circuits
            num_circuits = 1
            for col in ['circuits', 'num_parallel']:
                if col in network_row and pd.notna(network_row[col]):
                    num_circuits = int(network_row[col])
                    break

            # Create export row - Using pypsa_ prefix for parameter columns
            export_row = {
                'pypsa_id': pypsa_id,
                'network_id': network_id,
                'seg_len_km': network_length_km,
                'segment_ratio': segment_ratio,
                'pypsa_length_km': pypsa_length_km,
                'num_circuits': num_circuits,
                'pypsa_r_km_pc': r_km,
                'pypsa_x_km_pc': x_km,
                'pypsa_b_km_pc': b_km,
                'pypsa_g_km_pc': g_km,
                'v_nom': pypsa_row.get('voltage', 0)
            }

            export_rows.append(export_row)

    return export_rows


def export_allocation_details_csv(export_rows, out_csv_path, source_type="JAO"):
    """
    Write a 'long' table of the allocation with repeated line ids per matched segment/circuit.
    Adds per-segment per-circuit totals for r/x/b/g.

    Parameters:
    -----------
    export_rows : list
        List of dictionaries with allocation details
    out_csv_path : str
        Path to write the CSV file
    source_type : str
        'JAO' or 'PyPSA' to indicate the source of the allocation
    """
    import pandas as pd

    if not export_rows:
        print(f"Warning: export_rows empty; wrote empty file {out_csv_path}")
        pd.DataFrame().to_csv(out_csv_path, index=False)
        return

    er = pd.DataFrame(export_rows).copy()

    # Set column prefix based on source type
    prefix = source_type.lower()

    # Normalize expected columns
    param_cols = [f"{prefix}_r_km_pc", f"{prefix}_x_km_pc", f"{prefix}_b_km_pc", f"{prefix}_g_km_pc"]
    for c in ["network_id", "seg_len_km"] + param_cols:
        if c not in er.columns:
            er[c] = None

    # Segment totals (per circuit)
    er[f"r_seg_pc"] = er[f"{prefix}_r_km_pc"] * er["seg_len_km"]
    er[f"x_seg_pc"] = er[f"{prefix}_x_km_pc"] * er["seg_len_km"]
    er[f"b_seg_pc"] = er[f"{prefix}_b_km_pc"] * er["seg_len_km"]
    er[f"g_seg_pc"] = er[f"{prefix}_g_km_pc"] * er["seg_len_km"]  # will be NaN if no g provided

    # If g not provided → treat as 0 for totals
    er["g_seg_pc"] = er["g_seg_pc"].fillna(0.0)

    # Keep a tidy set of columns first
    core_cols = [
        "network_id", "seg_len_km",
        f"{prefix}_r_km_pc", f"{prefix}_x_km_pc", f"{prefix}_b_km_pc", f"{prefix}_g_km_pc",
        "r_seg_pc", "x_seg_pc", "b_seg_pc", "g_seg_pc"
    ]
    other_cols = [c for c in er.columns if c not in core_cols]
    er = er[core_cols + other_cols]

    er.to_csv(out_csv_path, index=False)
    print(f"Wrote {source_type} allocation details to: {out_csv_path} ({len(er)} rows)")


def export_updated_network_lines_csv(lines_df, export_rows, out_csv_path, source_type="JAO"):
    """
    Build a new lines CSV with r/x/b replaced by JAO/PyPSA-allocated values.
    - Uses weighted average of per-km (per-circuit) values over all segments that hit a network_id.
    - Converts to *corridor* parameters via num_parallel.
      For n parallel circuits:
        r_km_net = r_km_pc / n
        x_km_net = x_km_pc / n
        b_km_net = b_km_pc * n
    - Totals across the line = per-km_net * line.length
    - If missing, falls back to literature values by voltage (overhead).

    Parameters:
    -----------
    lines_df : DataFrame
        DataFrame with network lines data
    export_rows : list
        List of dictionaries with allocation details
    out_csv_path : str
        Path to write the CSV file
    source_type : str
        'JAO' or 'PyPSA' to indicate the source of the allocation
    """
    import pandas as pd
    import numpy as np

    # Set column prefix based on source type
    prefix = source_type.lower()

    # 1) Gather per-line weighted per-km (per circuit)
    by_line = {}
    for row in export_rows:
        nid = row['network_id']
        if nid not in by_line:
            by_line[nid] = {'len_w': [], 'r_pc': [], 'x_pc': [], 'b_pc': []}
        by_line[nid]['len_w'].append(row['seg_len_km'])
        by_line[nid]['r_pc'].append(row[f'{prefix}_r_km_pc'])
        by_line[nid]['x_pc'].append(row[f'{prefix}_x_km_pc'])
        by_line[nid]['b_pc'].append(row[f'{prefix}_b_km_pc'])

    # Helper function for weighted mean
    def _nanmean_weighted(vals, weights):
        import math
        num = den = 0.0
        for v, w in zip(vals, weights):
            if v is None or (isinstance(v, float) and math.isnan(v)) or w in (None, 0):
                continue
            num += float(v) * float(w)
            den += float(w)
        return (num / den) if den > 0 else None

    # Helper function for fallback values
    def _fallback_per_km_pc(v_nom):
        if v_nom is None:
            return None, None, None

        v = float(v_nom)
        if v >= 380:
            r, x, b = 0.03, 0.30, 3.5e-6
        elif v >= 200:
            r, x, b = 0.08, 0.40, 3.0e-6
        else:
            r, x, b = 0.12, 0.50, 2.5e-6
        return r, x, b

    # 2) Copy the DataFrame and add new columns
    df_new = lines_df.copy()
    df_new['r_alloc'] = pd.NA
    df_new['x_alloc'] = pd.NA
    df_new['b_alloc'] = pd.NA
    df_new['r_km_alloc'] = pd.NA
    df_new['x_km_alloc'] = pd.NA
    df_new['b_km_alloc'] = pd.NA
    df_new['alloc_source'] = pd.NA  # 'JAO', 'PyPSA', 'fallback' or 'original'

    # Check for and handle duplicates in the index
    if 'id' in df_new.columns:
        # Check for duplicates before setting index
        dup_count = df_new['id'].duplicated().sum()
        if dup_count > 0:
            print(f"Warning: {dup_count} duplicate IDs; keeping first")
            df_new = df_new.drop_duplicates('id', keep='first')

        # Set index but keep the 'id' column
        df_new = df_new.set_index('id', drop=False)

    # 3) Fill from weighted averages (or fallback)
    for nid, pack in by_line.items():
        # Skip if this network ID is not in the DataFrame
        if nid not in df_new.index:
            continue

        # Get values safely - handle as single row
        try:
            # Try to get values directly
            v_nom = df_new.loc[nid, 'v_nom']
            if isinstance(v_nom, pd.Series):
                v_nom = v_nom.iloc[0]  # Take first if it's a Series

            # Same for num_parallel
            npar_val = df_new.loc[nid, 'num_parallel'] if 'num_parallel' in df_new.columns else None
            if isinstance(npar_val, pd.Series):
                npar_val = npar_val.iloc[0]

            npar = int(float(npar_val)) if pd.notna(npar_val) else 1
            npar = max(npar, 1)

            # Same for length
            length_val = df_new.loc[nid, 'length'] if 'length' in df_new.columns else None
            if isinstance(length_val, pd.Series):
                length_val = length_val.iloc[0]

            length = float(length_val) if pd.notna(length_val) else 0.0

        except Exception as e:
            print(f"Error processing network ID {nid}: {e}")
            continue

        # Weighted means of PER-CIRCUIT per-km values
        r_pc = _nanmean_weighted(pack['r_pc'], pack['len_w'])
        x_pc = _nanmean_weighted(pack['x_pc'], pack['len_w'])
        b_pc = _nanmean_weighted(pack['b_pc'], pack['len_w'])

        # Fallback if any is missing
        src = source_type
        if (r_pc is None) or (x_pc is None) or (b_pc is None):
            fr, fx, fb = _fallback_per_km_pc(v_nom)
            r_pc = r_pc if r_pc is not None else fr
            x_pc = x_pc if x_pc is not None else fx
            b_pc = b_pc if b_pc is not None else fb
            src = 'fallback' if src != source_type else (f'{source_type}+fallback' if (fr or fx or fb) else source_type)

        if (r_pc is None) or (x_pc is None) or (b_pc is None) or length <= 0:
            # can't update this one, leave original
            # Handle the case where 'alloc_source' assignment might also return a Series
            try:
                if isinstance(df_new.loc[nid, 'alloc_source'], pd.Series):
                    df_new.loc[nid, 'alloc_source'].iloc[0] = 'original'
                else:
                    df_new.loc[nid, 'alloc_source'] = 'original'
            except Exception as e:
                print(f"Error setting alloc_source for {nid}: {e}")
            continue

        # Convert PER-CIRCUIT -> CORRIDOR equivalents
        r_km_net = r_pc / npar
        x_km_net = x_pc / npar
        b_km_net = b_pc * npar

        # Totals over the line
        r_tot = r_km_net * length
        x_tot = x_km_net * length
        b_tot = b_km_net * length

        # Update values - handle Series case
        try:
            # Safely set values, handling the case where a Series might be returned
            for col, val in [
                ('r_alloc', r_tot), ('x_alloc', x_tot), ('b_alloc', b_tot),
                ('r_km_alloc', r_km_net), ('x_km_alloc', x_km_net), ('b_km_alloc', b_km_net),
                ('alloc_source', src)
            ]:
                if isinstance(df_new.loc[nid, col], pd.Series):
                    df_new.loc[nid, col].iloc[0] = val
                else:
                    df_new.loc[nid, col] = val

        except Exception as e:
            print(f"Error updating values for {nid}: {e}")

    # 4) Overwrite r/x/b with allocated where available (keep originals otherwise)
    for col_src, col_dst in [('r_alloc', 'r'), ('x_alloc', 'x'), ('b_alloc', 'b')]:
        mask = df_new[col_src].notna()
        df_new.loc[mask, col_dst] = df_new.loc[mask, col_src].astype(float)

    # 5) Write to CSV
    df_new.reset_index(drop=True).to_csv(out_csv_path, index=False)
    print(f"Wrote updated lines with {source_type} allocations to: {out_csv_path}")


def export_ready_lines_csv(lines_df, export_rows, out_csv_path, source_type="JAO",
                           use_fallback=True, style="corridor_like_original",
                           set_num_parallel_to_one=False, update_per_km_cols=True):
    """
    Export lines CSV ready for use with either JAO or PyPSA parameters.

    Parameters:
    -----------
    lines_df : DataFrame
        DataFrame with lines data (JAO or PyPSA)
    export_rows : list
        List of dictionaries with allocation details
    out_csv_path : str
        Path to write the CSV file
    source_type : str
        'JAO' or 'PyPSA' to indicate the source of the allocation
    use_fallback : bool
        Whether to use fallback values when parameters are missing
    style : str
        'per_circuit' : write per-circuit totals (let PyPSA scale with num_parallel)
        'corridor_like_original' : write corridor totals (match your original numbers)
    set_num_parallel_to_one : bool
        If True and style='corridor_like_original', set num_parallel=1 to avoid double-scaling in PyPSA
    update_per_km_cols : bool
        If True, update r_per_km/x_per_km/b_per_km to be consistent with what's written
    """
    import pandas as pd
    import numpy as np

    # Set column prefix based on source type
    prefix = source_type.lower()

    df_new = lines_df.copy()
    df_new = df_new.drop(columns=["geometry"], errors="ignore")
    original_cols = list(df_new.columns)
    df_new['id'] = df_new['id'].astype(str)

    # Handle duplicates
    if df_new['id'].duplicated().any():
        dup_count = df_new['id'].duplicated().sum()
        print(f"Warning: {dup_count} duplicate IDs; keeping first")
        df_new = df_new.drop_duplicates('id', keep='first')

    df_new = df_new.set_index("id", drop=False)

    # bucket segments by network_id
    by_line = {}
    for row in export_rows or []:
        nid = str(row.get("network_id"))
        by_line.setdefault(nid, {"w": [], "r": [], "x": [], "b": [], "g": []})
        by_line[nid]["w"].append(row.get("seg_len_km"))
        by_line[nid]["r"].append(row.get(f"{prefix}_r_km_pc"))
        by_line[nid]["x"].append(row.get(f"{prefix}_x_km_pc"))
        by_line[nid]["b"].append(row.get(f"{prefix}_b_km_pc"))
        by_line[nid]["g"].append(row.get(f"{prefix}_g_km_pc", None))

    def _nanmean_weighted(vals, weights):
        import math
        num = den = 0.0
        for v, w in zip(vals, weights):
            if v is None or (isinstance(v, float) and math.isnan(v)) or w in (None, 0):
                continue
            num += float(v) * float(w)
            den += float(w)
        return (num / den) if den > 0 else None

    def _fallback_per_km_pc(v_nom_kv):
        if v_nom_kv is None: return None, None, None, 0.0
        v = float(v_nom_kv)
        if v >= 380:
            r, x, b = 0.03, 0.30, 3.5e-6
        elif v >= 200:
            r, x, b = 0.08, 0.40, 3.0e-6
        else:
            r, x, b = 0.12, 0.50, 2.5e-6
        return r, x, b, 0.0

    for c in ["r_alloc", "x_alloc", "b_alloc", "g_alloc",
              "r_km_alloc", "x_km_alloc", "b_km_alloc", "g_km_alloc",
              "alloc_source"]:
        df_new[c] = pd.NA

    for nid, pack in by_line.items():
        if nid not in df_new.index:
            continue

        # Get values safely - handle case where it might return a Series
        try:
            # Base data - handle Series
            v_nom_val = df_new.loc[nid, "v_nom"] if "v_nom" in df_new.columns else None
            if isinstance(v_nom_val, pd.Series):
                v_nom_val = v_nom_val.iloc[0]
            v_nom = float(v_nom_val) if pd.notna(v_nom_val) else None

            length_val = df_new.loc[nid, "length"] if "length" in df_new.columns else None
            if isinstance(length_val, pd.Series):
                length_val = length_val.iloc[0]
            length = float(length_val) if pd.notna(length_val) else 0.0
        except Exception as e:
            print(f"Error getting base data for network ID {nid}: {e}")
            continue

        # num_parallel/circuits - check multiple columns and handle Series
        npar = 1
        for cand in ["circuits", "num_parallel", "parallel_num", "n_circuits", "parallels", "nparallel"]:
            if cand in df_new.columns:
                try:
                    val = df_new.loc[nid, cand]
                    if isinstance(val, pd.Series):
                        val = val.iloc[0]

                    if pd.notna(val):
                        npar = max(1, int(round(float(val))))
                        break
                except Exception:
                    pass

        # per-circuit per-km from source
        r_pc = _nanmean_weighted(pack["r"], pack["w"])
        x_pc = _nanmean_weighted(pack["x"], pack["w"])
        b_pc = _nanmean_weighted(pack["b"], pack["w"])
        g_pc = _nanmean_weighted(pack["g"], pack["w"])  # may be None
        src = source_type
        if (r_pc is None) or (x_pc is None) or (b_pc is None) or (g_pc is None):
            fr, fx, fb, fg = _fallback_per_km_pc(v_nom) if use_fallback else (None, None, None, None)
            if r_pc is None: r_pc = fr
            if x_pc is None: x_pc = fx
            if b_pc is None: b_pc = fb
            if g_pc is None: g_pc = fg
            src = f"{source_type}+fallback"

        if (r_pc is None) or (x_pc is None) or (b_pc is None) or length <= 0:
            # Handle assigning to potentially Series values
            try:
                if isinstance(df_new.loc[nid, "alloc_source"], pd.Series):
                    df_new.loc[nid, "alloc_source"].iloc[0] = "original"
                else:
                    df_new.loc[nid, "alloc_source"] = "original"
            except Exception as e:
                print(f"Error setting alloc_source for {nid}: {e}")
            continue

        # choose what to WRITE
        if style == "per_circuit":
            # write per-circuit totals
            r_write = r_pc * length
            x_write = x_pc * length
            b_write = b_pc * length
            g_write = (g_pc or 0.0) * length
            r_km_write, x_km_write, b_km_write, g_km_write = r_pc, x_pc, b_pc, (g_pc or 0.0)

        elif style == "corridor_like_original":
            # write corridor totals (divide series by n, multiply shunt by n)
            r_km_net = r_pc / npar
            x_km_net = x_pc / npar
            b_km_net = b_pc * npar
            g_km_net = (g_pc or 0.0) * npar

            r_write = r_km_net * length
            x_write = x_km_net * length
            b_write = b_km_net * length
            g_write = g_km_net * length

            r_km_write, x_km_write, b_km_write, g_km_write = r_km_net, x_km_net, b_km_net, g_km_net
        else:
            raise ValueError("style must be 'per_circuit' or 'corridor_like_original'")

        # assign - handle Series
        try:
            for col, val in [
                ("r_alloc", r_write), ("x_alloc", x_write),
                ("b_alloc", b_write), ("g_alloc", g_write),
                ("r_km_alloc", r_km_write), ("x_km_alloc", x_km_write),
                ("b_km_alloc", b_km_write), ("g_km_alloc", g_km_write),
                ("alloc_source", src)
            ]:
                if isinstance(df_new.loc[nid, col], pd.Series):
                    df_new.loc[nid, col].iloc[0] = val
                else:
                    df_new.loc[nid, col] = val
        except Exception as e:
            print(f"Error setting values for {nid}: {e}")
            continue

        # optionally update *_per_km columns if they exist
        if update_per_km_cols:
            try:
                for col, val in [
                    ("r_per_km", r_km_write),
                    ("x_per_km", x_km_write),
                    ("b_per_km", b_km_write)
                ]:
                    if col in df_new.columns:
                        if isinstance(df_new.loc[nid, col], pd.Series):
                            df_new.loc[nid, col].iloc[0] = val
                        else:
                            df_new.loc[nid, col] = val
            except Exception as e:
                print(f"Error updating per_km columns for {nid}: {e}")

        # optionally set num_parallel=1 when writing corridor totals (avoid double scaling)
        if style == "corridor_like_original" and set_num_parallel_to_one:
            try:
                for cand in ["num_parallel", "circuits", "parallel_num", "n_circuits", "parallels", "nparallel"]:
                    if cand in df_new.columns:
                        if isinstance(df_new.loc[nid, cand], pd.Series):
                            df_new.loc[nid, cand].iloc[0] = 1
                        else:
                            df_new.loc[nid, cand] = 1
            except Exception as e:
                print(f"Error setting num_parallel to 1 for {nid}: {e}")

    # overwrite only r/x/b/g
    for s, d in [("r_alloc", "r"), ("x_alloc", "x"), ("b_alloc", "b"), ("g_alloc", "g")]:
        if d not in df_new.columns: df_new[d] = pd.NA
        m = df_new[s].notna()
        df_new.loc[m, d] = df_new.loc[m, s].astype(float)

    out_df = df_new[original_cols].reset_index(drop=True)
    out_df.to_csv(out_csv_path, index=False)
    print(f"Wrote {source_type}-ready lines to: {out_csv_path} "
          f"(style={style}, set_num_parallel_to_one={set_num_parallel_to_one})")

def generate_comparison_files(jao_df, pypsa_df, network_df, matching_results, pypsa_matches, output_dir="output"):
    """
    Generate all necessary files for parameter comparison.

    Parameters:
    -----------
    jao_df : DataFrame
        DataFrame with JAO lines data
    pypsa_df : DataFrame
        DataFrame with PyPSA lines data
    network_df : DataFrame
        DataFrame with network lines data
    matching_results : list
        List of dictionaries with JAO-Network matching results
    pypsa_matches : dict
        Dictionary mapping PyPSA IDs to matched network IDs
    output_dir : str
        Directory to write output files
    """
    import os
    import pandas as pd

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # 1. Prepare allocation rows for JAO
    print("Preparing JAO allocation rows...")
    jao_allocation_rows = prepare_jao_allocation_rows(matching_results, jao_df, network_df)

    # 2. Prepare allocation rows for PyPSA
    print("Preparing PyPSA allocation rows...")
    pypsa_allocation_rows = prepare_pypsa_allocation_rows(pypsa_df, network_df, pypsa_matches)

    # 3. Export allocation details
    print("Exporting allocation details...")
    jao_allocation_path = os.path.join(output_dir, "jao_allocation_details.csv")
    pypsa_allocation_path = os.path.join(output_dir, "pypsa_allocation_details.csv")

    export_allocation_details_csv(jao_allocation_rows, jao_allocation_path, source_type="JAO")
    export_allocation_details_csv(pypsa_allocation_rows, pypsa_allocation_path, source_type="PyPSA")

    # 4. Export updated network lines
    print("Exporting updated network lines...")
    jao_network_path = os.path.join(output_dir, "network_lines_jao.csv")
    pypsa_network_path = os.path.join(output_dir, "network_lines_pypsa.csv")

    export_updated_network_lines_csv(network_df, jao_allocation_rows, jao_network_path, source_type="JAO")
    export_updated_network_lines_csv(network_df, pypsa_allocation_rows, pypsa_network_path, source_type="PyPSA")

    # 5. Export ready lines for JAO and PyPSA
    print("Exporting ready lines...")
    jao_ready_path = os.path.join(output_dir, "jao_lines_ready.csv")
    pypsa_ready_path = os.path.join(output_dir, "pypsa_lines_ready.csv")

    export_ready_lines_csv(jao_df, jao_allocation_rows, jao_ready_path, source_type="JAO")
    export_ready_lines_csv(pypsa_df, pypsa_allocation_rows, pypsa_ready_path, source_type="PyPSA")

    print(f"\nAll files exported to {output_dir}:")
    print(f"  - {jao_allocation_path}")
    print(f"  - {pypsa_allocation_path}")
    print(f"  - {jao_network_path}")
    print(f"  - {pypsa_network_path}")
    print(f"  - {jao_ready_path}")
    print(f"  - {pypsa_ready_path}")

    return {
        "jao_allocation": jao_allocation_path,
        "pypsa_allocation": pypsa_allocation_path,
        "network_jao": jao_network_path,
        "network_pypsa": pypsa_network_path,
        "jao_ready": jao_ready_path,
        "pypsa_ready": pypsa_ready_path
    }


def match_parallel_circuits_robustly(matching_results, jao_gdf, network_gdf):
    """
    Robust approach to match parallel circuits properly.
    """
    import numpy as np  # Import numpy here to fix the referenced before assignment error

    print("\n=== ROBUST PARALLEL CIRCUIT MATCHING ===")

    # 1. Build comprehensive tracking of matches and parallel circuits

    # Track which JAO lines are matched
    matched_jao_ids = set()
    jao_to_network = {}  # Maps JAO ID to list of network IDs it matched with
    network_to_jao = {}  # Maps network ID to list of JAO IDs it matched with
    jao_to_result = {}  # Maps JAO ID to its match result

    for result in matching_results:
        if result.get('matched', False) and result.get('network_ids'):
            jao_id = result['jao_id']
            matched_jao_ids.add(jao_id)
            jao_to_result[jao_id] = result

            # Track which network lines this JAO matched with
            jao_to_network[jao_id] = result.get('network_ids', [])

            # Track which JAO lines each network line matched with
            for network_id in result.get('network_ids', []):
                network_to_jao.setdefault(network_id, []).append(jao_id)

    # Helper function for calculating parallel score with proper error handling
    def calculate_robust_parallel_score(geom1, geom2, buffer_meters=300):
        """Calculate how parallel two geometries are with proper error handling."""
        try:
            # Calculate buffer based on latitude
            avg_lat = (geom1.centroid.y + geom2.centroid.y) / 2
            meters_per_degree = 111111 * np.cos(np.radians(abs(avg_lat)))
            buffer_deg = buffer_meters / meters_per_degree

            # Create buffers
            buffer1 = geom1.buffer(buffer_deg)
            buffer2 = geom2.buffer(buffer_deg)

            # Calculate mutual overlap
            overlap1 = geom1.intersection(buffer2)
            overlap2 = geom2.intersection(buffer1)

            # Calculate overlap ratios
            ratio1 = overlap1.length / geom1.length if geom1.length > 0 else 0
            ratio2 = overlap2.length / geom2.length if geom2.length > 0 else 0

            # Average ratio (higher is better)
            avg_ratio = (ratio1 + ratio2) / 2

            # Calculate Hausdorff distance (lower is better)
            hausdorff_dist = geom1.hausdorff_distance(geom2)
            hausdorff_meters = hausdorff_dist * meters_per_degree

            # Normalize Hausdorff (1 when distance is 0, 0 when distance is large)
            norm_hausdorff = max(0, 1 - (hausdorff_meters / 1000))

            # Calculate direction alignment
            # Get unit vectors from start to end for both geometries
            def get_vector(geom):
                if geom.geom_type == 'MultiLineString':
                    # Use the longest component
                    geom = max(geom.geoms, key=lambda g: g.length)

                coords = np.array(list(geom.coords))
                vec = coords[-1] - coords[0]
                norm = np.linalg.norm(vec)
                return vec / norm if norm > 0 else vec

            try:
                vec1 = get_vector(geom1)
                vec2 = get_vector(geom2)
                alignment = abs(np.dot(vec1, vec2))
            except Exception:
                # If direction calculation fails, assume moderate alignment
                alignment = 0.7

            # Combined score (weighted average)
            score = 0.5 * avg_ratio + 0.3 * norm_hausdorff + 0.2 * alignment

            return score
        except Exception as e:
            print(f"Error calculating parallel score: {e}")
            return 0

    # 2. Find JAO lines with similar geometries (potential parallel circuits)
    print("Identifying parallel JAO geometries...")

    # Group JAO lines by geometry similarity
    jao_groups = []
    processed_jao_ids = set()

    for idx1, row1 in jao_gdf.iterrows():
        jao_id1 = str(row1['id'])

        # Skip if already processed
        if jao_id1 in processed_jao_ids:
            continue

        geom1 = row1.geometry
        voltage1 = int(row1['v_nom'])

        # Start a new group with this JAO
        current_group = [jao_id1]
        processed_jao_ids.add(jao_id1)

        # Compare with all other JAO lines
        for idx2, row2 in jao_gdf.iterrows():
            jao_id2 = str(row2['id'])

            # Skip if already processed or same line
            if jao_id2 in processed_jao_ids or jao_id1 == jao_id2:
                continue

            geom2 = row2.geometry
            voltage2 = int(row2['v_nom'])

            # Only group lines with same voltage
            if voltage1 != voltage2:
                continue

            # Calculate geometry similarity
            similarity = calculate_robust_parallel_score(geom1, geom2, buffer_meters=300)

            # If geometries are similar enough, add to group
            if similarity > 0.8:  # High threshold for being considered parallel
                current_group.append(jao_id2)
                processed_jao_ids.add(jao_id2)

        # Only keep groups with multiple JAO lines
        if len(current_group) > 1:
            jao_groups.append({
                'jao_ids': current_group,
                'voltage': voltage1,
                'geometry': geom1  # Use first JAO's geometry as reference
            })

    print(f"Found {len(jao_groups)} parallel JAO geometry groups")

    # 3. Find network lines with num_parallel > 1
    print("Identifying network lines with multiple circuits...")

    parallel_network_lines = {}

    for idx, row in network_gdf.iterrows():
        network_id = str(row['id'])
        num_parallel = int(row.get('num_parallel', 1)) if pd.notna(row.get('num_parallel')) else 1
        num_parallel = max(num_parallel, 1)

        # Special handling for specific lines known to have parallel circuits
        if network_id == 'Line_17046':
            num_parallel = max(num_parallel, 2)  # Ensure this line has at least 2 circuits

        if num_parallel > 1:
            parallel_network_lines[network_id] = {
                'num_parallel': num_parallel,
                'geometry': row.geometry,
                'voltage': _safe_int(row['v_nom']),
                'length': calculate_length_meters(row.geometry),
                'matched_jao_count': len(network_to_jao.get(network_id, []))
            }

    print(f"Found {len(parallel_network_lines)} network lines with multiple circuits")

    # 4. Part A: Ensure network lines with num_parallel > 1 match with the right number of JAO lines
    print("\nFilling in missing parallel circuit matches...")

    new_matches = []

    for network_id, info in parallel_network_lines.items():
        num_parallel = info['num_parallel']
        matched_count = info['matched_jao_count']
        network_voltage = info['voltage']
        network_geom = info['geometry']

        # Skip if already matched to enough or too many JAO lines
        if matched_count >= num_parallel:
            continue

        print(f"  Network line {network_id} has {num_parallel} circuits but only matched {matched_count} JAO lines")

        # Get already matched JAO IDs for this network line
        matched_jao_ids_for_network = set(network_to_jao.get(network_id, []))

        # See if any of these JAO IDs are in a parallel group
        candidate_jao_ids = set()

        # First try to find from parallel JAO groups
        for matched_jao_id in matched_jao_ids_for_network:
            for group in jao_groups:
                if matched_jao_id in group['jao_ids']:
                    # Add other JAOs from the same group that aren't already matched to this network
                    for jao_id in group['jao_ids']:
                        if jao_id not in matched_jao_ids_for_network:
                            candidate_jao_ids.add(jao_id)

        # If we don't have enough candidates, find unmatched JAO lines with similar geometry
        additional_needed = num_parallel - matched_count - len(candidate_jao_ids)

        if additional_needed > 0:
            unmatched_jao_ids = set(str(row['id']) for _, row in jao_gdf.iterrows()) - matched_jao_ids

            # Find JAO lines with similar geometry and voltage
            for jao_id in unmatched_jao_ids:
                if jao_id in candidate_jao_ids:
                    continue  # Skip if already a candidate

                jao_rows = jao_gdf[jao_gdf['id'].astype(str) == jao_id]
                if jao_rows.empty:
                    continue

                jao_row = jao_rows.iloc[0]
                jao_geom = jao_row.geometry
                jao_voltage = int(jao_row['v_nom'])

                # Check voltage compatibility
                voltage_match = _same_voltage(network_voltage, jao_voltage)


                if not voltage_match:
                    continue

                # Calculate geometry similarity
                similarity = calculate_robust_parallel_score(network_geom, jao_geom, buffer_meters=500)

                if similarity > 0.7:  # Lower threshold for matching unmatched JAOs
                    candidate_jao_ids.add(jao_id)
                    if len(candidate_jao_ids) >= num_parallel - matched_count:
                        break

        # Create matches for candidate JAO IDs
        for jao_id in candidate_jao_ids:
            # Check if JAO already has a match
            existing_match = jao_to_result.get(jao_id)

            if existing_match:
                # Add this network line to existing match if not already there
                if 'network_ids' not in existing_match:
                    existing_match['network_ids'] = []

                if network_id not in existing_match['network_ids']:
                    print(f"    Adding network line {network_id} to existing match for JAO {jao_id}")
                    existing_match['network_ids'].append(network_id)

                    # Update path length
                    if 'path_length' not in existing_match:
                        existing_match['path_length'] = 0
                    existing_match['path_length'] = float(existing_match['path_length'] + info['length'])

                    # Update length ratio
                    if 'jao_length' in existing_match and existing_match['jao_length'] > 0:
                        existing_match['length_ratio'] = float(
                            existing_match['path_length'] / existing_match['jao_length'])

                    # Mark as parallel circuit if not already
                    if not existing_match.get('is_parallel_circuit', False):
                        existing_match['is_parallel_circuit'] = True
                        existing_match['match_quality'] = f'Parallel Circuit ({network_voltage} kV) - Robust Matching'

                    # Update tracking
                    network_to_jao.setdefault(network_id, []).append(jao_id)
                    jao_to_network.setdefault(jao_id, []).append(network_id)
            else:
                # Create new match
                jao_rows = jao_gdf[jao_gdf['id'].astype(str) == jao_id]
                if jao_rows.empty:
                    continue

                jao_row = jao_rows.iloc[0]
                jao_length = calculate_length_meters(jao_row.geometry)

                new_match = {
                    'jao_id': jao_id,
                    'jao_name': str(jao_row.get('NE_name', '')),
                    'v_nom': int(jao_row['v_nom']),
                    'matched': True,
                    'is_duplicate': False,
                    'is_parallel_circuit': True,
                    'network_ids': [network_id],
                    'path_length': float(info['length']),
                    'jao_length': float(jao_length),
                    'length_ratio': float(info['length'] / jao_length) if jao_length > 0 else 1.0,
                    'match_quality': f'Parallel Circuit ({network_voltage} kV) - Robust Matching'
                }

                print(f"    Creating new match for JAO {jao_id} with network line {network_id}")
                new_matches.append(new_match)

                # Update tracking
                matched_jao_ids.add(jao_id)
                jao_to_result[jao_id] = new_match
                network_to_jao.setdefault(network_id, []).append(jao_id)
                jao_to_network[jao_id] = [network_id]

    # 5. Part B: Direct handling of specific cases
    print("\nHandling specific known parallel circuit cases...")

    special_cases = {
        "Line_17046": ["2282", "2283"]
    }

    for network_id, jao_ids in special_cases.items():
        print(f"  Processing special case: network line {network_id} should match JAO lines {jao_ids}")

        # Get network line details
        network_rows = network_gdf[network_gdf['id'].astype(str) == network_id]
        if network_rows.empty:
            print(f"    Network line {network_id} not found")
            continue

        network_row = network_rows.iloc[0]
        network_voltage = int(network_row['v_nom'])
        network_length = calculate_length_meters(network_row.geometry)

        # Check each JAO to see if it's already matched with this network line
        for jao_id in jao_ids:
            current_network_ids = jao_to_network.get(jao_id, [])

            # If this network ID is not already matched to this JAO, add it
            if network_id not in current_network_ids:
                # Get the JAO match result or create one
                match_result = jao_to_result.get(jao_id)

                if match_result:
                    # Add this network line to the existing match
                    if 'network_ids' not in match_result:
                        match_result['network_ids'] = []

                    print(f"    Adding network line {network_id} to existing match for JAO {jao_id}")
                    ok = _try_append_with_path_lock(match_result, network_id, network_gdf,
                                                    max_offcorridor_m=300, max_len_ratio_after=1.30)
                    if ok is None:  # not path-based → keep old behavior
                        match_result.setdefault('network_ids', []).append(network_id)
                        _normalize_network_ids_and_path_length(match_result, network_gdf)
                    elif ok is False:
                        continue

                    # Update path length
                    if 'path_length' not in match_result:
                        match_result['path_length'] = 0
                    match_result['path_length'] = float(match_result['path_length'] + network_length)

                    # Update length ratio
                    if 'jao_length' in match_result and match_result['jao_length'] > 0:
                        match_result['length_ratio'] = float(match_result['path_length'] / match_result['jao_length'])

                    # Mark as parallel circuit if not already
                    if not match_result.get('is_parallel_circuit', False):
                        match_result['is_parallel_circuit'] = True
                        match_result['match_quality'] = f'Parallel Circuit ({network_voltage} kV) - Special Case'
                else:
                    # Create a new match
                    jao_rows = jao_gdf[jao_gdf['id'].astype(str) == jao_id]
                    if jao_rows.empty:
                        print(f"    JAO {jao_id} not found")
                        continue

                    jao_row = jao_rows.iloc[0]
                    jao_length = calculate_length_meters(jao_row.geometry)

                    new_match = {
                        'jao_id': jao_id,
                        'jao_name': str(jao_row.get('NE_name', '')),
                        'v_nom': int(jao_row['v_nom']),
                        'matched': True,
                        'is_duplicate': False,
                        'is_parallel_circuit': True,
                        'network_ids': [network_id],
                        'path_length': float(network_length),
                        'jao_length': float(jao_length),
                        'length_ratio': float(network_length / jao_length) if jao_length > 0 else 1.0,
                        'match_quality': f'Parallel Circuit ({jao_row["v_nom"]} kV) - Special Case'
                    }

                    print(f"    Creating new match for JAO {jao_id} with network line {network_id}")
                    new_matches.append(new_match)

                    # Update tracking
                    matched_jao_ids.add(jao_id)
                    jao_to_result[jao_id] = new_match
                    jao_to_network[jao_id] = [network_id]

                # Update network_to_jao tracking
                network_to_jao.setdefault(network_id, []).append(jao_id)

    # Add new matches to results
    matching_results.extend(new_matches)
    print(f"\nAdded {len(new_matches)} new matches through robust parallel circuit matching")

    return matching_results

def _ids_from_shortest_path(G, start_node, end_node):
    import networkx as nx
    try:
        path = nx.shortest_path(G, start_node, end_node, weight='weight')
    except nx.NetworkXNoPath:
        return [], []
    edge_ids = []
    edges = []
    for u, v in zip(path[:-1], path[1:]):
        data = G.get_edge_data(u, v)
        if not data:
            continue
        # If multi-edge, pick the one with an 'id' and not a connector
        if isinstance(data, dict) and 0 in data and 'id' not in data:
            # MultiGraph style
            best = None
            for k, ed in data.items():
                if ed.get('connector'):
                    continue
                if 'id' in ed:
                    best = ed
                    break
            if best:
                edges.append((u, v, best))
                edge_ids.append(str(best['id']))
        else:
            # Single edge
            if data.get('connector'):
                continue
            if 'id' in data:
                edges.append((u, v, data))
                edge_ids.append(str(data['id']))
    return edge_ids, edges


def _near_enough(g_line, g_ref, max_off_m=800):
    # keep only parallel/duplicate lines that sit near the chosen path
    try:
        return calculate_geometry_coverage(g_ref, g_line, buffer_meters=max_off_m) > 0.6
    except Exception:
        return False


def trim_to_shortest_paths(matching_results, G, jao_gdf, network_gdf, nearest_points_dict,
                           keep_parallel=True, max_offcorridor_m=800, max_len_ratio_after=1.35):
    """
    For each matched JAO:
      1) compute graph shortest path between its start/end nearest nodes
      2) keep ONLY edges that lie on that path
      3) optionally keep true parallel duplicates that sit near the path
      4) recompute lengths/ratio
    """
    import numpy as np

    # index JAO rows once
    jao_index_by_id = {str(r['id']): i for i, r in jao_gdf.reset_index().iterrows()}

    trimmed = 0
    for r in matching_results:
        if not r.get('matched') or not r.get('network_ids'):
            continue

        jao_id = str(r['jao_id'])
        if jao_id not in jao_index_by_id:
            continue
        jidx = jao_index_by_id[jao_id]

        # must have both endpoints
        np_info = nearest_points_dict.get(jidx)
        if not np_info or not np_info.get('start_nearest') or not np_info.get('end_nearest'):
            continue

        s_idx, s_pos = np_info['start_nearest']
        e_idx, e_pos = np_info['end_nearest']
        start_node = f"node_{s_idx}_{s_pos}"
        end_node   = f"node_{e_idx}_{e_pos}"

        # 1) shortest path → network ids
        path_ids, _ = _ids_from_shortest_path(G, start_node, end_node)
        if not path_ids:
            continue

        keep = set(path_ids)

        # 2) optionally retain *true* parallels that hug the chosen path
        if keep_parallel:
            # build a union/path geometry from kept segments to test proximity
            kept_geoms = []
            for nid in keep:
                rows = network_gdf[network_gdf['id'].astype(str) == str(nid)]
                if not rows.empty:
                    kept_geoms.append(rows.iloc[0].geometry)
            if kept_geoms:
                from shapely.ops import unary_union
                path_geom = unary_union(kept_geoms)
                for nid in r['network_ids']:
                    if nid in keep:
                        continue
                    rows = network_gdf[network_gdf['id'].astype(str) == str(nid)]
                    if rows.empty:
                        continue
                    cand = rows.iloc[0].geometry
                    if _near_enough(cand, path_geom, max_off_m=max_offcorridor_m):
                        keep.add(str(nid))

        # 3) apply trimming only if we actually drop something or if ratio is too big
        if (set(r['network_ids']) - keep) or (r.get('jao_length', 0) and r.get('path_length', 0) / r['jao_length'] > max_len_ratio_after):
            r['network_ids'] = list(dict.fromkeys([nid for nid in r['network_ids'] if str(nid) in keep]))
            # recompute path_length and ratio
            _normalize_network_ids_and_path_length(r, network_gdf)
            if r.get('jao_length'):
                r['length_ratio'] = float(r['path_length'] / float(r['jao_length']))
            # annotate that this is path-based and cleaned
            r['is_geometric_match'] = False
            r['match_quality'] = (r.get('match_quality') or '') + ' | Trimmed to Shortest Path'
            trimmed += 1

    print(f"Trimmed {trimmed} JAO matches to a single shortest path")
    return matching_results


def share_network_lines_among_parallel_jaos(matching_results, jao_gdf,
                                            *,
                                            buffer_m=800,
                                            min_avg_coverage=0.60,
                                            max_hausdorff_m=1000.0,
                                            verbose=True):
    """
    Share network_ids among JAO lines that are effectively parallel segments of the
    same corridor (same voltage + high mutual coverage + small Hausdorff distance).

    - Uses a Shapely STRtree to avoid O(n^2) pairwise loops.
    - Respects path-based matches: if result.get('is_path_based') is True, we do NOT
      append any new network_ids to that result (soft lock).
    - Does not require network_gdf; it shares IDs only (lengths/ratios are NOT recomputed here).

    Parameters
    ----------
    matching_results : list[dict]
        Your per-JAO matching results (each dict is a result row).
    jao_gdf : GeoDataFrame
        Must have columns: 'id', 'v_nom', 'geometry'.
    buffer_m : float
        Corridor half-width (meters) used to preselect candidate parallels quickly.
    min_avg_coverage : float
        Threshold on average mutual coverage (0..1) to consider two JAOs parallel.
        We compute coverage in both directions with `calculate_geometry_coverage(...)`
        and take the average.
    max_hausdorff_m : float
        Maximum Hausdorff distance (meters) for the pair to be considered parallel.
    verbose : bool
        Print progress.

    Returns
    -------
    list[dict] : the same matching_results list (modified in place) is returned for convenience.
    """
    import numpy as np
    from shapely.strtree import STRtree

    if verbose:
        print("\n=== SHARING NETWORK LINES AMONG PARALLEL JAO LINES (STRtree) ===")

    # ---- fast lookups between results and JAO rows --------------------------
    # map jao_id -> result dict (only those that are "matched" and not duplicates)
    result_by_jao = {}
    for r in matching_results:
        if not r.get('matched'):
            continue
        if r.get('is_duplicate', False):
            continue
        jid = str(r.get('jao_id'))
        if jid:
            result_by_jao[jid] = r

    # quick access to geometry & voltage by JAO id
    geom_by_jao = {}
    volt_by_jao = {}
    for _, row in jao_gdf.iterrows():
        jid = str(row.get('id'))
        if not jid:
            continue
        geom_by_jao[jid] = row.geometry
        try:
            volt_by_jao[jid] = int(row.get('v_nom'))
        except Exception:
            volt_by_jao[jid] = None

    # keep only JAOs that have both a result and a geometry (and a voltage)
    jao_ids = [jid for jid in result_by_jao.keys() if jid in geom_by_jao and volt_by_jao.get(jid) is not None]
    if not jao_ids:
        if verbose:
            print("No eligible JAO results to process.")
        return matching_results

    geoms = [geom_by_jao[jid] for jid in jao_ids]
    voltages = [volt_by_jao[jid] for jid in jao_ids]

    # STRtree over JAO geometries
    tree = STRtree(geoms)
    geom_to_idx = {id(g): i for i, g in enumerate(geoms)}  # map geom identity -> index in our arrays

    def _same_voltage(a, b):
        try:
            A = 400 if int(a) in (380, 400) else int(a)
            B = 400 if int(b) in (380, 400) else int(b)
            return A == B
        except Exception:
            return False

    def _buffer_deg_at_lat(meters, lat_deg):
        return float(meters) / float(_meters_per_degree(lat_deg) or 111111.0)

    shares_made = 0
    seen_pairs = set()  # avoid double work (i,j) and (j,i)

    # Iterate each JAO and query candidates via its corridor buffer
    for i, jid in enumerate(jao_ids):
        r_i = result_by_jao.get(jid)
        if not r_i or not r_i.get('network_ids'):
            continue

        g_i = geoms[i]
        v_i = voltages[i]

        # corridor buffer in degrees at this latitude
        lat = g_i.centroid.y
        buf_deg = _buffer_deg_at_lat(buffer_m, lat)
        corridor = g_i.buffer(buf_deg)

        try:
            hits = tree.query(corridor)  # returns geometries intersecting the corridor
        except TypeError:
            # Older Shapely may not accept polygon query cleanly; fallback to line itself
            hits = tree.query(g_i)

        # Map to indices; ignore self
        cand_idxs = []
        for h in hits if isinstance(hits, (list, tuple)) else [hits]:
            j = geom_to_idx.get(id(h))
            if j is None or j == i:
                continue
            cand_idxs.append(j)

        if not cand_idxs:
            continue

        for j in cand_idxs:
            key = (min(i, j), max(i, j))
            if key in seen_pairs:
                continue
            seen_pairs.add(key)

            jid_j = jao_ids[j]
            r_j = result_by_jao.get(jid_j)
            if not r_j or not r_j.get('network_ids'):
                continue

            # same voltage only
            if not _same_voltage(v_i, voltages[j]):
                continue

            g_j = geoms[j]

            # geometric similarity checks (cheap -> expensive)
            try:
                cov_ij = calculate_geometry_coverage(g_i, g_j, buffer_meters=buffer_m)
                cov_ji = calculate_geometry_coverage(g_j, g_i, buffer_meters=buffer_m)
            except Exception:
                # if coverage fails (e.g., invalid geometry), skip this pair
                continue

            avg_cov = (float(cov_ij or 0.0) + float(cov_ji or 0.0)) / 2.0
            if avg_cov < float(min_avg_coverage):
                continue

            # Hausdorff in meters
            try:
                hd_deg = g_i.hausdorff_distance(g_j)
                m_per_deg = _meters_per_degree((g_i.centroid.y + g_j.centroid.y) / 2.0)
                hd_m = float(hd_deg) * float(m_per_deg)
            except Exception:
                hd_m = max_hausdorff_m + 1  # force reject if we can't compute

            if hd_m > float(max_hausdorff_m):
                continue

            # At this point, we consider them parallel enough to share
            if verbose:
                print(f"  Parallel JAOs: {jid} ↔ {jid_j}  (avg_cov={avg_cov:.2f}, hd={hd_m:.0f} m)")

            set_i = set(r_i.get('network_ids') or [])
            set_j = set(r_j.get('network_ids') or [])

            # Respect path lock: don't append to a path-based result
            add_to_i = (set_j - set_i) if not r_i.get('is_path_based') else set()
            add_to_j = (set_i - set_j) if not r_j.get('is_path_based') else set()

            if add_to_i:
                # keep stable order: existing + new (dedup preserving order)
                r_i['network_ids'] = list(dict.fromkeys(list(r_i['network_ids']) + list(add_to_i)))
                r_i['matched'] = True
                if not r_i.get('is_parallel_circuit', False) and not r_i.get('is_duplicate', False):
                    r_i['is_parallel_circuit'] = True
                    r_i['match_quality'] = f"Parallel Circuit ({v_i} kV) - Shared"
                shares_made += len(add_to_i)

            if add_to_j:
                r_j['network_ids'] = list(dict.fromkeys(list(r_j['network_ids']) + list(add_to_j)))
                r_j['matched'] = True
                if not r_j.get('is_parallel_circuit', False) and not r_j.get('is_duplicate', False):
                    r_j['is_parallel_circuit'] = True
                    r_j['match_quality'] = f"Parallel Circuit ({voltages[j]} kV) - Shared"
                shares_made += len(add_to_j)

    if verbose:
        print(f"Shared {shares_made} network lines between parallel JAO lines")

    return matching_results


def detect_and_prune_branches(network_ids, network_gdf, jao_geom):
    """Detect and remove branches that deviate from the main path."""
    import networkx as nx
    from shapely.geometry import Point, LineString

    # SAFEGUARD: If we have 3 or fewer network IDs, don't try to prune
    # This avoids over-pruning simple paths
    if len(network_ids) <= 3:
        return network_ids

    # Build a small graph of just these segments
    G = nx.Graph()
    geom_dict = {}  # Store geometries for validation

    # First pass: add all nodes
    for nid in network_ids:
        rows = network_gdf[network_gdf['id'].astype(str) == str(nid)]
        if rows.empty:
            continue

        geom = rows.iloc[0].geometry
        geom_dict[str(nid)] = geom

        if hasattr(geom, 'geoms'):  # MultiLineString
            for line in geom.geoms:
                start = tuple(line.coords[0])
                end = tuple(line.coords[-1])
                G.add_node(start)
                G.add_node(end)
        elif geom.geom_type == "LineString":
            start = tuple(geom.coords[0])
            end = tuple(geom.coords[-1])
            G.add_node(start)
            G.add_node(end)

    # Second pass: add all edges
    for nid in network_ids:
        rows = network_gdf[network_gdf['id'].astype(str) == str(nid)]
        if rows.empty:
            continue

        geom = rows.iloc[0].geometry

        if hasattr(geom, 'geoms'):  # MultiLineString
            for line in geom.geoms:
                start = tuple(line.coords[0])
                end = tuple(line.coords[-1])
                G.add_edge(start, end, id=nid)
        elif geom.geom_type == "LineString":
            start = tuple(geom.coords[0])
            end = tuple(geom.coords[-1])
            G.add_edge(start, end, id=nid)

    # Find endpoints (nodes with degree 1)
    endpoints = [n for n in G.nodes if G.degree(n) == 1]

    # SAFEGUARD: If we don't have at least 2 endpoints, don't prune
    if len(endpoints) < 2:
        return network_ids

    # If we have more than 2 endpoints, we have branches
    if len(endpoints) > 2:
        # If JAO geom is a LineString, get its endpoints
        jao_start = Point(jao_geom.coords[0]) if jao_geom.geom_type == "LineString" else None
        jao_end = Point(jao_geom.coords[-1]) if jao_geom.geom_type == "LineString" else None

        if jao_start is None or jao_end is None:
            print("  Warning: JAO geometry is not a LineString, skipping branch pruning")
            return network_ids

        # Find closest network endpoint to each JAO endpoint
        best_start = min(endpoints, key=lambda p: Point(p).distance(jao_start))
        best_end = min(endpoints, key=lambda p: Point(p).distance(jao_end))

        # SAFEGUARD: Don't try to find a path from a point to itself
        if best_start == best_end:
            print("  Warning: Best start and end points are the same, skipping branch pruning")
            return network_ids

        # Find shortest path between these endpoints
        try:
            path = nx.shortest_path(G, best_start, best_end)

            # Extract network IDs along this path
            main_path_ids = []
            for i in range(len(path) - 1):
                edge_data = G.get_edge_data(path[i], path[i + 1])
                if edge_data and 'id' in edge_data:
                    main_path_ids.append(edge_data['id'])

            # SAFEGUARD: Ensure we have at least one network ID in the path
            if not main_path_ids:
                print("  Warning: No network IDs found in the main path, skipping branch pruning")
                return network_ids

            # SAFEGUARD: Check if the pruned path is reasonable compared to the JAO line
            # Create LineString from the pruned path
            pruned_lines = [geom_dict.get(str(nid)) for nid in main_path_ids if str(nid) in geom_dict]
            if not pruned_lines:
                print("  Warning: No geometries found for pruned path, skipping branch pruning")
                return network_ids

            # Combine the pruned lines
            pruned_geoms = []
            for geom in pruned_lines:
                if hasattr(geom, 'geoms'):
                    pruned_geoms.extend(list(geom.geoms))
                else:
                    pruned_geoms.append(geom)

            # Check the total length of the pruned path vs. JAO
            pruned_length = sum(line.length for line in pruned_geoms)
            jao_length = jao_geom.length

            # If the pruned path is less than 50% of the JAO length, it's likely we pruned too much
            if pruned_length < 0.5 * jao_length:
                print(
                    f"  Warning: Pruned path is only {pruned_length / jao_length:.1%} of JAO length, skipping branch pruning")
                return network_ids

            # Return only the network IDs on the main path
            return main_path_ids
        except nx.NetworkXNoPath:
            print("  Warning: No path found between endpoints, skipping branch pruning")
            pass
        except Exception as e:
            print(f"  Error in branch pruning: {e}, skipping branch pruning")
            return network_ids

    # If no branches or couldn't find a path, return original IDs
    return network_ids


def calculate_length_km(geometry):
    """Calculate the length of a geometry in kilometers."""
    import numpy as np

    # Handle different geometry types
    if geometry is None:
        return 0.0

    if hasattr(geometry, 'geoms'):  # MultiLineString
        total_length_degrees = sum(line.length for line in geometry.geoms)
    elif geometry.geom_type == "LineString":
        total_length_degrees = geometry.length
    else:
        return 0.0  # Unsupported geometry type

    # Get an approximate center latitude for conversion
    if geometry.geom_type == "LineString" and len(geometry.coords) > 0:
        center_lat = np.mean([p[1] for p in geometry.coords])
    elif hasattr(geometry, 'geoms') and len(geometry.geoms) > 0:
        # For MultiLineString, get first line's coordinates
        first_line = list(geometry.geoms)[0]
        if len(first_line.coords) > 0:
            center_lat = np.mean([p[1] for p in first_line.coords])
        else:
            center_lat = 0.0
    else:
        center_lat = 0.0

    # Convert degrees to kilometers (approximate)
    # 1 degree of longitude at the equator is approximately 111.32 km
    # Adjust for latitude using cos(lat)
    meters_per_degree = 111320 * np.cos(np.radians(abs(center_lat)))
    length_km = total_length_degrees * meters_per_degree / 1000.0

    return length_km


def prune_all_branches(matching_results, jao_gdf, network_gdf):
    """Apply branch pruning to all matches to ensure only main paths are kept."""
    print("\nPruning branch lines from matches...")
    from shapely.geometry import Point
    import networkx as nx

    pruned_count = 0

    for result in matching_results:
        if not result.get('matched') or not result.get('network_ids'):
            continue

        jao_id = result.get('jao_id')

        # Get JAO geometry
        jao_rows = jao_gdf[jao_gdf['id'].astype(str) == str(jao_id)]
        if jao_rows.empty:
            continue

        jao_geom = jao_rows.iloc[0].geometry

        # Prune branches
        original_network_ids = result['network_ids']
        pruned_network_ids = detect_and_prune_branches(original_network_ids, network_gdf, jao_geom)

        # SAFEGUARD: If pruning would leave us with no network IDs, keep the original
        if not pruned_network_ids:
            print(f"  Warning: Pruning would leave JAO {jao_id} with no network lines. Keeping original match.")
            continue

        # Only update if we actually pruned something and have lines left
        if len(pruned_network_ids) < len(original_network_ids) and len(pruned_network_ids) > 0:
            # Update network IDs
            result['network_ids'] = pruned_network_ids

            # IMPORTANT: Remove any existing matched_lines_data to force recalculation
            if 'matched_lines_data' in result:
                result.pop('matched_lines_data')

            # Add a special flag for each network ID that wasn't in the original
            # This tells the allocation function that it needs to create entries for these
            new_network_ids = [nid for nid in pruned_network_ids if nid not in original_network_ids]
            if new_network_ids:
                if 'new_network_ids' not in result:
                    result['new_network_ids'] = []
                result['new_network_ids'].extend(new_network_ids)
                print(f"  Added {len(new_network_ids)} new network IDs after pruning for JAO {jao_id}")

            # Recalculate path length and ratio
            path_length = 0
            for nid in pruned_network_ids:
                rows = network_gdf[network_gdf['id'].astype(str) == str(nid)]
                if not rows.empty:
                    path_length += calculate_length_meters(rows.iloc[0].geometry)

            # Update path length and ratio
            result['path_length'] = float(path_length)
            if result.get('jao_length'):
                result['length_ratio'] = float(path_length / result['jao_length'])

            # Update match status to reflect pruning
            if 'match_status' in result:
                result['match_status'] = result['match_status'] + " | Branch lines pruned"

            # Ensure the matched flag stays true
            result['matched'] = True

            # Mark that this match was pruned
            result['pruned_branches'] = True
            pruned_count += 1
            print(
                f"  Pruned branches from JAO {jao_id}: removed {len(original_network_ids) - len(pruned_network_ids)} segments")

    print(f"Pruned branches from {pruned_count} matches")
    return matching_results


def ensure_match_consistency(matching_results):
    """Ensure all match-related fields are consistent across the results."""
    print("\nEnsuring match consistency...")

    fixed = 0
    for result in matching_results:
        # Check if we have network_ids but not marked as matched
        if result.get('network_ids') and not result.get('matched'):
            result['matched'] = True
            fixed += 1

        # Check if we have matched_lines_data but no network_ids
        elif result.get('matched_lines_data') and not result.get('network_ids'):
            result['network_ids'] = [str(seg['network_id']) for seg in result['matched_lines_data']]
            result['matched'] = True
            fixed += 1

        # Check if we're marked as matched but have no network_ids or matched_lines_data
        elif result.get('matched') and not (result.get('network_ids') or result.get('matched_lines_data')):
            # This is a problem - decide whether to mark as unmatched or try to recover
            print(
                f"  Warning: JAO {result.get('jao_id')} marked as matched but has no network_ids or matched_lines_data")

            # If there's allocation data, try to recover
            if any(k in result for k in ['allocated_r_sum', 'allocated_x_sum', 'allocated_b_sum']):
                print(f"  Attempting to recover match for JAO {result.get('jao_id')} from allocation data")
                # Could implement recovery logic here if needed
            else:
                # Can't recover, mark as unmatched
                result['matched'] = False
                fixed += 1

    print(f"Fixed consistency issues in {fixed} matches")
    return matching_results


def fix_inconsistent_matches(matching_results):
    """Fix any inconsistencies between network_ids, matched flag, and matched_lines_data."""
    print("\nFixing inconsistent matches...")

    fixed_count = 0
    for result in matching_results:
        original_matched = result.get('matched', False)
        has_network_ids = bool(result.get('network_ids', []))
        has_matched_lines = bool(result.get('matched_lines_data', []))

        # Case 1: Has network IDs but not marked as matched
        if has_network_ids and not original_matched:
            result['matched'] = True
            fixed_count += 1
            print(f"  Fixed JAO {result.get('jao_id')}: Has network IDs but wasn't marked as matched")

        # Case 2: Has matched lines data but no network IDs
        elif has_matched_lines and not has_network_ids:
            # Extract network IDs from matched_lines_data
            result['network_ids'] = [str(line.get('network_id')) for line in result.get('matched_lines_data', [])
                                     if line.get('network_id')]
            result['matched'] = True
            fixed_count += 1
            print(f"  Fixed JAO {result.get('jao_id')}: Had matched_lines_data but no network_ids")

        # Case 3: Marked as matched but has neither network IDs nor matched lines
        elif original_matched and not (has_network_ids or has_matched_lines):
            # This is likely an error, mark as unmatched
            result['matched'] = False
            fixed_count += 1
            print(f"  Fixed JAO {result.get('jao_id')}: Was marked as matched but had no network data")

    print(f"Fixed {fixed_count} inconsistent matches")
    return matching_results


def reallocate_parameters_after_pruning(result, jao_gdf, network_gdf):
    """Reallocate electrical parameters after branch pruning."""
    import pandas as pd
    import numpy as np

    if not result.get('matched') or not result.get('network_ids'):
        return result

    jao_id = result.get('jao_id')
    network_ids = result.get('network_ids', [])

    # Check if we have the necessary JAO values
    jao_rows = jao_gdf[jao_gdf['id'].astype(str) == str(jao_id)]
    if jao_rows.empty:
        return result

    jao_row = jao_rows.iloc[0]

    # Get JAO parameters
    jao_r = _get_first_existing(jao_row, 'r', 'R', 'resistance')
    jao_x = _get_first_existing(jao_row, 'x', 'X', 'reactance')
    jao_b = _get_first_existing(jao_row, 'b', 'B', 'susceptance')

    if all(v is None for v in [jao_r, jao_x, jao_b]):
        return result

    # Create matched_lines_data if it doesn't exist
    if 'matched_lines_data' not in result or not result['matched_lines_data']:
        result['matched_lines_data'] = []

        # Collect data for each network line
        for nid in network_ids:
            net_rows = network_gdf[network_gdf['id'].astype(str) == str(nid)]
            if net_rows.empty:
                continue

            net_row = net_rows.iloc[0]
            length_km = _to_km(_get_first_existing(net_row, 'length_km', 'length'))
            circuits = int(net_row.get('circuits', 1))

            # Create entry
            result['matched_lines_data'].append({
                'network_id': str(nid),
                'length_km': length_km,
                'circuits': circuits
            })

    # Calculate total length and weighted parameters
    total_length_km = sum(seg.get('length_km', 0) for seg in result['matched_lines_data'])

    if total_length_km <= 0:
        return result

    # Update JAO values in the result
    result['jao_r'] = jao_r
    result['jao_x'] = jao_x
    result['jao_b'] = jao_b

    # Calculate per-km values
    jao_length_km = _get_first_existing(jao_row, 'jao_length_km', 'length_km', 'length')
    if jao_length_km is not None and jao_length_km > 0:
        result['jao_r_per_km'] = jao_r / jao_length_km if jao_r is not None else None
        result['jao_x_per_km'] = jao_x / jao_length_km if jao_x is not None else None
        result['jao_b_per_km'] = jao_b / jao_length_km if jao_b is not None else None

    # Allocate parameters based on length proportion
    for seg in result['matched_lines_data']:
        length_ratio = seg['length_km'] / total_length_km

        # Allocate total values
        if jao_r is not None:
            seg['allocated_r'] = jao_r * length_ratio
        if jao_x is not None:
            seg['allocated_x'] = jao_x * length_ratio
        if jao_b is not None:
            seg['allocated_b'] = jao_b * length_ratio

        # Calculate per-km values
        length_km = seg.get('length_km', 0)
        if length_km > 0:
            if jao_r is not None:
                seg['jao_r_per_km'] = seg['allocated_r'] / length_km
            if jao_x is not None:
                seg['jao_x_per_km'] = seg['allocated_x'] / length_km
            if jao_b is not None:
                seg['jao_b_per_km'] = seg['allocated_b'] / length_km

        # Calculate per-circuit values
        circuits = seg.get('circuits', 1)
        if circuits > 0:
            if jao_r is not None:
                seg['jao_r_per_km_pc'] = seg['jao_r_per_km'] * circuits if 'jao_r_per_km' in seg else None
            if jao_x is not None:
                seg['jao_x_per_km_pc'] = seg['jao_x_per_km'] * circuits if 'jao_x_per_km' in seg else None
            if jao_b is not None:
                seg['jao_b_per_km_pc'] = seg['jao_b_per_km'] * circuits if 'jao_b_per_km' in seg else None

    # Compute sums for validation
    result['allocated_r_sum'] = sum(seg.get('allocated_r', 0) or 0 for seg in result['matched_lines_data'])
    result['allocated_x_sum'] = sum(seg.get('allocated_x', 0) or 0 for seg in result['matched_lines_data'])
    result['allocated_b_sum'] = sum(seg.get('allocated_b', 0) or 0 for seg in result['matched_lines_data'])

    # Calculate residuals
    if jao_r is not None and jao_r != 0:
        result['residual_r_percent'] = 100.0 * (jao_r - result['allocated_r_sum']) / abs(jao_r)
    if jao_x is not None and jao_x != 0:
        result['residual_x_percent'] = 100.0 * (jao_x - result['allocated_x_sum']) / abs(jao_x)
    if jao_b is not None and jao_b != 0:
        result['residual_b_percent'] = 100.0 * (jao_b - result['allocated_b_sum']) / abs(jao_b)

    return result


def remove_duplicate_jao_entries(matching_results):
    """Remove any duplicate entries for the same JAO ID, keeping the matched one if available."""
    print("\nRemoving duplicate JAO entries...")

    # Group by JAO ID
    jao_id_groups = {}
    for result in matching_results:
        jao_id = result.get('jao_id')
        if jao_id not in jao_id_groups:
            jao_id_groups[jao_id] = []
        jao_id_groups[jao_id].append(result)

    # Create new list with duplicates removed
    new_results = []
    duplicates_removed = 0

    for jao_id, group in jao_id_groups.items():
        if len(group) > 1:
            # Keep the matched one if available
            matched_entries = [r for r in group if r.get('matched', False)]
            if matched_entries:
                # Keep the first matched entry
                new_results.append(matched_entries[0])
            else:
                # No matched entries, keep the first one
                new_results.append(group[0])
            duplicates_removed += len(group) - 1
            print(f"  Removed {len(group) - 1} duplicate entries for JAO {jao_id}")
        else:
            # Only one entry, keep it
            new_results.append(group[0])

    print(f"Removed {duplicates_removed} duplicate JAO entries")
    return new_results


def visualize_results_with_path(jao_gdf, network_gdf, matching_results, pypsa_gdf=None, output_path=None):
    """Wrapper for visualize_results that allows specifying a custom output path."""
    # Call the original function
    result_path = visualize_results(jao_gdf, network_gdf, matching_results, pypsa_gdf)

    # If a custom path is specified and the result exists, copy it
    if output_path and result_path and os.path.exists(result_path):
        import shutil
        try:
            # Make sure the directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            # Copy the file to the desired location
            shutil.copy(result_path, output_path)
            return output_path
        except Exception as e:
            print(f"Warning: Failed to copy visualization to {output_path}: {e}")

    return result_path


def create_enhanced_summary_table_with_path(jao_gdf, network_gdf, matching_results, pypsa_gdf=None, output_path=None):
    """Wrapper for create_enhanced_summary_table that allows specifying a custom output path."""
    # Call the original function
    result_path = create_enhanced_summary_table(jao_gdf, network_gdf, matching_results, pypsa_gdf)

    # If a custom path is specified and the result exists, copy it
    if output_path and result_path and os.path.exists(result_path):
        import shutil
        try:
            # Make sure the directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            # Copy the file to the desired location
            shutil.copy(result_path, output_path)
            return output_path
        except Exception as e:
            print(f"Warning: Failed to copy summary table to {output_path}: {e}")

    return result_path


# Add this function after your other wrappers:
def create_enhanced_summary_table_with_pypsa_debug(jao_gdf, network_gdf, matching_results, pypsa_gdf=None,
                                                   output_path=None):
    """Enhanced version of create_enhanced_summary_table with better PyPSA support."""

    # Print debugging info
    print(f"PyPSA GDF columns: {pypsa_gdf.columns.tolist() if pypsa_gdf is not None else 'None'}")

    # Check if any PyPSA lines are in matching_results
    pypsa_lines_in_results = [r for r in matching_results if r.get('is_pypsa_line', False)]
    print(f"PyPSA lines in matching results: {len(pypsa_lines_in_results)}")

    # Make sure the combined_results has the PyPSA lines explicitly marked
    for r in matching_results:
        if 'jao_id' in r and r['jao_id'].startswith('pypsa_'):
            r['is_pypsa_line'] = True

    # Call the original function
    result_path = create_enhanced_summary_table(jao_gdf, network_gdf, matching_results, pypsa_gdf)

    # If the output exists, let's directly add the PyPSA tab content
    if result_path and os.path.exists(result_path) and pypsa_gdf is not None and len(pypsa_gdf) > 0:
        try:
            with open(result_path, 'r', encoding='utf-8') as f:
                html_content = f.read()

            # Check if tab navigation exists
            if '<div class="tab">' in html_content:
                print("Tab navigation found, ensuring PyPSA tab exists...")

                # Check if PyPSA tab button exists, add it if not
                if 'onclick="openTab(event, \'pypsa-lines-tab\')"' not in html_content:
                    tab_nav = '<div class="tab">'
                    pypsa_tab_button = '<button class="tablinks" onclick="openTab(event, \'pypsa-lines-tab\')">PyPSA Lines</button>'

                    # Add PyPSA tab button to navigation
                    html_content = html_content.replace(tab_nav, tab_nav + "\n    " + pypsa_tab_button)

                # Create PyPSA tab content
                pypsa_rows = []
                for _, row in pypsa_gdf.iterrows():
                    pypsa_id = str(row.get('id', ''))
                    # Create a basic row for each PyPSA line
                    pypsa_rows.append(f"""
                    <tr>
                        <td>{pypsa_id}</td>
                        <td>{row.get('bus0', '')}-{row.get('bus1', '')}</td>
                        <td>{int(row.get('voltage', 0))}</td>
                        <td>{row.get('length', 0):.2f}</td>
                        <td>{row.get('circuits', 1)}</td>
                        <td>Unknown</td>
                        <td>-</td>
                    </tr>
                    """)

                # Create the complete PyPSA tab content
                pypsa_tab_content = f"""
                <div id="pypsa-lines-tab" class="tabcontent">
                    <h2>PyPSA Lines</h2>
                    <p>This tab shows all {len(pypsa_gdf)} PyPSA lines.</p>
                    <table id="pypsa-lines-table" class="results-table">
                        <thead>
                            <tr>
                                <th>PyPSA ID</th>
                                <th>Buses</th>
                                <th>Voltage (kV)</th>
                                <th>Length (km)</th>
                                <th>Circuits</th>
                                <th>Matched</th>
                                <th>Network Lines</th>
                            </tr>
                        </thead>
                        <tbody>
                            {"".join(pypsa_rows)}
                        </tbody>
                    </table>
                </div>
                """

                # Check if PyPSA tab already exists
                if 'id="pypsa-lines-tab"' in html_content:
                    print("PyPSA tab already exists, replacing content...")
                    # Use regex to replace the entire tab content
                    import re
                    pattern = r'<div id="pypsa-lines-tab" class="tabcontent">.*?</div>\s*(?=<div id|<script)'
                    html_content = re.sub(pattern, pypsa_tab_content, html_content, flags=re.DOTALL)
                else:
                    print("PyPSA tab doesn't exist, adding it...")
                    # Add PyPSA tab content before the script tag
                    script_tag = '<script>'
                    html_content = html_content.replace(script_tag, pypsa_tab_content + "\n" + script_tag)

                # Write the updated HTML back to file
                with open(result_path, 'w', encoding='utf-8') as f:
                    f.write(html_content)
                print(f"Added {len(pypsa_rows)} PyPSA lines to the HTML table")
            else:
                print("No tab navigation found in HTML file")

            # If a custom path is specified, copy it
            if output_path and output_path != result_path:
                import shutil
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                shutil.copy(result_path, output_path)
                return output_path

        except Exception as e:
            print(f"Error enhancing PyPSA tab: {e}")
            traceback.print_exc()

    return result_path


def main():
    import os
    import pandas as pd
    from time import time
    import numpy as np
    import traceback
    import json

    # ---------- config / output ----------
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    # Track execution time
    start_time = time()

    # ---------- load ----------
    print("\n=== LOADING DATA ===")
    print("Loading JAO lines...")
    jao_gdf = load_jao_lines()
    print(f"Loaded {len(jao_gdf)} JAO lines")

    # Quick schema probe
    print("JAO DataFrame Column Check:")
    print(f"Columns: {jao_gdf.columns.tolist()}")
    if len(jao_gdf) > 0:
        first_row = jao_gdf.iloc[0]
        for col in ['id', 'NE_name', 'r', 'x', 'b', 'length', 'jao_length_km', 'R_per_km', 'X_per_km', 'B_per_km']:
            val = first_row.get(col, 'NOT FOUND')
            print(f"  {col}: {val} (type: {type(val).__name__})")

    print("\nLoading network lines (excluding 110kV)...")
    network_gdf = load_network_lines()
    print(f"Loaded {len(network_gdf)} network lines")

    # Load PyPSA lines early to make them available for all visualizations
    print("\nLoading PyPSA lines...")
    from pypsa_integration import load_pypsa_data
    pypsa_path = os.path.join(data_dir, 'pypsa-eur-lines-germany.csv')
    pypsa_gdf = None
    try:
        pypsa_gdf = load_pypsa_data(pypsa_path)
        if pypsa_gdf is not None:
            print(f"Loaded {len(pypsa_gdf)} PyPSA lines")
    except Exception as e:
        print(f"Failed to load PyPSA lines: {e}")
        pypsa_gdf = None

    # Fix parallel circuits information
    network_gdf = _ensure_circuits_col(network_gdf)

    # Check problematic lines that might have missing num_parallel
    for special_line in ['Line_16621', 'Line_16622', 'Line_6481', 'Line_17046']:
        line_rows = network_gdf.loc[network_gdf['id'].astype(str) == special_line]
        if not line_rows.empty:
            print(f"Checking {special_line} parallel circuit info: {line_rows.iloc[0]['circuits']} circuits")

    # Compute normalized per-circuit parameters
    network_gdf = compute_network_per_circuit_params(network_gdf)

    # ---------- duplicates / nearest / graph ----------
    print("\n=== PREPROCESSING ===")
    print("Identifying duplicate JAO geometries...")
    jao_to_group, geometry_groups = identify_duplicate_jao_lines(jao_gdf)
    print(f"Found {len([g for g, ids in geometry_groups.items() if len(ids) > 1])} duplicate JAO groups")
    print(f"Total duplicate JAO lines: {len(jao_to_group)}")

    print("\nFinding nearest points for JAO endpoints (enhanced version)...")
    nearest_points_dict = find_nearest_points(
        jao_gdf,
        network_gdf,
        max_alternatives=10,
        distance_threshold_meters=1500,  # Increased from 500m
        substation_cluster_radius_meters=350  # Clustering radius for substations
    )

    print("\nBuilding network graph...")
    G = build_network_graph(network_gdf)
    G = add_station_hubs_to_graph(G, network_gdf, radius_m=350)  # <<< new

    # Convert edge weights from degrees to meters (only once)
    for u, v, d in G.edges(data=True):
        if '_unit' not in d:  # don't double-convert
            lat = (G.nodes[u]['y'] + G.nodes[v]['y']) / 2
            d['weight_deg'] = d['weight']  # keep the raw value
            d['weight'] = d['weight'] * _meters_per_degree(lat)
            d['_unit'] = 'm'  # mark as converted

    print(f"Base graph has {len(G.nodes)} nodes and {len(G.edges)} edges")

    # Repair graph connectivity with enhanced approach
    print("\nRepairing network graph connectivity...")
    G = repair_network_graph(G, network_gdf, connection_threshold_meters=200)

    # Add more aggressive connectivity to handle disconnected segments
    G = augment_graph_with_bridges(G, network_gdf, max_gap_m=300)  # Increased from 150m

    print(f"Enhanced graph has {len(G.nodes)} nodes and {len(G.edges)} edges")

    # ---------- MULTI-STAGE MATCHING STRATEGY ----------
    print("\n=== STAGE 1: PRIMARY GRAPH-BASED MATCHING ===")
    print("Finding matching network lines with special handling for duplicates and electrical scoring...")
    matching_results = find_matching_network_lines_with_duplicates(
        jao_gdf,
        network_gdf,
        nearest_points_dict,
        G,
        duplicate_groups=geometry_groups,
        max_reuse=5,  # Increased from 4
        max_paths_to_try=150,  # Increased from 100
        min_length_ratio=0.4,  # More permissive (was 0.5)
        max_length_ratio=5.0,  # More permissive (was 4.0)
        corridor_km_strict=4.0,  # Increased from 3.0
        corridor_km_relaxed=8.0,  # Increased from 7.0
        time_budget_s=150.0  # Increased from 100.0
    )

    # Calculate initial match rate
    total_jao = len(jao_gdf)
    matched_jao_ids = set(r['jao_id'] for r in matching_results if r.get('matched', False))
    initial_match_rate = len(matched_jao_ids) / total_jao
    print(f"Initial matching rate: {len(matched_jao_ids)}/{total_jao} ({initial_match_rate:.1%})")

    print("\n=== STAGE 2: GEOMETRIC MATCHING FOR UNMATCHED LINES ===")
    # Identify unmatched JAO lines
    unmatched_jao_ids = set(str(r['id']) for _, r in jao_gdf.iterrows()) - matched_jao_ids
    print(f"Found {len(unmatched_jao_ids)} unmatched JAO lines to process in Stage 2")

    # After the initial graph-based matching
    print("\n=== IMPROVING PARALLEL CIRCUIT HANDLING ===")
    matching_results = match_parallel_circuits_robustly(matching_results, jao_gdf, network_gdf)

    # Update matched JAO IDs after parallel circuit improvement
    matched_jao_ids = set(r['jao_id'] for r in matching_results if r.get('matched', False))
    parallel_improved_rate = len(matched_jao_ids) / total_jao
    print(
        f"Match rate after parallel circuit improvement: {len(matched_jao_ids)}/{total_jao} ({parallel_improved_rate:.1%})")

    # Run geometric matching with enhanced parameters
    matching_results = match_remaining_lines_by_geometry(
        jao_gdf,
        network_gdf,
        matching_results,
        buffer_distance=0.008,  # Increased
        snap_tolerance=500,  # Increased from 300
        angle_tolerance=45,  # More permissive (was 30)
        min_dir_cos=0.7,  # More permissive (was 0.866)
        min_length_ratio=0.3,  # More permissive
        max_length_ratio=3.5  # More permissive
    )

    # Update matched JAO IDs after geometric matching
    matched_jao_ids = set(r['jao_id'] for r in matching_results if r.get('matched', False))
    geo_match_rate = len(matched_jao_ids) / total_jao
    print(f"Match rate after geometric matching: {len(matched_jao_ids)}/{total_jao} ({geo_match_rate:.1%})")

    print("\n=== STAGE 3: PARALLEL CIRCUIT MATCHING ===")
    # Identify remaining unmatched JAO lines
    unmatched_jao_ids = set(str(r['id']) for _, r in jao_gdf.iterrows()) - matched_jao_ids
    print(f"Found {len(unmatched_jao_ids)} unmatched JAO lines to process in Stage 3")

    # Run parallel circuit matching
    matching_results = match_parallel_circuit_jao_with_network(
        matching_results,
        jao_gdf,
        network_gdf,
        G,
        nearest_points_dict
    )

    # Update matched JAO IDs after parallel matching
    matched_jao_ids = set(r['jao_id'] for r in matching_results if r.get('matched', False))
    parallel_match_rate = len(matched_jao_ids) / total_jao
    print(f"Match rate after parallel circuit matching: {len(matched_jao_ids)}/{total_jao} ({parallel_match_rate:.1%})")

    print("\n=== STAGE 4: PARALLEL VOLTAGE CIRCUITS ===")
    # Match lines with same geometry but different voltage
    matching_results = match_parallel_voltage_circuits(
        jao_gdf,
        network_gdf,
        matching_results
    )

    # Update matched JAO IDs
    matched_jao_ids = set(r['jao_id'] for r in matching_results if r.get('matched', False))
    voltage_match_rate = len(matched_jao_ids) / total_jao
    print(f"Match rate after parallel voltage matching: {len(matched_jao_ids)}/{total_jao} ({voltage_match_rate:.1%})")

    print("\n=== STAGE 5: GEOMETRIC FALLBACK ===")
    # Final geometric fallback for any remaining unmatched lines
    matching_results = geometric_fallback_and_enhance(
        matching_results,
        jao_gdf,
        network_gdf,
        buffer_m=1500  # Increased from 1200
    )

    # Update matched JAO IDs
    matched_jao_ids = set(r['jao_id'] for r in matching_results if r.get('matched', False))
    fallback_match_rate = len(matched_jao_ids) / total_jao
    print(f"Match rate after fallback: {len(matched_jao_ids)}/{total_jao} ({fallback_match_rate:.1%})")

    # Sample check for specific JAO IDs
    for r in matching_results:
        if r.get('jao_id') in ['2282', '2283']:
            print(f"POST-ALLOC {r['jao_id']}: segments={len(r.get('matched_lines_data') or [])}")

    print("Normalizing fields and backfilling parameters from JAO dataframe...")
    matching_results = normalize_and_fill_params(matching_results, jao_gdf, network_gdf)

    print("\nHandling extreme length ratio matches...")
    matching_results = fix_extreme_ratio_matches(matching_results, jao_gdf, network_gdf)

    print("Backfilling network segment tables for simple matches...")
    matching_results, export_rows = ensure_network_segment_tables(
        matching_results,
        jao_gdf,
        network_gdf,
        freq_hz=50.0
    )

    # Add this right before visualization
    matching_results = protect_fixed_allocations(matching_results)

    # ---------- SPECIAL CASE HANDLING ----------
    print("\n=== HANDLING SPECIAL CASES ===")
    # Special handling for JAO 97
    matching_results = debug_specific_jao_match(
        jao_gdf,
        network_gdf,
        matching_results,
        jao_id_to_debug="97",
        target_network_ids=["Line_8160", "Line_30733", "Line_30181", "Line_17856"]
    )

    # Find network lines that are used multiple times
    print("\nAnalyzing network line usage...")
    network_line_usage, reused_lines = find_network_line_usage(matching_results)

    # Try to improve matches by grouping identical network geometries
    print("\nClustering identical network geometries...")
    matching_results = cluster_identical_network_lines(matching_results, network_gdf)

    # Try to convert geometric matches to path-based matches when possible
    print("\nAttempting to convert geometric matches to path-based matches...")
    matching_results = convert_geometric_to_path_matches(
        matching_results,
        G,
        jao_gdf,
        network_gdf,
        nearest_points_dict
    )

    # Match remaining parallel lines that might be the same
    print("\nMatching remaining identical network geometries...")
    matching_results = match_identical_network_geometries_aggressive(
        matching_results,
        jao_gdf,
        network_gdf
    )

    # Corridor-based matching as a final approach
    print("\nApplying corridor-based matching...")
    matching_results = corridor_parallel_match(
        matching_results,
        jao_gdf,
        network_gdf,
        corridor_w_220=300,  # Increased
        corridor_w_400=400  # Increased
    )

    # Share network lines among parallel JAOs when appropriate
    print("\nSharing network lines among parallel JAO lines...")
    matching_results = share_network_lines_among_parallel_jaos(
        matching_results,
        jao_gdf
    )

    matching_results = trim_to_shortest_paths(
        matching_results, G, jao_gdf, network_gdf, nearest_points_dict,
        keep_parallel=True, max_offcorridor_m=800, max_len_ratio_after=1.35
    )

    def finalize_results(matching_results, network_gdf):
        for r in matching_results:
            # If anything got appended anywhere, dedupe it
            r['network_ids'] = list(dict.fromkeys(r.get('network_ids', [])))

            # If it has network_ids, it's matched
            r['matched'] = bool(r['network_ids'])

            # Recompute path_length from the dataframe so tiny stubs don't inflate/deflate
            total_m = 0.0
            for nid in r['network_ids']:
                rows = network_gdf[network_gdf['id'].astype(str) == str(nid)]
                if rows.empty:
                    continue
                row = rows.iloc[0]
                try:
                    total_m += float(calculate_length_meters(row.geometry))
                except Exception:
                    # fallbacks if you store lengths in km
                    if 'length' in row and pd.notna(row['length']):
                        total_m += float(row['length']) * 1000.0
                    elif 'length_km' in row and pd.notna(row['length_km']):
                        total_m += float(row['length_km']) * 1000.0
            r['path_length'] = float(total_m)

            # Keep ratio consistent if JAO length is known (meters)
            jao_len_m = float(r.get('jao_length') or 0.0)
            if jao_len_m > 0:
                r['length_ratio'] = float(r['path_length'] / jao_len_m)

        return matching_results

    matching_results = share_among_same_endpoints(matching_results, jao_gdf, nearest_points_dict)
    matching_results = finalize_results(matching_results, network_gdf)

    # Add this early in the process, after initial matching but before parameter allocation
    print("\nHandling extreme length ratio matches...")
    matching_results = handle_extreme_ratio_matches(matching_results, jao_gdf, network_gdf)
    # Add this line right before creating visualizations
    matching_results = protect_fixed_allocations(matching_results)

    # Before parameter allocation
    print("\nPruning branch lines from matches...")
    matching_results = prune_all_branches(matching_results, jao_gdf, network_gdf)

    # Fix any inconsistencies that might have been introduced
    print("\nFixing any inconsistent matches...")
    matching_results = fix_inconsistent_matches(matching_results)

    # Remove duplicate JAO entries
    print("\nRemoving duplicate JAO entries...")
    matching_results = remove_duplicate_jao_entries(matching_results)

    print("\n=== STAGE 6: ELECTRICAL PARAMETERS ALLOCATION ===")
    print("Allocating electrical parameters...")
    matching_results = allocate_electrical_parameters(jao_gdf, network_gdf, matching_results)

    # ---------- PYPSA INTEGRATION ----------
    print("\n=== MATCHING PYPSA LINES TO NETWORK ===")
    # Only attempt PyPSA integration if we have PyPSA data
    combined_results = matching_results.copy()
    pypsa_matching_results = None
    pypsa_network_matches = {}

    if pypsa_gdf is not None:
        try:
            from pypsa_integration import match_pypsa_to_network_path_based, convert_pypsa_results_to_jao_format

            # Use only one PyPSA matching function - path_based is more accurate
            print("Running PyPSA to network matching...")
            pypsa_matching_results = match_pypsa_to_network_path_based(pypsa_gdf, network_gdf, G)

            # Create a dictionary for the PyPSA tab in HTML
            pypsa_network_matches = {}
            for result in pypsa_matching_results:
                if result.get('matched', False):
                    pypsa_network_matches[str(result.get('pypsa_id', ''))] = result.get('network_ids', [])
                else:
                    pypsa_network_matches[str(result.get('pypsa_id', ''))] = []

            # Convert results to JAO format for visualization
            pypsa_jao_format = convert_pypsa_results_to_jao_format(pypsa_matching_results)

            # Add voltage information from PyPSA data
            for result in pypsa_jao_format:
                pypsa_id = result['jao_id'].replace('pypsa_', '')  # Extract original ID
                pypsa_row = pypsa_gdf[pypsa_gdf['id'].astype(str) == pypsa_id]
                if not pypsa_row.empty:
                    result['v_nom'] = int(pypsa_row.iloc[0]['voltage']) if 'voltage' in pypsa_row.iloc[0] else 0

            # Combine with existing results for visualization
            combined_results = matching_results + pypsa_jao_format

            # Count matched PyPSA lines
            pypsa_matched_count = sum(1 for r in pypsa_matching_results if r.get('matched', False))
            print(
                f"Matched PyPSA lines: {pypsa_matched_count}/{len(pypsa_gdf)} ({pypsa_matched_count / len(pypsa_gdf) * 100:.1f}%)")

        except Exception as e:
            print(f"Error in PyPSA integration: {str(e)}")
            traceback.print_exc()
    else:
        print("Skipping PyPSA integration - no PyPSA data available")

    # ---------- PARAMETER COMPARISON AND ALLOCATION EXPORTS ----------
    print("\n=== GENERATING PARAMETER COMPARISON FILES ===")

    # Save matching results and PyPSA matches to JSON for parameter comparison
    try:
        matching_results_file = os.path.join(output_dir, "matching_results.json")
        with open(matching_results_file, 'w') as f:
            # Convert to simpler structure to avoid serialization issues
            simplified_results = []
            for r in matching_results:
                # Create a copy without geometry or DataFrame objects
                simplified_r = {}
                for k, v in r.items():
                    # Skip complex objects that can't be serialized
                    if k == 'geometry' or isinstance(v, (pd.DataFrame, pd.Series)):
                        continue
                    # Convert numpy types to Python native types
                    if isinstance(v, (np.int64, np.int32, np.int16, np.int8)):
                        simplified_r[k] = int(v)
                    elif isinstance(v, (np.float64, np.float32)):
                        simplified_r[k] = float(v)
                    # Handle lists with geometry objects
                    elif isinstance(v, list):
                        # Try to convert each item, skip if it fails
                        try:
                            simplified_r[k] = [item if not hasattr(item, 'wkt') else item.wkt for item in v]
                        except:
                            # If conversion fails, just keep basic types
                            simplified_r[k] = [item for item in v if
                                               isinstance(item, (str, int, float, bool, type(None)))]
                    else:
                        # Regular values
                        simplified_r[k] = v
                simplified_results.append(simplified_r)
            json.dump(simplified_results, f)

        pypsa_matches_file = os.path.join(output_dir, "pypsa_network_matches.json")
        # Simplify pypsa_network_matches to avoid serialization issues
        simplified_matches = {}
        for k, v in pypsa_network_matches.items():
            # Convert any complex objects to strings
            simplified_matches[str(k)] = [str(item) for item in v]
        with open(pypsa_matches_file, 'w') as f:
            json.dump(simplified_matches, f)

        print(f"Saved matching results to {matching_results_file}")
        print(f"Saved PyPSA matches to {pypsa_matches_file}")
    except Exception as e:
        print(f"Error saving matching results to JSON: {e}")

    # Prepare JAO allocation rows
    print("Preparing JAO allocation rows...")
    jao_allocation_rows = prepare_jao_allocation_rows(matching_results, jao_gdf, network_gdf)

    # Prepare PyPSA allocation rows if PyPSA data is available
    pypsa_allocation_rows = []
    if pypsa_gdf is not None and pypsa_network_matches:
        print("Preparing PyPSA allocation rows...")
        pypsa_allocation_rows = prepare_pypsa_allocation_rows(pypsa_gdf, network_gdf, pypsa_network_matches)

    # Export allocation details
    print("Exporting allocation details...")
    jao_allocation_path = os.path.join(output_dir, "jao_allocation_details.csv")
    export_allocation_details_csv(jao_allocation_rows, jao_allocation_path, source_type="JAO")

    if pypsa_allocation_rows:
        pypsa_allocation_path = os.path.join(output_dir, "pypsa_allocation_details.csv")
        export_allocation_details_csv(pypsa_allocation_rows, pypsa_allocation_path, source_type="PyPSA")

    # Export updated network lines
    print("Exporting updated network lines...")
    jao_network_path = os.path.join(output_dir, "network_lines_jao.csv")
    export_updated_network_lines_csv(network_gdf, jao_allocation_rows, jao_network_path, source_type="JAO")

    if pypsa_allocation_rows:
        pypsa_network_path = os.path.join(output_dir, "network_lines_pypsa.csv")
        export_updated_network_lines_csv(network_gdf, pypsa_allocation_rows, pypsa_network_path, source_type="PyPSA")

    # Export ready lines for JAO and PyPSA
    print("Exporting ready lines...")
    jao_ready_path = os.path.join(output_dir, "jao_lines_ready.csv")
    export_ready_lines_csv(jao_gdf, jao_allocation_rows, jao_ready_path, source_type="JAO")

    if pypsa_gdf is not None and pypsa_allocation_rows:
        pypsa_ready_path = os.path.join(output_dir, "pypsa_lines_ready.csv")
        export_ready_lines_csv(pypsa_gdf, pypsa_allocation_rows, pypsa_ready_path, source_type="PyPSA")

    # Generate parameter comparison visualization
    print("\nGenerating parameter comparison visualization...")
    try:
        # Create the parameter comparison HTML
        parameter_comparison_script = os.path.join(output_dir, "parameter_comparison.py")
        with open(parameter_comparison_script, 'w') as f:
            f.write("""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import warnings
warnings.filterwarnings('ignore')

# Define file paths
base_dir = "."  # Current directory
jao_file = os.path.join(base_dir, "jao_lines.csv")
pypsa_file = os.path.join(base_dir, "pypsa_lines.csv")
network_jao_file = os.path.join(base_dir, "network_lines_jao.csv")
network_pypsa_file = os.path.join(base_dir, "network_lines_pypsa.csv")

# Load data
jao_df = pd.read_csv(jao_file)
network_jao_df = pd.read_csv(network_jao_file)

# Try to load PyPSA data
try:
    pypsa_df = pd.read_csv(pypsa_file)
    network_pypsa_df = pd.read_csv(network_pypsa_file)
    has_pypsa = True
except:
    has_pypsa = False
    print("PyPSA data not available, will only show JAO comparison")

# Ensure IDs are strings
jao_df['id'] = jao_df['id'].astype(str)
network_jao_df['id'] = network_jao_df['id'].astype(str)

if has_pypsa:
    pypsa_df['id'] = pypsa_df['id'].astype(str)
    network_pypsa_df['id'] = network_pypsa_df['id'].astype(str)

# Prepare merged dataframes for analysis
jao_merged_df = jao_df.copy()
jao_merged_df['r_original'] = jao_merged_df['r']
jao_merged_df['x_original'] = jao_merged_df['x']
jao_merged_df['b_original'] = jao_merged_df['b']
jao_merged_df['r_allocated'] = network_jao_df['r']
jao_merged_df['x_allocated'] = network_jao_df['x']
jao_merged_df['b_allocated'] = network_jao_df['b']

if has_pypsa:
    pypsa_merged_df = pypsa_df.copy()
    pypsa_merged_df['r_original'] = pypsa_merged_df['r']
    pypsa_merged_df['x_original'] = pypsa_merged_df['x']
    pypsa_merged_df['b_original'] = pypsa_merged_df['b']
    pypsa_merged_df['r_allocated'] = network_pypsa_df['r']
    pypsa_merged_df['x_allocated'] = network_pypsa_df['x']
    pypsa_merged_df['b_allocated'] = network_pypsa_df['b']

# Function to calculate statistics
def calculate_stats(df, param):
    '''Calculate statistics for parameter comparison'''
    original_col = f"{param}_original"
    allocated_col = f"{param}_allocated"

    # Filter out rows where either value is 0 or NaN
    valid_rows = df[(df[original_col] > 0) & (df[allocated_col] > 0)]

    if len(valid_rows) == 0:
        return {
            "count": 0,
            "median_ratio": np.nan,
            "mean_ratio": np.nan,
            "increased": 0,
            "decreased": 0,
            "similar": 0
        }

    # Calculate ratios
    ratios = valid_rows[allocated_col] / valid_rows[original_col]

    # Calculate statistics
    median_ratio = ratios.median()
    mean_ratio = ratios.mean()
    increased = sum(ratios > 1.1)
    decreased = sum(ratios < 0.9)
    similar = sum((ratios >= 0.9) & (ratios <= 1.1))

    return {
        "count": len(valid_rows),
        "median_ratio": median_ratio,
        "mean_ratio": mean_ratio,
        "increased": increased,
        "decreased": decreased,
        "similar": similar,
        "min_ratio": ratios.min(),
        "max_ratio": ratios.max()
    }

# Calculate statistics for each parameter and dataset
jao_stats = {}
pypsa_stats = {}

for param in ['r', 'x', 'b']:
    jao_stats[param] = calculate_stats(jao_merged_df, param)

    print(f"\\nJAO {param.upper()} Statistics:")
    if jao_stats[param]['count'] > 0:
        print(f"  Valid comparisons: {jao_stats[param]['count']}")
        print(f"  Median ratio: {jao_stats[param]['median_ratio']:.2f}")
        print(f"  Mean ratio: {jao_stats[param]['mean_ratio']:.2f}")
        print(f"  Min ratio: {jao_stats[param]['min_ratio']:.2f}")
        print(f"  Max ratio: {jao_stats[param]['max_ratio']:.2f}")
        print(f"  Increased (>10%): {jao_stats[param]['increased']} ({jao_stats[param]['increased'] / jao_stats[param]['count'] * 100:.1f}%)")
        print(f"  Similar (±10%): {jao_stats[param]['similar']} ({jao_stats[param]['similar'] / jao_stats[param]['count'] * 100:.1f}%)")
        print(f"  Decreased (<10%): {jao_stats[param]['decreased']} ({jao_stats[param]['decreased'] / jao_stats[param]['count'] * 100:.1f}%)")
    else:
        print("  No valid data points")

    if has_pypsa:
        pypsa_stats[param] = calculate_stats(pypsa_merged_df, param)
        print(f"\\nPyPSA {param.upper()} Statistics:")
        if pypsa_stats[param]['count'] > 0:
            print(f"  Valid comparisons: {pypsa_stats[param]['count']}")
            print(f"  Median ratio: {pypsa_stats[param]['median_ratio']:.2f}")
            print(f"  Mean ratio: {pypsa_stats[param]['mean_ratio']:.2f}")
            print(f"  Min ratio: {pypsa_stats[param]['min_ratio']:.2f}")
            print(f"  Max ratio: {pypsa_stats[param]['max_ratio']:.2f}")
            print(f"  Increased (>10%): {pypsa_stats[param]['increased']} ({pypsa_stats[param]['increased'] / pypsa_stats[param]['count'] * 100:.1f}%)")
            print(f"  Similar (±10%): {pypsa_stats[param]['similar']} ({pypsa_stats[param]['similar'] / pypsa_stats[param]['count'] * 100:.1f}%)")
            print(f"  Decreased (<10%): {pypsa_stats[param]['decreased']} ({pypsa_stats[param]['decreased'] / pypsa_stats[param]['count'] * 100:.1f}%)")
        else:
            print("  No valid data points")

# Define color function for data points
def ratio_to_color(ratio):
    '''Convert ratio to a color for visualization'''
    if ratio > 10:
        return 'rgba(255, 0, 0, 0.7)'  # Red for much larger
    elif ratio > 5:
        return 'rgba(255, 100, 100, 0.7)'  # Light red
    elif ratio > 2:
        return 'rgba(200, 150, 200, 0.7)'  # Purple
    elif 0.8 <= ratio <= 1.25:
        return 'rgba(0, 0, 255, 0.7)'  # Blue for approximately equal
    elif ratio < 0.5:
        return 'rgba(0, 255, 0, 0.7)'  # Green for much smaller
    else:
        return 'rgba(150, 150, 150, 0.7)'  # Gray for other cases

# Create interactive plots
if has_pypsa:
    # Create 2x3 grid for JAO and PyPSA
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=(
            f"JAO Resistance (r) - Median ratio: {jao_stats['r'].get('median_ratio', 'N/A') if not np.isnan(jao_stats['r'].get('median_ratio', np.nan)) else 'N/A'}",
            f"JAO Reactance (x) - Median ratio: {jao_stats['x'].get('median_ratio', 'N/A') if not np.isnan(jao_stats['x'].get('median_ratio', np.nan)) else 'N/A'}",
            f"JAO Susceptance (b) - Median ratio: {jao_stats['b'].get('median_ratio', 'N/A') if not np.isnan(jao_stats['b'].get('median_ratio', np.nan)) else 'N/A'}",
            f"PyPSA Resistance (r) - Median ratio: {pypsa_stats['r'].get('median_ratio', 'N/A') if not np.isnan(pypsa_stats['r'].get('median_ratio', np.nan)) else 'N/A'}",
            f"PyPSA Reactance (x) - Median ratio: {pypsa_stats['x'].get('median_ratio', 'N/A') if not np.isnan(pypsa_stats['x'].get('median_ratio', np.nan)) else 'N/A'}",
            f"PyPSA Susceptance (b) - Median ratio: {pypsa_stats['b'].get('median_ratio', 'N/A') if not np.isnan(pypsa_stats['b'].get('median_ratio', np.nan)) else 'N/A'}"
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
        specs=[
            [{"type": "scatter"}, {"type": "scatter"}, {"type": "scatter"}],
            [{"type": "scatter"}, {"type": "scatter"}, {"type": "scatter"}]
        ]
    )
else:
    # Create 1x3 grid for JAO only
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=(
            f"JAO Resistance (r) - Median ratio: {jao_stats['r'].get('median_ratio', 'N/A') if not np.isnan(jao_stats['r'].get('median_ratio', np.nan)) else 'N/A'}",
            f"JAO Reactance (x) - Median ratio: {jao_stats['x'].get('median_ratio', 'N/A') if not np.isnan(jao_stats['x'].get('median_ratio', np.nan)) else 'N/A'}",
            f"JAO Susceptance (b) - Median ratio: {jao_stats['b'].get('median_ratio', 'N/A') if not np.isnan(jao_stats['b'].get('median_ratio', np.nan)) else 'N/A'}"
        ),
        horizontal_spacing=0.08
    )

param_names = {
    'r': 'Resistance',
    'x': 'Reactance',
    'b': 'Susceptance'
}

# Add data for JAO lines (first row)
for i, param in enumerate(['r', 'x', 'b'], 1):
    original_col = f"{param}_original"
    allocated_col = f"{param}_allocated"

    # Filter valid data points for JAO
    valid_data = jao_merged_df[(jao_merged_df[original_col] > 0) & (jao_merged_df[allocated_col] > 0)]

    if len(valid_data) > 0:
        # Calculate ratios for coloring
        ratios = valid_data[allocated_col] / valid_data[original_col]
        colors = [ratio_to_color(r) for r in ratios]

        # Add scatter plot for JAO
        fig.add_trace(
            go.Scatter(
                x=valid_data[original_col],
                y=valid_data[allocated_col],
                mode='markers',
                marker=dict(
                    size=8,
                    color=colors,
                ),
                text=[
                    f"ID: {id}<br>Original: {orig:.6e}<br>Allocated: {alloc:.6e}<br>Ratio: {ratio:.2f}<br>Voltage: {v_nom} kV"
                    for id, orig, alloc, ratio, v_nom in zip(
                        valid_data['id'],
                        valid_data[original_col],
                        valid_data[allocated_col],
                        ratios,
                        valid_data['v_nom']
                    )],
                hoverinfo='text',
                name=f"JAO {param_names[param]} Data",
                showlegend=(i == 1)  # Only show legend for first parameter
            ),
            row=1, col=i
        )

        # Add equality line (y=x) for JAO
        min_val = min(valid_data[original_col].min(), valid_data[allocated_col].min())
        max_val = max(valid_data[original_col].max(), valid_data[allocated_col].max())

        # Extend the line a bit beyond the data range
        min_line = min_val * 0.1
        max_line = max_val * 10

        fig.add_trace(
            go.Scatter(
                x=[min_line, max_line],
                y=[min_line, max_line],
                mode='lines',
                line=dict(color='red', dash='dash', width=2),
                name='Equal values (y = x)',
                showlegend=(i == 1)  # Only show in legend once
            ),
            row=1, col=i
        )

        # Configure axes to be logarithmic for JAO
        fig.update_xaxes(
            title_text=f"Original {param} value",
            type="log",
            row=1, col=i
        )

        fig.update_yaxes(
            title_text=f"Allocated {param} value",
            type="log",
            row=1, col=i
        )
    else:
        # Add empty trace if no data
        fig.add_trace(
            go.Scatter(
                x=[],
                y=[],
                mode='markers',
                name=f"No valid JAO {param_names[param]} data",
                showlegend=(i == 1)
            ),
            row=1, col=i
        )

# Add data for PyPSA lines (second row) if available
if has_pypsa:
    for i, param in enumerate(['r', 'x', 'b'], 1):
        original_col = f"{param}_original"
        allocated_col = f"{param}_allocated"

        # Filter valid data points for PyPSA
        valid_data = pypsa_merged_df[(pypsa_merged_df[original_col] > 0) & (pypsa_merged_df[allocated_col] > 0)]

        if len(valid_data) > 0:
            # Calculate ratios for coloring
            ratios = valid_data[allocated_col] / valid_data[original_col]
            colors = [ratio_to_color(r) for r in ratios]

            # Add scatter plot for PyPSA
            fig.add_trace(
                go.Scatter(
                    x=valid_data[original_col],
                    y=valid_data[allocated_col],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=colors,
                    ),
                    text=[
                        f"ID: {id}<br>Original: {orig:.6e}<br>Allocated: {alloc:.6e}<br>Ratio: {ratio:.2f}<br>Voltage: {v_nom} kV"
                        for id, orig, alloc, ratio, v_nom in zip(
                            valid_data['id'],
                            valid_data[original_col],
                            valid_data[allocated_col],
                            ratios,
                            valid_data['v_nom']
                        )],
                    hoverinfo='text',
                    name=f"PyPSA {param_names[param]} Data",
                    showlegend=False  # Don't show in legend since JAO already shows it
                ),
                row=2, col=i
            )

            # Add equality line (y=x) for PyPSA
            min_val = min(valid_data[original_col].min(), valid_data[allocated_col].min())
            max_val = max(valid_data[original_col].max(), valid_data[allocated_col].max())

            # Extend the line a bit beyond the data range
            min_line = min_val * 0.1
            max_line = max_val * 10

            fig.add_trace(
                go.Scatter(
                    x=[min_line, max_line],
                    y=[min_line, max_line],
                    mode='lines',
                    line=dict(color='red', dash='dash', width=2),
                    name='Equal values (y = x)',
                    showlegend=False  # Don't show in legend since JAO already shows it
                ),
                row=2, col=i
            )

            # Configure axes to be logarithmic for PyPSA
            fig.update_xaxes(
                title_text=f"Original {param} value",
                type="log",
                row=2, col=i
            )

            fig.update_yaxes(
                title_text=f"Allocated {param} value",
                type="log",
                row=2, col=i
            )
        else:
            # Add empty trace if no data
            fig.add_trace(
                go.Scatter(
                    x=[],
                    y=[],
                    mode='markers',
                    name=f"No valid PyPSA {param_names[param]} data",
                    showlegend=False
                ),
                row=2, col=i
            )

# Update layout
if has_pypsa:
    title = "JAO and PyPSA Parameter Comparison (log scale)"
    height = 1000
else:
    title = "JAO Parameter Comparison (log scale)"
    height = 600

fig.update_layout(
    title_text=title,
    height=height,
    width=1500,
    template="plotly_white",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    )
)

# Add color scale legend as a separate trace
legend_items = [
    ("Ratio > 10x", 'rgba(255, 0, 0, 0.7)'),
    ("Ratio 5-10x", 'rgba(255, 100, 100, 0.7)'),
    ("Ratio 2-5x", 'rgba(200, 150, 200, 0.7)'),
    ("Ratio 0.8-1.25x (similar)", 'rgba(0, 0, 255, 0.7)'),
    ("Ratio < 0.5x", 'rgba(0, 255, 0, 0.7)'),
    ("Other ratios", 'rgba(150, 150, 150, 0.7)')
]

for label, color in legend_items:
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode='markers',
            marker=dict(size=10, color=color),
            name=label
        )
    )

# Add annotation explaining the charts
fig.add_annotation(
    xref="paper",
    yref="paper",
    x=0.5,
    y=-0.05,
    text="These charts compare original parameter values (x-axis) with allocated values (y-axis). " +
         "Points close to the red dashed line have similar original and allocated values. " +
         "Points are colored based on the ratio of allocated/original values.",
    showarrow=False,
    font=dict(size=12)
)

# Save the figure
output_path = "parameter_comparison.html"
fig.write_html(output_path)
print(f"\\nComparison chart saved to: {output_path}")

try:
    image_path = "parameter_comparison.png"
    fig.write_image(image_path, scale=2)
    print(f"Image saved to: {image_path}")
except Exception as e:
    print(f"Could not save image (requires kaleido package): {e}")
    print("You can install it with: pip install kaleido")

# Print any high ratio outliers for both datasets
print("\\nHighest ratio outliers:")
for dataset_name, dataset in [("JAO", jao_merged_df)] + ([("PyPSA", pypsa_merged_df)] if has_pypsa else []):
    for param in ['r', 'x', 'b']:
        original_col = f"{param}_original"
        allocated_col = f"{param}_allocated"

        # Filter valid data points
        valid_data = dataset[(dataset[original_col] > 0) & (dataset[allocated_col] > 0)].copy()

        if len(valid_data) == 0:
            print(f"No valid data for {dataset_name} {param}")
            continue

        # Calculate ratios
        valid_data['ratio'] = valid_data[allocated_col] / valid_data[original_col]

        # Get top 5 outliers
        outliers = valid_data.nlargest(5, 'ratio')

        print(f"\\nTop 5 {dataset_name} {param.upper()} ratio outliers (Allocated/Original):")
        for _, row in outliers.iterrows():
            print(
                f"  ID: {row['id']}, Original: {row[original_col]:.6e}, Allocated: {row[allocated_col]:.6e}, " +
                f"Ratio: {row['ratio']:.2f}, Voltage: {row['v_nom']} kV")

        # Also print bottom 5 (where original is much larger than allocated)
        bottom_outliers = valid_data.nsmallest(5, 'ratio')
        print(f"\\nBottom 5 {dataset_name} {param.upper()} ratio outliers (Allocated/Original):")
        for _, row in bottom_outliers.iterrows():
            print(
                f"  ID: {row['id']}, Original: {row[original_col]:.6e}, Allocated: {row[allocated_col]:.6e}, " +
                f"Ratio: {row['ratio']:.2f}, Voltage: {row['v_nom']} kV")
            """)

        # Run the script
        parameter_comparison_file = os.path.join(output_dir, "parameter_comparison.html")
        import subprocess
        try:
            result = subprocess.run(["python", parameter_comparison_script], cwd=output_dir, capture_output=True,
                                    text=True)
            print(result.stdout)
            if result.stderr:
                print(f"Errors: {result.stderr}")

            if os.path.exists(parameter_comparison_file):
                print(f"Parameter comparison visualization saved to {parameter_comparison_file}")
            else:
                print("Warning: Parameter comparison visualization not generated")
        except Exception as e:
            print(f"Error running parameter comparison script: {e}")
    except Exception as e:
        print(f"Error generating parameter comparison visualization: {e}")

    # ---------- VISUALIZATIONS AND EXPORTS ----------
    print("\n=== CREATING VISUALIZATIONS AND EXPORTS ===")

    # Create visualizations with PyPSA integration
    # With this:
    print("Creating enhanced visualization with duplicate handling...")
    duplicate_map_file = visualize_results(
        jao_gdf,
        network_gdf,
        matching_results,  # Use original results without PyPSA for duplicate view
        pypsa_gdf=None  # Don't include PyPSA data in duplicates view
    )

    try:
        with open(duplicate_map_file, 'r', encoding='utf-8') as f:
            html_content = f.read()
        improved_html = improve_visualization_of_unmatched_network_lines(html_content)
        with open(duplicate_map_file, 'w', encoding='utf-8') as f:
            f.write(improved_html)
    except Exception as e:
        print("Warning: could not post-process duplicate map HTML:", e)

    print("Creating regular visualization and summary...")
    map_file = visualize_results(jao_gdf, network_gdf, matching_results, pypsa_gdf)
    try:
        with open(map_file, 'r', encoding='utf-8') as f:
            html_content = f.read()
        improved_html = improve_visualization_of_unmatched_network_lines(html_content)
        with open(map_file, 'w', encoding='utf-8') as f:
            f.write(improved_html)
    except Exception as e:
        print("Warning: could not post-process regular map HTML:", e)

    # Build the parameters HTML with PyPSA integration
    print("Building enhanced HTML with parameters, coverage, and totals checks...")
    enhanced_summary_file = create_enhanced_summary_table(jao_gdf, network_gdf, matching_results, pypsa_gdf)

    # Create combined visualization with PyPSA lines if available
    if pypsa_gdf is not None and pypsa_matching_results:
        print("\nCreating combined visualization with PyPSA lines...")
        combined_map_file = visualize_results_with_path(
            jao_gdf,
            network_gdf,
            combined_results,
            pypsa_gdf,
            output_path=os.path.join(output_dir, "combined_jao_network_pypsa_map.html")
        )

        # Create enhanced summary table with combined results
        print("Creating combined parameters table with PyPSA lines...")
        combined_summary_file = create_enhanced_summary_table_with_pypsa_debug(
            jao_gdf,
            network_gdf,
            combined_results,
            pypsa_gdf,
            output_path=os.path.join(output_dir, "combined_jao_network_pypsa_parameters.html")
        )

        print(f"Combined map with PyPSA lines: {combined_map_file}")
        print(f"Combined parameters table with PyPSA lines: {combined_summary_file}")

    # Create comprehensive visualizations with PyPSA if those functions exist
    print("\n=== CREATING COMPREHENSIVE VISUALIZATIONS WITH PYPSA LINES ===")
    try:
        # If we have PyPSA data and PyPSA integration worked
        if pypsa_gdf is not None and pypsa_matching_results:
            # Import any additional visualization functions
            from pypsa_integration import create_comprehensive_map, create_comprehensive_params_table

            # Check if these functions accept an output_file parameter
            import inspect
            if 'output_file' in inspect.signature(create_comprehensive_map).parameters:
                # Create comprehensive map
                comprehensive_map_file = create_comprehensive_map(
                    jao_gdf, network_gdf, pypsa_gdf, combined_results,
                    output_file=os.path.join(output_dir, "comprehensive_jao_network_pypsa_map.html")
                )
            else:
                # Use default output path
                comprehensive_map_file = create_comprehensive_map(
                    jao_gdf, network_gdf, pypsa_gdf, combined_results
                )
            print(f"Created comprehensive map at: {comprehensive_map_file}")

            # Check if this function accepts an output_file parameter
            if 'output_file' in inspect.signature(create_comprehensive_params_table).parameters:
                # Create comprehensive parameters table
                comprehensive_params_file = create_comprehensive_params_table(
                    jao_gdf, network_gdf, pypsa_gdf, combined_results,
                    output_file=os.path.join(output_dir, "comprehensive_jao_network_pypsa_parameters.html")
                )
            else:
                # Use default output path
                comprehensive_params_file = create_comprehensive_params_table(
                    jao_gdf, network_gdf, pypsa_gdf, combined_results
                )
            print(f"Created comprehensive parameters table at: {comprehensive_params_file}")
    except Exception as e:
        print(f"Error creating comprehensive visualizations: {str(e)}")
        traceback.print_exc()

    # Properly generate and sanitize export rows - JUST ONCE
    if not export_rows:
        print("Warning: No export rows collected. Attempting to regenerate from matching results...")
        export_rows = []
        for result in matching_results:
            if result.get('matched') and result.get('matched_lines_data'):
                for segment in result['matched_lines_data']:
                    if 'network_id' in segment and 'length_km' in segment:
                        export_rows.append({
                            'network_id': segment['network_id'],
                            'seg_len_km': segment['length_km'],
                            'jao_r_km_pc': segment.get('jao_r_per_km_pc', None),
                            'jao_x_km_pc': segment.get('jao_x_per_km_pc', None),
                            'jao_b_km_pc': segment.get('jao_b_per_km_pc', None)
                        })

    # Export detailed allocation CSV
    alloc_detail_csv = os.path.join(output_dir, "allocation_details.csv")
    export_allocation_details_csv(
        export_rows=export_rows,
        out_csv_path=alloc_detail_csv
    )

    # Export PyPSA-ready file (per-circuit totals written into r/x/b/g, columns preserved)
    pypsa_lines_csv = os.path.join(output_dir, "network_lines_pypsa.csv")
    export_ready_lines_csv(  # Changed from export_pypsa_ready_lines_csv
        lines_df=network_gdf.copy(),
        export_rows=export_rows,
        out_csv_path=pypsa_lines_csv,
        source_type="JAO",  # Added source_type parameter
        use_fallback=True,
        style="corridor_like_original"
    )

    # Also export a version with per-circuit parameters
    pypsa_lines_per_circuit_csv = os.path.join(output_dir, "network_lines_pypsa_per_circuit.csv")
    export_ready_lines_csv(  # Changed from export_pypsa_ready_lines_csv
        lines_df=network_gdf.copy(),
        export_rows=export_rows,
        out_csv_path=pypsa_lines_per_circuit_csv,
        source_type="JAO",  # Added source_type parameter
        use_fallback=True,
        style="per_circuit"
    )

    # Updated network lines with JAO params
    updated_network_csv = os.path.join(output_dir, "network_lines_updated.csv")
    export_updated_network_lines_csv(
        lines_df=network_gdf.copy(),
        export_rows=export_rows,
        out_csv_path=updated_network_csv
    )

    # ---------- SUMMARY STATS ----------
    total_jao_lines = len(matching_results)
    matched_lines = sum(1 for r in matching_results if r.get('matched'))
    duplicate_count = sum(1 for r in matching_results if r.get('is_duplicate', False))
    print(f"JAO lines marked as duplicates: {duplicate_count}")
    parallel_count = sum(1 for r in matching_results if r.get('is_parallel_circuit', False))
    geometric_count = sum(1 for r in matching_results if r.get('is_geometric_match', False))
    parallel_voltage_count = sum(1 for r in matching_results if r.get('is_parallel_voltage_circuit', False))
    regular_matches = matched_lines - duplicate_count - parallel_count - geometric_count - parallel_voltage_count

    # Calculate final match percentages
    print(f"\n=== FINAL STATISTICS ===")
    print(f"Total JAO lines: {total_jao_lines}")
    print(f"Matched lines: {matched_lines} ({matched_lines / total_jao_lines * 100:.1f}%)")
    print(f"  - Regular matches: {regular_matches} ({regular_matches / total_jao_lines * 100:.1f}%)")
    print(f"  - Geometric matches: {geometric_count} ({geometric_count / total_jao_lines * 100:.1f}%)")
    print(f"  - Duplicate JAO lines: {duplicate_count} ({duplicate_count / total_jao_lines * 100:.1f}%)")
    print(f"  - Parallel circuit JAO lines: {parallel_count} ({parallel_count / total_jao_lines * 100:.1f}%)")
    print(
        f"  - Parallel voltage circuit JAO lines: {parallel_voltage_count} ({parallel_voltage_count / total_jao_lines * 100:.1f}%)")
    print(
        f"Unmatched lines: {total_jao_lines - matched_lines} ({(total_jao_lines - matched_lines) / total_jao_lines * 100:.1f}%)")

    v220_lines = sum(1 for r in matching_results if r.get('v_nom') == 220)
    v220_matched = sum(1 for r in matching_results if r.get('v_nom') == 220 and r.get('matched'))
    v400_lines = sum(1 for r in matching_results if r.get('v_nom') in [380, 400])

    v400_matched = sum(1 for r in matching_results if r.get('v_nom') in [380, 400] and r.get('matched'))
    print("\nStatistics by voltage level:")
    print(
        f"220 kV lines: {v220_matched}/{v220_lines} matched ({(v220_matched / v220_lines * 100) if v220_lines else 0.0:.1f}%)")
    print(
        f"400 kV lines: {v400_matched}/{v400_lines} matched ({(v400_matched / v400_lines * 100) if v400_lines else 0.0:.1f}%)")

    total_network_lines = len(network_gdf)
    matched_network_ids = set()
    for r in matching_results:
        if r.get('matched') and r.get('network_ids'):
            for nid in r['network_ids']:
                matched_network_ids.add(str(nid))
    matched_network_count = len(matched_network_ids)
    unmatched_network_count = total_network_lines - matched_network_count
    print(f"\nNetwork Lines Statistics:")
    print(f"Total Network Lines: {total_network_lines}")
    print(f"Matched Network Lines: {matched_network_count} ({matched_network_count / total_network_lines * 100:.1f}%)")
    print(
        f"Unmatched Network Lines: {unmatched_network_count} ({unmatched_network_count / total_network_lines * 100:.1f}%)")

    n220_lines = sum(1 for _, row in network_gdf.iterrows() if row.get('v_nom') == 220)
    n220_matched = sum(1 for _, row in network_gdf.iterrows()
                       if row.get('v_nom') == 220 and str(row.get('id')) in matched_network_ids)

    n400_lines = sum(1 for _, row in network_gdf.iterrows() if row.get('v_nom') in [380, 400])
    n400_matched = sum(1 for _, row in network_gdf.iterrows() if
                       row.get('v_nom') in [380, 400] and str(row.get('id')) in matched_network_ids)
    print(
        f"220 kV network lines: {n220_matched}/{n220_lines} matched ({(n220_matched / n220_lines * 100) if n220_lines else 0.0:.1f}%)")
    print(
        f"400 kV network lines: {n400_matched}/{n400_lines} matched ({(n400_matched / n400_lines * 100) if n400_lines else 0.0:.1f}%)")

    # Add PyPSA statistics if available
    if pypsa_gdf is not None:
        print("\nPyPSA Lines Statistics:")
        print(f"Total PyPSA Lines: {len(pypsa_gdf)}")
        pypsa_220kv_count = len(pypsa_gdf[pypsa_gdf['voltage'].between(200, 300)])
        pypsa_400kv_count = len(pypsa_gdf[pypsa_gdf['voltage'] >= 300])
        print(f"220 kV PyPSA lines: {pypsa_220kv_count}")
        print(f"400 kV PyPSA lines: {pypsa_400kv_count}")

        # If we have PyPSA matching results, show those statistics too
        if pypsa_matching_results:
            pypsa_matched_count = sum(1 for r in pypsa_matching_results if r.get('matched', False))
            print(
                f"Matched PyPSA lines: {pypsa_matched_count}/{len(pypsa_gdf)} ({pypsa_matched_count / len(pypsa_gdf) * 100:.1f}%)")

    # Results location summary
    print("\nResults saved to:")
    print(f"  - Regular Map: {map_file}")
    print(f"  - Map with Duplicate Handling: {duplicate_map_file}")
    print(f"  - Enhanced Summary with Parameters: {enhanced_summary_file}")
    print(
        f"  - Parameter Comparison: {parameter_comparison_file if 'parameter_comparison_file' in locals() else 'Not generated'}")
    print(f"  - JAO Allocation Details: {jao_allocation_path}")
    if 'pypsa_allocation_path' in locals():
        print(f"  - PyPSA Allocation Details: {pypsa_allocation_path}")
    print(f"  - Network Lines with JAO Params: {jao_network_path}")
    if 'pypsa_network_path' in locals():
        print(f"  - Network Lines with PyPSA Params: {pypsa_network_path}")
    print(f"  - JAO Lines Ready: {jao_ready_path}")
    if 'pypsa_ready_path' in locals():
        print(f"  - PyPSA Lines Ready: {pypsa_ready_path}")

    if pypsa_gdf is not None and 'combined_map_file' in locals():
        print(f"  - Combined Map with PyPSA: {combined_map_file}")
        print(f"  - Combined Parameters with PyPSA: {combined_summary_file}")
    if 'comprehensive_map_file' in locals():
        print(f"  - Comprehensive PyPSA Map: {comprehensive_map_file}")
        print(f"  - Comprehensive PyPSA Parameters: {comprehensive_params_file}")
    print(f"  - Allocation Details: {alloc_detail_csv}")
    print(f"  - PyPSA Lines (Corridor): {pypsa_lines_csv}")
    print(f"  - PyPSA Lines (Per-Circuit): {pypsa_lines_per_circuit_csv}")
    print(f"  - Updated Network Lines: {updated_network_csv}")

    # Report total execution time
    elapsed = time() - start_time
    hours, remainder = divmod(elapsed, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"\nTotal execution time: {int(hours)}h {int(minutes)}m {seconds:.1f}s")


if __name__ == "__main__":
    main()