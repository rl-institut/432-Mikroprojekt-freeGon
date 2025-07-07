from __future__ import annotations

# ──────────────────────────────────────────────────────────────
#  src/matching/hybrid_matching.py
#  (complete replacement – paste the whole file)
# ──────────────────────────────────────────────────────────────

import math, logging
from typing import Dict, Tuple, List

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from shapely.geometry import LineString, MultiLineString, Point
from shapely.ops import linemerge, unary_union
from shapely.strtree import STRtree



# -------------------------------------------------------------
#  global parameters – can be overridden per-call
# -------------------------------------------------------------
METRIC_CRS        = 3035          # metres everywhere (ETRS-LAEA Europe)
TOL               = 600           # half corridor width for overlap [m]
SNAP_M            = 1_200         # stroke-merger snap distance   [m]
MIN_DIR_COS       = .80           # direction similarity threshold
OVERLAP_MIN_FRAC  = .10           # keep if ≥ 10 % of shorter line overlap
OVERLAP_MIN_ABS_M = 2_000         # …or ≥ 2 km absolute
MAX_OVERRUN       = 1.15          # allocate ≤ 115 % of DLR length
LOGGER = logging.getLogger(__name__)



# ── voltage comparison ----------------------------------------
V_EQ_TOL = 25           # kV – treat values within ±25 kV as equal


def _same_voltage(a: float, b: float) -> bool:
    """Return True if *a* and *b* are considered the same voltage."""
    if a == 0 or b == 0:  # If either is unknown/zero, don't compare
        return True

    return (
            abs(a - b) <= V_EQ_TOL  # 225 kV ≈ 220 kV, etc.
            or abs(a / b - 1.0) <= 0.1  # Within 10% of each other
            or {a, b} == {380, 400}  # explicit 380/400 rule
    )

# -------------------------------------------------------------
#  utilities – CRS, vectors, metrics
# -------------------------------------------------------------
def _to_metric(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.crs is None or gdf.crs.is_geographic:
        return gdf.to_crs(METRIC_CRS)
    return gdf

def _main_vec(ls: LineString) -> np.ndarray:
    if ls.geom_type == "MultiLineString":
        ls = max(ls.geoms, key=lambda p: p.length)
    c = np.asarray(ls.coords)
    v = c[-1] - c[0]
    n = np.linalg.norm(v)
    return v / n if n else v

def _dir_cos(a: LineString, b: LineString) -> float:
    return float(abs(np.dot(_main_vec(a), _main_vec(b))))

def _endpts(g) -> list[Point]:
    if g is None or g.is_empty:
        return []
    if g.geom_type == "LineString":
        c = list(g.coords); return [Point(c[0]), Point(c[-1])]
    out = []
    for part in g.geoms:
        c = list(part.coords); out += [Point(c[0]), Point(c[-1])]
    return out

def _endpts_inside(src: LineString, cand: LineString, width: float) -> bool:
    corr = src.buffer(width)
    return all(corr.contains(pt) for pt in _endpts(cand))

def _overlap_km(a: LineString, b: LineString) -> float:
    inter = b.intersection(a.buffer(TOL))
    return inter.length / 1_000

def _frechet(a: LineString, b: LineString) -> float:
    try:
        return float(a.frechet_distance(b))
    except Exception:                       # Shapely <2 → graceful fallback
        return float(a.hausdorff_distance(b))

def _hausdorff(a: LineString, b: LineString) -> float:
    return float(a.hausdorff_distance(b))

# helper – unique key for endpoints (≈ 10-m grid avoids numeric noise)
def _snap_key(pt: Point, grid: float = 1e-4):        # ~11 m at mid-lat
    return (round(pt.x / grid), round(pt.y / grid))


# -------------------------------------------------------------
#  stroke-merging on the network side
# -------------------------------------------------------------
def _merge_strokes(net: gpd.GeoDataFrame,
                   snap_m: float = SNAP_M,
                   min_cos: float = MIN_DIR_COS) -> gpd.GeoDataFrame:
    """
    Merge end-to-end network segments that touch within *snap_m*
    and share a similar direction (+ equal voltage).
    Returns a *copy* of *net*.
    """
    if net.empty:
        return net

    net = net.reset_index(drop=True).copy()
    sindex = STRtree(net.geometry.values)
    G = nx.Graph()                             # <- keep one graph here

    MAX_MERGE_ANGLE = 30 * np.pi / 180  # ≤ 30 ° bend is OK

    def _angle(a: LineString, b: LineString) -> float:
        return np.arccos(_dir_cos(a, b))  # 0…π

    # --------- FIXED LINES ------------------------------------------------
    for idx, geom in enumerate(net.geometry):
        for p in _endpts(geom):
            for j in sindex.query(p.buffer(snap_m)):
                if j == idx or G.has_edge(idx, j):
                    continue
                # ‼ same voltage as before
                if net.at[idx, "v_nom"] != net.at[j, "v_nom"]:
                    continue
                # ‼ NEW: forbid sharp bends
                if _angle(geom, net.geometry.iloc[j]) > MAX_MERGE_ANGLE:
                    continue
                if any(p.distance(q) <= snap_m for q in _endpts(net.geometry.iloc[j])):
                    G.add_edge(idx, j)
    # ---------------------------------------------------------------------

    merged_rows, visited = [], set()
    for cid, comp in enumerate(nx.connected_components(G), 1):
        if len(comp) == 1:
            continue
        geoms   = [net.geometry.iloc[i] for i in comp]
        merged  = linemerge(unary_union(geoms))
        merged_rows.append(
            dict(
                id       = f"stroke_{cid}",
                geometry = merged,
                v_nom    = net.at[next(iter(comp)), "v_nom"],
                src_ids  = ",".join(str(net.at[i, "id"]) for i in comp),
            )
        )
        visited.update(comp)

    leftover = net.loc[[i for i in net.index if i not in visited]].copy()
    return pd.concat([leftover,
                      gpd.GeoDataFrame(merged_rows, crs=net.crs)],
                     ignore_index=True)


# ──────────────────────────────────────────────────────────────
#  Same-voltage end-point graph
# ──────────────────────────────────────────────────────────────
# -------------------------------------------------------------
def _build_endpoint_graph(net: gpd.GeoDataFrame) -> nx.MultiGraph:
    """
    Build a graph where nodes are the endpoints of network segments with
    additional connections between nearby endpoints to handle small gaps.
    """
    # Ensure data is in metric CRS
    net_m = _to_metric(net)
    G = nx.MultiGraph()

    # Add edges for all network lines
    for idx, row in net_m.iterrows():
        ls = row.geometry
        if ls is None or ls.is_empty:
            continue

        # Extract endpoints
        endpoints = []

        if ls.geom_type == "LineString":
            coords = list(ls.coords)
            if len(coords) < 2:
                continue
            endpoints = [coords[0], coords[-1]]
        elif ls.geom_type == "MultiLineString":
            merged = linemerge(ls)
            if merged.geom_type == "LineString":
                coords = list(merged.coords)
                if len(coords) < 2:
                    continue
                endpoints = [coords[0], coords[-1]]
            else:
                # Handle disconnected MultiLineString
                parts = list(merged.geoms)
                if not parts:
                    continue

                first = min(parts, key=lambda g: g.bounds[0])
                last = max(parts, key=lambda g: g.bounds[2])
                endpoints = [list(first.coords)[0], list(last.coords)[-1]]

        if len(endpoints) != 2:
            continue

        # Convert tuple coords to standard format
        start_point = tuple(map(float, endpoints[0]))
        end_point = tuple(map(float, endpoints[-1]))

        # Add the edge with metadata
        G.add_edge(
            start_point, end_point,
            idx=idx,
            v_nom=float(row.get("v_nom", 0.0) or 0.0),
            length=float(ls.length),
            id=row.get("id", f"line_{idx}")
        )

    # Add virtual edges to connect nearby endpoints (to handle small gaps)
    # This is crucial for finding paths through the network
    nodes = list(G.nodes())

    # Build spatial index for efficient nearest neighbor search
    node_points = [Point(node) for node in nodes]
    tree = STRtree(node_points)

    # Connect nearby endpoints with virtual edges
    connect_radius = 300.0  # meters - smaller than snap_tolerance to avoid excessive edges

    for i, node in enumerate(nodes):
        pt = Point(node)
        nearby_idx = list(tree.query(pt.buffer(connect_radius)))

        for j in nearby_idx:
            if i == j:  # Skip self
                continue

            other_node = nodes[j]

            # Only add virtual edge if not already directly connected
            if not G.has_edge(node, other_node):
                # Measure exact distance
                distance = pt.distance(Point(other_node))
                if distance <= connect_radius:
                    G.add_edge(
                        node, other_node,
                        idx=-1,  # Use -1 to mark as virtual
                        v_nom=0.0,  # No voltage for virtual edges
                        length=distance,
                        virtual=True
                    )

    LOGGER.info(f"Built network graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    LOGGER.info(f"Graph has {len(list(nx.connected_components(G)))} connected components")

    return G


def _walk_same_voltage(
        dlr_row,
        net: gpd.GeoDataFrame,
        G: nx.MultiGraph,
        hop_cap: int = 20,  # Increased to allow longer paths
        snap_tolerance: float = 3000.0  # Increased for better endpoint matching
) -> list[int]:
    """
    Find the shortest path in the network graph that connects the endpoints of a DLR line
    with matching voltage level.

    Returns a list of network line indices that form the path.
    """
    try:
        # Extract DLR line information
        ls_dlr = dlr_row.geometry
        if ls_dlr is None or ls_dlr.is_empty:
            return []

        v_target = float(dlr_row.v_nom or 0.0)

        # Get DLR endpoints
        start_point, end_point = None, None

        if ls_dlr.geom_type == "LineString":
            start_point = Point(ls_dlr.coords[0])
            end_point = Point(ls_dlr.coords[-1])
        elif ls_dlr.geom_type == "MultiLineString":
            merged = linemerge(ls_dlr)
            if merged.geom_type == "LineString":
                start_point = Point(merged.coords[0])
                end_point = Point(merged.coords[-1])
            else:
                # Handle disconnected MultiLineString
                first = min(merged.geoms, key=lambda g: g.bounds[0])
                last = max(merged.geoms, key=lambda g: g.bounds[2])
                start_point = Point(first.coords[0])
                end_point = Point(last.coords[-1])

        if start_point is None or end_point is None:
            return []

        # Find all network nodes within tolerance of the DLR endpoints
        nodes = list(G.nodes())

        # Calculate distances from start point to all nodes
        start_distances = [(node, start_point.distance(Point(node))) for node in nodes]
        # Sort by distance
        start_distances.sort(key=lambda x: x[1])

        # Calculate distances from end point to all nodes
        end_distances = [(node, end_point.distance(Point(node))) for node in nodes]
        # Sort by distance
        end_distances.sort(key=lambda x: x[1])

        # Get a reasonable number of potential start and end nodes
        max_candidates = 10  # Try up to 10 closest points at each end

        start_nodes = [node for node, dist in start_distances[:max_candidates]
                       if dist <= snap_tolerance]
        end_nodes = [node for node, dist in end_distances[:max_candidates]
                     if dist <= snap_tolerance]

        if not start_nodes or not end_nodes:
            return []

        # Try to find the shortest path between any start-end pair
        best_path = None
        best_path_length = float('inf')  # To track the shortest path

        for start_node in start_nodes:
            for end_node in end_nodes:
                if start_node == end_node:
                    continue  # Skip self-loops

                try:
                    # Use NetworkX's built-in shortest path algorithm
                    path = nx.shortest_path(G, start_node, end_node)

                    # Extract edges along this path
                    edges_on_path = []

                    for i in range(len(path) - 1):
                        u, v = path[i], path[i + 1]
                        # Get the best edge between these nodes
                        best_edge_data = None
                        best_edge_match = False

                        # FIXED: Correctly iterate over all edges between u and v
                        if G.has_edge(u, v):
                            # Get all edges between u and v
                            for edge_key in G[u][v]:
                                edge_data = G[u][v][edge_key]

                                idx = edge_data.get("idx", -1)
                                if idx < 0:  # Skip virtual edges for now
                                    continue

                                if idx >= len(net):  # Skip invalid indices
                                    continue

                                # Check voltage match
                                try:
                                    edge_voltage = float(net.iloc[idx].get("v_nom", 0.0) or 0.0)
                                    is_match = _same_voltage(edge_voltage, v_target)

                                    # Prefer voltage-matching edges
                                    if is_match or best_edge_data is None:
                                        best_edge_data = edge_data
                                        best_edge_match = is_match
                                        if is_match:  # If we found a matching edge, break
                                            break
                                except Exception as e:
                                    continue

                        # If we found an edge between these nodes, add it to our path
                        if best_edge_data is not None:
                            edges_on_path.append(best_edge_data.get("idx", -1))

                    # Filter out any virtual edges or invalid indices
                    valid_edges = [idx for idx in edges_on_path if idx >= 0 and idx < len(net)]

                    # Check if this is a valid path
                    if valid_edges:
                        # Calculate the total length of this path
                        try:
                            path_length = sum(net.iloc[idx].geometry.length for idx in valid_edges)

                            # Only update if this path is shorter than our current best
                            if path_length < best_path_length:
                                best_path = valid_edges
                                best_path_length = path_length
                        except Exception as e:
                            # If we can't calculate length, use the number of segments
                            if len(valid_edges) < best_path_length:
                                best_path = valid_edges
                                best_path_length = len(valid_edges)
                except nx.NetworkXNoPath:
                    # No path exists between these nodes
                    continue
                except Exception as e:
                    LOGGER.error(f"Error finding path for DLR {dlr_row.id}: {e}")
                    continue

        if best_path:
            return best_path
        else:
            return []

    except Exception as e:
        LOGGER.error(f"Error in _walk_same_voltage for DLR {dlr_row.id}: {e}")
        return []



# -------------------------------------------------------------
#  chords – straight-line proxies for every network line
# -------------------------------------------------------------
def _as_chord(ls):
    """Return a straight two-point LineString connecting first↔last vertex."""
    if ls is None or ls.is_empty:
        return ls
    if ls.geom_type == "MultiLineString":
        merged = linemerge(ls)
        if merged.geom_type == "LineString":
            ls = merged
        else:                                # still multi – take extremes
            first = min(merged.geoms, key=lambda g: g.bounds[0])
            last  = max(merged.geoms, key=lambda g: g.bounds[2])
            return LineString([list(first.coords)[0],
                               list(last.coords)[-1]])
    c = list(ls.coords)
    return LineString([c[0], c[-1]])

from shapely.geometry import GeometryCollection
from shapely.geometry.base import BaseGeometry           # add at top of file

def _safe_linemerge(parts):
    """
    Robust union-and-merge that never passes non-geometry objects to Shapely.
    Accepts *any* iterable of geometries (list, tuple, generator).
    """
    parts = [
        g for g in list(parts)           # force generator → list
        if isinstance(g, BaseGeometry) and g is not None and not g.is_empty
    ]
    if not parts:
        return GeometryCollection()

    merged = unary_union(parts)

    if merged.geom_type == "LineString":
        return merged
    if merged.geom_type == "GeometryCollection":
        merged = [
            g for g in merged.geoms
            if g.geom_type in ("LineString", "MultiLineString")
        ]
    return linemerge(merged)



# -------------------------------------------------------------
#  similarity & scoring (real vs. chord)
# -------------------------------------------------------------
def _sim_real(src: LineString, cand: LineString) -> dict:
    """Metrics for *real* geometries."""
    overlap_km = _overlap_km(src, cand)
    shorter    = min(src.length, cand.length) / 1_000
    overlap    = overlap_km / shorter if shorter else 0.0        # 0…1
    L          = max(src.length, 1.0)        # avoid /0
    return dict(
        overlap    = overlap,
        frechet    = _frechet(src, cand),
        hausdorff  = _hausdorff(src, cand),
        direction  = _dir_cos(src, cand),
        endpoints  = int(_endpts_inside(src, cand, TOL*1.5)),
        overlap_km = overlap_km
    )

def _score_real(m: dict, L: float) -> float:
    f_norm  = math.exp(-m["frechet"]   / L)     # 0 … 1
    h_norm  = math.exp(-m["hausdorff"] / L)
    return (0.40 * m["overlap"] +
            0.10 * f_norm +
            0.5 * h_norm +
            0.35 * m["direction"] +
            0.10 * m["endpoints"])

def _sim_chord(src: LineString, cand: LineString) -> dict:
    """Metrics for straight-line chords (no overlap possible)."""
    return dict(
        overlap    = 0.0,
        frechet    = _frechet(src, cand),
        hausdorff  = _hausdorff(src, cand),
        direction  = _dir_cos(src, cand),
        endpoints  = int(_endpts_inside(src, cand, TOL*1.5)),
        overlap_km = 0.0
    )

def _score_chord(m: dict) -> float:
    return 0.75 * m["endpoints"] + 0.25 * m["direction"]


def _first_last_keys(ls: LineString | MultiLineString) -> tuple[tuple, tuple]:
    """Return the actual coordinates of the extreme vertices of *ls*."""
    if ls.geom_type == "MultiLineString":
        ls = linemerge(ls)
        if ls.geom_type != "LineString":
            # Still MultiLineString - use extremes
            first = min(ls.geoms, key=lambda g: g.bounds[0])
            last = max(ls.geoms, key=lambda g: g.bounds[2])
            return tuple(list(first.coords)[0]), tuple(list(last.coords)[-1])

    coords = list(ls.coords)
    return tuple(coords[0]), tuple(coords[-1])


# -------------------------------------------------------------
#  core matcher – one geometry flavour at a time
# -------------------------------------------------------------
def _match(
        dlr: gpd.GeoDataFrame,
        cand: gpd.GeoDataFrame,
        *,
        buf_m     : float,
        dir_thr   : float,
        enforceV  : bool,
        use_end   : bool,
        mode      : str         # 'real' | 'chord'
) -> Tuple[List[dict], Dict[str, str]]:

    sim_fn   = _sim_real   if mode == "real"  else _sim_chord
    score_fn = _score_real if mode == "real"  else _score_chord

    # --- spatial index for quick seed lookup (as before) ------------------
    tree = STRtree(cand.geometry.values)

    # --- NEW: topo-graph for path assembly --------------------------------
    G_topo = nx.Graph()

    for idx, geom in enumerate(cand.geometry):
        pts = _endpts(geom)
        if len(pts) < 2:
            continue
        a_key = _snap_key(pts[0])
        b_key = _snap_key(pts[-1])

        # ── NEW: accumulate *all* segment indices between the same endpoints
        if G_topo.has_edge(a_key, b_key):
            G_topo[a_key][b_key]['seg_idx'].append(idx)
        else:
            G_topo.add_edge(a_key, b_key, seg_idx=[idx])

    rows, best_map = [], {}

    for _, s in dlr.iterrows():
        s_geom = s.geometry
        if s_geom is None or s_geom.is_empty:
            continue
        s_len   = max(s_geom.length, 1.0)                 # metres
        cand_idx = list(map(int, tree.query(s_geom.buffer(buf_m))))
        if not cand_idx:
            continue

        # ── coarse filters ──────────────────────────────────────────
        keep = []
        for i in cand_idx:
            g = cand.geometry.iloc[i]                     # ← .iloc !
            if enforceV:
                sv, nv = s.v_nom, cand.v_nom.iloc[i]      # ← .iloc !
                if enforceV:
                    sv, nv = s.v_nom, cand.v_nom.iloc[i]
                    if sv and nv and not _same_voltage(sv, nv):
                        continue
            if _dir_cos(s_geom, g) < dir_thr:
                continue
            if use_end and not _endpts_inside(s_geom, g, buf_m):
                continue
            keep.append(i)
        if not keep:
            continue

        # ----------------------------------------------------------------------
        # (A) pick the seed with the largest overlap among `keep`
        def _cum_ovl(seed_idx: int) -> float:
            a_key, b_key = _first_last_keys(cand.geometry.iloc[seed_idx])

            # keep only those keys that are actually present in G_topo
            start = [k for k in (a_key, b_key) if k in G_topo]
            if not start:  # nothing to explore → fall back
                return _overlap_km(s_geom, cand.geometry.iloc[seed_idx])

            seen, stack = set(start), list(start)
            segs: list[int] = []

            while stack:
                u = stack.pop()
                for v, data in G_topo[u].items():
                    if v in seen:
                        continue
                    # keep only segments that still pass the corridor & direction test
                    ok = [i for i in data["seg_idx"]
                          if cand.geometry.iloc[i].buffer(TOL).intersects(s_geom)
                          and _dir_cos(s_geom, cand.geometry.iloc[i]) >= dir_thr]
                    if not ok:
                        continue
                    seen.add(v)
                    stack.append(v)
                    segs.extend(ok)

            geom = _safe_linemerge([cand.geometry.iloc[i] for i in segs])

            return _overlap_km(s_geom, geom)

        best_seed = max(keep, key=_cum_ovl)



        # (B) initialise path with that single segment
        seed_seg = cand.geometry.iloc[best_seed]
        path_nodes   = _first_last_keys(seed_seg)

        # (C) Dijkstra between the DLR endpoints *in the topo-graph*
        try:
            pt_start, pt_end = _snap_key(_endpts(s_geom)[0]), _snap_key(_endpts(s_geom)[1])
            topo_path = nx.shortest_path(G_topo, pt_start, pt_end)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            topo_path = list(path_nodes)  # fall back: seed only

        # (D) collect all segments along that path
        # (D) collect all segments along that path
        path_seg_idx = []
        for u, v in zip(topo_path[:-1], topo_path[1:]):
            # --- guard against zero-length hops or grid-rounding artefacts
            if u == v or not G_topo.has_edge(u, v):
                continue
            path_seg_idx.extend(G_topo[u][v]["seg_idx"])

        # unique, original order preserved
        path_seg_idx = list(dict.fromkeys(path_seg_idx))

        # fallback – if nothing survived, use the seed only
        if not path_seg_idx:
            path_seg_idx = [best_seed]

        path_geom = _safe_linemerge(
            [cand.geometry.iloc[i] for i in path_seg_idx]
        )

        # ── fine metrics + hard overlap gate ───────────────────────
        metr = {}
        for idx_fake, i in enumerate(path_seg_idx):
            g = path_geom if idx_fake == 0 else None  # evaluate once
            if g is None:
                continue
            m = sim_fn(s_geom, g)
            if mode == "real":
                if not (m["overlap"] >= OVERLAP_MIN_FRAC or
                        m["overlap_km"] * 1_000 >= OVERLAP_MIN_ABS_M):
                    continue
            metr[i] = m  # key doesn't matter, we use the same `g`

        # ── scoring & greedy accumulation ──────────────────────────
        scored = []
        for i, m in metr.items():
            sc = score_fn(m, s_len) if mode == "real" else score_fn(m)
            scored.append((i, sc, m))
        scored.sort(key=lambda t: -t[1])  # best first

        # ── NEW: if everything was filtered-out, skip this DLR line
        if not scored:
            continue

        acc, alloc_km = [], 0.0
        for i, sc, m in scored:
            if alloc_km + m["overlap_km"] > MAX_OVERRUN * s_len/1_000:
                continue
            acc.append((i, sc, m))
            alloc_km += m["overlap_km"]
        if not acc:                              # fallback: top candidate
            acc = [scored[0]]

        best_map[str(s.id)] = str(cand.id.iloc[acc[0][0]])  # ← .iloc !

        for i, sc, m in acc:
            share = (1/len(acc)) if alloc_km == 0 else m["overlap_km"]/alloc_km
            rows.append(dict(
                match_id   = str(s.id),
                dlr_id     = str(s.id),
                network_id  = ";".join(str(cand.id.iloc[i]) for i in path_seg_idx),
                score      = sc,
                v_dlr      = s.v_nom,
                v_net      = cand.v_nom.iloc[i],            # ← .iloc !
                overlap_km = m["overlap_km"],
                frechet_m  = m["frechet"],
                hausdorff_m= m["hausdorff"],
                dir_cos    = m["direction"],
                endpoints_ok = bool(m["endpoints"]),
                network_r  = cand.get("r", np.nan).iloc[i] if "r" in cand else np.nan,
                network_x  = cand.get("x", np.nan).iloc[i] if "x" in cand else np.nan,
                network_b  = cand.get("b", np.nan).iloc[i] if "b" in cand else np.nan,
                allocated_r= s.get("r", np.nan)*share if "r" in s else np.nan,
                allocated_x= s.get("x", np.nan)*share if "x" in s else np.nan,
                allocated_b= s.get("b", np.nan)*share if "b" in s else np.nan,
            ))

    return rows, best_map

# ──────────────────────────────────────────────────────────────
# NEW SECTION – build graph & network criteria
# ⬇ add this just after the “stroke-merging” helpers
# ──────────────────────────────────────────────────────────────
import networkx as nx
from collections import Counter

def _build_graph(lines: gpd.GeoDataFrame) -> nx.Graph:
    """
    Turn every LineString (or MultiLineString) in *lines* into
    an undirected edge between its two *extreme* vertices.

    Nodes carry the attributes:
        x, y   – longitude / latitude  (EPSG:4326)
    Edges keep:
        id     – original line id (string)
        v_nom  – voltage, if present
    """
    if lines.empty:
        return nx.Graph()

    # ensure geographic CRS for node coords
    if lines.crs and not lines.crs.is_geographic:
        tmp = lines.to_crs(4326)
    else:
        tmp = lines

    G = nx.Graph()
    for row in tmp.itertuples():
        geom = row.geometry
        if not geom or geom.is_empty:
            continue
        # extremes: first & last vertex of *merged* geometry
        if geom.geom_type == "MultiLineString":
            geom = linemerge(geom)
        coords = list(geom.coords) if geom.geom_type == "LineString" else []
        if len(coords) < 2:
            continue
        a, b = coords[0], coords[-1]
        # add / update nodes
        for pt in (a, b):
            if pt not in G:
                G.add_node(
                    pt,
                    x=pt[0],
                    y=pt[1],
                )
        # add edge with attributes
        G.add_edge(
            a,
            b,
            id=str(row.id),
            v_nom=float(getattr(row, "v_nom", np.nan)),
        )
    return G


def _compute_graph_metrics(G: nx.Graph) -> pd.DataFrame:
    """
    Return a DataFrame with one row per *edge* containing

        id, degree_u, degree_v, betweenness, clustering_u, clustering_v
    """
    if G.number_of_edges() == 0:
        return pd.DataFrame(
            columns=[
                "id", "degree_u", "degree_v",
                "betweenness", "clustering_u", "clustering_v"
            ]
        )

    # 1. per-node metrics
    degree     = dict(G.degree())
    clustering = nx.clustering(G)
    # 2. link betweenness centrality (normalised – eq. (2))
    bc_raw = nx.edge_betweenness_centrality(G, normalized=True)

    rows = []
    for u, v, data in G.edges(data=True):
        rows.append(
            dict(
                id=data["id"],
                degree_u=degree[u],
                degree_v=degree[v],
                betweenness=bc_raw[(u, v)],
                clustering_u=clustering[u],
                clustering_v=clustering[v],
            )
        )
    return pd.DataFrame(rows)



# ──────────────────────────────────────────────────────────────
#  THE PUBLIC ENTRY POINT
# ──────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────
#  src/matching/hybrid_matching.py
#  public entry-point – COMPLETE function
# ──────────────────────────────────────────────────────────────

import logging, pandas as pd
from typing import Dict, Tuple
import geopandas as gpd

LOGGER = logging.getLogger(__name__)

# (all helper utilities - _to_metric, _merge_strokes, _as_chord, _match …)
# must live in the same module – omitted here for brevity
# ------------------------------------------------------------------------

# ------------------------------------------------------------------
#  PUBLIC ENTRY-POINT
# ------------------------------------------------------------------
def match_lines_real_and_chord(
    dlr : gpd.GeoDataFrame,
    net : gpd.GeoDataFrame,
    *,
    cfg : dict,
) -> Tuple[
        pd.DataFrame,           # df_real   – best matches vs. real geometry
        pd.DataFrame,           # df_chord  – best matches vs. chord geometry
        Dict[str, str],         # best_real  (DLR id → best real-network id)
        Dict[str, str],         # best_chord (DLR id → best chord id)
]:
    """
    Run the matcher twice **independently**

        ① against the *original* network geometry  → df_real
        ② against straight-line “chords” built from stroke-merged network
           lines                                   → df_chord

    No post-merging between the two tables – the caller can combine or
    de-duplicate them later.
    """
    # ── read parameters from *cfg* (fall back to defaults) ─────────────
    buf_deg   = cfg.get("buffer_distance"         , 0.020)   # deg  → corridor
    snap_deg  = cfg.get("snap_distance"           , 0.010)   # deg  → stroke merge
    dir_thr   = cfg.get("direction_threshold"     , 0.45)    # cos θ – coarse filter
    use_end   = cfg.get("use_endpoint_filter"     , True)
    enforce_v = cfg.get("enforce_voltage_matching", True)

    buf_m  = buf_deg  * 111_000                    # rough deg→m for Europe
    snap_m = snap_deg * 111_000

    # ── transform to metric CRS once ──────────────────────────────────
    dlr_m = _to_metric(dlr)
    net_m = _to_metric(net)      # untouched geometry for the “real” run

    # ── stroke-merge a *copy* (only needed to generate chords) ────────
    net_strokes = _merge_strokes(net_m, snap_m, min_cos=dir_thr)

    chords          = net_strokes.copy()
    chords.geometry = chords.geometry.apply(_as_chord)
    chords["id"]    = "chord_" + chords["id"].astype(str)

    chords[["id", "src_ids"]].to_csv("output/chord_lookup.csv", index=False)

    # ── ensure numeric voltage everywhere ─────────────────────────────
    for g in (dlr_m, net_m, chords):
        g["v_nom"] = g["v_nom"].fillna(0).astype(float)

        # ------------------------------------------------------------------
        # 0️⃣  same-voltage endpoint walker (deterministic)
        # ------------------------------------------------------------------
        # In your main function:
        LOGGER.info("Running endpoint-BFS matcher with improved snapping...")
        G_endpts = _build_endpoint_graph(net_m)  # Build improved graph

        LOGGER.info(f"Graph built with {G_endpts.number_of_nodes()} nodes and {G_endpts.number_of_edges()} edges")

        # Check graph connectivity
        connected_components = list(nx.connected_components(G_endpts))
        LOGGER.info(f"Graph has {len(connected_components)} connected components")
        LOGGER.info(f"Largest component has {max(len(c) for c in connected_components)} nodes")

        # Optional: visualize the graph for debugging
        if hasattr(nx, 'draw_networkx'):  # Only if matplotlib is available
            try:
                import matplotlib.pyplot as plt
                plt.figure(figsize=(15, 12))
                pos = {node: (node[0], node[1]) for node in G_endpts.nodes()}
                nx.draw_networkx(G_endpts, pos, node_size=20, with_labels=False)
                plt.savefig("output/network_graph.png", dpi=300)
                plt.close()
                LOGGER.info("Graph visualization saved to output/network_graph.png")
            except Exception as e:
                LOGGER.warning(f"Could not visualize graph: {e}")

        success_count = 0
        total_count = len(dlr_m)
        bfs_rows, bfs_best = [], {}

        try:
            for row in dlr_m.itertuples():
                # Use an appropriate snap tolerance in meters (not degrees)
                try:
                    segs = _walk_same_voltage(row, net_m, G_endpts, snap_tolerance=1000.0)
                    if not segs:
                        continue

                    # Filter out virtual edges (added to connect components)
                    segs = [idx for idx in segs if idx >= 0]

                    # Only continue if we have real segments after filtering
                    if not segs:
                        continue

                    success_count += 1
                    bfs_best[str(row.id)] = ";".join(str(net_m.id.iloc[i]) for i in segs)

                    bfs_rows.append(dict(
                        match_id=row.id,
                        dlr_id=row.id,
                        network_id=bfs_best[str(row.id)],
                        network_path=";".join(str(net_m.id.iloc[i]) for i in segs),  # NEW
                        score=1.0,  # constant -- purely deterministic match
                        v_dlr=row.v_nom,
                        v_net=net_m.v_nom.iloc[segs[0]],
                        overlap_km=np.nan,
                        frechet_m=np.nan,
                        hausdorff_m=np.nan,
                        dir_cos=np.nan,
                        endpoints_ok=True,
                        network_r=np.nan,
                        network_x=np.nan,
                        network_b=np.nan,
                        allocated_r=np.nan,
                        allocated_x=np.nan,
                        allocated_b=np.nan,
                    ))

                except Exception as e:
                    LOGGER.error(f"Error processing DLR line {row.id}: {e}")
                    import traceback
                    LOGGER.error(traceback.format_exc())
                    continue

        except Exception as e:
            LOGGER.error(f"Error in endpoint-BFS matcher: {e}")
            import traceback
            LOGGER.error(traceback.format_exc())

        # Turn rows into DataFrame
        COLS = [
            "match_id", "dlr_id", "network_id", "score",
            "v_dlr", "v_net", "network_path",
            "overlap_km", "frechet_m", "hausdorff_m",
            "dir_cos", "endpoints_ok",
            "network_r", "network_x", "network_b",
            "allocated_r", "allocated_x", "allocated_b",
        ]
        df_bfs = pd.DataFrame(bfs_rows, columns=COLS)

        # Log coverage
        cov = 0.0 if df_bfs.empty else 100 * df_bfs.dlr_id.nunique() / len(dlr_m)
        LOGGER.info("Endpoint-BFS matcher: %d rows  (coverage %.1f %%)",
                    len(df_bfs), cov)

        # optional: write to disk
        df_bfs.to_csv("output/matches_endpoint_bfs.csv", index=False)

        # --------------------------------------------------------------
        #   RETURN – all other matchers are ignored for now
        # --------------------------------------------------------------

        if len(dlr_m) > 0:
            sample_dlr = dlr_m.iloc[0]
            LOGGER.info(f"Sample DLR line: id={sample_dlr.id}, v_nom={sample_dlr.v_nom}")
            LOGGER.info(f"Geometry type: {sample_dlr.geometry.geom_type}")
            LOGGER.info(f"Endpoints: {_first_last_keys(sample_dlr.geometry)}")

        if len(net_m) > 0:
            sample_net = net_m.iloc[0]
            LOGGER.info(f"Sample Network line: id={sample_net.id}, v_nom={sample_net.v_nom}")
            LOGGER.info(f"Geometry type: {sample_net.geometry.geom_type}")
            LOGGER.info(f"Endpoints: {_first_last_keys(sample_net.geometry)}")


        return (
            df_bfs,  # put same df in both slots for legacy callers
            df_bfs,
            bfs_best,  # best_real   (legacy)
            bfs_best,  # best_chord  (legacy)
            pd.DataFrame()  # empty metrics – not used
        )



