# ------------------------- build_pypsa_chains --------------------------
def build_pypsa_chains(pypsa_gdf, endpoint_grid_m=50.0, id_col="id", geom_col="geometry"):
    """
    Build linear 'chains' of connected PyPSA line segments.

    Parameters
    ----------
    pypsa_gdf : GeoDataFrame
        PyPSA lines with geometries
    endpoint_grid_m : float
        Grid size (meters) for snapping endpoints
    id_col : str
        Column name containing PyPSA line ID
    geom_col : str
        Column name containing geometry

    Returns
    -------
    chains : dict
        Dictionary mapping chain_id to dict with:
        - segment_ids: list of PyPSA IDs
        - geometry_m: merged geometry
        - length_km: total length in km
        - circuits: number of circuits (min of all segments)

    segment_to_chain : dict
        Maps PyPSA segment ID to its chain ID
    """
    import networkx as nx
    from shapely.geometry import LineString, Point, MultiLineString
    from shapely.ops import linemerge, unary_union

    # Helper to get segment endpoints
    def get_endpoints(geom):
        if geom is None or geom.is_empty:
            return None, None

        if isinstance(geom, LineString):
            coords = list(geom.coords)
            if len(coords) < 2:
                return None, None
            return Point(coords[0]), Point(coords[-1])

        # Handle MultiLineString
        try:
            if hasattr(geom, 'geoms'):
                parts = list(geom.geoms)
                if not parts:
                    return None, None
                first_coords = list(parts[0].coords)
                last_coords = list(parts[-1].coords)
                return Point(first_coords[0]), Point(last_coords[-1])
        except:
            pass

        return None, None

    # Snap point to grid
    def snap_to_grid(point, grid_m):
        if point is None:
            return None
        x = round(point.x / grid_m) * grid_m
        y = round(point.y / grid_m) * grid_m
        return (x, y)

    # Build graph with endpoints
    G = nx.Graph()

    # Add nodes and edges for each segment
    for _, row in pypsa_gdf.iterrows():
        # Use provided column names instead of hardcoded ones
        pypsa_id = str(row.get(id_col, row.get('line_id', '')))
        geom = row.get(geom_col)

        if geom is None or geom.is_empty:
            continue

        # Get endpoints
        start, end = get_endpoints(geom)
        if start is None or end is None:
            continue

        # Snap to grid
        start_key = snap_to_grid(start, endpoint_grid_m)
        end_key = snap_to_grid(end, endpoint_grid_m)

        if start_key is None or end_key is None:
            continue

        # Add to graph
        G.add_node(start_key, is_endpoint=True)
        G.add_node(end_key, is_endpoint=True)

        # Create edge with segment info
        if G.has_edge(start_key, end_key):
            G[start_key][end_key]['segments'].append(pypsa_id)
        else:
            G.add_edge(start_key, end_key, segments=[pypsa_id])

    # Now traverse the graph to build chains
    chains = {}
    segment_to_chain = {}
    visited_segments = set()
    chain_id = 1

    def walk_chain(start_node):
        """Walk graph from start node until hitting a junction or dead end"""
        path = []
        segments = []
        current = start_node
        prev = None

        while True:
            neighbors = list(G.neighbors(current))

            # Remove the previous node from neighbors if we're not at start
            if prev is not None and prev in neighbors:
                neighbors.remove(prev)

            # If dead end or junction, stop
            if len(neighbors) != 1:
                break

            # Get next node and segments on this edge
            next_node = neighbors[0]
            edge_segments = G[current][next_node]['segments']

            # Add unvisited segments
            for s in edge_segments:
                if s not in visited_segments:
                    segments.append(s)
                    visited_segments.add(s)

            # Move to next node
            path.append(next_node)
            prev = current
            current = next_node

        return path, segments

    # Find all endpoints or junctions as starting points
    start_points = [node for node, degree in G.degree() if degree != 2]

    # Also find any unvisited cycles (all nodes degree 2)
    for node in G.nodes():
        if node not in start_points:
            neighbors = list(G.neighbors(node))
            if len(neighbors) == 2:
                edge1_segments = G[node][neighbors[0]]['segments']
                edge2_segments = G[node][neighbors[1]]['segments']

                # Check if any segments are unvisited
                has_unvisited = False
                for s in edge1_segments + edge2_segments:
                    if s not in visited_segments:
                        has_unvisited = True
                        break

                if has_unvisited:
                    start_points.append(node)

    # Walk chains from all start points
    for start in start_points:
        # Skip if no unvisited segments connected to this node
        has_unvisited = False
        for neighbor in G.neighbors(start):
            for s in G[start][neighbor]['segments']:
                if s not in visited_segments:
                    has_unvisited = True
                    break
            if has_unvisited:
                break

        if not has_unvisited:
            continue

        # Walk chains in both directions from this node
        forward_path, forward_segments = walk_chain(start)

        # If we found segments, create a chain
        if forward_segments:
            chain_segments = forward_segments

            # Get chain properties
            pypsa_rows = [row for _, row in pypsa_gdf.iterrows()
                          if str(row.get(id_col, row.get('line_id', ''))) in chain_segments]

            # Merge geometries
            geoms = [row.get(geom_col) for row in pypsa_rows if row.get(geom_col) is not None]
            if geoms:
                try:
                    merged = linemerge(unary_union(geoms))
                except:
                    # Fallback to simple union if merge fails
                    merged = unary_union(geoms)
            else:
                merged = None

            # Calculate total length
            total_length_km = 0
            for row in pypsa_rows:
                length = row.get('length', 0)
                # Convert m to km if needed
                length_km = length / 1000.0 if length > 1000 else length
                total_length_km += length_km

            # Get minimum circuit count
            circuits = min([row.get('circuits', 1) for row in pypsa_rows]) if pypsa_rows else 1

            # Create chain
            chain_name = f"chain_{chain_id:04d}"
            chains[chain_name] = {
                'segment_ids': chain_segments,
                'geometry_m': merged,
                'length_km': total_length_km,
                'circuits': circuits
            }

            # Map segments to chain
            for seg_id in chain_segments:
                segment_to_chain[seg_id] = chain_name

            chain_id += 1

    print(f"Built {len(chains)} PyPSA chains from {len(segment_to_chain)} segments")
    return chains, segment_to_chain


# ----------------------- chain_corridor_rematch ------------------------
def chain_corridor_rematch(
    pypsa_gdf,
    jao_matches,
    endpoint_grid_m: float = 200.0,
    match_buffer_m: float = 400.0,
    min_coverage: float = 0.35,
    id_col: str = "id",
    geom_col: str = "geometry",
    verbose: bool = False,
    existing_results=None,
    **_ignore,
):
    """
    Re-match JAO corridors to *paths of chains* (one or more adjacent chains).

    Steps
    -----
    1) Build chains from PyPSA lines (already robust in build_pypsa_chains).
    2) For each JAO row, pick the best seed chain by overlap.
    3) Greedily extend the seed by adding adjacent chains that improve overlap
       with the JAO line, until no beneficial neighbor remains.
    4) Skip JAO rows locked_by_corridor and never add PyPSA chains that are
       entirely locked.
    """

    import pandas as pd
    from math import cos, radians
    from shapely.geometry import LineString, MultiLineString, Point
    from shapely.ops import unary_union

    # ---------- helpers ----------
    def _as_rows(x):
        if x is None:
            return []
        if isinstance(x, pd.DataFrame):
            return [dict(r) for _, r in x.iterrows()]
        if isinstance(x, (list, tuple)):
            return [dict(r) for r in x]
        if isinstance(x, pd.Series):
            return [dict(x)]
        return [dict(x)]

    def _jao_id(d):
        return str(d.get("jao_id", d.get("id", "")))

    def _is_line(g):
        return isinstance(g, (LineString, MultiLineString))

    def _geom_len_m(geom):
        return float(getattr(geom, "length", 0.0) or 0.0)

    # meters→degrees conversion (only used if CRS is geographic)
    def _meters_to_degrees(m, lat_deg=50.0):
        # ~111.32 km per degree lon at equator; scale lon by cos(lat)
        if m is None:
            return 0.0
        lon_deg = m / (111320.0 * max(0.15, cos(radians(lat_deg))))
        lat_deg_per_m = 1.0 / 110574.0
        lat_deg_val = m * lat_deg_per_m
        # we use the *max* to avoid creating skinny buffers; OK for snapping/buffering
        return max(lon_deg, lat_deg_val)

    # snap key that works in degrees or meters
    def _snap_key(p: Point, tol_m: float, mean_lat: float, is_geographic: bool):
        if is_geographic:
            # anisotropic quantization by converting meters to degrees separately
            dx = max(1e-10, tol_m / (111320.0 * max(0.15, cos(radians(mean_lat)))))
            dy = max(1e-10, tol_m / 110574.0)
            return (round(p.x / dx) * dx, round(p.y / dy) * dy)
        else:
            s = tol_m
            return (round(p.x / s) * s, round(p.y / s) * s)

    # ---------- inputs & precompute ----------
    chains, _seg_to_chain = build_pypsa_chains(
        pypsa_gdf, endpoint_grid_m=endpoint_grid_m, id_col=id_col, geom_col=geom_col
    )

    # detect CRS to convert meters → degrees where needed
    crs = getattr(pypsa_gdf, "crs", None)
    is_geographic = False
    if crs is not None:
        try:
            is_geographic = bool(getattr(crs, "is_geographic", False))
        except Exception:
            is_geographic = False

    # Mean latitude for deg↔m conversions
    sample_lat = None
    try:
        # take first geometry we find
        for _, r in pypsa_gdf.iterrows():
            g = r.get(geom_col)
            if _is_line(g):
                from grid_matcher.geo.geometry import _iter_line_parts
                coords = list(next(iter(_iter_line_parts(g))).coords)
                if coords:
                    sample_lat = coords[0][1]
                    break
    except Exception:
        pass
    if sample_lat is None:
        sample_lat = 50.0

    # Build endpoint index over CHAINS (to allow chain-to-chain joins)
    endpoint_index = {}  # snapped_key -> set(chain_ids)
    chain_endkeys = {}   # chain_id -> (keyA, keyB)
    for cid, cinfo in chains.items():
        g = cinfo.get("geometry_m")
        if not _is_line(g):
            continue
        try:
            first_ls = next(_iter_line_parts(g))
            coords = list(first_ls.coords)
            # try last part's end as well
            last_ls = list(_iter_line_parts(g))[-1]
            coords_last = list(last_ls.coords)
            pA = Point(coords[0])
            pB = Point(coords_last[-1])
        except Exception:
            continue
        kA = _snap_key(pA, endpoint_grid_m, sample_lat, is_geographic)
        kB = _snap_key(pB, endpoint_grid_m, sample_lat, is_geographic)
        chain_endkeys[cid] = (kA, kB)
        endpoint_index.setdefault(kA, set()).add(cid)
        endpoint_index.setdefault(kB, set()).add(cid)

    # Locked PyPSA ids
    locked_pypsa = set(
        str(r[id_col])
        for _, r in pypsa_gdf.iterrows()
        if bool(r.get("locked_by_corridor", False))
    )

    # input rows
    jao_rows = _as_rows(jao_matches)
    baseline_rows = _as_rows(existing_results)
    baseline_by_id = {_jao_id(r): r for r in baseline_rows}

    # ---------- overlap helpers ----------
    def _buffer_for(g_line):
        if not is_geographic:
            return g_line.buffer(match_buffer_m)
        # geographic: convert meters→degrees using local latitude
        try:
            # get a representative latitude from the geometry itself
            coords = list(next(_iter_line_parts(g_line)).coords)
            lat_here = coords[0][1] if coords else sample_lat
        except Exception:
            lat_here = sample_lat
        deg = _meters_to_degrees(match_buffer_m, lat_here)
        return g_line.buffer(deg)

    def _coverage_on_chain(jbuf, cgeom):
        L = _geom_len_m(cgeom)
        if L <= 0:
            return 0.0, 0.0
        try:
            inter = cgeom.intersection(jbuf).length
        except Exception:
            inter = 0.0
        cov = inter / L
        return cov, inter

    # ---------- greedy extension ----------
    def _extend_best(jgeom, seed_id):
        """
        Starting from seed chain, keep appending adjacent chains
        that have the highest overlap with jbuf, until no gain.
        """
        jbuf = _buffer_for(jgeom)
        chosen = [seed_id]
        chosen_set = {seed_id}

        # frontier = both endpoints of currently chosen chains
        frontier_keys = set(chain_endkeys.get(seed_id, ()))
        current_geom = chains[seed_id]["geometry_m"]
        try:
            current_inter_len = current_geom.intersection(jbuf).length
        except Exception:
            current_inter_len = 0.0

        # hard cap to prevent runaway on dense hubs
        MAX_STEPS = 20
        steps = 0

        while steps < MAX_STEPS:
            steps += 1
            # gather neighbors touching the frontier
            neighbor_ids = set()
            for k in list(frontier_keys):
                neighbor_ids |= endpoint_index.get(k, set())
            neighbor_ids -= chosen_set
            if not neighbor_ids:
                break

            # evaluate each neighbor by overlap with JAO buffer
            best_cand = None
            best_gain = 0.0
            for nid in neighbor_ids:
                cgeom = chains[nid]["geometry_m"]
                cov, inter_len = _coverage_on_chain(jbuf, cgeom)
                if cov < min_coverage * 0.6:  # relaxed threshold for incremental pieces
                    continue

                # skip if fully locked
                nids = chains[nid].get("segment_ids", [])
                if nids and all(str(pid) in locked_pypsa for pid in nids):
                    continue

                gain = inter_len  # absolute overlap with buffer; simple and effective
                if gain > best_gain:
                    best_gain = gain
                    best_cand = nid

            if best_cand is None:
                break

            # accept the best neighbor and update frontier + geometry
            chosen.append(best_cand)
            chosen_set.add(best_cand)

            kA, kB = chain_endkeys.get(best_cand, (None, None))
            # frontier update: remove the shared node, keep the new free end
            for k in (kA, kB):
                if k in frontier_keys:
                    frontier_keys.remove(k)
                else:
                    frontier_keys.add(k)

            # merge geometries incrementally (union is fine for length/intersection)
            try:
                current_geom = unary_union([current_geom, chains[best_cand]["geometry_m"]])
                current_inter_len = current_geom.intersection(jbuf).length
            except Exception:
                pass

        return chosen

    # ---------- main loop per JAO ----------
    out = []
    for row in jao_rows:
        jid = _jao_id(row)
        base = baseline_by_id.get(jid, {})
        jgeom = row.get("geometry", base.get("geometry"))
        locked_jao = bool(row.get("locked_by_corridor", base.get("locked_by_corridor", False)))

        res = dict(base)
        res.update(row)
        res.setdefault("jao_id", jid)

        if locked_jao:
            res["note"] = "locked: left existing pypsa_ids unchanged"
            out.append(res)
            continue

        if not _is_line(jgeom):
            res["note"] = "skipped: JAO geometry is not a line"
            out.append(res)
            continue

        jao_len_km = _geom_len_m(jgeom) / 1000.0

        # pick best seed chain
        jbuf = _buffer_for(jgeom)
        best = None  # (score, coverage, inter_len, cid)
        for cid, cinfo in chains.items():
            cgeom = cinfo.get("geometry_m")
            L = _geom_len_m(cgeom)
            if L <= 0:
                continue
            try:
                inter_len = cgeom.intersection(jbuf).length
            except Exception:
                inter_len = 0.0
            cov = inter_len / L if L > 0 else 0.0
            if cov < min_coverage:
                continue
            # prefer high coverage and long absolute overlap
            score = (cov * 100.0) + (inter_len / 1000.0)
            if (best is None) or (score > best[0]):
                best = (score, cov, inter_len, cid)

        if best is None:
            res.update({
                "note": f"no chain >= {int(min_coverage*100)}% coverage",
                "coverage": 0.0,
                "chain_id": None,
                "chain_length_km": 0.0,
                "jao_length_km": jao_len_km,
            })
            out.append(res)
            continue

        _, seed_cov, _, seed_cid = best

        # extend across adjacent chains that also follow the corridor
        chosen_chain_ids = _extend_best(jgeom, seed_cid)

        # collect PyPSA ids from all chosen chains (respect locks)
        all_pids = []
        for cid in chosen_chain_ids:
            all_pids.extend(map(str, chains[cid].get("segment_ids", [])))

        unlocked_pids = [pid for pid in all_pids if pid not in locked_pypsa]
        if not unlocked_pids:
            res.update({
                "note": "selected path consists entirely of locked PyPSA rows",
                "coverage": float(seed_cov),
                "chain_id": ";".join(chosen_chain_ids),
                "chain_length_km": sum(float(chains[c].get("length_km", 0.0) or 0.0) for c in chosen_chain_ids),
                "jao_length_km": jao_len_km,
            })
            out.append(res)
            continue

        # summarize combined geometry to compute useful coverage metric
        try:
            combined_geom = unary_union([chains[c]["geometry_m"] for c in chosen_chain_ids])
            inter_len = combined_geom.intersection(jbuf).length
            comb_len_km = float(getattr(combined_geom, "length", 0.0) or 0.0) / 1000.0
            coverage = inter_len / (comb_len_km * 1000.0) if comb_len_km > 0 else 0.0
        except Exception:
            comb_len_km = sum(float(chains[c].get("length_km", 0.0) or 0.0) for c in chosen_chain_ids)
            coverage = seed_cov

        res.update({
            "pypsa_ids": unlocked_pids,
            "coverage": float(coverage),
            "chain_id": ";".join(chosen_chain_ids),
            "chain_length_km": comb_len_km,
            "jao_length_km": jao_len_km,
        })

        if verbose:
            print(f"[chain-rematch] JAO {jid} -> {res['chain_id']} | "
                  f"chains={len(chosen_chain_ids)} | pypsa_ids={len(unlocked_pids)}")

        out.append(res)

    return out


def auto_extend_short_matches(
    matching_results,
    jao_gdf,
    pypsa_gdf,
    target_ratio=0.80,
    buffer_m=120.0,
    endpoint_grid_m=50.0,
    max_additions_per_line=40,
):
    """
    Extend 'short' JAO→PyPSA matches by traversing adjacent PyPSA segments to
    build the full corridor chain.

    Strategy
    --------
    1) Precompute an adjacency graph for PyPSA lines:
       - Two PyPSA lines are adjacent if they share an endpoint (rounded to an
         endpoint grid in meters).
    2) For each matched JAO record that is NOT locked-by-corridor and whose
       covered PyPSA length << JAO length, we BFS from the current PyPSA seed(s)
       through adjacent lines. We only accept neighbors that stay near the JAO
       geometry (hausdorff <= ~buffer_m or buffered IoU ≥ 0.2).
    3) Stop when coverage ≥ target_ratio or no better neighbors exist.

    Notes
    -----
    - Works with shapely Points or (x, y) tuples safely.
    - Uses EPSG:3857 projection internally for metric distances.
    - Never edits records marked with `locked_by_corridor=True`.

    Parameters
    ----------
    matching_results : list[dict]
    jao_gdf, pypsa_gdf : GeoDataFrames
    target_ratio : float
        Minimum (sum(PyPSA km)/JAO km) coverage to achieve by extension.
    buffer_m : float
        Corridor buffer (meters) used to decide if a PyPSA segment hugs the JAO line.
    endpoint_grid_m : float
        Grid size (meters) to cluster endpoints and build adjacency.
    max_additions_per_line : int
        Safety cap on how many extra PyPSA segments we can add per JAO record.

    Returns
    -------
    list[dict] (the same list, modified in place and also returned)
    """
    from collections import defaultdict, deque
    import math
    import numpy as np
    from shapely.geometry import Point, LineString
    from shapely.ops import unary_union

    # ----------------- helpers -----------------
    def _ensure_crs(gdf, epsg=4326):
        try:
            if gdf.crs is None:
                gdf = gdf.set_crs(epsg, allow_override=True)
            return gdf
        except Exception:
            return gdf

    def _to_meters(gdf):
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

    def _endpoint_key(pt, grid=endpoint_grid_m):
        """Round a *projected* point to a grid key."""
        p = _as_point(pt)
        if p is None:
            return None
        x = round(p.x / grid) * grid
        y = round(p.y / grid) * grid
        return (float(x), float(y))

    def _endpoints(ls):
        """Return start & end shapely Points from a LineString (projected)."""
        try:
            coords = list(ls.coords)
            if len(coords) < 2:
                return None, None
            return Point(coords[0]), Point(coords[-1])
        except Exception:
            return None, None

    def _parse_ids(v):
        """pypsa_ids can be list or str with ',' or ';' separators."""
        if not v:
            return []
        if isinstance(v, (list, tuple, set)):
            out = [str(x).strip() for x in v if str(x).strip()]
        else:
            s = str(v)
            out = []
            for sep in (";", ","):
                if sep in s:
                    out = [t.strip() for t in s.split(sep)]
                    break
            if not out:
                out = [s.strip()]
        return [x for x in out if x]

    def _length_km_from_row(row):
        try:
            lm = float(row.get("length", 0.0) or 0.0)
            return lm / 1000.0
        except Exception:
            return 0.0

    def _hausdorff_m(a, b):
        try:
            return float(a.hausdorff_distance(b))
        except Exception:
            return float("inf")

    def _buffered_iou(a, b, w=buffer_m):
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

    def _segment_fits_jao(seg, jao_line):
        """Decide if a PyPSA seg lies along the JAO corridor."""
        try:
            dH = _hausdorff_m(seg, jao_line)  # in meters (same CRS)
            if dH <= buffer_m * 1.25:
                return True
            iou = _buffered_iou(seg, jao_line, w=buffer_m)
            return iou >= 0.20
        except Exception:
            return False

    # ------------- prepare projected copies -------------
    jao_gdf = _ensure_crs(jao_gdf)
    pypsa_gdf = _ensure_crs(pypsa_gdf)
    try:
        jao_m = _to_meters(jao_gdf)
        py_m = _to_meters(pypsa_gdf)
    except Exception:
        jao_m = jao_gdf
        py_m = pypsa_gdf

    # quick lookups
    jao_by_id = {str(r.get("id")): r for _, r in jao_gdf.iterrows()}
    jao_m_by_id = {str(r.get("id")): r for _, r in jao_m.iterrows()}
    py_by_id = {str(r.get("line_id", r.get("id", ""))): r for _, r in pypsa_gdf.iterrows()}
    py_m_by_id = {str(r.get("line_id", r.get("id", ""))): r for _, r in py_m.iterrows()}

    # ------------- build endpoint adjacency for PyPSA -------------
    # Map endpoint key -> set of pypsa ids
    endpoint_index = defaultdict(set)
    # For faster neighbor membership
    id_to_endkeys = {}

    for pid, rowm in py_m_by_id.items():
        geom = rowm.geometry
        if geom is None or geom.is_empty:
            continue
        ls = geom
        if not isinstance(ls, LineString) and hasattr(ls, "geoms"):
            # MultiLineString: take the longest
            try:
                ls = max(list(ls.geoms), key=lambda g: g.length)
            except Exception:
                continue
        p0, p1 = _endpoints(ls)
        if p0 is None or p1 is None:
            continue
        k0 = _endpoint_key(p0)
        k1 = _endpoint_key(p1)
        id_to_endkeys[pid] = (k0, k1)
        if k0:
            endpoint_index[k0].add(pid)
        if k1:
            endpoint_index[k1].add(pid)

    # Neighbor map: share an endpoint key
    neighbors = defaultdict(set)
    for key, ids in endpoint_index.items():
        ids = list(ids)
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = ids[i], ids[j]
                neighbors[a].add(b)
                neighbors[b].add(a)

    # ------------- main loop over results -------------
    changed = 0
    for res in (matching_results or []):
        # keep locked-by-corridor intact
        if res.get("locked_by_corridor", False):
            continue
        if not res.get("matched", False):
            continue

        jao_id = str(res.get("jao_id", ""))
        if not jao_id or jao_id not in jao_by_id:
            continue

        # source-of-truth JAO length in km
        try:
            jlen_km = float(jao_by_id[jao_id].get("length", 0.0) or 0.0)
        except Exception:
            jlen_km = 0.0
        if jlen_km <= 0:
            continue

        # current pypsa ids
        curr_ids = set(_parse_ids(res.get("pypsa_ids", [])))
        if not curr_ids:
            continue

        # current coverage
        curr_km = sum(_length_km_from_row(py_by_id.get(pid, {})) for pid in curr_ids)
        coverage = (curr_km / jlen_km) if jlen_km > 0 else 0.0
        if coverage >= target_ratio:
            # already good enough
            continue

        # get JAO geometry (projected)
        jrow_m = jao_m_by_id.get(jao_id)
        if jrow_m is None or jrow_m.geometry is None or jrow_m.geometry.is_empty:
            continue
        jao_line = jrow_m.geometry
        if not isinstance(jao_line, LineString) and hasattr(jao_line, "geoms"):
            try:
                jao_line = max(list(jao_line.geoms), key=lambda g: g.length)
            except Exception:
                continue

        # BFS expansion from seeds
        selected = set(curr_ids)
        frontier = deque(curr_ids)
        additions = 0

        while frontier and coverage < target_ratio and additions < max_additions_per_line:
            seed = frontier.popleft()
            for nb in neighbors.get(seed, []):
                if nb in selected:
                    continue

                nb_row_m = py_m_by_id.get(nb)
                if nb_row_m is None or nb_row_m.geometry is None or nb_row_m.geometry.is_empty:
                    continue

                seg = nb_row_m.geometry
                if not isinstance(seg, LineString) and hasattr(seg, "geoms"):
                    try:
                        seg = max(list(seg.geoms), key=lambda g: g.length)
                    except Exception:
                        continue

                # Accept neighbor only if it stays close to the JAO corridor
                if not _segment_fits_jao(seg, jao_line):
                    continue

                # Add it
                selected.add(nb)
                frontier.append(nb)
                additions += 1

                # update coverage
                curr_km += _length_km_from_row(py_by_id.get(nb, {}))
                coverage = (curr_km / jlen_km) if jlen_km > 0 else 0.0

                if coverage >= target_ratio or additions >= max_additions_per_line:
                    break

        if selected != set(_parse_ids(res.get("pypsa_ids", []))):
            # Update result
            res["pypsa_ids"] = sorted(selected)
            # Keep a length indicator that downstream may use
            res["coverage_ratio"] = coverage
            res["matched_km"] = curr_km
            mq = res.get("match_quality", "")
            tag = "Auto-extended"
            if tag not in mq:
                res["match_quality"] = (mq + " | " + tag).strip(" |")
            changed += 1

    if changed:
        print(f"auto_extend_short_matches: extended {changed} matches toward target coverage {target_ratio:.0%}")
    else:
        print("auto_extend_short_matches: no changes needed (all under/locked or already sufficient coverage).")

    return matching_results