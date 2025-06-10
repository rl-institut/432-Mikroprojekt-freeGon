import logging
import itertools
import networkx as nx
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import (
    LineString, MultiLineString, LinearRing,
    GeometryCollection, Point
)

# -----------------------------------------------------------------------------
logger = logging.getLogger(__name__)
# -----------------------------------------------------------------------------

def preprocess_network_segments(
    network_lines: gpd.GeoDataFrame,
    *,
    max_gap: float = 1000.0,
    min_dir_similarity: float = 0.8,
) -> tuple[gpd.GeoDataFrame, dict[str, str]]:
    """Return *(merged_gdf, original→merged-id map)*.

    * `max_gap`           – max endpoint distance in **metres** to consider two
      segments connectable.
    * `min_dir_similarity`– cosine similarity threshold between direction
      vectors (1.0 = perfectly aligned).

    A very lightweight algorithm: build an undirected graph whose nodes are
    segment indices and connect two nodes if **any** endpoints lie within
    *max_gap* and the direction similarity is above the threshold.  Every
    connected component with >1 node is merged into a MultiLineString.
    """

    if network_lines.empty:
        return network_lines, {}

    # ensure metric CRS ---------------------------------------------------
    if network_lines.crs is None:
        network_lines = network_lines.set_crs(4326)
    if network_lines.crs.is_geographic:
        network_lines = network_lines.to_crs(3857)

    # spatial index for endpoint search ----------------------------------
    sindex = network_lines.sindex

    def _endpoints(geom):
        pts = []
        if geom.geom_type == "LineString":
            c = list(geom.coords)
            pts.extend([Point(c[0]), Point(c[-1])])
        elif geom.geom_type == "MultiLineString":
            for part in geom.geoms:
                c = list(part.coords)
                pts.extend([Point(c[0]), Point(c[-1])])
        return pts

    # direction similarity -----------------------------------------------
    def _dir_sim(g1, g2):
        def _vec(g):
            if g.geom_type == "LineString":
                c0, c1 = Point(g.coords[0]), Point(g.coords[-1])
            else:
                # take first sub‑line's ends
                gg = list(g.geoms)[0]
                c0, c1 = Point(gg.coords[0]), Point(gg.coords[-1])
            v = np.array([c1.x - c0.x, c1.y - c0.y])
            n = np.linalg.norm(v)
            return v / n if n else v
        v1, v2 = _vec(g1), _vec(g2)
        return abs(np.clip(np.dot(v1, v2), -1, 1))  # absolute cos θ

    # ── build graph ------------------------------------------------------
    G = nx.Graph()
    for pos, geom in enumerate(network_lines.geometry):  # ← positional
        G.add_node(pos)
        for p in _endpoints(geom):
            for other_pos in sindex.intersection(p.buffer(max_gap).bounds):
                if other_pos == pos or G.has_edge(pos, other_pos):
                    continue
                other_geom = network_lines.geometry.iloc[int(other_pos)]
                if any(p.distance(q) <= max_gap for q in _endpoints(other_geom)):
                    if _dir_sim(geom, other_geom) >= min_dir_similarity:
                        G.add_edge(pos, other_pos)

    id_map: dict[str, str] = {}
    merged_rows = []
    for comp_id, component in enumerate(nx.connected_components(G), 1):
        if len(component) == 1:
            continue  # singletons kept as-is later
        geoms = []
        length = 0.0
        for pos in component:
            row = network_lines.iloc[int(pos)]
            g = row.geometry
            # ---- flatten any MultiLineString -------------------
            if g.geom_type == "MultiLineString":
                geoms.extend(list(g.geoms))
            else:
                geoms.append(g)
            # ----------------------------------------------------
            length += row.get("length", g.length)
            id_map[str(row.id)] = f"merged_{comp_id}"

        merged_rows.append({
            "id": f"merged_{comp_id}",
            "geometry": MultiLineString(geoms) if len(geoms) > 1 else geoms[0],
            "length": length,
            "is_merged": True,
        })

    merged_gdf = gpd.GeoDataFrame(merged_rows, crs=network_lines.crs)
    singletons = network_lines[~network_lines.index.isin(id_map.keys())].copy()
    singletons["is_merged"] = False

    combined = pd.concat([merged_gdf, singletons], ignore_index=True)
    return combined, id_map

def _segment_graph(gdf: gpd.GeoDataFrame, snap_m: float) -> nx.Graph:
    """Return an undirected graph in which every **row** of *gdf* is a node and
    an edge exists when two segments have endpoints closer than *snap_m* metres."""
    g = nx.Graph()
    coords = []  # (index, [Point, Point, ...])

    for idx, geom in zip(gdf.index, gdf.geometry):
        if geom is None or geom.is_empty:
            continue
        ends: list[Point] = []
        if geom.geom_type == "LineString":
            c = list(geom.coords)
            ends.extend([Point(c[0]), Point(c[-1])])
        elif geom.geom_type == "MultiLineString":
            for part in geom.geoms:
                c = list(part.coords)
                ends.extend([Point(c[0]), Point(c[-1])])
        coords.append((idx, ends))
        g.add_node(idx)

    # naive O(n²) – acceptable for a few thousand candidates
    for (i, ends_i), (j, ends_j) in itertools.combinations(coords, 2):
        if any(e1.distance(e2) <= snap_m for e1 in ends_i for e2 in ends_j):
            g.add_edge(i, j)

    return g

# -----------------------------------------------------------------------------

def _length_km(geom) -> float:
    """Return length (km) of *geom* (expects CRS in **metres**)."""
    if geom is None or geom.is_empty:
        return 0.0
    if geom.geom_type in {"LineString", "LinearRing"}:
        return geom.length / 1000.0
    if geom.geom_type == "MultiLineString":
        return sum(part.length for part in geom.geoms) / 1000.0
    if geom.geom_type == "GeometryCollection":
        return sum(_length_km(g) for g in geom.geoms)
    return 0.0

# -----------------------------------------------------------------------------

TOL = 3000.0      # metres  (tune later)

def _overlap_km(src_geom, net_geom) -> float:
    """
    Length (km) of the part of *net_geom* that lies within ±TOL of *src_geom*.
    CRS must be metres (EPSG:3857).
    """
    band = src_geom.buffer(TOL)                     # polygon corridor
    inter = net_geom.intersection(band)             # returns lines
    return _length_km(inter)                        # now non-zero



# -----------------------------------------------------------------------------

def match_lines_detailed(
    source_lines: gpd.GeoDataFrame,
    network_lines: gpd.GeoDataFrame,
    *,
    buffer_distance: float = 0.05,   # degrees – converted internally
    snap_distance:   float = 0.010,  # degrees – converted internally
    direction_threshold: float = 0.65,
    enforce_voltage_matching: bool = False,
    dataset_name: str = "DLR",
    merge_segments: bool = True,
    max_matches_per_source: int = 20,
) -> pd.DataFrame:
    """Match *source_lines* (long) with connected *network_lines* (shorter).

    All geometrical work is done in **metres** (EPSG:3857).  Electrical
    parameters of a source line are allocated to every matched network
    segment proportionally to the overlap length.
    """

    logger.info(
        f"Matching {dataset_name} (buffer {buffer_distance}° / snap {snap_distance}° / dir ≥ {direction_threshold})")

    # ------------------------------------------------------------------
    # 1. Re‑project to a metric CRS (Web‑Mercator) -----------------------
    if source_lines.crs is None:
        source_lines = source_lines.set_crs(4326)
    if network_lines.crs is None:
        network_lines = network_lines.set_crs(4326)

    if source_lines.crs.is_geographic:
        source_lines = source_lines.to_crs(3857)
    if network_lines.crs.is_geographic:
        network_lines = network_lines.to_crs(3857)

    # convert angular distances to metres once
    DEG2M = 111_000.0
    buffer_m = buffer_distance * DEG2M
    snap_m   = snap_distance   * DEG2M

    # ------------------------------------------------------------------
    # 2. Optional segment merge (expects a helper in the same module) ----
    if merge_segments:
        try:
            merged, _ = preprocess_network_segments(
                network_lines,
                max_gap=snap_distance * 2,  # degrees, fine here
                min_dir_similarity=0.8,
            )
            network_lines = merged
            logger.info(f"Segment merge: {len(merged)} rows after merge")
        except Exception as e:
            logger.warning(f"Segment merge failed – using raw segments: {e}")

    # ------------------------------------------------------------------
    # 3. Spatial index for network lines --------------------------------
    try:
        net_sindex = network_lines.sindex
    except Exception as e:
        logger.warning(f"Spatial index failed ({e}) – falling back to brute force")
        net_sindex = None

    # helper to normalise per‑km parameters -----------------------------
    def _per_km(row, field):
        if row.get(f"{field}_per_km", 0):
            return row[f"{field}_per_km"]
        length_km = row.get("length", _length_km(row.geometry))
        return row.get(field, 0) / length_km if length_km else 0

    # ------------------------------------------------------------------
    matches = []

    for _, s_row in source_lines.iterrows():
        s_id   = str(s_row.get("id", _))
        s_geom = s_row.geometry
        if s_geom is None or s_geom.is_empty:
            continue

        s_len_km = _length_km(s_geom)
        s_volt   = s_row.get("v_nom", 0)

        buf = s_geom.buffer(max(buffer_m, min(5_000, s_len_km * 1000 * 0.0005)))

        # candidate network rows
        if net_sindex is not None:
            cand_idx = list(net_sindex.intersection(buf.bounds))
            cand = network_lines.iloc[cand_idx]
        else:
            cand = network_lines.copy()

        cand = cand[cand.geometry.intersects(buf)]
        if cand.empty:
            continue

        # direction + voltage filtering
        keep = []
        for n_idx, n_row in cand.iterrows():
            if direction_threshold:
                from src.matching.utils import direction_similarity
                if direction_similarity(s_geom, n_row.geometry) < direction_threshold:
                    continue
            if enforce_voltage_matching:
                n_volt = n_row.get("v_nom", 0)
                if not (s_volt == n_volt or {s_volt, n_volt} == {380, 400}):
                    continue
            keep.append(n_idx)

        if not keep:
            continue
        cand = cand.loc[keep]

        # compute overlaps (km) ---------------------------------------
        cand["overlap_km"] = cand.geometry.apply(lambda g: _overlap_km(s_geom, g))
        cand = cand[cand["overlap_km"] > 0]
        if cand.empty:
            continue

        # connectivity graph in this local subset ----------------------
        graph = _segment_graph(cand, snap_m)
        for component in nx.connected_components(graph):
            group = cand.loc[list(component)].copy()
            tot_ov = group["overlap_km"].sum()
            if tot_ov == 0:
                continue

            for _, n_row in group.iterrows():
                n_id   = str(n_row.get("id", _))
                n_len  = _length_km(n_row.geometry)
                ov_km  = n_row["overlap_km"]
                share  = ov_km / tot_ov if tot_ov else 0.0

                matches.append({
                    "dlr_id": s_id,
                    "network_id": n_id,
                    "overlap_km": ov_km,
                    "dlr_length_km": s_len_km,
                    "network_length_km": n_len,
                    "allocated_r": s_row.get("r", 0) * share,
                    "allocated_x": s_row.get("x", 0) * share,
                    "allocated_b": s_row.get("b", 0) * share,
                    "dlr_r": s_row.get("r", 0),
                    "dlr_x": s_row.get("x", 0),
                    "dlr_b": s_row.get("b", 0),
                    "network_r": n_row.get("r", 0),
                    "network_x": n_row.get("x", 0),
                    "network_b": n_row.get("b", 0),
                    "dlr_voltage": s_volt,
                    "network_voltage": n_row.get("v_nom", 0),
                    "dataset": dataset_name,
                })

    if not matches:
        logger.warning(f"No matches produced for {dataset_name}")
        return pd.DataFrame()

    df = pd.DataFrame(matches)

    # one best match per network segment -------------------------------
    df.sort_values("overlap_km", ascending=False, inplace=True)
    df = df.drop_duplicates(subset="network_id", keep="first")

    logger.info(f"{dataset_name}: {len(df)} matches for {len(source_lines)} source lines")
    return df
