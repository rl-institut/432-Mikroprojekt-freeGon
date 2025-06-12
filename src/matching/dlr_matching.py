# -*- coding: utf-8 -*-
"""
DLR ↔ Network line matcher – _compat version_
--------------------------------------------
* metric CRS everywhere  (EPSG:3035 – ETRS-LAEA Europe)
* two-stage buffer:      primary = buffer_distance · 111 000 m
                         secondary = primary × 3  (fallback)
* coarse direction filter on both passes
* endpoint snap graph to join broken segments (snap_distance)
* optional voltage filter  (enforce_voltage_matching)
* overlap corridor half-width = TOL  (metres)
* result rows are **individual network segments** – merged corridors are
  automatically split and electrical parameters allocated **proportional
  to overlap length**, never more than 115 % of the DLR length.

The external API is 100 % compatible with the older matcher, so
`run_matching.py` can stay unchanged.
"""
from __future__ import annotations

import itertools
import logging
from typing import Dict, List

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from shapely.geometry import LineString, MultiLineString, Point

# ─────────────────────────────  constants  ────────────────────────────── #
METRIC_CRS            = 3035          # metric projection for Europe
TOL                   = 700           # corridor half-width  (metres)
MAX_DLR_OVERRUN       = 1.15          # at most +15 % of dlr_length allocated
MIN_SHARE_ABS   =  2_000   # metres  – we want at least 2 km overlap
MIN_SHARE_REL   =  0.10    #       – or ≥ 10 % of dlr_length, whichever is bigger


logger = logging.getLogger(__name__)

# ───────────────────────────  tiny helpers  ───────────────────────────── #

def _length_km(g) -> float:
    if g is None or g.is_empty:
        return 0.0
    if g.geom_type == "LineString":
        return g.length / 1_000
    if g.geom_type == "MultiLineString":
        return sum(p.length for p in g.geoms) / 1_000
    return 0.0


def _overlap_km(src, tgt) -> float:
    """Length of *tgt* lying inside a ±TOL corridor around *src* (km)."""
    inter = tgt.intersection(src.buffer(TOL))
    return _length_km(inter)


def _dir_sim(a, b) -> float:
    """Absolute cosine between the main vectors of *a* and *b*."""
    def _vec(line):
        if line.geom_type == "MultiLineString":
            line = max(line.geoms, key=lambda p: p.length)
        c = list(line.coords)
        v = np.array(c[-1]) - np.array(c[0])
        n = np.linalg.norm(v)
        return v / n if n else v
    return float(abs(np.clip(np.dot(_vec(a), _vec(b)), -1, 1)))


def _endpts(g):
    if g is None or g.is_empty:
        return []
    if g.geom_type == "LineString":
        c = list(g.coords);                      return [Point(c[0]), Point(c[-1])]
    if g.geom_type == "MultiLineString":
        out = []
        for part in g.geoms:
            c = list(part.coords);  out += [Point(c[0]), Point(c[-1])]
        return out
    return []

# ───────────────────────────  public pre-merger  ────────────────────── #
def preprocess_network_segments(
    net: gpd.GeoDataFrame,
    *,
    max_gap: float = 600.0,          # metres – same default as SNAP_M
    min_dir_similarity: float = 0.8,
) -> tuple[gpd.GeoDataFrame, Dict[str, str]]:
    """
    Tiny convenience wrapper that

    1. re-projects *net* to the metric CRS if necessary,
    2. calls the private `_merge_segments` helper, and
    3. builds a mapping **orig-id → merged-id** that older code still needs.

    Parameters
    ----------
    net : GeoDataFrame
        Raw network segments.
    max_gap : float
        Maximum endpoint distance (in **metres**) to snap segments together.
    min_dir_similarity : float
        Minimum direction cosine (0–1) for two segments to be considered
        aligned.

    Returns
    -------
    merged_gdf : GeoDataFrame
        Network lines after the light-weight merge.
    id_map : Dict[str, str]
        Maps every **original** ``id`` that was merged to the new
        ``merged_XXX`` id.  Un-merged rows are **not** included.
    """

    # ---- CRS ----------------------------------------------------------------
    if net.crs is None or net.crs.is_geographic:
        net = net.to_crs(METRIC_CRS)      # 3035

    # ---- run the merger -----------------------------------------------------
    merged = _merge_segments(net, max_gap, min_dir_similarity)

    # ---- build mapping orig-id → merged-id ----------------------------------
    id_map: Dict[str, str] = {}
    merged_only = merged[merged["is_merged"]]

    # every merged row is composed of many original geometries; we stored
    # the originals’ endpoints in the geometry but we now need the ids.
    # simplest: spatial join to capture all originals that lie exactly on
    # the new MultiLineString (fast – they share coordinates).
    joined = gpd.sjoin(net[["id", "geometry"]],
                       merged_only[["id", "geometry"]],
                       how="inner",
                       predicate="intersects")

    for orig, new in zip(joined["id_left"], joined["id_right"]):
        id_map[str(orig)] = str(new)

    return merged, id_map



# ───────────────────────  segment pre-merger  ─────────────────────────── #

def _merge_segments(net: gpd.GeoDataFrame,
                    snap_m: float,
                    min_sim: float = .8) -> gpd.GeoDataFrame:
    """Very lightweight merger: joins segments whose *end-points* touch within
    *snap_m* **and** whose direction similarity ≥ *min_sim*."""
    if net.empty:
        return net

    sindex = net.sindex
    G = nx.Graph()
    for i, g in enumerate(net.geometry):
        G.add_node(i)
        for p in _endpts(g):
            for j in sindex.intersection(p.buffer(snap_m).bounds):
                if j == i or G.has_edge(i, j):
                    continue
                gj = net.geometry.iloc[int(j)]
                if any(p.distance(q) <= snap_m for q in _endpts(gj)) \
                   and _dir_sim(g, gj) >= min_sim:
                    G.add_edge(i, j)

    rows, seen = [], set()
    for cid, comp in enumerate(nx.connected_components(G), 1):
        if len(comp) == 1:
            continue

        geoms: list[LineString] = []
        ln_m = 0.0
        origs: list[str] = []  # ← NEW – collect original ids

        for idx in comp:
            r = net.iloc[int(idx)]
            g = r.geometry
            ln_m += _length_km(g) * 1_000
            origs.append(str(r.id))  # ← NEW
            if g.geom_type == "MultiLineString":
                geoms.extend(list(g.geoms))
            else:
                geoms.append(g)
            seen.add(idx)

        rows.append({
            "id": f"merged_{cid}",
            "geometry": MultiLineString(geoms) if len(geoms) > 1 else geoms[0],
            "length": ln_m / 1_000,
            "is_merged": True,
            "orig_ids": ",".join(origs),  # ← NEW
        })

    merged = gpd.GeoDataFrame(rows, crs=net.crs)
    single = net.loc[[i for i in net.index if i not in seen]].copy()
    single["is_merged"] = False
    return pd.concat([merged, single], ignore_index=True)

# ─────────────────────────  local connectivity graph  ────────────────── #
def _segment_graph(sub: gpd.GeoDataFrame, snap_m: float) -> nx.Graph:
    """Return an undirected graph whose nodes are *rows of* sub and whose
    edges connect segments that have endpoints ≤ snap_m metres apart."""
    G = nx.Graph()
    endpoints: list[tuple[int, list[Point]]] = [
        (idx, _endpts(geom)) for idx, geom in zip(sub.index, sub.geometry)
    ]

    for (i, ends_i), (j, ends_j) in itertools.combinations(endpoints, 2):
        if any(a.distance(b) <= snap_m for a in ends_i for b in ends_j):
            G.add_edge(i, j)

    return G
# ─────────────────────────────────────────────────────────────────────── #

# ─────────────────────────  main matcher  ─────────────────────────────── #

def match_lines_detailed(                           #  ⟸  *unchanged*
    source_lines: gpd.GeoDataFrame,
    network_lines: gpd.GeoDataFrame,
    *,
    buffer_distance: float      = .05,              # deg
    snap_distance:   float      = .010,             # deg
    direction_threshold: float  = .75,
    enforce_voltage_matching: bool = True,
    dataset_name: str = "DLR",
    merge_segments: bool = True,
    max_matches_per_source: int = 20,               # (kept for API-compat)
) -> pd.DataFrame:
    """DLR ↔ network matcher (API compatible with the old version)."""

    # ── CRS normalisation ────────────────────────────────────────────── #
    if source_lines.crs  is None or source_lines.crs.is_geographic:
        source_lines  = source_lines.to_crs(METRIC_CRS)
    if network_lines.crs is None or network_lines.crs.is_geographic:
        network_lines = network_lines.to_crs(METRIC_CRS)

    # ── optional segment merge ───────────────────────────────────────── #
    if merge_segments and snap_distance:
        network_lines = _merge_segments(network_lines,
                                        snap_distance * 111_000)

    sindex = network_lines.sindex
    records: List[dict] = []

    primary_m   = buffer_distance * 111_000          # ≈ deg→m
    fallback_m  = primary_m * 3                      # ~ ×3
    passes      = [(primary_m,  direction_threshold),
                   (fallback_m, max(.90, direction_threshold))]

    for _, s in source_lines.iterrows():
        s_id, s_g = str(s.id), s.geometry
        if s_g is None or s_g.is_empty:
            continue
        s_len = _length_km(s_g)
        s_v   = s.get("v_nom", 0)

        for buf_m, dir_thr in passes:
            ring = s_g.buffer(buf_m)
            cand_idx = list(sindex.intersection(ring.bounds))
            cand = network_lines.iloc[cand_idx]
            if cand.empty:
                continue

            good = []
            for n_idx, n in cand.iterrows():
                if _dir_sim(s_g, n.geometry) < dir_thr:
                    continue
                if enforce_voltage_matching:
                    nv = n.get("v_nom", 0)
                    if nv and s_v and (nv != s_v and {nv, s_v} != {380, 400}):
                        continue
                good.append(n_idx)
            if not good:
                continue
            cand = cand.loc[good].copy()
            cand["ov_km"] = cand.geometry.apply(lambda g: _overlap_km(s_g, g))
            cand = cand[cand.ov_km > 0]
            if cand.empty:
                continue

            # connectivity graph in ring
            Gsub = _segment_graph(cand, snap_distance*111_000)
            for comp in nx.connected_components(Gsub):
                grp = cand.loc[list(comp)].copy()
                grp["score"] = grp.apply(
                    lambda r: .6*_dir_sim(s_g, r.geometry) +
                              .4*min(r.ov_km/s_len,
                                     r.ov_km/_length_km(r.geometry)),
                    axis=1)
                grp.sort_values("score", ascending=False, inplace=True)

                acc, acc_len = [], 0.0
                for _, r in grp.iterrows():

                    # -------- NEW: minimum useful overlap --------------------------
                    if (r.ov_km * 1_000 < MIN_SHARE_ABS) or (r.ov_km < s_len * MIN_SHARE_REL):
                        continue
                    # ----------------------------------------------------------------

                    # existing length-cap
                    if acc_len + r.ov_km > s_len * MAX_DLR_OVERRUN:
                        continue

                    acc.append(r)
                    acc_len += r.ov_km

                if not acc:
                    continue
                tot_ov = sum(r.ov_km for r in acc)
                for r in acc:
                    share = r.ov_km / tot_ov if tot_ov else 0
                    records.append({
                        "dlr_id": s_id,
                        "network_id": str(r.id),
                        "merged_id": str(r.id) if r.get("is_merged", False) else "",
                        "overlap_km": r.ov_km,
                        "dlr_length_km": s_len,
                        "network_length_km": _length_km(r.geometry),
                        "allocated_r": s.get("r", 0)*share,
                        "allocated_x": s.get("x", 0)*share,
                        "allocated_b": s.get("b", 0)*share,
                        "dlr_r": s.get("r", 0),
                        "dlr_x": s.get("x", 0),
                        "dlr_b": s.get("b", 0),
                        "network_r": r.get("r", 0),
                        "network_x": r.get("x", 0),
                        "network_b": r.get("b", 0),
                        "dlr_voltage": s_v,
                        "network_voltage": r.get("v_nom", 0),
                        "dataset": dataset_name,
                    })
            if records:      # stop after successful primary pass
                break

    if not records:
        logger.warning("No matches produced for %s", dataset_name)
        return pd.DataFrame()

    df = pd.DataFrame(records)

    # ── split merged corridors into individual segments ─────────────── #
    if df.merged_id.any():
        len_lookup = (network_lines
                      .assign(_m=network_lines.geometry.length)
                      .astype({"id": str})
                      .set_index("id")["_m"]
                      .to_dict())

        exploded: List[dict] = []
        for _, row in df.iterrows():
            if not row.merged_id:           # already a single segment
                exploded.append(row)
                continue

            try:
                orig_str = network_lines.loc[network_lines.id == row.merged_id, "orig_ids"].iloc[0]
                comp_ids = [i for i in orig_str.split(",") if i]
            except Exception:
                # fallback – treat merged_xx as a single segment
                comp_ids = [row.merged_id]

            tot_m = sum(len_lookup.get(i, 0) for i in comp_ids)
            for seg_id in comp_ids:
                seg_m = len_lookup.get(seg_id, 0)
                share = seg_m / tot_m if tot_m else 0
                new = row.copy()
                new["network_id"]        = seg_id
                new["network_length_km"] = seg_m / 1_000
                new["merged_length_km"]  = tot_m / 1_000
                new["allocated_r"]       = row.dlr_r * share
                new["allocated_x"]       = row.dlr_x * share
                new["allocated_b"]       = row.dlr_b * share
                exploded.append(new)

        df = pd.DataFrame(exploded)

    logger.info("%s: %d matches / %d DLR lines", dataset_name,
                len(df), df.dlr_id.nunique())
    return df
