# -*- coding: utf-8 -*-
"""
DLR ↔ Network line matcher  —  robust, self-contained implementation
===================================================================

Main public entry-point
----------------------
    • **match_lines_detailed(source_lines, network_lines, …) → (df, best_dict)**

Key features
------------
* Metric CRS everywhere (ETRS-LAEA Europe – EPSG:3035)
* Two-stage search ring (primary = buffer_distance × 111 000 m, fallback = ×3)
* Coarse *direction* filter **and** optional *endpoint* corridor filter
* Lightweight segment merger (endpoint snap + direction check) before matching
* Optional voltage filter (tolerates 380 kV ↔ 400 kV substitutions)
* Per-segment *overlap* is measured inside an ± `TOL` corridor
* Greedy allocation: never more than *MAX_DLR_OVERRUN* × DLR length
* Merged corridors are exploded back to individual network segments
* Returned DataFrame contains one row **per network segment**

The file is *stand-alone* – drop it in ``src/matching/dlr_matching.py`` and make
sure your caller passes the YAML parameters through.
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

# ─────────────────────────────  constants  ───────────────────────────── #

METRIC_CRS        = 3035      # ETRS-LAEA Europe (metres)
TOL               = 800       # half-width of “overlap corridor” (m)
MAX_DLR_OVERRUN   = 1.15      # allocate ≤ 115 % of the DLR length
MIN_SHARE_ABS     = 1_500     # ≥ 1.5 km overlap …
MIN_SHARE_REL     = 0.05      # … or ≥ 5 % of DLR length (whichever is larger)

logger = logging.getLogger(__name__)

# ───────────────────────────── helpers ──────────────────────────────── #

def _pick_best(df: pd.DataFrame) -> dict[str, str]:
    """{dlr_id → network_id} with *highest score* per DLR line."""
    if df.empty:
        return {}
    idx = df.groupby("dlr_id")["score"].idxmax()
    best = df.loc[idx, ["dlr_id", "network_id"]]
    return dict(zip(best.dlr_id.astype(str), best.network_id.astype(str)))


def _length_km(g) -> float:
    if g is None or g.is_empty:
        return 0.0
    if g.geom_type == "LineString":
        return g.length / 1_000
    if g.geom_type == "MultiLineString":
        return sum(p.length for p in g.geoms) / 1_000
    return 0.0


def _overlap_km(src: LineString, tgt: LineString) -> float:
    """Length of *tgt* inside ± TOL corridor around *src* (km)."""
    inter = tgt.intersection(src.buffer(TOL))
    return float(inter.length) / 1_000


def _dir_sim(a: LineString, b: LineString, *, allow_reverse=False) -> float:
    """Cosine of the main vectors (-1…1).  If *allow_reverse*: absolute value."""
    def _vec(line):
        if line.geom_type == "MultiLineString":
            line = max(line.geoms, key=lambda p: p.length)
        c = np.asarray(line.coords)
        v = c[-1] - c[0]
        n = np.linalg.norm(v)
        return v / n if n else v
    sim = float(np.dot(_vec(a), _vec(b)))
    return abs(sim) if allow_reverse else sim


def _endpts(g):
    if g is None or g.is_empty:
        return []
    if g.geom_type == "LineString":
        c = list(g.coords)
        return [Point(c[0]), Point(c[-1])]
    if g.geom_type == "MultiLineString":
        pts: List[Point] = []
        for part in g.geoms:
            c = list(part.coords)
            pts += [Point(c[0]), Point(c[-1])]
        return pts
    return []


def _endpts_inside(src: LineString, cand: LineString, width: float) -> bool:
    """True ↔ all endpoints of *cand* lie within *width* corridor around *src*."""
    corr = src.buffer(width)
    return all(corr.contains(pt) for pt in _endpts(cand))

# ───────────────────── lightweight network corridor merge ───────────── #

def _merge_segments(net: gpd.GeoDataFrame,
                    snap_m: float,
                    min_sim: float = 0.8) -> gpd.GeoDataFrame:
    """Merge chains of tiny network segments into longer corridors.

    Two segments are merged if **any** endpoints are ≤ *snap_m* m apart
    *and* the direction similarity ≥ *min_sim*.  Keeps track of original IDs.
    """
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
                if any(p.distance(q) <= snap_m for q in _endpts(gj)) and _dir_sim(g, gj) >= min_sim:
                    G.add_edge(i, j)

    rows, seen = [], set()
    for cid, comp in enumerate(nx.connected_components(G), 1):
        if len(comp) == 1:
            continue
        geoms, ln_m, origs = [], 0.0, []
        for idx in comp:
            r = net.iloc[int(idx)]
            g = r.geometry
            ln_m += _length_km(g) * 1_000
            origs.append(str(r.id))
            geoms.extend(list(g.geoms) if g.geom_type == "MultiLineString" else [g])
            seen.add(idx)
        rows.append({
            "id": f"merged_{cid}",
            "geometry": MultiLineString(geoms) if len(geoms) > 1 else geoms[0],
            "length": ln_m / 1_000,
            "is_merged": True,
            "orig_ids": ",".join(origs),
        })

    merged = gpd.GeoDataFrame(rows, crs=net.crs)
    single = net.loc[[i for i in net.index if i not in seen]].copy()
    single["is_merged"] = False
    return pd.concat([merged, single], ignore_index=True)

# ─────────────────────  connectivity helper graph  ──────────────────── #

def _segment_graph(sub: gpd.GeoDataFrame, snap_m: float) -> nx.Graph:
    """Graph where edges link segments with endpoints ≤ *snap_m* m apart."""
    G = nx.Graph()
    endpoints = [(idx, _endpts(geom)) for idx, geom in zip(sub.index, sub.geometry)]
    for (i, ends_i), (j, ends_j) in itertools.combinations(endpoints, 2):
        if any(a.distance(b) <= snap_m for a in ends_i for b in ends_j):
            G.add_edge(i, j)
    return G

# ─────────────────────────  main matcher  ───────────────────────────── #

def match_lines_detailed(
    source_lines: gpd.GeoDataFrame,
    network_lines: gpd.GeoDataFrame,
    *,
    buffer_distance: float = 0.03,     # deg
    snap_distance: float   = 0.010,    # deg
    direction_threshold: float = 0.45,
    use_endpoint_filter: bool = True,
    enforce_voltage_matching: bool = True,
    dataset_name: str = "DLR",
    merge_segments: bool = True,
    max_matches_per_source: int = 20,  # kept for API-compat (unused)
):
    """Geometric + electrical matcher.  Returns ``(matches_df, best_dict)``."""
    # ────────── CRS normalisation ───────────────────────────────────── #
    if source_lines.crs is None or source_lines.crs.is_geographic:
        source_lines  = source_lines.to_crs(METRIC_CRS)
    if network_lines.crs is None or network_lines.crs.is_geographic:
        network_lines = network_lines.to_crs(METRIC_CRS)

    # ensure numeric voltages
    for gdf in (source_lines, network_lines):
        gdf["v_nom"] = gdf.get("v_nom", 0).fillna(0).astype(float)

    # ────────── optional corridor merge ─────────────────────────────── #
    if merge_segments and snap_distance:
        network_lines = _merge_segments(network_lines, snap_distance * 111_000)

    sindex  = network_lines.sindex
    records = []

    primary_m  = buffer_distance * 111_000
    fallback_m = primary_m * 3
    passes     = [(primary_m,  direction_threshold),
                  (fallback_m, max(0.90, direction_threshold))]

    for _, src in source_lines.iterrows():
        s_id, s_geom = str(src.id), src.geometry
        if s_geom is None or s_geom.is_empty:
            continue
        s_len = _length_km(s_geom)
        s_v   = src.get("v_nom", 0)

        for buf_m, dir_thr in passes:
            ring   = s_geom.buffer(buf_m)
            cand_i = list(sindex.intersection(ring.bounds))
            cand   = network_lines.iloc[cand_i]
            if cand.empty:
                continue

            # ── pre-filters (direction, endpoints, voltage) ─────────── #
            good_idx = []
            for n_idx in cand.index:
                geom_n = cand.at[n_idx, "geometry"]

                if _dir_sim(s_geom, geom_n) < dir_thr:
                    continue
                if use_endpoint_filter and not _endpts_inside(s_geom, geom_n, buf_m):
                    continue
                if enforce_voltage_matching:
                    nv = cand.at[n_idx, "v_nom"]
                    if nv and s_v and (nv != s_v and {nv, s_v} != {380, 400}):
                        continue
                good_idx.append(n_idx)

            cand = cand.loc[good_idx].copy()
            if cand.empty:
                continue

            cand["ov_km"] = cand.geometry.apply(lambda g: _overlap_km(s_geom, g))
            cand = cand[cand["ov_km"] > 0]
            if cand.empty:
                continue

            # ── connectivity graph inside ring ──────────────────────── #
            Gsub = _segment_graph(cand, snap_distance * 111_000)
            for comp in nx.connected_components(Gsub):
                grp = cand.loc[list(comp)].copy()

                grp["score"] = grp.apply(
                    lambda r: 0.60 * _dir_sim(s_geom, r.geometry) +
                              0.40 * min(r["ov_km"] / s_len,
                                         r["ov_km"] / _length_km(r.geometry)),
                    axis=1,
                )
                grp.sort_values("score", ascending=False, inplace=True)

                acc, acc_len = [], 0.0
                for _, r in grp.iterrows():
                    if (r["ov_km"] * 1_000 < MIN_SHARE_ABS) or (r["ov_km"] < s_len * MIN_SHARE_REL):
                        continue
                    if acc_len + r["ov_km"] > s_len * MAX_DLR_OVERRUN:
                        continue
                    acc.append(r)
                    acc_len += r["ov_km"]

                if not acc:
                    continue

                tot_ov = sum(r["ov_km"] for r in acc)
                for r in acc:
                    share = r["ov_km"] / tot_ov if tot_ov else 0.0
                    records.append({
                        "dlr_id":            s_id,
                        "score":             r.score,
                        "network_id":        str(r.id),
                        "merged_id":         str(r.id) if r.get("is_merged", False) else "",
                        "overlap_km":        r["ov_km"],
                        "dlr_length_km":     s_len,
                        "network_length_km": _length_km(r.geometry),
                        "allocated_r":       src.get("r", 0) * share,
                        "allocated_x":       src.get("x", 0) * share,
                        "allocated_b":       src.get("b", 0) * share,
                        "dlr_r":             src.get("r", 0),
                        "dlr_x":             src.get("x", 0),
                        "dlr_b":             src.get("b", 0),
                        "network_r":         r.get("r", 0),
                        "network_x":         r.get("x", 0),
                        "network_b":         r.get("b", 0),
                        "dlr_voltage":       s_v,
                        "network_voltage":   r.get("v_nom", 0),
                        "dataset":           dataset_name,
                    })
            # stop after first successful pass
            if any(rec["dlr_id"] == s_id for rec in records):
                break

    if not records:
        logger.warning("No matches produced for %s", dataset_name)
        return pd.DataFrame(), {}

    df = pd.DataFrame(records)

    # ── best partner dict ───────────────────────────────────────────── #
    best_partner: Dict[str, str] = _pick_best(df)

    # ── explode merged corridors back to segments ──────────────────── #
    if df.merged_id.any():
        len_lookup = (
            network_lines.assign(_m=network_lines.geometry.length)
            .astype({"id": str})
            .set_index("id")["_m"].to_dict()
        )
        exploded: List[dict] = []
        for _, row in df.iterrows():
            if not row.merged_id:
                exploded.append(row)
                continue
            try:
                orig_str = network_lines.loc[network_lines.id == row.merged_id, "orig_ids"].iloc[0]
                comp_ids = [i for i in orig_str.split(",") if i]
            except Exception:
                comp_ids = [row.merged_id]
            tot_m = sum(len_lookup.get(i, 0) for i in comp_ids)
            for seg_id in comp_ids:
                seg_m  = len_lookup.get(seg_id, 0)
                share  = seg_m / tot_m if tot_m else 0.0
                newrow = row.copy()
                newrow["network_id"]        = seg_id
                newrow["network_length_km"] = seg_m / 1_000
                newrow["merged_length_km"]  = tot_m / 1_000
                newrow["allocated_r"]       = row.dlr_r * share
                newrow["allocated_x"]       = row.dlr_x * share
                newrow["allocated_b"]       = row.dlr_b * share
                exploded.append(newrow)
        df = pd.DataFrame(exploded)

    logger.info(
        "%s: %d matches / %d source lines (%.1f %%)",
        dataset_name, len(df), df.dlr_id.nunique(),
        100 * df.dlr_id.nunique() / len(source_lines)
    )
    return df, best_partner
