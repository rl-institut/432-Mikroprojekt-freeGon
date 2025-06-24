# ──────────────────────────────────────────────────────────────
#  src/matching/hybrid_matching.py
# ──────────────────────────────────────────────────────────────
from __future__ import annotations
import itertools, math, logging
from pathlib import Path
from typing import List, Dict, Tuple

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from shapely.geometry import LineString, MultiLineString, Point
from shapely.ops import linemerge, unary_union
from shapely.strtree import STRtree
from sklearn.neighbors import BallTree


# -------------------------------------------------------------
#  global parameters – can be overridden per-call
# -------------------------------------------------------------
METRIC_CRS          = 3035        # ETRS-LAEA Europe (metre units)
TOL                 = 600         # corridor half-width for overlap (m)
SNAP_M              = 1_200       # stroke-merger “touch” distance  (m)
MIN_DIR_COS         = .80         # …and min. direction similarity
OVERLAP_MIN_FRAC    = .10         # Min. 10 % of shorter line must overlap
OVERLAP_MIN_ABS_M   = 2_000       # Min. 2 km overlap anyway
SCORE_W             = dict(overlap=.40, frechet=.30, hausdorff=.10,
                           direction=.15, endpoints=.05)
MAX_OVERRUN         = 1.15        # never allocate >115 % of DLR length
LOGGER = logging.getLogger(__name__)


# ──────────────────────  geometry helpers  ──────────────────────
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

def _length_km(g) -> float:
    return (g.length if g is not None else 0.0) / 1_000

def _frechet(a: LineString, b: LineString) -> float:
    try:
        return float(a.frechet_distance(b))
    except Exception:                       # Shapely <2 → graceful fallback
        return float(a.hausdorff_distance(b))

def _hausdorff(a: LineString, b: LineString) -> float:
    return float(a.hausdorff_distance(b))

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

# ────────────────────  stroke-merging (network)  ────────────────────
def _merge_strokes(net: gpd.GeoDataFrame,
                   snap_m: float = SNAP_M,
                   min_cos: float = MIN_DIR_COS) -> gpd.GeoDataFrame:
    """
    Merge end-to-end segments that (i) touch within *snap_m* metres and
    (ii) have similar direction.  Voltage must be equal.
    """
    if net.empty: return net
    net = net.reset_index(drop=True).copy()
    sindex = STRtree(net.geometry.values)
    G = nx.Graph()

    for idx, geom in enumerate(net.geometry):
        G.add_node(idx, voltage=net.at[idx, "v_nom"])
        for p in _endpts(geom):
            for j in sindex.query(p.buffer(snap_m)):
                if j == idx or G.has_edge(idx, j): continue
                if net.at[idx, "v_nom"] != net.at[j, "v_nom"]:  # voltage mismatch
                    continue
                if _dir_cos(geom, net.geometry.iloc[j]) < min_cos:
                    continue
                if any(p.distance(q) <= snap_m for q in _endpts(net.geometry.iloc[j])):
                    G.add_edge(idx, j)

    merged_rows = []
    visited = set()
    for cid, comp in enumerate(nx.connected_components(G), 1):
        if len(comp) == 1:
            continue
        geoms = [net.geometry.iloc[i] for i in comp]
        merged_geom = linemerge(unary_union(geoms))
        merged_rows.append(dict(id=f"stroke_{cid}",
                                geometry=merged_geom,
                                v_nom=net.at[next(iter(comp)), "v_nom"],
                                src_ids=",".join(str(net.at[i, "id"]) for i in comp)))
        visited.update(comp)

    leftover = net.loc[[i for i in net.index if i not in visited]].copy()
    return pd.concat([leftover, gpd.GeoDataFrame(merged_rows, crs=net.crs)],
                     ignore_index=True)

from shapely.ops import linemerge

def _as_chord(ls):
    """
    Return a straight two-point LineString that connects the first and
    last vertex of *ls* – works for LineString **and** MultiLineString.
    Empty / None geometries are passed through unchanged.
    """
    if ls is None or ls.is_empty:
        return ls

    # If it is a MultiLineString try to glue the pieces together first
    if ls.geom_type == "MultiLineString":
        ls_m = linemerge(ls)            # may still be MultiLineString
        if ls_m.geom_type == "LineString":
            ls = ls_m
        else:
            # take the very first and very last coordinate among parts
            first_part  = min(ls.geoms, key=lambda g: g.bounds[0])  # arbitrary but stable
            last_part   = max(ls.geoms, key=lambda g: g.bounds[2])
            start = list(first_part.coords)[0]
            end   = list(last_part.coords)[-1]
            return LineString([start, end])

    # now we are sure we have a LineString
    coords = list(ls.coords)
    return LineString([coords[0], coords[-1]])



# ────────────────────────────  scoring  ────────────────────────────
def _similarity_metrics(src: LineString, cand: LineString) -> dict:
    """
    Compute the similarity metrics **for the current geometry flavour**.
    When we use CHORDS we can’t measure corridor–overlap any more, so we
    just fill it with 0.0 but keep the key so that later code works.
    """
    return dict(
        overlap    = 0.0,                    # <── placeholder
        frechet    = _frechet(src, cand),
        hausdorff  = _hausdorff(src, cand),
        direction  = _dir_cos(src, cand),
        endpoints  = int(_endpts_inside(src, cand, TOL*1.5))
    )


def _score(m: dict) -> float:
    # Simple two-factor score for chord version
    return (
        0.75 * m["endpoints"] +               # 0 or 1
        0.25 * m["direction"]                 # cosine ∈ [0…1]
    )



# ─────────────────────────  public function  ─────────────────────────
def match_lines_hybrid(
        dlr: gpd.GeoDataFrame,
        net: gpd.GeoDataFrame,
        *,
        cfg: dict           # section from YAML  (see example below)
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Returns (matches_df,  best_partner_mapping).
    *matches_df* contains **one row per matched network segment**,
    with a shared `match_id` to indicate which belong to the same DLR line.
    """
    # --- configuration --------------------------------------------------
    buf_deg = cfg.get("buffer_distance", .03)      # degrees
    snap_deg = cfg.get("snap_distance",  .010)
    dir_thr  = cfg.get("direction_threshold", .45)
    use_end  = cfg.get("use_endpoint_filter", False)
    enforceV = cfg.get("enforce_voltage_matching", True)
    # --------------------------------------------------------------------

    # 1) project to metric CRS
    dlr  = _to_metric(dlr)
    net0 = _to_metric(net)

    # 2) stroke merge on the **network** side to reduce fragmentation
    net = _merge_strokes(net0, snap_deg*111_000, min_cos=dir_thr)

    # 2b) create straight-line (chord) geometries  ────────────────────
    net_chord = net.copy()
    net_chord["geometry"] = net_chord.geometry.apply(_as_chord)

    dlr_chord = dlr.copy()
    dlr_chord["geometry"] = dlr_chord.geometry.apply(_as_chord)


    # 3) voltage clean-up
    for g in (dlr_chord, net_chord):
        g["v_nom"] = g["v_nom"].fillna(0).astype(float)

    # build spatial index on net
    tree = STRtree(net_chord.geometry.values)

    rows, best_score = [], {}

    for _, s in dlr_chord.iterrows():
        s_geom = s.geometry
        if s_geom is None or s_geom.is_empty: continue
        s_len  = s_geom.length
        buf_m  = buf_deg * 111_000
        buf_geom = s_geom.buffer(buf_m)
        cand_idxs = list(map(int, tree.query(buf_geom)))
        if not cand_idxs:
            continue

        # --- coarse voltage & dir filter ------------------------------
        cand_idxs2 = []
        for i in cand_idxs:
            g = net_chord.geometry[i]
            if enforceV:
                sv, nv = s.v_nom, net_chord.v_nom[i]
                if sv and nv and (sv != nv and {sv, nv} != {380, 400}):
                    continue
            if _dir_cos(s_geom, g) < dir_thr:   # coarse dir check
                continue
            if use_end and not _endpts_inside(s_geom, g, buf_m):
                continue
            cand_idxs2.append(i)
        if not cand_idxs2:
            continue

        # --- compute similarity & score ------------------------------
        metr = {i: _similarity_metrics(s_geom, net_chord.geometry[i])
                for i in cand_idxs2}

        # --- discard clearly bad overlaps -----------------------------
        #if "overlap" in m and (
                #m["overlap"] < OVERLAP_MIN_FRAC and
                #m["overlap"] * s_geom.length < OVERLAP_MIN_ABS_M):
            #continue

        # --- compute similarity for every candidate -----------------
        metr: dict[int, dict] = {
            i: _similarity_metrics(s_geom, net_chord.geometry[i])
            for i in cand_idxs2
        }

        # ── OPTIONAL: discard very poor overlaps (only useful
        #              if you ever run with real geometries)
        keep = {}
        for i, m in metr.items():
            if m["overlap"] == 0:  # <- chord mode: keep everything
                keep[i] = m
            else:
                too_little = (
                        m["overlap"] < OVERLAP_MIN_FRAC and
                        m["overlap"] * s_geom.length < OVERLAP_MIN_ABS_M
                )
                if not too_little:
                    keep[i] = m
        metr = keep

        scored = sorted(
            ((i, _score(m), m) for i, m in metr.items()),
            key=lambda t: -t[1]
        )

        acc, total_ov = [], 0.0
        for i, sc, m in scored:
            if total_ov + m["overlap"]*s_geom.length/1_000 > MAX_OVERRUN * s_len/1_000:
                continue
            acc.append((i, sc, m))
            total_ov += m["overlap"]*s_geom.length/1_000

        if not acc:
            # fallback – take the highest-score candidate
            i_best, sc_best, m_best = max(scored, key=lambda t: t[1])
            acc = [(i_best, sc_best, m_best)]

        # save best network partner id
        best_score[str(s.id)] = str(net_chord.id[acc[0][0]])

        match_id = f"{s.id}"
        for i, sc, m in acc:
            share = 1.0 / len(acc) if total_ov == 0 else (
                    m["overlap"] * s_geom.length / 1_000 / total_ov)
            rows.append(dict(
                match_id=match_id,
                dlr_id=str(s.id),
                network_id=str(net_chord.id[i]),
                score=sc,
                v_dlr=s.v_nom,
                v_net=net_chord.v_nom[i],
                overlap_km=m["overlap"] * s_geom.length / 1_000,
                frechet_m=m["frechet"],
                hausdorff_m=m["hausdorff"],
                dir_cos=m["direction"],
                endpoints_ok=bool(m["endpoints"]),
                # ── NEW – provide the six expected columns ───────────
                network_r=net_chord.get("r", np.nan).iloc[i] if "r" in net_chord else np.nan,
                network_x=net_chord.get("x", np.nan).iloc[i] if "x" in net_chord else np.nan,
                network_b=net_chord.get("b", np.nan).iloc[i] if "b" in net_chord else np.nan,
                allocated_r=s.get("r", np.nan) * share if "r" in s else np.nan,
                allocated_x=s.get("x", np.nan) * share if "x" in s else np.nan,
                allocated_b=s.get("b", np.nan) * share if "b" in s else np.nan,
            ))

    df = pd.DataFrame(rows)
    if df.empty:
        coverage = 0.0
    else:
        coverage = 100 * df["dlr_id"].nunique() / len(dlr_chord)

    LOGGER.info(
        "Hybrid matcher: %d DLR lines, %d matched rows, %.1f %% coverage",
        len(dlr_chord), len(df), coverage
    )

    return df, best_score, net_chord