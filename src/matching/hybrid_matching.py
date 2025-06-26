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

# -------------------------------------------------------------
#  stroke-merging on the network side
# -------------------------------------------------------------
def _merge_strokes(net: gpd.GeoDataFrame,
                   snap_m: float = SNAP_M,
                   min_cos: float = MIN_DIR_COS) -> gpd.GeoDataFrame:
    """Merge end-to-end network segments that touch within *snap_m* and share
    a similar direction (+ equal voltage).  Returns a *copy* of *net*."""
    if net.empty:
        return net

    net = net.reset_index(drop=True).copy()
    sindex = STRtree(net.geometry.values)
    G = nx.Graph()

    for idx, geom in enumerate(net.geometry):
        G.add_node(idx)
        for p in _endpts(geom):
            for j in sindex.query(p.buffer(snap_m)):
                if j == idx or G.has_edge(idx, j):            # same segment
                    continue
                if net.at[idx, "v_nom"] != net.at[j, "v_nom"]:
                    continue
                if _dir_cos(geom, net.geometry.iloc[j]) < min_cos:
                    continue
                if any(p.distance(q) <= snap_m for q in _endpts(net.geometry.iloc[j])):
                    G.add_edge(idx, j)

    merged_rows, visited = [], set()
    for cid, comp in enumerate(nx.connected_components(G), 1):
        if len(comp) == 1:                    # nothing to merge
            continue
        geoms = [net.geometry.iloc[i] for i in comp]
        merged = linemerge(unary_union(geoms))
        merged_rows.append(dict(
            id      = f"stroke_{cid}",
            geometry= merged,
            v_nom   = net.at[next(iter(comp)), "v_nom"],
            src_ids = ",".join(str(net.at[i, "id"]) for i in comp)
        ))
        visited.update(comp)

    leftover = net.loc[[i for i in net.index if i not in visited]].copy()
    return pd.concat([leftover, gpd.GeoDataFrame(merged_rows, crs=net.crs)],
                     ignore_index=True)

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

    tree = STRtree(cand.geometry.values)
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
                if sv and nv and (sv != nv and {sv, nv} != {380, 400}):
                    continue
            if _dir_cos(s_geom, g) < dir_thr:
                continue
            if use_end and not _endpts_inside(s_geom, g, buf_m):
                continue
            keep.append(i)
        if not keep:
            continue

        # ── fine metrics + hard overlap gate ───────────────────────
        metr = {}
        for i in keep:
            m = sim_fn(s_geom, cand.geometry.iloc[i])     # ← .iloc !
            if mode == "real":
                if not (m["overlap"] >= OVERLAP_MIN_FRAC or
                        m["overlap_km"]*1_000 >= OVERLAP_MIN_ABS_M):
                    continue
            metr[i] = m
        if not metr:
            continue

        # ── scoring & greedy accumulation ──────────────────────────
        scored = []
        for i, m in metr.items():
            sc = score_fn(m, s_len) if mode == "real" else score_fn(m)
            scored.append((i, sc, m))
        scored.sort(key=lambda t: -t[1])         # best first

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
                network_id = str(cand.id.iloc[i]),          # ← .iloc !
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

    # ── ensure numeric voltage everywhere ─────────────────────────────
    for g in (dlr_m, net_m, chords):
        g["v_nom"] = g["v_nom"].fillna(0).astype(float)

    # ─────────────────────────────────────────────────────────────────
    # 1️⃣  match vs. real geometry
    # ─────────────────────────────────────────────────────────────────
    real_rows, best_real = _match(
        dlr      = dlr_m,
        cand     = net_m,
        buf_m    = buf_m,
        dir_thr  = dir_thr,
        enforceV = enforce_v,
        use_end  = use_end,
        mode     = "real",
    )

    # ─────────────────────────────────────────────────────────────────
    # 2️⃣  match vs. chord geometry
    # ─────────────────────────────────────────────────────────────────
    chord_rows, best_chord = _match(
        dlr      = dlr_m,
        cand     = chords,
        buf_m    = buf_m,
        dir_thr  = dir_thr,
        enforceV = enforce_v,
        use_end  = use_end,
        mode     = "chord",
    )

    # ── uniform column order ─────────────────────────────────────────
    COLS = [
        "match_id", "dlr_id", "network_id", "score",
        "v_dlr", "v_net",
        "overlap_km", "frechet_m", "hausdorff_m",
        "dir_cos", "endpoints_ok",
        "network_r", "network_x", "network_b",
        "allocated_r", "allocated_x", "allocated_b",
    ]

    df_real  = pd.DataFrame(real_rows , columns=COLS)
    df_chord = pd.DataFrame(chord_rows, columns=COLS)

    # ── quick log ────────────────────────────────────────────────────
    def _coverage(df):
        return 0.0 if df.empty else 100 * df.dlr_id.nunique() / len(dlr_m)

    LOGGER.info("Real matcher : %4d rows  (coverage %.1f %%)",
                len(df_real ), _coverage(df_real ))
    LOGGER.info("Chord matcher: %4d rows  (coverage %.1f %%)",
                len(df_chord), _coverage(df_chord))

    return df_real, df_chord, best_real, best_chord



