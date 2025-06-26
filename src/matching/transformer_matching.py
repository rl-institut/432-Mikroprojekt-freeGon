"""
Match DLR ↔ Network transformers by proximity (EPSG:3035 metres).

Returns
-------
match_df         : pandas.DataFrame[dlr_id, network_id, dist_m]
matched_dlr_ids  : set[str]
matched_net_ids  : set[str]
"""

from __future__ import annotations
import logging
import pandas as pd
import geopandas as gpd

log = logging.getLogger(__name__)


def _project(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Project to EU-LAEA (EPSG:3035) if not already metric."""
    return gdf if gdf.crs.to_epsg() == 3035 else gdf.to_crs(epsg=3035)


def match_transformers(
    dlr_gdf: gpd.GeoDataFrame,
    net_gdf: gpd.GeoDataFrame,
    buffer_m: float = 1000.0,
):
    if dlr_gdf.empty or net_gdf.empty:
        log.warning("Transformer matcher: one or both inputs are empty.")
        return pd.DataFrame(columns=["dlr_id", "network_id", "dist_m"]), set(), set()

    d3035 = _project(dlr_gdf)
    n3035 = _project(net_gdf)

    # spatial index speeds things up
    n_sindex = n3035.sindex
    matches, used_dlr, used_net = [], set(), set()

    for d_idx, d_row in d3035.iterrows():
        d_id = d_row["id"]
        if d_id in used_dlr:
            continue
        buf = d_row.geometry.buffer(buffer_m)

        # candidate net indices
        cand_idx = list(n_sindex.query(buf, predicate="intersects"))
        cand_idx = [i for i in cand_idx if n3035.iloc[i]["id"] not in used_net]
        if not cand_idx:
            continue

        # pick closest
        cand_geoms = n3035.iloc[cand_idx].geometry
        distances = cand_geoms.distance(d_row.geometry)
        best_i = distances.idxmin()
        best_dist = float(distances.loc[best_i])

        if best_dist <= buffer_m:
            n_id = n3035.at[best_i, "id"]
            matches.append(
                dict(dlr_id=d_id, network_id=n_id, dist_m=best_dist)
            )
            used_dlr.add(d_id)
            used_net.add(n_id)

    log.info("Transformer matcher: %d pairs (%d/%d DLR, %d/%d Net)",
             len(matches), len(used_dlr), len(dlr_gdf),
             len(used_net), len(net_gdf))

    return pd.DataFrame(matches), used_dlr, used_net
