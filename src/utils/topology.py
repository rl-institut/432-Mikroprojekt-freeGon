# src/utils/topology.py       (final robust version)
import geopandas as gpd
from shapely.geometry import LineString, MultiLineString, Point
from shapely.ops import linemerge, unary_union, snap

def _endpoints(geom):
    if geom.is_empty:
        return []
    if isinstance(geom, LineString):
        coords = list(geom.coords)
        return [Point(coords[0]), Point(coords[-1])]
    if isinstance(geom, MultiLineString):
        pts = []
        for part in geom.geoms:
            if part.is_empty:
                continue
            coords = list(part.coords)
            pts.append(Point(coords[0]))
            pts.append(Point(coords[-1]))
        return pts
    return []                 # unsupported geometry (Polygon etc.)

def reconnect_segments(gdf: gpd.GeoDataFrame,
                       id_col: str = "id",
                       tolerance_deg: float = 0.001) -> gpd.GeoDataFrame:
    """Snap close end-points together and dissolve segments per `id_col`."""
    if gdf.empty:
        return gdf.copy()

    # ---------- 1) snap -----------------------------------------------------
    endpoints = []
    for geom in gdf.geometry:
        endpoints.extend(_endpoints(geom))

    endpoint_union = unary_union(endpoints)          # MultiPoint
    snapped_geoms = [
        snap(geom, endpoint_union, tolerance_deg) if not geom.is_empty else geom
        for geom in gdf.geometry
    ]
    gdf_snapped = gdf.copy()
    gdf_snapped.geometry = snapped_geoms

    # ---------- 2) dissolve -------------------------------------------------
    dissolved_rows = []
    for _id, grp in gdf_snapped.groupby(id_col, dropna=False):
        unioned = unary_union(grp.geometry)

        # only merge if unioned is multi-part and mergeable
        if isinstance(unioned, (MultiLineString, list, tuple)):
            try:
                merged = linemerge(unioned)
            except ValueError:
                merged = unioned        # parts don’t touch – keep as is
        else:                           # already a LineString
            merged = unioned

        attrs = grp.iloc[0].drop("geometry").to_dict()
        dissolved_rows.append({**attrs, "geometry": merged, id_col: _id})

    return gpd.GeoDataFrame(dissolved_rows,
                            geometry="geometry",
                            crs=gdf.crs).reset_index(drop=True)