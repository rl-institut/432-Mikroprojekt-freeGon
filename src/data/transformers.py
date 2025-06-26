"""
Light-weight helpers for transformer CSVs
(identical logic to your visualise_transformers tool, but bullet-proof
for MultiLineString geometries)
"""
from pathlib import Path
import pandas as pd
import geopandas as gpd
from shapely import wkt
from shapely.geometry import Point, LineString, MultiLineString
import logging

log = logging.getLogger(__name__)
EPSG = "EPSG:4326"          # always keep lat/lon


# ──────────────────────────────────────────────────────────────
# generic helpers
# ──────────────────────────────────────────────────────────────
def _as_shape(txt: str):
    """Return a Shapely geometry (or None) from one WKT string."""
    try:
        return wkt.loads(txt) if isinstance(txt, str) else None
    except Exception:
        return None


def _representative_point(g):
    """
    Convert any geometry *g* into one representative Point:

    * Point           → itself
    * LineString      → first vertex
    * MultiLineString → first vertex of first non-empty part
    * other           → centroid (representative_point)
    """
    if g is None:
        return None

    gtype = g.geom_type
    if gtype == "Point":
        return g

    if gtype == "LineString" and g.coords:
        return Point(g.coords[0])

    if gtype == "MultiLineString":
        for part in g.geoms:
            if part.coords:
                return Point(part.coords[0])
        return None          # empty multilinestring

    # fallback – works for Polygon, MultiPolygon, …
    return g.representative_point()


# ──────────────────────────────────────────────────────────────
# DLR transformers
# ──────────────────────────────────────────────────────────────
def load_dlr_transformers(csv) -> gpd.GeoDataFrame:
    df = pd.read_csv(csv)
    if "geometry" not in df.columns:
        raise ValueError(f"{csv} has no geometry column")

    df["geometry"] = df["geometry"].apply(_as_shape)
    gdf = gpd.GeoDataFrame(df[df.geometry.notnull()],
                           geometry="geometry", crs=EPSG)

    # ensure “id”
    if "id" not in gdf.columns:
        gdf["id"] = gdf.get("name",
                            pd.Series([f"dlr_{i}" for i in range(len(gdf))]))
    gdf["id"] = gdf["id"].astype(str)

    log.info("DLR-TRF: %d points loaded", len(gdf))
    return gdf


# ──────────────────────────────────────────────────────────────
# Network transformers  ← fixed here
# ──────────────────────────────────────────────────────────────
def load_network_transformers(csv) -> gpd.GeoDataFrame:
    df = pd.read_csv(csv)
    if "geom" not in df.columns:
        raise ValueError(f"{csv} has no geom column")

    df["geometry"] = df["geom"].apply(_as_shape)
    # normalise every geometry to a point
    df["geometry"] = df["geometry"].apply(_representative_point)
    gdf = gpd.GeoDataFrame(df[df.geometry.notnull()],
                           geometry="geometry", crs=EPSG)

    # ensure “id”
    if "id" not in gdf.columns:
        gdf["id"] = [f"net_{i}" for i in range(len(gdf))]
    gdf["id"] = gdf["id"].astype(str)

    log.info("NET-TRF: %d points loaded", len(gdf))
    return gdf


# ──────────────────────────────────────────────────────────────
# spatial clip helper
# ──────────────────────────────────────────────────────────────
def clip_to_germany(gdf: gpd.GeoDataFrame,
                    germany_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Return only those *gdf* rows inside the (dissolved) *germany_gdf*
    polygon.  CRS is harmonised automatically.
    """
    if gdf.empty or germany_gdf.empty:
        return gdf

    if gdf.crs != germany_gdf.crs:
        gdf = gdf.to_crs(germany_gdf.crs)

    boundary = germany_gdf.iloc[0].geometry
    inside = gdf.geometry.apply(boundary.contains)

    return gdf.loc[inside].reset_index(drop=True)
