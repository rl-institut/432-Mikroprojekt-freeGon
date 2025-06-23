#!/usr/bin/env python3
"""
quick_plot_network_chords_v3.py
--------------------------------
Preview network lines (except 110 kV) + straight-line chords.

• works with Shapely 1.x  *and*  2.x
"""

import pathlib, sys, logging, webbrowser
import pandas as pd
import geopandas as gpd
from shapely import wkt
from shapely.geometry import LineString, MultiLineString
from shapely.ops import linemerge
import folium

# ───────────────────────────── CONFIG ──────────────────────────────
DATA_CSV   = pathlib.Path("data/input/network-lines.csv")     # your file
GEOM_FIELD = "geom"                                           # WKT column
VOLT_FIELD = "v_nom"                                          # voltage column
CRS_WGS84  = "EPSG:4326"                                      # Leaflet CRS
OUT_HTML   = pathlib.Path("network_chords_preview.html")
# ───────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

if not DATA_CSV.exists():
    sys.exit(f"❌  {DATA_CSV} not found – fix DATA_CSV in the script.")

# ------------------------------------------------------------------
# 1. Read CSV → GeoDataFrame  (exclude 110 kV straight away)
# ------------------------------------------------------------------
logging.info("Reading CSV %s …", DATA_CSV)
df_raw = pd.read_csv(DATA_CSV)

for col in (GEOM_FIELD, VOLT_FIELD):
    if col not in df_raw.columns:
        sys.exit(f"❌  Column “{col}” not found – adjust script.")

df_raw = df_raw[df_raw[VOLT_FIELD] != 110]          # ⚡ drop 110 kV
logging.info("Keeping %d line(s) (voltage ≠ 110 kV).", len(df_raw))

geoms = df_raw[GEOM_FIELD].apply(wkt.loads)
gdf   = gpd.GeoDataFrame(df_raw.drop(columns=[GEOM_FIELD]),
                         geometry=geoms, crs=CRS_WGS84)

if gdf.empty:
    sys.exit("❌  No geometries left after filtering.")

# ------------------------------------------------------------------
# 2. Build chords (first ↔ last coord of merged geometry)
# ------------------------------------------------------------------
def first_last_coord(geom):
    """
    Return (first, last) coordinate tuple of the whole route.

    • For a single LineString → use it directly.
    • For MultiLineString   → linemerge, then take the longest piece.
    """
    if geom is None or geom.is_empty:
        return None

    # ─── 1. get one continuous LineString ─────────────────────────
    if isinstance(geom, LineString):
        merged = geom                                    # already simple
    else:                                                # MultiLineString
        merged = linemerge(geom)                         # join parts
        if isinstance(merged, MultiLineString):          # still multiple?
            merged = max(merged.geoms, key=lambda ls: ls.length)

    # ─── 2. extract first / last coord ────────────────────────────
    coords = list(merged.coords)
    if len(coords) >= 2 and coords[0] != coords[-1]:
        return coords[0], coords[-1]                     # (x0,y0), (xN,yN)
    return None


chord_geoms = []
for g in gdf.geometry:
    ends = first_last_coord(g)
    if ends:
        chord_geoms.append(LineString(ends))

gdf_chords = gpd.GeoDataFrame(geometry=chord_geoms, crs=CRS_WGS84)
logging.info("Created %d chord(s).", len(gdf_chords))

# ------------------------------------------------------------------
# 3. Folium preview
# ------------------------------------------------------------------
centre = gdf.unary_union.centroid
m = folium.Map(location=[centre.y, centre.x], zoom_start=6,
               tiles="OpenStreetMap", control_scale=True)

# original lines (red)
fg_net = folium.FeatureGroup("Network lines (original)", show=True)
for geom in gdf.geometry:
    for seg in (geom.geoms if isinstance(geom, MultiLineString) else [geom]):
        folium.PolyLine([(y, x) for x, y in seg.coords],
                        color="red", weight=2, opacity=0.9).add_to(fg_net)
fg_net.add_to(m)

# chord lines (dark-grey, thicker)
fg_ch = folium.FeatureGroup("Chord (straight) lines", show=True)
for chord in gdf_chords.geometry:
    folium.PolyLine([(y, x) for x, y in chord.coords],
                    color="#222", weight=6, opacity=0.9).add_to(fg_ch)
fg_ch.add_to(m)

folium.LayerControl(collapsed=False).add_to(m)

# ------------------------------------------------------------------
# 4. Save & open
# ------------------------------------------------------------------
m.save(OUT_HTML)
logging.info("✔  Preview written to %s", OUT_HTML)
webbrowser.open_new_tab('file://' + str(OUT_HTML.resolve()))
