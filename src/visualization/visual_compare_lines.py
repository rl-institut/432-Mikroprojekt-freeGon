# ── src/visualization/visual_compare_lines.py ───────────────────────────
import logging
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import Normalize
from shapely import wkt
# plotting helpers
from mpl_toolkits.axes_grid1.inset_locator import inset_axes   # <-- add this


logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

ROOT            = Path(__file__).resolve().parents[2]
DATA_DIR        = ROOT / "data" / "clipped"
OUT_DIR         = ROOT / "output" / "charts" / "comparison"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FILES = {
    "JAO"   : DATA_DIR / "dlr-lines-germany.csv",
    "eGon"       : DATA_DIR / "network-lines-germany.csv",
    "PyPSA-Eur"  : DATA_DIR / "pypsa-eur-lines-germany.csv",
}
BOUNDARY_GEOJSON = ROOT / "data" / "input" / "georef-germany-gemeinde@public.geojson"

COMMON_CRS = 4326           # lon/lat for plotting
MIN_KV     = 220             # filter for eGon


# --------------------------------------------------------------------- #
def _read_csv(path: Path) -> gpd.GeoDataFrame:
    df = pd.read_csv(path)
    if "geometry" not in df.columns:
        raise ValueError(f"{path.name}: no 'geometry' column with WKT")
    df["geometry"] = df["geometry"].apply(wkt.loads)
    return gpd.GeoDataFrame(df, geometry="geometry", crs=COMMON_CRS)


def _load_dataset(name: str, file: Path) -> gpd.GeoDataFrame:
    log.info("Loading %s (%s)", name, file.name)
    gdf = _read_csv(file)

    # homogenise column names ------------------------------------------------
    if "s_nom" not in gdf.columns:
        # PyPSA csv has s_nom already; eGon may use s_nom_opt
        if "s_nom_opt" in gdf.columns:
            gdf["s_nom"] = gdf["s_nom_opt"]
        else:
            gdf["s_nom"] = 0.0

    if name == "eGon":
        # voltage column may be v_nom or v_nom_kv
        vcol = "v_nom" if "v_nom" in gdf.columns else "v_nom_kv"
        gdf = gdf[gdf[vcol] >= MIN_KV].copy()
        log.info("   filtered to %d lines ≥ %dkV", len(gdf), MIN_KV)

    return gdf.to_crs(COMMON_CRS)


# --------------------------------------------------------------------- #
def _common_extent(boundary: gpd.GeoSeries):
    xmin, ymin, xmax, ymax = boundary.total_bounds
    dx, dy = xmax - xmin, ymax - ymin
    pad_x, pad_y = dx * 0.03, dy * 0.03          # small padding
    return xmin - pad_x, xmax + pad_x, ymin - pad_y, ymax + pad_y


def _plot_single(ax, gdf: gpd.GeoDataFrame, title: str,
                 boundary: gpd.GeoSeries, cmap="turbo"):
    ax.set_title(title, fontsize=12)
    boundary.plot(ax=ax, fc="none", ec="black", lw=.5, zorder=1)

    if gdf.empty:
        ax.set_axis_off();  return

    norm  = Normalize(vmin=gdf["s_nom"].quantile(.05),
                      vmax=gdf["s_nom"].quantile(.95))
    gdf.plot(ax=ax,
             column="s_nom",
             cmap=cmap,
             norm=norm,
             linewidth=1.0,
             zorder=2)

    ax.set_axis_off()


# --------------------------------------------------------------------- #
def main():
    # load datasets ------------------------------------------------------
    data = {name: _load_dataset(name, path) for name, path in FILES.items()}

    # load Germany outline ----------------------------------------------
    boundary = gpd.read_file(BOUNDARY_GEOJSON, engine="pyogrio")
    boundary = boundary.dissolve()[["geometry"]].to_crs(COMMON_CRS)

    xmin, xmax, ymin, ymax = _common_extent(boundary)

    # -------------------------------------------------------------------
    fig, axes = plt.subplots(
        1, 3,
        figsize=(16, 7),
        gridspec_kw={"wspace": 0.02}
    )
    fig.suptitle("JAO | eGon | PyPSA-Eur   –   Line capacity $s_{nom}$ [MW]",
                 fontsize=18, y=0.97)

    for ax, (name, gdf) in zip(axes, data.items()):
        _plot_single(ax, gdf, name, boundary)
        ax.set_xlim(xmin, xmax);  ax.set_ylim(ymin, ymax)

    # colourbar (take the middle axis’ norm as reference) ---------------
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    sm = plt.cm.ScalarMappable(cmap="turbo",
                               norm=Normalize(vmin=0, vmax=10_000))
    cax = inset_axes(axes[-1], width="2%", height="70%", loc="center left",
                     bbox_to_anchor=(1.05, 0, 1, 1), bbox_transform=axes[-1].transAxes,
                     borderpad=0)
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("Capacity $s_{nom}$  [MW]")

    out = OUT_DIR / "jao_egon_pypsa_capacity.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    log.info("✓ saved %s", out)


# ────────────────────────────────────────────────────────────────────────
#  EXTRA:  three more sheets — r, x, b  (per-km)                         #
# ────────────────────────────────────────────────────────────────────────
def _plot_triptych(param: str, nice: str, out_name: str, cmap="plasma"):
    """
    Draw the three maps for a given *per-km* parameter and write PNG.
    *param*     – column name  (r_per_km, x_per_km, b_per_km)
    *nice*      – pretty label for the colour-bar
    *out_name*  – filename stem inside OUT_DIR
    """
    data = {name: _load_dataset(name, path) for name, path in FILES.items()}

    # guarantee column exists & coerce to numeric -----------------------
    for gdf in data.values():
        if param not in gdf.columns:
            gdf[param] = np.nan
        gdf[param] = pd.to_numeric(gdf[param], errors="coerce")

    boundary = gpd.read_file(BOUNDARY_GEOJSON, engine="pyogrio").dissolve().to_crs(COMMON_CRS)
    xmin, xmax, ymin, ymax = _common_extent(boundary)

    fig, axes = plt.subplots(1, 3, figsize=(16, 7), gridspec_kw={"wspace": .02})
    fig.suptitle(f"JAO | eGon | PyPSA-Eur   –  {nice}", fontsize=18, y=.97)

    # individual stretch per panel, but identical colour-bar -----------
    global_min = min(np.nanmin(gdf[param]) for gdf in data.values())
    global_max = max(np.nanmax(gdf[param]) for gdf in data.values())
    norm_global = Normalize(vmin=global_min, vmax=global_max)

    for ax, (name, gdf) in zip(axes, data.items()):
        _plot_single(ax, gdf, name, boundary, cmap=cmap)
        ax.set_xlim(xmin, xmax);  ax.set_ylim(ymin, ymax)

    cax = inset_axes(axes[-1], width="2%", height="70%", loc="center left",
                     bbox_to_anchor=(1.05, 0, 1, 1), bbox_transform=axes[-1].transAxes,
                     borderpad=0)
    sm = plt.cm.ScalarMappable(norm=norm_global, cmap=cmap)
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label(nice)

    outfile = OUT_DIR / f"{out_name}.png"
    fig.savefig(outfile, dpi=300, bbox_inches="tight")
    log.info("✓ saved %s", outfile)


if __name__ == "__main__":
    # first plot (s_nom) ------------------------------------------------
    main()

    # three extra parameter maps ---------------------------------------
    import numpy as np
    _plot_triptych("r_per_km", r"$r$ per km [$\Omega$/km]",    "lines_r_per_km")
    _plot_triptych("x_per_km", r"$x$ per km [$\Omega$/km]",    "lines_x_per_km",
                   cmap="viridis")
    _plot_triptych("b_per_km", r"$b$ per km [$S$/km]",         "lines_b_per_km",
                   cmap="coolwarm")

