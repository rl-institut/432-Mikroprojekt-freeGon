#!/usr/bin/env python3
#  plot_line_histograms.py --------------------------------------------------
"""
Histograms for per-km line parameters (r, x, b) of
CORE-TSO (DLR lines) | eGon | PyPSA-Eur.

Creates three PNG files:

    hist_r_per_km.png
    hist_x_per_km.png
    hist_b_per_km.png
"""
from __future__ import annotations
import argparse, logging
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from shapely import wkt

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger()

# ---------------------------------------------------------------------------

DEF_CORE  = Path("data/clipped/dlr-lines-germany.csv")
DEF_EGON  = Path("data/clipped/network-lines-germany.csv")
DEF_PYPSA = Path("data/clipped/pypsa-eur-lines-germany.csv")
BOUNDARY  = Path("data/input/georef-germany-gemeinde@public.geojson")   # not used here

MIN_KV    = 220          # keep only ≥220 kV in eGon
COMMON_CRS = 4326

COLOURS = {"CORE-TSO": "tab:red",
           "eGon"    : "tab:green",
           "PyPSA-Eur": "tab:blue"}

PARAMS = {
    "r_per_km": (r"$r$ per km  [$\Omega$/km]", "hist_r_per_km.png"),
    "x_per_km": (r"$x$ per km  [$\Omega$/km]", "hist_x_per_km.png"),
    "b_per_km": (r"$b$ per km  [$S$/km]"      , "hist_b_per_km.png"),
}

# ---------------------------------------------------------------------------

def _read_csv(path: Path) -> gpd.GeoDataFrame:
    df = pd.read_csv(path)
    if "geometry" not in df.columns:
        raise ValueError(f"{path} has no geometry column!")
    df["geometry"] = df["geometry"].apply(wkt.loads)
    return gpd.GeoDataFrame(df, geometry="geometry", crs=COMMON_CRS)

# ──────────────────────────────────────────────────────────────────
def _auto_length_km(geoms: gpd.GeoSeries) -> pd.Series:
    """Return line length in **kilometres** from the geometry itself."""
    # project once to a metric CRS (Web-Mercator is fine here)
    g_proj = geoms.to_crs(3857) if geoms.crs.is_geographic else geoms
    return g_proj.length / 1_000
# ──────────────────────────────────────────────────────────────────

def _load(name: str, path: Path) -> gpd.GeoDataFrame:
    gdf = _read_csv(path)

    # ----------------------------------------------------------------
    #  ALWAYS recompute length_km from geometry  ⟶  units are correct
    # ----------------------------------------------------------------
    length_km = _auto_length_km(gdf.geometry)

    # build r_per_km, x_per_km, b_per_km if missing ------------------
    for per_km, raw in {"r_per_km": "r",
                        "x_per_km": "x",
                        "b_per_km": "b"}.items():
        if per_km not in gdf.columns:
            gdf[per_km] = gdf.get(raw, np.nan) / length_km

    # eGon voltage filter -------------------------------------------
    if name == "eGon":
        vcol = "v_nom" if "v_nom" in gdf.columns else "v_nom_kv"
        gdf = gdf[gdf[vcol] >= MIN_KV].copy()
        log.info("eGon – kept %d lines ≥%dkV", len(gdf), MIN_KV)

    return gdf


# ---------------------------------------------------------------------------

def histogram_panel(param: str,
                    nice_label: str,
                    outfile: str,
                    datasets: dict[str, gpd.GeoDataFrame]) -> None:
    """Draw a 1×3 histogram figure for *param*."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    fig.suptitle(f"Distribution of {nice_label}", fontsize=16, y=.98)

    # common bin edges (robust: clip extreme 1 %)
    all_vals = pd.concat([ds[param] for ds in datasets.values()]).dropna()
    vmin, vmax = all_vals.quantile([.01, .99])
    bins = np.linspace(vmin, vmax, 41)

    for ax, (name, gdf) in zip(axes, datasets.items()):
        vals = gdf[param].dropna()
        sns.histplot(vals, bins=bins,
                     ax=ax, color=COLOURS[name],
                     edgecolor="black", linewidth=.3)
        ax.set_xlabel(nice_label + f"\n({name})")
        ax.set_ylabel("count" if ax is axes[0] else "")
        ax.tick_params(labelsize=8, length=2)

    fig.tight_layout()
    out = Path("output/charts/comparison") / outfile
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300)
    log.info("✓ saved %s", out)

# ---------------------------------------------------------------------------

def main(core: Path, egon: Path, pypsa: Path):
    data = {
        "CORE-TSO" : _load("CORE-TSO",  core),
        "eGon"     : _load("eGon",      egon),
        "PyPSA-Eur": _load("PyPSA-Eur", pypsa),
    }

    for param, (nice, fname) in PARAMS.items():
        histogram_panel(param, nice, fname, data)

# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Plot per-km parameter histograms")
    p.add_argument("--core",  type=Path, default=DEF_CORE,
                   help="CSV for CORE-TSO (DLR) lines")
    p.add_argument("--egon",  type=Path, default=DEF_EGON,
                   help="CSV for eGon network lines")
    p.add_argument("--pypsa", type=Path, default=DEF_PYPSA,
                   help="CSV for PyPSA-Eur lines")
    args = p.parse_args()

    main(args.core, args.egon, args.pypsa)
