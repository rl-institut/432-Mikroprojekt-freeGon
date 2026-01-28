#!/usr/bin/env python3
"""
Script to process PyPSA and JAO line data, selecting lines within or crossing Germany's borders.
"""

import os
import geopandas as gpd
import pandas as pd
from pathlib import Path
import requests
from typing import Optional
from shapely.geometry import LineString
from shapely import wkt

# Define paths
BASE_DIR = Path("/home/mohsen/PycharmProjects/freeGon/grid-matcher-original/432-Mikroprojekt-freeGon")
RAW_DATA_DIR = BASE_DIR / "grid_matcher/data/raw"
OUTPUT_DIR = BASE_DIR / "grid_matcher/data"

# Ensure output directory exists
OUTPUT_DIR.mkdir(exist_ok=True)

import csv

CSV_WRITE_KW = dict(
    index=False,
    sep=",",
    quoting=csv.QUOTE_ALL,   # <-- force quotes on EVERY field (safest for WKT)
    quotechar='"',
    escapechar="\\",
    lineterminator="\n",
    # Optional: keep small numbers out of scientific notation
    # float_format="%.12f",
)


# ---- Exact Germany boundary (municipal dissolve) ----
GER_DATA_DIR = BASE_DIR / "grid_matcher/data"
GER_BOUNDARY_FILE = GER_DATA_DIR / "georef-germany-gemeinde@public.geojson"  # already on disk

from shapely.ops import unary_union
from shapely.prepared import prep

def load_germany_boundary_exact() -> gpd.GeoSeries:
    """Read municipalities, dissolve to one polygon, return GeoSeries in EPSG:4326."""
    gdf = gpd.read_file(GER_BOUNDARY_FILE)
    if gdf.crs is None or gdf.crs.to_string().upper() != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")
    # dissolve to a single (Multi)Polygon; buffer(0) cleans minor slivers
    poly = unary_union(gdf.geometry).buffer(0)
    return gpd.GeoSeries([poly], crs="EPSG:4326")

def select_lines_intersecting_germany(
    gdf: gpd.GeoDataFrame,
    germany: Optional[gpd.GeoSeries] = None,
    buffer_km: float = 0.0,
) -> gpd.GeoDataFrame:
    """
    Keep lines whose geometry INTERSECTS Germany (full lines kept; no clipping).
    """
    if germany is None:
        germany = load_germany_boundary_exact()

    # CRS align
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    elif gdf.crs.to_string() != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")

    geom = germany.unary_union
    if buffer_km and buffer_km > 0:
        # crude deg ≈ km/111
        geom = geom.buffer(buffer_km / 111.0)

    P = prep(geom)   # speed
    mask = gdf.geometry.apply(lambda x: bool(x) and not x.is_empty and P.intersects(x))
    return gdf.loc[mask].copy()



def parse_geometry_column(df):
    """
    Improved function to parse geometry column with better error handling and debugging.
    """
    print(f"Columns in dataframe: {df.columns.tolist()}")

    if "geometry" in df.columns:
        print(f"Geometry column exists with {df['geometry'].notna().sum()} non-null values")
        print(f"Sample geometry values: {df['geometry'].head(3).tolist()}")

        # Clean up geometry strings first
        df["geometry"] = df["geometry"].apply(
            lambda x: x.replace("'", "").strip() if isinstance(x, str) else x
        )

        # Try different parsing approaches
        # 1. Try direct WKT parsing
        try:
            valid_mask = df["geometry"].str.contains("LINESTRING", na=False)
            print(f"Found {valid_mask.sum()} rows with LINESTRING text")

            if valid_mask.sum() > 0:
                geometries = []
                valid_indices = []

                for idx, geom_str in df.loc[valid_mask, "geometry"].items():
                    try:
                        # Try to parse with shapely's WKT loader
                        geom = wkt.loads(geom_str)
                        geometries.append(geom)
                        valid_indices.append(idx)
                    except Exception as e1:
                        # If that fails, try manual parsing
                        try:
                            # Extract coordinates
                            start = geom_str.find("(")
                            end = geom_str.rfind(")")
                            if start > 0 and end > start:
                                coords_text = geom_str[start + 1:end]

                                # Try different coordinate formats
                                points = []
                                if "," in coords_text:  # comma-separated format
                                    for pair in coords_text.split(","):
                                        x, y = map(float, pair.strip().split())
                                        points.append((x, y))
                                else:  # space-separated format
                                    coords = coords_text.split()
                                    for i in range(0, len(coords), 2):
                                        if i + 1 < len(coords):
                                            x, y = float(coords[i]), float(coords[i + 1])
                                            points.append((x, y))

                                if len(points) >= 2:
                                    geom = LineString(points)
                                    geometries.append(geom)
                                    valid_indices.append(idx)
                        except Exception as e2:
                            print(f"Failed to parse geometry: {geom_str[:50]}... Error: {e2}")
                            continue

                if geometries:
                    print(f"Successfully parsed {len(geometries)} geometries")
                    return gpd.GeoDataFrame(df.loc[valid_indices], geometry=geometries, crs="EPSG:4326")
                else:
                    print("No valid geometries were created from LINESTRING strings")
            else:
                print("No LINESTRING values found in geometry column")
        except Exception as e:
            print(f"Error in WKT parsing approach: {e}")

    # Try to create from x/y coordinates
    coord_cols = [
        ("x0", "y0", "x1", "y1"),  # Standard naming
        ("lon0", "lat0", "lon1", "lat1"),  # Alternative naming
        ("longitude0", "latitude0", "longitude1", "latitude1")  # Full names
    ]

    for cols in coord_cols:
        if all(col in df.columns for col in cols):
            x0_col, y0_col, x1_col, y1_col = cols
            try:
                print(f"Attempting to create geometries from coordinate columns: {cols}")
                geometries = []
                valid_indices = []

                for idx, row in df.iterrows():
                    try:
                        if pd.notnull(row[x0_col]) and pd.notnull(row[y0_col]) and pd.notnull(
                                row[x1_col]) and pd.notnull(row[y1_col]):
                            x0, y0, x1, y1 = float(row[x0_col]), float(row[y0_col]), float(row[x1_col]), float(
                                row[y1_col])
                            geom = LineString([(x0, y0), (x1, y1)])
                            geometries.append(geom)
                            valid_indices.append(idx)
                    except Exception as e:
                        print(f"Error processing row {idx}: {e}")

                if geometries:
                    print(f"Created {len(geometries)} geometries from coordinate columns")
                    return gpd.GeoDataFrame(df.loc[valid_indices], geometry=geometries, crs="EPSG:4326")
            except Exception as e:
                print(f"Failed to create geometries from {cols}: {e}")

    # Try the zenodo fallback
    try:
        try:
            from grid_matcher.io.zenodo_fetch import download_prepare_pypsa_lines_from_zenodo
            print("Attempting zenodo_fetch fallback...")
            output = download_prepare_pypsa_lines_from_zenodo(
                output_csv=RAW_DATA_DIR / "pypsa_fixed_lines.csv",
                verbose=True
            )
            return gpd.read_file(output)
        except ImportError:
            print("grid_matcher module not found, skipping zenodo_fetch fallback")
    except Exception as e:
        print(f"Failed to use zenodo_fetch fallback: {e}")

    raise ValueError("Could not find or parse valid geometry information in the CSV file. "
                     "Please check the format of your input data.")


def fix_pypsa_geometries(df):
    """Properly parse PyPSA geometries with quotes."""
    if 'geometry' not in df.columns:
        return None

    # Print raw content to better understand the data format
    print("Full raw geometry examples (first 3):")
    for i, geo in enumerate(df['geometry'].dropna().head(3)):
        print(f"  {i}: Full value: {repr(geo)}")  # Use repr to see exact string representation

    # Better handling of single quoted LINESTRING data
    def extract_linestring(value):
        if pd.isnull(value):
            return None

        value_str = str(value)
        # Look for LINESTRING pattern
        if 'LINESTRING' in value_str:
            # Already contains LINESTRING, just clean quotes if needed
            if value_str.startswith("'") and value_str.endswith("'"):
                return value_str[1:-1]  # Remove outer quotes
            return value_str

        # If LINESTRING is missing but we have coordinate pairs
        if '(' in value_str and ')' in value_str and ',' in value_str:
            # Try to extract coordinates and construct a proper LINESTRING
            coords_start = value_str.find('(')
            coords_end = value_str.rfind(')')
            if coords_start >= 0 and coords_end > coords_start:
                coords = value_str[coords_start:coords_end + 1]
                return f"LINESTRING {coords}"

        return None

    df['clean_geom'] = df['geometry'].apply(extract_linestring)

    # Show what we extracted
    print("Extracted geometries (first 3):")
    for i, geo in enumerate(df['clean_geom'].dropna().head(3)):
        print(f"  {i}: {geo[:50]}...")

    valid_count = df['clean_geom'].notna().sum()
    print(f"Found {valid_count} valid geometry strings")

    if valid_count == 0:
        return None

    # Create GeoDataFrame from valid geometries
    try:
        valid_rows = df['clean_geom'].notna()
        gdf = df[valid_rows].copy()
        gdf['geometry'] = gdf['clean_geom'].apply(wkt.loads)
        gdf = gdf.drop(columns=['clean_geom'])
        return gpd.GeoDataFrame(gdf, geometry='geometry', crs="EPSG:4326")
    except Exception as e:
        print(f"Error creating GeoDataFrame: {e}")
        if valid_count > 0:
            print(f"Sample value causing error: {df.loc[df['clean_geom'].notna()].iloc[0]['clean_geom'][:100]}")
        return None


import re
import numpy as np

def _coerce_voltage_smart(series: pd.Series) -> pd.Series:
    """
    Extract the *maximum* numeric voltage from free-text v_nom strings.
    Handles patterns like '380/220 kV', '225,0', '400 kV – 220 kV', etc.
    Returns a float (kV) or NaN if nothing numeric is found.
    """
    s = series.astype(str).fillna("").str.strip()

    # Normalize decimal commas to dots (e.g., 225,0 -> 225.0)
    s = s.str.replace(",", ".", regex=False)

    # Remove trailing unit tokens like 'kV' (various capitalizations and spacing)
    s = s.str.replace(r"\s*[kK]\s*[vV]\b", "", regex=True)

    # Pull out *all* numbers (ints/floats) and take the maximum
    def pick_max(text: str):
        nums = re.findall(r"\d+(?:\.\d+)?", text)
        if not nums:
            return np.nan
        vals = [float(n) for n in nums]
        # Sometimes raw volts appear (e.g., 220000). If a value is >= 1000 and divisible by 1000,
        # treat it as volts and convert to kV.
        fixed = []
        for v in vals:
            if v >= 1000 and abs((v / 1000) - round(v / 1000)) < 1e-6:
                fixed.append(v / 1000.0)
            else:
                fixed.append(v)
        return max(fixed)

    return s.map(pick_max)

def split_pypsa_voltage_from_saved(base_csv: Path, threshold: float = 220.0, write_debug: bool = True):
    """
    Read the saved pypsa_clipped_lines.csv and write:
      - pypsa_clipped_LV_lines.csv     (v_nom <  threshold)
      - pypsa_clipped_MVHV_lines.csv   (v_nom >= threshold)
    Also writes a debug file for rows where voltage couldn't be parsed.
    """
    # Read exactly what we saved
    df = pd.read_csv(base_csv, dtype=str)

    # Ensure we have a v_nom-like column
    if "v_nom" not in df.columns and "voltage" in df.columns:
        df["v_nom"] = df["voltage"]
    elif "v_nom" not in df.columns:
        raise ValueError("No 'v_nom' (or 'voltage') column in saved CSV.")

    # Robust, multi-number aware parsing
    v_num = _coerce_voltage_smart(df["v_nom"])
    df["__v_num__"] = v_num  # keep this only for debugging

    # Split
    lv_df   = df[v_num.lt(threshold).fillna(False)].copy()
    mvhv_df = df[v_num.ge(threshold).fillna(False)].copy()

    # Save with safe quoting
    lv_path   = base_csv.with_name("pypsa_clipped_LV_lines.csv")
    mvhv_path = base_csv.with_name("pypsa_clipped_MVHV_lines.csv")
    lv_df.to_csv(lv_path, **CSV_WRITE_KW)
    mvhv_df.to_csv(mvhv_path, **CSV_WRITE_KW)

    # Optional: rows we couldn't parse at all
    if write_debug:
        excl = df[v_num.isna()].copy()
        if not excl.empty:
            excl_path = base_csv.with_name("pypsa_clipped_excluded_no_vnom.csv")
            # Keep just the helpful columns for inspection
            keep_cols = [c for c in ["id","name","bus0","bus1","v_nom","voltage","length","type","tags","geometry","__v_num__"] if c in excl.columns]
            excl[keep_cols].to_csv(excl_path, **CSV_WRITE_KW)
            print(f"[PYPSA] Excluded (unparseable v_nom): {len(excl)} → {excl_path}")

        # quick bin counts to verify distribution
        try:
            bins = pd.cut(v_num, bins=[0,110,150,220,300,400,800], include_lowest=True)
            bin_counts = bins.value_counts(dropna=False).sort_index()
            bin_counts.to_csv(base_csv.with_name("pypsa_clipped_voltage_bins_debug.csv"), **CSV_WRITE_KW)
        except Exception:
            pass

    # Sanity: total length vs. parts (length is meters)
    try:
        base = pd.read_csv(base_csv)
        base_len_km = pd.to_numeric(base.get("length"), errors="coerce").div(1000).sum()
        lv_len_km   = pd.to_numeric(lv_df.get("length"),   errors="coerce").div(1000).sum()
        mvhv_len_km = pd.to_numeric(mvhv_df.get("length"), errors="coerce").div(1000).sum()
        print(f"[PYPSA] Length km → total:{base_len_km:,.1f} | LV:{lv_len_km:,.1f} | MVHV:{mvhv_len_km:,.1f} | LV+MVHV:{(lv_len_km+mvhv_len_km):,.1f}")
    except Exception:
        pass

    # Drop debug helper col from memory copies (not strictly needed since we saved already)
    for _df in (lv_df, mvhv_df):
        if "__v_num__" in _df.columns:
            _df.drop(columns=["__v_num__"], inplace=True)

    return lv_path, mvhv_path



def process_pypsa_lines(buffer_km: float = 0.0, verbose: bool = True):
    """
    Read PyPSA lines, keep full geometries that intersect Germany, and
    write a standardized CSV (id first, v_nom, per-km columns) + GeoPackage.
    PyPSA 'length' is in METERS → convert to km for per-km fields.
    """
    import geopandas as gpd
    import pandas as pd
    from shapely import wkt

    input_file = RAW_DATA_DIR / "pypsa_raw_lines.csv"
    output_file = OUTPUT_DIR / "pypsa_clipped_lines.csv"

    if verbose:
        print(f"Reading PyPSA data from {input_file}")
    if not input_file.exists():
        raise FileNotFoundError(input_file)

    # robust read (geometry often single-quoted)
    try:
        df = pd.read_csv(input_file, sep=",", engine="python", dtype=str,
                         on_bad_lines="skip", quotechar="'", escapechar="\\")
    except Exception:
        df = pd.read_csv(input_file, sep=",", engine="python", dtype=str,
                         on_bad_lines="skip", quotechar='"', escapechar="\\")

    if verbose:
        print(f"DataFrame shape after reading: {df.shape}")
        print(f"Columns: {list(df.columns)}")
    if "geometry" not in df.columns:
        raise ValueError("No 'geometry' column in PyPSA CSV.")

    # parse WKT → geometry
    def _parse_wkt(s: str):
        if pd.isna(s):
            return None
        s = str(s).strip()
        if not s:
            return None
        if (len(s) >= 2) and (s[0] == s[-1]) and s[0] in ("'", '"'):
            s = s[1:-1].strip()
        if not s.upper().startswith("LINESTRING"):
            return None
        try:
            return wkt.loads(s)
        except Exception:
            return None

    geoms = df["geometry"].apply(_parse_wkt)
    gdf = gpd.GeoDataFrame(df[geoms.notna()].copy(), geometry=geoms[geoms.notna()], crs="EPSG:4326")
    if verbose:
        print(f"Parsed {len(gdf)} valid LINESTRING geometries out of {len(df)}")

    # keep in/crossing Germany (full geometry retained)
    # keep lines that intersect Germany (full geometry retained)
    germany = load_germany_boundary_exact()
    gdf_de = select_lines_intersecting_germany(gdf, germany, buffer_km=buffer_km)
    if verbose:
        print(f"[PYPSA] Rows after DE + cross-border selection: {len(gdf_de)}")

    # ---- flat DataFrame + WKT ----
    from shapely.geometry.base import BaseGeometry
    df_out = gdf_de.drop(columns="geometry").copy()
    df_out["geometry"] = gdf_de.geometry.apply(lambda g: g.wkt if isinstance(g, BaseGeometry) else None)
    df_out = df_out[df_out["geometry"].notna()]

    # rename voltage → v_nom
    if "v_nom" not in df_out.columns and "voltage" in df_out.columns:
        df_out.rename(columns={"voltage": "v_nom"}, inplace=True)

    # ids
    if "id" not in df_out.columns:
        if "line_id" in df_out.columns:
            df_out["id"] = df_out["line_id"].astype(str)
        elif "name" in df_out.columns:
            df_out["id"] = df_out["name"].astype(str)
        else:
            df_out["id"] = [f"pypsa_line_{i}" for i in range(len(df_out))]
    if "line_id" in df_out.columns:
        df_out.drop(columns=["line_id"], inplace=True)

    # numeric coercions
    for c in ["r", "x", "b", "length", "v_nom", "i_nom", "circuits", "s_nom"]:
        if c in df_out.columns:
            df_out[c] = pd.to_numeric(df_out[c], errors="coerce")

    # per-km (length is meters)
    L_km = df_out["length"] / 1000.0

    def _safe_div(n, d):
        try:
            return float(n) / float(d) if pd.notna(n) and pd.notna(d) and float(d) != 0 else None
        except Exception:
            return None

    df_out["r_per_km"] = [_safe_div(r, L) for r, L in zip(df_out.get("r"), L_km)]
    df_out["x_per_km"] = [_safe_div(x, L) for x, L in zip(df_out.get("x"), L_km)]
    df_out["b_per_km"] = [_safe_div(b, L) for b, L in zip(df_out.get("b"), L_km)]

    # final schema
    schema = [
        "id", "bus0", "bus1", "v_nom", "i_nom", "circuits", "s_nom",
        "r", "x", "b", "length", "underground", "under_construction",
        "type", "tags", "geometry", "r_per_km", "x_per_km", "b_per_km",
    ]
    for col in schema:
        if col not in df_out.columns: df_out[col] = None
    df_out = df_out[schema]

    # write once, with safe quoting
    df_out.to_csv(output_file, **CSV_WRITE_KW)

    # GeoPackage for mapping
    gdf_de.to_file(output_file.with_suffix(".gpkg"), driver="GPKG")
    if verbose:
        print(f"Saved PyPSA CSV → {output_file} and GPKG → {output_file.with_suffix('.gpkg')}")

    # ---- build flat DataFrame and WKT geometry ----
    df_out = pd.DataFrame(gdf_de.drop(columns="geometry"))
    df_out["geometry"] = gdf_de.geometry.apply(lambda g: g.wkt if g is not None else None)

    # rename voltage → v_nom
    if "v_nom" not in df_out.columns and "voltage" in df_out.columns:
        df_out.rename(columns={"voltage": "v_nom"}, inplace=True)

    # ids (prefer original line_id for id)
    if "id" not in df_out.columns:
        if "line_id" in df_out.columns:
            df_out["id"] = df_out["line_id"].astype(str)
        elif "name" in df_out.columns:
            df_out["id"] = df_out["name"].astype(str)
        else:
            df_out["id"] = [f"pypsa_line_{i}" for i in range(len(df_out))]
    if "line_id" in df_out.columns:
        df_out.drop(columns=["line_id"], inplace=True)

    # numeric coercions
    for c in ["r", "x", "b", "length", "v_nom", "i_nom", "circuits", "s_nom"]:
        if c in df_out.columns:
            df_out[c] = pd.to_numeric(df_out[c], errors="coerce")

    # per-km using length in METERS → km
    L_km = df_out["length"] / 1000.0
    def _safe_div(n, d):
        try:
            return float(n) / float(d) if pd.notna(n) and pd.notna(d) and float(d) != 0 else None
        except Exception:
            return None
    df_out["r_per_km"] = [_safe_div(r, L) for r, L in zip(df_out.get("r"), L_km)]
    df_out["x_per_km"] = [_safe_div(x, L) for x, L in zip(df_out.get("x"), L_km)]
    df_out["b_per_km"] = [_safe_div(b, L) for b, L in zip(df_out.get("b"), L_km)]

    # final column order
    schema = [
        "id", "bus0", "bus1", "v_nom", "i_nom", "circuits", "s_nom",
        "r", "x", "b", "length", "underground", "under_construction",
        "type", "tags", "geometry", "r_per_km", "x_per_km", "b_per_km",
    ]
    for col in schema:
        if col not in df_out.columns:
            df_out[col] = None
    df_out = df_out[schema]

    if verbose:
        print(f"Saving processed PyPSA data with {len(df_out.columns)} columns to {output_file}")

    # quote everything so WKT survives
    # Save the main CSV once
    df_out["geometry"] = df_out["geometry"].astype(str)
    df_out.to_csv(output_file, **CSV_WRITE_KW)

    # GPKG for mapping
    gdf_de.to_file(output_file.with_suffix(".gpkg"), driver="GPKG")
    if verbose:
        print(f"Saved PyPSA CSV → {output_file} and GPKG → {output_file.with_suffix('.gpkg')}")

    # NOW split from the saved file
    split_pypsa_voltage_from_saved(output_file, threshold=220.0)

    return output_file

def _is_german_tso(tso_str):
    """Case-insensitive check for German TSOs in a free-text TSO field."""
    if not isinstance(tso_str, str):
        return False
    s = tso_str.lower()
    german = ("50hertz", "amprion", "tennet", "transnetbw")
    return any(t in s for t in german)

def _ensure_id_from_name(df: pd.DataFrame, prefix="jao_line_") -> pd.DataFrame:
    """
    Ensure an 'id' column exists and is non-empty.
    Prefer 'name', then 'NE_name'; otherwise generate sequential IDs.
    """
    need_id = ("id" not in df.columns) or df["id"].isna().all() or (df["id"].astype(str).str.strip() == "").all()
    if need_id:
        if "name" in df.columns and df["name"].notna().any():
            df["id"] = df["name"].astype(str)
        elif "NE_name" in df.columns and df["NE_name"].notna().any():
            df["id"] = df["NE_name"].astype(str)
        else:
            df["id"] = [f"{prefix}{i}" for i in range(len(df))]
    # make sure it's plain strings (no NaN)
    df["id"] = df["id"].astype(str).fillna("").str.strip()
    return df


def process_jao_lines(buffer_km: float = 0.0, verbose: bool = True):
    """
    Read JAO lines, keep full geometries intersecting Germany, include rows
    where any TSO field is a German TSO, and write standardized CSV + GPKG.
    JAO 'length' is in KILOMETERS → use directly for per-km fields.
    """
    import geopandas as gpd
    import pandas as pd

    input_file = RAW_DATA_DIR / "jao_raw_lines.csv"
    output_file = OUTPUT_DIR / "jao_clipped_lines.csv"

    if verbose:
        print(f"Reading JAO data from {input_file}")
    if not input_file.exists():
        raise FileNotFoundError(input_file)

    df = pd.read_csv(input_file)
    if verbose:
        print(f"Read JAO data, shape: {df.shape}, columns: {df.columns.tolist()}")

    gdf = parse_geometry_column(df)
    if verbose:
        print(f"[JAO] Loaded {len(gdf)} rows")

    germany = load_germany_boundary_exact()
    gdf_de = select_lines_intersecting_germany(gdf, germany, buffer_km=buffer_km)
    if verbose:
        print(f"[JAO] {len(gdf)} total rows → {len(gdf_de)} after Germany + cross-border selection")

    # optional TSO filter exactly as before…
    if any(c in gdf_de.columns for c in ["TSO","CORE-TSO_bus0","CORE-TSO_bus1"]):
        mask = False
        if "TSO" in gdf_de.columns:            mask = gdf_de["TSO"].apply(_is_german_tso)
        if "CORE-TSO_bus0" in gdf_de.columns:  mask = mask | gdf_de["CORE-TSO_bus0"].apply(_is_german_tso)
        if "CORE-TSO_bus1" in gdf_de.columns:  mask = mask | gdf_de["CORE-TSO_bus1"].apply(_is_german_tso)
        gdf_de = gdf_de[mask].copy()
        if verbose: print(f"[JAO] {len(gdf_de)} rows after German TSO filtering")

    # ---- ensure 'id' exists (from 'name' or 'NE_name') BEFORE we drop columns ----
    gdf_de = _ensure_id_from_name(gdf_de, prefix="jao_line_")

    # flat DF + WKT
    from shapely.geometry.base import BaseGeometry
    df_out = gdf_de.drop(columns="geometry").copy()
    df_out["geometry"] = gdf_de.geometry.apply(lambda g: g.wkt if isinstance(g, BaseGeometry) else None)
    df_out = df_out[df_out["geometry"].notna()]

    # rename Imax
    if "Imax" not in df_out.columns and "Maximum Current Imax (A) Period 1" in df_out.columns:
        df_out.rename(columns={"Maximum Current Imax (A) Period 1": "Imax"}, inplace=True)

    # numeric + per-km (length already km)
    for c in ["r","x","b","length","v_nom","Imax","s_nom"]:
        if c in df_out.columns:
            df_out[c] = pd.to_numeric(df_out[c], errors="coerce")
    def _safe_div(n,d):
        try:
            return float(n)/float(d) if pd.notna(n) and pd.notna(d) and float(d)!=0 else None
        except Exception:
            return None
    df_out["r_per_km"] = [_safe_div(r,L) for r,L in zip(df_out.get("r"), df_out.get("length"))]
    df_out["x_per_km"] = [_safe_div(x,L) for x,L in zip(df_out.get("x"), df_out.get("length"))]
    df_out["b_per_km"] = [_safe_div(b,L) for b,L in zip(df_out.get("b"), df_out.get("length"))]

    # final schema (yours)
    schema = [
        "id","NE_name","r","x","b","length","Comment","geometry",
        "bus0","bus1","EIC_Code","TSO","CORE-TSO_bus0","CORE-TSO_bus1","v_nom",
        "Imax","Maximum Current Imax (A) Period 2","Maximum Current Imax (A) Period 3",
        "Maximum Current Imax (A) Period 4","Maximum Current Imax (A) Period 5",
        "Maximum Current Imax (A) Period 6","Maximum Current Imax (A) Fixed",
        "Dynamic line rating (DLR) DLRmin(A)","Dynamic line rating (DLR) DLRmax(A)",
        "s_nom","tie_line","Unnamed: 26","r_per_km","x_per_km","b_per_km",
    ]
    for col in schema:
        if col not in df_out.columns: df_out[col] = None
    df_out = df_out[schema]

    df_out.to_csv(output_file, **CSV_WRITE_KW)
    gdf_de.to_file(output_file.with_suffix(".gpkg"), driver="GPKG")
    if verbose:
        print(f"Saved JAO CSV → {output_file} and GPKG → {output_file.with_suffix('.gpkg')}")




    # quote everything so WKT survives
    df_out["geometry"] = df_out["geometry"].astype(str)
    df_out.to_csv(output_file, **CSV_WRITE_KW)

    # GeoPackage
    gpkg_output = output_file.with_suffix('.gpkg')
    gdf_de.to_file(gpkg_output, driver="GPKG")
    if verbose:
        print(f"Also saved as GeoPackage: {gpkg_output}")

    return output_file


# --- Simple HTML map for Germany + PyPSA/JAO lines --------------------------------
def _load_lines_any(path: Path) -> gpd.GeoDataFrame:
    """Load lines from GPKG/GeoJSON/shape; fallback to CSV (WKT in 'geometry')."""
    path = Path(path)
    if path.suffix.lower() in {".gpkg", ".geojson", ".json", ".shp"}:
        gdf = gpd.read_file(path)
    else:
        # CSV fallback with WKT
        df = pd.read_csv(path, dtype=str)
        if "geometry" not in df.columns:
            raise ValueError(f"No 'geometry' column in {path}")
        geom = df["geometry"].apply(lambda s: wkt.loads(str(s)) if pd.notna(s) else None)
        gdf = gpd.GeoDataFrame(df, geometry=geom, crs="EPSG:4326")
    if gdf.crs is None:
        gdf.set_crs("EPSG:4326", inplace=True)
    elif gdf.crs.to_string() != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")
    return gdf


def plot_lines_map(
    pypsa_path: Optional[Path] = None,
    jao_path: Optional[Path] = None,
    germany_gdf: Optional[gpd.GeoDataFrame] = None,
    out_html: Path = OUTPUT_DIR / "grid_lines_map.html",
    simplify_tolerance: float = 0.0,  # degrees; e.g. 0.0005 to shrink file size
) -> Path:
    """
    Build a simple Leaflet (Folium) HTML map with toggleable layers:
    - Germany border
    - PyPSA lines
    - JAO lines
    """
    import folium

    # Germany border
    if germany_gdf is None:
        # load the same accurate boundary you used for clipping
        germany_gdf = load_germany_boundary_exact().to_frame(name="geometry")

    if germany_gdf.crs is None or germany_gdf.crs.to_string() != "EPSG:4326":
        germany_gdf = germany_gdf.to_crs("EPSG:4326")
    if simplify_tolerance > 0:
        germany_gdf = germany_gdf.copy()
        germany_gdf["geometry"] = germany_gdf.geometry.simplify(simplify_tolerance, preserve_topology=True)

    # Optional line layers
    gdf_pypsa = _load_lines_any(pypsa_path) if pypsa_path else None
    gdf_jao   = _load_lines_any(jao_path)   if jao_path else None

    # Simplify if requested (can dramatically shrink HTML for many lines)
    for _gdf in (gdf_pypsa, gdf_jao):
        if _gdf is not None and simplify_tolerance > 0:
            _gdf["geometry"] = _gdf.geometry.simplify(simplify_tolerance, preserve_topology=True)

    # Map center/zoom from Germany bounds
    minx, miny, maxx, maxy = germany_gdf.total_bounds
    center = [(miny + maxy) / 2.0, (minx + maxx) / 2.0]

    m = folium.Map(location=center, zoom_start=6, control_scale=True, tiles="cartodbpositron")

    # Germany border layer
    folium.GeoJson(
        germany_gdf.to_json(),
        name="Germany border",
        style_function=lambda _f: {"color": "#111111", "weight": 2, "fill": False},
        highlight_function=lambda _f: {"weight": 3},
        tooltip="Germany",
    ).add_to(m)

    # Helper to add a line layer
    def _add_line_layer(gdf: gpd.GeoDataFrame, name: str, color: str, popup_fields=None):
        if gdf is None or gdf.empty:
            return
        # Choose a few popup fields if present
        if popup_fields is None:
            candidates = [c for c in ["id", "NE_name", "bus0", "bus1", "v_nom", "length", "TSO", "s_nom"] if c in gdf.columns]
            popup_fields = candidates[:6]

        gj = folium.GeoJson(
            gdf.to_json(),
            name=name,
            style_function=lambda _f, col=color: {"color": col, "weight": 2, "opacity": 0.8},
            highlight_function=lambda _f: {"weight": 4, "opacity": 1.0},
            tooltip=folium.GeoJsonTooltip(fields=popup_fields, aliases=[f"{f}:" for f in popup_fields]),
        )
        gj.add_to(m)

    _add_line_layer(gdf_pypsa, "PyPSA lines", "#1f77b4")
    _add_line_layer(gdf_jao,   "JAO lines",   "#d62728")

    folium.LayerControl(collapsed=False).add_to(m)

    # Fit bounds to Germany (with a little padding)
    m.fit_bounds([[miny, minx], [maxy, maxx]])

    # Save
    out_html = Path(out_html)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out_html))
    return out_html

import re, numpy as np

# From your notes
TSO_GERMANY_TOTAL_CIRCUIT_KM = 36002
TSO_BREAKDOWN = [
    {"TSO":"TenneT TSO", "total_km":12333, "kv220":None, "kv380":None},
    {"TSO":"Amprion",    "total_km":10151, "kv220":None, "kv380":None},
    {"TSO":"TransnetBW", "total_km": 2985, "kv220":None, "kv380":None},
    {"TSO":"50Hertz",    "total_km":10533, "kv220":2638, "kv380":7895},
]

import re, numpy as np

# From your notes
TSO_GERMANY_TOTAL_CIRCUIT_KM = 36002

def build_mvhv_vs_tso_report(
    mvhv_csv: Path = OUTPUT_DIR / "pypsa_clipped_MVHV_lines.csv",
    out_html: Path = OUTPUT_DIR / "pypsa_mvhv_vs_tso.html",
) -> Optional[Path]:
    """Compact HTML report: MV/HV circuit-km by voltage (PyPSA),
    % difference vs TSO national total, with tidy small charts."""
    if not mvhv_csv.exists():
        print(f"[REPORT] Missing {mvhv_csv}; run the PyPSA processing/split first.")
        return None

    df = pd.read_csv(mvhv_csv, dtype=str)

    # --- Robust voltage parsing (take the max number, handles '380/220 kV', '225,0', etc.) ---
    def _coerce_voltage_smart(series: pd.Series) -> pd.Series:
        s = series.astype(str).fillna("").str.strip()
        s = s.str.replace(",", ".", regex=False)
        s = s.str.replace(r"\s*[kK]\s*[vV]\b", "", regex=True)
        def pick_max(text: str):
            nums = re.findall(r"\d+(?:\.\d+)?", text)
            if not nums: return np.nan
            vals = [float(n) for n in nums]
            fixed = [(v/1000.0 if v >= 1000 and abs((v/1000)-round(v/1000)) < 1e-6 else v) for v in vals]
            return max(fixed)
        return s.map(pick_max)

    v = _coerce_voltage_smart(df.get("v_nom") if "v_nom" in df.columns else df.get("voltage"))
    length_km  = pd.to_numeric(df.get("length"),   errors="coerce").div(1000.0)   # meters -> km
    circuits   = pd.to_numeric(df.get("circuits"), errors="coerce").fillna(1.0).clip(lower=1.0)

    # route-km vs circuit-km
    route_km   = length_km
    circuit_km = length_km * circuits

    # Buckets for headline chart
    def _bucket_voltage_kv(x: float) -> str:
        if pd.isna(x): return "unknown"
        if 200 <= x < 260: return "220–225 kV"
        if 360 <= x <= 420: return "380–400 kV"
        return "other"

    buckets = v.map(_bucket_voltage_kv)
    by_bucket = (pd.DataFrame({"bucket":buckets, "route_km":route_km,
                               "circuit_km":circuit_km, "circuits":circuits})
                 .groupby("bucket", dropna=False)
                 .agg(routes=("bucket","size"),
                      route_km=("route_km","sum"),
                      circuit_km=("circuit_km","sum"),
                      avg_circuits=("circuits","mean"))
                 .reset_index())

    # Exact voltages (rounded) – 220/225/380/400 first, then top others
    v_round = v.round().dropna()
    by_voltage = (pd.DataFrame({"v_kV":v_round, "circuit_km":circuit_km.loc[v_round.index]})
                  .groupby("v_kV").sum().reset_index())
    wanted = [220,225,380,400]
    front = by_voltage[by_voltage["v_kV"].isin(wanted)]
    rest  = by_voltage[~by_voltage["v_kV"].isin(wanted)].sort_values("circuit_km", ascending=False)
    by_voltage = pd.concat([front, rest.head(8)], ignore_index=True)  # keep compact

    pypsa_total_ckm = float(by_bucket["circuit_km"].sum())
    pct_diff_total  = 100.0 * (pypsa_total_ckm - TSO_GERMANY_TOTAL_CIRCUIT_KM) / TSO_GERMANY_TOTAL_CIRCUIT_KM

    # ---------- Plots (matplotlib, compact, professional) ----------
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, MaxNLocator

    def _thousands(x, _pos):
        try:
            return f"{x:,.0f}"
        except Exception:
            return str(x)

    # 1) PyPSA MV/HV circuit-km by bucket
    plot_bucket = OUTPUT_DIR / "pypsa_mvhv_circuit_km_by_voltage.png"
    fig = plt.figure(figsize=(4.4, 3.1))
    sub = by_bucket[by_bucket["bucket"].isin(["220–225 kV","380–400 kV","other","unknown"])]
    ax = plt.gca()
    ax.bar(sub["bucket"].astype(str), sub["circuit_km"])
    ax.set_ylabel("Circuit-km")
    ax.yaxis.set_major_formatter(FuncFormatter(_thousands))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.grid(axis='y', linestyle=':', alpha=0.6)
    for x, y in enumerate(sub["circuit_km"]):
        ax.text(x, y, f"{y:,.0f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout(); fig.savefig(plot_bucket, dpi=150); plt.close(fig)

    # 2) PyPSA MV/HV circuit-km by exact voltage (focused list)
    plot_exact = OUTPUT_DIR / "pypsa_mvhv_circuit_km_by_exact_voltage.png"
    fig = plt.figure(figsize=(5.6, 3.2))
    labels = by_voltage["v_kV"].astype(int).astype(str)
    vals   = by_voltage["circuit_km"].values
    ax = plt.gca()
    ax.bar(labels, vals)
    ax.set_ylabel("Circuit-km")
    ax.yaxis.set_major_formatter(FuncFormatter(_thousands))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.grid(axis='y', linestyle=':', alpha=0.6)
    plt.xticks(rotation=0)
    for x, y in enumerate(vals):
        ax.text(x, y, f"{y:,.0f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout(); fig.savefig(plot_exact, dpi=150); plt.close(fig)

    # 3) National total: PyPSA vs TSO
    plot_vs = OUTPUT_DIR / "pypsa_vs_tso_total_circuit_km.png"
    fig = plt.figure(figsize=(3.9, 3.0))
    labels = ["PyPSA", "TSO"]
    vals   = [pypsa_total_ckm, TSO_GERMANY_TOTAL_CIRCUIT_KM]
    ax = plt.gca()
    ax.bar(labels, vals)
    ax.set_ylabel("Circuit-km")
    ax.yaxis.set_major_formatter(FuncFormatter(_thousands))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.grid(axis='y', linestyle=':', alpha=0.6)
    for x, y in enumerate(vals):
        ax.text(x, y, f"{y:,.0f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout(); fig.savefig(plot_vs, dpi=150); plt.close(fig)

    # ---------- Tables ----------
    bucket_tbl = (by_bucket.assign(
                    route_km=lambda d: d["route_km"].round(0).map("{:,.0f}".format),
                    circuit_km=lambda d: d["circuit_km"].round(0).map("{:,.0f}".format),
                    avg_circuits=lambda d: d["avg_circuits"].round(2),
                  )
                  .rename(columns={"bucket":"Voltage bucket",
                                   "routes":"# routes",
                                   "route_km":"Route-km",
                                   "circuit_km":"Circuit-km",
                                   "avg_circuits":"Avg circuits"})
                  .to_html(index=False, escape=False))

    exact_tbl = (by_voltage
                 .assign(v_kV=lambda d: d["v_kV"].astype(int),
                         circuit_km=lambda d: d["circuit_km"].round(0).map("{:,.0f}".format))
                 .rename(columns={"v_kV":"Voltage [kV]","circuit_km":"Circuit-km"})
                 .to_html(index=False, escape=False))

    delta_txt = f"{pct_diff_total:+.1f}%"
    delta_class = "pos" if pct_diff_total >= 0 else "neg"

    # ---------- Compact, polished HTML (no per-TSO charts) ----------
    html = f"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>PyPSA MV/HV vs TSO — Summary</title>
<style>
  :root {{
    --bg:#0b1020; --card:#111633; --ink:#e8ecff; --muted:#aab3d8; --accent:#7aa2ff; --soft:#1b2349; --good:#20c997; --bad:#ff6b6b;
  }}
  * {{ box-sizing: border-box; }}
  body {{ margin: 22px; color: var(--ink); background: radial-gradient(1200px 800px at 10% -10%, #1a2055, var(--bg)); font: 14px/1.5 ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial; }}
  h1 {{ font-size: 24px; margin: 0 0 10px; letter-spacing:.2px; }}
  h2 {{ font-size: 17px; margin: 14px 0 8px; color: var(--accent); }}
  .grid {{ display: grid; grid-template-columns: repeat(12, 1fr); gap: 14px; }}
  .card {{ background: linear-gradient(180deg, var(--card), var(--soft)); border: 1px solid #222a55; border-radius: 12px; padding: 12px 14px; box-shadow: 0 10px 28px rgba(0,0,0,.22); }}
  .kpis {{ grid-column: span 12; display: grid; grid-template-columns: repeat(auto-fit, minmax(220px,1fr)); gap: 12px; }}
  .kpi h3 {{ margin:0 0 4px; font-size: 11px; text-transform: uppercase; letter-spacing:.4px; color: var(--muted); }}
  .kpi .v {{ font-size: 20px; font-weight: 700; }}
  .delta {{ font-weight:700; padding:2px 8px; border-radius:999px; margin-left:8px; }}
  .delta.pos {{ background: rgba(32,201,151,.15); color: var(--good); }}
  .delta.neg {{ background: rgba(255,107,107,.15); color: var(--bad); }}
  .pair {{ grid-column: span 6; }}
  .wide {{ grid-column: span 12; }}
  table {{ width:100%; border-collapse: collapse; border: 1px solid #283069; font-size: 13px; }}
  th, td {{ border:1px solid #283069; padding:7px 9px; }}
  th {{ background:#141b3f; color: var(--muted); text-align:left; }}
  img {{ width:100%; height:auto; border-radius: 10px; border: 1px solid #283069; background:#0e1433; }}
  .note {{ color: var(--muted); font-size: 12px; }}
</style>
</head>
<body>
  <h1>PyPSA MV/HV vs TSO — Germany</h1>

  <div class="kpis card">
    <div class="kpi"><h3>PyPSA MV/HV total (circuit-km)</h3><div class="v">{pypsa_total_ckm:,.0f}</div></div>
    <div class="kpi"><h3>TSO reported national total (circuit-km)</h3><div class="v">{TSO_GERMANY_TOTAL_CIRCUIT_KM:,.0f}</div></div>
    <div class="kpi"><h3>Δ PyPSA vs TSO</h3><div class="v">{pypsa_total_ckm-TSO_GERMANY_TOTAL_CIRCUIT_KM:,.0f}<span class="delta {delta_class}">{delta_txt}</span></div></div>
  </div>

  <div class="grid" style="margin-top:12px;">
    <div class="pair card">
      <h2>By voltage bucket (circuit-km)</h2>
      <img src="{plot_bucket.name}" alt="PyPSA by bucket"/>
    </div>
    <div class="pair card">
      <h2>National total (circuit-km)</h2>
      <img src="{plot_vs.name}" alt="PyPSA vs TSO"/>
    </div>

    <div class="wide card">
      <h2>PyPSA circuit-km by exact voltage</h2>
      <img src="{plot_exact.name}" alt="PyPSA by exact voltage"/>
    </div>

    <div class="pair card">
      <h2>Voltage buckets — table</h2>
      {bucket_tbl}
    </div>
    <div class="pair card">
      <h2>Exact voltages — table</h2>
      {exact_tbl}
    </div>
  </div>

  <p class="note" style="margin-top:10px;">
    Method: PyPSA line lengths (meters) × <code>circuits</code> → circuit-km. HVDC is modeled as <code>links</code> in PyPSA and is not counted here.
    International corridors aren’t clipped at borders in this run. Differences can also stem from asset definitions, in-service status, and publication years.
  </p>
</body>
</html>
"""
    out_html.parent.mkdir(parents=True, exist_ok=True)
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"[REPORT] Wrote {out_html}")
    return out_html







def main():
    """
    Main function to process both datasets.
    """
    print("Processing PyPSA and JAO line data...")
    buffer_km = 0.0  # You can adjust this buffer value if needed

    # Process both datasets
    try:
        pypsa_output = process_pypsa_lines(buffer_km=buffer_km)
        print(f"PyPSA output: {pypsa_output}")
    except Exception as e:
        print(f"PyPSA processing failed: {e}")

    try:
        jao_output = process_jao_lines(buffer_km=buffer_km)
        print(f"JAO output: {jao_output}")
    except Exception as e:
        print(f"JAO processing failed: {e}")

    # After successful processing, try to build an interactive map
    try:
        pypsa_gpkg = (OUTPUT_DIR / "pypsa_clipped_lines.gpkg")
        jao_gpkg   = (OUTPUT_DIR / "jao_clipped_lines.gpkg")

        # You can pass None for either layer if one failed/was skipped
        map_path = plot_lines_map(
            pypsa_path=pypsa_gpkg if pypsa_gpkg.exists() else None,
            jao_path=jao_gpkg if jao_gpkg.exists() else None,
            germany_gdf=None,                 # auto-load from your boundary function
            out_html=OUTPUT_DIR / "grid_lines_map.html",
            simplify_tolerance=0.0            # e.g. 0.0005 to reduce file size
        )
        print(f"Map saved to: {map_path}")
    except Exception as e:
        print(f"Map creation failed: {e}")

    try:
        mvhv_csv = OUTPUT_DIR / "pypsa_clipped_MVHV_lines.csv"
        build_mvhv_vs_tso_report(mvhv_csv=mvhv_csv,
                                 out_html=OUTPUT_DIR / "pypsa_mvhv_vs_tso.html")
    except Exception as e:
        print(f"[REPORT] Failed to build MV/HV vs TSO report: {e}")

    print("\nProcessing completed!")


if __name__ == "__main__":
    main()