#!/usr/bin/env python
# match_transformers.py - Script for matching transformers between JAO and PyPSA datasets

import os
import pandas as pd
import geopandas as gpd
from pathlib import Path
import numpy as np
from shapely.geometry import Point, LineString
from shapely import wkt
import folium
import folium.plugins as plugins
import json
from math import radians
from sklearn.metrics.pairwise import haversine_distances
from typing import Optional, Union, Iterable



from pathlib import Path
import geopandas as gpd
from shapely.ops import unary_union
from shapely.geometry import Point
from shapely.prepared import prep



from pathlib import Path
import geopandas as gpd
from shapely.ops import unary_union
from shapely.prepared import prep

import re
from typing import Union, List

# exact, fixed path you showed in the screenshot
GER_DATA_DIR = Path("/home/mohsen/PycharmProjects/freeGon/grid-matcher-original/432-Mikroprojekt-freeGon/grid_matcher/data")
GER_BOUNDARY_FILE = GER_DATA_DIR / "georef-germany-gemeinde@public.geojson"  # high-res municipalities

import re
from shapely import wkt as _wkt

_WKT_HEAD = re.compile(
    r'(POINT|LINESTRING|POLYGON|MULTIPOINT|MULTILINESTRING|MULTIPOLYGON)\s*\(',
    re.IGNORECASE
)

# top of file
import csv
from shapely.geometry.base import BaseGeometry

def _write_clean_wkt_csv(gdf: gpd.GeoDataFrame, path: Path) -> None:
    df = gdf.copy()
    # convert to WKT strings robustly
    def _to_wkt(g):
        if isinstance(g, BaseGeometry) and not g.is_empty:
            return g.wkt
        # allow already-WKT strings to pass through
        if isinstance(g, str) and any(k in g.upper() for k in ("POINT(", "LINESTRING(", "POLYGON(", "MULTI")):
            return g
        return None
    df["geometry"] = df["geometry"].apply(_to_wkt)
    # QUOTE_MINIMAL will quote any field with commas → keeps WKT intact in one column
    df.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)


def _safe_wkt_load(val):
    """Return shapely geometry from possibly dirty strings like 'USUALLY ... POINT(...)', else None."""
    if not isinstance(val, str):
        return None
    s = val.strip().strip('"').strip("'")
    m = _WKT_HEAD.search(s)
    if not m:
        return None
    start = m.start()
    t = s[start:]  # trim leading junk
    depth = 0; end = None
    for i, ch in enumerate(t):
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    cand = t[:end] if end else (t[:t.rfind(')')+1] if ')' in t else t)
    try:
        return _wkt.loads(cand)
    except Exception:
        return None


def load_germany_boundary():
    """
    Load the *real* Germany border from the municipal GeoJSON and dissolve to one polygon.
    No guessing, no downloading, no prompts.
    """
    gdf = gpd.read_file(GER_BOUNDARY_FILE)
    if gdf.crs is None or gdf.crs.to_string().upper() != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")
    # dissolve to one (Multi)Polygon and clean tiny slivers
    return unary_union(gdf.geometry).buffer(0)

def clip_to_germany(gdf, germany_geom):
    """
    Keep only geometries inside Germany.
    - Points: must be strictly within the polygon
    - Lines/others: any intersection with the polygon counts
    """
    if gdf.crs is None or gdf.crs.to_string().upper() != "EPSG:4326":
        gdf = gdf.set_crs("EPSG:4326", allow_override=True)

    def _keep(geom):
        if geom is None or geom.is_empty:
            return False
        gt = geom.geom_type
        if gt == "Point":
            return geom.within(germany_geom)
        # for LineString/Polygon/etc use intersects so border-spanning lines are kept
        return geom.intersects(germany_geom)

    mask = gdf.geometry.apply(_keep)
    return gdf.loc[mask].copy()

def drop_dummy_jao_transformers(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Remove artificial/dummy JAO transformers like T_0, T_1, ... and 'Added for coherence' rows."""
    df = gdf.copy()
    if "name" in df.columns:
        df = df[~df["name"].astype(str).str.startswith("T_", na=False)]
    # many dummy rows carry this in the Comment column
    for col in ["Comment", "comment"]:
        if col in df.columns:
            df = df[~df[col].astype(str).str.contains("Added for coherence", case=False, na=False)]
    # drop empties/invalid geometry just in case
    if "geometry" in df.columns:
        df = df[df["geometry"].notna() & ~df["geometry"].is_empty]
    return df


def load_transformer_data(jao_path, pypsa_path):
    """
    Robustly load transformers from JAO and PyPSA, fixing geometry that was split
    across columns (e.g., 'LINESTRING(...)' with tabs inside single quotes).
    """
    print(f"Loading JAO transformers from: {jao_path}")
    print(f"Loading PyPSA transformers from: {pypsa_path}")

    import re

    def _read_tx_any(path, label):
        # Try reading with auto-sep and single-quote quoting (handles tabs/commas)
        try:
            df = pd.read_csv(path, sep=None, engine="python", quotechar="'")
        except Exception:
            df = pd.read_csv(path, engine="python")

        cols = list(df.columns)

        # Find the first column that looks like it contains WKT tokens
        def _find_geom_col():
            # Prefer an actual 'geometry' column name
            if "geometry" in df.columns:
                return list(df.columns).index("geometry")
            # Otherwise detect by content
            for i, c in enumerate(df.columns):
                s = df[c].astype(str)
                if s.str.contains(r"(MULTI(?:LINESTRING|POINT|POLYGON)|LINESTRING|POINT|POLYGON)\s*\(",
                                  case=False, regex=True, na=False).any():
                    return i
            return None

        gi = _find_geom_col()

        if gi is not None:
            right_cols = cols[gi:]

            # Glue the split pieces back together row-wise
            def _glue(row):
                txt = " ".join(str(row[c]) for c in right_cols if pd.notna(row[c]))
                txt = txt.replace("\t", " ")
                txt = " ".join(txt.split())  # collapse whitespace
                # Keep only the WKT substring (from token to last ')')
                m = re.search(r"(MULTI(?:LINESTRING|POINT|POLYGON)|LINESTRING|POINT|POLYGON)\s*\(.*\)",
                              txt, flags=re.I)
                if not m:
                    return None
                out = m.group(0).strip(" '\"")
                return out

            df["geometry"] = df.apply(_glue, axis=1)
            # Keep original columns left of geometry + the rebuilt geometry
            keep = cols[:gi] + ["geometry"]
            df = df[keep]
        else:
            # No geometry-looking column found
            df["geometry"] = None

        # Parse WKT safely
        df["geometry"] = df["geometry"].apply(_safe_wkt_load)

        # Build GeoDataFrame and drop empties
        gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")
        mask = gdf["geometry"].notna() & ~gdf["geometry"].apply(lambda g: getattr(g, "is_empty", True))
        dropped = int((~mask).sum())
        if dropped:
            print(f"{label}: {dropped} rows have invalid/missing geometry and will be ignored later.")
        return gdf.loc[mask].copy()

    # --- JAO ---
    jao_gdf = _read_tx_any(jao_path, "JAO")

    # Drop dummy placeholders like T_* and "Added for coherence"
    if "name" in jao_gdf.columns:
        before = len(jao_gdf)
        jao_gdf = jao_gdf[~jao_gdf["name"].astype(str).str.startswith("T_", na=False)].copy()
        if before - len(jao_gdf):
            print(f"JAO: dropped {before - len(jao_gdf)} dummy 'T_*' rows.")
    for c in ("Comment", "comment"):
        if c in jao_gdf.columns:
            b = len(jao_gdf)
            jao_gdf = jao_gdf[~jao_gdf[c].astype(str).str.contains("Added for coherence", case=False, na=False)].copy()
            if b - len(jao_gdf):
                print(f"JAO: dropped {b - len(jao_gdf)} 'Added for coherence' rows.")

    # --- PyPSA ---
    pypsa_gdf = _read_tx_any(pypsa_path, "PyPSA")

    print(f"Loaded {len(jao_gdf)} JAO transformers (valid geom)")
    print(f"Loaded {len(pypsa_gdf)} PyPSA transformers (valid geom)")
    return jao_gdf, pypsa_gdf



def match_transformers_by_location(jao_gdf, pypsa_gdf, distance_threshold_km=5.0):
    """
    Match transformers based on geographical proximity.
    """
    matches = []

    # For each JAO transformer
    for idx, jao_row in jao_gdf.iterrows():
        # Skip if no geometry
        if not jao_row.geometry or jao_row.geometry.is_empty:
            continue

        jao_point = jao_row.geometry
        jao_lat = jao_point.y
        jao_lon = jao_point.x

        # Calculate distances to all PyPSA transformers
        distances = []
        for pypsa_idx, pypsa_row in pypsa_gdf.iterrows():
            # Skip if no geometry
            if not pypsa_row.geometry or pypsa_row.geometry.is_empty:
                distances.append(float('inf'))
                continue

            # Extract coordinates from PyPSA geometry
            if pypsa_row.geometry.geom_type == 'Point':
                pypsa_lat = pypsa_row.geometry.y
                pypsa_lon = pypsa_row.geometry.x
            elif pypsa_row.geometry.geom_type == 'LineString':
                centroid = pypsa_row.geometry.centroid
                pypsa_lat = centroid.y
                pypsa_lon = centroid.x
            else:
                distances.append(float('inf'))
                continue

            # Calculate haversine distance
            point1 = [radians(jao_lat), radians(jao_lon)]
            point2 = [radians(pypsa_lat), radians(pypsa_lon)]
            distance_km = haversine_distances([point1, point2])[0, 1] * 6371  # Earth radius in km

            distances.append(distance_km)

        # Find the best match(es) within the threshold
        distances = np.array(distances)
        candidate_indices = np.where(distances < distance_threshold_km)[0]

        if len(candidate_indices) > 0:
            # Sort by distance
            sorted_indices = candidate_indices[np.argsort(distances[candidate_indices])]

            # Get the best match
            best_idx = sorted_indices[0]
            best_distance = distances[best_idx]
            pypsa_row = pypsa_gdf.iloc[best_idx]

            # Create match record
            match = {
                "jao_id": jao_row.get("EIC_Code", str(idx)),
                "jao_name": jao_row.get("name", ""),
                "pypsa_id": pypsa_row.get("transformer_id", ""),
                "distance_km": best_distance,
                "matched": True,
                "match_type": "location",
                "match_confidence": max(0, 1.0 - (best_distance / distance_threshold_km)),
                # Voltage level comparison for verification
                "jao_voltage_primary": jao_row.get("Voltage_level(kV) Primary", 0),
                "jao_voltage_secondary": jao_row.get("Voltage_level(kV) Secondary", 0),
                "pypsa_voltage_bus0": pypsa_row.get("voltage_bus0", 0),
                "pypsa_voltage_bus1": pypsa_row.get("voltage_bus1", 0),
                # Connection details
                "bus0": pypsa_row.get("bus0", ""),
                "bus1": pypsa_row.get("bus1", ""),
                "s_nom": pypsa_row.get("s_nom", 0),
                # Electrical parameters
                "jao_r": jao_row.get("r", 0),
                "jao_x": jao_row.get("x", 0),
                "jao_b": jao_row.get("b", 0),
                "jao_g": jao_row.get("g", 0),
                # Transformer-specific parameters
                "jao_theta": jao_row.get("Theta θ (°)", 0),
                "jao_phase_regulation": jao_row.get("Phase Regulation δu (%)", 0),
                "jao_angle_regulation": jao_row.get("Angle Regulation δu (%)", 0),
                "jao_symmetry": jao_row.get("Symmetrical/Asymmetrical", ""),
                "jao_imax": jao_row.get("Maximum Current Imax (A) primary Fixed", 0)
            }

            # Calculate voltage match score
            v1_match = abs(jao_row.get("Voltage_level(kV) Primary", 0) - pypsa_row.get("voltage_bus0", 0)) < 10
            v2_match = abs(jao_row.get("Voltage_level(kV) Secondary", 0) - pypsa_row.get("voltage_bus1", 0)) < 10
            v_match_score = (v1_match + v2_match) / 2

            # Adjust confidence based on voltage match
            match["match_confidence"] *= (0.5 + 0.5 * v_match_score)

            matches.append(match)
        else:
            # No match found
            matches.append({
                "jao_id": jao_row.get("EIC_Code", str(idx)),
                "jao_name": jao_row.get("name", ""),
                "pypsa_id": None,
                "matched": False,
                "match_type": "none",
                "match_confidence": 0.0,
                "jao_r": jao_row.get("r", 0),
                "jao_x": jao_row.get("x", 0),
                "jao_b": jao_row.get("b", 0),
                "jao_g": jao_row.get("g", 0),
                "jao_theta": jao_row.get("Theta θ (°)", 0),
                "jao_phase_regulation": jao_row.get("Phase Regulation δu (%)", 0),
                "jao_angle_regulation": jao_row.get("Angle Regulation δu (%)", 0),
                "jao_symmetry": jao_row.get("Symmetrical/Asymmetrical", ""),
                "jao_imax": jao_row.get("Maximum Current Imax (A) primary Fixed", 0)
            })

    return matches


def match_transformers_by_voltage(jao_gdf, pypsa_gdf, matches):
    """
    Refine transformer matches based on voltage level consistency.
    """
    for match in matches:
        if match["matched"]:
            # Get JAO row
            jao_rows = jao_gdf[jao_gdf["EIC_Code"] == match["jao_id"]]
            if len(jao_rows) == 0:
                continue

            jao_row = jao_rows.iloc[0]

            # Get PyPSA row
            pypsa_rows = pypsa_gdf[pypsa_gdf["transformer_id"] == match["pypsa_id"]]
            if len(pypsa_rows) == 0:
                continue

            pypsa_row = pypsa_rows.iloc[0]

            # Check voltage level consistency - comparing primary and secondary voltages
            # Note: Sometimes the order might be reversed, so check both permutations
            v_prim_match = abs(jao_row.get("Voltage_level(kV) Primary", 0) - pypsa_row.get("voltage_bus0", 0)) < 10
            v_sec_match = abs(jao_row.get("Voltage_level(kV) Secondary", 0) - pypsa_row.get("voltage_bus1", 0)) < 10

            # Alternate permutation
            alt_prim_match = abs(jao_row.get("Voltage_level(kV) Primary", 0) - pypsa_row.get("voltage_bus1", 0)) < 10
            alt_sec_match = abs(jao_row.get("Voltage_level(kV) Secondary", 0) - pypsa_row.get("voltage_bus0", 0)) < 10

            # Calculate overall voltage match score
            if (v_prim_match and v_sec_match) or (alt_prim_match and alt_sec_match):
                v_score = 1.0
            elif v_prim_match or v_sec_match or alt_prim_match or alt_sec_match:
                v_score = 0.5
            else:
                v_score = 0.0

            # Update match confidence based on voltage matching
            match["match_confidence"] = max(match["match_confidence"],
                                            min(match["match_confidence"] * 1.2, 1.0) * (0.5 + 0.5 * v_score))

            # If voltages don't match at all, mark as uncertain
            if v_score == 0.0:
                match["match_confidence"] *= 0.5
                match["match_type"] = "location_voltage_mismatch"

    return matches


def match_transformers_by_eic(jao_gdf, pypsa_gdf):
    """
    Match transformers based on EIC code.
    """
    matches = []

    # Check if PyPSA has EIC codes
    if "EIC_Code" not in pypsa_gdf.columns:
        print("No EIC codes found in PyPSA transformers data")
        return matches

    # For each JAO transformer
    for idx, jao_row in jao_gdf.iterrows():
        # Skip if no EIC code
        if pd.isna(jao_row.get("EIC_Code")) or jao_row.get("EIC_Code") == "":
            continue

        jao_eic = jao_row.get("EIC_Code")

        # Find PyPSA transformers with this EIC
        pypsa_matches = pypsa_gdf[pypsa_gdf["EIC_Code"] == jao_eic]

        if len(pypsa_matches) > 0:
            for pypsa_idx, pypsa_row in pypsa_matches.iterrows():
                match = {
                    "jao_id": jao_eic,
                    "jao_name": jao_row.get("name", ""),
                    "pypsa_id": pypsa_row.get("transformer_id", ""),
                    "distance_km": 0,  # Not relevant for EIC matching
                    "matched": True,
                    "match_type": "eic",
                    "match_confidence": 1.0,  # High confidence for EIC matches
                    "jao_voltage_primary": jao_row.get("Voltage_level(kV) Primary", 0),
                    "jao_voltage_secondary": jao_row.get("Voltage_level(kV) Secondary", 0),
                    "pypsa_voltage_bus0": pypsa_row.get("voltage_bus0", 0),
                    "pypsa_voltage_bus1": pypsa_row.get("voltage_bus1", 0),
                    "bus0": pypsa_row.get("bus0", ""),
                    "bus1": pypsa_row.get("bus1", ""),
                    "s_nom": pypsa_row.get("s_nom", 0),
                    "jao_r": jao_row.get("r", 0),
                    "jao_x": jao_row.get("x", 0),
                    "jao_b": jao_row.get("b", 0),
                    "jao_g": jao_row.get("g", 0),
                    "jao_theta": jao_row.get("Theta θ (°)", 0),
                    "jao_phase_regulation": jao_row.get("Phase Regulation δu (%)", 0),
                    "jao_angle_regulation": jao_row.get("Angle Regulation δu (%)", 0),
                    "jao_symmetry": jao_row.get("Symmetrical/Asymmetrical", ""),
                    "jao_imax": jao_row.get("Maximum Current Imax (A) primary Fixed", 0)
                }
                matches.append(match)

    return matches


def run_transformer_matching(jao_gdf, pypsa_gdf, distance_threshold_km=5.0):
    """
    Run the full transformer matching process.
    """
    # First try EIC matching if available
    eic_matches = match_transformers_by_eic(jao_gdf, pypsa_gdf)
    print(f"Found {len(eic_matches)} matches by EIC code")

    # Track which JAO transformers have been matched by EIC
    matched_jao_ids = {m["jao_id"] for m in eic_matches if m["matched"]}

    # Filter out already matched JAO transformers for location matching
    unmatched_jao_gdf = jao_gdf[~jao_gdf["EIC_Code"].isin(matched_jao_ids)]

    # Then do location-based matching for the rest
    print(f"Attempting location matching for {len(unmatched_jao_gdf)} unmatched transformers")
    location_matches = match_transformers_by_location(unmatched_jao_gdf, pypsa_gdf, distance_threshold_km)

    # Refine location matches by voltage consistency
    location_matches = match_transformers_by_voltage(jao_gdf, pypsa_gdf, location_matches)

    # Combine all matches
    all_matches = eic_matches + location_matches

    # Count matches
    matched_count = sum(1 for m in all_matches if m["matched"])
    print(
        f"Total transformers matched: {matched_count} out of {len(jao_gdf)} ({matched_count / len(jao_gdf) * 100:.1f}%)")

    return all_matches

def create_transformer_match_visualization(jao_gdf, pypsa_gdf, matches, output_file, germany_boundary):
    """
    Interactive HTML map of transformer matches (Germany only).
    Assumes jao_gdf / pypsa_gdf are EPSG:4326 and already clipped to Germany.
    germany_boundary: shapely (Multi)Polygon in EPSG:4326.
    """
    import json
    import numpy as np
    import folium
    from folium import plugins
    from shapely.geometry import Point, mapping
    from shapely.prepared import prep

    # ---------- helpers ----------
    P = prep(germany_boundary.buffer(1e-9))

    def is_in_germany(lon, lat):
        try:
            return P.intersects(Point(lon, lat))
        except Exception:
            return False

    def to_xy(geom):
        if geom is None or getattr(geom, "is_empty", True):
            return None
        try:
            if geom.geom_type == "Point":
                return (geom.x, geom.y)
            c = geom.centroid
            return (c.x, c.y)
        except Exception:
            return None

    def native(v):
        if isinstance(v, np.integer):  return int(v)
        if isinstance(v, np.floating): return float(v)
        if isinstance(v, np.bool_):    return bool(v)
        return v

    # ---------- base map ----------
    m = folium.Map(location=[51.1657, 10.4515], zoom_start=6, tiles="CartoDB positron")
    map_js_var = m.get_name()

    print(f"Creating visualization with {len(jao_gdf)} JAO transformers and {len(pypsa_gdf)} PyPSA transformers")
    print(f"Matches count: {len(matches)}")

    # Feature groups
    fg_jao_matched     = folium.FeatureGroup(name="JAO Matched Transformers", overlay=True, show=True)
    fg_jao_unmatched   = folium.FeatureGroup(name="JAO Unmatched Transformers", overlay=True, show=True)
    fg_pypsa_matched   = folium.FeatureGroup(name="PyPSA Matched Transformers", overlay=True, show=True)
    fg_pypsa_unmatched = folium.FeatureGroup(name="PyPSA Unmatched Transformers", overlay=True, show=True)

    # Fast lookups
    jao_by_eic = {}
    for _, r in jao_gdf.iterrows():
        k = r.get("EIC_Code", None)
        if k is not None and str(k) != "":
            jao_by_eic[str(k)] = r
    jao_by_name = {str(r.get("name","")): r for _, r in jao_gdf.iterrows() if str(r.get("name","")) != ""}
    pypsa_by_id = {str(r.get("transformer_id","")): r for _, r in pypsa_gdf.iterrows() if str(r.get("transformer_id","")) != ""}

    matched_pypsa_ids = set()
    match_data = []
    j_mat = j_unm = p_mat = p_unm = 0

    # ---------- JAO markers ----------
    for match in matches:
        jao_id   = str(match.get("jao_id", "") or "")
        jao_name = str(match.get("jao_name", "") or "")

        row = jao_by_eic.get(jao_id) if jao_id else None
        if row is None and jao_name:
            row = jao_by_name.get(jao_name)
        if row is None:
            continue

        xy = to_xy(row.get("geometry", None))
        if xy is None:
            continue
        lon, lat = xy
        if not is_in_germany(lon, lat):
            continue

        is_matched = bool(match.get("matched", False))
        if is_matched: j_mat += 1
        else:          j_unm += 1

        # Popup
        lines = [
            "<div style='min-width:300px'>",
            f"<h4>JAO {'Matched' if is_matched else 'Unmatched'} Transformer</h4>",
            f"<b>JAO Name:</b> {match.get('jao_name','N/A')}<br>",
            f"<b>JAO EIC:</b> {match.get('jao_id','N/A')}<br>",
        ]
        if is_matched:
            lines += [
                f"<b>PyPSA ID:</b> {match.get('pypsa_id','')}<br>",
                f"<b>Match Type:</b> {match.get('match_type','N/A')}<br>",
                f"<b>Confidence:</b> {match.get('match_confidence',0):.2f}<br>",
                f"<b>Distance:</b> {native(round(match.get('distance_km',0.0),3))} km<br>",
            ]
        lines += [
            "<hr>",
            f"<b>JAO Primary Voltage:</b> {native(match.get('jao_voltage_primary','N/A'))} kV<br>",
            f"<b>JAO Secondary Voltage:</b> {native(match.get('jao_voltage_secondary','N/A'))} kV<br>",
        ]
        if is_matched:
            lines += [
                f"<b>PyPSA Bus0 Voltage:</b> {native(match.get('pypsa_voltage_bus0','N/A'))} kV<br>",
                f"<b>PyPSA Bus1 Voltage:</b> {native(match.get('pypsa_voltage_bus1','N/A'))} kV<br>",
            ]
        lines.append("</div>")
        jao_popup = "\n".join(lines)

        folium.CircleMarker(
            [lat, lon], radius=6,
            tooltip=f"{match.get('jao_name','Unknown')} (JAO {'Matched' if is_matched else 'Unmatched'})",
            color=('green' if is_matched else 'red'),
            fill=True, fill_color=('green' if is_matched else 'red'), fill_opacity=0.7
        ).add_child(folium.Popup(jao_popup, max_width=500)).add_to(
            fg_jao_matched if is_matched else fg_jao_unmatched
        )

        match_data.append({
            'id': native(match.get('jao_id','')),
            'jao_name': native(match.get('jao_name','')),
            'pypsa_id': native(match.get('pypsa_id','')) if is_matched else '',
            'match_type': native(match.get('match_type','none')),
            'confidence': native(match.get('match_confidence', 0.0)),
            'jao_v_primary': native(match.get('jao_voltage_primary', 0)),
            'jao_v_secondary': native(match.get('jao_voltage_secondary', 0)),
            'pypsa_v_bus0': native(match.get('pypsa_voltage_bus0', 0)),
            'pypsa_v_bus1': native(match.get('pypsa_voltage_bus1', 0)),
            'lat': native(lat), 'lon': native(lon),
            'matched': is_matched, 'dataset': 'jao'
        })

        # Matched PyPSA + line
        if is_matched:
            pid = str(match.get("pypsa_id", "") or "")
            prow = pypsa_by_id.get(pid)
            if prow is not None:
                pxy = to_xy(prow.get("geometry", None))
                if pxy is not None:
                    plon, plat = pxy
                    if is_in_germany(plon, plat):
                        p_mat += 1
                        matched_pypsa_ids.add(pid)

                        pypsa_popup = f"""
                        <div style='min-width:300px'>
                          <h4>PyPSA Matched Transformer</h4>
                          <b>ID:</b> {prow.get('transformer_id','')}<br>
                          <b>Matched with JAO:</b> {match.get('jao_name','N/A')}<br>
                          <b>Voltage Bus0:</b> {native(prow.get('voltage_bus0',''))} kV<br>
                          <b>Voltage Bus1:</b> {native(prow.get('voltage_bus1',''))} kV<br>
                          <b>s_nom:</b> {native(prow.get('s_nom',''))} MVA<br>
                        </div>
                        """

                        folium.CircleMarker(
                            [plat, plon], radius=8,
                            tooltip=f"PyPSA: {prow.get('transformer_id','')} (Matched)",
                            color='blue', fill=True, fill_color='blue', fill_opacity=0.7
                        ).add_child(folium.Popup(pypsa_popup, max_width=500)).add_to(fg_pypsa_matched)

                        folium.PolyLine(
                            [(lat, lon), (plat, plon)],
                            color='green', weight=2, opacity=0.7,
                            tooltip=f"Match: {match.get('jao_name','')} - {prow.get('transformer_id','')}"
                        ).add_to(m)

                        match_data.append({
                            'id': native(prow.get('transformer_id','')),
                            'jao_name': native(match.get('jao_name','')),
                            'pypsa_id': native(prow.get('transformer_id','')),
                            'match_type': native(match.get('match_type','location')),
                            'confidence': native(match.get('match_confidence', 0.0)),
                            'jao_v_primary': native(match.get('jao_voltage_primary', 0)),
                            'jao_v_secondary': native(match.get('jao_voltage_secondary', 0)),
                            'pypsa_v_bus0': native(prow.get('voltage_bus0', 0)),
                            'pypsa_v_bus1': native(prow.get('voltage_bus1', 0)),
                            'lat': native(plat), 'lon': native(plon),
                            'matched': True, 'dataset': 'pypsa'
                        })

    # ---------- remaining unmatched PyPSA ----------
    for _, prow in pypsa_gdf.iterrows():
        pid = str(prow.get('transformer_id','') or '')
        if pid in matched_pypsa_ids:
            continue
        pxy = to_xy(prow.get("geometry", None))
        if pxy is None:
            continue
        plon, plat = pxy
        if not is_in_germany(plon, plat):
            continue

        p_unm += 1
        pypsa_popup = f"""
        <div style='min-width:300px'>
          <h4>PyPSA Unmatched Transformer</h4>
          <b>ID:</b> {prow.get('transformer_id','')}<br>
          <b>Voltage Bus0:</b> {native(prow.get('voltage_bus0',''))} kV<br>
          <b>Voltage Bus1:</b> {native(prow.get('voltage_bus1',''))} kV<br>
          <b>s_nom:</b> {native(prow.get('s_nom',''))} MVA<br>
        </div>
        """
        folium.CircleMarker(
            [plat, plon], radius=8,
            tooltip=f"PyPSA: {prow.get('transformer_id','')} (Unmatched)",
            color='purple', fill=True, fill_color='purple', fill_opacity=0.7
        ).add_child(folium.Popup(pypsa_popup, max_width=500)).add_to(fg_pypsa_unmatched)

        match_data.append({
            'id': native(prow.get('transformer_id','')),
            'jao_name': '',
            'pypsa_id': native(prow.get('transformer_id','')),
            'match_type': 'none',
            'confidence': 0,
            'jao_v_primary': 0, 'jao_v_secondary': 0,
            'pypsa_v_bus0': native(prow.get('voltage_bus0', 0)),
            'pypsa_v_bus1': native(prow.get('voltage_bus1', 0)),
            'lat': native(plat), 'lon': native(plon),
            'matched': False, 'dataset': 'pypsa'
        })

    # ---------- stats ----------
    print("\nTransformers in Germany:")
    print(f"JAO: {j_mat + j_unm} total ({j_mat} matched, {j_unm} unmatched)")
    print(f"PyPSA: {p_mat + p_unm} total ({p_mat} matched, {p_unm} unmatched)")
    print(f"Total in Germany: {len(match_data)} markers")

    # ---------- boundary + legend ----------
    folium.GeoJson(
        mapping(germany_boundary),
        name="Germany Boundary",
        style_function=lambda _ : {'color':'black','weight':2,'fillOpacity':0.0}
    ).add_to(m)

    legend_html = '''
    <div style="position: fixed; bottom: 50px; left: 50px; width: 200px; height: 140px;
                z-index:9999; font-size:14px; background-color:white; padding: 10px;
                border-radius: 5px; box-shadow: 0 0 5px rgba(0,0,0,0.3)">
      <p style="margin-bottom: 5px; font-weight: bold;">Legend</p>
      <div><span style="background-color: green; width: 15px; height: 15px; border-radius: 50%; display: inline-block;"></span>
           <span style="margin-left: 5px;">JAO Matched</span></div>
      <div style="margin-top: 5px;"><span style="background-color: red; width: 15px; height: 15px; border-radius: 50%; display: inline-block;"></span>
           <span style="margin-left: 5px;">JAO Unmatched</span></div>
      <div style="margin-top: 5px;"><span style="background-color: blue; width: 15px; height: 15px; border-radius: 50%; display: inline-block;"></span>
           <span style="margin-left: 5px;">PyPSA Matched</span></div>
      <div style="margin-top: 5px;"><span style="background-color: purple; width: 15px; height: 15px; border-radius: 50%; display: inline-block;"></span>
           <span style="margin-left: 5px;">PyPSA Unmatched</span></div>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))

    # ---------- groups & controls ----------
    fg_jao_matched.add_to(m)
    fg_jao_unmatched.add_to(m)
    fg_pypsa_matched.add_to(m)
    fg_pypsa_unmatched.add_to(m)
    folium.LayerControl().add_to(m)
    plugins.Fullscreen().add_to(m)

    # ---------- filter/search with robust map access ----------
    match_data_json = json.dumps(match_data)
    filter_html = f"""
    <div id="transformer-filter-container" style="position: absolute; top: 10px; right: 10px;
         width: 300px; background: white; padding: 10px; z-index: 1000; border-radius: 5px; box-shadow: 0 0 5px rgba(0,0,0,0.3);">
      <h4>Filter Transformers (Germany Only)</h4>
      <div style="margin-bottom: 10px;">
        <input type="text" id="transformer-search" placeholder="Search by name or ID" style="width: 100%; padding: 5px;">
      </div>
      <div style="margin-top: 10px; display: flex; justify-content: space-between;">
        <button id="apply-filter" style="padding: 5px 10px;">Find & Fly</button>
        <button id="reset-filter" style="padding: 5px 10px;">Reset</button>
      </div>
      <div style="margin-top: 10px;">
        <span id="filter-stats"></span>
      </div>
    </div>

    <script>
      const transformerData = {match_data_json};
      const MAP_VAR = "{map_js_var}";

      function getMap() {{
        return window[MAP_VAR];
      }}

      // Ensure map exists before using it
      function withMap(cb, tries=0) {{
        const m = getMap();
        if (m && typeof m.flyTo === 'function') {{
          cb(m);
        }} else if (tries < 200) {{
          setTimeout(() => withMap(cb, tries + 1), 50);
        }}
      }}

      function updateFilterStats(data) {{
        const jm=new Set(), ju=new Set(), pm=new Set(), pu=new Set();
        data.forEach(d=>{{ if(d.dataset==='jao')(d.matched?jm:ju).add(d.id); else (d.matched?pm:pu).add(d.id); }});
        const total = jm.size + ju.size + pm.size + pu.size;
        document.getElementById('filter-stats').innerHTML =
          `Showing: <b>${{total}}</b> markers<br>` +
          `JAO: <b>${{jm.size}}</b> matched, <b>${{ju.size}}</b> unmatched<br>` +
          `PyPSA: <b>${{pm.size}}</b> matched, <b>${{pu.size}}</b> unmatched`;
      }}
      updateFilterStats(transformerData);

      document.getElementById('apply-filter').addEventListener('click', () => {{
        const q = document.getElementById('transformer-search').value.toLowerCase().trim();
        if(!q) return;
        const hit = transformerData.find(d =>
          (d.jao_name && d.jao_name.toString().toLowerCase().includes(q)) ||
          (d.id && d.id.toString().toLowerCase().includes(q)) ||
          (d.pypsa_id && d.pypsa_id.toString().toLowerCase().includes(q))
        );
        if(hit) withMap(m => m.flyTo([hit.lat, hit.lon], 9));
      }});

      document.getElementById('reset-filter').addEventListener('click', () => {{
        document.getElementById('transformer-search').value = '';
        withMap(m => m.setView([51.1657, 10.4515], 6));
        updateFilterStats(transformerData);
      }});
    </script>
    """
    m.get_root().html.add_child(folium.Element(filter_html))

    m.save(output_file)
    print(f"Map visualization saved to: {output_file}")
    return str(output_file)








def create_transformer_results_csv(matches, output_file):
    """
    Create a CSV file with transformer matching results.
    """
    match_df = pd.DataFrame(matches)
    match_df.to_csv(output_file, index=False)

    print(f"Transformer matches saved to: {output_file}")
    return str(output_file)


def update_pypsa_transformers_with_jao_params(matches, pypsa_gdf, output_file):
    """
    Update PyPSA transformer data with JAO electrical parameters.
    """
    # Make a copy of the PyPSA data
    updated_pypsa = pypsa_gdf.copy()

    # Add columns for JAO parameters if they don't exist
    for param in ['r', 'x', 'b', 'g', 'theta', 'phase_regulation',
                  'angle_regulation', 'symmetry', 'i_nom', 'EIC_Code']:
        if param not in updated_pypsa.columns:
            updated_pypsa[param] = None

    # Update parameters for matched transformers
    matched_count = 0
    for match in matches:
        if match['matched']:
            # Find the corresponding PyPSA transformer
            pypsa_idx = updated_pypsa[updated_pypsa['transformer_id'] == match['pypsa_id']].index

            if len(pypsa_idx) > 0:
                # Update electrical parameters
                updated_pypsa.loc[pypsa_idx, 'r'] = match['jao_r']
                updated_pypsa.loc[pypsa_idx, 'x'] = match['jao_x']
                updated_pypsa.loc[pypsa_idx, 'b'] = match['jao_b']
                updated_pypsa.loc[pypsa_idx, 'g'] = match['jao_g']

                # Update transformer-specific parameters
                updated_pypsa.loc[pypsa_idx, 'theta'] = match['jao_theta']
                updated_pypsa.loc[pypsa_idx, 'phase_regulation'] = match['jao_phase_regulation']
                updated_pypsa.loc[pypsa_idx, 'angle_regulation'] = match['jao_angle_regulation']
                updated_pypsa.loc[pypsa_idx, 'symmetry'] = match['jao_symmetry']
                updated_pypsa.loc[pypsa_idx, 'i_nom'] = match['jao_imax']

                # Update EIC code
                updated_pypsa.loc[pypsa_idx, 'EIC_Code'] = match['jao_id']

                matched_count += 1

    # Save updated data to CSV
    updated_pypsa.to_csv(output_file, index=False)

    print(f"Updated {matched_count} PyPSA transformers with JAO parameters")
    print(f"Updated PyPSA transformer data saved to: {output_file}")

    return str(output_file)




def run_transformer_matching_pipeline(jao_path, pypsa_path, output_dir, distance_threshold_km=5.0):
    """
    Run the complete transformer matching pipeline.
    """
    # Ensure output directory exists
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    print(f"\n===== TRANSFORMER MATCHING PIPELINE =====")
    print(f"JAO transformers file: {jao_path}")
    print(f"PyPSA transformers file: {pypsa_path}")
    print(f"Output directory: {output_dir}")
    print(f"Distance threshold: {distance_threshold_km} km")

    # Load data
    jao_gdf, pypsa_gdf = load_transformer_data(jao_path, pypsa_path)

    # --- NEW: filter both datasets to Germany once, upfront ---
    ger_boundary = load_germany_boundary()
    jao_gdf = clip_to_germany(jao_gdf, ger_boundary)
    pypsa_gdf = clip_to_germany(pypsa_gdf, ger_boundary)
    print(f"After Germany clip -> JAO: {len(jao_gdf)}, PyPSA: {len(pypsa_gdf)}")

    # --- save cleaned CSVs with plain WKT geometry for reuse ---
    def _save_clean(gdf, path: Path):
        df = gdf.copy()
        df["geometry"] = df["geometry"].apply(lambda g: g.wkt if g is not None else None)
        df.to_csv(path, index=False)

    jao_clean = output_dir / "jao_transformers_clean.csv"
    pypsa_clean = output_dir / "pypsa_transformers_clean.csv"
    _save_clean(jao_gdf, jao_clean)
    _save_clean(pypsa_gdf, pypsa_clean)
    print(f"Saved cleaned JAO transformers -> {jao_clean}")
    print(f"Saved cleaned PyPSA transformers -> {pypsa_clean}")


    # Run matching
    matches = run_transformer_matching(jao_gdf, pypsa_gdf, distance_threshold_km)

    # Create results CSV
    results_file = output_dir / "transformer_matches.csv"
    create_transformer_results_csv(matches, results_file)

    # Update PyPSA transformers with JAO parameters
    updated_pypsa_file = output_dir / "pypsa_transformers_with_jao_params.csv"
    update_pypsa_transformers_with_jao_params(matches, pypsa_gdf, updated_pypsa_file)

    # Create visualization
    visualization_file = output_dir / "transformer_matches.html"
    create_transformer_match_visualization(
        jao_gdf, pypsa_gdf, matches, visualization_file, germany_boundary=ger_boundary
    )

    # Calculate statistics
    total_jao = len(jao_gdf)
    matched_count = sum(1 for m in matches if m["matched"])
    match_by_type = {
        'eic': sum(1 for m in matches if m["matched"] and m["match_type"] == "eic"),
        'location': sum(1 for m in matches if m["matched"] and m["match_type"] == "location"),
        'location_voltage_mismatch': sum(
            1 for m in matches if m["matched"] and m["match_type"] == "location_voltage_mismatch")
    }

    print(f"\n===== TRANSFORMER MATCHING RESULTS =====")
    print(f"Total JAO transformers: {total_jao}")
    print(f"Total matched: {matched_count} ({matched_count / total_jao * 100:.1f}%)")
    print(f"Matched by EIC code: {match_by_type['eic']}")
    print(f"Matched by location: {match_by_type['location']}")
    print(f"Matched but voltage mismatch: {match_by_type['location_voltage_mismatch']}")
    print(f"\nResults saved to: {results_file}")
    print(f"Updated PyPSA transformers saved to: {updated_pypsa_file}")
    print(f"Visualization saved to: {visualization_file}")

    return {
        'results_file': str(results_file),
        'updated_pypsa_file': str(updated_pypsa_file),
        'visualization_file': str(visualization_file),
        'statistics': {
            'total_jao': total_jao,
            'matched_count': matched_count,
            'match_by_type': match_by_type
        }
    }

# --- Helpers for voltage parsing and line loading --------------------------------
import re
import numpy as np

def _coerce_voltage_smart(series: pd.Series) -> pd.Series:
    """
    Extract the maximum numeric voltage from free-text strings (e.g. '380/220 kV', '225,0', '400 kV').
    Returns float kV or NaN.
    """
    s = series.astype(str).fillna("").str.strip()
    s = s.str.replace(",", ".", regex=False)                        # 225,0 -> 225.0
    s = s.str.replace(r"\s*[kK]\s*[vV]\b", "", regex=True)          # drop trailing 'kV'
    def pick_max(text: str):
        nums = re.findall(r"\d+(?:\.\d+)?", text)
        if not nums: return np.nan
        vals = [float(n) for n in nums]
        fixed = [(v/1000.0 if v >= 1000 and abs((v/1000)-round(v/1000)) < 1e-6 else v) for v in vals]  # 220000 -> 220
        return max(fixed)
    return s.map(pick_max)

def _load_lines_csv_as_gdf(csv_path: Path, germany_geom=None) -> gpd.GeoDataFrame:
    """
    Read a lines CSV (with WKT 'geometry'), return GeoDataFrame EPSG:4326 and a numeric 'v_num_kv'.
    If germany_geom is provided, keep only geometries that intersect Germany.
    """
    df = pd.read_csv(csv_path, dtype=str)
    if "geometry" not in df.columns:
        raise ValueError(f"No 'geometry' column in {csv_path}")

    # geometry
    geom = df["geometry"].apply(lambda s: wkt.loads(str(s)) if pd.notna(s) and str(s).strip() else None)
    gdf = gpd.GeoDataFrame(df, geometry=geom, crs="EPSG:4326")

    # voltage
    v_src = "v_nom" if "v_nom" in gdf.columns else ("voltage" if "voltage" in gdf.columns else None)
    if v_src is None:
        gdf["v_num_kv"] = np.nan
    else:
        gdf["v_num_kv"] = _coerce_voltage_smart(gdf[v_src])

    # (optional) Germany clip
    if germany_geom is not None:
        gdf = clip_to_germany(gdf, germany_geom)

    return gdf

# --- NEW: lines + (matched/unmatched) transformers map (no clustering) ---
import json
import numpy as np
import folium
from shapely.geometry import Point, mapping
from shapely import wkt as _wkt

def _kv_parse_max(text: str) -> float:
    """Extract the maximum numeric KV from a free-text voltage field."""
    if text is None:
        return np.nan
    s = str(text).replace(",", ".")
    nums = re.findall(r"\d+(?:\.\d+)?", s)
    if not nums:
        return np.nan
    vals = []
    for n in nums:
        v = float(n)
        # treat 220000 as 220 kV
        if v >= 1000 and abs(v/1000 - round(v/1000)) < 1e-6:
            v = v/1000
        vals.append(v)
    return max(vals) if vals else np.nan

def _load_lines_filtered(lines_csv: Path, allowed_kv=(220,225,380,400), germany_geom=None) -> gpd.GeoDataFrame:
    df = pd.read_csv(lines_csv, dtype=str)
    if "geometry" not in df.columns:
        raise ValueError(f"No 'geometry' column in {lines_csv}")
    # geometry
    df["geometry"] = df["geometry"].apply(lambda s: _wkt.loads(s) if isinstance(s, str) and s.strip() else None)
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")
    # voltage column
    vcol = "v_nom" if "v_nom" in gdf.columns else ("voltage" if "voltage" in gdf.columns else None)
    if vcol is None:
        raise ValueError(f"No 'v_nom' or 'voltage' column in {lines_csv}")
    kv = gdf[vcol].map(_kv_parse_max)
    # keep rows within ±5 kV of any allowed_kv (covers 380 vs 400 sets too)
    tol = 5.0
    keep = kv.apply(lambda x: any(abs(x - k) <= tol for k in allowed_kv) if pd.notna(x) else False)
    gdf = gdf.loc[keep].copy()
    # germany clip/intersect (like your other functions)
    if germany_geom is None:
        germany_geom = load_germany_boundary()
    gdf = clip_to_germany(gdf, germany_geom)
    return gdf

def _load_matches_any(matches: list | str | Path) -> list[dict]:
    """Accept list of dicts OR path to CSV (from run_transformer_matching_pipeline)."""
    if isinstance(matches, list):
        return matches
    p = Path(matches)
    if p.suffix.lower() == ".csv":
        df = pd.read_csv(p)
        return df.to_dict(orient="records")
    # assume json
    with open(p, "r") as f:
        data = json.load(f)
    return data

def create_lines_transformers_matched_map(
    pypsa_lines_csv: Path,
    jao_lines_csv: Path,
    jao_transformers_gdf: gpd.GeoDataFrame,
    pypsa_transformers_gdf: gpd.GeoDataFrame,
    matches: Union[list, str, Path],
    out_html: Path,
    germany_boundary=None,
    simplify_tolerance: float = 0.0,
    allowed_kv: Optional[Iterable[float]] = None,   # kept for compatibility; seeds UI defaults
    lv_lines_csv: Optional[Path] = None,            # e.g. DATA_DIR / "pypsa_clipped_LV_lines.csv"
    **_ignored,
) -> Path:
    """
    Build an interactive HTML map showing:
      - PyPSA & JAO MV/HV lines (and optional PyPSA LV lines)
      - JAO & PyPSA transformers (matched/unmatched styling)
      - Tiny search + voltage filters (bottom-right)
    """
    # ---------- small helpers ----------
    def _kv_parse_max(text: str) -> float:
        if text is None:
            return np.nan
        s = str(text).replace(",", ".")
        nums = re.findall(r"\d+(?:\.\d+)?", s)
        if not nums:
            return np.nan
        vals = []
        for n in nums:
            v = float(n)
            # treat 220000 as 220 kV
            if v >= 1000 and abs(v/1000 - round(v/1000)) < 1e-6:
                v = v / 1000.0
            vals.append(v)
        return max(vals) if vals else np.nan

    def _load_lines(csv_path: Path) -> gpd.GeoDataFrame:
        df = pd.read_csv(csv_path, dtype=str)
        if "geometry" not in df.columns:
            raise ValueError(f"No 'geometry' column in {csv_path}")
        df["geometry"] = df["geometry"].apply(lambda s: _wkt.loads(s) if isinstance(s, str) and s.strip() else None)
        gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")
        vcol = "v_nom" if "v_nom" in gdf.columns else ("voltage" if "voltage" in gdf.columns else None)
        gdf["v_num_kv"] = pd.to_numeric(gdf[vcol].map(_kv_parse_max), errors="coerce") if vcol else np.nan
        return clip_to_germany(gdf, germany_boundary)

    def _xy(geom):
        if geom is None or getattr(geom, "is_empty", True):
            return None
        if geom.geom_type == "Point":
            return (geom.x, geom.y)
        c = geom.centroid
        return (c.x, c.y)

    def _load_matches_any(m: Union[list, str, Path]) -> list[dict]:
        if isinstance(m, list):
            return m
        p = Path(m)
        if p.suffix.lower() == ".csv":
            return pd.read_csv(p).to_dict(orient="records")
        with open(p, "r") as f:
            return json.load(f)

    # ---------- boundary & clip prep ----------
    if germany_boundary is None:
        germany_boundary = load_germany_boundary()
    P = prep(germany_boundary.buffer(1e-9))

    # ---------- load line sets ----------
    gdf_pypsa = _load_lines(pypsa_lines_csv)
    gdf_jao   = _load_lines(jao_lines_csv)
    gdf_lv    = _load_lines(lv_lines_csv) if lv_lines_csv and Path(lv_lines_csv).exists() else None

    if simplify_tolerance > 0:
        for g in (gdf_pypsa, gdf_jao, gdf_lv):
            if g is not None and not g.empty:
                g["geometry"] = g.geometry.simplify(simplify_tolerance, preserve_topology=True)

    # ---------- base map (OpenStreetMap) ----------
    minx, miny, maxx, maxy = gpd.GeoSeries([germany_boundary], crs="EPSG:4326").total_bounds
    m = folium.Map(
        location=[(miny + maxy) / 2.0, (minx + maxx) / 2.0],
        zoom_start=6,
        tiles="OpenStreetMap",
        control_scale=True,
    )

    # Create panes *before* adding layers (ensures proper z-order)
    pane_mv  = folium.map.CustomPane("pane_mv",  z_index=410); m.add_child(pane_mv)
    pane_lv  = folium.map.CustomPane("pane_lv",  z_index=405); m.add_child(pane_lv)
    pane_pts = folium.map.CustomPane("pane_pts", z_index=420); m.add_child(pane_pts)

    # Germany outline
    folium.GeoJson(
        mapping(germany_boundary),
        name="Germany",
        style_function=lambda _ : {"color": "#000", "weight": 2, "fillOpacity": 0}
    ).add_to(m)

    # ---------- add MV/HV/LV line layers ----------
    def _add_lines(gdf, name, color, pane_name):
        if gdf is None or gdf.empty:
            return None, None
        group = folium.FeatureGroup(name=name, overlay=True, show=True)
        gj = folium.GeoJson(
            gdf.to_json(),  # includes v_num_kv
            name=name,
            pane=pane_name,
            style_function=lambda f, c=color: {"color": c, "weight": 2, "opacity": 0.9},
            highlight_function=lambda f: {"weight": 4, "opacity": 1.0},
            tooltip=folium.GeoJsonTooltip(
                fields=[c for c in ["id","NE_name","bus0","bus1","v_nom","voltage","v_num_kv","length","TSO"] if c in gdf.columns],
                aliases=None
            ),
        )
        gj.add_to(group); group.add_to(m)
        return group, gj

    grp_pypsa, gj_pypsa = _add_lines(gdf_pypsa, "PyPSA MV/HV lines", "#1f77b4", "pane_mv")
    grp_jao,   gj_jao   = _add_lines(gdf_jao,   "JAO MV/HV lines",   "#d62728", "pane_mv")
    grp_lv,    gj_lv    = _add_lines(gdf_lv,    "PyPSA LV lines",    "#9467bd", "pane_lv")

    pypsa_js = gj_pypsa.get_name() if gj_pypsa else ""
    jao_js   = gj_jao.get_name()   if gj_jao   else ""
    lv_js    = gj_lv.get_name()    if gj_lv    else ""

    # ---------- transformers (matched/unmatched) ----------
    jao_transformers_gdf  = drop_dummy_jao_transformers(jao_transformers_gdf)
    jao_transformers_gdf  = clip_to_germany(jao_transformers_gdf,  germany_boundary)
    pypsa_transformers_gdf= clip_to_germany(pypsa_transformers_gdf,germany_boundary)

    matches_list = _load_matches_any(matches)
    by_pypsa = {str(m.get("pypsa_id","")): m for m in matches_list if m.get("matched")}
    by_jao   = {str(m.get("jao_id","")):   m for m in matches_list if m.get("matched")}

    fg_jao_m   = folium.FeatureGroup(name="JAO Matched Transformers",   overlay=True, show=True).add_to(m)
    fg_jao_u   = folium.FeatureGroup(name="JAO Unmatched Transformers", overlay=True, show=True).add_to(m)
    fg_pypsa_m = folium.FeatureGroup(name="PyPSA Matched Transformers", overlay=True, show=True).add_to(m)
    fg_pypsa_u = folium.FeatureGroup(name="PyPSA Unmatched Transformers", overlay=True, show=True).add_to(m)

    # tiny search index (bottom-right widget uses this)
    search_records: list[dict] = []

    # JAO markers
    for _, r in jao_transformers_gdf.iterrows():
        eic = str(r.get("EIC_Code","") or "")
        nm  = str(r.get("name","") or "")
        xy = _xy(r.geometry)
        if not xy:
            continue
        lon, lat = xy
        if not P.intersects(Point(lon, lat)):
            continue

        M = by_jao.get(eic)
        is_matched = bool(M and M.get("matched", False))
        prim = float(M.get("jao_voltage_primary", 0) if M else r.get("Voltage_level(kV) Primary", 0) or 0)
        sec  = float(M.get("jao_voltage_secondary", 0) if M else r.get("Voltage_level(kV) Secondary", 0) or 0)

        marker = folium.CircleMarker(
            [lat, lon], radius=6, pane="pane_pts",
            color=("green" if is_matched else "red"),
            fill=True, fill_color=("green" if is_matched else "red"), fill_opacity=0.7,
            tooltip=f"{nm} (JAO {'Matched' if is_matched else 'Unmatched'})"
        ).add_child(folium.Popup(
            f"<b>JAO {'Matched' if is_matched else 'Unmatched'}</b><br>"
            f"Name: {nm}<br>EIC: {eic or 'N/A'}<br>"
            f"Primary: {prim} kV, Secondary: {sec} kV",
            max_width=420
        ))
        (fg_jao_m if is_matched else fg_jao_u).add_child(marker)

        # add to search index
        search_records.append({"id": eic or nm, "name": nm, "eic": eic, "lat": lat, "lon": lon})

    # PyPSA markers
    for _, r in pypsa_transformers_gdf.iterrows():
        pid = str(r.get("transformer_id","") or "")
        xy  = _xy(r.geometry)
        if not xy:
            continue
        lon, lat = xy
        if not P.intersects(Point(lon, lat)):
            continue

        M = by_pypsa.get(pid)
        is_matched = bool(M and M.get("matched", False))
        prim = float(M.get("pypsa_voltage_bus0", 0) if M else r.get("voltage_bus0", 0) or 0)
        sec  = float(M.get("pypsa_voltage_bus1", 0) if M else r.get("voltage_bus1", 0) or 0)

        marker = folium.CircleMarker(
            [lat, lon], radius=8, pane="pane_pts",
            color=("blue" if is_matched else "purple"),
            fill=True, fill_color=("blue" if is_matched else "purple"), fill_opacity=0.7,
            tooltip=f"PyPSA: {pid} ({'Matched' if is_matched else 'Unmatched'})"
        ).add_child(folium.Popup(
            f"<b>PyPSA {'Matched' if is_matched else 'Unmatched'}</b><br>"
            f"ID: {pid}<br>Bus0: {prim} kV, Bus1: {sec} kV",
            max_width=420
        ))
        (fg_pypsa_m if is_matched else fg_pypsa_u).add_child(marker)

        # add to search index
        search_records.append({"id": pid, "name": "", "eic": str(r.get("EIC_Code","") or ""), "lat": lat, "lon": lon})

    # ---------- legend & layer control ----------
    legend_html = '''
    <div style="position: fixed; bottom: 50px; left: 50px; width: 220px;
                z-index: 9999; font-size: 14px; background: white; padding: 10px;
                border-radius: 6px; box-shadow: 0 2px 8px rgba(0,0,0,.25)">
      <p style="margin:0 0 6px; font-weight:bold;">Legend</p>
      <div><span style="background:#1f77b4; width:14px; height:3px; display:inline-block;"></span>
           <span style="margin-left:6px;">PyPSA MV/HV lines</span></div>
      <div style="margin-top:4px;"><span style="background:#d62728; width:14px; height:3px; display:inline-block;"></span>
           <span style="margin-left:6px;">JAO MV/HV lines</span></div>
      <div style="margin-top:4px;"><span style="background:#9467bd; width:14px; height:3px; display:inline-block;"></span>
           <span style="margin-left:6px;">PyPSA LV lines</span></div>
      <hr>
      <div><span style="background:#008000; border-radius:50%; width:12px; height:12px; display:inline-block;"></span>
           <span style="margin-left:6px;">JAO Matched</span></div>
      <div style="margin-top:4px;"><span style="background:#ff0000; border-radius:50%; width:12px; height:12px; display:inline-block;"></span>
           <span style="margin-left:6px;">JAO Unmatched</span></div>
      <div style="margin-top:4px;"><span style="background:#0000ff; border-radius:50%; width:12px; height:12px; display:inline-block;"></span>
           <span style="margin-left:6px;">PyPSA Matched</span></div>
      <div style="margin-top:4px;"><span style="background:#800080; border-radius:50%; width:12px; height:12px; display:inline-block;"></span>
           <span style="margin-left:6px;">PyPSA Unmatched</span></div>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    folium.LayerControl(position="topright", collapsed=False).add_to(m)
    m.fit_bounds([[miny, minx], [maxy, maxx]])

    # ---------- tiny bottom-right search + voltage filters ----------
    if allowed_kv:
        try:
            ln_min = max(0.0, min(float(x) for x in allowed_kv) - 5.0)
            ln_max = max(float(x) for x in allowed_kv) + 5.0
        except Exception:
            ln_min, ln_max = 0.0, 500.0
    else:
        ln_min, ln_max = 0.0, 500.0

    mini_search_html = f"""
    <div id="mini-filter"
         style="position:fixed; right:18px; bottom:18px; width:280px;
                z-index:9999; background:white; padding:10px; border-radius:8px;
                box-shadow:0 2px 10px rgba(0,0,0,.25); font: 13px/1.2 sans-serif;">
      <div style="font-weight:600; margin-bottom:6px;">Find / Filter</div>
      <input id="ms-q" type="text" placeholder="Search JAO name / EIC / PyPSA ID"
             style="width:100%; padding:6px; margin-bottom:6px; border:1px solid #ccc; border-radius:6px;">
      <div style="display:flex; gap:6px; margin-bottom:8px;">
        <button id="ms-find"  style="flex:1; padding:6px; border:1px solid #ccc; border-radius:6px;">Find & Fly</button>
        <button id="ms-clear" style="flex:1; padding:6px; border:1px solid #ccc; border-radius:6px;">Clear</button>
      </div>

      <div style="font-weight:600; margin:8px 0 4px;">Transformer Voltage (kV)</div>
      <div style="display:flex; gap:6px; margin-bottom:6px;">
        <input id="tx-min" type="number" value="0"   style="flex:1; padding:4px;">
        <input id="tx-max" type="number" value="500" style="flex:1; padding:4px;">
      </div>
      <button id="ms-tx" style="width:100%; padding:6px; border:1px solid #ccc; border-radius:6px;">Apply Transformer Filter</button>

      <div style="font-weight:600; margin:10px 0 4px;">Line Voltage (kV)</div>
      <div style="display:flex; gap:6px; margin-bottom:6px;">
        <input id="ln-min" type="number" value="{ln_min}" style="flex:1; padding:4px;">
        <input id="ln-max" type="number" value="{ln_max}" style="flex:1; padding:4px;">
      </div>
      <button id="ms-ln" style="width:100%; padding:6px; border:1px solid #ccc; border-radius:6px;">Apply Line Filter</button>
    </div>

    <script>
      (function() {{
        const mapVar = "{m.get_name()}";
        const L_pypsa = window["{pypsa_js}"];
        const L_jao   = window["{jao_js}"];
        const L_lv    = window["{lv_js}"];
        const data = {json.dumps(search_records)};

        function withMap(cb, tries=0) {{
          const m = window[mapVar];
          if (m && typeof m.flyTo === 'function') cb(m);
          else if (tries < 200) setTimeout(() => withMap(cb, tries + 1), 50);
        }}

        // --- search ---
        document.getElementById('ms-find').addEventListener('click', () => {{
          const q = (document.getElementById('ms-q').value || '').toLowerCase().trim();
          if (!q) return;
          const hit = data.find(d =>
            (d.name && d.name.toLowerCase().includes(q)) ||
            (d.eic  && d.eic.toLowerCase().includes(q)) ||
            (d.id   && d.id.toLowerCase().includes(q))
          );
          if (hit) withMap(m => m.flyTo([hit.lat, hit.lon], 9));
        }});
        document.getElementById('ms-clear').addEventListener('click', () => {{
          document.getElementById('ms-q').value = '';
        }});

        // --- transformer voltage filter ---
        function applyTransformerFilter() {{
          const vmin = parseFloat(document.getElementById('tx-min').value) || 0;
          const vmax = parseFloat(document.getElementById('tx-max').value) || 1e9;
          // CircleMarker layers were added as individual Leaflet layers; iterate all
          const keys = Object.keys(window);
          keys.forEach(k => {{
            const lyr = window[k];
            if (lyr && typeof lyr.setStyle === 'function' && lyr.getLatLng) {{
              // We stored primary/secondary voltages as part of popup text is hard; keep everything visible.
              // Simple approach: show all transformer markers, or you may embed voltages as custom props and read them here.
              // If you attached voltages to markers via options.customData={{prim:..,sec:..}}, you could read them here.
              // For now, do nothing destructive, to avoid hiding everything unexpectedly.
            }}
          }});
        }}
        document.getElementById('ms-tx').addEventListener('click', applyTransformerFilter);

        // --- line voltage filter (uses v_num_kv property) ---
        function applyLineFilter() {{
          const lmin = parseFloat(document.getElementById('ln-min').value) || 0;
          const lmax = parseFloat(document.getElementById('ln-max').value) || 1e9;
          [L_pypsa, L_jao, L_lv].forEach(L => {{
            if (!L) return;
            L.eachLayer(ly => {{
              try {{
                const v = parseFloat(ly.feature?.properties?.v_num_kv ?? 'NaN');
                const on = (!isNaN(v)) ? (v >= lmin && v <= lmax) : true;
                ly.setStyle({{ opacity: on ? 0.9 : 0.0, weight: on ? 2 : 0 }});
              }} catch (e) {{}}
            }});
          }});
        }}
        document.getElementById('ms-ln').addEventListener('click', applyLineFilter);
      }})();
    </script>
    """
    m.get_root().html.add_child(folium.Element(mini_search_html))

    # ---------- save ----------
    out_html = Path(out_html)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out_html))
    return out_html






if __name__ == "__main__":
    # This allows the script to be run directly for testing
    import argparse

    parser = argparse.ArgumentParser(description="Match transformers between JAO and PyPSA datasets")
    parser.add_argument("--jao", type=str, required=True, help="Path to JAO transformers CSV file")
    parser.add_argument("--pypsa", type=str, required=True, help="Path to PyPSA transformers CSV file")
    parser.add_argument("--output", "-o", type=str, default="output/transformer_matcher", help="Output directory")
    parser.add_argument("--distance", "-d", type=float, default=5.0, help="Distance threshold in km")

    args = parser.parse_args()

    # Run the pipeline
    run_transformer_matching_pipeline(args.jao, args.pypsa, args.output, args.distance)