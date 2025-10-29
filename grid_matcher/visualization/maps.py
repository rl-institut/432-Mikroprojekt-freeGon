import pandas as pd
from folium import folium
from folium.plugins import MeasureControl


def create_jao_pypsa_visualization(source_gdf, pypsa_gdf, results, map_file):
    # Add this debugging code at the beginning
    print(f"Visualization received {len(results)} total results")
    matched_count = sum(1 for r in results if r.get("matched", False))
    print(f"Results contain {matched_count} matched lines")

    # Print a sample match to check format
    if matched_count > 0:
        sample = next(r for r in results if r.get("matched", False))
        print(f"Sample match format: {sample}")

    # Rename variables to match the rest of the function
    jao_gdf = source_gdf
    matching_results = results
    output_path = map_file

    print(f"Creating visualization at {output_path}...")
    """
    Create an interactive Leaflet visualization comparing JAO and PyPSA grid lines,
    including rich filters, search, stats, and parameter tables.

    Parameters
    ----------
    jao_gdf : GeoDataFrame
        JAO lines. Geometry must be in lon/lat (EPSG:4326) or convertible to it.
        Length columns may be `length_km` (km) or `length` (km or m).
    pypsa_gdf : GeoDataFrame
        PyPSA lines. Geometry must be in lon/lat (EPSG:4326) or convertible to it.
        Length column typically `length` in meters.
    matching_results : list[dict]
        Results of the matching step. Each dict may include:
        - 'matched' (bool)
        - 'jao_id' (str/int)
        - 'pypsa_ids' (list[str] | str with ;-sep)
        - flags: 'is_duplicate', 'is_geometric_match', 'is_parallel_circuit',
        'is_parallel_voltage_circuit'
        - 'match_quality' (str)
    output_path : str | pathlib.Path
        Path where the HTML will be written.

    Returns
    -------
    str
        Path to the generated HTML file (as a string).
    """
    import json
    import os
    from pathlib import Path

    # Optional GeoPandas / Shapely imports are kept inside in case the caller runs this in a slim env
    try:
        import geopandas as gpd  # noqa: F401
    except Exception:
        gpd = None
    try:
        import shapely.geometry as sgeom
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Shapely is required for geometry handling.") from e

    output_path = map_file  # Define output_path for consistency

    print(f"Creating visualization at {output_path}...")

    # -------------------------------
    # Helpers
    # -------------------------------
    def _safe_int(v, default=0):
        try:
            if v is None:
                return default
            return int(v)
        except Exception:
            try:
                return int(float(v))
            except Exception:
                return default

    def _safe_float(v, default=0.0):
        try:
            if v is None:
                return default
            return float(v)
        except Exception:
            return default

    def _to_km_from_unknown(x):
        """
        Convert length to km with a simple heuristic:
        - if >= 1000, assume meters
        - otherwise assume km
        """
        val = _safe_float(x, None)
        if val is None:
            return None
        return val / 1000.0 if abs(val) >= 1000.0 else val

    def _meters_to_km(m):
        val = _safe_float(m, None)
        if val is None:
            return None
        return val / 1000.0

    def _get_val(row, *names, default=None):
        for n in names:
            if (hasattr(row, "__contains__") and n in row) or hasattr(row, "get"):
                try:
                    v = row[n] if n in row else row.get(n, None)
                except Exception:
                    v = None
                if v is None:
                    continue
                # pandas NA safe-ish check
                try:
                    import pandas as pd  # local import
                    if pd.isna(v):
                        continue
                except Exception:
                    pass
                if isinstance(v, str) and v.strip() == "":
                    continue
                return v
        return default

    def _ensure_wgs84(gdf):
        """Try to reproject to EPSG:4326 if possible/needed."""
        try:
            if gdf is None:
                return gdf
            if hasattr(gdf, "crs") and gdf.crs:
                # already lon/lat?
                if getattr(gdf.crs, "to_epsg", lambda: None)() == 4326:
                    return gdf
                if hasattr(gdf, "to_crs"):
                    return gdf.to_crs(4326)
        except Exception:
            # if anything fails, just return as-is (better than crashing);
            # the map will render with whatever coordinates are present.
            return gdf
        return gdf

    def _explode_to_lines(geom):
        """
        Yield LineStrings for any geometry: LineString -> itself,
        MultiLineString -> parts, GeometryCollection -> lines within,
        Point/Polygon -> skip; for Polygon, use exterior as a fallback LineString.
        """
        if geom is None:
            return
        if isinstance(geom, sgeom.LineString):
            yield geom
        elif isinstance(geom, sgeom.MultiLineString):
            for part in geom.geoms:
                if isinstance(part, sgeom.LineString):
                    yield part
        elif isinstance(geom, sgeom.GeometryCollection):
            for part in geom.geoms:
                yield from _explode_to_lines(part)
        elif isinstance(geom, sgeom.Polygon):
            # fallback: exterior as line (rare, but avoids dropping)
            try:
                ext = geom.exterior
                if ext:
                    yield sgeom.LineString(ext.coords)
            except Exception:
                return
        # Points and other unsupported geometries are ignored

    def _coords(ls):
        # Return [ [x,y], ... ] safely
        try:
            return [[float(x), float(y)] for (x, y) in ls.coords]
        except Exception:
            return []

    # -------------------------------
    # Ensure lon/lat and prep data
    # -------------------------------
    jao_gdf = _ensure_wgs84(jao_gdf)
    pypsa_gdf = _ensure_wgs84(pypsa_gdf)

    # -------------------------------
    # Parse matching results -> sets/mappings
    # -------------------------------
    matched_jao_ids = set()
    geometric_match_jao_ids = set()
    parallel_circuit_jao_ids = set()
    parallel_voltage_jao_ids = set()
    duplicate_jao_ids = set()

    # {pypsa_id: {'jao_ids': set(), 'match_types': set(), 'quality': str}}
    matched_pypsa_info = {}

    for res in (matching_results or []):
        if not res or not res.get("matched", False):
            continue

        jao_id = str(res.get("jao_id", ""))
        pypsa_ids = res.get("pypsa_ids", [])
        if isinstance(pypsa_ids, str):
            pypsa_ids = [pid.strip() for pid in pypsa_ids.replace(",", ";").split(";") if pid.strip()]

        if jao_id:
            matched_jao_ids.add(jao_id)

        is_dup = res.get("is_duplicate", False)
        is_geom = res.get("is_geometric_match", False)
        is_par = res.get("is_parallel_circuit", False)
        is_parv = res.get("is_parallel_voltage_circuit", False)

        if is_dup:
            duplicate_jao_ids.add(jao_id)
        if is_geom:
            geometric_match_jao_ids.add(jao_id)
        if is_par:
            parallel_circuit_jao_ids.add(jao_id)
        if is_parv:
            parallel_voltage_jao_ids.add(jao_id)

        # Determine a single label for JS "status" styling
        mtype = (
            "duplicate" if is_dup else
            "geometric" if is_geom else
            "parallel" if is_par else
            "parallel_voltage" if is_parv else
            "matched"
        )

        for pid in pypsa_ids:
            ent = matched_pypsa_info.setdefault(str(pid), {"jao_ids": set(), "match_types": set(), "quality": None})
            ent["jao_ids"].add(jao_id)
            ent["match_types"].add(mtype)
            # prefer first non-empty quality string
            if ent["quality"] in (None, "", "Match"):
                ent["quality"] = res.get("match_quality", "Match") or "Match"

    # -------------------------------
    # Build GeoJSON Features
    # -------------------------------
    jao_features = []
    pypsa_features = []

    # Count features by type for logging
    dc_count = 0
    v110_count = 0
    v220_count = 0
    v400_count = 0

    # JAO features
    for _, row in jao_gdf.iterrows():
        jao_id = str(_get_val(row, "id", "jao_id", default=""))
        if not jao_id:
            continue

        name = str(_get_val(row, "NE_name", "name", default="") or "")
        v_nom = _get_val(row, "v_nom", "voltage", default=0)

        # Add DC detection
        is_dc = (isinstance(v_nom, str) and v_nom.upper() == "DC") or _get_val(row, "is_dc", "dc", default=False)

        # Update voltage classification to include DC and 110kV
        if is_dc:
            voltage_class = "DC"
            v_nom = "DC"  # Standardize the representation
        else:
            v_nom = _safe_int(v_nom, 0)
            if 100 <= v_nom < 200:
                voltage_class = "110kV"
            elif 200 <= v_nom < 300:
                voltage_class = "220kV"
            elif v_nom >= 300:
                voltage_class = "400kV"
            else:
                voltage_class = "other"

        # Preferred length column: length_km -> length (km or m)
        length_km = _get_val(row, "length_km", default=None)
        if length_km is not None:
            length_km = _safe_float(length_km, None)
        if length_km is None:
            # Try to infer from 'length'
            length_km = _to_km_from_unknown(_get_val(row, "length", "len", default=None)) or 0.0

        r_per_km = _safe_float(_get_val(row, "r_per_km", "r_km", "R_per_km", default=0))
        x_per_km = _safe_float(_get_val(row, "x_per_km", "x_km", "X_per_km", default=0))
        b_per_km = _safe_float(_get_val(row, "b_per_km", "b_km", "B_per_km", default=0))

        # Status/style
        if jao_id in duplicate_jao_ids:
            status = "duplicate"
            tooltip_status = "Duplicate"
        elif jao_id in parallel_circuit_jao_ids:
            status = "parallel"
            tooltip_status = "Parallel Circuit"
        elif jao_id in parallel_voltage_jao_ids:
            status = "parallel_voltage"
            tooltip_status = "Parallel Voltage"
        elif jao_id in geometric_match_jao_ids:
            status = "geometric"
            tooltip_status = "Geometric Match"
        elif jao_id in matched_jao_ids:
            status = "matched"
            tooltip_status = "Regular Match"
        else:
            status = "unmatched"
            tooltip_status = "Unmatched"

        # Geometry: split into individual LineStrings to avoid spurious connections
        parts = list(_explode_to_lines(row.geometry))
        if not parts:
            continue

        for idx_part, ls in enumerate(parts):
            fid = f"jao_{jao_id}" if idx_part == 0 else f"jao_{jao_id}_p{idx_part}"
            tooltip = f"JAO: {jao_id} - {name} ({v_nom} kV) - {length_km:.2f} km - {tooltip_status}"

            # Adjust tooltip for DC links
            if is_dc:
                tooltip = f"JAO: {jao_id} - {name} (DC) - {length_km:.2f} km - {tooltip_status}"

            jao_features.append({
                "type": "Feature",
                "id": fid,
                "properties": {
                    "type": "jao",
                    "id": jao_id,
                    "name": name,
                    "voltage": v_nom,
                    "voltageClass": voltage_class,
                    "is_dc": is_dc,
                    "status": status,
                    "matchStatus": "matched" if status != "unmatched" else "unmatched",
                    "tooltip": tooltip,
                    "length_km": float(length_km or 0.0),
                    "r_per_km": r_per_km,
                    "x_per_km": x_per_km,
                    "b_per_km": b_per_km,
                },
                "geometry": {"type": "LineString", "coordinates": _coords(ls)}
            })

    # After all JAO features are created
    print(f"Created {len(jao_features)} JAO features")
    if len(jao_features) > 0:
        print("Debug: First 5 JAO features status values:")
        for i, feature in enumerate(jao_features[:min(5, len(jao_features))]):
            print(
                f"  Feature {i}: id={feature['properties']['id']}, status={feature['properties']['status']}, matchStatus={feature['properties']['matchStatus']}")

    # PyPSA features
    for _, row in pypsa_gdf.iterrows():
        # id may be in 'line_id' or 'id'
        pypsa_id = str(_get_val(row, "line_id", "id", default=""))
        if not pypsa_id or row.geometry is None:
            continue

        v_nom = _get_val(row, "voltage", "v_nom", default=0)

        # Add DC detection
        is_dc = (isinstance(v_nom, str) and v_nom.upper() == "DC") or _get_val(row, "is_dc", "dc", default=False)

        # Update voltage classification to include DC and 110kV
        if is_dc:
            voltage_class = "DC"
            v_nom = "DC"  # Standardize the representation
            dc_count += 1
        else:
            v_nom = _safe_int(v_nom, 0)
            if 100 <= v_nom < 200:
                voltage_class = "110kV"
                v110_count += 1
            elif 200 <= v_nom < 300:
                voltage_class = "220kV"
                v220_count += 1
            elif v_nom >= 300:
                voltage_class = "400kV"
                v400_count += 1
            else:
                voltage_class = "other"

        length_m = _safe_float(_get_val(row, "length", default=0.0))
        length_km = _meters_to_km(length_m) or 0.0

        bus0 = str(_get_val(row, "bus0", default="") or "")
        bus1 = str(_get_val(row, "bus1", default="") or "")

        # Matched info
        m_info = matched_pypsa_info.get(pypsa_id, None)
        is_matched = m_info is not None
        match_status = "matched" if is_matched else "unmatched"
        matched_jao_list = sorted(list(m_info["jao_ids"])) if is_matched else []
        # choose one representative match type for styling; if multiple, prefer 'geometric'>'parallel'>'duplicate'>'matched'
        match_type_order = ["geometric", "parallel", "parallel_voltage", "duplicate", "matched"]
        mtype = match_status
        if is_matched and m_info["match_types"]:
            for t in match_type_order:
                if t in m_info["match_types"]:
                    mtype = t
                    break
        match_quality = (m_info["quality"] if is_matched else "N/A") or "N/A"

        # Params
        r_total = _safe_float(_get_val(row, "r", default=0.0))
        x_total = _safe_float(_get_val(row, "x", default=0.0))
        b_total = _safe_float(_get_val(row, "b", default=0.0))
        g_total = _safe_float(_get_val(row, "g", default=0.0))

        r_per_km = _safe_float(_get_val(row, "r_per_km", default=None))
        if r_per_km is None:
            r_per_km = (r_total / length_km) if length_km > 0 else 0.0
        x_per_km = _safe_float(_get_val(row, "x_per_km", default=None))
        if x_per_km is None:
            x_per_km = (x_total / length_km) if length_km > 0 else 0.0
        b_per_km = _safe_float(_get_val(row, "b_per_km", default=None))
        if b_per_km is None:
            b_per_km = (b_total / length_km) if length_km > 0 else 0.0
        g_per_km = _safe_float(_get_val(row, "g_per_km", default=None))
        if g_per_km is None:
            g_per_km = (g_total / length_km) if length_km > 0 else 0.0

        # Tooltip
        status_detail_map = {
            "geometric": " (Geometric Match)",
            "parallel": " (Parallel Circuit)",
            "parallel_voltage": " (Parallel Voltage)",
            "duplicate": " (Duplicate)",
            "matched": " (Regular Match)",
            "unmatched": ""
        }

        if is_dc:
            tooltip = f"PyPSA: {pypsa_id} (DC) - {length_km:.2f} km - {'Matched' if is_matched else 'Unmatched'}{status_detail_map.get(mtype, '')}"
        else:
            tooltip = f"PyPSA: {pypsa_id} ({v_nom} kV) - {length_km:.2f} km - {'Matched' if is_matched else 'Unmatched'}{status_detail_map.get(mtype, '')}"

        if bus0 or bus1:
            tooltip += f" - {bus0} to {bus1}"
        if matched_jao_list:
            tooltip += f" - Matched to JAO ID(s): {', '.join(matched_jao_list)}"

        # Geometry -> split multilines into individual features
        parts = list(_explode_to_lines(row.geometry))
        if not parts:
            continue
        for idx_part, ls in enumerate(parts):
            fid = f"pypsa_{pypsa_id}" if idx_part == 0 else f"pypsa_{pypsa_id}_p{idx_part}"
            pypsa_features.append({
                "type": "Feature",
                "id": fid,
                "properties": {
                    "type": "pypsa",
                    "id": pypsa_id,
                    "voltage": v_nom,
                    "voltageClass": voltage_class,
                    "is_dc": is_dc,
                    "matchStatus": match_status,
                    "status": mtype,
                    "matchQuality": match_quality,
                    "matchedJaoIds": matched_jao_list,
                    "tooltip": tooltip,
                    "length_km": float(length_km),
                    "bus0": bus0,
                    "bus1": bus1,
                    "r_ohm": r_total,
                    "x_ohm": x_total,
                    "b_siemens": b_total,
                    "g_siemens": g_total,
                    "r_per_km": r_per_km,
                    "x_per_km": x_per_km,
                    "b_per_km": b_per_km,
                    "g_per_km": g_per_km,
                },
                "geometry": {"type": "LineString", "coordinates": _coords(ls)}
            })

    # Print summary of loaded features
    print(f"Loaded {v400_count} 400kV lines")
    print(f"Loaded {v220_count} 220kV lines")
    print(f"Loaded {v110_count} 110kV lines")
    print(f"Loaded {dc_count} DC links")

    # Collections -> JSON strings for embedding
    jao_json = json.dumps({"type": "FeatureCollection", "features": jao_features}, ensure_ascii=False)
    pypsa_json = json.dumps({"type": "FeatureCollection", "features": pypsa_features}, ensure_ascii=False)

    # -------------------------------
    # Stats
    # -------------------------------
    total_jao = len(jao_gdf)
    total_pypsa = len(pypsa_gdf)

    jao_matched_cnt = len(matched_jao_ids)
    jao_geom_cnt = len(geometric_match_jao_ids)
    jao_parallel_cnt = len(parallel_circuit_jao_ids)
    jao_parvolt_cnt = len(parallel_voltage_jao_ids)
    jao_dup_cnt = len(duplicate_jao_ids)
    jao_unmatched_cnt = max(0, total_jao - jao_matched_cnt)

    pypsa_matched_cnt = len(matched_pypsa_info)
    pypsa_unmatched_cnt = max(0, total_pypsa - pypsa_matched_cnt)

    # -------------------------------
    # HTML (Leaflet app + sidebar)
    # -------------------------------
    html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>JAO--PyPSA Line Matching Results</title>

<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css">
<link rel="stylesheet" href="https://ppete2.github.io/Leaflet.PolylineMeasure/Leaflet.PolylineMeasure.css" />

<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script src="https://ppete2.github.io/Leaflet.PolylineMeasure/Leaflet.PolylineMeasure.js"></script>

<style>
html, body {{
height: 100%;
margin: 0;
padding: 0;
font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
}}
#map {{ height: 100%; width: 100%; }}

.sidebar {{
position: absolute; top: 10px; right: 10px; width: 320px;
background: #fff; border-radius: 8px; box-shadow: 0 8px 24px rgba(0,0,0,0.15);
max-height: calc(100% - 20px); overflow-y: auto; z-index: 1000;
}}
.sidebar-header {{
padding: 12px 14px; background: #f7f7f8; border-bottom: 1px solid #eaeaea;
font-weight: 600; border-radius: 8px 8px 0 0;
}}
.sidebar-content {{ padding: 12px 14px 18px; }}
.section-header {{
margin: 14px 0 8px; font-weight: 600; color: #444; border-bottom: 1px solid #eee; padding-bottom: 4px;
}}
.search-box {{
width: 100%; padding: 9px 10px; border: 1px solid #ddd; border-radius: 6px; margin-bottom: 8px;
}}
.search-results {{
position: absolute; width: calc(320px - 28px); background: #fff; border: 1px solid #ddd;
border-radius: 6px; z-index: 1010; display: none; max-height: 220px; overflow-y: auto;
}}
.search-result {{ padding: 8px 10px; cursor: pointer; border-bottom: 1px solid #f0f0f0; }}
.search-result:hover {{ background: #f5f5f5; }}

.filter-option {{
padding: 6px 8px; margin: 3px 0; cursor: pointer; display: flex; align-items: center;
background: #f6f6f6; border: 1px solid #ddd; border-radius: 6px;
}}
.filter-option.active {{ background: #2f855a; color: #fff; border-color: #2f855a; }}
.legend-color {{ display: inline-block; width: 18px; height: 4px; margin-right: 8px; border-radius: 2px; }}

.voltage-group {{ border: 1px solid #eee; border-radius: 6px; padding: 8px; margin-bottom: 12px; }}
.voltage-header {{ font-weight: 700; color: #333; cursor: pointer; }}
.voltage-header::before {{ content: "▼ "; font-size: 10px; }}
.voltage-header.collapsed::before {{ content: "► "; }}
.voltage-content.collapsed {{ display: none; }}

.status-group {{ margin: 10px 0 8px 0; }}
.status-header {{ font-weight: 600; color: #555; cursor: pointer; }}
.status-header::before {{ content: "▼ "; font-size: 10px; }}
.status-header.collapsed::before {{ content: "► "; }}
.status-content.collapsed {{ display: none; }}
.type-group {{ margin-left: 12px; }}

.toggle-all-btn {{
margin: 6px 0 10px; padding: 6px 10px; border: 1px solid #ddd; border-radius: 6px;
background: #fafafa; cursor: pointer; font-size: 12px; text-align: center;
}}
.toggle-all-btn:hover {{ background: #f1f1f1; }}

.stats-box {{
margin-top: 10px; background: #fafafa; border: 1px solid #eee; border-radius: 6px; padding: 10px; font-size: 13px;
}}
.stats-item {{ margin-bottom: 4px; }}

.tab-container {{ width: 100%; margin-top: 16px; }}
.tab-nav {{ display: flex; gap: 4px; border-bottom: 1px solid #ddd; }}
.tab-btn {{
padding: 8px 10px; cursor: pointer; background: #f3f4f6; border: 1px solid #ddd; border-bottom: none;
border-radius: 6px 6px 0 0; font-size: 13px;
}}
.tab-btn.active {{ background: #fff; border-bottom: 1px solid #fff; }}
.tab-content {{ display: none; padding: 12px; border: 1px solid #ddd; border-top: none; }}
.tab-content.active {{ display: block; }}

.results-table {{ width: 100%; border-collapse: collapse; margin-top: 8px; font-size: 13px; }}
.results-table th, .results-table td {{ padding: 8px; border-bottom: 1px solid #eee; text-align: left; }}
.results-table th {{ background: #f9fafb; position: sticky; top: 0; z-index: 1; }}

.modal {{ display: none; position: fixed; z-index: 1050; inset: 0; background: rgba(0,0,0,0.45); }}
.modal-content {{
background: #fff; margin: 6% auto; padding: 16px; border: 1px solid #ddd;
width: 90%; max-width: 760px; border-radius: 8px;
}}
.close-btn {{ float: right; font-size: 24px; cursor: pointer; color: #777; }}
.close-btn:hover {{ color: #222; }}

.param-table {{ width: 100%; border-collapse: collapse; }}
.param-table th, .param-table td {{ padding: 8px; border-bottom: 1px solid #eee; text-align: left; }}
.param-table th {{ background: #f9fafb; }}

.show-params-btn {{
padding: 6px 10px; background: #2f855a; color: #fff; border: none; border-radius: 6px; cursor: pointer;
}}

.highlighted {{ stroke-width: 6px !important; stroke-opacity: 1 !important; animation: pulse 1.5s infinite; }}
@keyframes pulse {{ 0% {{opacity: 1;}} 50% {{opacity: .5;}} 100% {{opacity: 1;}} }}

.leaflet-control-polylinemeasure {{
background-color: white !important; padding: 4px !important; border-radius: 6px !important;
}}
</style>
</head>
<body>
<div id="map"></div>

<div class="sidebar" id="controlPanel">
<div class="sidebar-header"><i class="fas fa-search"></i> Search & Filter</div>
<div class="sidebar-content">
<input type="text" id="searchInput" class="search-box" placeholder="Search for lines...">
<div id="searchResults" class="search-results"></div>

<div class="section-header">By Voltage</div>
<div class="toggle-all-btn" id="toggleAllVoltages">Toggle All Voltages</div>

<!-- DC -->
<div class="voltage-group" data-voltage="DC">
<div class="voltage-header" data-voltage="DC">DC Links</div>
<div class="voltage-content">
<!-- JAO matched -->
<div class="status-group">
<div class="status-header" data-type="jao" data-voltage="DC" data-status="matched">JAO Matched</div>
<div class="status-content">
<div class="type-group">
<div class="filter-option active" data-type="jao" data-voltage="DC" data-status="matched" data-subtype="matched">
<div class="legend-color" style="background:#006400;"></div>
<span>Regular Match</span>
</div>
<div class="filter-option active" data-type="jao" data-voltage="DC" data-status="matched" data-subtype="geometric">
<div class="legend-color" style="background:#00688B;"></div>
<span>Geometric Match</span>
</div>
<div class="filter-option active" data-type="jao" data-voltage="DC" data-status="matched" data-subtype="parallel">
<div class="legend-color" style="background:#8B008B;"></div>
<span>Parallel Circuit</span>
</div>
<div class="filter-option active" data-type="jao" data-voltage="DC" data-status="matched" data-subtype="parallel_voltage">
<div class="legend-color" style="background:#CD6600;"></div>
<span>Parallel Voltage</span>
</div>
<div class="filter-option active" data-type="jao" data-voltage="DC" data-status="matched" data-subtype="duplicate">
<div class="legend-color" style="background:#DA70D6;"></div>
<span>Duplicate</span>
</div>
</div>
</div>
</div>
<!-- JAO unmatched -->
<div class="status-group">
<div class="status-header" data-type="jao" data-voltage="DC" data-status="unmatched">JAO Unmatched</div>
<div class="status-content">
<div class="type-group">
<div class="filter-option active" data-type="jao" data-voltage="DC" data-status="unmatched" data-subtype="unmatched">
<div class="legend-color" style="background:#8B0000;"></div>
<span>Unmatched JAO</span>
</div>
</div>
</div>
</div>
<!-- PyPSA matched -->
<div class="status-group">
<div class="status-header" data-type="pypsa" data-voltage="DC" data-status="matched">PyPSA Matched</div>
<div class="status-content">
<div class="type-group">
<div class="filter-option active" data-type="pypsa" data-voltage="DC" data-status="matched" data-subtype="matched">
<div class="legend-color" style="background:#006400;"></div>
<span>Regular Match</span>
</div>
<div class="filter-option active" data-type="pypsa" data-voltage="DC" data-status="matched" data-subtype="geometric">
<div class="legend-color" style="background:#1E90FF;"></div>
<span>Geometric Match</span>
</div>
<div class="filter-option active" data-type="pypsa" data-voltage="DC" data-status="matched" data-subtype="parallel">
<div class="legend-color" style="background:#8B008B;"></div>
<span>Parallel Circuit</span>
</div>
<div class="filter-option active" data-type="pypsa" data-voltage="DC" data-status="matched" data-subtype="parallel_voltage">
<div class="legend-color" style="background:#FF4500;"></div>
<span>Parallel Voltage</span>
</div>
<div class="filter-option active" data-type="pypsa" data-voltage="DC" data-status="matched" data-subtype="duplicate">
<div class="legend-color" style="background:#FF00FF;"></div>
<span>Duplicate</span>
</div>
</div>
</div>
</div>
<!-- PyPSA unmatched -->
<div class="status-group">
<div class="status-header" data-type="pypsa" data-voltage="DC" data-status="unmatched">PyPSA Unmatched</div>
<div class="status-content">
<div class="type-group">
<div class="filter-option active" data-type="pypsa" data-voltage="DC" data-status="unmatched" data-subtype="unmatched">
<div class="legend-color" style="background:#CD6600;"></div>
<span>Unmatched PyPSA</span>
</div>
</div>
</div>
</div>
</div>
</div>

<!-- 110kV -->
<div class="voltage-group" data-voltage="110">
<div class="voltage-header" data-voltage="110">110 kV</div>
<div class="voltage-content">
<!-- JAO matched -->
<div class="status-group">
<div class="status-header" data-type="jao" data-voltage="110" data-status="matched">JAO Matched</div>
<div class="status-content">
<div class="type-group">
<div class="filter-option active" data-type="jao" data-voltage="110" data-status="matched" data-subtype="matched">
<div class="legend-color" style="background:#008000;"></div>
<span>Regular Match</span>
</div>
<div class="filter-option active" data-type="jao" data-voltage="110" data-status="matched" data-subtype="geometric">
<div class="legend-color" style="background:#00BFFF;"></div>
<span>Geometric Match</span>
</div>
<div class="filter-option active" data-type="jao" data-voltage="110" data-status="matched" data-subtype="parallel">
<div class="legend-color" style="background:#9932CC;"></div>
<span>Parallel Circuit</span>
</div>
<div class="filter-option active" data-type="jao" data-voltage="110" data-status="matched" data-subtype="parallel_voltage">
<div class="legend-color" style="background:#FF8C00;"></div>
<span>Parallel Voltage</span>
</div>
<div class="filter-option active" data-type="jao" data-voltage="110" data-status="matched" data-subtype="duplicate">
<div class="legend-color" style="background:#DA70D6;"></div>
<span>Duplicate</span>
</div>
</div>
</div>
</div>
<!-- JAO unmatched -->
<div class="status-group">
<div class="status-header" data-type="jao" data-voltage="110" data-status="unmatched">JAO Unmatched</div>
<div class="status-content">
<div class="type-group">
<div class="filter-option active" data-type="jao" data-voltage="110" data-status="unmatched" data-subtype="unmatched">
<div class="legend-color" style="background:#ff0000;"></div>
<span>Unmatched JAO</span>
</div>
</div>
</div>
</div>
<!-- PyPSA matched -->
<div class="status-group">
<div class="status-header" data-type="pypsa" data-voltage="110" data-status="matched">PyPSA Matched</div>
<div class="status-content">
<div class="type-group">
<div class="filter-option active" data-type="pypsa" data-voltage="110" data-status="matched" data-subtype="matched">
<div class="legend-color" style="background:#006400;"></div>
<span>Regular Match</span>
</div>
<div class="filter-option active" data-type="pypsa" data-voltage="110" data-status="matched" data-subtype="geometric">
<div class="legend-color" style="background:#1E90FF;"></div>
<span>Geometric Match</span>
</div>
<div class="filter-option active" data-type="pypsa" data-voltage="110" data-status="matched" data-subtype="parallel">
<div class="legend-color" style="background:#8B008B;"></div>
<span>Parallel Circuit</span>
</div>
<div class="filter-option active" data-type="pypsa" data-voltage="110" data-status="matched" data-subtype="parallel_voltage">
<div class="legend-color" style="background:#FF4500;"></div>
<span>Parallel Voltage</span>
</div>
<div class="filter-option active" data-type="pypsa" data-voltage="110" data-status="matched" data-subtype="duplicate">
<div class="legend-color" style="background:#FF00FF;"></div>
<span>Duplicate</span>
</div>
</div>
</div>
</div>
<!-- PyPSA unmatched -->
<div class="status-group">
<div class="status-header" data-type="pypsa" data-voltage="110" data-status="unmatched">PyPSA Unmatched</div>
<div class="status-content">
<div class="type-group">
<div class="filter-option active" data-type="pypsa" data-voltage="110" data-status="unmatched" data-subtype="unmatched">
<div class="legend-color" style="background:#CD8500;"></div>
<span>Unmatched PyPSA</span>
</div>
</div>
</div>
</div>
</div>
</div>

<!-- 220kV -->
<div class="voltage-group" data-voltage="220">
<div class="voltage-header" data-voltage="220">220 kV</div>
<div class="voltage-content">
<!-- JAO matched -->
<div class="status-group">
<div class="status-header" data-type="jao" data-voltage="220" data-status="matched">JAO Matched</div>
<div class="status-content">
<div class="type-group">
<div class="filter-option active" data-type="jao" data-voltage="220" data-status="matched" data-subtype="matched">
<div class="legend-color" style="background:#008000;"></div>
<span>Regular Match</span>
</div>
<div class="filter-option active" data-type="jao" data-voltage="220" data-status="matched" data-subtype="geometric">
<div class="legend-color" style="background:#00BFFF;"></div>
<span>Geometric Match</span>
</div>
<div class="filter-option active" data-type="jao" data-voltage="220" data-status="matched" data-subtype="parallel">
<div class="legend-color" style="background:#9932CC;"></div>
<span>Parallel Circuit</span>
</div>
<div class="filter-option active" data-type="jao" data-voltage="220" data-status="matched" data-subtype="parallel_voltage">
<div class="legend-color" style="background:#FF8C00;"></div>
<span>Parallel Voltage</span>
</div>
<div class="filter-option active" data-type="jao" data-voltage="220" data-status="matched" data-subtype="duplicate">
<div class="legend-color" style="background:#DA70D6;"></div>
<span>Duplicate</span>
</div>
</div>
</div>
</div>
<!-- JAO unmatched -->
<div class="status-group">
<div class="status-header" data-type="jao" data-voltage="220" data-status="unmatched">JAO Unmatched</div>
<div class="status-content">
<div class="type-group">
<div class="filter-option active" data-type="jao" data-voltage="220" data-status="unmatched" data-subtype="unmatched">
<div class="legend-color" style="background:#ff0000;"></div>
<span>Unmatched JAO</span>
</div>
</div>
</div>
</div>
<!-- PyPSA matched -->
<div class="status-group">
<div class="status-header" data-type="pypsa" data-voltage="220" data-status="matched">PyPSA Matched</div>
<div class="status-content">
<div class="type-group">
<div class="filter-option active" data-type="pypsa" data-voltage="220" data-status="matched" data-subtype="matched">
<div class="legend-color" style="background:#008000;"></div>
<span>Regular Match</span>
</div>
<div class="filter-option active" data-type="pypsa" data-voltage="220" data-status="matched" data-subtype="geometric">
<div class="legend-color" style="background:#1E90FF;"></div>
<span>Geometric Match</span>
</div>
<div class="filter-option active" data-type="pypsa" data-voltage="220" data-status="matched" data-subtype="parallel">
<div class="legend-color" style="background:#8B008B;"></div>
<span>Parallel Circuit</span>
</div>
<div class="filter-option active" data-type="pypsa" data-voltage="220" data-status="matched" data-subtype="parallel_voltage">
<div class="legend-color" style="background:#FF4500;"></div>
<span>Parallel Voltage</span>
</div>
<div class="filter-option active" data-type="pypsa" data-voltage="220" data-status="matched" data-subtype="duplicate">
<div class="legend-color" style="background:#FF00FF;"></div>
<span>Duplicate</span>
</div>
</div>
</div>
</div>
<!-- PyPSA unmatched -->
<div class="status-group">
<div class="status-header" data-type="pypsa" data-voltage="220" data-status="unmatched">PyPSA Unmatched</div>
<div class="status-content">
<div class="type-group">
<div class="filter-option active" data-type="pypsa" data-voltage="220" data-status="unmatched" data-subtype="unmatched">
<div class="legend-color" style="background:#ffa500;"></div>
<span>Unmatched PyPSA</span>
</div>
</div>
</div>
</div>
</div>
</div>

<!-- 400kV -->
<div class="voltage-group" data-voltage="400">
<div class="voltage-header" data-voltage="400">400 kV</div>
<div class="voltage-content">
<!-- JAO matched -->
<div class="status-group">
<div class="status-header" data-type="jao" data-voltage="400" data-status="matched">JAO Matched</div>
<div class="status-content">
<div class="type-group">
<div class="filter-option active" data-type="jao" data-voltage="400" data-status="matched" data-subtype="matched">
<div class="legend-color" style="background:#008000;"></div>
<span>Regular Match</span>
</div>
<div class="filter-option active" data-type="jao" data-voltage="400" data-status="matched" data-subtype="geometric">
<div class="legend-color" style="background:#00BFFF;"></div>
<span>Geometric Match</span>
</div>
<div class="filter-option active" data-type="jao" data-voltage="400" data-status="matched" data-subtype="parallel">
<div class="legend-color" style="background:#9932CC;"></div>
<span>Parallel Circuit</span>
</div>
<div class="filter-option active" data-type="jao" data-voltage="400" data-status="matched" data-subtype="parallel_voltage">
<div class="legend-color" style="background:#FF8C00;"></div>
<span>Parallel Voltage</span>
</div>
<div class="filter-option active" data-type="jao" data-voltage="400" data-status="matched" data-subtype="duplicate">
<div class="legend-color" style="background:#DA70D6;"></div>
<span>Duplicate</span>
</div>
</div>
</div>
</div>
<!-- JAO unmatched -->
<div class="status-group">
<div class="status-header" data-type="jao" data-voltage="400" data-status="unmatched">JAO Unmatched</div>
<div class="status-content">
<div class="type-group">
<div class="filter-option active" data-type="jao" data-voltage="400" data-status="unmatched" data-subtype="unmatched">
<div class="legend-color" style="background:#ff0000;"></div>
<span>Unmatched JAO</span>
</div>
</div>
</div>
</div>
<!-- PyPSA matched -->
<div class="status-group">
<div class="status-header" data-type="pypsa" data-voltage="400" data-status="matched">PyPSA Matched</div>
<div class="status-content">
<div class="type-group">
<div class="filter-option active" data-type="pypsa" data-voltage="400" data-status="matched" data-subtype="matched">
<div class="legend-color" style="background:#4b0082;"></div>
<span>Regular Match</span>
</div>
<div class="filter-option active" data-type="pypsa" data-voltage="400" data-status="matched" data-subtype="geometric">
<div class="legend-color" style="background:#1E90FF;"></div>
<span>Geometric Match</span>
</div>
<div class="filter-option active" data-type="pypsa" data-voltage="400" data-status="matched" data-subtype="parallel">
<div class="legend-color" style="background:#8B008B;"></div>
<span>Parallel Circuit</span>
</div>
<div class="filter-option active" data-type="pypsa" data-voltage="400" data-status="matched" data-subtype="parallel_voltage">
<div class="legend-color" style="background:#FF4500;"></div>
<span>Parallel Voltage</span>
</div>
<div class="filter-option active" data-type="pypsa" data-voltage="400" data-status="matched" data-subtype="duplicate">
<div class="legend-color" style="background:#FF00FF;"></div>
<span>Duplicate</span>
</div>
</div>
</div>
</div>
<!-- PyPSA unmatched -->
<div class="status-group">
<div class="status-header" data-type="pypsa" data-voltage="400" data-status="unmatched">PyPSA Unmatched</div>
<div class="status-content">
<div class="type-group">
<div class="filter-option active" data-type="pypsa" data-voltage="400" data-status="unmatched" data-subtype="unmatched">
<div class="legend-color" style="background:#7d1d88;"></div>
<span>Unmatched PyPSA</span>
</div>
</div>
</div>
</div>
</div>
</div>

<div class="section-header">Tools</div>
<div class="type-group"><p>Click the ruler icon on the map to measure distances.</p></div>

<div class="stats-box">
<h3><i class="fas fa-chart-pie"></i> Statistics</h3>
<div class="stats-item">JAO Lines: {total_jao} total</div>
<div class="stats-item">- Matched: {jao_matched_cnt} ({(jao_matched_cnt / max(1, total_jao)) * 100:.1f}%)</div>
<div class="stats-item">- Geometric: {jao_geom_cnt} ({(jao_geom_cnt / max(1, total_jao)) * 100:.1f}%)</div>
<div class="stats-item">- Parallel Circuit: {jao_parallel_cnt} ({(jao_parallel_cnt / max(1, total_jao)) * 100:.1f}%)</div>
<div class="stats-item">- Parallel Voltage: {jao_parvolt_cnt} ({(jao_parvolt_cnt / max(1, total_jao)) * 100:.1f}%)</div>
<div class="stats-item">- Duplicate: {jao_dup_cnt} ({(jao_dup_cnt / max(1, total_jao)) * 100:.1f}%)</div>
<div class="stats-item">- Unmatched: {jao_unmatched_cnt} ({(jao_unmatched_cnt / max(1, total_jao)) * 100:.1f}%)</div>
<hr>
<div class="stats-item">PyPSA Lines: {total_pypsa} total</div>
<div class="stats-item">- Matched: {pypsa_matched_cnt} ({(pypsa_matched_cnt / max(1, total_pypsa)) * 100:.1f}%)</div>
<div class="stats-item">- Unmatched: {pypsa_unmatched_cnt} ({(pypsa_unmatched_cnt / max(1, total_pypsa)) * 100:.1f}%)</div>
<div class="stats-item">- 400kV Lines: {v400_count}</div>
<div class="stats-item">- 220kV Lines: {v220_count}</div>
<div class="stats-item">- 110kV Lines: {v110_count}</div>
<div class="stats-item">- DC Links: {dc_count}</div>
</div>

<div class="section-header">Detailed Results</div>
<div class="tab-container">
<div class="tab-nav">
<div class="tab-btn active" data-tab="jaoTab">JAO Lines</div>
<div class="tab-btn" data-tab="pypsaTab">PyPSA Lines</div>
<div class="tab-btn" data-tab="comparisonTab">Parameter Comparison</div>
</div>

<div id="jaoTab" class="tab-content active">
<input type="text" id="jaoSearch" placeholder="Search for JAO lines..." class="search-box">
<table id="jaoTable" class="results-table">
<thead>
<tr>
<th>JAO ID</th>
<th>Name</th>
<th>Voltage (kV)</th>
<th>Length (km)</th>
<th>Match Status</th>
<th>PyPSA IDs</th>
</tr>
</thead>
<tbody></tbody>
</table>
</div>

<div id="pypsaTab" class="tab-content">
<input type="text" id="pypsaSearch" placeholder="Search for PyPSA lines..." class="search-box">
<table id="pypsaTable" class="results-table">
<thead>
<tr>
<th>PyPSA ID</th>
<th>Bus 0</th>
<th>Bus 1</th>
<th>Voltage (kV)</th>
<th>Length (km)</th>
<th>Match Status</th>
<th>Matched JAO IDs</th>
<th>Electrical Parameters</th>
</tr>
</thead>
<tbody></tbody>
</table>
</div>

<div id="comparisonTab" class="tab-content">
<input type="text" id="comparisonSearch" placeholder="Search for comparisons..." class="search-box">
<table id="comparisonTable" class="results-table">
<thead>
<tr>
<th>JAO ID</th>
<th>PyPSA ID</th>
<th>Voltage (kV)</th>
<th>Length Ratio</th>
<th>R (ohm) Ratio</th>
<th>X (ohm) Ratio</th>
<th>B (S) Ratio</th>
<th>Details</th>
</tr>
</thead>
<tbody></tbody>
</table>
</div>
</div>
</div>
</div>

<!-- Modal -->
<div id="paramModal" class="modal">
<div class="modal-content">
<span class="close-btn">&times;</span>
<h2 id="paramModalTitle">Electrical Parameters</h2>
<div id="paramModalContent"></div>
</div>
</div>

<script>
// Map init
var map = L.map('map').setView([51.1657, 10.4515], 6);
L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
attribution: '&copy; OpenStreetMap contributors'
}}).addTo(map);

// Measure tool
L.control.polylineMeasure({{
position: 'topleft', unit: 'metres', showBearings: true,
clearMeasurementsOnStop: false, showClearControl: true, showUnitControl: true
}}).addTo(map);

// Data
var jaoLines = {jao_json};
var pypsaLines = {pypsa_json};

// Styles
function jaoStyle(feature) {{
    // Simplified version to test basic matching
    console.log("Styling JAO feature:", feature.properties.id, "status:", feature.properties.status, "matchStatus:", feature.properties.matchStatus);
    
    // First try the matchStatus
    if (feature.properties.matchStatus === "matched") {{
        return {{color: "green", weight: 3, opacity: 0.85}};
    }} else {{
        return {{color: "red", weight: 3, opacity: 0.85}};
    }}
}}

function pypsaStyle(feature) {{
    let v = feature.properties.voltage;
    let isMatched = feature.properties.matchStatus === "matched";
    let mt = feature.properties.status;
    let isDC = feature.properties.voltageClass === "DC";

    if (isDC) {{
        if (isMatched) {{
            switch (mt) {{
                case "geometric": return {{color:"#1E90FF", weight:3.5, opacity:0.8, dashArray:"10,5"}};
                case "parallel": return {{color:"#8B008B", weight:3.5, opacity:0.8, dashArray:"5,5"}};
                case "parallel_voltage": return {{color:"#FF4500", weight:3.5, opacity:0.8, dashArray:"10,2,2,2"}};
                case "duplicate": return {{color:"#FF00FF", weight:3.5, opacity:0.8, dashArray:"2,5"}};
                default: return {{color:"#006400", weight:3.5, opacity:0.85, dashArray:"8,4"}};
            }}
        }} else {{
            return {{color:"#CD6600", weight:3.5, opacity:0.85, dashArray:"8,4"}};
        }}
    }} else if (isMatched) {{
        switch (mt) {{
            case "geometric": return {{color:"#1E90FF", weight:2.5, opacity:0.8}};
            case "parallel": return {{color:"#8B008B", weight:2.5, opacity:0.8, dashArray:"5,5"}};
            case "parallel_voltage": return {{color:"#FF4500", weight:2.5, opacity:0.8, dashArray:"10,2,2,2"}};
            case "duplicate": return {{color:"#FF00FF", weight:2.5, opacity:0.8, dashArray:"2,5"}};
            default:
                if (v >= 300) return {{color:"#4b0082", weight:3, opacity:0.85}};
                else if (v >= 200) return {{color:"#008000", weight:2.5, opacity:0.8}};
                else return {{color:"#006400", weight:2.5, opacity:0.8}};
        }}
    }} else {{
        if (v >= 300) return {{color:"#7d1d88", weight:3, opacity:0.85, dashArray:"5,5"}};
        else if (v >= 200) return {{color:"#ffa500", weight:2.5, opacity:0.8, dashArray:"5,5"}};
        else return {{color:"#CD8500", weight:2.5, opacity:0.8, dashArray:"5,5"}};
    }}
}}

// Layers registry - now including DC and 110kV
var layers = {{
    jao: {{
        "DC": {{ matched: {{"matched":L.geoJSON(null,{{style:jaoStyle}}),
                        "geometric":L.geoJSON(null,{{style:jaoStyle}}),
                        "parallel":L.geoJSON(null,{{style:jaoStyle}}),
                        "parallel_voltage":L.geoJSON(null,{{style:jaoStyle}}),
                        "duplicate":L.geoJSON(null,{{style:jaoStyle}})}},
                unmatched: {{"unmatched":L.geoJSON(null,{{style:jaoStyle}})}} }},
        "110kV": {{ matched: {{"matched":L.geoJSON(null,{{style:jaoStyle}}),
                        "geometric":L.geoJSON(null,{{style:jaoStyle}}),
                        "parallel":L.geoJSON(null,{{style:jaoStyle}}),
                        "parallel_voltage":L.geoJSON(null,{{style:jaoStyle}}),
                        "duplicate":L.geoJSON(null,{{style:jaoStyle}})}},
                unmatched: {{"unmatched":L.geoJSON(null,{{style:jaoStyle}})}} }},
        "220kV": {{ matched: {{"matched":L.geoJSON(null,{{style:jaoStyle}}),
                        "geometric":L.geoJSON(null,{{style:jaoStyle}}),
                        "parallel":L.geoJSON(null,{{style:jaoStyle}}),
                        "parallel_voltage":L.geoJSON(null,{{style:jaoStyle}}),
                        "duplicate":L.geoJSON(null,{{style:jaoStyle}})}},
                unmatched: {{"unmatched":L.geoJSON(null,{{style:jaoStyle}})}} }},
        "400kV": {{ matched: {{"matched":L.geoJSON(null,{{style:jaoStyle}}),
                        "geometric":L.geoJSON(null,{{style:jaoStyle}}),
                        "parallel":L.geoJSON(null,{{style:jaoStyle}}),
                        "parallel_voltage":L.geoJSON(null,{{style:jaoStyle}}),
                        "duplicate":L.geoJSON(null,{{style:jaoStyle}})}},
                unmatched: {{"unmatched":L.geoJSON(null,{{style:jaoStyle}})}} }}
    }},
    pypsa: {{
        "DC": {{ matched: {{"matched":L.geoJSON(null,{{style:pypsaStyle}}),
                        "geometric":L.geoJSON(null,{{style:pypsaStyle}}),
                        "parallel":L.geoJSON(null,{{style:pypsaStyle}}),
                        "parallel_voltage":L.geoJSON(null,{{style:pypsaStyle}}),
                        "duplicate":L.geoJSON(null,{{style:pypsaStyle}})}},
                unmatched: {{"unmatched":L.geoJSON(null,{{style:pypsaStyle}})}} }},
        "110kV": {{ matched: {{"matched":L.geoJSON(null,{{style:pypsaStyle}}),
                        "geometric":L.geoJSON(null,{{style:pypsaStyle}}),
                        "parallel":L.geoJSON(null,{{style:pypsaStyle}}),
                        "parallel_voltage":L.geoJSON(null,{{style:pypsaStyle}}),
                        "duplicate":L.geoJSON(null,{{style:pypsaStyle}})}},
                unmatched: {{"unmatched":L.geoJSON(null,{{style:pypsaStyle}})}} }},
        "220kV": {{ matched: {{"matched":L.geoJSON(null,{{style:pypsaStyle}}),
                        "geometric":L.geoJSON(null,{{style:pypsaStyle}}),
                        "parallel":L.geoJSON(null,{{style:pypsaStyle}}),
                        "parallel_voltage":L.geoJSON(null,{{style:pypsaStyle}}),
                        "duplicate":L.geoJSON(null,{{style:pypsaStyle}})}},
                unmatched: {{"unmatched":L.geoJSON(null,{{style:pypsaStyle}})}} }},
        "400kV": {{ matched: {{"matched":L.geoJSON(null,{{style:pypsaStyle}}),
                        "geometric":L.geoJSON(null,{{style:pypsaStyle}}),
                        "parallel":L.geoJSON(null,{{style:pypsaStyle}}),
                        "parallel_voltage":L.geoJSON(null,{{style:pypsaStyle}}),
                        "duplicate":L.geoJSON(null,{{style:pypsaStyle}})}},
                unmatched: {{"unmatched":L.geoJSON(null,{{style:pypsaStyle}})}} }}
    }}
}};

function addFeatureToLayer(root, feature) {{
    var props = feature.properties;
    var vClass = props.voltageClass;
    var mStatus = props.matchStatus;
    var status = props.status;
    if (root[vClass] && root[vClass][mStatus] && root[vClass][mStatus][status]) {{
        root[vClass][mStatus][status].addData(feature);
        root[vClass][mStatus][status].eachLayer(function(layer) {{
            if (layer.feature && layer.feature.id === feature.id) {{
                layer.bindTooltip(props.tooltip);
                layer.on('click', function() {{
                    highlightFeature(feature.id);
                    if (props.matchedJaoIds && props.matchedJaoIds.length) {{
                        setTimeout(function() {{ highlightFeature("jao_" + props.matchedJaoIds[0]); }}, 800);
                    }}
                }});
            }}
        }});
    }}
}}

// Load features
jaoLines.features.forEach(function(f) {{ addFeatureToLayer(layers.jao, f); }});
pypsaLines.features.forEach(function(f) {{ addFeatureToLayer(layers.pypsa, f); }});

// Highlight utilities
function clearHighlights() {{
    function resetSet(layerSet, styler) {{
        if (!layerSet) return;
        Object.keys(layerSet).forEach(function(v) {{
            Object.keys(layerSet[v]).forEach(function(ms) {{
                Object.keys(layerSet[v][ms]).forEach(function(sub) {{
                    layerSet[v][ms][sub].eachLayer(function(layer) {{
                        layer.setStyle(styler(layer.feature));
                        if (layer._path) layer._path.classList.remove('highlighted');
                    }});
                }});
            }});
        }});
    }}
    resetSet(layers.jao, jaoStyle);
    resetSet(layers.pypsa, pypsaStyle);
}}

function highlightFeature(id) {{
    clearHighlights();
    function checkSet(layerSet) {{
        if (!layerSet) return;
        Object.keys(layerSet).forEach(function(v) {{
            Object.keys(layerSet[v]).forEach(function(ms) {{
                Object.keys(layerSet[v][ms]).forEach(function(sub) {{
                    layerSet[v][ms][sub].eachLayer(function(layer) {{
                        if (layer.feature && layer.feature.id === id) {{
                            layer.setStyle({{weight: 6, opacity: 1}});
                            if (layer._path) layer._path.classList.add('highlighted');
                            var b = layer.getBounds && layer.getBounds();
                            if (b && b.isValid && b.isValid()) map.fitBounds(b, {{padding:[50,50]}});
                        }}
                    }});
                }});
            }});
        }});
    }}
    checkSet(layers.jao);
    checkSet(layers.pypsa);
}}

// Filters
function applyFilters() {{
    document.querySelectorAll('.filter-option').forEach(function(el) {{
        var type = el.getAttribute('data-type');
        var voltage = el.getAttribute('data-voltage');
        var status = el.getAttribute('data-status');
        var subtype = el.getAttribute('data-subtype');
        var isActive = el.classList.contains('active');
        var vClass = voltage === "DC" ? "DC" : 
                    voltage === "110" ? "110kV" :
                    voltage === "220" ? "220kV" : 
                    voltage === "400" ? "400kV" : "other";
        var root = type === 'jao' ? layers.jao : layers.pypsa;

        if (root[vClass] && root[vClass][status] && root[vClass][status][subtype]) {{
            var layer = root[vClass][status][subtype];
            if (isActive) layer.addTo(map);
            else map.removeLayer(layer);
        }}
    }});
}}

function setupFilters() {{
    document.querySelectorAll('.filter-option').forEach(function(opt) {{
        opt.addEventListener('click', function(e) {{
            this.classList.toggle('active');
            applyFilters();
            e.stopPropagation();
        }});
    }});
    document.querySelectorAll('.voltage-header').forEach(function(h) {{
        h.addEventListener('click', function() {{
            this.classList.toggle('collapsed');
            var c = this.nextElementSibling;
            if (c) c.classList.toggle('collapsed');
            applyFilters();
        }});
    }});
    document.querySelectorAll('.status-header').forEach(function(h) {{
        h.addEventListener('click', function() {{
            this.classList.toggle('collapsed');
            var c = this.nextElementSibling;
            if (c) c.classList.toggle('collapsed');
            applyFilters();
        }});
    }});
    document.getElementById('toggleAllVoltages').addEventListener('click', function() {{
        var all = document.querySelectorAll('.filter-option[data-type]');
        var activeCount = document.querySelectorAll('.filter-option[data-type].active').length;
        var turnOff = activeCount === all.length;
        all.forEach(function(el) {{ el.classList.toggle('active', !turnOff); }});
        applyFilters();
    }});
}}

// Search
function createSearchIndex() {{
    var data = [];
    jaoLines.features.forEach(function(f) {{
        data.push({{id:f.id, text:f.properties.tooltip, feature:f}});
    }});
    pypsaLines.features.forEach(function(f) {{
        data.push({{id:f.id, text:f.properties.tooltip, feature:f}});
    }});
    // Relationships: PyPSA->JAO
    pypsaLines.features.forEach(function(f) {{
        var p = f.properties;
        if (p.matchedJaoIds && p.matchedJaoIds.length) {{
            p.matchedJaoIds.forEach(function(jid) {{
                data.push({{
                    id: "ref_"+f.id+"_"+jid,
                    text: "JAO "+jid+" is matched to PyPSA "+p.id+" ("+p.status+")",
                    type: "reference", jaoId: jid, pypsaId: p.id
                }});
            }});
        }}
    }});
    return data;
}}

function initializeSearch() {{
    var input = document.getElementById('searchInput');
    var results = document.getElementById('searchResults');
    var idx = createSearchIndex();

    input.addEventListener('input', function() {{
        var q = this.value.toLowerCase();
        if (q.length < 2) {{ results.innerHTML = ""; results.style.display = "none"; return; }}
        var hits = idx.filter(function(it) {{ return it.text.toLowerCase().includes(q); }});
        results.innerHTML = "";
        hits.slice(0,12).forEach(function(r) {{
            var div = document.createElement('div');
            div.className = 'search-result';
            div.textContent = r.text;
            div.onclick = function() {{
                if (r.type === "reference") {{
                    if (r.jaoId) highlightFeature("jao_"+r.jaoId);
                    if (r.pypsaId) setTimeout(function() {{ highlightFeature("pypsa_"+r.pypsaId); }}, 700);
                }} else {{
                    highlightFeature(r.id);
                }}
                input.value = r.text;
                results.style.display = "none";
            }};
            results.appendChild(div);
        }});
        results.style.display = hits.length ? "block" : "none";
    }});
    document.addEventListener('click', function(e) {{
        if (!input.contains(e.target) && !results.contains(e.target)) results.style.display = 'none';
    }});
}}

// Tabs & tables
function setupTabs() {{
    document.querySelectorAll('.tab-btn').forEach(function(btn) {{
        btn.addEventListener('click', function() {{
            var id = this.getAttribute('data-tab');
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
            this.classList.add('active');
            document.getElementById(id).classList.add('active');
            if (id === 'jaoTab') updateJaoTable();
            else if (id === 'pypsaTab') updatePypsaTable();
            else if (id === 'comparisonTab') updateComparisonTable();
        }});
    }});
    updateJaoTable();
}}

function updateJaoTable() {{
    var tbody = document.querySelector('#jaoTable tbody');
    tbody.innerHTML = '';
    jaoLines.features.forEach(function(f) {{
        var p = f.properties;
        if (!p || !p.id) return;
        // Only render first part row if feature id has _p suffix
        if (f.id.includes("_p")) return;

        var tr = document.createElement('tr');
        var matchedPypsa = [];
        pypsaLines.features.forEach(function(P) {{
            if (P.properties.matchedJaoIds && P.properties.matchedJaoIds.includes(p.id)) {{
                if (!matchedPypsa.includes(P.properties.id)) matchedPypsa.push(P.properties.id);
            }}
        }});
        tr.innerHTML = `
        <td>${{p.id}}</td>
        <td>${{p.name || ""}}</td>
        <td>${{p.voltage}}</td>
        <td>${{(p.length_km||0).toFixed(2)}}</td>
        <td>${{p.status.charAt(0).toUpperCase()+p.status.slice(1)}}</td>
        <td>${{matchedPypsa.join(', ') || "None"}}</td>`;
        tr.addEventListener('click', function() {{ highlightFeature(f.id); }});
        tbody.appendChild(tr);
    }});
    document.getElementById('jaoSearch').addEventListener('input', function() {{
        var q = this.value.toLowerCase();
        Array.from(document.querySelectorAll('#jaoTable tbody tr')).forEach(function(r){{
            r.style.display = r.textContent.toLowerCase().includes(q) ? '' : 'none';
        }});
    }});
}}

function showElectricalParams(props) {{
    var modal = document.getElementById('paramModal');
    var title = document.getElementById('paramModalTitle');
    var content = document.getElementById('paramModalContent');

    title.textContent = 'Electrical Parameters for PyPSA Line ' + props.id;
    var html = '<table class="param-table">';
    html += '<tr><th>Parameter</th><th>Value</th><th>Unit</th></tr>';
    html += '<tr><td>Voltage</td><td>' + props.voltage + '</td><td>kV</td></tr>';
    html += '<tr><td>Length</td><td>' + (props.length_km||0).toFixed(2) + '</td><td>km</td></tr>';
    html += '<tr><td>Resistance (R)</td><td>' + (props.r_ohm||0).toFixed(6) + '</td><td>Ω</td></tr>';
    html += '<tr><td>Reactance (X)</td><td>' + (props.x_ohm||0).toFixed(6) + '</td><td>Ω</td></tr>';
    html += '<tr><td>Susceptance (B)</td><td>' + (props.b_siemens||0).toFixed(6) + '</td><td>S</td></tr>';
    html += '<tr><td>Conductance (G)</td><td>' + (props.g_siemens||0).toFixed(6) + '</td><td>S</td></tr>';
    html += '<tr><td colspan="3"><b>Per km</b></td></tr>';
    html += '<tr><td>R per km</td><td>' + (props.r_per_km||0).toFixed(6) + '</td><td>Ω/km</td></tr>';
    html += '<tr><td>X per km</td><td>' + (props.x_per_km||0).toFixed(6) + '</td><td>Ω/km</td></tr>';
    html += '<tr><td>B per km</td><td>' + (props.b_per_km||0).toFixed(6) + '</td><td>S/km</td></tr>';
    html += '<tr><td>G per km</td><td>' + (props.g_per_km||0).toFixed(6) + '</td><td>S/km</td></tr>';
    html += '<tr><td colspan="3"><b>Connection</b></td></tr>';
    html += '<tr><td>Bus 0</td><td colspan="2">' + (props.bus0||"") + '</td></tr>';
    html += '<tr><td>Bus 1</td><td colspan="2">' + (props.bus1||"") + '</td></tr>';
    if (props.matchStatus === 'matched' && props.matchedJaoIds && props.matchedJaoIds.length) {{
        html += '<tr><td colspan="3"><b>Match</b></td></tr>';
        html += '<tr><td>JAO IDs</td><td colspan="2">' + props.matchedJaoIds.join(', ') + '</td></tr>';
        html += '<tr><td>Quality</td><td colspan="2">' + (props.matchQuality||'N/A') + '</td></tr>';
    }}
    html += '</table>';
    content.innerHTML = html;
    modal.style.display = 'block';

    var closeBtn = document.getElementsByClassName('close-btn')[0];
    closeBtn.onclick = function() {{ modal.style.display = 'none'; }}
    window.onclick = function(e) {{ if (e.target === modal) modal.style.display = 'none'; }}
}}

function updatePypsaTable() {{
    var tbody = document.querySelector('#pypsaTable tbody');
    tbody.innerHTML = '';
    pypsaLines.features.forEach(function(f) {{
        var p = f.properties;
        if (!p || !p.id) return;
        if (f.id.includes("_p")) return;

        var tr = document.createElement('tr');
        var btn = document.createElement('button');
        btn.className = 'show-params-btn';
        btn.textContent = 'Show Parameters';
        btn.onclick = function(e) {{ e.stopPropagation(); showElectricalParams(p); }};

        tr.innerHTML = `
        <td>${{p.id}}</td>
        <td>${{p.bus0 || 'N/A'}}</td>
        <td>${{p.bus1 || 'N/A'}}</td>
        <td>${{p.voltage}}</td>
        <td>${{(p.length_km||0).toFixed(2)}}</td>
        <td>${{p.matchStatus.charAt(0).toUpperCase()+p.matchStatus.slice(1)}}</td>
        <td>${{p.matchedJaoIds && p.matchedJaoIds.length ? p.matchedJaoIds.join(', ') : "None"}}</td>
        <td class="params-cell"></td>`;
        tr.querySelector('.params-cell').appendChild(btn);
        tr.addEventListener('click', function() {{ highlightFeature(f.id); }});
        tbody.appendChild(tr);
    }});
    document.getElementById('pypsaSearch').addEventListener('input', function() {{
        var q = this.value.toLowerCase();
        Array.from(document.querySelectorAll('#pypsaTable tbody tr')).forEach(function(r){{
            r.style.display = r.textContent.toLowerCase().includes(q) ? '' : 'none';
        }});
    }});
}}

function updateComparisonTable() {{
    var tbody = document.querySelector('#comparisonTable tbody');
    tbody.innerHTML = '';
    pypsaLines.features.forEach(function(pf) {{
        var pp = pf.properties;
        if (!pp.matchedJaoIds || !pp.matchedJaoIds.length) return;

        pp.matchedJaoIds.forEach(function(jid) {{
            // find first (base) JAO feature
            var jf = jaoLines.features.find(function(f) {{ return f.properties && f.properties.id === jid; }});
            if (!jf) return;
            var jp = jf.properties;

            var lengthRatio = (jp.length_km && pp.length_km) ? (jp.length_km / pp.length_km) : null;
            var rRatio = (jp.r_per_km>0 && pp.r_per_km>0) ? (pp.r_per_km / jp.r_per_km) : null;
            var xRatio = (jp.x_per_km>0 && pp.x_per_km>0) ? (pp.x_per_km / jp.x_per_km) : null;
            var bRatio = (jp.b_per_km>0 && pp.b_per_km>0) ? (pp.b_per_km / jp.b_per_km) : null;

            var tr = document.createElement('tr');
            var compareBtn = document.createElement('button');
            compareBtn.className = 'show-params-btn';
            compareBtn.textContent = 'Compare';
            compareBtn.onclick = function(e) {{
                e.stopPropagation();
                showParameterComparison(jp, pp);
            }};
            tr.innerHTML = `
            <td>${{jid}}</td>
            <td>${{pp.id}}</td>
            <td>${{jp.voltage}}</td>
            <td>${{lengthRatio ? lengthRatio.toFixed(2) : 'N/A'}}</td>
            <td>${{rRatio ? rRatio.toFixed(2) : 'N/A'}}</td>
            <td>${{xRatio ? xRatio.toFixed(2) : 'N/A'}}</td>
            <td>${{bRatio ? bRatio.toFixed(2) : 'N/A'}}</td>
            <td class="cmp-cell"></td>`;
            tr.querySelector('.cmp-cell').appendChild(compareBtn);
            tr.addEventListener('click', function() {{
                highlightFeature("jao_"+jid);
                setTimeout(function() {{ highlightFeature("pypsa_"+pp.id); }}, 700);
            }});
            tbody.appendChild(tr);
        }});
    }});
    document.getElementById('comparisonSearch').addEventListener('input', function() {{
        var q = this.value.toLowerCase();
        Array.from(document.querySelectorAll('#comparisonTable tbody tr')).forEach(function(r){{
            r.style.display = r.textContent.toLowerCase().includes(q) ? '' : 'none';
        }});
    }});
}}

function showParameterComparison(jaoProps, pypsaProps) {{
    var modal = document.getElementById('paramModal');
    var title = document.getElementById('paramModalTitle');
    var content = document.getElementById('paramModalContent');

    title.textContent = 'Parameter Comparison: JAO ' + jaoProps.id + ' vs PyPSA ' + pypsaProps.id;

    var jaoRkm = (jaoProps.r_per_km||0), jaoXkm = (jaoProps.x_per_km||0), jaoBkm = (jaoProps.b_per_km||0);
    var pRkm = (pypsaProps.r_per_km||0), pXkm = (pypsaProps.x_per_km||0), pBkm = (pypsaProps.b_per_km||0);

    var Ljao = (jaoProps.length_km||0), Lp = (pypsaProps.length_km||0);
    var jaoRtot = jaoRkm * Ljao, jaoXtot = jaoXkm * Ljao, jaoBtot = jaoBkm * Ljao;

    var html = '<div>';
    html += '<h3>Basic Information</h3>';
    var lenRatio = (Ljao && Lp) ? (Ljao/Lp).toFixed(2) : 'N/A';
    html += '<p><b>Length (km):</b> JAO '+Ljao.toFixed(2)+' | PyPSA '+Lp.toFixed(2)+' | Ratio '+lenRatio+'</p>';
    html += '<p><b>Voltage:</b> JAO '+(jaoProps.voltage||'')+' kV | PyPSA '+(pypsaProps.voltage||'')+' kV</p>';

    html += '<h3>Per-km Parameters</h3>';
    function pct(a,b) {{ if (!b || b===0) return 'N/A'; return ((a-b)/b*100).toFixed(2)+'%'; }}
    html += '<p><b>R/km:</b> JAO '+jaoRkm.toFixed(6)+' Ω/km | PyPSA '+pRkm.toFixed(6)+' Ω/km | Δ '+pct(jaoRkm,pRkm)+'</p>';
    html += '<p><b>X/km:</b> JAO '+jaoXkm.toFixed(6)+' Ω/km | PyPSA '+pXkm.toFixed(6)+' Ω/km | Δ '+pct(jaoXkm,pXkm)+'</p>';
    html += '<p><b>B/km:</b> JAO '+jaoBkm.toFixed(6)+' S/km | PyPSA '+pBkm.toFixed(6)+' S/km | Δ '+pct(jaoBkm,pBkm)+'</p>';

    html += '<h3>Total Parameters</h3>';
    html += '<p><b>R total:</b> JAO '+jaoRtot.toFixed(6)+' Ω | PyPSA '+(pypsaProps.r_ohm||0).toFixed(6)+' Ω</p>';
    html += '<p><b>X total:</b> JAO '+jaoXtot.toFixed(6)+' Ω | PyPSA '+(pypsaProps.x_ohm||0).toFixed(6)+' Ω</p>';
    html += '<p><b>B total:</b> JAO '+jaoBtot.toFixed(6)+' S | PyPSA '+(pypsaProps.b_siemens||0).toFixed(6)+' S</p>';
    html += '</div>';

    content.innerHTML = html;
    modal.style.display = 'block';
    var closeBtn = document.getElementsByClassName('close-btn')[0];
    closeBtn.onclick = function() {{ modal.style.display = 'none'; }}
    window.onclick = function(e) {{ if (e.target === modal) modal.style.display = 'none'; }}
}}

// Init
document.addEventListener('DOMContentLoaded', function() {{
    // Initialize 110kV and DC filters to be checked by default
    document.querySelectorAll('.filter-option[data-voltage="DC"]').forEach(function(el) {{
        el.classList.add('active');
    }});
    document.querySelectorAll('.filter-option[data-voltage="110"]').forEach(function(el) {{
        el.classList.add('active');
    }});

    // Initialize search and filters
    initializeSearch();
    setupFilters();
    setupTabs();
    applyFilters();

    // Debug logging
    console.log("Initialized with all voltage classes enabled including 110kV and DC");
}});
</script>
</body>
</html>
"""

    # -------------------------------
    # Write file
    # -------------------------------
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Visualization saved to {out_path}")
    return str(out_path)



def create_enhanced_visualization(jao_gdf, pypsa_gdf, matches, output_path, dc_links_gdf=None, lines_110kv_gdf=None):
    """
    Create an interactive visualization comparing JAO and PyPSA grid lines with comprehensive filtering.

    Parameters:
    -----------
    jao_gdf : GeoDataFrame
        JAO transmission line data with geometries
    pypsa_gdf : GeoDataFrame
        PyPSA transmission line data with geometries
    matches : list of dict
        Matching results between JAO and PyPSA lines
    output_path : str
        Path to save the HTML visualization
    dc_links_gdf : GeoDataFrame, optional
        DC transmission link data with geometries
    lines_110kv_gdf : GeoDataFrame, optional
        110kV transmission line data with geometries

    The visualization includes:
    - Color-coded lines by data source, voltage level, and match status
    - Detailed popups with electrical parameters
    - Comprehensive filtering by data source, voltage level, match status, and line type
    - Search functionality for specific lines by ID or name
    - Problem area highlighting
    - Layer toggling for complex analysis
    """
    print(f"Creating enhanced visualization at {output_path}...")

    # Create base map centered on Germany
    m = folium.Map(
        location=[51.1657, 10.4515],
        zoom_start=6,
        control_scale=True,
        tiles="OpenStreetMap"
    )
    m.add_child(MeasureControl())

    # Create feature groups for organized layer control
    # PyPSA AC layers
    pypsa_matched_400 = folium.FeatureGroup(name="PyPSA-Matched-400kV", show=True)
    pypsa_matched_220 = folium.FeatureGroup(name="PyPSA-Matched-220kV", show=True)
    pypsa_unmatched_400 = folium.FeatureGroup(name="PyPSA-Unmatched-400kV", show=True)
    pypsa_unmatched_220 = folium.FeatureGroup(name="PyPSA-Unmatched-220kV", show=True)

    # JAO layers
    jao_matched_400 = folium.FeatureGroup(name="JAO-Matched-400kV", show=True)
    jao_matched_220 = folium.FeatureGroup(name="JAO-Matched-220kV", show=True)
    jao_unmatched_400 = folium.FeatureGroup(name="JAO-Unmatched-400kV", show=True)
    jao_unmatched_220 = folium.FeatureGroup(name="JAO-Unmatched-220kV", show=True)

    # Additional network layers - initially hidden
    pypsa_110kv = folium.FeatureGroup(name="PyPSA-110kV", show=True)
    pypsa_dc_links = folium.FeatureGroup(name="PyPSA-DC-Links", show=True)

    # Build sets of matched IDs for quick lookup - enhanced for circuit matching
    matched_jao_ids = set()
    matched_pypsa_ids = set()

    # Ensure consistent string formatting for ALL IDs
    for i, match in enumerate(matches):
        if 'jao_id' in match:
            # Convert to string and strip whitespace
            matches[i]['jao_id'] = str(match['jao_id']).strip()

        if 'pypsa_ids' in match:
            # Normalize pypsa_ids format to always be a list
            if isinstance(match['pypsa_ids'], str):
                # Split by both semicolon and comma, then clean up
                ids = []
                for separator in [';', ',']:
                    for pid in match['pypsa_ids'].split(separator):
                        if pid.strip():
                            ids.append(str(pid).strip())
                matches[i]['pypsa_ids'] = ids
            elif isinstance(match['pypsa_ids'], list):
                # Convert all to strings and strip whitespace
                matches[i]['pypsa_ids'] = [str(pid).strip() for pid in match['pypsa_ids']]

    # Collect all matched IDs
    for match in matches:
        if match.get('matched', False):
            matched_jao_ids.add(str(match.get('jao_id', '')).strip())

            if 'pypsa_ids' in match and match['pypsa_ids']:
                if isinstance(match['pypsa_ids'], list):
                    matched_pypsa_ids.update([str(pid).strip() for pid in match['pypsa_ids']])
                elif isinstance(match['pypsa_ids'], str):
                    for separator in [';', ',']:
                        for pid in match['pypsa_ids'].split(separator):
                            if pid.strip():
                                matched_pypsa_ids.add(str(pid).strip())

    print(
        f"Visualization: collected {len(matched_jao_ids)} matched JAO IDs and {len(matched_pypsa_ids)} matched PyPSA IDs")

    # Helper function to safely format electrical parameter values
    def safe_format(value, precision=4):
        """Format a value to a specified precision, handling None/NA values."""
        try:
            if value is None or value == 'N/A' or pd.isna(value):
                return 'N/A'
            return f"{float(value):.{precision}f}"
        except (ValueError, TypeError):
            return 'N/A'

    # Helper function to safely convert to integer
    def _safe_int(value, default=0):
        """Convert a value to integer, handling None/NA values."""
        try:
            if value is None or pd.isna(value):
                return default
            return int(value)
        except (ValueError, TypeError):
            return default

    # Process PYPSA lines
    for _, pypsa_row in pypsa_gdf.iterrows():
        pypsa_id = str(pypsa_row['id'])
        pypsa_geom = pypsa_row.geometry

        if pypsa_geom is None:
            continue

        # Determine match status for styling
        is_matched = str(pypsa_id).strip() in matched_pypsa_ids

        # Style based on match status
        if is_matched:
            color = '#0070FF'  # Bright blue for matched
            weight = 6
            opacity = 1.0
        else:
            color = '#FF8000'  # Orange for unmatched
            weight = 5
            opacity = 0.9

        # Adjust line weight for multi-circuit lines
        circuits = _safe_int(pypsa_row.get('circuits', 1))
        if circuits > 1:
            weight += 1

        # Get voltage level
        pypsa_voltage = _safe_int(pypsa_row.get('voltage', 0))
        is_400kv = pypsa_voltage >= 380  # 380kV and 400kV grouped together
        is_220kv = pypsa_voltage == 220

        # Skip if not a standard voltage
        if not (is_400kv or is_220kv):
            continue

        # Create popup content with electrical parameters
        popup_content = f"""
        <div style="font-family: Arial; max-width: 300px;">
            <h4>PyPSA Line {pypsa_id}</h4>
            <b>Voltage:</b> {pypsa_voltage} kV<br>
            <b>Length:</b> {pypsa_row.get('length_km', 0):.2f} km<br>
            <b>Circuits:</b> {circuits}<br>
            <hr style="margin: 5px 0;">
            <b>Electrical Parameters:</b><br>
            <b>r:</b> {safe_format(pypsa_row.get('r_per_km'))} Ω/km<br>
            <b>x:</b> {safe_format(pypsa_row.get('x_per_km'))} Ω/km<br>
            <b>b:</b> {safe_format(pypsa_row.get('b_per_km'))} μS/km<br>
            <hr style="margin: 5px 0;">
            <b>Status:</b> {'Matched' if is_matched else 'Unmatched'}<br>
        </div>
        """

        # Create GeoJson with properties for filtering
        gj = folium.GeoJson(
            pypsa_geom,
            name=f"PyPSA Line {pypsa_id}",
            tooltip=f"PyPSA Line {pypsa_id}",
            popup=folium.Popup(popup_content, max_width=300),
            style_function=lambda x, c=color, w=weight, o=opacity: {
                'color': c, 'weight': w, 'opacity': o
            }
        )

        # Add to appropriate group based on match status and voltage
        if is_matched and is_400kv:
            gj.add_to(pypsa_matched_400)
        elif is_matched and is_220kv:
            gj.add_to(pypsa_matched_220)
        elif not is_matched and is_400kv:
            gj.add_to(pypsa_unmatched_400)
        elif not is_matched and is_220kv:
            gj.add_to(pypsa_unmatched_220)


    # Process JAO lines
    for _, jao_row in jao_gdf.iterrows():
        jao_id = str(jao_row['id'])
        jao_geom = jao_row.geometry

        if jao_geom is None:
            continue

        # Determine match status for styling
        is_matched = jao_id in matched_jao_ids

        # Style based on match status
        color = '#007200' if is_matched else '#CC0000'  # Green if matched, Red if unmatched
        weight = 3 if is_matched else 2
        opacity = 0.8 if is_matched else 0.6

        # Get voltage level
        jao_voltage = _safe_int(jao_row.get('v_nom', 0))
        is_400kv = jao_voltage >= 380  # 380kV and 400kV grouped together
        is_220kv = jao_voltage == 220

        # Skip if not a standard voltage
        if not (is_400kv or is_220kv):
            continue

        # Create popup content with electrical parameters
        popup_content = f"""
        <div style="font-family: Arial; max-width: 300px;">
            <h4>JAO Line {jao_id}</h4>
            <b>Name:</b> {jao_row.get('NE_name', '')}<br>
            <b>Voltage:</b> {jao_voltage} kV<br>
            <b>Length:</b> {jao_row.get('length_km', 0):.2f} km<br>
            <hr style="margin: 5px 0;">
            <b>Electrical Parameters:</b><br>
            <b>r:</b> {safe_format(jao_row.get('r_per_km'))} Ω/km<br>
            <b>x:</b> {safe_format(jao_row.get('x_per_km'))} Ω/km<br>
            <b>b:</b> {safe_format(jao_row.get('b_per_km'))} μS/km<br>
            <hr style="margin: 5px 0;">
            <b>Status:</b> {'Matched' if is_matched else 'Unmatched'}<br>
        </div>
        """

        # Create GeoJson with properties for filtering
        gj = folium.GeoJson(
            jao_geom,
            name=f"JAO Line {jao_id}",
            tooltip=f"JAO Line {jao_id}",
            popup=folium.Popup(popup_content, max_width=300),
            style_function=lambda x, c=color, w=weight, o=opacity: {
                'color': c, 'weight': w, 'opacity': o
            }
        )

        # Add to appropriate group based on match status and voltage
        if is_matched and is_400kv:
            gj.add_to(jao_matched_400)
        elif is_matched and is_220kv:
            gj.add_to(jao_matched_220)
        elif not is_matched and is_400kv:
            gj.add_to(jao_unmatched_400)
        elif not is_matched and is_220kv:
            gj.add_to(jao_unmatched_220)

    # Process 110kV lines if provided
    if lines_110kv_gdf is not None:
        print("Processing 110kV lines...")
        for _, line_row in lines_110kv_gdf.iterrows():
            line_id = str(line_row['id'])
            line_geom = line_row.geometry

            if line_geom is None:
                continue

            # Style for 110kV lines - purple
            color = '#8A2BE2'  # BlueViolet
            weight = 3
            opacity = 0.7

            # Create popup content
            popup_content = f"""
            <div style="font-family: Arial; max-width: 300px;">
                <h4>110kV Line {line_id}</h4>
                <b>Voltage:</b> 110 kV<br>
                <b>Length:</b> {line_row.get('length_km', 0):.2f} km<br>
                <b>Circuits:</b> {_safe_int(line_row.get('circuits', 1))}<br>
                <hr style="margin: 5px 0;">
                <b>Electrical Parameters:</b><br>
                <b>r:</b> {safe_format(line_row.get('r_per_km'))} Ω/km<br>
                <b>x:</b> {safe_format(line_row.get('x_per_km'))} Ω/km<br>
                <b>b:</b> {safe_format(line_row.get('b_per_km'))} μS/km<br>
            </div>
            """

            # Create GeoJson with properties
            gj = folium.GeoJson(
                line_geom,
                name=f"110kV Line {line_id}",
                tooltip=f"110kV Line {line_id}",
                popup=folium.Popup(popup_content, max_width=300),
                style_function=lambda x, c=color, w=weight, o=opacity: {
                    'color': c, 'weight': w, 'opacity': o
                }
            )

            # Add to 110kV group
            gj.add_to(pypsa_110kv)

    # Process DC links if provided
    if dc_links_gdf is not None:
        print("Processing DC links...")
        for _, dc_row in dc_links_gdf.iterrows():
            dc_id = str(dc_row['id'])
            dc_geom = dc_row.geometry

            if dc_geom is None:
                continue

            # Style for DC links - black with dashed pattern
            color = '#000000'  # Black
            weight = 4
            opacity = 0.85
            dash_array = '5, 10'  # Dashed line pattern

            # Get voltage level
            dc_voltage = _safe_int(dc_row.get('voltage', 0))

            # Create popup content
            popup_content = f"""
            <div style="font-family: Arial; max-width: 300px;">
                <h4>DC Link {dc_id}</h4>
                <b>Name:</b> {dc_row.get('name', '')}<br>
                <b>Voltage:</b> {dc_voltage} kV (DC)<br>
                <b>Length:</b> {dc_row.get('length_km', 0):.2f} km<br>
                <b>Capacity:</b> {dc_row.get('capacity_mw', 0)} MW<br>
                <hr style="margin: 5px 0;">
                <b>Type:</b> HVDC Link<br>
            </div>
            """

            # Create GeoJson with properties
            gj = folium.GeoJson(
                dc_geom,
                name=f"DC Link {dc_id}",
                tooltip=f"DC Link {dc_id}",
                popup=folium.Popup(popup_content, max_width=300),
                style_function=lambda x, c=color, w=weight, o=opacity: {
                    'color': c,
                    'weight': w,
                    'opacity': o,
                    'dashArray': dash_array
                }
            )

            # Add to DC links group
            gj.add_to(pypsa_dc_links)

    # Add legend and filtering controls
    legend_html = """
    <div style="position: fixed; bottom: 50px; left: 50px; width: 210px; 
                background-color: white; border: 2px solid grey; z-index: 1000; padding: 10px; 
                font-family: Arial; font-size: 12px;">
        <div style="font-weight: bold; margin-bottom: 5px;">Legend</div>
        <div><span style="background-color: #007200; display: inline-block; width: 20px; height: 3px;"></span> Matched JAO</div>
        <div><span style="background-color: #CC0000; display: inline-block; width: 20px; height: 3px;"></span> Unmatched JAO</div>
        <div><span style="background-color: #0070FF; display: inline-block; width: 20px; height: 6px;"></span> Matched PyPSA</div>
        <div><span style="background-color: #FF8000; display: inline-block; width: 20px; height: 5px;"></span> Unmatched PyPSA</div>
        <div><span style="background-color: #8A2BE2; display: inline-block; width: 20px; height: 3px;"></span> 110kV Lines</div>
        <div><span style="background-color: #000000; display: inline-block; width: 20px; height: 4px; border-top: 1px dashed #000;"></span> DC Links</div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    # Add filtering interface
    filter_html = """
    <div style="position: fixed; top: 10px; left: 50px; z-index: 1000;">
        <div style="background-color: white; padding: 12px; border-radius: 4px; box-shadow: 0 1px 5px rgba(0,0,0,0.4); max-width: 330px;">
            <div style="font-weight: bold; margin-bottom: 10px; font-size: 14px;">GRID VISUALIZATION FILTERING</div>

            <div style="display: flex; margin-bottom: 12px;">
                <div style="flex: 1;">
                    <div style="font-weight: bold; margin-bottom: 5px;">Data Source:</div>
                    <div>
                        <input type="checkbox" id="jao-filter" checked onchange="applyFilters()">
                        <label for="jao-filter">JAO</label>
                    </div>
                    <div>
                        <input type="checkbox" id="pypsa-filter" checked onchange="applyFilters()">
                        <label for="pypsa-filter">PyPSA</label>
                    </div>
                </div>

                <div style="flex: 1;">
                    <div style="font-weight: bold; margin-bottom: 5px;">Match Status:</div>
                    <div>
                        <input type="checkbox" id="matched-filter" checked onchange="applyFilters()">
                        <label for="matched-filter">Matched</label>
                    </div>
                    <div>
                        <input type="checkbox" id="unmatched-filter" checked onchange="applyFilters()">
                        <label for="unmatched-filter">Unmatched</label>
                    </div>
                </div>

                <div style="flex: 1;">
                    <div style="font-weight: bold; margin-bottom: 5px;">AC Voltage:</div>
                    <div>
                        <input type="checkbox" id="400kv-filter" checked onchange="applyFilters()">
                        <label for="400kv-filter">400kV</label>
                    </div>
                    <div>
                        <input type="checkbox" id="220kv-filter" checked onchange="applyFilters()">
                        <label for="220kv-filter">220kV</label>
                    </div>
                </div>
            </div>

            <div style="display: flex; margin-bottom: 12px;">
                <div style="flex: 1;">
                    <div style="font-weight: bold; margin-bottom: 5px;">Additional Lines:</div>
                    <div>
                        <input type="checkbox" id="110kv-filter" onchange="applyFilters()">
                        <label for="110kv-filter">110kV Lines</label>
                    </div>
                    <div>
                        <input type="checkbox" id="dc-filter" onchange="applyFilters()">
                        <label for="dc-filter">DC Links</label>
                    </div>
                </div>

                <div style="flex: 2;">
                    <div style="font-weight: bold; margin-bottom: 5px;">Quick Filters:</div>
                    <div>
                        <button onclick="showACOnly()">AC Lines Only</button>
                        <button onclick="showAllTypes()">All Types</button>
                    </div>
                </div>
            </div>

            <div style="margin-bottom: 10px;">
                <button onclick="checkAll(true)">Show All</button>
                <button onclick="checkAll(false)">Hide All</button>
                <button onclick="resetFilters()">Reset Filters</button>
            </div>

            <div style="font-weight: bold; margin-bottom: 5px;">Search:</div>
            <div>
                <input type="text" id="line-search" placeholder="Enter line ID or name..." style="width:170px; padding:4px;">
                <button onclick="searchLine()">Search</button>
                <button onclick="clearSearch()">Clear</button>
            </div>
        </div>
    </div>
    """

    # JavaScript for filtering and interaction
    filter_script = """
    <script>
    // Function to check or uncheck all filters
    function checkAll(checked) {
        document.getElementById('jao-filter').checked = checked;
        document.getElementById('pypsa-filter').checked = checked;
        document.getElementById('matched-filter').checked = checked;
        document.getElementById('unmatched-filter').checked = checked;
        document.getElementById('400kv-filter').checked = checked;
        document.getElementById('220kv-filter').checked = checked;
        document.getElementById('110kv-filter').checked = checked;
        document.getElementById('dc-filter').checked = checked;
        applyFilters();
    }

    // Reset to default view
    function resetFilters() {
        document.getElementById('jao-filter').checked = true;
        document.getElementById('pypsa-filter').checked = true;
        document.getElementById('matched-filter').checked = true;
        document.getElementById('unmatched-filter').checked = true;
        document.getElementById('400kv-filter').checked = true;
        document.getElementById('220kv-filter').checked = true;
        document.getElementById('110kv-filter').checked = false;
        document.getElementById('dc-filter').checked = false;
        applyFilters();
    }

    // Show only AC lines
    function showACOnly() {
        document.getElementById('jao-filter').checked = true;
        document.getElementById('pypsa-filter').checked = true;
        document.getElementById('matched-filter').checked = true;
        document.getElementById('unmatched-filter').checked = true;
        document.getElementById('400kv-filter').checked = true;
        document.getElementById('220kv-filter').checked = true;
        document.getElementById('110kv-filter').checked = false;
        document.getElementById('dc-filter').checked = false;
        applyFilters();
    }

    // Show all line types
    function showAllTypes() {
        document.getElementById('jao-filter').checked = true;
        document.getElementById('pypsa-filter').checked = true;
        document.getElementById('matched-filter').checked = true;
        document.getElementById('unmatched-filter').checked = true;
        document.getElementById('400kv-filter').checked = true;
        document.getElementById('220kv-filter').checked = true;
        document.getElementById('110kv-filter').checked = true;
        document.getElementById('dc-filter').checked = true;
        applyFilters();
    }

    // Function to click layer controls
    function setLayerVisibility(layerName, shouldBeVisible) {
        setTimeout(function() {
            var layerControls = document.querySelectorAll('.leaflet-control-layers-overlays label');
            for (var i = 0; i < layerControls.length; i++) {
                var label = layerControls[i];
                var text = label.textContent.trim();
                var checkbox = label.querySelector('input');
                if (text === layerName) {
                    if (checkbox.checked !== shouldBeVisible) {
                        checkbox.click();
                    }
                    break;
                }
            }
        }, 10);
    }

    // Main filtering function that applies all selected filters
    function applyFilters() {
        // Get filter states
        var showJAO = document.getElementById('jao-filter').checked;
        var showPyPSA = document.getElementById('pypsa-filter').checked;
        var showMatched = document.getElementById('matched-filter').checked;
        var showUnmatched = document.getElementById('unmatched-filter').checked;
        var show400kV = document.getElementById('400kv-filter').checked;
        var show220kV = document.getElementById('220kv-filter').checked;
        var show110kV = document.getElementById('110kv-filter').checked;
        var showDC = document.getElementById('dc-filter').checked;

        // Apply JAO filters
        setLayerVisibility("JAO-Matched-400kV", showJAO && showMatched && show400kV);
        setLayerVisibility("JAO-Matched-220kV", showJAO && showMatched && show220kV);
        setLayerVisibility("JAO-Unmatched-400kV", showJAO && showUnmatched && show400kV);
        setLayerVisibility("JAO-Unmatched-220kV", showJAO && showUnmatched && show220kV);

        // Apply PyPSA filters
        setLayerVisibility("PyPSA-Matched-400kV", showPyPSA && showMatched && show400kV);
        setLayerVisibility("PyPSA-Matched-220kV", showPyPSA && showMatched && show220kV);
        setLayerVisibility("PyPSA-Unmatched-400kV", showPyPSA && showUnmatched && show400kV);
        setLayerVisibility("PyPSA-Unmatched-220kV", showPyPSA && showUnmatched && show220kV);

        // Apply additional network layers
        setLayerVisibility("PyPSA-110kV", show110kV);
        setLayerVisibility("PyPSA-DC-Links", showDC);
    }

    // Search for a specific line ID or name
    function searchLine() {
        var searchText = document.getElementById('line-search').value.trim().toLowerCase();
        if (!searchText) return;

        var found = false;
        var foundLayers = [];

        // Search through all layers
        map.eachLayer(function(layer) {
            // Look for GeoJSON layers with matching ID in name or tooltip
            if (layer.feature && layer.feature.properties) {
                var props = layer.feature.properties;
                var name = (props.name || '').toLowerCase();
                var tooltip = (props.tooltip || '').toLowerCase();
                var popup = layer._popup ? layer._popup._content.toLowerCase() : '';

                if (name.includes(searchText) || tooltip.includes(searchText) || popup.includes(searchText)) {
                    found = true;
                    foundLayers.push(layer);

                    // Make sure the layer's group is visible
                    var layerGroup = layer._group;
                    if (layerGroup && layerGroup._map === null) {
                        // Find the group name and enable it
                        for (var key in map._layers) {
                            if (map._layers[key] === layerGroup) {
                                // Enable this layer in the filter UI
                                enableLayerInUI(layerGroup.options.name);
                                break;
                            }
                        }
                    }

                    // Highlight the found line
                    var originalStyle = layer.options || {};
                    var originalColor = layer.options.color || '#000';
                    var originalWeight = layer.options.weight || 3;
                    var originalDashArray = layer.options.dashArray || '';

                    // Store original style for reset
                    layer._originalStyle = {
                        color: originalColor,
                        weight: originalWeight,
                        dashArray: originalDashArray
                    };

                    // Highlight with bright pink and increased width
                    layer.setStyle({
                        color: '#FF00FF',
                        weight: originalWeight + 2,
                        dashArray: originalDashArray  // Maintain dash pattern if it exists
                    });

                    // Open popup if possible
                    if (layer.getLatLng) {
                        map.setView(layer.getLatLng(), 10);
                        layer.openPopup();
                    } else if (layer.getBounds) {
                        map.fitBounds(layer.getBounds());
                        if (layer.openPopup) {
                            layer.openPopup();
                        }
                    }
                }
            }
        });

        if (found) {
            // If multiple matches, fit bounds to include all
            if (foundLayers.length > 1) {
                var group = new L.featureGroup(foundLayers);
                map.fitBounds(group.getBounds());
            }

            alert('Found ' + foundLayers.length + ' matching lines. They are highlighted in pink.');
        } else {
            alert('No lines found matching: ' + searchText);
        }
    }

    // Enable a specific layer in the UI based on its name
    function enableLayerInUI(layerName) {
        // Parse the layer name to determine what filters need to be enabled
        if (layerName.includes("JAO")) {
            document.getElementById('jao-filter').checked = true;
        }
        if (layerName.includes("PyPSA")) {
            document.getElementById('pypsa-filter').checked = true;
        }
        if (layerName.includes("Matched")) {
            document.getElementById('matched-filter').checked = true;
        }
        if (layerName.includes("Unmatched")) {
            document.getElementById('unmatched-filter').checked = true;
        }
        if (layerName.includes("400kV")) {
            document.getElementById('400kv-filter').checked = true;
        }
        if (layerName.includes("220kV")) {
            document.getElementById('220kv-filter').checked = true;
        }
        if (layerName.includes("110kV")) {
            document.getElementById('110kv-filter').checked = true;
        }
        if (layerName.includes("DC-Links")) {
            document.getElementById('dc-filter').checked = true;
        }

        // Apply the updated filters
        applyFilters();
    }

    // Clear search results
    function clearSearch() {
        // Reset all highlighted layers
        map.eachLayer(function(layer) {
            if (layer._originalStyle) {
                layer.setStyle(layer._originalStyle);
                delete layer._originalStyle;
            }
        });

        // Clear search box
        document.getElementById('line-search').value = '';
    }
    </script>
    """

    # Add HTML elements to the map
    m.get_root().html.add_child(folium.Element(filter_html))
    m.get_root().html.add_child(folium.Element(filter_script))

    # Add all feature groups to map
    m.add_child(jao_matched_400)
    m.add_child(jao_matched_220)
    m.add_child(jao_unmatched_400)
    m.add_child(jao_unmatched_220)
    m.add_child(pypsa_matched_400)
    m.add_child(pypsa_matched_220)
    m.add_child(pypsa_unmatched_400)
    m.add_child(pypsa_unmatched_220)
    m.add_child(pypsa_110kv)
    m.add_child(pypsa_dc_links)

    # Add layer control
    folium.LayerControl().add_to(m)

    # Save the map
    m.save(output_path)
    print(f"Enhanced visualization saved to {output_path}")
    return output_path


def create_specialized_visualization(gdf, output_path, line_type="110kV"):
    """
    Create a specialized visualization for a specific type of grid lines (110kV or DC).

    Parameters:
    -----------
    gdf : GeoDataFrame
        The transmission line data with geometries (either 110kV or DC links)
    output_path : str
        Path to save the HTML visualization
    line_type : str
        Type of lines being visualized ("110kV" or "DC")
    """
    import folium
    from folium.plugins import MeasureControl

    print(f"Creating specialized {line_type} visualization at {output_path}...")

    # Create base map centered on Germany
    m = folium.Map(
        location=[51.1657, 10.4515],
        zoom_start=6,
        control_scale=True,
        tiles="OpenStreetMap"
    )
    m.add_child(MeasureControl())

    # Create feature group for the lines
    feature_group = folium.FeatureGroup(name=f"PyPSA-{line_type}", show=True)

    # Style for different line types
    if line_type == "DC":
        color = '#000000'  # Black for DC
        weight = 4
        dash_array = '5, 10'  # Dashed line for DC
    else:  # 110kV
        color = '#8A2BE2'  # BlueViolet for 110kV
        weight = 3
        dash_array = None

    # Count processed lines
    processed_count = 0

    # Process each line
    for _, row in gdf.iterrows():
        line_id = str(row.get('id', ''))
        if not line_id or row.geometry is None:
            continue

        # Get voltage
        if line_type == "DC":
            voltage = "DC"
            is_dc = True
        else:
            voltage = row.get('voltage', 110)
            is_dc = False

        # Get length in km
        length_km = row.get('length_km', 0)
        if not length_km:
            # Try to convert from meters
            length_m = row.get('length', 0)
            if length_m > 1000:  # Assume it's in meters
                length_km = length_m / 1000
            else:
                length_km = length_m

        # Create popup content
        if is_dc:
            popup_content = f"""
            <div style="font-family: Arial; max-width: 300px;">
                <h4>DC Link {line_id}</h4>
                <b>Voltage:</b> DC<br>
                <b>Length:</b> {length_km:.2f} km<br>
                <b>Capacity:</b> {row.get('capacity_mw', 0)} MW<br>
                <hr style="margin: 5px 0;">
                <b>Type:</b> HVDC Link<br>
            </div>
            """
        else:
            popup_content = f"""
            <div style="font-family: Arial; max-width: 300px;">
                <h4>110kV Line {line_id}</h4>
                <b>Voltage:</b> {voltage} kV<br>
                <b>Length:</b> {length_km:.2f} km<br>
                <b>Circuits:</b> {row.get('circuits', 1)}<br>
                <hr style="margin: 5px 0;">
                <b>Electrical Parameters:</b><br>
                <b>r:</b> {row.get('r_per_km', 'N/A')} Ω/km<br>
                <b>x:</b> {row.get('x_per_km', 'N/A')} Ω/km<br>
                <b>b:</b> {row.get('b_per_km', 'N/A')} μS/km<br>
            </div>
            """

        # Style options
        style = {
            'color': color,
            'weight': weight,
            'opacity': 0.8
        }
        if dash_array:
            style['dashArray'] = dash_array

        # Create the line and add to map
        folium.GeoJson(
            row.geometry,
            name=f"{line_type} Line {line_id}",
            tooltip=f"{line_type} Line {line_id} - {length_km:.2f} km",
            popup=folium.Popup(popup_content, max_width=300),
            style_function=lambda x, s=style: s
        ).add_to(feature_group)

        processed_count += 1

    # Add feature group to map
    feature_group.add_to(m)

    # Add layer control
    folium.LayerControl().add_to(m)

    # Add info box
    info_html = f"""
    <div style="position: fixed; bottom: 50px; left: 50px; width: 250px; 
                background-color: white; border: 2px solid grey; z-index: 1000; padding: 10px; 
                font-family: Arial; font-size: 14px;">
        <div style="font-weight: bold; margin-bottom: 10px;">{line_type} Lines Visualization</div>
        <div>Total lines: {processed_count}</div>
        <div style="margin-top: 8px;">
            <span style="background-color: {color}; display: inline-block; width: 20px; height: {weight}px;
                  {f'border-top: 1px dashed {color};' if dash_array else ''}"></span>
            <span style="margin-left: 6px;">{line_type} Line</span>
        </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(info_html))

    # Save the map
    m.save(output_path)
    print(f"Specialized {line_type} visualization saved to {output_path} with {processed_count} lines")
    return output_path


def create_osm_pypsa_match_visualization(osm_gdf, pypsa_gdf, matching_results, output_path):
    """
    Create an interactive Leaflet visualization showing matches between OSM and PyPSA grid lines.

    Parameters
    ----------
    osm_gdf : GeoDataFrame
        OpenStreetMap lines with geometry in lon/lat (EPSG:4326).
        Should include columns: 'id', 'voltage', 'length_km' (or 'length')
    pypsa_gdf : GeoDataFrame
        PyPSA lines with geometry in lon/lat (EPSG:4326).
        Should include columns: 'id', 'v_nom', 'length' (typically in meters)
    matching_results : list[dict]
        Results of the matching step. Each dict should include:
        - 'matched' (bool): Whether this represents a match
        - 'osm_id' (str/int): ID of the OSM line
        - 'pypsa_ids' (list[str] or str): PyPSA IDs that match this OSM line
        - 'match_type' (str): Type of match ('regular', 'geometric', 'parallel', etc.)
        - 'match_quality' (str, optional): Quality indicator
    output_path : str | Path
        Path where the HTML file will be written

    Returns
    -------
    str
        Path to the generated HTML file
    """
    import json
    import folium
    from pathlib import Path
    import pandas as pd
    from folium.plugins import MarkerCluster

    # Ensure output_path is a Path object
    output_path = Path(output_path)

    print(f"Creating OSM-PyPSA match visualization at {output_path}...")

    # Convert to WGS84 if needed
    def ensure_wgs84(gdf):
        """Reproject to EPSG:4326 if needed"""
        if gdf is None:
            return gdf
        if hasattr(gdf, "crs") and gdf.crs:
            if getattr(gdf.crs, "to_epsg", lambda: None)() == 4326:
                return gdf
            if hasattr(gdf, "to_crs"):
                return gdf.to_crs(4326)
        return gdf

    osm_gdf = ensure_wgs84(osm_gdf)
    pypsa_gdf = ensure_wgs84(pypsa_gdf)

    # Create base map - find center of all geometries
    bounds = osm_gdf.total_bounds
    center_lat = (bounds[1] + bounds[3]) / 2
    center_lon = (bounds[0] + bounds[2]) / 2

    m = folium.Map(location=[center_lat, center_lon], zoom_start=6,
                   tiles='CartoDB positron', control_scale=True)

    # Prepare match type information
    match_types = {
        "regular": {"color": "#00AA00", "label": "Regular Match"},
        "geometric": {"color": "#0000FF", "label": "Geometric Match"},
        "parallel": {"color": "#FFA500", "label": "Parallel Circuit"},
        "parallel_voltage": {"color": "#AA00AA", "label": "Parallel Voltage"},
        "duplicate": {"color": "#FF0000", "label": "Duplicate"},
        "unmatched": {"color": "#888888", "label": "Unmatched"}
    }

    # Create feature groups for different match types
    feature_groups = {
        match_type: folium.FeatureGroup(name=info["label"])
        for match_type, info in match_types.items()
    }

    # Add unmatched feature groups
    feature_groups["osm_unmatched"] = folium.FeatureGroup(name="Unmatched OSM")
    feature_groups["pypsa_unmatched"] = folium.FeatureGroup(name="Unmatched PyPSA")

    # Track which OSM and PyPSA lines are matched
    matched_osm_ids = set()
    matched_pypsa_ids = set()

    # Add matched lines to map
    for result in matching_results:
        if not result.get("matched", False):
            continue

        osm_id = str(result.get("osm_id", ""))
        if not osm_id:
            continue

        matched_osm_ids.add(osm_id)

        # Get match type
        match_type = result.get("match_type", "regular").lower()
        if match_type not in match_types:
            match_type = "regular"

        # Get matched PyPSA IDs
        pypsa_ids = result.get("pypsa_ids", [])
        if isinstance(pypsa_ids, str):
            pypsa_ids = [pid.strip() for pid in pypsa_ids.replace(",", ";").split(";") if pid.strip()]

        matched_pypsa_ids.update(pypsa_ids)

        # Get OSM geometry
        try:
            osm_row = osm_gdf[osm_gdf["id"] == osm_id].iloc[0]
            osm_geom = osm_row.geometry

            # Convert to list of coordinates for folium
            osm_coords = [[y, x] for x, y in zip(*osm_geom.xy)]

            # Add to map with popup info
            voltage = osm_row.get("voltage", "Unknown")
            length = osm_row.get("length_km", osm_row.get("length", "Unknown"))

            popup_text = f"""
            <b>OSM ID:</b> {osm_id}<br>
            <b>Voltage:</b> {voltage} kV<br>
            <b>Length:</b> {length} km<br>
            <b>Match Type:</b> {match_types[match_type]['label']}<br>
            <b>Matched PyPSA IDs:</b> {', '.join(pypsa_ids)}
            """

            folium.PolyLine(
                osm_coords,
                color=match_types[match_type]["color"],
                weight=3,
                opacity=0.8,
                popup=folium.Popup(popup_text, max_width=300)
            ).add_to(feature_groups[match_type])

        except (IndexError, KeyError) as e:
            print(f"Error adding OSM line {osm_id}: {e}")

    # Add unmatched OSM lines
    for _, row in osm_gdf.iterrows():
        osm_id = str(row.get("id", ""))
        if osm_id not in matched_osm_ids:
            try:
                osm_geom = row.geometry
                osm_coords = [[y, x] for x, y in zip(*osm_geom.xy)]

                voltage = row.get("voltage", "Unknown")
                length = row.get("length_km", row.get("length", "Unknown"))

                popup_text = f"""
                <b>OSM ID:</b> {osm_id}<br>
                <b>Voltage:</b> {voltage} kV<br>
                <b>Length:</b> {length} km<br>
                <b>Status:</b> Unmatched
                """

                folium.PolyLine(
                    osm_coords,
                    color="#888888",
                    weight=2,
                    opacity=0.6,
                    popup=folium.Popup(popup_text, max_width=300)
                ).add_to(feature_groups["osm_unmatched"])

            except Exception as e:
                print(f"Error adding unmatched OSM line {osm_id}: {e}")

    # Add unmatched PyPSA lines
    for _, row in pypsa_gdf.iterrows():
        pypsa_id = str(row.get("id", ""))
        if pypsa_id not in matched_pypsa_ids:
            try:
                pypsa_geom = row.geometry
                pypsa_coords = [[y, x] for x, y in zip(*pypsa_geom.xy)]

                voltage = row.get("v_nom", "Unknown")
                length = row.get("length", "Unknown")
                if isinstance(length, (int, float)) and length > 1000:
                    length = length / 1000  # Convert to km if in meters

                popup_text = f"""
                <b>PyPSA ID:</b> {pypsa_id}<br>
                <b>Voltage:</b> {voltage} kV<br>
                <b>Length:</b> {length} km<br>
                <b>Status:</b> Unmatched
                """

                folium.PolyLine(
                    pypsa_coords,
                    color="#AAAAAA",
                    weight=2,
                    opacity=0.6,
                    dashArray="5, 5",
                    popup=folium.Popup(popup_text, max_width=300)
                ).add_to(feature_groups["pypsa_unmatched"])

            except Exception as e:
                print(f"Error adding unmatched PyPSA line {pypsa_id}: {e}")

    # Add all feature groups to map
    for group in feature_groups.values():
        group.add_to(m)

    # Add layer control
    folium.LayerControl(collapsed=False).add_to(m)

    # Add map title and legend
    title_html = '''
    <div style="position: fixed; 
                top: 10px; left: 50px; width: 300px; height: 30px; 
                background-color: white; border-radius: 5px;
                z-index: 900; font-size: 18px; font-weight: bold;
                text-align: center; padding: 5px;">
        OSM-PyPSA Line Matches
    </div>
    '''
    m.get_root().html.add_child(folium.Element(title_html))

    # Add legend
    legend_html = '''
    <div style="position: fixed; 
                bottom: 50px; right: 50px; 
                background-color: white; border-radius: 5px;
                z-index: 900; padding: 10px;
                font-size: 14px;">
        <div style="font-weight: bold; margin-bottom: 5px;">Legend</div>
    '''

    for match_type, info in match_types.items():
        legend_html += f'''
        <div>
            <span style="background-color: {info['color']}; 
                         display: inline-block; width: 15px; height: 15px;
                         margin-right: 5px;"></span>
            {info['label']}
        </div>
        '''

    legend_html += '''
        <div>
            <span style="background-color: #AAAAAA; 
                         display: inline-block; width: 15px; height: 15px;
                         margin-right: 5px; border: 1px dashed black;"></span>
            Unmatched PyPSA
        </div>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))

    # Save the map
    m.save(output_path)
    print(f"Map saved to {output_path}")

    return str(output_path)
