import os
import logging
import json
import folium
from folium.plugins import MeasureControl
from shapely.geometry import Point, box
import pandas as pd
import geopandas as gpd
logger = logging.getLogger(__name__)

from src.utils.topology import reconnect_segments



def add_connection_notification_to_html(html_file):
    """
    Add a fixed notification about the special connection that doesn't rely on JavaScript.
    This is the simplest possible approach that will always work.
    """
    try:
        # Read the HTML file
        with open(html_file, 'r', encoding='utf-8') as f:
            html_content = f.read()

        # Create a simple HTML notification that shows immediately and doesn't use JavaScript
        notification_html = """
        <div style="
            position: fixed;
            top: 100px;
            left: 50%;
            transform: translateX(-50%);
            padding: 15px;
            background-color: rgba(255, 235, 59, 0.9);
            border: 2px solid #e65100;
            border-radius: 10px;
            box-shadow: 0 0 15px rgba(0,0,0,0.2);
            color: black;
            font-weight: bold;
            z-index: 9999;
            text-align: center;
            max-width: 90%;
        ">
            <div style="font-size: 18px; margin-bottom: 8px;">⚡ Special Connection ⚡</div>
            <div>Line_10688 connects to Line_10682</div>
            <div style="margin-top: 5px;">Coordinates: [53.343675, 7.972087]</div>
            <button style="
                margin-top: 10px;
                padding: 5px 10px;
                background: #e65100;
                color: white;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                font-weight: bold;
            " onclick="this.parentNode.style.display='none'">Close</button>
        </div>
        """

        # Find the end of the <body> tag to insert our notification
        if '<body>' in html_content:
            # Insert right after the opening <body> tag
            html_content = html_content.replace('<body>', '<body>\n' + notification_html)
        else:
            # As a fallback, add to the end of the head
            if '</head>' in html_content:
                html_content = html_content.replace('</head>', notification_html + '\n</head>')
            else:
                # Last resort: add at the beginning of the HTML
                html_content = notification_html + '\n' + html_content

        ##Write the modified HTML back to the file
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)

        return True
    except Exception as e:
        logger.error(f"Error adding connection notification to HTML: {e}")
        return False

def _coords_from_linestring(linestring):
    # Leaflet order is [lat, lon] == [y, x]
    return [[y, x] for x, y in linestring.coords]


def _best_match_table(*match_dfs):
    table = {}
    for df in match_dfs:
        if df is None or df.empty:
            continue
        ranked = (df.sort_values(["network_id", "overlap_km"],
                                 ascending=[True, False]))
        for net_id, grp in ranked.groupby("network_id", sort=False):
            table.setdefault(str(net_id), []).extend(
                grp["dlr_id"].astype(str)
            )
    return table



def create_comprehensive_map(dlr_lines, network_lines, matches_dlr, pypsa_lines=None, pypsa_lines_new=None,
                             matches_pypsa=None, matches_pypsa_new=None, fifty_hertz_lines=None, tennet_lines=None,
                             matches_fifty_hertz=None, matches_tennet=None, germany_gdf=None,
                             output_file='comprehensive_grid_map.html', matched_ids=None,
                             dlr_lines_germany_count=None, network_lines_germany_count=None,
                             pypsa_lines_germany_count=None, pypsa_lines_new_germany_count=None,
                             fifty_hertz_lines_germany_count=None, tennet_lines_germany_count=None,
                             detect_connections=False):
    """
    Create a completely standalone map with no reliance on folium.
    """
    import os
    import json
    import logging
    from shapely.geometry import Point, box
    import pandas as pd
    import geopandas as gpd

    logger = logging.getLogger(__name__)
    logger.info("Creating standalone map with direct JavaScript...")

    # Format match rate
    def format_rate(numerator, denominator):
        if denominator == 0 or denominator is None:
            return "0.0"
        try:
            return f"{(numerator / denominator * 100):.1f}"
        except:
            return "0.0"

    # Use provided counts if available, otherwise calculate from dataframes
    if dlr_lines_germany_count is None:
        dlr_lines_germany_count = len(dlr_lines) if dlr_lines is not None else 0

    if network_lines_germany_count is None:
        network_lines_germany_count = len(network_lines) if network_lines is not None else 0

    if pypsa_lines_germany_count is None:
        pypsa_lines_germany_count = len(pypsa_lines) if pypsa_lines is not None else 0

    if fifty_hertz_lines_germany_count is None:
        fifty_hertz_lines_germany_count = len(fifty_hertz_lines) if fifty_hertz_lines is not None else 0

    if tennet_lines_germany_count is None:
        tennet_lines_germany_count = len(tennet_lines) if tennet_lines is not None else 0

    # Calculate map center (center of Germany)
    centroid_y = 51.1657  # latitude
    centroid_x = 10.4515  # longitude

    # Extract matched IDs from the match dataframes
    matched_dlr_ids = set()
    all_matched_network_ids = set()
    matched_pypsa_eur_ids = set()
    matched_fifty_hertz_ids = set()
    matched_tennet_ids = set()

    # If matched_ids is provided, use that directly
    if matched_ids is not None:
        matched_dlr_ids = matched_ids.get('dlr', set())
        matched_pypsa_eur_ids = matched_ids.get('pypsa', set())
        matched_fifty_hertz_ids = matched_ids.get('fifty_hertz', set())
        matched_tennet_ids = matched_ids.get('tennet', set())
        all_matched_network_ids = matched_ids.get('network', set())
    # Otherwise extract from dataframes
    else:
        # Process matches_dlr
        if matches_dlr is not None and not matches_dlr.empty:
            if 'dlr_id' in matches_dlr.columns and 'network_id' in matches_dlr.columns:
                # Convert to strings for consistent comparison
                matches_dlr['dlr_id'] = matches_dlr['dlr_id'].astype(str)
                matches_dlr['network_id'] = matches_dlr['network_id'].astype(str)
                matched_dlr_ids = set(matches_dlr['dlr_id'])
                network_ids = set(matches_dlr['network_id'])
                all_matched_network_ids.update(network_ids)

        # Similarly process other match dataframes
        if matches_pypsa is not None and not matches_pypsa.empty:
            if 'dlr_id' in matches_pypsa.columns and 'network_id' in matches_pypsa.columns:
                matches_pypsa['dlr_id'] = matches_pypsa['dlr_id'].astype(str)
                matches_pypsa['network_id'] = matches_pypsa['network_id'].astype(str)
                matched_pypsa_eur_ids = set(matches_pypsa['dlr_id'])
                network_ids = set(matches_pypsa['network_id'])
                all_matched_network_ids.update(network_ids)

        if matches_fifty_hertz is not None and not matches_fifty_hertz.empty:
            if 'dlr_id' in matches_fifty_hertz.columns and 'network_id' in matches_fifty_hertz.columns:
                matches_fifty_hertz['dlr_id'] = matches_fifty_hertz['dlr_id'].astype(str)
                matches_fifty_hertz['network_id'] = matches_fifty_hertz['network_id'].astype(str)
                matched_fifty_hertz_ids = set(matches_fifty_hertz['dlr_id'])
                network_ids = set(matches_fifty_hertz['network_id'])
                all_matched_network_ids.update(network_ids)

        if matches_tennet is not None and not matches_tennet.empty:
            if 'dlr_id' in matches_tennet.columns and 'network_id' in matches_tennet.columns:
                matches_tennet['dlr_id'] = matches_tennet['dlr_id'].astype(str)
                matches_tennet['network_id'] = matches_tennet['network_id'].astype(str)
                matched_tennet_ids = set(matches_tennet['dlr_id'])
                network_ids = set(matches_tennet['network_id'])
                all_matched_network_ids.update(network_ids)

    # ------------------------------------------------------------------
    # 1. Build {dlr_id: [network_id,…]}, {network_id: [dlr_id,…]}, …
    # ------------------------------------------------------------------
    dlr_matches, network_matches  = {}, {}
    pypsa_matches, fifty_matches, tennet_matches = {}, {}, {}

    def _add_pair(a_dict, a_id, b_id):
        if a_id not in a_dict:
            a_dict[a_id] = []
        if b_id not in a_dict[a_id]:
            a_dict[a_id].append(b_id)

    def _collect_matches(df, a_dict, b_dict):
        if df is None or df.empty:
            return
        for _, r in df.iterrows():
            a_id = str(r["dlr_id"])
            b_id = str(r["network_id"])
            _add_pair(a_dict, a_id, b_id)
            _add_pair(b_dict, b_id, a_id)

    _collect_matches(matches_dlr,        dlr_matches,   network_matches)
    _collect_matches(matches_pypsa,      pypsa_matches, network_matches)
    _collect_matches(matches_fifty_hertz,fifty_matches, network_matches)
    _collect_matches(matches_tennet,     tennet_matches,network_matches)

    def _best_of(dct):
        """return {key: first_item_of_value_list}"""
        return {k: v[0] for k, v in dct.items() if v}

    dlr_best = _best_of(dlr_matches)  # DLR  → best Network
    pypsa_best = _best_of(pypsa_matches)  # PyPSA→ best Network
    net_best = _best_of(network_matches)  # Network→ best DLR / PyPSA


    # Calculate match statistics
    dlr_matched_count = len(matched_dlr_ids)
    dlr_unmatched_count = dlr_lines_germany_count - dlr_matched_count
    dlr_match_rate = format_rate(dlr_matched_count, dlr_lines_germany_count)

    network_matched_count = len(all_matched_network_ids)
    network_unmatched_count = network_lines_germany_count - network_matched_count
    network_match_rate = format_rate(network_matched_count, network_lines_germany_count)

    pypsa_matched_count = len(matched_pypsa_eur_ids)
    pypsa_unmatched_count = pypsa_lines_germany_count - pypsa_matched_count
    pypsa_match_rate = format_rate(pypsa_matched_count, pypsa_lines_germany_count)

    fifty_hertz_matched_count = len(matched_fifty_hertz_ids)
    fifty_hertz_unmatched_count = fifty_hertz_lines_germany_count - fifty_hertz_matched_count
    fifty_hertz_match_rate = format_rate(fifty_hertz_matched_count, fifty_hertz_lines_germany_count)

    tennet_matched_count = len(matched_tennet_ids)
    tennet_unmatched_count = tennet_lines_germany_count - tennet_matched_count
    tennet_match_rate = format_rate(tennet_matched_count, tennet_lines_germany_count)

    # Process lines for the map
    logger.info("Processing lines for map display...")
    net_to_matches = {}

    if matches_dlr is not None and not matches_dlr.empty:
        # sort so the first row of every network_id is the largest overlap
        ranked = (matches_dlr
                  .sort_values(["network_id", "overlap_km"],
                               ascending=[True, False]))

        for net_id, grp in ranked.groupby("network_id", sort=False):
            net_to_matches[str(net_id)] = list(grp["dlr_id"].astype(str))

    # PyPSA → Network
    if matches_pypsa is not None and not matches_pypsa.empty:
        for net_id, grp in matches_pypsa.groupby("network_id"):
            net_to_matches.setdefault(str(net_id), []).extend(
                grp["dlr_id"].astype(str)
            )

    # 50 Hertz → Network
    if matches_fifty_hertz is not None and not matches_fifty_hertz.empty:
        for net_id, grp in matches_fifty_hertz.groupby("network_id"):
            net_to_matches.setdefault(str(net_id), []).extend(
                grp["dlr_id"].astype(str)
            )

    # TenneT → Network
    if matches_tennet is not None and not matches_tennet.empty:
        for net_id, grp in matches_tennet.groupby("network_id"):
            net_to_matches.setdefault(str(net_id), []).extend(
                grp["dlr_id"].astype(str)
            )

    # Process DLR lines
    dlr_lines_data = []
    for idx, row in dlr_lines.iterrows():
        g = row.geometry
        if g is None or g.is_empty:
            continue

        coords = [_coords_from_linestring(seg) for seg in g.geoms] \
            if g.geom_type == "MultiLineString" else _coords_from_linestring(g)
        if not coords:
            continue

        line_id = str(row.get("id", idx))
        dlr_lines_data.append({
            "id": line_id,
            "best": dlr_best.get(line_id, ""),
            "is_matched": line_id in matched_dlr_ids,
            "v_nom": str(row.get("v_nom", "N/A")),
            "s_nom": str(row.get("s_nom", "N/A")),
            "r": str(row.get("r", "N/A")),
            "x": str(row.get("x", "N/A")),
            "b": str(row.get("b", "N/A")),
            "matches": net_to_matches.get(line_id, [])[:1],  # ★ take first only
            "coords": coords
        })

    # -------------------------------------------------------------------
    # 2️⃣  Network lines
    # -------------------------------------------------------------------
    network_lines_data = []
    for idx, row in network_lines.iterrows():
        g = row.geometry
        if g is None or g.is_empty:
            continue

        coords = [_coords_from_linestring(seg) for seg in g.geoms] \
            if g.geom_type == "MultiLineString" else _coords_from_linestring(g)
        if not coords:
            continue

        line_id = str(row.get("id", idx))
        network_lines_data.append({
            "id": line_id,
            "best": net_best.get(line_id, ""),
            "is_matched": line_id in all_matched_network_ids,
            "v_nom": str(row.get("v_nom", "N/A")),
            "s_nom": str(row.get("s_nom", "N/A")),
            "r": str(row.get("r", "N/A")),
            "x": str(row.get("x", "N/A")),
            "b": str(row.get("b", "N/A")),
            "matches": net_to_matches.get(line_id, [])[:1],  # ★ take first only
            "coords": coords
        })

    # -------------------------------------------------------------------
    # 3️⃣  PyPSA-EUR lines
    # -------------------------------------------------------------------
    pypsa_lines_data = []
    if pypsa_lines is not None and not pypsa_lines.empty:
        for idx, row in pypsa_lines.iterrows():
            g = row.geometry
            if g is None or g.is_empty:
                continue
            coords = [_coords_from_linestring(seg) for seg in g.geoms] \
                if g.geom_type == "MultiLineString" else _coords_from_linestring(g)
            if not coords:
                continue

            line_id = str(row.get("id", idx))
            pypsa_lines_data.append({
                "id": line_id,
                "best": pypsa_best.get(line_id, ""),
                "is_matched": line_id in matched_pypsa_eur_ids,
                "v_nom": str(row.get("v_nom", row.get("voltage", "N/A"))),
                "s_nom": str(row.get("s_nom", "N/A")),
                "r": str(row.get("r", "N/A")),
                "x": str(row.get("x", "N/A")),
                "b": str(row.get("b", "N/A")),
                "matches": net_to_matches.get(line_id, [])[:1],
                "coords": coords
            })

    # -------------------------------------------------------------------
    # 4️⃣  50 Hertz lines
    # -------------------------------------------------------------------
    fifty_hertz_lines_data = []


    # -------------------------------------------------------------------
    # 5️⃣  TenneT lines
    # -------------------------------------------------------------------
    tennet_lines_data = []


    # Create JSON data for JavaScript
    map_data = {
        'center': [centroid_y, centroid_x],
        'dlr_lines': dlr_lines_data,
        'network_lines': network_lines_data,
        'pypsa_lines': pypsa_lines_data,
        'fifty_hertz_lines': fifty_hertz_lines_data,
        'tennet_lines': tennet_lines_data
    }

    map_data_json = json.dumps(map_data, separators=(',', ':'))

    # Create standalone HTML
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>German Grid Network Comparison</title>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.7.1/dist/leaflet.css" />
    <style>
        html, body {{
            height: 100%;
            margin: 0;
            padding: 0;
            font-family: Arial, sans-serif;
        }}
        #map {{
            height: 100%;
            width: 100%;
            z-index: 1;
        }}
        #stats-panel {{
            position: fixed;
            top: 10px;
            left: 10px;
            background: white;
            padding: 10px;
            border-radius: 5px;
            box-shadow: 0 0 10px rgba(0,0,0,0.3);
            z-index: 1000;
            max-width: 280px;
            max-height: 80vh;
            overflow-y: auto;
        }}
        #search-panel {{
            position: fixed;
            top: 10px;
            right: 10px;
            background: white;
            padding: 10px;
            border-radius: 5px;
            box-shadow: 0 0 10px rgba(0,0,0,0.3);
            z-index: 1000;
            width: 250px;
        }}
        #filter-panel {{
            position: fixed;
            bottom: 40px;
            right: 10px;
            background: white;
            padding: 10px;
            border-radius: 5px;
            box-shadow: 0 0 10px rgba(0,0,0,0.3);
            z-index: 1000;
            width: 250px;
        }}
        #legend-panel {{
            position: fixed;
            bottom: 10px;
            left: 10px;
            background: white;
            padding: 10px;
            border-radius: 5px;
            box-shadow: 0 0 10px rgba(0,0,0,0.3);
            z-index: 1000;
            max-width: 280px;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            margin-bottom: 5px;
        }}
        .legend-color {{
            width: 20px;
            height: 5px;
            margin-right: 10px;
        }}
        h4 {{
            margin-top: 0;
            margin-bottom: 10px;
        }}
        h5 {{
            margin: 10px 0 5px 0;
        }}
        p {{
            margin: 3px 0;
        }}
        .connection-notification {{
            position: fixed;
            top: 100px;
            left: 50%;
            transform: translateX(-50%);
            background-color: rgba(255, 235, 59, 0.9);
            border: 2px solid #e65100;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 0 15px rgba(0,0,0,0.2);
            z-index: 2000;
            text-align: center;
            max-width: 300px;
        }}
    </style>
</head>
<body>
    <div id="map"></div>

    <div id="stats-panel">
        <h4>Statistics</h4>

        <h5>Network Lines: {network_lines_germany_count}</h5>
        <p>- Matched to any: {network_matched_count} ({network_match_rate}%)</p>
        <p>- Unmatched: {network_unmatched_count}</p>

        <h5>DLR Lines: {dlr_lines_germany_count}</h5>
        <p>- Matched: {dlr_matched_count}</p>
        <p>- Unmatched: {dlr_unmatched_count}</p>
        <p>- Match Rate: {dlr_match_rate}%</p>

        <h5>PyPSA-EUR Lines: {pypsa_lines_germany_count}</h5>
        <p>- Matched: {pypsa_matched_count}</p>
        <p>- Unmatched: {pypsa_unmatched_count}</p>
        <p>- Match Rate: {pypsa_match_rate}%</p>

        <h5>50Hertz Lines: {fifty_hertz_lines_germany_count}</h5>
        <p>- Matched: {fifty_hertz_matched_count}</p>
        <p>- Unmatched: {fifty_hertz_unmatched_count}</p>
        <p>- Match Rate: {fifty_hertz_match_rate}%</p>

        <h5>TenneT Lines: {tennet_lines_germany_count}</h5>
        <p>- Matched: {tennet_matched_count}</p>
        <p>- Unmatched: {tennet_unmatched_count}</p>
        <p>- Match Rate: {tennet_match_rate}%</p>

        <p style="font-style: italic; font-size: 10px; margin-top: 8px;">All statistics are for lines inside Germany only.</p>
    </div>

    <div id="search-panel">
        <h4>Search Lines</h4>
        <input type="text" id="line-search" placeholder="Enter line ID..." style="width: 100%; padding: 5px; margin-bottom: 10px;">
        <div style="margin-bottom: 10px;">
            <input type="radio" id="all-lines" name="line-type" value="all" checked>
            <label for="all-lines">All Lines</label><br>

            <input type="radio" id="dlr-lines" name="line-type" value="dlr">
            <label for="dlr-lines">DLR Lines</label><br>

            <input type="radio" id="network-lines" name="line-type" value="network">
            <label for="network-lines">Network Lines</label><br>

            <input type="radio" id="pypsa-lines" name="line-type" value="pypsa">
            <label for="pypsa-lines">PyPSA Lines</label><br>

            <input type="radio" id="fifty-hertz-lines" name="line-type" value="fifty_hertz">
            <label for="fifty-hertz-lines">50Hertz Lines</label><br>

            <input type="radio" id="tennet-lines" name="line-type" value="tennet">
            <label for="tennet-lines">TenneT Lines</label>
        </div>
        <button id="search-button" style="width: 100%; padding: 5px; background: #4CAF50; color: white; border: none; border-radius: 3px; cursor: pointer;">Search</button>
        <div id="search-results" style="margin-top: 10px; max-height: 200px; overflow-y: auto;"></div>
    </div>

    <div id="filter-panel">
        <h4>Filter Map Lines</h4>
        <div>
            <h5>DLR Lines</h5>
            <input type="checkbox" id="filter-dlr-matched" checked>
            <label for="filter-dlr-matched">Matched</label><br>
            <input type="checkbox" id="filter-dlr-unmatched" checked>
            <label for="filter-dlr-unmatched">Unmatched</label>

            <h5>Network Lines</h5>
            <input type="checkbox" id="filter-network-matched" checked>
            <label for="filter-network-matched">Matched</label><br>
            <input type="checkbox" id="filter-network-unmatched" checked>
            <label for="filter-network-unmatched">Unmatched</label>

            <h5>PyPSA Lines</h5>
            <input type="checkbox" id="filter-pypsa-matched" checked>
            <label for="filter-pypsa-matched">Matched</label><br>
            <input type="checkbox" id="filter-pypsa-unmatched" checked>
            <label for="filter-pypsa-unmatched">Unmatched</label>

            <h5>50Hertz Lines</h5>
            <input type="checkbox" id="filter-fifty-hertz-matched" checked>
            <label for="filter-fifty-hertz-matched">Matched</label><br>
            <input type="checkbox" id="filter-fifty-hertz-unmatched" checked>
            <label for="filter-fifty-hertz-unmatched">Unmatched</label>

            <h5>TenneT Lines</h5>
            <input type="checkbox" id="filter-tennet-matched" checked>
            <label for="filter-tennet-matched">Matched</label><br>
            <input type="checkbox" id="filter-tennet-unmatched" checked>
            <label for="filter-tennet-unmatched">Unmatched</label>
        </div>
        <div style="margin-top: 10px; display: flex; justify-content: space-between;">
            <button id="apply-filter" style="flex: 1; margin-right: 5px; padding: 5px; background: #4CAF50; color: white; border: none; border-radius: 3px; cursor: pointer;">Apply</button>
            <button id="reset-filter" style="flex: 1; margin-left: 5px; padding: 5px; background: #f44336; color: white; border: none; border-radius: 3px; cursor: pointer;">Reset</button>
        </div>
    </div>

    <div id="legend-panel">
        <h4>Map Legend</h4>
        <div class="legend-item">
            <div class="legend-color" style="background-color: blue;"></div>
            <span>DLR Lines (Unmatched)</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background-color: green;"></div>
            <span>DLR Lines (Matched)</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background-color: red;"></div>
            <span>Network Lines (Unmatched)</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background-color: orange;"></div>
            <span>Network Lines (Matched)</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background-color: magenta; border-top: 1px dashed #000;"></div>
            <span>PyPSA Lines (Unmatched)</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background-color: purple; border-top: 1px dashed #000;"></div>
            <span>PyPSA Lines (Matched)</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background-color: green;"></div>
            <span>50Hertz Lines (Unmatched)</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background-color: darkgreen;"></div>
            <span>50Hertz Lines (Matched)</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background-color: blue;"></div>
            <span>TenneT Lines (Unmatched)</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background-color: darkblue;"></div>
            <span>TenneT Lines (Matched)</span>
        </div>
    </div>

    <div class="connection-notification">
        <div style="font-size: 18px; margin-bottom: 8px;">⚡ Special Connection ⚡</div>
        <div>Line_10688 connects to Line_10682</div>
        <div style="margin-top: 5px;">Coordinates: [53.343675, 7.972087]</div>
        <button onclick="this.parentNode.style.display='none'" style="margin-top: 10px; padding: 5px 10px; background: #e65100; color: white; border: none; border-radius: 5px; cursor: pointer; font-weight: bold;">Close</button>
    </div>

    <script src="https://unpkg.com/leaflet@1.7.1/dist/leaflet.js"></script>
    <script>
        // Map data from Python
        const mapData = {map_data_json};

        // Initialize map
        const map = L.map('map').setView(mapData.center, 6);

        // Add tile layer
        L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
            attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
        }}).addTo(map);

        // Line collections for filtering
        const lineCollections = {{
            dlr_matched: [],
            dlr_unmatched: [],
            network_matched: [],
            network_unmatched: [],
            pypsa_matched: [],
            pypsa_unmatched: [],
            fifty_hertz_matched: [],
            fifty_hertz_unmatched: [],
            tennet_matched: [],
            tennet_unmatched: []
        }};

        // Dictionary to store all lines by ID
        const linesById = {{}};

        // Add DLR lines
        mapData.dlr_lines.forEach(line => {{
            const color = line.is_matched ? 'green' : 'blue';
            const weight = line.is_matched ? 3 : 2;
            const opacity = line.is_matched ? 0.8 : 0.6;
            const category = 'dlr_' + (line.is_matched ? 'matched' : 'unmatched');

            const polyline = L.polyline(line.coords, {{
                color: color,
                weight: weight,
                opacity: opacity
            }}).addTo(map);

            const tooltip = `DLR Line ${{line.id}} ${{line.is_matched ? '(matched)' : '(unmatched)'}} - ${{line.v_nom}} kV`;
            polyline.bindTooltip(tooltip);

            const popupContent = `
    <div style="min-width:220px;">
        <h4>DLR Line ${{line.id}}</h4>

        <p><b>Status:</b> ${{line.is_matched ? 'Matched' : 'Unmatched'}}</p>
        <p><b>Voltage:</b> ${{line.v_nom}} kV</p>
        <p><b>s<sub>nom</sub>:</b> ${{line.s_nom}} MW</p>

        <p><b>r / x / b&nbsp;per km:</b><br>
           ${{line.r}} Ω&nbsp;/&nbsp;${{line.x}} Ω&nbsp;/&nbsp;${{line.b}} S
        </p>

        ${{line.best
            ? `<p><b>Best&nbsp;match:</b> ${{line.best}}</p>`
            : ''}}
    </div>`;

            polyline.bindPopup(popupContent);

            // Store line for filtering
            lineCollections[category].push(polyline);

            // Store line by ID for search
            const uniqueId = `dlr_${{line.id}}`;
            linesById[uniqueId] = {{
                line: polyline,
                id: line.id,
                type: 'dlr',
                is_matched: line.is_matched,
                coords: line.coords
            }};

            // Add click handler
            polyline.on('click', function() {{
                highlightLine(uniqueId);
            }});
        }});

        // Add Network lines
        mapData.network_lines.forEach(line => {{
            const color = line.is_matched ? 'orange' : 'red';
            const weight = line.is_matched ? 3 : 2;
            const opacity = line.is_matched ? 0.8 : 0.6;
            const category = 'network_' + (line.is_matched ? 'matched' : 'unmatched');

            const polyline = L.polyline(line.coords, {{
                color: color,
                weight: weight,
                opacity: opacity
            }}).addTo(map);

            const tooltip = `Network Line ${{line.id}} ${{line.is_matched ? '(matched)' : '(unmatched)'}} - ${{line.v_nom}} kV`;
            polyline.bindTooltip(tooltip);

            const popupContent = `
    <div style="min-width:220px;">
        <h4>Network Line ${{line.id}}</h4>

        <p><b>Status:</b> ${{line.is_matched ? 'Matched' : 'Unmatched'}}</p>
        <p><b>Voltage:</b> ${{line.v_nom}} kV</p>
        <p><b>s<sub>nom</sub>:</b> ${{line.s_nom}} MW</p>

        <p><b>r / x / b&nbsp;per km:</b><br>
           ${{line.r}} Ω&nbsp;/&nbsp;${{line.x}} Ω&nbsp;/&nbsp;${{line.b}} S
        </p>

        ${{line.best
            ? `<p><b>Best&nbsp;match:</b> ${{line.best}}</p>`
            : ''}}
    </div>`;


            polyline.bindPopup(popupContent);

            // Store line for filtering
            lineCollections[category].push(polyline);

            // Store line by ID for search
            const uniqueId = `network_${{line.id}}`;
            linesById[uniqueId] = {{
                line: polyline,
                id: line.id,
                type: 'network',
                is_matched: line.is_matched,
                coords: line.coords
            }};

            // Add click handler
            polyline.on('click', function() {{
                highlightLine(uniqueId);
            }});
        }});

        // Add PyPSA lines
        mapData.pypsa_lines.forEach(line => {{
            const color = line.is_matched ? 'purple' : 'magenta';
            const weight = line.is_matched ? 3 : 2;
            const opacity = line.is_matched ? 0.8 : 0.6;
            const category = 'pypsa_' + (line.is_matched ? 'matched' : 'unmatched');

            const polyline = L.polyline(line.coords, {{
                color: color,
                weight: weight,
                opacity: opacity,
                dashArray: "5,5"  // Make PyPSA lines dashed
            }}).addTo(map);

            const tooltip = `PyPSA Line ${{line.id}} ${{line.is_matched ? '(matched)' : '(unmatched)'}} - ${{line.v_nom}} kV`;
            polyline.bindTooltip(tooltip);

            const popupContent = `
    <div style="min-width:220px;">
        <h4>PyPSA Line ${{line.id}}</h4>

        <p><b>Status:</b> ${{line.is_matched ? 'Matched' : 'Unmatched'}}</p>
        <p><b>Voltage:</b> ${{line.v_nom}} kV</p>
        <p><b>s<sub>nom</sub>:</b> ${{line.s_nom}} MW</p>

        <p><b>r / x / b&nbsp;per km:</b><br>
           ${{line.r}} Ω&nbsp;/&nbsp;${{line.x}} Ω&nbsp;/&nbsp;${{line.b}} S
        </p>

        ${{line.best
            ? `<p><b>Best&nbsp;match:</b> ${{line.best}}</p>`
            : ''}}
    </div>`;

            polyline.bindPopup(popupContent);

            // Store line for filtering
            lineCollections[category].push(polyline);

            // Store line by ID for search
            const uniqueId = `pypsa_${{line.id}}`;
            linesById[uniqueId] = {{
                line: polyline,
                id: line.id,
                type: 'pypsa',
                is_matched: line.is_matched,
                coords: line.coords
            }};

            // Add click handler
            polyline.on('click', function() {{
                highlightLine(uniqueId);
            }});
        }});

        // Add 50Hertz lines
        mapData.fifty_hertz_lines.forEach(line => {{
            const color = line.is_matched ? 'darkgreen' : 'green';
            const weight = line.is_matched ? 3 : 2;
            const opacity = line.is_matched ? 0.8 : 0.6;
            const category = 'fifty_hertz_' + (line.is_matched ? 'matched' : 'unmatched');

            const polyline = L.polyline(line.coords, {{
                color: color,
                weight: weight,
                opacity: opacity
            }}).addTo(map);

            const tooltip = `50Hertz Line ${{line.id}} ${{line.is_matched ? '(matched)' : '(unmatched)'}} - ${{line.v_nom}} kV`;
            polyline.bindTooltip(tooltip);

            const popupContent = `
    <div style="min-width:220px;">
        <h4>50Hertz Line ${{line.id}}</h4>

        <p><b>Status:</b> ${{line.is_matched ? 'Matched' : 'Unmatched'}}</p>
        <p><b>Voltage:</b> ${{line.v_nom}} kV</p>
        <p><b>s<sub>nom</sub>:</b> ${{line.s_nom}} MW</p>

        <p><b>r / x / b&nbsp;per km:</b><br>
           ${{line.r}} Ω&nbsp;/&nbsp;${{line.x}} Ω&nbsp;/&nbsp;${{line.b}} S
        </p>

        ${{(line.matches && line.matches.length)
            ? `<p><b>Match&nbsp;ID:</b> ${{line.matches[0]}}</p>`
            : ''}}
    </div>`;
            polyline.bindPopup(popupContent);

            // Store line for filtering
            lineCollections[category].push(polyline);

            // Store line by ID for search
            const uniqueId = `fifty_hertz_${{line.id}}`;
            linesById[uniqueId] = {{
                line: polyline,
                id: line.id,
                type: 'fifty_hertz',
                is_matched: line.is_matched,
                coords: line.coords
            }};

            // Add click handler
            polyline.on('click', function() {{
                highlightLine(uniqueId);
            }});
        }});

        // Add TenneT lines
        mapData.tennet_lines.forEach(line => {{
            const color = line.is_matched ? 'darkblue' : 'blue';
            const weight = line.is_matched ? 3 : 2;
            const opacity = line.is_matched ? 0.8 : 0.6;
            const category = 'tennet_' + (line.is_matched ? 'matched' : 'unmatched');

            const polyline = L.polyline(line.coords, {{
                color: color,
                weight: weight,
                opacity: opacity
            }}).addTo(map);

            const tooltip = `TenneT Line ${{line.id}} ${{line.is_matched ? '(matched)' : '(unmatched)'}} - ${{line.v_nom}} kV`;
            polyline.bindTooltip(tooltip);

            const popupContent = `
    <div style="min-width:220px;">
        <h4>Tennet Line ${{line.id}}</h4>

        <p><b>Status:</b> ${{line.is_matched ? 'Matched' : 'Unmatched'}}</p>
        <p><b>Voltage:</b> ${{line.v_nom}} kV</p>
        <p><b>s<sub>nom</sub>:</b> ${{line.s_nom}} MW</p>

        <p><b>r / x / b&nbsp;per km:</b><br>
           ${{line.r}} Ω&nbsp;/&nbsp;${{line.x}} Ω&nbsp;/&nbsp;${{line.b}} S
        </p>

        ${{(line.matches && line.matches.length)
            ? `<p><b>Match&nbsp;ID:</b> ${{line.matches[0]}}</p>`
            : ''}}
    </div>`;

            // Store line for filtering
            lineCollections[category].push(polyline);

            // Store line by ID for search
            const uniqueId = `tennet_${{line.id}}`;
            linesById[uniqueId] = {{
                line: polyline,
                id: line.id,
                type: 'tennet',
                is_matched: line.is_matched,
                coords: line.coords
            }};

            // Add click handler
            polyline.on('click', function() {{
                highlightLine(uniqueId);
            }});
        }});

        // Add special connection marker
        const connectionPoint = [53.343675, 7.972087];
        const marker = L.circleMarker(connectionPoint, {{
            radius: 8,
            color: '#FF0000',
            fillColor: '#FF0000',
            fillOpacity: 0.8,
            weight: 3
        }}).addTo(map);

        marker.bindPopup(`
            <strong>Connection Point</strong><br>
            Line_10688 connects to Line_10682<br>
            Coordinates: ${{connectionPoint[0]}}, ${{connectionPoint[1]}}
        `);

        // Add click handler to marker
        marker.on('click', function() {{
            // Highlight both connected lines
            highlightLine('dlr_Line_10688');
            setTimeout(() => highlightLine('network_Line_10682'), 1000);
        }});

        // Variables to track current highlight
        let currentHighlight = null;
        let currentHighlightedLineId = null;

        // Function to highlight a line
        function highlightLine(lineId) {{
            // Clear any existing highlight
            if (currentHighlight) {{
                map.removeLayer(currentHighlight);
                currentHighlight = null;
                currentHighlightedLineId = null;
            }}

            // Get line data
            const lineData = linesById[lineId];
            if (!lineData) {{
                console.error("Line not found:", lineId);
                return;
            }}

            // Create highlight polyline
            currentHighlight = L.polyline(lineData.coords, {{
                color: 'white',
                weight: 6,
                opacity: 1.0
            }}).addTo(map);

            currentHighlightedLineId = lineId;

            // Zoom to fit the line
            map.fitBounds(currentHighlight.getBounds(), {{ padding: [50, 50] }});

            // Open popup
            lineData.line.openPopup();
        }}

        // Search functionality
        document.getElementById('search-button').addEventListener('click', function() {{
            const searchText = document.getElementById('line-search').value.trim().toLowerCase();
            const resultsDiv = document.getElementById('search-results');

            if (!searchText) {{
                resultsDiv.innerHTML = '<p>Please enter a line ID to search for.</p>';
                return;
            }}

            // Get selected line type
            let selectedType = 'all';
            const radioButtons = document.querySelectorAll('input[name="line-type"]');
            for (const radio of radioButtons) {{
                if (radio.checked) {{
                    selectedType = radio.value;
                    break;
                }}
            }}

            // Search for lines
            const matches = [];
            for (const lineId in linesById) {{
                const line = linesById[lineId];

                // Filter by type if not "all"
                if (selectedType !== 'all' && line.type !== selectedType) {{
                    continue;
                }}

                // Check if ID contains search text
                if (line.id.toLowerCase().includes(searchText)) {{
                    matches.push(line);
                }}
            }}

            // Display results
            if (matches.length === 0) {{
                resultsDiv.innerHTML = '<p>No lines found matching "' + searchText + '".</p>';
            }} else {{
                // Sort by ID
                matches.sort((a, b) => a.id.localeCompare(b.id, undefined, {{numeric: true}}));

                // Create results HTML
                let resultsHtml = '<p>Found ' + matches.length + ' matching lines:</p><ul>';

                // Display up to 50 results for performance
                const displayCount = Math.min(matches.length, 50);
                for (let i = 0; i < displayCount; i++) {{
                    const line = matches[i];
                    const uniqueId = Object.keys(linesById).find(key => linesById[key] === line);

                    resultsHtml += `
                        <li>
                            <a href="#" onclick="highlightLine('${{uniqueId}}'); return false;">
                                ${{line.type.toUpperCase()}} Line ${{line.id}} ${{line.is_matched ? '(matched)' : '(unmatched)'}}
                            </a>
                        </li>
                    `;
                }}

                if (matches.length > displayCount) {{
                    resultsHtml += '<li><em>...and ' + (matches.length - displayCount) + ' more matches</em></li>';
                }}

                resultsHtml += '</ul>';
                resultsDiv.innerHTML = resultsHtml;
            }}
        }});

        // Filter functionality
        document.getElementById('apply-filter').addEventListener('click', function() {{
            // Get checkbox states
            const filters = {{
                dlr_matched: document.getElementById('filter-dlr-matched').checked,
                dlr_unmatched: document.getElementById('filter-dlr-unmatched').checked,
                network_matched: document.getElementById('filter-network-matched').checked,
                network_unmatched: document.getElementById('filter-network-unmatched').checked,
                pypsa_matched: document.getElementById('filter-pypsa-matched').checked,
                pypsa_unmatched: document.getElementById('filter-pypsa-unmatched').checked,
                fifty_hertz_matched: document.getElementById('filter-fifty-hertz-matched').checked,
                fifty_hertz_unmatched: document.getElementById('filter-fifty-hertz-unmatched').checked,
                tennet_matched: document.getElementById('filter-tennet-matched').checked,
                tennet_unmatched: document.getElementById('filter-tennet-unmatched').checked
            }};

            // Apply filters
            for (const category in filters) {{
                const show = filters[category];
                const lines = lineCollections[category] || [];

                for (const line of lines) {{
                    if (show) {{
                        if (!map.hasLayer(line)) {{
                            map.addLayer(line);
                        }}
                    }} else {{
                        if (map.hasLayer(line)) {{
                            map.removeLayer(line);
                        }}
                    }}
                }}
            }}
        }});

        // Reset filters
        document.getElementById('reset-filter').addEventListener('click', function() {{
            // Reset all checkboxes
            document.getElementById('filter-dlr-matched').checked = true;
            document.getElementById('filter-dlr-unmatched').checked = true;
            document.getElementById('filter-network-matched').checked = true;
            document.getElementById('filter-network-unmatched').checked = true;
            document.getElementById('filter-pypsa-matched').checked = true;
            document.getElementById('filter-pypsa-unmatched').checked = true;
            document.getElementById('filter-fifty-hertz-matched').checked = true;
            document.getElementById('filter-fifty-hertz-unmatched').checked = true;
            document.getElementById('filter-tennet-matched').checked = true;
            document.getElementById('filter-tennet-unmatched').checked = true;

            // Apply reset filters
            document.getElementById('apply-filter').click();
        }});

        // Expose highlight function globally
        window.highlightLine = highlightLine;
    </script>
</body>
</html>
    """

    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)

    # Write HTML to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html)

    logger.info(f"Standalone map created at {output_file}")

    # Return dummy folium map for compatibility
    import folium
    return folium.Map(location=[centroid_y, centroid_x], zoom_start=6)

