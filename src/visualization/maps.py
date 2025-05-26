import os
import logging
import json
import folium
from folium.plugins import MeasureControl
from shapely.geometry import Point, box
import pandas as pd
import geopandas as gpd
logger = logging.getLogger(__name__)


def create_comprehensive_map(dlr_lines, network_lines, matches_dlr, pypsa_lines=None, pypsa_lines_new=None,
                             matches_pypsa=None, matches_pypsa_new=None, fifty_hertz_lines=None, tennet_lines=None,
                             matches_fifty_hertz=None, matches_tennet=None, germany_gdf=None,
                             output_file='comprehensive_grid_map.html', matched_ids=None,
                             dlr_lines_germany_count=None, network_lines_germany_count=None,
                             pypsa_lines_germany_count=None, pypsa_lines_new_germany_count=None,
                             fifty_hertz_lines_germany_count=None, tennet_lines_germany_count=None):
    """
    Create a comprehensive map with DLR, network, PyPSA-EUR, 50Hertz, and TenneT lines with color-coded matches.
    """
    logger.info("Creating comprehensive map with all line datasets...")

    # Format match rate more carefully to avoid NaN or infinity
    def format_rate(numerator, denominator):
        if denominator == 0:
            return "0.0"  # Avoid division by zero
        return f"{(numerator / denominator * 100):.1f}"

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

    # Create map with performance settings for many lines
    m = folium.Map(location=[centroid_y, centroid_x],
                   zoom_start=6,
                   tiles='CartoDB positron',
                   prefer_canvas=True,
                   control_scale=True,
                   name='map')  # Use canvas renderer for better performance

    # Add measure control
    folium.plugins.MeasureControl(
        position='topleft',
        primary_length_unit='kilometers',
        secondary_length_unit='miles'
    ).add_to(m)

    # Create feature groups for all line types
    dlr_matched_group = folium.FeatureGroup(name="DLR Lines (Matched)")
    dlr_unmatched_group = folium.FeatureGroup(name="DLR Lines (Unmatched)")
    network_matched_group = folium.FeatureGroup(name="Network Lines (Matched)")
    network_unmatched_group = folium.FeatureGroup(name="Network Lines (Unmatched)")

    # PyPSA-EUR groups
    pypsa_eur_matched_group = folium.FeatureGroup(name="PyPSA-EUR Lines (Matched)")
    pypsa_eur_unmatched_group = folium.FeatureGroup(name="PyPSA-EUR Lines (Unmatched)")

    # 50Hertz groups
    fifty_hertz_matched_group = folium.FeatureGroup(name="50Hertz Lines (Matched)")
    fifty_hertz_unmatched_group = folium.FeatureGroup(name="50Hertz Lines (Unmatched)")

    # TenneT groups
    tennet_matched_group = folium.FeatureGroup(name="TenneT Lines (Matched)")
    tennet_unmatched_group = folium.FeatureGroup(name="TenneT Lines (Unmatched)")

    # Search results group for highlighting
    search_results_group = folium.FeatureGroup(name="Search Results")

    # Add Germany boundary and mask areas outside Germany
    if germany_gdf is not None:
        germany_group = folium.FeatureGroup(name="Germany Boundary")

        # First add the Germany polygon with no fill and just a border
        folium.GeoJson(
            germany_gdf,
            style_function=lambda x: {
                'fillColor': 'none',
                'color': 'black',
                'weight': 2,
                'opacity': 0.5
            }
        ).add_to(germany_group)

        # Create a mask layer for areas outside Germany
        # Create a large rectangle covering the whole map area
        world_box = box(-20, 35, 40, 70)  # (min_lon, min_lat, max_lon, max_lat) covering all of Europe
        world_gdf = gpd.GeoDataFrame(geometry=[world_box], crs="EPSG:4326")

        # Create a "negative" of Germany - this is the area we want to mask
        if not germany_gdf.empty:
            # Get the German polygon
            germany_poly = germany_gdf.geometry.unary_union
            # Difference between the world box and Germany = everything except Germany
            outside_germany = world_box.difference(germany_poly)

            # Create GeoDataFrame for the area outside Germany
            outside_germany_gdf = gpd.GeoDataFrame(geometry=[outside_germany], crs="EPSG:4326")

            # Add a semi-transparent mask layer for areas outside Germany
            folium.GeoJson(
                outside_germany_gdf,
                style_function=lambda x: {
                    'fillColor': 'white',
                    'color': 'none',  # no border
                    'fillOpacity': 0.6  # semi-transparent
                }
            ).add_to(germany_group)

        germany_group.add_to(m)

    # Extract matched IDs from the passed dictionary or from match dataframes
    matched_dlr_ids = set()
    matched_network_ids_dlr = set()
    matched_pypsa_eur_ids = set()  # Only need this for PyPSA-EUR
    matched_network_ids_pypsa = set()
    matched_fifty_hertz_ids = set()
    matched_network_ids_fifty_hertz = set()
    matched_tennet_ids = set()
    matched_network_ids_tennet = set()
    all_matched_network_ids = set()

    # If matched_ids is provided, use that as the source of truth
    if matched_ids is not None:
        matched_dlr_ids = matched_ids.get('dlr', set())
        matched_pypsa_eur_ids = matched_ids.get('pypsa', set())  # Use 'pypsa' key for PyPSA-EUR
        matched_fifty_hertz_ids = matched_ids.get('fifty_hertz', set())
        matched_tennet_ids = matched_ids.get('tennet', set())
        all_matched_network_ids = matched_ids.get('network', set())
    else:
        # Extract from match dataframes (fallback)
        if matches_dlr is not None and len(matches_dlr) > 0:
            # Find the ID columns
            dlr_id_col = next((col for col in ['dlr_id', 'id'] if col in matches_dlr.columns), None)
            network_id_col = next((col for col in ['network_id', 'netw_id', 'net_id'] if col in matches_dlr.columns),
                                  None)

            if dlr_id_col and network_id_col:
                # Convert IDs to strings for matching
                matches_dlr[dlr_id_col] = matches_dlr[dlr_id_col].astype(str)
                matches_dlr[network_id_col] = matches_dlr[network_id_col].astype(str)
                matched_dlr_ids = set(matches_dlr[dlr_id_col])
                matched_network_ids_dlr = set(matches_dlr[network_id_col])
                all_matched_network_ids.update(matched_network_ids_dlr)

        if matches_pypsa is not None and len(matches_pypsa) > 0:
            # Find the ID columns
            pypsa_id_col = next((col for col in ['dlr_id', 'pypsa_id', 'id'] if col in matches_pypsa.columns), None)
            network_id_col = next((col for col in ['network_id', 'netw_id', 'net_id'] if col in matches_pypsa.columns),
                                  None)

            if pypsa_id_col and network_id_col:
                # Ensure string types for matching
                matches_pypsa[pypsa_id_col] = matches_pypsa[pypsa_id_col].astype(str)
                matches_pypsa[network_id_col] = matches_pypsa[network_id_col].astype(str)
                matched_pypsa_eur_ids = set(matches_pypsa[pypsa_id_col].values)  # Use for PyPSA-EUR
                matched_network_ids_pypsa = set(matches_pypsa[network_id_col].values)
                all_matched_network_ids.update(matched_network_ids_pypsa)

        if matches_fifty_hertz is not None and len(matches_fifty_hertz) > 0:
            # Find the ID columns
            fifty_hertz_id_col = next(
                (col for col in ['dlr_id', 'fifty_hertz_id', 'id'] if col in matches_fifty_hertz.columns), None)
            network_id_col = next(
                (col for col in ['network_id', 'netw_id', 'net_id'] if col in matches_fifty_hertz.columns), None)

            if fifty_hertz_id_col and network_id_col:
                # Ensure string types for matching
                matches_fifty_hertz[fifty_hertz_id_col] = matches_fifty_hertz[fifty_hertz_id_col].astype(str)
                matches_fifty_hertz[network_id_col] = matches_fifty_hertz[network_id_col].astype(str)
                matched_fifty_hertz_ids = set(matches_fifty_hertz[fifty_hertz_id_col].values)
                matched_network_ids_fifty_hertz = set(matches_fifty_hertz[network_id_col].values)
                all_matched_network_ids.update(matched_network_ids_fifty_hertz)

        if matches_tennet is not None and len(matches_tennet) > 0:
            # Find the ID columns
            tennet_id_col = next((col for col in ['dlr_id', 'tennet_id', 'id'] if col in matches_tennet.columns), None)
            network_id_col = next((col for col in ['network_id', 'netw_id', 'net_id'] if col in matches_tennet.columns),
                                  None)

            if tennet_id_col and network_id_col:
                # Ensure string types for matching
                matches_tennet[tennet_id_col] = matches_tennet[tennet_id_col].astype(str)
                matches_tennet[network_id_col] = matches_tennet[network_id_col].astype(str)
                matched_tennet_ids = set(matches_tennet[tennet_id_col].values)
                matched_network_ids_tennet = set(matches_tennet[network_id_col].values)
                all_matched_network_ids.update(matched_network_ids_tennet)

    # Track displayed counts for statistics
    displayed_counts = {
        'dlr_matched': 0,
        'dlr_unmatched': 0,
        'network_matched': 0,
        'network_unmatched': 0,
        'pypsa_eur_matched': 0,
        'pypsa_eur_unmatched': 0,
        'fifty_hertz_matched': 0,
        'fifty_hertz_unmatched': 0,
        'tennet_matched': 0,
        'tennet_unmatched': 0
    }

    # Dictionary to store all lines with their coordinates for search
    line_coordinates = {}

    # Function to format parameters for display
    def format_param(value, unit="", scientific=False):
        if value == 'N/A' or pd.isna(value):
            return 'N/A'
        try:
            value = float(value)
            if scientific and 0 < abs(value) < 0.001:
                return f"{value:.6e} {unit}"
            else:
                return f"{value:.6f} {unit}"
        except (ValueError, TypeError):
            return f"{value} {unit}"

    # Function to add DLR or network lines to the map
    # Function to add DLR or network lines to the map
    # Function to add DLR or network lines to the map
    def add_line_to_map(row, is_dlr, is_matched, group):
        if row.geometry is None or row.geometry.is_empty:
            return

        line_id = str(row.get('id', ''))
        color = 'blue' if is_dlr else 'red'
        if is_matched:
            color = 'green' if is_dlr else 'orange'

        # Create tooltip
        v_nom = row.get('v_nom', 'N/A')
        tooltip = f"{'DLR' if is_dlr else 'Network'} Line {line_id}"
        if is_matched:
            tooltip += " (matched)"
        tooltip += f" - {v_nom} kV"

        # Create a unique line ID for search
        unique_id = f"{'dlr' if is_dlr else 'net'}_{line_id}"
        match_status = "matched" if is_matched else "unmatched"

        # Prepare match information
        primary_match_id = None
        primary_match_data = None
        all_matches = []
        additional_matches = 0
        matched_to = ""

        if is_matched:
            if is_dlr:
                matched_to = "Network"
                # Safely get match IDs in case columns are missing
                if matches_dlr is not None and 'dlr_id' in matches_dlr and 'network_id' in matches_dlr:
                    # Get all matching network IDs
                    all_matches = [net_id for net_id, dlr_id in zip(matches_dlr['network_id'], matches_dlr['dlr_id'])
                                   if dlr_id == line_id]

                    # Get the primary match (first one)
                    if all_matches:
                        primary_match_id = all_matches[0]
                        additional_matches = len(all_matches) - 1

                        # Find the matched network line data for comparison
                        if network_lines is not None and not network_lines.empty:
                            for idx, net_row in network_lines.iterrows():
                                if str(net_row.get('id', '')) == primary_match_id:
                                    primary_match_data = net_row
                                    break
            else:
                matched_sources = []
                # Check if matched to DLR
                if line_id in matched_network_ids_dlr:
                    matched_sources.append("DLR")
                # Check if matched to PyPSA-EUR
                if line_id in matched_network_ids_pypsa:
                    matched_sources.append("PyPSA-EUR")
                # Check if matched to 50Hertz
                if line_id in matched_network_ids_fifty_hertz:
                    matched_sources.append("50Hertz")
                # Check if matched to TenneT
                if line_id in matched_network_ids_tennet:
                    matched_sources.append("TenneT")

                matched_to = ", ".join(matched_sources)
                match_ids = []
                all_matches = []
                additional_matches = 0

                # For network lines matched to DLR, show only the first match from each source
                if "DLR" in matched_sources and matches_dlr is not None and 'dlr_id' in matches_dlr and 'network_id' in matches_dlr:
                    dlr_matches = [f"DLR: {dlr_id}" for dlr_id, net_id in
                                   zip(matches_dlr['dlr_id'], matches_dlr['network_id'])
                                   if net_id == line_id]

                    all_matches.extend(dlr_matches)
                    if dlr_matches:
                        match_ids.append(dlr_matches[0])
                        additional_matches += len(dlr_matches) - 1

                # Similar code for other sources (PyPSA, 50Hertz, TenneT)
                # For PyPSA-EUR
                if "PyPSA-EUR" in matched_sources and matches_pypsa is not None and 'dlr_id' in matches_pypsa and 'network_id' in matches_pypsa:
                    pypsa_matches = [f"PyPSA-EUR: {pypsa_id}" for pypsa_id, net_id in
                                     zip(matches_pypsa['dlr_id'], matches_pypsa['network_id'])
                                     if net_id == line_id]

                    all_matches.extend(pypsa_matches)
                    if pypsa_matches:
                        match_ids.append(pypsa_matches[0])
                        additional_matches += len(pypsa_matches) - 1

                # For 50Hertz
                if "50Hertz" in matched_sources and matches_fifty_hertz is not None and 'dlr_id' in matches_fifty_hertz and 'network_id' in matches_fifty_hertz:
                    fifty_hertz_matches = [f"50Hertz: {fifty_id}" for fifty_id, net_id in
                                           zip(matches_fifty_hertz['dlr_id'], matches_fifty_hertz['network_id'])
                                           if net_id == line_id]

                    all_matches.extend(fifty_hertz_matches)
                    if fifty_hertz_matches:
                        match_ids.append(fifty_hertz_matches[0])
                        additional_matches += len(fifty_hertz_matches) - 1

                # For TenneT
                if "TenneT" in matched_sources and matches_tennet is not None and 'dlr_id' in matches_tennet and 'network_id' in matches_tennet:
                    tennet_matches = [f"TenneT: {tennet_id}" for tennet_id, net_id in
                                      zip(matches_tennet['dlr_id'], matches_tennet['network_id'])
                                      if net_id == line_id]

                    all_matches.extend(tennet_matches)
                    if tennet_matches:
                        match_ids.append(tennet_matches[0])
                        additional_matches += len(tennet_matches) - 1

        # Create popup content - with comparison if matched
        if is_dlr and is_matched and primary_match_data is not None:
            # Two-column layout for comparison
            popup_html = f"""
            <div style="min-width: 550px; max-width: 600px;">
                <h4>DLR Line {line_id} matched with Network Line {primary_match_id}</h4>
                <table style="width: 100%; border-collapse: collapse; margin-bottom: 10px; table-layout: fixed;">
                    <tr>
                        <th style="text-align: left; padding: 3px; border-bottom: 1px solid #ddd; width: 34%;">Parameter</th>
                        <th style="text-align: right; padding: 3px; border-bottom: 1px solid #ddd; width: 33%;">DLR Value</th>
                        <th style="text-align: right; padding: 3px; border-bottom: 1px solid #ddd; width: 33%;">Network Value</th>
                    </tr>
                    <tr>
                        <td style="padding: 3px;">Voltage</td>
                        <td style="text-align: right; padding: 3px;">{row.get('v_nom', 'N/A')} kV</td>
                        <td style="text-align: right; padding: 3px;">{primary_match_data.get('v_nom', 'N/A')} kV</td>
                    </tr>
                    <tr>
                        <td style="padding: 3px;">Apparent Power (s_nom)</td>
                        <td style="text-align: right; padding: 3px;">{row.get('s_nom', 'N/A')} MVA</td>
                        <td style="text-align: right; padding: 3px;">{primary_match_data.get('s_nom', 'N/A')} MVA</td>
                    </tr>
                    <tr>
                        <td style="padding: 3px;">Length</td>
                        <td style="text-align: right; padding: 3px;">{row.get('length', 0):.2f} km</td>
                        <td style="text-align: right; padding: 3px;">{primary_match_data.get('length', 0):.2f} km</td>
                    </tr>
                    <tr>
                        <td style="padding: 3px;">Resistance (r)</td>
                        <td style="text-align: right; padding: 3px;">{format_param(row.get('r', 'N/A'), 'Ω')}</td>
                        <td style="text-align: right; padding: 3px;">{format_param(primary_match_data.get('r', 'N/A'), 'Ω')}</td>
                    </tr>
                    <tr>
                        <td style="padding: 3px;">Reactance (x)</td>
                        <td style="text-align: right; padding: 3px;">{format_param(row.get('x', 'N/A'), 'Ω')}</td>
                        <td style="text-align: right; padding: 3px;">{format_param(primary_match_data.get('x', 'N/A'), 'Ω')}</td>
                    </tr>
                    <tr>
                        <td style="padding: 3px;">Susceptance (b)</td>
                        <td style="text-align: right; padding: 3px;">{format_param(row.get('b', 'N/A'), 'S', True)}</td>
                        <td style="text-align: right; padding: 3px;">{format_param(primary_match_data.get('b', 'N/A'), 'S', True)}</td>
                    </tr>
                    <tr>
                        <td style="padding: 3px;">r per km</td>
                        <td style="text-align: right; padding: 3px;">{format_param(row.get('r_per_km', 'N/A'), 'Ω/km')}</td>
                        <td style="text-align: right; padding: 3px;">{format_param(primary_match_data.get('r_per_km', 'N/A'), 'Ω/km')}</td>
                    </tr>
                    <tr>
                        <td style="padding: 3px;">x per km</td>
                        <td style="text-align: right; padding: 3px;">{format_param(row.get('x_per_km', 'N/A'), 'Ω/km')}</td>
                        <td style="text-align: right; padding: 3px;">{format_param(primary_match_data.get('x_per_km', 'N/A'), 'Ω/km')}</td>
                    </tr>
                    <tr>
                        <td style="padding: 3px;">b per km</td>
                        <td style="text-align: right; padding: 3px;">{format_param(row.get('b_per_km', 'N/A'), 'S/km', True)}</td>
                        <td style="text-align: right; padding: 3px;">{format_param(primary_match_data.get('b_per_km', 'N/A'), 'S/km', True)}</td>
                    </tr>
                    <tr>
                        <td style="padding: 3px;">ID</td>
                        <td style="text-align: right; padding: 3px;">{line_id}</td>
                        <td style="text-align: right; padding: 3px;">{primary_match_id}</td>
                    </tr>
                    <tr>
                        <td style="padding: 3px;">Status</td>
                        <td style="text-align: right; padding: 3px;">Matched</td>
                        <td style="text-align: right; padding: 3px;">Matched</td>
                    </tr>
                </table>
            """

            # Add additional matches information
            if additional_matches > 0:
                popup_html += f"""
                <p>+{additional_matches} additional matches</p>
                <div id="all-matches-{line_id}" style="display: none;">
                    <p><b>All Network Line Matches:</b></p>
                    <ul style="max-height: 150px; overflow-y: auto; font-size: 12px;">
                """

                for match_id in all_matches[1:]:  # Skip the first one (primary)
                    popup_html += f"<li>Network: {match_id}</li>"

                popup_html += f"""
                    </ul>
                </div>

                <button id="toggle-btn-{line_id}" onclick="toggleMatches('{line_id}')" 
                   style="font-size: 11px; padding: 3px 6px; background-color: #f0f0f0; border: 1px solid #ccc; border-radius: 3px; margin-top: 5px; cursor: pointer;">
                   Show All Matches
                </button>
                """
        else:
            # Regular single-column layout for unmatched lines or network lines
            popup_html = f"""
            <div style="min-width: 300px;">
                <h4>{'DLR' if is_dlr else 'Network'} Line {line_id}</h4>
                <table style="width: 100%; border-collapse: collapse; margin-bottom: 10px;">
                    <tr>
                        <th style="text-align: left; padding: 3px; border-bottom: 1px solid #ddd;">Parameter</th>
                        <th style="text-align: right; padding: 3px; border-bottom: 1px solid #ddd;">Value</th>
                    </tr>
                    <tr>
                        <td style="padding: 3px;">Voltage</td>
                        <td style="text-align: right; padding: 3px;">{row.get('v_nom', 'N/A')} kV</td>
                    </tr>
                    <tr>
                        <td style="padding: 3px;">Apparent Power (s_nom)</td>
                        <td style="text-align: right; padding: 3px;">{row.get('s_nom', 'N/A')} MVA</td>
                    </tr>
                    <tr>
                        <td style="padding: 3px;">Length</td>
                        <td style="text-align: right; padding: 3px;">{row.get('length', 0):.2f} km</td>
                    </tr>
                    <tr>
                        <td style="padding: 3px;">Resistance (r)</td>
                        <td style="text-align: right; padding: 3px;">{format_param(row.get('r', 'N/A'), 'Ω')}</td>
                    </tr>
                    <tr>
                        <td style="padding: 3px;">Reactance (x)</td>
                        <td style="text-align: right; padding: 3px;">{format_param(row.get('x', 'N/A'), 'Ω')}</td>
                    </tr>
                    <tr>
                        <td style="padding: 3px;">Susceptance (b)</td>
                        <td style="text-align: right; padding: 3px;">{format_param(row.get('b', 'N/A'), 'S', True)}</td>
                    </tr>
                    <tr>
                        <td style="padding: 3px;">r per km</td>
                        <td style="text-align: right; padding: 3px;">{format_param(row.get('r_per_km', 'N/A'), 'Ω/km')}</td>
                    </tr>
                    <tr>
                        <td style="padding: 3px;">x per km</td>
                        <td style="text-align: right; padding: 3px;">{format_param(row.get('x_per_km', 'N/A'), 'Ω/km')}</td>
                    </tr>
                    <tr>
                        <td style="padding: 3px;">b per km</td>
                        <td style="text-align: right; padding: 3px;">{format_param(row.get('b_per_km', 'N/A'), 'S/km', True)}</td>
                    </tr>
                    <tr>
                        <td style="padding: 3px;">ID</td>
                        <td style="text-align: right; padding: 3px;">{line_id}</td>
                    </tr>
                    <tr>
                        <td style="padding: 3px;">Status</td>
                        <td style="text-align: right; padding: 3px;">{'Matched' if is_matched else 'Unmatched'}</td>
                    </tr>
                </table>
            """

            # Add match information if this is a matched line
            if is_matched:
                if is_dlr:
                    popup_html += f"""
                    <p><b>Matched with:</b> {matched_to}</p>
                    <p><b>Primary Match:</b> {primary_match_id or 'N/A'}</p>
                    """

                    # Add additional matches information if any
                    if additional_matches > 0:
                        popup_html += f"""
                        <p>+{additional_matches} additional matches</p>
                        <div id="all-matches-{line_id}" style="display: none;">
                            <p><b>All Network Line Matches:</b></p>
                            <ul style="max-height: 150px; overflow-y: auto; font-size: 12px;">
                        """

                        for match_id in all_matches[1:]:  # Skip the first one (primary)
                            popup_html += f"<li>Network: {match_id}</li>"

                        popup_html += f"""
                            </ul>
                        </div>

                        <button id="toggle-btn-{line_id}" onclick="toggleMatches('{line_id}')" 
                           style="font-size: 11px; padding: 3px 6px; background-color: #f0f0f0; border: 1px solid #ccc; border-radius: 3px; margin-top: 5px; cursor: pointer;">
                           Show All Matches
                        </button>
                        """
                else:
                    # For network lines that are matched to other sources
                    popup_html += f"<p><b>Matched with:</b> {matched_to}</p>"

                    if match_ids:
                        popup_html += "<p><b>Primary Matches:</b></p><ul>"
                        for match_id in match_ids:
                            popup_html += f"<li>{match_id}</li>"
                        popup_html += "</ul>"

                    # If there are additional matches, show a count and toggle
                    if additional_matches > 0:
                        popup_html += f"""
                        <p>+{additional_matches} additional matches</p>
                        <div id="all-matches-{line_id}" style="display: none;">
                            <p><b>All Matches:</b></p>
                            <ul style="max-height: 150px; overflow-y: auto; font-size: 12px;">
                        """

                        shown_matches = set(match_ids)
                        for match in all_matches:
                            if match not in shown_matches:
                                popup_html += f"<li>{match}</li>"

                        popup_html += f"""
                            </ul>
                        </div>

                        <button id="toggle-btn-{line_id}" onclick="toggleMatches('{line_id}')" 
                           style="font-size: 11px; padding: 3px 6px; background-color: #f0f0f0; border: 1px solid #ccc; border-radius: 3px; margin-top: 5px; cursor: pointer;">
                           Show All Matches
                        </button>
                        """

        popup_html += "</div>"

        # Store all coordinates for search
        all_coords = []
        added_to_map = False

        try:
            # For MultiLineString, add each segment separately
            if row.geometry.geom_type == 'MultiLineString':
                for segment in row.geometry.geoms:
                    if segment.is_empty or len(list(segment.coords)) < 2:
                        continue

                    # Convert to folium coordinates
                    latlngs = [[y, x] for x, y in segment.coords]
                    all_coords.extend(latlngs)

                    # Add segment to map
                    folium.PolyLine(
                        latlngs,
                        popup=folium.Popup(popup_html, max_width=600),  # Increased max width
                        tooltip=tooltip,
                        color=color,
                        weight=3 if is_matched else 2,
                        opacity=0.8 if is_matched else 0.6,
                        name=unique_id,
                        className=f"{unique_id} {match_status}"
                    ).add_to(group)
                    added_to_map = True

            else:
                # Handle LineString
                if hasattr(row.geometry, 'coords') and len(list(row.geometry.coords)) >= 2:
                    coords = list(row.geometry.coords)
                    latlngs = [[y, x] for x, y in coords]
                    all_coords.extend(latlngs)

                    # Add line to map
                    folium.PolyLine(
                        latlngs,
                        popup=folium.Popup(popup_html, max_width=600),  # Increased max width
                        tooltip=tooltip,
                        color=color,
                        weight=3 if is_matched else 2,
                        opacity=0.8 if is_matched else 0.6,
                        name=unique_id,
                        className=f"{unique_id} {match_status}"
                    ).add_to(group)
                    added_to_map = True

        except Exception as e:
            logger.error(f"Error adding {'DLR' if is_dlr else 'Network'} line {line_id} to map: {str(e)}")

        # Only count lines actually added to the map
        if added_to_map:
            # Update display counts for statistics
            if is_dlr:
                if is_matched:
                    displayed_counts['dlr_matched'] += 1
                else:
                    displayed_counts['dlr_unmatched'] += 1
            else:
                if is_matched:
                    displayed_counts['network_matched'] += 1
                else:
                    displayed_counts['network_unmatched'] += 1

            # Calculate midpoint for search functionality
            if all_coords:
                lat_sum = sum(coord[0] for coord in all_coords)
                lng_sum = sum(coord[1] for coord in all_coords)
                midpoint = [lat_sum / len(all_coords), lng_sum / len(all_coords)]

                # Store the coordinates and midpoint for this line
                simplified_coords = all_coords
                if len(all_coords) > 50:  # Simplify if too many points
                    step = max(1, len(all_coords) // 50)
                    simplified_coords = all_coords[::step]
                    if simplified_coords[-1] != all_coords[-1]:
                        simplified_coords.append(all_coords[-1])

                line_coordinates[unique_id] = {
                    'id': line_id,
                    'type': 'dlr' if is_dlr else 'network',
                    'coords': simplified_coords,
                    'midpoint': midpoint,
                    'is_matched': is_matched,
                    'match_status': match_status
                }

    # Function to add PyPSA-EUR lines to the map
    def add_pypsa_eur_line_to_map(row, is_matched, group):
        """Special function just for PyPSA-EUR lines"""
        if row.geometry is None or row.geometry.is_empty:
            return False

        # Get the line ID, handling different column names that might be present
        line_id = str(row.get('id', row.get('line_id', '')))

        # Debug this line
        logger.debug(f"Processing PyPSA-EUR line {line_id}, match status: {is_matched}")

        # Double-check match status
        actual_match = line_id in matched_pypsa_eur_ids
        if actual_match != is_matched:
            logger.warning(
                f"Match status inconsistency for PyPSA-EUR line {line_id}: passed {is_matched}, actual {actual_match}")
            is_matched = actual_match  # Use actual match status

        # Set color - using purple colors to match the legend
        color = 'purple' if is_matched else 'magenta'  # Changed from orange to purple/magenta
        line_type = "PyPSA-EUR"

        # Create tooltip
        v_nom = row.get('v_nom', row.get('voltage', 'N/A'))
        tooltip = f"PyPSA-EUR Line {line_id}"
        if is_matched:
            tooltip += " (matched)"
        tooltip += f" - {v_nom} kV"

        # Create unique ID
        unique_id = f"pypsa_eur_{line_id}"
        match_status = "matched" if is_matched else "unmatched"

        # Create popup content
        popup_html = f"""
        <div style="min-width: 300px;">
            <h4>PyPSA-EUR Line {line_id}</h4>
            <table style="width: 100%; border-collapse: collapse; margin-bottom: 10px;">
                <tr>
                    <th style="text-align: left; padding: 3px; border-bottom: 1px solid #ddd;">Parameter</th>
                    <th style="text-align: right; padding: 3px; border-bottom: 1px solid #ddd;">Value</th>
                </tr>
                <tr>
                    <td style="padding: 3px;">Voltage</td>
                    <td style="text-align: right; padding: 3px;">{row.get('v_nom', row.get('voltage', 'N/A'))} kV</td>
                </tr>
                <tr>
                    <td style="padding: 3px;">Apparent Power (s_nom)</td>
                    <td style="text-align: right; padding: 3px;">{row.get('s_nom', 'N/A')} MVA</td>
                </tr>
                <tr>
                    <td style="padding: 3px;">Length</td>
                    <td style="text-align: right; padding: 3px;">{row.get('length', 0):.2f} km</td>
                </tr>
                <tr>
                    <td style="padding: 3px;">Resistance (r)</td>
                    <td style="text-align: right; padding: 3px;">{format_param(row.get('r', 'N/A'), 'Ω')}</td>
                </tr>
                <tr>
                    <td style="padding: 3px;">Reactance (x)</td>
                    <td style="text-align: right; padding: 3px;">{format_param(row.get('x', 'N/A'), 'Ω')}</td>
                </tr>
                <tr>
                    <td style="padding: 3px;">Susceptance (b)</td>
                    <td style="text-align: right; padding: 3px;">{format_param(row.get('b', 'N/A'), 'S', True)}</td>
                </tr>
                <tr>
                    <td style="padding: 3px;">r per km</td>
                    <td style="text-align: right; padding: 3px;">{format_param(row.get('r_per_km', 'N/A'), 'Ω/km')}</td>
                </tr>
                <tr>
                    <td style="padding: 3px;">x per km</td>
                    <td style="text-align: right; padding: 3px;">{format_param(row.get('x_per_km', 'N/A'), 'Ω/km')}</td>
                </tr>
                <tr>
                    <td style="padding: 3px;">b per km</td>
                    <td style="text-align: right; padding: 3px;">{format_param(row.get('b_per_km', 'N/A'), 'S/km', True)}</td>
                </tr>
                <tr>
                    <td style="padding: 3px;">ID</td>
                    <td style="text-align: right; padding: 3px;">{line_id}</td>
                </tr>
                <tr>
                    <td style="padding: 3px;">Status</td>
                    <td style="text-align: right; padding: 3px;">{'Matched' if is_matched else 'Unmatched'}</td>
                </tr>
            </table>
        """

        if is_matched:
            # Find matched network IDs
            match_ids = []
            if matches_pypsa is not None and 'dlr_id' in matches_pypsa and 'network_id' in matches_pypsa:
                match_ids = [net_id for net_id, pypsa_id in
                             zip(matches_pypsa['network_id'], matches_pypsa['dlr_id'])
                             if pypsa_id == line_id]

            if match_ids:
                popup_html += "<p><b>Matched with Network Lines:</b></p><ul>"
                for match_id in match_ids:
                    popup_html += f"<li>Network: {match_id}</li>"
                popup_html += "</ul>"

        popup_html += "</div>"

        # Store coordinates for search
        all_coords = []
        added_to_map = False

        try:
            # Process geometry - enhanced error handling
            if row.geometry.geom_type == 'MultiLineString':
                for segment in row.geometry.geoms:
                    if segment.is_empty or len(list(segment.coords)) < 2:
                        continue

                    # Convert to folium coordinates
                    latlngs = [[y, x] for x, y in segment.coords]
                    all_coords.extend(latlngs)

                    # Add segment to map
                    folium.PolyLine(
                        latlngs,
                        popup=folium.Popup(popup_html, max_width=300),
                        tooltip=tooltip,
                        color=color,
                        weight=4,  # Thicker to be more visible
                        opacity=0.8,
                        dash_array="5,5",  # Distinctive dash pattern
                        name=unique_id,
                        className=f"{unique_id} {match_status}"
                    ).add_to(group)
                    added_to_map = True

            else:  # LineString or other geometry
                if hasattr(row.geometry, 'coords') and len(list(row.geometry.coords)) >= 2:
                    coords = list(row.geometry.coords)
                    latlngs = [[y, x] for x, y in coords]
                    all_coords.extend(latlngs)

                    # Add line to map
                    folium.PolyLine(
                        latlngs,
                        popup=folium.Popup(popup_html, max_width=300),
                        tooltip=tooltip,
                        color=color,
                        weight=4,  # Thicker to be more visible
                        opacity=0.8,
                        dash_array="5,5",  # Distinctive dash pattern
                        name=unique_id,
                        className=f"{unique_id} {match_status}"
                    ).add_to(group)
                    added_to_map = True

        except Exception as e:
            logger.error(f"Error adding PyPSA-EUR line {line_id} to map: {str(e)}")
            return False

        # Only continue if actually added to map
        if added_to_map:
            # Update display counts
            if is_matched:
                displayed_counts['pypsa_eur_matched'] += 1
            else:
                displayed_counts['pypsa_eur_unmatched'] += 1

            # Calculate midpoint for search
            if all_coords:
                lat_sum = sum(coord[0] for coord in all_coords)
                lng_sum = sum(coord[1] for coord in all_coords)
                midpoint = [lat_sum / len(all_coords), lng_sum / len(all_coords)]

                # Store simplified coordinates
                simplified_coords = all_coords
                if len(all_coords) > 50:
                    step = max(1, len(all_coords) // 50)
                    simplified_coords = all_coords[::step]
                    if simplified_coords[-1] != all_coords[-1]:
                        simplified_coords.append(all_coords[-1])

                line_coordinates[unique_id] = {
                    'id': line_id,
                    'type': 'pypsa_eur',  # Specific type for PyPSA-EUR
                    'coords': simplified_coords,
                    'midpoint': midpoint,
                    'is_matched': is_matched,
                    'match_status': match_status
                }

            # Log that we added a line
            logger.debug(f"Added PyPSA-EUR line {line_id} to map (matched: {is_matched})")
            return True

        return False

    # Function to add TSO lines to the map
    def add_tso_line_to_map(row, tso_name, is_matched, group):
        if row.geometry is None or row.geometry.is_empty:
            return False

        line_id = str(row.get('id', ''))
        # Assign different colors for each TSO
        if tso_name == '50Hertz':
            color = 'darkgreen' if is_matched else 'green'
        else:  # TenneT
            color = 'darkblue' if is_matched else 'blue'

        # Create tooltip
        v_nom = row.get('v_nom', 'N/A')
        tooltip = f"{tso_name} Line {line_id}"
        if is_matched:
            tooltip += " (matched)"
        tooltip += f" - {v_nom} kV"

        # Create a unique line ID for search
        unique_id = f"{tso_name.lower()}_{line_id}"
        match_status = "matched" if is_matched else "unmatched"

        # Create popup content
        popup_html = f"""
        <div style="min-width: 300px;">
            <h4>{tso_name} Line {line_id}</h4>
            <table style="width: 100%; border-collapse: collapse; margin-bottom: 10px;">
                <tr>
                    <th style="text-align: left; padding: 3px; border-bottom: 1px solid #ddd;">Parameter</th>
                    <th style="text-align: right; padding: 3px; border-bottom: 1px solid #ddd;">Value</th>
                </tr>
                <tr>
                    <td style="padding: 3px;">Voltage</td>
                    <td style="text-align: right; padding: 3px;">{row.get('v_nom', 'N/A')} kV</td>
                </tr>
                <tr>
                    <td style="padding: 3px;">Apparent Power (s_nom)</td>
                    <td style="text-align: right; padding: 3px;">{row.get('s_nom', 'N/A')} MVA</td>
                </tr>
                <tr>
                    <td style="padding: 3px;">Length</td>
                    <td style="text-align: right; padding: 3px;">{row.get('length', 0):.2f} km</td>
                </tr>
                <tr>
                    <td style="padding: 3px;">Resistance (r)</td>
                    <td style="text-align: right; padding: 3px;">{format_param(row.get('r', 'N/A'), 'Ω')}</td>
                </tr>
                <tr>
                    <td style="padding: 3px;">Reactance (x)</td>
                    <td style="text-align: right; padding: 3px;">{format_param(row.get('x', 'N/A'), 'Ω')}</td>
                </tr>
                <tr>
                    <td style="padding: 3px;">Susceptance (b)</td>
                    <td style="text-align: right; padding: 3px;">{format_param(row.get('b', 'N/A'), 'S', True)}</td>
                </tr>
                <tr>
                    <td style="padding: 3px;">r per km</td>
                    <td style="text-align: right; padding: 3px;">{format_param(row.get('r_per_km', 'N/A'), 'Ω/km')}</td>
                </tr>
                <tr>
                    <td style="padding: 3px;">x per km</td>
                    <td style="text-align: right; padding: 3px;">{format_param(row.get('x_per_km', 'N/A'), 'Ω/km')}</td>
                </tr>
                <tr>
                    <td style="padding: 3px;">b per km</td>
                    <td style="text-align: right; padding: 3px;">{format_param(row.get('b_per_km', 'N/A'), 'S/km', True)}</td>
                </tr>
                <tr>
                    <td style="padding: 3px;">ID</td>
                    <td style="text-align: right; padding: 3px;">{line_id}</td>
                </tr>
                <tr>
                    <td style="padding: 3px;">Status</td>
                    <td style="text-align: right; padding: 3px;">{'Matched' if is_matched else 'Unmatched'}</td>
                </tr>
            </table>
        """

        if is_matched:
            # Find matched network IDs
            match_ids = []
            match_dataframe = matches_fifty_hertz if tso_name == '50Hertz' else matches_tennet

            if match_dataframe is not None and 'dlr_id' in match_dataframe.columns and 'network_id' in match_dataframe.columns:
                for match_id, net_id in zip(match_dataframe['dlr_id'], match_dataframe['network_id']):
                    if str(match_id) == line_id:
                        match_ids.append(net_id)

            if match_ids:
                popup_html += "<p><b>Matched with Network Lines:</b></p><ul>"
                for net_id in match_ids:
                    popup_html += f"<li>Network: {net_id}</li>"
                popup_html += "</ul>"

        popup_html += "</div>"

        # Store all coordinates for search
        all_coords = []
        added_to_map = False

        try:
            # Process the geometry
            if row.geometry.geom_type == 'MultiLineString':
                for segment in row.geometry.geoms:
                    if segment.is_empty or len(list(segment.coords)) < 2:
                        continue

                    # Convert to folium coordinates
                    latlngs = [[y, x] for x, y in segment.coords]
                    all_coords.extend(latlngs)

                    # Add segment to map
                    folium.PolyLine(
                        latlngs,
                        popup=folium.Popup(popup_html, max_width=300),
                        tooltip=tooltip,
                        color=color,
                        weight=3 if is_matched else 2,
                        opacity=0.8 if is_matched else 0.6,
                        name=unique_id,
                        className=f"{unique_id} {match_status}"
                    ).add_to(group)
                    added_to_map = True

            else:
                # Handle LineString
                if hasattr(row.geometry, 'coords') and len(list(row.geometry.coords)) >= 2:
                    coords = list(row.geometry.coords)
                    latlngs = [[y, x] for x, y in coords]
                    all_coords.extend(latlngs)

                    # Add line to map
                    folium.PolyLine(
                        latlngs,
                        popup=folium.Popup(popup_html, max_width=300),
                        tooltip=tooltip,
                        color=color,
                        weight=3 if is_matched else 2,
                        opacity=0.8 if is_matched else 0.6,
                        name=unique_id,
                        className=f"{unique_id} {match_status}"
                    ).add_to(group)
                    added_to_map = True

        except Exception as e:
            logger.error(f"Error adding {tso_name} line {line_id} to map: {str(e)}")
            return False

        # Only count lines actually added to the map
        if added_to_map:
            # Update display counts for statistics
            if tso_name == '50Hertz':
                if is_matched:
                    displayed_counts['fifty_hertz_matched'] += 1
                else:
                    displayed_counts['fifty_hertz_unmatched'] += 1
            else:  # TenneT
                if is_matched:
                    displayed_counts['tennet_matched'] += 1
                else:
                    displayed_counts['tennet_unmatched'] += 1

            # Calculate midpoint for search functionality
            if all_coords:
                lat_sum = sum(coord[0] for coord in all_coords)
                lng_sum = sum(coord[1] for coord in all_coords)
                midpoint = [lat_sum / len(all_coords), lng_sum / len(all_coords)]

                # Store the coordinates and midpoint for this line
                simplified_coords = all_coords
                if len(all_coords) > 50:  # Simplify if too many points
                    step = max(1, len(all_coords) // 50)
                    simplified_coords = all_coords[::step]
                    if simplified_coords[-1] != all_coords[-1]:
                        simplified_coords.append(all_coords[-1])

                line_coordinates[unique_id] = {
                    'id': line_id,
                    'type': tso_name.lower(),
                    'coords': simplified_coords,
                    'midpoint': midpoint,
                    'is_matched': is_matched,
                    'match_status': match_status
                }

            return True

        return False

    # Add DLR lines to the map
    if dlr_lines is not None and not dlr_lines.empty:
        logger.info(f"Adding {len(dlr_lines)} DLR lines to the map...")
        for idx, row in dlr_lines.iterrows():
            # Get line ID safely
            line_id = str(row.get('id', idx))
            # Check if line ID is in matched set
            is_matched = line_id in matched_dlr_ids
            group = dlr_matched_group if is_matched else dlr_unmatched_group
            add_line_to_map(row, is_dlr=True, is_matched=is_matched, group=group)

    # Add network lines to the map
    if network_lines is not None and not network_lines.empty:
        logger.info(f"Adding {len(network_lines)} network lines to the map...")
        for idx, row in network_lines.iterrows():
            # Get line ID safely
            line_id = str(row.get('id', idx))
            # A network line can be matched to any source dataset
            is_matched = line_id in all_matched_network_ids
            group = network_matched_group if is_matched else network_unmatched_group
            add_line_to_map(row, is_dlr=False, is_matched=is_matched, group=group)

    # Add PyPSA-EUR lines to the map if available
    if pypsa_lines is not None and not pypsa_lines.empty:
        logger.info(f"Adding {len(pypsa_lines)} PyPSA-EUR lines to the map...")

        # Check the first few lines to debug
        for idx, row in pypsa_lines.head(3).iterrows():
            logger.info(f"Sample PyPSA-EUR line {idx}: {row.get('id', 'No ID')} - Geometry: {row.geometry.wkt[:100]}")

        added_count = 0
        for idx, row in pypsa_lines.iterrows():
            # Get the ID safely
            line_id = str(row.get('id', ''))
            # Determine match status
            is_matched = line_id in matched_pypsa_eur_ids
            # Add to appropriate group
            group = pypsa_eur_matched_group if is_matched else pypsa_eur_unmatched_group

            # Use the specialized function
            if add_pypsa_eur_line_to_map(row, is_matched, group):
                added_count += 1

        logger.info(f"Added {added_count} PyPSA-EUR lines to the map out of {len(pypsa_lines)} total")

    # Add the 50Hertz lines to the map
    if fifty_hertz_lines is not None and not fifty_hertz_lines.empty:
        logger.info(f"Adding {len(fifty_hertz_lines)} 50Hertz lines to the map...")
        for idx, row in fifty_hertz_lines.iterrows():
            # Get line ID safely
            line_id = str(row.get('id', idx))
            # Check if line ID is in matched set
            is_matched = line_id in matched_fifty_hertz_ids
            group = fifty_hertz_matched_group if is_matched else fifty_hertz_unmatched_group
            add_tso_line_to_map(row, tso_name='50Hertz', is_matched=is_matched, group=group)

    # Add the TenneT lines to the map
    if tennet_lines is not None and not tennet_lines.empty:
        logger.info(f"Adding {len(tennet_lines)} TenneT lines to the map...")
        for idx, row in tennet_lines.iterrows():
            # Get line ID safely
            line_id = str(row.get('id', idx))
            # Check if line ID is in matched set
            is_matched = line_id in matched_tennet_ids
            group = tennet_matched_group if is_matched else tennet_unmatched_group
            add_tso_line_to_map(row, tso_name='TenneT', is_matched=is_matched, group=group)

    # Count lines by category for layer naming
    dlr_matched_count = len(matched_dlr_ids)
    dlr_unmatched_count = dlr_lines_germany_count - dlr_matched_count

    network_matched_count = len(all_matched_network_ids)
    network_unmatched_count = network_lines_germany_count - network_matched_count

    pypsa_eur_matched_count = len(matched_pypsa_eur_ids) if pypsa_lines is not None else 0
    pypsa_eur_unmatched_count = pypsa_lines_germany_count - pypsa_eur_matched_count if pypsa_lines is not None else 0

    fifty_hertz_matched_count = len(matched_fifty_hertz_ids) if fifty_hertz_lines is not None else 0
    fifty_hertz_unmatched_count = fifty_hertz_lines_germany_count - fifty_hertz_matched_count if fifty_hertz_lines is not None else 0

    tennet_matched_count = len(matched_tennet_ids) if tennet_lines is not None else 0
    tennet_unmatched_count = tennet_lines_germany_count - tennet_matched_count if tennet_lines is not None else 0

    # Update layer names with counts
    dlr_matched_group.layer_name = f"DLR Lines (Matched) ({dlr_matched_count})"
    dlr_unmatched_group.layer_name = f"DLR Lines (Unmatched) ({dlr_unmatched_count})"
    network_matched_group.layer_name = f"Network Lines (Matched) ({network_matched_count})"
    network_unmatched_group.layer_name = f"Network Lines (Unmatched) ({network_unmatched_count})"

    # Add layer names for PyPSA-EUR if available
    if pypsa_lines is not None:
        pypsa_eur_matched_group.layer_name = f"PyPSA-EUR Lines (Matched) ({pypsa_eur_matched_count})"
        pypsa_eur_unmatched_group.layer_name = f"PyPSA-EUR Lines (Unmatched) ({pypsa_eur_unmatched_count})"

    # Add layer names for 50Hertz if available
    if fifty_hertz_lines is not None:
        fifty_hertz_matched_group.layer_name = f"50Hertz Lines (Matched) ({fifty_hertz_matched_count})"
        fifty_hertz_unmatched_group.layer_name = f"50Hertz Lines (Unmatched) ({fifty_hertz_unmatched_count})"

    # Add layer names for TenneT if available
    if tennet_lines is not None:
        tennet_matched_group.layer_name = f"TenneT Lines (Matched) ({tennet_matched_count})"
        tennet_unmatched_group.layer_name = f"TenneT Lines (Unmatched) ({tennet_unmatched_count})"

    search_results_group.layer_name = "Search Results"

    # Add feature groups to map in order of importance
    search_results_group.add_to(m)
    if 'germany_group' in locals():
        germany_group.add_to(m)

    # Add matched lines first (most important)
    network_matched_group.add_to(m)
    dlr_matched_group.add_to(m)

    if pypsa_lines is not None:
        pypsa_eur_matched_group.add_to(m)

    if fifty_hertz_lines is not None:
        fifty_hertz_matched_group.add_to(m)

    if tennet_lines is not None:
        tennet_matched_group.add_to(m)

    # Then add unmatched lines
    network_unmatched_group.add_to(m)
    dlr_unmatched_group.add_to(m)

    if pypsa_lines is not None:
        pypsa_eur_unmatched_group.add_to(m)

    if fifty_hertz_lines is not None:
        fifty_hertz_unmatched_group.add_to(m)

    if tennet_lines is not None:
        tennet_unmatched_group.add_to(m)

    # Add layer control
    folium.LayerControl().add_to(m)

    # Convert line coordinates dictionary to JSON for JavaScript
    import json
    try:
        line_coords_json = json.dumps(line_coordinates)
    except Exception as e:
        logger.error(f"Error converting coordinates to JSON: {e}")
        # Simplify coordinates further if JSON conversion fails
        for line_id in line_coordinates:
            if 'coords' in line_coordinates[line_id] and len(line_coordinates[line_id]['coords']) > 10:
                line_coordinates[line_id]['coords'] = line_coordinates[line_id]['coords'][::5]
        try:
            line_coords_json = json.dumps(line_coordinates)
        except Exception as e2:
            logger.error(f"Still unable to convert to JSON after simplification: {e2}")
            line_coords_json = "{}"  # Empty JSON as last resort

    # Add stats panel
    # Replace all JavaScript components with this consolidated version
    javascript = """
    <script>
    // Store line coordinates data for search
    var lineCoordinates = """ + line_coords_json + """;
    var currentSearchMarker = null;
    var currentHighlightedLines = [];

    // Add custom CSS for animations and styling
    var style = document.createElement('style');
    style.textContent = `
        @keyframes flashBg {
            0% { background-color: #e6f7ff; }
            100% { background-color: #f8f8f8; }
        }

        #search-results a:hover {
            text-decoration: underline !important;
            background-color: #f0f0f0 !important;
        }

        #search-panel, #filter-panel {
            transition: all 0.3s ease;
        }

        .search-result {
            display: block;
            padding: 3px 6px;
            margin: 2px 0;
            border-radius: 3px;
            transition: background-color 0.2s;
        }

        /* Ensure message display is visible above other elements */
        #message-element {
            z-index: 9999 !important;
        }
    `;
    document.head.appendChild(style);

    // Function to toggle display of all matches
    function toggleMatches(lineId) {
        var matchesDiv = document.getElementById('all-matches-' + lineId);
        var button = document.getElementById('toggle-btn-' + lineId);

        if (matchesDiv) {
            if (matchesDiv.style.display === 'none') {
                matchesDiv.style.display = 'block';
                if (button) button.textContent = 'Hide All Matches';
            } else {
                matchesDiv.style.display = 'none';
                if (button) button.textContent = 'Show All Matches';
            }
        }
    }

    // SEARCH FUNCTION - Direct implementation
    function searchLines() {
        console.log("Search function called");
        var searchInput = document.getElementById('line-search');
        if (!searchInput) {
            console.error("Search input element not found");
            return;
        }

        var searchText = searchInput.value.trim().toLowerCase();
        console.log("Search text: '" + searchText + "'");

        if (!searchText) {
            showSearchResults('<p>Please enter an ID to search for.</p>');
            return;
        }

        var lineType = "all"; // Default value
        var radioButtons = document.querySelectorAll('input[name="line-type"]');
        for (var i = 0; i < radioButtons.length; i++) {
            if (radioButtons[i].checked) {
                lineType = radioButtons[i].value;
                break;
            }
        }
        console.log("Line type filter: " + lineType);

        var matches = [];
        var matchCount = 0;

        // Search through line coordinates
        for (var lineId in lineCoordinates) {
            var line = lineCoordinates[lineId];
            var type = line.type || '';
            var id = line.id || '';

            // Convert both to strings and lowercase for comparison
            id = id.toString().toLowerCase();

            // Filter by type if not "all"
            if (lineType !== 'all' && type !== lineType) {
                continue;
            }

            // Check if ID contains the search text
            if (id.includes(searchText)) {
                matchCount++;
                matches.push({
                    id: line.id,
                    fullId: lineId,
                    type: type,
                    isMatched: line.is_matched || false,
                    coordinates: line.coords || []
                });
            }
        }

        console.log("Search found " + matchCount + " matches");

        // Display results
        if (matches.length === 0) {
            showSearchResults('<p>No matches found for "' + searchText + '".</p>');
        } else {
            displaySearchResults(matches);
        }
    }

    // Display search results in HTML
    function displaySearchResults(matches) {
        var resultsDiv = document.getElementById('search-results');
        if (!resultsDiv) {
            console.error("Search results container not found");
            return;
        }

        // Clear previous results
        resultsDiv.innerHTML = '';

        // Limit to 20 results for performance
        var displayCount = Math.min(matches.length, 20);

        // Create results HTML
        var html = '<div style="font-weight: bold; margin-bottom: 5px; border-bottom: 1px solid #ddd; padding-bottom: 3px;">' +
                   'Found ' + matches.length + ' lines' + 
                   (matches.length > displayCount ? ' (showing first ' + displayCount + ')' : '') +
                   '</div>';

        html += '<ul style="padding-left: 15px; margin: 5px 0;">';

        // Sort by ID for better readability
        matches.sort(function(a, b) {
            return a.id.toString().localeCompare(b.id.toString());
        });

        // Add each match to the results
        for (var i = 0; i < displayCount; i++) {
            var match = matches[i];
            var typeName = getTypeDisplayName(match.type);
            var color = getLineColor(match.type, match.isMatched);

            html += '<li style="margin-bottom: 3px;">' +
                    '<a href="#" class="search-result" onclick="highlightLine(\'' + match.fullId + '\'); return false;" ' +
                    'style="color: ' + color + '; text-decoration: none; display: block;">' +
                    typeName + ' Line ' + match.id + (match.isMatched ? ' (matched)' : '') +
                    '</a></li>';
        }

        html += '</ul>';
        resultsDiv.innerHTML = html;

        // Make results stand out
        resultsDiv.style.backgroundColor = '#f8f8f8';
        resultsDiv.style.border = '1px solid #ddd';
        resultsDiv.style.borderRadius = '3px';
        resultsDiv.style.animation = 'flashBg 1s ease';
    }

    // Show search results with HTML
    function showSearchResults(html) {
        var resultsDiv = document.getElementById('search-results');
        if (resultsDiv) {
            resultsDiv.innerHTML = html;
            resultsDiv.style.display = html ? 'block' : 'none';
            resultsDiv.style.backgroundColor = '#f8f8f8';
            resultsDiv.style.border = '1px solid #ddd';
            resultsDiv.style.padding = '8px';

            // Add a subtle flash effect to draw attention to new results
            resultsDiv.style.animation = 'none';
            setTimeout(function() {
                resultsDiv.style.animation = 'flashBg 0.5s ease';
            }, 10);
        } else {
            console.error("Search results div not found");
        }
    }

    // Get display name for line type
    function getTypeDisplayName(type) {
        switch (type) {
            case 'dlr': return 'DLR';
            case 'network': return 'Network';
            case 'pypsa': return 'PyPSA';
            case 'pypsa_eur': return 'PyPSA-EUR';
            case 'pypsa_new': return 'PyPSA-new';
            case 'fifty_hertz': return '50Hertz';
            case 'tennet': return 'TenneT';
            default: return type;
        }
    }

    // Get color for line type
    function getLineColor(type, isMatched) {
        if (type === 'dlr') {
            return isMatched ? 'green' : 'blue';
        } else if (type === 'network') {
            return isMatched ? 'orange' : 'red';
        } else if (type === 'pypsa' || type === 'pypsa_eur') {
            return isMatched ? 'purple' : 'magenta';
        } else if (type === 'pypsa_new') {
            return isMatched ? 'yellow' : 'cyan';
        } else if (type === 'fifty_hertz') {
            return isMatched ? 'darkgreen' : 'green';
        } else if (type === 'tennet') {
            return isMatched ? 'darkblue' : 'blue';
        }
        return 'black'; // Default
    }

    // Highlight a specific line on the map
    function highlightLine(lineId) {
        // Clear any existing highlights
        clearHighlights();

        console.log("Highlighting line: " + lineId);

        var line = lineCoordinates[lineId];
        if (!line || !line.coords || line.coords.length < 2) {
            console.error("Line not found or has no coordinates", lineId);
            showMessage('Line not found or has no coordinates.', 3000);
            return;
        }

        // Determine which layer should be visible
        var layerName = "";
        if (line.type === 'dlr') {
            layerName = "DLR Lines (" + (line.is_matched ? "Matched" : "Unmatched") + ")";
        } else if (line.type === 'network') {
            layerName = "Network Lines (" + (line.is_matched ? "Matched" : "Unmatched") + ")";
        } else if (line.type === 'pypsa_eur') {
            layerName = "PyPSA-EUR Lines (" + (line.is_matched ? "Matched" : "Unmatched") + ")";
        } else if (line.type === 'pypsa_new') {
            layerName = "PyPSA-new Lines (" + (line.is_matched ? "Matched" : "Unmatched") + ")";
        } else if (line.type === 'fifty_hertz') {
            layerName = "50Hertz Lines (" + (line.is_matched ? "Matched" : "Unmatched") + ")";
        } else if (line.type === 'tennet') {
            layerName = "TenneT Lines (" + (line.is_matched ? "Matched" : "Unmatched") + ")";
        }

        // Find all paths with this line ID
        var paths = document.querySelectorAll('path.leaflet-interactive');
        var matchingPaths = [];

        // Try different methods to match the paths
        for (var i = 0; i < paths.length; i++) {
            var path = paths[i];
            // Method 1: Check className
            if (path.className && path.className.baseVal && 
                path.className.baseVal.includes(lineId)) {
                matchingPaths.push(path);
            }
            // Method 2: Check name attribute
            else if (path.getAttribute('name') === lineId) {
                matchingPaths.push(path);
            }
            // Method 3: Check data attributes
            else if (path.dataset && path.dataset.id === lineId) {
                matchingPaths.push(path);
            }
        }

        if (matchingPaths.length === 0) {
            // If no matching paths found with exact ID, try to match by line ID part
            // This handles cases where the className structure might be different
            var lineIdParts = lineId.split('_');
            var lineTypePart = lineIdParts[0];
            var lineIdPart = lineIdParts.length > 1 ? lineIdParts[1] : '';

            for (var i = 0; i < paths.length; i++) {
                var path = paths[i];
                var className = path.className && path.className.baseVal ? path.className.baseVal : '';

                // Check if both the line type and ID part are in the className
                if (className.includes(lineTypePart) && lineIdPart && className.includes(lineIdPart)) {
                    matchingPaths.push(path);
                }
            }
        }

        if (matchingPaths.length === 0) {
            // If still no matching paths found, check if the layer is hidden and enable it
            var layerWasEnabled = enableLayerByName(layerName);

            if (layerWasEnabled) {
                // Wait a moment for the layer to be displayed, then try again
                setTimeout(function() {
                    // Search again for the paths after enabling the layer
                    paths = document.querySelectorAll('path.leaflet-interactive');
                    for (var i = 0; i < paths.length; i++) {
                        var path = paths[i];
                        if (path.className && path.className.baseVal && 
                            path.className.baseVal.includes(lineId)) {
                            matchingPaths.push(path);
                        }
                    }

                    if (matchingPaths.length > 0) {
                        // Now we have matching paths, continue highlighting
                        highlightMatchingPaths(matchingPaths, line);
                    } else {
                        // Even if we couldn't find the exact path, zoom to the coordinates
                        // This ensures the user at least sees the right area
                        if (line.coords && line.coords.length > 0) {
                            var bounds = L.latLngBounds(line.coords);
                            map.fitBounds(bounds, { padding: [50, 50] });

                            // Create a temporary highlight along the line's coordinates
                            var tempLine = L.polyline(line.coords, {
                                color: 'white',
                                weight: 6,
                                opacity: 0.8,
                                dashArray: '5,10',
                                className: 'temp-highlight'
                            }).addTo(map);

                            // Remove after a few seconds
                            setTimeout(function() {
                                map.removeLayer(tempLine);
                            }, 5000);

                            showMessage("Zoomed to line area - enable layer in control panel to see it", 3000);
                        } else {
                            showMessage("Line found but couldn't highlight it or zoom to area", 3000);
                        }
                    }
                }, 500); // Wait 500ms for the layer to render

                return;
            } else {
                // Even if we couldn't enable the layer, zoom to the coordinates if available
                if (line.coords && line.coords.length > 0) {
                    var bounds = L.latLngBounds(line.coords);
                    map.fitBounds(bounds, { padding: [50, 50] });
                    showMessage("Line found - enable the \"" + layerName + "\" layer to see it", 3000);
                } else {
                    showMessage("Couldn't find or enable the required layer: " + layerName, 3000);
                }
                return;
            }
        }

        // Highlight the matching paths
        highlightMatchingPaths(matchingPaths, line);
    }

    // Function to highlight matching paths
    function highlightMatchingPaths(matchingPaths, line) {
        // Highlight matching paths
        matchingPaths.forEach(function(path) {
            // Store original style
            path._originalStyle = {
                'stroke': path.getAttribute('stroke'),
                'stroke-width': path.getAttribute('stroke-width'),
                'stroke-opacity': path.getAttribute('stroke-opacity')
            };

            // Apply highlight style
            path.setAttribute('stroke', 'white');
            path.setAttribute('stroke-width', '6');
            path.setAttribute('stroke-opacity', '1');

            currentHighlightedLines.push(path);
        });

        // Add a marker at the midpoint
        if (line.midpoint && line.midpoint.length === 2) {
            var markerColor = getLineColor(line.type, line.is_matched);

            var icon = L.divIcon({
                html: '<div style="background-color: white; border: 3px solid ' + markerColor + 
                    '; width: 16px; height: 16px; border-radius: 50%;"></div>',
                className: 'search-marker',
                iconSize: [16, 16],
                iconAnchor: [8, 8]
            });

            currentSearchMarker = L.marker(line.midpoint, { icon: icon }).addTo(map);
        }

        // Zoom to fit the line
        if (line.coords && line.coords.length > 0) {
            var bounds = L.latLngBounds(line.coords);
            map.fitBounds(bounds, { padding: [50, 50] });
        }

        // Find and open popup if possible
        map.eachLayer(function(layer) {
            if (layer instanceof L.Polyline && 
                layer.options && 
                layer.options.className && 
                layer.options.className.includes(line.id)) {
                if (layer.getPopup) {
                    try {
                        layer.openPopup();
                    } catch(e) {
                        console.error("Error opening popup:", e);
                    }
                }
            }
        });
    }

    // Function to enable a layer by name
    function enableLayerByName(layerName) {
        var layerControl = document.querySelector('.leaflet-control-layers-list');
        if (!layerControl) {
            console.error("Layer control not found");
            return false;
        }

        var labels = layerControl.querySelectorAll('span');
        var layerFound = false;

        for (var i = 0; i < labels.length; i++) {
            var label = labels[i];
            var labelText = label.textContent || label.innerText;

            // Check if this label contains our layer name
            if (labelText.includes(layerName)) {
                var input = label.parentNode.querySelector('input[type="checkbox"]');
                layerFound = true;

                if (input && !input.checked) {
                    console.log("Enabling layer: " + layerName);
                    input.click(); // Toggle the checkbox to enable
                    showMessage("Enabling layer: " + layerName, 2000);
                    return true;
                } else if (input && input.checked) {
                    // Layer is already enabled
                    return true;
                }
                break;
            }
        }

        // If we get here, either the layer wasn't found or it couldn't be enabled
        if (!layerFound) {
            console.error("Layer not found: " + layerName);
        }
        return false;
    }

    // Clear highlighted lines
    function clearHighlights() {
        // Remove marker
        if (currentSearchMarker) {
            map.removeLayer(currentSearchMarker);
            currentSearchMarker = null;
        }

        // Reset highlighted lines
        currentHighlightedLines.forEach(function(path) {
            if (path._originalStyle) {
                path.setAttribute('stroke', path._originalStyle['stroke']);
                path.setAttribute('stroke-width', path._originalStyle['stroke-width']);
                path.setAttribute('stroke-opacity', path._originalStyle['stroke-opacity']);
                delete path._originalStyle;
            }
        });

        currentHighlightedLines = [];
    }

    // Show a temporary message
    function showMessage(text, duration) {
        var messageElement = document.getElementById('message-element');

        if (!messageElement) {
            messageElement = document.createElement('div');
            messageElement.id = 'message-element';
            messageElement.style.position = 'fixed';
            messageElement.style.top = '50%';
            messageElement.style.left = '50%';
            messageElement.style.transform = 'translate(-50%, -50%)';
            messageElement.style.padding = '10px 20px';
            messageElement.style.backgroundColor = 'rgba(0, 0, 0, 0.7)';
            messageElement.style.color = 'white';
            messageElement.style.borderRadius = '5px';
            messageElement.style.zIndex = '9999';
            messageElement.style.pointerEvents = 'none';
            document.body.appendChild(messageElement);
        }

        messageElement.textContent = text;
        messageElement.style.display = 'block';

        setTimeout(function() {
            messageElement.style.display = 'none';
        }, duration || 2000);
    }

    // Filter functionality
    function applyFilters() {
        console.log("Applying filters");

        // Get checkbox states
        var showDlrMatched = document.getElementById('filter-dlr-matched').checked;
        var showDlrUnmatched = document.getElementById('filter-dlr-unmatched').checked;
        var showNetworkMatched = document.getElementById('filter-network-matched').checked;
        var showNetworkUnmatched = document.getElementById('filter-network-unmatched').checked;

        // Toggle layers
        toggleLayer('DLR Lines (Matched)', showDlrMatched);
        toggleLayer('DLR Lines (Unmatched)', showDlrUnmatched);
        toggleLayer('Network Lines (Matched)', showNetworkMatched);
        toggleLayer('Network Lines (Unmatched)', showNetworkUnmatched);

        // Handle PyPSA-EUR layers
        var pypsa_eur_matched = document.getElementById('filter-pypsa-eur-matched');
        var pypsa_eur_unmatched = document.getElementById('filter-pypsa-eur-unmatched');

        if (pypsa_eur_matched && pypsa_eur_unmatched) {
            toggleLayer('PyPSA-EUR Lines (Matched)', pypsa_eur_matched.checked);
            toggleLayer('PyPSA-EUR Lines (Unmatched)', pypsa_eur_unmatched.checked);
        }

        // Handle 50Hertz layers
        var fifty_hertz_matched = document.getElementById('filter-fifty-hertz-matched');
        var fifty_hertz_unmatched = document.getElementById('filter-fifty-hertz-unmatched');

        if (fifty_hertz_matched && fifty_hertz_unmatched) {
            toggleLayer('50Hertz Lines (Matched)', fifty_hertz_matched.checked);
            toggleLayer('50Hertz Lines (Unmatched)', fifty_hertz_unmatched.checked);
        }

        // Handle TenneT layers
        var tennet_matched = document.getElementById('filter-tennet-matched');
        var tennet_unmatched = document.getElementById('filter-tennet-unmatched');

        if (tennet_matched && tennet_unmatched) {
            toggleLayer('TenneT Lines (Matched)', tennet_matched.checked);
            toggleLayer('TenneT Lines (Unmatched)', tennet_unmatched.checked);
        }

        showMessage('Filters applied', 2000);
    }

    // Toggle a layer's visibility
    function toggleLayer(layerName, show) {
        console.log("Toggling layer: " + layerName + " to " + (show ? "show" : "hide"));

        // Find layer controls
        var layerControl = document.querySelector('.leaflet-control-layers-list');
        if (!layerControl) {
            console.error("Layer control not found");
            return;
        }

        var labels = layerControl.querySelectorAll('span');

        for (var i = 0; i < labels.length; i++) {
            var label = labels[i];
            var labelText = label.textContent || label.innerText;

            // Check if this label contains our layer name
            if (labelText.includes(layerName)) {
                var input = label.parentNode.querySelector('input[type="checkbox"]');
                if (input && input.checked !== show) {
                    console.log("Clicking layer checkbox for: " + layerName);
                    input.click(); // Toggle the checkbox
                }
                break;
            }
        }
    }

    // Reset filters
    function resetFilters() {
        console.log("Resetting filters");

        document.getElementById('filter-dlr-matched').checked = true;
        document.getElementById('filter-dlr-unmatched').checked = true;
        document.getElementById('filter-network-matched').checked = true;
        document.getElementById('filter-network-unmatched').checked = true;

        // Reset PyPSA-EUR filters if they exist
        var pypsa_eur_matched = document.getElementById('filter-pypsa-eur-matched');
        var pypsa_eur_unmatched = document.getElementById('filter-pypsa-eur-unmatched');

        if (pypsa_eur_matched && pypsa_eur_unmatched) {
            pypsa_eur_matched.checked = true;
            pypsa_eur_unmatched.checked = true;
        }

        // Reset 50Hertz filters if they exist
        var fifty_hertz_matched = document.getElementById('filter-fifty-hertz-matched');
        var fifty_hertz_unmatched = document.getElementById('filter-fifty-hertz-unmatched');

        if (fifty_hertz_matched && fifty_hertz_unmatched) {
            fifty_hertz_matched.checked = true;
            fifty_hertz_unmatched.checked = true;
        }

        // Reset TenneT filters if they exist
        var tennet_matched = document.getElementById('filter-tennet-matched');
        var tennet_unmatched = document.getElementById('filter-tennet-unmatched');

        if (tennet_matched && tennet_unmatched) {
            tennet_matched.checked = true;
            tennet_unmatched.checked = true;
        }

        // Apply the reset filters
        applyFilters();

        showMessage('All filters reset', 2000);
    }

    // Add a special debug function to check what's happening
    function debugMap() {
        console.log("Debug function called");

        // Check if the search button is found
        var searchBtn = document.getElementById('search-button');
        console.log("Search button exists:", searchBtn !== null);

        // Check input field
        var searchInput = document.getElementById('line-search');
        console.log("Search input exists:", searchInput !== null);

        // Test path counts
        var paths = document.querySelectorAll('path.leaflet-interactive');
        console.log("Total interactive paths:", paths.length);

        // Test layer controls
        var layerControl = document.querySelector('.leaflet-control-layers-list');
        console.log("Layer control exists:", layerControl !== null);

        if (layerControl) {
            var labels = layerControl.querySelectorAll('span');
            console.log("Layer labels found:", labels.length);
        }

        // Show message
        showMessage("Debug info logged to console (F12)", 3000);
    }

    // SETUP THE EVENT LISTENERS
    // Use multiple approaches to ensure they're attached

    // Method 1: Window load event
    window.onload = function() {
        console.log("Window loaded, setting up handlers");
        setupEventHandlers();
    };

    // Method 2: DOMContentLoaded event
    document.addEventListener('DOMContentLoaded', function() {
        console.log("DOM content loaded, setting up handlers");
        setupEventHandlers();
    });

    // Method 3: Direct script execution with a slight delay
    setTimeout(function() {
        console.log("Delayed execution, setting up handlers");
        setupEventHandlers();
    }, 500);

    // Method 4: Retry a few times if needed
    var setupAttempts = 0;
    var setupInterval = setInterval(function() {
        setupAttempts++;
        var searchBtn = document.getElementById('search-button');
        if (searchBtn || setupAttempts >= 5) {
            console.log("Setup attempt " + setupAttempts + ", search button found: " + (searchBtn !== null));
            setupEventHandlers();
            clearInterval(setupInterval);
        }
    }, 1000);

    // Central setup function
    function setupEventHandlers() {
        // Set up search button using multiple methods
        var searchButton = document.getElementById('search-button');
        if (searchButton) {
            // Method 1: addEventListener
            searchButton.addEventListener('click', function() {
                console.log("Search button clicked via addEventListener");
                searchLines();
            });

            // Method 2: onclick property
            searchButton.onclick = function() {
                console.log("Search button clicked via onclick");
                searchLines();
            };

            console.log("Search button handlers attached");
        }

        // Set up search input Enter key
        var searchInput = document.getElementById('line-search');
        if (searchInput) {
            searchInput.addEventListener('keypress', function(e) {
                if (e.key === 'Enter') {
                    console.log("Enter key pressed in search input");
                    searchLines();
                }
            });
        }

        // Set up filter buttons
        var applyFilterButton = document.getElementById('apply-filter');
        if (applyFilterButton) {
            applyFilterButton.addEventListener('click', applyFilters);
        }

        var resetFilterButton = document.getElementById('reset-filter');
        if (resetFilterButton) {
            resetFilterButton.addEventListener('click', resetFilters);
        }

        // Add a debug button
        var debugBtn = document.createElement('button');
        debugBtn.innerHTML = 'Debug Map';
        debugBtn.style.position = 'fixed';
        debugBtn.style.bottom = '10px';
        debugBtn.style.right = '10px';
        debugBtn.style.zIndex = '9999';
        debugBtn.style.backgroundColor = '#f0f0f0';
        debugBtn.style.border = '1px solid #ccc';
        debugBtn.style.borderRadius = '4px';
        debugBtn.style.padding = '5px 10px';
        debugBtn.onclick = debugMap;
        document.body.appendChild(debugBtn);
    }
    </script>
    """

    # Define HTML variables to avoid IDE warnings
    # Add stats panel
    stats_html = f"""
    <div id="stats-panel" style="
        position: fixed; 
        top: 10px; 
        left: 10px; 
        width: 220px; 
        padding: 10px; 
        background-color: white; 
        border-radius: 5px; 
        box-shadow: 0 0 10px rgba(0,0,0,0.2); 
        z-index: 999;">
        <h4 style="margin-top: 0;">Statistics</h4>

        <h5 style="margin: 10px 0 5px;">Network Lines: {network_lines_germany_count}</h5>
        <p style="margin: 3px 0; padding-left: 10px;">- Matched to any: {len(all_matched_network_ids)} ({format_rate(len(all_matched_network_ids), network_lines_germany_count)}%)</p>
        <p style="margin: 3px 0; padding-left: 10px;">- Unmatched: {network_lines_germany_count - len(all_matched_network_ids)}</p>

        <h5 style="margin: 10px 0 5px;">DLR Lines: {dlr_lines_germany_count}</h5>
        <p style="margin: 3px 0; padding-left: 10px;">- Matched: {len(matched_dlr_ids)}</p>
        <p style="margin: 3px 0; padding-left: 10px;">- Unmatched: {dlr_lines_germany_count - len(matched_dlr_ids)}</p>
        <p style="margin: 3px 0; padding-left: 10px;">- Match Rate: {format_rate(len(matched_dlr_ids), dlr_lines_germany_count)}%</p>
    """

    # Add PyPSA-EUR stats if available
    if pypsa_lines is not None and not pypsa_lines.empty:
        stats_html += f"""
        <h5 style="margin: 10px 0 5px;">PyPSA-EUR Lines: {pypsa_lines_germany_count}</h5>
        <p style="margin: 3px 0; padding-left: 10px;">- Matched: {len(matched_pypsa_eur_ids)}</p>
        <p style="margin: 3px 0; padding-left: 10px;">- Unmatched: {pypsa_lines_germany_count - len(matched_pypsa_eur_ids)}</p>
        <p style="margin: 3px 0; padding-left: 10px;">- Match Rate: {format_rate(len(matched_pypsa_eur_ids), pypsa_lines_germany_count)}%</p>
        """

    # Add 50Hertz stats if available
    if fifty_hertz_lines is not None and not fifty_hertz_lines.empty:
        stats_html += f"""
        <h5 style="margin: 10px 0 5px;">50Hertz Lines: {fifty_hertz_lines_germany_count}</h5>
        <p style="margin: 3px 0; padding-left: 10px;">- Matched: {len(matched_fifty_hertz_ids)}</p>
        <p style="margin: 3px 0; padding-left: 10px;">- Unmatched: {fifty_hertz_lines_germany_count - len(matched_fifty_hertz_ids)}</p>
        <p style="margin: 3px 0; padding-left: 10px;">- Match Rate: {format_rate(len(matched_fifty_hertz_ids), fifty_hertz_lines_germany_count)}%</p>
        """

    # Add TenneT stats if available
    if tennet_lines is not None and not tennet_lines.empty:
        stats_html += f"""
        <h5 style="margin: 10px 0 5px;">TenneT Lines: {tennet_lines_germany_count}</h5>
        <p style="margin: 3px 0; padding-left: 10px;">- Matched: {len(matched_tennet_ids)}</p>
        <p style="margin: 3px 0; padding-left: 10px;">- Unmatched: {tennet_lines_germany_count - len(matched_tennet_ids)}</p>
        <p style="margin: 3px 0; padding-left: 10px;">- Match Rate: {format_rate(len(matched_tennet_ids), tennet_lines_germany_count)}%</p>
        """

    stats_html += "<p style='font-style: italic; font-size: 10px; margin-top: 8px;'>All statistics are for lines inside Germany only.</p>"
    stats_html += "</div>"

    # Add PyPSA-EUR stats if available
    if pypsa_lines is not None and not pypsa_lines.empty:
        stats_html += f"""
            <h5 style="margin: 10px 0 5px;">PyPSA-EUR Lines: {pypsa_lines_germany_count}</h5>
            <p style="margin: 3px 0; padding-left: 10px;">- Matched: {len(matched_pypsa_eur_ids)}</p>
            <p style="margin: 3px 0; padding-left: 10px;">- Unmatched: {pypsa_lines_germany_count - len(matched_pypsa_eur_ids)}</p>
            <p style="margin: 3px 0; padding-left: 10px;">- Match Rate: {len(matched_pypsa_eur_ids) / pypsa_lines_germany_count * 100 if pypsa_lines_germany_count > 0 else 0:.1f}%</p>
            """

    # Add 50Hertz stats if available
    if fifty_hertz_lines is not None and not fifty_hertz_lines.empty:
        stats_html += f"""
            <h5 style="margin: 10px 0 5px;">50Hertz Lines: {fifty_hertz_lines_germany_count}</h5>
            <p style="margin: 3px 0; padding-left: 10px;">- Matched: {len(matched_fifty_hertz_ids)}</p>
            <p style="margin: 3px 0; padding-left: 10px;">- Unmatched: {fifty_hertz_lines_germany_count - len(matched_fifty_hertz_ids)}</p>
            <p style="margin: 3px 0; padding-left: 10px;">- Match Rate: {len(matched_fifty_hertz_ids) / fifty_hertz_lines_germany_count * 100 if fifty_hertz_lines_germany_count > 0 else 0:.1f}%</p>
            """

    # Add TenneT stats if available
    if tennet_lines is not None and not tennet_lines.empty:
        stats_html += f"""
            <h5 style="margin: 10px 0 5px;">TenneT Lines: {tennet_lines_germany_count}</h5>
            <p style="margin: 3px 0; padding-left: 10px;">- Matched: {len(matched_tennet_ids)}</p>
            <p style="margin: 3px 0; padding-left: 10px;">- Unmatched: {tennet_lines_germany_count - len(matched_tennet_ids)}</p>
            <p style="margin: 3px 0; padding-left: 10px;">- Match Rate: {len(matched_tennet_ids) / tennet_lines_germany_count * 100 if tennet_lines_germany_count > 0 else 0:.1f}%</p>
            """

    stats_html += "<p style='font-style: italic; font-size: 10px; margin-top: 8px;'>All statistics are for lines inside Germany only.</p>"
    stats_html += "</div>"

    # Add legend HTML
    legend_html = """
    <div id="legend-panel" style="
        position: fixed; 
        bottom: 10px; 
        left: 10px; 
        width: 300px; 
        padding: 10px; 
        background-color: white; 
        border-radius: 5px; 
        box-shadow: 0 0 10px rgba(0,0,0,0.2); 
        z-index: 1000;">
        <h4 style="margin-top: 0;">Map Legend</h4>

        <!-- DLR Lines -->
        <div style="display: flex; align-items: center; margin-bottom: 5px;">
            <div style="width: 20px; height: 5px; background-color: blue; margin-right: 10px;"></div>
            <span>DLR Lines (Unmatched)</span>
        </div>
        <div style="display: flex; align-items: center; margin-bottom: 5px;">
            <div style="width: 20px; height: 5px; background-color: green; margin-right: 10px;"></div>
            <span>DLR Lines (Matched)</span>
        </div>

        <!-- Network Lines -->
        <div style="display: flex; align-items: center; margin-bottom: 5px;">
            <div style="width: 20px; height: 5px; background-color: red; margin-right: 10px;"></div>
            <span>Network Lines (Unmatched)</span>
        </div>
        <div style="display: flex; align-items: center; margin-bottom: 5px;">
            <div style="width: 20px; height: 5px; background-color: orange; margin-right: 10px;"></div>
            <span>Network Lines (Matched)</span>
        </div>
    """

    # Add PyPSA legend if available
    if pypsa_lines is not None:
        legend_html += """
        <!-- PyPSA Lines -->
        <div style="display: flex; align-items: center; margin-bottom: 5px;">
            <div style="width: 20px; height: 5px; background-color: magenta; margin-right: 10px; border-top: 1px dashed #000;"></div>
            <span>PyPSA Lines (Unmatched)</span>
        </div>
        <div style="display: flex; align-items: center; margin-bottom: 5px;">
            <div style="width: 20px; height: 5px; background-color: purple; margin-right: 10px; border-top: 1px dashed #000;"></div>
            <span>PyPSA Lines (Matched)</span>
        </div>
        """

    # Add PyPSA-new legend if available
    if pypsa_lines_new is not None:
        legend_html += """
        <!-- PyPSA-new Lines -->
        <div style="display: flex; align-items: center; margin-bottom: 5px;">
            <div style="width: 20px; height: 5px; background-color: cyan; margin-right: 10px;"></div>
            <span>PyPSA-new Lines (Unmatched)</span>
        </div>
        <div style="display: flex; align-items: center; margin-bottom: 5px;">
            <div style="width: 20px; height: 5px; background-color: yellow; margin-right: 10px;"></div>
            <span>PyPSA-new Lines (Matched)</span>
        </div>
        """

    # Add 50Hertz legend if available
    if fifty_hertz_lines is not None:
        legend_html += """
        <!-- 50Hertz Lines -->
        <div style="display: flex; align-items: center; margin-bottom: 5px;">
            <div style="width: 20px; height: 5px; background-color: green; margin-right: 10px;"></div>
            <span>50Hertz Lines (Unmatched)</span>
        </div>
        <div style="display: flex; align-items: center; margin-bottom: 5px;">
            <div style="width: 20px; height: 5px; background-color: darkgreen; margin-right: 10px;"></div>
            <span>50Hertz Lines (Matched)</span>
        </div>
        """

    # Add TenneT legend if available
    if tennet_lines is not None:
        legend_html += """
        <!-- TenneT Lines -->
        <div style="display: flex; align-items: center; margin-bottom: 5px;">
            <div style="width: 20px; height: 5px; background-color: blue; margin-right: 10px;"></div>
            <span>TenneT Lines (Unmatched)</span>
        </div>
        <div style="display: flex; align-items: center; margin-bottom: 5px;">
            <div style="width: 20px; height: 5px; background-color: darkblue; margin-right: 10px;"></div>
            <span>TenneT Lines (Matched)</span>
        </div>
        """

    legend_html += "</div>"

    # Add search box interface
    search_html = """
    <div id="search-panel" style="
        position: fixed; 
        top: 10px;
        right: 10px; 
        width: 250px; 
        padding: 10px; 
        background-color: white; 
        border-radius: 5px; 
        box-shadow: 0 0 10px rgba(0,0,0,0.2); 
        z-index: 1000;">
        <h4 style="margin-top: 0; margin-bottom: 10px;">Search Lines</h4>
        <input type="text" id="line-search" placeholder="Enter line ID..." 
            style="width: 100%; padding: 5px; border: 1px solid #ccc; border-radius: 3px; margin-bottom: 10px;"
            onkeypress="if(event.key==='Enter') { searchLines(); return false; }">

        <div style="max-height: 120px; overflow-y: auto; margin-bottom: 10px;">
            <div>
                <label style="display: block; margin-bottom: 5px;">
                    <input type="radio" name="line-type" value="all" checked> All Lines
                </label>
            </div>
            <div>
                <label style="display: block; margin-bottom: 5px;">
                    <input type="radio" name="line-type" value="dlr"> DLR Lines
                </label>
            </div>
            <div>
                <label style="display: block; margin-bottom: 5px;">
                    <input type="radio" name="line-type" value="network"> Network Lines
                </label>
            </div>
    """

    # Add the other radio buttons
    if pypsa_lines is not None:
        search_html += """
        <div>
            <label style="display: block; margin-bottom: 5px;">
                <input type="radio" name="line-type" value="pypsa_eur"> PyPSA Lines
            </label>
        </div>
        """

    if fifty_hertz_lines is not None:
        search_html += """
        <div>
            <label style="display: block; margin-bottom: 5px;">
                <input type="radio" name="line-type" value="fifty_hertz"> 50Hertz Lines
            </label>
        </div>
        """

    if tennet_lines is not None:
        search_html += """
        <div>
            <label style="display: block; margin-bottom: 5px;">
                <input type="radio" name="line-type" value="tennet"> TenneT Lines
            </label>
        </div>
        """

    search_html += """
        </div>
        <button id="search-button" onclick="searchLines(); return false;" 
            style="width: 100%; margin-top: 5px; padding: 5px; background-color: #4CAF50; color: white; border: none; border-radius: 3px; cursor: pointer;">
            Search
        </button>

        <div id="search-results" style="
            margin-top: 10px; 
            max-height: 150px; 
            overflow-y: auto;
            background-color: white;
            padding: 5px;
            border: 1px solid #ddd;
            border-radius: 3px;
        "></div>
    </div>
    """

    # Add filter panel
    filter_html = """
    <div id="filter-panel" style="
        position: fixed; 
        bottom: 40px;  /* Position at bottom instead of top */
        right: 10px; 
        width: 250px; 
        padding: 10px; 
        background-color: white; 
        border-radius: 5px; 
        box-shadow: 0 0 10px rgba(0,0,0,0.2); 
        z-index: 999;
        max-height: 50vh;
        overflow-y: auto;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
            <h4 style="margin: 0;">Filter Map Lines</h4>
            <button id="toggle-filter" style="
                padding: 2px 6px;
                background-color: #f0f0f0;
                border: 1px solid #ccc;
                border-radius: 3px;
                cursor: pointer;
                font-size: 10px;">
                Collapse
            </button>
        </div>
        <div id="filter-content">
            <h5 style="margin: 10px 0 5px;">DLR Lines</h5>
            <div class="filter-item">
                <input type="checkbox" id="filter-dlr-matched" class="filter-checkbox" checked>
                <label for="filter-dlr-matched">Matched</label>
            </div>
            <div class="filter-item">
                <input type="checkbox" id="filter-dlr-unmatched" class="filter-checkbox" checked>
                <label for="filter-dlr-unmatched">Unmatched</label>
            </div>

            <h5 style="margin: 10px 0 5px;">Network Lines</h5>
            <div class="filter-item">
                <input type="checkbox" id="filter-network-matched" class="filter-checkbox" checked>
                <label for="filter-network-matched">Matched</label>
            </div>
            <div class="filter-item">
                <input type="checkbox" id="filter-network-unmatched" class="filter-checkbox" checked>
                <label for="filter-network-unmatched">Unmatched</label>
            </div>
    """

    # Add PyPSA-EUR filters if available
    if pypsa_lines is not None:
        filter_html += """
        <h5 style="margin: 10px 0 5px;">PyPSA-EUR Lines</h5>
        <div class="filter-item">
            <input type="checkbox" id="filter-pypsa-eur-matched" class="filter-checkbox" checked>
            <label for="filter-pypsa-eur-matched">Matched</label>
        </div>
        <div class="filter-item">
            <input type="checkbox" id="filter-pypsa-eur-unmatched" class="filter-checkbox" checked>
            <label for="filter-pypsa-eur-unmatched">Unmatched</label>
        </div>
        """

    # Add 50Hertz filters if available
    if fifty_hertz_lines is not None:
        filter_html += """
        <h5 style="margin: 10px 0 5px;">50Hertz Lines</h5>
        <div class="filter-item">
            <input type="checkbox" id="filter-fifty-hertz-matched" class="filter-checkbox" checked>
            <label for="filter-fifty-hertz-matched">Matched</label>
        </div>
        <div class="filter-item">
            <input type="checkbox" id="filter-fifty-hertz-unmatched" class="filter-checkbox" checked>
            <label for="filter-fifty-hertz-unmatched">Unmatched</label>
        </div>
        """

    # Add TenneT filters if available
    if tennet_lines is not None:
        filter_html += """
        <h5 style="margin: 10px 0 5px;">TenneT Lines</h5>
        <div class="filter-item">
            <input type="checkbox" id="filter-tennet-matched" class="filter-checkbox" checked>
            <label for="filter-tennet-matched">Matched</label>
        </div>
        <div class="filter-item">
            <input type="checkbox" id="filter-tennet-unmatched" class="filter-checkbox" checked>
            <label for="filter-tennet-unmatched">Unmatched</label>
        </div>
        """

    filter_html += """
            <div style="margin-top: 15px; display: flex; justify-content: space-between;">
                <button id="apply-filter" style="padding: 5px 10px; background-color: #4CAF50; color: white; border: none; border-radius: 3px; cursor: pointer; flex: 1; margin-right: 5px;">Apply</button>
                <button id="reset-filter" style="padding: 5px 10px; background-color: #f44336; color: white; border: none; border-radius: 3px; cursor: pointer; flex: 1; margin-left: 5px;">Reset</button>
            </div>
        </div>
    </div>
    """

    # Add title
    title_html = """
    <div id="title-panel" style="
        position: fixed; 
        top: 10px; 
        left: 50%; 
        transform: translateX(-50%);
        background-color: white; 
        border-radius: 5px; 
        padding: 10px; 
        z-index: 9999; 
        font-size: 16px;
        font-weight: bold;
        box-shadow: 0 0 15px rgba(0,0,0,0.2);">
        German Grid Network Comparison
        <button onclick="debugMap()" style="margin-left: 10px; font-size: 10px; padding: 2px 5px;">Debug</button>
    </div>
    """

    # Force HTML elements to be treated as strings to avoid IDE warnings
    m.get_root().html.add_child(folium.Element(str(stats_html)))
    m.get_root().html.add_child(folium.Element(str(legend_html)))
    m.get_root().html.add_child(folium.Element(str(search_html)))
    m.get_root().html.add_child(folium.Element(str(filter_html)))
    m.get_root().html.add_child(folium.Element(str(title_html)))
    m.get_root().html.add_child(folium.Element(str(javascript)))

    # Get the bounds of Germany
    if germany_gdf is not None and not germany_gdf.empty:
        bounds = germany_gdf.total_bounds  # (minx, miny, maxx, maxy)

        # Set the map bounds with a small buffer
        buffer = 0.5  # degrees
        sw = [bounds[1] - buffer, bounds[0] - buffer]  # [lat, lon] for southwest
        ne = [bounds[3] + buffer, bounds[2] + buffer]  # [lat, lon] for northeast

        # Fit bounds to Germany
        m.fit_bounds([sw, ne])
        logger.info(f"Set map bounds to Germany: {sw} to {ne}")

    # Save map
    m.save(output_file)
    logger.info(f"Comprehensive map saved to {output_file}")

    return m
