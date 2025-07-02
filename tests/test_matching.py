import pandas as pd
import networkx as nx
import folium
from folium.plugins import MarkerCluster
from shapely.geometry import LineString, Point
import geopandas as gpd
import numpy as np
from pathlib import Path
import os
from shapely.ops import nearest_points
from scipy.spatial import cKDTree
import json
import uuid

# Define file paths
data_dir = Path('/home/mohsen/PycharmProjects/freeGon/grid-matching-tool/data/clipped')
dlr_lines_path = data_dir / 'dlr-lines-germany.csv'
network_lines_path = data_dir / 'network-lines-germany.csv'
output_dir = Path('/home/mohsen/PycharmProjects/freeGon/grid-matching-tool/tests')


# Function to parse linestring from CSV
from shapely import wkt               # add once at the top

def parse_linestring(wkt_str: str):
    """Return the exact geometry written in WKT (LINESTRING or MULTILINESTRING)."""
    try:
        return wkt.loads(wkt_str)
    except Exception as exc:
        print(f"[parse_linestring] bad WKT → {exc}  |  {wkt_str[:80]}…")
        return None



# Load DLR lines data
def load_dlr_lines():
    dlr_df = pd.read_csv(dlr_lines_path)
    # Create GeoDataFrame
    geometry = dlr_df['geometry'].apply(parse_linestring)
    dlr_gdf = gpd.GeoDataFrame(dlr_df, geometry=geometry)
    dlr_gdf = dlr_gdf.explode(index_parts=False, ignore_index=True)


    # Extract start and end points
    dlr_gdf['start_point'] = dlr_gdf.geometry.apply(lambda x: Point(x.coords[0]))
    dlr_gdf['end_point'] = dlr_gdf.geometry.apply(lambda x: Point(x.coords[-1]))

    return dlr_gdf


# Load network lines data, excluding 110kV lines
def load_network_lines():
    network_df = pd.read_csv(network_lines_path)

    # Exclude 110kV lines
    network_df = network_df[network_df['v_nom'] != 110]

    # Create GeoDataFrame
    geometry = network_df['geometry'].apply(parse_linestring)
    network_gdf = gpd.GeoDataFrame(network_df, geometry=geometry)
    network_gdf = network_gdf.explode(index_parts=False, ignore_index=True)

    # Extract start and end points
    network_gdf['start_point'] = network_gdf.geometry.apply(lambda x: Point(x.coords[0]))
    network_gdf['end_point'] = network_gdf.geometry.apply(lambda x: Point(x.coords[-1]))

    # Reset index to make sure it's continuous
    network_gdf = network_gdf.reset_index(drop=True)

    return network_gdf


# Find the nearest points in the network for DLR endpoints
def find_nearest_points(dlr_gdf, network_gdf):
    # Create a function to find the nearest point
    def find_nearest_point(point, points_array, points_indices):
        # Find the closest point (without using a buffer)
        closest_idx = np.argmin([point.distance(Point(coords)) for coords in points_array])
        return points_indices[closest_idx]

    # Create arrays of network endpoints for search
    network_points = []
    network_point_indices = []

    # Collect all network endpoints (both start and end)
    for idx, row in network_gdf.iterrows():
        network_points.append([row.start_point.x, row.start_point.y])
        network_point_indices.append((idx, 'start'))

        network_points.append([row.end_point.x, row.end_point.y])
        network_point_indices.append((idx, 'end'))

    # Find nearest network points for each DLR line
    nearest_points_dict = {}

    for idx, row in dlr_gdf.iterrows():
        # Find nearest point for DLR start point
        start_nearest = find_nearest_point(row.start_point, network_points, network_point_indices)

        # Find nearest point for DLR end point
        end_nearest = find_nearest_point(row.end_point, network_points, network_point_indices)

        nearest_points_dict[idx] = {
            'dlr_id': row['id'],
            'start_nearest': start_nearest,
            'end_nearest': end_nearest,
            'v_nom': row['v_nom']
        }

    return nearest_points_dict


# Build network graph from network lines
# Build network graph from network lines
def build_network_graph(network_gdf):
    # Create graph
    G = nx.Graph()

    # Add nodes and edges for each network line
    for idx, row in network_gdf.iterrows():
        # Add unique node IDs for start and end points
        start_node = f"node_{idx}_start"
        end_node = f"node_{idx}_end"

        # Add nodes with coordinates and voltage info
        G.add_node(start_node, pos=(row.start_point.x, row.start_point.y), voltage=int(row['v_nom']))
        G.add_node(end_node, pos=(row.end_point.x, row.end_point.y), voltage=int(row['v_nom']))

        # Add edge for the line itself - this is a real network line, not a connector
        G.add_edge(start_node, end_node, weight=float(row.geometry.length), id=str(row['id']), idx=int(idx),
                   voltage=int(row['v_nom']), connector=False)

    # Add connections only between nodes that are very close to each other (endpoints of different lines)
    nodes = list(G.nodes())
    positions = nx.get_node_attributes(G, 'pos')

    # Very small buffer for connecting only coincident points (1 meter ~ 0.00001 degrees)
    buffer_distance = 0.00001

    for i, node1 in enumerate(nodes):
        for j, node2 in enumerate(nodes[i + 1:], i + 1):
            # Don't connect nodes from the same line
            if node1.split('_')[1] == node2.split('_')[1]:
                continue

            pos1 = positions[node1]
            pos2 = positions[node2]

            # Calculate distance
            dist = ((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2) ** 0.5

            # Only connect if very close
            if dist < buffer_distance:
                # This is a connector edge, not a real network line
                G.add_edge(node1, node2, weight=float(dist), connector=True)

    return G


# Find matching network lines for DLR lines
# Find matching network lines for DLR lines - modified to only track actual network segments
def find_matching_network_lines(dlr_gdf, network_gdf, nearest_points_dict, G):
    results = []

    for idx, row in dlr_gdf.iterrows():
        nearest_points = nearest_points_dict[idx]
        dlr_voltage = int(row['v_nom'])

        if nearest_points['start_nearest'] is None or nearest_points['end_nearest'] is None:
            # No matching endpoints found
            results.append({
                'dlr_id': str(row['id']),
                'dlr_name': str(row['NE_name']),
                'v_nom': int(dlr_voltage),
                'matched': False,
                'path': [],
                'network_ids': [],
                'match_quality': 'No match - endpoints not found'
            })
            continue

        # Get node IDs for start and end points
        start_idx, start_pos = nearest_points['start_nearest']
        end_idx, end_pos = nearest_points['end_nearest']

        start_node = f"node_{start_idx}_{start_pos}"
        end_node = f"node_{end_idx}_{end_pos}"

        # Skip if start and end are the same
        if start_node == end_node:
            results.append({
                'dlr_id': str(row['id']),
                'dlr_name': str(row['NE_name']),
                'v_nom': int(dlr_voltage),
                'matched': False,
                'path': [],
                'network_ids': [],
                'match_quality': 'No match - start and end are the same'
            })
            continue

        try:
            # Find shortest path between the endpoints
            path = nx.shortest_path(G, start_node, end_node, weight='weight')

            # Get the network line IDs for this path
            network_ids = []

            for i in range(len(path) - 1):
                edge_data = G.get_edge_data(path[i], path[i + 1])
                if 'id' in edge_data and not edge_data.get('connector', False):  # Only include actual network lines
                    network_ids.append(str(edge_data['id']))

            # If no network lines in the path (only connectors), treat as unmatched
            if not network_ids:
                results.append({
                    'dlr_id': str(row['id']),
                    'dlr_name': str(row['NE_name']),
                    'v_nom': int(dlr_voltage),
                    'matched': False,
                    'path': [],
                    'network_ids': [],
                    'match_quality': 'No match - no network lines in path'
                })
                continue

            # Calculate path length to compare with DLR line length
            path_length = 0
            path_lines = []

            for network_id in network_ids:
                network_line = network_gdf[network_gdf['id'].astype(str) == network_id]
                if not network_line.empty:
                    path_length += float(network_line.iloc[0].geometry.length)
                    path_lines.append(network_line.iloc[0].geometry)

            # Compare lengths to determine match quality
            dlr_length = float(row.geometry.length)
            length_ratio = path_length / dlr_length if dlr_length > 0 else float('inf')

            # Determine match quality based on length ratio
            if 0.9 <= length_ratio <= 1.1:
                match_quality = 'Excellent'
            elif 0.8 <= length_ratio <= 1.2:
                match_quality = 'Good'
            elif 0.7 <= length_ratio <= 1.3:
                match_quality = 'Fair'
            else:
                match_quality = 'Poor'

            results.append({
                'dlr_id': str(row['id']),
                'dlr_name': str(row['NE_name']),
                'v_nom': int(dlr_voltage),
                'matched': True,
                'path': [str(p) for p in path],  # Convert path nodes to strings
                'network_ids': network_ids,
                'path_length': float(path_length),
                'dlr_length': float(dlr_length),
                'length_ratio': float(length_ratio),
                'match_quality': match_quality
            })

        except nx.NetworkXNoPath:
            results.append({
                'dlr_id': str(row['id']),
                'dlr_name': str(row['NE_name']),
                'v_nom': int(dlr_voltage),
                'matched': False,
                'path': [],
                'network_ids': [],
                'match_quality': 'No path found'
            })

    return results


# Create complete HTML file for visualization - WITHOUT SHOWING CONNECTOR LINES
# Create complete HTML file for visualization with simplified coloring
def visualize_results(dlr_gdf, network_gdf, matching_results):
    # Create GeoJSON data for lines
    dlr_features = []
    network_features = []
    matched_features = []

    # Create sets to track which lines are matched
    matched_dlr_ids = set()
    matched_network_ids = set()

    # First, identify all matched DLR and network lines
    for result in matching_results:
        if result['matched'] and result['network_ids']:
            matched_dlr_ids.add(str(result['dlr_id']))
            for network_id in result['network_ids']:
                matched_network_ids.add(str(network_id))

    # Add DLR lines to GeoJSON with matched/unmatched status
    for idx, row in dlr_gdf.iterrows():
        # Create a unique ID for each line
        line_id = f"dlr_{row['id']}"
        coords = list(row.geometry.coords)

        # Check if this DLR line is matched
        is_matched = str(row['id']) in matched_dlr_ids

        # Create GeoJSON feature
        feature = {
            "type": "Feature",
            "id": line_id,
            "properties": {
                "type": "dlr",
                "id": str(row['id']),
                "name": str(row['NE_name']),
                "voltage": int(row['v_nom']),
                "is_matched": is_matched,
                "tooltip": f"DLR: {row['id']} - {row['NE_name']} ({row['v_nom']} kV) - {'Matched' if is_matched else 'Unmatched'}"
            },
            "geometry": {
                "type": "LineString",
                "coordinates": [[float(x), float(y)] for x, y in coords]
            }
        }
        dlr_features.append(feature)

    # Add network lines to GeoJSON with matched/unmatched status
    for idx, row in network_gdf.iterrows():
        line_id = f"network_{row['id']}"
        coords = list(row.geometry.coords)

        # Check if this network line is matched
        is_matched = str(row['id']) in matched_network_ids

        feature = {
            "type": "Feature",
            "id": line_id,
            "properties": {
                "type": "network",
                "id": str(row['id']),
                "voltage": int(row['v_nom']),
                "is_matched": is_matched,
                "tooltip": f"Network: {row['id']} ({row['v_nom']} kV) - {'Matched' if is_matched else 'Unmatched'}"
            },
            "geometry": {
                "type": "LineString",
                "coordinates": [[float(x), float(y)] for x, y in coords]
            }
        }
        network_features.append(feature)

    # Create GeoJSON collections
    dlr_collection = {"type": "FeatureCollection", "features": dlr_features}
    network_collection = {"type": "FeatureCollection", "features": network_features}

    # Convert to JSON strings
    dlr_json = json.dumps(dlr_collection)
    network_json = json.dumps(network_collection)

    # Create a complete HTML file from scratch
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>DLR-Network Line Matching Results</title>

        <link rel="stylesheet" href="https://unpkg.com/leaflet@1.7.1/dist/leaflet.css" />
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css">

        <script src="https://unpkg.com/leaflet@1.7.1/dist/leaflet.js"></script>

        <style>
            html, body, #map {{
                height: 100%;
                width: 100%;
                margin: 0;
                padding: 0;
            }}

            .control-panel {{
                position: absolute;
                top: 10px;
                right: 10px;
                z-index: 1000;
                background: white;
                padding: 15px;
                border-radius: 5px;
                box-shadow: 0 1px 5px rgba(0,0,0,0.4);
                max-width: 300px;
                max-height: calc(100vh - 50px);
                overflow-y: auto;
            }}

            .control-section {{
                margin-bottom: 15px;
            }}

            .control-section h3 {{
                margin: 0 0 10px 0;
                font-size: 16px;
            }}

            .search-input {{
                width: 100%;
                padding: 8px;
                margin-bottom: 10px;
                border: 1px solid #ccc;
                border-radius: 4px;
            }}

            .search-results {{
                max-height: 200px;
                overflow-y: auto;
                margin-bottom: 10px;
                display: none;
            }}

            .search-result {{
                padding: 8px;
                cursor: pointer;
                border-bottom: 1px solid #eee;
            }}

            .search-result:hover {{
                background-color: #f0f0f0;
            }}

            .filter-section {{
                margin-bottom: 10px;
            }}

            .filter-options {{
                display: flex;
                flex-wrap: wrap;
                gap: 5px;
                margin-bottom: 8px;
            }}

            .filter-option {{
                background: #f0f0f0;
                border: 1px solid #ccc;
                padding: 5px 10px;
                border-radius: 3px;
                cursor: pointer;
                font-size: 12px;
            }}

            .filter-option.active {{
                background: #4CAF50;
                color: white;
            }}

            .legend {{
                padding: 10px;
                background: white;
                border-radius: 5px;
                box-shadow: 0 1px 5px rgba(0,0,0,0.4);
                line-height: 1.5;
            }}

            .legend-item {{
                display: flex;
                align-items: center;
                margin-bottom: 5px;
            }}

            .legend-color {{
                width: 20px;
                height: 3px;
                margin-right: 8px;
            }}

            .highlighted {{
                stroke-width: 6px !important;
                stroke-opacity: 1 !important;
                animation: pulse 1.5s infinite;
            }}

            @keyframes pulse {{
                0% {{ stroke-opacity: 1; }}
                50% {{ stroke-opacity: 0.5; }}
                100% {{ stroke-opacity: 1; }}
            }}
        </style>
    </head>
    <body>
        <div id="map"></div>

        <div class="control-panel" id="controlPanel">
            <div class="control-section">
                <h3><i class="fas fa-search"></i> Search</h3>
                <input type="text" id="searchInput" class="search-input" placeholder="Search for lines...">
                <div id="searchResults" class="search-results"></div>
            </div>

            <div class="control-section">
                <h3><i class="fas fa-filter"></i> Filters</h3>

                <div class="filter-section">
                    <h4>Line Types</h4>
                    <div class="filter-options">
                        <div class="filter-option active" data-filter="dlr-matched">Matched DLR Lines</div>
                        <div class="filter-option active" data-filter="dlr-unmatched">Unmatched DLR Lines</div>
                        <div class="filter-option active" data-filter="network-matched">Matched Network Lines</div>
                        <div class="filter-option active" data-filter="network-unmatched">Unmatched Network Lines</div>
                    </div>
                </div>

                <div class="filter-section">
                    <h4>Voltage</h4>
                    <div class="filter-options">
                        <div class="filter-option active" data-filter-voltage="all">All</div>
                        <div class="filter-option" data-filter-voltage="220">220 kV</div>
                        <div class="filter-option" data-filter-voltage="400">400 kV</div>
                    </div>
                </div>
            </div>

            <div class="control-section">
                <h3><i class="fas fa-info-circle"></i> Legend</h3>
                <div class="legend">
                    <div class="legend-item">
                        <div class="legend-color" style="background-color: green;"></div>
                        <div>Matched DLR Lines</div>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background-color: red;"></div>
                        <div>Unmatched DLR Lines</div>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background-color: purple;"></div>
                        <div>Matched Network Lines</div>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background-color: blue;"></div>
                        <div>Unmatched Network Lines</div>
                    </div>
                </div>
            </div>
        </div>

        <script>
            // Initialize the map
            var map = L.map('map').setView([51.1657, 10.4515], 6);

            // Add base tile layer
            L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
                attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
            }}).addTo(map);

            // Load the GeoJSON data - DIRECTLY EMBEDDED IN THE HTML
            var dlrLines = {dlr_json};
            var networkLines = {network_json};

            // Define styling for the lines with simplified coloring
            function dlrStyle(feature) {{
                return {{
                    "color": feature.properties.is_matched ? "green" : "red",
                    "weight": 3,
                    "opacity": 0.8
                }};
            }};

            function networkStyle(feature) {{
                return {{
                    "color": feature.properties.is_matched ? "purple" : "blue",
                    "weight": 2,
                    "opacity": 0.6
                }};
            }};

            // Create the layers
            var dlrMatchedLayer = L.geoJSON(dlrLines, {{
                filter: function(feature) {{
                    return feature.properties.is_matched === true;
                }},
                style: dlrStyle,
                onEachFeature: function(feature, layer) {{
                    layer.bindTooltip(feature.properties.tooltip);
                    layer.on('click', function() {{
                        highlightFeature(feature.id);
                    }});
                }}
            }}).addTo(map);

            var dlrUnmatchedLayer = L.geoJSON(dlrLines, {{
                filter: function(feature) {{
                    return feature.properties.is_matched === false;
                }},
                style: dlrStyle,
                onEachFeature: function(feature, layer) {{
                    layer.bindTooltip(feature.properties.tooltip);
                    layer.on('click', function() {{
                        highlightFeature(feature.id);
                    }});
                }}
            }}).addTo(map);

            var networkMatchedLayer = L.geoJSON(networkLines, {{
                filter: function(feature) {{
                    return feature.properties.is_matched === true;
                }},
                style: networkStyle,
                onEachFeature: function(feature, layer) {{
                    layer.bindTooltip(feature.properties.tooltip);
                    layer.on('click', function() {{
                        highlightFeature(feature.id);
                    }});
                }}
            }}).addTo(map);

            var networkUnmatchedLayer = L.geoJSON(networkLines, {{
                filter: function(feature) {{
                    return feature.properties.is_matched === false;
                }},
                style: networkStyle,
                onEachFeature: function(feature, layer) {{
                    layer.bindTooltip(feature.properties.tooltip);
                    layer.on('click', function() {{
                        highlightFeature(feature.id);
                    }});
                }}
            }}).addTo(map);

            // Function to highlight a feature
            function highlightFeature(id) {{
                // Clear any existing highlights
                clearHighlights();

                // Apply highlight to the specific feature
                dlrMatchedLayer.eachLayer(function(layer) {{
                    if (layer.feature.id === id || (layer.feature.properties.id === id)) {{
                        layer.setStyle({{className: 'highlighted'}});
                        if (layer._path) layer._path.classList.add('highlighted');
                    }}
                }});

                dlrUnmatchedLayer.eachLayer(function(layer) {{
                    if (layer.feature.id === id || (layer.feature.properties.id === id)) {{
                        layer.setStyle({{className: 'highlighted'}});
                        if (layer._path) layer._path.classList.add('highlighted');
                    }}
                }});

                networkMatchedLayer.eachLayer(function(layer) {{
                    if (layer.feature.id === id || (layer.feature.properties.id === id)) {{
                        layer.setStyle({{className: 'highlighted'}});
                        if (layer._path) layer._path.classList.add('highlighted');
                    }}
                }});

                networkUnmatchedLayer.eachLayer(function(layer) {{
                    if (layer.feature.id === id || (layer.feature.properties.id === id)) {{
                        layer.setStyle({{className: 'highlighted'}});
                        if (layer._path) layer._path.classList.add('highlighted');
                    }}
                }});
            }}

            // Function to clear highlights
            function clearHighlights() {{
                dlrMatchedLayer.eachLayer(function(layer) {{
                    layer.setStyle(dlrStyle(layer.feature));
                    if (layer._path) layer._path.classList.remove('highlighted');
                }});

                dlrUnmatchedLayer.eachLayer(function(layer) {{
                    layer.setStyle(dlrStyle(layer.feature));
                    if (layer._path) layer._path.classList.remove('highlighted');
                }});

                networkMatchedLayer.eachLayer(function(layer) {{
                    layer.setStyle(networkStyle(layer.feature));
                    if (layer._path) layer._path.classList.remove('highlighted');
                }});

                networkUnmatchedLayer.eachLayer(function(layer) {{
                    layer.setStyle(networkStyle(layer.feature));
                    if (layer._path) layer._path.classList.remove('highlighted');
                }});
            }}

            // Function to create the search index
            function createSearchIndex() {{
                var searchData = [];

                // Add DLR lines to search data
                dlrLines.features.forEach(function(feature) {{
                    searchData.push({{
                        id: feature.id,
                        text: "DLR: " + feature.properties.id + " - " + feature.properties.name + " (" + feature.properties.voltage + " kV)",
                        type: "dlr",
                        feature: feature
                    }});
                }});

                // Add network lines to search data
                networkLines.features.forEach(function(feature) {{
                    searchData.push({{
                        id: feature.id,
                        text: "Network: " + feature.properties.id + " (" + feature.properties.voltage + " kV)",
                        type: "network",
                        feature: feature
                    }});
                }});

                return searchData;
            }}

            // Function to filter the features
            function filterFeatures() {{
                // Get active filters
                var activeLineTypes = [];
                document.querySelectorAll('.filter-option[data-filter].active').forEach(function(element) {{
                    activeLineTypes.push(element.getAttribute('data-filter'));
                }});

                var activeVoltages = [];
                var allVoltages = document.querySelector('.filter-option[data-filter-voltage="all"]').classList.contains('active');
                if (!allVoltages) {{
                    document.querySelectorAll('.filter-option[data-filter-voltage].active').forEach(function(element) {{
                        var voltage = element.getAttribute('data-filter-voltage');
                        if (voltage !== "all") activeVoltages.push(parseInt(voltage));
                    }});
                }}

                // Show/hide DLR matched layer
                if (activeLineTypes.includes("dlr-matched")) {{
                    dlrMatchedLayer.addTo(map);

                    // Apply voltage filter
                    if (!allVoltages) {{
                        dlrMatchedLayer.eachLayer(function(layer) {{
                            var visible = activeVoltages.includes(layer.feature.properties.voltage);
                            if (visible) {{
                                if (layer._path) layer._path.style.display = "block";
                            }} else {{
                                if (layer._path) layer._path.style.display = "none";
                            }}
                        }});
                    }}
                }} else {{
                    map.removeLayer(dlrMatchedLayer);
                }}

                // Show/hide DLR unmatched layer
                if (activeLineTypes.includes("dlr-unmatched")) {{
                    dlrUnmatchedLayer.addTo(map);

                    // Apply voltage filter
                    if (!allVoltages) {{
                        dlrUnmatchedLayer.eachLayer(function(layer) {{
                            var visible = activeVoltages.includes(layer.feature.properties.voltage);
                            if (visible) {{
                                if (layer._path) layer._path.style.display = "block";
                            }} else {{
                                if (layer._path) layer._path.style.display = "none";
                            }}
                        }});
                    }}
                }} else {{
                    map.removeLayer(dlrUnmatchedLayer);
                }}

                // Show/hide Network matched layer
                if (activeLineTypes.includes("network-matched")) {{
                    networkMatchedLayer.addTo(map);

                    // Apply voltage filter
                    if (!allVoltages) {{
                        networkMatchedLayer.eachLayer(function(layer) {{
                            var visible = activeVoltages.includes(layer.feature.properties.voltage);
                            if (visible) {{
                                if (layer._path) layer._path.style.display = "block";
                            }} else {{
                                if (layer._path) layer._path.style.display = "none";
                            }}
                        }});
                    }}
                }} else {{
                    map.removeLayer(networkMatchedLayer);
                }}

                // Show/hide Network unmatched layer
                if (activeLineTypes.includes("network-unmatched")) {{
                    networkUnmatchedLayer.addTo(map);

                    // Apply voltage filter
                    if (!allVoltages) {{
                        networkUnmatchedLayer.eachLayer(function(layer) {{
                            var visible = activeVoltages.includes(layer.feature.properties.voltage);
                            if (visible) {{
                                if (layer._path) layer._path.style.display = "block";
                            }} else {{
                                if (layer._path) layer._path.style.display = "none";
                            }}
                        }});
                    }}
                }} else {{
                    map.removeLayer(networkUnmatchedLayer);
                }}
            }}

            // Initialize the search functionality
            function initializeSearch() {{
                var searchInput = document.getElementById('searchInput');
                var searchResults = document.getElementById('searchResults');
                var searchData = createSearchIndex();

                searchInput.addEventListener('input', function() {{
                    var query = this.value.toLowerCase();

                    if (query.length < 2) {{
                        searchResults.style.display = 'none';
                        return;
                    }}

                    var results = searchData.filter(function(item) {{
                        return item.text.toLowerCase().includes(query);
                    }});

                    searchResults.innerHTML = '';

                    results.forEach(function(result) {{
                        var div = document.createElement('div');
                        div.className = 'search-result';
                        div.textContent = result.text;
                        div.onclick = function() {{
                            // Get the center of the feature
                            var coords = getCenterOfFeature(result.feature);

                            // Zoom to the feature
                            map.setView([coords[1], coords[0]], 10);

                            // Highlight the feature
                            highlightFeature(result.id);

                            // Hide search results
                            searchResults.style.display = 'none';

                            // Set input value
                            searchInput.value = result.text;
                        }};
                        searchResults.appendChild(div);
                    }});

                    searchResults.style.display = results.length > 0 ? 'block' : 'none';
                }});

                document.addEventListener('click', function(e) {{
                    if (!searchInput.contains(e.target) && !searchResults.contains(e.target)) {{
                        searchResults.style.display = 'none';
                    }}
                }});
            }}

            // Function to get center of a feature
            function getCenterOfFeature(feature) {{
                var coords = feature.geometry.coordinates;
                var sum = [0, 0];

                for (var i = 0; i < coords.length; i++) {{
                    sum[0] += coords[i][0];
                    sum[1] += coords[i][1];
                }}

                return [sum[0] / coords.length, sum[1] / coords.length];
            }}

            // Initialize the filter functionality
            function initializeFilters() {{
                // Line type filters
                document.querySelectorAll('.filter-option[data-filter]').forEach(function(element) {{
                    element.addEventListener('click', function() {{
                        this.classList.toggle('active');
                        filterFeatures();
                    }});
                }});

                // Voltage filters
                document.querySelectorAll('.filter-option[data-filter-voltage]').forEach(function(element) {{
                    element.addEventListener('click', function() {{
                        var isAll = this.getAttribute('data-filter-voltage') === 'all';

                        if (isAll) {{
                            // If "All" is clicked, toggle it and set others accordingly
                            this.classList.toggle('active');

                            if (this.classList.contains('active')) {{
                                // If "All" is now active, make all others inactive
                                document.querySelectorAll('.filter-option[data-filter-voltage]:not([data-filter-voltage="all"])').forEach(function(el) {{
                                    el.classList.remove('active');
                                }});
                            }}
                        }} else {{
                            // If a specific voltage is clicked, make "All" inactive
                            document.querySelector('.filter-option[data-filter-voltage="all"]').classList.remove('active');
                            this.classList.toggle('active');

                            // If no specific voltage is active, activate "All"
                            var anyActive = false;
                            document.querySelectorAll('.filter-option[data-filter-voltage]:not([data-filter-voltage="all"])').forEach(function(el) {{
                                if (el.classList.contains('active')) anyActive = true;
                            }});

                            if (!anyActive) {{
                                document.querySelector('.filter-option[data-filter-voltage="all"]').classList.add('active');
                            }}
                        }}

                        filterFeatures();
                    }});
                }});
            }}

            // Initialize everything once the document is loaded
            document.addEventListener('DOMContentLoaded', function() {{
                initializeSearch();
                initializeFilters();
            }});
        </script>
    </body>
    </html>
    """

    # Save the HTML file
    output_file = output_dir / 'dlr_network_matching_results.html'
    with open(output_file, 'w') as f:
        f.write(html)

    return output_file


# Create a summary table as HTML
# Create a summary table as HTML with simplified colors
def create_summary_table(matching_results):
    # Convert results to DataFrame
    results_df = pd.DataFrame(matching_results)

    # Create summary statistics
    total_dlr_lines = len(matching_results)
    matched_lines = sum(result['matched'] for result in matching_results)
    unmatched_lines = total_dlr_lines - matched_lines

    match_quality_counts = {
        'Excellent': sum(1 for result in matching_results if result.get('match_quality') == 'Excellent'),
        'Good': sum(1 for result in matching_results if result.get('match_quality') == 'Good'),
        'Fair': sum(1 for result in matching_results if result.get('match_quality') == 'Fair'),
        'Poor': sum(1 for result in matching_results if result.get('match_quality') == 'Poor'),
        'No match - endpoints not found': sum(
            1 for result in matching_results if result.get('match_quality') == 'No match - endpoints not found'),
        'No match - no network lines in path': sum(
            1 for result in matching_results if result.get('match_quality') == 'No match - no network lines in path'),
        'No match - start and end are the same': sum(
            1 for result in matching_results if result.get('match_quality') == 'No match - start and end are the same'),
        'No path found': sum(1 for result in matching_results if result.get('match_quality') == 'No path found')
    }

    # Create HTML for summary
    html_summary = f"""
    <html>
    <head>
        <title>DLR-Network Line Matching Summary</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 20px;
            }}
            h1, h2 {{
                color: #333;
            }}
            .summary {{
                margin-bottom: 20px;
                padding: 15px;
                background-color: #f5f5f5;
                border-radius: 5px;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                margin-top: 20px;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }}
            th {{
                background-color: #4CAF50;
                color: white;
            }}
            tr:nth-child(even) {{
                background-color: #f2f2f2;
            }}
            .filter-controls {{
                margin: 20px 0;
                padding: 10px;
                background-color: #eee;
                border-radius: 5px;
            }}
            .matched {{
                background-color: #90EE90;  /* Light green */
            }}
            .unmatched {{
                background-color: #ffcccb;  /* Light red */
            }}
        </style>
        <script>
            function filterTable() {{
                const filter = document.getElementById('filter').value.toLowerCase();
                const rows = document.getElementById('resultsTable').getElementsByTagName('tr');

                for (let i = 1; i < rows.length; i++) {{
                    const row = rows[i];
                    const text = row.textContent.toLowerCase();
                    if (text.includes(filter)) {{
                        row.style.display = '';
                    }} else {{
                        row.style.display = 'none';
                    }}
                }}
            }}

            function filterByMatchStatus(status) {{
                const rows = document.getElementById('resultsTable').getElementsByTagName('tr');

                for (let i = 1; i < rows.length; i++) {{
                    const row = rows[i];
                    const matchedCell = row.cells[3].textContent;

                    if (status === 'all' || 
                        (status === 'matched' && matchedCell === 'Yes') || 
                        (status === 'unmatched' && matchedCell === 'No')) {{
                        row.style.display = '';
                    }} else {{
                        row.style.display = 'none';
                    }}
                }}
            }}

            function filterByVoltage(voltage) {{
                const rows = document.getElementById('resultsTable').getElementsByTagName('tr');

                for (let i = 1; i < rows.length; i++) {{
                    const row = rows[i];
                    const voltageCell = row.cells[2].textContent;

                    if (voltage === 'all' || voltageCell === voltage) {{
                        row.style.display = '';
                    }} else {{
                        row.style.display = 'none';
                    }}
                }}
            }}
        </script>
    </head>
    <body>
        <h1>DLR-Network Line Matching Results</h1>

        <div class="summary">
            <h2>Summary</h2>
            <p>Total DLR Lines: {total_dlr_lines}</p>
            <p>Matched Lines: {matched_lines} ({matched_lines / total_dlr_lines * 100:.1f}%)</p>
            <p>Unmatched Lines: {unmatched_lines} ({unmatched_lines / total_dlr_lines * 100:.1f}%)</p>
            <p>Match Quality Details:</p>
            <ul>
                <li>Excellent Matches: {match_quality_counts['Excellent']} ({match_quality_counts['Excellent'] / total_dlr_lines * 100:.1f}%)</li>
                <li>Good Matches: {match_quality_counts['Good']} ({match_quality_counts['Good'] / total_dlr_lines * 100:.1f}%)</li>
                <li>Fair Matches: {match_quality_counts['Fair']} ({match_quality_counts['Fair'] / total_dlr_lines * 100:.1f}%)</li>
                <li>Poor Matches: {match_quality_counts['Poor']} ({match_quality_counts['Poor'] / total_dlr_lines * 100:.1f}%)</li>
                <li>No Match (Endpoints not found): {match_quality_counts['No match - endpoints not found']} ({match_quality_counts['No match - endpoints not found'] / total_dlr_lines * 100:.1f}%)</li>
                <li>No Match (Start and end are the same): {match_quality_counts.get('No match - start and end are the same', 0)} ({match_quality_counts.get('No match - start and end are the same', 0) / total_dlr_lines * 100:.1f}%)</li>
                <li>No Match (No network lines in path): {match_quality_counts['No match - no network lines in path']} ({match_quality_counts['No match - no network lines in path'] / total_dlr_lines * 100:.1f}%)</li>
                <li>No Path Found: {match_quality_counts['No path found']} ({match_quality_counts['No path found'] / total_dlr_lines * 100:.1f}%)</li>
            </ul>
        </div>

        <div class="filter-controls">
            <h2>Filter Results</h2>
            <input type="text" id="filter" onkeyup="filterTable()" placeholder="Search for DLR lines...">
            <h3>By Match Status:</h3>
            <button onclick="filterByMatchStatus('all')">All</button>
            <button onclick="filterByMatchStatus('matched')">Matched</button>
            <button onclick="filterByMatchStatus('unmatched')">Unmatched</button>
            <h3>By Voltage Level:</h3>
            <button onclick="filterByVoltage('all')">All</button>
            <button onclick="filterByVoltage('220')">220 kV</button>
            <button onclick="filterByVoltage('400')">400 kV</button>
        </div>

        <h2>Detailed Results</h2>
        <table id="resultsTable">
            <tr>
                <th>DLR ID</th>
                <th>DLR Name</th>
                <th>Voltage (kV)</th>
                <th>Matched</th>
                <th>Network IDs</th>
                <th>Length Ratio</th>
                <th>Match Quality</th>
            </tr>
    """

    # Add rows for each result
    for result in matching_results:
        network_ids = ", ".join(result['network_ids']) if result['matched'] and 'network_ids' in result and result[
            'network_ids'] else "-"
        length_ratio = f"{result.get('length_ratio', '-'):.2f}" if result.get('length_ratio') is not None else "-"

        # Determine CSS class based on match status (not quality)
        css_class = "matched" if result['matched'] else "unmatched"

        html_summary += f"""
            <tr class="{css_class}">
                <td>{result['dlr_id']}</td>
                <td>{result['dlr_name']}</td>
                <td>{result['v_nom']}</td>
                <td>{"Yes" if result['matched'] else "No"}</td>
                <td>{network_ids}</td>
                <td>{length_ratio}</td>
                <td>{result.get('match_quality', '-')}</td>
            </tr>
        """

    html_summary += """
        </table>
    </body>
    </html>
    """

    # Save the HTML file
    output_file = output_dir / 'dlr_network_matching_summary.html'
    with open(output_file, 'w') as f:
        f.write(html_summary)

    return output_file


def main():
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    print("Loading DLR lines...")
    dlr_gdf = load_dlr_lines()
    print(f"Loaded {len(dlr_gdf)} DLR lines")

    print("Loading network lines (excluding 110kV)...")
    network_gdf = load_network_lines()
    print(f"Loaded {len(network_gdf)} network lines")

    # Find nearest points in the network for DLR endpoints
    print("Finding nearest points for DLR endpoints...")
    nearest_points_dict = find_nearest_points(dlr_gdf, network_gdf)

    # Build network graph with NO automatic connections
    print("Building network graph (without extra connections)...")
    G = build_network_graph(network_gdf)
    print(f"Graph has {len(G.nodes)} nodes and {len(G.edges)} edges")

    # Find matching network lines
    print("Finding matching network lines...")
    matching_results = find_matching_network_lines(dlr_gdf, network_gdf, nearest_points_dict, G)

    # Visualize results
    print("Visualizing results...")
    map_file = visualize_results(dlr_gdf, network_gdf, matching_results)
    summary_file = create_summary_table(matching_results)

    print(f"Results saved to:")
    print(f"  - Map: {map_file}")
    print(f"  - Summary: {summary_file}")

    # Print summary statistics
    total_dlr_lines = len(matching_results)
    matched_lines = sum(result['matched'] for result in matching_results)
    print(f"Total DLR lines: {total_dlr_lines}")
    print(f"Matched lines: {matched_lines} ({matched_lines / total_dlr_lines * 100:.1f}%)")
    print(
        f"Unmatched lines: {total_dlr_lines - matched_lines} ({(total_dlr_lines - matched_lines) / total_dlr_lines * 100:.1f}%)")

    # Print statistics by voltage level
    v220_lines = sum(1 for result in matching_results if result.get('v_nom') == 220)
    v220_matched = sum(1 for result in matching_results if result.get('v_nom') == 220 and result['matched'])
    v400_lines = sum(1 for result in matching_results if result.get('v_nom') == 400)
    v400_matched = sum(1 for result in matching_results if result.get('v_nom') == 400 and result['matched'])

    print("\nStatistics by voltage level:")
    if v220_lines > 0:
        print(f"220 kV lines: {v220_matched}/{v220_lines} matched ({v220_matched / v220_lines * 100:.1f}%)")
    else:
        print("220 kV lines: 0/0 matched (0.0%)")

    if v400_lines > 0:
        print(f"400 kV lines: {v400_matched}/{v400_lines} matched ({v400_matched / v400_lines * 100:.1f}%)")
    else:
        print("400 kV lines: 0/0 matched (0.0%)")


if __name__ == "__main__":
    main()