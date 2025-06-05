import folium
import geopandas as gpd
import pandas as pd
import json
import os
from shapely.geometry import LineString
import random
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_simple_network_map(output_file='simple_network_map.html'):
    """Create a simple interactive network map with functioning search"""

    # Create sample data (50 random lines)
    network_lines = create_sample_data(50)

    # Calculate map center
    centroid_x = network_lines.geometry.centroid.x.mean()
    centroid_y = network_lines.geometry.centroid.y.mean()

    # Create base folium map (we won't use this for rendering)
    m = folium.Map(location=[centroid_y, centroid_x], zoom_start=6, tiles='CartoDB positron')

    # Prepare line coordinates for JavaScript
    line_coordinates = {}
    line_popups = {}
    all_line_ids = []

    # Process each line
    for idx, row in network_lines.iterrows():
        line_id = row['id']

        # Convert geometry to coordinates
        if row.geometry.geom_type == 'MultiLineString':
            all_coords = []
            for segment in row.geometry.geoms:
                latlngs = [[y, x] for x, y in segment.coords]
                all_coords.extend(latlngs)
        else:
            all_coords = [[y, x] for x, y in row.geometry.coords]

        # Calculate midpoint
        if all_coords:
            lat_sum = sum(coord[0] for coord in all_coords)
            lng_sum = sum(coord[1] for coord in all_coords)
            midpoint = [lat_sum / len(all_coords), lng_sum / len(all_coords)]
        else:
            midpoint = None

        # Create popup content
        popup_html = f"""
        <div style="width: 200px;">
            <h3>{line_id}</h3>
            <p><b>Voltage:</b> {row['voltage']} kV</p>
            <p><b>Length:</b> {row['length']:.2f} km</p>
        </div>
        """

        # Store line information
        line_coordinates[line_id] = {
            'id': line_id,
            'type': 'network',
            'coords': all_coords,
            'midpoint': midpoint,
            'voltage': row['voltage'],
            'length': row['length']
        }

        # Store popup content
        line_popups[line_id] = popup_html

        # Add to list of all line IDs
        all_line_ids.append({
            'id': line_id,
            'type': 'network',
            'voltage': row['voltage']
        })

    # Convert data to JSON for JavaScript
    line_coords_json = json.dumps(line_coordinates)
    line_popups_json = json.dumps(line_popups)
    all_line_ids_json = json.dumps(all_line_ids)

    # Create custom HTML template
    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Simple Network Map</title>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">

        <!-- Leaflet CSS -->
        <link rel="stylesheet" href="https://unpkg.com/leaflet@1.7.1/dist/leaflet.css" />

        <style>
            body {{
                margin: 0;
                padding: 0;
                font-family: Arial, sans-serif;
            }}

            #map {{
                position: absolute;
                top: 0;
                bottom: 0;
                width: 100%;
                z-index: 1;
            }}

            /* Search Container */
            #search-container {{
                position: fixed;
                top: 10px;
                right: 10px;
                width: 250px;
                background-color: white;
                padding: 10px;
                border-radius: 5px;
                box-shadow: 0 0 10px rgba(0,0,0,0.2);
                z-index: 1000;
            }}

            #search-container h4 {{
                margin-top: 0;
                margin-bottom: 10px;
            }}

            #line-search {{
                width: 100%;
                padding: 5px;
                box-sizing: border-box;
                margin-bottom: 10px;
                border: 1px solid #ccc;
                border-radius: 3px;
            }}

            #search-button {{
                width: 100%;
                margin-top: 10px;
                padding: 5px;
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 3px;
                cursor: pointer;
            }}

            #search-results {{
                margin-top: 10px;
                max-height: 150px;
                overflow-y: auto;
            }}

            /* Stats Container */
            #stats-container {{
                position: fixed;
                top: 10px;
                left: 10px;
                width: 200px;
                background-color: white;
                padding: 10px;
                border-radius: 5px;
                box-shadow: 0 0 10px rgba(0,0,0,0.2);
                z-index: 999;
            }}

            #stats-container h4 {{
                margin-top: 0;
            }}

            #stats-container p {{
                margin: 5px 0;
            }}

            /* Message container */
            #message-container {{
                display: none;
                position: fixed;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                background-color: rgba(0, 0, 0, 0.7);
                color: white;
                padding: 15px 20px;
                border-radius: 5px;
                z-index: 2000;
                max-width: 80%;
                text-align: center;
            }}
        </style>
    </head>
    <body>
        <!-- Main map container -->
        <div id="map"></div>

        <!-- Search container -->
        <div id="search-container">
            <h4>Search Lines</h4>
            <input type="text" id="line-search" placeholder="Enter line ID...">
            <button id="search-button">Search</button>
            <div id="search-results"></div>
        </div>

        <!-- Stats container -->
        <div id="stats-container">
            <h4>Network Statistics</h4>
            <p>Total Lines: {len(network_lines)}</p>
            <p>Voltage Levels:</p>
            <p style="padding-left: 15px;">- 110 kV: {len(network_lines[network_lines['voltage'] == 110])}</p>
            <p style="padding-left: 15px;">- 220 kV: {len(network_lines[network_lines['voltage'] == 220])}</p>
            <p style="padding-left: 15px;">- 380 kV: {len(network_lines[network_lines['voltage'] == 380])}</p>
        </div>

        <!-- Message container -->
        <div id="message-container"></div>

        <!-- Leaflet JS -->
        <script src="https://unpkg.com/leaflet@1.7.1/dist/leaflet.js"></script>

        <script>
        // Self-executing function to avoid global scope pollution
        (function() {{
            // Log function with timestamps
            function log(message) {{
                var now = new Date();
                console.log('[' + now.toLocaleTimeString() + '] ' + message);
            }}

            // Show temporary message
            function showMessage(message, duration) {{
                var container = document.getElementById('message-container');
                container.innerText = message;
                container.style.display = 'block';

                setTimeout(function() {{
                    container.style.display = 'none';
                }}, duration || 3000);
            }}

            // Data from Python
            var lineCoordinates = {line_coords_json};
            var allLineIds = {all_line_ids_json};
            var linePopups = {line_popups_json};
            var map = null;
            var currentMarker = null;
            var currentHighlightedLines = [];
            var allLines = [];

            // Initialize map
            function initMap() {{
                log('Initializing map');

                try {{
                    // Create the map
                    map = L.map('map').setView([{centroid_y}, {centroid_x}], 6);

                    // Add tile layer
                    L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
                        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
                    }}).addTo(map);

                    // Add all lines to the map
                    addAllLinesToMap();

                    // Add click handler to map to clear selection
                    map.on('click', function() {{
                        clearHighlights();
                    }});

                    log('Map initialized successfully');
                    return true;
                }} catch (e) {{
                    log('Error initializing map: ' + e.message);
                    showMessage('Error initializing map. Please refresh the page.', 5000);
                    return false;
                }}
            }}

            // Add all lines to the map
            function addAllLinesToMap() {{
                log('Adding lines to map');

                var count = 0;
                for (var lineId in lineCoordinates) {{
                    var line = lineCoordinates[lineId];

                    // Skip if no coordinates
                    if (!line.coords || line.coords.length < 2) continue;

                    // Create polyline
                    var polyline = L.polyline(line.coords, {{
                        color: 'blue',
                        weight: 2,
                        opacity: 0.6
                    }});

                    // Add popup with the detailed HTML content
                    if (linePopups[lineId]) {{
                        var popupContent = linePopups[lineId];
                        var popup = L.popup({{
                            maxWidth: 400
                        }}).setContent(popupContent);

                        polyline.bindPopup(popup);
                    }}

                    // Add tooltip
                    polyline.bindTooltip("Line " + lineId);

                    // Store line ID for later reference
                    polyline.lineId = lineId;

                    // Add event handler for highlighting
                    polyline.on('click', function(e) {{
                        // Prevent map click from firing
                        L.DomEvent.stopPropagation(e);

                        // Highlight this line
                        highlightLine(this.lineId);
                    }});

                    // Add to map
                    polyline.addTo(map);

                    // Add to array of all lines
                    allLines.push(polyline);

                    count++;
                }}

                log('Added ' + count + ' lines to map');
            }}

            // Clear all highlights
            function clearHighlights() {{
                // Remove current marker
                if (currentMarker && map) {{
                    map.removeLayer(currentMarker);
                    currentMarker = null;
                }}

                // Reset highlighted lines
                currentHighlightedLines.forEach(function(polyline) {{
                    if (polyline && polyline.setStyle) {{
                        polyline.setStyle({{
                            color: 'blue',
                            weight: 2,
                            opacity: 0.6
                        }});
                    }}
                }});

                currentHighlightedLines = [];
            }}

            // Highlight a specific line
            function highlightLine(lineId) {{
                log('Highlighting line: ' + lineId);

                try {{
                    // Clear previous highlights
                    clearHighlights();

                    // Get line data
                    var line = lineCoordinates[lineId];
                    if (!line || !line.coords || line.coords.length < 2) {{
                        log('Line not found or has no coordinates');
                        showMessage('Line not found or has no coordinates', 3000);
                        return;
                    }}

                    // Find matching polylines
                    var matchingPolylines = [];

                    // Check all lines for this ID
                    allLines.forEach(function(polyline) {{
                        if (polyline.lineId === lineId) {{
                            matchingPolylines.push(polyline);
                        }}
                    }});

                    log('Found ' + matchingPolylines.length + ' matching polylines');

                    if (matchingPolylines.length === 0) {{
                        // If no polylines found, create a temporary one from coordinates
                        var tempPolyline = L.polyline(line.coords, {{
                            color: 'red',
                            weight: 5,
                            opacity: 1
                        }}).addTo(map);

                        // Add popup with the detailed HTML content
                        if (linePopups[lineId]) {{
                            var popupContent = linePopups[lineId];
                            var popup = L.popup({{
                                maxWidth: 400
                            }}).setContent(popupContent);

                            tempPolyline.bindPopup(popup);

                            // Open the popup
                            setTimeout(function() {{
                                tempPolyline.openPopup();
                            }}, 500);
                        }}

                        // Highlight it
                        tempPolyline.lineId = lineId;
                        currentHighlightedLines.push(tempPolyline);

                    }} else {{
                        // Highlight found polylines
                        matchingPolylines.forEach(function(polyline) {{
                            polyline.setStyle({{
                                color: 'red',
                                weight: 5,
                                opacity: 1
                            }});

                            currentHighlightedLines.push(polyline);

                            // Open popup for the first polyline
                            if (matchingPolylines.indexOf(polyline) === 0) {{
                                setTimeout(function() {{
                                    polyline.openPopup();
                                }}, 500);
                            }}
                        }});
                    }}

                    // Add marker at midpoint
                    if (line.midpoint) {{
                        var icon = L.divIcon({{
                            html: '<div style="background-color: blue; border: 3px solid white; width: 20px; height: 20px; border-radius: 50%; box-shadow: 0 0 5px rgba(0,0,0,0.5);"></div>',
                            className: 'search-marker',
                            iconSize: [20, 20],
                            iconAnchor: [10, 10]
                        }});

                        currentMarker = L.marker(line.midpoint, {{ icon: icon }}).addTo(map);
                    }}

                    // Zoom to line bounds
                    var bounds = L.latLngBounds(line.coords);
                    map.fitBounds(bounds, {{ padding: [50, 50] }});

                }} catch (e) {{
                    log('Error highlighting line: ' + e.message);
                    showMessage('Error highlighting line: ' + e.message, 3000);
                }}
            }}

            // Search for lines
            function searchLines() {{
                log('Searching for lines');

                try {{
                    // Get search parameters
                    var searchText = document.getElementById('line-search').value.trim().toLowerCase();
                    var resultsDiv = document.getElementById('search-results');

                    // Validate search text
                    if (!searchText) {{
                        resultsDiv.innerHTML = '<p>Please enter a line ID.</p>';
                        return;
                    }}

                    log('Search text: "' + searchText + '"');

                    // Find matching lines
                    var matchingLines = [];

                    for (var i = 0; i < allLineIds.length; i++) {{
                        var line = allLineIds[i];
                        var id = line.id.toString().toLowerCase();

                        // Check if ID contains search text
                        if (id.includes(searchText)) {{
                            matchingLines.push({{
                                id: line.id,
                                type: line.type,
                                voltage: line.voltage
                            }});
                        }}
                    }}

                    log('Found ' + matchingLines.length + ' matching lines');

                    // Display results
                    if (matchingLines.length === 0) {{
                        resultsDiv.innerHTML = '<p>No matching lines found.</p>';
                    }} else {{
                        // Sort results by ID
                        matchingLines.sort(function(a, b) {{
                            return a.id.toString().localeCompare(b.id.toString());
                        }});

                        // Create results HTML
                        var resultHtml = '<p>Found ' + matchingLines.length + ' matching lines:</p>';
                        resultHtml += '<ul style="padding-left: 20px; margin-top: 5px;">';

                        for (var j = 0; j < matchingLines.length; j++) {{
                            var matchLine = matchingLines[j];

                            resultHtml += '<li>' +
                                '<button onclick="highlightLine(\\\'' + matchLine.id + '\\\')" ' +
                                'style="background: none; border: none; color: blue; text-decoration: underline; cursor: pointer; padding: 0; font: inherit; text-align: left;">' +
                                'Line ' + matchLine.id + ' - ' + matchLine.voltage + ' kV' +
                                '</button>' +
                                '</li>';
                        }}

                        resultHtml += '</ul>';
                        resultsDiv.innerHTML = resultHtml;
                    }}
                }} catch (e) {{
                    log('Error searching: ' + e.message);
                    resultsDiv.innerHTML = '<p>Error searching: ' + e.message + '</p>';
                }}
            }}

            // Set up event listeners
            function setupEventListeners() {{
                log('Setting up event listeners');

                // Search button
                var searchButton = document.getElementById('search-button');
                if (searchButton) {{
                    searchButton.addEventListener('click', searchLines);
                    log('Search button handler attached');
                }} else {{
                    log('Search button not found');
                }}

                // Search input enter key
                var searchInput = document.getElementById('line-search');
                if (searchInput) {{
                    searchInput.addEventListener('keypress', function(e) {{
                        if (e.key === 'Enter') {{
                            searchLines();
                        }}
                    }});
                    log('Search input handler attached');
                }} else {{
                    log('Search input not found');
                }}
            }}

            // Initialize the application
            function init() {{
                log('Initializing application');

                // Initialize map
                if (initMap()) {{
                    // Set up event listeners
                    setupEventListeners();
                    log('Application initialized successfully');
                }} else {{
                    // Show error message
                    showMessage('Failed to initialize map. Please reload the page.', 5000);
                }}
            }}

            // Start initialization when page loads
            window.addEventListener('load', init);

            // Expose functions globally for use in HTML
            window.searchLines = searchLines;
            window.highlightLine = highlightLine;
        }})();
        </script>
    </body>
    </html>
    """

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)

    # Write directly to file
    with open(output_file, 'w') as f:
        f.write(html_template)

    logger.info(f"Simple network map saved to {output_file}")

    return m  # Return the folium map even though we're not using it


def create_sample_data(num_lines=50):
    """Create sample network lines data"""
    lines = []
    for i in range(1, num_lines + 1):
        # Create random lines within Germany
        start_lat = random.uniform(47.0, 55.0)
        start_lng = random.uniform(6.0, 15.0)
        end_lat = start_lat + random.uniform(-1.0, 1.0)
        end_lng = start_lng + random.uniform(-1.0, 1.0)

        # Create LineString geometry
        geometry = LineString([(start_lng, start_lat), (end_lng, end_lat)])

        # Create line properties
        line_id = f"Line_{i}"
        voltage = random.choice([110, 220, 380])
        length = random.uniform(5, 50)

        lines.append({
            'id': line_id,
            'voltage': voltage,
            'length': length,
            'geometry': geometry
        })

    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(lines, crs="EPSG:4326")
    return gdf


# Create the map
output_file = "output/simple_map/simple_network_map.html"
create_simple_network_map(output_file)