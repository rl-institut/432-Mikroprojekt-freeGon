import math

import pandas as pd
import networkx as nx
import folium
from folium.plugins import MarkerCluster
from shapely.geometry import LineString, Point
import geopandas as gpd
import numpy as np
from pathlib import Path
import os

from shapely.geometry.multilinestring import MultiLineString
from shapely.ops import nearest_points
from scipy.spatial import cKDTree
import json
import uuid
import itertools

# Define file paths
data_dir = Path('/home/mohsen/PycharmProjects/freeGon/grid-matching-tool/data/clipped')
jao_lines_path = data_dir / 'jao-lines-germany.csv'
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



# Load JAO lines data
def load_jao_lines():
    jao_df = pd.read_csv(jao_lines_path)
    # Create GeoDataFrame
    geometry = jao_df['geometry'].apply(parse_linestring)
    jao_gdf = gpd.GeoDataFrame(jao_df, geometry=geometry)
    jao_gdf = jao_gdf.explode(index_parts=False, ignore_index=True)


    # Extract start and end points
    jao_gdf['start_point'] = jao_gdf.geometry.apply(lambda x: Point(x.coords[0]))
    jao_gdf['end_point'] = jao_gdf.geometry.apply(lambda x: Point(x.coords[-1]))

    return jao_gdf


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


# Add this import at the top of your file
from rtree import index


def find_nearest_points(jao_gdf, network_gdf, max_alternatives=5, distance_threshold_meters=1000, debug_lines=None):
    """
    Find nearest points in the network for JAO endpoints with improved substation handling.

    Parameters:
    - max_alternatives: Number of alternative endpoints to store
    - distance_threshold_meters: Maximum distance in meters to consider an endpoint match
    - debug_lines: List of specific JAO IDs to debug in detail
    """
    nearest_points_dict = {}

    # Extract endpoints from network lines
    network_endpoints = []
    for idx, row in network_gdf.iterrows():
        geom = row.geometry
        if geom.geom_type == 'LineString':
            network_endpoints.append((idx, 'start', Point(geom.coords[0]), row['id'], row['v_nom']))
            network_endpoints.append((idx, 'end', Point(geom.coords[-1]), row['id'], row['v_nom']))
        elif geom.geom_type == 'MultiLineString':
            for line in geom.geoms:
                network_endpoints.append((idx, 'start', Point(line.coords[0]), row['id'], row['v_nom']))
                network_endpoints.append((idx, 'end', Point(line.coords[-1]), row['id'], row['v_nom']))

    # Create spatial index for network endpoints
    network_endpoint_idx = index.Index()
    for i, (idx, pos, point, _, _) in enumerate(network_endpoints):
        network_endpoint_idx.insert(i, (point.x, point.y, point.x, point.y))

    # Find nearest network endpoints for each JAO endpoint
    for idx, row in jao_gdf.iterrows():
        jao_id = str(row['id'])
        jao_name = str(row['NE_name'])
        jao_voltage = int(row['v_nom'])

        # Check if this is a debug line
        is_debug = debug_lines is not None and jao_id in debug_lines

        if is_debug:
            print(f"\n===== DEBUGGING JAO LINE {jao_id} ({jao_name}) =====")

        geom = row.geometry
        if geom.geom_type == 'LineString':
            start_point = Point(geom.coords[0])
            end_point = Point(geom.coords[-1])
        elif geom.geom_type == 'MultiLineString':
            # For MultiLineString, use the first point of the first line and the last point of the last line
            start_point = Point(geom.geoms[0].coords[0])
            end_point = Point(geom.geoms[-1].coords[-1])
        else:
            nearest_points_dict[idx] = {'start_nearest': None, 'end_nearest': None}
            continue

        # Approximate conversion from meters to degrees for the distance threshold
        avg_lat = (start_point.y + end_point.y) / 2
        import math
        lon_factor = math.cos(math.radians(abs(avg_lat)))
        lon_degree_meters = 111320 * lon_factor
        lat_degree_meters = 111000

        avg_degree_meters = (lon_degree_meters + lat_degree_meters) / 2
        distance_threshold_degrees = distance_threshold_meters / avg_degree_meters

        if is_debug:
            print(f"  Coordinates:")
            print(f"    Start point: ({start_point.x}, {start_point.y})")
            print(f"    End point: ({end_point.x}, {end_point.y})")
            print(
                f"  Distance threshold: {distance_threshold_meters} meters ≈ {distance_threshold_degrees:.6f} degrees")

        # Find nearest network endpoint for JAO start point
        start_nearest = None
        start_dist_meters = float('inf')
        start_alternatives = []

        # Find up to 20 closest candidates for debug lines to see what's happening
        num_candidates = 20 if is_debug else max_alternatives * 5

        if is_debug:
            print("\n  START POINT CANDIDATES:")

        for i in network_endpoint_idx.nearest((start_point.x, start_point.y, start_point.x, start_point.y),
                                              num_candidates):
            network_idx, pos, network_point, network_id, network_voltage = network_endpoints[i]

            # Calculate distance in degrees
            dist_degrees = start_point.distance(network_point)
            # Convert to approximate meters
            dist_meters = dist_degrees * avg_degree_meters

            if is_debug:
                voltage_match = (jao_voltage == 220 and network_voltage == 220) or (
                            jao_voltage == 400 and network_voltage == 380)
                print(
                    f"    Network {network_id} ({pos} point) - Distance: {dist_meters:.2f}m, Voltage: {network_voltage} kV (match: {voltage_match})")
                print(f"      Coordinates: ({network_point.x}, {network_point.y})")
                print(f"      Within threshold: {dist_meters <= distance_threshold_meters}")

            # Consider voltage constraint - try to match same voltage if possible
            voltage_match = (jao_voltage == 220 and network_voltage == 220) or (
                        jao_voltage == 400 and network_voltage == 380)

            # First try to find exact voltage matches within threshold
            if voltage_match and dist_meters <= distance_threshold_meters:
                if len(start_alternatives) < max_alternatives:
                    start_alternatives.append((network_idx, pos))

                if dist_meters < start_dist_meters:
                    start_nearest = (network_idx, pos)
                    start_dist_meters = dist_meters

        # If no voltage matches found, accept any within threshold
        if start_nearest is None and distance_threshold_meters > 0:
            if is_debug:
                print("  No voltage-matching endpoints found within threshold, trying any voltage")

            for i in network_endpoint_idx.nearest((start_point.x, start_point.y, start_point.x, start_point.y),
                                                  num_candidates):
                network_idx, pos, network_point, network_id, network_voltage = network_endpoints[i]

                # Calculate distance
                dist_degrees = start_point.distance(network_point)
                dist_meters = dist_degrees * avg_degree_meters

                if dist_meters <= distance_threshold_meters:
                    if len(start_alternatives) < max_alternatives:
                        start_alternatives.append((network_idx, pos))

                    if dist_meters < start_dist_meters:
                        start_nearest = (network_idx, pos)
                        start_dist_meters = dist_meters

        # Find nearest network endpoint for JAO end point
        end_nearest = None
        end_dist_meters = float('inf')
        end_alternatives = []

        if is_debug:
            print("\n  END POINT CANDIDATES:")

        for i in network_endpoint_idx.nearest((end_point.x, end_point.y, end_point.x, end_point.y), num_candidates):
            network_idx, pos, network_point, network_id, network_voltage = network_endpoints[i]

            # Calculate distance
            dist_degrees = end_point.distance(network_point)
            dist_meters = dist_degrees * avg_degree_meters

            if is_debug:
                voltage_match = (jao_voltage == 220 and network_voltage == 220) or (
                            jao_voltage == 400 and network_voltage == 380)
                print(
                    f"    Network {network_id} ({pos} point) - Distance: {dist_meters:.2f}m, Voltage: {network_voltage} kV (match: {voltage_match})")
                print(f"      Coordinates: ({network_point.x}, {network_point.y})")
                print(f"      Within threshold: {dist_meters <= distance_threshold_meters}")

            # Consider voltage constraint
            voltage_match = (jao_voltage == 220 and network_voltage == 220) or (
                        jao_voltage == 400 and network_voltage == 380)

            # First try to find exact voltage matches within threshold
            if voltage_match and dist_meters <= distance_threshold_meters:
                if len(end_alternatives) < max_alternatives:
                    end_alternatives.append((network_idx, pos))

                if dist_meters < end_dist_meters:
                    end_nearest = (network_idx, pos)
                    end_dist_meters = dist_meters

        # If no voltage matches found, accept any within threshold
        if end_nearest is None and distance_threshold_meters > 0:
            if is_debug:
                print("  No voltage-matching endpoints found within threshold, trying any voltage")

            for i in network_endpoint_idx.nearest((end_point.x, end_point.y, end_point.x, end_point.y), num_candidates):
                network_idx, pos, network_point, network_id, network_voltage = network_endpoints[i]

                # Calculate distance
                dist_degrees = end_point.distance(network_point)
                dist_meters = dist_degrees * avg_degree_meters

                if dist_meters <= distance_threshold_meters:
                    if len(end_alternatives) < max_alternatives:
                        end_alternatives.append((network_idx, pos))

                    if dist_meters < end_dist_meters:
                        end_nearest = (network_idx, pos)
                        end_dist_meters = dist_meters

        # Special handling for substations - if endpoint is extremely close (within ~10m),
        # consider it a match even if not the absolute closest
        substation_threshold = 10  # meters

        if is_debug:
            print(f"\n  MATCHING RESULTS:")
            print(f"    Start point matched: {start_nearest is not None}, distance: {start_dist_meters:.2f}m")
            print(f"    End point matched: {end_nearest is not None}, distance: {end_dist_meters:.2f}m")

            if start_nearest:
                network_idx, pos = start_nearest
                _, _, _, network_id, network_voltage = next(
                    (ne for ne in network_endpoints if ne[0] == network_idx and ne[1] == pos),
                    (None, None, None, "Unknown", 0))
                print(f"    Start matched with: Network {network_id} ({network_voltage} kV)")

            if end_nearest:
                network_idx, pos = end_nearest
                _, _, _, network_id, network_voltage = next(
                    (ne for ne in network_endpoints if ne[0] == network_idx and ne[1] == pos),
                    (None, None, None, "Unknown", 0))
                print(f"    End matched with: Network {network_id} ({network_voltage} kV)")

        nearest_points_dict[idx] = {
            'start_nearest': start_nearest,
            'end_nearest': end_nearest,
            'start_alternatives': start_alternatives,
            'end_alternatives': end_alternatives,
            'start_distance_meters': start_dist_meters if start_nearest else float('inf'),
            'end_distance_meters': end_dist_meters if end_nearest else float('inf')
        }

    # Print statistics
    total_jao = len(jao_gdf)
    start_matches = sum(1 for data in nearest_points_dict.values() if data['start_nearest'] is not None)
    end_matches = sum(1 for data in nearest_points_dict.values() if data['end_nearest'] is not None)
    both_matches = sum(1 for data in nearest_points_dict.values()
                       if data['start_nearest'] is not None and data['end_nearest'] is not None)

    print(f"Endpoint matching statistics (distance threshold: {distance_threshold_meters} meters):")
    print(f"  JAO lines with start point matched: {start_matches}/{total_jao} ({start_matches / total_jao * 100:.1f}%)")
    print(f"  JAO lines with end point matched: {end_matches}/{total_jao} ({end_matches / total_jao * 100:.1f}%)")
    print(
        f"  JAO lines with both endpoints matched: {both_matches}/{total_jao} ({both_matches / total_jao * 100:.1f}%)")

    return nearest_points_dict



# Function to allocate electrical parameters from JAO to network lines - using CSV length data
def allocate_electrical_parameters(jao_gdf, network_gdf, matching_results):
    # Enhanced results with electrical parameters
    enhanced_results = []

    for result in matching_results:
        if result['matched'] and result['network_ids']:
            jao_id = result['jao_id']
            jao_rows = jao_gdf[jao_gdf['id'].astype(str) == jao_id]

            if jao_rows.empty:
                print(f"Warning: JAO ID {jao_id} not found in JAO GeoDataFrame")
                enhanced_results.append(result.copy())
                continue

            jao_row = jao_rows.iloc[0]

            # Get JAO electrical parameters (if available)
            jao_r = float(jao_row.get('r', 0)) if 'r' in jao_row else 0
            jao_x = float(jao_row.get('x', 0)) if 'x' in jao_row else 0
            jao_b = float(jao_row.get('b', 0)) if 'b' in jao_row else 0

            # Prioritize using the 'length' column for JAO
            if 'length' in jao_row:
                jao_length_km = float(jao_row['length'])
                print(f"Using 'length' column for JAO {jao_id}: {jao_length_km} km")
            else:
                # Fallback to other methods if 'length' is not available
                jao_length_km = result.get('jao_length', 0) / 1000 if 'jao_length' in result else 0

                # If length is still 0, try to calculate it from geometry
                if jao_length_km == 0 and hasattr(jao_row, 'geometry') and jao_row.geometry is not None:
                    try:
                        # Try to get length in meters and convert to km
                        jao_length_m = float(jao_row.geometry.length)
                        jao_length_km = jao_length_m / 1000
                        print(f"Calculated JAO length for {jao_id}: {jao_length_km} km")
                    except Exception as e:
                        print(f"Error calculating JAO length for {jao_id}: {e}")

            # Calculate JAO per-km values (with safety check for zero division)
            jao_r_per_km = jao_r / jao_length_km if jao_length_km > 0 else 0
            jao_x_per_km = jao_x / jao_length_km if jao_length_km > 0 else 0
            jao_b_per_km = jao_b / jao_length_km if jao_length_km > 0 else 0

            # Get matched network lines
            matched_lines = []
            total_network_length_km = 0

            # First, calculate total network length using the 'length' column from CSV
            for network_id in result['network_ids']:
                network_rows = network_gdf[network_gdf['id'].astype(str) == network_id]

                if not network_rows.empty:
                    line = network_rows.iloc[0]
                    line_length_km = 0

                    # Prioritize using the 'length' column from CSV
                    if 'length' in line:
                        try:
                            line_length_km = float(line['length'])
                            print(f"Using 'length' from CSV for {network_id}: {line_length_km} km")
                        except Exception as e:
                            print(f"Error using 'length' from CSV for {network_id}: {e}")

                    # Only fallback to geometry if 'length' is not available or zero
                    if line_length_km == 0 and hasattr(line, 'geometry') and line.geometry is not None:
                        try:
                            line_length_m = float(line.geometry.length)
                            line_length_km = line_length_m / 1000
                            print(f"Calculated length from geometry for {network_id}: {line_length_km} km")
                        except Exception as e:
                            print(f"Error calculating length from geometry for {network_id}: {e}")

                    # Add to total network length
                    total_network_length_km += line_length_km

            # If total network length is still 0, use the JAO length as a fallback
            if total_network_length_km == 0:
                total_network_length_km = jao_length_km
                print(f"Warning: Using JAO length as fallback for network length for JAO {jao_id}")

            # Process each matched network line
            for network_id in result['network_ids']:
                network_rows = network_gdf[network_gdf['id'].astype(str) == network_id]

                if not network_rows.empty:
                    line = network_rows.iloc[0]
                    line_length_km = 0

                    # Prioritize using the 'length' column from CSV
                    if 'length' in line:
                        try:
                            line_length_km = float(line['length'])
                        except Exception as e:
                            print(f"Error using 'length' from CSV for {network_id}: {e}")

                    # Only fallback to geometry if 'length' is not available or zero
                    if line_length_km == 0 and hasattr(line, 'geometry') and line.geometry is not None:
                        try:
                            line_length_m = float(line.geometry.length)
                            line_length_km = line_length_m / 1000
                        except Exception as e:
                            print(f"Error calculating length from geometry for {network_id}: {e}")

                    # Calculate length ratio for this segment
                    segment_ratio = line_length_km / total_network_length_km if total_network_length_km > 0 else 0

                    # Allocate JAO parameters based on length ratio
                    allocated_r = jao_r * segment_ratio
                    allocated_x = jao_x * segment_ratio
                    allocated_b = jao_b * segment_ratio

                    # Get original network parameters (if available)
                    original_r = float(line.get('r', 0)) if 'r' in line else 0
                    original_x = float(line.get('x', 0)) if 'x' in line else 0
                    original_b = float(line.get('b', 0)) if 'b' in line else 0

                    # Calculate per-km values for the network line (with safety checks)
                    if line_length_km > 0:
                        original_r_per_km = original_r / line_length_km
                        original_x_per_km = original_x / line_length_km
                        original_b_per_km = original_b / line_length_km

                        allocated_r_per_km = allocated_r / line_length_km
                        allocated_x_per_km = allocated_x / line_length_km
                        allocated_b_per_km = allocated_b / line_length_km
                    else:
                        # If length is zero, use JAO per-km values as fallback
                        original_r_per_km = 0
                        original_x_per_km = 0
                        original_b_per_km = 0

                        allocated_r_per_km = jao_r_per_km
                        allocated_x_per_km = jao_x_per_km
                        allocated_b_per_km = jao_b_per_km

                    # Calculate differences (with safety checks)
                    if original_r != 0:
                        r_diff_percent = ((allocated_r - original_r) / original_r * 100)
                    else:
                        r_diff_percent = float('inf')

                    if original_x != 0:
                        x_diff_percent = ((allocated_x - original_x) / original_x * 100)
                    else:
                        x_diff_percent = float('inf')

                    if original_b != 0:
                        b_diff_percent = ((allocated_b - original_b) / original_b * 100)
                    else:
                        b_diff_percent = float('inf')

                    # Add to matched lines
                    matched_lines.append({
                        'network_id': network_id,
                        'length_km': line_length_km,
                        'segment_ratio': segment_ratio,
                        'allocated_r': allocated_r,
                        'allocated_x': allocated_x,
                        'allocated_b': allocated_b,
                        'original_r': original_r,
                        'original_x': original_x,
                        'original_b': original_b,
                        'allocated_r_per_km': allocated_r_per_km,
                        'allocated_x_per_km': allocated_x_per_km,
                        'allocated_b_per_km': allocated_b_per_km,
                        'original_r_per_km': original_r_per_km,
                        'original_x_per_km': original_x_per_km,
                        'original_b_per_km': original_b_per_km,
                        'r_diff_percent': r_diff_percent,
                        'x_diff_percent': x_diff_percent,
                        'b_diff_percent': b_diff_percent
                    })

            # Add enhanced data to result
            result_with_parameters = result.copy()
            result_with_parameters['matched_lines_data'] = matched_lines
            result_with_parameters['jao_r'] = jao_r
            result_with_parameters['jao_x'] = jao_x
            result_with_parameters['jao_b'] = jao_b
            result_with_parameters['jao_length_km'] = jao_length_km
            result_with_parameters['jao_r_per_km'] = jao_r_per_km
            result_with_parameters['jao_x_per_km'] = jao_x_per_km
            result_with_parameters['jao_b_per_km'] = jao_b_per_km
            result_with_parameters['total_network_length_km'] = total_network_length_km

            enhanced_results.append(result_with_parameters)
        else:
            # Just copy the unmatched result
            enhanced_results.append(result.copy())

    return enhanced_results


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


def identify_duplicate_jao_lines(jao_gdf):
    """
    Identify groups of JAO lines with identical geometries.
    Returns a dictionary mapping JAO IDs to their group ID.
    """
    print("Identifying duplicate JAO lines...")

    # Create a dictionary to store groups
    geometry_groups = {}
    jao_to_group = {}

    # Use WKT representation of geometry as key to group identical geometries
    for idx, row in jao_gdf.iterrows():
        jao_id = str(row['id'])
        geom_wkt = row.geometry.wkt

        if geom_wkt in geometry_groups:
            # Add to existing group
            geometry_groups[geom_wkt].append(jao_id)
        else:
            # Create new group
            geometry_groups[geom_wkt] = [jao_id]

    # Assign group IDs only to geometries with multiple JAO lines
    group_id = 0
    for geom_wkt, jao_ids in geometry_groups.items():
        if len(jao_ids) > 1:
            group_id += 1
            print(f"Group {group_id}: Found {len(jao_ids)} duplicate JAO lines with identical geometry:")
            for jao_id in jao_ids:
                jao_to_group[jao_id] = group_id
                print(f"  - JAO {jao_id}")

    print(f"Found {group_id} groups of duplicate JAO lines")
    return jao_to_group, geometry_groups


def extract_path_details(G, path, network_gdf):
    """Helper function to extract network IDs and calculate path length from a path."""
    network_ids = []
    unique_ids = set()
    path_edges = []

    for i in range(len(path) - 1):
        edge_data = G.get_edge_data(path[i], path[i + 1])

        if edge_data and 'id' in edge_data and not edge_data.get('connector', False):
            if edge_data['id'] not in unique_ids:
                network_ids.append(str(edge_data['id']))
                unique_ids.add(edge_data['id'])

            path_edges.append((path[i], path[i + 1], edge_data))

    # If no network lines in the path, return empty
    if not network_ids:
        return [], 0, []

    # Calculate path length
    path_length = 0
    for network_id in network_ids:
        network_line = network_gdf[network_gdf['id'].astype(str) == network_id]
        if not network_line.empty:
            if 'length' in network_line.iloc[0] and network_line.iloc[0]['length']:
                line_length = float(network_line.iloc[0]['length']) * 1000  # Convert to meters
            else:
                line_length = float(network_line.iloc[0].geometry.length)
            path_length += line_length

    return network_ids, path_length, path_edges


# Helper function for proper length calculation
def calculate_length_meters(geometry):
    """Calculate length in meters for a geometry, accounting for coordinate system."""
    if geometry is None:
        return 0

    # If the geometry uses geographic coordinates (lon/lat)
    if isinstance(geometry, (LineString, MultiLineString)):
        # Get centroid latitude for conversion
        centroid_lat = geometry.centroid.y
        # Approximate meters per degree at this latitude
        meters_per_degree = 111111 * np.cos(np.radians(abs(centroid_lat)))
        # Convert length from degrees to meters
        return float(geometry.length) * meters_per_degree
    else:
        # For other coordinate systems, just return the length
        return float(geometry.length)


def find_matching_network_lines_with_duplicates(jao_gdf, network_gdf, nearest_points_dict, G,
                                                duplicate_groups, max_reuse=3, max_paths_to_try=20,
                                                min_length_ratio=0.5, max_length_ratio=1.7):
    """
    Find matching network lines with special handling for duplicate JAO lines.
    Evaluates multiple possible paths and selects the best one based on length and voltage match.
    """
    results = []

    # Track how many times each network line has been used
    network_line_usage = {str(row['id']): 0 for _, row in network_gdf.iterrows()}

    # Create a reverse lookup from JAO ID to its duplicate group
    jao_to_group = {}
    for geom_wkt, jao_ids in duplicate_groups.items():
        if len(jao_ids) > 1:
            # Only track duplicates (groups with multiple JAO lines)
            for jao_id in jao_ids:
                jao_to_group[jao_id] = geom_wkt

    # Track which duplicate groups have been matched
    matched_groups = {}

    # Process JAO lines by ID order for reproducibility
    jao_with_idx = [(idx, row) for idx, row in jao_gdf.iterrows()]
    jao_with_idx.sort(key=lambda x: str(x[1]['id']))

    print(f"Processing {len(jao_with_idx)} JAO lines...")
    print(f"Using strict reuse limit of {max_reuse}")
    print(f"Evaluating up to {max_paths_to_try} possible paths per JAO line")
    print(f"Length ratio constraints: network/JAO must be between {min_length_ratio:.1f} and {max_length_ratio:.1f}")

    for idx, row in jao_with_idx:
        jao_id = str(row['id'])
        jao_name = str(row['NE_name'])
        jao_voltage = int(row['v_nom'])

        # Calculate JAO length properly
        jao_length_meters = calculate_length_meters(row.geometry)
        jao_length_km = jao_length_meters / 1000.0

        print(f"\nProcessing JAO line {jao_id} ({jao_name}) with length {jao_length_km:.2f} km")

        # Handle duplicate JAO lines
        is_duplicate = jao_id in jao_to_group
        duplicate_group = jao_to_group.get(jao_id, None)

        if is_duplicate and duplicate_group in matched_groups:
            # This is a duplicate of an already matched JAO line
            print(f"\nProcessing duplicate JAO line {jao_id} ({jao_name})")
            print(f"  This is a duplicate of already matched JAO(s): {matched_groups[duplicate_group]}")

            # Copy the match from the first matched JAO in this group
            first_match = None
            for result in results:
                if result['jao_id'] in matched_groups[duplicate_group]:
                    first_match = result
                    break

            if first_match and first_match['matched']:
                # Create a duplicate match with special status
                duplicate_result = {
                    'jao_id': jao_id,
                    'jao_name': jao_name,
                    'v_nom': int(jao_voltage),
                    'matched': True,
                    'is_duplicate': True,  # Make sure this is set to True
                    'duplicate_of': first_match['jao_id'],
                    'path': first_match['path'].copy() if 'path' in first_match else [],
                    'network_ids': first_match['network_ids'].copy() if 'network_ids' in first_match else [],
                    'path_length': first_match.get('path_length', 0),
                    'jao_length': float(jao_length_meters),  # Use meters consistently
                    'length_ratio': first_match.get('length_ratio', 1.0),
                    'match_quality': 'Parallel Circuit (duplicate geometry)'
                }

                results.append(duplicate_result)
                print(f"  Marked as parallel circuit of {first_match['jao_id']}")

                # Make sure this duplicate is recorded in matched_groups
                if duplicate_group not in matched_groups:
                    matched_groups[duplicate_group] = []
                matched_groups[duplicate_group].append(jao_id)

                continue

        # Get node IDs for start and end points
        if idx not in nearest_points_dict or nearest_points_dict[idx]['start_nearest'] is None:
            print(f"  No start point match for {jao_id}")
            results.append({
                'jao_id': jao_id,
                'jao_name': jao_name,
                'v_nom': int(jao_voltage),
                'matched': False,
                'path': [],
                'network_ids': [],
                'match_quality': 'No match - endpoints not found'
            })
            continue

        start_idx, start_pos = nearest_points_dict[idx]['start_nearest']
        if idx not in nearest_points_dict or nearest_points_dict[idx]['end_nearest'] is None:
            print(f"  No end point match for {jao_id}")
            results.append({
                'jao_id': jao_id,
                'jao_name': jao_name,
                'v_nom': int(jao_voltage),
                'matched': False,
                'path': [],
                'network_ids': [],
                'match_quality': 'No match - endpoints not found'
            })
            continue

        end_idx, end_pos = nearest_points_dict[idx]['end_nearest']

        start_node = f"node_{start_idx}_{start_pos}"
        end_node = f"node_{end_idx}_{end_pos}"

        # Skip if start and end are the same
        if start_node == end_node:
            print(f"  Start and end points are the same for {jao_id}")
            results.append({
                'jao_id': jao_id,
                'jao_name': jao_name,
                'v_nom': int(jao_voltage),
                'matched': False,
                'path': [],
                'network_ids': [],
                'match_quality': 'No match - start and end are the same'
            })
            continue

        # Check if start and end indices refer to the same network line
        if start_idx == end_idx and start_idx is not None:
            # This might be a case where a single network line matches the JAO
            network_id = None
            for u, v, data in G.edges(data=True):
                if 'idx' in data and data['idx'] == start_idx and not data.get('connector', False):
                    network_id = data.get('id')
                    break

            if network_id:
                # Found a direct single-line match
                network_row = network_gdf[network_gdf['id'].astype(str) == str(network_id)]
                if not network_row.empty:
                    network_length = calculate_length_meters(network_row.iloc[0].geometry)
                    length_ratio = network_length / jao_length_meters if jao_length_meters > 0 else float('inf')

                    if min_length_ratio <= length_ratio <= max_length_ratio:
                        print(f"  Found direct single-line match with network line {network_id}")
                        print(f"  Length ratio: {length_ratio:.2f}")

                        # Create result
                        result = {
                            'jao_id': jao_id,
                            'jao_name': jao_name,
                            'v_nom': int(jao_voltage),
                            'matched': True,
                            'is_duplicate': False,
                            'path': [start_node, end_node],  # Simple direct path
                            'network_ids': [str(network_id)],
                            'path_length': float(network_length),
                            'jao_length': float(jao_length_meters),
                            'length_ratio': float(length_ratio),
                            'match_quality': 'Direct Single-Line Match'
                        }

                        # Add voltage mismatch note if needed
                        if network_row.iloc[0]['v_nom'] != jao_voltage:
                            result['match_quality'] += ' (voltage mismatch)'

                        results.append(result)

                        # Increment usage count
                        network_line_usage[str(network_id)] += 1

                        # If this is a duplicate JAO line, record that its group has been matched
                        if is_duplicate:
                            if duplicate_group not in matched_groups:
                                matched_groups[duplicate_group] = []
                            matched_groups[duplicate_group].append(jao_id)

                        print(f"  Successfully matched {jao_id} with network line {network_id}")
                        continue
                    else:
                        print(
                            f"  Direct match found but length ratio {length_ratio:.2f} outside limits {min_length_ratio:.1f}-{max_length_ratio:.1f}")

        # Create copies of the graph with constraints
        H_strict = G.copy()
        H_reuse = G.copy()

        # Apply constraints to each graph
        edges_to_remove_strict = []
        edges_to_remove_reuse = []

        for u, v, data in G.edges(data=True):
            if 'connector' not in data or not data['connector']:  # Only check actual network lines
                if 'id' in data:
                    # Check reuse limit for both graphs
                    if network_line_usage[data['id']] >= max_reuse:
                        edges_to_remove_strict.append((u, v))
                        edges_to_remove_reuse.append((u, v))

                    # Check voltage constraint only for strict graph
                    if 'voltage' in data:
                        # Modified voltage matching to handle 380/400 equivalence
                        if not ((jao_voltage == 220 and data['voltage'] == 220) or
                                (jao_voltage == 400 and data['voltage'] in [380, 400]) or
                                (jao_voltage == 380 and data['voltage'] in [380, 400])):
                            edges_to_remove_strict.append((u, v))

        # Remove edges from each graph
        for edge in edges_to_remove_strict:
            if H_strict.has_edge(*edge):
                H_strict.remove_edge(*edge)

        for edge in edges_to_remove_reuse:
            if H_reuse.has_edge(*edge):
                H_reuse.remove_edge(*edge)

        # Structure to store candidate paths
        candidate_paths = []

        # Try different path finding approaches
        try:
            # 1. First try with strict constraints (voltage + reuse)
            print("  Finding paths with strict constraints (voltage + reuse limit)...")
            try:
                k_paths = list(itertools.islice(
                    nx.shortest_simple_paths(H_strict, start_node, end_node, weight='weight'),
                    max_paths_to_try))

                # Process strict paths...
                for i, path in enumerate(k_paths):
                    network_ids, path_length, path_edges = extract_path_details(G, path, network_gdf)

                    if network_ids:  # Only consider if there are actual network lines in the path
                        length_ratio = path_length / jao_length_meters if jao_length_meters > 0 else float('inf')

                        # More permissive filtering of paths with length ratio
                        if min_length_ratio <= length_ratio <= max_length_ratio:
                            # Determine match quality based on length ratio
                            if 0.9 <= length_ratio <= 1.1:
                                match_quality = 'Excellent'
                                quality_score = 4
                            elif 0.8 <= length_ratio <= 1.2:
                                match_quality = 'Good'
                                quality_score = 3
                            elif 0.7 <= length_ratio <= 1.3:
                                match_quality = 'Fair'
                                quality_score = 2
                            else:
                                match_quality = 'Poor (length mismatch)'
                                quality_score = 1

                            # More emphasis on ratio score
                            ratio_score = 1.0 - abs(length_ratio - 1.0)
                            path_score = 50 + (ratio_score * 50) + quality_score

                            print(
                                f"    Path {i + 1}: Score {path_score:.2f}, length ratio {length_ratio:.2f}, quality: {match_quality}")

                            candidate_paths.append({
                                'path': path,
                                'network_ids': network_ids,
                                'path_length': path_length,
                                'length_ratio': length_ratio,
                                'match_quality': match_quality,
                                'constraints_used': 'strict',
                                'score': path_score,
                                'path_edges': path_edges
                            })
            except nx.NetworkXNoPath:
                print("  No path found with strict constraints")
            except Exception as e:
                print(f"  Error in strict path finding: {e}")

            # 2. If strict approach found nothing, try with relaxed voltage constraints
            if not candidate_paths:
                print("  Finding paths with reuse limit only (no voltage constraint)...")
                try:
                    k_paths = list(itertools.islice(
                        nx.shortest_simple_paths(H_reuse, start_node, end_node, weight='weight'),
                        max_paths_to_try))

                    # Process paths with relaxed voltage constraints...
                    for i, path in enumerate(k_paths):
                        network_ids, path_length, path_edges = extract_path_details(G, path, network_gdf)

                        if network_ids:
                            length_ratio = path_length / jao_length_meters if jao_length_meters > 0 else float('inf')

                            if min_length_ratio <= length_ratio <= max_length_ratio:
                                # Similar quality assessment but always note voltage mismatch
                                if 0.9 <= length_ratio <= 1.1:
                                    base_quality = 'Excellent'
                                    quality_score = 3
                                elif 0.8 <= length_ratio <= 1.2:
                                    base_quality = 'Good'
                                    quality_score = 2
                                elif 0.7 <= length_ratio <= 1.3:
                                    base_quality = 'Fair'
                                    quality_score = 1
                                else:
                                    base_quality = 'Poor'
                                    quality_score = 0

                                match_quality = f"{base_quality} (voltage mismatch)"
                                ratio_score = 1.0 - abs(length_ratio - 1.0)
                                path_score = 40 + (ratio_score * 40) + quality_score  # Slightly lower base score

                                print(
                                    f"    Path {i + 1}: Score {path_score:.2f}, length ratio {length_ratio:.2f}, quality: {match_quality}")

                                candidate_paths.append({
                                    'path': path,
                                    'network_ids': network_ids,
                                    'path_length': path_length,
                                    'length_ratio': length_ratio,
                                    'match_quality': match_quality,
                                    'constraints_used': 'reuse_only',
                                    'score': path_score,
                                    'path_edges': path_edges
                                })
                except nx.NetworkXNoPath:
                    print("  No path found with relaxed voltage constraints")
                except Exception as e:
                    print(f"  Error in relaxed path finding: {e}")

            # 3. If still no paths, try with alternative endpoints
            if not candidate_paths and 'start_alternatives' in nearest_points and 'end_alternatives' in nearest_points:
                print("  Trying alternative endpoints...")

                # Try combinations of alternative start and end points
                for start_alt in nearest_points['start_alternatives'][:3]:  # Limit to top 3 alternatives
                    for end_alt in nearest_points['end_alternatives'][:3]:
                        alt_start_node = f"node_{start_alt[0]}_{start_alt[1]}"
                        alt_end_node = f"node_{end_alt[0]}_{end_alt[1]}"

                        if alt_start_node == alt_end_node:
                            continue

                        try:
                            # Try to find path with these alternative endpoints
                            alt_path = next(
                                nx.shortest_simple_paths(H_reuse, alt_start_node, alt_end_node, weight='weight'))

                            network_ids, path_length, path_edges = extract_path_details(G, alt_path, network_gdf)

                            if network_ids:
                                length_ratio = path_length / jao_length_meters if jao_length_meters > 0 else float(
                                    'inf')

                                if min_length_ratio <= length_ratio <= max_length_ratio:
                                    # Use a lower score for alternative endpoints
                                    ratio_score = 1.0 - abs(length_ratio - 1.0)
                                    path_score = 30 + (ratio_score * 30)
                                    match_quality = "Alternative Endpoints"

                                    print(
                                        f"    Alternative path: Score {path_score:.2f}, length ratio {length_ratio:.2f}")

                                    candidate_paths.append({
                                        'path': alt_path,
                                        'network_ids': network_ids,
                                        'path_length': path_length,
                                        'length_ratio': length_ratio,
                                        'match_quality': match_quality,
                                        'constraints_used': 'alternative_endpoints',
                                        'score': path_score,
                                        'path_edges': path_edges
                                    })
                        except (nx.NetworkXNoPath, StopIteration):
                            # No path for this alternative pair
                            continue
                        except Exception as e:
                            print(f"  Error with alternative endpoints: {e}")
                            continue

        except Exception as e:
            print(f"  Unexpected error in path finding: {e}")

        # If no paths found, mark as unmatched
        if not candidate_paths:
            print(f"  No valid paths found for {jao_id}")
            results.append({
                'jao_id': jao_id,
                'jao_name': jao_name,
                'v_nom': int(jao_voltage),
                'matched': False,
                'path': [],
                'network_ids': [],
                'match_quality': 'No path found with constraints'
            })
            continue

        # Sort candidates by score (highest first)
        candidate_paths.sort(key=lambda x: x['score'], reverse=True)

        # Select the best path
        best_path = candidate_paths[0]

        print(f"  Selected best path with score {best_path['score']:.2f}, length ratio {best_path['length_ratio']:.2f}")
        print(f"  Using constraint type: {best_path['constraints_used']}")
        print(f"  Match quality: {best_path['match_quality']}")
        print(f"  Network path length: {best_path['path_length'] / 1000:.2f} km vs JAO length: {jao_length_km:.2f} km")

        # Print information about alternatives if available
        if len(candidate_paths) > 1:
            print(f"  Alternative paths were available:")
            for i, alt_path in enumerate(candidate_paths[1:3]):  # Show up to 2 alternatives
                print(
                    f"    Alt {i + 1}: score {alt_path['score']:.2f}, length ratio {alt_path['length_ratio']:.2f}, quality: {alt_path['match_quality']}")

        # Increment usage count for matched network lines
        for network_id in best_path['network_ids']:
            network_line_usage[network_id] += 1
            print(f"  Network line {network_id} usage count now: {network_line_usage[network_id]}")

        # Create the result
        result = {
            'jao_id': jao_id,
            'jao_name': jao_name,
            'v_nom': int(jao_voltage),
            'matched': True,
            'is_duplicate': False,
            'path': [str(p) for p in best_path['path']],
            'network_ids': best_path['network_ids'],
            'path_length': float(best_path['path_length']),
            'jao_length': float(jao_length_meters),
            'length_ratio': float(best_path['length_ratio']),
            'match_quality': best_path['match_quality'],
            'constraints_used': best_path['constraints_used']
        }

        results.append(result)

        # If this is a duplicate JAO line, record that its group has been matched
        if is_duplicate:
            if duplicate_group not in matched_groups:
                matched_groups[duplicate_group] = []
            matched_groups[duplicate_group].append(jao_id)

        print(f"  Successfully matched {jao_id} with {len(best_path['network_ids'])} network lines")

    # Print usage statistics and other info as before...
    # (rest of the function remains the same)

    return results


def debug_specific_jao_match(jao_gdf, network_gdf, matching_results, jao_id_to_debug="97",
                             target_network_ids=["Line_8160", "Line_30733", "Line_30181", "Line_17856"]):
    """
    Debug function to investigate why a specific JAO is not being matched correctly.
    """
    print(f"\n=== DEBUGGING JAO {jao_id_to_debug} MATCHING ===")

    # Find the JAO in the dataframe
    jao_rows = jao_gdf[jao_gdf['id'].astype(str) == jao_id_to_debug]
    if jao_rows.empty:
        print(f"ERROR: JAO {jao_id_to_debug} not found in dataset")
        return matching_results

    jao_row = jao_rows.iloc[0]
    jao_name = str(jao_row['NE_name'])
    jao_voltage = int(jao_row['v_nom'])
    jao_geom = jao_row.geometry
    jao_length = calculate_length_meters(jao_geom)

    print(f"JAO {jao_id_to_debug}: {jao_name}")
    print(f"Voltage: {jao_voltage} kV")
    print(f"Length: {jao_length / 1000:.2f} km")

    # Check if this JAO is already matched
    is_matched = False
    existing_match = None
    for result in matching_results:
        if result['jao_id'] == jao_id_to_debug and result['matched']:
            is_matched = True
            existing_match = result
            break

    if is_matched:
        print(f"JAO is currently matched with: {existing_match['network_ids']}")
        print(f"Match quality: {existing_match['match_quality']}")
        if 'is_parallel_circuit' in existing_match and existing_match['is_parallel_circuit']:
            print(f"Marked as parallel circuit")
            if 'parallel_to_jao' in existing_match:
                print(f"Parallel to JAO: {existing_match['parallel_to_jao']}")
    else:
        print("JAO is currently UNMATCHED")

    # Check target network lines
    print("\nChecking target network lines...")
    target_network_geoms = []
    target_network_lengths = 0

    for network_id in target_network_ids:
        network_rows = network_gdf[network_gdf['id'].astype(str) == network_id]
        if network_rows.empty:
            print(f"  Network line {network_id}: NOT FOUND in dataset")
            continue

        network_row = network_rows.iloc[0]
        network_geom = network_row.geometry
        network_voltage = int(network_row['v_nom'])
        network_length = calculate_length_meters(network_geom)
        target_network_geoms.append(network_geom)
        target_network_lengths += network_length

        # Check if already used in another match
        used_in = []
        for result in matching_results:
            if result['matched'] and 'network_ids' in result and network_id in result['network_ids']:
                used_in.append(result['jao_id'])

        used_status = f"Used by JAO(s): {', '.join(used_in)}" if used_in else "Not used in any match"

        print(f"  Network line {network_id}: {network_voltage} kV, {network_length / 1000:.2f} km - {used_status}")

    # Create a MultiLineString from all target network lines
    if target_network_geoms:
        from shapely.geometry import MultiLineString
        target_network_multi = MultiLineString(target_network_geoms)

        # Calculate coverage and Hausdorff distance
        avg_lat = (target_network_multi.centroid.y + jao_geom.centroid.y) / 2
        meters_per_degree = 111111 * np.cos(np.radians(abs(avg_lat)))

        # Create buffer around the JAO
        buffer_m = 200  # 200m buffer
        buffer_deg = buffer_m / meters_per_degree
        jao_buffer = jao_geom.buffer(buffer_deg)

        # Check how much of network is within this buffer
        network_in_buffer = target_network_multi.intersection(jao_buffer)
        network_coverage = network_in_buffer.length / target_network_multi.length if target_network_multi.length > 0 else 0

        # Check how much of JAO is covered by network buffer
        network_buffer = target_network_multi.buffer(buffer_deg)
        jao_in_network_buffer = jao_geom.intersection(network_buffer)
        jao_coverage = jao_in_network_buffer.length / jao_geom.length if jao_geom.length > 0 else 0

        # Calculate Hausdorff distance
        hausdorff_dist = jao_geom.hausdorff_distance(target_network_multi)
        hausdorff_meters = hausdorff_dist * meters_per_degree

        # Calculate length ratio
        length_ratio = target_network_lengths / jao_length if jao_length > 0 else float('inf')

        print(f"\nGeometric analysis of target network path:")
        print(f"  Total network length: {target_network_lengths / 1000:.2f} km")
        print(f"  Length ratio (network/jao): {length_ratio:.2f}")
        print(f"  Network coverage by JAO buffer: {network_coverage:.2f}")
        print(f"  JAO coverage by network buffer: {jao_coverage:.2f}")
        print(f"  Hausdorff distance: {hausdorff_meters:.1f} meters")

        # Check if this would pass our matching criteria
        match_quality = (
                network_coverage >= 0.7 and
                jao_coverage >= 0.7 and
                hausdorff_meters <= 1000 and
                0.8 <= length_ratio <= 1.3
        )

        print(f"  Would this match pass our criteria? {'YES' if match_quality else 'NO'}")

        # Find if this JAO is parallel to any other JAO
        print("\nChecking for parallel JAOs...")
        best_parallel = None
        best_parallel_score = 0

        for idx, row in jao_gdf.iterrows():
            other_id = str(row['id'])

            # Skip self
            if other_id == jao_id_to_debug:
                continue

            other_geom = row.geometry
            other_voltage = int(row['v_nom'])

            # Only consider same voltage
            if other_voltage != jao_voltage:
                continue

            # Check if parallel
            other_buffer = other_geom.buffer(buffer_deg)
            jao_in_other_buffer = jao_geom.intersection(other_buffer)
            coverage = jao_in_other_buffer.length / jao_geom.length if jao_geom.length > 0 else 0

            # Calculate Hausdorff distance
            try:
                h_dist = jao_geom.hausdorff_distance(other_geom)
                h_meters = h_dist * meters_per_degree

                # Calculate similarity score
                if coverage >= 0.7 and h_meters <= 1000:
                    score = 0.6 * coverage + 0.4 * (1 - h_meters / 1000)

                    # Check if this JAO is matched
                    other_is_matched = False
                    for result in matching_results:
                        if result['jao_id'] == other_id and result['matched']:
                            other_is_matched = True
                            break

                    if score > best_parallel_score and other_is_matched:
                        best_parallel_score = score
                        best_parallel = {
                            'id': other_id,
                            'name': str(row['NE_name']),
                            'coverage': coverage,
                            'hausdorff_meters': h_meters,
                            'score': score
                        }
            except Exception as e:
                print(f"  Error calculating distance to JAO {other_id}: {e}")

        if best_parallel:
            print(f"  Found parallel JAO: {best_parallel['id']} ({best_parallel['name']})")
            print(f"  Coverage: {best_parallel['coverage']:.2f}")
            print(f"  Hausdorff distance: {best_parallel['hausdorff_meters']:.1f} meters")
            print(f"  Similarity score: {best_parallel['score']:.3f}")

            # Check if that JAO uses any of our target network lines
            for result in matching_results:
                if result['jao_id'] == best_parallel['id'] and result['matched']:
                    common_network_ids = set(result.get('network_ids', [])).intersection(set(target_network_ids))
                    if common_network_ids:
                        print(
                            f"  Parallel JAO uses {len(common_network_ids)} of our target network lines: {common_network_ids}")
                    else:
                        print(f"  Parallel JAO doesn't use any of our target network lines")

                    print(f"  Network lines used by parallel JAO: {result.get('network_ids', [])}")
                    break
        else:
            print("  No parallel JAO found")

    # Create a direct match with the target network lines
    print("\nCreating direct match with target network lines...")

    # First check if all target network lines exist
    all_exist = True
    for network_id in target_network_ids:
        network_rows = network_gdf[network_gdf['id'].astype(str) == network_id]
        if network_rows.empty:
            all_exist = False
            print(f"  Cannot create match - Network line {network_id} not found")

    if all_exist:
        # Remove existing match if it exists
        if is_matched:
            print(f"  Removing existing match for JAO {jao_id_to_debug}")
            matching_results = [r for r in matching_results if not (r['jao_id'] == jao_id_to_debug and r['matched'])]

        # Create new match
        new_match = {
            'jao_id': jao_id_to_debug,
            'jao_name': jao_name,
            'v_nom': jao_voltage,
            'matched': True,
            'is_duplicate': False,
            'is_parallel_circuit': True,
            'path': [],
            'network_ids': target_network_ids.copy(),
            'path_length': float(target_network_lengths),
            'jao_length': float(jao_length),
            'length_ratio': float(length_ratio),
            'match_quality': f'Parallel Circuit ({jao_voltage} kV) - Manual Override',
            'match_method': 'manual_override',
            'coverage_ratio': float(jao_coverage),
            'hausdorff_meters': float(hausdorff_meters)
        }

        matching_results.append(new_match)
        print(f"  Created manual match for JAO {jao_id_to_debug} with network lines: {target_network_ids}")

    return matching_results


def match_parallel_circuit_jao_with_network(matching_results, jao_gdf, network_gdf, G, nearest_points_dict):
    """
    Directly match parallel circuit JAO lines with unmatched network lines.
    Uses very relaxed constraints focused on geometric proximity rather than
    graph connectivity.
    """
    print("\nMatching parallel circuit JAO lines with unmatched network lines...")

    # PART 1: Handle special case for JAO 97
    special_case_matches = []
    special_case = {
        "97": ["Line_8160", "Line_30733", "Line_30181", "Line_17856"]
    }

    for jao_id, network_ids in special_case.items():
        jao_rows = jao_gdf[jao_gdf['id'].astype(str) == jao_id]
        if not jao_rows.empty:
            jao_row = jao_rows.iloc[0]

            special_match = {
                'jao_id': jao_id,
                'jao_name': str(jao_row['NE_name']),
                'v_nom': int(jao_row['v_nom']),
                'matched': True,
                'is_duplicate': False,
                'is_parallel_circuit': True,
                'path': [],
                'network_ids': network_ids,
                'jao_length': float(calculate_length_meters(jao_row.geometry)),
                'match_quality': f'Parallel Circuit ({jao_row["v_nom"]} kV) - Special Case'
            }

            # Calculate path length
            path_length = 0
            for network_id in network_ids:
                network_rows = network_gdf[network_gdf['id'].astype(str) == network_id]
                if not network_rows.empty:
                    path_length += calculate_length_meters(network_rows.iloc[0].geometry)

            special_match['path_length'] = float(path_length)
            special_match['length_ratio'] = float(path_length / special_match['jao_length']) if special_match[
                                                                                                    'jao_length'] > 0 else 1.0

            special_case_matches.append(special_match)
            print(f"  Created special match for JAO {jao_id} with network lines: {network_ids}")

    # PART 2: Find all unmatched network lines
    used_network_ids = set()
    for result in matching_results:
        if result['matched'] and 'network_ids' in result:
            for network_id in result['network_ids']:
                used_network_ids.add(str(network_id))

    # Add special case network IDs to used set
    for match in special_case_matches:
        for network_id in match.get('network_ids', []):
            used_network_ids.add(str(network_id))

    # Get all unmatched network lines
    unmatched_network_lines = []
    for idx, row in network_gdf.iterrows():
        network_id = str(row['id'])
        if network_id not in used_network_ids:
            unmatched_network_lines.append({
                'id': network_id,
                'geometry': row.geometry,
                'voltage': int(row['v_nom']),
                'length': calculate_length_meters(row.geometry)
            })

    print(f"Found {len(unmatched_network_lines)} unmatched network lines")

    # PART 3: Find all JAO lines that appear to be parallel circuits

    # First identify all matched JAO lines
    matched_jao_ids = set()
    for result in matching_results:
        if result['matched']:
            matched_jao_ids.add(str(result['jao_id']))

    # Add special case JAO IDs to matched set
    for match in special_case_matches:
        matched_jao_ids.add(str(match['jao_id']))

    # Get geometries of all matched JAO lines for comparison
    matched_jao_geometries = {}
    for idx, row in jao_gdf.iterrows():
        jao_id = str(row['id'])
        if jao_id in matched_jao_ids:
            matched_jao_geometries[jao_id] = {
                'geometry': row.geometry,
                'voltage': int(row['v_nom'])
            }

    # Find all unmatched JAO lines that are likely parallel circuits
    potential_parallel_circuits = []

    for idx, row in jao_gdf.iterrows():
        jao_id = str(row['id'])

        # Skip if already matched
        if jao_id in matched_jao_ids:
            continue

        jao_geom = row.geometry
        jao_voltage = int(row['v_nom'])

        # Check if this JAO runs parallel to any matched JAO
        is_parallel = False
        for matched_id, matched_info in matched_jao_geometries.items():
            # Only compare same voltage lines
            if matched_info['voltage'] != jao_voltage:
                continue

            # Check if geometries are parallel (very relaxed criteria)
            if is_geometry_parallel(jao_geom, matched_info['geometry'],
                                    buffer_meters=1000, min_coverage=0.4):
                is_parallel = True
                break

        # Add to potential parallel circuits if it runs parallel to a matched JAO
        if is_parallel:
            potential_parallel_circuits.append({
                'id': jao_id,
                'name': str(row['NE_name']),
                'geometry': jao_geom,
                'voltage': jao_voltage,
                'length': calculate_length_meters(jao_geom)
            })

    print(f"Found {len(potential_parallel_circuits)} potential parallel circuit JAO lines")

    # PART 4: Match the parallel circuit JAO lines with unmatched network lines
    parallel_matches = []

    for parallel_jao in potential_parallel_circuits:
        jao_id = parallel_jao['id']
        jao_name = parallel_jao['name']
        jao_voltage = parallel_jao['voltage']
        jao_length = parallel_jao['length']
        jao_geom = parallel_jao['geometry']

        print(f"\nProcessing parallel circuit JAO {jao_id} ({jao_name})")

        # Find matching network lines with very relaxed constraints
        matching_network_lines = find_matching_network_lines(
            jao_geom, unmatched_network_lines, jao_voltage,
            buffer_meters=1500,  # Very large buffer
            min_coverage=0.3,  # Very low coverage requirement
            max_lines=4  # Allow up to 4 lines to match
        )

        if matching_network_lines:
            network_ids = [line['id'] for line in matching_network_lines]
            path_length = sum(line['length'] for line in matching_network_lines)
            length_ratio = path_length / jao_length if jao_length > 0 else float('inf')

            print(f"  Found matching network lines: {network_ids}")
            print(f"  Length ratio: {length_ratio:.2f}")

            # Create match result
            match_result = {
                'jao_id': jao_id,
                'jao_name': jao_name,
                'v_nom': jao_voltage,
                'matched': True,
                'is_duplicate': False,
                'is_parallel_circuit': True,
                'path': [],  # No path for direct geometric matching
                'network_ids': network_ids,
                'path_length': float(path_length),
                'jao_length': float(jao_length),
                'length_ratio': float(length_ratio),
                'match_quality': f'Parallel Circuit ({jao_voltage} kV)'
            }

            parallel_matches.append(match_result)

            # Remove these network lines from unmatched pool
            unmatched_network_lines = [line for line in unmatched_network_lines
                                       if line['id'] not in network_ids]
        else:
            print(f"  No matching network lines found for JAO {jao_id}")

    # PART 5: Combine all matches
    all_matches = special_case_matches + parallel_matches

    # Remove any existing matches for these JAOs
    jao_ids_to_match = set(match['jao_id'] for match in all_matches)
    filtered_results = [r for r in matching_results if r['jao_id'] not in jao_ids_to_match]

    # Add parallel matches
    filtered_results.extend(all_matches)

    # Count match statistics
    print(f"\nParallel circuit matching summary:")
    print(f"  {len(special_case_matches)} special case matches")
    print(f"  {len(parallel_matches)} general parallel circuit matches")
    print(f"  Total: {len(all_matches)} parallel circuit matches")

    return filtered_results


# Helper functions

def is_geometry_parallel(geom1, geom2, buffer_meters=1000, min_coverage=0.4):
    """
    Check if two geometries are roughly parallel using a very relaxed buffer-based approach.

    Parameters:
    - geom1, geom2: The geometries to compare
    - buffer_meters: Buffer size in meters
    - min_coverage: Minimum coverage ratio to consider parallel

    Returns:
    - True if geometries are parallel, False otherwise
    """
    try:
        # Calculate buffer size in degrees based on latitude
        avg_lat = (geom1.centroid.y + geom2.centroid.y) / 2
        meters_per_degree = 111111 * np.cos(np.radians(abs(avg_lat)))
        buffer_deg = buffer_meters / meters_per_degree

        # Create buffers
        buffer1 = geom1.buffer(buffer_deg)

        # Check overlap
        intersection = geom2.intersection(buffer1)
        coverage = intersection.length / geom2.length if geom2.length > 0 else 0

        return coverage >= min_coverage
    except Exception as e:
        print(f"  Error checking if geometries are parallel: {e}")
        return False


def find_matching_network_lines(jao_geom, network_lines, jao_voltage,
                                buffer_meters=1500, min_coverage=0.3, max_lines=4):
    """
    Find network lines that match a JAO geometry with very relaxed constraints.

    Parameters:
    - jao_geom: JAO geometry
    - network_lines: List of network line dictionaries
    - jao_voltage: JAO voltage
    - buffer_meters: Buffer size in meters
    - min_coverage: Minimum coverage ratio
    - max_lines: Maximum number of lines to match

    Returns:
    - List of matching network lines
    """
    # Calculate buffer size in degrees
    avg_lat = jao_geom.centroid.y
    meters_per_degree = 111111 * np.cos(np.radians(abs(avg_lat)))
    buffer_deg = buffer_meters / meters_per_degree

    # Create buffer around JAO
    jao_buffer = jao_geom.buffer(buffer_deg)

    # Find matching network lines
    matches = []

    for line in network_lines:
        # Check voltage compatibility (very relaxed)
        voltage_match = False
        if (jao_voltage == 220 and line['voltage'] == 220) or \
                (jao_voltage == 400 and line['voltage'] in [380, 400]) or \
                (jao_voltage == 380 and line['voltage'] in [380, 400]):
            voltage_match = True

        if not voltage_match:
            continue

        # Check if line intersects buffer
        if jao_buffer.intersects(line['geometry']):
            # Calculate coverage
            intersection = line['geometry'].intersection(jao_buffer)
            coverage = intersection.length / line['geometry'].length if line['geometry'].length > 0 else 0

            if coverage >= min_coverage:
                matches.append({
                    'id': line['id'],
                    'geometry': line['geometry'],
                    'voltage': line['voltage'],
                    'length': line['length'],
                    'coverage': coverage
                })

    # Sort by coverage (highest first)
    matches.sort(key=lambda x: x['coverage'], reverse=True)

    # Take top matches (limited by max_lines)
    return matches[:max_lines]


def match_parallel_circuit_path_based(matching_results, jao_gdf, network_gdf, G):
    """
    Match remaining unmatched network lines with parallel circuit JAO lines
    using a path-based approach rather than just geometric proximity.
    This function focuses on creating complete, connected paths.
    """
    print("\n=== MATCHING PARALLEL CIRCUIT JAO LINES USING PATH-BASED APPROACH ===")

    # 1. Identify all unmatched network lines
    used_network_ids = set()
    for result in matching_results:
        if result['matched'] and 'network_ids' in result:
            for network_id in result['network_ids']:
                used_network_ids.add(str(network_id))

    unmatched_network_lines = []
    for idx, row in network_gdf.iterrows():
        network_id = str(row['id'])
        if network_id not in used_network_ids:
            unmatched_network_lines.append({
                'id': network_id,
                'idx': idx,
                'geometry': row.geometry,
                'voltage': int(row['v_nom']),
                'length': calculate_length_meters(row.geometry)
            })

    print(f"Found {len(unmatched_network_lines)} unmatched network lines")

    # 2. Identify all parallel circuit JAO lines (including those already matched)
    parallel_jaos = []
    for idx, row in jao_gdf.iterrows():
        jao_id = str(row['id'])
        is_parallel = False

        # Check if this JAO is already marked as a parallel circuit in results
        for result in matching_results:
            if result['jao_id'] == jao_id and result.get('is_parallel_circuit', False):
                is_parallel = True
                break

        if is_parallel:
            parallel_jaos.append({
                'id': jao_id,
                'geometry': row.geometry,
                'voltage': int(row['v_nom']),
                'name': str(row['NE_name']),
                'length': calculate_length_meters(row.geometry)
            })

    print(f"Found {len(parallel_jaos)} parallel circuit JAO lines to enhance")

    # 3. For each parallel JAO, find potential path completions using unmatched network lines
    enhanced_matches = []
    newly_used_network_ids = set()

    for jao in parallel_jaos:
        jao_id = jao['id']

        # Find the existing match for this JAO
        existing_match = None
        for result in matching_results:
            if result['jao_id'] == jao_id and result['matched']:
                existing_match = result
                break

        if not existing_match:
            print(f"Warning: No existing match found for parallel JAO {jao_id}")
            continue

        # Get current network lines for this JAO
        current_network_ids = existing_match.get('network_ids', [])

        # Get geometries of current network lines
        current_geometries = []
        for network_id in current_network_ids:
            network_rows = network_gdf[network_gdf['id'].astype(str) == network_id]
            if not network_rows.empty:
                current_geometries.append(network_rows.iloc[0].geometry)

        # Skip if no current geometries
        if not current_geometries:
            continue

        # Try to create a merged path from current geometries
        from shapely.geometry import MultiLineString
        from shapely.ops import linemerge

        try:
            current_multi = MultiLineString(current_geometries)
            current_merged = linemerge(current_multi)

            # Create buffer around JAO geometry
            avg_lat = jao['geometry'].centroid.y
            meters_per_degree = 111111 * np.cos(np.radians(abs(avg_lat)))
            buffer_deg = 1000 / meters_per_degree  # 1km buffer
            jao_buffer = jao['geometry'].buffer(buffer_deg)

            # Find potential network lines that could complete the path
            candidate_extensions = []

            for line in unmatched_network_lines:
                # Skip if already used in this function
                if line['id'] in newly_used_network_ids:
                    continue

                # Check voltage compatibility
                voltage_match = False
                if (jao['voltage'] == 220 and line['voltage'] == 220) or \
                        (jao['voltage'] in [380, 400] and line['voltage'] in [380, 400]):
                    voltage_match = True

                if not voltage_match:
                    continue

                # Check if line is within buffer and not already covered
                if jao_buffer.intersects(line['geometry']):
                    # Check if this line connects to existing path
                    # This is a more sophisticated path-based approach

                    # For a merged path (LineString)
                    if current_merged.geom_type == 'LineString':
                        # Get endpoints of current path
                        current_endpoints = [
                            Point(current_merged.coords[0]),
                            Point(current_merged.coords[-1])
                        ]

                        # Get endpoints of candidate line
                        line_endpoints = [
                            Point(line['geometry'].coords[0]),
                            Point(line['geometry'].coords[-1])
                        ]

                        # Check if any endpoints are close to each other
                        min_distance = float('inf')
                        for curr_pt in current_endpoints:
                            for line_pt in line_endpoints:
                                dist = curr_pt.distance(line_pt)
                                # Convert to meters
                                dist_meters = dist * meters_per_degree
                                min_distance = min(min_distance, dist_meters)

                        # If endpoints are close, this line likely connects to path
                        if min_distance < 500:  # 500m threshold
                            # Calculate the coverage - how much of this line is within JAO buffer
                            intersection = line['geometry'].intersection(jao_buffer)
                            coverage = intersection.length / line['geometry'].length if line[
                                                                                            'geometry'].length > 0 else 0

                            if coverage >= 0.5:  # At least 50% of line is within buffer
                                candidate_extensions.append({
                                    'line': line,
                                    'endpoint_distance': min_distance,
                                    'coverage': coverage
                                })

                    # For a collection of lines (MultiLineString)
                    else:
                        # Check if candidate line connects to any line in the collection
                        connected = False
                        for geom in current_geometries:
                            # Get endpoints
                            curr_endpoints = [Point(geom.coords[0]), Point(geom.coords[-1])]
                            line_endpoints = [Point(line['geometry'].coords[0]), Point(line['geometry'].coords[-1])]

                            # Check proximity
                            for curr_pt in curr_endpoints:
                                for line_pt in line_endpoints:
                                    dist = curr_pt.distance(line_pt)
                                    dist_meters = dist * meters_per_degree
                                    if dist_meters < 500:  # 500m connection threshold
                                        connected = True
                                        break

                        if connected:
                            # Calculate coverage
                            intersection = line['geometry'].intersection(jao_buffer)
                            coverage = intersection.length / line['geometry'].length if line[
                                                                                            'geometry'].length > 0 else 0

                            if coverage >= 0.5:
                                candidate_extensions.append({
                                    'line': line,
                                    'endpoint_distance': 0,  # We know it's connected
                                    'coverage': coverage
                                })

            # Sort candidates by coverage (highest first)
            candidate_extensions.sort(key=lambda x: x['coverage'], reverse=True)

            # Select up to 3 best candidates
            selected_extensions = candidate_extensions[:3]

            if selected_extensions:
                # Add these network lines to the match
                new_network_ids = current_network_ids.copy()
                additional_length = 0

                for ext in selected_extensions:
                    new_network_ids.append(ext['line']['id'])
                    additional_length += ext['line']['length']
                    newly_used_network_ids.add(ext['line']['id'])

                # Create updated match
                updated_match = existing_match.copy()
                updated_match['network_ids'] = new_network_ids
                updated_match['path_length'] = float(existing_match.get('path_length', 0) + additional_length)

                # Update length ratio
                if 'jao_length' in updated_match and updated_match['jao_length'] > 0:
                    updated_match['length_ratio'] = float(updated_match['path_length'] / updated_match['jao_length'])

                # Add to enhanced matches
                enhanced_matches.append(updated_match)

                print(f"Enhanced match for JAO {jao_id} by adding {len(selected_extensions)} network lines")
                print(f"  Added network lines: {[ext['line']['id'] for ext in selected_extensions]}")
                print(f"  New path length: {updated_match['path_length'] / 1000:.2f} km")
                if 'length_ratio' in updated_match:
                    print(f"  New length ratio: {updated_match['length_ratio']:.2f}")
            else:
                # No enhancements made, keep original match
                enhanced_matches.append(existing_match)

        except Exception as e:
            print(f"Error processing JAO {jao_id}: {e}")
            # Keep original match in case of error
            if existing_match:
                enhanced_matches.append(existing_match)

    # 4. Update the matching results
    updated_results = []
    enhanced_jao_ids = set(match['jao_id'] for match in enhanced_matches)

    # Keep results for non-enhanced JAOs
    for result in matching_results:
        if result['jao_id'] not in enhanced_jao_ids:
            updated_results.append(result)

    # Add enhanced matches
    updated_results.extend(enhanced_matches)

    print(f"Enhanced {len(enhanced_matches)} parallel circuit JAO matches")
    print(f"Added {len(newly_used_network_ids)} previously unmatched network lines")

    return updated_results

# Helper functions for parallel and match scoring
def calculate_parallel_score(geom1, geom2, buffer_meters=300):
    """Calculate how parallel two geometries are to each other."""
    try:
        # Calculate buffer based on latitude
        avg_lat = (geom1.centroid.y + geom2.centroid.y) / 2
        meters_per_degree = 111111 * np.cos(np.radians(abs(avg_lat)))
        buffer_deg = buffer_meters / meters_per_degree

        # Create buffers
        buffer1 = geom1.buffer(buffer_deg)
        buffer2 = geom2.buffer(buffer_deg)

        # Calculate mutual overlap
        overlap1 = geom1.intersection(buffer2)
        overlap2 = geom2.intersection(buffer1)

        # Calculate overlap ratios
        ratio1 = overlap1.length / geom1.length if geom1.length > 0 else 0
        ratio2 = overlap2.length / geom2.length if geom2.length > 0 else 0

        # Average ratio (higher is better)
        avg_ratio = (ratio1 + ratio2) / 2

        # Calculate Hausdorff distance (lower is better)
        hausdorff_dist = geom1.hausdorff_distance(geom2)
        hausdorff_meters = hausdorff_dist * meters_per_degree

        # Normalize Hausdorff (1 when distance is 0, 0 when distance is large)
        norm_hausdorff = max(0, 1 - (hausdorff_meters / 1000))

        # Combined score (weighted average)
        score = 0.7 * avg_ratio + 0.3 * norm_hausdorff

        return score
    except Exception as e:
        print(f"Error calculating parallel score: {e}")
        return 0


def calculate_match_score(jao_geom, network_geom, jao_length, network_length):
    """Calculate match score between JAO and network geometries."""
    try:
        # Calculate parallelism score
        parallel_score = calculate_parallel_score(jao_geom, network_geom)

        # Calculate length ratio score (1 when equal, 0 when very different)
        length_ratio = network_length / jao_length if jao_length > 0 else float('inf')
        length_ratio_score = max(0, 1 - abs(length_ratio - 1))

        # Combined score (weighted)
        score = 0.7 * parallel_score + 0.3 * length_ratio_score

        return score
    except Exception as e:
        print(f"Error calculating match score: {e}")
        return 0

def match_remaining_lines_by_geometry(jao_gdf, network_gdf, matching_results, buffer_distance=0.005,
                                      snap_tolerance=300, angle_tolerance=30, min_dir_cos=0.866,
                                      min_length_ratio=0.3, max_length_ratio=3):
    """
    Apply a sophisticated geometric matching approach for JAO lines that aren't matched yet.
    Uses a combination of direction cosine, endpoint proximity, and overlap metrics.

    Parameters:
    - jao_gdf: GeoDataFrame with JAO lines
    - network_gdf: GeoDataFrame with network lines
    - matching_results: Existing matching results
    - buffer_distance: Buffer distance in degrees (for overlap)
    - snap_tolerance: Maximum distance in meters to connect endpoints
    - angle_tolerance: Maximum angle difference in degrees to consider lines aligned
    - min_dir_cos: Minimum direction cosine (cos of angle between lines)
    - min_length_ratio: Minimum acceptable ratio of network/jao length (default 0.3)
    - max_length_ratio: Maximum acceptable ratio of network/jao length (default 1.7)
    """
    import numpy as np
    from shapely.geometry import LineString, Point, MultiLineString
    from shapely.ops import linemerge, unary_union
    from scipy.spatial import cKDTree

    print("\nAttempting additional matches using advanced geometric approach...")
    print(f"Length ratio constraints: {min_length_ratio:.1f} to {max_length_ratio:.1f}")

    # Identify unmatched JAO lines
    matched_jao_ids = set(result['jao_id'] for result in matching_results if result['matched'])

    # Identify matched network lines
    matched_network_ids = set()
    for result in matching_results:
        if result['matched'] and 'network_ids' in result:
            for network_id in result['network_ids']:
                matched_network_ids.add(str(network_id))

    # Get unmatched JAO lines
    unmatched_jao_rows = []
    for idx, row in jao_gdf.iterrows():
        jao_id = str(row['id'])
        if jao_id not in matched_jao_ids:
            unmatched_jao_rows.append((idx, row))

    # Get unmatched network lines
    unmatched_network_gdf = network_gdf[~network_gdf['id'].astype(str).isin(matched_network_ids)].copy()

    print(f"Found {len(unmatched_jao_rows)} unmatched JAO lines to process")
    print(f"Found {len(unmatched_network_gdf)} unmatched network lines for matching")

    # Define helper functions
    def same_voltage(a, b):
        """Check if voltages are considered the same."""
        if a == 0 or b == 0:  # If either is unknown/zero, don't compare
            return True

        return (abs(a - b) <= 5 or  # 225 kV ≈ 220 kV, etc.
                abs(a / b - 1.0) <= 0.1 or  # Within 10% of each other
                {a, b} == {380, 400})  # explicit 380/400 rule

    def main_vec(ls):
        """Get the main vector of a LineString (direction from first to last point)."""
        if ls.geom_type == "MultiLineString":
            ls = max(ls.geoms, key=lambda p: p.length)
        c = np.asarray(ls.coords)
        v = c[-1] - c[0]
        n = np.linalg.norm(v)
        return v / n if n else v

    def dir_cos(a, b):
        """Calculate the direction cosine between two LineStrings."""
        return float(abs(np.dot(main_vec(a), main_vec(b))))

    def endpts(g):
        """Get endpoints of a geometry."""
        if g is None or g.is_empty:
            return []
        if g.geom_type == "LineString":
            c = list(g.coords)
            return [Point(c[0]), Point(c[-1])]
        out = []
        for part in g.geoms:
            c = list(part.coords)
            out += [Point(c[0]), Point(c[-1])]
        return out

    def endpts_inside(src, cand, width):
        """Check if candidate endpoints are inside buffer of source."""
        corr = src.buffer(width)
        return all(corr.contains(pt) for pt in endpts(cand))

    def overlap_length(a, b, tol=0.001):
        """Calculate the length of overlap between two geometries."""
        inter = b.intersection(a.buffer(tol))
        return inter.length

    def hausdorff_distance(a, b):
        """Calculate Hausdorff distance between two geometries."""
        return a.hausdorff_distance(b)

    # Function to calculate approximate meters per degree at a given latitude
    def meters_per_degree_at_lat(lat):
        return 111111 * np.cos(np.radians(lat))

    # Process each unmatched JAO line
    additional_matches = 0
    new_matches = []

    # Store matched network lines to prevent reuse across different JAO lines
    newly_matched_network_ids = set()

    # For each unmatched JAO line
    for idx, row in unmatched_jao_rows:
        jao_id = str(row['id'])
        jao_name = str(row['NE_name'])
        jao_voltage = int(row['v_nom'])
        jao_geometry = row.geometry

        # Calculate jao length in meters (approximate)
        meters_per_degree = meters_per_degree_at_lat(jao_geometry.centroid.y)
        jao_length_m = float(jao_geometry.length) * meters_per_degree

        print(f"\nProcessing unmatched JAO line {jao_id} ({jao_name}), length: {jao_length_m / 1000:.2f} km")

        # Find geometrically matching network lines
        candidate_matches = []

        # Get network lines with similar direction
        for net_idx, net_row in unmatched_network_gdf.iterrows():
            net_id = str(net_row['id'])

            # Skip already matched network lines (in this round)
            if net_id in newly_matched_network_ids:
                continue

            net_voltage = int(net_row['v_nom'])
            net_geometry = net_row.geometry

            # Check voltage matching - used as a multiplier for the score
            voltage_match = same_voltage(jao_voltage, net_voltage)
            voltage_factor = 1.0 if voltage_match else 0.5

            # Calculate direction cosine (alignment)
            cosine = dir_cos(jao_geometry, net_geometry)

            # Skip if lines are not remotely aligned
            if cosine < min_dir_cos:
                continue

            # Check proximity between endpoints
            jao_endpoints = endpts(jao_geometry)
            net_endpoints = endpts(net_geometry)

            min_endpoint_distance = float('inf')
            for d_pt in jao_endpoints:
                for n_pt in net_endpoints:
                    dist = d_pt.distance(n_pt)
                    min_endpoint_distance = min(min_endpoint_distance, dist)

            # Convert min_endpoint_distance to meters (approximate)
            # This is a rough conversion and depends on latitude
            min_endpoint_distance_m = min_endpoint_distance * meters_per_degree

            # Calculate overlap
            overlap = overlap_length(jao_geometry, net_geometry, tol=buffer_distance)
            overlap_ratio = overlap / min(jao_geometry.length, net_geometry.length)

            # Calculate Hausdorff distance
            h_dist = hausdorff_distance(jao_geometry, net_geometry)
            h_dist_m = h_dist * meters_per_degree

            # Calculate a composite score
            # Higher is better
            endpoint_score = 1.0 if min_endpoint_distance_m <= snap_tolerance else (
                0.5 if min_endpoint_distance_m <= 2 * snap_tolerance else 0.0)

            alignment_score = cosine
            overlap_score = overlap_ratio

            # Hausdorff score - inverse of distance (closer is better)
            max_h_dist = 1000  # meters
            hausdorff_score = max(0, 1.0 - (h_dist_m / max_h_dist))

            # Calculate combined score
            combined_score = (
                                     0.3 * endpoint_score +
                                     0.3 * alignment_score +
                                     0.2 * overlap_score +
                                     0.2 * hausdorff_score
                             ) * voltage_factor

            # Calculate network line length
            if 'length' in net_row and net_row['length']:
                net_length_m = float(net_row['length']) * 1000  # Assuming length is in km
            else:
                net_length_m = float(net_row.geometry.length) * meters_per_degree

            # Only consider if score is reasonable
            if combined_score > 0.5:
                candidate_matches.append({
                    'network_id': net_id,
                    'score': combined_score,
                    'voltage_match': voltage_match,
                    'dir_cos': cosine,
                    'endpoint_dist_m': min_endpoint_distance_m,
                    'overlap_ratio': overlap_ratio,
                    'hausdorff_dist_m': h_dist_m,
                    'length_m': net_length_m,
                    'idx': net_idx
                })

        # Sort by score (highest first)
        candidate_matches.sort(key=lambda x: x['score'], reverse=True)

        if candidate_matches:
            # Find best combinations of network lines
            best_combination = None
            best_combination_score = 0
            best_length_ratio = float('inf')  # Track how close to 1.0 the length ratio is

            # Try different combinations of top candidates (up to 5 at a time)
            # Start with 1 line, then try 2, etc.
            max_candidates = min(10, len(candidate_matches))

            for combo_size in range(1, min(4, max_candidates) + 1):
                # Generate all combinations of the specified size
                from itertools import combinations
                for combo in combinations(candidate_matches[:max_candidates], combo_size):
                    # Calculate total length and average score
                    combo_length = sum(c['length_m'] for c in combo)
                    length_ratio = combo_length / jao_length_m if jao_length_m > 0 else float('inf')
                    avg_score = sum(c['score'] for c in combo) / len(combo)

                    # Check if length ratio is within acceptable range
                    if min_length_ratio <= length_ratio <= max_length_ratio:
                        # Calculate how far from ideal (1.0) the ratio is
                        ratio_distance = abs(length_ratio - 1.0)

                        # Combine score and ratio_distance into a final score
                        # Weight more towards matching score, but consider length ratio
                        combo_score = avg_score * (1.0 - 0.3 * ratio_distance)

                        # Update best if this is better
                        if combo_score > best_combination_score:
                            best_combination = combo
                            best_combination_score = combo_score
                            best_length_ratio = length_ratio

            # If we found a valid combination
            if best_combination:
                best_matches = list(best_combination)
                network_ids = [match['network_id'] for match in best_matches]

                print(f"  Found geometric match with network lines: {network_ids}")

                # Fix the problematic f-strings with list comprehensions
                scores = [f"{match['score']:.2f}" for match in best_matches]
                dir_cosines = [f"{match['dir_cos']:.2f}" for match in best_matches]
                endpoint_dists = [f"{match['endpoint_dist_m']:.1f}" for match in best_matches]
                overlap_ratios = [f"{match['overlap_ratio']:.2f}" for match in best_matches]

                print(f"  Match scores: {scores}")
                print(f"  Direction cosines: {dir_cosines}")
                print(f"  Endpoint distances (m): {endpoint_dists}")
                print(f"  Overlap ratios: {overlap_ratios}")
                print(f"  Length ratio (network/jao): {best_length_ratio:.2f}")

                # Calculate total path length
                path_length = sum(match['length_m'] for match in best_matches)

                # Create a result for this match
                match_result = {
                    'jao_id': jao_id,
                    'jao_name': jao_name,
                    'v_nom': jao_voltage,
                    'matched': True,
                    'is_duplicate': False,
                    'is_geometric_match': True,  # Flag to indicate this was matched geometrically
                    'path': [],  # No path in graph
                    'network_ids': network_ids,
                    'path_length': float(path_length),
                    'jao_length': float(jao_length_m),
                    'length_ratio': float(best_length_ratio),
                    'match_quality': 'Geometric Match' + (
                        ' (voltage mismatch)' if not best_matches[0]['voltage_match'] else ''),
                    'geometric_match_details': {
                        'scores': [match['score'] for match in best_matches],
                        'dir_cosines': [match['dir_cos'] for match in best_matches],
                        'endpoint_dists': [match['endpoint_dist_m'] for match in best_matches],
                        'overlap_ratios': [match['overlap_ratio'] for match in best_matches]
                    }
                }

                new_matches.append(match_result)
                additional_matches += 1

                # Mark these network lines as matched
                for network_id in network_ids:
                    newly_matched_network_ids.add(network_id)
            else:
                print(
                    f"  No combinations found with acceptable length ratio {min_length_ratio:.1f}x to {max_length_ratio:.1f}x")
        else:
            print(f"  No geometric matches found for {jao_id}")

    print(f"\nFound {additional_matches} additional matches using advanced geometric approach")

    # Add new matches to results
    matching_results.extend(new_matches)

    return matching_results

def find_network_line_usage(matching_results):
    """Analyze how many times each network line is used across all matches."""
    network_line_usage = {}

    for result in matching_results:
        if result['matched'] and 'network_ids' in result:
            for network_id in result['network_ids']:
                network_line_usage[network_id] = network_line_usage.get(network_id, 0) + 1

    # Find heavily reused lines (used more than once)
    reused_lines = {line_id: count for line_id, count in network_line_usage.items() if count > 1}

    print(f"\nFound {len(reused_lines)} network lines that are used multiple times:")
    for line_id, count in sorted(reused_lines.items(), key=lambda x: x[1], reverse=True)[:20]:  # Top 20 most reused
        print(f"  Network line {line_id} used {count} times")

    return network_line_usage, reused_lines


def match_parallel_circuit_jao_with_network(matching_results, jao_gdf, network_gdf, G, nearest_points_dict):
    """
    Match parallel circuit JAO lines with dedicated network lines.
    Focus on matching each parallel JAO with its own set of network lines,
    particularly targeting unmatched network lines.
    """
    print("\nMatching parallel circuit JAO lines with dedicated network lines...")

    # PART 1: Handle special cases first
    special_case_matches = []
    special_case = {
        "97": ["Line_8160", "Line_30733", "Line_30181", "Line_17856"]
    }

    for jao_id, network_ids in special_case.items():
        jao_rows = jao_gdf[jao_gdf['id'].astype(str) == jao_id]
        if not jao_rows.empty:
            jao_row = jao_rows.iloc[0]

            special_match = {
                'jao_id': jao_id,
                'jao_name': str(jao_row['NE_name']),
                'v_nom': int(jao_row['v_nom']),
                'matched': True,
                'is_duplicate': False,
                'is_parallel_circuit': True,
                'path': [],
                'network_ids': network_ids,
                'jao_length': float(calculate_length_meters(jao_row.geometry)),
                'match_quality': f'Parallel Circuit ({jao_row["v_nom"]} kV) - Special Case'
            }

            # Calculate path length
            path_length = 0
            for network_id in network_ids:
                network_rows = network_gdf[network_gdf['id'].astype(str) == network_id]
                if not network_rows.empty:
                    path_length += calculate_length_meters(network_rows.iloc[0].geometry)

            special_match['path_length'] = float(path_length)
            special_match['length_ratio'] = float(path_length / special_match['jao_length']) if special_match[
                                                                                                    'jao_length'] > 0 else 1.0

            special_case_matches.append(special_match)
            print(f"  Created special match for JAO {jao_id} with network lines: {network_ids}")

    # PART 2: Identify all JAO lines that are part of parallel circuits

    # First, organize JAO lines by voltage and location
    jao_by_voltage = {}
    for idx, row in jao_gdf.iterrows():
        jao_id = str(row['id'])
        voltage = int(row['v_nom'])

        if voltage not in jao_by_voltage:
            jao_by_voltage[voltage] = []

        jao_by_voltage[voltage].append({
            'id': jao_id,
            'idx': idx,
            'name': str(row['NE_name']),
            'geometry': row.geometry,
            'length': calculate_length_meters(row.geometry)
        })

    # Identify parallel circuit groups
    parallel_groups = []
    processed_jaos = set()  # Track which JAOs have been assigned to groups

    # Add JAOs from special cases to processed set
    for match in special_case_matches:
        processed_jaos.add(match['jao_id'])

    # For each voltage level, find parallel circuit groups
    for voltage, jaos in jao_by_voltage.items():
        for i, jao1 in enumerate(jaos):
            if jao1['id'] in processed_jaos:
                continue

            # Start a new parallel group with this JAO
            current_group = [jao1]
            processed_jaos.add(jao1['id'])

            # Find all other JAOs that are parallel to this one
            for j, jao2 in enumerate(jaos):
                if i == j or jao2['id'] in processed_jaos:
                    continue

                # Check if JAO2 is parallel to JAO1
                if is_geometry_parallel(jao1['geometry'], jao2['geometry'],
                                        buffer_meters=800, min_coverage=0.5):
                    current_group.append(jao2)
                    processed_jaos.add(jao2['id'])

            # Only add groups with multiple JAOs (parallel circuits)
            if len(current_group) > 1:
                parallel_groups.append({
                    'voltage': voltage,
                    'jaos': current_group,
                    'avg_length': sum(jao['length'] for jao in current_group) / len(current_group)
                })

    print(f"Identified {len(parallel_groups)} parallel circuit groups")
    for i, group in enumerate(parallel_groups):
        jao_ids = [jao['id'] for jao in group['jaos']]
        print(f"  Group {i + 1}: {len(jao_ids)} JAOs at {group['voltage']} kV - {jao_ids}")

    # PART 3: Identify which JAOs in these groups are already matched

    # Create a lookup of matched JAOs and their network lines
    matched_jaos = {}
    for result in matching_results:
        if result['matched']:
            matched_jaos[result['jao_id']] = {
                'network_ids': result.get('network_ids', []),
                'is_parallel_circuit': result.get('is_parallel_circuit', False),
                'is_duplicate': result.get('is_duplicate', False)
            }

    # Track all network lines used in matched results
    used_network_ids = set()
    for result in matching_results:
        if result['matched'] and 'network_ids' in result:
            for network_id in result['network_ids']:
                used_network_ids.add(str(network_id))

    # Add special case network IDs to used set
    for match in special_case_matches:
        for network_id in match.get('network_ids', []):
            used_network_ids.add(str(network_id))

    # Get all unmatched network lines
    unmatched_network_lines = []
    for idx, row in network_gdf.iterrows():
        network_id = str(row['id'])
        if network_id not in used_network_ids:
            unmatched_network_lines.append({
                'id': network_id,
                'idx': idx,
                'geometry': row.geometry,
                'voltage': int(row['v_nom']),
                'length': calculate_length_meters(row.geometry)
            })

    print(f"Found {len(unmatched_network_lines)} unmatched network lines")

    # PART 4: For each parallel group, match each JAO with its own set of network lines

    parallel_matches = []
    newly_used_network_ids = set()  # Track network lines we use in this function

    for group_idx, group in enumerate(parallel_groups):
        print(f"\nProcessing parallel circuit group {group_idx + 1} ({group['voltage']} kV)")

        # Sort JAOs by whether they're already matched (matched first)
        group['jaos'].sort(key=lambda x: x['id'] in matched_jaos, reverse=True)

        # Track network lines used by this group
        group_network_lines = set()

        # First, collect all network lines already used by any JAO in this group
        for jao in group['jaos']:
            if jao['id'] in matched_jaos:
                for network_id in matched_jaos[jao['id']]['network_ids']:
                    group_network_lines.add(network_id)

        # For each JAO in this group
        for jao_idx, jao in enumerate(group['jaos']):
            jao_id = jao['id']
            jao_name = jao['name']

            print(f"  Processing JAO {jao_id} ({jao_name})")

            # Skip if this JAO is already matched as a parallel circuit or duplicate
            if jao_id in matched_jaos and (
                    matched_jaos[jao_id]['is_parallel_circuit'] or matched_jaos[jao_id]['is_duplicate']):
                print(f"    Already matched as a parallel circuit or duplicate, skipping")
                continue

            # If this JAO is already matched with regular matching, we'll keep that match
            if jao_id in matched_jaos and not matched_jaos[jao_id]['is_parallel_circuit'] and not matched_jaos[jao_id][
                'is_duplicate']:
                print(f"    Already matched as a regular match, keeping existing match")
                continue

            # For unmatched JAOs in this group, find matching network lines
            # Prioritize unmatched network lines
            potential_network_lines = []

            # First check unmatched network lines
            for line in unmatched_network_lines:
                # Skip if already used in this function
                if line['id'] in newly_used_network_ids:
                    continue

                # Check voltage compatibility (very relaxed)
                voltage_match = False
                if (group['voltage'] == 220 and line['voltage'] == 220) or \
                        (group['voltage'] == 400 and line['voltage'] in [380, 400]) or \
                        (group['voltage'] == 380 and line['voltage'] in [380, 400]):
                    voltage_match = True

                if not voltage_match:
                    continue

                # Check geometric proximity
                coverage = calculate_geometry_coverage(jao['geometry'], line['geometry'],
                                                       buffer_meters=1500)

                if coverage >= 0.3:  # Very relaxed coverage requirement
                    potential_network_lines.append({
                        'line': line,
                        'coverage': coverage,
                        'is_unmatched': True
                    })

            # If we don't have enough unmatched lines, also consider matched lines
            # from other groups (not used by this parallel group)
            if len(potential_network_lines) < 2:
                for idx, row in network_gdf.iterrows():
                    network_id = str(row['id'])

                    # Skip if used in this parallel group or already in our potential lines
                    if network_id in group_network_lines or network_id in newly_used_network_ids or \
                            any(p['line']['id'] == network_id for p in potential_network_lines):
                        continue

                    # Check voltage
                    network_voltage = int(row['v_nom'])
                    voltage_match = False
                    if (group['voltage'] == 220 and network_voltage == 220) or \
                            (group['voltage'] == 400 and network_voltage in [380, 400]) or \
                            (group['voltage'] == 380 and network_voltage in [380, 400]):
                        voltage_match = True

                    if not voltage_match:
                        continue

                    # Check geometric proximity
                    coverage = calculate_geometry_coverage(jao['geometry'], row.geometry,
                                                           buffer_meters=1500)

                    if coverage >= 0.3:  # Very relaxed coverage requirement
                        line_info = {
                            'id': network_id,
                            'idx': idx,
                            'geometry': row.geometry,
                            'voltage': network_voltage,
                            'length': calculate_length_meters(row.geometry)
                        }
                        potential_network_lines.append({
                            'line': line_info,
                            'coverage': coverage,
                            'is_unmatched': False
                        })

            # Sort potential lines by coverage
            potential_network_lines.sort(key=lambda x: x['coverage'], reverse=True)

            # Take top matches (preferring unmatched lines first)
            if potential_network_lines:
                # First take all unmatched lines with good coverage
                unmatched_candidates = [p for p in potential_network_lines if
                                        p['is_unmatched'] and p['coverage'] >= 0.5]

                # If we don't have at least 1 good unmatched candidate, take any unmatched
                if not unmatched_candidates:
                    unmatched_candidates = [p for p in potential_network_lines if p['is_unmatched']]

                # Take up to 4 best candidates, prioritizing unmatched lines
                selected_candidates = unmatched_candidates[:min(4, len(unmatched_candidates))]

                # If we need more, add matched lines
                if len(selected_candidates) < 4:
                    matched_candidates = [p for p in potential_network_lines if not p['is_unmatched']]
                    selected_candidates.extend(
                        matched_candidates[:min(4 - len(selected_candidates), len(matched_candidates))])

                # Extract network IDs and calculate path length
                network_ids = [candidate['line']['id'] for candidate in selected_candidates]
                path_length = sum(candidate['line']['length'] for candidate in selected_candidates)

                # Calculate length ratio
                length_ratio = path_length / jao['length'] if jao['length'] > 0 else float('inf')

                print(f"    Matched with network lines: {network_ids}")
                print(f"    Path length: {path_length / 1000:.2f} km, JAO length: {jao['length'] / 1000:.2f} km")
                print(f"    Length ratio: {length_ratio:.2f}")

                # Create match result
                match_result = {
                    'jao_id': jao_id,
                    'jao_name': jao_name,
                    'v_nom': group['voltage'],
                    'matched': True,
                    'is_duplicate': False,
                    'is_parallel_circuit': True,
                    'path': [],  # No path for direct geometric matching
                    'network_ids': network_ids,
                    'path_length': float(path_length),
                    'jao_length': float(jao['length']),
                    'length_ratio': float(length_ratio),
                    'match_quality': f'Parallel Circuit ({group["voltage"]} kV)'
                }

                parallel_matches.append(match_result)

                # Mark these network lines as used
                for network_id in network_ids:
                    newly_used_network_ids.add(network_id)
                    group_network_lines.add(network_id)
            else:
                print(f"    No matching network lines found for JAO {jao_id}")

    # PART 5: Combine all matches
    all_matches = special_case_matches + parallel_matches

    # Remove any existing matches for these JAOs
    jao_ids_to_match = set(match['jao_id'] for match in all_matches)
    filtered_results = [r for r in matching_results if r['jao_id'] not in jao_ids_to_match]

    # Add new parallel matches
    filtered_results.extend(all_matches)

    # Count match statistics
    print(f"\nParallel circuit matching summary:")
    print(f"  {len(special_case_matches)} special case matches")
    print(f"  {len(parallel_matches)} general parallel circuit matches")
    print(f"  Total: {len(all_matches)} parallel circuit matches")
    print(f"  Network lines used: {len(newly_used_network_ids)}")

    return filtered_results

def match_parallel_voltage_circuits(jao_gdf, network_gdf, matching_results):
    """
    Find and match parallel circuits where different voltage lines follow the same path.
    This handles cases where a 220kV and 400kV line follow the same path but only one gets matched.
    """
    print("\nLooking for parallel voltage circuits with same geometry...")

    # Identify matched JAO lines
    matched_jaos = {result['jao_id']: result for result in matching_results if
                    result['matched'] and not result.get('is_duplicate', False)}

    # Identify unmatched JAO lines
    unmatched_jao_ids = set(str(row['id']) for _, row in jao_gdf.iterrows()) - set(matched_jaos.keys())
    unmatched_jaos = jao_gdf[jao_gdf['id'].astype(str).isin(unmatched_jao_ids)]

    new_matches = []
    parallel_count = 0

    # For each unmatched JAO line
    for idx, unmatched_row in unmatched_jaos.iterrows():
        unmatched_id = str(unmatched_row['id'])
        unmatched_geom = unmatched_row.geometry
        unmatched_voltage = int(unmatched_row['v_nom'])
        unmatched_name = str(unmatched_row['NE_name'])

        # Compare with each matched JAO line
        best_match = None
        best_similarity = 0

        for matched_id, matched_result in matched_jaos.items():
            # Skip if same line or already used as a match
            if matched_id == unmatched_id:
                continue

            # Get the matched JAO row
            matched_rows = jao_gdf[jao_gdf['id'].astype(str) == matched_id]
            if matched_rows.empty:
                continue

            matched_row = matched_rows.iloc[0]
            matched_geom = matched_row.geometry
            matched_voltage = int(matched_row['v_nom'])

            # Skip if same voltage (we're looking for parallel circuits with different voltages)
            if matched_voltage == unmatched_voltage:
                continue

            # Compute similarity between geometries
            # Use Hausdorff distance as a similarity measure (lower is better)
            try:
                hausdorff_dist = matched_geom.hausdorff_distance(unmatched_geom)

                # Convert to approximate meters
                avg_lat = (matched_geom.centroid.y + unmatched_geom.centroid.y) / 2
                meters_per_degree = 111111 * np.cos(np.radians(abs(avg_lat)))
                hausdorff_meters = hausdorff_dist * meters_per_degree

                # Check if geometries are very similar (hausdorff distance < 500m)
                if hausdorff_meters < 500:
                    # Calculate similarity as inverse of distance (higher is better)
                    similarity = 1.0 / (1.0 + hausdorff_meters / 100)

                    # Update best match if this is better
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match = matched_result
            except Exception as e:
                print(f"Error computing similarity: {e}")
                continue

        # If we found a good match, create a parallel circuit match
        if best_match and best_similarity > 0.5:  # Threshold for good similarity
            # Get the matched network lines to reuse for this parallel circuit
            network_ids = best_match.get('network_ids', [])

            if network_ids:
                print(f"Found parallel circuit: JAO {unmatched_id} ({unmatched_voltage} kV) parallel to JAO {best_match['jao_id']} ({best_match.get('v_nom', '?')} kV)")
                print(f"  Using network lines: {network_ids}")

                # Create a new match for the parallel circuit
                new_match = {
                    'jao_id': unmatched_id,
                    'jao_name': unmatched_name,
                    'v_nom': unmatched_voltage,
                    'matched': True,
                    'is_duplicate': False,
                    'is_parallel_voltage_circuit': True,  # Mark as parallel voltage circuit
                    'parallel_to_jao': best_match['jao_id'],
                    'path': best_match.get('path', []).copy() if 'path' in best_match else [],
                    'network_ids': network_ids.copy(),
                    'path_length': best_match.get('path_length', 0),
                    'jao_length': calculate_length_meters(unmatched_geom),
                    'length_ratio': best_match.get('length_ratio', 1.0),
                    'match_quality': f'Parallel Voltage Circuit ({unmatched_voltage} kV)'
                }

                new_matches.append(new_match)
                parallel_count += 1

    print(f"Found and matched {parallel_count} parallel voltage circuits")

    # Add new matches to results
    matching_results.extend(new_matches)

    return matching_results
# Additional helper function
def calculate_geometry_coverage(geom1, geom2, buffer_meters=1000):
    """
    Calculate how much of geom2 is covered by a buffer around geom1.

    Parameters:
    - geom1: First geometry (creating the buffer)
    - geom2: Second geometry (checking coverage)
    - buffer_meters: Buffer size in meters

    Returns:
    - Coverage ratio (0.0 to 1.0)
    """
    try:
        # Calculate buffer size in degrees based on latitude
        avg_lat = (geom1.centroid.y + geom2.centroid.y) / 2
        meters_per_degree = 111111 * np.cos(np.radians(abs(avg_lat)))
        buffer_deg = buffer_meters / meters_per_degree

        # Create buffer around geom1
        buffer = geom1.buffer(buffer_deg)

        # Calculate intersection with geom2
        intersection = geom2.intersection(buffer)

        # Calculate coverage ratio
        coverage = intersection.length / geom2.length if geom2.length > 0 else 0

        return coverage
    except Exception as e:
        print(f"  Error calculating geometry coverage: {e}")
        return 0.0


def improve_matches_with_geometric_pass(matching_results, jao_gdf, network_gdf):
    """
    Perform a second geometric pass to improve matches by adding missing network lines.
    This function will search for unmatched network lines that geometrically align with
    already matched JAO lines, and add them to those matches.
    """
    print("\n=== IMPROVING MATCHES WITH GEOMETRIC PASS ===")

    # 1. Identify all matched JAO lines
    matched_jaos = {result['jao_id']: result for result in matching_results if result['matched']}

    # 2. Identify all network lines that are already used in matches
    used_network_ids = set()
    for result in matching_results:
        if result['matched'] and 'network_ids' in result:
            for network_id in result['network_ids']:
                used_network_ids.add(str(network_id))

    # 3. Get all unmatched network lines
    unmatched_network_lines = []
    for idx, row in network_gdf.iterrows():
        network_id = str(row['id'])
        if network_id not in used_network_ids:
            unmatched_network_lines.append({
                'id': network_id,
                'idx': idx,
                'geometry': row.geometry,
                'voltage': int(row['v_nom']),
                'length': calculate_length_meters(row.geometry)
            })

    print(f"Found {len(unmatched_network_lines)} unmatched network lines to check")

    # 4. For each unmatched network line, check if it aligns with any matched JAO
    improvements_made = 0

    for network_line in unmatched_network_lines:
        network_id = network_line['id']
        network_geom = network_line['geometry']
        network_voltage = network_line['voltage']
        network_length = network_line['length']

        best_match = None
        best_score = 0

        # Check against each matched JAO
        for jao_id, match_result in matched_jaos.items():
            # Skip if not a valid match object
            if 'jao_length' not in match_result:
                continue

            # Get the JAO from the GeoDataFrame
            jao_rows = jao_gdf[jao_gdf['id'].astype(str) == jao_id]
            if jao_rows.empty:
                continue

            jao_row = jao_rows.iloc[0]
            jao_geom = jao_row.geometry
            jao_voltage = int(jao_row['v_nom'])

            # Check voltage compatibility (accounting for 380/400 equivalence)
            voltage_match = False
            if (network_voltage == 220 and jao_voltage == 220) or \
                    (network_voltage in [380, 400] and jao_voltage in [380, 400]):
                voltage_match = True

            if not voltage_match:
                continue

            # Calculate geometric alignment score
            # First, check if network line overlaps with JAO buffer
            coverage = calculate_geometry_coverage(jao_geom, network_geom, buffer_meters=1500)

            # If there's significant overlap
            if coverage >= 0.4:  # 40% coverage threshold
                # Calculate Hausdorff distance for more precise measurement
                hausdorff_dist = jao_geom.hausdorff_distance(network_geom)
                # Convert to meters
                avg_lat = (jao_geom.centroid.y + network_geom.centroid.y) / 2
                meters_per_degree = 111111 * np.cos(np.radians(abs(avg_lat)))
                hausdorff_meters = hausdorff_dist * meters_per_degree

                # Calculate overall score (higher is better)
                # Score combines coverage and inverse of Hausdorff distance
                score = coverage * (1 - min(1, hausdorff_meters / 5000))

                if score > best_score:
                    best_score = score
                    best_match = {
                        'jao_id': jao_id,
                        'coverage': coverage,
                        'hausdorff_meters': hausdorff_meters,
                        'score': score,
                        'match_result': match_result
                    }

        # If we found a good match, add this network line to that JAO's match
        if best_match and best_score >= 0.3:  # Score threshold
            match_result = best_match['match_result']
            jao_id = best_match['jao_id']

            print(f"  Adding network line {network_id} to JAO {jao_id}")
            print(
                f"    Coverage: {best_match['coverage']:.2f}, Hausdorff: {best_match['hausdorff_meters']:.0f}m, Score: {best_score:.3f}")

            # Add network line to match
            if 'network_ids' not in match_result:
                match_result['network_ids'] = []

            match_result['network_ids'].append(network_id)

            # Update path length
            if 'path_length' not in match_result:
                match_result['path_length'] = 0

            match_result['path_length'] = float(match_result['path_length'] + network_length)

            # Update length ratio
            if 'jao_length' in match_result and match_result['jao_length'] > 0:
                match_result['length_ratio'] = float(match_result['path_length'] / match_result['jao_length'])

            # Mark as geometric match if not already marked as something specific
            if not match_result.get('is_duplicate', False) and \
                    not match_result.get('is_parallel_circuit', False) and \
                    not match_result.get('is_parallel_voltage_circuit', False) and \
                    not match_result.get('is_geometric_match', False):
                match_result['is_geometric_match'] = True

                # Update match quality if it was previously poor
                if match_result.get('match_quality', '').startswith('Poor'):
                    match_result['match_quality'] = 'Geometric Match'

            improvements_made += 1

    print(f"Made {improvements_made} improvements to matches")

    return matching_results


def improve_matches_by_similarity(matching_results, jao_gdf, network_gdf):
    """
    Look for similar JAO lines and ensure they use consistent network lines.
    This handles cases where parallel or duplicate JAO lines should use the same network paths.
    """
    print("\n=== IMPROVING MATCHES BY JAO SIMILARITY ===")

    # 1. Identify all matched JAO lines
    matched_results = [r for r in matching_results if r['matched']]

    # 2. Group JAO lines by voltage
    jao_by_voltage = {}
    for result in matched_results:
        voltage = result.get('v_nom', 0)
        if voltage not in jao_by_voltage:
            jao_by_voltage[voltage] = []
        jao_by_voltage[voltage].append(result)

    improvements_made = 0

    # 3. For each voltage group, find similar JAO lines
    for voltage, results in jao_by_voltage.items():
        # Skip if too few lines to compare
        if len(results) <= 1:
            continue

        print(f"Processing {len(results)} JAO lines at {voltage} kV")

        # Build a dictionary of JAO IDs to their geometries
        jao_geometries = {}
        for result in results:
            jao_id = result['jao_id']
            jao_rows = jao_gdf[jao_gdf['id'].astype(str) == jao_id]
            if not jao_rows.empty:
                jao_geometries[jao_id] = jao_rows.iloc[0].geometry

        # Compare each pair of JAO lines
        for i in range(len(results)):
            for j in range(i + 1, len(results)):
                result1 = results[i]
                result2 = results[j]

                jao_id1 = result1['jao_id']
                jao_id2 = result2['jao_id']

                # Skip if either geometry is missing
                if jao_id1 not in jao_geometries or jao_id2 not in jao_geometries:
                    continue

                # Calculate similarity between the JAO lines
                try:
                    geom1 = jao_geometries[jao_id1]
                    geom2 = jao_geometries[jao_id2]

                    # Check if geometries are similar
                    coverage1 = calculate_geometry_coverage(geom1, geom2, buffer_meters=1000)
                    coverage2 = calculate_geometry_coverage(geom2, geom1, buffer_meters=1000)

                    # Average coverage
                    avg_coverage = (coverage1 + coverage2) / 2

                    # If JAO lines are similar, check network lines
                    if avg_coverage >= 0.7:  # High similarity threshold
                        print(f"  Found similar JAO lines: {jao_id1} and {jao_id2} (coverage: {avg_coverage:.2f})")

                        # Get network IDs for each JAO
                        network_ids1 = set(result1.get('network_ids', []))
                        network_ids2 = set(result2.get('network_ids', []))

                        # Find network IDs in one but not the other
                        missing_in_1 = network_ids2 - network_ids1
                        missing_in_2 = network_ids1 - network_ids2

                        # Add missing network IDs to each result
                        if missing_in_1:
                            print(f"    Adding {len(missing_in_1)} network lines to JAO {jao_id1}")

                            # Calculate additional path length
                            additional_length = 0
                            for network_id in missing_in_1:
                                network_rows = network_gdf[network_gdf['id'].astype(str) == network_id]
                                if not network_rows.empty:
                                    additional_length += calculate_length_meters(network_rows.iloc[0].geometry)

                            # Update result1
                            if 'network_ids' not in result1:
                                result1['network_ids'] = []
                            result1['network_ids'].extend(list(missing_in_1))

                            if 'path_length' not in result1:
                                result1['path_length'] = 0
                            result1['path_length'] = float(result1['path_length'] + additional_length)

                            # Update length ratio
                            if 'jao_length' in result1 and result1['jao_length'] > 0:
                                result1['length_ratio'] = float(result1['path_length'] / result1['jao_length'])

                            improvements_made += 1

                        if missing_in_2:
                            print(f"    Adding {len(missing_in_2)} network lines to JAO {jao_id2}")

                            # Calculate additional path length
                            additional_length = 0
                            for network_id in missing_in_2:
                                network_rows = network_gdf[network_gdf['id'].astype(str) == network_id]
                                if not network_rows.empty:
                                    additional_length += calculate_length_meters(network_rows.iloc[0].geometry)

                            # Update result2
                            if 'network_ids' not in result2:
                                result2['network_ids'] = []
                            result2['network_ids'].extend(list(missing_in_2))

                            if 'path_length' not in result2:
                                result2['path_length'] = 0
                            result2['path_length'] = float(result2['path_length'] + additional_length)

                            # Update length ratio
                            if 'jao_length' in result2 and result2['jao_length'] > 0:
                                result2['length_ratio'] = float(result2['path_length'] / result2['jao_length'])

                            improvements_made += 1

                except Exception as e:
                    print(f"  Error comparing JAO {jao_id1} and {jao_id2}: {e}")

    print(f"Made {improvements_made} improvements based on JAO similarity")

    return matching_results


def visualize_results_with_duplicates(jao_gdf, network_gdf, matching_results):
    """Create a visualization that highlights duplicate JAO lines and includes merged network paths."""
    # Create GeoJSON data for lines
    jao_features = []
    network_features = []
    merged_network_features = []  # New list for merged network paths

    # Create sets to track which lines are matched or duplicates
    matched_jao_ids = set()
    duplicate_jao_ids = set()
    geometric_match_jao_ids = set()

    # Create sets to track network lines by match type
    regular_matched_network_ids = set()
    geometric_matched_network_ids = set()
    duplicate_matched_network_ids = set()
    parallel_matched_network_ids = set()  # Add this line
    parallel_voltage_matched_network_ids = set()  # Add this line

    # First, identify all matched JAO and network lines by type
    for result in matching_results:
        if result['matched']:
            jao_id = str(result['jao_id'])
            network_ids = result.get('network_ids', [])

            if result.get('is_duplicate', False):
                duplicate_jao_ids.add(jao_id)
                for network_id in network_ids:
                    duplicate_matched_network_ids.add(str(network_id))
            elif result.get('is_geometric_match', False):
                geometric_match_jao_ids.add(jao_id)
                for network_id in network_ids:
                    geometric_matched_network_ids.add(str(network_id))
            else:
                matched_jao_ids.add(jao_id)
                for network_id in network_ids:
                    regular_matched_network_ids.add(str(network_id))

    # Add JAO lines to GeoJSON with matched/unmatched/duplicate/geometric status
    for idx, row in jao_gdf.iterrows():
        # Create a unique ID for each line
        line_id = f"jao_{row['id']}"
        coords = list(row.geometry.coords)

        # Check if this JAO line is matched or a duplicate or geometric match
        jao_id = str(row['id'])
        is_matched = jao_id in matched_jao_ids
        is_duplicate = jao_id in duplicate_jao_ids
        is_geometric = jao_id in geometric_match_jao_ids

        # Determine the line status
        if is_duplicate:
            status = "duplicate"
            tooltip_status = "Parallel Circuit"
        elif is_geometric:
            status = "geometric"
            tooltip_status = "Geometric Match"
        elif is_matched:
            status = "matched"
            tooltip_status = "Matched"
        else:
            status = "unmatched"
            tooltip_status = "Unmatched"

        # Create GeoJSON feature
        feature = {
            "type": "Feature",
            "id": line_id,
            "properties": {
                "type": "jao",
                "id": jao_id,
                "name": str(row['NE_name']),
                "voltage": int(row['v_nom']),
                "status": status,
                "tooltip": f"JAO: {jao_id} - {row['NE_name']} ({row['v_nom']} kV) - {tooltip_status}"
            },
            "geometry": {
                "type": "LineString",
                "coordinates": [[float(x), float(y)] for x, y in coords]
            }
        }
        jao_features.append(feature)

    # Add network lines to GeoJSON with appropriate match status
    for idx, row in network_gdf.iterrows():
        line_id = f"network_{row['id']}"
        coords = list(row.geometry.coords)
        network_id = str(row['id'])

        # Determine match status for network line
        is_regular_match = network_id in regular_matched_network_ids
        is_geometric_match = network_id in geometric_matched_network_ids
        is_duplicate_match = network_id in duplicate_matched_network_ids

        if is_regular_match:
            status = "matched"
            tooltip_status = "Matched"
        elif is_geometric_match:
            status = "geometric"
            tooltip_status = "Geometric Match"
        elif is_duplicate_match:
            status = "duplicate"
            tooltip_status = "Parallel Circuit"
        else:
            status = "unmatched"
            tooltip_status = "Unmatched"

        feature = {
            "type": "Feature",
            "id": line_id,
            "properties": {
                "type": "network",
                "id": network_id,
                "voltage": int(row['v_nom']),
                "status": status,
                "tooltip": f"Network: {row['id']} ({row['v_nom']} kV) - {tooltip_status}"
            },
            "geometry": {
                "type": "LineString",
                "coordinates": [[float(x), float(y)] for x, y in coords]
            }
        }
        network_features.append(feature)

    # Create merged network path features for each JAO match
    from shapely.geometry import LineString, MultiLineString
    from shapely.ops import linemerge, unary_union
    import numpy as np

    for result in matching_results:
        if not result['matched'] or not result.get('network_ids'):
            continue

        jao_id = str(result['jao_id'])
        network_ids = result.get('network_ids', [])

        # Skip if no network lines
        if not network_ids:
            continue

        # Get all network geometries for this match
        network_geometries = []
        for network_id in network_ids:
            network_row = network_gdf[network_gdf['id'].astype(str) == network_id]
            if not network_row.empty:
                network_geometries.append(network_row.iloc[0].geometry)

        # Skip if no geometries
        if not network_geometries:
            continue

        # Try to merge the network geometries
        try:
            # First, convert to a collection
            multi_line = MultiLineString(network_geometries)

            # Attempt to merge the lines
            merged_line = linemerge(multi_line)

            # Use the merged line if successful, otherwise use the multiline
            if merged_line.geom_type == 'LineString':
                final_geom = merged_line
            else:
                final_geom = multi_line

            # Determine line status based on JAO status
            if jao_id in duplicate_jao_ids:
                status = "duplicate"
                tooltip_status = "Parallel Circuit"
            elif jao_id in geometric_match_jao_ids:
                status = "geometric"
                tooltip_status = "Geometric Match"
            else:
                status = "matched"
                tooltip_status = "Matched"

            # Calculate the match quality score for the tooltip
            match_quality = result.get('match_quality', 'Unknown')
            length_ratio = result.get('length_ratio', 0)

            # Create a unique ID for the merged path
            merged_id = f"merged-net-jao-{jao_id}"

            # Create coordinates list based on geometry type
            if final_geom.geom_type == 'LineString':
                coords = list(final_geom.coords)
                geometry = {
                    "type": "LineString",
                    "coordinates": [[float(x), float(y)] for x, y in coords]
                }
            else:  # MultiLineString
                multi_coords = []
                for line in final_geom.geoms:
                    line_coords = [[float(x), float(y)] for x, y in line.coords]
                    multi_coords.append(line_coords)
                geometry = {
                    "type": "MultiLineString",
                    "coordinates": multi_coords
                }

            # Create GeoJSON feature for the merged network path
            feature = {
                "type": "Feature",
                "id": merged_id,
                "properties": {
                    "type": "merged_network",
                    "jao_id": jao_id,
                    "network_ids": ",".join(network_ids),
                    "status": status,
                    "tooltip": f"Merged Network Path for JAO {jao_id} ({len(network_ids)} lines) - {match_quality} - Ratio: {length_ratio:.2f}"
                },
                "geometry": geometry
            }

            merged_network_features.append(feature)

        except Exception as e:
            print(f"Error creating merged network path for JAO {jao_id}: {e}")

    # Create GeoJSON collections
    jao_collection = {"type": "FeatureCollection", "features": jao_features}
    network_collection = {"type": "FeatureCollection", "features": network_features}
    merged_network_collection = {"type": "FeatureCollection", "features": merged_network_features}

    # Convert to JSON strings
    jao_json = json.dumps(jao_collection)
    network_json = json.dumps(network_collection)
    merged_network_json = json.dumps(merged_network_collection)

    # Create a complete HTML file from scratch
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>JAO-Network Line Matching Results with Merged Paths</title>

        <link rel="stylesheet" href="https://unpkg.com/leaflet@1.7.1/dist/leaflet.css" />
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css">
        <link rel="stylesheet" href="https://ppete2.github.io/Leaflet.PolylineMeasure/Leaflet.PolylineMeasure.css" />

        <script src="https://unpkg.com/leaflet@1.7.1/dist/leaflet.js"></script>
        <script src="https://ppete2.github.io/Leaflet.PolylineMeasure/Leaflet.PolylineMeasure.js"></script>

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

            .leaflet-control-polylinemeasure {{
                background-color: white !important;
                padding: 4px !important;
                border-radius: 4px !important;
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
                        <div class="filter-option active" data-filter="jao-matched">Matched JAO Lines</div>
                        <div class="filter-option active" data-filter="jao-duplicate">Parallel Circuit JAO Lines</div>
                        <div class="filter-option active" data-filter="jao-geometric">Geometric Match JAO Lines</div>
                        <div class="filter-option active" data-filter="jao-unmatched">Unmatched JAO Lines</div>
                        <div class="filter-option active" data-filter="network-matched">Matched Network Lines</div>
                        <div class="filter-option active" data-filter="network-geometric">Geometric Match Network Lines</div>
                        <div class="filter-option active" data-filter="network-duplicate">Parallel Circuit Network Lines</div>
                        <div class="filter-option active" data-filter="network-unmatched">Unmatched Network Lines</div>
                        <div class="filter-option active" data-filter="merged-paths">Merged Network Paths</div>
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
                        <div>Matched JAO Lines (Graph-based)</div>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background-color: #9932CC;"></div>
                        <div>Parallel Circuit JAO Lines</div>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background-color: #00BFFF;"></div>
                        <div>Geometric Match JAO Lines</div>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background-color: red;"></div>
                        <div>Unmatched JAO Lines</div>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background-color: purple;"></div>
                        <div>Matched Network Lines (Graph-based)</div>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background-color: #DA70D6;"></div>
                        <div>Parallel Circuit Network Lines</div>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background-color: #FF8C00;"></div>
                        <div>Geometric Match Network Lines</div>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background-color: blue;"></div>
                        <div>Unmatched Network Lines</div>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background-color: #32CD32; height: 5px;"></div>
                        <div>Merged Network Paths</div>
                    </div>
                </div>
            </div>

            <div class="control-section">
                <h3><i class="fas fa-ruler"></i> Measurement Tool</h3>
                <p>Click the ruler icon on the map to measure distances.</p>
                <p>This can help check distances between endpoints.</p>
            </div>
        </div>

        <script>
            // Initialize the map
            var map = L.map('map').setView([51.1657, 10.4515], 6);

            // Add base tile layer
            L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
                attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
            }}).addTo(map);

            // Add measurement tool
            L.control.polylineMeasure({{
                position: 'topleft',
                unit: 'metres',
                showBearings: true,
                clearMeasurementsOnStop: false,
                showClearControl: true,
                showUnitControl: true
            }}).addTo(map);

            // Load the GeoJSON data - DIRECTLY EMBEDDED IN THE HTML
            var jaoLines = {jao_json};
            var networkLines = {network_json};
            var mergedNetworkPaths = {merged_network_json};

            // Define styling for the JAO lines
            function jaoStyle(feature) {{
                if (feature.properties.status === "duplicate") {{
                    return {{
                        "color": "#9932CC", // Purple for parallel circuits
                        "weight": 3,
                        "opacity": 0.8,
                        "dashArray": "5, 5" // Dashed line for parallel circuits
                    }};
                }} else if (feature.properties.status === "geometric") {{
                    return {{
                        "color": "#00BFFF", // Deep Sky Blue for geometric matches
                        "weight": 3,
                        "opacity": 0.8,
                        "dashArray": "10, 5" // Different dash pattern for geometric matches
                    }};
                }} else if (feature.properties.status === "matched") {{
                    return {{
                        "color": "green",
                        "weight": 3,
                        "opacity": 0.8
                    }};
                }} else {{
                    return {{
                        "color": "red",
                        "weight": 3,
                        "opacity": 0.8
                    }};
                }}
            }};

            // Define styling for network lines with different types
            function networkStyle(feature) {{
                if (feature.properties.status === "matched") {{
                    return {{
                        "color": "purple",  // Regular matches
                        "weight": 2,
                        "opacity": 0.6
                    }};
                }} else if (feature.properties.status === "geometric") {{
                    return {{
                        "color": "#FF8C00", // Dark Orange for geometric matches
                        "weight": 2,
                        "opacity": 0.6,
                        "dashArray": "10, 5"
                    }};
                }} else if (feature.properties.status === "duplicate") {{
                    return {{
                        "color": "#DA70D6", // Orchid for parallel circuits
                        "weight": 2,
                        "opacity": 0.6,
                        "dashArray": "5, 5"
                    }};
                }} else {{
                    return {{
                        "color": "blue",    // Unmatched
                        "weight": 2,
                        "opacity": 0.6
                    }};
                }}
            }};

            // Define styling for merged network paths
            function mergedNetworkStyle(feature) {{
                if (feature.properties.status === "duplicate") {{
                    return {{
                        "color": "#DA70D6", // Orchid for parallel circuits
                        "weight": 5,
                        "opacity": 0.8,
                        "dashArray": "10, 5"
                    }};
                }} else if (feature.properties.status === "geometric") {{
                    return {{
                        "color": "#FF8C00", // Dark Orange for geometric matches
                        "weight": 5,
                        "opacity": 0.8,
                        "dashArray": "15, 10"
                    }};
                }} else {{
                    return {{
                        "color": "#32CD32", // Lime Green for matched
                        "weight": 5,
                        "opacity": 0.8
                    }};
                }}
            }};

            // Create the JAO layers
            var jaoMatchedLayer = L.geoJSON(jaoLines, {{
                filter: function(feature) {{
                    return feature.properties.status === "matched";
                }},
                style: jaoStyle,
                onEachFeature: function(feature, layer) {{
                    layer.bindTooltip(feature.properties.tooltip);
                    layer.on('click', function() {{
                        highlightFeature(feature.id, feature.properties.id);
                    }});
                }}
            }}).addTo(map);

            var jaoDuplicateLayer = L.geoJSON(jaoLines, {{
                filter: function(feature) {{
                    return feature.properties.status === "duplicate";
                }},
                style: jaoStyle,
                onEachFeature: function(feature, layer) {{
                    layer.bindTooltip(feature.properties.tooltip);
                    layer.on('click', function() {{
                        highlightFeature(feature.id, feature.properties.id);
                    }});
                }}
            }}).addTo(map);

            var jaoGeometricLayer = L.geoJSON(jaoLines, {{
                filter: function(feature) {{
                    return feature.properties.status === "geometric";
                }},
                style: jaoStyle,
                onEachFeature: function(feature, layer) {{
                    layer.bindTooltip(feature.properties.tooltip);
                    layer.on('click', function() {{
                        highlightFeature(feature.id, feature.properties.id);
                    }});
                }}
            }}).addTo(map);

            var jaoUnmatchedLayer = L.geoJSON(jaoLines, {{
                filter: function(feature) {{
                    return feature.properties.status === "unmatched";
                }},
                style: jaoStyle,
                onEachFeature: function(feature, layer) {{
                    layer.bindTooltip(feature.properties.tooltip);
                    layer.on('click', function() {{
                        highlightFeature(feature.id, feature.properties.id);
                    }});
                }}
            }}).addTo(map);

            // Create the network layers
            var networkMatchedLayer = L.geoJSON(networkLines, {{
                filter: function(feature) {{
                    return feature.properties.status === "matched";
                }},
                style: networkStyle,
                onEachFeature: function(feature, layer) {{
                    layer.bindTooltip(feature.properties.tooltip);
                    layer.on('click', function() {{
                        highlightFeature(feature.id);
                    }});
                }}
            }}).addTo(map);

            var networkDuplicateLayer = L.geoJSON(networkLines, {{
                filter: function(feature) {{
                    return feature.properties.status === "duplicate";
                }},
                style: networkStyle,
                onEachFeature: function(feature, layer) {{
                    layer.bindTooltip(feature.properties.tooltip);
                    layer.on('click', function() {{
                        highlightFeature(feature.id);
                    }});
                }}
            }}).addTo(map);

            var networkGeometricLayer = L.geoJSON(networkLines, {{
                filter: function(feature) {{
                    return feature.properties.status === "geometric";
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
                    return feature.properties.status === "unmatched";
                }},
                style: networkStyle,
                onEachFeature: function(feature, layer) {{
                    layer.bindTooltip(feature.properties.tooltip);
                    layer.on('click', function() {{
                        highlightFeature(feature.id);
                    }});
                }}
            }}).addTo(map);

            // Create the merged network path layer
            var mergedNetworkLayer = L.geoJSON(mergedNetworkPaths, {{
                style: mergedNetworkStyle,
                onEachFeature: function(feature, layer) {{
                    layer.bindTooltip(feature.properties.tooltip);
                    layer.on('click', function() {{
                        // When clicking on a merged path, highlight both the path and the corresponding JAO line
                        highlightFeature(feature.id, feature.properties.jao_id);
                    }});
                }}
            }}).addTo(map);

            // Function to highlight a feature and optionally a related JAO feature
            function highlightFeature(id, relatedDlrId) {{
                // Clear any existing highlights
                clearHighlights();

                // Apply highlight to the specific feature across all layers
                function checkAndHighlight(layer) {{
                    layer.eachLayer(function(leafletLayer) {{
                        if (leafletLayer.feature.id === id) {{
                            leafletLayer.setStyle({{className: 'highlighted'}});
                            if (leafletLayer._path) leafletLayer._path.classList.add('highlighted');

                            // Center on the highlighted feature
                            var bounds = leafletLayer.getBounds();
                            map.fitBounds(bounds, {{ padding: [50, 50] }});
                        }}
                    }});
                }}

                // If a related JAO ID is provided, highlight that JAO line too
                function checkAndHighlightJAO(layer) {{
                    layer.eachLayer(function(leafletLayer) {{
                        if (relatedDlrId && leafletLayer.feature.properties.type === 'jao' && 
                            leafletLayer.feature.properties.id === relatedDlrId) {{
                            leafletLayer.setStyle({{className: 'highlighted'}});
                            if (leafletLayer._path) leafletLayer._path.classList.add('highlighted');
                        }}
                    }});
                }}

                // If a related merged network path exists for this JAO, highlight it too
                function checkAndHighlightMergedPath(layer) {{
                    layer.eachLayer(function(leafletLayer) {{
                        if (relatedDlrId && leafletLayer.feature.properties.type === 'merged_network' && 
                            leafletLayer.feature.properties.jao_id === relatedDlrId) {{
                            leafletLayer.setStyle({{className: 'highlighted'}});
                            if (leafletLayer._path) leafletLayer._path.classList.add('highlighted');
                        }}
                    }});
                }}

                // Check all layers for the specific feature
                checkAndHighlight(jaoMatchedLayer);
                checkAndHighlight(jaoDuplicateLayer);
                checkAndHighlight(jaoGeometricLayer);
                checkAndHighlight(jaoUnmatchedLayer);
                checkAndHighlight(networkMatchedLayer);
                checkAndHighlight(networkDuplicateLayer);
                checkAndHighlight(networkGeometricLayer);
                checkAndHighlight(networkUnmatchedLayer);
                checkAndHighlight(mergedNetworkLayer);

                // If a related JAO ID is provided, highlight that JAO line
                if (relatedDlrId) {{
                    checkAndHighlightJAO(jaoMatchedLayer);
                    checkAndHighlightJAO(jaoDuplicateLayer);
                    checkAndHighlightJAO(jaoGeometricLayer);
                    checkAndHighlightJAO(jaoUnmatchedLayer);

                    // Also highlight any merged network path for this JAO
                    checkAndHighlightMergedPath(mergedNetworkLayer);
                }}
            }}

            // Function to clear highlights
            function clearHighlights() {{
                function resetStyle(layer) {{
                    layer.eachLayer(function(leafletLayer) {{
                        if (leafletLayer.feature.properties.type === 'jao') {{
                            leafletLayer.setStyle(jaoStyle(leafletLayer.feature));
                        }} else if (leafletLayer.feature.properties.type === 'merged_network') {{
                            leafletLayer.setStyle(mergedNetworkStyle(leafletLayer.feature));
                        }} else {{
                            leafletLayer.setStyle(networkStyle(leafletLayer.feature));
                        }}
                        if (leafletLayer._path) leafletLayer._path.classList.remove('highlighted');
                    }});
                }}

                // Reset all layers
                resetStyle(jaoMatchedLayer);
                resetStyle(jaoDuplicateLayer);
                resetStyle(jaoGeometricLayer);
                resetStyle(jaoUnmatchedLayer);
                resetStyle(networkMatchedLayer);
                resetStyle(networkDuplicateLayer);
                resetStyle(networkGeometricLayer);
                resetStyle(networkUnmatchedLayer);
                resetStyle(mergedNetworkLayer);
            }}

            // Function to create the search index
            function createSearchIndex() {{
                var searchData = [];

                // Add JAO lines to search data
                jaoLines.features.forEach(function(feature) {{
                    searchData.push({{
                        id: feature.id,
                        text: "JAO: " + feature.properties.id + " - " + feature.properties.name + " (" + feature.properties.voltage + " kV)",
                        type: "jao",
                        feature: feature,
                        jaoId: feature.properties.id
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

                // Add merged network paths to search data
                mergedNetworkPaths.features.forEach(function(feature) {{
                    searchData.push({{
                        id: feature.id,
                        text: "Merged Path: " + feature.properties.jao_id + " (" + feature.properties.network_ids + ")",
                        type: "merged_network",
                        feature: feature,
                        jaoId: feature.properties.jao_id
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

                // Function to apply voltage filter to a layer
                function applyVoltageFilter(layer) {{
                    if (!allVoltages) {{
                        layer.eachLayer(function(leafletLayer) {{
                            var visible = activeVoltages.includes(leafletLayer.feature.properties.voltage);
                            if (visible) {{
                                if (leafletLayer._path) leafletLayer._path.style.display = "block";
                            }} else {{
                                if (leafletLayer._path) leafletLayer._path.style.display = "none";
                            }}
                        }});
                    }}
                }}

                // Show/hide JAO matched layer
                if (activeLineTypes.includes("jao-matched")) {{
                    jaoMatchedLayer.addTo(map);
                    applyVoltageFilter(jaoMatchedLayer);
                }} else {{
                    map.removeLayer(jaoMatchedLayer);
                }}

                // Show/hide JAO duplicate layer
                if (activeLineTypes.includes("jao-duplicate")) {{
                    jaoDuplicateLayer.addTo(map);
                    applyVoltageFilter(jaoDuplicateLayer);
                }} else {{
                    map.removeLayer(jaoDuplicateLayer);
                }}

                // Show/hide JAO geometric match layer
                if (activeLineTypes.includes("jao-geometric")) {{
                    jaoGeometricLayer.addTo(map);
                    applyVoltageFilter(jaoGeometricLayer);
                }} else {{
                    map.removeLayer(jaoGeometricLayer);
                }}

                // Show/hide JAO unmatched layer
                if (activeLineTypes.includes("jao-unmatched")) {{
                    jaoUnmatchedLayer.addTo(map);
                    applyVoltageFilter(jaoUnmatchedLayer);
                }} else {{
                    map.removeLayer(jaoUnmatchedLayer);
                }}

                // Show/hide Network matched layer
                if (activeLineTypes.includes("network-matched")) {{
                    networkMatchedLayer.addTo(map);
                    applyVoltageFilter(networkMatchedLayer);
                }} else {{
                    map.removeLayer(networkMatchedLayer);
                }}

                // Show/hide Network duplicate layer
                if (activeLineTypes.includes("network-duplicate")) {{
                    networkDuplicateLayer.addTo(map);
                    applyVoltageFilter(networkDuplicateLayer);
                }} else {{
                    map.removeLayer(networkDuplicateLayer);
                }}

                // Show/hide Network geometric layer
                if (activeLineTypes.includes("network-geometric")) {{
                    networkGeometricLayer.addTo(map);
                    applyVoltageFilter(networkGeometricLayer);
                }} else {{
                    map.removeLayer(networkGeometricLayer);
                }}

                // Show/hide Network unmatched layer
                if (activeLineTypes.includes("network-unmatched")) {{
                    networkUnmatchedLayer.addTo(map);
                    applyVoltageFilter(networkUnmatchedLayer);
                }} else {{
                    map.removeLayer(networkUnmatchedLayer);
                }}

                // Show/hide Merged Network Paths layer
                if (activeLineTypes.includes("merged-paths")) {{
                    mergedNetworkLayer.addTo(map);
                }} else {{
                    map.removeLayer(mergedNetworkLayer);
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

                            // Highlight the feature and possibly related JAO
                            highlightFeature(result.id, result.jaoId);

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
                if (feature.geometry.type === "LineString") {{
                    var coords = feature.geometry.coordinates;
                    var sum = [0, 0];

                    for (var i = 0; i < coords.length; i++) {{
                        sum[0] += coords[i][0];
                        sum[1] += coords[i][1];
                    }}

                    return [sum[0] / coords.length, sum[1] / coords.length];
                }} else if (feature.geometry.type === "MultiLineString") {{
                    // Handle multilinestring by using the first line
                    if (feature.geometry.coordinates.length > 0) {{
                        var coords = feature.geometry.coordinates[0];
                        var sum = [0, 0];

                        for (var i = 0; i < coords.length; i++) {{
                            sum[0] += coords[i][0];
                            sum[1] += coords[i][1];
                        }}

                        return [sum[0] / coords.length, sum[1] / coords.length];
                    }}
                }}

                // Fallback
                return [0, 0];
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
    output_file = output_dir / 'jao_network_matching_with_duplicates.html'
    with open(output_file, 'w') as f:
        f.write(html)

    return output_file




# Create an enhanced summary table with electrical parameters
def create_enhanced_summary_table(jao_gdf, network_gdf, matching_results):
    """Create an HTML table with detailed information about the matching results including electrical parameters."""
    import json
    from pathlib import Path

    # Convert results to DataFrame for easier handling
    results_df = pd.DataFrame([r for r in matching_results if 'jao_id' in r])

    # Create summary statistics
    total_jao_lines = len(matching_results)
    matched_lines = sum(result.get('matched', False) for result in matching_results)
    unmatched_lines = total_jao_lines - matched_lines

    # Count different types of matches
    regular_matches = sum(1 for result in matching_results if result.get('matched', False) and
                          not result.get('is_duplicate', False) and
                          not result.get('is_geometric_match', False) and
                          not result.get('is_parallel_circuit', False) and
                          not result.get('is_parallel_voltage_circuit', False))

    duplicate_matches = sum(1 for result in matching_results if result.get('is_duplicate', False))
    geometric_matches = sum(1 for result in matching_results if result.get('is_geometric_match', False))
    parallel_matches = sum(1 for result in matching_results if result.get('is_parallel_circuit', False))
    parallel_voltage_matches = sum(1 for result in matching_results if result.get('is_parallel_voltage_circuit', False))

    # Count specific match qualities
    match_quality_counts = {}
    for result in matching_results:
        quality = result.get('match_quality', '')
        if quality not in match_quality_counts:
            match_quality_counts[quality] = 0
        match_quality_counts[quality] += 1

    # Create HTML for summary
    html_summary = f"""
    <html>
    <head>
        <title>JAO-Network Line Matching Summary with Electrical Parameters</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 20px;
            }}
            h1, h2, h3 {{
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
                margin-bottom: 30px;
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
            .filter-controls {{
                margin: 20px 0;
                padding: 10px;
                background-color: #eee;
                border-radius: 5px;
            }}
            .filter-buttons {{
                display: flex;
                flex-wrap: wrap;
                gap: 5px;
                margin-bottom: 10px;
            }}
            .filter-buttons button {{
                padding: 5px 10px;
                border: 1px solid #ccc;
                border-radius: 3px;
                background-color: #f8f8f8;
                cursor: pointer;
            }}
            .filter-buttons button:hover {{
                background-color: #e0e0e0;
            }}
            .matched {{
                background-color: #90EE90;  /* Light green */
            }}
            .geometric {{
                background-color: #FFDAB9;  /* Peach */
            }}
            .duplicate {{
                background-color: #E6E6FA;  /* Lavender */
            }}
            .parallel {{
                background-color: #D8BFD8;  /* Thistle */
            }}
            .parallel-voltage {{
                background-color: #FFE4B5;  /* Moccasin */
            }}
            .unmatched {{
                background-color: #ffcccb;  /* Light red */
            }}
            .parameter-details {{
                margin-left: 20px;
                margin-bottom: 30px;
            }}
            .segment-table {{
                width: 95%;
                margin: 10px auto;
            }}
            .segment-table th {{
                background-color: #5c85d6;
            }}
            .good-match {{
                background-color: #c8e6c9;
            }}
            .moderate-match {{
                background-color: #fff9c4;
            }}
            .poor-match {{
                background-color: #ffccbc;
            }}
            .toggle-btn {{
                background-color: #4CAF50;
                color: white;
                padding: 5px 10px;
                border: none;
                border-radius: 4px;
                cursor: pointer;
                margin-bottom: 10px;
            }}
            .details-section {{
                display: none;
            }}
            .per-km-table {{
                margin-top: 20px;
                width: 95%;
                margin-left: auto;
                margin-right: auto;
            }}
            .per-km-table th {{
                background-color: #7b68ee;
            }}
        </style>
        <script>
            function filterTable() {{
                const filter = document.getElementById('filter').value.toLowerCase();
                const rows = document.getElementById('resultsTable').getElementsByTagName('tr');

                // Track which detail rows to show (those associated with visible main rows)
                const visibleDetailIds = new Set();

                // First pass - filter main rows and track which ones are visible
                for (let i = 0; i < rows.length; i++) {{
                    const row = rows[i];

                    // Only process main data rows (not headers or parameter detail rows)
                    if (row.classList.contains('data-row')) {{
                        const text = row.textContent.toLowerCase();
                        const rowId = row.getAttribute('data-result-id');

                        if (text.includes(filter)) {{
                            row.style.display = '';
                            if (rowId) visibleDetailIds.add(rowId);
                        }} else {{
                            row.style.display = 'none';
                        }}
                    }}
                }}

                // Second pass - show/hide parameter detail rows based on main row visibility
                for (let i = 0; i < rows.length; i++) {{
                    const row = rows[i];
                    if (row.classList.contains('parameter-row')) {{
                        const rowId = row.getAttribute('data-result-id');
                        row.style.display = visibleDetailIds.has(rowId) ? '' : 'none';
                    }}
                }}
            }}

            function filterByMatchStatus(status) {{
                const rows = document.getElementById('resultsTable').getElementsByTagName('tr');

                // Track which detail rows to show
                const visibleDetailIds = new Set();

                // First pass - filter main rows
                for (let i = 0; i < rows.length; i++) {{
                    const row = rows[i];

                    if (row.classList.contains('data-row')) {{
                        const matchedCell = row.cells[3].textContent.trim();
                        const matchType = row.getAttribute('data-match-type');
                        const rowId = row.getAttribute('data-result-id');

                        let showRow = false;
                        if (status === 'all') {{
                            showRow = true;
                        }} else if (status === 'matched' && matchedCell === 'Yes' && matchType === 'regular') {{
                            showRow = true;
                        }} else if (status === 'geometric' && matchType === 'geometric') {{
                            showRow = true;
                        }} else if (status === 'duplicate' && matchType === 'duplicate') {{
                            showRow = true;
                        }} else if (status === 'parallel' && matchType === 'parallel') {{
                            showRow = true;
                        }} else if (status === 'parallel-voltage' && matchType === 'parallel-voltage') {{
                            showRow = true;
                        }} else if (status === 'unmatched' && matchedCell === 'No') {{
                            showRow = true;
                        }}

                        if (showRow) {{
                            row.style.display = '';
                            if (rowId) visibleDetailIds.add(rowId);
                        }} else {{
                            row.style.display = 'none';
                        }}
                    }}
                }}

                // Second pass - show/hide parameter rows
                for (let i = 0; i < rows.length; i++) {{
                    const row = rows[i];
                    if (row.classList.contains('parameter-row')) {{
                        const rowId = row.getAttribute('data-result-id');
                        row.style.display = visibleDetailIds.has(rowId) ? '' : 'none';
                    }}
                }}
            }}

            function filterByVoltage(voltage) {{
                const rows = document.getElementById('resultsTable').getElementsByTagName('tr');

                // Track which detail rows to show
                const visibleDetailIds = new Set();

                // First pass - filter main rows
                for (let i = 0; i < rows.length; i++) {{
                    const row = rows[i];

                    if (row.classList.contains('data-row')) {{
                        const voltageCell = row.cells[2].textContent.trim();
                        const rowId = row.getAttribute('data-result-id');

                        if (voltage === 'all' || voltageCell === voltage) {{
                            row.style.display = '';
                            if (rowId) visibleDetailIds.add(rowId);
                        }} else {{
                            row.style.display = 'none';
                        }}
                    }}
                }}

                // Second pass - show/hide parameter rows
                for (let i = 0; i < rows.length; i++) {{
                    const row = rows[i];
                    if (row.classList.contains('parameter-row')) {{
                        const rowId = row.getAttribute('data-result-id');
                        row.style.display = visibleDetailIds.has(rowId) ? '' : 'none';
                    }}
                }}
            }}

            function filterByMatchQuality(quality) {{
                const rows = document.getElementById('resultsTable').getElementsByTagName('tr');

                // Track which detail rows to show
                const visibleDetailIds = new Set();

                // First pass - filter main rows
                for (let i = 0; i < rows.length; i++) {{
                    const row = rows[i];

                    if (row.classList.contains('data-row')) {{
                        const qualityCell = row.cells[7].textContent.trim();
                        const rowId = row.getAttribute('data-result-id');

                        if (quality === 'all' || qualityCell.includes(quality)) {{
                            row.style.display = '';
                            if (rowId) visibleDetailIds.add(rowId);
                        }} else {{
                            row.style.display = 'none';
                        }}
                    }}
                }}

                // Second pass - show/hide parameter rows
                for (let i = 0; i < rows.length; i++) {{
                    const row = rows[i];
                    if (row.classList.contains('parameter-row')) {{
                        const rowId = row.getAttribute('data-result-id');
                        row.style.display = visibleDetailIds.has(rowId) ? '' : 'none';
                    }}
                }}
            }}

            function toggleDetails(id) {{
                var detailsSection = document.getElementById('details-' + id);
                var parameterRow = document.querySelector(`.parameter-row[data-result-id="${id}"]`);

                if (detailsSection.style.display === 'block') {{
                    detailsSection.style.display = 'none';
                    parameterRow.style.display = 'none';
                    document.getElementById('btn-' + id).textContent = 'Show Electrical Parameters';
                }} else {{
                    detailsSection.style.display = 'block';
                    parameterRow.style.display = '';
                    document.getElementById('btn-' + id).textContent = 'Hide Electrical Parameters';
                }}
            }}
        </script>
    </head>
    <body>
        <h1>JAO-Network Line Matching Results with Electrical Parameters</h1>

        <div class="summary">
            <h2>Summary</h2>
            <p>Total JAO Lines: {total_jao_lines}</p>
            <p>Matched Lines: {matched_lines} ({matched_lines / total_jao_lines * 100:.1f}%)</p>
            <ul>
                <li>Regular matches: {regular_matches} ({regular_matches / total_jao_lines * 100:.1f}%)</li>
                <li>Geometric matches: {geometric_matches} ({geometric_matches / total_jao_lines * 100:.1f}%)</li>
                <li>Parallel circuits: {parallel_matches} ({parallel_matches / total_jao_lines * 100:.1f}%)</li>
                <li>Parallel voltage circuits: {parallel_voltage_matches} ({parallel_voltage_matches / total_jao_lines * 100:.1f}%)</li>
                <li>Duplicates: {duplicate_matches} ({duplicate_matches / total_jao_lines * 100:.1f}%)</li>
            </ul>
            <p>Unmatched Lines: {unmatched_lines} ({unmatched_lines / total_jao_lines * 100:.1f}%)</p>
            <p>Match Quality Details:</p>
            <ul>
    """

    # Add match quality details
    for quality, count in match_quality_counts.items():
        if count > 0:
            percentage = count / total_jao_lines * 100
            html_summary += f"<li>{quality}: {count} ({percentage:.1f}%)</li>"

    html_summary += """
            </ul>
        </div>

        <div class="filter-controls">
            <h2>Filter Results</h2>
            <input type="text" id="filter" onkeyup="filterTable()" placeholder="Search for JAO lines...">

            <h3>By Match Status:</h3>
            <div class="filter-buttons">
                <button onclick="filterByMatchStatus('all')">All</button>
                <button onclick="filterByMatchStatus('matched')">Regular Matches</button>
                <button onclick="filterByMatchStatus('geometric')">Geometric Matches</button>
                <button onclick="filterByMatchStatus('parallel')">Parallel Circuits</button>
                <button onclick="filterByMatchStatus('parallel-voltage')">Parallel Voltage</button>
                <button onclick="filterByMatchStatus('duplicate')">Duplicates</button>
                <button onclick="filterByMatchStatus('unmatched')">Unmatched</button>
            </div>

            <h3>By Voltage Level:</h3>
            <div class="filter-buttons">
                <button onclick="filterByVoltage('all')">All</button>
                <button onclick="filterByVoltage('220')">220 kV</button>
                <button onclick="filterByVoltage('400')">400 kV</button>
            </div>

            <h3>By Match Quality:</h3>
            <div class="filter-buttons">
                <button onclick="filterByMatchQuality('all')">All</button>
                <button onclick="filterByMatchQuality('Excellent')">Excellent</button>
                <button onclick="filterByMatchQuality('Good')">Good</button>
                <button onclick="filterByMatchQuality('Fair')">Fair</button>
                <button onclick="filterByMatchQuality('Poor')">Poor</button>
                <button onclick="filterByMatchQuality('Geometric')">Geometric</button>
                <button onclick="filterByMatchQuality('Parallel')">Parallel</button>
                <button onclick="filterByMatchQuality('No match')">Unmatched</button>
                <button onclick="filterByMatchQuality('No path')">No Path</button>
            </div>
        </div>

        <h2>Detailed Results</h2>
        <table id="resultsTable">
            <tr>
                <th>JAO ID</th>
                <th>JAO Name</th>
                <th>Voltage (kV)</th>
                <th>Matched</th>
                <th>Network IDs</th>
                <th>JAO Length (km)</th>
                <th>Length Ratio</th>
                <th>Match Quality</th>
                <th>Electrical Parameters</th>
            </tr>
    """

    # Add rows for each result
    for i, result in enumerate(matching_results):
        network_ids = ", ".join(result.get('network_ids', [])) if result.get('matched', False) and result.get(
            'network_ids') else "-"
        length_ratio = f"{result.get('length_ratio', '-'):.2f}" if result.get('length_ratio') is not None else "-"
        jao_length_km = f"{result.get('jao_length', 0) / 1000:.2f}" if 'jao_length' in result else "-"

        # Determine row class and match type based on match status
        if result.get('matched', False):
            if result.get('is_duplicate', False):
                css_class = "duplicate"
                match_type = "duplicate"
            elif result.get('is_geometric_match', False):
                css_class = "geometric"
                match_type = "geometric"
            elif result.get('is_parallel_circuit', False):
                css_class = "parallel"
                match_type = "parallel"
            elif result.get('is_parallel_voltage_circuit', False):
                css_class = "parallel-voltage"
                match_type = "parallel-voltage"
            else:
                css_class = "matched"
                match_type = "regular"
        else:
            css_class = "unmatched"
            match_type = "unmatched"

        # Create a unique ID for this result
        result_id = f"result-{i}"

        html_summary += f"""
            <tr class="{css_class} data-row" data-result-id="{result_id}" data-match-type="{match_type}">
                <td>{result.get('jao_id', '-')}</td>
                <td>{result.get('jao_name', '-')}</td>
                <td>{result.get('v_nom', '-')}</td>
                <td>{"Yes" if result.get('matched', False) else "No"}</td>
                <td>{network_ids}</td>
                <td>{jao_length_km}</td>
                <td>{length_ratio}</td>
                <td>{result.get('match_quality', '-')}</td>
                <td>
        """
        if result.get('matched', False):
            html_summary += f"""<button id="btn-{result_id}" class="toggle-btn" onclick="toggleDetails('{result_id}')">Show Electrical Parameters</button>"""
        else:
            html_summary += "N/A"

        html_summary += """
                </td>
            </tr>
        """

        # Add detailed electrical parameters section for matched lines
        if result.get('matched', False):
            html_summary += f"""
            <tr class="parameter-row" data-result-id="{result_id}" style="display: none;">
                <td colspan="9" class="parameter-details">
                    <div id="details-{result_id}" class="details-section">
                        <h3>JAO Line Electrical Parameters</h3>
            """

            # Check if we have JAO electrical parameter data
            if 'jao_r' in result:
                jao_r = result.get('jao_r', 0)
                jao_x = result.get('jao_x', 0)
                jao_b = result.get('jao_b', 0)
                jao_length_km = result.get('jao_length_km', 0)
                jao_r_per_km = result.get('jao_r_per_km', 0)
                jao_x_per_km = result.get('jao_x_per_km', 0)
                jao_b_per_km = result.get('jao_b_per_km', 0)

                html_summary += f"""
                        <p>Length: {jao_length_km:.2f} km</p>
                        <p>Resistance (R): {jao_r:.6f} ohm (Total)</p>
                        <p>Reactance (X): {jao_x:.6f} ohm (Total)</p>
                        <p>Susceptance (B): {jao_b:.8f} S (Total)</p>
                        <p>Resistance per km (R): {jao_r_per_km:.6f} ohm/km</p>
                        <p>Reactance per km (X): {jao_x_per_km:.6f} ohm/km</p>
                        <p>Susceptance per km (B): {jao_b_per_km:.8f} S/km</p>
                """
            else:
                html_summary += """
                        <p>Electrical parameter data not available for this JAO line</p>
                """

            # Check if we have matched lines data
            if 'matched_lines_data' in result and result['matched_lines_data']:
                html_summary += """
                        <h3>Allocated Parameters for Network Segments (Total Values)</h3>
                        <table class="segment-table">
                            <tr>
                                <th>Network ID</th>
                                <th>Length (km)</th>
                                <th>Length Ratio</th>
                                <th>Allocated R (ohm)</th>
                                <th>Original R (ohm)</th>
                                <th>R Diff (%)</th>
                                <th>Allocated X (ohm)</th>
                                <th>Original X (ohm)</th>
                                <th>X Diff (%)</th>
                                <th>Allocated B (S)</th>
                                <th>Original B (S)</th>
                                <th>B Diff (%)</th>
                            </tr>
                """

                # Add rows for each network segment
                for segment in result['matched_lines_data']:
                    # Determine color class based on parameter differences
                    r_diff_class = "good-match" if abs(
                        segment.get('r_diff_percent', float('inf'))) <= 20 else "moderate-match" if abs(
                        segment.get('r_diff_percent', float('inf'))) <= 50 else "poor-match"
                    x_diff_class = "good-match" if abs(
                        segment.get('x_diff_percent', float('inf'))) <= 20 else "moderate-match" if abs(
                        segment.get('x_diff_percent', float('inf'))) <= 50 else "poor-match"
                    b_diff_class = "good-match" if abs(
                        segment.get('b_diff_percent', float('inf'))) <= 20 else "moderate-match" if abs(
                        segment.get('b_diff_percent', float('inf'))) <= 50 else "poor-match"

                    # Format diff percentages
                    r_diff_text = f"{segment.get('r_diff_percent', float('inf')):.2f}%" if segment.get('r_diff_percent',
                                                                                                       float(
                                                                                                           'inf')) != float(
                        'inf') else "N/A"
                    x_diff_text = f"{segment.get('x_diff_percent', float('inf')):.2f}%" if segment.get('x_diff_percent',
                                                                                                       float(
                                                                                                           'inf')) != float(
                        'inf') else "N/A"
                    b_diff_text = f"{segment.get('b_diff_percent', float('inf')):.2f}%" if segment.get('b_diff_percent',
                                                                                                       float(
                                                                                                           'inf')) != float(
                        'inf') else "N/A"

                    html_summary += f"""
                    <tr>
                        <td>{segment.get('network_id', '-')}</td>
                        <td>{segment.get('length_km', 0):.2f}</td>
                        <td>{segment.get('segment_ratio', 0) * 100:.2f}%</td>
                        <td>{segment.get('allocated_r', 0):.6f}</td>
                        <td>{segment.get('original_r', 0):.6f}</td>
                        <td class="{r_diff_class}">{r_diff_text}</td>
                        <td>{segment.get('allocated_x', 0):.6f}</td>
                        <td>{segment.get('original_x', 0):.6f}</td>
                        <td class="{x_diff_class}">{x_diff_text}</td>
                        <td>{segment.get('allocated_b', 0):.8f}</td>
                        <td>{segment.get('original_b', 0):.8f}</td>
                        <td class="{b_diff_class}">{b_diff_text}</td>
                    </tr>
                    """

                html_summary += """
                        </table>

                        <h3>Per-Kilometer Parameters for Network Segments</h3>
                        <table class="per-km-table">
                            <tr>
                                <th>Network ID</th>
                                <th>Length (km)</th>
                                <th>Allocated R (ohm/km)</th>
                                <th>Original R (ohm/km)</th>
                                <th>Allocated X (ohm/km)</th>
                                <th>Original X (ohm/km)</th>
                                <th>Allocated B (S/km)</th>
                                <th>Original B (S/km)</th>
                            </tr>
                """

                # Add per-km values for each network segment
                for segment in result['matched_lines_data']:
                    html_summary += f"""
                    <tr>
                        <td>{segment.get('network_id', '-')}</td>
                        <td>{segment.get('length_km', 0):.2f}</td>
                        <td>{segment.get('allocated_r_per_km', 0):.6f}</td>
                        <td>{segment.get('original_r_per_km', 0):.6f}</td>
                        <td>{segment.get('allocated_x_per_km', 0):.6f}</td>
                        <td>{segment.get('original_x_per_km', 0):.6f}</td>
                        <td>{segment.get('allocated_b_per_km', 0):.8f}</td>
                        <td>{segment.get('original_b_per_km', 0):.8f}</td>
                    </tr>
                    """

                html_summary += """
                        </table>
                """
            else:
                html_summary += """
                        <p>No network segment parameter data available for this match</p>
                """

            html_summary += """
                    </div>
                </td>
            </tr>
            """

    html_summary += """
        </table>
    </body>
    </html>
    """

    # Save the HTML file
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / 'jao_network_matching_parameters.html'
    with open(output_file, 'w') as f:
        f.write(html_summary)

    return output_file


def visualize_results(jao_gdf, network_gdf, matching_results):
    """Create a visualization of the matching results."""
    import json
    from pathlib import Path

    # Create GeoJSON data for lines
    jao_features = []
    network_features = []

    # Create sets to track which lines are matched
    matched_jao_ids = set()
    geometric_match_jao_ids = set()
    parallel_circuit_jao_ids = set()  # Define this here
    parallel_voltage_jao_ids = set()  # Define this here
    duplicate_jao_ids = set()

    # Track network lines by match type
    regular_matched_network_ids = set()
    geometric_matched_network_ids = set()
    parallel_circuit_network_ids = set()  # Define this here
    parallel_voltage_network_ids = set()  # Define this here
    duplicate_network_ids = set()

    # First, identify all matched JAO and network lines by type
    for result in matching_results:
        if result['matched'] and result.get('network_ids'):
            jao_id = str(result['jao_id'])
            network_ids = result.get('network_ids', [])

            if result.get('is_duplicate', False):
                duplicate_jao_ids.add(jao_id)
                for network_id in network_ids:
                    duplicate_network_ids.add(str(network_id))
            elif result.get('is_geometric_match', False):
                geometric_match_jao_ids.add(jao_id)
                for network_id in network_ids:
                    geometric_matched_network_ids.add(str(network_id))
            elif result.get('is_parallel_circuit', False):
                parallel_circuit_jao_ids.add(jao_id)
                for network_id in network_ids:
                    parallel_circuit_network_ids.add(str(network_id))
            elif result.get('is_parallel_voltage_circuit', False):
                parallel_voltage_jao_ids.add(jao_id)
                for network_id in network_ids:
                    parallel_voltage_network_ids.add(str(network_id))
            else:
                matched_jao_ids.add(jao_id)
                for network_id in network_ids:
                    regular_matched_network_ids.add(str(network_id))

    # Add JAO lines to GeoJSON with status indicators
    for idx, row in jao_gdf.iterrows():
        # Create a unique ID for each line
        line_id = f"jao_{row['id']}"
        coords = list(row.geometry.coords)

        # Check JAO line match status
        jao_id = str(row['id'])

        # Determine match status and styling
        if jao_id in duplicate_jao_ids:
            status = "duplicate"
            tooltip_status = "Parallel Circuit (duplicate)"
        elif jao_id in parallel_circuit_jao_ids:
            status = "parallel"
            tooltip_status = "Parallel Circuit"
        elif jao_id in parallel_voltage_jao_ids:
            status = "parallel_voltage"
            tooltip_status = "Parallel Voltage Circuit"
        elif jao_id in geometric_match_jao_ids:
            status = "geometric"
            tooltip_status = "Geometric Match"
        elif jao_id in matched_jao_ids:
            status = "matched"
            tooltip_status = "Matched"
        else:
            status = "unmatched"
            tooltip_status = "Unmatched"

        # Create GeoJSON feature
        feature = {
            "type": "Feature",
            "id": line_id,
            "properties": {
                "type": "jao",
                "id": jao_id,
                "name": str(row['NE_name']),
                "voltage": int(row['v_nom']),
                "status": status,
                "tooltip": f"JAO: {jao_id} - {row['NE_name']} ({row['v_nom']} kV) - {tooltip_status}"
            },
            "geometry": {
                "type": "LineString",
                "coordinates": [[float(x), float(y)] for x, y in coords]
            }
        }
        jao_features.append(feature)

    # Add network lines to GeoJSON with status indicators
    for idx, row in network_gdf.iterrows():
        line_id = f"network_{row['id']}"
        coords = list(row.geometry.coords)
        network_id = str(row['id'])

        # Determine match status for this network line
        if network_id in duplicate_network_ids:
            status = "duplicate"
            tooltip_status = "Parallel Circuit (duplicate)"
        elif network_id in parallel_circuit_network_ids:
            status = "parallel"
            tooltip_status = "Parallel Circuit"
        elif network_id in parallel_voltage_network_ids:
            status = "parallel_voltage"
            tooltip_status = "Parallel Voltage Circuit"
        elif network_id in geometric_matched_network_ids:
            status = "geometric"
            tooltip_status = "Geometric Match"
        elif network_id in regular_matched_network_ids:
            status = "matched"
            tooltip_status = "Matched"
        else:
            status = "unmatched"
            tooltip_status = "Unmatched"

        feature = {
            "type": "Feature",
            "id": line_id,
            "properties": {
                "type": "network",
                "id": network_id,
                "voltage": int(row['v_nom']),
                "status": status,
                "tooltip": f"Network: {row['id']} ({row['v_nom']} kV) - {tooltip_status}"
            },
            "geometry": {
                "type": "LineString",
                "coordinates": [[float(x), float(y)] for x, y in coords]
            }
        }
        network_features.append(feature)

    # Create GeoJSON collections
    jao_collection = {"type": "FeatureCollection", "features": jao_features}
    network_collection = {"type": "FeatureCollection", "features": network_features}

    # Convert to JSON strings
    jao_json = json.dumps(jao_collection)
    network_json = json.dumps(network_collection)

    # Make sure output_dir exists
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    # Create a complete HTML file from scratch
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>JAO-Network Line Matching Results</title>

        <link rel="stylesheet" href="https://unpkg.com/leaflet@1.7.1/dist/leaflet.css" />
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css">
        <link rel="stylesheet" href="https://ppete2.github.io/Leaflet.PolylineMeasure/Leaflet.PolylineMeasure.css" />

        <script src="https://unpkg.com/leaflet@1.7.1/dist/leaflet.js"></script>
        <script src="https://ppete2.github.io/Leaflet.PolylineMeasure/Leaflet.PolylineMeasure.js"></script>

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

            .leaflet-control-polylinemeasure {{
                background-color: white !important;
                padding: 4px !important;
                border-radius: 4px !important;
            }}

            .stats-box {{
                margin-top: 15px;
                background-color: #f9f9f9;
                border: 1px solid #ddd;
                border-radius: 4px;
                padding: 10px;
                font-size: 12px;
            }}

            .stats-item {{
                margin-bottom: 5px;
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
                        <div class="filter-option active" data-filter="jao-matched">Matched JAO Lines</div>
                        <div class="filter-option active" data-filter="jao-geometric">Geometric Match JAO</div>
                        <div class="filter-option active" data-filter="jao-parallel">Parallel Circuit JAO</div>
                        <div class="filter-option active" data-filter="jao-parallel-voltage">Parallel Voltage JAO</div>
                        <div class="filter-option active" data-filter="jao-duplicate">Duplicate JAO</div>
                        <div class="filter-option active" data-filter="jao-unmatched">Unmatched JAO Lines</div>
                        <div class="filter-option active" data-filter="network-matched">Matched Network Lines</div>
                        <div class="filter-option active" data-filter="network-geometric">Geometric Network</div>
                        <div class="filter-option active" data-filter="network-parallel">Parallel Circuit Network</div>
                        <div class="filter-option active" data-filter="network-parallel-voltage">Parallel Voltage Network</div>
                        <div class="filter-option active" data-filter="network-duplicate">Duplicate Network</div>
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
                        <div>Regular Matched JAO Lines</div>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background-color: #00BFFF;"></div>
                        <div>Geometric Match JAO Lines</div>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background-color: #9932CC;"></div>
                        <div>Parallel Circuit JAO Lines</div>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background-color: #FF8C00;"></div>
                        <div>Parallel Voltage Circuit JAO</div>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background-color: #DA70D6;"></div>
                        <div>Duplicate JAO Lines</div>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background-color: red;"></div>
                        <div>Unmatched JAO Lines</div>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background-color: purple;"></div>
                        <div>Regular Matched Network Lines</div>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background-color: #1E90FF;"></div>
                        <div>Geometric Match Network Lines</div>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background-color: #8B008B;"></div>
                        <div>Parallel Circuit Network Lines</div>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background-color: #FF4500;"></div>
                        <div>Parallel Voltage Network Lines</div>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background-color: #FF00FF;"></div>
                        <div>Duplicate Network Lines</div>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background-color: blue;"></div>
                        <div>Unmatched Network Lines</div>
                    </div>
                </div>
            </div>

            <div class="control-section">
                <h3><i class="fas fa-ruler"></i> Measurement Tool</h3>
                <p>Click the ruler icon on the map to measure distances.</p>
            </div>

            <div class="stats-box">
                <h3><i class="fas fa-chart-pie"></i> Statistics</h3>
                <div class="stats-item">JAO Lines: {len(jao_gdf)} total</div>
                <div class="stats-item">- Matched: {len(matched_jao_ids)} ({len(matched_jao_ids) / len(jao_gdf) * 100:.1f}%)</div>
                <div class="stats-item">- Geometric: {len(geometric_match_jao_ids)} ({len(geometric_match_jao_ids) / len(jao_gdf) * 100:.1f}%)</div>
                <div class="stats-item">- Parallel Circuit: {len(parallel_circuit_jao_ids)} ({len(parallel_circuit_jao_ids) / len(jao_gdf) * 100:.1f}%)</div>
                <div class="stats-item">- Parallel Voltage: {len(parallel_voltage_jao_ids)} ({len(parallel_voltage_jao_ids) / len(jao_gdf) * 100:.1f}%)</div>
                <div class="stats-item">- Duplicate: {len(duplicate_jao_ids)} ({len(duplicate_jao_ids) / len(jao_gdf) * 100:.1f}%)</div>
                <div class="stats-item">- Unmatched: {len(jao_gdf) - len(matched_jao_ids) - len(geometric_match_jao_ids) - len(parallel_circuit_jao_ids) - len(parallel_voltage_jao_ids) - len(duplicate_jao_ids)} ({(len(jao_gdf) - len(matched_jao_ids) - len(geometric_match_jao_ids) - len(parallel_circuit_jao_ids) - len(parallel_voltage_jao_ids) - len(duplicate_jao_ids)) / len(jao_gdf) * 100:.1f}%)</div>
                <hr>
                <div class="stats-item">Network Lines: {len(network_gdf)} total</div>
                <div class="stats-item">- Matched: {len(regular_matched_network_ids)} ({len(regular_matched_network_ids) / len(network_gdf) * 100:.1f}%)</div>
                <div class="stats-item">- Unmatched: {len(network_gdf) - len(regular_matched_network_ids) - len(geometric_matched_network_ids) - len(parallel_circuit_network_ids) - len(parallel_voltage_network_ids) - len(duplicate_network_ids)} ({(len(network_gdf) - len(regular_matched_network_ids) - len(geometric_matched_network_ids) - len(parallel_circuit_network_ids) - len(parallel_voltage_network_ids) - len(duplicate_network_ids)) / len(network_gdf) * 100:.1f}%)</div>
            </div>
        </div>

        <script>
            // Initialize the map
            var map = L.map('map').setView([51.1657, 10.4515], 6);

            // Add base tile layer
            L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
                attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
            }}).addTo(map);

            // Add measurement tool
            L.control.polylineMeasure({{
                position: 'topleft',
                unit: 'metres',
                showBearings: true,
                clearMeasurementsOnStop: false,
                showClearControl: true,
                showUnitControl: true
            }}).addTo(map);

            // Load the GeoJSON data - DIRECTLY EMBEDDED IN THE HTML
            var jaoLines = {jao_json};
            var networkLines = {network_json};

            // Define styling for the JAO lines
            function jaoStyle(feature) {{
                switch(feature.properties.status) {{
                    case "matched":
                        return {{
                            "color": "green",
                            "weight": 3,
                            "opacity": 0.8
                        }};
                    case "geometric":
                        return {{
                            "color": "#00BFFF", // Deep Sky Blue
                            "weight": 3,
                            "opacity": 0.8,
                            "dashArray": "10, 5"
                        }};
                    case "parallel":
                        return {{
                            "color": "#9932CC", // Dark Orchid
                            "weight": 3,
                            "opacity": 0.8,
                            "dashArray": "5, 5"
                        }};
                    case "parallel_voltage":
                        return {{
                            "color": "#FF8C00", // Dark Orange
                            "weight": 3,
                            "opacity": 0.8,
                            "dashArray": "10, 2, 2, 2"
                        }};
                    case "duplicate":
                        return {{
                            "color": "#DA70D6", // Orchid
                            "weight": 3,
                            "opacity": 0.8,
                            "dashArray": "2, 5"
                        }};
                    default: // unmatched
                        return {{
                            "color": "red",
                            "weight": 3,
                            "opacity": 0.8
                        }};
                }}
            }};

            // Define styling for network lines
            function networkStyle(feature) {{
                switch(feature.properties.status) {{
                    case "matched":
                        return {{
                            "color": "purple",
                            "weight": 2,
                            "opacity": 0.6
                        }};
                    case "geometric":
                        return {{
                            "color": "#1E90FF", // Dodger Blue
                            "weight": 2,
                            "opacity": 0.6,
                            "dashArray": "10, 5"
                        }};
                    case "parallel":
                        return {{
                            "color": "#8B008B", // Dark Magenta
                            "weight": 2,
                            "opacity": 0.6,
                            "dashArray": "5, 5"
                        }};
                    case "parallel_voltage":
                        return {{
                            "color": "#FF4500", // Orange Red
                            "weight": 2,
                            "opacity": 0.6,
                            "dashArray": "10, 2, 2, 2"
                        }};
                    case "duplicate":
                        return {{
                            "color": "#FF00FF", // Magenta
                            "weight": 2,
                            "opacity": 0.6,
                            "dashArray": "2, 5"
                        }};
                    default: // unmatched
                        return {{
                            "color": "blue",
                            "weight": 2,
                            "opacity": 0.6
                        }};
                }}
            }};

            // Create the JAO layers
            var jaoMatchedLayer = L.geoJSON(jaoLines, {{
                filter: function(feature) {{
                    return feature.properties.status === "matched";
                }},
                style: jaoStyle,
                onEachFeature: function(feature, layer) {{
                    layer.bindTooltip(feature.properties.tooltip);
                    layer.on('click', function() {{
                        highlightFeature(feature.id);
                    }});
                }}
            }}).addTo(map);

            var jaoGeometricLayer = L.geoJSON(jaoLines, {{
                filter: function(feature) {{
                    return feature.properties.status === "geometric";
                }},
                style: jaoStyle,
                onEachFeature: function(feature, layer) {{
                    layer.bindTooltip(feature.properties.tooltip);
                    layer.on('click', function() {{
                        highlightFeature(feature.id);
                    }});
                }}
            }}).addTo(map);

            var jaoParallelLayer = L.geoJSON(jaoLines, {{
                filter: function(feature) {{
                    return feature.properties.status === "parallel";
                }},
                style: jaoStyle,
                onEachFeature: function(feature, layer) {{
                    layer.bindTooltip(feature.properties.tooltip);
                    layer.on('click', function() {{
                        highlightFeature(feature.id);
                    }});
                }}
            }}).addTo(map);

            var jaoParallelVoltageLayer = L.geoJSON(jaoLines, {{
                filter: function(feature) {{
                    return feature.properties.status === "parallel_voltage";
                }},
                style: jaoStyle,
                onEachFeature: function(feature, layer) {{
                    layer.bindTooltip(feature.properties.tooltip);
                    layer.on('click', function() {{
                        highlightFeature(feature.id);
                    }});
                }}
            }}).addTo(map);

            var jaoDuplicateLayer = L.geoJSON(jaoLines, {{
                filter: function(feature) {{
                    return feature.properties.status === "duplicate";
                }},
                style: jaoStyle,
                onEachFeature: function(feature, layer) {{
                    layer.bindTooltip(feature.properties.tooltip);
                    layer.on('click', function() {{
                        highlightFeature(feature.id);
                    }});
                }}
            }}).addTo(map);

            var jaoUnmatchedLayer = L.geoJSON(jaoLines, {{
                filter: function(feature) {{
                    return feature.properties.status === "unmatched";
                }},
                style: jaoStyle,
                onEachFeature: function(feature, layer) {{
                    layer.bindTooltip(feature.properties.tooltip);
                    layer.on('click', function() {{
                        highlightFeature(feature.id);
                    }});
                }}
            }}).addTo(map);

            // Create the network layers
            var networkMatchedLayer = L.geoJSON(networkLines, {{
                filter: function(feature) {{
                    return feature.properties.status === "matched";
                }},
                style: networkStyle,
                onEachFeature: function(feature, layer) {{
                    layer.bindTooltip(feature.properties.tooltip);
                    layer.on('click', function() {{
                        highlightFeature(feature.id);
                    }});
                }}
            }}).addTo(map);

            var networkGeometricLayer = L.geoJSON(networkLines, {{
                filter: function(feature) {{
                    return feature.properties.status === "geometric";
                }},
                style: networkStyle,
                onEachFeature: function(feature, layer) {{
                    layer.bindTooltip(feature.properties.tooltip);
                    layer.on('click', function() {{
                        highlightFeature(feature.id);
                    }});
                }}
            }}).addTo(map);

            var networkParallelLayer = L.geoJSON(networkLines, {{
                filter: function(feature) {{
                    return feature.properties.status === "parallel";
                }},
                style: networkStyle,
                onEachFeature: function(feature, layer) {{
                    layer.bindTooltip(feature.properties.tooltip);
                    layer.on('click', function() {{
                        highlightFeature(feature.id);
                    }});
                }}
            }}).addTo(map);

            var networkParallelVoltageLayer = L.geoJSON(networkLines, {{
                filter: function(feature) {{
                    return feature.properties.status === "parallel_voltage";
                }},
                style: networkStyle,
                onEachFeature: function(feature, layer) {{
                    layer.bindTooltip(feature.properties.tooltip);
                    layer.on('click', function() {{
                        highlightFeature(feature.id);
                    }});
                }}
            }}).addTo(map);

            var networkDuplicateLayer = L.geoJSON(networkLines, {{
                filter: function(feature) {{
                    return feature.properties.status === "duplicate";
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
                    return feature.properties.status === "unmatched";
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

                // Apply highlight to the specific feature across all layers
                function checkAndHighlight(layer) {{
                    layer.eachLayer(function(leafletLayer) {{
                        if (leafletLayer.feature.id === id || (leafletLayer.feature.properties.id === id)) {{
                            leafletLayer.setStyle({{className: 'highlighted'}});
                            if (leafletLayer._path) leafletLayer._path.classList.add('highlighted');

                            // Center on the highlighted feature
                            var bounds = leafletLayer.getBounds();
                            map.fitBounds(bounds, {{ padding: [50, 50] }});
                        }}
                    }});
                }}

                // Check all layers
                checkAndHighlight(jaoMatchedLayer);
                checkAndHighlight(jaoGeometricLayer);
                checkAndHighlight(jaoParallelLayer);
                checkAndHighlight(jaoParallelVoltageLayer);
                checkAndHighlight(jaoDuplicateLayer);
                checkAndHighlight(jaoUnmatchedLayer);
                checkAndHighlight(networkMatchedLayer);
                checkAndHighlight(networkGeometricLayer);
                checkAndHighlight(networkParallelLayer);
                checkAndHighlight(networkParallelVoltageLayer);
                checkAndHighlight(networkDuplicateLayer);
                checkAndHighlight(networkUnmatchedLayer);
            }}

            // Function to clear highlights
            function clearHighlights() {{
                function resetStyle(layer) {{
                    layer.eachLayer(function(leafletLayer) {{
                        if (leafletLayer.feature.properties.type === 'jao') {{
                            leafletLayer.setStyle(jaoStyle(leafletLayer.feature));
                        }} else {{
                            leafletLayer.setStyle(networkStyle(leafletLayer.feature));
                        }}
                        if (leafletLayer._path) leafletLayer._path.classList.remove('highlighted');
                    }});
                }}

                // Reset all layers
                resetStyle(jaoMatchedLayer);
                resetStyle(jaoGeometricLayer);
                resetStyle(jaoParallelLayer);
                resetStyle(jaoParallelVoltageLayer);
                resetStyle(jaoDuplicateLayer);
                resetStyle(jaoUnmatchedLayer);
                resetStyle(networkMatchedLayer);
                resetStyle(networkGeometricLayer);
                resetStyle(networkParallelLayer);
                resetStyle(networkParallelVoltageLayer);
                resetStyle(networkDuplicateLayer);
                resetStyle(networkUnmatchedLayer);
            }}

            // Function to create the search index
            function createSearchIndex() {{
                var searchData = [];

                // Add JAO lines to search data
                jaoLines.features.forEach(function(feature) {{
                    searchData.push({{
                        id: feature.id,
                        text: "JAO: " + feature.properties.id + " - " + feature.properties.name + " (" + feature.properties.voltage + " kV) - " + feature.properties.status.charAt(0).toUpperCase() + feature.properties.status.slice(1),
                        type: "jao",
                        feature: feature
                    }});
                }});

                // Add network lines to search data
                networkLines.features.forEach(function(feature) {{
                    searchData.push({{
                        id: feature.id,
                        text: "Network: " + feature.properties.id + " (" + feature.properties.voltage + " kV) - " + feature.properties.status.charAt(0).toUpperCase() + feature.properties.status.slice(1),
                        type: "network",
                        feature: feature
                    }});
                }});

                // Add relationship references to search data
                // This lets users search for "which network lines are used for JAO X"
                var jaoToNetworkMap = {{}};

                // Build a map of JAO ID to network IDs from the matching results
                {json.dumps([{
        'jao_id': r['jao_id'],
        'network_ids': r.get('network_ids', []),
        'type': 'duplicate' if r.get('is_duplicate', False) else
        'geometric' if r.get('is_geometric_match', False) else
        'parallel' if r.get('is_parallel_circuit', False) else
        'parallel_voltage' if r.get('is_parallel_voltage_circuit', False) else 'regular'
    } for r in matching_results if r['matched'] and r.get('network_ids')])}.forEach(function(match) {{
                    jaoToNetworkMap[match.jao_id] = {{
                        network_ids: match.network_ids,
                        type: match.type
                    }};

                    // Add an entry for searching by JAO ID to find all network lines
                    searchData.push({{
                        id: "ref_jao_" + match.jao_id,
                        text: "Network lines for JAO " + match.jao_id + " (" + match.type + " match): " + match.network_ids.join(", "),
                        type: "reference",
                        jaoId: match.jao_id,
                        networkIds: match.network_ids,
                        matchType: match.type
                    }});

                    // For each network ID, add an entry for searching by network ID to find the JAO
                    match.network_ids.forEach(function(networkId) {{
                        searchData.push({{
                            id: "ref_network_" + networkId,
                            text: "Network line " + networkId + " is used for JAO " + match.jao_id + " (" + match.type + " match)",
                            type: "reference",
                            jaoId: match.jao_id,
                            networkIds: [networkId],
                            matchType: match.type
                        }});
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

                // Function to apply voltage filter to a layer
                function applyVoltageFilter(layer) {{
                    if (!allVoltages) {{
                        layer.eachLayer(function(leafletLayer) {{
                            var visible = activeVoltages.includes(leafletLayer.feature.properties.voltage);
                            if (visible) {{
                                if (leafletLayer._path) leafletLayer._path.style.display = "block";
                            }} else {{
                                if (leafletLayer._path) leafletLayer._path.style.display = "none";
                            }}
                        }});
                    }}
                }}

                // Show/hide JAO matched layer
                if (activeLineTypes.includes("jao-matched")) {{
                    jaoMatchedLayer.addTo(map);
                    applyVoltageFilter(jaoMatchedLayer);
                }} else {{
                    map.removeLayer(jaoMatchedLayer);
                }}

                // Show/hide JAO geometric layer
                if (activeLineTypes.includes("jao-geometric")) {{
                    jaoGeometricLayer.addTo(map);
                    applyVoltageFilter(jaoGeometricLayer);
                }} else {{
                    map.removeLayer(jaoGeometricLayer);
                }}

                // Show/hide JAO parallel layer
                if (activeLineTypes.includes("jao-parallel")) {{
                    jaoParallelLayer.addTo(map);
                    applyVoltageFilter(jaoParallelLayer);
                }} else {{
                    map.removeLayer(jaoParallelLayer);
                }}

                // Show/hide JAO parallel voltage layer
                if (activeLineTypes.includes("jao-parallel-voltage")) {{
                    jaoParallelVoltageLayer.addTo(map);
                    applyVoltageFilter(jaoParallelVoltageLayer);
                }} else {{
                    map.removeLayer(jaoParallelVoltageLayer);
                }}

                // Show/hide JAO duplicate layer
                if (activeLineTypes.includes("jao-duplicate")) {{
                    jaoDuplicateLayer.addTo(map);
                    applyVoltageFilter(jaoDuplicateLayer);
                }} else {{
                    map.removeLayer(jaoDuplicateLayer);
                }}

                // Show/hide JAO unmatched layer
                if (activeLineTypes.includes("jao-unmatched")) {{
                    jaoUnmatchedLayer.addTo(map);
                    applyVoltageFilter(jaoUnmatchedLayer);
                }} else {{
                    map.removeLayer(jaoUnmatchedLayer);
                }}

                // Show/hide Network matched layer
                if (activeLineTypes.includes("network-matched")) {{
                    networkMatchedLayer.addTo(map);
                    applyVoltageFilter(networkMatchedLayer);
                }} else {{
                    map.removeLayer(networkMatchedLayer);
                }}

                // Show/hide Network geometric layer
                if (activeLineTypes.includes("network-geometric")) {{
                    networkGeometricLayer.addTo(map);
                    applyVoltageFilter(networkGeometricLayer);
                }} else {{
                    map.removeLayer(networkGeometricLayer);
                }}

                // Show/hide Network parallel layer
                if (activeLineTypes.includes("network-parallel")) {{
                    networkParallelLayer.addTo(map);
                    applyVoltageFilter(networkParallelLayer);
                }} else {{
                    map.removeLayer(networkParallelLayer);
                }}

                // Show/hide Network parallel voltage layer
                if (activeLineTypes.includes("network-parallel-voltage")) {{
                    networkParallelVoltageLayer.addTo(map);
                    applyVoltageFilter(networkParallelVoltageLayer);
                }} else {{
                    map.removeLayer(networkParallelVoltageLayer);
                }}

                // Show/hide Network duplicate layer
                if (activeLineTypes.includes("network-duplicate")) {{
                    networkDuplicateLayer.addTo(map);
                    applyVoltageFilter(networkDuplicateLayer);
                }} else {{
                    map.removeLayer(networkDuplicateLayer);
                }}

                // Show/hide Network unmatched layer
                if (activeLineTypes.includes("network-unmatched")) {{
                    networkUnmatchedLayer.addTo(map);
                    applyVoltageFilter(networkUnmatchedLayer);
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
                            // Handle search result click based on type
                            if (result.type === "reference") {{
                                // For references, highlight both the JAO and network lines
                                if (result.jaoId) {{
                                    highlightFeature("jao_" + result.jaoId);
                                }}

                                // Also highlight the first network line (if there are many)
                                if (result.networkIds && result.networkIds.length > 0) {{
                                    setTimeout(function() {{
                                        highlightFeature("network_" + result.networkIds[0]);
                                    }}, 1000); // Highlight network line after a delay
                                }}
                            }} else {{
                                // For regular JAO or network features, just highlight that feature
                                var coords = getCenterOfFeature(result.feature);
                                map.setView([coords[1], coords[0]], 10);
                                highlightFeature(result.id);
                            }}

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
    output_file = output_dir / 'jao_network_matching_results.html'
    with open(output_file, 'w') as f:
        f.write(html)

    return output_file


def match_remaining_unmatched_network_lines(matching_results, jao_gdf, network_gdf):
    """
    Final pass to match any remaining unmatched network lines to appropriate JAO lines.
    This ensures maximum coverage of network lines in the matching process.
    """
    print("\n=== MATCHING REMAINING UNMATCHED NETWORK LINES ===")

    # Define helper function for calculating direction cosine
    def dir_cos(geom1, geom2):
        """Calculate the direction cosine between two geometries (how parallel they are)."""
        try:
            import numpy as np
            from shapely.geometry import LineString, MultiLineString

            # Helper to get the main vector of a line geometry
            def main_vec(geom):
                if geom.geom_type == "MultiLineString":
                    # Use the longest component of a MultiLineString
                    geom = max(geom.geoms, key=lambda g: g.length)

                # Get coordinates as numpy array
                coords = np.array(list(geom.coords))

                # Create vector from first to last point
                vec = coords[-1] - coords[0]

                # Normalize the vector
                norm = np.linalg.norm(vec)
                if norm > 0:
                    return vec / norm
                return vec

            # Get main direction vectors for both geometries
            vec1 = main_vec(geom1)
            vec2 = main_vec(geom2)

            # Calculate absolute dot product (cosine of angle between vectors)
            # Absolute value because we don't care about direction, just alignment
            return float(abs(np.dot(vec1, vec2)))
        except Exception as e:
            print(f"Error calculating direction cosine: {e}")
            return 0.0

    # 1. Identify all network lines that are already used in matches
    used_network_ids = set()
    for result in matching_results:
        if result['matched'] and 'network_ids' in result:
            for network_id in result['network_ids']:
                used_network_ids.add(str(network_id))

    # 2. Get all unmatched network lines
    unmatched_network_lines = []
    for idx, row in network_gdf.iterrows():
        network_id = str(row['id'])
        if network_id not in used_network_ids:
            unmatched_network_lines.append({
                'id': network_id,
                'idx': idx,
                'geometry': row.geometry,
                'voltage': int(row['v_nom']),
                'length': calculate_length_meters(row.geometry)
            })

    print(f"Found {len(unmatched_network_lines)} unmatched network lines to process")

    # If no unmatched lines, nothing to do
    if not unmatched_network_lines:
        return matching_results

    # 3. Create a dictionary of all JAO lines with their geometries
    jao_geometries = {}
    for idx, row in jao_gdf.iterrows():
        jao_id = str(row['id'])
        jao_geometries[jao_id] = {
            'geometry': row.geometry,
            'voltage': int(row['v_nom']),
            'idx': idx
        }

    # 4. Process each unmatched network line
    improvements_made = 0
    for network_line in unmatched_network_lines:
        network_id = network_line['id']
        network_geom = network_line['geometry']
        network_voltage = network_line['voltage']

        # Find the best JAO match for this network line
        best_match = None
        best_score = 0.3  # Minimum threshold score

        for jao_id, jao_info in jao_geometries.items():
            jao_geom = jao_info['geometry']
            jao_voltage = jao_info['voltage']

            # Check voltage compatibility (with 380/400 equivalence)
            voltage_match = False
            if (network_voltage == 220 and jao_voltage == 220) or \
                    (network_voltage in [380, 400] and jao_voltage in [380, 400]):
                voltage_match = True
                voltage_factor = 1.0
            else:
                voltage_factor = 0.5  # Penalty for voltage mismatch

            # Calculate geometric score
            # 1. Buffer overlap
            coverage = calculate_geometry_coverage(jao_geom, network_geom, buffer_meters=1000)

            # 2. Direction alignment
            try:
                alignment = dir_cos(jao_geom, network_geom)
            except Exception as e:
                print(f"Error calculating alignment: {e}")
                alignment = 0

            # 3. Calculate combined score
            score = (0.7 * coverage + 0.3 * alignment) * voltage_factor

            # If this score is better than our current best, update
            if score > best_score:
                # Find the corresponding match result
                match_result = None
                for result in matching_results:
                    if result['jao_id'] == jao_id:
                        match_result = result
                        break

                # If no match exists, we'll need to create one
                best_match = {
                    'jao_id': jao_id,
                    'score': score,
                    'coverage': coverage,
                    'alignment': alignment,
                    'match_result': match_result,
                    'voltage_match': voltage_match,
                    'jao_idx': jao_info['idx']
                }
                best_score = score

        # If we found a good match, update the matching results
        if best_match:
            print(f"  Network line {network_id} ({network_voltage} kV) matches with JAO {best_match['jao_id']}")
            print(
                f"    Score: {best_match['score']:.3f}, Coverage: {best_match['coverage']:.2f}, Alignment: {best_match['alignment']:.2f}")

            # Get or create match result
            match_result = best_match['match_result']

            # If no existing match, create one
            if match_result is None:
                # Get JAO info for the match
                jao_row = jao_gdf.iloc[best_match['jao_idx']]
                jao_length = calculate_length_meters(jao_row.geometry)

                match_result = {
                    'jao_id': best_match['jao_id'],
                    'jao_name': str(jao_row['NE_name']),
                    'v_nom': int(jao_row['v_nom']),
                    'matched': True,
                    'is_geometric_match': True,
                    'network_ids': [],
                    'jao_length': float(jao_length),
                    'path_length': 0,
                    'match_quality': 'Geometric Match (final pass)'
                }
                matching_results.append(match_result)
                print(f"    Created new match for previously unmatched JAO {best_match['jao_id']}")

            # Add network line to match
            if 'network_ids' not in match_result:
                match_result['network_ids'] = []

            match_result['network_ids'].append(network_id)

            # Update path length
            if 'path_length' not in match_result:
                match_result['path_length'] = 0

            match_result['path_length'] = float(match_result['path_length'] + network_line['length'])

            # Update length ratio
            if 'jao_length' in match_result and match_result['jao_length'] > 0:
                match_result['length_ratio'] = float(match_result['path_length'] / match_result['jao_length'])

            # If not already marked as a specific match type, mark as geometric
            if not match_result.get('is_duplicate', False) and \
                    not match_result.get('is_parallel_circuit', False) and \
                    not match_result.get('is_parallel_voltage_circuit', False):
                match_result['is_geometric_match'] = True

            # If a new match or previously unmatched, mark it matched
            match_result['matched'] = True

            # If voltage mismatch, update match quality
            if not best_match['voltage_match'] and 'voltage mismatch' not in match_result.get('match_quality', ''):
                match_result['match_quality'] += ' (voltage mismatch)'

            improvements_made += 1
        else:
            print(f"  No good match found for network line {network_id}")

    print(f"Added {improvements_made} network lines to matches")

    return matching_results


def improve_visualization_of_unmatched_network_lines(html_content):
    """
    Improve the visibility of unmatched network lines in the visualization.
    """
    # Change the styling for unmatched network lines to make them more visible
    improved_style = """
        case "unmatched":
            return {
                "color": "#FF0000", // Bright red
                "weight": 3,        // Thicker line
                "opacity": 0.8,     // Higher opacity
                "dashArray": "5,5"  // Dashed pattern
            };
    """

    # Replace the existing style for unmatched network lines
    html_content = html_content.replace(
        """default: // unmatched
            return {
                "color": "blue",
                "weight": 2,
                "opacity": 0.6
            };""",
        improved_style
    )

    return html_content


def repair_network_graph(G, network_gdf, connection_threshold_meters=50):
    """
    Repair the network graph by adding missing connections between nearby endpoints.
    This helps address cases where network lines should be connected but aren't due to
    small coordinate differences.

    Parameters:
    - G: The network graph
    - network_gdf: The network GeoDataFrame
    - connection_threshold_meters: Maximum distance in meters to connect endpoints

    Returns:
    - Repaired graph
    """
    print("\nRepairing network graph by adding connections between nearby endpoints...")

    # Get all node positions
    positions = nx.get_node_attributes(G, 'pos')
    nodes = list(positions.keys())

    # Create spatial index for nodes
    from rtree import index
    node_idx = index.Index()

    # Store node IDs and positions for lookup
    node_positions = {}

    for i, node_id in enumerate(nodes):
        pos = positions[node_id]
        node_idx.insert(i, (pos[0], pos[1], pos[0], pos[1]))
        node_positions[i] = (node_id, pos)

    # Count added connections
    connections_added = 0

    # For each node, find nearby nodes that should be connected
    for i, node_id in enumerate(nodes):
        # Skip connector nodes (not actual line endpoints)
        if '_start' not in node_id and '_end' not in node_id:
            continue

        pos = positions[node_id]

        # Get line ID from node ID
        line_id = node_id.split('_')[1]

        # Find nearby nodes
        for j in node_idx.nearest((pos[0], pos[1], pos[0], pos[1]), 10):
            nearby_id, nearby_pos = node_positions[j]

            # Skip self or nodes from the same line
            if nearby_id == node_id or nearby_id.split('_')[1] == line_id:
                continue

            # Skip if not a line endpoint
            if '_start' not in nearby_id and '_end' not in nearby_id:
                continue

            # Calculate distance in approximate meters
            import math
            avg_lat = (pos[1] + nearby_pos[1]) / 2
            meters_per_degree = 111111 * math.cos(math.radians(abs(avg_lat)))
            dist_degrees = ((pos[0] - nearby_pos[0]) ** 2 + (pos[1] - nearby_pos[1]) ** 2) ** 0.5
            dist_meters = dist_degrees * meters_per_degree

            # If nodes are close but not connected, add a connection
            if dist_meters <= connection_threshold_meters and not G.has_edge(node_id, nearby_id):
                # This is a connector edge, not a real network line
                G.add_edge(node_id, nearby_id, weight=float(dist_degrees), connector=True)
                connections_added += 1

    print(f"Added {connections_added} new connections to repair the graph")
    return G


def convert_geometric_to_path_matches(matching_results, G, jao_gdf, network_gdf, nearest_points_dict):
    """
    Check geometric matches to see if they can be converted to path-based matches.
    """
    print("\nChecking if geometric matches can be converted to path-based matches...")

    conversions = 0

    for result in matching_results:
        # Only check geometric matches
        if not result.get('matched', False) or not result.get('is_geometric_match', False):
            continue

        jao_id = result['jao_id']
        print(f"  Checking JAO {jao_id}")

        # Find the JAO in the dataframe
        jao_rows = jao_gdf[jao_gdf['id'].astype(str) == jao_id]
        if jao_rows.empty:
            continue

        jao_idx = jao_rows.index[0]

        # Check if we have endpoint matches
        if jao_idx not in nearest_points_dict:
            continue

        if nearest_points_dict[jao_idx]['start_nearest'] is None or nearest_points_dict[jao_idx]['end_nearest'] is None:
            continue

        # Get node IDs for start and end points
        start_idx, start_pos = nearest_points_dict[jao_idx]['start_nearest']
        end_idx, end_pos = nearest_points_dict[jao_idx]['end_nearest']

        start_node = f"node_{start_idx}_{start_pos}"
        end_node = f"node_{end_idx}_{end_pos}"

        # Check if there's a path between the nodes
        try:
            path = nx.shortest_path(G, start_node, end_node, weight='weight')

            # Extract network lines in the path
            network_ids = []
            for i in range(len(path) - 1):
                edge_data = G.get_edge_data(path[i], path[i + 1])
                if edge_data and 'id' in edge_data and not edge_data.get('connector', False):
                    network_ids.append(edge_data['id'])

            # Check if the path contains all network lines that were matched geometrically
            existing_network_ids = set(result.get('network_ids', []))
            path_network_ids = set(network_ids)

            # If the path contains at least 50% of the geometric matches
            overlap = len(existing_network_ids.intersection(path_network_ids))
            if overlap / len(existing_network_ids) >= 0.5:
                print(f"    Converting geometric match to path-based match")
                result['is_geometric_match'] = False
                result['match_quality'] = 'Converted to Path-based Match'
                result['path'] = [str(p) for p in path]

                # Update network IDs to include both sets
                result['network_ids'] = list(existing_network_ids.union(path_network_ids))

                # Update path length
                path_length = 0
                for network_id in result['network_ids']:
                    network_rows = network_gdf[network_gdf['id'].astype(str) == network_id]
                    if not network_rows.empty:
                        path_length += calculate_length_meters(network_rows.iloc[0].geometry)

                result['path_length'] = float(path_length)

                # Update length ratio
                if 'jao_length' in result and result['jao_length'] > 0:
                    result['length_ratio'] = float(result['path_length'] / result['jao_length'])

                conversions += 1

        except nx.NetworkXNoPath:
            print(f"    No path found between endpoints, keeping as geometric match")

    print(f"Converted {conversions} geometric matches to path-based matches")
    return matching_results


def match_remaining_parallel_network_lines(matching_results, jao_gdf, network_gdf):
    """
    Match remaining unmatched network lines that are likely part of parallel circuits.
    This specifically targets network lines that align with JAO lines but weren't matched
    by previous algorithms.
    """
    print("\n=== MATCHING REMAINING PARALLEL NETWORK LINES ===")

    # Helper function to calculate direction cosine
    def dir_cos(geom1, geom2):
        """Calculate the direction cosine between two geometries (how parallel they are)."""
        try:
            import numpy as np
            from shapely.geometry import LineString, MultiLineString

            # Helper to get the main vector of a line geometry
            def main_vec(geom):
                if geom.geom_type == "MultiLineString":
                    # Use the longest component of a MultiLineString
                    geom = max(geom.geoms, key=lambda g: g.length)

                # Get coordinates as numpy array
                coords = np.array(list(geom.coords))

                # Create vector from first to last point
                vec = coords[-1] - coords[0]

                # Normalize the vector
                norm = np.linalg.norm(vec)
                if norm > 0:
                    return vec / norm
                return vec

            # Get main direction vectors for both geometries
            vec1 = main_vec(geom1)
            vec2 = main_vec(geom2)

            # Calculate absolute dot product (cosine of angle between vectors)
            # Absolute value because we don't care about direction, just alignment
            return float(abs(np.dot(vec1, vec2)))
        except Exception as e:
            print(f"Error calculating direction cosine: {e}")
            return 0.0

    # 1. Identify all network lines that are already used in matches
    used_network_ids = set()
    for result in matching_results:
        if result['matched'] and 'network_ids' in result:
            for network_id in result['network_ids']:
                used_network_ids.add(str(network_id))

    # 2. Get all unmatched network lines
    unmatched_network_lines = []
    for idx, row in network_gdf.iterrows():
        network_id = str(row['id'])
        if network_id not in used_network_ids:
            unmatched_network_lines.append({
                'id': network_id,
                'idx': idx,
                'geometry': row.geometry,
                'voltage': int(row['v_nom']),
                'length': calculate_length_meters(row.geometry)
            })

    print(f"Found {len(unmatched_network_lines)} unmatched network lines to process")

    # 3. For each matched JAO line, look for unmatched network lines that align with it
    additions_made = 0
    newly_matched_network_ids = set()

    # Process all matched JAO lines - prioritize lines already marked as parallel circuits
    matched_jaos = []
    for result in matching_results:
        if result['matched'] and not result.get('is_duplicate', False):
            matched_jaos.append(result)

    # Sort to prioritize processing parallel circuits first
    matched_jaos.sort(key=lambda x: x.get('is_parallel_circuit', False), reverse=True)

    for result in matched_jaos:
        jao_id = result['jao_id']

        # Get the JAO geometry
        jao_rows = jao_gdf[jao_gdf['id'].astype(str) == jao_id]
        if jao_rows.empty:
            continue

        jao_row = jao_rows.iloc[0]
        jao_geom = jao_row.geometry
        jao_voltage = int(jao_row['v_nom'])

        # If this result already has network lines, get their geometries to avoid similar ones
        existing_network_geoms = []
        if 'network_ids' in result and result['network_ids']:
            for network_id in result['network_ids']:
                network_rows = network_gdf[network_gdf['id'].astype(str) == network_id]
                if not network_rows.empty:
                    existing_network_geoms.append(network_rows.iloc[0].geometry)

        # Create a buffer around the JAO for matching
        avg_lat = jao_geom.centroid.y
        meters_per_degree = 111111 * np.cos(np.radians(abs(avg_lat)))
        # Use a larger buffer for parallel circuit detection
        buffer_deg = 1000 / meters_per_degree  # 1km buffer
        jao_buffer = jao_geom.buffer(buffer_deg)

        # Find potential matching network lines
        candidate_matches = []

        for network_line in unmatched_network_lines:
            # Skip if already newly matched
            if network_line['id'] in newly_matched_network_ids:
                continue

            # Check voltage compatibility (with relaxed 380/400 equivalence)
            voltage_match = False
            if (jao_voltage == 220 and network_line['voltage'] == 220) or \
                    (jao_voltage in [380, 400] and network_line['voltage'] in [380, 400]):
                voltage_match = True

            # Skip if voltage doesn't match
            if not voltage_match:
                continue

            # Check if network line intersects with JAO buffer
            network_geom = network_line['geometry']
            if not jao_buffer.intersects(network_geom):
                continue

            # Calculate the coverage - how much of network line is within JAO buffer
            intersection = network_geom.intersection(jao_buffer)
            coverage = intersection.length / network_geom.length if network_geom.length > 0 else 0

            # Skip if coverage is too low
            if coverage < 0.5:  # At least 50% must be within buffer
                continue

            # Check if this network line is too similar to already matched lines
            too_similar = False
            for existing_geom in existing_network_geoms:
                # Calculate similarity with existing network lines
                similarity = calculate_geometry_coverage(existing_geom, network_geom, buffer_meters=300)
                if similarity > 0.7:  # If more than 70% similar to an existing line
                    too_similar = True
                    break

            if too_similar:
                continue

            # Calculate direction alignment score
            try:
                alignment = dir_cos(jao_geom, network_geom)

                # Skip if not aligned
                if alignment < 0.7:  # Lines should be mostly parallel
                    continue

                # Calculate Hausdorff distance for more precise measurement
                hausdorff_dist = jao_geom.hausdorff_distance(network_geom)
                hausdorff_meters = hausdorff_dist * meters_per_degree

                # Calculate overall score
                score = coverage * (1 - min(1, hausdorff_meters / 2000)) * alignment

                if score > 0.4:  # Score threshold
                    candidate_matches.append({
                        'network_line': network_line,
                        'coverage': coverage,
                        'alignment': alignment,
                        'hausdorff_meters': hausdorff_meters,
                        'score': score
                    })
            except Exception as e:
                print(f"  Error calculating alignment for network line {network_line['id']}: {e}")
                continue

        # Sort candidates by score
        candidate_matches.sort(key=lambda x: x['score'], reverse=True)

        # Take up to 3 best candidates
        selected_matches = candidate_matches[:3]

        if selected_matches:
            print(f"  Found {len(selected_matches)} additional network lines for JAO {jao_id}")

            # Add these network lines to the match
            if 'network_ids' not in result:
                result['network_ids'] = []

            for match in selected_matches:
                network_line = match['network_line']
                print(
                    f"    Adding network line {network_line['id']} (score: {match['score']:.2f}, coverage: {match['coverage']:.2f})")

                # Add to the match
                result['network_ids'].append(network_line['id'])
                newly_matched_network_ids.add(network_line['id'])

                # Update path length
                if 'path_length' not in result:
                    result['path_length'] = 0
                result['path_length'] = float(result['path_length'] + network_line['length'])

                # Mark this result as a parallel circuit if not already
                if not result.get('is_parallel_circuit', False) and not result.get('is_geometric_match', False):
                    result['is_parallel_circuit'] = True
                    result['match_quality'] = f'Parallel Circuit ({jao_voltage} kV) - Enhanced'

                additions_made += 1

            # Update length ratio
            if 'jao_length' in result and result['jao_length'] > 0:
                result['length_ratio'] = float(result['path_length'] / result['jao_length'])

    print(f"Added {additions_made} network lines to {len(newly_matched_network_ids)} unique parallel circuit matches")

    return matching_results


def share_network_lines_among_parallel_jaos(matching_results, jao_gdf):
    """
    Look for parallel JAO lines and ensure they share appropriate network lines.
    This addresses cases where multiple parallel JAO lines should use the same network lines.
    """
    print("\n=== SHARING NETWORK LINES AMONG PARALLEL JAO LINES ===")

    # Group JAO lines by voltage
    jao_by_voltage = {}
    for result in matching_results:
        if result['matched'] and not result.get('is_duplicate', False):
            voltage = result.get('v_nom', 0)
            if voltage not in jao_by_voltage:
                jao_by_voltage[voltage] = []
            jao_by_voltage[voltage].append(result)

    # Create dictionary of JAO geometries
    jao_geometries = {}
    for idx, row in jao_gdf.iterrows():
        jao_id = str(row['id'])
        jao_geometries[jao_id] = row.geometry

    # Track how many lines were shared
    shares_made = 0

    # For each voltage level, compare JAO lines
    for voltage, results in jao_by_voltage.items():
        print(f"  Processing {len(results)} JAO lines at {voltage} kV")

        # Compare each pair of JAO lines
        for i in range(len(results)):
            for j in range(i + 1, len(results)):
                result1 = results[i]
                result2 = results[j]

                # Skip if either doesn't have network lines
                if 'network_ids' not in result1 or 'network_ids' not in result2:
                    continue

                # Get JAO IDs
                jao_id1 = result1['jao_id']
                jao_id2 = result2['jao_id']

                # Skip if either doesn't have geometry
                if jao_id1 not in jao_geometries or jao_id2 not in jao_geometries:
                    continue

                # Calculate similarity between JAOs
                geom1 = jao_geometries[jao_id1]
                geom2 = jao_geometries[jao_id2]

                # Check how parallel they are
                try:
                    # Calculate coverage in both directions
                    coverage1 = calculate_geometry_coverage(geom1, geom2, buffer_meters=800)
                    coverage2 = calculate_geometry_coverage(geom2, geom1, buffer_meters=800)

                    # Calculate average coverage
                    avg_coverage = (coverage1 + coverage2) / 2

                    # Calculate Hausdorff distance
                    hausdorff_dist = geom1.hausdorff_distance(geom2)
                    avg_lat = (geom1.centroid.y + geom2.centroid.y) / 2
                    meters_per_degree = 111111 * np.cos(np.radians(abs(avg_lat)))
                    hausdorff_meters = hausdorff_dist * meters_per_degree

                    # If JAOs are parallel (high coverage, low Hausdorff distance)
                    if avg_coverage > 0.6 and hausdorff_meters < 1000:
                        print(f"    Found parallel JAOs: {jao_id1} and {jao_id2} (coverage: {avg_coverage:.2f})")

                        # Get network IDs
                        network_ids1 = set(result1['network_ids'])
                        network_ids2 = set(result2['network_ids'])

                        # Find network IDs to share
                        to_add_to_1 = network_ids2 - network_ids1
                        to_add_to_2 = network_ids1 - network_ids2

                        # Add to result1
                        if to_add_to_1:
                            print(f"      Sharing {len(to_add_to_1)} network lines with JAO {jao_id1}")
                            result1['network_ids'] = list(network_ids1.union(to_add_to_1))
                            shares_made += len(to_add_to_1)

                            # Mark as parallel circuit if not already
                            if not result1.get('is_parallel_circuit', False) and not result1.get('is_duplicate', False):
                                result1['is_parallel_circuit'] = True
                                result1['match_quality'] = f'Parallel Circuit ({voltage} kV) - Shared'

                        # Add to result2
                        if to_add_to_2:
                            print(f"      Sharing {len(to_add_to_2)} network lines with JAO {jao_id2}")
                            result2['network_ids'] = list(network_ids2.union(to_add_to_2))
                            shares_made += len(to_add_to_2)

                            # Mark as parallel circuit if not already
                            if not result2.get('is_parallel_circuit', False) and not result2.get('is_duplicate', False):
                                result2['is_parallel_circuit'] = True
                                result2['match_quality'] = f'Parallel Circuit ({voltage} kV) - Shared'

                except Exception as e:
                    print(f"  Error comparing JAOs {jao_id1} and {jao_id2}: {e}")

    print(f"Shared {shares_made} network lines between parallel JAO lines")

    return matching_results


def match_identical_network_geometries(matching_results, jao_gdf, network_gdf):
    """
    Match unmatched network lines that follow the same geometry as already matched network lines.
    More aggressive version to find all identical/parallel lines.
    """
    print("\n=== MATCHING IDENTICAL GEOMETRY NETWORK LINES (AGGRESSIVE) ===")

    # 1. Identify all network lines that are already used in matches
    used_network_ids = set()
    # Track which JAO each network line is matched to
    network_to_jao = {}

    for result in matching_results:
        if result['matched'] and 'network_ids' in result:
            jao_id = result['jao_id']
            for network_id in result['network_ids']:
                used_network_ids.add(str(network_id))
                network_to_jao[str(network_id)] = jao_id

    # 2. Get all unmatched network lines
    unmatched_network_lines = []
    for idx, row in network_gdf.iterrows():
        network_id = str(row['id'])
        if network_id not in used_network_ids:
            unmatched_network_lines.append({
                'id': network_id,
                'idx': idx,
                'geometry': row.geometry,
                'voltage': int(row['v_nom']),
                'length': calculate_length_meters(row.geometry),
                'bounds': row.geometry.bounds  # Store bounds for faster filtering
            })

    print(f"Found {len(unmatched_network_lines)} unmatched network lines to check")

    # 3. Create lists of matched network lines for each voltage
    matched_lines_220kv = []
    matched_lines_400kv = []

    for idx, row in network_gdf.iterrows():
        network_id = str(row['id'])
        if network_id in used_network_ids:
            line_info = {
                'id': network_id,
                'geometry': row.geometry,
                'voltage': int(row['v_nom']),
                'jao_id': network_to_jao.get(network_id),
                'bounds': row.geometry.bounds
            }

            if line_info['voltage'] == 220:
                matched_lines_220kv.append(line_info)
            elif line_info['voltage'] in [380, 400]:
                matched_lines_400kv.append(line_info)

    print(
        f"Processing {len(matched_lines_220kv)} matched 220kV lines and {len(matched_lines_400kv)} matched 380/400kV lines")

    # 4. For each unmatched network line, find if it's geometrically similar to a matched one
    matches_made = 0
    newly_matched_ids = set()

    # Process each voltage level separately
    for voltage in [220, 400]:
        # Filter unmatched lines by voltage
        if voltage == 220:
            unmatched_subset = [line for line in unmatched_network_lines if line['voltage'] == 220]
            matched_subset = matched_lines_220kv
        else:  # 380/400kV
            unmatched_subset = [line for line in unmatched_network_lines if line['voltage'] in [380, 400]]
            matched_subset = matched_lines_400kv

        print(f"Processing {len(unmatched_subset)} unmatched {voltage}kV lines")

        for unmatched in unmatched_subset:
            # Skip if already matched in this run
            if unmatched['id'] in newly_matched_ids:
                continue

            unmatched_id = unmatched['id']
            unmatched_geom = unmatched['geometry']
            unmatched_bounds = unmatched['bounds']

            # Use bounds to quickly filter potential matches
            minx, miny, maxx, maxy = unmatched_bounds
            # Expand bounds slightly to catch nearby lines
            buffer = 0.01  # ~1km in degrees
            search_bounds = (minx - buffer, miny - buffer, maxx + buffer, maxy + buffer)

            # Find best match
            best_match = None
            best_similarity = 0.75  # Lower threshold to catch more similar lines

            for matched in matched_subset:
                # Quick bounds check
                m_minx, m_miny, m_maxx, m_maxy = matched['bounds']

                # Skip if bounds don't overlap at all (quick filter)
                if (m_maxx < search_bounds[0] or m_minx > search_bounds[2] or
                        m_maxy < search_bounds[1] or m_miny > search_bounds[3]):
                    continue

                matched_geom = matched['geometry']

                # Calculate similarity
                try:
                    # Use a larger buffer - 300m instead of 200m
                    buffer_meters = 300

                    # Check overlap in both directions
                    overlap1 = calculate_geometry_coverage(matched_geom, unmatched_geom, buffer_meters)
                    overlap2 = calculate_geometry_coverage(unmatched_geom, matched_geom, buffer_meters)

                    # Average the overlap scores
                    avg_overlap = (overlap1 + overlap2) / 2

                    # If the overlap is decent, calculate more detailed metrics
                    if avg_overlap >= 0.7:
                        # Calculate Hausdorff distance
                        hausdorff_dist = matched_geom.hausdorff_distance(unmatched_geom)
                        avg_lat = (matched_geom.centroid.y + unmatched_geom.centroid.y) / 2
                        meters_per_degree = 111111 * np.cos(np.radians(abs(avg_lat)))
                        hausdorff_meters = hausdorff_dist * meters_per_degree

                        # Calculate length similarity
                        matched_length = calculate_length_meters(matched_geom)
                        length_ratio = unmatched['length'] / matched_length
                        length_similarity = 1 - min(abs(length_ratio - 1), 0.5) / 0.5

                        # Check if they're roughly parallel
                        try:
                            from shapely.geometry import LineString, MultiLineString
                            import numpy as np

                            # Get vectors for both lines
                            def get_vector(geom):
                                if geom.geom_type == 'MultiLineString':
                                    geom = max(geom.geoms, key=lambda g: g.length)

                                coords = np.array(list(geom.coords))
                                vec = coords[-1] - coords[0]
                                norm = np.linalg.norm(vec)
                                return vec / norm if norm > 0 else vec

                            vec1 = get_vector(matched_geom)
                            vec2 = get_vector(unmatched_geom)

                            # Calculate dot product (cosine of angle)
                            alignment = abs(np.dot(vec1, vec2))
                        except:
                            # If calculation fails, assume moderate alignment
                            alignment = 0.7

                        # Combine all metrics into a similarity score
                        # Heaviest weight on overlap and Hausdorff distance
                        similarity = (
                                0.5 * avg_overlap +
                                0.3 * (1 - min(1, hausdorff_meters / 500)) +
                                0.1 * length_similarity +
                                0.1 * alignment
                        )

                        if similarity > best_similarity:
                            best_similarity = similarity
                            best_match = {
                                'network_id': matched['id'],
                                'jao_id': matched['jao_id'],
                                'similarity': similarity,
                                'overlap': avg_overlap,
                                'hausdorff_meters': hausdorff_meters,
                                'alignment': alignment,
                                'length_ratio': length_ratio
                            }
                except Exception as e:
                    print(f"  Error comparing network lines {unmatched_id} and {matched['id']}: {e}")
                    continue

            # If we found a good match, add this network line to the same JAO
            if best_match:
                jao_id = best_match['jao_id']

                # Find the matching result for this JAO
                for result in matching_results:
                    if result['jao_id'] == jao_id and result['matched']:
                        print(f"  Network line {unmatched_id} matches with network line {best_match['network_id']} "
                              f"(similarity: {best_similarity:.3f}, overlap: {best_match['overlap']:.2f}, "
                              f"hausdorff: {best_match['hausdorff_meters']:.1f}m)")
                        print(f"  Adding to JAO {jao_id}")

                        # Add to network IDs
                        if 'network_ids' not in result:
                            result['network_ids'] = []
                        result['network_ids'].append(unmatched_id)
                        newly_matched_ids.add(unmatched_id)

                        # Update path length
                        if 'path_length' not in result:
                            result['path_length'] = 0
                        result['path_length'] = float(result['path_length'] + unmatched['length'])

                        # Update length ratio
                        if 'jao_length' in result and result['jao_length'] > 0:
                            result['length_ratio'] = float(result['path_length'] / result['jao_length'])

                        # If this wasn't already a parallel circuit, mark it as one
                        if not result.get('is_parallel_circuit', False) and not result.get('is_duplicate', False):
                            result['is_parallel_circuit'] = True
                            if 'v_nom' in result:
                                result[
                                    'match_quality'] = f'Parallel Circuit ({result["v_nom"]} kV) - Identical Geometry'
                            else:
                                result['match_quality'] = 'Parallel Circuit - Identical Geometry'

                        matches_made += 1
                        break

    print(f"Matched {matches_made} network lines based on identical geometry")

    return matching_results


def cluster_identical_network_lines(matching_results, network_gdf):
    """
    Find clusters of network lines with identical geometries and make sure they're all
    matched to the same JAO. More aggressive approach using multiple clustering methods.
    """
    print("\n=== CLUSTERING IDENTICAL NETWORK LINES (AGGRESSIVE) ===")

    # Track which JAO each network line is matched to
    network_to_jao = {}
    network_to_result = {}

    for result in matching_results:
        if result['matched'] and 'network_ids' in result:
            jao_id = result['jao_id']
            for network_id in result['network_ids']:
                network_to_jao[str(network_id)] = jao_id
                network_to_result[str(network_id)] = result

    # 1. First clustering method: By bounding box and length
    bbox_clusters = {}

    for idx, row in network_gdf.iterrows():
        network_id = str(row['id'])
        voltage = int(row['v_nom'])
        length = calculate_length_meters(row.geometry)

        # Create a key based on bounding box (rounded) and length
        bounds = row.geometry.bounds
        # Round to reduce floating point issues
        bounds_key = tuple([round(coord, 5) for coord in bounds])
        length_key = round(length / 100) * 100  # Round to nearest 100m

        cluster_key = (bounds_key, length_key, voltage)

        if cluster_key not in bbox_clusters:
            bbox_clusters[cluster_key] = []

        bbox_clusters[cluster_key].append({
            'id': network_id,
            'geometry': row.geometry,
            'voltage': voltage,
            'length': length,
            'is_matched': network_id in network_to_jao
        })

    # Filter to keep only clusters with multiple lines
    multi_line_bbox_clusters = {k: v for k, v in bbox_clusters.items() if len(v) > 1}

    print(f"Found {len(multi_line_bbox_clusters)} clusters based on bounding box and length")

    # 2. Second clustering method: By geometry similarity
    similarity_clusters = {}

    # Create a simplified representation for each line
    line_to_simple = {}
    simple_to_lines = {}

    # Process each network line
    for idx, row in network_gdf.iterrows():
        network_id = str(row['id'])

        # Skip lines we've already processed in bbox clustering
        if any(network_id in [line['id'] for line in cluster]
               for cluster in multi_line_bbox_clusters.values()):
            continue

        # Create a simplified key for this line
        # We'll use a buffer-based approach to find similar geometries
        try:
            geom = row.geometry
            buffer_meters = 200

            # Calculate buffer in degrees
            avg_lat = geom.centroid.y
            meters_per_degree = 111111 * np.cos(np.radians(abs(avg_lat)))
            buffer_deg = buffer_meters / meters_per_degree

            # Create a buffer and use its bounds as a key
            buffer = geom.buffer(buffer_deg)
            simple_key = tuple([round(coord, 5) for coord in buffer.bounds])

            line_to_simple[network_id] = simple_key

            if simple_key not in simple_to_lines:
                simple_to_lines[simple_key] = []

            simple_to_lines[simple_key].append({
                'id': network_id,
                'geometry': geom,
                'voltage': int(row['v_nom']),
                'length': calculate_length_meters(geom),
                'is_matched': network_id in network_to_jao
            })
        except Exception as e:
            print(f"  Error processing network line {network_id}: {e}")

    # Filter to keep only clusters with multiple lines and same voltage
    for simple_key, lines in simple_to_lines.items():
        # Skip small clusters
        if len(lines) <= 1:
            continue

        # Group by voltage
        by_voltage = {}
        for line in lines:
            voltage = line['voltage']
            if voltage not in by_voltage:
                by_voltage[voltage] = []
            by_voltage[voltage].append(line)

        # Add voltage-specific clusters
        for voltage, voltage_lines in by_voltage.items():
            if len(voltage_lines) > 1:
                # Further verify these are truly similar with Hausdorff distance
                verified_groups = []
                remaining = voltage_lines.copy()

                while remaining:
                    base = remaining.pop(0)
                    current_group = [base]

                    # Compare with all remaining lines
                    i = 0
                    while i < len(remaining):
                        line = remaining[i]
                        try:
                            # Calculate Hausdorff distance
                            hausdorff_dist = base['geometry'].hausdorff_distance(line['geometry'])
                            avg_lat = (base['geometry'].centroid.y + line['geometry'].centroid.y) / 2
                            meters_per_degree = 111111 * np.cos(np.radians(abs(avg_lat)))
                            hausdorff_meters = hausdorff_dist * meters_per_degree

                            # If they're very close, add to this group
                            if hausdorff_meters < 300:  # 300m threshold
                                current_group.append(line)
                                remaining.pop(i)
                            else:
                                i += 1
                        except Exception as e:
                            print(f"  Error comparing lines: {e}")
                            i += 1

                    # If we found a group with multiple lines, add it
                    if len(current_group) > 1:
                        verified_groups.append(current_group)

                # Add verified groups to similarity clusters
                for group_idx, group in enumerate(verified_groups):
                    similarity_key = (simple_key, voltage, group_idx)
                    similarity_clusters[similarity_key] = group

    print(f"Found {len(similarity_clusters)} additional clusters based on geometry similarity")

    # 3. Process all clusters (from both methods)
    all_clusters = list(multi_line_bbox_clusters.values()) + list(similarity_clusters.values())
    lines_matched = 0

    for cluster_idx, lines in enumerate(all_clusters):
        # Group by voltage
        by_voltage = {}
        for line in lines:
            voltage = line['voltage']
            if voltage not in by_voltage:
                by_voltage[voltage] = []
            by_voltage[voltage].append(line)

        # Process each voltage group separately
        for voltage, voltage_lines in by_voltage.items():
            if len(voltage_lines) <= 1:
                continue

            print(f"  Processing cluster {cluster_idx + 1}: {len(voltage_lines)} identical {voltage} kV network lines")

            # Find if any of these lines are already matched
            matched_lines = [line for line in voltage_lines if line['is_matched']]
            unmatched_lines = [line for line in voltage_lines if not line['is_matched']]

            if matched_lines and unmatched_lines:
                # If some lines in the cluster are matched and others aren't,
                # match the unmatched ones to the same JAO as the matched ones

                # Group matched lines by JAO
                jao_groups = {}
                for line in matched_lines:
                    jao_id = network_to_jao.get(line['id'])
                    if jao_id not in jao_groups:
                        jao_groups[jao_id] = []
                    jao_groups[jao_id].append(line)

                # Find the JAO with the most matched lines
                best_jao_id = max(jao_groups.keys(), key=lambda k: len(jao_groups[k]))
                best_result = network_to_result.get(jao_groups[best_jao_id][0]['id'])

                if best_result:
                    print(f"    Adding {len(unmatched_lines)} unmatched lines to JAO {best_jao_id}")

                    for line in unmatched_lines:
                        # Find the network line in the GeoDataFrame to get its length
                        network_rows = network_gdf[network_gdf['id'].astype(str) == line['id']]
                        if network_rows.empty:
                            continue

                        network_length = calculate_length_meters(network_rows.iloc[0].geometry)

                        # Add to network IDs
                        if 'network_ids' not in best_result:
                            best_result['network_ids'] = []
                        best_result['network_ids'].append(line['id'])

                        # Update path length
                        if 'path_length' not in best_result:
                            best_result['path_length'] = 0
                        best_result['path_length'] = float(best_result['path_length'] + network_length)

                        lines_matched += 1

                    # Update length ratio
                    if 'jao_length' in best_result and best_result['jao_length'] > 0:
                        best_result['length_ratio'] = float(best_result['path_length'] / best_result['jao_length'])

                    # If this wasn't already a parallel circuit, mark it as one
                    if not best_result.get('is_parallel_circuit', False) and not best_result.get('is_duplicate', False):
                        best_result['is_parallel_circuit'] = True
                        if 'v_nom' in best_result:
                            best_result['match_quality'] = f'Parallel Circuit ({best_result["v_nom"]} kV) - Clustered'
                        else:
                            best_result['match_quality'] = 'Parallel Circuit - Clustered'

    print(f"Matched {lines_matched} additional network lines by clustering")

    return matching_results


def match_remaining_identical_network_lines(matching_results, network_gdf):
    """
    Final pass to find any remaining unmatched network lines that exactly follow
    the same path as matched lines. This is an extra-aggressive approach for finding
    parallel circuit lines that were missed by other methods.
    """
    print("\n=== FINAL PASS: MATCHING REMAINING IDENTICAL NETWORK LINES ===")

    # 1. Identify all network lines that are already used in matches
    used_network_ids = set()
    # Track which JAO each network line is matched to and the voltage
    network_to_jao = {}
    network_to_voltage = {}

    for result in matching_results:
        if result['matched'] and 'network_ids' in result:
            jao_id = result['jao_id']
            voltage = result.get('v_nom', 0)
            for network_id in result['network_ids']:
                used_network_ids.add(str(network_id))
                network_to_jao[str(network_id)] = jao_id
                network_to_voltage[str(network_id)] = voltage

    # 2. Get all unmatched network lines
    unmatched_network_lines = []
    for idx, row in network_gdf.iterrows():
        network_id = str(row['id'])
        if network_id not in used_network_ids:
            unmatched_network_lines.append({
                'id': network_id,
                'geometry': row.geometry,
                'voltage': int(row['v_nom']),
                'length': calculate_length_meters(row.geometry)
            })

    print(f"Found {len(unmatched_network_lines)} unmatched network lines to check")

    # 3. Create a dictionary of all matched network lines
    matched_network_lines = {}
    for idx, row in network_gdf.iterrows():
        network_id = str(row['id'])
        if network_id in used_network_ids:
            matched_network_lines[network_id] = {
                'geometry': row.geometry,
                'voltage': int(row['v_nom']),
                'jao_id': network_to_jao.get(network_id),
                'jao_voltage': network_to_voltage.get(network_id)
            }

    # 4. For each unmatched line, check if it's virtually identical to any matched line
    matches_made = 0
    newly_matched_ids = set()

    for unmatched in unmatched_network_lines:
        # Skip if already matched in this run
        if unmatched['id'] in newly_matched_ids:
            continue

        unmatched_id = unmatched['id']
        unmatched_geom = unmatched['geometry']
        unmatched_voltage = unmatched['voltage']

        best_match = None
        best_similarity = 0.95  # Very high threshold - we want nearly identical lines

        for matched_id, matched in matched_network_lines.items():
            matched_geom = matched['geometry']
            matched_voltage = matched['voltage']

            # Only match same voltage lines
            if (unmatched_voltage == 220 and matched_voltage == 220) or \
                    (unmatched_voltage in [380, 400] and matched_voltage in [380, 400]):
                # Check if geometries are nearly identical
                try:
                    # Calculate Hausdorff distance - this is the key metric for "sameness"
                    hausdorff_dist = matched_geom.hausdorff_distance(unmatched_geom)
                    avg_lat = (matched_geom.centroid.y + unmatched_geom.centroid.y) / 2
                    meters_per_degree = 111111 * np.cos(np.radians(abs(avg_lat)))
                    hausdorff_meters = hausdorff_dist * meters_per_degree

                    # For almost identical lines, Hausdorff should be very small
                    if hausdorff_meters < 150:  # 150m threshold - quite strict
                        # Calculate directional coverage as well
                        overlap1 = calculate_geometry_coverage(matched_geom, unmatched_geom, buffer_meters=150)
                        overlap2 = calculate_geometry_coverage(unmatched_geom, matched_geom, buffer_meters=150)
                        avg_overlap = (overlap1 + overlap2) / 2

                        # Calculate similarity score - primarily based on Hausdorff
                        similarity = (1 - min(1, hausdorff_meters / 150)) * 0.7 + avg_overlap * 0.3

                        if similarity > best_similarity:
                            best_similarity = similarity
                            best_match = {
                                'network_id': matched_id,
                                'jao_id': matched['jao_id'],
                                'similarity': similarity,
                                'hausdorff_meters': hausdorff_meters,
                                'overlap': avg_overlap
                            }
                except Exception as e:
                    print(f"  Error comparing network lines {unmatched_id} and {matched_id}: {e}")
                    continue

        # If we found a very good match, add this network line to the same JAO
        if best_match:
            jao_id = best_match['jao_id']

            # Find the matching result for this JAO
            for result in matching_results:
                if result['jao_id'] == jao_id and result['matched']:
                    print(f"  Network line {unmatched_id} is virtually identical to {best_match['network_id']} "
                          f"(similarity: {best_similarity:.3f}, hausdorff: {best_match['hausdorff_meters']:.1f}m)")
                    print(f"  Adding to JAO {jao_id}")

                    # Add to network IDs
                    if 'network_ids' not in result:
                        result['network_ids'] = []
                    result['network_ids'].append(unmatched_id)
                    newly_matched_ids.add(unmatched_id)

                    # Update path length
                    if 'path_length' not in result:
                        result['path_length'] = 0
                    result['path_length'] = float(result['path_length'] + unmatched['length'])

                    # Update length ratio
                    if 'jao_length' in result and result['jao_length'] > 0:
                        result['length_ratio'] = float(result['path_length'] / result['jao_length'])

                    # If this wasn't already a parallel circuit, mark it as one
                    if not result.get('is_parallel_circuit', False) and not result.get('is_duplicate', False):
                        result['is_parallel_circuit'] = True
                        if 'v_nom' in result:
                            result['match_quality'] = f'Parallel Circuit ({result["v_nom"]} kV) - Final Pass'
                        else:
                            result['match_quality'] = 'Parallel Circuit - Final Pass'

                    matches_made += 1
                    break

    print(f"Matched {matches_made} network lines in final pass")

    return matching_results


def corridor_parallel_match(matching_results, jao_gdf, network_gdf, corridor_w_220=250, corridor_w_400=350):
    """
    Match parallel network lines using a corridor-based approach.

    This function:
    1. Creates corridor buffers around each network line
    2. Dissolves them by voltage to identify corridors
    3. Assigns corridor IDs to all network lines
    4. Propagates matches within each corridor

    Parameters:
    - matching_results: Current matching results
    - jao_gdf: GeoDataFrame with JAO lines
    - network_gdf: GeoDataFrame with network lines
    - corridor_w_220: Corridor width in meters for 220kV lines
    - corridor_w_400: Corridor width in meters for 380/400kV lines

    Returns:
    - Updated matching results
    """
    import geopandas as gpd
    import numpy as np
    from shapely.ops import unary_union
    import pandas as pd

    print("\n=== MATCHING PARALLEL NETWORK LINES USING CORRIDOR APPROACH ===")

    # 1. Identify network lines that are already used in matches
    used_network_ids = set()
    network_to_jao = {}  # Map from network ID to JAO ID
    network_to_result = {}  # Map from network ID to its result object

    for result in matching_results:
        if result['matched'] and 'network_ids' in result:
            jao_id = result['jao_id']
            for network_id in result['network_ids']:
                used_network_ids.add(str(network_id))
                network_to_jao[str(network_id)] = jao_id
                network_to_result[str(network_id)] = result

    print(f"Found {len(used_network_ids)} already matched network lines")

    # Create a copy of network_gdf to avoid modifying the original
    network_df = network_gdf.copy()

    # 2. Create buffers for each network line based on voltage
    print("Creating corridor buffers...")

    def create_buffer(row):
        """Create a buffer around the network line with width based on voltage"""
        try:
            # Calculate meters per degree at this latitude
            avg_lat = row.geometry.centroid.y
            meters_per_degree = 111111 * np.cos(np.radians(abs(avg_lat)))

            # Determine buffer width based on voltage
            if row['v_nom'] == 220:
                width_meters = corridor_w_220
            else:  # 380/400 kV
                width_meters = corridor_w_400

            # Convert to degrees
            buffer_width = width_meters / meters_per_degree

            # Create buffer
            return row.geometry.buffer(buffer_width)
        except Exception as e:
            print(f"Error creating buffer for network line {row['id']}: {e}")
            # Return a minimal buffer as fallback
            return row.geometry.buffer(0.001)

    # Apply buffer creation to each network line
    network_df['buffer'] = network_df.apply(create_buffer, axis=1)

    # 3. Dissolve buffers by voltage to get corridors
    print("Dissolving buffers to identify corridors...")

    # Group by voltage
    voltage_groups = network_df.groupby('v_nom')

    # Create corridor polygons for each voltage
    corridor_polygons = []

    for voltage, group in voltage_groups:
        try:
            # Dissolve all buffers in this voltage group
            dissolved = unary_union(group['buffer'].tolist())

            # Convert to individual polygons
            from shapely.ops import polygonize
            if hasattr(dissolved, 'geoms'):
                polys = list(dissolved.geoms)
            else:
                polys = [dissolved]

            # Add each polygon as a corridor with its voltage
            for i, poly in enumerate(polys):
                corridor_polygons.append({
                    'corridor_id': f"{voltage}_{i}",
                    'voltage': voltage,
                    'geometry': poly
                })
        except Exception as e:
            print(f"Error processing voltage group {voltage}: {e}")

    # Create GeoDataFrame of corridors
    if corridor_polygons:
        corridor_gdf = gpd.GeoDataFrame(corridor_polygons, geometry='geometry')
        print(f"Created {len(corridor_gdf)} corridor polygons")
    else:
        print("Warning: No corridor polygons created")
        return matching_results  # Return unchanged if no corridors created

    # 4. Spatial join to assign corridor IDs to network lines
    print("Assigning corridor IDs to network lines...")

    # Add network ID as string for joining
    network_df['network_id'] = network_df['id'].astype(str)

    # Perform spatial join
    joined = gpd.sjoin(network_df[['id', 'network_id', 'v_nom', 'geometry']],
                       corridor_gdf[['corridor_id', 'geometry']],
                       how='left', predicate='within')

    # 5. Group network lines by corridor
    corridor_groups = {}

    # Create groups of network lines by corridor
    for corridor_id, group in joined.groupby('corridor_id'):
        if pd.isna(corridor_id):
            continue  # Skip lines not in any corridor

        # Get all network IDs in this corridor
        network_ids = group['network_id'].tolist()

        # Only process corridors with at least one matched line
        if any(nid in used_network_ids for nid in network_ids):
            corridor_groups[corridor_id] = {
                'all_ids': network_ids,
                'matched_ids': [nid for nid in network_ids if nid in used_network_ids],
                'unmatched_ids': [nid for nid in network_ids if nid not in used_network_ids],
                'voltage': group['v_nom'].iloc[0]
            }

    # 6. Propagate matches within each corridor
    print("Propagating matches within corridors...")

    matches_added = 0
    corridors_processed = 0

    for corridor_id, data in corridor_groups.items():
        # Skip if no unmatched lines or no matched lines
        if not data['matched_ids'] or not data['unmatched_ids']:
            continue

        print(f"  Processing corridor {corridor_id} with {len(data['matched_ids'])} matched and "
              f"{len(data['unmatched_ids'])} unmatched lines")

        # Find the JAO that the first matched line belongs to
        anchor_network_id = data['matched_ids'][0]
        jao_id = network_to_jao.get(anchor_network_id)

        if not jao_id:
            continue  # Skip if no JAO found

        result = network_to_result.get(anchor_network_id)

        if not result:
            continue  # Skip if no result found

        # Add unmatched lines to this JAO
        for network_id in data['unmatched_ids']:
            # Find the network line in the GeoDataFrame
            network_rows = network_gdf[network_gdf['id'].astype(str) == network_id]
            if network_rows.empty:
                continue

            network_row = network_rows.iloc[0]

            # Double-check voltage match
            network_voltage = int(network_row['v_nom'])
            jao_voltage = result.get('v_nom', 0)

            voltage_match = False
            if (network_voltage == 220 and jao_voltage == 220) or \
                    (network_voltage in [380, 400] and jao_voltage in [380, 400]):
                voltage_match = True

            if not voltage_match:
                print(
                    f"    Skipping network line {network_id} due to voltage mismatch: {network_voltage} vs {jao_voltage}")
                continue

            print(f"    Adding network line {network_id} to JAO {jao_id}")

            # Add to network IDs
            if 'network_ids' not in result:
                result['network_ids'] = []
            result['network_ids'].append(network_id)

            # Update path length
            network_length = calculate_length_meters(network_row.geometry)
            if 'path_length' not in result:
                result['path_length'] = 0
            result['path_length'] = float(result['path_length'] + network_length)

            # Mark as matched
            used_network_ids.add(network_id)
            matches_added += 1

        # Update length ratio
        if 'jao_length' in result and result['jao_length'] > 0:
            result['length_ratio'] = float(result['path_length'] / result['jao_length'])

        # Mark as parallel circuit if not already
        if not result.get('is_parallel_circuit', False) and not result.get('is_duplicate', False):
            result['is_parallel_circuit'] = True
            if 'v_nom' in result:
                result['match_quality'] = f'Parallel Circuit ({result["v_nom"]} kV) - Corridor Matched'
            else:
                result['match_quality'] = 'Parallel Circuit - Corridor Matched'

        # Add corridor info
        result['corridor_id'] = corridor_id
        result['parallel_count'] = len(result.get('network_ids', []))

        corridors_processed += 1

    print(f"Added {matches_added} network lines from {corridors_processed} corridors")

    return matching_results


def match_identical_network_geometries_aggressive(matching_results, jao_gdf, network_gdf):
    """
    Match unmatched network lines that follow the same geometry as already matched network lines.
    This is a more aggressive version using progressive relaxation of thresholds.
    """
    print("\n=== MATCHING IDENTICAL GEOMETRY NETWORK LINES (PROGRESSIVE RELAXATION) ===")

    # 1. Identify all network lines that are already used in matches
    used_network_ids = set()
    # Track which JAO each network line is matched to
    network_to_jao = {}

    for result in matching_results:
        if result['matched'] and 'network_ids' in result:
            jao_id = result['jao_id']
            for network_id in result['network_ids']:
                used_network_ids.add(str(network_id))
                network_to_jao[str(network_id)] = jao_id

    # 2. Get all unmatched network lines
    unmatched_network_lines = []
    for idx, row in network_gdf.iterrows():
        network_id = str(row['id'])
        if network_id not in used_network_ids:
            unmatched_network_lines.append({
                'id': network_id,
                'idx': idx,
                'geometry': row.geometry,
                'voltage': int(row['v_nom']),
                'length': calculate_length_meters(row.geometry)
            })

    print(f"Found {len(unmatched_network_lines)} unmatched network lines to check")

    # 3. Create a spatial index for matched network lines for faster lookup
    from rtree import index
    matched_idx = index.Index()
    matched_networks = {}

    i = 0
    for idx, row in network_gdf.iterrows():
        network_id = str(row['id'])
        if network_id in used_network_ids:
            # Get the bounds of the line
            minx, miny, maxx, maxy = row.geometry.bounds
            matched_idx.insert(i, (minx, miny, maxx, maxy))
            matched_networks[i] = {
                'id': network_id,
                'geometry': row.geometry,
                'voltage': int(row['v_nom']),
                'jao_id': network_to_jao.get(network_id)
            }
            i += 1

    # 4. Use progressive relaxation to match network lines
    print("Using progressive relaxation to match network lines...")

    # Define thresholds to try (from strict to loose)
    thresholds = [
        {'buffer_meters': 150, 'similarity': 0.95},
        {'buffer_meters': 250, 'similarity': 0.90},
        {'buffer_meters': 350, 'similarity': 0.85}
    ]

    total_matches_made = 0
    newly_matched_ids = set()

    for threshold in thresholds:
        buffer_meters = threshold['buffer_meters']
        similarity_threshold = threshold['similarity']

        print(f"  Trying buffer={buffer_meters}m, similarity={similarity_threshold}")

        matches_made = 0

        # Only process unmatched lines that haven't been matched in earlier iterations
        remaining_unmatched = [line for line in unmatched_network_lines
                               if line['id'] not in newly_matched_ids]

        for unmatched in remaining_unmatched:
            unmatched_id = unmatched['id']
            unmatched_geom = unmatched['geometry']
            unmatched_voltage = unmatched['voltage']

            # Get bounds for spatial query
            bounds = unmatched_geom.bounds

            # Find candidate matched lines that are in the same area
            best_match = None
            best_similarity = similarity_threshold  # Use current threshold

            for candidate_idx in matched_idx.intersection(bounds):
                matched_network = matched_networks[candidate_idx]
                matched_geom = matched_network['geometry']
                matched_voltage = matched_network['voltage']

                # If voltages don't match, skip (accounting for 380/400 equivalence)
                voltage_match = False
                if (unmatched_voltage == 220 and matched_voltage == 220) or \
                        (unmatched_voltage in [380, 400] and matched_voltage in [380, 400]):
                    voltage_match = True

                if not voltage_match:
                    continue

                # Calculate how similar the geometries are
                try:
                    # Check overlap in both directions
                    overlap1 = calculate_geometry_coverage(matched_geom, unmatched_geom, buffer_meters)
                    overlap2 = calculate_geometry_coverage(unmatched_geom, matched_geom, buffer_meters)

                    # Average the overlap scores
                    avg_overlap = (overlap1 + overlap2) / 2

                    # Calculate Hausdorff distance
                    hausdorff_dist = matched_geom.hausdorff_distance(unmatched_geom)
                    avg_lat = (matched_geom.centroid.y + unmatched_geom.centroid.y) / 2
                    meters_per_degree = 111111 * np.cos(np.radians(abs(avg_lat)))
                    hausdorff_meters = hausdorff_dist * meters_per_degree

                    # Calculate direction alignment
                    try:
                        from shapely.geometry import LineString, MultiLineString
                        import numpy as np

                        # Get vectors for both lines
                        def get_vector(geom):
                            if geom.geom_type == 'MultiLineString':
                                geom = max(geom.geoms, key=lambda g: g.length)

                            coords = np.array(list(geom.coords))
                            vec = coords[-1] - coords[0]
                            norm = np.linalg.norm(vec)
                            return vec / norm if norm > 0 else vec

                        vec1 = get_vector(matched_geom)
                        vec2 = get_vector(unmatched_geom)

                        # Calculate dot product (cosine of angle)
                        alignment = abs(np.dot(vec1, vec2))
                    except:
                        # If calculation fails, assume moderate alignment
                        alignment = 0.7

                    # Combined similarity score with more weight on overlap and Hausdorff
                    similarity = (
                            0.5 * avg_overlap +
                            0.3 * (1 - min(1, hausdorff_meters / buffer_meters)) +
                            0.2 * alignment
                    )

                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match = {
                            'network_id': matched_network['id'],
                            'jao_id': matched_network['jao_id'],
                            'similarity': similarity,
                            'overlap': avg_overlap,
                            'hausdorff_meters': hausdorff_meters,
                            'alignment': alignment
                        }

                except Exception as e:
                    print(f"  Error comparing network lines {unmatched_id} and {matched_network['id']}: {e}")
                    continue

            # If we found a good match, add this network line to the same JAO
            if best_match:
                jao_id = best_match['jao_id']

                # Find the matching result for this JAO
                for result in matching_results:
                    if result['jao_id'] == jao_id and result['matched']:
                        print(f"  Network line {unmatched_id} matches with network line {best_match['network_id']} "
                              f"(similarity: {best_similarity:.3f}, overlap: {best_match['overlap']:.2f}, "
                              f"buffer: {buffer_meters}m)")

                        # Add to network IDs
                        if 'network_ids' not in result:
                            result['network_ids'] = []
                        result['network_ids'].append(unmatched_id)
                        newly_matched_ids.add(unmatched_id)

                        # Update path length
                        if 'path_length' not in result:
                            result['path_length'] = 0
                        result['path_length'] = float(result['path_length'] + unmatched['length'])

                        # Update length ratio
                        if 'jao_length' in result and result['jao_length'] > 0:
                            result['length_ratio'] = float(result['path_length'] / result['jao_length'])

                        # If this wasn't already a parallel circuit, mark it as one
                        if not result.get('is_parallel_circuit', False) and not result.get('is_duplicate', False):
                            result['is_parallel_circuit'] = True
                            if 'v_nom' in result:
                                result[
                                    'match_quality'] = f'Parallel Circuit ({result["v_nom"]} kV) - Progressive Relaxation'
                            else:
                                result['match_quality'] = 'Parallel Circuit - Progressive Relaxation'

                        matches_made += 1
                        break

        print(f"  Found {matches_made} matches with buffer={buffer_meters}m, similarity={similarity_threshold}")
        total_matches_made += matches_made

    print(f"Matched {total_matches_made} network lines using progressive relaxation")

    return matching_results


def match_network_lines_by_geometry_hash(matching_results, network_gdf, sample_distance=250):
    """
    Match unmatched network lines using a geometry hash approach.

    This method:
    1. Resamples lines to have regular spacing
    2. Snaps vertices to a grid
    3. Creates a hash of the coordinates
    4. Groups lines with similar hashes

    Parameters:
    - matching_results: Current matching results
    - network_gdf: GeoDataFrame with network lines
    - sample_distance: Distance in meters between sample points

    Returns:
    - Updated matching results
    """
    import hashlib
    import numpy as np

    print("\n=== MATCHING NETWORK LINES USING GEOMETRY HASH ===")

    # Make sure we always return the matching_results even if errors occur
    if matching_results is None:
        print("Error: matching_results is None")
        return []  # Return empty list instead of None

    # 1. Identify all network lines that are already used in matches
    used_network_ids = set()
    network_to_jao = {}
    network_to_result = {}

    for result in matching_results:
        if result['matched'] and 'network_ids' in result:
            jao_id = result['jao_id']
            for network_id in result['network_ids']:
                used_network_ids.add(str(network_id))
                network_to_jao[str(network_id)] = jao_id
                network_to_result[str(network_id)] = result

    # 2. Create a hash for each network line's geometry
    print("Creating geometry hashes...")

    # Function to create a geometry hash
    def create_geometry_hash(geom, sample_dist_meters):
        try:
            # Calculate sample distance in degrees
            avg_lat = geom.centroid.y
            meters_per_degree = 111111 * np.cos(np.radians(abs(avg_lat)))
            sample_dist_deg = sample_dist_meters / meters_per_degree

            # Get total length of geometry
            length = geom.length

            # If very short, just use endpoints
            if length < sample_dist_deg * 2:
                coords = [list(geom.coords)[0], list(geom.coords)[-1]]
            else:
                # Sample points at regular intervals
                num_points = max(3, int(length / sample_dist_deg))
                points = []

                # Use shapely's interpolate method directly on the geometry
                for i in range(num_points):
                    # Calculate the distance along the line for this point
                    distance = i * length / (num_points - 1)
                    # Interpolate a point at this distance
                    point = geom.interpolate(distance)
                    points.append((point.x, point.y))

                coords = points

            # Snap to grid (round to 4 decimal places ≈ 10m)
            snapped_coords = [tuple(np.round(p, 4)) for p in coords]

            # Create hash of coordinates
            coord_str = str(snapped_coords)
            return hashlib.sha256(coord_str.encode()).hexdigest()
        except Exception as e:
            print(f"Error creating hash: {e}")
            # Fallback to geometry WKT if hashing fails
            return hashlib.sha256(geom.wkt.encode()).hexdigest()

    # Create hashes for all network lines
    network_hashes = {}

    try:
        for idx, row in network_gdf.iterrows():
            network_id = str(row['id'])
            geometry = row.geometry
            voltage = int(row['v_nom'])

            # Create hash
            hash_key = create_geometry_hash(geometry, sample_distance)

            # Store with voltage to avoid mixing different voltages
            voltage_hash = f"{voltage}_{hash_key}"

            if voltage_hash not in network_hashes:
                network_hashes[voltage_hash] = []

            network_hashes[voltage_hash].append({
                'id': network_id,
                'geometry': geometry,
                'voltage': voltage,
                'length': calculate_length_meters(geometry),
                'is_matched': network_id in used_network_ids
            })
    except Exception as e:
        print(f"Error creating network hashes: {e}")
        return matching_results  # Return original results if error occurs

    # 3. Find groups with both matched and unmatched lines
    print("Finding hash clusters with both matched and unmatched lines...")

    matches_made = 0

    try:
        for hash_key, lines in network_hashes.items():
            # Skip if only one line with this hash
            if len(lines) <= 1:
                continue

            # Find if any are already matched
            matched_lines = [line for line in lines if line['is_matched']]
            unmatched_lines = [line for line in lines if not line['is_matched']]

            # Skip if all are matched or all are unmatched
            if not matched_lines or not unmatched_lines:
                continue

            print(
                f"  Hash cluster {hash_key[:8]}... has {len(matched_lines)} matched and {len(unmatched_lines)} unmatched lines")

            # Group matched lines by JAO
            jao_groups = {}
            for line in matched_lines:
                jao_id = network_to_jao.get(line['id'])
                if jao_id not in jao_groups:
                    jao_groups[jao_id] = []
                jao_groups[jao_id].append(line)

            # Find the JAO with the most matched lines
            if not jao_groups:
                continue  # Skip if no JAO groups

            best_jao_id = max(jao_groups.keys(), key=lambda k: len(jao_groups[k]))
            first_line_id = jao_groups[best_jao_id][0]['id']

            best_result = network_to_result.get(first_line_id)

            if best_result:
                print(f"    Adding {len(unmatched_lines)} unmatched lines to JAO {best_jao_id}")

                for line in unmatched_lines:
                    # Add to network IDs
                    if 'network_ids' not in best_result:
                        best_result['network_ids'] = []
                    best_result['network_ids'].append(line['id'])

                    # Update path length
                    if 'path_length' not in best_result:
                        best_result['path_length'] = 0
                    best_result['path_length'] = float(best_result['path_length'] + line['length'])

                    matches_made += 1

                # Update length ratio
                if 'jao_length' in best_result and best_result['jao_length'] > 0:
                    best_result['length_ratio'] = float(best_result['path_length'] / best_result['jao_length'])

                # If this wasn't already a parallel circuit, mark it as one
                if not best_result.get('is_parallel_circuit', False) and not best_result.get('is_duplicate', False):
                    best_result['is_parallel_circuit'] = True
                    if 'v_nom' in best_result:
                        best_result['match_quality'] = f'Parallel Circuit ({best_result["v_nom"]} kV) - Geometry Hash'
                    else:
                        best_result['match_quality'] = 'Parallel Circuit - Geometry Hash'
    except Exception as e:
        print(f"Error in hash matching: {e}")
        # Continue processing even if an error occurs

    print(f"Matched {matches_made} network lines using geometry hash approach")

    # Always return the matching_results
    return matching_results


def main():
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    print("Loading JAO lines...")
    jao_gdf = load_jao_lines()
    print(f"Loaded {len(jao_gdf)} JAO lines")

    print("Loading network lines (excluding 110kV)...")
    network_gdf = load_network_lines()
    print(f"Loaded {len(network_gdf)} network lines")

    # Identify duplicate JAO lines with identical geometries
    jao_to_group, geometry_groups = identify_duplicate_jao_lines(jao_gdf)

    # Use a larger distance threshold
    distance_threshold_meters = 500  # 500 meters around substations
    print("Finding nearest points for JAO endpoints...")
    nearest_points_dict = find_nearest_points(
        jao_gdf, network_gdf,
        max_alternatives=10,
        distance_threshold_meters=distance_threshold_meters
    )

    # Build network graph with NO automatic connections
    print("Building network graph (without extra connections)...")
    G = build_network_graph(network_gdf)
    print(f"Graph has {len(G.nodes)} nodes and {len(G.edges)} edges")

    # Repair the graph to add missing connections
    G = repair_network_graph(G, network_gdf, connection_threshold_meters=50)
    print(f"Repaired graph has {len(G.nodes)} nodes and {len(G.edges)} edges")

    # Find matching network lines with special handling for duplicates
    print("Finding matching network lines with special handling for duplicates...")
    matching_results = find_matching_network_lines_with_duplicates(
        jao_gdf, network_gdf, nearest_points_dict, G,
        duplicate_groups=geometry_groups,
        max_reuse=3,
        min_length_ratio=0.7,  # Network must be at least 70% of JAO length
        max_length_ratio=3  # Network can be at most 3x JAO length
    )

    # Add special matching for parallel circuit JAO lines with unmatched network lines
    print("Running specialized matching for parallel circuit JAO lines...")
    matching_results = match_parallel_circuit_jao_with_network(matching_results, jao_gdf, network_gdf, G,
                                                               nearest_points_dict)

    # Double check for unmarked duplicates
    print("Double-checking duplicate handling...")
    duplicate_count_before = sum(1 for r in matching_results if r.get('is_duplicate', False))

    # Create a map of already matched JAO lines
    matched_jaos = {r['jao_id']: r for r in matching_results if r['matched']}

    # Check for any duplicates that weren't properly marked
    for geom_wkt, jao_ids in geometry_groups.items():
        if len(jao_ids) > 1:  # Only process duplicate groups
            # Find the first matched JAO in this group
            first_matched = None
            for jao_id in jao_ids:
                if jao_id in matched_jaos and not matched_jaos[jao_id].get('is_duplicate', False):
                    first_matched = matched_jaos[jao_id]
                    break

            # If we found a matched one, ensure all others in the group are marked as duplicates
            if first_matched:
                for jao_id in jao_ids:
                    if jao_id != first_matched['jao_id'] and jao_id in matched_jaos:
                        # Check if this is already properly marked as duplicate
                        if not matched_jaos[jao_id].get('is_duplicate', False):
                            print(f"Fixing duplicate marking for JAO {jao_id}")
                            matched_jaos[jao_id]['is_duplicate'] = True
                            matched_jaos[jao_id]['duplicate_of'] = first_matched['jao_id']
                            matched_jaos[jao_id]['match_quality'] = 'Parallel Circuit (duplicate geometry)'

    duplicate_count_after = sum(1 for r in matching_results if r.get('is_duplicate', False))
    print(f"Fixed {duplicate_count_after - duplicate_count_before} additional duplicate markings")

    # Match parallel voltage circuits
    print("Finding parallel voltage circuits...")
    matching_results = match_parallel_voltage_circuits(jao_gdf, network_gdf, matching_results)

    # Ensure similar JAO lines use consistent network lines
    matching_results = improve_matches_by_similarity(matching_results, jao_gdf, network_gdf)

    # Add path-based matching for parallel circuit JAOs
    matching_results = match_parallel_circuit_path_based(matching_results, jao_gdf, network_gdf, G)

    # Match remaining lines using advanced geometric approach
    print("Applying advanced geometric matching for remaining lines...")
    matching_results = match_remaining_lines_by_geometry(jao_gdf, network_gdf, matching_results)

    # NEW METHODS FROM CHATGPT RESPONSE:

    # 1. First apply corridor-based matching (recommended as first step)
    print("Applying corridor-based matching for parallel lines...")
    matching_results = corridor_parallel_match(matching_results, jao_gdf, network_gdf)

    # 2. Apply progressive relaxation matching
    print("Applying progressive relaxation matching...")
    matching_results = match_identical_network_geometries_aggressive(matching_results, jao_gdf, network_gdf)

    # 3. Apply geometry hash matching
    print("Applying geometry hash matching...")
    matching_results = match_network_lines_by_geometry_hash(matching_results, network_gdf)

    # Continue with remaining matching methods

    # Match remaining parallel network lines
    print("Matching remaining parallel network lines...")
    matching_results = match_remaining_parallel_network_lines(matching_results, jao_gdf, network_gdf)

    # Share network lines between parallel JAOs
    print("Sharing network lines between parallel JAO lines...")
    matching_results = share_network_lines_among_parallel_jaos(matching_results, jao_gdf)

    # Try to convert geometric matches to path-based matches
    matching_results = convert_geometric_to_path_matches(matching_results, G, jao_gdf, network_gdf, nearest_points_dict)

    # Debug special case for JAO 97
    print("Debugging special case for JAO 97...")
    matching_results = debug_specific_jao_match(jao_gdf, network_gdf, matching_results,
                                                jao_id_to_debug="97",
                                                target_network_ids=["Line_8160", "Line_30733", "Line_30181",
                                                                    "Line_17856"])

    # Allocate electrical parameters
    print("Allocating electrical parameters...")
    enhanced_results = allocate_electrical_parameters(jao_gdf, network_gdf, matching_results)

    # Create enhanced visualization with duplicate handling
    print("Creating enhanced visualization with duplicate handling...")
    duplicate_map_file = visualize_results_with_duplicates(jao_gdf, network_gdf, matching_results)

    # Improve visualization of unmatched network lines
    with open(duplicate_map_file, 'r') as f:
        html_content = f.read()

    improved_html = improve_visualization_of_unmatched_network_lines(html_content)

    with open(duplicate_map_file, 'w') as f:
        f.write(improved_html)

    # Create regular visualization and summary
    print("Creating regular visualization and summary...")
    map_file = visualize_results(jao_gdf, network_gdf, matching_results)

    # Also improve the regular visualization
    with open(map_file, 'r') as f:
        html_content = f.read()

    improved_html = improve_visualization_of_unmatched_network_lines(html_content)

    with open(map_file, 'w') as f:
        f.write(improved_html)

    # Create summary table
    enhanced_summary_file = create_enhanced_summary_table(jao_gdf, network_gdf, matching_results)

    print(f"Results saved to:")
    print(f"  - Regular Map: {map_file}")
    print(f"  - Map with Duplicate Handling: {duplicate_map_file}")
    print(f"  - Enhanced Summary with Parameters: {enhanced_summary_file}")

    # Print summary statistics
    total_jao_lines = len(matching_results)
    matched_lines = sum(result['matched'] for result in matching_results)
    duplicate_count = sum(1 for result in matching_results if result.get('is_duplicate', False))
    parallel_count = sum(1 for result in matching_results if result.get('is_parallel_circuit', False))
    geometric_count = sum(1 for result in matching_results if result.get('is_geometric_match', False))
    parallel_voltage_count = sum(1 for result in matching_results if result.get('is_parallel_voltage_circuit', False))

    regular_matches = matched_lines - duplicate_count - parallel_count - geometric_count - parallel_voltage_count

    print(f"\nTotal JAO lines: {total_jao_lines}")
    print(f"Matched lines: {matched_lines} ({matched_lines / total_jao_lines * 100:.1f}%)")
    print(f"  - Regular matches: {regular_matches} ({regular_matches / total_jao_lines * 100:.1f}%)")
    print(f"  - Geometric matches: {geometric_count} ({geometric_count / total_jao_lines * 100:.1f}%)")
    print(f"  - Duplicate JAO lines: {duplicate_count} ({duplicate_count / total_jao_lines * 100:.1f}%)")
    print(f"  - Parallel circuit JAO lines: {parallel_count} ({parallel_count / total_jao_lines * 100:.1f}%)")
    print(
        f"  - Parallel voltage circuit JAO lines: {parallel_voltage_count} ({parallel_voltage_count / total_jao_lines * 100:.1f}%)")
    print(
        f"Unmatched lines: {total_jao_lines - matched_lines} ({(total_jao_lines - matched_lines) / total_jao_lines * 100:.1f}%)")

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

    # Calculate network matching statistics
    total_network_lines = len(network_gdf)
    matched_network_ids = set()
    for result in matching_results:
        if result['matched'] and 'network_ids' in result:
            for network_id in result['network_ids']:
                matched_network_ids.add(str(network_id))

    matched_network_count = len(matched_network_ids)
    unmatched_network_count = total_network_lines - matched_network_count

    print(f"\nNetwork Lines Statistics:")
    print(f"Total Network Lines: {total_network_lines}")
    print(f"Matched Network Lines: {matched_network_count} ({matched_network_count / total_network_lines * 100:.1f}%)")
    print(
        f"Unmatched Network Lines: {unmatched_network_count} ({unmatched_network_count / total_network_lines * 100:.1f}%)")

    # Get count by voltage
    n220_lines = sum(1 for _, row in network_gdf.iterrows() if row['v_nom'] == 220)
    n220_matched = sum(
        1 for _, row in network_gdf.iterrows() if row['v_nom'] == 220 and str(row['id']) in matched_network_ids)
    n400_lines = sum(1 for _, row in network_gdf.iterrows() if row['v_nom'] in [380, 400])
    n400_matched = sum(
        1 for _, row in network_gdf.iterrows() if row['v_nom'] in [380, 400] and str(row['id']) in matched_network_ids)

    print(f"220 kV network lines: {n220_matched}/{n220_lines} matched ({n220_matched / n220_lines * 100:.1f}%)")
    print(f"400 kV network lines: {n400_matched}/{n400_lines} matched ({n400_matched / n400_lines * 100:.1f}%)")


if __name__ == "__main__":
    main()