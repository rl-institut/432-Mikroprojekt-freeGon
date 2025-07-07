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


# Add this import at the top of your file
from rtree import index


def find_nearest_points(dlr_gdf, network_gdf, max_alternatives=5, distance_threshold_meters=1000, debug_lines=None):
    """
    Find nearest points in the network for DLR endpoints with improved substation handling.

    Parameters:
    - max_alternatives: Number of alternative endpoints to store
    - distance_threshold_meters: Maximum distance in meters to consider an endpoint match
    - debug_lines: List of specific DLR IDs to debug in detail
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

    # Find nearest network endpoints for each DLR endpoint
    for idx, row in dlr_gdf.iterrows():
        dlr_id = str(row['id'])
        dlr_name = str(row['NE_name'])
        dlr_voltage = int(row['v_nom'])

        # Check if this is a debug line
        is_debug = debug_lines is not None and dlr_id in debug_lines

        if is_debug:
            print(f"\n===== DEBUGGING DLR LINE {dlr_id} ({dlr_name}) =====")

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

        # Find nearest network endpoint for DLR start point
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
                voltage_match = (dlr_voltage == 220 and network_voltage == 220) or (
                            dlr_voltage == 400 and network_voltage == 380)
                print(
                    f"    Network {network_id} ({pos} point) - Distance: {dist_meters:.2f}m, Voltage: {network_voltage} kV (match: {voltage_match})")
                print(f"      Coordinates: ({network_point.x}, {network_point.y})")
                print(f"      Within threshold: {dist_meters <= distance_threshold_meters}")

            # Consider voltage constraint - try to match same voltage if possible
            voltage_match = (dlr_voltage == 220 and network_voltage == 220) or (
                        dlr_voltage == 400 and network_voltage == 380)

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

        # Find nearest network endpoint for DLR end point
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
                voltage_match = (dlr_voltage == 220 and network_voltage == 220) or (
                            dlr_voltage == 400 and network_voltage == 380)
                print(
                    f"    Network {network_id} ({pos} point) - Distance: {dist_meters:.2f}m, Voltage: {network_voltage} kV (match: {voltage_match})")
                print(f"      Coordinates: ({network_point.x}, {network_point.y})")
                print(f"      Within threshold: {dist_meters <= distance_threshold_meters}")

            # Consider voltage constraint
            voltage_match = (dlr_voltage == 220 and network_voltage == 220) or (
                        dlr_voltage == 400 and network_voltage == 380)

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
    total_dlr = len(dlr_gdf)
    start_matches = sum(1 for data in nearest_points_dict.values() if data['start_nearest'] is not None)
    end_matches = sum(1 for data in nearest_points_dict.values() if data['end_nearest'] is not None)
    both_matches = sum(1 for data in nearest_points_dict.values()
                       if data['start_nearest'] is not None and data['end_nearest'] is not None)

    print(f"Endpoint matching statistics (distance threshold: {distance_threshold_meters} meters):")
    print(f"  DLR lines with start point matched: {start_matches}/{total_dlr} ({start_matches / total_dlr * 100:.1f}%)")
    print(f"  DLR lines with end point matched: {end_matches}/{total_dlr} ({end_matches / total_dlr * 100:.1f}%)")
    print(
        f"  DLR lines with both endpoints matched: {both_matches}/{total_dlr} ({both_matches / total_dlr * 100:.1f}%)")

    return nearest_points_dict



# Function to allocate electrical parameters from DLR to network lines - using CSV length data
def allocate_electrical_parameters(dlr_gdf, network_gdf, matching_results):
    # Enhanced results with electrical parameters
    enhanced_results = []

    for result in matching_results:
        if result['matched'] and result['network_ids']:
            dlr_id = result['dlr_id']
            dlr_rows = dlr_gdf[dlr_gdf['id'].astype(str) == dlr_id]

            if dlr_rows.empty:
                print(f"Warning: DLR ID {dlr_id} not found in DLR GeoDataFrame")
                enhanced_results.append(result.copy())
                continue

            dlr_row = dlr_rows.iloc[0]

            # Get DLR electrical parameters (if available)
            dlr_r = float(dlr_row.get('r', 0)) if 'r' in dlr_row else 0
            dlr_x = float(dlr_row.get('x', 0)) if 'x' in dlr_row else 0
            dlr_b = float(dlr_row.get('b', 0)) if 'b' in dlr_row else 0

            # Prioritize using the 'length' column for DLR
            if 'length' in dlr_row:
                dlr_length_km = float(dlr_row['length'])
                print(f"Using 'length' column for DLR {dlr_id}: {dlr_length_km} km")
            else:
                # Fallback to other methods if 'length' is not available
                dlr_length_km = result.get('dlr_length', 0) / 1000 if 'dlr_length' in result else 0

                # If length is still 0, try to calculate it from geometry
                if dlr_length_km == 0 and hasattr(dlr_row, 'geometry') and dlr_row.geometry is not None:
                    try:
                        # Try to get length in meters and convert to km
                        dlr_length_m = float(dlr_row.geometry.length)
                        dlr_length_km = dlr_length_m / 1000
                        print(f"Calculated DLR length for {dlr_id}: {dlr_length_km} km")
                    except Exception as e:
                        print(f"Error calculating DLR length for {dlr_id}: {e}")

            # Calculate DLR per-km values (with safety check for zero division)
            dlr_r_per_km = dlr_r / dlr_length_km if dlr_length_km > 0 else 0
            dlr_x_per_km = dlr_x / dlr_length_km if dlr_length_km > 0 else 0
            dlr_b_per_km = dlr_b / dlr_length_km if dlr_length_km > 0 else 0

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

            # If total network length is still 0, use the DLR length as a fallback
            if total_network_length_km == 0:
                total_network_length_km = dlr_length_km
                print(f"Warning: Using DLR length as fallback for network length for DLR {dlr_id}")

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

                    # Allocate DLR parameters based on length ratio
                    allocated_r = dlr_r * segment_ratio
                    allocated_x = dlr_x * segment_ratio
                    allocated_b = dlr_b * segment_ratio

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
                        # If length is zero, use DLR per-km values as fallback
                        original_r_per_km = 0
                        original_x_per_km = 0
                        original_b_per_km = 0

                        allocated_r_per_km = dlr_r_per_km
                        allocated_x_per_km = dlr_x_per_km
                        allocated_b_per_km = dlr_b_per_km

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
            result_with_parameters['dlr_r'] = dlr_r
            result_with_parameters['dlr_x'] = dlr_x
            result_with_parameters['dlr_b'] = dlr_b
            result_with_parameters['dlr_length_km'] = dlr_length_km
            result_with_parameters['dlr_r_per_km'] = dlr_r_per_km
            result_with_parameters['dlr_x_per_km'] = dlr_x_per_km
            result_with_parameters['dlr_b_per_km'] = dlr_b_per_km
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


def identify_duplicate_dlr_lines(dlr_gdf):
    """
    Identify groups of DLR lines with identical geometries.
    Returns a dictionary mapping DLR IDs to their group ID.
    """
    print("Identifying duplicate DLR lines...")

    # Create a dictionary to store groups
    geometry_groups = {}
    dlr_to_group = {}

    # Use WKT representation of geometry as key to group identical geometries
    for idx, row in dlr_gdf.iterrows():
        dlr_id = str(row['id'])
        geom_wkt = row.geometry.wkt

        if geom_wkt in geometry_groups:
            # Add to existing group
            geometry_groups[geom_wkt].append(dlr_id)
        else:
            # Create new group
            geometry_groups[geom_wkt] = [dlr_id]

    # Assign group IDs only to geometries with multiple DLR lines
    group_id = 0
    for geom_wkt, dlr_ids in geometry_groups.items():
        if len(dlr_ids) > 1:
            group_id += 1
            print(f"Group {group_id}: Found {len(dlr_ids)} duplicate DLR lines with identical geometry:")
            for dlr_id in dlr_ids:
                dlr_to_group[dlr_id] = group_id
                print(f"  - DLR {dlr_id}")

    print(f"Found {group_id} groups of duplicate DLR lines")
    return dlr_to_group, geometry_groups


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


def find_matching_network_lines_with_duplicates(dlr_gdf, network_gdf, nearest_points_dict, G,
                                               duplicate_groups, max_reuse=2, max_paths_to_try=20,
                                               min_length_ratio=0.5, max_length_ratio=1.7):
    """
    Find matching network lines with special handling for duplicate DLR lines.
    Evaluates multiple possible paths and selects the best one based on length and voltage match.
    """
    results = []

    # Track how many times each network line has been used
    network_line_usage = {str(row['id']): 0 for _, row in network_gdf.iterrows()}

    # Create a reverse lookup from DLR ID to its duplicate group
    dlr_to_group = {}
    for geom_wkt, dlr_ids in duplicate_groups.items():
        if len(dlr_ids) > 1:
            # Only track duplicates (groups with multiple DLR lines)
            for dlr_id in dlr_ids:
                dlr_to_group[dlr_id] = geom_wkt

    # Track which duplicate groups have been matched
    matched_groups = {}

    # Process DLR lines by ID order for reproducibility
    dlr_with_idx = [(idx, row) for idx, row in dlr_gdf.iterrows()]
    dlr_with_idx.sort(key=lambda x: str(x[1]['id']))

    print(f"Processing {len(dlr_with_idx)} DLR lines...")
    print(f"Using strict reuse limit of {max_reuse}")
    print(f"Evaluating up to {max_paths_to_try} possible paths per DLR line")
    print(f"Length ratio constraints: network/DLR must be between {min_length_ratio:.1f} and {max_length_ratio:.1f}")

    for idx, row in dlr_with_idx:
        dlr_id = str(row['id'])
        dlr_name = str(row['NE_name'])
        dlr_voltage = int(row['v_nom'])

        # Calculate DLR length properly
        dlr_length_meters = calculate_length_meters(row.geometry)
        dlr_length_km = dlr_length_meters / 1000.0

        print(f"\nProcessing DLR line {dlr_id} ({dlr_name}) with length {dlr_length_km:.2f} km")

        # Handle duplicate DLR lines
        is_duplicate = dlr_id in dlr_to_group
        duplicate_group = dlr_to_group.get(dlr_id, None)

        if is_duplicate and duplicate_group in matched_groups:
            # This is a duplicate of an already matched DLR line
            print(f"\nProcessing duplicate DLR line {dlr_id} ({dlr_name})")
            print(f"  This is a duplicate of already matched DLR(s): {matched_groups[duplicate_group]}")

            # Copy the match from the first matched DLR in this group
            first_match = None
            for result in results:
                if result['dlr_id'] in matched_groups[duplicate_group]:
                    first_match = result
                    break

            if first_match and first_match['matched']:
                # Create a duplicate match with special status
                duplicate_result = {
                    'dlr_id': dlr_id,
                    'dlr_name': dlr_name,
                    'v_nom': int(dlr_voltage),
                    'matched': True,
                    'is_duplicate': True,  # Make sure this is set to True
                    'duplicate_of': first_match['dlr_id'],
                    'path': first_match['path'].copy() if 'path' in first_match else [],
                    'network_ids': first_match['network_ids'].copy() if 'network_ids' in first_match else [],
                    'path_length': first_match.get('path_length', 0),
                    'dlr_length': float(dlr_length_meters),  # Use meters consistently
                    'length_ratio': first_match.get('length_ratio', 1.0),
                    'match_quality': 'Parallel Circuit (duplicate geometry)'
                }

                results.append(duplicate_result)
                print(f"  Marked as parallel circuit of {first_match['dlr_id']}")

                # Make sure this duplicate is recorded in matched_groups
                if duplicate_group not in matched_groups:
                    matched_groups[duplicate_group] = []
                matched_groups[duplicate_group].append(dlr_id)

                continue

        # Get node IDs for start and end points
        if idx not in nearest_points_dict or nearest_points_dict[idx]['start_nearest'] is None:
            print(f"  No start point match for {dlr_id}")
            results.append({
                'dlr_id': dlr_id,
                'dlr_name': dlr_name,
                'v_nom': int(dlr_voltage),
                'matched': False,
                'path': [],
                'network_ids': [],
                'match_quality': 'No match - endpoints not found'
            })
            continue

        start_idx, start_pos = nearest_points_dict[idx]['start_nearest']
        if idx not in nearest_points_dict or nearest_points_dict[idx]['end_nearest'] is None:
            print(f"  No end point match for {dlr_id}")
            results.append({
                'dlr_id': dlr_id,
                'dlr_name': dlr_name,
                'v_nom': int(dlr_voltage),
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
            print(f"  Start and end points are the same for {dlr_id}")
            results.append({
                'dlr_id': dlr_id,
                'dlr_name': dlr_name,
                'v_nom': int(dlr_voltage),
                'matched': False,
                'path': [],
                'network_ids': [],
                'match_quality': 'No match - start and end are the same'
            })
            continue

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
                        if not ((dlr_voltage == 220 and data['voltage'] == 220) or
                                (dlr_voltage == 400 and data['voltage'] in [380, 400]) or
                                (dlr_voltage == 380 and data['voltage'] in [380, 400])):
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
                        length_ratio = path_length / dlr_length_meters if dlr_length_meters > 0 else float('inf')

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
                            length_ratio = path_length / dlr_length_meters if dlr_length_meters > 0 else float('inf')

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
                                length_ratio = path_length / dlr_length_meters if dlr_length_meters > 0 else float(
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
            print(f"  No valid paths found for {dlr_id}")
            results.append({
                'dlr_id': dlr_id,
                'dlr_name': dlr_name,
                'v_nom': int(dlr_voltage),
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
        print(f"  Network path length: {best_path['path_length'] / 1000:.2f} km vs DLR length: {dlr_length_km:.2f} km")

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
            'dlr_id': dlr_id,
            'dlr_name': dlr_name,
            'v_nom': int(dlr_voltage),
            'matched': True,
            'is_duplicate': False,
            'path': [str(p) for p in best_path['path']],
            'network_ids': best_path['network_ids'],
            'path_length': float(best_path['path_length']),
            'dlr_length': float(dlr_length_meters),
            'length_ratio': float(best_path['length_ratio']),
            'match_quality': best_path['match_quality'],
            'constraints_used': best_path['constraints_used']
        }

        results.append(result)

        # If this is a duplicate DLR line, record that its group has been matched
        if is_duplicate:
            if duplicate_group not in matched_groups:
                matched_groups[duplicate_group] = []
            matched_groups[duplicate_group].append(dlr_id)

        print(f"  Successfully matched {dlr_id} with {len(best_path['network_ids'])} network lines")

    # Print usage statistics and other info as before...
    # (rest of the function remains the same)

    return results


def match_remaining_lines_by_geometry(dlr_gdf, network_gdf, matching_results, buffer_distance=0.005,
                                      snap_tolerance=300, angle_tolerance=30, min_dir_cos=0.866,
                                      min_length_ratio=0.3, max_length_ratio=1.7):
    """
    Apply a sophisticated geometric matching approach for DLR lines that aren't matched yet.
    Uses a combination of direction cosine, endpoint proximity, and overlap metrics.

    Parameters:
    - dlr_gdf: GeoDataFrame with DLR lines
    - network_gdf: GeoDataFrame with network lines
    - matching_results: Existing matching results
    - buffer_distance: Buffer distance in degrees (for overlap)
    - snap_tolerance: Maximum distance in meters to connect endpoints
    - angle_tolerance: Maximum angle difference in degrees to consider lines aligned
    - min_dir_cos: Minimum direction cosine (cos of angle between lines)
    - min_length_ratio: Minimum acceptable ratio of network/dlr length (default 0.3)
    - max_length_ratio: Maximum acceptable ratio of network/dlr length (default 1.7)
    """
    import numpy as np
    from shapely.geometry import LineString, Point, MultiLineString
    from shapely.ops import linemerge, unary_union
    from scipy.spatial import cKDTree

    print("\nAttempting additional matches using advanced geometric approach...")
    print(f"Length ratio constraints: {min_length_ratio:.1f} to {max_length_ratio:.1f}")

    # Identify unmatched DLR lines
    matched_dlr_ids = set(result['dlr_id'] for result in matching_results if result['matched'])

    # Identify matched network lines
    matched_network_ids = set()
    for result in matching_results:
        if result['matched'] and 'network_ids' in result:
            for network_id in result['network_ids']:
                matched_network_ids.add(str(network_id))

    # Get unmatched DLR lines
    unmatched_dlr_rows = []
    for idx, row in dlr_gdf.iterrows():
        dlr_id = str(row['id'])
        if dlr_id not in matched_dlr_ids:
            unmatched_dlr_rows.append((idx, row))

    # Get unmatched network lines
    unmatched_network_gdf = network_gdf[~network_gdf['id'].astype(str).isin(matched_network_ids)].copy()

    print(f"Found {len(unmatched_dlr_rows)} unmatched DLR lines to process")
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

    # Process each unmatched DLR line
    additional_matches = 0
    new_matches = []

    # Store matched network lines to prevent reuse across different DLR lines
    newly_matched_network_ids = set()

    # For each unmatched DLR line
    for idx, row in unmatched_dlr_rows:
        dlr_id = str(row['id'])
        dlr_name = str(row['NE_name'])
        dlr_voltage = int(row['v_nom'])
        dlr_geometry = row.geometry

        # Calculate dlr length in meters (approximate)
        meters_per_degree = meters_per_degree_at_lat(dlr_geometry.centroid.y)
        dlr_length_m = float(dlr_geometry.length) * meters_per_degree

        print(f"\nProcessing unmatched DLR line {dlr_id} ({dlr_name}), length: {dlr_length_m / 1000:.2f} km")

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
            voltage_match = same_voltage(dlr_voltage, net_voltage)
            voltage_factor = 1.0 if voltage_match else 0.5

            # Calculate direction cosine (alignment)
            cosine = dir_cos(dlr_geometry, net_geometry)

            # Skip if lines are not remotely aligned
            if cosine < min_dir_cos:
                continue

            # Check proximity between endpoints
            dlr_endpoints = endpts(dlr_geometry)
            net_endpoints = endpts(net_geometry)

            min_endpoint_distance = float('inf')
            for d_pt in dlr_endpoints:
                for n_pt in net_endpoints:
                    dist = d_pt.distance(n_pt)
                    min_endpoint_distance = min(min_endpoint_distance, dist)

            # Convert min_endpoint_distance to meters (approximate)
            # This is a rough conversion and depends on latitude
            min_endpoint_distance_m = min_endpoint_distance * meters_per_degree

            # Calculate overlap
            overlap = overlap_length(dlr_geometry, net_geometry, tol=buffer_distance)
            overlap_ratio = overlap / min(dlr_geometry.length, net_geometry.length)

            # Calculate Hausdorff distance
            h_dist = hausdorff_distance(dlr_geometry, net_geometry)
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
                    length_ratio = combo_length / dlr_length_m if dlr_length_m > 0 else float('inf')
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
                print(f"  Length ratio (network/dlr): {best_length_ratio:.2f}")

                # Calculate total path length
                path_length = sum(match['length_m'] for match in best_matches)

                # Create a result for this match
                match_result = {
                    'dlr_id': dlr_id,
                    'dlr_name': dlr_name,
                    'v_nom': dlr_voltage,
                    'matched': True,
                    'is_duplicate': False,
                    'is_geometric_match': True,  # Flag to indicate this was matched geometrically
                    'path': [],  # No path in graph
                    'network_ids': network_ids,
                    'path_length': float(path_length),
                    'dlr_length': float(dlr_length_m),
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
            print(f"  No geometric matches found for {dlr_id}")

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


def visualize_results_with_duplicates(dlr_gdf, network_gdf, matching_results):
    """Create a visualization that highlights duplicate DLR lines and includes merged network paths."""
    # Create GeoJSON data for lines
    dlr_features = []
    network_features = []
    merged_network_features = []  # New list for merged network paths

    # Create sets to track which lines are matched or duplicates
    matched_dlr_ids = set()
    duplicate_dlr_ids = set()
    geometric_match_dlr_ids = set()

    # Create sets to track network lines by match type
    regular_matched_network_ids = set()
    geometric_matched_network_ids = set()
    duplicate_matched_network_ids = set()

    # First, identify all matched DLR and network lines by type
    for result in matching_results:
        if result['matched']:
            dlr_id = str(result['dlr_id'])
            network_ids = result.get('network_ids', [])

            if result.get('is_duplicate', False):
                duplicate_dlr_ids.add(dlr_id)
                for network_id in network_ids:
                    duplicate_matched_network_ids.add(str(network_id))
            elif result.get('is_geometric_match', False):
                geometric_match_dlr_ids.add(dlr_id)
                for network_id in network_ids:
                    geometric_matched_network_ids.add(str(network_id))
            else:
                matched_dlr_ids.add(dlr_id)
                for network_id in network_ids:
                    regular_matched_network_ids.add(str(network_id))

    # Add DLR lines to GeoJSON with matched/unmatched/duplicate/geometric status
    for idx, row in dlr_gdf.iterrows():
        # Create a unique ID for each line
        line_id = f"dlr_{row['id']}"
        coords = list(row.geometry.coords)

        # Check if this DLR line is matched or a duplicate or geometric match
        dlr_id = str(row['id'])
        is_matched = dlr_id in matched_dlr_ids
        is_duplicate = dlr_id in duplicate_dlr_ids
        is_geometric = dlr_id in geometric_match_dlr_ids

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
                "type": "dlr",
                "id": dlr_id,
                "name": str(row['NE_name']),
                "voltage": int(row['v_nom']),
                "status": status,
                "tooltip": f"DLR: {dlr_id} - {row['NE_name']} ({row['v_nom']} kV) - {tooltip_status}"
            },
            "geometry": {
                "type": "LineString",
                "coordinates": [[float(x), float(y)] for x, y in coords]
            }
        }
        dlr_features.append(feature)

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

    # Create merged network path features for each DLR match
    from shapely.geometry import LineString, MultiLineString
    from shapely.ops import linemerge, unary_union
    import numpy as np

    for result in matching_results:
        if not result['matched'] or not result.get('network_ids'):
            continue

        dlr_id = str(result['dlr_id'])
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

            # Determine line status based on DLR status
            if dlr_id in duplicate_dlr_ids:
                status = "duplicate"
                tooltip_status = "Parallel Circuit"
            elif dlr_id in geometric_match_dlr_ids:
                status = "geometric"
                tooltip_status = "Geometric Match"
            else:
                status = "matched"
                tooltip_status = "Matched"

            # Calculate the match quality score for the tooltip
            match_quality = result.get('match_quality', 'Unknown')
            length_ratio = result.get('length_ratio', 0)

            # Create a unique ID for the merged path
            merged_id = f"merged-net-dlr-{dlr_id}"

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
                    "dlr_id": dlr_id,
                    "network_ids": ",".join(network_ids),
                    "status": status,
                    "tooltip": f"Merged Network Path for DLR {dlr_id} ({len(network_ids)} lines) - {match_quality} - Ratio: {length_ratio:.2f}"
                },
                "geometry": geometry
            }

            merged_network_features.append(feature)

        except Exception as e:
            print(f"Error creating merged network path for DLR {dlr_id}: {e}")

    # Create GeoJSON collections
    dlr_collection = {"type": "FeatureCollection", "features": dlr_features}
    network_collection = {"type": "FeatureCollection", "features": network_features}
    merged_network_collection = {"type": "FeatureCollection", "features": merged_network_features}

    # Convert to JSON strings
    dlr_json = json.dumps(dlr_collection)
    network_json = json.dumps(network_collection)
    merged_network_json = json.dumps(merged_network_collection)

    # Create a complete HTML file from scratch
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>DLR-Network Line Matching Results with Merged Paths</title>

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
                        <div class="filter-option active" data-filter="dlr-matched">Matched DLR Lines</div>
                        <div class="filter-option active" data-filter="dlr-duplicate">Parallel Circuit DLR Lines</div>
                        <div class="filter-option active" data-filter="dlr-geometric">Geometric Match DLR Lines</div>
                        <div class="filter-option active" data-filter="dlr-unmatched">Unmatched DLR Lines</div>
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
                        <div>Matched DLR Lines (Graph-based)</div>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background-color: #9932CC;"></div>
                        <div>Parallel Circuit DLR Lines</div>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background-color: #00BFFF;"></div>
                        <div>Geometric Match DLR Lines</div>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background-color: red;"></div>
                        <div>Unmatched DLR Lines</div>
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
            var dlrLines = {dlr_json};
            var networkLines = {network_json};
            var mergedNetworkPaths = {merged_network_json};

            // Define styling for the DLR lines
            function dlrStyle(feature) {{
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

            // Create the DLR layers
            var dlrMatchedLayer = L.geoJSON(dlrLines, {{
                filter: function(feature) {{
                    return feature.properties.status === "matched";
                }},
                style: dlrStyle,
                onEachFeature: function(feature, layer) {{
                    layer.bindTooltip(feature.properties.tooltip);
                    layer.on('click', function() {{
                        highlightFeature(feature.id, feature.properties.id);
                    }});
                }}
            }}).addTo(map);

            var dlrDuplicateLayer = L.geoJSON(dlrLines, {{
                filter: function(feature) {{
                    return feature.properties.status === "duplicate";
                }},
                style: dlrStyle,
                onEachFeature: function(feature, layer) {{
                    layer.bindTooltip(feature.properties.tooltip);
                    layer.on('click', function() {{
                        highlightFeature(feature.id, feature.properties.id);
                    }});
                }}
            }}).addTo(map);

            var dlrGeometricLayer = L.geoJSON(dlrLines, {{
                filter: function(feature) {{
                    return feature.properties.status === "geometric";
                }},
                style: dlrStyle,
                onEachFeature: function(feature, layer) {{
                    layer.bindTooltip(feature.properties.tooltip);
                    layer.on('click', function() {{
                        highlightFeature(feature.id, feature.properties.id);
                    }});
                }}
            }}).addTo(map);

            var dlrUnmatchedLayer = L.geoJSON(dlrLines, {{
                filter: function(feature) {{
                    return feature.properties.status === "unmatched";
                }},
                style: dlrStyle,
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
                        // When clicking on a merged path, highlight both the path and the corresponding DLR line
                        highlightFeature(feature.id, feature.properties.dlr_id);
                    }});
                }}
            }}).addTo(map);

            // Function to highlight a feature and optionally a related DLR feature
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

                // If a related DLR ID is provided, highlight that DLR line too
                function checkAndHighlightDLR(layer) {{
                    layer.eachLayer(function(leafletLayer) {{
                        if (relatedDlrId && leafletLayer.feature.properties.type === 'dlr' && 
                            leafletLayer.feature.properties.id === relatedDlrId) {{
                            leafletLayer.setStyle({{className: 'highlighted'}});
                            if (leafletLayer._path) leafletLayer._path.classList.add('highlighted');
                        }}
                    }});
                }}

                // If a related merged network path exists for this DLR, highlight it too
                function checkAndHighlightMergedPath(layer) {{
                    layer.eachLayer(function(leafletLayer) {{
                        if (relatedDlrId && leafletLayer.feature.properties.type === 'merged_network' && 
                            leafletLayer.feature.properties.dlr_id === relatedDlrId) {{
                            leafletLayer.setStyle({{className: 'highlighted'}});
                            if (leafletLayer._path) leafletLayer._path.classList.add('highlighted');
                        }}
                    }});
                }}

                // Check all layers for the specific feature
                checkAndHighlight(dlrMatchedLayer);
                checkAndHighlight(dlrDuplicateLayer);
                checkAndHighlight(dlrGeometricLayer);
                checkAndHighlight(dlrUnmatchedLayer);
                checkAndHighlight(networkMatchedLayer);
                checkAndHighlight(networkDuplicateLayer);
                checkAndHighlight(networkGeometricLayer);
                checkAndHighlight(networkUnmatchedLayer);
                checkAndHighlight(mergedNetworkLayer);

                // If a related DLR ID is provided, highlight that DLR line
                if (relatedDlrId) {{
                    checkAndHighlightDLR(dlrMatchedLayer);
                    checkAndHighlightDLR(dlrDuplicateLayer);
                    checkAndHighlightDLR(dlrGeometricLayer);
                    checkAndHighlightDLR(dlrUnmatchedLayer);

                    // Also highlight any merged network path for this DLR
                    checkAndHighlightMergedPath(mergedNetworkLayer);
                }}
            }}

            // Function to clear highlights
            function clearHighlights() {{
                function resetStyle(layer) {{
                    layer.eachLayer(function(leafletLayer) {{
                        if (leafletLayer.feature.properties.type === 'dlr') {{
                            leafletLayer.setStyle(dlrStyle(leafletLayer.feature));
                        }} else if (leafletLayer.feature.properties.type === 'merged_network') {{
                            leafletLayer.setStyle(mergedNetworkStyle(leafletLayer.feature));
                        }} else {{
                            leafletLayer.setStyle(networkStyle(leafletLayer.feature));
                        }}
                        if (leafletLayer._path) leafletLayer._path.classList.remove('highlighted');
                    }});
                }}

                // Reset all layers
                resetStyle(dlrMatchedLayer);
                resetStyle(dlrDuplicateLayer);
                resetStyle(dlrGeometricLayer);
                resetStyle(dlrUnmatchedLayer);
                resetStyle(networkMatchedLayer);
                resetStyle(networkDuplicateLayer);
                resetStyle(networkGeometricLayer);
                resetStyle(networkUnmatchedLayer);
                resetStyle(mergedNetworkLayer);
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
                        feature: feature,
                        dlrId: feature.properties.id
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
                        text: "Merged Path: " + feature.properties.dlr_id + " (" + feature.properties.network_ids + ")",
                        type: "merged_network",
                        feature: feature,
                        dlrId: feature.properties.dlr_id
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

                // Show/hide DLR matched layer
                if (activeLineTypes.includes("dlr-matched")) {{
                    dlrMatchedLayer.addTo(map);
                    applyVoltageFilter(dlrMatchedLayer);
                }} else {{
                    map.removeLayer(dlrMatchedLayer);
                }}

                // Show/hide DLR duplicate layer
                if (activeLineTypes.includes("dlr-duplicate")) {{
                    dlrDuplicateLayer.addTo(map);
                    applyVoltageFilter(dlrDuplicateLayer);
                }} else {{
                    map.removeLayer(dlrDuplicateLayer);
                }}

                // Show/hide DLR geometric match layer
                if (activeLineTypes.includes("dlr-geometric")) {{
                    dlrGeometricLayer.addTo(map);
                    applyVoltageFilter(dlrGeometricLayer);
                }} else {{
                    map.removeLayer(dlrGeometricLayer);
                }}

                // Show/hide DLR unmatched layer
                if (activeLineTypes.includes("dlr-unmatched")) {{
                    dlrUnmatchedLayer.addTo(map);
                    applyVoltageFilter(dlrUnmatchedLayer);
                }} else {{
                    map.removeLayer(dlrUnmatchedLayer);
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

                            // Highlight the feature and possibly related DLR
                            highlightFeature(result.id, result.dlrId);

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
    output_file = output_dir / 'dlr_network_matching_with_duplicates.html'
    with open(output_file, 'w') as f:
        f.write(html)

    return output_file




# Create an enhanced summary table with electrical parameters
def create_enhanced_summary_table(enhanced_results):
    # Convert results to DataFrame
    results_df = pd.DataFrame([r for r in enhanced_results if 'dlr_id' in r])

    # Create summary statistics
    total_dlr_lines = len(enhanced_results)
    matched_lines = sum(result.get('matched', False) for result in enhanced_results)
    unmatched_lines = total_dlr_lines - matched_lines

    # Count different types of matches
    regular_matches = sum(1 for result in enhanced_results if result.get('matched', False) and
                          not result.get('is_duplicate', False) and
                          not result.get('is_geometric_match', False))

    duplicate_matches = sum(1 for result in enhanced_results if result.get('is_duplicate', False))
    geometric_matches = sum(1 for result in enhanced_results if result.get('is_geometric_match', False))

    # Count specific match qualities
    match_quality_counts = {
        'Excellent': sum(1 for result in enhanced_results if result.get('match_quality') == 'Excellent'),
        'Good': sum(1 for result in enhanced_results if result.get('match_quality') == 'Good'),
        'Fair': sum(1 for result in enhanced_results if 'Fair' in str(result.get('match_quality', ''))),
        'Poor': sum(1 for result in enhanced_results if 'Poor' in str(result.get('match_quality', ''))),
        'Geometric Match': sum(
            1 for result in enhanced_results if 'Geometric Match' in str(result.get('match_quality', ''))),
        'Parallel Circuit': sum(
            1 for result in enhanced_results if 'Parallel Circuit' in str(result.get('match_quality', ''))),
        'No match - endpoints not found': sum(
            1 for result in enhanced_results if result.get('match_quality') == 'No match - endpoints not found'),
        'No match - start and end are the same': sum(
            1 for result in enhanced_results if result.get('match_quality') == 'No match - start and end are the same'),
        'No match - no network lines in path': sum(
            1 for result in enhanced_results if result.get('match_quality') == 'No match - no network lines in path'),
        'No path found': sum(
            1 for result in enhanced_results if 'No path found' in str(result.get('match_quality', '')))
    }

    # Create HTML for summary
    html_summary = f"""
    <html>
    <head>
        <title>DLR-Network Line Matching Summary with Electrical Parameters</title>
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
        <h1>DLR-Network Line Matching Results with Electrical Parameters</h1>

        <div class="summary">
            <h2>Summary</h2>
            <p>Total DLR Lines: {total_dlr_lines}</p>
            <p>Matched Lines: {matched_lines} ({matched_lines / total_dlr_lines * 100:.1f}%)</p>
            <ul>
                <li>Regular matches: {regular_matches} ({regular_matches / total_dlr_lines * 100:.1f}%)</li>
                <li>Geometric matches: {geometric_matches} ({geometric_matches / total_dlr_lines * 100:.1f}%)</li>
                <li>Parallel circuits: {duplicate_matches} ({duplicate_matches / total_dlr_lines * 100:.1f}%)</li>
            </ul>
            <p>Unmatched Lines: {unmatched_lines} ({unmatched_lines / total_dlr_lines * 100:.1f}%)</p>
            <p>Match Quality Details:</p>
            <ul>
                <li>Excellent Matches: {match_quality_counts['Excellent']} ({match_quality_counts['Excellent'] / total_dlr_lines * 100:.1f}%)</li>
                <li>Good Matches: {match_quality_counts['Good']} ({match_quality_counts['Good'] / total_dlr_lines * 100:.1f}%)</li>
                <li>Fair Matches: {match_quality_counts['Fair']} ({match_quality_counts['Fair'] / total_dlr_lines * 100:.1f}%)</li>
                <li>Poor Matches: {match_quality_counts['Poor']} ({match_quality_counts['Poor'] / total_dlr_lines * 100:.1f}%)</li>
                <li>Geometric Matches: {match_quality_counts['Geometric Match']} ({match_quality_counts['Geometric Match'] / total_dlr_lines * 100:.1f}%)</li>
                <li>Parallel Circuits: {match_quality_counts['Parallel Circuit']} ({match_quality_counts['Parallel Circuit'] / total_dlr_lines * 100:.1f}%)</li>
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
            <div class="filter-buttons">
                <button onclick="filterByMatchStatus('all')">All</button>
                <button onclick="filterByMatchStatus('matched')">Regular Matches</button>
                <button onclick="filterByMatchStatus('geometric')">Geometric Matches</button>
                <button onclick="filterByMatchStatus('duplicate')">Parallel Circuits</button>
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
                <th>DLR ID</th>
                <th>DLR Name</th>
                <th>Voltage (kV)</th>
                <th>Matched</th>
                <th>Network IDs</th>
                <th>DLR Length (km)</th>
                <th>Length Ratio</th>
                <th>Match Quality</th>
                <th>Electrical Parameters</th>
            </tr>
    """

    # Add rows for each result
    for i, result in enumerate(enhanced_results):
        network_ids = ", ".join(result.get('network_ids', [])) if result.get('matched', False) and result.get(
            'network_ids') else "-"
        length_ratio = f"{result.get('length_ratio', '-'):.2f}" if result.get('length_ratio') is not None else "-"
        dlr_length_km = f"{result.get('dlr_length_km', 0):.2f}" if 'dlr_length_km' in result else "-"

        # Determine row class and match type based on match status
        if result.get('matched', False):
            if result.get('is_duplicate', False):
                css_class = "duplicate"
                match_type = "duplicate"
            elif result.get('is_geometric_match', False):
                css_class = "geometric"
                match_type = "geometric"
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
                <td>{result.get('dlr_id', '-')}</td>
                <td>{result.get('dlr_name', '-')}</td>
                <td>{result.get('v_nom', '-')}</td>
                <td>{"Yes" if result.get('matched', False) else "No"}</td>
                <td>{network_ids}</td>
                <td>{dlr_length_km}</td>
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
        if result.get('matched', False) and 'matched_lines_data' in result:
            html_summary += f"""
            <tr class="parameter-row" data-result-id="{result_id}" style="display: none;">
                <td colspan="9" class="parameter-details">
                    <div id="details-{result_id}" class="details-section">
                        <h3>DLR Line Electrical Parameters</h3>
                        <p>Length: {result.get('dlr_length_km', 0):.2f} km</p>
                        <p>Resistance (R): {result.get('dlr_r', 0):.6f} ohm (Total)</p>
                        <p>Reactance (X): {result.get('dlr_x', 0):.6f} ohm (Total)</p>
                        <p>Susceptance (B): {result.get('dlr_b', 0):.8f} S (Total)</p>
                        <p>Resistance per km (R): {result.get('dlr_r_per_km', 0):.6f} ohm/km</p>
                        <p>Reactance per km (X): {result.get('dlr_x_per_km', 0):.6f} ohm/km</p>
                        <p>Susceptance per km (B): {result.get('dlr_b_per_km', 0):.8f} S/km</p>

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
    output_file = output_dir / 'dlr_network_matching_parameters.html'
    with open(output_file, 'w') as f:
        f.write(html_summary)

    return output_file

def visualize_results(dlr_gdf, network_gdf, matching_results):
    """Create a visualization of the matching results."""
    # Create GeoJSON data for lines
    dlr_features = []
    network_features = []

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

    # Identify duplicate DLR lines with identical geometries
    dlr_to_group, geometry_groups = identify_duplicate_dlr_lines(dlr_gdf)

    # Use a larger distance threshold
    distance_threshold_meters = 500  # 500 meters around substations
    print("Finding nearest points for DLR endpoints...")
    nearest_points_dict = find_nearest_points(
        dlr_gdf, network_gdf,
        max_alternatives=10,
        distance_threshold_meters=distance_threshold_meters
    )

    # Build network graph with NO automatic connections
    print("Building network graph (without extra connections)...")
    G = build_network_graph(network_gdf)
    print(f"Graph has {len(G.nodes)} nodes and {len(G.edges)} edges")

    # Find matching network lines with special handling for duplicates
    print("Finding matching network lines with special handling for duplicates...")
    matching_results = find_matching_network_lines_with_duplicates(
        dlr_gdf, network_gdf, nearest_points_dict, G,
        duplicate_groups=geometry_groups,
        max_reuse=5,
        min_length_ratio=0.7,  # Network must be at least 70% of DLR length
        max_length_ratio=3  # Network can be at most 1.8x DLR length
    )

    # ADD THE NEW CODE HERE - AFTER finding matches but BEFORE geometric matching
    # Double check for unmarked duplicates
    print("Double-checking duplicate handling...")
    duplicate_count_before = sum(1 for r in matching_results if r.get('is_duplicate', False))

    # Create a map of already matched DLR lines
    matched_dlrs = {r['dlr_id']: r for r in matching_results if r['matched']}

    # Check for any duplicates that weren't properly marked
    for geom_wkt, dlr_ids in geometry_groups.items():
        if len(dlr_ids) > 1:  # Only process duplicate groups
            # Find the first matched DLR in this group
            first_matched = None
            for dlr_id in dlr_ids:
                if dlr_id in matched_dlrs and not matched_dlrs[dlr_id].get('is_duplicate', False):
                    first_matched = matched_dlrs[dlr_id]
                    break

            # If we found a matched one, ensure all others in the group are marked as duplicates
            if first_matched:
                for dlr_id in dlr_ids:
                    if dlr_id != first_matched['dlr_id'] and dlr_id in matched_dlrs:
                        # Check if this is already properly marked as duplicate
                        if not matched_dlrs[dlr_id].get('is_duplicate', False):
                            print(f"Fixing duplicate marking for DLR {dlr_id}")
                            matched_dlrs[dlr_id]['is_duplicate'] = True
                            matched_dlrs[dlr_id]['duplicate_of'] = first_matched['dlr_id']
                            matched_dlrs[dlr_id]['match_quality'] = 'Parallel Circuit (duplicate geometry)'

    duplicate_count_after = sum(1 for r in matching_results if r.get('is_duplicate', False))
    print(f"Fixed {duplicate_count_after - duplicate_count_before} additional duplicate markings")
    # END OF NEW CODE

    # Add a second pass using direct geometric matching for remaining unmatched lines
    print("\nAttempting second pass with advanced geometric matching...")
    matching_results = match_remaining_lines_by_geometry(
        dlr_gdf, network_gdf, matching_results,
        buffer_distance=0.005,  # ~500m in degrees
        snap_tolerance=500,  # 500 meters for endpoint matching
        angle_tolerance=40,  # 30 degrees max angle difference
        min_dir_cos=0.866,  # cos(30°) ≈ 0.866
        min_length_ratio=0.5,  # Network must be at least 30% of DLR length
        max_length_ratio=1.5  # Network can be at most 170% of DLR length
    )

    # Continue with the rest of your code...
    # Allocate electrical parameters
    print("Allocating electrical parameters...")
    enhanced_results = allocate_electrical_parameters(dlr_gdf, network_gdf, matching_results)



    # Create enhanced visualization with duplicate handling
    print("Creating enhanced visualization with duplicate handling...")
    duplicate_map_file = visualize_results_with_duplicates(dlr_gdf, network_gdf, matching_results)

    # Create regular visualization and summary
    print("Creating regular visualization and summary...")
    map_file = visualize_results(dlr_gdf, network_gdf, matching_results)
    enhanced_summary_file = create_enhanced_summary_table(enhanced_results)

    print(f"Results saved to:")
    print(f"  - Regular Map: {map_file}")
    print(f"  - Map with Duplicate Handling: {duplicate_map_file}")
    print(f"  - Enhanced Summary with Parameters: {enhanced_summary_file}")

    # Find DLR lines that use the same network lines
    dlr_by_network_line = {}
    for result in matching_results:
        if result['matched'] and not result.get('is_duplicate', False) and 'network_ids' in result:
            for network_id in result['network_ids']:
                if network_id not in dlr_by_network_line:
                    dlr_by_network_line[network_id] = []
                dlr_by_network_line[network_id].append(result['dlr_id'])

    # Count duplicate DLR lines
    duplicate_count = sum(1 for result in matching_results if result.get('is_duplicate', False))

    # Print summary statistics
    total_dlr_lines = len(matching_results)
    matched_lines = sum(result['matched'] for result in matching_results)
    print(f"\nTotal DLR lines: {total_dlr_lines}")
    print(f"Matched lines: {matched_lines} ({matched_lines / total_dlr_lines * 100:.1f}%)")
    print(
        f"  - Regular matches: {matched_lines - duplicate_count} ({(matched_lines - duplicate_count) / total_dlr_lines * 100:.1f}%)")
    print(f"  - Parallel circuits: {duplicate_count} ({duplicate_count / total_dlr_lines * 100:.1f}%)")
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