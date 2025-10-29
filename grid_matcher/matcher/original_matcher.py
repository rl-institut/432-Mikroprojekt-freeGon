"""matcher matching algorithm with original high match rate."""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import math
from pathlib import Path
from collections import defaultdict, deque
import networkx as nx
from shapely.geometry import LineString, Point, MultiLineString
from shapely.ops import linemerge, unary_union, nearest_points

# Import internal modules to reuse some utility functions
from grid_matcher.io.exporters import (
    create_results_csv, generate_pypsa_with_eic, generate_jao_with_pypsa,
    generate_matching_statistics
)
from grid_matcher.visualization.maps import create_jao_pypsa_visualization
from grid_matcher.visualization.reports import create_enhanced_summary_table, export_unmatched_pypsa_details, random_match_quality_check



# ----------------- Utility Functions -----------------

def _safe_int(val):
    """Safely convert a value to int."""
    try:
        if val is None:
            return 0
        return int(float(val))
    except (ValueError, TypeError):
        return 0


def _num(val):
    """Convert a value to float or return None."""
    try:
        if val is None or pd.isna(val):
            return None
        return float(val)
    except Exception:
        return None


def _to_km(length_value):
    """Convert a length to kilometers."""
    v = _num(length_value)
    if v is None:
        return None
    return v / 1000.0 if abs(v) >= 1000.0 else v


def calculate_length_meters(geometry):
    """Calculate length in meters for a geometry."""
    if geometry is None:
        return 0

    # Get centroid latitude for conversion
    centroid_lat = geometry.centroid.y
    # Approximate meters per degree at this latitude
    meters_per_deg = 111111 * np.cos(np.radians(abs(centroid_lat)))
    # Convert length from degrees to meters
    return float(geometry.length) * meters_per_deg


def get_start_point(geometry):
    """Safely extract the start point from a geometry."""
    if geometry is None:
        return None

    if isinstance(geometry, LineString):
        if len(geometry.coords) > 0:
            return Point(geometry.coords[0])
    elif isinstance(geometry, MultiLineString):
        if len(geometry.geoms) > 0 and len(geometry.geoms[0].coords) > 0:
            return Point(geometry.geoms[0].coords[0])

    return None


def get_end_point(geometry):
    """Safely extract the end point from a geometry."""
    if geometry is None:
        return None

    if isinstance(geometry, LineString):
        if len(geometry.coords) > 0:
            return Point(geometry.coords[-1])
    elif isinstance(geometry, MultiLineString):
        if len(geometry.geoms) > 0 and len(geometry.geoms[-1].coords) > 0:
            return Point(geometry.geoms[-1].coords[-1])

    return None


def parse_linestring(wkt_str):
    """Return the exact geometry from WKT."""
    from shapely import wkt
    try:
        return wkt.loads(wkt_str)
    except Exception as exc:
        print(f"[parse_linestring] bad WKT → {exc}")
        return None


# ----------------- Core Matching Functions -----------------

def load_data(jao_path, pypsa_path):
    """Load JAO and PyPSA data from CSV files with enhanced preprocessing."""
    print(f"Loading JAO lines from {jao_path}")
    jao_df = pd.read_csv(jao_path)
    jao_geometry = jao_df['geometry'].apply(parse_linestring)
    jao_gdf = gpd.GeoDataFrame(jao_df, geometry=jao_geometry)

    print(f"Loading PyPSA lines from {pypsa_path}")
    pypsa_df = pd.read_csv(pypsa_path)
    pypsa_geometry = pypsa_df['geometry'].apply(parse_linestring)
    pypsa_gdf = gpd.GeoDataFrame(pypsa_df, geometry=pypsa_geometry)

    # Ensure IDs are strings
    jao_gdf['id'] = jao_gdf['id'].astype(str)
    pypsa_gdf['id'] = pypsa_gdf['id'].astype(str)

    # Add endpoints for matching
    jao_gdf['start_point'] = jao_gdf.geometry.apply(get_start_point)
    jao_gdf['end_point'] = jao_gdf.geometry.apply(get_end_point)

    pypsa_gdf['start_point'] = pypsa_gdf.geometry.apply(get_start_point)
    pypsa_gdf['end_point'] = pypsa_gdf.geometry.apply(get_end_point)

    # Calculate lengths in km
    jao_gdf['length_km'] = jao_gdf.geometry.apply(
        lambda g: calculate_length_meters(g) / 1000 if g is not None else 0
    )

    pypsa_gdf['length_km'] = pypsa_gdf.geometry.apply(
        lambda g: calculate_length_meters(g) / 1000 if g is not None else 0
    )

    # Ensure circuits column exists (default to 1)
    if 'circuits' not in pypsa_gdf.columns:
        pypsa_gdf['circuits'] = 1
    else:
        pypsa_gdf['circuits'] = pd.to_numeric(pypsa_gdf['circuits'], errors='coerce').fillna(1)

    pypsa_gdf['circuits'] = pypsa_gdf['circuits'].apply(lambda x: max(1, int(x)))

    # Standardize voltage columns
    if 'v_nom' not in jao_gdf.columns and 'voltage' in jao_gdf.columns:
        jao_gdf['v_nom'] = jao_gdf['voltage']

    if 'voltage' not in pypsa_gdf.columns and 'v_nom' in pypsa_gdf.columns:
        pypsa_gdf['voltage'] = pypsa_gdf['v_nom']

    # Ensure electrical parameters are available for matching
    for df in [jao_gdf, pypsa_gdf]:
        # Standardize parameter names
        if 'r_ohm_per_km' in df.columns and 'r_per_km' not in df.columns:
            df['r_per_km'] = df['r_ohm_per_km']
        if 'x_ohm_per_km' in df.columns and 'x_per_km' not in df.columns:
            df['x_per_km'] = df['x_ohm_per_km']
        if 'b_mho_per_km' in df.columns and 'b_per_km' not in df.columns:
            df['b_per_km'] = df['b_mho_per_km']

    # Identify duplicate geometries in JAO data (parallel circuits)
    jao_geometry_groups = defaultdict(list)
    for idx, row in jao_gdf.iterrows():
        if row.geometry is not None:
            wkt_str = row.geometry.wkt
            jao_geometry_groups[wkt_str].append(row['id'])

    # Mark parallel circuits
    jao_gdf['is_parallel_circuit'] = False
    jao_gdf['parallel_group'] = None

    for wkt_str, ids in jao_geometry_groups.items():
        if len(ids) > 1:
            group_id = '_'.join(sorted(ids))
            jao_gdf.loc[jao_gdf['id'].isin(ids), 'is_parallel_circuit'] = True
            jao_gdf.loc[jao_gdf['id'].isin(ids), 'parallel_group'] = group_id

    print(f"Loaded {len(jao_gdf)} JAO lines and {len(pypsa_gdf)} PyPSA lines")
    print(f"Found {sum(jao_gdf['is_parallel_circuit'])} JAO lines that are part of parallel circuits")

    return jao_gdf, pypsa_gdf


def build_network_graph(pypsa_gdf):
    """Build a network graph from PyPSA lines for path-based matching."""
    print("Building network graph from PyPSA lines...")

    G = nx.Graph()

    # Add edges for each PyPSA line
    for idx, row in pypsa_gdf.iterrows():
        if row.geometry is None:
            continue

        try:
            # Get endpoints
            start_point = get_start_point(row.geometry)
            end_point = get_end_point(row.geometry)

            if start_point is None or end_point is None:
                continue

            # Create node IDs
            start_id = f"node_{start_point.x:.6f}_{start_point.y:.6f}"
            end_id = f"node_{end_point.x:.6f}_{end_point.y:.6f}"

            # Add nodes with coordinates
            G.add_node(start_id, x=start_point.x, y=start_point.y)
            G.add_node(end_id, x=end_point.x, y=end_point.y)

            # Add edge with attributes
            length = row['length_km']
            weight = length if length > 0 else 0.001  # Avoid zero weights

            G.add_edge(
                start_id,
                end_id,
                id=row['id'],
                weight=weight,
                length=length,
                voltage=row.get('voltage', 0),
                circuits=int(row.get('circuits', 1))
            )
        except Exception as e:
            print(f"Error adding line {row['id']} to graph: {e}")

    print(f"Created graph with {len(G.nodes)} nodes and {len(G.edges)} edges")
    return G


def identify_duplicate_geometries(jao_gdf):
    """Identify JAO lines with identical geometries (parallel circuits)."""
    print("Identifying duplicate JAO geometries (parallel circuits)...")

    duplicate_groups = {}

    for idx, row in jao_gdf.iterrows():
        if row.geometry is None:
            continue

        wkt_str = row.geometry.wkt
        duplicate_groups.setdefault(wkt_str, []).append(row['id'])

    # Filter to groups with multiple lines
    parallel_groups = {wkt: ids for wkt, ids in duplicate_groups.items() if len(ids) > 1}

    print(f"Found {len(parallel_groups)} groups of parallel JAO circuits")

    # Create a mapping from JAO ID to its group
    jao_to_group = {}
    for wkt, ids in parallel_groups.items():
        for jao_id in ids:
            jao_to_group[jao_id] = wkt

    return parallel_groups, jao_to_group


def identify_parallel_pypsa_circuits(pypsa_gdf):
    """
    Identify groups of PyPSA lines that share very similar geometry
    and are likely to be parallel circuits.
    """
    print("Identifying parallel PyPSA circuits...")

    parallel_groups = {}
    processed = set()

    for idx1, line1 in pypsa_gdf.iterrows():
        if idx1 in processed:
            continue

        line_id1 = line1['id']
        geom1 = line1.geometry

        if geom1 is None:
            continue

        # Start a new group
        group = [line_id1]
        processed.add(idx1)

        # Find all lines with similar geometry
        for idx2, line2 in pypsa_gdf.iterrows():
            if idx2 == idx1 or idx2 in processed:
                continue

            line_id2 = line2['id']
            geom2 = line2.geometry

            if geom2 is None:
                continue

            # Check if lines have high similarity
            # For parallel circuits, use a very strict comparison
            try:
                # Check if lines follow the same route (with small tolerance)
                hausdorff_dist = geom1.hausdorff_distance(geom2)

                # If the maximum distance between lines is very small
                if hausdorff_dist < 0.0005:  # About 50m
                    # Also check if lengths are similar
                    if 0.95 <= (geom1.length / geom2.length) <= 1.05:
                        # Check voltage levels are the same
                        if _safe_int(line1.get('voltage', 0)) == _safe_int(line2.get('voltage', 0)):
                            group.append(line_id2)
                            processed.add(idx2)
            except Exception:
                continue

        if len(group) > 1:  # Only store actual parallel groups
            parallel_groups[line_id1] = group

    print(f"Found {len(parallel_groups)} groups of parallel PyPSA circuits")
    return parallel_groups


def find_nearest_endpoints(jao_gdf, pypsa_gdf, G, max_dist_km=5):
    """Find the nearest PyPSA endpoints for each JAO endpoint using buffers and direction."""
    print("Finding nearest endpoints with buffer-based approach...")

    # Convert distance to degrees (approximate)
    max_dist_deg = max_dist_km / 111

    nearest_points = {}

    # Process each JAO line
    for idx, jao_row in jao_gdf.iterrows():
        jao_id = jao_row['id']
        jao_start = jao_row['start_point']
        jao_end = jao_row['end_point']

        if jao_start is None or jao_end is None:
            continue

        # Create direction vector for JAO line
        jao_vector = None
        if jao_row.geometry and jao_row.geometry.geom_type in ('LineString', 'MultiLineString'):
            if jao_row.geometry.geom_type == 'LineString':
                coords = list(jao_row.geometry.coords)
                if len(coords) >= 2:
                    start = coords[0]
                    end = coords[-1]
                    jao_vector = np.array([end[0] - start[0], end[1] - start[1]])
                    jao_vector_len = np.linalg.norm(jao_vector)
                    if jao_vector_len > 0:
                        jao_vector = jao_vector / jao_vector_len
            else:  # MultiLineString
                start = list(jao_row.geometry.geoms[0].coords)[0]
                end = list(jao_row.geometry.geoms[-1].coords)[-1]
                jao_vector = np.array([end[0] - start[0], end[1] - start[1]])
                jao_vector_len = np.linalg.norm(jao_vector)
                if jao_vector_len > 0:
                    jao_vector = jao_vector / jao_vector_len

        # Create buffers around endpoints
        start_buffer = jao_start.buffer(max_dist_deg)
        end_buffer = jao_end.buffer(max_dist_deg)

        # Find nodes within buffer
        start_nodes = []
        end_nodes = []

        for node_id, data in G.nodes(data=True):
            if 'x' not in data or 'y' not in data:
                continue

            node_point = Point(data['x'], data['y'])

            # Check start point buffer
            if start_buffer.contains(node_point):
                # Calculate distance
                distance = jao_start.distance(node_point)
                start_nodes.append({
                    'node_id': node_id,
                    'distance': distance
                })

            # Check end point buffer
            if end_buffer.contains(node_point):
                # Calculate distance
                distance = jao_end.distance(node_point)
                end_nodes.append({
                    'node_id': node_id,
                    'distance': distance
                })

        # Sort by distance
        start_nodes.sort(key=lambda x: x['distance'])
        end_nodes.sort(key=lambda x: x['distance'])

        # Store results along with direction vector
        nearest_points[jao_id] = {
            'start_nodes': start_nodes[:10],  # Keep more candidates for path finding
            'end_nodes': end_nodes[:10],
            'direction_vector': jao_vector
        }

    print(f"Found endpoint nodes for {len(nearest_points)} JAO lines")
    return nearest_points


def enhanced_path_finding(G, start_node, end_node, jao_length, max_segments=3):
    """Find paths with multiple segments between start and end nodes."""
    # Try direct path first
    try:
        direct_path = nx.shortest_path(G, start_node, end_node, weight='weight')
        return [direct_path]
    except nx.NetworkXNoPath:
        pass

    # If no direct path, try paths with intermediate nodes
    candidate_paths = []

    # Get nodes within reasonable distance of start and end
    start_area = []
    end_area = []

    for node_id, data in G.nodes(data=True):
        if 'x' not in data or 'y' not in data:
            continue

        start_node_data = G.nodes[start_node]
        end_node_data = G.nodes[end_node]

        # Calculate distance in km (approximate)
        if 'x' in start_node_data and 'y' in start_node_data:
            dist_to_start = math.sqrt(
                (data['x'] - start_node_data['x']) ** 2 +
                (data['y'] - start_node_data['y']) ** 2
            ) * 111  # ~111km per degree

            if dist_to_start < jao_length * 0.3:  # Within 30% of total line length
                start_area.append(node_id)

        if 'x' in end_node_data and 'y' in end_node_data:
            dist_to_end = math.sqrt(
                (data['x'] - end_node_data['x']) ** 2 +
                (data['y'] - end_node_data['y']) ** 2
            ) * 111

            if dist_to_end < jao_length * 0.3:
                end_area.append(node_id)

    # For each potential intermediate node near the middle
    middle_nodes = set(start_area).intersection(set(end_area))

    # If too many middle nodes, take a sample
    if len(middle_nodes) > 20:
        middle_nodes = list(middle_nodes)[:20]

    # Try paths through intermediate nodes
    for mid_node in middle_nodes:
        if mid_node == start_node or mid_node == end_node:
            continue

        try:
            # Find path from start to middle
            path1 = nx.shortest_path(G, start_node, mid_node, weight='weight')

            # Find path from middle to end
            path2 = nx.shortest_path(G, mid_node, end_node, weight='weight')

            # Combine paths (remove duplicate middle node)
            combined_path = path1 + path2[1:]

            # Calculate path length
            path_length = 0
            for i in range(len(combined_path) - 1):
                u, v = combined_path[i], combined_path[i + 1]
                path_length += G[u][v].get('length', 0)

            # Only consider if length is reasonable
            if 0.7 <= (path_length / jao_length) <= 1.3:
                candidate_paths.append((combined_path, path_length))

        except nx.NetworkXNoPath:
            continue

    # Sort by length similarity to JAO line
    candidate_paths.sort(key=lambda x: abs(x[1] - jao_length))

    # Return the paths
    return [path for path, _ in candidate_paths[:5]]  # Return top 5 candidates


def calculate_electrical_similarity(jao_row, pypsa_row):
    """Calculate electrical parameter similarity between JAO and PyPSA lines."""
    similarity = 0.5  # Default neutral score

    # Extract per-km electrical parameters
    jao_r = _num(jao_row.get('r_per_km'))
    jao_x = _num(jao_row.get('x_per_km'))
    jao_b = _num(jao_row.get('b_per_km'))

    pypsa_r = _num(pypsa_row.get('r_per_km'))
    pypsa_x = _num(pypsa_row.get('x_per_km'))
    pypsa_b = _num(pypsa_row.get('b_per_km'))

    # If no parameters available, return default
    if jao_r is None and jao_x is None and jao_b is None:
        return similarity

    if pypsa_r is None and pypsa_x is None and pypsa_b is None:
        return similarity

    # Calculate relative errors for each parameter
    errors = []

    if jao_r is not None and pypsa_r is not None and jao_r > 0 and pypsa_r > 0:
        r_err = abs(jao_r - pypsa_r) / jao_r
        errors.append(min(r_err, 1.0))  # Cap at 100% error

    if jao_x is not None and pypsa_x is not None and jao_x > 0 and pypsa_x > 0:
        x_err = abs(jao_x - pypsa_x) / jao_x
        errors.append(min(x_err, 1.0))

    if jao_b is not None and pypsa_b is not None and jao_b > 0 and pypsa_b > 0:
        b_err = abs(jao_b - pypsa_b) / jao_b
        errors.append(min(b_err, 1.0))

    # If no comparable parameters, return default
    if not errors:
        return similarity

    # Calculate average similarity (1 - average error)
    avg_err = sum(errors) / len(errors)
    return max(0.0, 1.0 - avg_err)


def match_multi_segment_corridors(jao_gdf, pypsa_gdf):
    """
    Match JAO lines to multi-segment corridors made up of connected PyPSA segments.
    Respects circuit constraints - only match as many JAO lines as there are circuits.
    """
    from collections import defaultdict
    import re
    import numpy as np
    from shapely.geometry import LineString, Point
    from shapely.ops import linemerge, unary_union

    print("\n===== MATCHING MULTI-SEGMENT CORRIDORS =====")

    # Define known problematic corridors with exact mappings and circuit sharing
    known_corridors = [
        {
            'name': 'Gundelfingen-Voehringen',
            'shared_circuit_groups': [
                # Define groups of JAO lines that should share circuit capacity
                {
                    'jao_ids': ['244', '245'],  # These JAO lines compete for the same circuits
                    'circuit_capacity': 1,  # Total capacity they must share
                    'mappings': [
                        # Each JAO has its exact mapping, but they compete for capacity
                        {
                            'jao_id': '244',
                            'pypsa_segments': ['relation/1641463-380-a', 'relation/1641463-380-b',
                                               'relation/1641463-380-c', 'relation/1641463-380-e']
                        },
                        {
                            'jao_id': '245',
                            'pypsa_segments': ['relation/1641474-380-a', 'relation/1641474-380-b',
                                               'relation/1641474-380-c', 'relation/1641474-380-e']
                        }
                    ]
                }
            ]
        }
    ]

    results = []
    forced_unmatched_jao_ids = []  # Keep track of JAO IDs that should never be matched

    # Process each known corridor
    for corridor in known_corridors:
        print(f"Processing known corridor: {corridor['name']}")

        # Process each shared circuit group
        for group in corridor.get('shared_circuit_groups', []):
            jao_ids = group.get('jao_ids', [])
            circuit_capacity = group.get('circuit_capacity', 1)
            mappings = group.get('mappings', [])

            print(f"  Processing shared circuit group with {len(jao_ids)} JAO lines, capacity: {circuit_capacity}")

            # Sort JAO IDs for deterministic matching
            sorted_jao_ids = sorted(jao_ids)

            # Only match up to the circuit capacity
            matched_count = 0

            for jao_id in sorted_jao_ids:
                # Find the mapping for this JAO ID
                mapping = next((m for m in mappings if m.get('jao_id') == jao_id), None)
                if not mapping:
                    print(f"  Warning: No mapping found for JAO {jao_id}")
                    continue

                pypsa_segments = mapping.get('pypsa_segments', [])

                # Check if JAO ID exists
                jao_rows = jao_gdf[jao_gdf['id'].astype(str) == jao_id]
                if jao_rows.empty:
                    print(f"  Warning: JAO ID {jao_id} not found, skipping")
                    continue

                jao_row = jao_rows.iloc[0]

                # Verify all segments exist
                all_segments_found = True
                found_segments = []

                for segment_id in pypsa_segments:
                    segments_match = pypsa_gdf[(pypsa_gdf['id'].astype(str) == segment_id) |
                                               (pypsa_gdf.get('line_id', '').astype(str) == segment_id)]

                    if segments_match.empty:
                        print(f"  Warning: PyPSA segment {segment_id} not found")
                        all_segments_found = False
                    else:
                        found_segments.append(segment_id)

                # Only proceed if all segments were found
                if not all_segments_found:
                    print(f"  Skipping JAO {jao_id} - not all segments found")
                    continue

                # Check if we're within the circuit capacity
                if matched_count < circuit_capacity:
                    # We can match this JAO line
                    matched_count += 1

                    # Calculate path length and stats
                    jao_length = float(jao_row.get('length_km', jao_row.get('length', 0)) or 0)
                    path_length = 0

                    for segment_id in found_segments:
                        segment_row = pypsa_gdf[(pypsa_gdf['id'].astype(str) == segment_id) |
                                                (pypsa_gdf.get('line_id', '').astype(str) == segment_id)].iloc[0]

                        length = float(segment_row.get('length', 0) or 0)
                        # Convert to km if needed
                        if length > 1000:  # Likely in meters
                            length /= 1000.0
                        path_length += length

                    # Calculate length ratio
                    length_ratio = path_length / jao_length if jao_length > 0 else 1.0

                    # Create match result
                    results.append({
                        'matched': True,
                        'jao_id': jao_id,
                        'pypsa_ids': found_segments,
                        'path_length': path_length,
                        'length_ratio': length_ratio,
                        'match_quality': f"Exact Multi-Segment Circuit ({len(found_segments)} segments)",
                        'is_geometric_match': True,
                        'is_parallel_circuit': True,
                        'is_duplicate': False,
                        'locked_by_corridor': True  # Lock this match
                    })

                    print(f"  Successfully matched JAO {jao_id} to {len(found_segments)} segments")
                else:
                    # We've exceeded the circuit capacity, so this JAO line remains unmatched
                    # Mark unmatched corridor JAO lines as locked too, to prevent other
                    # functions from trying to match them later
                    results.append({
                        'matched': False,
                        'jao_id': jao_id,
                        'pypsa_ids': [],
                        'match_quality': f"Unmatched - No available circuit (capacity of {circuit_capacity} already allocated)",
                        'is_geometric_match': False,
                        'is_parallel_circuit': False,
                        'is_duplicate': False,
                        'locked_by_corridor': True,
                        'forced_unmatched': True
                    })

                    # Add to list of JAO IDs that should never be matched
                    forced_unmatched_jao_ids.append(jao_id)

                    print(f"  JAO {jao_id} could not be matched - all {circuit_capacity} circuit(s) already allocated")

    matched_count = sum(1 for r in results if r.get('matched', False))
    print(f"Multi-segment corridor matching complete: matched {matched_count} JAO lines")

    # Add explicit logging for debugging
    locked_count = sum(1 for r in results if r.get('locked_by_corridor', False))
    forced_unmatched_count = len(forced_unmatched_jao_ids)
    print(f"CORRIDOR DEBUG: Setting locked_by_corridor=True for {locked_count} matches")
    print(f"CORRIDOR DEBUG: Forced {forced_unmatched_count} JAO IDs to remain unmatched: {forced_unmatched_jao_ids}")

    return results, forced_unmatched_jao_ids


def path_based_line_matching(jao_gdf, pypsa_gdf, G, nearest_points, parallel_groups, jao_to_group,
                             existing_matches=None):
    """Match JAO lines to PyPSA lines using enhanced path-based matching."""
    print("Performing enhanced path-based line matching...")

    results = []

    # Check for already locked matches
    locked_jao_ids = set()
    if existing_matches:
        for match in existing_matches:
            if match.get('locked_by_corridor', False) and match.get('matched', False):
                locked_jao_ids.add(str(match.get('jao_id', '')))

        if locked_jao_ids:
            print(f"Skipping {len(locked_jao_ids)} JAO lines already matched with locked_by_corridor")

    # Track PyPSA line usage for circuit constraints
    pypsa_usage = {}
    group_used_pypsa = {}  # Track which PyPSA lines are used by each parallel group

    # Process each JAO line in sorted order for determinism
    for _, jao_row in sorted(jao_gdf.iterrows(), key=lambda it: str(it[1]['id'])):
        jao_id = jao_row['id']

        # Skip if this JAO ID is in a locked corridor match
        if str(jao_id) in locked_jao_ids:
            print(f"Skipping JAO {jao_id} - already matched by corridor")
            continue

        jao_voltage = _safe_int(jao_row.get('v_nom', 0))
        jao_length = jao_row['length_km']

        # Skip if no nearest points found
        if jao_id not in nearest_points:
            results.append({
                'jao_id': jao_id,
                'jao_name': str(jao_row.get('NE_name', '')),
                'jao_voltage': jao_voltage,
                'matched': False,
                'match_quality': 'No Endpoint Matches'
            })
            continue

        # Get parallel circuit group if applicable
        group_key = jao_to_group.get(jao_id)
        group_used_pypsa.setdefault(group_key, set())

        # Get nearest endpoint nodes
        np_info = nearest_points[jao_id]
        start_nodes = np_info['start_nodes']
        end_nodes = np_info['end_nodes']

        if not start_nodes or not end_nodes:
            results.append({
                'jao_id': jao_id,
                'jao_name': str(jao_row.get('NE_name', '')),
                'jao_voltage': jao_voltage,
                'matched': False,
                'match_quality': 'Insufficient Endpoint Matches'
            })
            continue

        # Try to find paths between start and end nodes
        candidates = []

        # Try up to 5 start nodes and 5 end nodes
        for start_info in start_nodes[:5]:
            start_node = start_info['node_id']

            for end_info in end_nodes[:5]:
                end_node = end_info['node_id']

                if start_node == end_node:
                    continue

                # Use enhanced path finding for multiple potential paths
                potential_paths = enhanced_path_finding(G, start_node, end_node, jao_length, max_segments=3)

                for path in potential_paths:
                    # Extract PyPSA line IDs from the path
                    pypsa_ids = []
                    path_length = 0

                    for i in range(len(path) - 1):
                        u, v = path[i], path[i + 1]
                        edge_data = G[u][v]

                        # Skip edges with no ID (should not happen)
                        if 'id' not in edge_data:
                            continue

                        pypsa_id = edge_data['id']

                        # Check circuit constraints
                        if pypsa_usage.get(pypsa_id, 0) >= G[u][v].get('circuits', 1):
                            # This PyPSA line is already at capacity
                            continue

                        # Skip if already used by another line in the same parallel group
                        if group_key and pypsa_id in group_used_pypsa[group_key]:
                            continue

                        # Check voltage compatibility
                        edge_voltage = edge_data.get('voltage', 0)
                        if not ((jao_voltage == 220 and edge_voltage == 220) or
                                (jao_voltage in [380, 400] and edge_voltage in [380, 400])):
                            # Voltage mismatch
                            continue

                        pypsa_ids.append(pypsa_id)
                        path_length += edge_data.get('length', 0)

                    # Skip if no valid PyPSA lines in the path
                    if not pypsa_ids:
                        continue

                    # Calculate length ratio
                    length_ratio = path_length / jao_length if jao_length > 0 else 0

                    # Skip if length ratio is unreasonable
                    if not (0.4 <= length_ratio <= 2.5):
                        continue

                    # Calculate electrical similarity
                    electrical_score = 0.5  # Default

                    # If single PyPSA line, directly compare electrical parameters
                    if len(pypsa_ids) == 1:
                        pypsa_rows = pypsa_gdf[pypsa_gdf['id'] == pypsa_ids[0]]
                        if not pypsa_rows.empty:
                            electrical_score = calculate_electrical_similarity(jao_row, pypsa_rows.iloc[0])
                    else:
                        # For multi-segment paths, calculate weighted average of electrical parameters
                        total_r = 0
                        total_x = 0
                        total_length = 0

                        for pypsa_id in pypsa_ids:
                            pypsa_rows = pypsa_gdf[pypsa_gdf['id'] == pypsa_id]
                            if not pypsa_rows.empty:
                                segment_r = _num(pypsa_rows.iloc[0].get('r_per_km'))
                                segment_x = _num(pypsa_rows.iloc[0].get('x_per_km'))
                                segment_length = pypsa_rows.iloc[0].get('length_km', 0)

                                if segment_r is not None and segment_length > 0:
                                    total_r += segment_r * segment_length
                                if segment_x is not None and segment_length > 0:
                                    total_x += segment_x * segment_length
                                if segment_length > 0:
                                    total_length += segment_length

                        if total_length > 0:
                            avg_r = total_r / total_length
                            avg_x = total_x / total_length

                            # Compare with JAO parameters
                            jao_r = _num(jao_row.get('r_per_km'))
                            jao_x = _num(jao_row.get('x_per_km'))

                            r_score = 0.5
                            x_score = 0.5

                            if jao_r is not None and avg_r > 0:
                                r_error = abs(jao_r - avg_r) / jao_r
                                r_score = max(0, 1 - min(r_error, 1))

                            if jao_x is not None and avg_x > 0:
                                x_error = abs(jao_x - avg_x) / jao_x
                                x_score = max(0, 1 - min(x_error, 1))

                            electrical_score = (r_score + x_score) / 2

                    # Calculate geometric similarity using buffer overlap
                    jao_buffer = jao_row.geometry.buffer(0.01)  # ~1km buffer
                    path_geoms = []

                    for pypsa_id in pypsa_ids:
                        pypsa_rows = pypsa_gdf[pypsa_gdf['id'] == pypsa_id]
                        if not pypsa_rows.empty and pypsa_rows.iloc[0].geometry is not None:
                            path_geoms.append(pypsa_rows.iloc[0].geometry)

                    geometric_score = 0.5  # Default

                    if path_geoms:
                        try:
                            # Try to merge the geometries
                            merged_geom = linemerge(path_geoms)
                            if merged_geom is None or merged_geom.is_empty:
                                merged_geom = path_geoms[0]  # Fallback to first geometry

                            # Calculate overlap
                            merged_buffer = merged_geom.buffer(0.01)
                            intersection = jao_buffer.intersection(merged_buffer)

                            if hasattr(intersection, 'area') and jao_buffer.area > 0:
                                geometric_score = min(1.0, intersection.area / jao_buffer.area)
                        except Exception:
                            pass

                    # Calculate direction similarity
                    direction_score = 0.5  # Default neutral value
                    if 'direction_vector' in np_info and np_info['direction_vector'] is not None:
                        # Extract path endpoints
                        path_start = G.nodes[path[0]]
                        path_end = G.nodes[path[-1]]

                        # Calculate path direction vector
                        path_vector = np.array([path_end['x'] - path_start['x'], path_end['y'] - path_start['y']])
                        path_vector_len = np.linalg.norm(path_vector)

                        if path_vector_len > 0:
                            path_vector = path_vector / path_vector_len

                            # Calculate direction similarity (dot product)
                            direction_score = abs(np.dot(np_info['direction_vector'], path_vector))

                            # Skip paths with very different direction (e.g., < 0.7 = > 45 degree difference)
                            if direction_score < 0.7:
                                continue

                    # Combined score
                    length_score = 1.0 - min(1.0, abs(length_ratio - 1.0))

                    # For multi-segment paths, give higher weight to geometry and length
                    if len(pypsa_ids) > 1:
                        combined_score = (
                                0.25 * electrical_score +  # Electrical parameters
                                0.35 * geometric_score +  # More weight on geometry for multi-segments
                                0.25 * length_score +  # More weight on length for multi-segments
                                0.15 * direction_score  # Direction similarity
                        )
                        # Bonus for multi-segment paths that closely match JAO length
                        if 0.95 <= length_ratio <= 1.05:
                            combined_score += 0.05
                    else:
                        combined_score = (
                                0.35 * electrical_score +  # Electrical parameters
                                0.25 * geometric_score +  # Geometric similarity
                                0.20 * length_score +  # Length ratio
                                0.20 * direction_score  # Direction similarity
                        )

                    candidates.append({
                        'pypsa_ids': pypsa_ids,
                        'path_length': path_length,
                        'length_ratio': length_ratio,
                        'electrical_score': electrical_score,
                        'geometric_score': geometric_score,
                        'length_score': length_score,
                        'direction_score': direction_score,
                        'score': combined_score,
                        'segments': len(pypsa_ids)  # Track number of segments
                    })

        # If no candidates found, mark as unmatched
        if not candidates:
            results.append({
                'jao_id': jao_id,
                'jao_name': str(jao_row.get('NE_name', '')),
                'jao_voltage': jao_voltage,
                'matched': False,
                'match_quality': 'No Valid Paths Found'
            })
            continue

        # Sort candidates by score
        candidates.sort(key=lambda x: x['score'], reverse=True)
        best_candidate = candidates[0]

        # Determine match quality
        score = best_candidate['score']
        segments = best_candidate['segments']

        if segments > 1:
            quality_prefix = f"{segments}-Segment "
        else:
            quality_prefix = ""

        if score >= 0.8:
            quality = f"{quality_prefix}Excellent Match"
        elif score >= 0.6:
            quality = f"{quality_prefix}Good Match"
        elif score >= 0.4:
            quality = f"{quality_prefix}Fair Match"
        else:
            quality = f"{quality_prefix}Poor Match"

        # Add parallel circuit indicator if applicable
        if jao_row.get('is_parallel_circuit', False):
            quality = f"Parallel Circuit - {quality}"

        # Create result
        result = {
            'jao_id': jao_id,
            'jao_name': str(jao_row.get('NE_name', '')),
            'jao_voltage': jao_voltage,
            'matched': True,
            'pypsa_ids': best_candidate['pypsa_ids'],
            'path_length': best_candidate['path_length'],
            'jao_length': jao_length,
            'length_ratio': best_candidate['length_ratio'],
            'match_score': score,
            'electrical_score': best_candidate['electrical_score'],
            'geometric_score': best_candidate['geometric_score'],
            'direction_score': best_candidate['direction_score'],
            'match_quality': quality,
            'is_parallel_circuit': jao_row.get('is_parallel_circuit', False),
            'parallel_group': jao_row.get('parallel_group'),
            'segments': segments  # Save number of segments
        }

        # Update PyPSA line usage
        for pypsa_id in best_candidate['pypsa_ids']:
            pypsa_usage[pypsa_id] = pypsa_usage.get(pypsa_id, 0) + 1

            # Mark as used by this parallel group
            if group_key:
                group_used_pypsa[group_key].add(pypsa_id)

        results.append(result)

    # Enforce circuit constraints
    enforce_circuit_constraints(results, pypsa_gdf, pypsa_usage)

    # Process parallel circuits to ensure they match different lines
    process_parallel_circuits(results, jao_gdf, pypsa_gdf, jao_to_group)

    return results


def enforce_circuit_constraints(results, pypsa_gdf, pypsa_usage):
    """Enforce circuit constraints on the matching results."""
    print("Enforcing circuit constraints...")

    # Find overused PyPSA lines
    overused_lines = {}
    for pypsa_id, usage in pypsa_usage.items():
        pypsa_rows = pypsa_gdf[pypsa_gdf['id'] == pypsa_id]
        if len(pypsa_rows) > 0:
            circuits = int(pypsa_rows.iloc[0].get('circuits', 1))
            if usage > circuits:
                overused_lines[pypsa_id] = {'usage': usage, 'circuits': circuits}

    if not overused_lines:
        print("No circuit constraints violated")
        return

    print(f"Found {len(overused_lines)} overused PyPSA lines")

    # For each overused line, keep only the best matches
    for pypsa_id, info in overused_lines.items():
        circuits = info['circuits']

        # Find all matches using this PyPSA line
        using_matches = []
        for i, result in enumerate(results):
            if result.get('matched', False) and pypsa_id in result.get('pypsa_ids', []):
                using_matches.append({
                    'index': i,
                    'jao_id': result['jao_id'],
                    'score': result.get('match_score', 0),
                    'is_parallel': result.get('is_parallel_circuit', False)
                })

        # Sort by score, but prioritize parallel circuits
        using_matches.sort(key=lambda x: (x['is_parallel'], x['score']), reverse=True)

        # Keep only the top N matches where N is the circuit count
        keep_matches = using_matches[:circuits]
        remove_matches = using_matches[circuits:]

        print(f"  PyPSA line {pypsa_id}: keeping {len(keep_matches)} of {len(using_matches)} matches")

        # For each match to remove, either remove just this PyPSA line or mark as unmatched
        for match_info in remove_matches:
            result = results[match_info['index']]

            # If this is the only PyPSA line in the match, mark as unmatched
            if len(result['pypsa_ids']) == 1:
                result['matched'] = False
                result['match_quality'] = 'Unmatched (Circuit Constraint)'
                result.pop('pypsa_ids', None)
                print(f"    Unmatched JAO {result['jao_id']} due to circuit constraints")
            else:
                # Remove this PyPSA line from the match
                result['pypsa_ids'].remove(pypsa_id)

                # Recalculate path length
                path_length = 0
                for pid in result['pypsa_ids']:
                    pypsa_rows = pypsa_gdf[pypsa_gdf['id'] == pid]
                    if not pypsa_rows.empty:
                        path_length += pypsa_rows.iloc[0]['length_km']

                result['path_length'] = path_length
                result['length_ratio'] = path_length / result['jao_length'] if result['jao_length'] > 0 else 0
                result['match_quality'] = f"Modified (Circuit Constraint) - {result['match_quality']}"
                print(f"    Modified match for JAO {result['jao_id']}, removed {pypsa_id}")


def process_parallel_circuits(results, jao_gdf, pypsa_gdf, jao_to_group):
    """
    Enhanced function to handle parallel circuits with improved path consistency.
    This function ensures that JAO lines with identical geometries are properly matched
    to available PyPSA lines, respecting existing matches and circuit constraints.
    """
    print("\n=== PROCESSING PARALLEL CIRCUITS WITH IMPROVED PATH CONSISTENCY ===")

    # 1. Collect all JAO parallel groups and their matches
    jao_parallel_groups = {}
    for result in results:
        group_key = jao_to_group.get(result.get('jao_id'))
        if group_key:
            jao_parallel_groups.setdefault(group_key, []).append(result)

    if not jao_parallel_groups:
        print("No JAO parallel circuit groups found")
        return

    print(f"Found {len(jao_parallel_groups)} JAO parallel circuit groups")

    # 2. Process each parallel group
    for group_key, group_results in jao_parallel_groups.items():
        jao_ids = [r['jao_id'] for r in group_results]
        print(f"Processing JAO parallel group: {', '.join(jao_ids)}")

        # Skip if none are matched
        matched_results = [r for r in group_results if r.get('matched', False)]
        if not matched_results:
            print("  No matched lines in this group")
            continue

        # Check if all lines are already matched to different PyPSA lines
        if len(matched_results) == len(group_results):
            # Get all PyPSA IDs used by this group
            used_pypsa_ids = set()
            for r in matched_results:
                pypsa_ids = r.get('pypsa_ids', [])
                if isinstance(pypsa_ids, str):
                    pypsa_ids = [pid.strip() for pid in pypsa_ids.split(';') if pid.strip()]
                used_pypsa_ids.update(pypsa_ids)

            # If each JAO has a match, and we have at least as many PyPSA IDs as JAO lines,
            # then we likely have good one-to-one matching already
            if len(used_pypsa_ids) >= len(jao_ids):
                print("  All lines already have good matches - preserving original mapping")
                continue

        # 3. Find the best match in the group (prioritize multi-segment with best coverage)
        best_match = None
        best_score = -1

        for result in matched_results:
            # Calculate a score based on number of segments and coverage
            segments = result.get('pypsa_ids', [])
            if isinstance(segments, str):
                segments = [s.strip() for s in segments.split(';') if s.strip()]

            num_segments = len(segments)
            coverage = result.get('length_ratio', 0)

            # Score calculation: prioritize multi-segment paths with good coverage
            score = num_segments * 10  # Each segment adds 10 points

            if 0.9 <= coverage <= 1.1:
                score += 20  # Excellent coverage
            elif 0.8 <= coverage <= 1.2:
                score += 15  # Good coverage
            elif 0.7 <= coverage <= 1.3:
                score += 10  # Fair coverage
            elif 0.5 <= coverage <= 1.5:
                score += 5  # Poor but acceptable coverage

            # Additional score for match quality
            quality = result.get('match_quality', '')
            if 'Excellent' in quality:
                score += 15
            elif 'Good' in quality:
                score += 10
            elif 'Fair' in quality:
                score += 5

            if score > best_score:
                best_score = score
                best_match = result

        if not best_match:
            print("  No suitable match found in the group")
            continue

        # 4. Get the best match details
        best_pypsa_ids = best_match.get('pypsa_ids', [])
        if isinstance(best_pypsa_ids, str):
            best_pypsa_ids = [pid.strip() for pid in best_pypsa_ids.split(';') if pid.strip()]

        best_match_quality = best_match.get('match_quality', 'Matched')
        best_path_length = best_match.get('path_length', 0)
        best_length_ratio = best_match.get('length_ratio', 0)

        print(f"  Best match found: {best_match['jao_id']} → {best_pypsa_ids}")

        # 5. Check which JAO lines are currently unmatched
        unmatched_results = [r for r in group_results if not r.get('matched', False)]

        if unmatched_results:
            print(f"  Applying to {len(unmatched_results)} unmatched lines in the group")
        else:
            print("  No unmatched lines to update in this group")
            continue

        # 6. Apply ONLY to unmatched lines in the group
        for result in unmatched_results:
            result['matched'] = True
            result['pypsa_ids'] = best_pypsa_ids.copy()
            result['path_length'] = best_path_length
            result['length_ratio'] = best_length_ratio
            result['match_quality'] = f"Parallel Circuit - {best_match_quality}"
            print(f"    Updated {result['jao_id']}: {result['match_quality']}")

    # 7. Validate circuit constraints for the entire result set
    # Mapping of PyPSA ID to list of JAO IDs using it
    pypsa_usage = {}
    for result in results:
        if result.get('matched', False) and 'pypsa_ids' in result:
            pypsa_ids = result['pypsa_ids']
            if isinstance(pypsa_ids, str):
                pypsa_ids = [pid.strip() for pid in pypsa_ids.split(';') if pid.strip()]

            for pypsa_id in pypsa_ids:
                pypsa_usage.setdefault(pypsa_id, []).append(result['jao_id'])

    # Check for oversubscribed PyPSA lines
    for pypsa_id, jao_ids in pypsa_usage.items():
        # Find the PyPSA row to get circuit count
        pypsa_rows = pypsa_gdf[pypsa_gdf['id'].astype(str) == str(pypsa_id)]
        if pypsa_rows.empty:
            # Try with 'line_id' if 'id' doesn't match
            pypsa_rows = pypsa_gdf[pypsa_gdf['line_id'].astype(str) == str(pypsa_id)]

        if not pypsa_rows.empty:
            circuits = int(pypsa_rows.iloc[0].get('circuits', 1))
            if len(jao_ids) > circuits:
                print(
                    f"  Warning: PyPSA line {pypsa_id} is used by {len(jao_ids)} JAO lines ({', '.join(jao_ids)}) but has only {circuits} circuits")
                # This is just a warning - we could implement more sophisticated resolution logic here

    return results


def find_bus_based_paths(jao_gdf, pypsa_gdf, G):
    """Find complete paths between buses specified in JAO lines."""
    print("Finding bus-based paths...")

    results = []

    for _, jao_row in jao_gdf.iterrows():
        jao_id = jao_row['id']
        jao_bus0 = str(jao_row.get('bus0', ''))
        jao_bus1 = str(jao_row.get('bus1', ''))
        jao_length = jao_row.get('length', 0)
        jao_voltage = _safe_int(jao_row.get('v_nom', 0))

        # Skip if buses aren't specified
        if not jao_bus0 or not jao_bus1:
            continue

        print(f"Finding path for JAO {jao_id} from {jao_bus0} to {jao_bus1}")

        # Map JAO buses to PyPSA nodes
        start_nodes = []
        end_nodes = []

        # Look for nodes containing the bus IDs
        for node_id in G.nodes():
            if jao_bus0 in node_id:
                start_nodes.append(node_id)
            if jao_bus1 in node_id:
                end_nodes.append(node_id)

        if not start_nodes or not end_nodes:
            print(f"  Could not find matching nodes for buses {jao_bus0}, {jao_bus1}")
            continue

        print(f"  Found {len(start_nodes)} potential start nodes and {len(end_nodes)} end nodes")

        # Find paths between all combinations
        best_path = None
        best_score = 0

        for start_node in start_nodes:
            for end_node in end_nodes:
                try:
                    # Find shortest path
                    path = nx.shortest_path(G, start_node, end_node, weight='weight')

                    # Extract PyPSA IDs and calculate path properties
                    pypsa_ids = []
                    path_length = 0
                    path_voltage_matches = True

                    for i in range(len(path) - 1):
                        u, v = path[i], path[i + 1]
                        if 'id' in G[u][v]:
                            pypsa_ids.append(G[u][v]['id'])
                            path_length += G[u][v].get('length', 0)

                            # Check voltage compatibility
                            edge_voltage = _safe_int(G[u][v].get('voltage', 0))
                            if not ((jao_voltage == edge_voltage) or
                                    (jao_voltage in [380, 400] and edge_voltage in [380, 400])):
                                path_voltage_matches = False

                    # Skip if voltage doesn't match
                    if not path_voltage_matches:
                        continue

                    # Calculate length similarity
                    length_ratio = path_length / jao_length if jao_length > 0 else 0
                    length_score = 1.0 - min(1.0, abs(length_ratio - 1.0))

                    # Calculate geometric similarity
                    path_geoms = []
                    for pypsa_id in pypsa_ids:
                        pypsa_rows = pypsa_gdf[pypsa_gdf['id'] == pypsa_id]
                        if not pypsa_rows.empty and pypsa_rows.iloc[0].geometry is not None:
                            path_geoms.append(pypsa_rows.iloc[0].geometry)

                    geometric_score = 0.5  # Default
                    if path_geoms and jao_row.geometry is not None:
                        try:
                            # Try to merge the geometries
                            merged_geom = linemerge(path_geoms)
                            if merged_geom is None or merged_geom.is_empty:
                                merged_geom = path_geoms[0]

                            # Calculate Hausdorff distance
                            hausdorff_dist = jao_row.geometry.hausdorff_distance(merged_geom)
                            hausdorff_score = max(0, 1 - (hausdorff_dist * 111000 / 5000))  # ~5km tolerance
                            geometric_score = hausdorff_score
                        except Exception as e:
                            print(f"  Error calculating geometric similarity: {e}")

                    # Combined score
                    combined_score = 0.6 * length_score + 0.4 * geometric_score

                    # Track best path
                    if combined_score > best_score:
                        best_score = combined_score
                        best_path = {
                            'path': path,
                            'pypsa_ids': pypsa_ids,
                            'path_length': path_length,
                            'length_ratio': length_ratio,
                            'geometric_score': geometric_score,
                            'score': combined_score
                        }

                except nx.NetworkXNoPath:
                    continue

        # If found a good path
        if best_path and best_path['score'] > 0.6:  # Reasonable threshold
            print(f"  Found path with score {best_path['score']:.2f}")
            print(f"  Path: {' -> '.join(best_path['pypsa_ids'])}")
            print(f"  Length ratio: {best_path['length_ratio']:.2f}")

            results.append({
                'jao_id': jao_id,
                'matched': True,
                'pypsa_ids': best_path['pypsa_ids'],
                'path_length': best_path['path_length'],
                'jao_length': jao_length,
                'length_ratio': best_path['length_ratio'],
                'match_score': best_path['score'],
                'match_quality': "Bus-Based Path Match"
            })
        else:
            print(f"  No suitable path found")

    return results


def find_visually_matching_lines(jao_gdf, pypsa_gdf, matches, batch_size=500, max_candidates=50,
                                 time_limit_sec=300, buffer_distance=0.005, min_score=0.65):
    """
    Find lines that visually match but weren't detected by other matching methods.

    Uses spatial indexing, batching and timeouts to efficiently process large datasets.

    Parameters
    ----------
    jao_gdf : GeoDataFrame
        GeoDataFrame with JAO lines
    pypsa_gdf : GeoDataFrame
        GeoDataFrame with PyPSA lines
    matches : list
        List of existing match dictionaries
    batch_size : int, optional
        Number of JAO lines to process in one batch
    max_candidates : int, optional
        Maximum PyPSA candidates to check for each JAO line
    time_limit_sec : int, optional
        Maximum seconds to run before returning results
    buffer_distance : float, optional
        Buffer distance in coordinate units (~500m at equator)
    min_score : float, optional
        Minimum score to consider a match valid

    Returns
    -------
    list
        Updated matches list
    """
    import time
    from shapely.strtree import STRtree
    from rtree import index
    start_time = time.time()

    print("\n===== FINDING VISUALLY MATCHING LINES =====")

    # Get all unmatched JAO lines
    unmatched_jao_ids = [r['jao_id'] for r in matches if not r.get('matched', False)]
    print(f"Total unmatched JAO lines: {len(unmatched_jao_ids)}")

    if not unmatched_jao_ids:
        print("No unmatched JAO lines to process")
        return matches

    # Get all unmatched PyPSA lines
    matched_pypsa_ids = set()
    for match in matches:
        if match.get('matched', False) and 'pypsa_ids' in match:
            if isinstance(match['pypsa_ids'], list):
                matched_pypsa_ids.update(match['pypsa_ids'])
            else:
                matched_pypsa_ids.add(match['pypsa_ids'])

    unmatched_pypsa = [str(id) for id in pypsa_gdf['id'].astype(str) if str(id) not in matched_pypsa_ids]
    unmatched_pypsa_gdf = pypsa_gdf[pypsa_gdf['id'].astype(str).isin(unmatched_pypsa)]
    print(f"Unmatched PyPSA lines: {len(unmatched_pypsa)}")

    # Build spatial index for unmatched PyPSA lines
    print("Building spatial index for PyPSA lines...")

    # Check if we have valid geometries
    valid_pypsa_geoms = unmatched_pypsa_gdf[unmatched_pypsa_gdf.geometry.notna()]
    if len(valid_pypsa_geoms) == 0:
        print("No valid PyPSA geometries for spatial indexing")
        return matches

    try:
        # Create spatial index - use STRtree if possible, otherwise use bounds-based filtering
        use_spatial_index = False
        try:
            pypsa_geoms = list(valid_pypsa_geoms.geometry)
            pypsa_idx = STRtree(pypsa_geoms)
            pypsa_id_map = {i: id for i, id in enumerate(valid_pypsa_geoms['id'])}
            use_spatial_index = True
            print(f"Created STRtree spatial index with {len(pypsa_geoms)} geometries")
        except Exception as e:
            print(f"STRtree creation failed: {e}, falling back to bounding box filtering")
            use_spatial_index = False

        # Count of fixed matches
        fixed_count = 0
        processed_count = 0

        # Process in batches to avoid memory issues and allow timeouts
        print(f"Processing in batches of {batch_size} lines...")

        for batch_start in range(0, len(unmatched_jao_ids), batch_size):
            batch_end = min(batch_start + batch_size, len(unmatched_jao_ids))
            batch_ids = unmatched_jao_ids[batch_start:batch_end]

            print(f"Processing batch {batch_start // batch_size + 1}: "
                  f"{len(batch_ids)} lines (overall progress: {processed_count}/{len(unmatched_jao_ids)})")

            batch_fixed = 0
            batch_jao_gdf = jao_gdf[jao_gdf['id'].astype(str).isin(batch_ids)]

            # For each unmatched JAO line in this batch
            for _, jao_row in batch_jao_gdf.iterrows():
                jao_id = str(jao_row['id'])
                processed_count += 1

                # Check timeout periodically
                if time.time() - start_time > time_limit_sec:
                    print(f"\nTIMEOUT: Visual matching stopped after {time_limit_sec} seconds")
                    print(f"Processed {processed_count}/{len(unmatched_jao_ids)} JAO lines")
                    print(f"Fixed {fixed_count} matches before timeout")
                    return matches

                # Skip if no valid geometry
                if jao_row.geometry is None or jao_row.geometry.is_empty:
                    continue

                jao_geom = jao_row.geometry
                jao_voltage = jao_row.get('v_nom', 0)

                # Create a buffer around the JAO line
                jao_buffer = jao_geom.buffer(buffer_distance)

                # Get direction vector for JAO line
                jao_vector = None
                try:
                    if jao_geom.geom_type == 'LineString':
                        coords = list(jao_geom.coords)
                        if len(coords) >= 2:
                            start = coords[0]
                            end = coords[-1]
                            jao_vector = np.array([end[0] - start[0], end[1] - start[1]])
                            jao_vector_len = np.linalg.norm(jao_vector)
                            if jao_vector_len > 0:
                                jao_vector = jao_vector / jao_vector_len
                    elif jao_geom.geom_type == 'MultiLineString':
                        if len(jao_geom.geoms) > 0:
                            start = list(jao_geom.geoms[0].coords)[0]
                            end = list(jao_geom.geoms[-1].coords)[-1]
                            jao_vector = np.array([end[0] - start[0], end[1] - start[1]])
                            jao_vector_len = np.linalg.norm(jao_vector)
                            if jao_vector_len > 0:
                                jao_vector = jao_vector / jao_vector_len
                except Exception as e:
                    print(f"Error calculating JAO vector for {jao_id}: {e}")
                    jao_vector = None

                # Create endpoint buffers if available
                try:
                    start_point = get_start_point(jao_geom)
                    end_point = get_end_point(jao_geom)

                    start_buffer = start_point.buffer(buffer_distance * 2) if start_point else None
                    end_buffer = end_point.buffer(buffer_distance * 2) if end_point else None
                except Exception as e:
                    print(f"Error creating endpoint buffers for {jao_id}: {e}")
                    start_buffer, end_buffer = None, None

                # Find candidate PyPSA lines using spatial index
                candidate_pypsa_ids = []

                if use_spatial_index:
                    try:
                        # Query spatial index with the buffer
                        candidate_indices = pypsa_idx.query(jao_buffer)
                        candidate_pypsa_ids = [pypsa_id_map[i] for i in candidate_indices[:max_candidates]]
                    except Exception as e:
                        print(f"Spatial query error for {jao_id}: {e}")
                        candidate_pypsa_ids = []
                else:
                    # Fallback: filter by bounding box overlap
                    jao_bounds = jao_buffer.bounds
                    for _, pypsa_row in unmatched_pypsa_gdf.iterrows():
                        if pypsa_row.geometry is None:
                            continue

                        pypsa_bounds = pypsa_row.geometry.bounds
                        if (jao_bounds[0] <= pypsa_bounds[2] and jao_bounds[2] >= pypsa_bounds[0] and
                                jao_bounds[1] <= pypsa_bounds[3] and jao_bounds[3] >= pypsa_bounds[1]):
                            candidate_pypsa_ids.append(pypsa_row['id'])
                            if len(candidate_pypsa_ids) >= max_candidates:
                                break

                # Find best matching PyPSA line among candidates
                best_match = None
                best_score = min_score  # Minimum threshold

                for pypsa_id in candidate_pypsa_ids:
                    pypsa_rows = unmatched_pypsa_gdf[unmatched_pypsa_gdf['id'] == pypsa_id]
                    if pypsa_rows.empty or pypsa_rows.iloc[0].geometry is None:
                        continue

                    pypsa_row = pypsa_rows.iloc[0]
                    pypsa_geom = pypsa_row.geometry
                    pypsa_voltage = pypsa_row.get('voltage', 0)

                    # Skip if voltage doesn't match
                    if not ((jao_voltage == 220 and pypsa_voltage == 220) or
                            (jao_voltage in [380, 400] and pypsa_voltage in [380, 400])):
                        continue

                    try:
                        if jao_buffer.intersects(pypsa_geom):
                            # Calculate percentage of PyPSA line within JAO buffer
                            intersection = jao_buffer.intersection(pypsa_geom)
                            overlap = intersection.length / pypsa_geom.length

                            # Check endpoint buffers for better matching
                            endpoint_match = 0

                            # Get PyPSA endpoints
                            try:
                                pypsa_start = get_start_point(pypsa_geom)
                                pypsa_end = get_end_point(pypsa_geom)

                                # Check if PyPSA endpoints are in JAO endpoint buffers
                                if start_buffer and pypsa_start and start_buffer.contains(pypsa_start):
                                    endpoint_match += 0.5
                                if end_buffer and pypsa_end and end_buffer.contains(pypsa_end):
                                    endpoint_match += 0.5
                            except Exception:
                                # If endpoint extraction fails, skip this part of scoring
                                pass

                            # Check direction similarity
                            direction_score = 0.5  # Default neutral
                            if jao_vector is not None and pypsa_geom.geom_type in ('LineString', 'MultiLineString'):
                                # Calculate PyPSA direction vector
                                pypsa_vector = None
                                try:
                                    if pypsa_geom.geom_type == 'LineString':
                                        coords = list(pypsa_geom.coords)
                                        if len(coords) >= 2:
                                            p_start = coords[0]
                                            p_end = coords[-1]
                                            pypsa_vector = np.array([p_end[0] - p_start[0], p_end[1] - p_start[1]])
                                    else:  # MultiLineString
                                        p_start = list(pypsa_geom.geoms[0].coords)[0]
                                        p_end = list(pypsa_geom.geoms[-1].coords)[-1]
                                        pypsa_vector = np.array([p_end[0] - p_start[0], p_end[1] - p_start[1]])

                                    if pypsa_vector is not None:
                                        pypsa_vector_len = np.linalg.norm(pypsa_vector)
                                        if pypsa_vector_len > 0:
                                            pypsa_vector = pypsa_vector / pypsa_vector_len
                                            # Calculate dot product (cosine of angle)
                                            direction_score = abs(np.dot(jao_vector, pypsa_vector))
                                except Exception:
                                    # If vector calculation fails, keep default score
                                    pass

                            # Combine scores: overlap, endpoint match, direction
                            combined_score = (
                                    0.5 * overlap +  # Line overlap
                                    0.3 * endpoint_match +  # Endpoint matching
                                    0.2 * direction_score  # Direction similarity
                            )

                            # If good match and better than previous matches
                            if combined_score > best_score:
                                best_match = pypsa_id
                                best_score = combined_score
                    except Exception as e:
                        continue  # Skip errors and continue with next candidate

                # If we found a good match
                if best_match:
                    # Find this JAO line in the results
                    for match in matches:
                        if str(match.get('jao_id', '')) == jao_id:
                            match['matched'] = True
                            match['pypsa_ids'] = [best_match]
                            match['match_quality'] = f"Visual Match ({best_score:.2f})"

                            # Remove from unmatched lists
                            if best_match in unmatched_pypsa:
                                unmatched_pypsa.remove(best_match)

                            batch_fixed += 1
                            fixed_count += 1
                            break

                # Progress indicator for large batches
                if len(batch_ids) > 50 and processed_count % 50 == 0:
                    elapsed = time.time() - start_time
                    print(f"  Progress: {processed_count}/{len(unmatched_jao_ids)} lines "
                          f"processed, {fixed_count} fixed ({elapsed:.1f}s elapsed)")

            print(f"  Batch complete: fixed {batch_fixed} matches in this batch")

            # Report time usage after each batch
            elapsed = time.time() - start_time
            remaining = len(unmatched_jao_ids) - processed_count
            print(f"  Time elapsed: {elapsed:.1f} seconds, "
                  f"estimated remaining: {remaining / max(1, processed_count) * elapsed:.1f} seconds")

        print(f"\nVisual matching complete: fixed {fixed_count}/{len(unmatched_jao_ids)} matches")
        return matches

    except Exception as e:
        import traceback
        print(f"ERROR in visual matching: {e}")
        print(traceback.format_exc())
        print(f"Returning {fixed_count} fixed matches before error")
        return matches

def catch_remaining_visual_matches(matching_results, jao_gdf, pypsa_gdf, buffer_meters=5000):
    """
    Catch remaining visual matches with extremely robust matching strategies:
    - Very aggressive buffering for problematic geometries
    - More lenient thresholds for simplified geometries
    - Multiple passes with different strategies
    - Special handling for parallel circuit recognition
    """
    print("\n=== ENHANCED FINAL PASS: AGGRESSIVE VISUAL MATCHING ===")

    # Get unmatched JAO lines
    matched_jao_ids = {r['jao_id'] for r in matching_results if r.get('matched', False)}
    unmatched_jao = jao_gdf[~jao_gdf['id'].astype(str).isin(matched_jao_ids)]

    # Get matched PyPSA IDs
    matched_pypsa_ids = set()
    for r in matching_results:
        if r.get('matched', False) and 'pypsa_ids' in r:
            if isinstance(r['pypsa_ids'], list):
                matched_pypsa_ids.update(r['pypsa_ids'])
            else:
                matched_pypsa_ids.update([pid.strip() for pid in r['pypsa_ids'].split(';')])

    # Get unmatched PyPSA lines
    unmatched_pypsa = pypsa_gdf[~pypsa_gdf['id'].astype(str).isin(matched_pypsa_ids)]

    print(f"Checking {len(unmatched_jao)} unmatched JAO lines against {len(unmatched_pypsa)} unmatched PyPSA lines")

    # Create a temporary working copy of unmatched PyPSA to update as we match
    available_pypsa_ids = list(unmatched_pypsa['id'])

    # Track matches found
    matches_found = 0
    multi_pass_matched = set()  # Track JAO lines that get matched

    # PASS 1: MATCH SIMPLIFIED GEOMETRIES WITH EXTREME BUFFERS
    print("\nPASS 1: Matching simplified geometries with extreme buffers")

    for _, jao_row in unmatched_jao.iterrows():
        jao_id = str(jao_row['id'])
        jao_geom = jao_row.geometry
        jao_voltage = _safe_int(jao_row.get('v_nom', 0))
        jao_length_km = jao_row.get('length_km', 0)

        if jao_geom is None or jao_id in multi_pass_matched:
            continue

        # Count points in geometry to determine simplification level
        point_count = 0
        if jao_geom.geom_type == 'LineString':
            point_count = len(list(jao_geom.coords))
        elif jao_geom.geom_type == 'MultiLineString':
            point_count = sum(len(list(part.coords)) for part in jao_geom.geoms)

        # Only process simplified geometries in this pass
        is_simplified = point_count <= 5
        if not is_simplified:
            continue

        print(f"  JAO {jao_id}: {point_count} points, {jao_length_km:.2f}km, {jao_voltage}kV")

        # Get direction vector (for extremely simplified geometries, this is crucial)
        jao_vector = None
        if jao_geom.geom_type == 'LineString' and len(list(jao_geom.coords)) >= 2:
            start_pt = jao_geom.coords[0]
            end_pt = jao_geom.coords[-1]
            jao_vector = np.array([end_pt[0] - start_pt[0], end_pt[1] - start_pt[1]])
            jao_vector_len = np.linalg.norm(jao_vector)
            if jao_vector_len > 0:
                jao_vector = jao_vector / jao_vector_len
        elif jao_geom.geom_type == 'MultiLineString' and len(jao_geom.geoms) > 0:
            first_line = jao_geom.geoms[0]
            last_line = jao_geom.geoms[-1]
            if len(first_line.coords) > 0 and len(last_line.coords) > 0:
                start_pt = first_line.coords[0]
                end_pt = last_line.coords[-1]
                jao_vector = np.array([end_pt[0] - start_pt[0], end_pt[1] - start_pt[1]])
                jao_vector_len = np.linalg.norm(jao_vector)
                if jao_vector_len > 0:
                    jao_vector = jao_vector / jao_vector_len

        # Use a VERY large buffer for these simplified geometries
        # The smaller the point count, the larger the buffer
        adaptive_buffer = buffer_meters * (6.0 / max(point_count, 2))  # Inverse scaling

        # Convert to degrees based on latitude
        lat = jao_geom.centroid.y
        buffer_deg = adaptive_buffer / (111111 * np.cos(np.radians(abs(lat))))

        print(f"    Using {adaptive_buffer:.0f}m buffer for {point_count} points")

        # Create buffer
        line_buffer = jao_geom.buffer(buffer_deg)

        # Track best match
        best_match = None
        best_score = 0.4  # Lower threshold for simplified geometries

        # For each available PyPSA line
        for pypsa_id in available_pypsa_ids:
            pypsa_row = pypsa_gdf[pypsa_gdf['id'] == pypsa_id]
            if pypsa_row.empty or pypsa_row.iloc[0].geometry is None:
                continue

            pypsa_geom = pypsa_row.iloc[0].geometry
            pypsa_voltage = _safe_int(pypsa_row.iloc[0].get('voltage', 0))
            pypsa_length_km = pypsa_row.iloc[0].get('length_km', 0)

            # Check voltage compatibility (more lenient: allow 220kV to match with 380/400kV)
            # This is needed because sometimes voltage levels are recorded differently
            voltage_match = False
            if jao_voltage == pypsa_voltage:
                voltage_match = True
            elif jao_voltage in [380, 400] and pypsa_voltage in [380, 400]:
                voltage_match = True
            elif jao_voltage == 220 and pypsa_voltage in [220, 380, 400]:
                # Allow 220kV JAO to match with 380kV PyPSA in desperate cases
                # This addresses voltage recording discrepancies
                voltage_match = True

            if not voltage_match:
                continue

            # Initial check with buffer
            if not line_buffer.intersects(pypsa_geom):
                continue

            # For extremely simplified geometries, focus on:
            # 1. Direction vector similarity (most important)
            # 2. Length similarity (very important)
            # 3. Minimal buffered overlap (less important)

            # Calculate direction similarity
            direction_score = 0.5  # Default neutral
            if jao_vector is not None:
                pypsa_vector = None
                if pypsa_geom.geom_type == 'LineString' and len(list(pypsa_geom.coords)) >= 2:
                    p_start = pypsa_geom.coords[0]
                    p_end = pypsa_geom.coords[-1]
                    pypsa_vector = np.array([p_end[0] - p_start[0], p_end[1] - p_start[1]])
                    pypsa_vector_len = np.linalg.norm(pypsa_vector)
                    if pypsa_vector_len > 0:
                        pypsa_vector = pypsa_vector / pypsa_vector_len
                elif pypsa_geom.geom_type == 'MultiLineString' and len(pypsa_geom.geoms) > 0:
                    p_start = list(pypsa_geom.geoms[0].coords)[0]
                    p_end = list(pypsa_geom.geoms[-1].coords)[-1]
                    pypsa_vector = np.array([p_end[0] - p_start[0], p_end[1] - p_start[1]])
                    pypsa_vector_len = np.linalg.norm(pypsa_vector)
                    if pypsa_vector_len > 0:
                        pypsa_vector = pypsa_vector / pypsa_vector_len

                if pypsa_vector is not None:
                    # Calculate dot product (cosine of angle)
                    dot_product = np.dot(jao_vector, pypsa_vector)
                    direction_score = abs(dot_product)

            # Calculate length similarity
            length_ratio = 0.0
            if jao_length_km > 0 and pypsa_length_km > 0:
                length_ratio = min(jao_length_km, pypsa_length_km) / max(jao_length_km, pypsa_length_km)

            # Check if this is likely to be a line segment (part of a longer line)
            is_segment = pypsa_length_km < jao_length_km * 0.7

            # Calculate overlap percentage (less important for simplified geoms)
            overlap = 0.0
            try:
                intersection = line_buffer.intersection(pypsa_geom)
                if hasattr(intersection, 'length'):
                    overlap = min(1.0, intersection.length / pypsa_geom.length)
            except:
                overlap = 0.2  # Default if calculation fails

            # Scoring strategy for extremely simplified geometries
            if point_count <= 3:  # Extremely simplified
                # Almost entirely direction and length based
                combined_score = (
                        0.60 * direction_score +  # Direction is by far the most important
                        0.35 * length_ratio +  # Length is next most important
                        0.05 * overlap  # Minimal weight on overlap
                )

                # Bonus for length near-matches
                if 0.9 <= length_ratio <= 1.1:
                    combined_score += 0.1

                # If this is likely a segment, be more lenient
                if is_segment and direction_score > 0.9:  # Very similar direction
                    combined_score += 0.1
            else:  # Somewhat simplified (4-5 points)
                combined_score = (
                        0.45 * direction_score +
                        0.30 * length_ratio +
                        0.25 * overlap
                )

            # Track best match
            if combined_score > best_score:
                best_score = combined_score
                best_match = {
                    'pypsa_id': pypsa_id,
                    'score': combined_score,
                    'details': {
                        'direction': direction_score,
                        'length_ratio': length_ratio,
                        'overlap': overlap,
                        'is_segment': is_segment
                    }
                }

        # If found a match
        if best_match:
            for match in matching_results:
                if match['jao_id'] == jao_id:
                    match['matched'] = True
                    match['pypsa_ids'] = [best_match['pypsa_id']]
                    match['match_quality'] = f"Simplified Geometry Match ({best_score:.2f})"
                    match['match_details'] = best_match['details']
                    multi_pass_matched.add(jao_id)
                    matches_found += 1

                    # Remove from available PyPSA IDs
                    if best_match['pypsa_id'] in available_pypsa_ids:
                        available_pypsa_ids.remove(best_match['pypsa_id'])

                    print(f"  ✓ Matched JAO {jao_id} to PyPSA {best_match['pypsa_id']} (score: {best_score:.2f})")
                    break
            else:
                # Create new match
                new_match = {
                    'jao_id': jao_id,
                    'jao_name': str(jao_row.get('NE_name', '')),
                    'jao_voltage': jao_voltage,
                    'matched': True,
                    'pypsa_ids': [best_match['pypsa_id']],
                    'match_quality': f"Simplified Geometry Match ({best_score:.2f})",
                    'match_details': best_match['details']
                }
                matching_results.append(new_match)
                multi_pass_matched.add(jao_id)
                matches_found += 1

                # Remove from available PyPSA IDs
                if best_match['pypsa_id'] in available_pypsa_ids:
                    available_pypsa_ids.remove(best_match['pypsa_id'])

                print(f"  ✓ Matched JAO {jao_id} to PyPSA {best_match['pypsa_id']} (score: {best_score:.2f})")

    print(f"\nAggressively matched {matches_found} additional lines")
    return matching_results


def fix_parallel_circuit_matching(matches, jao_gdf, pypsa_gdf):
    """
    Fix matching for parallel circuits with identical geometries.
    Preserves any matches marked with locked_by_corridor=True.
    """
    print("Fixing parallel circuit matching...")

    # Split matches into locked and modifiable groups
    locked_matches = []
    modifiable_matches = []
    locked_jao_ids = set()

    for match in matches:
        if match.get("locked_by_corridor", False):
            # Deep copy to ensure no accidental modifications
            locked_match = match.copy()
            locked_matches.append(locked_match)
            locked_jao_ids.add(str(locked_match.get('jao_id', '')))
        else:
            modifiable_matches.append(match)

    print(f"Found {len(locked_matches)} matches locked by corridor matching - will preserve these")
    print(f"Locked JAO IDs that will be preserved: {', '.join(sorted(locked_jao_ids))}")

    # If no modifiable matches, just return the locked ones
    if not modifiable_matches:
        print("No modifiable matches to process")
        return locked_matches

    # ---------- Process only modifiable matches ----------

    # Group JAO lines by geometry to find parallel circuits
    jao_by_geometry = defaultdict(list)
    for _, row in jao_gdf.iterrows():
        if row.geometry is not None:
            geom_wkt = row.geometry.wkt
            jao_by_geometry[geom_wkt].append(str(row['id']))

    # Find JAO parallel groups (multiple lines with identical geometry)
    jao_parallel_groups = {wkt: ids for wkt, ids in jao_by_geometry.items() if len(ids) > 1}
    print(f"Found {len(jao_parallel_groups)} JAO parallel groups")

    # Group PyPSA lines by geometry to find parallel circuits
    pypsa_by_geometry = defaultdict(list)
    for _, row in pypsa_gdf.iterrows():
        if row.geometry is not None:
            geom_wkt = row.geometry.wkt
            line_id = str(row.get('line_id', row.get('id', '')))
            pypsa_by_geometry[geom_wkt].append(line_id)

    # Find PyPSA parallel groups
    pypsa_parallel_groups = {wkt: ids for wkt, ids in pypsa_by_geometry.items() if len(ids) > 1}
    print(f"Found {len(pypsa_parallel_groups)} PyPSA parallel groups")

    # For each JAO parallel group
    for jao_wkt, jao_ids in jao_parallel_groups.items():
        # SKIP if ANY JAO in this group is locked - critical protection
        if any(str(jid) in locked_jao_ids for jid in jao_ids):
            print(f"  Skipping parallel group with JAO IDs {jao_ids} - contains locked corridor matches")
            continue

        print(f"\nProcessing JAO parallel group: {', '.join(jao_ids)}")

        # Get all matches for this group
        jao_matches = []
        for jao_id in jao_ids:
            for match in modifiable_matches:
                if str(match.get('jao_id', '')) == str(jao_id):
                    jao_matches.append(match)
                    break

        # Check if any are matched
        matched_jao_matches = [m for m in jao_matches if m.get('matched', False)]
        if not matched_jao_matches:
            print(f"  No matches found for this group")
            continue

        # Collect all PyPSA IDs used by any JAO in this group
        used_pypsa_ids = set()
        for match in matched_jao_matches:
            pypsa_ids = match.get('pypsa_ids', [])
            if isinstance(pypsa_ids, str):
                pypsa_ids = [id.strip() for id in pypsa_ids.split(';') if id.strip()]
            elif isinstance(pypsa_ids, list):
                used_pypsa_ids.update(pypsa_ids)
            else:
                continue

            for pypsa_id in pypsa_ids:
                used_pypsa_ids.add(str(pypsa_id))

        if not used_pypsa_ids:
            print(f"  No PyPSA IDs found for this group")
            continue

        # Find which PyPSA parallel group these IDs belong to
        target_pypsa_wkt = None
        target_pypsa_ids = []

        # First, check if any of the used PyPSA IDs belong to a parallel group
        for pypsa_id in used_pypsa_ids:
            for wkt, ids in pypsa_parallel_groups.items():
                if pypsa_id in ids:
                    target_pypsa_wkt = wkt
                    target_pypsa_ids = ids
                    break
            if target_pypsa_wkt:
                break

        # If no parallel group found, just use the first PyPSA ID's geometry
        if not target_pypsa_wkt and used_pypsa_ids:
            first_pypsa_id = list(used_pypsa_ids)[0]
            pypsa_rows = pypsa_gdf[pypsa_gdf['id'].astype(str) == first_pypsa_id]

            if not pypsa_rows.empty and pypsa_rows.iloc[0].geometry is not None:
                geom_wkt = pypsa_rows.iloc[0].geometry.wkt
                target_pypsa_wkt = geom_wkt
                target_pypsa_ids = pypsa_by_geometry.get(geom_wkt, [first_pypsa_id])

        if not target_pypsa_ids:
            print(f"  Could not identify target PyPSA lines")
            continue

        # Check if we have enough PyPSA lines for all JAO lines
        if len(target_pypsa_ids) < len(jao_ids):
            print(f"  Warning: Not enough PyPSA lines ({len(target_pypsa_ids)}) for all JAO lines ({len(jao_ids)})")
            print(f"  Will match as many as possible")

        # Match them one-to-one in sorted order
        sorted_jao_ids = sorted(jao_ids)
        sorted_pypsa_ids = sorted(target_pypsa_ids)

        print(f"  Found matching PyPSA parallel group: {', '.join(sorted_pypsa_ids)}")
        print(f"  JAO lines: {', '.join(sorted_jao_ids)}")
        for i, jao_id in enumerate(sorted_jao_ids):
            if i >= len(sorted_pypsa_ids):
                break

            # Find the match for this JAO ID in modifiable_matches (not locked)
            for match in modifiable_matches:
                if str(match.get('jao_id', '')) == str(jao_id):
                    # Double-check this isn't locked (should be impossible but just to be safe)
                    if match.get('locked_by_corridor', False):
                        print(f"    WARNING: JAO {jao_id} is locked but in modifiable matches - skipping!")
                        continue

                    pypsa_id = sorted_pypsa_ids[i]
                    print(f"    Matched JAO {jao_id} to PyPSA {pypsa_id}")

                    # Update the match
                    match['matched'] = True
                    match['pypsa_ids'] = [pypsa_id]
                    match['match_quality'] = "Parallel Circuit - One-to-One"
                    break

    # Handle special case for Altenfeld-Redwitz lines
    altenfeld_jao = []
    for _, row in jao_gdf.iterrows():
        jao_id = str(row['id'])
        # Skip locked JAO IDs
        if jao_id in locked_jao_ids:
            continue

        name = str(row.get('NE_name', ''))
        if 'Altenfeld - Redwitz' in name:
            altenfeld_jao.append(jao_id)

    if len(altenfeld_jao) >= 2:
        print(f"Found Altenfeld-Redwitz JAO lines: {', '.join(altenfeld_jao)}")

        # Find matching PyPSA lines with relation/5486612-380 pattern
        altenfeld_pypsa = []
        for _, row in pypsa_gdf.iterrows():
            line_id = str(row.get('line_id', row.get('id', '')))
            if 'relation/5486612-380' in line_id:
                altenfeld_pypsa.append(line_id)

        if len(altenfeld_pypsa) >= 1:
            print(f"Found matching PyPSA line: {', '.join(altenfeld_pypsa)}")

            # Update all Altenfeld JAO lines to match to this PyPSA line
            for jao_id in altenfeld_jao:
                # Skip if locked
                if jao_id in locked_jao_ids:
                    print(f"  Skipping locked JAO {jao_id}")
                    continue

                for result in modifiable_matches:
                    if str(result.get('jao_id', '')) == jao_id:
                        result['matched'] = True
                        result['pypsa_ids'] = [altenfeld_pypsa[0]]  # Use the first one
                        result['match_quality'] = "Parallel Circuit - Special Case (Altenfeld-Redwitz)"
                        print(f"    Matched JAO {jao_id} to PyPSA {altenfeld_pypsa[0]}")
                        break

    # Check for Mecklar - Dipperz lines and similar cases
    mecklar_jao = []
    for _, row in jao_gdf.iterrows():
        jao_id = str(row['id'])
        # Skip locked JAO IDs
        if jao_id in locked_jao_ids:
            continue

        name = str(row.get('NE_name', ''))
        if 'Mecklar - Dipperz' in name:
            mecklar_jao.append(jao_id)

    if len(mecklar_jao) >= 2:
        print(f"Found Mecklar-Dipperz JAO lines: {', '.join(mecklar_jao)}")

        # Find matching PyPSA lines with relation/12819660-380 and relation/3688563-380
        mecklar_pypsa = []
        for _, row in pypsa_gdf.iterrows():
            line_id = str(row.get('line_id', row.get('id', '')))
            if 'relation/12819660-380' in line_id or 'relation/3688563-380' in line_id:
                mecklar_pypsa.append(line_id)

        if len(mecklar_pypsa) >= 2:
            print(f"Found matching PyPSA lines: {', '.join(mecklar_pypsa)}")

            # Sort both lists
            sorted_jao = sorted(mecklar_jao)
            sorted_pypsa = sorted(mecklar_pypsa)

            # Match them one-to-one
            for i, jao_id in enumerate(sorted_jao[:len(sorted_pypsa)]):
                # Skip if locked
                if jao_id in locked_jao_ids:
                    print(f"  Skipping locked JAO {jao_id}")
                    continue

                for result in modifiable_matches:
                    if str(result.get('jao_id', '')) == jao_id:
                        result['matched'] = True
                        result['pypsa_ids'] = [sorted_pypsa[i]]
                        result['match_quality'] = "Parallel Circuit - Special Case (Mecklar-Dipperz)"
                        print(f"    Matched JAO {jao_id} to PyPSA {sorted_pypsa[i]}")
                        break

    # Final verification step: Ensure no locked JAO IDs were modified
    for i, result in enumerate(modifiable_matches):
        jao_id = str(result.get('jao_id', ''))
        if jao_id in locked_jao_ids:
            print(f"WARNING: Found modified match for locked JAO {jao_id} - fixing!")
            # Find the original locked match
            for locked in locked_matches:
                if str(locked.get('jao_id', '')) == jao_id:
                    # Replace with the locked version
                    modifiable_matches[i] = locked.copy()
                    break

    # Return combined results
    return locked_matches + modifiable_matches


def enhanced_parallel_circuit_matching(matches, jao_gdf, pypsa_gdf):
    """Enhance parallel circuit matching for better results."""
    print("\n===== ENHANCED PARALLEL CIRCUIT MATCHING =====")

    # First filter out locked matches - don't try to modify them
    modifiable_matches = []
    locked_matches = []

    for match in matches:
        if match.get("locked_by_corridor", False):
            locked_matches.append(match)
        else:
            modifiable_matches.append(match)

    print(f"Found {len(locked_matches)} matches locked by corridor matching - will preserve these")

    # 1. Index all geometries by a simplified hash to find similar geometries
    def geometry_signature(geom):
        """Create a simplified signature of a geometry to find similar ones."""
        if geom is None:
            return None

        # For LineString, use first and last point plus total length
        if geom.geom_type == 'LineString':
            if len(geom.coords) < 2:
                return None

            first = geom.coords[0]
            last = geom.coords[-1]
            length = geom.length

            # Round coordinates to reduce noise
            return (round(first[0], 3), round(first[1], 3),
                    round(last[0], 3), round(last[1], 3),
                    round(length, 3))

        # For MultiLineString, use combined signature
        elif geom.geom_type == 'MultiLineString':
            if len(geom.geoms) == 0:
                return None

            # Use first and last points of the entire multilinestring
            first = geom.geoms[0].coords[0]
            last = geom.geoms[-1].coords[-1]
            length = sum(part.length for part in geom.geoms)

            return (round(first[0], 3), round(first[1], 3),
                    round(last[0], 3), round(last[1], 3),
                    round(length, 3))

        return None

    # Index JAO lines by geometry signature
    jao_by_signature = defaultdict(list)
    for _, row in jao_gdf.iterrows():
        jao_id = str(row['id'])
        if row.geometry is not None:
            sig = geometry_signature(row.geometry)
            if sig:
                jao_by_signature[sig].append(jao_id)

    # Index PyPSA lines by geometry signature
    pypsa_by_signature = defaultdict(list)
    for _, row in pypsa_gdf.iterrows():
        pypsa_id = str(row.get('line_id', row.get('id', '')))
        if row.geometry is not None:
            sig = geometry_signature(row.geometry)
            if sig:
                pypsa_by_signature[sig].append(pypsa_id)

    # Find parallel circuit groups
    parallel_jao_groups = {sig: ids for sig, ids in jao_by_signature.items() if len(ids) > 1}
    parallel_pypsa_groups = {sig: ids for sig, ids in pypsa_by_signature.items() if len(ids) > 1}

    print(f"Found {len(parallel_jao_groups)} JAO parallel circuit groups")
    print(f"Found {len(parallel_pypsa_groups)} PyPSA parallel circuit groups")

    # 2. For each JAO parallel group, find matching PyPSA parallel groups
    matches_to_update = []

    for jao_sig, jao_ids in parallel_jao_groups.items():
        print(f"\nProcessing JAO parallel group: {', '.join(jao_ids)}")

        # Skip if any JAO in this group is locked
        if any(jao_id in locked_matches for jao_id in jao_ids):
            print("  Skipping: contains locked corridor matches")
            continue

        # Find matching PyPSA groups
        matching_pypsa_groups = []

        # First try to find exact signature match
        if jao_sig in pypsa_by_signature:
            matching_pypsa_groups.append((jao_sig, pypsa_by_signature[jao_sig]))

        # If no exact match, find similar signatures (same endpoints but slightly different length)
        if not matching_pypsa_groups:
            jao_endpoints = jao_sig[:4]  # First 4 elements are the endpoints

            for pypsa_sig, pypsa_ids in pypsa_by_signature.items():
                pypsa_endpoints = pypsa_sig[:4]

                # Check if endpoints are close
                endpoints_match = all(abs(a - b) < 0.01 for a, b in zip(jao_endpoints, pypsa_endpoints))

                if endpoints_match:
                    matching_pypsa_groups.append((pypsa_sig, pypsa_ids))

        if not matching_pypsa_groups:
            print(f"  No matching PyPSA parallel groups found")
            continue

        # Sort by length similarity to find best match
        matching_pypsa_groups.sort(key=lambda x: abs(x[0][4] - jao_sig[4]))

        # Get best matching group
        best_pypsa_sig, best_pypsa_ids = matching_pypsa_groups[0]

        print(f"  Found matching PyPSA group: {', '.join(best_pypsa_ids)}")

        # Skip if we don't have enough PyPSA lines
        if len(best_pypsa_ids) < len(jao_ids):
            print(f"  Not enough PyPSA lines ({len(best_pypsa_ids)}) for JAO lines ({len(jao_ids)})")
            continue

        # Sort both lists
        sorted_jao_ids = sorted(jao_ids)
        sorted_pypsa_ids = sorted(best_pypsa_ids)

        # 3. Match them one-to-one
        for i, jao_id in enumerate(sorted_jao_ids):
            if i < len(sorted_pypsa_ids):
                # Find matching result for this JAO ID
                for result in matches:
                    if str(result.get('jao_id', '')) == jao_id:
                        # Skip if locked
                        if result.get("locked_by_corridor", False):
                            print(f"  Skipping locked JAO {jao_id}")
                            continue

                        # Store current match for logging
                        old_match = result.get('pypsa_ids', [])
                        if isinstance(old_match, list) and old_match:
                            old_match = old_match[0] if len(old_match) == 1 else str(old_match)
                        elif isinstance(old_match, str):
                            old_match = old_match
                        else:
                            old_match = "None"

                        # Update match
                        result['matched'] = True
                        result['pypsa_ids'] = [sorted_pypsa_ids[i]]
                        result['match_quality'] = f"Parallel Circuit - Comprehensive 1:1 (was: {old_match})"

                        matches_to_update.append((jao_id, sorted_pypsa_ids[i], old_match))
                        break

    # 4. Process unmatched JAO lines - try to match with any available parallel PyPSA lines
    unmatched_jao = [r for r in matches if not r.get('matched', False)]
    print(f"\nProcessing {len(unmatched_jao)} unmatched JAO lines")

    for result in unmatched_jao:
        jao_id = str(result.get('jao_id', ''))

        # Skip if locked
        if result.get("locked_by_corridor", False):
            print(f"  Skipping locked JAO {jao_id}")
            continue

        # Get JAO geometry
        jao_rows = jao_gdf[jao_gdf['id'].astype(str) == jao_id]
        if jao_rows.empty or jao_rows.iloc[0].geometry is None:
            continue

        jao_geom = jao_rows.iloc[0].geometry
        jao_sig = geometry_signature(jao_geom)

        if not jao_sig:
            continue

        # Find similar PyPSA groups
        similar_pypsa_ids = []

        # Check for similar endpoints with tolerance
        jao_endpoints = jao_sig[:4]

        for pypsa_sig, pypsa_ids in pypsa_by_signature.items():
            pypsa_endpoints = pypsa_sig[:4]

            # Check if endpoints are close
            endpoints_match = all(abs(a - b) < 0.02 for a, b in zip(jao_endpoints, pypsa_endpoints))

            # Check length similarity (within 10%)
            length_ratio = abs(jao_sig[4] - pypsa_sig[4]) / max(jao_sig[4], pypsa_sig[4])
            length_match = length_ratio < 0.1

            if endpoints_match and length_match:
                # Check if any of these PyPSA lines are still unmatched
                for pypsa_id in pypsa_ids:
                    # Count how many JAO lines are already matched to this PyPSA ID
                    usage_count = 0
                    for m in matches:
                        if not m.get('matched', False):
                            continue

                        m_pypsa_ids = m.get('pypsa_ids', [])
                        if isinstance(m_pypsa_ids, list):
                            if pypsa_id in m_pypsa_ids:
                                usage_count += 1
                        elif isinstance(m_pypsa_ids, str):
                            if pypsa_id in m_pypsa_ids.split(';') or pypsa_id in m_pypsa_ids.split(','):
                                usage_count += 1

                    # Get number of circuits for this PyPSA line
                    pypsa_rows = pypsa_gdf[pypsa_gdf['id'].astype(str) == pypsa_id]
                    circuits = 1
                    if not pypsa_rows.empty:
                        circuits = int(pypsa_rows.iloc[0].get('circuits', 1))

                    # If PyPSA line still has available capacity, use it
                    if usage_count < circuits:
                        similar_pypsa_ids.append(pypsa_id)

        # If we found matching PyPSA lines, use the first one
        if similar_pypsa_ids:
            print(f"  Found {len(similar_pypsa_ids)} potential PyPSA matches for unmatched JAO {jao_id}")

            # Sort by ID for determinism
            similar_pypsa_ids.sort()

            # Update match
            result['matched'] = True
            result['pypsa_ids'] = [similar_pypsa_ids[0]]
            result['match_quality'] = f"Parallel Circuit - Late Match"

            matches_to_update.append((jao_id, similar_pypsa_ids[0], "unmatched"))

    # Print summary of updates
    print("\nUpdated parallel circuit matches:")
    for jao_id, pypsa_id, old_match in matches_to_update:
        print(f"  JAO {jao_id}: {old_match} → {pypsa_id}")

    return locked_matches + modifiable_matches


def auto_extend_short_matches(matches, jao_gdf, pypsa_gdf, target_ratio=0.80, buffer_m=120.0, endpoint_grid_m=50.0):
    """
    Extend 'short' JAO→PyPSA matches by traversing adjacent PyPSA segments to
    build the full corridor chain.
    """
    from collections import defaultdict, deque
    import math
    import numpy as np
    from shapely.geometry import Point, LineString, MultiLineString
    from shapely.ops import unary_union

    print(f"\nExtending short matches (target ratio: {target_ratio:.0%})...")

    # ----- helpers -----
    def _snap_key(pt, grid_m=endpoint_grid_m):
        """Round projected point to a grid key."""
        if pt is None:
            return None
        x = round(pt.x / grid_m) * grid_m
        y = round(pt.y / grid_m) * grid_m
        return (float(x), float(y))

    def _endpoints(ls):
        """Return start & end shapely Points from a LineString."""
        try:
            coords = list(ls.coords)
            if len(coords) < 2:
                return None, None
            return Point(coords[0]), Point(coords[-1])
        except Exception:
            return None, None

    def _listify_ids(v):
        """pypsa_ids can be list or str with ',' or ';' separators."""
        if not v:
            return []
        if isinstance(v, (list, tuple, set)):
            out = [str(x).strip() for x in v if str(x).strip()]
        else:
            s = str(v)
            out = []
            for sep in (";", ","):
                if sep in s:
                    out = [t.strip() for t in s.split(sep)]
                    break
            if not out:
                out = [s.strip()]
        return [x for x in out if x]

    def _iter_line_parts(g):
        """Yield only valid LineStrings from g (deeply), ignoring empties."""
        if g is None:
            return
        if isinstance(g, LineString):
            if not g.is_empty:
                yield g
        elif isinstance(g, MultiLineString):
            for ls in g.geoms:
                if ls is not None and not ls.is_empty:
                    yield ls
        else:
            try:
                for sub in getattr(g, "geoms", []):
                    yield from _iter_line_parts(sub)
            except Exception:
                return

    def _segment_fits_jao(seg, jao_line, buffer_m_val=buffer_m):
        """Decide if a PyPSA seg lies along the JAO corridor."""
        try:
            dH = seg.hausdorff_distance(jao_line)  # in meters (same CRS)
            if dH <= buffer_m_val * 1.25:
                return True

            # Try buffer overlap
            jao_buffer = jao_line.buffer(buffer_m_val / 111000)  # approx degrees
            inter = jao_buffer.intersection(seg)
            if hasattr(inter, 'length') and inter.length > 0:
                iou = inter.length / seg.length
                return iou >= 0.20
            return False
        except Exception:
            return False

    # ----- build endpoint index -----
    # Index all PyPSA lines by endpoints for quick lookup
    endpoint_index = defaultdict(set)
    segments_by_id = {}

    for _, row in pypsa_gdf.iterrows():
        pypsa_id = str(row.get('line_id', row.get('id', '')))
        geom = row.geometry

        if geom is None or geom.is_empty:
            continue

        # Store segments for quick lookup
        segments_by_id[pypsa_id] = geom

        # Index endpoints (all line parts)
        for ls in _iter_line_parts(geom):
            p0, p1 = _endpoints(ls)
            if p0 is not None and p1 is not None:
                k0, k1 = _snap_key(p0), _snap_key(p1)
                endpoint_index[k0].add(pypsa_id)
                endpoint_index[k1].add(pypsa_id)

    # ----- process matches -----
    extended_count = 0

    for match in matches:
        # Skip unmatched or locked matches
        if not match.get('matched', False) or match.get('locked_by_corridor', False):
            continue

        # Get current length ratio
        jao_id = match.get('jao_id')
        length_ratio = float(match.get('length_ratio', 0) or match.get('coverage_ratio', 0) or 0)

        # Skip if already good enough
        if length_ratio >= target_ratio:
            continue

        # Get current PyPSA IDs
        pypsa_ids = _listify_ids(match.get('pypsa_ids', []))
        if not pypsa_ids:
            continue

        # Get JAO geometry and details
        jao_row = jao_gdf[jao_gdf['id'].astype(str) == str(jao_id)]
        if jao_row.empty:
            continue

        jao_geom = jao_row.iloc[0].geometry
        if jao_geom is None:
            continue

        jao_length_km = float(jao_row.iloc[0].get('length_km', 0) or 0)
        if jao_length_km <= 0:
            continue

        print(f"Extending match for JAO {jao_id}: current ratio {length_ratio:.1%} (target: {target_ratio:.0%})")

        # Build a set of current endpoint keys
        current_endpoints = set()
        for pid in pypsa_ids:
            segment = segments_by_id.get(pid)
            if segment is None:
                continue

            for ls in _iter_line_parts(segment):
                p0, p1 = _endpoints(ls)
                if p0 is not None and p1 is not None:
                    k0, k1 = _snap_key(p0), _snap_key(p1)
                    current_endpoints.add(k0)
                    current_endpoints.add(k1)

        # Find free endpoints (only appear once)
        endpoint_count = defaultdict(int)
        for pid in pypsa_ids:
            segment = segments_by_id.get(pid)
            if segment is None:
                continue

            for ls in _iter_line_parts(segment):
                p0, p1 = _endpoints(ls)
                if p0 is None or p1 is None:
                    continue
                k0, k1 = _snap_key(p0), _snap_key(p1)
                endpoint_count[k0] += 1
                endpoint_count[k1] += 1

        frontier = {k for k, count in endpoint_count.items() if count == 1}

        if not frontier:
            print(f"  No free endpoints found to extend from")
            continue

        print(f"  Found {len(frontier)} free endpoint(s)")

        # Get current path length and coverage
        current_path_length = 0
        for pid in pypsa_ids:
            row = pypsa_gdf[pypsa_gdf['id'].astype(str) == pid]
            if row.empty:
                continue

            length = row.iloc[0].get('length', 0)
            current_path_length += length / 1000.0  # Convert to km

        # Set of IDs already used
        used_ids = set(pypsa_ids)

        # BFS from frontier endpoints
        additions = 0
        MAX_ADDITIONS = 10  # Safety cap
        added_segments = []

        while frontier and additions < MAX_ADDITIONS:
            # Get neighbors from all frontier endpoints
            neighbors = set()
            for k in frontier:
                neighbors.update(endpoint_index.get(k, set()))

            # Remove already used segments
            neighbors = neighbors - used_ids

            if not neighbors:
                print(f"  No more connected segments to add")
                break

            # Find the best neighbor that fits along the JAO corridor
            best_pid = None
            best_score = 0

            for pid in neighbors:
                segment = segments_by_id.get(pid)
                if segment is None:
                    continue

                # Check if segment fits along the JAO corridor
                if not _segment_fits_jao(segment, jao_geom):
                    continue

                # Get length
                row = pypsa_gdf[pypsa_gdf['id'].astype(str) == pid]
                if row.empty:
                    continue

                segment_length = float(row.iloc[0].get('length', 0) or 0) / 1000.0  # km

                # Simple scoring: just use segment length as score
                # This prioritizes longer segments that fill more of the gap
                score = segment_length

                if score > best_score:
                    best_score = score
                    best_pid = pid

            # If no good neighbor found, stop extending
            if best_pid is None:
                print(f"  No suitable segment found to extend path")
                break

            # Add the best segment
            used_ids.add(best_pid)
            added_segments.append(best_pid)
            additions += 1

            # Update path length
            row = pypsa_gdf[pypsa_gdf['id'].astype(str) == best_pid]
            segment_length = float(row.iloc[0].get('length', 0) or 0) / 1000.0  # km
            current_path_length += segment_length

            # Update ratio
            new_ratio = current_path_length / jao_length_km

            print(f"  Added {best_pid} (length: {segment_length:.2f} km, new ratio: {new_ratio:.1%})")

            # If we've reached the target ratio, we can stop
            if new_ratio >= target_ratio:
                print(f"  Reached target ratio of {target_ratio:.0%}")
                break

            # Update frontier with the new segment's endpoints
            segment = segments_by_id.get(best_pid)
            new_frontier = set()

            for ls in _iter_line_parts(segment):
                p0, p1 = _endpoints(ls)
                if p0 is None or p1 is None:
                    continue

                k0, k1 = _snap_key(p0), _snap_key(p1)

                # Only consider endpoints that connect to unused segments
                for k in (k0, k1):
                    potential_neighbors = endpoint_index.get(k, set()) - used_ids
                    if potential_neighbors:
                        new_frontier.add(k)

            frontier = new_frontier

        # If we added segments, update the match
        if added_segments:
            # Update PyPSA IDs with the original plus added segments
            match['pypsa_ids'] = pypsa_ids + added_segments

            # Update length ratio
            match['length_ratio'] = current_path_length / jao_length_km

            # Add extension note to match quality
            quality = match.get('match_quality', '')
            if 'Extended' not in quality:
                match['match_quality'] = f"{quality} | Extended"

            print(f"  Extended match with {len(added_segments)} additional segments")
            print(f"  New ratio: {match['length_ratio']:.1%} (was {length_ratio:.1%})")
            extended_count += 1

    print(f"Extended {extended_count} matches to improve coverage")
    return matches


# ----------------- Main matcher Function -----------------

def run_original_matching(
        jao_gdf,
        pypsa_gdf,
        output_dir,
        include_110kv=True,
        include_dc_links=True,
        pypsa_110kv_path=None,
        pypsa_dc_path=None,
        endpoint_grid_m=50.0,
        match_buffer_m=200.0,
        min_coverage=0.35,
        target_ratio=0.80,
        verbose=False
):
    """Run the original matching algorithm with the original high match rate."""
    # Prepare output directory
    os.makedirs(output_dir, exist_ok=True)

    # ----- Load additional data sources if requested -----
    if include_110kv and pypsa_110kv_path:
        print("\n===== INCLUDING 110kV LINES IN ANALYSIS =====")
        pypsa_110kv_gdf = load_110kv_data(pypsa_110kv_path, verbose=verbose)

        if pypsa_110kv_gdf is not None and len(pypsa_110kv_gdf) > 0:
            print(f"Adding {len(pypsa_110kv_gdf)} 110kV lines to analysis")
            # Combine with main PyPSA dataframe
            pypsa_gdf = pd.concat([pypsa_gdf, pypsa_110kv_gdf])
            print(f"Combined PyPSA dataset now has {len(pypsa_gdf)} lines")
        else:
            print("No 110kV lines loaded or found")

    if include_dc_links and pypsa_dc_path:
        print("\n===== INCLUDING DC LINKS IN ANALYSIS =====")
        pypsa_dc_gdf = load_dc_links(pypsa_dc_path, verbose=verbose)

        if pypsa_dc_gdf is not None and len(pypsa_dc_gdf) > 0:
            print(f"Adding {len(pypsa_dc_gdf)} DC links to analysis")
            # Combine with main PyPSA dataframe
            pypsa_gdf = pd.concat([pypsa_gdf, pypsa_dc_gdf])
            print(f"Combined PyPSA dataset now has {len(pypsa_gdf)} lines")
        else:
            print("No DC links loaded or found")


    # ----- preliminary analyses -----
    parallel_groups, jao_to_group = identify_duplicate_geometries(jao_gdf)
    pypsa_parallel_groups = identify_parallel_pypsa_circuits(pypsa_gdf)
    G = build_network_graph(pypsa_gdf)

    # ----- matching phases -----

    # STAGE 1: Match known problem corridors
    print("\n===== MATCHING MULTI-SEGMENT CORRIDORS =====")
    corridor_matches, forced_unmatched_jao_ids = match_multi_segment_corridors(jao_gdf, pypsa_gdf)

    # Create a global set of JAO IDs that should never be matched
    never_match_jao_ids = set(str(id) for id in forced_unmatched_jao_ids)
    print(f"Created never-match list with {len(never_match_jao_ids)} JAO IDs")

    all_matches = corridor_matches.copy()

    # Track matched JAO IDs to avoid duplicate matches
    matched_jao_ids = {str(m['jao_id']) for m in all_matches if m.get('matched', False)}
    print(f"From corridor matching: {len(matched_jao_ids)} matched JAO IDs")

    # STAGE 2: Bus-based path matching
    print("\n===== BUS-BASED PATH MATCHING =====")
    # Only apply to unmatched JAO lines that aren't in never_match_jao_ids
    unmatched_jao_gdf = jao_gdf[
        (~jao_gdf['id'].astype(str).isin(matched_jao_ids)) &
        (~jao_gdf['id'].astype(str).isin(never_match_jao_ids))
        ]

    if len(unmatched_jao_gdf) > 0:
        print(f"Bus matching: Processing {len(unmatched_jao_gdf)} unmatched JAO lines")
        bus_matches = find_bus_based_paths(unmatched_jao_gdf, pypsa_gdf, G)

        # Add new matches to results
        for m in bus_matches:
            if m.get('matched', False) and str(m['jao_id']) not in matched_jao_ids and str(
                    m['jao_id']) not in never_match_jao_ids:
                all_matches.append(m)
                matched_jao_ids.add(str(m['jao_id']))

    # STAGE 3: Enhanced path-based matching
    print("\n===== ENHANCED PATH-BASED MATCHING =====")
    # Only apply to unmatched JAO lines that aren't in never_match_jao_ids
    unmatched_jao_gdf = jao_gdf[
        (~jao_gdf['id'].astype(str).isin(matched_jao_ids)) &
        (~jao_gdf['id'].astype(str).isin(never_match_jao_ids))
        ]

    if len(unmatched_jao_gdf) > 0:
        print(f"Path matching: Processing {len(unmatched_jao_gdf)} unmatched JAO lines")
        nearest_points = find_nearest_endpoints(unmatched_jao_gdf, pypsa_gdf, G)
        path_matches = path_based_line_matching(
            unmatched_jao_gdf, pypsa_gdf, G, nearest_points, parallel_groups, jao_to_group,
            existing_matches=all_matches
        )

        # Add new matches to results
        for m in path_matches:
            if m.get('matched', False) and str(m['jao_id']) not in matched_jao_ids and str(
                    m['jao_id']) not in never_match_jao_ids:
                all_matches.append(m)
                matched_jao_ids.add(str(m['jao_id']))

    # STAGE 4: Fix parallel circuits and do visual matching
    all_matches = fix_parallel_circuit_matching(all_matches, jao_gdf, pypsa_gdf)

    # Verify locked matches remained intact and re-enforce never_match status
    for jid in never_match_jao_ids:
        for i, m in enumerate(all_matches):
            if str(m.get('jao_id', '')) == jid:
                print(f"RE-ENFORCING: JAO {jid} - Setting matched=False, forced_unmatched=True")
                all_matches[i]['matched'] = False
                all_matches[i]['pypsa_ids'] = []
                all_matches[i]['locked_by_corridor'] = True
                all_matches[i]['forced_unmatched'] = True

    # Try visual matching for remaining unmatched lines
    find_visually_matching_lines(
        jao_gdf[~jao_gdf['id'].astype(str).isin(never_match_jao_ids)],
        pypsa_gdf,
        all_matches
    )

    # Final visual matching pass
    all_matches = catch_remaining_visual_matches(
        all_matches,
        jao_gdf[~jao_gdf['id'].astype(str).isin(never_match_jao_ids)],
        pypsa_gdf,
        buffer_meters=2000
    )

    # STAGE 5: Apply comprehensive parallel circuit handling
    print("\n===== APPLYING COMPREHENSIVE PARALLEL CIRCUIT MATCHING =====")
    all_matches = comprehensive_parallel_circuit_matching(all_matches, jao_gdf, pypsa_gdf)

    # Re-enforce never_match status
    for jid in never_match_jao_ids:
        for i, m in enumerate(all_matches):
            if str(m.get('jao_id', '')) == jid:
                print(f"RE-ENFORCING AFTER PARALLEL: JAO {jid} - Setting matched=False")
                all_matches[i]['matched'] = False
                all_matches[i]['pypsa_ids'] = []
                all_matches[i]['locked_by_corridor'] = True
                all_matches[i]['forced_unmatched'] = True

    # Additional parallel circuit refinement
    print("\n===== ENHANCED PARALLEL CIRCUIT MATCHING =====")
    all_matches = enhanced_parallel_circuit_matching(all_matches, jao_gdf, pypsa_gdf)

    # STAGE 6: Extend and complete matches
    print("\n===== EXTENDING SHORT MATCHES =====")
    all_matches = auto_extend_short_matches(all_matches, jao_gdf, pypsa_gdf, target_ratio=target_ratio,
                                            buffer_m=match_buffer_m)

    # First, modify your run_original_matching function to respect the fixed_jao_ids
    print("\n===== FINAL INSPECTION FOR MISSED PARALLEL CIRCUIT MATCHES =====")
    all_matches, fixed_jao_ids = find_parallel_missed_matches(all_matches, jao_gdf, pypsa_gdf)

    # Modify the RE-ENFORCING AFTER PARALLEL step (if you have this step)
    print("\n===== RE-ENFORCING AFTER PARALLEL =====")
    for i, match in enumerate(all_matches):
        jao_id = str(match.get('jao_id', ''))

        # Skip re-enforcement for lines that were fixed by parallel circuit finder
        if jao_id in fixed_jao_ids:
            print(f"Skipping re-enforcement for JAO {jao_id} - matched by parallel circuit finder")
            continue

        # Your existing re-enforcement code here

    # Final enforcement for forced unmatched lines
    print("\n===== FINAL ENFORCEMENT FOR FORCED UNMATCHED LINES =====")
    for i, match in enumerate(all_matches):
        jao_id = str(match.get('jao_id', ''))

        # Skip enforcement for lines fixed by parallel circuit finder
        if jao_id in fixed_jao_ids:
            print(f"Skipping enforcement for JAO {jao_id} - matched by parallel circuit finder")
            continue

        if jao_id in never_match_jao_ids or match.get('forced_unmatched', False):
            print(f"FINAL ENFORCEMENT: Forcing JAO {jao_id} to remain unmatched")
            all_matches[i]['matched'] = False
            all_matches[i]['pypsa_ids'] = []
            all_matches[i]['locked_by_corridor'] = True
            all_matches[i]['forced_unmatched'] = True

    # Modified failsafe specifically for JAO 245
    for i, match in enumerate(all_matches):
        jao_id = str(match.get('jao_id', ''))
        if jao_id == '245' and jao_id not in fixed_jao_ids:  # Skip if it was fixed by parallel circuit finder
            print(f"FAILSAFE: Ensuring JAO 245 remains unmatched due to circuit constraint with JAO 244")
            all_matches[i] = {
                'jao_id': '245',
                'matched': False,
                'pypsa_ids': [],
                'locked_by_corridor': True,
                'forced_unmatched': True,
                'match_quality': 'Unmatched - Circuit constraint with JAO 244'
            }
    # With this more general approach:
    print("\n===== FINAL ENFORCEMENT WITH PARALLEL CIRCUIT AWARENESS =====")

    # First, identify matches that were made by our parallel circuit finder
    parallel_matched_ids = set()
    for match in all_matches:
        if match.get('matched', False) and "Parallel Circuit Match" in str(match.get('match_quality', '')):
            parallel_matched_ids.add(str(match.get('jao_id', '')))

    print(f"Found {len(parallel_matched_ids)} lines matched by parallel circuit finder")

    # Apply forced unmatching, but respect parallel circuit matches
    for i, match in enumerate(all_matches):
        jao_id = str(match.get('jao_id', ''))

        # If it was successfully matched by our parallel circuit finder, preserve it
        if jao_id in parallel_matched_ids:
            print(f"Preserving parallel circuit match for JAO {jao_id}")
            continue

        # Otherwise apply normal enforcement
        if jao_id in never_match_jao_ids or match.get('forced_unmatched', False):
            print(f"FINAL ENFORCEMENT: Forcing JAO {jao_id} to remain unmatched")
            all_matches[i]['matched'] = False
            all_matches[i]['pypsa_ids'] = []
            all_matches[i]['locked_by_corridor'] = True
            all_matches[i]['forced_unmatched'] = True

    # Fail-safe specifically for JAO 245 (the known problem case)
    # Modified failsafe for JAO 245
    for i, match in enumerate(all_matches):
        jao_id = str(match.get('jao_id', ''))
        if jao_id == '245' and jao_id not in fixed_jao_ids:  # Skip if fixed by parallel circuit finder
            print(f"FAILSAFE: Ensuring JAO 245 remains unmatched due to circuit constraint with JAO 244")
            all_matches[i] = {
                'jao_id': '245',
                'matched': False,
                'pypsa_ids': [],
                'locked_by_corridor': True,
                'forced_unmatched': True,
                'match_quality': 'Unmatched - Circuit constraint with JAO 244'
            }
        elif jao_id == '245' and jao_id in fixed_jao_ids:
            print(f"Skipping failsafe for JAO 245 - matched by parallel circuit finder")

    # Fill in unmatched results for any JAO lines not in the results
    all_jao_ids = set(str(id) for id in jao_gdf['id'])
    result_jao_ids = set(str(match.get('jao_id', '')) for match in all_matches)
    missing_jao_ids = all_jao_ids - result_jao_ids

    if missing_jao_ids:
        print(f"Adding {len(missing_jao_ids)} missing JAO lines to results")
        for jao_id in missing_jao_ids:
            jao_rows = jao_gdf[jao_gdf['id'].astype(str) == jao_id]
            if jao_rows.empty:
                continue

            jao_row = jao_rows.iloc[0]

            all_matches.append({
                'jao_id': jao_id,
                'jao_name': str(jao_row.get('NE_name', '')),
                'jao_voltage': _safe_int(jao_row.get('v_nom', 0)),
                'matched': False,
                'match_quality': 'Unmatched - No suitable match found'
            })

    # Calculate final statistics
    total_jao = len(jao_gdf)
    matched_jao = sum(1 for m in all_matches if m.get('matched', False))
    match_percentage = matched_jao / total_jao * 100 if total_jao > 0 else 0

    print("\n===== MATCHING RESULTS =====")
    print(f"Total JAO lines: {total_jao}")
    print(f"Matched JAO lines: {matched_jao}")
    print(f"Match percentage: {match_percentage:.1f}%")

    # Export results to CSV
    csv_file = os.path.join(output_dir, 'jao_pypsa_matches.csv')
    create_results_csv(all_matches, csv_file)

    # Generate PyPSA with EIC codes
    pypsa_match_count, pypsa_with_eic, pypsa_eic_files = generate_pypsa_with_eic(
        all_matches, jao_gdf, pypsa_gdf, output_dir
    )

    # Generate JAO with PyPSA electrical parameters
    jao_with_pypsa = generate_jao_with_pypsa(
        all_matches, jao_gdf, pypsa_gdf, output_dir
    )

    # Create visualization
    map_file = os.path.join(output_dir, 'jao_pypsa_matches.html')
    create_jao_pypsa_visualization(jao_gdf, pypsa_gdf, all_matches, map_file)

    # Generate JAO with PyPSA electrical parameters (the line we added earlier)
    jao_with_pypsa = generate_jao_with_pypsa(
        all_matches, jao_gdf, pypsa_gdf, output_dir
    )

    # Generate enhanced summary table
    # Generate enhanced summary table
    from pathlib import Path
    summary_file = create_enhanced_summary_table(jao_gdf, pypsa_gdf, all_matches, output_dir=Path(output_dir))

    # Generate random match quality check report
    quality_report = random_match_quality_check(all_matches, jao_gdf, pypsa_gdf, output_dir)

    # Generate unmatched PyPSA analysis
    unmatched_analysis = export_unmatched_pypsa_details(all_matches, jao_gdf, pypsa_gdf, output_dir)

    print(f"Results saved to {output_dir}")
    return all_matches

def comprehensive_parallel_circuit_matching(matches, jao_gdf, pypsa_gdf):
    """
    Comprehensive parallel circuit matching that handles complex cases
    by analyzing electrical characteristics and fixing circuit allocations.
    """
    print("\n===== COMPREHENSIVE PARALLEL CIRCUIT MATCHING =====")

    # Track locked matches that we shouldn't modify
    locked_jao_ids = set()
    for match in matches:
        if match.get('locked_by_corridor', False):
            locked_jao_ids.add(str(match.get('jao_id', '')))

    # Group JAO lines by parallel circuits
    jao_groups = defaultdict(list)
    for _, row in jao_gdf.iterrows():
        jao_id = str(row['id'])

        # Skip locked JAO IDs
        if jao_id in locked_jao_ids:
            continue

        if row.get('parallel_group'):
            group_id = str(row.get('parallel_group'))
            jao_groups[group_id].append(jao_id)
        elif row.get('is_parallel_circuit', False):
            # Fallback for older datasets that don't have parallel_group
            geom_hash = row.geometry.wkt if row.geometry else None
            if geom_hash:
                jao_groups[geom_hash].append(jao_id)

    # Filter to only groups with multiple JAO lines
    jao_groups = {g: ids for g, ids in jao_groups.items() if len(ids) > 1}

    print(f"Found {len(jao_groups)} JAO parallel circuit groups")

    # Process each group
    changes_made = 0

    for group_id, jao_ids in jao_groups.items():
        # Skip if any JAO in this group is locked
        if any(jao_id in locked_jao_ids for jao_id in jao_ids):
            print(f"  Skipping group {group_id}: contains locked JAO IDs")
            continue

        print(f"\nProcessing JAO parallel group: {', '.join(jao_ids)}")

        # Get match details for this group
        group_matches = {}
        for jao_id in jao_ids:
            for match in matches:
                if str(match.get('jao_id', '')) == jao_id:
                    group_matches[jao_id] = match
                    break

        # Count how many are matched
        matched_count = sum(1 for m in group_matches.values() if m.get('matched', False))

        if matched_count == 0:
            print(f"  No matches in this group")
            continue

        if matched_count < len(jao_ids):
            print(f"  Only {matched_count} of {len(jao_ids)} JAO lines are matched")

            # Find all PyPSA IDs used by this group
            used_pypsa_ids = set()
            for match in group_matches.values():
                if not match.get('matched', False):
                    continue

                pypsa_ids = match.get('pypsa_ids', [])
                if isinstance(pypsa_ids, list):
                    used_pypsa_ids.update(pypsa_ids)
                elif isinstance(pypsa_ids, str):
                    used_pypsa_ids.update([id.strip() for id in pypsa_ids.split(';')])

            # Find unmatched JAO IDs in this group
            unmatched_jao = [jid for jid in jao_ids if not group_matches.get(jid, {}).get('matched', False)]

            # Try to find matching PyPSA lines for unmatched JAO lines
            for jao_id in unmatched_jao:
                jao_row = jao_gdf[jao_gdf['id'].astype(str) == jao_id].iloc[0]
                jao_geom = jao_row.geometry
                jao_buffer = jao_geom.buffer(0.005)  # ~500m buffer

                # Find potential PyPSA lines
                candidates = []

                for _, pypsa_row in pypsa_gdf.iterrows():
                    pypsa_id = str(pypsa_row.get('id', ''))

                    # Skip if already used by this group
                    if pypsa_id in used_pypsa_ids:
                        continue

                    # Check voltage compatibility
                    jao_voltage = _safe_int(jao_row.get('v_nom', 0))
                    pypsa_voltage = _safe_int(pypsa_row.get('voltage', 0))

                    if not ((jao_voltage == 220 and pypsa_voltage == 220) or
                            (jao_voltage in [380, 400] and pypsa_voltage in [380, 400])):
                        continue

                    # Check geometric overlap
                    if pypsa_row.geometry is None:
                        continue

                    if jao_buffer.intersects(pypsa_row.geometry):
                        # Calculate overlap percentage
                        intersection = jao_buffer.intersection(pypsa_row.geometry)
                        if hasattr(intersection, 'length') and intersection.length > 0:
                            overlap = intersection.length / pypsa_row.geometry.length

                            # Calculate length similarity
                            jao_length = jao_row.get('length_km', 0)
                            pypsa_length = pypsa_row.get('length_km', 0)

                            length_ratio = min(jao_length, pypsa_length) / max(jao_length, pypsa_length) if max(jao_length, pypsa_length) > 0 else 0

                            # Combine scores
                            score = 0.7 * overlap + 0.3 * length_ratio

                            candidates.append({
                                'pypsa_id': pypsa_id,
                                'score': score,
                                'overlap': overlap,
                                'length_ratio': length_ratio
                            })

                # Sort candidates by score
                candidates.sort(key=lambda x: x['score'], reverse=True)

                # If found good candidates
                if candidates and candidates[0]['score'] >= 0.5:
                    best = candidates[0]
                    print(f"  Found match for unmatched JAO {jao_id}: {best['pypsa_id']} (score: {best['score']:.2f})")

                    # Update match
                    match = group_matches[jao_id]
                    match['matched'] = True
                    match['pypsa_ids'] = [best['pypsa_id']]
                    match['match_quality'] = f"Parallel Circuit - Comprehensive Match ({best['score']:.2f})"

                    # Mark as used
                    used_pypsa_ids.add(best['pypsa_id'])
                    changes_made += 1

        # Ensure consistent match quality label for all matches in the group
        if matched_count > 0:
            for match in group_matches.values():
                if match.get('matched', False) and 'Parallel Circuit' not in str(match.get('match_quality', '')):
                    match['match_quality'] = f"Parallel Circuit - {match.get('match_quality', 'Matched')}"
                    changes_made += 1

    print(f"Made {changes_made} changes to parallel circuit matches")
    return matches


def find_parallel_missed_matches(matches, jao_gdf, pypsa_gdf):
    """
    Enhanced final inspection for parallel circuit matching using multiple strategies:
    1. Pattern-based matching (JAO to PyPSA)
    2. Path-based matching (JAO to PyPSA)
    3. Reverse matching (PyPSA to JAO)
    """
    from collections import defaultdict
    import re
    from shapely.geometry import LineString
    from shapely.ops import linemerge, unary_union

    print("\n===== ENHANCED FINAL INSPECTION FOR PARALLEL CIRCUIT MATCHES =====")

    # Track fixed matches and IDs
    matches_fixed = 0
    fixed_jao_ids = set()

    # Helper function for safe integer conversion
    def _safe_int(val, default=0):
        try:
            if val is None:
                return default
            return int(val)
        except (ValueError, TypeError):
            try:
                return int(float(val))
            except (ValueError, TypeError):
                return default

    # ===== STRATEGY 1: PATTERN-BASED MATCHING =====
    print("\n--- Strategy 1: Pattern-Based Matching ---")

    # First, identify all known parallel circuits by geometry
    jao_by_geometry = defaultdict(list)
    for _, row in jao_gdf.iterrows():
        if row.geometry is not None:
            geom_wkt = row.geometry.wkt
            jao_id = str(row['id'])
            jao_by_geometry[geom_wkt].append(jao_id)

    # Filter to groups with multiple lines (parallel circuits)
    parallel_groups = {wkt: ids for wkt, ids in jao_by_geometry.items() if len(ids) > 1}
    print(f"Found {len(parallel_groups)} JAO parallel groups by geometry")

    # Process each parallel group
    pattern_matches_fixed = 0

    for wkt, jao_ids in parallel_groups.items():
        print(f"\nInspecting parallel JAO group: {', '.join(jao_ids)}")

        # Get match status for all lines in this group
        group_status = {}
        for jao_id in jao_ids:
            for match in matches:
                if str(match.get('jao_id', '')) == jao_id:
                    status = "matched" if match.get('matched', False) else "unmatched"
                    group_status[jao_id] = {
                        'match': match,
                        'status': status,
                        'pypsa_ids': match.get('pypsa_ids', []) if match.get('matched', False) else [],
                        'reason': match.get('match_quality', 'Unknown')
                    }
                    break

        # Print status of this group
        for jao_id, info in group_status.items():
            pypsa_str = ", ".join(info['pypsa_ids']) if info['pypsa_ids'] else "none"
            print(f"  JAO {jao_id}: {info['status']}, PyPSA: {pypsa_str}, Reason: {info['reason']}")

        # Count matched and unmatched in this group
        matched = [jid for jid, info in group_status.items() if info['status'] == 'matched']
        unmatched = [jid for jid, info in group_status.items() if info['status'] == 'unmatched']

        if not matched:
            print("  No matches in this group to use as reference")
            continue

        if not unmatched:
            print("  All lines in this group are already matched")
            continue

        print(f"  Group has {len(matched)} matched and {len(unmatched)} unmatched lines")

        # For each matched line, extract pattern information from its PyPSA IDs
        matched_patterns = {}
        for jao_id in matched:
            info = group_status[jao_id]
            pypsa_ids = info['pypsa_ids']
            if not pypsa_ids:
                continue

            # Extract patterns (e.g., relation/1641463-380-a -> 1641463)
            patterns = {}
            for pypsa_id in pypsa_ids:
                # Try multiple pattern extraction approaches
                pattern_matches = []
                pattern_matches.extend(re.findall(r'\/(\d+)-', str(pypsa_id)))
                pattern_matches.extend(re.findall(r'relation\/(\d+)', str(pypsa_id)))
                pattern_matches.extend(re.findall(r'way\/(\d+)', str(pypsa_id)))

                if pattern_matches:
                    pattern = pattern_matches[0]
                    suffix = str(pypsa_id)[str(pypsa_id).find(pattern) + len(pattern):]
                    patterns[suffix] = pattern
                    print(f"    JAO {jao_id}: PyPSA {pypsa_id} -> pattern {pattern}, suffix {suffix}")

            if patterns:
                matched_patterns[jao_id] = patterns

        # For each unmatched line, try to find corresponding PyPSA IDs
        for jao_id in unmatched:
            print(f"  Trying to find match for unmatched JAO {jao_id}")

            # Get the JAO row and metadata
            jao_row = jao_gdf[jao_gdf['id'].astype(str) == jao_id]
            if jao_row.empty:
                print("    JAO row not found, skipping")
                continue

            # Get JAO properties
            jao_voltage = _safe_int(jao_row.iloc[0].get('v_nom', 0))
            jao_name = str(jao_row.iloc[0].get('NE_name', ''))

            # Try to find a match based on each matched reference
            candidates = []

            for ref_jao_id, patterns in matched_patterns.items():
                print(f"    Using JAO {ref_jao_id} as reference with {len(patterns)} patterns")

                # For each pattern+suffix in the reference
                for suffix, pattern in patterns.items():
                    print(f"      Searching for alternate to pattern {pattern} with suffix {suffix}")

                    # Look for PyPSA lines with different patterns but matching suffixes
                    for _, pypsa_row in pypsa_gdf.iterrows():
                        pypsa_id = str(pypsa_row.get('line_id', pypsa_row.get('id', '')))
                        pypsa_voltage = _safe_int(pypsa_row.get('voltage', 0))

                        # Skip if voltage doesn't match
                        if not ((jao_voltage == 220 and pypsa_voltage == 220) or
                                (jao_voltage in [380, 400] and pypsa_voltage in [380, 400])):
                            continue

                        # Skip if it contains the same pattern (we want different patterns)
                        if pattern in pypsa_id:
                            continue

                        # Check if suffix matches
                        if suffix in pypsa_id:
                            # Get the new pattern
                            new_pattern_matches = []
                            new_pattern_matches.extend(re.findall(r'\/(\d+)-', str(pypsa_id)))
                            new_pattern_matches.extend(re.findall(r'relation\/(\d+)', str(pypsa_id)))
                            new_pattern_matches.extend(re.findall(r'way\/(\d+)', str(pypsa_id)))

                            if new_pattern_matches:
                                new_pattern = new_pattern_matches[0]

                                # Skip if it's the same pattern (shouldn't happen due to earlier check)
                                if new_pattern == pattern:
                                    continue

                                # We found a candidate with different pattern but same suffix!
                                print(f"        Found candidate: {pypsa_id} with pattern {new_pattern}")
                                candidates.append({
                                    'pypsa_id': pypsa_id,
                                    'pattern': new_pattern,
                                    'suffix': suffix,
                                    'ref_jao_id': ref_jao_id
                                })

            # Analyze candidates - group by pattern
            pattern_groups = {}
            for candidate in candidates:
                pattern = candidate['pattern']
                pattern_groups.setdefault(pattern, []).append(candidate)

            # Find most promising pattern group
            best_pattern = None
            most_candidates = 0

            for pattern, group in pattern_groups.items():
                print(f"    Pattern {pattern} has {len(group)} candidates")
                if len(group) > most_candidates:
                    most_candidates = len(group)
                    best_pattern = pattern

            if best_pattern:
                print(f"    Best pattern: {best_pattern} with {most_candidates} candidates")

                # Get PyPSA IDs for this pattern
                best_pypsa_ids = [c['pypsa_id'] for c in pattern_groups[best_pattern]]

                # Find the match object for this JAO
                for i, match in enumerate(matches):
                    if str(match.get('jao_id', '')) == jao_id:
                        # Set the match fields
                        matches[i]['matched'] = True
                        matches[i]['pypsa_ids'] = best_pypsa_ids

                        # Reference the matched JAO
                        ref_jao_id = pattern_groups[best_pattern][0]['ref_jao_id']
                        matches[i]['match_quality'] = f"Parallel Circuit Match (pattern, parallel to JAO {ref_jao_id})"

                        # Add to tracking
                        pattern_matches_fixed += 1
                        matches_fixed += 1
                        fixed_jao_ids.add(jao_id)

                        print(f"    SUCCESS: Matched JAO {jao_id} to {len(best_pypsa_ids)} PyPSA lines")
                        break
            else:
                print(f"    No consistent pattern found for JAO {jao_id}")

    print(f"Pattern-based matching fixed {pattern_matches_fixed} parallel circuit matches")

    # ===== STRATEGY 2: PATH-BASED MATCHING =====
    print("\n--- Strategy 2: Path-Based Matching ---")

    # Helper to calculate path similarity
    def calculate_path_similarity(geom1, geom2, buffer_distance=0.005):  # ~500m buffer
        try:
            # Handle MultiLineString by converting to LineString if possible
            if hasattr(geom1, 'geoms'):
                try:
                    geom1 = linemerge(geom1)
                except:
                    # Use first geometry component if merge fails
                    geom1 = list(geom1.geoms)[0]

            if hasattr(geom2, 'geoms'):
                try:
                    geom2 = linemerge(geom2)
                except:
                    # Use first geometry component if merge fails
                    geom2 = list(geom2.geoms)[0]

            buffer1 = geom1.buffer(buffer_distance)
            buffer2 = geom2.buffer(buffer_distance)

            # Calculate IoU (Intersection over Union) to measure similarity
            intersection = buffer1.intersection(buffer2)
            union = buffer1.union(buffer2)

            if union.area > 0:
                return intersection.area / union.area
            return 0.0
        except Exception as e:
            print(f"Error in path similarity calculation: {str(e)}")
            return 0.0

    # Find unmatched JAO lines not already fixed by pattern matching
    unmatched_jao = []
    for i, match in enumerate(matches):
        jao_id = str(match.get('jao_id', ''))
        if not match.get('matched', False) and jao_id not in fixed_jao_ids:
            jao_row = jao_gdf[jao_gdf['id'].astype(str) == jao_id]
            if not jao_row.empty and jao_row.iloc[0].geometry is not None:
                unmatched_jao.append({
                    'index': i,
                    'jao_id': jao_id,
                    'match': match,
                    'geometry': jao_row.iloc[0].geometry,
                    'voltage': _safe_int(jao_row.iloc[0].get('v_nom', 0))
                })

    print(f"Found {len(unmatched_jao)} unmatched JAO lines to analyze with path-based matching")

    # Build reference of matched JAO lines with their PyPSA IDs
    matched_jao_refs = []
    for match in matches:
        if match.get('matched', False):
            jao_id = str(match.get('jao_id', ''))
            pypsa_ids = match.get('pypsa_ids', [])
            if isinstance(pypsa_ids, str):
                pypsa_ids = [p.strip() for p in pypsa_ids.split(';') if p.strip()]

            jao_row = jao_gdf[jao_gdf['id'].astype(str) == jao_id]
            if not jao_row.empty and jao_row.iloc[0].geometry is not None:
                voltage = _safe_int(jao_row.iloc[0].get('v_nom', 0))
                matched_jao_refs.append({
                    'jao_id': jao_id,
                    'geometry': jao_row.iloc[0].geometry,
                    'voltage': voltage,
                    'pypsa_ids': pypsa_ids
                })

    print(f"Found {len(matched_jao_refs)} matched JAO references to use as templates")

    path_matches_fixed = 0

    # For each unmatched line, find similar matched lines as references
    for unmatched in unmatched_jao:
        jao_id = unmatched['jao_id']
        jao_voltage = unmatched['voltage']
        jao_geom = unmatched['geometry']

        print(f"\nAnalyzing path-based match for JAO {jao_id}")

        # Find the best reference by geometry similarity
        best_ref = None
        best_similarity = 0.0

        for ref in matched_jao_refs:
            # Skip if voltage doesn't match
            if not ((jao_voltage == 220 and ref['voltage'] == 220) or
                    (jao_voltage in [380, 400] and ref['voltage'] in [380, 400])):
                continue

            # Calculate path similarity
            similarity = calculate_path_similarity(jao_geom, ref['geometry'])

            # Update if better match found (at least 60% overlap)
            if similarity > best_similarity and similarity > 0.6:
                best_similarity = similarity
                best_ref = ref

        if not best_ref:
            print(f"  No similar matched JAO line found as reference")
            continue

        print(f"  Found similar matched JAO {best_ref['jao_id']} with {best_similarity:.2f} path similarity")

        # Get the pattern structure from reference's PyPSA IDs
        ref_patterns = {}
        for pypsa_id in best_ref['pypsa_ids']:
            pattern_matches = []
            pattern_matches.extend(re.findall(r'\/(\d+)-', str(pypsa_id)))
            pattern_matches.extend(re.findall(r'relation\/(\d+)', str(pypsa_id)))
            pattern_matches.extend(re.findall(r'way\/(\d+)', str(pypsa_id)))

            if pattern_matches:
                pattern = pattern_matches[0]
                suffix = str(pypsa_id)[str(pypsa_id).find(pattern) + len(pattern):]
                ref_patterns[suffix] = pattern

        if not ref_patterns:
            print(f"  Couldn't extract pattern information from reference")
            continue

        # Find PyPSA lines that follow a similar path to the unmatched JAO line
        candidates = []

        for _, pypsa_row in pypsa_gdf.iterrows():
            pypsa_id = str(pypsa_row.get('line_id', pypsa_row.get('id', '')))
            pypsa_voltage = _safe_int(pypsa_row.get('voltage', 0))
            pypsa_geom = pypsa_row.geometry

            # Skip if voltage doesn't match
            if not ((jao_voltage == 220 and pypsa_voltage == 220) or
                    (jao_voltage in [380, 400] and pypsa_voltage in [380, 400])):
                continue

            # Skip if no geometry
            if pypsa_geom is None:
                continue

            # Calculate path similarity
            similarity = calculate_path_similarity(jao_geom, pypsa_geom)

            # Add candidate if similarity is good (50% overlap)
            if similarity > 0.5:
                # Extract pattern from PyPSA ID
                pattern_matches = []
                pattern_matches.extend(re.findall(r'\/(\d+)-', str(pypsa_id)))
                pattern_matches.extend(re.findall(r'relation\/(\d+)', str(pypsa_id)))
                pattern_matches.extend(re.findall(r'way\/(\d+)', str(pypsa_id)))

                if not pattern_matches:
                    continue

                pattern = pattern_matches[0]
                full_suffix = str(pypsa_id)[str(pypsa_id).find(pattern) + len(pattern):]

                # Check if this is a circuit pattern match with reference
                is_circuit_match = False
                matching_suffix = ""

                for ref_suffix in ref_patterns:
                    if ref_suffix in full_suffix:
                        is_circuit_match = True
                        matching_suffix = ref_suffix
                        break

                candidates.append({
                    'pypsa_id': pypsa_id,
                    'similarity': similarity,
                    'pattern': pattern,
                    'is_circuit_match': is_circuit_match,
                    'matching_suffix': matching_suffix
                })

        print(f"  Found {len(candidates)} candidate PyPSA lines with similar paths")

        # Group candidates by pattern
        pattern_groups = {}
        for candidate in candidates:
            pattern = candidate['pattern']
            pattern_groups.setdefault(pattern, []).append(candidate)

        # Find best pattern group
        best_pattern = None
        best_score = 0

        for pattern, group in pattern_groups.items():
            # Calculate group score: average similarity * (1 + bonus for circuit matches)
            circuit_matches = sum(1 for c in group if c['is_circuit_match'])
            avg_similarity = sum(c['similarity'] for c in group) / len(group)
            score = avg_similarity * (1 + 0.5 * circuit_matches / max(1, len(group)))

            print(
                f"    Pattern {pattern}: {len(group)} candidates, {circuit_matches} circuit matches, score: {score:.2f}")

            if score > best_score:
                best_score = score
                best_pattern = pattern

        if best_pattern and best_score > 0.6:
            print(f"    Best pattern: {best_pattern} with score {best_score:.2f}")

            # Get PyPSA IDs for this pattern
            best_pypsa_ids = [c['pypsa_id'] for c in pattern_groups[best_pattern]]

            # Update match
            match_idx = unmatched['index']
            matches[match_idx]['matched'] = True
            matches[match_idx]['pypsa_ids'] = best_pypsa_ids
            matches[match_idx][
                'match_quality'] = f"Parallel Circuit Match (path, similar to JAO {best_ref['jao_id']}, similarity: {best_similarity:.2f})"

            path_matches_fixed += 1
            matches_fixed += 1
            fixed_jao_ids.add(jao_id)

            print(f"    SUCCESS: Matched JAO {jao_id} to {len(best_pypsa_ids)} PyPSA lines using path-based matching")
        else:
            print(f"    No suitable pattern group found for JAO {jao_id}")

    print(f"Path-based matching fixed {path_matches_fixed} parallel circuit matches")

    # ===== STRATEGY 3: REVERSE MATCHING (PYPSA TO JAO) =====
    print("\n--- Strategy 3: Reverse Matching (PyPSA to JAO) ---")

    # First identify which PyPSA lines are already matched
    matched_pypsa_ids = set()
    for match in matches:
        if match.get('matched', False):
            pypsa_ids = match.get('pypsa_ids', [])
            if isinstance(pypsa_ids, str):
                pypsa_ids = [p.strip() for p in pypsa_ids.split(';') if p.strip()]
            matched_pypsa_ids.update(pypsa_ids)

    # Find unmatched PyPSA lines of the correct voltage class
    unmatched_pypsa = []
    for i, row in pypsa_gdf.iterrows():
        pypsa_id = str(row.get('line_id', row.get('id', '')))

        if pypsa_id not in matched_pypsa_ids:
            # Only consider high-voltage lines (220kV, 380kV, 400kV)
            voltage = _safe_int(row.get('voltage', 0))
            if voltage in [220, 380, 400] and row.geometry is not None:
                unmatched_pypsa.append({
                    'id': pypsa_id,
                    'voltage': voltage,
                    'geometry': row.geometry,
                    'original_row': row
                })

    print(f"Found {len(unmatched_pypsa)} unmatched PyPSA lines to analyze")

    # Look for pattern matches in existing matched PyPSA IDs
    # (e.g., if relation/1234-380-a is matched, relation/1234-380-b might be unmatched)
    pattern_to_jao = {}

    for match in matches:
        if match.get('matched', False):
            jao_id = str(match.get('jao_id', ''))
            pypsa_ids = match.get('pypsa_ids', [])
            if isinstance(pypsa_ids, str):
                pypsa_ids = [p.strip() for p in pypsa_ids.split(';') if p.strip()]

            # Extract patterns from matched PyPSA IDs
            for pypsa_id in pypsa_ids:
                pattern_matches = []
                pattern_matches.extend(re.findall(r'\/(\d+)-', str(pypsa_id)))
                pattern_matches.extend(re.findall(r'relation\/(\d+)', str(pypsa_id)))
                pattern_matches.extend(re.findall(r'way\/(\d+)', str(pypsa_id)))

                if pattern_matches:
                    pattern = pattern_matches[0]
                    suffix = str(pypsa_id)[str(pypsa_id).find(pattern) + len(pattern):]

                    # Store pattern -> JAO ID mapping
                    key = (pattern, suffix)
                    pattern_to_jao[key] = jao_id

    print(f"Found {len(pattern_to_jao)} pattern-suffix combinations from matched lines")

    # For each unmatched PyPSA line, try to find a JAO match
    reverse_matches_fixed = 0

    for pypsa_item in unmatched_pypsa:
        pypsa_id = pypsa_item['id']
        pypsa_voltage = pypsa_item['voltage']
        pypsa_geom = pypsa_item['geometry']

        print(f"\nAnalyzing reverse match for PyPSA {pypsa_id}")

        # Extract pattern from PyPSA ID
        pattern_matches = []
        pattern_matches.extend(re.findall(r'\/(\d+)-', str(pypsa_id)))
        pattern_matches.extend(re.findall(r'relation\/(\d+)', str(pypsa_id)))
        pattern_matches.extend(re.findall(r'way\/(\d+)', str(pypsa_id)))

        if not pattern_matches:
            print(f"  No pattern found in PyPSA ID, skipping")
            continue

        pypsa_pattern = pattern_matches[0]
        pypsa_suffix = str(pypsa_id)[str(pypsa_id).find(pypsa_pattern) + len(pypsa_pattern):]

        print(f"  PyPSA pattern: {pypsa_pattern}, suffix: {pypsa_suffix}")

        # APPROACH 1: Check if there's a direct pattern match
        potential_jao_ids = []

        # Check if this pattern matches any known pattern->JAO mapping
        for (pattern, suffix), jao_id in pattern_to_jao.items():
            # Check if this is a different pattern with same suffix
            if pattern != pypsa_pattern and suffix == pypsa_suffix:
                potential_jao_ids.append({
                    'jao_id': jao_id,
                    'match_type': 'pattern',
                    'score': 0.9,
                    'pattern': pattern
                })
                print(f"  Found pattern match: JAO {jao_id} with pattern {pattern}, suffix {suffix}")

        # APPROACH 2: Check for geometric similarity
        # Find JAO lines that follow a similar path
        for _, jao_row in jao_gdf.iterrows():
            jao_id = str(jao_row['id'])
            jao_voltage = _safe_int(jao_row.get('v_nom', 0))

            # Skip already matched JAO lines
            if jao_id in fixed_jao_ids:
                continue

            # Skip if already matched through normal process
            already_matched = False
            for match in matches:
                if str(match.get('jao_id', '')) == jao_id and match.get('matched', False):
                    already_matched = True
                    break

            if already_matched:
                continue

            # Skip if voltage doesn't match
            if not ((pypsa_voltage == 220 and jao_voltage == 220) or
                    (pypsa_voltage in [380, 400] and jao_voltage in [380, 400])):
                continue

            # Skip if no geometry
            if jao_row.geometry is None:
                continue

            # Calculate path similarity
            similarity = calculate_path_similarity(pypsa_geom, jao_row.geometry)

            # Add candidate if similarity is good (50% overlap)
            if similarity > 0.5:
                potential_jao_ids.append({
                    'jao_id': jao_id,
                    'match_type': 'geometry',
                    'score': similarity,
                    'similarity': similarity
                })
                print(f"  Found geometry match: JAO {jao_id} with similarity {similarity:.2f}")

        # If no potential matches, continue to next PyPSA line
        if not potential_jao_ids:
            print("  No potential JAO matches found")
            continue

        # Sort potential matches by score
        potential_jao_ids.sort(key=lambda x: x['score'], reverse=True)

        # Choose best JAO match
        best_match = potential_jao_ids[0]
        best_jao_id = best_match['jao_id']

        print(f"  Best match: JAO {best_jao_id} with score {best_match['score']:.2f} ({best_match['match_type']})")

        # Find JAO match object to update
        for i, match in enumerate(matches):
            if str(match.get('jao_id', '')) == best_jao_id:
                # Check if it's already matched
                if match.get('matched', True):
                    print(f"  WARNING: JAO {best_jao_id} is already matched, skipping")
                    break

                # Update the match
                existing_pypsa_ids = match.get('pypsa_ids', [])
                if not existing_pypsa_ids:
                    existing_pypsa_ids = []
                elif isinstance(existing_pypsa_ids, str):
                    existing_pypsa_ids = [p.strip() for p in existing_pypsa_ids.split(';') if p.strip()]

                # Add this PyPSA ID
                if pypsa_id not in existing_pypsa_ids:
                    existing_pypsa_ids.append(pypsa_id)

                # Update match
                matches[i]['matched'] = True
                matches[i]['pypsa_ids'] = existing_pypsa_ids

                if best_match['match_type'] == 'pattern':
                    matches[i]['match_quality'] = f"Parallel Circuit Match (reverse pattern, suffix match)"
                else:
                    matches[i][
                        'match_quality'] = f"Parallel Circuit Match (reverse geometry, similarity: {best_match['similarity']:.2f})"

                reverse_matches_fixed += 1
                matches_fixed += 1
                fixed_jao_ids.add(best_jao_id)

                print(f"  SUCCESS: Matched PyPSA {pypsa_id} to JAO {best_jao_id}")
                break

    print(f"Reverse matching fixed {reverse_matches_fixed} matches")

    print(f"\nTotal parallel circuit matches fixed across all strategies: {matches_fixed}")
    return matches, fixed_jao_ids

# ----- matcher Module Init -----

def load_dc_links(pypsa_dc_path, verbose=False):
    """Load DC links from CSV file."""
    if not os.path.isfile(pypsa_dc_path):
        if verbose:
            print(f"DC links file not found: {pypsa_dc_path}")
        return None

    try:
        pypsa_dc_df = pd.read_csv(pypsa_dc_path)
        pypsa_dc_geometry = pypsa_dc_df['geometry'].apply(parse_linestring)
        pypsa_dc_gdf = gpd.GeoDataFrame(pypsa_dc_df, geometry=pypsa_dc_geometry)

        # Add circuits column if missing
        if 'circuits' not in pypsa_dc_gdf.columns:
            pypsa_dc_gdf['circuits'] = 1

        # Ensure IDs are strings
        pypsa_dc_gdf['id'] = pypsa_dc_gdf['id'].astype(str)

        if verbose:
            print(f"Loaded {len(pypsa_dc_gdf)} DC links")

        return pypsa_dc_gdf
    except Exception as e:
        if verbose:
            print(f"Error loading DC links: {e}")
        return None

def load_110kv_data(pypsa_110kv_path, verbose=False):
    """Load 110kV PyPSA data from CSV file."""
    if not os.path.isfile(pypsa_110kv_path):
        if verbose:
            print(f"110kV PyPSA file not found: {pypsa_110kv_path}")
        return None

    try:
        pypsa_110_df = pd.read_csv(pypsa_110kv_path)
        pypsa_110_geometry = pypsa_110_df['geometry'].apply(parse_linestring)
        pypsa_110_gdf = gpd.GeoDataFrame(pypsa_110_df, geometry=pypsa_110_geometry)

        # Add circuits column if missing
        if 'circuits' not in pypsa_110_gdf.columns:
            pypsa_110_gdf['circuits'] = 1

        # Ensure IDs are strings
        pypsa_110_gdf['id'] = pypsa_110_gdf['id'].astype(str)

        if verbose:
            print(f"Loaded {len(pypsa_110_gdf)} 110kV PyPSA lines")

        return pypsa_110_gdf
    except Exception as e:
        if verbose:
            print(f"Error loading 110kV PyPSA data: {e}")
        return None