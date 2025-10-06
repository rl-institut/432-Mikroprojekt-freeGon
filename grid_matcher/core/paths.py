import math

import networkx as nx
import numpy as np
from shapely.ops import linemerge

from grid_matcher.core.parallel import enforce_circuit_constraints, process_parallel_circuits
from grid_matcher.utils.helpers import _safe_int


def path_based_line_matching(jao_gdf, pypsa_gdf, G, nearest_points, parallel_groups, jao_to_group,
                             existing_matches=None):
    """Match JAO lines to PyPSA lines using enhanced path-based matching."""
    print("Performing enhanced path-based line matching...")

    results = []

    # NEW: Check for already locked matches
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

        # NEW: Skip if this JAO ID is in a locked corridor match
        if str(jao_id) in locked_jao_ids:
            print(f"Skipping JAO {jao_id} - already matched by corridor")
            continue

        from grid_matcher.utils.helpers import _num
        jao_voltage = _num(jao_row.get('v_nom', 0))
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


def extend_incomplete_matches(matches, jao_gdf, pypsa_gdf, G, min_ratio=0.90, max_angle_diff=30):
    """
    Find and extend matches that have poor length coverage by adding connected
    PyPSA segments to create complete paths.
    """
    from collections import deque
    import numpy as np
    import math
    import pandas as pd
    from shapely.geometry import LineString, Point, MultiLineString
    from shapely.ops import linemerge

    print(f"\n===== EXTENDING INCOMPLETE MATCHES (target ratio: {min_ratio:.0%}) =====")

    # Create lookups for quick access - extract as dictionaries to avoid Series comparisons
    pypsa_by_id = {}
    for _, row in pypsa_gdf.iterrows():
        row_id = str(row.get('line_id', row.get('id', '')))
        pypsa_by_id[row_id] = dict(row)

    jao_by_id = {}
    for _, row in jao_gdf.iterrows():
        row_id = str(row['id'])
        jao_by_id[row_id] = dict(row)

    # Helper to safely get coordinates from geometry
    def get_geometry_coords(geom):
        if geom is None:
            return []

        if isinstance(geom, LineString):
            return list(geom.coords)
        elif isinstance(geom, MultiLineString):
            if len(geom.geoms) == 0:
                return []
            # Extract first and last coords from first and last parts
            first_coords = list(geom.geoms[0].coords)
            last_coords = list(geom.geoms[-1].coords)
            if not first_coords or not last_coords:
                return []
            return [first_coords[0], last_coords[-1]]
        elif hasattr(geom, 'coords'):
            return list(geom.coords)
        return []

    # Create endpoint index for PyPSA lines
    endpoint_index = {}
    for pid, row in pypsa_by_id.items():
        geom = row.get('geometry')
        if geom is None:
            continue

        coords = get_geometry_coords(geom)

        if len(coords) >= 2:
            start = tuple(coords[0])
            end = tuple(coords[-1])

            # Round to 5 decimal places for approximate matching
            start_key = (round(start[0], 5), round(start[1], 5))
            end_key = (round(end[0], 5), round(end[1], 5))

            # Index both endpoints
            if start_key not in endpoint_index:
                endpoint_index[start_key] = []
            if end_key not in endpoint_index:
                endpoint_index[end_key] = []

            endpoint_index[start_key].append(pid)
            endpoint_index[end_key].append(pid)

    # Helper to calculate angle between two segments
    def angle_between(g1, g2):
        # Extract direction vectors
        try:
            coords1 = get_geometry_coords(g1)
            coords2 = get_geometry_coords(g2)

            if len(coords1) < 2 or len(coords2) < 2:
                return 90  # Default if not enough points

            # Get direction vectors
            v1 = np.array([coords1[-1][0] - coords1[0][0], coords1[-1][1] - coords1[0][1]])
            v2 = np.array([coords2[-1][0] - coords2[0][0], coords2[-1][1] - coords2[0][1]])

            # Normalize vectors
            v1_norm = np.linalg.norm(v1)
            v2_norm = np.linalg.norm(v2)

            if v1_norm == 0 or v2_norm == 0:
                return 90

            v1 = v1 / v1_norm
            v2 = v2 / v2_norm

            # Calculate angle using dot product
            dot_product = np.clip(np.dot(v1, v2), -1.0, 1.0)
            angle_rad = np.arccos(dot_product)
            angle_deg = math.degrees(angle_rad)

            return angle_deg
        except Exception as e:
            print(f"Error calculating angle: {e}")
            return 90  # Default value on error

    # Helper to get endpoints of a PyPSA segment
    def get_endpoints(pid):
        row = pypsa_by_id.get(pid)
        if row is None:
            return None, None

        geom = row.get('geometry')
        if geom is None:
            return None, None

        coords = get_geometry_coords(geom)
        if len(coords) < 2:
            return None, None

        return tuple(coords[0]), tuple(coords[-1])

    # Helper to find the free endpoint of a path
    def find_free_endpoint(segment_ids):
        if not segment_ids:
            return None

        # Track all endpoints and how many times they appear
        endpoint_count = {}

        for pid in segment_ids:
            start, end = get_endpoints(pid)
            if start is None or end is None:
                continue

            start_key = (round(start[0], 5), round(start[1], 5))
            end_key = (round(end[0], 5), round(end[1], 5))

            endpoint_count[start_key] = endpoint_count.get(start_key, 0) + 1
            endpoint_count[end_key] = endpoint_count.get(end_key, 0) + 1

        # Free endpoints only appear once
        free_endpoints = [ep for ep, count in endpoint_count.items() if count == 1]

        # Return the first free endpoint (should be exactly 0 or 2 for a complete path)
        return free_endpoints[0] if free_endpoints else None

    # Process each match
    extended_count = 0

    for match in matches:
        # Skip unmatched or locked matches
        if not match.get('matched', False) or match.get('locked_by_corridor', False):
            continue

        # Get current length ratio
        jao_id = match.get('jao_id')
        length_ratio = float(match.get('length_ratio', 0) or match.get('coverage_ratio', 0) or 0)

        # Skip if already good enough
        if length_ratio >= min_ratio:
            continue

        # Get current PyPSA IDs
        pypsa_ids = match.get('pypsa_ids', [])
        if isinstance(pypsa_ids, str):
            pypsa_ids = [pid.strip() for pid in pypsa_ids.split(';') if pid.strip()]

        # Get JAO geometry and details
        jao_row = jao_by_id.get(jao_id)
        if jao_row is None:
            continue

        jao_geom = jao_row.get('geometry')
        if jao_geom is None:
            continue

        jao_voltage = _safe_int(jao_row.get('v_nom', jao_row.get('voltage', 0)))
        jao_length_km = float(jao_row.get('length_km', jao_row.get('length', 0)) or 0)

        # Skip if JAO length is zero
        if jao_length_km <= 0:
            continue

        print(f"Extending match for JAO {jao_id}: current ratio {length_ratio:.1%} (target: {min_ratio:.0%})")
        print(f"  Current PyPSA segments: {pypsa_ids}")

        # Calculate current path length
        current_path_length = 0
        for pid in pypsa_ids:
            row = pypsa_by_id.get(pid)
            if row:
                length_m = float(row.get('length', 0) or 0)
                current_path_length += length_m / 1000.0  # Convert to km

        # Find the free endpoint to extend from
        free_endpoint = find_free_endpoint(pypsa_ids)
        if not free_endpoint:
            print(f"  No free endpoint found to extend from")
            continue

        print(f"  Found free endpoint at {free_endpoint}")

        # Look for connected segments
        connected_segments = endpoint_index.get(free_endpoint, [])
        connected_segments = [pid for pid in connected_segments if pid not in pypsa_ids]

        if not connected_segments:
            print(f"  No connected segments found")
            continue

        print(f"  Found {len(connected_segments)} potential connecting segments")

        # Select the best segment to add
        best_segment = None
        best_score = -1

        for pid in connected_segments:
            row = pypsa_by_id.get(pid)
            if row is None:
                continue

            geom = row.get('geometry')
            if geom is None:
                continue

            # Check voltage compatibility
            pypsa_voltage = _safe_int(row.get('voltage', row.get('v_nom', 0)))
            if not ((jao_voltage == pypsa_voltage) or
                    (jao_voltage in [380, 400] and pypsa_voltage in [380, 400])):
                continue

            # Check circuit compatibility
            circuits = int(row.get('circuits', 1) or 1)

            # Calculate alignment with the JAO line
            angle = angle_between(jao_geom, geom)
            angle_score = 1.0 - min(angle, 90) / 90.0  # 1.0 for perfect alignment, 0.0 for perpendicular

            # Get segment length
            length_m = float(row.get('length', 0) or 0)
            segment_length_km = length_m / 1000.0  # Convert to km

            # Skip very short segments
            if segment_length_km < 0.5:
                continue

            # Score based on length contribution and alignment
            score = (segment_length_km / jao_length_km) * 0.7 + angle_score * 0.3

            # Check if adding this segment improves the ratio significantly
            new_length_ratio = (current_path_length + segment_length_km) / jao_length_km
            if new_length_ratio <= length_ratio + 0.05:  # Must improve by at least 5%
                continue

            if score > best_score:
                best_score = score
                best_segment = pid

        if best_segment:
            print(f"  Adding segment {best_segment} to improve match")

            # Add the segment to the match
            pypsa_ids.append(best_segment)
            match['pypsa_ids'] = pypsa_ids

            # Recalculate path length
            new_path_length = 0
            for pid in pypsa_ids:
                row = pypsa_by_id.get(pid)
                if row:
                    length_m = float(row.get('length', 0) or 0)
                    new_path_length += length_m / 1000.0

            # Update match metrics
            new_length_ratio = new_path_length / jao_length_km
            match['length_ratio'] = new_length_ratio
            match['path_length'] = new_path_length

            # Add extension note to match quality
            quality = match.get('match_quality', '')
            if 'Extended' not in quality:
                match['match_quality'] = f"{quality} | Extended"

            print(f"  New length ratio: {new_length_ratio:.1%} (was {length_ratio:.1%})")
            extended_count += 1
        else:
            print(f"  No suitable segment found to extend this match")

    print(f"Extended {extended_count} matches to improve coverage")
    return matches
