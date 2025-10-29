"""Matching algorithm for OpenStreetMap to PyPSA 110kV lines with comprehensive debugging."""

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

# Import shared utility functions
from grid_matcher.utils.helpers import calculate_length_meters, get_start_point, get_end_point
from grid_matcher.matcher.original_matcher import (
    build_network_graph, identify_duplicate_geometries, identify_parallel_pypsa_circuits,
    find_nearest_endpoints, fix_parallel_circuit_matching, find_visually_matching_lines,
    catch_remaining_visual_matches, enhanced_parallel_circuit_matching, auto_extend_short_matches,
    _safe_int, _num, _to_km
)


def load_osm_data(osm_path, verbose=False):
    """
    Load OpenStreetMap transmission line data.
    """
    print(f"\n===== LOADING OSM DATA =====")
    print(f"Loading OSM lines from {osm_path}")

    try:
        osm_df = pd.read_csv(osm_path)
        print(f"Loaded raw OSM dataframe with {len(osm_df)} rows and {len(osm_df.columns)} columns")
        print(f"Available columns: {', '.join(osm_df.columns)}")

        # Try WKT format first (from 'geometry' column)
        from shapely import wkt
        import binascii
        from shapely import wkb

        valid_geometries = None

        # Try the 'geometry' column first (WKT format)
        if 'geometry' in osm_df.columns:
            print("Trying to parse WKT geometries from 'geometry' column...")

            def parse_wkt(geom_str):
                try:
                    if isinstance(geom_str, str) and 'LINESTRING' in geom_str:
                        return wkt.loads(geom_str)
                    elif isinstance(geom_str, str) and 'MULTILINESTRING' in geom_str:
                        return wkt.loads(geom_str)
                    return None
                except Exception as e:
                    if verbose:
                        print(f"WKT parsing error: {e}")
                    return None

            wkt_geometries = osm_df['geometry'].apply(parse_wkt)
            valid_wkt = sum(1 for g in wkt_geometries if g is not None)
            print(f"  Parsed {valid_wkt} valid WKT geometries")

            if valid_wkt > 0:
                valid_geometries = wkt_geometries

        # If WKT failed, try the 'geom' column (WKB hex format)
        if valid_geometries is None or sum(1 for g in valid_geometries if g is not None) == 0:
            if 'geom' in osm_df.columns:
                print("Trying to parse WKB geometries from 'geom' column...")

                def parse_wkb_hex(geom_str):
                    try:
                        if isinstance(geom_str, str) and len(geom_str) > 10:
                            # Convert hex to binary and parse as WKB
                            binary_wkb = binascii.unhexlify(geom_str)
                            return wkb.loads(binary_wkb)
                        return None
                    except Exception as e:
                        if verbose:
                            print(f"WKB parsing error: {e}")
                        return None

                wkb_geometries = osm_df['geom'].apply(parse_wkb_hex)
                valid_wkb = sum(1 for g in wkb_geometries if g is not None)
                print(f"  Parsed {valid_wkb} valid WKB geometries")

                if valid_wkb > 0:
                    valid_geometries = wkb_geometries

        # If no valid geometries found
        if valid_geometries is None or sum(1 for g in valid_geometries if g is not None) == 0:
            print("CRITICAL ERROR: Could not parse any valid geometries")
            print("Checking the first few records of each geometry column:")

            for col in ['geom', 'geometry']:
                if col in osm_df.columns:
                    print(f"\nSample {col} values:")
                    for i in range(min(3, len(osm_df))):
                        val = osm_df[col].iloc[i]
                        print(f"  Row {i}: {val[:50]}...")

            # Create empty geometries as a last resort
            print("Creating empty GeoDataFrame with NULL geometries")
            valid_geometries = [None] * len(osm_df)

        # Create the GeoDataFrame
        osm_gdf = gpd.GeoDataFrame(osm_df, geometry=valid_geometries)

        # Set coordinate reference system
        if osm_gdf.crs is None:
            print("Setting CRS to EPSG:4326 (WGS84)")
            osm_gdf.set_crs(epsg=4326, inplace=True)

        # Ensure IDs are strings - use result_id as the main identifier
        osm_gdf['id'] = osm_df['result_id'].astype(str)
        print(f"Using 'result_id' as primary identifier")

        # Add endpoints for matching (only for valid geometries)
        def safe_get_point(func, geom):
            if geom is not None:
                try:
                    return func(geom)
                except:
                    return None
            return None

        osm_gdf['start_point'] = osm_gdf.geometry.apply(lambda g: safe_get_point(get_start_point, g))
        osm_gdf['end_point'] = osm_gdf.geometry.apply(lambda g: safe_get_point(get_end_point, g))

        valid_start_points = sum(1 for p in osm_gdf['start_point'] if p is not None)
        valid_end_points = sum(1 for p in osm_gdf['end_point'] if p is not None)
        print(f"Valid start points: {valid_start_points}/{len(osm_gdf)}")
        print(f"Valid end points: {valid_end_points}/{len(osm_gdf)}")

        # Calculate lengths in km
        def safe_length(geom):
            if geom is not None:
                try:
                    return calculate_length_meters(geom) / 1000
                except:
                    return 0
            return 0

        if 'length' in osm_df.columns:
            osm_gdf['length_km'] = osm_df['length'] / 1000.0
            print(f"Using existing 'length' field divided by 1000 for km")
        else:
            osm_gdf['length_km'] = osm_gdf.geometry.apply(safe_length)
            print(f"Calculated length_km from geometries")

        # Log length statistics
        lengths = osm_gdf['length_km'].dropna()
        if len(lengths) > 0:
            print(f"Length statistics (km): min={lengths.min():.2f}, max={lengths.max():.2f}, avg={lengths.mean():.2f}")
        else:
            print("Warning: No valid length values found")

        # Set voltage to 110kV for all lines
        osm_gdf['voltage'] = 110
        osm_gdf['v_nom'] = 110
        print(f"Set voltage=110 and v_nom=110 for all OSM lines")

        # Extract electrical parameters if available
        params_available = []
        if 'br_r' in osm_df.columns:
            osm_gdf['r'] = osm_df['br_r']
            params_available.append('r')
        if 'br_x' in osm_df.columns:
            osm_gdf['x'] = osm_df['br_x']
            params_available.append('x')
        if 'br_b' in osm_df.columns:
            osm_gdf['b'] = osm_df['br_b']
            params_available.append('b')

        print(f"Electrical parameters available: {', '.join(params_available) if params_available else 'None'}")

        # Calculate per-km values
        for param in ['r', 'x', 'b']:
            if param in osm_gdf.columns:
                osm_gdf[f'{param}_per_km'] = osm_gdf.apply(
                    lambda row: row[param] / row['length_km'] if row['length_km'] > 0 else 0, axis=1
                )
                non_zero_values = sum(1 for v in osm_gdf[f'{param}_per_km'] if v > 0)
                print(f"{param}_per_km: {non_zero_values} non-zero values")

        # Add number of circuits
        if 'cables' in osm_df.columns:
            osm_gdf['circuits'] = osm_df['cables'] // 3  # Assuming 3 cables per circuit
            # Ensure at least 1 circuit
            osm_gdf['circuits'] = osm_gdf['circuits'].apply(lambda x: max(1, x))
            circuit_counts = osm_gdf['circuits'].value_counts()
            print(f"Circuit counts from 'cables' column: {dict(circuit_counts)}")
        else:
            osm_gdf['circuits'] = 1
            print(f"No 'cables' column found, defaulting to 1 circuit for all lines")

        # Identify parallel circuits by looking at identical paths
        valid_geoms = osm_gdf[osm_gdf.geometry.notna()]
        osm_geometry_groups = defaultdict(list)

        for idx, row in valid_geoms.iterrows():
            wkt_str = row.geometry.wkt
            osm_geometry_groups[wkt_str].append(row['id'])

        # Mark parallel circuits
        osm_gdf['is_parallel_circuit'] = False
        osm_gdf['parallel_group'] = None

        parallel_groups = 0
        parallel_lines = 0

        for wkt_str, ids in osm_geometry_groups.items():
            if len(ids) > 1:
                group_id = '_'.join(sorted(ids))
                osm_gdf.loc[osm_gdf['id'].isin(ids), 'is_parallel_circuit'] = True
                osm_gdf.loc[osm_gdf['id'].isin(ids), 'parallel_group'] = group_id
                parallel_groups += 1
                parallel_lines += len(ids)

        print(f"Found {parallel_groups} parallel circuit groups with a total of {parallel_lines} lines")

        # Report valid geometries
        valid_geom_count = osm_gdf.geometry.notna().sum()
        print(f"Loaded {len(osm_gdf)} OSM lines with {valid_geom_count} valid geometries")

        # Check for potential issues
        null_geoms = osm_gdf.geometry.isna().sum()
        if null_geoms > 0:
            print(f"WARNING: {null_geoms} OSM lines have NULL geometries")

        invalid_geoms = sum(1 for g in osm_gdf.geometry if g is not None and not g.is_valid)
        if invalid_geoms > 0:
            print(f"WARNING: {invalid_geoms} OSM lines have invalid geometries")

        zero_length = sum(1 for l in osm_gdf['length_km'] if l == 0)
        if zero_length > 0:
            print(f"WARNING: {zero_length} OSM lines have zero length")

        # Filter to keep only rows with valid geometries
        if valid_geom_count > 0:
            valid_osm_gdf = osm_gdf[osm_gdf.geometry.notna()].copy()
            print(f"Filtered to {len(valid_osm_gdf)} OSM lines with valid geometries")
            return valid_osm_gdf
        else:
            print("WARNING: Returning GeoDataFrame with no valid geometries")
            return osm_gdf

    except Exception as e:
        print(f"ERROR loading OSM data: {e}")
        import traceback
        print(traceback.format_exc())
        return None


def path_based_osm_matching(osm_gdf, pypsa_gdf, G, nearest_points, parallel_groups, osm_to_group):
    """
    Match OSM lines to PyPSA lines using enhanced path-based matching for 110kV lines.
    This is similar to JAO-PyPSA matching but with parameters optimized for 110kV networks.
    """
    print("\n===== PATH-BASED MATCHING DEBUG =====")
    print(f"Starting path-based matching for OSM-PyPSA 110kV lines...")
    print(f"OSM lines: {len(osm_gdf)}")
    print(f"PyPSA lines: {len(pypsa_gdf)}")
    print(f"Graph nodes: {len(G.nodes)}")
    print(f"Graph edges: {len(G.edges)}")
    print(f"OSM lines with nearest points data: {len(nearest_points)}")
    print(f"Parallel groups: {len(parallel_groups)}")

    # DEBUG: Track statistics
    candidates_counts = []
    paths_found = 0
    paths_attempted = 0
    paths_rejected = {'no_endpoints': 0, 'length_ratio': 0, 'voltage': 0,
                      'circuits': 0, 'path_not_found': 0, 'other': 0}

    results = []

    # Track PyPSA line usage for circuit constraints
    pypsa_usage = {}
    group_used_pypsa = {}  # Track which PyPSA lines are used by each parallel group

    # Process each OSM line in sorted order for determinism
    print(f"\nProcessing OSM lines one by one...")
    for _, osm_row in sorted(osm_gdf.iterrows(), key=lambda it: str(it[1]['id'])):
        osm_id = osm_row['id']
        osm_voltage = _safe_int(osm_row.get('v_nom', 0))
        osm_length = osm_row['length_km']

        paths_attempted += 1
        if paths_attempted % 100 == 0:
            print(f"Processed {paths_attempted}/{len(osm_gdf)} OSM lines...")

        # Skip if no nearest points found
        if osm_id not in nearest_points:
            paths_rejected['no_endpoints'] += 1
            results.append({
                'osm_id': osm_id,
                'osm_name': str(osm_row.get('name', '')),
                'osm_voltage': osm_voltage,
                'matched': False,
                'match_quality': 'No Endpoint Matches'
            })
            continue

        # Get parallel circuit group if applicable
        group_key = osm_to_group.get(osm_id)
        group_used_pypsa.setdefault(group_key, set())

        # Get nearest endpoint nodes
        np_info = nearest_points[osm_id]
        start_nodes = np_info.get('start_nodes', [])
        end_nodes = np_info.get('end_nodes', [])

        if not start_nodes or not end_nodes:
            paths_rejected['no_endpoints'] += 1
            results.append({
                'osm_id': osm_id,
                'osm_name': str(osm_row.get('name', '')),
                'osm_voltage': osm_voltage,
                'matched': False,
                'match_quality': 'Insufficient Endpoint Matches'
            })
            continue

        # Try to find paths between start and end nodes
        candidates = []

        # Log the number of endpoints we're trying
        print(f"\nTrying to find paths for OSM ID {osm_id} - Length: {osm_length:.2f}km")
        print(f"  Start nodes: {len(start_nodes)}, End nodes: {len(end_nodes)}")

        # Try up to 5 start nodes and 5 end nodes
        total_path_attempts = 0
        successful_path_attempts = 0
        failed_path_attempts = 0

        for start_info in start_nodes[:5]:
            start_node = start_info['node_id']

            for end_info in end_nodes[:5]:
                end_node = end_info['node_id']
                total_path_attempts += 1

                if start_node == end_node:
                    continue

                # Find paths between nodes, limited to 3 segments for 110kV (usually shorter paths)
                potential_paths = find_paths(G, start_node, end_node, osm_length, max_segments=5)

                if not potential_paths:
                    failed_path_attempts += 1
                    continue

                successful_path_attempts += 1

                for path in potential_paths:
                    # Extract PyPSA line IDs from the path
                    pypsa_ids = []
                    path_length = 0

                    for i in range(len(path) - 1):
                        u, v = path[i], path[i + 1]
                        edge_data = G[u][v]

                        # Skip edges with no ID
                        if 'id' not in edge_data:
                            continue

                        pypsa_id = edge_data['id']

                        # Check circuit constraints
                        if pypsa_usage.get(pypsa_id, 0) >= G[u][v].get('circuits', 1):
                            # This PyPSA line is already at capacity
                            paths_rejected['circuits'] += 1
                            continue

                        # Skip if already used by another line in the same parallel group
                        if group_key and pypsa_id in group_used_pypsa[group_key]:
                            paths_rejected['circuits'] += 1
                            continue

                        # Voltage check - both should be 110kV for this matcher
                        edge_voltage = edge_data.get('voltage', 0)
                        if not (osm_voltage == 110 and edge_voltage == 110):
                            paths_rejected['voltage'] += 1
                            continue

                        pypsa_ids.append(pypsa_id)
                        path_length += edge_data.get('length', 0)

                    # Skip if no valid PyPSA lines in the path
                    if not pypsa_ids:
                        continue

                    # Calculate length ratio
                    length_ratio = path_length / osm_length if osm_length > 0 else 0

                    # 110kV lines can have more variation in length due to local routing
                    if not (0.5 <= length_ratio <= 2.0):
                        paths_rejected['length_ratio'] += 1
                        continue

                    # Calculate electrical similarity
                    electrical_score = 0.5  # Default

                    if len(pypsa_ids) == 1:
                        pypsa_rows = pypsa_gdf[pypsa_gdf['id'] == pypsa_ids[0]]
                        if not pypsa_rows.empty:
                            electrical_score = calculate_electrical_similarity(osm_row, pypsa_rows.iloc[0])
                    else:
                        # For multi-segment paths, use weighted average
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

                            # Compare with OSM parameters
                            osm_r = _num(osm_row.get('r_per_km'))
                            osm_x = _num(osm_row.get('x_per_km'))

                            r_score = 0.5
                            x_score = 0.5

                            if osm_r is not None and avg_r > 0:
                                r_error = abs(osm_r - avg_r) / osm_r
                                r_score = max(0, 1 - min(r_error, 1))

                            if osm_x is not None and avg_x > 0:
                                x_error = abs(osm_x - avg_x) / osm_x
                                x_score = max(0, 1 - min(x_error, 1))

                            electrical_score = (r_score + x_score) / 2

                    # Calculate geometric similarity using buffer overlap
                    osm_buffer = osm_row.geometry.buffer(0.005)  # ~500m buffer for 110kV
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
                            merged_buffer = merged_geom.buffer(0.005)
                            intersection = osm_buffer.intersection(merged_buffer)

                            if hasattr(intersection, 'area') and osm_buffer.area > 0:
                                geometric_score = min(1.0, intersection.area / osm_buffer.area)
                        except Exception as e:
                            print(f"    Error calculating geometric score: {e}")
                            pass

                    # Calculate direction similarity
                    direction_score = 0.5  # Default
                    if 'direction_vector' in np_info and np_info['direction_vector'] is not None:
                        # Extract path endpoints
                        path_start = G.nodes[path[0]]
                        path_end = G.nodes[path[-1]]

                        # Calculate path direction vector
                        path_vector = np.array([path_end['x'] - path_start['x'], path_end['y'] - path_start['y']])
                        path_vector_len = np.linalg.norm(path_vector)

                        if path_vector_len > 0:
                            path_vector = path_vector / path_vector_len
                            direction_score = abs(np.dot(np_info['direction_vector'], path_vector))

                    # Combined score
                    length_score = 1.0 - min(1.0, abs(length_ratio - 1.0))
                    # Combined score - weighted for 110kV network characteristics
                    combined_score = (
                            0.20 * electrical_score +  # Less weight on electrical params for 110kV
                            0.40 * geometric_score +  # More weight on geometry (routing is key)
                            0.25 * length_score +  # Length is important
                            0.15 * direction_score  # Direction matters but less so for local routing
                    )

                    candidates.append({
                        'pypsa_ids': pypsa_ids,
                        'path_length': path_length,
                        'length_ratio': length_ratio,
                        'electrical_score': electrical_score,
                        'geometric_score': geometric_score,
                        'direction_score': direction_score,
                        'score': combined_score,
                        'segments': len(pypsa_ids)
                    })

        print(f"  Path finding: {successful_path_attempts} successful, {failed_path_attempts} failed out of {total_path_attempts} attempts")

        # If no candidates found, mark as unmatched
        if not candidates:
            if not start_nodes or not end_nodes:
                paths_rejected['no_endpoints'] += 1
            elif failed_path_attempts > 0:
                paths_rejected['path_not_found'] += 1
            else:
                paths_rejected['other'] += 1

            print(f"  No valid candidates found for OSM ID {osm_id}")

            results.append({
                'osm_id': osm_id,
                'osm_name': str(osm_row.get('name', '')),
                'osm_voltage': osm_voltage,
                'matched': False,
                'match_quality': 'No Valid Paths Found'
            })
            continue

        # Found candidates
        paths_found += 1
        candidates_counts.append(len(candidates))
        print(f"  Found {len(candidates)} candidate paths for OSM ID {osm_id}")

        # Sort candidates by score
        candidates.sort(key=lambda x: x['score'], reverse=True)
        best_candidate = candidates[0]
        print(f"  Best candidate: score={best_candidate['score']:.3f}, length_ratio={best_candidate['length_ratio']:.2f}, segments={best_candidate['segments']}")

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
        if osm_row.get('is_parallel_circuit', False):
            quality = f"Parallel Circuit - {quality}"

        # Create result
        result = {
            'osm_id': osm_id,
            'osm_name': str(osm_row.get('name', '')),
            'osm_voltage': osm_voltage,
            'matched': True,
            'pypsa_ids': best_candidate['pypsa_ids'],
            'path_length': best_candidate['path_length'],
            'osm_length': osm_length,
            'length_ratio': best_candidate['length_ratio'],
            'match_score': score,
            'electrical_score': best_candidate['electrical_score'],
            'geometric_score': best_candidate['geometric_score'],
            'direction_score': best_candidate['direction_score'],
            'match_quality': quality,
            'is_parallel_circuit': osm_row.get('is_parallel_circuit', False),
            'parallel_group': osm_row.get('parallel_group'),
            'segments': segments
        }

        # Update PyPSA line usage
        for pypsa_id in best_candidate['pypsa_ids']:
            pypsa_usage[pypsa_id] = pypsa_usage.get(pypsa_id, 0) + 1

            # Mark as used by this parallel group
            if group_key:
                group_used_pypsa[group_key].add(pypsa_id)

        results.append(result)

    # Print matching statistics
    print(f"\n===== PATH-BASED MATCHING STATISTICS =====")
    print(f"OSM lines processed: {paths_attempted}/{len(osm_gdf)}")
    print(f"OSM lines with path candidates: {paths_found}/{len(osm_gdf)} ({paths_found/len(osm_gdf)*100:.1f}%)")

    if candidates_counts:
        avg_candidates = sum(candidates_counts) / len(candidates_counts)
        print(f"Average candidates per OSM line with paths: {avg_candidates:.1f}")

    print(f"Rejection reasons:")
    for reason, count in paths_rejected.items():
        print(f"  - {reason}: {count} ({count/len(osm_gdf)*100:.1f}%)")

    # Count matched lines
    matched_count = sum(1 for r in results if r.get('matched', False))
    print(f"Total matched lines: {matched_count}/{len(osm_gdf)} ({matched_count/len(osm_gdf)*100:.1f}%)")

    return results


def find_paths(G, start_node, end_node, osm_length, max_segments=5):
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

            if dist_to_start < osm_length * 0.4:  # Within 40% of total line length
                start_area.append(node_id)

        if 'x' in end_node_data and 'y' in end_node_data:
            dist_to_end = math.sqrt(
                (data['x'] - end_node_data['x']) ** 2 +
                (data['y'] - end_node_data['y']) ** 2
            ) * 111

            if dist_to_end < osm_length * 0.4:
                end_area.append(node_id)

    # For each potential intermediate node near the middle
    middle_nodes = set(start_area).intersection(set(end_area))

    # If too many middle nodes, take a sample
    if len(middle_nodes) > 25:
        middle_nodes = list(middle_nodes)[:25]

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

            # Only consider if length is reasonable (bit looser for 110kV)
            if 0.6 <= (path_length / osm_length) <= 1.4:
                candidate_paths.append((combined_path, path_length))

        except nx.NetworkXNoPath:
            continue

    # Sort by length similarity to OSM line
    candidate_paths.sort(key=lambda x: abs(x[1] - osm_length))

    # Return the paths
    return [path for path, _ in candidate_paths[:5]]


def calculate_electrical_similarity(osm_row, pypsa_row):
    """Calculate electrical parameter similarity between OSM and PyPSA lines."""
    similarity = 0.5  # Default neutral score

    # Extract per-km electrical parameters
    osm_r = _num(osm_row.get('r_per_km'))
    osm_x = _num(osm_row.get('x_per_km'))
    osm_b = _num(osm_row.get('b_per_km'))

    pypsa_r = _num(pypsa_row.get('r_per_km'))
    pypsa_x = _num(pypsa_row.get('x_per_km'))
    pypsa_b = _num(pypsa_row.get('b_per_km'))

    # If no parameters available, return default
    if osm_r is None and osm_x is None and osm_b is None:
        return similarity

    if pypsa_r is None and pypsa_x is None and pypsa_b is None:
        return similarity

    # Calculate relative errors for each parameter
    errors = []

    if osm_r is not None and pypsa_r is not None and osm_r > 0 and pypsa_r > 0:
        r_err = abs(osm_r - pypsa_r) / osm_r
        errors.append(min(r_err, 1.0))  # Cap at 100% error

    if osm_x is not None and pypsa_x is not None and osm_x > 0 and pypsa_x > 0:
        x_err = abs(osm_x - pypsa_x) / osm_x
        errors.append(min(x_err, 1.0))

    if osm_b is not None and pypsa_b is not None and osm_b > 0 and pypsa_b > 0:
        b_err = abs(osm_b - pypsa_b) / osm_b
        errors.append(min(b_err, 1.0))

    # If no comparable parameters, return default
    if not errors:
        return similarity

    # Calculate average similarity (1 - average error)
    avg_err = sum(errors) / len(errors)
    return max(0.0, 1.0 - avg_err)


def run_osm_pypsa_matching(osm_gdf, pypsa_gdf, output_dir, include_dc_links=False,
                           skip_visual_matching=False, verbose=True,
                           ignore_voltage_differences=True):
    """
    Match OpenStreetMap and PyPSA 110kV transmission lines using multiple geometric and topological approaches.

    This function implements a multi-stage matching algorithm that combines endpoint matching,
    path-based matching, parallel circuit detection, and visual similarity analysis to identify
    corresponding transmission lines between OSM and PyPSA datasets.

    Parameters
    ----------
    osm_gdf : GeoDataFrame
        OpenStreetMap transmission line data with geometry and attributes
    pypsa_gdf : GeoDataFrame
        PyPSA 110kV transmission line data with geometry and attributes
    output_dir : str or Path
        Directory where output files (CSV, HTML visualization) will be saved
    include_dc_links : bool, optional
        Whether to include DC links in the matching process (default: False)
    skip_visual_matching : bool, optional
        Whether to skip the computationally expensive visual matching step (default: False)
    verbose : bool, optional
        Whether to print detailed progress information (default: True)
    ignore_voltage_differences : bool, optional
        Whether to ignore voltage level differences during matching (default: False)

    Returns
    -------
    list
        List of dictionaries containing match information
    """
    import os
    import traceback
    import pandas as pd
    import numpy as np
    import networkx as nx
    from pathlib import Path
    from shapely.geometry import box

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # --------------------------------------------------------------------------
    # STAGE 1: DATA VALIDATION AND PREPARATION
    # --------------------------------------------------------------------------
    print("\n===== DATA VALIDATION =====")

    # Handle missing input data
    if osm_gdf is None or pypsa_gdf is None:
        missing = "OSM" if osm_gdf is None else "PyPSA"
        print(f"ERROR: {missing} data is None, cannot proceed with matching")
        results_file = os.path.join(output_dir, 'osm_pypsa_matches.csv')
        pd.DataFrame(columns=['osm_id', 'pypsa_id', 'matched', 'match_quality']).to_csv(results_file, index=False)
        print(f"Created empty results file: {results_file}")
        return []

    # Report data sizes
    print(f"OSM data: {len(osm_gdf)} lines")
    print(f"PyPSA data: {len(pypsa_gdf)} lines")

    # Ensure consistent CRS between datasets
    print(f"Original OSM CRS: {osm_gdf.crs}")
    print(f"Original PyPSA CRS: {pypsa_gdf.crs}")

    # Handle missing CRS
    osm_gdf = osm_gdf.copy()
    pypsa_gdf = pypsa_gdf.copy()

    if osm_gdf.crs is None:
        print("WARNING: OSM data has no CRS defined, setting to EPSG:4326")
        osm_gdf.set_crs(epsg=4326, inplace=True)

    if pypsa_gdf.crs is None:
        print("WARNING: PyPSA data has no CRS defined, setting to EPSG:4326")
        pypsa_gdf.set_crs(epsg=4326, inplace=True)

    # Ensure both datasets use same CRS
    if osm_gdf.crs != pypsa_gdf.crs:
        print(f"WARNING: CRS mismatch - converting OSM data to match PyPSA CRS {pypsa_gdf.crs}")
        osm_gdf = osm_gdf.to_crs(pypsa_gdf.crs)
        print(f"After conversion - OSM CRS: {osm_gdf.crs}, PyPSA CRS: {pypsa_gdf.crs}")

    # Handle NULL geometries
    osm_null_geoms = osm_gdf.geometry.isna().sum()
    pypsa_null_geoms = pypsa_gdf.geometry.isna().sum()

    if osm_null_geoms > 0 or pypsa_null_geoms > 0:
        if osm_null_geoms > 0:
            print(f"WARNING: {osm_null_geoms} OSM lines have NULL geometries")
            osm_gdf = osm_gdf[osm_gdf.geometry.notna()].copy()
            print(f"Filtered OSM data to {len(osm_gdf)} lines with non-null geometries")

        if pypsa_null_geoms > 0:
            print(f"WARNING: {pypsa_null_geoms} PyPSA lines have NULL geometries")
            pypsa_gdf = pypsa_gdf[pypsa_gdf.geometry.notna()].copy()
            print(f"Filtered PyPSA data to {len(pypsa_gdf)} lines with non-null geometries")

    # Stop if either dataset is empty
    if len(osm_gdf) == 0 or len(pypsa_gdf) == 0:
        missing = "OSM" if len(osm_gdf) == 0 else "PyPSA"
        print(f"ERROR: No {missing} lines with valid geometries, cannot proceed with matching")
        results_file = os.path.join(output_dir, 'osm_pypsa_matches.csv')
        pd.DataFrame(columns=['osm_id', 'pypsa_id', 'matched', 'match_quality']).to_csv(results_file, index=False)
        print(f"Created empty results file: {results_file}")
        return []

    # Repair invalid geometries
    osm_invalid_geoms = sum(1 for g in osm_gdf.geometry if not g.is_valid)
    pypsa_invalid_geoms = sum(1 for g in pypsa_gdf.geometry if not g.is_valid)

    if osm_invalid_geoms > 0:
        print(f"WARNING: {osm_invalid_geoms} OSM lines have invalid geometries, attempting to fix...")
        osm_gdf['geometry'] = osm_gdf.geometry.apply(lambda g: g.buffer(0) if not g.is_valid else g)
        osm_invalid_after = sum(1 for g in osm_gdf.geometry if not g.is_valid)
        print(f"After fixing: {osm_invalid_after} OSM lines still have invalid geometries")

    if pypsa_invalid_geoms > 0:
        print(f"WARNING: {pypsa_invalid_geoms} PyPSA lines have invalid geometries, attempting to fix...")
        pypsa_gdf['geometry'] = pypsa_gdf.geometry.apply(lambda g: g.buffer(0) if not g.is_valid else g)
        pypsa_invalid_after = sum(1 for g in pypsa_gdf.geometry if not g.is_valid)
        print(f"After fixing: {pypsa_invalid_after} PyPSA lines still have invalid geometries")

    # Display geometry type distribution
    osm_types = osm_gdf.geometry.type.value_counts()
    pypsa_types = pypsa_gdf.geometry.type.value_counts()
    print(f"OSM geometry types: {dict(osm_types)}")
    print(f"PyPSA geometry types: {dict(pypsa_types)}")

    # Check geographic overlap
    try:
        osm_bounds = osm_gdf.total_bounds
        pypsa_bounds = pypsa_gdf.total_bounds

        if np.any(np.isnan(osm_bounds)) or np.any(np.isnan(pypsa_bounds)):
            print("WARNING: NaN values detected in data bounds")
        else:
            print(f"OSM bounds: {osm_bounds}")
            print(f"PyPSA bounds: {pypsa_bounds}")

            # Calculate overlap
            osm_min_x, osm_min_y, osm_max_x, osm_max_y = osm_bounds
            pypsa_min_x, pypsa_min_y, pypsa_max_x, pypsa_max_y = pypsa_bounds

            overlap_x = osm_min_x <= pypsa_max_x and pypsa_min_x <= osm_max_x
            overlap_y = osm_min_y <= pypsa_max_y and pypsa_min_y <= osm_max_y

            if not (overlap_x and overlap_y):
                print("WARNING: OSM and PyPSA data do not overlap geographically")
    except Exception as e:
        print(f"WARNING: Error checking bounds: {e}")

    # Check voltage distribution
    try:
        osm_voltages = osm_gdf['voltage'].value_counts() if 'voltage' in osm_gdf.columns else "Not available"
        pypsa_voltages = pypsa_gdf['voltage'].value_counts() if 'voltage' in pypsa_gdf.columns else "Not available"
        print(f"OSM voltages: {osm_voltages}")
        print(f"PyPSA voltages: {pypsa_voltages}")

        if ignore_voltage_differences:
            print("NOTE: Ignoring voltage differences during matching (all voltage levels will be considered)")
    except Exception as e:
        print(f"WARNING: Error checking voltage distribution: {e}")

    # --------------------------------------------------------------------------
    # STAGE 2: NETWORK GRAPH CONSTRUCTION
    # --------------------------------------------------------------------------
    print("\n===== BUILDING NETWORK GRAPH =====")
    try:
        G = build_network_graph(pypsa_gdf)
        print(f"Graph nodes: {len(G.nodes)}")
        print(f"Graph edges: {len(G.edges)}")

        # Validate graph by checking sample nodes and edges
        if len(G.nodes) > 0:
            sample_node = list(G.nodes)[0]
            print(f"Sample node data: {G.nodes[sample_node]}")

        if len(G.edges) > 0:
            sample_edge = list(G.edges)[0]
            print(f"Sample edge data: {G[sample_edge[0]][sample_edge[1]]}")
    except Exception as e:
        print(f"ERROR building network graph: {e}")
        print(traceback.format_exc())

        # Create empty fallback graph
        G = nx.Graph()
        print("Created empty graph as fallback")

    # --------------------------------------------------------------------------
    # STAGE 3: PARALLEL CIRCUIT ANALYSIS
    # --------------------------------------------------------------------------
    print("\n===== ANALYZING PARALLEL CIRCUITS =====")
    try:
        parallel_groups, osm_to_group = identify_duplicate_geometries(osm_gdf)
        print(f"OSM parallel groups: {len(parallel_groups)}")

        pypsa_parallel_groups = identify_parallel_pypsa_circuits(pypsa_gdf)
        print(f"PyPSA parallel groups: {len(pypsa_parallel_groups)}")
    except Exception as e:
        print(f"ERROR analyzing parallel circuits: {e}")
        print(traceback.format_exc())

        # Empty fallback data
        parallel_groups = {}
        osm_to_group = {}
        pypsa_parallel_groups = {}

    # --------------------------------------------------------------------------
    # STAGE 4: ENDPOINT MATCHING
    # --------------------------------------------------------------------------
    print("\n===== FINDING NEAREST ENDPOINTS =====")
    try:
        # Use 2km distance threshold initially
        print("Using max distance of 2km for endpoint matching")
        # Remove ignore_voltage_differences parameter if it causes issues
        nearest_points = find_nearest_endpoints(osm_gdf, pypsa_gdf, G, max_dist_km=2)

        if nearest_points:
            print(f"OSM lines with endpoint matches: {len(nearest_points)}/{len(osm_gdf)}")

            # Calculate endpoint match statistics
            start_nodes_count = sum(1 for v in nearest_points.values() if v.get('start_nodes', []))
            end_nodes_count = sum(1 for v in nearest_points.values() if v.get('end_nodes', []))
            both_endpoints = sum(1 for v in nearest_points.values()
                                 if v.get('start_nodes', []) and v.get('end_nodes', []))

            print(f"OSM lines with start nodes: {start_nodes_count}")
            print(f"OSM lines with end nodes: {end_nodes_count}")
            print(f"OSM lines with both start and end nodes: {both_endpoints}")

            # Show sample data for debugging
            if nearest_points:
                sample_osm_id = list(nearest_points.keys())[0]
                sample_data = nearest_points[sample_osm_id]
                print(f"\nSample nearest points for OSM ID {sample_osm_id}:")
                print(f"  Start nodes: {len(sample_data.get('start_nodes', []))}")
                print(f"  End nodes: {len(sample_data.get('end_nodes', []))}")
                if sample_data.get('start_nodes'):
                    print(f"  First start node: {sample_data['start_nodes'][0]}")
                if sample_data.get('end_nodes'):
                    print(f"  First end node: {sample_data['end_nodes'][0]}")

            # Try with larger buffer if insufficient matches
            if both_endpoints < len(osm_gdf) * 0.5:
                print("\nNot enough endpoint matches found with 2km buffer, trying with 5km buffer...")
                nearest_points = find_nearest_endpoints(osm_gdf, pypsa_gdf, G, max_dist_km=5)

                # Recalculate statistics
                start_nodes_count = sum(1 for v in nearest_points.values() if v.get('start_nodes', []))
                end_nodes_count = sum(1 for v in nearest_points.values() if v.get('end_nodes', []))
                both_endpoints = sum(1 for v in nearest_points.values()
                                     if v.get('start_nodes', []) and v.get('end_nodes', []))

                print(f"With 5km buffer:")
                print(f"OSM lines with both start and end nodes: {both_endpoints}/{len(osm_gdf)}")
        else:
            print("WARNING: No nearest points found")
    except Exception as e:
        print(f"ERROR finding nearest endpoints: {e}")
        print(traceback.format_exc())

        # Empty fallback data
        nearest_points = {}

    # --------------------------------------------------------------------------
    # STAGE 5: MULTI-STAGE MATCHING PROCESS
    # --------------------------------------------------------------------------
    # Initialize results container
    all_matches = []

    # Step 1: Path-based matching
    print("\n===== PATH-BASED MATCHING FOR OSM-PYPSA 110KV LINES =====")
    try:
        # Remove ignore_voltage_differences parameter
        path_matches = path_based_osm_matching(
            osm_gdf, pypsa_gdf, G, nearest_points, parallel_groups, osm_to_group
        )

        # Ensure compatibility with existing functions by adding jao_id field
        for match in path_matches:
            if 'osm_id' in match and 'jao_id' not in match:
                match['jao_id'] = match['osm_id']

        matched_count = sum(1 for m in path_matches if m.get('matched', False))
        print(f"Path-based matching found {matched_count}/{len(path_matches)} matches")
        all_matches.extend(path_matches)
    except Exception as e:
        print(f"ERROR in path-based matching: {e}")
        print(traceback.format_exc())
        # Continue with empty matches

    # Step 2: Parallel circuit matching
    print("\n===== APPLYING PARALLEL CIRCUIT MATCHING =====")
    try:
        original_match_count = sum(1 for m in all_matches if m.get('matched', False))
        # Remove ignore_voltage_differences parameter
        all_matches = fix_parallel_circuit_matching(all_matches, osm_gdf, pypsa_gdf)
        new_match_count = sum(1 for m in all_matches if m.get('matched', False))
        print(f"Parallel circuit matching added {new_match_count - original_match_count} new matches")
    except Exception as e:
        print(f"ERROR in parallel circuit matching: {e}")
        print(traceback.format_exc())
        # Continue with existing matches

    # Step 3: Visual matching for unmatched lines
    print("\n===== FINDING VISUALLY MATCHING LINES =====")

    if skip_visual_matching:
        print("Skipping visual matching step as requested")
    else:
        try:
            # Get unmatched OSM IDs
            unmatched_ids = set(row['id'] for _, row in osm_gdf.iterrows() if 'id' in row) - set(
                m.get('osm_id', '') for m in all_matches if m.get('matched', False))
            print(f"Unmatched OSM lines before visual matching: {len(unmatched_ids)}")

            # Apply spatial filtering to reduce search space
            try:
                osm_bounds = osm_gdf.total_bounds
                osm_box = box(*osm_bounds)

                # Only consider PyPSA lines that intersect with the OSM area
                pypsa_filtered = pypsa_gdf[pypsa_gdf.intersects(osm_box)]
                print(f"Filtered PyPSA lines from {len(pypsa_gdf)} to {len(pypsa_filtered)} based on spatial bounds")

                # Use filtered data if it's not empty, otherwise use original
                if len(pypsa_filtered) > 0:
                    pypsa_for_matching = pypsa_filtered
                else:
                    print("Spatial filter removed all PyPSA lines, using original data")
                    pypsa_for_matching = pypsa_gdf
            except Exception as e:
                print(f"ERROR applying spatial filter: {e}")
                print(traceback.format_exc())
                pypsa_for_matching = pypsa_gdf

            # Define safe visual matcher function
            def safe_visual_matcher():
                try:
                    # Remove ignore_voltage_differences parameter
                    new_matches = find_visually_matching_lines(
                        osm_gdf, pypsa_for_matching, all_matches
                    )

                    # Handle case where function might return count instead of matches
                    if isinstance(new_matches, int):
                        print(f"Visual matcher returned a count: {new_matches}")
                        return all_matches

                    # Count new matches
                    new_match_count = sum(1 for m in new_matches
                                          if m.get('matched', False) and m.get('osm_id', '') in unmatched_ids)
                    print(f"Visual matching found {new_match_count} new matches")

                    return new_matches
                except Exception as e:
                    print(f"Error in visual matching: {e}")
                    print(traceback.format_exc())
                    return all_matches  # Return existing matches on error

            all_matches = safe_visual_matcher()
        except Exception as e:
            print(f"ERROR in visual matching outer block: {e}")
            print(traceback.format_exc())
            # Continue with existing matches

    # Step 4: Final aggressive buffer matching
    print("\n===== FINAL PASS FOR REMAINING VISUAL MATCHES =====")
    try:
        print("Using buffer of 1000 meters for remaining matches")

        # Count unmatched before final pass
        unmatched_before = len(set(row['id'] for _, row in osm_gdf.iterrows() if 'id' in row) -
                               set(m.get('osm_id', '') for m in all_matches if m.get('matched', False)))

        # Remove ignore_voltage_differences parameter
        all_matches = catch_remaining_visual_matches(
            all_matches, osm_gdf, pypsa_gdf, buffer_meters=1000  # Smaller buffer for 110kV
        )

        # Count unmatched after final pass
        unmatched_after = len(set(row['id'] for _, row in osm_gdf.iterrows() if 'id' in row) -
                              set(m.get('osm_id', '') for m in all_matches if m.get('matched', False)))

        print(f"Final visual pass added {unmatched_before - unmatched_after} new matches")
    except Exception as e:
        print(f"ERROR in final visual pass: {e}")
        print(traceback.format_exc())
        # Continue with existing matches

    # Step 5: Enhanced parallel circuit matching
    print("\n===== ENHANCED PARALLEL CIRCUIT MATCHING =====")
    try:
        matched_before = sum(1 for m in all_matches if m.get('matched', False))
        # Remove ignore_voltage_differences parameter
        all_matches = enhanced_parallel_circuit_matching(all_matches, osm_gdf, pypsa_gdf)
        matched_after = sum(1 for m in all_matches if m.get('matched', False))
        print(f"Enhanced parallel circuit matching added {matched_after - matched_before} new matches")
    except Exception as e:
        print(f"ERROR in enhanced parallel circuit matching: {e}")
        print(traceback.format_exc())
        # Continue with existing matches

    # Step 6: Extend short matches
    print("\n===== EXTENDING SHORT MATCHES =====")
    try:
        matched_before = sum(1 for m in all_matches if m.get('matched', False))
        # Remove ignore_voltage_differences parameter
        all_matches = auto_extend_short_matches(all_matches, osm_gdf, pypsa_gdf, target_ratio=0.75)
        matched_after = sum(1 for m in all_matches if m.get('matched', False))
        print(f"Short match extension added {matched_after - matched_before} new matches")
    except Exception as e:
        print(f"ERROR in extending short matches: {e}")
        print(traceback.format_exc())
        # Continue with existing matches

    # --------------------------------------------------------------------------
    # STAGE 6: RESULTS ANALYSIS AND EXPORT
    # --------------------------------------------------------------------------
    # Calculate final statistics
    total_osm = len(osm_gdf)
    matched_osm = sum(1 for m in all_matches if m.get('matched', False))
    match_percentage = matched_osm / total_osm * 100 if total_osm > 0 else 0

    print("\n===== OSM-PYPSA MATCHING RESULTS =====")
    print(f"Total OSM lines: {total_osm}")
    print(f"Matched OSM lines: {matched_osm}")
    print(f"Match percentage: {match_percentage:.1f}%")
    if ignore_voltage_differences:
        print(f"Note: Voltage differences were ignored during matching")

    # Summarize match quality
    qualities = {}
    for match in all_matches:
        if match.get('matched', False):
            quality = match.get('match_quality', 'Unknown')
            qualities[quality] = qualities.get(quality, 0) + 1

    print("\nMatch quality distribution:")
    for quality, count in sorted(qualities.items()):
        percentage = count / matched_osm * 100 if matched_osm > 0 else 0
        print(f"  {quality}: {count} ({percentage:.1f}%)")

    # Export results to CSV
    try:
        from grid_matcher.io.exporters import create_results_csv
        csv_file = os.path.join(output_dir, 'osm_pypsa_matches.csv')
        create_results_csv(all_matches, csv_file)
        print(f"\nResults saved to CSV: {csv_file}")
    except Exception as e:
        print(f"ERROR exporting results to CSV: {e}")
        # Try basic fallback CSV export
        try:
            results_df = pd.DataFrame(all_matches)
            results_df.to_csv(os.path.join(output_dir, 'osm_pypsa_matches_basic.csv'), index=False)
            print("Created basic results CSV as fallback")
        except Exception:
            print("Could not create any results CSV")

    # Create visualization
    try:
        from grid_matcher.visualization.maps import create_jao_pypsa_visualization
        map_file = os.path.join(output_dir, 'osm_pypsa_matches.html')
        create_jao_pypsa_visualization(osm_gdf, pypsa_gdf, all_matches, map_file)
        print(f"Visualization created: {map_file}")
    except Exception as e:
        print(f"Error creating visualization: {e}")

    # Generate enhanced summary table
    try:
        from grid_matcher.visualization.reports import create_enhanced_summary_table
        from pathlib import Path
        # Remove ignore_voltage_differences parameter
        summary_file = create_enhanced_summary_table(
            osm_gdf, pypsa_gdf, all_matches, output_dir=Path(output_dir)
        )
        print(f"Enhanced summary created: {summary_file}")
    except Exception as e:
        print(f"Error creating enhanced summary: {e}")

    print(f"All results saved to {output_dir}")
    return all_matches


def create_osm_pypsa_results_csv(matching_results, output_path):
    """
    Create a CSV file summarizing the OSM-PyPSA matching results.

    Parameters
    ----------
    matching_results : list
        List of matching result dictionaries
    output_path : str
        Path to output CSV file
    """
    results_data = []

    for result in matching_results:
        row = {
            'osm_id': result.get('osm_id', result.get('jao_id', '')),
            'matched': result.get('matched', False),
            'match_quality': result.get('match_quality', 'N/A'),
            'pypsa_ids': result.get('pypsa_ids', []),
        }

        # Format pypsa_ids as semicolon-separated string if it's a list
        if isinstance(row['pypsa_ids'], list):
            row['pypsa_ids'] = ';'.join(map(str, row['pypsa_ids']))

        # Add any other available fields
        for key in ['osm_name', 'osm_voltage', 'length_ratio', 'overall_score']:
            if key in result:
                row[key] = result[key]

        results_data.append(row)

    # Create DataFrame and save to CSV
    df = pd.DataFrame(results_data)

    # Reorder columns for clarity
    column_order = ['osm_id', 'matched', 'match_quality', 'pypsa_ids']
    other_columns = [col for col in df.columns if col not in column_order]
    df = df[column_order + other_columns]

    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Matching results saved to {output_path}")

    return output_path


def generate_pypsa_with_osm_parameters(matching_results, osm_gdf, pypsa_gdf, output_dir):
    """
    Generate PyPSA lines with OSM electrical parameters.

    Parameters
    ----------
    matching_results : list
        List of matching result dictionaries
    osm_gdf : GeoDataFrame
        OSM transmission line data
    pypsa_gdf : GeoDataFrame
        PyPSA transmission line data
    output_dir : str or Path
        Directory to write output files

    Returns
    -------
    tuple
        (match_count, pypsa_with_osm, output_files)
    """
    print("\n===== GENERATING PYPSA WITH OSM PARAMETERS =====")
    # Create a copy of PyPSA GDF for modification
    pypsa_enhanced = pypsa_gdf.copy()

    # Add OSM ID and parameter columns
    pypsa_enhanced['osm_id'] = None
    pypsa_enhanced['osm_r'] = None
    pypsa_enhanced['osm_x'] = None
    pypsa_enhanced['osm_b'] = None
    pypsa_enhanced['match_quality'] = None

    # Create lookup for OSM data
    osm_by_id = {str(row['id']): row for _, row in osm_gdf.iterrows()}
    print(f"Created lookup for {len(osm_by_id)} OSM lines")

    # Track which PyPSA lines get matched
    matched_count = 0
    matched_pypsa_ids = set()

    # Process matches
    print("Processing matches...")
    for match in matching_results:
        if not match.get('matched', False):
            continue

        osm_id = str(match.get('osm_id', match.get('jao_id', '')))
        osm_row = osm_by_id.get(osm_id)

        if osm_row is None:
            print(f"  Warning: OSM ID {osm_id} not found in OSM data")
            continue

        # Get PyPSA IDs for this match
        pypsa_ids = match.get('pypsa_ids', [])
        if isinstance(pypsa_ids, str):
            pypsa_ids = [pid.strip() for pid in pypsa_ids.split(';') if pid.strip()]

        for pypsa_id in pypsa_ids:
            # Find this PyPSA ID in the dataframe
            mask = pypsa_enhanced['id'].astype(str) == str(pypsa_id)
            if not any(mask):
                # Try line_id if id doesn't match
                mask = pypsa_enhanced['line_id'].astype(str) == str(pypsa_id)

            if any(mask):
                # Update this PyPSA line with OSM parameters
                pypsa_enhanced.loc[mask, 'osm_id'] = osm_id
                pypsa_enhanced.loc[mask, 'osm_r'] = osm_row.get('r', None)
                pypsa_enhanced.loc[mask, 'osm_x'] = osm_row.get('x', None)
                pypsa_enhanced.loc[mask, 'osm_b'] = osm_row.get('b', None)
                pypsa_enhanced.loc[mask, 'match_quality'] = match.get('match_quality', 'Matched')
                matched_count += 1
                matched_pypsa_ids.add(str(pypsa_id))
            else:
                print(f"  Warning: PyPSA ID {pypsa_id} not found in PyPSA data")

    # Log statistics
    print(f"Matched {matched_count} PyPSA lines with OSM parameters")
    print(f"Total unique PyPSA lines matched: {len(matched_pypsa_ids)}")

    # Save to CSV
    output_file = os.path.join(output_dir, 'pypsa_with_osm.csv')
    pypsa_enhanced.to_csv(output_file, index=False)
    print(f"Saved to CSV: {output_file}")

    # Save GeoJSON for visualization
    try:
        geojson_file = os.path.join(output_dir, 'pypsa_with_osm.geojson')
        pypsa_enhanced.to_file(geojson_file, driver='GeoJSON')
        output_files = [output_file, geojson_file]
        print(f"Saved to GeoJSON: {geojson_file}")
    except Exception as e:
        print(f"Warning: Could not save GeoJSON ({e})")
        output_files = [output_file]

    print(f"Generated PyPSA with OSM parameters - {matched_count} lines matched")
    return matched_count, pypsa_enhanced, output_files