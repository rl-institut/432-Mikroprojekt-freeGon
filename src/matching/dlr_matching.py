import logging
import pandas as pd
import numpy as np
from geopandas import GeoDataFrame
from shapely.geometry import LineString, MultiLineString

# Import the utility functions
from src.matching.utils import direction_similarity, calculate_line_direction, get_geometry_coords, create_geometry_hash

logger = logging.getLogger(__name__)
import logging
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import LineString, MultiLineString
from typing import Dict, List, Tuple, Optional, Union, Any

from src.matching.utils import direction_similarity, calculate_line_direction

def match_lines_detailed(source_lines, network_lines, buffer_distance=0.020, snap_distance=0.009,
                         direction_threshold=0.65, enforce_voltage_matching=False, dataset_name="Source"):
    """
    Match lines using the sophisticated approach with direction similarity and detailed intersection analysis.
    Properly handles parallel circuits with identical geometries.

    Parameters:
    - source_lines: GeoDataFrame of source lines
    - network_lines: GeoDataFrame of network lines
    - buffer_distance: Distance in degrees to buffer around source lines for intersection checks
    - snap_distance: Maximum distance in degrees to consider lines for snapping
    - direction_threshold: Minimum direction similarity value (0-1) to consider lines parallel
    - enforce_voltage_matching: If True, only match lines with matching voltages
    - dataset_name: Name of the source dataset for logging

    Returns:
    - DataFrame of matched line pairs
    """
    logger.info(
        f"Matching {dataset_name} lines with buffer distance {buffer_distance}, "
        f"snap_distance {snap_distance}, and direction threshold {direction_threshold}...")

    matches = []
    match_count = 0

    # Remove rows with None geometry
    source_lines = source_lines[source_lines['geometry'].notna()].copy()
    network_lines = network_lines[network_lines['geometry'].notna()].copy()

    # Ensure source_lines has string IDs
    if 'id' in source_lines.columns:
        source_lines['id'] = source_lines['id'].astype(str)

    # Ensure network_lines has string IDs
    if 'id' in network_lines.columns:
        network_lines['id'] = network_lines['id'].astype(str)

    logger.info(
        f"Starting matching process with {len(source_lines)} {dataset_name} lines and {len(network_lines)} network lines")

    # Create spatial index for network lines
    if len(network_lines) > 0:
        try:
            logger.info("Creating spatial index for network lines...")
            network_sindex = network_lines.sindex
            logger.info("Spatial index created successfully")
        except Exception as e:
            logger.error(f"Error creating spatial index: {str(e)}. Using brute force approach.")
            network_sindex = None
    else:
        logger.warning("No valid network lines found.")
        return pd.DataFrame()

    # Create geometry hash for detecting parallel circuits
    logger.info(f"Indexing {dataset_name} lines to detect parallel circuits...")
    geom_hash_to_ids = create_geometry_hash(source_lines)

    # Log parallel circuits found
    parallel_circuits = [ids for ids in geom_hash_to_ids.values() if len(ids) > 1]
    if parallel_circuits:
        logger.info(f"Found {len(parallel_circuits)} sets of parallel circuits in {dataset_name} data")
        for i, circuit_set in enumerate(parallel_circuits[:5]):  # Log first 5 sets
            logger.info(f"  Parallel circuit set {i + 1}: {circuit_set}")
        if len(parallel_circuits) > 5:
            logger.info(f"  ...and {len(parallel_circuits) - 5} more sets")

    # Process each source line
    batch_size = max(1, len(source_lines) // 10)
    match_details = []

    # Keep track of which network lines have been matched
    network_matches = {}  # network_id -> {source_id, match_data}

    for idx, source_line in source_lines.iterrows():
        if idx % batch_size == 0:
            logger.info(f"Processing {dataset_name} line {idx + 1}/{len(source_lines)}...")

        if source_line.geometry is None or source_line.geometry.is_empty:
            logger.debug(f"Skipping {dataset_name} line {source_line.get('id', idx)} with empty geometry")
            continue

        try:
            # Get source line information
            source_id = str(source_line.get('id', idx))
            source_geom_type = source_line.geometry.geom_type
            source_length = source_line.get('length', source_line.geometry.length * 111)

            # Check if this is part of a parallel circuit
            for geom_hash, ids in geom_hash_to_ids.items():
                if source_id in ids and len(ids) > 1:
                    logger.debug(f"{dataset_name} line {source_id} is part of parallel circuit with {ids}")
                    break

            # Buffer the line
            buffered_line = source_line.geometry.buffer(buffer_distance)

            # Find candidate lines using spatial index or brute force
            if network_sindex is not None:
                possible_matches_idx = list(network_sindex.intersection(buffered_line.bounds))
                possible_matches = network_lines.iloc[possible_matches_idx]
                logger.debug(f"Found {len(possible_matches)} possible matches using spatial index")
            else:
                # Brute force - check all network lines
                possible_matches = network_lines
                logger.debug(f"Using brute force approach to check all {len(possible_matches)} network lines")

            # No possible matches found
            if len(possible_matches) == 0:
                logger.debug(f"No possible matches found for {dataset_name} line {source_id}")
                continue

            # Check each possible match for actual intersection
            matched_this_line = False
            for net_idx, net_line in possible_matches.iterrows():
                try:
                    if net_line.geometry is None or net_line.geometry.is_empty:
                        continue

                    net_id = str(net_line.get('id', net_idx))
                    net_geom_type = net_line.geometry.geom_type

                    # Check voltage matching if enforced
                    if enforce_voltage_matching and 'v_nom' in source_line and 'v_nom' in net_line:
                        source_voltage = source_line['v_nom']
                        net_voltage = net_line['v_nom']

                        # Skip if voltages don't match (allowing 380/400 kV equivalence)
                        if not (
                                (source_voltage == net_voltage) or
                                (source_voltage == 380 and net_voltage == 400) or
                                (source_voltage == 400 and net_voltage == 380)
                        ):
                            logger.debug(
                                f"Voltage mismatch: {dataset_name} {source_id} ({source_voltage} kV) vs Net {net_id} ({net_voltage} kV)")
                            continue

                    # Check spatial relationship
                    intersects_buffer = False

                    # Try different methods to determine intersection
                    try:
                        # Method 1: Direct intersection with buffer
                        if net_line.geometry.intersects(buffered_line):
                            intersects_buffer = True
                        # Method 2: Distance check
                        elif net_line.geometry.distance(source_line.geometry) <= snap_distance:
                            intersects_buffer = True
                    except Exception as e:
                        logger.warning(f"Error in intersection test: {str(e)}. Trying alternative method.")
                        # Fallback method: use bounds to estimate
                        net_bounds = net_line.geometry.bounds
                        source_bounds = source_line.geometry.bounds
                        if (
                                net_bounds[0] <= source_bounds[2] + buffer_distance and
                                net_bounds[2] >= source_bounds[0] - buffer_distance and
                                net_bounds[1] <= source_bounds[3] + buffer_distance and
                                net_bounds[3] >= source_bounds[1] - buffer_distance
                        ):
                            intersects_buffer = True

                    if not intersects_buffer:
                        continue

                    # Calculate direction similarity
                    dir_sim = direction_similarity(source_line.geometry, net_line.geometry)

                    # Skip if direction similarity is too low
                    if dir_sim < direction_threshold:
                        logger.debug(f"Direction similarity too low: {dir_sim:.2f} < {direction_threshold}")
                        continue

                    # Rest of the function remains the same...
                    # (Abbreviated for brevity in this example)

                    # Calculate intersection metrics
                    intersection_length = 0
                    overlap_percentage = 0
                    net_overlap_percentage = 0

                    # Different approach based on geometry types
                    if isinstance(net_line.geometry, MultiLineString):
                        # For MultiLineString, calculate intersection for each segment separately
                        for segment in net_line.geometry.geoms:
                            try:
                                seg_buffer = segment.buffer(buffer_distance)
                                if source_line.geometry.intersects(seg_buffer):
                                    seg_intersection = source_line.geometry.intersection(seg_buffer)

                                    if not seg_intersection.is_empty:
                                        if seg_intersection.geom_type == 'LineString':
                                            intersection_length += seg_intersection.length
                                        elif seg_intersection.geom_type == 'MultiLineString':
                                            intersection_length += sum(line.length for line in seg_intersection.geoms)
                            except Exception as e:
                                logger.warning(f"Error calculating intersection for segment: {str(e)}")
                    else:
                        # Standard intersection calculation for LineString
                        try:
                            intersection = source_line.geometry.intersection(net_line.geometry.buffer(buffer_distance))

                            if not intersection.is_empty:
                                if intersection.geom_type == 'LineString':
                                    intersection_length = intersection.length
                                elif intersection.geom_type == 'MultiLineString':
                                    intersection_length = sum(line.length for line in intersection.geoms)
                        except Exception as e:
                            logger.warning(f"Error calculating intersection: {str(e)}")
                            # Fallback: estimate based on distance
                            if net_line.geometry.distance(source_line.geometry) <= snap_distance:
                                intersection_length = min(source_line.geometry.length, net_line.geometry.length) * 0.5

                    # Skip if no meaningful intersection
                    if intersection_length <= 0:
                        continue

                    # Calculate overlap percentages
                    source_length = source_line.get('length', source_line.geometry.length * 111)
                    net_length = net_line.get('length', net_line.geometry.length * 111)

                    # Convert intersection_length from degrees to km (approximation)
                    intersection_length_km = intersection_length * 111

                    overlap_percentage = intersection_length_km / source_length if source_length > 0 else 0
                    net_overlap_percentage = intersection_length_km / net_length if net_length > 0 else 0

                    # Record detailed match information for debugging
                    match_details.append({
                        'source_id': source_id,
                        'network_id': net_id,
                        'source_type': source_geom_type,
                        'network_type': net_geom_type,
                        'direction_sim': dir_sim,
                        'intersection_length_km': intersection_length_km,
                        'overlap_pct': overlap_percentage,
                        'net_overlap_pct': net_overlap_percentage
                    })

                    # Prepare parameter values
                    source_r_per_km = source_line.get('r_per_km', 0)
                    source_x_per_km = source_line.get('x_per_km', 0)
                    source_b_per_km = source_line.get('b_per_km', 0)

                    # Use original values if per_km not available
                    if source_r_per_km == 0 and 'r' in source_line and source_length > 0:
                        source_r_per_km = source_line['r'] / source_length
                    if source_x_per_km == 0 and 'x' in source_line and source_length > 0:
                        source_x_per_km = source_line['x'] / source_length
                    if source_b_per_km == 0 and 'b' in source_line and source_length > 0:
                        source_b_per_km = source_line['b'] / source_length

                    # Calculate allocated parameters
                    allocated_r = source_r_per_km * net_length
                    allocated_x = source_x_per_km * net_length
                    allocated_b = source_b_per_km * net_length

                    # Handle voltage allocation
                    source_v_nom = source_line.get('v_nom', 0)
                    network_v_nom = net_line.get('v_nom', 0)
                    if network_v_nom == 380 and source_v_nom == 400:
                        allocated_v_nom = 400
                    else:
                        allocated_v_nom = network_v_nom

                    # Create match data
                    match_data = {
                        'dlr_id': source_id,  # Keep column name as dlr_id for compatibility
                        'network_id': net_id,
                        'source_length': source_length,
                        'network_length': net_length,
                        'intersection_length': intersection_length_km,
                        'overlap_percentage': overlap_percentage,
                        'net_overlap_percentage': net_overlap_percentage,
                        'direction_similarity': dir_sim,
                        # Source parameters
                        'source_r': source_line.get('r', 0),
                        'source_x': source_line.get('x', 0),
                        'source_b': source_line.get('b', 0),
                        'source_r_per_km': source_r_per_km,
                        'source_x_per_km': source_x_per_km,
                        'source_b_per_km': source_b_per_km,
                        # Network parameters
                        'network_r': net_line.get('r', 0),
                        'network_x': net_line.get('x', 0),
                        'network_b': net_line.get('b', 0),
                        'network_r_per_km': net_line.get('r_per_km', 0),
                        'network_x_per_km': net_line.get('x_per_km', 0),
                        'network_b_per_km': net_line.get('b_per_km', 0),
                        # Allocated parameters
                        'allocated_r': allocated_r,
                        'allocated_x': allocated_x,
                        'allocated_b': allocated_b,
                        # Voltage information
                        'source_voltage': source_v_nom,
                        'network_voltage': network_v_nom,
                        'allocated_v_nom': allocated_v_nom,
                        # Metadata
                        'TSO': source_line.get('TSO', 'Unknown'),
                        'match_type': 'intersection',
                        'dataset': dataset_name
                    }

                    # Add to matches list
                    matches.append(match_data)

                    # Track network matches
                    if net_id not in network_matches:
                        network_matches[net_id] = []
                    network_matches[net_id].append({
                        'source_id': source_id,
                        'overlap_percentage': overlap_percentage,
                        'match_data': match_data
                    })

                    match_count += 1
                    matched_this_line = True

                    logger.debug(
                        f"Match found: {dataset_name} {source_id} to Network {net_id} "
                        f"(dir_sim: {dir_sim:.2f}, overlap: {overlap_percentage:.2f})")

                except Exception as e:
                    logger.error(
                        f"Error processing potential match between {dataset_name} {source_id} and "
                        f"Network {net_line.get('id', net_idx)}: {str(e)}")

            if matched_this_line:
                logger.debug(f"Found matches for {dataset_name} line {source_id}")
            else:
                logger.debug(f"No matches found for {dataset_name} line {source_id}")

        except Exception as e:
            logger.error(f"Error processing {dataset_name} line {source_line.get('id', idx)}: {str(e)}")

    # Create DataFrame from matches
    if not matches:
        logger.warning(f"No matches found for {dataset_name} lines.")
        return pd.DataFrame()

    match_df = pd.DataFrame(matches)
    logger.info(f"Found {len(match_df)} potential matches before best-match selection")

    # Select best matches for each network line
    best_matches = []

    for net_id, matches_list in network_matches.items():
        # Sort matches by overlap percentage
        matches_list.sort(key=lambda x: x['overlap_percentage'], reverse=True)

        # Process parallel circuits specifically
        source_ids = [match['source_id'] for match in matches_list]
        parallel_matches = []

        # Check if any of these source lines are part of parallel circuits
        for geom_hash, parallel_ids in geom_hash_to_ids.items():
            # Find overlap between matched source IDs and parallel circuit IDs
            overlapping_ids = set(source_ids).intersection(set(parallel_ids))

            if len(overlapping_ids) > 0 and len(parallel_ids) > 1:
                logger.info(f"Processing parallel circuit match for network line {net_id}")
                logger.info(f"  Matched IDs: {overlapping_ids}")
                logger.info(f"  Full parallel circuit: {parallel_ids}")

                # If not all parallel lines are matched to this network line,
                # we need to handle the unmatched ones
                unmatched_parallel_ids = set(parallel_ids) - set(source_ids)

                if unmatched_parallel_ids:
                    logger.info(f"  Adding matches for unmatched parallel lines: {unmatched_parallel_ids}")

                    # Get the best match as template
                    best_match = matches_list[0]['match_data']

                    # For each unmatched parallel line, create a new match based on the best match
                    for unmatched_id in unmatched_parallel_ids:
                        # Get the unmatched source line
                        unmatched_line = source_lines[source_lines['id'] == unmatched_id]
                        if len(unmatched_line) == 0:
                            continue

                        unmatched_line = unmatched_line.iloc[0]

                        # Create a new match based on the best match
                        new_match = best_match.copy()
                        new_match['dlr_id'] = unmatched_id

                        # Update source parameters
                        new_match['source_r'] = unmatched_line.get('r', 0)
                        new_match['source_x'] = unmatched_line.get('x', 0)
                        new_match['source_b'] = unmatched_line.get('b', 0)
                        new_match['source_length'] = unmatched_line.get('length', 0)

                        # Recalculate per-km values
                        source_length = unmatched_line.get('length', unmatched_line.geometry.length * 111)
                        if source_length > 0:
                            new_match['source_r_per_km'] = unmatched_line.get('r', 0) / source_length
                            new_match['source_x_per_km'] = unmatched_line.get('x', 0) / source_length
                            new_match['source_b_per_km'] = unmatched_line.get('b', 0) / source_length

                        # Recalculate allocated parameters
                        net_length = new_match['network_length']
                        new_match['allocated_r'] = new_match['source_r_per_km'] * net_length
                        new_match['allocated_x'] = new_match['source_x_per_km'] * net_length
                        new_match['allocated_b'] = new_match['source_b_per_km'] * net_length

                        # Add to parallel matches
                        parallel_matches.append(new_match)

                # Add all the existing matches for this network line
                for match in matches_list:
                    best_matches.append(match['match_data'])

                # Add the new parallel matches
                best_matches.extend(parallel_matches)

                # Skip regular processing for this network line
                break
        else:
            # No parallel circuits found, just add the best match
            best_matches.append(matches_list[0]['match_data'])

    best_match_df = pd.DataFrame(best_matches)

    if len(best_match_df) == 0:
        logger.warning(f"No best matches found for {dataset_name} lines.")
        return pd.DataFrame()

    # Calculate change percentages
    if 'network_r' in best_match_df.columns and 'allocated_r' in best_match_df.columns:
        best_match_df['r_change_pct'] = np.where(
            best_match_df['network_r'] != 0,
            ((best_match_df['allocated_r'] / best_match_df['network_r']) - 1) * 100,
            0
        )

        best_match_df['x_change_pct'] = np.where(
            best_match_df['network_x'] != 0,
            ((best_match_df['allocated_x'] / best_match_df['network_x']) - 1) * 100,
            0
        )

        best_match_df['b_change_pct'] = np.where(
            best_match_df['network_b'] != 0,
            ((best_match_df['allocated_b'] / best_match_df['network_b']) - 1) * 100,
            0
        )

    # Add dlr_r and dlr_x columns for compatibility with the rest of the code
    if 'source_r' in best_match_df.columns and 'dlr_r' not in best_match_df.columns:
        best_match_df['dlr_r'] = best_match_df['source_r']
        best_match_df['dlr_x'] = best_match_df['source_x']
        best_match_df['dlr_b'] = best_match_df['source_b']
        best_match_df['dlr_length'] = best_match_df['source_length']
        best_match_df['dlr_voltage'] = best_match_df['source_voltage']

    logger.info(f"Final number of matches for {dataset_name}: {len(best_match_df)}")

    # Log information about parallel circuits that were matched
    parallel_match_count = len(best_match_df) - len(set(best_match_df['network_id']))
    if parallel_match_count > 0:
        logger.info(f"Matched {parallel_match_count} additional parallel circuits")

    return best_match_df