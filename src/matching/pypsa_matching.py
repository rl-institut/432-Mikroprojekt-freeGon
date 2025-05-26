import logging
import pandas as pd
logger = logging.getLogger(__name__)

def identify_visual_overlaps(pypsa_lines, network_lines, max_distance=0.0001, max_hausdorff=0.001):
    """
    Identify visual overlaps between PyPSA-EUR and network lines.

    Parameters:
    - pypsa_lines: GeoDataFrame of PyPSA-EUR lines
    - network_lines: GeoDataFrame of network lines
    - max_distance: Maximum distance between lines to consider them overlapping
    - max_hausdorff: Maximum Hausdorff distance to consider lines visually similar

    Returns:
    - List of dictionaries with overlap information
    """
    logger.info("Identifying visual overlaps between PyPSA-EUR and network lines...")
    logger.info(f"Using distance threshold: {max_distance} degrees")
    logger.info(f"Using Hausdorff threshold: {max_hausdorff} degrees")

    # Ensure IDs are strings
    if 'id' in pypsa_lines.columns:
        pypsa_lines['id'] = pypsa_lines['id'].astype(str)
    if 'id' in network_lines.columns:
        network_lines['id'] = network_lines['id'].astype(str)

    overlapping_pairs = []

    # Try to create spatial index for efficiency
    try:
        network_sindex = network_lines.sindex
        use_index = True
        logger.info("Using spatial index for efficient searching")
    except:
        use_index = False
        logger.warning("Spatial index creation failed, using brute force approach")

    # Process each PyPSA-EUR line
    total_pypsa = len(pypsa_lines)
    for p_idx, pypsa_line in pypsa_lines.iterrows():
        if p_idx % 10 == 0:
            logger.info(f"Processing PyPSA-EUR line {p_idx + 1}/{total_pypsa}...")

        pypsa_id = str(pypsa_line.get('id', p_idx))

        if pypsa_line.geometry is None or pypsa_line.geometry.is_empty:
            continue

        # Get candidate network lines
        candidates = network_lines
        if use_index:
            # Use bounds with small buffer to find candidates
            bounds = pypsa_line.geometry.buffer(max_distance * 10).bounds
            candidate_idxs = list(network_sindex.intersection(bounds))
            candidates = network_lines.iloc[candidate_idxs]

        # Check each candidate for direct overlap
        for n_idx, net_line in candidates.iterrows():
            net_id = str(net_line.get('id', n_idx))

            if net_line.geometry is None or net_line.geometry.is_empty:
                continue

            # Use multiple checks for visual overlap
            try:
                # Check 1: Simple distance
                distance = float(pypsa_line.geometry.distance(net_line.geometry))

                # Check 2: Hausdorff distance (more robust for comparing line shapes)
                try:
                    hausdorff_dist = float(pypsa_line.geometry.hausdorff_distance(net_line.geometry))
                except:
                    hausdorff_dist = float('inf')

                # Check 3: Length of common section
                try:
                    intersection = pypsa_line.geometry.buffer(max_distance).intersection(net_line.geometry)
                    if intersection.is_empty:
                        intersection_length = 0.0
                    elif intersection.geom_type == 'LineString':
                        intersection_length = float(intersection.length)
                    elif intersection.geom_type == 'MultiLineString':
                        intersection_length = float(sum(line.length for line in intersection.geoms))
                    else:
                        intersection_length = 0.0

                    # Calculate overlap percentages
                    pypsa_length = float(pypsa_line.geometry.length)
                    net_length = float(net_line.geometry.length)

                    pypsa_overlap_pct = float(intersection_length / pypsa_length if pypsa_length > 0 else 0)
                    net_overlap_pct = float(intersection_length / net_length if net_length > 0 else 0)
                except:
                    intersection_length = 0.0
                    pypsa_overlap_pct = 0.0
                    net_overlap_pct = 0.0

                # Is this a visual overlap?
                is_overlap = False

                # Criterion 1: Very small distance
                if distance < max_distance:
                    is_overlap = True

                # Criterion 2: Small Hausdorff distance (similar shape)
                if hausdorff_dist < max_hausdorff:
                    is_overlap = True

                # Criterion 3: Significant overlap percentage
                if pypsa_overlap_pct > 0.9 or net_overlap_pct > 0.9:
                    is_overlap = True

                if is_overlap:
                    # Get voltage levels
                    pypsa_voltage = float(pypsa_line.get('v_nom', pypsa_line.get('voltage', 0)))
                    net_voltage = float(net_line.get('v_nom', 0))

                    match_data = {
                        'dlr_id': pypsa_id,  # Use dlr_id for compatibility
                        'network_id': net_id,
                        'source_length': pypsa_line.get('length', 0),
                        'network_length': net_line.get('length', 0),
                        'distance': distance,
                        'hausdorff_distance': hausdorff_dist,
                        'intersection_length': intersection_length * 111,  # Convert to approx km
                        'overlap_percentage': pypsa_overlap_pct,
                        'net_overlap_percentage': net_overlap_pct,
                        'direction_similarity': 1.0,  # Visual overlap implies good direction
                        'source_voltage': pypsa_voltage,
                        'network_voltage': net_voltage,
                        'source_r': pypsa_line.get('r', 0),
                        'source_x': pypsa_line.get('x', 0),
                        'source_b': pypsa_line.get('b', 0),
                        'network_r': net_line.get('r', 0),
                        'network_x': net_line.get('x', 0),
                        'network_b': net_line.get('b', 0),
                        'match_type': 'visual_overlap',
                        'dataset': 'PyPSA-EUR'
                    }

                    overlapping_pairs.append(match_data)
            except Exception as e:
                logger.warning(f"Error checking overlap between PyPSA {pypsa_id} and Network {net_id}: {e}")

    logger.info(f"Found {len(overlapping_pairs)} visually overlapping lines")
    return overlapping_pairs


def match_pypsa_eur_lines(pypsa_lines, network_lines, config=None):
    """
    Match PyPSA-EUR lines to network lines using a multi-stage approach.

    Parameters:
    - pypsa_lines: GeoDataFrame of PyPSA-EUR lines
    - network_lines: GeoDataFrame of network lines
    - config: Optional configuration dictionary with matching parameters

    Returns:
    - DataFrame of matched line pairs
    """
    logger.info("Matching PyPSA-EUR lines to network lines with multi-stage approach...")

    if pypsa_lines.empty or network_lines.empty:
        logger.warning("Empty input data, cannot perform matching")
        return pd.DataFrame()

    # Get configuration parameters or use defaults
    if config is None:
        config = {}

    max_distance = config.get('max_distance', 0.0001)
    max_hausdorff = config.get('max_hausdorff', 0.0005)
    relaxed_max_distance = config.get('relaxed_max_distance', 0.001)
    relaxed_max_hausdorff = config.get('relaxed_max_hausdorff', 0.005)

    # Stage 1: Try direct visual overlap matching
    logger.info("Stage 1: Identifying visual overlaps...")
    visual_matches = identify_visual_overlaps(
        pypsa_lines,
        network_lines,
        max_distance=max_distance,
        max_hausdorff=max_hausdorff
    )

    # Stage 2: For lines that didn't match, try with more relaxed parameters
    matched_pypsa_ids = {match['dlr_id'] for match in visual_matches}
    matched_network_ids = {match['network_id'] for match in visual_matches}

    # Filter unmatched lines
    unmatched_pypsa = pypsa_lines[~pypsa_lines['id'].astype(str).isin(matched_pypsa_ids)]
    unmatched_network = network_lines[~network_lines['id'].astype(str).isin(matched_network_ids)]

    logger.info(f"After Stage 1: {len(matched_pypsa_ids)} PyPSA-EUR lines matched")
    logger.info(f"Trying Stage 2 with {len(unmatched_pypsa)} unmatched PyPSA-EUR lines...")

    # More relaxed parameters for stage 2
    relaxed_matches = identify_visual_overlaps(
        unmatched_pypsa,
        unmatched_network,
        max_distance=relaxed_max_distance,
        max_hausdorff=relaxed_max_hausdorff
    )

    # Combine results from both stages
    all_matches = visual_matches + relaxed_matches
    logger.info(f"Total matched pairs after Stage 2: {len(all_matches)}")

    # Convert to DataFrame
    if all_matches:
        # Create final match DataFrame
        matches_df = pd.DataFrame(all_matches)

        # Calculate allocated parameters
        matches_df['allocated_r'] = matches_df.apply(
            lambda row: row['source_r'] * (row['network_length'] / row['source_length'])
            if row['source_length'] > 0 else 0,
            axis=1
        )

        matches_df['allocated_x'] = matches_df.apply(
            lambda row: row['source_x'] * (row['network_length'] / row['source_length'])
            if row['source_length'] > 0 else 0,
            axis=1
        )

        matches_df['allocated_b'] = matches_df.apply(
            lambda row: row['source_b'] * (row['network_length'] / row['source_length'])
            if row['source_length'] > 0 else 0,
            axis=1
        )

        # Add compatibility columns
        matches_df['dlr_r'] = matches_df['source_r']
        matches_df['dlr_x'] = matches_df['source_x']
        matches_df['dlr_b'] = matches_df['source_b']
        matches_df['dlr_length'] = matches_df['source_length']
        matches_df['dlr_voltage'] = matches_df['source_voltage']
        matches_df['allocated_v_nom'] = matches_df['network_voltage']

        # Calculate change percentages
        matches_df['r_change_pct'] = matches_df.apply(
            lambda row: ((row['allocated_r'] / row['network_r']) - 1) * 100
            if row['network_r'] != 0 else 0,
            axis=1
        )

        matches_df['x_change_pct'] = matches_df.apply(
            lambda row: ((row['allocated_x'] / row['network_x']) - 1) * 100
            if row['network_x'] != 0 else 0,
            axis=1
        )

        matches_df['b_change_pct'] = matches_df.apply(
            lambda row: ((row['allocated_b'] / row['network_b']) - 1) * 100
            if row['network_b'] != 0 else 0,
            axis=1
        )

        # Set a default TSO value
        matches_df['TSO'] = 'PyPSA-EUR'

        return matches_df
    else:
        logger.warning("No PyPSA-EUR matches found")
        return pd.DataFrame()