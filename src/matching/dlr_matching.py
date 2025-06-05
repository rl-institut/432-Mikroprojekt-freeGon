
import logging




logger = logging.getLogger(__name__)


from src.matching.utils import direction_similarity


# ---------------------------------------------------------------------------
def _segment_graph(gdf, snap_dist):
    """
    Build an undirected graph: each network segment is a node; an edge is
    added when the minimal distance between the two segments' end-points is
    ≤ snap_dist  (degrees).
    Returns a networkx.Graph.
    """
    g = nx.Graph()
    coords = []

    for idx, geom in zip(gdf.index, gdf.geometry):
        if geom is None or geom.is_empty:
            continue
        ends = []
        if geom.geom_type == "LineString":
            p0, p1 = Point(list(geom.coords)[0]), Point(list(geom.coords)[-1])
            ends.extend([p0, p1])
        elif geom.geom_type == "MultiLineString":
            for part in geom.geoms:
                p0, p1 = Point(list(part.coords)[0]), Point(list(part.coords)[-1])
                ends.extend([p0, p1])
        coords.append((idx, ends))
        g.add_node(idx)

    # naive O(n²) – fine for at most a few hundred candidates
    for (i, ends_i), (j, ends_j) in itertools.combinations(coords, 2):
        if any(e1.distance(e2) <= snap_dist for e1 in ends_i for e2 in ends_j):
            g.add_edge(i, j)

    return g
# ---------------------------------------------------------------------------


def detect_bus_connections(network_lines):
    """
    Detect connections between lines based on shared bus nodes.
    Returns a dictionary of line_id -> [connected_line_ids]
    """
    connections = {}

    # Create mapping of buses to lines
    bus_to_lines = {}

    for idx, line in network_lines.iterrows():
        line_id = str(line.get('id', idx))
        bus0 = str(line.get('bus0', ''))
        bus1 = str(line.get('bus1', ''))

        # Skip if missing bus information
        if not bus0 or not bus1:
            continue

        # Add to mapping
        if bus0 not in bus_to_lines:
            bus_to_lines[bus0] = []
        bus_to_lines[bus0].append(line_id)

        if bus1 not in bus_to_lines:
            bus_to_lines[bus1] = []
        bus_to_lines[bus1].append(line_id)

    # Find connections through shared buses
    for bus, lines in bus_to_lines.items():
        if len(lines) > 1:
            # These lines connect through this bus
            for line_id in lines:
                if line_id not in connections:
                    connections[line_id] = []

                # Add all other lines at this bus as connections
                for other_line in lines:
                    if other_line != line_id and other_line not in connections[line_id]:
                        connections[line_id].append(other_line)

    return connections


def get_all_endpoints(geometry):
    """
    Extract all endpoints from a geometry, handling both LineString and MultiLineString.
    Returns a list of Point objects.
    """
    endpoints = []

    try:
        if geometry is None or geometry.is_empty:
            return endpoints

        if geometry.geom_type == 'LineString':
            coords = list(geometry.coords)
            if len(coords) >= 2:
                endpoints.append(Point(coords[0]))
                endpoints.append(Point(coords[-1]))

        elif geometry.geom_type == 'MultiLineString':
            for line in geometry.geoms:
                coords = list(line.coords)
                if len(coords) >= 2:
                    endpoints.append(Point(coords[0]))
                    endpoints.append(Point(coords[-1]))
    except Exception as e:
        logger.warning(f"Error extracting endpoints: {str(e)}")

    return endpoints


def preprocess_network_segments(network_lines, max_gap=0.010, min_dir_similarity=0.8,
                                min_voltage_match=False, max_r_x_difference=0.5,
                                known_connections=None):
    """
    Preprocess network lines to identify and merge segments that form continuous lines.

    Parameters:
    - network_lines: GeoDataFrame of network lines
    - max_gap: Maximum distance (in degrees) between segments to consider them connectable
    - min_dir_similarity: Minimum direction similarity for segments to be considered aligned
    - min_voltage_match: Whether to require voltage matching for merging segments
    - max_r_x_difference: Maximum allowed difference in r_per_km and x_per_km values (as a ratio)
    - known_connections: Dictionary mapping line_id -> [connected_line_ids] for forcing connections

    Returns:
    - merged_network_lines: GeoDataFrame with merged segments
    - segment_to_merged_map: Dictionary mapping original segment IDs to merged line IDs
    """
    logger.info("Starting network segment preprocessing...")

    # Initialize known_connections if not provided
    if known_connections is None:
        known_connections = {}

    # Create a copy to avoid modifying the original
    merged_lines = network_lines.copy()

    # Dictionary to track which segments have been merged
    segment_to_merged_map = {}

    # Create a spatial index for efficient spatial queries
    try:
        sindex = merged_lines.sindex
        logger.info("Created spatial index for network lines")
    except Exception as e:
        logger.error(f"Error creating spatial index: {str(e)}. Skipping segment merging.")
        # Return original with an empty mapping
        return network_lines, {}

    # Create unique IDs for merged lines
    next_merged_id = 1

    # First pass - identify all potential segment connections
    segment_connections = {}

    logger.info("Identifying potential connections between line segments...")

    # Add connections from known_connections
    for line_id, conn_ids in known_connections.items():
        if line_id not in segment_connections:
            segment_connections[line_id] = []

        for conn_id in conn_ids:
            # Find the connected line
            conn_rows = merged_lines[merged_lines['id'].astype(str) == conn_id]
            if len(conn_rows) == 0:
                logger.warning(f"Known connection {conn_id} for line {line_id} not found")
                continue

            # Add as a forced connection with high direction similarity
            segment_connections[line_id].append({
                'id': conn_id,
                'direction_sim': 1.0,  # Assume perfect direction similarity for known connections
                'is_known_connection': True
            })

            logger.debug(f"Added known connection between {line_id} and {conn_id}")

    # Helper function to get endpoints from a geometry (handling both LineString and MultiLineString)
    def get_endpoints(geom):
        endpoints = []

        if geom is None or geom.is_empty:
            return endpoints

        if geom.geom_type == 'LineString':
            coords = list(geom.coords)
            if len(coords) >= 2:
                endpoints.append(Point(coords[0]))
                endpoints.append(Point(coords[-1]))
        elif geom.geom_type == 'MultiLineString':
            for line in geom.geoms:
                coords = list(line.coords)
                if len(coords) >= 2:
                    endpoints.append(Point(coords[0]))
                    endpoints.append(Point(coords[-1]))

        return endpoints

    # Process each line to identify connections based on geometry
    for idx, line in merged_lines.iterrows():
        if line.geometry is None or line.geometry.is_empty:
            continue

        line_id = str(line.get('id', idx))

        if line_id not in segment_connections:
            segment_connections[line_id] = []

        # Get all endpoints for this line (handles MultiLineString properly)
        line_endpoints = get_endpoints(line.geometry)

        if not line_endpoints:
            continue

        # Extract line properties
        line_voltage = line.get('v_nom', 0)
        line_r_per_km = line.get('r_per_km', 0)
        line_x_per_km = line.get('x_per_km', 0)

        # For each endpoint, look for nearby line endpoints
        for endpoint in line_endpoints:
            endpoint_buffer = endpoint.buffer(max_gap)

            # Find potential connections using the buffer
            possible_connections_idx = list(sindex.intersection(endpoint_buffer.bounds))

            # Check each potential connection
            for conn_idx in possible_connections_idx:
                if conn_idx == idx:  # Skip self
                    continue

                conn_line = merged_lines.iloc[conn_idx]
                conn_id = str(conn_line.get('id', conn_idx))

                # Skip if already in known connections
                if any(conn['id'] == conn_id for conn in segment_connections[line_id]):
                    continue

                # Get all endpoints for the potential connection
                conn_endpoints = get_endpoints(conn_line.geometry)

                if not conn_endpoints:
                    continue

                # Check if any endpoint of this line is near any endpoint of the connection line
                connects = False
                for conn_endpoint in conn_endpoints:
                    if endpoint.distance(conn_endpoint) <= max_gap:
                        connects = True
                        break

                if not connects:
                    continue

                # Check direction similarity
                dir_sim = direction_similarity(line.geometry, conn_line.geometry)

                # Special handling for end-to-start connections which might appear to go in opposite directions
                if dir_sim < min_dir_similarity:
                    # For now, just skip if direction similarity is too low
                    # This is a simplified version that avoids the complexity with reversing geometries
                    continue

                # Check voltage match if required
                if min_voltage_match:
                    conn_voltage = conn_line.get('v_nom', 0)

                    # Allow 380/400 kV equivalence
                    voltage_match = (line_voltage == conn_voltage) or \
                                    (line_voltage == 380 and conn_voltage == 400) or \
                                    (line_voltage == 400 and conn_voltage == 380)

                    if not voltage_match:
                        logger.debug(
                            f"Voltage mismatch between {line_id} ({line_voltage} kV) and {conn_id} ({conn_voltage} kV)")
                        continue

                # Check electrical parameter similarity if needed
                if max_r_x_difference < 1.0:
                    conn_r_per_km = conn_line.get('r_per_km', 0)
                    conn_x_per_km = conn_line.get('x_per_km', 0)

                    # Calculate ratios for non-zero values
                    r_ratio = (max(line_r_per_km, conn_r_per_km) / min(line_r_per_km, conn_r_per_km)
                               if min(line_r_per_km, conn_r_per_km) > 0 else 1.0)
                    x_ratio = (max(line_x_per_km, conn_x_per_km) / min(line_x_per_km, conn_x_per_km)
                               if min(line_x_per_km, conn_x_per_km) > 0 else 1.0)

                    if r_ratio > (1.0 + max_r_x_difference) or x_ratio > (1.0 + max_r_x_difference):
                        logger.debug(f"Electrical parameters too different between {line_id} and {conn_id}: "
                                     f"r_ratio={r_ratio:.2f}, x_ratio={x_ratio:.2f}")
                        continue

                # This is a valid connection
                segment_connections[line_id].append({
                    'id': conn_id,
                    'direction_sim': dir_sim
                })

                logger.debug(
                    f"Found connection between {line_id} and {conn_id} with direction similarity {dir_sim:.2f}")

    logger.info("Building chains of connected segments...")

    # Second pass - build chains of connected segments
    processed_segments = set()
    segments_to_merge = {}

    # Function to recursively build a chain of connected segments
    def build_connection_chain(start_id, current_chain=None):
        if current_chain is None:
            current_chain = []

        if start_id in current_chain:
            return current_chain

        current_chain.append(start_id)
        processed_segments.add(start_id)

        # Process all connections of this segment
        for connection in segment_connections.get(start_id, []):
            conn_id = connection['id']
            if conn_id not in processed_segments and conn_id not in current_chain:
                build_connection_chain(conn_id, current_chain)

        return current_chain

    # Find all chains of connected segments
    for segment_id in segment_connections:
        if segment_id in processed_segments:
            continue

        # Start a new chain with this segment
        chain = build_connection_chain(segment_id)

        # If we found a chain of multiple connected segments, create a merged group
        if len(chain) > 1:
            merged_id = f"merged_{next_merged_id}"
            next_merged_id += 1

            # Get segments in the chain
            chain_segments = []

            # Get representative values for the merged line
            chain_voltages = []
            chain_r_per_km = []
            chain_x_per_km = []
            chain_b_per_km = []

            for chain_segment_id in chain:
                # Find the segment row
                segment_rows = merged_lines[merged_lines['id'].astype(str) == chain_segment_id]

                if len(segment_rows) == 0:
                    logger.warning(f"Segment {chain_segment_id} in chain not found in network_lines")
                    continue

                segment_row = segment_rows.iloc[0]
                segment_length = segment_row.get('length', 0)
                if segment_length == 0 and segment_row.geometry:
                    # Calculate length if not available in the data
                    if segment_row.geometry.geom_type == 'LineString':
                        segment_length = segment_row.geometry.length * 111  # Convert to km
                    elif segment_row.geometry.geom_type == 'MultiLineString':
                        segment_length = sum(line.length for line in segment_row.geometry.geoms) * 111

                chain_segments.append({
                    'id': chain_segment_id,
                    'idx': segment_row.name,  # Get the actual index
                    'geometry': segment_row.geometry,
                    'length': segment_length
                })

                # Collect electrical parameters (only use non-zero values)
                if segment_row.get('v_nom', 0) > 0:
                    chain_voltages.append(segment_row.get('v_nom', 0))

                if segment_row.get('r_per_km', 0) > 0:
                    chain_r_per_km.append(segment_row.get('r_per_km', 0))

                if segment_row.get('x_per_km', 0) > 0:
                    chain_x_per_km.append(segment_row.get('x_per_km', 0))

                if segment_row.get('b_per_km', 0) > 0:
                    chain_b_per_km.append(segment_row.get('b_per_km', 0))

            # Skip if no valid segments found
            if not chain_segments:
                continue

            # Get the most common voltage
            voltage = max(set(chain_voltages), key=chain_voltages.count) if chain_voltages else 0

            # Get median values for electrical parameters (more robust than mean)
            r_per_km = np.median(chain_r_per_km) if chain_r_per_km else 0
            x_per_km = np.median(chain_x_per_km) if chain_x_per_km else 0
            b_per_km = np.median(chain_b_per_km) if chain_b_per_km else 0

            # Create the merged group
            segments_to_merge[merged_id] = {
                'segments': chain_segments,
                'voltage': voltage,
                'r_per_km': r_per_km,
                'x_per_km': x_per_km,
                'b_per_km': b_per_km
            }

            # Update the mapping
            for segment in chain_segments:
                segment_to_merged_map[segment['id']] = merged_id

            logger.info(
                f"Created merged line {merged_id} from {len(chain_segments)} segments: {[s['id'] for s in chain_segments]}")

    # Now create merged lines
    if segments_to_merge:
        logger.info(
            f"Creating {len(segments_to_merge)} merged lines from {len(segment_to_merged_map)} original segments")

        # Create a list for the new merged lines
        new_lines = []

        # Process each merged group
        for merged_id, group in segments_to_merge.items():
            try:
                # Extract segment geometries
                segment_geoms = []

                for segment in group['segments']:
                    if segment['geometry'] is None or segment['geometry'].is_empty:
                        continue

                    if segment['geometry'].geom_type == 'LineString':
                        segment_geoms.append(segment['geometry'])
                    elif segment['geometry'].geom_type == 'MultiLineString':
                        segment_geoms.extend(list(segment['geometry'].geoms))

                if not segment_geoms:
                    logger.warning(f"No valid geometries for merged line {merged_id}")
                    continue

                # Create MultiLineString from all segments
                merged_geom = MultiLineString(segment_geoms)

                # Calculate total length
                total_length = sum(segment['length'] for segment in group['segments'])

                # Get the electrical parameters
                voltage = group['voltage']
                r_per_km = group['r_per_km']
                x_per_km = group['x_per_km']
                b_per_km = group['b_per_km']

                # Calculate absolute values
                r_abs = r_per_km * total_length
                x_abs = x_per_km * total_length
                b_abs = b_per_km * total_length

                # Create new line
                new_line = {
                    'id': merged_id,
                    'geometry': merged_geom,
                    'length': total_length,
                    'v_nom': voltage,
                    'r_per_km': r_per_km,
                    'x_per_km': x_per_km,
                    'b_per_km': b_per_km,
                    'r': r_abs,
                    'x': x_abs,
                    'b': b_abs,
                    'num_segments': len(group['segments']),
                    'is_merged': True,
                    'original_ids': ','.join([segment['id'] for segment in group['segments']])
                }

                # Add in any other columns that might be needed
                for col in merged_lines.columns:
                    if col not in new_line and col not in ['id', 'geometry', 'length', 'v_nom',
                                                           'r_per_km', 'x_per_km', 'b_per_km',
                                                           'r', 'x', 'b']:
                        # Use the value from the first segment as default
                        if group['segments']:
                            first_seg_idx = group['segments'][0]['idx']
                            new_line[col] = merged_lines.iloc[first_seg_idx].get(col)
                        else:
                            new_line[col] = None

                new_lines.append(new_line)

            except Exception as e:
                logger.error(f"Error creating merged line {merged_id}: {str(e)}")

        # Create a GeoDataFrame with the merged lines
        if new_lines:
            # Create the GeoDataFrame with the new lines
            merged_gdf = gpd.GeoDataFrame(new_lines, crs=merged_lines.crs)

            # Identify segments that weren't merged
            unmerged_ids = [str(id) for id in merged_lines['id'].astype(str) if id not in segment_to_merged_map]
            unmerged_lines = merged_lines[merged_lines['id'].astype(str).isin(unmerged_ids)]

            # Add is_merged column to unmerged lines
            unmerged_lines['is_merged'] = False

            # Combine merged and unmerged lines
            combined_gdf = pd.concat([merged_gdf, unmerged_lines], ignore_index=True)

            logger.info(
                f"Created GeoDataFrame with {len(merged_gdf)} merged lines and {len(unmerged_lines)} unmerged lines")

            return combined_gdf, segment_to_merged_map

    # If no merging happened, return original data
    logger.info("No segments were merged, returning original network lines")
    return network_lines, {}

import itertools
import networkx as nx                      # ← requires networkx>=2.0
from shapely.geometry import Point, MultiLineString
import numpy as np
import pandas as pd
import geopandas as gpd
import logging

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------
# small helpers that already existed in your code base ----------------


def calculate_electrical_similarity(src, net):
    """Simple euclidean distance in (r,x,b)_per_km space, turned into similarity 0-1."""
    vals = ["r_per_km", "x_per_km", "b_per_km"]
    diff = np.sqrt(sum((src[v] - net[v]) ** 2 for v in vals))
    max_norm = 1e-3  # avoid div/0
    return np.exp(-diff / max_norm)

def calculate_combined_score(dir_sim, ov_src, ov_net, elec_sim,
                             volt_match, length_ratio):
    score = 0.5 * dir_sim + 0.3 * max(ov_src, ov_net) + 0.2 * elec_sim
    if volt_match:
        score += 0.05
    # penalise extremely different length ratios
    if length_ratio > 3:
        score *= 0.2
    return score
# --------------------------------------------------------------------


def _endpoints(geom):
    """Return a list of Point objects for every end-point of the geometry."""
    pts = []
    if geom.geom_type == "LineString":
        c = list(geom.coords)
        pts.extend([Point(c[0]), Point(c[-1])])
    elif geom.geom_type == "MultiLineString":
        for part in geom.geoms:
            c = list(part.coords)
            pts.extend([Point(c[0]), Point(c[-1])])
    return pts



def match_lines_detailed(
        source_lines, network_lines,
        buffer_distance=0.020, snap_distance=0.009,
        direction_threshold=0.65, enforce_voltage_matching=False,
        dataset_name="Source",
        merge_segments=True,
        max_matches_per_source=20):
    """
    Match one (long) source line against *all* connected network segments that
    overlap it.  Electrical parameters of the source are distributed to every
    matched segment proportionally to its length.
    """
    logger.info(
        f"Matching {dataset_name} lines "
        f"(buffer={buffer_distance}, snap={snap_distance}, "
        f"dir>= {direction_threshold})")

    # --- clean -------------------------------------------------------------
    source_lines   = source_lines[source_lines.geometry.notna()].copy()
    network_lines  = network_lines[network_lines.geometry.notna()].copy()
    if 'id' in source_lines.columns:
        source_lines['id'] = source_lines['id'].astype(str)
    if 'id' in network_lines.columns:
        network_lines['id'] = network_lines['id'].astype(str)

    # ----------------------------------------------------------------------
    # (optional) merge segments – keep your existing implementation -------
    if merge_segments:
        try:
            import preprocess_network_segments  # your function
            logger.info("Merging network segments …")
            network_lines = preprocess_network_segments(network_lines,
                                                        max_gap=snap_distance*2,
                                                        min_dir_similarity=0.8)
        except Exception as e:
            logger.warning(f"Segment merge failed – using raw segments: {e}")

    # spatial index ---------------------------------------------------------
    try:
        net_sindex = network_lines.sindex
    except Exception as e:
        logger.error(f"Could not build spatial index – fallback brute force ({e})")
        net_sindex = None

    # cache per-km electrical data -----------------------------------------
    def _per_km(row, field):
        if field + "_per_km" in row and row[field + "_per_km"] != 0:
            return row[field + "_per_km"]
        length = row.get("length", row.geometry.length * 111)
        return row.get(field, 0) / length if length else 0

    src_profiles  = {}
    for _, r in source_lines.iterrows():
        src_profiles[str(r["id"])] = dict(
            voltage=r.get("v_nom", 0),
            r_per_km=_per_km(r, "r"),
            x_per_km=_per_km(r, "x"),
            b_per_km=_per_km(r, "b"),
            length=r.get("length", r.geometry.length * 111)
        )

    net_profiles  = {}
    for _, r in network_lines.iterrows():
        net_profiles[str(r["id"])] = dict(
            voltage=r.get("v_nom", 0),
            r_per_km=_per_km(r, "r"),
            x_per_km=_per_km(r, "x"),
            b_per_km=_per_km(r, "b"),
            length=r.get("length", r.geometry.length * 111)
        )

    matches = []

    # ================= PER SOURCE LINE =====================================
    for s_idx, s_row in source_lines.iterrows():
        s_id   = str(s_row["id"])
        s_geom = s_row.geometry
        if s_geom is None or s_geom.is_empty:
            continue

        s_len  = src_profiles[s_id]["length"]
        s_volt = src_profiles[s_id]["voltage"]

        buf    = s_geom.buffer(max(buffer_distance, min(0.05, s_len*0.0005)))

        # candidate network rows
        if net_sindex is not None:
            cand_idx = list(net_sindex.intersection(buf.bounds))
            cand     = network_lines.iloc[cand_idx]
        else:
            cand     = network_lines

        cand = cand[cand.geometry.intersects(buf)]
        if cand.empty:
            continue

        # further filter by direction & voltage if requested
        good_idx = []
        for n_idx, n_row in cand.iterrows():
            dir_sim = direction_similarity(s_geom, n_row.geometry)
            if dir_sim < direction_threshold:
                continue
            if enforce_voltage_matching:
                n_volt = n_row.get("v_nom", 0)
                if not (s_volt == n_volt or
                        {s_volt, n_volt} == {380, 400}):
                    continue
            good_idx.append(n_idx)

        if not good_idx:
            continue

        cand = cand.loc[good_idx]

        # ------------- build connectivity graph in that local subset -------
        sub_net = cand.copy()
        graph   = _segment_graph(sub_net, snap_distance)

        # seeds = every segment that intersects the buffer (already true)
        seed_nodes = list(sub_net.index)

        visited = set(seed_nodes)
        q       = list(seed_nodes)
        while q:
            cur = q.pop(0)
            for neigh in graph.neighbors(cur):
                if neigh not in visited:
                    if sub_net.loc[neigh].geometry.intersects(buf):
                        visited.add(neigh)
                        q.append(neigh)

        group = network_lines.loc[list(visited)].copy()
        if group.empty:
            continue

        total_group_len = group["length"].fillna(
            group.geometry.length * 111).sum()

        # -------- allocate parameters --------------------------------------
        for _, n_row in group.iterrows():
            n_id   = str(n_row["id"])
            n_len  = n_row.get("length", n_row.geometry.length * 111)
            share  = (n_len / total_group_len) if total_group_len else 0

            matches.append({
                "dlr_id":          s_id,
                "network_id":      n_id,
                "source_length":   s_len,
                "network_length":  n_len,
                "direction_similarity": 1.0,   # already checked
                "allocated_r":     s_row.get("r", 0) * share,
                "allocated_x":     s_row.get("x", 0) * share,
                "allocated_b":     s_row.get("b", 0) * share,
                "network_r":       n_row.get("r", 0),
                "network_x":       n_row.get("x", 0),
                "network_b":       n_row.get("b", 0),
                "dlr_r":           s_row.get("r", 0),
                "dlr_x":           s_row.get("x", 0),
                "dlr_b":           s_row.get("b", 0),
                "dlr_voltage":     s_volt,
                "network_voltage": n_row.get("v_nom", 0),
                "match_type":      "multi_segment",
                "dataset":         dataset_name
            })

    # ----------------------------------------------------------------------
    if not matches:
        logger.warning(f"No matches produced for {dataset_name}")
        return pd.DataFrame()

    match_df = pd.DataFrame(matches)

    # optional change percentages (same as before) --------------------------
    for p in ("r", "x", "b"):
        match_df[f"{p}_change_pct"] = np.where(
            match_df[f"network_{p}"] != 0,
            ((match_df[f"allocated_{p}"] / match_df[f"network_{p}"]) - 1) * 100,
            0
        )

    logger.info(f"{dataset_name}: produced {len(match_df)} segment matches "
                f"for {len(source_lines)} source lines")
    return match_df