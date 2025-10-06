import networkx as nx
import numpy as np
from grid_matcher.utils.helpers import get_start_point, get_end_point

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
