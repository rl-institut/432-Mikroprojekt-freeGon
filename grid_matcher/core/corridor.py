
def match_identical_or_parallel_corridors(
        jao_gdf,
        pypsa_gdf,
        tol_m=60.0,
        overlap_iou=0.60,
        prefer_same_voltage=True,
):
    """
    One-to-one assignment for identical/parallel corridors (full or partial overlap).

    - Groups JAO & PyPSA lines into 'corridors' (same endpoints within tol, undirected).
    - Inside each corridor, builds candidate edges using Hausdorff distance (<= tol_m)
      OR buffered-IoU (>= overlap_iou).
    - Greedy maximum matching (no reuse): each JAO twin matched to a *distinct* PyPSA twin
      whenever available.

    Returns
    -------
    list[dict]
        Each dict like:
        {
          'matched': True/False,
          'jao_id': <str>,
          'pypsa_ids': [<str>, ...],        # one assigned id per JAO (or empty)
          'is_geometric_match': True/False,
          'is_parallel_circuit': True/False,
          'is_parallel_voltage_circuit': True/False,
          'is_duplicate': False,
          'match_quality': 'Corridor-Greedy' | 'Unmatched'
        }
    """
    from collections import defaultdict
    from math import isfinite
    import numpy as np

    try:
        import geopandas as gpd
        from shapely.geometry import LineString, Point
        from shapely.ops import transform
    except Exception as e:
        raise RuntimeError("geopandas and shapely are required") from e

    # --- Helpers --------------------------------------------------------------
    def _ensure_crs(gdf, epsg=4326):
        try:
            if gdf.crs is None:
                # Best effort: assume lon/lat if missing
                gdf = gdf.set_crs(epsg, allow_override=True)
            return gdf
        except Exception:
            return gdf

    def _to_meters(gdf):
        # Web Mercator is fine for sub-country extents; keeps things simple
        try:
            return gdf.to_crs(3857)
        except Exception:
            return gdf  # fallback (won't break, but distances become deg)

    def _round_xy(pt, grid_m=50.0):
        # Round projected coords to a grid (undirected endpoint key)
        # FIX: Properly handle both Point objects and coordinate tuples
        if pt is None:
            return None

        if hasattr(pt, 'x') and hasattr(pt, 'y'):
            # It's a Point object
            x = round(pt.x / grid_m) * grid_m
            y = round(pt.y / grid_m) * grid_m
        else:
            # Assume it's a coordinate tuple
            try:
                x = round(pt[0] / grid_m) * grid_m
                y = round(pt[1] / grid_m) * grid_m
            except (IndexError, TypeError):
                return None

        return (float(x), float(y))

    def _ensure_linestring(geom):
        """Return a LineString for consistent endpoint ops."""
        if geom is None:
            return None
        if isinstance(geom, LineString):
            return geom
        try:
            # Try to get first part of MultiLineString
            if hasattr(geom, 'geoms'):
                parts = list(geom.geoms)
                return max(parts, key=lambda g: g.length) if parts else None
            # Last resort: try to build from coords
            return LineString(list(geom.coords))
        except Exception:
            return None

    def _endpoint_key(geom, grid_m=50.0):
        """Create a unique key for line endpoints (undirected)"""
        if geom is None or geom.is_empty:
            return None

        # Ensure we have a LineString to work with
        ls = _ensure_linestring(geom)
        if ls is None:
            return None

        try:
            # Get first and last points
            coords = list(ls.coords)
            if len(coords) < 2:
                return None

            # Create Point objects for the endpoints
            start_point = Point(coords[0])
            end_point = Point(coords[-1])

            # Round the coordinates
            p0 = _round_xy(start_point, grid_m)
            p1 = _round_xy(end_point, grid_m)

            # Return sorted tuple for undirected matching
            if p0 is None or p1 is None:
                return None
            return tuple(sorted((p0, p1)))
        except Exception:
            return None

    def _voltage(row):
        try:
            return int(round(float(row.get("voltage", row.get("v_nom", 0)) or 0)))
        except Exception:
            return 0

    def _voltage_bin(v):
        if v >= 300: return "400kV"
        if v >= 200: return "220kV"
        return "other"

    def _get_id(row, keys):
        for k in keys:
            v = row.get(k)
            if v not in (None, "", np.nan):
                return str(v)
        return None

    def _hausdorff(a, b):
        try:
            d = a.hausdorff_distance(b)
            return float(d) if isfinite(d) else float("inf")
        except Exception:
            return float("inf")

    def _buffer_iou(a, b, w=30.0):
        try:
            ab = a.buffer(w)
            bb = b.buffer(w)
            inter = ab.intersection(bb).area
            den = ab.union(bb).area
            if den <= 0: return 0.0
            return float(inter / den)
        except Exception:
            return 0.0

    # --- Prep in meters -------------------------------------------------------
    jao_gdf = _ensure_crs(jao_gdf)
    pypsa_gdf = _ensure_crs(pypsa_gdf)

    try:
        jao_m = _to_meters(jao_gdf)
        pys_m = _to_meters(pypsa_gdf)
    except Exception as e:
        print(f"Warning: Unable to convert to meters projection: {str(e)}")
        print("Continuing with original coordinates.")
        jao_m = jao_gdf
        pys_m = pypsa_gdf

    # --- Build corridor groups -----------------------------------------------
    # key: (endpoint_pair, voltage_bin)  -> lists of indices
    corridors = defaultdict(lambda: {"jao": [], "pypsa": []})

    # Process JAO lines
    for i, row in jao_m.iterrows():
        try:
            jid = _get_id(row, ["id", "jao_id"])
            if not jid or row.geometry is None:
                continue

            # Get endpoint key for corridor matching
            k_end = _endpoint_key(row.geometry)
            if k_end is None:
                continue

            vbin = _voltage_bin(_voltage(row))
            corridors[(k_end, vbin)]["jao"].append(i)
        except Exception as e:
            print(f"Error processing JAO line at index {i}: {str(e)}")
            continue

    # Process PyPSA lines
    for i, row in pys_m.iterrows():
        try:
            pid = _get_id(row, ["line_id", "id"])
            if not pid or row.geometry is None:
                continue

            # Get endpoint key for corridor matching
            k_end = _endpoint_key(row.geometry)
            if k_end is None:
                continue

            vbin = _voltage_bin(_voltage(row))
            corridors[(k_end, vbin)]["pypsa"].append(i)
        except Exception as e:
            print(f"Error processing PyPSA line at index {i}: {str(e)}")
            continue

    # --- Candidate edges & greedy assignment ----------------------------------
    results = []

    # Report on corridors found
    total_corridors = len(corridors)
    matched_corridors = sum(1 for (_, _), members in corridors.items()
                            if members["jao"] and members["pypsa"])
    print(f"Found {total_corridors} total corridors, {matched_corridors} with both JAO and PyPSA lines")

    # Process each corridor
    for (ekey, vbin), members in corridors.items():
        jao_idxs = members["jao"]
        py_idxs = members["pypsa"]

        if not jao_idxs:
            continue  # nothing to do

        # If no PyPSA in corridor, mark JAO unmatched
        if not py_idxs:
            for j in jao_idxs:
                try:
                    jr = jao_gdf.iloc[j]
                    results.append({
                        "matched": False,
                        "jao_id": str(_get_id(jr, ["id", "jao_id"]) or ""),
                        "pypsa_ids": [],
                        "is_geometric_match": False,
                        "is_parallel_circuit": len(jao_idxs) > 1,
                        "is_parallel_voltage_circuit": False,
                        "is_duplicate": len(jao_idxs) > 1,
                        "match_quality": "Unmatched"
                    })
                except Exception as e:
                    print(f"Error creating unmatched result for JAO index {j}: {str(e)}")
            continue

        # Build all edges with cost
        edges = []  # (cost, j_idx, p_idx, flags)
        for j in jao_idxs:
            try:
                gJ = jao_m.iloc[j].geometry
                vJ = _voltage(jao_gdf.iloc[j])

                for p in py_idxs:
                    try:
                        gP = pys_m.iloc[p].geometry
                        vP = _voltage(pypsa_gdf.iloc[p])

                        # Optionally deprioritize different voltage
                        vpen = 0.0 if (not prefer_same_voltage or vJ == vP) else 5.0

                        dH = _hausdorff(gJ, gP)  # meters
                        iou = _buffer_iou(gJ, gP, w=30.0)

                        if (dH <= tol_m) or (iou >= overlap_iou):
                            cost = dH + vpen - (iou * 10.0)  # low is good
                            edges.append((cost, j, p, {"dH": dH, "IoU": iou, "sameV": (vJ == vP)}))
                    except Exception as e:
                        print(f"Error comparing JAO {j} with PyPSA {p}: {str(e)}")
                        continue
            except Exception as e:
                print(f"Error processing JAO index {j} for edge building: {str(e)}")
                continue

        # If no edges, they don't overlap enough; mark unmatched
        if not edges:
            for j in jao_idxs:
                try:
                    jr = jao_gdf.iloc[j]
                    results.append({
                        "matched": False,
                        "jao_id": str(_get_id(jr, ["id", "jao_id"]) or ""),
                        "pypsa_ids": [],
                        "is_geometric_match": False,
                        "is_parallel_circuit": len(jao_idxs) > 1,
                        "is_parallel_voltage_circuit": False,
                        "is_duplicate": len(jao_idxs) > 1,
                        "match_quality": "Unmatched"
                    })
                except Exception as e:
                    print(f"Error creating unmatched result for JAO index {j}: {str(e)}")
            continue

        # Greedy maximum matching: sort by cost, then pick if both free
        edges.sort(key=lambda t: t[0])
        used_j = set()
        used_p = set()
        assignments = []  # (j_idx, p_idx, flags)

        for cost, j, p, flags in edges:
            if j in used_j or p in used_p:
                continue
            used_j.add(j)
            used_p.add(p)
            assignments.append((j, p, flags))

            # Early exit if perfect one-to-one
            if len(used_j) == len(jao_idxs) or len(used_p) == len(py_idxs):
                # no more pairs possible
                break

        # Emit results for matched JAO in this corridor
        assigned_j = {j for j, _, _ in assignments}
        for j, p, flags in assignments:
            try:
                jr = jao_gdf.iloc[j]
                pr = pypsa_gdf.iloc[p]

                results.append({
                    "matched": True,
                    "jao_id": str(_get_id(jr, ["id", "jao_id"]) or ""),
                    "pypsa_ids": [str(_get_id(pr, ["line_id", "id"]) or "")],
                    "is_geometric_match": True,
                    "is_parallel_circuit": (len(jao_idxs) > 1 or len(py_idxs) > 1),
                    "is_parallel_voltage_circuit": False,
                    "is_duplicate": (len(jao_idxs) > 1),
                    "match_quality": "Corridor-Greedy"
                })
            except Exception as e:
                print(f"Error creating match result for JAO {j} to PyPSA {p}: {str(e)}")

        # Any remaining JAO in the corridor that didn't get a unique PyPSA
        # are left unmatched (instead of reusing a PyPSA id)
        for j in (set(jao_idxs) - assigned_j):
            try:
                jr = jao_gdf.iloc[j]
                results.append({
                    "matched": False,
                    "jao_id": str(_get_id(jr, ["id", "jao_id"]) or ""),
                    "pypsa_ids": [],
                    "is_geometric_match": False,
                    "is_parallel_circuit": (len(jao_idxs) > 1 or len(py_idxs) > 1),
                    "is_parallel_voltage_circuit": False,
                    "is_duplicate": (len(jao_idxs) > 1),
                    "match_quality": "Unmatched"
                })
            except Exception as e:
                print(f"Error creating unmatched result for JAO index {j}: {str(e)}")

    # Print summary
    matched_count = sum(1 for r in results if r.get("matched", False))
    total_count = len(results)
    print(f"Corridor matching complete: {matched_count}/{total_count} JAO lines matched")

    return results


def match_multi_segment_corridors(jao_gdf, pypsa_gdf):
    """
    Match JAO lines to multi-segment corridors made up of connected PyPSA segments.
    Respects circuit constraints - only match as many JAO lines as there are circuits.

    Returns:
    --------
    tuple
        (matching_results, forced_unmatched_jao_ids)
        - matching_results: List of match dictionaries
        - forced_unmatched_jao_ids: List of JAO IDs that should never be matched elsewhere
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
                # Add more shared circuit groups as needed
            ]
        }
        # Add more corridors as needed
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
                    # KEY CHANGE: Mark unmatched corridor JAO lines as locked too, to prevent other
                    # functions from trying to match them later, and add to forced_unmatched list
                    results.append({
                        'matched': False,
                        'jao_id': jao_id,
                        'pypsa_ids': [],
                        'match_quality': f"Unmatched - No available circuit (capacity of {circuit_capacity} already allocated)",
                        'is_geometric_match': False,
                        'is_parallel_circuit': False,
                        'is_duplicate': False,
                        'locked_by_corridor': True,  # IMPORTANT: Lock unmatched JAO lines too!
                        'forced_unmatched': True  # New flag to indicate this should stay unmatched
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