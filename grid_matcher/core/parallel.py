from collections import defaultdict

from grid_matcher.utils.helpers import _safe_int


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

def circuit_aware_matching(matching_results, jao_gdf, pypsa_gdf):
    """
    Circuit-aware matching that:
      • respects corridor-locked matches,
      • enforces per-edge circuit capacity without breaking corridor one-to-one,
      • reassigns excess uses to parallel connectors when possible,
      • optionally attaches unmatched JAOs to under-utilized edges,
      • annotates each result with 'circuit_allocation'.

    Returns the updated list of results (locked + modified).
    """
    print("\n===== ENHANCED CIRCUIT-AWARE MATCHING =====")

    from collections import defaultdict

    # --- split: never modify corridor-locked results ---
    locked = [r for r in matching_results if r.get("locked_by_corridor", False)]
    work   = [r for r in matching_results if not r.get("locked_by_corridor", False)]

    # --- helpers ---
    def _listify_ids(v):
        if not v:
            return []
        if isinstance(v, str):
            return [t.strip() for t in v.replace(";", ",").split(",") if t.strip()]
        if isinstance(v, (list, tuple, set)):
            return [str(x) for x in v if str(x)]
        return [str(v)]

    def _score_result(res):
        """Higher = better. Favor good coverage and quality words."""
        s = 0.0
        cov = float(res.get("coverage_ratio") or res.get("length_ratio") or 0.0)
        # bound cov to reasonable range so outliers don't dominate
        s += min(max(cov, 0.0), 2.0) * 100.0
        q = (res.get("match_quality") or "").lower()
        if "excellent" in q: s += 40
        elif "good" in q:   s += 25
        elif "fair" in q:   s += 10
        if "parallel circuit" in q: s += 10
        return s

    # --- lookups from PyPSA ---
    pypsa_by_id = {str(r.get("line_id", r.get("id", ""))): r for _, r in pypsa_gdf.iterrows()}

    def _circuits(pid):
        row = pypsa_by_id.get(pid)
        if row is None: return 1
        try:
            return int(row.get("circuits", 1) or 1)
        except Exception:
            return 1

    def _buspair(pid):
        row = pypsa_by_id.get(pid)
        if row is None: return None
        a, b = str(row.get("bus0") or ""), str(row.get("bus1") or "")
        if not a or not b: return None
        return tuple(sorted((a, b)))

    # index: bus pair -> list of edge IDs
    pair_index = defaultdict(list)
    for pid in pypsa_by_id.keys():
        key = _buspair(pid)
        if key:
            pair_index[key].append(pid)

    # --- usage accounting (include locked for truth) ---
    all_results = locked + work
    pypsa_usage = defaultdict(list)  # pid -> [jao_id,...]
    for res in all_results:
        if not res.get("matched"):
            continue
        for pid in _listify_ids(res.get("pypsa_ids")):
            pypsa_usage[pid].append(res.get("jao_id"))

    # --- UNDER-UTILIZED: optionally place unmatched JAOs there (light heuristic) ---
    under = {
        pid: {
            "cap": _circuits(pid),
            "used": len(jaos),
            "remain": max(0, _circuits(pid) - len(jaos))
        }
        for pid, jaos in pypsa_usage.items()
        if len(jaos) < _circuits(pid)
    }

    if under:
        print(f"Found {len(under)} PyPSA edges with unused circuit capacity")
        unmatched = [r for r in work if not r.get("matched")]
        # very permissive: just fill remaining capacity
        for pid, info in under.items():
            k = info["remain"]
            if k <= 0:
                continue
            take = unmatched[:k]
            unmatched = unmatched[k:]
            for r in take:
                r["matched"] = True
                r["pypsa_ids"] = [pid]
                r["match_quality"] = ("Circuit Match | " + (r.get("match_quality") or "")).strip(" |")
                pypsa_usage[pid].append(r.get("jao_id"))

    # --- OVER-USED: keep within capacity; try to reassign to parallel connectors ---
    over = {pid: jaos for pid, jaos in pypsa_usage.items() if len(jaos) > _circuits(pid)}
    if over:
        print(f"Found {len(over)} overused PyPSA edges (exceed circuits). Resolving...")

        # quick map: jao_id -> result (for edits)
        by_jao = {str(r.get("jao_id")): r for r in work}

        for pid, jao_ids in over.items():
            cap = _circuits(pid)

            # 1) keep all locked matches using this pid
            locked_users = [r for r in locked if pid in _listify_ids(r.get("pypsa_ids"))]
            keep_locked = {r.get("jao_id") for r in locked_users}

            # 2) among UNlocked users, keep the best (up to remaining capacity)
            unlocked_ids = [j for j in jao_ids if j not in keep_locked]
            unlocked_results = [(by_jao.get(j), j) for j in unlocked_ids if by_jao.get(j)]
            scored = sorted(
                [(res, _score_result(res)) for res, _ in unlocked_results],
                key=lambda t: t[1],
                reverse=True
            )
            remaining_slots = max(0, cap - len(keep_locked))
            keep_unlocked = {res.get("jao_id") for res, _ in scored[:remaining_slots]}
            to_move = [res for res, _ in scored[remaining_slots:]]

            # 3) try to reassign each "to_move" to a parallel connector with spare capacity
            key = _buspair(pid)
            alternatives = [q for q in pair_index.get(key, []) if q != pid] if key else []
            # track dynamic usage while we reassign
            dyn_used = {q: len(pypsa_usage.get(q, [])) for q in alternatives}
            dyn_cap  = {q: _circuits(q) for q in alternatives}

            for res in to_move:
                # if this res doesn't actually use pid anymore (changed earlier), skip
                ids_now = _listify_ids(res.get("pypsa_ids"))
                if pid not in ids_now:
                    continue

                moved = False
                # prefer alternatives with most spare capacity
                alternatives.sort(key=lambda q: (dyn_cap[q] - dyn_used[q]), reverse=True)
                for q in alternatives:
                    if dyn_used[q] < dyn_cap[q]:
                        # replace 'pid' with 'q' in this result's pypsa_ids
                        new_ids = [q if x == pid else x for x in ids_now]
                        res["pypsa_ids"] = new_ids
                        # update dynamic usage
                        dyn_used[q] += 1
                        # global usage bookkeeping
                        try:
                            pypsa_usage[pid].remove(res.get("jao_id"))
                        except ValueError:
                            pass
                        pypsa_usage.setdefault(q, []).append(res.get("jao_id"))
                        # annotate
                        mq = res.get("match_quality", "")
                        res["match_quality"] = ("Reassigned to parallel connector | " + mq).strip(" |")
                        moved = True
                        break

                if not moved:
                    # 4) could not reassign: mark as sharing (do not break the match)
                    res["is_sharing_circuit"] = True
                    mq = res.get("match_quality", "")
                    if "Circuit Sharing" not in mq:
                        res["match_quality"] = ("Circuit Sharing | " + mq).strip(" |")

    # --- annotate allocation info for ALL matched results (locked + work) ---
    def _fill_allocation(res_list):
        for r in res_list:
            if not r.get("matched"):
                continue
            pids = _listify_ids(r.get("pypsa_ids"))
            if not pids:
                r["matched"] = False
                continue
            r["circuit_allocation"] = {}
            for pid in pids:
                total = _circuits(pid)
                shared = len(pypsa_usage.get(pid, []))
                r["circuit_allocation"][pid] = {
                    "total_circuits": total,
                    "shared_with": shared,
                    "allocation_factor": 1.0 / max(1, shared)
                }

    _fill_allocation(locked)
    _fill_allocation(work)

    print("Enhanced circuit-aware matching completed")
    return locked + work


def comprehensive_parallel_circuit_matching(matches, jao_gdf, pypsa_gdf):
    """Comprehensively match parallel circuits across the entire network."""
    print("\n===== COMPREHENSIVE PARALLEL CIRCUIT MATCHING =====")

    # First filter out locked matches - don't try to modify them
    modifiable_matches = []
    locked_matches = []

    for match in matches:
        if match.get("locked_by_corridor", False):
            locked_matches.append(match)
        else:
            modifiable_matches.append(match)

    print(f"Found {len(locked_matches)} matches locked by corridor matching - will preserve these")
    """
    Comprehensively match parallel circuits across the entire network.

    This function identifies all cases where:
    1. Multiple JAO lines have identical or very similar geometries
    2. Multiple PyPSA lines have identical or very similar geometries
    3. Matches them in a distributed fashion to ensure maximum coverage

    Parameters:
    -----------
    matches : list
        List of match dictionaries from the matching process
    jao_gdf : GeoDataFrame
        GeoDataFrame containing JAO lines
    pypsa_gdf : GeoDataFrame
        GeoDataFrame containing PyPSA lines

    Returns:
    --------
    list
        Updated matching results with correct parallel circuit matches
    """
    print("\n===== COMPREHENSIVE PARALLEL CIRCUIT MATCHING =====")

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


def enhanced_parallel_circuit_matching(matches, jao_gdf, pypsa_gdf):
    """
    Enhanced function to properly match JAO parallel circuits to PyPSA parallel circuits.
    This function specifically focuses on cases where multiple JAO lines with identical
    geometries need to be matched to multiple PyPSA lines with identical geometries.
    """
    from collections import defaultdict
    print("\n=== ENHANCED PARALLEL CIRCUIT MATCHING ===")

    # 1. Group JAO lines by geometry to find parallel circuits
    jao_by_geometry = defaultdict(list)
    for _, row in jao_gdf.iterrows():
        if row.geometry is not None:
            geom_wkt = row.geometry.wkt
            jao_by_geometry[geom_wkt].append(str(row['id']))

    # Find JAO parallel groups (multiple lines with identical geometry)
    jao_parallel_groups = {wkt: ids for wkt, ids in jao_by_geometry.items() if len(ids) > 1}
    print(f"Found {len(jao_parallel_groups)} JAO parallel groups")

    # 2. Group PyPSA lines by geometry to find parallel circuits
    pypsa_by_geometry = defaultdict(list)
    for _, row in pypsa_gdf.iterrows():
        if row.geometry is not None:
            geom_wkt = row.geometry.wkt
            line_id = str(row.get('line_id', row.get('id', '')))
            pypsa_by_geometry[geom_wkt].append(line_id)

    # Find PyPSA parallel groups
    pypsa_parallel_groups = {wkt: ids for wkt, ids in pypsa_by_geometry.items() if len(ids) > 1}
    print(f"Found {len(pypsa_parallel_groups)} PyPSA parallel groups")

    # 3. For each JAO parallel group
    for jao_wkt, jao_ids in jao_parallel_groups.items():
        print(f"\nProcessing JAO parallel group: {', '.join(jao_ids)}")

        # Get all matches for this group
        jao_matches = []
        for jao_id in jao_ids:
            for match in matches:
                if str(match.get('jao_id', '')) == jao_id:
                    jao_matches.append(match)
                    break

        # >>> NEW: respect corridor locks (don’t remap anything in this group)
        if any(m.get("locked_by_corridor", False) for m in jao_matches):
            print("  Skipping group: at least one JAO is locked_by_corridor")
            continue
        # <<< NEW

        # Check if any are matched
        matched_jao_matches = [m for m in jao_matches if m.get('matched', False)]
        if not matched_jao_matches:
            print(f"  No matches found for this group")
            continue

        # 4. Collect all PyPSA IDs used by any JAO in this group
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
                used_pypsa_ids.add(pypsa_id)

        if not used_pypsa_ids:
            print(f"  No PyPSA IDs found for this group")
            continue

        # 5. Find which PyPSA parallel group these IDs belong to
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

        # 6. Check if we have enough PyPSA lines for all JAO lines
        if len(target_pypsa_ids) < len(jao_ids):
            print(f"  Warning: Not enough PyPSA lines ({len(target_pypsa_ids)}) for all JAO lines ({len(jao_ids)})")
            print(f"  Will match as many as possible")

        # 7. Match them one-to-one in sorted order
        sorted_jao_ids = sorted(jao_ids)
        sorted_pypsa_ids = sorted(target_pypsa_ids)

        print(f"  Matching {len(sorted_jao_ids)} JAO lines to {len(sorted_pypsa_ids)} PyPSA lines")
        for i, jao_id in enumerate(sorted_jao_ids):
            if i >= len(sorted_pypsa_ids):
                break

            # Find the match for this JAO ID
            for match in matches:
                if str(match.get('jao_id', '')) == jao_id:
                    pypsa_id = sorted_pypsa_ids[i]
                    old_pypsa_ids = match.get('pypsa_ids', [])

                    # Update the match
                    match['matched'] = True
                    match['pypsa_ids'] = [pypsa_id]
                    match['match_quality'] = f"Enhanced Parallel Circuit (was: {old_pypsa_ids})"
                    print(f"    Matched JAO {jao_id} to PyPSA {pypsa_id}")
                    break

    return matches


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

    Parameters:
    -----------
    results : list
        List of match dictionaries from the matching process
    jao_gdf : GeoDataFrame
        GeoDataFrame containing JAO lines
    pypsa_gdf : GeoDataFrame
        GeoDataFrame containing PyPSA lines
    jao_to_group : dict
        Mapping from JAO ID to its parallel group key
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



def identify_parallel_pypsa_circuits(pypsa_gdf):
    """
    Identify groups of PyPSA lines that share very similar geometry
    and are likely to be parallel circuits.

    Parameters:
    -----------
    pypsa_gdf : GeoDataFrame
        GeoDataFrame containing PyPSA lines

    Returns:
    --------
    dict
        Dictionary mapping group keys to lists of PyPSA line IDs
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
