from grid_matcher.utils.helpers import _num, _to_km


def allocate_electrical_parameters(jao_gdf, pypsa_gdf, matching_results):
    """Allocate electrical parameters from JAO lines to PyPSA segments."""
    import numpy as np

    # Create lookups
    pypsa_by_id = {str(row.get('line_id', row.get('id', ''))): row for _, row in pypsa_gdf.iterrows()}
    jao_by_id = {str(row['id']): row for _, row in jao_gdf.iterrows()}

    print("\n=== ALLOCATING ELECTRICAL PARAMETERS (DIRECT METHOD) ===")

    # Track which PyPSA lines have been allocated
    allocated_pypsa_ids = set()

    for result in matching_results:
        jao_id = result.get('jao_id', 'unknown')
        print(f"Processing JAO {jao_id}")

        # Skip if not matched or no PyPSA lines
        if not result.get('matched', False) or not result.get('pypsa_ids'):
            continue

        # Get JAO parameters
        jao_row = jao_by_id.get(str(jao_id))
        if jao_row is None:
            print(f"  Cannot find JAO {jao_id} in the data")
            continue

        # Get JAO parameters
        jao_length_km = float(jao_row.get('length', 0))
        jao_r_total = float(jao_row.get('r', 0))
        jao_x_total = float(jao_row.get('x', 0))
        jao_b_total = float(jao_row.get('b', 0))

        # Store in result
        result['jao_length_km'] = jao_length_km
        result['jao_r'] = jao_r_total
        result['jao_x'] = jao_x_total
        result['jao_b'] = jao_b_total

        # Calculate per-km values
        jao_r_per_km = jao_r_total / jao_length_km if jao_length_km > 0 else 0
        jao_x_per_km = jao_x_total / jao_length_km if jao_length_km > 0 else 0
        jao_b_per_km = jao_b_total / jao_length_km if jao_length_km > 0 else 0

        result['jao_r_per_km'] = jao_r_per_km
        result['jao_x_per_km'] = jao_x_per_km
        result['jao_b_per_km'] = jao_b_per_km

        print(f"  JAO {jao_id}: R={jao_r_total:.6f}Ω X={jao_x_total:.6f}Ω B={jao_b_total:.8f}S")

        # Get PyPSA ids
        pypsa_ids = result.get('pypsa_ids', [])
        if isinstance(pypsa_ids, str):
            pypsa_ids = [id.strip() for id in pypsa_ids.split(';') if id.strip()]

        # Initialize
        segments = []
        total_length_km = 0

        # HACK: Special case for Vieselbach-Remptendorf - use safer pandas access
        is_vieselbach_remptendorf = False
        if jao_id == "26":
            is_vieselbach_remptendorf = True
            print("  SPECIAL CASE: Vieselbach-Remptendorf detected by ID")
        elif jao_row is not None:
            # Safely check if NE_name column exists and matches
            if 'NE_name' in jao_row and jao_row['NE_name'] == "Vieselbach - Remptendorf 416":
                is_vieselbach_remptendorf = True
                print("  SPECIAL CASE: Vieselbach-Remptendorf detected by name")

        # Calculate total length for allocation
        for pid in pypsa_ids:
            pypsa_row = pypsa_by_id.get(pid)
            if pypsa_row is None:
                continue

            length_m = float(pypsa_row.get('length', 0))
            length_km = length_m / 1000.0
            total_length_km += length_km

        # Allocate parameters
        allocated_r_sum = 0
        allocated_x_sum = 0
        allocated_b_sum = 0

        for pid in pypsa_ids:
            pypsa_row = pypsa_by_id.get(pid)
            if pypsa_row is None:
                continue

            # Get PyPSA properties
            length_m = float(pypsa_row.get('length', 0))
            length_km = length_m / 1000.0
            circuits = int(pypsa_row.get('circuits', 1))

            # Calculate proportion of total length
            length_proportion = length_km / total_length_km

            # DIRECT ALLOCATION - no circuit adjustment
            alloc_r = jao_r_total * length_proportion
            alloc_x = jao_x_total * length_proportion
            alloc_b = jao_b_total * length_proportion

            print(f"  Allocating to {pid}: {length_proportion:.4f} of JAO total")
            print(f"    R: {alloc_r:.6f}Ω X: {alloc_x:.6f}Ω B: {alloc_b:.8f}S")

            # Add to running sums
            allocated_r_sum += alloc_r
            allocated_x_sum += alloc_x
            allocated_b_sum += alloc_b

            # Create segment record
            segment = {
                'network_id': pid,
                'length_km': length_km,
                'num_parallel': circuits,
                'allocation_factor': 1.0,
                'segment_ratio': length_km / max(jao_length_km, 1e-9),
                'allocated_r': alloc_r,
                'allocated_x': alloc_x,
                'allocated_b': alloc_b,
                'allocation_status': 'Parallel Circuit' if circuits > 1 else 'Applied'
            }

            segments.append(segment)

        # Update result
        result['matched_lines_data'] = segments
        result['matched_km'] = total_length_km
        result['coverage_ratio'] = total_length_km / max(jao_length_km, 1e-9)
        result['allocated_r_sum'] = allocated_r_sum
        result['allocated_x_sum'] = allocated_x_sum
        result['allocated_b_sum'] = allocated_b_sum

        # Calculate residuals
        r_residual = jao_r_total - allocated_r_sum
        x_residual = jao_x_total - allocated_x_sum
        b_residual = jao_b_total - allocated_b_sum

        def safe_pct(num, den):
            if abs(den) < 1e-9:
                return 0
            return 100.0 * num / abs(den)

        result['residual_r_percent'] = safe_pct(r_residual, jao_r_total)
        result['residual_x_percent'] = safe_pct(x_residual, jao_x_total)
        result['residual_b_percent'] = safe_pct(b_residual, jao_b_total)

        print(
            f"  Allocation complete: R sum={allocated_r_sum:.6f}Ω X sum={allocated_x_sum:.6f}Ω B sum={allocated_b_sum:.8f}S")
        print(
            f"  Residuals: R={r_residual:.6f}Ω ({result['residual_r_percent']:.2f}%) X={x_residual:.6f}Ω ({result['residual_x_percent']:.2f}%) B={b_residual:.8f}S ({result['residual_b_percent']:.2f}%)")

    return matching_results

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


def _extract_jao_params(result, jao_gdf):
    """
    Extract JAO parameters from the authoritative JAO GeoDataFrame with a
    robust fallback to the 'result' dict. All returned lengths are in km.

    Returns dict with keys:
      - jao_length_km
      - jao_r_total, jao_x_total, jao_b_total
      - jao_r_per_km, jao_x_per_km, jao_b_per_km
    """
    import pandas as pd

    jao_id = result.get("jao_id")
    out = {
        "jao_length_km": None,
        "jao_r_total": None,
        "jao_x_total": None,
        "jao_b_total": None,
        "jao_r_per_km": None,
        "jao_x_per_km": None,
        "jao_b_per_km": None,
    }

    # 1) Try authoritative source: JAO GDF
    jao_row = None
    try:
        if jao_id is not None and "id" in jao_gdf.columns:
            sel = jao_gdf[jao_gdf["id"].astype(str) == str(jao_id)]
            if not sel.empty:
                jao_row = sel.iloc[0]
    except Exception:
        jao_row = None

    # Length (km)
    if jao_row is not None:
        out["jao_length_km"] = _to_km(
            _get_first_existing(jao_row, "length_km", "length", "len_km", "len")
        )

    # Totals from authoritative row first
    if jao_row is not None:
        out["jao_r_total"] = _num(
            _get_first_existing(jao_row, "r", "R", "r_total", "R_total")
        )
        out["jao_x_total"] = _num(
            _get_first_existing(jao_row, "x", "X", "x_total", "X_total")
        )
        out["jao_b_total"] = _num(
            _get_first_existing(jao_row, "b", "B", "b_total", "B_total")
        )

        out["jao_r_per_km"] = _num(
            _get_first_existing(jao_row, "r_per_km", "R_per_km", "r_km", "R_km")
        )
        out["jao_x_per_km"] = _num(
            _get_first_existing(jao_row, "x_per_km", "X_per_km", "x_km", "X_km")
        )
        out["jao_b_per_km"] = _num(
            _get_first_existing(jao_row, "b_per_km", "B_per_km", "b_km", "B_km")
        )

    # 2) Fill any gaps from 'result' dict
    if out["jao_length_km"] is None:
        # result may store km directly or meters as 'jao_length'
        length_km = None
        if result.get("jao_length_km") is not None:
            length_km = _num(result.get("jao_length_km"))
        elif result.get("jao_length") is not None:
            length_km = _to_km(result.get("jao_length"))
        out["jao_length_km"] = length_km

    if out["jao_r_total"] is None:
        out["jao_r_total"] = _num(result.get("jao_r"))
    if out["jao_x_total"] is None:
        out["jao_x_total"] = _num(result.get("jao_x"))
    if out["jao_b_total"] is None:
        out["jao_b_total"] = _num(result.get("jao_b"))

    if out["jao_r_per_km"] is None:
        out["jao_r_per_km"] = _num(result.get("jao_r_per_km"))
    if out["jao_x_per_km"] is None:
        out["jao_x_per_km"] = _num(result.get("jao_x_per_km"))
    if out["jao_b_per_km"] is None:
        out["jao_b_per_km"] = _num(result.get("jao_b_per_km"))

    # 3) Cross-derive missing per-km or totals from length
    L = out["jao_length_km"]
    if L is not None and L > 0:
        if out["jao_r_per_km"] is None and out["jao_r_total"] is not None:
            out["jao_r_per_km"] = out["jao_r_total"] / L
        if out["jao_x_per_km"] is None and out["jao_x_total"] is not None:
            out["jao_x_per_km"] = out["jao_x_total"] / L
        if out["jao_b_per_km"] is None and out["jao_b_total"] is not None:
            out["jao_b_per_km"] = out["jao_b_total"] / L

        if out["jao_r_total"] is None and out["jao_r_per_km"] is not None:
            out["jao_r_total"] = out["jao_r_per_km"] * L
        if out["jao_x_total"] is None and out["jao_x_per_km"] is not None:
            out["jao_x_total"] = out["jao_x_per_km"] * L
        if out["jao_b_total"] is None and out["jao_b_per_km"] is not None:
            out["jao_b_total"] = out["jao_b_per_km"] * L

    return out


def _get_first_existing(row, *names):
    """
    Get the first present, non-null value from a row (Series/dict-like) by name.
    Returns None if none of the provided names exist or all are null/NaN.
    """
    try:
        import pandas as pd
    except Exception:
        pd = None

    for name in names:
        try:
            val = None
            if hasattr(row, "__contains__") and name in row:
                val = row[name]
            elif hasattr(row, "get"):
                val = row.get(name, None)
            # Treat pandas NA/None/empty string as missing
            if val is None:
                continue
            if pd is not None:
                try:
                    if pd.isna(val):
                        continue
                except Exception:
                    pass
            if isinstance(val, str) and val.strip() == "":
                continue
            return val
        except Exception:
            continue
    return None