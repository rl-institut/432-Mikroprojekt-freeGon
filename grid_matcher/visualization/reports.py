
def create_enhanced_summary_table(jao_gdf, pypsa_gdf, matching_results):
    """
    Create an HTML table summarizing electrical parameter allocation from JAO to PyPSA.
    Safe against missing fields and unit mismatches (assumes JAO in km, PyPSA in m).
    Includes total length statistics for matching quality assessment.
    """
    import os
    from pathlib import Path
    import pandas as pd

    print("Creating enhanced parameter summary table...")

    # ---- helpers ----
    def meters_to_km(val):
        try:
            if val is None or pd.isna(val):
                return None
            return float(val) / 1000.0
        except Exception:
            return None

    def _sf(val, fmt=".6f", na="-"):
        try:
            if val is None or (isinstance(val, float) and (val != val)):  # NaN check
                return na
            return f"{float(val):{fmt}}"
        except Exception:
            return na

    def _safe_pct(num, den):
        try:
            den = float(den)
            if den == 0:
                return None
            return 100.0 * float(num) / den
        except Exception:
            return None

    # ---- quick lookups ----
    jao_by_id = {str(r["id"]): r for _, r in jao_gdf.iterrows()}
    pypsa_by_id = {str(r.get("line_id", r.get("id", ""))): r for _, r in pypsa_gdf.iterrows()}

    # ---- count statistics ----
    total_jao = len(jao_gdf)
    total_pypsa = len(pypsa_gdf)

    matched_jao = sum(1 for r in (matching_results or []) if r.get("matched", False))

    matched_pypsa_ids = set()
    for r in (matching_results or []):
        if r.get("matched", False) and r.get("pypsa_ids"):
            ids = r["pypsa_ids"]
            if isinstance(ids, str):
                ids = [p.strip() for p in ids.split(";") if p.strip()]
            matched_pypsa_ids.update(map(str, ids))
    matched_pypsa = len(matched_pypsa_ids)

    # ---- length statistics ----
    # Calculate total lengths for JAO and PyPSA networks
    total_jao_length_km = 0.0
    for _, row in jao_gdf.iterrows():
        if "length" in row and row["length"] is not None:
            lv = float(row["length"])
            jao_length = lv / 1000.0 if lv > 1000 else lv
        else:
            jao_length = float(row.get("length_km", 0) or 0)
        total_jao_length_km += jao_length

    # Track which PyPSA lines have already been counted in matched calculations
    # to prevent double-counting when the same line is matched multiple times
    matched_pypsa_line_counter = {}

    # First calculate total PyPSA length (physical, regardless of circuits)
    total_pypsa_length_km = 0.0
    for _, row in pypsa_gdf.iterrows():
        pypsa_id = str(row.get("line_id", row.get("id", "")))
        pypsa_length = meters_to_km(row.get("length", None)) or 0.0
        total_pypsa_length_km += pypsa_length
        # Initialize matched length counter for this line
        matched_pypsa_line_counter[pypsa_id] = 0.0

    # Calculate matched lengths
    matched_jao_length_km = 0.0
    matched_pypsa_length_km = 0.0

    for result in (matching_results or []):
        if not result.get("matched", False):
            continue

        # Add JAO matched length
        jao_id = str(result.get("jao_id", ""))
        jao_row = jao_by_id.get(jao_id)
        if jao_row is not None:
            if "length" in jao_row and jao_row["length"] is not None:
                lv = float(jao_row["length"])
                jao_length_km = lv / 1000.0 if lv > 1000 else lv
            else:
                jao_length_km = float(jao_row.get("length_km", 0) or 0)
        else:
            jao_length_km = float(result.get("jao_length_km", 0) or 0)

        matched_jao_length_km += jao_length_km

        # Add PyPSA matched length without double-counting
        segs = result.get("matched_lines_data") or []
        if segs:
            for seg in segs:
                pid = str(seg.get("network_id", ""))
                prow = pypsa_by_id.get(pid)

                if prow is not None:
                    km = meters_to_km(prow.get("length", None)) or 0.0
                    circuits = int(prow.get("circuits", 1) or 1)

                    # Calculate the portion of this line being used by this match
                    # Don't add more than the physical length (adjusted for remaining circuits)
                    remaining_length = km - matched_pypsa_line_counter.get(pid, 0.0)
                    if remaining_length > 0:
                        allocated_length = min(remaining_length, km)
                        matched_pypsa_line_counter[pid] += allocated_length
                        matched_pypsa_length_km += allocated_length
                else:
                    # If we can't find the line in pypsa_by_id, use the segment length
                    # but still track to avoid double counting
                    km = float(seg.get("length_km", 0) or 0.0)
                    if pid not in matched_pypsa_line_counter:
                        matched_pypsa_line_counter[pid] = 0.0

                    remaining_length = km - matched_pypsa_line_counter.get(pid, 0.0)
                    if remaining_length > 0:
                        allocated_length = min(remaining_length, km)
                        matched_pypsa_line_counter[pid] += allocated_length
                        matched_pypsa_length_km += allocated_length

    # Calculate percentages
    jao_length_pct = _safe_pct(matched_jao_length_km, total_jao_length_km)
    pypsa_length_pct = _safe_pct(matched_pypsa_length_km, total_pypsa_length_km)

    # ---- HTML header ----
    html = f"""<!DOCTYPE html>
<html>
<head>
  <title>JAO-PyPSA Electrical Parameter Allocation</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 20px; }}
    h1, h2, h3 {{ color: #333; }}
    .summary {{ margin-bottom: 20px; padding: 15px; background-color: #f5f5f5; border-radius: 5px; }}
    .stats-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 15px; }}
    .stats-card {{ background-color: white; padding: 15px; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
    .stats-title {{ font-size: 16px; font-weight: bold; margin-bottom: 10px; color: #4CAF50; }}
    .stat-row {{ display: flex; justify-content: space-between; margin-bottom: 5px; }}
    .stat-label {{ color: #555; }}
    .stat-value {{ font-weight: bold; }}
    .progress-bar-container {{ width: 100%; height: 20px; background-color: #e0e0e0; border-radius: 10px; margin-top: 5px; overflow: hidden; }}
    .progress-bar {{ height: 100%; background-color: #4CAF50; border-radius: 10px; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 20px; margin-bottom: 30px; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
    th {{ background-color: #4CAF50; color: white; }}
    .filter-controls {{ margin: 20px 0; padding: 10px; background-color: #eee; border-radius: 5px; }}
    .toggle-btn {{ background-color: #4CAF50; color: white; padding: 5px 10px; border: none; border-radius: 4px; cursor: pointer; margin-bottom: 10px; }}
    .details-section {{ display: none; }}
    .good-match {{ background-color: #c8e6c9; }}
    .moderate-match {{ background-color: #fff9c4; }}
    .poor-match {{ background-color: #ffccbc; }}
    .debug-info {{ font-family: monospace; font-size: 12px; color: #555; background-color: #f9f9f9; padding: 8px; margin-bottom: 10px; border-radius: 3px; display: none; }}
    .show-debug {{ cursor: pointer; color: #0066cc; margin-bottom: 5px; }}
    .show-debug:hover {{ text-decoration: underline; }}
  </style>
  <script>
    function filterTable() {{
      const q = document.getElementById('filter').value.toLowerCase();
      const rows = document.getElementById('resultsTable').getElementsByTagName('tr');
      for (let i=1; i<rows.length; i++) {{
        rows[i].style.display = rows[i].textContent.toLowerCase().includes(q) ? '' : 'none';
      }}
    }}
    function toggleDetails(id) {{
      const details = document.getElementById('details-'+id);
      const btn = document.getElementById('btn-'+id);
      const open = details.style.display === 'block';
      details.style.display = open ? 'none' : 'block';
      btn.textContent = open ? 'Show Parameters' : 'Hide Parameters';
    }}
    function toggleDebugInfo(id) {{
      const el = document.getElementById('debug-'+id);
      el.style.display = el.style.display === 'block' ? 'none' : 'block';
    }}
  </script>
</head>
<body>
  <h1>JAO-PyPSA Electrical Parameter Allocation</h1>

  <div class="summary">
    <h2>Matching Summary</h2>

    <div class="stats-grid">
      <div class="stats-card">
        <div class="stats-title">Line Count Statistics</div>
        <div class="stat-row">
          <span class="stat-label">JAO Lines:</span>
          <span class="stat-value">{total_jao} (Matched: {matched_jao}, {_safe_pct(matched_jao, max(1, total_jao)):.1f}%)</span>
        </div>
        <div class="progress-bar-container">
          <div class="progress-bar" style="width: {_safe_pct(matched_jao, max(1, total_jao)):.1f}%"></div>
        </div>

        <div class="stat-row" style="margin-top: 15px;">
          <span class="stat-label">PyPSA Lines:</span>
          <span class="stat-value">{total_pypsa} (Matched: {matched_pypsa}, {_safe_pct(matched_pypsa, max(1, total_pypsa)):.1f}%)</span>
        </div>
        <div class="progress-bar-container">
          <div class="progress-bar" style="width: {_safe_pct(matched_pypsa, max(1, total_pypsa)):.1f}%"></div>
        </div>
      </div>

      <div class="stats-card">
        <div class="stats-title">Length Coverage Statistics</div>
        <div class="stat-row">
          <span class="stat-label">JAO Total Length:</span>
          <span class="stat-value">{_sf(total_jao_length_km, '.1f')} km</span>
        </div>
        <div class="stat-row">
          <span class="stat-label">JAO Matched Length:</span>
          <span class="stat-value">{_sf(matched_jao_length_km, '.1f')} km ({_sf(jao_length_pct, '.1f')}%)</span>
        </div>
        <div class="progress-bar-container">
          <div class="progress-bar" style="width: {_sf(jao_length_pct, '.1f')}%"></div>
        </div>

        <div class="stat-row" style="margin-top: 15px;">
          <span class="stat-label">PyPSA Total Length:</span>
          <span class="stat-value">{_sf(total_pypsa_length_km, '.1f')} km</span>
        </div>
        <div class="stat-row">
          <span class="stat-label">PyPSA Matched Length:</span>
          <span class="stat-value">{_sf(matched_pypsa_length_km, '.1f')} km ({_sf(pypsa_length_pct, '.1f')}%)</span>
        </div>
        <div class="progress-bar-container">
          <div class="progress-bar" style="width: {_sf(pypsa_length_pct, '.1f')}%"></div>
        </div>
      </div>
    </div>

    <p><strong>Note:</strong> JAO lengths are treated as km; PyPSA lengths are converted from meters to km for display.</p>
  </div>

  <div class="filter-controls">
    <h2>Filter Results</h2>
    <input type="text" id="filter" onkeyup="filterTable()" placeholder="Search for lines...">
  </div>

  <h2>Parameter Allocation Results</h2>
  <table id="resultsTable">
    <tr>
      <th>JAO ID</th><th>JAO Name</th><th>Voltage (kV)</th>
      <th>PyPSA IDs</th><th>JAO Length (km)</th><th>PyPSA Length (km)</th>
      <th>Coverage (%)</th><th>Parameters</th>
    </tr>
"""

    # ---- rows ----
    row_counter = 0
    for result in (matching_results or []):
        if not result.get("matched", False):
            continue

        row_counter += 1
        jao_id = str(result.get("jao_id", ""))
        jao_row = jao_by_id.get(jao_id)
        if jao_row is not None:
            jao_name = jao_row.get("NE_name", "")
            voltage = jao_row.get("v_nom", "")
            # length in km (source of truth)
            if "length" in jao_row and jao_row["length"] is not None:
                # JAO in km unless absurd
                lv = float(jao_row["length"])
                jao_length_km = lv / 1000.0 if lv > 1000 else lv
            else:
                jao_length_km = float(jao_row.get("length_km", 0) or 0)
            jao_r_total = float(jao_row.get("r", 0) or 0)
            jao_x_total = float(jao_row.get("x", 0) or 0)
            jao_b_total = float(jao_row.get("b", 0) or 0)
        else:
            jao_name = result.get("jao_name", "")
            voltage = result.get("v_nom", "")
            jao_length_km = float(result.get("jao_length_km", 0) or 0)
            jao_r_total = float(result.get("jao_r", 0) or 0)
            jao_x_total = float(result.get("jao_x", 0) or 0)
            jao_b_total = float(result.get("jao_b", 0) or 0)

        jao_r_km = (jao_r_total / jao_length_km) if jao_length_km > 0 else 0.0
        jao_x_km = (jao_x_total / jao_length_km) if jao_length_km > 0 else 0.0
        jao_b_km = (jao_b_total / jao_length_km) if jao_length_km > 0 else 0.0

        # pypsa ids string
        pypsa_ids = result.get("pypsa_ids", [])
        if isinstance(pypsa_ids, str):
            pypsa_ids = [p.strip() for p in pypsa_ids.split(";") if p.strip()]
        pypsa_ids_str = ", ".join(map(str, pypsa_ids))

        # sum of matched PyPSA km from source data
        total_pypsa_km = 0.0
        segs = result.get("matched_lines_data") or []
        if segs:
            for seg in segs:
                pid = str(seg.get("network_id", ""))
                prow = pypsa_by_id.get(pid)
                if prow is not None:
                    km = meters_to_km(prow.get("length", None))
                    total_pypsa_km += (km or 0.0)
                else:
                    total_pypsa_km += float(seg.get("length_km", 0) or 0.0)
        else:
            # NEW: if segments haven't been materialized yet, sum by pypsa_ids
            for pid in (pypsa_ids or []):
                prow = pypsa_by_id.get(str(pid))
                if prow is not None:
                    km = meters_to_km(prow.get("length", None))
                    total_pypsa_km += (km or 0.0)

        coverage = _safe_pct(total_pypsa_km, jao_length_km)

        html += f"""
    <tr>
      <td>{jao_id}</td>
      <td>{jao_name}</td>
      <td>{voltage}</td>
      <td>{pypsa_ids_str}</td>
      <td>{_sf(jao_length_km, '.2f')}</td>
      <td>{_sf(total_pypsa_km, '.2f')}</td>
      <td>{_sf(coverage, '.1f')}%</td>
      <td><button id="btn-{row_counter}" class="toggle-btn" onclick="toggleDetails('{row_counter}')">Show Parameters</button></td>
    </tr>
    <tr>
      <td colspan="8">
        <div id="details-{row_counter}" class="details-section">
          <div class="show-debug" onclick="toggleDebugInfo('{row_counter}')">Show/Hide Debug Info</div>
          <div id="debug-{row_counter}" class="debug-info">
            <strong>Debug Information:</strong><br>
            JAO ID: {jao_id}<br>
            JAO length (km): {_sf(jao_length_km, '.6f')}<br>
            PyPSA total length (km): {_sf(total_pypsa_km, '.6f')}<br>
            Coverage ratio: {_sf(coverage, '.2f')}%<br>
            Raw JAO R/X/B: {_sf(jao_r_total)} / {_sf(jao_x_total)} / {_sf(jao_b_total, '.8f')}<br>
            JAO per-km R/X/B: {_sf(jao_r_km)} / {_sf(jao_x_km)} / {_sf(jao_b_km, '.8f')}<br>
          </div>
"""

        if segs:
            # allocation table
            html += """
          <h3>JAO Electrical Parameters</h3>
          <p>R: """ + _sf(jao_r_total) + """ Ω (Total) | """ + _sf(jao_r_km) + """ Ω/km</p>
          <p>X: """ + _sf(jao_x_total) + """ Ω (Total) | """ + _sf(jao_x_km) + """ Ω/km</p>
          <p>B: """ + _sf(jao_b_total, '.8f') + """ S (Total) | """ + _sf(jao_b_km, '.8f') + """ S/km</p>

          <h3>Parameter Allocation to PyPSA Lines</h3>
          <table>
            <tr><th>PyPSA ID</th><th>Length (km)</th><th>Circuits</th>
                <th>Allocated R (Ω)</th><th>Allocated X (Ω)</th><th>Allocated B (S)</th><th>Status</th></tr>
"""
            for seg in segs:
                pid = str(seg.get("network_id", ""))
                prow = pypsa_by_id.get(pid)
                if prow is not None:
                    km = meters_to_km(prow.get("length", None)) or 0.0
                    circuits = int(prow.get("circuits", 1) or 1)
                else:
                    km = float(seg.get("length_km", 0) or 0.0)
                    circuits = int(seg.get("num_parallel", 1) or 1)

                html += f"""
            <tr class="{'good-match' if str(seg.get('allocation_status', '')) in ['Applied', 'Parallel Circuit'] else ''}">
              <td>{pid}</td>
              <td>{_sf(km, '.2f')}</td>
              <td>{circuits}</td>
              <td>{_sf(seg.get('allocated_r', None))}</td>
              <td>{_sf(seg.get('allocated_x', None))}</td>
              <td>{_sf(seg.get('allocated_b', None), '.8f')}</td>
              <td>{seg.get('allocation_status', '')}</td>
            </tr>
"""

            # per-km comparison - UPDATED WITH CIRCUIT ADJUSTMENT
            html += """
          </table>
          <h3>Per-Kilometer Comparison</h3>
          <table>
            <tr>
              <th>PyPSA ID</th>
              <th>JAO R (Ω/km)</th><th>PyPSA R (Ω/km)</th><th>R Diff (%)</th>
              <th>JAO X (Ω/km)</th><th>PyPSA X (Ω/km)</th><th>X Diff (%)</th>
              <th>JAO B (S/km)</th><th>PyPSA B (S/km)</th><th>B Diff (%)</th>
            </tr>
"""

            def diff_pct(a, b):
                try:
                    b = float(b)
                    if b == 0:
                        return None
                    return 100.0 * (float(a) - b) / abs(b)
                except Exception:
                    return None

            def cls(d):
                if d is None: return ""
                if abs(d) <= 20: return "good-match"
                if abs(d) <= 50: return "moderate-match"
                return "poor-match"

            for seg in segs:
                pid = str(seg.get("network_id", ""))
                prow = pypsa_by_id.get(pid)
                if prow is not None:
                    km = meters_to_km(prow.get("length", None)) or 0.0
                    r = float(prow.get("r", 0) or 0.0)
                    x = float(prow.get("x", 0) or 0.0)
                    b = float(prow.get("b", 0) or 0.0)
                    circuits = int(prow.get("circuits", 1) or 1)
                    if km > 0:
                        rpk, xpk, bpk = r / km, x / km, b / km
                    else:
                        rpk = xpk = bpk = 0.0
                else:
                    rpk = float(seg.get("original_r_per_km", 0) or 0.0)
                    xpk = float(seg.get("original_x_per_km", 0) or 0.0)
                    bpk = float(seg.get("original_b_per_km", 0) or 0.0)
                    circuits = int(seg.get("num_parallel", 1) or 1)

                # Apply circuit adjustment for comparison
                # For series elements (R, X): divide JAO by circuits
                # For parallel elements (B): multiply JAO by circuits
                jao_r_km_adjusted = jao_r_km / circuits
                jao_x_km_adjusted = jao_x_km / circuits
                jao_b_km_adjusted = jao_b_km * circuits

                # Calculate differences with circuit-adjusted values
                rd = diff_pct(jao_r_km_adjusted, rpk)
                xd = diff_pct(jao_x_km_adjusted, xpk)
                bd = diff_pct(jao_b_km_adjusted, bpk)

                html += f"""
            <tr>
              <td>{pid}</td>
              <td>{_sf(jao_r_km_adjusted)}</td><td>{_sf(rpk)}</td><td class="{cls(rd)}">{_sf(rd, '.2f') if rd is not None else 'N/A'}%</td>
              <td>{_sf(jao_x_km_adjusted)}</td><td>{_sf(xpk)}</td><td class="{cls(xd)}">{_sf(xd, '.2f') if xd is not None else 'N/A'}%</td>
              <td>{_sf(jao_b_km_adjusted, '.8f')}</td><td>{_sf(bpk, '.8f')}</td><td class="{cls(bd)}">{_sf(bd, '.2f') if bd is not None else 'N/A'}%</td>
            </tr>
"""

            # residuals (use fields if present; else compute)
            alloc_r = float(result.get("allocated_r_sum", 0) or 0.0)
            alloc_x = float(result.get("allocated_x_sum", 0) or 0.0)
            alloc_b = float(result.get("allocated_b_sum", 0) or 0.0)

            def res_pct(total, alloc):
                try:
                    if abs(total) < 1e-12:
                        return None
                    return 100.0 * (total - alloc) / abs(total)
                except Exception:
                    return None

            r_res, x_res, b_res = jao_r_total - alloc_r, jao_x_total - alloc_x, jao_b_total - alloc_b
            r_res_pct, x_res_pct, b_res_pct = res_pct(jao_r_total, alloc_r), res_pct(jao_x_total, alloc_x), res_pct(
                jao_b_total, alloc_b)

            def cls_res(v):
                if v is None: return ""
                if abs(v) <= 10: return "good-match"
                if abs(v) <= 30: return "moderate-match"
                return "poor-match"

            html += f"""
          <h3>Parameter Allocation Consistency</h3>
          <table>
            <tr><th>Parameter</th><th>JAO Total</th><th>Sum Allocated</th><th>Residual</th><th>Residual %</th></tr>
            <tr><td>R (Ω)</td><td>{_sf(jao_r_total)}</td><td>{_sf(alloc_r)}</td><td>{_sf(r_res)}</td><td class="{cls_res(r_res_pct)}">{_sf(r_res_pct, '.2f') if r_res_pct is not None else 'N/A'}%</td></tr>
            <tr><td>X (Ω)</td><td>{_sf(jao_x_total)}</td><td>{_sf(alloc_x)}</td><td>{_sf(x_res)}</td><td class="{cls_res(x_res_pct)}">{_sf(x_res_pct, '.2f') if x_res_pct is not None else 'N/A'}%</td></tr>
            <tr><td>B (S)</td><td>{_sf(jao_b_total, '.8f')}</td><td>{_sf(alloc_b, '.8f')}</td><td>{_sf(b_res, '.8f')}</td><td class="{cls_res(b_res_pct)}">{_sf(b_res_pct, '.2f') if b_res_pct is not None else 'N/A'}%</td></tr>
          </table>
"""
        else:
            html += "<p>No parameter allocation data available for this match.</p>"

        html += """
        </div>
      </td>
    </tr>
"""

    html += """
  </table>
</body>
</html>
"""

    # write file
    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / "jao_pypsa_electrical_parameters.html"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Electrical parameters summary saved to {out_file}")
    return out_file


def random_match_quality_check(matches, jao_gdf, pypsa_gdf, output_dir, sample_size=20):
    """
    Randomly sample matched lines and evaluate their quality, generating
    an HTML report with visualizations and statistics.
    """
    import pandas as pd
    import numpy as np
    import random
    from pathlib import Path
    import matplotlib.pyplot as plt
    import io
    import base64
    from shapely.geometry import LineString, MultiLineString
    from datetime import datetime

    print(f"\n===== RANDOM MATCH QUALITY CHECK (n={sample_size}) =====")

    # Filter to matched results only
    matched_results = [m for m in matches if m.get('matched', False)]
    if not matched_results:
        print("No matched lines to evaluate")
        return None

    # Create lookups for quick access - convert to dictionaries to avoid Series truth value issues
    jao_by_id = {}
    for _, row in jao_gdf.iterrows():
        jao_by_id[str(row['id'])] = dict(row)

    pypsa_by_id = {}
    for _, row in pypsa_gdf.iterrows():
        row_id = str(row.get('line_id', row.get('id', '')))
        pypsa_by_id[row_id] = dict(row)

    # Helper to calculate length in km
    def length_km(row):
        try:
            length = float(row.get('length', 0) or 0)
            return length / 1000.0 if length > 1000 else length
        except (TypeError, ValueError):
            return 0

    # Randomly sample matches
    sample_count = min(len(matched_results), sample_size)
    sampled_matches = random.sample(matched_results, sample_count)

    print(f"Sampled {sample_count} matches for quality evaluation")

    # Evaluate each sampled match
    quality_checks = []

    for match in sampled_matches:
        jao_id = str(match.get('jao_id', ''))
        match_quality = match.get('match_quality', '')

        # Get JAO data
        jao_row = jao_by_id.get(jao_id)
        if jao_row is None:
            continue

        jao_name = jao_row.get('NE_name', '')
        jao_voltage = jao_row.get('v_nom', jao_row.get('voltage', 0))
        jao_length = length_km(jao_row)
        jao_geom = jao_row.get('geometry')

        # Get PyPSA data
        pypsa_ids = match.get('pypsa_ids', [])
        if isinstance(pypsa_ids, str):
            pypsa_ids = [pid.strip() for pid in pypsa_ids.split(';') if pid.strip()]

        # Calculate total PyPSA path length
        path_length = 0
        pypsa_segments = []

        for pid in pypsa_ids:
            pypsa_row = pypsa_by_id.get(pid)
            if pypsa_row is not None:
                path_length += length_km(pypsa_row)
                pypsa_segments.append({
                    'id': pid,
                    'length_km': length_km(pypsa_row),
                    'voltage': pypsa_row.get('voltage', pypsa_row.get('v_nom', 0)),
                    'geometry': pypsa_row.get('geometry')
                })

        # Calculate length ratio
        length_ratio = path_length / jao_length if jao_length > 0 else 0

        # Calculate parameter similarity if possible
        param_similarity = {}

        # Calculate spatial metrics
        spatial_metrics = {}
        if jao_geom is not None:
            try:
                # Try to merge PyPSA geometries
                from shapely.ops import linemerge, unary_union
                pypsa_geoms = [s['geometry'] for s in pypsa_segments if s['geometry'] is not None]

                if pypsa_geoms:
                    # Try to merge the geometries
                    merged_geom = linemerge(unary_union(pypsa_geoms))

                    # Calculate Hausdorff distance
                    try:
                        hausdorff_dist = jao_geom.hausdorff_distance(merged_geom)
                        spatial_metrics['hausdorff_distance'] = hausdorff_dist
                    except Exception as e:
                        print(f"Error calculating Hausdorff distance: {e}")

                    # Calculate buffer overlap
                    try:
                        jao_buffer = jao_geom.buffer(0.01)  # ~1km buffer
                        pypsa_buffer = merged_geom.buffer(0.01)

                        overlap_area = jao_buffer.intersection(pypsa_buffer).area
                        union_area = jao_buffer.union(pypsa_buffer).area

                        if union_area > 0:
                            iou = overlap_area / union_area
                            spatial_metrics['iou'] = iou
                    except Exception as e:
                        print(f"Error calculating buffer overlap: {e}")
            except Exception as e:
                print(f"Error calculating spatial metrics for JAO {jao_id}: {e}")

        # Get electrical parameters for comparison
        if len(pypsa_ids) == 1:  # Direct comparison is only meaningful for 1:1 matches
            jao_r = jao_row.get('r_per_km')
            jao_x = jao_row.get('x_per_km')
            jao_b = jao_row.get('b_per_km')

            pypsa_row = pypsa_by_id.get(pypsa_ids[0])
            if pypsa_row is not None:
                # Get PyPSA parameters
                pypsa_r = pypsa_row.get('r_per_km')
                if pypsa_r is None and 'r' in pypsa_row and length_km(pypsa_row) > 0:
                    pypsa_r = pypsa_row.get('r', 0) / length_km(pypsa_row)

                pypsa_x = pypsa_row.get('x_per_km')
                if pypsa_x is None and 'x' in pypsa_row and length_km(pypsa_row) > 0:
                    pypsa_x = pypsa_row.get('x', 0) / length_km(pypsa_row)

                pypsa_b = pypsa_row.get('b_per_km')
                if pypsa_b is None and 'b' in pypsa_row and length_km(pypsa_row) > 0:
                    pypsa_b = pypsa_row.get('b', 0) / length_km(pypsa_row)

                # Calculate parameter differences if possible
                try:
                    if jao_r is not None and pypsa_r is not None and float(jao_r) != 0:
                        r_diff = (float(pypsa_r) - float(jao_r)) / float(jao_r)
                        param_similarity['r_diff'] = r_diff
                except Exception as e:
                    print(f"Error calculating r_diff: {e}")

                try:
                    if jao_x is not None and pypsa_x is not None and float(jao_x) != 0:
                        x_diff = (float(pypsa_x) - float(jao_x)) / float(jao_x)
                        param_similarity['x_diff'] = x_diff
                except Exception as e:
                    print(f"Error calculating x_diff: {e}")

                try:
                    if jao_b is not None and pypsa_b is not None and float(jao_b) != 0:
                        b_diff = (float(pypsa_b) - float(jao_b)) / float(jao_b)
                        param_similarity['b_diff'] = b_diff
                except Exception as e:
                    print(f"Error calculating b_diff: {e}")

        # Calculate overall quality score
        score_components = []

        # 1. Length ratio component (ideal: 1.0, range: 0-1)
        length_score = max(0, 1 - abs(length_ratio - 1))
        score_components.append(('length', length_score))

        # 2. Spatial component from IoU if available (ideal: 1.0, range: 0-1)
        spatial_score = spatial_metrics.get('iou', 0.5)  # Default to neutral if not available
        score_components.append(('spatial', spatial_score))

        # 3. Parameter similarity if available (ideal: 0 difference, range: 0-1)
        param_score = 0.5  # Default to neutral
        if param_similarity:
            # Average the absolute differences, capped at 100%
            diffs = [min(1.0, abs(d)) for d in param_similarity.values()]
            if diffs:
                param_score = 1.0 - sum(diffs) / len(diffs)
        score_components.append(('parameters', param_score))

        # Calculate weighted score
        weights = {'length': 0.4, 'spatial': 0.4, 'parameters': 0.2}
        overall_score = sum(weights[c] * s for c, s in score_components)

        # Determine quality rating
        if overall_score >= 0.85:
            quality_rating = "Excellent"
        elif overall_score >= 0.7:
            quality_rating = "Good"
        elif overall_score >= 0.5:
            quality_rating = "Fair"
        else:
            quality_rating = "Poor"

        # Collect all metrics
        quality_checks.append({
            'jao_id': jao_id,
            'jao_name': jao_name,
            'jao_voltage': jao_voltage,
            'jao_length_km': jao_length,
            'pypsa_ids': pypsa_ids,
            'pypsa_count': len(pypsa_ids),
            'pypsa_path_length': path_length,
            'length_ratio': length_ratio,
            'match_quality': match_quality,
            'spatial_metrics': spatial_metrics,
            'param_similarity': param_similarity,
            'score_components': score_components,
            'overall_score': overall_score,
            'quality_rating': quality_rating
        })

    if not quality_checks:
        print("No valid quality checks could be performed")
        return None

    # Create summary statistics
    quality_counts = {}
    for check in quality_checks:
        rating = check['quality_rating']
        quality_counts[rating] = quality_counts.get(rating, 0) + 1

    # Calculate averages
    avg_score = sum(c['overall_score'] for c in quality_checks) / len(quality_checks) if quality_checks else 0
    avg_length_ratio = sum(c['length_ratio'] for c in quality_checks) / len(quality_checks) if quality_checks else 0

    # Generate charts for the report (as base64 embedded images)
    def create_pie_chart():
        plt.figure(figsize=(6, 4))
        labels = list(quality_counts.keys())
        sizes = list(quality_counts.values())
        colors = {'Excellent': '#4CAF50', 'Good': '#8BC34A', 'Fair': '#FFC107', 'Poor': '#FF5722'}
        plt_colors = [colors.get(label, '#CCCCCC') for label in labels]

        plt.pie(sizes, labels=labels, colors=plt_colors, autopct='%1.1f%%', startangle=90)
        plt.axis('equal')
        plt.title('Match Quality Distribution')

        # Save to base64 string
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()

        return base64.b64encode(image_png).decode('utf-8')

    def create_length_ratio_histogram():
        plt.figure(figsize=(8, 4))
        ratios = [c['length_ratio'] for c in quality_checks]

        plt.hist(ratios, bins=10, alpha=0.7, color='#2196F3')
        plt.axvline(x=1.0, color='r', linestyle='--', label='Ideal Ratio (1.0)')
        plt.xlabel('Length Ratio (PyPSA Path / JAO Line)')
        plt.ylabel('Count')
        plt.title('Distribution of Length Ratios')
        plt.legend()

        # Save to base64 string
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()

        return base64.b64encode(image_png).decode('utf-8')

    # Generate pie chart and histogram
    try:
        pie_chart_data = create_pie_chart()
        histogram_data = create_length_ratio_histogram()
    except Exception as e:
        print(f"Error generating charts: {e}")
        pie_chart_data = ""
        histogram_data = ""

    # Create HTML report
    html_path = Path(output_dir) / f'match_quality_check_{datetime.now().strftime("%Y%m%d_%H%M%S")}.html'

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Match Quality Random Check</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
            h1, h2, h3 {{ color: #333; }}
            .container {{ max-width: 1200px; margin: 0 auto; }}
            .summary {{ background-color: #f9f9f9; padding: 20px; border-radius: 5px; margin-bottom: 20px; }}
            .charts {{ display: flex; justify-content: space-between; margin-bottom: 20px; }}
            .chart {{ background: white; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
            table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; position: sticky; top: 0; }}
            .excellent {{ background-color: #E8F5E9; }}
            .good {{ background-color: #F1F8E9; }}
            .fair {{ background-color: #FFF8E1; }}
            .poor {{ background-color: #FFEBEE; }}
            .filters {{ margin-bottom: 20px; }}
            input[type=text] {{ padding: 8px; width: 250px; }}
            .score-bar {{ width: 100px; height: 15px; background: #eee; position: relative; }}
            .score-fill {{ height: 100%; background: linear-gradient(to right, #FF5722, #FFC107, #8BC34A, #4CAF50); position: absolute; top: 0; left: 0; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Match Quality Random Check</h1>

            <div class="summary">
                <h2>Summary</h2>
                <p>Random sample of {len(quality_checks)} matches evaluated on {datetime.now().strftime("%Y-%m-%d %H:%M")}</p>
                <p><strong>Average quality score:</strong> {avg_score:.2f} out of 1.0</p>
                <p><strong>Average length ratio:</strong> {avg_length_ratio:.2f}</p>

                <h3>Quality Distribution:</h3>
                <ul>
    """

    for rating, count in quality_counts.items():
        percentage = 100 * count / len(quality_checks) if quality_checks else 0
        html_content += f"<li><strong>{rating}:</strong> {count} matches ({percentage:.1f}%)</li>\n"

    html_content += """
                </ul>
            </div>
    """

    if pie_chart_data and histogram_data:
        html_content += """
            <div class="charts">
                <div class="chart">
                    <img src="data:image/png;base64,""" + pie_chart_data + """" alt="Quality Distribution">
                </div>
                <div class="chart">
                    <img src="data:image/png;base64,""" + histogram_data + """" alt="Length Ratio Distribution">
                </div>
            </div>
        """

    html_content += """
            <div class="filters">
                <input type="text" id="searchInput" onkeyup="filterTable()" placeholder="Search for IDs, quality ratings...">
            </div>

            <table id="resultsTable">
                <tr>
                    <th>JAO ID</th>
                    <th>JAO Name</th>
                    <th>Voltage</th>
                    <th>JAO Length</th>
                    <th>PyPSA Segments</th>
                    <th>Length Ratio</th>
                    <th>Quality Score</th>
                    <th>Rating</th>
                    <th>Match Type</th>
                </tr>
    """

    for check in quality_checks:
        rating_class = check['quality_rating'].lower()
        score_percentage = int(check['overall_score'] * 100)

        html_content += f"""
            <tr class="{rating_class}">
                <td>{check['jao_id']}</td>
                <td>{check['jao_name']}</td>
                <td>{check['jao_voltage']} kV</td>
                <td>{check['jao_length_km']:.2f} km</td>
                <td>{len(check['pypsa_ids'])} ({', '.join(check['pypsa_ids'][:2]) + ('...' if len(check['pypsa_ids']) > 2 else '')})</td>
                <td>{check['length_ratio']:.2f}</td>
                <td>
                    <div class="score-bar">
                        <div class="score-fill" style="width: {score_percentage}%;"></div>
                    </div>
                    {check['overall_score']:.2f}
                </td>
                <td>{check['quality_rating']}</td>
                <td>{check['match_quality']}</td>
            </tr>
        """

    html_content += """
            </table>

            <script>
                function filterTable() {
                    var input = document.getElementById("searchInput");
                    var filter = input.value.toLowerCase();
                    var table = document.getElementById("resultsTable");
                    var tr = table.getElementsByTagName("tr");

                    for (var i = 1; i < tr.length; i++) {
                        var td = tr[i].getElementsByTagName("td");
                        var found = false;

                        for (var j = 0; j < td.length; j++) {
                            if (td[j].innerHTML.toLowerCase().indexOf(filter) > -1) {
                                found = true;
                                break;
                            }
                        }

                        if (found) {
                            tr[i].style.display = "";
                        } else {
                            tr[i].style.display = "none";
                        }
                    }
                }
            </script>
        </div>
    </body>
    </html>
    """

    with open(html_path, 'w') as f:
        f.write(html_content)

    print(f"Random quality check report saved to {html_path}")

    # Also save raw data to CSV for further analysis
    try:
        csv_path = Path(output_dir) / f'match_quality_check_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'

        # Flatten data for CSV
        csv_data = []
        for check in quality_checks:
            row = {
                'jao_id': check['jao_id'],
                'jao_name': check['jao_name'],
                'jao_voltage': check['jao_voltage'],
                'jao_length_km': check['jao_length_km'],
                'pypsa_count': check['pypsa_count'],
                'pypsa_ids': ';'.join(check['pypsa_ids']),
                'pypsa_path_length': check['pypsa_path_length'],
                'length_ratio': check['length_ratio'],
                'overall_score': check['overall_score'],
                'quality_rating': check['quality_rating'],
                'match_quality': check['match_quality']
            }

            # Add spatial metrics
            for key, value in check['spatial_metrics'].items():
                row[f'spatial_{key}'] = value

            # Add parameter similarity
            for key, value in check['param_similarity'].items():
                row[f'param_{key}'] = value

            csv_data.append(row)

        pd.DataFrame(csv_data).to_csv(csv_path, index=False)
        print(f"Quality check data saved to {csv_path}")
    except Exception as e:
        print(f"Error saving CSV data: {e}")

    return html_path



def export_unmatched_pypsa_details(matches, jao_gdf, pypsa_gdf, output_dir):
    """
    Analyze and export details about unmatched PyPSA lines including likely reasons
    for not being matched.

    Parameters:
    -----------
    matches : list
        List of match dictionaries from the matching process
    jao_gdf : GeoDataFrame
        GeoDataFrame containing JAO lines
    pypsa_gdf : GeoDataFrame
        GeoDataFrame containing PyPSA lines
    output_dir : Path
        Directory to save output files

    Returns:
    --------
    Path
        Path to the output file
    """
    import pandas as pd
    from pathlib import Path
    import numpy as np
    from shapely.geometry import Point

    print("\n===== ANALYZING UNMATCHED PYPSA LINES =====")

    # Get all matched PyPSA IDs
    matched_pypsa_ids = set()
    for match in matches:
        if match.get('matched', False):
            pypsa_ids = match.get('pypsa_ids', [])
            if isinstance(pypsa_ids, str):
                pypsa_ids = [pid.strip() for pid in pypsa_ids.split(';') if pid.strip()]
            matched_pypsa_ids.update(pypsa_ids)

    print(f"Found {len(matched_pypsa_ids)} matched PyPSA lines")

    # Find unmatched PyPSA lines
    unmatched_pypsa = []
    for _, row in pypsa_gdf.iterrows():
        pypsa_id = str(row.get('line_id', row.get('id', '')))
        if pypsa_id not in matched_pypsa_ids:
            unmatched_pypsa.append(row)

    print(f"Found {len(unmatched_pypsa)} unmatched PyPSA lines")

    if not unmatched_pypsa:
        print("No unmatched PyPSA lines to analyze")
        return None

    # Function to safely get length in km
    def length_km(row):
        try:
            length = float(row.get('length', 0))
            return length / 1000.0 if length > 1000 else length
        except (TypeError, ValueError):
            return 0

    # Analyze unmatched lines
    results = []
    for row in unmatched_pypsa:
        pypsa_id = str(row.get('line_id', row.get('id', '')))
        voltage = row.get('voltage', row.get('v_nom', 0))
        bus0 = row.get('bus0', '')
        bus1 = row.get('bus1', '')
        length = length_km(row)
        geom = row.geometry

        # Determine likely reasons for not matching
        reasons = []

        # Reason 1: Voltage - non-standard voltage classes are harder to match
        if voltage not in [220, 380, 400]:
            reasons.append("Non-standard voltage class")

        # Reason 2: Length - very short lines often don't have matches
        if length < 1.0:
            reasons.append("Very short line (<1km)")

        # Reason 3: Isolated - not connected to main network
        if not (bus0 and bus1):
            reasons.append("Missing bus connections")

        # Reason 4: Geographic isolation - check if there are JAO lines nearby
        has_nearby_jao = False
        if geom is not None:
            buffer_dist = 0.01  # ~1km in degrees
            buffer = geom.buffer(buffer_dist)
            for _, jao_row in jao_gdf.iterrows():
                if jao_row.geometry is not None and buffer.intersects(jao_row.geometry):
                    has_nearby_jao = True
                    break

        if not has_nearby_jao:
            reasons.append("No nearby JAO lines")

        # Reason 5: Already at circuit capacity
        # (This would require tracking which PyPSA lines were considered but rejected)

        # If no specific reasons identified
        if not reasons:
            if length > 20:
                reasons.append("Long line, possible cross-border or in area not covered by JAO")
            else:
                reasons.append("Unknown (may be duplicate path or parallel circuit)")

        results.append({
            'pypsa_id': pypsa_id,
            'voltage': voltage,
            'bus0': bus0,
            'bus1': bus1,
            'length_km': length,
            'reasons_unmatched': '; '.join(reasons)
        })

    # Create DataFrame and save to CSV
    df = pd.DataFrame(results)

    # Sort by voltage then length
    df = df.sort_values(['voltage', 'length_km'], ascending=[False, False])

    # Save to CSV
    output_path = Path(output_dir) / 'unmatched_pypsa_analysis.csv'
    df.to_csv(output_path, index=False)

    # Generate HTML report
    html_path = Path(output_dir) / 'unmatched_pypsa_analysis.html'

    # Group by reason for summary
    reason_counts = {}
    for r in results:
        reasons = r['reasons_unmatched'].split('; ')
        for reason in reasons:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

    # Sort reasons by frequency
    sorted_reasons = sorted(reason_counts.items(), key=lambda x: x[1], reverse=True)

    # Create HTML content with advanced filtering and sorting
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Unmatched PyPSA Lines Analysis</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
            h1, h2 {{ color: #333; }}
            table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; position: sticky; top: 0; }}
            .summary {{ background-color: #f9f9f9; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
            .filters {{ margin-bottom: 20px; display: flex; flex-wrap: wrap; gap: 10px; }}
            .filter-group {{ display: flex; flex-direction: column; margin-right: 15px; }}
            .filter-group label {{ font-weight: bold; margin-bottom: 5px; }}
            input[type=text], select {{ padding: 8px; width: 180px; }}
            th.sortable {{ cursor: pointer; }}
            th.sortable::after {{ content: "⇅"; margin-left: 5px; color: #999; }}
            th.sort-asc::after {{ content: "↑"; color: #333; }}
            th.sort-desc::after {{ content: "↓"; color: #333; }}
            .table-container {{ max-height: 70vh; overflow-y: auto; }}
            .hidden {{ display: none; }}
        </style>
    </head>
    <body>
        <h1>Unmatched PyPSA Lines Analysis</h1>

        <div class="summary">
            <h2>Summary</h2>
            <p>Total unmatched PyPSA lines: {len(unmatched_pypsa)}</p>

            <h3>Reasons for Unmatched Status:</h3>
            <ul>
    """

    for reason, count in sorted_reasons:
        percentage = 100 * count / len(unmatched_pypsa)
        html_content += f"<li><strong>{reason}</strong>: {count} lines ({percentage:.1f}%)</li>\n"

    html_content += """
            </ul>
        </div>

        <div class="filters">
            <div class="filter-group">
                <label for="filterPyPSAID">PyPSA ID</label>
                <input type="text" id="filterPyPSAID" placeholder="Filter PyPSA ID">
            </div>
            <div class="filter-group">
                <label for="filterVoltage">Voltage (kV)</label>
                <input type="text" id="filterVoltage" placeholder="Filter voltage">
            </div>
            <div class="filter-group">
                <label for="filterBus0">Bus0</label>
                <input type="text" id="filterBus0" placeholder="Filter bus0">
            </div>
            <div class="filter-group">
                <label for="filterBus1">Bus1</label>
                <input type="text" id="filterBus1" placeholder="Filter bus1">
            </div>
            <div class="filter-group">
                <label for="filterLength">Length (km)</label>
                <input type="text" id="filterLength" placeholder="Filter length">
            </div>
            <div class="filter-group">
                <label for="filterReasons">Reasons</label>
                <input type="text" id="filterReasons" placeholder="Filter reasons">
            </div>
            <div class="filter-group">
                <label>&nbsp;</label>
                <button onclick="clearFilters()">Clear Filters</button>
            </div>
        </div>

        <div class="table-container">
            <table id="resultsTable">
                <thead>
                    <tr>
                        <th class="sortable" onclick="sortTable(0)">PyPSA ID</th>
                        <th class="sortable" onclick="sortTable(1)">Voltage (kV)</th>
                        <th class="sortable" onclick="sortTable(2)">Bus0</th>
                        <th class="sortable" onclick="sortTable(3)">Bus1</th>
                        <th class="sortable" onclick="sortTable(4)">Length (km)</th>
                        <th class="sortable" onclick="sortTable(5)">Reasons Unmatched</th>
                    </tr>
                </thead>
                <tbody>
    """

    for r in results:
        html_content += f"""
                    <tr>
                        <td>{r['pypsa_id']}</td>
                        <td>{r['voltage']}</td>
                        <td>{r['bus0']}</td>
                        <td>{r['bus1']}</td>
                        <td>{r['length_km']:.2f}</td>
                        <td>{r['reasons_unmatched']}</td>
                    </tr>
        """

    html_content += """
                </tbody>
            </table>
        </div>

        <script>
            // Column filtering functionality
            const filters = {
                'filterPyPSAID': 0,
                'filterVoltage': 1,
                'filterBus0': 2,
                'filterBus1': 3,
                'filterLength': 4,
                'filterReasons': 5
            };

            // Add event listeners to all filter inputs
            for (const [filterId, colIndex] of Object.entries(filters)) {
                document.getElementById(filterId).addEventListener('input', function() {
                    filterTable();
                });
            }

            function filterTable() {
                const table = document.getElementById('resultsTable');
                const rows = table.getElementsByTagName('tr');

                // Skip header row
                for (let i = 1; i < rows.length; i++) {
                    let showRow = true;
                    const row = rows[i];

                    // Check each filter
                    for (const [filterId, colIndex] of Object.entries(filters)) {
                        const filterValue = document.getElementById(filterId).value.toLowerCase();
                        if (filterValue) {
                            const cell = row.getElementsByTagName('td')[colIndex];
                            const cellText = cell.textContent.toLowerCase();

                            if (cellText.indexOf(filterValue) === -1) {
                                showRow = false;
                                break;
                            }
                        }
                    }

                    // Show or hide row
                    row.style.display = showRow ? '' : 'none';
                }
            }

            function clearFilters() {
                for (const filterId of Object.keys(filters)) {
                    document.getElementById(filterId).value = '';
                }
                filterTable();
            }

            // Sorting functionality
            let currentSortCol = -1;
            let sortAscending = true;

            function sortTable(colIndex) {
                const table = document.getElementById('resultsTable');
                const tbody = table.getElementsByTagName('tbody')[0];
                const rows = Array.from(tbody.getElementsByTagName('tr'));

                // Update sort direction
                if (currentSortCol === colIndex) {
                    sortAscending = !sortAscending;
                } else {
                    sortAscending = true;

                    // Reset all column headers
                    const headers = table.getElementsByTagName('th');
                    for (let i = 0; i < headers.length; i++) {
                        headers[i].classList.remove('sort-asc', 'sort-desc');
                    }
                }

                // Update sort indicator
                const header = table.getElementsByTagName('th')[colIndex];
                header.classList.remove('sort-asc', 'sort-desc');
                header.classList.add(sortAscending ? 'sort-asc' : 'sort-desc');

                currentSortCol = colIndex;

                // Sort rows
                rows.sort((a, b) => {
                    let valueA = a.getElementsByTagName('td')[colIndex].textContent;
                    let valueB = b.getElementsByTagName('td')[colIndex].textContent;

                    // Handle numeric values (voltage, length)
                    if (colIndex === 1 || colIndex === 4) {
                        valueA = parseFloat(valueA) || 0;
                        valueB = parseFloat(valueB) || 0;
                        return sortAscending ? valueA - valueB : valueB - valueA;
                    } 
                    // Text values
                    else {
                        return sortAscending 
                            ? valueA.localeCompare(valueB) 
                            : valueB.localeCompare(valueA);
                    }
                });

                // Reorder rows
                rows.forEach(row => tbody.appendChild(row));
            }

            // Initial sort by voltage (descending)
            document.addEventListener('DOMContentLoaded', function() {
                sortTable(1);
                sortTable(1); // Toggle to descending
            });
        </script>
    </body>
    </html>
    """

    with open(html_path, 'w') as f:
        f.write(html_content)

    print(f"Saved unmatched PyPSA analysis to {output_path} and {html_path}")
    return html_path
