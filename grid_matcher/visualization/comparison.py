def calculate_difference_percentage(original, allocated):
    """
    calculate the percentage difference between original and allocated values.

    Returns a percentage value between -100% and infinity where:
    - 0% means identical values
    - 100% means allocated is double the original
    - -50% means allocated is half the original
    """
    if original == 0:
        if allocated == 0:
            return 0  # Both zero means no difference
        else:
            return 100  # Original is zero, allocated is non-zero

    # Calculate percentage difference
    percent_diff = ((allocated - original) / original) * 100

    return percent_diff


def prepare_visualization_data(matching_results, pypsa_gdf, jao_gdf=None, use_demo_jao_values=False):
    """
    Transform matching results into the format expected by the parameter comparison visualization.
    """
    print("Preparing visualization data...")

    # Create a lookup for JAO parameters
    jao_lookup = {}
    if jao_gdf is not None:
        for _, row in jao_gdf.iterrows():
            jao_id = str(row['id'])
            jao_lookup[jao_id] = {
                'r': float(row.get('r', 0) or 0),
                'x': float(row.get('x', 0) or 0),
                'b': float(row.get('b', 0) or 0),
                'length_km': float(row.get('length_km', row.get('length', 0)) or 0)
            }
        print(f"Created JAO lookup table with {len(jao_lookup)} entries")

    # Create enhanced results with the expected structure
    enhanced_results = []

    # Count how many results have matched_lines_data with allocated parameters
    has_allocation_data = 0
    missing_allocation_data = 0

    for result in matching_results:
        if not result.get('matched', False):
            continue

        # IMPORTANT: Check if result already has matched_lines_data with allocated parameters
        # This happens for manual matches which already have allocated parameters
        segs = result.get('matched_lines_data', [])
        has_all_params = all('allocated_r' in seg for seg in segs) if segs else False

        if has_all_params:
            has_allocation_data += 1
            # Already has properly allocated parameters, use it directly
            enhanced_result = {
                'jao_id': result.get('jao_id', ''),
                'matched': True,
                'jao_r': result.get('jao_r', 0),
                'jao_x': result.get('jao_x', 0),
                'jao_b': result.get('jao_b', 0),
                'jao_length_km': result.get('jao_length_km', 0),
                'matched_lines_data': segs
            }
            enhanced_results.append(enhanced_result)
            continue

        missing_allocation_data += 1
        # Standard processing for results without allocated parameters
        # Get JAO ID and length
        jao_id = str(result.get('jao_id', ''))
        jao_length_km = float(result.get('jao_length_km', result.get('jao_length', 0)) or 0)

        # Get JAO parameters
        jao_r = float(result.get('jao_r', 0) or 0)
        jao_x = float(result.get('jao_x', 0) or 0)
        jao_b = float(result.get('jao_b', 0) or 0)

        # Try lookup if parameters are missing
        if (jao_r == 0 and jao_x == 0 and jao_b == 0) and jao_id in jao_lookup:
            jao_data = jao_lookup[jao_id]
            jao_r = jao_data['r']
            jao_x = jao_data['x']
            jao_b = jao_data['b']
            if jao_length_km <= 0:
                jao_length_km = jao_data['length_km']

        # Get matched PyPSA IDs
        pypsa_ids = result.get('pypsa_ids', [])
        if not isinstance(pypsa_ids, list):
            pypsa_ids = [pypsa_ids]

        if not pypsa_ids:
            continue

        # Create matched_lines_data array expected by visualization
        matched_lines_data = []
        total_pypsa_r = 0
        total_pypsa_x = 0
        total_pypsa_b = 0
        total_pypsa_length = 0

        for pypsa_id in pypsa_ids:
            # Find the pypsa line in the dataframe
            matching_rows = pypsa_gdf[pypsa_gdf['id'].astype(str) == str(pypsa_id)]
            if len(matching_rows) == 0 and 'line_id' in pypsa_gdf.columns:
                matching_rows = pypsa_gdf[pypsa_gdf['line_id'].astype(str) == str(pypsa_id)]

            if len(matching_rows) == 0:
                continue

            pypsa_row = matching_rows.iloc[0]

            # Extract and convert parameters
            try:
                r = float(pypsa_row.get('r', 0) or 0)
                x = float(pypsa_row.get('x', 0) or 0)
                b = float(pypsa_row.get('b', 0) or 0)
                length = float(pypsa_row.get('length', 0) or 0)
                circuits = int(pypsa_row.get('circuits', 1) or 1)

                total_pypsa_r += r
                total_pypsa_x += x
                total_pypsa_b += b
                total_pypsa_length += length

                # Create line data entry
                matched_line = {
                    'network_id': pypsa_id,
                    'length_km': length,
                    'num_parallel': circuits,
                    'allocation_status': 'Applied',
                    'original_r': r,
                    'original_x': x,
                    'original_b': b
                }
                matched_lines_data.append(matched_line)
            except Exception as e:
                print(f"Error processing PyPSA ID {pypsa_id}: {str(e)}")

        # Only process if we have lines with data and valid JAO parameters
        if matched_lines_data and (jao_r > 0 or jao_x > 0 or jao_b > 0 or use_demo_jao_values):
            # Use demo values if needed and allowed
            if (jao_r == 0 and jao_x == 0 and jao_b == 0) and use_demo_jao_values:
                import random
                variation_factor = lambda: random.uniform(0.8, 1.2)
                jao_r = total_pypsa_r * variation_factor()
                jao_x = total_pypsa_x * variation_factor()
                jao_b = total_pypsa_b * variation_factor()

                if jao_length_km <= 0:
                    jao_length_km = total_pypsa_length / 1000

                jao_id = f"{jao_id} (demo values)"

            # Skip if still no valid parameters
            if jao_r == 0 and jao_x == 0 and jao_b == 0:
                continue

            # Calculate per-km values
            jao_r_per_km = jao_r / jao_length_km if jao_length_km > 0 else 0
            jao_x_per_km = jao_x / jao_length_km if jao_length_km > 0 else 0
            jao_b_per_km = jao_b / jao_length_km if jao_length_km > 0 else 0

            # Allocate parameters to each line
            for line in matched_lines_data:
                circuits = line.get('num_parallel', 1)

                # Store allocated parameters
                line['allocated_r'] = jao_r
                line['allocated_x'] = jao_x
                line['allocated_b'] = jao_b

                # Store per-km values
                line['allocated_r_per_km'] = jao_r_per_km
                line['allocated_x_per_km'] = jao_x_per_km
                line['allocated_b_per_km'] = jao_b_per_km

            # Create the enhanced result
            enhanced_result = {
                'jao_id': jao_id,
                'matched': True,
                'jao_r': jao_r,
                'jao_x': jao_x,
                'jao_b': jao_b,
                'jao_length_km': jao_length_km,
                'matched_lines_data': matched_lines_data
            }
            enhanced_results.append(enhanced_result)

    print(f"Created {len(enhanced_results)} enhanced results with line data")
    print(f"Results with pre-allocated parameters: {has_allocation_data}")
    print(f"Results needing parameter allocation: {missing_allocation_data}")

    return enhanced_results
def visualize_parameter_comparison(matching_results, pypsa_gdf, output_dir="output"):
    """
    Builds a self-contained HTML report comparing original (PyPSA) and allocated (JAO) parameters.

    Keeps:
      • Scientific-notation tick labels,
      • Larger charts,
      • y=x identity line,
      • Chart legends hidden (cleaner UI).

    New:
      • When downloading PNGs, a drawn legend is added on the image:
        - good match (|Δ| ≤ 20%)
        - moderate (20–50%, JAO > PyPSA)
        - large positive Δ (JAO ≫ PyPSA)
        - negative Δ (JAO < PyPSA)

    Returns: str path to the generated HTML file.
    """
    import json
    from pathlib import Path
    import pandas as pd
    from scipy.stats import pearsonr

    print("Creating parameter comparison visualizations with PNG legend overlay...")

    # ---------- helpers ----------
    def pct_diff(a, b):
        try:
            b = float(b); a = float(a)
        except Exception:
            return None
        if abs(b) < 1e-9:
            return None
        return ((a - b) / abs(b)) * 100.0

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- fast lookup from PyPSA gdf ----------
    pypsa_lookup = {}
    for _, row in pypsa_gdf.iterrows():
        key = str(row.get("line_id", row.get("id", "")))
        if key:
            pypsa_lookup[key] = row

    # ---------- build comparison rows ----------
    rows = []
    for res in matching_results:
        if not res.get("matched"):
            continue
        segs = res.get("matched_lines_data") or []
        if not segs:
            continue

        jao_id = res.get("jao_id", "unknown")
        jao_len_km = float(res.get("jao_length_km") or 0.0)
        jao_r_tot = float(res.get("jao_r") or 0.0)
        jao_x_tot = float(res.get("jao_x") or 0.0)
        jao_b_tot = float(res.get("jao_b") or 0.0)

        jao_r_km = jao_r_tot / jao_len_km if jao_len_km > 0 else 0.0
        jao_x_km = jao_x_tot / jao_len_km if jao_len_km > 0 else 0.0
        jao_b_km = jao_b_tot / jao_len_km if jao_len_km > 0 else 0.0

        for seg in segs:
            if seg.get("allocation_status") not in ("Applied", "Parallel Circuit"):
                continue

            pid = str(seg.get("network_id", ""))
            p_row = pypsa_lookup.get(pid)
            if p_row is None:
                continue

            length_km = float(seg.get("length_km") or 0.0)
            length_km_conv = (length_km / 1000.0) if length_km > 1000 else length_km
            circuits = int(seg.get("num_parallel") or 1)

            a_r = float(seg.get("allocated_r") or 0.0)
            a_x = float(seg.get("allocated_x") or 0.0)
            a_b = float(seg.get("allocated_b") or 0.0)

            o_r = float(p_row.get("r") or 0.0)
            o_x = float(p_row.get("x") or 0.0)
            o_b = float(p_row.get("b") or 0.0)

            o_r_km = (o_r / length_km_conv) if length_km_conv > 0 else 0.0
            o_x_km = (o_x / length_km_conv) if length_km_conv > 0 else 0.0
            o_b_km = (o_b / length_km_conv) if length_km_conv > 0 else 0.0

            j_r_km_adj = jao_r_km / circuits if circuits > 0 else 0.0
            j_x_km_adj = jao_x_km / circuits if circuits > 0 else 0.0
            j_b_km_adj = jao_b_km * circuits

            rows.append({
                "jao_id": jao_id,
                "pypsa_id": pid,
                "length_km": float(length_km),
                "circuits": circuits,
                # totals
                "original_r": o_r, "allocated_r": a_r, "diff_r_pct": pct_diff(a_r, o_r),
                "original_x": o_x, "allocated_x": a_x, "diff_x_pct": pct_diff(a_x, o_x),
                "original_b": o_b, "allocated_b": a_b, "diff_b_pct": pct_diff(a_b, o_b),
                # per-km
                "original_r_per_km": o_r_km, "original_x_per_km": o_x_km, "original_b_per_km": o_b_km,
                "jao_r_km_adjusted": j_r_km_adj, "jao_x_km_adjusted": j_x_km_adj, "jao_b_km_adjusted": j_b_km_adj,
                "diff_r_pct_km": pct_diff(j_r_km_adj, o_r_km),
                "diff_x_pct_km": pct_diff(j_x_km_adj, o_x_km),
                "diff_b_pct_km": pct_diff(j_b_km_adj, o_b_km),
            })

    df = pd.DataFrame(rows)

    # ---------- correlations for titles ----------
    r_vals = {}
    for p in ("r", "x", "b"):
        if len(df) >= 2:
            oc, ac = f"original_{p}", f"allocated_{p}"
            mask = df[oc].astype(float) > 0
            mask &= df[ac].astype(float) > 0
            if mask.sum() >= 2:
                r, _ = pearsonr(df.loc[mask, oc], df.loc[mask, ac])
                r_vals[p] = round(float(r), 4)
                continue
        r_vals[p] = None

    # ---------- HTML ----------
    html = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Electrical Parameter Comparison</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
  body{font-family:Arial,Helvetica,sans-serif;background:#f5f5f5;margin:20px;}
  h1,h2{color:#333}
  .container{background:#fff;border-radius:8px;box-shadow:0 2px 10px rgba(0,0,0,.08);padding:18px;margin:18px 0;}
  .chart-container{position:relative;width:100%;height:560px;}
  .download-bar{margin-top:8px;text-align:right}
  .download-btn{border:0;border-radius:6px;padding:8px 12px;cursor:pointer;background:#4CAF50;color:#fff}
  .section-divider{border-top:2px solid #4CAF50;margin:30px 0 10px;padding-top:6px}
  table{border-collapse:collapse;width:100%;margin-top:12px}
  th,td{border:1px solid #ddd;padding:6px 8px;text-align:left}
  th{background:#4CAF50;color:#fff;position:sticky;top:0}
  .filters input{width:90%;padding:5px;border:1px solid #ddd}
  .good-match{background:#c8e6c9}
  .moderate-match{background:#fff9c4}
  .poor-match{background:#ffccbc}
  .data-table-container{max-height:600px;overflow:auto}
  .tabs{display:flex;margin-bottom:10px}
  .tab{padding:8px 14px;background:#ddd;margin-right:6px;border-radius:6px 6px 0 0;cursor:pointer}
  .tab.active{background:#4CAF50;color:#fff}
  .note{font-style:italic;color:#555;background:#f9f9f9;border-left:4px solid #4CAF50;padding:10px;margin:10px 0}
  .legend{font-size:14px;color:#333;background:#f9f9f9;border-left:4px solid #4CAF50;padding:10px;margin:10px 0;border-radius:4px}
  .swatch{display:inline-block;width:14px;height:10px;border:1px solid #ccc;border-radius:2px;vertical-align:middle;margin:0 6px 0 10px}
</style>
</head>
<body>
  <h1>Electrical Parameter Comparison</h1>

  <div class="note">
    <strong>Note:</strong> Per-km charts use circuit-adjusted values:
    <ul>
      <li>Series terms (R, X): JAO per-km ÷ number of circuits</li>
      <li>Shunt term (B): JAO per-km × number of circuits</li>
    </ul>
  </div>

  <h2 class="section-divider">Total Parameter Comparison</h2>

  <div class="container">
    <h2>Resistance (R) — Totals</h2>
    <div class="chart-container"><canvas id="r_tot"></canvas></div>
    <div class="download-bar"><button class="download-btn" data-target="r_tot">Download PNG</button></div>
  </div>

  <div class="legend">
    <strong>Color encoding for points (Δ% = 100·(JAO − PyPSA)/|PyPSA|):</strong>
    <span class="swatch" style="background:#4CAF50"></span> good match (|Δ| ≤ 20%)
    <span class="swatch" style="background:#ffb74d"></span> moderate (20–50%, JAO &gt; PyPSA)
    <span class="swatch" style="background:#d32f2f"></span> large positive Δ (JAO ≫ PyPSA)
    <span class="swatch" style="background:#1976D2"></span> negative Δ (JAO &lt; PyPSA)
  </div>

  <div class="container">
    <h2>Reactance (X) — Totals</h2>
    <div class="chart-container"><canvas id="x_tot"></canvas></div>
    <div class="download-bar"><button class="download-btn" data-target="x_tot">Download PNG</button></div>
  </div>

  <div class="container">
    <h2>Susceptance (B) — Totals</h2>
    <div class="chart-container"><canvas id="b_tot"></canvas></div>
    <div class="download-bar"><button class="download-btn" data-target="b_tot">Download PNG</button></div>
  </div>

  <h2 class="section-divider">Per-km Parameter Comparison (Circuit-adjusted)</h2>

  <div class="container">
    <h2>R (Ω/km) — Per-km</h2>
    <div class="chart-container"><canvas id="r_km"></canvas></div>
    <div class="download-bar"><button class="download-btn" data-target="r_km">Download PNG</button></div>
  </div>

  <div class="container">
    <h2>X (Ω/km) — Per-km</h2>
    <div class="chart-container"><canvas id="x_km"></canvas></div>
    <div class="download-bar"><button class="download-btn" data-target="x_km">Download PNG</button></div>
  </div>

  <div class="container">
    <h2>B (S/km) — Per-km</h2>
    <div class="chart-container"><canvas id="b_km"></canvas></div>
    <div class="download-bar"><button class="download-btn" data-target="b_km">Download PNG</button></div>
  </div>

  <h2 class="section-divider">Data Tables</h2>
  <div class="container">
    <div class="tabs">
      <div class="tab active" onclick="showTable('tbl_tot')">Total Values</div>
      <div class="tab" onclick="showTable('tbl_km')">Per-km Values</div>
    </div>

    <div id="tbl_tot" class="data-table-container">
      <table id="tot_table">
        <thead>
          <tr>
            <th>JAO ID</th><th>PyPSA ID</th><th>Length (km)</th><th>Circuits</th>
            <th>Original R (Ω)</th><th>Allocated R (Ω)</th><th>R Diff (%)</th>
            <th>Original X (Ω)</th><th>Allocated X (Ω)</th><th>X Diff (%)</th>
            <th>Original B (S)</th><th>Allocated B (S)</th><th>B Diff (%)</th>
          </tr>
          <tr class="filters">
            <td><input data-col="0" placeholder="Filter JAO"></td>
            <td><input data-col="1" placeholder="Filter PyPSA"></td>
            <td><input data-col="2" placeholder="Length"></td>
            <td><input data-col="3" placeholder="Circuits"></td>
            <td><input data-col="4" placeholder="Orig R"></td>
            <td><input data-col="5" placeholder="Alloc R"></td>
            <td><input data-col="6" placeholder="R %"></td>
            <td><input data-col="7" placeholder="Orig X"></td>
            <td><input data-col="8" placeholder="Alloc X"></td>
            <td><input data-col="9" placeholder="X %"></td>
            <td><input data-col="10" placeholder="Orig B"></td>
            <td><input data-col="11" placeholder="Alloc B"></td>
            <td><input data-col="12" placeholder="B %"></td>
          </tr>
        </thead>
        <tbody></tbody>
      </table>
    </div>

    <div id="tbl_km" class="data-table-container" style="display:none">
      <table id="km_table">
        <thead>
          <tr>
            <th>JAO ID</th><th>PyPSA ID</th><th>Length (km)</th><th>Circuits</th>
            <th>PyPSA R (Ω/km)</th><th>JAO R (Ω/km)</th><th>R Diff (%)</th>
            <th>PyPSA X (Ω/km)</th><th>JAO X (Ω/km)</th><th>X Diff (%)</th>
            <th>PyPSA B (S/km)</th><th>JAO B (S/km)</th><th>B Diff (%)</th>
          </tr>
          <tr class="filters">
            <td><input data-col="0" placeholder="Filter JAO"></td>
            <td><input data-col="1" placeholder="Filter PyPSA"></td>
            <td><input data-col="2" placeholder="Length"></td>
            <td><input data-col="3" placeholder="Circuits"></td>
            <td><input data-col="4" placeholder="PyPSA R"></td>
            <td><input data-col="5" placeholder="JAO R"></td>
            <td><input data-col="6" placeholder="R %"></td>
            <td><input data-col="7" placeholder="PyPSA X"></td>
            <td><input data-col="8" placeholder="JAO X"></td>
            <td><input data-col="9" placeholder="X %"></td>
            <td><input data-col="10" placeholder="PyPSA B"></td>
            <td><input data-col="11" placeholder="JAO B"></td>
            <td><input data-col="12" placeholder="B %"></td>
          </tr>
        </thead>
        <tbody></tbody>
      </table>
    </div>
  </div>

<script>
  const DATA = __DATA_JSON__;
  const RVALUES = __R_VALUES__;

  function sci(n){
    if (!isFinite(n) || n === 0) return '0';
    const sign = n < 0 ? '-' : '';
    n = Math.abs(n);
    const e = Math.floor(Math.log10(n));
    const a = n / Math.pow(10, e);
    return `${sign}${a.toFixed(1)}×10^${e}`;
  }
  function logTick(val){
    const e = Math.log10(val);
    return Math.abs(e - Math.round(e)) < 1e-9 ? `10^${Math.round(e)}` : '';
  }

  function colorByDiff(p){
    if (p === null || isNaN(p)) return 'rgba(150,150,150,0.85)';
    const clamped = Math.max(-100, Math.min(100, p));
    if (clamped < 0){
      const t = Math.min(1, Math.abs(clamped)/50);
      const r = Math.round(25 + (76 - 25)*t);
      const g = Math.round(118 + (175 - 118)*t);
      const b = Math.round(210 + (80 - 210)*t);
      return `rgba(${r},${g},${b},0.85)`; // blue→green
    }else{
      const t = Math.min(1, clamped/50);
      const r = Math.round(76 + (211 - 76)*t);
      const g = Math.round(175 + (47 - 175)*t);
      const b = Math.round(80 + (47 - 80)*t);
      return `rgba(${r},${g},${b},0.85)`; // green→red
    }
  }

  function axisLabel(param, perKm){
    if (param==='r') return perKm ? 'R (Ω/km)' : 'R (Ω)';
    if (param==='x') return perKm ? 'X (Ω/km)' : 'X (Ω)';
    return perKm ? 'B (S/km)' : 'B (S)';
  }
  function titleText(prefix, param){
    const lab = axisLabel(param,false);
    const r = (RVALUES && RVALUES[param]!=null) ? RVALUES[param] : 'N/A';
    return `${prefix} ${lab} (r=${r})`;
  }

  Chart.defaults.font.size = 14;

  function domainFor(items){
    const xs = items.map(d=>d.x), ys = items.map(d=>d.y);
    const lo = Math.min(Math.min(...xs), Math.min(...ys));
    const hi = Math.max(Math.max(...xs), Math.max(...ys));
    const pad = 0.1*(hi - lo || 1);
    return {min: lo - pad, max: hi + pad};
  }

  function buildTotals(param, canvasId){
    const items = DATA.map(it => ({
      x: +it['original_'+param], y: +it['allocated_'+param],
      diff: it['diff_'+param+'_pct']
    })).filter(d => isFinite(d.x) && isFinite(d.y) && d.x>0 && d.y>0);

    const dom = domainFor(items);
    const ctx = document.getElementById(canvasId).getContext('2d');

    new Chart(ctx, {
      type:'scatter',
      data:{
        datasets:[
          {data:items.map(d=>({x:d.x, y:d.y})),
           backgroundColor: items.map(d=>colorByDiff(d.diff)), pointRadius:6, pointHoverRadius:8},
          {data:[{x:dom.min,y:dom.min},{x:dom.max,y:dom.max}],
           showLine:true, pointRadius:0, borderColor:'rgba(80,80,80,.8)', borderDash:[6,6], borderWidth:2}
        ]
      },
      options:{
        responsive:true, maintainAspectRatio:false,
        plugins:{
          title:{display:true, text:titleText('Original vs Allocated', param), font:{size:18, weight:'600'}},
          legend:{display:false},  // keep charts clean
          tooltip:{
            callbacks:{
              label:(ctx)=>{
                const p = items[ctx.dataIndex];
                const diff = (p.diff==null||isNaN(p.diff)) ? 'N/A' : p.diff.toFixed(2)+'%';
                return [`Original: ${p.x.toExponential(3)}`, `Allocated: ${p.y.toExponential(3)}`, `Δ%: ${diff}`];
              }
            }
          }
        },
        scales:{
          x:{title:{display:true, text:`Original ${axisLabel(param,false)}`},
             ticks:{callback:(v)=>sci(v)}},
          y:{title:{display:true, text:`Allocated ${axisLabel(param,false)}`},
             ticks:{callback:(v)=>sci(v)}}
        }
      }
    });
  }

  function buildPerKm(param, canvasId){
    const items = DATA.map(it => ({
      x: +it['original_'+param+'_per_km'],
      y: +it['jao_'+param+'_km_adjusted'],
      diff: it['diff_'+param+'_pct_km']
    })).filter(d => isFinite(d.x) && isFinite(d.y) && d.x>0 && d.y>0);

    const xs = items.map(d=>d.x), ys = items.map(d=>d.y);
    const lo = Math.min(Math.min(...xs), Math.min(...ys));
    const hi = Math.max(Math.max(...xs), Math.max(...ys));
    const ctx = document.getElementById(canvasId).getContext('2d');

    new Chart(ctx, {
      type:'scatter',
      data:{
        datasets:[
          {data:items.map(d=>({x:d.x, y:d.y})),
           backgroundColor: items.map(d=>colorByDiff(d.diff)), pointRadius:6, pointHoverRadius:8},
          {data:[{x:lo,y:lo},{x:hi,y:hi}],
           showLine:true, pointRadius:0, borderColor:'rgba(80,80,80,.8)', borderDash:[6,6], borderWidth:2}
        ]
      },
      options:{
        responsive:true, maintainAspectRatio:false,
        plugins:{
          title:{display:true, text:titleText('PyPSA vs JAO — Per-km (circuit-adjusted)', param), font:{size:18, weight:'600'}},
          legend:{display:false},
          tooltip:{
            callbacks:{
              label:(ctx)=>{
                const p = items[ctx.dataIndex];
                const diff = (p.diff==null||isNaN(p.diff)) ? 'N/A' : p.diff.toFixed(2)+'%';
                return [`PyPSA: ${p.x.toExponential(3)}`, `JAO: ${p.y.toExponential(3)}`, `Δ%: ${diff}`];
              }
            }
          }
        },
        scales:{
          x:{type:'logarithmic', title:{display:true, text:`PyPSA ${axisLabel(param,true)}`},
             ticks:{callback:(v)=>logTick(v)}},
          y:{type:'logarithmic', title:{display:true, text:`JAO ${axisLabel(param,true)} (adjusted)`},
             ticks:{callback:(v)=>logTick(v)}}
        }
      }
    });
  }

  // ------- PNG export with legend overlay -------
  function drawRoundedRect(ctx, x, y, w, h, r){
    ctx.beginPath();
    ctx.moveTo(x+r, y);
    ctx.arcTo(x+w, y, x+w, y+h, r);
    ctx.arcTo(x+w, y+h, x, y+h, r);
    ctx.arcTo(x, y+h, x, y, r);
    ctx.arcTo(x, y, x+w, y, r);
    ctx.closePath();
  }

  function drawLegend(ctx, canvasWidth, canvasHeight){
    // Colors synced with UI legend
    const items = [
      {label: 'good match (|Δ| ≤ 20%)', color: '#4CAF50'},
      {label: 'moderate (20–50%, JAO > PyPSA)', color: '#ffb74d'},
      {label: 'large positive Δ (JAO ≫ PyPSA)', color: '#d32f2f'},
      {label: 'negative Δ (JAO < PyPSA)', color: '#1976D2'},
    ];

    const pad = 12;
    const sw = 14;          // swatch width
    const sh = 10;          // swatch height
    const gap = 8;
    const lineH = 20;
    const header = 'Δ% legend (color encoding)';
    ctx.font = '14px Arial';

    // Compute box size
    let textW = ctx.measureText(header).width;
    items.forEach(it => { textW = Math.max(textW, ctx.measureText(it.label).width); });
    const boxW = Math.ceil(pad*3 + sw + gap + textW);
    const boxH = Math.ceil(pad*2 + 16 + items.length*lineH);

    const x = canvasWidth - boxW - 16;
    const y = canvasHeight - boxH - 16;

    // Background
    ctx.save();
    ctx.globalAlpha = 0.95;
    ctx.fillStyle = '#ffffff';
    ctx.strokeStyle = 'rgba(0,0,0,0.15)';
    drawRoundedRect(ctx, x, y, boxW, boxH, 8);
    ctx.fill();
    ctx.stroke();

    // Header
    ctx.fillStyle = '#333';
    ctx.font = 'bold 14px Arial';
    ctx.fillText(header, x + pad, y + pad + 12);

    // Items
    ctx.font = '14px Arial';
    let yy = y + pad + 20;
    items.forEach(it => {
      ctx.fillStyle = it.color;
      ctx.fillRect(x + pad, yy - sh + 8, sw, sh);
      ctx.strokeStyle = 'rgba(0,0,0,0.2)';
      ctx.strokeRect(x + pad, yy - sh + 8, sw, sh);

      ctx.fillStyle = '#333';
      ctx.fillText(it.label, x + pad + sw + gap, yy + 6);
      yy += lineH;
    });
    ctx.restore();
  }

  function canvasToPngWithLegend(canvas, filename){
    const w = canvas.width, h = canvas.height;
    const tmp = document.createElement('canvas');
    tmp.width = w; tmp.height = h;
    const ctx = tmp.getContext('2d');

    // white background + chart
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0,0,w,h);
    ctx.drawImage(canvas, 0, 0);

    // draw legend overlay
    drawLegend(ctx, w, h);

    // download
    if (tmp.toBlob) {
      tmp.toBlob((blob)=>{
        const a = document.createElement('a');
        a.href = URL.createObjectURL(blob);
        a.download = filename || 'chart.png';
        document.body.appendChild(a);
        a.click();
        URL.revokeObjectURL(a.href);
        a.remove();
      }, 'image/png', 1);
    } else {
      const uri = tmp.toDataURL('image/png');
      const a = document.createElement('a');
      a.href = uri; a.download = filename || 'chart.png';
      document.body.appendChild(a); a.click(); a.remove();
    }
  }

  // ------- tables & filters -------
  function fillTotalsTable(){
    const tb = document.querySelector('#tot_table tbody');
    tb.innerHTML = '';
    DATA.forEach(it=>{
      const tr = document.createElement('tr');
      const cells = [
        it.jao_id, it.pypsa_id,
        (isFinite(+it.length_km)? (+it.length_km).toFixed(2):''),
        it.circuits,
        (+it.original_r).toExponential(6), (+it.allocated_r).toExponential(6),
        it.diff_r_pct==null?'N/A':(+it.diff_r_pct).toFixed(2)+'%',
        (+it.original_x).toExponential(6), (+it.allocated_x).toExponential(6),
        it.diff_x_pct==null?'N/A':(+it.diff_x_pct).toFixed(2)+'%',
        (+it.original_b).toExponential(6), (+it.allocated_b).toExponential(6),
        it.diff_b_pct==null?'N/A':(+it.diff_b_pct).toFixed(2)+'%'
      ];
      cells.forEach((txt, idx)=>{
        const td = document.createElement('td'); td.textContent = txt;
        if ([6,9,12].includes(idx)){
          const v = parseFloat(String(txt).replace('%',''));
          if (!isNaN(v)){
            td.className = Math.abs(v)<=20 ? 'good-match' : (Math.abs(v)<=50 ? 'moderate-match' : 'poor-match');
          }
        }
        tr.appendChild(td);
      });
      tb.appendChild(tr);
    });
  }

  function fillPerKmTable(){
    const tb = document.querySelector('#km_table tbody');
    tb.innerHTML = '';
    DATA.forEach(it=>{
      const tr = document.createElement('tr');
      const cells = [
        it.jao_id, it.pypsa_id,
        (isFinite(+it.length_km)? (+it.length_km).toFixed(2):''),
        it.circuits,
        (+it.original_r_per_km).toExponential(6), (+it.jao_r_km_adjusted).toExponential(6),
        it.diff_r_pct_km==null?'N/A':(+it.diff_r_pct_km).toFixed(2)+'%',
        (+it.original_x_per_km).toExponential(6), (+it.jao_x_km_adjusted).toExponential(6),
        it.diff_x_pct_km==null?'N/A':(+it.diff_x_pct_km).toFixed(2)+'%',
        (+it.original_b_per_km).toExponential(6), (+it.jao_b_km_adjusted).toExponential(6),
        it.diff_b_pct_km==null?'N/A':(+it.diff_b_pct_km).toFixed(2)+'%'
      ];
      cells.forEach((txt, idx)=>{
        const td = document.createElement('td'); td.textContent = txt;
        if ([6,9,12].includes(idx)){
          const v = parseFloat(String(txt).replace('%',''));
          if (!isNaN(v)){
            td.className = Math.abs(v)<=20 ? 'good-match' : (Math.abs(v)<=50 ? 'moderate-match' : 'poor-match');
          }
        }
        tr.appendChild(td);
      });
      tb.appendChild(tr);
    });
  }

  function setupFilters(){
    document.querySelectorAll('.filters input').forEach(inp=>{
      inp.addEventListener('input', ()=>{
        const table = inp.closest('table');
        const col = +inp.dataset.col;
        const val = inp.value.toLowerCase();
        table.querySelectorAll('tbody tr').forEach(tr=>{
          const cell = tr.children[col];
          const txt = (cell?.textContent || '').toLowerCase();
          tr.style.display = (val==='' || txt.includes(val)) ? '' : 'none';
        });
      });
    });
  }

  function showTable(id){
    document.getElementById('tbl_tot').style.display = (id==='tbl_tot') ? 'block':'none';
    document.getElementById('tbl_km').style.display  = (id==='tbl_km') ? 'block':'none';
    const tabs = document.querySelectorAll('.tab');
    tabs[0].classList.toggle('active', id==='tbl_tot');
    tabs[1].classList.toggle('active', id==='tbl_km');
  }

  // ------- download buttons -------
  function bindDownloadButtons(){
    document.querySelectorAll('.download-btn').forEach(btn=>{
      btn.addEventListener('click', ()=>{
        const id = btn.getAttribute('data-target');
        const cv = document.getElementById(id);
        if (cv) canvasToPngWithLegend(cv, id + '.png');
      });
    });
  }

  // init
  document.addEventListener('DOMContentLoaded', ()=>{
    buildTotals('r', 'r_tot'); buildTotals('x', 'x_tot'); buildTotals('b', 'b_tot');
    buildPerKm('r', 'r_km'); buildPerKm('x', 'x_km'); buildPerKm('b', 'b_km');
    fillTotalsTable(); fillPerKmTable(); setupFilters(); bindDownloadButtons();
  });
</script>
</body>
</html>
"""

    html = html.replace("__DATA_JSON__", json.dumps(rows))
    html = html.replace("__R_VALUES__", json.dumps(r_vals))

    out_file = out_dir / "parameter_comparison.html"
    out_file.write_text(html, encoding="utf-8")
    print(f"Parameter comparison visualization saved to {out_file}")
    return str(out_file)





import pandas as pd
from pathlib import Path

def create_updated_pypsa_with_jao_params(
    pypsa_csv: str,
    jao_csv: str,
    matches,                 # either a list[dict] from your matcher or a path to CSV
    out_csv: str,
    overwrite_s_nom: bool = False,   # set True if you want to REPLACE PyPSA s_nom by JAO values
    add_eic: bool = True             # add EIC_Code to PyPSA when available
):
    """
    Update a PyPSA transformer CSV by adding JAO electrical parameters for matched rows.

    Creates columns:
      - jao_s_nom, jao_r, jao_x, jao_b, jao_g
      - (optionally) EIC_Code copied from JAO
      - (optionally) overwrite s_nom with jao_s_nom

    Args
    ----
    pypsa_csv : path to original pypsa_transformers.csv
    jao_csv   : path to jao_transformers.csv
    matches   : list of match dicts (from your pipeline) OR path to transformer_matches.csv
    out_csv   : path to write the updated PyPSA CSV
    overwrite_s_nom : if True, overwrite the existing PyPSA `s_nom` with JAO value
    add_eic   : if True, write JAO EIC code into PyPSA column `EIC_Code`
    """
    pypsa_df = pd.read_csv(pypsa_csv)
    jao_df   = pd.read_csv(jao_csv)

    # Normalize keys (some JAO files vary slightly in column names)
    # Required: r, x, b, g; Optional/guessed: s_nom
    def find_col(df, candidates):
        for c in candidates:
            if c in df.columns:
                return c
        # case-insensitive fallback
        lower_map = {c.lower(): c for c in df.columns}
        for c in candidates:
            if c.lower() in lower_map:
                return lower_map[c.lower()]
        return None

    col_r = find_col(jao_df, ["r"])
    col_x = find_col(jao_df, ["x"])
    col_b = find_col(jao_df, ["b"])
    col_g = find_col(jao_df, ["g"])
    col_eic = find_col(jao_df, ["EIC_Code", "eic_code", "EIC", "eic"])

    # Try to locate an apparent S (MVA) column
    col_s = find_col(
        jao_df,
        ["s_nom", "S_nom", "Snom", "S_nom (MVA)", "Rated Power (MVA)", "Power (MVA)", "S (MVA)"]
    )

    # Coerce numeric where present
    for c in [col_r, col_x, col_b, col_g, col_s]:
        if c and c in jao_df.columns:
            jao_df[c] = pd.to_numeric(jao_df[c], errors="coerce")

    # Build a fast lookup for JAO rows by EIC and by name
    jao_by_eic  = {}
    if col_eic:
        jao_by_eic = {str(v): row for v, row in jao_df.set_index(col_eic).iterrows() if pd.notna(v)}
    jao_by_name = {str(row.get("name", "")): row for _, row in jao_df.iterrows() if str(row.get("name","")) != ""}

    # Load matches (list or CSV)
    if isinstance(matches, (str, Path)):
        matches_df = pd.read_csv(matches)
        # normalize column names we need
        need_cols = {
            "pypsa_id": ["pypsa_id", "transformer_id", "pypsa_transformer_id"],
            "jao_id":   ["jao_id", "EIC_Code", "jao_eic", "jao_eic_code"],
            "jao_name": ["jao_name", "name"]
        }
        def pick(src, options):
            for o in options:
                if o in src.columns:
                    return o
            return None

        c_pypsa = pick(matches_df, need_cols["pypsa_id"])
        c_jao   = pick(matches_df, need_cols["jao_id"])
        c_jname = pick(matches_df, need_cols["jao_name"])

        match_list = []
        for _, r in matches_df.iterrows():
            match_list.append({
                "pypsa_id": str(r.get(c_pypsa, "")),
                "jao_id":   (str(r.get(c_jao, "")) if c_jao else ""),
                "jao_name": (str(r.get(c_jname, "")) if c_jname else "")
            })
    else:
        # assume a list[dict] from your pipeline
        match_list = [
            {
                "pypsa_id": str(m.get("pypsa_id", "")),
                "jao_id":   str(m.get("jao_id", "")),
                "jao_name": str(m.get("jao_name", "")),
                # if your matcher already carries jao_r etc., we could also use those directly
                "jao_r":    m.get("jao_r", None),
                "jao_x":    m.get("jao_x", None),
                "jao_b":    m.get("jao_b", None),
                "jao_g":    m.get("jao_g", None),
            }
            for m in matches if bool(m.get("matched", False))
        ]

    # Ensure output columns exist on PyPSA side
    for c in ["jao_s_nom", "jao_r", "jao_x", "jao_b", "jao_g"]:
        if c not in pypsa_df.columns:
            pypsa_df[c] = pd.NA
    if add_eic and "EIC_Code" not in pypsa_df.columns:
        pypsa_df["EIC_Code"] = pd.NA

    # Index for fast row selection in PyPSA
    if "transformer_id" not in pypsa_df.columns:
        raise ValueError("PyPSA CSV must have a 'transformer_id' column.")

    pypsa_df.set_index("transformer_id", inplace=True, drop=False)

    # Apply updates for matched rows
    updated = 0
    for m in match_list:
        pid = m.get("pypsa_id", "")
        if not pid or pid not in pypsa_df.index:
            continue

        # Prefer values directly from the match dict if present; otherwise pull from JAO row
        jao_row = None
        if col_eic and str(m.get("jao_id","")) in jao_by_eic:
            jao_row = jao_by_eic[str(m.get("jao_id",""))]
        elif m.get("jao_name","") in jao_by_name:
            jao_row = jao_by_name[m.get("jao_name","")]

        # Resolve parameters
        def get_val(key_from_match, jao_colname):
            # 1) from match list if present
            if key_from_match in m and m[key_from_match] is not None:
                return m[key_from_match]
            # 2) from JAO row if available
            if jao_row is not None and jao_colname and jao_colname in jao_row:
                return jao_row[jao_colname]
            return None

        v_r = get_val("jao_r", col_r)
        v_x = get_val("jao_x", col_x)
        v_b = get_val("jao_b", col_b)
        v_g = get_val("jao_g", col_g)
        v_s = (jao_row[col_s] if (jao_row is not None and col_s and col_s in jao_row) else None)

        if pd.notna(v_r): pypsa_df.at[pid, "jao_r"] = pd.to_numeric(v_r, errors="coerce")
        if pd.notna(v_x): pypsa_df.at[pid, "jao_x"] = pd.to_numeric(v_x, errors="coerce")
        if pd.notna(v_b): pypsa_df.at[pid, "jao_b"] = pd.to_numeric(v_b, errors="coerce")
        if pd.notna(v_g): pypsa_df.at[pid, "jao_g"] = pd.to_numeric(v_g, errors="coerce")
        if pd.notna(v_s): pypsa_df.at[pid, "jao_s_nom"] = pd.to_numeric(v_s, errors="coerce")

        if add_eic:
            eid = m.get("jao_id", None)
            if eid:
                pypsa_df.at[pid, "EIC_Code"] = eid

        if overwrite_s_nom and pd.notna(pypsa_df.at[pid, "jao_s_nom"]):
            pypsa_df.at[pid, "s_nom"] = pypsa_df.at[pid, "jao_s_nom"]

        updated += 1

    # Final tidy: ensure numeric dtype where possible
    for c in ["jao_s_nom", "jao_r", "jao_x", "jao_b", "jao_g", "s_nom"]:
        if c in pypsa_df.columns:
            pypsa_df[c] = pd.to_numeric(pypsa_df[c], errors="coerce")

    # Write output
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    pypsa_df.to_csv(out_csv, index=False)
    print(f"Updated {updated} PyPSA transformers with JAO parameters -> {out_csv}")
