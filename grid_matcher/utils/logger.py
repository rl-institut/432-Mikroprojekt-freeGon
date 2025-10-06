
import sys
import logging
import os


def setup_logger(log_level=logging.INFO, log_file=None, name='grid_matcher'):
    """
    Set up and configure a logger for the application.

    Parameters
    ----------
    log_level : int
        Logging level (e.g., logging.DEBUG, logging.INFO)
    log_file : str, optional
        Path to log file. If provided, logs will be written to this file
        in addition to the console.
    name : str
        Logger name, defaults to 'grid_matcher'

    Returns
    -------
    logging.Logger
        Configured logger
    """
    from ..config import LOG_FORMAT

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Remove existing handlers to avoid duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(LOG_FORMAT)

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Create file handler if log_file is provided
    if log_file:
        # Ensure directory exists
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name):
    """Get an existing logger or create a new one"""
    return logging.getLogger(name)

def install_stdout_tee(log_path="output/chain_debug.log"):
    log_path = str(log_path)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    class _Tee:
        def __init__(self, path, base):
            self.file = open(path, "a", buffering=1, encoding="utf-8")
            self.base = base  # sys.__stdout__ or sys.__stderr__

        def write(self, s):
            try:
                self.base.write(s)
            except Exception:
                pass
            try:
                self.file.write(s)
            except Exception:
                pass

        def flush(self):
            try:
                self.base.flush()
            except Exception:
                pass
            try:
                self.file.flush()
            except Exception:
                pass

    # Tee BOTH stdout and stderr
    sys.stdout = _Tee(log_path, sys.__stdout__)
    sys.stderr = _Tee(log_path, sys.__stderr__)
    _force_unbuffered()
def _force_unbuffered():
    os.environ["PYTHONUNBUFFERED"] = "1"
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(line_buffering=True)  # Py3.7+
        except Exception:
            pass



def print_debug_banner(msg: str):
    import sys
    sys.stdout.write("\n" + "="*12 + f" {msg} " + "="*12 + "\n")
    sys.stdout.flush()

def dump_matched_jao_ids(matching_results, limit=20):
    """Print a sample of matched JAO ids so you can verify the ID you’re debugging exists."""
    import sys
    print_debug_banner("Matched JAO IDs (sample)")
    cnt = 0
    for r in (matching_results or []):
        if r.get("matched"):
            print(f"  jao_id={r.get('jao_id')}  pypsa_ids={r.get('pypsa_ids')}", flush=True)
            cnt += 1
            if cnt >= limit:
                break
    if cnt == 0:
        print("  (none matched yet)", flush=True)
    sys.stdout.flush()



def debug_chain_growth(
    jao_id,
    matching_results,
    jao_gdf,
    pypsa_gdf,
    buffer_m=120.0,
    endpoint_grid_m=50.0,
    max_additions=40,
):
    """
    Verbose debugger for chain growth around a single JAO record.
    Prints why neighbor PyPSA segments were accepted/rejected during BFS.

    Call example:
        debug_chain_growth("2607", matches, jao_gdf, pypsa_gdf, buffer_m=120, endpoint_grid_m=50)
    """
    from collections import defaultdict, deque
    from shapely.geometry import Point, LineString

    # ----------------- helpers -----------------
    def _ensure_crs(gdf, epsg=4326):
        try:
            if gdf.crs is None:
                gdf = gdf.set_crs(epsg, allow_override=True)
            return gdf
        except Exception:
            return gdf

    def _to_meters(gdf):
        try:
            return gdf.to_crs(3857)
        except Exception:
            return gdf

    def _as_point(obj):
        if obj is None:
            return None
        if hasattr(obj, "x") and hasattr(obj, "y"):
            return obj
        try:
            x, y = float(obj[0]), float(obj[1])
            return Point(x, y)
        except Exception:
            return None

    def _endpoint_key(pt, grid=endpoint_grid_m):
        p = _as_point(pt)
        if p is None:
            return None
        x = round(p.x / grid) * grid
        y = round(p.y / grid) * grid
        return (float(x), float(y))

    def _endpoints(ls):
        try:
            coords = list(ls.coords)
            if len(coords) < 2:
                return None, None
            return Point(coords[0]), Point(coords[-1])
        except Exception:
            return None, None

    def _parse_ids(v):
        if not v:
            return []
        if isinstance(v, (list, tuple, set)):
            out = [str(x).strip() for x in v if str(x).strip()]
        else:
            s = str(v)
            out = []
            for sep in (";", ","):
                if sep in s:
                    out = [t.strip() for t in s.split(sep)]
                    break
            if not out:
                out = [s.strip()]
        return [x for x in out if x]

    def _len_km(row):
        try:
            return float(row.get("length", 0.0) or 0.0) / 1000.0
        except Exception:
            return 0.0

    def _hausdorff_m(a, b):
        try:
            return float(a.hausdorff_distance(b))
        except Exception:
            return float("inf")

    def _buffered_iou(a, b, w=buffer_m):
        try:
            ab = a.buffer(w)
            bb = b.buffer(w)
            inter = ab.intersection(bb).area
            den = ab.union(bb).area
            if den <= 0:
                return 0.0
            return float(inter / den)
        except Exception:
            return 0.0

    def _segment_fits_jao(seg, jao_line):
        dH = _hausdorff_m(seg, jao_line)
        iou = _buffered_iou(seg, jao_line, w=buffer_m)
        ok = (dH <= 1.25 * buffer_m) or (iou >= 0.20)
        return ok, dH, iou

    # ------------- prepare data -------------
    jao_gdf = _ensure_crs(jao_gdf)
    pypsa_gdf = _ensure_crs(pypsa_gdf)
    jao_m = _to_meters(jao_gdf)
    py_m = _to_meters(pypsa_gdf)

    # lookups
    jao_by_id = {str(r.get("id")): r for _, r in jao_gdf.iterrows()}
    jao_m_by_id = {str(r.get("id")): r for _, r in jao_m.iterrows()}
    py_by_id = {str(r.get("line_id", r.get("id", ""))): r for _, r in pypsa_gdf.iterrows()}
    py_m_by_id = {str(r.get("line_id", r.get("id", ""))): r for _, r in py_m.iterrows()}

    # find the result for this JAO id
    res = next((r for r in (matching_results or []) if str(r.get("jao_id")) == str(jao_id)), None)
    if not res:
        print(f"[DEBUG] No matching result with jao_id={jao_id}")
        return

    seeds = _parse_ids(res.get("pypsa_ids", []))
    if not seeds:
        print(f"[DEBUG] JAO {jao_id}: no seed pypsa_ids in matching_results")
        return

    if jao_id not in jao_by_id:
        print(f"[DEBUG] JAO {jao_id}: not found in jao_gdf")
        return

    jlen_km = float(jao_by_id[jao_id].get("length", 0.0) or 0.0)
    jr_m = jao_m_by_id.get(jao_id)
    if jr_m is None or jr_m.geometry is None or jr_m.geometry.is_empty:
        print(f"[DEBUG] JAO {jao_id}: missing geometry")
        return
    jao_line = jr_m.geometry
    if not isinstance(jao_line, LineString) and hasattr(jao_line, "geoms"):
        try:
            jao_line = max(list(jao_line.geoms), key=lambda g: g.length)
        except Exception:
            pass

    # ------------- build endpoint adjacency for PyPSA -------------
    from collections import defaultdict
    endpoint_index = defaultdict(set)
    id_to_endkeys = {}

    for pid, rowm in py_m_by_id.items():
        geom = rowm.geometry
        if geom is None or geom.is_empty:
            continue
        ls = geom
        if not isinstance(ls, LineString) and hasattr(ls, "geoms"):
            try:
                ls = max(list(ls.geoms), key=lambda g: g.length)
            except Exception:
                continue
        p0, p1 = _endpoints(ls)
        if p0 is None or p1 is None:
            continue
        k0, k1 = _endpoint_key(p0), _endpoint_key(p1)
        id_to_endkeys[pid] = (k0, k1)
        if k0: endpoint_index[k0].add(pid)
        if k1: endpoint_index[k1].add(pid)

    neighbors = defaultdict(set)
    for key, ids in endpoint_index.items():
        ids = list(ids)
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = ids[i], ids[j]
                neighbors[a].add(b)
                neighbors[b].add(a)

    # ------------- print header -------------
    print(f"\n[DEBUG] ===== Chain growth debugger for JAO {jao_id} =====")
    print(f"[DEBUG] JAO length (km): {jlen_km:.3f}")
    print(f"[DEBUG] Initial seed PyPSA IDs: {seeds}")

    # current coverage
    curr_km = sum(_len_km(py_by_id.get(pid, {})) for pid in seeds)
    if jlen_km > 0:
        print(f"[DEBUG] Initial coverage: {curr_km:.2f} / {jlen_km:.2f} km ({100.0*curr_km/jlen_km:.1f}%)")
    else:
        print(f"[DEBUG] Initial coverage: {curr_km:.2f} km (JAO length unknown)")

    # ------------- BFS with verbose logging -------------
    selected = set(seeds)
    frontier = deque(seeds)
    additions = 0
    step = 0

    while frontier and additions < max_additions:
        step += 1
        seed = frontier.popleft()
        print(f"\n[DEBUG] Step {step}: expanding from seed {seed}")

        # list neighbors
        nbs = sorted(list(neighbors.get(seed, [])))
        if not nbs:
            print(f"[DEBUG]   No neighbors via endpoint grid for {seed}")
            continue
        print(f"[DEBUG]   Neighbors of {seed}: {nbs}")

        s_endkeys = id_to_endkeys.get(seed)
        print(f"[DEBUG]   Endpoint keys(seed): {s_endkeys}")

        for nb in nbs:
            if nb in selected:
                print(f"[DEBUG]     - {nb}: already selected, skip")
                continue

            nb_row_m = py_m_by_id.get(nb)
            if nb_row_m is None or nb_row_m.geometry is None or nb_row_m.geometry.is_empty:
                print(f"[DEBUG]     - {nb}: missing geometry, skip")
                continue

            seg = nb_row_m.geometry
            if not isinstance(seg, LineString) and hasattr(seg, "geoms"):
                try:
                    seg = max(list(seg.geoms), key=lambda g: g.length)
                except Exception:
                    print(f"[DEBUG]     - {nb}: cannot resolve to LineString, skip")
                    continue

            ok, dH, iou = _segment_fits_jao(seg, jao_line)
            print(f"[DEBUG]     - {nb}: fits={ok}  hausdorff={dH:.1f} m  IoU={iou:.3f}")

            nb_keys = id_to_endkeys.get(nb)
            print(f"[DEBUG]         endpoint keys(nb): {nb_keys}")

            if ok:
                selected.add(nb)
                frontier.append(nb)
                additions += 1
                add_km = _len_km(py_by_id.get(nb, {}))
                curr_km += add_km
                cov = (curr_km / jlen_km) if jlen_km > 0 else 0.0
                print(f"[DEBUG]         ACCEPTED. +{add_km:.2f} km; coverage now {curr_km:.2f}/{jlen_km:.2f} km ({100.0*cov:.1f}%)")
            else:
                reason = []
                if dH > 1.25 * buffer_m:
                    reason.append(f"Hausdorff>{1.25*buffer_m:.0f}m")
                if iou < 0.20:
                    reason.append("IoU<0.20")
                print(f"[DEBUG]         REJECTED due to {', '.join(reason) if reason else 'unknown'}")

            if additions >= max_additions:
                print(f"[DEBUG]   Reached max_additions={max_additions}, stopping.")
                break

    print(f"\n[DEBUG] ===== Final selection for JAO {jao_id} =====")
    final_ids = sorted(selected)
    print(f"[DEBUG] Selected PyPSA IDs ({len(final_ids)}): {final_ids}")
    if jlen_km > 0:
        print(f"[DEBUG] Final coverage: {curr_km:.2f} / {jlen_km:.2f} km ({100.0*curr_km/jlen_km:.1f}%)")
    else:
        print(f"[DEBUG] Final coverage: {curr_km:.2f} km")
    print("[DEBUG] =============================================\n")


def run_chain_debug(jao_ids, matches, jao_gdf, pypsa_gdf, buffer_m=120.0, endpoint_grid_m=50.0):
    """
    Wrapper that ensures you see logs from debug_chain_growth and also shows
    a few actually matched JAO ids in case the one you typed isn't present yet.
    """
    import sys
    print_debug_banner("BEGIN CHAIN DEBUG")
    dump_matched_jao_ids(matches, limit=10)

    # Make sure the function exists
    if 'debug_chain_growth' not in globals():
        print("[DEBUG] debug_chain_growth is not defined in this module.", flush=True)
        print_debug_banner("END CHAIN DEBUG")
        sys.stdout.flush()
        return

    for jid in jao_ids:
        print(f"\n>>> Running chain-growth debug for JAO {jid}", flush=True)
        try:
            debug_chain_growth(str(jid), matches, jao_gdf, pypsa_gdf,
                               buffer_m=buffer_m, endpoint_grid_m=endpoint_grid_m)
        except Exception as e:
            print(f"[DEBUG] Error inside debug_chain_growth({jid}): {e}", flush=True)

    print_debug_banner("END CHAIN DEBUG")
    sys.stdout.flush()