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
