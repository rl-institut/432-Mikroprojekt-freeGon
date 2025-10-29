#!/usr/bin/env python
# fix_pypsa_geometries.py
# Robust parser for PyPSA line CSV/TSV where 'geometry' is malformed and 'tags' can span multiple tokens.

from __future__ import annotations
import argparse
import csv
import re
from dataclasses import dataclass
from typing import List, Tuple, Optional

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, Point
from shapely.wkt import dumps as wkt_dumps


WKT_RE = re.compile(r"LINESTRING\s*\((.*?)\)", re.IGNORECASE | re.DOTALL)


@dataclass
class Stats:
    total_lines: int = 0
    parsed_rows: int = 0
    no_geom: int = 0
    unparsable_geom: int = 0
    one_point_fixed: int = 0


def detect_delimiter(header: str) -> str:
    return "\t" if "\t" in header else ","


def is_balanced(s: str) -> bool:
    return s.count("(") > 0 and s.count("(") == s.count(")")


def extract_linestring_block(rec: str) -> Optional[Tuple[int, int, str]]:
    """
    Returns (start_idx, end_idx, inner_coords_text) for the LINESTRING block, or None.
    """
    m = WKT_RE.search(rec)
    if not m:
        return None
    return m.start(), m.end(), m.group(1)


def parse_coords(inner_text: str) -> List[Tuple[float, float]]:
    """
    Parses coordinate pairs from *anything* that looks like 'x y' separated by whitespace,
    ignoring commas/tabs/linebreaks between pairs.
    """
    pair_re = re.compile(r"(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)")
    coords = [(float(x), float(y)) for x, y in pair_re.findall(inner_text)]
    return coords


def to_linestring(coords: List[Tuple[float, float]], epsilon: float = 1e-6) -> Optional[LineString]:
    """
    Build a LineString. If only one point is found, create a tiny 2-point segment.
    """
    if len(coords) >= 2:
        return LineString(coords)
    if len(coords) == 1:
        x, y = coords[0]
        # Create a tiny offset to make a valid 2-point line
        return LineString([(x, y), (x + epsilon, y + epsilon)])
    return None


def main(input_path: str, output_path: str, geojson_out: Optional[str]) -> None:
    rows = []
    stats = Stats()

    with open(input_path, "r", encoding="utf-8", newline="") as f:
        header_line = f.readline().rstrip("\n")
        delimiter = detect_delimiter(header_line)
        headers = header_line.split(delimiter)

        if "geometry" not in headers or "tags" not in headers:
            raise ValueError("Input header must contain 'tags' and 'geometry' columns.")

        tags_idx = headers.index("tags")

        # We'll build non-geometry columns using the header minus 'geometry'
        headers_no_geom = [h for h in headers if h != "geometry"]

        # Iterate physical lines; buffer if geometry spans multiple lines
        while True:
            first = f.readline()
            if not first:
                break
            stats.total_lines += 1
            rec = first.rstrip("\n")

            # If geometry appears and parentheses aren't balanced, keep reading
            if "LINESTRING" in rec and not is_balanced(rec[rec.find("LINESTRING"):]):
                while True:
                    nxt = f.readline()
                    if not nxt:
                        break
                    rec += nxt.rstrip("\n")
                    if is_balanced(rec[rec.find("LINESTRING"):]):
                        break

            # Extract the LINESTRING block (remove geometry before CSV tokenization)
            ls_block = extract_linestring_block(rec)
            if not ls_block:
                stats.no_geom += 1
                continue
            start_idx, end_idx, inner_coords_text = ls_block

            # Remove geometry block from the record so the csv parser sees everything except geometry
            rec_wo_geom = (rec[:start_idx] + rec[end_idx:]).strip(delimiter)

            # Tokenize the non-geometry fields
            parts = next(csv.reader([rec_wo_geom], delimiter=delimiter))

            # Collapse all tokens from 'tags' onward into one 'tags' field
            # (because 'tags' may contain multiple 'way/...' tokens)
            left = parts[:tags_idx]
            tags_joined = " ".join(parts[tags_idx:]).strip()
            parts_fixed = left + [tags_joined]

            # Map to header names (header without 'geometry')
            row = {}
            for i, col in enumerate(headers_no_geom):
                row[col] = parts_fixed[i] if i < len(parts_fixed) else ""

            # Duplicate id like the user expects (line_id and id both present)
            if "line_id" in row:
                row["id"] = row.get("id", row["line_id"])

            # Build geometry from raw coords (robust to tabs/newlines/no commas)
            coords = parse_coords(inner_coords_text)
            geom = to_linestring(coords)
            if geom is None:
                stats.unparsable_geom += 1
                continue
            if len(coords) == 1:
                stats.one_point_fixed += 1

            row["geometry"] = geom
            rows.append(row)
            stats.parsed_rows += 1

    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")

    # Column order: everything except geometry (in header order), ensure 'id' after 'line_id', then geometry
    base_cols = [c for c in headers_no_geom if c in gdf.columns and c != "geometry"]
    if "id" in gdf.columns and "line_id" in base_cols:
        # place id right after line_id if not already
        if "id" in base_cols:
            base_cols.remove("id")
        base_cols.insert(base_cols.index("line_id") + 1, "id")

    gdf = gdf.loc[:, base_cols + ["geometry"]]

    # Write GeoJSON (true geometry)
    if geojson_out:
        gdf.to_file(geojson_out, driver="GeoJSON")
        print(f"Saved GeoJSON to {geojson_out}")

    # Write TSV with canonical WKT (quoted)
    df_out = pd.DataFrame(gdf.drop(columns="geometry"))
    df_out["geometry"] = gdf.geometry.apply(lambda g: wkt_dumps(g, rounding_precision=14, trim=True))

    df_out.to_csv(
        output_path,
        sep="\t",
        index=False,
        quoting=csv.QUOTE_NONNUMERIC,  # quote non-numeric columns (incl. geometry & tags)
        quotechar='"',
        escapechar="\\",
    )
    print(f"Saved {len(df_out)} rows -> {output_path}")

    # Basic validations
    tabs_in_geom = df_out["geometry"].astype(str).str.contains("\t", na=False).sum()
    end_paren_miss = (~df_out["geometry"].astype(str).str.endswith(")", na=False)).sum()
    no_comma = (~df_out["geometry"].astype(str).str.contains(",", na=False)).sum()

    print(
        f"Stats: total_lines={stats.total_lines}, parsed={stats.parsed_rows}, "
        f"no_geom={stats.no_geom}, unparsable_geom={stats.unparsable_geom}, "
        f"one_point_fixed={stats.one_point_fixed}"
    )
    print(f"Sanity: tabs_in_geometry={tabs_in_geom}, missing_closing_paren={end_paren_miss}, no_comma_in_wkt={no_comma}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fix PyPSA line geometries and tags, write clean TSV + optional GeoJSON.")
    parser.add_argument(
        "-i", "--input", default="grid_matcher/data/pypsa_lines_110kv.csv",
        help="Path to source CSV/TSV with malformed geometry."
    )
    parser.add_argument(
        "-o", "--output", default="grid_matcher/data/pypsa_lines_110kv_fixed.csv",
        help="Path to write fixed TSV (geometry as quoted WKT)."
    )
    parser.add_argument(
        "-g", "--geojson", default="grid_matcher/data/pypsa_lines_110kv_fixed.geojson",
        help="Optional GeoJSON output path (set to '' to skip)."
    )
    args = parser.parse_args()
    geojson_out = args.geojson if args.geojson.strip() else None
    main(args.input, args.output, geojson_out)
