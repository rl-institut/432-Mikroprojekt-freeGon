#!/usr/bin/env python3
import os
import argparse
import logging


# scripts/run_matching.py
import yaml
import sys
import pandas as pd
from pathlib import Path
import time
import geopandas as gpd

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loaders import load_data, load_pypsa_eur_data, load_tso_data, load_germany_boundary, load_tennet_data, create_tennet_geojson
from src.utils.geometry import clip_to_germany_strict
from src.matching.dlr_matching import match_lines_detailed
from src.matching.pypsa_matching import match_pypsa_eur_lines
from src.matching.tso_matching import match_fifty_hertz_lines, match_tennet_lines
from src.data.exporters import export_results
from src.visualization.charts import generate_parameter_comparison_charts
from src.visualization.maps import create_comprehensive_map, build_network_chords
from src.data.processors import filter_lines_by_voltage
from src.matching.hybrid_matching import match_lines_real_and_chord
from src.matching.transformer_matching import match_transformers
from src.data.transformers import (
    load_dlr_transformers,
    load_network_transformers,
    clip_to_germany as clip_trf_to_germany,
)




logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
#  Transformer helpers – 100 % self-contained
# ──────────────────────────────────────────────────────────────
from pathlib import Path
import logging
import pandas as pd
import geopandas as gpd
from shapely import wkt, wkb

log = logging.getLogger(__name__)


def _to_shape(txt: str):
    """
    Convert one string to a Shapely geometry.

    • Accepts either WKT (e.g. 'POINT (13 52)')
      or WKB **hex** (e.g. '0101000020E610...').

    • Returns None on any parsing error → caller will drop invalid rows.
    """
    if not isinstance(txt, str) or not txt.strip():
        return None

    # try WKT first (fast for POINT / LINESTRING / …)
    try:
        return wkt.loads(txt)
    except Exception:
        pass

    # fall back to WKB-hex
    try:
        return wkb.loads(bytes.fromhex(txt))
    except Exception:
        return None




def prepare_tennet_data(input_file, output_file='tennet_fixed.csv'):
    """Prepare TenneT data by renaming duplicate column names"""
    logger.info(f"Preparing TenneT data from {input_file}")

    try:
        # Read the CSV file
        df = pd.read_csv(input_file)
        logger.info(f"Loaded {len(df)} rows from TenneT data")

        # Get the column names
        columns = list(df.columns)
        logger.info(f"Original columns: {columns}")

        # Handle duplicate column names
        new_columns = []
        col_counts = {}

        for col in columns:
            if col in col_counts:
                col_counts[col] += 1
                new_columns.append(f"{col}.{col_counts[col]}")
            else:
                col_counts[col] = 0
                new_columns.append(col)

        # Rename the columns in the DataFrame
        df.columns = new_columns
        logger.info(f"New columns: {new_columns}")

        # Save the fixed data
        df.to_csv(output_file, index=False)
        logger.info(f"Fixed TenneT data saved to {output_file}")

        return output_file
    except Exception as e:
        logger.error(f"Error preparing TenneT data: {e}")
        return input_file


def parse_args():
    """
    Parse command line arguments.

    Returns:
    - Parsed arguments object
    """
    parser = argparse.ArgumentParser(
        description="Grid Matching Tool - Match grid lines from different datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config/default_config.yaml",
        help="Path to configuration YAML file"
    )

    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set logging level"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        help="Override output directory from config"
    )

    parser.add_argument(
        "--dlr-file",
        type=str,
        help="Override DLR input file from config"
    )

    parser.add_argument(
        "--network-file",
        type=str,
        help="Override network input file from config"
    )

    parser.add_argument(
        "--pypsa-eur-file",
        type=str,
        help="Override PyPSA-EUR input file from config"
    )

    parser.add_argument(
        "--skip-visualizations",
        action="store_true",
        help="Skip generating visualizations"
    )

    parser.add_argument(
        "--use-preclipped",
        action="store_true",
        help="Use pre-clipped data if available"
    )

    return parser.parse_args()


def load_config(config_path):
    """
    Load configuration from YAML file.

    Parameters:
    - config_path: Path to the YAML configuration file

    Returns:
    - Dictionary containing configuration settings
    """
    logger.info(f"Loading configuration from {config_path}")
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        logger.info("Creating a default configuration")
        default_config = {
            "paths": {
                "input": {
                    "dlr_file": "data/input/dlr-lines.csv",
                    "network_file": "data/input/network-lines.csv",
                    "pypsa_eur_file": "data/input/pypsa-eur-lines.csv",
                    "fifty_hertz_file": "data/input/50hertz-lines.csv",
                    "tennet_file": "data/input/tennet-lines.csv",
                    "boundary_file": "data/input/germany-boundary.geojson"
                },
                "output": {
                    "matches_dir": "output/matches/",
                    "charts_dir": "output/charts/",
                    "maps_dir": "output/maps/"
                },
                "processed": {
                    "clipped_dir": "data/clipped/"
                }
            },
            "matching": {
                "min_line_length_km": 1.0,
                "filter_voltage": 110.0,
                "dlr": {
                    "buffer_distance": 0.020,
                    "snap_distance": 0.009,
                    "direction_threshold": 0.2,
                    "enforce_voltage_matching": False
                },
                "pypsa_eur": {
                    "max_distance": 0.0001,
                    "max_hausdorff": 0.001,
                    "relaxed_max_distance": 0.001,
                    "relaxed_max_hausdorff": 0.005
                },
                "fifty_hertz": {
                    "buffer_distance": 0.020,
                    "snap_distance": 0.010,
                    "direction_threshold": 0.65,
                    "enforce_voltage_matching": True
                },
                "tennet": {
                    "buffer_distance": 0.020,
                    "snap_distance": 0.010,
                    "direction_threshold": 0.65,
                    "enforce_voltage_matching": True
                }
            },
            "visualization": {
                "map": {
                    "start_zoom": 6,
                    "tiles": "CartoDB positron",
                    "germany_center": [51.1657, 10.4515]
                },
                "charts": {
                    "dpi": 300,
                    "width": 10,
                    "height": 8
                }
            },
            "logging": {
                "level": "INFO",
                "file": "output/grid_matching.log"
            }
        }

        # Create config directory if it doesn't exist
        os.makedirs("config", exist_ok=True)

        # Save default config
        with open("config/default_config.yaml", 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False, sort_keys=False)

        logger.info("Created default configuration at config/default_config.yaml")
        return default_config
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        raise







def setup_logging(config, args):
    """
    Set up logging based on configuration and command line arguments.

    Parameters:
    - config: Configuration dictionary
    - args: Command line arguments
    """
    # Get log level from args or config
    log_level = args.log_level or config.get('logging', {}).get('level', 'INFO')

    # Set up root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level))

    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler if specified in config
    log_file = config.get('logging', {}).get('file')
    if log_file:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def create_directories(config):
    """
    Create necessary directories specified in the configuration.

    Parameters:
    - config: Configuration dictionary
    """
    # Create output directories
    os.makedirs(config['paths']['output']['matches_dir'], exist_ok=True)
    os.makedirs(config['paths']['output']['charts_dir'], exist_ok=True)
    os.makedirs(config['paths']['output']['maps_dir'], exist_ok=True)

    # Create processed data directories
    os.makedirs(config['paths']['processed']['clipped_dir'], exist_ok=True)

    # Create directory structure for charts
    for dataset in ['dlr', 'pypsa_eur', 'fifty_hertz', 'tennet']:
        os.makedirs(os.path.join(config['paths']['output']['charts_dir'], dataset), exist_ok=True)


import os, logging
from pathlib import Path

logger = logging.getLogger(__name__)

def export_results(df, output_file, index=False):
    """
    Write *df* to CSV.  Creates the parent directory if needed and turns
    any Shapely geometry column into WKT strings.
    """
    try:
        # make sure the destination exists
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)

        # serialise geometry if present
        if "geometry" in df.columns:
            df = df.copy()
            df["geometry"] = df["geometry"].apply(
                lambda g: g.wkt if g is not None else ""
            )

        df.to_csv(output_file, index=index)
        logger.info(f"✔ results written to {output_file}")
    except Exception as e:
        logger.error(f"✖ could not save {output_file}: {e}")
        raise



def main():
    """
    Main function to run the grid line matching process.
    """
    # Start timer
    start_time = time.time()

    # Parse command line arguments
    args = parse_args()

    # Load configuration
    config = load_config(args.config)

    # Set up logging
    setup_logging(config, args)

    logger.info("Starting Grid Matching Tool")

    # Override config with command line arguments if provided
    if args.output_dir:
        config['paths']['output']['matches_dir'] = os.path.join(args.output_dir, 'matches/')
        config['paths']['output']['charts_dir'] = os.path.join(args.output_dir, 'charts/')
        config['paths']['output']['maps_dir'] = os.path.join(args.output_dir, 'maps/')

    if args.dlr_file:
        config['paths']['input']['dlr_file'] = args.dlr_file

    if args.network_file:
        config['paths']['input']['network_file'] = args.network_file

    if args.pypsa_eur_file:
        config['paths']['input']['pypsa_eur_file'] = args.pypsa_eur_file

    # Create necessary directories
    create_directories(config)

    # Define file paths
    dlr_file = config['paths']['input']['dlr_file']
    network_file = config['paths']['input']['network_file']
    dlr_trf_file = "data/input/dlr_transformers.csv"
    net_trf_file = "data/input/network_transformers.csv"
    pypsa_eur_file = config['paths']['input']['pypsa_eur_file']
    fifty_hertz_file = config['paths']['input']['fifty_hertz_file']
    tennet_file = config['paths']['input']['tennet_file']
    boundary_file = config['paths']['input']['boundary_file']
    # ───────────────── Germany boundary ────────────────────────────────
    germany_gdf = load_germany_boundary(boundary_file)
    if germany_gdf is None:
        logger.error("Could not load Germany boundary. Exiting.")
        return

    # ───────────────── transformers (need germany_gdf) ─────────────────
    logger.info("Loading transformer CSV files …")
    dlr_trf = clip_trf_to_germany(
        load_dlr_transformers(dlr_trf_file), germany_gdf
    )
    net_trf = clip_trf_to_germany(
        load_network_transformers(net_trf_file), germany_gdf
    )

    logger.info("DLR-TRF: %d points, NET-TRF: %d points", len(dlr_trf), len(net_trf))

    # Define clipped file paths
    clipped_dir = config['paths']['processed']['clipped_dir']
    clipped_dlr_file = os.path.join(clipped_dir, "dlr-lines-germany.csv")
    clipped_network_file = os.path.join(clipped_dir, "network-lines-germany.csv")
    clipped_pypsa_eur_file = os.path.join(clipped_dir, "pypsa-eur-lines-germany.csv")
    clipped_fifty_hertz_file = os.path.join(clipped_dir, "50hertz-lines-germany.csv")
    clipped_tennet_file = os.path.join(clipped_dir, "tennet-lines-germany.csv")

    # Determine whether to use pre-clipped data
    use_preclipped = args.use_preclipped
    if use_preclipped:
        use_preclipped_dlr = os.path.exists(clipped_dlr_file)
        use_preclipped_network = os.path.exists(clipped_network_file)
        use_preclipped_pypsa_eur = os.path.exists(clipped_pypsa_eur_file)
        use_preclipped_fifty_hertz = os.path.exists(clipped_fifty_hertz_file)
        use_preclipped_tennet = os.path.exists(clipped_tennet_file)

        if use_preclipped_dlr and use_preclipped_network:
            logger.info("Using pre-clipped data files for faster processing")
        else:
            logger.warning("Some pre-clipped files not found. Will perform clipping as needed.")
    else:
        use_preclipped_dlr = False
        use_preclipped_network = False
        use_preclipped_pypsa_eur = False
        use_preclipped_fifty_hertz = False
        use_preclipped_tennet = False

    # Choose which files to use
    dlr_file_to_use = clipped_dlr_file if use_preclipped_dlr else dlr_file
    network_file_to_use = clipped_network_file if use_preclipped_network else network_file
    pypsa_eur_file_to_use = clipped_pypsa_eur_file if use_preclipped_pypsa_eur else pypsa_eur_file
    fifty_hertz_file_to_use = clipped_fifty_hertz_file if use_preclipped_fifty_hertz else fifty_hertz_file
    tennet_file_to_use = clipped_tennet_file if use_preclipped_tennet else tennet_file

    # Load Germany boundary
    germany_gdf = load_germany_boundary(boundary_file)
    if germany_gdf is None:
        logger.error("Could not load Germany boundary. Exiting.")
        return

    # Load data
    logger.info("Loading grid data...")

    # Load DLR and network data (required)
    dlr_lines, network_lines = load_data(dlr_file_to_use, network_file_to_use)

    # Load optional datasets
    pypsa_eur_lines = None
    fifty_hertz_lines = None
    tennet_lines = None


    from shapely import wkt

    def _ensure_geodf(df):
        """Turn bare DataFrame (+geometry col) into a proper GeoDataFrame."""
        if isinstance(df, gpd.GeoDataFrame):
            return df
        if "geometry" not in df.columns:
            return df  # nothing we can do
        # dtype will often still be "object" – convert strings → Shapely
        if df["geometry"].dtype == "object" and isinstance(df["geometry"].iloc[0], str):
            df = df.copy()
            df["geometry"] = df["geometry"].apply(wkt.loads)
        return gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")

    # make sure both are GeoDataFrames
    dlr_lines = _ensure_geodf(dlr_lines)
    network_lines = _ensure_geodf(network_lines)

    for g in (dlr_lines, network_lines):
        if g.geometry.dtype == "object":
            g["geometry"] = gpd.GeoSeries.from_wkt(g.geometry)
            g.set_geometry("geometry", inplace=True)

    if os.path.exists(pypsa_eur_file_to_use):
        pypsa_eur_lines = load_pypsa_eur_data(pypsa_eur_file_to_use)

    if os.path.exists(fifty_hertz_file_to_use):
        fifty_hertz_lines = load_tso_data(fifty_hertz_file_to_use, '50Hertz')

    # In your main function, replace the TenneT loading code with:
    tennet_lines = None
    tennet_geojson_file = "data/processed/tennet_lines.geojson"

    # Check if we need to generate the GeoJSON
    if not os.path.exists(tennet_geojson_file) and os.path.exists(tennet_file_to_use):
        logger.info("Generating TenneT GeoJSON from CSV...")
        from src.data.generate_tennet_geojson import generate_tennet_geojson
        generate_tennet_geojson(tennet_file_to_use, tennet_geojson_file)

    # Load the TenneT data from GeoJSON
    if os.path.exists(tennet_geojson_file):
        logger.info(f"Loading TenneT data from GeoJSON: {tennet_geojson_file}")
        tennet_lines = gpd.read_file(tennet_geojson_file)
        logger.info(f"Loaded {len(tennet_lines)} TenneT lines from GeoJSON")
    else:
        logger.warning("TenneT GeoJSON file not found, TenneT data will not be included")
        tennet_lines = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    # Clip to Germany if needed
    min_line_length_km = config['matching']['min_line_length_km']

    # Clip DLR lines if needed
    if not use_preclipped_dlr:
        logger.info("Clipping DLR lines to Germany...")
        dlr_lines = clip_to_germany_strict(dlr_lines, germany_gdf, min_length_km=min_line_length_km)
        # Optionally save clipped data
        if not dlr_lines.empty:
            dlr_lines.to_csv(clipped_dlr_file, index=False)

    # Clip network lines if needed
    if not use_preclipped_network:
        logger.info("Clipping network lines to Germany...")
        network_lines = clip_to_germany_strict(network_lines, germany_gdf, min_length_km=min_line_length_km)
        # Optionally save clipped data
        if not network_lines.empty:
            network_lines.to_csv(clipped_network_file, index=False)

    # In your main function or wherever you process the data
    network_lines = filter_lines_by_voltage(network_lines, min_voltage=110.0)

    # Clip PyPSA-EUR lines if needed
    if pypsa_eur_lines is not None and not use_preclipped_pypsa_eur:
        logger.info("Clipping PyPSA-EUR lines to Germany...")
        pypsa_eur_lines = clip_to_germany_strict(pypsa_eur_lines, germany_gdf, min_length_km=min_line_length_km)
        # Optionally save clipped data
        if not pypsa_eur_lines.empty:
            pypsa_eur_lines.to_csv(clipped_pypsa_eur_file, index=False)

    # Clip 50Hertz lines if needed
    if fifty_hertz_lines is not None and not use_preclipped_fifty_hertz:
        logger.info("Clipping 50Hertz lines to Germany...")
        fifty_hertz_lines = clip_to_germany_strict(fifty_hertz_lines, germany_gdf, min_length_km=min_line_length_km)
        # Optionally save clipped data
        if not fifty_hertz_lines.empty:
            fifty_hertz_lines.to_csv(clipped_fifty_hertz_file, index=False)

    # Clip TenneT lines if needed
    if tennet_lines is not None and not use_preclipped_tennet:
        logger.info("Clipping TenneT lines to Germany...")
        tennet_lines = clip_to_germany_strict(tennet_lines, germany_gdf, min_length_km=min_line_length_km)
        # Optionally save clipped data
        if not tennet_lines.empty:
            tennet_lines.to_csv(clipped_tennet_file, index=False)

    # Store counts for reporting
    dlr_lines_germany_count = len(dlr_lines)
    network_lines_germany_count = len(network_lines)
    pypsa_eur_lines_germany_count = len(pypsa_eur_lines) if pypsa_eur_lines is not None else 0
    fifty_hertz_lines_germany_count = len(fifty_hertz_lines) if fifty_hertz_lines is not None else 0
    tennet_lines_germany_count = len(tennet_lines) if tennet_lines is not None else 0

    # Initialize matched IDs tracking
    all_matched_dlr_ids = set()
    all_matched_pypsa_eur_ids = set()
    all_matched_fifty_hertz_ids = set()
    all_matched_tennet_ids = set()
    all_matched_network_ids = set()

    # Match DLR lines
    logger.info("Matching DLR lines...")
    dlr_cfg = config['matching']['dlr']

    # Generate chord lines
    net_chord = build_network_chords(network_lines)

    # Add necessary columns to chord lines
    net_chord["v_nom"] = network_lines["v_nom"].mean()
    net_chord["r"] = net_chord["x"] = net_chord["b"] = 0  # dummy impedances
    net_chord["s_nom"] = 0

    # Combine with network lines for matching
    combined_network = pd.concat([network_lines, net_chord], ignore_index=True)            # chords only
    logger.info(
        f"Combined network has {len(combined_network)} lines ({len(network_lines)} original + {len(net_chord)} chords)")

    # 1) raw tables -------------------------------------------------------
    df_real, df_chord, best_real, best_chord = match_lines_real_and_chord(
        dlr_lines, network_lines, cfg=dlr_cfg)

    # 2) one-to-one view --------------------------------------------------
    df_real = (df_real.sort_values("score", ascending=False)
               .drop_duplicates("dlr_id", keep="first"))
    df_chord = (df_chord.sort_values("score", ascending=False)
                .drop_duplicates("dlr_id", keep="first"))

    # →  new line: discard chord rows that clash with a real partner
    df_chord = df_chord[~df_chord.dlr_id.isin(df_real.dlr_id)]

    # 3) helper sets & export --------------------------------------------
    # ── 3) helper sets & export  ----------------------------------
    all_matched_dlr_ids.update(df_real.dlr_id.astype(str))
    all_matched_dlr_ids.update(df_chord.dlr_id.astype(str))

    all_matched_network_ids.update(
        nid for nid in df_real.network_id.astype(str)
        if not nid.startswith("chord_")
    )

    matched_chord_ids = set(df_chord.network_id.astype(str))


    # ➜ add this one line -----------------------------------------
    matches_dir = Path(config["paths"]["output"]["matches_dir"])

    # ─── 4) TRANSFORMER matcher ───────────────────────────────
    logger.info("Matching transformers (≤ 10 m)…")
    df_trf_match, matched_dlr_trf, matched_net_trf = match_transformers(
        dlr_trf, net_trf, buffer_m=10.0)

    matches_dir = Path(config["paths"]["output"]["matches_dir"])
    export_results(df_trf_match, matches_dir / "matched_transformers.csv")

    matched_dlr_trf = {str(i) for i in matched_dlr_trf}
    matched_net_trf = {str(i) for i in matched_net_trf}

    logger.info("Network-TRF exported to map: %d  (matched %d)",
                len(net_trf), len(matched_net_trf))
    logger.info("DLR-TRF exported to map:     %d  (matched %d)",
                len(dlr_trf), len(matched_dlr_trf))

    all_matched_dlr_ids.update(matched_dlr_trf)
    all_matched_network_ids.update(matched_net_trf)

    # and replace the two ellipsis lines by these -----------------
    export_results(df_real, output_file=matches_dir / "matched_dlr_real.csv")
    export_results(df_chord, output_file=matches_dir / "matched_dlr_chord.csv")

    dlr_match_rate = 100 * len(all_matched_dlr_ids) / len(dlr_lines) \
        if len(dlr_lines) else 0

    network_coverage_rate = 100 * len(all_matched_network_ids) / len(network_lines) \
        if len(network_lines) else 0

    logger.info(f"DLR Lines   : {len(dlr_lines)}  (matched {len(all_matched_dlr_ids)}, "
                f"{dlr_match_rate:5.2f} %)")
    logger.info(f"Network real: {len(network_lines)}  (covered {len(all_matched_network_ids)}, "
                f"{network_coverage_rate:5.2f} %)")

    # Match PyPSA-EUR lines if available
    final_matches_pypsa_eur = None
    if pypsa_eur_lines is not None and not pypsa_eur_lines.empty:
        logger.info("Matching PyPSA-EUR lines using visual overlap approach...")
        pypsa_config = config['matching']['pypsa_eur']
        final_matches_pypsa_eur = match_pypsa_eur_lines(
            pypsa_eur_lines,
            network_lines,
            config=pypsa_config
        )

        # Process PyPSA-EUR matches
        if not final_matches_pypsa_eur.empty:
            if 'dlr_id' in final_matches_pypsa_eur.columns:
                all_matched_pypsa_eur_ids = set(final_matches_pypsa_eur['dlr_id'].astype(str))
            if 'network_id' in final_matches_pypsa_eur.columns:
                all_matched_network_ids.update(set(final_matches_pypsa_eur['network_id'].astype(str)))

            # Export results
            export_file = os.path.join(config['paths']['output']['matches_dir'], 'matched_pypsa_eur_lines.csv')
            export_results(final_matches_pypsa_eur, output_file=export_file)

            # Generate charts if not skipped
            if not args.skip_visualizations:
                generate_parameter_comparison_charts(
                    final_matches_pypsa_eur,
                    dataset_name='PyPSA-EUR',
                    output_dir=os.path.join(config['paths']['output']['charts_dir'], 'pypsa_eur')
                )

    # Match 50Hertz lines if available
    final_matches_fifty_hertz = None
    if fifty_hertz_lines is not None and not fifty_hertz_lines.empty:
        logger.info("Matching 50Hertz lines...")
        fifty_hertz_config = config['matching']['fifty_hertz']
        final_matches_fifty_hertz, _ = match_fifty_hertz_lines(  # ignore “best”
            fifty_hertz_lines,
            network_lines,
            config=fifty_hertz_config
        )

        # Process 50Hertz matches
        if not final_matches_fifty_hertz.empty:
            if 'dlr_id' in final_matches_fifty_hertz.columns:
                all_matched_fifty_hertz_ids = set(final_matches_fifty_hertz['dlr_id'].astype(str))
            if 'network_id' in final_matches_fifty_hertz.columns:
                all_matched_network_ids.update(set(final_matches_fifty_hertz['network_id'].astype(str)))

            # Export results
            export_file = os.path.join(config['paths']['output']['matches_dir'], 'matched_50hertz_lines.csv')
            export_results(final_matches_fifty_hertz, output_file=export_file)

            # Generate charts if not skipped
            if not args.skip_visualizations:
                generate_parameter_comparison_charts(
                    final_matches_fifty_hertz,
                    dataset_name='50Hertz',
                    output_dir=os.path.join(config['paths']['output']['charts_dir'], '50hertz')
                )

    # Match TenneT lines if available
    final_matches_tennet = None
    if tennet_lines is not None and not tennet_lines.empty:
        logger.info("Matching TenneT lines...")
        tennet_config = config['matching']['tennet']
        final_matches_tennet, _ = match_tennet_lines(
            tennet_lines,
            network_lines,
            config=tennet_config
        )

        # Process TenneT matches
        if not final_matches_tennet.empty:
            if 'dlr_id' in final_matches_tennet.columns:
                all_matched_tennet_ids = set(final_matches_tennet['dlr_id'].astype(str))
            if 'network_id' in final_matches_tennet.columns:
                all_matched_network_ids.update(set(final_matches_tennet['network_id'].astype(str)))

            # Export results
            export_file = os.path.join(config['paths']['output']['matches_dir'], 'matched_tennet_lines.csv')
            export_results(final_matches_tennet, output_file=export_file)

            # Generate charts if not skipped
            if not args.skip_visualizations:
                generate_parameter_comparison_charts(
                    final_matches_tennet,
                    dataset_name='TenneT',
                    output_dir=os.path.join(config['paths']['output']['charts_dir'], 'tennet')
                )

    # Create comprehensive map if visualizations are not skipped
    if not args.skip_visualizations:
        logger.info("Creating comprehensive map...")

        map_file = os.path.join(config['paths']['output']['maps_dir'], 'comprehensive_grid_map.html')

        print('CHORDS  ➜', len(net_chord), 'features')

        # Create the map with all datasets
        create_comprehensive_map(
            dlr_lines=dlr_lines,
            network_lines=network_lines,
            network_lines_chord= net_chord,
            network_chord_matched=matched_chord_ids,
            matches_dlr_real=df_real,
            matches_dlr_chord=df_chord,
            pypsa_lines=pypsa_eur_lines if pypsa_eur_lines_germany_count > 0 else None,
            matches_pypsa=final_matches_pypsa_eur if not (
                    final_matches_pypsa_eur is None or final_matches_pypsa_eur.empty) else None,
            fifty_hertz_lines=fifty_hertz_lines if fifty_hertz_lines_germany_count > 0 else None,
            tennet_lines=tennet_lines if tennet_lines_germany_count > 0 else None,
            network_trf=net_trf,
            dlr_trf=dlr_trf,
            matched_net_trf_ids=matched_net_trf,
            matched_dlr_trf_ids=matched_dlr_trf,
            matches_fifty_hertz=final_matches_fifty_hertz if not (
                    final_matches_fifty_hertz is None or final_matches_fifty_hertz.empty) else None,
            matches_tennet=final_matches_tennet if not (
                    final_matches_tennet is None or final_matches_tennet.empty) else None,
            germany_gdf=germany_gdf,
            output_file=map_file,
            matched_ids={
                'dlr': all_matched_dlr_ids,
                'pypsa': all_matched_pypsa_eur_ids,
                'fifty_hertz': all_matched_fifty_hertz_ids,
                'tennet': all_matched_tennet_ids,
                'network': all_matched_network_ids
            },
            dlr_lines_germany_count=dlr_lines_germany_count,
            network_lines_germany_count=network_lines_germany_count,
            pypsa_lines_germany_count=pypsa_eur_lines_germany_count,
            fifty_hertz_lines_germany_count=fifty_hertz_lines_germany_count,
            tennet_lines_germany_count=tennet_lines_germany_count,
            detect_connections=True  # Enable automatic connection detection
        )

        logger.info(f"Comprehensive map saved to {map_file}")





    # #Calculate match rates
    dlr_match_rate = len(all_matched_dlr_ids) / dlr_lines_germany_count * 100 if dlr_lines_germany_count > 0 else 0
    pypsa_eur_match_rate = len(
        all_matched_pypsa_eur_ids) / pypsa_eur_lines_germany_count * 100 if pypsa_eur_lines_germany_count > 0 else 0
    fifty_hertz_match_rate = len(
        all_matched_fifty_hertz_ids) / fifty_hertz_lines_germany_count * 100 if fifty_hertz_lines_germany_count > 0 else 0
    tennet_match_rate = len(
        all_matched_tennet_ids) / tennet_lines_germany_count * 100 if tennet_lines_germany_count > 0 else 0
    network_coverage_rate = len(
        all_matched_network_ids) / network_lines_germany_count * 100 if network_lines_germany_count > 0 else 0

    # Print final statistics
    logger.info("\n============ FINAL STATISTICS ============")
    logger.info(
        f"DLR Lines: {dlr_lines_germany_count} (Matched: {len(all_matched_dlr_ids)}, Rate: {dlr_match_rate:.2f}%)")
    logger.info(
        f"Network Lines: {network_lines_germany_count} (Matched to any: {len(all_matched_network_ids)}, Rate: {network_coverage_rate:.2f}%)")

    if pypsa_eur_lines is not None:
        logger.info(
            f"PyPSA-EUR Lines: {pypsa_eur_lines_germany_count} (Matched: {len(all_matched_pypsa_eur_ids)}, Rate: {pypsa_eur_match_rate:.2f}%)")

    if fifty_hertz_lines is not None:
        logger.info(
            f"50Hertz Lines: {fifty_hertz_lines_germany_count} (Matched: {len(all_matched_fifty_hertz_ids)}, Rate: {fifty_hertz_match_rate:.2f}%)")

    if tennet_lines is not None:
        logger.info(
            f"TenneT Lines: {tennet_lines_germany_count} (Matched: {len(all_matched_tennet_ids)}, Rate: {tennet_match_rate:.2f}%)")

    # Calculate run time
    end_time = time.time()
    run_time = end_time - start_time
    hours, remainder = divmod(run_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    logger.info(f"Total run time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")

    logger.info("=========================================")
    logger.info("Grid Matching Tool completed successfully")


if __name__ == "__main__":
    main()