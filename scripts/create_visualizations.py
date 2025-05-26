#!/usr/bin/env python3
import os
import argparse
import logging
import yaml
import pandas as pd
import geopandas as gpd
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loaders import load_germany_boundary
from src.visualization.charts import generate_parameter_comparison_charts
from src.visualization.maps import create_comprehensive_map

logger = logging.getLogger(__name__)


def parse_args():
    """
    Parse command line arguments.

    Returns:
    - Parsed arguments object
    """
    parser = argparse.ArgumentParser(
        description="Grid Data Visualization Tool",
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
        "--input-dir",
        type=str,
        help="Input directory containing data and match files"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for visualizations"
    )

    parser.add_argument(
        "--visualizations",
        type=str,
        choices=["all", "charts", "map"],
        default="all",
        help="Which visualizations to create"
    )

    parser.add_argument(
        "--datasets",
        type=str,
        choices=["all", "dlr", "pypsa", "fifty_hertz", "tennet"],
        default="all",
        help="Which datasets to visualize"
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Force regeneration even if output files exist"
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
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        raise


def setup_logging(level):
    """
    Set up logging with specified level.

    Parameters:
    - level: Logging level (e.g., "INFO", "DEBUG")
    """
    # Set up root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level))

    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Add console handler
    root_logger.addHandler(console_handler)


def load_dataset(file_path, dataset_type="generic"):
    """
    Load a dataset from a CSV or GeoJSON file.

    Parameters:
    - file_path: Path to the data file
    - dataset_type: Type of dataset for specialized loading

    Returns:
    - GeoDataFrame with loaded data
    """
    logger.info(f"Loading {dataset_type} dataset from {file_path}")

    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return None

    try:
        # Determine file type
        if file_path.endswith('.geojson'):
            gdf = gpd.read_file(file_path)
        else:
            # Assume CSV with WKT geometries
            df = pd.read_csv(file_path)

            # Check if geometry column exists
            if 'geometry' in df.columns:
                # Try to convert WKT to geometries
                try:
                    from shapely import wkt
                    df['geometry'] = df['geometry'].apply(lambda x: wkt.loads(x) if isinstance(x, str) else None)
                    gdf = gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")
                except Exception as e:
                    logger.error(f"Error parsing geometries: {e}")
                    return None
            else:
                logger.error("No geometry column found in CSV file")
                return None

        logger.info(f"Successfully loaded {len(gdf)} features")
        return gdf

    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        return None


def load_matches(file_path):
    """
    Load match results from a CSV file.

    Parameters:
    - file_path: Path to the matches CSV file

    Returns:
    - DataFrame with match results
    """
    logger.info(f"Loading matches from {file_path}")

    if not os.path.exists(file_path):
        logger.error(f"Match file not found: {file_path}")
        return None

    try:
        matches = pd.read_csv(file_path)
        logger.info(f"Successfully loaded {len(matches)} matches")
        return matches
    except Exception as e:
        logger.error(f"Error loading matches: {e}")
        return None


def create_charts(matches, dataset_name, output_dir, force=False):
    """
    Create parameter comparison charts for matched lines.

    Parameters:
    - matches: DataFrame with match results
    - dataset_name: Name of the dataset (e.g., 'DLR', 'PyPSA-EUR')
    - output_dir: Directory to save charts
    - force: Whether to force regeneration if files exist

    Returns:
    - Path to the output directory
    """
    if matches is None or matches.empty:
        logger.warning(f"No matches available for {dataset_name}. Skipping chart generation.")
        return None

    # Define chart filenames
    r_chart = os.path.join(output_dir, f"{dataset_name.lower()}_r_parameter_comparison.png")
    x_chart = os.path.join(output_dir, f"{dataset_name.lower()}_x_parameter_comparison.png")
    b_chart = os.path.join(output_dir, f"{dataset_name.lower()}_b_parameter_comparison.png")
    ratio_chart = os.path.join(output_dir, f"{dataset_name.lower()}_parameter_ratio_histograms.png")

    # Check if output files already exist
    if not force and all(os.path.exists(chart) for chart in [r_chart, x_chart, b_chart, ratio_chart]):
        logger.info(f"Chart files for {dataset_name} already exist. Use --force to regenerate.")
        return output_dir

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Generate charts
    logger.info(f"Generating parameter comparison charts for {dataset_name}")
    try:
        generate_parameter_comparison_charts(matches, dataset_name=dataset_name, output_dir=output_dir)
        logger.info(f"Charts for {dataset_name} saved to {output_dir}")
        return output_dir
    except Exception as e:
        logger.error(f"Error generating charts for {dataset_name}: {e}")
        return None


def main():
    """
    Main function for visualization creation.
    """
    # Parse command line arguments
    args = parse_args()

    # Set up logging
    setup_logging(args.log_level)

    logger.info("Starting Grid Data Visualization Tool")

    # Load configuration
    config = load_config(args.config)

    # Override config with command line arguments if provided
    if args.input_dir:
        # Update paths for input data
        input_dir = args.input_dir
        clipped_dir = os.path.join(input_dir, 'clipped')
        matches_dir = os.path.join(input_dir, 'matches')
    else:
        clipped_dir = config['paths']['processed']['clipped_dir']
        matches_dir = config['paths']['output']['matches_dir']

    if args.output_dir:
        output_dir = args.output_dir
        charts_dir = os.path.join(output_dir, 'charts')
        maps_dir = os.path.join(output_dir, 'maps')
    else:
        charts_dir = config['paths']['output']['charts_dir']
        maps_dir = config['paths']['output']['maps_dir']

    # Create output directories
    os.makedirs(charts_dir, exist_ok=True)
    os.makedirs(maps_dir, exist_ok=True)

    # Determine which visualizations to create
    create_all = args.visualizations == 'all'
    create_charts_vis = create_all or args.visualizations == 'charts'
    create_map_vis = create_all or args.visualizations == 'map'

    # Determine which datasets to visualize
    process_all = args.datasets == 'all'
    process_dlr = process_all or args.datasets == 'dlr'
    process_pypsa = process_all or args.datasets == 'pypsa'
    process_fifty_hertz = process_all or args.datasets == 'fifty_hertz'
    process_tennet = process_all or args.datasets == 'tennet'

    # Define file paths
    dlr_data = os.path.join(clipped_dir, "dlr-lines-germany.csv")
    network_data = os.path.join(clipped_dir, "network-lines-germany.csv")
    pypsa_data = os.path.join(clipped_dir, "pypsa-eur-lines-germany.csv")
    fifty_hertz_data = os.path.join(clipped_dir, "50hertz-lines-germany.csv")
    tennet_data = os.path.join(clipped_dir, "tennet-lines-germany.csv")

    dlr_matches = os.path.join(matches_dir, "matched_dlr_lines.csv")
    pypsa_matches = os.path.join(matches_dir, "matched_pypsa_eur_lines.csv")
    fifty_hertz_matches = os.path.join(matches_dir, "matched_50hertz_lines.csv")
    tennet_matches = os.path.join(matches_dir, "matched_tennet_lines.csv")

    # Load Germany boundary
    boundary_file = config['paths']['input']['boundary_file']
    germany_gdf = load_germany_boundary(boundary_file)
    if germany_gdf is None:
        logger.error("Could not load Germany boundary. Exiting.")
        return

    # Load network data (required for map)
    network_gdf = None
    if os.path.exists(network_data):
        network_gdf = load_dataset(network_data, dataset_type='network')

    if create_map_vis and network_gdf is None:
        logger.error("Network data is required for map visualization. Exiting.")
        return

    # Track what was loaded for comprehensive map
    dlr_gdf = None
    pypsa_gdf = None
    fifty_hertz_gdf = None
    tennet_gdf = None

    dlr_matches_df = None
    pypsa_matches_df = None
    fifty_hertz_matches_df = None
    tennet_matches_df = None

    # Create charts for each dataset
    if create_charts_vis:
        # Create individual chart directories
        os.makedirs(os.path.join(charts_dir, 'dlr'), exist_ok=True)
        os.makedirs(os.path.join(charts_dir, 'pypsa_eur'), exist_ok=True)
        os.makedirs(os.path.join(charts_dir, 'fifty_hertz'), exist_ok=True)
        os.makedirs(os.path.join(charts_dir, 'tennet'), exist_ok=True)

        # DLR charts
        if process_dlr and os.path.exists(dlr_matches):
            dlr_matches_df = load_matches(dlr_matches)
            create_charts(
                dlr_matches_df,
                dataset_name='DLR',
                output_dir=os.path.join(charts_dir, 'dlr'),
                force=args.force
            )

        # PyPSA-EUR charts
        if process_pypsa and os.path.exists(pypsa_matches):
            pypsa_matches_df = load_matches(pypsa_matches)
            create_charts(
                pypsa_matches_df,
                dataset_name='PyPSA-EUR',
                output_dir=os.path.join(charts_dir, 'pypsa_eur'),
                force=args.force
            )

        # 50Hertz charts
        if process_fifty_hertz and os.path.exists(fifty_hertz_matches):
            fifty_hertz_matches_df = load_matches(fifty_hertz_matches)
            create_charts(
                fifty_hertz_matches_df,
                dataset_name='50Hertz',
                output_dir=os.path.join(charts_dir, 'fifty_hertz'),
                force=args.force
            )

        # TenneT charts
        if process_tennet and os.path.exists(tennet_matches):
            tennet_matches_df = load_matches(tennet_matches)
            create_charts(
                tennet_matches_df,
                dataset_name='TenneT',
                output_dir=os.path.join(charts_dir, 'tennet'),
                force=args.force
            )

    # Create comprehensive map
    if create_map_vis:
        # Define map output file
        map_file = os.path.join(maps_dir, 'comprehensive_grid_map.html')

        # Check if map already exists
        if not args.force and os.path.exists(map_file):
            logger.info(f"Map file already exists: {map_file}. Use --force to regenerate.")
        else:
            # Load all necessary datasets for the map
            if process_dlr and os.path.exists(dlr_data):
                dlr_gdf = load_dataset(dlr_data, dataset_type='dlr')
                if dlr_matches_df is None and os.path.exists(dlr_matches):
                    dlr_matches_df = load_matches(dlr_matches)

            if process_pypsa and os.path.exists(pypsa_data):
                pypsa_gdf = load_dataset(pypsa_data, dataset_type='pypsa')
                if pypsa_matches_df is None and os.path.exists(pypsa_matches):
                    pypsa_matches_df = load_matches(pypsa_matches)

            if process_fifty_hertz and os.path.exists(fifty_hertz_data):
                fifty_hertz_gdf = load_dataset(fifty_hertz_data, dataset_type='fifty_hertz')
                if fifty_hertz_matches_df is None and os.path.exists(fifty_hertz_matches):
                    fifty_hertz_matches_df = load_matches(fifty_hertz_matches)

            if process_tennet and os.path.exists(tennet_data):
                tennet_gdf = load_dataset(tennet_data, dataset_type='tennet')
                if tennet_matches_df is None and os.path.exists(tennet_matches):
                    tennet_matches_df = load_matches(tennet_matches)

            # Extract matched IDs
            all_matched_dlr_ids = set()
            all_matched_pypsa_eur_ids = set()
            all_matched_fifty_hertz_ids = set()
            all_matched_tennet_ids = set()
            all_matched_network_ids = set()

            if dlr_matches_df is not None and 'dlr_id' in dlr_matches_df:
                all_matched_dlr_ids = set(dlr_matches_df['dlr_id'].astype(str))
                if 'network_id' in dlr_matches_df:
                    all_matched_network_ids.update(set(dlr_matches_df['network_id'].astype(str)))

            if pypsa_matches_df is not None and 'dlr_id' in pypsa_matches_df:
                all_matched_pypsa_eur_ids = set(pypsa_matches_df['dlr_id'].astype(str))
                if 'network_id' in pypsa_matches_df:
                    all_matched_network_ids.update(set(pypsa_matches_df['network_id'].astype(str)))

            if fifty_hertz_matches_df is not None and 'dlr_id' in fifty_hertz_matches_df:
                all_matched_fifty_hertz_ids = set(fifty_hertz_matches_df['dlr_id'].astype(str))
                if 'network_id' in fifty_hertz_matches_df:
                    all_matched_network_ids.update(set(fifty_hertz_matches_df['network_id'].astype(str)))

            if tennet_matches_df is not None and 'dlr_id' in tennet_matches_df:
                all_matched_tennet_ids = set(tennet_matches_df['dlr_id'].astype(str))
                if 'network_id' in tennet_matches_df:
                    all_matched_network_ids.update(set(tennet_matches_df['network_id'].astype(str)))

            # Get counts
            dlr_lines_germany_count = len(dlr_gdf) if dlr_gdf is not None else 0
            network_lines_germany_count = len(network_gdf) if network_gdf is not None else 0
            pypsa_eur_lines_germany_count = len(pypsa_gdf) if pypsa_gdf is not None else 0
            fifty_hertz_lines_germany_count = len(fifty_hertz_gdf) if fifty_hertz_gdf is not None else 0
            tennet_lines_germany_count = len(tennet_gdf) if tennet_gdf is not None else 0

            # Create the map
            logger.info("Creating comprehensive map...")
            create_comprehensive_map(
                dlr_lines=dlr_gdf,
                network_lines=network_gdf,
                matches_dlr=dlr_matches_df,
                pypsa_lines=pypsa_gdf,
                matches_pypsa=pypsa_matches_df,
                fifty_hertz_lines=fifty_hertz_gdf,
                tennet_lines=tennet_gdf,
                matches_fifty_hertz=fifty_hertz_matches_df,
                matches_tennet=tennet_matches_df,
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
                tennet_lines_germany_count=tennet_lines_germany_count
            )

            logger.info(f"Comprehensive map saved to {map_file}")

    logger.info("Visualization creation complete")


if __name__ == "__main__":
    main()