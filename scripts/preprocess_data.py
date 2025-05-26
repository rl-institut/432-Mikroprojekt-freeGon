#!/usr/bin/env python3
import os
import argparse
import logging
import yaml
import sys
import pandas as pd
import geopandas as gpd
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loaders import load_data, load_pypsa_eur_data, load_tso_data, load_germany_boundary
from src.utils.geometry import clip_to_germany_strict
from src.utils.validators import (
    validate_required_columns,
    validate_geometry_column,
    validate_line_geometries,
    validate_numeric_columns,
    validate_coordinate_ranges,
    validate_line_lengths,
    validate_and_set_crs
)

logger = logging.getLogger(__name__)


def parse_args():
    """
    Parse command line arguments.

    Returns:
    - Parsed arguments object
    """
    parser = argparse.ArgumentParser(
        description="Grid Data Preprocessing Tool",
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
        help="Input directory containing data files"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for processed data"
    )

    parser.add_argument(
        "--datasets",
        type=str,
        choices=["all", "dlr", "network", "pypsa", "fifty_hertz", "tennet"],
        default="all",
        help="Which datasets to process"
    )

    parser.add_argument(
        "--min-length",
        type=float,
        help="Minimum line length in km to keep"
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reprocessing even if output files exist"
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


def preprocess_dataset(input_file, output_file, germany_gdf, min_length_km=1.0, dataset_type="generic", force=False):
    """
    Preprocess a dataset by validating, fixing, and clipping to Germany.

    Parameters:
    - input_file: Path to input CSV file
    - output_file: Path to output CSV file
    - germany_gdf: GeoDataFrame containing Germany boundary
    - min_length_km: Minimum line length in kilometers to keep
    - dataset_type: Type of dataset ('dlr', 'network', 'pypsa', etc.)
    - force: Whether to force reprocessing even if output file exists

    Returns:
    - GeoDataFrame with preprocessed data
    """
    if os.path.exists(output_file) and not force:
        logger.info(f"Output file {output_file} already exists. Use --force to reprocess.")
        try:
            # Try to load existing file
            return gpd.read_file(output_file)
        except:
            logger.warning(f"Could not load existing file {output_file}. Reprocessing...")

    logger.info(f"Preprocessing {dataset_type} dataset")
    logger.info(f"Input file: {input_file}")
    logger.info(f"Output file: {output_file}")

    # Check if input file exists
    if not os.path.exists(input_file):
        logger.error(f"Input file not found: {input_file}")
        return None

    # Load data based on dataset type
    try:
        if dataset_type == 'pypsa':
            gdf = load_pypsa_eur_data(input_file)
        elif dataset_type in ['fifty_hertz', 'tennet']:
            gdf = load_tso_data(input_file, dataset_type)
        else:
            # For DLR and network datasets
            if dataset_type == 'dlr':
                dlr_gdf, _ = load_data(input_file, None)
                gdf = dlr_gdf
            elif dataset_type == 'network':
                _, network_gdf = load_data(None, input_file)
                gdf = network_gdf
            else:
                # Generic dataset
                df = pd.read_csv(input_file)
                gdf = gpd.GeoDataFrame(df)
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        return None

    if gdf is None or gdf.empty:
        logger.error(f"Failed to load dataset or dataset is empty: {input_file}")
        return None

    logger.info(f"Loaded {len(gdf)} lines from {input_file}")

    # Validation and cleaning steps
    logger.info("Validating and cleaning data...")

    # Required columns check
    required_columns = ['geometry']
    if not validate_required_columns(gdf, required_columns):
        logger.error(f"Dataset missing required columns. Cannot proceed.")
        return None

    # Validate geometry column
    gdf = validate_geometry_column(gdf)
    if gdf.empty:
        logger.error("No valid geometries found after validation")
        return None

    # Validate line geometries
    gdf = validate_line_geometries(gdf)
    if gdf.empty:
        logger.error("No valid line geometries found after validation")
        return None

    # Validate coordinate ranges
    gdf = validate_coordinate_ranges(gdf)
    if gdf.empty:
        logger.error("No geometries with valid coordinate ranges found")
        return None

    # Numeric columns check
    numeric_columns = ['length', 'r', 'x', 'b', 'v_nom']
    gdf = validate_numeric_columns(gdf, numeric_columns)

    # Validate CRS
    gdf = validate_and_set_crs(gdf)

    # Clip to Germany
    logger.info("Clipping to Germany boundary...")
    gdf = clip_to_germany_strict(gdf, germany_gdf, min_length_km=min_length_km)

    if gdf.empty:
        logger.warning("No lines remain after clipping to Germany")
        return gdf

    # Validate line lengths
    gdf = validate_line_lengths(gdf, min_length_km=min_length_km)

    if gdf.empty:
        logger.warning(f"No lines longer than {min_length_km} km remain after filtering")
        return gdf

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Save preprocessed data
    gdf.to_csv(output_file, index=False)
    logger.info(f"Saved {len(gdf)} preprocessed lines to {output_file}")

    return gdf


def main():
    """
    Main function for data preprocessing.
    """
    # Parse command line arguments
    args = parse_args()

    # Set up logging
    setup_logging(args.log_level)

    logger.info("Starting Grid Data Preprocessing Tool")

    # Load configuration
    config = load_config(args.config)

    # Override config with command line arguments if provided
    if args.input_dir:
        # Update all input file paths
        config['paths']['input']['dlr_file'] = os.path.join(args.input_dir,
                                                            os.path.basename(config['paths']['input']['dlr_file']))
        config['paths']['input']['network_file'] = os.path.join(args.input_dir, os.path.basename(
            config['paths']['input']['network_file']))
        config['paths']['input']['pypsa_eur_file'] = os.path.join(args.input_dir, os.path.basename(
            config['paths']['input']['pypsa_eur_file']))
        config['paths']['input']['fifty_hertz_file'] = os.path.join(args.input_dir, os.path.basename(
            config['paths']['input']['fifty_hertz_file']))
        config['paths']['input']['tennet_file'] = os.path.join(args.input_dir, os.path.basename(
            config['paths']['input']['tennet_file']))

    if args.output_dir:
        config['paths']['processed']['clipped_dir'] = args.output_dir

    if args.min_length:
        config['matching']['min_line_length_km'] = args.min_length

    # Make sure output directory exists
    os.makedirs(config['paths']['processed']['clipped_dir'], exist_ok=True)

    # Define paths
    dlr_input = config['paths']['input']['dlr_file']
    network_input = config['paths']['input']['network_file']
    pypsa_input = config['paths']['input']['pypsa_eur_file']
    fifty_hertz_input = config['paths']['input']['fifty_hertz_file']
    tennet_input = config['paths']['input']['tennet_file']

    dlr_output = os.path.join(config['paths']['processed']['clipped_dir'], "dlr-lines-germany.csv")
    network_output = os.path.join(config['paths']['processed']['clipped_dir'], "network-lines-germany.csv")
    pypsa_output = os.path.join(config['paths']['processed']['clipped_dir'], "pypsa-eur-lines-germany.csv")
    fifty_hertz_output = os.path.join(config['paths']['processed']['clipped_dir'], "50hertz-lines-germany.csv")
    tennet_output = os.path.join(config['paths']['processed']['clipped_dir'], "tennet-lines-germany.csv")

    # Load Germany boundary
    boundary_file = config['paths']['input']['boundary_file']
    germany_gdf = load_germany_boundary(boundary_file)
    if germany_gdf is None:
        logger.error("Could not load Germany boundary. Exiting.")
        return

    # Determine which datasets to process
    process_all = args.datasets == 'all'
    process_dlr = process_all or args.datasets == 'dlr'
    process_network = process_all or args.datasets == 'network'
    process_pypsa = process_all or args.datasets == 'pypsa'
    process_fifty_hertz = process_all or args.datasets == 'fifty_hertz'
    process_tennet = process_all or args.datasets == 'tennet'

    # Get minimum line length
    min_length_km = config['matching']['min_line_length_km']

    # Preprocess datasets
    processed_datasets = 0

    # Process DLR dataset
    if process_dlr and os.path.exists(dlr_input):
        dlr_gdf = preprocess_dataset(
            dlr_input,
            dlr_output,
            germany_gdf,
            min_length_km=min_length_km,
            dataset_type='dlr',
            force=args.force
        )
        if dlr_gdf is not None and not dlr_gdf.empty:
            processed_datasets += 1

    # Process network dataset
    if process_network and os.path.exists(network_input):
        network_gdf = preprocess_dataset(
            network_input,
            network_output,
            germany_gdf,
            min_length_km=min_length_km,
            dataset_type='network',
            force=args.force
        )
        if network_gdf is not None and not network_gdf.empty:
            processed_datasets += 1

    # Process PyPSA-EUR dataset
    if process_pypsa and os.path.exists(pypsa_input):
        pypsa_gdf = preprocess_dataset(
            pypsa_input,
            pypsa_output,
            germany_gdf,
            min_length_km=min_length_km,
            dataset_type='pypsa',
            force=args.force
        )
        if pypsa_gdf is not None and not pypsa_gdf.empty:
            processed_datasets += 1

    # Process 50Hertz dataset
    if process_fifty_hertz and os.path.exists(fifty_hertz_input):
        fifty_hertz_gdf = preprocess_dataset(
            fifty_hertz_input,
            fifty_hertz_output,
            germany_gdf,
            min_length_km=min_length_km,
            dataset_type='fifty_hertz',
            force=args.force
        )
        if fifty_hertz_gdf is not None and not fifty_hertz_gdf.empty:
            processed_datasets += 1

    # Process TenneT dataset
    if process_tennet and os.path.exists(tennet_input):
        tennet_gdf = preprocess_dataset(
            tennet_input,
            tennet_output,
            germany_gdf,
            min_length_km=min_length_km,
            dataset_type='tennet',
            force=args.force
        )
        if tennet_gdf is not None and not tennet_gdf.empty:
            processed_datasets += 1

    logger.info(f"Preprocessing complete. Processed {processed_datasets} datasets.")


if __name__ == "__main__":
    main()