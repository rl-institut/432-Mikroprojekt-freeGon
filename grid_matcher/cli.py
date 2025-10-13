"""Command-line interface for Grid Matcher."""

import os
import sys
import argparse
from pathlib import Path
import logging

from .io.loaders import load_data, load_dc_links, load_110kv_data
from .core.matcher import GridMatcher
from .utils.logger import setup_logger, install_stdout_tee
from .io.exporters import create_results_csv, generate_pypsa_with_eic
from .visualization.maps import create_jao_pypsa_visualization
from .config import DEFAULT_OUTPUT_DIR, DEBUG_LOG_PATH

def setup_parser():
    """Set up the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Grid Matcher: Match PyPSA lines to JAO lines"
    )

    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Parser for the 'match' command
    match_parser = subparsers.add_parser("match", help="Run the matching algorithm")

    # Input arguments
    match_parser.add_argument("--jao-path", type=str, required=True,
                             help="Path to JAO lines CSV file")
    match_parser.add_argument("--pypsa-path", type=str, required=True,
                             help="Path to PyPSA lines CSV file")

    # Optional data sources
    match_parser.add_argument("--include-dc", action="store_true",
                             help="Include DC links")
    match_parser.add_argument("--pypsa-dc-path", type=str, default="pypsa-links.csv",
                             help="Path to DC links CSV file")
    match_parser.add_argument("--include-110kv", action="store_true",
                             help="Include 110kV PyPSA dataset")
    match_parser.add_argument("--include-unmatched-110kv", action="store_true",
                             help="Include unmatched 110kV JAO lines in results")
    match_parser.add_argument("--pypsa-110kv-path", type=str, default="pypsa-110.csv",
                             help="Path to pypsa-110.csv file")

    # Output arguments
    match_parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR,
                             help="Directory to save output files")

    # Algorithm parameters
    match_parser.add_argument("--endpoint-grid-m", type=float, default=50.0,
                             help="Grid size (meters) for endpoint snapping")
    match_parser.add_argument("--match-buffer-m", type=float, default=200.0,
                             help="Buffer size (meters) for line matching")
    match_parser.add_argument("--min-coverage", type=float, default=0.35,
                             help="Minimum coverage ratio for chain matching")
    match_parser.add_argument("--target-ratio", type=float, default=0.80,
                             help="Target ratio for extending short matches")

    # Verbosity
    match_parser.add_argument("--verbose", action="store_true",
                             help="Enable verbose output")
    match_parser.add_argument("--debug", action="store_true",
                             help="Enable debug mode with detailed logging")

    # Add matcher mode option to match parser
    match_parser.add_argument("--matcher-mode", action="store_true",
                              help="Use the original matching algorithm with higher match rate")

    # Clean command parser
    clean_parser = subparsers.add_parser("clean", help="Generate a clean PyPSA CSV file")

    clean_parser.add_argument("--original-pypsa", type=str, required=True,
                             help="Path to the original PyPSA CSV file")
    clean_parser.add_argument("--pypsa-with-eic", type=str, required=True,
                             help="Path to the PyPSA with EIC codes file")
    clean_parser.add_argument("--output", type=str, default="output/clean_pypsa.csv",
                             help="Path to save the clean PyPSA file")
    clean_parser.add_argument("--include-110kv", action="store_true", default=True,
                             help="Include 110kV and below lines")
    clean_parser.add_argument("--exclude-110kv", action="store_false", dest="include_110kv",
                             help="Exclude 110kV and below lines")
    clean_parser.add_argument("--include-dc", action="store_true", default=True,
                             help="Include DC lines")
    clean_parser.add_argument("--exclude-dc", action="store_false", dest="include_dc",
                             help="Exclude DC lines")
    clean_parser.add_argument("--pypsa-110kv-path", type=str, default="pypsa-110.csv",
                             help="Path to the 110kV PyPSA file")
    clean_parser.add_argument("--pypsa-dc-path", type=str, default="pypsa-links.csv",
                             help="Path to the DC links file")

    return parser

def run_matching(args):
    """Run the matching algorithm with the given arguments."""
    # Setup logging
    os.makedirs(args.output_dir, exist_ok=True)
    log_level = logging.DEBUG if args.debug else (logging.INFO if args.verbose else logging.WARNING)

    if args.debug:
        log_path = os.path.join(args.output_dir, DEBUG_LOG_PATH)
        install_stdout_tee(log_path)
        print(f"DEBUG mode enabled: logging to {log_path}")

    setup_logger(log_level, os.path.join(args.output_dir, "grid_matcher.log"))

    # Load data
    jao_gdf, pypsa_gdf = load_data(args.jao_path, args.pypsa_path, verbose=args.verbose)

    # Load additional data if requested
    dc_gdf = None
    pypsa_110_gdf = None

    if args.include_dc:
        dc_gdf = load_dc_links(args.pypsa_dc_path, verbose=args.verbose)
        if dc_gdf is not None and not dc_gdf.empty:
            # Combine with main pypsa_gdf for matching
            pypsa_gdf = pypsa_gdf.append(dc_gdf)

    if args.include_110kv:
        pypsa_110_gdf = load_110kv_data(args.pypsa_110kv_path, verbose=args.verbose)
        if pypsa_110_gdf is not None and not pypsa_110_gdf.empty:
            # Combine with main pypsa_gdf for matching
            pypsa_gdf = pypsa_gdf.append(pypsa_110_gdf)

    # Choose matching method based on mode
    if args.matcher_mode:
        print("Using matcher matching algorithm (original high match rate)")
        from grid_matcher.matcher.original_matcher import run_original_matching

        results = run_original_matching(
            jao_gdf,
            pypsa_gdf,
            args.output_dir,
            endpoint_grid_m=args.endpoint_grid_m,
            match_buffer_m=args.match_buffer_m,
            min_coverage=args.min_coverage,
            target_ratio=args.target_ratio,
            verbose=args.verbose
        )
    else:
        # Use the structured GridMatcher implementation
        matcher = GridMatcher(
            jao_gdf,
            pypsa_gdf,
            endpoint_grid_m=args.endpoint_grid_m,
            match_buffer_m=args.match_buffer_m,
            min_coverage=args.min_coverage,
            target_ratio=args.target_ratio
        )
        results = matcher.run_matching()



    # Create visualization
    map_file = os.path.join(args.output_dir, 'jao_pypsa_matches.html')
    create_jao_pypsa_visualization(jao_gdf, pypsa_gdf, results, map_file)

    # Export results to CSV
    csv_file = os.path.join(args.output_dir, 'jao_pypsa_matches.csv')
    create_results_csv(results, csv_file)

    # Generate PyPSA with EIC codes
    pypsa_match_count, pypsa_with_eic, pypsa_eic_files = generate_pypsa_with_eic(
        results, jao_gdf, pypsa_gdf, args.output_dir
    )

    # Print summary
    matched_count = sum(1 for m in results if m.get('matched', False))
    print(f"\nMatching Results Summary:")
    print(f"  Total JAO Lines: {len(jao_gdf)}")
    print(f"  Total PyPSA Lines: {len(pypsa_gdf)}")
    print(f"  Matched JAO Lines: {matched_count}/{len(jao_gdf)} ({matched_count/len(jao_gdf)*100:.1f}%)")
    print(f"  Matched PyPSA Lines: {pypsa_match_count}/{len(pypsa_gdf)} ({pypsa_match_count/len(pypsa_gdf)*100:.1f}%)")

    return results

def main():
    """Main entry point for the command-line interface."""
    parser = setup_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 1

    if args.command == "match":
        run_matching(args)
    elif args.command == "clean":
        from .io.exporters import generate_clean_pypsa_csv
        generate_clean_pypsa_csv(
            args.original_pypsa,
            args.pypsa_with_eic,
            args.output,
            args.include_110kv,
            args.include_dc,
            args.pypsa_110kv_path,
            args.pypsa_dc_path
        )
    else:
        print(f"Unknown command: {args.command}")
        parser.print_help()
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())