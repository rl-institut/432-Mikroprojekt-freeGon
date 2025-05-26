
import logging
import pandas as pd
from typing import Optional
logger = logging.getLogger(__name__)


def export_results(matches: pd.DataFrame, output_file: str = 'matched_lines.csv') -> Optional[pd.DataFrame]:
    """Export matched lines with their attributes to a CSV file"""
    if len(matches) == 0:
        logger.warning("No matches found, nothing to export.")
        return None

    # Select relevant columns for export
    export_cols = [col for col in [
        'dlr_id', 'network_id', 'dlr_length', 'network_length',
        'overlap_percentage', 'direction_similarity',
        'dlr_r', 'dlr_x', 'dlr_b',
        'network_r', 'network_x', 'network_b',
        'allocated_r', 'allocated_x', 'allocated_b',
        'r_change_pct', 'x_change_pct', 'b_change_pct',
        'dlr_voltage', 'network_voltage', 'allocated_v_nom',
        'TSO'
    ] if col in matches.columns]

    export_df = matches[export_cols].copy()

    # Sort by network_id for easier reading
    export_df = export_df.sort_values('network_id')

    # Export to CSV
    export_df.to_csv(output_file, index=False)
    logger.info(f"Results exported to {output_file}")

    return export_df