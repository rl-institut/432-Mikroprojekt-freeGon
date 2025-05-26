import logging
import pandas as pd
import geopandas as gpd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
logger = logging.getLogger(__name__)




def safe_calc_per_km(row: pd.Series, param: str) -> float:
    """
    Safely calculate per-km values handling zero lengths.
    """
    if pd.isna(row[param]) or pd.isna(row['length']) or row['length'] <= 0:
        return 0.0
    return row[param] / row['length']


def filter_lines_by_voltage(network_lines_gdf, min_voltage=110.0, max_voltage=None):
    """Filter network lines by voltage level."""
    logger.info(f"Filtering network lines by voltage (> {min_voltage}kV)")
    original_count = len(network_lines_gdf)

    if 'v_nom' in network_lines_gdf.columns:
        network_lines_gdf['v_nom'] = pd.to_numeric(network_lines_gdf['v_nom'], errors='coerce')

        if max_voltage is None:
            network_lines_filtered = network_lines_gdf[network_lines_gdf['v_nom'] > min_voltage]
        else:
            network_lines_filtered = network_lines_gdf[
                (network_lines_gdf['v_nom'] > min_voltage) &
                (network_lines_gdf['v_nom'] <= max_voltage)
                ]
    else:
        logger.warning("v_nom column not found. Cannot filter by voltage.")
        network_lines_filtered = network_lines_gdf

    after_count = len(network_lines_filtered)
    logger.info(f"Filtered out {original_count - after_count} lines with voltage <= {min_voltage}kV")
    logger.info(f"Remaining network lines: {after_count}")

    return network_lines_filtered