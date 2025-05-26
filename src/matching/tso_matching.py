import logging



# Import the match_lines_detailed function
from src.matching.dlr_matching import match_lines_detailed

logger = logging.getLogger(__name__)


def match_fifty_hertz_lines(fifty_hertz_lines, network_lines, config=None):
    """
    Match 50Hertz lines to network lines.

    Parameters:
    - fifty_hertz_lines: GeoDataFrame of 50Hertz lines
    - network_lines: GeoDataFrame of network lines
    - config: Optional configuration dictionary

    Returns:
    - DataFrame of matched line pairs
    """
    if config is None:
        config = {}

    # Get parameters from config or use defaults
    buffer_distance = config.get('buffer_distance', 0.020)
    snap_distance = config.get('snap_distance', 0.010)
    direction_threshold = config.get('direction_threshold', 0.65)
    enforce_voltage_matching = config.get('enforce_voltage_matching', True)

    logger.info(f"Matching 50Hertz lines with buffer distance {buffer_distance}...")

    return match_lines_detailed(
        fifty_hertz_lines,
        network_lines,
        buffer_distance=buffer_distance,
        snap_distance=snap_distance,
        direction_threshold=direction_threshold,
        enforce_voltage_matching=enforce_voltage_matching,
        dataset_name="50Hertz"
    )


def match_tennet_lines(tennet_lines, network_lines, config=None):
    """
    Match TenneT lines to network lines.

    Parameters:
    - tennet_lines: GeoDataFrame of TenneT lines
    - network_lines: GeoDataFrame of network lines
    - config: Optional configuration dictionary

    Returns:
    - DataFrame of matched line pairs
    """
    if config is None:
        config = {}

    # Get parameters from config or use defaults
    buffer_distance = config.get('buffer_distance', 0.020)
    snap_distance = config.get('snap_distance', 0.010)
    direction_threshold = config.get('direction_threshold', 0.65)
    enforce_voltage_matching = config.get('enforce_voltage_matching', True)

    logger.info(f"Matching TenneT lines with buffer distance {buffer_distance}...")

    return match_lines_detailed(
        tennet_lines,
        network_lines,
        buffer_distance=buffer_distance,
        snap_distance=snap_distance,
        direction_threshold=direction_threshold,
        enforce_voltage_matching=enforce_voltage_matching,
        dataset_name="TenneT"
    )