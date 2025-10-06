"""Configuration settings and defaults for the Grid Matcher."""

# Default matching parameters
DEFAULT_ENDPOINT_GRID_M = 50.0
DEFAULT_MATCH_BUFFER_M = 200.0
DEFAULT_MIN_COVERAGE = 0.35
DEFAULT_TARGET_RATIO = 0.80
DEFAULT_OVERLAP_IOU = 0.60

# Parallel circuit handling
MAX_ADDITIONS_PER_LINE = 40
MAX_ANGLE_DIFF = 30

# Known corridors with exact mappings
KNOWN_CORRIDORS = [
    {
        'name': 'Gundelfingen-Voehringen',
        'shared_circuit_groups': [
            {
                'jao_ids': ['244', '245'],
                'circuit_capacity': 1,
                'mappings': [
                    {
                        'jao_id': '244',
                        'pypsa_segments': ['relation/1641463-380-a', 'relation/1641463-380-b',
                                          'relation/1641463-380-c', 'relation/1641463-380-e']
                    },
                    {
                        'jao_id': '245',
                        'pypsa_segments': ['relation/1641474-380-a', 'relation/1641474-380-b',
                                          'relation/1641474-380-c', 'relation/1641474-380-e']
                    }
                ]
            }
        ]
    }
]

# Logging configuration
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_LEVEL = "INFO"
DEBUG_LOG_PATH = "output/grid_matcher_debug.log"

# Default output paths
DEFAULT_OUTPUT_DIR = "output"
DEFAULT_VISUALIZATION_FILE = "jao_pypsa_matches.html"
DEFAULT_RESULTS_CSV = "jao_pypsa_matches.csv"
DEFAULT_PARAMS_TABLE = "jao_pypsa_electrical_parameters.html"