# Grid Matcher

Grid Matcher is a specialized tool for matching transmission line data between JAO (Joint Allocation Office) and PyPSA (Python for Power System Analysis) formats. It provides automated spatial matching with manual override options, enabling accurate conversion between these formats, side-by-side parameter comparisons, and export/visualization utilities.

## Features

- **Automated Line Matching** – Spatially match transmission lines between JAO and PyPSA datasets.  
- **Parameter Transfer** – Transfer electrical parameters between matched lines.  
- **Manual Matching** – Define and apply manual matches for complex cases.  
- **DC Link Support** – Optionally include DC links in matching and outputs.  
- **110 kV Line Support** – Optionally include 110 kV voltage level lines.  
- **Visualization** – Generate interactive maps and parameter comparison visuals.  
- **Comprehensive Reports** – Produce detailed CSV exports and summary tables.

## Requirements

- **Python**: 3.8+  
- **Dependencies**:
  - `pandas`
  - `geopandas`
  - `shapely`
  - `matplotlib`
  - `folium` (for visualization)
  - `numpy`
  - `requests` (for data downloading)
- **Optional** (for troubleshooting / extra visuals):
  - `plotly`

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/yourusername/grid_matcher.git
cd grid_matcher
pip install -r requirements.txt
```

## Data Acquisition

Before running the matcher, you must download the required dataset files from Zenodo. The project includes helper scripts to automate this process.

### Downloading JAO Data

Run the provided script to download JAO transmission line data:

```bash
python scripts/materialize_jao.py
```

This script will:

- Download the JAO dataset from Zenodo record ID `13367535`.  
- Process and standardize the data format.  
- Save the result to `grid_matcher/data/jao_lines.csv`.

> **Note:** The CORE-TSO data has been georeferenced using the methodology described in https://zenodo.org/records/13367535
To use the most up-to-date TSO lines with proper geometry, please follow that methodology; the dataset provided here reflects the state of September 2022.


### Downloading PyPSA Data

To download the PyPSA line data, use the following Python code:

```python
from pathlib import Path
from grid_matcher.io.zenodo_fetch import download_prepare_pypsa_lines_from_zenodo

download_prepare_pypsa_lines_from_zenodo(
    url="https://zenodo.org/records/14144752/files/lines.csv?download=1",
    output_csv=Path("grid_matcher/data/pypsa_lines.csv"),
    verbose=True
)
```

> **Note:** Both download functions have parameters to control Germany-specific clipping, caching, and other options. See the docstrings for details.

## Data Preparation

After downloading the core datasets, you can optionally prepare additional input files:

- `pypsa_lines_110kv.csv` – PyPSA 110 kV line data (optional)  
- `pypsa_dc_links.csv` – PyPSA DC link data (optional)  
- `manual_matches.json` – Manual matches file (created automatically if it doesn't exist)

Additional notes:

- The download scripts ensure both input formats contain geometry information in WKT (Well-Known Text) format and appropriate electrical parameters.  
- Both datasets are standardized to use a consistent CRS (`EPSG:4326`).

## Basic Usage

After downloading the required data, run the matcher with default settings:

```bash
python run_matcher.py
```

This will:

1. Load JAO and PyPSA transmission line data.  
2. Apply any predefined manual matches.  
3. Run the automated matching algorithm.  
4. Generate output files in the `output/matcher` directory.  
5. Create visualizations for parameter comparison.

## Command-Line Options

```text
usage: run_matcher.py [-h] [--include-dc-matching] [--include-110kv-matching] 
                      [--no-dc-output] [--no-110kv-output] [--no-viz] [--quiet]
                      [--manual] [--no-manual] [--add-predefined] [--no-predefined]
                      [--import-new-lines] [--output OUTPUT]

Grid Matcher Mode

options:
  -h, --help            show this help message and exit
  --include-dc-matching
                        Include DC links in matching
  --include-110kv-matching
                        Include 110kV lines in matching
  --no-dc-output        Exclude DC links from output
  --no-110kv-output     Exclude 110kV lines from output
  --no-viz              Skip parameter visualization
  --quiet               Disable verbose output
  --manual              Enable manual matching
  --no-manual           Disable manual matching
  --add-predefined      Add predefined manual matches
  --no-predefined       Don't add predefined matches
  --import-new-lines    Import new lines from CSV files
  --output OUTPUT, -o OUTPUT
                        Output directory (default: output/matcher)
```

## Example Commands

Match only high-voltage AC lines (exclude DC and 110 kV from output):

```bash
python run_matcher.py --no-dc-output --no-110kv-output
```

Include DC links in both matching and output:

```bash
python run_matcher.py --include-dc-matching
```

Disable manual matches entirely:

```bash
python run_matcher.py --no-manual
```

Import new lines before matching:

```bash
python run_matcher.py --import-new-lines
```

## Output Files

Generated in the specified output directory (default: `output/matcher`):

- `jao_pypsa_matches.csv` – All matched lines and their parameters.  
- `pypsa_with_eic.csv` – PyPSA lines with added JAO identifiers.  
- `pypsa_with_eic_enhanced.csv` – Enhanced PyPSA set including DC/110 kV lines.  
- `jao_with_pypsa.csv` – JAO lines with PyPSA electrical parameters.  
- `jao_pypsa_matches.html` – Interactive map visualization of matches.  
- `parameter_comparison.html` – Visualization of electrical parameter comparison.  
- `parameter_summary.html` – Summary table of parameter statistics.

## Manual Matching

For complex cases where automated matching doesn't provide satisfactory results, you can define manual matches in `grid_matcher/manual/manual_matching.py`. Manual matches override automated results for the specified JAO IDs.

Example structure:

```python
# Dictionary of predefined matches (JAO ID -> PyPSA IDs)
predefined_matches = {
    "jao_id": ["pypsa_id1", "pypsa_id2", ...],
    # Example:
    "2611": ["merged_relation/3916226-380-c+3", "merged_way/240543053-1-380-a+5"]
}
```

You can also keep a `manual_matches.json` file (created on first run if missing) that persists manual mappings between runs.

## Importing New Lines

If you have new transmission line data to add:

1. Place new JAO lines in `grid_matcher/data/jao-new-lines.csv`.  
2. Place new PyPSA lines in `grid_matcher/data/pypsa-new-lines.csv`.  
3. Run with the import flag:

```bash
python run_matcher.py --import-new-lines
```

## Troubleshooting

### Missing Geometry Data

- Ensure input files contain valid geometry in WKT format.  
- Verify CRS consistency between datasets.

### No Matches Found

- Adjust matching parameters in `grid_matcher/matcher/original_matcher.py`.  
- Add manual matches for problematic lines.

### Visualization Errors

Ensure visualization dependencies are installed:

```bash
pip install folium matplotlib plotly
```

### Memory Errors with Large Datasets

Exclude 110 kV lines and DC links to reduce memory usage:

```bash
python run_matcher.py --no-dc-output --no-110kv-output
```

### Data Download Errors

- Check internet connection.  
- Verify Zenodo record IDs and URLs are correct.  
- Look in the `.cache/zenodo` directory for cached files that may be used as fallbacks.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request with improvements, bug fixes, or feature proposals.

## License

This project is licensed under the MIT License — see the `LICENSE` file for details.