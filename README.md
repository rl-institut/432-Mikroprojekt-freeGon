# Grid Matching Tool

A tool for matching grid line data from different datasets (PyPSA-EUR, JAO, 50Hertz, TenneT) to reference network data for Germany.

## Features

- Load and process line data from different sources (JAO, PyPSA-EUR, 50Hertz, TenneT)
- Clip line geometries to Germany's boundaries
- Match lines between datasets using visual overlap detection
- Generate parameter comparison charts
- Create interactive maps for visualization
- Export matching results for further processing

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/grid-matching-tool.git
cd grid-matching-tool

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

## Data Sources

- Transmission lines and transformers in the Core-TSO (JAO): [https://www.jao.eu/static-grid-model](https://www.jao.eu/static-grid-model). This data is of public domain. Second release. Accessed: 06.09.2022.
- PyPSA Eur lines: Prebuilt Electricity Network for PyPSA-Eur based on OpenStreetMap Data (Published November 13, 2024 | Version 0.6) (https://zenodo.org/records/14144752)
