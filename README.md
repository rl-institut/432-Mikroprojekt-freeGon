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