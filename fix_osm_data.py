#!/usr/bin/env python
# fix_osm_data_wkb.py

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from pathlib import Path
import os
from shapely import wkb
import binascii

DATA_DIR = Path("grid_matcher/data")
OSM_PATH = DATA_DIR / "osm_lines.csv"
FIXED_OSM_PATH = DATA_DIR / "osm_lines_fixed.csv"

print("===== FIXING OSM DATA (WKB FORMAT) =====")

# Load OSM data
print("Loading OSM data...")
osm_df = pd.read_csv(OSM_PATH)
print(f"Loaded {len(osm_df)} OSM lines")

# Check the format of the geometry column
print("\nInspecting geom column format:")
geom_sample = osm_df['geom'].head(3).tolist()
for i, g in enumerate(geom_sample):
    print(f"Row {i}: {type(g).__name__} - {g[:50]}..." if isinstance(g, str) and len(
        g) > 50 else f"Row {i}: {type(g).__name__} - {g}")


# Function to safely parse WKB hex geometry
def parse_wkb_geometry(geom_hex):
    if not isinstance(geom_hex, str):
        return None

    try:
        # Try parsing as WKB hex string
        return wkb.loads(geom_hex, hex=True)
    except Exception as e1:
        # If that fails, try to clean up the hex string
        try:
            # Some WKB hex strings might have unexpected formatting
            clean_hex = geom_hex.strip()
            return wkb.loads(clean_hex, hex=True)
        except Exception as e2:
            print(f"Failed to parse geometry: {e2}, value: {geom_hex[:50]}..." if len(geom_hex) > 50 else geom_hex)
            return None


# Try to parse geometries
print("\nParsing geometries using WKB hex format...")
osm_df['geometry'] = osm_df['geom'].apply(parse_wkb_geometry)

# Count valid geometries
valid_geoms = osm_df['geometry'].notnull().sum()
print(f"Successfully parsed {valid_geoms} out of {len(osm_df)} geometries")

# Create GeoDataFrame
if valid_geoms > 0:
    print("Creating GeoDataFrame...")
    osm_gdf = gpd.GeoDataFrame(osm_df.dropna(subset=['geometry']), geometry='geometry')
    osm_gdf = osm_gdf.set_crs("EPSG:4326")

    # Check bounds after fixing
    print(f"Fixed OSM bounds: {osm_gdf.total_bounds}")
    print(f"Geometry types: {osm_gdf.geometry.geom_type.value_counts()}")

    # Save fixed data
    print(f"Saving {len(osm_gdf)} valid geometries to {FIXED_OSM_PATH}")
    osm_gdf.to_csv(FIXED_OSM_PATH, index=False)

    # Visualize sample of fixed data
    print("Creating sample visualization...")
    sample_size = min(1000, len(osm_gdf))
    sample = osm_gdf.sample(sample_size)

    fig, ax = plt.subplots(figsize=(12, 10))
    sample.plot(ax=ax, color='blue', linewidth=0.5)
    plt.title(f'Sample of {sample_size} valid OSM geometries')

    os.makedirs(DATA_DIR.parent / "output", exist_ok=True)
    viz_path = DATA_DIR.parent / "output" / 'osm_sample_viz.png'
    plt.savefig(viz_path)
    print(f"Visualization saved to {viz_path}")
else:
    print("No valid geometries found. Saving empty dataset.")
    empty_gdf = gpd.GeoDataFrame(osm_df.copy(), geometry='geometry')
    empty_gdf.to_csv(FIXED_OSM_PATH, index=False)

print("\n===== OSM DATA FIX COMPLETE =====")
print(f"Fixed OSM data saved to {FIXED_OSM_PATH}")

# If we're still having issues, try a full diagnostic with the original CSV
if valid_geoms == 0:
    print("\n===== ADDITIONAL DIAGNOSTIC =====")
    print("Examining original CSV file structure...")

    # Get column names and sample values
    print(f"CSV Columns: {osm_df.columns.tolist()}")
    print("\nFirst 5 rows of data:")
    for col in osm_df.columns:
        print(f"\nColumn '{col}' - first 5 values:")
        for i, val in enumerate(osm_df[col].head(5)):
            print(f"  Row {i}: {type(val).__name__} - {val[:50]}..." if isinstance(val, str) and len(
                str(val)) > 50 else f"  Row {i}: {type(val).__name__} - {val}")