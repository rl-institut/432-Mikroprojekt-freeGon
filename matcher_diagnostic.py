#!/usr/bin/env python
# matcher_diagnostic.py

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from pathlib import Path
from shapely import wkt

# Import the necessary functions
from grid_matcher.utils.helpers import calculate_length_meters, get_start_point, get_end_point

# Define paths
DATA_DIR = Path("grid_matcher/data")
OUTPUT_DIR = Path("output/matcher")
OSM_PATH = DATA_DIR / "osm_lines.csv"
PYPSA_PATH = DATA_DIR / "pypsa_lines_110kv_fixed.csv"

print("===== DIAGNOSTIC TOOL FOR OSM-PYPSA MATCHING =====")

# Load OSM data
print("\nLoading OSM data...")
osm_df = pd.read_csv(OSM_PATH)
# Parse geometry
osm_df['geometry'] = osm_df['geom'].apply(lambda g: wkt.loads(g) if isinstance(g, str) and 'LINESTRING' in g else None)
osm_gdf = gpd.GeoDataFrame(osm_df, geometry='geometry')
print(f"Loaded {len(osm_gdf)} OSM lines")

# Load PyPSA data
print("\nLoading PyPSA data...")
pypsa_df = pd.read_csv(PYPSA_PATH)
pypsa_df['geometry'] = pypsa_df['geometry'].apply(lambda g: wkt.loads(g) if isinstance(g, str) else None)
pypsa_gdf = gpd.GeoDataFrame(pypsa_df, geometry='geometry')
print(f"Loaded {len(pypsa_gdf)} PyPSA lines")

print("\n===== DIAGNOSTIC INFO =====")

# Check CRS
print(f"OSM CRS: {osm_gdf.crs}")
print(f"PyPSA CRS: {pypsa_gdf.crs}")

# Enforce CRS if needed
if osm_gdf.crs is None:
    osm_gdf = osm_gdf.set_crs("EPSG:4326")
    print("Set OSM CRS to EPSG:4326")
if pypsa_gdf.crs is None:
    pypsa_gdf = pypsa_gdf.set_crs("EPSG:4326")
    print("Set PyPSA CRS to EPSG:4326")

# Check bounds
osm_bounds = osm_gdf.total_bounds if len(osm_gdf) > 0 else "Empty"
pypsa_bounds = pypsa_gdf.total_bounds if len(pypsa_gdf) > 0 else "Empty"
print(f"OSM bounds: {osm_bounds}")
print(f"PyPSA bounds: {pypsa_bounds}")

# Check geometry types
osm_geom_types = osm_gdf.geometry.apply(lambda g: g.geom_type if g is not None else None).value_counts()
pypsa_geom_types = pypsa_gdf.geometry.apply(lambda g: g.geom_type if g is not None else None).value_counts()
print(f"OSM geometry types: {osm_geom_types}")
print(f"PyPSA geometry types: {pypsa_geom_types}")

# Test endpoint extraction on first few rows
print("\nEndpoint Extraction Test:")
for i, (name, dataset) in enumerate([("OSM", osm_gdf), ("PyPSA", pypsa_gdf)]):
    if len(dataset) > 0:
        for j in range(min(3, len(dataset))):
            geom = dataset.iloc[j].geometry
            if geom is not None:
                start = get_start_point(geom)
                end = get_end_point(geom)
                print(f"{name} row {j} - Start: {start}, End: {end}")

# Create a debug visualization
print("\nCreating debug visualization...")
try:
    fig, ax = plt.subplots(figsize=(12, 8))
    osm_sample = osm_gdf.sample(min(1000, len(osm_gdf))) if len(osm_gdf) > 1000 else osm_gdf
    pypsa_sample = pypsa_gdf.sample(min(1000, len(pypsa_gdf))) if len(pypsa_gdf) > 1000 else pypsa_gdf

    osm_sample.plot(ax=ax, color='blue', alpha=0.5, label='OSM')
    pypsa_sample.plot(ax=ax, color='red', alpha=0.5, label='PyPSA')

    plt.title('OSM and PyPSA Lines Overlay')
    plt.legend()

    # Save the plot
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    debug_map = OUTPUT_DIR / 'debug_map.png'
    plt.savefig(debug_map)
    print(f"Debug map saved to {debug_map}")
except Exception as e:
    print(f"Error creating visualization: {e}")

# Check for any null geometries
osm_null = osm_gdf[osm_gdf.geometry.isna()].shape[0]
pypsa_null = pypsa_gdf[pypsa_gdf.geometry.isna()].shape[0]
print(f"\nNull geometries - OSM: {osm_null}, PyPSA: {pypsa_null}")

# Check for actual spatial overlap
print("\nChecking for spatial overlap...")
try:
    from shapely.geometry import box

    osm_envelope = box(*osm_bounds)
    pypsa_envelope = box(*pypsa_bounds)

    overlap = osm_envelope.intersection(pypsa_envelope)
    overlap_area = overlap.area
    osm_area = osm_envelope.area
    pypsa_area = pypsa_envelope.area

    print(f"Overlap area: {overlap_area}")
    print(f"OSM area: {osm_area}")
    print(f"PyPSA area: {pypsa_area}")
    print(f"Overlap percentage of OSM: {(overlap_area / osm_area) * 100:.2f}%")
    print(f"Overlap percentage of PyPSA: {(overlap_area / pypsa_area) * 100:.2f}%")

    # Test a small buffer match to see if any would match
    print("\nTesting a simple buffer match on sample...")
    matches_found = 0

    for i, osm_row in osm_sample.iterrows():
        if osm_row.geometry is None:
            continue

        osm_buffer = osm_row.geometry.buffer(0.01)  # ~1km buffer

        for j, pypsa_row in pypsa_sample.iterrows():
            if pypsa_row.geometry is None:
                continue

            if osm_buffer.intersects(pypsa_row.geometry):
                matches_found += 1
                print(f"Found potential match: OSM {i} -> PyPSA {j}")
                if matches_found >= 5:  # Just show a few
                    break

        if matches_found >= 5:
            break

    print(f"Found {matches_found} potential matches in sample")

except Exception as e:
    print(f"Error in spatial analysis: {e}")

print("\n===== DIAGNOSTIC COMPLETE =====")
print("Based on this analysis:")
print("1. Check if the coordinate systems are aligned")
print("2. Verify the geometries are properly formed")
print("3. Check if the datasets actually cover the same area")
print("4. Consider increasing the buffer distance for matching")