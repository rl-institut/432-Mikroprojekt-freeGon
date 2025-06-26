#!/usr/bin/env python3
"""
Transformer Visualization Tool - Germany Edition

This script displays transformers within Germany's boundaries with a clean OpenStreetMap base layer.
"""

import os
import sys
import logging
import pandas as pd
import geopandas as gpd
from pathlib import Path
from shapely import wkt
import folium
from folium.plugins import MarkerCluster, MeasureControl
import argparse
from shapely.geometry import Point

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Find project root directory (2 levels up from the script location)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "input"


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Germany Transformer Visualization")
    parser.add_argument("--dlr-trf", type=str,
                        default=str(DATA_DIR / "dlr_transformers.csv"),
                        help="Path to DLR transformers CSV file")
    parser.add_argument("--net-trf", type=str,
                        default=str(DATA_DIR / "network_transformers.csv"),
                        help="Path to Network transformers CSV file")
    parser.add_argument("--germany-geojson", type=str,
                        default=str(DATA_DIR / "georef-germany-gemeinde@public.geojson"),
                        help="Path to Germany GeoJSON boundary file")
    parser.add_argument("--output", type=str, default="transformer_map.html",
                        help="Output HTML map file")
    parser.add_argument("--buffer", type=float, default=1000.0,
                        help="Buffer distance for matching (meters)")
    return parser.parse_args()


def load_germany_boundary(geojson_path):
    """Load Germany boundary from GeoJSON file"""
    if not os.path.exists(geojson_path):
        logger.error(f"Germany boundary file not found: {geojson_path}")
        # Try to find the file in various locations
        possible_locations = [
            Path.cwd() / "data" / "input" / "georef-germany-gemeinde@public.geojson",
            PROJECT_ROOT / "data" / "input" / "georef-germany-gemeinde@public.geojson",
            Path.cwd() / ".." / "data" / "input" / "georef-germany-gemeinde@public.geojson",
        ]

        for location in possible_locations:
            if location.exists():
                geojson_path = str(location)
                logger.info(f"Found Germany boundary at: {location}")
                break

    try:
        # Load the GeoJSON file
        germany_gdf = gpd.read_file(geojson_path)

        # Dissolve to get a single polygon for all of Germany
        germany_boundary = germany_gdf.dissolve().reset_index(drop=True)

        logger.info(f"Successfully loaded Germany boundary with {len(germany_gdf)} features")
        return germany_boundary
    except Exception as e:
        logger.error(f"Error loading Germany boundary: {e}")
        return None


def load_dlr_transformers(csv_path):
    """Load DLR transformers with special handling for the format"""
    if not os.path.exists(csv_path):
        logger.error(f"File not found: {csv_path}")
        # Try relative path from current directory
        alt_path = os.path.join("../", csv_path)
        if os.path.exists(alt_path):
            logger.info(f"Found file at alternate path: {alt_path}")
            csv_path = alt_path
        else:
            logger.error(f"Also checked {alt_path} but file not found")
            return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    logger.info(f"Loading DLR transformers from {csv_path}")
    try:
        # Read the CSV file
        df = pd.read_csv(csv_path)
        logger.info(f"Read {len(df)} rows from {csv_path}")
        logger.info(f"Columns: {', '.join(df.columns)}")

        # Convert geometry strings to Shapely objects
        if 'geometry' in df.columns:
            df['geometry'] = df['geometry'].apply(lambda x: wkt.loads(x) if isinstance(x, str) else None)
            df = df[df['geometry'].notnull()]

            # Create GeoDataFrame
            gdf = gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")

            # Ensure ID column
            if 'name' in gdf.columns:
                gdf['id'] = gdf['name']
            elif not 'id' in gdf.columns:
                gdf['id'] = [f"dlr_{i}" for i in range(len(gdf))]

            logger.info(f"Successfully loaded {len(gdf)} DLR transformers with geometry")

            # Check if we have points in both Austria and Germany
            logger.info(f"DLR transformer bounds: {gdf.total_bounds}")
            return gdf
        else:
            logger.error("No 'geometry' column found in DLR transformer file")
            return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    except Exception as e:
        logger.error(f"Error loading DLR transformers: {e}")
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")


def load_network_transformers(csv_path):
    """Load network transformers with special handling for their format"""
    if not os.path.exists(csv_path):
        logger.error(f"File not found: {csv_path}")
        # Try relative path from current directory
        alt_path = os.path.join("../", csv_path)
        if os.path.exists(alt_path):
            logger.info(f"Found file at alternate path: {alt_path}")
            csv_path = alt_path
        else:
            logger.error(f"Also checked {alt_path} but file not found")
            return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    logger.info(f"Loading network transformers from {csv_path}")
    try:
        # Read the CSV file
        df = pd.read_csv(csv_path)
        logger.info(f"Read {len(df)} rows from {csv_path}")
        logger.info(f"Columns: {', '.join(df.columns)}")

        # Handle the specific format of network transformers
        if 'geom' in df.columns:
            # Parse WKT geometry
            df['geometry'] = df['geom'].apply(lambda x: wkt.loads(x) if isinstance(x, str) else None)
            df = df[df['geometry'].notnull()]

            # Convert to point geometry (take centroid of multilinestring)
            df['geometry'] = df['geometry'].apply(lambda g:
                                                  Point(g.coords[0]) if g.geom_type == 'LineString' else
                                                  Point(
                                                      g.geoms[0].coords[0]) if g.geom_type == 'MultiLineString' else g)

            # Create GeoDataFrame
            gdf = gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")

            # Ensure ID column
            if not 'id' in gdf.columns:
                gdf['id'] = [f"net_{i}" for i in range(len(gdf))]

            logger.info(f"Successfully loaded {len(gdf)} network transformers with geometry")

            # Check bounds
            logger.info(f"Network transformer bounds: {gdf.total_bounds}")
            return gdf
        else:
            logger.error("No 'geom' column found in network transformer file")
            return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    except Exception as e:
        logger.error(f"Error loading network transformers: {e}")
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")


def clip_to_germany(gdf, germany_boundary):
    """Clip a GeoDataFrame to the Germany boundary"""
    if gdf.empty or germany_boundary is None or germany_boundary.empty:
        return gdf

    # Ensure both have the same CRS
    if gdf.crs != germany_boundary.crs:
        gdf = gdf.to_crs(germany_boundary.crs)

    # Spatial join to check which points are within Germany
    germany_geom = germany_boundary.iloc[0].geometry
    in_germany = gdf.geometry.apply(lambda point: germany_geom.contains(point))

    # Filter to keep only points within Germany
    clipped_gdf = gdf[in_germany].copy()

    # Reset index to ensure continuous indices
    clipped_gdf = clipped_gdf.reset_index(drop=True)

    logger.info(f"Clipped from {len(gdf)} to {len(clipped_gdf)} points within Germany")
    return clipped_gdf


def match_transformers(dlr_gdf, net_gdf, buffer_m=1000.0):
    """Match transformers using a spatial join with buffer"""
    if dlr_gdf.empty or net_gdf.empty:
        logger.warning("Cannot match transformers: one or both datasets are empty")
        return pd.DataFrame(columns=["dlr_id", "network_id", "dist_m"]), set(), set()

    # Project to a metric CRS for distance calculations
    dlr_metric = dlr_gdf.to_crs(epsg=3035)
    net_metric = net_gdf.to_crs(epsg=3035)

    # Initialize results
    matches = []
    used_dlr_ids = set()
    used_net_ids = set()

    # For each DLR transformer, find the closest network transformer within buffer
    for idx_d, row_d in dlr_metric.iterrows():
        dlr_id = row_d["id"]
        if dlr_id in used_dlr_ids:
            continue

        # Create buffer around point
        buffer = row_d.geometry.buffer(buffer_m)

        # Find candidates within buffer
        candidates = []
        for idx_n, row_n in net_metric.iterrows():
            net_id = row_n["id"]
            if net_id in used_net_ids:
                continue

            dist = row_d.geometry.distance(row_n.geometry)
            if dist <= buffer_m:
                candidates.append((idx_n, net_id, dist))

        # Sort by distance and pick closest
        if candidates:
            candidates.sort(key=lambda x: x[2])
            idx_n, net_id, dist = candidates[0]

            # Mark as used
            used_dlr_ids.add(dlr_id)
            used_net_ids.add(net_id)

            # Add to matches
            matches.append({
                "dlr_id": dlr_id,
                "network_id": net_id,
                "dist_m": dist,
                "dlr_lat": row_d.geometry.y,
                "dlr_lon": row_d.geometry.x,
                "net_lat": net_metric.iloc[idx_n].geometry.y,
                "net_lon": net_metric.iloc[idx_n].geometry.x
            })

    # Create match DataFrame
    match_df = pd.DataFrame(matches)

    logger.info(f"Found {len(matches)} transformer matches")
    logger.info(f"Matched {len(used_dlr_ids)}/{len(dlr_gdf)} DLR transformers")
    logger.info(f"Matched {len(used_net_ids)}/{len(net_gdf)} Network transformers")

    return match_df, used_dlr_ids, used_net_ids


def create_transformer_map(dlr_gdf, net_gdf, matched_dlr_ids, matched_net_ids, germany_boundary, output_file):
    """Create an interactive map showing transformers from both datasets"""
    # Set the map center to Germany
    center_lat = 51.1657
    center_lon = 10.4515
    zoom_start = 6

    # Create base map with OpenStreetMap as the base layer
    m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom_start,
                   tiles='OpenStreetMap',
                   attr='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors')

    # Add measurement control
    MeasureControl(position='bottomright', primary_length_unit='kilometers').add_to(m)

    # Add Germany boundary
    if germany_boundary is not None and not germany_boundary.empty:
        style_function = lambda x: {
            'fillColor': 'transparent',
            'color': '#3388ff',
            'weight': 2,
            'fillOpacity': 0
        }
        folium.GeoJson(
            germany_boundary,
            style_function=style_function,
            name="Germany Boundary"
        ).add_to(m)

    # Add marker cluster groups for better visibility
    dlr_matched_cluster = MarkerCluster(name="DLR Transformers (Matched)")
    dlr_unmatched_cluster = MarkerCluster(name="DLR Transformers (Unmatched)")
    net_matched_cluster = MarkerCluster(name="Network Transformers (Matched)")
    net_unmatched_cluster = MarkerCluster(name="Network Transformers (Unmatched)")

    # Debug counters
    dlr_matched_count = 0
    dlr_unmatched_count = 0
    net_matched_count = 0
    net_unmatched_count = 0

    # Add DLR transformers
    for _, row in dlr_gdf.iterrows():
        is_matched = str(row["id"]) in matched_dlr_ids
        lat, lon = row.geometry.y, row.geometry.x

        if is_matched:
            dlr_matched_count += 1
            color = 'blue'
            cluster = dlr_matched_cluster
            popup_text = f"""
            <b>DLR TRF {row['id']} (Matched)</b><br>
            Coordinates: {lat:.6f}, {lon:.6f}<br>
            """
            if 'name' in row and 'Full Name' in row:
                popup_text += f"Name: {row['name']}<br>"
                popup_text += f"Full Name: {row['Full Name']}<br>"
            if 'r' in row and 'x' in row and 'b' in row:
                popup_text += f"r: {row['r']}<br>x: {row['x']}<br>b: {row['b']}<br>"
        else:
            dlr_unmatched_count += 1
            color = 'darkblue'
            cluster = dlr_unmatched_cluster
            popup_text = f"""
            <b>DLR TRF {row['id']} (Unmatched)</b><br>
            Coordinates: {lat:.6f}, {lon:.6f}<br>
            """
            if 'name' in row and 'Full Name' in row:
                popup_text += f"Name: {row['name']}<br>"
                popup_text += f"Full Name: {row['Full Name']}<br>"
            if 'r' in row and 'x' in row and 'b' in row:
                popup_text += f"r: {row['r']}<br>x: {row['x']}<br>b: {row['b']}<br>"

        # Add to marker cluster
        folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(popup_text, max_width=300),
            icon=folium.Icon(color='blue' if is_matched else 'cadetblue',
                             icon='bolt', prefix='fa')
        ).add_to(cluster)

        # Also add circle marker (will be visible at all zoom levels)
        folium.CircleMarker(
            location=[lat, lon],
            radius=8 if is_matched else 6,
            color='black',
            weight=2,
            fill=True,
            fill_color=color,
            fill_opacity=0.9,
            tooltip=f"DLR TRF {row['id']} ({'Matched' if is_matched else 'Unmatched'})"
        ).add_to(m)

    # Add Network transformers
    for _, row in net_gdf.iterrows():
        is_matched = str(row["id"]) in matched_net_ids
        lat, lon = row.geometry.y, row.geometry.x

        if is_matched:
            net_matched_count += 1
            color = 'red'
            cluster = net_matched_cluster
            popup_text = f"""
            <b>Network TRF {row['id']} (Matched)</b><br>
            Coordinates: {lat:.6f}, {lon:.6f}<br>
            """
            if 'bus0' in row and 'bus1' in row:
                popup_text += f"Bus0: {row['bus0']}<br>Bus1: {row['bus1']}<br>"
            if 'r' in row and 'x' in row:
                popup_text += f"r: {row['r']}<br>x: {row['x']}<br>"
        else:
            net_unmatched_count += 1
            color = 'gray'
            cluster = net_unmatched_cluster
            popup_text = f"""
            <b>Network TRF {row['id']} (Unmatched)</b><br>
            Coordinates: {lat:.6f}, {lon:.6f}<br>
            """
            if 'bus0' in row and 'bus1' in row:
                popup_text += f"Bus0: {row['bus0']}<br>Bus1: {row['bus1']}<br>"
            if 'r' in row and 'x' in row:
                popup_text += f"r: {row['r']}<br>x: {row['x']}<br>"

        # Add to marker cluster
        folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(popup_text, max_width=300),
            icon=folium.Icon(color='red' if is_matched else 'lightgray',
                             icon='flash', prefix='fa')
        ).add_to(cluster)

        # Also add circle marker
        folium.CircleMarker(
            location=[lat, lon],
            radius=8 if is_matched else 6,
            color='black',
            weight=2,
            fill=True,
            fill_color=color,
            fill_opacity=0.9,
            tooltip=f"Network TRF {row['id']} ({'Matched' if is_matched else 'Unmatched'})"
        ).add_to(m)

    # Add match lines if there are matches
    match_lines = folium.FeatureGroup(name="Transformer Match Lines")

    # Add all the feature groups to the map
    dlr_matched_cluster.add_to(m)
    dlr_unmatched_cluster.add_to(m)
    net_matched_cluster.add_to(m)
    net_unmatched_cluster.add_to(m)
    match_lines.add_to(m)

    # Add layer control
    folium.LayerControl().add_to(m)

    # Add legend as HTML
    legend_html = """
    <div style="position: fixed; 
                bottom: 50px; 
                left: 50px; 
                background-color: white; 
                padding: 10px; 
                border: 1px solid grey; 
                border-radius: 5px; 
                z-index: 1000;">
      <h4>Transformer Legend</h4>
      <div style="display: flex; align-items: center; margin-bottom: 5px;">
        <div style="width: 20px; height: 20px; border-radius: 50%; background-color: blue; margin-right: 10px;"></div>
        <span>DLR Matched ({0})</span>
      </div>
      <div style="display: flex; align-items: center; margin-bottom: 5px;">
        <div style="width: 20px; height: 20px; border-radius: 50%; background-color: darkblue; margin-right: 10px;"></div>
        <span>DLR Unmatched ({1})</span>
      </div>
      <div style="display: flex; align-items: center; margin-bottom: 5px;">
        <div style="width: 20px; height: 20px; border-radius: 50%; background-color: red; margin-right: 10px;"></div>
        <span>Network Matched ({2})</span>
      </div>
      <div style="display: flex; align-items: center;">
        <div style="width: 20px; height: 20px; border-radius: 50%; background-color: gray; margin-right: 10px;"></div>
        <span>Network Unmatched ({3})</span>
      </div>
    </div>
    """.format(dlr_matched_count, dlr_unmatched_count, net_matched_count, net_unmatched_count)

    m.get_root().html.add_child(folium.Element(legend_html))

    # Add title
    title_html = """
    <div style="position: fixed; top: 10px; left: 50%; transform: translateX(-50%); 
                background-color: white; padding: 10px; border: 1px solid grey; 
                border-radius: 5px; z-index: 1000;">
      <h3 style="margin: 0;">Germany Transformer Visualization</h3>
      <p style="margin: 5px 0 0 0; text-align: center;">
        DLR: {0}/{1} matched ({2:.1f}%) | Network: {3}/{4} matched ({5:.1f}%)
      </p>
    </div>
    """.format(
        dlr_matched_count, len(dlr_gdf),
        100 * dlr_matched_count / len(dlr_gdf) if len(dlr_gdf) > 0 else 0,
        net_matched_count, len(net_gdf),
        100 * net_matched_count / len(net_gdf) if len(net_gdf) > 0 else 0
    )

    m.get_root().html.add_child(folium.Element(title_html))

    # Add debug panel
    debug_html = """
    <div style="position: fixed; 
                bottom: 10px; 
                right: 10px; 
                background-color: white; 
                padding: 10px; 
                border: 1px solid grey; 
                border-radius: 5px; 
                z-index: 1000;
                font-family: monospace;
                font-size: 12px;">
        <h4>Debug Info</h4>
        <p>Map center: {0:.6f}, {1:.6f}</p>
        <p>DLR trf total: {2}</p>
        <p>Net trf total: {3}</p>
        <p>Buffer: {4}m</p>
    </div>
    """.format(
        center_lat, center_lon,
        len(dlr_gdf),
        len(net_gdf),
        args.buffer
    )
    m.get_root().html.add_child(folium.Element(debug_html))

    # Save map
    m.save(output_file)
    logger.info(f"Map saved to {output_file}")

    return m


def main():
    """Main function"""
    global args
    args = parse_args()

    # List existing files and directories to help debug
    logger.info(f"Project root: {PROJECT_ROOT}")
    logger.info(f"Data directory: {DATA_DIR}")

    if DATA_DIR.exists():
        logger.info(f"Files in {DATA_DIR}:")
        for file in DATA_DIR.iterdir():
            logger.info(f"  {file.name}")
    else:
        logger.warning(f"Data directory {DATA_DIR} does not exist")

    # Load Germany boundary
    germany_boundary = load_germany_boundary(args.germany_geojson)

    # Check for the exact files we need
    dlr_file = Path(args.dlr_trf)
    net_file = Path(args.net_trf)

    logger.info(f"Looking for DLR file at: {dlr_file}")
    logger.info(f"Looking for Network file at: {net_file}")

    # Try to find the files in various locations
    possible_locations = [
        dlr_file,
        PROJECT_ROOT / "data" / "input" / dlr_file.name,
        Path.cwd() / ".." / "data" / "input" / dlr_file.name,
    ]

    for location in possible_locations:
        if location.exists():
            args.dlr_trf = str(location)
            logger.info(f"Found DLR file at: {location}")
            break

    possible_locations = [
        net_file,
        PROJECT_ROOT / "data" / "input" / net_file.name,
        Path.cwd() / ".." / "data" / "input" / net_file.name,
    ]

    for location in possible_locations:
        if location.exists():
            args.net_trf = str(location)
            logger.info(f"Found Network file at: {location}")
            break

    # Load transformers with custom loaders for each file format
    dlr_trf_all = load_dlr_transformers(args.dlr_trf)
    net_trf_all = load_network_transformers(args.net_trf)

    # Check if we have data
    if dlr_trf_all.empty:
        logger.error("No DLR transformers loaded. Check your input file.")
        return 1

    if net_trf_all.empty:
        logger.error("No Network transformers loaded. Check your input file.")
        return 1

    # Clip to Germany boundary
    dlr_trf = clip_to_germany(dlr_trf_all, germany_boundary)
    net_trf = clip_to_germany(net_trf_all, germany_boundary)

    # Output transformer coordinates for debugging
    logger.info("Sample DLR transformer coordinates (within Germany):")
    for i, (_, row) in enumerate(dlr_trf.iterrows()):
        if i < 5:  # Just show the first 5
            logger.info(f"  {row['id']}: {row.geometry.y:.6f}, {row.geometry.x:.6f}")

    logger.info("Sample Network transformer coordinates (within Germany):")
    for i, (_, row) in enumerate(net_trf.iterrows()):
        if i < 5:  # Just show the first 5
            logger.info(f"  {row['id']}: {row.geometry.y:.6f}, {row.geometry.x:.6f}")

    # Match transformers
    match_df, matched_dlr_ids, matched_net_ids = match_transformers(
        dlr_trf, net_trf, buffer_m=args.buffer)

    # Display some match information
    if not match_df.empty:
        logger.info("Sample matches:")
        for _, row in match_df.head(5).iterrows():
            logger.info(f"  DLR {row['dlr_id']} ↔ Network {row['network_id']} (Distance: {row['dist_m']:.2f}m)")

    # Create map
    create_transformer_map(
        dlr_trf, net_trf, matched_dlr_ids, matched_net_ids,
        germany_boundary, args.output)

    logger.info("Done!")
    return 0


if __name__ == "__main__":
    sys.exit(main())