# Updated generate_tennet_geojson.py
import os
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def generate_tennet_geojson(input_file, output_file):
    """
    Generate a GeoJSON file from the TenneT CSV data,
    properly handling the specific format of the TenneT CSV file.
    """
    logger.info(f"Reading TenneT data from {input_file}")

    try:
        # Read the raw CSV file, skipping the header row and using the second row as header
        df = pd.read_csv(input_file, skiprows=[0])
        logger.info(f"Loaded {len(df)} rows from TenneT CSV")

        # Display available columns for debugging
        logger.info(f"Available columns: {df.columns.tolist()}")

        # Create features array for GeoJSON
        features = []

        # Process each row directly since each row represents a complete line
        for idx, row in df.iterrows():
            try:
                # Extract coordinates
                lon1 = float(row['Longitude'])
                lat1 = float(row['Latitude'])
                lon2 = float(row['Longitude.1'])
                lat2 = float(row['Latitude.1'])

                # Extract properties
                line_id = row['NE_name']
                voltage = float(row.get('Voltage_level(kV)', 380))
                resistance = float(row.get('Resistance_R(Ω)', 0))
                reactance = float(row.get('Reactance_X(Ω)', 0))
                susceptance = float(row.get('Susceptance_B(μS)', 0)) * 1e-6  # Convert μS to S
                length = float(row.get('Length_(km)', 0))

                # Create properties dictionary
                properties = {
                    'id': line_id,
                    'NE_name': line_id,
                    'EIC_Code': str(row.get('EIC_Code', '')),
                    'v_nom': voltage,
                    'r': resistance,
                    'x': reactance,
                    'b': susceptance,
                    'length': length,
                    'TSO': 'TenneT',
                    'substation1': str(row.get('Full_name', '')),
                    'substation2': str(row.get('Full_name.1', ''))
                }

                # Create GeoJSON feature
                feature = {
                    'type': 'Feature',
                    'properties': properties,
                    'geometry': {
                        'type': 'LineString',
                        'coordinates': [(lon1, lat1), (lon2, lat2)]
                    }
                }

                features.append(feature)

                if idx < 5:  # Log first few for verification
                    logger.info(f"Created line {idx}: {line_id} from ({lon1}, {lat1}) to ({lon2}, {lat2})")

            except Exception as e:
                logger.warning(f"Error processing row {idx}: {str(e)}")

        # Create the GeoJSON object
        geojson = {
            'type': 'FeatureCollection',
            'features': features
        }

        # Write to file
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(geojson, f)

        logger.info(f"Created GeoJSON with {len(features)} TenneT lines at {output_file}")

        # Create a GeoDataFrame from the features if we have any
        if features:
            gdf = gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")
            gpkg_file = output_file.replace('.geojson', '.gpkg')
            gdf.to_file(gpkg_file, driver="GPKG")
            logger.info(f"Created GeoPackage at {gpkg_file}")
        else:
            logger.warning("No features created, skipping GeoPackage creation")

        return output_file

    except Exception as e:
        logger.error(f"Error generating TenneT GeoJSON: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None


if __name__ == "__main__":
    # Example usage
    input_file = "data/input/tennet-lines.csv"
    output_file = "data/processed/tennet_lines.geojson"
    generate_tennet_geojson(input_file, output_file)