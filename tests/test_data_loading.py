import os
import pytest
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString
import sys
from pathlib import Path
import tempfile

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loaders import load_data, load_pypsa_eur_data, load_tso_data, load_germany_boundary


# Fixture for creating sample test files
@pytest.fixture
def sample_data_files():
    """Create temporary sample data files for testing loaders"""
    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create DLR test data
        dlr_data = """id,r,x,b,v_nom,length,geometry
1,10.0,100.0,0.001,380,111.0,LINESTRING(0 0, 0 1)
2,5.0,50.0,0.0005,220,111.0,LINESTRING(1 0, 2 0)
"""
        dlr_file = os.path.join(temp_dir, "dlr-test.csv")
        with open(dlr_file, 'w') as f:
            f.write(dlr_data)

        # Create network test data
        network_data = """id,r,x,b,v_nom,length,geometry
N1,9.0,90.0,0.0009,380,111.0,LINESTRING(0.01 0, 0.01 1)
N2,4.5,45.0,0.00045,220,111.0,LINESTRING(1 0.01, 2 0.01)
"""
        network_file = os.path.join(temp_dir, "network-test.csv")
        with open(network_file, 'w') as f:
            f.write(network_data)

        # Create PyPSA-EUR test data
        pypsa_data = """line_id,voltage,r,x,b,length,geometry
P1,380,11.0,110.0,0.0011,111.0,'LINESTRING(0 0, 0 1)'
P2,220,5.5,55.0,0.00055,111.0,'LINESTRING(1 0, 2 0)'
"""
        pypsa_file = os.path.join(temp_dir, "pypsa-eur-test.csv")
        with open(pypsa_file, 'w') as f:
            f.write(pypsa_data)

        # Create 50Hertz test data
        fifty_hertz_data = """NE_name,Longitude_Substation_1,Latitude_Substation_1,Longitude_Substation_2,Latitude_Substation_2,Voltage_level(kV),Length_(km),Resistance_R(Ω),Reactance_X(Ω),Susceptance_B(μS)
F1,0,0,0,1,380,111.0,12.0,120.0,1100
F2,1,0,2,0,220,111.0,6.0,60.0,600
"""
        fifty_hertz_file = os.path.join(temp_dir, "fifty-hertz-test.csv")
        with open(fifty_hertz_file, 'w') as f:
            f.write(fifty_hertz_data)

        # Create TenneT test data
        tennet_data = """NE_name,Longitude,Latitude,Longitude.1,Latitude.1,Voltage_level(kV),Length_(km),Resistance_R(Ω),Reactance_X(Ω),Susceptance_B(μS)
T1,0,0,0,1,380,111.0,12.5,125.0,1250
T2,1,0,2,0,220,111.0,6.25,62.5,625
"""
        tennet_file = os.path.join(temp_dir, "tennet-test.csv")
        with open(tennet_file, 'w') as f:
            f.write(tennet_data)

        # Create a simple Germany boundary GeoJSON
        germany_data = """
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "properties": {},
      "geometry": {
        "type": "Polygon",
        "coordinates": [
          [
            [5.0, 47.0],
            [15.0, 47.0],
            [15.0, 55.0],
            [5.0, 55.0],
            [5.0, 47.0]
          ]
        ]
      }
    }
  ]
}
"""
        germany_file = os.path.join(temp_dir, "germany-test.geojson")
        with open(germany_file, 'w') as f:
            f.write(germany_data)

        # Return the file paths
        yield {
            'dlr': dlr_file,
            'network': network_file,
            'pypsa': pypsa_file,
            'fifty_hertz': fifty_hertz_file,
            'tennet': tennet_file,
            'germany': germany_file
        }


def test_load_data(sample_data_files):
    """Test loading DLR and network data"""
    dlr_gdf, network_gdf = load_data(sample_data_files['dlr'], sample_data_files['network'])

    # Check DLR data
    assert isinstance(dlr_gdf, gpd.GeoDataFrame)
    assert len(dlr_gdf) == 2
    assert 'geometry' in dlr_gdf.columns
    assert 'r' in dlr_gdf.columns
    assert 'x' in dlr_gdf.columns
    assert 'b' in dlr_gdf.columns
    assert 'v_nom' in dlr_gdf.columns
    assert 'length' in dlr_gdf.columns
    assert 'r_per_km' in dlr_gdf.columns
    assert 'x_per_km' in dlr_gdf.columns
    assert 'b_per_km' in dlr_gdf.columns

    # Check network data
    assert isinstance(network_gdf, gpd.GeoDataFrame)
    assert len(network_gdf) == 2
    assert 'geometry' in network_gdf.columns
    assert 'r' in network_gdf.columns
    assert 'x' in network_gdf.columns
    assert 'b' in network_gdf.columns
    assert 'v_nom' in network_gdf.columns
    assert 'length' in network_gdf.columns
    assert 'r_per_km' in network_gdf.columns
    assert 'x_per_km' in network_gdf.columns
    assert 'b_per_km' in network_gdf.columns

    # Check that geometries are valid
    assert all(g.is_valid for g in dlr_gdf.geometry)
    assert all(g.is_valid for g in network_gdf.geometry)

    # Check CRS is set
    assert dlr_gdf.crs == "EPSG:4326"
    assert network_gdf.crs == "EPSG:4326"


def test_load_pypsa_eur_data(sample_data_files):
    """Test loading PyPSA-EUR data"""
    pypsa_gdf = load_pypsa_eur_data(sample_data_files['pypsa'])

    # Check PyPSA-EUR data
    assert isinstance(pypsa_gdf, gpd.GeoDataFrame)
    assert len(pypsa_gdf) == 2
    assert 'geometry' in pypsa_gdf.columns
    assert 'r' in pypsa_gdf.columns
    assert 'x' in pypsa_gdf.columns
    assert 'b' in pypsa_gdf.columns
    assert 'v_nom' in pypsa_gdf.columns
    assert 'length' in pypsa_gdf.columns
    assert 'r_per_km' in pypsa_gdf.columns
    assert 'x_per_km' in pypsa_gdf.columns
    assert 'b_per_km' in pypsa_gdf.columns

    # Check ID mapping
    assert 'id' in pypsa_gdf.columns
    assert 'P1' in pypsa_gdf['id'].values
    assert 'P2' in pypsa_gdf['id'].values

    # Check voltage mapping
    assert all(pypsa_gdf['v_nom'] == pypsa_gdf['voltage'])

    # Check that geometries are valid
    assert all(g.is_valid for g in pypsa_gdf.geometry)

    # Check CRS is set
    assert pypsa_gdf.crs == "EPSG:4326"


def test_load_tso_data(sample_data_files):
    """Test loading TSO data (50Hertz and TenneT)"""
    # Load 50Hertz data
    fifty_hertz_gdf = load_tso_data(sample_data_files['fifty_hertz'], 'fifty_hertz')

    # Load TenneT data
    tennet_gdf = load_tso_data(sample_data_files['tennet'], 'tennet')

    # Check 50Hertz data
    assert isinstance(fifty_hertz_gdf, gpd.GeoDataFrame)
    assert len(fifty_hertz_gdf) == 2
    assert 'geometry' in fifty_hertz_gdf.columns
    assert 'r' in fifty_hertz_gdf.columns
    assert 'x' in fifty_hertz_gdf.columns
    assert 'b' in fifty_hertz_gdf.columns
    assert 'v_nom' in fifty_hertz_gdf.columns
    assert 'length' in fifty_hertz_gdf.columns

    # Check TenneT data
    assert isinstance(tennet_gdf, gpd.GeoDataFrame)
    assert len(tennet_gdf) == 2
    assert 'geometry' in tennet_gdf.columns
    assert 'r' in tennet_gdf.columns
    assert 'x' in tennet_gdf.columns
    assert 'b' in tennet_gdf.columns
    assert 'v_nom' in tennet_gdf.columns
    assert 'length' in tennet_gdf.columns

    # Check that geometries are created correctly (should be LineStrings)
    assert all(isinstance(g, LineString) for g in fifty_hertz_gdf.geometry)
    assert all(isinstance(g, LineString) for g in tennet_gdf.geometry)

    # Check ID mapping
    assert 'id' in fifty_hertz_gdf.columns
    assert 'id' in tennet_gdf.columns
    assert 'F1' in fifty_hertz_gdf['id'].values
    assert 'T1' in tennet_gdf['id'].values

    # Check TSO field is set
    assert 'TSO' in fifty_hertz_gdf.columns
    assert 'TSO' in tennet_gdf.columns
    assert all(fifty_hertz_gdf['TSO'] == '50Hertz')
    assert all(tennet_gdf['TSO'] == 'TenneT')


def test_load_germany_boundary(sample_data_files):
    """Test loading Germany boundary"""
    germany_gdf = load_germany_boundary(sample_data_files['germany'])

    # Check Germany boundary data
    assert isinstance(germany_gdf, gpd.GeoDataFrame)
    assert len(germany_gdf) == 1
    assert 'geometry' in germany_gdf.columns

    # Check that geometry is valid
    assert all(g.is_valid for g in germany_gdf.geometry)

    # Check CRS is set
    assert germany_gdf.crs == "EPSG:4326"

    # Verify it's a Polygon or MultiPolygon
    geom_type = germany_gdf.geometry.iloc[0].geom_type
    assert geom_type in ['Polygon', 'MultiPolygon']


def test_load_data_missing_files():
    """Test loading data with missing files"""
    # Test with missing DLR file
    with pytest.raises(FileNotFoundError):
        dlr_gdf, network_gdf = load_data("nonexistent-file.csv", "also-nonexistent.csv")


def test_load_pypsa_eur_data_empty_file(sample_data_files):
    """Test loading PyPSA-EUR data from an empty file"""
    # Create an empty file
    empty_file = os.path.join(os.path.dirname(sample_data_files['pypsa']), "empty-pypsa.csv")
    with open(empty_file, 'w') as f:
        f.write("line_id,voltage,r,x,b,length,geometry\n")

    # Load the empty file
    pypsa_gdf = load_pypsa_eur_data(empty_file)

    # Should return an empty GeoDataFrame
    assert isinstance(pypsa_gdf, gpd.GeoDataFrame)
    assert len(pypsa_gdf) == 0


def test_load_tso_data_invalid_tso_type(sample_data_files):
    """Test loading TSO data with invalid TSO type"""
    # Try with invalid TSO type
    with pytest.raises(ValueError):
        tso_gdf = load_tso_data(sample_data_files['fifty_hertz'], 'invalid_tso')