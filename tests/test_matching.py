
import sys
from pathlib import Path
import tempfile

# Add the project root to the Python path
project_root = str(Path(__file__).parent.parent.absolute())
sys.path.insert(0, project_root)

# Third-party imports
import pytest
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.matching.dlr_matching import match_lines_detailed
from src.matching.pypsa_matching import identify_visual_overlaps, match_pypsa_eur_lines
from src.matching.utils import direction_similarity, calculate_line_direction


# Fixture for creating test lines
@pytest.fixture
def sample_lines():
    """Create sample line geometries for testing"""
    # Create two parallel lines
    line1 = LineString([(0, 0), (0, 1)])
    line2 = LineString([(0.01, 0), (0.01, 1)])

    # Create two perpendicular lines
    line3 = LineString([(0, 0), (1, 0)])
    line4 = LineString([(0, 0.01), (0, 1.01)])

    # Create two overlapping lines (one is partly along the other)
    line5 = LineString([(0, 0), (0, 1)])
    line6 = LineString([(0, 0.5), (0, 1.5)])

    # Create two nearly identical lines with slight offset
    line7 = LineString([(0, 0), (0.5, 0.5), (1, 1)])
    line8 = LineString([(0.001, 0.001), (0.501, 0.501), (1.001, 1.001)])

    return {
        'parallel': (line1, line2),
        'perpendicular': (line3, line4),
        'overlapping': (line5, line6),
        'nearly_identical': (line7, line8)
    }


@pytest.fixture
def sample_geodataframes(sample_lines):
    """Create sample GeoDataFrames for testing"""
    # Source lines GeoDataFrame
    source_data = [
        {
            'id': '1',
            'r': 10.0,
            'x': 100.0,
            'b': 0.001,
            'v_nom': 380,
            'length': 111.0,  # ~1 degree in km
            'geometry': sample_lines['parallel'][0]
        },
        {
            'id': '2',
            'r': 5.0,
            'x': 50.0,
            'b': 0.0005,
            'v_nom': 220,
            'length': 111.0,
            'geometry': sample_lines['perpendicular'][0]
        },
        {
            'id': '3',
            'r': 15.0,
            'x': 150.0,
            'b': 0.0015,
            'v_nom': 380,
            'length': 111.0,
            'geometry': sample_lines['overlapping'][0]
        },
        {
            'id': '4',
            'r': 20.0,
            'x': 200.0,
            'b': 0.002,
            'v_nom': 380,
            'length': 155.6,  # ~sqrt(2) degrees in km
            'geometry': sample_lines['nearly_identical'][0]
        }
    ]

    # Network lines GeoDataFrame
    network_data = [
        {
            'id': 'N1',
            'r': 9.0,
            'x': 90.0,
            'b': 0.0009,
            'v_nom': 380,
            'length': 111.0,
            'geometry': sample_lines['parallel'][1]
        },
        {
            'id': 'N2',
            'r': 4.5,
            'x': 45.0,
            'b': 0.00045,
            'v_nom': 220,
            'length': 111.0,
            'geometry': sample_lines['perpendicular'][1]
        },
        {
            'id': 'N3',
            'r': 12.0,
            'x': 120.0,
            'b': 0.0012,
            'v_nom': 380,
            'length': 111.0,
            'geometry': sample_lines['overlapping'][1]
        },
        {
            'id': 'N4',
            'r': 18.0,
            'x': 180.0,
            'b': 0.0018,
            'v_nom': 380,
            'length': 155.6,
            'geometry': sample_lines['nearly_identical'][1]
        }
    ]

    source_gdf = gpd.GeoDataFrame(source_data, crs='EPSG:4326')
    network_gdf = gpd.GeoDataFrame(network_data, crs='EPSG:4326')

    # Calculate per_km values
    for gdf in [source_gdf, network_gdf]:
        gdf['r_per_km'] = gdf.apply(lambda row: row['r'] / row['length'] if row['length'] > 0 else 0, axis=1)
        gdf['x_per_km'] = gdf.apply(lambda row: row['x'] / row['length'] if row['length'] > 0 else 0, axis=1)
        gdf['b_per_km'] = gdf.apply(lambda row: row['b'] / row['length'] if row['length'] > 0 else 0, axis=1)

    return {'source': source_gdf, 'network': network_gdf}


def test_direction_similarity(sample_lines):
    """Test the direction similarity function"""
    # Test parallel lines (should be close to 1)
    parallel_sim = direction_similarity(sample_lines['parallel'][0], sample_lines['parallel'][1])
    assert parallel_sim > 0.99

    # Test perpendicular lines (should be close to 0)
    perp_sim = direction_similarity(sample_lines['perpendicular'][0], sample_lines['perpendicular'][1])
    assert perp_sim < 0.1

    # Test nearly identical lines (should be close to 1)
    identical_sim = direction_similarity(sample_lines['nearly_identical'][0], sample_lines['nearly_identical'][1])
    assert identical_sim > 0.99

    # Test overlapping lines (should be 1 since they're colinear)
    overlap_sim = direction_similarity(sample_lines['overlapping'][0], sample_lines['overlapping'][1])
    assert overlap_sim > 0.99


def test_calculate_line_direction(sample_lines):
    """Test the line direction calculation"""
    # Test a vertical line (should be approximately (0,1))
    vertical_dx, vertical_dy = calculate_line_direction(sample_lines['parallel'][0])
    assert abs(vertical_dx) < 0.001
    assert abs(vertical_dy - 1) < 0.001

    # Test a horizontal line (should be approximately (1,0))
    horizontal_dx, horizontal_dy = calculate_line_direction(sample_lines['perpendicular'][0])
    assert abs(horizontal_dx - 1) < 0.001
    assert abs(horizontal_dy) < 0.001

    # Test a diagonal line (should be approximately (√2/2, √2/2))
    diagonal_dx, diagonal_dy = calculate_line_direction(sample_lines['nearly_identical'][0])
    assert abs(diagonal_dx - diagonal_dy) < 0.001
    assert 0.7 < diagonal_dx < 0.8  # Approximately √2/2 = ~0.707


def test_match_lines_detailed(sample_geodataframes):
    """Test the main line matching algorithm"""
    source_gdf = sample_geodataframes['source']
    network_gdf = sample_geodataframes['network']

    # Run the matching algorithm
    matches = match_lines_detailed(
        source_gdf,
        network_gdf,
        buffer_distance=0.02,
        snap_distance=0.01,
        direction_threshold=0.9,
        enforce_voltage_matching=False,
        dataset_name="Test"
    )

    # Check that we got matches
    assert not matches.empty

    # Should find at least 2 matches (the parallel and nearly identical lines)
    assert len(matches) >= 2

    # Check that the IDs match correctly - parallel lines
    parallel_matches = matches[matches['dlr_id'] == '1']
    assert not parallel_matches.empty
    assert 'N1' in parallel_matches['network_id'].values

    # Check nearly identical lines match
    identical_matches = matches[matches['dlr_id'] == '4']
    assert not identical_matches.empty
    assert 'N4' in identical_matches['network_id'].values

    # Test parameters are copied correctly
    test_match = parallel_matches.iloc[0]
    assert abs(test_match['source_r'] - 10.0) < 0.001
    assert abs(test_match['network_r'] - 9.0) < 0.001

    # Test allocation is working
    assert 'allocated_r' in test_match
    assert 'allocated_x' in test_match
    assert 'allocated_b' in test_match


def test_match_lines_detailed_with_voltage_matching(sample_geodataframes):
    """Test the line matching algorithm with voltage matching enforced"""
    source_gdf = sample_geodataframes['source']
    network_gdf = sample_geodataframes['network']

    # Run with voltage matching enforced
    matches = match_lines_detailed(
        source_gdf,
        network_gdf,
        buffer_distance=0.02,
        snap_distance=0.01,
        direction_threshold=0.7,
        enforce_voltage_matching=True,
        dataset_name="Test"
    )

    # Should still find matches for voltage-compatible lines
    assert not matches.empty

    # The 380kV lines should match
    assert '1' in matches['dlr_id'].values
    assert '4' in matches['dlr_id'].values

    # The 220kV line should also match if there's a 220kV network line
    if '2' in matches['dlr_id'].values:
        voltage_match = matches[matches['dlr_id'] == '2'].iloc[0]
        assert voltage_match['source_voltage'] == voltage_match['network_voltage']


def test_identify_visual_overlaps(sample_geodataframes):
    """Test the PyPSA-EUR visual overlap detection function"""
    # For this test, pretend the source GeoDataFrame is PyPSA-EUR data
    pypsa_lines = sample_geodataframes['source']
    network_lines = sample_geodataframes['network']

    # Run the visual overlap detection
    overlaps = identify_visual_overlaps(
        pypsa_lines,
        network_lines,
        max_distance=0.015,
        max_hausdorff=0.02
    )

    # Check that we got overlaps
    assert len(overlaps) > 0

    # The nearly identical lines should be detected
    nearly_identical_found = False
    for overlap in overlaps:
        if overlap['dlr_id'] == '4' and overlap['network_id'] == 'N4':
            nearly_identical_found = True
            break

    assert nearly_identical_found, "Nearly identical lines should be detected as visual overlaps"

    # Check that the overlap data is structured correctly
    sample_overlap = overlaps[0]
    assert 'dlr_id' in sample_overlap
    assert 'network_id' in sample_overlap
    assert 'distance' in sample_overlap
    assert 'hausdorff_distance' in sample_overlap
    assert 'overlap_percentage' in sample_overlap
    assert 'match_type' in sample_overlap
    assert sample_overlap['match_type'] == 'visual_overlap'


def test_match_pypsa_eur_lines(sample_geodataframes):
    """Test the PyPSA-EUR multi-stage matching function"""
    # For this test, pretend the source GeoDataFrame is PyPSA-EUR data
    pypsa_lines = sample_geodataframes['source']
    network_lines = sample_geodataframes['network']

    # Configure the matcher
    config = {
        'max_distance': 0.01,
        'max_hausdorff': 0.01,
        'relaxed_max_distance': 0.02,
        'relaxed_max_hausdorff': 0.03
    }

    # Run the multi-stage matching
    matches = match_pypsa_eur_lines(
        pypsa_lines,
        network_lines,
        config=config
    )

    # Check that we got matches
    assert not matches.empty

    # Check that allocated parameters are calculated
    assert 'allocated_r' in matches.columns
    assert 'allocated_x' in matches.columns
    assert 'allocated_b' in matches.columns

    # Check that change percentages are calculated
    assert 'r_change_pct' in matches.columns
    assert 'x_change_pct' in matches.columns
    assert 'b_change_pct' in matches.columns

    # Verify that the nearly identical lines match
    assert '4' in matches['dlr_id'].values
    nearly_identical_match = matches[matches['dlr_id'] == '4'].iloc[0]
    assert nearly_identical_match['network_id'] == 'N4'

    # The TSO field should be set to 'PyPSA-EUR'
    assert nearly_identical_match['TSO'] == 'PyPSA-EUR'


def test_match_lines_detailed_empty_input():
    """Test the matching algorithm with empty inputs"""
    # Create empty GeoDataFrames
    empty_source = gpd.GeoDataFrame(geometry=[], crs='EPSG:4326')
    empty_network = gpd.GeoDataFrame(geometry=[], crs='EPSG:4326')

    # Run with empty source
    matches1 = match_lines_detailed(
        empty_source,
        gpd.GeoDataFrame({'id': ['N1'], 'geometry': [LineString([(0, 0), (1, 1)])]}, crs='EPSG:4326'),
        buffer_distance=0.02,
        snap_distance=0.01,
        direction_threshold=0.7,
        enforce_voltage_matching=False
    )

    # Run with empty network
    matches2 = match_lines_detailed(
        gpd.GeoDataFrame({'id': ['1'], 'geometry': [LineString([(0, 0), (1, 1)])]}, crs='EPSG:4326'),
        empty_network,
        buffer_distance=0.02,
        snap_distance=0.01,
        direction_threshold=0.7,
        enforce_voltage_matching=False
    )

    # Run with both empty
    matches3 = match_lines_detailed(
        empty_source,
        empty_network,
        buffer_distance=0.02,
        snap_distance=0.01,
        direction_threshold=0.7,
        enforce_voltage_matching=False
    )

    # All should return empty DataFrames without errors
    assert isinstance(matches1, pd.DataFrame)
    assert matches1.empty

    assert isinstance(matches2, pd.DataFrame)
    assert matches2.empty

    assert isinstance(matches3, pd.DataFrame)
    assert matches3.empty