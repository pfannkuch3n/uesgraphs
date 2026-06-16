"""
This module contains unit tests for uesgraphs analyze module
"""

import pytest
import os
from pathlib import Path
from datetime import datetime

import pandas as pd
from shapely.geometry import Point

import uesgraphs as ug
from uesgraphs.analyze.data_handling.data_handling import check_input_file
import uesgraphs.analyze as analyze
from uesgraphs.analyze import return_temp_reduction_potential

_K0 = 273.15


class TestAnalyzeDataHandling:
    """Test class for analyze data handling functionality"""
    
    def test_mat_file_conversion(self):
        """
        Test that check_input_file correctly handles .mat file conversion to .gzip
        """
        # Use the persistent .mat file in data directory
        mat_file_path = Path(__file__).parent.parent / "uesgraphs" / "data" / "Pinola_low_temp_network_inputs.mat"
        
        # Verify the test file exists
        assert mat_file_path.exists(), f"Test .mat file not found at {mat_file_path}"
        
        # Call check_input_file
        processed_file = check_input_file(file_path=str(mat_file_path))
        
        # Verify return value
        assert isinstance(processed_file, str), "Should return a string path"
        assert len(processed_file) > 0, "Should return non-empty path"
        assert os.path.exists(processed_file), f"Processed file should exist: {processed_file}"
        
        # Should return .gzip file (either existing or newly created)
        assert processed_file.endswith('.gzip'), "Should return .gzip file extension"
        
        # Verify it's the expected .gzip path
        expected_gzip_path = str(mat_file_path).replace('.mat', '.gzip')
        assert processed_file == expected_gzip_path, "Should return corresponding .gzip file"
    
    def test_assign_data_pipeline_integration(self):
        """
        Test that assign_data_pipeline works with real Pinola network data
        """
        # Get test data paths
        test_data_dir = Path(__file__).parent / "test_analyze_data"
        mat_file_path = Path(__file__).parent.parent / "uesgraphs" / "data" / "Pinola_low_temp_network_inputs.mat"
        nodes_json_path = test_data_dir / "nodes.json"
        sysm_json_path = test_data_dir / "pinola_sysm.json"
        
        # Verify all required files exist
        assert mat_file_path.exists(), f"Simulation data file not found: {mat_file_path}"
        assert nodes_json_path.exists(), f"Network topology file not found: {nodes_json_path}"
        assert sysm_json_path.exists(), f"System model file not found: {sysm_json_path}"
        
        # Create and configure graph
        graph = ug.UESGraph()
        graph.from_json(path=str(nodes_json_path), network_type="heating")
        graph.graph["name"] = "pinola_test"
        graph.graph["supply_type"] = "supply"
        
        # Set up test parameters (small time window for faster testing)
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 2)  # Just one day for testing
        time_interval = "15min"
        
        # Run the data assignment pipeline
        result_graph = analyze.assign_data_pipeline(
            graph=graph,
            simulation_data_path=str(mat_file_path),
            time_interval=time_interval,
            start_date=start_date,
            end_date=end_date,
            aixlib_version="2.1.0",
            system_model_path=str(sysm_json_path)
        )
        
        # Verify the pipeline completed successfully
        assert result_graph is not None, "Pipeline should return a graph"
        assert len(result_graph.nodes) > 0, "Graph should have nodes"
        assert len(result_graph.edges) > 0, "Graph should have edges"
        
        # Verify that data was assigned to edges
        edges_with_data = 0
        for edge in result_graph.edges:
            if "m_flow" in result_graph.edges[edge]:
                edges_with_data += 1
                # Verify data is time series
                m_flow_data = result_graph.edges[edge]["m_flow"]
                assert hasattr(m_flow_data, '__len__'), "m_flow should be a time series"
                assert len(m_flow_data) > 0, "m_flow should contain data points"
        
        assert edges_with_data > 0, "At least some edges should have mass flow data"
        
        # Verify that data was assigned to nodes  
        nodes_with_data = 0
        for node in result_graph.nodes:
            if "pressure" in result_graph.nodes[node]:
                nodes_with_data += 1
                # Verify data is time series
                pressure_data = result_graph.nodes[node]["pressure"]
                assert hasattr(pressure_data, '__len__'), "pressure should be a time series"
                assert len(pressure_data) > 0, "pressure should contain data points"
        
        assert nodes_with_data > 0, "At least some nodes should have pressure data"


def _toy_network():
    """Supply + 1 junction + 2 substations (b1, b2) with known 2-step series.

    target = 55 degC; deficits (degC): b1 = [5, -5], b2 = [15, 15];
    connection mass flows (kg/s): b1 = [2, 2], b2 = [2, 6].
    """
    graph = ug.UESGraph()
    supply = graph.add_building(name="supply", position=Point(0, 0),
                                is_supply_heating=True)
    junction = graph.add_network_node("heating", name="j", position=Point(1, 0))
    b1 = graph.add_building(name="b1", position=Point(2, 1))
    b2 = graph.add_building(name="b2", position=Point(2, -1))
    graph.add_edge(supply, junction)
    e1, e2 = (junction, b1), (junction, b2)
    graph.add_edge(*e1)
    graph.add_edge(*e2)

    idx = pd.date_range("2024-01-01", periods=2, freq="h")
    graph.nodes[b1]["temperature_return"] = pd.Series([60 + _K0, 50 + _K0], index=idx)
    graph.nodes[b2]["temperature_return"] = pd.Series([70 + _K0, 70 + _K0], index=idx)
    graph.edges[e1]["m_flow"] = pd.Series([2.0, 2.0], index=idx)
    graph.edges[e2]["m_flow"] = pd.Series([2.0, 6.0], index=idx)
    return graph


class TestReturnTempReductionPotential:
    """Eq. 13 (Oltmanns 2020) on a hand-computable toy network."""

    def test_clamped_potential_matches_hand_calc(self):
        df = return_temp_reduction_potential(_toy_network(), target_temp=55)

        # b2: mean(15*2/4, 15*6/8) = mean(7.5, 11.25) = 9.375
        # b1: mean(5*2/4, max(0,-5)*2/8) = mean(2.5, 0) = 1.25  (clamp kills step 2)
        assert df.loc["b2", "potential_K"] == pytest.approx(9.375)
        assert df.loc["b1", "potential_K"] == pytest.approx(1.25)
        assert df.attrs["network_total_K"] == pytest.approx(10.625)

        # Sorted descending -> highest-leverage substation first.
        assert list(df.index) == ["b2", "b1"]

        # Context columns and bookkeeping.
        assert df.loc["b1", "mean_return_C"] == pytest.approx(55.0)
        assert df.loc["b2", "mean_mflow_kgs"] == pytest.approx(4.0)
        assert df["mean_share"].sum() == pytest.approx(1.0)  # no skipped substations
        assert df.attrs["n_steps_used"] == 2
        assert df.attrs["n_substations"] == 2
        assert df.attrs["n_skipped"] == 0

    def test_no_clamp_keeps_signed_deficit(self):
        df = return_temp_reduction_potential(_toy_network(), target_temp=55,
                                             clamp=False)
        # b1 step 2 now contributes -5 * 2/8 = -1.25 -> mean(2.5, -1.25) = 0.625
        assert df.loc["b1", "potential_K"] == pytest.approx(0.625)
        assert df.loc["b2", "potential_K"] == pytest.approx(9.375)

    def test_target_in_kelvin_equivalent(self):
        df_c = return_temp_reduction_potential(_toy_network(), target_temp=55)
        df_k = return_temp_reduction_potential(_toy_network(),
                                               target_temp=55 + _K0,
                                               target_in_celsius=False)
        assert df_k.loc["b2", "potential_K"] == pytest.approx(
            df_c.loc["b2", "potential_K"])