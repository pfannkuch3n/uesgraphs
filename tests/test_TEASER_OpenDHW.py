# -*- coding: utf-8 -*-
"""
Tests for the TEASER and OpenDHW integration.

This module tests if TEASER and OpenDHW are able to generate the expected demand profiles 
with the provided GeoJSON, so that the installation of TEASER and OpenDHW is verified.
"""

import os
import pytest
import tempfile
from uesgraphs.DHW_estimation.utilities import generate_DHW_profiles_from_geojson
from uesgraphs.teaser_integration.utilities import run_sim_teaser
# Close all loggers to release file handles
import logging

# Skip test if dymola is not available
try:
    import dymola
    HAS_DYMOLA = True
except ImportError:
    HAS_DYMOLA = False

class TestE17IntegrationTEASER_OpenDHW:
    """Integration test using example e17."""

    @staticmethod
    def _setup_paths():
        """Helper method to set up common paths."""
        data_dir = os.path.join('uesgraphs', 'data')
        data_examples_dir = os.path.join(data_dir, 'examples')
        geojson_dir = os.path.join(data_examples_dir, 'e15_geojson')
        
        return {
            'network_geojson': os.path.join(geojson_dir, 'network.geojson'),
            'buildings_geojson': os.path.join(geojson_dir, 'buildings_teaser_OpenDHW_info.geojson'),
            'supply_geojson': os.path.join(geojson_dir, 'supply.geojson'),
            'ground_temps': os.path.join(data_examples_dir, 'ground_temps_hassel.csv'),
            'params_template': os.path.join(data_dir, 'uesgraphs_parameters_template_pp.xlsx'),
        }

    @pytest.mark.skipif(not HAS_DYMOLA, reason="dymola not installed")
    def test_e17_TEASER(self):
        """
        Test the TEASER integration for demand estimation.
        """
        paths = self._setup_paths()
        
        # Check if all required files exist
        required_files = [
            paths['network_geojson'], paths['buildings_geojson'], 
            paths['supply_geojson'], paths['ground_temps'], paths['params_template']
        ]
        for file_path in required_files:
            if not os.path.exists(file_path):
                pytest.skip(f"Required file not found: {file_path}")
        
        # Create temporary workspace - ignore cleanup errors on Windows
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as workspace:
            try:                
                # Step 1: Run TEASER simulations
                input_heating, input_cooling = run_sim_teaser(
                    buildings_info_path=paths['buildings_geojson'],
                    save_path=workspace,
                    sim_setup_path=paths['params_template'],
                    log_level=logging.INFO,
                    number_of_workers=1
                )
                
                # Step 2: Verify output
                demands_dir = os.path.join(workspace, 'demand_csv')
                assert os.path.exists(demands_dir), "Demand estimation directory was not created"
                
                # Check that demand estimation files were generated
                csv_files = []
                for root, dirs, files in os.walk(demands_dir):
                    csv_files.extend([f for f in files if f.endswith('.csv')])
                
                assert len(csv_files) > 0, f"Not all files were generated in {demands_dir}"
                print(f"✓ Test passed: Found {len(csv_files)} CSV files")
                
            except Exception as e:
                pytest.fail(f"TEASER failed: {e}")
            finally:
                logging.shutdown()

    def test_e17_OpenDHW(self):
        """
        Test the OpenDHW integration for DHW demand estimation.
        """
        paths = self._setup_paths()
        
        # Check if all required files exist
        required_files = [
            paths['network_geojson'], paths['buildings_geojson'], 
            paths['supply_geojson'], paths['ground_temps'], paths['params_template']
        ]
        for file_path in required_files:
            if not os.path.exists(file_path):
                pytest.skip(f"Required file not found: {file_path}")
        
        # Create temporary workspace - ignore cleanup errors on Windows
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as workspace:
            try:                
                # Step 1: Run OpenDHW for demand generations
                input_dhw = generate_DHW_profiles_from_geojson(
                    buildings_info_path=paths['buildings_geojson'],
                    save_path=workspace,
                    sim_setup_path=paths['params_template'],
                    log_level=logging.INFO
                )

                # Step 2: Verify output
                demands_dir = os.path.join(workspace, 'demand_csv')
                assert os.path.exists(demands_dir), "Demand estimation directory was not created"
                
                # Check that demand estimation files were generated
                csv_files = []
                for root, dirs, files in os.walk(demands_dir):
                    csv_files.extend([f for f in files if f.endswith('.csv')])
                
                assert len(csv_files) > 0, f"No CSV files were generated in {demands_dir}"
                print(f"✓ Test passed: Found {len(csv_files)} CSV files")
                
            except Exception as e:
                pytest.fail(f"OpenDHW failed: {e}")
            finally:
                # Close all loggers to release file handles
                logging.shutdown()


