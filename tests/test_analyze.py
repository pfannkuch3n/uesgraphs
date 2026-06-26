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
from uesgraphs.analyze.data_handling.data_handling import (
    check_input_file, mask_keep_suffixes, AIXLIB_MASKS,
    SUPPLY_MASKS, resolve_supply_mask, assign_supply_values,
    build_filter_list_loss, build_filter_list_supply,
)
import uesgraphs.analyze as analyze
from uesgraphs.analyze import return_temp_reduction_potential, pump_vs_loss
from uesgraphs.analyze.convert import convert_mat

_K0 = 273.15


def _parquet_scope(path):
    """Read the informational ``ues_scope`` label from a parquet cache."""
    import pyarrow.parquet as pq
    meta = pq.ParquetFile(str(path)).schema_arrow.metadata or {}
    return meta.get(b"ues_scope", b"").decode()


def _mask_template_tails(masks):
    """All template tails (after the last ``{...}``) in a nested mask dict."""
    tails = set()

    def _walk(d):
        for v in d.values():
            if isinstance(v, dict):
                _walk(v)
            elif isinstance(v, str):
                tails.add(v.rsplit("}", 1)[-1])

    _walk(masks)
    return tails


class TestAnalyzeDataHandling:
    """Test class for analyze data handling functionality"""
    
    def test_mat_file_conversion(self):
        """
        Test that check_input_file converts a .mat file to the .parquet cache.
        """
        # Use the persistent .mat file in data directory
        mat_file_path = Path(__file__).parent.parent / "uesgraphs" / "data" / "Pinola_low_temp_network_inputs.mat"

        # Verify the test file exists
        assert mat_file_path.exists(), f"Test .mat file not found at {mat_file_path}"

        # Force a reconvert so the assertions don't depend on a pre-existing cache.
        processed_file = check_input_file(file_path=str(mat_file_path), force_reconvert=True)

        # Verify return value
        assert isinstance(processed_file, str), "Should return a string path"
        assert len(processed_file) > 0, "Should return non-empty path"
        assert os.path.exists(processed_file), f"Processed file should exist: {processed_file}"

        # Should return the .parquet cache (the old .gzip name was a misnomer).
        assert processed_file.endswith('.parquet'), "Should return .parquet file extension"
        expected_parquet_path = str(mat_file_path).replace('.mat', '.parquet')
        assert processed_file == expected_parquet_path, "Should return corresponding .parquet file"

        # Default scope is the mask-derived fast subset.
        assert _parquet_scope(processed_file) == "fast", "Default cache scope should be 'fast'"

        # A second (non-forced) call must reuse the fresh cache, not reconvert.
        again = check_input_file(file_path=str(mat_file_path))
        assert again == expected_parquet_path, "Should reuse the existing .parquet cache"

    def test_mat_file_conversion_full_scope(self):
        """all_vars=True / scope='full' keeps every variable and labels the cache.

        fast and full caches share the same path (<base>.parquet), so measure each
        right after its own conversion (the later write overwrites the earlier).
        """
        import pyarrow.parquet as pq
        mat_file_path = Path(__file__).parent.parent / "uesgraphs" / "data" / "Pinola_low_temp_network_inputs.mat"
        assert mat_file_path.exists()

        full = convert_mat(mat_file_path, force=True, all_vars=True)
        assert _parquet_scope(full) == "full"
        n_full = len(pq.ParquetFile(str(full)).schema_arrow.names)

        fast = convert_mat(mat_file_path, force=True)
        assert fast == full  # same cache path (<base>.parquet)
        assert _parquet_scope(fast) == "fast"
        n_fast = len(pq.ParquetFile(str(fast)).schema_arrow.names)

        # The fast cache must be a strict subset (far fewer columns).
        assert n_fast < n_full, f"fast={n_fast} should be < full={n_full}"

    def test_stale_keep_cache_is_rebuilt(self):
        """A fast cache built with FEWER kept suffixes (an older mask set) is
        detected stale via its ues_keep fingerprint and rebuilt, so columns added
        to the masks later (e.g. heat loss) reappear without a manual force."""
        import pyarrow.parquet as pq
        from uesgraphs.analyze.data_handling.mat_handler import mat_to_parquet

        mat = Path(__file__).parent.parent / "uesgraphs" / "data" / "Pinola_low_temp_network_inputs.mat"
        assert mat.exists()
        base = str(mat)[:-4]
        parquet = base + ".parquet"

        # Build a deliberately narrow cache (mimics masks before heat loss existed).
        mat_to_parquet(save_as=base, fname=str(mat), with_unit=False,
                       keep_suffixes=(".port_a.m_flow", "Time"))
        narrow = set(pq.ParquetFile(parquet).schema.names)
        assert not any(c.endswith(".heatPort.Q_flow") for c in narrow)

        # A normal (non-forced) resolve with the CURRENT masks must rebuild it.
        out = check_input_file(file_path=str(mat))
        cols = set(pq.ParquetFile(out).schema.names)
        assert any(c.endswith(".heatPort.Q_flow") for c in cols), \
            "stale cache was not rebuilt with the current kept-column set"
    
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


class TestMaskKeepSuffixes:
    """The fast-track filter must be derived from AIXLIB_MASKS (single source)."""

    def test_covers_all_mask_tails(self):
        suffixes = set(mask_keep_suffixes())
        for version, masks in AIXLIB_MASKS.items():
            for tail in _mask_template_tails(masks):
                assert tail in suffixes, (
                    f"AixLib {version}: template tail {tail!r} not covered by "
                    f"mask_keep_suffixes() -> fast cache would drop it")
        assert "Time" in suffixes

    def test_keeps_relevant_drops_unrelated(self):
        keep = tuple(mask_keep_suffixes())
        # A standard pipe/demand variable matches and is kept ...
        assert "networkModel.pipe5.port_a.m_flow".endswith(keep)
        assert "networkModel.house3Q_flow_input".endswith(keep)
        # ... an out-of-pattern quantity (e.g. heat pump power) is dropped.
        assert not "networkModel.heatPump.compressor.P_el".endswith(keep)


class TestConvertCLI:
    """Pre-convert helper: produces the cache and skips when current."""

    def test_convert_mat_skip_when_current(self):
        mat = Path(__file__).parent.parent / "uesgraphs" / "data" / "Pinola_low_temp_network_inputs.mat"
        assert mat.exists()
        out = convert_mat(mat, force=True)
        assert out.exists() and out.suffix == ".parquet"
        mtime1 = out.stat().st_mtime
        # A current cache must be skipped (file untouched), not rewritten.
        out2 = convert_mat(mat)
        assert out2 == out
        assert out2.stat().st_mtime == mtime1, "current cache should not be rewritten"

    def test_find_mats(self):
        from uesgraphs.analyze.convert import find_mats
        data_dir = Path(__file__).parent.parent / "uesgraphs" / "data"
        mats = find_mats(data_dir)
        assert any(m.name == "Pinola_low_temp_network_inputs.mat" for m in mats)
        # A single .mat path resolves to just that file.
        assert find_mats(mats[0]) == [mats[0]]


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


# ---------------------------------------------------------------------------
# Pump power vs. heat loss
# ---------------------------------------------------------------------------

def _pump_loss_toy():
    """Supply S + 2 substations (b1, b2), 2 pipes, hand-computable 2-step series.

    Q_loss (W): e1=[5,10], e2=[15,20]  -> per-pipe sum [20,30]
    supply Q  (W): [100,200];  demand prescribed (W): b1=[60,90], b2=[20,30]
                   -> balance [100-80, 200-120] = [20, 80]
    pump_power_hydraulic (W): [10, 30]
    """
    graph = ug.UESGraph()
    supply = graph.add_building(name="S", position=Point(0, 0),
                                is_supply_heating=True)
    b1 = graph.add_building(name="b1", position=Point(2, 1))
    b2 = graph.add_building(name="b2", position=Point(2, -1))
    graph.add_edge(supply, b1)
    graph.add_edge(supply, b2)
    e1, e2 = (supply, b1), (supply, b2)

    idx = pd.date_range("2024-01-01", periods=2, freq="h")
    graph.edges[e1]["Q_loss"] = pd.Series([5.0, 10.0], index=idx)
    graph.edges[e2]["Q_loss"] = pd.Series([15.0, 20.0], index=idx)
    graph.nodes[supply]["heat_power_supply"] = pd.Series([100.0, 200.0], index=idx)
    graph.nodes[supply]["pump_power_hydraulic"] = pd.Series([10.0, 30.0], index=idx)
    graph.nodes[b1]["heat_power_prescribed"] = pd.Series([60.0, 90.0], index=idx)
    graph.nodes[b2]["heat_power_prescribed"] = pd.Series([20.0, 30.0], index=idx)
    return graph


class TestPumpVsLoss:
    """pump_vs_loss reduction on a hand-computable toy network."""

    def test_matches_hand_calc(self):
        res = pump_vs_loss(_pump_loss_toy(), timestep_hours=1.0)

        # per-pipe loss: [20,30] W -> 0.05 kWh, peak 0.03 kW
        assert res["loss_per_pipe_kWh"] == pytest.approx(0.05)
        assert res["loss_per_pipe_peak_kW"] == pytest.approx(0.03)
        # supply heat (context): [100,200] W -> 0.3 kWh, peak 0.2 kW
        assert res["supply_heat_kWh"] == pytest.approx(0.3)
        assert res["supply_heat_peak_kW"] == pytest.approx(0.2)
        # demand heat (context, prescribed): [80,120] W -> 0.2 kWh, peak 0.12 kW
        assert res["demand_heat_kWh"] == pytest.approx(0.2)
        assert res["demand_heat_peak_kW"] == pytest.approx(0.12)
        # pump: [10,30] W -> 0.04 kWh, peak 0.03 kW
        assert res["pump_kWh"] == pytest.approx(0.04)
        assert res["pump_peak_kW"] == pytest.approx(0.03)
        # ratio pump / per-pipe loss
        assert res["ratio_pump_per_loss"] == pytest.approx(0.8)
        # supply - demand is NOT exposed as a loss claim
        assert "loss_balance_kWh" not in res
        # bookkeeping
        assert res["n_pipes"] == 2
        assert res["n_supply"] == 1
        assert res["n_demand"] == 2
        assert res["includes_return"] is False
        assert res["pump_attr"] == "pump_power_hydraulic"

    def test_missing_inputs_give_nan_not_raise(self):
        graph = ug.UESGraph()
        graph.add_building(name="S", position=Point(0, 0), is_supply_heating=True)
        res = pump_vs_loss(graph, timestep_hours=1.0)
        # No series anywhere -> NaN metrics, no exception.
        assert res["loss_per_pipe_kWh"] != res["loss_per_pipe_kWh"]  # NaN
        assert res["pump_kWh"] != res["pump_kWh"]
        assert res["n_pipes"] == 0

    def test_return_loss_added_to_per_pipe(self):
        """A supply graph that also carries Q_loss_return (return-line loss) sums
        BOTH lines into the per-pipe figure (VL+RL)."""
        g = _pump_loss_toy()
        for e, vals in zip(g.edges, ([1.0, 2.0], [3.0, 4.0])):
            idx = g.edges[e]["Q_loss"].index
            g.edges[e]["Q_loss_return"] = pd.Series(vals, index=idx)
        res = pump_vs_loss(g, timestep_hours=1.0)
        assert res["includes_return"] is True
        # per step: (5+15)+(1+3)=24, (10+20)+(2+4)=36 -> (24+36)/1000 = 0.06 kWh
        assert res["loss_per_pipe_kWh"] == pytest.approx(0.06)


class TestSupplyMaskRegistry:
    """The supply mask is exchangeable and auto-selected from comp_model."""

    def test_resolve_by_full_comp_model(self):
        entry, key = resolve_supply_mask(
            comp_model="AixLib.Fluid.DistrictHeatingCooling.Supplies.OpenLoop.SourceIdeal")
        assert key == "OpenLoop.SourceIdeal"
        assert entry is SUPPLY_MASKS["OpenLoop.SourceIdeal"]

    def test_resolve_by_short_key(self):
        entry, key = resolve_supply_mask(supply_model="OpenLoop.SourceIdeal")
        assert key == "OpenLoop.SourceIdeal"

    def test_unknown_falls_back_to_default(self):
        # Unknown comp_model -> default model (SourceIdeal) still resolves.
        entry, key = resolve_supply_mask(comp_model="Some.Unregistered.Station")
        assert key == "OpenLoop.SourceIdeal"

    def test_assign_supply_values_hydraulic(self):
        """SourceIdeal mask -> dp_pump and ideal hydraulic pump power."""
        graph = ug.UESGraph()
        s = graph.add_building(name="S1", position=Point(0, 0),
                               is_supply_heating=True)
        idx = pd.date_range("2024-01-01", periods=2, freq="h")
        df = pd.DataFrame({
            "networkModel.supplyS1.Q_flow":            [100.0, 200.0],
            "networkModel.supplyS1.senMasFlo.m_flow":  [2.0, 4.0],
            "networkModel.supplyS1.port_b.p":          [600000.0, 600000.0],
            "networkModel.supplyS1.port_a.p":          [200000.0, 200000.0],
        }, index=idx)

        MASK = {"supply": SUPPLY_MASKS["OpenLoop.SourceIdeal"]}
        n = assign_supply_values(graph, df, MASK, rho=1000.0)

        assert n == 1
        node = graph.nodes[s]
        assert list(node["heat_power_supply"]) == [100.0, 200.0]
        # dp_pump = port_b.p - port_a.p = 400000 Pa
        assert list(node["dp_pump"]) == [400000.0, 400000.0]
        # P_hyd = |m_flow|/rho * dp = [2/1000*4e5, 4/1000*4e5] = [800, 1600] W
        assert list(node["pump_power_hydraulic"]) == [800.0, 1600.0]


class TestLossAndPumpIntegration:
    """End-to-end on the real Pinola .mat: loss on edges, pump on supply."""

    def test_pipeline_assigns_loss_and_pump(self):
        test_data_dir = Path(__file__).parent / "test_analyze_data"
        mat_file_path = Path(__file__).parent.parent / "uesgraphs" / "data" / "Pinola_low_temp_network_inputs.mat"
        nodes_json_path = test_data_dir / "nodes.json"
        sysm_json_path = test_data_dir / "pinola_sysm.json"
        assert mat_file_path.exists()

        # Rebuild the cache with the current masks so the optional loss/supply
        # columns are present regardless of any stale cache on disk.
        check_input_file(file_path=str(mat_file_path), force_reconvert=True)

        graph = ug.UESGraph()
        graph.from_json(path=str(nodes_json_path), network_type="heating")
        graph.graph["name"] = "pinola_loss_pump"
        graph.graph["supply_type"] = "supply"

        result = analyze.assign_data_pipeline(
            graph=graph,
            simulation_data_path=str(mat_file_path),
            time_interval="15min",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 2),
            aixlib_version="2.1.0",
            system_model_path=str(sysm_json_path),
        )

        # Per-pipe heat loss assigned to edges - supply (Q_loss) AND return
        # (Q_loss_return), so a single supply graph covers VL+RL.
        edges_with_loss = [e for e in result.edges if "Q_loss" in result.edges[e]]
        assert edges_with_loss, "expected Q_loss on at least some edges"
        sample = result.edges[edges_with_loss[0]]["Q_loss"]
        assert hasattr(sample, "__len__") and len(sample) > 0
        assert any("Q_loss_return" in result.edges[e] for e in result.edges), \
            "expected return-line loss (Q_loss_return) mapped onto the supply graph"

        # Supply node carries thermal power, derived dp and hydraulic pump power.
        supply_nodes = [n for n in result.nodelist_building
                        if result.nodes[n].get("is_supply_heating")]
        assert supply_nodes
        sup = result.nodes[supply_nodes[0]]
        assert "heat_power_supply" in sup
        assert "dp_pump" in sup
        assert "pump_power_hydraulic" in sup

        # Comparison reduces to finite energies (loss may be negative by the
        # model's heatPort sign convention - that is fine, we keep it signed).
        res = pump_vs_loss(result, timestep_hours=0.25)
        assert res["pump_kWh"] == res["pump_kWh"] and res["pump_kWh"] >= 0
        assert res["loss_per_pipe_kWh"] == res["loss_per_pipe_kWh"]
        assert res["includes_return"] is True   # VL+RL from the single supply graph
        assert res["supply_heat_kWh"] == res["supply_heat_kWh"]
        assert res["demand_heat_kWh"] == res["demand_heat_kWh"]
        assert res["pump_attr"] == "pump_power_hydraulic"