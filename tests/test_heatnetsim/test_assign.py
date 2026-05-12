"""Tests for uesgraphs.heatnetsim.assign — round-trip export → assign."""

import pandas as pd
import pytest

from uesgraphs.heatnetsim import assign_hns_results_to_uesgraph, uesgraph_to_hns_csv
from uesgraphs.heatnetsim.assign import _resolve_edge


def test_roundtrip_attaches_node_and_edge_series(small_graph, fake_results, tmp_path):
    paths = uesgraph_to_hns_csv(small_graph, tmp_path)

    assign_hns_results_to_uesgraph(
        small_graph,
        results=fake_results,
        mapping_path=paths["mapping"],
        include_return=True,
        derive_edge_dp=True,
    )

    for node_id in ["S", "N1", "N2", "D1", "D2"]:
        nd = small_graph.nodes[node_id]
        for key in ("temperature", "pressure", "temperature_return", "pressure_return"):
            assert isinstance(nd[key], pd.Series)
            assert len(nd[key]) == 3

    for edge in [("S", "N1"), ("N1", "N2"), ("N1", "D1"), ("N2", "D2")]:
        ed = small_graph.edges[edge]
        for key in ("m_flow", "m_flow_return", "dp"):
            assert isinstance(ed[key], pd.Series)
            assert len(ed[key]) == 3
        assert (ed["dp"] >= 0).all()


def test_assign_warm_side_uses_warm_indices(small_graph, fake_results, tmp_path):
    """Side='warm' must pull warm_idx (1..5) values, not cold_idx (6..10)."""
    paths = uesgraph_to_hns_csv(small_graph, tmp_path)

    assign_hns_results_to_uesgraph(
        small_graph,
        results=fake_results,
        mapping_path=paths["mapping"],
        side="warm",
        include_return=False,
        derive_edge_dp=False,
    )

    # Temperature for HNS index i, step s = 350 + 10*s + i.
    # S is the first node (warm_idx=1), so temperature[0] should be 351.0.
    assert small_graph.nodes["S"]["temperature"].iloc[0] == 351.0
    assert small_graph.nodes["S"]["temperature"].iloc[1] == 361.0
    assert small_graph.nodes["S"]["temperature"].iloc[2] == 371.0


def test_assign_cold_side_uses_cold_indices(small_graph, fake_results, tmp_path):
    paths = uesgraph_to_hns_csv(small_graph, tmp_path)

    assign_hns_results_to_uesgraph(
        small_graph,
        results=fake_results,
        mapping_path=paths["mapping"],
        side="cold",
        include_return=False,
        derive_edge_dp=False,
    )

    # S cold_idx = 6 → temperature[0] = 350 + 0 + 6 = 356.
    assert small_graph.nodes["S"]["temperature"].iloc[0] == 356.0


def test_resolve_edge_both_orientations(small_graph):
    """Either orientation must resolve to a key the graph accepts."""
    for u, v in [("S", "N1"), ("N1", "S")]:
        key = _resolve_edge(small_graph, u, v)
        assert small_graph.has_edge(*key)
    with pytest.raises(KeyError):
        _resolve_edge(small_graph, "S", "D2")


def test_assign_rejects_inconsistent_timestep_counts(small_graph, fake_results, tmp_path):
    paths = uesgraph_to_hns_csv(small_graph, tmp_path)
    broken = dict(fake_results)
    broken["pipeline_massflow"] = broken["pipeline_massflow"][:-1]  # 2 steps not 3

    with pytest.raises(ValueError, match="Inconsistent timestep counts"):
        assign_hns_results_to_uesgraph(small_graph, broken, paths["mapping"])


def test_assign_rejects_invalid_side(small_graph, fake_results, tmp_path):
    paths = uesgraph_to_hns_csv(small_graph, tmp_path)
    with pytest.raises(ValueError, match="side must be"):
        assign_hns_results_to_uesgraph(
            small_graph, fake_results, paths["mapping"], side="lukewarm"
        )


def test_assign_raises_on_missing_node(small_graph, fake_results, tmp_path):
    """If the mapping references a node id not in the graph, raise KeyError."""
    paths = uesgraph_to_hns_csv(small_graph, tmp_path)
    # mutate the small graph to drop a node that's still in the mapping
    del small_graph._node_data["D2"]

    with pytest.raises(KeyError, match="D2"):
        assign_hns_results_to_uesgraph(small_graph, fake_results, paths["mapping"])
