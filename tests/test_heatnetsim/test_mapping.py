"""Tests for uesgraphs.heatnetsim.mapping."""

from uesgraphs.heatnetsim.mapping import (
    build_heat_mapping,
    build_index_map,
)
from uesgraphs.heatnetsim.export import (
    build_ebu_records,
    build_pipe_records,
    build_substation_records,
)


PIPE_PARAMS = {
    "diameter_default": 0.1,
    "roughness_default": 2e-4,
    "thickness_insulation_default": 0.05,
    "heat_conductivity_insulation_default": 0.026,
}


def test_index_map_pairs_warm_and_cold(small_graph):
    mapping = build_index_map(small_graph)

    assert len(mapping) == 5
    n = len(mapping)
    for entry in mapping.values():
        assert 1 <= entry.warm_idx <= n
        assert n + 1 <= entry.cold_idx <= 2 * n
        assert entry.cold_idx == entry.warm_idx + n


def test_index_map_classifies_roles(small_graph):
    mapping = build_index_map(small_graph)

    assert mapping["S"].role == "supply"
    assert mapping["N1"].role == "network"
    assert mapping["N2"].role == "network"
    assert mapping["D1"].role == "demand"
    assert mapping["D2"].role == "demand"


def test_build_heat_mapping_invariants(small_graph):
    mapping = build_index_map(small_graph)
    pipe_recs = build_pipe_records(small_graph, mapping, PIPE_PARAMS)
    sub_recs = build_substation_records(
        small_graph, mapping,
        {**PIPE_PARAMS, "T_set_sec": 333.15, "delta_T_sec": 10.0,
         "delta_T_pri": 10.0, "heat_exchange": "heat_exchanger",
         "cooling_exchange": "heat_exchanger"},
    )
    ebu_recs = build_ebu_records(small_graph, mapping)

    hm = build_heat_mapping(small_graph, mapping, pipe_recs, sub_recs, ebu_recs)

    assert hm["meta"]["n_ues_nodes"] == 5
    assert hm["meta"]["n_ues_edges"] == 4
    assert hm["meta"]["cold_offset"] == 5

    for entry in hm["nodes"]:
        assert entry["warm_idx"] + hm["meta"]["cold_offset"] == entry["cold_idx"]

    for entry in hm["edges"]:
        # warm pipes get the first 4 indices, cold the next 4
        assert 1 <= entry["warm_pipe_idx"] <= 4
        assert 5 <= entry["cold_pipe_idx"] <= 8
