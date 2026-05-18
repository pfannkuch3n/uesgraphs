"""Tests for uesgraphs.heatnetsim.export — port of the legacy __main__ smoke test."""

import json

import pytest

from uesgraphs.heatnetsim import uesgraph_to_hns_csv


def test_export_writes_expected_csv_shapes(small_graph, tmp_path):
    paths = uesgraph_to_hns_csv(small_graph, tmp_path)

    node_csv = paths["nodes"].read_text().splitlines()
    pipe_csv = paths["pipelines"].read_text().splitlines()
    sub_csv  = paths["substations"].read_text().splitlines()
    ebu_csv  = paths["ebu"].read_text().splitlines()

    assert len(node_csv) == 1 + 2 * 5   # header + 2 HNS nodes per ues node
    assert len(pipe_csv) == 1 + 2 * 4   # header + warm/cold per edge
    assert len(sub_csv)  == 1 + 2       # header + 2 demand buildings
    assert len(ebu_csv)  == 1 + 1       # header + 1 supply


def test_export_ebu_direction(small_graph, tmp_path):
    """EBU must go cold→warm (return → supply)."""
    paths = uesgraph_to_hns_csv(small_graph, tmp_path)
    ebu_row = paths["ebu"].read_text().splitlines()[1].split(";")
    inlet, outlet = int(ebu_row[1]), int(ebu_row[2])
    assert inlet > 5 and outlet <= 5


def test_export_substation_direction(small_graph, tmp_path):
    """Substation must go warm→cold (supply → return)."""
    paths = uesgraph_to_hns_csv(small_graph, tmp_path)
    for sub_line in paths["substations"].read_text().splitlines()[1:]:
        cells = sub_line.split(";")
        si, so = int(cells[1]), int(cells[2])
        assert si <= 5 and so > 5


def test_export_cold_pipe_reversed(small_graph, tmp_path):
    """Each cold pipe is the reverse of its warm twin (shifted by cold_offset)."""
    paths = uesgraph_to_hns_csv(small_graph, tmp_path)
    pipe_csv = paths["pipelines"].read_text().splitlines()
    warm_pipes = [p.split(";") for p in pipe_csv[1:5]]
    cold_pipes = [p.split(";") for p in pipe_csv[5:9]]
    for w, c in zip(warm_pipes, cold_pipes):
        w_in, w_out = int(w[1]), int(w[2])
        c_in, c_out = int(c[1]), int(c[2])
        assert c_in == w_out + 5 and c_out == w_in + 5


def test_export_uses_edge_diameter_not_default(small_graph, tmp_path):
    paths = uesgraph_to_hns_csv(small_graph, tmp_path)
    warm_pipes = paths["pipelines"].read_text().splitlines()[1:5]
    for row in warm_pipes:
        diameter = float(row.split(";")[3])
        assert diameter != 0.1, "diameter looks like default — edge attr not picked up"


def test_export_raises_without_supply(tmp_path):
    """Removing the supply node must trip the no-EBU guard."""
    from tests.test_heatnetsim.conftest import FakeGraph

    nodes = [
        ("N1", {"node_type": "network_heating"}),
        ("D1", {"node_type": "building"}),
    ]
    edges = [("N1", "D1", {"diameter": 0.05, "length": 10.0})]
    g = FakeGraph(nodes, edges)

    with pytest.raises(ValueError, match="No supply node"):
        uesgraph_to_hns_csv(g, tmp_path)


def test_heat_mapping_json_invariants(small_graph, tmp_path):
    paths = uesgraph_to_hns_csv(small_graph, tmp_path)
    hm = json.loads(paths["mapping"].read_text())

    assert len(hm["nodes"]) == 5
    assert len(hm["edges"]) == 4

    cold_offset = hm["meta"]["cold_offset"]
    for n in hm["nodes"]:
        assert n["warm_idx"] + cold_offset == n["cold_idx"]

    # cross-check against the CSV indices that were actually written
    node_csv = paths["nodes"].read_text().splitlines()
    pipe_csv = paths["pipelines"].read_text().splitlines()
    node_indices = {int(line.split(";")[0]) for line in node_csv[1:]}
    pipe_indices = {int(line.split(";")[0]) for line in pipe_csv[1:]}

    for n in hm["nodes"]:
        assert n["warm_idx"] in node_indices
        assert n["cold_idx"] in node_indices
    for e in hm["edges"]:
        assert e["warm_pipe_idx"] in pipe_indices
        assert e["cold_pipe_idx"] in pipe_indices
