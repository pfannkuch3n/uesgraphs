"""Roundtrip tests for uesgraph_io (node-link save/load)."""

import json
import os
import pytest

from uesgraphs import UESGraph
from uesgraphs.uesgraph_io import (
    FORMAT_TAG,
    graph_from_json,
    graph_to_json,
    load_graph,
)


@pytest.fixture(scope="module")
def datadir():
    test_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(test_dir, "test_from_geojson")
    assert os.path.isdir(data_dir)
    return data_dir


@pytest.fixture()
def graph_from_geojson(datadir):
    graph = UESGraph()
    graph.from_geojson(
        network_path=os.path.join(datadir, "network.geojson"),
        buildings_path=os.path.join(datadir, "buildings.geojson"),
        supply_path=os.path.join(datadir, "supply.geojson"),
        name="roundtrip_test",
    )
    return graph


def test_format_tag_in_payload(graph_from_geojson, tmp_path):
    out = graph_to_json(graph_from_geojson, str(tmp_path / "graph.json"))
    with open(out) as f:
        payload = json.load(f)
    assert payload["meta"]["format"] == FORMAT_TAG


def test_roundtrip_preserves_topology(graph_from_geojson, tmp_path):
    g1 = graph_from_geojson
    path = graph_to_json(g1, str(tmp_path / "graph.json"))
    g2 = graph_from_json(path, graph_class=UESGraph)

    assert set(g1.nodes()) == set(g2.nodes()), "Node IDs differ"
    assert set(map(frozenset, g1.edges())) == set(map(frozenset, g2.edges())), \
        "Edges differ"


def test_roundtrip_preserves_node_attributes(graph_from_geojson, tmp_path):
    g1 = graph_from_geojson
    path = graph_to_json(g1, str(tmp_path / "graph.json"))
    g2 = graph_from_json(path, graph_class=UESGraph)

    for n in g1.nodes():
        a1 = dict(g1.nodes[n])
        a2 = dict(g2.nodes[n])
        # position is a Shapely Point — compare coordinates separately
        if "position" in a1:
            assert a2["position"].x == pytest.approx(a1["position"].x)
            assert a2["position"].y == pytest.approx(a1["position"].y)
            a1.pop("position")
            a2.pop("position")
        for k, v1 in a1.items():
            assert k in a2, f"Node {n}: attribute {k!r} missing after roundtrip"
            v2 = a2[k]
            if isinstance(v1, tuple):
                v1 = list(v1)
            assert v2 == v1, f"Node {n}: attribute {k!r} differs ({v1!r} vs {v2!r})"


def test_roundtrip_preserves_edge_attributes(graph_from_geojson, tmp_path):
    g1 = graph_from_geojson
    path = graph_to_json(g1, str(tmp_path / "graph.json"))
    g2 = graph_from_json(path, graph_class=UESGraph)

    for u, v in g1.edges():
        a1 = dict(g1.edges[u, v])
        a2 = dict(g2.edges[u, v])
        for k, val in a1.items():
            assert k in a2, f"Edge ({u},{v}): attribute {k!r} missing"
            ref = list(val) if isinstance(val, tuple) else val
            assert a2[k] == ref, f"Edge ({u},{v}): {k!r} differs"


def test_roundtrip_restores_uesgraph_state(graph_from_geojson, tmp_path):
    g1 = graph_from_geojson
    path = graph_to_json(g1, str(tmp_path / "graph.json"))
    g2 = graph_from_json(path, graph_class=UESGraph)

    # nodelist_building must be populated
    assert len(g2.nodelist_building) == len(g1.nodelist_building)
    assert set(g2.nodelist_building) == set(g1.nodelist_building)

    # nodelists_heating must not be empty for a heating graph
    flat_h1 = [n for ids in g1.nodelists_heating.values() for n in ids]
    flat_h2 = [n for ids in g2.nodelists_heating.values() for n in ids]
    assert set(flat_h1) == set(flat_h2), "nodelists_heating mismatch"

    # next_node_number must be > max int node id
    max_id = max(n for n in g2.nodes() if isinstance(n, int))
    assert g2.next_node_number > max_id

    # nodes_by_name should be populated for named nodes
    assert len(g2.nodes_by_name) > 0


def test_load_graph_detects_node_link_format(graph_from_geojson, tmp_path):
    path = graph_to_json(graph_from_geojson, str(tmp_path / "graph.json"))
    g = load_graph(path, graph_class=UESGraph)
    assert set(g.nodes()) == set(graph_from_geojson.nodes())


def test_load_graph_falls_back_to_legacy(datadir):
    # reference.json was written by the legacy UESGraph.to_json
    g = load_graph(os.path.join(datadir, "reference.json"))
    assert len(g.nodes) > 0
