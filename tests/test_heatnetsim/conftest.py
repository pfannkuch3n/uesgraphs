"""Shared fixtures for heatnetsim tests.

Provides a small 5-node graph (1 EBU, 2 demand buildings, 2 network
junctions) duck-typed to match what ``uesgraph_to_hns_csv`` and
``assign_hns_results_to_uesgraph`` consume — no UESGraph dependency, so the
HNS modules can be tested in isolation.
"""

import pytest


class _CallableView:
    """networkx-style view: callable (graph.nodes()) and subscriptable (graph.nodes[id])."""

    def __init__(self, store, edges=False):
        self._store = store
        self._edges = edges

    def __call__(self, data=False):
        if self._edges:
            if data:
                return [(u, v, d) for (u, v), d in self._store.items()]
            return list(self._store)
        if data:
            return list(self._store.items())
        return list(self._store)

    def __iter__(self):
        return iter(self._store)

    def __contains__(self, key):
        if self._edges:
            u, v = key
            return (u, v) in self._store or (v, u) in self._store
        return key in self._store

    def __getitem__(self, key):
        if self._edges:
            u, v = key
            if (u, v) in self._store:
                return self._store[(u, v)]
            if (v, u) in self._store:
                return self._store[(v, u)]
            raise KeyError(key)
        return self._store[key]


class FakeGraph:
    """networkx.Graph-shaped duck for the heatnetsim modules."""

    def __init__(self, node_dicts, edge_tuples):
        self._node_data = {nid: dict(d) for nid, d in node_dicts}
        self._edge_data = {(u, v): dict(d) for u, v, d in edge_tuples}
        self.nodes = _CallableView(self._node_data)
        self.edges = _CallableView(self._edge_data, edges=True)

    def has_edge(self, u, v):
        return (u, v) in self._edge_data or (v, u) in self._edge_data


@pytest.fixture
def small_graph():
    """5-node toy graph: S(supply) --- N1 --- D1; N1 --- N2 --- D2."""
    nodes = [
        ("S",  {"node_type": "building", "is_supply_heating": True,
                "name": "ebu1", "altitude": 100.0}),
        ("N1", {"node_type": "network_heating", "altitude": 95.0}),
        ("N2", {"node_type": "network_heating", "altitude": 90.0}),
        ("D1", {"node_type": "building", "is_supply_heating": False,
                "name": "bldg_a"}),
        ("D2", {"node_type": "building", "is_supply_heating": False,
                "name": "bldg_b"}),
    ]
    edges = [
        ("S",  "N1", {"diameter": 0.2101, "dIns": 0.10, "length": 50.0,
                      "attr_dict": {"DN": 200, "id": "p1"}}),
        ("N1", "N2", {"diameter": 0.1325, "dIns": 0.10, "length": 30.0,
                      "attr_dict": {"DN": 125, "id": "p2"}}),
        ("N1", "D1", {"diameter": 0.0825, "dIns": 0.08, "length": 20.0,
                      "attr_dict": {"DN": 80,  "id": "p3"}}),
        ("N2", "D2", {"diameter": 0.0539, "dIns": 0.05, "length": 25.0,
                      "attr_dict": {"DN": 50,  "id": "p4"}}),
    ]
    return FakeGraph(nodes, edges)


@pytest.fixture
def fake_results():
    """Fake HNS results dict with T=3 timesteps for the small_graph.

    Indices follow build_index_map: 5 nodes × 2 (warm 1..5, cold 6..10), and
    4 edges × 2 (warm pipes 1..4, cold pipes 5..8). Values are deterministic
    so tests can assert exact numbers.
    """
    n_nodes_hns = 10
    n_pipes_hns = 8
    n_steps = 3

    nodal_temperature = [
        [float(350 + step * 10 + idx) for idx in range(1, n_nodes_hns + 1)]
        for step in range(n_steps)
    ]
    nodal_pressure = [
        [float(3e5 - 1000 * idx + step * 100) for idx in range(1, n_nodes_hns + 1)]
        for step in range(n_steps)
    ]
    pipeline_massflow = [
        [float(0.1 * idx + step * 0.01) for idx in range(1, n_pipes_hns + 1)]
        for step in range(n_steps)
    ]

    return {
        "nodal_temperature": nodal_temperature,
        "nodal_pressure": nodal_pressure,
        "pipeline_massflow": pipeline_massflow,
    }
