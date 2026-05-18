"""Native JSON I/O for UESGraph using NetworkX's node-link format.

Drop-in alternative to ``UESGraph.to_json`` / ``UESGraph.from_json`` that:
  - preserves internal node IDs (edge tuples are stable across roundtrips)
  - preserves ALL node and edge attributes verbatim, including nested dicts
    like ``attr_dict`` (no ``all_data`` flag required, no implicit dropping)
  - preserves graph-level attributes (``uesgraph.graph`` dict)
  - reconstructs internal state (``nodelists_heating``, ``nodelist_building``,
    ``nodes_by_name``, ``next_node_number``) from node attributes on load
  - handles shapely Point positions transparently (serialized as x/y)

Tuples are converted to lists during serialization (JSON has no tuple type).
"""

from __future__ import annotations

import datetime
import json
import os
import uuid
from typing import Any

import networkx as nx
from shapely.geometry import Point


FORMAT_TAG = "node_link_v1"
_POSITION_KEYS = ("position",)


def _node_attrs_to_jsonable(node_data: dict) -> dict:
    out = {}
    for k, v in node_data.items():
        if k in _POSITION_KEYS and isinstance(v, Point):
            out["_position_x"] = v.x
            out["_position_y"] = v.y
        else:
            out[k] = _make_jsonable(v)
    return out


def _node_attrs_from_jsonable(node_data: dict) -> dict:
    out = dict(node_data)
    if "_position_x" in out and "_position_y" in out:
        out["position"] = Point(out.pop("_position_x"), out.pop("_position_y"))
    return out


def _make_jsonable(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_make_jsonable(v) for v in value]
    if isinstance(value, list):
        return [_make_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: _make_jsonable(v) for k, v in value.items()}
    return value


def graph_to_json(
    graph: nx.Graph,
    path: str,
    description: str = "uesgraph node-link export",
    prettyprint: bool = True,
) -> str:
    """Write graph to JSON using NetworkX's node-link format."""
    nodes_jsonable = []
    for n in graph.nodes():
        entry = {"id": n}
        entry.update(_node_attrs_to_jsonable(dict(graph.nodes[n])))
        nodes_jsonable.append(entry)

    edges_jsonable = []
    for u, v in graph.edges():
        entry = {"source": u, "target": v}
        for k, val in graph.edges[u, v].items():
            entry[k] = _make_jsonable(val)
        edges_jsonable.append(entry)

    graph_attrs = {}
    for k, v in graph.graph.items():
        try:
            json.dumps(v)
            graph_attrs[k] = v
        except (TypeError, ValueError):
            pass

    payload = {
        "meta": {
            "description": description,
            "source": "uesgraph (node_link_data)",
            "created": str(datetime.datetime.now()),
            "input_id": str(uuid.uuid4()),
            "format": FORMAT_TAG,
        },
        "graph": graph_attrs,
        "directed": graph.is_directed(),
        "multigraph": graph.is_multigraph(),
        "nodes": nodes_jsonable,
        "links": edges_jsonable,
    }

    path = os.path.abspath(path)
    with open(path, "w") as f:
        if prettyprint:
            json.dump(payload, f, indent=4, default=str)
        else:
            json.dump(payload, f, default=str)
    return path


def graph_from_json(path: str, graph_class=None) -> nx.Graph:
    """Load a graph written by ``graph_to_json``.

    Pass ``graph_class=UESGraph`` to get UESGraph internals (nodelists,
    counters) restored too.
    """
    with open(path, "r") as f:
        payload = json.load(f)

    if graph_class is None:
        graph_class = nx.Graph

    g = graph_class()
    g.graph.update(payload.get("graph", {}))

    for entry in payload["nodes"]:
        nid = entry["id"]
        attrs = {k: v for k, v in entry.items() if k != "id"}
        attrs = _node_attrs_from_jsonable(attrs)
        g.add_node(nid, **attrs)

    for entry in payload["links"]:
        u, v = entry["source"], entry["target"]
        attrs = {k: v for k, v in entry.items() if k not in ("source", "target")}
        g.add_edge(u, v, **attrs)

    if hasattr(g, "nodelist_building"):
        _rebuild_uesgraph_state(g)
    return g


def _rebuild_uesgraph_state(g) -> None:
    g.nodelist_building = []
    g.nodes_by_name = {}

    nodelist_attrs = {
        "network_heating": "nodelists_heating",
        "network_cooling": "nodelists_cooling",
        "network_electricity": "nodelists_electricity",
        "network_gas": "nodelists_gas",
        "network_others": "nodelists_others",
    }
    for attr in nodelist_attrs.values():
        d = getattr(g, attr, None)
        if d is None:
            continue
        for k in d:
            d[k] = []

    if hasattr(g, "nodelist_street"):
        g.nodelist_street = []

    max_int_id = 1000
    for n, data in g.nodes(data=True):
        if isinstance(n, int):
            max_int_id = max(max_int_id, n)

        node_type = data.get("node_type", "")
        if "name" in data:
            g.nodes_by_name[data["name"]] = n

        if "building" in node_type or "supply" in node_type:
            g.nodelist_building.append(n)
        elif "street" in node_type and hasattr(g, "nodelist_street"):
            g.nodelist_street.append(n)
        else:
            for prefix, attr in nodelist_attrs.items():
                if prefix in node_type:
                    container = getattr(g, attr, None)
                    if container is not None:
                        container.setdefault("default", []).append(n)
                    break

    g.next_node_number = max_int_id + 1


def load_graph(path: str, graph_class=None) -> nx.Graph:
    """Load a graph, auto-detecting node-link vs legacy uesgraphs format.

    For node-link files, calls :func:`graph_from_json`. For legacy files
    (written by ``UESGraph.to_json``), falls back to ``UESGraph.from_json``
    so older artifacts still load. Pass ``graph_class=UESGraph`` to fully
    restore internal state from node-link files.
    """
    with open(path, "r") as f:
        payload = json.load(f)

    if payload.get("meta", {}).get("format") == FORMAT_TAG:
        return graph_from_json(path, graph_class=graph_class)

    from uesgraphs.uesgraph import UESGraph
    g = UESGraph()
    g.from_json(path, network_type="heating")
    return g
