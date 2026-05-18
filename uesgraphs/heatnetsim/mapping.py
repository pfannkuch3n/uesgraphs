"""Index mapping between UESGraph nodes/edges and HNS indices.

Single source of truth for warm/cold pairing. The cold-strand offset equals
the total number of nodes, so ``warm_idx + cold_offset == cold_idx`` for every
node. All downstream record builders consume only ``entry.warm_idx`` /
``entry.cold_idx`` — the offset is computed in exactly one place
(:func:`build_index_map`).
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal, Optional


# ---------------------------------------------------------------------------
# HNS CSV schemas. Hardcoded — HNS's loader expects exactly these column names.
# ---------------------------------------------------------------------------

HNS_NODE_FIELDS = [
    "node_index", "pressure_pa", "altitude_m", "node_type", "node_mode",
    # debug-only, ignored by HNS:
    "node_id", "name",
]

HNS_PIPE_FIELDS = [
    "pipeline_index", "inlet_index", "outlet_index",
    "diameter_m", "length_m", "roughness_m",
    "thickness_insulation_m", "heat_conductivity_insulation",
    "pipe_mode",
    # debug-only:
    "drawing_pipe_id",
]

HNS_SUBSTATION_FIELDS = [
    "substation_index", "inlet_index", "outlet_index",
    "T_set_sec", "delta_T_sec", "delta_T_pri",
    "heat_exchange", "cooling_exchange",
    # debug-only:
    "substation_name",
]

HNS_EBU_FIELDS = [
    "ebu_index", "inlet_index", "outlet_index",
    # debug-only:
    "ebu_name",
]


@dataclass
class IndexEntry:
    """Maps a single uesgraphs node to its pair of HNS node indices."""
    warm_idx: int
    cold_idx: int
    role: Literal["network", "demand", "supply"]
    name: Optional[str] = None
    altitude: Optional[float] = None


def _classify_node(data: dict) -> str:
    """Categorize a uesgraphs node by role.

    Buildings flagged with ``is_supply_heating`` *or* ``is_supply_cooling`` are
    EBUs (supply). All other buildings are demand. Anything else is a plain
    network junction.
    """
    node_type = (data.get("node_type") or "").lower()
    if "building" in node_type:
        if data.get("is_supply_heating") or data.get("is_supply_cooling"):
            return "supply"
        return "demand"
    return "network"


def build_index_map(graph) -> dict:
    """Build ``{uesgraph_node_id: IndexEntry}`` for all nodes.

    The cold-strand offset equals the total number of nodes, so::

        warm_idx in [1 .. N]
        cold_idx in [N+1 .. 2N]
    """
    nodes = list(graph.nodes(data=True))
    n = len(nodes)
    mapping = {}
    for i, (node_id, data) in enumerate(nodes, start=1):
        mapping[node_id] = IndexEntry(
            warm_idx=i,
            cold_idx=i + n,
            role=_classify_node(data),
            name=data.get("name"),
            altitude=data.get("altitude"),
        )
    return mapping


def _extract_position(node_data: dict):
    """Return [x, y] from node data, or None if no position is set.

    Handles shapely Point objects and plain (x, y) sequences.
    """
    pos = node_data.get("position")
    if pos is None:
        return None
    if hasattr(pos, "x") and hasattr(pos, "y"):          # shapely Point
        return [pos.x, pos.y]
    try:
        lst = list(pos)
        if len(lst) >= 2:
            return [float(lst[0]), float(lst[1])]
    except TypeError:
        pass
    return None


def build_heat_mapping(
    graph,
    mapping: dict,
    pipe_recs: list,
    sub_recs: list,
    ebu_recs: list,
) -> dict:
    """Assemble the heat_mapping dict (serialised later as JSON).

    Correlates every uesgraph node / edge with the HNS indices that were
    actually written to the CSV files, so simulation results can be mapped
    back to UESGraph entities downstream.

    Parameters
    ----------
    graph     : UESGraph (or compatible fake)
    mapping   : {node_id: IndexEntry} from build_index_map
    pipe_recs : list of dicts from build_pipe_records
                (first n_edges entries = warm strand, next n_edges = cold strand)
    sub_recs  : list of dicts from build_substation_records
    ebu_recs  : list of dicts from build_ebu_records

    Returns
    -------
    dict with keys "meta", "nodes", "edges" — ready for json.dump.
    """
    n_ues_nodes = len(mapping)
    n_ues_edges = len(pipe_recs) // 2   # warm + cold split is always 50/50

    sub_by_warm = {rec["inlet_index"]:  rec["substation_index"] for rec in sub_recs}
    ebu_by_warm = {rec["outlet_index"]: rec["ebu_index"]        for rec in ebu_recs}

    node_data_by_id = dict(graph.nodes(data=True))

    node_entries = []
    for node_id, entry in mapping.items():
        ndata = node_data_by_id.get(node_id) or {}
        node_entries.append({
            "ues_node_id":       node_id,
            "name":              entry.name,
            "node_type":         ndata.get("node_type"),
            "role":              entry.role,
            "warm_idx":          entry.warm_idx,
            "cold_idx":          entry.cold_idx,
            "position":          _extract_position(ndata),
            "substation_index":  sub_by_warm.get(entry.warm_idx),
            "ebu_index":         ebu_by_warm.get(entry.warm_idx),
        })

    # pipe_recs layout guaranteed by build_pipe_records:
    #   [0 .. n-1]   warm pipes  (same order as graph.edges)
    #   [n .. 2n-1]  cold pipes  (same order as graph.edges)
    warm_pipes = pipe_recs[:n_ues_edges]
    cold_pipes = pipe_recs[n_ues_edges:]
    graph_edges = list(graph.edges(data=True))

    edge_entries = []
    for (u, v, edata), wp, cp in zip(graph_edges, warm_pipes, cold_pipes):
        attr_dict = edata.get("attr_dict") or {}
        edge_entries.append({
            "ues_u":           u,
            "ues_v":           v,
            "drawing_pipe_id": attr_dict.get("id"),
            "warm_pipe_idx":   wp["pipeline_index"],
            "cold_pipe_idx":   cp["pipeline_index"],
            "length_m":        float(wp.get("length_m") or 0),
        })

    return {
        "meta": {
            "n_ues_nodes": n_ues_nodes,
            "cold_offset":  n_ues_nodes,
            "n_ues_edges":  n_ues_edges,
            "created_at":   datetime.now(timezone.utc).isoformat(),
        },
        "nodes": node_entries,
        "edges": edge_entries,
    }
