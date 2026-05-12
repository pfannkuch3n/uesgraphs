"""Export a UESGraph to HeatNetSim CSV files.

Position in the export pipeline:

    1. graph = UESGraph(); graph.from_geojson(...)
    2. apply_pipe_specs_to_graph(graph, "pipe_specs.csv")     # optional fallback
    3. uesgraph_to_hns_csv(graph, "output_dir/")              # this module

Step 2 is only needed when the GeoJSON does not already carry the catalog
columns (inner_diameter, d_ins, wall_thickness). The GeoJSON importer pulls
these in automatically when present; apply_pipe_specs_to_graph fills the gap
when only DN is on the edges.

The output directory will contain four CSV files that HeatNetSim's
``create_network_from_csv`` picks up via filename suffix:

    {prefix}_nodes.csv
    {prefix}_pipelines.csv
    {prefix}_substations.csv
    {prefix}_energybalancingunit.csv

Topology
--------
Dual-strand 4G/2G heat-only. For each uesgraph node, two HNS nodes are emitted
- a warm twin (index N) and a cold twin (index N + total_nodes). The cold
strand is the *return line*; HNS requires it for a closed hydraulic loop.
There is no second cooling network and no bidirectional flow.

Conventions enforced (read straight off ``HeatNetSim/network.py`` and
``substation.py``):

    * EBU:        inlet = cold (return),   outlet = warm (supply).
                  Both endpoints are reference nodes carrying ``pressure_pa``.
    * Substation: inlet = warm,            outlet = cold.
    * Cold pipe:  flow direction reversed relative to its warm twin.
    * cooling_exchange = 'heat_exchanger'  (HNS-recognised string; combined
                  with cooling_demand=0 in the time-series, this acts as an
                  inert dummy and gives 4G/2G heat-only behaviour).

Attribute resolution per edge: ``edge_data[<key>]`` first, then the matching
``*_default`` parameter. Existing edge values are never overwritten.
"""

import logging
from pathlib import Path

from uesgraphs.heatnetsim.io import write_csv, write_heat_mapping
from uesgraphs.heatnetsim.mapping import (
    HNS_EBU_FIELDS,
    HNS_NODE_FIELDS,
    HNS_PIPE_FIELDS,
    HNS_SUBSTATION_FIELDS,
    build_heat_mapping,
    build_index_map,
)


logger = logging.getLogger(__name__)


def _first_non_none(*values):
    """Return the first argument that is not None. Treats 0 / 0.0 as valid."""
    for v in values:
        if v is not None:
            return v
    return None


def _node_type_string(role: str, is_warm: bool) -> str:
    """Translate (role, side) into the HNS node_type string.

    Only ``'reference'`` has functional meaning to HNS (find_reference_nodes,
    find_non_junction_nodes); the other strings are descriptive only.
    """
    if role == "supply":
        return "reference"
    if role == "demand":
        return "building_inlet" if is_warm else "building_outlet"
    return "network"


def build_node_records(graph, mapping: dict, params: dict) -> list:
    """Emit one dict per HNS node (= 2 per uesgraph node)."""
    records = []
    p_warm = params["pressure_warm_pa"]
    p_cold = params["pressure_cold_pa"]

    for node_id, entry in mapping.items():
        is_ref = entry.role == "supply"
        altitude = entry.altitude if entry.altitude is not None else ""

        records.append({
            "node_index": entry.warm_idx,
            "pressure_pa": p_warm if is_ref else "",
            "altitude_m": altitude,
            "node_type": _node_type_string(entry.role, is_warm=True),
            "node_mode": "warm",
            "node_id": node_id,
            "name": entry.name or "",
        })
        records.append({
            "node_index": entry.cold_idx,
            "pressure_pa": p_cold if is_ref else "",
            "altitude_m": altitude,
            "node_type": _node_type_string(entry.role, is_warm=False),
            "node_mode": "cold",
            "node_id": node_id,
            "name": entry.name or "",
        })
    return records


def _resolve_pipe_attrs(edge_data: dict, params: dict) -> dict:
    """Pull pipe attributes from edge data; fall back to defaults if missing."""
    return {
        "diameter_m": _first_non_none(
            edge_data.get("diameter"),
            params["diameter_default"],
        ),
        "length_m": _first_non_none(
            edge_data.get("length"),
            0,
        ),
        "roughness_m": _first_non_none(
            edge_data.get("roughness"),
            params["roughness_default"],
        ),
        "thickness_insulation_m": _first_non_none(
            edge_data.get("dIns"),
            params["thickness_insulation_default"],
        ),
        "heat_conductivity_insulation": _first_non_none(
            edge_data.get("heat_conductivity_insulation"),
            params["heat_conductivity_insulation_default"],
        ),
    }


def build_pipe_records(graph, mapping: dict, params: dict) -> list:
    """Emit two pipes per uesgraph edge: one warm, one cold (reversed direction)."""
    records = []
    pipeline_index = 0
    edges = list(graph.edges(data=True))

    # Warm strand: same direction as the uesgraph edge.
    for u, v, data in edges:
        pipeline_index += 1
        attrs = _resolve_pipe_attrs(data, params)
        attr_dict = data.get("attr_dict") or {}
        records.append({
            "pipeline_index": pipeline_index,
            "inlet_index": mapping[u].warm_idx,
            "outlet_index": mapping[v].warm_idx,
            "pipe_mode": "warm",
            "drawing_pipe_id": attr_dict.get("id", ""),
            **attrs,
        })

    # Cold strand: reversed direction (return flow).
    for u, v, data in edges:
        pipeline_index += 1
        attrs = _resolve_pipe_attrs(data, params)
        attr_dict = data.get("attr_dict") or {}
        records.append({
            "pipeline_index": pipeline_index,
            "inlet_index": mapping[v].cold_idx,
            "outlet_index": mapping[u].cold_idx,
            "pipe_mode": "cold",
            "drawing_pipe_id": attr_dict.get("id", ""),
            **attrs,
        })

    return records


def build_substation_records(graph, mapping: dict, params: dict) -> list:
    """Emit one substation per demand building. inlet=warm, outlet=cold."""
    records = []
    sub_idx = 0
    for node_id, entry in mapping.items():
        if entry.role != "demand":
            continue
        sub_idx += 1
        records.append({
            "substation_index": sub_idx,
            "inlet_index": entry.warm_idx,
            "outlet_index": entry.cold_idx,
            "T_set_sec": params["T_set_sec"],
            "delta_T_sec": params["delta_T_sec"],
            "delta_T_pri": params["delta_T_pri"],
            "heat_exchange": params["heat_exchange"],
            "cooling_exchange": params["cooling_exchange"],
            "substation_name": entry.name or f"sub_{sub_idx}",
        })
    return records


def build_ebu_records(graph, mapping: dict) -> list:
    """Emit one EBU per supply building. inlet=cold (return), outlet=warm (supply)."""
    records = []
    ebu_idx = 0
    for node_id, entry in mapping.items():
        if entry.role != "supply":
            continue
        ebu_idx += 1
        records.append({
            "ebu_index": ebu_idx,
            "inlet_index": entry.cold_idx,
            "outlet_index": entry.warm_idx,
            "ebu_name": entry.name or f"ebu_{ebu_idx}",
        })
    return records


def uesgraph_to_hns_csv(
    graph,
    output_dir,
    file_prefix="heat",
    delimiter=";",
    # Pipe attribute defaults (used when the edge doesn't have the value).
    diameter_default=0.1,
    roughness_default=2e-4,
    thickness_insulation_default=0.05,
    heat_conductivity_insulation_default=0.026,
    # Reference-node pressures at the EBU.
    pressure_warm_pa=3e5,
    pressure_cold_pa=2e5,
    # Substation defaults.
    T_set_sec=333.15,
    delta_T_sec=10.0,
    delta_T_pri=10.0,
    heat_exchange="heat_exchanger",
    cooling_exchange="heat_exchanger",
    logger=None,
):
    """Convert a uesgraphs UESGraph into HeatNetSim-compatible CSV files.

    Writes four CSV files plus a ``{file_prefix}_mapping.json`` sidecar
    into ``output_dir``, and returns a dict of the paths written. The graph
    is *not* modified.

    Mode is fixed to dual-strand 4G/2G heat-only. The cold strand is emitted
    as the return line because HNS requires both sides for a closed loop;
    cooling_demand should be set to zero everywhere in the time-series CSVs.

    Parameters
    ----------
    graph : UESGraph
        The graph to export. Must have at least one node with
        ``is_supply_heating=True`` or ``is_supply_cooling=True`` (becomes EBU).
        Pipe attributes (``diameter``, ``dIns``, etc.) are read directly from
        edge data; missing attributes fall back to the ``*_default`` parameters.
    output_dir : str or Path
    file_prefix : str, default "heat"
        Filename stem for the four output CSV files.
    delimiter : str, default ";"
        CSV delimiter — HNS's loader uses semicolons.
    diameter_default, roughness_default, thickness_insulation_default,         heat_conductivity_insulation_default : float
        Pipe attribute fallbacks if the edge does not carry the value.
    pressure_warm_pa, pressure_cold_pa : float
        Reference-node pressures at the EBU (warm and cold side).
    T_set_sec, delta_T_sec, delta_T_pri : float
        Substation defaults.
    heat_exchange, cooling_exchange : str
        HNS-recognised strings. Valid: 'heat_pump' / 'heat_exchanger' for
        ``heat_exchange``; 'chiller' / 'heat_exchanger' for ``cooling_exchange``.
        Default ``cooling_exchange='heat_exchanger'`` is the 4G/2G heat-only
        configuration: HNS won't run a cooling-side calculation.
    logger : logging.Logger, optional

    Returns
    -------
    dict
        ``{'nodes', 'pipelines', 'substations', 'ebu', 'mapping'} -> Path``.

    Raises
    ------
    ValueError
        If the graph has no supply node — HNS requires at least one EBU and
        ``Network.__init__`` would crash on construction.
    """
    log = logger if logger is not None else globals()["logger"]
    output_dir = Path(output_dir)

    params = {
        "diameter_default": diameter_default,
        "roughness_default": roughness_default,
        "thickness_insulation_default": thickness_insulation_default,
        "heat_conductivity_insulation_default": heat_conductivity_insulation_default,
        "pressure_warm_pa": pressure_warm_pa,
        "pressure_cold_pa": pressure_cold_pa,
        "T_set_sec": T_set_sec,
        "delta_T_sec": delta_T_sec,
        "delta_T_pri": delta_T_pri,
        "heat_exchange": heat_exchange,
        "cooling_exchange": cooling_exchange,
    }

    # Phase 1 — index mapping.
    mapping = build_index_map(graph)
    n_supply = sum(1 for e in mapping.values() if e.role == "supply")
    n_demand = sum(1 for e in mapping.values() if e.role == "demand")
    n_network = sum(1 for e in mapping.values() if e.role == "network")
    log.info(
        "Index map built: %d nodes total (%d network, %d demand, %d supply).",
        len(mapping), n_network, n_demand, n_supply,
    )
    if n_supply == 0:
        raise ValueError(
            "No supply node found (no node with is_supply_heating=True or "
            "is_supply_cooling=True). HNS requires at least one EBU."
        )

    # Phase 2 — build records.
    node_recs = build_node_records(graph, mapping, params)
    pipe_recs = build_pipe_records(graph, mapping, params)
    sub_recs = build_substation_records(graph, mapping, params)
    ebu_recs = build_ebu_records(graph, mapping)
    log.info(
        "Records built: %d node rows, %d pipe rows, %d substation rows, %d EBU rows.",
        len(node_recs), len(pipe_recs), len(sub_recs), len(ebu_recs),
    )

    # Phase 3 — write CSVs.
    paths = {
        "nodes":       output_dir / f"{file_prefix}_nodes.csv",
        "pipelines":   output_dir / f"{file_prefix}_pipelines.csv",
        "substations": output_dir / f"{file_prefix}_substations.csv",
        "ebu":         output_dir / f"{file_prefix}_energybalancingunit.csv",
    }
    write_csv(paths["nodes"],       node_recs, HNS_NODE_FIELDS,       delimiter)
    write_csv(paths["pipelines"],   pipe_recs, HNS_PIPE_FIELDS,       delimiter)
    write_csv(paths["substations"], sub_recs,  HNS_SUBSTATION_FIELDS, delimiter)
    write_csv(paths["ebu"],         ebu_recs,  HNS_EBU_FIELDS,        delimiter)
    log.info("CSV files written to %s", output_dir)

    # Phase 4 — sidecar mapping JSON.
    heat_mapping = build_heat_mapping(graph, mapping, pipe_recs, sub_recs, ebu_recs)
    paths["mapping"] = output_dir / f"{file_prefix}_mapping.json"
    write_heat_mapping(paths["mapping"], heat_mapping)
    log.info("Heat mapping written to %s", paths["mapping"])

    return paths
