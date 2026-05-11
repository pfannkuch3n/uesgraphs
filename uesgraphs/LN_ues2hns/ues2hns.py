"""
Convert a uesgraphs UESGraph to HeatNetSim CSV files.

Position in the export pipeline:

    1. graph = UESGraph(); graph.from_geojson(...)
    2. apply_pipe_specs_to_graph(graph, "pipe_specs.csv")     # optional
    3. uesgraph_to_hns_csv(graph, "output_dir/")              # this module

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

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional
import json
from datetime import datetime, timezone


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. HNS schemas. Hardcoded — HNS's loader expects exactly these column names.
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


# ---------------------------------------------------------------------------
# 2. Index mapping — single source of truth for warm/cold pairing.
# ---------------------------------------------------------------------------

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

    This is the *only* place where the ``+ N`` offset is computed. All
    downstream builders consume only ``entry.warm_idx`` / ``entry.cold_idx``.
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


# ---------------------------------------------------------------------------
# 3. Record builders — pure functions, return list of dicts ready to write.
# ---------------------------------------------------------------------------

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
    """Pull pipe attributes from edge data; fall back to defaults if missing.

    Source priority for each field is: direct edge attribute -> default.
    Existing values on the edge always win - this is what
    ``apply_pipe_specs_to_graph`` is for upstream.
    """
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


# ---------------------------------------------------------------------------
# 4. CSV writer.
# ---------------------------------------------------------------------------

def _write_csv(path: Path, records: list, fieldnames: list, delimiter: str = ";"):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=delimiter)
        writer.writeheader()
        for record in records:
            writer.writerow({k: record.get(k, "") for k in fieldnames})

# ---------------------------------------------------------------------------
# 5. Mapping writer to reverse engineer sim data onto uesgraph
# ---------------------------------------------------------------------------

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

    # Fast lookups: warm_idx -> assigned sub / ebu index (None if not applicable)
    sub_by_warm = {rec["inlet_index"]:  rec["substation_index"] for rec in sub_recs}
    ebu_by_warm = {rec["outlet_index"]: rec["ebu_index"]        for rec in ebu_recs}

    node_data_by_id = dict(graph.nodes(data=True))

    # --- node entries ---
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

    # --- edge entries ---
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
            "drawing_pipe_id": attr_dict.get("id"),          # None → JSON null
            "warm_pipe_idx":   wp["pipeline_index"],
            "cold_pipe_idx":   cp["pipeline_index"],
            "length_m":        float(wp.get("length_m") or 0),
        })

    return {
        "meta": {
            "n_ues_nodes": n_ues_nodes,
            "cold_offset":  n_ues_nodes,   # warm_idx + cold_offset == cold_idx always
            "n_ues_edges":  n_ues_edges,
            "created_at":   datetime.now(timezone.utc).isoformat(),
        },
        "nodes": node_entries,
        "edges": edge_entries,
    }

def _write_heat_mapping(path: Path, heat_mapping: dict) -> None:
    """Serialise *heat_mapping* to JSON at *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(heat_mapping, f, indent=2)

# ---------------------------------------------------------------------------
# 6. Public API
# ---------------------------------------------------------------------------

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

    Writes four files into ``output_dir`` and returns a dict of the paths
    written. The graph is *not* modified.

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
        ``{'nodes', 'pipelines', 'substations', 'ebu'} -> Path``.

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

    # Phase 3 — write.
    paths = {
        "nodes":       output_dir / f"{file_prefix}_nodes.csv",
        "pipelines":   output_dir / f"{file_prefix}_pipelines.csv",
        "substations": output_dir / f"{file_prefix}_substations.csv",
        "ebu":         output_dir / f"{file_prefix}_energybalancingunit.csv",
    }
    _write_csv(paths["nodes"],       node_recs, HNS_NODE_FIELDS,       delimiter)
    _write_csv(paths["pipelines"],   pipe_recs, HNS_PIPE_FIELDS,       delimiter)
    _write_csv(paths["substations"], sub_recs,  HNS_SUBSTATION_FIELDS, delimiter)
    _write_csv(paths["ebu"],         ebu_recs,  HNS_EBU_FIELDS,        delimiter)
    log.info("CSV files written to %s", output_dir)

    # Phase 4 — sidecar mapping JSON.
    heat_mapping = build_heat_mapping(graph, mapping, pipe_recs, sub_recs, ebu_recs)
    paths["mapping"] = output_dir / f"{file_prefix}_heat_mapping.json"
    _write_heat_mapping(paths["mapping"], heat_mapping)
    log.info("Heat mapping written to %s", paths["mapping"])

    return paths


# ---------------------------------------------------------------------------
# Test — run with `python ues2hns.py`
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """Synthetic 5-node graph: 1 EBU, 2 demand buildings, 2 network junctions.

    Topology:
        supply --- net1 --- demand1
                     |
                   net2 --- demand2
    """
    import tempfile, shutil

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    class _FakeGraph:
        def __init__(self, node_dicts, edge_tuples):
            self._nodes = node_dicts
            self._edges = edge_tuples

        def nodes(self, data=False):
            if data:
                return [(nid, d) for (nid, d) in self._nodes]
            return [nid for (nid, _) in self._nodes]

        def edges(self, data=False):
            if data:
                return [(u, v, d) for (u, v, d) in self._edges]
            return [(u, v) for (u, v, _) in self._edges]

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
    g = _FakeGraph(nodes, edges)

    tmp = Path(tempfile.mkdtemp(prefix="ues2hns_smoke_"))
    try:
        paths = uesgraph_to_hns_csv(g, tmp)

        for label, p in paths.items():
            print(f"\n===== {label}  ->  {p.name} =====")
            print(p.read_text())

        node_csv = paths["nodes"].read_text().splitlines()
        pipe_csv = paths["pipelines"].read_text().splitlines()
        sub_csv  = paths["substations"].read_text().splitlines()
        ebu_csv  = paths["ebu"].read_text().splitlines()

        assert len(node_csv) == 1 + 2 * 5,  f"expected 11 lines, got {len(node_csv)}"
        assert len(pipe_csv) == 1 + 2 * 4,  f"expected 9 lines,  got {len(pipe_csv)}"
        assert len(sub_csv)  == 1 + 2,      f"expected 3 lines,  got {len(sub_csv)}"
        assert len(ebu_csv)  == 1 + 1,      f"expected 2 lines,  got {len(ebu_csv)}"

        ebu_data = ebu_csv[1].split(";")
        ebu_inlet, ebu_outlet = int(ebu_data[1]), int(ebu_data[2])
        assert ebu_inlet > 5 and ebu_outlet <= 5, (
            f"EBU direction wrong: inlet={ebu_inlet} (should be cold>5), "
            f"outlet={ebu_outlet} (should be warm<=5)"
        )

        for sub_line in sub_csv[1:]:
            cells = sub_line.split(";")
            si, so = int(cells[1]), int(cells[2])
            assert si <= 5 and so > 5, f"sub direction wrong: inlet={si}, outlet={so}"

        warm_pipes = [p.split(";") for p in pipe_csv[1:5]]
        cold_pipes = [p.split(";") for p in pipe_csv[5:9]]
        for w, c in zip(warm_pipes, cold_pipes):
            w_in, w_out = int(w[1]), int(w[2])
            c_in, c_out = int(c[1]), int(c[2])
            assert c_in == w_out + 5 and c_out == w_in + 5, (
                f"cold pipe direction wrong: warm ({w_in}->{w_out}), "
                f"cold ({c_in}->{c_out})"
            )

        for w in warm_pipes:
            assert float(w[3]) != 0.1, f"diameter looks like default: {w[3]}"

        print("\nAll invariants passed.")
        
        # --- heat_mapping.json ---
        with open(paths["mapping"]) as f:
            hm = json.load(f)

        assert len(hm["nodes"]) == 5, f"expected 5 node entries, got {len(hm['nodes'])}"
        assert len(hm["edges"]) == 4, f"expected 4 edge entries, got {len(hm['edges'])}"

        cold_offset = hm["meta"]["cold_offset"]
        for n in hm["nodes"]:
            assert n["warm_idx"] + cold_offset == n["cold_idx"], (
                f"cold_offset invariant broken for {n['ues_node_id']}: "
                f"{n['warm_idx']} + {cold_offset} != {n['cold_idx']}"
            )

        node_indices = {int(line.split(";")[0]) for line in node_csv[1:]}
        pipe_indices = {int(line.split(";")[0]) for line in pipe_csv[1:]}

        for n in hm["nodes"]:
            assert n["warm_idx"] in node_indices, f"warm_idx {n['warm_idx']} missing from CSV"
            assert n["cold_idx"] in node_indices, f"cold_idx {n['cold_idx']} missing from CSV"

        for e in hm["edges"]:
            assert e["warm_pipe_idx"] in pipe_indices, f"warm_pipe_idx {e['warm_pipe_idx']} missing"
            assert e["cold_pipe_idx"] in pipe_indices, f"cold_pipe_idx {e['cold_pipe_idx']} missing"

        print("heat_mapping.json invariants passed.")
    finally:
        shutil.rmtree(tmp)