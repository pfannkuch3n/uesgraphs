"""
GeoJSON Import to HeatNetSim CSV export (and result re-attachment)
==================================================================

End-to-end example for the uesgraphs.heatnetsim subpackage. Loads a small
district network from GeoJSON, resolves missing pipe properties from a
DN-keyed isoplus catalog, writes HeatNetSim-compatible CSV files, and then
demonstrates the result-assignment API with a synthetic results dict (since
HeatNetSim itself is not a uesgraphs dependency).

Workflow Overview:
-----------------
1. **GeoJSON Import**: Load network topology, buildings, and supply stations
2. **Pipe Specs**: Fill in diameter / d_ins / wall_thickness from the isoplus
   catalog based on each edge's DN
3. **HeatNetSim Export**: Write 4 CSVs plus heat_mapping.json sidecar
4. **External HeatNetSim Run**: not executed here; placeholder explains how
5. **Synthetic Results**: build a fake results dict for 3 timesteps to
   exercise the assign API
6. **Result Re-Attachment**: attach the timeseries to graph nodes and edges
7. **Inspection**: print a few attached series for verification

Example Directory Structure:
----------------------------
workspace/e19/
  - simple_district_graph.json   # saved UESGraph after GeoJSON import
  - heat_nodes.csv               # HNS node table
  - heat_pipelines.csv           # HNS pipe table (warm + cold strands)
  - heat_substations.csv         # HNS substation table
  - heat_energybalancingunit.csv # HNS EBU table
  - heat_mapping.json            # mapping for result re-attachment
  - e19_heatnetsim_<ts>.log      # run log

Notes:
-----
- HeatNetSim is not a uesgraphs dependency. STEP 4 only describes how to
  invoke it; the actual simulation must be run externally.
- Synthetic results in STEP 5 use a deterministic pattern so the printed
  values are predictable and the script can be used as a smoke test.
"""

import logging
import os
import sys

# Allow running this file directly from a development clone.
script_dir = os.path.dirname(os.path.abspath(__file__))
uesgraphs_root = os.path.dirname(os.path.dirname(script_dir))
if uesgraphs_root not in sys.path:
    sys.path.insert(0, uesgraphs_root)

from uesgraphs import UESGraph
from uesgraphs.utilities import set_up_file_logger
from uesgraphs.heatnetsim import (
    apply_pipe_specs_to_graph,
    assign_hns_results_to_uesgraph,
    uesgraph_to_hns_csv,
)


def workspace_example(name_workspace=None):
    """Create a local workspace with given name (copied from e1_readme_example).

    Parameters
    ----------
    name_workspace : str
        Name of the local workspace to be created

    Returns
    -------
    workspace : str
        Full path to the new workspace
    """
    this_dir = os.path.dirname(__file__)
    ues_dir = os.path.dirname(os.path.dirname(this_dir))
    workspace = os.path.join(ues_dir, "workspace")
    if not os.path.exists(workspace):
        os.mkdir(workspace)

    if name_workspace is not None:
        workspace = os.path.join(workspace, name_workspace)
        if not os.path.exists(workspace):
            os.mkdir(workspace)

    return workspace


def _build_synthetic_results(graph, n_steps=3):
    """Build a deterministic fake HNS results dict.

    Index layout matches build_index_map: warm indices 1..N, cold indices
    N+1..2N for N graph nodes. Pipes follow the same warm-first / cold-second
    convention.
    """
    n_nodes = len(graph.nodes())
    n_edges = len(graph.edges())
    n_nodes_hns = 2 * n_nodes
    n_pipes_hns = 2 * n_edges

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


def main():
    print("=" * 80)
    print("E19: GeoJSON to HeatNetSim CSV export and result re-attachment")
    print("=" * 80)

    # =========================================================================
    # STEP 1: Setup Workspace and Logger
    # =========================================================================
    print("\n STEP 1: Setting up workspace and logger...")

    workspace = workspace_example("e19")
    print(f"   Workspace: {workspace}")

    logger = set_up_file_logger(
        "e19_heatnetsim", log_dir=workspace, level=logging.INFO,
    )

    uesgraphs_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(uesgraphs_dir, "uesgraphs", "data", "examples")
    geojson_dir = os.path.join(data_dir, "e15_geojson")
    catalog_path = os.path.join(
        uesgraphs_dir, "uesgraphs", "data", "pipe_catalogs", "isoplus.csv",
    )

    network_geojson = os.path.join(geojson_dir, "network.geojson")
    buildings_geojson = os.path.join(geojson_dir, "buildings.geojson")
    supply_geojson = os.path.join(geojson_dir, "supply.geojson")

    print("   Paths configured.")

    # =========================================================================
    # STEP 2: Import District Network from GeoJSON
    # =========================================================================
    print("\n STEP 2: Importing district network from GeoJSON files...")

    graph = UESGraph()
    graph.from_geojson(
        network_path=network_geojson,
        buildings_path=buildings_geojson,
        supply_path=supply_geojson,
        name="simple_district",
        save_path=workspace,
        generate_visualizations=False,
    )

    n_nodes = len(graph.nodes())
    n_edges = len(graph.edges())
    print(f"   Loaded graph: {n_nodes} nodes, {n_edges} edges.")
    logger.info("Imported graph: %d nodes, %d edges", n_nodes, n_edges)

    # =========================================================================
    # STEP 3: Resolve Pipe Specs from Isoplus Catalog
    # =========================================================================
    print("\n STEP 3: Resolving pipe specs from isoplus catalog...")
    print(f"   Catalog: {catalog_path}")

    apply_pipe_specs_to_graph(graph, csv_path=catalog_path, logger=logger)

    sample_edge = next(iter(graph.edges(data=True)), None)
    if sample_edge is not None:
        u, v, data = sample_edge
        print(f"   Sample edge ({u}, {v}) attrs after lookup:")
        for key in ("DN", "diameter", "dIns", "wall_thickness"):
            if key in data:
                print(f"     {key}: {data[key]}")

    # =========================================================================
    # STEP 4: Export to HeatNetSim CSV files
    # =========================================================================
    print("\n STEP 4: Writing HeatNetSim CSV files...")

    try:
        paths = uesgraph_to_hns_csv(
            graph, workspace, file_prefix="heat", logger=logger,
        )
    except ValueError as exc:
        print(f"   ERROR: {exc}")
        print("   The example data should contain a supply node; check the GeoJSON.")
        raise

    print("   Files written:")
    for label, path in paths.items():
        print(f"     {label}: {os.path.basename(path)}")

    # =========================================================================
    # STEP 5: External HeatNetSim Run (placeholder)
    # =========================================================================
    print("\n STEP 5: HeatNetSim simulation (external)...")
    print("   This example does NOT run HeatNetSim itself.")
    print("   To run a real simulation, point HNS at the CSVs in:")
    print(f"     {workspace}")
    print("   Typical HNS call (outside uesgraphs):")
    print("     from heatnetsim import create_network_from_csv, run_simulation")
    print(f'     net = create_network_from_csv("{workspace}", prefix="heat")')
    print("     results = run_simulation(net, demand_profiles, ground_temps)")
    print("   For this example we use synthetic results in STEP 6.")

    # =========================================================================
    # STEP 6: Build Synthetic Results and Attach to Graph
    # =========================================================================
    print("\n STEP 6: Building synthetic results and attaching to graph...")

    results = _build_synthetic_results(graph, n_steps=3)
    logger.info(
        "Synthetic results: %d timesteps, %d nodes per step, %d pipes per step",
        len(results["nodal_temperature"]),
        len(results["nodal_temperature"][0]),
        len(results["pipeline_massflow"][0]),
    )

    assign_hns_results_to_uesgraph(
        graph,
        results=results,
        mapping_path=paths["mapping"],
        include_return=True,
        derive_edge_dp=True,
    )
    print("   Results attached. Each node has temperature/pressure series;")
    print("   each edge has m_flow / m_flow_return / dp.")

    # =========================================================================
    # STEP 7: Inspect Attached Series
    # =========================================================================
    print("\n STEP 7: Inspecting attached series for a few entities...")

    sample_node_ids = list(graph.nodes())[:2]
    for node_id in sample_node_ids:
        nd = graph.nodes[node_id]
        print(f"   Node {node_id}:")
        print(f"     temperature: {list(nd['temperature'].values)}")
        print(f"     pressure:    {list(nd['pressure'].values)}")
        print(f"     temperature_return: {list(nd['temperature_return'].values)}")

    sample_edge = next(iter(graph.edges()), None)
    if sample_edge is not None:
        u, v = sample_edge
        ed = graph.edges[(u, v)]
        print(f"   Edge ({u}, {v}):")
        print(f"     m_flow:        {list(ed['m_flow'].values)}")
        print(f"     m_flow_return: {list(ed['m_flow_return'].values)}")
        print(f"     dp:            {list(ed['dp'].values)}")

    print("\n" + "=" * 80)
    print(" E19 Example Completed Successfully!")
    print("=" * 80)
    print(f"\n Output Locations:")
    print(f"   Workspace:    {workspace}")
    print(f"   HNS CSVs:     heat_*.csv")
    print(f"   Sidecar JSON: heat_mapping.json")
    print(f"   Logfile:      e19_heatnetsim_<timestamp>.log")
    print("=" * 80)


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("UESGraphs Example 19: Complete GeoJSON to HeatNetSim Workflow")
    print("=" * 80)

    main()

    print("\n" + "=" * 80)
    print("Example script completed. Check your workspace for outputs!")
    print("=" * 80 + "\n")
