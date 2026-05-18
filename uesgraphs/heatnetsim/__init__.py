"""HeatNetSim interop: export UESGraph to HNS CSVs and assign results back.

Three public entry points:

    apply_pipe_specs_to_graph(graph, csv_path, ...)
        Optional preprocessing: resolve edge attributes (diameter, dIns,
        wall_thickness) from a DN-keyed catalog CSV. Used when the GeoJSON
        only carries DN on the edges.

    uesgraph_to_hns_csv(graph, output_dir, ...)
        Write {prefix}_nodes.csv, _pipelines.csv, _substations.csv,
        _energybalancingunit.csv and a heat_mapping.json sidecar.

    assign_hns_results_to_uesgraph(graph, results, mapping_path, ...)
        Attach simulation timeseries (per-node temperature/pressure,
        per-edge mass flow / pressure drop) to graph nodes and edges
        in place.
"""

from uesgraphs.heatnetsim.export import uesgraph_to_hns_csv
from uesgraphs.heatnetsim.assign import assign_hns_results_to_uesgraph
from uesgraphs.heatnetsim.pipe_specs import apply_pipe_specs_to_graph

__all__ = [
    "uesgraph_to_hns_csv",
    "assign_hns_results_to_uesgraph",
    "apply_pipe_specs_to_graph",
]
