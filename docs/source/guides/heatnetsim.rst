HeatNetSim Interop
==================

The :mod:`uesgraphs.heatnetsim` subpackage provides a round-trip between a
:class:`uesgraphs.UESGraph` and HeatNetSim (HNS) simulations: export the graph
to HNS-compatible CSV files, run HNS externally, then attach the simulation
results back onto the graph for downstream analysis and visualisation.

Overview
--------

::

    graph  --uesgraph_to_hns_csv-->   heat_*.csv + heat_mapping.json
                                            |
                                            v
                                      run HeatNetSim (externally)
                                            |
                                            v
    graph  <--assign_hns_results_to_uesgraph-- results dict / CSV

The export emits four CSV files (``_nodes``, ``_pipelines``,
``_substations``, ``_energybalancingunit``) and a sidecar
``heat_mapping.json`` that records which HNS index corresponds to which
UESGraph node/edge.

Topology
--------

Dual-strand 4G/2G heat-only. For each UESGraph node, two HNS nodes are
emitted — a warm twin (index ``N``) and a cold twin (index ``N +
total_nodes``). The cold strand is the return line; HNS requires it for a
closed hydraulic loop.

Conventions:

* **EBU**: inlet = cold (return), outlet = warm (supply); both endpoints are
  reference nodes carrying ``pressure_pa``.
* **Substation**: inlet = warm, outlet = cold.
* **Cold pipe**: flow direction reversed relative to its warm twin.

Quickstart
----------

Optional preprocessing — resolve pipe attributes from a DN catalog when the
GeoJSON only carries DN on the edges:

.. code-block:: python

    from uesgraphs.heatnetsim import apply_pipe_specs_to_graph

    apply_pipe_specs_to_graph(
        graph,
        csv_path="uesgraphs/data/pipe_catalogs/isoplus.csv",
    )
    # Sets edge attributes: diameter, dIns, wall_thickness. Existing
    # attributes are preserved by default; pass overwrite=True to replace.

Export:

.. code-block:: python

    from uesgraphs.heatnetsim import uesgraph_to_hns_csv

    paths = uesgraph_to_hns_csv(graph, "out/", file_prefix="heat")
    # paths = {"nodes": ..., "pipelines": ..., "substations": ...,
    #          "ebu": ..., "mapping": ...}

Assign results back:

.. code-block:: python

    from uesgraphs.heatnetsim import assign_hns_results_to_uesgraph

    assign_hns_results_to_uesgraph(
        graph,
        results=results_dict_or_csv_path,
        mapping_path=paths["mapping"],
        include_return=True,
        derive_edge_dp=True,
    )

    # graph.nodes[id]["temperature"], ["pressure"]      pd.Series
    # graph.edges[u,v]["m_flow"], ["dp"]                pd.Series
    # With include_return=True, additionally:
    #   *_return on nodes, m_flow_return on edges

API
---

.. autofunction:: uesgraphs.heatnetsim.uesgraph_to_hns_csv

.. autofunction:: uesgraphs.heatnetsim.assign_hns_results_to_uesgraph
