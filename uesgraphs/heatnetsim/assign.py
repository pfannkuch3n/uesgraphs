"""Attach HNS simulation results to a UESGraph in place.

Attribute names mirror the Modelica-mask convention used in uesgraphs:

    graph.nodes[id]["temperature"]   pd.Series  (K)
    graph.nodes[id]["pressure"]      pd.Series  (Pa)
    graph.edges[u,v]["m_flow"]       pd.Series  (kg/s)
    graph.edges[u,v]["dp"]           pd.Series  (Pa, always >= 0)

With ``include_return=True``, the cold-side values are additionally stored as:

    graph.nodes[id]["temperature_return"]
    graph.nodes[id]["pressure_return"]
    graph.edges[u,v]["m_flow_return"]
"""

import pandas as pd

from uesgraphs.heatnetsim.io import build_time_index, load_mapping, load_results


def _extract_series(results: dict, key: str, hns_idx: int, index) -> pd.Series:
    """Pull a per-entity timeseries for HNS index *hns_idx* (1-based).

    ``results[key]`` is a list of T lists, each of length N_entities.
    Position ``hns_idx - 1`` within each inner list gives the value at that
    timestep.
    """
    values = [timestep_list[hns_idx - 1] for timestep_list in results[key]]
    return pd.Series(values, index=index, name=key)


def _resolve_edge(graph, u, v):
    """Return the canonical (u, v) key present in graph.edges, or raise."""
    if graph.has_edge(u, v):
        return (u, v)
    if graph.has_edge(v, u):
        return (v, u)
    raise KeyError(
        f"Edge ({u}, {v}) not found in graph (tried both orientations)."
    )


def assign_hns_results_to_uesgraph(
    graph,
    results,
    mapping_path,
    side: str = "warm",
    include_return: bool = False,
    derive_edge_dp: bool = True,
    start_date=None,
    time_interval=None,
):
    """Attach HNS simulation timeseries to *graph* in place and return it.

    Parameters
    ----------
    graph : uesgraphs.UESGraph  (or any networkx.Graph duck-type)
    results : dict or str/Path
        Either the raw results dict from ``save_time_series_results``, or a
        path to the CSV written by ``runner.py``.
    mapping_path : str or Path
        Path to ``heat_mapping.json`` produced by :func:`uesgraph_to_hns_csv`.
    side : {"warm", "cold"}
        Which pipeline strand to treat as the primary assignment.
        "warm" = supply (default), "cold" = return.
    include_return : bool
        If True, also attach the opposite strand under ``*_return`` keys.
    derive_edge_dp : bool
        If True, compute ``dp = |p_u - p_v|`` from nodal pressures and store
        it on each edge.
    start_date : datetime-like, optional
        If given together with *time_interval*, build a DatetimeIndex.
    time_interval : str, optional
        Pandas frequency string, e.g. ``"15min"``.

    Returns
    -------
    graph
        The same object, mutated in place.

    Raises
    ------
    KeyError
        If mapping references a node/edge not present in the graph.
    ValueError
        If timestep counts are inconsistent across result columns.
    """
    if side not in ("warm", "cold"):
        raise ValueError(f"side must be 'warm' or 'cold', got {side!r}")

    results_dict = load_results(results)
    mapping = load_mapping(mapping_path)

    list_keys = [
        k for k, v in results_dict.items()
        if isinstance(v, list) and v and isinstance(v[0], list)
    ]
    n_steps_per_key = {k: len(results_dict[k]) for k in list_keys}
    unique_counts = set(n_steps_per_key.values())
    if len(unique_counts) > 1:
        raise ValueError(
            f"Inconsistent timestep counts across result columns: {n_steps_per_key}"
        )
    n_timesteps = unique_counts.pop() if unique_counts else 0
    time_index = build_time_index(n_timesteps, start_date, time_interval)

    for entry in mapping["nodes"]:
        node_id = entry["ues_node_id"]
        if node_id not in graph.nodes:
            raise KeyError(
                f"UESGraph node {node_id!r} from mapping not found in graph."
            )

        primary_idx = entry["warm_idx"] if side == "warm" else entry["cold_idx"]
        return_idx  = entry["cold_idx"] if side == "warm" else entry["warm_idx"]

        graph.nodes[node_id]["temperature"] = _extract_series(
            results_dict, "nodal_temperature", primary_idx, time_index
        )
        graph.nodes[node_id]["pressure"] = _extract_series(
            results_dict, "nodal_pressure", primary_idx, time_index
        )

        if include_return:
            graph.nodes[node_id]["temperature_return"] = _extract_series(
                results_dict, "nodal_temperature", return_idx, time_index
            )
            graph.nodes[node_id]["pressure_return"] = _extract_series(
                results_dict, "nodal_pressure", return_idx, time_index
            )

    for entry in mapping["edges"]:
        u, v = entry["ues_u"], entry["ues_v"]
        edge_key = _resolve_edge(graph, u, v)

        primary_pipe = entry["warm_pipe_idx"] if side == "warm" else entry["cold_pipe_idx"]
        return_pipe  = entry["cold_pipe_idx"] if side == "warm" else entry["warm_pipe_idx"]

        graph.edges[edge_key]["m_flow"] = _extract_series(
            results_dict, "pipeline_massflow", primary_pipe, time_index
        )

        if include_return:
            graph.edges[edge_key]["m_flow_return"] = _extract_series(
                results_dict, "pipeline_massflow", return_pipe, time_index
            )

        if derive_edge_dp:
            p_u = graph.nodes[u]["pressure"]
            p_v = graph.nodes[v]["pressure"]
            graph.edges[edge_key]["dp"] = (p_u - p_v).abs()

    return graph
