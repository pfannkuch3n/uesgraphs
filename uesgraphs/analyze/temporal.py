"""
Temporal reduction of time-series graph attributes
===================================================

The data-assignment pipeline (``analyze.data_handling``) writes time series
(pandas Series) onto graph edges/nodes - e.g. ``m_flow`` per edge or
``heat_power_prescribed`` per building node. The plotting layer
(``uesgraphs.visuals``) is deliberately kept "dumb": it can only render scalar
attributes, not Series (see example e13).

This module bridges the two by reducing time-series attributes to a single
representative scalar, so a coincident network snapshot - e.g. mass flow AND
building power *at the same instant* - can be plotted in one figure.

Keeping this here (and out of both ``data_handling`` and ``visuals``) preserves
the separation: data assignment writes the series, ``temporal`` reduces it,
``visuals`` renders the resulting scalar.
"""

from uesgraphs.utilities import set_up_terminal_logger


def snapshot_at(graph, timestamp, edge_keys=None, node_keys=None,
                suffix="_t", logger=None):
    """Materialize scalar snapshot attributes at a single timestamp for plotting.

    For each key in ``edge_keys`` writes ``graph.edges[e][key + suffix] =
    series.loc[timestamp]``; for each key in ``node_keys`` writes
    ``graph.nodes[n][key + suffix] = series.loc[timestamp]`` - but only where the
    source attribute exists and is a time series (entries that are missing or
    already scalar are skipped).

    Args:
        graph: uesgraphs object carrying time-series attributes.
        timestamp: a value usable with ``Series.loc`` (e.g. a ``pandas.Timestamp``
            / ``datetime``, or a string like ``"2024-01-13 14:00"``). Must match
            the series index exactly.
        edge_keys: list of edge attribute names to snapshot (e.g. ``["m_flow"]``).
        node_keys: list of node attribute names to snapshot
            (e.g. ``["heat_power_prescribed"]``).
        suffix: appended to each key to form the scalar attribute name
            (default ``"_t"``, so ``m_flow`` -> ``m_flow_t``).
        logger: optional logger.

    Returns:
        The ``timestamp`` (unchanged), handy as the ``show_network(timestamp=...)``
        label.

    Example
    -------
    >>> from uesgraphs.analyze import snapshot_at
    >>> t = snapshot_at(graph, "2023-04-13 14:00",
    ...                 edge_keys=["m_flow"],
    ...                 node_keys=["heat_power_prescribed"])
    >>> vis.show_network(generic_extensive_size="m_flow_t",
    ...                  generic_node_size="heat_power_prescribed_t",
    ...                  timestamp=str(t))
    """
    if logger is None:
        logger = set_up_terminal_logger(f"{__name__}.snapshot_at")

    edge_keys = edge_keys or []
    node_keys = node_keys or []

    for key in edge_keys:
        count = 0
        for edge in graph.edges:
            series = graph.edges[edge].get(key)
            if series is None or not hasattr(series, "loc"):
                continue
            graph.edges[edge][key + suffix] = series.loc[timestamp]
            count += 1
        logger.info(f"Snapshot edge '{key}' -> '{key + suffix}' at {timestamp} "
                    f"for {count} edges")

    for key in node_keys:
        count = 0
        for node in graph.nodes:
            series = graph.nodes[node].get(key)
            if series is None or not hasattr(series, "loc"):
                continue
            graph.nodes[node][key + suffix] = series.loc[timestamp]
            count += 1
        logger.info(f"Snapshot node '{key}' -> '{key + suffix}' at {timestamp} "
                    f"for {count} nodes")

    return timestamp


def value_range(graph, attribute, level="edge", timestamps=None):
    """Global ``(min, max)`` of a time-series (or scalar) attribute.

    Computes the extremes of ``attribute`` across every edge (``level="edge"``)
    or node (``level="node"``) and across all timestamps. This pins a consistent
    color/size scale for ``Visuals.show_network`` (``minmax`` / ``ring_minmax``),
    so a batch of per-timestamp snapshots shares one legend and the resulting
    image series is comparable frame to frame.

    Series attributes are reduced over their time index; scalar attributes count
    as a single value. Elements where the attribute is missing or non-numeric are
    skipped, and ``NaN`` values are ignored.

    Args:
        graph: uesgraphs object carrying the attribute.
        attribute: attribute name to scan.
        level: ``"edge"`` or ``"node"``.
        timestamps: optional iterable of index values restricting the time range
            (e.g. a sub-window of the simulation). Default: the full series index.

    Returns:
        ``(min, max)`` as floats, or ``(None, None)`` if the attribute is absent
        or non-numeric everywhere.

    Example
    -------
    >>> lo, hi = value_range(graph, "m_flow", level="edge")
    >>> vis.show_network(generic_extensive_size="m_flow_t", minmax=[lo, hi])
    """
    import numpy as np

    if level == "edge":
        values = (graph.edges[e].get(attribute) for e in graph.edges)
    elif level == "node":
        values = (graph.nodes[n].get(attribute) for n in graph.nodes)
    else:
        raise ValueError(f"level must be 'edge' or 'node', got {level!r}")

    ts_filter = None if timestamps is None else list(timestamps)

    lo = hi = None
    for val in values:
        if val is None:
            continue
        if hasattr(val, "index"):  # pandas Series
            series = val if ts_filter is None else val[val.index.isin(ts_filter)]
            if len(series) == 0 or series.isna().all():
                continue  # nothing to contribute (e.g. an unmeasured/NaN-only edge)
            vmin = float(np.nanmin(series.values))
            vmax = float(np.nanmax(series.values))
        else:
            try:
                vmin = vmax = float(val)
            except (TypeError, ValueError):
                continue
        if np.isnan(vmin) or np.isnan(vmax):
            continue
        lo = vmin if lo is None else min(lo, vmin)
        hi = vmax if hi is None else max(hi, vmax)

    return lo, hi


def abs_attr(graph, attribute, level="edge", suffix="_abs", logger=None):
    """Write the absolute value of an attribute as ``<attribute><suffix>``.

    For each edge (``level="edge"``) or node (``level="node"``) carrying
    ``attribute``, store ``|attribute|`` under ``attribute + suffix``: a pandas
    Series is reduced element-wise (``Series.abs()``, index preserved), a plain
    numeric scalar via ``abs(float(...))``; missing or non-numeric entries are
    skipped.

    This belongs here (not in ``visuals`` or any renderer) for the same reason as
    :func:`snapshot_at` / :func:`value_range`: transforming graph attributes is the
    analyze layer's job, the renderer only reads scalars. It exists because some
    quantities flip sign with the modelled flow direction (``m_flow`` is positive
    or negative depending on a pipe's orientation), so a topology map or a
    comparison wants to colour by *magnitude*. Pair it with :func:`snapshot_at`
    (snapshot the ``_abs`` series) and :func:`value_range` (range over ``_abs``).

    Args:
        graph: uesgraphs object carrying the attribute.
        attribute: source attribute name (e.g. ``"m_flow"``).
        level: ``"edge"`` or ``"node"``.
        suffix: appended to form the target name (default ``"_abs"`` ->
            ``m_flow_abs``).
        logger: optional logger.

    Returns:
        The new attribute name (``attribute + suffix``).
    """
    if logger is None:
        logger = set_up_terminal_logger(f"{__name__}.abs_attr")

    if level == "edge":
        elements = list(graph.edges)
        bag_of = lambda el: graph.edges[el]
    elif level == "node":
        elements = list(graph.nodes)
        bag_of = lambda el: graph.nodes[el]
    else:
        raise ValueError(f"level must be 'edge' or 'node', got {level!r}")

    target = attribute + suffix
    count = 0
    for el in elements:
        bag = bag_of(el)
        val = bag.get(attribute)
        if val is None:
            continue
        if hasattr(val, "index") and hasattr(val, "loc"):  # pandas Series
            bag[target] = val.abs()
            count += 1
        else:
            try:
                bag[target] = abs(float(val))
                count += 1
            except (TypeError, ValueError):
                continue
    logger.info(f"abs '{attribute}' -> '{target}' for {count} {level}s")
    return target
