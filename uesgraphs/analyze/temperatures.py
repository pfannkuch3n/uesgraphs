"""
Return-temperature metrics for district heating graphs
======================================================

Sibling of :mod:`uesgraphs.analyze.temporal`: where ``temporal`` reduces a
single time-series attribute to a representative scalar for plotting, this
module computes a *network-level* temperature KPI that couples each
substation's return-temperature series with its connection-pipe mass-flow
series.

It currently provides :func:`return_temp_reduction_potential`, the
mass-flow-weighted, time-averaged return temperature reduction potential per
substation after Oltmanns (2020), Eq. 13.

Like ``temporal`` it only reads Series already attached to the graph (no IO, no
rendering) and deliberately stays out of ``data_handling`` / ``visuals``.
"""

import pandas as pd

from uesgraphs.utilities import set_up_terminal_logger

_K0 = 273.15  # Kelvin <-> Celsius offset


def _substations(graph):
    """Non-supply building nodes — the substations / heat transfer stations.

    Prefers ``graph.nodelist_building``; falls back to ``node_type ==
    "building"``. Supply buildings (``is_supply_heating``) are excluded.
    """
    nodes = getattr(graph, "nodelist_building", None)
    if not nodes:
        nodes = [n for n, d in graph.nodes(data=True)
                 if d.get("node_type") == "building"]
    return [n for n in nodes if not graph.nodes[n].get("is_supply_heating")]


def _connection_edge(graph, node):
    """The ``(u, v)`` key of a leaf substation's single connection pipe, else None."""
    incident = list(graph.edges(node))
    if len(incident) != 1:
        return None
    return incident[0]


def return_temp_reduction_potential(graph, target_temp, *,
                                    t_return_attr="temperature_return",
                                    m_flow_attr="m_flow",
                                    target_in_celsius=True,
                                    clamp=True,
                                    logger=None):
    """Mass-flow-weighted, time-averaged return temperature reduction potential.

    Implements the return temperature reduction potential after Oltmanns (2020,
    Eq. 13). For each substation (non-supply building) *j* it quantifies, in
    Kelvin, how much the network's mass-flow-mixed average return temperature
    would drop if *that* substation's return temperature were lowered to
    ``target_temp`` — time-averaged over the run::

        dT_R,j = mean_i  max(0, T_R,i,j - T_target) * M_i,j / sum_j' M_i,j'

    The mass-flow share ``M_i,j / sum_j' M_i,j'`` is exactly the weight with
    which substation *j*'s return temperature enters the mixed network return
    temperature, so the per-substation value is its contribution to the network
    return-temperature reduction. The sum over all substations
    (``attrs["network_total_K"]``) is the total reduction achievable if every
    substation is brought to target simultaneously — successive lowering yields
    less, because a colder return reduces a substation's mass-flow share.

    Reads the return-temperature Series (Kelvin) on each substation node and the
    mass flow on its single connection pipe (``abs`` of the edge series, since
    ``m_flow`` flips sign with pipe orientation). Within one simulation run all
    Series share a common index; ``pandas`` aligns them on the index defensively.

    Time steps with zero total network mass flow carry no return-temperature
    mixing and are dropped from the average (so the mean is over the ``mean_i``
    of the non-zero-flow steps, see ``attrs["n_steps_used"]``, rather than over
    all ``n_ts`` steps).

    Args:
        graph: uesgraphs object carrying the time-series attributes.
        target_temp: target return temperature. Interpreted in degrees Celsius
            unless ``target_in_celsius=False`` (then Kelvin, matching the stored
            Series).
        t_return_attr: node attribute holding the return-temperature Series in
            Kelvin (default ``"temperature_return"``).
        m_flow_attr: connection-edge attribute holding the mass-flow Series in
            kg/s (default ``"m_flow"``).
        target_in_celsius: if True (default), ``target_temp`` is given in deg C
            and converted to Kelvin internally to match the stored Series.
        clamp: if True (default), negative deficits ``(T_R - T_target) < 0`` are
            clamped to 0 — a substation already returning below target has no
            reduction potential. If False, the literal signed difference of
            Eq. 13 is kept (the net network effect of forcing every substation
            exactly to target, including raising the cold ones).
        logger: optional logger.

    Returns:
        pandas.DataFrame indexed by substation name, sorted by ``potential_K``
        descending, with columns:

          - ``potential_K``    return temp reduction potential dT_R,j in K
            (equivalently a degC difference)
          - ``mean_return_C``  time-mean return temperature in degC (context)
          - ``mean_mflow_kgs`` time-mean |mass flow| in kg/s (context)
          - ``mean_share``     time-mean mass-flow share of the network (-)

        ``df.attrs`` carries ``network_total_K`` (sum of ``potential_K``),
        ``n_steps_used`` (time steps with non-zero network flow that entered the
        average), ``n_substations`` and ``n_skipped`` (substations dropped for a
        missing return-temperature or mass-flow series).

    Raises:
        ValueError: if no substation carries a mass-flow series on its
            connection pipe (the network weighting is then undefined).

    Example
    -------
    >>> from uesgraphs.analyze import return_temp_reduction_potential
    >>> df = return_temp_reduction_potential(graph, target_temp=55)
    >>> df.head()
    >>> df.attrs["network_total_K"]   # total achievable network return-temp drop
    """
    if logger is None:
        logger = set_up_terminal_logger(
            f"{__name__}.return_temp_reduction_potential")

    target_K = target_temp + _K0 if target_in_celsius else target_temp

    t_cols = {}   # substations with a return-temp series (numerator)
    m_cols = {}   # substations with a mass-flow series (denominator)
    skipped = 0
    for node in _substations(graph):
        name = graph.nodes[node].get("name", node)
        edge = _connection_edge(graph, node)
        t_ser = graph.nodes[node].get(t_return_attr)
        m_ser = graph.edges[edge].get(m_flow_attr) if edge is not None else None
        has_t = hasattr(t_ser, "index")
        has_m = hasattr(m_ser, "index")
        if has_m:
            m_cols[name] = m_ser.abs()  # magnitude: m_flow sign flips with direction
        if has_t and has_m:
            t_cols[name] = t_ser
        else:
            skipped += 1

    if not m_cols:
        raise ValueError(
            f"no substation carries a '{m_flow_attr}' series on its connection "
            f"pipe — cannot compute the network mass-flow weighting")

    # Denominator of Eq. 13: total network mass flow per time step over ALL
    # substation connection flows (even those without a known return temp).
    Mall = pd.concat(m_cols, axis=1)
    total_flow = Mall.sum(axis=1, min_count=1)
    total_flow = total_flow.where(total_flow > 0)  # 0-flow steps -> NaN (dropped)

    # Numerator: per-substation deficit, only where the return temp is known.
    Tframe = pd.concat(t_cols, axis=1)
    Mframe = Mall[list(t_cols)]
    deficit = Tframe - target_K
    if clamp:
        deficit = deficit.clip(lower=0)
    share = Mframe.div(total_flow, axis=0)
    contrib = deficit * share

    potential = contrib.mean(axis=0)            # nanmean over non-zero-flow steps
    df = pd.DataFrame({
        "potential_K": potential,
        "mean_return_C": Tframe.mean(axis=0) - _K0,
        "mean_mflow_kgs": Mframe.mean(axis=0),
        "mean_share": share.mean(axis=0),
    }).sort_values("potential_K", ascending=False)

    df.attrs["network_total_K"] = float(potential.sum())
    df.attrs["n_steps_used"] = int(total_flow.notna().sum())
    df.attrs["n_substations"] = int(len(t_cols))
    df.attrs["n_skipped"] = int(skipped)

    logger.info(
        f"return_temp_reduction_potential: {len(t_cols)} substations, "
        f"{skipped} skipped (missing series), "
        f"{df.attrs['n_steps_used']} time steps used, network total "
        f"{df.attrs['network_total_K']:.3f} K "
        f"(target {target_K - _K0:.1f} degC, clamp={clamp})")
    return df
