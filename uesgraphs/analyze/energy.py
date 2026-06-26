"""
Pump power vs. heat loss — energy comparison
============================================

Reduces the time-series attributes written by :func:`assign_data_pipeline`
(per-pipe heat loss, supply thermal power, demand power, hydraulic pump power) to
scalar energies, for a side-by-side comparison of *pumping* vs. *network heat
losses*.

This is a pure series->scalar reduction and does no plotting (the ``visuals``
layer stays "dumb"; series->scalar belongs here in ``analyze``). It mirrors the
pandapipes-path ``analysis_pp.thermal_loss_analysis`` /
``analysis_pp.pump_power_analysis`` but operates on the in-memory graph produced
by the AixLib/Dymola mapping path instead of JSON files on disk.

Heat loss is taken from the pipes only:

* **per pipe** — sum of the pipes' own ``heatPort.Q_flow`` over ALL pipes. A
  supply-side graph carries both the supply-pipe loss (``edge["Q_loss"]``) and
  the return-pipe loss (``edge["Q_loss_return"]``), so the figure covers VL+RL
  from one graph (a separately mapped ``graph_return`` is summed too). This is
  THE heat-loss figure.

Alongside it (NOT subtracted, NOT relabelled as loss) the two thermal totals are
reported as separate context quantities:

* ``supply_heat`` — Σ of the supply stations' thermal power ``heat_power_supply``.
* ``demand_heat`` — Σ of the buildings' ``heat_power_prescribed`` (the prescribed
  load that drives the simulation; left untouched, never modified here).

Pump energy is the ideal hydraulic pumping power written by
``assign_supply_values`` (``supply["pump_power_hydraulic"]``), or a native pump
variable (``supply["pump_power"]``) for real-pump stations.
"""

import numpy as np
import pandas as pd

from uesgraphs.utilities import set_up_terminal_logger


def _sum_over(attr_dicts, key, t_min=None):
    """Element-wise sum of a time-series attribute over graph elements.

    Returns ``(total_series_or_None, n_contributing)``. Elements missing the key
    (or holding a scalar rather than a series) are skipped. With *t_min* set,
    samples before it are dropped first (warm-up / initialisation transients).
    """
    total = None
    n = 0
    for d in attr_dicts:
        s = d.get(key)
        if s is None or not hasattr(s, "__len__"):
            continue
        s = s if isinstance(s, pd.Series) else pd.Series(s)
        if t_min is not None:
            try:
                s = s[s.index >= t_min]
            except TypeError:
                pass  # non-comparable index -> no trim
        if len(s) == 0:
            continue
        total = s.copy() if total is None else total.add(s, fill_value=0.0)
        n += 1
    return total, n


def _energy_peak(series, timestep_hours):
    """Reduce a power series [W] to (energy [kWh], peak [kW]).

    Returns (nan, nan) for a missing series. Energy = Σ power * dt / 1000.
    """
    if series is None:
        return float("nan"), float("nan")
    arr = np.asarray(series, dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    energy_kWh = float(np.nansum(arr) * timestep_hours / 1e3)
    peak_kW = float(np.nanmax(arr) / 1e3)
    return energy_kWh, peak_kW


def pump_vs_loss(graph, graph_return=None, *, timestep_hours=1.0, t_min=None,
                 loss_key="Q_loss", supply_power_key="heat_power_supply",
                 demand_key="heat_power_prescribed",
                 pump_keys=("pump_power_hydraulic", "pump_power"),
                 logger=None):
    """Compare pumping energy against network heat losses for a mapped graph.

    The graph must already carry the time-series attributes written by
    :func:`uesgraphs.analyze.assign_data_pipeline` with ``with_heat_loss`` and
    ``with_pump_power`` enabled.

    Args:
        graph: mapped supply-side UESGraph (edges carry ``Q_loss``; the supply
            node carries ``heat_power_supply`` / ``pump_power_hydraulic``;
            building nodes carry ``heat_power_prescribed``).
        graph_return: optional mapped return-side UESGraph; if given, its pipe
            losses are added to the per-pipe figure (full-system loss).
        timestep_hours: sample spacing in hours (W * h -> Wh -> /1000 = kWh).
        t_min: optional timestamp; samples before it are dropped from every series
            before reducing (warm-up / initialisation transients).
        loss_key / supply_power_key / demand_key / pump_keys: attribute names;
            ``pump_keys`` is tried in order (hydraulic first, then a native var).
        logger: optional logger.

    Returns:
        dict of scalars (all energies in kWh, peaks in kW):
            loss_per_pipe_kWh, loss_per_pipe_peak_kW   # THE heat loss (pipes)
            supply_heat_kWh,   supply_heat_peak_kW     # Σ supply station power
            demand_heat_kWh,   demand_heat_peak_kW     # Σ prescribed building load
            pump_kWh,          pump_peak_kW            # hydraulic pump energy
            ratio_pump_per_loss                        # pump / per-pipe loss
            n_pipes, n_supply, n_demand, includes_return, pump_attr,
            timestep_hours
        supply_heat and demand_heat are reported side by side as context; they are
        deliberately NOT subtracted into a "loss" figure. Missing inputs yield NaN
        for the affected metric rather than raising.
    """
    if logger is None:
        logger = set_up_terminal_logger(f"{__name__}.pump_vs_loss")

    supply_nodes = [n for n in graph.nodelist_building
                    if graph.nodes[n].get("is_supply_heating")]
    demand_nodes = [n for n in graph.nodelist_building
                    if not graph.nodes[n].get("is_supply_heating")]

    # --- heat loss: per pipe, BOTH lines ---------------------------------
    # A supply-side graph carries the supply-pipe loss ("Q_loss") AND the
    # return-pipe loss ("Q_loss_return"), so summing both keys gives the full
    # VL+RL pipe loss from a single graph. A separately mapped return graph
    # (graph_return) is also summed, for callers that keep the sides apart.
    loss_keys = (loss_key, "Q_loss_return")
    loss_series = None
    n_pipes = 0
    ret_count = 0
    edge_graphs = [(graph, False)]
    if graph_return is not None:
        edge_graphs.append((graph_return, True))
    for g, is_return_graph in edge_graphs:
        for k in loss_keys:
            s, n = _sum_over((g.edges[e] for e in g.edges), k, t_min)
            if s is None:
                continue
            loss_series = s if loss_series is None else loss_series.add(s, fill_value=0.0)
            n_pipes += n
            if is_return_graph or k == "Q_loss_return":
                ret_count += n
    includes_return = ret_count > 0

    # --- thermal totals (context only; NOT subtracted into a loss) -------
    supply_power, n_supply = _sum_over(
        (graph.nodes[n] for n in supply_nodes), supply_power_key, t_min)
    demand_power, n_demand = _sum_over(
        (graph.nodes[n] for n in demand_nodes), demand_key, t_min)

    # --- pump (hydraulic) power ------------------------------------------
    pump_series, pump_attr = None, None
    for key in pump_keys:
        pump_series, _ = _sum_over((graph.nodes[n] for n in supply_nodes), key, t_min)
        if pump_series is not None:
            pump_attr = key
            break

    loss_pipe_kWh, loss_pipe_peak = _energy_peak(loss_series, timestep_hours)
    supply_kWh, supply_peak = _energy_peak(supply_power, timestep_hours)
    demand_kWh, demand_peak = _energy_peak(demand_power, timestep_hours)
    pump_kWh, pump_peak = _energy_peak(pump_series, timestep_hours)

    def _ratio(num, den):
        if not np.isfinite(num) or not np.isfinite(den) or den == 0:
            return float("nan")
        return num / den

    result = {
        "loss_per_pipe_kWh": loss_pipe_kWh,
        "loss_per_pipe_peak_kW": loss_pipe_peak,
        "supply_heat_kWh": supply_kWh,
        "supply_heat_peak_kW": supply_peak,
        "demand_heat_kWh": demand_kWh,
        "demand_heat_peak_kW": demand_peak,
        "pump_kWh": pump_kWh,
        "pump_peak_kW": pump_peak,
        "ratio_pump_per_loss": _ratio(pump_kWh, loss_pipe_kWh),
        "n_pipes": n_pipes,
        "n_supply": n_supply,
        "n_demand": n_demand,
        "includes_return": includes_return,
        "pump_attr": pump_attr,
        "timestep_hours": timestep_hours,
    }

    logger.info(
        "pump_vs_loss: loss(per-pipe%s)=%.1f kWh | supply=%.1f kWh, "
        "demand=%.1f kWh | pump=%.1f kWh (attr=%s)" % (
            "+R" if includes_return else "", loss_pipe_kWh, supply_kWh,
            demand_kWh, pump_kWh, pump_attr))
    return result
