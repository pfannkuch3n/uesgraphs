"""Hydraulics-only pandapipes builder for pressure-drop (uncertainty) studies.

Lean alternative to :mod:`systemmodelheating`: no thermal model, no Excel
setup, no demand-to-massflow conversion. Consumers are `flow_control`
components fed with prescribed mass flows (e.g. measured volume flows), the
network is solved with ``pp.pipeflow(mode="hydraulics")`` per timestep.

Topology conventions match :class:`SystemModelHeating`: every uesgraph node
becomes two junctions (supply/return), every edge two pipes. Pipe geometry
reuses ``find_pipe_parameter`` (diameter snap to standard sizes) and
``estimate_xi`` (fittings heuristic).

The ``k_factor``/``d_factor``/``zeta_factor`` arguments scale roughness,
inner diameter and loss coefficient globally for parameter studies.
"""

import logging

import numpy as np
import pandas as pd
import pandapipes as pp

from .systemmodelheating import SystemModelHeating

# flow_control with exactly 0 kg/s can upset the hydraulic solver; clamp to a
# value that is hydraulically negligible for DN20+ pipes.
MDOT_MIN_KG_S = 1e-4

logger = logging.getLogger(__name__)


def build_hydraulic_net(
    uesgraph,
    p_flow_bar=2.3,
    plift_bar=0.3,
    t_supply_k=273.15 + 90.0,
    t_return_k=273.15 + 60.0,
    k_factor=1.0,
    d_factor=1.0,
    zeta_factor=1.0,
    default_k_mm=0.075,
):
    """Builds a hydraulics-only pandapipes net from a uesgraph.

    Parameters
    ----------
    uesgraph : uesgraphs.UESGraph
        Graph with one heating network. Edge attributes used: ``length`` [m],
        ``diameter`` [m] (snapped to standard sizes), optional ``roughness``
        [mm] and ``pipeID``. The supply is the building node with
        ``is_supply_heating``.
    p_flow_bar, plift_bar : float
        Pressure level and lift of the circulation pump at the supply node.
    t_supply_k, t_return_k : float
        Fluid temperatures of supply/return junctions — hydraulically only
        relevant through density/viscosity.
    k_factor, d_factor, zeta_factor : float
        Global multipliers on roughness, inner diameter and loss coefficient.
    default_k_mm : float
        Roughness fallback for edges without a ``roughness`` attribute.

    Returns
    -------
    net : pandapipes net
    consumers : dict
        ``{building_name: {"node", "fc", "supply_junction", "return_junction"}}``
    pipes : pandas.DataFrame
        One row per pipe pair with ``pipe_supply``/``pipe_return`` indices and
        the uesgraph edge ``(u, v)``.
    """
    net = pp.create_empty_network(fluid="water")

    junctions = {}
    supply_node = None
    building_nodes = {}
    for node, nd in uesgraph.nodes(data=True):
        pos = nd.get("position")
        xy = (pos.x, pos.y) if pos is not None else (0.0, 0.0)
        j_supply = pp.create_junction(
            net, pn_bar=p_flow_bar, tfluid_k=t_supply_k, name=f"{node}0",
            geodata=xy, position_x=xy[0], position_y=xy[1],
        )
        j_return = pp.create_junction(
            net, pn_bar=p_flow_bar - plift_bar, tfluid_k=t_return_k,
            name=f"{node}1", geodata=xy, position_x=xy[0], position_y=xy[1],
        )
        junctions[node] = (j_supply, j_return)
        if nd.get("node_type") == "building":
            if nd.get("is_supply_heating", False):
                if supply_node is not None:
                    raise ValueError(
                        "Mehrere Supply-Knoten gefunden — hydraulics.py "
                        "unterstützt aktuell genau einen."
                    )
                supply_node = node
            else:
                building_nodes[node] = str(nd.get("name", node))

    if supply_node is None:
        raise ValueError("Kein Knoten mit is_supply_heating im Graphen.")
    j_supply, j_return = junctions[supply_node]
    pp.create_circ_pump_const_pressure(
        net, return_junction=j_return, flow_junction=j_supply,
        p_flow_bar=p_flow_bar, plift_bar=plift_bar, t_flow_k=t_supply_k,
        type="auto", name="energy_hub",
    )

    pipe_rows = []
    for u, v, ed in uesgraph.edges(data=True):
        d_in, _ = SystemModelHeating.find_pipe_parameter(ed["diameter"] * 1000)
        d_in *= d_factor
        length = ed["length"]
        k_mm = ed.get("roughness", default_k_mm) * k_factor
        # Σζ aus gezählten Formteilen bevorzugen (uesgraphs.fittings schreibt
        # ``sum_zetas`` je Strang auf die Kante). Fällt zurück auf die alte
        # Längenheuristik, wenn der Graph noch keine Formteile kennt — analog
        # zur roughness-Zeile darüber.
        zeta = ed.get("sum_zetas", SystemModelHeating.estimate_xi(length)) * zeta_factor
        pair = {"u": u, "v": v, "length_m": length, "d_in_m": d_in}
        for idx, side in enumerate(("supply", "return")):
            pid = pp.create_pipe_from_parameters(
                net,
                from_junction=junctions[u][idx],
                to_junction=junctions[v][idx],
                length_km=length / 1000,
                inner_diameter_mm=d_in * 1000,
                k_mm=k_mm,
                loss_coefficient=zeta,
                sections=1,
                name=f"{side}_{ed.get('diameter')}_{ed.get('pipeID', '')}",
            )
            pair[f"pipe_{side}"] = pid
        pipe_rows.append(pair)

    consumers = {}
    for node, bname in building_nodes.items():
        js, jr = junctions[node]
        fc = pp.create_flow_control(
            net, from_junction=js, to_junction=jr,
            controlled_mdot_kg_per_s=MDOT_MIN_KG_S, name=bname,
        )
        consumers[bname] = {
            "node": node, "fc": fc,
            "supply_junction": js, "return_junction": jr,
        }

    return net, consumers, pd.DataFrame(pipe_rows)


def run_hydraulics_timeseries(net, consumers, mdot_df, mdot_min_kg_s=MDOT_MIN_KG_S):
    """Solves the hydraulic net for every row of ``mdot_df``.

    Parameters
    ----------
    net, consumers
        Output of :func:`build_hydraulic_net`.
    mdot_df : pandas.DataFrame
        Index = timestamps, columns = building names (keys of ``consumers``),
        values = mass flow in kg/s. NaN or missing columns fall back to
        ``mdot_min_kg_s``.

    Returns
    -------
    dict of DataFrames (index = mdot_df.index):
        ``dp_building_bar`` — supply-minus-return pressure at each building,
        ``pipe_dp_bar`` — pressure drop per pipe (columns = pipe index),
        ``pump`` — mdot [kg/s] and dp [bar] of the circulation pump.
    """
    missing = sorted(set(consumers) - set(mdot_df.columns))
    if missing:
        logger.warning(
            "Kein Massenstrom für %d Gebäude — laufen mit %g kg/s: %s",
            len(missing), mdot_min_kg_s, ", ".join(missing),
        )
    unknown = sorted(set(mdot_df.columns) - set(consumers))
    if unknown:
        logger.warning("Spalten ohne Gebäude im Netz (ignoriert): %s",
                       ", ".join(unknown))

    dp_rows, pipe_rows, pump_rows = [], [], []
    for ts, row in mdot_df.iterrows():
        for bname, info in consumers.items():
            val = row.get(bname, np.nan)
            if not np.isfinite(val):
                val = 0.0
            net.flow_control.at[info["fc"], "controlled_mdot_kg_per_s"] = max(
                float(val), mdot_min_kg_s
            )
        pp.pipeflow(net, mode="hydraulics", stop_condition="tol", iter=100,
                    tol_p=1e-7, tol_v=1e-7)

        p_junction = net.res_junction["p_bar"]
        dp_rows.append({
            bname: p_junction[info["supply_junction"]]
            - p_junction[info["return_junction"]]
            for bname, info in consumers.items()
        })
        pipe_rows.append(
            (net.res_pipe["p_from_bar"] - net.res_pipe["p_to_bar"]).to_dict()
        )
        pump = net.res_circ_pump_pressure.iloc[0]
        pump_rows.append({
            "mdot_kg_per_s": pump["mdot_from_kg_per_s"],
            "dp_bar": pump["p_to_bar"] - pump["p_from_bar"],
        })

    return {
        "dp_building_bar": pd.DataFrame(dp_rows, index=mdot_df.index),
        "pipe_dp_bar": pd.DataFrame(pipe_rows, index=mdot_df.index),
        "pump": pd.DataFrame(pump_rows, index=mdot_df.index),
    }
