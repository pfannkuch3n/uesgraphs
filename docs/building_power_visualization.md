# Building thermal-power visualization

*Added 2026-05-29.* Plots **building (demand) thermal power** on a uesgraph network —
as a secondary overlay next to the existing edge/node painters — for a single
coincident timestamp (e.g. "mass flow AND building demand at 2023-04-13 14:00").

Design principle: the power is assigned as a real **node attribute at the source**
(data pipeline); `visuals.py` stays "dumb" and only renders **scalars**; the
time-series → scalar reduction lives separately in `analyze`.

---

## 1. Data side — `analyze/data_handling/data_handling.py`

Building power is keyed by the **building/demand node name** (`node["name"]`),
i.e. `networkModel.<name>Q_flow_input` — *not* by pipe code like the existing
`pressure`/`temperature` node masks. So it gets its own path (the per-pipe
`assign_node_values` does not fit).

- **`AIXLIB_MASKS[...]["building"]`** (new section in `2.1.0` and `2.0.0`):
  - `heat_power_prescribed = "networkModel.{name}Q_flow_input"` — the **prescribed**
    demand setpoint [W] (the input fed to the AixLib demand model).
  - `_realized` (optional): `senT_supply.T`, `senT_return.T`, `senT_supply.m_flow`
    on `demand{name}` — to reconstruct delivered power (no prebuilt realized-Q var
    exists in `VarTSupplyDp`).
- **`build_filter_list_demand(graph, MASK, realized=False)`** — column list per
  building name (supply buildings skipped).
- **`assign_demand_power(graph, df, MASK, realized=False, cp=4184)`** — writes
  `nodes[n]["heat_power_prescribed"]` (pd.Series, W); with `realized=True` also
  `nodes[n]["heat_power_realized"] = m_flow·cp·(T_supply − T_return)`.
- **`assign_data_pipeline(..., with_demand_power=True, demand_power_realized=False)`** —
  demand columns are intersected with the file's available columns (missing →
  *warning*, not a hard error), then assigned after `assign_edge_data`.

## 2. Temporal reduction — `analyze/temporal.py` (new)

- **`snapshot_at(graph, timestamp, edge_keys=None, node_keys=None, suffix="_t")`** —
  reduces time-series attributes to a scalar via `series.loc[timestamp]` for edges
  **and** nodes (e.g. `m_flow` → `m_flow_t`), returns the timestamp (handy as the
  `show_network(timestamp=...)` label). Exported from `uesgraphs.analyze`.
- **`value_range(graph, attribute, level="edge", timestamps=None)`** *(added
  2026-05-29)* — the global `(min, max)` of a (series **or** scalar) attribute
  across all edges/nodes and all timestamps. Pass it as `show_network(minmax=...)`
  (edges) or `ring_minmax=...` (rings) to **pin one colour/ring scale across a batch
  of per-timestamp snapshots**, so an exported image series shares a single legend.
  `timestamps` restricts the window; returns `(None, None)` if the attribute is
  absent/non-numeric everywhere. Exported from `uesgraphs.analyze`. Used by the
  interactive panel in `hns_tools.analyze.interactive` (see that module's
  `docs/interactive_plotting.md`) for both the "pin scale" toggle and the batch
  image-series export.

## 3. Visuals — `visuals.py` (`create_plot` + `show_network`)

All new params default to `None`/neutral → **existing calls are unchanged**.
Normalization is computed over **building nodes only** (avoids the
`get_min_max(mode="node")` KeyError on street/heat nodes); a Series attribute
raises a clear `TypeError` (reduce it first).

- **`generic_node_size`** / **`generic_node_color`** (+ `node_size_min=10`,
  `node_size_max=120`, `node_minmax`, `node_ylabel`) — size/color building dots by
  a scalar node attribute. Color reuses the existing `gs[1]` colorbar.
- **`generic_node_ring`** (+ `ring_size_min=60`, `ring_size_max=600`,
  `ring_width=1.5`, `ring_color`, `ring_alpha`, `ring_minmax`, `ring_legend=False`,
  `ring_legend_scale`, `ring_legend_unit`, `ring_legend_title`) — **recommended
  secondary overlay**: a hollow halo ring around the building node, area ∝ value,
  **anchored at 0** so value 0 draws *no ring* (distinguishes "no demand").
  `ring_size_min` keeps the smallest non-zero ring outside the base node dot.
  Optional discrete legend (min/mid/max). Uses **no** colorbar → the primary
  edge/node colorbar stays for the main attribute.

## Minimal usage

```python
import uesgraphs as ug
from uesgraphs.analyze import assign_data_pipeline, snapshot_at

graph = assign_data_pipeline(graph, simulation_data_path=..., start_date=...,
                             end_date=..., time_interval="15min",
                             aixlib_version="2.1.0", system_model_path=...)
# with_demand_power=True (default) -> nodes carry "heat_power_prescribed" (pd.Series, W)

t = snapshot_at(graph, "2023-04-13 14:00",
                edge_keys=["m_flow"], node_keys=["heat_power_prescribed"])

ug.Visuals(graph).show_network(
    generic_extensive_size="m_flow_t", ylabel="Mass flow [kg/s]",   # PRIMARY (edges)
    generic_node_ring="heat_power_prescribed_t",                    # SECONDARY (demand)
    ring_legend=True, ring_legend_scale=1e-3, ring_legend_unit="kW",
    timestamp=str(t),
)
```

## Honest notes / caveats

- **Prescribed vs realized:** the default is the prescribed setpoint (`Q_flow_input`).
  Label it "demand", not "delivered power". For realized power use
  `demand_power_realized=True` (needs the `senT_*` columns in the result).
- **Encoding:** prefer the **ring** (secondary) for demand so it does not compete
  with the primary edge/node color+size. Color-on-nodes collides with the edge
  colorbar (only one colorbar) — use it only when edges are uncolored.
- **Number of timesteps** = rows in the result `.mat`, set by the Modelica
  experiment `StopTime`/`Interval` (model generation `sim_params['stop_time']`,
  `timestep`). `time_interval` in `assign_data_pipeline` only *labels* rows at that
  spacing — it does **not** resample. Few timesteps ⇒ short `StopTime`.
- **Result cache:** `check_input_file` reuses an existing `dsres.gzip` without an
  mtime check. After re-simulating into the same folder, delete the stale `.gzip`.
- **Units:** `Q_flow_input` is in **W**; use `ring_legend_scale=1e-3` for kW labels.
