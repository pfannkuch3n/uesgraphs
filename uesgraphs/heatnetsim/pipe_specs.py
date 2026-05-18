"""Apply pipe attributes from a DN-keyed catalog CSV to UESGraph edges.

When a UESGraph is built from GeoJSON it often only carries the DN (nominal
diameter) on each edge — but HeatNetSim and the Modelica export need
``diameter``, ``dIns`` (insulation thickness) and ``wall_thickness``. This
module is the fallback that resolves those values from a catalog CSV after
the import, without overwriting any attribute the edge already has.

The catalog CSV must have one row per DN with at least the DN column; the
``column_mapping`` argument controls which CSV columns get written to which
edge attribute.
"""

import logging
from pathlib import Path

import pandas as pd

from uesgraphs.utilities import set_up_file_logger


DEFAULT_COLUMN_MAPPING = {
    "inner_diameter": "diameter",
    "d_ins":          "dIns",
    "wall_thickness": "wall_thickness",
}


def _parse_dn(value):
    """Normalize a DN value to int. Accepts 'DN200', '200', 200, None.

    Returns None if the value can't be interpreted (caller decides what to do).
    """
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    if isinstance(value, str):
        cleaned = value.strip().upper().lstrip("DN").strip()
        try:
            return int(float(cleaned))
        except ValueError:
            return None
    try:
        return int(value)
    except (ValueError, TypeError):
        return None


def _read_pipe_specs(csv_path, dn_column, logger):
    """Read the catalog CSV and return it as a DataFrame indexed by DN (int).

    - Lines starting with '#' are treated as comments.
    - DN values are normalized via ``_parse_dn``.
    - Rows with unparseable DN are dropped (with a warning).
    - Duplicate DN entries: first occurrence wins (with a warning).
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Pipe specs CSV not found: {csv_path}")

    df = pd.read_csv(csv_path, comment="#")

    if dn_column not in df.columns:
        raise ValueError(
            f"DN column '{dn_column}' not found in {csv_path}. "
            f"Available columns: {list(df.columns)}"
        )

    df["_dn_norm"] = df[dn_column].apply(_parse_dn)

    bad = df[df["_dn_norm"].isna()]
    if not bad.empty:
        logger.warning(
            "Dropping %d row(s) with unparseable DN values: %s",
            len(bad), bad[dn_column].tolist(),
        )
        df = df.dropna(subset=["_dn_norm"])

    duplicate_mask = df["_dn_norm"].duplicated(keep="first")
    if duplicate_mask.any():
        dups = df.loc[duplicate_mask, "_dn_norm"].astype(int).tolist()
        logger.warning(
            "Pipe specs CSV has duplicate DN entries; keeping first occurrence "
            "for each: %s", dups,
        )
        df = df.loc[~duplicate_mask]

    df["_dn_norm"] = df["_dn_norm"].astype(int)
    df = df.set_index("_dn_norm").drop(columns=[dn_column])
    return df


def _get_edge_dn(edge_data, dn_attr):
    """Retrieve and normalize the DN of an edge.

    Looks first at edge_data[dn_attr], then at edge_data['attr_dict'][dn_attr]
    (where uesgraphs stores raw GeoJSON properties).
    """
    dn = edge_data.get(dn_attr)
    if dn is None:
        nested = edge_data.get("attr_dict") or {}
        dn = nested.get(dn_attr)
    return _parse_dn(dn)


def apply_pipe_specs_to_graph(
    graph,
    csv_path,
    column_mapping=None,
    dn_column="DN",
    edge_dn_attr="DN",
    overwrite=False,
    logger=None,
):
    """Assign pipe properties from a DN-keyed catalog CSV to graph edges.

    For each edge, reads its DN, looks up the matching catalog row, and writes
    the catalog values as edge attributes. By default existing attributes are
    preserved - the catalog only fills in what's missing.

    Parameters
    ----------
    graph : uesgraphs.UESGraph
        Modified in-place.
    csv_path : str or Path
        Path to a CSV with one row per DN. '#' comments allowed. Must contain a
        column named ``dn_column`` (default: 'DN'). DN values may be given as
        int, plain string ('200'), or 'DN'-prefixed string ('DN200').
    column_mapping : dict, optional
        Dictionary ``{csv_column: edge_attribute_name}``. Only listed columns
        are applied to edges. If None, uses :data:`DEFAULT_COLUMN_MAPPING`.
        Pass an explicit mapping to add new columns or rename targets.
    dn_column : str, default 'DN'
        Name of the DN column in the CSV.
    edge_dn_attr : str, default 'DN'
        Name of the DN attribute on edges. Looked up first as
        ``edge_data[edge_dn_attr]``, then as
        ``edge_data['attr_dict'][edge_dn_attr]``.
    overwrite : bool, default False
        If False (recommended), existing edge attributes are kept and the
        catalog only writes attributes that aren't already there. If True,
        catalog values replace existing ones.
    logger : logging.Logger, optional
        If None, a project file logger is created lazily (writes to the OS
        temp directory; the file path is printed at setup time).

    Returns
    -------
    graph : uesgraphs.UESGraph
        Same object, modified in-place. Returned for chaining.

    Notes
    -----
    Edges whose DN can't be determined or whose DN is not in the catalog are
    skipped (logged at WARNING level). The function never raises on missing
    DNs - it's expected that some edges might not have one and you'll fall
    back to other defaults later in the pipeline.
    """
    if logger is None:
        logger = set_up_file_logger(
            f"{__name__}.apply_pipe_specs_to_graph",
            level=logging.INFO,
        )

    if column_mapping is None:
        column_mapping = dict(DEFAULT_COLUMN_MAPPING)

    specs = _read_pipe_specs(csv_path, dn_column=dn_column, logger=logger)

    available = set(specs.columns)
    missing_cols = [c for c in column_mapping if c not in available]
    if missing_cols:
        logger.warning(
            "Columns in column_mapping not present in CSV (skipped): %s",
            missing_cols,
        )
        column_mapping = {k: v for k, v in column_mapping.items() if k in available}

    if not column_mapping:
        logger.warning("No usable columns left in column_mapping - nothing to apply.")
        return graph

    n_set, n_kept, n_no_dn, n_unknown_dn = 0, 0, 0, 0

    for u, v, data in graph.edges(data=True):
        dn = _get_edge_dn(data, dn_attr=edge_dn_attr)
        if dn is None:
            n_no_dn += 1
            continue
        if dn not in specs.index:
            n_unknown_dn += 1
            logger.warning("Edge (%s, %s) has DN=%s, not in catalog - skipped.", u, v, dn)
            continue

        row = specs.loc[dn]
        for csv_col, edge_attr in column_mapping.items():
            value = float(row[csv_col])
            if pd.isna(value):
                continue
            if edge_attr in data and not overwrite:
                n_kept += 1
                continue
            data[edge_attr] = value
            n_set += 1

    logger.info(
        "Pipe specs applied: %d attribute(s) set, %d preserved, "
        "%d edge(s) without DN, %d edge(s) with unknown DN.",
        n_set, n_kept, n_no_dn, n_unknown_dn,
    )
    return graph
