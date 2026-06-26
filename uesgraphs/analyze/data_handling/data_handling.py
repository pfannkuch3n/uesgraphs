
import pyarrow.parquet as pq

import pandas as pd


from typing import List, Dict, Generator, Optional, Union, Tuple, Set, Any
import os
from pathlib import Path

import logging
import tempfile
from datetime import datetime

import uesgraphs as ug
from uesgraphs.systemmodels import utilities as ut
from uesgraphs.analyze.data_handling import graph_transformation
from uesgraphs.analyze.data_handling.mat_handler import mat_to_parquet, keep_fingerprint
from uesgraphs.utilities import set_up_terminal_logger, set_up_file_logger


#### Global Variables ####
AIXLIB_MASKS = None  # Dictionary to store masks for column names



#### Functions 2: Data Aquisition ####

def _read_keep_marker(parquet_path: str):
    """The ``ues_keep`` kept-column fingerprint stored in a parquet cache, or None
    if absent/unreadable (legacy cache or non-parquet)."""
    try:
        meta = pq.ParquetFile(parquet_path).schema_arrow.metadata or {}
    except Exception:
        return None
    val = meta.get(b"ues_keep")
    return val.decode() if val is not None else None


def check_input_file(file_path: str, logger=None, force_reconvert: bool = False,
                     convert_kwargs: Optional[dict] = None) -> str:
    """
    Check and prepare an input file for processing.

    Resolves the parquet cache for a Dymola ``.mat`` result, converting on demand.
    Resolution order (so a present ``.mat`` always yields a current cache):

    1. a fresh ``<base>.parquet`` (newer than the ``.mat``)      -> use it
    2. a ``.mat``                                                -> (re)convert to ``<base>.parquet``
    3. an existing cache with no ``.mat`` to rebuild from        -> read it (incl. legacy ``.gzip``)
    4. the original file                                         -> use as-is

    Args:
        file_path: Path to the input file (typically the ``.mat``).
        logger: optional logger.
        force_reconvert: skip step 1 and rebuild from the ``.mat``. Used by the
            self-heal path when a required column is absent from a fast cache.
        convert_kwargs: kwargs forwarded to :func:`mat_to_parquet` (e.g.
            ``{"keep_suffixes": (...)}``). Defaults to the project scope
            (:data:`DEFAULT_CACHE_SCOPE`).

    Returns:
        Path to the file to read (usually ``<base>.parquet``).

    Raises:
        ValueError: If file doesn't exist or conversion fails.
    """
    if logger is None:
        logger = set_up_terminal_logger(f"{__name__}.check_input_file")

    if not file_path:
        raise ValueError("File path cannot be empty")

    if convert_kwargs is None:
        convert_kwargs = _default_convert_kwargs()

    base_path = os.path.splitext(file_path)[0]
    parquet_path = f"{base_path}.parquet"
    gzip_legacy = f"{base_path}.gzip"
    mat_path = f"{base_path}.mat"

    def _fresh(cache: str) -> bool:
        # Valid if there is no .mat to compare against, or the cache is newer.
        return (not os.path.exists(mat_path)
                or os.path.getmtime(cache) >= os.path.getmtime(mat_path))

    def _keep_ok(cache: str) -> bool:
        """Whether *cache* was built with the column set we now expect. Only
        actionable when a .mat exists (otherwise we cannot rebuild, so keep it).
        A cache predating the fingerprint is rebuilt for a fast expectation so it
        picks up columns added to the masks later; a full cache already has all."""
        if not os.path.exists(mat_path):
            return True
        expected = keep_fingerprint(convert_kwargs.get("keep_suffixes"))
        actual = _read_keep_marker(cache)
        if actual is None:
            return expected == "full"
        return actual == expected

    # 1) Fresh new-format cache, built with the column set we now expect.
    if not force_reconvert and os.path.exists(parquet_path) and _fresh(parquet_path):
        if _keep_ok(parquet_path):
            logger.info(f"Using parquet cache: {parquet_path}")
            return parquet_path
        logger.info("Parquet cache kept-column set is stale (masks changed) -> "
                    "rebuilding once from .mat")

    # 2) (Re)convert from .mat. Also migrates legacy gzip-only setups to parquet
    #    and refreshes a stale cache after a re-simulation (mtime check above).
    if os.path.exists(mat_path):
        try:
            logger.info(f"Converting .mat file to parquet: {mat_path}")
            out = mat_to_parquet(save_as=base_path, fname=mat_path,
                                 with_unit=False, **convert_kwargs)
            logger.info(f"Successfully converted .mat file to: {out}")
            return out
        except Exception as e:
            logger.error(f"Failed to convert .mat file: {mat_path}")
            raise ValueError(f"Could not convert .mat file to parquet: {mat_path}") from e

    # 3) An existing cache, when there is no .mat to rebuild from (e.g. archived
    #    result). New format wins; legacy .gzip stays readable.
    if os.path.exists(parquet_path):
        logger.info(f"Using parquet cache (no .mat to refresh): {parquet_path}")
        return parquet_path
    if os.path.exists(gzip_legacy):
        logger.info(f"Using legacy .gzip cache (no .mat to refresh): {gzip_legacy}")
        return gzip_legacy

    # 4) Original file.
    if not os.path.exists(file_path):
        raise ValueError(f"File does not exist: {file_path}")

    logger.info(f"Using original file: {file_path}")
    return file_path

def validate_columns_exist(file_path: str, required_columns: List[str],
                          logger: Optional[logging.Logger] = None) -> Set[str]:
    """
    Check if all required columns exist in the simulation data file.
    
    Args:
        file_path: Path to the parquet/simulation file
        required_columns: List of exact column names that must exist
        logger: Logger instance (optional)
        
    Returns:
        Set of available columns from the file
        
    Raises:
        KeyError: If any required columns are missing
        ValueError: If file cannot be read
    """
    if logger is None:
        logger = set_up_terminal_logger(f"{__name__}.validate_columns_exist")
    
    logger.info(f"Validating {len(required_columns)} required columns in: {file_path}")
    
    # Read file metadata (no data loading)
    try:
        parquet_file = pq.ParquetFile(file_path)
        available_columns = set(parquet_file.schema.names)
        logger.debug(f"File contains {len(available_columns)} total columns")
    except Exception as e:
        error_msg = f"Could not read file metadata: {e}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Check for missing columns
    required_set = set(required_columns)
    missing_columns = required_set - available_columns
    
    if missing_columns:
        missing_list = sorted(missing_columns)
        error_msg = f"Missing required columns: {missing_list}"
        logger.error(error_msg)
        
        # Raise KeyError with first missing column for auto-retry compatibility
        first_missing = missing_list[0]
        raise KeyError(first_missing)
    
    logger.info("SUCCESS: All required columns found in data file")
    return available_columns

def process_simulation_result(file_path: str, filter_list: List[str], 
                        chunk_size: int = 100000, logger=None) -> Generator[pd.DataFrame, None, None]:
    """
    Process a parquet file in chunks to reduce memory usage.
    
    Args:
        file_path: Path to the parquet file
        filter_list: List of column patterns to filter
        chunk_size: Number of rows to process at once
        logger: Optional logger instance
        
    Yields:
        pd.DataFrame: Processed chunks of the parquet file
    """
    if logger is None:
        logger = set_up_terminal_logger(f"{__name__}.process_parquet_file")
    
    # Step 1: Check if file exists and convert .mat if needed
    processed_file_path = check_input_file(file_path=file_path, logger=logger)

    # Step 1: Validate all columns exist (will raise KeyError if missing)
    available_columns = validate_columns_exist(processed_file_path, required_columns=filter_list, logger=logger)
    
    # Step 2: Check if any columns match the filter_list
    logger.info(f"Starting parquet file processing: {processed_file_path}")
    logger.debug(f"Filter patterns: {filter_list}")
    logger.debug(f"Chunk size: {chunk_size}")
    
    try:
        # Read parquet file metadata to get columns
        parquet_file = pq.ParquetFile(processed_file_path,
                                  thrift_string_size_limit=2_000_000_000,
                                  thrift_container_size_limit=2_000_000_000)
        chunks = []
        total_rows = 0
       
        
        # Read and process the file in chunks
        for batch in parquet_file.iter_batches(batch_size=chunk_size, columns=filter_list):
            total_rows += 1
            chunk_df = batch.to_pandas()
            chunks.append(chunk_df)
            if len(chunks) % 10 == 0:  # Log every 10 chunks
                logger.debug(f"Loaded {len(chunks)} chunks, {total_rows} rows so far")

        if not chunks:
            logger.warning("No data loaded from file")
            return pd.DataFrame()
        
        result_df = pd.concat(chunks, axis = 0, ignore_index=True)
        
        logger.info(f"Successfully loaded {len(result_df)} rows, {len(result_df.columns)} columns")
        logger.debug(f"DataFrame memory usage: {result_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        return result_df
    except Exception as e:
        logger.error(f"Error processing parquet file {processed_file_path}: {str(e)}")
        raise e



#### Functions 3: Data Processing ####

def prepare_DataFrame(df, base_date=datetime(2024, 1, 1), time_interval="15min", 
                      start_date=None, end_date=None, logger=None) -> pd.DataFrame:
    """
    Prepare a DataFrame with a datetime index using customizable parameters.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame to be processed
    base_date : datetime, optional
        The starting date for the index (default: 2024-01-01)
    time_interval : str, optional
        Frequency of the time intervals (e.g., '15min', '1h', '30min', default: '15min')
    start_date : datetime, optional
        If provided, slice the DataFrame from this date (inclusive)
    end_date : datetime, optional
        If provided, slice the DataFrame until this date (inclusive)
    logger : logging.Logger, optional
        Logger instance for logging operations
    
    Returns:
    --------
    DataFrame: A DataFrame containing the data from the parquet file for the specified time period.
    """
    if logger is None:
        logger = set_up_terminal_logger(f"{__name__}.prepare_DataFrame")
    
    logger.info(f"Preparing DataFrame with {len(df)} rows and {len(df.columns)} columns")
    logger.debug(f"Parameters - base_date: {base_date}, time_interval: {time_interval}")
    logger.debug(f"Date filtering - start_date: {start_date}, end_date: {end_date}")
    
    try:
        # Create datetime index with specified frequency    
        logger.debug(f"Creating datetime index with frequency '{time_interval}'")
        datetime_index = pd.date_range(start=base_date, periods=len(df), freq=time_interval)
        logger.debug(f"Created datetime index from {datetime_index[0]} to {datetime_index[-1]}")
        
        # Set the index of the DataFrame to the datetime index
        df.index = datetime_index
        df.index.name = 'DateTime'
        logger.info(f"Applied datetime index to DataFrame")
        
        # Filter by date range if specified
        original_length = len(df)
        if start_date is not None and end_date is not None:
            logger.info(f"Filtering DataFrame from {start_date} to {end_date}")
            df = df.loc[start_date:end_date]
        elif start_date is not None:
            logger.info(f"Filtering DataFrame from {start_date} onwards")
            df = df.loc[start_date:]
        elif end_date is not None:
            logger.info(f"Filtering DataFrame up to {end_date}")
            df = df.loc[:end_date]
        
        # Log filtering results if any filtering was applied
        if len(df) != original_length:
            logger.info(f"Date filtering applied: {original_length} -> {len(df)} rows ({len(df)/original_length*100:.1f}% retained)")
            if len(df) == 0:
                logger.warning("Date filtering resulted in empty DataFrame - check date range parameters")
        else:
            logger.debug("No date filtering applied")
        
        logger.info(f"Successfully prepared DataFrame: {len(df)} rows, index range: {df.index[0]} to {df.index[-1]}")
        return df
        
    except ValueError as e:
        error_msg = f"Error creating date range with frequency {time_interval} and base date {base_date}. Original error: {e}"
        logger.error(error_msg)
        raise ValueError(error_msg) from e
    except Exception as e:
        error_msg = f"Unexpected error in prepare_DataFrame: {str(e)}"
        logger.error(error_msg)
        raise

#### Functions 4: Data Assignment to UESGraph ####

AIXLIB_MASKS = {
    "2.1.0": {
        "edge": {
            # Extensive properties - same value at both ports
            "m_flow": "networkModel.pipe{pipe_code}{type}.port_a.m_flow",
            "dp": "networkModel.pipe{pipe_code}{type}.dp",
            "velocity": "networkModel.pipe{pipe_code}{type}.v_med",
        },
        "node": {
            # Intensive properties - may differ between ports
            "pressure": {
                "port_a": "networkModel.pipe{pipe_code}{type}.port_a.p",
                "port_b": "networkModel.pipe{pipe_code}{type}.port_b.p"
            },
            "temperature": {
                "port_a": "networkModel.pipe{pipe_code}{type}.sta_a.T",
                "port_b": "networkModel.pipe{pipe_code}{type}.sta_b.T"
            }
        },
        "edge_optional": {
            # OPTIONAL per-pipe heat loss to the surroundings: the pipe model's
            # OWN loss term (heatPort is connected to TGround in the generated
            # network), in W with >0 = loss. Kept SEPARATE from the required
            # "edge" block so a data file WITHOUT heatPort logging degrades to a
            # warning instead of raising (see assign_data_pipeline /
            # assign_edge_loss). heatPort is exposed top-level by every DHC pipe
            # wrapper (DHCPipe, PlugFlowPipe*, StaticPipe).
            "Q_loss": "networkModel.pipe{pipe_code}{type}.heatPort.Q_flow",
            # Return-line pipe loss (literal "R", no {type}): so a SUPPLY-side
            # graph also carries the return pipe loss, and the per-pipe total in
            # pump_vs_loss covers BOTH lines (VL+RL). Only mapped on a supply
            # graph (skipped when the graph itself is the return side, to avoid
            # double counting) - see assign_edge_loss.
            "Q_loss_return": "networkModel.pipe{pipe_code}R.heatPort.Q_flow",
        },
        "building": {
            # Demand (building) thermal power - keyed by the BUILDING/demand node
            # NAME, not by pipe code. Flat network-model RealInput
            # "<name>Q_flow_input" (NOT "demand<name>.Q_flow_input").
            # Default = PRESCRIBED demand setpoint [W].
            "heat_power_prescribed": "networkModel.{name}Q_flow_input",
            # Substation return temperature [K] as a BUILDING quantity (keyed by
            # demand name), assigned to the node like the prescribed power. In the
            # open-loop demand model the return is a fixed TReturn, so it is a
            # building/substation quantity, not a network-node value from a pipe.
            "temperature_return": "networkModel.demand{name}.senT_return.T",
            # Optional realized-power reconstruction inputs (only pulled when
            # realized=True): realized = m_flow * cp * (T_supply - T_return).
            # No prebuilt realized-Q variable exists in the AixLib demand models.
            "_realized": {
                "T_supply": "networkModel.demand{name}.senT_supply.T",
                "T_return": "networkModel.demand{name}.senT_return.T",
                "m_flow": "networkModel.demand{name}.senT_supply.m_flow",
            },
        }
    },
    "2.0.0": {
        "edge": {
            # Extensive properties - same value at both ports
            "m_flow": "networkModel.pipe{pipe_code}{type}.port_a.m_flow",
        },
        "node": {
            # Intensive properties - may differ between ports
            "pressure": {
                "port_a": "networkModel.pipe{pipe_code}{type}.port_a.p",
                "port_b": "networkModel.pipe{pipe_code}{type}.ports_b[1].p"
            },
            "temperature": {
                "port_a": "networkModel.pipe{pipe_code}{type}.sta_a.T",
                "port_b": "networkModel.pipe{pipe_code}{type}.sta_b[1].T"
            }
        },
        "edge_optional": {
            # See AIXLIB_MASKS["2.1.0"]["edge_optional"] for documentation.
            "Q_loss": "networkModel.pipe{pipe_code}{type}.heatPort.Q_flow",
            "Q_loss_return": "networkModel.pipe{pipe_code}R.heatPort.Q_flow",
        },
        "building": {
            # See AIXLIB_MASKS["2.1.0"]["building"] for documentation.
            "heat_power_prescribed": "networkModel.{name}Q_flow_input",
            "temperature_return": "networkModel.demand{name}.senT_return.T",
            "_realized": {
                "T_supply": "networkModel.demand{name}.senT_supply.T",
                "T_return": "networkModel.demand{name}.senT_return.T",
                "m_flow": "networkModel.demand{name}.senT_supply.m_flow",
            },
        }
    }
}


# --------------------------------------------------------------------------
# Supply-station masks: EXCHANGEABLE per supply model
# --------------------------------------------------------------------------
# The supply station is swappable (open-loop ideal today, closed-loop or a model
# with a real pump later). Different stations expose different variable names AND
# a different pump-power semantics, so the supply mask must NOT be hard-wired.
#
# Keyed by the supply *model* name (the ``comp_model`` the model generator uses).
# assign_data_pipeline auto-selects the entry per supply node from the system
# model's ``comp_model`` (falling back to the ``supply_model`` argument). Add a
# new station = add a new entry here.
#
# Each entry:
#   "vars":       attribute -> Modelica column template (keyed by supply NAME).
#   "pump_power": how to obtain pump power:
#       {"mode": "hydraulic"}            -> reconstruct ideal hydraulic power
#                                           P = |m_flow|/rho * (p_flow - p_return)
#                                           (ideal pressure source, no efficiency)
#       {"mode": "native", "var": "..."} -> read a logged pump-power variable
#                                           directly (real-pump models).
SUPPLY_MASKS = {
    # AixLib.Fluid.DistrictHeatingCooling.Supplies.OpenLoop.SourceIdeal:
    # ideal Boundary_pT pressure source + sink, NO native pump-power variable.
    "OpenLoop.SourceIdeal": {
        "vars": {
            "heat_power_supply": "networkModel.supply{name}.Q_flow",
            "m_flow":            "networkModel.supply{name}.senMasFlo.m_flow",
            "pressure_flow":     "networkModel.supply{name}.port_b.p",
            "pressure_return":   "networkModel.supply{name}.port_a.p",
        },
        "pump_power": {"mode": "hydraulic"},
    },
}

# Default station used when no system-model comp_model is available.
DEFAULT_SUPPLY_MODEL = "OpenLoop.SourceIdeal"


def resolve_supply_mask(comp_model=None, supply_model=None):
    """Pick a SUPPLY_MASKS entry for a supply node.

    Matches ``comp_model`` (a full Modelica path like
    ``AixLib...Supplies.OpenLoop.SourceIdeal``) against the registry keys by
    suffix, so both the short key and the full path resolve. Falls back to the
    explicit ``supply_model`` argument, then to ``DEFAULT_SUPPLY_MODEL``.

    Returns (mask_entry, resolved_key) or (None, None) if nothing matches.
    """
    candidates = [c for c in (comp_model, supply_model, DEFAULT_SUPPLY_MODEL) if c]
    for cand in candidates:
        if cand in SUPPLY_MASKS:
            return SUPPLY_MASKS[cand], cand
        for key in SUPPLY_MASKS:
            if cand.endswith(key):
                return SUPPLY_MASKS[key], key
    return None, None


# Cache scope used when converting .mat -> parquet. "fast" keeps only the
# columns the analysis pipeline actually reads (derived from AIXLIB_MASKS);
# "full" keeps every variable (the original behaviour). Flip to "full" project-
# wide to restore the old all-variables cache.
DEFAULT_CACHE_SCOPE = "fast"  # "fast" | "full"


def mask_keep_suffixes(masks=None):
    """The stable name suffixes of all mask templates — the part after the last
    ``{...}`` placeholder (e.g. ``networkModel.pipe{pipe_code}{type}.port_a.m_flow``
    -> ``.port_a.m_flow``).

    Derived from :data:`AIXLIB_MASKS` so there is a single source of truth: extend
    AIXLIB_MASKS with a new quantity and both reading (``build_filter_list_*``) and
    fast conversion pick it up automatically — no second list to maintain. ``Time``
    is always included.
    """
    if masks is None:
        masks = AIXLIB_MASKS
    out = set()

    def _walk(d):
        for value in d.values():
            if isinstance(value, dict):
                _walk(value)
            elif isinstance(value, str):
                out.add(value.rsplit("}", 1)[-1])

    _walk(masks)
    # Also keep the exchangeable supply-station columns (registry lives outside
    # AIXLIB_MASKS) so the fast cache does not drop them.
    _walk(SUPPLY_MASKS)
    return tuple(out) + ("Time",)


def _default_convert_kwargs():
    """Conversion kwargs implied by :data:`DEFAULT_CACHE_SCOPE`."""
    if DEFAULT_CACHE_SCOPE == "full":
        return {"keep_suffixes": None}
    return {"keep_suffixes": mask_keep_suffixes()}


def build_filter_list_pipe(graph, mask, logger=None):
    """
    Build a list of filter variables for pipes in a district heating network graph.
    
    This function extracts patterns from a mask data structure and formats them
    with specific pipe codes and type information for each edge (pipe) in the graph.
    This is useful for filtering and analyzing district heating networks.
    
    Args:
        graph : uesgraphs object.
        mask (dict): Hierarchical data structure containing filter patterns.
                    Expected categories are 'edge' and 'node':
                    - 'edge': Dict with direct pattern values
                    - 'node': Dict with nested pattern values
        logger (logging.Logger, optional): Logger instance for debug output.
                                          Will be created automatically if None.
    
    Returns:
        list: List of formatted variable names for all pipes in the graph,
              based on the mask patterns.
    
    Example:
        >>> mask = {
        ...     'edge': {'pressure': 'p_{pipe_code}_{type}'},
        ...     'node': {'inlet': {'temp': 'T_in_{pipe_code}_{type}'}}
        ... }
        >>> filter_list = build_filter_list_pipe(graph, mask)
        >>> print(filter_list)
        ['p_PIPE001_supply', 'T_in_PIPE001_supply', ...]
    
    Raises:
        KeyError: If pipe edges don't have a 'name' attribute
        AttributeError: If graph doesn't have edges attribute
    """
    # Initialize logger if not provided
    if logger is None:
        logger = set_up_terminal_logger("BuildFilterListPipe")
    
    # Collection of all simulation patterns from the mask structure
    simulation_patterns = []
    
    # Iterate through all categories in the mask
    for category_name, category_data in mask.items():
        if category_name not in ["edge", "node"]:
            # These are handled by their own builders/assigners, not by pipe-code:
            #   "building"     -> build_filter_list_demand / assign_demand_power
            #   "edge_optional"-> build_filter_list_loss   / assign_edge_loss
            #   "supply"       -> build_filter_list_supply / assign_supply_values
            if category_name not in ("building", "edge_optional", "supply"):
                logger.warning(f"Unknown category '{category_name}' in mask, skipping")
            continue
            
        if category_name == "edge":
            # Edge category: direct extraction of pattern values
            # Example: {'m_flow': 'p_{pipe_code}_{type}'} -> ['p_{pipe_code}_{type}']
            simulation_patterns.extend(category_data.values())
            logger.debug(f"Added {len(category_data)} edge patterns")
            
        elif category_name == "node":
            # Node category: nested structure - extract all port patterns
            # Example: {'temperature': {'port_a': 'T_in_{pipe_code}_{type}'}} 
            #          -> ['T_in_{pipe_code}_{type}']
            for port_name, attribute_patterns in category_data.items():
                if isinstance(attribute_patterns, dict):
                    simulation_patterns.extend(attribute_patterns.values())
                    logger.debug(f"Added {len(attribute_patterns)} node patterns "
                               f"for port '{port_name}'")
                else:
                    logger.warning(f"Expected dict for node port '{port_name}', "
                                 f"got {type(attribute_patterns)}")
    
    logger.info(f"Extracted {len(simulation_patterns)} simulation patterns total")
    
    # Get type prefix from graph (e.g., 'supply', 'return')
    type_prefix = get_supply_type_prefix(graph)
    logger.debug(f"Using type prefix: '{type_prefix}'")
    
    # List for all generated filter variables
    filter_variables = []
    
    # Generate filter variables for each edge (pipe) in the graph
    for edge in graph.edges:
        try:
            # Extract pipe code from edge attributes
            pipe_code = graph.edges[edge]["name"]
            
            # Generate a variable for each simulation pattern
            for pattern in simulation_patterns:
                # Format pattern with specific values
                variable_name = pattern.format(
                    pipe_code=pipe_code,
                    type=type_prefix
                )
                filter_variables.append(variable_name)
                
            logger.debug(f"Generated {len(simulation_patterns)} variables "
                        f"for pipe '{pipe_code}'")
                        
        except KeyError as e:
            logger.error(f"Edge {edge} missing required attribute: {e}")
            raise
        except Exception as e:
            logger.error(f"Error processing edge {edge}: {e}")
            raise
    
    logger.info(f"Created filter list with {len(filter_variables)} entries "
               f"for {len(graph.edges)} pipes")

    return filter_variables


def build_filter_list_demand(graph, MASK, realized=False, logger=None):
    """Build the list of simulation columns for building (demand) thermal power.

    Unlike build_filter_list_pipe, these variables are keyed by the BUILDING /
    demand node NAME (networkModel.<name>Q_flow_input), not by pipe code. Supply
    buildings are skipped (they have no demand input).

    Args:
        graph: uesgraphs object with a nodelist_building.
        MASK: mask dict; the "building" section drives the patterns.
        realized: if True, also request the sensor columns needed to reconstruct
                  realized power (m_flow * cp * dT).
        logger: optional logger.

    Returns:
        list[str]: column names; empty if MASK has no "building" section.
    """
    if logger is None:
        logger = set_up_terminal_logger("BuildFilterListDemand")

    bldg_mask = MASK.get("building")
    if not bldg_mask:
        logger.debug("No 'building' section in MASK; no demand-power columns.")
        return []

    pattern = bldg_mask["heat_power_prescribed"]
    tret_pattern = bldg_mask.get("temperature_return")
    realized_patterns = bldg_mask.get("_realized", {}) if realized else {}

    columns = []
    for node in graph.nodelist_building:
        if graph.nodes[node].get("is_supply_heating"):
            continue
        name = graph.nodes[node].get("name")
        if name is None:
            continue
        columns.append(pattern.format(name=name))
        if tret_pattern:  # load the substation return-temp sensor too (always)
            columns.append(tret_pattern.format(name=name))
        for realized_pattern in realized_patterns.values():
            columns.append(realized_pattern.format(name=name))

    logger.info(f"Created demand-power filter list with {len(columns)} entries "
                f"(realized={realized})")
    return columns


def build_filter_list_loss(graph, MASK, logger=None):
    """Build the list of simulation columns for the OPTIONAL per-pipe heat loss.

    Reads the ``edge_optional`` mask section (keyed by pipe code, like the
    required ``edge`` section) and formats one column per pipe per variable.
    Returns an empty list if MASK has no ``edge_optional`` section.
    """
    opt = MASK.get("edge_optional")
    if not opt:
        return []
    type_prefix = get_supply_type_prefix(graph)
    columns = []
    for edge in graph.edges:
        pipe_code = graph.edges[edge]["name"]
        for var, pattern in opt.items():
            # The return-line loss is only meaningful on a supply graph; on a
            # return graph its pipes are already the Q_loss, so skip it.
            if var == "Q_loss_return" and type_prefix == "R":
                continue
            columns.append(pattern.format(pipe_code=pipe_code, type=type_prefix))
    if logger:
        logger.info(f"Created heat-loss filter list with {len(columns)} entries")
    return columns


def build_filter_list_supply(graph, MASK, logger=None):
    """Build the list of simulation columns for the OPTIONAL supply-node
    quantities (thermal power, mass flow, flow/return pressure).

    Keyed by the SUPPLY building NAME (e.g. "S1"), driven by the ``supply``
    mask section (injected from SUPPLY_MASKS by assign_data_pipeline). Returns
    an empty list if MASK has no ``supply`` section.
    """
    sup = MASK.get("supply")
    if not sup:
        return []
    vars_ = sup.get("vars", {})
    columns = []
    for node in graph.nodelist_building:
        if not graph.nodes[node].get("is_supply_heating"):
            continue
        name = graph.nodes[node].get("name")
        if name is None:
            continue
        for pattern in vars_.values():
            columns.append(pattern.format(name=name))
    if logger:
        logger.info(f"Created supply filter list with {len(columns)} entries")
    return columns


##### Assign node values

def assign_node_values(graph, df, port_mapping, mask, time_index=0, logger=None):
    """
    Assigns node values from simulation data using flexible mask configuration.
    
    Processes intensive properties (pressure, temperature) that may differ between 
    ports of the same pipe, unlike extensive properties (mass flow) that are 
    identical at both ports.
    
    Args:
        graph: NetworkX or uesgraphs graph with nodes to assign values to
        df: DataFrame containing simulation data
        port_mapping: Dict mapping node_ids to list of connected ports.
                     Example: {1: ['pipe001.port_a', 'pipe002.port_b'], 
                              2: ['pipe001.port_b', 'pipe003.port_a']}
                     Source: [Method name if known, otherwise leave empty]
        mask: Mask dictionary containing node configuration for intensive properties
        time_index: Time step index to extract from df (default: 0)
        logger: Logger instance (optional, creates terminal logger if None)
    """
    if logger is None:
        logger = set_up_terminal_logger(f"{__name__}.assign_node_values")
    
    # Extract node configuration
    node_config = mask.get("node", {})
    if not node_config:
        logger.error("No 'node' configuration found in mask")
        return
    
    type_suffix = get_supply_type_prefix(graph)
    
    stats =  {
        'processed_count': 0,
        'inconsistency_count': 0, 
        'pattern_conflicts': 0
    }
    
    for node_id, node_ports in port_mapping.items():
        if not node_ports:
            continue
            
        _assign_attributes_to_node(
            graph, node_id, node_ports, df, node_config,
            type_suffix, stats, logger
        )
        
        stats['processed_count'] += 1
    
    logger.info(f"Assignment completed:")
    logger.info(f"  Processed nodes: {stats['processed_count']}")
    logger.info(f"  Within-pattern inconsistencies: {stats['inconsistency_count']}")
    logger.info(f"  Cross-pattern conflicts: {stats['pattern_conflicts']}")

def _assign_attributes_to_node(graph, node_id, node_ports, df, config, 
                              type_suffix, stats, logger):
    """Assign all attributes to a single node.
        config: {"attribute": {"port_suffix": "pattern_with_{pipe_code}"}}
                ex.: {"temperature": {"port_a": "networkModel.pipe{pipe_code}{type}.sta_a.T"}}
    """
    
    for attribute_name, port_patterns in config.items():
        """Collect values for a specific attribute from all relevant ports.
        attribute_name: e.g. "temperature"
        port_patterns: e.g. {"port_a": "networkModel.pipe{pipe_code}{type}.sta_a.T",
                        "port_b": "networkModel.pipe{pipe_code}{type}.sta_b.T"}
        """
        series_list = []
        for port in node_ports:
            
            pipe_name, port_suffix = _parse_port_identifier(port)
            if port_suffix in port_patterns:
                pattern = port_patterns[port_suffix]
                column_name = pattern.format(pipe_code=pipe_name, type=type_suffix)
            
                if column_name in df.columns:
                    series = df[column_name]
                    series_list.append(series)
                else:
                    logger.debug(f"Column not found: {column_name}")

        if len(series_list) == 0:
            logger.debug(f"No data found for {attribute_name} at node {node_id}")
            continue

        # Check if all series are identical
        if len(series_list) > 1:
            all_equal = all(series_list[0].equals(series) for series in series_list[1:])
            if not all_equal:
                logger.warning(f"Node {node_id}: Inconsistent {attribute_name} time series found")
                stats['inconsistency_count'] += 1
        
        # Use first series as result
        graph.nodes[node_id][attribute_name] = series_list[0]        

def _parse_port_identifier(port):
    """Parse port identifier to extract pipe name and port suffix."""
    port_parts = port.split(".")
    if len(port_parts) < 2:
        raise ValueError(f"Invalid port format: {port}. Expected 'pipe_name.port_suffix'")
    return port_parts[0].replace("pipe", ""), port_parts[1]

##### Assign edge data

def assign_edge_data(graph, MASK, df):
    type_suffix = get_supply_type_prefix(graph)
    for edge in graph.edges:
        for edge_variable, variable_mask in MASK["edge"].items():
            pipe_name = graph.edges[edge]["name"]
            variable_name = variable_mask.format(pipe_code=pipe_name, type=type_suffix)
            graph.edges[edge][edge_variable] = df[variable_name]


def derive_specific_pressure_loss(graph):
    """Computes dp_spec [Pa/m] = dp / length for each edge and stores it on the graph."""
    for edge in graph.edges:
        dp = graph.edges[edge]["dp"]
        length = graph.edges[edge]["length"]
        graph.edges[edge]["dp_spec"] = dp / length


def assign_edge_loss(graph, MASK, df, logger=None):
    """Assign the OPTIONAL per-pipe heat loss (edge_optional) onto edges.

    Writes graph.edges[e]["Q_loss"] (pd.Series, W; >0 = loss) from the pipe's
    own heatPort.Q_flow. GUARDED: a missing column degrades to a debug log
    instead of raising, so a model/cache without heatPort logging is tolerated.

    Returns the number of (edge, variable) assignments made.
    """
    if logger is None:
        logger = set_up_terminal_logger(f"{__name__}.assign_edge_loss")
    opt = MASK.get("edge_optional")
    if not opt:
        return 0
    type_suffix = get_supply_type_prefix(graph)
    assigned = 0
    missing = 0
    for edge in graph.edges:
        pipe_name = graph.edges[edge]["name"]
        for edge_variable, variable_mask in opt.items():
            # Return-line loss only on a supply graph (avoid double counting / a
            # non-existent pipe<code>RR on a return graph).
            if edge_variable == "Q_loss_return" and type_suffix == "R":
                continue
            column = variable_mask.format(pipe_code=pipe_name, type=type_suffix)
            if column in df.columns:
                graph.edges[edge][edge_variable] = df[column]
                assigned += 1
            else:
                missing += 1
                logger.debug(f"Heat-loss column not found for pipe '{pipe_name}': {column}")
    logger.info(f"Assigned heat loss to {assigned} edge slots "
                f"({missing} missing)")
    return assigned


##### Assign building (demand) power

CP_WATER_DEFAULT = 4184.0  # J/(kg*K), AixLib water default for realized-power reconstruction
RHO_WATER_DEFAULT = 983.0  # kg/m3, ~60 degC; for ideal-hydraulic pump power V*dp


def assign_demand_power(graph, df, MASK, realized=False, cp=CP_WATER_DEFAULT, logger=None):
    """Assign building thermal power onto demand nodes as a time series.

    Writes graph.nodes[n]["heat_power_prescribed"] (pd.Series, W) for every
    non-supply building node carrying a "name", read from the flat network-model
    RealInput networkModel.<name>Q_flow_input. With realized=True, also writes
    graph.nodes[n]["heat_power_realized"] = m_flow * cp * (T_supply - T_return),
    reconstructed from the substation sensors (no prebuilt realized-Q variable
    exists in the AixLib demand models).

    Keyed by node["name"] (the demand-instance name), NOT by pipe code - which is
    why the per-pipe assign_node_values cannot be reused here.

    Args:
        graph: uesgraphs object.
        df: DataFrame with simulation columns (already time-indexed).
        MASK: mask dict; the "building" section drives the patterns.
        realized: also reconstruct realized delivered power.
        cp: specific heat capacity [J/(kg*K)] for the realized reconstruction.
        logger: optional logger.
    """
    if logger is None:
        logger = set_up_terminal_logger(f"{__name__}.assign_demand_power")

    bldg_mask = MASK.get("building")
    if not bldg_mask:
        logger.warning("No 'building' section in MASK; skipping demand power.")
        return

    pattern = bldg_mask["heat_power_prescribed"]
    tret_pattern = bldg_mask.get("temperature_return")
    realized_patterns = bldg_mask.get("_realized", {})
    assigned = 0
    tret_assigned = 0
    realized_assigned = 0

    for node in graph.nodelist_building:
        if graph.nodes[node].get("is_supply_heating"):
            continue
        name = graph.nodes[node].get("name")
        if name is None:
            continue

        column = pattern.format(name=name)
        if column in df.columns:
            graph.nodes[node]["heat_power_prescribed"] = df[column]
            assigned += 1
        else:
            logger.debug(f"Prescribed power column not found for '{name}': {column}")

        # Substation return temperature [K] -> building node (fixed TReturn in the
        # open-loop model, so a building quantity rather than a pipe/node value).
        if tret_pattern:
            tcol = tret_pattern.format(name=name)
            if tcol in df.columns:
                graph.nodes[node]["temperature_return"] = df[tcol]
                tret_assigned += 1
            else:
                logger.debug(f"Return-temp column not found for '{name}': {tcol}")

        if realized and realized_patterns:
            try:
                t_supply = df[realized_patterns["T_supply"].format(name=name)]
                t_return = df[realized_patterns["T_return"].format(name=name)]
                m_flow = df[realized_patterns["m_flow"].format(name=name)]
                graph.nodes[node]["heat_power_realized"] = m_flow * cp * (t_supply - t_return)
                realized_assigned += 1
            except KeyError as missing:
                logger.debug(f"Realized power columns missing for '{name}': {missing}")

    logger.info(f"Assigned prescribed demand power to {assigned} building nodes")
    if tret_pattern:
        logger.info(f"Assigned return temperature to {tret_assigned} building nodes")
    if realized:
        logger.info(f"Assigned realized demand power to {realized_assigned} building nodes")


##### Assign supply (pump power + supply thermal power)

def assign_supply_values(graph, df, MASK, rho=RHO_WATER_DEFAULT, logger=None):
    """Assign supply-node quantities and derive the pump power.

    For every is_supply_heating node carrying a "name", writes the time series
    listed in the (injected) ``supply`` mask section - e.g. heat_power_supply
    [W], m_flow [kg/s], pressure_flow / pressure_return [Pa] - and derives:

      dp_pump [Pa]            = pressure_flow - pressure_return
      pump_power_hydraulic [W]= |m_flow| / rho * dp_pump      (mode "hydraulic")
        OR pump_power_native, read directly                   (mode "native")

    SourceIdeal is an ideal pressure source with no native pump-power output, so
    the hydraulic reconstruction is the ideal pumping power (NO efficiency). The
    pump-power mode comes from the supply mask entry, so a real-pump station can
    expose a logged variable instead (see SUPPLY_MASKS).

    GUARDED: missing columns degrade to a debug log instead of raising.

    Returns the number of supply nodes that received at least one value.
    """
    if logger is None:
        logger = set_up_terminal_logger(f"{__name__}.assign_supply_values")
    sup = MASK.get("supply")
    if not sup:
        logger.debug("No 'supply' section in MASK; skipping supply values.")
        return 0

    vars_ = sup.get("vars", {})
    pump_spec = sup.get("pump_power", {})
    pump_mode = pump_spec.get("mode", "hydraulic")
    nodes_touched = 0
    pump_assigned = 0

    for node in graph.nodelist_building:
        if not graph.nodes[node].get("is_supply_heating"):
            continue
        name = graph.nodes[node].get("name")
        if name is None:
            continue

        present = {}
        for var, pattern in vars_.items():
            column = pattern.format(name=name)
            if column in df.columns:
                graph.nodes[node][var] = df[column]
                present[var] = df[column]
            else:
                logger.debug(f"Supply column not found for '{name}': {column}")
        if present:
            nodes_touched += 1

        # Pressure rise across the supply (pump head).
        if "pressure_flow" in present and "pressure_return" in present:
            dp_pump = present["pressure_flow"] - present["pressure_return"]
            graph.nodes[node]["dp_pump"] = dp_pump
        else:
            dp_pump = None

        # Pump power.
        if pump_mode == "native":
            native_var = pump_spec.get("var")
            if native_var and native_var in present:
                graph.nodes[node]["pump_power"] = present[native_var]
                pump_assigned += 1
            else:
                logger.debug(f"Native pump-power var missing for supply '{name}'")
        else:  # "hydraulic": ideal pumping power V*dp = |m_flow|/rho * dp_pump
            if dp_pump is not None and "m_flow" in present:
                graph.nodes[node]["pump_power_hydraulic"] = (
                    present["m_flow"].abs() / rho * dp_pump
                )
                pump_assigned += 1
            else:
                logger.debug(f"Cannot derive hydraulic pump power for supply '{name}' "
                             f"(need m_flow + both pressures)")

    logger.info(f"Assigned supply values to {nodes_touched} supply nodes "
                f"({pump_assigned} with pump power, mode='{pump_mode}')")
    return nodes_touched


##### Validation functions

def validate_edge_attributes(graph, edge_attributes, reference_df, logger=None):
    """
    Validates graph edge attributes against a reference DataFrame.
    
    Checks if the length of attribute arrays in edges matches the 
    number of rows in the reference DataFrame.
    
    Args:
        graph: uesgraphs
        edge_attributes (dict): Dictionary containing edge attributes to validate
        reference_df (pd.DataFrame): Reference DataFrame for length comparison
        logger (logging.Logger, optional): Logger instance. If None, a 
                                         terminal logger will be created.
    
    Returns:
        bool: True if all validations pass, False otherwise
        
    Raises:
        ValueError: For critical validation errors
    """
    if logger is None:
        logger = set_up_terminal_logger(f"{__name__}.validate_edge_attributes")
    
    logger.info("Starting edge attribute validation...")
    
    expected_length = reference_df.shape[0]
    validation_passed = True
    errors = []
    
    # Validate edge attributes
    logger.info(f"Validating edge attributes for {len(graph.edges)} edges...")
    
    if edge_attributes:
        for edge_idx, edge in enumerate(graph.edges):
            for edge_attr in edge_attributes.keys():
                if edge_attr in graph.edges[edge]:
                    actual_length = len(graph.edges[edge][edge_attr])
                    
                    if actual_length != expected_length:
                        error_msg = (
                            f"Edge {edge} - Attribute '{edge_attr}': "
                            f"length {actual_length} != expected length {expected_length}"
                        )
                        logger.error(error_msg)
                        errors.append(error_msg)
                        validation_passed = False
                    else:
                        logger.debug(
                            f"Edge {edge} - Attribute '{edge_attr}': OK "
                            f"(length: {actual_length})"
                        )
                else:
                    warning_msg = f"Edge {edge} - Attribute '{edge_attr}' not found"
                    logger.warning(warning_msg)
    else:
        logger.warning("No edge attributes provided for validation")
    
    # Validation summary
    if validation_passed:
        logger.info("SUCCESS: Edge attribute validation completed successfully")
    else:
        logger.error(f"FAILED: Edge attribute validation failed: {len(errors)} errors")
        for error in errors[:5]:  # Show maximum 5 errors in summary
            logger.error(f"  - {error}")
        if len(errors) > 5:
            logger.error(f"  ... and {len(errors) - 5} more errors")
    
    return validation_passed

def validate_node_attributes(graph, node_attributes, reference_df, logger=None):
    """
    Validates graph node attributes against a reference DataFrame.
    
    Checks if the length of attribute arrays in nodes matches the 
    number of rows in the reference DataFrame.
    
    Args:
        graph: uesgraphs
        node_attributes (dict): Dictionary containing node attributes to validate
        reference_df (pd.DataFrame): Reference DataFrame for length comparison
        logger (logging.Logger, optional): Logger instance. If None, a 
                                         terminal logger will be created.
    
    Returns:
        bool: True if all validations pass, False otherwise
        
    Raises:
        ValueError: For critical validation errors
    """
    if logger is None:
        logger = set_up_terminal_logger(f"{__name__}.validate_node_attributes")
    
    logger.info("Starting node attribute validation...")
    
    expected_length = reference_df.shape[0]
    validation_passed = True
    errors = []
    
    # Validate node attributes
    logger.info(f"Validating node attributes for {len(graph.nodes)} nodes...")
    
    if node_attributes:
        for node_idx, node in enumerate(graph.nodes):
            for node_attr in node_attributes.keys():
                if node_attr in graph.nodes[node]:
                    actual_length = len(graph.nodes[node][node_attr])
                    
                    if actual_length != expected_length:
                        error_msg = (
                            f"Node {node} - Attribute '{node_attr}': "
                            f"length {actual_length} != expected length {expected_length}"
                        )
                        logger.error(error_msg)
                        errors.append(error_msg)
                        validation_passed = False
                    else:
                        logger.debug(
                            f"Node {node} - Attribute '{node_attr}': OK "
                            f"(length: {actual_length})"
                        )
                else:
                    warning_msg = f"Node {node} - Attribute '{node_attr}' not found"
                    logger.warning(warning_msg)
    else:
        logger.warning("No node attributes provided for validation")
    
    # Validation summary
    if validation_passed:
        logger.info("SUCCESS: Node attribute validation completed successfully")
    else:
        logger.error(f"FAILED: Node attribute validation failed: {len(errors)} errors")
        for error in errors[:5]:  # Show maximum 5 errors in summary
            logger.error(f"  - {error}")
        if len(errors) > 5:
            logger.error(f"  ... and {len(errors) - 5} more errors")
    
    return validation_passed


def assess_dp_quality(graph, 
                      negligible_abs_threshold=1.0, 
                      negligible_rel_threshold=0.001,
                      acceptable_abs_threshold=10.0, 
                      acceptable_rel_threshold=0.01):
    """
    Assesses the quality of node assignments based on pressure and pressure difference
    
    Parameters:
    -----------
    graph : uesgraph
        The district heating network graph object
    negligible_abs_threshold : float, default=1.0
        Absolute threshold for negligible deviations (Pa)
    negligible_rel_threshold : float, default=0.001
        Relative threshold for negligible deviations (0.1%)
    acceptable_abs_threshold : float, default=10.0
        Absolute threshold for acceptable deviations (Pa)
    acceptable_rel_threshold : float, default=0.01
        Relative threshold for acceptable deviations (1%)
    
    Returns:
    --------
    dict
        Dictionary with categories 'negligible', 'acceptable', 'investigate'
        and corresponding edges with timestamp information
    """
    stats = {
        'negligible': [],
        'acceptable': [], 
        'investigate': []
    }
    
    for edge in graph.edges:
        node1, node2 = list(edge)
        p1 = graph.nodes[node1]["pressure"]
        p2 = graph.nodes[node2]["pressure"]
        dp_calc = p1 - p2
        dp_sim = graph.edges[edge]["dp"]
        
        # Check for each timestamp
        for i in range(len(p1)):
            timestamp = p1.index[i] if hasattr(p1, 'index') else i
            
            abs_diff = abs(dp_calc.iloc[i] - dp_sim.iloc[i])
            rel_error = (abs_diff / abs(dp_sim.iloc[i]) 
                        if dp_sim.iloc[i] != 0 else float('inf'))
            
            edge_info = {
                'edge': edge,
                'timestamp': timestamp,
                'abs_diff': abs_diff,
                'rel_error': rel_error,
                'dp_calculated': dp_calc.iloc[i],
                'dp_simulated': dp_sim.iloc[i]
            }
            
            # Categorization based on thresholds
            if (abs_diff < negligible_abs_threshold or 
                rel_error < negligible_rel_threshold):
                stats['negligible'].append(edge_info)
                
            elif (abs_diff < acceptable_abs_threshold and 
                  rel_error < acceptable_rel_threshold):
                stats['acceptable'].append(edge_info)
                
            else:
                stats['investigate'].append(edge_info)
    
    return stats


## Final pipeline function


def assign_data_pipeline(
    graph: ug.UESGraph,
    simulation_data_path: Union[str, Path], 
    start_date: datetime,
    end_date: datetime,
    time_interval: str,
    MASK: Optional[Dict[str, str]] = None,
    aixlib_version: str = "2.1.0",
    system_model_path: Optional[Union[str, Path]] = None,
    node_to_port_mapping: Optional[Dict] = None,
    with_demand_power: bool = True,
    demand_power_realized: bool = False,
    with_heat_loss: bool = True,
    with_pump_power: bool = True,
    supply_model: str = DEFAULT_SUPPLY_MODEL,
    rho: float = RHO_WATER_DEFAULT,
    logger: Optional[logging.Logger] = None
) -> ug.UESGraph:
    """
    Assign simulation data to a UESGraph network.
    
    This function processes simulation results and assigns time series data
    to network components (nodes and edges). It supports two modes:
    
    1. **Full assignment** (with node data): Requires either `node_to_port_mapping`
       or `system_model_path` to map simulation variables to graph nodes
    2. **Edge-only assignment**: When no mapping is available, only assigns
       data to edges (mass flows, pressure drops)
    
    Args:
        graph: UESGraph instance to assign data to
        simulation_data_path: Path to simulation results (.mat or .parquet)
        start_date: Start date for data processing
        end_date: End date for data processing  
        time_interval: Time interval for resampling (e.g., "15min", "1H")
                      No default - user must specify explicitly
        MASK: Custom variable name masks. If None, uses AixLib standard masks
        aixlib_version: AixLib version for standard masks (default: "2.1.0")
        system_model_path: Path to system model JSON (for creating port mapping)
        node_to_port_mapping: Pre-computed mapping from nodes to simulation ports
        with_demand_power: If True (default), also assign building thermal power to
                      building nodes from the MASK "building" section (keyed by
                      building name). Missing columns degrade to a warning.
        demand_power_realized: If True, additionally reconstruct realized delivered
                      power (m_flow*cp*dT) per substation from the sensor columns.
        with_heat_loss: If True (default), assign the per-pipe heat loss
                      (edge["Q_loss"], W) from the pipe's heatPort.Q_flow. Missing
                      columns degrade to a warning (no exception); see assign_edge_loss.
        with_pump_power: If True (default), assign supply-node quantities
                      (heat_power_supply, m_flow, pressures) and derive the pump
                      power (pump_power_hydraulic by default). Missing columns
                      degrade to a warning; see assign_supply_values.
        supply_model: Supply-station model key for SUPPLY_MASKS (default
                      "OpenLoop.SourceIdeal"). Per supply node this is overridden
                      by the system model's comp_model when available, so the
                      supply mask is EXCHANGEABLE per station.
        rho: water density [kg/m3] for the ideal-hydraulic pump power V*dp.
        logger: Logger instance. If None, creates a new file logger
        
    Returns:
        UESGraph instance with assigned simulation data
        
    Raises:
        FileNotFoundError: If simulation_data_path doesn't exist
        ValueError: If graph has no name set or data validation fails
        KeyError: If required simulation variables are missing
        
    Notes:
        - Either `node_to_port_mapping` OR `system_model_path` is required 
          for full data assignment including nodes
        - If both are None, only edge data (mass flows) will be assigned
        - Graph must have a name set in graph.graph["name"]
        
    Example:
        >>> import uesgraphs as ug
        >>> from datetime import datetime
        >>> 
        >>> # Load your network
        >>> graph = ug.UESGraph()  
        >>> graph.from_json("network.json", network_type="heating")
        >>> graph.graph["name"] = "my_network"
        >>> 
        >>> # Assign simulation data
        >>> graph_with_data = assign_data_pipeline(
        ...     graph=graph,
        ...     simulation_data_path="results.mat",
        ...     start_date=datetime(2024, 1, 1),
        ...     end_date=datetime(2024, 1, 7), 
        ...     time_interval="15min",
        ...     system_model_path="system_model.json"
        ... )
        >>> 
        >>> # With custom masks
        >>> custom_masks = {
        ...     "m_flow": "custom.pipe{pipe_code}.flow",
        ...     "p_a": "custom.pipe{pipe_code}.pressure_in", 
        ...     "p_b": "custom.pipe{pipe_code}.pressure_out"
        ... }
        >>> graph_with_data = assign_data_pipeline(
        ...     graph=graph,
        ...     simulation_data_path="results.mat", 
        ...     start_date=datetime(2024, 1, 1),
        ...     end_date=datetime(2024, 1, 7),
        ...     time_interval="1H",
        ...     MASK=custom_masks
        ... )
    """
    # Set up logging
    if logger is None:
        logger = set_up_file_logger("assign_data_pipeline", level=logging.INFO)
    
    # Convert simulation_data_path to Path object  
    simulation_data_path = Path(simulation_data_path)
    
    # Validate graph has a name
    network_name = graph.graph.get("name")
    if not network_name:
        raise ValueError("Graph must have a name set in graph.graph['name']")
    
    # Check supply type from graph
    supply_type = graph.graph.get("supply_type", "supply")
    supply_type_prefix = {"supply": "", "return": "R"}
    
    logger.info("="*70)
    logger.info("STARTING DATA ASSIGNMENT PIPELINE")
    logger.info("="*70)
    logger.info(f"Network: {network_name}")
    logger.info(f"Graph: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
    logger.info(f"Simulation data: {simulation_data_path}")
    logger.info(f"Time period: {start_date} to {end_date}")
    logger.info(f"Time interval: {time_interval}")
    logger.info(f"Supply type: {supply_type}")
    
    # Determine assignment mode based on available mapping options
    assignment_mode = _determine_assignment_mode(
        system_model_path, node_to_port_mapping, logger
    )
    
    try:
        # Step 1: Determine variable masks
        logger.info("Step 1/6: Setting up variable masks")
        if MASK is None:
            MASK = AIXLIB_MASKS[aixlib_version]
            logger.info(f"SUCCESS: Using AixLib {aixlib_version} standard masks")
        else:
            logger.info("SUCCESS: Using custom variable masks")
        
        # Step 2: Create or use port mapping (if available)
        port_mapping = None
        supply_comp_model = None  # detected from the system model, drives supply mask
        if assignment_mode == "full":
            logger.info("Step 2/6: Setting up port mapping for node assignment")

            if node_to_port_mapping is not None:
                port_mapping = node_to_port_mapping
                logger.info("SUCCESS: Using provided node-to-port mapping")

            elif system_model_path is not None:
                system_model_path = Path(system_model_path)
                if not system_model_path.exists():
                    raise FileNotFoundError(f"System model file not found: {system_model_path}")

                sysm_graph = ut.load_system_model_from_json(str(system_model_path))
                port_mapping = graph_transformation.map_system_model_to_uesgraph(sysm_graph, graph)
                logger.info(f"SUCCESS: Created port mapping from system model ({len(port_mapping)} components)")
                supply_comp_model = _detect_supply_comp_model(sysm_graph, logger)

        elif assignment_mode == "edges_only":
            logger.info("Step 2/6: Skipping port mapping - edge-only assignment mode")
            logger.warning("WARNING: No node data will be assigned (temperatures, pressures)")

        # Resolve the EXCHANGEABLE supply mask (pump power + supply thermal power).
        # The system model's comp_model wins; otherwise the supply_model argument.
        # Never mutate the shared AIXLIB_MASKS dict - inject into a shallow copy.
        if with_pump_power and "supply" not in MASK:
            supply_entry, supply_key = resolve_supply_mask(
                comp_model=supply_comp_model, supply_model=supply_model)
            if supply_entry is not None:
                MASK = {**MASK, "supply": supply_entry}
                logger.info(f"SUCCESS: Using supply mask '{supply_key}' "
                            f"(comp_model={supply_comp_model})")
            else:
                logger.warning(f"WARNING: No supply mask for model "
                               f"'{supply_comp_model or supply_model}'; pump power skipped")

        # Step 3: Process simulation data
        logger.info("Step 3/6: Processing simulation data")
        
        # Check if simulation data file exists and convert .mat files if needed
        if not simulation_data_path.exists():
            raise FileNotFoundError(f"Simulation data file not found: {simulation_data_path}")
        
        # Resolve/convert to the parquet cache (handles .mat -> .parquet) and get
        # the processed file path
        processed_simulation_path = check_input_file(file_path=str(simulation_data_path), logger=logger)
        logger.info(f"SUCCESS: Using processed simulation file: {processed_simulation_path}")
        
        # Build filter list for required variables. Graph+mask only, so it is
        # stable across a cache rebuild and computed once, outside the retry.
        filter_list_pipe = build_filter_list_pipe(graph, mask=MASK, logger=logger)
        logger.info(f"SUCCESS: Built filter list with {len(filter_list_pipe)} variables")

        # Building (demand) power columns - keyed by building name, not pipe code.
        demand_filter = (build_filter_list_demand(
            graph, MASK, realized=demand_power_realized, logger=logger)
            if with_demand_power else [])

        # OPTIONAL columns (heat loss per pipe, supply quantities). Like the
        # demand columns these are NOT required: missing ones degrade to a
        # warning instead of raising, so an older model/cache still works.
        loss_filter = (build_filter_list_loss(graph, MASK, logger=logger)
                       if with_heat_loss else [])
        supply_filter = (build_filter_list_supply(graph, MASK, logger=logger)
                         if with_pump_power else [])

        # Self-heal: a fast cache that predates an AIXLIB_MASKS extension can miss a
        # required (mask-derived) pipe column. On the first KeyError, force a
        # reconvert with the *current* masks and retry once — requested columns
        # always come from AIXLIB_MASKS, so the rebuilt (still lean) cache has them.
        df = None
        for attempt in (1, 2):
            try:
                filter_list = list(filter_list_pipe)

                # Optional column groups (demand power, per-pipe heat loss, supply
                # quantities) may be absent. Intersect each with the available
                # columns and degrade missing ones to a warning - never required.
                if demand_filter or loss_filter or supply_filter:
                    available_columns = set(
                        pq.ParquetFile(processed_simulation_path).schema.names)

                    def _add_optional(group, label):
                        if not group:
                            return
                        present = [c for c in group if c in available_columns]
                        missing = [c for c in group if c not in available_columns]
                        if missing:
                            logger.warning(
                                f"WARNING: {len(missing)}/{len(group)} {label} columns "
                                f"not in data file; skipped. Rebuild the cache if they "
                                f"were just added. e.g. {missing[:3]}")
                        filter_list.extend(present)
                        logger.info(f"SUCCESS: Added {len(present)} {label} columns")

                    _add_optional(demand_filter, "demand-power")
                    _add_optional(loss_filter, "heat-loss")
                    _add_optional(supply_filter, "supply")

                # Validate that all required columns exist in the processed file.
                column_validation = validate_columns_exist(
                    file_path=processed_simulation_path,
                    required_columns=filter_list,
                    logger=logger
                )
                if not column_validation:
                    raise KeyError("Required simulation variables not found in data file")

                # Load and process simulation results.
                df = process_simulation_result(
                    file_path=processed_simulation_path,
                    filter_list=filter_list,
                    logger=logger
                )
                break
            except KeyError:
                mat_sibling = Path(
                    os.path.splitext(str(simulation_data_path))[0] + ".mat")
                if attempt == 2 or not mat_sibling.exists():
                    raise
                logger.warning(
                    "Required column missing from cache -> forcing a reconvert with "
                    "current masks and retrying once.")
                processed_simulation_path = check_input_file(
                    file_path=str(simulation_data_path), force_reconvert=True,
                    logger=logger)
        logger.info(f"SUCCESS: Loaded simulation data: {df.shape[0]} timesteps")
        
        # Step 4: Prepare DataFrame
        logger.info("Step 4/6: Preparing time series data")
        df = prepare_DataFrame(
            df,
            base_date=start_date,
            start_date=start_date,
            end_date=end_date,
            time_interval=time_interval,
            logger=logger
        )
        logger.info(f"SUCCESS: Prepared DataFrame: {df.shape[0]} timesteps after filtering")
        
        # Step 5: Assign data to graph components
        logger.info("Step 5/6: Assigning data to graph components")
        
        if assignment_mode == "full":
            # Assign values to nodes (temperature and pressure)
            assign_node_values(graph, df, port_mapping, MASK, logger=logger)
            logger.info(f"SUCCESS: Assigned node data (temperature, pressure) to {len(graph.nodes)} nodes")
        
        # Assign values to edges (mass flow, pressure drop, velocity) - always done
        assign_edge_data(graph, MASK, df)
        derive_specific_pressure_loss(graph)
        logger.info(f"SUCCESS: Assigned edge data (m_flow, dp, velocity, dp_spec) to {len(graph.edges)} edges")

        # Assign building (demand) power to building nodes (keyed by building name)
        if with_demand_power:
            assign_demand_power(graph, df, MASK,
                                realized=demand_power_realized, logger=logger)

        # Assign optional per-pipe heat loss onto edges (guarded; no-op if absent)
        if with_heat_loss:
            assign_edge_loss(graph, MASK, df, logger=logger)

        # Assign supply quantities + derive pump power onto supply nodes (guarded)
        if with_pump_power:
            assign_supply_values(graph, df, MASK, rho=rho, logger=logger)

        # Step 6: Validate results
        logger.info("Step 6/6: Validating assignment results")
        
        validate_edge_attributes(graph, MASK["edge"], df, logger=logger)
        logger.info("SUCCESS: Edge validation completed successfully")

        if assignment_mode == "full":
            validate_node_attributes(graph,MASK["node"],df, logger=logger)

            ## Additional test to asses node assignment based on pressures
            stats = assess_dp_quality(graph) 
            if len(stats['investigate']) > 1:
                logger.warning("WARNING: Critical pressure difference deviations found!")
            else:
                logger.info("SUCCESS: Full validation completed successfully")

        
        logger.info("="*70)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info(f"SUCCESS: Network '{network_name}' ready for analysis")
        logger.info(f"SUCCESS: Data period: {df.index.min()} to {df.index.max()}")
        logger.info(f"SUCCESS: Time resolution: {time_interval}")
        logger.info(f"SUCCESS: Assignment mode: {assignment_mode}")
        logger.info("="*70)
        
        return graph
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        logger.error("="*70)
        raise


def _detect_supply_comp_model(sysm_graph, logger=None):
    """Read the supply station's ``comp_model`` from the system-model graph.

    Used to auto-select the EXCHANGEABLE supply mask (SUPPLY_MASKS). Scans the
    system model for a heating-supply node and returns its comp_model string
    (e.g. "AixLib...Supplies.OpenLoop.SourceIdeal"), or None if not found. If
    several supply nodes disagree, the first is used and a warning is logged
    (per-node masks are future work).
    """
    found = []
    try:
        for _, attrs in sysm_graph.nodes(data=True):
            if attrs.get("is_supply_heating") and attrs.get("comp_model"):
                found.append(attrs["comp_model"])
    except Exception as exc:  # pragma: no cover - defensive
        if logger:
            logger.debug(f"Could not read supply comp_model from system model: {exc}")
        return None
    if not found:
        return None
    if logger and len(set(found)) > 1:
        logger.warning(f"Multiple supply comp_models {set(found)}; using '{found[0]}'")
    return found[0]


def _determine_assignment_mode(
    system_model_path: Optional[Union[str, Path]],
    node_to_port_mapping: Optional[Dict],
    logger: logging.Logger
) -> str:
    """
    Determine the assignment mode based on available mapping options.
    
    Returns:
        "full": Full assignment including nodes (requires mapping)
        "edges_only": Only edge assignment (no node data)
    """
    if node_to_port_mapping is not None:
        logger.info("Port mapping provided - Full assignment mode")
        return "full"
    elif system_model_path is not None:
        logger.info("System model provided - Full assignment mode")
        return "full"
    else:
        logger.warning("No port mapping or system model - Edge-only assignment mode")
        logger.warning("Node temperatures and pressures will NOT be assigned")
        return "edges_only"


##### Helper functions

def get_supply_type_prefix(graph):
    supply_type = graph.graph.get("supply_type", "supply")
    supply_type_prefix = {"supply": "", "return": "R"}
    return supply_type_prefix.get(supply_type, "")
