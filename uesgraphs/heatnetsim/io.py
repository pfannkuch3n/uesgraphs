"""IO helpers for HeatNetSim interop: load results CSVs / mapping JSON, write CSVs."""

import ast
import csv
import json
from pathlib import Path

import pandas as pd


def write_csv(path: Path, records: list, fieldnames: list, delimiter: str = ";") -> None:
    """Write *records* as CSV with *fieldnames* header, creating parent dirs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=delimiter)
        writer.writeheader()
        for record in records:
            writer.writerow({k: record.get(k, "") for k in fieldnames})


def write_heat_mapping(path: Path, heat_mapping: dict) -> None:
    """Serialise *heat_mapping* to JSON at *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(heat_mapping, f, indent=2)


def load_results(results) -> dict:
    """Accept a dict (already in memory) or a path to a runner-produced CSV.

    Each cell in the CSV is a stringified Python list, e.g. "[3.1e5, 2.9e5]".
    Scalar cells (e.g. summary columns) are passed through unchanged.
    """
    if isinstance(results, dict):
        return results

    df = pd.read_csv(Path(results))
    out = {}
    for col in df.columns:
        parsed = []
        for cell in df[col]:
            try:
                parsed.append(ast.literal_eval(str(cell)))
            except (ValueError, SyntaxError):
                parsed.append(cell)
        out[col] = parsed
    return out


def load_mapping(mapping_path) -> dict:
    with open(Path(mapping_path), "r") as f:
        return json.load(f)


def build_time_index(n: int, start_date=None, time_interval=None) -> pd.Index:
    if start_date is not None and time_interval is not None:
        return pd.date_range(start=start_date, periods=n, freq=time_interval)
    return pd.RangeIndex(n)
