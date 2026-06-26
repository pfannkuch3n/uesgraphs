"""Pre-convert Dymola/OpenModelica ``.mat`` results to the fast parquet cache.

The analysis app converts a ``.mat`` to ``<base>.parquet`` the first time a result
is opened — which is slow. This module exposes that same conversion so it can run
*ahead of time*, e.g. right after copying ``.mat`` files over from a simulation VM:

    python -m uesgraphs.analyze.convert <path> [--force] [--level N] [--all-vars]

``<path>`` may be a single ``.mat`` file, a ``Sim*`` directory, or any directory
(searched recursively for ``*.mat``). A cache that is already current (newer than
its ``.mat``) is skipped unless ``--force``.

Defaults match the app (mask-derived "fast" scope), so a pre-built cache is used
1:1 by :func:`uesgraphs.analyze.data_handling.data_handling.check_input_file`
without any re-conversion. ``--all-vars`` keeps every variable ("full" scope).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Optional

from uesgraphs.analyze.data_handling.data_handling import mask_keep_suffixes
from uesgraphs.analyze.data_handling.mat_handler import mat_to_parquet


def _cache_path(mat_path) -> Path:
    """The parquet cache that belongs to *mat_path* (``dsres.mat`` -> ``dsres.parquet``)."""
    return Path(mat_path).with_suffix(".parquet")


def cache_is_current(mat_path) -> bool:
    """True if a parquet cache exists and is at least as new as its ``.mat``."""
    cache = _cache_path(mat_path)
    return (cache.exists()
            and cache.stat().st_mtime >= Path(mat_path).stat().st_mtime)


def convert_mat(mat_path, *, force: bool = False, compression: str = "zstd",
                compression_level: int = 3, all_vars: bool = False) -> Path:
    """Convert one ``.mat`` to ``<base>.parquet`` and return the cache path.

    Skips (and returns the existing cache) when it is already current, unless
    *force*. ``all_vars=True`` keeps every variable (``keep_suffixes=None`` /
    "full" scope); otherwise only the mask-derived columns are kept.
    """
    mat_path = Path(mat_path)
    if not force and cache_is_current(mat_path):
        return _cache_path(mat_path)

    keep = None if all_vars else mask_keep_suffixes()
    base = str(mat_path.with_suffix(""))
    saved = mat_to_parquet(save_as=base, fname=str(mat_path), with_unit=False,
                           keep_suffixes=keep, compression=compression,
                           compression_level=compression_level)
    return Path(saved)


def find_mats(root) -> List[Path]:
    """All ``.mat`` files for *root*: the file itself if it is a ``.mat``, else
    every ``*.mat`` found recursively under the directory (sorted)."""
    root = Path(root)
    if root.is_file() and root.suffix.lower() == ".mat":
        return [root]
    return sorted(root.rglob("*.mat"))


def convert_tree(root, *, force: bool = False, compression: str = "zstd",
                 compression_level: int = 3, all_vars: bool = False,
                 verbose: bool = False):
    """Convert every ``.mat`` under *root*. Returns ``[(mat, cache, converted)]``
    where *converted* is False when a current cache let us skip the work."""
    results = []
    for mat in find_mats(root):
        skipped = (not force) and cache_is_current(mat)
        cache = convert_mat(mat, force=force, compression=compression,
                            compression_level=compression_level, all_vars=all_vars)
        results.append((mat, cache, not skipped))
        if verbose:
            print(f"{'convert' if not skipped else 'skip   '} {cache}")
    return results


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m uesgraphs.analyze.convert",
        description="Pre-convert Dymola .mat results to the fast .parquet cache.")
    parser.add_argument(
        "path",
        help="a .mat file, a Sim* dir, or a directory searched recursively for *.mat")
    parser.add_argument("--force", action="store_true",
                        help="reconvert even if a current cache exists")
    parser.add_argument("--level", type=int, default=3, dest="level",
                        help="zstd compression level (default 3; offline runs can "
                             "afford 9-15 for smaller files, reads stay fast)")
    parser.add_argument("--all-vars", action="store_true", dest="all_vars",
                        help="keep every variable (scope=full) instead of the "
                             "mask-derived fast subset")
    args = parser.parse_args(argv)

    root = Path(args.path)
    if not root.exists():
        parser.error(f"path does not exist: {root}")

    results = convert_tree(root, force=args.force, compression_level=args.level,
                           all_vars=args.all_vars, verbose=True)
    if not results:
        print(f"No .mat files found under {root}")
        return 1
    n_conv = sum(1 for _, _, conv in results if conv)
    print(f"Done: {n_conv} converted, {len(results) - n_conv} skipped "
          f"({'full' if args.all_vars else 'fast'} scope).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
