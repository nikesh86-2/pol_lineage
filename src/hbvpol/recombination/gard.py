"""GARD wrapper and JSON parser.

GARD (Genetic Algorithm for Recombination Detection) is a HyPhy method.  It
scans an alignment for a genome-wide set of breakpoints (it does not identify
recombinant/parent pairs), so its rows use a scan-level identifier and leave
``partner`` empty.  As with the other tool wrappers, GARD is probed at call
time and its absence only produces a warning plus an empty frame.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ..config import get
from ..pipeline import StageError, get_logger, have_executable, run_command
from .partition import BREAKPOINT_COLUMNS

__all__ = ["run_gard", "parse_gard_json"]

logger = get_logger("recombination.gard")


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=BREAKPOINT_COLUMNS)


def _collect_breakpoints(node, acc: set[int]) -> None:
    """Recursively collect integer breakpoint positions from a GARD JSON tree."""
    if isinstance(node, dict):
        for key, value in node.items():
            if "breakpoint" in str(key).lower():
                if isinstance(value, (list, tuple)):
                    for item in value:
                        if isinstance(item, (int, float)) and not isinstance(item, bool):
                            if item >= 0:
                                acc.add(int(item))
                elif isinstance(value, dict):
                    for nested_key in value:
                        try:
                            position = int(float(nested_key))
                            if position >= 0:
                                acc.add(position)
                        except (TypeError, ValueError):
                            continue
            _collect_breakpoints(value, acc)
    elif isinstance(node, (list, tuple)):
        for item in node:
            _collect_breakpoints(item, acc)


def parse_gard_json(path: str | Path) -> pd.DataFrame:
    """Parse a HyPhy GARD JSON report into the tidy breakpoint schema.

    Breakpoint positions are located recursively under any key containing
    "breakpoint".  GARD does not assign parent partners, so ``partner`` is
    empty and rows are labelled with a scan-level ``recombinant_id``.
    """
    path = Path(path)
    if not path.exists():
        return _empty()
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (json.JSONDecodeError, OSError) as error:  # pragma: no cover - defensive
        logger.warning("could not read GARD JSON %s: %s", path, error)
        return _empty()

    positions: set[int] = set()
    _collect_breakpoints(data, positions)
    if not positions:
        return _empty()

    rows = [
        {
            "recombinant_id": "GARD",
            "partner": "",
            "tool": "gard",
            "bp_start": position,
            "bp_end": position,
            "support": 1.0,
            "region": "",
        }
        for position in sorted(positions)
    ]
    return pd.DataFrame(rows, columns=BREAKPOINT_COLUMNS)


def run_gard(alignment, config, workdir) -> pd.DataFrame:
    """Run HyPhy ``gard`` on ``alignment`` and return its breakpoints.

    Honours ``recombination.gard.rate_variation`` and
    ``recombination.gard.n_categories`` by passing the corresponding HyPhy
    options.  Fails soft when HyPhy is unavailable.
    """
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    alignment_path = Path(alignment)

    executable_name = str(get(config, "recombination.gard.executable", "hyphy") or "hyphy")
    if not (Path(executable_name).exists() or have_executable(executable_name)):
        logger.warning("HyPhy/GARD executable %r not found; skipping GARD", executable_name)
        return _empty()

    json_out = workdir / "gard.GARD.json"
    argv = [
        executable_name, "gard",
        "--alignment", str(alignment_path),
        "--output", str(json_out),
    ]
    rate_variation = get(config, "recombination.gard.rate_variation")
    n_categories = get(config, "recombination.gard.n_categories")
    if rate_variation:
        argv += ["--rate-variation", str(rate_variation)]
    if n_categories:
        argv += ["--rate-classes", str(int(n_categories))]

    try:
        run_command(argv, cwd=workdir, log_path=workdir / "gard.log", check=False)
    except StageError as error:  # pragma: no cover - external tool behaviour
        logger.warning("GARD invocation failed: %s", error)
        return _empty()

    if not json_out.exists():
        logger.warning("GARD produced no JSON at %s; skipping", json_out)
        return _empty()
    return parse_gard_json(json_out)
