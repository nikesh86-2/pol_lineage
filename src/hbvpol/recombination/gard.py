"""GARD wrapper and JSON parser.

GARD (Genetic Algorithm for Recombination Detection) is a HyPhy method.  It
scans an alignment for a genome-wide set of breakpoints (it does not identify
recombinant/parent pairs), so its rows use a scan-level identifier and leave
``partner`` empty.  As with the other tool wrappers, GARD is probed at call
time and its absence only produces a warning plus an empty frame.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pandas as pd

from ..config import get
from ..pipeline import StageError, get_logger, have_executable, run_command
from .partition import BREAKPOINT_COLUMNS

__all__ = ["run_gard", "parse_gard_json"]

logger = get_logger("recombination.gard")


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=BREAKPOINT_COLUMNS)


def _as_step(key: object) -> int:
    """Order GARD's string step/model keys numerically (``-1`` if non-numeric)."""
    try:
        return int(str(key))
    except (TypeError, ValueError):
        return -1


def _positions(value: object) -> list[int]:
    """Positive integer positions from a (possibly nested) GARD position list."""
    positions: set[int] = set()
    if not isinstance(value, (list, tuple)):
        return []
    for item in value:
        if isinstance(item, bool):
            continue
        if isinstance(item, (int, float)):
            if item > 0:
                positions.add(int(item))
        elif isinstance(item, (list, tuple)):
            for position in item:
                if isinstance(position, bool):
                    continue
                if isinstance(position, (int, float)) and position > 0:
                    positions.add(int(position))
    return sorted(positions)


def _improvement_breakpoints(improvements: object) -> list[int]:
    """Breakpoints of the selected model, from GARD's ``improvements``."""
    if not isinstance(improvements, dict) or not improvements:
        return []
    entry = improvements[max(improvements, key=_as_step)]
    if not isinstance(entry, dict):
        return []
    return _positions(entry.get("breakpoints"))


def _partition_breakpoints(breakpoint_data: object) -> list[int]:
    """Fallback: breakpoints from GARD's partition intervals.

    ``breakpointData`` is keyed by *partition*; each value's ``bps`` lists that
    partition's interval(s), and the partitions tile the alignment in order.  The
    breakpoints are therefore the end of every partition but the last.  Used only
    when ``improvements`` is absent (older GARD reports).
    """
    if not isinstance(breakpoint_data, dict) or not breakpoint_data:
        return []
    intervals: list[tuple[float, int]] = []
    for entry in breakpoint_data.values():
        bps = entry.get("bps") if isinstance(entry, dict) else None
        if not isinstance(bps, (list, tuple)):
            continue
        for interval in bps:
            if not isinstance(interval, (list, tuple)) or not interval:
                continue
            start, end = interval[0], interval[-1]
            if not isinstance(start, (int, float)) or isinstance(start, bool):
                continue
            if not isinstance(end, (int, float)) or isinstance(end, bool) or end <= 0:
                continue
            intervals.append((start, int(end)))
    if len(intervals) < 2:
        return []
    intervals.sort()
    return sorted({end for _, end in intervals[:-1]})


def _selected_breakpoints(data: object) -> list[int]:
    """Selected breakpoints from a parsed GARD JSON report.

    GARD's step-up procedure records each accepted model under ``improvements``
    (keyed by step), listing the breakpoints it added; the highest step is the
    selected model.  ``breakpointData`` holds the resulting partition intervals
    and ``siteBreakPointSupport`` the per-site score for the *next* candidate,
    so neither may be mined for breakpoint positions.
    """
    if not isinstance(data, dict):
        return []
    if "improvements" in data:
        return _improvement_breakpoints(data.get("improvements"))
    return _partition_breakpoints(data.get("breakpointData"))


def parse_gard_json(path: str | Path) -> pd.DataFrame:
    """Parse a HyPhy GARD JSON report into the tidy breakpoint schema.

    The breakpoints of the selected model are read from ``improvements`` (its
    highest step).  GARD does not assign parent partners, so ``partner`` is
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

    positions = _selected_breakpoints(data)
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
        for position in positions
    ]
    return pd.DataFrame(rows, columns=BREAKPOINT_COLUMNS)


def run_gard(alignment, config, workdir) -> pd.DataFrame:
    """Run HyPhy ``gard`` on ``alignment`` and return its breakpoints.

    Honours ``recombination.gard.rate_variation`` and
    ``recombination.gard.n_categories`` by passing the corresponding HyPhy
    options.  ``recombination.gard.max_breakpoints`` caps the step-up search and
    ``recombination.gard.timeout_s`` bounds the wall-clock run; because GARD
    spools its JSON after every breakpoint search, a capped or timed-out run
    still returns the best partitions found so far.  Fails soft when HyPhy is
    unavailable.
    """
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    # Absolute paths: run_command sets cwd=workdir, so relative inputs/outputs
    # would be re-rooted under it and HyPhy could not find the alignment.
    alignment_path = Path(alignment).resolve()

    executable_name = str(get(config, "recombination.gard.executable", "hyphy") or "hyphy")
    if not (Path(executable_name).exists() or have_executable(executable_name)):
        logger.warning("HyPhy/GARD executable %r not found; skipping GARD", executable_name)
        return _empty()

    json_out = (workdir / "gard.GARD.json").resolve()
    argv = [
        executable_name, "gard",
        "--alignment", str(alignment_path),
        "--output", str(json_out),
        # HyPhy aborts GARD on reversible-model numerical instability with an
        # internal ComputeBranchCache error; its documented remedy is to treat
        # those as warnings.  Without this GARD silently yields a 0-byte JSON.
        "ENV=TOLERATE_NUMERICAL_ERRORS=1;",
    ]
    rate_variation = get(config, "recombination.gard.rate_variation")
    n_categories = get(config, "recombination.gard.n_categories")
    if rate_variation:
        argv += ["--rate-variation", str(rate_variation)]
    if n_categories:
        argv += ["--rate-classes", str(int(n_categories))]
    max_breakpoints = get(config, "recombination.gard.max_breakpoints")
    if max_breakpoints not in (None, ""):
        argv += ["--max-breakpoints", str(int(max_breakpoints))]

    timeout_raw = get(config, "recombination.gard.timeout_s")
    try:
        timeout = float(timeout_raw)
    except (TypeError, ValueError):
        timeout = None
    if timeout is not None and timeout <= 0:
        timeout = None

    # Drop any report left by a previous run so a failed or timed-out invocation
    # cannot be mistaken for a stale success.
    json_out.unlink(missing_ok=True)

    try:
        run_command(
            argv, cwd=workdir, log_path=workdir / "gard.log", check=False, timeout=timeout
        )
    except subprocess.TimeoutExpired:
        # GARD spools its JSON after each breakpoint search, so a bounded run
        # still yields the best partitions found so far.
        logger.warning(
            "GARD exceeded recombination.gard.timeout_s=%s s; using the partial report",
            timeout,
        )
    except StageError as error:  # pragma: no cover - external tool behaviour
        logger.warning("GARD invocation failed: %s", error)

    if not json_out.exists():
        logger.warning("GARD produced no JSON at %s; skipping", json_out)
        return _empty()
    return parse_gard_json(json_out)
