"""3SEQ wrapper and output parser.

3SEQ (Boni et al.) detects recombination between triplets of sequences and
reports the child plus the two parent sequences with a p-value and breakpoint
positions.

Bioconda's ``3seq`` uses *attached* options and run modes, which differ from
RDP's 3SEQ interface:

* full analysis is ``3seq -full <alignment> [options]``;
* ``-t`` is the rejection threshold and must be attached (``-t0.05``);
* ``-id <name>`` prefixes the output files (``<name>.3s.rec.csv``);
* ``-f``/``-l`` mean *first/last nucleotide*, **not** input file.

The report is a CSV with a fixed schema::

    P_ACCNUM,Q_ACCNUM,C_ACCNUM,m,n,k,p,HS?,log(p),DS(p),DS(p),min_rec_length,breakpoints

``C_ACCNUM`` is the child (recombinant), ``P_ACCNUM``/``Q_ACCNUM`` the parents,
the second ``DS(p)`` the corrected p-value, and ``breakpoints`` the detected
segments (``"100-100 & 200-200"``).

Missing executable or a failed run yields an empty breakpoint frame with a
warning; the stage never hard-fails on this tool.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from ..config import get
from ..pipeline import StageError, get_logger, have_executable, run_command
from .partition import BREAKPOINT_COLUMNS, normalise_breakpoints

__all__ = ["run_threeseq", "parse_threeseq_output", "parse_breakpoints_field"]

logger = get_logger("recombination.threeseq")

_BREAKPOINT_PAIR_RE = re.compile(r"(\d+)\s*-\s*(\d+)")


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=BREAKPOINT_COLUMNS)


def parse_breakpoints_field(text: object) -> tuple[int, int]:
    """Parse 3SEQ's ``breakpoints`` field into an overall ``(start, end)`` span.

    3SEQ reports one or two segments as ``"100-100 & 200-200"``; the returned
    span is the minimum start to the maximum end, i.e. the full tract that
    differs from the parents.
    """
    pairs = _BREAKPOINT_PAIR_RE.findall(str(text or ""))
    if not pairs:
        return 1, 1
    starts = [int(start) for start, _ in pairs]
    ends = [int(end) for _, end in pairs]
    return min(starts), max(ends)


def _from_biondi_schema(frame: pd.DataFrame) -> pd.DataFrame:
    """Map bioconda 3SEQ's fixed-column report onto the tidy schema."""
    rows: list[dict] = []
    for _, row in frame.iterrows():
        bp_start, bp_end = parse_breakpoints_field(row.iloc[-1])
        # Column 10 is the second (multiple-testing-corrected) DS(p); fall back
        # to the raw p at column 6.  Kept as a p-value; support is tool-specific.
        support_column = 10 if len(row) > 10 else (6 if len(row) > 6 else None)
        support = 1.0
        if support_column is not None:
            try:
                support = float(row.iloc[support_column])
            except (TypeError, ValueError):
                support = 1.0
        rows.append({
            "recombinant_id": str(row.iloc[2]),
            "partner": f"{row.iloc[0]}|{row.iloc[1]}",
            "tool": "threeseq",
            "bp_start": bp_start,
            "bp_end": bp_end,
            "support": support,
            "region": "",
        })
    if not rows:
        return _empty()
    return pd.DataFrame(rows, columns=BREAKPOINT_COLUMNS)


def parse_threeseq_output(path: str | Path) -> pd.DataFrame:
    """Parse a 3SEQ report into the tidy breakpoint schema.

    The bioconda report (``C_ACCNUM`` columns) is mapped explicitly; anything
    else is passed through :func:`normalise_breakpoints`, which recognises the
    common column aliases.
    """
    path = Path(path)
    if not path.exists():
        return _empty()
    try:
        frame = pd.read_csv(path)
    except Exception as error:  # pragma: no cover - defensive
        logger.warning("could not parse 3SEQ output %s: %s", path, error)
        return _empty()
    if frame is None or len(frame) == 0:
        return _empty()
    columns = {str(column).strip().upper() for column in frame.columns}
    if {"P_ACCNUM", "Q_ACCNUM", "C_ACCNUM"} <= columns:
        return _from_biondi_schema(frame)
    return normalise_breakpoints(frame, tool="threeseq")


def run_threeseq(alignment, config, workdir) -> pd.DataFrame:
    """Run ``3seq -full`` on ``alignment`` and return its breakpoints.

    Fails soft when the executable is missing or the run/report fails.
    """
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    # run_command sets cwd=workdir, so the alignment path must be absolute.
    alignment_path = Path(alignment).resolve()

    executable_name = str(get(config, "recombination.threeseq.executable", "3seq") or "3seq")
    if not (Path(executable_name).exists() or have_executable(executable_name)):
        logger.warning("3SEQ executable %r not found; skipping 3SEQ", executable_name)
        return _empty()

    threshold = get(config, "recombination.threeseq.threshold", 0.05)
    threshold = float(threshold) if threshold not in (None, "") else 0.05
    run_id = str(get(config, "recombination.threeseq.id", "hbvpol") or "hbvpol")
    # Options must be attached (`-t0.05`), and the positional alignment comes
    # before the option list.
    argv = [executable_name, "-full", str(alignment_path), f"-t{threshold}", "-id", run_id]

    try:
        run_command(argv, cwd=workdir, log_path=workdir / "threeseq.log", check=False)
    except StageError as error:  # pragma: no cover - external tool behaviour
        logger.warning("3SEQ invocation failed: %s", error)
        return _empty()

    out_path = workdir / f"{run_id}.3s.rec.csv"
    if not out_path.exists():
        fallback = workdir / "3s.rec.csv"  # 3seq ignores -id in some builds
        if fallback.exists():
            out_path = fallback
        else:
            logger.warning("3SEQ produced no report at %s; skipping", out_path)
            return _empty()
    return parse_threeseq_output(out_path)
