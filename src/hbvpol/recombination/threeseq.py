"""3SEQ wrapper and output parser.

3SEQ detects recombination between triplets of sequences and reports the
recombinant and the two breakpoint positions together with a p-value.  The
wrapper is lazy: the ``3seq`` binary is located at call time and a missing tool
only yields a warning and an empty breakpoint frame.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..config import get
from ..pipeline import StageError, get_logger, have_executable, run_command
from .partition import BREAKPOINT_COLUMNS, normalise_breakpoints

__all__ = ["run_threeseq", "parse_threeseq_output"]

logger = get_logger("recombination.threeseq")


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=BREAKPOINT_COLUMNS)


def _parse_table(path: Path) -> pd.DataFrame:
    """Read a whitespace- or comma-delimited 3SEQ table, skipping comments."""
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = [line for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")]
    if not lines:
        return pd.DataFrame()
    import io

    buffer = io.StringIO("\n".join(lines))
    separator = "\t" if "\t" in lines[0] else ("," if "," in lines[0] else r"\s+")
    return pd.read_csv(buffer, sep=separator)


def parse_threeseq_output(path: str | Path) -> pd.DataFrame:
    """Parse a 3SEQ report into the tidy breakpoint schema.

    Comments (``#``) are ignored and the table is normalised through
    :func:`normalise_breakpoints`, which recognises 3SEQ's common column
    aliases.
    """
    path = Path(path)
    if not path.exists():
        return _empty()
    try:
        frame = _parse_table(path)
    except Exception as error:  # pragma: no cover - defensive
        logger.warning("could not parse 3SEQ output %s: %s", path, error)
        return _empty()
    return normalise_breakpoints(frame, tool="threeseq")


def run_threeseq(alignment, config, workdir) -> pd.DataFrame:
    """Run ``3seq`` on ``alignment`` and return its breakpoints.

    Fails soft when the executable is missing or the run/report fails.
    """
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    alignment_path = Path(alignment)

    executable_name = str(get(config, "recombination.threeseq.executable", "3seq") or "3seq")
    if not (Path(executable_name).exists() or have_executable(executable_name)):
        logger.warning("3SEQ executable %r not found; skipping 3SEQ", executable_name)
        return _empty()

    out_path = workdir / "threeseq_output.txt"
    argv = [executable_name, "-f", str(alignment_path), "-o", str(out_path)]

    try:
        run_command(argv, cwd=workdir, log_path=workdir / "threeseq.log", check=False)
    except StageError as error:  # pragma: no cover - external tool behaviour
        logger.warning("3SEQ invocation failed: %s", error)
        return _empty()

    if not out_path.exists():
        logger.warning("3SEQ produced no report at %s; skipping", out_path)
        return _empty()
    return parse_threeseq_output(out_path)
