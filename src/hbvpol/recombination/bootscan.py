"""Bootscan-style sliding-window recombination scan.

Choice of implementation
------------------------
RDP5 ships a bootscan module but no standalone, portable bootscan binary is
guaranteed to exist on the target systems.  This module therefore defaults to a
**self-contained, pure-Python scan** that needs no external tool, no network and
no random bootstrapping, which keeps the recombination stage fully offline and
unit-testable.  If ``recombination.bootscan.executable`` is configured *and*
present, the external RDP5 bootscan is preferred and its report is parsed with
:func:`parse_bootscan_output`.

The pure-Python scan
--------------------
For each query sequence the full-length distance to every other sequence picks
a primary reference.  A window (``window`` nt, advanced by ``step``) is then
slid across the alignment; in each window the query's similarity to every other
sequence is computed.  Where the window's best-matching sequence changes away
from the primary reference and the best similarity is at least ``threshold``,
the change is recorded.  Consecutive swapped windows sharing a partner are
collapsed into a single breakpoint interval.

Similarity is the fraction of identical (non-gap) positions, so this is a
distance-based scan rather than a true bootstrap, and is intended as a
recombination *screen* consistent with the other tools' outputs.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..config import get
from ..io import GenomeRecord
from ..pipeline import get_logger, have_executable, run_command, StageError
from .partition import BREAKPOINT_COLUMNS, load_alignment, normalise_breakpoints

__all__ = ["run_bootscan", "bootscan_scan", "parse_bootscan_output", "window_similarity"]

logger = get_logger("recombination.bootscan")


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=BREAKPOINT_COLUMNS)


def window_similarity(a: str, b: str, start: int, end: int) -> float:
    """Fraction of identical non-gap positions between two windows."""
    left = a[start:end].upper()
    right = b[start:end].upper()
    if not left or len(left) != len(right):
        return 0.0
    compared = 0
    matches = 0
    for x, y in zip(left, right):
        if x == "-" or y == "-":
            continue
        compared += 1
        if x == y:
            matches += 1
    return matches / compared if compared else 0.0


def bootscan_scan(
    records: list[GenomeRecord],
    window: int = 500,
    step: int = 50,
    threshold: float = 0.70,
    replicates: int = 100,
    min_score_margin: float = 0.0,
    min_region_len: int = 0,
    min_support: float = 0.0,
) -> pd.DataFrame:
    """Run the pure-Python bootscan scan over an in-memory alignment.

    ``replicates`` is accepted for interface parity with RDP5's bootscan (it
    would govern bootstrap resampling); this deterministic implementation does
    not resample and therefore ignores it, which is recorded in the returned
    ``tool`` column as ``bootscan``.

    ``min_score_margin`` requires the swapped window's best similarity to beat
    the primary reference's similarity by at least that margin.  This is the key
    specificity guard for HBV: with thousands of near-identical genomes the
    nearest neighbour flips between equally-similar references, so a bare
    ``threshold`` produces thousands of spurious calls.  ``min_region_len`` and
    ``min_support`` then drop short or weakly supported regions.
    """
    if len(records) < 3:
        return _empty()
    width = min(len(record.seq) for record in records)
    window = min(window, width)
    if window <= 0 or step <= 0:
        return _empty()

    sequence_by_id = {record.id: record.seq.upper() for record in records}
    ids = [record.id for record in records]

    rows: list[dict] = []
    for query_id in ids:
        query = sequence_by_id[query_id]
        others = [other for other in ids if other != query_id]

        full_similarity = {
            other: window_similarity(query, sequence_by_id[other], 0, width)
            for other in others
        }
        primary = max(full_similarity, key=full_similarity.get)

        swapped_regions: list[dict] = []
        current: dict | None = None
        for start in range(0, max(1, width - window + 1), step):
            end = min(start + window, width)
            window_scores = {
                other: window_similarity(query, sequence_by_id[other], start, end)
                for other in others
            }
            best = max(window_scores, key=window_scores.get)
            score = window_scores[best]
            primary_score = window_scores.get(primary, 0.0)
            is_swap = (
                best != primary
                and score >= threshold
                and (score - primary_score) >= min_score_margin
            )
            if is_swap:
                if current is not None and current["partner"] == best:
                    current["end"] = end
                    current["scores"].append(score)
                else:
                    if current is not None:
                        swapped_regions.append(current)
                    current = {"partner": best, "start": start + 1, "end": end, "scores": [score]}
            else:
                if current is not None:
                    swapped_regions.append(current)
                    current = None
        if current is not None:
            swapped_regions.append(current)

        for region in swapped_regions:
            support = sum(region["scores"]) / len(region["scores"])
            region_len = region["end"] - region["start"] + 1
            if region_len < min_region_len or support < min_support:
                continue
            rows.append({
                "recombinant_id": query_id,
                "partner": region["partner"],
                "tool": "bootscan",
                "bp_start": int(region["start"]),
                "bp_end": int(region["end"]),
                "support": float(support),
                "region": "",
            })

    return pd.DataFrame(rows, columns=BREAKPOINT_COLUMNS)


def parse_bootscan_output(path: str | Path) -> pd.DataFrame:
    """Parse a bootscan report (RDP5 or this module's tidy TSV) into schema."""
    path = Path(path)
    if not path.exists():
        return _empty()
    try:
        frame = pd.read_csv(path, sep=None, engine="python", comment="#")
    except Exception as error:  # pragma: no cover - defensive
        logger.warning("could not parse bootscan output %s: %s", path, error)
        return _empty()
    return normalise_breakpoints(frame, tool="bootscan")


def run_bootscan(alignment, config, workdir) -> pd.DataFrame:
    """Run a bootscan-style scan and return breakpoints.

    Prefers a configured external bootscan executable when present; otherwise
    runs the fully offline pure-Python scan.  Never raises for tool absence.
    """
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    executable_name = get(config, "recombination.bootscan.executable")
    if executable_name and (Path(str(executable_name)).exists() or have_executable(str(executable_name))):
        # Absolute: run_command sets cwd=workdir (see the other tool wrappers).
        alignment_path = Path(alignment).resolve()
        out_path = (workdir / "bootscan_output.txt").resolve()
        try:
            run_command(
                [str(executable_name), "-f", str(alignment_path), "-o", str(out_path)],
                cwd=workdir,
                log_path=workdir / "bootscan.log",
                check=False,
            )
        except StageError as error:  # pragma: no cover - external tool behaviour
            logger.warning("external bootscan failed: %s", error)
        else:
            if out_path.exists():
                return parse_bootscan_output(out_path)

    records = load_alignment(alignment)
    window = int(get(config, "recombination.bootscan.window", 500) or 500)
    step = int(get(config, "recombination.bootscan.step", 50) or 50)
    replicates = int(get(config, "recombination.bootscan.replicates", 100) or 100)
    threshold = float(get(config, "recombination.bootscan.threshold", 0.70) or 0.70)
    min_score_margin = float(get(config, "recombination.bootscan.min_score_margin", 0.0) or 0.0)
    min_region_len = int(get(config, "recombination.bootscan.min_region_len", 0) or 0)
    min_support = float(get(config, "recombination.bootscan.min_support", 0.0) or 0.0)
    return bootscan_scan(
        records,
        window=window,
        step=step,
        threshold=threshold,
        replicates=replicates,
        min_score_margin=min_score_margin,
        min_region_len=min_region_len,
        min_support=min_support,
    )
