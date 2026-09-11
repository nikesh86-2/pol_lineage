"""Experimental fitness anchor stage.

Loads and annotates the 2024 HBV polymerase deep mutational scanning map and
applies the six mechanistic-target criteria to joined per-position evidence.

Public entry point is :func:`hbvpol.fitness.pipeline.run`.
"""

from __future__ import annotations

from .dms import (
    CRITERIA,
    DMS_COLUMNS,
    annotate_dms,
    intolerant_sites,
    load_dms,
    mechanistic_candidates,
    score_criteria,
)

__all__ = [
    "load_dms",
    "annotate_dms",
    "intolerant_sites",
    "score_criteria",
    "mechanistic_candidates",
    "DMS_COLUMNS",
    "CRITERIA",
    "run",
]


def run(config, root):  # pragma: no cover - thin re-export
    from .pipeline import run as _run

    return _run(config, root)
