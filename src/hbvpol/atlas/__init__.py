"""Residue-atlas stage.

Joins all upstream per-position scores into a single outer-joined atlas, applies
the mechanistic-target criteria, ranks validation targets, builds the conserved
interaction network and conformational-switch shortlist, and writes a
self-contained HTML report.

Public entry point is :func:`hbvpol.atlas.pipeline.run`.
"""

from __future__ import annotations

from .report import build_report, render_table
from .targets import (
    INTERACTION_COLUMNS,
    RANKED_TARGET_COLUMNS,
    SWITCH_COLUMNS,
    conformational_switch_candidates,
    interaction_network,
    rank_targets,
)

__all__ = [
    "rank_targets",
    "interaction_network",
    "conformational_switch_candidates",
    "build_report",
    "render_table",
    "RANKED_TARGET_COLUMNS",
    "INTERACTION_COLUMNS",
    "SWITCH_COLUMNS",
    "run",
]


def run(config, root):  # pragma: no cover - thin re-export
    from .pipeline import run as _run

    return _run(config, root)
