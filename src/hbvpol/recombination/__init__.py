"""Recombination stage — detection tools, reconciliation and partitioning.

Public entry point is :func:`hbvpol.recombination.pipeline.run`.
"""

from __future__ import annotations

from .bootscan import bootscan_scan, parse_bootscan_output, run_bootscan
from .gard import parse_gard_json, run_gard
from .partition import (
    BREAKPOINT_COLUMNS,
    extract_domain_subalignments,
    normalise_breakpoints,
    partition_alignment,
    reconcile_breakpoints,
)
from .rdp5 import parse_rdp5_output, run_rdp5
from .threeseq import parse_threeseq_output, run_threeseq

__all__ = [
    "run_rdp5",
    "parse_rdp5_output",
    "run_gard",
    "parse_gard_json",
    "run_threeseq",
    "parse_threeseq_output",
    "run_bootscan",
    "bootscan_scan",
    "parse_bootscan_output",
    "reconcile_breakpoints",
    "partition_alignment",
    "extract_domain_subalignments",
    "normalise_breakpoints",
    "BREAKPOINT_COLUMNS",
    "run",
]


def run(config, root):  # pragma: no cover - thin re-export
    from .pipeline import run as _run

    return _run(config, root)
