"""Phylogeny stage — per-block and per-domain trees, ancestral states.

Public entry point is :func:`hbvpol.phylogeny.pipeline.run`.
"""

from __future__ import annotations

from .align import (
    alignment_length,
    as_pairs,
    read_alignment,
    slice_alignment,
    translate_alignment_for_domain,
)
from .ancestral import parsimony_ancestral, reconstruct_ancestral
from .trees import (
    apply_rooting,
    infer_tree,
    neighbor_joining,
    p_distance_matrix,
    parse_tree_summary,
)

__all__ = [
    "as_pairs",
    "read_alignment",
    "alignment_length",
    "slice_alignment",
    "translate_alignment_for_domain",
    "p_distance_matrix",
    "neighbor_joining",
    "infer_tree",
    "apply_rooting",
    "parse_tree_summary",
    "parsimony_ancestral",
    "reconstruct_ancestral",
    "run",
]


def run(config, root):  # pragma: no cover - thin re-export
    from .pipeline import run as _run

    return _run(config, root)
