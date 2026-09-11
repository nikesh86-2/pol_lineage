"""Selection stage — entropy, dual-frame codons, covariation, resistance.

Public entry point is :func:`hbvpol.selection.pipeline.run`.
"""

from __future__ import annotations

from .covariation import (
    COVARIATION_COLUMNS,
    EPISTASIS_COLUMNS,
    apc_correct,
    covarying_pairs,
    dca_scores,
    epistasis_pairs,
    mutual_information,
)
from .dualframe import (
    DUAL_FRAME_CLASSES,
    DUAL_FRAME_COLUMNS,
    classify_substitution,
    dual_frame_table,
)
from .entropy import ENTROPY_COLUMNS, lineage_entropy, per_position_entropy, shannon_entropy
from .genotype import GENOTYPE_COLUMNS, genotype_specificity
from .resistance import (
    RESISTANCE_COLUMNS,
    canonical_motif_at,
    load_catalogue,
    parse_rt_mutation,
    resistance_table,
)

__all__ = [
    "shannon_entropy",
    "per_position_entropy",
    "lineage_entropy",
    "ENTROPY_COLUMNS",
    "classify_substitution",
    "dual_frame_table",
    "DUAL_FRAME_CLASSES",
    "DUAL_FRAME_COLUMNS",
    "mutual_information",
    "apc_correct",
    "dca_scores",
    "covarying_pairs",
    "epistasis_pairs",
    "COVARIATION_COLUMNS",
    "EPISTASIS_COLUMNS",
    "genotype_specificity",
    "GENOTYPE_COLUMNS",
    "load_catalogue",
    "resistance_table",
    "parse_rt_mutation",
    "canonical_motif_at",
    "RESISTANCE_COLUMNS",
    "run",
]


def run(config, root):  # pragma: no cover - thin re-export
    from .pipeline import run as _run

    return _run(config, root)
