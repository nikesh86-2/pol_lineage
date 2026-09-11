"""Epsilon RNA / Pol coevolution stage.

Extracts the HBV epsilon element from oriented genomes, folds it (RNAfold or an
offline Nussinov fallback), quantifies Pol-TP/epsilon covariation and builds a
cross-genotype compatibility matrix.

Public entry point is :func:`hbvpol.epsilon.pipeline.run`.
"""

from __future__ import annotations

from .coevolve import (
    COMPATIBILITY_COLUMNS,
    COEVOLUTION_COLUMNS,
    compatibility_matrix,
    contact_map_energy,
    mutual_information,
    pol_epsilon_coevolution,
)
from .fold import (
    DEFAULT_EPSILON_SPAN,
    epsilon_features,
    extract_epsilon,
    fold_epsilon,
    nussinov_fold,
)

__all__ = [
    "extract_epsilon",
    "fold_epsilon",
    "nussinov_fold",
    "epsilon_features",
    "DEFAULT_EPSILON_SPAN",
    "pol_epsilon_coevolution",
    "compatibility_matrix",
    "mutual_information",
    "contact_map_energy",
    "COEVOLUTION_COLUMNS",
    "COMPATIBILITY_COLUMNS",
    "run",
]


def run(config, root):  # pragma: no cover - thin re-export
    from .pipeline import run as _run

    return _run(config, root)
