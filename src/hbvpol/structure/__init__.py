"""Structural ensemble stage.

Builds AlphaFold2/3 and ESMFold models for each (genotype state) combination,
computes structural metrics (pLDDT, PAE, SASA, domain orientation, catalytic
geometry, nucleic-acid compatibility, conserved packing), reclassifies
low-pLDDT regions as candidate hinges, and emits a coarse-grained MD plan.

Public entry point is :func:`hbvpol.structure.pipeline.run`.
"""

from __future__ import annotations

from .md import build_md_manifest, pocket_persistence
from .metrics import (
    candidate_hinges,
    catalytic_geometry,
    compute_metrics,
    conserved_packing,
    domain_orientation,
    nucleic_acid_compatibility,
    pae_from_json,
    per_residue_sasa,
    plddt_from_pdb,
)
from .models import build_model_manifest, run_predictor, sequence_for_lineage

__all__ = [
    "build_model_manifest",
    "sequence_for_lineage",
    "run_predictor",
    "plddt_from_pdb",
    "pae_from_json",
    "per_residue_sasa",
    "domain_orientation",
    "catalytic_geometry",
    "nucleic_acid_compatibility",
    "conserved_packing",
    "compute_metrics",
    "candidate_hinges",
    "build_md_manifest",
    "pocket_persistence",
    "run",
]


def run(config, root):  # pragma: no cover - thin re-export
    from .pipeline import run as _run

    return _run(config, root)
