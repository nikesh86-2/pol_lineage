"""QC stage — circularisation, ORF integrity and dereplication.

Public entry point is :func:`hbvpol.qc.pipeline.run`.
"""

from __future__ import annotations

from .circularise import circularise_genomes, find_origin_by_reference, orient_all
from .deduplicate import dereplicate, sequence_identity
from .orfcheck import QC_COLUMNS, check_genome, check_orf

__all__ = [
    "circularise_genomes",
    "find_origin_by_reference",
    "orient_all",
    "dereplicate",
    "sequence_identity",
    "check_genome",
    "check_orf",
    "QC_COLUMNS",
    "run",
]


def run(config, root):  # pragma: no cover - thin re-export
    from .pipeline import run as _run

    return _run(config, root)
