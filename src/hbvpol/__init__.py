"""hbvpol — recombination-aware evolution of hepatitis B virus polymerase.

The package is organised by analysis stage, mirroring the study design:

    io            shared FASTA/TSV/parquet and coordinate helpers
    domain        Pol domain model, reference coordinates, reading-frame maths
    datasets      Stage 1 — human-HBV and deep-hepadnavirus sequence acquisition
    qc            Stage 2a — circularisation and ORF integrity filtering
    recombination Stage 2b — RDP5/GARD/3SEQ/bootscan and block partitioning
    phylogeny     Stage 2c — per-block and per-domain trees, ancestral states
    selection     Stage 3 — entropy, dual-frame codons, covariation, epistasis
    structure     Stage 4 — multi-predictor structural ensembles and metrics
    epsilun       Stage 5 — epsilon-RNA folding and Pol:epsilon coevolution
    fitness       Stage 6 — deep-mutational-scanning integration
    atlas         Stage 7 — unified residue atlas and ranked target list
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
