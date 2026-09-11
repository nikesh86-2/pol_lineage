"""Deterministic synthetic data for offline development, CI and smoke tests.

The real pipeline is dominated by network acquisition and by external tools.
This module fabricates a *small but structurally faithful* dataset — genomes in
the standard HBV numbering, an intact (stop-free) Pol ORF with the priming
tyrosine and the YMDD catalytic motif planted, genotype labels, longitudinal
metadata and a handful of deep-hepadnavirus Pol proteins — so that every
downstream stage can be exercised end to end with no network and no external
programs.

The sequences are synthetic and must never be used for biological inference;
they exist only to make the wiring testable.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Iterable

from .config import load_config

__all__ = [
    "SENSE_CODONS",
    "build_synthetic",
    "synthetic_overrides",
    "run_synthetic_pipeline",
    "SYNTHETIC_STAGES",
]

#: All 61 sense codons (TAA/TAG/TGA excluded).
SENSE_CODONS: tuple[str, ...] = tuple(
    a + b + c
    for a in "ACGT"
    for b in "ACGT"
    for c in "ACGT"
    if a + b + c not in {"TAA", "TAG", "TGA"}
)

_BASES = "ACGT"

#: Stages runnable without network/tools, in dependency order.
SYNTHETIC_STAGES: tuple[str, ...] = (
    "qc",
    "recombine",
    "tree",
    "select",
    "epsilon",
    "fitness",
    "structure",
    "atlas",
)

#: Planted functional anchors (Pol amino-acid positions, genotype A2 numbering).
_PRIMING_TYROSINE = (63, "TAT")
_YMDD = ((549, "TAT"), (550, "ATG"), (551, "GAT"), (552, "GAT"))


def _pol_start_index(pol_start_nt: int, length: int) -> int:
    return (pol_start_nt - 1) % length


def _pol_coding_length(length: int, pol_start_nt: int, pol_end_nt: int) -> int:
    start0 = _pol_start_index(pol_start_nt, length)
    return (length - start0) + pol_end_nt


def _pol_codon_indices(length: int, pol_start_nt: int, pol_end_nt: int) -> list[list[int]]:
    start0 = _pol_start_index(pol_start_nt, length)
    n_codons = _pol_coding_length(length, pol_start_nt, pol_end_nt) // 3
    return [
        [(start0 + 3 * k + j) % length for j in range(3)]
        for k in range(n_codons)
    ]


def _set_codon(seq: list[str], indices: list[int], codon: str) -> None:
    for index, base in zip(indices, codon):
        seq[index] = base


def make_founder_genome(
    rng: random.Random,
    length: int = 3215,
    pol_start_nt: int = 2307,
    pol_end_nt: int = 1623,
) -> str:
    """Build one synthetic genome with a stop-free Pol ORF and planted motifs."""
    seq = [rng.choice(_BASES) for _ in range(length)]
    codons = _pol_codon_indices(length, pol_start_nt, pol_end_nt)
    for indices in codons:
        _set_codon(seq, indices, rng.choice(SENSE_CODONS))

    # Plant the priming tyrosine and the YMDD catalytic motif in the Pol frame.
    aa_pos, codon = _PRIMING_TYROSINE
    if aa_pos <= len(codons):
        _set_codon(seq, codons[aa_pos - 1], codon)
    for aa_pos, codon in _YMDD:
        if aa_pos <= len(codons):
            _set_codon(seq, codons[aa_pos - 1], codon)
    return "".join(seq)


def _repair_pol_stops(seq: list[str], codons: list[list[int]], rng: random.Random) -> None:
    """Replace any mutated stop codon inside the Pol ORF with a sense codon."""
    from .domain import translate

    for indices in codons:
        codon = "".join(seq[i] for i in indices)
        if translate(codon) == "*":
            _set_codon(seq, indices, rng.choice(SENSE_CODONS))


def mutate_genome(founder: str, rng: random.Random, rate: float, pol_codons: list[list[int]]) -> str:
    """Introduce point substitutions, keeping the Pol ORF stop-free."""
    seq = list(founder)
    for index in range(len(seq)):
        if rng.random() < rate:
            seq[index] = rng.choice([b for b in _BASES if b != seq[index]])
    # Deterministically repair any stop codon introduced inside Pol.
    _repair_pol_stops(seq, pol_codons, rng)
    return "".join(seq)


def _write_fasta(records: Iterable[tuple[str, str]], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for ident, seq in records:
            handle.write(f">{ident}\n")
            for i in range(0, len(seq), 70):
                handle.write(seq[i:i + 70] + "\n")
    return path


def _build_deephep_alignment(rng: random.Random, path: Path) -> Path:
    """A tiny protein alignment standing in for the deep-hepadnavirus Pol set."""
    groups = ["primate", "rodent", "bat", "avian", "reptile", "amphibian", "fish", "nackednavirus"]
    length = 180
    founders = {
        group: [rng.choice("ACDEFGHIKLMNPQRSTVWY") for _ in range(length)]
        for group in groups
    }
    records: list[tuple[str, str]] = []
    for group, founder in founders.items():
        for copy in range(2):
            seq = list(founder)
            for index in range(length):
                if rng.random() < 0.05:
                    seq[index] = rng.choice("ACDEFGHIKLMNPQRSTVWY")
            records.append((f"{group}_{copy}", "".join(seq)))
    return _write_fasta(records, path)


def build_synthetic(
    outdir: str | Path,
    n_genomes: int = 8,
    seed: int = 20240911,
    length: int = 3215,
    pol_start_nt: int = 2307,
    pol_end_nt: int = 1623,
    mutation_rate: float = 0.03,
) -> dict[str, Path]:
    """Fabricate a full synthetic ``datasets/`` stage output under ``outdir``.

    Returns a mapping of artefact name to path, mirroring the dataset stage.
    Two genotypes (A and D) are produced with a shared within-genotype founder,
    plus a longitudinal pair sharing a patient id, so per-lineage and
    longitudinal code paths are actually exercised.
    """
    rng = random.Random(seed)
    outdir = Path(outdir)
    datasets = outdir / "datasets"
    datasets.mkdir(parents=True, exist_ok=True)

    genotypes = ("A", "D")
    codons = _pol_codon_indices(length, pol_start_nt, pol_end_nt)
    founders = {genotype: make_founder_genome(rng, length, pol_start_nt, pol_end_nt) for genotype in genotypes}

    records: list[tuple[str, str]] = []
    rows: list[dict[str, object]] = []
    countries = ["US", "CN", "DE", "IN"]
    years = [1998, 2007, 2014, 2021]
    for index in range(n_genomes):
        genotype = genotypes[index % len(genotypes)]
        accession = f"SYN{index:03d}"
        seq = mutate_genome(founders[genotype], rng, mutation_rate, codons)
        records.append((accession, seq))
        rows.append({
            "accession": accession,
            "genotype": genotype,
            "subgenotype": f"{genotype}{1 + index % 2}",
            "country": countries[index % len(countries)],
            "year": years[index % len(years)],
            "isolation_source": "serum",
            "treatment_status": "treated" if index % 3 == 0 else "naive",
            "patient_id": f"P{index // 2}",
            "length": len(seq),
            "source": "synthetic",
            "description": f"synthetic genotype {genotype} genome",
        })

    fasta = _write_fasta(records, datasets / "hbv_genomes.fasta")

    metadata = datasets / "hbv_metadata.tsv"
    columns = [
        "accession", "genotype", "subgenotype", "country", "year",
        "isolation_source", "treatment_status", "patient_id", "length",
        "source", "description",
    ]
    with metadata.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    deephep = _build_deephep_alignment(rng, datasets / "deephep_alignment.fasta")

    return {
        "hbv_genomes": fasta,
        "hbv_metadata": metadata,
        "deephep_alignment": deephep,
    }


def synthetic_overrides(outdir: str | Path, n_genomes: int = 8) -> list[str]:
    """Config overrides that make ``config.yaml`` run offline on synthetic data.

    ``outdir`` is the output root that :func:`build_synthetic` populated (the
    ``datasets/`` directory lives directly beneath it).
    """
    outdir = Path(outdir).resolve()
    return [
        f"project.output_root={outdir}",
        "datasets.hbv.sources=[]",
        # The synthetic Pol ORF is stop-free; S/C ORFs are not modelled.
        "qc.require_complete_orf=[]",
        "qc.exclude_stop_in_pol=true",
        "recombination.tools=[bootscan]",
        "recombination.min_block_len=200",
        "recombination.min_seqs_for_scan=4",
        "phylogeny.bootstrap=0",
        "selection.methods=[]",
        "selection.covariation.min_seqs=4",
        "selection.covariation.max_positions=150",
        "selection.epistasis.min_support=0.0",
        "selection.epistasis.max_positions=150",
        "structure.strategies=[]",
        "structure.seeds=1",
        "structure.states=[apo]",
        "structure.lineages=['A', 'D']",
        "structure.md.n_models=2",
        "fitness.dms.map=null",
        "epsilun.fold_models=[nussinov]",
        "atlas.interactive=false",
    ]


def run_synthetic_pipeline(config: dict, root: str | Path) -> dict[str, dict[str, Path]]:
    """Run every network-free stage in order, returning ``{stage: artefacts}``."""
    from .cli import STAGES, _resolve

    results: dict[str, dict[str, Path]] = {}
    for stage in SYNTHETIC_STAGES:
        func = _resolve(STAGES[stage])
        results[stage] = func(config, root)
    return results


def synthetic_config(config_path: str | Path, outdir: str | Path, n_genomes: int = 8) -> dict:
    """Load the real config with :func:`synthetic_overrides` applied."""
    return load_config(config_path, overrides=synthetic_overrides(outdir, n_genomes))
