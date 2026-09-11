"""Genotype specificity of Polymorphism positions.

For every Pol codon the fixation index ``Fst`` is computed over the amino-acid
state distribution across genotype groups:

    Ht = 1 - sum_i p_i^2                 (total heterozygosity)
    Hs = sum_g (n_g / N) * (1 - sum_i p_gi^2)   (mean within-group heterozygosity)
    Fst = (Ht - Hs) / Ht   (0 when Ht == 0)

A site is flagged ``genotype_informative`` when ``Fst >=
selection.genotype_specificity.min_fst`` (default 0.25).  High-Fst sites are
those where genotypes carry different, internally conserved residues — the
signature of genotype-specific adaptation rather than shared constraint.

Pure and offline; the pipeline adds no lineage column because this table is
genome-wide by design.
"""

from __future__ import annotations

from collections import Counter
from typing import Mapping

import pandas as pd

from ..config import get
from ..domain import domain_of, translate
from .dualframe import alignment_width, coerce_alignment, pol_codon_columns

__all__ = [
    "GENOTYPE_COLUMNS",
    "genotype_specificity",
]

GENOTYPE_COLUMNS = ["pol_position", "domain", "fst", "genotype_informative"]

_ID_COLUMNS = ("accession", "isolate", "id", "sequence_id", "name", "strain", "seq_id")
_GENOTYPE_COLUMNS = ("genotype", "genotype_group", "lineage")


def _id_column(metadata: pd.DataFrame) -> str | None:
    for candidate in _ID_COLUMNS:
        if candidate in metadata.columns:
            return candidate
    return None


def _genotype_column(metadata: pd.DataFrame) -> str | None:
    for candidate in _GENOTYPE_COLUMNS:
        if candidate in metadata.columns:
            return candidate
    return None


def _heterozygosity(counts: Counter) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    return 1.0 - sum((value / total) ** 2 for value in counts.values())


def _domain_label(position: int) -> str:
    try:
        return domain_of(position).value
    except ValueError:
        return ""


def genotype_specificity(alignment, metadata, config: Mapping[str, object]) -> pd.DataFrame:
    """Per-Pol-position Fst across genotype groups.

    ``metadata`` is a DataFrame with an id column (``accession``/``isolate``/…)
    and a ``genotype`` column.  Returns columns ``pol_position, domain, fst,
    genotype_informative``; an empty frame (correct schema) is returned when
    metadata is missing, genotypes are unknown, or fewer than two genotype
    groups are available.
    """
    records = coerce_alignment(alignment)
    if metadata is None or len(metadata) == 0 or not records:
        return pd.DataFrame(columns=GENOTYPE_COLUMNS)

    genotype_column = _genotype_column(metadata)
    if genotype_column is None:
        return pd.DataFrame(columns=GENOTYPE_COLUMNS)
    id_column = _id_column(metadata)

    genotype_of: dict[str, str] = {}
    for _, row in metadata.iterrows():
        key = str(row[id_column]) if id_column is not None else str(row.name)
        genotype = str(row[genotype_column]).strip()
        if genotype and genotype.lower() not in {"nan", "none", "na"}:
            genotype_of[key] = genotype

    groups: dict[str, list[str]] = {}
    for record in records:
        genotype = genotype_of.get(record.id)
        if genotype:
            groups.setdefault(genotype, []).append(record.seq.upper())
    if len(groups) < 2:
        return pd.DataFrame(columns=GENOTYPE_COLUMNS)

    width = alignment_width(records)
    codon_columns = pol_codon_columns(config, width)
    min_fst = float(get(config, "selection.genotype_specificity.min_fst", 0.25) or 0.0)
    total_sequences = sum(len(seqs) for seqs in groups.values())

    rows: list[dict] = []
    for index, columns in enumerate(codon_columns):
        overall: Counter = Counter()
        group_counts: list[tuple[int, Counter]] = []
        for sequences in groups.values():
            counts: Counter = Counter()
            for sequence in sequences:
                padded = sequence.ljust(width, "-")
                codon = "".join(padded[column] if column < len(padded) else "-" for column in columns)
                amino_acid = translate(codon, frame=0)
                if amino_acid and amino_acid != "X":
                    counts[amino_acid] += 1
                    overall[amino_acid] += 1
            group_counts.append((len(sequences), counts))

        total_heterozygosity = _heterozygosity(overall)
        if total_heterozygosity > 0:
            within = sum(
                size * _heterozygosity(counts) for size, counts in group_counts
            ) / total_sequences
            fst = (total_heterozygosity - within) / total_heterozygosity
        else:
            fst = 0.0
        position = index + 1
        rows.append({
            "pol_position": position,
            "domain": _domain_label(position),
            "fst": float(fst),
            "genotype_informative": bool(fst >= min_fst),
        })

    return pd.DataFrame(rows, columns=GENOTYPE_COLUMNS)
