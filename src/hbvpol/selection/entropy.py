"""Sequence entropy per Pol position (amino-acid and nucleotide).

Entropy is reported in **bits** (base-2 logarithm).  For each Pol codon the
amino-acid distribution across sequences gives ``aa_entropy``; the three codon
columns pooled give ``nt_entropy``.  Both profiles are smoothed with a centred
sliding window whose width is ``selection.window`` codons (default 21, forced
odd) to reduce the sampling noise of sparse alignments.  ``gap_frac`` is the
fraction of sequences whose codon contains a gap/ambiguous base, and ``n_seqs``
is the number of sequences contributing to the position.

All functions are pure and offline.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Mapping

import pandas as pd

from ..config import get
from ..domain import domain_of, domain_spans_from_config, translate
from .dualframe import (
    _codon_string,
    alignment_width,
    coerce_alignment,
    codon_columns_from_positions,
    pol_codon_columns,
)

__all__ = [
    "ENTROPY_COLUMNS",
    "shannon_entropy",
    "per_position_entropy",
    "lineage_entropy",
]

ENTROPY_COLUMNS = [
    "lineage",
    "pol_position",
    "domain",
    "aa_entropy",
    "nt_entropy",
    "gap_frac",
    "n_seqs",
]

_PER_POSITION_COLUMNS = [column for column in ENTROPY_COLUMNS if column != "lineage"]

_ACGT = frozenset("ACGT")


def shannon_entropy(counts) -> float:
    """Shannon entropy (bits) of a count distribution.

    ``counts`` may be a mapping ``{symbol: count}`` or any iterable of counts.
    Zero counts are ignored; an empty/zero-total input returns ``0.0``.
    """
    values = list(counts.values()) if isinstance(counts, Mapping) else list(counts)
    total = sum(values)
    if total <= 0:
        return 0.0
    entropy = 0.0
    for value in values:
        if value > 0:
            probability = value / total
            entropy -= probability * math.log2(probability)
    return entropy


def _window_width(config: Mapping[str, object]) -> int:
    window = int(get(config, "selection.window", 21) or 21)
    window = max(1, window)
    if window % 2 == 0:
        window += 1
    return window


def _smooth(values: list[float], window: int) -> list[float]:
    half = window // 2
    count = len(values)
    smoothed: list[float] = []
    for index in range(count):
        low = max(0, index - half)
        high = min(count, index + half + 1)
        segment = values[low:high]
        smoothed.append(sum(segment) / len(segment) if segment else 0.0)
    return smoothed


def _domain_label(position: int, spans=None) -> str:
    try:
        return (domain_of(position, spans) if spans else domain_of(position)).value
    except ValueError:
        return ""


def per_position_entropy(
    alignment,
    config: Mapping[str, object],
    positions=None,
    genome_length: int | None = None,
) -> pd.DataFrame:
    """Per-Pol-position entropy table (without the ``lineage`` column).

    Returns columns ``pol_position, domain, aa_entropy, nt_entropy, gap_frac,
    n_seqs``.  The pipeline adds ``lineage`` and concatenates across lineages.

    Passing the column -> reference-position ``positions`` map selects codons by
    reference coordinate (correct for gapped alignments and indel-bearing
    genotypes); otherwise the historical reference-index arithmetic is used.
    """
    records = coerce_alignment(alignment)
    if not records:
        return pd.DataFrame(columns=_PER_POSITION_COLUMNS)

    width = alignment_width(records)
    if positions is not None and genome_length:
        codon_columns: list[list[int | None]] = codon_columns_from_positions(
            positions, config, genome_length
        )
    else:
        codon_columns = pol_codon_columns(config, width)
    if not codon_columns:
        return pd.DataFrame(columns=_PER_POSITION_COLUMNS)

    sequences = [record.seq.upper().ljust(width, "-") for record in records]
    n_seqs = len(sequences)
    spans = domain_spans_from_config(config)

    aa_raw: list[float] = []
    nt_raw: list[float] = []
    gap_fracs: list[float] = []

    for columns in codon_columns:
        amino_counts: Counter = Counter()
        nucleotide_counts: Counter = Counter()
        gap_count = 0
        for sequence in sequences:
            codon = _codon_string(sequence, columns)
            if any(base not in _ACGT for base in codon):
                gap_count += 1
            amino_acid = translate(codon, frame=0)
            if amino_acid and amino_acid != "X":
                amino_counts[amino_acid] += 1
            for base in codon:
                if base in _ACGT:
                    nucleotide_counts[base] += 1
        aa_raw.append(shannon_entropy(amino_counts))
        nt_raw.append(shannon_entropy(nucleotide_counts))
        gap_fracs.append(gap_count / n_seqs if n_seqs else 0.0)

    window = _window_width(config)
    aa_smoothed = _smooth(aa_raw, window)
    nt_smoothed = _smooth(nt_raw, window)

    rows: list[dict] = []
    for index in range(len(codon_columns)):
        position = index + 1
        rows.append({
            "pol_position": position,
            "domain": _domain_label(position, spans),
            "aa_entropy": aa_smoothed[index],
            "nt_entropy": nt_smoothed[index],
            "gap_frac": gap_fracs[index],
            "n_seqs": n_seqs,
        })
    return pd.DataFrame(rows, columns=_PER_POSITION_COLUMNS)


def lineage_entropy(records_by_lineage, config: Mapping[str, object]) -> pd.DataFrame:
    """Compute :func:`per_position_entropy` for each lineage and concatenate.

    ``records_by_lineage`` maps a lineage label to an alignment (any spelling
    accepted by :func:`coerce_alignment`).  The returned frame has the full
    :data:`ENTROPY_COLUMNS` schema.
    """
    frames: list[pd.DataFrame] = []
    for lineage, alignment in (records_by_lineage or {}).items():
        frame = per_position_entropy(alignment, config)
        frame.insert(0, "lineage", lineage)
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=ENTROPY_COLUMNS)
    combined = pd.concat(frames, ignore_index=True)
    return combined.reindex(columns=ENTROPY_COLUMNS)
