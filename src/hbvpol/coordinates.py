"""Reference-frame column mapping for (gapped) alignments.

Every position the pipeline reports is expressed in the reference genome's own
numbering (the same convention as clinical rt numbering).  That is only
well-defined when alignment columns can be mapped to reference positions.
Individual records recut at the reference origin are *nearly* reference-indexed,
but a multiple alignment inserts gap columns and genotypes carry indels, so the
invariant "column ``i`` == reference position ``i + 1'`` breaks.

:func:`reference_positions` restores the mapping:

* pick a representative row of the alignment (the reference record itself when
  present, otherwise the record whose length is closest to the reference);
* align the configured reference genome to that row once;
* walk the row to build a per-column ``reference position | None`` map, where
  ``None`` marks a column that is an insertion relative to the reference.

Callers then select columns by *reference coordinate* rather than by index, and
:func:`columns_for_nt_range` additionally assigns insertion columns to the
domain of their nearest reference-anchored neighbour, so genotype-specific
insertions are not silently dropped from a domain sub-alignment.

When no reference sequence is configured the functions return ``None`` and
callers fall back to the historical index arithmetic (documented, and what the
offline tests exercise).
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

from .config import get
from .io import read_fasta
from .pipeline import get_logger
from .selection.dualframe import alignment_width, coerce_alignment

__all__ = [
    "reference_positions",
    "reference_sequence_from_config",
    "columns_for_nt_range",
    "assign_domains",
]

logger = get_logger("coordinates")

_BASES = set("ACGTUNRYKMSWBDHV")


def reference_sequence_from_config(config: Mapping[str, object]) -> str | None:
    """Return the configured reference genome, or ``None``.

    ``reference.sequence`` may be a path (as written by the reference stage) or
    an inlined nucleotide string; both are accepted.
    """
    value = get(config, "reference.sequence")
    if not value:
        return None
    text = str(value)
    candidate = Path(text)
    if len(text) < 512 and candidate.exists() and candidate.is_file():
        records = read_fasta(candidate)
        return records[0].seq.upper() if records else None
    if len(text) >= 100 and set(text.upper()) <= set("ACGTUNRYKMSWBDHV"):
        return text.upper()
    return None


def _pairwise_target_to_reference(reference: str, target: str) -> dict[int, int]:
    """Map 0-based target indices to 0-based reference indices by alignment."""
    from Bio.Align import PairwiseAligner, substitution_matrices

    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.substitution_matrix = substitution_matrices.load("NUC.4.4")
    aligner.open_gap_score = -10
    aligner.extend_gap_score = -0.5
    alignment = next(iter(aligner.align(reference, target)))
    reference_blocks = alignment.aligned[0]
    target_blocks = alignment.aligned[1]
    mapping: dict[int, int] = {}
    for (reference_start, reference_end), (target_start, target_end) in zip(
        reference_blocks, target_blocks
    ):
        length = min(int(reference_end) - int(reference_start), int(target_end) - int(target_start))
        for offset in range(length):
            mapping[int(target_start) + offset] = int(reference_start) + offset
    return mapping


def _pick_representative(records, reference: str, accession: object):
    """Prefer the reference record itself; otherwise the closest-length record."""
    if accession:
        key = str(accession).strip().lower().split(".")[0]
        for record in records:
            if record.id.strip().lower().split(".")[0] == key:
                return record
    reference_length = len(reference)
    return min(records, key=lambda record: abs(len(record.seq) - reference_length))


def reference_positions(alignment, config: Mapping[str, object]) -> list[int | None] | None:
    """Map each alignment column to a 1-based reference position, or ``None``.

    ``None`` is returned (the whole call) when no reference sequence is
    configured, so callers can fall back to index arithmetic.  Within the
    returned list, ``None`` entries are columns that are insertions relative to
    the reference.
    """
    records = coerce_alignment(alignment)
    if not records:
        return []
    reference = reference_sequence_from_config(config)
    if not reference:
        return None

    width = alignment_width(records)
    representative = _pick_representative(records, reference, get(config, "reference.accession"))
    representative_bases = "".join(
        symbol for symbol in representative.seq.upper() if symbol in _BASES
    )
    if not representative_bases:
        return None
    target_to_reference = _pairwise_target_to_reference(reference, representative_bases)
    row = representative.seq.upper().ljust(width, "-")

    positions: list[int | None] = []
    representative_index = 0
    for column in range(width):
        symbol = row[column]
        if symbol not in _BASES:
            positions.append(None)
            continue
        reference_index = target_to_reference.get(representative_index)
        positions.append(reference_index + 1 if reference_index is not None else None)
        representative_index += 1
    return positions


def _nearest_anchors(positions: Sequence[int | None]) -> list[int | None]:
    """Each column's nearest reference-anchored position (left first, then right)."""
    anchors: list[int | None] = [None] * len(positions)
    last: int | None = None
    for index, position in enumerate(positions):
        if position is not None:
            last = position
        anchors[index] = last
    following: int | None = None
    for index in range(len(positions) - 1, -1, -1):
        if positions[index] is not None:
            following = positions[index]
        if anchors[index] is None:
            anchors[index] = following
    return anchors


def _in_wrapped_range(position: int, start_nt: int, end_nt: int, genome_length: int) -> bool:
    if genome_length <= 0:
        return start_nt <= position <= end_nt
    start = ((start_nt - 1) % genome_length) + 1
    end = ((end_nt - 1) % genome_length) + 1
    if start <= end:
        return start <= position <= end
    return position >= start or position <= end


def columns_for_nt_range(
    positions: Sequence[int | None],
    start_nt: int,
    end_nt: int,
    genome_length: int,
) -> list[int]:
    """Columns whose nearest reference position falls in a (wrapping) nt range.

    Insertion columns (``positions`` entry ``None``) inherit the domain of their
    nearest anchored neighbour, so a genotype insertion internal to a domain is
    retained rather than dropped.
    """
    anchors = _nearest_anchors(positions)
    return [
        column
        for column, anchor in enumerate(anchors)
        if anchor is not None and _in_wrapped_range(anchor, start_nt, end_nt, genome_length)
    ]


def assign_domains(
    positions: Sequence[int | None],
    config: Mapping[str, object],
) -> list[str | None]:
    """Label each column with a Pol domain name (via its nearest anchor), or ``None``.

    Uses ``reference.domain_spans`` geometry converted to reference nucleotides.
    """
    from .domain import domain_spans_from_config

    reference = reference_sequence_from_config(config)
    genome_length = len(reference) if reference else max((len(str(value)) for value in positions), default=0)
    pol_start = int(get(config, "reference.pol_start_nt", 1) or 1)
    anchors = _nearest_anchors(positions)

    labels: list[str | None] = [None] * len(positions)
    for span in domain_spans_from_config(config):
        start_nt = pol_start + (span.start - 1) * 3
        end_nt = pol_start + (span.end - 1) * 3 + 2
        for column, anchor in enumerate(anchors):
            if anchor is None or labels[column] is not None:
                continue
            if _in_wrapped_range(anchor, start_nt, end_nt, genome_length):
                labels[column] = span.domain.value
    return labels
