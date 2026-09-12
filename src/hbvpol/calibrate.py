"""Alignment-based calibration of per-genotype element spans.

Why this exists
---------------
The shipped epsilon (:data:`hbvpol.epsilon.fold.DEFAULT_EPSILON_SPAN`) and Pol
domain (:data:`hbvpol.domain.DEFAULT_DOMAIN_SPANS`) spans are single-genotype
fallbacks.  Genotype reference genomes are **not** the same length (a 6-nt
indel upstream of preS1 makes genotype A ~3221 nt versus genotype D's 3182 nt),
so copying one genotype's coordinates onto another silently misplaces the
element.  Because these elements are sequence-conserved, the reliable
calibration is alignment, not arithmetic: locate the element by aligning a
conserved query to the genotype's own reference sequence, then read the
coordinates off *that* sequence.

Two flavours are implemented here:

* nucleotide local alignment for epsilon -- a conserved epsilon nt query against
  a genotype genome (:func:`calibrate_epsilon_span`);
* protein alignment for the Pol domains -- a query Pol protein with known
  domain boundaries against a genotype's Pol protein
  (:func:`calibrate_domain_spans`), because domain boundaries are defined in
  amino-acid space and genotype differences (mostly indels in the spacer) shift
  them.

The module is deliberately dependency-light at import time; Biopython's
``PairwiseAligner`` is imported lazily so the rest of the pipeline does not pay
for it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .domain import (
    DEFAULT_DOMAIN_SPANS,
    DomainSpan,
    PolDomain,
    reverse_complement,
    translate,
)

__all__ = [
    "AlignmentHit",
    "DomainCalibration",
    "calibrate_epsilon_span",
    "calibrate_domain_spans",
    "map_span_by_alignment",
    "detect_pol_protein",
]


# Canonical reporting order for the four domains.
_DOMAIN_ORDER: tuple[PolDomain, ...] = (
    PolDomain.TP,
    PolDomain.SPACER,
    PolDomain.RT,
    PolDomain.RNASEH,
)


@dataclass(frozen=True)
class AlignmentHit:
    """A 1-based inclusive interval located in the *target* sequence."""

    start: int
    end: int
    identity: float
    coverage: float = 1.0


@dataclass(frozen=True)
class DomainCalibration:
    """Calibrated Pol domain spans plus the alignment's quality."""

    spans: tuple[DomainSpan, ...]
    identity: float
    coverage: float
    warnings: tuple[str, ...] = ()


# --------------------------------------------------------------------------- #
# aligner construction
# --------------------------------------------------------------------------- #
def _nucleotide_aligner():
    from Bio.Align import PairwiseAligner, substitution_matrices

    aligner = PairwiseAligner()
    aligner.mode = "local"
    aligner.substitution_matrix = substitution_matrices.load("NUC.4.4")
    aligner.open_gap_score = -10
    aligner.extend_gap_score = -0.5
    return aligner


def _protein_aligner(mode: str = "global"):
    from Bio.Align import PairwiseAligner, substitution_matrices

    aligner = PairwiseAligner()
    aligner.mode = mode
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    aligner.open_gap_score = -11
    aligner.extend_gap_score = -1
    return aligner


def _best_alignment(aligner, target: str, query: str):
    """Return the highest-scoring alignment (or ``None``)."""
    alignments = aligner.align(target, query)
    try:
        return next(iter(alignments))
    except StopIteration:
        return None


def _aln_stats(alignment, query_length: int) -> tuple[float, float]:
    """Return ``(identity, coverage)`` as fractions of the ungapped query.

    ``coverage`` is the fraction of query residues placed opposite a target
    residue; ``identity`` is the fraction of query residues placed opposite an
    identical target residue.  Identity over the query length (rather than over
    aligned columns) means a short, near-perfect local hit cannot masquerade as
    a confident full-length calibration.
    """
    if query_length <= 0:
        return 0.0, 0.0
    target_row = str(alignment[0])
    query_row = str(alignment[1])
    placed = 0
    matches = 0
    for target_char, query_char in zip(target_row, query_row):
        if query_char == "-" or target_char == "-":
            continue
        placed += 1
        if target_char == query_char:
            matches += 1
    return matches / query_length, placed / query_length


def _query_to_target_map(alignment) -> dict[int, int]:
    """Map 0-based query indices to 0-based target indices along the alignment.

    Only query residues placed opposite a target residue get an entry; residues
    aligned to a target gap are filled by :func:`_nearest` when mapping a span
    boundary.
    """
    mapping: dict[int, int] = {}
    target_blocks = alignment.aligned[0]
    query_blocks = alignment.aligned[1]
    for (target_start, target_end), (query_start, query_end) in zip(target_blocks, query_blocks):
        length = min(int(target_end) - int(target_start), int(query_end) - int(query_start))
        for offset in range(length):
            mapping[int(query_start) + offset] = int(target_start) + offset
    return mapping


def _nearest(mapping: dict[int, int], index: int, limit: int) -> int | None:
    """Nearest mapped target index to ``index`` (ties prefer the left)."""
    if index in mapping:
        return mapping[index]
    for distance in range(1, limit + 1):
        if index - distance in mapping:
            return mapping[index - distance]
        if index + distance in mapping:
            return mapping[index + distance]
    return None


def map_span_by_alignment(alignment, start: int, end: int, query_length: int | None = None):
    """Map a 1-based inclusive query span to target coordinates.

    Returns ``(start, end)`` (1-based inclusive) or ``None`` when the alignment
    places none of the span on the target.  Boundary residues aligned to target
    gaps snap to the nearest placed residue, so a boundary inside a target
    insertion is not lost.
    """
    mapping = _query_to_target_map(alignment)
    if not mapping:
        return None
    if query_length is None:
        query_length = max(mapping) + 1
    left = _nearest(mapping, start - 1, query_length)
    right = _nearest(mapping, end - 1, query_length)
    if left is None or right is None:
        return None
    if left > right:
        left, right = right, left
    return left + 1, right + 1


# --------------------------------------------------------------------------- #
# public calibration entry points
# --------------------------------------------------------------------------- #
def calibrate_epsilon_span(query: str, target: str, min_identity: float = 0.85) -> AlignmentHit:
    """Locate a conserved epsilon nt query in a genotype genome by local alignment.

    Raises :class:`ValueError` when no alignment is found.
    """
    query = str(query).upper().replace("U", "T")
    target = str(target).upper().replace("U", "T")
    alignment = _best_alignment(_nucleotide_aligner(), target, query)
    if alignment is None:
        raise ValueError("epsilon query did not align to the target genome")
    identity, coverage = _aln_stats(alignment, len(query))
    mapped = map_span_by_alignment(alignment, 1, len(query), len(query))
    if mapped is None:
        raise ValueError("epsilon query aligned but no residues could be mapped")
    start, end = mapped
    _ = min_identity  # documented threshold; the caller decides how to act on it
    return AlignmentHit(start, end, identity, coverage)


def calibrate_domain_spans(
    query_pol: str,
    target_pol: str,
    query_spans: Iterable[DomainSpan] = DEFAULT_DOMAIN_SPANS,
    min_identity: float = 0.6,
) -> DomainCalibration:
    """Transfer Pol domain boundaries from a query Pol protein to a target Pol.

    A global BLOSUM62 alignment is used because the boundaries are internal and
    span the whole polypeptide.  Each query span is mapped through the alignment
    and clamped to the target length, so a shorter target genotype cannot spill
    past its own end.
    """
    query_pol = str(query_pol).upper()
    target_pol = str(target_pol).upper()
    if not query_pol or not target_pol:
        raise ValueError("query and target Pol proteins must be non-empty")
    alignment = _best_alignment(_protein_aligner("global"), target_pol, query_pol)
    if alignment is None:
        raise ValueError("query Pol did not align to the target Pol")
    identity, coverage = _aln_stats(alignment, len(query_pol))

    warnings: list[str] = []
    spans: list[DomainSpan] = []
    for span in query_spans:
        mapped = map_span_by_alignment(alignment, span.start, span.end, len(query_pol))
        if mapped is None:
            warnings.append(
                f"{span.domain.value}: could not map {span.start}-{span.end}; "
                "keeping query coordinates"
            )
            mapped = (span.start, span.end)
        start, end = mapped
        start = max(1, min(start, len(target_pol)))
        end = max(1, min(end, len(target_pol)))
        if start > end:
            start, end = end, start
        spans.append(DomainSpan(span.domain, start, end))

    if identity < min_identity:
        warnings.append(
            f"alignment identity {identity:.1%} is below min_identity "
            f"{min_identity:.0%}; inspect before trusting these spans"
        )
    if coverage < 0.8:
        warnings.append(f"alignment covers only {coverage:.1%} of the query Pol")

    order = {domain: index for index, domain in enumerate(_DOMAIN_ORDER)}
    spans.sort(key=lambda span: (order.get(span.domain, len(order)), span.start))
    return DomainCalibration(tuple(spans), identity, coverage, tuple(warnings))


def detect_pol_protein(
    query_pol: str,
    genome: str,
    min_identity: float = 0.6,
) -> tuple[str, str, int, float, float]:
    """Find the Pol ORF in an unannotated circular genome by six-frame translation.

    The genome is searched in both senses and as a **doubled** copy, because the
    HBV Pol ORF routinely wraps the numbering origin (e.g. 2309 -> 1625 in
    NC_003977.2) and so is not contiguous in a linear record.  The returned
    protein is trimmed to the region the query aligns to, so its residue 1 is
    Pol residue 1 and domain boundaries transfer without an offset.

    Returns ``(protein, strand, frame, identity, coverage)``; ``frame`` is 0-based.
    Raises :class:`ValueError` when nothing aligns or the best hit is too weak.
    """
    query_pol = str(query_pol).upper()
    genome = str(genome).upper().replace("U", "T")
    doubled = genome + genome
    best: tuple[float, str, str, int, float, float] | None = None
    for strand, oriented in (("+", doubled), ("-", reverse_complement(doubled))):
        for frame in range(3):
            protein = translate(oriented, frame=frame)
            if len(protein) < 30:
                continue
            alignment = _best_alignment(_protein_aligner("local"), protein, query_pol)
            if alignment is None:
                continue
            identity, coverage = _aln_stats(alignment, len(query_pol))
            blocks = alignment.aligned[0]
            first = int(min(block[0] for block in blocks))
            last = int(max(block[1] for block in blocks))
            trimmed = protein[first:last]
            score = identity * coverage
            if best is None or score > best[0]:
                best = (score, trimmed, strand, frame, identity, coverage)
    if best is None:
        raise ValueError("no Pol-like ORF found in any reading frame")
    _, protein, strand, frame, identity, coverage = best
    if identity < min_identity:
        raise ValueError(
            f"best Pol-like ORF identity {identity:.1%} is below min_identity "
            f"{min_identity:.0%}; the reference may be the wrong accession"
        )
    return protein, strand, frame, identity, coverage
