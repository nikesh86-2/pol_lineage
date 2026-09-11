"""Alignment helpers for the phylogeny stage.

This module is deliberately dependency-light: Biopython's ``AlignIO`` is only
imported inside :func:`read_alignment`, so importing this module (and therefore
the whole phylogeny package) works in a fully offline interpreter with no
alignment tools installed.

An *alignment* is represented throughout the package as an ordered list of
``(name, sequence)`` pairs.  Every helper accepts any of the following and
coerces it with :func:`as_pairs`:

* a path (``str`` or :class:`~pathlib.Path`) to a FASTA file,
* a list of :class:`~hbvpol.io.GenomeRecord`,
* a list of ``(name, sequence)`` pairs,
* a mapping of ``name -> sequence``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from ..config import get
from ..io import GenomeRecord, read_fasta
from ..domain import DEFAULT_DOMAIN_SPANS, DomainSpan, PolDomain, frame_indices, translate

__all__ = [
    "as_pairs",
    "read_alignment",
    "alignment_length",
    "slice_alignment",
    "translate_alignment_for_domain",
    "alignment_names",
]


def as_pairs(alignment) -> list[tuple[str, str]]:
    """Coerce an alignment argument into a list of ``(name, sequence)`` pairs.

    Accepts a path, a list of :class:`GenomeRecord`, a list of pairs, a mapping
    ``name -> sequence``, ``None`` (empty list), or any iterable of the above.
    """
    if alignment is None:
        return []
    if isinstance(alignment, (str, Path)):
        return read_alignment(alignment)
    if isinstance(alignment, Mapping):
        return [(str(name), str(seq)) for name, seq in alignment.items()]
    if isinstance(alignment, list):
        if not alignment:
            return []
        first = alignment[0]
        if isinstance(first, GenomeRecord):
            return [(record.id, record.seq) for record in alignment]
        pairs: list[tuple[str, str]] = []
        for index, item in enumerate(alignment):
            if isinstance(item, GenomeRecord):
                pairs.append((item.id, item.seq))
            elif isinstance(item, (tuple, list)) and len(item) >= 2:
                pairs.append((str(item[0]), str(item[1])))
            elif hasattr(item, "id") and hasattr(item, "seq"):
                pairs.append((str(item.id), str(item.seq)))
            else:
                pairs.append((str(index + 1), str(item)))
        return pairs
    # Generic iterable of GenomeRecord / pairs.
    return as_pairs(list(alignment))


def read_alignment(path: str | Path) -> list[tuple[str, str]]:
    """Read a FASTA alignment into ``(name, sequence)`` pairs.

    Uses Biopython ``AlignIO`` (imported here, not at module import) and falls
    back to the project's own FASTA reader if the file is not a valid alignment.
    """
    from Bio import AlignIO

    try:
        alignment = AlignIO.read(str(path), "fasta")
    except Exception:
        return [(record.id, record.seq) for record in read_fasta(path)]
    return [(record.id, str(record.seq)) for record in alignment]


def alignment_length(alignment) -> int:
    """Return the number of columns in the alignment (0 for an empty one)."""
    pairs = as_pairs(alignment)
    if not pairs:
        return 0
    return max(len(seq) for _, seq in pairs)


def alignment_names(alignment) -> list[str]:
    """Return the sequence names in alignment order."""
    return [name for name, _ in as_pairs(alignment)]


def slice_alignment(alignment, start: int, end: int) -> list[tuple[str, str]]:
    """Return alignment columns ``start..end`` (1-based, inclusive)."""
    pairs = as_pairs(alignment)
    if start < 1:
        start = 1
    return [(name, seq[start - 1:end]) for name, seq in pairs]


def _span_for_domain(domain: PolDomain | str) -> DomainSpan:
    parsed = domain if isinstance(domain, PolDomain) else PolDomain.parse(str(domain))
    for span in DEFAULT_DOMAIN_SPANS:
        if span.domain == parsed:
            return span
    raise ValueError(f"no span configured for domain {parsed!r}")


def translate_alignment_for_domain(
    alignment, domain: PolDomain | str, config: Mapping[str, object]
) -> list[tuple[str, str]]:
    """Translate the Pol codons of a domain into amino acids.

    The domain's amino-acid span (from :data:`hbvpol.domain.DEFAULT_DOMAIN_SPANS`)
    is converted to nucleotide columns using ``reference.pol_start_nt`` and the
    triplet periodicity of the Pol frame, then each sequence is translated in
    frame 0 of that slice.

    Coordinate contract: alignment columns are assumed to be in the oriented
    reference frame (column ``i`` is reference position ``i + 1``), matching the
    output of the QC recut and the recombination stage's domain sub-alignments.
    With a domain sub-alignment (which already starts at the domain's first
    codon) this is equivalent to translating the whole slice in frame 0.

    Returns a list of ``(name, amino_acid_sequence)`` pairs.
    """
    pairs = as_pairs(alignment)
    span = _span_for_domain(domain)
    width = alignment_length(pairs)
    if width == 0:
        return [(name, "") for name, _ in pairs]

    pol_start = int(get(config, "reference.pol_start_nt", 1) or 1)
    nt_start = pol_start + (span.start - 1) * 3
    nt_length = span.length * 3
    indices = frame_indices(nt_start, nt_length, width)

    translated: list[tuple[str, str]] = []
    for name, seq in pairs:
        sub = "".join(seq[i] if i < len(seq) else "-" for i in indices)
        translated.append((name, translate(sub, frame=0)))
    return translated
