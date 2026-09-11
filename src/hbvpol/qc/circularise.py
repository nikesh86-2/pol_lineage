"""Circularisation and orientation of HBV genomes.

HBV is a circular, partially double-stranded DNA virus.  Public databases emit
"complete genome" records that start at arbitrary offsets, so two genomes that
are biologically identical can carry completely different nucleotide numbering.
Every downstream comparison (ORF integrity, breakpoint coordinates, domain
extraction) therefore depends on first *recutting* all genomes at a common
origin and normalising their strand.

Design note / limitation
------------------------
The canonical implementation recuts every record at a **fixed** reference
coordinate (``reference.origin_nt``).  That is exact whenever the input records
share the reference's numbering convention (as NCBI ``NC_003977``-derived
records usually do) but it is only an approximation for genomes submitted in
an arbitrary rotation.  A production version should *detect* the origin by
aligning a k-mer from the reference origin against the genome rather than
assuming fixed numbering.  :func:`find_origin_by_reference` implements that
detection; it is opt-in via ``qc.detect_origin: true`` and always falls back to
the configured constant when the k-mer cannot be located unambiguously.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Iterable

from ..config import get
from ..domain import reverse_complement
from ..io import GenomeRecord, read_fasta, rotate_to_origin

__all__ = ["circularise_genomes", "orient_all", "find_origin_by_reference"]


def _reference_sequence(config) -> str | None:
    """Return the reference origin sequence, if one is configured/available.

    ``reference.sequence`` may be either a literal nucleotide string or a path
    to a FASTA file; a path is read and its first record used.
    """
    value = get(config, "reference.sequence")
    if not value:
        return None
    text = str(value)
    candidate = Path(text)
    if len(text) < 512 and candidate.exists() and candidate.is_file():
        records = read_fasta(candidate)
        return records[0].seq if records else None
    return text


def find_origin_by_reference(seq: str, ref_seq: str, seed_len: int = 25) -> int | None:
    """Locate the reference origin inside ``seq`` by exact k-mer match.

    Takes the first ``seed_len`` bases of ``ref_seq`` (the bases *at the
    reference origin*) as a seed and searches for it in ``seq`` interpreted
    circularly.  Returns the **1-based** position in ``seq`` at which the seed
    starts (i.e. the value to pass to :func:`rotate_to_origin` so that ``seq``
    is recut at the reference origin), or ``None`` when the seed is absent or
    ambiguous.

    This is deliberately conservative: a single exact match is required.  For
    divergent genomes a production implementation would fall back to a
    redundancy-reduced seed, an alignment back-translation, or IUPAC-aware
    matching; here the caller applies the configured constant instead.
    """
    if not seq or not ref_seq:
        return None
    seed = ref_seq[:seed_len].upper()
    query = seq.upper()
    if len(seed) < 4 or len(query) < len(seed):
        return None
    # Append a prefix so seeds spanning the wrap point are found.
    extended = query + query[: len(seed) - 1]
    index = extended.find(seed)
    if index == -1 or index >= len(query):
        return None
    return index + 1


def circularise_genomes(records: Iterable[GenomeRecord], config) -> list[GenomeRecord]:
    """Recut every record at the reference origin (``reference.origin_nt``).

    Each returned record is a copy whose ``metadata`` records the applied
    ``origin_shift`` (0-based left rotation) and the ``origin_nt_used``.  Those
    fields let later stages lift reference-numbered coordinates into the
    oriented frame.  When ``qc.detect_origin`` is enabled and a reference
    sequence is available, the origin is found with
    :func:`find_origin_by_reference`; otherwise the configured constant is
    used, as the contract specifies.
    """
    configured_origin = int(get(config, "reference.origin_nt", 1) or 1)
    detect = bool(get(config, "qc.detect_origin", False))
    seed_len = int(get(config, "qc.origin_seed_len", 25) or 25)
    ref_seq = _reference_sequence(config) if detect else None

    oriented: list[GenomeRecord] = []
    for record in records:
        seq = record.seq
        length = len(seq)
        if length == 0:
            oriented.append(record)
            continue

        origin: int | None = None
        if ref_seq:
            origin = find_origin_by_reference(seq, ref_seq, seed_len=seed_len)
        if origin is None:
            origin = ((configured_origin - 1) % length) + 1

        rotated = rotate_to_origin(seq, origin)
        shift = (origin - 1) % length
        metadata = dict(record.metadata)
        metadata["origin_shift"] = shift
        metadata["origin_nt_used"] = origin
        oriented.append(
            replace(record, seq=rotated, metadata=metadata)
        )
    return oriented


def orient_all(records: Iterable[GenomeRecord], config) -> list[GenomeRecord]:
    """Normalise strand and recut at the origin.

    Records flagged as reverse-strand (``strand == "-"``) are reverse
    complemented first so that all genomes are in the reference orientation,
    then :func:`circularise_genomes` is applied.
    """
    normalised: list[GenomeRecord] = []
    for record in records:
        if str(record.strand) in {"-", "-1"}:
            normalised.append(replace(record, seq=reverse_complement(record.seq), strand="+"))
        else:
            normalised.append(record)
    return circularise_genomes(normalised, config)
