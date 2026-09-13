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

from ..calibrate import orient_to_reference
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
    ``origin_shift`` (0-based left rotation), the ``origin_nt_used`` and the
    ``origin_method`` that resolved it.  Three strategies are tried in order
    when ``qc.detect_origin`` is on and a reference sequence is available:

    1. ``qc.origin_method: ymdd`` (default) -- align the invariant YMDD catalytic
       motif (which tolerates synonymous codon variants and detects the opposite
       strand) with :func:`hbvpol.calibrate.orient_to_reference`.  This works for
       every human genotype, unlike the exact-25-mer seed, and re-orients
       reverse-complemented records such as V01460.  Its assumption is that no
       indel separates the origin from the anchor motif (position 738 in the
       reference), which holds across human genotypes.
    2. the exact reference-origin k-mer seed (:func:`find_origin_by_reference`);
    3. the configured constant (``reference.origin_nt``), which assumes the
       record already uses the reference numbering.
    """
    configured_origin = int(get(config, "reference.origin_nt", 1) or 1)
    detect = bool(get(config, "qc.detect_origin", False))
    seed_len = int(get(config, "qc.origin_seed_len", 25) or 25)
    method = str(get(config, "qc.origin_method", "ymdd") or "ymdd").strip().lower()
    ref_seq = _reference_sequence(config) if detect else None

    oriented: list[GenomeRecord] = []
    for record in records:
        seq = record.seq
        length = len(seq)
        if length == 0:
            oriented.append(record)
            continue

        oriented_seq: str | None = None
        strand = str(record.strand)
        shift = 0
        origin = configured_origin
        resolved_by = "constant"

        if ref_seq and method in {"ymdd", "auto"}:
            anchored = orient_to_reference(seq, ref_seq)
            if anchored is not None:
                oriented_seq, detected_strand, shift = anchored
                if detected_strand in {"-", "-1"}:
                    strand = "+"
                origin = shift + 1
                resolved_by = "ymdd"

        if oriented_seq is None and ref_seq:
            found = find_origin_by_reference(seq, ref_seq, seed_len=seed_len)
            if found is not None:
                oriented_seq = rotate_to_origin(seq, found)
                shift = (found - 1) % length
                origin = found
                resolved_by = "seed"

        if oriented_seq is None:
            origin = ((configured_origin - 1) % length) + 1
            oriented_seq = rotate_to_origin(seq, origin)
            shift = (origin - 1) % length
            resolved_by = "constant"

        metadata = dict(record.metadata)
        metadata["origin_shift"] = shift
        metadata["origin_nt_used"] = origin
        metadata["origin_method"] = resolved_by
        oriented.append(
            replace(record, seq=oriented_seq, strand=strand, metadata=metadata)
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
