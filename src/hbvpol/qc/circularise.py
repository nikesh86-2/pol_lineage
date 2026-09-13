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


def _orf_stop_score(seq: str, config) -> int:
    """Total internal stop codons across the required ORFs (lower is better).

    Used to arbitrate between the anchor rotation and leaving a record in its
    own numbering: a genuine rotation takes a record from broken to intact
    ORFs, whereas a small anchor offset driven by an *indel* would break an
    otherwise-correct frame.  This makes QC's own criterion the arbiter rather
    than committing to one heuristic.
    """
    from .orfcheck import check_orf

    length = len(seq)
    if length == 0:
        return 0
    spans = (
        (int(get(config, "reference.pol_start_nt", 1) or 1),
         int(get(config, "reference.pol_end_nt", length) or length)),
        (int(get(config, "reference.s_start_nt", 155) or 155),
         int(get(config, "reference.s_end_nt", 835) or 835)),
        (int(get(config, "reference.c_start_nt", 1901) or 1901),
         int(get(config, "reference.c_end_nt", 2450) or 2450)),
    )
    total = 0
    for start, end in spans:
        start = ((start - 1) % length) + 1
        end = ((end - 1) % length) + 1
        total += int(check_orf(seq, start, end, length, max_internal_stops=0)["internal_stops"])
    return total


def circularise_genomes(records: Iterable[GenomeRecord], config) -> list[GenomeRecord]:
    """Recut every record at the reference origin (``reference.origin_nt``).

    Each returned record is a copy whose ``metadata`` records the applied
    ``origin_shift`` (0-based left rotation), the ``origin_nt_used`` and the
    ``origin_method`` that resolved it.  With ``qc.detect_origin`` on and a
    reference sequence available, ``qc.origin_method: auto`` (default) tries:

    1. the exact reference-origin k-mer seed (:func:`find_origin_by_reference`)
       -- origin-anchored and unambiguous, and it resolves the bulk of GenBank
       records (which are rotated, not standard-numbered);
    2. the invariant YMDD catalytic motif
       (:func:`hbvpol.calibrate.orient_to_reference`), consulted only when the
       seed fails -- it tolerates synonymous codon variants and detects the
       opposite strand, which is what rescue divergent-genotype and
       reverse-oriented records.  Because one anchor cannot tell a rotation from
       an indel upstream of the motif, a small implied offset is arbitrated by
       ORF integrity (keep the rotation only when it reduces internal stops);
    3. the configured constant (``reference.origin_nt``).

    ``origin_method: seed`` or ``ymdd`` force one detector; ``constant``
    disables detection.
    """
    configured_origin = int(get(config, "reference.origin_nt", 1) or 1)
    detect = bool(get(config, "qc.detect_origin", False))
    seed_len = int(get(config, "qc.origin_seed_len", 25) or 25)
    method = str(get(config, "qc.origin_method", "auto") or "auto").strip().lower()
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

        # 1. exact reference-origin seed (origin-anchored, unambiguous).
        if ref_seq and method in {"auto", "seed"}:
            found = find_origin_by_reference(seq, ref_seq, seed_len=seed_len)
            if found is not None:
                oriented_seq = rotate_to_origin(seq, found)
                shift = (found - 1) % length
                origin = found
                resolved_by = "seed"

        # 2. YMDD anchor: works for genotypes whose origin k-mer is not
        #    conserved, and detects the opposite strand.  Only consulted when the
        #    seed fails, because the anchor cannot distinguish a rotation from an
        #    indel upstream of the motif.
        if oriented_seq is None and ref_seq and method in {"auto", "ymdd"}:
            anchored = orient_to_reference(seq, ref_seq)
            if anchored is not None:
                oriented_seq, detected_strand, shift = anchored
                strand = "+"
                origin = shift + 1
                resolved_by = "ymdd"
                if bool(get(config, "qc.ymdd_arbitrate_rotation", True)):
                    # A small anchor offset is more often an indel than a real
                    # rotation; let the ORF criterion decide.  A large offset is
                    # an unambiguous rotation.
                    min_rotation = int(get(config, "qc.ymdd_min_rotation", 30) or 0)
                    distance = min(shift % length, (-shift) % length)
                    if distance < min_rotation:
                        unrotated = (
                            reverse_complement(seq)
                            if detected_strand in {"-", "-1"}
                            else seq
                        )
                        if _orf_stop_score(unrotated, config) < _orf_stop_score(oriented_seq, config):
                            oriented_seq = unrotated
                            shift = 0
                            origin = 1
                            resolved_by = "unrotated"

        # 3. configured constant: assume the record already uses the reference
        #    numbering (or that the caller knowingly disabled detection).
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
