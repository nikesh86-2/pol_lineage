"""Open-reading-frame integrity checks used by the QC stage.

HBV packs five overlapping genes and regulatory elements into ~3.2 kb, so a
single indel can destroy one reading frame while leaving its neighbours intact.
This module translates the canonical Pol ORF (which wraps the origin) and the
two other protein-coding frames required by configuration, and counts internal
stop codons as the primary indicator of a broken sequence.

Coordinate frames
-----------------
``reference.pol_start_nt`` / ``reference.pol_end_nt`` (and the optional S/C
coordinates) are expressed in the **reference genome's native numbering**.
:func:`check_genome` lifts them into the oriented frame produced by
:mod:`hbvpol.qc.circularise` using the ``origin_shift`` custom field, so the
checks are applied to the actual recut sequence.  S and C coordinates are not
present in the shipped config; the defaults below are the standard genotype A2
(NC_003977) spans over the gene bodies and are **fully configurable** via
``reference.s_start_nt``/``reference.s_end_nt`` and
``reference.c_start_nt``/``reference.c_end_nt``.
"""

from __future__ import annotations

from ..config import get
from ..domain import frame_indices, translate
from ..io import GenomeRecord, ambiguous_fraction

__all__ = ["check_orf", "check_genome", "QC_COLUMNS"]

# Columns of ``hbv_qc_pass.tsv`` / ``hbv_qc_fail.tsv``, in order.
QC_COLUMNS = [
    "accession",
    "length",
    "ambiguous_frac",
    "pol_start",
    "pol_end",
    "pol_stops",
    "orf_pol",
    "orf_s",
    "orf_c",
    "qc_pass",
    "fail_reason",
]

# Default genotype A2 ORF spans (see module docstring: configurable).
_DEFAULT_S_SPAN = (155, 835)      # HBsAg / surface
_DEFAULT_C_SPAN = (1901, 2450)    # core / capsid


def check_orf(
    seq: str,
    start_nt: int,
    end_nt: int,
    genome_length: int,
    max_internal_stops: int = 0,
    frame_offset: int = 0,
) -> dict:
    """Translate an ORF (which may wrap the origin) and count stop codons.

    Coordinates are 1-based and inclusive.  When ``end_nt < start_nt`` the ORF
    wraps past ``genome_length``; :func:`hbvpol.domain.frame_indices` is used to
    walk the circular indices, with ``frame_offset`` shifting the reading frame
    (used for the overlapping surface frame).

    Returns a dictionary with the translated ``protein``, raw ``n_stops``, the
    number of ``internal_stops`` (a terminal stop is not counted) and a
    ``complete`` flag that is true when the internal stop count does not exceed
    ``max_internal_stops``.
    """
    if genome_length <= 0:
        raise ValueError("genome_length must be positive")
    if end_nt >= start_nt:
        length_nt = end_nt - start_nt + 1
    else:
        # Origin-wrapping ORF: start -> genome_length, then 1 -> end.
        length_nt = (genome_length - start_nt + 1) + end_nt

    indices = frame_indices(start_nt, length_nt, genome_length, frame_offset)
    n = len(seq)
    if n:
        nt = "".join(seq[i % n] for i in indices)
    else:
        nt = ""
    protein = translate(nt, frame=0) if nt else ""

    n_stops = protein.count("*")
    internal_stops = n_stops - (1 if protein.endswith("*") else 0)
    internal_stops = max(0, internal_stops)

    return {
        "start_nt": start_nt,
        "end_nt": end_nt,
        "length_nt": length_nt,
        "frame_offset": frame_offset,
        "nt": nt,
        "protein": protein,
        "n_stops": n_stops,
        "internal_stops": internal_stops,
        "has_start": protein.startswith("M"),
        "has_stop": protein.endswith("*"),
        "complete": internal_stops <= max_internal_stops and length_nt >= 3,
    }


def _lift(record: GenomeRecord, position_nt: int, genome_length: int) -> int:
    """Lift a native reference coordinate into the record's oriented frame."""
    if genome_length <= 0:
        return position_nt
    shift = int(record.metadata.get("origin_shift", 0) or 0)
    if shift == 0:
        return position_nt
    return ((position_nt - 1 - shift) % genome_length) + 1


def check_genome(record: GenomeRecord, config) -> dict:
    """Run all QC checks for one genome and return a ``hbv_qc_pass.tsv`` row.

    The row carries the fields consumed downstream: length, ambiguous
    fraction, Pol ORF coordinates and internal stop count, per-ORF integrity
    flags for Pol/S/C, an overall ``qc_pass`` decision and a ``fail_reason``
    string (semicolon-separated codes, empty when passing).
    """
    seq = record.seq
    length = len(seq)

    if length == 0:
        return {
            "accession": record.id,
            "length": 0,
            "ambiguous_frac": 0.0,
            "pol_start": 0,
            "pol_end": 0,
            "pol_stops": 0,
            "orf_pol": False,
            "orf_s": False,
            "orf_c": False,
            "qc_pass": False,
            "fail_reason": "empty",
        }

    min_length = int(get(config, "qc.min_length", 0) or 0)
    max_length = get(config, "qc.max_length", None)
    max_length = int(max_length) if max_length not in (None, "", 0) else None
    max_ambiguous = float(get(config, "qc.max_ambiguous_frac", 1.0) or 1.0)
    max_stops = int(get(config, "qc.max_internal_stops", 0) or 0)
    exclude_pol_stop = bool(get(config, "qc.exclude_stop_in_pol", True))
    require = list(get(config, "qc.require_complete_orf", ["pol", "s", "c"]) or [])

    raw_spans = {
        "pol": (
            int(get(config, "reference.pol_start_nt", 1) or 1),
            int(get(config, "reference.pol_end_nt", length) or length),
        ),
        "s": (
            int(get(config, "reference.s_start_nt", _DEFAULT_S_SPAN[0]) or _DEFAULT_S_SPAN[0]),
            int(get(config, "reference.s_end_nt", _DEFAULT_S_SPAN[1]) or _DEFAULT_S_SPAN[1]),
        ),
        "c": (
            int(get(config, "reference.c_start_nt", _DEFAULT_C_SPAN[0]) or _DEFAULT_C_SPAN[0]),
            int(get(config, "reference.c_end_nt", _DEFAULT_C_SPAN[1]) or _DEFAULT_C_SPAN[1]),
        ),
    }

    spans: dict[str, tuple[int, int]] = {}
    orf_results: dict[str, dict] = {}
    for name, (start, end) in raw_spans.items():
        lifted = (_lift(record, start, length), _lift(record, end, length))
        spans[name] = lifted
        orf_results[name] = check_orf(seq, lifted[0], lifted[1], length, max_internal_stops=max_stops)

    reasons: list[str] = []
    if length < min_length:
        reasons.append("length")
    if max_length is not None and length > max_length:
        # Concatemers and multi-genome constructs survive a min-length filter.
        reasons.append("too_long")
    if ambiguous_fraction(seq) > max_ambiguous:
        reasons.append("ambiguous")

    for name in ("pol", "s", "c"):
        if name in require and not orf_results[name]["complete"]:
            reasons.append(f"orf_{name}")

    if exclude_pol_stop and orf_results["pol"]["internal_stops"] > 0:
        reasons.append("stop_in_pol")

    return {
        "accession": record.id,
        "length": length,
        "ambiguous_frac": round(ambiguous_fraction(seq), 6),
        "pol_start": spans["pol"][0],
        "pol_end": spans["pol"][1],
        "pol_stops": int(orf_results["pol"]["internal_stops"]),
        "orf_pol": bool(orf_results["pol"]["complete"]),
        "orf_s": bool(orf_results["s"]["complete"]),
        "orf_c": bool(orf_results["c"]["complete"]),
        "qc_pass": not reasons,
        "fail_reason": ";".join(reasons),
    }
