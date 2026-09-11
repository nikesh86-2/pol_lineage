"""Dual-frame (Pol / surface) substitution classification — the core method.

HBV polymerase (P) and surface antigen (S) are translated from *overlapping*
open reading frames in different reading frames.  A single nucleotide
substitution therefore has to be interpreted twice: once in the Pol frame and
once in the overlapping surface frame.  Some substitutions are synonymous in
one frame and amino-acid-changing in the other, which is the central reason why
naive single-frame selection scans mis-annotate HBV.

Frame arithmetic
----------------
Let ``w`` be the alignment width (columns, 0-based), ``P`` the 1-based reference
position of the Pol initiator codon (``reference.pol_start_nt``) and ``o`` the
configurable ``reference.surface_frame_offset`` (default ``1``).  The surface
frame is offset by ``o`` nucleotides relative to Pol, so its first codon begins
at

    S = ((P - 1 + o) mod w) + 1        (1-based)

For an alignment column ``c`` in a frame whose first codon base is ``F``, the
position within the codon and the three column indices are

    rel     = (c - (F - 1)) mod 3
    start   = c - rel
    codon   = [(start + k) mod w for k in range(3)]

The Pol codon index of that column is ``((start - (P - 1)) mod w) // 3 + 1``;
the surface codon index is computed the same way with ``S``.  A substitution at
column ``c`` is applied to the reference codon by replacing exactly one base,
so each variable site is classified independently of the others.  The origin
offset (``reference.origin_nt``) only affects the *reported* reference
coordinate of a column, which is how optional ``reference.rna_elements``
(``{start, end, name}`` in reference coordinates) are matched.

All functions here are pure and work offline on an in-memory multiple alignment
of nucleotide sequences.
"""

from __future__ import annotations

from collections import Counter
from typing import Mapping

import pandas as pd

from ..config import get
from ..domain import dual_frame_consequence, frame_indices, translate
from ..io import GenomeRecord, read_fasta

__all__ = [
    "DUAL_FRAME_CLASSES",
    "DUAL_FRAME_COLUMNS",
    "coerce_alignment",
    "default_reference_sequence",
    "pol_codon_columns",
    "codon_bounds",
    "reference_nt_position",
    "classify_substitution",
    "dual_frame_table",
]

#: The five documented outcome classes (plus ``unknown``) from hbvpol.domain.
DUAL_FRAME_CLASSES = (
    "synonymous_both",
    "nonsynonymous_pol_only",
    "nonsynonymous_surface_only",
    "nonsynonymous_both",
    "disruptive_rna_element",
    "unknown",
)

#: Full output schema of ``dual_frame.tsv`` (``lineage`` is added by the pipeline).
DUAL_FRAME_COLUMNS = [
    "lineage",
    "nucleotide_position",
    "ref_nt",
    "pol_codon_position",
    "pol_ref_aa",
    "pol_alt_aa",
    "surface_position",
    "surface_ref_aa",
    "surface_alt_aa",
    "consequence_class",
    "disrupts_rna",
]

_ACGT = frozenset("ACGT")


def coerce_alignment(alignment) -> list[GenomeRecord]:
    """Coerce a path/record list/pair list/mapping into ``GenomeRecord`` objects.

    Shared by the selection modules so that every analysis accepts the same
    alignment spellings while keeping its public function signatures clean.
    """
    if alignment is None:
        return []
    if isinstance(alignment, (str,)):
        return read_fasta(alignment)
    from pathlib import Path

    if isinstance(alignment, Path):
        return read_fasta(alignment)
    if isinstance(alignment, Mapping):
        return _to_records(list(alignment.items()))
    if isinstance(alignment, list):
        if not alignment:
            return []
        if isinstance(alignment[0], GenomeRecord):
            return list(alignment)
        return _to_records(alignment)
    return coerce_alignment(list(alignment))


def _to_records(items) -> list[GenomeRecord]:
    records: list[GenomeRecord] = []
    for index, item in enumerate(items):
        if isinstance(item, GenomeRecord):
            records.append(item)
        elif isinstance(item, (tuple, list)) and len(item) >= 2:
            records.append(GenomeRecord(id=str(item[0]), seq=str(item[1])))
        elif hasattr(item, "id") and hasattr(item, "seq"):
            records.append(GenomeRecord(id=str(item.id), seq=str(item.seq)))
        else:
            records.append(GenomeRecord(id=str(index + 1), seq=str(item)))
    return records


def alignment_width(records: list[GenomeRecord]) -> int:
    """Width (number of columns) of the alignment implied by ``records``."""
    return max((len(record.seq) for record in records), default=0)


def default_reference_sequence(records: list[GenomeRecord]) -> str:
    """Build a consensus reference row (majority unambiguous base per column).

    Ties are broken alphabetically, so the result is deterministic.  Columns
    with no unambiguous base become ``-``.
    """
    width = alignment_width(records)
    if width == 0:
        return ""
    padded = [record.seq.upper().ljust(width, "-") for record in records]
    reference: list[str] = []
    for column in range(width):
        counts = Counter(seq[column] for seq in padded if seq[column] in _ACGT)
        if counts:
            reference.append(sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0])
        else:
            reference.append("-")
    return "".join(reference)


def _pol_start(config: Mapping[str, object]) -> int:
    return int(get(config, "reference.pol_start_nt", 1) or 1)


def pol_codon_columns(config: Mapping[str, object], width: int) -> list[list[int]]:
    """Return the 0-based column indices of every Pol codon, in order.

    The Pol ORF is walked from ``reference.pol_start_nt``.  When
    ``reference.pol_end_nt`` is set the ORF length wraps the origin; otherwise
    the ORF is taken to run to the end of the alignment.  This mirrors the
    coordinate handling of :func:`hbvpol.domain.frame_indices`.
    """
    if width <= 0:
        return []
    pol_start = _pol_start(config)
    pol_end = get(config, "reference.pol_end_nt", None)
    if pol_end is not None:
        try:
            length_nt = (int(pol_end) - pol_start) % width + 1
        except (TypeError, ValueError):
            length_nt = width - (pol_start - 1)
    else:
        length_nt = width - (pol_start - 1)
    n_codons = max(0, length_nt // 3)
    return [frame_indices(pol_start + 3 * index, 3, width) for index in range(n_codons)]


def codon_bounds(nt_index: int, frame_start: int, width: int) -> tuple[list[int], int]:
    """Return the codon column indices and within-codon offset for a column.

    ``frame_start`` is the 1-based position of the frame's first codon base.
    Codons wrap the origin, so indices are taken modulo ``width``.
    """
    relative = (nt_index - (frame_start - 1)) % 3
    start = nt_index - relative
    columns = [(start + k) % width for k in range(3)]
    return columns, relative


def _codon_index(start_column: int, frame_start: int, width: int) -> int:
    return ((start_column - (frame_start - 1)) % width) // 3 + 1


def surface_frame_start(config: Mapping[str, object], width: int) -> int:
    """1-based position of the surface frame's first codon base."""
    offset = int(get(config, "reference.surface_frame_offset", 1) or 0)
    return ((_pol_start(config) - 1 + offset) % width) + 1


def reference_nt_position(nt_index: int, config: Mapping[str, object], width: int) -> int:
    """Map a 0-based alignment column to a 1-based reference coordinate.

    The alignment is oriented at the reference origin, so column 0 is
    ``reference.origin_nt``.  Missing/zero values default to identity
    (``nt_index + 1``).
    """
    origin = int(get(config, "reference.origin_nt", 1) or 1)
    genome_length = int(get(config, "reference.length", width) or width)
    if genome_length <= 0:
        genome_length = width or 1
    return ((origin - 1 + nt_index) % genome_length) + 1


def _disrupts_rna(reference_position: int, config: Mapping[str, object]) -> bool:
    elements = get(config, "reference.rna_elements", None) or []
    for element in elements:
        try:
            start = int(element.get("start"))
            end = int(element.get("end"))
        except (AttributeError, TypeError, ValueError):
            continue
        if start <= reference_position <= end:
            return True
    return False


def classify_substitution(
    ref_codon_pol: str,
    alt_codon_pol: str,
    ref_codon_surface: str,
    alt_codon_surface: str,
    disrupts_rna: bool = False,
) -> dict:
    """Classify one substitution across the Pol and surface frames.

    Thin, documented wrapper around
    :func:`hbvpol.domain.dual_frame_consequence` that also reports the
    translated reference/alternate amino acids in both frames.
    """
    consequence = dual_frame_consequence(
        ref_codon_pol, alt_codon_pol, ref_codon_surface, alt_codon_surface,
        disrupts_rna=disrupts_rna,
    )
    return {
        "consequence_class": consequence.value,
        "pol_ref_aa": translate(ref_codon_pol),
        "pol_alt_aa": translate(alt_codon_pol),
        "surface_ref_aa": translate(ref_codon_surface),
        "surface_alt_aa": translate(alt_codon_surface),
        "disrupts_rna": bool(disrupts_rna),
    }


def _substitute(codon_columns: list[int], reference: str, relative: int, alt_nt: str) -> str:
    bases = [reference[column] if column < len(reference) else "-" for column in codon_columns]
    bases[relative] = alt_nt
    return "".join(bases)


def dual_frame_table(alignment, config: Mapping[str, object]) -> pd.DataFrame:
    """Classify every variable nucleotide site in both reading frames.

    For each alignment column with two or more unambiguous alleles, the
    majority base is the reference and every other allele is classified once.
    The returned frame uses :data:`DUAL_FRAME_COLUMNS` *without* the leading
    ``lineage`` column, which the pipeline adds when it iterates lineages.

    Works offline on any multiple alignment of nucleotide sequences.
    """
    records = coerce_alignment(alignment)
    output_columns = [column for column in DUAL_FRAME_COLUMNS if column != "lineage"]
    if not records:
        return pd.DataFrame(columns=output_columns)

    width = alignment_width(records)
    if width == 0:
        return pd.DataFrame(columns=output_columns)

    pol_start = _pol_start(config)
    surface_start = surface_frame_start(config, width)
    reference = default_reference_sequence(records)
    sequences = [record.seq.upper().ljust(width, "-") for record in records]

    # Only columns inside the Pol ORF carry a meaningful Pol codon/amino acid;
    # classifying genome-wide sites would report Pol positions that do not exist.
    pol_columns_valid: set[int] = set()
    for codon in pol_codon_columns(config, width):
        pol_columns_valid.update(codon)

    rows: list[dict] = []
    for column in range(width):
        if pol_columns_valid and column not in pol_columns_valid:
            continue
        counts = Counter(seq[column] for seq in sequences if seq[column] in _ACGT)
        if len(counts) < 2:
            continue
        ref_nt = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]

        pol_columns, pol_relative = codon_bounds(column, pol_start, width)
        surface_columns, surface_relative = codon_bounds(column, surface_start, width)
        ref_codon_pol = "".join(
            reference[index] if index < len(reference) else "-" for index in pol_columns
        )
        ref_codon_surface = "".join(
            reference[index] if index < len(reference) else "-" for index in surface_columns
        )
        reference_position = reference_nt_position(column, config, width)
        disrupts = _disrupts_rna(reference_position, config)

        for alt_nt in sorted(counts):
            if alt_nt == ref_nt:
                continue
            alt_codon_pol = _substitute(pol_columns, reference, pol_relative, alt_nt)
            alt_codon_surface = _substitute(surface_columns, reference, surface_relative, alt_nt)
            result = classify_substitution(
                ref_codon_pol, alt_codon_pol, ref_codon_surface, alt_codon_surface,
                disrupts_rna=disrupts,
            )
            rows.append({
                "nucleotide_position": reference_position,
                "ref_nt": ref_nt,
                "pol_codon_position": _codon_index(pol_columns[0], pol_start, width),
                "pol_ref_aa": result["pol_ref_aa"],
                "pol_alt_aa": result["pol_alt_aa"],
                "surface_position": _codon_index(surface_columns[0], surface_start, width),
                "surface_ref_aa": result["surface_ref_aa"],
                "surface_alt_aa": result["surface_alt_aa"],
                "consequence_class": result["consequence_class"],
                "disrupts_rna": result["disrupts_rna"],
            })

    return pd.DataFrame(rows, columns=output_columns)
