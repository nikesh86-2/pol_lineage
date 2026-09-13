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
    "codon_index_from_position",
    "codon_columns_from_positions",
    "reference_codon",
    "reference_nt_position",
    "classify_substitution",
    "dual_frame_table",
    "translate_pol_alignment",
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


# --- reference-frame (column-mapped) codon arithmetic ----------------------- #
#
# When a gapped multiple alignment is used, alignment columns are not reference
# positions, so the codon arithmetic above cannot be applied to a column index.
# The helpers below work in **reference coordinates** and are paired with the
# column -> reference-position map from :mod:`hbvpol.coordinates`.


def codon_index_from_position(position: int, frame_start: int, genome_length: int) -> int:
    """1-based codon index (residue number) of a reference nucleotide position."""
    if genome_length <= 0:
        return 1
    return ((position - frame_start) % genome_length) // 3 + 1


def reference_codon(
    position: int, frame_start: int, genome_length: int
) -> tuple[int, int, list[int]]:
    """Reference positions of the codon containing ``position`` in a frame.

    Returns ``(codon_start_position, within_codon_offset, [positions])`` where
    the three positions wrap the origin.  Mirrors :func:`codon_bounds` but in
    reference coordinates rather than alignment columns.
    """
    if genome_length <= 0:
        return position, 0, [position, position, position]
    relative = (position - frame_start) % 3
    start = ((position - 1 - relative) % genome_length) + 1
    positions = [((start - 1 + k) % genome_length) + 1 for k in range(3)]
    return start, relative, positions


def _position_to_column(positions) -> dict[int, int]:
    """First column holding each reference position (deterministic)."""
    mapping: dict[int, int] = {}
    for column, position in enumerate(positions):
        if position is not None and position not in mapping:
            mapping[position] = column
    return mapping


def codon_columns_from_positions(
    positions, config: Mapping[str, object], genome_length: int
) -> list[list[int | None]]:
    """Pol codon columns in reference order, with ``None`` for missing positions.

    One entry per Pol codon (``reference.pol_start_nt`` -> ``pol_end_nt``), each
    a three-element list of alignment column indices.  A position that the
    column map does not cover (e.g. deleted in the representative) yields
    ``None`` so the codon translates to a gap rather than shifting the frame.
    """
    if genome_length <= 0:
        return []
    position_to_column = _position_to_column(positions)
    pol_start = _pol_start(config)
    pol_end = get(config, "reference.pol_end_nt", None)
    try:
        pol_end = int(pol_end) if pol_end is not None else None
    except (TypeError, ValueError):
        pol_end = None
    if pol_end is not None:
        length_nt = (pol_end - pol_start) % genome_length + 1
    else:
        length_nt = genome_length - pol_start + 1
    n_codons = max(0, length_nt // 3)
    codons: list[list[int | None]] = []
    for index in range(n_codons):
        start = ((pol_start - 1 + 3 * index) % genome_length) + 1
        codons.append(
            [position_to_column.get(((start - 1 + k) % genome_length) + 1) for k in range(3)]
        )
    return codons


def _codon_string(sequence: str, columns: list[int | None]) -> str:
    """Bases at the given columns, or ``-`` for absent (``None``) columns."""
    return "".join(
        sequence[column] if column is not None and column < len(sequence) else "-"
        for column in columns
    )


def translate_pol_alignment(
    alignment,
    config: Mapping[str, object],
    positions=None,
    genome_length: int | None = None,
) -> list[GenomeRecord]:
    """Translate a nucleotide alignment into a Pol **protein** alignment.

    One output character per Pol codon, so output position *i* corresponds to
    Pol amino acid *i* -- the coordinate the atlas and the selection tables use.
    Codons containing a gap or an ambiguous base become ``-`` so that
    DCA/covariation see an aligned protein alignment rather than codon-level
    noise.

    Passing the column -> reference-position ``positions`` map (from
    :func:`hbvpol.coordinates.reference_positions`) selects codons by reference
    coordinate, which is what makes the translation correct on a gapped
    alignment and for genotypes with indels.  Without it the historical
    reference-index arithmetic is used.
    """
    records = coerce_alignment(alignment)
    if not records:
        return []
    width = alignment_width(records)
    if positions is not None and genome_length:
        codon_columns: list[list[int | None]] = codon_columns_from_positions(
            positions, config, genome_length
        )
    else:
        codon_columns = pol_codon_columns(config, width)
    if not codon_columns:
        return []

    translated: list[GenomeRecord] = []
    for record in records:
        seq = record.seq.upper().ljust(width, "-")
        chars: list[str] = []
        for columns in codon_columns:
            codon = _codon_string(seq, columns)
            if any(base not in _ACGT for base in codon):
                chars.append("-")
                continue
            amino_acid = translate(codon, frame=0)
            chars.append(amino_acid if amino_acid and amino_acid != "X" else "-")
        translated.append(
            GenomeRecord(
                id=record.id,
                seq="".join(chars),
                description=record.description,
                source=record.source,
            )
        )
    return translated


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


def _substitute(codon_columns: list[int | None], reference: str, relative: int, alt_nt: str) -> str:
    bases = [
        reference[column] if column is not None and column < len(reference) else "-"
        for column in codon_columns
    ]
    bases[relative] = alt_nt
    return "".join(bases)


def _position_in_frame_range(
    position: int, start_nt: int, end_nt: int | None, genome_length: int
) -> bool:
    """True when ``position`` falls in a (possibly wrapping) reference range."""
    if end_nt is None:
        return position >= start_nt
    start = ((start_nt - 1) % genome_length) + 1
    end = ((end_nt - 1) % genome_length) + 1
    if start <= end:
        return start <= position <= end
    return position >= start or position <= end


def _dual_frame_table_mapped(
    records,
    sequences,
    reference: str,
    positions,
    genome_length: int,
    config: Mapping[str, object],
    output_columns: list[str],
) -> pd.DataFrame:
    """Dual-frame classification using reference coordinates (column-mapped)."""
    pol_start = _pol_start(config)
    offset = int(get(config, "reference.surface_frame_offset", 1) or 0)
    surface_start = ((pol_start - 1 + offset) % genome_length) + 1
    pol_end = get(config, "reference.pol_end_nt", None)
    try:
        pol_end = int(pol_end) if pol_end is not None else None
    except (TypeError, ValueError):
        pol_end = None
    position_to_column = _position_to_column(positions)

    rows: list[dict] = []
    for column, position in enumerate(positions):
        if position is None:
            continue
        if not _position_in_frame_range(position, pol_start, pol_end, genome_length):
            continue
        counts = Counter(seq[column] for seq in sequences if seq[column] in _ACGT)
        if len(counts) < 2:
            continue
        ref_nt = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]

        _, pol_relative, pol_positions = reference_codon(position, pol_start, genome_length)
        _, surface_relative, surface_positions = reference_codon(
            position, surface_start, genome_length
        )
        pol_columns = [position_to_column.get(p) for p in pol_positions]
        surface_columns = [position_to_column.get(p) for p in surface_positions]
        ref_codon_pol = _codon_string(reference, pol_columns)
        ref_codon_surface = _codon_string(reference, surface_columns)
        disrupts = _disrupts_rna(position, config)

        for alt_nt in sorted(counts):
            if alt_nt == ref_nt:
                continue
            result = classify_substitution(
                ref_codon_pol,
                _substitute(pol_columns, reference, pol_relative, alt_nt),
                ref_codon_surface,
                _substitute(surface_columns, reference, surface_relative, alt_nt),
                disrupts_rna=disrupts,
            )
            rows.append({
                "nucleotide_position": position,
                "ref_nt": ref_nt,
                "pol_codon_position": codon_index_from_position(position, pol_start, genome_length),
                "pol_ref_aa": result["pol_ref_aa"],
                "pol_alt_aa": result["pol_alt_aa"],
                "surface_position": codon_index_from_position(position, surface_start, genome_length),
                "surface_ref_aa": result["surface_ref_aa"],
                "surface_alt_aa": result["surface_alt_aa"],
                "consequence_class": result["consequence_class"],
                "disrupts_rna": result["disrupts_rna"],
            })
    return pd.DataFrame(rows, columns=output_columns)


def dual_frame_table(
    alignment,
    config: Mapping[str, object],
    positions=None,
    genome_length: int | None = None,
) -> pd.DataFrame:
    """Classify every variable nucleotide site in both reading frames.

    For each alignment column with two or more unambiguous alleles, the
    majority base is the reference and every other allele is classified once.
    The returned frame uses :data:`DUAL_FRAME_COLUMNS` *without* the leading
    ``lineage`` column, which the pipeline adds when it iterates lineages.

    Passing the column -> reference-position ``positions`` map (and the reference
    ``genome_length``) makes every reported position a reference coordinate and
    handles gapped alignments/indels correctly; without it the historical
    reference-index arithmetic is used.
    """
    records = coerce_alignment(alignment)
    output_columns = [column for column in DUAL_FRAME_COLUMNS if column != "lineage"]
    if not records:
        return pd.DataFrame(columns=output_columns)

    width = alignment_width(records)
    if width == 0:
        return pd.DataFrame(columns=output_columns)

    reference = default_reference_sequence(records)
    sequences = [record.seq.upper().ljust(width, "-") for record in records]

    if positions is not None and genome_length:
        return _dual_frame_table_mapped(
            records, sequences, reference, positions, genome_length, config, output_columns
        )

    pol_start = _pol_start(config)
    surface_start = surface_frame_start(config, width)

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
        ref_codon_pol = _codon_string(reference, pol_columns)
        ref_codon_surface = _codon_string(reference, surface_columns)
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
