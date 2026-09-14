"""Reconciliation and partitioning of recombination breakpoints.

The individual detection tools (RDP5, GARD, 3SEQ, bootscan) return breakpoints
in different formats and with different notions of what a "breakpoint" is.
This module provides the pure, side-effect-free core that turns those frames
into a consensus set and then into non-recombinant alignment blocks:

    normalise_breakpoints   map a tool-specific frame onto the tidy schema
    reconcile_breakpoints   cluster nearby calls and mark tool consensus
    partition_alignment     cut the alignment into >= min_block_len blocks
    extract_domain_subalignments   slice Pol domains out of the alignment

All functions are deterministic and operate on in-memory data, so they can be
unit-tested without any external binary.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import get
from ..coordinates import columns_for_nt_range, reference_positions, reference_sequence_from_config
from ..domain import PolDomain, domain_spans_from_config, frame_indices
from ..io import GenomeRecord, read_fasta, write_fasta
from ..pipeline import get_logger

logger = get_logger("recombination.partition")

__all__ = [
    "BREAKPOINT_COLUMNS",
    "BLOCK_COLUMNS",
    "normalise_breakpoints",
    "reconcile_breakpoints",
    "partition_alignment",
    "extract_domain_subalignments",
    "load_alignment",
]

# Tidy breakpoint schema written to ``breakpoints.tsv``.
BREAKPOINT_COLUMNS = [
    "recombinant_id",
    "partner",
    "tool",
    "bp_start",
    "bp_end",
    "support",
    "region",
]

# Tidy block schema written to ``blocks.tsv``.
BLOCK_COLUMNS = ["block_id", "start_nt", "end_nt", "n_seqs", "n_tools_supporting"]


def _empty_breakpoints() -> pd.DataFrame:
    return pd.DataFrame(columns=BREAKPOINT_COLUMNS)


def load_alignment(alignment) -> list[GenomeRecord]:
    """Coerce an alignment argument to a list of :class:`GenomeRecord`.

    Accepts a path to a FASTA file, a list of ``GenomeRecord`` objects, or a
    list/iterable of ``(id, sequence)`` pairs (and bare sequences, which get
    positional ids).  This keeps the parsers and partitioner usable from both
    the pipeline (paths) and tests (in-memory data).
    """
    if isinstance(alignment, (str, Path)):
        return read_fasta(alignment)
    if isinstance(alignment, list):
        if not alignment:
            return []
        if isinstance(alignment[0], GenomeRecord):
            return list(alignment)
        records: list[GenomeRecord] = []
        for index, item in enumerate(alignment):
            if isinstance(item, (tuple, list)) and len(item) >= 2:
                records.append(GenomeRecord(id=str(item[0]), seq=str(item[1])))
            else:
                records.append(GenomeRecord(id=str(index + 1), seq=str(item)))
        return records
    if alignment is None:
        return []
    # Generic iterable of GenomeRecord.
    return list(alignment)


def _pick(columns: dict[str, str], *names: str) -> str | None:
    for name in names:
        if name in columns:
            return columns[name]
    return None


def normalise_breakpoints(frame: pd.DataFrame, tool: str) -> pd.DataFrame:
    """Map a tool-specific breakpoint frame onto the tidy schema.

    Column matching is case-insensitive and tolerant of the many aliases used
    by RDP5/GARD/3SEQ/bootscan.  Missing coordinates fall back sensibly (a
    single-site breakpoint) and missing support defaults to 1.0.
    """
    if frame is None or len(frame) == 0:
        return _empty_breakpoints()

    df = frame.copy()
    lower = {str(column).strip().lower(): column for column in df.columns}

    rcol = _pick(lower, "recombinant_id", "recombinant", "recombinant sequence",
                 "recombinant sequence(s)", "query", "sequence", "c_accnum")
    pcol = _pick(lower, "partner", "minor parent", "major parent", "parent",
                 "reference", "parent1", "parent2", "p_accnum", "q_accnum")
    scol = _pick(lower, "bp_start", "begin", "start", "breakpoint begin",
                 "breakpoint_start", "position", "breakpoint")
    ecol = _pick(lower, "bp_end", "end", "breakpoint end", "breakpoint_end")
    supcol = _pick(lower, "support", "p-value", "p value", "pvalue", "score")
    regcol = _pick(lower, "region", "domain", "gene")

    starts = pd.to_numeric(df[scol], errors="coerce") if scol else pd.Series(np.nan, index=df.index)
    ends = pd.to_numeric(df[ecol], errors="coerce") if ecol else starts.copy()

    out = pd.DataFrame(index=df.index)
    out["recombinant_id"] = df[rcol].astype(str) if rcol else "unknown"
    out["partner"] = df[pcol].astype(str) if pcol else ""
    out["tool"] = tool
    out["bp_start"] = starts.fillna(1).astype(int)
    out["bp_end"] = ends.fillna(starts).fillna(1).astype(int)
    out["support"] = (
        pd.to_numeric(df[supcol], errors="coerce").fillna(1.0) if supcol else 1.0
    )
    out["region"] = df[regcol].astype(str) if regcol else ""

    return out[BREAKPOINT_COLUMNS].reset_index(drop=True)


def reconcile_breakpoints(frames: list[pd.DataFrame], config) -> pd.DataFrame:
    """Merge tool outputs and mark breakpoints supported by enough tools.

    Nearby calls (midpoints within ``recombination.breakpoint_tolerance`` nt)
    on the same ``recombinant_id`` are clustered.  A cluster is retained as a
    consensus breakpoint when the number of *distinct* tools supporting it is
    at least ``ceil(recombination.consensus_frac * n_tools_effective)``, where
    ``n_tools_effective`` counts only the tools that actually produced calls
    (an uninstalled tool cannot support anything, and counting it would discard
    every breakpoint when only the offline scan is available).

    Returns one row per cluster with the tidy schema plus ``cluster_id``,
    ``n_tools_supporting``, ``n_tools_total`` and a boolean ``consensus``.
    """
    tolerance = int(get(config, "recombination.breakpoint_tolerance", 50) or 50)
    consensus_frac = float(get(config, "recombination.consensus_frac", 0.5) or 0.5)

    non_empty = [frame for frame in frames if frame is not None and len(frame) > 0]
    extra = ["cluster_id", "n_tools_supporting", "n_tools_total", "consensus"]
    if not non_empty:
        return pd.DataFrame(columns=BREAKPOINT_COLUMNS + extra)

    combined = pd.concat(
        [normalise_breakpoints(frame, tool=_tool_label(frame)) for frame in non_empty],
        ignore_index=True,
    )

    observed_tools = {str(tool) for tool in combined["tool"].dropna().unique()}
    configured = get(config, "recombination.tools", None)
    configured_count = len(configured) if configured else len(observed_tools)
    # Count only tools that actually ran: an uninstalled tool cannot support a
    # breakpoint, and counting it silently discards every call when, e.g., only
    # the offline bootscan scan is available.
    n_tools_total = max(1, len(observed_tools) or configured_count)
    needed = max(1, math.ceil(consensus_frac * n_tools_total))
    if observed_tools and len(observed_tools) < configured_count:
        logger.warning(
            "consensus computed from %d of %d configured tools (%s)",
            len(observed_tools), configured_count, ",".join(sorted(observed_tools)),
        )

    combined = combined.assign(
        _mid=(combined["bp_start"].astype(float) + combined["bp_end"].astype(float)) / 2.0
    )

    clusters: list[dict] = []
    cluster_id = 0
    for recombinant_id, group in combined.groupby("recombinant_id", sort=True, dropna=False):
        group = group.sort_values("_mid")
        local: list[dict] = []
        for _, row in group.iterrows():
            placed = False
            for cluster in local:
                if row["_mid"] - cluster["_max_mid"] <= tolerance:
                    cluster["rows"].append(row)
                    cluster["_max_mid"] = max(cluster["_max_mid"], row["_mid"])
                    placed = True
                    break
            if not placed:
                local.append({"rows": [row], "_max_mid": row["_mid"]})

        for cluster in local:
            rows = cluster["rows"]
            cluster_id += 1
            tools = sorted({str(row["tool"]) for row in rows})
            partners = [str(row["partner"]) for row in rows if str(row["partner"]).strip()]
            # Deterministic tie-break (most frequent, then lexicographic) so the
            # chosen partner does not depend on set iteration order.
            partner = (
                min(partners, key=lambda value: (-partners.count(value), value))
                if partners else ""
            )
            supports = [
                float(row["support"]) if pd.notna(row["support"]) else 1.0 for row in rows
            ]
            clusters.append({
                "recombinant_id": recombinant_id,
                "partner": partner,
                "tool": "+".join(tools),
                "bp_start": int(min(row["bp_start"] for row in rows)),
                "bp_end": int(max(row["bp_end"] for row in rows)),
                "support": float(np.mean(supports)) if supports else 1.0,
                "region": str(rows[0].get("region", "") or ""),
                "cluster_id": cluster_id,
                "n_tools_supporting": len(tools),
                "n_tools_total": n_tools_total,
                "consensus": len(tools) >= needed,
            })

    return pd.DataFrame(clusters, columns=BREAKPOINT_COLUMNS + extra)


def _tool_label(frame: pd.DataFrame) -> str:
    """Recover the tool name from a frame if it carries one, else 'combined'."""
    if "tool" in frame.columns and len(frame):
        values = [str(value) for value in frame["tool"].dropna().unique()]
        if len(values) == 1:
            return values[0]
    return "combined"


def _merge_short_blocks(blocks: list[tuple[int, int]], min_block_len: int) -> list[tuple[int, int]]:
    """Merge any block shorter than ``min_block_len`` into a neighbour."""
    merged = list(blocks)
    changed = True
    while changed and len(merged) > 1:
        changed = False
        for index, (start, end) in enumerate(merged):
            if end - start + 1 >= min_block_len:
                continue
            if index == 0:
                _, next_end = merged[1]
                merged[0] = (start, next_end)
                del merged[1]
            else:
                prev_start, _ = merged[index - 1]
                merged[index - 1] = (prev_start, end)
                del merged[index]
            changed = True
            break
    return merged


def partition_alignment(
    alignment, breakpoints: pd.DataFrame, config
) -> tuple[pd.DataFrame, dict[str, Path]]:
    """Cut an alignment into non-recombinant blocks.

    Consensus breakpoint midpoints become cut sites; the resulting intervals
    are merged until every block is at least ``recombination.min_block_len`` nt
    (a block is merged into its predecessor, or into the following block when
    it is the first).  Each retained block is written to
    ``<recombination.block_aln_dir>/block_<id>.fasta`` and described by a
    ``blocks.tsv``-compatible row.

    Returns the block table and a mapping ``{"block_<id>": path}``.
    """
    records = load_alignment(alignment)
    if not records:
        return pd.DataFrame(columns=BLOCK_COLUMNS), {}

    width = max(len(record.seq) for record in records)
    min_block_len = int(get(config, "recombination.min_block_len", 1) or 1)
    block_dir = Path(str(get(config, "recombination.block_aln_dir", "block_alns")))

    tools_cfg = get(config, "recombination.tools", None)
    n_tools_total = len(tools_cfg) if tools_cfg else 1

    support_at: dict[int, int] = {}
    cuts: list[int] = []
    if breakpoints is not None and len(breakpoints) > 0:
        table = breakpoints
        if "consensus" in table.columns:
            table = table[table["consensus"].astype(bool)]
        for _, row in table.iterrows():
            mid = int((int(row["bp_start"]) + int(row["bp_end"])) // 2)
            if 1 <= mid < width:
                cuts.append(mid)
                support = int(row.get("n_tools_supporting", 1) or 1)
                support_at[mid] = max(support_at.get(mid, 0), support)
        if "n_tools_total" in table.columns and len(table):
            n_tools_total = max(n_tools_total, int(table["n_tools_total"].max()))

    cuts = sorted(set(cuts))
    ends = cuts + [width]
    starts = [1] + [cut + 1 for cut in cuts]
    raw_blocks = list(zip(starts, ends))
    merged_blocks = _merge_short_blocks(raw_blocks, min_block_len)

    rows: list[dict] = []
    paths: dict[str, Path] = {}
    for block_id, (start, end) in enumerate(merged_blocks, start=1):
        boundary_supports = [
            support_at[position]
            for position in (start - 1, end)
            if position in support_at
        ]
        n_supporting = min(boundary_supports) if boundary_supports else n_tools_total

        sub_records = [
            GenomeRecord(
                id=record.id,
                seq=record.seq[start - 1:end],
                description=record.description,
                source=record.source,
                strand=record.strand,
                metadata=dict(record.metadata),
            )
            for record in records
        ]
        path = block_dir / f"block_{block_id}.fasta"
        write_fasta(sub_records, path)
        paths[f"block_{block_id}"] = path
        rows.append({
            "block_id": block_id,
            "start_nt": start,
            "end_nt": end,
            "n_seqs": len(sub_records),
            "n_tools_supporting": n_supporting,
        })

    blocks = pd.DataFrame(rows, columns=BLOCK_COLUMNS)
    return blocks, paths


def extract_domain_subalignments(alignment, config) -> dict[PolDomain, str]:
    """Slice the Pol TP/spacer/RT/RNaseH domains out of an alignment.

    ``DEFAULT_DOMAIN_SPANS`` are amino-acid spans within the Pol polypeptide;
    they are converted to nucleotide ranges using ``reference.pol_start_nt``
    and the triplet periodicity of the Pol frame, then mapped onto alignment
    columns.

    Coordinate-system contract
    --------------------------
    Column indices are selected by **reference coordinate**, using the
    column -> reference-position map from :mod:`hbvpol.coordinates` (built from a
    single alignment of the configured reference genome to a representative row).
    That is what makes the slices correct on a gapped multiple alignment and on
    genotypes that carry indels: insertion columns are assigned to the domain of
    their nearest reference-anchored neighbour rather than being dropped or
    shifting everything downstream.  When no reference sequence is configured,
    the historical "column ``i`` == reference position ``i + 1``" arithmetic is
    used instead (documented fallback, exercised by the offline tests).

    Returns a mapping from :class:`PolDomain` to a FASTA-format string.
    """
    records = load_alignment(alignment)
    if not records:
        return {}

    width = max(len(record.seq) for record in records)
    pol_start = int(get(config, "reference.pol_start_nt", 1) or 1)
    positions = reference_positions(records, config)
    reference = reference_sequence_from_config(config)
    genome_length = len(reference) if reference else width

    result: dict[PolDomain, str] = {}
    for span in domain_spans_from_config(config):
        aa_start, aa_end = span.start, span.end
        nt_start = pol_start + (aa_start - 1) * 3
        if positions is None:
            indices = frame_indices(nt_start, (aa_end - aa_start + 1) * 3, width)
        else:
            nt_end = pol_start + (aa_end - 1) * 3 + 2
            indices = columns_for_nt_range(positions, nt_start, nt_end, genome_length)

        lines: list[str] = []
        for record in records:
            seq = record.seq
            sub = "".join(seq[i] if i < len(seq) else "-" for i in indices)
            lines.append(f">{record.id}\n{sub}")
        result[span.domain] = "\n".join(lines) + "\n"

    return result
