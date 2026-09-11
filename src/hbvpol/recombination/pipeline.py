"""Recombination stage pipeline.

Orchestrates the available detection tools, reconciles their breakpoints into a
consensus set, partitions the alignment into non-recombinant blocks, extracts
Pol-domain sub-alignments and writes the inter-stage contract artefacts:

    <outroot>/recombination/breakpoints.tsv
    <outroot>/recombination/blocks.tsv
    <outroot>/recombination/block_alns/block_<id>.fasta
    <outroot>/recombination/domain_subalns/<domain>.fasta
    <outroot>/recombination/recombination_summary.json

Every external tool is optional: tools that are unavailable are skipped with a
warning and the stage still completes using the tools that are present (and,
for bootscan, the offline pure-Python scan).
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pandas as pd

from ..config import get
from ..pipeline import StageError, get_logger, have_executable, output_dir, run_command, stage_dir
from .bootscan import run_bootscan
from .gard import run_gard
from .partition import (
    BREAKPOINT_COLUMNS,
    extract_domain_subalignments,
    partition_alignment,
    reconcile_breakpoints,
)
from .rdp5 import run_rdp5
from .threeseq import run_threeseq

__all__ = ["run"]

logger = get_logger("recombination")

_TOOL_RUNNERS = {
    "rdp5": run_rdp5,
    "gard": run_gard,
    "threeseq": run_threeseq,
    "bootscan": run_bootscan,
}


def _locate_input(config: dict, root: Path) -> Path:
    """Prefer the QC-oriented FASTA, falling back to the raw dataset."""
    outroot = output_dir(config, root)
    oriented = outroot / "qc" / "hbv_oriented.fasta"
    if oriented.exists():
        return oriented
    raw = outroot / "datasets" / "hbv_genomes.fasta"
    if raw.exists():
        return raw
    raise StageError(
        "no input alignment found; expected "
        f"{oriented} or {raw} (run the QC stage first)"
    )


def _align(input_fasta: Path, outdir: Path, config: dict) -> Path:
    """Align with MAFFT when available, otherwise pass the input through."""
    aligned = outdir / "hbv_aligned.fasta"
    if not have_executable("mafft"):
        logger.warning("mafft not found; using %s unaligned", input_fasta)
        return input_fasta

    threads = int(get(config, "project.threads", 1) or 1)
    try:
        result = run_command(
            ["mafft", "--auto", "--thread", str(threads), str(input_fasta)],
            cwd=outdir,
            check=True,
        )
    except StageError as error:
        logger.warning("mafft failed (%s); using input unaligned", error)
        return input_fasta

    stdout = result.stdout or ""
    if not stdout.lstrip().startswith(">"):
        logger.warning("mafft output did not look like FASTA; using input unaligned")
        return input_fasta
    aligned.write_text(stdout, encoding="utf-8")
    logger.info("wrote alignment to %s", aligned)
    return aligned


def run(config: dict, root) -> dict[str, Path]:
    """Execute the recombination stage and return the artefact mapping."""
    root = Path(root)
    outdir = stage_dir(config, root, "recombination")
    workdir = outdir / "work"
    workdir.mkdir(parents=True, exist_ok=True)

    input_fasta = _locate_input(config, root)
    alignment = _align(input_fasta, outdir, config)

    # Inject the resolved output directory so partition_alignment can write
    # block alignments without needing the project root.
    run_config = copy.deepcopy(config)
    run_config.setdefault("recombination", {})
    run_config["recombination"]["block_aln_dir"] = str(outdir / "block_alns")

    configured_tools = list(get(config, "recombination.tools", list(_TOOL_RUNNERS)) or [])
    frames: list[pd.DataFrame] = []
    tool_counts: dict[str, int] = {}
    tools_run: list[str] = []
    for tool in configured_tools:
        runner = _TOOL_RUNNERS.get(tool)
        if runner is None:
            logger.warning("unknown recombination tool %r; skipping", tool)
            continue
        tools_run.append(tool)
        frame = runner(alignment, run_config, workdir)
        tool_counts[tool] = int(len(frame)) if frame is not None else 0
        if frame is not None and len(frame):
            frames.append(frame)
        else:
            logger.warning("tool %r produced no breakpoints", tool)

    breakpoints = reconcile_breakpoints(frames, run_config)
    if "consensus" in breakpoints.columns and len(breakpoints):
        consensus = breakpoints[breakpoints["consensus"].astype(bool)]
    else:
        consensus = breakpoints

    breakpoint_path = outdir / "breakpoints.tsv"
    pd.DataFrame(breakpoints, columns=BREAKPOINT_COLUMNS).to_csv(
        breakpoint_path, sep="\t", index=False
    )

    blocks, block_paths = partition_alignment(alignment, breakpoints, run_config)
    block_table_path = outdir / "blocks.tsv"
    blocks.to_csv(block_table_path, sep="\t", index=False)

    domain_dir = outdir / "domain_subalns"
    domain_dir.mkdir(parents=True, exist_ok=True)
    domain_paths: dict[str, Path] = {}
    for domain, fasta_text in extract_domain_subalignments(alignment, run_config).items():
        domain_path = domain_dir / f"{domain.value}.fasta"
        domain_path.write_text(fasta_text, encoding="utf-8")
        domain_paths[domain.value] = domain_path

    summary = {
        "input": str(input_fasta),
        "alignment": str(alignment),
        "tools_configured": configured_tools,
        "tools_run": tools_run,
        "breakpoints_per_tool": tool_counts,
        "n_breakpoints": int(len(breakpoints)),
        "n_consensus_breakpoints": int(len(consensus)),
        "n_blocks": int(len(blocks)),
        "min_block_len": int(get(config, "recombination.min_block_len", 0) or 0),
        "consensus_frac": float(get(config, "recombination.consensus_frac", 0.5) or 0.5),
        "block_alns": {name: str(path) for name, path in block_paths.items()},
        "domain_subalns": {name: str(path) for name, path in domain_paths.items()},
    }
    summary_path = outdir / "recombination_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    logger.info(
        "recombination: %d breakpoints (%d consensus), %d blocks",
        len(breakpoints), len(consensus), len(blocks),
    )

    artefacts: dict[str, Path] = {
        "breakpoints": breakpoint_path,
        "blocks": block_table_path,
        "summary": summary_path,
        "block_alns": outdir / "block_alns",
        "domain_subalns": domain_dir,
    }
    return artefacts
