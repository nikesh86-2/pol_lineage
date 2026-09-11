"""Phylogeny stage pipeline (Stage 2c).

Reads the recombination stage's non-recombinant blocks and Pol-domain
sub-alignments, infers trees, reconstructs ancestral sequences and writes:

    <outroot>/phylogeny/genotype_tree.treefile
    <outroot>/phylogeny/block_trees/block_<id>.treefile
    <outroot>/phylogeny/domain_trees/<domain>.treefile
    <outroot>/phylogeny/ancestral/ancestral_<domain>.fasta
    <outroot>/phylogeny/ancestral/ancestral_states.tsv
    <outroot>/phylogeny/tree_summary.json

Recombination-aware genotype tree
----------------------------------
Recombination breaks the assumption of a single tree for the whole genome, so
the stage infers a tree for every non-recombinant block independently and then
builds the *genotype tree* by concatenating the block alignments into one
partitioned supermatrix (each block a partition) and inferring a single tree
over it.  When the blocks cannot be concatenated (mismatched taxa), the tree
from the largest block by nucleotide length is used instead.  Per-block trees
are always retained so block-level support can be compared afterwards.  This is
the "infer per-block, then summarise" approach; the concatenated tree is the
partition-aware consensus.

Every external tool is optional.  IQ-TREE is used when available and the pure
Python Neighbor-Joining fallback otherwise (see :mod:`hbvpol.phylogeny.trees`),
and ancestral reconstruction falls back to Fitch parsimony, so the stage
completes offline.  Missing upstream inputs are logged and produce
empty-but-correctly-schemed outputs rather than raising.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ..config import get
from ..domain import DEFAULT_DOMAIN_SPANS, PolDomain, translate
from ..io import read_fasta, read_table, write_table
from ..pipeline import get_logger, output_dir, stage_dir
from .align import as_pairs, read_alignment
from .ancestral import reconstruct_ancestral
from .trees import _iqtree_executable, infer_tree, parse_tree_summary

__all__ = ["run", "ANCESTRAL_STATE_COLUMNS"]

logger = get_logger("phylogeny")

ANCESTRAL_STATE_COLUMNS = ["domain", "node", "pol_position", "aa"]

_FALLBACK_ALIGNMENTS = (
    ("qc", "hbv_oriented.fasta"),
    ("datasets", "hbv_genomes.fasta"),
    ("datasets", "deephep_alignment.fasta"),
)


def _collect_blocks(recomb_dir: Path) -> list[tuple[str, Path, int]]:
    """Return ``(label, path, length_nt)`` for every available block alignment."""
    blocks: list[tuple[str, Path, int]] = []
    block_dir = recomb_dir / "block_alns"
    blocks_tsv = recomb_dir / "blocks.tsv"

    if blocks_tsv.exists():
        try:
            table = read_table(blocks_tsv)
        except Exception as error:  # pragma: no cover - defensive
            logger.warning("could not read %s (%s)", blocks_tsv, error)
            table = pd.DataFrame()
        for _, row in table.iterrows():
            block_id = row.get("block_id")
            if pd.isna(block_id):
                continue
            label = f"block_{int(block_id)}"
            path = block_dir / f"{label}.fasta"
            if not path.exists():
                logger.warning("block alignment missing for %s: %s", label, path)
                continue
            try:
                length = int(row.get("end_nt")) - int(row.get("start_nt")) + 1
            except (TypeError, ValueError):
                length = 0
            blocks.append((label, path, length))
        if blocks:
            return blocks

    if block_dir.is_dir():
        for path in sorted(block_dir.glob("block_*.fasta")):
            label = path.stem
            try:
                length = max(len(record.seq) for record in read_fasta(path))
            except Exception:
                length = 0
            blocks.append((label, path, length))
    return blocks


def _concatenate_blocks(blocks: list[tuple[str, Path, int]]) -> list[tuple[str, str]] | None:
    """Column-wise concatenate block alignments that share the same taxa."""
    if len(blocks) < 2:
        return None
    per_block: list[list[tuple[str, str]]] = []
    for _, path, _ in blocks:
        pairs = as_pairs(read_alignment(path))
        if not pairs:
            return None
        per_block.append(pairs)

    names = [name for name, _ in per_block[0]]
    for pairs in per_block[1:]:
        if [name for name, _ in pairs] != names:
            return None

    concatenated: dict[str, list[str]] = {name: [] for name in names}
    for pairs in per_block:
        for name, seq in pairs:
            concatenated[name].append(seq)
    return [(name, "".join(parts)) for name, parts in concatenated.items()]


def _fallback_alignment(outroot: Path) -> Path | None:
    for stage, filename in _FALLBACK_ALIGNMENTS:
        candidate = outroot / stage / filename
        if candidate.exists():
            return candidate
    return None


def _domain_span(domain: PolDomain):
    for span in DEFAULT_DOMAIN_SPANS:
        if span.domain == domain:
            return span
    return None


def _write_empty_ancestral_states(path: Path) -> Path:
    return write_table(pd.DataFrame(columns=ANCESTRAL_STATE_COLUMNS), path)


def run(config: dict, root) -> dict[str, Path]:
    """Execute the phylogeny stage and return the artefact mapping."""
    root = Path(root)
    outdir = stage_dir(config, root, "phylogeny")
    outroot = output_dir(config, root)
    recomb_dir = outroot / "recombination"

    block_trees_dir = outdir / "block_trees"
    domain_trees_dir = outdir / "domain_trees"
    ancestral_dir = outdir / "ancestral"
    workdir = outdir / "work"
    for directory in (block_trees_dir, domain_trees_dir, ancestral_dir, workdir):
        directory.mkdir(parents=True, exist_ok=True)

    blocks = _collect_blocks(recomb_dir)
    if not blocks:
        logger.warning(
            "no block alignments found under %s; falling back to a whole-genome alignment",
            recomb_dir,
        )

    tool = _iqtree_executable() or "neighbor_joining"

    # --- per-block trees ---------------------------------------------------
    block_trees: dict[str, Path] = {}
    block_summaries: dict[str, dict] = {}
    for label, path, _ in blocks:
        tree_path = block_trees_dir / f"{label}.treefile"
        try:
            infer_tree(path, tree_path, config, prefix=label)
        except Exception as error:
            logger.warning("tree inference failed for %s (%s); writing placeholder", label, error)
            tree_path.write_text("();\n", encoding="utf-8")
        block_trees[label] = tree_path
        block_summaries[label] = parse_tree_summary(tree_path)

    # --- genotype tree -----------------------------------------------------
    genotype_tree = outdir / "genotype_tree.treefile"
    genotype_source = "none"
    concatenated = _concatenate_blocks(blocks)
    if concatenated:
        genotype_source = "concatenated_blocks"
        genotype_alignment = concatenated
    elif blocks:
        largest = max(blocks, key=lambda item: item[2])
        genotype_source = largest[0]
        genotype_alignment = largest[1]
    else:
        fallback = _fallback_alignment(outroot)
        if fallback is not None:
            genotype_source = str(fallback)
            genotype_alignment = fallback
        else:
            genotype_alignment = None

    if genotype_alignment is not None:
        try:
            infer_tree(genotype_alignment, genotype_tree, config, prefix="genotype")
        except Exception as error:
            logger.warning("genotype tree inference failed (%s); writing placeholder", error)
            genotype_tree.write_text("();\n", encoding="utf-8")
    else:
        logger.warning("no alignment available for a genotype tree; writing an empty tree")
        genotype_tree.write_text("();\n", encoding="utf-8")
    genotype_summary = parse_tree_summary(genotype_tree)

    # --- domain trees + ancestral reconstruction ---------------------------
    domain_files: dict[PolDomain, Path] = {}
    subaln_dir = recomb_dir / "domain_subalns"
    if subaln_dir.is_dir():
        for path in sorted(subaln_dir.glob("*.fasta")):
            try:
                domain = PolDomain.parse(path.stem)
            except ValueError:
                logger.warning("unrecognised domain sub-alignment %s; skipping", path)
                continue
            domain_files[domain] = path

    if not domain_files:
        fallback = _fallback_alignment(outroot)
        if fallback is not None:
            from ..recombination.partition import extract_domain_subalignments

            try:
                derived = extract_domain_subalignments(fallback, config)
            except Exception as error:  # pragma: no cover - defensive
                logger.warning("could not derive domain sub-alignments (%s)", error)
                derived = {}
            for domain, text in derived.items():
                path = workdir / f"{domain.value}.fasta"
                path.write_text(text, encoding="utf-8")
                domain_files[domain] = path

    domain_trees: dict[str, Path] = {}
    ancestral_fastas: dict[str, Path] = {}
    ancestral_rows: list[dict] = []

    for domain, alignment_path in sorted(domain_files.items(), key=lambda item: item[0].value):
        tree_path = domain_trees_dir / f"{domain.value}.treefile"
        try:
            infer_tree(alignment_path, tree_path, config, prefix=domain.value)
        except Exception as error:
            logger.warning("domain tree inference failed for %s (%s)", domain.value, error)
            tree_path.write_text("();\n", encoding="utf-8")
        domain_trees[domain.value] = tree_path

        ancestral_path = ancestral_dir / f"ancestral_{domain.value}.fasta"
        try:
            reconstruct_ancestral(tree_path, alignment_path, ancestral_path, config)
        except Exception as error:
            logger.warning("ancestral reconstruction failed for %s (%s)", domain.value, error)
            ancestral_path.write_text("", encoding="utf-8")
        ancestral_fastas[domain.value] = ancestral_path

        span = _domain_span(domain)
        span_start = span.start if span is not None else 1
        try:
            ancestral_records = read_fasta(ancestral_path)
        except Exception:
            ancestral_records = []
        for record in ancestral_records:
            protein = translate(record.seq, frame=0)
            for index, aa in enumerate(protein):
                ancestral_rows.append({
                    "domain": domain.value,
                    "node": record.id,
                    "pol_position": span_start + index,
                    "aa": aa,
                })

    ancestral_states_path = write_table(
        pd.DataFrame(ancestral_rows, columns=ANCESTRAL_STATE_COLUMNS),
        outdir / "ancestral" / "ancestral_states.tsv",
    )

    summary = {
        "tool": tool,
        "root": get(config, "phylogeny.root", None),
        "bootstrap": int(get(config, "phylogeny.bootstrap", 0) or 0),
        "ancestral_method": get(config, "phylogeny.ancestral_method", "auto"),
        "n_blocks": len(blocks),
        "block_trees": {label: str(path) for label, path in block_trees.items()},
        "block_summaries": block_summaries,
        "genotype_tree": str(genotype_tree),
        "genotype_source": genotype_source,
        "genotype_summary": genotype_summary,
        "domain_trees": {name: str(path) for name, path in domain_trees.items()},
        "ancestral": {name: str(path) for name, path in ancestral_fastas.items()},
        "n_ancestral_states": len(ancestral_rows),
    }
    summary_path = outdir / "tree_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    logger.info(
        "phylogeny: %d block trees, %d domain trees, genotype source=%s",
        len(block_trees), len(domain_trees), genotype_source,
    )

    return {
        "genotype_tree": genotype_tree,
        "block_trees": block_trees_dir,
        "domain_trees": domain_trees_dir,
        "ancestral_states": ancestral_states_path,
        "summary": summary_path,
    }
