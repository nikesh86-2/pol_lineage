"""Tree inference with IQ-TREE and an offline Neighbor-Joining fallback.

The public entry point is :func:`infer_tree`.  It runs IQ-TREE (``iqtree2`` or
``iqtree``) when one is on ``PATH`` and otherwise falls back to a pure-Python
Neighbor-Joining (NJ) tree over a p-distance matrix.  The fallback makes the
phylogeny stage fully testable offline; it is *not* a substitute for a proper
model-based ML tree, and :func:`infer_tree` marks which route was taken via
:func:`parse_tree_summary`.

All heavy imports (Biopython) happen inside functions so this module imports
cleanly with no external tools or optional libraries present.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Mapping

import numpy as np

from ..config import get
from ..io import GenomeRecord, write_fasta
from ..pipeline import StageError, effective_threads, get_logger, have_executable, run_command, strict_tools
from .align import as_pairs

__all__ = [
    "p_distance_matrix",
    "neighbor_joining",
    "infer_tree",
    "parse_tree_summary",
    "apply_rooting",
    "newick_leaf_names",
]

logger = get_logger("phylogeny.trees")

_ACGT = frozenset("ACGT")


# --------------------------------------------------------------------------- #
# Pure algorithms
# --------------------------------------------------------------------------- #
def p_distance_matrix(alignment) -> tuple[list[str], np.ndarray]:
    """Compute the pairwise p-distance matrix of an alignment.

    Only columns where *both* sequences carry an unambiguous A/C/G/T are
    compared.  Pairs with no comparable columns are assigned a distance of 1.0
    (maximal separation) so that Neighbor-Joining always has a valid matrix.

    Returns ``(names, matrix)`` where ``matrix`` is symmetric with a zero
    diagonal.
    """
    pairs = as_pairs(alignment)
    names = [name for name, _ in pairs]
    seqs = [seq.upper() for _, seq in pairs]
    n = len(seqs)
    if n == 0:
        return names, np.zeros((0, 0), dtype=float)

    width = max(len(seq) for seq in seqs)
    padded = [seq.ljust(width, "-") for seq in seqs]
    matrix = np.zeros((n, n), dtype=float)
    for i in range(n):
        a = padded[i]
        for j in range(i + 1, n):
            b = padded[j]
            comparable = 0
            differences = 0
            for x, y in zip(a, b):
                if x in _ACGT and y in _ACGT:
                    comparable += 1
                    if x != y:
                        differences += 1
            distance = differences / comparable if comparable else 1.0
            matrix[i, j] = matrix[j, i] = distance
    return names, matrix


def _render_newick(node) -> str:
    if isinstance(node, str):
        return node
    left, left_len, right, right_len = node
    return f"({_render_newick(left)}:{left_len:.6f},{_render_newick(right)}:{right_len:.6f})"


def _sanitize_name(name: str) -> str:
    cleaned = re.sub(r"[\s,:;()\[\]']", "_", str(name))
    return cleaned or "leaf"


def neighbor_joining(distance_matrix, names) -> str:
    """Build a Neighbor-Joining tree and return it as a Newick string.

    Implements the classical Saitou & Nei (1987) algorithm with non-negative
    branch lengths.  ``distance_matrix`` is a square, symmetric array and
    ``names`` the leaf labels.  This is a pure function: it has no side effects
    and is unit-tested directly.
    """
    matrix = np.asarray(distance_matrix, dtype=float)
    labels = [_sanitize_name(name) for name in names]
    n = len(labels)
    if matrix.shape != (n, n):
        raise ValueError("distance_matrix shape does not match the number of names")
    if n == 0:
        return "();"
    if n == 1:
        return f"({labels[0]}:0.000000);"
    if n == 2:
        d = float(matrix[0, 1]) / 2.0
        return f"({labels[0]}:{d:.6f},{labels[1]}:{d:.6f});"

    dist: dict[tuple[int, int], float] = {}
    for i in range(n):
        for j in range(n):
            dist[(i, j)] = float(matrix[i, j])

    active = list(range(n))
    trees: dict[int, object] = {i: labels[i] for i in range(n)}
    next_id = n

    while len(active) > 2:
        r = len(active)
        totals = {i: sum(dist[(i, k)] for k in active) for i in active}
        best = None  # (Q, i, j)
        for x in range(r):
            i = active[x]
            for y in range(x + 1, r):
                j = active[y]
                q = (r - 2) * dist[(i, j)] - totals[i] - totals[j]
                if best is None or q < best[0]:
                    best = (q, i, j)
        _, i, j = best
        d_ij = dist[(i, j)]
        delta_i = 0.5 * d_ij + (totals[i] - totals[j]) / (2 * (r - 2))
        delta_j = d_ij - delta_i

        new = next_id
        next_id += 1
        trees[new] = (trees[i], max(delta_i, 0.0), trees[j], max(delta_j, 0.0))

        for k in active:
            if k in (i, j):
                continue
            d_new = 0.5 * (dist[(i, k)] + dist[(j, k)] - d_ij)
            dist[(new, k)] = d_new
            dist[(k, new)] = d_new
        dist[(new, new)] = 0.0
        active = [k for k in active if k not in (i, j)]
        active.append(new)

    if len(active) == 2:
        a, b = active
        half = dist[(a, b)] / 2.0
        root = (trees[a], half, trees[b], half)
    else:  # pragma: no cover - only for degenerate n==1 handled above
        root = trees[active[0]]
    return _render_newick(root) + ";"


def newick_leaf_names(newick: str) -> list[str]:
    """Return the leaf labels of a Newick string (parsed with Biopython)."""
    from io import StringIO

    from Bio import Phylo

    tree = Phylo.read(StringIO(newick), "newick")
    return [clade.name for clade in tree.get_terminals()]


# --------------------------------------------------------------------------- #
# Rooting
# --------------------------------------------------------------------------- #
def apply_rooting(newick: str, config: Mapping[str, object]) -> str:
    """Optionally re-root a Newick string per ``phylogeny.root``.

    ``phylogeny.root`` may be an outgroup leaf name or the literal ``midpoint``.
    Any failure (unknown outgroup, malformed tree) leaves the tree unchanged.
    """
    root_option = get(config, "phylogeny.root", None)
    if not root_option:
        return newick
    try:
        from io import StringIO

        from Bio import Phylo

        tree = Phylo.read(StringIO(newick), "newick")
        option = str(root_option).strip()
        if option.lower() == "midpoint":
            tree.root_at_midpoint()
        else:
            tree.root_with_outgroup(option)
        handle = StringIO()
        Phylo.write(tree, handle, "newick")
        rooted = handle.getvalue().strip()
        return rooted if rooted else newick
    except Exception as error:  # pragma: no cover - depends on tree shape
        logger.warning("rooting with %r failed (%s); leaving tree unrooted", root_option, error)
        return newick


# --------------------------------------------------------------------------- #
# IQ-TREE wrapper + fallback
# --------------------------------------------------------------------------- #
def _iqtree_executable() -> str | None:
    for candidate in ("iqtree2", "iqtree"):
        if have_executable(candidate):
            return candidate
    return None


def _ensure_alignment_file(alignment, fallback_path: Path) -> Path:
    if isinstance(alignment, (str, Path)) and Path(alignment).exists():
        return Path(alignment)
    pairs = as_pairs(alignment)
    records = [GenomeRecord(id=name, seq=seq) for name, seq in pairs]
    return write_fasta(records, fallback_path)


def infer_tree(alignment, out_path, config, prefix: str = "") -> Path:
    """Infer a tree for ``alignment`` and write Newick to ``out_path``.

    IQ-TREE is used when available with model finder ``-m MFP``, ultrafast
    bootstrap ``-B`` (from ``phylogeny.bootstrap``) and ``project.threads``
    threads.  On any absence or failure the function falls back to a
    Neighbor-Joining tree over the alignment's p-distance matrix, so the
    returned path always exists and contains a parseable Newick tree.
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    stem = prefix or out.stem or "tree"

    executable = _iqtree_executable()
    if executable:
        try:
            # Absolute paths are required: run_command sets cwd to out.parent, so
            # a relative `-s`/`-pre` would be re-rooted under it and IQ-TREE
            # cannot open its own log ("Could not open .../block_1.log").
            alignment_path = Path(
                _ensure_alignment_file(alignment, out.parent / f"{stem}.fasta")
            ).resolve()
            bootstrap = int(get(config, "phylogeny.bootstrap", 1000) or 1000)
            threads = effective_threads(config)
            # Pin the seed so UFBoot resampling and model search are reproducible.
            seed = int(get(config, "project.seed", 1) or 1)
            pre = (out.parent / stem).resolve()
            command = [
                executable,
                "-s", str(alignment_path),
                "-m", "MFP",
                "-B", str(bootstrap),
                "-T", str(max(1, threads)),
                "--seed", str(seed),
                "-pre", str(pre),
                "-redo",
            ]
            run_command(command, cwd=out.parent, log_path=out.parent / f"{stem}.iqtree.log", check=True)
            produced = Path(str(pre) + ".treefile")
            if produced.exists():
                newick = apply_rooting(produced.read_text(encoding="utf-8"), config)
                out.write_text(newick, encoding="utf-8")
                return out
            logger.warning("IQ-TREE produced no treefile for %s; using Neighbor-Joining", stem)
        except Exception as error:
            logger.warning("IQ-TREE failed for %s (%s); using Neighbor-Joining", stem, error)

    # A publication run must not silently substitute a different method for a
    # present-but-failing IQ-TREE.
    if executable and strict_tools(config):
        raise StageError(
            f"{stem}: IQ-TREE was available but produced no usable tree; refusing the "
            "Neighbor-Joining fallback (project.strict_tools)"
        )

    # Guard the pure-Python fallback: p-distances are O(N^2) and Neighbor-Joining
    # is O(N^3), so a large taxon set never finishes.  Bail out loudly instead of
    # spinning for days; callers cap taxa via `phylogeny.max_taxa`.
    n_taxa = len(as_pairs(alignment))
    max_nj = get(config, "phylogeny.max_nj_taxa", 1000)
    max_nj = int(max_nj) if max_nj not in (None, "", 0) else None
    if max_nj is not None and n_taxa > max_nj:
        raise StageError(
            f"refusing Neighbor-Joining on {n_taxa} taxa (phylogeny.max_nj_taxa={max_nj}); "
            f"install IQ-TREE or lower phylogeny.max_taxa"
        )

    names, matrix = p_distance_matrix(alignment)
    newick = neighbor_joining(matrix, names)
    newick = apply_rooting(newick, config)
    out.write_text(newick, encoding="utf-8")
    return out


# --------------------------------------------------------------------------- #
# Summary parsing
# --------------------------------------------------------------------------- #
_IQTREE_PATTERNS = {
    "best_fit_model": [r"Best-fit model:\s*(\S+)", r"Model of substitution:\s*(\S+)"],
    "log_likelihood": [r"Log-likelihood of the tree:\s*(-?[\d.eE+-]+)"],
    "tree_length": [r"Total tree length \(sum of branch lengths\):\s*([\d.eE+-]+)"],
    "n_taxa": [r"Input sequences:\s*(\d+)", r"Number of taxa:\s*(\d+)"],
    "n_sites": [r"Input alignment:\s*(\d+)", r"Number of sites:\s*(\d+)"],
    "bootstrap": [r"Ultrafast bootstrap \(UFBoot\):\s*(\d+)"],
}


def parse_tree_summary(path) -> dict:
    """Summarise an IQ-TREE report (``*.iqtree``) or a Newick treefile.

    Returns a plain ``dict``.  When the file looks like an IQ-TREE report the
    recognised numeric/string fields are extracted; when it looks like Newick
    the leaf count and string length are reported.  A permissive implementation
    is intentional: the function is used for logging/JSON summaries, not for
    any downstream numeric decision.
    """
    summary: dict[str, object] = {"path": str(path)}
    file_path = Path(path)
    if not file_path.exists():
        return summary
    text = file_path.read_text(encoding="utf-8", errors="replace")

    looks_like_newick = text.lstrip().startswith("(") or ("(" in text and ";" in text and "IQ-TREE" not in text)
    if looks_like_newick:
        single_line = text.strip()
        try:
            summary["n_leaves"] = len(newick_leaf_names(single_line))
            summary["format"] = "newick"
        except Exception:
            # A cheap fallback leaf count avoids failing on exotic trees.
            summary["n_leaves"] = len(re.findall(r"[,\(]([^,\(\):;]+):", single_line))
            summary["format"] = "newick-unparsed"
        summary["newick_length"] = len(single_line)
        return summary

    summary["format"] = "iqtree" if "IQ-TREE" in text else "unknown"
    for key, patterns in _IQTREE_PATTERNS.items():
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                raw = match.group(1)
                try:
                    summary[key] = int(raw)
                except ValueError:
                    try:
                        summary[key] = float(raw)
                    except ValueError:
                        summary[key] = raw
                break
    if "Log-likelihood" in text and "log_likelihood" not in summary:
        summary["log_likelihood"] = None
    return summary
