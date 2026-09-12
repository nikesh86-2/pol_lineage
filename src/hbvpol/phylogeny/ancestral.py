"""Ancestral sequence reconstruction.

Two routes are provided:

* **Maximum likelihood** via IQ-TREE's ancestral reconstruction
  (``-as`` / ``--ancestral``) when IQ-TREE is on ``PATH`` and the configured
  ``phylogeny.ancestral_method`` asks for it (``auto``/``ml``).
* **Parsimony** (Fitch/Sankoff) implemented in pure Python, which works fully
  offline.  This is the default fallback and the route used by the tests.

The parsimony implementation is deterministic: when several states are equally
parsimonious the alphabetically-first unambiguous nucleotide is chosen, so
repeated runs give byte-identical output.
"""

from __future__ import annotations

from pathlib import Path

from ..config import get
from ..io import GenomeRecord, write_fasta
from ..pipeline import effective_threads, get_logger, run_command
from .align import as_pairs
from .trees import _iqtree_executable

__all__ = ["parsimony_ancestral", "reconstruct_ancestral"]

logger = get_logger("phylogeny.ancestral")

_PREFERRED = ("A", "C", "G", "T")


def _choose_state(states: set[str]) -> str:
    for nucleotide in _PREFERRED:
        if nucleotide in states:
            return nucleotide
    return min(states) if states else "-"


def parsimony_ancestral(tree_newick: str, alignment) -> dict[str, str]:
    """Reconstruct ancestral sequences by Fitch parsimony.

    Walks the tree for every alignment column: a first (post-order) pass
    propagates the set of possible states, a second (pre-order) pass fixes one
    realisation.  Internal nodes are named ``Node1``, ``Node2``… in traversal
    order when the tree does not already name them.

    Returns a mapping ``node_name -> ancestral_sequence`` for the internal
    (ancestral) nodes only.
    """
    from io import StringIO

    from Bio import Phylo

    pairs = as_pairs(alignment)
    sequences = {name: seq.upper() for name, seq in pairs}
    width = max((len(seq) for seq in sequences.values()), default=0)

    tree = Phylo.read(StringIO(tree_newick), "newick")
    internal = [clade for clade in tree.find_clades() if not clade.is_terminal()]
    for index, clade in enumerate(internal, start=1):
        if not clade.name:
            clade.name = f"Node{index}"

    if width == 0:
        return {clade.name: "" for clade in internal}

    parents = {child: parent for parent in tree.find_clades() for child in parent.clades}
    result: dict[str, list[str]] = {clade.name: [] for clade in internal}

    for column in range(width):
        states: dict[object, set[str]] = {}
        for clade in tree.find_clades(order="postorder"):
            if clade.is_terminal():
                sequence = sequences.get(clade.name, "")
                states[clade] = {sequence[column] if column < len(sequence) else "-"}
            else:
                child_sets = [states[child] for child in clade.clades]
                if not child_sets:
                    states[clade] = {"-"}
                    continue
                intersection = set.intersection(*child_sets)
                states[clade] = intersection if intersection else set.union(*child_sets)

        assigned: dict[object, str] = {}
        root = tree.root
        assigned[root] = _choose_state(states[root])
        for clade in tree.find_clades(order="preorder"):
            if clade is root:
                continue
            parent_state = assigned[parents[clade]]
            possible = states[clade]
            assigned[clade] = parent_state if parent_state in possible else _choose_state(possible)

        for clade in internal:
            result[clade.name].append(assigned[clade])

    return {name: "".join(chars) for name, chars in result.items()}


def _iqtree_ancestral(tree_path, alignment, config, out_path: Path) -> dict[str, str] | None:
    """Best-effort IQ-TREE ancestral reconstruction; returns ``None`` on failure.

    IQ-TREE writes a ``.state`` file when invoked with ``-as``; its format is
    version-dependent, so any parse problem simply triggers the parsimony
    fallback in :func:`reconstruct_ancestral`.
    """
    executable = _iqtree_executable()
    if executable is None:
        return None
    pairs = as_pairs(alignment)
    workdir = out_path.parent
    workdir.mkdir(parents=True, exist_ok=True)
    # Absolute paths: run_command sets cwd to workdir, so relative paths would be
    # re-rooted under it and IQ-TREE could not open its log/state files.
    alignment_path = (workdir / f"{out_path.stem}.fasta").resolve()
    write_fasta([GenomeRecord(id=name, seq=seq) for name, seq in pairs], alignment_path)
    threads = effective_threads(config)
    pre = (workdir / out_path.stem).resolve()
    tree_path = Path(tree_path).resolve()
    run_command(
        [
            executable,
            "-s", str(alignment_path),
            "-te", str(tree_path),
            "-as",
            "-m", "MFP",
            "-T", str(max(1, threads)),
            "-pre", str(pre),
            "-redo",
        ],
        cwd=workdir,
        log_path=workdir / f"{out_path.stem}.ancestral.log",
        check=True,
    )
    state_file = Path(str(pre) + ".state")
    if not state_file.exists():
        return None
    return _parse_state_file(state_file)


def _parse_state_file(state_file: Path) -> dict[str, str] | None:
    """Parse an IQ-TREE ``.state`` file into ``node -> sequence``.

    The file contains one ``Site`` block per alignment column with node rows
    ``Node  <name>  <state>``.  This parser is intentionally permissive; it
    returns ``None`` if the expected structure is not found.
    """
    sequences: dict[str, list[str]] = {}
    found = False
    current_column: dict[str, str] = {}
    for raw_line in state_file.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if line.startswith("Site"):
            if current_column:
                for node, state in current_column.items():
                    sequences.setdefault(node, []).append(state)
                current_column = {}
            found = True
            continue
        parts = line.split()
        if len(parts) >= 3 and parts[0].lower() == "node":
            current_column[parts[1]] = parts[2]
    if current_column:
        for node, state in current_column.items():
            sequences.setdefault(node, []).append(state)
    if not found or not sequences:
        return None
    return {name: "".join(chars) for name, chars in sequences.items()}


def reconstruct_ancestral(tree_path, alignment, out_fasta, config) -> Path:
    """Reconstruct ancestral sequences and write them to ``out_fasta``.

    Uses IQ-TREE when the configured method permits and the tool is available,
    otherwise Fitch parsimony.  Always returns the path written.
    """
    out = Path(out_fasta)
    out.parent.mkdir(parents=True, exist_ok=True)
    method = str(get(config, "phylogeny.ancestral_method", "auto") or "auto").lower()

    if method in {"auto", "ml", "iqtree", "maximum_likelihood", "maximum-likelihood"}:
        try:
            inferred = _iqtree_ancestral(tree_path, alignment, config, out)
        except Exception as error:
            logger.warning("IQ-TREE ancestral reconstruction failed (%s); using parsimony", error)
            inferred = None
        if inferred:
            records = [GenomeRecord(id=name, seq=seq) for name, seq in inferred.items()]
            write_fasta(records, out)
            return out

    tree_newick = Path(tree_path).read_text(encoding="utf-8")
    ancestral = parsimony_ancestral(tree_newick, alignment)
    records = [GenomeRecord(id=name, seq=seq) for name, seq in ancestral.items()]
    write_fasta(records, out)
    return out
