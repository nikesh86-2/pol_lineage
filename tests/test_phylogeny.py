"""Offline unit tests for the phylogeny package.

External tools (IQ-TREE) are absent in the test environment, so these tests
exercise the pure-Python Neighbor-Joining and parsimony fallbacks that keep the
stage functional offline.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from hbvpol.io import GenomeRecord, read_fasta, write_fasta, write_table
from hbvpol.phylogeny import pipeline as phylo_pipeline
from hbvpol.phylogeny.align import (
    alignment_length,
    as_pairs,
    read_alignment,
    slice_alignment,
    translate_alignment_for_domain,
)
from hbvpol.phylogeny.ancestral import parsimony_ancestral, reconstruct_ancestral
from hbvpol.phylogeny.trees import (
    infer_tree,
    neighbor_joining,
    newick_leaf_names,
    p_distance_matrix,
    parse_tree_summary,
)


# --------------------------------------------------------------------------- #
# p-distance + Neighbor-Joining
# --------------------------------------------------------------------------- #
def test_p_distance_matrix_known_values_and_symmetry():
    names, matrix = p_distance_matrix([
        ("a", "ACGT"),
        ("b", "ACGT"),
        ("c", "ACGA"),
        ("d", "ACGG"),
    ])
    assert names == ["a", "b", "c", "d"]
    assert matrix.shape == (4, 4)
    assert np.allclose(matrix, matrix.T)
    assert np.allclose(np.diag(matrix), 0.0)
    assert matrix[0, 1] == 0.0
    # One of four comparable sites differs.
    assert matrix[0, 2] == pytest.approx(0.25)
    assert matrix[0, 3] == pytest.approx(0.25)
    assert matrix[2, 3] == pytest.approx(0.25)


def test_p_distance_ignores_gaps_and_counts_uncomparable_as_one():
    _, matrix = p_distance_matrix([("a", "ACGT"), ("b", "----")])
    assert matrix[0, 1] == 1.0


def test_neighbor_joining_produces_valid_newick_with_expected_leaves():
    names = ["s1", "s2", "s3", "s4", "s5"]
    matrix = np.array([
        [0.0, 0.05, 0.40, 0.42, 0.44],
        [0.05, 0.0, 0.41, 0.43, 0.45],
        [0.40, 0.41, 0.0, 0.06, 0.40],
        [0.42, 0.43, 0.06, 0.0, 0.42],
        [0.44, 0.45, 0.40, 0.42, 0.0],
    ])
    newick = neighbor_joining(matrix, names)
    assert newick.endswith(";")
    assert set(newick_leaf_names(newick)) == set(names)
    # The two near-identical pairs should be sisters; the tree must be rooted on
    # an internal node, not a leaf.
    assert newick.startswith("(")


def test_neighbor_joining_degenerate_inputs():
    assert neighbor_joining(np.zeros((0, 0)), []) == "();"
    assert "solo" in neighbor_joining(np.zeros((1, 1)), ["solo"])
    two = neighbor_joining(np.array([[0.0, 0.2], [0.2, 0.0]]), ["a", "b"])
    assert set(newick_leaf_names(two)) == {"a", "b"}


# --------------------------------------------------------------------------- #
# alignment helpers
# --------------------------------------------------------------------------- #
def test_read_slice_and_length_roundtrip(tmp_path):
    path = write_fasta([
        GenomeRecord(id="s1", seq="ACGTACGT"),
        GenomeRecord(id="s2", seq="ACGTTCGT"),
    ], tmp_path / "aln.fasta")
    pairs = read_alignment(path)
    assert as_pairs(path) == pairs
    assert alignment_length(pairs) == 8
    sliced = slice_alignment(pairs, 2, 4)
    assert sliced[0] == ("s1", "CGT")
    assert as_pairs([("x", "AC"), ("y", "GT")]) == [("x", "AC"), ("y", "GT")]


def test_translate_alignment_for_domain_maps_pol_codons():
    # TP spans Pol residues 1-183; encode a poly-Ala ORF.
    sequence = "GCT" * 185
    config = {"reference": {"pol_start_nt": 1}}
    translated = translate_alignment_for_domain([("s", sequence)], "TP", config)
    assert translated[0][0] == "s"
    assert translated[0][1] == "A" * 183


# --------------------------------------------------------------------------- #
# parsimony ancestral reconstruction
# --------------------------------------------------------------------------- #
def test_parsimony_ancestral_returns_internal_sequences_of_right_length():
    tree = "((a:1,b:1):1,c:1);"
    alignment = [("a", "AAAA"), ("b", "AAAT"), ("c", "GGGG")]
    ancestral = parsimony_ancestral(tree, alignment)
    assert ancestral
    assert all(len(sequence) == 4 for sequence in ancestral.values())
    # Deterministic: the tie between {A,T} resolves to A.
    assert all(sequence == "AAAA" for sequence in ancestral.values())


def test_reconstruct_ancestral_writes_fasta(tmp_path):
    tree_path = tmp_path / "t.treefile"
    tree_path.write_text("((a:1,b:1):1,c:1);", encoding="utf-8")
    alignment = [("a", "AAAA"), ("b", "AAAT"), ("c", "GGGG")]
    out = reconstruct_ancestral(tree_path, alignment, tmp_path / "anc.fasta", {})
    records = read_fasta(out)
    assert records
    assert all(len(record.seq) == 4 for record in records)


# --------------------------------------------------------------------------- #
# infer_tree fallback + summary parsing
# --------------------------------------------------------------------------- #
def test_infer_tree_offline_writes_parseable_tree(tmp_path):
    alignment = [
        ("a", "ACGTACGTAC"),
        ("b", "ACGTACGTAC"),
        ("c", "ACGTTCGTAG"),
        ("d", "TCGTACGTAC"),
    ]
    config = {"project": {"threads": 1}, "phylogeny": {"bootstrap": 100}}
    path = infer_tree(alignment, tmp_path / "tree.treefile", config, prefix="blk")
    assert path.exists()
    assert set(newick_leaf_names(path.read_text())) == {"a", "b", "c", "d"}


def test_parse_tree_summary_newick(tmp_path):
    tree_path = tmp_path / "tree.treefile"
    tree_path.write_text("(a:0.1,b:0.1,c:0.2);", encoding="utf-8")
    summary = parse_tree_summary(tree_path)
    assert summary["format"] == "newick"
    assert summary["n_leaves"] == 3


def test_parse_tree_summary_iqtree_report(tmp_path):
    report = tmp_path / "x.iqtree"
    report.write_text(
        "IQ-TREE 2.3\n"
        "Best-fit model: GTR+F+I+G4\n"
        "Log-likelihood of the tree: -1234.5678\n"
        "Total tree length (sum of branch lengths): 1.2345\n",
        encoding="utf-8",
    )
    summary = parse_tree_summary(report)
    assert summary["format"] == "iqtree"
    assert summary["best_fit_model"] == "GTR+F+I+G4"
    assert summary["log_likelihood"] == pytest.approx(-1234.5678)


# --------------------------------------------------------------------------- #
# pipeline.run end-to-end (offline; no IQ-TREE -> Neighbor-Joining)
# --------------------------------------------------------------------------- #
def _fabricate_upstream(root: Path) -> Path:
    outroot = root / "output"
    (outroot / "recombination" / "block_alns").mkdir(parents=True)
    (outroot / "recombination" / "domain_subalns").mkdir(parents=True)

    base = {
        "s1": "ACGTACGTACGTACGTAC",
        "s2": "ACGTACGTACGTACGTAC",
        "s3": "ACGTTCGTACGTTCGTAG",
        "s4": "TCGTACGTACGTACGTAC",
        "s5": "ACGTACGTACGTACGTTT",
    }
    for block_id, cut in (("block_1", 18), ("block_2", 12)):
        write_fasta(
            [GenomeRecord(id=name, seq=seq[:cut]) for name, seq in base.items()],
            outroot / "recombination" / "block_alns" / f"{block_id}.fasta",
        )
    write_table(pd.DataFrame([
        {"block_id": 1, "start_nt": 1, "end_nt": 18, "n_seqs": 5, "n_tools_supporting": 2},
        {"block_id": 2, "start_nt": 19, "end_nt": 30, "n_seqs": 5, "n_tools_supporting": 2},
    ]), outroot / "recombination" / "blocks.tsv")
    for domain in ("TP", "RT"):
        write_fasta(
            [GenomeRecord(id=name, seq=seq[:18]) for name, seq in base.items()],
            outroot / "recombination" / "domain_subalns" / f"{domain}.fasta",
        )
    return outroot


def test_phylogeny_pipeline_run_writes_contract_artefacts(tmp_path):
    _fabricate_upstream(tmp_path)
    config = {
        "project": {"output_root": "output", "threads": 1},
        "reference": {"pol_start_nt": 1, "origin_nt": 1, "length": 3215},
        "phylogeny": {"bootstrap": 100, "ancestral_method": "parsimony"},
    }

    artefacts = phylo_pipeline.run(config, tmp_path)

    outdir = tmp_path / "output" / "phylogeny"
    for key in ("genotype_tree", "summary", "ancestral_states"):
        assert artefacts[key].exists()
    assert artefacts["block_trees"].is_dir()
    assert artefacts["domain_trees"].is_dir()

    for path in outdir.glob("block_trees/*.treefile"):
        assert set(newick_leaf_names(path.read_text())) == {"s1", "s2", "s3", "s4", "s5"}
    for path in outdir.glob("domain_trees/*.treefile"):
        assert newick_leaf_names(path.read_text())

    ancestral = read_fasta(outdir / "ancestral" / "ancestral_TP.fasta")
    assert ancestral and all(record.seq for record in ancestral)

    states = pd.read_csv(outdir / "ancestral" / "ancestral_states.tsv", sep="\t")
    assert list(states.columns) == ["domain", "node", "pol_position", "aa"]
    assert not states.empty

    summary = json.loads(artefacts["summary"].read_text())
    assert summary["tool"] == "neighbor_joining"
    assert summary["n_blocks"] == 2
    assert summary["genotype_source"] == "concatenated_blocks"
    assert summary["genotype_summary"]["n_leaves"] == 5


def test_phylogeny_pipeline_degrades_without_upstream(tmp_path):
    config = {"project": {"output_root": "output", "threads": 1}, "phylogeny": {}}

    artefacts = phylo_pipeline.run(config, tmp_path)

    outdir = tmp_path / "output" / "phylogeny"
    assert artefacts["genotype_tree"].exists()
    assert artefacts["genotype_tree"].read_text().strip() == "();"
    states = pd.read_csv(outdir / "ancestral" / "ancestral_states.tsv", sep="\t")
    assert list(states.columns) == ["domain", "node", "pol_position", "aa"]
    summary = json.loads(artefacts["summary"].read_text())
    assert summary["n_blocks"] == 0
