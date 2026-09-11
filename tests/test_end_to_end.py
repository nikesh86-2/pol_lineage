"""End-to-end offline test of the full stage chain on synthetic data.

This is the integration guard: it fabricates a tiny dataset and runs every
network-free stage (QC -> recombination -> phylogeny -> selection -> epsilon ->
fitness -> structure -> atlas), asserting that the inter-stage contract holds and
that the final residue atlas is usable.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from hbvpol.synthetic import build_synthetic, run_synthetic_pipeline, synthetic_config

CONFIG = Path(__file__).resolve().parents[1] / "config" / "config.yaml"


@pytest.fixture(scope="module")
def synthetic_run(tmp_path_factory):
    outdir = tmp_path_factory.mktemp("synthetic") / "output"
    root = outdir.parent
    build_synthetic(outdir, n_genomes=8, seed=7)
    config = synthetic_config(CONFIG, outdir, n_genomes=8)
    results = run_synthetic_pipeline(config, root)
    return {"outdir": outdir, "config": config, "results": results}


def test_synthetic_pol_orf_is_stop_free(tmp_path):
    from hbvpol.config import get
    from hbvpol.domain import frame_indices, translate
    from hbvpol.io import read_fasta
    from hbvpol.qc.circularise import orient_all
    from hbvpol.qc.orfcheck import check_genome

    outdir = tmp_path / "output"
    build_synthetic(outdir, n_genomes=2, seed=11)
    records = read_fasta(outdir / "datasets" / "hbv_genomes.fasta")
    assert records, "synthetic dataset should contain genomes"

    config = synthetic_config(CONFIG, outdir, n_genomes=2)
    start = int(get(config, "reference.pol_start_nt", 1))
    end = int(get(config, "reference.pol_end_nt", 1))

    for record in records:
        oriented = orient_all([record], config)[0]
        row = check_genome(oriented, config)
        assert row["pol_stops"] == 0, f"{record.id} has internal Pol stops"

        length = len(oriented.seq)
        span = (end - start + 1) if end >= start else (length - start + 1) + end
        indices = frame_indices(start, span, length)
        protein = translate("".join(oriented.seq[i] for i in indices))
        assert "*" not in protein[:-1]


def test_qc_passes_most_genomes(synthetic_run):
    outdir = synthetic_run["outdir"]
    passed = pd.read_csv(outdir / "qc" / "hbv_qc_pass.tsv", sep="\t")
    assert len(passed) >= 6
    assert set(passed["accession"].head())  # non-empty identifiers


def test_recombination_contract(synthetic_run):
    outdir = synthetic_run["outdir"]
    blocks = pd.read_csv(outdir / "recombination" / "blocks.tsv", sep="\t")
    assert list(blocks.columns[:3]) == ["block_id", "start_nt", "end_nt"]
    assert (outdir / "recombination" / "block_alns").is_dir()


def test_phylogeny_outputs(synthetic_run):
    outdir = synthetic_run["outdir"]
    tree = outdir / "phylogeny" / "genotype_tree.treefile"
    assert tree.exists() and tree.stat().st_size > 0
    assert (outdir / "phylogeny" / "tree_summary.json").exists()


def test_selection_tables_have_expected_schema(synthetic_run):
    outdir = synthetic_run["outdir"]
    entropy = pd.read_csv(outdir / "selection" / "entropy.tsv", sep="\t")
    assert {"lineage", "pol_position", "aa_entropy", "nt_entropy"}.issubset(entropy.columns)
    assert len(entropy) > 0

    dual = pd.read_csv(outdir / "selection" / "dual_frame.tsv", sep="\t")
    assert "consequence_class" in dual.columns
    # The atlas normalises the dual-frame position column onto `pol_position`.
    assert {"pol_codon_position", "surface_position"}.intersection(dual.columns)


def test_epsilon_and_coevolution(synthetic_run):
    outdir = synthetic_run["outdir"]
    eps = outdir / "epsilon" / "epsilon_sequences.fasta"
    assert eps.exists() and eps.stat().st_size > 0
    folds = list((outdir / "epsilon" / "epsilon_folds").glob("*.dotbracket"))
    assert folds, "at least one epsilon fold should be produced"


def test_atlas_is_joined_and_round_trips(synthetic_run):
    outdir = synthetic_run["outdir"]
    atlas = pd.read_parquet(outdir / "atlas" / "residue_atlas.parquet")
    assert len(atlas) > 0
    assert {"lineage", "pol_position", "domain"}.issubset(atlas.columns)
    # The atlas is one row per (lineage, Pol position): upstream tables that are
    # finer-grained (e.g. dual-frame nucleotide rows) must be collapsed first.
    assert not atlas.duplicated(["lineage", "pol_position"]).any()
    assert (outdir / "atlas" / "ranked_targets.tsv").exists()
    assert (outdir / "atlas" / "report.html").exists()
    report = (outdir / "atlas" / "report.html").read_text(encoding="utf-8")
    assert "<html" in report.lower()


def test_pipeline_returns_paths_for_every_stage(synthetic_run):
    results = synthetic_run["results"]
    for stage in ("qc", "recombine", "tree", "select", "epsilon", "fitness", "structure", "atlas"):
        assert stage in results
        assert isinstance(results[stage], dict)


def test_cli_accepts_trailing_overrides(tmp_path):
    from hbvpol.cli import main

    outdir = tmp_path / "output"
    build_synthetic(outdir, n_genomes=4, seed=3)
    code = main(["qc", "-c", str(CONFIG), "--root", str(tmp_path),
                 f"project.output_root={outdir}"])
    assert code == 0
    assert (outdir / "qc" / "hbv_qc_pass.tsv").exists()
