"""Offline unit tests for the atlas package."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hbvpol.atlas import pipeline as atlas_pipeline
from hbvpol.atlas.targets import (
    INTERACTION_COLUMNS,
    RANKED_TARGET_COLUMNS,
    SWITCH_COLUMNS,
    conformational_switch_candidates,
    interaction_network,
    rank_targets,
)


# --------------------------------------------------------------------------- #
# rank_targets
# --------------------------------------------------------------------------- #
def test_rank_targets_orders_by_weighted_score():
    atlas = pd.DataFrame({
        "lineage": ["A", "A", "A"],
        "pol_position": [1, 2, 3],
        "conservation": [0.1, 0.9, 0.5],
    })
    config = {
        "atlas": {
            "ranked_targets": {
                "weights": {
                    "conservation": 1.0,
                    "dms_intolerance": 0.0,
                    "structural_interface": 0.0,
                    "covariation": 0.0,
                },
                "top_n": 3,
            }
        }
    }
    ranked = rank_targets(atlas, config)
    assert list(ranked["pol_position"]) == [2, 3, 1]
    assert list(ranked["rank"]) == [1, 2, 3]
    assert ranked["score"].is_monotonic_decreasing


def test_rank_targets_empty_atlas_has_schema():
    ranked = rank_targets(pd.DataFrame(), {})
    assert ranked.empty
    assert list(ranked.columns) == RANKED_TARGET_COLUMNS


def test_rank_targets_respects_top_n():
    atlas = pd.DataFrame({
        "lineage": ["A"] * 5,
        "pol_position": list(range(1, 6)),
        "conservation": [0.1, 0.2, 0.3, 0.4, 0.5],
    })
    config = {"atlas": {"ranked_targets": {"weights": {"conservation": 1.0}, "top_n": 2}}}
    ranked = rank_targets(atlas, config)
    assert len(ranked) == 2
    assert list(ranked["pol_position"]) == [5, 4]


# --------------------------------------------------------------------------- #
# interaction network and switches
# --------------------------------------------------------------------------- #
def test_interaction_network_classifies_conserved_pairs():
    atlas = pd.DataFrame({
        "lineage": ["A", "B", "A", "B"],
        "pol_position": [10, 10, 20, 20],
        "covariation_partner": [30, 30, 40, 50],
        "covariation_support": [0.4, 0.3, 0.2, 0.1],
    })
    config = {"atlas": {"interaction_network": {"min_support": 0.05}}}
    network = interaction_network(atlas, config)
    assert list(network.columns) == INTERACTION_COLUMNS
    assert len(network) == 4
    pair_10 = network[network["pol_position"] == 10]
    assert set(pair_10["interaction_class"]) == {"conserved"}


def test_interaction_network_empty_without_support_columns():
    atlas = pd.DataFrame({"lineage": ["A"], "pol_position": [1]})
    network = interaction_network(atlas, {"atlas": {}})
    assert network.empty
    assert list(network.columns) == INTERACTION_COLUMNS


def test_conformational_switch_candidates_flags_spacer_hinge():
    atlas = pd.DataFrame({
        "lineage": ["A", "A"],
        "pol_position": [200, 400],  # spacer, RT
        "domain": ["spacer", "RT"],
        "is_hinge": [True, True],
        "aa_entropy": [0.8, 0.9],
    })
    config = {"atlas": {"switch_domains": ["spacer"]}}
    switches = conformational_switch_candidates(atlas, config)
    assert list(switches.columns) == SWITCH_COLUMNS
    assert list(switches["pol_position"]) == [200]


# --------------------------------------------------------------------------- #
# pipeline smoke test on a fabricated upstream tree
# --------------------------------------------------------------------------- #
def _write_upstream(root: Path) -> None:
    output = root / "output"
    for sub in ("selection", "fitness", "structure"):
        (output / sub).mkdir(parents=True, exist_ok=True)

    pd.DataFrame({
        "lineage": ["A", "A", "B", "B"],
        "position": [100, 200, 100, 200],
        "entropy": [0.2, 1.5, 0.3, 1.2],
    }).to_csv(output / "selection" / "entropy.tsv", sep="\t", index=False)

    # Documented covariation schema: pair-indexed, with position_i/position_j/score.
    pd.DataFrame({
        "lineage": ["A", "B", "A"],
        "position_i": [100, 100, 200],
        "position_j": [200, 210, 100],
        "method": ["apc", "apc", "apc"],
        "score": [0.4, 0.3, 0.4],
        "pvalue": [0.01, 0.02, 0.01],
    }).to_csv(output / "selection" / "covariation.tsv", sep="\t", index=False)

    pd.DataFrame({
        "model_id": ["apo__A__colabfold__seed1"],
        "hinge_start": [198],
        "hinge_end": [202],
        "length": [5],
    }).to_csv(output / "structure" / "hinges.tsv", sep="\t", index=False)

    pd.DataFrame({
        "wt_nt": ["A"],
        "position": [2606],
        "mut_nt": ["C"],
        "fitness": [0.05],
        "pol_position": [100],
        "surface_consequence": ["nonsynonymous_pol_only"],
        "is_intolerant": [True],
    }).to_csv(output / "fitness" / "dms_annotated.tsv", sep="\t", index=False)


def _config() -> dict:
    return {
        "project": {"output_root": "output", "name": "test"},
        "reference": {"length": 3215, "pol_start_nt": 2307},
        "structure": {"lineages": ["A", "B"]},
        "selection": {"genotype_specificity": {"min_fst": 0.25}},
        "fitness": {"criteria": {"distinct_from_canonical_motifs": ["YMDD", "A-F", "NJG"]}},
        "atlas": {
            "interactive": False,
            "switch_domains": ["spacer"],
            "ranked_targets": {
                "weights": {
                    "conservation": 0.3,
                    "dms_intolerance": 0.3,
                    "structural_interface": 0.2,
                    "covariation": 0.2,
                },
                "top_n": 50,
            },
        },
    }


def test_atlas_pipeline_run_writes_contract_artefacts(tmp_path):
    pytest.importorskip("pyarrow")
    _write_upstream(tmp_path)
    artefacts = atlas_pipeline.run(_config(), tmp_path)

    expected = {
        "atlas_parquet", "atlas_tsv", "ranked_targets",
        "conserved_interactions", "conformational_switches", "report", "summary",
    }
    assert expected.issubset(artefacts)
    for path in artefacts.values():
        assert Path(path).exists()

    atlas = pd.read_parquet(artefacts["atlas_parquet"])
    tsv = pd.read_csv(artefacts["atlas_tsv"], sep="\t")
    assert len(atlas) == len(tsv)
    assert {"lineage", "pol_position"}.issubset(atlas.columns)
    # The outer join keeps the hinge-only and DMS-only rows too.
    assert {"A", "B"}.issubset(set(atlas["lineage"]))
    assert "passes_criteria" in atlas.columns

    ranked = pd.read_csv(artefacts["ranked_targets"], sep="\t")
    assert list(ranked.columns)[:4] == ["rank", "lineage", "pol_position", "score"]
    if len(ranked) > 1:
        assert ranked["score"].is_monotonic_decreasing

    # Covariation carries a partner, so the interaction network is populated.
    network = pd.read_csv(artefacts["conserved_interactions"], sep="\t")
    assert len(network) >= 1
    assert "partner_position" in network.columns

    summary = json.loads(Path(artefacts["summary"]).read_text())
    assert summary["key"] == ["lineage", "pol_position"]
    assert summary["n_atlas_rows"] == len(atlas)


def test_atlas_pipeline_emits_valid_atlas_without_upstream(tmp_path):
    pytest.importorskip("pyarrow")
    config = {
        "project": {"output_root": "output", "name": "test"},
        "atlas": {"ranked_targets": {"top_n": 10}},
    }
    artefacts = atlas_pipeline.run(config, tmp_path)
    for path in artefacts.values():
        assert Path(path).exists()
    atlas = pd.read_parquet(artefacts["atlas_parquet"])
    assert "lineage" in atlas.columns
    assert "pol_position" in atlas.columns
    # A sparse but valid report must still be produced.
    assert Path(artefacts["report"]).read_text(encoding="utf-8").startswith("<!DOCTYPE html>")


# --------------------------------------------------------------------------- #
# contract: one row per (lineage, pol_position)
# --------------------------------------------------------------------------- #
def test_atlas_one_row_per_key_with_overlapping_hinges(tmp_path):
    pytest.importorskip("pyarrow")
    _write_upstream(tmp_path)
    hinges_path = tmp_path / "output" / "structure" / "hinges.tsv"
    hinges = pd.read_csv(hinges_path, sep="\t")
    # A second model of the *same* lineage with an overlapping hinge range.
    extra = hinges.copy()
    extra["model_id"] = "apo__A__colabfold__seed2"
    extra["hinge_start"] = 200
    extra["hinge_end"] = 204
    pd.concat([hinges, extra], ignore_index=True).to_csv(hinges_path, sep="\t", index=False)

    artefacts = atlas_pipeline.run(_config(), tmp_path)
    atlas = pd.read_parquet(artefacts["atlas_parquet"])

    assert not atlas.duplicated(["lineage", "pol_position"]).any()
    a_positions = set(atlas.loc[atlas["lineage"] == "A", "pol_position"])
    assert {198, 200, 202, 204}.issubset(a_positions)


def test_collapse_duplicates_keeps_disruptive_rna_element():
    frame = pd.DataFrame({
        "lineage": ["A", "A"],
        "pol_position": [5, 5],
        "consequence_class": ["synonymous_both", "disruptive_rna_element"],
    })
    collapsed = atlas_pipeline._collapse_duplicates(frame)
    assert len(collapsed) == 1
    assert collapsed.iloc[0]["consequence_class"] == "disruptive_rna_element"


def test_atlas_consumes_interface_residues(tmp_path):
    pytest.importorskip("pyarrow")
    _write_upstream(tmp_path)
    pd.DataFrame({
        "model_id": ["apo__A__colabfold__seed1"],
        "pol_position": [100],
        "min_distance": [3.0],
        "is_interface": [True],
    }).to_csv(
        tmp_path / "output" / "structure" / "interface_residues.tsv", sep="\t", index=False
    )

    artefacts = atlas_pipeline.run(_config(), tmp_path)
    atlas = pd.read_parquet(artefacts["atlas_parquet"])
    row = atlas[(atlas["lineage"] == "A") & (atlas["pol_position"] == 100)]
    assert len(row) == 1
    assert bool(row.iloc[0]["is_interface"]) is True
