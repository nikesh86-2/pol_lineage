"""Offline unit tests for the fitness package."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hbvpol.fitness.dms import (
    CRITERIA,
    DMS_COLUMNS,
    annotate_dms,
    intolerant_sites,
    load_dms,
    mechanistic_candidates,
    score_criteria,
)


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def test_load_dms_missing_file_returns_empty_schema(tmp_path, caplog):
    config = {"fitness": {"dms": {"map": "resources/absent.tsv"}}}
    frame = load_dms(config, tmp_path)
    assert frame.empty
    assert list(frame.columns) == DMS_COLUMNS


def test_load_dms_reads_tidy_tsv(tmp_path):
    path = tmp_path / "resources" / "dms.tsv"
    path.parent.mkdir(parents=True)
    pd.DataFrame({
        "wt_nt": ["A", "C"],
        "position": [2606, 2607],
        "mut_nt": ["G", "T"],
        "fitness": [0.1, 0.9],
    }).to_csv(path, sep="\t", index=False)
    config = {"fitness": {"dms": {"map": "resources/dms.tsv"}}}
    frame = load_dms(config, tmp_path)
    assert len(frame) == 2
    assert set(DMS_COLUMNS).issubset(frame.columns)


# --------------------------------------------------------------------------- #
# annotation
# --------------------------------------------------------------------------- #
def _reference() -> str:
    seq = ["A"] * 3215
    seq[2306], seq[2307], seq[2308] = "T", "A", "C"  # pol codon 1 == TAC (Y)
    return "".join(seq)


def test_annotate_dms_maps_pol_codon_and_intent():
    dms = pd.DataFrame({
        "wt_nt": ["T"],
        "position": [2307],
        "mut_nt": ["C"],
        "fitness": [0.0],
    })
    config = {
        "reference": {"length": 3215, "pol_start_nt": 2307},
        "fitness": {"reference_sequence": _reference(), "intolerance_quantile": 0.1},
    }
    annotated = annotate_dms(dms, config)
    row = annotated.iloc[0]
    assert int(row["pol_position"]) == 1
    assert row["pol_aa_wt"] == "Y"
    assert row["pol_aa_mut"] == "H"
    assert row["surface_consequence"] in {c.value for c in _consequence_classes()}
    assert bool(row["is_intolerant"]) is True
    assert row["intent"] == "loss_of_function_candidate"


def _consequence_classes():
    from hbvpol.domain import ConsequenceClass

    return list(ConsequenceClass)


def test_intolerant_sites_selects_low_fitness_tail():
    dms = pd.DataFrame({
        "pol_position": [1, 2, 3, 4],
        "fitness": [0.1, 0.2, 0.9, 1.0],
    })
    assert intolerant_sites(dms, quantile=0.5) == {1, 2}


# --------------------------------------------------------------------------- #
# criteria truth table
# --------------------------------------------------------------------------- #
def _criteria_config():
    return {
        "fitness": {
            "criteria": {
                "conservation_entropy_max": 0.5,
                "covariation_min": 0.05,
                "distinct_from_canonical_motifs": ["YMDD", "A-F", "NJG"],
            }
        }
    }


def _base_row(**overrides):
    row = {
        "pol_position": 500,
        "aa_entropy": 0.1,
        "is_intolerant": True,
        "is_hinge": True,
        "covariation_support": 0.2,
        "surface_consequence": "nonsynonymous_pol_only",
    }
    row.update(overrides)
    return row


def test_score_criteria_truth_table():
    table = pd.DataFrame([
        _base_row(),                                              # passes all
        _base_row(aa_entropy=2.0),                                # not conserved
        _base_row(pol_position=550),                              # inside YMDD
        _base_row(surface_consequence="synonymous_both"),         # explained by surface
        _base_row(is_hinge=False, is_intolerant=False),           # no interface/intolerance
    ])
    scored = score_criteria(table, _criteria_config())

    for criterion in CRITERIA:
        assert criterion in scored.columns
    assert list(scored["passes_criteria"]) == [True, False, False, False, False]
    # YMDD (549-552) is excluded from the canonical-motif-distinct criterion.
    assert bool(scored.iloc[2]["distinct_from_canonical_motifs"]) is False
    assert int(scored.iloc[0]["n_criteria_met"]) == 6


def test_mechanistic_candidates_filters_to_passers():
    table = pd.DataFrame([_base_row(), _base_row(aa_entropy=2.0)])
    candidates = mechanistic_candidates(table, _criteria_config())
    assert len(candidates) == 1
    assert bool(candidates.iloc[0]["passes_criteria"]) is True


def test_score_criteria_handles_missing_columns_gracefully():
    table = pd.DataFrame({"pol_position": [1, 2]})
    scored = score_criteria(table, _criteria_config())
    assert len(scored) == 2
    assert not scored["passes_criteria"].any()
