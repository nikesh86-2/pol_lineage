"""Offline unit tests for the selection package.

HyPhy is absent in the test environment, so the pure-Python entropy,
dual-frame, covariation and genotype analyses are tested directly and the HyPhy
delegation is checked to fail soft with a schema-correct empty frame.
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

from hbvpol.io import GenomeRecord, write_fasta, write_table
from hbvpol.selection import pipeline as selection_pipeline
from hbvpol.selection.covariation import (
    COVARIATION_COLUMNS,
    EPISTASIS_COLUMNS,
    apc_correct,
    covarying_pairs,
    epistasis_pairs,
    mutual_information,
)
from hbvpol.selection.dualframe import (
    DUAL_FRAME_CLASSES,
    DUAL_FRAME_COLUMNS,
    classify_substitution,
    dual_frame_table,
)
from hbvpol.selection.entropy import ENTROPY_COLUMNS, per_position_entropy, shannon_entropy
from hbvpol.selection.genotype import GENOTYPE_COLUMNS, genotype_specificity
from hbvpol.selection.pipeline import SELECTION_SITE_COLUMNS, run_hyphy_method
from hbvpol.selection.resistance import (
    RESISTANCE_COLUMNS,
    canonical_motif_at,
    load_catalogue,
    parse_rt_mutation,
    resistance_table,
)


# --------------------------------------------------------------------------- #
# entropy
# --------------------------------------------------------------------------- #
def test_shannon_entropy_known_values():
    assert shannon_entropy({"A": 2, "B": 2}) == pytest.approx(1.0)
    assert shannon_entropy([1, 1, 1, 1]) == pytest.approx(2.0)
    assert shannon_entropy({"A": 5}) == 0.0
    assert shannon_entropy([]) == 0.0
    assert shannon_entropy([]) == 0.0


def test_per_position_entropy_amino_acid_value():
    alignment = [
        ("s1", "GCTGCT"),
        ("s2", "GCTGCT"),
        ("s3", "GATGAT"),
        ("s4", "GATGAT"),
    ]
    config = {"reference": {"pol_start_nt": 1}, "selection": {"window": 1}}
    frame = per_position_entropy(alignment, config)
    assert list(frame.columns) == [column for column in ENTROPY_COLUMNS if column != "lineage"]
    # Ala/Ala/Asp/Asp -> two equally frequent amino acids -> 1 bit.
    assert frame.iloc[0]["aa_entropy"] == pytest.approx(1.0)
    assert frame.iloc[0]["gap_frac"] == 0.0
    assert frame.iloc[0]["n_seqs"] == 4
    assert frame.iloc[0]["domain"] == "TP"
    assert frame.iloc[0]["pol_position"] == 1


def test_per_position_entropy_counts_gaps():
    alignment = [("s1", "GCT"), ("s2", "---")]
    config = {"reference": {"pol_start_nt": 1}, "selection": {"window": 1}}
    frame = per_position_entropy(alignment, config)
    assert frame.iloc[0]["gap_frac"] == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# dual-frame classification
# --------------------------------------------------------------------------- #
# Reference and a variant engineered so that single substitutions at columns
# 0, 3, 4, 5 and 10 yield, respectively, all five consequence classes with the
# surface frame offset by +1 and an RNA element covering reference position 11.
_DUAL_REFERENCE = "TTATGTAGCAGC"   # cols 0..11
_DUAL_VARIANT = "CTACACAGCAAC"     # differs at cols 0, 3, 4, 5, 10


def _dual_alignment():
    return [
        GenomeRecord(id=f"ref{i}", seq=_DUAL_REFERENCE) for i in range(3)
    ] + [GenomeRecord(id="var", seq=_DUAL_VARIANT)]


def _dual_config():
    return {
        "reference": {
            "pol_start_nt": 1,
            "surface_frame_offset": 1,
            "origin_nt": 1,
            "rna_elements": [{"start": 11, "end": 11, "name": "test_element"}],
        }
    }


def test_classify_substitution_covers_all_consequence_classes():
    cases = {
        "synonymous_both": (("GGT", "GGC"), ("TTA", "TTG")),
        "nonsynonymous_pol_only": (("GGT", "GAT"), ("TTA", "TTG")),
        "nonsynonymous_surface_only": (("GGT", "GGC"), ("TTA", "TCA")),
        "nonsynonymous_both": (("GGT", "GAT"), ("TTA", "TCA")),
    }
    for expected, ((ref_pol, alt_pol), (ref_surface, alt_surface)) in cases.items():
        result = classify_substitution(ref_pol, alt_pol, ref_surface, alt_surface)
        assert result["consequence_class"] == expected
    disrupted = classify_substitution("GGT", "GGC", "TTA", "TTG", disrupts_rna=True)
    assert disrupted["consequence_class"] == "disruptive_rna_element"
    assert disrupted["disrupts_rna"] is True


def test_dual_frame_table_produces_all_consequence_classes():
    frame = dual_frame_table(_dual_alignment(), _dual_config())
    assert list(frame.columns) == [column for column in DUAL_FRAME_COLUMNS if column != "lineage"]
    classes = set(frame["consequence_class"])
    assert classes == set(DUAL_FRAME_CLASSES) - {"unknown"}
    assert bool(frame[frame["consequence_class"] == "disruptive_rna_element"].iloc[0]["disrupts_rna"])
    # Spot-check the synonymous-both row's amino acids.
    syn = frame[frame["consequence_class"] == "synonymous_both"].iloc[0]
    assert syn["pol_ref_aa"] == syn["pol_alt_aa"]
    assert syn["surface_ref_aa"] == syn["surface_alt_aa"]


# --------------------------------------------------------------------------- #
# covariation / epistasis
# --------------------------------------------------------------------------- #
def _covarying_alignment():
    rows = []
    for k in range(16):
        bit = k & 1
        base0 = "A" if bit else "G"
        base1 = "C" if bit else "T"
        base2 = "A" if (k >> 1) & 1 else "C"
        rows.append((f"s{k}", base0 + base1 + base2 + "ACG"))
    return rows


def test_mutual_information_detects_the_true_covarying_pair():
    matrix = mutual_information(_covarying_alignment())
    assert matrix.shape == (6, 6)
    assert matrix[0, 1] == pytest.approx(1.0, abs=1e-9)
    assert matrix[0, 2] == pytest.approx(0.0, abs=1e-9)
    assert matrix[0, 3] == pytest.approx(0.0, abs=1e-9)


def test_apc_correction_keeps_true_pair_on_top():
    corrected = apc_correct(mutual_information(_covarying_alignment()))
    upper = np.triu_indices(corrected.shape[0], k=1)
    best = int(np.argmax(corrected[upper]))
    i, j = upper[0][best], upper[1][best]
    assert {int(i), int(j)} == {0, 1}
    assert corrected[i, j] > 0


def test_covarying_pairs_ranks_true_pair_first():
    config = {
        "selection": {
            "covariation": {"methods": ["mutual_information"], "min_seqs": 0, "min_score": 0.0}
        }
    }
    frame = covarying_pairs(_covarying_alignment(), config)
    assert list(frame.columns) == [column for column in COVARIATION_COLUMNS if column != "lineage"]
    top = frame.iloc[0]
    assert {int(top["position_i"]), int(top["position_j"])} == {1, 2}
    assert top["method"] == "mutual_information"
    assert top["score"] == pytest.approx(1.0, abs=1e-9)


def test_epistasis_pairs_flags_coupled_sites():
    config = {"selection": {"epistasis": {"method": "pairwise_epistasis", "min_support": 0.0}}}
    frame = epistasis_pairs(_covarying_alignment(), config)
    assert list(frame.columns) == [column for column in EPISTASIS_COLUMNS if column != "lineage"]
    paired = frame[
        (frame["position_i"] == 1) & (frame["position_j"] == 2)
    ]
    assert len(paired) == 1
    assert paired.iloc[0]["score"] >= 0.5
    assert paired.iloc[0]["support"] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# genotype specificity
# --------------------------------------------------------------------------- #
def test_genotype_specificity_fst_and_informative_flag():
    metadata = pd.DataFrame({
        "accession": ["a1", "a2", "b1", "b2"],
        "genotype": ["A", "A", "B", "B"],
    })
    alignment = [
        GenomeRecord(id="a1", seq="GCTGCTGCT"),
        GenomeRecord(id="a2", seq="GCTGCTGCT"),
        GenomeRecord(id="b1", seq="GATGATGAT"),
        GenomeRecord(id="b2", seq="GATGATGAT"),
    ]
    config = {
        "reference": {"pol_start_nt": 1},
        "selection": {"genotype_specificity": {"min_fst": 0.25}},
    }
    frame = genotype_specificity(alignment, metadata, config)
    assert list(frame.columns) == GENOTYPE_COLUMNS
    assert frame.iloc[0]["fst"] == pytest.approx(1.0)
    assert bool(frame.iloc[0]["genotype_informative"]) is True


def test_genotype_specificity_empty_without_metadata():
    alignment = [GenomeRecord(id="a", seq="GCT")]
    frame = genotype_specificity(alignment, pd.DataFrame(), {})
    assert frame.empty
    assert list(frame.columns) == GENOTYPE_COLUMNS


# --------------------------------------------------------------------------- #
# resistance
# --------------------------------------------------------------------------- #
def test_parse_rt_mutation_and_canonical_motif():
    assert parse_rt_mutation("rtM204V") == ("M", 204, "V")
    assert parse_rt_mutation("M204V") is None
    assert canonical_motif_at(550) is True   # YMDD (549-552)
    assert canonical_motif_at(582) is False  # rtN236T


def test_resistance_table_builtin_catalogue():
    frame = resistance_table({})
    assert list(frame.columns) == [column for column in RESISTANCE_COLUMNS if column != "lineage"]
    m204v = frame[frame["rt_mutation"] == "rtM204V"].iloc[0]
    assert int(m204v["pol_position"]) == 550
    assert bool(m204v["canonical_motif"]) is True
    n236t = frame[frame["rt_mutation"] == "rtN236T"].iloc[0]
    assert bool(n236t["canonical_motif"]) is False


def test_load_catalogue_falls_back_to_builtin(tmp_path):
    config = {"selection": {"drug_resistance": {"catalogue": str(tmp_path / "missing.tsv")}}}
    frame = load_catalogue(config, tmp_path)
    assert not frame.empty
    assert "rt_mutation" in frame.columns


# --------------------------------------------------------------------------- #
# HyPhy delegation fails soft offline
# --------------------------------------------------------------------------- #
def test_run_hyphy_method_returns_empty_schema_when_unavailable():
    frame = run_hyphy_method([("a", "ACGT")], None, "fel", {})
    assert frame.empty
    assert list(frame.columns) == [column for column in SELECTION_SITE_COLUMNS if column != "lineage"]


# --------------------------------------------------------------------------- #
# pipeline.run end-to-end (offline)
# --------------------------------------------------------------------------- #
def _fabricate_selection_inputs(root: Path) -> Path:
    outroot = root / "output"
    (outroot / "qc").mkdir(parents=True)
    (outroot / "datasets").mkdir(parents=True)

    sequences = {
        "A1": "GCT" + "AAA" * 9,
        "A2": "GCC" + "AAA" * 9,
        "A3": "GAT" + "AAA" * 9,
        "A4": "GAC" + "AAA" * 9,
        "B1": "TTA" + "CCC" * 9,
        "B2": "TTG" + "CCC" * 9,
        "B3": "TTA" + "CCA" * 9,
        "B4": "TTA" + "CCC" * 9,
    }
    write_fasta(
        [GenomeRecord(id=name, seq=seq) for name, seq in sequences.items()],
        outroot / "qc" / "hbv_oriented.fasta",
    )
    metadata = pd.DataFrame({
        "accession": list(sequences),
        "genotype": ["A", "A", "A", "A", "B", "B", "B", "B"],
    })
    write_table(metadata, outroot / "datasets" / "hbv_metadata.tsv")
    return outroot


def test_selection_pipeline_run_writes_contract_artefacts(tmp_path):
    _fabricate_selection_inputs(tmp_path)
    config = {
        "project": {"output_root": "output", "threads": 1},
        "reference": {"pol_start_nt": 1, "origin_nt": 1, "surface_frame_offset": 1},
        "selection": {
            "window": 3,
            "methods": ["fel"],
            "covariation": {"methods": ["mutual_information"], "min_seqs": 0},
            "epistasis": {"method": "pairwise_epistasis", "min_support": 0.0},
            "genotype_specificity": {"min_fst": 0.25},
        },
    }

    artefacts = selection_pipeline.run(config, tmp_path)
    outdir = tmp_path / "output" / "selection"
    for key in (
        "entropy", "dual_frame", "selection_sites", "covariation",
        "epistasis", "genotype_specificity", "resistance", "summary",
    ):
        assert artefacts[key].exists()

    entropy = pd.read_csv(outdir / "entropy.tsv", sep="\t")
    assert list(entropy.columns) == ENTROPY_COLUMNS
    assert set(entropy["lineage"]) == {"A", "B"}
    assert len(entropy) == 2 * 10  # two lineages x ten codons

    dual_frame = pd.read_csv(outdir / "dual_frame.tsv", sep="\t")
    assert list(dual_frame.columns) == DUAL_FRAME_COLUMNS
    assert not dual_frame.empty
    assert set(dual_frame["lineage"]) == {"A", "B"}

    selection_sites = pd.read_csv(outdir / "selection_sites.tsv", sep="\t")
    assert list(selection_sites.columns) == SELECTION_SITE_COLUMNS

    genotype = pd.read_csv(outdir / "genotype_specificity.tsv", sep="\t")
    assert list(genotype.columns) == GENOTYPE_COLUMNS
    assert genotype["genotype_informative"].any()

    resistance = pd.read_csv(outdir / "resistance.tsv", sep="\t")
    assert list(resistance.columns) == RESISTANCE_COLUMNS
    assert set(resistance["lineage"]) == {"A", "B"}
    assert not resistance.empty

    summary = json.loads(artefacts["summary"].read_text())
    assert summary["lineages"] == ["A", "B"]
    assert summary["hyphy_available"] is False


def test_selection_pipeline_degrades_without_upstream(tmp_path):
    config = {
        "project": {"output_root": "output", "threads": 1},
        "reference": {"pol_start_nt": 1},
        "selection": {"methods": []},
    }
    artefacts = selection_pipeline.run(config, tmp_path)
    outdir = tmp_path / "output" / "selection"
    for name, columns in (
        ("entropy", ENTROPY_COLUMNS),
        ("dual_frame", DUAL_FRAME_COLUMNS),
        ("selection_sites", SELECTION_SITE_COLUMNS),
        ("covariation", COVARIATION_COLUMNS),
        ("epistasis", EPISTASIS_COLUMNS),
        ("genotype_specificity", GENOTYPE_COLUMNS),
        ("resistance", RESISTANCE_COLUMNS),
    ):
        frame = pd.read_csv(outdir / f"{name}.tsv", sep="\t")
        assert list(frame.columns) == columns
    assert artefacts["summary"].exists()
