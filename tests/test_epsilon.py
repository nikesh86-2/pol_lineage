"""Offline unit tests for the epsilon package."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hbvpol.epsilon.coevolve import (
    COMPATIBILITY_COLUMNS,
    COEVOLUTION_COLUMNS,
    compatibility_matrix,
    mutual_information,
    pol_epsilon_coevolution,
)
from hbvpol.epsilon.fold import (
    DEFAULT_EPSILON_SPAN,
    extract_epsilon,
    fold_epsilon,
    nussinov_fold,
)
from hbvpol.io import GenomeRecord


# --------------------------------------------------------------------------- #
# Nussinov folding
# --------------------------------------------------------------------------- #
def _balanced(dotbracket: str) -> bool:
    depth = 0
    for char in dotbracket:
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


def test_nussinov_fold_is_valid_and_canonical():
    seq = "GGGGAAAACCCC"
    dotbracket, n_pairs, pairs = nussinov_fold(seq, min_loop=3)
    assert len(dotbracket) == len(seq)
    assert _balanced(dotbracket)
    assert n_pairs >= 1
    canonical = {"GC", "CG", "AU", "UA", "GU", "UG"}
    for i, j in pairs:
        assert seq[i - 1].upper() + seq[j - 1].upper() in canonical


def test_nussinov_fold_respects_min_loop():
    # A perfect 4 bp stem needs a large enough loop; with min_loop 3 it folds.
    seq = "GGGG" + "AAAA" + "CCCC"
    dotbracket, n_pairs, _ = nussinov_fold(seq, min_loop=3)
    assert _balanced(dotbracket)
    assert n_pairs >= 1
    # Every pair must be separated by more than min_loop nucleotides.
    for i, j in nussinov_fold(seq, min_loop=3)[2]:
        assert j - i - 1 >= 3


def test_fold_epsilon_returns_serialisable_result():
    config = {"epsilon": {"fold_models": ["literature"], "length_nt": [55, 70]}}
    result = fold_epsilon("GGGGAAAACCCC", config)
    assert result["method"] in {"nussinov", "rnafold"}
    assert len(result["dotbracket"]) == result["length"]
    assert _balanced(result["dotbracket"])
    assert isinstance(result["pairs"], list)


# --------------------------------------------------------------------------- #
# epsilon extraction
# --------------------------------------------------------------------------- #
def test_extract_epsilon_slices_default_span():
    seq = "A" * 3215
    start, end = DEFAULT_EPSILON_SPAN
    seq = seq[: start - 1] + "G" * (end - start + 1) + seq[end:]
    record = GenomeRecord(id="x", seq=seq)
    epsilon = extract_epsilon(record, {"epsilon": {"length_nt": [55, 70]}})
    assert len(epsilon) == end - start + 1
    assert set(epsilon) == {"G"}


def test_extract_epsilon_handles_origin_wrap():
    seq = "".join(chr(ord("A") + (i % 4)) for i in range(100))
    record = GenomeRecord(id="x", seq=seq)
    config = {"epsilon": {"genome_span": [95, 5], "length_nt": [5, 20]}}
    epsilon = extract_epsilon(record, config)
    assert epsilon == seq[94:100] + seq[0:5]


# --------------------------------------------------------------------------- #
# mutual information and coevolution ranking
# --------------------------------------------------------------------------- #
def test_mutual_information_perfect_correlation():
    x = ["A"] * 10 + ["C"] * 10
    y = ["C"] * 10 + ["G"] * 10
    assert mutual_information(x, y) == pytest.approx(np.log(2), rel=1e-6)


def test_pol_epsilon_coevolution_ranks_coupled_position():
    n = 24
    pol_seqs = []
    epsilon_seqs = []
    for i in range(n):
        pol_base = "A" if i < n // 2 else "C"
        eps_base = "C" if i < n // 2 else "G"
        epsilon = list("GGGGAAAACCCC")
        epsilon[1] = eps_base
        ident = f"seq{i}"
        pol_seqs.append((ident, pol_base * 5))
        epsilon_seqs.append((ident, "".join(epsilon)))

    config = {"epsilon": {"coevolve": {"residue_window": "TP", "rna_features": ["base_pair", "bulge", "loop"]}}}
    frame = pol_epsilon_coevolution(pol_seqs, epsilon_seqs, config)

    assert list(frame.columns) == COEVOLUTION_COLUMNS
    assert not frame.empty
    top = frame.iloc[0]
    assert int(top["pol_position"]) == 1
    assert int(top["epsilon_position"]) == 2
    assert float(top["mutual_information"]) > 0.5


def test_pol_epsilon_coevolution_empty_without_shared_ids():
    config = {"epsilon": {"coevolve": {"residue_window": "TP"}}}
    frame = pol_epsilon_coevolution([("a", "AAAA")], [("b", "CCCC")], config)
    assert frame.empty
    assert list(frame.columns) == COEVOLUTION_COLUMNS


# --------------------------------------------------------------------------- #
# compatibility matrix
# --------------------------------------------------------------------------- #
def test_compatibility_matrix_shape_and_range():
    pol = {"A": "M" * 200, "B": "K" * 200}
    epsilon = {"A": "GGGGAAAACCCC", "B": "CCCCAAAAGGGG"}
    config = {"epsilon": {"compatibility": {"predictor": "contact_map_energy"}}}

    frame = compatibility_matrix(pol, epsilon, config)

    assert list(frame.columns) == COMPATIBILITY_COLUMNS
    assert len(frame) == 4
    assert frame["compatibility"].between(0.0, 1.0).all()
    assert set(frame["pol_genotype"]) == {"A", "B"}
    assert set(frame["rank"]) == {1, 2}


def test_compatibility_matrix_empty_without_genotypes():
    config = {"epsilon": {"compatibility": {"predictor": "contact_map_energy"}}}
    assert compatibility_matrix({}, {}, config).empty
