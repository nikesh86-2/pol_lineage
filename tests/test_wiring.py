"""Tests for the wiring of epsilon spans, the DCA backend and bootscan guards."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hbvpol.io import GenomeRecord  # noqa: E402
from hbvpol.pipeline import StageError  # noqa: E402


# --------------------------------------------------------------------------- #
# epsilon span calibration
# --------------------------------------------------------------------------- #
def test_resolve_epsilon_span_from_table(tmp_path):
    from hbvpol.epsilon.fold import resolve_epsilon_span

    table = tmp_path / "spans.tsv"
    table.write_text(
        "genotype\tstart_nt\tend_nt\tsource\n"
        "default\t1846\t1905\tliterature\n"
        "d\t1850\t1909\tcalibrated\n",
        encoding="utf-8",
    )
    config = {"epsilon": {"spans_file": str(table)}}
    assert resolve_epsilon_span(config, "D") == (1850, 1909)
    assert resolve_epsilon_span(config, "d") == (1850, 1909)
    assert resolve_epsilon_span(config, "A") == (1846, 1905)  # default row
    assert resolve_epsilon_span(config, None) == (1846, 1905)


def test_resolve_epsilon_span_precedence(tmp_path):
    from hbvpol.epsilon.fold import DEFAULT_EPSILON_SPAN, resolve_epsilon_span

    # Missing file -> config list -> built-in default.
    missing = {"epsilon": {"spans_file": str(tmp_path / "nope.tsv"), "genome_span": [10, 70]}}
    assert resolve_epsilon_span(missing, "A") == (10, 70)
    assert resolve_epsilon_span({}) == DEFAULT_EPSILON_SPAN
    # Table wins over the config list.
    table = tmp_path / "spans.tsv"
    table.write_text("genotype\tstart_nt\tend_nt\ndefault\t1\t60\n", encoding="utf-8")
    both = {"epsilon": {"spans_file": str(table), "genome_span": [10, 70]}}
    assert resolve_epsilon_span(both, "A") == (1, 60)


def test_extract_epsilon_uses_genotype_span(tmp_path):
    from hbvpol.epsilon.fold import extract_epsilon

    seq = list("A" * 3215)
    for position in range(1846, 1906):
        seq[position - 1] = "C"
    for position in range(1850, 1910):
        seq[position - 1] = "G"
    record = GenomeRecord(id="x", seq="".join(seq))
    config = {"epsilon": {"length_nt": [55, 70]}}
    assert extract_epsilon(record, config, genotype="A") == "C" * 4 + "G" * 56
    # A genotype-specific span selects a different slice.
    table = tmp_path / "spans.tsv"
    table.write_text("genotype\tstart_nt\tend_nt\nd\t1850\t1909\n", encoding="utf-8")
    config = {"epsilon": {"length_nt": [55, 70], "spans_file": str(table)}}
    assert extract_epsilon(record, config, genotype="D") == "G" * 60


# --------------------------------------------------------------------------- #
# DCA backend
# --------------------------------------------------------------------------- #
def _variable_alignment():
    # Columns are A|{A,C,A,C}|{A,A,T,T}|A, so the variable columns are 2 and 3
    # (1-based) and the invariant ones are 1 and 4.
    return [("s1", "AAAA"), ("s2", "ACAA"), ("s3", "AATA"), ("s4", "ACTA")]


def _dca_config(**covariation):
    base = {
        "methods": ["dca"],
        "dca_impl": "fake_dca_wiring",
        "min_seqs": 0,
        "min_entropy": 0.0,
        "max_positions": 500,
        "min_score": 0.0,
    }
    base.update(covariation)
    return {"selection": {"covariation": base}}


def test_covarying_pairs_uses_external_dca(monkeypatch):
    from hbvpol.selection.covariation import covarying_pairs

    module = types.ModuleType("fake_dca_wiring")
    calls = {"count": 0}

    def dca_scores(alignment):
        calls["count"] += 1
        width = len(alignment[0].seq) if hasattr(alignment[0], "seq") else len(alignment[0])
        matrix = np.zeros((width, width))
        # Physical columns 1 and 2 are variable; give that pair a strong score.
        matrix[1, 2] = matrix[2, 1] = 5.0
        return matrix

    module.dca_scores = dca_scores
    monkeypatch.setitem(sys.modules, "fake_dca_wiring", module)

    frame = covarying_pairs(_variable_alignment(), _dca_config())
    assert calls["count"] >= 1, "external DCA backend was not called"
    hit = frame[(frame["position_i"] == 2) & (frame["position_j"] == 3)]
    assert not hit.empty
    assert float(hit.iloc[0]["score"]) == pytest.approx(5.0)


def test_require_backend_raises_when_unavailable():
    from hbvpol.selection.covariation import covarying_pairs

    config = _dca_config(dca_impl="definitely_not_a_real_module_xyz", require_backend=True)
    with pytest.raises(StageError):
        covarying_pairs(_variable_alignment(), config)


def test_dca_falls_back_without_backend():
    from hbvpol.selection.covariation import covarying_pairs

    config = _dca_config(dca_impl="definitely_not_a_real_module_xyz", require_backend=False)
    frame = covarying_pairs(_variable_alignment(), config)  # APC proxy, no raise
    assert list(frame.columns) == ["position_i", "position_j", "method", "score", "pvalue"]


# --------------------------------------------------------------------------- #
# bootscan specificity
# --------------------------------------------------------------------------- #
def _flip_alignment():
    """Three sequences where the nearest neighbour flips on a zero margin.

    ``query`` equals ``ref2``, so ``ref2`` is the primary reference overall, yet
    across the all-A windows ``ref1`` ties it exactly -- the near-identical case
    that produced thousands of spurious calls.
    """
    ref1 = "A" * 1000
    ref2 = "A" * 500 + "T" * 100 + "A" * 400
    return [
        GenomeRecord(id="ref1", seq=ref1),
        GenomeRecord(id="ref2", seq=ref2),
        GenomeRecord(id="query", seq=ref2),
    ]


def test_bootscan_margin_suppresses_zero_margin_flip():
    from hbvpol.recombination.bootscan import bootscan_scan

    records = _flip_alignment()
    permissive = bootscan_scan(
        records, window=200, step=100, threshold=0.7,
        min_score_margin=0.0, min_region_len=0, min_support=0.0,
    )
    strict = bootscan_scan(
        records, window=200, step=100, threshold=0.7,
        min_score_margin=0.05, min_region_len=0, min_support=0.0,
    )
    assert len(permissive) > 0
    assert len(strict) == 0


def test_bootscan_min_region_len_filters_short_regions():
    from hbvpol.recombination.bootscan import bootscan_scan

    records = _flip_alignment()
    kept = bootscan_scan(
        records, window=200, step=100, threshold=0.7,
        min_score_margin=0.0, min_region_len=0, min_support=0.0,
    )
    filtered = bootscan_scan(
        records, window=200, step=100, threshold=0.7,
        min_score_margin=0.0, min_region_len=100000, min_support=0.0,
    )
    assert len(kept) > 0
    assert len(filtered) == 0
