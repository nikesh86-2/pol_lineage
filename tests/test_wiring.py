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
    calls = {"count": 0, "width": None}

    def dca_scores(alignment, config=None):
        calls["count"] += 1
        width = len(alignment[0].seq) if hasattr(alignment[0], "seq") else len(alignment[0])
        calls["width"] = width
        matrix = np.zeros((width, width))
        # The backend receives only the selected variable columns, so the strong
        # score belongs at (0, 1) of the sub-alignment == original columns 2, 3.
        matrix[0, 1] = matrix[1, 0] = 5.0
        return matrix

    module.dca_scores = dca_scores
    monkeypatch.setitem(sys.modules, "fake_dca_wiring", module)

    frame = covarying_pairs(_variable_alignment(), _dca_config())
    assert calls["count"] >= 1, "external DCA backend was not called"
    assert calls["width"] == 2, "backend should receive only the selected columns"
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
# plmc backend
# --------------------------------------------------------------------------- #
def test_plmc_parse_couplings(tmp_path):
    from hbvpol.selection.plmc_backend import parse_couplings

    path = tmp_path / "couplings.txt"
    path.write_text(
        "1 - 2 - 0 0.5\n1 - 3 - 0 -0.25\n2 - 3 - 0 0.75\n# trailing junk\n",
        encoding="utf-8",
    )
    pairs = parse_couplings(path)
    assert pairs[(0, 1)] == pytest.approx(0.5)
    assert pairs[(0, 2)] == pytest.approx(-0.25)
    assert pairs[(1, 2)] == pytest.approx(0.75)


def test_plmc_requires_executable():
    from hbvpol.pipeline import StageError
    from hbvpol.selection.plmc_backend import dca_scores

    config = {"selection": {"covariation": {"plmc_executable": "definitely_missing_plmc"}}}
    with pytest.raises(StageError):
        dca_scores([("s1", "ACGT"), ("s2", "ACGA")], config)


def test_plmc_runs_and_returns_symmetric_matrix():
    import random
    import shutil

    if shutil.which("plmc") is None:
        pytest.skip("plmc is not installed")
    from hbvpol.selection.plmc_backend import dca_scores

    rng = random.Random(1)
    base = "".join(rng.choice("ACGT") for _ in range(14))
    records = []
    for index in range(40):
        seq = list(base)
        for _ in range(5):
            seq[rng.randrange(len(seq))] = rng.choice("ACGT")
        records.append((f"s{index}", "".join(seq)))

    config = {"selection": {"covariation": {"plmc_executable": "plmc"}}}
    matrix = dca_scores(records, config)
    assert matrix.shape == (14, 14)
    assert np.allclose(matrix, matrix.T)
    assert np.count_nonzero(matrix) > 0


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


# --------------------------------------------------------------------------- #
# deep-hepadnavirus group queries
# --------------------------------------------------------------------------- #
def test_group_query_list_override_and_suffix_control():
    from hbvpol.datasets.deephep import _group_query

    config = {
        "deephep": {
            "group_queries": {
                "bat": ["Bat hepatitis B virus[Organism]", "Bat hepadnavirus[Organism]"],
            },
            "group_query_suffixes": {"bat": ""},
        }
    }
    query = _group_query("bat", config)
    assert "Bat hepatitis B virus[Organism]" in query
    assert "Bat hepadnavirus[Organism]" in query
    assert " OR " in query
    # An empty per-group suffix disables the Pol restriction entirely.
    assert "polymerase[Title]" not in query


def test_group_query_appends_default_suffix_once():
    from hbvpol.datasets.deephep import _group_query

    # A taxon query with no Pol terms gets the generic suffix appended.
    appended = _group_query("primate", {})
    assert appended.count("polymerase[Title]") == 1

    # The nackednavirus default already embeds its own Pol restriction
    # (indexed by protein title), so the generic suffix must NOT be added.
    nacked = _group_query("nackednavirus", {})
    assert "nackednavirus[Title]" in nacked
    assert nacked.count("polymerase[Title]") == 1
    assert "P[Title]" in nacked and "ORF2[Title]" in nacked


def test_probe_group_queries_counts(monkeypatch):
    Bio = pytest.importorskip("Bio")
    from hbvpol.datasets.deephep import probe_group_queries

    calls = []

    class _Handle:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    class _FakeEntrez:
        email = None
        tool = None

        def esearch(self, **kwargs):
            calls.append(kwargs.get("term"))
            return _Handle()

        def read(self, handle):
            return {"Count": "42"}

    monkeypatch.setattr(Bio, "Entrez", _FakeEntrez(), raising=False)
    config = {"datasets": {"hbv": {"genbank": {"email": "x@y"}}},
              "deephep": {"taxonomic_groups": ["bat", "rodent"]}}
    counts = probe_group_queries(config)
    assert counts == {"bat": 42, "rodent": 42}
    assert any("Bat hepatitis B virus[Organism]" in term for term in calls)


# --------------------------------------------------------------------------- #
# recombination tool wrappers
# --------------------------------------------------------------------------- #
def test_threeseq_parses_biondi_schema(tmp_path):
    from hbvpol.recombination.partition import BREAKPOINT_COLUMNS
    from hbvpol.recombination.threeseq import parse_breakpoints_field, parse_threeseq_output

    assert parse_breakpoints_field("100-100 & 200-200") == (100, 200)
    assert parse_breakpoints_field("44-44") == (44, 44)
    assert parse_breakpoints_field("") == (1, 1)

    path = tmp_path / "3s.rec.csv"
    path.write_text(
        "P_ACCNUM,Q_ACCNUM,C_ACCNUM,m,n,k,p,HS?,log(p),DS(p),DS(p),min_rec_length,breakpoints\n"
        "B,C,R,74,76,76,0.0,0,-42.0871,0.0,1.718552e-40,99,100-100 & 200-200\n",
        encoding="utf-8",
    )
    frame = parse_threeseq_output(path)
    assert list(frame.columns) == BREAKPOINT_COLUMNS
    row = frame.iloc[0]
    assert row["recombinant_id"] == "R"
    assert row["partner"] == "B|C"
    assert (int(row["bp_start"]), int(row["bp_end"])) == (100, 200)
    assert row["tool"] == "threeseq"


def test_openrdp_parses_csv(tmp_path):
    from hbvpol.recombination.openrdp import parse_openrdp_output
    from hbvpol.recombination.partition import BREAKPOINT_COLUMNS

    path = tmp_path / "openrdp.csv"
    path.write_text(
        "Method,Start,End,Recombinant,Parent1,Parent2,Pvalue\n"
        "Rdp,0,80,A,B,C,4.875e-08\n"
        "Threeseq,100,200,B,C,R,1.718e-40\n",
        encoding="utf-8",
    )
    frame = parse_openrdp_output(path)
    assert list(frame.columns) == BREAKPOINT_COLUMNS
    assert set(frame["recombinant_id"]) == {"A", "B"}
    assert frame.iloc[0]["partner"] == "B|C"
    assert frame.iloc[0]["region"] == "Rdp"
    assert frame.iloc[0]["tool"] == "openrdp"


def test_recombination_wrappers_fail_soft(tmp_path):
    from hbvpol.recombination.openrdp import run_openrdp
    from hbvpol.recombination.partition import BREAKPOINT_COLUMNS
    from hbvpol.recombination.threeseq import run_threeseq

    config = {"recombination": {
        "openrdp": {"executable": "definitely_missing_openrdp"},
        "threeseq": {"executable": "definitely_missing_3seq"},
    }}
    for frame in (run_openrdp("missing.fa", config, tmp_path),
                  run_threeseq("missing.fa", config, tmp_path)):
        assert frame.empty
        assert list(frame.columns) == BREAKPOINT_COLUMNS


# --------------------------------------------------------------------------- #
# protein-space covariation
# --------------------------------------------------------------------------- #
def test_translate_pol_alignment_positions_are_residues():
    from hbvpol.selection.dualframe import translate_pol_alignment

    config = {"reference": {"pol_start_nt": 1}}
    # Two codons: ATG GCT / ATG GCC -> Met Ala (synonymous at codon 2).
    translated = translate_pol_alignment([("s1", "ATGGCT"), ("s2", "ATGGCC")], config)
    assert [record.seq for record in translated] == ["MA", "MA"]

    # Gaps and ambiguous codons become '-', keeping the alignment rectangular.
    gapped = translate_pol_alignment([("s1", "ATG---"), ("s2", "ATGNNT")], config)
    assert [record.seq for record in gapped] == ["M-", "M-"]
    assert len({len(record.seq) for record in gapped}) == 1


def test_covariation_on_protein_reports_residue_positions():
    from hbvpol.selection.covariation import covarying_pairs

    # Length-4 protein alignment; residues 2 and 3 vary A/C and are perfectly
    # coupled, so the pair has MI = 1 bit.  (An A/C x A/T layout would be
    # statistically independent and give MI = 0.)
    protein = [("s1", "MAAA"), ("s2", "MCCA"), ("s3", "MAAA"), ("s4", "MCCA")]
    config = {"selection": {"covariation": {
        "methods": ["mutual_information"], "min_seqs": 0,
        "min_entropy": 0.0, "max_positions": 50, "min_score": 0.0,
    }}}
    frame = covarying_pairs(protein, config)
    assert not frame.empty
    # Positions are Pol residues (<= protein length), not genome columns.
    assert int(frame["position_i"].max()) <= 4
    assert int(frame["position_j"].max()) <= 4
    assert set(frame["position_i"]) == {2}
    assert set(frame["position_j"]) == {3}
