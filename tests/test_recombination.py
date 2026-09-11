"""Offline unit tests for the recombination package."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hbvpol.domain import PolDomain
from hbvpol.io import GenomeRecord, read_fasta, write_fasta
from hbvpol.recombination import bootscan, gard, rdp5
from hbvpol.recombination.partition import (
    BREAKPOINT_COLUMNS,
    extract_domain_subalignments,
    partition_alignment,
    reconcile_breakpoints,
)


# --------------------------------------------------------------------------- #
# reconcile_breakpoints
# --------------------------------------------------------------------------- #
def _frame(rows):
    return pd.DataFrame(rows, columns=BREAKPOINT_COLUMNS)


def test_reconcile_clusters_nearby_breakpoints_and_marks_consensus():
    tool_a = _frame([
        ("R1", "P1", "toolA", 100, 110, 0.90, ""),
        ("R1", "P1", "toolA", 500, 510, 0.80, ""),
        ("R2", "P2", "toolA", 200, 210, 0.70, ""),
    ])
    tool_b = _frame([
        ("R1", "P1", "toolB", 105, 115, 0.95, ""),
        ("R1", "P1", "toolB", 505, 515, 0.85, ""),
    ])
    config = {
        "recombination": {
            "tools": ["toolA", "toolB"],
            "consensus_frac": 0.75,
            "breakpoint_tolerance": 20,
        }
    }

    result = reconcile_breakpoints([tool_a, tool_b], config)

    # Three clusters: R1 x2 (supported by both tools) and R2 x1 (one tool).
    assert len(result) == 3
    by_id = result.set_index("recombinant_id")
    assert bool(by_id.loc["R2", "consensus"]) is False
    assert int(by_id.loc["R2", "n_tools_supporting"]) == 1

    r1 = result[result["recombinant_id"] == "R1"]
    assert r1["consensus"].all()
    assert set(r1["n_tools_supporting"]) == {2}
    # Nearby calls are merged into one cluster spanning both tools' intervals.
    assert int(r1.iloc[0]["bp_start"]) == 100
    assert int(r1.iloc[0]["bp_end"]) == 115


def test_reconcile_consensus_frac_controls_acceptance():
    tool_a = _frame([("R1", "P1", "toolA", 100, 100, 0.9, "")])
    tool_b = _frame([("R1", "P1", "toolB", 500, 500, 0.9, "")])
    base = {"tools": ["toolA", "toolB"], "breakpoint_tolerance": 10}

    strict = reconcile_breakpoints(
        [tool_a, tool_b], {"recombination": {**base, "consensus_frac": 0.75}}
    )
    assert not strict["consensus"].any()

    loose = reconcile_breakpoints(
        [tool_a, tool_b], {"recombination": {**base, "consensus_frac": 0.5}}
    )
    assert loose["consensus"].all()


def test_reconcile_empty_frames_returns_empty_schema():
    result = reconcile_breakpoints([], {"recombination": {"tools": []}})
    assert result.empty
    assert "consensus" in result.columns


def test_reconcile_separates_breakpoints_beyond_tolerance():
    tool_a = _frame([
        ("R1", "P1", "toolA", 100, 100, 0.9, ""),
        ("R1", "P1", "toolA", 400, 400, 0.9, ""),
    ])
    config = {"recombination": {"tools": ["toolA"], "consensus_frac": 0.5, "breakpoint_tolerance": 20}}
    result = reconcile_breakpoints([tool_a], config)
    assert len(result) == 2


# --------------------------------------------------------------------------- #
# partition_alignment
# --------------------------------------------------------------------------- #
@pytest.fixture()
def synthetic_alignment(tmp_path):
    sequences = [
        GenomeRecord(id="s1", seq=("ACGT" * 250)),
        GenomeRecord(id="s2", seq=("ACGT" * 250)),
        GenomeRecord(id="s3", seq=("ACGT" * 250)),
    ]
    path = write_fasta(sequences, tmp_path / "aln.fasta")
    return path


def test_partition_alignment_block_coordinates_and_files(synthetic_alignment, tmp_path):
    breakpoints = pd.DataFrame([{
        "recombinant_id": "R1",
        "partner": "P1",
        "tool": "toolA",
        "bp_start": 400,
        "bp_end": 400,
        "support": 0.9,
        "region": "",
        "cluster_id": 1,
        "n_tools_supporting": 2,
        "n_tools_total": 2,
        "consensus": True,
    }])
    config = {
        "recombination": {
            "min_block_len": 300,
            "block_aln_dir": str(tmp_path / "block_alns"),
            "tools": ["toolA", "toolB"],
        }
    }

    blocks, paths = partition_alignment(synthetic_alignment, breakpoints, config)

    assert list(blocks["start_nt"]) == [1, 401]
    assert list(blocks["end_nt"]) == [400, 1000]
    assert list(blocks["n_seqs"]) == [3, 3]
    assert list(blocks["n_tools_supporting"]) == [2, 2]
    assert set(paths) == {"block_1", "block_2"}
    assert paths["block_1"].exists()

    block1 = read_fasta(paths["block_1"])
    assert len(block1) == 3
    assert all(len(record.seq) == 400 for record in block1)
    block2 = read_fasta(paths["block_2"])
    assert all(len(record.seq) == 600 for record in block2)


def test_partition_alignment_merges_blocks_below_min_length(synthetic_alignment, tmp_path):
    breakpoints = pd.DataFrame([
        {"recombinant_id": "R1", "partner": "P1", "tool": "t", "bp_start": 100,
         "bp_end": 100, "support": 0.9, "region": "", "n_tools_supporting": 2,
         "n_tools_total": 2, "consensus": True},
        {"recombinant_id": "R1", "partner": "P1", "tool": "t", "bp_start": 150,
         "bp_end": 150, "support": 0.9, "region": "", "n_tools_supporting": 2,
         "n_tools_total": 2, "consensus": True},
    ])
    config = {
        "recombination": {
            "min_block_len": 300,
            "block_aln_dir": str(tmp_path / "block_alns"),
            "tools": ["a", "b"],
        }
    }

    blocks, paths = partition_alignment(synthetic_alignment, breakpoints, config)

    # (1-100) and (101-150) are both under 300 nt, so everything merges.
    assert len(blocks) == 1
    assert int(blocks.iloc[0]["start_nt"]) == 1
    assert int(blocks.iloc[0]["end_nt"]) == 1000
    assert list(paths) == ["block_1"]


def test_partition_alignment_ignores_non_consensus_breakpoints(synthetic_alignment, tmp_path):
    breakpoints = pd.DataFrame([{
        "recombinant_id": "R1", "partner": "P1", "tool": "t", "bp_start": 500,
        "bp_end": 500, "support": 0.9, "region": "", "n_tools_supporting": 1,
        "n_tools_total": 2, "consensus": False,
    }])
    config = {
        "recombination": {
            "min_block_len": 100,
            "block_aln_dir": str(tmp_path / "block_alns"),
            "tools": ["a", "b"],
        }
    }
    blocks, _ = partition_alignment(synthetic_alignment, breakpoints, config)
    assert len(blocks) == 1
    assert int(blocks.iloc[0]["end_nt"]) == 1000


# --------------------------------------------------------------------------- #
# extract_domain_subalignments
# --------------------------------------------------------------------------- #
def test_extract_domain_subalignments_covers_all_domains(tmp_path):
    width = 3215
    records = [
        GenomeRecord(id="s1", seq="A" * width),
        GenomeRecord(id="s2", seq="C" * width),
    ]
    path = write_fasta(records, tmp_path / "aln.fasta")
    config = {"reference": {"pol_start_nt": 1}}

    domains = extract_domain_subalignments(path, config)

    assert set(domains) == set(PolDomain)
    tp_lines = [line for line in domains[PolDomain.TP].splitlines() if line.startswith(">")]
    assert len(tp_lines) == 2
    # TP spans 183 aa -> 549 nt.
    tp_seq = domains[PolDomain.TP].splitlines()[1]
    assert len(tp_seq) == 183 * 3


# --------------------------------------------------------------------------- #
# tool wrappers fail soft when the executable is absent
# --------------------------------------------------------------------------- #
def test_rdp5_fails_soft_when_unavailable(synthetic_alignment, tmp_path):
    config = {"recombination": {"rdp5": {"executable": "definitely-not-rdp5-xyz"}}}
    frame = rdp5.run_rdp5(synthetic_alignment, config, tmp_path / "work")
    assert frame.empty
    assert list(frame.columns) == BREAKPOINT_COLUMNS


def test_gard_fails_soft_when_unavailable(synthetic_alignment, tmp_path):
    config = {"recombination": {"gard": {"executable": "definitely-not-hyphy-xyz"}}}
    frame = gard.run_gard(synthetic_alignment, config, tmp_path / "work")
    assert frame.empty
    assert list(frame.columns) == BREAKPOINT_COLUMNS


# --------------------------------------------------------------------------- #
# parsers
# --------------------------------------------------------------------------- #
def test_parse_rdp5_output(tmp_path):
    csv_path = tmp_path / "rdp5.csv"
    csv_path.write_text(
        "# RDP5 report\n"
        "Recombinant,Minor Parent,Start,End,P-value\n"
        "R1,P1,100,120,0.01\n"
        "R2,P2,300,330,0.05\n"
    )
    frame = rdp5.parse_rdp5_output(csv_path)
    assert list(frame.columns) == BREAKPOINT_COLUMNS
    assert len(frame) == 2
    assert frame.iloc[0]["tool"] == "rdp5"
    assert int(frame.iloc[0]["bp_start"]) == 100
    assert frame.iloc[0]["recombinant_id"] == "R1"


def test_parse_gard_json(tmp_path):
    json_path = tmp_path / "gard.GARD.json"
    json_path.write_text(json.dumps({"breakpoints": [100, 500, 900]}))
    frame = gard.parse_gard_json(json_path)
    assert list(frame.columns) == BREAKPOINT_COLUMNS
    assert list(frame["bp_start"]) == [100, 500, 900]
    assert set(frame["tool"]) == {"gard"}


# --------------------------------------------------------------------------- #
# pure-Python bootscan
# --------------------------------------------------------------------------- #
def test_bootscan_scan_detects_a_mosaic():
    a = "ACGT" * 250
    b = "TTAA" * 250
    c = "GGCC" * 250
    query = a[:800] + b[800:1000]
    records = [
        GenomeRecord(id="q", seq=query),
        GenomeRecord(id="A", seq=a),
        GenomeRecord(id="B", seq=b),
        GenomeRecord(id="C", seq=c),
    ]

    frame = bootscan.bootscan_scan(records, window=200, step=50, threshold=0.6)

    assert not frame.empty
    assert (frame["tool"] == "bootscan").all()
    assert set(frame[frame["recombinant_id"] == "q"]["partner"]) == {"B"}


def test_run_bootscan_uses_offline_scan(synthetic_alignment, tmp_path):
    config = {"recombination": {"bootscan": {"window": 100, "step": 50, "threshold": 0.6, "replicates": 10}}}
    frame = bootscan.run_bootscan(synthetic_alignment, config, tmp_path / "work")
    assert list(frame.columns) == BREAKPOINT_COLUMNS


# --------------------------------------------------------------------------- #
# pipeline.run end-to-end (offline; external tools absent -> skipped)
# --------------------------------------------------------------------------- #
def test_recombination_pipeline_run_writes_contract_artefacts(tmp_path):
    from hbvpol.recombination import pipeline as recomb_pipeline

    qc_dir = tmp_path / "output" / "qc"
    qc_dir.mkdir(parents=True)
    a = "ACGT" * 250
    b = "TTAA" * 250
    c = "GGCC" * 250
    d = "CTAG" * 250
    records = [
        GenomeRecord(id="q", seq=a[:800] + b[800:]),
        GenomeRecord(id="a", seq=a),
        GenomeRecord(id="b", seq=b),
        GenomeRecord(id="c", seq=c),
        GenomeRecord(id="d", seq=d),
    ]
    write_fasta(records, qc_dir / "hbv_oriented.fasta")

    config = {
        "project": {"output_root": "output", "threads": 1},
        "reference": {"pol_start_nt": 1},
        "recombination": {
            "tools": ["rdp5", "gard", "threeseq", "bootscan"],
            "consensus_frac": 0.5,
            "min_block_len": 200,
            "rdp5": {"executable": "definitely-not-rdp5-xyz"},
            "gard": {"executable": "definitely-not-hyphy-xyz"},
            "threeseq": {"executable": "definitely-not-3seq-xyz"},
            "bootscan": {"window": 200, "step": 50, "replicates": 10, "threshold": 0.6},
        },
    }

    artefacts = recomb_pipeline.run(config, tmp_path)

    outdir = tmp_path / "output" / "recombination"
    for key in ("breakpoints", "blocks", "summary"):
        assert artefacts[key].exists()
    assert artefacts["block_alns"].is_dir()
    assert artefacts["domain_subalns"].is_dir()

    breakpoints = pd.read_csv(outdir / "breakpoints.tsv", sep="\t")
    assert list(breakpoints.columns) == BREAKPOINT_COLUMNS
    blocks = pd.read_csv(outdir / "blocks.tsv", sep="\t")
    assert len(blocks) >= 1
    assert int(blocks.iloc[0]["start_nt"]) == 1
    # Every written block alignment is a real FASTA file.
    for path in outdir.glob("block_alns/block_*.fasta"):
        assert read_fasta(path)
    summary = json.loads(artefacts["summary"].read_text())
    assert summary["tools_run"] == ["rdp5", "gard", "threeseq", "bootscan"]
    assert "bootscan" in summary["breakpoints_per_tool"]
