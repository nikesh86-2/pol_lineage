"""Tests for alignment-based span calibration (epsilon and Pol domains)."""

from __future__ import annotations

import json
import random
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hbvpol.calibrate import (  # noqa: E402
    calibrate_domain_spans,
    calibrate_epsilon_span,
    detect_pol_protein,
    find_unique_motif,
    looks_like_nucleotide,
    orient_to_reference,
    slice_wrapped,
)
from hbvpol.domain import (  # noqa: E402
    DEFAULT_DOMAIN_SPANS,
    DomainSpan,
    PolDomain,
    domain_spans_from_config,
    load_domain_spans,
    reverse_complement,
)

# Amino-acid alphabet without P, so an inserted "PPP" is an unambiguous marker.
_PROTEIN_ALPHABET = "ACDEFGHIKLMNQRSTVWY"

_CODON = {
    "A": "GCT", "C": "TGT", "D": "GAT", "E": "GAA", "F": "TTT", "G": "GGT",
    "H": "CAT", "I": "ATT", "K": "AAA", "L": "CTT", "M": "ATG", "N": "AAT",
    "P": "CCT", "Q": "CAA", "R": "CGT", "S": "TCT", "T": "ACT", "V": "GTT",
    "W": "TGG", "Y": "TAT",
}


def _random_protein(length: int, seed: int) -> str:
    rng = random.Random(seed)
    return "".join(rng.choice(_PROTEIN_ALPHABET) for _ in range(length))


# --------------------------------------------------------------------------- #
# protein-level Pol domain transfer
# --------------------------------------------------------------------------- #
def test_calibrate_domain_spans_shifts_with_spacer_insertion():
    query = _random_protein(80, seed=3)
    # Insert a unique 3-aa marker inside the spacer (after query residue 30).
    target = query[:30] + "PPP" + query[30:]
    assert len(target) == 83

    query_spans = (
        DomainSpan(PolDomain.TP, 1, 25),
        DomainSpan(PolDomain.SPACER, 26, 40),
        DomainSpan(PolDomain.RT, 41, 65),
        DomainSpan(PolDomain.RNASEH, 66, 80),
    )

    result = calibrate_domain_spans(query, target, query_spans, min_identity=0.6)

    by_domain = {span.domain: (span.start, span.end) for span in result.spans}
    assert by_domain[PolDomain.TP] == (1, 25)          # before the insertion
    assert by_domain[PolDomain.SPACER] == (26, 43)     # insertion inside it
    assert by_domain[PolDomain.RT] == (44, 68)         # shifted by +3
    assert by_domain[PolDomain.RNASEH] == (69, 83)
    assert result.identity > 0.9
    assert not result.warnings


def test_calibrate_domain_spans_keeps_canonical_order():
    query = _random_protein(60, seed=7)
    # Deliberately pass the spans out of order.
    shuffled = (
        DomainSpan(PolDomain.RT, 31, 50),
        DomainSpan(PolDomain.TP, 1, 20),
        DomainSpan(PolDomain.RNASEH, 51, 60),
        DomainSpan(PolDomain.SPACER, 21, 30),
    )
    result = calibrate_domain_spans(query, query, shuffled)
    assert [span.domain for span in result.spans] == [
        PolDomain.TP, PolDomain.SPACER, PolDomain.RT, PolDomain.RNASEH,
    ]


def test_calibrate_domain_spans_clamps_to_short_target():
    query = _random_protein(80, seed=11)
    target = query[:60]  # truncated target
    result = calibrate_domain_spans(query, target, DEFAULT_DOMAIN_SPANS)
    assert all(1 <= span.start <= span.end <= len(target) for span in result.spans)


def test_calibrate_domain_spans_form_gap_free_partition():
    query = _random_protein(80, seed=41)
    target = query[:30] + "PPP" + query[30:]  # insertion at the TP/spacer junction
    spans = (
        DomainSpan(PolDomain.TP, 1, 30),
        DomainSpan(PolDomain.SPACER, 31, 40),
        DomainSpan(PolDomain.RT, 41, 70),
        DomainSpan(PolDomain.RNASEH, 71, 80),
    )
    result = calibrate_domain_spans(query, target, spans)

    ordered = result.spans
    assert ordered[0].start == 1
    assert ordered[-1].end == len(target)
    for previous, current in zip(ordered, ordered[1:]):
        assert current.start == previous.end + 1  # no gaps, no overlap
    # The junction insertion is absorbed by the following domain.
    assert (ordered[0].start, ordered[0].end) == (1, 30)
    assert ordered[1].start == 31


# --------------------------------------------------------------------------- #
# nucleotide epsilon transfer
# --------------------------------------------------------------------------- #
def test_calibrate_epsilon_span_recovers_embedded_element():
    rng = random.Random(5)
    query = "".join(rng.choice("ACGT") for _ in range(60))
    genome = _random_nt(40, seed=6) + query + _random_nt(40, seed=8)

    hit = calibrate_epsilon_span(query, genome)
    assert (hit.start, hit.end) == (41, 100)
    assert hit.identity == pytest.approx(1.0)


def _random_nt(length: int, seed: int) -> str:
    rng = random.Random(seed)
    return "".join(rng.choice("ACGT") for _ in range(length))


# --------------------------------------------------------------------------- #
# six-frame Pol detection
# --------------------------------------------------------------------------- #
def test_detect_pol_protein_finds_offset_frame():
    query = _random_protein(40, seed=13)
    cds = "".join(_CODON[aa] for aa in query)
    genome = "A" + cds + _random_nt(30, seed=14)  # CDS sits in 0-based frame 1

    protein, strand, frame, identity, coverage = detect_pol_protein(query, genome)
    assert strand == "+"
    assert frame == 1
    assert query in protein
    assert identity > 0.9
    assert coverage > 0.9


# --------------------------------------------------------------------------- #
# per-genotype resolver
# --------------------------------------------------------------------------- #
def _write_span_table(path: Path) -> Path:
    path.write_text(
        "genotype\tdomain\tstart\tend\tsource\n"
        "default\tTP\t1\t183\tfallback\n"
        "default\tspacer\t184\t336\tfallback\n"
        "default\tRT\t337\t681\tfallback\n"
        "default\tRNaseH\t682\t832\tfallback\n"
        "a\tTP\t1\t180\tcalibrated\n"
        "a\tspacer\t181\t350\tcalibrated\n"
        "a\tRT\t351\t700\tcalibrated\n"
        "a\tRNaseH\t701\t845\tcalibrated\n",
        encoding="utf-8",
    )
    return path


def test_genotype_row_used_verbatim_and_default_clamped(tmp_path):
    table = _write_span_table(tmp_path / "pol_domain_spans.tsv")
    config = {"reference": {"domain_spans_file": str(table), "pol_length_aa": 800}}

    spans = domain_spans_from_config(config, genotype="A")
    by_domain = {span.domain: (span.start, span.end) for span in spans}
    assert by_domain[PolDomain.RT] == (351, 700)
    # A calibrated genotype row is already in that genotype's coordinates and is
    # NOT clamped to the reference's pol_length_aa.
    assert by_domain[PolDomain.RNASEH] == (701, 845)
    assert by_domain[PolDomain.TP] == (1, 180)

    # Unknown genotype falls back to the default row, which IS clamped.
    fallback = domain_spans_from_config(config, genotype="Z")
    by_domain = {span.domain: (span.start, span.end) for span in fallback}
    assert by_domain[PolDomain.RT] == (337, 681)
    assert by_domain[PolDomain.RNASEH] == (682, 800)


def test_config_list_beats_default_row_but_not_genotype_row(tmp_path):
    table = _write_span_table(tmp_path / "pol_domain_spans.tsv")
    config = {
        "reference": {
            "domain_spans_file": str(table),
            "domain_spans": [
                {"domain": "TP", "start": 1, "end": 100},
                {"domain": "RT", "start": 101, "end": 700},
            ],
        }
    }
    # No genotype -> the explicit list wins over the table's default row.
    listed = domain_spans_from_config(config)
    assert [span.domain for span in listed] == [PolDomain.TP, PolDomain.RT]
    # A calibrated genotype row is more specific still.
    calibrated = domain_spans_from_config(config, genotype="a")
    assert {span.domain: span.start for span in calibrated}[PolDomain.RT] == 351


def test_load_domain_spans_missing_file_is_empty(tmp_path):
    assert load_domain_spans(tmp_path / "nope.tsv") == {}


# --------------------------------------------------------------------------- #
# reference-frame helpers
# --------------------------------------------------------------------------- #
def test_slice_wrapped_circular():
    assert slice_wrapped("ACGTACGTAC", 3, 6) == "GTAC"
    assert slice_wrapped("ACGTACGTAC", 8, 2) == "TACAC"  # wraps the origin
    assert slice_wrapped("ACGT", 1, 4) == "ACGT"


def test_looks_like_nucleotide():
    assert looks_like_nucleotide("ACGTNacgtn")
    assert not looks_like_nucleotide("MPLQ")


def test_find_unique_motif_uniqueness():
    assert find_unique_motif("TTTTACGTACGTGGGG", ("ACGTACGT",)) == (5, "ACGTACGT")
    assert find_unique_motif("ACGTACGTACGTACGT", ("ACGTACGT",)) is None  # ambiguous
    assert find_unique_motif("ACGTACGT", ("TTTTTTTT",)) is None  # absent


def test_orient_to_reference_recovers_rotation_and_strand():
    motif = "ACGTACGTAC"
    reference = _random_nt(50, seed=1) + motif + _random_nt(140, seed=2)

    rotated = reference[30:] + reference[:30]
    result = orient_to_reference(rotated, reference, motifs=(motif,))
    assert result is not None
    oriented, strand, _ = result
    assert strand == "+"
    assert oriented == reference

    # Reverse-complemented and rotated: orientation must be detected and undone.
    flipped = reverse_complement(reference)
    rotated_flipped = flipped[37:] + flipped[:37]
    result2 = orient_to_reference(rotated_flipped, reference, motifs=(motif,))
    assert result2 is not None
    oriented2, strand2, _ = result2
    assert strand2 == "-"
    assert oriented2 == reference


def test_orient_to_reference_returns_none_without_anchor():
    assert orient_to_reference("ACGTACGT", "ACGTACGT", motifs=("TTTTTTTT",)) is None


def test_locate_epsilon_cli_normalises_rotation_and_strand(tmp_path):
    motif = "TATATGGATGAT"
    reference = _random_nt(50, seed=61) + motif + _random_nt(248, seed=62)
    ref_path = tmp_path / "reference.fasta"
    ref_path.write_text(f">ref\n{reference}\n", encoding="utf-8")

    # Rotate and reverse-complement the reference: the CLI must recover the
    # reference-frame coordinates of the 101-160 query.
    rotated = reverse_complement(reference[37:] + reference[:37])
    target_path = tmp_path / "target.fasta"
    target_path.write_text(f">tgt\n{rotated}\n", encoding="utf-8")
    out_path = tmp_path / "epsilon_spans.tsv"

    script = Path(__file__).resolve().parents[1] / "scripts" / "locate_epsilon.py"
    completed = subprocess.run(
        [
            sys.executable, str(script),
            "--reference-fasta", str(ref_path),
            "--target-fasta", str(target_path),
            "--genotype", "Z",
            "--query-span", "101", "160",
            "--out", str(out_path),
            "--min-identity", "0.9",
        ],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    rows = out_path.read_text(encoding="utf-8").strip().splitlines()
    assert rows[0].split("\t") == ["genotype", "start_nt", "end_nt", "source"]
    fields = rows[1].split("\t")
    assert fields[0] == "Z"
    assert (fields[1], fields[2]) == ("101", "160")


# --------------------------------------------------------------------------- #
# CLI smoke test
# --------------------------------------------------------------------------- #
def _load_cli_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "locate_pol_domains",
        Path(__file__).resolve().parents[1] / "scripts" / "locate_pol_domains.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pol_from_reference_features_reads_sequence_path(tmp_path):
    module = _load_cli_module()
    query = _random_protein(40, seed=31)
    cds = "".join(_CODON[aa] for aa in query)
    genome = "AC" + cds + "TT"  # Pol ORF starts at nt 3 (1-based)
    fasta = tmp_path / "ref.fasta"
    fasta.write_text(f">ref\n{genome}\n", encoding="utf-8")
    features = tmp_path / "reference_features.json"
    features.write_text(
        json.dumps({
            "reference": {
                "sequence": str(fasta),
                "sequence_path": str(fasta),
                "pol_start_nt": 3,
                "pol_end_nt": 2 + len(cds),
            }
        }),
        encoding="utf-8",
    )
    assert module.pol_from_reference_features(features) == query


def test_locate_pol_domains_cli_appends_rows(tmp_path):
    query = _random_protein(80, seed=21)
    target = query[:30] + "PPP" + query[30:]
    query_path = tmp_path / "query_pol.fasta"
    query_path.write_text(f">query\n{query}\n", encoding="utf-8")
    target_path = tmp_path / "target_pol.fasta"
    target_path.write_text(f">target\n{target}\n", encoding="utf-8")
    out_path = tmp_path / "pol_domain_spans.tsv"

    script = Path(__file__).resolve().parents[1] / "scripts" / "locate_pol_domains.py"
    completed = subprocess.run(
        [
            sys.executable, str(script),
            "--target-fasta", str(target_path),
            "--genotype", "A",
            "--query-pol-fasta", str(query_path),
            "--target-is-protein",
            "--out", str(out_path),
        ],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    rows = out_path.read_text(encoding="utf-8").strip().splitlines()
    assert rows[0].split("\t") == ["genotype", "domain", "start", "end", "source"]
    assert len(rows) == 5  # header + 4 domains
    assert rows[1].split("\t")[0] == "A"
    assert rows[1].split("\t")[1] == "TP"
