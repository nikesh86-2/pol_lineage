"""Offline unit tests for the QC package (circularisation, ORF check, dedupe)."""

from __future__ import annotations

import random
import sys
from pathlib import Path

# Make the in-repo package importable without an editable install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hbvpol.io import GenomeRecord, read_fasta, read_table, rotate_to_origin, write_fasta
from hbvpol.qc.circularise import circularise_genomes, find_origin_by_reference, orient_all
from hbvpol.qc.deduplicate import dereplicate, sequence_identity
from hbvpol.qc.orfcheck import check_genome, check_orf


# --------------------------------------------------------------------------- #
# rotate_to_origin / circularisation
# --------------------------------------------------------------------------- #
def test_rotate_to_origin_reindexes_at_origin():
    assert rotate_to_origin("ABCDE", 3) == "CDEAB"
    assert rotate_to_origin("ABCDE", 1) == "ABCDE"
    # Position 6 on a 5-mer wraps to position 1.
    assert rotate_to_origin("ABCDE", 6) == "ABCDE"
    assert rotate_to_origin("", 3) == ""


def test_circularise_genomes_uses_configured_origin():
    records = [GenomeRecord(id="g1", seq="ACGTACGT")]
    config = {"reference": {"origin_nt": 3}}

    (oriented,) = circularise_genomes(records, config)

    assert oriented.seq == "GTACGTAC"
    assert oriented.metadata["origin_shift"] == 2
    assert oriented.metadata["origin_nt_used"] == 3
    # Original record is not mutated.
    assert records[0].seq == "ACGTACGT"


def test_orient_all_reverse_complements_negative_strand():
    records = [GenomeRecord(id="minus", seq="AAAACCCC", strand="-")]
    config = {"reference": {"origin_nt": 1}}

    (oriented,) = orient_all(records, config)

    assert oriented.strand == "+"
    assert oriented.seq == "GGGGTTTT"


# --------------------------------------------------------------------------- #
# find_origin_by_reference
# --------------------------------------------------------------------------- #
def _random_dna(length: int, seed: int = 1234) -> str:
    rng = random.Random(seed)
    return "".join(rng.choice("ACGT") for _ in range(length))


def test_find_origin_by_reference_recovers_origin():
    reference = _random_dna(200)
    rotated = rotate_to_origin(reference, 75)

    position = find_origin_by_reference(rotated, reference, seed_len=25)

    assert position is not None
    assert rotate_to_origin(rotated, position) == reference
    # Reference position 1 sits at index len - shift = 126 -> 1-based 127.
    assert position == 127


def test_find_origin_by_reference_returns_none_when_seed_absent():
    reference = _random_dna(200, seed=7)
    assert find_origin_by_reference("T" * 200, reference) is None
    assert find_origin_by_reference("", reference) is None


def test_find_origin_by_reference_handles_origin_spanning_wrap():
    reference = _random_dna(120, seed=99)
    # Recut at position 100: the seed starts near the end and wraps.
    rotated = rotate_to_origin(reference, 100)
    position = find_origin_by_reference(rotated, reference, seed_len=25)
    assert position is not None
    assert rotate_to_origin(rotated, position) == reference


# --------------------------------------------------------------------------- #
# check_orf
# --------------------------------------------------------------------------- #
def test_check_orf_without_internal_stops():
    seq = "AAA" * 9 + "TAA"  # nine Lys then a terminal stop
    result = check_orf(seq, 1, 30, 30)

    assert result["length_nt"] == 30
    assert result["protein"] == "KKKKKKKKK*"
    assert result["n_stops"] == 1
    assert result["internal_stops"] == 0
    assert result["complete"] is True


def test_check_orf_with_internal_stop_fails():
    seq = "AAA" * 3 + "TAA" + "AAA" * 5 + "TAA"  # internal stop at codon 4
    result = check_orf(seq, 1, 30, 30, max_internal_stops=0)

    assert result["protein"] == "KKK*KKKKK*"
    assert result["n_stops"] == 2
    assert result["internal_stops"] == 1
    assert result["complete"] is False


def test_check_orf_origin_wrapping():
    genome = list("C" * 20)
    # Wrapped ORF covers indices 16,17,18,19,0,1,2,3,4 -> "ATGAAATAA".
    genome[16:20] = list("ATGA")
    genome[0:5] = list("AATAA")
    seq = "".join(genome)

    result = check_orf(seq, 17, 5, 20)

    assert result["length_nt"] == 9
    assert result["protein"] == "MK*"
    assert result["internal_stops"] == 0
    assert result["complete"] is True


def test_check_orf_frame_offset_shifts_reading_frame():
    seq = "ACGTACGTACG"
    frame0 = check_orf(seq, 1, 12, 12)
    frame1 = check_orf(seq, 1, 12, 12, frame_offset=1)
    assert frame0["nt"][0] == "A"
    assert frame1["nt"][0] == "C"


# --------------------------------------------------------------------------- #
# check_genome
# --------------------------------------------------------------------------- #
def _small_config(**qc_overrides) -> dict:
    qc = {
        "min_length": 10,
        "max_ambiguous_frac": 0.1,
        "require_complete_orf": ["pol"],
        "exclude_stop_in_pol": True,
        "max_internal_stops": 0,
    }
    qc.update(qc_overrides)
    return {
        "qc": qc,
        "reference": {
            "pol_start_nt": 1,
            "pol_end_nt": 30,
            "s_start_nt": 1,
            "s_end_nt": 30,
            "c_start_nt": 1,
            "c_end_nt": 30,
        },
    }


def test_check_genome_passes_clean_orfs():
    record = GenomeRecord(id="clean", seq="AAA" * 9 + "TAA")
    row = check_genome(record, _small_config())

    assert row["accession"] == "clean"
    assert row["qc_pass"] is True
    assert row["fail_reason"] == ""
    assert row["orf_pol"] is True
    assert row["pol_stops"] == 0


def test_check_genome_fails_on_internal_stop():
    record = GenomeRecord(id="broken", seq="AAA" * 3 + "TAA" + "AAA" * 5 + "TAA")
    row = check_genome(record, _small_config())

    assert row["qc_pass"] is False
    assert row["pol_stops"] == 1
    assert "stop_in_pol" in row["fail_reason"]


def test_check_genome_fails_short_and_ambiguous():
    record = GenomeRecord(id="short", seq="ACGTN")
    row = check_genome(record, _small_config(min_length=10))

    assert row["qc_pass"] is False
    assert "length" in row["fail_reason"]


# --------------------------------------------------------------------------- #
# dereplicate
# --------------------------------------------------------------------------- #
def test_sequence_identity():
    assert sequence_identity("ACGT", "ACGT") == 1.0
    assert sequence_identity("ACGT", "ACGA") == 0.75
    assert sequence_identity("ACGT", "ACG") == 0.0
    assert sequence_identity("acgt", "ACGT") == 1.0


def test_dereplicate_exact_removes_identical_records():
    records = [
        GenomeRecord(id="a", seq="ACGTACGTAA"),
        GenomeRecord(id="b", seq="ACGTACGTAA"),  # exact duplicate
        GenomeRecord(id="c", seq="TTTTTTTTTT"),
    ]
    kept = dereplicate(records)

    assert [record.id for record in kept] == ["a", "c"]


def test_dereplicate_approximate_collapses_near_duplicates():
    records = [
        GenomeRecord(id="a", seq="ACGTACGTAA"),
        GenomeRecord(id="near", seq="ACGTACGTAT"),  # 9/10 identical
        GenomeRecord(id="c", seq="TTTTTTTTTT"),
    ]
    kept = dereplicate(records, identity=0.9)

    assert [record.id for record in kept] == ["a", "c"]


def test_dereplicate_preserves_first_representative():
    records = [
        GenomeRecord(id="first", seq="ACGT"),
        GenomeRecord(id="second", seq="ACGT"),
    ]
    kept = dereplicate(records, identity=1.0)
    assert [record.id for record in kept] == ["first"]


# --------------------------------------------------------------------------- #
# pipeline.run end-to-end (offline)
# --------------------------------------------------------------------------- #
def test_qc_pipeline_run_writes_contract_artefacts(tmp_path):
    from hbvpol.qc import pipeline as qc_pipeline

    datasets = tmp_path / "output" / "datasets"
    datasets.mkdir(parents=True)
    clean_a = "AAA" * 9 + "TAA"
    clean_b = "CCC" * 9 + "TAA"
    write_fasta(
        [
            GenomeRecord(id="g1", seq=clean_a),
            GenomeRecord(id="g2", seq=clean_b),
            GenomeRecord(id="g1_dup", seq=clean_a),  # exact duplicate
        ],
        datasets / "hbv_genomes.fasta",
    )

    config = {
        "project": {"output_root": "output"},
        "qc": {
            "min_length": 10,
            "max_ambiguous_frac": 0.1,
            "require_complete_orf": ["pol"],
            "exclude_stop_in_pol": True,
            "max_internal_stops": 0,
            "circularise": True,
            "dereplicate": True,
            "dereplicate_identity": 0.9999,
        },
        "reference": {
            "origin_nt": 1,
            "pol_start_nt": 1,
            "pol_end_nt": 30,
            "s_start_nt": 1,
            "s_end_nt": 30,
            "c_start_nt": 1,
            "c_end_nt": 30,
        },
    }

    artefacts = qc_pipeline.run(config, tmp_path)

    assert set(artefacts) == {"oriented_fasta", "qc_pass", "qc_fail"}
    for path in artefacts.values():
        assert path.exists()
    oriented = read_fasta(artefacts["oriented_fasta"])
    assert [record.id for record in oriented] == ["g1", "g2"]
    pass_table = read_table(artefacts["qc_pass"])
    assert list(pass_table["accession"]) == ["g1", "g2"]
    assert pass_table["qc_pass"].all()
