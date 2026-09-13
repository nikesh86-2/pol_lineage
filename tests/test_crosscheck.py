"""Tests for the curated deep-reference manifest and provenance cross-check."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hbvpol.datasets.crosscheck import (  # noqa: E402
    CROSSCHECK_FIELDS,
    STATUS_STATES,
    local_checks,
    run_crosscheck,
    source_statuses,
    write_crosscheck,
)
from hbvpol.datasets.deephep import (  # noqa: E402
    SEQUENCE_ORIGINS,
    load_reference_manifest,
    manifest_provenance,
)
from hbvpol.io import GenomeRecord, write_fasta  # noqa: E402

CONFIG = Path(__file__).resolve().parents[1] / "config" / "config.yaml"
MANIFEST = Path(__file__).resolve().parents[1] / "config" / "deep_hepadnavirus_references.tsv"

_CODON = {"M": "ATG", "A": "GCT", "Y": "TAT", "D": "GAT"}


def _pol_sequence(with_stop: bool = False) -> str:
    protein = "M" + "A" * 20 + "YMDD" + "A" * 20
    sequence = "".join(_CODON[aa] for aa in protein)
    if with_stop:
        # Insert an in-frame stop near the middle.
        middle = (len(sequence) // 6) * 3
        sequence = sequence[:middle] + "TAA" + sequence[middle + 3:]
    return sequence


# --------------------------------------------------------------------------- #
# curated manifest
# --------------------------------------------------------------------------- #
def test_shipped_manifest_is_well_formed():
    rows = load_reference_manifest(MANIFEST)
    assert rows, "the curated manifest should not be empty"
    accessions = {row["accession"] for row in rows}
    # Amphibian + fish references that taxon-only queries miss.
    assert "NC_030446" in accessions  # Tibetan frog
    assert "NC_030445" in accessions  # bluegill
    assert "NC_003977" in accessions  # human control
    for row in rows:
        assert row["sequence_origin"] in SEQUENCE_ORIGINS
        assert row["include_pol_tree"] in {"yes", "no", "conditional", "true", "false"}


def test_load_reference_manifest_missing_file(tmp_path):
    assert load_reference_manifest(tmp_path / "nope.tsv") == []


def test_pol_from_genbank_recognises_p_protein_and_falls_back():
    from Bio.Seq import Seq
    from Bio.SeqFeature import SimpleLocation, SeqFeature
    from Bio.SeqRecord import SeqRecord

    from hbvpol.datasets.deephep import _pol_from_genbank

    annotated = SeqRecord(Seq("ATG" * 10), id="x")
    annotated.features = [
        SeqFeature(
            SimpleLocation(0, 30),
            type="CDS",
            qualifiers={"product": ["P-protein"], "translation": ["M" * 10]},
        )
    ]
    protein, how = _pol_from_genbank(annotated, {})
    assert how == "annotated"
    assert protein == "M" * 10

    # No polymerase annotation: fall back to the longest CDS, recorded as such.
    hypothetical = SeqRecord(Seq("ATG" * 20), id="y")
    hypothetical.features = [
        SeqFeature(SimpleLocation(0, 30), type="CDS", qualifiers={"translation": ["M" * 10]}),
        SeqFeature(SimpleLocation(30, 60), type="CDS", qualifiers={"translation": ["M" * 20]}),
    ]
    protein2, how2 = _pol_from_genbank(hypothetical, {})
    assert how2 == "longest_cds_fallback"
    assert len(protein2) == 20


def test_manifest_provenance_filters_to_manifest_group():
    manifest_record = GenomeRecord(
        id="NC_030446",
        seq="MAAA",
        metadata={"group": "reference_manifest", "accession": "NC_030446", "genus": "Herpetohepadnavirus"},
    )
    query_record = GenomeRecord(id="P12345", seq="MAAA", metadata={"group": "bat"})
    rows = manifest_provenance([manifest_record, query_record])
    assert len(rows) == 1
    assert rows[0]["accession"] == "NC_030446"
    assert rows[0]["genus"] == "Herpetohepadnavirus"


# --------------------------------------------------------------------------- #
# local checks
# --------------------------------------------------------------------------- #
def test_local_checks_pass_a_clean_pol_orf():
    sequence = _pol_sequence()
    config = {"reference": {"pol_start_nt": 1, "pol_end_nt": len(sequence)}, "qc": {"min_length": 0}}
    result = local_checks([GenomeRecord(id="x", seq=sequence)], config)
    assert result["n_checked"] == 1
    assert result["pol_orf_stop_free_frac"] == 1.0
    assert result["ymdd_motif_frac"] == 1.0


def test_local_checks_flag_an_internal_stop():
    sequence = _pol_sequence(with_stop=True)
    config = {"reference": {"pol_start_nt": 1, "pol_end_nt": len(sequence)}, "qc": {"min_length": 0}}
    result = local_checks([GenomeRecord(id="x", seq=sequence)], config)
    assert result["pol_orf_stop_free_frac"] == 0.0


# --------------------------------------------------------------------------- #
# cross-check manifest
# --------------------------------------------------------------------------- #
def test_status_states_include_service_unavailable():
    assert "service_unavailable" in STATUS_STATES
    assert "not_applicable" in STATUS_STATES


def test_source_statuses_records_unavailable_sources(tmp_path):
    config = {
        "datasets": {
            "hbv": {"sources": ["genbank", "hbvdb", "glue"]},
            "crosscheck": {
                "hbvdb_snapshot": str(tmp_path / "missing_snapshot"),
                "hbv_glue_path": str(tmp_path / "missing_glue"),
                "hepadnaviridae_glue_path": str(tmp_path / "missing_glue2"),
                "stanford_catalogue": str(tmp_path / "missing_catalogue"),
            },
        }
    }
    statuses = source_statuses(config, tmp_path, {})
    assert statuses["hbvdb_status"] == "service_unavailable"
    assert statuses["hbv_glue_status"] == "not_found"
    assert statuses["stanford_hbvseq_status"] == "not_attempted"


def test_run_crosscheck_is_partial_and_writes_manifest(tmp_path):
    sequence = _pol_sequence()
    fasta = write_fasta([GenomeRecord(id="x", seq=sequence)], tmp_path / "hbv_genomes.fasta")
    config = {
        "reference": {"pol_start_nt": 1, "pol_end_nt": len(sequence)},
        "qc": {"min_length": 0},
        "datasets": {
            "hbv": {"sources": ["genbank", "hbvdb", "glue"]},
            "crosscheck": {"enabled": True, "max_records": 10},
        },
    }
    payload = run_crosscheck(config, tmp_path, fasta)
    for field in CROSSCHECK_FIELDS:
        assert field in payload
    # Local checks pass but no external source matched -> partial, not invalid.
    assert payload["crosscheck_status"] == "partial"
    assert payload["hbvdb_status"] == "service_unavailable"
    assert payload["crosscheck_details"]["local_checks"]["pol_orf_stop_free_frac"] == 1.0

    out = write_crosscheck(payload, tmp_path / "crosscheck_status.json")
    assert json.loads(out.read_text(encoding="utf-8"))["crosscheck_status"] == "partial"
