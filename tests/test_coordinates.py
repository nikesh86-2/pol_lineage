"""Tests for reference-frame column mapping on gapped alignments."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hbvpol.coordinates import (  # noqa: E402
    assign_domains,
    columns_for_nt_range,
    reference_positions,
    reference_sequence_from_config,
)
from hbvpol.io import GenomeRecord, write_fasta  # noqa: E402
from hbvpol.recombination.partition import extract_domain_subalignments  # noqa: E402


def _random_nt(length: int, seed: int) -> str:
    rng = random.Random(seed)
    return "".join(rng.choice("ACGT") for _ in range(length))


def _reference_config(tmp_path: Path, reference: str, **reference_overrides) -> dict:
    path = write_fasta([GenomeRecord(id="ref", seq=reference)], tmp_path / "reference.fasta")
    config = {"reference": {"sequence": str(path), "pol_start_nt": 1, **reference_overrides}}
    return config


# --------------------------------------------------------------------------- #
# reference_positions
# --------------------------------------------------------------------------- #
def test_reference_positions_none_without_reference():
    records = [GenomeRecord(id="a", seq="ACGTACGT")]
    assert reference_positions(records, {}) is None


def test_reference_positions_identity_for_the_reference_record(tmp_path):
    reference = _random_nt(60, seed=1)
    config = _reference_config(tmp_path, reference, accession="ref")
    positions = reference_positions([GenomeRecord(id="ref", seq=reference)], config)
    assert positions == list(range(1, 61))


def test_reference_positions_mark_insertions_as_none(tmp_path):
    reference = _random_nt(60, seed=2)
    representative = reference[:30] + "TTT" + reference[30:]
    config = _reference_config(tmp_path, reference, accession="ref")
    positions = reference_positions([GenomeRecord(id="rep", seq=representative)], config)

    assert positions is not None
    assert len(positions) == 63
    assert positions[29] == 30           # last residue before the insertion
    assert positions[30:33] == [None, None, None]  # the TTT insertion
    assert positions[33] == 31           # downstream numbering is not shifted


def test_columns_for_nt_range_assigns_insertions_to_the_neighbouring_domain(tmp_path):
    reference = _random_nt(60, seed=3)
    # Insertion internal to the first half.
    representative = reference[:20] + "AA" + reference[20:]
    config = _reference_config(tmp_path, reference, accession="ref")
    positions = reference_positions([GenomeRecord(id="rep", seq=representative)], config)

    first = columns_for_nt_range(positions, 1, 20, genome_length=60)
    second = columns_for_nt_range(positions, 21, 60, genome_length=60)
    assert first == list(range(0, 22))   # 20 reference bases + the 2-nt insertion
    assert second == list(range(22, 62))


# --------------------------------------------------------------------------- #
# extract_domain_subalignments on a gapped alignment
# --------------------------------------------------------------------------- #
def test_extract_domain_subalignments_uses_reference_coordinates(tmp_path):
    reference = _random_nt(60, seed=4)
    # A genuine gapped MSA: the reference row carries gaps opposite the insertion.
    ref_gapped = reference[:30] + "---" + reference[30:]
    ins_gapped = reference[:30] + "TTT" + reference[30:]
    alignment = [
        GenomeRecord(id="ref", seq=ref_gapped),
        GenomeRecord(id="ins", seq=ins_gapped),
    ]
    config = _reference_config(
        tmp_path,
        reference,
        accession="ref",
        pol_length_aa=20,
        domain_spans=[
            {"domain": "TP", "start": 1, "end": 10},      # nt 1..30
            {"domain": "spacer", "start": 11, "end": 20},  # nt 31..60
        ],
    )

    subalignments = extract_domain_subalignments(alignment, config)
    tp = subalignments[next(domain for domain in subalignments if domain.value == "TP")]
    spacer = subalignments[next(domain for domain in subalignments if domain.value == "spacer")]

    def parse(text: str) -> dict[str, str]:
        records: dict[str, str] = {}
        ident = None
        for line in text.splitlines():
            if line.startswith(">"):
                ident = line[1:]
                records[ident] = ""
            elif ident is not None:
                records[ident] += line
        return records

    tp_records = parse(tp)
    # Reference positions 1-30 plus the boundary insertion assigned to TP.
    assert tp_records["ref"] == reference[:30] + "---"
    assert tp_records["ins"] == reference[:30] + "TTT"
    spacer_records = parse(spacer)
    assert spacer_records["ref"] == reference[30:]
    assert spacer_records["ins"] == reference[30:]


# --------------------------------------------------------------------------- #
# assign_domains
# --------------------------------------------------------------------------- #
def test_assign_domains_labels_by_reference_position(tmp_path):
    reference = _random_nt(60, seed=5)
    config = _reference_config(
        tmp_path,
        reference,
        accession="ref",
        pol_length_aa=20,
        domain_spans=[
            {"domain": "TP", "start": 1, "end": 10},
            {"domain": "spacer", "start": 11, "end": 20},
        ],
    )
    positions = reference_positions([GenomeRecord(id="ref", seq=reference)], config)
    labels = assign_domains(positions, config)
    assert labels[0] == "TP"
    assert labels[29] == "TP"
    assert labels[30] == "spacer"
    assert labels[59] == "spacer"


def test_reference_sequence_from_config_reads_path(tmp_path):
    reference = _random_nt(40, seed=6)
    config = _reference_config(tmp_path, reference)
    assert reference_sequence_from_config(config) == reference
