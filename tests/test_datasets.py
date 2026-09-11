"""Offline unit tests for :mod:`hbvpol.datasets`.

Everything here avoids the network and external tools: network-capable code paths
are exercised only for their pre-flight validation, and the stage-level test uses
a config with no configured sources.
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import pytest

# The package uses a src/ layout and may not be installed in the test env.
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from hbvpol.io import GenomeRecord  # noqa: E402
from hbvpol.pipeline import StageError  # noqa: E402

from hbvpol.datasets import (  # noqa: E402
    HBV_METADATA_COLUMNS,
    check_release_freshness,
    fetch_genbank,
    filter_by_length,
    normalize_metadata,
    parse_hbvdb_record,
    run,
)


# --------------------------------------------------------------------------- #
# HBVdb record parsing
# --------------------------------------------------------------------------- #


def test_parse_hbvdb_record_basic() -> None:
    raw = {
        "accession": "AB123456",
        "sequence": "ACGTACGTACGT",
        "genotype": "C2",
        "country": "Japan",
        "collection_date": "2015-06-01",
        "isolation_source": "serum",
        "patient_id": "P42",
        "definition": "Hepatitis B virus isolate P42",
    }
    record = parse_hbvdb_record(raw)

    assert isinstance(record, GenomeRecord)
    assert record.id == "AB123456"
    assert record.seq == "ACGTACGTACGT"
    assert record.source == "hbvdb"
    assert record.metadata["genotype"] == "C"
    assert record.metadata["subgenotype"] == "C2"
    assert record.metadata["country"] == "Japan"
    assert record.metadata["year"] == "2015"
    assert record.metadata["isolation_source"] == "serum"
    assert record.metadata["patient_id"] == "P42"
    assert record.metadata["length"] == len("ACGTACGTACGT")


def test_parse_hbvdb_record_is_total_on_empty_input() -> None:
    record = parse_hbvdb_record({})
    assert record.seq == ""
    assert record.id == "hbvdb_record"
    assert record.metadata["genotype"] == ""
    assert record.metadata["year"] == ""


def test_parse_hbvdb_record_infers_labelled_genotype_from_description() -> None:
    record = parse_hbvdb_record(
        {
            "accession": "X1",
            "sequence": "ACGT",
            "description": "Hepatitis B virus genotype D3 complete genome",
        }
    )
    assert record.metadata["genotype"] == "D"
    assert record.metadata["subgenotype"] == "D3"


# --------------------------------------------------------------------------- #
# metadata normalisation
# --------------------------------------------------------------------------- #


def test_normalize_metadata_pins_documented_columns() -> None:
    frame = normalize_metadata(
        [
            {"accession": "X", "genotype": "D"},
            {"accession": "Y", "year": 2019, "length": "1234"},
        ]
    )
    assert list(frame.columns) == list(HBV_METADATA_COLUMNS)
    assert frame.loc[0, "country"] == ""  # missing qualifier -> empty string, not NaN
    assert frame.loc[1, "year"] == "2019"
    assert frame.loc[1, "length"] == 1234


def test_normalize_metadata_empty_is_well_formed() -> None:
    frame = normalize_metadata([])
    assert list(frame.columns) == list(HBV_METADATA_COLUMNS)
    assert len(frame) == 0


# --------------------------------------------------------------------------- #
# GenBank metadata parsing (Biopython-dependent, skipped if absent)
# --------------------------------------------------------------------------- #


def test_parse_genbank_metadata_constructed_record() -> None:
    pytest.importorskip("Bio")
    from Bio.Seq import Seq
    from Bio.SeqFeature import FeatureLocation, SeqFeature
    from Bio.SeqRecord import SeqRecord

    from hbvpol.datasets import parse_genbank_metadata

    sequence = Seq("ACGT" * 60)
    record = SeqRecord(
        sequence,
        id="AB123456.1",
        name="AB123456",
        description="Hepatitis B virus isolate patient42 genotype C2, complete genome",
    )
    source = SeqFeature(FeatureLocation(0, len(sequence)), type="source")
    source.qualifiers = {
        "country": ["Japan"],
        "collection_date": ["2015"],
        "isolate": ["patient42"],
        "isolation_source": ["serum"],
        "genotype": ["C2"],
    }
    record.features.append(source)

    metadata = parse_genbank_metadata(record)

    assert metadata["accession"] == "AB123456.1"
    assert metadata["genotype"] == "C"
    assert metadata["subgenotype"] == "C2"
    assert metadata["country"] == "Japan"
    assert metadata["year"] == "2015"
    assert metadata["isolation_source"] == "serum"
    assert metadata["patient_id"] == "patient42"
    assert metadata["length"] == len(sequence)
    assert metadata["source"] == "genbank"


def test_parse_genbank_metadata_missing_qualifiers_are_empty() -> None:
    pytest.importorskip("Bio")
    from Bio.Seq import Seq
    from Bio.SeqRecord import SeqRecord

    from hbvpol.datasets import parse_genbank_metadata

    record = SeqRecord(Seq("ACGT"), id="ZZ1", description="unannotated")
    metadata = parse_genbank_metadata(record)

    assert metadata["genotype"] == ""
    assert metadata["country"] == ""
    assert metadata["year"] == ""
    assert metadata["patient_id"] == ""


# --------------------------------------------------------------------------- #
# deephep filtering
# --------------------------------------------------------------------------- #


def test_filter_by_length_is_inclusive() -> None:
    records = [GenomeRecord(id=str(n), seq="A" * n) for n in (100, 500, 800, 2000)]
    kept = filter_by_length(records, min_length=500, max_length=1200)
    assert [record.id for record in kept] == ["500", "800"]


def test_filter_by_length_all_rejected_when_bounds_exclude() -> None:
    records = [GenomeRecord(id="a", seq="A" * 10)]
    assert filter_by_length(records, 500, 1200) == []


# --------------------------------------------------------------------------- #
# pre-flight validation (never reaches the network)
# --------------------------------------------------------------------------- #


def test_fetch_genbank_requires_email(tmp_path: Path) -> None:
    with pytest.raises(StageError):
        fetch_genbank({}, tmp_path / "a.fasta", tmp_path / "a.tsv")


def test_fetch_genbank_requires_query(tmp_path: Path) -> None:
    config = {"datasets": {"hbv": {"genbank": {"email": "a@b.org", "query": None}}}}
    with pytest.raises(StageError):
        fetch_genbank(config, tmp_path / "a.fasta", tmp_path / "a.tsv")


# --------------------------------------------------------------------------- #
# HBVdb release freshness
# --------------------------------------------------------------------------- #


def test_check_release_freshness_recent() -> None:
    config = {
        "datasets": {
            "hbv": {
                "hbvdb": {
                    "release": date.today().strftime("%Y-%m-%d"),
                    "max_release_age_days": 730,
                }
            }
        }
    }
    fresh, message = check_release_freshness(config)
    assert fresh is True
    assert "0 days old" in message


def test_check_release_freshness_stale_and_undeclared() -> None:
    stale = {
        "datasets": {
            "hbv": {
                "hbvdb": {
                    "release": (date.today() - timedelta(days=900)).strftime("%Y-%m-%d"),
                    "max_release_age_days": 730,
                }
            }
        }
    }
    assert check_release_freshness(stale)[0] is False
    assert check_release_freshness({})[0] is False


# --------------------------------------------------------------------------- #
# stage orchestration
# --------------------------------------------------------------------------- #


def _offline_config(output_root: Path) -> dict:
    return {
        "project": {"output_root": str(output_root)},
        "datasets": {"hbv": {"sources": ["genbank"]}},  # no email -> GenBank is skipped
        "deephep": {
            "taxonomic_groups": [],
            "include_nackednavirus": False,
            "include_rt_outgroups": False,
            "aligner": "none",
            "min_pol_length": 500,
            "max_pol_length": 1200,
        },
    }


def test_run_offline_returns_dict_without_raising(tmp_path: Path) -> None:
    artefacts = run(_offline_config(tmp_path / "out"), tmp_path)
    assert isinstance(artefacts, dict)
    # The deep-hepadnavirus outputs are always materialised (possibly empty).
    assert artefacts["deephep_pol"].exists()
    assert artefacts["deephep_alignment"].exists()
    assert all(isinstance(path, Path) for path in artefacts.values())


def test_run_offline_with_no_sources(tmp_path: Path) -> None:
    config = _offline_config(tmp_path / "out")
    config["datasets"]["hbv"]["sources"] = []
    artefacts = run(config, tmp_path)
    assert isinstance(artefacts, dict)
    assert "deephep_pol" in artefacts
