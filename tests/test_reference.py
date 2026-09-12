"""Offline tests for reference-feature derivation and config wiring."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hbvpol.config import get, load_config  # noqa: E402
from hbvpol.reference import (  # noqa: E402
    ReferenceFeatures,
    derive_features,
    features_to_config,
    load_features,
    load_reference_record,
    write_features,
)

# 100 bp, Pol annotated as a two-part join across the origin (70..100,1..30),
# surface at 5..40, core at 45..68.
GENBANK = """LOCUS       TESTREF                 100 bp    DNA     circular
DEFINITION  synthetic test reference.
ACCESSION   TESTREF
FEATURES             Location/Qualifiers
     source          1..100
                     /organism="Hepatitis B virus"
     CDS             join(70..100,1..30)
                     /gene="P"
                     /product="polymerase"
     CDS             5..40
                     /gene="S"
                     /product="surface"
     CDS             45..68
                     /gene="C"
                     /product="core"
ORIGIN
        1 atgcatgcat gcatgcatgc atgcatgcat gcatgcatgc atgcatgcat gcatgcatgc
       61 atgcatgcat gcatgcatgc atgcatgcat gcatgcatgc
//
"""


@pytest.fixture()
def record(tmp_path):
    pytest.importorskip("Bio")
    path = tmp_path / "ref.gb"
    path.write_text(GENBANK, encoding="utf-8")
    return load_reference_record(path)


def test_derive_features_from_annotated_record(record):
    features = derive_features(record, origin_nt=1)
    assert features.length == 100
    assert (features.pol_start_nt, features.pol_end_nt) == (70, 30)
    assert (features.s_start_nt, features.s_end_nt) == (5, 40)
    assert (features.c_start_nt, features.c_end_nt) == (45, 68)
    # The shipped default is +1; the true offset for these coordinates is 1.
    assert features.surface_frame_offset == (5 - 70) % 3
    # Origin-wrapping span (31 + 30) / 3 = 20 codons.
    assert features.pol_length_aa == 20


def test_derive_features_requires_pol_and_surface(tmp_path):
    pytest.importorskip("Bio")
    from hbvpol.pipeline import StageError

    # Only a Pol CDS: derivation must fail loudly rather than guess a frame.
    text = (
        "LOCUS       TESTREF                 100 bp    DNA     circular\n"
        "FEATURES             Location/Qualifiers\n"
        "     CDS             join(70..100,1..30)\n"
        '                     /gene="P"\n'
        "ORIGIN\n"
        "        1 atgcatgcat gcatgcatgc atgcatgcat gcatgcatgc atgcatgcat gcatgcatgc\n"
        "       61 atgcatgcat gcatgcatgc atgcatgcat gcatgcatgc\n"
        "//\n"
    )
    path = tmp_path / "incomplete.gb"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(StageError):
        derive_features(load_reference_record(path))


def test_features_to_config_maps_offsets():
    features = ReferenceFeatures(
        accession="TESTREF", length=100, origin_nt=1,
        pol_start_nt=70, pol_end_nt=30, pol_length_aa=20,
        s_start_nt=5, s_end_nt=40, c_start_nt=45, c_end_nt=68,
        surface_frame_offset=1,
    )
    config = features_to_config(features)
    assert config["pol_start_nt"] == 70
    assert config["pol_end_nt"] == 30
    assert config["surface_frame_offset"] == 1
    assert "sequence" not in config  # no path supplied


def test_write_load_roundtrip(tmp_path):
    features = ReferenceFeatures(
        accession="TESTREF", length=100, origin_nt=1,
        pol_start_nt=70, pol_end_nt=30, pol_length_aa=20,
        s_start_nt=5, s_end_nt=40, c_start_nt=45, c_end_nt=68,
        surface_frame_offset=1,
    )
    path = write_features(features, tmp_path / "reference_features.json")
    assert path.exists()
    assert load_features(path) == features
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["reference"]["pol_start_nt"] == 70


def test_load_config_merges_derived_features(tmp_path):
    """The derived JSON must override the shipped constants, not vice versa."""
    output_root = tmp_path / "output"
    (output_root / "reference").mkdir(parents=True)
    features = ReferenceFeatures(
        accession="TESTREF", length=100, origin_nt=1,
        pol_start_nt=70, pol_end_nt=30, pol_length_aa=20,
        s_start_nt=5, s_end_nt=40, c_start_nt=45, c_end_nt=68,
        surface_frame_offset=1,
    )
    write_features(features, output_root / "reference" / "reference_features.json")

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "project:\n  output_root: output\n"
        "reference:\n  auto_derive: true\n  pol_start_nt: 2307\n  surface_frame_offset: 1\n",
        encoding="utf-8",
    )
    config = load_config(config_path, root=tmp_path)
    assert get(config, "reference.pol_start_nt") == 70
    assert get(config, "reference.s_start_nt") == 5

    # Explicit overrides still win over derived features.
    overridden = load_config(config_path, overrides=["reference.pol_start_nt=999"], root=tmp_path)
    assert get(overridden, "reference.pol_start_nt") == 999

    # auto_derive: false leaves the config file's constants untouched.
    config_path.write_text(
        "project:\n  output_root: output\n"
        "reference:\n  auto_derive: false\n  pol_start_nt: 2307\n",
        encoding="utf-8",
    )
    raw = load_config(config_path, root=tmp_path)
    assert get(raw, "reference.pol_start_nt") == 2307


def test_domain_spans_from_config_clamps_and_overrides():
    from hbvpol.domain import PolDomain, domain_spans_from_config

    # Clamped to the derived Pol length (RNaseH must not spill past the end).
    clamped = domain_spans_from_config({"reference": {"pol_length_aa": 800}})
    assert clamped[-1].end == 800

    # Explicit spans win and are parsed from the config list.
    custom = domain_spans_from_config({
        "reference": {
            "domain_spans": [
                {"domain": "TP", "start": 1, "end": 100},
                {"domain": "RT", "start": 101, "end": 700},
            ]
        }
    })
    assert [span.domain for span in custom] == [PolDomain.TP, PolDomain.RT]
