"""Offline unit tests for the structure package."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hbvpol.io import GenomeRecord, write_fasta
from hbvpol.structure.metrics import candidate_hinges, compute_metrics, interface_residues, plddt_from_pdb
from hbvpol.structure.models import build_model_manifest, sequence_for_lineage


# --------------------------------------------------------------------------- #
# pure hinge calling
# --------------------------------------------------------------------------- #
def test_candidate_hinges_finds_contiguous_low_regions():
    plddt = np.array([90, 88, 40, 35, 92, 91, 60, 61, 62, 90], dtype=float)
    assert candidate_hinges(plddt, threshold=70) == [(3, 4), (7, 9)]


def test_candidate_hinges_respects_threshold_and_min_length():
    plddt = np.array([80, 65, 64, 80], dtype=float)
    assert candidate_hinges(plddt, threshold=70, min_length=2) == [(2, 3)]
    assert candidate_hinges(plddt, threshold=70, min_length=3) == []


def test_candidate_hinges_treats_nan_as_assessed():
    plddt = np.array([80.0, np.nan, 60.0], dtype=float)
    assert candidate_hinges(plddt, threshold=70) == [(3, 3)]
    assert candidate_hinges(np.array([]), threshold=70) == []


# --------------------------------------------------------------------------- #
# a tiny hand-written PDB
# --------------------------------------------------------------------------- #
def _atom_line(serial, name, resname, chain, resseq, xyz, b, element):
    x, y, z = xyz
    return (
        f"ATOM  {serial:5d} {name:<4} {resname:>3} {chain}{resseq:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}{1.00:6.2f}{b:6.2f}          {element:>2}\n"
    )


def _write_pdb(path: Path, plddt) -> Path:
    atoms = [
        ("N", 0.0, 0.0, 0.0, "N"),
        ("CA", 0.0, 1.5, 0.0, "C"),
        ("C", 0.0, 2.5, 0.0, "C"),
        ("O", 0.0, 3.2, 0.0, "O"),
        ("CB", 0.0, 2.0, 1.5, "C"),
    ]
    lines = []
    serial = 1
    for index, value in enumerate(plddt, start=1):
        base_x = index * 3.8
        for name, dx, dy, dz, element in atoms:
            lines.append(
                _atom_line(serial, name, "ALA", "A", index,
                           (base_x + dx, dy, dz), value, element)
            )
            serial += 1
    lines.append("END\n")
    path.write_text("".join(lines), encoding="utf-8")
    return path


def test_plddt_from_pdb_matches_b_factors(tmp_path):
    pytest.importorskip("Bio.PDB")
    values = [90.0, 88.0, 40.0, 35.0, 92.0, 91.0]
    path = _write_pdb(tmp_path / "model.pdb", values)
    parsed = plddt_from_pdb(path)
    assert parsed.shape == (6,)
    assert np.allclose(parsed, values)


def test_compute_metrics_on_tiny_pdb(tmp_path):
    pytest.importorskip("Bio.PDB")
    values = [90.0, 88.0, 40.0, 35.0, 92.0, 91.0]
    path = _write_pdb(tmp_path / "model.pdb", values)

    record = compute_metrics(path, {"structure": {"hinge_plddt": 70}})

    assert record["status"] == "ok"
    assert record["n_residues"] == 6
    assert record["plddt_mean"] == pytest.approx(float(np.mean(values)))
    assert record["plddt_min"] == pytest.approx(35.0)
    # No YMDD motif is present in a six-residue model.
    assert record["catalytic_distance"] is None
    assert record["nucleic_has_chain"] is False


def test_compute_metrics_is_graceful_on_missing_file(tmp_path):
    pytest.importorskip("Bio.PDB")
    record = compute_metrics(tmp_path / "does_not_exist.pdb", {})
    assert record["status"] == "error"
    assert "error_plddt" in record


# --------------------------------------------------------------------------- #
# manifest and lineage sequences
# --------------------------------------------------------------------------- #
def test_build_model_manifest_enumerates_ensemble():
    config = {
        "structure": {
            "states": ["apo", "eps_bound"],
            "lineages": ["A", "B"],
            "strategies": ["colabfold"],
            "seeds": 2,
        }
    }
    manifest = build_model_manifest(config)
    assert len(manifest) == 8
    assert set(manifest["state"]) == {"apo", "eps_bound"}
    assert set(manifest["model_id"]) == {
        "apo__A__colabfold__seed1", "apo__A__colabfold__seed2",
        "apo__B__colabfold__seed1", "apo__B__colabfold__seed2",
        "eps_bound__A__colabfold__seed1", "eps_bound__A__colabfold__seed2",
        "eps_bound__B__colabfold__seed1", "eps_bound__B__colabfold__seed2",
    }


def test_build_model_manifest_includes_mutants(tmp_path):
    mutants = tmp_path / "resources" / "mutants.tsv"
    mutants.parent.mkdir(parents=True)
    pd.DataFrame({"mutant": ["rtA181T"], "lineage": ["A"]}).to_csv(mutants, sep="\t", index=False)
    config = {
        "structure": {
            "states": ["apo"],
            "lineages": ["A"],
            "strategies": ["colabfold"],
            "seeds": 1,
            "mutants_file": "resources/mutants.tsv",
        }
    }
    manifest = build_model_manifest(config, tmp_path)
    assert len(manifest) == 2
    assert "mutant__A__rtA181T__seed1" in set(manifest["model_id"])


def test_sequence_for_lineage_reads_ancestral_file(tmp_path):
    ancestral_dir = tmp_path / "output" / "phylogeny" / "ancestral"
    ancestral_dir.mkdir(parents=True)
    write_fasta(
        [GenomeRecord(id="A", seq="ATGGCC" * 10)],
        ancestral_dir / "ancestral_A.fasta",
    )
    config = {"project": {"output_root": "output"}}
    record = sequence_for_lineage("A", config, tmp_path)
    assert record is not None
    assert record.seq == "ATGGCC" * 10


def test_sequence_for_lineage_returns_none_when_absent(tmp_path, caplog):
    config = {"project": {"output_root": "output"}}
    assert sequence_for_lineage("Z", config, tmp_path) is None


# --------------------------------------------------------------------------- #
# nucleic-acid interface residues
# --------------------------------------------------------------------------- #
def test_interface_residues_measures_distance_to_nucleic_chain(tmp_path):
    pytest.importorskip("Bio.PDB")
    path = _write_pdb(tmp_path / "model.pdb", [90.0, 90.0, 90.0])
    # Place a nucleic atom next to residue 2's backbone (residues are 3.8 A apart).
    with path.open("a", encoding="utf-8") as handle:
        handle.write(_atom_line(100, "P", "DA", "N", 1, (7.6, 3.5, 0.0), 0.0, "P"))
        handle.write("TER\n")

    distances = interface_residues(path)

    assert distances.shape == (3,)
    assert distances[1] < 1.0
    assert distances[0] > 3.0
    assert distances[2] > 3.0


def test_interface_residues_empty_without_nucleic_chain(tmp_path):
    pytest.importorskip("Bio.PDB")
    path = _write_pdb(tmp_path / "model.pdb", [90.0, 90.0])
    assert interface_residues(path).size == 0
