"""Tests for the OpenMM MD runner (plan, force-field mapping, CLI surface)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hbvpol.structure.md import _command_plan  # noqa: E402
from hbvpol.structure.openmm_md import (  # noqa: E402
    _select_platform,
    forcefield_files,
    openmm_command_plan,
)


def test_forcefield_files_maps_names():
    assert forcefield_files("amber14sb", "tip3p") == ["amber14-all.xml", "amber14/tip3pfb.xml"]
    assert forcefield_files("charmm36", "tip3p") == ["charmm36.xml", "amber14/tip3pfb.xml"]
    # An unknown water model is dropped rather than guessed.
    assert forcefield_files("amber14sb", "unknownmodel") == ["amber14-all.xml"]


def test_forcefield_files_explicit_override():
    config = {"structure": {"md": {"openmm": {"forcefield_files": ["a.xml", "b.xml"]}}}}
    assert forcefield_files("amber14sb", "tip3p", config) == ["a.xml", "b.xml"]


def test_openmm_command_plan_points_at_the_runner():
    plan = openmm_command_plan("models/m.pdb", "m1", {})
    assert "scripts/run_md_openmm.py" in plan
    assert "--pdb models/m.pdb" in plan
    assert "trajectory.dcd" in plan


def test_md_manifest_plan_uses_the_openmm_runner_for_openmm_engine():
    config = {"structure": {"md": {"engine": "openmm"}}}
    plan = _command_plan("models/m.pdb", "m1", config, 8)
    assert "scripts/run_md_openmm.py" in plan


def test_md_manifest_plan_uses_gromacs_for_gromacs_engine():
    config = {"structure": {"md": {"engine": "gromacs"}}}
    plan = _command_plan("models/m.pdb", "m1", config, 8)
    assert plan.startswith("gmx pdb2gmx")


def test_select_platform_returns_an_available_platform():
    pytest.importorskip("openmm")
    platform = _select_platform("auto")
    assert platform is not None
    assert platform.getName() in {"CUDA", "OpenCL", "CPU", "Reference"}
