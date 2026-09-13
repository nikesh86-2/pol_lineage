"""Molecular-dynamics planning and cavity-persistence accounting.

This module does not run MD.  It (a) selects a small, prioritised subset of the
structural ensemble for simulation and emits an explicit, inspectable command
plan, and (b) turns either an MDAnalysis trajectory or a supplied cavity
catalogue into the ``pocket_persistence.tsv`` contract.

Design note
-----------
A pocket is only a credible drug target once its persistence has been
*demonstrated* across a trajectory.  ``pocket_persistence`` therefore reports
the fraction of frames in which each cavity is present and flags those passing
``structure.md.pocket_persistence_frac``; a single-frame or
catalogue-only estimate is explicitly marked ``demonstrated=False``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..config import get
from ..io import read_table
from ..pipeline import get_logger
from .openmm_md import openmm_command_plan

__all__ = [
    "MD_MANIFEST_COLUMNS",
    "CAVITY_COLUMNS",
    "build_md_manifest",
    "pocket_persistence",
]

logger = get_logger("structure.md")

MD_MANIFEST_COLUMNS = [
    "model_id",
    "kind",
    "state",
    "lineage",
    "priority",
    "engine",
    "forcefield",
    "water",
    "ns_per_model",
    "workdir",
    "commands",
]

CAVITY_COLUMNS = [
    "model_id",
    "cavity_id",
    "n_frames",
    "n_frames_present",
    "persistence",
    "threshold",
    "passes",
    "demonstrated",
]


def _empty_md_manifest() -> pd.DataFrame:
    return pd.DataFrame(columns=MD_MANIFEST_COLUMNS)


def _empty_cavities() -> pd.DataFrame:
    return pd.DataFrame(columns=CAVITY_COLUMNS)


def _priority(row: pd.Series, config: dict) -> int:
    """Lower is more important: mutants, then apo, then bound states."""
    override = get(config, "structure.md.priority")
    if override:
        order = {str(k): int(v) for k, v in dict(override).items()}
        for key in (str(row.get("state")), str(row.get("kind"))):
            if key in order:
                return order[key]
    kind = str(row.get("kind", ""))
    state = str(row.get("state", ""))
    if kind == "mutant":
        return 0
    if state == "apo":
        return 1
    return 2


def _command_plan(model_path: str, model_id: str, config: dict, threads: int) -> str:
    engine = str(get(config, "structure.md.engine", "gromacs") or "gromacs").lower()
    forcefield = str(get(config, "structure.md.forcefield", "amber14sb") or "amber14sb")
    water = str(get(config, "structure.md.water", "tip3p") or "tip3p")
    if engine == "openmm":
        return openmm_command_plan(model_path, model_id, config)
    return "\n".join([
        f"gmx pdb2gmx -f {model_path} -o {model_id}.gro -ff {forcefield} -water {water}",
        f"gmx editconf -f {model_id}.gro -o {model_id}_box.gro -c -d 1.0 -bt cubic",
        f"gmx solvate -cp {model_id}_box.gro -cs spc216.gro -o {model_id}_solv.gro -p topol.top",
        f"gmx grompp -f ions.mdp -c {model_id}_solv.gro -o ions.tpr -p topol.top",
        f"gmx genion -s ions.tpr -o {model_id}_ions.gro -p topol.top -neutral -conc 0.15",
        f"gmx grompp -f md.mdp -c {model_id}_ions.gro -o {model_id}.tpr -p topol.top",
        f"gmx mdrun -deffnm {model_id} -nt {threads}",
    ])


def build_md_manifest(models, config: dict) -> pd.DataFrame:
    """Select the top ``structure.md.n_models`` models and plan their MD runs."""
    if models is None or len(models) == 0:
        return _empty_md_manifest()
    frame = pd.DataFrame(models).copy()
    for column in ("model_id", "kind", "state", "lineage", "rel_path"):
        if column not in frame.columns:
            frame[column] = ""
    n_models = int(get(config, "structure.md.n_models", 20) or 20)
    threads = int(get(config, "project.threads", 1) or 1)
    frame["priority"] = frame.apply(lambda row: _priority(row, config), axis=1)
    frame = frame.sort_values(["priority", "model_id"]).head(n_models).reset_index(drop=True)

    engine = str(get(config, "structure.md.engine", "gromacs") or "gromacs")
    forcefield = str(get(config, "structure.md.forcefield", "amber14sb") or "amber14sb")
    water = str(get(config, "structure.md.water", "tip3p") or "tip3p")
    ns_per_model = float(get(config, "structure.md.ns_per_model", 100) or 100)
    workroot = Path(str(get(config, "structure.md.workdir", "md")))
    rows = []
    for _, row in frame.iterrows():
        model_id = str(row["model_id"])
        rel_path = str(row.get("rel_path", ""))
        rows.append({
            "model_id": model_id,
            "kind": row.get("kind", ""),
            "state": row.get("state", ""),
            "lineage": row.get("lineage", ""),
            "priority": int(row["priority"]),
            "engine": engine,
            "forcefield": forcefield,
            "water": water,
            "ns_per_model": ns_per_model,
            "workdir": str(workroot / model_id),
            "commands": _command_plan(rel_path, model_id, config, threads),
        })
    return pd.DataFrame(rows, columns=MD_MANIFEST_COLUMNS)


def _catalogue_persistence(catalogue: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Aggregate a per-frame cavity catalogue into per-cavity persistence."""
    required = {"model_id", "cavity_id", "frame", "present"}
    if not required.issubset(catalogue.columns):
        logger.warning(
            "cavity catalogue is missing columns %s; cannot compute persistence",
            sorted(required - set(catalogue.columns)),
        )
        return _empty_cavities()
    rows = []
    for (model_id, cavity_id), group in catalogue.groupby(["model_id", "cavity_id"], dropna=False):
        present = pd.to_numeric(group["present"], errors="coerce").fillna(0.0)
        n_frames = int(len(group))
        n_present = int((present > 0).sum())
        fraction = n_present / n_frames if n_frames else 0.0
        rows.append({
            "model_id": model_id,
            "cavity_id": cavity_id,
            "n_frames": n_frames,
            "n_frames_present": n_present,
            "persistence": fraction,
            "threshold": threshold,
            "passes": bool(fraction >= threshold),
            "demonstrated": False,
        })
    return pd.DataFrame(rows, columns=CAVITY_COLUMNS)


def _mdanalysis_persistence(trajectory_dir: Path, threshold: float, config: dict) -> pd.DataFrame:
    """Estimate cavity persistence from a real trajectory via MDAnalysis.

    The cavity is defined by ``structure.md.pocket_residues`` (1-based residue
    ids).  A frame counts as "cavity present" when the centroid of those
    residues is within ``structure.md.pocket_radius`` of the origin of the
    selected pocket sphere — a deliberately simple, transparent proxy that can
    be replaced by a proper cavity detector (e.g. fpocket/MDPocket) without
    changing the output schema.
    """
    try:
        import MDAnalysis as mda  # type: ignore
    except Exception:
        return _empty_cavities()

    topologies = [p for p in sorted(trajectory_dir.glob("*")) if p.suffix.lower() in {".pdb", ".gro", ".psf", ".tpr"}]
    trajectories = [p for p in sorted(trajectory_dir.glob("*")) if p.suffix.lower() in {".xtc", ".dcd", ".trr"}]
    if not topologies or not trajectories:
        return _empty_cavities()
    pocket_residues = list(get(config, "structure.md.pocket_residues", []) or [])
    if not pocket_residues:
        logger.warning("structure.md.pocket_residues not configured; skipping MDAnalysis persistence")
        return _empty_cavities()
    radius = float(get(config, "structure.md.pocket_radius", 6.0) or 6.0)

    try:
        universe = mda.Universe(str(topologies[0]), str(trajectories[0]))
    except Exception as error:  # pragma: no cover - environment dependent
        logger.warning("MDAnalysis could not open trajectory: %s", error)
        return _empty_cavities()

    selection = universe.select_atoms("resid " + " ".join(str(r) for r in pocket_residues))
    if selection.n_atoms == 0:
        logger.warning("no atoms matched pocket residues %s", pocket_residues)
        return _empty_cavities()
    centre = selection.center_of_mass()
    present = 0
    n_frames = 0
    for _ in universe.trajectory:
        n_frames += 1
        distance = float(np.linalg.norm(selection.center_of_mass() - centre))
        if distance <= radius:
            present += 1
    fraction = present / n_frames if n_frames else 0.0
    model_id = trajectory_dir.name
    return pd.DataFrame([{
        "model_id": model_id,
        "cavity_id": "pocket_residues",
        "n_frames": n_frames,
        "n_frames_present": present,
        "persistence": fraction,
        "threshold": threshold,
        "passes": bool(fraction >= threshold),
        "demonstrated": True,
    }], columns=CAVITY_COLUMNS)


def pocket_persistence(trajectory_dir, config: dict) -> pd.DataFrame:
    """Fraction of frames each cavity is present, with a pass/fail flag.

    Priority: an MDAnalysis trajectory (``demonstrated=True``) if one is present
    and usable, else a supplied per-frame cavity catalogue, else an empty but
    correctly-schemad frame.
    """
    threshold = float(get(config, "structure.md.pocket_persistence_frac", 0.5) or 0.5)
    directory = Path(trajectory_dir) if trajectory_dir is not None else Path()
    if not directory or not directory.is_dir():
        return _empty_cavities()

    trajectory = _mdanalysis_persistence(directory, threshold, config)
    if len(trajectory):
        return trajectory

    catalogue_path = directory / "cavity_catalogue.tsv"
    if not catalogue_path.exists():
        for candidate in sorted(directory.glob("*cavit*")):
            catalogue_path = candidate
            break
    if catalogue_path.exists():
        try:
            catalogue = read_table(catalogue_path)
        except Exception as error:  # pragma: no cover - defensive
            logger.warning("could not read cavity catalogue %s: %s", catalogue_path, error)
            return _empty_cavities()
        return _catalogue_persistence(catalogue, threshold)

    logger.warning(
        "no trajectory or cavity catalogue under %s; emitting empty persistence table",
        directory,
    )
    return _empty_cavities()
