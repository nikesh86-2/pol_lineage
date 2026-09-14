"""OpenMM molecular-dynamics runner (drop-in for GROMACS).

The structure stage only *plans* MD (:func:`hbvpol.structure.md.build_md_manifest`)
and *analyses* a trajectory through MDAnalysis
(:func:`hbvpol.structure.md.pocket_persistence`).  The engine is therefore
interchangeable as long as it writes a topology plus a trajectory into the
directory the analysis reads (default ``<outroot>/structure/md/trajectories/``).

This module runs the GROMACS-equivalent protocol with OpenMM -- add hydrogens,
solvate, add ions, minimise, NVT/NPT equilibrate, production -- and writes
``topology.pdb`` and ``trajectory.dcd`` so ``pocket_persistence`` picks them up
alongside a GROMACS ``.gro``/``.xtc`` pair.  OpenMM is imported lazily, so no
other stage depends on it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from ..config import get
from ..pipeline import get_logger

__all__ = ["openmm_command_plan", "forcefield_files", "run_openmm_md"]

logger = get_logger("structure.openmm")

#: Amber-14 water model XML files keyed by the config ``water`` name.
_WATER_FILES = {
    "tip3p": "amber14/tip3pfb.xml",
    "tip3pfb": "amber14/tip3pfb.xml",
    "tip4pew": "amber14/tip4pew.xml",
    "tip4p": "amber14/tip4pew.xml",
    "spce": "amber14/spce.xml",
}

#: Protein force-field XML files keyed by the config ``forcefield`` name.
_FORCEFIELD_FILES = {
    "amber14sb": "amber14-all.xml",
    "amber14-all": "amber14-all.xml",
    "amber99sb": "amber99sbildn.xml",
    "charmm36": "charmm36.xml",
    "charmm36m": "charmm36.xml",
}


def forcefield_files(forcefield: str, water: str, config: Mapping[str, object] | None = None):
    """Resolve the OpenMM force-field XML files for the configured names.

    An explicit ``structure.md.openmm.forcefield_files`` list wins, so an
    unusual force field does not need a code change.
    """
    if config is not None:
        explicit = get(config, "structure.md.openmm.forcefield_files", None)
        if explicit:
            return [str(name) for name in explicit]
    protein = _FORCEFIELD_FILES.get(str(forcefield).strip().lower(), str(forcefield))
    solvent = _WATER_FILES.get(str(water).strip().lower())
    return [protein, solvent] if solvent else [protein]


def _select_platform(name: str | None):
    """Return an OpenMM platform, preferring CUDA/OpenCL and falling back to CPU."""
    from openmm import Platform

    requested = str(name or "auto").strip()
    if requested and requested.lower() != "auto":
        try:
            return Platform.getPlatformByName(requested)
        except Exception as error:  # pragma: no cover - environment dependent
            logger.warning("OpenMM platform %r unavailable (%s); falling back", requested, error)
    for candidate in ("CUDA", "OpenCL", "CPU", "Reference"):
        try:
            return Platform.getPlatformByName(candidate)
        except Exception:
            continue
    return None  # pragma: no cover - OpenMM always ships Reference


def openmm_command_plan(pdb_path: str, model_id: str, config: Mapping[str, object]) -> str:
    """The manifest command text for an OpenMM run."""
    return "\n".join([
        f"# OpenMM protocol for {model_id}: add H -> solvate -> ions -> minimise -> equilibrate -> production",
        "python scripts/run_md_openmm.py "
        f"--pdb {pdb_path} --model-id {model_id} -c config/config.yaml",
        "# writes topology.pdb + trajectory.dcd under <outroot>/structure/md/trajectories/",
    ])


def run_openmm_md(pdb_path, outdir, config: Mapping[str, object] | None = None) -> dict[str, Path]:
    """Run a short OpenMM protocol and write ``topology.pdb`` + ``trajectory.dcd``.

    Parameters mirror ``structure.md.*`` so this is a drop-in for the GROMACS plan.
    The ensemble is built at 300 K with a Monte-Carlo barostat (NPT); production
    length is ``structure.md.ns_per_model`` nanoseconds at
    ``structure.md.openmm.timestep_fs``.  Returns a mapping with the written
    ``topology``, ``trajectory`` and ``log`` paths.
    """
    try:
        from openmm import LangevinMiddleIntegrator, MonteCarloBarostat, unit
        from openmm.app import (
            DCDReporter,
            ForceField,
            HBonds,
            Modeller,
            PDBFile,
            PME,
            Simulation,
            StateDataReporter,
        )
    except Exception as error:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "OpenMM is required to run the MD protocol; install it with "
            "`conda install -c conda-forge openmm`"
        ) from error

    pdb_path = Path(pdb_path)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    forcefield = str(get(config, "structure.md.forcefield", "amber14sb") or "amber14sb")
    water = str(get(config, "structure.md.water", "tip3p") or "tip3p")
    files = forcefield_files(forcefield, water, config)
    platform_name = get(config, "structure.md.openmm.platform", "auto")
    padding_nm = float(get(config, "structure.md.openmm.padding_nm", 1.0) or 1.0)
    ionic = float(get(config, "structure.md.openmm.ionic_strength_molar", 0.15) or 0.0)
    temperature = float(get(config, "structure.md.openmm.temperature_k", 300) or 300)
    friction = float(get(config, "structure.md.openmm.friction_per_ps", 1.0) or 1.0)
    timestep_fs = float(get(config, "structure.md.openmm.timestep_fs", 2.0) or 2.0)
    equilibration_ps = float(get(config, "structure.md.openmm.equilibration_ps", 100) or 0.0)
    ns_per_model = float(get(config, "structure.md.ns_per_model", 100) or 0.0)
    report_interval = int(get(config, "structure.md.openmm.report_interval", 5000) or 5000)
    max_iterations = int(get(config, "structure.md.openmm.minimize_max_iterations", 1000) or 1000)

    logger.info(
        "OpenMM MD: %s (files=%s, water=%s, %.1f ns, %.1f fs)",
        pdb_path.name, ",".join(str(f) for f in files), water, ns_per_model, timestep_fs,
    )
    pdb = PDBFile(str(pdb_path))
    field = ForceField(*[str(name) for name in files])
    modeller = Modeller(pdb.topology, pdb.positions)
    modeller.addHydrogens(field, pH=7.0)
    modeller.addSolvent(
        field,
        model=water,
        padding=padding_nm * unit.nanometer,
        ionicStrength=ionic * unit.molar,
        neutralize=True,
    )

    system = field.createSystem(
        modeller.topology,
        nonbondedMethod=PME,
        nonbondedCutoff=1.0 * unit.nanometer,
        constraints=HBonds,
    )
    # Pin the integrator/barostat RNG so a trajectory is reproducible.
    seed = int(get(config, "project.seed", 1) or 1)
    barostat = MonteCarloBarostat(1.0 * unit.bar, temperature * unit.kelvin)
    barostat.setRandomNumberSeed(seed)
    system.addForce(barostat)
    integrator = LangevinMiddleIntegrator(
        temperature * unit.kelvin,
        friction / unit.picosecond,
        timestep_fs * unit.femtoseconds,
    )
    integrator.setRandomNumberSeed(seed)
    platform = _select_platform(platform_name)
    simulation = Simulation(
        modeller.topology, system, integrator, platform
    ) if platform is not None else Simulation(modeller.topology, system, integrator)
    simulation.context.setPositions(modeller.positions)

    log_path = outdir / "openmm_md.log"
    simulation.minimizeEnergy(maxIterations=max_iterations)

    equilibration_steps = int(round(equilibration_ps * 1000.0 / timestep_fs))
    if equilibration_steps > 0:
        simulation.step(equilibration_steps)

    topology_path = outdir / "topology.pdb"
    with topology_path.open("w", encoding="utf-8") as handle:
        PDBFile.writeFile(
            simulation.topology,
            simulation.context.getState(getPositions=True).getPositions(),
            handle,
        )

    trajectory_path = outdir / "trajectory.dcd"
    simulation.reporters.append(DCDReporter(str(trajectory_path), max(1, report_interval)))
    simulation.reporters.append(
        StateDataReporter(
            str(log_path), max(1, report_interval),
            step=True, potentialEnergy=True, temperature=True, density=True,
            speed=True, remainingTime=True, totalSteps=int(
                round(ns_per_model * 1_000_000.0 / timestep_fs)
            ),
        )
    )
    production_steps = int(round(ns_per_model * 1_000_000.0 / timestep_fs))
    if production_steps > 0:
        simulation.step(production_steps)

    final_path = outdir / "final.pdb"
    with final_path.open("w", encoding="utf-8") as handle:
        PDBFile.writeFile(
            simulation.topology,
            simulation.context.getState(getPositions=True).getPositions(),
            handle,
        )

    return {
        "topology": topology_path,
        "trajectory": trajectory_path,
        "log": log_path,
        "final": final_path,
    }
