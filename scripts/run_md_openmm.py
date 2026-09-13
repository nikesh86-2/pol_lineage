#!/usr/bin/env python3
"""run_md_openmm.py -- run the MD protocol with OpenMM (GROMACS replacement).

The structure stage plans MD and analyses the resulting trajectory with
MDAnalysis; this script produces that trajectory with OpenMM:
add hydrogens -> solvate -> add ions -> minimise -> equilibrate -> production,
writing ``topology.pdb`` + ``trajectory.dcd`` into the directory
``pocket_persistence`` reads (default ``<output_root>/structure/md/trajectories``).

Typical use::

    python scripts/run_md_openmm.py \\
        --pdb output/structure/models/apo/D/colabfold_seed0.pdb --model-id apo_D

Then re-run the structure stage so the persistence table is recomputed::

    hbvpol structure -c config/config.yaml

Everything is configured under ``structure.md`` / ``structure.md.openmm`` (force
field, water, platform, padding, ionic strength, temperature, timestep,
equilibration and production length).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hbvpol.config import get, load_config  # noqa: E402
from hbvpol.pipeline import output_dir  # noqa: E402
from hbvpol.structure.openmm_md import run_openmm_md  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--pdb", required=True, help="Input model PDB (one of the predicted ensemble)")
    parser.add_argument("--model-id", default=None, help="Model label for logs (default: PDB stem)")
    parser.add_argument("-c", "--config", default="config/config.yaml", help="Pipeline config YAML")
    parser.add_argument("--root", default=".", help="Project root for relative paths")
    parser.add_argument("--outdir", default=None,
                        help="Trajectory directory (default: structure.md.trajectory_dir or <outroot>/structure/md/trajectories)")
    parser.add_argument("--ns", type=float, default=None, help="Override structure.md.ns_per_model (nanoseconds)")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    config = load_config(root / args.config, root=root)
    if args.ns is not None:
        config.setdefault("structure", {}).setdefault("md", {})["ns_per_model"] = args.ns

    if args.outdir:
        outdir = Path(args.outdir)
    else:
        configured = get(config, "structure.md.trajectory_dir", None)
        outdir = Path(configured) if configured else output_dir(config, root) / "structure" / "md" / "trajectories"

    model_id = args.model_id or Path(args.pdb).stem
    print(f"OpenMM MD: model={model_id} pdb={args.pdb} outdir={outdir}")
    artefacts = run_openmm_md(Path(args.pdb), outdir, config)
    for name, path in artefacts.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
