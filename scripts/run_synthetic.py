#!/usr/bin/env python
"""Fabricate a synthetic dataset and run the full offline pipeline.

Example::

    python scripts/run_synthetic.py --outdir output-synthetic --genomes 8

This needs no network and no external bioinformatics programs: recombination
falls back to the pure-Python bootscan, trees to Neighbor-Joining, ancestral
reconstruction to parsimony and epsilon folding to a Nussinov fold.  It is the
fastest way to confirm an installation and to see the expected output tree.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from hbvpol.synthetic import build_synthetic, run_synthetic_pipeline, synthetic_config  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default=str(ROOT / "output-synthetic"),
                        help="output root to populate (created if needed)")
    parser.add_argument("--genomes", type=int, default=8, help="number of synthetic genomes")
    parser.add_argument("--seed", type=int, default=20240911, help="RNG seed")
    parser.add_argument("--config", default=str(ROOT / "config" / "config.yaml"))
    parser.add_argument("--data-only", action="store_true",
                        help="only write the synthetic dataset, do not run stages")
    args = parser.parse_args(argv)

    outdir = Path(args.outdir).resolve()
    artefacts = build_synthetic(outdir, n_genomes=args.genomes, seed=args.seed)
    print(f"wrote synthetic dataset under {outdir / 'datasets'}:")
    for name, path in artefacts.items():
        print(f"  {name}: {path}")

    if args.data_only:
        return 0

    config = synthetic_config(args.config, outdir, n_genomes=args.genomes)
    results = run_synthetic_pipeline(config, outdir.parent)
    print(f"\nran {len(results)} stages; key outputs:")
    for relative in (
        "qc/hbv_qc_pass.tsv",
        "recombination/blocks.tsv",
        "phylogeny/genotype_tree.treefile",
        "selection/dual_frame.tsv",
        "epsilon/pol_epsilon_coevolution.tsv",
        "atlas/residue_atlas.parquet",
        "atlas/ranked_targets.tsv",
        "atlas/report.html",
    ):
        marker = "ok " if (outdir / relative).exists() else "MISSING"
        print(f"  [{marker}] {outdir / relative}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
