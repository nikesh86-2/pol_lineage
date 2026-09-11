"""Command-line entry point.

Each analysis stage is a package exposing ``pipeline.run(config, root) -> dict``
mapping logical artefact names to paths.  This module wires those stages into a
single ``hbvpol`` command with import-on-demand so that a missing optional
dependency (for example a structural predictor) never breaks unrelated stages.

    hbvpol fetch      -c config/config.yaml
    hbvpol qc         -c config/config.yaml
    hbvpol recombine  -c config/config.yaml
    hbvpol tree       -c config/config.yaml
    hbvpol select     -c config/config.yaml
    hbvpol structure  -c config/config.yaml
    hbvpol epsilon    -c config/config.yaml
    hbvpol fitness    -c config/config.yaml
    hbvpol atlas      -c config/config.yaml
    hbvpol all        -c config/config.yaml
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from typing import Callable

from .config import load_config

# stage name -> "module:callable"
STAGES: dict[str, str] = {
    "fetch": "hbvpol.datasets.pipeline:run",
    "deephep": "hbvpol.datasets.deephep:run",
    "qc": "hbvpol.qc.pipeline:run",
    "recombine": "hbvpol.recombination.pipeline:run",
    "tree": "hbvpol.phylogeny.pipeline:run",
    "select": "hbvpol.selection.pipeline:run",
    "structure": "hbvpol.structure.pipeline:run",
    "epsilon": "hbvpol.epsilon.pipeline:run",
    "fitness": "hbvpol.fitness.pipeline:run",
    "atlas": "hbvpol.atlas.pipeline:run",
}

# Execution order for `all`.  Structural sampling is the expensive tail.
ORDER = [
    "fetch",
    "deephep",
    "qc",
    "recombine",
    "tree",
    "select",
    "epsilon",
    "fitness",
    "structure",
    "atlas",
]


def _resolve(target: str) -> Callable:
    module_name, _, attr = target.partition(":")
    module = importlib.import_module(module_name)
    return getattr(module, attr)


def _run_stage(stage: str, config: dict, root: Path) -> dict:
    func = _resolve(STAGES[stage])
    artefacts = func(config, root)
    if artefacts:
        for name, path in artefacts.items():
            print(f"[{stage}] {name}: {path}")
    else:
        print(f"[{stage}] done")
    return artefacts or {}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hbvpol", description=__doc__)
    parser.add_argument("stage", choices=[*STAGES, "all"], help="pipeline stage to run")
    parser.add_argument("-c", "--config", default="config/config.yaml", help="path to config YAML")
    parser.add_argument("--root", default=".", help="project root for relative paths")
    parser.add_argument("overrides", nargs="*", help="dotted config overrides, e.g. selection.window=50")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    # `parse_known_args` keeps dotted overrides working even when they follow
    # options (argparse cannot reliably bind a trailing nargs="*" positional
    # after interleaved optionals).
    args, extras = parser.parse_known_args(argv)
    overrides = [*args.overrides, *(token for token in extras if "=" in token)]
    config = load_config(args.config, overrides=overrides)
    root = Path(args.root)

    if args.stage == "all":
        for stage in ORDER:
            _run_stage(stage, config, root)
        return 0

    _run_stage(args.stage, config, root)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
