"""Ad-hoc timing harness for the offline synthetic pipeline (dev utility)."""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hbvpol.cli import STAGES, _resolve  # noqa: E402
from hbvpol.synthetic import SYNTHETIC_STAGES, build_synthetic, synthetic_config  # noqa: E402

CONFIG = ROOT / "config" / "config.yaml"


def main() -> int:
    outdir = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "output-profile"
    build_synthetic(outdir, n_genomes=8, seed=7)
    config = synthetic_config(CONFIG, outdir, n_genomes=8)
    for stage in SYNTHETIC_STAGES:
        func = _resolve(STAGES[stage])
        start = time.perf_counter()
        func(config, outdir.parent)
        print(f"{stage:>10}: {time.perf_counter() - start:8.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
