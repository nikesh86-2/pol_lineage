"""OpenRDP wrapper and output parser.

`OpenRDP <https://github.com/PoonLab/OpenRDP>`_ is an open-source Python
re-implementation of RDP4/RDP5 (PoonLab).  Its methods include ``rdp``,
``geneconv``, ``bootscan``, ``maxchi``, ``siscan``, ``chimaera`` and ``threeseq``;
it bundles the third-party 3Seq and GENECONV binaries.

OpenRDP pins ``numpy<2`` and ``h5py<3.11``, so it is best installed in its own
environment and invoked as a subprocess::

    mamba create -p <envs>/openrdp -c conda-forge python=3.11 "numpy<2" scipy "h5py<3.11" pip
    <envs>/openrdp/bin/pip install "git+https://github.com/PoonLab/OpenRDP.git"

then point ``recombination.openrdp.executable`` at ``<envs>/openrdp/bin/openrdp``.

CLI notes (from ``openrdp -h``):

* the alignment is a **positional** argument and must come *before* ``-m``;
* ``-m`` is greedy (``nargs='+'``), so list methods before any other flag;
* ``-o <csv>`` writes a CSV with ``Method,Start,End,Recombinant,Parent1,Parent2,Pvalue``.

OpenRDP is explicitly under development; some methods raise on some inputs.  The
wrapper runs the configured subset and fails soft, returning an empty frame when
the tool is absent, errors, or writes no report.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..config import get
from ..pipeline import StageError, get_logger, have_executable, run_command
from .partition import BREAKPOINT_COLUMNS, normalise_breakpoints

__all__ = ["run_openrdp", "parse_openrdp_output"]

logger = get_logger("recombination.openrdp")


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=BREAKPOINT_COLUMNS)


def parse_openrdp_output(path: str | Path) -> pd.DataFrame:
    """Parse an OpenRDP CSV report into the tidy breakpoint schema.

    ``Parent1``/``Parent2`` are combined into a single ``partner`` field, the
    method name is kept in ``region``, and the p-value is mapped onto
    ``support`` (as for the RDP5/3SEQ parsers — support is tool-specific:
    a similarity for bootscan, a p-value here).
    """
    path = Path(path)
    if not path.exists():
        return _empty()
    try:
        frame = pd.read_csv(path)
    except Exception as error:  # pragma: no cover - defensive
        logger.warning("could not parse OpenRDP output %s: %s", path, error)
        return _empty()
    if frame is None or len(frame) == 0:
        return _empty()

    lower = {str(column).strip().lower(): column for column in frame.columns}
    frame = frame.copy()
    parent1 = frame[lower["parent1"]] if "parent1" in lower else ""
    parent2 = frame[lower["parent2"]] if "parent2" in lower else ""
    if "parent1" in lower or "parent2" in lower:
        frame["partner"] = [
            "|".join(str(value) for value in (one, two) if str(value) not in {"", "nan", "-"})
            for one, two in zip(parent1, parent2)
        ]
    if "method" in lower:
        frame["region"] = frame[lower["method"]].astype(str)
    return normalise_breakpoints(frame, tool="openrdp")


def run_openrdp(alignment, config, workdir) -> pd.DataFrame:
    """Run OpenRDP on ``alignment`` and return its breakpoints.

    Fails soft: a missing executable, a non-zero exit (some OpenRDP methods are
    unstable) or an empty report all yield an empty frame with a warning.
    """
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    alignment_path = Path(alignment).resolve()

    executable_name = str(get(config, "recombination.openrdp.executable", "openrdp") or "openrdp")
    if not (Path(executable_name).exists() or have_executable(executable_name)):
        logger.warning("OpenRDP executable %r not found; skipping OpenRDP", executable_name)
        return _empty()

    methods = get(config, "recombination.openrdp.methods", ["rdp", "threeseq"]) or []
    if isinstance(methods, str):
        methods = [methods]
    methods = [str(method).strip().lower() for method in methods if str(method).strip()]

    # Absolute: run_command sets cwd=workdir, so a relative -o would be
    # re-rooted under it and OpenRDP could not create its report.
    out_csv = (workdir / "openrdp.csv").resolve()
    # Positional alignment first, then the greedy -m list, then other options.
    argv = [executable_name, str(alignment_path)]
    if methods:
        argv += ["-m", *methods]
    argv += ["-o", str(out_csv)]

    cfg_path = get(config, "recombination.openrdp.cfg")
    if cfg_path:
        path = Path(str(cfg_path))
        argv += ["-c", str(path if path.is_absolute() else (workdir / path).resolve())]
    seed = get(config, "recombination.openrdp.seed")
    if seed not in (None, ""):
        argv += ["-s", str(seed)]
    fail = get(config, "recombination.openrdp.fail")
    if fail not in (None, ""):
        argv += ["-f", str(fail)]

    try:
        run_command(argv, cwd=workdir, log_path=workdir / "openrdp.log", check=False)
    except StageError as error:  # pragma: no cover - external tool behaviour
        logger.warning("OpenRDP invocation failed: %s", error)
        return _empty()

    if not out_csv.exists() or out_csv.stat().st_size == 0:
        logger.warning("OpenRDP produced no report at %s; skipping", out_csv)
        return _empty()
    return parse_openrdp_output(out_csv)
