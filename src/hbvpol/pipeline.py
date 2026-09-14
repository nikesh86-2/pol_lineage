"""Shared runtime utilities for stage pipelines.

Every stage's ``pipeline.run(config, root)`` uses these helpers so that output
layout, logging and external-tool invocation are consistent across the project.
"""

from __future__ import annotations

import logging
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .config import get

__all__ = [
    "get_logger",
    "output_dir",
    "stage_dir",
    "run_command",
    "require_executable",
    "have_executable",
    "available_cpus",
    "effective_threads",
    "verify_required_tools",
    "strict_tools",
    "StageError",
]


class StageError(RuntimeError):
    """Raised when a pipeline stage cannot complete."""


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(f"hbvpol.{name}")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def output_dir(config: Mapping[str, object], root: str | Path) -> Path:
    """The per-project output root (``project.output_root``)."""
    path = Path(str(get(config, "project.output_root", "output")))
    return path if path.is_absolute() else Path(root) / path


def stage_dir(config: Mapping[str, object], root: str | Path, stage: str) -> Path:
    """Create and return the output directory for a named stage."""
    path = output_dir(config, root) / stage
    path.mkdir(parents=True, exist_ok=True)
    return path


def have_executable(name: str) -> bool:
    return shutil.which(name) is not None


def available_cpus() -> int:
    """CPUs actually usable by this process (respects SLURM/cgroup affinity)."""
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):  # pragma: no cover - non-Linux fallback
        return max(1, os.cpu_count() or 1)


def effective_threads(config: Mapping[str, object], key: str = "project.threads", default: int = 1) -> int:
    """Configured thread count, clamped to the CPUs available to this process.

    A SLURM allocation can be smaller than ``project.threads``, and some tools
    (notably IQ-TREE 3) refuse to run when given more threads than cores.  The
    clamp keeps a large config value safe inside a small allocation without the
    user having to edit the config per job.
    """
    try:
        requested = int(get(config, key, default) or default)
    except (TypeError, ValueError):
        requested = default
    return max(1, min(requested, available_cpus()))


def verify_required_tools(config: Mapping[str, object]) -> None:
    """Fail fast when tools listed in ``project.required_tools`` are missing.

    Offline fallbacks (Neighbor-Joining, parsimony, Nussinov, pure-Python
    bootscan) exist so the pipeline runs without external software, but a
    publication run must not silently substitute them.  Set
    ``project.required_tools: [mafft, iqtree, hyphy]`` to make their absence a
    hard error up front instead of a warning buried in a stage log.
    """
    required = get(config, "project.required_tools", []) or []
    if isinstance(required, str):
        required = [required]
    missing = [str(tool) for tool in required if tool and not have_executable(str(tool))]
    if missing:
        raise StageError(
            "required external tools are missing: " + ", ".join(sorted(missing))
            + " (install them, or clear project.required_tools to allow fallbacks)"
        )


def strict_tools(config: Mapping[str, object]) -> bool:
    """Whether a degraded external-tool fallback must be a hard error.

    :func:`verify_required_tools` only catches *missing* binaries.  A publication
    run also wants a crash, a timeout or unparseable output to fail loudly rather
    than silently substituting a heuristic, so set ``project.strict_tools: true``
    alongside a populated ``project.required_tools``.
    """
    return bool(get(config, "project.strict_tools", False))


def require_executable(name: str, hint: str = "") -> str:
    """Return the path to an executable or raise a helpful error."""
    path = shutil.which(name)
    if path is None:
        message = f"required executable {name!r} not found on PATH"
        if hint:
            message += f" — {hint}"
        raise StageError(message)
    return path


def run_command(
    command: Sequence[str] | str,
    *,
    cwd: str | Path | None = None,
    log_path: str | Path | None = None,
    stderr_path: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    check: bool = True,
    timeout: float | None = None,
) -> subprocess.CompletedProcess:
    """Run an external tool, tee-ing stdout/stderr to a log file if given.

    Commands are passed as argument vectors (never ``shell=True``) so that
    user-supplied metadata cannot be interpreted by the shell.  ``timeout``
    bounds the wall-clock run (seconds); on expiry ``subprocess.TimeoutExpired``
    is raised so a caller can fall back to any partial output already written.
    """
    argv = shlex.split(command) if isinstance(command, str) else list(command)
    full_env = {**os.environ, **(env or {})}
    logger = get_logger("command")
    logger.info("run: %s", " ".join(shlex.quote(a) for a in argv))

    if log_path is not None:
        log_file = Path(log_path)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with log_file.open("w", encoding="utf-8") as handle:
            handle.write(f"# cwd={cwd}\n# {' '.join(shlex.quote(a) for a in argv)}\n\n")
            handle.flush()
            result = subprocess.run(
                argv, cwd=cwd, env=full_env, stdout=handle, stderr=subprocess.STDOUT,
                text=True, timeout=timeout,
            )
    elif stderr_path is not None:
        # Capture stdout (the useful output, e.g. an alignment) while teeing
        # stderr to a file, so a tool's full diagnostic survives the error tail.
        err_file = Path(stderr_path)
        err_file.parent.mkdir(parents=True, exist_ok=True)
        with err_file.open("w", encoding="utf-8") as err_handle:
            err_handle.write(f"# cwd={cwd}\n# {' '.join(shlex.quote(a) for a in argv)}\n\n")
            err_handle.flush()
            result = subprocess.run(
                argv, cwd=cwd, env=full_env, stdout=subprocess.PIPE, stderr=err_handle,
                text=True, timeout=timeout,
            )
    else:
        result = subprocess.run(argv, cwd=cwd, env=full_env, capture_output=True, text=True, timeout=timeout)

    if check and result.returncode != 0:
        detail = ""
        if log_path is not None:
            detail = f" (see {log_path})"
        elif getattr(result, "stderr", None):
            detail = f": {result.stderr.strip()[-800:]}"
        raise StageError(f"command failed with exit {result.returncode}: {' '.join(argv)}{detail}")
    return result


def write_records(artefacts: Iterable[tuple[str, Path]]) -> dict[str, Path]:
    """Small helper to build the ``{name: path}`` return value of a stage."""
    return {name: Path(path) for name, path in artefacts}
