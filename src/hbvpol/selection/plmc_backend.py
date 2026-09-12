"""PLMC (``plmc``) backend for the DCA covariation hook.

``pydca``/``plmDCA`` (what the pipeline originally imported) is unmaintained.
``plmc`` is the maintained C++ implementation of plmDCA and is packaged on
bioconda::

    mamba install -p <env> -c bioconda plmc

``plmc`` is a **binary**, not an importable Python module, so it cannot satisfy
``selection.covariation.dca_impl`` on its own.  This adapter exposes the
``dca_scores(alignment, config=None) -> np.ndarray`` interface that
:func:`hbvpol.selection.covariation._try_external_dca` expects.  Select it with::

    selection:
      covariation:
        dca_impl: hbvpol.selection.plmc_backend

It writes the (already column-restricted) alignment to FASTA, runs
``plmc -c couplings.txt`` and parses the coupling scores -- one line per
unordered pair, ``i - j - 0 score`` -- into a symmetric L x L matrix.

Alphabet: sequences drawn from the nucleotide set use ``-a "-ACGT"`` (the
leading ``-`` is the gap state); anything else is left to plmc's protein
default.  For a large alignment set ``selection.covariation.plmc_fast: true`` to
enable plmc's stochastic-gradient ``--fast`` mode.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Mapping

import numpy as np

from ..config import get
from ..io import GenomeRecord, write_fasta
from ..pipeline import StageError, effective_threads, get_logger, have_executable, run_command

__all__ = ["dca_scores", "parse_couplings", "NUCLEOTIDE_ALPHABET"]

logger = get_logger("selection.plmc")

#: plmc alphabet for nucleotide alignments (leading "-" is the gap state).
NUCLEOTIDE_ALPHABET = "-ACGT"

_NUCLEOTIDE_SYMBOLS = set("ACGTUN-.")


def _records(alignment) -> list[GenomeRecord]:
    from .dualframe import coerce_alignment

    return coerce_alignment(alignment)


def _alphabet(records: list[GenomeRecord]) -> str | None:
    """Return the plmc alphabet for these records, or ``None`` for the default."""
    symbols: set[str] = set()
    for record in records:
        symbols.update(record.seq.upper())
    if symbols and symbols <= _NUCLEOTIDE_SYMBOLS and symbols & set("ACGT"):
        return NUCLEOTIDE_ALPHABET
    return None


def parse_couplings(path: str | Path) -> dict[tuple[int, int], float]:
    """Parse a plmc ``-c`` couplings file into ``{(i, j): score}`` (0-based).

    Each line has six whitespace-separated fields (``i - j - 0 score``); the
    pairs are unordered and 1-based.  Unparseable lines are skipped, which keeps
    the adapter robust to plmc version changes.
    """
    pairs: dict[tuple[int, int], float] = {}
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        fields = line.split()
        if len(fields) < 3:
            continue
        try:
            i = int(fields[0])
            j = int(fields[2])
            score = float(fields[-1])
        except ValueError:
            continue
        pairs[(i - 1, j - 1)] = score
    return pairs


def dca_scores(alignment, config: Mapping[str, object] | None = None) -> np.ndarray:
    """Return the plmc coupling-score matrix for an alignment.

    Raises :class:`StageError` when plmc is missing or produces no scores, so
    that ``selection.covariation.require_backend`` can turn that into a hard
    failure instead of a silent proxy.
    """
    records = _records(alignment)
    if not records:
        return np.zeros((0, 0), dtype=float)

    width = max(len(record.seq) for record in records)
    executable = "plmc"
    if config is not None:
        executable = str(get(config, "selection.covariation.plmc_executable", "plmc") or "plmc")
    if not (Path(executable).exists() or have_executable(executable)):
        raise StageError(
            f"plmc executable {executable!r} not found; install it "
            "(mamba install -c bioconda plmc) or choose another "
            "selection.covariation.dca_impl"
        )

    threads = effective_threads(config) if config is not None else 1
    fast = bool(get(config, "selection.covariation.plmc_fast", False)) if config is not None else False
    alphabet = _alphabet(records)

    with tempfile.TemporaryDirectory(prefix="hbvpol_plmc_") as tmp:
        workdir = Path(tmp)
        fasta = write_fasta(records, workdir / "alignment.fasta")
        couplings = workdir / "couplings.txt"
        argv = [executable]
        if alphabet:
            argv += ["-a", alphabet]
        argv += ["-n", str(max(1, threads))]
        if fast:
            argv += ["--fast"]
        argv += ["-c", str(couplings), str(fasta)]
        run_command(argv, cwd=workdir, log_path=workdir / "plmc.log", check=True)
        pairs = parse_couplings(couplings)

    if not pairs:
        raise StageError("plmc produced no coupling scores")

    matrix = np.zeros((width, width), dtype=float)
    for (i, j), score in pairs.items():
        if 0 <= i < width and 0 <= j < width:
            matrix[i, j] = matrix[j, i] = score
    return matrix
