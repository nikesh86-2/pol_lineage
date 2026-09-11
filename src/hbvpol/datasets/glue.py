"""HBV-GLUE integration.

HBV-GLUE ships a maintained nucleotide alignment and a genotype reference set.
We do not re-implement GLUE; we locate a local installation and copy the files
this project consumes into the datasets stage directory.

.. note::
   **Documented assumption.**  GLUE installations differ in their internal
   directory names.  :func:`fetch_glue_alignments` therefore searches the
   installation recursively for a small set of permissive filename globs rather
   than hard-coding a layout from a specific release.

Everything here is local-only; no network access is performed.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Iterable, Mapping

from ..config import get
from ..pipeline import get_logger

__all__ = ["verify_glue_install", "fetch_glue_alignments"]

GLUE_ENV_VAR = "HBV_GLUE_HOME"

#: Filenames that suggest a directory really is a GLUE install / checkout.
_INSTALL_MARKERS = ("glue", "glue.jar", "hbv", "hbv-glue", "build.xml", "glue_config")

#: Globs used to locate the two maintained artefacts (assumption, see docstring).
_ALIGNMENT_GLOBS = (
    "**/*maintained*align*.fasta",
    "**/*align*.fasta",
    "**/*.aln.fasta",
    "**/*.fasta.aln",
)
_REFERENCE_GLOBS = (
    "**/*genotype*ref*.fasta",
    "**/*ref*genotype*.fasta",
    "**/*reference*.fasta",
    "**/*ref*.fasta",
)


def _candidate_paths(config: Mapping[str, object]) -> Iterable[Path]:
    """Yield plausible GLUE locations in priority order."""
    for dotted in (
        "datasets.hbv.glue.home",
        "datasets.hbv.glue.path",
        "datasets.hbv.glue.install",
        "datasets.hbv.glue.root",
    ):
        value = get(config, dotted)
        if value:
            yield Path(str(value))

    env_home = os.environ.get(GLUE_ENV_VAR)
    if env_home:
        yield Path(env_home)

    for executable in ("hbv-glue", "HBV-GLUE", "glue", "GLUE"):
        found = shutil.which(executable)
        if found:
            yield Path(found)


def verify_glue_install(path: str | Path) -> bool:
    """Best-effort check that ``path`` looks like an HBV-GLUE installation."""
    root = Path(path)
    if root.is_file():
        return root.suffix.lower() in {".jar", ""} and "glue" in root.name.lower()
    if not root.is_dir():
        return False
    try:
        names = {child.name.lower() for child in root.iterdir()}
    except OSError:
        return False
    if any(marker.lower() in names for marker in _INSTALL_MARKERS):
        return True
    # A source checkout may keep the payload one level down (e.g. HBV/ or hbv/).
    return any((root / sub).is_dir() for sub in ("HBV", "hbv", "hbv-glue", "HBV-GLUE"))


def _locate_glue(config: Mapping[str, object]) -> Path | None:
    for candidate in _candidate_paths(config):
        if candidate.exists() and verify_glue_install(candidate):
            return candidate
    return None


def _find_first(root: Path, globs: Iterable[str]) -> Path | None:
    for pattern in globs:
        for match in sorted(root.rglob(pattern)):
            if match.is_file() and match.stat().st_size > 0:
                return match
    return None


def fetch_glue_alignments(
    config: Mapping[str, object],
    out_dir: str | Path,
) -> dict[str, Path]:
    """Copy maintained HBV-GLUE alignments/reference into ``out_dir``.

    Returns a mapping of logical artefact name to copied path, or ``{}`` when no
    usable installation is found.
    """
    logger = get_logger("datasets.glue")
    out_dir = Path(out_dir)

    install = _locate_glue(config)
    if install is None:
        logger.warning(
            "HBV-GLUE installation not found (set %s or datasets.hbv.glue.home); "
            "skipping GLUE-derived alignments",
            GLUE_ENV_VAR,
        )
        return {}

    out_dir.mkdir(parents=True, exist_ok=True)
    artefacts: dict[str, Path] = {}

    if get(config, "datasets.hbv.glue.use_maintained_alignment", True):
        source = _find_first(install, _ALIGNMENT_GLOBS)
        if source is not None:
            artefacts["glue_alignment"] = _copy_into(source, out_dir, "glue_alignment.fasta")

    if get(config, "datasets.hbv.glue.use_genotype_reference", True):
        source = _find_first(install, _REFERENCE_GLOBS)
        if source is not None:
            artefacts["glue_genotype_reference"] = _copy_into(
                source, out_dir, "glue_genotype_reference.fasta"
            )

    logger.info("HBV-GLUE: copied %d artefact(s) from %s", len(artefacts), install)
    return artefacts


def _copy_into(source: Path, out_dir: Path, filename: str) -> Path:
    destination = out_dir / filename
    shutil.copyfile(source, destination)
    return destination
