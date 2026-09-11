"""Stage 1 orchestration: acquire the human-HBV and deep-hepadnavirus datasets.

``run(config, root)`` returns a mapping of logical artefact name to path.  Every
source is optional: a missing NCBI e-mail, an unreachable HBVdb, absent HBV-GLUE
or a missing MAFFT all degrade to a warning, and only the artefacts that were
actually produced are returned.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Mapping

from ..config import get
from ..pipeline import StageError, get_logger, stage_dir
from .deephep import run as run_deephep
from .glue import fetch_glue_alignments
from .hbvdb import check_release_freshness, fetch_hbvdb
from .ncbi import fetch_genbank

__all__ = ["run", "ARTEFACT_NAMES"]

#: Names returned by :func:`run` under the canonical layout.
ARTEFACT_NAMES: tuple[str, ...] = (
    "hbv_genomes",
    "hbv_metadata",
    "hbvdb_genomes",
    "hbvdb_metadata",
    "deephep_pol",
    "deephep_alignment",
)


def _nonempty(path: str | Path) -> bool:
    candidate = Path(path)
    return candidate.exists() and candidate.stat().st_size > 0


def _configured_sources(config: Mapping[str, Any]) -> list[str]:
    sources = get(config, "datasets.hbv.sources", ["genbank"]) or []
    return [str(source).strip().lower() for source in sources]


def run(config: Mapping[str, Any], root: str | Path) -> dict[str, Path]:
    """Acquire every configured dataset source and return the artefacts produced."""
    root = Path(root)
    stage = stage_dir(config, root, "datasets")
    logger = get_logger("datasets")
    artefacts: dict[str, Path] = {}
    summary: dict[str, Any] = {}
    sources = _configured_sources(config)

    # --- human HBV: GenBank is the sequence-of-record, HBVdb the cross-check ---
    hbv_fasta = stage / "hbv_genomes.fasta"
    hbv_metadata = stage / "hbv_metadata.tsv"
    hbvdb_fasta = stage / "hbvdb_genomes.fasta"
    hbvdb_metadata = stage / "hbvdb_metadata.tsv"

    genbank_ok = False
    if "genbank" in sources:
        try:
            _fasta, _metadata = fetch_genbank(config, hbv_fasta, hbv_metadata)
            genbank_ok = _nonempty(hbv_fasta)
            summary["genbank"] = "ok" if genbank_ok else "no records"
        except StageError as exc:
            summary["genbank"] = f"unavailable ({exc})"
            logger.warning("GenBank source unavailable: %s", exc)
    else:
        summary["genbank"] = "disabled"

    hbvdb_ok = False
    if "hbvdb" in sources:
        try:
            fetch_hbvdb(config, hbvdb_fasta, hbvdb_metadata)
            hbvdb_ok = _nonempty(hbvdb_fasta)
            summary["hbvdb"] = "ok" if hbvdb_ok else "no records"
            if hbvdb_ok:
                fresh, message = check_release_freshness(config)
                summary["hbvdb_release"] = message
                if not fresh:
                    logger.warning("HBVdb release freshness: %s", message)
        except StageError as exc:
            summary["hbvdb"] = f"unavailable ({exc})"
            logger.warning("HBVdb source unavailable: %s", exc)
    else:
        summary["hbvdb"] = "disabled"

    if genbank_ok:
        artefacts["hbv_genomes"] = hbv_fasta
        artefacts["hbv_metadata"] = hbv_metadata
        if hbvdb_ok:
            artefacts["hbvdb_genomes"] = hbvdb_fasta
            artefacts["hbvdb_metadata"] = hbvdb_metadata
    elif hbvdb_ok:
        logger.warning("GenBank unavailable; using HBVdb as the human-HBV sequence-of-record")
        shutil.copyfile(hbvdb_fasta, hbv_fasta)
        shutil.copyfile(hbvdb_metadata, hbv_metadata)
        artefacts["hbv_genomes"] = hbv_fasta
        artefacts["hbv_metadata"] = hbv_metadata

    # --- HBV-GLUE maintained alignments (local installation only) -------------
    if "glue" in sources:
        try:
            glue = fetch_glue_alignments(config, stage / "glue")
            artefacts.update(glue)
            summary["glue"] = "ok" if glue else "unavailable"
        except StageError as exc:
            summary["glue"] = f"unavailable ({exc})"
            logger.warning("HBV-GLUE integration unavailable: %s", exc)
    else:
        summary["glue"] = "disabled"

    # --- deep hepadnavirus / nackednavirus Pol --------------------------------
    try:
        deephep = run_deephep(config, root)
        artefacts.update(deephep)
        summary["deephep"] = f"{len(deephep)} artefact(s)"
    except StageError as exc:
        summary["deephep"] = f"unavailable ({exc})"
        logger.warning("deep-hepadnavirus dataset unavailable: %s", exc)

    logger.info("dataset summary: %s", summary)
    return artefacts
