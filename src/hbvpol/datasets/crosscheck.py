"""Provenance-aware cross-validation that never blocks the pipeline.

NCBI GenBank is the sequence-of-record.  Specialist resources are *validation
layers*, not dependencies: HBV-GLUE (pinned clone) for independent annotation,
Stanford HBVseq for RT-resistance interpretation, a checksummed legacy HBVdb
snapshot for historical comparison, and local translation/motif/coordinate
checks.  Any of them being unavailable is recorded as a status, not an error.

The emitted manifest uses these states::

    matched | conflict | not_found | not_attempted
    | service_unavailable | not_applicable

so "HBVdb unreachable" becomes ``hbvdb_status = service_unavailable`` and
``crosscheck_status = partial``, never "record invalid".
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from ..config import get
from ..domain import frame_indices, translate
from ..io import read_fasta
from ..pipeline import get_logger, output_dir

__all__ = [
    "STATUS_STATES",
    "CROSSCHECK_FIELDS",
    "local_checks",
    "source_statuses",
    "run_crosscheck",
    "write_crosscheck",
]

logger = get_logger("datasets.crosscheck")

STATUS_STATES = (
    "matched",
    "conflict",
    "not_found",
    "not_attempted",
    "service_unavailable",
    "not_applicable",
)

CROSSCHECK_FIELDS = (
    "hbvdb_status",
    "hbv_glue_status",
    "stanford_hbvseq_status",
    "crosscheck_status",
    "crosscheck_date",
    "crosscheck_details",
)


def _pol_protein(sequence: str, pol_start: int, pol_end: int) -> str:
    length = len(sequence)
    if length == 0:
        return ""
    span = (pol_end - pol_start) % length + 1
    indices = frame_indices(pol_start, span, length)
    return translate("".join(sequence[index % length] for index in indices), frame=0)


def local_checks(records, config: Mapping[str, Any]) -> dict[str, Any]:
    """Independent local validation of the sequence-of-record.

    Checks that the polymerase ORF translates without internal stops, that the
    invariant YMDD catalytic motif is recoverable, and that lengths are coherent.
    This is the part of the cross-check that does not need any external service.
    """
    limit = int(get(config, "datasets.crosscheck.max_records", 500) or 500)
    sample = list(records)[:limit]
    if not sample:
        return {
            "n_checked": 0,
            "pol_orf_stop_free_frac": 0.0,
            "ymdd_motif_frac": 0.0,
            "length_coherent_frac": 0.0,
        }
    pol_start = int(get(config, "reference.pol_start_nt", 1) or 1)
    pol_end = int(get(config, "reference.pol_end_nt", 1) or 1)
    min_length = int(get(config, "qc.min_length", 0) or 0)

    stop_free = 0
    motif = 0
    coherent = 0
    for record in sample:
        sequence = record.seq.upper()
        protein = _pol_protein(sequence, pol_start, pol_end)
        internal = protein.count("*") - (1 if protein.endswith("*") else 0)
        if protein and internal <= 0:
            stop_free += 1
        if "YMDD" in protein:
            motif += 1
        if min_length <= len(sequence):
            coherent += 1
    n = len(sample)
    return {
        "n_checked": n,
        "pol_orf_stop_free_frac": round(stop_free / n, 4),
        "ymdd_motif_frac": round(motif / n, 4),
        "length_coherent_frac": round(coherent / n, 4),
    }


def _exists(path_value: Any, root: Path) -> bool:
    if not path_value:
        return False
    path = Path(str(path_value))
    if not path.is_absolute():
        path = root / path
    return path.exists()


def source_statuses(config: Mapping[str, Any], root: Path, artefacts: Mapping[str, Any]) -> dict[str, Any]:
    """Status of every validation source, derived without making any request.

    A source is ``matched`` only when a local artefact from it exists;
    ``service_unavailable`` when it was configured but produced nothing; and
    ``not_attempted`` when it was not configured or is a local clone that has not
    been analysed.
    """
    sources = [str(item).strip().lower() for item in (get(config, "datasets.hbv.sources", []) or [])]

    def _has(artefact: str) -> bool:
        path = artefacts.get(artefact)
        return bool(path) and Path(str(path)).exists() and Path(str(path)).stat().st_size > 0

    # HBVdb: live service or a checksummed legacy snapshot.
    snapshot = _exists(get(config, "datasets.crosscheck.hbvdb_snapshot"), root)
    if _has("hbvdb_genomes"):
        hbvdb_status = "matched"
    elif "hbvdb" in sources and not snapshot:
        hbvdb_status = "service_unavailable"
    elif snapshot:
        hbvdb_status = "not_attempted"
    else:
        hbvdb_status = "not_applicable"

    # HBV-GLUE / Hepadnaviridae-GLUE: local clones, pinned by commit.
    glue_clone = _exists(get(config, "datasets.crosscheck.hbv_glue_path"), root)
    if glue_clone:
        hbv_glue_status = "not_attempted"
    elif "glue" in sources:
        hbv_glue_status = "not_found"
    else:
        hbv_glue_status = "not_applicable"

    hepadnaviridae_glue = _exists(
        get(config, "datasets.crosscheck.hepadnaviridae_glue_path"), root
    )

    # Stanford HBVseq: an RT-focused catalogue, used for interpretation.
    stanford = _exists(get(config, "datasets.crosscheck.stanford_catalogue"), root)
    stanford_status = "not_applicable" if stanford else "not_attempted"

    return {
        "hbvdb_status": hbvdb_status,
        "hbv_glue_status": hbv_glue_status,
        "stanford_hbvseq_status": stanford_status,
        "glue_clone_present": glue_clone,
        "hepadnaviridae_glue_present": hepadnaviridae_glue,
        "hbvdb_snapshot_present": snapshot,
    }


def run_crosscheck(
    config: Mapping[str, Any],
    root: str | Path,
    records_path: str | Path | None = None,
    artefacts: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute the cross-check manifest for the sequence-of-record.

    Never raises: an unreadable alignment yields ``not_attempted`` local checks
    and the pipeline continues.
    """
    root = Path(root)
    if records_path is None:
        records_path = output_dir(config, root) / "datasets" / "hbv_genomes.fasta"
    records_path = Path(records_path)

    records: list[Any] = []
    try:
        if records_path.exists():
            records = read_fasta(records_path)
    except Exception as error:  # pragma: no cover - defensive
        logger.warning("crosscheck: could not read %s (%s)", records_path, error)

    checks = local_checks(records, config) if records else {
        "n_checked": 0,
        "pol_orf_stop_free_frac": 0.0,
        "ymdd_motif_frac": 0.0,
        "length_coherent_frac": 0.0,
    }
    statuses = source_statuses(config, root, dict(artefacts or {}))

    min_stop_free = float(get(config, "datasets.crosscheck.pol_orf_stop_free_frac", 0.95) or 0.95)
    min_motif = float(get(config, "datasets.crosscheck.ymdd_motif_frac", 0.95) or 0.95)
    local_ok = (
        checks["n_checked"] > 0
        and checks["pol_orf_stop_free_frac"] >= min_stop_free
        and checks["ymdd_motif_frac"] >= min_motif
    )
    external_ok = any(
        statuses[field] == "matched"
        for field in ("hbvdb_status", "hbv_glue_status", "stanford_hbvseq_status")
    )

    if checks["n_checked"] == 0:
        crosscheck_status = "not_attempted"
    elif local_ok and external_ok:
        crosscheck_status = "complete"
    else:
        crosscheck_status = "partial"

    details = {
        "local_checks": checks,
        "sources": statuses,
        "local_ok": local_ok,
        "external_any_matched": external_ok,
        "records_path": str(records_path),
        "note": (
            "GenBank is the sequence-of-record. Local translation/motif checks "
            "run always; external sources are recorded, never required."
        ),
    }

    payload: dict[str, Any] = {
        "hbvdb_status": statuses["hbvdb_status"],
        "hbv_glue_status": statuses["hbv_glue_status"],
        "stanford_hbvseq_status": statuses["stanford_hbvseq_status"],
        "crosscheck_status": crosscheck_status,
        "crosscheck_date": datetime.now(timezone.utc).date().isoformat(),
        "crosscheck_details": details,
    }
    return payload


def write_crosscheck(payload: Mapping[str, Any], path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out
