"""Structural ensemble stage pipeline.

manifest -> predict (skipping unavailable predictors) -> metrics for any models
that exist -> hinge calling -> MD plan.

Outputs (``<outroot>/structure/``):

    models/<state>/<lineage>/<strategy>_seed<k>.pdb|cif
    model_manifest.tsv
    metrics.tsv
    hinges.tsv
    interface_residues.tsv
    md/pocket_persistence.tsv
    md/md_manifest.tsv

None of the external predictors (ColabFold, AlphaFold3, ESMFold) are required:
when they are absent every model is simply skipped and the manifest, an empty
metrics/hinges table and the MD plan are still written.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import get
from ..io import write_table
from ..pipeline import get_logger, stage_dir
from ..provenance import provenance
from .md import CAVITY_COLUMNS, MD_MANIFEST_COLUMNS, build_md_manifest, pocket_persistence
from .metrics import candidate_hinges, compute_metrics, interface_residues, metrics_table, plddt_from_pdb
from .models import build_model_manifest, run_predictor, sequence_for_lineage

__all__ = ["run"]

logger = get_logger("structure")

HINGE_COLUMNS = ["model_id", "hinge_start", "hinge_end", "length", "plddt_mean", "plddt_min"]

#: Per-residue nucleic-acid interface flags.  ``pol_position`` is the file-order
#: residue index, matching hinges and the atlas key.
INTERFACE_COLUMNS = ["model_id", "pol_position", "min_distance", "is_interface"]


def _empty_hinges() -> pd.DataFrame:
    return pd.DataFrame(columns=HINGE_COLUMNS)


def _iter_models(models_dir: Path):
    for pattern in ("*.pdb", "*.ent", "*.cif", "*.mmcif"):
        yield from sorted(models_dir.rglob(pattern))


def _model_id(path: Path, models_dir: Path) -> str:
    relative = path.relative_to(models_dir)
    return "__".join(relative.with_suffix("").parts)


def run(config: dict, root) -> dict[str, Path]:
    """Execute the structural stage and return the artefact mapping."""
    root = Path(root)
    outdir = stage_dir(config, root, "structure")
    models_dir = outdir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    manifest = build_model_manifest(config, root)
    manifest_path = write_table(manifest, outdir / "model_manifest.tsv")

    # --- predict (offline-safe: every predictor may be skipped) -------------
    predicted = 0
    skipped = 0
    for _, row in manifest.iterrows():
        target = models_dir / str(row["rel_path"]).removeprefix("models/")
        if any(target.with_suffix(suffix).exists() for suffix in (".pdb", ".ent", ".cif", ".mmcif")):
            continue
        record = sequence_for_lineage(str(row["lineage"]), config, root)
        if record is None:
            skipped += 1
            continue
        produced = run_predictor(record, target.parent, config, strategy=str(row["strategy"]))
        if produced is not None:
            predicted += 1
        else:
            skipped += 1
    logger.info("structure: predicted=%d skipped=%d", predicted, skipped)

    # --- metrics for whatever models exist ---------------------------------
    records = [compute_metrics(path, config) for path in _iter_models(models_dir)]
    metrics = metrics_table(records)
    metrics_path = write_table(metrics, outdir / "metrics.tsv")

    # --- hinges -------------------------------------------------------------
    threshold = float(get(config, "structure.hinge_plddt", 70.0) or 70.0)
    hinge_rows: list[dict[str, object]] = []
    for path in _iter_models(models_dir):
        model_id = _model_id(path, models_dir)
        try:
            plddt = plddt_from_pdb(path)
        except Exception as error:  # pragma: no cover - defensive
            logger.warning("could not read pLDDT for %s: %s", path, error)
            continue
        for start, end in candidate_hinges(plddt, threshold=threshold):
            segment = plddt[start - 1:end]
            hinge_rows.append({
                "model_id": model_id,
                "hinge_start": int(start),
                "hinge_end": int(end),
                "length": int(end - start + 1),
                "plddt_mean": float(np.nanmean(segment)) if segment.size else None,
                "plddt_min": float(np.nanmin(segment)) if segment.size else None,
            })
    hinges = pd.DataFrame(hinge_rows, columns=HINGE_COLUMNS) if hinge_rows else _empty_hinges()
    hinges_path = write_table(hinges, outdir / "hinges.tsv")

    # --- nucleic-acid interface residues ------------------------------------
    cutoff = float(get(config, "structure.interface_cutoff", 4.5) or 4.5)
    interface_rows: list[dict[str, object]] = []
    for path in _iter_models(models_dir):
        model_id = _model_id(path, models_dir)
        try:
            distances = interface_residues(path)
        except Exception as error:  # pragma: no cover - defensive
            logger.warning("could not compute interface residues for %s: %s", path, error)
            continue
        for index, distance in enumerate(distances, start=1):
            interface_rows.append({
                "model_id": model_id,
                "pol_position": int(index),
                "min_distance": float(distance),
                "is_interface": bool(np.isfinite(distance) and distance <= cutoff),
            })
    interface = (
        pd.DataFrame(interface_rows, columns=INTERFACE_COLUMNS)
        if interface_rows else pd.DataFrame(columns=INTERFACE_COLUMNS)
    )
    interface_path = write_table(interface, outdir / "interface_residues.tsv")

    # --- MD plan ------------------------------------------------------------
    md_dir = outdir / "md"
    md_dir.mkdir(parents=True, exist_ok=True)
    md_manifest = build_md_manifest(manifest, config)
    if md_manifest.empty:
        md_manifest = pd.DataFrame(columns=MD_MANIFEST_COLUMNS)
    md_manifest_path = write_table(md_manifest, md_dir / "md_manifest.tsv")

    trajectory_dir = get(config, "structure.md.trajectory_dir") or (md_dir / "trajectories")
    persistence = pocket_persistence(trajectory_dir, config)
    if persistence.empty:
        persistence = pd.DataFrame(columns=CAVITY_COLUMNS)
    persistence_path = write_table(persistence, md_dir / "pocket_persistence.tsv")

    summary = {
        "n_planned_models": int(len(manifest)),
        "n_predicted": predicted,
        "n_skipped": skipped,
        "n_models_present": int(len(records)),
        "n_hinges": int(len(hinges)),
        "n_interface_residues": int(interface["is_interface"].sum()) if len(interface) else 0,
        "n_md_models": int(len(md_manifest)),
        "n_persistent_cavities": int(persistence["passes"].sum()) if len(persistence) else 0,
        "hinge_plddt": threshold,
        "predictors": list(get(config, "structure.strategies", []) or []),
        "provenance": provenance(["colabfold_batch", "run_alphafold", "esm-fold"]),
    }
    summary_path = outdir / "structure_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return {
        "models": models_dir,
        "model_manifest": manifest_path,
        "metrics": metrics_path,
        "hinges": hinges_path,
        "interface_residues": interface_path,
        "md_manifest": md_manifest_path,
        "pocket_persistence": persistence_path,
        "summary": summary_path,
    }
