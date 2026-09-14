"""Experimental fitness anchor stage pipeline.

load -> annotate -> write, plus a best-effort candidate shortlist that joins in
whatever conservation/structure evidence is already on disk.

Outputs (``<outroot>/fitness/``):

    dms_annotated.tsv
    dms_candidates.tsv
    fitness_summary.json

The candidate shortlist is *also* produced by the atlas stage, which has access
to the fully joined per-position table; here it is computed opportunistically so
the stage is useful on its own.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ..config import get
from ..io import read_table, write_table
from ..pipeline import get_logger, output_dir, stage_dir
from ..provenance import provenance
from .dms import (
    CRITERIA,
    DMS_COLUMNS,
    annotate_dms,
    load_dms,
    mechanistic_candidates,
)

__all__ = ["run"]

logger = get_logger("fitness")

ANNOTATION_COLUMNS = [
    "pol_position",
    "pol_codon",
    "pol_aa_wt",
    "pol_aa_mut",
    "surface_consequence",
    "intent",
    "is_intolerant",
]

CANDIDATE_COLUMNS = DMS_COLUMNS + ANNOTATION_COLUMNS + CRITERIA + ["n_criteria_met", "passes_criteria"]


def _empty_candidates() -> pd.DataFrame:
    return pd.DataFrame(columns=CANDIDATE_COLUMNS)


def _augment_with_upstream(annotated: pd.DataFrame, config: dict, root: Path) -> pd.DataFrame:
    """Best-effort join of selection/structure evidence onto annotated DMS rows."""
    if annotated.empty:
        return annotated
    outroot = output_dir(config, root)
    frame = annotated.copy()
    if "pol_position" not in frame.columns:
        return frame

    entropy_path = outroot / "selection" / "entropy.tsv"
    if entropy_path.exists():
        try:
            entropy = read_table(entropy_path)
            position_col = next((c for c in ("pol_position", "position", "site") if c in entropy.columns), None)
            entropy_col = next((c for c in ("aa_entropy", "entropy", "shannon_entropy") if c in entropy.columns), None)
            if position_col and entropy_col:
                subset = entropy[[position_col, entropy_col]].rename(
                    columns={position_col: "pol_position", entropy_col: "aa_entropy"}
                )
                frame = frame.merge(subset, on="pol_position", how="left")
        except Exception as error:  # pragma: no cover - defensive
            logger.warning("could not join entropy table: %s", error)

    covariation_path = outroot / "selection" / "covariation.tsv"
    if covariation_path.exists():
        try:
            covariation = read_table(covariation_path)
            position_col = next((c for c in ("pol_position", "position", "site") if c in covariation.columns), None)
            support_col = next((c for c in ("covariation_support", "support", "mi") if c in covariation.columns), None)
            if position_col and support_col:
                subset = covariation[[position_col, support_col]].rename(
                    columns={position_col: "pol_position", support_col: "covariation_support"}
                )
                frame = frame.merge(subset, on="pol_position", how="left")
        except Exception as error:  # pragma: no cover - defensive
            logger.warning("could not join covariation table: %s", error)

    hinges_path = outroot / "structure" / "hinges.tsv"
    if hinges_path.exists():
        try:
            hinges = read_table(hinges_path)
            if {"hinge_start", "hinge_end"}.issubset(hinges.columns) and "pol_position" in frame.columns:
                mask = pd.Series(False, index=frame.index)
                for _, hinge in hinges.iterrows():
                    mask |= frame["pol_position"].between(int(hinge["hinge_start"]), int(hinge["hinge_end"]))
                frame["is_hinge"] = mask
        except Exception as error:  # pragma: no cover - defensive
            logger.warning("could not join hinge table: %s", error)
    return frame


def run(config: dict, root) -> dict[str, Path]:
    """Execute the fitness stage and return the artefact mapping."""
    root = Path(root)
    outdir = stage_dir(config, root, "fitness")

    dms = load_dms(config, root)
    logger.info("loaded %d DMS rows", len(dms))
    annotated = annotate_dms(dms, config, root)
    for column in ANNOTATION_COLUMNS:
        if column not in annotated.columns:
            annotated[column] = None
    annotated_path = write_table(annotated, outdir / "dms_annotated.tsv")

    joined = _augment_with_upstream(annotated, config, root)
    if joined.empty:
        candidates = _empty_candidates()
    else:
        candidates = mechanistic_candidates(joined, config)
        for column in CANDIDATE_COLUMNS:
            if column not in candidates.columns:
                candidates[column] = None
        candidates = candidates[CANDIDATE_COLUMNS]
    candidates_path = write_table(candidates, outdir / "dms_candidates.tsv")

    summary = {
        "dms_map": str(get(config, "fitness.dms.map", "")),
        "n_dms_rows": int(len(dms)),
        "n_annotated_rows": int(len(annotated)),
        "n_candidates": int(len(candidates)),
        "criteria_enabled": {
            name: bool(get(config, f"fitness.criteria.{name}", True)) for name in CRITERIA
        },
        "provenance": provenance([]),
    }
    summary_path = outdir / "fitness_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return {
        "dms_annotated": annotated_path,
        "dms_candidates": candidates_path,
        "summary": summary_path,
    }
