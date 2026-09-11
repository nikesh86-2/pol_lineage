"""Residue-atlas stage pipeline.

Joins every upstream per-position table on ``atlas.key`` = [lineage,
pol_position] with an **outer** join, so partial availability (missing
selection, structure, epsilon or fitness files) still yields a usable table.
Derived columns, the six mechanistic-target criteria, ranked targets, the
interaction network, conformational-switch candidates and a self-contained HTML
report are then written out.

Outputs (``<outroot>/atlas/``):

    residue_atlas.parquet
    residue_atlas.tsv
    ranked_targets.tsv
    conserved_interactions.tsv
    conformational_switches.tsv
    report.html
    atlas_summary.json
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ..config import get
from ..domain import domain_of
from ..fitness.dms import CRITERIA, score_criteria
from ..io import read_table, write_table
from ..pipeline import get_logger, output_dir, stage_dir
from .report import build_report
from .targets import (
    INTERACTION_COLUMNS,
    RANKED_TARGET_COLUMNS,
    SWITCH_COLUMNS,
    conformational_switch_candidates,
    interaction_network,
    rank_targets,
)

__all__ = ["run"]

logger = get_logger("atlas")

KEY = ["lineage", "pol_position"]

_POSITION_ALIASES = (
    "pol_position",
    "position",
    "site",
    "codon",
    "residue",
    "pol_site",
    "pol_codon_position",
    "codon_position",
)
_LINEAGE_ALIASES = ("lineage", "genotype", "clade", "subgenotype", "group")

#: Files joined into the atlas, with per-file column renames.
_UPSTREAM = [
    ("selection/entropy.tsv", {"entropy": "aa_entropy", "shannon_entropy": "aa_entropy"}),
    ("selection/dual_frame.tsv", {"consequence": "surface_consequence", "dual_frame": "surface_consequence"}),
    ("selection/genotype_specificity.tsv", {"specificity": "lineage_specific", "fst": "fst"}),
    ("selection/resistance.tsv", {"resistance": "is_resistance_site"}),
    ("fitness/dms_annotated.tsv", {}),
    ("epsilon/pol_epsilon_coevolution.tsv", {}),
]

#: Pair-indexed tables that must be reduced to one support score per position.
_PAIR_UPSTREAM = [
    ("selection/covariation.tsv", "covariation_support"),
    ("selection/epistasis.tsv", "epistasis_support"),
]


def _first_column(frame: pd.DataFrame, aliases) -> str | None:
    return next((alias for alias in aliases if alias in frame.columns), None)


def _lineage_from_model_id(model_id: str, lineages: list[str]) -> str:
    tokens = {token for token in model_id.replace("-", "_").split("_") if token}
    for lineage in lineages:
        if lineage in tokens:
            return lineage
    return "all"


#: Severity ordering for collapsing nucleotide-level dual-frame rows to codons.
_SEVERITY = {
    "rna_element": 4,
    "nonsynonymous_both": 3,
    "nonsynonymous_pol_only": 2,
    "nonsynonymous_surface_only": 2,
    "synonymous_both": 1,
    "unknown": 0,
}


def _collapse_duplicates(frame: pd.DataFrame) -> pd.DataFrame:
    """Reduce a table to one row per ``KEY`` before it is joined.

    Some upstream tables are finer-grained than the atlas key: ``dual_frame``
    has up to three nucleotide rows per Pol codon, and the resistance catalogue
    can list several substitutions at one residue.  Joining those directly would
    fan out the atlas, so numeric fields are maximised, categorical fields are
    semicolon-joined, and the dual-frame class keeps its most severe value.
    """
    if frame.empty or not frame.duplicated(KEY).any():
        return frame
    logger.info("collapsing %d duplicate %s rows", int(frame.duplicated(KEY).sum()), KEY)
    frame = frame.copy()
    has_severity = "consequence_class" in frame.columns
    if has_severity:
        frame["_severity"] = frame["consequence_class"].map(_SEVERITY).fillna(-1)

    numeric = [
        column for column in frame.columns
        if column not in KEY
        and column != "_severity"
        and pd.api.types.is_numeric_dtype(frame[column])
    ]

    def _join_unique(values: pd.Series) -> str:
        seen = {str(value) for value in values.dropna() if str(value) not in {"", "nan", "None"}}
        return ";".join(sorted(seen))

    aggregation = {}
    for column in frame.columns:
        if column in KEY or column == "_severity":
            continue
        if column in numeric:
            aggregation[column] = "max"
        elif column == "consequence_class":
            aggregation[column] = "first"  # replaced by the most severe below
        else:
            aggregation[column] = _join_unique

    collapsed = frame.groupby(KEY, as_index=False).agg(aggregation)
    if has_severity:
        worst = (
            frame.sort_values("_severity", ascending=False)
            .drop_duplicates(subset=KEY, keep="first")[KEY + ["consequence_class"]]
        )
        collapsed = collapsed.drop(columns=["consequence_class"]).merge(worst, on=KEY, how="left")
    return collapsed


def _normalise(table: pd.DataFrame, renames: dict, default_lineage: str = "all") -> pd.DataFrame | None:
    if table is None or len(table) == 0:
        return None
    frame = table.copy()
    frame = frame.rename(columns={k: v for k, v in renames.items() if k in frame.columns})

    position_col = _first_column(frame, _POSITION_ALIASES)
    if position_col is None:
        logger.warning("table has no recognisable position column; skipping")
        return None
    if position_col != "pol_position":
        frame = frame.rename(columns={position_col: "pol_position"})

    lineage_col = _first_column(frame, _LINEAGE_ALIASES)
    if lineage_col is not None and lineage_col != "lineage":
        frame = frame.rename(columns={lineage_col: "lineage"})
    elif "lineage" not in frame.columns:
        frame["lineage"] = default_lineage

    frame["pol_position"] = pd.to_numeric(frame["pol_position"], errors="coerce")
    frame = frame.dropna(subset=["pol_position"])
    if frame.empty:
        return None
    frame["pol_position"] = frame["pol_position"].astype(int)
    frame["lineage"] = frame["lineage"].astype("string").fillna(default_lineage).astype(str)
    return _collapse_duplicates(frame)


def _hinge_frame(config: dict, root: Path) -> pd.DataFrame | None:
    path = output_dir(config, root) / "structure" / "hinges.tsv"
    if not path.exists():
        return None
    try:
        hinges = read_table(path)
    except Exception as error:  # pragma: no cover - defensive
        logger.warning("could not read %s: %s", path, error)
        return None
    if not {"hinge_start", "hinge_end"}.issubset(hinges.columns):
        return None
    lineages = [str(x) for x in (get(config, "structure.lineages", []) or [])]
    rows = []
    for _, hinge in hinges.iterrows():
        lineage = _lineage_from_model_id(str(hinge.get("model_id", "")), lineages)
        start, end = int(hinge["hinge_start"]), int(hinge["hinge_end"])
        for position in range(start, end + 1):
            rows.append({"lineage": lineage, "pol_position": position, "is_hinge": True})
    if not rows:
        return None
    return pd.DataFrame(rows)


def _pair_support_frame(outroot: Path, relative: str, value_name: str) -> pd.DataFrame | None:
    """Reduce a pair-indexed table (covariation/epistasis) to per-position support.

    Pair tables are keyed by ``(position_i, position_j)``, which cannot be joined
    on the atlas key.  Each pair contributes its score to both endpoints, and the
    maximum over a position's partners is used as that position's support.
    """
    path = outroot / relative
    if not path.exists():
        return None
    try:
        table = read_table(path)
    except Exception as error:  # pragma: no cover - defensive
        logger.warning("could not read %s: %s", path, error)
        return None
    if table is None or len(table) == 0 or not {"position_i", "position_j"}.issubset(table.columns):
        return None
    if "score" not in table.columns:
        return None

    lineage = (table["lineage"].astype(str) if "lineage" in table.columns
               else pd.Series("all", index=table.index))
    score = pd.to_numeric(table["score"], errors="coerce")
    endpoints = []
    for column in ("position_i", "position_j"):
        endpoints.append(pd.DataFrame({
            "lineage": lineage,
            "pol_position": pd.to_numeric(table[column], errors="coerce"),
            value_name: score,
        }))
    long = pd.concat(endpoints, ignore_index=True).dropna(subset=["pol_position", "lineage"])
    if long.empty:
        return None
    long["pol_position"] = long["pol_position"].astype(int)
    return long.groupby(["lineage", "pol_position"], as_index=False)[value_name].max()


def _collect_frames(config: dict, root: Path) -> list[pd.DataFrame]:
    outroot = output_dir(config, root)
    frames: list[pd.DataFrame] = []
    for relative, renames in _UPSTREAM:
        path = outroot / relative
        if not path.exists():
            continue
        try:
            table = read_table(path)
        except Exception as error:  # pragma: no cover - defensive
            logger.warning("could not read %s: %s", path, error)
            continue
        normalised = _normalise(table, renames)
        if normalised is not None:
            frames.append(normalised)
    for relative, value_name in _PAIR_UPSTREAM:
        support = _pair_support_frame(outroot, relative, value_name)
        if support is not None:
            frames.append(support)
    hinges = _hinge_frame(config, root)
    if hinges is not None:
        frames.append(hinges)
    return frames


def _minmax(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").fillna(0.0)
    low, high = values.min(), values.max()
    if high > low:
        return (values - low) / (high - low)
    return pd.Series(0.0, index=values.index, dtype=float)


def _join(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame(columns=KEY)
    atlas = frames[0]
    used_columns = set(atlas.columns)
    for frame in frames[1:]:
        overlap = (set(frame.columns) & used_columns) - set(KEY)
        if overlap:
            frame = frame.drop(columns=sorted(overlap))
        atlas = atlas.merge(frame, on=KEY, how="outer")
        used_columns |= set(frame.columns)
    return atlas


def _derive(atlas: pd.DataFrame, config: dict) -> pd.DataFrame:
    frame = atlas.copy()
    if frame.empty:
        return frame
    frame["pol_position"] = pd.to_numeric(frame["pol_position"], errors="coerce").astype("Int64")

    def _domain(position):
        if pd.isna(position):
            return "unknown"
        try:
            return domain_of(int(position)).value
        except ValueError:
            return "unknown"

    # `domain` may already exist (from entropy) but only for its own rows; the
    # outer join leaves other rows blank, so backfill every missing value.
    derived_domain = frame["pol_position"].map(_domain)
    if "domain" in frame.columns:
        existing = frame["domain"].astype("object")
        frame["domain"] = existing.where(existing.notna(), derived_domain)
    else:
        frame["domain"] = derived_domain

    if "aa_entropy" in frame.columns:
        frame["conservation"] = 1.0 - _minmax(frame["aa_entropy"])

    for column in ("is_hinge", "is_interface"):
        if column in frame.columns:
            frame[column] = frame[column].fillna(False).astype(bool)
        else:
            frame[column] = False

    if "fitness" in frame.columns:
        fitness = pd.to_numeric(frame["fitness"], errors="coerce")
        if fitness.notna().any() and "is_intolerant" not in frame.columns:
            frame["is_intolerant"] = fitness <= float(fitness.quantile(0.1))
        frame["fitness"] = fitness
    if "is_intolerant" in frame.columns:
        frame["is_intolerant"] = frame["is_intolerant"].fillna(False).astype(bool)
    return frame


def _parquet_safe(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column in out.columns:
        if out[column].dtype == object:
            out[column] = out[column].astype("string")
    return out


def run(config: dict, root) -> dict[str, Path]:
    """Execute the atlas stage and return the artefact mapping."""
    root = Path(root)
    outdir = stage_dir(config, root, "atlas")

    frames = _collect_frames(config, root)
    atlas = _join(frames)
    atlas = _derive(atlas, config)

    # The atlas is defined per Pol residue; drop positions that fall outside the
    # annotated domains unless the caller explicitly wants them retained.
    if not bool(get(config, "atlas.include_unannotated", False)) and "domain" in atlas.columns:
        before = len(atlas)
        atlas = atlas[atlas["domain"] != "unknown"].reset_index(drop=True)
        if before != len(atlas):
            logger.info(
                "dropped %d atlas rows outside annotated Pol domains", before - len(atlas)
            )

    atlas = score_criteria(atlas, config)

    ranked = rank_targets(atlas, config)
    if ranked.empty:
        ranked = pd.DataFrame(columns=RANKED_TARGET_COLUMNS)

    network = interaction_network(atlas, config)
    if network.empty:
        network = pd.DataFrame(columns=INTERACTION_COLUMNS)

    switches = conformational_switch_candidates(atlas, config)
    if switches.empty:
        switches = pd.DataFrame(columns=SWITCH_COLUMNS)

    atlas_parquet = write_table(_parquet_safe(atlas), outdir / "residue_atlas.parquet")
    atlas_tsv = write_table(atlas, outdir / "residue_atlas.tsv")
    ranked_path = write_table(ranked, outdir / "ranked_targets.tsv")
    network_path = write_table(network, outdir / "conserved_interactions.tsv")
    switches_path = write_table(switches, outdir / "conformational_switches.tsv")

    report_path = build_report(atlas, ranked, config, outdir / "report.html")

    summary = {
        "n_atlas_rows": int(len(atlas)),
        "n_lineages": int(atlas["lineage"].nunique()) if "lineage" in atlas.columns and len(atlas) else 0,
        "n_candidates": int(atlas["passes_criteria"].fillna(False).astype(bool).sum())
        if "passes_criteria" in atlas.columns else 0,
        "n_ranked_targets": int(len(ranked)),
        "n_interactions": int(len(network)),
        "n_switches": int(len(switches)),
        "upstream_tables": [relative for relative, _ in _UPSTREAM
                            if (output_dir(config, root) / relative).exists()],
        "criteria": CRITERIA,
        "weights": dict(get(config, "atlas.ranked_targets.weights", {}) or {}),
        "key": KEY,
    }
    summary_path = outdir / "atlas_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return {
        "atlas_parquet": atlas_parquet,
        "atlas_tsv": atlas_tsv,
        "ranked_targets": ranked_path,
        "conserved_interactions": network_path,
        "conformational_switches": switches_path,
        "report": report_path,
        "summary": summary_path,
    }
