"""Selection stage pipeline (Stage 3).

Computes, per lineage, the multi-level evolutionary-constraint tables:

    <outroot>/selection/entropy.tsv
    <outroot>/selection/dual_frame.tsv
    <outroot>/selection/selection_sites.tsv
    <outroot>/selection/covariation.tsv
    <outroot>/selection/epistasis.tsv
    <outroot>/selection/genotype_specificity.tsv
    <outroot>/selection/resistance.tsv
    <outroot>/selection/selection_summary.json

Lineages come from the genotype column of ``datasets/hbv_metadata.tsv`` (or a
single ``all`` lineage when no genotype metadata is available).  The pure-Python
entropy / dual-frame / covariation / epistasis / genotype analyses never depend
on HyPhy; per-site dN/dS-style methods from ``selection.methods`` are delegated
to HyPhy *only* when it is installed, and otherwise produce schema-correct empty
frames with a warning.

Every upstream input is optional: a missing alignment logs a warning and yields
empty-but-correctly-schemed outputs rather than raising.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ..config import get
from ..io import GenomeRecord, read_fasta, read_table, write_table
from ..pipeline import get_logger, have_executable, output_dir, run_command, stage_dir
from .covariation import COVARIATION_COLUMNS, EPISTASIS_COLUMNS, covarying_pairs, epistasis_pairs
from .dualframe import DUAL_FRAME_COLUMNS, coerce_alignment, dual_frame_table
from .entropy import ENTROPY_COLUMNS, per_position_entropy
from .genotype import GENOTYPE_COLUMNS, genotype_specificity
from .resistance import RESISTANCE_COLUMNS, resistance_table

__all__ = ["run", "run_hyphy_method", "SELECTION_SITE_COLUMNS"]

logger = get_logger("selection")

SELECTION_SITE_COLUMNS = ["lineage", "pol_position", "method", "statistic", "pvalue", "significant"]
_SITE_BASE_COLUMNS = [column for column in SELECTION_SITE_COLUMNS if column != "lineage"]

_LINEAGE_COLUMN_CANDIDATES = ("genotype", "genotype_group", "lineage")
_ID_COLUMN_CANDIDATES = ("accession", "isolate", "id", "sequence_id", "name", "strain", "seq_id")

_FALLBACK_ALIGNMENTS = (
    ("qc", "hbv_oriented.fasta"),
    ("datasets", "hbv_genomes.fasta"),
    ("datasets", "deephep_alignment.fasta"),
)


# --------------------------------------------------------------------------- #
# HyPhy delegation
# --------------------------------------------------------------------------- #
def _parse_hyphy_sites(payload: dict, method: str) -> list[dict]:
    """Best-effort extraction of per-site results from a HyPhy JSON payload.

    Handles the ``MLE -> content -> {site: {...}}`` shape used by FEL/MEME and
    similar per-site methods.  Global tests (e.g. BUSTED) legitimately return no
    per-site rows.
    """
    rows: list[dict] = []
    mle = payload.get("MLE") or payload.get("mle") or {}
    content = mle.get("content") if isinstance(mle, dict) else None
    if not isinstance(content, dict):
        return rows
    for site_key, site in content.items():
        if not isinstance(site, dict):
            continue
        try:
            position = int(site_key) + 1
        except (TypeError, ValueError):
            continue
        pvalue = site.get("p-value", site.get("pvalue", site.get("p_val")))
        statistic = site.get("beta", site.get("alpha", site.get("omega")))
        try:
            pvalue_value = float(pvalue)
        except (TypeError, ValueError):
            pvalue_value = 1.0
        try:
            statistic_value = float(statistic)
        except (TypeError, ValueError):
            statistic_value = float("nan")
        rows.append({
            "pol_position": position,
            "method": method,
            "statistic": statistic_value,
            "pvalue": pvalue_value,
            "significant": bool(pvalue_value < 0.05),
        })
    return rows


def run_hyphy_method(alignment, tree, method: str, config) -> pd.DataFrame:
    """Run a HyPhy per-site selection method and return a tidy frame.

    Returns columns ``pol_position, method, statistic, pvalue, significant``
    (the pipeline adds ``lineage``).  HyPhy (and ``subprocess``) are only
    touched when the executable is actually present; otherwise — or on any
    failure — a schema-correct empty frame is returned with a warning.
    """
    if not have_executable("hyphy"):
        logger.warning("hyphy not found; %s selection sites will be empty", method)
        return pd.DataFrame(columns=_SITE_BASE_COLUMNS)
    if tree is None or not Path(tree).exists():
        logger.warning("no tree supplied for hyphy %s; selection sites will be empty", method)
        return pd.DataFrame(columns=_SITE_BASE_COLUMNS)

    workdir = Path(tree).parent
    outdir = workdir / "hyphy"
    outdir.mkdir(parents=True, exist_ok=True)

    records = coerce_alignment(alignment)
    if not records:
        return pd.DataFrame(columns=_SITE_BASE_COLUMNS)
    alignment_path = workdir / f"selection_{method}.fasta"
    alignment_path.write_text(
        "".join(f">{record.id}\n{record.seq}\n" for record in records), encoding="utf-8"
    )
    output_path = outdir / f"{method}.json"

    try:
        run_command(
            [
                "hyphy", str(method),
                "--alignment", str(alignment_path),
                "--tree", str(tree),
                "--output", str(output_path),
            ],
            cwd=outdir,
            log_path=outdir / f"{method}.log",
            check=True,
        )
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        rows = _parse_hyphy_sites(payload, str(method))
    except Exception as error:
        logger.warning("hyphy %s failed (%s); writing empty selection sites", method, error)
        return pd.DataFrame(columns=_SITE_BASE_COLUMNS)

    if not rows:
        return pd.DataFrame(columns=_SITE_BASE_COLUMNS)
    return pd.DataFrame(rows, columns=_SITE_BASE_COLUMNS)


# --------------------------------------------------------------------------- #
# Input discovery
# --------------------------------------------------------------------------- #
def _load_alignment(outroot: Path) -> list[GenomeRecord]:
    for stage, filename in _FALLBACK_ALIGNMENTS:
        candidate = outroot / stage / filename
        if candidate.exists():
            logger.info("selection input alignment: %s", candidate)
            return read_fasta(candidate)
    logger.warning("no upstream alignment found under %s; writing empty selection tables", outroot)
    return []


def _load_metadata(outroot: Path) -> pd.DataFrame:
    path = outroot / "datasets" / "hbv_metadata.tsv"
    if not path.exists():
        logger.warning("no metadata table at %s; treating data as a single 'all' lineage", path)
        return pd.DataFrame()
    try:
        return read_table(path)
    except Exception as error:  # pragma: no cover - defensive
        logger.warning("could not read metadata %s (%s)", path, error)
        return pd.DataFrame()


def _pick_column(frame: pd.DataFrame, candidates) -> str | None:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    return None


def _lineages(metadata: pd.DataFrame) -> list[str]:
    if metadata is None or len(metadata) == 0:
        return ["all"]
    lineage_column = _pick_column(metadata, _LINEAGE_COLUMN_CANDIDATES)
    if lineage_column is None:
        return ["all"]
    values = [
        str(value).strip()
        for value in metadata[lineage_column].dropna().unique()
        if str(value).strip() and str(value).strip().lower() not in {"nan", "none", "na"}
    ]
    return sorted(set(values)) or ["all"]


def _subset_by_lineage(
    records: list[GenomeRecord], metadata: pd.DataFrame, lineages: list[str]
) -> dict[str, list[GenomeRecord]]:
    if lineages == ["all"] or metadata is None or len(metadata) == 0:
        return {"all": records}

    lineage_column = _pick_column(metadata, _LINEAGE_COLUMN_CANDIDATES)
    id_column = _pick_column(metadata, _ID_COLUMN_CANDIDATES)
    if lineage_column is None:
        return {"all": records}

    genotype_of: dict[str, str] = {}
    for _, row in metadata.iterrows():
        key = str(row[id_column]) if id_column is not None else str(row.name)
        genotype_of[key] = str(row[lineage_column]).strip()

    subsets: dict[str, list[GenomeRecord]] = {}
    for lineage in lineages:
        subsets[lineage] = [record for record in records if genotype_of.get(record.id) == lineage]
    return subsets


def _find_tree(outroot: Path) -> Path | None:
    candidate = outroot / "phylogeny" / "genotype_tree.treefile"
    return candidate if candidate.exists() else None


def _empty(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)


def _with_lineage(frame: pd.DataFrame, lineage: str, columns: list[str]) -> pd.DataFrame:
    if frame is None or len(frame) == 0:
        result = _empty(columns)
    else:
        result = frame.copy()
        if "lineage" not in result.columns:
            result.insert(0, "lineage", lineage)
    return result.reindex(columns=columns)


def _concat(frames: list[pd.DataFrame], columns: list[str]) -> pd.DataFrame:
    non_empty = [frame for frame in frames if frame is not None and len(frame) > 0]
    if not non_empty:
        return _empty(columns)
    return pd.concat(non_empty, ignore_index=True).reindex(columns=columns)


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
def run(config: dict, root) -> dict[str, Path]:
    """Execute the selection stage and return the artefact mapping."""
    root = Path(root)
    outdir = stage_dir(config, root, "selection")
    outroot = output_dir(config, root)

    records = _load_alignment(outroot)
    metadata = _load_metadata(outroot)
    lineages = _lineages(metadata)
    subsets = _subset_by_lineage(records, metadata, lineages)
    tree = _find_tree(outroot)

    methods = get(config, "selection.methods", []) or []
    if isinstance(methods, str):
        methods = [methods]

    entropy_frames: list[pd.DataFrame] = []
    dual_frames: list[pd.DataFrame] = []
    site_frames: list[pd.DataFrame] = []
    covariation_frames: list[pd.DataFrame] = []
    epistasis_frames: list[pd.DataFrame] = []

    for lineage in lineages:
        subset = subsets.get(lineage, [])
        entropy_frames.append(
            _with_lineage(per_position_entropy(subset, config), lineage, ENTROPY_COLUMNS)
        )
        dual_frames.append(
            _with_lineage(dual_frame_table(subset, config), lineage, DUAL_FRAME_COLUMNS)
        )
        covariation_frames.append(
            _with_lineage(covarying_pairs(subset, config), lineage, COVARIATION_COLUMNS)
        )
        epistasis_frames.append(
            _with_lineage(epistasis_pairs(subset, config), lineage, EPISTASIS_COLUMNS)
        )
        for method in methods:
            site_frames.append(
                _with_lineage(run_hyphy_method(subset, tree, str(method), config), lineage, SELECTION_SITE_COLUMNS)
            )
        logger.info("selection: lineage %r has %d sequences", lineage, len(subset))

    entropy = _concat(entropy_frames, ENTROPY_COLUMNS)
    dual_frame = _concat(dual_frames, DUAL_FRAME_COLUMNS)
    selection_sites = _concat(site_frames, SELECTION_SITE_COLUMNS)
    covariation = _concat(covariation_frames, COVARIATION_COLUMNS)
    epistasis = _concat(epistasis_frames, EPISTASIS_COLUMNS)

    genotype = genotype_specificity(records, metadata, config)
    if genotype is None or len(genotype) == 0:
        genotype = _empty(GENOTYPE_COLUMNS)
    else:
        genotype = genotype.reindex(columns=GENOTYPE_COLUMNS)

    resistance_base = resistance_table(config, root)
    resistance_frames = [
        _with_lineage(resistance_base, lineage, RESISTANCE_COLUMNS) for lineage in lineages
    ]
    resistance = _concat(resistance_frames, RESISTANCE_COLUMNS)

    paths = {
        "entropy": write_table(entropy, outdir / "entropy.tsv"),
        "dual_frame": write_table(dual_frame, outdir / "dual_frame.tsv"),
        "selection_sites": write_table(selection_sites, outdir / "selection_sites.tsv"),
        "covariation": write_table(covariation, outdir / "covariation.tsv"),
        "epistasis": write_table(epistasis, outdir / "epistasis.tsv"),
        "genotype_specificity": write_table(genotype, outdir / "genotype_specificity.tsv"),
        "resistance": write_table(resistance, outdir / "resistance.tsv"),
    }

    summary = {
        "lineages": lineages,
        "n_sequences": len(records),
        "methods": [str(method) for method in methods],
        "hyphy_available": have_executable("hyphy"),
        "tree": str(tree) if tree is not None else None,
        "n_entropy": int(len(entropy)),
        "n_dual_frame": int(len(dual_frame)),
        "n_selection_sites": int(len(selection_sites)),
        "n_covariation": int(len(covariation)),
        "n_epistasis": int(len(epistasis)),
        "n_genotype_sites": int(len(genotype)),
        "n_resistance": int(len(resistance)),
    }
    summary_path = outdir / "selection_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    paths["summary"] = summary_path

    logger.info(
        "selection: %d lineages, %d entropy rows, %d dual-frame rows, %d genotype sites",
        len(lineages), len(entropy), len(dual_frame), len(genotype),
    )
    return paths
