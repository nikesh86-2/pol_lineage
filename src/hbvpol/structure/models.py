"""Structural model manifest construction and predictor dispatch.

The stage enumerates a *planned* ensemble — every combination of state,
lineage, predictor and seed — plus any explicitly listed mutants — and records
where each model *would* live.  Building the manifest never touches the
network or a structural predictor, so it is safe to run offline and inside
tests.

Actual prediction is delegated to :func:`run_predictor`, which only ever calls
an external tool when the corresponding executable is found on ``PATH``.  When
nothing is available the function logs and returns ``None`` so the pipeline can
degrade gracefully.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from ..config import get
from ..domain import DEFAULT_DOMAIN_SPANS
from ..io import GenomeRecord, read_fasta, read_table
from ..pipeline import StageError, get_logger, have_executable, output_dir, run_command

__all__ = [
    "MANIFEST_COLUMNS",
    "build_model_manifest",
    "sequence_for_lineage",
    "run_predictor",
]

logger = get_logger("structure.models")

#: Tidy schema of ``model_manifest.tsv``.
MANIFEST_COLUMNS = [
    "model_id",
    "kind",
    "state",
    "lineage",
    "strategy",
    "seed",
    "variant",
    "rel_path",
    "sequence_source",
]

# Default executables per predictor.  These can be overridden with
# ``structure.<strategy>.executable`` (e.g. ``structure.colabfold.executable``).
_DEFAULT_EXECUTABLES = {
    "colabfold": "colabfold_batch",
    "alphafold3": "run_alphafold",
    "esmfold": "esm-fold",
}

_PREDICTOR_EXTENSIONS = (".pdb", ".ent", ".cif", ".mmcif")


def _root(config: dict, root=None) -> Path:
    """Resolve the project root, honouring an explicit value first."""
    if root is not None:
        return Path(root)
    injected = get(config, "_root")
    return Path(injected) if injected else Path(".")


def _resolve(config: dict, dotted: str, root: Path) -> Path:
    value = get(config, dotted)
    if value is None:
        return Path("")
    path = Path(str(value))
    return path if path.is_absolute() else root / path


def _tokens(text: str) -> set[str]:
    return {tok for tok in re.split(r"[^0-9A-Za-z]+", str(text).lower()) if tok}


def _pick_record(records: list[GenomeRecord], lineage: str) -> GenomeRecord | None:
    """Prefer a record whose id explicitly names the lineage."""
    if not records:
        return None
    want = lineage.lower()
    for record in records:
        if want in _tokens(record.id):
            return record
    return records[0]


def build_model_manifest(config: dict, root=None) -> pd.DataFrame:
    """Enumerate the planned structural ensemble as a tidy DataFrame.

    Rows cover the Cartesian product of ``structure.states`` x
    ``structure.lineages`` x ``structure.strategies`` x ``seeds`` plus any
    mutants listed in ``structure.mutants_file`` when that file exists.
    """
    states = list(get(config, "structure.states", ["apo"]) or ["apo"])
    lineages = list(get(config, "structure.lineages", ["A"]) or ["A"])
    strategies = list(get(config, "structure.strategies", ["colabfold"]) or ["colabfold"])
    seeds = int(get(config, "structure.seeds", 1) or 1)

    rows: list[dict[str, object]] = []
    for state in states:
        for lineage in lineages:
            for strategy in strategies:
                for seed in range(1, seeds + 1):
                    model_id = f"{state}__{lineage}__{strategy}__seed{seed}"
                    rows.append({
                        "model_id": model_id,
                        "kind": "state",
                        "state": state,
                        "lineage": lineage,
                        "strategy": strategy,
                        "seed": seed,
                        "variant": "",
                        "rel_path": f"models/{state}/{lineage}/{strategy}_seed{seed}.pdb",
                        "sequence_source": "",
                    })

    mutants_path = _resolve(config, "structure.mutants_file", _root(config, root))
    if mutants_path and mutants_path.exists():
        try:
            mutants = read_table(mutants_path)
        except Exception as error:  # pragma: no cover - defensive
            logger.warning("could not read mutants file %s: %s", mutants_path, error)
            mutants = pd.DataFrame()
        for _, row in mutants.iterrows():
            variant = str(
                row.get("mutant")
                or row.get("variant")
                or row.get("name")
                or row.iloc[0]
            )
            lineage = str(
                row.get("lineage") or row.get("genotype") or (lineages[0] if lineages else "unknown")
            )
            state = str(row.get("state") or "mutant")
            strategy = str(row.get("strategy") or (strategies[0] if strategies else "colabfold"))
            model_id = f"mutant__{lineage}__{variant}__seed1"
            rows.append({
                "model_id": model_id,
                "kind": "mutant",
                "state": state,
                "lineage": lineage,
                "strategy": strategy,
                "seed": 1,
                "variant": variant,
                "rel_path": f"models/{state}/{lineage}/mutant_{variant}_seed1.pdb",
                "sequence_source": str(mutants_path),
            })
    elif get(config, "structure.mutants_file"):
        logger.warning("mutants file %s not found; manifest will omit mutants", mutants_path)

    return pd.DataFrame(rows, columns=MANIFEST_COLUMNS)


def sequence_for_lineage(lineage: str, config: dict, root=None) -> GenomeRecord | None:
    """Return the consensus/ancestral Pol sequence for a lineage, or ``None``.

    Sources are tried in order:

    1. ``phylogeny/ancestral`` — an ``ancestral_<lineage>`` file, or a file
       containing a record whose id names the lineage.
    2. ``recombination/domain_subalns`` — per-domain subalignments, whose
       consensus/first records are concatenated in domain order (TP, spacer,
       RT, RNaseH) to reconstruct the full Pol coding sequence.

    When nothing can be resolved the function logs a warning and returns
    ``None``; callers are expected to skip that lineage rather than fail.
    """
    outroot = output_dir(config, _root(config, root))
    search_dirs = [
        outroot / "phylogeny" / "ancestral",
        outroot / "recombination" / "domain_subalns",
    ]
    want = lineage.lower()

    for directory in search_dirs:
        if not directory.is_dir():
            continue
        for fasta in sorted(directory.glob("*.fasta")) + sorted(directory.glob("*.fa")):
            if want not in _tokens(fasta.stem):
                continue
            record = _pick_record(read_fasta(fasta), lineage)
            if record is not None:
                record.source = str(fasta)
                return record
        # Second pass: match on record ids rather than file names.
        for fasta in sorted(directory.glob("*.fasta")) + sorted(directory.glob("*.fa")):
            records = read_fasta(fasta)
            for record in records:
                if want in _tokens(record.id):
                    record.source = str(fasta)
                    return record

    # Fallback: concatenate per-domain subalignments in domain order.
    domain_dir = outroot / "recombination" / "domain_subalns"
    if domain_dir.is_dir():
        chunks: list[str] = []
        for span in DEFAULT_DOMAIN_SPANS:
            fasta = domain_dir / f"{span.domain.value}.fasta"
            if not fasta.exists():
                chunks = []
                break
            record = _pick_record(read_fasta(fasta), lineage)
            if record is None:
                chunks = []
                break
            chunks.append(record.seq)
        if chunks:
            return GenomeRecord(
                id=lineage,
                seq="".join(chunks),
                source=str(domain_dir),
                description="concatenated domain subalignments",
            )

    logger.warning("no consensus/ancestral Pol sequence found for lineage %r", lineage)
    return None


def _predictor_plan(strategy: str, fasta: Path, out_dir: Path, config: dict):
    """Return ``(executable, argv)`` for a predictor, or ``(None, None)``."""
    executable = get(config, f"structure.{strategy}.executable", _DEFAULT_EXECUTABLES.get(strategy))
    if strategy == "colabfold":
        return executable or "colabfold_batch", [executable or "colabfold_batch", str(fasta), str(out_dir)]
    if strategy == "alphafold3":
        exe = executable or "run_alphafold"
        return exe, [exe, f"--fasta_paths={fasta}", f"--output_dir={out_dir}"]
    if strategy == "esmfold":
        exe = executable or "esm-fold"
        return exe, [exe, "-i", str(fasta), "-o", str(out_dir)]
    return None, None


def _find_model(out_dir: Path) -> Path | None:
    candidates: list[Path] = []
    for pattern in ("*.pdb", "*.ent", "*.cif", "*.mmcif"):
        candidates.extend(sorted(out_dir.rglob(pattern)))
    return candidates[0] if candidates else None


def run_predictor(record: GenomeRecord, out_dir, config: dict, strategy: str | None = None):
    """Run the first available structural predictor for ``record``.

    Returns the path to the first produced model file, or ``None`` when no
    predictor executable is available (or every predictor fails).  This function
    is offline-safe: with no predictors installed it emits a warning and returns.
    """
    # Absolute: run_command sets cwd=out_dir, so relative predictor paths would
    # be re-rooted under it and the predictor could not open them.
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    fasta = out_dir / f"{record.id}.fasta"
    fasta.write_text(record.to_fasta(), encoding="utf-8")

    strategies = [strategy] if strategy else list(
        get(config, "structure.strategies", ["colabfold"]) or ["colabfold"]
    )
    for name in strategies:
        executable, argv = _predictor_plan(name, fasta, out_dir, config)
        if executable is None:
            logger.warning("unknown structural predictor %r; skipping", name)
            continue
        if not have_executable(executable):
            logger.warning(
                "predictor %r executable %r not on PATH; skipping", name, executable
            )
            continue
        log_path = out_dir / f"{record.id}.{name}.log"
        try:
            run_command(argv, cwd=out_dir, log_path=log_path, check=True)
        except StageError as error:
            logger.warning("predictor %r failed for %s: %s", name, record.id, error)
            continue
        produced = _find_model(out_dir)
        if produced is not None:
            return produced
        logger.warning("predictor %r produced no model file for %s", name, record.id)
    return None
