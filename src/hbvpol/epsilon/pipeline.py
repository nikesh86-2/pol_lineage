"""Epsilon RNA / Pol coevolution stage pipeline.

extract epsilon per genome -> fold -> coevolve TP vs epsilon -> cross-genotype
compatibility -> write.

Outputs (``<outroot>/epsilon/``):

    epsilon_sequences.fasta
    epsilon_folds/<id>.dotbracket
    epsilon_folds/<id>.json
    pol_epsilon_coevolution.tsv
    compatibility.tsv
    epsilon_summary.json

Every upstream input is optional: with no oriented genomes or no Pol alignment
the stage still writes correctly-schemad (possibly empty) outputs.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd

from ..config import get
from ..io import GenomeRecord, read_fasta, read_table, write_fasta, write_table
from ..pipeline import get_logger, output_dir, stage_dir
from .coevolve import COEVOLUTION_COLUMNS, compatibility_matrix, pol_epsilon_coevolution
from .fold import extract_epsilon, fold_epsilon

__all__ = ["run"]

logger = get_logger("epsilon")

_GENOTYPE_RE = re.compile(r"genotype[ _-]?([A-HJ])\b", re.IGNORECASE)


def _consensus(sequences: list[str]) -> str:
    if not sequences:
        return ""
    width = min(len(seq) for seq in sequences)
    out = []
    for index in range(width):
        counts = Counter(seq[index] for seq in sequences if seq[index] not in "-.")
        out.append(counts.most_common(1)[0][0] if counts else "-")
    return "".join(out)


def _load_genotypes(config: dict, root: Path) -> dict[str, str]:
    """Map sequence id -> genotype from QC/metadata tables, if available."""
    outroot = output_dir(config, root)
    candidates = [
        outroot / "qc" / "hbv_qc_pass.tsv",
        outroot / "datasets" / "hbv_metadata.tsv",
        outroot / "qc" / "hbv_qc_fail.tsv",
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            table = read_table(path)
        except Exception as error:  # pragma: no cover - defensive
            logger.warning("could not read %s: %s", path, error)
            continue
        genotype_col = next((c for c in ("genotype", "genotype_id", "clade") if c in table.columns), None)
        id_col = next(
            (c for c in ("accession", "id", "accession_id", "isolate", "sequence_id") if c in table.columns),
            None,
        )
        if genotype_col and id_col:
            mapping = {
                str(row[id_col]): str(row[genotype_col])
                for _, row in table.iterrows()
                if pd.notna(row[genotype_col])
            }
            if mapping:
                logger.info("loaded %d genotype assignments from %s", len(mapping), path)
                return mapping
    return {}


def _genotype_of(record: GenomeRecord, genotype_map: dict[str, str]) -> str:
    if record.id in genotype_map:
        return genotype_map[record.id]
    match = _GENOTYPE_RE.search(record.description or "")
    if match:
        return match.group(1).upper()
    return "unknown"


def _genotype_consensus(records: list[GenomeRecord], genotype_map: dict[str, str], kind: str) -> dict[str, str]:
    grouped: dict[str, list[str]] = {}
    for record in records:
        genotype = _genotype_of(record, genotype_map)
        grouped.setdefault(genotype, []).append(record.seq)
    result = {genotype: _consensus(seqs) for genotype, seqs in grouped.items() if seqs}
    if not result:
        logger.warning("no %s sequences available for compatibility analysis", kind)
    return result


def _locate_pol_alignment(config: dict, root: Path) -> Path | None:
    outroot = output_dir(config, root)
    for relative in ("recombination/domain_subalns/TP.fasta", "recombination/hbv_aligned.fasta"):
        candidate = outroot / relative
        if candidate.exists():
            return candidate
    return None


def run(config: dict, root) -> dict[str, Path]:
    """Execute the epsilon stage and return the artefact mapping."""
    root = Path(root)
    outdir = stage_dir(config, root, "epsilon")
    outroot = output_dir(config, root)

    oriented = outroot / "qc" / "hbv_oriented.fasta"
    if oriented.exists():
        genomes = read_fasta(oriented, source="qc")
    else:
        logger.warning("no oriented genomes at %s; writing empty epsilon outputs", oriented)
        genomes = []

    genotype_map = _load_genotypes(config, root)

    fold_dir = outdir / "epsilon_folds"
    fold_dir.mkdir(parents=True, exist_ok=True)
    epsilon_records: list[GenomeRecord] = []
    methods: Counter[str] = Counter()
    for record in genomes:
        epsilon = extract_epsilon(record, config, genotype=_genotype_of(record, genotype_map))
        if not epsilon:
            continue
        epsilon_records.append(GenomeRecord(id=record.id, seq=epsilon, description="epsilon"))
        fold = fold_epsilon(epsilon, config)
        methods[str(fold["method"])] += 1
        (fold_dir / f"{record.id}.dotbracket").write_text(
            fold["dotbracket"] + "\n", encoding="utf-8"
        )
        (fold_dir / f"{record.id}.json").write_text(
            json.dumps(fold, indent=2), encoding="utf-8"
        )

    epsilon_fasta = write_fasta(epsilon_records, outdir / "epsilon_sequences.fasta")

    pol_alignment = _locate_pol_alignment(config, root)
    if pol_alignment is None:
        logger.warning("no Pol alignment available; coevolution table will be empty")
        coevolution = pd.DataFrame(columns=COEVOLUTION_COLUMNS)
    else:
        coevolution = pol_epsilon_coevolution(pol_alignment, epsilon_records, config)
        if coevolution.empty:
            logger.warning("coevolution produced no rows (no shared ids between alignments?)")
    coevolution_path = write_table(coevolution, outdir / "pol_epsilon_coevolution.tsv")

    pol_records = read_fasta(pol_alignment) if pol_alignment else []
    pol_map = _genotype_consensus(pol_records, genotype_map, "Pol")
    eps_map = _genotype_consensus(epsilon_records, genotype_map, "epsilon")
    compatibility = compatibility_matrix(pol_map, eps_map, config)
    compatibility_path = write_table(compatibility, outdir / "compatibility.tsv")

    summary = {
        "n_genomes": int(len(genomes)),
        "n_epsilon": int(len(epsilon_records)),
        "fold_methods": dict(methods),
        "pol_alignment": str(pol_alignment) if pol_alignment else None,
        "n_coevolution_rows": int(len(coevolution)),
        "n_genotypes": len(set(pol_map) | set(eps_map)),
        "genotypes": sorted(set(pol_map) | set(eps_map)),
        "n_compatibility_pairs": int(len(compatibility)),
        "compatibility_predictor": str(
            get(config, "epsilon.compatibility.predictor", "contact_map_energy")
        ),
    }
    summary_path = outdir / "epsilon_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return {
        "epsilon_sequences": epsilon_fasta,
        "epsilon_folds": fold_dir,
        "coevolution": coevolution_path,
        "compatibility": compatibility_path,
        "summary": summary_path,
    }
