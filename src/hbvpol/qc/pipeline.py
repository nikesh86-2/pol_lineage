"""QC stage pipeline: circularise -> ORF check -> dereplicate -> write outputs.

Reads the human-HBV FASTA produced by the dataset stage, recuts every genome
at the reference origin, evaluates ORF integrity, removes duplicate genomes and
writes the inter-stage contract artefacts:

    <outroot>/qc/hbv_oriented.fasta   all genomes recut at the reference origin
    <outroot>/qc/hbv_qc_pass.tsv      QC rows that passed
    <outroot>/qc/hbv_qc_fail.tsv      QC rows that failed, with reasons
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..config import get
from ..io import read_fasta, write_fasta, write_table
from ..pipeline import StageError, get_logger, output_dir, stage_dir
from .circularise import orient_all
from .deduplicate import dereplicate
from .orfcheck import QC_COLUMNS, check_genome

__all__ = ["run"]

logger = get_logger("qc")


def run(config: dict, root) -> dict[str, Path]:
    """Execute the QC stage and return the artefact mapping."""
    root = Path(root)
    outdir = stage_dir(config, root, "qc")
    datasets = output_dir(config, root) / "datasets"

    genomes_path = datasets / "hbv_genomes.fasta"
    metadata_path = datasets / "hbv_metadata.tsv"
    if not genomes_path.exists():
        raise StageError(
            f"input FASTA not found: {genomes_path} (run the dataset stage first)"
        )

    records = read_fasta(genomes_path, source="datasets")
    logger.info("read %d genomes from %s", len(records), genomes_path)
    if metadata_path.exists():
        logger.info("metadata table present at %s", metadata_path)

    if bool(get(config, "qc.circularise", True)):
        records = orient_all(records, config)
        logger.info("circularised %d genomes at the reference origin", len(records))

    rows = [check_genome(record, config) for record in records]
    table = pd.DataFrame(rows, columns=QC_COLUMNS)

    # Only records that passed QC may flow downstream.  Writing every
    # dereplicated record (including failures) previously let short/truncated
    # genomes into alignment, epsilon extraction and tree inference.
    if table.empty:
        passing_ids: set[str] = {record.id for record in records}
    else:
        passing_ids = set(table.loc[table["qc_pass"].astype(bool), "accession"].astype(str))
    passing_records = [record for record in records if record.id in passing_ids]
    logger.info(
        "QC pass=%d fail=%d", len(passing_records), len(records) - len(passing_records)
    )

    if bool(get(config, "qc.dereplicate", True)) and passing_records:
        identity = float(get(config, "qc.dereplicate_identity", 0.9999) or 0.9999)
        kept = dereplicate(passing_records, identity=identity)
        logger.info("dereplication kept %d of %d passing genomes", len(kept), len(passing_records))
    else:
        kept = passing_records

    oriented_path = write_fasta(kept, outdir / "hbv_oriented.fasta")

    if table.empty:
        pass_table = table.copy()
        fail_table = table.copy()
    else:
        kept_ids = {record.id for record in kept}
        keep_mask = table["accession"].isin(kept_ids)
        pass_mask = table["qc_pass"].astype(bool)
        pass_table = table[keep_mask & pass_mask].reset_index(drop=True)
        fail_table = table[~pass_mask].reset_index(drop=True)

    pass_path = write_table(pass_table[QC_COLUMNS], outdir / "hbv_qc_pass.tsv")
    fail_path = write_table(fail_table[QC_COLUMNS], outdir / "hbv_qc_fail.tsv")

    return {
        "oriented_fasta": oriented_path,
        "qc_pass": pass_path,
        "qc_fail": fail_path,
    }
