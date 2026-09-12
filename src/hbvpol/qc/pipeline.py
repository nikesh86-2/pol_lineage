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
from ..pipeline import (
    StageError,
    effective_threads,
    get_logger,
    have_executable,
    output_dir,
    run_command,
    stage_dir,
)
from .circularise import orient_all
from .deduplicate import dereplicate
from .orfcheck import QC_COLUMNS, check_genome

__all__ = ["run"]

logger = get_logger("qc")

#: Above this many same-length sequences, the pure-Python approximate pass is
#: abandoned (it is O(N^2) within a length bucket) in favour of exact dedup.
_APPROX_BUCKET_CAP = 2000


def _dereplicate_records(records, config, workdir: Path):
    """Dereplicate records, preferring cd-hit-est for the approximate pass.

    The pure-Python approximate dereplication compares every same-length pair
    (O(N^2)), which is intractable for thousands of genomes.  cd-hit-est is
    used when available; otherwise the approximate pass is skipped above
    ``_APPROX_BUCKET_CAP`` in favour of exact-hash dedup only.
    """
    identity = float(get(config, "qc.dereplicate_identity", 0.9999) or 0.9999)
    if identity >= 1.0 - 1e-9:
        return dereplicate(records, identity=identity), "exact"

    if have_executable("cd-hit-est"):
        in_fasta = write_fasta(records, workdir / "derep_in.fasta")
        out_fasta = workdir / "derep_out.fasta"
        try:
            run_command(
                [
                    "cd-hit-est",
                    "-i", str(in_fasta), "-o", str(out_fasta),
                    "-c", str(identity), "-n", "5", "-M", "0", "-d", "0",
                    "-T", str(effective_threads(config)),
                ],
                cwd=workdir,
                log_path=workdir / "cd-hit-est.log",
                check=True,
            )
            representative_ids = {record.id for record in read_fasta(out_fasta)}
            kept = [record for record in records if record.id in representative_ids]
            if kept:
                return kept, "cd-hit-est"
            logger.warning("cd-hit-est produced no representatives; using exact dedup")
        except Exception as error:  # pragma: no cover - defensive
            logger.warning("cd-hit-est failed (%s); using exact dedup", error)

    # Fallback: exact dedup, plus the approximate pass only when buckets are small.
    lengths: dict[int, int] = {}
    for record in records:
        lengths[len(record.seq)] = lengths.get(len(record.seq), 0) + 1
    largest_bucket = max(lengths.values(), default=0)
    if largest_bucket > _APPROX_BUCKET_CAP:
        logger.warning(
            "skipping O(N^2) approximate dereplication: %d same-length sequences "
            "exceed the %d cap and cd-hit-est is unavailable",
            largest_bucket, _APPROX_BUCKET_CAP,
        )
        return dereplicate(records, identity=1.0), "exact"
    return dereplicate(records, identity=identity), "approximate"


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
        kept, method = _dereplicate_records(passing_records, config, outdir)
        logger.info(
            "dereplication kept %d of %d passing genomes (%s)",
            len(kept), len(passing_records), method,
        )
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
