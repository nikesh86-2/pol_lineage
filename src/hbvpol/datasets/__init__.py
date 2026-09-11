"""Stage 1 — dataset acquisition.

Two datasets are assembled here:

* **human HBV** — GenBank as the sequence-of-record, HBVdb as an independent
  cross-check, and HBV-GLUE's maintained alignments when a local installation is
  available;
* **deep hepadnaviruses** — Pol protein sequences spanning the hepadnavirus host
  range, optionally with nackednaviruses and RT outgroups, aligned for the
  conservation analyses.

All network and external-tool access is lazy (inside functions), so importing any
of these modules works offline and without MAFFT/triMAL/HBV-GLUE installed.
"""

from .deephep import (
    align_sequences,
    build_deephep_dataset,
    fetch_group_sequences,
    filter_by_length,
)
from .glue import fetch_glue_alignments, verify_glue_install
from .hbvdb import check_release_freshness, fetch_hbvdb, parse_hbvdb_record
from .ncbi import (
    HBV_METADATA_COLUMNS,
    fetch_genbank,
    normalize_metadata,
    parse_genbank_metadata,
)
from .pipeline import run

__all__ = [
    "run",
    "HBV_METADATA_COLUMNS",
    "fetch_genbank",
    "parse_genbank_metadata",
    "normalize_metadata",
    "fetch_hbvdb",
    "parse_hbvdb_record",
    "check_release_freshness",
    "fetch_glue_alignments",
    "verify_glue_install",
    "build_deephep_dataset",
    "fetch_group_sequences",
    "filter_by_length",
    "align_sequences",
]
