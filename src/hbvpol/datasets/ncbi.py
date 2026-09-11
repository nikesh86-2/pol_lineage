"""NCBI GenBank acquisition for complete/near-complete human-HBV genomes.

The public surface is deliberately small:

* :func:`fetch_genbank` runs an E-utilities ``esearch``/``efetch`` round trip and
  writes ``<accession> <definition>`` FASTA plus a tidy metadata table.
* :func:`parse_genbank_metadata` is a pure record -> mapping function, kept
  separate so it can be unit-tested without any network access.
* :func:`normalize_metadata` pins the metadata table to the documented columns
  and fills every missing qualifier with ``""`` (never ``NaN``), which keeps the
  downstream merge logic predictable.

Biopython is imported inside the fetch functions so that importing this module
(and the rest of :mod:`hbvpol.datasets`) never requires network access.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import pandas as pd

from ..config import get
from ..io import GenomeRecord, write_fasta, write_table
from ..pipeline import StageError, get_logger

__all__ = [
    "HBV_METADATA_COLUMNS",
    "fetch_genbank",
    "parse_genbank_metadata",
    "normalize_metadata",
]

#: The documented metadata columns, in their canonical order.
HBV_METADATA_COLUMNS: tuple[str, ...] = (
    "accession",
    "genotype",
    "subgenotype",
    "country",
    "year",
    "isolation_source",
    "treatment_status",
    "patient_id",
    "length",
    "source",
    "description",
)

_YEAR_RE = re.compile(r"(1[89]\d{2}|20\d{2})")


# --------------------------------------------------------------------------- #
# pure helpers
# --------------------------------------------------------------------------- #


def _first(qualifiers: Mapping[str, Any], *keys: str) -> str:
    """Return the first non-empty qualifier value among ``keys``."""
    for key in keys:
        values = qualifiers.get(key)
        if not values:
            continue
        if isinstance(values, (list, tuple)):
            for value in values:
                text = str(value).strip()
                if text:
                    return text
        else:
            text = str(values).strip()
            if text:
                return text
    return ""


def _extract_year(value: Any) -> str:
    """Pull a 4-digit year out of a free-text date, else ``""``."""
    match = _YEAR_RE.search(str(value or ""))
    return match.group(1) if match else ""


def split_genotype(value: str) -> tuple[str, str]:
    """Split a genotype string into ``(genotype, subgenotype)``.

    Handles ``"C"``, ``"C2"``, ``"genotype C2"`` and ``"subgenotype C2"``;
    anything unrecognised yields ``("", "")`` rather than raising.
    """
    text = str(value or "").strip()
    if not text:
        return "", ""
    match = re.search(r"([A-Ja-j])\s*([0-9][0-9A-Za-z]*)", text)
    if match:
        genotype = match.group(1).upper()
        return genotype, (genotype + match.group(2)).upper()
    match = re.search(r"([A-Ja-j])\b", text)
    if match:
        return match.group(1).upper(), ""
    return "", ""


def _infer_genotype_from_definition(definition: str) -> tuple[str, str]:
    """Infer genotype/subgenotype from a definition line *only* when labelled."""
    text = str(definition or "")
    match = re.search(r"genotype\s+([A-Ja-j])\s*([0-9][0-9A-Za-z]*)", text)
    if match:
        genotype = match.group(1).upper()
        return genotype, (genotype + match.group(2)).upper()
    match = re.search(r"subgenotype\s+([A-Ja-j])\s*([0-9][0-9A-Za-z]*)", text)
    if match:
        genotype = match.group(1).upper()
        return genotype, (genotype + match.group(2)).upper()
    match = re.search(r"genotype\s+([A-Ja-j])\b", text)
    if match:
        return match.group(1).upper(), ""
    return "", ""


def _record_accession(record: Any) -> str:
    for attr in ("id", "name"):
        value = getattr(record, attr, None)
        if value:
            return str(value)
    annotations = getattr(record, "annotations", {}) or {}
    accessions = annotations.get("accessions") or []
    return str(accessions[0]) if accessions else ""


def _source_qualifiers(record: Any) -> dict[str, list[str]]:
    """Collect qualifiers from every ``source`` feature of a GenBank record."""
    qualifiers: dict[str, list[str]] = {}
    for feature in getattr(record, "features", None) or []:
        if getattr(feature, "type", "") != "source":
            continue
        for key, value in (getattr(feature, "qualifiers", {}) or {}).items():
            bucket = qualifiers.setdefault(key, [])
            if isinstance(value, (list, tuple)):
                bucket.extend(str(item) for item in value)
            else:
                bucket.append(str(value))
    return qualifiers


def parse_genbank_metadata(record: Any) -> dict[str, object]:
    """Extract the documented metadata columns from a GenBank ``SeqRecord``.

    Every field is defensive: absent qualifiers become ``""`` so the returned
    mapping can be fed straight into a DataFrame without ``NaN`` churn.
    """
    qualifiers = _source_qualifiers(record)

    genotype_raw = _first(qualifiers, "genotype")
    subgenotype_raw = _first(qualifiers, "subgenotype")
    genotype, subgenotype = split_genotype(genotype_raw)
    if subgenotype_raw:
        fallback_g, fallback_sg = split_genotype(subgenotype_raw)
        genotype = genotype or fallback_g
        subgenotype = subgenotype or fallback_sg
    if not genotype:
        inferred_g, inferred_sg = _infer_genotype_from_definition(
            getattr(record, "description", "") or ""
        )
        genotype = inferred_g
        subgenotype = subgenotype or inferred_sg

    seq = getattr(record, "seq", None)
    return {
        "accession": _record_accession(record),
        "genotype": genotype,
        "subgenotype": subgenotype,
        "country": _first(qualifiers, "country", "geo_loc_name"),
        "year": _extract_year(_first(qualifiers, "collection_date", "year")),
        "isolation_source": _first(qualifiers, "isolation_source", "host"),
        "treatment_status": _first(qualifiers, "treatment_status", "treatment"),
        "patient_id": _first(qualifiers, "patient", "isolate", "strain", "specimen_voucher"),
        "length": len(str(seq)) if seq is not None else 0,
        "source": "genbank",
        "description": str(getattr(record, "description", "") or ""),
    }


def _coerce_int(value: Any) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return 0


def normalize_metadata(
    rows: Iterable[Mapping[str, Any]],
    columns: Sequence[str] = HBV_METADATA_COLUMNS,
) -> pd.DataFrame:
    """Return a DataFrame pinned to ``columns``, missing values as ``""``."""
    columns = list(columns)
    data: list[dict[str, Any]] = []
    for row in rows:
        record: dict[str, Any] = {}
        for column in columns:
            value = row.get(column, "")
            if column == "length":
                record[column] = _coerce_int(value)
            else:
                record[column] = "" if value is None else str(value)
        data.append(record)
    return pd.DataFrame(data, columns=columns)


# --------------------------------------------------------------------------- #
# network layer (lazy)
# --------------------------------------------------------------------------- #


def _iter_genbank(
    entrez: Any,
    seqio: Any,
    query: str,
    max_records: int,
    batch_size: int,
) -> Iterator[Any]:
    """Yield GenBank records for ``query``, batching ``efetch`` via history."""
    with entrez.esearch(db="nucleotide", term=query, retmax=0, usehistory="y") as handle:
        search = entrez.read(handle)
    count = int(search.get("Count", 0) or 0)
    if count == 0:
        return
    total = count if max_records <= 0 else min(count, max_records)
    webenv = search.get("WebEnv")
    query_key = search.get("QueryKey")
    for start in range(0, total, batch_size):
        fetch_n = min(batch_size, total - start)
        with entrez.efetch(
            db="nucleotide",
            rettype="gb",
            retmode="text",
            retstart=start,
            retmax=fetch_n,
            webenv=webenv,
            query_key=query_key,
        ) as handle:
            yield from seqio.parse(handle, "gb")


def fetch_genbank(
    config: Mapping[str, Any],
    out_fasta: str | Path,
    out_metadata: str | Path,
) -> tuple[Path, Path]:
    """Download human-HBV genomes from NCBI and write FASTA + metadata TSV."""
    email = get(config, "datasets.hbv.genbank.email")
    if not email:
        raise StageError(
            "datasets.hbv.genbank.email is required for NCBI E-utilities; "
            "set it in the config or pass datasets.hbv.genbank.email=you@example.org"
        )
    query = get(config, "datasets.hbv.genbank.query")
    if not query:
        raise StageError("datasets.hbv.genbank.query is required for the GenBank source")

    max_records = int(get(config, "datasets.hbv.genbank.max_records", 0) or 0)
    api_key = get(config, "datasets.hbv.genbank.api_key")
    batch_size = int(get(config, "datasets.hbv.genbank.batch_size", 500) or 500)

    from Bio import Entrez, SeqIO  # lazy: only needed when actually fetching

    Entrez.email = str(email)
    Entrez.tool = "hbvpol"
    if api_key:
        Entrez.api_key = str(api_key)

    logger = get_logger("datasets.ncbi")
    out_fasta = Path(out_fasta)
    out_metadata = Path(out_metadata)

    records: list[GenomeRecord] = []
    rows: list[dict[str, object]] = []
    try:
        for record in _iter_genbank(Entrez, SeqIO, str(query), max_records, batch_size):
            metadata = parse_genbank_metadata(record)
            accession = str(metadata["accession"]) or f"record_{len(rows)}"
            records.append(
                GenomeRecord(
                    id=accession,
                    seq=str(record.seq),
                    description=str(getattr(record, "description", "") or ""),
                    source="genbank",
                    metadata=metadata,
                )
            )
            rows.append(metadata)
    except Exception as exc:  # any network/parse error is treated as partial success
        logger.warning("GenBank retrieval interrupted after %d records: %s", len(records), exc)
        if not records:
            raise StageError(f"GenBank retrieval failed: {exc}") from exc

    logger.info("GenBank: %d genome records", len(records))
    write_fasta(records, out_fasta)
    write_table(normalize_metadata(rows), out_metadata)
    return out_fasta, out_metadata
