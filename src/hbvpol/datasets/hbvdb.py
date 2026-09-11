"""HBVdb cross-check source.

HBVdb is an independently curated HBV database.  We use it as a second opinion
on genotype/subgenotype assignment rather than as the sequence-of-record.

.. warning::
   **Documented assumption.**  The exact HBVdb REST interface could not be
   verified from an offline environment.  All HTTP access is therefore isolated
   in :class:`HbvdbHttpAdapter`, whose endpoint constants
   (``_SEARCH_PATH``, ``_PARAM_*``) are the *only* place that needs to change if
   the real API differs.  Everything else — record parsing, output layout,
   freshness checking — is independent of the wire format.

Network failures are non-fatal: the fetch functions log a warning and return
empty outputs so the stage can continue with the GenBank sequence-of-record.
"""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from ..config import get
from ..io import GenomeRecord, write_fasta, write_table
from ..pipeline import get_logger
from .ncbi import (
    _extract_year,
    _infer_genotype_from_definition,
    normalize_metadata,
    split_genotype,
)

__all__ = [
    "HbvdbHttpAdapter",
    "parse_hbvdb_record",
    "fetch_hbvdb",
    "check_release_freshness",
]

_ACCESSION_KEYS = ("accession", "accession_number", "genbank_accession", "id", "name", "hbvdb_id")
_SEQUENCE_KEYS = ("sequence", "seq", "nucleotide_sequence", "genome")
_DESCRIPTION_KEYS = ("description", "definition", "title", "isolate_name")

_DATE_FORMATS = ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%Y/%m/%d", "%Y-%m", "%B %Y", "%b %Y", "%Y")


class HbvdbHttpAdapter:
    """Minimal JSON-over-HTTP adapter for the HBVdb REST endpoint.

    The endpoint shape is a *documented assumption* (see the module docstring);
    subclasses may override the constants or :meth:`search` to match the real
    service without touching the rest of the stage.
    """

    _SEARCH_PATH = "/api/search"
    _PARAM_QUERY = "query"
    _PARAM_LIMIT = "limit"
    _PARAM_FORMAT = "format"
    _FORMAT_JSON = "json"

    def __init__(self, base_url: str, timeout: float = 30.0) -> None:
        self.base_url = str(base_url).rstrip("/")
        self.timeout = float(timeout)

    def _url(self, term: str, limit: int) -> str:
        params = {self._PARAM_QUERY: term, self._PARAM_FORMAT: self._FORMAT_JSON}
        if limit > 0:
            params[self._PARAM_LIMIT] = str(limit)
        return f"{self.base_url}{self._SEARCH_PATH}?{urlencode(params)}"

    def search(self, term: str, limit: int = 0) -> list[dict[str, Any]]:
        """Return raw HBVdb records for a free-text search term."""
        request = Request(
            self._url(term, limit),
            headers={"Accept": "application/json", "User-Agent": "hbvpol/0.1"},
        )
        with urlopen(request, timeout=self.timeout) as response:  # configured URL
            payload = response.read().decode("utf-8", "replace")
        return _coerce_payload(payload)


def _coerce_payload(payload: str) -> list[dict[str, Any]]:
    """Normalise a JSON payload (list or wrapper object) into a record list."""
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError(f"HBVdb returned a non-JSON payload: {payload[:120]!r}") from exc
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        for key in ("records", "results", "data", "hits"):
            value = data.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    raise ValueError("HBVdb JSON payload had no recognisable record list")


def _pick(raw: Mapping[str, Any], keys: tuple[str, ...]) -> str:
    lowered = {str(key).lower(): value for key, value in raw.items()}
    for key in keys:
        if key.lower() not in lowered:
            continue
        value = lowered[key.lower()]
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            value = value[0] if value else ""
        text = str(value).strip()
        if text:
            return text
    return ""


def parse_hbvdb_record(raw: Mapping[str, Any]) -> GenomeRecord:
    """Turn a raw HBVdb record into a :class:`GenomeRecord`.

    Pure and total: unknown keys are ignored and missing ones become ``""``, so
    this is safe to call on whatever the adapter happens to return.
    """
    record = dict(raw or {})
    accession = _pick(record, _ACCESSION_KEYS)
    sequence = _pick(record, _SEQUENCE_KEYS)
    description = _pick(record, _DESCRIPTION_KEYS)

    genotype, subgenotype = split_genotype(_pick(record, ("genotype", "genotype_group")))
    subgenotype_raw = _pick(record, ("subgenotype", "sub_genotype"))
    if subgenotype_raw:
        fallback_g, fallback_sg = split_genotype(subgenotype_raw)
        genotype = genotype or fallback_g
        subgenotype = subgenotype or fallback_sg
    if not genotype:
        inferred_g, inferred_sg = _infer_genotype_from_definition(description)
        genotype = inferred_g
        subgenotype = subgenotype or inferred_sg

    metadata: dict[str, object] = {
        "accession": accession,
        "genotype": genotype,
        "subgenotype": subgenotype,
        "country": _pick(record, ("country", "geo_loc_name", "location")),
        "year": _extract_year(_pick(record, ("year", "collection_date", "date", "isolation_year"))),
        "isolation_source": _pick(record, ("isolation_source", "source", "host")),
        "treatment_status": _pick(record, ("treatment_status", "treatment")),
        "patient_id": _pick(record, ("patient_id", "patient", "isolate")),
        "length": len(sequence),
        "source": "hbvdb",
        "description": description,
    }
    return GenomeRecord(
        id=accession or description or "hbvdb_record",
        seq=sequence,
        description=description,
        source="hbvdb",
        metadata=metadata,
    )


def _write_empty(out_fasta: Path, out_metadata: Path) -> None:
    write_fasta([], out_fasta)
    write_table(normalize_metadata([]), out_metadata)


def fetch_hbvdb(
    config: Mapping[str, Any],
    out_fasta: str | Path,
    out_metadata: str | Path,
) -> tuple[Path, Path]:
    """Fetch HBVdb records, writing FASTA + metadata TSV.

    On any network/parse failure this logs and writes empty (well-formed)
    outputs, because HBVdb is only a cross-check source.
    """
    logger = get_logger("datasets.hbvdb")
    out_fasta = Path(out_fasta)
    out_metadata = Path(out_metadata)

    base_url = get(config, "datasets.hbv.hbvdb.base_url", "")
    if not base_url:
        logger.warning("HBVdb: no datasets.hbv.hbvdb.base_url configured; skipping")
        _write_empty(out_fasta, out_metadata)
        return out_fasta, out_metadata

    term = get(config, "datasets.hbv.hbvdb.query") or get(
        config, "datasets.hbv.genbank.query", "Hepatitis B virus complete genome"
    )
    limit = int(get(config, "datasets.hbv.hbvdb.max_records", 0) or 0)
    timeout = float(get(config, "datasets.hbv.hbvdb.timeout", 30) or 30)

    adapter = HbvdbHttpAdapter(str(base_url), timeout=timeout)
    try:
        raw_records = adapter.search(str(term), limit)
    except Exception as exc:  # noqa: BLE001 - network failure is explicitly non-fatal
        logger.warning("HBVdb request failed (%s); continuing without it", exc)
        _write_empty(out_fasta, out_metadata)
        return out_fasta, out_metadata

    records = [parse_hbvdb_record(item) for item in raw_records]
    records = [record for record in records if record.seq]
    logger.info("HBVdb: %d genome records", len(records))
    write_fasta(records, out_fasta)
    write_table(normalize_metadata(record.metadata for record in records), out_metadata)
    return out_fasta, out_metadata


def _parse_date(value: str) -> date | None:
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(value.strip(), fmt).date()
        except ValueError:
            continue
    return None


def check_release_freshness(config: Mapping[str, Any]) -> tuple[bool, str]:
    """Report whether the declared HBVdb release is within the allowed age.

    Returns ``(is_fresh, message)``.  A missing or unparseable release is
    treated as *not* fresh (and logged) so that a curated cross-check cannot be
    silently trusted.
    """
    logger = get_logger("datasets.hbvdb")
    release = get(config, "datasets.hbv.hbvdb.release") or get(
        config, "datasets.hbv.hbvdb.release_date"
    )
    max_age = float(get(config, "datasets.hbv.hbvdb.max_release_age_days", 730) or 730)

    if not release:
        message = "HBVdb release date not declared; freshness cannot be verified"
        logger.warning(message)
        return False, message

    released = _parse_date(str(release))
    if released is None:
        message = f"could not parse HBVdb release date {release!r}"
        logger.warning(message)
        return False, message

    age_days = (date.today() - released).days
    if age_days > max_age:
        message = f"HBVdb release {release!r} is {age_days} days old (> {max_age:.0f} days)"
        logger.warning(message)
        return False, message
    return True, f"HBVdb release {release!r} is {age_days} days old"
