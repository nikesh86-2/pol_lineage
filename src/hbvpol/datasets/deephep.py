"""Deep hepadnavirus / nackednavirus Pol dataset.

This stage builds a broad Pol protein alignment spanning the hepadnavirus host
range (primate, rodent, bat, avian, reptile, amphibian, fish), optionally adding
fish nackednaviruses and reverse-transcriptase outgroups.  The alignment is the
reference frame for the conservation half of the study, so it is kept separate
from the human-HBV dataset.

Retrieval, alignment and trimming are all *pluggable*:

* :func:`fetch_group_sequences` owns the only network access and can be replaced
  with a local-data loader in tests or air-gapped runs;
* :func:`align_sequences` degrades gracefully to the unaligned FASTA when MAFFT
  is unavailable or fails;
* trimming (triMAL) is applied only when configured *and* present on ``PATH``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, cast

from ..config import get
from ..io import GenomeRecord, write_fasta
from ..pipeline import (
    StageError,
    get_logger,
    have_executable,
    require_executable,
    run_command,
    stage_dir,
)

__all__ = [
    "run",
    "build_deephep_dataset",
    "fetch_group_sequences",
    "fetch_manifest_sequences",
    "load_reference_manifest",
    "manifest_provenance",
    "probe_group_queries",
    "filter_by_length",
    "align_sequences",
    "maybe_trim_alignment",
]

#: Taxon queries for the configured host groups.  These are working defaults —
#: they are overridable per group via ``deephep.group_queries``.
#:
#: Note: ``[Host]`` qualifiers are *not* indexed in the NCBI protein database
#: (``Hepadnaviridae[Organism] AND Rodentia[Host]`` returns zero), so the
#: queries below target the actual virus taxa.  Reptile and amphibian
#: hepadnaviruses have no NCBI protein records at all, so those groups
#: legitimately return nothing.
DEFAULT_GROUP_QUERIES: dict[str, object] = {
    "primate": 'Hepadnaviridae[Organism] AND (primates[Host] OR Homo sapiens[Host])',
    "rodent": (
        '(Woodchuck hepatitis virus[Organism] OR Ground squirrel hepatitis virus[Organism]'
        ' OR "Arctic ground squirrel hepatitis virus"[Organism])'
    ),
    "bat": '(Bat hepatitis B virus[Organism] OR Bat hepadnavirus[Organism])',
    "avian": "Avihepadnavirus[Organism]",
    "reptile": (
        "(Hepadnaviridae[Organism] AND (Testudines[All Fields] OR Squamata[All Fields]"
        " OR Crocodilia[All Fields]))"
    ),
    "amphibian": (
        "(Hepadnaviridae[Organism] AND (Anura[All Fields] OR Caudata[All Fields]"
        " OR Gymnophiona[All Fields]))"
    ),
    "fish": (
        '(Slender scalyhead hepatitis B virus[Organism] OR Scaly rockcod hepatitis B virus[Organism]'
        ' OR (Hepadnaviridae[Organism] AND fish[All Fields] NOT "Hepatitis B virus"[Organism]))'
    ),
    # Nackednaviruses are indexed by protein title, not as an organism taxon, so
    # this query embeds its own Pol restriction ("polymerase" present means the
    # generic suffix below is not appended).
    "nackednavirus": (
        'nackednavirus[Title] AND (polymerase[Title] OR "reverse transcriptase"[Title]'
        ' OR P[Title] OR ORF2[Title] OR "RNA-dependent RNA-polymerase"[Title])'
    ),
}

#: Appended to taxon queries so that only Pol-family proteins are retrieved.
#: Override per group with ``deephep.group_query_suffixes`` (empty = none).
DEFAULT_QUERY_SUFFIX = 'AND (polymerase[Title] OR "reverse transcriptase"[Title])'


# --------------------------------------------------------------------------- #
# pure filtering helpers
# --------------------------------------------------------------------------- #


def filter_by_length(
    records: Iterable[GenomeRecord],
    min_length: int,
    max_length: int,
) -> list[GenomeRecord]:
    """Keep records whose sequence length is within the inclusive bounds."""
    return [record for record in records if min_length <= len(record.seq) <= max_length]


def _dedupe(records: Iterable[GenomeRecord]) -> list[GenomeRecord]:
    seen: set[str] = set()
    unique: list[GenomeRecord] = []
    for record in records:
        if record.id in seen:
            continue
        seen.add(record.id)
        unique.append(record)
    return unique


def _group_query(group: str, config: Mapping[str, Any]) -> str:
    """Resolve the Entrez query for a taxonomic group or outgroup name.

    A group value may be a string or a list of alternative terms (OR-joined).
    The Pol-restriction suffix is appended unless the base query already
    mentions a polymerase, and can be overridden per group via
    ``deephep.group_query_suffixes`` (an empty string disables it).
    """
    overrides = get(config, "deephep.group_queries", {}) or {}
    if group in overrides:
        base: object = overrides[group]
    elif group in DEFAULT_GROUP_QUERIES:
        base = DEFAULT_GROUP_QUERIES[group]
    else:
        base = group  # free-text term, e.g. a plant pararetrovirus family

    if isinstance(base, (list, tuple)):
        base = "(" + " OR ".join(str(item) for item in base) + ")"
    base = str(base)

    suffixes = get(config, "deephep.group_query_suffixes", {}) or {}
    if group in suffixes:
        suffix = str(suffixes[group] or "")
    else:
        suffix = get(config, "deephep.query_suffix", DEFAULT_QUERY_SUFFIX) or ""
    if suffix and "polymerase" not in base.lower():
        return f"({base}) {suffix}"
    return base


# --------------------------------------------------------------------------- #
# retrieval (network is confined here)
# --------------------------------------------------------------------------- #


def _iter_protein_batches(entrez: Any, seqio: Any, ids: list[str], batch_size: int):
    """Yield protein FASTA records for ``ids`` in ``efetch`` batches."""
    for start in range(0, len(ids), batch_size):
        chunk = ids[start : start + batch_size]
        with entrez.efetch(
            db="protein", id=",".join(chunk), rettype="fasta", retmode="text"
        ) as handle:
            yield from seqio.parse(handle, "fasta")


def fetch_group_sequences(group: str, config: Mapping[str, Any]) -> list[GenomeRecord]:
    """Retrieve Pol protein sequences for one taxonomic group via NCBI.

    ``group`` is a key in :data:`DEFAULT_GROUP_QUERIES` (primate, rodent, bat,
    avian, reptile, amphibian, fish, nackednavirus) or an arbitrary free-text
    term used for outgroups.  All network access lives in this function.
    """
    from Bio import Entrez, SeqIO  # lazy: importing the module needs no network

    email = get(config, "datasets.hbv.genbank.email") or get(config, "deephep.email")
    if not email:
        raise StageError(
            "deep hepadnavirus retrieval requires datasets.hbv.genbank.email "
            "(or deephep.email) to be set for NCBI E-utilities"
        )

    Entrez.email = str(email)
    Entrez.tool = "hbvpol"
    api_key = get(config, "datasets.hbv.genbank.api_key")
    if api_key:
        Entrez.api_key = str(api_key)

    query = _group_query(group, config)
    limit = int(get(config, "deephep.max_per_group", 500) or 500)
    batch_size = int(get(config, "deephep.fetch_batch_size", 200) or 200)
    logger = get_logger("datasets.deephep")

    with Entrez.esearch(db="protein", term=query, retmax=limit, idtype="acc") as handle:
        search = cast(dict[str, Any], Entrez.read(handle))
    ids = [str(item) for item in search.get("IdList", [])]
    if not ids:
        # Log the exact query: a zero here is usually a bad taxon name, not an
        # absence of data.
        logger.warning("deephep: no protein hits for %s (query: %s)", group, query)
        return []

    records: list[GenomeRecord] = []
    for record in _iter_protein_batches(Entrez, SeqIO, ids, batch_size):
        seq = str(record.seq).upper()
        records.append(
            GenomeRecord(
                id=str(record.id),
                seq=seq,
                description=str(getattr(record, "description", "") or ""),
                source=f"deephep:{group}",
                metadata={"group": group},
            )
        )
    logger.info("deephep: %s -> %d protein sequences", group, len(records))
    return records


# --------------------------------------------------------------------------- #
# curated deep-reference manifest (version-controlled)
# --------------------------------------------------------------------------- #

#: Columns of ``config/deep_hepadnavirus_references.tsv``.
REFERENCE_MANIFEST_COLUMNS = (
    "accession", "display_name", "genus", "host_class", "host_species",
    "sequence_origin", "expected_complete", "include_pol_tree",
    "include_codon_analysis", "include_structure_analysis",
    "source_publication", "notes",
)

#: Honest provenance categories.  Endogenous elements and assembly-derived
#: candidates must never be pooled silently with extant viral genomes.
SEQUENCE_ORIGINS = (
    "exogenous_complete",
    "exogenous_partial",
    "assembly_derived_complete",
    "assembly_derived_partial",
    "endogenous_complete_or_near_complete",
    "endogenous_fragment",
    "metagenomic_unverified",
)

_TRUTHY = {"yes", "true", "1", "y"}
_TRUTHY_CONDITIONAL = _TRUTHY | {"conditional"}


def load_reference_manifest(path: str | Path) -> list[dict[str, str]]:
    """Parse the curated deep-reference manifest TSV into rows.

    The manifest is the publication dataset: database queries discover
    candidates, but only rows here are guaranteed to be retrieved.  Blank
    accessions are skipped; unknown columns are preserved.
    """
    import csv

    manifest_path = Path(path)
    if not manifest_path.exists():
        return []
    rows: list[dict[str, str]] = []
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            accession = str(row.get("accession", "") or "").strip()
            if not accession:
                continue
            rows.append({str(key): str(value or "").strip() for key, value in row.items()})
    return rows


def _flag(value: str, conditional: bool = False) -> bool:
    allowed = _TRUTHY_CONDITIONAL if conditional else _TRUTHY
    return str(value).strip().lower() in allowed


def _pol_from_genbank(record: Any, entry: Mapping[str, str]) -> tuple[str | None, str]:
    """Extract the Pol protein from an annotated GenBank record.

    Prefers the deposited ``translation`` qualifier (exact); otherwise
    translates the CDS location.  Recognises ``polymerase``/``reverse
    transcriptase`` products, the ``P``/``P-protein`` gene names of the avian
    and orthohepadnaviruses, and finally falls back to the longest CDS
    (polymerase is the largest hepadnaviral protein), recording which route was
    used so the provenance table stays honest.

    Returns ``(protein, how)`` where ``how`` is ``annotated``,
    ``longest_cds_fallback`` or ``none``.
    """
    from ..domain import translate

    candidates: list[tuple[int, str]] = []
    for feature in getattr(record, "features", []):
        if getattr(feature, "type", None) != "CDS":
            continue
        quals = feature.qualifiers
        gene = " ".join(quals.get("gene", [])).lower().strip()
        product = " ".join(quals.get("product", [])).lower()
        translation = (quals.get("translation") or [""])[0]
        protein = str(translation).upper() if translation else ""
        if not protein:
            try:
                nt = str(feature.extract(record.seq)).upper()
            except Exception:  # pragma: no cover - malformed location
                continue
            protein = translate(nt, frame=0).rstrip("*")
        if not protein:
            continue
        looks_like_pol = (
            "polymerase" in product
            or "reverse transcriptase" in product
            or "p-protein" in product
            or "p protein" in product
            or gene in {"pol", "p", "polymerase"}
        )
        if looks_like_pol:
            return protein, "annotated"
        candidates.append((len(protein), protein))
    if candidates:
        candidates.sort(reverse=True)
        return candidates[0][1], "longest_cds_fallback"
    return None, "none"


def fetch_manifest_sequences(config: Mapping[str, Any]) -> list[GenomeRecord]:
    """Retrieve Pol proteins for the curated reference manifest.

    Each accession is fetched as a GenBank nucleotide record and its polymerase
    CDS is extracted.  Per-accession failures are logged and skipped, so a single
    stale accession cannot abort the deep dataset.
    """
    from Bio import Entrez, SeqIO  # lazy: no network at import time

    logger = get_logger("datasets.deephep")
    manifest_path = get(config, "deephep.references_file")
    if not manifest_path:
        return []
    entries = load_reference_manifest(str(manifest_path))
    if not entries:
        logger.warning("deephep: reference manifest %s is empty or missing", manifest_path)
        return []

    email = get(config, "datasets.hbv.genbank.email") or get(config, "deephep.email")
    if not email:
        raise StageError(
            "the deep-reference manifest requires datasets.hbv.genbank.email "
            "(or deephep.email) for NCBI E-utilities"
        )
    Entrez.email = str(email)
    Entrez.tool = "hbvpol"
    api_key = get(config, "datasets.hbv.genbank.api_key")
    if api_key:
        Entrez.api_key = str(api_key)

    records: list[GenomeRecord] = []
    for entry in entries:
        if not _flag(entry.get("include_pol_tree", ""), conditional=True):
            continue
        accession = entry["accession"]
        try:
            with Entrez.efetch(db="nucleotide", id=accession, rettype="gb", retmode="text") as handle:
                record = SeqIO.read(handle, "genbank")
        except Exception as exc:  # noqa: BLE001 - per-accession resilience
            logger.warning("deephep: manifest accession %s could not be fetched (%s)", accession, exc)
            continue
        protein, how = _pol_from_genbank(record, entry)
        if not protein:
            logger.warning("deephep: no polymerase CDS found for manifest accession %s", accession)
            continue
        if how != "annotated":
            logger.warning(
                "deephep: %s has no annotated polymerase CDS; using the longest CDS (%d aa)",
                accession, len(protein),
            )
        records.append(
            GenomeRecord(
                id=accession,
                seq=protein,
                description=entry.get("display_name", ""),
                source="deephep:reference_manifest",
                metadata={
                    "group": "reference_manifest",
                    "accession": accession,
                    "pol_annotation": how,
                    "display_name": entry.get("display_name", ""),
                    "genus": entry.get("genus", ""),
                    "host_class": entry.get("host_class", ""),
                    "host_species": entry.get("host_species", ""),
                    "sequence_origin": entry.get("sequence_origin", ""),
                    "include_codon_analysis": entry.get("include_codon_analysis", ""),
                    "include_structure_analysis": entry.get("include_structure_analysis", ""),
                    "source_publication": entry.get("source_publication", ""),
                },
            )
        )
    logger.info("deephep: reference manifest -> %d Pol sequences", len(records))
    return records


def manifest_provenance(records: Iterable[GenomeRecord]) -> list[dict[str, str]]:
    """Tidy provenance rows for manifest-sourced records (the audit table)."""
    rows: list[dict[str, str]] = []
    for record in records:
        metadata = dict(record.metadata or {})
        if metadata.get("group") != "reference_manifest":
            continue
        rows.append({
            "accession": metadata.get("accession", record.id),
            "display_name": metadata.get("display_name", ""),
            "genus": metadata.get("genus", ""),
            "host_class": metadata.get("host_class", ""),
            "host_species": metadata.get("host_species", ""),
            "sequence_origin": metadata.get("sequence_origin", ""),
            "length": str(len(record.seq)),
            "pol_annotation": metadata.get("pol_annotation", ""),
            "source_publication": metadata.get("source_publication", ""),
            "include_codon_analysis": metadata.get("include_codon_analysis", ""),
            "include_structure_analysis": metadata.get("include_structure_analysis", ""),
        })
    return rows


def _write_provenance(path: Path, records: Iterable[GenomeRecord]) -> None:
    """Write the manifest provenance audit table (TSV)."""
    import csv

    columns = [
        "accession", "display_name", "genus", "host_class", "host_species",
        "sequence_origin", "length", "pol_annotation", "source_publication",
        "include_codon_analysis", "include_structure_analysis",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(columns)
        for row in manifest_provenance(records):
            writer.writerow([row.get(column, "") for column in columns])


# --------------------------------------------------------------------------- #
# alignment / trimming
# --------------------------------------------------------------------------- #


def _copy_unaligned(in_fasta: Path, out_fasta: Path) -> Path:
    out_fasta.parent.mkdir(parents=True, exist_ok=True)
    text = in_fasta.read_text(encoding="utf-8") if in_fasta.exists() else ""
    out_fasta.write_text(text, encoding="utf-8")
    return out_fasta


def _format_tool_log(argv: list[str], result: Any) -> str:
    header = f"# command: {' '.join(argv)}\n# exit: {result.returncode}\n\n"
    stdout = getattr(result, "stdout", "") or ""
    stderr = getattr(result, "stderr", "") or ""
    return f"{header}{stdout}\n{stderr}"


def align_sequences(in_fasta: str | Path, out_fasta: str | Path, config: Mapping[str, Any]) -> Path:
    """Align ``in_fasta`` to ``out_fasta``, falling back to the unaligned copy.

    Only MAFFT is currently wired up.  Absence of the executable, a non-zero
    exit, or empty output all result in the unaligned sequences being copied and
    a warning being logged, so the stage never hard-fails on tooling.
    """
    in_fasta = Path(in_fasta)
    out_fasta = Path(out_fasta)
    logger = get_logger("datasets.deephep")

    aligner = str(get(config, "deephep.aligner", "") or "").strip().lower()
    if not aligner or aligner in {"none", "false", "unaligned", "off"}:
        logger.warning("deephep: no aligner configured; writing unaligned sequences")
        return _copy_unaligned(in_fasta, out_fasta)

    if not aligner.startswith("mafft"):
        logger.warning("deephep: unsupported aligner %r; writing unaligned sequences", aligner)
        return _copy_unaligned(in_fasta, out_fasta)

    try:
        executable = require_executable(
            "mafft", hint="install MAFFT or set deephep.aligner to 'none'"
        )
    except StageError as exc:
        logger.warning("deephep: %s; writing unaligned sequences", exc)
        return _copy_unaligned(in_fasta, out_fasta)

    argv = [executable]
    if "linsi" in aligner:
        argv += ["--localpair", "--maxiterate", "1000"]
    elif "einsi" in aligner:
        argv += ["--genafpair", "--maxiterate", "1000"]
    elif "ginsi" in aligner:
        argv += ["--globalpair", "--maxiterate", "1000"]
    else:
        argv += ["--auto"]
    argv += ["--quiet", str(in_fasta)]

    log_path = out_fasta.with_suffix(".log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    result = run_command(argv, check=False)
    log_path.write_text(_format_tool_log(argv, result), encoding="utf-8")

    if result.returncode != 0 or not (result.stdout or "").strip():
        logger.warning(
            "deephep: MAFFT failed (exit %s); writing unaligned sequences (see %s)",
            result.returncode,
            log_path,
        )
        return _copy_unaligned(in_fasta, out_fasta)

    out_fasta.parent.mkdir(parents=True, exist_ok=True)
    out_fasta.write_text(result.stdout, encoding="utf-8")
    logger.info("deephep: MAFFT alignment written to %s (log: %s)", out_fasta, log_path)
    return out_fasta


def maybe_trim_alignment(path: str | Path, config: Mapping[str, Any]) -> Path:
    """Trim an alignment in place with triMAL when configured and available."""
    path = Path(path)
    logger = get_logger("datasets.deephep")

    requested = [str(item).strip().lower() for item in (get(config, "deephep.trim", []) or [])]
    if "trimal" not in requested and "trimal-automated1" not in requested:
        return path
    if not have_executable("trimal"):
        logger.warning("deephep: triMAL requested but not on PATH; keeping untrimmed alignment")
        return path

    trimmed = path.with_suffix(".trim.fasta")
    result = run_command(
        ["trimal", "-in", str(path), "-out", str(trimmed), "-automated1"],
        check=False,
    )
    if result.returncode != 0 or not trimmed.exists() or trimmed.stat().st_size == 0:
        logger.warning("deephep: triMAL failed; keeping untrimmed alignment")
        return path
    trimmed.replace(path)
    logger.info("deephep: triMAL trimming applied to %s", path)
    return path


# --------------------------------------------------------------------------- #
# stage entry points
# --------------------------------------------------------------------------- #


def _collect_terms(config: Mapping[str, Any]) -> list[str]:
    terms: list[str] = [str(g) for g in (get(config, "deephep.taxonomic_groups", []) or [])]
    if get(config, "deephep.include_nackednavirus", False):
        terms.append("nackednavirus")
    if get(config, "deephep.include_rt_outgroups", False):
        terms.extend(str(item) for item in (get(config, "deephep.outgroups", []) or []))
    return terms


def probe_group_queries(config: Mapping[str, Any]) -> dict[str, int]:
    """Return the esearch hit count for every configured group query.

    A zero count almost always means the taxon name is wrong rather than that
    the data are absent, so this is the first thing to run when a group comes
    back empty.  Network access is confined here.
    """
    from Bio import Entrez

    email = get(config, "datasets.hbv.genbank.email") or get(config, "deephep.email")
    if not email:
        raise StageError(
            "probing deep-hepadnavirus queries requires datasets.hbv.genbank.email "
            "(or deephep.email) for NCBI E-utilities"
        )
    Entrez.email = str(email)
    Entrez.tool = "hbvpol"

    counts: dict[str, int] = {}
    for term in _collect_terms(config):
        with Entrez.esearch(db="protein", term=_group_query(term, config), retmax=0) as handle:
            payload = cast(dict[str, Any], Entrez.read(handle))
        counts[term] = int(payload.get("Count", 0) or 0)
    return counts


def build_deephep_dataset(
    config: Mapping[str, Any],
    out_fasta: str | Path,
    out_alignment: str | Path,
) -> dict[str, Path]:
    """Build the deep-hepadnavirus Pol FASTA and its alignment.

    Partial retrieval failures are logged per group and do not abort the stage.
    """
    logger = get_logger("datasets.deephep")
    out_fasta = Path(out_fasta)
    out_alignment = Path(out_alignment)

    terms: list[str] = _collect_terms(config)

    collected: list[GenomeRecord] = []
    counts: dict[str, int] = {}
    for term in terms:
        try:
            group_records = fetch_group_sequences(term, config)
        except Exception as exc:  # noqa: BLE001 - one bad group must not kill the stage
            logger.warning("deephep: could not retrieve %s (%s)", term, exc)
            group_records = []
        counts[term] = len(group_records)
        collected.extend(group_records)

    # Curated, version-controlled references: database queries discover
    # candidates, but the manifest controls the publication dataset.
    manifest_records: list[GenomeRecord] = []
    if get(config, "deephep.references_file"):
        try:
            manifest_records = fetch_manifest_sequences(config)
        except Exception as exc:  # noqa: BLE001 - optional source
            logger.warning("deephep: reference manifest unavailable (%s)", exc)
            manifest_records = []
        counts["reference_manifest"] = len(manifest_records)
        collected.extend(manifest_records)

    unique = _dedupe(collected)
    min_length = int(get(config, "deephep.min_pol_length", 0) or 0)
    max_length = int(get(config, "deephep.max_pol_length", 10**9) or 10**9)
    filtered = filter_by_length(unique, min_length, max_length)
    logger.info(
        "deephep: retrieved %d sequences (%s), %d pass length filter %d-%d",
        len(unique),
        counts,
        len(filtered),
        min_length,
        max_length,
    )

    write_fasta(filtered, out_fasta)

    provenance_path = out_fasta.parent / "deephep_provenance.tsv"
    _write_provenance(provenance_path, manifest_records)
    result = {
        "deephep_pol": out_fasta,
        "deephep_provenance": provenance_path,
    }
    if not filtered:
        out_alignment.parent.mkdir(parents=True, exist_ok=True)
        out_alignment.write_text("", encoding="utf-8")
        result["deephep_alignment"] = out_alignment
        return result

    aligned = align_sequences(out_fasta, out_alignment, config)
    aligned = maybe_trim_alignment(aligned, config)
    result["deephep_alignment"] = aligned
    return result


def run(config: Mapping[str, Any], root: str | Path) -> dict[str, Path]:
    """Package entry point: build the deep hepadnavirus dataset under the stage dir."""
    stage = stage_dir(config, root, "datasets")
    return build_deephep_dataset(
        config,
        out_fasta=stage / "deephep_pol.fasta",
        out_alignment=stage / "deephep_alignment.fasta",
    )
