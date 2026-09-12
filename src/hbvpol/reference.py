"""Reference-genome feature derivation.

The pipeline ships genotype-A2 constants for the Pol/S/C ORF spans and the
Pol-to-surface frame offset.  Those are assumptions, not facts: they vary by
genotype and by accession, and a wrong surface-frame offset silently
mis-classifies every dual-frame site.

This module derives them from an *annotated* reference genome (a GenBank record
with CDS features) and emits a small JSON fragment that
:func:`hbvpol.config.load_config` merges into ``reference`` for every stage.
That single merge point is what "wires" the derived coordinates into QC,
recombination, selection, epsilon and the atlas.

Nothing here is HBV-specific beyond the gene aliases; supply any annotated
reference to make the pipeline coordinate-exact for that accession.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from .config import get
from .pipeline import StageError, get_logger

__all__ = [
    "ReferenceFeatures",
    "derive_features",
    "load_reference_record",
    "write_features",
    "load_features",
    "features_to_config",
    "fetch_reference_genbank",
    "run",
]

logger = get_logger("reference")

#: Gene aliases seen in hepadnavirus GenBank records (lower-cased).
GENE_ALIASES: dict[str, set[str]] = {
    "pol": {"p", "pol", "polymerase", "p protein"},
    "s": {"s", "surface", "hbsag", "s protein", "large s", "pre-s1/pre-s2/s"},
    "c": {"c", "core", "capsid", "hbcag", "core protein"},
    "x": {"x", "x protein", "hbx"},
}


@dataclass
class ReferenceFeatures:
    """Coordinates and frame relationships derived from a reference record."""

    accession: str
    length: int
    origin_nt: int
    pol_start_nt: int
    pol_end_nt: int
    pol_length_aa: int
    s_start_nt: int
    s_end_nt: int
    c_start_nt: int
    c_end_nt: int
    surface_frame_offset: int
    sequence_path: str = ""
    source: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ReferenceFeatures":
        fields = cls.__dataclass_fields__
        return cls(**{key: payload[key] for key in fields if key in payload})


def _cds_gene(feature) -> str | None:
    """Map a CDS feature's qualifiers onto a canonical gene name."""
    qualifiers = getattr(feature, "qualifiers", {}) or {}
    tokens: list[str] = []
    for key in ("gene", "product", "note", "standard_name"):
        for value in qualifiers.get(key, []) or []:
            tokens.extend(str(value).lower().replace("_", " ").split())
    tokens.append(" ".join(str(v).lower() for v in qualifiers.get("product", []) or []))
    for gene, aliases in GENE_ALIASES.items():
        for token in tokens:
            token = token.strip()
            if not token:
                continue
            if token in aliases or token.rstrip(".") in aliases:
                return gene
            if gene == "pol" and token in {"p", "pol"}:
                return gene
    return None


def _location_span(location, genome_length: int) -> tuple[int, int]:
    """Return the 1-based inclusive ``(start, end)`` of a (possibly wrapping) CDS.

    HBV Pol is annotated ``join(<end>..<length>,1..<start>)`` because it spans
    the origin, so the overall start is the first part's start and the overall
    end the last part's end.  Plus-strand genes are assumed (all HBV ORFs are).
    """
    parts = getattr(location, "parts", None)
    if parts:
        first, last = parts[0], parts[-1]
        return int(first.start) + 1, int(last.end)
    start = int(location.start) + 1
    end = int(location.end)
    return start, end


def derive_features(record, origin_nt: int = 1, sequence_path: str = "", source: str = "") -> ReferenceFeatures:
    """Derive coordinates and the Pol/surface frame offset from a record.

    Raises :class:`StageError` when the record lacks the Pol or surface CDS,
    because a wrong frame offset is worse than a loud failure.
    """
    genome_length = len(record.seq)
    spans: dict[str, tuple[int, int]] = {}
    translations: dict[str, str] = {}
    for feature in getattr(record, "features", []):
        if getattr(feature, "type", None) != "CDS":
            continue
        gene = _cds_gene(feature)
        if gene is None or gene in spans:
            continue
        spans[gene] = _location_span(feature.location, genome_length)
        translation = (feature.qualifiers or {}).get("translation")
        if translation:
            translations[gene] = str(translation[0])

    if "pol" not in spans or "s" not in spans:
        raise StageError(
            "reference annotation must contain CDS features for Pol and surface "
            f"(found: {sorted(spans)})"
        )

    pol_start, pol_end = spans["pol"]
    s_start, s_end = spans["s"]
    c_start, c_end = spans.get("c", (0, 0))

    pol_length_aa = len(translations["pol"]) if "pol" in translations else 0
    if pol_length_aa <= 0:
        span_nt = (genome_length - pol_start + 1) + pol_end if pol_start > pol_end else pol_end - pol_start + 1
        pol_length_aa = span_nt // 3

    # Frame offset such that the surface frame shares a reading frame with the
    # surface ORF.  The shipped default (+1) is not correct for genotype A2,
    # where the true offset is 2; deriving it is the point of this module.
    surface_frame_offset = (s_start - pol_start) % 3

    return ReferenceFeatures(
        accession=str(getattr(record, "id", "") or ""),
        length=genome_length,
        origin_nt=int(origin_nt),
        pol_start_nt=int(pol_start),
        pol_end_nt=int(pol_end),
        pol_length_aa=int(pol_length_aa),
        s_start_nt=int(s_start),
        s_end_nt=int(s_end),
        c_start_nt=int(c_start),
        c_end_nt=int(c_end),
        surface_frame_offset=int(surface_frame_offset),
        sequence_path=str(sequence_path),
        source=str(source),
    )


def load_reference_record(path: str | Path):
    """Read a single GenBank record (Biopython imported lazily)."""
    from Bio import SeqIO

    records = list(SeqIO.parse(str(path), "genbank"))
    if not records:
        raise StageError(f"no GenBank record in {path}")
    return records[0]


def features_to_config(features: ReferenceFeatures) -> dict[str, Any]:
    """Map derived features onto the ``reference`` config subtree."""
    config: dict[str, Any] = {
        "accession": features.accession or None,
        "length": features.length,
        "origin_nt": features.origin_nt,
        "pol_start_nt": features.pol_start_nt,
        "pol_end_nt": features.pol_end_nt,
        "pol_length_aa": features.pol_length_aa,
        "s_start_nt": features.s_start_nt,
        "s_end_nt": features.s_end_nt,
        "surface_frame_offset": features.surface_frame_offset,
    }
    if features.c_start_nt and features.c_end_nt:
        config["c_start_nt"] = features.c_start_nt
        config["c_end_nt"] = features.c_end_nt
    if features.sequence_path:
        # Enables reference-anchored origin detection in the QC stage.
        config["sequence"] = features.sequence_path
    return {key: value for key, value in config.items() if value is not None}


def write_features(features: ReferenceFeatures, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"reference": features_to_config(features), "features": features.to_dict()}
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out


def load_features(path: str | Path) -> ReferenceFeatures:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return ReferenceFeatures.from_dict(payload.get("features", payload))


def fetch_reference_genbank(config: Mapping[str, Any], out_gb: Path) -> Path:
    """Fetch the reference GenBank record via Entrez (network; lazy import)."""
    email = get(config, "datasets.hbv.genbank.email") or get(config, "reference.email")
    if not email:
        raise StageError(
            "an NCBI e-mail is required to fetch the reference "
            "(datasets.hbv.genbank.email), or set reference.genbank_file"
        )
    from Bio import Entrez

    Entrez.email = str(email)
    api_key = get(config, "datasets.hbv.genbank.api_key")
    if api_key:
        Entrez.api_key = str(api_key)
    accession = str(get(config, "reference.accession", "NC_003977") or "NC_003977")
    logger.info("fetching reference %s from NCBI", accession)
    handle = Entrez.efetch(db="nucleotide", id=accession, rettype="gbwithparts", retmode="text")
    out_gb.parent.mkdir(parents=True, exist_ok=True)
    out_gb.write_text(handle.read(), encoding="utf-8")
    return out_gb


def run(config: dict, root) -> dict[str, Path]:
    """Derive reference coordinates and write ``reference_features.json``.

    Reads a local GenBank file when ``reference.genbank_file`` is set, otherwise
    fetches the accession from NCBI.  The JSON is merged into ``reference`` by
    :func:`hbvpol.config.load_config`, which is what wires the derived values
    into every downstream stage.
    """
    from .io import write_fasta
    from .io.sequences import GenomeRecord
    from .pipeline import stage_dir

    root = Path(root)
    outdir = stage_dir(config, root, "reference")

    genbank_path = get(config, "reference.genbank_file")
    if genbank_path:
        gb_path = Path(str(genbank_path))
        if not gb_path.is_absolute():
            gb_path = root / gb_path
        if not gb_path.exists():
            raise StageError(f"reference.genbank_file not found: {gb_path}")
    else:
        gb_path = fetch_reference_genbank(config, outdir / "reference.gb")

    record = load_reference_record(gb_path)
    fasta_path = write_fasta(
        [GenomeRecord(id=str(record.id), seq=str(record.seq).upper(), description=str(record.description or ""))],
        outdir / "reference.fasta",
    )
    features = derive_features(
        record,
        origin_nt=int(get(config, "reference.origin_nt", 1) or 1),
        sequence_path=str(fasta_path.resolve()),
        source=str(gb_path),
    )
    features_path = write_features(features, outdir / "reference_features.json")

    logger.info(
        "reference %s: %d bp, Pol %d..%d (%d aa), surface %d..%d, frame offset %d",
        features.accession, features.length, features.pol_start_nt, features.pol_end_nt,
        features.pol_length_aa, features.s_start_nt, features.s_end_nt,
        features.surface_frame_offset,
    )
    return {"features": features_path, "fasta": fasta_path, "genbank": Path(gb_path)}
