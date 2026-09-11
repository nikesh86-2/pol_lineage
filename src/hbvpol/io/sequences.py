"""FASTA handling built around a light ``GenomeRecord`` container.

A purpose-built parser is used in preference to ``Bio.SeqIO`` on the hot path
because the pipeline routinely streams tens of thousands of genomes; the
Biopython path is still exercised by the tests for validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

__all__ = [
    "GenomeRecord",
    "read_fasta",
    "write_fasta",
    "rotate_to_origin",
    "ambiguous_fraction",
    "find_orfs",
]

_AMBIGUOUS = set("NRYKMSWBDHVnrykmswbdhv")

_AA_COMPLEMENT_IRRELEVANT = str.maketrans(
    "ACGTUNRYKMSWBDHVacgtunrykmswbdhv",
    "TGCAANYRMKSWVHDBtgcaanyrmkswvhdb",
)


@dataclass
class GenomeRecord:
    """A nucleotide sequence plus its provenance and metadata."""

    id: str
    seq: str
    description: str = ""
    source: str = ""
    strand: str = "+"
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def length(self) -> int:
        return len(self.seq)

    @property
    def uppercase(self) -> str:
        return self.seq.upper()

    def to_fasta(self, wrap: int = 70) -> str:
        header = self.id if not self.description else f"{self.id} {self.description}"
        if wrap <= 0:
            return f">{header}\n{self.seq}\n"
        lines = [self.seq[i:i + wrap] for i in range(0, len(self.seq), wrap)]
        return f">{header}\n" + "\n".join(lines) + "\n"


def _iter_fasta(path: str | Path) -> Iterator[tuple[str, str]]:
    header: str | None = None
    chunks: list[str] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(chunks)
                header = line[1:].strip()
                chunks = []
            else:
                chunks.append(line.strip())
    if header is not None:
        yield header, "".join(chunks)


def read_fasta(path: str | Path, source: str = "") -> list[GenomeRecord]:
    """Read a FASTA file into ``GenomeRecord`` objects (first word is the id)."""
    records: list[GenomeRecord] = []
    for header, seq in _iter_fasta(path):
        ident, _, description = header.partition(" ")
        records.append(GenomeRecord(id=ident, seq=seq, description=description, source=source))
    return records


def write_fasta(records: Iterable[GenomeRecord], path: str | Path, wrap: int = 70) -> Path:
    """Write records to ``path``; returns the path written."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(record.to_fasta(wrap=wrap))
    return out


def rotate_to_origin(seq: str, origin_nt: int) -> str:
    """Circularly rotate ``seq`` so that 1-based ``origin_nt`` becomes index 0.

    Genomes are supplied in arbitrary orientations by different databases;
    recutting them all at a common origin is a prerequisite for comparing
    nucleotide positions and for aligning overlapping frames.
    """
    if not seq:
        return seq
    shift = (origin_nt - 1) % len(seq)
    return seq[shift:] + seq[:shift]


def ambiguous_fraction(seq: str) -> float:
    if not seq:
        return 0.0
    return sum(1 for ch in seq if ch in _AMBIGUOUS) / len(seq)


def find_orfs(seq: str, min_aa: int = 30, both_strands: bool = False) -> list[dict[str, object]]:
    """Very light ORF finder returning coordinates and lengths.

    This is intentionally simple: the pipeline uses canonical HBV ORF
    coordinates mapped from the reference rather than *de novo* prediction,
    but a permissive ORF scan is a useful sanity check for integrity.
    """
    from ..domain import translate

    strands = ("+", "-") if both_strands else ("+",)
    hits: list[dict[str, object]] = []
    for strand in strands:
        seq_strand = seq if strand == "+" else seq.translate(_AA_COMPLEMENT_IRRELEVANT)[::-1]
        for frame in range(3):
            protein = translate(seq_strand, frame=frame)
            start = 0
            for i, aa in enumerate(protein):
                if aa == "*":
                    length = i - start
                    if length >= min_aa:
                        hits.append({
                            "strand": strand,
                            "frame": frame,
                            "start_nt": start * 3 + frame + 1,
                            "length_aa": length,
                        })
                    start = i + 1
            length = len(protein) - start
            if length >= min_aa:
                hits.append({
                    "strand": strand,
                    "frame": frame,
                    "start_nt": start * 3 + frame + 1,
                    "length_aa": length,
                })
    return sorted(hits, key=lambda hit: hit["length_aa"], reverse=True)
