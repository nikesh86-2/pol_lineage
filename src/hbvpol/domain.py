"""Pol domain model, reference coordinates and reading-frame arithmetic.

Hepatitis B virus polymerase (P) is a single ~843-residue polypeptide with four
functionally distinct regions:

    TP      terminal protein — carries the priming tyrosine (Tyr63 in genotype A2)
    spacer  a variable linker whose role is not purely passive
    RT      reverse transcriptase — the catalytic core (YMDD, motifs A–F, NJG)
    RNaseH  ribonuclease H — degrades the RNA strand of the RNA:DNA hybrid

The P open reading frame overlaps the surface (S) open reading frame in a
*shifted* frame, so a single nucleotide substitution can be synonymous for one
protein and nonsynonymous for the other.  Everything needed to reason about
that overlap is encapsulated here.

Coordinate conventions
-----------------------
* ``pol_position`` is 1-based and counts amino acids from the Pol initiator.
* ``nt_position`` uses standard HBV genome numbering (origin at the unique
  EcoRI site), so the P ORF wraps 3215 -> 1.
* ``frame_offset`` is the nucleotide offset of the overlapping frame relative
  to Pol: for the S/P overlap this is +1.  Treat the default as a documented
  assumption to be confirmed against the reference annotation, not a universal
  constant.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable

__all__ = [
    "PolDomain",
    "DomainSpan",
    "DEFAULT_DOMAIN_SPANS",
    "CANONICAL_MOTIFS",
    "domain_of",
    "translate",
    "reverse_complement",
    "revcomp_if_needed",
    "frame_indices",
    "dual_frame_consequence",
    "ConsequenceClass",
]


class PolDomain(str, Enum):
    """The four regions of the HBV polymerase polypeptide."""

    TP = "TP"
    SPACER = "spacer"
    RT = "RT"
    RNASEH = "RNaseH"

    @classmethod
    def parse(cls, value: str) -> "PolDomain":
        normalised = value.strip().lower().replace("-", "").replace("_", "")
        aliases = {
            "tp": cls.TP,
            "terminalprotein": cls.TP,
            "spacer": cls.SPACER,
            "rt": cls.RT,
            "reversetranscriptase": cls.RT,
            "rnaseh": cls.RNASEH,
        }
        if normalised not in aliases:
            raise ValueError(f"unknown Pol domain: {value!r}")
        return aliases[normalised]


@dataclass(frozen=True)
class DomainSpan:
    """An inclusive 1-based amino-acid interval within the Pol polypeptide."""

    domain: PolDomain
    start: int
    end: int

    def contains(self, position: int) -> bool:
        return self.start <= position <= self.end

    @property
    def length(self) -> int:
        return self.end - self.start + 1

    def to_dict(self) -> dict[str, object]:
        return {"domain": self.domain.value, "start": self.start, "end": self.end}


# Approximate spans for the genotype A2 reference (NC_003977, ~843 aa Pol).
# These are deliberately config-overridable: the pipeline refines them by
# lifting the reference annotation onto the curated alignment, and different
# genotypes differ in the exact spacer boundaries.  Use them as defaults only.
DEFAULT_DOMAIN_SPANS: tuple[DomainSpan, ...] = (
    DomainSpan(PolDomain.TP, 1, 183),
    DomainSpan(PolDomain.SPACER, 184, 336),
    DomainSpan(PolDomain.RT, 337, 681),
    DomainSpan(PolDomain.RNASEH, 682, 843),
)

# Canonical, well-characterised motifs that a *novel* mechanistic target should
# be distinct from.  Positions are Pol-relative defaults for genotype A2.
CANONICAL_MOTIFS: dict[str, tuple[int, int]] = {
    "priming_tyrosine": (63, 63),
    "RT_motif_A": (338, 348),
    "RT_motif_B": (407, 413),
    "RT_motif_C_YMDD": (549, 552),
    "RT_motif_D": (589, 590),
    "RT_motif_E": (620, 628),
    "RNaseH_motif_C": (737, 739),
}


def domain_of(position: int, spans: Iterable[DomainSpan] = DEFAULT_DOMAIN_SPANS) -> PolDomain:
    """Return the Pol domain containing a 1-based amino-acid position."""
    for span in spans:
        if span.contains(position):
            return span.domain
    raise ValueError(f"position {position} falls outside all Pol domains")


# --- sequence / frame utilities ---------------------------------------------

_CODON_TABLE = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}

_COMPLEMENT = str.maketrans("ACGTUNRYKMSWBDHVacgtunrykmswbdhv",
                            "TGCAANYRMKSWVHDBtgcaanyrmkswvhdb")


def translate(nt: str, frame: int = 0) -> str:
    """Translate ``nt`` in the given 0-based ``frame`` to a protein string.

    Ambiguous codons become ``X``; a trailing partial codon is ignored.
    """
    protein = []
    for i in range(frame, len(nt) - 2, 3):
        codon = nt[i:i + 3].upper().replace("U", "T")
        protein.append(_CODON_TABLE.get(codon, "X"))
    return "".join(protein)


def reverse_complement(nt: str) -> str:
    return nt.translate(_COMPLEMENT)[::-1]


def revcomp_if_needed(nt: str, strand: str = "+") -> str:
    return nt if strand in {"+", "1", 1} else reverse_complement(nt)


def frame_indices(start_nt: int, length_nt: int, genome_length: int, frame_offset: int = 0) -> list[int]:
    """Return 0-based genome indices for a coding region, wrapping the origin.

    ``start_nt`` is 1-based; ``frame_offset`` shifts the reading frame relative
    to ``start_nt`` (used to walk the overlapping surface frame).
    """
    start0 = (start_nt - 1 + frame_offset) % genome_length
    return [(start0 + i) % genome_length for i in range(length_nt)]


class ConsequenceClass(str, Enum):
    """How a nucleotide change manifests across the overlapping frames."""

    SYN_BOTH = "synonymous_both"
    NONSYN_POL = "nonsynonymous_pol_only"
    NONSYN_SURFACE = "nonsynonymous_surface_only"
    NONSYN_BOTH = "nonsynonymous_both"
    RNA_ELEMENT = "disruptive_rna_element"
    UNKNOWN = "unknown"


def dual_frame_consequence(
    ref_codon_pol: str,
    alt_codon_pol: str,
    ref_codon_surface: str,
    alt_codon_surface: str,
    disrupts_rna: bool = False,
) -> ConsequenceClass:
    """Classify a substitution across the Pol and overlapping surface frames.

    The ordering encodes priority: a hit to a known/predicted RNA element
    outranks protein-level effects, because regulatory-RNA constraints can
    dominate the visible nucleotide conservation.
    """
    if disrupts_rna:
        return ConsequenceClass.RNA_ELEMENT
    pol_syn = translate(ref_codon_pol) == translate(alt_codon_pol)
    surface_syn = translate(ref_codon_surface) == translate(alt_codon_surface)
    if pol_syn and surface_syn:
        return ConsequenceClass.SYN_BOTH
    if pol_syn and not surface_syn:
        return ConsequenceClass.NONSYN_SURFACE
    if surface_syn and not pol_syn:
        return ConsequenceClass.NONSYN_POL
    return ConsequenceClass.NONSYN_BOTH
