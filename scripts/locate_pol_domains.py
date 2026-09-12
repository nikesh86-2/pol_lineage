#!/usr/bin/env python3
"""locate_pol_domains.py -- calibrate HBV Pol domain boundaries per genotype.

Why this exists
---------------
``DEFAULT_DOMAIN_SPANS`` (TP 1-183, spacer 184-336, RT 337-681, RNaseH 682-832)
are a single reference's amino-acid coordinates (NC_003977.2, genotype D, 832 aa
Pol).  Pol domain boundaries are protein-level, and genotypes differ in Pol
length -- mostly through insertions in the spacer -- so copying genotype D's
numbers onto another genotype misplaces the spacer/RT boundary.

The reliable calibration is sequence alignment, not arithmetic: take a query Pol
protein with known boundaries, align it to the genotype's own Pol protein, and
read the boundaries off that alignment (see ``hbvpol.calibrate``).

Targets may be given as a Pol protein, or as a genome (with ``--pol-start-nt`` /
``--pol-end-nt``, or detected automatically by six-frame translation).

Fetch genotype reference genomes with Biopython Entrez, e.g.::

    from Bio import Entrez, SeqIO
    Entrez.email = "you@example.com"
    handle = Entrez.efetch(db="nucleotide", id="X02763", rettype="fasta",
                           retmode="text")
    SeqIO.write(SeqIO.read(handle, "fasta"), "genotype_A.fasta", "fasta")

Suggested accessions (verify G/H against HBVdb before trusting):

    A X02763   B D00329 (alt AB219428)   C AB014362 (alt GQ924620)
    D V01460 (alt AF121240)   E X75657   F X75658 (alt AY090458)
    G AF160501 [verify]       H AY090454 [verify]

Usage::

    python scripts/locate_pol_domains.py \\
        --target-fasta genotype_A.fasta --genotype A \\
        --out resources/pol_domain_spans.tsv

The query Pol defaults to the pipeline's derived reference
(``output/reference/reference_features.json``); override with
``--query-pol-fasta``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hbvpol.calibrate import calibrate_domain_spans, detect_pol_protein  # noqa: E402
from hbvpol.domain import DEFAULT_DOMAIN_SPANS, translate  # noqa: E402

_NUCLEOTIDE_SYMBOLS = set("ACGTUNRYKMSWBDHV-.")


def slice_wrapped(sequence: str, start_nt: int, end_nt: int) -> str:
    """Slice a 1-based inclusive interval from a circular sequence."""
    sequence = sequence.upper()
    length = len(sequence)
    start0 = (start_nt - 1) % length
    if start_nt <= end_nt:
        return sequence[start0:end_nt]
    return sequence[start0:] + sequence[: end_nt % length]


def pol_from_reference_features(path: Path) -> str:
    """Extract the reference Pol protein from ``reference_features.json``.

    The ``sequence``/``sequence_path`` fields written by the reference stage hold
    a path to the reference FASTA (not the sequence itself), so both a path and
    an inlined nucleotide sequence are accepted.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    reference = data.get("reference", data)
    start_nt = int(reference["pol_start_nt"])
    end_nt = int(reference["pol_end_nt"])

    sequence = ""
    for candidate in (reference.get("sequence_path"), reference.get("sequence")):
        if not candidate:
            continue
        candidate_path = Path(str(candidate))
        if candidate_path.is_file():
            sequence = read_first_sequence(candidate_path)
            break
        if set(str(candidate).upper()) <= _NUCLEOTIDE_SYMBOLS:
            sequence = str(candidate).upper()
            break
    if not sequence:
        raise SystemExit(
            f"could not resolve the reference sequence from {path}; expected a "
            "sequence_path/sequence file or an inlined nucleotide sequence"
        )
    return translate(slice_wrapped(sequence, start_nt, end_nt), frame=0)


def read_first_sequence(path: Path) -> str:
    """Return the first sequence from a (possibly multi-record) FASTA."""
    lines: list[str] = []
    seen_header = False
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith(">"):
            if seen_header:
                break
            seen_header = True
            continue
        if seen_header:
            lines.append(line.strip())
    if not lines:
        raise SystemExit(f"no sequence found in {path}")
    return "".join(lines).upper()


def looks_like_nucleotide(sequence: str) -> bool:
    return set(sequence.upper()) <= _NUCLEOTIDE_SYMBOLS


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--target-fasta", required=True, help="Genotype reference genome or Pol protein FASTA")
    parser.add_argument("--genotype", required=True, help="Genotype label to write into the TSV rows, e.g. A")
    parser.add_argument("--query-pol-fasta", default=None, help="Query Pol protein FASTA (default: reference features)")
    parser.add_argument(
        "--reference-features",
        default="output/reference/reference_features.json",
        help="Pipeline reference_features.json used to derive the query Pol",
    )
    parser.add_argument("--pol-start-nt", type=int, default=None, help="Target Pol ORF start (1-based) if target is a genome")
    parser.add_argument("--pol-end-nt", type=int, default=None, help="Target Pol ORF end (1-based, may wrap the origin)")
    parser.add_argument("--target-is-protein", action="store_true", help="Force protein interpretation of the target")
    parser.add_argument("--min-identity", type=float, default=0.6, help="Warn below this alignment identity")
    parser.add_argument("--out", default=None, help="TSV path to append rows to (default: print to stdout)")
    parser.add_argument("--source-note", default=None, help="Free-text note for the source column")
    args = parser.parse_args()

    if args.query_pol_fasta:
        query_pol = read_first_sequence(Path(args.query_pol_fasta))
    else:
        features_path = Path(args.reference_features)
        if not features_path.exists():
            raise SystemExit(
                f"no query Pol: {features_path} not found and --query-pol-fasta not given"
            )
        query_pol = pol_from_reference_features(features_path)

    target_path = Path(args.target_fasta)
    if not target_path.exists():
        raise SystemExit(f"target FASTA not found: {target_path}")
    target = read_first_sequence(target_path)

    is_protein = args.target_is_protein or not looks_like_nucleotide(target)

    detection_note = ""
    if is_protein:
        target_pol = target
    elif args.pol_start_nt and args.pol_end_nt:
        target_pol = translate(
            slice_wrapped(target, args.pol_start_nt, args.pol_end_nt), frame=0
        )
    else:
        target_pol, strand, frame, identity, coverage = detect_pol_protein(
            query_pol, target, min_identity=args.min_identity
        )
        detection_note = f" (detected {strand} frame {frame}, identity {identity:.1%})"

    result = calibrate_domain_spans(
        query_pol, target_pol, DEFAULT_DOMAIN_SPANS, min_identity=args.min_identity
    )
    for warning in result.warnings:
        print(f"WARNING [{args.genotype}]: {warning}", file=sys.stderr)

    source = args.source_note or (
        f"auto-calibrated by Pol protein alignment against {target_path.name}"
        f"{detection_note}; identity {result.identity:.1%}, coverage {result.coverage:.1%}; "
        "verify against the annotation before use"
    )

    rows = [[args.genotype, span.domain.value, span.start, span.end, source] for span in result.spans]
    if args.out:
        out_path = Path(args.out)
        write_header = not out_path.exists()
        with out_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle, delimiter="\t")
            if write_header:
                writer.writerow(["genotype", "domain", "start", "end", "source"])
            writer.writerows(rows)
        print(f"Appended {len(rows)} rows to {out_path}")
    else:
        for row in rows:
            print("\t".join(str(value) for value in row))


if __name__ == "__main__":
    main()
