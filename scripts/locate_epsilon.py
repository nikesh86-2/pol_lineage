#!/usr/bin/env python3
"""locate_epsilon.py -- calibrate the HBV epsilon (ε) span per genotype.

Why this exists
---------------
``epsilon_spans.tsv`` ships a single default row (1846-1905, the genotype D
reference's numbering).  Public records carry arbitrary rotation and sometimes
the opposite strand, so aligning a conserved ε query directly to a genotype
reference only yields comparable coordinates once the reference has been
normalised to the *reference frame* -- origin and strand.

This script does that normalisation with a genotype-independent anchor (the
invariant YMDD catalytic motif, whose codons vary synonymously), aligns a
conserved ε query, and writes the 1-based coordinates in the reference frame::

    python scripts/locate_epsilon.py \\
        --target-fasta resources/genotype_references/genotype_A.fasta \\
        --genotype A --query-span 1846 1905 \\
        --out resources/epsilon_spans.tsv

The query defaults to the reference's own span, so the reference row is a
no-op control.  Fetch genotype references as FASTA (see
``locate_pol_domains.py`` for the accessions; genotype I is provisional and
has no consensus type strain) and confirm any new accession before trusting it.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hbvpol.calibrate import (  # noqa: E402
    calibrate_epsilon_span,
    orient_to_reference,
    read_reference_sequence,
    slice_wrapped,
)
from hbvpol.io import read_fasta  # noqa: E402


def _first_sequence(path: Path) -> str:
    records = read_fasta(path)
    if not records:
        raise SystemExit(f"no sequence in {path}")
    return records[0].seq.upper()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--target-fasta", required=True, help="Genotype reference genome FASTA")
    parser.add_argument("--genotype", required=True, help="Genotype label for the TSV row, e.g. A")
    parser.add_argument("--reference-features", default="output/reference/reference_features.json",
                        help="reference_features.json used to resolve the reference genome")
    parser.add_argument("--reference-fasta", default=None,
                        help="Reference genome FASTA (alternative to --reference-features)")
    parser.add_argument("--query-span", type=int, nargs=2, metavar=("START", "END"),
                        default=[1846, 1905], help="Reference-frame span to use as the epsilon query")
    parser.add_argument("--query-seq", default=None, help="Use this epsilon query directly")
    parser.add_argument("--min-identity", type=float, default=0.85,
                        help="Warn below this alignment identity")
    parser.add_argument("--length-bounds", type=int, nargs=2, metavar=("MIN", "MAX"),
                        default=[55, 70], help="Expected epsilon length window")
    parser.add_argument("--out", default=None, help="TSV path to append the row to (default: stdout)")
    parser.add_argument("--source-note", default=None, help="Free-text note for the source column")
    args = parser.parse_args()

    if args.query_seq:
        query = args.query_seq.upper().replace("U", "T")
        reference = None
    elif args.reference_fasta:
        reference = _first_sequence(Path(args.reference_fasta))
    else:
        features_path = Path(args.reference_features)
        if not features_path.exists():
            raise SystemExit(
                f"no reference genome: {features_path} not found; give --reference-fasta"
            )
        reference = read_reference_sequence(features_path)

    if not args.query_seq:
        query = slice_wrapped(reference, args.query_span[0], args.query_span[1])
    if not query:
        raise SystemExit("epsilon query is empty")

    target_path = Path(args.target_fasta)
    if not target_path.exists():
        raise SystemExit(f"target FASTA not found: {target_path}")
    target = _first_sequence(target_path)

    note = ""
    if reference is not None:
        oriented = orient_to_reference(target, reference)
        if oriented is None:
            print(
                f"WARNING [{args.genotype}]: could not normalise orientation (no unique "
                "YMDD anchor); coordinates may be in the record's own frame",
                file=sys.stderr,
            )
            candidate, strand, shift = target, "?", 0
            note = " (orientation UNVERIFIED)"
        else:
            candidate, strand, shift = oriented
            note = f" (strand {strand}, shift {shift})"
    else:
        candidate, strand, shift = target, "?", 0

    hit = calibrate_epsilon_span(query, candidate, min_identity=args.min_identity)
    length = hit.end - hit.start + 1
    low, high = args.length_bounds
    if not (low <= length <= high):
        print(
            f"WARNING [{args.genotype}]: calibrated span is {length} nt "
            f"(expected {low}-{high}); inspect before trusting it",
            file=sys.stderr,
        )
    if hit.identity < args.min_identity:
        print(
            f"WARNING [{args.genotype}]: identity {hit.identity:.1%} is below "
            f"{args.min_identity:.0%}",
            file=sys.stderr,
        )

    source = (
        f"auto-calibrated by local alignment of a {len(query)}-nt epsilon query to "
        f"{target_path.name} normalised to the reference frame{note}; "
        f"identity {hit.identity:.1%}, coverage {hit.coverage:.1%}; verify before use"
    )
    if args.source_note:
        source = f"{source}; NOTE: {args.source_note}"
    row = [args.genotype, hit.start, hit.end, source]

    if args.out:
        out_path = Path(args.out)
        write_header = not out_path.exists()
        with out_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle, delimiter="\t")
            if write_header:
                writer.writerow(["genotype", "start_nt", "end_nt", "source"])
            writer.writerow(row)
        print(f"Appended: {args.genotype} {hit.start}-{hit.end}")
    else:
        print("\t".join(str(value) for value in row))


if __name__ == "__main__":
    main()
