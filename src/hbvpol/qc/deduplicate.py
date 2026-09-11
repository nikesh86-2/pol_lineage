"""Dereplication of near-identical genomes.

``dereplicate`` always removes *exact* sequence duplicates using a hash of the
upper-cased sequence.  For thresholds below identity 1.0 it additionally
performs a greedy, length-bucketed approximate dereplication in pure Python.

For production-scale datasets that approximate pass is better delegated to
``cd-hit-est`` (``-c <identity>``); the pure-Python path exists so the stage is
unit-testable offline and so small inputs do not require an external binary.
The function is deliberately conservative: the first sequence in input order
becomes the cluster representative.
"""

from __future__ import annotations

from typing import Iterable

from ..io import GenomeRecord

__all__ = ["dereplicate", "sequence_identity"]


def sequence_identity(a: str, b: str) -> float:
    """Fraction of identical positions between two equal-length sequences.

    Sequences of different length are treated as non-identical (0.0); the
    comparison is case-insensitive.
    """
    if len(a) != len(b):
        return 0.0
    if not a:
        return 1.0
    a_up, b_up = a.upper(), b.upper()
    matches = sum(1 for x, y in zip(a_up, b_up) if x == y)
    return matches / len(a_up)


def dereplicate(records: Iterable[GenomeRecord], identity: float = 0.9999) -> list[GenomeRecord]:
    """Return records with duplicates removed, preserving input order.

    Exact duplicates are always collapsed via a hash of the upper-cased
    sequence.  When ``identity < 1.0``, a greedy approximate pass merges any
    same-length record that is at least ``identity`` identical to an already
    kept representative.  Documented production substitute: ``cd-hit-est``.
    """
    indexed = list(enumerate(records))

    seen: set[str] = set()
    unique: list[tuple[int, GenomeRecord]] = []
    for index, record in indexed:
        key = record.seq.upper()
        if key in seen:
            continue
        seen.add(key)
        unique.append((index, record))

    if identity >= 1.0 - 1e-9:
        return [record for _, record in unique]

    buckets: dict[int, list[tuple[int, GenomeRecord]]] = {}
    for index, record in unique:
        buckets.setdefault(len(record.seq), []).append((index, record))

    kept: list[tuple[int, GenomeRecord]] = []
    for group in buckets.values():
        representatives: list[tuple[int, GenomeRecord]] = []
        for index, record in group:
            if any(sequence_identity(record.seq, rep.seq) >= identity for _, rep in representatives):
                continue
            representatives.append((index, record))
        kept.extend(representatives)

    kept.sort(key=lambda item: item[0])
    return [record for _, record in kept]
