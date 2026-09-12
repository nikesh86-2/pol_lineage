"""Covariation and epistasis between alignment columns.

Two complementary scores are provided:

* :func:`mutual_information` — the full mutual-information matrix (bits),
  estimated only from sequences where both columns are unambiguous.
* :func:`dca_scores` — direct-coupling-like scores.  When the configured
  ``selection.covariation.dca_impl`` names an importable implementation that is
  used, otherwise an APC-corrected mutual-information matrix is returned as a
  documented mean-field-like proxy.  This keeps the whole stage offline.
* :func:`epistasis_pairs` — pairwise epistasis measured as the total-variation
  distance between the observed joint distribution and the product of marginals,
  with a ``support`` column for the fraction of sequences observed at both
  sites.

All core functions are pure and side-effect-free.  ``*_pairs`` helpers return
frames without the ``lineage`` column, which the pipeline adds.
"""

from __future__ import annotations

import importlib
import math
from typing import Mapping

import numpy as np
import pandas as pd

from ..config import get
from ..io import GenomeRecord
from ..pipeline import StageError, get_logger
from .dualframe import alignment_width, coerce_alignment

__all__ = [
    "COVARIATION_COLUMNS",
    "EPISTASIS_COLUMNS",
    "mutual_information",
    "apc_correct",
    "dca_scores",
    "covarying_pairs",
    "epistasis_pairs",
]

logger = get_logger("selection.covariation")

_COVARIATION_COLUMNS = ["position_i", "position_j", "method", "score", "pvalue"]
_EPISTASIS_COLUMNS = ["position_i", "position_j", "method", "score", "support"]
COVARIATION_COLUMNS = ["lineage", *_COVARIATION_COLUMNS]
EPISTASIS_COLUMNS = ["lineage", *_EPISTASIS_COLUMNS]


def _empty(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)


def _encode(records: list[GenomeRecord]):
    """Encode an alignment as an integer array (``-1`` = gap/ambiguous).

    Returns ``(array, alphabet, n_sequences, width)``.
    """
    if not records:
        return np.zeros((0, 0), dtype=int), [], 0, 0
    width = alignment_width(records)
    sequences = [record.seq.upper().ljust(width, "-") for record in records]
    alphabet = sorted(set("".join(sequences)) - {"-"})
    index = {symbol: position for position, symbol in enumerate(alphabet)}
    array = np.full((len(sequences), width), -1, dtype=int)
    for row, sequence in enumerate(sequences):
        for column, symbol in enumerate(sequence):
            array[row, column] = index.get(symbol, -1)
    return array, alphabet, len(sequences), width


def _column_entropies(array: np.ndarray, present: np.ndarray, states: int) -> np.ndarray:
    """Per-column Shannon entropy in bits (0 for invariant or empty columns)."""
    width = array.shape[1] if array.ndim == 2 else 0
    entropies = np.zeros(width, dtype=float)
    for column in range(width):
        values = array[present[:, column], column]
        if values.size == 0:
            continue
        counts = np.bincount(values, minlength=states).astype(float)
        counts = counts[counts > 0]
        probabilities = counts / counts.sum()
        entropies[column] = float(-(probabilities * np.log2(probabilities)).sum())
    return entropies


def _variable_columns(
    entropies: np.ndarray,
    min_entropy: float = 0.0,
    max_positions: int | None = None,
) -> list[int]:
    """Select informative columns, capped by entropy to bound the O(L^2) scan.

    Pairwise metrics are quadratic in the number of columns, which is
    prohibitive for whole genomes.  Restricting the scan to variable columns
    (and, when still too many, to the ``max_positions`` most entropic ones) keeps
    the stage tractable; the latter is a documented approximation that a
    production run should replace with a vectorised or external DCA backend.
    """
    selected = [i for i, value in enumerate(entropies) if value > min_entropy]
    if max_positions and len(selected) > max_positions:
        logger.warning(
            "%d variable columns exceed max_positions=%d; keeping the most entropic",
            len(selected), max_positions,
        )
        selected = sorted(sorted(selected, key=lambda i: entropies[i], reverse=True)[:max_positions])
    return selected


def _mi_submatrix(array: np.ndarray, present: np.ndarray, states: int, columns: list[int]) -> np.ndarray:
    """Mutual-information matrix (bits) restricted to ``columns``."""
    width = len(columns)
    matrix = np.zeros((width, width), dtype=float)
    if not columns:
        return matrix
    sub = array[:, columns]
    sub_present = present[:, columns]
    for i in range(width):
        a = sub[:, i]
        a_ok = sub_present[:, i]
        if not a_ok.any():
            continue
        for j in range(i + 1, width):
            both = a_ok & sub_present[:, j]
            if not both.any():
                continue
            joint = np.bincount(a[both] * states + sub[both, j], minlength=states * states)
            joint = joint.reshape(states, states).astype(float)
            joint /= joint.sum()
            pi = joint.sum(axis=1, keepdims=True)
            pj = joint.sum(axis=0, keepdims=True)
            product = pi * pj
            nonzero = joint > 0
            matrix[i, j] = matrix[j, i] = float(
                np.sum(joint[nonzero] * np.log2(joint[nonzero] / product[nonzero]))
            )
    return matrix


def mutual_information(alignment) -> np.ndarray:
    """Full mutual-information matrix (bits) between alignment columns.

    MI is estimated from the sequences where both columns carry an unambiguous
    symbol.  Pairs with no shared observations get ``0``.  This is O(L^2) in the
    number of columns; callers that only need informative sites should select
    columns first (see :func:`covarying_pairs`) or delegate to an external DCA
    implementation.
    """
    records = coerce_alignment(alignment)
    array, alphabet, _, width = _encode(records)
    if width == 0 or not alphabet:
        return np.zeros((width, width), dtype=float)
    present = array >= 0
    return _mi_submatrix(array, present, len(alphabet), list(range(width)))


def apc_correct(matrix) -> np.ndarray:
    """Average Product Correction (Dunn et al. 2008) of a score matrix.

    Subtracts the expected background ``row_mean * col_mean / grand_mean``.
    The diagonal is zeroed because self-pairs are not of interest.
    """
    values = np.array(matrix, dtype=float)
    if values.size == 0:
        return values
    row_mean = values.mean(axis=1, keepdims=True)
    col_mean = values.mean(axis=0, keepdims=True)
    grand_mean = values.mean()
    if grand_mean <= 0:
        return values
    corrected = values - row_mean * col_mean / grand_mean
    np.fill_diagonal(corrected, 0.0)
    return corrected


def _try_external_dca(records: list[GenomeRecord], config: Mapping[str, object]) -> np.ndarray | None:
    """Call the configured external DCA implementation, or return ``None``.

    ``selection.covariation.dca_impl`` names an importable module exposing
    ``dca_scores``.  It is tried with and without ``config`` and with either the
    record list or a plain list of sequence strings, so both simple wrappers and
    the shipped ``hbvpol.selection.plmc_backend`` adapter work.  Any absence,
    exception or shape mismatch falls back to the APC-corrected MI proxy in the
    caller.
    """
    implementation = str(get(config, "selection.covariation.dca_impl", "") or "").strip()
    if not implementation:
        return None
    try:
        module = importlib.import_module(implementation)
    except Exception:
        logger.warning(
            "dca_impl %r could not be imported; using APC-corrected mutual information",
            implementation,
        )
        return None

    imported = coerce_alignment(records)
    sequences = [record.seq for record in records]
    candidates = [
        lambda: module.dca_scores(imported, config),
        lambda: module.dca_scores(imported),
        lambda: module.dca_scores(sequences, config),
        lambda: module.dca_scores(sequences),
    ]
    for call in candidates:
        try:
            matrix = np.asarray(call(), dtype=float)
        except Exception:
            continue
        if matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1]:
            return matrix
    logger.warning(
        "dca_impl %r returned no square score matrix; using APC-corrected mutual information",
        implementation,
    )
    return None


def dca_scores(alignment, config: Mapping[str, object]) -> np.ndarray:
    """Direct-coupling-like scores for every pair of columns.

    Uses the configured ``selection.covariation.dca_impl`` when it can be
    imported; otherwise (and on any failure) returns the APC-corrected
    mutual-information matrix as a documented mean-field-like proxy, so the
    stage never depends on an external package.
    """
    records = coerce_alignment(alignment)
    external = _try_external_dca(records, config)
    if external is not None:
        return external
    return apc_correct(mutual_information(records))


def _pvalue_from_scores(matrix: np.ndarray) -> np.ndarray:
    """Two-sided normal-tail p-values for each entry of an upper-triangle score."""
    width = matrix.shape[0]
    pvalues = np.ones((width, width), dtype=float)
    if width < 3:
        return pvalues
    iu, ju = np.triu_indices(width, k=1)
    scores = matrix[iu, ju]
    mean = float(scores.mean())
    std = float(scores.std())
    if std <= 0:
        return pvalues
    for i, j in zip(iu, ju):
        z = (float(matrix[i, j]) - mean) / std
        pvalues[i, j] = pvalues[j, i] = math.erfc(abs(z) / math.sqrt(2.0))
    return pvalues


def _score_matrix(alignment, method: str, config: Mapping[str, object]) -> np.ndarray | None:
    normalised = method.strip().lower()
    if normalised in {"mutual_information", "mi"}:
        return mutual_information(alignment)
    if normalised in {"dca", "dca_scores", "mean_field", "plmdca"}:
        return dca_scores(alignment, config)
    logger.warning("unknown covariation method %r; skipping", method)
    return None


def _subset_records(records: list[GenomeRecord], columns: list[int]) -> list[GenomeRecord]:
    """Return records restricted to ``columns`` (padding short sequences).

    External DCA engines estimate an O(L^2) parameter set, so they must be run
    on the selected variable columns rather than the whole genome; computing a
    full-width matrix and indexing it afterwards is intractable.
    """
    subset: list[GenomeRecord] = []
    for record in records:
        seq = record.seq
        subset.append(
            GenomeRecord(
                id=record.id,
                seq="".join(seq[index] if index < len(seq) else "-" for index in columns),
            )
        )
    return subset


def covarying_pairs(alignment, config: Mapping[str, object]) -> pd.DataFrame:
    """Covarying column pairs for each configured method.

    Thresholds by ``selection.covariation.min_score`` and (optionally) caps the
    output at ``selection.covariation.top_n`` pairs per method.  Returns columns
    ``position_i, position_j, method, score, pvalue`` (1-based positions).
    """
    records = coerce_alignment(alignment)
    methods = get(config, "selection.covariation.methods", ["mutual_information"]) or []
    if isinstance(methods, str):
        methods = [methods]
    min_seqs = int(get(config, "selection.covariation.min_seqs", 0) or 0)
    if len(records) < max(1, min_seqs):
        logger.warning(
            "only %d sequences (< selection.covariation.min_seqs=%d); writing empty covariation",
            len(records), min_seqs,
        )
        return _empty(_COVARIATION_COLUMNS)

    min_score = float(get(config, "selection.covariation.min_score", 0.0) or 0.0)
    min_entropy = float(get(config, "selection.covariation.min_entropy", 0.0) or 0.0)
    max_positions = get(config, "selection.covariation.max_positions", None)
    max_positions = int(max_positions) if max_positions not in (None, "", 0) else None
    top_n = get(config, "selection.covariation.top_n", None)
    top_n = int(top_n) if top_n not in (None, "", 0) else None

    array, alphabet, _, width = _encode(records)
    if width == 0 or not alphabet:
        return _empty(_COVARIATION_COLUMNS)
    present = array >= 0
    states = len(alphabet)
    entropies = _column_entropies(array, present, states)
    columns = _variable_columns(entropies, min_entropy=min_entropy, max_positions=max_positions)
    if not columns:
        logger.warning("no variable columns above min_entropy=%.3f; writing empty covariation", min_entropy)
        return _empty(_COVARIATION_COLUMNS)
    base = _mi_submatrix(array, present, states, columns)

    rows: list[dict] = []
    for method in methods:
        normalised = str(method).strip().lower()
        if normalised in {"mutual_information", "mi"}:
            matrix, positions = base, columns
        elif normalised in {"dca", "dca_scores", "mean_field", "plmdca"}:
            # Run the backend on the selected columns only (O(L^2) parameters).
            external = _try_external_dca(_subset_records(records, columns), config)
            if external is None:
                if bool(get(config, "selection.covariation.require_backend", False)):
                    raise StageError(
                        "selection.covariation.require_backend is set but dca_impl "
                        f"{get(config, 'selection.covariation.dca_impl')!r} is unavailable"
                    )
                matrix, positions = apc_correct(base), columns
            elif external.shape[0] == len(columns):
                matrix, positions = external, columns
            else:
                logger.warning(
                    "external DCA matrix is %dx%d but %d columns were selected; "
                    "using the APC-corrected proxy",
                    external.shape[0], external.shape[1], len(columns),
                )
                matrix, positions = apc_correct(base), columns
        else:
            external = _score_matrix(records, str(method), config)
            if external is None or external.size == 0:
                continue
            matrix, positions = external, list(range(external.shape[0]))

        pvalues = _pvalue_from_scores(matrix)
        size = matrix.shape[0]
        candidates: list[tuple[float, int, int]] = []
        for i in range(size):
            for j in range(i + 1, size):
                score = float(matrix[i, j])
                if score > min_score:
                    candidates.append((score, i, j))
        candidates.sort(key=lambda item: item[0], reverse=True)
        if top_n is not None:
            candidates = candidates[:top_n]
        for score, i, j in candidates:
            rows.append({
                "position_i": positions[i] + 1,
                "position_j": positions[j] + 1,
                "method": str(method),
                "score": score,
                "pvalue": float(pvalues[i, j]),
            })

    if not rows:
        return _empty(_COVARIATION_COLUMNS)
    return pd.DataFrame(rows, columns=_COVARIATION_COLUMNS)


def epistasis_pairs(alignment, config: Mapping[str, object]) -> pd.DataFrame:
    """Pairwise epistasis from the deviation of the joint from independence.

    ``score`` is the total-variation distance ``0.5 * sum |p(a,b) -
    p(a)p(b)|`` and ``support`` the fraction of sequences with an unambiguous
    state at both positions.  Pairs below ``selection.epistasis.min_support``
    are dropped.  Returns columns ``position_i, position_j, method, score,
    support``.
    """
    records = coerce_alignment(alignment)
    method = str(get(config, "selection.epistasis.method", "pairwise_epistasis") or "pairwise_epistasis")
    min_support = float(get(config, "selection.epistasis.min_support", 0.0) or 0.0)
    min_entropy = float(get(config, "selection.epistasis.min_entropy", 0.0) or 0.0)
    max_positions = get(config, "selection.epistasis.max_positions", None)
    max_positions = int(max_positions) if max_positions not in (None, "", 0) else None
    top_n = get(config, "selection.epistasis.top_n", None)
    top_n = int(top_n) if top_n not in (None, "", 0) else None

    array, alphabet, n_sequences, width = _encode(records)
    if width == 0 or not alphabet or n_sequences == 0:
        return _empty(_EPISTASIS_COLUMNS)

    states = len(alphabet)
    present = array >= 0
    entropies = _column_entropies(array, present, states)
    columns = _variable_columns(entropies, min_entropy=min_entropy, max_positions=max_positions)
    if len(columns) < 2:
        return _empty(_EPISTASIS_COLUMNS)

    rows: list[dict] = []
    for a_index in range(len(columns)):
        i = columns[a_index]
        a_ok = present[:, i]
        if not a_ok.any():
            continue
        for b_index in range(a_index + 1, len(columns)):
            j = columns[b_index]
            both = a_ok & present[:, j]
            n_both = int(both.sum())
            support = n_both / n_sequences
            if support < min_support or n_both == 0:
                continue
            ai = array[both, i]
            bj = array[both, j]
            joint = np.bincount(ai * states + bj, minlength=states * states).reshape(states, states)
            joint = joint.astype(float) / n_both
            pi = joint.sum(axis=1, keepdims=True)
            pj = joint.sum(axis=0, keepdims=True)
            score = 0.5 * float(np.abs(joint - pi * pj).sum())
            if score > 0:
                rows.append({
                    "position_i": i + 1,
                    "position_j": j + 1,
                    "method": method,
                    "score": score,
                    "support": support,
                })

    if not rows:
        return _empty(_EPISTASIS_COLUMNS)
    rows.sort(key=lambda row: row["score"], reverse=True)
    if top_n is not None:
        rows = rows[:top_n]
    return pd.DataFrame(rows, columns=_EPISTASIS_COLUMNS)
