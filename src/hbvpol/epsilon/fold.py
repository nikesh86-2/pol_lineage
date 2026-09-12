"""Epsilon RNA extraction and secondary-structure folding.

The HBV epsilon element is a ~60 nt cis-acting stem-loop at the 5' end of the
pregenomic RNA that acts as the encapsidation signal and the template for
protein priming.  This module slices it out of an *oriented* genome and predicts
its secondary structure.

Folding uses ViennaRNA's ``RNAfold`` when it is installed; otherwise a
self-contained Nussinov-style minimum-free-energy fold provides a documented,
offline fallback so the stage always produces a dot-bracket structure.

Coordinate note
---------------
``epsilon.genome_span`` is interpreted in the *oriented* genome frame produced
by the QC stage (origin recut at the EcoRI site, so oriented index 0 == standard
nt 1).  The default ``[1846, 1905]`` is the approximate genotype-A2 epsilon span
in that convention; it is **origin-aware and should be refined** per genotype and
against the reference annotation.  The span may wrap the origin.
"""

from __future__ import annotations

import re
from pathlib import Path

from ..config import get
from ..io import GenomeRecord
from ..pipeline import get_logger, have_executable, run_command

__all__ = [
    "DEFAULT_EPSILON_SPAN",
    "extract_epsilon",
    "resolve_epsilon_span",
    "load_epsilon_spans",
    "fold_epsilon",
    "nussinov_fold",
    "epsilon_features",
]

logger = get_logger("epsilon.fold")

#: Approximate genotype-A2 epsilon span (1-based, inclusive) in oriented coords.
DEFAULT_EPSILON_SPAN = (1846, 1905)

# Watson–Crick plus the G–U wobble pair.
_CANONICAL_PAIRS = {
    ("A", "U"), ("U", "A"),
    ("G", "C"), ("C", "G"),
    ("G", "U"), ("U", "G"),
}


def can_pair(left: str, right: str) -> bool:
    return (str(left).upper().replace("T", "U"), str(right).upper().replace("T", "U")) in _CANONICAL_PAIRS


def _parse_span(raw) -> tuple[int, int]:
    if isinstance(raw, dict):
        return int(raw.get("start", raw.get("begin", DEFAULT_EPSILON_SPAN[0]))), int(
            raw.get("end", raw.get("stop", DEFAULT_EPSILON_SPAN[1]))
        )
    if isinstance(raw, (list, tuple)) and len(raw) >= 2:
        return int(raw[0]), int(raw[1])
    return DEFAULT_EPSILON_SPAN


def load_epsilon_spans(path: str | Path) -> dict[str, tuple[int, int]]:
    """Load a ``genotype -> (start, end)`` epsilon-span table.

    The table is a TSV with columns ``genotype, start_nt, end_nt`` (extra
    columns such as ``source`` are ignored).  Keys are lower-cased; a
    ``default`` row applies when the genotype is absent.  Returns ``{}`` when
    the file is missing or unreadable, so the caller falls back to config.
    """
    spans_path = Path(path)
    if not spans_path.exists():
        return {}
    try:
        import csv

        spans: dict[str, tuple[int, int]] = {}
        with spans_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                genotype = str(row.get("genotype", "")).strip().lower()
                if not genotype:
                    continue
                try:
                    spans[genotype] = (int(row["start_nt"]), int(row["end_nt"]))
                except (KeyError, TypeError, ValueError):
                    continue
        return spans
    except OSError as error:  # pragma: no cover - defensive
        logger.warning("could not read epsilon span table %s: %s", spans_path, error)
        return {}


def resolve_epsilon_span(config: dict, genotype: str | None = None) -> tuple[int, int]:
    """Resolve the epsilon span for a genotype.

    Precedence: the table row for ``genotype`` (``epsilon.spans_file``), then a
    ``default`` row, then ``epsilon.genome_span`` (a ``[start, end]`` list or a
    ``genotype -> [start, end]`` dict), then :data:`DEFAULT_EPSILON_SPAN`.
    """
    key = str(genotype).strip().lower() if genotype else ""
    configured = get(config, "epsilon.spans_file", None)
    if configured:
        table = load_epsilon_spans(configured)
        if key and key in table:
            return table[key]
        if "default" in table:
            return table["default"]

    raw = get(config, "epsilon.genome_span", None)
    if isinstance(raw, dict):
        for candidate, value in raw.items():
            if key and str(candidate).lower() == key:
                return _parse_span(value)
        if "default" in raw:
            return _parse_span(raw["default"])
    elif raw:
        return _parse_span(raw)
    return DEFAULT_EPSILON_SPAN


def extract_epsilon(record: GenomeRecord | str, config: dict, genotype: str | None = None) -> str:
    """Slice the epsilon element from an oriented genome and validate its length.

    The span is resolved per genotype (see :func:`resolve_epsilon_span`); it may
    wrap the origin.  Lengths outside ``epsilon.length_nt`` are reported as a
    warning (the sequence is still returned so downstream steps can proceed, but
    the log makes a wrong span obvious).
    """
    seq = record.seq if isinstance(record, GenomeRecord) else str(record)
    seq = seq.upper().replace("U", "T")
    start, end = resolve_epsilon_span(config, genotype)
    length = len(seq)
    if length == 0:
        return ""
    start0 = (start - 1) % length
    if start <= end and end <= length:
        sub = seq[start0:end]
    else:
        end0 = end % length if end else length
        sub = seq[start0:] + seq[:end0]

    bounds = get(config, "epsilon.length_nt", [55, 70]) or [55, 70]
    low, high = int(bounds[0]), int(bounds[1])
    if not (low <= len(sub) <= high):
        logger.warning(
            "epsilon span %s..%s yielded %d nt (expected %d-%d); refine epsilon.genome_span",
            start, end, len(sub), low, high,
        )
    return sub


# --------------------------------------------------------------------------- #
# pure-Python Nussinov MFE fold (offline fallback)
# --------------------------------------------------------------------------- #
def _traceback(dp, i: int, j: int, seq: str, min_loop: int, structure: list[str], pairs: list[tuple[int, int]]) -> None:
    while i < j:
        if dp[i][j] == dp[i + 1][j]:
            i += 1
        elif dp[i][j] == dp[i][j - 1]:
            j -= 1
        elif j - i > min_loop and can_pair(seq[i], seq[j]) and dp[i][j] == dp[i + 1][j - 1] + 1:
            structure[i] = "("
            structure[j] = ")"
            pairs.append((i + 1, j + 1))
            i += 1
            j -= 1
        else:
            for k in range(i + 1, j):
                if dp[i][j] == dp[i][k] + dp[k + 1][j]:
                    _traceback(dp, i, k, seq, min_loop, structure, pairs)
                    _traceback(dp, k + 1, j, seq, min_loop, structure, pairs)
                    return
            break  # pragma: no cover - defensive: unreachable for a consistent DP


def nussinov_fold(seq: str, min_loop: int = 3) -> tuple[str, float, list[tuple[int, int]]]:
    """Maximum-base-pair (Nussinov) secondary structure.

    Returns ``(dotbracket, n_pairs, [(i, j), ...])`` with 1-based pair indices.
    Only canonical/Watson–Crick/wobble pairs are formed and every hairpin loop
    is at least ``min_loop`` nucleotides.
    """
    seq = str(seq).upper().replace("T", "U")
    n = len(seq)
    if n == 0:
        return "", 0.0, []
    dp = [[0] * n for _ in range(n)]
    for span in range(1, n):
        for i in range(0, n - span):
            j = i + span
            best = dp[i + 1][j]
            if dp[i][j - 1] > best:
                best = dp[i][j - 1]
            if j - i > min_loop and can_pair(seq[i], seq[j]):
                paired = dp[i + 1][j - 1] + 1
                if paired > best:
                    best = paired
            for k in range(i + 1, j):
                bifurcation = dp[i][k] + dp[k + 1][j]
                if bifurcation > best:
                    best = bifurcation
            dp[i][j] = best
    structure = ["."] * n
    pairs: list[tuple[int, int]] = []
    _traceback(dp, 0, n - 1, seq, min_loop, structure, pairs)
    return "".join(structure), float(dp[0][n - 1]), pairs


# --------------------------------------------------------------------------- #
# RNAfold (optional)
# --------------------------------------------------------------------------- #
_BRACKETS = set(".()[]{}<>,")


def _run_rnafold(seq: str, config: dict):
    """Fold with ViennaRNA RNAfold; return ``(dotbracket, mfe)`` or ``None``."""
    executable = str(get(config, "epsilon.rnafold.executable", "RNAfold") or "RNAfold")
    if not have_executable(executable):
        return None
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        fasta = Path(tmp) / "epsilon.fa"
        fasta.write_text(f">epsilon\n{seq}\n", encoding="utf-8")
        try:
            result = run_command([executable, "--noPS", str(fasta)], check=True)
        except Exception as error:  # pragma: no cover - external tool
            logger.warning("RNAfold failed (%s); using Nussinov fallback", error)
            return None
    stdout = result.stdout or ""
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        token = line.split()[0]
        if token and set(token) <= _BRACKETS and any(c in "()" for c in token):
            energy = None
            match = re.search(r"\(\s*([-+]?\d+(?:\.\d+)?)\s*\)", line)
            if match:
                energy = float(match.group(1))
            return token, energy
    logger.warning("could not parse RNAfold output; using Nussinov fallback")
    return None


def fold_epsilon(seq: str, config: dict) -> dict:
    """Fold an epsilon sequence to a dot-bracket structure.

    Tries ``RNAfold`` when it is both configured and available, otherwise falls
    back to :func:`nussinov_fold`.  The returned dict is JSON-serialisable.
    """
    seq = str(seq).upper().replace("T", "U")
    models = list(get(config, "epsilon.fold_models", ["literature", "rnafold"]) or [])
    result = None
    if "rnafold" in models:
        result = _run_rnafold(seq, config)
    if result is not None:
        dotbracket, mfe = result
        method = "rnafold"
        pairs = [(i + 1, j + 1) for i, j in _pair_indices(dotbracket)]
    else:
        dotbracket, n_pairs, pairs = nussinov_fold(seq)
        mfe = -float(n_pairs)
        method = "nussinov"
    return {
        "sequence": seq,
        "length": len(seq),
        "dotbracket": dotbracket,
        "mfe": mfe,
        "method": method,
        "n_pairs": int(len(pairs)),
        "pairs": [[int(i), int(j)] for i, j in pairs],
    }


def _pair_indices(dotbracket: str):
    stack: list[int] = []
    for idx, char in enumerate(dotbracket):
        if char in "([{<":
            stack.append(idx)
        elif char in ")]}>":
            if stack:
                yield stack.pop(), idx


def epsilon_features(dotbracket: str) -> list[str]:
    """Per-position structural feature: ``base_pair``, ``bulge`` or ``loop``.

    An unpaired position with at least one paired neighbour is classed as a
    ``bulge`` (covering bulges and internal loops); otherwise it is a ``loop``.
    """
    length = len(dotbracket)
    paired = [char in "()" for char in dotbracket]
    features: list[str] = []
    for idx in range(length):
        if paired[idx]:
            features.append("base_pair")
            continue
        left = paired[idx - 1] if idx > 0 else False
        right = paired[idx + 1] if idx + 1 < length else False
        features.append("bulge" if (left or right) else "loop")
    return features
