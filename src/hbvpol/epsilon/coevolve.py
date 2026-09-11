"""Pol–epsilon coevolution and cross-genotype compatibility.

Two ideas are implemented here, both pure and offline:

``pol_epsilon_coevolution``
    Mutual information between TP-domain residues and epsilon structural
    features (base pairs, bulges, loops).  Each feature is scored by the
    strongest covariation over all epsilon positions carrying that feature, so
    the result is a compact residue x feature table that ranks which TP
    positions are coupled to the RNA element.

``compatibility_matrix``
    A documented, deliberately coarse *contact-map energy* heuristic for every
    (Pol genotype, epsilon genotype) pair.  It is **not** a calibrated affinity
    predictor — it is a screening heuristic that must be validated
    experimentally; the ``predictor`` config key records which heuristic was used.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import get
from ..domain import DEFAULT_DOMAIN_SPANS, PolDomain
from ..io import GenomeRecord, read_fasta
from ..pipeline import get_logger
from .fold import epsilon_features, fold_epsilon

__all__ = [
    "COEVOLUTION_COLUMNS",
    "COMPATIBILITY_COLUMNS",
    "mutual_information",
    "pol_epsilon_coevolution",
    "compatibility_matrix",
    "contact_map_energy",
]

logger = get_logger("epsilon.coevolve")

COEVOLUTION_COLUMNS = [
    "pol_position",
    "pol_residue",
    "epsilon_feature",
    "epsilon_position",
    "mutual_information",
    "n_sequences",
]

COMPATIBILITY_COLUMNS = [
    "pol_genotype",
    "epsilon_genotype",
    "energy",
    "compatibility",
    "rank",
]


def _empty(columns) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)


# --------------------------------------------------------------------------- #
# alignment coercion and information theory
# --------------------------------------------------------------------------- #
def _resolve_alignment(obj) -> list[tuple[str, str]]:
    """Coerce a path, record list or string iterable to ``[(id, seq), ...]``."""
    if obj is None:
        return []
    if isinstance(obj, (str, Path)) and Path(obj).exists():
        return [(record.id, record.seq) for record in read_fasta(obj)]
    if isinstance(obj, (str, bytes)):
        return [("seq1", str(obj))]
    resolved: list[tuple[str, str]] = []
    for index, item in enumerate(obj):
        if isinstance(item, GenomeRecord):
            resolved.append((item.id, item.seq))
        elif isinstance(item, (tuple, list)) and len(item) >= 2:
            resolved.append((str(item[0]), str(item[1])))
        else:
            resolved.append((f"seq{index + 1}", str(item)))
    return resolved


def _entropy(labels) -> float:
    counts = np.asarray(list(Counter(labels).values()), dtype=float)
    if counts.sum() == 0:
        return 0.0
    probs = counts / counts.sum()
    return float(-(probs * np.log(probs)).sum())


def mutual_information(x, y) -> float:
    """Mutual information (nats) between two categorical vectors."""
    x = list(x)
    y = list(y)
    n = len(x)
    if n == 0 or n != len(y):
        return 0.0
    joint = Counter(zip(x, y))
    probs = np.asarray(list(joint.values()), dtype=float) / n
    return float(_entropy(x) + _entropy(y) + (probs * np.log(probs)).sum())


def _consensus_char(column) -> str:
    counts = Counter(ch for ch in column if ch not in "-.")
    if not counts:
        return "-"
    return counts.most_common(1)[0][0]


def _consensus_sequence(sequences: list[str]) -> str:
    if not sequences:
        return ""
    width = min(len(s) for s in sequences) if sequences else 0
    return "".join(_consensus_char([s[i] for s in sequences]) for i in range(width))


def _window(config: dict, width: int) -> tuple[int, int]:
    """Resolve ``epsilon.coevolve.residue_window`` to a 0-based column range."""
    window = get(config, "epsilon.coevolve.residue_window", "TP")
    if isinstance(window, bool):
        return 0, width
    if isinstance(window, int):
        return 0, min(int(window), width)
    if isinstance(window, (list, tuple)) and len(window) == 2:
        return max(0, int(window[0]) - 1), min(int(window[1]), width)
    if isinstance(window, str):
        text = window.strip()
        if "-" in text and text.replace("-", "").isdigit():
            low, _, high = text.partition("-")
            return max(0, int(low) - 1), min(int(high), width)
        try:
            domain = PolDomain.parse(text)
        except ValueError:
            logger.warning("unknown residue_window %r; scanning all columns", window)
            return 0, width
        for span in DEFAULT_DOMAIN_SPANS:
            if span.domain is domain:
                return max(0, span.start - 1), min(span.end, width)
    return 0, width


def pol_epsilon_coevolution(pol_alignment, epsilon_alignment, config: dict) -> pd.DataFrame:
    """Mutual information between TP residues and epsilon structural features."""
    pol = _resolve_alignment(pol_alignment)
    epsilon = _resolve_alignment(epsilon_alignment)
    epsilon_by_id = {ident: seq for ident, seq in epsilon}
    pairs: list[tuple[str, str]] = [
        (seq, epsilon_by_id[ident]) for ident, seq in pol if ident in epsilon_by_id
    ]
    if not pairs:
        logger.warning(
            "no shared ids between Pol and epsilon alignments; coevolution table will be empty"
        )
        return _empty(COEVOLUTION_COLUMNS)

    width = min(len(p) for p, _ in pairs)
    ewidth = min(len(e) for _, e in pairs)
    if width == 0 or ewidth == 0:
        return _empty(COEVOLUTION_COLUMNS)

    pol_seqs = [p[:width] for p, _ in pairs]
    eps_seqs = [e[:ewidth] for _, e in pairs]
    pol_columns = list(zip(*pol_seqs))
    eps_columns = list(zip(*eps_seqs))

    consensus_epsilon = _consensus_sequence(eps_seqs)
    structure = fold_epsilon(consensus_epsilon, config)["dotbracket"]
    feature_by_column = [
        epsilon_features(structure)[j] if j < len(structure) else "loop"
        for j in range(ewidth)
    ]

    requested = list(
        get(config, "epsilon.coevolve.rna_features", ["base_pair", "bulge", "loop"])
        or ["base_pair", "bulge", "loop"]
    )
    start, stop = _window(config, width)
    n_sequences = len(pairs)

    rows: list[dict[str, object]] = []
    for column_index in range(start, stop):
        column = pol_columns[column_index]
        for feature in requested:
            best_mi = 0.0
            best_position = None
            for epsilon_index in range(ewidth):
                if feature_by_column[epsilon_index] != feature:
                    continue
                score = mutual_information(column, eps_columns[epsilon_index])
                if best_position is None or score > best_mi:
                    best_mi = score
                    best_position = epsilon_index + 1
            rows.append({
                "pol_position": column_index + 1,
                "pol_residue": _consensus_char(column),
                "epsilon_feature": feature,
                "epsilon_position": best_position,
                "mutual_information": float(best_mi),
                "n_sequences": int(n_sequences),
            })

    frame = pd.DataFrame(rows, columns=COEVOLUTION_COLUMNS)
    if len(frame):
        frame = frame.sort_values(
            ["mutual_information", "pol_position"], ascending=[False, True]
        ).reset_index(drop=True)
    return frame


# --------------------------------------------------------------------------- #
# cross-genotype compatibility
# --------------------------------------------------------------------------- #
def _resample(values: np.ndarray, n_bins: int) -> np.ndarray:
    if values.size == 0:
        return np.zeros(n_bins, dtype=float)
    if values.size == 1:
        return np.full(n_bins, float(values[0]), dtype=float)
    source = np.linspace(0.0, 1.0, values.size)
    target = np.linspace(0.0, 1.0, n_bins)
    return np.interp(target, source, values)


def _affinity_profile(pol_seq, n_bins: int = 32) -> np.ndarray:
    seq = str(pol_seq).upper()
    span = DEFAULT_DOMAIN_SPANS[0]  # TP
    tp = seq[span.start - 1:span.end]
    if not tp:
        tp = seq
    values = []
    for aa in tp:
        if aa in "KRH":
            values.append(1.0)
        elif aa in "DE":
            values.append(-0.5)
        elif aa in "STNQY":
            values.append(0.6)
        elif aa in "AGP":
            values.append(0.2)
        else:
            values.append(0.4)
    return _resample(np.asarray(values, dtype=float), n_bins)


def _accessibility_profile(epsilon_seq, config: dict, n_bins: int = 32) -> np.ndarray:
    structure = fold_epsilon(str(epsilon_seq), config)["dotbracket"]
    features = epsilon_features(structure)
    values = np.asarray(
        [1.0 if feature in {"loop", "bulge"} else 0.3 for feature in features],
        dtype=float,
    )
    return _resample(values, n_bins)


def contact_map_energy(pol_seq, epsilon_seq, config: dict) -> tuple[float, float]:
    """Coarse contact-map energy for a Pol/epsilon sequence pair.

    The Pol TP side-chain affinity profile is dotted with the epsilon structural
    accessibility profile on a common 32-bin grid.  A higher overlap yields a
    more negative energy and a compatibility in ``(0, 1)``.  **Uncalibrated** —
    use for ranking, never as a quantitative Kd.
    """
    affinity = _affinity_profile(pol_seq)
    accessibility = _accessibility_profile(epsilon_seq, config)
    bins = min(affinity.size, accessibility.size)
    if bins == 0:
        return 0.0, 0.5
    overlap = float(np.dot(affinity[:bins], accessibility[:bins]) / bins)
    energy = -overlap
    compatibility = float(1.0 / (1.0 + np.exp(-(overlap - 0.3) / 0.1)))
    return energy, compatibility


def _as_genotype_map(obj, kind: str) -> dict[str, str]:
    """Coerce a genotype->sequence mapping from a dict or a tidy DataFrame."""
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return {str(key): str(value) for key, value in obj.items()}
    if isinstance(obj, pd.DataFrame):
        genotype_col = next(
            (c for c in ("genotype", "lineage", "clade", "id") if c in obj.columns), None
        )
        sequence_col = next(
            (c for c in (kind, "sequence", "seq", "consensus") if c in obj.columns), None
        )
        if genotype_col and sequence_col:
            return {
                str(row[genotype_col]): str(row[sequence_col])
                for _, row in obj.iterrows()
            }
    logger.warning("could not interpret %s genotype mapping; ignoring", kind)
    return {}


def compatibility_matrix(pol_by_genotype, epsilon_by_genotype, config: dict) -> pd.DataFrame:
    """Contact-map-energy compatibility for every (Pol, epsilon) genotype pair."""
    pol_map = _as_genotype_map(pol_by_genotype, "pol")
    eps_map = _as_genotype_map(epsilon_by_genotype, "epsilon")
    if not pol_map or not eps_map:
        return _empty(COMPATIBILITY_COLUMNS)

    predictor = str(get(config, "epsilon.compatibility.predictor", "contact_map_energy"))
    if predictor != "contact_map_energy":
        logger.warning("unknown compatibility predictor %r; using contact_map_energy", predictor)

    rows: list[dict[str, object]] = []
    for pol_genotype in sorted(pol_map):
        pairs = []
        for eps_genotype in sorted(eps_map):
            energy, compatibility = contact_map_energy(
                pol_map[pol_genotype], eps_map[eps_genotype], config
            )
            pairs.append((eps_genotype, energy, compatibility))
        ordered = sorted(pairs, key=lambda item: item[1])
        for rank, (eps_genotype, energy, compatibility) in enumerate(ordered, start=1):
            rows.append({
                "pol_genotype": pol_genotype,
                "epsilon_genotype": eps_genotype,
                "energy": float(energy),
                "compatibility": float(compatibility),
                "rank": int(rank),
            })
    frame = pd.DataFrame(rows, columns=COMPATIBILITY_COLUMNS)
    if len(frame):
        frame = frame.sort_values(["pol_genotype", "rank"]).reset_index(drop=True)
    return frame
