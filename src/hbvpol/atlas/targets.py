"""Target ranking, interaction networks and conformational-switch detection.

All three functions consume the unified per-position atlas produced by
:func:`hbvpol.atlas.pipeline.run` and are pure (no I/O, no side effects), so
they are straightforward to unit-test on small synthetic frames.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import get
from ..domain import domain_of
from ..pipeline import get_logger

__all__ = [
    "RANKED_TARGET_COLUMNS",
    "INTERACTION_COLUMNS",
    "SWITCH_COLUMNS",
    "rank_targets",
    "interaction_network",
    "conformational_switch_candidates",
]

logger = get_logger("atlas.targets")

RANKED_TARGET_COLUMNS = [
    "rank",
    "lineage",
    "pol_position",
    "score",
    "norm_conservation",
    "norm_dms_intolerance",
    "norm_structural_interface",
    "norm_covariation",
]

INTERACTION_COLUMNS = [
    "lineage",
    "pol_position",
    "partner_position",
    "interaction_type",
    "support",
    "interaction_class",
]

SWITCH_COLUMNS = [
    "lineage",
    "pol_position",
    "domain",
    "pol_aa",
    "entropy",
    "interface_flag",
    "lineage_specific",
    "switch_score",
]

_DEFAULT_WEIGHTS = {
    "conservation": 0.3,
    "dms_intolerance": 0.3,
    "structural_interface": 0.2,
    "covariation": 0.2,
}


def _numeric(df: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(df[column], errors="coerce")


def _minmax(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    values = values.fillna(0.0)
    low, high = values.min(), values.max()
    if high > low:
        return (values - low) / (high - low)
    # Constant column: neutral 0.5 if non-zero, else 0.0.
    return pd.Series(0.5 if high > 0 else 0.0, index=values.index, dtype=float)


def _conservation_component(dist: pd.DataFrame) -> pd.Series:
    if "conservation" in dist.columns:
        return _minmax(dist["conservation"])
    if "aa_entropy" in dist.columns:
        return 1.0 - _minmax(dist["aa_entropy"])
    return pd.Series(0.0, index=dist.index)


def _intolerance_component(dist: pd.DataFrame) -> pd.Series:
    if "dms_intolerance" in dist.columns:
        return _minmax(dist["dms_intolerance"])
    if "is_intolerant" in dist.columns:
        return dist["is_intolerant"].fillna(False).astype(float)
    if "fitness" in dist.columns:
        return 1.0 - _minmax(dist["fitness"])
    return pd.Series(0.0, index=dist.index)


def _interface_component(dist: pd.DataFrame) -> pd.Series:
    if "structural_interface" in dist.columns:
        return _minmax(dist["structural_interface"])
    combined = pd.Series(0.0, index=dist.index)
    found = False
    for column in ("is_interface", "is_hinge"):
        if column in dist.columns:
            combined = np.maximum(combined, dist[column].fillna(False).astype(float))
            found = True
    return combined if found else pd.Series(0.0, index=dist.index)


def _covariation_component(dist: pd.DataFrame) -> pd.Series:
    for column in ("covariation_support", "covariation"):
        if column in dist.columns:
            return _minmax(dist[column])
    return pd.Series(0.0, index=dist.index)


def rank_targets(atlas: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Weighted ranking over normalised {conservation, DMS intolerance, interface, covariation}."""
    if atlas is None or len(atlas) == 0:
        return pd.DataFrame(columns=RANKED_TARGET_COLUMNS)
    if "pol_position" not in atlas.columns:
        logger.warning("atlas has no pol_position column; cannot rank targets")
        return pd.DataFrame(columns=RANKED_TARGET_COLUMNS)

    dist = atlas.copy().reset_index(drop=True)
    if "lineage" not in dist.columns:
        dist["lineage"] = "all"

    weights = dict(get(config, "atlas.ranked_targets.weights", _DEFAULT_WEIGHTS) or _DEFAULT_WEIGHTS)
    weights = {key: float(weights.get(key, 0.0)) for key in _DEFAULT_WEIGHTS}
    total = sum(weights.values()) or 1.0

    components = {
        "conservation": _conservation_component(dist),
        "dms_intolerance": _intolerance_component(dist),
        "structural_interface": _interface_component(dist),
        "covariation": _covariation_component(dist),
    }
    score = pd.Series(0.0, index=dist.index)
    for key, values in components.items():
        score += (weights[key] / total) * values

    dist["norm_conservation"] = components["conservation"]
    dist["norm_dms_intolerance"] = components["dms_intolerance"]
    dist["norm_structural_interface"] = components["structural_interface"]
    dist["norm_covariation"] = components["covariation"]
    dist["score"] = score

    top_n = int(get(config, "atlas.ranked_targets.top_n", 50) or 50)
    ranked = dist.sort_values(["score", "pol_position"], ascending=[False, True]).head(top_n)
    ranked = ranked.reset_index(drop=True)
    ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))

    keep = [column for column in RANKED_TARGET_COLUMNS if column in ranked.columns]
    extra = [
        column for column in ("domain", "pol_aa", "pol_residue", "aa_entropy", "fitness")
        if column in ranked.columns and column not in keep
    ]
    return ranked[keep + extra]


def _candidate_column(df: pd.DataFrame) -> str | None:
    for column in ("covariation_partner", "partner_position", "contact_partner", "structural_contacts"):
        if column in df.columns:
            return column
    return None


def interaction_network(atlas: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Conserved and lineage-specific residue contacts from covariation/structure."""
    if atlas is None or len(atlas) == 0 or "pol_position" not in atlas.columns:
        return pd.DataFrame(columns=INTERACTION_COLUMNS)
    partner_col = _candidate_column(atlas)
    if partner_col is None or "covariation_support" not in atlas.columns:
        return pd.DataFrame(columns=INTERACTION_COLUMNS)

    min_support = float(get(config, "atlas.interaction_network.min_support", 0.05))
    frame = atlas.copy()
    frame["partner_position"] = pd.to_numeric(frame[partner_col], errors="coerce")
    frame["support"] = pd.to_numeric(frame["covariation_support"], errors="coerce")
    frame = frame.dropna(subset=["partner_position", "support"])
    frame = frame[frame["support"] >= min_support]
    if frame.empty:
        return pd.DataFrame(columns=INTERACTION_COLUMNS)

    if "lineage" not in frame.columns:
        frame["lineage"] = "all"
    frame["interaction_type"] = "covariation"

    lineage_counts = frame.groupby(["pol_position", "partner_position"])["lineage"].nunique()
    frame["interaction_class"] = [
        "conserved" if lineage_counts.get((row["pol_position"], row["partner_position"]), 0) >= 2
        else "lineage_specific"
        for _, row in frame.iterrows()
    ]
    return frame[INTERACTION_COLUMNS].reset_index(drop=True)


def conformational_switch_candidates(atlas: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Flag lineage-specific spacer substitutions near domain interfaces/hinges."""
    if atlas is None or len(atlas) == 0 or "pol_position" not in atlas.columns:
        return pd.DataFrame(columns=SWITCH_COLUMNS)

    frame = atlas.copy()
    if "lineage" not in frame.columns:
        frame["lineage"] = "all"

    if "domain" not in frame.columns:
        spans = None
        frame["domain"] = frame["pol_position"].map(
            lambda position: _safe_domain(position, spans)
        )

    switch_domains = [str(d).lower() for d in get(config, "atlas.switch_domains", ["spacer"]) or ["spacer"]]
    domain_mask = frame["domain"].astype("string").str.lower().isin(switch_domains)

    interface = pd.Series(False, index=frame.index)
    for column in ("is_hinge", "is_interface"):
        if column in frame.columns:
            interface = interface | frame[column].fillna(False).astype(bool)

    if "lineage_specific" in frame.columns:
        lineage_specific = frame["lineage_specific"].fillna(False).astype(bool)
    elif "genotype_specific" in frame.columns:
        lineage_specific = frame["genotype_specific"].fillna(False).astype(bool)
    elif "fst" in frame.columns:
        min_fst = float(get(config, "selection.genotype_specificity.min_fst", 0.25))
        lineage_specific = pd.to_numeric(frame["fst"], errors="coerce").fillna(0.0) >= min_fst
    elif "aa_entropy" in frame.columns:
        lineage_specific = pd.to_numeric(frame["aa_entropy"], errors="coerce").fillna(0.0) > 0.0
    else:
        lineage_specific = pd.Series(False, index=frame.index)

    mask = domain_mask & interface & lineage_specific
    selected = frame[mask].copy()
    if selected.empty:
        return pd.DataFrame(columns=SWITCH_COLUMNS)

    if "aa_entropy" in selected.columns:
        entropy = pd.to_numeric(selected["aa_entropy"], errors="coerce").fillna(0.0)
    else:
        entropy = pd.Series(0.0, index=selected.index)
    selected["switch_score"] = _minmax(entropy)

    output = pd.DataFrame({
        "lineage": selected.get("lineage", "all"),
        "pol_position": selected["pol_position"],
        "domain": selected["domain"],
        "pol_aa": selected["pol_aa"] if "pol_aa" in selected.columns else selected.get("pol_residue"),
        "entropy": entropy,
        "interface_flag": True,
        "lineage_specific": True,
        "switch_score": selected["switch_score"],
    })
    return output.sort_values(
        ["switch_score", "pol_position"], ascending=[False, True]
    ).reset_index(drop=True)


def _safe_domain(position, spans=None) -> str:
    try:
        return domain_of(int(position), spans).value if spans else domain_of(int(position)).value
    except (ValueError, TypeError):
        return "unknown"
