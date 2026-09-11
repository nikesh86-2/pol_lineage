"""Deep mutational scanning (DMS) loading, annotation and criteria scoring.

The 2024 single-nucleotide-resolution HBV polymerase fitness map is the
experimental anchor of the project.  It is an **optional input**: when it is
absent the stage logs loudly and emits a correctly-schemad empty table so that
downstream stages can still run.

``score_criteria`` is the reusable core that turns a joined per-position table
(into which upstream conservation, structure and covariation scores have been
merged) into the six boolean mechanistic-target criteria.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import get
from ..domain import CANONICAL_MOTIFS, ConsequenceClass, dual_frame_consequence, translate
from ..pipeline import get_logger

__all__ = [
    "DMS_COLUMNS",
    "CRITERIA",
    "load_dms",
    "annotate_dms",
    "intolerant_sites",
    "score_criteria",
    "mechanistic_candidates",
]

logger = get_logger("fitness.dms")

#: Tidy schema expected at ``fitness.dms.map``.
DMS_COLUMNS = ["wt_nt", "position", "mut_nt", "fitness"]

#: The six mechanistic-target criteria, in report order.
CRITERIA = [
    "conserved_deep_hepadna",
    "intolerant_dms",
    "interface_or_hinge",
    "supported_covariation",
    "not_explained_by_surface_frame",
    "distinct_from_canonical_motifs",
]

_SURFACE_EXPLAINED = {
    ConsequenceClass.SYN_BOTH.value,
    ConsequenceClass.NONSYN_SURFACE.value,
}

_DEFAULT_ENTROPY_MAX = 0.7
_DEFAULT_COVARIATION_MIN = 0.05
_DEFAULT_INTOLERANCE_QUANTILE = 0.1


def _root(config: dict, root=None) -> Path:
    if root is not None:
        return Path(root)
    injected = get(config, "_root")
    return Path(injected) if injected else Path(".")


def _resolve(config: dict, dotted: str, root: Path) -> Path:
    value = get(config, dotted)
    if value is None:
        return Path("")
    path = Path(str(value))
    return path if path.is_absolute() else root / path


def load_dms(config: dict, root=None) -> pd.DataFrame:
    """Load the tidy DMS map, or return an empty schema if it is absent.

    A missing map is a *loud* condition: the 2024 DMS map must be supplied for
    the fitness-anchored candidates to be meaningful.
    """
    path = _resolve(config, "fitness.dms.map", _root(config, root))
    if not path or not path.exists():
        logger.error(
            "DMS map not found at %s — the 2024 HBV Pol DMS map must be supplied "
            "(fitness.dms.map); emitting an empty, correctly-schemad table",
            path or "<unset>",
        )
        return pd.DataFrame(columns=DMS_COLUMNS)
    try:
        frame = pd.read_csv(path, sep="\t")
    except Exception as error:  # pragma: no cover - defensive
        logger.error("could not read DMS map %s: %s", path, error)
        return pd.DataFrame(columns=DMS_COLUMNS)

    for column in DMS_COLUMNS:
        if column not in frame.columns:
            logger.warning("DMS map %s is missing column %r", path, column)
            frame[column] = np.nan
    for column in ("position", "fitness"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame[DMS_COLUMNS + [c for c in frame.columns if c not in DMS_COLUMNS]]


def _reference_sequence(config: dict, root: Path) -> str | None:
    inline = get(config, "fitness.reference_sequence")
    if inline:
        return str(inline).upper().replace("U", "T")
    reference_fasta = get(config, "fitness.dms.reference_fasta")
    if reference_fasta:
        candidate = Path(str(reference_fasta))
        if not candidate.is_absolute():
            candidate = root / candidate
        if candidate.exists():
            from ..io import read_fasta

            records = read_fasta(candidate)
            if records:
                return records[0].seq.upper().replace("U", "T")
    return None


def _orf_codon(sequence: str, position: int, orf_start_nt: int, genome_length: int) -> tuple[str, int]:
    """Return the codon (in the ORF frame) containing ``position`` and its offset."""
    if not sequence:
        return "", 0
    offset = (position - orf_start_nt) % genome_length
    start0 = (position - 1 - (offset % 3)) % genome_length
    indices = [(start0 + k) % genome_length for k in range(3)]
    codon = "".join(sequence[i] for i in indices)
    within = (position - 1 - start0) % genome_length
    return codon, int(within)


def annotate_dms(dms: pd.DataFrame, config: dict, root=None) -> pd.DataFrame:
    """Annotate DMS rows with Pol codon/position, surface-frame consequence and intent."""
    root = _root(config, root)
    frame = dms.copy()
    if frame.empty:
        for column in (
            "pol_position", "pol_codon", "pol_aa_wt", "pol_aa_mut",
            "surface_consequence", "intent", "is_intolerant",
        ):
            frame[column] = pd.Series(dtype="object")
        return frame

    genome_length = int(get(config, "reference.length", 3215) or 3215)
    pol_start_nt = int(get(config, "reference.pol_start_nt", 2307) or 2307)
    surface_offset = int(get(config, "reference.surface_frame_offset", 1) or 1)
    sequence = _reference_sequence(config, root)

    quantile = float(get(config, "fitness.intolerance_quantile", _DEFAULT_INTOLERANCE_QUANTILE))
    fitness_values = pd.to_numeric(frame.get("fitness"), errors="coerce") if "fitness" in frame else None
    threshold = None
    if fitness_values is not None and fitness_values.notna().any():
        threshold = float(fitness_values.quantile(quantile))

    rows: list[dict[str, object]] = []
    for _, row in frame.iterrows():
        position = row.get("position")
        record: dict[str, object] = {}
        if pd.isna(position):
            rows.append(record)
            continue
        position = int(position)
        record["pol_position"] = ((position - pol_start_nt) % genome_length) // 3 + 1
        if sequence:
            pol_codon, within = _orf_codon(sequence, position, pol_start_nt, genome_length)
            surface_codon, surface_within = _orf_codon(
                sequence, position, pol_start_nt + surface_offset, genome_length
            )
            mut_nt = str(row.get("mut_nt") or "").upper()
            alt_pol = pol_codon
            alt_surface = surface_codon
            if 0 <= within < 3 and len(mut_nt) == 1:
                alt_pol = pol_codon[:within] + mut_nt + pol_codon[within + 1:]
            if 0 <= surface_within < 3 and len(mut_nt) == 1:
                alt_surface = surface_codon[:surface_within] + mut_nt + surface_codon[surface_within + 1:]
            record["pol_codon"] = pol_codon
            record["pol_aa_wt"] = translate(pol_codon)
            record["pol_aa_mut"] = translate(alt_pol)
            consequence = dual_frame_consequence(pol_codon, alt_pol, surface_codon, alt_surface)
            record["surface_consequence"] = consequence.value
        else:
            record["pol_codon"] = None
            record["pol_aa_wt"] = None
            record["pol_aa_mut"] = None
            record["surface_consequence"] = ConsequenceClass.UNKNOWN.value

        fitness = row.get("fitness")
        is_intolerant = bool(threshold is not None and pd.notna(fitness) and float(fitness) <= threshold)
        record["is_intolerant"] = is_intolerant
        if record.get("pol_aa_mut") == "*":
            record["intent"] = "premature_stop"
        elif is_intolerant:
            record["intent"] = "loss_of_function_candidate"
        elif pd.notna(fitness) and float(fitness) >= 0.9:
            record["intent"] = "tolerated"
        else:
            record["intent"] = "intermediate"
        rows.append(record)

    annotations = pd.DataFrame(rows)
    result = pd.concat([frame.reset_index(drop=True), annotations.reset_index(drop=True)], axis=1)
    return result


def intolerant_sites(dms: pd.DataFrame, quantile: float = 0.1) -> set[int]:
    """Return the positions in the least-fit tail of the distribution."""
    if dms is None or len(dms) == 0:
        return set()
    position_col = "pol_position" if "pol_position" in dms.columns else "position"
    if position_col not in dms.columns or "fitness" not in dms.columns:
        return set()
    fitness = pd.to_numeric(dms["fitness"], errors="coerce")
    valid = dms.assign(_fitness=fitness).dropna(subset=["_fitness"])
    if valid.empty:
        return set()
    threshold = float(valid["_fitness"].quantile(quantile))
    selected = valid[valid["_fitness"] <= threshold][position_col].dropna()
    return {int(value) for value in selected}


# --------------------------------------------------------------------------- #
# criteria scoring
# --------------------------------------------------------------------------- #
def _canonical_intervals(config: dict, motifs: dict | None = None) -> list[tuple[int, int]]:
    """Interpret the ``distinct_from_canonical_motifs`` tokens as intervals."""
    motifs = motifs or CANONICAL_MOTIFS
    tokens = get(config, "fitness.criteria.distinct_from_canonical_motifs", list(motifs.keys()))
    intervals: list[tuple[int, int]] = []
    for token in tokens or []:
        text = str(token).strip()
        upper = text.upper()
        if upper in {"YMDD", "RT_MOTIF_C_YMDD"}:
            intervals.append(tuple(motifs["RT_motif_C_YMDD"]))
            continue
        letters = re.fullmatch(r"([A-Za-z])\s*[-–]\s*([A-Za-z])", text)
        if letters:
            low, high = sorted((letters.group(1).upper(), letters.group(2).upper()))
            for key, span in motifs.items():
                name = key.upper()
                if re.search(rf"_{low}$", name) or re.search(rf"_{high}$", name):
                    intervals.append(tuple(span))
                elif low <= key[-1].upper() <= high and "MOTIF" in name:
                    intervals.append(tuple(span))
            continue
        matched = False
        for key, span in motifs.items():
            if upper in key.upper() or key.upper() in upper:
                intervals.append(tuple(span))
                matched = True
        if not matched:
            logger.warning("unknown canonical-motif token %r; ignoring", token)
    return intervals


def _truthy(value, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() not in {"false", "0", "no", ""}
    return bool(value)


def _enabled(config: dict, name: str) -> bool:
    value = get(config, f"fitness.criteria.{name}")
    if value is None:
        # A list value (canonical motifs) means "enabled"; otherwise default on.
        return True
    return _truthy(value)


def score_criteria(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Add a boolean per criterion plus ``passes_criteria`` to a per-position table.

    Missing input columns make the corresponding criterion ``False`` rather than
    raising, so the function is safe to call from the atlas stage on a sparse,
    outer-joined table.
    """
    result = df.copy()
    if len(result) == 0:
        for name in CRITERIA:
            result[name] = pd.Series(dtype=bool)
        result["passes_criteria"] = pd.Series(dtype=bool)
        result["n_criteria_met"] = pd.Series(dtype=int)
        return result

    entropy_max = float(get(config, "fitness.criteria.conservation_entropy_max", _DEFAULT_ENTROPY_MAX))
    covariation_min = float(get(config, "fitness.criteria.covariation_min", _DEFAULT_COVARIATION_MIN))

    # conservation
    if "aa_entropy" in result.columns:
        entropy = pd.to_numeric(result["aa_entropy"], errors="coerce")
        conserved = entropy <= entropy_max
    elif "conservation" in result.columns:
        conserved = pd.to_numeric(result["conservation"], errors="coerce") >= (1.0 - entropy_max)
    else:
        conserved = pd.Series(False, index=result.index)
    result["conserved_deep_hepadna"] = conserved.fillna(False)

    # DMS intolerance
    if "is_intolerant" in result.columns:
        intolerant = result["is_intolerant"].fillna(False).astype(bool)
    elif "dms_intolerance" in result.columns:
        cutoff = float(get(config, "fitness.criteria.intolerance_threshold", 0.5))
        intolerant = pd.to_numeric(result["dms_intolerance"], errors="coerce").fillna(0.0) >= cutoff
    elif "fitness" in result.columns:
        fitness = pd.to_numeric(result["fitness"], errors="coerce")
        explicit = get(config, "fitness.criteria.intolerance_fitness_max")
        cutoff = float(explicit) if explicit is not None else (
            float(fitness.quantile(_DEFAULT_INTOLERANCE_QUANTILE)) if fitness.notna().any() else np.nan
        )
        intolerant = (fitness <= cutoff) if pd.notna(cutoff) else pd.Series(False, index=result.index)
    else:
        intolerant = pd.Series(False, index=result.index)
    result["intolerant_dms"] = intolerant.fillna(False).astype(bool)

    # interface or hinge
    interface = pd.Series(False, index=result.index)
    have_interface = False
    for column in ("is_interface", "is_hinge", "interface_or_hinge"):
        if column in result.columns:
            interface = interface | result[column].fillna(False).astype(bool)
            have_interface = True
    result["interface_or_hinge"] = interface if have_interface else pd.Series(False, index=result.index)

    # covariation support
    if "covariation_support" in result.columns:
        support = pd.to_numeric(result["covariation_support"], errors="coerce").fillna(0.0)
        covariation = support >= covariation_min
    elif "covariation" in result.columns:
        support = pd.to_numeric(result["covariation"], errors="coerce").fillna(0.0)
        covariation = support >= covariation_min
    else:
        covariation = pd.Series(False, index=result.index)
    result["supported_covariation"] = covariation.fillna(False).astype(bool)

    # not explained by the surface frame
    if "surface_consequence" in result.columns:
        consequence = result["surface_consequence"].astype("string").str.lower()
        not_explained = ~consequence.isin(_SURFACE_EXPLAINED)
        not_explained = not_explained.fillna(False)
    else:
        not_explained = pd.Series(False, index=result.index)
    result["not_explained_by_surface_frame"] = not_explained.astype(bool)

    # distinct from canonical motifs
    intervals = _canonical_intervals(config)
    if "pol_position" in result.columns and intervals:
        positions = pd.to_numeric(result["pol_position"], errors="coerce")

        def _distinct(value) -> bool:
            if pd.isna(value):
                return False
            return not any(start <= int(value) <= end for start, end in intervals)

        result["distinct_from_canonical_motifs"] = positions.map(_distinct)
    elif "pol_position" in result.columns:
        result["distinct_from_canonical_motifs"] = True
    else:
        result["distinct_from_canonical_motifs"] = pd.Series(False, index=result.index)

    enabled = [name for name in CRITERIA if _enabled(config, name)]
    result["n_criteria_met"] = result[CRITERIA].astype(bool).sum(axis=1)
    if enabled:
        result["passes_criteria"] = result[enabled].astype(bool).all(axis=1)
    else:
        result["passes_criteria"] = True
    return result


def mechanistic_candidates(atlas_like_table: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Rows satisfying every enabled criterion, with a boolean per criterion."""
    scored = score_criteria(atlas_like_table, config)
    if len(scored) == 0:
        return scored
    return scored[scored["passes_criteria"].astype(bool)].reset_index(drop=True)
