"""Drug-resistance catalogue annotation.

The resistance table maps reverse-transcriptase (rt) substitutions onto Pol
coordinates and flags whether each falls inside a canonical motif from
:data:`hbvpol.domain.CANONICAL_MOTIFS` (YMDD, motifs A–F, RNaseH motif C, …).

A curated catalogue may be supplied as a TSV via
``selection.drug_resistance.catalogue``.  When it is absent (or unreadable) a
small documented built-in table of well-characterised RT substitutions is used,
so the stage always produces annotated output offline.

rt→Pol numbering
----------------
The rt domain numbering used clinically is offset from the Pol (genotype A2)
numbering.  The default offset is ``346`` so that the catalytic YMDD motif maps
to Pol positions 549–552, matching ``CANONICAL_MOTIFS["RT_motif_C_YMDD"]``;
this is genotype-dependent and can be overridden with
``selection.drug_resistance.rt_pol_offset``.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Mapping

import pandas as pd

from ..config import get
from ..domain import CANONICAL_MOTIFS
from ..io import read_table
from ..pipeline import get_logger

__all__ = [
    "RESISTANCE_COLUMNS",
    "RT_TO_POL_OFFSET",
    "BUILTIN_RESISTANCE",
    "parse_rt_mutation",
    "canonical_motif_at",
    "load_catalogue",
    "resistance_table",
]

logger = get_logger("selection.resistance")

#: Full output schema of ``resistance.tsv`` (``lineage`` is added by the pipeline).
RESISTANCE_COLUMNS = [
    "lineage",
    "pol_position",
    "rt_mutation",
    "drug_class",
    "associated",
    "canonical_motif",
]

_RT_COLUMNS = [column for column in RESISTANCE_COLUMNS if column != "lineage"]

#: Default rt→Pol offset (see module docstring).
RT_TO_POL_OFFSET = 346

_MUTATION_RE = re.compile(r"^rt\s*([A-Za-z])(\d+)([A-Za-z*])$", re.IGNORECASE)

#: Built-in, documented RT resistance substitutions (mutation, drug class).
BUILTIN_RESISTANCE: tuple[tuple[str, str], ...] = (
    ("rtM204V", "lamivudine"),
    ("rtM204I", "lamivudine"),
    ("rtM204V", "entecavir"),
    ("rtM204I", "entecavir"),
    ("rtL180M", "lamivudine"),
    ("rtA181T", "adefovir"),
    ("rtA181V", "adefovir"),
    ("rtA181T", "tenofovir"),
    ("rtA181V", "tenofovir"),
    ("rtT184G", "entecavir"),
    ("rtS202G", "entecavir"),
    ("rtM250V", "entecavir"),
    ("rtN236T", "adefovir"),
    ("rtN236T", "tenofovir"),
    ("rtI169T", "entecavir"),
    ("rtV173L", "lamivudine"),
)


def parse_rt_mutation(mutation: str) -> tuple[str, int, str] | None:
    """Parse ``rtM204V`` into ``("M", 204, "V")`` (case-insensitive)."""
    match = _MUTATION_RE.match(str(mutation).strip())
    if not match:
        return None
    return match.group(1).upper(), int(match.group(2)), match.group(3).upper()


def canonical_motif_at(position: int) -> bool:
    """Whether a 1-based Pol position falls inside any canonical motif."""
    for start, end in CANONICAL_MOTIFS.values():
        if start <= position <= end:
            return True
    return False


def _rt_pol_offset(config: Mapping[str, object]) -> int:
    value = get(config, "selection.drug_resistance.rt_pol_offset", RT_TO_POL_OFFSET)
    try:
        return int(value)
    except (TypeError, ValueError):
        return RT_TO_POL_OFFSET


def _pick_column(frame: pd.DataFrame, candidates) -> str | None:
    lowered = {str(column).strip().lower(): column for column in frame.columns}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]
    return None


def _read_catalogue_file(path: Path) -> pd.DataFrame | None:
    try:
        frame = read_table(path)
    except Exception as error:
        logger.warning("could not read resistance catalogue %s (%s)", path, error)
        return None
    if frame is None or frame.empty:
        return None

    mutation_column = _pick_column(
        frame, ("rt_mutation", "mutation", "substitution", "rt", "name", "variant")
    )
    if mutation_column is None:
        logger.warning("resistance catalogue %s has no mutation column; using built-in table", path)
        return None
    drug_column = _pick_column(frame, ("drug_class", "drug", "drugs", "class", "antiviral"))
    associated_column = _pick_column(frame, ("associated", "known", "resistant", "resistance"))
    position_column = _pick_column(frame, ("pol_position", "position", "pol_pos"))

    normalised = pd.DataFrame()
    normalised["rt_mutation"] = frame[mutation_column].astype(str)
    normalised["drug_class"] = frame[drug_column].astype(str) if drug_column else ""
    if associated_column:
        normalised["associated"] = frame[associated_column].astype(bool)
    else:
        normalised["associated"] = True
    if position_column:
        normalised["pol_position"] = pd.to_numeric(frame[position_column], errors="coerce")
    return normalised


def load_catalogue(config: Mapping[str, object], root=None) -> pd.DataFrame:
    """Load the configured resistance catalogue, else the built-in table.

    Returns a frame with at least ``rt_mutation``, ``drug_class`` and
    ``associated`` columns (and optionally ``pol_position``).
    """
    configured = get(config, "selection.drug_resistance.catalogue", None)
    if configured:
        path = Path(str(configured))
        if not path.is_absolute() and root is not None:
            path = Path(root) / path
        if path.exists():
            loaded = _read_catalogue_file(path)
            if loaded is not None:
                logger.info("loaded %d resistance entries from %s", len(loaded), path)
                return loaded
        else:
            logger.warning("resistance catalogue %s not found; using built-in table", path)

    builtin = pd.DataFrame(BUILTIN_RESISTANCE, columns=["rt_mutation", "drug_class"])
    builtin["associated"] = True
    return builtin


def resistance_table(config: Mapping[str, object], root=None) -> pd.DataFrame:
    """Annotate resistance substitutions with Pol positions and motif flags.

    Returns columns ``pol_position, rt_mutation, drug_class, associated,
    canonical_motif`` (the pipeline adds ``lineage``).
    """
    catalogue = load_catalogue(config, root)
    offset = _rt_pol_offset(config)
    has_position = "pol_position" in catalogue.columns

    rows: list[dict] = []
    for _, entry in catalogue.iterrows():
        mutation = str(entry["rt_mutation"])
        parsed = parse_rt_mutation(mutation)
        if parsed is None:
            logger.warning("could not parse resistance mutation %r; skipping", mutation)
            continue
        _, rt_position, _ = parsed
        pol_position = None
        if has_position:
            value = entry.get("pol_position")
            if pd.notna(value):
                pol_position = int(value)
        if pol_position is None:
            pol_position = rt_position + offset

        associated = entry.get("associated", True)
        rows.append({
            "pol_position": pol_position,
            "rt_mutation": mutation,
            "drug_class": str(entry.get("drug_class", "") or ""),
            "associated": bool(associated) if pd.notna(associated) else True,
            "canonical_motif": canonical_motif_at(pol_position),
        })

    return pd.DataFrame(rows, columns=_RT_COLUMNS)
