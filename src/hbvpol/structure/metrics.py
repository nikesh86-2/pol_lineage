"""Structure-level metrics on PDB/mmCIF files.

Everything here is a *pure* function of a structural file (plus, occasionally, a
configuration mapping): the heavy dependencies (``Bio.PDB``, ``freesasa``,
``scipy``) are imported inside functions so that importing this module — and
therefore the whole stage — works with none of them installed.

Conventions
-----------
* pLDDT is read from the B-factor column, which is the convention used by
  AlphaFold2/3 and ColabFold (values 0–100, where higher is more confident).
* Residues are numbered sequentially (1-based) in file order; alignments are
  mapped onto that ordering by :func:`compute_metrics`.
* A contiguous run of residues below ``structure.hinge_plddt`` is reported as a
  *candidate hinge* — a flexible, biologically meaningful region — rather than
  treated as a failed prediction.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..config import get
from ..domain import CANONICAL_MOTIFS, DEFAULT_DOMAIN_SPANS, DomainSpan, PolDomain

__all__ = [
    "HINGE_PLDDT_DEFAULT",
    "plddt_from_pdb",
    "pae_from_json",
    "per_residue_sasa",
    "domain_orientation",
    "catalytic_geometry",
    "nucleic_acid_compatibility",
    "conserved_packing",
    "compute_metrics",
    "candidate_hinges",
]

#: Fallback pLDDT threshold when ``structure.hinge_plddt`` is absent.
HINGE_PLDDT_DEFAULT = 70.0

_VAN_DER_WAALS = {
    "H": 1.20,
    "C": 1.70,
    "N": 1.55,
    "O": 1.52,
    "S": 1.80,
    "P": 1.80,
    "F": 1.47,
    "CL": 1.75,
    "BR": 1.85,
    "I": 1.98,
}
_DEFAULT_RADIUS = 1.70

_NUCLEIC_RESIDUES = {
    "A", "C", "G", "U", "I", "DA", "DC", "DG", "DT", "DI", "DU",
    "RA", "RC", "RG", "RU",
}


# --------------------------------------------------------------------------- #
# parsing helpers (Bio.PDB imported lazily)
# --------------------------------------------------------------------------- #
def _load_structure(path: str | Path):
    from Bio.PDB import MMCIFParser, PDBParser

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    name = path.stem
    if path.suffix.lower() in {".cif", ".mmcif"}:
        return MMCIFParser(QUIET=True).get_structure(name, str(path))
    return PDBParser(QUIET=True).get_structure(name, str(path))


def _is_aa(residue) -> bool:
    from Bio.PDB.Polypeptide import is_aa

    return bool(is_aa(residue))


def _protein_residues(structure):
    """Yield ``(chain_id, resid, resname, [atoms])`` for amino-acid residues."""
    for model in structure:
        for chain in model:
            for residue in chain:
                if _is_aa(residue):
                    yield chain.id, residue.id[1], residue.get_resname(), list(residue)
        break


def _nucleic_residues(structure):
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.get_resname().strip().upper() in _NUCLEIC_RESIDUES:
                    yield chain.id, residue.id[1], residue.get_resname(), list(residue)
        break


def _atom_radius(atom) -> float:
    element = (getattr(atom, "element", "") or "").strip().upper()
    if not element:
        element = atom.get_name().strip()[:1].upper()
    return _VAN_DER_WAALS.get(element, _DEFAULT_RADIUS)


# --------------------------------------------------------------------------- #
# per-model metrics
# --------------------------------------------------------------------------- #
def plddt_from_pdb(path: str | Path) -> np.ndarray:
    """Per-residue pLDDT from the B-factor column (CA atoms where available)."""
    structure = _load_structure(path)
    values: list[float] = []
    for _, _, _, atoms in _protein_residues(structure):
        chosen = None
        for atom in atoms:
            if atom.get_name().strip() == "CA":
                chosen = atom
                break
        if chosen is None and atoms:
            chosen = atoms[0]
        if chosen is not None:
            values.append(float(getattr(chosen, "bfactor", 0.0) or 0.0))
    return np.asarray(values, dtype=float)


def pae_from_json(path: str | Path) -> np.ndarray:
    """Load a predicted-aligned-error matrix from an AlphaFold/ColabFold JSON."""
    import json

    path = Path(path)
    if not path.exists():
        return np.empty((0, 0), dtype=float)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return np.empty((0, 0), dtype=float)
    if isinstance(payload, list):
        payload = payload[0] if payload else {}
    matrix = None
    if isinstance(payload, dict):
        for key in ("predicted_aligned_error", "pae", "distance_matrix", "pae_matrix"):
            if key in payload:
                matrix = payload[key]
                break
    if matrix is None:
        return np.empty((0, 0), dtype=float)
    try:
        return np.asarray(matrix, dtype=float)
    except (TypeError, ValueError):
        return np.empty((0, 0), dtype=float)


def _sphere_points(n_points: int) -> np.ndarray:
    i = np.arange(n_points, dtype=float) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n_points)
    theta = np.pi * (1.0 + 5.0 ** 0.5) * i
    return np.column_stack([
        np.sin(phi) * np.cos(theta),
        np.sin(phi) * np.sin(theta),
        np.cos(phi),
    ])


def _shrake_rupley(coords: np.ndarray, radii: np.ndarray, probe: float = 1.4, n_points: int = 92) -> np.ndarray:
    """Numerical solvent-accessible surface per atom (Å²)."""
    n_atoms = len(coords)
    if n_atoms == 0:
        return np.zeros(0, dtype=float)
    points = _sphere_points(n_points)
    sasa = np.zeros(n_atoms, dtype=float)
    # Pairwise distances with a generous neighbour cutoff.
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=-1))
    for i in range(n_atoms):
        radius_i = radii[i] + probe
        neighbours = np.where((dist[i] <= radius_i + radii + probe) & (np.arange(n_atoms) != i))[0]
        sphere = coords[i] + radius_i * points
        accessible = np.ones(len(points), dtype=bool)
        for j in neighbours:
            d = np.sqrt(((sphere - coords[j]) ** 2).sum(axis=1))
            accessible &= d >= (radii[j] + probe)
        sasa[i] = 4.0 * np.pi * radius_i ** 2 * accessible.mean()
    return sasa


def per_residue_sasa(path: str | Path) -> np.ndarray:
    """Per-residue solvent-accessible surface area (Å²).

    Uses ``freesasa`` when importable *and* the input is a PDB file, otherwise a
    self-contained Shrake–Rupley implementation so the metric is always
    available offline.
    """
    path = Path(path)
    if path.suffix.lower() in {".pdb", ".ent"}:
        try:
            import freesasa  # type: ignore

            structure = freesasa.Structure(str(path))
            result = freesasa.calc(structure)
            areas = result.residueAreas()
            values = []
            for chain in areas.values():
                for area in chain.values():
                    values.append(float(area.total))
            if values:
                return np.asarray(values, dtype=float)
        except Exception:  # pragma: no cover - fall through to Shrake-Rupley
            pass

    structure = _load_structure(path)
    coords: list[list[float]] = []
    radii: list[float] = []
    residue_index: list[int] = []
    for idx, (_, _, _, atoms) in enumerate(_protein_residues(structure)):
        for atom in atoms:
            coords.append(list(atom.get_coord()))
            radii.append(_atom_radius(atom))
            residue_index.append(idx)
    if not coords:
        return np.zeros(0, dtype=float)
    atom_sasa = _shrake_rupley(np.asarray(coords, dtype=float), np.asarray(radii, dtype=float))
    n_residues = max(residue_index) + 1
    out = np.zeros(n_residues, dtype=float)
    for value, idx in zip(atom_sasa, residue_index):
        out[idx] += value
    return out


def domain_orientation(path: str | Path, spans=DEFAULT_DOMAIN_SPANS) -> dict:
    """Inter-domain geometry: per-domain centroids and a hinge angle."""
    structure = _load_structure(path)
    positions: dict[str, list[list[float]]] = {}
    for idx, (_, _, _, atoms) in enumerate(_protein_residues(structure), start=1):
        ca = next((a for a in atoms if a.get_name().strip() == "CA"), atoms[0] if atoms else None)
        if ca is None:
            continue
        domain = next((span.domain.value for span in spans if span.contains(idx)), None)
        if domain is None:
            continue
        positions.setdefault(domain, []).append(list(ca.get_coord()))

    centroids = {
        domain: np.asarray(coords, dtype=float).mean(axis=0).tolist()
        for domain, coords in positions.items()
        if coords
    }
    angle = None
    tp = centroids.get(PolDomain.TP.value)
    rt = centroids.get(PolDomain.RT.value)
    rnaseh = centroids.get(PolDomain.RNASEH.value)
    if tp is not None and rt is not None and rnaseh is not None:
        v1 = np.asarray(rt) - np.asarray(tp)
        v2 = np.asarray(rnaseh) - np.asarray(tp)
        denom = np.linalg.norm(v1) * np.linalg.norm(v2)
        if denom > 0:
            cosine = float(np.clip(np.dot(v1, v2) / denom, -1.0, 1.0))
            angle = float(np.degrees(np.arccos(cosine)))
    return {
        "centroids": centroids,
        "angle_deg": angle,
        "n_domains": len(centroids),
    }


def _motif_span(motifs, key: str):
    for name, span in motifs.items():
        if name == key:
            return tuple(span)
    return None


def catalytic_geometry(path: str | Path, motifs: dict | None = None) -> float | None:
    """CA–CA distance (Å) between the priming Tyr and the RT YMDD motif."""
    motifs = motifs or CANONICAL_MOTIFS
    tyrosine = _motif_span(motifs, "priming_tyrosine")
    ymdd = _motif_span(motifs, "RT_motif_C_YMDD")
    if tyrosine is None or ymdd is None:
        return None

    structure = _load_structure(path)
    tyro_coords: list[list[float]] = []
    ymdd_coords: list[list[float]] = []
    for idx, (_, _, _, atoms) in enumerate(_protein_residues(structure), start=1):
        ca = next((a for a in atoms if a.get_name().strip() == "CA"), None)
        if ca is None:
            continue
        coord = list(ca.get_coord())
        if tyrosine[0] <= idx <= tyrosine[1]:
            tyro_coords.append(coord)
        if ymdd[0] <= idx <= ymdd[1]:
            ymdd_coords.append(coord)
    if not tyro_coords or not ymdd_coords:
        return None
    a = np.asarray(tyro_coords, dtype=float)
    b = np.asarray(ymdd_coords, dtype=float)
    return float(np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=-1)).min())


def nucleic_acid_compatibility(path: str | Path, nucleic_chain_ids=None, cutoff: float = 4.5) -> dict:
    """How well the protein is positioned to contact a nucleic-acid chain."""
    structure = _load_structure(path)
    protein_atoms: list[list[float]] = []
    for _, _, _, atoms in _protein_residues(structure):
        for atom in atoms:
            protein_atoms.append(list(atom.get_coord()))

    allowed = {str(c) for c in nucleic_chain_ids} if nucleic_chain_ids else None
    nucleic_atoms: list[list[float]] = []
    chains: set[str] = set()
    for chain_id, _, _, atoms in _nucleic_residues(structure):
        if allowed is not None and chain_id not in allowed:
            continue
        for atom in atoms:
            nucleic_atoms.append(list(atom.get_coord()))
        chains.add(chain_id)

    result = {
        "has_nucleic": bool(chains),
        "n_nucleic_chains": len(chains),
        "n_nucleic_atoms": len(nucleic_atoms),
        "contact_fraction": 0.0,
        "min_distance": None,
    }
    if not chains or not nucleic_atoms or not protein_atoms:
        return result

    protein = np.asarray(protein_atoms, dtype=float)
    nucleic = np.asarray(nucleic_atoms, dtype=float)
    diff = protein[:, None, :] - nucleic[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=-1))
    result["min_distance"] = float(dist.min())
    contacting = (dist.min(axis=1) <= cutoff)
    result["contact_fraction"] = float(contacting.mean())
    return result


def conserved_packing(path: str | Path, conservation_vector) -> float | None:
    """Pearson correlation between per-residue burial and conservation."""
    vector = np.asarray(list(conservation_vector), dtype=float)
    if vector.size == 0:
        return None
    sasa = per_residue_sasa(path)
    n = min(len(sasa), len(vector))
    if n < 3:
        return None
    burial = 1.0 - (sasa[:n] / (sasa[:n].max() or 1.0))
    values = vector[:n]
    if np.std(burial) == 0 or np.std(values) == 0:
        return None
    corr = float(np.corrcoef(burial, values)[0, 1])
    return None if np.isnan(corr) else corr


# --------------------------------------------------------------------------- #
# hinge calling
# --------------------------------------------------------------------------- #
def candidate_hinges(plddt, threshold: float | None = None, min_length: int = 1) -> list[tuple[int, int]]:
    """Return 1-based inclusive intervals of low-pLDDT (hinge) residues.

    ``None`` values and NaNs are treated as unassessed rather than low, so a
    truncated model does not manufacture hinges.
    """
    if threshold is None:
        threshold = HINGE_PLDDT_DEFAULT
    values = np.asarray(plddt, dtype=float)
    if values.size == 0:
        return []
    low = np.nan_to_num(values, nan=threshold + 1.0) < float(threshold)
    hinges: list[tuple[int, int]] = []
    start: int | None = None
    for i, flag in enumerate(low):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            if i - start >= min_length:
                hinges.append((start + 1, i))
            start = None
    if start is not None and len(low) - start >= min_length:
        hinges.append((start + 1, len(low)))
    return hinges


# --------------------------------------------------------------------------- #
# aggregation
# --------------------------------------------------------------------------- #
def _spans(config: dict | None):
    if config:
        raw = get(config, "structure.domain_spans")
        if raw:
            spans = []
            for item in raw:
                if isinstance(item, dict) and "domain" in item:
                    spans.append(DomainSpan(PolDomain.parse(item["domain"]), int(item["start"]), int(item["end"])))
            if spans:
                return tuple(spans)
    return DEFAULT_DOMAIN_SPANS


def compute_metrics(model_path: str | Path, config: dict | None = None) -> dict:
    """Assemble a flat metric record for one model, degrading per metric.

    Any metric that cannot be computed is recorded as ``None`` with an
    ``error_<metric>`` note instead of aborting the record.
    """
    config = config or {}
    path = Path(model_path)
    record: dict[str, object] = {
        "model": path.name,
        "path": str(path),
        "status": "ok",
        "n_residues": 0,
        "plddt_mean": None,
        "plddt_min": None,
        "plddt_max": None,
        "pae_mean": None,
        "sasa_mean": None,
        "orientation_angle_deg": None,
        "catalytic_distance": None,
        "nucleic_has_chain": False,
        "nucleic_contact_fraction": None,
        "packing_correlation": None,
    }

    try:
        plddt = plddt_from_pdb(path)
        record["n_residues"] = int(plddt.size)
        if plddt.size:
            record["plddt_mean"] = float(np.nanmean(plddt))
            record["plddt_min"] = float(np.nanmin(plddt))
            record["plddt_max"] = float(np.nanmax(plddt))
    except Exception as error:  # pragma: no cover - defensive
        record["status"] = "error"
        record["error_plddt"] = str(error)

    pae_path = path.with_suffix(".json")
    if not pae_path.exists():
        candidate = path.parent / (path.stem + "_pae.json")
        pae_path = candidate if candidate.exists() else pae_path
    try:
        pae = pae_from_json(pae_path)
        if pae.size:
            record["pae_mean"] = float(np.nanmean(pae))
    except Exception as error:  # pragma: no cover - defensive
        record["error_pae"] = str(error)

    try:
        sasa = per_residue_sasa(path)
        if sasa.size:
            record["sasa_mean"] = float(np.nanmean(sasa))
    except Exception as error:  # pragma: no cover - defensive
        record["error_sasa"] = str(error)

    try:
        orientation = domain_orientation(path, _spans(config))
        record["orientation_angle_deg"] = orientation.get("angle_deg")
        record["n_domains"] = orientation.get("n_domains")
    except Exception as error:  # pragma: no cover - defensive
        record["error_orientation"] = str(error)

    try:
        record["catalytic_distance"] = catalytic_geometry(path)
    except Exception as error:  # pragma: no cover - defensive
        record["error_catalytic"] = str(error)

    try:
        nucleic = nucleic_acid_compatibility(
            path, get(config, "structure.metrics.nucleic_chains")
        )
        record["nucleic_has_chain"] = nucleic["has_nucleic"]
        record["nucleic_contact_fraction"] = nucleic["contact_fraction"]
        record["nucleic_min_distance"] = nucleic["min_distance"]
    except Exception as error:  # pragma: no cover - defensive
        record["error_nucleic"] = str(error)

    conservation = get(config, "structure.conservation_vector")
    if conservation:
        try:
            record["packing_correlation"] = conserved_packing(path, conservation)
        except Exception as error:  # pragma: no cover - defensive
            record["error_packing"] = str(error)

    return record


def metrics_table(records: list[dict]) -> pd.DataFrame:
    """Build a stable-column metrics DataFrame from :func:`compute_metrics` rows."""
    columns = [
        "model", "path", "status", "n_residues",
        "plddt_mean", "plddt_min", "plddt_max", "pae_mean", "sasa_mean",
        "orientation_angle_deg", "n_domains", "catalytic_distance",
        "nucleic_has_chain", "nucleic_contact_fraction", "nucleic_min_distance",
        "packing_correlation",
    ]
    frame = pd.DataFrame(records)
    for column in columns:
        if column not in frame.columns:
            frame[column] = None
    extra = [c for c in frame.columns if c not in columns]
    return frame[columns + extra]
