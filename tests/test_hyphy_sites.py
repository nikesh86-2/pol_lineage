"""Tests for HyPhy site mapping and taxa/tree matching."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hbvpol.io import GenomeRecord  # noqa: E402
from hbvpol.selection.pipeline import _map_hyphy_sites, _restrict_and_prune  # noqa: E402

_SITE = {"method": "fel", "statistic": 0.5, "pvalue": 0.01, "significant": True}


def _row(site: int) -> dict:
    return {"pol_position": site, **_SITE}


def test_map_hyphy_sites_reference_frame_and_drops_unmapped():
    # 63-column gapped MSA: reference positions 1..20, a 3-nt insertion, 21..60.
    positions = list(range(1, 21)) + [None, None, None] + list(range(21, 61))
    config = {"reference": {"pol_start_nt": 1, "pol_end_nt": 60}}
    rows = [_row(45), _row(21), _row(63), _row(64)]

    mapped = _map_hyphy_sites(rows, positions, 60, 63, config)
    by_site = {row["pol_position"] for row in mapped}
    # site 45 -> column 44 -> reference 42 -> codon 14
    # site 63 -> column 62 -> reference 60 -> codon 20
    assert by_site == {14, 20}


def test_map_hyphy_sites_detects_codon_scale():
    positions = list(range(1, 61))
    config = {"reference": {"pol_start_nt": 1, "pol_end_nt": 60}}
    # 20 reported sites over a 60-column alignment -> HyPhy reported codons.
    rows = [_row(index) for index in range(1, 21)]

    mapped = _map_hyphy_sites(rows, positions, 60, 60, config)
    by_site = {row["pol_position"] for row in mapped}
    assert by_site == set(range(1, 21))


def test_map_hyphy_sites_no_map_is_passthrough():
    rows = [_row(7)]
    assert _map_hyphy_sites(rows, None, None, 60, {}) == rows


def test_restrict_and_prune_matches_taxa(tmp_path):
    tree = tmp_path / "tree.nwk"
    tree.write_text("(A:0.1,B:0.1,(C:0.1,D:0.1):0.1);\n", encoding="utf-8")
    records = [GenomeRecord(id=name, seq="ACGT") for name in "ABCDE"]

    keep, pruned = _restrict_and_prune(records, tree, tmp_path, "fel")
    assert [record.id for record in keep] == ["A", "B", "C", "D"]
    assert pruned is not None and Path(pruned).exists()

    from hbvpol.phylogeny.trees import newick_leaf_names

    assert set(newick_leaf_names(Path(pruned).read_text(encoding="utf-8"))) == {"A", "B", "C", "D"}


def test_restrict_and_prune_disjoint_returns_empty(tmp_path):
    tree = tmp_path / "tree.nwk"
    tree.write_text("(X:0.1,Y:0.1);\n", encoding="utf-8")
    records = [GenomeRecord(id=name, seq="ACGT") for name in "AB"]
    keep, _ = _restrict_and_prune(records, tree, tmp_path, "fel")
    assert keep == []
