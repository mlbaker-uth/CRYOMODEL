"""MolProbity-style clashscore (≥0.4 Å overlap, heavy atoms)."""
from __future__ import annotations

import gemmi

from cryomodel.validation.geometry_priors import (
    build_heavy_atom_clash_context,
    molprobity_like_clashscore_heavy,
)


def _two_carbon_residues(distance_A: float) -> gemmi.Structure:
    st = gemmi.Structure()
    st.cell = gemmi.UnitCell(100, 100, 100, 90, 90, 90)
    st.add_model(gemmi.Model("1"))
    ch = gemmi.Chain("A")
    for i, x in enumerate([0.0, distance_A]):
        r = gemmi.Residue()
        r.seqid = gemmi.SeqId(str(i + 1))
        r.name = "ALA"
        a = gemmi.Atom()
        a.name = "CB"
        a.element = gemmi.Element("C")
        a.pos = gemmi.Position(x, 0.0, 0.0)
        r.add_atom(a)
        ch.add_residue(r)
    st[0].add_chain(ch)
    return st


def test_molprobity_clashscore_severe_overlap() -> None:
    st = _two_carbon_residues(2.3)
    score, n_pairs, n_h = molprobity_like_clashscore_heavy(st)
    assert n_h == 2
    assert n_pairs == 1
    assert score == 500.0


def test_molprobity_clashscore_no_overlap() -> None:
    st = _two_carbon_residues(4.0)
    score, n_pairs, n_h = molprobity_like_clashscore_heavy(st)
    assert n_h == 2
    assert n_pairs == 0
    assert score == 0.0


def test_clash_context_fallback_distance_for_minimal_cb_residues() -> None:
    """Gemmi topology yields no bonds when standard atoms are missing; use distance graph."""
    st = _two_carbon_residues(4.0)
    ctx = build_heavy_atom_clash_context(st)
    assert ctx.bond_topology == "distance"
