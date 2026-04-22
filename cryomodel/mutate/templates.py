"""Ideal residue templates from OpenMM PDBFixer (data/templates/*.pdb; Apache-2.0)."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import gemmi

_TEMPLATE_DIR = Path(__file__).resolve().parent / "data" / "templates"


@lru_cache(maxsize=32)
def load_template_residue(resname: str) -> gemmi.Residue:
    """Load a single-residue template (heavy atoms, first conformer)."""
    p = _TEMPLATE_DIR / f"{resname.upper()}.pdb"
    if not p.exists():
        raise FileNotFoundError(f"No template for residue {resname!r}: {p}")
    st = gemmi.read_structure(str(p))
    return st[0][0][0].clone()


def superpose_and_copy_template(
    template_name: str,
    target_residue: gemmi.Residue,
) -> gemmi.Residue:
    """
    Superpose template N–Cα–C onto target backbone; return cloned residue with
    transformed coordinates (same seqid/name as caller should set afterward).
    """
    tpl = load_template_residue(template_name)
    tr = _superpose_transform(tpl, target_residue)
    out = tpl.clone()
    out.name = template_name.upper()
    for atom in out:
        atom.pos = gemmi.Position(tr.apply(atom.pos))
    return out


def _superpose_transform(tpl_res: gemmi.Residue, target_res: gemmi.Residue) -> gemmi.Transform:
    def ncac(r: gemmi.Residue):
        return [r.sole_atom("N").pos, r.get_ca().pos, r.sole_atom("C").pos]

    fixed = ncac(target_res)
    mov = ncac(tpl_res)
    sup = gemmi.superpose_positions(fixed, mov)
    return sup.transform
