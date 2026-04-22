"""Minimal backbone φ/ψ rotations on a gemmi chain (local zonal refinement)."""
from __future__ import annotations

from typing import List

import gemmi
import numpy as np


def _pos(atom: gemmi.Atom) -> np.ndarray:
    return np.array([atom.pos.x, atom.pos.y, atom.pos.z], dtype=np.float64)


def _set_pos(atom: gemmi.Atom, p: np.ndarray) -> None:
    atom.pos = gemmi.Position(float(p[0]), float(p[1]), float(p[2]))


def _rodrigues_rotate_point(p: np.ndarray, pivot: np.ndarray, axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    v = p - pivot
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    k = np.array(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]],
        dtype=np.float64,
    )
    r = np.eye(3) + sin_a * k + (1.0 - cos_a) * (k @ k)
    return r @ v + pivot


def _h_on_n_atoms(res: gemmi.Residue, n_atom: gemmi.Atom) -> List[gemmi.Atom]:
    """Backbone H atoms attached to N (distance heuristic)."""
    pn = _pos(n_atom)
    out: List[gemmi.Atom] = []
    for atom in res:
        if atom.element.name != "H":
            continue
        if np.linalg.norm(_pos(atom) - pn) < 1.25:
            out.append(atom)
    return out


def rotate_phi(chain: gemmi.Chain, res_index: int, delta_deg: float) -> None:
    """
    Rotate φ(i) by ``delta_deg`` about the N(i)–CA(i) axis.

    Moves the N-terminal fragment (all atoms in residues before ``res_index``,
    plus H atoms on N(i)). N(i) and CA(i) lie on the rotation axis and are fixed.
    """
    if res_index <= 0 or abs(delta_deg) < 1e-12:
        return
    res_i = chain[res_index]
    n_atom = res_i.sole_atom("N")
    ca_atom = res_i.sole_atom("CA")
    axis = _pos(ca_atom) - _pos(n_atom)
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    pivot = _pos(n_atom)
    ang = float(np.deg2rad(delta_deg))

    moving: List[gemmi.Atom] = []
    for ri, res in enumerate(chain):
        if ri < res_index:
            for atom in res:
                moving.append(atom)
        elif ri == res_index:
            moving.extend(_h_on_n_atoms(res, n_atom))

    for atom in moving:
        p = _pos(atom)
        q = _rodrigues_rotate_point(p, pivot, axis, ang)
        _set_pos(atom, q)


def rotate_psi(chain: gemmi.Chain, res_index: int, delta_deg: float) -> None:
    """
    Rotate ψ(i) about the CA(i)–C(i) axis.

    Moves O(i) and all atoms in residues after ``res_index``.
    """
    if res_index >= len(chain) - 1 or abs(delta_deg) < 1e-12:
        return
    res_i = chain[res_index]
    ca_atom = res_i.sole_atom("CA")
    c_atom = res_i.sole_atom("C")
    axis = _pos(c_atom) - _pos(ca_atom)
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    pivot = _pos(ca_atom)
    ang = float(np.deg2rad(delta_deg))

    moving: List[gemmi.Atom] = []
    try:
        moving.append(res_i.sole_atom("O"))
    except Exception:
        pass
    for ri in range(res_index + 1, len(chain)):
        for atom in chain[ri]:
            moving.append(atom)

    for atom in moving:
        p = _pos(atom)
        q = _rodrigues_rotate_point(p, pivot, axis, ang)
        _set_pos(atom, q)
