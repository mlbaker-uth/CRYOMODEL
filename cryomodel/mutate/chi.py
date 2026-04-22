"""χ1 rotamer sampling and scoring helpers."""
from __future__ import annotations

import math
from typing import Callable, Dict, Optional, Tuple

import gemmi
import numpy as np

# Typical χ1 rotamers (degrees) — approximate Dunbrack g-, g+, trans.
CHI1_TRIAL_ANGLES = (-60.0, 60.0, 180.0)

# Relative priors P(rotamer) for the three χ1 bins — rough; sum ≈ 1.
CHI1_PRIOR_TRIPLET: Dict[str, Tuple[float, float, float]] = {
    "VAL": (0.35, 0.35, 0.30),
    "LEU": (0.35, 0.35, 0.30),
    "ILE": (0.35, 0.35, 0.30),
    "PHE": (0.25, 0.25, 0.50),
    "TYR": (0.25, 0.25, 0.50),
    "TRP": (0.25, 0.25, 0.50),
    "HIS": (0.30, 0.30, 0.40),
    "ASP": (0.33, 0.33, 0.34),
    "ASN": (0.33, 0.33, 0.34),
    "GLU": (0.33, 0.33, 0.34),
    "GLN": (0.33, 0.33, 0.34),
    "MET": (0.33, 0.33, 0.34),
    "LYS": (0.33, 0.33, 0.34),
    "ARG": (0.33, 0.33, 0.34),
    "SER": (0.33, 0.33, 0.34),
    "CYS": (0.33, 0.33, 0.34),
    "THR": (0.35, 0.35, 0.30),
}


def chi1_quadruple(resname: str, res: gemmi.Residue) -> Optional[Tuple[str, str, str, str]]:
    """Atom names (N, CA, CB, X) defining χ1, or None if not applicable."""
    r = resname.upper()
    if r in ("GLY", "ALA", "PRO"):
        return None
    try:
        if r in ("VAL", "ILE"):
            res.sole_atom("CG1")
            return ("N", "CA", "CB", "CG1")
        if r == "THR":
            return ("N", "CA", "CB", "OG1")
        if r == "SER":
            return ("N", "CA", "CB", "OG")
        if r == "CYS":
            return ("N", "CA", "CB", "SG")
        res.sole_atom("CG")
        return ("N", "CA", "CB", "CG")
    except Exception:
        return None


def _atom_by_name(res: gemmi.Residue, name: str) -> gemmi.Atom:
    return res.sole_atom(name)


def chi1_dihedral_deg(res: gemmi.Residue, quad: Tuple[str, str, str, str]) -> float:
    pts = [_atom_by_name(res, q).pos for q in quad]
    rad = gemmi.calculate_dihedral(pts[0], pts[1], pts[2], pts[3])
    return float(np.rad2deg(rad))


def _wrap_delta_deg(delta: float) -> float:
    x = (delta + 180.0) % 360.0 - 180.0
    return x


def rotate_sidechain_chi1(
    residue: gemmi.Residue,
    quad: Tuple[str, str, str, str],
    delta_deg: float,
) -> None:
    """Rotate side-chain atoms distal to CB by delta around CA–CB axis (in place)."""
    n_name, ca_name, cb_name, _ = quad
    ca = residue.sole_atom(ca_name)
    cb = residue.sole_atom(cb_name)

    axis_start = np.array([ca.pos.x, ca.pos.y, ca.pos.z], dtype=np.float64)
    axis_end = np.array([cb.pos.x, cb.pos.y, cb.pos.z], dtype=np.float64)
    axis = axis_end - axis_start
    axis = axis / (np.linalg.norm(axis) + 1e-12)

    angle_rad = np.deg2rad(delta_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    K = np.array(
        [
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0],
        ],
        dtype=np.float64,
    )
    R = np.eye(3) + sin_a * K + (1 - cos_a) * (K @ K)
    cb_pos = axis_end.copy()

    for atom in residue:
        if atom.name in (n_name, ca_name, cb_name, "C", "O"):
            continue
        atom_pos = np.array([atom.pos.x, atom.pos.y, atom.pos.z], dtype=np.float64)
        rel = atom_pos - cb_pos
        new_rel = R @ rel + cb_pos
        atom.pos = gemmi.Position(float(new_rel[0]), float(new_rel[1]), float(new_rel[2]))


def map_fit_anchor_term(
    m_trial: float,
    m_anchor: float,
    *,
    weight_anchor: float = 0.0,
    weight_gain: float = 0.0,
    eps: float = 1e-5,
) -> float:
    """
    Penalize losing map fit when the anchor was above threshold-derived signal;
    when the anchor was weak (≤ eps), reward gains only (no penalty for staying weak).

    ``m_*`` are mean sampled map values after ``density_threshold`` (higher = better fit).
    """
    if weight_anchor <= 0.0 and weight_gain <= 0.0:
        return 0.0
    delta = m_trial - m_anchor
    if m_anchor > eps:
        if weight_anchor > 0.0:
            return weight_anchor * max(0.0, -delta)
        return 0.0
    if weight_gain > 0.0:
        return -weight_gain * max(0.0, delta)
    return 0.0


def chi1_prior_penalty(resname: str, trial_index: int) -> float:
    triplet = CHI1_PRIOR_TRIPLET.get(resname.upper(), (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0))
    p = triplet[trial_index]
    p = max(p, 1e-6)
    return float(-math.log(p))


def pick_best_chi1(
    resname: str,
    residue: gemmi.Residue,
    quad: Tuple[str, str, str, str],
    clash_fn: Callable[[gemmi.Residue], float],
    map_score_fn: Callable[[gemmi.Residue], float],
    *,
    weight_rot: float = 0.15,
    weight_map: float = 0.0,
    weight_density_anchor: float = 0.0,
    weight_density_gain: float = 0.0,
    map_anchor_eps: float = 1e-5,
) -> None:
    """
    Try χ1 trials in place on `residue`: minimize
    clash + weight_rot * (-log P) - weight_map * mean(map at side-chain atoms),
    plus optional map-fit anchoring vs the starting pose (see ``map_fit_anchor_term``).
    """
    need_map = (
        weight_map > 0.0
        or weight_density_anchor > 0.0
        or weight_density_gain > 0.0
    )
    m_anchor = float(map_score_fn(residue)) if need_map else 0.0
    best_score = float("inf")
    best_trial: Optional[gemmi.Residue] = None
    for idx, target_chi in enumerate(CHI1_TRIAL_ANGLES):
        trial = residue.clone()
        cur = chi1_dihedral_deg(trial, quad)
        delta = _wrap_delta_deg(target_chi - cur)
        rotate_sidechain_chi1(trial, quad, delta)
        c = clash_fn(trial)
        rot_pen = weight_rot * chi1_prior_penalty(resname, idx)
        ms = map_score_fn(trial) if need_map else 0.0
        fit_term = map_fit_anchor_term(
            ms,
            m_anchor,
            weight_anchor=weight_density_anchor,
            weight_gain=weight_density_gain,
            eps=map_anchor_eps,
        )
        score = c + rot_pen - weight_map * ms + fit_term
        if score < best_score:
            best_score = score
            best_trial = trial
    if best_trial is None:
        return
    name_to_pos = {a.name: a.pos for a in best_trial}
    for atom in residue:
        if atom.name in name_to_pos:
            atom.pos = name_to_pos[atom.name]


def mean_sidechain_map_value(
    residue: gemmi.Residue,
    map_vol,
    *,
    density_threshold: float = 0.0,
) -> float:
    from ..validation.ringer_lite import sample_density_at_position

    vals = []
    for atom in residue:
        if atom.element.name == "H":
            continue
        if atom.name in ("N", "CA", "C", "O"):
            continue
        p = np.array([atom.pos.x, atom.pos.y, atom.pos.z], dtype=np.float64)
        vals.append(sample_density_at_position(map_vol, p, density_threshold=density_threshold))
    return float(sum(vals) / max(len(vals), 1))
