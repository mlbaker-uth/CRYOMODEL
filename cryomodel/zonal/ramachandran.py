"""Ramachandran classification (general case) for zonal refinement."""
from __future__ import annotations

import math
from typing import Literal, Optional, Tuple

import gemmi
import numpy as np

RamaClass = Literal["favored", "allowed", "outlier"]

# Approximate (phi, psi) elliptical cores for alanine-like residues (degrees).
# Favored: tighter; allowed: expanded. Not identical to MolProbity but captures
# helix / sheet / L-helix lobes for outlier vs allowed/favored decisions.
_FAVORED_ELLIPSES: Tuple[Tuple[float, float, float, float], ...] = (
    (-62.0, -43.0, 30.0, 24.0),  # alpha
    (-118.0, 128.0, 38.0, 45.0),  # beta
    (62.0, 28.0, 28.0, 22.0),  # L-helix
)
_ALLOWED_ELLIPSES: Tuple[Tuple[float, float, float, float], ...] = (
    (-62.0, -43.0, 44.0, 36.0),
    (-118.0, 128.0, 52.0, 58.0),
    (62.0, 28.0, 40.0, 32.0),
    (-90.0, 5.0, 55.0, 55.0),  # bridge / polyproline II neighborhood
)


def _wrap180(deg: float) -> float:
    x = (deg + 180.0) % 360.0 - 180.0
    return x


def _in_ellipse(phi: float, psi: float, cx: float, cy: float, rx: float, ry: float) -> bool:
    dx = (_wrap180(phi - cx)) / max(rx, 1e-6)
    dy = (_wrap180(psi - cy)) / max(ry, 1e-6)
    return dx * dx + dy * dy <= 1.0


def classify_phi_psi_general(phi_deg: float, psi_deg: float) -> RamaClass:
    """Classify (phi, psi) for non-Gly, non-Pro general case."""
    for e in _FAVORED_ELLIPSES:
        if _in_ellipse(phi_deg, psi_deg, e[0], e[1], e[2], e[3]):
            return "favored"
    for e in _ALLOWED_ELLIPSES:
        if _in_ellipse(phi_deg, psi_deg, e[0], e[1], e[2], e[3]):
            return "allowed"
    return "outlier"


def rama_penalty(class_: RamaClass) -> float:
    """Smooth scalar penalty: favored < allowed << outlier."""
    if class_ == "favored":
        return 0.05
    if class_ == "allowed":
        return 1.0
    return 12.0


def phi_psi_deg(
    prev_res: Optional[gemmi.Residue],
    res: gemmi.Residue,
    next_res: Optional[gemmi.Residue],
) -> Optional[Tuple[float, float]]:
    """Return (phi, psi) in degrees, or None if undefined."""
    pp = gemmi.calculate_phi_psi(prev_res, res, next_res)
    if len(pp) < 2:
        return None
    phi_rad, psi_rad = pp[0], pp[1]
    if math.isnan(phi_rad) or math.isnan(psi_rad):
        return None
    return (float(np.rad2deg(phi_rad)), float(np.rad2deg(psi_rad)))


def classify_residue_backbone(
    prev_res: Optional[gemmi.Residue],
    res: gemmi.Residue,
    next_res: Optional[gemmi.Residue],
) -> Optional[RamaClass]:
    """Rama class for a residue, or None if not applicable (terminal, GLY, PRO)."""
    rn = res.name.strip().upper().split()[0][:3]
    if rn in ("GLY", "PRO"):
        return None
    pp = phi_psi_deg(prev_res, res, next_res)
    if pp is None:
        return None
    return classify_phi_psi_general(pp[0], pp[1])
