"""Lightweight guide metrics: clash and map coverage at side-chain atoms."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..io.mrc import MapVolume
from ..validation.ringer_lite import sample_density_at_position
from .clash import clash_score_for_residue, self_clash_backbone_sidechain


def map_volume_mean_std(map_vol: MapVolume) -> Tuple[float, float]:
    """Global mean and std of map voxel values (one-time per run)."""
    arr = np.asarray(map_vol.data_zyx, dtype=np.float64).ravel()
    if arr.size == 0:
        return 0.0, 1.0
    mu = float(arr.mean())
    sig = float(arr.std())
    if sig < 1e-12:
        sig = 1.0
    return mu, sig


def sidechain_atom_densities(residue, map_vol: MapVolume) -> List[float]:
    """Trilinear density at each side-chain heavy atom (excludes backbone)."""
    vals: List[float] = []
    for atom in residue:
        if atom.element.name == "H":
            continue
        if atom.name in ("N", "CA", "C", "O"):
            continue
        p = np.array([atom.pos.x, atom.pos.y, atom.pos.z], dtype=np.float64)
        vals.append(sample_density_at_position(map_vol, p))
    return vals


def density_sidechain_stats(
    residue,
    map_vol: MapVolume,
    map_mu: float,
    map_sig: float,
    sigma_mult: float,
) -> Tuple[Optional[float], Optional[float]]:
    """
    Mean density at side-chain atoms, and fraction of those atoms with
    density > map_mu + sigma_mult * map_sig.
    """
    vals = sidechain_atom_densities(residue, map_vol)
    if not vals:
        return None, None
    mean_d = float(sum(vals) / len(vals))
    thr = map_mu + float(sigma_mult) * map_sig
    frac = float(sum(1 for v in vals if v > thr) / len(vals))
    return mean_d, frac


def total_clash_for_residue(st, chain_id: str, residue) -> float:
    """Same combined clash as used during rotamer scoring."""
    return clash_score_for_residue(st, chain_id, residue) + self_clash_backbone_sidechain(residue)


def guide_for_residue(
    st,
    chain_id: str,
    residue,
    map_vol: Optional[MapVolume],
    map_mu: float,
    map_sig: float,
    sigma_mult: float,
) -> Dict[str, Any]:
    """Clash + optional density guide for one residue snapshot."""
    out: Dict[str, Any] = {
        "clash": float(total_clash_for_residue(st, chain_id, residue)),
    }
    if map_vol is None:
        out["sidechain_density_mean"] = None
        out["frac_atoms_above_global_mu_plus_sigma"] = None
        return out
    mean_d, frac = density_sidechain_stats(residue, map_vol, map_mu, map_sig, sigma_mult)
    out["sidechain_density_mean"] = mean_d
    out["frac_atoms_above_global_mu_plus_sigma"] = frac
    return out


def delta_guide(
    before: Dict[str, Any],
    after: Dict[str, Any],
    *,
    map_mu: float,
    map_sig: float,
    sigma_mult: float,
) -> Dict[str, Any]:
    """Deltas (after − before); positive delta_clash = more clash after mutation."""
    out: Dict[str, Any] = {
        "delta_clash": float(after["clash"]) - float(before["clash"]),
    }
    mb, ma = before.get("sidechain_density_mean"), after.get("sidechain_density_mean")
    fb, fa = before.get("frac_atoms_above_global_mu_plus_sigma"), after.get(
        "frac_atoms_above_global_mu_plus_sigma"
    )
    if mb is not None and ma is not None:
        out["delta_sidechain_density_mean"] = float(ma - mb)
    else:
        out["delta_sidechain_density_mean"] = None
    if fb is not None and fa is not None:
        out["delta_frac_atoms_above_threshold"] = float(fa - fb)
    else:
        out["delta_frac_atoms_above_threshold"] = None
    # Threshold uses whole-map μ, σ and k (see map_guide_reference on MutateResult).
    out["threshold_density"] = float(map_mu + float(sigma_mult) * map_sig)
    return out
