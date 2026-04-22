"""Spatial masks for zonal refinement."""
from __future__ import annotations

from typing import List, Optional, Set, Tuple

import gemmi
import numpy as np


def parse_center_xyz(center_s: str) -> np.ndarray:
    """Parse comma-separated ``x,y,z`` in Angstrom."""
    parts = [p.strip() for p in center_s.replace(" ", "").split(",")]
    if len(parts) != 3:
        raise ValueError(f"Expected three comma-separated numbers (x,y,z), got {center_s!r}")
    return np.array([float(parts[0]), float(parts[1]), float(parts[2])], dtype=np.float64)


def residues_in_sphere(
    st: gemmi.Structure,
    center_xyz: np.ndarray,
    radius: float,
    *,
    chain_filter: Optional[Set[str]] = None,
) -> List[Tuple[str, gemmi.Residue]]:
    """
    Residues whose **any** heavy atom lies within ``radius`` Å of ``center_xyz``.

    ``chain_filter`` is an optional set of chain IDs to include; ``None`` means all chains.
    """
    if radius <= 0:
        raise ValueError("radius must be positive")
    center_xyz = np.asarray(center_xyz, dtype=np.float64).reshape(3)
    r2 = radius * radius
    out: List[Tuple[str, gemmi.Residue]] = []
    seen: set[Tuple[str, str]] = set()
    for model in st:
        for chain in model:
            cid = chain.name
            if chain_filter is not None and cid not in chain_filter:
                continue
            for res in chain:
                for atom in res:
                    if atom.element.name == "H":
                        continue
                    dx = atom.pos.x - center_xyz[0]
                    dy = atom.pos.y - center_xyz[1]
                    dz = atom.pos.z - center_xyz[2]
                    if dx * dx + dy * dy + dz * dz <= r2:
                        key = (cid, str(res.seqid))
                        if key not in seen:
                            seen.add(key)
                            out.append((cid, res))
                        break
    return out


def partition_hard_soft_spherical(
    st: gemmi.Structure,
    center_xyz: np.ndarray,
    hard_radius: float,
    soft_buffer: float,
    *,
    chain_filter: Optional[Set[str]] = None,
) -> Tuple[List[Tuple[str, gemmi.Residue]], List[Tuple[str, gemmi.Residue]]]:
    """
    **Hard:** residues with any heavy atom within ``hard_radius`` Å of ``center_xyz``.

    **Soft:** residues in the larger ball of radius ``hard_radius + soft_buffer`` that are
    **not** in the hard set (spherical shell occupancy for whole residues).

    If ``soft_buffer <= 0``, soft list is empty (A0-only behavior).
    """
    hard = residues_in_sphere(st, center_xyz, hard_radius, chain_filter=chain_filter)
    if soft_buffer <= 0:
        return hard, []

    outer_r = hard_radius + float(soft_buffer)
    outer = residues_in_sphere(st, center_xyz, outer_r, chain_filter=chain_filter)
    hard_keys: Set[Tuple[str, str]] = {(c, str(r.seqid)) for c, r in hard}
    soft = [(c, r) for c, r in outer if (c, str(r.seqid)) not in hard_keys]
    return hard, soft
