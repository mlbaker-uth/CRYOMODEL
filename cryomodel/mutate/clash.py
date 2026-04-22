"""Simple van der Waals clash score vs the rest of the structure."""
from __future__ import annotations

from typing import List, Tuple

import gemmi
import numpy as np


def clash_score_for_residue(
    st: gemmi.Structure,
    chain_id: str,
    residue: gemmi.Residue,
    *,
    cutoff_close: float = 2.4,
    cutoff_soft: float = 3.5,
) -> float:
    """
    Score side-chain heavy atoms vs all other atoms, including this residue's backbone.
    """
    points: List[np.ndarray] = []
    for model in st:
        for chain in model:
            for res in chain:
                for atom in res:
                    if atom.element.name == "H":
                        continue
                    same = chain.name == chain_id and res.seqid == residue.seqid
                    if same and atom.name not in ("N", "CA", "C", "O"):
                        # Skip current residue side-chain positions from the environment
                        # (they are the ones being scored from `residue` argument).
                        continue
                    points.append(np.array([atom.pos.x, atom.pos.y, atom.pos.z], dtype=np.float64))
    if not points:
        return 0.0
    tree = None
    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(np.vstack(points))
    except Exception:
        tree = None

    score = 0.0
    for atom in residue:
        if atom.element.name == "H":
            continue
        if atom.name in ("N", "CA", "C", "O"):
            continue
        p = np.array([atom.pos.x, atom.pos.y, atom.pos.z], dtype=np.float64)
        if tree is not None:
            dist, _ = tree.query(p, k=1)
            d = float(dist)
        else:
            d = min(np.linalg.norm(p - q) for q in points)
        if d >= cutoff_soft:
            continue
        if d < cutoff_close:
            score += (cutoff_close - d) ** 2 * 10.0
        else:
            score += (cutoff_soft - d) ** 2
    return float(score)


def self_clash_backbone_sidechain(residue: gemmi.Residue) -> float:
    """Light penalty for side-chain atoms clashing with backbone of same residue."""
    bb: List[np.ndarray] = []
    sc: List[Tuple[str, np.ndarray]] = []
    for atom in residue:
        if atom.element.name == "H":
            continue
        p = np.array([atom.pos.x, atom.pos.y, atom.pos.z], dtype=np.float64)
        if atom.name in ("N", "CA", "C", "O"):
            bb.append(p)
        else:
            sc.append((atom.name, p))
    if not bb or not sc:
        return 0.0
    s = 0.0
    for _, p in sc:
        for q in bb:
            d = float(np.linalg.norm(p - q))
            if d < 2.0:
                s += (2.0 - d) ** 2
    return s
