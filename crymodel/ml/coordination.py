# crymodel/ml/coordination.py
"""Coordination geometry features for ion classification."""
from __future__ import annotations
import numpy as np
from scipy.spatial import cKDTree
from typing import Optional


def coordination_number(
    point: np.ndarray,
    coords: np.ndarray,
    mask: np.ndarray,
    cutoff_A: float = 3.0,
) -> int:
    """Count atoms within cutoff distance."""
    idx = np.where(mask)[0]
    if idx.size == 0:
        return 0
    sub = coords[idx]
    d = np.linalg.norm(sub - point, axis=1)
    return int(np.sum(d <= cutoff_A))


def coordination_geometry(
    point: np.ndarray,
    coords: np.ndarray,
    mask: np.ndarray,
    cutoff_A: float = 3.0,
    k: int = 6,
) -> dict[str, float]:
    """Compute coordination geometry features.
    
    Returns features that help distinguish:
    - Tetrahedral (e.g., Mg2+, Zn2+): 4-coordinate, ~109° angles
    - Octahedral (e.g., Ca2+, Mn2+): 6-coordinate, ~90° angles
    - Linear/bent (e.g., water): 2-coordinate
    
    Args:
        point: Central point
        coords: All atom coordinates
        mask: Boolean mask for atoms to consider
        cutoff_A: Coordination distance cutoff
        k: Maximum neighbors to consider
        
    Returns:
        Dictionary with coordination features
    """
    idx = np.where(mask)[0]
    if idx.size == 0:
        return {
            "coord_number": 0,
            "coord_mean_dist": np.nan,
            "coord_std_dist": np.nan,
            "coord_angle_mean": np.nan,
            "coord_angle_std": np.nan,
            "coord_tetrahedral_score": np.nan,
            "coord_octahedral_score": np.nan,
        }
    
    sub = coords[idx]
    d = np.linalg.norm(sub - point, axis=1)
    within_cutoff = d <= cutoff_A
    
    if within_cutoff.sum() == 0:
        return {
            "coord_number": 0,
            "coord_mean_dist": np.nan,
            "coord_std_dist": np.nan,
            "coord_angle_mean": np.nan,
            "coord_angle_std": np.nan,
            "coord_tetrahedral_score": np.nan,
            "coord_octahedral_score": np.nan,
        }
    
    coord_atoms = sub[within_cutoff]
    coord_dists = d[within_cutoff]
    
    # Sort by distance and take nearest k
    sort_idx = np.argsort(coord_dists)[:min(k, len(coord_dists))]
    coord_atoms = coord_atoms[sort_idx]
    coord_dists = coord_dists[sort_idx]
    
    coord_number = len(coord_atoms)
    
    # Distance statistics
    mean_dist = float(np.mean(coord_dists))
    std_dist = float(np.std(coord_dists)) if len(coord_dists) > 1 else 0.0
    
    # Angle statistics (angles between vectors from center to neighbors)
    if len(coord_atoms) >= 2:
        vecs = coord_atoms - point
        # Normalize
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.where(norms < 1e-6, 1.0, norms)
        vecs_norm = vecs / norms
        
        angles = []
        for i in range(len(vecs_norm)):
            for j in range(i + 1, len(vecs_norm)):
                cos_angle = np.clip(np.dot(vecs_norm[i], vecs_norm[j]), -1.0, 1.0)
                angle = np.degrees(np.arccos(cos_angle))
                angles.append(angle)
        
        angle_mean = float(np.mean(angles)) if angles else np.nan
        angle_std = float(np.std(angles)) if angles and len(angles) > 1 else 0.0
        
        # Geometric scores: how close to ideal geometries
        # Tetrahedral: ~109.5° angles, 4-coordinate
        tet_angles = [109.5] * 6  # 4 choose 2 = 6 angles
        tet_score = 1.0 - min(1.0, abs(angle_mean - 109.5) / 30.0) if not np.isnan(angle_mean) else 0.0
        tet_score = tet_score * (1.0 if coord_number == 4 else 0.5)  # Penalize if not 4-coord
        
        # Octahedral: ~90° angles, 6-coordinate
        oct_score = 1.0 - min(1.0, abs(angle_mean - 90.0) / 30.0) if not np.isnan(angle_mean) else 0.0
        oct_score = oct_score * (1.0 if coord_number == 6 else 0.5)  # Penalize if not 6-coord
    else:
        angle_mean = np.nan
        angle_std = 0.0
        tet_score = 0.0
        oct_score = 0.0
    
    return {
        "coord_number": coord_number,
        "coord_mean_dist": mean_dist,
        "coord_std_dist": std_dist,
        "coord_angle_mean": angle_mean,
        "coord_angle_std": angle_std,
        "coord_tetrahedral_score": float(tet_score),
        "coord_octahedral_score": float(oct_score),
    }

