# crymodel/validation/geometry_priors.py
"""Geometry-vs-resolution priors and features."""
from __future__ import annotations
import numpy as np
from typing import Dict, Optional
import gemmi

# Note: Full MolProbity/CaBLAM integration would require additional dependencies
# This is a simplified version that computes basic geometry metrics


def compute_ramachandran_prob(
    residue: gemmi.Residue,
    prev_residue: Optional[gemmi.Residue] = None,
) -> float:
    """Compute Ramachandran probability (simplified).
    
    Returns probability-like score (0-1) based on phi/psi angles.
    """
    if not prev_residue:
        return 0.5  # Default if no previous residue
    
    # Get atoms
    n_prev = prev_residue.find('N')
    ca_prev = prev_residue.find('CA')
    c_prev = prev_residue.find('C')
    n = residue.find('N')
    ca = residue.find('CA')
    c = residue.find('C')
    
    if not all([n_prev, ca_prev, c_prev, n, ca, c]):
        return 0.5
    
    # Compute phi (C_prev - N - CA - C)
    phi = _dihedral_angle(
        [c_prev.pos, n.pos, ca.pos, c.pos]
    )
    
    # Compute psi (N - CA - C - N_next) - would need next residue
    # For now, use phi only
    phi_deg = np.rad2deg(phi)
    
    # Simplified: check if in favorable regions
    # Core regions: phi ~ -60, psi ~ -45 (helix) or phi ~ -120, psi ~ 120 (sheet)
    if -180 <= phi_deg <= 180:
        # Very simplified scoring
        if -90 <= phi_deg <= -30:  # Helix-like
            return 0.8
        elif -150 <= phi_deg <= -90:  # Sheet-like
            return 0.7
        else:
            return 0.4
    
    return 0.5


def compute_clashscore_z(
    residue: gemmi.Residue,
    all_atoms: list,
    clash_cutoff: float = 2.0,
) -> float:
    """Compute clashscore Z-score (simplified).
    
    Returns Z-score based on number of clashes.
    """
    res_atoms = [a for a in all_atoms if a.resi == residue.seqid.num]
    
    clashes = 0
    for atom1 in res_atoms:
        for atom2 in all_atoms:
            if atom1 == atom2:
                continue
            dist = np.sqrt(
                (atom1.pos.x - atom2.pos.x)**2 +
                (atom1.pos.y - atom2.pos.y)**2 +
                (atom1.pos.z - atom2.pos.z)**2
            )
            # Simplified clash detection
            if dist < clash_cutoff:
                clashes += 1
    
    # Normalize by number of atoms
    if len(res_atoms) > 0:
        clash_density = clashes / len(res_atoms)
        # Convert to Z-score (simplified)
        z_score = (clash_density - 0.1) / 0.05  # Rough normalization
        return float(z_score)
    
    return 0.0


def _dihedral_angle(positions: list) -> float:
    """Compute dihedral angle from 4 positions."""
    p0, p1, p2, p3 = [np.array([p.x, p.y, p.z]) for p in positions]
    
    v1 = p1 - p0
    v2 = p2 - p1
    v3 = p3 - p2
    
    n1 = np.cross(v1, v2)
    n2 = np.cross(v2, v3)
    
    n1 = n1 / (np.linalg.norm(n1) + 1e-12)
    n2 = n2 / (np.linalg.norm(n2) + 1e-12)
    
    angle = np.arccos(np.clip(np.dot(n1, n2), -1.0, 1.0))
    
    # Determine sign
    if np.dot(np.cross(n1, n2), v2) < 0:
        angle = -angle
    
    return angle


def compute_geometry_features(
    residue: gemmi.Residue,
    all_residues: list,
    local_res: Optional[float] = None,
) -> Dict[str, float]:
    """Compute geometry features for a residue.
    
    Args:
        residue: Current residue
        all_residues: List of all residues in structure
        local_res: Optional local resolution (Å)
        
    Returns:
        Dictionary with geometry metrics
    """
    all_atoms = []
    for res in all_residues:
        for atom in res:
            all_atoms.append(atom)
    
    # Find previous residue
    prev_residue = None
    for res in all_residues:
        if res.seqid.num == residue.seqid.num - 1 and res.chain == residue.chain:
            prev_residue = res
            break
    
    ramachandran_prob = compute_ramachandran_prob(residue, prev_residue)
    clashscore_z = compute_clashscore_z(residue, all_atoms)
    
    # Additional geometry features would go here:
    # - Rotamer probability
    # - CaBLAM flags
    # - Peptide planarity
    # - Cβ deviations
    
    return {
        'ramachandran_prob': float(ramachandran_prob),
        'clashscore_z': float(clashscore_z),
        # Placeholders for future features
        'rotamer_prob': 0.5,
        'cablam_flag': 0.0,
        'peptide_planarity_z': 0.0,
        'cb_deviation_z': 0.0,
    }

