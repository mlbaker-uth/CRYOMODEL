# crymodel/validation/ringer_lite.py
"""Ringer-Lite: χ1-scan side-chain density score."""
from __future__ import annotations
import numpy as np
from typing import Dict, List, Optional, Tuple
import gemmi

from ..io.mrc import MapVolume


# Canonical rotamer angles (degrees) for common amino acids
ROTAMER_ANGLES = {
    'ALA': [],  # No χ1
    'GLY': [],  # No χ1
    'PRO': [],  # Special case
    'SER': [-60, 60, 180],  # gauche-, gauche+, trans
    'CYS': [-60, 60, 180],
    'THR': [-60, 60, 180],
    'VAL': [-60, 60, 180],
    'ILE': [-60, 60, 180],
    'LEU': [-60, 60, 180],
    'ASP': [-60, 60, 180],
    'ASN': [-60, 60, 180],
    'PHE': [-60, 60, 180],
    'TYR': [-60, 60, 180],
    'TRP': [-60, 60, 180],
    'HIS': [-60, 60, 180],
    'GLU': [-60, 60, 180],
    'GLN': [-60, 60, 180],
    'MET': [-60, 60, 180],
    'LYS': [-60, 60, 180],
    'ARG': [-60, 60, 180],
}


def get_chi1_atoms(residue: gemmi.Residue) -> Optional[Tuple[str, str, str, str]]:
    """Get atom names for χ1 torsion angle.
    
    Returns (N, CA, CB, CG) or equivalent for χ1 calculation.
    """
    resname = residue.name.upper()
    
    # Standard χ1: N-CA-CB-CG (or equivalent)
    if resname in ['SER', 'CYS', 'THR', 'VAL']:
        # CB-CG is first side-chain bond
        if residue.find('N') and residue.find('CA') and residue.find('CB'):
            cb = residue.find('CB')
            # Find first heavy atom after CB
            for atom in residue:
                if atom.name not in ['N', 'CA', 'CB', 'C', 'O']:
                    if atom.name.startswith('CG') or atom.name.startswith('OG') or atom.name.startswith('SG'):
                        return ('N', 'CA', 'CB', atom.name)
    elif resname in ['ILE', 'LEU', 'ASP', 'ASN', 'PHE', 'TYR', 'TRP', 'HIS', 'GLU', 'GLN', 'MET', 'LYS', 'ARG']:
        if residue.find('N') and residue.find('CA') and residue.find('CB') and residue.find('CG'):
            return ('N', 'CA', 'CB', 'CG')
    
    return None


def rotate_sidechain_around_chi1(
    residue: gemmi.Residue,
    chi1_atoms: Tuple[str, str, str, str],
    angle_deg: float,
) -> Dict[str, np.ndarray]:
    """Rotate side-chain atoms around χ1 axis.
    
    Returns dictionary of {atom_name: new_position} for atoms distal to χ1.
    """
    n_name, ca_name, cb_name, cg_name = chi1_atoms
    
    n = residue.find(n_name)
    ca = residue.find(ca_name)
    cb = residue.find(cb_name)
    cg = residue.find(cg_name)
    
    if not all([n, ca, cb, cg]):
        return {}
    
    # Get rotation axis (CA-CB vector)
    axis_start = np.array([ca.pos.x, ca.pos.y, ca.pos.z])
    axis_end = np.array([cb.pos.x, cb.pos.y, cb.pos.z])
    axis = axis_end - axis_start
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    
    # Rotation matrix
    angle_rad = np.deg2rad(angle_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    
    # Rodrigues' rotation formula
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    R = np.eye(3) + sin_a * K + (1 - cos_a) * np.dot(K, K)
    
    # Get atoms to rotate (distal to CB)
    rotated_positions = {}
    cb_pos = np.array([cb.pos.x, cb.pos.y, cb.pos.z])
    
    for atom in residue:
        if atom.name in [n_name, ca_name, cb_name, 'C', 'O']:
            continue
        atom_pos = np.array([atom.pos.x, atom.pos.y, atom.pos.z])
        # Translate to origin (CB)
        rel_pos = atom_pos - cb_pos
        # Rotate
        rotated_rel = np.dot(R, rel_pos)
        # Translate back
        new_pos = rotated_rel + cb_pos
        rotated_positions[atom.name] = new_pos
    
    return rotated_positions


def sample_density_at_position(
    map_vol: MapVolume,
    position: np.ndarray,
) -> float:
    """Sample density at a position using trilinear interpolation."""
    origin = map_vol.origin_xyzA
    apix = map_vol.apix
    data = map_vol.data_zyx
    
    # Convert to voxel coordinates
    vox = (position - origin) / apix
    # Map: (x, y, z) -> (z, y, x) for data array
    z, y, x = vox[2], vox[1], vox[0]
    
    # Trilinear interpolation
    z0, y0, x0 = int(np.floor(z)), int(np.floor(y)), int(np.floor(x))
    z1, y1, x1 = z0 + 1, y0 + 1, x0 + 1
    
    # Clamp to bounds
    z0 = max(0, min(z0, data.shape[0] - 1))
    z1 = max(0, min(z1, data.shape[0] - 1))
    y0 = max(0, min(y0, data.shape[1] - 1))
    y1 = max(0, min(y1, data.shape[1] - 1))
    x0 = max(0, min(x0, data.shape[2] - 1))
    x1 = max(0, min(x1, data.shape[2] - 1))
    
    # Fractional parts
    dz, dy, dx = z - z0, y - y0, x - x0
    
    # Interpolate
    c000 = data[z0, y0, x0]
    c001 = data[z0, y0, x1]
    c010 = data[z0, y1, x0]
    c011 = data[z0, y1, x1]
    c100 = data[z1, y0, x0]
    c101 = data[z1, y0, x1]
    c110 = data[z1, y1, x0]
    c111 = data[z1, y1, x1]
    
    c00 = c000 * (1 - dx) + c001 * dx
    c01 = c010 * (1 - dx) + c011 * dx
    c10 = c100 * (1 - dx) + c101 * dx
    c11 = c110 * (1 - dx) + c111 * dx
    
    c0 = c00 * (1 - dy) + c01 * dy
    c1 = c10 * (1 - dy) + c11 * dy
    
    value = c0 * (1 - dz) + c1 * dz
    return float(value)


def ringer_scan_residue(
    residue: gemmi.Residue,
    map_vol: MapVolume,
    half1_vol: Optional[MapVolume] = None,
    half2_vol: Optional[MapVolume] = None,
    local_res: Optional[float] = None,
    step_deg: float = 10.0,
) -> Dict[str, float]:
    """Perform χ1-scan for a residue and compute Ringer-Lite metrics.
    
    Returns:
        Dictionary with keys:
        - ringer_Z: Peak Z-score
        - ringer_peak_deg: Peak angle (degrees)
        - ringer_to_rotamer_deg: Distance to nearest canonical rotamer
        - ringer_half_drop: Half-map stability (Δpeak)
    """
    resname = residue.name.upper()
    
    # Check if residue has χ1
    chi1_atoms = get_chi1_atoms(residue)
    if chi1_atoms is None:
        return {
            'ringer_Z': 0.0,
            'ringer_peak_deg': 0.0,
            'ringer_to_rotamer_deg': 0.0,
            'ringer_half_drop': 0.0,
        }
    
    # Get target atom (Cγ or equivalent)
    n_name, ca_name, cb_name, cg_name = chi1_atoms
    cg_atom = residue.find(cg_name)
    if not cg_atom:
        return {
            'ringer_Z': 0.0,
            'ringer_peak_deg': 0.0,
            'ringer_to_rotamer_deg': 0.0,
            'ringer_half_drop': 0.0,
        }
    
    # Scan angles
    angles = np.arange(-180, 180, step_deg)
    densities = []
    densities_half1 = []
    densities_half2 = []
    
    for angle in angles:
        # Rotate side-chain
        rotated = rotate_sidechain_around_chi1(residue, chi1_atoms, angle)
        if cg_name not in rotated:
            densities.append(0.0)
            if half1_vol:
                densities_half1.append(0.0)
            if half2_vol:
                densities_half2.append(0.0)
            continue
        
        # Sample density at rotated Cγ position
        cg_pos = rotated[cg_name]
        density = sample_density_at_position(map_vol, cg_pos)
        densities.append(density)
        
        if half1_vol:
            density_h1 = sample_density_at_position(half1_vol, cg_pos)
            densities_half1.append(density_h1)
        if half2_vol:
            density_h2 = sample_density_at_position(half2_vol, cg_pos)
            densities_half2.append(density_h2)
    
    densities = np.array(densities)
    
    # Compute peak Z-score
    if densities.std() > 1e-6:
        peak_idx = np.argmax(densities)
        peak_density = densities[peak_idx]
        mean_density = densities.mean()
        std_density = densities.std()
        ringer_Z = (peak_density - mean_density) / std_density
        ringer_peak_deg = float(angles[peak_idx])
    else:
        ringer_Z = 0.0
        ringer_peak_deg = 0.0
    
    # Distance to nearest canonical rotamer
    canonical_angles = ROTAMER_ANGLES.get(resname, [-60, 60, 180])
    if canonical_angles:
        min_dist = min([abs(ringer_peak_deg - ca) for ca in canonical_angles])
        # Handle periodicity
        min_dist = min(min_dist, 360 - min_dist)
        ringer_to_rotamer_deg = float(min_dist)
    else:
        ringer_to_rotamer_deg = 0.0
    
    # Half-map stability
    ringer_half_drop = 0.0
    if half1_vol and half2_vol and len(densities_half1) > 0 and len(densities_half2) > 0:
        peak_h1_idx = np.argmax(densities_half1)
        peak_h2_idx = np.argmax(densities_half2)
        peak_h1_deg = float(angles[peak_h1_idx])
        peak_h2_deg = float(angles[peak_h2_idx])
        # Difference in peak angles
        ringer_half_drop = abs(peak_h1_deg - peak_h2_deg)
        # Handle periodicity
        ringer_half_drop = min(ringer_half_drop, 360 - ringer_half_drop)
    
    # Apply local resolution penalty if provided
    if local_res and local_res > 3.8:
        # Downweight for low resolution
        penalty = max(0.0, 1.0 - (local_res - 3.8) / 2.0)
        ringer_Z *= penalty
    
    return {
        'ringer_Z': float(ringer_Z),
        'ringer_peak_deg': float(ringer_peak_deg),
        'ringer_to_rotamer_deg': float(ringer_to_rotamer_deg),
        'ringer_half_drop': float(ringer_half_drop),
    }

