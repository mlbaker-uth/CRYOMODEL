# crymodel/loops/loopcloud.py
"""Generate clash-free loop completions in weak density."""
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import gemmi

from ..io.mrc import MapVolume
from ..validation.local_cc import compute_local_cc_variants
from ..validation.q_lite import q_score_atom


# Ramachandran preferences (simplified)
RAMACHANDRAN_PRIOR = {
    'helix': {'phi': -60, 'psi': -45, 'phi_std': 20, 'psi_std': 20},
    'sheet': {'phi': -120, 'psi': 120, 'phi_std': 30, 'psi_std': 30},
    'loop': {'phi': -90, 'psi': 0, 'phi_std': 60, 'psi_std': 60},
}


def parse_anchor_spec(anchor_str: str) -> Tuple[str, int, str, int]:
    """Parse anchor specification.
    
    Format: "chainA:res123 -> chainA:res140"
    Returns: (start_chain, start_res, end_chain, end_res)
    """
    if '->' in anchor_str:
        start, end = anchor_str.split('->')
        start = start.strip()
        end = end.strip()
    else:
        raise ValueError(f"Invalid anchor format: {anchor_str}")
    
    start_chain, start_res = start.split(':')
    end_chain, end_res = end.split(':')
    
    return start_chain.strip(), int(start_res.strip()), end_chain.strip(), int(end_res.strip())


def get_flanking_residues(
    structure: gemmi.Structure,
    start_chain: str,
    start_res: int,
    end_chain: str,
    end_res: int,
) -> Tuple[Optional[gemmi.Residue], Optional[gemmi.Residue]]:
    """Get flanking residues for loop anchors."""
    start_residue = None
    end_residue = None
    
    for model in structure:
        for chain in model:
            if chain.name == start_chain:
                for residue in chain:
                    if residue.seqid.num == start_res:
                        start_residue = residue
            if chain.name == end_chain:
                for residue in chain:
                    if residue.seqid.num == end_res:
                        end_residue = residue
    
    return start_residue, end_residue


def compute_backbone_frame(residue: gemmi.Residue) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute backbone frame from residue."""
    n = residue.find('N')
    ca = residue.find('CA')
    c = residue.find('C')
    
    if not all([n, ca, c]):
        return None, None, None
    
    n_pos = np.array([n.pos.x, n.pos.y, n.pos.z])
    ca_pos = np.array([ca.pos.x, ca.pos.y, ca.pos.z])
    c_pos = np.array([c.pos.x, c.pos.y, c.pos.z])
    
    # Origin at CA
    origin = ca_pos
    
    # X-axis: CA -> C
    x_axis = c_pos - ca_pos
    x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-12)
    
    # Y-axis: perpendicular to plane (N-CA-C)
    v1 = n_pos - ca_pos
    v2 = c_pos - ca_pos
    y_axis = np.cross(v1, v2)
    y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-12)
    
    # Z-axis: X x Y
    z_axis = np.cross(x_axis, y_axis)
    z_axis = z_axis / (np.linalg.norm(z_axis) + 1e-12)
    
    return origin, x_axis, z_axis


def sample_phi_psi(ss_type: str = 'loop') -> Tuple[float, float]:
    """Sample phi/psi angles from Ramachandran prior."""
    prior = RAMACHANDRAN_PRIOR.get(ss_type, RAMACHANDRAN_PRIOR['loop'])
    phi = np.random.normal(prior['phi'], prior['phi_std'])
    psi = np.random.normal(prior['psi'], prior['psi_std'])
    return np.deg2rad(phi), np.deg2rad(psi)


def build_loop_backbone(
    start_residue: gemmi.Residue,
    end_residue: gemmi.Residue,
    num_residues: int,
    ss_type: str = 'loop',
) -> List[np.ndarray]:
    """Build loop backbone using kinematic closure (simplified).
    
    Returns list of Cα positions.
    """
    # Get anchor positions
    ca_start = start_residue.find('CA')
    ca_end = end_residue.find('CA')
    
    if not ca_start or not ca_end:
        return []
    
    start_pos = np.array([ca_start.pos.x, ca_start.pos.y, ca_start.pos.z])
    end_pos = np.array([ca_end.pos.x, ca_end.pos.y, ca_end.pos.z])
    
    # Generate Cα positions along path
    ca_positions = []
    
    # Simple interpolation with random perturbations
    for i in range(num_residues + 2):  # +2 for start and end
        t = i / (num_residues + 1)
        pos = start_pos + t * (end_pos - start_pos)
        
        # Add random perturbation
        if 0 < t < 1:
            phi, psi = sample_phi_psi(ss_type)
            # Simplified: add small random offset
            offset = np.random.normal(0, 0.5, 3)
            pos += offset
        
        ca_positions.append(pos)
    
    return ca_positions


def check_clashes(
    loop_ca_positions: List[np.ndarray],
    structure: gemmi.Structure,
    clash_cutoff: float = 2.4,
) -> int:
    """Check for clashes with fixed model."""
    clash_count = 0
    
    # Get all atoms from fixed model
    fixed_atoms = []
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    if atom.element.name != 'H':
                        fixed_atoms.append([atom.pos.x, atom.pos.y, atom.pos.z])
    
    if len(fixed_atoms) == 0:
        return 0
    
    fixed_atoms = np.array(fixed_atoms)
    
    # Check each Cα against fixed atoms
    for ca_pos in loop_ca_positions:
        distances = np.linalg.norm(fixed_atoms - ca_pos, axis=1)
        clashes = (distances < clash_cutoff).sum()
        clash_count += clashes
    
    return clash_count


def score_loop_density(
    loop_ca_positions: List[np.ndarray],
    map_vol: MapVolume,
    half1_vol: Optional[MapVolume] = None,
    half2_vol: Optional[MapVolume] = None,
) -> Dict[str, float]:
    """Score loop against density map."""
    if len(loop_ca_positions) == 0:
        return {'cc_mask': 0.0, 'q_mean': 0.0, 'score': 0.0}
    
    ca_positions = np.array(loop_ca_positions)
    
    # CC_mask
    cc_features = compute_local_cc_variants(ca_positions, map_vol, half1_vol, half2_vol)
    
    # Q-scores (simplified - use first Cα)
    q_scores = []
    for ca_pos in loop_ca_positions:
        q_feat = q_score_atom(ca_pos, map_vol, 3.0, half1_vol, half2_vol)
        q_scores.append(q_feat['Q'])
    
    q_mean = np.mean(q_scores) if q_scores else 0.0
    
    # Combined score
    score = 0.5 * cc_features['CC_mask'] + 0.3 * q_mean - 0.2 * cc_features.get('CC_delta', 0.0)
    
    return {
        'cc_mask': float(cc_features['CC_mask']),
        'q_mean': float(q_mean),
        'cc_delta': float(cc_features.get('CC_delta', 0.0)),
        'score': float(score),
    }


def generate_loop_candidates(
    structure: gemmi.Structure,
    start_chain: str,
    start_res: int,
    end_chain: str,
    end_res: int,
    sequence: str,
    num_candidates: int = 50,
    ss_type: str = 'loop',
) -> List[List[np.ndarray]]:
    """Generate loop candidates."""
    start_residue, end_residue = get_flanking_residues(
        structure, start_chain, start_res, end_chain, end_res
    )
    
    if not start_residue or not end_residue:
        return []
    
    num_residues = len(sequence)
    candidates = []
    
    for _ in range(num_candidates):
        ca_positions = build_loop_backbone(start_residue, end_residue, num_residues, ss_type)
        if len(ca_positions) > 0:
            candidates.append(ca_positions)
    
    return candidates


def score_loops(
    candidates: List[List[np.ndarray]],
    structure: gemmi.Structure,
    map_vol: Optional[MapVolume] = None,
    half1_vol: Optional[MapVolume] = None,
    half2_vol: Optional[MapVolume] = None,
) -> pd.DataFrame:
    """Score all loop candidates."""
    rows = []
    
    for i, ca_positions in enumerate(candidates):
        # Clash check
        clash_count = check_clashes(ca_positions, structure)
        
        # Density score (if map provided)
        density_score = {'cc_mask': 0.0, 'q_mean': 0.0, 'score': 0.0}
        if map_vol:
            density_score = score_loop_density(ca_positions, map_vol, half1_vol, half2_vol)
        
        # Combined score
        total_score = density_score['score'] - 0.1 * clash_count
        
        rows.append({
            'candidate_id': i,
            'clash_count': clash_count,
            'cc_mask': density_score['cc_mask'],
            'q_mean': density_score['q_mean'],
            'density_score': density_score['score'],
            'total_score': total_score,
        })
    
    df = pd.DataFrame(rows)
    df = df.sort_values('total_score', ascending=False)
    return df

