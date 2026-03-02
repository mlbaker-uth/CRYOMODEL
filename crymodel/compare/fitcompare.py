# crymodel/compare/fitcompare.py
"""Align and compare models across states."""
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import gemmi


def parse_anchor_selection(selection_str: str) -> List[Tuple[str, int, int]]:
    """Parse anchor selection string.
    
    Format: "A:100-160,B:20-45" or "A:100-160"
    Returns list of (chain, start_res, end_res) tuples.
    """
    anchors = []
    for part in selection_str.split(','):
        part = part.strip()
        if ':' in part:
            chain, range_str = part.split(':', 1)
            chain = chain.strip()
            if '-' in range_str:
                start, end = range_str.split('-')
                anchors.append((chain, int(start.strip()), int(end.strip())))
            else:
                resnum = int(range_str.strip())
                anchors.append((chain, resnum, resnum))
    return anchors


def get_ca_positions(
    structure: gemmi.Structure,
    anchors: Optional[List[Tuple[str, int, int]]] = None,
) -> np.ndarray:
    """Get Cα positions from structure, optionally filtered by anchors."""
    positions = []
    
    for model in structure:
        for chain in model:
            for residue in chain:
                if anchors:
                    # Check if residue is in anchor selection
                    in_anchor = False
                    for chain_id, start_res, end_res in anchors:
                        if chain.name == chain_id and start_res <= residue.seqid.num <= end_res:
                            in_anchor = True
                            break
                    if not in_anchor:
                        continue
                
                ca = residue.find('CA')
                if ca:
                    positions.append([ca.pos.x, ca.pos.y, ca.pos.z])
    
    return np.array(positions) if positions else np.zeros((0, 3))


def kabsch_align(
    P: np.ndarray,
    Q: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Kabsch algorithm for optimal rotation.
    
    Args:
        P: Reference points (N, 3)
        Q: Points to align (N, 3)
        
    Returns:
        (rotation_matrix, translation_vector)
    """
    # Center both sets
    P_centroid = P.mean(axis=0)
    Q_centroid = Q.mean(axis=0)
    
    P_centered = P - P_centroid
    Q_centered = Q - Q_centroid
    
    # Compute covariance matrix
    H = P_centered.T @ Q_centered
    
    # SVD
    U, S, Vt = np.linalg.svd(H)
    
    # Rotation matrix
    R = Vt.T @ U.T
    
    # Ensure proper rotation (det(R) = 1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # Translation
    t = P_centroid - R @ Q_centroid
    
    return R, t


def align_models(
    structure_A: gemmi.Structure,
    structure_B: gemmi.Structure,
    anchors: Optional[List[Tuple[str, int, int]]] = None,
) -> Tuple[gemmi.Structure, np.ndarray, np.ndarray]:
    """Align structure_B to structure_A.
    
    Returns:
        (aligned_structure_B, rotation_matrix, translation_vector)
    """
    # Get Cα positions
    ca_A = get_ca_positions(structure_A, anchors)
    ca_B = get_ca_positions(structure_B, anchors)
    
    if len(ca_A) == 0 or len(ca_B) == 0:
        raise ValueError("No matching Cα atoms found")
    
    if len(ca_A) != len(ca_B):
        # Use minimum length
        min_len = min(len(ca_A), len(ca_B))
        ca_A = ca_A[:min_len]
        ca_B = ca_B[:min_len]
    
    # Compute alignment
    R, t = kabsch_align(ca_A, ca_B)
    
    # Apply transformation to structure_B
    aligned_B = gemmi.Structure(structure_B)
    
    for model in aligned_B:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    pos = np.array([atom.pos.x, atom.pos.y, atom.pos.z])
                    pos_aligned = R @ pos + t
                    atom.pos.x = pos_aligned[0]
                    atom.pos.y = pos_aligned[1]
                    atom.pos.z = pos_aligned[2]
    
    return aligned_B, R, t


def compute_rmsd(
    structure_A: gemmi.Structure,
    structure_B: gemmi.Structure,
    atom_type: str = "CA",
) -> pd.DataFrame:
    """Compute per-residue RMSD between two structures.
    
    Returns:
        DataFrame with columns: chain, resi, resname, ca_rmsd, backbone_rmsd, sidechain_rmsd
    """
    rows = []
    
    # Build residue maps
    residues_A = {}
    residues_B = {}
    
    for model in structure_A:
        for chain in model:
            for residue in chain:
                key = (chain.name, residue.seqid.num)
                residues_A[key] = residue
    
    for model in structure_B:
        for chain in model:
            for residue in chain:
                key = (chain.name, residue.seqid.num)
                residues_B[key] = residue
    
    # Compute RMSD for matching residues
    for key in residues_A:
        if key not in residues_B:
            continue
        
        res_A = residues_A[key]
        res_B = residues_B[key]
        
        # Cα RMSD
        ca_A = res_A.find('CA')
        ca_B = res_B.find('CA')
        ca_rmsd = 0.0
        if ca_A and ca_B:
            diff = np.array([ca_A.pos.x - ca_B.pos.x,
                           ca_A.pos.y - ca_B.pos.y,
                           ca_A.pos.z - ca_B.pos.z])
            ca_rmsd = float(np.linalg.norm(diff))
        
        # Backbone RMSD (N, CA, C, O)
        backbone_atoms = ['N', 'CA', 'C', 'O']
        backbone_diffs = []
        for atom_name in backbone_atoms:
            atom_A = res_A.find(atom_name)
            atom_B = res_B.find(atom_name)
            if atom_A and atom_B:
                diff = np.array([atom_A.pos.x - atom_B.pos.x,
                               atom_A.pos.y - atom_B.pos.y,
                               atom_A.pos.z - atom_B.pos.z])
                backbone_diffs.append(np.linalg.norm(diff))
        
        backbone_rmsd = float(np.sqrt(np.mean([d**2 for d in backbone_diffs]))) if backbone_diffs else 0.0
        
        # Side-chain RMSD (centroid)
        sidechain_A = []
        sidechain_B = []
        for atom in res_A:
            if atom.name not in backbone_atoms:
                sidechain_A.append([atom.pos.x, atom.pos.y, atom.pos.z])
        for atom in res_B:
            if atom.name not in backbone_atoms:
                sidechain_B.append([atom.pos.x, atom.pos.y, atom.pos.z])
        
        sidechain_rmsd = 0.0
        if sidechain_A and sidechain_B and len(sidechain_A) == len(sidechain_B):
            sidechain_A = np.array(sidechain_A)
            sidechain_B = np.array(sidechain_B)
            centroid_A = sidechain_A.mean(axis=0)
            centroid_B = sidechain_B.mean(axis=0)
            diff = centroid_A - centroid_B
            sidechain_rmsd = float(np.linalg.norm(diff))
        
        rows.append({
            'chain': key[0],
            'resi': key[1],
            'resname': res_A.name,
            'ca_rmsd': ca_rmsd,
            'backbone_rmsd': backbone_rmsd,
            'sidechain_rmsd': sidechain_rmsd,
        })
    
    return pd.DataFrame(rows)


def compare_models(
    structure_A: gemmi.Structure,
    structure_B: gemmi.Structure,
    anchors: Optional[List[Tuple[str, int, int]]] = None,
) -> Dict:
    """Compare two models and return summary statistics."""
    # Align models
    aligned_B, R, t = align_models(structure_A, structure_B, anchors)
    
    # Compute RMSD
    rmsd_df = compute_rmsd(structure_A, aligned_B)
    
    # Summary statistics
    summary = {
        'ca_rmsd_mean': float(rmsd_df['ca_rmsd'].mean()),
        'ca_rmsd_max': float(rmsd_df['ca_rmsd'].max()),
        'backbone_rmsd_mean': float(rmsd_df['backbone_rmsd'].mean()),
        'sidechain_rmsd_mean': float(rmsd_df['sidechain_rmsd'].mean()),
        'num_residues': len(rmsd_df),
    }
    
    return {
        'aligned_structure': aligned_B,
        'rotation_matrix': R,
        'translation_vector': t,
        'rmsd_df': rmsd_df,
        'summary': summary,
    }

