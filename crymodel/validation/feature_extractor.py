# crymodel/validation/feature_extractor.py
"""Feature extraction pipeline for fitcheck."""
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import gemmi

from ..io.mrc import MapVolume, read_map
from .ringer_lite import ringer_scan_residue
from .q_lite import q_score_atom
from .ca_tube import backbone_continuity
from .local_cc import compute_local_cc_variants
from .geometry_priors import compute_geometry_features


def extract_residue_features(
    structure: gemmi.Structure,
    map_vol: MapVolume,
    half1_vol: Optional[MapVolume] = None,
    half2_vol: Optional[MapVolume] = None,
    local_res_map: Optional[MapVolume] = None,
) -> pd.DataFrame:
    """Extract all features for each residue.
    
    Returns:
        DataFrame with one row per residue and all computed features
    """
    rows = []
    
    for model in structure:
        for chain in model:
            residues = list(chain)
            ca_positions = []
            
            for i, residue in enumerate(residues):
                # Get Cα position
                ca = residue.find('CA')
                if not ca:
                    continue
                
                ca_pos = np.array([ca.pos.x, ca.pos.y, ca.pos.z])
                ca_positions.append(ca_pos)
                
                # Get local resolution
                local_res = None
                if local_res_map:
                    local_res = _get_local_resolution(local_res_map, ca_pos)
                
                # Get all atoms in residue
                atom_positions = []
                for atom in residue:
                    if atom.element.name != 'H':  # Skip hydrogens
                        atom_positions.append(np.array([atom.pos.x, atom.pos.y, atom.pos.z]))
                
                if len(atom_positions) == 0:
                    continue
                
                atom_positions = np.array(atom_positions)
                
                # Extract features
                features = {
                    'chain': chain.name,
                    'resi': residue.seqid.num,
                    'resname': residue.name,
                    'local_res': local_res if local_res else 0.0,
                }
                
                # Ringer-Lite
                ringer_features = ringer_scan_residue(
                    residue, map_vol, half1_vol, half2_vol, local_res
                )
                features.update(ringer_features)
                
                # Q-Lite (average over atoms)
                q_scores = []
                for atom_pos in atom_positions:
                    q_feat = q_score_atom(
                        atom_pos, map_vol, local_res or 3.0, half1_vol, half2_vol
                    )
                    q_scores.append(q_feat['Q'])
                
                if q_scores:
                    features['Q_mean'] = float(np.mean(q_scores))
                    features['Q_min'] = float(np.min(q_scores))
                else:
                    features['Q_mean'] = 0.0
                    features['Q_min'] = 0.0
                
                # Local CC variants
                cc_features = compute_local_cc_variants(
                    atom_positions, map_vol, half1_vol, half2_vol
                )
                features.update(cc_features)
                
                # Geometry features
                geometry_features = compute_geometry_features(
                    residue, residues, local_res
                )
                features.update(geometry_features)
                
                rows.append(features)
            
            # Backbone continuity (per chain)
            if len(ca_positions) >= 2:
                ca_positions_array = np.array(ca_positions)
                continuity_features = backbone_continuity(
                    ca_positions_array, map_vol, half1_vol, half2_vol
                )
                # Add to last residue of chain
                if rows:
                    rows[-1].update(continuity_features)
    
    return pd.DataFrame(rows)


def _get_local_resolution(local_res_map: MapVolume, position: np.ndarray) -> Optional[float]:
    """Get local resolution at a position."""
    origin = local_res_map.origin_xyzA
    apix = local_res_map.apix
    data = local_res_map.data_zyx
    
    vox = (position - origin) / apix
    z, y, x = int(round(vox[2])), int(round(vox[1])), int(round(vox[0]))
    
    if (0 <= z < data.shape[0] and 
        0 <= y < data.shape[1] and 
        0 <= x < data.shape[2]):
        return float(data[z, y, x])
    
    return None

