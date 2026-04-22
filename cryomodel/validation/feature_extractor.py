# cryomodel/validation/feature_extractor.py
"""Feature extraction pipeline for fitcheck."""
from __future__ import annotations
import sys
from typing import Dict, List, Optional

import gemmi
import numpy as np
import pandas as pd
from tqdm import tqdm

from ..gemmi_atoms import sole_atom
from ..io.mrc import MapVolume
from .ringer_lite import ringer_scan_residue
from .q_lite import q_score_atom
from .ca_tube import backbone_continuity
from .local_cc import compute_local_cc_variants
from .geometry_priors import (
    build_heavy_atom_clash_context,
    compute_geometry_features,
    compute_global_clash_z_stats,
    molprobity_like_clashscore_from_context,
    steric_clash_counts_from_context,
    steric_clash_pair_count,
)


def _n_ca_residues(structure: gemmi.Structure) -> int:
    n = 0
    for model in structure:
        for chain in model:
            for res in chain:
                if sole_atom(res, "CA"):
                    n += 1
    return n


def extract_residue_features(
    structure: gemmi.Structure,
    map_vol: MapVolume,
    half1_vol: Optional[MapVolume] = None,
    half2_vol: Optional[MapVolume] = None,
    local_res_map: Optional[MapVolume] = None,
    *,
    show_progress: bool = False,
    monomer_lib_dir: Optional[str] = None,
) -> pd.DataFrame:
    """Extract all features for each residue.
    
    Returns:
        DataFrame with one row per residue and all computed features
    """
    rows = []
    # With tqdm on stderr, keep clash lines on stderr; otherwise stdout so plain logs still show them.
    _log = (lambda m: print(m, file=sys.stderr, flush=True)) if show_progress else print
    _log("  Clash scope: building bond graph + steric/MolProbity scans (heavy atoms, O(N²))…")
    clash_ctx = build_heavy_atom_clash_context(structure, monomer_lib_dir=monomer_lib_dir)
    _log(
        f"  Clash bond graph: {clash_ctx.bond_topology} "
        f"({'Gemmi MonLib + peptide C-N' if clash_ctx.bond_topology == 'gemmi' else 'covalent radii + distance'})"
    )
    clash_counts = steric_clash_counts_from_context(clash_ctx)
    clash_mu, clash_sd = compute_global_clash_z_stats(structure, clash_counts)
    n_heavy_clash = len(clash_ctx.atoms)
    clash_pairs = steric_clash_pair_count(clash_counts)
    clash_per_1000 = (clash_pairs / n_heavy_clash) * 1000.0 if n_heavy_clash else 0.0
    mp_score, mp_pairs, _n_h_mp = molprobity_like_clashscore_from_context(clash_ctx)
    _log(
        f"  Clash — done: internal {clash_pairs} pair(s); "
        f"MolProbity-like {mp_pairs} pair(s), {mp_score:.2f} per 1000 heavy."
    )

    n_ca = _n_ca_residues(structure)
    pbar = tqdm(
        total=n_ca,
        desc="Per-residue features",
        unit="res",
        file=sys.stderr,
        disable=not show_progress,
        mininterval=0.3,
        leave=False,
    )

    for model in structure:
        for chain in model:
            residues = list(chain)
            ca_positions = []
            
            for i, residue in enumerate(residues):
                # Get Cα position
                ca = sole_atom(residue, "CA")
                if not ca:
                    continue
                pbar.update(1)

                ca_pos = np.array([ca.pos.x, ca.pos.y, ca.pos.z])
                ca_positions.append(ca_pos)
                
                # Local resolution (Å) from resmap; NaN if unavailable — do not use 0 (that would falsify Q priors)
                local_res = None
                if local_res_map:
                    local_res = _get_local_resolution(local_res_map, ca_pos)
                
                # Get all atoms in residue
                atom_positions = []
                for atom in residue:
                    if atom.is_hydrogen():
                        continue
                    atom_positions.append(np.array([atom.pos.x, atom.pos.y, atom.pos.z]))
                
                if len(atom_positions) == 0:
                    continue
                
                atom_positions = np.array(atom_positions)
                
                # Extract features
                features = {
                    'chain': chain.name,
                    'resi': residue.seqid.num,
                    'seqid': str(residue.seqid),
                    'resname': residue.name,
                    'local_res': float(local_res) if local_res is not None else float("nan"),
                    'clashscore_per_1000_atoms': float(clash_per_1000),
                    'molprobity_clashscore': float(mp_score),
                    'molprobity_clash_pairs': int(mp_pairs),
                }
                
                # Ringer-Lite
                ringer_features = ringer_scan_residue(
                    residue, map_vol, half1_vol, half2_vol, local_res if local_res is not None else None
                )
                features.update(ringer_features)
                
                # Q-Lite (average over atoms)
                q_scores = []
                for atom_pos in atom_positions:
                    q_feat = q_score_atom(
                        atom_pos,
                        map_vol,
                        float(local_res) if local_res is not None else 3.0,
                        half1_vol,
                        half2_vol,
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
                    atom_positions,
                    map_vol,
                    half1_vol,
                    half2_vol,
                    model_resolution_A=float(local_res) if local_res is not None else 3.0,
                )
                features.update(cc_features)
                
                # Geometry features
                geometry_features = compute_geometry_features(
                    residue,
                    residues,
                    i,
                    chain.name,
                    clash_counts,
                    clash_mu,
                    clash_sd,
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
    
    pbar.close()
    return pd.DataFrame(rows)


def _get_local_resolution(local_res_map: MapVolume, position: np.ndarray) -> Optional[float]:
    """Trilinear sample when a Gemmi grid is available; else nearest voxel (legacy)."""
    px, py, pz = float(position[0]), float(position[1]), float(position[2])
    if local_res_map.grid is not None:
        v = float(local_res_map.grid.interpolate_value(gemmi.Position(px, py, pz), order=1))
        if not np.isfinite(v):
            return None
        return v

    origin = local_res_map.origin_xyzA
    apix = local_res_map.apix
    data = local_res_map.data_zyx
    vox = (position - origin) / apix
    z, y, x = int(round(vox[2])), int(round(vox[1])), int(round(vox[0]))
    if 0 <= z < data.shape[0] and 0 <= y < data.shape[1] and 0 <= x < data.shape[2]:
        return float(data[z, y, x])
    return None

