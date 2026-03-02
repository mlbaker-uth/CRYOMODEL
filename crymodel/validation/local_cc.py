# crymodel/validation/local_cc.py
"""Local CC variants: CC_mask, CC_box, ZNCC."""
from __future__ import annotations
import numpy as np
from typing import Dict, Optional, Tuple
from scipy.ndimage import binary_dilation, generate_binary_structure

from ..io.mrc import MapVolume


def compute_local_cc_variants(
    atom_positions: np.ndarray,
    map_vol: MapVolume,
    half1_vol: Optional[MapVolume] = None,
    half2_vol: Optional[MapVolume] = None,
    mask_radius: float = 2.0,
    box_size: float = 4.0,
) -> Dict[str, float]:
    """Compute local CC variants for a set of atoms.
    
    Args:
        atom_positions: (N, 3) array of atom positions in Å
        map_vol: MapVolume with density data
        half1_vol: Optional half-map 1
        half2_vol: Optional half-map 2
        mask_radius: Radius for CC_mask (Å)
        box_size: Size of box for CC_box (Å)
        
    Returns:
        Dictionary with CC variants and half-map deltas
    """
    if len(atom_positions) == 0:
        return {
            'CC_mask': 0.0,
            'CC_box': 0.0,
            'ZNCC': 0.0,
            'CC_half1': 0.0,
            'CC_half2': 0.0,
            'CC_delta': 0.0,
        }
    
    # CC_mask: 2Å mask around atoms
    cc_mask = _compute_cc_mask(atom_positions, map_vol, mask_radius)
    
    # CC_box: Fixed 4Å cube
    cc_box = _compute_cc_box(atom_positions, map_vol, box_size)
    
    # ZNCC: Zero-mean normalized cross-correlation
    zncc = _compute_zncc(atom_positions, map_vol, mask_radius)
    
    # Half-map versions
    cc_half1 = 0.0
    cc_half2 = 0.0
    if half1_vol:
        cc_half1 = _compute_cc_mask(atom_positions, half1_vol, mask_radius)
    if half2_vol:
        cc_half2 = _compute_cc_mask(atom_positions, half2_vol, mask_radius)
    
    # Delta: full vs best half-map
    cc_delta = cc_mask - max(cc_half1, cc_half2) if (half1_vol or half2_vol) else 0.0
    
    return {
        'CC_mask': float(cc_mask),
        'CC_box': float(cc_box),
        'ZNCC': float(zncc),
        'CC_half1': float(cc_half1),
        'CC_half2': float(cc_half2),
        'CC_delta': float(cc_delta),
    }


def _compute_cc_mask(
    atom_positions: np.ndarray,
    map_vol: MapVolume,
    radius: float,
) -> float:
    """Compute CC with mask around atoms."""
    origin = map_vol.origin_xyzA
    apix = map_vol.apix
    data = map_vol.data_zyx
    
    # Create mask
    mask = np.zeros_like(data, dtype=bool)
    radius_vox = radius / apix
    
    for atom_pos in atom_positions:
        vox = (atom_pos - origin) / apix
        z, y, x = int(round(vox[2])), int(round(vox[1])), int(round(vox[0]))
        
        # Create spherical mask
        z_min = max(0, int(z - radius_vox))
        z_max = min(data.shape[0], int(z + radius_vox) + 1)
        y_min = max(0, int(y - radius_vox))
        y_max = min(data.shape[1], int(y + radius_vox) + 1)
        x_min = max(0, int(x - radius_vox))
        x_max = min(data.shape[2], int(x + radius_vox) + 1)
        
        for iz in range(z_min, z_max):
            for iy in range(y_min, y_max):
                for ix in range(x_min, x_max):
                    dz, dy, dx = iz - z, iy - y, ix - x
                    dist = np.sqrt(dz**2 + dy**2 + dx**2) * apix
                    if dist <= radius:
                        mask[iz, iy, ix] = True
    
    # Compute CC in masked region
    masked_data = data[mask]
    if len(masked_data) == 0:
        return 0.0
    
    # For CC, we need model density - use uniform for now
    # In practice, this would use a model map
    model_density = np.ones_like(masked_data)
    
    # Pearson correlation
    if masked_data.std() > 1e-6 and model_density.std() > 1e-6:
        cc = np.corrcoef(masked_data, model_density)[0, 1]
        return float(cc) if not np.isnan(cc) else 0.0
    return 0.0


def _compute_cc_box(
    atom_positions: np.ndarray,
    map_vol: MapVolume,
    box_size: float,
) -> float:
    """Compute CC in fixed-size box."""
    if len(atom_positions) == 0:
        return 0.0
    
    # Use centroid of atoms
    centroid = atom_positions.mean(axis=0)
    
    origin = map_vol.origin_xyzA
    apix = map_vol.apix
    data = map_vol.data_zyx
    
    # Extract box
    half_box = box_size / 2.0
    vox = (centroid - origin) / apix
    z, y, x = int(round(vox[2])), int(round(vox[1])), int(round(vox[0]))
    
    box_vox = int(box_size / apix)
    half_box_vox = box_vox // 2
    
    z_min = max(0, z - half_box_vox)
    z_max = min(data.shape[0], z + half_box_vox + 1)
    y_min = max(0, y - half_box_vox)
    y_max = min(data.shape[1], y + half_box_vox + 1)
    x_min = max(0, x - half_box_vox)
    x_max = min(data.shape[2], x + half_box_vox + 1)
    
    box_data = data[z_min:z_max, y_min:y_max, x_min:x_max]
    if box_data.size == 0:
        return 0.0
    
    # For CC, use uniform model density
    model_density = np.ones_like(box_data)
    
    if box_data.std() > 1e-6 and model_density.std() > 1e-6:
        cc = np.corrcoef(box_data.flatten(), model_density.flatten())[0, 1]
        return float(cc) if not np.isnan(cc) else 0.0
    return 0.0


def _compute_zncc(
    atom_positions: np.ndarray,
    map_vol: MapVolume,
    radius: float,
) -> float:
    """Compute zero-mean normalized cross-correlation."""
    # Similar to CC_mask but with zero-mean normalization
    origin = map_vol.origin_xyzA
    apix = map_vol.apix
    data = map_vol.data_zyx
    
    # Create mask
    mask = np.zeros_like(data, dtype=bool)
    radius_vox = radius / apix
    
    for atom_pos in atom_positions:
        vox = (atom_pos - origin) / apix
        z, y, x = int(round(vox[2])), int(round(vox[1])), int(round(vox[0]))
        
        z_min = max(0, int(z - radius_vox))
        z_max = min(data.shape[0], int(z + radius_vox) + 1)
        y_min = max(0, int(y - radius_vox))
        y_max = min(data.shape[1], int(y + radius_vox) + 1)
        x_min = max(0, int(x - radius_vox))
        x_max = min(data.shape[2], int(x + radius_vox) + 1)
        
        for iz in range(z_min, z_max):
            for iy in range(y_min, y_max):
                for ix in range(x_min, x_max):
                    dz, dy, dx = iz - z, iy - y, ix - x
                    dist = np.sqrt(dz**2 + dy**2 + dx**2) * apix
                    if dist <= radius:
                        mask[iz, iy, ix] = True
    
    masked_data = data[mask]
    if len(masked_data) == 0:
        return 0.0
    
    # Zero-mean normalization
    masked_data_norm = masked_data - masked_data.mean()
    model_density = np.ones_like(masked_data_norm)
    model_density_norm = model_density - model_density.mean()
    
    # Normalized cross-correlation
    numerator = np.sum(masked_data_norm * model_density_norm)
    denominator = np.sqrt(np.sum(masked_data_norm**2) * np.sum(model_density_norm**2))
    
    if denominator > 1e-6:
        zncc = numerator / denominator
        return float(zncc)
    return 0.0

