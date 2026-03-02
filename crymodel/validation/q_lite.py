# crymodel/validation/q_lite.py
"""Q-Lite: Atom resolvability index."""
from __future__ import annotations
import numpy as np
from typing import Dict, Optional, Tuple
from scipy.stats import pearsonr
from scipy.ndimage import gaussian_filter

from ..io.mrc import MapVolume


def compute_expected_profile(
    local_res: float,
    voxel_size: float = 0.05,
    max_radius: float = 2.0,
) -> np.ndarray:
    """Compute expected radial density profile for an atom.
    
    Args:
        local_res: Local resolution (Å)
        voxel_size: Sampling step size (Å)
        max_radius: Maximum radius to sample (Å)
        
    Returns:
        Expected profile as 1D array
    """
    # Effective sigma from local resolution
    # FWHM ≈ k * local_res, σ = FWHM / 2.355
    k = 0.4  # Empirically fit
    fwhm = k * local_res
    sigma = fwhm / 2.355
    
    # Generate radial profile
    radii = np.arange(0, max_radius + voxel_size, voxel_size)
    
    # Normalized Gaussian profile
    profile = np.exp(-0.5 * (radii / sigma) ** 2)
    profile = profile / (np.sqrt(2 * np.pi) * sigma)  # Normalize
    
    return profile


def _sample_density_at_position(map_vol: MapVolume, position: np.ndarray) -> float:
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


def sample_radial_profile(
    map_vol: MapVolume,
    atom_pos: np.ndarray,
    max_radius: float = 2.0,
    step_size: float = 0.05,
) -> np.ndarray:
    """Sample radial density profile around an atom.
    
    Args:
        map_vol: MapVolume with density data
        atom_pos: Atom position (x, y, z) in Å
        max_radius: Maximum radius to sample (Å)
        step_size: Sampling step size (Å)
        
    Returns:
        Density profile as 1D array
    """
    apix = map_vol.apix
    
    # Sample radial profile
    radii = np.arange(0, max_radius + step_size, step_size)
    profile = []
    
    for r in radii:
        if r == 0:
            # Sample at center
            density = _sample_density_at_position(map_vol, atom_pos)
            profile.append(density)
        else:
            # Sample at radius r
            densities_at_r = []
            # Use spherical sampling
            n_samples = max(8, int(2 * np.pi * r / step_size))
            for _ in range(n_samples):
                # Random direction on sphere
                theta = np.random.uniform(0, 2 * np.pi)
                phi = np.random.uniform(0, np.pi)
                direction = np.array([
                    np.sin(phi) * np.cos(theta),
                    np.sin(phi) * np.sin(theta),
                    np.cos(phi)
                ])
                sample_pos = atom_pos + r * direction
                density = _sample_density_at_position(map_vol, sample_pos)
                densities_at_r.append(density)
            profile.append(np.mean(densities_at_r))
    
    return np.array(profile)


def q_score_atom(
    atom_pos: np.ndarray,
    map_vol: MapVolume,
    local_res: float,
    half1_vol: Optional[MapVolume] = None,
    half2_vol: Optional[MapVolume] = None,
    max_radius: float = 2.0,
    step_size: float = 0.05,
) -> Dict[str, float]:
    """Compute Q-Lite score for an atom.
    
    Returns:
        Dictionary with keys:
        - Q: Normalized correlation with expected profile
        - Q_half1: Q score for half-map 1
        - Q_half2: Q score for half-map 2
        - Q_delta: Difference between full and best half-map
        - A: Amplitude ratio (best fit scale)
    """
    # Sample observed profile
    observed = sample_radial_profile(map_vol, atom_pos, max_radius, step_size)
    
    # Compute expected profile
    expected = compute_expected_profile(local_res, step_size, max_radius)
    
    # Normalize profiles
    observed_norm = (observed - observed.mean()) / (observed.std() + 1e-12)
    expected_norm = (expected - expected.mean()) / (expected.std() + 1e-12)
    
    # Compute correlation (Q score)
    if len(observed_norm) > 1 and observed_norm.std() > 1e-6:
        corr, _ = pearsonr(observed_norm, expected_norm)
        Q = float(corr) if not np.isnan(corr) else 0.0
    else:
        Q = 0.0
    
    # Compute amplitude ratio (best fit scale)
    if expected.std() > 1e-6:
        A = float(np.sum(observed * expected) / np.sum(expected ** 2))
    else:
        A = 0.0
    
    # Half-map versions
    Q_half1 = 0.0
    Q_half2 = 0.0
    if half1_vol:
        observed_h1 = sample_radial_profile(half1_vol, atom_pos, max_radius, step_size)
        observed_h1_norm = (observed_h1 - observed_h1.mean()) / (observed_h1.std() + 1e-12)
        if len(observed_h1_norm) > 1 and observed_h1_norm.std() > 1e-6:
            corr_h1, _ = pearsonr(observed_h1_norm, expected_norm)
            Q_half1 = float(corr_h1) if not np.isnan(corr_h1) else 0.0
    
    if half2_vol:
        observed_h2 = sample_radial_profile(half2_vol, atom_pos, max_radius, step_size)
        observed_h2_norm = (observed_h2 - observed_h2.mean()) / (observed_h2.std() + 1e-12)
        if len(observed_h2_norm) > 1 and observed_h2_norm.std() > 1e-6:
            corr_h2, _ = pearsonr(observed_h2_norm, expected_norm)
            Q_half2 = float(corr_h2) if not np.isnan(corr_h2) else 0.0
    
    # Delta Q (full vs best half-map)
    Q_delta = Q - max(Q_half1, Q_half2) if (half1_vol or half2_vol) else 0.0
    
    return {
        'Q': Q,
        'Q_half1': Q_half1,
        'Q_half2': Q_half2,
        'Q_delta': Q_delta,
        'A': A,
    }

