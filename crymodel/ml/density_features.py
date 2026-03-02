# crymodel/ml/density_features.py
"""Extract density map features at candidate locations."""
from __future__ import annotations
import numpy as np
from pathlib import Path
from typing import Optional

from ..io.mrc import MapVolume, read_map


def extract_density_features(
    point_xyzA: np.ndarray,
    density_map: MapVolume,
    radius_A: float = 3.0,
) -> dict[str, float]:
    """Extract density statistics around a point in the density map.
    
    Args:
        point_xyzA: Point coordinates in Å (x, y, z)
        density_map: MapVolume containing density data
        radius_A: Radius in Å to sample around point
        
    Returns:
        Dictionary with density features: peak, mean, std, min, max, local_SNR
    """
    apix = float(density_map.apix)
    origin = density_map.origin_xyzA
    data = density_map.data_zyx
    
    # Convert point to voxel coordinates (Z, Y, X)
    x, y, z = point_xyzA
    vx = (x - origin[0]) / apix
    vy = (y - origin[1]) / apix
    vz = (z - origin[2]) / apix
    
    # Get bounding box in voxels
    r_vox = radius_A / apix
    z0 = max(0, int(np.floor(vz - r_vox)))
    z1 = min(data.shape[0], int(np.ceil(vz + r_vox)) + 1)
    y0 = max(0, int(np.floor(vy - r_vox)))
    y1 = min(data.shape[1], int(np.ceil(vy + r_vox)) + 1)
    x0 = max(0, int(np.floor(vx - r_vox)))
    x1 = min(data.shape[2], int(np.ceil(vx + r_vox)) + 1)
    
    if z0 >= z1 or y0 >= y1 or x0 >= x1:
        # Point outside map
        return {
            "density_peak": np.nan,
            "density_mean": np.nan,
            "density_std": np.nan,
            "density_min": np.nan,
            "density_max": np.nan,
            "density_local_SNR": np.nan,
        }
    
    # Extract local region
    local_region = data[z0:z1, y0:y1, x0:x1].copy().astype(np.float32)
    
    # Distance mask (within radius)
    zz, yy, xx = np.meshgrid(
        np.arange(z0, z1),
        np.arange(y0, y1),
        np.arange(x0, x1),
        indexing='ij'
    )
    dists = np.sqrt(
        (zz - vz)**2 + (yy - vy)**2 + (xx - vx)**2
    ) * apix
    mask = dists <= radius_A
    
    if mask.sum() == 0:
        return {
            "density_peak": np.nan,
            "density_mean": np.nan,
            "density_std": np.nan,
            "density_min": np.nan,
            "density_max": np.nan,
            "density_local_SNR": np.nan,
        }
    
    masked_vals = local_region[mask]
    
    peak = float(np.max(masked_vals))
    mean = float(np.mean(masked_vals))
    std = float(np.std(masked_vals))
    min_val = float(np.min(masked_vals))
    max_val = float(np.max(masked_vals))
    
    # Local SNR: peak / std (or peak / mean if std is 0)
    local_snr = peak / std if std > 1e-6 else (peak / mean if mean > 1e-6 else np.nan)
    
    return {
        "density_peak": peak,
        "density_mean": mean,
        "density_std": std,
        "density_min": min_val,
        "density_max": max_val,
        "density_local_SNR": local_snr,
    }


def extract_halfmap_features(
    point_xyzA: np.ndarray,
    half1: MapVolume,
    half2: MapVolume,
    radius_A: float = 3.0,
) -> dict[str, float]:
    """Extract half-map FSC-like features at a point.
    
    Args:
        point_xyzA: Point coordinates in Å (x, y, z)
        half1: First half-map
        half2: Second half-map
        radius_A: Radius in Å to sample
        
    Returns:
        Dictionary with half-map features: half1_val, half2_val, half_corr, half_SNR
    """
    # Sample values at point (simple nearest neighbor for speed)
    apix = float(half1.apix)
    origin = half1.origin_xyzA
    
    x, y, z = point_xyzA
    vx = int(round((x - origin[0]) / apix))
    vy = int(round((y - origin[1]) / apix))
    vz = int(round((z - origin[2]) / apix))
    
    # Check bounds
    def safe_sample(data, vz, vy, vx):
        if 0 <= vz < data.shape[0] and 0 <= vy < data.shape[1] and 0 <= vx < data.shape[2]:
            return float(data[vz, vy, vx])
        return np.nan
    
    h1_val = safe_sample(half1.data_zyx, vz, vy, vx)
    h2_val = safe_sample(half2.data_zyx, vz, vy, vx)
    
    # Correlation and SNR-like metrics
    half_corr = np.nan  # Could compute over local region if needed
    half_snr = (h1_val + h2_val) / (abs(h1_val - h2_val) + 1e-6) if not (np.isnan(h1_val) or np.isnan(h2_val)) else np.nan
    
    return {
        "half1_val": h1_val,
        "half2_val": h2_val,
        "half_corr": half_corr,
        "half_SNR": half_snr,
    }

