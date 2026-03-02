# crymodel/validation/ca_tube.py
"""Cα-Tube: Backbone trace continuity."""
from __future__ import annotations
import numpy as np
from typing import Dict, List, Optional, Tuple
try:
    from scipy.interpolate import splprep, splev
except ImportError:
    # Fallback if scipy.interpolate not available
    splprep = None
    splev = None

from ..io.mrc import MapVolume


def build_ca_spline(ca_positions: np.ndarray) -> Optional[Tuple]:
    """Build spline through Cα positions.
    
    Returns:
        Spline tuple from scipy.interpolate.splprep, or None if unavailable
    """
    if splprep is None:
        return None
    
    if len(ca_positions) < 2:
        return None
    
    # Transpose for splprep (expects (3, N))
    ca_t = ca_positions.T
    
    # Build spline
    tck, u = splprep(ca_t, s=0, k=min(3, len(ca_positions) - 1))
    return tck, u


def sample_tube_along_spline(
    tck: tuple,
    u: np.ndarray,
    map_vol: MapVolume,
    tube_radius: float = 1.25,
    n_points_per_residue: int = 10,
) -> np.ndarray:
    """Sample density along cylindrical tube around spline.
    
    Returns:
        Array of density values along the tube
    """
    if splev is None:
        # Fallback if scipy.interpolate not available
        return np.zeros(10)
    
    origin = map_vol.origin_xyzA
    apix = map_vol.apix
    data = map_vol.data_zyx
    
    # Generate sample points along spline
    u_samples = np.linspace(0, 1, len(u) * n_points_per_residue)
    points = splev(u_samples, tck)
    points = np.array(points).T  # (N, 3)
    
    # Sample density in cylindrical tube around each point
    densities = []
    
    for i, point in enumerate(points):
        # Get tangent direction
        if i == 0:
            tangent = points[1] - points[0]
        elif i == len(points) - 1:
            tangent = points[-1] - points[-2]
        else:
            tangent = points[i + 1] - points[i - 1]
        tangent = tangent / (np.linalg.norm(tangent) + 1e-12)
        
        # Sample in circle perpendicular to tangent
        n_samples = max(8, int(2 * np.pi * tube_radius / apix))
        densities_at_point = []
        
        for j in range(n_samples):
            angle = 2 * np.pi * j / n_samples
            # Generate perpendicular vector
            if abs(tangent[0]) < 0.9:
                perp = np.array([1, 0, 0])
            else:
                perp = np.array([0, 1, 0])
            perp = perp - np.dot(perp, tangent) * tangent
            perp = perp / (np.linalg.norm(perp) + 1e-12)
            
            # Second perpendicular vector
            perp2 = np.cross(tangent, perp)
            perp2 = perp2 / (np.linalg.norm(perp2) + 1e-12)
            
            # Sample position
            sample_pos = point + tube_radius * (np.cos(angle) * perp + np.sin(angle) * perp2)
            
            # Sample density
            vox = (sample_pos - origin) / apix
            z, y, x = int(round(vox[2])), int(round(vox[1])), int(round(vox[0]))
            
            if (0 <= z < data.shape[0] and 
                0 <= y < data.shape[1] and 
                0 <= x < data.shape[2]):
                densities_at_point.append(float(data[z, y, x]))
        
        if densities_at_point:
            densities.append(np.mean(densities_at_point))
        else:
            densities.append(0.0)
    
    return np.array(densities)


def backbone_continuity(
    ca_positions: np.ndarray,
    map_vol: MapVolume,
    half1_vol: Optional[MapVolume] = None,
    half2_vol: Optional[MapVolume] = None,
    tube_radius: float = 1.25,
) -> Dict[str, float]:
    """Compute backbone continuity metrics.
    
    Args:
        ca_positions: (N, 3) array of Cα positions in Å
        map_vol: MapVolume with density data
        half1_vol: Optional half-map 1
        half2_vol: Optional half-map 2
        tube_radius: Radius of sampling tube (Å)
        
    Returns:
        Dictionary with keys:
        - continuity_score: Fraction of points above threshold
        - continuity_mean: Mean density in tube
        - continuity_std: Std density in tube
        - continuity_half_drop: Half-map stability
    """
    if len(ca_positions) < 2:
        return {
            'continuity_score': 0.0,
            'continuity_mean': 0.0,
            'continuity_std': 0.0,
            'continuity_half_drop': 0.0,
        }
    
    # Build spline
    spline_result = build_ca_spline(ca_positions)
    if spline_result is None:
        return {
            'continuity_score': 0.0,
            'continuity_mean': 0.0,
            'continuity_std': 0.0,
            'continuity_half_drop': 0.0,
        }
    
    tck, u = spline_result
    
    # Sample tube
    densities = sample_tube_along_spline(tck, u, map_vol, tube_radius)
    
    # Compute statistics
    mean_density = float(densities.mean())
    std_density = float(densities.std())
    threshold = mean_density + 1.5 * std_density
    continuity_score = float((densities > threshold).sum() / len(densities))
    
    # Half-map stability
    continuity_half_drop = 0.0
    if half1_vol and half2_vol:
        densities_h1 = sample_tube_along_spline(tck, u, half1_vol, tube_radius)
        densities_h2 = sample_tube_along_spline(tck, u, half2_vol, tube_radius)
        mean_h1 = densities_h1.mean()
        mean_h2 = densities_h2.mean()
        continuity_half_drop = float(abs(mean_h1 - mean_h2))
    
    return {
        'continuity_score': continuity_score,
        'continuity_mean': mean_density,
        'continuity_std': std_density,
        'continuity_half_drop': continuity_half_drop,
    }

