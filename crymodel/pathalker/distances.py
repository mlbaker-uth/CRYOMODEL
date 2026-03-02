# crymodel/pathalker/distances.py
"""Distance matrix calculation for pathwalking."""
from __future__ import annotations
import numpy as np
from scipy.spatial.distance import cdist
from typing import Optional

from ..io.mrc import MapVolume


def compute_distance_matrix(
    pseudoatoms: np.ndarray,
    map_vol: Optional[MapVolume] = None,
    map_weighted: bool = False,
    threshold: float = 0.5,
) -> np.ndarray:
    """Compute distance matrix between pseudoatoms.
    
    Args:
        pseudoatoms: (N, 3) array of pseudoatom coordinates in Å (x, y, z)
        map_vol: Optional MapVolume for map-weighted distances
        map_weighted: If True, weight distances by map density
        threshold: Density threshold for map weighting
        
    Returns:
        (N, N) distance matrix in Å
    """
    if map_weighted and map_vol is not None:
        return _compute_map_weighted_distances(pseudoatoms, map_vol, threshold)
    else:
        return _compute_euclidean_distances(pseudoatoms)


def _compute_euclidean_distances(pseudoatoms: np.ndarray) -> np.ndarray:
    """Compute Euclidean distance matrix."""
    return cdist(pseudoatoms, pseudoatoms, metric="euclidean").astype(np.float32)


def _compute_map_weighted_distances(
    pseudoatoms: np.ndarray,
    map_vol: MapVolume,
    threshold: float,
) -> np.ndarray:
    """Compute map-weighted distance matrix.
    
    Distances are weighted by inverse of average map density along the path.
    Low density paths are penalized (higher effective distance).
    
    Args:
        pseudoatoms: (N, 3) array of pseudoatom coordinates in Å
        map_vol: MapVolume with density data
        threshold: Density threshold
        
    Returns:
        (N, N) weighted distance matrix in Å
    """
    n = len(pseudoatoms)
    apix = map_vol.apix
    origin = map_vol.origin_xyzA
    data = map_vol.data_zyx
    
    # Convert pseudoatoms to voxel coordinates
    xyz_vox = (pseudoatoms - origin) / apix
    # Map: (x, y, z) Å -> (z, y, x) voxels
    zyx_vox = xyz_vox[:, [2, 1, 0]]
    
    distances = np.zeros((n, n), dtype=np.float32)
    
    for i in range(n):
        for j in range(n):
            if i == j:
                distances[i, j] = 0.0
                continue
            
            p1_zyx = zyx_vox[i]
            p2_zyx = zyx_vox[j]
            
            # Euclidean distance
            euclidean_dist = np.linalg.norm(pseudoatoms[i] - pseudoatoms[j])
            
            # Sample points along the line between p1 and p2
            n_samples = max(10, int(euclidean_dist / apix))
            t_values = np.linspace(0, 1, n_samples)
            
            map_values = []
            for t in t_values:
                point_zyx = (1 - t) * p1_zyx + t * p2_zyx
                iz, iy, ix = int(round(point_zyx[0])), int(round(point_zyx[1])), int(round(point_zyx[2]))
                
                # Clip to bounds
                if (0 <= iz < data.shape[0] and 
                    0 <= iy < data.shape[1] and 
                    0 <= ix < data.shape[2]):
                    map_value = data[iz, iy, ix]
                    map_values.append(map_value)
            
            if len(map_values) > 0:
                avg_map_value = np.mean(map_values)
                # Weight: inverse of map value (low density = high weight)
                # Add small epsilon to avoid division by zero
                if avg_map_value <= 0.0001:
                    avg_map_value = 0.0001
                weight = 1.0 / avg_map_value
                weighted_dist = weight * euclidean_dist
            else:
                # No valid map samples - use high penalty
                weighted_dist = euclidean_dist * 10.0
            
            distances[i, j] = weighted_dist
    
    return distances


def prepare_tsp_distance_matrix(
    distance_matrix: np.ndarray,
    add_depot: bool = True,
) -> np.ndarray:
    """Prepare distance matrix for TSP solver.
    
    Adds depot row/column (zeros) if needed for TSP formulation.
    
    Args:
        distance_matrix: (N, N) distance matrix in Å
        add_depot: If True, add depot row/column (required for some TSP solvers)
        
    Returns:
        Distance matrix ready for TSP solver (integers, scaled by 100)
    """
    if add_depot:
        n = len(distance_matrix)
        # Add zero row at top
        zero_row = np.zeros((1, n), dtype=distance_matrix.dtype)
        dm_with_row = np.vstack([zero_row, distance_matrix])
        # Add zero column at left
        zero_col = np.zeros((n + 1, 1), dtype=distance_matrix.dtype)
        dm_with_depot = np.hstack([zero_col, dm_with_row])
    else:
        dm_with_depot = distance_matrix
    
    # Convert to integers (scale by 100 for precision)
    # TSP solvers typically expect integer distances
    dm_int = (dm_with_depot * 100).astype(int)
    
    return dm_int

