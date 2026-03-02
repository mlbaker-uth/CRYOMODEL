# crymodel/pathalker/pseudoatoms.py
"""Pseudoatom generation for pathwalking."""
from __future__ import annotations
import numpy as np
from typing import Literal
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from sklearn.mixture import GaussianMixture as GMM
from sklearn.cluster import Birch, MeanShift

from ..io.mrc import MapVolume


PseudoatomMethod = Literal["kmeans", "sc", "ac", "ms", "gmm", "birch"]


def generate_pseudoatoms(
    map_vol: MapVolume,
    threshold: float,
    n_pseudoatoms: int,
    method: PseudoatomMethod = "kmeans",
    random_state: int = 42,
) -> np.ndarray:
    """Generate pseudoatoms from density map.
    
    Args:
        map_vol: MapVolume with density data
        threshold: Density threshold for selecting voxels
        n_pseudoatoms: Target number of pseudoatoms (should match number of C-alpha atoms)
        method: Clustering method ('kmeans', 'sc', 'ac', 'ms', 'gmm', 'birch')
        random_state: Random seed for reproducibility
        
    Returns:
        (n_pseudoatoms, 3) array of pseudoatom coordinates in Å (x, y, z)
    """
    # Get voxel coordinates above threshold
    data = map_vol.data_zyx
    apix = map_vol.apix
    origin = map_vol.origin_xyzA
    
    # Find voxels above threshold
    above_threshold = data > threshold
    voxel_indices = np.array(np.where(above_threshold)).T  # (z, y, x)
    
    if len(voxel_indices) == 0:
        raise ValueError(f"No voxels above threshold {threshold}")
    
    # Convert voxel indices to Å coordinates
    # Map: (z, y, x) voxels -> (x, y, z) Å
    xyz_vox = np.zeros((len(voxel_indices), 3))
    xyz_vox[:, 0] = origin[0] + voxel_indices[:, 2] * apix  # x
    xyz_vox[:, 1] = origin[1] + voxel_indices[:, 1] * apix  # y
    xyz_vox[:, 2] = origin[2] + voxel_indices[:, 0] * apix  # z
    
    print(f"  Found {len(voxel_indices)} voxels above threshold {threshold}")
    print(f"  Generating {n_pseudoatoms} pseudoatoms using {method}")
    
    # Generate pseudoatoms using clustering
    if method == "kmeans":
        km = KMeans(
            n_clusters=n_pseudoatoms,
            max_iter=1000,
            n_init=100,
            tol=1e-9,
            random_state=random_state,
        ).fit(xyz_vox)
        centers = km.cluster_centers_
        
    elif method == "sc":
        sc = SpectralClustering(
            n_clusters=n_pseudoatoms,
            random_state=random_state,
        ).fit(xyz_vox)
        centers = _find_cluster_centers(sc.labels_, xyz_vox, n_pseudoatoms)
        
    elif method == "ac":
        ac = AgglomerativeClustering(n_clusters=n_pseudoatoms).fit(xyz_vox)
        centers = _find_cluster_centers(ac.labels_, xyz_vox, n_pseudoatoms)
        
    elif method == "ms":
        ms = MeanShift(bandwidth=2.0).fit(xyz_vox)
        centers = ms.cluster_centers_
        # MeanShift doesn't guarantee n_pseudoatoms, so we may need to adjust
        if len(centers) != n_pseudoatoms:
            print(f"  Warning: MeanShift produced {len(centers)} clusters, expected {n_pseudoatoms}")
            # Optionally re-cluster to get exact number
            if len(centers) > n_pseudoatoms:
                # Use KMeans on the MeanShift centers to reduce to target number
                km = KMeans(n_clusters=n_pseudoatoms, random_state=random_state).fit(centers)
                centers = km.cluster_centers_
            elif len(centers) < n_pseudoatoms:
                # Need to add more centers (could use KMeans on original data)
                km = KMeans(n_clusters=n_pseudoatoms, random_state=random_state).fit(xyz_vox)
                centers = km.cluster_centers_
        
    elif method == "gmm":
        gm = GMM(
            n_components=n_pseudoatoms,
            covariance_type="tied",
            random_state=random_state,
        ).fit(xyz_vox)
        centers = gm.means_
        
    elif method == "birch":
        birch = Birch(n_clusters=n_pseudoatoms, threshold=1e-8).fit(xyz_vox)
        centers = _find_cluster_centers(birch.subcluster_labels_, xyz_vox, n_pseudoatoms)
        
    else:
        raise ValueError(f"Unknown pseudoatom method: {method}")
    
    return centers.astype(np.float32)


def _find_cluster_centers(labels: np.ndarray, points: np.ndarray, n_clusters: int) -> np.ndarray:
    """Find cluster centers from labels and points."""
    centers = []
    for i in range(n_clusters):
        cluster_points = points[labels == i]
        if len(cluster_points) > 0:
            centers.append(cluster_points.mean(axis=0))
        else:
            # Empty cluster - use a random point or mean of all points
            centers.append(points.mean(axis=0))
    return np.array(centers, dtype=np.float32)


def add_noise_to_pseudoatoms(
    pseudoatoms: np.ndarray,
    noise_level: float,
    random_state: int = 42,
) -> np.ndarray:
    """Add Gaussian noise to pseudoatom positions.
    
    Args:
        pseudoatoms: (N, 3) array of pseudoatom coordinates in Å
        noise_level: Standard deviation of noise in Å
        random_state: Random seed
        
    Returns:
        Noisy pseudoatom coordinates
    """
    if noise_level <= 0:
        return pseudoatoms.copy()
    
    rng = np.random.RandomState(random_state)
    noise = rng.normal(0, noise_level, pseudoatoms.shape)
    return (pseudoatoms + noise).astype(np.float32)

