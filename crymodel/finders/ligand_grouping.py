# crymodel/finders/ligand_grouping.py
"""Group near-contiguous ligand density components."""
from __future__ import annotations
import numpy as np
from scipy.spatial.distance import cdist
from scipy.ndimage import label
from scipy.ndimage.morphology import binary_dilation
from typing import List, Tuple, Optional

from ..io.mrc import MapVolume


def group_near_contiguous_components(
    components_xyz: List[np.ndarray],
    ligand_map: MapVolume,
    threshold_original: float = 0.5,
    threshold_lower: float = 0.3,
    min_pseudoatoms: int = 2,
    max_pseudoatoms: int = 80,
    distance_threshold_A: float = 5.0,
    connectivity_radius_A: float = 3.0,
    component_ids: Optional[List[str]] = None
) -> Tuple[List[np.ndarray], List[List[str]]]:
    """Group near-contiguous ligand components.
    
    This function:
    1. Filters out components with < min_pseudoatoms
    2. Checks connectivity at original threshold
    3. Checks connectivity at lower threshold to find near-contiguous regions
    4. Groups components that are connected at the lower threshold or very close
    
    Args:
        components_xyz: List of (N_i, 3) arrays of pseudoatom coordinates in Å
        ligand_map: MapVolume with ligand density
        threshold_original: Original threshold used for segmentation
        threshold_lower: Lower threshold for checking continuity
        min_pseudoatoms: Minimum pseudoatoms to keep a component
        max_pseudoatoms: Maximum pseudoatoms per grouped component (prevents over-grouping)
        distance_threshold_A: Maximum distance between components to consider grouping (Å)
        connectivity_radius_A: Radius for checking connectivity in map (Å)
        component_ids: Optional list of component IDs (e.g., chain IDs) corresponding to components_xyz
        
    Returns:
        Tuple of:
        - List of grouped component coordinates (merged components)
        - List of lists of original component IDs for each grouped component
    """
    # Generate component IDs if not provided
    if component_ids is None:
        component_ids = [f"component_{i}" for i in range(len(components_xyz))]
    
    # Step 1: Filter out small components
    filtered_components = []
    filtered_indices = []
    filtered_component_ids = []
    for i, comp_xyz in enumerate(components_xyz):
        if len(comp_xyz) >= min_pseudoatoms:
            filtered_components.append(comp_xyz)
            filtered_indices.append(i)
            filtered_component_ids.append(component_ids[i])
    
    if not filtered_components:
        return [], []
    
    print(f"  Filtered {len(components_xyz) - len(filtered_components)} components with < {min_pseudoatoms} pseudoatoms")
    
    # Step 2: Build distance matrix between component centroids
    centroids = np.array([comp.mean(axis=0) for comp in filtered_components])
    n_components = len(filtered_components)
    
    if n_components == 1:
        return filtered_components
    
    # Distance matrix
    dist_matrix = cdist(centroids, centroids)
    
    # Step 3: Check connectivity in density map at lower threshold
    # BUT: Only consider density from filtered components (>= min_pseudoatoms)
    # Create a mask of density that belongs to filtered components only
    apix = ligand_map.apix
    origin = ligand_map.origin_xyzA
    
    # First, create a mask of density that belongs ONLY to filtered components
    # This prevents small components from bridging connections
    filtered_components_mask = np.zeros(ligand_map.data_zyx.shape, dtype=bool)
    for comp_xyz in filtered_components:
        comp_xyz_vox = (comp_xyz - origin) / apix
        comp_zyx_vox = comp_xyz_vox[:, [2, 1, 0]]  # (x,y,z) -> (z,y,x)
        
        # Mark voxels near pseudoatoms (within radius)
        radius_vox = connectivity_radius_A / apix
        for z, y, x in comp_zyx_vox:
            iz, iy, ix = int(round(z)), int(round(y)), int(round(x))
            if (0 <= iz < filtered_components_mask.shape[0] and 
                0 <= iy < filtered_components_mask.shape[1] and 
                0 <= ix < filtered_components_mask.shape[2]):
                # Mark a small region around each pseudoatom
                z_min = max(0, iz - int(radius_vox))
                z_max = min(filtered_components_mask.shape[0], iz + int(radius_vox) + 1)
                y_min = max(0, iy - int(radius_vox))
                y_max = min(filtered_components_mask.shape[1], iy + int(radius_vox) + 1)
                x_min = max(0, ix - int(radius_vox))
                x_max = min(filtered_components_mask.shape[2], ix + int(radius_vox) + 1)
                filtered_components_mask[z_min:z_max, y_min:y_max, x_min:x_max] = True
    
    # Now check connectivity at lower threshold, BUT only within the filtered components mask
    # This ensures we only connect components that are actually part of larger structures
    lower_threshold_mask = (ligand_map.data_zyx >= threshold_lower) & filtered_components_mask
    labels_lower, n_labels_lower = label(lower_threshold_mask, structure=np.ones((3, 3, 3)))
    
    # Convert component centroids to voxel coordinates
    centroids_vox = (centroids - origin) / apix
    centroids_zyx_vox = centroids_vox[:, [2, 1, 0]]  # (x,y,z) -> (z,y,x)
    
    # Create connectivity graph based on density continuity at lower threshold
    connectivity_graph = np.zeros((n_components, n_components), dtype=bool)
    
    # For each filtered component, find which label(s) it touches by checking pseudoatom positions
    component_labels = []
    connectivity_radius_vox = connectivity_radius_A / apix
    
    for comp_idx, comp_xyz in enumerate(filtered_components):
        # Convert component pseudoatoms to voxel coordinates
        comp_xyz_vox = (comp_xyz - origin) / apix
        comp_zyx_vox = comp_xyz_vox[:, [2, 1, 0]]  # (x,y,z) -> (z,y,x)
        
        # Find labels that this component touches (only in filtered components mask)
        touched_labels = set()
        for z, y, x in comp_zyx_vox:
            iz, iy, ix = int(round(z)), int(round(y)), int(round(x))
            # Clip to bounds
            if (0 <= iz < labels_lower.shape[0] and 
                0 <= iy < labels_lower.shape[1] and 
                0 <= ix < labels_lower.shape[2]):
                label_val = labels_lower[iz, iy, ix]
                if label_val > 0:
                    touched_labels.add(label_val)
                else:
                    # Check nearby voxels if not directly on a label
                    z_min = max(0, iz - int(connectivity_radius_vox))
                    z_max = min(labels_lower.shape[0], iz + int(connectivity_radius_vox) + 1)
                    y_min = max(0, iy - int(connectivity_radius_vox))
                    y_max = min(labels_lower.shape[1], iy + int(connectivity_radius_vox) + 1)
                    x_min = max(0, ix - int(connectivity_radius_vox))
                    x_max = min(labels_lower.shape[2], ix + int(connectivity_radius_vox) + 1)
                    
                    nearby_labels = labels_lower[z_min:z_max, y_min:y_max, x_min:x_max]
                    unique_labels = np.unique(nearby_labels)
                    touched_labels.update(unique_labels[unique_labels > 0])
        
        component_labels.append(touched_labels)
    
    # Build connectivity: two components are connected if they share a label
    # BUT: Add additional constraints to prevent over-grouping
    for i in range(n_components):
        for j in range(i + 1, n_components):
            # Check if components share a label (connected at lower threshold)
            if component_labels[i] & component_labels[j]:
                # Additional check: components should be reasonably close spatially
                # and have reasonable size (not connect tiny to huge components)
                dist_ij = dist_matrix[i, j]
                size_i = len(filtered_components[i])
                size_j = len(filtered_components[j])
                
                # Only connect if:
                # 1. Combined size would not exceed max_pseudoatoms
                # 2. They're within distance threshold, OR
                # 3. They're close (< 1.5x distance threshold) AND similar size (within 2x)
                combined_size = size_i + size_j
                if combined_size > max_pseudoatoms:
                    # Skip - would exceed maximum size
                    continue
                
                if dist_ij < distance_threshold_A:
                    # Close and within size limit - connect
                    connectivity_graph[i, j] = True
                    connectivity_graph[j, i] = True
                elif dist_ij < distance_threshold_A * 1.5 and (
                    size_i <= size_j * 2 and size_j <= size_i * 2  # Similar size (within 2x)
                ):
                    # Moderately close and similar size - connect
                    connectivity_graph[i, j] = True
                    connectivity_graph[j, i] = True
            # Also check spatial proximity as fallback (but be very conservative)
            else:
                size_i = len(filtered_components[i])
                size_j = len(filtered_components[j])
                combined_size = size_i + size_j
                
                # Only consider fallback if very close and combined size is reasonable
                if (dist_matrix[i, j] < distance_threshold_A * 0.6 and 
                    combined_size <= max_pseudoatoms and
                    size_i <= size_j * 2 and size_j <= size_i * 2):
                    # Check if there's a path through density between them
                    if _check_density_path(
                        centroids_zyx_vox[i], centroids_zyx_vox[j],
                        lower_threshold_mask, connectivity_radius_A / apix
                    ):
                        connectivity_graph[i, j] = True
                        connectivity_graph[j, i] = True
    
    # Step 4: Find connected components (graph clustering)
    grouped = _cluster_connected_components(connectivity_graph)
    
    # Step 5: Merge grouped components and track original component IDs
    # Apply max_pseudoatoms constraint during merging
    merged_components = []
    merged_component_id_groups = []
    for group_indices in grouped:
        if len(group_indices) == 1:
            # Single component, keep as-is (already checked for min_pseudoatoms)
            merged_components.append(filtered_components[group_indices[0]])
            merged_component_id_groups.append([filtered_component_ids[group_indices[0]]])
        else:
            # Merge multiple components, but check total size
            group_components = [filtered_components[i] for i in group_indices]
            total_size = sum(len(comp) for comp in group_components)
            
            if total_size <= max_pseudoatoms:
                # Merge all components in group
                merged_xyz = np.concatenate(group_components, axis=0)
                merged_components.append(merged_xyz)
                # Track which original component IDs were merged
                original_ids = [filtered_component_ids[i] for i in group_indices]
                merged_component_id_groups.append(original_ids)
                # Clean up the indices display
                ids_str = ",".join(original_ids[:5])  # Show first 5 IDs
                if len(original_ids) > 5:
                    ids_str += f",... (+{len(original_ids)-5} more)"
                print(f"  Merged {len(group_indices)} components into one ({total_size} pseudoatoms, original IDs: {ids_str})")
            else:
                # Group too large - split into smaller groups or keep separate
                # For now, keep components separate if merging would exceed max
                print(f"  Warning: Group of {len(group_indices)} components would exceed max_pseudoatoms ({total_size} > {max_pseudoatoms}), keeping separate")
                for i in group_indices:
                    merged_components.append(filtered_components[i])
                    merged_component_id_groups.append([filtered_component_ids[i]])
    
    print(f"  Grouped {len(filtered_components)} components into {len(merged_components)} groups (max_pseudoatoms={max_pseudoatoms})")
    
    return merged_components, merged_component_id_groups


def _check_density_path(
    point1_zyx_vox: np.ndarray,
    point2_zyx_vox: np.ndarray,
    density_mask: np.ndarray,
    max_radius_vox: float
) -> bool:
    """Check if there's a path through density between two points.
    
    Uses binary dilation to see if the two points are connected via density.
    """
    p1 = point1_zyx_vox.astype(int)
    p2 = point2_zyx_vox.astype(int)
    
    # Clip to bounds
    p1 = np.clip(p1, 0, np.array(density_mask.shape) - 1)
    p2 = np.clip(p2, 0, np.array(density_mask.shape) - 1)
    
    # Check if both points are in density
    if not (density_mask[tuple(p1)] and density_mask[tuple(p2)]):
        return False
    
    # Use distance check: if points are close and both in density, consider connected
    dist_vox = np.linalg.norm(p2 - p1)
    if dist_vox < max_radius_vox * 2:
        return True
    
    # For longer distances, could use pathfinding, but for now just check proximity
    return False


def _cluster_connected_components(connectivity_graph: np.ndarray) -> List[List[int]]:
    """Find connected components in a graph using DFS.
    
    Args:
        connectivity_graph: (n, n) boolean adjacency matrix
        
    Returns:
        List of lists, where each inner list contains indices of connected components
    """
    n = connectivity_graph.shape[0]
    visited = np.zeros(n, dtype=bool)
    clusters = []
    
    for start_idx in range(n):
        if visited[start_idx]:
            continue
        
        # DFS to find all connected nodes
        cluster = []
        stack = [start_idx]
        
        while stack:
            node = stack.pop()
            if visited[node]:
                continue
            visited[node] = True
            cluster.append(node)
            
            # Add neighbors to stack
            neighbors = np.where(connectivity_graph[node])[0]
            for neighbor in neighbors:
                if not visited[neighbor]:
                    stack.append(neighbor)
        
        if cluster:
            clusters.append(cluster)
    
    return clusters


def extract_component_voxels_at_threshold(
    component_xyz: np.ndarray,
    ligand_map: MapVolume,
    threshold: float,
    padding_A: float = 6.0
) -> np.ndarray:
    """Extract voxel coordinates for a component at a given threshold.
    
    Args:
        component_xyz: (N, 3) pseudoatom coordinates in Å
        ligand_map: MapVolume with ligand density
        threshold: Density threshold
        padding_A: Padding around bbox
        
    Returns:
        (M, 3) array of voxel indices (z, y, x) above threshold
    """
    apix = ligand_map.apix
    origin = ligand_map.origin_xyzA
    
    # Convert to voxel coordinates
    xyz_vox = (component_xyz - origin) / apix
    zyx_vox = xyz_vox[:, [2, 1, 0]]
    
    # Get bounding box
    z_min, y_min, x_min = zyx_vox.min(axis=0) - padding_A / apix
    z_max, y_max, x_max = zyx_vox.max(axis=0) + padding_A / apix
    
    z_min, y_min, x_min = int(np.floor(z_min)), int(np.floor(y_min)), int(np.floor(x_min))
    z_max, y_max, x_max = int(np.ceil(z_max)), int(np.ceil(y_max)), int(np.ceil(x_max))
    
    # Clip to bounds
    z_min = max(0, z_min)
    y_min = max(0, y_min)
    x_min = max(0, x_min)
    z_max = min(ligand_map.data_zyx.shape[0], z_max)
    y_max = min(ligand_map.data_zyx.shape[1], y_max)
    x_max = min(ligand_map.data_zyx.shape[2], x_max)
    
    if z_max <= z_min or y_max <= y_min or x_max <= x_min:
        return np.zeros((0, 3), dtype=int)
    
    # Extract subvolume and threshold
    subvol = ligand_map.data_zyx[z_min:z_max, y_min:y_max, x_min:x_max]
    mask = subvol >= threshold
    
    # Get voxel coordinates
    voxels_local = np.array(np.where(mask)).T  # (z, y, x) in local coordinates
    if len(voxels_local) == 0:
        return np.zeros((0, 3), dtype=int)
    
    # Convert to global coordinates
    voxels_global = voxels_local + np.array([z_min, y_min, x_min])
    
    return voxels_global

