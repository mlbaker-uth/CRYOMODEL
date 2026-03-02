# crymodel/pathalker/averaging.py
"""Averaging multiple pathwalking runs."""
from __future__ import annotations
import numpy as np
from pathlib import Path
from scipy.spatial.distance import cdist
from typing import List, Optional


def _read_pdb_coordinates(path: str) -> np.ndarray:
    """Read coordinates from PDB file."""
    coords = []
    with open(path, 'r') as f:
        for line in f:
            if line.startswith("ATOM"):
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
                coords.append([x, y, z])
    return np.array(coords, dtype=np.float32) if coords else np.zeros((0, 3), dtype=np.float32)


def average_paths(
    path_files: List[Path],
    reference_coordinates: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Average multiple pathwalking runs.
    
    Aligns paths by matching coordinates and computes average positions.
    
    Args:
        path_files: List of PDB file paths containing path coordinates
        reference_coordinates: Optional reference coordinates to align to
            (if None, uses first path as reference)
        
    Returns:
        Tuple of (averaged coordinates, probabilities)
    """
    if not path_files:
        raise ValueError("No path files provided")
    
    # Load all paths
    all_paths = []
    for path_file in path_files:
        coords = _read_pdb_coordinates(str(path_file))
        if len(coords) > 0:
            all_paths.append(coords)
    
    if not all_paths:
        raise ValueError("No valid paths found in files")
    
    # Use first path as reference if not provided
    if reference_coordinates is None:
        reference_coordinates = all_paths[0]
    
    n_ref = len(reference_coordinates)
    
    # Align each path to reference and compute average
    averaged_coords = np.zeros_like(reference_coordinates)
    
    for path_coords in all_paths:
        # Match each reference point to closest point in path
        distances = cdist(reference_coordinates, path_coords)
        matched_indices = distances.argmin(axis=1)
        
        # Add matched coordinates
        for i, matched_idx in enumerate(matched_indices):
            averaged_coords[i] += path_coords[matched_idx]
    
    # Average
    averaged_coords /= len(all_paths)
    
    # Compute probabilities (how often each reference position is visited)
    probabilities = np.zeros(n_ref, dtype=np.float32)
    for path_coords in all_paths:
        distances = cdist(reference_coordinates, path_coords)
        matched_indices = distances.argmin(axis=1)
        
        # Count visits to each reference position
        for matched_idx in matched_indices:
            if matched_idx < n_ref:
                probabilities[matched_idx] += 1.0
    
    probabilities /= len(all_paths)
    
    return averaged_coords.astype(np.float32), probabilities


def compute_path_probabilities(
    path_files: List[Path],
    reference_coordinates: np.ndarray,
) -> np.ndarray:
    """Compute probability of each position in reference path.
    
    Args:
        path_files: List of PDB file paths containing path coordinates
        reference_coordinates: Reference path coordinates
        
    Returns:
        (N,) array of probabilities for each reference position
    """
    n_ref = len(reference_coordinates)
    visit_counts = np.zeros(n_ref, dtype=int)
    
    for path_file in path_files:
        path_coords = _read_pdb_coordinates(str(path_file))
        if len(path_coords) == 0:
            continue
        
        # Match reference to path
        distances = cdist(reference_coordinates, path_coords)
        
        # For each reference position, find if it's visited in this path
        for i in range(n_ref):
            # Check if any path coordinate is close to this reference position
            min_dist = distances[i].min()
            if min_dist < 2.0:  # Within 2Å
                visit_counts[i] += 1
    
    probabilities = visit_counts.astype(np.float32) / len(path_files)
    return probabilities

