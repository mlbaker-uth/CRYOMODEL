# crymodel/pathalker/path_evaluation.py
"""Path evaluation and geometry analysis for pathwalking."""
from __future__ import annotations
import numpy as np
from typing import Optional


TARGET_CA_CA_DISTANCE = 3.8  # Å
MIN_CA_CA_DISTANCE = 2.8  # Å
MAX_CA_CA_DISTANCE = 4.8  # Å


def evaluate_path(
    path_coordinates: np.ndarray,
    verbose: bool = True,
) -> dict[str, float | list[int]]:
    """Evaluate path geometry (C-alpha distances).
    
    Args:
        path_coordinates: (N, 3) array of path coordinates in Å (ordered along path)
        verbose: Print evaluation results
        
    Returns:
        Dictionary with evaluation metrics
    """
    n = len(path_coordinates)
    if n < 2:
        return {
            "path_length": 0.0,
            "expected_length": 0.0,
            "mean_ca_ca_distance": 0.0,
            "std_ca_ca_distance": 0.0,
            "too_short_indices": [],
            "too_long_indices": [],
        }
    
    # Compute C-alpha to C-alpha distances
    ca_ca_distances = []
    for i in range(n - 1):
        dist = np.linalg.norm(path_coordinates[i + 1] - path_coordinates[i])
        ca_ca_distances.append(dist)
    
    ca_ca_distances = np.array(ca_ca_distances)
    
    # Statistics
    path_length = float(ca_ca_distances.sum())
    expected_length = (n - 1) * TARGET_CA_CA_DISTANCE
    mean_ca_ca = float(ca_ca_distances.mean())
    std_ca_ca = float(ca_ca_distances.std())
    
    # Find problematic distances
    too_short = np.where(ca_ca_distances < MIN_CA_CA_DISTANCE)[0].tolist()
    too_long = np.where(ca_ca_distances > MAX_CA_CA_DISTANCE)[0].tolist()
    
    results = {
        "path_length": path_length,
        "expected_length": expected_length,
        "mean_ca_ca_distance": mean_ca_ca,
        "std_ca_ca_distance": std_ca_ca,
        "too_short_indices": [int(i) for i in too_short],
        "too_long_indices": [int(i) for i in too_long],
    }
    
    if verbose:
        print(f"  Path evaluation:")
        print(f"    Path length: {path_length:.2f} Å")
        print(f"    Expected length: {expected_length:.2f} Å")
        print(f"    Mean C-alpha distance: {mean_ca_ca:.2f} Å (target: {TARGET_CA_CA_DISTANCE} Å)")
        print(f"    Std C-alpha distance: {std_ca_ca:.2f} Å")
        if too_short:
            print(f"    Too short distances (< {MIN_CA_CA_DISTANCE} Å): {len(too_short)} at indices {too_short[:10]}{'...' if len(too_short) > 10 else ''}")
        if too_long:
            print(f"    Too long distances (> {MAX_CA_CA_DISTANCE} Å): {len(too_long)} at indices {too_long[:10]}{'...' if len(too_long) > 10 else ''}")
    
    return results


def calculate_path_statistics(path_coordinates: np.ndarray) -> dict[str, float]:
    """Calculate additional path statistics.
    
    Args:
        path_coordinates: (N, 3) array of path coordinates in Å
        
    Returns:
        Dictionary with path statistics
    """
    if len(path_coordinates) < 2:
        return {}
    
    # Two-step distances (i to i+2)
    two_step_distances = []
    for i in range(len(path_coordinates) - 2):
        dist = np.linalg.norm(path_coordinates[i + 2] - path_coordinates[i])
        two_step_distances.append(dist)
    
    # Neighboring bond distances (sum of two consecutive bonds)
    ca_ca_distances = []
    for i in range(len(path_coordinates) - 1):
        dist = np.linalg.norm(path_coordinates[i + 1] - path_coordinates[i])
        ca_ca_distances.append(dist)
    
    neighboring_sums = []
    for i in range(len(ca_ca_distances) - 1):
        neighboring_sums.append(ca_ca_distances[i] + ca_ca_distances[i + 1])
    
    return {
        "mean_two_step_distance": float(np.mean(two_step_distances)) if two_step_distances else 0.0,
        "std_two_step_distance": float(np.std(two_step_distances)) if two_step_distances else 0.0,
        "mean_neighboring_sum": float(np.mean(neighboring_sums)) if neighboring_sums else 0.0,
        "std_neighboring_sum": float(np.std(neighboring_sums)) if neighboring_sums else 0.0,
    }

