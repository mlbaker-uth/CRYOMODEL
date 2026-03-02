# crymodel/nucleotide/basehunter.py
"""BaseHunter: Compare and sort nucleotide density at near-atomic resolutions."""
from __future__ import annotations
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import mrcfile
from scipy.stats import wasserstein_distance

from ..io.mrc import read_map, write_map, MapVolume


def read_mrc_file(filepath: str) -> Tuple[Optional[np.ndarray], Optional[mrcfile.MrcFile]]:
    """Read MRC file."""
    try:
        with mrcfile.open(filepath, permissive=True) as mrc:
            print(f"Reading file: {filepath}")
            print(f"Shape: {mrc.data.shape}")
            return np.copy(mrc.data), mrc
    except Exception as e:
        print(f"Error reading MRC file: {e}")
        return None, None


def write_mrc_file(filepath: str, volume: np.ndarray, original_mrc: mrcfile.MrcFile, origin: Optional[Tuple[int, int, int]] = None) -> None:
    """Write MRC file."""
    try:
        if original_mrc is None:
            raise ValueError("Original MRC file is required for writing.")
        with mrcfile.new(filepath, overwrite=True) as mrc:
            mrc.set_data(volume.astype(np.float32))
            mrc.header.nx = original_mrc.header.nx
            mrc.header.ny = original_mrc.header.ny
            mrc.header.nz = original_mrc.header.nz
            mrc.header.mode = original_mrc.header.mode
            mrc.header.dmin = np.min(volume)
            mrc.header.dmax = np.max(volume)
            mrc.header.dmean = np.mean(volume)
            
            if origin:
                mrc.header.nxstart, mrc.header.nystart, mrc.header.nzstart = origin
            else:
                mrc.header.nxstart = mrc.header.nystart = mrc.header.nzstart = 0
            
            mrc.voxel_size = original_mrc.voxel_size
            print(f"Saved volume to: {filepath}")
    except Exception as e:
        print(f"Error writing MRC file: {e}")


def threshold_volume(volume: np.ndarray, threshold: float) -> np.ndarray:
    """Threshold volume."""
    return np.where(volume >= threshold, volume, 0)


def generate_point_cloud(volume: np.ndarray) -> np.ndarray:
    """Generate a point cloud from all non-zero points in the volume."""
    coords = np.array(np.nonzero(volume)).T  # Coordinates of all non-zero points
    return coords


def compare_point_clouds_emd(cloud1: np.ndarray, cloud2: np.ndarray) -> float:
    """Compare point clouds using Earth Mover's Distance (EMD)."""
    cloud1_flat = cloud1.flatten()
    cloud2_flat = cloud2.flatten()
    return wasserstein_distance(cloud1_flat, cloud2_flat)


def evaluate_group_consistency(group: List[str], point_clouds: Dict[str, np.ndarray]) -> float:
    """Evaluate group consistency using average pairwise Earth Mover's Distance (EMD)."""
    if len(group) <= 1:
        return 0  # Cannot evaluate consistency with a single volume
    
    similarities = [
        compare_point_clouds_emd(point_clouds[v1], point_clouds[v2])
        for i, v1 in enumerate(group)
        for v2 in group[i + 1:]
    ]
    return np.mean(similarities)


def sort_volumes_into_groups(volume_pairs: List[Tuple[str, str]], point_clouds: Dict[str, np.ndarray]) -> Tuple[List[str], List[str]]:
    """Sort volumes into groups."""
    group1, group2 = [], []
    
    for vol1, vol2 in volume_pairs:
        # Compute similarity to each group
        group1_similarity_vol1 = (
            np.mean([compare_point_clouds_emd(point_clouds[vol1], point_clouds[v]) for v in group1]) if group1 else float('inf')
        )
        group2_similarity_vol1 = (
            np.mean([compare_point_clouds_emd(point_clouds[vol1], point_clouds[v]) for v in group2]) if group2 else float('inf')
        )
        
        group1_similarity_vol2 = (
            np.mean([compare_point_clouds_emd(point_clouds[vol2], point_clouds[v]) for v in group1]) if group1 else float('inf')
        )
        group2_similarity_vol2 = (
            np.mean([compare_point_clouds_emd(point_clouds[vol2], point_clouds[v]) for v in group2]) if group2 else float('inf')
        )
        
        # Determine assignment
        if len(group1) < len(group2):
            group1.append(vol1)
            group2.append(vol2)
        elif len(group2) < len(group1):
            group1.append(vol2)
            group2.append(vol1)
        else:
            # Choose based on similarity
            if group1_similarity_vol1 + group2_similarity_vol2 < group2_similarity_vol1 + group1_similarity_vol2:
                group1.append(vol1)
                group2.append(vol2)
            elif group2_similarity_vol1 + group1_similarity_vol2 < group1_similarity_vol1 + group2_similarity_vol2:
                group2.append(vol1)
                group1.append(vol2)
            else:
                # Fallback: Random assignment to maintain balance
                if random.random() < 0.5:
                    group1.append(vol1)
                    group2.append(vol2)
                else:
                    group1.append(vol2)
                    group2.append(vol1)
    
    # Ensure both groups are balanced
    if len(group1) != len(group2):
        raise ValueError("Groups are not balanced after sorting.")
    
    return group1, group2


def monte_carlo_refine(
    groups: Tuple[List[str], List[str]],
    point_clouds: Dict[str, np.ndarray],
    volume_pairs: List[Tuple[str, str]],
    max_iterations: int = 1000,
    min_stability: int = 100000,
    min_improvement: float = 1e-4,
    exploration_chance: float = 0.1,
    force_iterations: int = 250,
) -> Tuple[List[str], List[str]]:
    """Refine group assignments using Monte Carlo-style swapping."""
    group1, group2 = groups
    group1, group2 = group1[:], group2[:]
    stability_history = []
    consistency_scores = []
    
    for iteration in range(max_iterations):
        print(f"Iteration {iteration + 1}: Starting refinement.")
        
        # Randomly pick a pair to consider for swapping
        pair = random.choice(volume_pairs)
        vol1, vol2 = pair
        
        if vol1 in group1 and vol2 in group2:
            current_group1 = group1[:]
            current_group2 = group2[:]
            
            # Perform the swap
            current_group1.remove(vol1)
            current_group2.remove(vol2)
            current_group1.append(vol2)
            current_group2.append(vol1)
            
            # Compute new group scores
            score1_original = evaluate_group_consistency(group1, point_clouds)
            score2_original = evaluate_group_consistency(group2, point_clouds)
            
            score1_swapped = evaluate_group_consistency(current_group1, point_clouds)
            score2_swapped = evaluate_group_consistency(current_group2, point_clouds)
            
            print(
                f"Swap evaluation: Original score = {score1_original + score2_original}, "
                f"Swapped score = {score1_swapped + score2_swapped}"
            )
            
            # Decide whether to accept the swap
            if score1_swapped + score2_swapped < score1_original + score2_original:
                # Accept if the swap improves the score
                group1, group2 = current_group1, current_group2
                print("Swap accepted (improved score).")
            elif random.random() < exploration_chance:
                # Accept a suboptimal swap with a small probability
                group1, group2 = current_group1, current_group2
                print("Swap accepted (exploration).")
            else:
                print("Swap rejected.")
        
        # Track stability and consistency
        stability_history.append((tuple(sorted(group1)), tuple(sorted(group2))))
        consistency_scores.append(evaluate_group_consistency(group1, point_clouds) + evaluate_group_consistency(group2, point_clouds))
        
        # Ensure a minimum number of iterations before checking convergence
        if iteration + 1 < force_iterations:
            continue
        
        # Check convergence: stability
        if len(stability_history) > min_stability and len(set(stability_history[-min_stability:])) == 1:
            print(f"Convergence achieved at iteration {iteration + 1} (stability).")
            break
        
        # Check convergence: minimal improvement
        if len(consistency_scores) > 1 and abs(consistency_scores[-1] - consistency_scores[-2]) < min_improvement:
            print(f"Convergence achieved at iteration {iteration + 1} (minimal improvement).")
            break
    
    print("Final group consistency scores:")
    print(f"Group 1 score: {evaluate_group_consistency(group1, point_clouds)}")
    print(f"Group 2 score: {evaluate_group_consistency(group2, point_clouds)}")
    
    return group1, group2


def compute_average_volume(group: List[str], volumes: Dict[str, np.ndarray], output_path: str, reference_mrc: mrcfile.MrcFile) -> None:
    """Compute and save the average volume for a group."""
    if reference_mrc is None:
        raise ValueError("Reference MRC file is required to save average volume.")
    group_volumes = [volumes[vol_path] for vol_path in group]
    if not group_volumes:
        print(f"No volumes in group {output_path}.")
        return
    
    avg_volume = np.mean(group_volumes, axis=0)
    write_mrc_file(output_path, avg_volume, reference_mrc)


def calculate_normalized_cross_correlation(volume1: np.ndarray, volume2: np.ndarray) -> float:
    """Calculate normalized cross-correlation (NCC) between two volumes."""
    if volume1.shape != volume2.shape:
        raise ValueError("Volumes must have the same shape for NCC calculation.")
    
    mean1 = np.mean(volume1)
    mean2 = np.mean(volume2)
    
    numerator = np.sum((volume1 - mean1) * (volume2 - mean2))
    denominator = np.sqrt(np.sum((volume1 - mean1) ** 2) * np.sum((volume2 - mean2) ** 2))
    
    if denominator == 0:
        return 0  # Avoid division by zero
    
    return numerator / denominator


def calculate_group_ncc(group: List[str], volumes: Dict[str, np.ndarray]) -> Optional[float]:
    """Calculate average normalized cross-correlation (NCC) for all pairs in a group."""
    if len(group) <= 1:
        return None  # Cannot calculate NCC for a single volume
    
    ncc_values = [
        calculate_normalized_cross_correlation(volumes[v1], volumes[v2])
        for i, v1 in enumerate(group)
        for v2 in group[i + 1:]
    ]
    return np.mean(ncc_values)


def write_group_with_ncc(group: List[str], volumes: Dict[str, np.ndarray], filepath: str) -> None:
    """Write group members and their pairwise NCC values to a text file."""
    with open(filepath, "w") as f:
        # Write group members
        f.write("Group Members:\n")
        for volume in group:
            f.write(f"{volume}\n")
        
        # Calculate pairwise NCC
        if len(group) > 1:
            f.write("\nPairwise Normalized Cross-Correlation (NCC):\n")
            ncc_table = []
            for i, v1 in enumerate(group):
                for v2 in group[i + 1:]:
                    ncc_value = calculate_normalized_cross_correlation(volumes[v1], volumes[v2])
                    ncc_table.append((v1, v2, ncc_value))
                    f.write(f"{v1} vs {v2}: {ncc_value:.4f}\n")
            
            # Optionally, write as a structured table
            f.write("\nStructured NCC Table:\n")
            f.write("Volume1\tVolume2\tNCC\n")
            for row in ncc_table:
                f.write(f"{row[0]}\t{row[1]}\t{row[2]:.4f}\n")
        else:
            f.write("\nNo pairwise NCC calculations (single volume in group).\n")

