# crymodel/nucleotide/classification.py
"""Template-based classification for BaseHunter."""
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.ndimage import maximum_filter, gaussian_filter
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from ..io.mrc import read_map, MapVolume
from .templates import TemplateLibrary


def align_volume_to_template(
    volume: np.ndarray,
    template: np.ndarray,
    method: str = "ncc",
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Align volume to template using rigid-body transformation.
    
    Args:
        volume: Volume to align (z, y, x)
        template: Template volume (z, y, x)
        method: Alignment method ("ncc" for normalized cross-correlation)
        
    Returns:
        (aligned_volume, transformation_matrix, correlation_score)
    """
    # For now, use FFT-based cross-correlation for translation
    # Then refine with small rotations if needed
    
    # Ensure same shape (pad if needed)
    if volume.shape != template.shape:
        # Pad smaller volume to match larger
        max_shape = tuple(max(s1, s2) for s1, s2 in zip(volume.shape, template.shape))
        volume_padded = np.zeros(max_shape, dtype=volume.dtype)
        template_padded = np.zeros(max_shape, dtype=template.dtype)
        
        v0, v1, v2 = volume.shape
        t0, t1, t2 = template.shape
        
        volume_padded[:v0, :v1, :v2] = volume
        template_padded[:t0, :t1, :t2] = template
        
        volume = volume_padded
        template = template_padded
    
    # Normalize
    volume_norm = volume - np.mean(volume)
    volume_std = np.std(volume_norm)
    if volume_std > 1e-6:
        volume_norm /= volume_std
    
    template_norm = template - np.mean(template)
    template_std = np.std(template_norm)
    if template_std > 1e-6:
        template_norm /= template_std
    
    # FFT-based cross-correlation
    volume_fft = np.fft.fftn(volume_norm)
    template_fft = np.fft.fftn(template_norm)
    
    corr_fft = np.conj(volume_fft) * template_fft
    corr = np.fft.ifftn(corr_fft).real
    
    # Find peak (best translation)
    peak_idx = np.unravel_index(np.argmax(corr), corr.shape)
    peak_corr = corr[peak_idx]
    
    # Apply translation (circular shift)
    aligned = np.roll(volume, 
                     (peak_idx[0] - volume.shape[0]//2,
                      peak_idx[1] - volume.shape[1]//2,
                      peak_idx[2] - volume.shape[2]//2),
                     axis=(0, 1, 2))
    
    # Identity transformation matrix (for now - could add rotation refinement)
    transform = np.eye(4, dtype=np.float32)
    
    return aligned, transform, float(peak_corr)


def extract_features(volume: np.ndarray) -> Dict[str, float]:
    """
    Extract features that distinguish purine (2-ring) from pyrimidine (1-ring).
    
    Args:
        volume: Density volume (z, y, x)
        
    Returns:
        Dictionary with features
    """
    from scipy.ndimage import label
    
    # Threshold to get base region
    threshold = np.mean(volume) + np.std(volume)
    binary = volume > threshold
    
    # Connected components
    labeled, n_components = label(binary)
    
    # Size of largest component (base size)
    component_sizes = []
    for i in range(1, n_components + 1):
        size = np.sum(labeled == i)
        component_sizes.append(size)
    
    max_component_size = max(component_sizes) if component_sizes else 0
    
    # Volume statistics
    total_volume = np.sum(binary)
    mean_density = np.mean(volume[binary]) if np.any(binary) else 0.0
    max_density = np.max(volume) if volume.size > 0 else 0.0
    
    # Compactness (volume / surface area approximation)
    # For 2-ring purine, expect larger, less compact
    # For 1-ring pyrimidine, expect smaller, more compact
    # Approximate surface area as 6 * (volume^(2/3)) for cube-like shapes
    if total_volume > 0:
        surface_area_approx = 6 * (total_volume ** (2.0/3.0))
        compactness = max_component_size / (surface_area_approx + 1e-6)
    else:
        compactness = 0.0
    
    # Density distribution (skewness, kurtosis)
    density_values = volume[binary] if np.any(binary) else np.array([0.0])
    if len(density_values) > 0:
        from scipy.stats import skew, kurtosis
        density_skew = float(skew(density_values))
        density_kurtosis = float(kurtosis(density_values))
    else:
        density_skew = 0.0
        density_kurtosis = 0.0
    
    return {
        "max_component_size": float(max_component_size),
        "total_volume": float(total_volume),
        "mean_density": float(mean_density),
        "max_density": float(max_density),
        "n_components": int(n_components),
        "compactness": float(compactness),
        "density_skew": density_skew,
        "density_kurtosis": density_kurtosis,
    }


def feature_similarity(features1: Dict[str, float], features2: Dict[str, float]) -> float:
    """Compute similarity between feature vectors (0-1, higher = more similar)."""
    # Normalize features to [0, 1] range for comparison
    # Use cosine similarity or normalized Euclidean distance
    
    keys = set(features1.keys()) & set(features2.keys())
    if not keys:
        return 0.0
    
    vec1 = np.array([features1[k] for k in keys])
    vec2 = np.array([features2[k] for k in keys])
    
    # Normalize
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 < 1e-6 or norm2 < 1e-6:
        return 0.0
    
    vec1_norm = vec1 / norm1
    vec2_norm = vec2 / norm2
    
    # Cosine similarity
    similarity = np.dot(vec1_norm, vec2_norm)
    
    # Convert to [0, 1] range (cosine similarity is [-1, 1])
    return float((similarity + 1.0) / 2.0)


def compute_emd_similarity(
    volume1: np.ndarray,
    volume2: np.ndarray,
) -> float:
    """
    Compute Earth Mover's Distance (EMD) between two volumes.
    
    EMD is useful for noisy/anisotropic maps where NCC may be unreliable.
    Lower EMD = more similar.
    
    Args:
        volume1: First volume (z, y, x)
        volume2: Second volume (z, y, x)
        
    Returns:
        EMD value (lower = more similar), normalized to [0, 1] range
    """
    from scipy.stats import wasserstein_distance
    
    # Flatten volumes for EMD computation
    # EMD works on 1D distributions, so we flatten the volumes
    vol1_flat = volume1.flatten()
    vol2_flat = volume2.flatten()
    
    # Normalize to probability distributions
    vol1_sum = np.sum(vol1_flat)
    vol2_sum = np.sum(vol2_flat)
    
    if vol1_sum < 1e-6 or vol2_sum < 1e-6:
        return 1.0  # Maximum distance if one volume is empty
    
    vol1_norm = vol1_flat / vol1_sum
    vol2_norm = vol2_flat / vol2_sum
    
    # Compute EMD (Wasserstein distance)
    # For 1D distributions, wasserstein_distance computes EMD
    emd = wasserstein_distance(vol1_norm, vol2_norm)
    
    # Normalize EMD to [0, 1] range (approximate - EMD can vary)
    # Use a heuristic: divide by max possible distance (rough estimate)
    # For normalized distributions, max EMD is roughly 2.0
    emd_normalized = min(1.0, emd / 2.0)
    
    # Convert to similarity (1 - normalized EMD, so higher = more similar)
    similarity = 1.0 - emd_normalized
    
    return float(similarity)


def classify_with_templates(
    volume: np.ndarray,
    purine_template: np.ndarray,
    pyrimidine_template: np.ndarray,
    alignment_threshold: float = 0.3,
    use_emd: bool = True,
    emd_weight: float = 0.2,
) -> Tuple[str, float, float, float]:
    """
    Classify volume as purine-like, pyrimidine-like, or unclassified.
    
    Args:
        volume: Volume to classify (z, y, x)
        purine_template: Purine template (z, y, x)
        pyrimidine_template: Pyrimidine template (z, y, x)
        alignment_threshold: Minimum correlation to classify (default 0.3)
        use_emd: If True, include EMD in scoring (default True)
        emd_weight: Weight for EMD in combined score (default 0.2)
        
    Returns:
        (class, purine_score, pyrimidine_score, confidence)
        class: "purine", "pyrimidine", or "unclassified"
        purine_score: Alignment score with purine template (0-1)
        pyrimidine_score: Alignment score with pyrimidine template (0-1)
        confidence: Confidence in classification (0-1, higher = more confident)
    """
    # Align to both templates
    aligned_purine, _, purine_corr = align_volume_to_template(volume, purine_template)
    aligned_pyrimidine, _, pyrimidine_corr = align_volume_to_template(volume, pyrimidine_template)
    
    # Extract features
    purine_features = extract_features(aligned_purine)
    pyrimidine_features = extract_features(aligned_pyrimidine)
    template_purine_features = extract_features(purine_template)
    template_pyrimidine_features = extract_features(pyrimidine_template)
    
    # Feature similarity
    purine_feat_sim = feature_similarity(purine_features, template_purine_features)
    pyrimidine_feat_sim = feature_similarity(pyrimidine_features, template_pyrimidine_features)
    
    # EMD similarity (if enabled) - useful for noisy/anisotropic maps
    if use_emd:
        purine_emd_sim = compute_emd_similarity(aligned_purine, purine_template)
        pyrimidine_emd_sim = compute_emd_similarity(aligned_pyrimidine, pyrimidine_template)
        
        # Combined score: correlation + features + EMD
        # Adjust weights so they sum to 1.0
        corr_weight = 0.6 * (1.0 - emd_weight)
        feat_weight = 0.4 * (1.0 - emd_weight)
        
        purine_score = (corr_weight * purine_corr + 
                       feat_weight * purine_feat_sim + 
                       emd_weight * purine_emd_sim)
        pyrimidine_score = (corr_weight * pyrimidine_corr + 
                           feat_weight * pyrimidine_feat_sim + 
                           emd_weight * pyrimidine_emd_sim)
    else:
        # Original scoring without EMD
        purine_score = 0.6 * purine_corr + 0.4 * purine_feat_sim
        pyrimidine_score = 0.6 * pyrimidine_corr + 0.4 * pyrimidine_feat_sim
    
    # Normalize scores to [0, 1] range
    purine_score = max(0.0, min(1.0, purine_score))
    pyrimidine_score = max(0.0, min(1.0, pyrimidine_score))
    
    # Classification logic
    score_diff = abs(purine_score - pyrimidine_score)
    max_score = max(purine_score, pyrimidine_score)
    
    if max_score < alignment_threshold:
        # Too low confidence
        return ("unclassified", purine_score, pyrimidine_score, 1.0 - max_score)
    elif score_diff < 0.1:  # Too close to call
        return ("unclassified", purine_score, pyrimidine_score, score_diff)
    elif purine_score > pyrimidine_score:
        confidence = score_diff / (purine_score + 1e-6)
        return ("purine", purine_score, pyrimidine_score, confidence)
    else:
        confidence = score_diff / (pyrimidine_score + 1e-6)
        return ("pyrimidine", purine_score, pyrimidine_score, confidence)


def enforce_base_pair_constraint(
    classifications: Dict[str, Tuple[str, float, float, float]],
    base_pairs: List[Tuple[str, str]],
) -> Dict[str, str]:
    """
    Enforce constraint: each base pair must have 1 purine + 1 pyrimidine.
    
    If one is unclassified, assign based on the other.
    If both are unclassified, use scores to assign.
    If both are same class, resolve by confidence.
    
    Args:
        classifications: Dict mapping volume_path -> (class, purine_score, pyrimidine_score, confidence)
        base_pairs: List of (volume1_path, volume2_path) tuples
        
    Returns:
        Dict mapping volume_path -> resolved_class ("purine" or "pyrimidine")
    """
    resolved = {}
    
    for vol1, vol2 in base_pairs:
        if vol1 not in classifications or vol2 not in classifications:
            continue
        
        class1, score1_pur, score1_pyr, conf1 = classifications[vol1]
        class2, score2_pur, score2_pyr, conf2 = classifications[vol2]
        
        # Both classified - check constraint
        if class1 != "unclassified" and class2 != "unclassified":
            if class1 == class2:
                # Conflict: both same class - resolve by confidence
                if conf1 > conf2:
                    # Reclassify vol2
                    class2 = "pyrimidine" if class1 == "purine" else "purine"
                else:
                    # Reclassify vol1
                    class1 = "pyrimidine" if class2 == "purine" else "purine"
        
        # One unclassified - assign based on other
        elif class1 == "unclassified" and class2 != "unclassified":
            class1 = "pyrimidine" if class2 == "purine" else "purine"
        elif class2 == "unclassified" and class1 != "unclassified":
            class2 = "pyrimidine" if class1 == "purine" else "purine"
        
        # Both unclassified - use scores to assign
        elif class1 == "unclassified" and class2 == "unclassified":
            # Use combined scores to determine assignment
            vol1_pur_score = score1_pur
            vol1_pyr_score = score1_pyr
            vol2_pur_score = score2_pur
            vol2_pyr_score = score2_pyr
            
            # Try both assignments and pick best
            # Assignment 1: vol1=purine, vol2=pyrimidine
            score1 = vol1_pur_score + vol2_pyr_score
            # Assignment 2: vol1=pyrimidine, vol2=purine
            score2 = vol1_pyr_score + vol2_pur_score
            
            if score1 > score2:
                class1 = "purine"
                class2 = "pyrimidine"
            else:
                class1 = "pyrimidine"
                class2 = "purine"
        
        resolved[vol1] = class1
        resolved[vol2] = class2
    
    return resolved


def calculate_normalized_cross_correlation(volume1: np.ndarray, volume2: np.ndarray) -> float:
    """Calculate normalized cross-correlation (NCC) between two volumes."""
    if volume1.shape != volume2.shape:
        raise ValueError("Volumes must have the same shape for NCC calculation.")
    
    mean1 = np.mean(volume1)
    mean2 = np.mean(volume2)
    
    numerator = np.sum((volume1 - mean1) * (volume2 - mean2))
    denominator = np.sqrt(np.sum((volume1 - mean1) ** 2) * np.sum((volume2 - mean2) ** 2))
    
    if denominator == 0:
        return 0.0
    
    return float(numerator / denominator)


def compute_class_statistics(
    volumes: Dict[str, np.ndarray],
    classifications: Dict[str, str],
) -> Dict[str, float]:
    """
    Compute inter- and intra-class statistics.
    
    Args:
        volumes: Dict mapping volume_path -> volume array
        classifications: Dict mapping volume_path -> class ("purine" or "pyrimidine")
        
    Returns:
        Dictionary with statistics
    """
    
    purine_volumes = [v for v, c in classifications.items() if c == "purine"]
    pyrimidine_volumes = [v for v, c in classifications.items() if c == "pyrimidine"]
    
    # Intra-class: within purine group
    purine_intra = []
    for i, v1_path in enumerate(purine_volumes):
        for v2_path in purine_volumes[i+1:]:
            ncc = calculate_normalized_cross_correlation(volumes[v1_path], volumes[v2_path])
            purine_intra.append(ncc)
    
    # Intra-class: within pyrimidine group
    pyrimidine_intra = []
    for i, v1_path in enumerate(pyrimidine_volumes):
        for v2_path in pyrimidine_volumes[i+1:]:
            ncc = calculate_normalized_cross_correlation(volumes[v1_path], volumes[v2_path])
            pyrimidine_intra.append(ncc)
    
    # Inter-class: between purine and pyrimidine
    inter_class = []
    for v1_path in purine_volumes:
        for v2_path in pyrimidine_volumes:
            ncc = calculate_normalized_cross_correlation(volumes[v1_path], volumes[v2_path])
            inter_class.append(ncc)
    
    purine_intra_mean = np.mean(purine_intra) if purine_intra else 0.0
    purine_intra_std = np.std(purine_intra) if purine_intra else 0.0
    pyrimidine_intra_mean = np.mean(pyrimidine_intra) if pyrimidine_intra else 0.0
    pyrimidine_intra_std = np.std(pyrimidine_intra) if pyrimidine_intra else 0.0
    inter_class_mean = np.mean(inter_class) if inter_class else 0.0
    inter_class_std = np.std(inter_class) if inter_class else 0.0
    
    # Separation score: higher = better separation between classes
    separation_score = (purine_intra_mean + pyrimidine_intra_mean) - 2 * inter_class_mean
    
    return {
        "purine_intra_mean": float(purine_intra_mean),
        "purine_intra_std": float(purine_intra_std),
        "purine_intra_n": len(purine_intra),
        "pyrimidine_intra_mean": float(pyrimidine_intra_mean),
        "pyrimidine_intra_std": float(pyrimidine_intra_std),
        "pyrimidine_intra_n": len(pyrimidine_intra),
        "inter_class_mean": float(inter_class_mean),
        "inter_class_std": float(inter_class_std),
        "inter_class_n": len(inter_class),
        "separation_score": float(separation_score),
    }


def bootstrap_classification(
    volumes: Dict[str, np.ndarray],
    purine_template: np.ndarray,
    pyrimidine_template: np.ndarray,
    base_pairs: List[Tuple[str, str]],
    n_bootstrap: int = 100,
    alignment_threshold: float = 0.3,
    use_emd: bool = True,
    emd_weight: float = 0.2,
) -> Dict[str, Dict[str, float]]:
    """
    Bootstrap classification to estimate likelihoods.
    
    Args:
        volumes: Dict mapping volume_path -> volume array
        purine_template: Purine template
        pyrimidine_template: Pyrimidine template
        base_pairs: List of (volume1_path, volume2_path) tuples
        n_bootstrap: Number of bootstrap iterations
        alignment_threshold: Minimum correlation threshold
        
    Returns:
        Dict mapping volume_path -> {"purine": likelihood, "pyrimidine": likelihood, "unclassified": likelihood}
    """
    from collections import defaultdict
    
    likelihoods = defaultdict(lambda: {"purine": 0.0, "pyrimidine": 0.0, "unclassified": 0.0})
    rng = np.random.default_rng(42)  # Deterministic seed
    
    for bootstrap_iter in range(n_bootstrap):
        # Add small random noise to volumes
        sampled_volumes = {}
        for vol_name, vol_data in volumes.items():
            noise_std = 0.05 * np.std(vol_data)
            noise = rng.normal(0, noise_std, vol_data.shape)
            sampled_volumes[vol_name] = np.clip(vol_data + noise, 0, None)
        
        # Classify with templates
        classifications = {}
        for vol_name, vol_data in sampled_volumes.items():
            class_name, _, _, _ = classify_with_templates(
                vol_data, purine_template, pyrimidine_template, 
                alignment_threshold, use_emd=use_emd, emd_weight=emd_weight
            )
            classifications[vol_name] = class_name
        
        # Enforce constraints
        resolved = enforce_base_pair_constraint(classifications, base_pairs)
        
        # Count assignments
        for vol_name, class_name in resolved.items():
            likelihoods[vol_name][class_name] += 1.0 / n_bootstrap
    
    return dict(likelihoods)

