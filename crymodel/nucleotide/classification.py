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


def _pad_to_match(volume: np.ndarray, template: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Pad smaller volume to match larger (z, y, x)."""
    if volume.shape == template.shape:
        return volume, template

    max_shape = tuple(max(s1, s2) for s1, s2 in zip(volume.shape, template.shape))
    volume_padded = np.zeros(max_shape, dtype=volume.dtype)
    template_padded = np.zeros(max_shape, dtype=template.dtype)

    v0, v1, v2 = volume.shape
    t0, t1, t2 = template.shape

    volume_padded[:v0, :v1, :v2] = volume
    template_padded[:t0, :t1, :t2] = template

    return volume_padded, template_padded


def _compute_alignment_shift(volume: np.ndarray, template: np.ndarray) -> Tuple[int, int, int]:
    """Compute best translation shift from volume to template using FFT correlation."""
    volume, template = _pad_to_match(volume, template)

    volume_norm = volume - np.mean(volume)
    volume_std = np.std(volume_norm)
    if volume_std > 1e-6:
        volume_norm /= volume_std

    template_norm = template - np.mean(template)
    template_std = np.std(template_norm)
    if template_std > 1e-6:
        template_norm /= template_std

    volume_fft = np.fft.fftn(volume_norm)
    template_fft = np.fft.fftn(template_norm)
    corr_fft = np.conj(volume_fft) * template_fft
    corr = np.fft.ifftn(corr_fft).real

    peak_idx = np.unravel_index(np.argmax(corr), corr.shape)
    shift = (
        peak_idx[0] - volume.shape[0] // 2,
        peak_idx[1] - volume.shape[1] // 2,
        peak_idx[2] - volume.shape[2] // 2,
    )
    return shift


def _compute_shift_from_reference_mask(
    volume_mask: np.ndarray,
    reference_mask: np.ndarray,
) -> Tuple[int, int, int]:
    """Compute translation shift that maximizes overlap with a reference mask."""
    volume_mask, reference_mask = _pad_to_match(volume_mask, reference_mask)

    vol_fft = np.fft.fftn(volume_mask)
    ref_fft = np.fft.fftn(reference_mask)
    corr_fft = np.conj(vol_fft) * ref_fft
    corr = np.fft.ifftn(corr_fft).real

    peak_idx = np.unravel_index(np.argmax(corr), corr.shape)
    shift = (
        peak_idx[0] - volume_mask.shape[0] // 2,
        peak_idx[1] - volume_mask.shape[1] // 2,
        peak_idx[2] - volume_mask.shape[2] // 2,
    )
    return shift


def _apply_shift(volume: np.ndarray, shift: Tuple[int, int, int]) -> np.ndarray:
    """Apply circular shift to volume."""
    return np.roll(volume, shift, axis=(0, 1, 2))


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
    shift = _compute_alignment_shift(volume, template)
    aligned = _apply_shift(volume, shift)
    transform = np.eye(4, dtype=np.float32)

    ncc = calculate_normalized_cross_correlation(aligned, template)
    corr_score = (ncc + 1.0) / 2.0

    return aligned, transform, float(corr_score)


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


def _normalize_class_scores(purine_score: float, pyrimidine_score: float) -> Tuple[float, float]:
    """Normalize purine/pyrimidine scores into probabilities."""
    pur = max(0.0, float(purine_score))
    pyr = max(0.0, float(pyrimidine_score))
    total = pur + pyr
    if total < 1e-8:
        return 0.5, 0.5
    return pur / total, pyr / total


def _resolve_pair_assignment(
    p1_pur: float,
    p1_pyr: float,
    p2_pur: float,
    p2_pyr: float,
    pair_mismatch_penalty: float,
) -> Tuple[str, str]:
    """
    Resolve a base-pair assignment using probabilistic scoring.

    The pair_mismatch_penalty down-weights assignments where both bases
    are purines or both are pyrimidines. A value of 0.1 matches the
    low-probability correction used in doubleHelix. A value of 0.0
    makes the constraint hard (only purine+pyrimidine allowed).
    """
    eps = 1e-8
    p1_pur = max(p1_pur, eps)
    p1_pyr = max(p1_pyr, eps)
    p2_pur = max(p2_pur, eps)
    p2_pyr = max(p2_pyr, eps)

    if pair_mismatch_penalty <= 0.0:
        # Hard constraint: disallow purine/purine and pyrimidine/pyrimidine
        score_pp = -np.inf
        score_yy = -np.inf
    else:
        penalty = float(min(pair_mismatch_penalty, 1.0))
        score_pp = np.log(p1_pur) + np.log(p2_pur) + np.log(penalty)
        score_yy = np.log(p1_pyr) + np.log(p2_pyr) + np.log(penalty)

    score_py = np.log(p1_pur) + np.log(p2_pyr)
    score_yp = np.log(p1_pyr) + np.log(p2_pur)

    scores = {
        ("purine", "purine"): score_pp,
        ("purine", "pyrimidine"): score_py,
        ("pyrimidine", "purine"): score_yp,
        ("pyrimidine", "pyrimidine"): score_yy,
    }
    best_assignment = max(scores.items(), key=lambda item: item[1])[0]
    return best_assignment[0], best_assignment[1]


def _apply_template_mask(
    volume: np.ndarray,
    template: np.ndarray,
    template_mask_threshold: Optional[float],
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply a template-derived mask to focus scoring on base density."""
    if template_mask_threshold is None:
        return volume, template

    mask = template > template_mask_threshold
    if not np.any(mask):
        return volume, template

    return volume * mask, template * mask


def _apply_center_sphere_mask(
    volume: np.ndarray,
    template: np.ndarray,
    radius_vox: Optional[float],
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply a spherical mask around the box center to reduce backbone signal."""
    if radius_vox is None or radius_vox <= 0:
        return volume, template

    z, y, x = volume.shape
    cz, cy, cx = (z - 1) / 2.0, (y - 1) / 2.0, (x - 1) / 2.0
    zz, yy, xx = np.ogrid[:z, :y, :x]
    dist2 = (zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2
    mask = dist2 <= (radius_vox ** 2)
    if not np.any(mask):
        return volume, template

    return volume * mask, template * mask

def compute_pair_assignment_probabilities(
    p1_pur: float,
    p1_pyr: float,
    p2_pur: float,
    p2_pyr: float,
    pair_mismatch_penalty: float,
) -> Dict[Tuple[str, str], float]:
    """
    Compute normalized probabilities for all pair assignments.

    Returns a dict keyed by (class1, class2) with probabilities that sum to 1.
    """
    eps = 1e-8
    p1_pur = max(p1_pur, eps)
    p1_pyr = max(p1_pyr, eps)
    p2_pur = max(p2_pur, eps)
    p2_pyr = max(p2_pyr, eps)

    if pair_mismatch_penalty <= 0.0:
        score_pp = -np.inf
        score_yy = -np.inf
    else:
        penalty = float(min(pair_mismatch_penalty, 1.0))
        score_pp = np.log(p1_pur) + np.log(p2_pur) + np.log(penalty)
        score_yy = np.log(p1_pyr) + np.log(p2_pyr) + np.log(penalty)

    score_py = np.log(p1_pur) + np.log(p2_pyr)
    score_yp = np.log(p1_pyr) + np.log(p2_pur)

    scores = {
        ("purine", "purine"): score_pp,
        ("purine", "pyrimidine"): score_py,
        ("pyrimidine", "purine"): score_yp,
        ("pyrimidine", "pyrimidine"): score_yy,
    }

    # Softmax over finite scores
    finite_scores = [s for s in scores.values() if np.isfinite(s)]
    if not finite_scores:
        # Fallback to uniform
        return {k: 0.25 for k in scores}

    max_score = max(finite_scores)
    exp_scores = {}
    total = 0.0
    for k, s in scores.items():
        if np.isfinite(s):
            exp_s = np.exp(s - max_score)
        else:
            exp_s = 0.0
        exp_scores[k] = exp_s
        total += exp_s

    if total < 1e-12:
        return {k: 0.25 for k in scores}

    return {k: v / total for k, v in exp_scores.items()}

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
    template_mask_threshold: Optional[float] = None,
    center_sphere_radius: Optional[float] = None,
    alignment_volume: Optional[np.ndarray] = None,
    alignment_reference_mask: Optional[np.ndarray] = None,
    apply_alignment: bool = True,
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
    if apply_alignment:
        align_vol = alignment_volume if alignment_volume is not None else volume
        if alignment_reference_mask is not None and align_vol is not None:
            shift = _compute_shift_from_reference_mask(align_vol, alignment_reference_mask)
            shift_pur = shift
            shift_pyr = shift
        else:
            shift_pur = _compute_alignment_shift(align_vol, purine_template)
            shift_pyr = _compute_alignment_shift(align_vol, pyrimidine_template)
        aligned_purine = _apply_shift(volume, shift_pur)
        aligned_pyrimidine = _apply_shift(volume, shift_pyr)
    else:
        aligned_purine = volume
        aligned_pyrimidine = volume

    aligned_purine, purine_template = _apply_template_mask(
        aligned_purine, purine_template, template_mask_threshold
    )
    aligned_pyrimidine, pyrimidine_template = _apply_template_mask(
        aligned_pyrimidine, pyrimidine_template, template_mask_threshold
    )

    aligned_purine, purine_template = _apply_center_sphere_mask(
        aligned_purine, purine_template, center_sphere_radius
    )
    aligned_pyrimidine, pyrimidine_template = _apply_center_sphere_mask(
        aligned_pyrimidine, pyrimidine_template, center_sphere_radius
    )

    purine_corr = calculate_normalized_cross_correlation(aligned_purine, purine_template)
    pyrimidine_corr = calculate_normalized_cross_correlation(aligned_pyrimidine, pyrimidine_template)
    purine_corr = (purine_corr + 1.0) / 2.0
    pyrimidine_corr = (pyrimidine_corr + 1.0) / 2.0
    
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
    pair_mismatch_penalty: float = 0.1,
) -> Dict[str, str]:
    """
    Enforce constraint: each base pair must have 1 purine + 1 pyrimidine.
    
    Uses a probabilistic pair assignment and applies a penalty for
    same-class pairs (purine/purine or pyrimidine/pyrimidine).
    
    Args:
        classifications: Dict mapping volume_path -> (class, purine_score, pyrimidine_score, confidence)
        base_pairs: List of (volume1_path, volume2_path) tuples
        pair_mismatch_penalty: Penalty for same-class pairs (0.0=hard constraint)
        
    Returns:
        Dict mapping volume_path -> resolved_class ("purine" or "pyrimidine")
    """
    resolved = {}
    
    for vol1, vol2 in base_pairs:
        if vol1 not in classifications or vol2 not in classifications:
            continue
        
        class1, score1_pur, score1_pyr, conf1 = classifications[vol1]
        class2, score2_pur, score2_pyr, conf2 = classifications[vol2]

        # Convert scores to normalized probabilities for pairwise resolution
        p1_pur, p1_pyr = _normalize_class_scores(score1_pur, score1_pyr)
        p2_pur, p2_pyr = _normalize_class_scores(score2_pur, score2_pyr)

        class1, class2 = _resolve_pair_assignment(
            p1_pur, p1_pyr, p2_pur, p2_pyr, pair_mismatch_penalty
        )
        
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
    pair_mismatch_penalty: float = 0.1,
    template_mask_threshold: Optional[float] = None,
    center_sphere_radius: Optional[float] = None,
    alignment_volumes: Optional[Dict[str, np.ndarray]] = None,
    alignment_reference_mask: Optional[np.ndarray] = None,
    apply_alignment: bool = True,
) -> Tuple[Dict[str, Dict[str, float]], Dict[Tuple[str, str], List[float]]]:
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
        (likelihoods, pair_confidences)
        likelihoods: Dict mapping volume_path -> {"purine": likelihood, "pyrimidine": likelihood, "unclassified": likelihood}
        pair_confidences: Dict mapping (vol1, vol2) -> list of pair_confidence values from bootstrap samples
    """
    from collections import defaultdict
    
    likelihoods = defaultdict(lambda: {"purine": 0.0, "pyrimidine": 0.0, "unclassified": 0.0})
    pair_confidences = defaultdict(list)
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
            align_vol = alignment_volumes.get(vol_name) if alignment_volumes else None
            class_name, pur_score, pyr_score, conf = classify_with_templates(
                vol_data, purine_template, pyrimidine_template, 
                alignment_threshold, use_emd=use_emd, emd_weight=emd_weight,
                template_mask_threshold=template_mask_threshold,
                center_sphere_radius=center_sphere_radius,
                alignment_volume=align_vol,
                alignment_reference_mask=alignment_reference_mask,
                apply_alignment=apply_alignment,
            )
            classifications[vol_name] = (class_name, pur_score, pyr_score, conf)
        
        # Enforce constraints
        resolved = enforce_base_pair_constraint(
            classifications, base_pairs, pair_mismatch_penalty=pair_mismatch_penalty
        )
        
        # Count assignments
        for vol_name, class_name in resolved.items():
            likelihoods[vol_name][class_name] += 1.0 / n_bootstrap

        # Pair-wise confidence distribution
        for vol1, vol2 in base_pairs:
            if vol1 not in classifications or vol2 not in classifications:
                continue
            _, score1_pur, score1_pyr, _ = classifications[vol1]
            _, score2_pur, score2_pyr, _ = classifications[vol2]
            p1_pur, p1_pyr = _normalize_class_scores(score1_pur, score1_pyr)
            p2_pur, p2_pyr = _normalize_class_scores(score2_pur, score2_pyr)
            pair_probs = compute_pair_assignment_probabilities(
                p1_pur,
                p1_pyr,
                p2_pur,
                p2_pyr,
                pair_mismatch_penalty=pair_mismatch_penalty,
            )
            class1 = resolved.get(vol1, "purine")
            class2 = resolved.get(vol2, "pyrimidine")
            pair_confidences[(vol1, vol2)].append(
                pair_probs.get((class1, class2), 0.0)
            )
    
    return dict(likelihoods), dict(pair_confidences)

