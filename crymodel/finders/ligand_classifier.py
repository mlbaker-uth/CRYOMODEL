# crymodel/finders/ligand_classifier.py
"""Rule-based ligand classification."""
from __future__ import annotations
import numpy as np
from typing import Dict


# Ligand classes
LIGAND_CLASSES = ["heme", "nad", "nucleotide", "phospholipid", "uq"]
CLASS_DESCRIPTIONS = {
    "heme": "Heme/porphyrin",
    "nad": "NAD(H)/NADP(H)",
    "nucleotide": "Nucleotide (ATP/ADP/GTP/GDP)",
    "phospholipid": "Phospholipid",
    "uq": "Ubiquinone/CoQ",
}


def rule_based_classify(features: Dict[str, float]) -> Dict[str, float]:
    """Rule-based classification of ligand components.
    
    Args:
        features: Dictionary of component features
        
    Returns:
        Dictionary of class scores (0-1)
    """
    scores = {cls: 0.0 for cls in LIGAND_CLASSES}
    
    volume = features.get("volume_A3", 0.0)
    planarity = features.get("planarity", 0.0)
    anisotropy = features.get("anisotropy", 0.0)
    n_peaks = features.get("n_peaks", 0)
    peak_spacing = features.get("peak_mean_spacing_A", np.nan)
    axial_HisCys = features.get("axial_HisCys", False)
    has_Fe = features.get("has_Fe", False)
    has_Mg = features.get("has_Mg", False)
    basic_count = features.get("basic_residue_count", 0)
    acidic_count = features.get("acidic_residue_count", 0)
    in_membrane = features.get("in_membrane_belt", False)  # Would need membrane detection
    
    # Heme/porphyrin: High planarity + axial ligation + Fe + compact
    if planarity > 0.85 and axial_HisCys and (has_Fe or volume < 600):
        scores["heme"] += 0.8
    if planarity > 0.75 and has_Fe:
        scores["heme"] += 0.4
    if volume > 300 and volume < 800 and planarity > 0.7:
        scores["heme"] += 0.2
    
    # Nucleotide: Phosphate chain (peaks with ~3-4Å spacing) + basic residues + Mg
    if n_peaks >= 2 and n_peaks <= 4:
        if not np.isnan(peak_spacing) and 2.5 <= peak_spacing <= 4.5:
            scores["nucleotide"] += 0.6
        if basic_count > acidic_count:
            scores["nucleotide"] += 0.3
        if has_Mg:
            scores["nucleotide"] += 0.4
    if volume > 400 and volume < 1200 and n_peaks >= 2:
        scores["nucleotide"] += 0.2
    
    # NAD/NADP: Two ring lobes (two planar regions) + elongated shape
    if volume > 600 and volume < 1400:
        if planarity < 0.6 and anisotropy > 0.5:  # Elongated, not planar
            scores["nad"] += 0.5
        if basic_count > 2:  # Often in basic pockets
            scores["nad"] += 0.3
        if n_peaks >= 2:
            scores["nad"] += 0.2
    
    # Phospholipid: Two tails (high anisotropy) + membrane + large volume
    if in_membrane or volume > 1200:
        if anisotropy > 0.6 and volume > 1000:
            scores["phospholipid"] += 0.7
        if volume > 1200:
            scores["phospholipid"] += 0.4
    if volume > 800 and anisotropy > 0.5:
        scores["phospholipid"] += 0.2
    
    # UQ/CoQ: Long tail (high anisotropy) + membrane + medium volume
    if in_membrane or (volume > 500 and volume < 1200):
        if anisotropy > 0.7 and volume < 1200:
            scores["uq"] += 0.6
        if volume > 400 and volume < 1000 and anisotropy > 0.6:
            scores["uq"] += 0.3
    
    # Volume-based filtering
    if volume < 120:
        # Too small - likely noise or residual ion
        for cls in LIGAND_CLASSES:
            scores[cls] *= 0.3
    
    # Normalize scores to 0-1 range
    max_score = max(scores.values()) if scores.values() else 1.0
    if max_score > 0:
        for cls in scores:
            scores[cls] = min(1.0, scores[cls] / max_score)
    
    return scores


def get_top_class(scores: Dict[str, float], confidence_threshold: float = 0.3) -> tuple[str, float, float]:
    """Get top predicted class and confidence.
    
    Args:
        scores: Dictionary of class scores
        confidence_threshold: Minimum score to consider a valid prediction
        
    Returns:
        (top_class, top_score, confidence) tuple
    """
    if not scores or max(scores.values()) < confidence_threshold:
        return ("unknown", 0.0, 0.0)
    
    sorted_classes = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_class, top_score = sorted_classes[0]
    
    # Confidence based on margin over second-best
    if len(sorted_classes) > 1:
        second_score = sorted_classes[1][1]
        margin = top_score - second_score
        confidence = min(1.0, top_score * 0.7 + margin * 0.3)
    else:
        confidence = top_score
    
    return (top_class, top_score, confidence)

