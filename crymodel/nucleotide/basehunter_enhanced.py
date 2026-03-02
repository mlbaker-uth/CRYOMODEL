# crymodel/nucleotide/basehunter_enhanced.py
"""Enhanced BaseHunter with template-based classification."""
from __future__ import annotations
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import mrcfile

from ..io.mrc import read_map, MapVolume
from .templates import TemplateLibrary
from .classification import (
    classify_with_templates,
    enforce_base_pair_constraint,
    compute_class_statistics,
    bootstrap_classification,
    calculate_normalized_cross_correlation,
)


def read_mrc_file(filepath: str) -> Tuple[Optional[np.ndarray], Optional[mrcfile.MrcFile]]:
    """Read MRC file."""
    try:
        with mrcfile.open(filepath, permissive=True) as mrc:
            return np.copy(mrc.data), mrc
    except Exception as e:
        print(f"Error reading MRC file {filepath}: {e}")
        return None, None


def threshold_volume(volume: np.ndarray, threshold: float) -> np.ndarray:
    """Threshold volume."""
    return np.where(volume >= threshold, volume, 0)


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


def classify_base_pairs(
    volume_pairs: List[Tuple[str, str]],
    template_dir: Path,
    threshold: float,
    target_resolution: Optional[float] = None,
    target_apix: Optional[float] = None,
    alignment_threshold: float = 0.3,
    use_bootstrap: bool = True,
    n_bootstrap: int = 100,
    use_emd: bool = True,
    emd_weight: float = 0.2,
    output_dir: Optional[Path] = None,
    volume_thresholds: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """
    Classify base pairs using template-based classification.
    
    Args:
        volume_pairs: List of (volume1_path, volume2_path) tuples
        template_dir: Directory containing templates
        threshold: Density threshold for volumes
        target_resolution: Target resolution in Å (auto-detect from first volume if None)
        target_apix: Target voxel size in Å (auto-detect if None)
        alignment_threshold: Minimum correlation for classification
        use_bootstrap: If True, perform bootstrap analysis for likelihoods
        n_bootstrap: Number of bootstrap iterations
        output_dir: Output directory for results
        volume_thresholds: Optional per-volume threshold overrides
        
    Returns:
        DataFrame with classifications and statistics
    """
    # Load template library
    template_lib = TemplateLibrary(template_dir=template_dir)
    
    # Load and preprocess volumes
    volumes = {}
    raw_volumes = {}
    reference_mrc = None
    thresholds = volume_thresholds or {}
    
    print(f"Loading {len(volume_pairs) * 2} volumes...")
    for vol_path1, vol_path2 in volume_pairs:
        for vol_path in [vol_path1, vol_path2]:
            if vol_path not in volumes:
                volume, mrc = read_mrc_file(vol_path)
                if volume is not None:
                    raw_volumes[vol_path] = volume
                    
                    if reference_mrc is None and mrc is not None:
                        reference_mrc = mrc
                        
                        # Auto-detect resolution and apix if not provided
                        if target_resolution is None:
                            # Try to get from map header or use default
                            target_resolution = 3.5  # Default
                        if target_apix is None:
                            # Get from map header
                            try:
                                target_apix = float(mrc.voxel_size.x)
                            except:
                                target_apix = 1.0  # Default
    
    if not raw_volumes:
        raise ValueError("No volumes loaded")
    
    # Apply per-volume thresholds (if provided), otherwise use default threshold
    for vol_path, volume in raw_volumes.items():
        threshold_value = thresholds.get(vol_path, threshold)
        volumes[vol_path] = threshold_volume(volume, threshold_value)
    
    print(f"Loaded {len(volumes)} volumes")
    print(f"Target resolution: {target_resolution} Å, apix: {target_apix} Å/voxel")
    
    # Get templates at target resolution
    # Use first volume's shape as reference
    first_volume = list(volumes.values())[0]
    target_shape = first_volume.shape
    target_origin = np.array([0.0, 0.0, 0.0])  # Assume centered
    
    print("Loading templates...")
    purine_template, pyrimidine_template = template_lib.get_templates_at_resolution(
        target_resolution=target_resolution,
        target_apix=target_apix,
        target_shape=target_shape,
        target_origin=target_origin,
        use_average=True,
    )
    print(f"Templates loaded: purine shape={purine_template.shape}, pyrimidine shape={pyrimidine_template.shape}")
    
    # Classify each volume
    print("\nClassifying volumes...")
    classifications = {}
    classification_details = {}
    
    for vol_path, volume in volumes.items():
        class_name, purine_score, pyrimidine_score, confidence = classify_with_templates(
            volume, purine_template, pyrimidine_template, 
            alignment_threshold, use_emd=use_emd, emd_weight=emd_weight
        )
        classifications[vol_path] = (class_name, purine_score, pyrimidine_score, confidence)
        classification_details[vol_path] = {
            "class": class_name,
            "purine_score": purine_score,
            "pyrimidine_score": pyrimidine_score,
            "confidence": confidence,
        }
        print(f"  {os.path.basename(vol_path)}: {class_name} (purine={purine_score:.3f}, pyrimidine={pyrimidine_score:.3f}, conf={confidence:.3f})")
    
    # Enforce base pair constraints
    print("\nEnforcing base pair constraints...")
    resolved = enforce_base_pair_constraint(classifications, volume_pairs)
    
    # Count changes
    n_changed = sum(1 for vol in resolved if vol in classifications and resolved[vol] != classifications[vol][0])
    print(f"  Resolved {n_changed} unclassified/conflicting assignments")
    
    # Compute inter/intra-class statistics
    print("\nComputing class statistics...")
    stats = compute_class_statistics(volumes, resolved)
    print(f"  Purine intra-class NCC: {stats['purine_intra_mean']:.4f} ± {stats['purine_intra_std']:.4f} (n={stats['purine_intra_n']})")
    print(f"  Pyrimidine intra-class NCC: {stats['pyrimidine_intra_mean']:.4f} ± {stats['pyrimidine_intra_std']:.4f} (n={stats['pyrimidine_intra_n']})")
    print(f"  Inter-class NCC: {stats['inter_class_mean']:.4f} ± {stats['inter_class_std']:.4f} (n={stats['inter_class_n']})")
    print(f"  Separation score: {stats['separation_score']:.4f}")
    
    # Bootstrap analysis for likelihoods
    likelihoods = {}
    if use_bootstrap:
        print(f"\nPerforming bootstrap analysis ({n_bootstrap} iterations)...")
        likelihoods = bootstrap_classification(
            volumes, purine_template, pyrimidine_template, volume_pairs,
            n_bootstrap=n_bootstrap, alignment_threshold=alignment_threshold,
            use_emd=use_emd, emd_weight=emd_weight
        )
        print("  Bootstrap analysis complete")
    
    # Build results DataFrame
    results = []
    for vol_path1, vol_path2 in volume_pairs:
        vol1_name = os.path.basename(vol_path1)
        vol2_name = os.path.basename(vol_path2)
        
        class1 = resolved.get(vol_path1, "unknown")
        class2 = resolved.get(vol_path2, "unknown")
        
        details1 = classification_details.get(vol_path1, {})
        details2 = classification_details.get(vol_path2, {})
        
        likelihood1 = likelihoods.get(vol_path1, {})
        likelihood2 = likelihoods.get(vol_path2, {})
        
        results.append({
            "volume1": vol1_name,
            "volume1_path": vol_path1,
            "volume1_class": class1,
            "volume1_purine_score": details1.get("purine_score", 0.0),
            "volume1_pyrimidine_score": details1.get("pyrimidine_score", 0.0),
            "volume1_confidence": details1.get("confidence", 0.0),
            "volume1_purine_likelihood": likelihood1.get("purine", 0.0),
            "volume1_pyrimidine_likelihood": likelihood1.get("pyrimidine", 0.0),
            "volume1_unclassified_likelihood": likelihood1.get("unclassified", 0.0),
            "volume2": vol2_name,
            "volume2_path": vol_path2,
            "volume2_class": class2,
            "volume2_purine_score": details2.get("purine_score", 0.0),
            "volume2_pyrimidine_score": details2.get("pyrimidine_score", 0.0),
            "volume2_confidence": details2.get("confidence", 0.0),
            "volume2_purine_likelihood": likelihood2.get("purine", 0.0),
            "volume2_pyrimidine_likelihood": likelihood2.get("pyrimidine", 0.0),
            "volume2_unclassified_likelihood": likelihood2.get("unclassified", 0.0),
            "base_pair_valid": (class1 != class2),  # Should be 1 purine + 1 pyrimidine
        })
    
    df = pd.DataFrame(results)
    
    # Add summary statistics
    df.attrs['statistics'] = stats
    df.attrs['n_purine'] = sum(1 for c in resolved.values() if c == "purine")
    df.attrs['n_pyrimidine'] = sum(1 for c in resolved.values() if c == "pyrimidine")
    df.attrs['n_unclassified'] = sum(1 for c in resolved.values() if c == "unclassified")
    
    # Save results
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save CSV
        csv_path = output_dir / "basehunter_classifications.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nResults saved to {csv_path}")
        
        # Save statistics
        stats_path = output_dir / "basehunter_statistics.txt"
        with open(stats_path, 'w') as f:
            f.write("BaseHunter Classification Statistics\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Total volumes: {len(volumes)}\n")
            f.write(f"Total base pairs: {len(volume_pairs)}\n")
            f.write(f"Purine assignments: {df.attrs['n_purine']}\n")
            f.write(f"Pyrimidine assignments: {df.attrs['n_pyrimidine']}\n")
            f.write(f"Unclassified: {df.attrs['n_unclassified']}\n\n")
            f.write("Inter/Intra-class Statistics:\n")
            f.write(f"  Purine intra-class NCC: {stats['purine_intra_mean']:.4f} ± {stats['purine_intra_std']:.4f} (n={stats['purine_intra_n']})\n")
            f.write(f"  Pyrimidine intra-class NCC: {stats['pyrimidine_intra_mean']:.4f} ± {stats['pyrimidine_intra_std']:.4f} (n={stats['pyrimidine_intra_n']})\n")
            f.write(f"  Inter-class NCC: {stats['inter_class_mean']:.4f} ± {stats['inter_class_std']:.4f} (n={stats['inter_class_n']})\n")
            f.write(f"  Separation score: {stats['separation_score']:.4f}\n")
        print(f"Statistics saved to {stats_path}")
    
    return df

