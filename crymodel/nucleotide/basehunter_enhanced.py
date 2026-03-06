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
import gemmi
from .classification import (
    classify_with_templates,
    enforce_base_pair_constraint,
    compute_class_statistics,
    bootstrap_classification,
    calculate_normalized_cross_correlation,
    compute_pair_assignment_probabilities,
    _compute_shift_from_reference_mask,
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
    pair_mismatch_penalty: float = 0.1,
    template_mask_threshold: Optional[float] = None,
    center_sphere_radius: Optional[float] = None,
    score_on_raw: bool = False,
    alignment_reference_pdb: Optional[Path] = None,
    alignment_reference_atom_radius_vox: float = 1.5,
    alignment_on_raw: bool = False,
    output_reference_models: bool = False,
    reference_model_output_dir: Optional[Path] = None,
    apply_alignment: bool = True,
    template_threshold: Optional[float] = None,
    backbone_mask_mrc: Optional[Path] = None,
    backbone_mask_threshold: Optional[float] = None,
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
        pair_mismatch_penalty: Penalty for same-class base pairs (0.0=hard constraint)
        template_mask_threshold: Template mask threshold to reduce backbone influence
        center_sphere_radius: Spherical mask radius (vox) around box center
        score_on_raw: If True, score using raw volumes and align using thresholded
        alignment_reference_pdb: Reference PDB for alignment (base-only)
        alignment_reference_atom_radius_vox: Atom mask radius in voxels
        alignment_on_raw: If True, align using raw volumes instead of thresholded
        output_reference_models: If True, write aligned reference PDB per volume
        reference_model_output_dir: Output directory for aligned reference PDBs
        apply_alignment: If False, skip alignment and score in place
        template_threshold: Threshold for template maps (zero below)
        backbone_mask_mrc: Backbone mask MRC to exclude sugar/phosphate
        backbone_mask_threshold: Threshold for backbone mask MRC
        output_dir: Output directory for results
        volume_thresholds: Optional per-volume threshold overrides
        
    Returns:
        DataFrame with classifications, probabilities, and statistics
    """
    # Load template library
    template_dir = Path(template_dir)
    template_lib = TemplateLibrary(template_dir=template_dir)

    # Auto-defaults for packaged DNA template sets (if not explicitly provided)
    if template_threshold is None:
        if (template_dir / "templateBP-purine.mrc").exists() and (template_dir / "templateBP-pyrimidine.mrc").exists():
            template_threshold = 1.7
    if backbone_mask_mrc is None:
        candidate = template_dir / "template-sp-backbone.mrc"
        if candidate.exists():
            backbone_mask_mrc = candidate
            if backbone_mask_threshold is None:
                backbone_mask_threshold = 0.45
    
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
    thresholded_volumes = {}
    for vol_path, volume in raw_volumes.items():
        threshold_value = thresholds.get(vol_path, threshold)
        thresholded_volumes[vol_path] = threshold_volume(volume, threshold_value)

    volumes = raw_volumes if score_on_raw else thresholded_volumes
    alignment_volumes = raw_volumes if alignment_on_raw else thresholded_volumes
    
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

    backbone_mask = None

    if template_threshold is not None:
        purine_template = threshold_volume(purine_template, template_threshold)
        pyrimidine_template = threshold_volume(pyrimidine_template, template_threshold)

    if backbone_mask_mrc is not None:
        mv = read_map(str(backbone_mask_mrc))
        mask_data = mv.data_zyx
        if backbone_mask_threshold is not None:
            mask_data = (mask_data >= backbone_mask_threshold).astype(np.float32)
        else:
            mask_data = (mask_data > 0).astype(np.float32)
        if mask_data.shape != target_shape:
            raise ValueError("Backbone mask shape does not match target shape")
        backbone_mask = mask_data

    if backbone_mask is not None:
        keep_mask = 1.0 - backbone_mask
        purine_template = purine_template * keep_mask
        pyrimidine_template = pyrimidine_template * keep_mask
        for vol_path in volumes:
            volumes[vol_path] = volumes[vol_path] * keep_mask
        for vol_path in alignment_volumes:
            alignment_volumes[vol_path] = alignment_volumes[vol_path] * keep_mask
    # Build reference alignment mask if provided
    alignment_reference_mask = None
    alignment_reference_path = None
    if alignment_reference_pdb:
        ref_path = Path(alignment_reference_pdb)
        if not ref_path.exists():
            raise ValueError(f"Alignment reference PDB not found: {ref_path}")

        alignment_reference_path = ref_path
        st = gemmi.read_structure(str(ref_path))
        atoms_xyz = []
        for model in st:
            for chain in model:
                for res in chain:
                    for atom in res:
                        element_name = atom.element.name if atom.element else atom.name.strip()[0] if atom.name.strip() else "C"
                        if element_name.upper() == "H":
                            continue
                        pos = atom.pos
                        atoms_xyz.append([float(pos.x), float(pos.y), float(pos.z)])

        if atoms_xyz:
            atoms_xyz = np.array(atoms_xyz, dtype=np.float32)
            alignment_reference_mask = np.zeros(target_shape, dtype=np.float32)
            z_shape, y_shape, x_shape = target_shape
            radius = max(0.5, float(alignment_reference_atom_radius_vox))
            radius2 = radius ** 2
            for x, y, z in atoms_xyz:
                vx = (x - target_origin[0]) / target_apix
                vy = (y - target_origin[1]) / target_apix
                vz = (z - target_origin[2]) / target_apix

                vx_int = int(np.round(vx))
                vy_int = int(np.round(vy))
                vz_int = int(np.round(vz))

                z0 = max(0, int(np.floor(vz_int - radius)))
                z1 = min(z_shape, int(np.ceil(vz_int + radius + 1)))
                y0 = max(0, int(np.floor(vy_int - radius)))
                y1 = min(y_shape, int(np.ceil(vy_int + radius + 1)))
                x0 = max(0, int(np.floor(vx_int - radius)))
                x1 = min(x_shape, int(np.ceil(vx_int + radius + 1)))

                for zz in range(z0, z1):
                    dz2 = (zz - vz_int) ** 2
                    for yy in range(y0, y1):
                        dy2 = (yy - vy_int) ** 2
                        for xx in range(x0, x1):
                            dx2 = (xx - vx_int) ** 2
                            if dx2 + dy2 + dz2 <= radius2:
                                alignment_reference_mask[zz, yy, xx] = 1.0
        else:
            alignment_reference_mask = None

    reference_output_dir = None
    if output_reference_models:
        reference_output_dir = Path(reference_model_output_dir) if reference_model_output_dir else None
        if reference_output_dir is None:
            if output_dir:
                reference_output_dir = Path(output_dir) / "aligned_reference_models"
            else:
                raise ValueError("reference_model_output_dir required when output_reference_models=True and output_dir is None")
        reference_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Templates loaded: purine shape={purine_template.shape}, pyrimidine shape={pyrimidine_template.shape}")
    
    # Classify each volume
    print("\nClassifying volumes...")
    classifications = {}
    classification_details = {}
    
    for vol_path, volume in volumes.items():
        shift = None
        if alignment_reference_mask is not None:
            align_vol = alignment_volumes.get(vol_path)
            if align_vol is not None:
                shift = _compute_shift_from_reference_mask(align_vol, alignment_reference_mask)

        class_name, purine_score, pyrimidine_score, confidence = classify_with_templates(
            volume, purine_template, pyrimidine_template, 
            alignment_threshold, use_emd=use_emd, emd_weight=emd_weight,
            template_mask_threshold=template_mask_threshold,
            center_sphere_radius=center_sphere_radius,
            alignment_volume=alignment_volumes.get(vol_path),
            alignment_reference_mask=alignment_reference_mask,
            apply_alignment=apply_alignment,
        )
        classifications[vol_path] = (class_name, purine_score, pyrimidine_score, confidence)
        classification_details[vol_path] = {
            "class": class_name,
            "purine_score": purine_score,
            "pyrimidine_score": pyrimidine_score,
            "confidence": confidence,
        }
        print(f"  {os.path.basename(vol_path)}: {class_name} (purine={purine_score:.3f}, pyrimidine={pyrimidine_score:.3f}, conf={confidence:.3f})")

        if output_reference_models and alignment_reference_path is not None and shift is not None:
            dz, dy, dx = shift
            translation = gemmi.Vec3(float(dx * target_apix), float(dy * target_apix), float(dz * target_apix))
            st_copy = gemmi.read_structure(str(alignment_reference_path))
            for model in st_copy:
                for chain in model:
                    for res in chain:
                        for atom in res:
                            atom.pos = gemmi.Position(
                                atom.pos.x + translation.x,
                                atom.pos.y + translation.y,
                                atom.pos.z + translation.z,
                            )
            out_name = f"{Path(vol_path).stem}_reference_aligned.pdb"
            out_path = reference_output_dir / out_name
            st_copy.write_pdb(str(out_path))
    
    # Enforce base pair constraints with soft penalty for same-class pairs
    print("\nEnforcing base pair constraints...")
    resolved = enforce_base_pair_constraint(
        classifications, volume_pairs, pair_mismatch_penalty=pair_mismatch_penalty
    )
    
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
    pair_confidence_bootstrap = {}
    if use_bootstrap:
        print(f"\nPerforming bootstrap analysis ({n_bootstrap} iterations)...")
        likelihoods, pair_confidence_bootstrap = bootstrap_classification(
            volumes, purine_template, pyrimidine_template, volume_pairs,
            n_bootstrap=n_bootstrap, alignment_threshold=alignment_threshold,
            use_emd=use_emd, emd_weight=emd_weight,
            pair_mismatch_penalty=pair_mismatch_penalty,
            template_mask_threshold=template_mask_threshold,
            center_sphere_radius=center_sphere_radius,
            alignment_volumes=alignment_volumes if score_on_raw else None,
            alignment_reference_mask=alignment_reference_mask,
            apply_alignment=apply_alignment,
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
        
        # Normalize to probabilities for reporting
        def _normalize_probs(pur_score: float, pyr_score: float) -> Tuple[float, float]:
            total = max(1e-8, pur_score + pyr_score)
            return pur_score / total, pyr_score / total

        v1_pur_prob, v1_pyr_prob = _normalize_probs(
            details1.get("purine_score", 0.0), details1.get("pyrimidine_score", 0.0)
        )
        v2_pur_prob, v2_pyr_prob = _normalize_probs(
            details2.get("purine_score", 0.0), details2.get("pyrimidine_score", 0.0)
        )

        pair_probs = compute_pair_assignment_probabilities(
            v1_pur_prob,
            v1_pyr_prob,
            v2_pur_prob,
            v2_pyr_prob,
            pair_mismatch_penalty=pair_mismatch_penalty,
        )
        pair_key = (class1, class2)
        pair_confidence = pair_probs.get(pair_key, 0.0)

        # Pair-aware p-value from bootstrap confidence distribution
        pair_p_value = 1.0
        if use_bootstrap:
            samples = pair_confidence_bootstrap.get((vol_path1, vol_path2), [])
            if samples:
                pair_p_value = float(np.mean([s >= pair_confidence for s in samples]))

        results.append({
            "volume1": vol1_name,
            "volume1_path": vol_path1,
            "volume1_class": class1,
            "volume1_purine_score": details1.get("purine_score", 0.0),
            "volume1_pyrimidine_score": details1.get("pyrimidine_score", 0.0),
            "volume1_purine_prob": v1_pur_prob,
            "volume1_pyrimidine_prob": v1_pyr_prob,
            "volume1_confidence": details1.get("confidence", 0.0),
            "volume1_purine_likelihood": likelihood1.get("purine", 0.0),
            "volume1_pyrimidine_likelihood": likelihood1.get("pyrimidine", 0.0),
            "volume1_unclassified_likelihood": likelihood1.get("unclassified", 0.0),
            "volume2": vol2_name,
            "volume2_path": vol_path2,
            "volume2_class": class2,
            "volume2_purine_score": details2.get("purine_score", 0.0),
            "volume2_pyrimidine_score": details2.get("pyrimidine_score", 0.0),
            "volume2_purine_prob": v2_pur_prob,
            "volume2_pyrimidine_prob": v2_pyr_prob,
            "volume2_confidence": details2.get("confidence", 0.0),
            "volume2_purine_likelihood": likelihood2.get("purine", 0.0),
            "volume2_pyrimidine_likelihood": likelihood2.get("pyrimidine", 0.0),
            "volume2_unclassified_likelihood": likelihood2.get("unclassified", 0.0),
            "pair_assignment_confidence": pair_confidence,
            "pair_assignment_p_value": pair_p_value,
            "pair_assignment_prob_pur_pur": pair_probs.get(("purine", "purine"), 0.0),
            "pair_assignment_prob_pur_pyr": pair_probs.get(("purine", "pyrimidine"), 0.0),
            "pair_assignment_prob_pyr_pur": pair_probs.get(("pyrimidine", "purine"), 0.0),
            "pair_assignment_prob_pyr_pyr": pair_probs.get(("pyrimidine", "pyrimidine"), 0.0),
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

