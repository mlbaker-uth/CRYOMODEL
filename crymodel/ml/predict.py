# crymodel/ml/predict.py
"""Prediction utilities for ion/water classification."""
from __future__ import annotations
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Optional

from .model import IonWaterMLP
from .featureizer import build_feature_row, build_kdtree
from .pdb_reader import read_pdb_to_dataframe, read_candidate_waters_to_dataframe
from .density_features import extract_density_features, extract_halfmap_features
from .ensemble import predict_ensemble as _predict_ensemble
from ..io.mrc import read_map, MapVolume

WATER_LABEL = "HOH"


def apply_resolution_filtering(
    results: pd.DataFrame,
    entry_resolution: Optional[float],
    classes: list[str],
    probs: np.ndarray,
) -> pd.DataFrame:
    """Apply resolution-based filtering to predictions.
    
    Resolution ranges (lower numerical value = higher resolution):
    - 0.1-2.5Å: Primarily waters (1-2% ions) - allow waters
    - 2.5-3.0Å: Primarily ions, but some coordinated waters possible - allow waters but prefer ions
    - 3.0-5.0Å: Almost exclusively ions - filter out waters
    - >5.0Å: Should not be annotated (error/warning)
    
    Args:
        results: DataFrame with predictions
        entry_resolution: Resolution in Å (None if unknown)
        classes: List of class names
        probs: Probability matrix (n_samples, n_classes)
        
    Returns:
        DataFrame with adjusted predictions
    """
    # Always add tracking columns
    results["pred_class_original"] = results["pred_class"].copy()
    results["resolution_filtered"] = False
    
    # Check for invalid resolution ranges
    if entry_resolution is not None:
        if entry_resolution > 5.0:
            # Resolution too low - warn but don't filter (let user decide)
            print(f"Warning: Resolution {entry_resolution}Å is outside recommended range (0-5Å)")
        if entry_resolution < 0.1:
            # Resolution too high - unlikely but allow
            print(f"Warning: Resolution {entry_resolution}Å is very high (<0.1Å)")
    
    # If resolution unknown, don't filter (assume high resolution)
    if entry_resolution is None:
        return results
    
    water_idx = classes.index(WATER_LABEL) if WATER_LABEL in classes else None
    if water_idx is None:
        return results
    
    # For each prediction, find the next most likely ion (excluding HOH)
    ion_indices = [i for i, cls in enumerate(classes) if cls != WATER_LABEL]
    
    # Resolution-based filtering thresholds
    if entry_resolution <= 2.5:
        # High resolution (0.1-2.5Å): Primarily waters (1-2% ions)
        # At high resolution, apply strong prior favoring waters
        # Since waters are 10-100x more common than ions at high res,
        # reclassify to water unless model is very confident in an ion
        
        # Calculate a resolution-dependent confidence threshold
        # At 2.5Å: require 0.4 confidence for ions
        # At 1.0Å: require 0.5 confidence for ions
        # Linear interpolation
        if entry_resolution >= 2.0:
            ion_confidence_threshold = 0.35  # Lower threshold for 2.0-2.5Å
        else:
            ion_confidence_threshold = 0.40 + (2.0 - entry_resolution) * 0.1  # Higher for better res
        
        water_reclassified = 0
        # Reset index to ensure we can iterate properly
        results = results.reset_index(drop=True)
        
        # Create new columns for updated predictions
        new_pred_class = results["pred_class"].copy()
        new_confidence = results["confidence"].copy()
        new_filtered = results["resolution_filtered"].copy()
        
        for idx in range(len(results)):
            current_pred = results.loc[idx, "pred_class"]
            if current_pred == WATER_LABEL:
                # Already water, no change needed
                continue
            
            hoh_prob = float(probs[idx, water_idx])
            current_confidence = float(results.loc[idx, "confidence"])
            
            # At high resolution, if model confidence in ion is low,
            # reclassify to water (waters are much more common than ions)
            # At high res (≤2.5Å), waters are 10-100x more common, so low-confidence
            # predictions should default to water unless model is very confident in an ion
            if current_confidence < ion_confidence_threshold:
                # Reclassify to water - low ion confidence + high res strongly favors water
                new_pred_class.iloc[idx] = WATER_LABEL
                # For confidence, use max of p_HOH or a minimum threshold (0.01) 
                # to indicate it's a resolution-based reclassification
                new_confidence.iloc[idx] = max(hoh_prob, 0.01)
                new_filtered.iloc[idx] = True
                water_reclassified += 1
        
        # Update results with new values
        results["pred_class"] = new_pred_class
        results["confidence"] = new_confidence
        results["resolution_filtered"] = new_filtered
        
        if water_reclassified > 0:
            print(f"High resolution ({entry_resolution}Å): Reclassified {water_reclassified} predictions from ions to waters (confidence threshold: {ion_confidence_threshold:.2f})")
        
        return results
    
    elif entry_resolution <= 3.0:
        # Medium resolution (2.5-3.0Å): Primarily ions, but some waters possible
        # Be more conservative - only keep waters if they have high confidence
        # and the best ion alternative is not much better
        for idx in results[hoh_mask].index:
            hoh_prob = probs[idx, water_idx]
            ion_probs = probs[idx, ion_indices]
            if len(ion_probs) > 0:
                best_ion_prob = ion_probs.max()
                # Keep water only if water prob is significantly higher than best ion
                # (water prob must be >1.5x the best ion prob, and >0.3)
                if not (hoh_prob > 1.5 * best_ion_prob and hoh_prob > 0.3):
                    # Reclassify to best ion
                    best_ion_idx = ion_indices[np.argmax(ion_probs)]
                    best_ion_class = classes[best_ion_idx]
                    results.loc[idx, "pred_class"] = best_ion_class
                    results.loc[idx, "confidence"] = best_ion_prob
                    results.loc[idx, "resolution_filtered"] = True
    
    else:
        # Low resolution (3.0-5.0Å): Almost exclusively ions - filter out all waters
        for idx in results[hoh_mask].index:
            ion_probs = probs[idx, ion_indices]
            if len(ion_probs) > 0 and ion_probs.max() > 0:
                # Reclassify to best ion
                best_ion_idx = ion_indices[np.argmax(ion_probs)]
                best_ion_class = classes[best_ion_idx]
                best_ion_prob = ion_probs.max()
                results.loc[idx, "pred_class"] = best_ion_class
                results.loc[idx, "confidence"] = best_ion_prob
                results.loc[idx, "resolution_filtered"] = True
            else:
                # No valid ion predictions, keep as HOH but flag it
                results.loc[idx, "resolution_filtered"] = True
    
    return results


def predict_water_identities(
    candidate_waters_pdb: str | Path,
    model_pdb: str | Path,
    model_checkpoint: str | Path,
    sites_csv: Optional[str | Path] = None,
    entry_resolution: Optional[float] = None,
    water_map: Optional[str | Path] = None,
    half1_map: Optional[str | Path] = None,
    half2_map: Optional[str | Path] = None,
    remove_hydrogens: bool = True,
    is_ensemble: bool = False,
    output_csv: Optional[str | Path] = None,
) -> pd.DataFrame:
    """Predict identities of water candidates using ML model.
    
    Args:
        candidate_waters_pdb: Path to candidate-waters.pdb
        model_pdb: Path to model PDB file (for feature extraction)
        model_checkpoint: Path to trained model checkpoint (.pt file) or ensemble directory
        sites_csv: Optional path to sites.csv (for additional features)
        entry_resolution: Optional resolution value for entry_res feature
        water_map: Optional path to water density map for density features
        half1_map: Optional path to half-map 1
        half2_map: Optional path to half-map 2
        remove_hydrogens: If True, filter out hydrogen atoms from model (default: True)
        is_ensemble: If True, treat model_checkpoint as ensemble directory
        output_csv: Optional path to write predictions CSV
        
    Returns:
        DataFrame with predictions (id, pred_class, confidence, p_<class> columns)
    """
    # Check if this is an ensemble
    if is_ensemble or Path(model_checkpoint).is_dir():
        # Extract features first, then use ensemble predictor
        # Load atom model for feature extraction
        atom_df = read_pdb_to_dataframe(str(model_pdb), remove_hydrogens=remove_hydrogens)
        kdtree, coords = build_kdtree(atom_df)
        
        # Load candidate waters
        waters_df = read_candidate_waters_to_dataframe(str(candidate_waters_pdb))
        
        # Load additional features from sites.csv if provided
        extra_cols = None
        if sites_csv and Path(sites_csv).exists():
            sites_df = pd.read_csv(sites_csv)
            sites_df = sites_df[sites_df["type"] == "water_candidate"].copy()
            if len(sites_df) > 0:
                if "id" in sites_df.columns:
                    if "id" in waters_df.columns:
                        waters_df = waters_df.merge(
                            sites_df[[c for c in sites_df.columns if c not in ["type", "center_x", "center_y", "center_z"]]],
                            on="id",
                            how="left",
                        )
                    else:
                        waters_df["id"] = sites_df["id"].values[:len(waters_df)] if len(sites_df) >= len(waters_df) else [f"W{i+1:05d}" for i in range(len(waters_df))]
                        waters_df = waters_df.merge(
                            sites_df[[c for c in sites_df.columns if c not in ["type", "center_x", "center_y", "center_z"]]],
                            on="id",
                            how="left",
                        )
                extra_cols = [
                    c
                    for c in sites_df.columns
                    if c not in ["id", "type", "center_x", "center_y", "center_z"]
                    and np.issubdtype(sites_df[c].dtype, np.number)
                ]
        
        # Load density maps if provided
        water_map_vol = None
        half1_vol = None
        half2_vol = None
        if water_map and Path(water_map).exists():
            water_map_vol = read_map(str(water_map))
        if half1_map and half2_map and Path(half1_map).exists() and Path(half2_map).exists():
            half1_vol = read_map(str(half1_map))
            half2_vol = read_map(str(half2_map))
        
        # Extract features for each candidate
        features = []
        for _, row in waters_df.iterrows():
            feat_dict = build_feature_row(row, atom_df, kdtree, coords, extra_cols=extra_cols, entry_resolution=entry_resolution)
            
            # Add density features if available
            point_xyzA = np.array([row["center_x"], row["center_y"], row["center_z"]], dtype=np.float32)
            if water_map_vol:
                density_feats = extract_density_features(point_xyzA, water_map_vol, radius_A=3.0)
                feat_dict.update(density_feats)
            
            # Add half-map features if available
            if half1_vol and half2_vol:
                half_feats = extract_halfmap_features(point_xyzA, half1_vol, half2_vol, radius_A=3.0)
                feat_dict.update(half_feats)
            
            features.append(feat_dict)
        
        feat_df = pd.DataFrame(features)
        
        # Add coordinates
        feat_df["x"] = waters_df["center_x"].values
        feat_df["y"] = waters_df["center_y"].values
        feat_df["z"] = waters_df["center_z"].values
        if "id" in waters_df.columns:
            feat_df["id"] = waters_df["id"].values
        
        # Use ensemble predictor
        ensemble_results = _predict_ensemble(
            features_csv=feat_df,
            ensemble_dir=Path(model_checkpoint),
            output_csv=None,  # Don't write yet, need to apply filtering
        )
        
        # Apply resolution-based filtering
        # Get classes from first model in ensemble
        model_dirs = sorted(Path(model_checkpoint).glob("model_*"))
        if model_dirs:
            ckpt = torch.load(str(model_dirs[0] / "model.pt"), map_location="cpu")
            classes = ckpt["classes"]
            # Reconstruct probabilities from ensemble results
            prob_cols = [f"p_{cls}" for cls in classes]
            probs = ensemble_results[prob_cols].values
            ensemble_results = apply_resolution_filtering(
                ensemble_results, entry_resolution, classes, probs
            )
        
        # Write output if requested
        if output_csv:
            ensemble_results.to_csv(output_csv, index=False)
        
        return ensemble_results
    
    # Single model prediction (existing code)
    # Load model checkpoint
    ck = torch.load(str(model_checkpoint), map_location="cpu")
    feat_cols = ck["feat_cols"]
    classes = ck["classes"]
    mean = np.array(ck["scaler_mean"])
    scale = np.array(ck["scaler_scale"])

    # Load atom model for feature extraction
    atom_df = read_pdb_to_dataframe(str(model_pdb), remove_hydrogens=remove_hydrogens)
    kdtree, coords = build_kdtree(atom_df)

    # Load candidate waters
    waters_df = read_candidate_waters_to_dataframe(str(candidate_waters_pdb))
    
    # Load additional features from sites.csv if provided
    extra_cols = None
    if sites_csv and Path(sites_csv).exists():
        sites_df = pd.read_csv(sites_csv)
        sites_df = sites_df[sites_df["type"] == "water_candidate"].copy()
        if len(sites_df) > 0:
            # Merge with waters_df by matching IDs or coordinates
            if "id" in sites_df.columns:
                # Merge by ID if available
                if "id" in waters_df.columns:
                    waters_df = waters_df.merge(
                        sites_df[[c for c in sites_df.columns if c not in ["type", "center_x", "center_y", "center_z"]]],
                        on="id",
                        how="left",
                    )
                else:
                    # Add IDs to waters_df based on order
                    waters_df["id"] = sites_df["id"].values[:len(waters_df)] if len(sites_df) >= len(waters_df) else [f"W{i+1:05d}" for i in range(len(waters_df))]
                    waters_df = waters_df.merge(
                        sites_df[[c for c in sites_df.columns if c not in ["type", "center_x", "center_y", "center_z"]]],
                        on="id",
                        how="left",
                    )
            # Use numeric columns from sites.csv as extra features
            extra_cols = [
                c
                for c in sites_df.columns
                if c not in ["id", "type", "center_x", "center_y", "center_z"]
                and np.issubdtype(sites_df[c].dtype, np.number)
            ]

    # Load density maps if provided
    water_map_vol = None
    half1_vol = None
    half2_vol = None
    if water_map and Path(water_map).exists():
        water_map_vol = read_map(str(water_map))
    if half1_map and half2_map and Path(half1_map).exists() and Path(half2_map).exists():
        half1_vol = read_map(str(half1_map))
        half2_vol = read_map(str(half2_map))
    
    # Extract features for each candidate
    features = []
    for _, row in waters_df.iterrows():
        feat_dict = build_feature_row(row, atom_df, kdtree, coords, extra_cols=extra_cols, entry_resolution=entry_resolution)
        
        # Add density features if available
        point_xyzA = np.array([row["center_x"], row["center_y"], row["center_z"]], dtype=np.float32)
        if water_map_vol:
            density_feats = extract_density_features(point_xyzA, water_map_vol, radius_A=3.0)
            feat_dict.update(density_feats)
        
        # Add half-map features if available
        if half1_vol and half2_vol:
            half_feats = extract_halfmap_features(point_xyzA, half1_vol, half2_vol, radius_A=3.0)
            feat_dict.update(half_feats)
        
        features.append(feat_dict)
    
    feat_df = pd.DataFrame(features)
    
    # Ensure all required feature columns are present
    missing_cols = set(feat_cols) - set(feat_df.columns)
    if missing_cols:
        raise ValueError(f"Missing feature columns: {missing_cols}. Available: {feat_df.columns.tolist()}")
    
    # Extract and scale features
    X = feat_df[feat_cols].to_numpy(dtype=np.float32)
    X = (X - mean) / scale
    
    # Replace inf/nan with reasonable defaults
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Load and run model
    model = IonWaterMLP(in_dim=X.shape[1], n_classes=len(classes))
    model.load_state_dict(ck["state_dict"])
    model.eval()

    with torch.no_grad():
        xb = torch.tensor(X, dtype=torch.float32)
        logits, aux = model(xb, calibrate=True)  # use calibrated temperature
        probs = torch.softmax(logits, dim=-1).numpy()
        aux_p = torch.softmax(aux, dim=-1).numpy()  # [:,1] = ion

    # Create output DataFrame
    top_idx = probs.argmax(axis=1)
    results = waters_df[["center_x", "center_y", "center_z"]].copy()
    if "id" in waters_df.columns:
        results["id"] = waters_df["id"]
    else:
        results["id"] = [f"W{i+1:05d}" for i in range(len(results))]
    
    results["pred_class"] = [classes[i] for i in top_idx]
    results["confidence"] = probs.max(axis=1)
    results["p_ion"] = aux_p[:, 1]
    
    # Add per-class probabilities
    for i, cls in enumerate(classes):
        results[f"p_{cls}"] = probs[:, i]
    
    # Apply resolution-based filtering
    results = apply_resolution_filtering(results, entry_resolution, classes, probs)
    
    # Write output if requested
    if output_csv:
        results.to_csv(output_csv, index=False)
    
    return results

