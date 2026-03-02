# crymodel/ml/ensemble.py
"""Ensemble training and prediction utilities."""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Optional

from .model import IonWaterMLP
from .train import main as train_main


def train_ensemble(
    train_csv: str,
    outdir: str = "ensemble_model",
    n_models: int = 3,
    epochs: int = 50,
    batch: int = 512,
    lr: float = 2e-4,
    focal: bool = True,
    class_weights: bool = True,
    group_col: str = "pdb_id",
) -> None:
    """Train an ensemble of models with different random seeds.
    
    Args:
        train_csv: Path to training CSV
        outdir: Output directory for ensemble models
        n_models: Number of models in ensemble
        epochs: Training epochs per model
        batch: Batch size
        lr: Learning rate
        focal: Use focal loss
        class_weights: Use class weights
        group_col: Column for group-based splitting
    """
    os.makedirs(outdir, exist_ok=True)
    
    for i in range(n_models):
        model_dir = os.path.join(outdir, f"model_{i+1}")
        print(f"\n{'='*60}")
        print(f"Training model {i+1}/{n_models}")
        print(f"{'='*60}")
        
        # Set random seed for reproducibility but different per model
        seed = 42 + i
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        train_main(
            train_csv=train_csv,
            outdir=model_dir,
            epochs=epochs,
            batch=batch,
            lr=lr,
            focal=focal,
            class_weights=class_weights,
            group_col=group_col,
        )
    
    print(f"\n{'='*60}")
    print(f"Ensemble training complete!")
    print(f"Models saved in: {outdir}")
    print(f"{'='*60}")


def predict_ensemble(
    features_csv: str | Path | pd.DataFrame,
    ensemble_dir: str | Path,
    output_csv: Optional[str | Path] = None,
) -> pd.DataFrame:
    """Predict using ensemble of models (average predictions).
    
    Args:
        features_csv: Path to features CSV or DataFrame
        ensemble_dir: Directory containing ensemble model subdirectories
        output_csv: Optional path to save predictions
        
    Returns:
        DataFrame with ensemble predictions
    """
    ensemble_dir = Path(ensemble_dir)
    model_dirs = sorted(ensemble_dir.glob("model_*"))
    
    if not model_dirs:
        raise ValueError(f"No model directories found in {ensemble_dir}")
    
    print(f"Loading ensemble of {len(model_dirs)} models...")
    
    # Load data
    if isinstance(features_csv, pd.DataFrame):
        df = features_csv
    else:
        df = pd.read_csv(features_csv)
    
    # Extract feature columns (exclude metadata columns)
    drop_cols = {"label", "x", "y", "z", "pdb_id", "resname", "chain", "resi", "id", "center_x", "center_y", "center_z"}
    feat_cols = [c for c in df.columns if c not in drop_cols and pd.api.types.is_numeric_dtype(df[c])]
    
    # Replace inf/nan in features
    for col in feat_cols:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        df[col] = df[col].fillna(0.0)
    
    all_probs = []
    all_aux_probs = []
    
    for model_dir in model_dirs:
        model_path = model_dir / "model.pt"
        if not model_path.exists():
            print(f"Warning: {model_path} not found, skipping")
            continue
        
        ckpt = torch.load(str(model_path), map_location="cpu")
        feat_cols_model = ckpt["feat_cols"]
        classes = ckpt["classes"]
        mean = np.array(ckpt["scaler_mean"])
        scale = np.array(ckpt["scaler_scale"])
        
        # Check which features are available and which are missing
        missing_features = set(feat_cols_model) - set(df.columns)
        if missing_features:
            print(f"Warning: Missing features in {model_dir}: {missing_features}")
            # Fill missing features with 0 (or NaN if appropriate)
            for feat in missing_features:
                df[feat] = 0.0
        
        # Only use features that the model expects
        available_feat_cols = [f for f in feat_cols_model if f in df.columns]
        if len(available_feat_cols) != len(feat_cols_model):
            print(f"Warning: Only {len(available_feat_cols)}/{len(feat_cols_model)} features available in {model_dir}")
            continue
        
        # Extract and scale features (in the order the model expects)
        X_model = df[feat_cols_model].to_numpy()
        X_model = np.nan_to_num(X_model, nan=0.0, posinf=1e6, neginf=-1e6)
        Xs = (X_model - mean) / scale
        Xs = np.nan_to_num(Xs, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Load model and predict
        model = IonWaterMLP(in_dim=Xs.shape[1], n_classes=len(classes))
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        
        with torch.no_grad():
            xb = torch.tensor(Xs, dtype=torch.float32)
            logits, aux = model(xb, calibrate=True)
            probs = torch.softmax(logits, dim=-1)
            aux_p = torch.softmax(aux, dim=-1)
            
            # Convert to numpy safely
            try:
                probs_np = probs.cpu().detach().numpy()
                aux_p_np = aux_p.cpu().detach().numpy()
            except RuntimeError:
                # Fallback
                probs_np = np.array([[float(p.item()) for p in row] for row in probs.cpu().detach()])
                aux_p_np = np.array([[float(p.item()) for p in row] for row in aux_p.cpu().detach()])
        
        all_probs.append(probs_np)
        all_aux_probs.append(aux_p_np)
    
    if not all_probs:
        raise ValueError("No valid models found in ensemble")
    
    # Average predictions
    avg_probs = np.mean(all_probs, axis=0)
    avg_aux = np.mean(all_aux_probs, axis=0)
    
    # Create output DataFrame
    top_idx = avg_probs.argmax(axis=1)
    
    # Get coordinates (handle both x/y/z and center_x/center_y/center_z)
    if "x" in df.columns:
        results = df[["x", "y", "z"]].copy()
    elif "center_x" in df.columns:
        results = pd.DataFrame({
            "center_x": df["center_x"].values,
            "center_y": df["center_y"].values,
            "center_z": df["center_z"].values,
        })
    else:
        results = pd.DataFrame()
    
    if "id" in df.columns:
        results["id"] = df["id"]
    elif "pdb_id" in df.columns:
        results["id"] = df["pdb_id"]
    else:
        results["id"] = [f"W{i+1:05d}" for i in range(len(results))]
    
    results["pred_class"] = [classes[i] for i in top_idx]
    results["confidence"] = avg_probs.max(axis=1)
    results["p_ion"] = avg_aux[:, 1]
    
    # Add per-class probabilities
    for i, cls in enumerate(classes):
        results[f"p_{cls}"] = avg_probs[:, i]
    
    # Add ensemble info
    results["n_models"] = len(all_probs)
    
    # Note: Resolution filtering should be applied by the caller (predict_water_identities)
    # since it needs the entry_resolution parameter
    
    if output_csv:
        results.to_csv(output_csv, index=False)
        print(f"Ensemble predictions saved to {output_csv}")
    
    return results

