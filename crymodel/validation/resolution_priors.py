# crymodel/validation/resolution_priors.py
"""Resolution-aware priors and calibration."""
from __future__ import annotations
import numpy as np
import yaml
from pathlib import Path
from typing import Dict, List, Optional
from scipy.stats import median_abs_deviation
import pandas as pd


def fit_resolution_priors(
    features_df: pd.DataFrame,
    resolution_bins: Optional[np.ndarray] = None,
    bin_size: float = 0.2,
    min_res: float = 2.0,
    max_res: float = 5.0,
) -> Dict:
    """Fit resolution-aware priors from feature data.
    
    Args:
        features_df: DataFrame with features and local_res column
        resolution_bins: Optional pre-defined bins
        bin_size: Size of resolution bins (Å)
        min_res: Minimum resolution (Å)
        max_res: Maximum resolution (Å)
        
    Returns:
        Dictionary with priors per bin
    """
    if resolution_bins is None:
        resolution_bins = np.arange(min_res, max_res + bin_size, bin_size)
    
    priors = {}
    
    # Feature columns to fit priors for
    feature_cols = [
        'ringer_Z', 'Q_mean', 'CC_mask', 'ZNCC',
        'ramachandran_prob', 'clashscore_z',
    ]
    
    for i in range(len(resolution_bins) - 1):
        bin_min = resolution_bins[i]
        bin_max = resolution_bins[i + 1]
        bin_key = f"{bin_min:.1f}-{bin_max:.1f}"
        
        # Filter data in this bin
        mask = (features_df['local_res'] >= bin_min) & (features_df['local_res'] < bin_max)
        bin_data = features_df[mask]
        
        if len(bin_data) == 0:
            continue
        
        priors[bin_key] = {}
        
        for col in feature_cols:
            if col not in bin_data.columns:
                continue
            
            values = bin_data[col].dropna()
            if len(values) == 0:
                continue
            
            # Robust statistics
            median = float(values.median())
            mad = float(median_abs_deviation(values, scale='normal'))
            mean = float(values.mean())
            std = float(values.std())
            
            priors[bin_key][col] = {
                'median': median,
                'mad': mad,
                'mean': mean,
                'std': std,
            }
    
    return priors


def compute_z_residuals(
    features_df: pd.DataFrame,
    priors: Dict,
) -> pd.DataFrame:
    """Convert raw features to Z-residuals based on resolution priors.
    
    Args:
        features_df: DataFrame with features
        priors: Prior dictionary from fit_resolution_priors
        
    Returns:
        DataFrame with Z-residual columns added
    """
    df = features_df.copy()
    
    # Feature columns
    feature_cols = [
        'ringer_Z', 'Q_mean', 'CC_mask', 'ZNCC',
        'ramachandran_prob', 'clashscore_z',
    ]
    
    for col in feature_cols:
        if col not in df.columns:
            continue
        
        z_residual_col = f"{col}_z_residual"
        z_residuals = []
        
        for _, row in df.iterrows():
            local_res = row.get('local_res', None)
            if local_res is None or pd.isna(local_res):
                z_residuals.append(0.0)
                continue
            
            # Find appropriate bin
            bin_key = None
            for key in priors.keys():
                bin_min, bin_max = map(float, key.split('-'))
                if bin_min <= local_res < bin_max:
                    bin_key = key
                    break
            
            if bin_key is None or col not in priors[bin_key]:
                z_residuals.append(0.0)
                continue
            
            # Compute Z-residual
            prior = priors[bin_key][col]
            raw_value = row[col]
            if pd.isna(raw_value):
                z_residuals.append(0.0)
                continue
            
            # Use median and MAD for robust Z-score
            median = prior['median']
            mad = prior['mad']
            
            if mad > 1e-6:
                z_residual = (raw_value - median) / mad
            else:
                z_residual = 0.0
            
            z_residuals.append(float(z_residual))
        
        df[z_residual_col] = z_residuals
    
    return df


def save_priors(priors: Dict, path: Path) -> None:
    """Save priors to YAML file."""
    with open(path, 'w') as f:
        yaml.dump(priors, f, default_flow_style=False)


def load_priors(path: Path) -> Dict:
    """Load priors from YAML file."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)

