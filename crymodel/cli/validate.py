# crymodel/cli/validate.py
"""CLI command for fitcheck model validation."""
from __future__ import annotations
from pathlib import Path
import typer
import pandas as pd
import gemmi

from ..io.mrc import read_map, read_map_with_halves
from ..validation.feature_extractor import extract_residue_features
from ..validation.resolution_priors import fit_resolution_priors, compute_z_residuals, save_priors, load_priors

app = typer.Typer(no_args_is_help=True)


@app.command()
def validate(
    model: str = typer.Option(..., "--model", help="Input model PDB/mmCIF"),
    map: str = typer.Option(..., "--map", help="Full map (.mrc)"),
    half1: str = typer.Option(None, "--half1", help="Half-map 1 (.mrc)"),
    half2: str = typer.Option(None, "--half2", help="Half-map 2 (.mrc)"),
    localres: str = typer.Option(None, "--localres", help="Local resolution map (.mrc)"),
    out_dir: str = typer.Option("outputs", "--out-dir", help="Output directory"),
    priors: str = typer.Option(None, "--priors", help="Resolution priors YAML file (optional)"),
    fit_priors: bool = typer.Option(False, "--fit-priors", help="Fit priors from this data"),
    weights: str = typer.Option(None, "--weights", help="Model weights file (for future ML)"),
):
    """Validate cryoEM model with resolution-aware metrics."""
    typer.echo("FitCheck: Resolution-aware model validation")
    typer.echo(f"  Model: {model}")
    typer.echo(f"  Map: {map}")
    
    # Load structure
    structure = gemmi.read_structure(model)
    
    # Load maps
    if half1 and half2:
        map_vol = read_map_with_halves(map, half1, half2)
        half1_vol = map_vol.halfmaps[0] if map_vol.halfmaps else None
        half2_vol = map_vol.halfmaps[1] if map_vol.halfmaps else None
    else:
        map_vol = read_map(map)
        half1_vol = None
        half2_vol = None
    
    # Load local resolution map if provided
    local_res_map = None
    if localres:
        local_res_map = read_map(localres)
        typer.echo(f"  Local resolution map: {localres}")
    
    # Extract features
    typer.echo("Extracting features...")
    features_df = extract_residue_features(
        structure,
        map_vol,
        half1_vol,
        half2_vol,
        local_res_map,
    )
    
    typer.echo(f"  Extracted features for {len(features_df)} residues")
    
    # Fit or load priors
    if fit_priors:
        typer.echo("Fitting resolution priors...")
        priors_dict = fit_resolution_priors(features_df)
        priors_path = Path(out_dir) / "priors.yaml"
        save_priors(priors_dict, priors_path)
        typer.echo(f"  Saved priors to {priors_path}")
    elif priors:
        typer.echo(f"Loading priors from {priors}...")
        priors_dict = load_priors(Path(priors))
    else:
        priors_dict = {}
    
    # Compute Z-residuals if priors available
    if priors_dict:
        typer.echo("Computing Z-residuals...")
        features_df = compute_z_residuals(features_df, priors_dict)
    
    # Save features
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    features_csv = out_path / "features.csv"
    features_df.to_csv(features_csv, index=False)
    typer.echo(f"  Saved features to {features_csv}")
    
    # Compute summary statistics
    typer.echo("\nSummary statistics:")
    if 'local_res' in features_df.columns:
        typer.echo(f"  Mean local resolution: {features_df['local_res'].mean():.2f} Å")
        typer.echo(f"  Resolution range: {features_df['local_res'].min():.2f} - {features_df['local_res'].max():.2f} Å")
    
    if 'Q_mean' in features_df.columns:
        typer.echo(f"  Mean Q-score: {features_df['Q_mean'].mean():.3f}")
    
    if 'CC_mask' in features_df.columns:
        typer.echo(f"  Mean CC_mask: {features_df['CC_mask'].mean():.3f}")
    
    if 'ringer_Z' in features_df.columns:
        typer.echo(f"  Mean Ringer Z: {features_df['ringer_Z'].mean():.2f}")
    
    typer.echo(f"\nValidation complete! Results in {out_path}")


if __name__ == "__main__":
    app()

