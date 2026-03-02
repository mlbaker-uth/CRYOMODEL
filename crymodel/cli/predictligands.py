# crymodel/cli/predictligands.py
"""CLI command for ligand identification."""
from __future__ import annotations
from pathlib import Path
import typer
import pandas as pd
import numpy as np

from ..finders.ligand_features import (
    read_ligand_components_pdb,
    extract_component_mask,
    compute_size_features,
    compute_peak_features,
    compute_environment_features,
)
from ..finders.ligand_grouping import group_near_contiguous_components
from ..finders.ligand_classifier import rule_based_classify, get_top_class, LIGAND_CLASSES
from ..io.mrc import read_map
from ..io.pdb import read_model_xyz
from ..ml.pdb_reader import read_pdb_to_dataframe

app = typer.Typer(no_args_is_help=True)


@app.command()
def predict(
    ligands_pdb: str = typer.Option(..., "--ligands-pdb", help="Ligand pseudoatoms PDB from findligands"),
    ligand_map: str = typer.Option(..., "--ligand-map", help="Ligand density map (.mrc)"),
    model: str = typer.Option(..., "--model", help="Model PDB (for environment features)"),
    sites_csv: str = typer.Option(None, "--sites-csv", help="Optional sites.csv from findligands"),
    entry_resolution: float = typer.Option(None, "--entry-resolution", help="Optional resolution (Å)"),
    keep_hydrogens: bool = typer.Option(False, "--keep-hydrogens", help="Keep hydrogen atoms in model"),
    min_pseudoatoms: int = typer.Option(3, "--min-pseudoatoms", help="Minimum pseudoatoms to keep a component"),
    max_pseudoatoms: int = typer.Option(80, "--max-pseudoatoms", help="Maximum pseudoatoms per grouped component (prevents over-grouping)"),
    threshold_original: float = typer.Option(0.5, "--threshold-original", help="Original threshold used for segmentation"),
    threshold_lower: float = typer.Option(0.3, "--threshold-lower", help="Lower threshold for checking continuity"),
    distance_threshold: float = typer.Option(5.0, "--distance-threshold", help="Max distance between components to consider grouping (Å)"),
    output_csv: str = typer.Option("ligand-predictions.csv", "--output-csv", help="Output predictions CSV"),
    out_dir: str = typer.Option("outputs", "--out-dir", help="Output directory"),
):
    """Identify ligand components using shape, environment, and template matching."""
    remove_hydrogens = not keep_hydrogens
    
    # Load inputs
    typer.echo("Loading inputs...")
    ligand_map_vol = read_map(ligand_map)
    model_xyz = read_model_xyz(model, remove_hydrogens=remove_hydrogens)
    model_df = read_pdb_to_dataframe(model, remove_hydrogens=remove_hydrogens)
    
    # Read ligand components
    components_raw = read_ligand_components_pdb(ligands_pdb)
    typer.echo(f"Found {len(components_raw)} ligand components (raw)")
    
    # Extract coordinates and chain IDs for grouping
    components_xyz = [xyz for _, xyz in components_raw]
    component_chain_ids = [chain_id for chain_id, _ in components_raw]
    
    # Group near-contiguous components
    typer.echo(f"Grouping near-contiguous components...")
    typer.echo(f"  Min pseudoatoms: {min_pseudoatoms}")
    typer.echo(f"  Max pseudoatoms: {max_pseudoatoms}")
    typer.echo(f"  Thresholds: original={threshold_original}, lower={threshold_lower}")
    typer.echo(f"  Distance threshold: {distance_threshold}Å")
    
    grouped_components_xyz, grouped_component_id_groups = group_near_contiguous_components(
        components_xyz,
        ligand_map_vol,
        threshold_original=threshold_original,
        threshold_lower=threshold_lower,
        min_pseudoatoms=min_pseudoatoms,
        max_pseudoatoms=max_pseudoatoms,
        distance_threshold_A=distance_threshold,
        component_ids=component_chain_ids,
    )
    
    typer.echo(f"Grouped into {len(grouped_components_xyz)} components after filtering and merging")
    
    # Load sites.csv if provided (for pre-computed features)
    sites_data = None
    if sites_csv and Path(sites_csv).exists():
        sites_df = pd.read_csv(sites_csv)
        sites_data = sites_df[sites_df["type"] == "ligand_component"].copy()
        typer.echo(f"Loaded {len(sites_data)} components from sites.csv")
    
    # Extract features for each grouped component
    results = []
    for comp_idx, (component_xyz, original_component_ids) in enumerate(
        zip(grouped_components_xyz, grouped_component_id_groups), start=1
    ):
        typer.echo(f"Processing component {comp_idx}/{len(grouped_components_xyz)} ({len(component_xyz)} pseudoatoms, from {len(original_component_ids)} original components)...")
        
        # Extract component mask
        try:
            component_mask, voxel_indices = extract_component_mask(component_xyz, ligand_map_vol)
        except Exception as e:
            typer.echo(f"  Warning: Failed to extract mask for component {comp_idx}: {e}", err=True)
            continue
        
        # Compute features
        size_feats = compute_size_features(component_xyz, component_mask, ligand_map_vol)
        peak_feats = compute_peak_features(component_mask, ligand_map_vol)
        env_feats = compute_environment_features(component_xyz, model_xyz, model_df)
        
        # Combine all features
        all_features = {**size_feats, **peak_feats, **env_feats}
        all_features["component_id"] = f"L{comp_idx:05d}"
        all_features["n_pseudoatoms_original"] = len(component_xyz)
        all_features["n_original_components"] = len(original_component_ids)
        all_features["original_component_ids"] = ",".join(original_component_ids)
        all_features["entry_resolution"] = entry_resolution if entry_resolution else np.nan
        
        # Add pre-computed features from sites.csv if available
        if sites_data is not None and len(sites_data) > 0:
            # Match by component index (assuming same order)
            if comp_idx <= len(sites_data):
                site_row = sites_data.iloc[comp_idx - 1]
                for col in ["nvox", "peak", "mean", "std_dev", "skewness", "elongation", "sphericity"]:
                    if col in site_row and pd.notna(site_row[col]):
                        all_features[f"site_{col}"] = float(site_row[col])
        
        # Rule-based classification
        class_scores = rule_based_classify(all_features)
        top_class, top_score, confidence = get_top_class(class_scores)
        
        # Create result row
        result = {
            "component_id": all_features["component_id"],
            "pred_class": top_class,
            "confidence": confidence,
            "pred_score": top_score,
            **all_features,
            **{f"score_{cls}": class_scores.get(cls, 0.0) for cls in LIGAND_CLASSES},
        }
        results.append(result)
    
    # Write results
    if results:
        results_df = pd.DataFrame(results)
        output_path = Path(out_dir) / output_csv
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(output_path, index=False)
        typer.echo(f"\nWrote {len(results)} predictions to {output_path}")
        
        # Print summary
        typer.echo("\nPrediction summary:")
        for cls in LIGAND_CLASSES:
            count = (results_df["pred_class"] == cls).sum()
            if count > 0:
                typer.echo(f"  {cls}: {count}")
        unknown_count = (results_df["pred_class"] == "unknown").sum()
        if unknown_count > 0:
            typer.echo(f"  unknown: {unknown_count}")
    else:
        typer.echo("No components processed!", err=True)


if __name__ == "__main__":
    app()

