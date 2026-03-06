# crymodel/cli/basehunter.py
"""CLI command for BaseHunter nucleotide density comparison."""
from __future__ import annotations
from pathlib import Path
from typing import Optional
import typer
import os

from ..nucleotide.basehunter_enhanced import classify_base_pairs

app = typer.Typer(no_args_is_help=True)

DEFAULT_TEMPLATE_DIR = Path(__file__).resolve().parents[2] / "data" / "DNA-TEMPLATES"


@app.command()
def compare(
    input_file: str = typer.Option(..., "--input-file", help="Input file with volume directory and pairs"),
    template_dir: str = typer.Option(
        str(DEFAULT_TEMPLATE_DIR),
        "--template-dir",
        help="Directory containing purine/pyrimidine templates",
    ),
    threshold: Optional[float] = typer.Option(None, "--threshold", help="Density threshold value (prompted if missing)"),
    target_resolution: Optional[float] = typer.Option(None, "--resolution", help="Target resolution in Å (prompted if missing)"),
    alignment_threshold: float = typer.Option(0.3, "--alignment-threshold", help="Minimum correlation for classification"),
    use_bootstrap: bool = typer.Option(True, "--bootstrap/--no-bootstrap", help="Perform bootstrap analysis for likelihoods"),
    n_bootstrap: int = typer.Option(100, "--n-bootstrap", help="Number of bootstrap iterations"),
    use_emd: bool = typer.Option(True, "--emd/--no-emd", help="Use Earth Mover's Distance for additional discrimination"),
    emd_weight: float = typer.Option(0.2, "--emd-weight", help="Weight for EMD in combined score (0.0-1.0, default 0.2)"),
    pair_mismatch_penalty: float = typer.Option(
        0.1,
        "--pair-mismatch-penalty",
        help="Penalty for same-class base pairs (0.0=hard constraint, 1.0=no penalty)",
    ),
    template_mask_threshold: Optional[float] = typer.Option(
        None,
        "--template-mask-threshold",
        help="Threshold for template-derived mask to reduce backbone influence",
    ),
    center_sphere_radius: Optional[float] = typer.Option(
        None,
        "--center-sphere-radius",
        help="Spherical mask radius (vox) around box center",
    ),
    score_on_raw: bool = typer.Option(
        False,
        "--score-on-raw/--score-on-thresholded",
        help="Score using raw volumes while alignment uses thresholded volumes",
    ),
    alignment_reference_pdb: Optional[str] = typer.Option(
        None,
        "--alignment-reference-pdb",
        help="Reference PDB for alignment (base-only template coordinates)",
    ),
    alignment_reference_atom_radius_vox: float = typer.Option(
        1.5,
        "--alignment-reference-atom-radius-vox",
        help="Atom mask radius in voxels for reference alignment",
    ),
    alignment_on_raw: bool = typer.Option(
        False,
        "--alignment-on-raw/--alignment-on-thresholded",
        help="Align using raw volumes instead of thresholded",
    ),
    output_reference_models: bool = typer.Option(
        False,
        "--output-reference-models/--no-output-reference-models",
        help="Write aligned reference PDB per input volume",
    ),
    reference_model_output_dir: Optional[str] = typer.Option(
        None,
        "--reference-model-output-dir",
        help="Output directory for aligned reference PDBs",
    ),
    apply_alignment: bool = typer.Option(
        True,
        "--apply-alignment/--no-alignment",
        help="Apply alignment before scoring",
    ),
    template_threshold: Optional[float] = typer.Option(
        None,
        "--template-threshold",
        help="Threshold for template maps (zero below)",
    ),
    backbone_mask_mrc: Optional[str] = typer.Option(
        None,
        "--backbone-mask-mrc",
        help="Backbone mask MRC to exclude sugar/phosphate",
    ),
    backbone_mask_threshold: Optional[float] = typer.Option(
        None,
        "--backbone-mask-threshold",
        help="Threshold for backbone mask MRC",
    ),
    out_dir: str = typer.Option("basehunter_outputs", "--out-dir", help="Output directory for results"),
):
    """
    Classify base pairs using template-based classification.
    
    Uses purine/pyrimidine templates to classify nucleotide density volumes.
    Enforces base pair constraints (1 purine + 1 pyrimidine per pair).
    """
    input_path = Path(input_file)
    if not input_path.exists():
        typer.echo(f"ERROR: Input file not found: {input_path}", err=True)
        raise typer.Exit(1)
    
    template_path = Path(template_dir)
    if not template_path.exists():
        typer.echo(f"ERROR: Template directory not found: {template_path}", err=True)
        raise typer.Exit(1)
    
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Prompt for required parameters if missing
    if target_resolution is None:
        target_resolution = typer.prompt("Target resolution (Å)", type=float)
    if threshold is None:
        threshold = typer.prompt("Default threshold", type=float)
    
    # Read input file
    with open(input_path, "r") as f:
        lines = f.readlines()
    
    # Filter non-empty, non-comment lines
    data_lines = [line.strip() for line in lines if line.strip() and not line.startswith("#")]
    if not data_lines:
        typer.echo("ERROR: Input file contains no data lines.", err=True)
        raise typer.Exit(1)
    
    # Support two formats:
    # 1) First line is a directory, remaining lines are pairs
    # 2) All lines are pairs (no directory)
    first_parts = data_lines[0].split()
    volume_dir = ""
    pair_lines = data_lines
    
    if len(first_parts) == 1 and Path(first_parts[0]).is_dir():
        volume_dir = first_parts[0]
        pair_lines = data_lines[1:]
    
    # Resolve volume_dir relative to input file location if needed
    if volume_dir and not Path(volume_dir).is_absolute():
        volume_dir = str((input_path.parent / volume_dir).resolve())
    
    volume_pairs = []
    volume_thresholds = {}
    
    for line in pair_lines:
        parts = line.split()
        if len(parts) >= 2:
            vol1_path = parts[0] if not volume_dir else os.path.join(volume_dir, parts[0])
            vol2_path = parts[1] if not volume_dir else os.path.join(volume_dir, parts[1])
            
            # Resolve relative paths relative to input file if no volume_dir
            if not volume_dir:
                if not Path(vol1_path).is_absolute():
                    vol1_path = str((input_path.parent / vol1_path).resolve())
                if not Path(vol2_path).is_absolute():
                    vol2_path = str((input_path.parent / vol2_path).resolve())
            
            volume_pairs.append((vol1_path, vol2_path))
            
            # Optional per-pair threshold: "A.mrc B.mrc 0.67"
            if len(parts) >= 3:
                try:
                    per_pair_threshold = float(parts[2])
                    for vol_path in (vol1_path, vol2_path):
                        if vol_path in volume_thresholds and abs(volume_thresholds[vol_path] - per_pair_threshold) > 1e-6:
                            typer.echo(
                                f"WARNING: Conflicting thresholds for {vol_path}. "
                                f"Keeping {volume_thresholds[vol_path]:.3f}, ignoring {per_pair_threshold:.3f}."
                            )
                        else:
                            volume_thresholds[vol_path] = per_pair_threshold
                except ValueError:
                    typer.echo(f"WARNING: Invalid threshold '{parts[2]}' in line: {line.strip()}")
    
    typer.echo(f"Found {len(volume_pairs)} volume pairs")
    typer.echo(f"Template directory: {template_path}")
    
    # Classify base pairs
    try:
        df = classify_base_pairs(
            volume_pairs=volume_pairs,
            template_dir=template_path,
            threshold=threshold,
            target_resolution=target_resolution,
            alignment_threshold=alignment_threshold,
            use_bootstrap=use_bootstrap,
            n_bootstrap=n_bootstrap,
            use_emd=use_emd,
            emd_weight=emd_weight,
            pair_mismatch_penalty=pair_mismatch_penalty,
            template_mask_threshold=template_mask_threshold,
            center_sphere_radius=center_sphere_radius,
            score_on_raw=score_on_raw,
            alignment_reference_pdb=Path(alignment_reference_pdb) if alignment_reference_pdb else None,
            alignment_reference_atom_radius_vox=alignment_reference_atom_radius_vox,
            alignment_on_raw=alignment_on_raw,
            output_reference_models=output_reference_models,
            reference_model_output_dir=Path(reference_model_output_dir) if reference_model_output_dir else None,
            apply_alignment=apply_alignment,
            template_threshold=template_threshold,
            backbone_mask_mrc=Path(backbone_mask_mrc) if backbone_mask_mrc else None,
            backbone_mask_threshold=backbone_mask_threshold,
            output_dir=out_path,
            volume_thresholds=volume_thresholds if volume_thresholds else None,
        )
        
        typer.echo(f"\n✓ Classification complete!")
        typer.echo(f"  Purine assignments: {df.attrs.get('n_purine', 0)}")
        typer.echo(f"  Pyrimidine assignments: {df.attrs.get('n_pyrimidine', 0)}")
        typer.echo(f"  Unclassified: {df.attrs.get('n_unclassified', 0)}")
        typer.echo(f"  Valid base pairs: {df['base_pair_valid'].sum()}/{len(df)}")
        
        # Text summary output (U=purine, Y=pyrimidine)
        def _class_to_letter(class_name: str) -> str:
            if class_name == "purine":
                return "U"
            if class_name == "pyrimidine":
                return "Y"
            return "?"
        
        summary_lines = []
        for _, row in df.iterrows():
            v1 = row["volume1"]
            v2 = row["volume2"]
            c1 = row["volume1_class"]
            c2 = row["volume2_class"]
            
            # Prefer bootstrap likelihoods when available
            use_likelihoods = use_bootstrap and (
                row.get("volume1_purine_likelihood", 0.0) > 0.0 or
                row.get("volume1_pyrimidine_likelihood", 0.0) > 0.0
            )
            
            if use_likelihoods:
                score1 = row.get(f"volume1_{c1}_likelihood", 0.0)
                score2 = row.get(f"volume2_{c2}_likelihood", 0.0)
            else:
                score1 = row.get("volume1_confidence", 0.0)
                score2 = row.get("volume2_confidence", 0.0)
            
            composite = (score1 + score2) / 2.0
            pair_conf = row.get("pair_assignment_confidence", 0.0)
            pair_p = row.get("pair_assignment_p_value", 1.0)
            summary_lines.append(
                f"{v1} - {_class_to_letter(c1)} {v2} - {_class_to_letter(c2)} "
                f"Likelihood: {composite:.2f} ({score1:.2f}, {score2:.2f}) "
                f"PairConf: {pair_conf:.2f} PairP: {pair_p:.3f}"
            )
        
        summary_path = out_path / "basehunter_assignments.txt"
        with open(summary_path, "w") as f:
            f.write("\n".join(summary_lines) + "\n")
        
        typer.echo("\nAssignments:")
        for line in summary_lines:
            typer.echo(f"  {line}")
        
        typer.echo(f"\nResults written to: {out_path}")
        
    except Exception as e:
        typer.echo(f"ERROR: {e}", err=True)
        import traceback
        traceback.print_exc()
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
