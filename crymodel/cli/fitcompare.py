# crymodel/cli/fitcompare.py
"""CLI command for fitcompare - align & compare models."""
from __future__ import annotations
from pathlib import Path
import typer
import gemmi

from ..compare.fitcompare import (
    parse_anchor_selection,
    compare_models,
)

app = typer.Typer(no_args_is_help=True)


@app.command()
def compare(
    model_a: str = typer.Option(..., "--model-a", help="Reference model PDB/mmCIF"),
    model_b: str = typer.Option(..., "--model-b", help="Model to align PDB/mmCIF"),
    out_dir: str = typer.Option("outputs", "--out-dir", help="Output directory"),
    anchors: str = typer.Option(None, "--anchors", help="Anchor selection (e.g., 'A:100-160,B:20-45')"),
    domains: str = typer.Option(None, "--domains", help="Domain specification JSON (for domain summaries)"),
):
    """Align and compare two models."""
    model_a_path = Path(model_a)
    model_b_path = Path(model_b)
    
    if not model_a_path.exists():
        typer.echo(f"ERROR: Model A not found: {model_a_path}", err=True)
        raise typer.Exit(1)
    if not model_b_path.exists():
        typer.echo(f"ERROR: Model B not found: {model_b_path}", err=True)
        raise typer.Exit(1)
    
    # Load structures
    structure_A = gemmi.read_structure(str(model_a_path))
    structure_B = gemmi.read_structure(str(model_b_path))
    
    # Parse anchors
    anchor_list = None
    if anchors:
        anchor_list = parse_anchor_selection(anchors)
        typer.echo(f"Using anchor selection: {anchors}")
    
    # Compare models
    typer.echo("Aligning and comparing models...")
    results = compare_models(structure_A, structure_B, anchor_list)
    
    # Write outputs
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Write aligned structure
    aligned_path = out_path / "fitcompare_superposed.pdb"
    results['aligned_structure'].write_minimal_pdb(str(aligned_path))
    typer.echo(f"Wrote: {aligned_path}")
    
    # Write per-residue RMSD
    rmsd_path = out_path / "per_residue_deltas.csv"
    results['rmsd_df'].to_csv(rmsd_path, index=False)
    typer.echo(f"Wrote: {rmsd_path}")
    
    # Print summary
    summary = results['summary']
    typer.echo("\nSummary:")
    typer.echo(f"  Cα RMSD (mean): {summary['ca_rmsd_mean']:.3f} Å")
    typer.echo(f"  Cα RMSD (max): {summary['ca_rmsd_max']:.3f} Å")
    typer.echo(f"  Backbone RMSD (mean): {summary['backbone_rmsd_mean']:.3f} Å")
    typer.echo(f"  Sidechain RMSD (mean): {summary['sidechain_rmsd_mean']:.3f} Å")
    typer.echo(f"  Residues compared: {summary['num_residues']}")


if __name__ == "__main__":
    app()

