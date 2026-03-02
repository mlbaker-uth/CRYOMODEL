# crymodel/cli/fitprep.py
"""CLI command for fitprep - preflight checker."""
from __future__ import annotations
from pathlib import Path
import typer

from ..prep.fitprep import check_map_model_alignment, generate_report, apply_fixes
from ..io.mrc import write_map
import gemmi
import numpy as np

app = typer.Typer(no_args_is_help=True)


@app.command()
def check(
    model: str = typer.Option(..., "--model", help="Input model PDB/mmCIF"),
    map: str = typer.Option(..., "--map", help="Full map (.mrc)"),
    half1: str = typer.Option(None, "--half1", help="Half-map 1 (.mrc)"),
    half2: str = typer.Option(None, "--half2", help="Half-map 2 (.mrc)"),
    out_dir: str = typer.Option("outputs", "--out-dir", help="Output directory"),
    apply: bool = typer.Option(False, "--apply", help="Apply suggested fixes"),
):
    """Preflight checker for maps & models."""
    model_path = Path(model)
    map_path = Path(map)
    
    if not model_path.exists():
        typer.echo(f"ERROR: Model not found: {model_path}", err=True)
        raise typer.Exit(1)
    if not map_path.exists():
        typer.echo(f"ERROR: Map not found: {map_path}", err=True)
        raise typer.Exit(1)
    
    # Run checks
    typer.echo("Running preflight checks...")
    results = check_map_model_alignment(str(map_path), str(model_path), half1, half2)
    
    # Write report
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    report_path = out_path / "fitprep_report.json"
    generate_report(results, report_path)
    typer.echo(f"Wrote: {report_path}")
    
    # Print summary
    typer.echo("\nPreflight Check Summary:")
    typer.echo("=" * 50)
    
    # Voxel grid
    vg = results['voxel_grid']
    typer.echo(f"Voxel size: {vg['voxel_size']:.3f} Å")
    typer.echo(f"Grid shape: {vg['grid_shape']}")
    
    # Origin alignment
    oa = results['origin_alignment']
    typer.echo(f"\nOrigin alignment:")
    typer.echo(f"  Model centroid: ({oa['model_centroid'][0]:.2f}, {oa['model_centroid'][1]:.2f}, {oa['model_centroid'][2]:.2f})")
    typer.echo(f"  Map origin: ({oa['map_origin'][0]:.2f}, {oa['map_origin'][1]:.2f}, {oa['map_origin'][2]:.2f})")
    typer.echo(f"  Offset: {np.linalg.norm(oa['offset']):.2f} Å")
    if 'suggested_shift' in oa:
        typer.echo(f"  Suggested shift: ({oa['suggested_shift'][0]:.2f}, {oa['suggested_shift'][1]:.2f}, {oa['suggested_shift'][2]:.2f})")
    
    # Intensity normalization
    inorm = results['intensity_normalization']
    typer.echo(f"\nIntensity normalization:")
    typer.echo(f"  Mean: {inorm['mean']:.4f}, Std: {inorm['std']:.4f}")
    
    # Quick fit
    qf = results['quick_fit']
    typer.echo(f"\nQuick fit metrics:")
    typer.echo(f"  CC_mask: {qf['cc_mask']:.3f}")
    typer.echo(f"  CC_box: {qf['cc_box']:.3f}")
    typer.echo(f"  ZNCC: {qf['zncc']:.3f}")
    
    # Warnings
    if results['all_warnings']:
        typer.echo(f"\n⚠️  Warnings ({len(results['all_warnings'])}):")
        for warning in results['all_warnings']:
            typer.echo(f"  - {warning}")
    
    # Apply fixes if requested
    if apply:
        typer.echo("\nApplying suggested fixes...")
        fixes = {}
        if 'suggested_shift' in oa and np.linalg.norm(oa['suggested_shift']) > 0.1:
            fixes['origin_shift'] = oa['suggested_shift']
        if 'suggested_normalization' in inorm:
            fixes['normalize'] = True
        
        if fixes:
            from ..io.mrc import read_map
            map_vol = read_map(str(map_path))
            structure = gemmi.read_structure(str(model_path))
            fixed_map, fixed_structure = apply_fixes(map_vol, structure, fixes)
            
            fixed_map_path = out_path / "map_fixed.mrc"
            fixed_model_path = out_path / "model_shifted.pdb"
            
            write_map(fixed_map, str(fixed_map_path))
            fixed_structure.write_minimal_pdb(str(fixed_model_path))
            
            typer.echo(f"Wrote: {fixed_map_path}")
            typer.echo(f"Wrote: {fixed_model_path}")
        else:
            typer.echo("No fixes to apply")


if __name__ == "__main__":
    app()

