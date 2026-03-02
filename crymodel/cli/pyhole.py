# crymodel/cli/pyhole.py
"""CLI command for pyHole pore analysis."""
from __future__ import annotations
from pathlib import Path
import typer
import json

from ..pore.pyhole import (
    load_pdb_atoms,
    parse_residue_tokens,
    ca_positions_for,
    profile_along_axis,
    construct_centers_curved,
    profile_along_centers,
    write_csv,
    write_centerline_pdb,
    write_mesh_pdb,
    calculate_pore_statistics,
    vdw_radius,
)

app = typer.Typer(no_args_is_help=True)


@app.command()
def analyze(
    pdb: str = typer.Option(..., "--pdb", help="Input PDB file"),
    top: str = typer.Option("", "--top", help="Top residue selection (e.g., 'A:123' or 'A:123,B:456')"),
    bottom: str = typer.Option("", "--bottom", help="Bottom residue selection"),
    step: float = typer.Option(1.0, "--step", help="Step size along axis (Å)"),
    eps: float = typer.Option(0.25, "--eps", help="Contact epsilon (Å)"),
    no_h: bool = typer.Option(False, "--no-h", help="Exclude hydrogen atoms"),
    vdw_json: str = typer.Option("", "--vdw-json", help="Custom VDW radii JSON file"),
    rings: int = typer.Option(24, "--rings", help="Number of rings for mesh PDB"),
    out_prefix: str = typer.Option("pyhole_out", "--out-prefix", help="Output file prefix"),
    probe: float = typer.Option(0.0, "--probe", help="Probe radius for accessible volume (Å)"),
    conductivity: float = typer.Option(1.5, "--conductivity", help="Conductivity (S/m)"),
    occupancy: str = typer.Option("hydro", "--occupancy", help="Occupancy metric: 'hydro' or 'electro'"),
    hydro_scale: str = typer.Option("raw", "--hydro-scale", help="Hydrophobicity scale: 'raw' or '01'"),
    electro_scale: str = typer.Option("raw", "--electro-scale", help="Electrostatics scale: 'raw' or '01'"),
    passable_json: str = typer.Option("", "--passable-json", help="Passability radii JSON file"),
    centerline: str = typer.Option("straight", "--centerline", help="Centerline type: 'straight' or 'curved'"),
    adaptive: bool = typer.Option(False, "--adaptive", help="Use adaptive sampling"),
    slope_thresh: float = typer.Option(0.5, "--slope-thresh", help="Slope threshold for adaptive sampling"),
    max_refine: int = typer.Option(3, "--max-refine", help="Maximum refinement iterations"),
    curve_radius: float = typer.Option(2.0, "--curve-radius", help="Curve radius for curved centerline (Å)"),
    curve_iters: int = typer.Option(3, "--curve-iters", help="Curve iteration count"),
    interactive: bool = typer.Option(False, "--interactive", help="Interactive mode for top/bottom selection"),
):
    """Calculate pore profile through a structure."""
    pdb_path = Path(pdb)
    if not pdb_path.exists():
        typer.echo(f"ERROR: PDB not found: {pdb_path}", err=True)
        raise typer.Exit(2)
    
    # Interactive mode
    if interactive or (not top and not bottom):
        top = typer.prompt("Enter TOP residue selection")
        bottom = typer.prompt("Enter BOTTOM residue selection")
    
    # Load atoms
    atoms = load_pdb_atoms(str(pdb_path), include_h=not no_h)
    
    # Parse selections
    top_sel = parse_residue_tokens(top)
    bot_sel = parse_residue_tokens(bottom)
    
    # Get C-alpha positions
    ca_top = ca_positions_for(atoms, top_sel)
    ca_bot = ca_positions_for(atoms, bot_sel)
    
    if ca_top.size == 0 or ca_bot.size == 0:
        typer.echo("ERROR: Could not find CA atoms.", err=True)
        raise typer.Exit(3)
    
    c_top = ca_top.mean(axis=0)
    c_bot = ca_bot.mean(axis=0)
    
    # Load custom VDW radii
    custom_vdw = {}
    if vdw_json:
        with open(vdw_json, 'r') as jf:
            custom_vdw = json.load(jf)
    
    # Prepare atom data
    import numpy as np
    coords = np.array([[a.x, a.y, a.z] for a in atoms], dtype=float)
    radii = np.array([vdw_radius(a.element, custom_vdw) for a in atoms], dtype=float)
    metas = [(a.chain, a.resname, a.resi) for a in atoms]
    
    # Calculate profile
    if centerline == 'straight':
        rows, u, L = profile_along_axis(
            coords, radii, c_bot, c_top, step, eps, metas,
            adaptive=adaptive,
            slope_thresh=slope_thresh,
            max_refine=max_refine,
            hydro_scale=hydro_scale,
            electro_scale=electro_scale,
            occupancy_metric=occupancy,
        )
    else:
        centers, u, L = construct_centers_curved(
            coords, radii, c_bot, c_top, step, curve_radius, curve_iters
        )
        rows = profile_along_centers(
            coords, radii, centers, eps, metas, hydro_scale, electro_scale, occupancy
        )
    
    # Calculate statistics
    pass_radii = {}
    if passable_json:
        try:
            with open(passable_json, 'r') as pf:
                pass_radii = json.load(pf)
        except Exception:
            pass_radii = {}
    if not pass_radii:
        pass_radii = {'water': 1.4, 'na': 1.02, 'k': 1.38, 'ca': 1.00}
    
    stats = calculate_pore_statistics(rows, probe, conductivity, pass_radii)
    
    # Prepare output paths
    outprefix = Path(out_prefix)
    if outprefix.suffix.lower() == '.csv':
        outprefix = outprefix.with_suffix('')
    
    csv_path = outprefix.with_suffix('.csv')
    pdb_center = outprefix.with_name(outprefix.stem + '_centerline.pdb')
    pdb_mesh = outprefix.with_name(outprefix.stem + '_mesh.pdb')
    summary_path = outprefix.with_name(outprefix.stem + '_summary.json')
    
    # Write outputs
    write_csv(csv_path, rows)
    write_centerline_pdb(pdb_center, rows)
    write_mesh_pdb(pdb_mesh, axis_u=(c_top - c_bot), rows=rows, rings=rings)
    
    # Write summary JSON
    summary = {
        'pdb': str(pdb_path),
        'top': top,
        'bottom': bottom,
        'centerline': centerline,
        'step_A': float(step),
        'eps_A': float(eps),
        'probe_A': float(probe),
        'length_A': float(rows[-1]['s_A'] if rows else 0.0),
        'num_samples': len(rows),
        'occupancy_metric': occupancy,
        'hydroscale': hydro_scale,
        'electroscale': electro_scale,
        **stats,
    }
    
    with open(summary_path, 'w') as jf:
        json.dump(summary, jf, indent=2)
    
    typer.echo(f"Wrote: {csv_path}")
    typer.echo(f"Wrote: {pdb_center}")
    typer.echo(f"Wrote: {pdb_mesh}")
    typer.echo(f"Wrote: {summary_path}")


if __name__ == "__main__":
    app()

