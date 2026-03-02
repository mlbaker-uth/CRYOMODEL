# crymodel/cli/pathwalk.py
"""CLI command for pathwalking."""
from __future__ import annotations
from pathlib import Path
import typer
import numpy as np

from ..pathalker.pathwalker import pathwalk, write_path_pdb, write_path_pdb_with_probabilities
from ..pathalker.averaging import average_paths, compute_path_probabilities
from ..pathalker.pseudoatoms import PseudoatomMethod
from ..io.mrc import read_map

app = typer.Typer(no_args_is_help=True)


@app.command()
def walk(
    map: str = typer.Option(..., "--map", help="Input density map (.mrc/.map)"),
    threshold: float = typer.Option(..., "--threshold", help="Density threshold for pseudoatom generation"),
    n_residues: int = typer.Option(..., "--n-residues", help="Number of residues (C-alpha atoms)"),
    pseudoatom_method: PseudoatomMethod = typer.Option("kmeans", "--pseudoatom-method", help="Pseudoatom generation method (kmeans, sc, ac, ms, gmm, birch)"),
    map_weighted: bool = typer.Option(False, "--map-weighted", help="Use map-weighted distances"),
    tsp_solver: str = typer.Option("ortools", "--tsp-solver", help="TSP solver (ortools or lkh)"),
    time_limit: int = typer.Option(30, "--time-limit", help="Time limit for TSP solver (seconds)"),
    noise_level: float = typer.Option(0.0, "--noise-level", help="Noise level to add to pseudoatoms (Å)"),
    random_state: int = typer.Option(42, "--random-state", help="Random seed"),
    output_pdb: str = typer.Option("pathwalk.pdb", "--output-pdb", help="Output PDB file"),
    out_dir: str = typer.Option("outputs", "--out-dir", help="Output directory"),
):
    """Run pathwalking on a density map to find optimal protein path."""
    typer.echo("Pathwalking: Finding optimal protein path through density")
    typer.echo(f"  Map: {map}")
    typer.echo(f"  Threshold: {threshold}")
    typer.echo(f"  Number of residues: {n_residues}")
    typer.echo(f"  Pseudoatom method: {pseudoatom_method}")
    typer.echo(f"  Map-weighted: {map_weighted}")
    typer.echo(f"  TSP solver: {tsp_solver}")
    
    # Load map
    map_vol = read_map(map)
    
    # Run pathwalking
    path_coords, route, path_length = pathwalk(
        map_vol=map_vol,
        threshold=threshold,
        n_pseudoatoms=n_residues,
        pseudoatom_method=pseudoatom_method,
        map_weighted=map_weighted,
        tsp_solver=tsp_solver,
        time_limit_seconds=time_limit,
        noise_level=noise_level,
        random_state=random_state,
        verbose=True,
    )
    
    # Write output
    output_path = Path(out_dir) / output_pdb
    write_path_pdb(path_coords, output_path)
    
    typer.echo(f"\nPathwalking complete!")
    typer.echo(f"  Path length: {path_length:.2f} Å")
    typer.echo(f"  Output: {output_path}")


@app.command()
def average(
    path_files: str = typer.Option(..., "--path-files", help="Comma-separated list of path PDB files"),
    output_pdb: str = typer.Option("pathwalk_averaged.pdb", "--output-pdb", help="Output averaged PDB file"),
    probabilistic: bool = typer.Option(False, "--probabilistic", help="Write probabilities to B-factor column"),
    out_dir: str = typer.Option("outputs", "--out-dir", help="Output directory"),
):
    """Average multiple pathwalking runs."""
    # Parse path files
    path_file_list = [Path(f.strip()) for f in path_files.split(",")]
    
    typer.echo(f"Averaging {len(path_file_list)} pathwalking runs...")
    
    # Average paths
    averaged_coords, probabilities = average_paths(path_file_list)
    
    # Write output
    output_path = Path(out_dir) / output_pdb
    if probabilistic:
        write_path_pdb_with_probabilities(averaged_coords, probabilities, output_path)
        typer.echo(f"Averaged path with probabilities written to {output_path}")
    else:
        write_path_pdb(averaged_coords, output_path)
        typer.echo(f"Averaged path written to {output_path}")


if __name__ == "__main__":
    app()

