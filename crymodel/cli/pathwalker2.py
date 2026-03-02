"""CLI entry point for Pathwalker2."""
from __future__ import annotations

from pathlib import Path
import typer

from ..pathwalker2 import Pathwalker2Config, run_pathwalker2
from ..pathwalker2.core import load_config_from_cli_kwargs

app = typer.Typer(no_args_is_help=True)


@app.command()
def discover(
    map: Path = typer.Option(..., "--map", help="Input density map (.mrc/.map)"),
    threshold: float = typer.Option(0.05, "--threshold", help="Density threshold"),
    n_residues: int = typer.Option(..., "--n-residues", help="Estimated number of residues"),
    out_dir: Path = typer.Option(Path("pw2_outputs"), "--out-dir", help="Output directory"),
    seed_method: str = typer.Option("ridge", "--seed-method", help="Seed method: ridge|grid"),
    grid_step: float = typer.Option(1.0, "--grid-step", help="Grid sampling step (Å)"),
    map_prep: str = typer.Option("locscale", "--map-prep", help="Map prep: locscale|gaussian|none"),
    gaussian_sigma: float = typer.Option(0.6, "--gaussian-sigma", help="Gaussian sigma when map-prep=gaussian"),
    k_neighbors: int = typer.Option(12, "--k-neighbors", help="kNN parameter for graph"),
    max_edge_length: float = typer.Option(5.5, "--max-edge-length", help="Maximum edge length (Å)"),
    fragments_per_partition: int = typer.Option(10, "--fragments-per-partition", help="Number of fragments to retain per partition"),
    random_state: int = typer.Option(0, "--random-state", help="Random seed"),
) -> None:
    """Run Pathwalker2 automatic trace discovery (Step 1)."""
    config = load_config_from_cli_kwargs(
        map=map,
        threshold=threshold,
        n_residues=n_residues,
        out_dir=out_dir,
        seed_method=seed_method,
        grid_step=grid_step,
        map_prep=map_prep,
        gaussian_sigma=gaussian_sigma,
        k_neighbors=k_neighbors,
        max_edge_length=max_edge_length,
        fragments_per_partition=fragments_per_partition,
        random_state=random_state,
    )

    typer.echo("Running Pathwalker2 Step 1 (trace discovery)...")
    outputs = run_pathwalker2(config)
    typer.echo("Completed Pathwalker2 trace discovery.")
    for key, path in outputs.items():
        typer.echo(f"  {key}: {path}")


if __name__ == "__main__":
    app()


