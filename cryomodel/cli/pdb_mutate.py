"""CLI: mutate PDB side chains from a target sequence (alignment or FASTA)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from cryomodel.mutate.engine import mutate_pdb

app = typer.Typer(no_args_is_help=True, help="Introduce mutations from sequence alignment")


@app.command("run")
def run_cmd(
    pdb: Path = typer.Argument(..., exists=True, help="Input PDB/mmCIF"),
    out: Path = typer.Argument(..., help="Output PDB path"),
    chain: str = typer.Option(..., "--chain", "-c", help="Chain ID (comma-separated for homomultimers)"),
    target_fasta: Optional[Path] = typer.Option(
        None,
        "--target-fasta",
        help="FASTA with a single target protein sequence",
    ),
    alignment_fasta: Optional[Path] = typer.Option(
        None,
        "--alignment-fasta",
        help=(
            "Multi-record FASTA: two equal-length rows (default first two records). "
            "Template vs target is auto-detected; template may be longer than the chain. "
            "Use --alignment-row-a / --alignment-row-b for other pairs in an MSA."
        ),
    ),
    alignment_row_a: int = typer.Option(
        0,
        "--alignment-row-a",
        min=0,
        help="0-based FASTA record index (file order) for the first alignment row",
    ),
    alignment_row_b: int = typer.Option(
        1,
        "--alignment-row-b",
        min=0,
        help="0-based FASTA record index for the second alignment row",
    ),
    map_path: Optional[Path] = typer.Option(None, "--map", help="Optional MRC/CCP4 map for rotamer scoring"),
    weight_rot: float = typer.Option(0.15, "--weight-rotamer", help="Weight for -log(rotamer prior)"),
    weight_map: float = typer.Option(0.5, "--weight-map", help="Weight for map density (if --map)"),
    density_sigma_k: float = typer.Option(
        1.0,
        "--density-sigma-k",
        help="For guide JSON: count side-chain atoms above map_mean + k*map_std (whole map stats)",
    ),
    json_log: Optional[Path] = typer.Option(None, "--json-log", help="Write mutation table JSON"),
):
    """Backbone-fixed mutations with χ1 rotamer choice (clash + prior + optional map)."""
    chains = [x.strip() for x in chain.split(",") if x.strip()]
    if not chains:
        raise typer.BadParameter("Provide at least one chain ID.")
    if (target_fasta is None) == (alignment_fasta is None):
        raise typer.BadParameter("Provide exactly one of --target-fasta or --alignment-fasta.")

    result = mutate_pdb(
        str(pdb),
        str(out),
        chains,
        target_fasta=str(target_fasta) if target_fasta else None,
        alignment_fasta=str(alignment_fasta) if alignment_fasta else None,
        alignment_row_a=alignment_row_a,
        alignment_row_b=alignment_row_b,
        map_path=str(map_path) if map_path else None,
        weight_rot=weight_rot,
        weight_map=weight_map,
        density_sigma_mult=density_sigma_k,
    )
    typer.echo(f"Wrote {out} ({len(result.mutations)} substitution(s))")
    if json_log:
        json_log = json_log.expanduser()
        json_log.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "mutations": result.mutations,
            "alignment": result.alignment,
            "map_guide_reference": result.map_guide_reference,
        }
        json_log.write_text(json.dumps(payload, indent=2))
        typer.echo(f"Log {json_log}")


if __name__ == "__main__":
    app()
