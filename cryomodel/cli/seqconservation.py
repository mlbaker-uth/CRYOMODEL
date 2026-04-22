"""CLI for mapping alignment conservation onto PDB residues."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from ..conservation import compute_conservation
from ..conservation_diffusion import run_conservation_diffusion


def seqconservation(
    pdb: Path = typer.Argument(..., exists=True, help="Input PDB/mmCIF"),
    alignment: Path = typer.Argument(..., exists=True, help="MSA FASTA (first seq = PDB reference)"),
    chain: str = typer.Option(
        ...,
        "--chain",
        "--chains",
        "-c",
        help="Chain ID(s), comma-separated for identical homomultimers (e.g. A,B,C,D)",
    ),
    out_csv: Path = typer.Option(..., "--out-csv", help="Per-residue metrics CSV"),
    out_json: Optional[Path] = typer.Option(None, "--out-json", help="Optional JSON output"),
    out_pdb: Optional[Path] = typer.Option(
        None, "--out-pdb", help="Optional PDB with selected metric in B-factor"
    ),
    bfactor_metric: str = typer.Option(
        "n_aa_types",
        "--bfactor-metric",
        help="Metric to write to B-factor if --out-pdb is set",
    ),
    occupancy_metric: Optional[str] = typer.Option(
        None,
        "--occupancy-metric",
        help="Optional metric to write to occupancy if --out-pdb is set",
    ),
    include_reference_in_stats: bool = typer.Option(
        False,
        "--include-reference-in-stats/--exclude-reference-in-stats",
        help="Include the first (reference) sequence in frequency statistics",
    ),
):
    """
    Map MSA conservation to PDB residues.

    Assumption: the first FASTA record is the **reference** for the chain(s). It may be longer
    than the coordinate sequence (extra termini are skipped like pdb-mutate) and may contain
    ``'-'`` gaps. Comma-separated chains must have identical polymer sequences (homomultimer).
    """
    result = compute_conservation(
        pdb_path=pdb,
        chains=chain,
        alignment_fasta=alignment,
        out_csv=out_csv,
        out_json=out_json,
        out_pdb=out_pdb,
        bfactor_metric=bfactor_metric,
        occupancy_metric=occupancy_metric,
        include_reference_in_stats=include_reference_in_stats,
    )
    typer.echo(f"Wrote CSV: {result.out_csv}")
    if result.out_json is not None:
        typer.echo(f"Wrote JSON: {result.out_json}")
    if result.out_pdb is not None:
        typer.echo(f"Wrote PDB: {result.out_pdb}")


def seqconservation_diffuse(
    pdb: Path = typer.Argument(..., exists=True, help="Input PDB/mmCIF"),
    alignment: Path = typer.Argument(..., exists=True, help="MSA FASTA (first seq = reference)"),
    chain: str = typer.Option(
        ...,
        "--chain",
        "--chains",
        "-c",
        help="Chain ID(s), comma-separated; all are nodes in one Cα graph (homomultimer)",
    ),
    out_csv: Path = typer.Option(..., "--out-csv", help="Per-residue CSV (+ diffusion columns)"),
    out_json: Optional[Path] = typer.Option(None, "--out-json", help="Optional JSON (meta + rows)"),
    out_pdb: Optional[Path] = typer.Option(
        None, "--out-pdb", help="Optional PDB: B-factor from diffused_score or seed_signal"
    ),
    include_reference_in_stats: bool = typer.Option(
        False,
        "--include-reference-in-stats/--exclude-reference-in-stats",
        help="Include first MSA row in frequency statistics",
    ),
    seed_metric: str = typer.Option(
        "p_nonref",
        "--seed-metric",
        help=(
            "Primitive: p_nonref, n_aa_types, entropy, mean_penalty, frac_nonconservative, p_gap. "
            "Composites: composite_nonref_penalty, composite_entropy_noncons, composite_diversity_penalty."
        ),
    ),
    seed_threshold: float = typer.Option(
        0.0,
        "--seed-threshold",
        help="Subtract from seed before diffusion (clamp at 0); higher = fewer seed sources",
    ),
    contact_radius: float = typer.Option(
        10.0,
        "--contact-radius",
        help="Cα–Cα edge if distance ≤ this (Å); links subunits in 3D",
    ),
    falloff_angstrom: float = typer.Option(
        3.0,
        "--falloff-angstrom",
        help="Soft edge weight exp(-d/d0); smaller d0 = shorter range",
    ),
    diffusion_steps: int = typer.Option(
        24,
        "--diffusion-steps",
        min=1,
        help="Relaxation iterations toward neighbor average",
    ),
    mix: float = typer.Option(
        0.4,
        "--mix",
        min=0.0,
        max=1.0,
        help="Blend toward neighbor mean each step (higher = faster spatial spread)",
    ),
    peak_min: float = typer.Option(
        0.02,
        "--peak-min",
        help="Local maxima in diffused field must exceed this to count as a peak",
    ),
    basin_mode: str = typer.Option(
        "nearest_peak",
        "--basin-mode",
        help="none | nearest_peak (3D distance / peak strength)",
    ),
    peak_weight_gamma: float = typer.Option(
        0.5,
        "--peak-weight-gamma",
        help="Basin assignment: score = distance / (peak_value^gamma)",
    ),
    bfactor_writes: str = typer.Option(
        "diffused_score",
        "--bfactor-writes",
        help="diffused_score | seed_signal for --out-pdb B-factors",
    ),
):
    """
    Tier 3: diffuse conservation variability on a **3D Cα graph** over all selected chains.

    High ``seed_metric`` sites seed the field; iterative mixing spreads signal across
    contacts (including inter-chain), surfacing spatial clusters of variation.
    """
    result = run_conservation_diffusion(
        pdb_path=pdb,
        chains=chain,
        alignment_fasta=alignment,
        out_csv=out_csv,
        out_json=out_json,
        out_pdb=out_pdb,
        include_reference_in_stats=include_reference_in_stats,
        seed_metric=seed_metric,
        seed_threshold=seed_threshold,
        contact_radius=contact_radius,
        falloff_angstrom=falloff_angstrom,
        diffusion_steps=diffusion_steps,
        mix=mix,
        peak_min=peak_min,
        basin_mode=basin_mode,
        peak_weight_gamma=peak_weight_gamma,
        bfactor_writes=bfactor_writes,
    )
    typer.echo(f"Wrote CSV: {result.out_csv}")
    if result.out_json is not None:
        typer.echo(f"Wrote JSON: {result.out_json}")
    if result.out_pdb is not None:
        typer.echo(f"Wrote PDB: {result.out_pdb}")

