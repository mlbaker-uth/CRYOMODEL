# cryomodel/cli/validate.py
"""CLI command for fitcheck model validation."""
from __future__ import annotations
from pathlib import Path
from typing import Optional

import gemmi
import numpy as np
import pandas as pd
import typer

from ..io.map_resample import maps_grid_compatible, resample_map_volume
from ..io.mrc import read_map
from ..io.structure_filter import filter_protein_only
from ..validation.composite_score import (
    add_composite_quality_columns,
    metric_higher_is_worse,
    resolve_bfactor_column,
    write_structure_with_composite_bfactors,
)
from ..validation.benchmark_priors import priors_dict_from_benchmark
from ..validation.residue_report import resolve_validate_score_column, write_residue_report
from ..validation.feature_extractor import extract_residue_features
from ..validation.resolution_priors import (
    compute_z_residuals,
    fit_resolution_priors,
    load_priors,
    merge_resolution_priors,
    save_priors,
)

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
    protein_only: bool = typer.Option(
        False,
        "--protein-only",
        help="Validate protein residues only (exclude waters, ligands, nucleic acids)",
    ),
    bfactor_pdb: str = typer.Option(
        None,
        "--bfactor-pdb",
        help="Write model with validation scores in the B-factor column (see --bfactor-color)",
    ),
    bfactor_color: str = typer.Option(
        "auto",
        "--bfactor-color",
        help="auto=band |z| vs resolution priors if available else percentile badness; band|badness|quality",
    ),
    bfactor_threshold: Optional[float] = typer.Option(
        None,
        "--bfactor-threshold",
        help="Hard cutoff (0–100): pass occupancy 1.0, fail 0.35 (band/badness: fail if value > T; quality: fail if < T). Adds bfactor_flagged to CSV.",
    ),
    bfactor_fail_occupancy: float = typer.Option(
        0.35,
        "--bfactor-fail-occupancy",
        help="Occupancy for residues that fail --bfactor-threshold",
    ),
    benchmark_priors: Optional[str] = typer.Option(
        None,
        "--benchmark-priors",
        help="Merge ModBench-style geometry priors (em|xray|combined|all); 0.1 Å bins from bundled JSON",
    ),
    benchmark_json: Optional[str] = typer.Option(
        None,
        "--benchmark-json",
        help="Path to benchmark_data.json (overrides bundled file)",
    ),
    no_progress: bool = typer.Option(
        False,
        "--no-progress",
        help="Disable the residue progress bar (still prints major steps)",
    ),
    monomer_lib: Optional[str] = typer.Option(
        None,
        "--monomer-lib",
        help=(
            "Directory with CCP4 monomers (list/mon_lib_list.cif) or residue *.cif files "
            "(e.g. CLIBD_MON); improves clash 1–4 exclusion vs distance-only bonds"
        ),
    ),
    residue_report: bool = typer.Option(
        True,
        "--residue-report/--no-residue-report",
        help="Write residue_report.txt and .csv (Coot-friendly per-residue summaries)",
    ),
    residue_report_score: str = typer.Option(
        "auto",
        "--residue-report-score",
        help="Headline score in report: auto, badness, quality, or band (same semantics as --bfactor-color)",
    ),
):
    """Validate cryoEM model with resolution-aware metrics."""
    typer.echo("FitCheck: Resolution-aware model validation")
    typer.echo(f"  Model: {model}")
    typer.echo(f"  Map: {map}")

    # Load structure (keep full copy for optional B-factor export)
    typer.echo("Loading model…")
    structure_full = gemmi.read_structure(model)
    structure = structure_full.clone()
    if protein_only:
        typer.echo("  Protein-only mode: excluding non-amino-acid residues")
        structure = filter_protein_only(structure)

    typer.echo("Loading primary map…")
    map_vol = read_map(map)

    half1_vol = None
    half2_vol = None
    if half1 and half2:
        typer.echo("Loading half-maps…")
        h1 = read_map(half1)
        h2 = read_map(half2)
        if not maps_grid_compatible(h1, map_vol):
            typer.echo("  Resampling half-map 1 onto primary map grid…")
            h1 = resample_map_volume(h1, map_vol)
        if not maps_grid_compatible(h2, map_vol):
            typer.echo("  Resampling half-map 2 onto primary map grid…")
            h2 = resample_map_volume(h2, map_vol)
        half1_vol, half2_vol = h1, h2

    local_res_map = None
    if localres:
        typer.echo("Loading local-resolution map…")
        local_res_map = read_map(localres)
        typer.echo(f"  Local resolution map: {localres}")
        if not maps_grid_compatible(local_res_map, map_vol):
            typer.echo("  Resampling local-resolution map onto primary map grid…")
            local_res_map = resample_map_volume(local_res_map, map_vol)
    
    # Extract features (tqdm on stderr when progress enabled; clash scan follows same stream)
    typer.echo(
        "Extracting per-residue features…"
        + ("" if no_progress else " (progress bar on stderr)"),
    )
    features_df = extract_residue_features(
        structure,
        map_vol,
        half1_vol,
        half2_vol,
        local_res_map,
        show_progress=not no_progress,
        monomer_lib_dir=monomer_lib,
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

    if benchmark_priors:
        bm_mod = benchmark_priors.strip().lower()
        if bm_mod not in ("xray", "em", "combined", "all"):
            typer.echo(
                f"Unknown --benchmark-priors {benchmark_priors!r} (use em, xray, combined, or all).",
                err=True,
            )
            raise typer.Exit(code=1)
        jp = Path(benchmark_json) if benchmark_json else None
        bm = priors_dict_from_benchmark(bm_mod, json_path=jp)
        priors_dict = merge_resolution_priors(priors_dict, bm)
        typer.echo(f"  ModBench geometry priors merged ({bm_mod}, {len(bm)} bins).")

    # Compute Z-residuals if priors available
    if priors_dict:
        typer.echo("Computing Z-residuals...")
        features_df = compute_z_residuals(features_df, priors_dict)

    typer.echo("Computing composite quality score…")
    features_df = add_composite_quality_columns(features_df)

    if bfactor_pdb:
        bc = bfactor_color.strip().lower()
        if bc not in ("auto", "band", "badness", "quality"):
            typer.echo(f"Unknown --bfactor-color {bfactor_color!r} (use auto, band, badness, quality).", err=True)
            raise typer.Exit(code=1)
        col = resolve_bfactor_column(features_df, bc)
        if col == "composite_band_deviation_0_100" and not features_df[col].notna().any():
            typer.echo(
                "  Note: no resolution-band z-scores (--priors or --fit-priors). "
                "Using composite_badness_0_100 for B-factors."
            )
            col = "composite_badness_0_100"
        hiw = metric_higher_is_worse(bc, col)
        if bfactor_threshold is not None:
            if hiw:
                features_df = features_df.copy()
                features_df["bfactor_flagged"] = (
                    features_df[col].astype(float) > float(bfactor_threshold)
                ).astype(int)
            else:
                features_df = features_df.copy()
                features_df["bfactor_flagged"] = (
                    features_df[col].astype(float) < float(bfactor_threshold)
                ).astype(int)
        outp = Path(bfactor_pdb)
        write_structure_with_composite_bfactors(
            structure_full,
            features_df,
            outp,
            column=col,
            threshold=float(bfactor_threshold) if bfactor_threshold is not None else None,
            higher_is_worse=hiw,
            fail_occupancy=float(bfactor_fail_occupancy),
        )
        typer.echo(f"  Wrote B-factor model to {outp} (column {col})")
        if col in ("composite_badness_0_100", "composite_band_deviation_0_100"):
            typer.echo(
                "  ChimeraX default ramp: blue = low (better), red = high (worse) for band/badness."
            )
        if bfactor_threshold is not None:
            nf = int(features_df["bfactor_flagged"].sum())
            typer.echo(
                f"  Threshold {bfactor_threshold}: {nf} residue(s) flagged "
                f"(occupancy={bfactor_fail_occupancy}; e.g. select occ < 0.9)."
            )

    # Save features
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    features_csv = out_path / "features.csv"
    features_df.to_csv(features_csv, index=False)
    typer.echo(f"  Saved features to {features_csv}")

    if residue_report:
        rs = residue_report_score.strip().lower()
        if rs not in ("auto", "band", "badness", "quality"):
            typer.echo(
                f"Unknown --residue-report-score {residue_report_score!r} "
                "(use auto, band, badness, quality).",
                err=True,
            )
            raise typer.Exit(code=1)
        score_col = resolve_validate_score_column(features_df, rs)
        csv_r, txt_r = write_residue_report(
            features_df,
            out_path,
            score_column=score_col,
        )
        typer.echo(f"  Saved per-residue report: {txt_r}")
        typer.echo(f"  Saved per-residue table: {csv_r}")
    
    # Compute summary statistics
    typer.echo("\nSummary statistics:")
    if "clashscore_per_1000_atoms" in features_df.columns:
        c1k = float(features_df["clashscore_per_1000_atoms"].iloc[0])
        typer.echo(
            f"  Internal steric pairs per 1000 heavy atoms (vdW + 0.25 Å probe): {c1k:.2f}"
        )
    if "molprobity_clashscore" in features_df.columns:
        mp = float(features_df["molprobity_clashscore"].iloc[0])
        typer.echo(
            f"  MolProbity clashscore (≥0.4 Å vdW overlap, per 1000 heavy atoms): {mp:.2f}"
        )
    if "steric_clashes" in features_df.columns:
        typer.echo(
            f"  Residues with ≥1 steric clash: {int((features_df['steric_clashes'] > 0).sum())}"
        )
    if "clashscore_z" in features_df.columns:
        nz = features_df["clashscore_z"].notna().sum()
        if nz == 0 and len(features_df) > 0:
            typer.echo(
                "  clashscore_z is NaN: no spread in clash density across residues "
                "(e.g. zero clashes everywhere — not the same as MolProbity clashscore)."
            )
    if "local_res" in features_df.columns:
        lr = features_df["local_res"].astype(float)
        typer.echo(f"  Mean local resolution: {lr.mean(skipna=True):.2f} Å")
        typer.echo(
            f"  Resolution range: {lr.min(skipna=True):.2f} - {lr.max(skipna=True):.2f} Å"
        )
    
    if 'Q_mean' in features_df.columns:
        typer.echo(f"  Mean Q-score: {features_df['Q_mean'].mean():.3f}")
    
    if 'CC_mask' in features_df.columns:
        typer.echo(f"  Mean CC_mask: {features_df['CC_mask'].mean():.3f}")
    
    if 'ringer_Z' in features_df.columns:
        typer.echo(f"  Mean Ringer Z: {features_df['ringer_Z'].mean():.2f}")
    
    typer.echo(f"\nValidation complete! Results in {out_path}")


if __name__ == "__main__":
    app()

