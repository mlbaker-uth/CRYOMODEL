"""CLI: helical symmetry finder (separate from Cn/Dn)."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from ..helical.finder import run_helical_find
from ..helical.local_refine import run_helical_refine_local
from ..helical.overlap_resolve import run_helical_resolve_overlaps
from ..helical.segmenter import run_helical_segment
from .command_log import log_command

app = typer.Typer(no_args_is_help=True, help="Helical symmetry finder (rise/twist).")


@app.command("find")
@log_command("helical find")
def find(
    map_path: Path = typer.Argument(..., exists=True, help="Input MRC/CCP4 map"),
    out_dir: Path = typer.Argument(..., help="Output directory"),
    density_threshold: Optional[float] = typer.Option(None, "--density-threshold"),
    density_percentile: float = typer.Option(90.0, "--density-percentile", min=0.0, max=100.0),
    axis_mode: str = typer.Option("cardinal_pca", "--axis-mode", help="cardinal | pca | cardinal_pca"),
    twist_min_deg: float = typer.Option(-20.0, "--twist-min-deg"),
    twist_max_deg: float = typer.Option(20.0, "--twist-max-deg"),
    twist_step_deg: float = typer.Option(0.5, "--twist-step-deg", min=0.05),
    rise_min_A: float = typer.Option(2.0, "--rise-min-A", min=0.1),
    rise_max_A: float = typer.Option(8.0, "--rise-max-A", min=0.1),
    rise_step_A: float = typer.Option(0.2, "--rise-step-A", min=0.02),
    max_voxels_score: int = typer.Option(200_000, "--max-voxels-score", min=5_000),
    seed: int = typer.Option(0, "--seed"),
    no_refine: bool = typer.Option(False, "--no-refine", help="Disable local fine refinement around coarse best"),
    refine_iters: int = typer.Option(2, "--refine-iters", min=0, max=8, help="Extra iterative local zoom refinements"),
    no_heatmap: bool = typer.Option(False, "--no-heatmap", help="Skip writing twist-vs-rise heatmap PNG"),
):
    """
    Estimate helical parameters (twist/rise) by maximizing one-step screw self-correlation.

    Notes:
    - Axis search is cardinal-first (x/y/z) with optional PCA candidate.
    - For very slow pitch fibrils (e.g. beta-amyloid), narrow around small |twist| and realistic rise:
      ``--twist-min-deg -5 --twist-max-deg 5 --twist-step-deg 0.2 --rise-min-A 3 --rise-max-A 6``.
    """
    if axis_mode not in ("cardinal", "pca", "cardinal_pca"):
        typer.echo("--axis-mode must be one of: cardinal, pca, cardinal_pca", err=True)
        raise typer.Exit(1)
    if twist_max_deg <= twist_min_deg:
        typer.echo("--twist-max-deg must be > --twist-min-deg", err=True)
        raise typer.Exit(1)
    if rise_max_A <= rise_min_A:
        typer.echo("--rise-max-A must be > --rise-min-A", err=True)
        raise typer.Exit(1)

    res = run_helical_find(
        map_path,
        out_dir,
        density_threshold=density_threshold,
        density_percentile=density_percentile,
        axis_mode=axis_mode,
        twist_min_deg=twist_min_deg,
        twist_max_deg=twist_max_deg,
        twist_step_deg=twist_step_deg,
        rise_min_A=rise_min_A,
        rise_max_A=rise_max_A,
        rise_step_A=rise_step_A,
        max_voxels_score=max_voxels_score,
        seed=seed,
        refine=not no_refine,
        refine_iters=refine_iters,
        write_heatmap=not no_heatmap,
    )
    typer.echo(f"Wrote {res.output_json}")
    if res.heatmap_png:
        typer.echo(f"Wrote {res.heatmap_png}")
    typer.echo(f"Best helix: rise={res.best_rise_A:.4f} Å, twist={res.best_twist_deg:.4f} deg, score={res.best_score:.4f}")
    typer.echo("Axis xyz: " + ", ".join(f"{v:.4f}" for v in res.axis_xyz))
    typer.echo("Pivot xyz (Å): " + ", ".join(f"{v:.2f}" for v in res.pivot_xyz))


@app.command("segment")
@log_command("helical segment")
def segment(
    map_path: Path = typer.Argument(..., exists=True, help="Input MRC/CCP4 map"),
    helical_json: Path = typer.Argument(..., exists=True, help="helical_find.json from `helical find`"),
    out_dir: Path = typer.Argument(..., help="Output directory"),
    density_threshold: Optional[float] = typer.Option(
        None,
        "--density-threshold",
        help="Approximate modeling threshold; defaults to helical_find threshold",
    ),
    k_window: int = typer.Option(3, "--k-window", min=1, max=12, help="Local k search half-window around t/rise"),
    sigma_t_A: Optional[float] = typer.Option(None, "--sigma-t-A", help="Axial tolerance scale (Å)"),
    sigma_phi_deg: Optional[float] = typer.Option(None, "--sigma-phi-deg", help="Angular tolerance scale (deg)"),
    max_norm_cost: float = typer.Option(12.0, "--max-norm-cost", min=0.1, max=100.0, help="Max normalized assignment cost"),
    min_cost_margin: float = typer.Option(0.05, "--min-cost-margin", min=0.0, max=50.0, help="Min gap between best and 2nd-best assignment"),
    mode: str = typer.Option("phase_peaks", "--mode", help="segmentation mode: phase_peaks | seeded_watershed | analytic"),
    radial_band_center_A: Optional[float] = typer.Option(None, "--radial-band-center-A", help="Radial center for axial profile (Å)"),
    radial_band_halfwidth_A: float = typer.Option(2.5, "--radial-band-halfwidth-A", min=0.3, max=30.0, help="Radial band halfwidth (Å)"),
    axial_window_halfwidth_A: Optional[float] = typer.Option(None, "--axial-window-halfwidth-A", help="Halfwidth for repeat assignment windows (Å)"),
    peak_min_prominence: float = typer.Option(0.0, "--peak-min-prominence", min=0.0, help="Min prominence for axial profile peaks"),
    shear_alpha_rad_per_A: Optional[float] = typer.Option(
        None,
        "--shear-alpha-rad-per-A",
        help="Single shear α (rad/Å) in (dθ - α·Δz); overrides two-slope mode",
    ),
    shear_alpha_pos_rad_per_A: Optional[float] = typer.Option(
        None,
        "--shear-alpha-pos-rad-per-A",
        help="Two-slope: α for Δz≥0 (Å⁻¹ rad); use with --shear-alpha-neg-rad-per-A",
    ),
    shear_alpha_neg_rad_per_A: Optional[float] = typer.Option(
        None,
        "--shear-alpha-neg-rad-per-A",
        help="Two-slope: α for Δz<0 (Å⁻¹ rad); use with --shear-alpha-pos-rad-per-A",
    ),
    largest_component: bool = typer.Option(
        False,
        "--largest-component",
        help="Representative MRC only: keep largest 26-connected component of the top label",
    ),
    prune_labels_largest_component: bool = typer.Option(
        False,
        "--prune-labels-largest-component",
        help="Label volume: per subunit ID, keep only the largest 26-connected component (drops detached blobs)",
    ),
    watershed_max_norm_cost: Optional[float] = typer.Option(
        None,
        "--watershed-max-norm-cost",
        help=(
            "After watershed: drop filled voxels whose best phase cost exceeds this; "
            "core seeds are always kept (same units as --max-norm-cost). "
            "Try ~1.3–2.0× your --max-norm-cost for the fill, not the same value."
        ),
    ),
    no_qc_png: bool = typer.Option(False, "--no-qc-png", help="Skip QC PNG diagnostic output"),
    no_average: bool = typer.Option(False, "--no-average", help="Skip average-subunit map output"),
    no_sequential_helical_labels: bool = typer.Option(
        False,
        "--no-sequential-helical-labels",
        help="Keep raw label IDs from segmentation (default: renumber 1..K along helical axis)",
    ),
):
    """
    Segment map voxels into helical subunits using the fitted axis/rise/twist.

    Outputs:
    - helical_subunit_labels.mrc      (0 background, positive integer subunit IDs)
    - helical_subunit_representative.mrc
    - helical_subunit_average.mrc     (optional)
    - helical_segment.json
    """
    if mode not in ("phase_peaks", "seeded_watershed", "analytic"):
        typer.echo("--mode must be one of: phase_peaks, seeded_watershed, analytic", err=True)
        raise typer.Exit(1)
    res = run_helical_segment(
        map_path,
        helical_json,
        out_dir,
        density_threshold=density_threshold,
        k_window=k_window,
        sigma_t_A=sigma_t_A,
        sigma_phi_deg=sigma_phi_deg,
        max_norm_cost=max_norm_cost,
        min_cost_margin=min_cost_margin,
        mode=mode,
        radial_band_center_A=radial_band_center_A,
        radial_band_halfwidth_A=radial_band_halfwidth_A,
        axial_window_halfwidth_A=axial_window_halfwidth_A,
        peak_min_prominence=peak_min_prominence,
        shear_alpha_rad_per_A=shear_alpha_rad_per_A,
        shear_alpha_pos_rad_per_A=shear_alpha_pos_rad_per_A,
        shear_alpha_neg_rad_per_A=shear_alpha_neg_rad_per_A,
        representative_largest_component=largest_component,
        prune_labels_largest_component=prune_labels_largest_component,
        watershed_max_norm_cost=watershed_max_norm_cost,
        sequential_helical_labels=not no_sequential_helical_labels,
        write_qc_png=not no_qc_png,
        write_average=not no_average,
    )
    typer.echo(f"Wrote {res.output_json}")
    typer.echo(f"Labels map: {res.labels_map} ({res.n_subunits} subunits)")
    typer.echo(f"Representative map: {res.representative_map}")
    if res.qc_png:
        typer.echo(f"QC PNG: {res.qc_png}")
    if res.average_map:
        typer.echo(f"Average map: {res.average_map}")


@app.command("resolve-overlaps")
@log_command("helical resolve-overlaps")
def resolve_overlaps(
    map_path: Path = typer.Argument(..., exists=True, help="Density map (same grid as masks)"),
    out_dir: Path = typer.Argument(..., help="Output directory"),
    masks: list[Path] = typer.Argument(
        ...,
        help="One or more mask MRCs (any positive voxel = that subunit claims the voxel); may overlap",
    ),
    tie_break: str = typer.Option(
        "density",
        "--tie-break",
        help="How to break ties at overlaps: density | mask_order",
    ),
    write_representative: bool = typer.Option(
        False,
        "--write-representative",
        help="Write helical_overlap_representative.mrc for one label (default: largest by voxel count)",
    ),
    representative_label: Optional[int] = typer.Option(
        None,
        "--representative-label",
        min=1,
        help="1-based label id for representative map (default: largest label)",
    ),
    largest_component: bool = typer.Option(
        False,
        "--largest-component",
        help="Representative: keep only largest 26-connected component of that label",
    ),
):
    """
    Merge overlapping binary masks into one label volume (exactly one label per voxel).

    Use when building an assembly from several subunit masks that claim the same voxels.
    Overlaps are resolved using the density map (higher value wins) unless --tie-break mask_order.
    Outputs helical_overlap_labels.mrc and helical_overlap_resolve.json.
    """
    if tie_break not in ("density", "mask_order"):
        typer.echo("--tie-break must be density or mask_order", err=True)
        raise typer.Exit(1)
    if len(masks) < 1:
        typer.echo("Provide at least one mask MRC.", err=True)
        raise typer.Exit(1)
    res = run_helical_resolve_overlaps(
        map_path,
        masks,
        out_dir,
        tie_break=tie_break,  # type: ignore[arg-type]
        write_representative=write_representative,
        representative_label=representative_label,
        representative_largest_component=largest_component,
    )
    typer.echo(f"Wrote {res.output_json}")
    typer.echo(f"Labels map: {res.labels_map}")
    typer.echo(f"Overlap voxels (before resolve): {res.n_overlap_voxels}")
    if res.representative_map:
        typer.echo(f"Representative map: {res.representative_map}")


@app.command("refine-local")
@log_command("helical refine-local")
def refine_local(
    map_path: Path = typer.Argument(..., exists=True, help="Same density map used for segmentation"),
    labels_path: Path = typer.Argument(..., exists=True, help="helical_subunit_labels.mrc (full volume)"),
    segment_json: Path = typer.Argument(..., exists=True, help="helical_segment.json from helical segment"),
    out_dir: Path = typer.Argument(..., help="Output directory"),
    neighbor_layers: int = typer.Option(
        2,
        "--neighbor-layers",
        min=0,
        max=24,
        help="Refine representative label ± this many neighbor IDs (end subunits use fewer neighbors)",
    ),
    pad_voxels: int = typer.Option(8, "--pad-voxels", min=0, max=64, help="Extra voxels around active-label bbox"),
    density_threshold: Optional[float] = typer.Option(
        None,
        "--density-threshold",
        help="Override threshold (default: value stored in segment JSON)",
    ),
    representative_label: Optional[int] = typer.Option(
        None,
        "--representative-label",
        min=1,
        help="1-based label to treat as representative (default: segment JSON or largest count)",
    ),
    largest_component: bool = typer.Option(
        False,
        "--largest-component",
        help="Representative map: keep largest 26-connected component only",
    ),
):
    """
    Refine boundaries in a **cropped** region around the representative subunit and ±N neighbor labels,
    using one full-size label map (no per-subunit mask files).

    Writes full-size ``helical_subunit_labels_refined.mrc``, a representative density map, and a binary
    ``helical_representative_mask.mrc``. Re-run ``helical segment`` (or keep an older JSON) with
    ``phase_peaks`` so ``repeat_center_t_A`` is present; missing entries are filled from medians or a
    linear axial fallback.
    """
    res = run_helical_refine_local(
        map_path,
        labels_path,
        segment_json,
        out_dir,
        neighbor_layers=neighbor_layers,
        pad_voxels=pad_voxels,
        density_threshold=density_threshold,
        representative_label=representative_label,
        representative_largest_component=largest_component,
    )
    typer.echo(f"Wrote {res.output_json}")
    typer.echo(f"Refined labels: {res.labels_map} ({res.n_voxels_refined} voxels re-assigned in crop)")
    typer.echo(f"Crop z,y,x index ranges: {res.crop_zyx_slices}")
    typer.echo(f"Representative (label {res.representative_label_id}): {res.representative_map}")
    typer.echo(f"Representative mask: {res.representative_mask_map}")

