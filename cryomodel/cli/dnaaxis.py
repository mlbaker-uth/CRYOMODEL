from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from ..nucleotide.dna_axis import extract_medial_axis
from .command_log import log_command

app = typer.Typer(no_args_is_help=True, help="Trace a medial axis through a dsDNA density map")


@app.command()
@log_command("dnaaxis extract")
def extract(
    map_path: str = typer.Option(..., "--map", help="Input map (.mrc) containing dsDNA"),
    threshold: float = typer.Option(..., "--threshold", help="Density threshold for axis extraction"),
    n_points: Optional[int] = typer.Option(None, "--n-points", help="Total points in final polyline (default: length/3.4Å)"),
    guides_pdb: str = typer.Option("", "--guides-pdb", help="Ordered guide PDB (ATOM/HETATM records in file order)"),
    endpoints_pdb: str = typer.Option("", "--endpoints-pdb", help="Alias for a 2-point guide PDB"),
    power: float = typer.Option(3.0, "--power", help="Cost exponent for inverse DT weighting"),
    eps: float = typer.Option(1e-3, "--eps", help="Stabilizer added to DT"),
    cap_fraction: float = typer.Option(0.03, "--cap-fraction", help="Fraction of voxels at each PC1 extreme for termini"),
    close_iters: int = typer.Option(0, "--close-iters", help="Binary closing iterations"),
    smooth: float = typer.Option(0.4, "--smooth", help="Spline smoothing factor per segment"),
    pre_smooth_sigma: float = typer.Option(1.0, "--pre-smooth-sigma", help="Gaussian sigma for segment smoothing"),
    pre_smooth_iters: int = typer.Option(2, "--pre-smooth-iters", help="Pre-smoothing passes per segment"),
    guide_search_radius: float = typer.Option(8.0, "--guide-search-radius", help="Search radius (Å) around each guide"),
    guide_dt_weight: float = typer.Option(2.0, "--guide-dt-weight", help="DT weight when anchoring guides"),
    max_guide_span_A: float = typer.Option(60.0, "--max-guide-span-A", help="Warn if guide span exceeds this distance (Å)"),
    point_refine_window_A: float = typer.Option(0.8, "--point-refine-window-A", help="Arc-length search window (Å)"),
    point_refine_dt_weight: float = typer.Option(0.35, "--point-refine-dt-weight", help="DT weight for point refinement"),
    recenter_radius_A: float = typer.Option(1.2, "--recenter-radius-A", help="Local recenter search radius (Å)"),
    recenter_dt_weight: float = typer.Option(1.0, "--recenter-dt-weight", help="DT weight for recentering"),
    recenter_disp_weight: float = typer.Option(0.35, "--recenter-disp-weight", help="Displacement penalty for recentering"),
    recenter_iters: int = typer.Option(1, "--recenter-iters", help="Recenter iterations"),
    rise_per_bp: float = typer.Option(3.4, "--rise-per-bp", help="Rise per base pair (Å)"),
    target_spacing: Optional[float] = typer.Option(None, "--target-spacing", help="Override spacing (Å)"),
    count_mode: str = typer.Option("round", "--count-mode", help="Point count mode: round, ceil, floor"),
    report: str = typer.Option("", "--report", help="Optional report output path"),
    out_mrc: str = typer.Option("dna_axis.mrc", "--out-mrc", help="Output axis MRC path"),
    out_pdb: str = typer.Option("dna_axis.pdb", "--out-pdb", help="Output CA PDB path"),
):
    map_path = Path(map_path)
    if not map_path.exists():
        raise typer.BadParameter(f"Map not found: {map_path}")

    extract_medial_axis(
        map_path=map_path,
        threshold=threshold,
        n_points=n_points,
        out_mrc=Path(out_mrc),
        out_pdb=Path(out_pdb),
        guides_pdb=Path(guides_pdb) if guides_pdb else None,
        endpoints_pdb=Path(endpoints_pdb) if endpoints_pdb else None,
        power=power,
        eps=eps,
        cap_fraction=cap_fraction,
        close_iters=close_iters,
        smooth=smooth,
        pre_smooth_sigma=pre_smooth_sigma,
        pre_smooth_iters=pre_smooth_iters,
        guide_search_radius_A=guide_search_radius,
        guide_dt_weight=guide_dt_weight,
        max_guide_span_A=max_guide_span_A,
        point_refine_window_A=point_refine_window_A,
        point_refine_dt_weight=point_refine_dt_weight,
        recenter_radius_A=recenter_radius_A,
        recenter_dt_weight=recenter_dt_weight,
        recenter_disp_weight=recenter_disp_weight,
        recenter_iters=recenter_iters,
        rise_per_bp_A=rise_per_bp,
        target_spacing_A=target_spacing,
        count_mode=count_mode,
        report_path=Path(report) if report else None,
    )

    typer.echo(f"Wrote: {out_mrc}")
    typer.echo(f"Wrote: {out_pdb}")
