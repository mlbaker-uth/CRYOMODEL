"""CLI: symmetry discovery (phase 0 preprocess, later full search)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from ..symmetry.axis_candidates import run_phase1_candidates
from ..symmetry.multishell_cn import run_multishell_cn_scores
from ..symmetry.phase2_cn import run_phase2_cn_scores
from ..symmetry.phase2_dn import run_phase2_dn_scores
from ..symmetry.phase3_dn_refine import run_phase3d_refine
from ..symmetry.phase3_refine import run_phase3_refine
from ..symmetry.phase4_axis_pdb import run_phase4_axis_pdb
from ..symmetry.pipeline_find import run_symmetry_find, run_symmetry_find_auto
from ..symmetry.preprocess import run_phase0_preprocess
from .command_log import log_command

app = typer.Typer(no_args_is_help=True, help="Symmetry axis discovery and related tools.")
C_ORDERS_DEFAULT_STR = ",".join(str(n) for n in range(2, 21))
D_ORDERS_DEFAULT_STR = ",".join(str(n) for n in range(2, 13))


@app.command("phase0")
@log_command("symmetry phase0")
def phase0(
    map_path: Path = typer.Argument(..., exists=True, help="Input MRC/CCP4 map"),
    out_dir: Path = typer.Argument(..., help="Output directory (created if missing)"),
    mask: Optional[Path] = typer.Option(None, "--mask", "-m", exists=True, help="Optional mask map (same grid or resampled to map)"),
    downsample: int = typer.Option(4, "--downsample", "-d", min=1, max=64, help="Integer block-mean downsample factor"),
    bandpass_low: Optional[float] = typer.Option(
        None,
        "--bandpass-low",
        help="Band-pass low-resolution cutoff (Å); use with --bandpass-high",
    ),
    bandpass_high: Optional[float] = typer.Option(
        None,
        "--bandpass-high",
        help="Band-pass high-resolution cutoff (Å); use with --bandpass-low",
    ),
    edge: str = typer.Option(
        "none",
        "--edge",
        help="Edge emphasis: none | laplacian | laplacian_sharpen",
    ),
    laplacian_strength: float = typer.Option(
        1.0,
        "--laplacian-strength",
        min=0.0,
        max=10.0,
        help="Strength for laplacian_sharpen edge mode",
    ),
    density_percentile: Optional[float] = typer.Option(
        90.0,
        "--density-percentile",
        help="Voxel selection percentile for inertia/PCA (ignored if --density-threshold set)",
    ),
    density_threshold: Optional[float] = typer.Option(
        None,
        "--density-threshold",
        help="Absolute map threshold for voxel selection (overrides percentile)",
    ),
    max_voxels_pca: int = typer.Option(
        400_000,
        "--max-voxels-pca",
        min=1000,
        help="Max voxels used for weighted PCA (subsample if more)",
    ),
    seed: int = typer.Option(0, "--seed", help="RNG seed for PCA subsampling"),
):
    """
    Phase 0 — preprocess map for symmetry search: optional mask, band-pass, edge emphasis,
    downsample, then weighted principal axes from high-density voxels.

    Writes ``symmetry_phase0_downsample.mrc`` and ``symmetry_phase0.json`` under OUT_DIR.
    """
    if (bandpass_low is None) ^ (bandpass_high is None):
        typer.echo("Provide both --bandpass-low and --bandpass-high, or neither.", err=True)
        raise typer.Exit(1)
    if edge not in ("none", "laplacian", "laplacian_sharpen"):
        typer.echo("--edge must be one of: none, laplacian, laplacian_sharpen", err=True)
        raise typer.Exit(1)

    bl = bandpass_low
    bh = bandpass_high
    res = run_phase0_preprocess(
        map_path,
        out_dir=out_dir,
        mask_path=mask,
        downsample_factor=downsample,
        bandpass_low_res_A=bl,
        bandpass_high_res_A=bh,
        edge_emphasis=edge,  # type: ignore[arg-type]
        laplacian_sharpen_strength=laplacian_strength,
        density_threshold=density_threshold,
        density_percentile=density_percentile if density_threshold is None else None,
        max_voxels_pca=max_voxels_pca,
        random_seed=seed,
    )
    typer.echo(f"Wrote {res.output_map}")
    typer.echo(f"Wrote {res.output_json}")
    typer.echo(
        "Primary principal axis (Å, xyz): "
        + ", ".join(f"{v:.4f}" for v in res.principal_axes_xyz[0])
    )


@app.command("phase1")
@log_command("symmetry phase1")
def phase1(
    phase0_dir: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=False,
        help="Directory containing symmetry_phase0.json and symmetry_phase0_downsample.mrc",
    ),
    tilt: str = typer.Option(
        "0,5,10,15",
        "--tilt-deg",
        help="Comma-separated tilt angles (degrees) for near-cardinal grids",
    ),
    no_diagonals: bool = typer.Option(False, "--no-diagonals", help="Omit (±1,±1,±1) body diagonals"),
    axial_bins: int = typer.Option(64, "--axial-bins", min=8, max=512, help="Bins for 1D axial mass profile"),
):
    """
    Phase 1 — discrete axis candidates (cardinal + tilt grid, optional diagonals, phase-0 PCA axes)
    and per-candidate axial mass profiles + radial statistics on the phase-0 downsampled map.

    Requires prior ``cryomodel symmetry phase0 ... PHASE0_DIR``. Writes ``symmetry_phase1.json``.
    """
    parts = [float(x.strip()) for x in tilt.split(",") if x.strip()]
    if not parts:
        typer.echo("--tilt-deg must list at least one angle.", err=True)
        raise typer.Exit(1)
    td = tuple(parts)
    res = run_phase1_candidates(
        phase0_dir,
        tilt_degrees=td,
        include_diagonals=not no_diagonals,
        n_axial_bins=axial_bins,
    )
    typer.echo(f"Wrote {res.output_json} ({res.n_candidates} candidates)")


@app.command("phase2")
@log_command("symmetry phase2")
def phase2(
    phase0_dir: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=False,
        help="Directory with symmetry_phase0.json, symmetry_phase1.json, symmetry_phase0_downsample.mrc",
    ),
    orders: str = typer.Option(
        C_ORDERS_DEFAULT_STR,
        "--orders",
        help="Comma-separated Cₙ orders to score (each n ≥ 2)",
    ),
    max_candidates: Optional[int] = typer.Option(
        None,
        "--max-candidates",
        min=1,
        help="Score only the first N phase-1 candidates (order preserved)",
    ),
):
    """
    Phase 2 — for each phase-1 axis candidate, score Cₙ (n from ``--orders``) by rotational
    self-correlation (Pearson r) on voxels above the phase-0 threshold on the downsampled map.

    Requires prior ``phase0`` and ``phase1`` in the same directory. Writes ``symmetry_phase2.json``.
    """
    parts = [int(x.strip()) for x in orders.split(",") if x.strip()]
    if not parts or any(n < 2 for n in parts):
        typer.echo("--orders must list integers n ≥ 2.", err=True)
        raise typer.Exit(1)
    res = run_phase2_cn_scores(phase0_dir, orders=tuple(parts), max_candidates=max_candidates)
    gb = res.global_best
    typer.echo(f"Wrote {res.output_json}")
    typer.echo(
        f"Global best: candidate_id={gb['candidate_id']} n={gb['n']} r={gb['score']:.4f} ({gb['source']})"
    )


@app.command("phase2d")
@log_command("symmetry phase2d")
def phase2d(
    phase0_dir: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=False,
        help="Directory with symmetry_phase0.json, symmetry_phase1.json, symmetry_phase0_downsample.mrc",
    ),
    orders: str = typer.Option(
        D_ORDERS_DEFAULT_STR,
        "--orders",
        help="Comma-separated Dₙ orders to score (each n ≥ 2)",
    ),
    max_candidates: Optional[int] = typer.Option(
        None,
        "--max-candidates",
        min=1,
        help="Score only the first N phase-1 candidates (order preserved)",
    ),
    inplane_samples: int = typer.Option(
        36,
        "--inplane-samples",
        min=8,
        max=360,
        help="Number of in-plane angles to search for perpendicular C2 axis",
    ),
):
    """
    Phase 2D — for each phase-1 axis candidate, score Dₙ by combining:
    - Cₙ self-correlation about the candidate axis
    - best C2 self-correlation about a perpendicular axis (searched over in-plane angle)

    Requires prior ``phase0`` and ``phase1`` in the same directory. Writes ``symmetry_phase2d.json``.
    """
    parts = [int(x.strip()) for x in orders.split(",") if x.strip()]
    if not parts or any(n < 2 for n in parts):
        typer.echo("--orders must list integers n ≥ 2.", err=True)
        raise typer.Exit(1)
    res = run_phase2_dn_scores(
        phase0_dir,
        orders=tuple(parts),
        max_candidates=max_candidates,
        inplane_samples=inplane_samples,
    )
    gb = res.global_best
    typer.echo(f"Wrote {res.output_json}")
    typer.echo(
        f"Global best D_n: candidate_id={gb['candidate_id']} n={gb['n']} score={gb['score']:.4f} ({gb['source']})"
    )


@app.command("phase3")
@log_command("symmetry phase3")
def phase3(
    phase0_dir: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=False,
        help="Directory with symmetry_phase0/1/2 artifacts and symmetry_phase2.json",
    ),
    top: int = typer.Option(3, "--top", min=1, max=32, help="Refine this many best phase-2 hypotheses"),
    max_tilt_deg: float = typer.Option(
        5.0,
        "--max-tilt-deg",
        min=0.1,
        max=30.0,
        help="Half-range (±) for small axis tilts in the tangent plane",
    ),
    max_shift_along: float = typer.Option(
        10.0,
        "--max-shift-along-A",
        min=0.0,
        max=80.0,
        help="Max pivot shift along the refined axis (Å)",
    ),
    max_shift_perp: float = typer.Option(
        6.0,
        "--max-shift-perp-A",
        min=0.0,
        max=40.0,
        help="Max pivot shift perpendicular to the axis (Å)",
    ),
    maxiter: int = typer.Option(80, "--maxiter", min=10, max=500, help="L-BFGS-B iteration cap per hypothesis"),
):
    """
    Phase 3 — local L-BFGS-B refinement of axis direction and pivot for the top phase-2 Cₙ
    hypotheses (same mask and map as phase 2). Writes ``symmetry_phase3.json``.
    """
    res = run_phase3_refine(
        phase0_dir,
        top_hypotheses=top,
        max_tilt_deg=max_tilt_deg,
        max_shift_along_axis_A=max_shift_along,
        max_shift_perp_A=max_shift_perp,
        maxiter=maxiter,
    )
    typer.echo(f"Wrote {res.output_json} ({len(res.refinements)} hypotheses)")
    if res.refinements:
        best = res.refinements[0]
        rs = best.get("refined_score", best.get("phase2_best_score"))
        typer.echo(
            f"Best refined: id={best.get('phase2_candidate_id')} n={best.get('n')} r={float(rs):.4f} "
            f"(Δ={float(best.get('score_delta', 0.0)):+.4f})"
        )


@app.command("phase3d")
@log_command("symmetry phase3d")
def phase3d(
    phase0_dir: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=False,
        help="Directory with symmetry_phase0 and symmetry_phase2d artifacts",
    ),
    top: int = typer.Option(3, "--top", min=1, max=32),
    inplane_samples: int = typer.Option(36, "--inplane-samples", min=8, max=360),
    max_tilt_deg: float = typer.Option(5.0, "--max-tilt-deg", min=0.1, max=30.0),
    max_shift_along: float = typer.Option(10.0, "--max-shift-along-A", min=0.0, max=80.0),
    max_shift_perp: float = typer.Option(6.0, "--max-shift-perp-A", min=0.0, max=40.0),
    maxiter: int = typer.Option(80, "--maxiter", min=10, max=500),
):
    """Phase 3D — local refinement of Dₙ objective for top phase-2D hypotheses."""
    res = run_phase3d_refine(
        phase0_dir,
        top_hypotheses=top,
        inplane_samples=inplane_samples,
        max_tilt_deg=max_tilt_deg,
        max_shift_along_axis_A=max_shift_along,
        max_shift_perp_A=max_shift_perp,
        maxiter=maxiter,
    )
    typer.echo(f"Wrote {res.output_json} ({len(res.refinements)} hypotheses)")
    if res.refinements:
        best = res.refinements[0]
        typer.echo(
            f"Best refined D_n: id={best.get('phase2d_candidate_id')} n={best.get('n')} "
            f"score={float(best.get('refined_score', best.get('phase2d_best_score', -2.0))):.4f} "
            f"(Δ={float(best.get('score_delta', 0.0)):+.4f})"
        )


@app.command("phase4")
@log_command("symmetry phase4")
def phase4(
    phase0_dir: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=False,
        help="Directory with symmetry_phase0.json, symmetry_phase2.json (and optional phase3)",
    ),
    out_pdb: Optional[Path] = typer.Option(
        None,
        "--out-pdb",
        "-o",
        help="Output PDB path (default: PHASE0_DIR/symmetry_axis_ca.pdb)",
    ),
    reference_map: Optional[Path] = typer.Option(
        None,
        "--map",
        "-m",
        exists=True,
        help="Full-resolution map for bbox/frame (default: input_map from symmetry_phase0.json)",
    ),
    slice_step: float = typer.Option(
        10.0,
        "--slice-step",
        min=1.0,
        max=10_000.0,
        help="Approximate spacing along axis in full-map voxels (spacing Å = step × apix)",
    ),
    refinement_index: int = typer.Option(
        0,
        "--refinement-index",
        min=0,
        max=31,
        help="Which phase-3 refinement row to use (by refined-score rank); ignored if no phase3",
    ),
    no_phase3: bool = typer.Option(
        False,
        "--no-phase3",
        help="Use phase-2 global-best axis and phase-0 COM as pivot (ignore symmetry_phase3.json)",
    ),
):
    """
    Phase 4 — write a CA trace PDB along the Cₙ symmetry axis in the **reference map** frame
    (same Å origin/apix as the MRC you open beside it). Points are spaced by ``--slice-step``
    times the reference-map voxel size along the axis, clipped to the map bounding box.

    Requires ``symmetry phase0`` and ``phase2``; uses ``symmetry_phase3.json`` when present unless
    ``--no-phase3`` is set.
    """
    res = run_phase4_axis_pdb(
        phase0_dir,
        out_pdb=out_pdb,
        reference_map_path=reference_map,
        slice_step_voxels=slice_step,
        refinement_index=refinement_index,
        prefer_phase3=not no_phase3,
    )
    typer.echo(f"Wrote {res.output_pdb} ({res.n_points} CA atoms, C{res.n_fold}, {res.axis_source})")
    typer.echo(f"Wrote {res.output_json}")


@app.command("multishell")
@log_command("symmetry multishell")
def multishell(
    phase0_dir: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=False,
        help="Directory with symmetry_phase0 outputs (and phase 2; phase 3 optional)",
    ),
    orders: str = typer.Option(
        C_ORDERS_DEFAULT_STR,
        "--orders",
        help="Comma-separated Cₙ orders per shell",
    ),
    n_shells: int = typer.Option(8, "--n-shells", min=1, max=64, help="Equal-width annuli in ρ from axis"),
    max_radius_A: Optional[float] = typer.Option(
        None,
        "--max-radius-A",
        help="Cap cylindrical radius (Å); default: percentile of ρ among masked voxels",
    ),
    radius_percentile: float = typer.Option(
        99.5,
        "--radius-percentile",
        min=50.0,
        max=100.0,
        help="When --max-radius-A unset, cap ρ at this percentile of masked voxels",
    ),
    min_voxels: int = typer.Option(
        64,
        "--min-voxels",
        min=8,
        max=1_000_000,
        help="Skip shells with fewer masked voxels",
    ),
    refinement_index: int = typer.Option(
        0,
        "--refinement-index",
        min=0,
        max=31,
        help="Phase-3 refinement row for axis/pivot (if symmetry_phase3.json exists)",
    ),
    no_phase3: bool = typer.Option(
        False,
        "--no-phase3",
        help="Use phase-2 global axis + phase-0 COM (ignore symmetry_phase3.json)",
    ),
):
    """
    Cylindrical multishell Cₙ scores: same correlation as phase 2, evaluated in radial annuli
    about the chosen symmetry axis. Writes ``symmetry_multishell.json``.
    """
    parts = [int(x.strip()) for x in orders.split(",") if x.strip()]
    if not parts or any(n < 2 for n in parts):
        typer.echo("--orders must list integers n ≥ 2.", err=True)
        raise typer.Exit(1)
    res = run_multishell_cn_scores(
        phase0_dir,
        orders=tuple(parts),
        n_shells=n_shells,
        max_radius_A=max_radius_A,
        radius_percentile_cap=radius_percentile,
        min_voxels_per_shell=min_voxels,
        refinement_index=refinement_index,
        prefer_phase3=not no_phase3,
    )
    typer.echo(f"Wrote {res.output_json} ({res.n_shells} shells, r_cap={res.radius_cap_A:.2f} Å, axis={res.axis_source})")


@app.command("find")
@log_command("symmetry find")
def find(
    map_path: Path = typer.Argument(..., exists=True, help="Input MRC/CCP4 map"),
    out_dir: Path = typer.Argument(..., help="Output directory (created if missing)"),
    mask: Optional[Path] = typer.Option(None, "--mask", "-m", exists=True, help="Optional mask map"),
    downsample: int = typer.Option(4, "--downsample", "-d", min=1, max=64),
    bandpass_low: Optional[float] = typer.Option(None, "--bandpass-low"),
    bandpass_high: Optional[float] = typer.Option(None, "--bandpass-high"),
    edge: str = typer.Option("none", "--edge", help="none | laplacian | laplacian_sharpen"),
    laplacian_strength: float = typer.Option(1.0, "--laplacian-strength", min=0.0, max=10.0),
    density_percentile: Optional[float] = typer.Option(90.0, "--density-percentile"),
    density_threshold: Optional[float] = typer.Option(None, "--density-threshold"),
    max_voxels_pca: int = typer.Option(400_000, "--max-voxels-pca", min=1000),
    seed: int = typer.Option(0, "--seed"),
    tilt: str = typer.Option("0,5,10,15", "--tilt-deg"),
    no_diagonals: bool = typer.Option(False, "--no-diagonals"),
    axial_bins: int = typer.Option(64, "--axial-bins", min=8, max=512),
    orders: Optional[str] = typer.Option(
        None,
        "--orders",
        help="Comma-separated orders; default family-specific (C: 2..20, D: 2..12, auto: each family default)",
    ),
    family: str = typer.Option("c", "--family", help="Symmetry family: c | d | auto"),
    mode: str = typer.Option("search", "--mode", help="Mode: search | guided"),
    guided_order: Optional[int] = typer.Option(None, "--guided-order", min=2, help="Required for --mode guided"),
    max_candidates: Optional[int] = typer.Option(None, "--max-candidates", min=1),
    no_phase3: bool = typer.Option(False, "--no-phase3", help="Skip phase 3; axis PDB/multishell use phase 2 only"),
    phase3_top: int = typer.Option(3, "--phase3-top", min=1, max=32),
    phase3_maxiter: int = typer.Option(80, "--phase3-maxiter", min=10, max=500),
    no_multishell: bool = typer.Option(False, "--no-multishell"),
    n_shells: int = typer.Option(8, "--n-shells", min=1, max=64),
    shell_radius_pct: float = typer.Option(99.5, "--shell-radius-percentile", min=50.0, max=100.0),
    no_axis_pdb: bool = typer.Option(False, "--no-axis-pdb"),
    slice_step: float = typer.Option(10.0, "--slice-step", min=1.0, max=10_000.0),
    out_pdb: Optional[Path] = typer.Option(None, "--out-pdb", "-o"),
    reference_map: Optional[Path] = typer.Option(
        None,
        "--map",
        exists=True,
        help="Reference map for axis PDB frame (default: same as input map)",
    ),
):
    """
    Run the full symmetry pipeline: phase 0 → 1 → 2 → 3 (optional) → multishell (optional) →
    axis CA PDB (optional). Writes the usual per-phase files plus ``symmetry_find.json`` summary.
    """
    if (bandpass_low is None) ^ (bandpass_high is None):
        typer.echo("Provide both --bandpass-low and --bandpass-high, or neither.", err=True)
        raise typer.Exit(1)
    if edge not in ("none", "laplacian", "laplacian_sharpen"):
        typer.echo("--edge must be one of: none, laplacian, laplacian_sharpen", err=True)
        raise typer.Exit(1)
    tdparts = [float(x.strip()) for x in tilt.split(",") if x.strip()]
    if not tdparts:
        typer.echo("--tilt-deg must list at least one angle.", err=True)
        raise typer.Exit(1)
    oparts: Optional[list[int]] = None
    if orders is not None:
        oparts = [int(x.strip()) for x in orders.split(",") if x.strip()]
        if not oparts or any(n < 2 for n in oparts):
            typer.echo("--orders must list integers n ≥ 2.", err=True)
            raise typer.Exit(1)

    fam = family.strip().lower()
    if fam not in ("c", "d", "auto"):
        typer.echo("--family must be one of: c, d, auto", err=True)
        raise typer.Exit(1)
    md = mode.strip().lower()
    if md not in ("search", "guided"):
        typer.echo("--mode must be one of: search, guided", err=True)
        raise typer.Exit(1)
    if md == "guided" and guided_order is None:
        typer.echo("--mode guided requires --guided-order >= 2", err=True)
        raise typer.Exit(1)

    kwargs = dict(
        mask_path=mask,
        downsample_factor=downsample,
        bandpass_low_res_A=bandpass_low,
        bandpass_high_res_A=bandpass_high,
        edge_emphasis=edge,  # type: ignore[arg-type]
        laplacian_sharpen_strength=laplacian_strength,
        density_threshold=density_threshold,
        density_percentile=density_percentile if density_threshold is None else None,
        max_voxels_pca=max_voxels_pca,
        random_seed=seed,
        tilt_degrees=tuple(tdparts),
        include_diagonals=not no_diagonals,
        n_axial_bins=axial_bins,
        orders=tuple(oparts) if oparts is not None else None,
        mode=md,  # type: ignore[arg-type]
        guided_order=guided_order,
        max_phase2_candidates=max_candidates,
        run_phase3_step=not no_phase3,
        phase3_top=phase3_top,
        phase3_maxiter=phase3_maxiter,
        run_multishell_step=not no_multishell,
        n_shells=n_shells,
        multishell_radius_percentile=shell_radius_pct,
        run_axis_pdb_step=not no_axis_pdb,
        axis_pdb_path=out_pdb,
        reference_map_path=reference_map,
        axis_slice_step_voxels=slice_step,
        prefer_phase3_geometry=not no_phase3,
    )
    def _extract_axis_and_pivot_from_result(r):
        gb = r.phase2.global_best
        cand_id = int(gb.get("candidate_id", -1))
        axis = None
        for c in r.phase2.candidates:
            if int(c.get("id", -2)) == cand_id:
                axis = c.get("direction_xyz")
                break
        pivot = None
        if r.phase4 and r.phase4.output_json:
            try:
                with open(r.phase4.output_json, encoding="utf-8") as fh:
                    p4 = json.load(fh)
                axis = p4.get("axis_xyz", axis)
                pivot = p4.get("pivot_xyz", None)
            except Exception:
                pass
        return axis, pivot

    if fam == "auto":
        res_auto = run_symmetry_find_auto(map_path, out_dir, **kwargs)
        typer.echo(f"Auto summary: {res_auto.auto_summary_json}")
        typer.echo(f"Winner family: {res_auto.winner_family.upper()} (score={res_auto.winner_score:.4f})")
        typer.echo(f"C summary: {res_auto.c_result.symmetry_find_json}")
        typer.echo(f"D summary: {res_auto.d_result.symmetry_find_json}")
        win = res_auto.c_result if res_auto.winner_family == "c" else res_auto.d_result
        wgb = win.phase2.global_best
        wlabel = "C" if res_auto.winner_family == "c" else "D"
        axis, pivot = _extract_axis_and_pivot_from_result(win)
        typer.echo(f"Best symmetry: {wlabel}{int(wgb['n'])}  score={float(wgb['score']):.4f}")
        if axis is not None:
            typer.echo("Axis xyz: " + ", ".join(f"{float(v):.4f}" for v in axis))
        if pivot is not None:
            typer.echo("Pivot xyz (Å): " + ", ".join(f"{float(v):.2f}" for v in pivot))
        return

    res = run_symmetry_find(map_path, out_dir, family=fam, **kwargs)  # type: ignore[arg-type]
    typer.echo(f"Summary: {res.symmetry_find_json}")
    gb = res.phase2.global_best
    label = "D_n" if fam == "d" else "C_n"
    typer.echo(f"Global {label} best: n={gb['n']} score={gb['score']:.4f} (candidate {gb['candidate_id']})")
    axis, pivot = _extract_axis_and_pivot_from_result(res)
    if axis is not None:
        typer.echo("Axis xyz: " + ", ".join(f"{float(v):.4f}" for v in axis))
    if pivot is not None:
        typer.echo("Pivot xyz (Å): " + ", ".join(f"{float(v):.2f}" for v in pivot))
    if res.score_plot_png:
        typer.echo(f"Score plot: {res.score_plot_png}")
    if res.phase4:
        typer.echo(f"Axis PDB: {res.phase4.output_pdb}")


@app.command("score-plot")
@log_command("symmetry score-plot")
def score_plot(
    phase_json: Path = typer.Argument(..., exists=True, help="symmetry_phase2.json or symmetry_phase2d.json"),
    family: str = typer.Option("c", "--family", help="c for C_n plot, d for D_n plot"),
    out_png: Optional[Path] = typer.Option(None, "--out-png", "-o"),
):
    """Plot n-vs-best score (best candidate per n) for Cₙ or Dₙ."""
    fam = family.strip().lower()
    if fam not in ("c", "d"):
        typer.echo("--family must be one of: c, d", err=True)
        raise typer.Exit(1)
    from ..symmetry.score_plot import write_family_score_plot

    out = write_family_score_plot(phase_json, family=fam, out_png=out_png)  # type: ignore[arg-type]
    typer.echo(f"Wrote {out}")
