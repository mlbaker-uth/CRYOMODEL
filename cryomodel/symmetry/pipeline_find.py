"""One-shot ``symmetry find``: run phases 0→4 with optional multishell."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional

from .axis_candidates import Phase1Result, run_phase1_candidates
from .multishell_cn import MultishellResult, run_multishell_cn_scores
from .phase2_cn import Phase2Result, run_phase2_cn_scores
from .phase2_dn import Phase2DResult, run_phase2_dn_scores
from .phase3_dn_refine import Phase3DResult, run_phase3d_refine
from .phase3_refine import Phase3Result, run_phase3_refine
from .phase4_axis_pdb import Phase4Result, run_phase4_axis_pdb
from .preprocess import Phase0Result, run_phase0_preprocess

EdgeMode = Literal["none", "laplacian", "laplacian_sharpen"]
FamilyMode = Literal["c", "d"]
SearchMode = Literal["guided", "search"]
C_DEFAULT_ORDERS: tuple[int, ...] = tuple(range(2, 21))
D_DEFAULT_ORDERS: tuple[int, ...] = tuple(range(2, 13))


@dataclass
class SymmetryFindResult:
    """Paths to artifacts written under ``out_dir``."""

    out_dir: str
    symmetry_find_json: str
    phase0: Phase0Result
    phase1: Phase1Result
    phase2: Phase2Result | Phase2DResult
    phase3: Optional[Phase3Result | Phase3DResult]
    multishell: Optional[MultishellResult]
    phase4: Optional[Phase4Result]
    score_plot_png: Optional[str]


@dataclass
class SymmetryFindAutoResult:
    out_dir: str
    auto_summary_json: str
    c_result: SymmetryFindResult
    d_result: SymmetryFindResult
    winner_family: str
    winner_score: float


def run_symmetry_find(
    map_path: Path,
    out_dir: Path,
    *,
    mask_path: Optional[Path] = None,
    downsample_factor: int = 4,
    bandpass_low_res_A: Optional[float] = None,
    bandpass_high_res_A: Optional[float] = None,
    edge_emphasis: EdgeMode = "none",
    laplacian_sharpen_strength: float = 1.0,
    density_threshold: Optional[float] = None,
    density_percentile: Optional[float] = 90.0,
    max_voxels_pca: int = 400_000,
    random_seed: int = 0,
    tilt_degrees: tuple[float, ...] = (0.0, 5.0, 10.0, 15.0),
    include_diagonals: bool = True,
    n_axial_bins: int = 64,
    orders: Optional[tuple[int, ...]] = None,
    family: FamilyMode = "c",
    mode: SearchMode = "search",
    guided_order: Optional[int] = None,
    max_phase2_candidates: Optional[int] = None,
    run_phase3_step: bool = True,
    phase3_top: int = 3,
    phase3_max_tilt_deg: float = 5.0,
    phase3_max_shift_along_A: float = 10.0,
    phase3_max_shift_perp_A: float = 6.0,
    phase3_maxiter: int = 80,
    run_multishell_step: bool = True,
    n_shells: int = 8,
    multishell_max_radius_A: Optional[float] = None,
    multishell_radius_percentile: float = 99.5,
    multishell_min_voxels: int = 64,
    multishell_refinement_index: int = 0,
    run_axis_pdb_step: bool = True,
    axis_pdb_path: Optional[Path] = None,
    reference_map_path: Optional[Path] = None,
    axis_slice_step_voxels: float = 10.0,
    prefer_phase3_geometry: bool = True,
    write_score_plot: bool = True,
) -> SymmetryFindResult:
    """
    Run symmetry discovery end-to-end in ``out_dir``: phase 0 preprocess, phase 1 candidates,
    phase 2 Cₙ scores, optional phase 3 refinement, optional cylindrical multishell Cₙ scores
    about the chosen axis, optional phase 4 CA trace PDB.

    If ``run_phase3_step`` is False, multishell and the axis PDB use phase-2 global geometry
    (pivot = phase-0 COM) by forcing ``prefer_phase3_geometry`` off for those steps even if an
    old ``symmetry_phase3.json`` is present.
    """
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    p0 = run_phase0_preprocess(
        Path(map_path).expanduser().resolve(),
        out_dir=out_dir,
        mask_path=Path(mask_path).expanduser().resolve() if mask_path else None,
        downsample_factor=downsample_factor,
        bandpass_low_res_A=bandpass_low_res_A,
        bandpass_high_res_A=bandpass_high_res_A,
        edge_emphasis=edge_emphasis,
        laplacian_sharpen_strength=laplacian_sharpen_strength,
        density_threshold=density_threshold,
        density_percentile=density_percentile if density_threshold is None else None,
        max_voxels_pca=max_voxels_pca,
        random_seed=random_seed,
    )
    p1 = run_phase1_candidates(
        out_dir,
        tilt_degrees=tilt_degrees,
        include_diagonals=include_diagonals,
        n_axial_bins=n_axial_bins,
    )
    base_orders = orders if orders is not None else (D_DEFAULT_ORDERS if family == "d" else C_DEFAULT_ORDERS)
    ord_used = tuple(sorted({int(n) for n in base_orders if int(n) >= 2}))
    if mode == "guided":
        if guided_order is None or int(guided_order) < 2:
            raise ValueError("guided mode requires guided_order >= 2")
        ord_used = (int(guided_order),)

    p2_cn: Optional[Phase2Result] = None
    p2_dn: Optional[Phase2DResult] = None
    if family == "d":
        p2_dn = run_phase2_dn_scores(out_dir, orders=ord_used, max_candidates=max_phase2_candidates)
    else:
        p2_cn = run_phase2_cn_scores(out_dir, orders=ord_used, max_candidates=max_phase2_candidates)

    p3: Optional[Phase3Result] = None
    p3d: Optional[Phase3DResult] = None
    if run_phase3_step:
        if family == "d":
            p3d = run_phase3d_refine(
                out_dir,
                top_hypotheses=phase3_top,
                inplane_samples=36,
                max_tilt_deg=phase3_max_tilt_deg,
                max_shift_along_axis_A=phase3_max_shift_along_A,
                max_shift_perp_A=phase3_max_shift_perp_A,
                maxiter=phase3_maxiter,
            )
        else:
            p3 = run_phase3_refine(
                out_dir,
                top_hypotheses=phase3_top,
                max_tilt_deg=phase3_max_tilt_deg,
                max_shift_along_axis_A=phase3_max_shift_along_A,
                max_shift_perp_A=phase3_max_shift_perp_A,
                maxiter=phase3_maxiter,
            )

    use_p3 = bool(run_phase3_step and prefer_phase3_geometry)

    ms: Optional[MultishellResult] = None
    if run_multishell_step:
        ms = run_multishell_cn_scores(
            out_dir,
            family=family,
            orders=ord_used,
            n_shells=n_shells,
            max_radius_A=multishell_max_radius_A,
            radius_percentile_cap=multishell_radius_percentile,
            min_voxels_per_shell=multishell_min_voxels,
            refinement_index=multishell_refinement_index,
            prefer_phase3=use_p3,
        )

    p4: Optional[Phase4Result] = None
    if run_axis_pdb_step:
        p4 = run_phase4_axis_pdb(
            out_dir,
            family=family,
            out_pdb=axis_pdb_path,
            reference_map_path=reference_map_path,
            slice_step_voxels=axis_slice_step_voxels,
            refinement_index=multishell_refinement_index,
            prefer_phase3=use_p3,
        )

    score_plot_png: Optional[str] = None
    if write_score_plot:
        from .score_plot import write_family_score_plot
        phase_json = out_dir / ("symmetry_phase2d.json" if family == "d" else "symmetry_phase2.json")
        try:
            score_plot_png = str(write_family_score_plot(phase_json, family=family))
        except Exception:
            score_plot_png = None

    summary_path = out_dir / "symmetry_find.json"
    gb = (p2_dn.global_best if p2_dn is not None else p2_cn.global_best)  # type: ignore[union-attr]
    summary: dict[str, Any] = {
        "map_path": str(Path(map_path).expanduser().resolve()),
        "out_dir": str(out_dir),
        "phase0_json": p0.output_json,
        "phase1_json": p1.output_json,
        "family": family,
        "mode": mode,
        "guided_order": guided_order,
        "phase2_json": p2_cn.output_json if p2_cn else None,
        "phase2d_json": p2_dn.output_json if p2_dn else None,
        "global_best": dict(gb),
        "phase3_json": p3.output_json if p3 else None,
        "phase3d_json": p3d.output_json if p3d else None,
        "multishell_json": ms.output_json if ms else None,
        "phase4_pdb": p4.output_pdb if p4 else None,
        "phase4_json": p4.output_json if p4 else None,
        "score_plot_png": score_plot_png,
        "prefer_phase3_geometry_used": use_p3,
    }
    if ms:
        per_shell = []
        for sh in ms.shells:
            if sh.get("skipped"):
                per_shell.append({"shell_index": sh["shell_index"], "skipped": True})
            else:
                per_shell.append(
                    {
                        "shell_index": sh["shell_index"],
                        "r_inner_A": sh["r_inner_A"],
                        "r_outer_A": sh["r_outer_A"],
                        "best_n": sh.get("best_n"),
                        "best_score": sh.get("best_score"),
                        "n_voxels": sh.get("n_voxels"),
                    }
                )
        summary["multishell_shell_summary"] = per_shell

    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    return SymmetryFindResult(
        out_dir=str(out_dir),
        symmetry_find_json=str(summary_path),
        phase0=p0,
        phase1=p1,
        phase2=p2_cn if p2_cn is not None else p2_dn,  # type: ignore[arg-type]
        phase3=p3 if p3 is not None else p3d,  # type: ignore[arg-type]
        multishell=ms,
        phase4=p4,
        score_plot_png=score_plot_png,
    )


def run_symmetry_find_auto(
    map_path: Path,
    out_dir: Path,
    **kwargs: Any,
) -> SymmetryFindAutoResult:
    """
    Run both families (C and D) in isolated subdirectories, then write a combined
    comparison summary in ``out_dir/symmetry_find_auto.json``.
    """
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    c_dir = out_dir / "family_c"
    d_dir = out_dir / "family_d"
    c_res = run_symmetry_find(map_path, c_dir, family="c", **kwargs)
    d_res = run_symmetry_find(map_path, d_dir, family="d", **kwargs)

    c_best = float(c_res.phase2.global_best.get("score", -2.0))
    d_best = float(d_res.phase2.global_best.get("score", -2.0))
    winner_family = "c" if c_best >= d_best else "d"
    winner_score = c_best if c_best >= d_best else d_best

    out_json = out_dir / "symmetry_find_auto.json"
    payload: dict[str, Any] = {
        "out_dir": str(out_dir),
        "map_path": str(Path(map_path).expanduser().resolve()),
        "families": {
            "c": {
                "summary_json": c_res.symmetry_find_json,
                "global_best": dict(c_res.phase2.global_best),
                "score_plot_png": c_res.score_plot_png,
                "phase4_pdb": c_res.phase4.output_pdb if c_res.phase4 else None,
            },
            "d": {
                "summary_json": d_res.symmetry_find_json,
                "global_best": dict(d_res.phase2.global_best),
                "score_plot_png": d_res.score_plot_png,
                "phase4_pdb": d_res.phase4.output_pdb if d_res.phase4 else None,
            },
        },
        "winner_family": winner_family,
        "winner_score": winner_score,
    }
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    return SymmetryFindAutoResult(
        out_dir=str(out_dir),
        auto_summary_json=str(out_json),
        c_result=c_res,
        d_result=d_res,
        winner_family=winner_family,
        winner_score=float(winner_score),
    )
