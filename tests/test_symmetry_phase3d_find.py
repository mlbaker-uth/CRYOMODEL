"""Tests for D-aware refinement and family-aware find pipeline."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from cryomodel.io.mrc import MapVolume, write_map
from cryomodel.symmetry.axis_candidates import run_phase1_candidates
from cryomodel.symmetry.phase2_dn import run_phase2_dn_scores
from cryomodel.symmetry.phase3_dn_refine import run_phase3d_refine
from cryomodel.symmetry.pipeline_find import run_symmetry_find
from cryomodel.symmetry.preprocess import run_phase0_preprocess


def _d2_four_blobs_zyx(nz: int = 40, ny: int = 40, nx: int = 40) -> np.ndarray:
    zz, yy, xx = np.meshgrid(
        np.arange(nz, dtype=np.float64),
        np.arange(ny, dtype=np.float64),
        np.arange(nx, dtype=np.float64),
        indexing="ij",
    )
    cz = (nz - 1) / 2.0
    cy = (ny - 1) / 2.0
    cx = (nx - 1) / 2.0
    dx = 7.0
    dy = 5.0
    sig2 = 2.5**2
    dens = np.zeros((nz, ny, nx), dtype=np.float64)
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            dens += np.exp(
                -0.5
                * (
                    ((xx - (cx + sx * dx)) ** 2)
                    + ((yy - (cy + sy * dy)) ** 2)
                    + ((zz - cz) ** 2)
                )
                / sig2
            )
    return dens.astype(np.float32)


def test_phase3d_writes_json(tmp_path: Path):
    dens = _d2_four_blobs_zyx()
    mv = MapVolume(
        data_zyx=dens,
        apix=1.0,
        origin_xyzA=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        halfmaps=None,
        grid=None,
        _ccp4=None,
    )
    mpath = tmp_path / "d2.mrc"
    write_map(mpath, mv, dens)
    out_dir = tmp_path / "sym"
    run_phase0_preprocess(mpath, out_dir=out_dir, downsample_factor=1, density_percentile=80.0, max_voxels_pca=50_000)
    run_phase1_candidates(out_dir, tilt_degrees=(0.0,), include_diagonals=False)
    run_phase2_dn_scores(out_dir, orders=(2, 3, 4), inplane_samples=24)
    res = run_phase3d_refine(out_dir, top_hypotheses=2, inplane_samples=24, maxiter=50)
    assert Path(res.output_json).is_file()
    assert len(res.refinements) >= 1


def test_find_d_guided_writes_plot(tmp_path: Path):
    dens = _d2_four_blobs_zyx()
    mv = MapVolume(
        data_zyx=dens,
        apix=1.0,
        origin_xyzA=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        halfmaps=None,
        grid=None,
        _ccp4=None,
    )
    mpath = tmp_path / "d2.mrc"
    write_map(mpath, mv, dens)
    out_dir = tmp_path / "findd"
    res = run_symmetry_find(
        mpath,
        out_dir,
        downsample_factor=1,
        density_percentile=80.0,
        max_voxels_pca=50_000,
        tilt_degrees=(0.0,),
        include_diagonals=False,
        family="d",
        mode="guided",
        guided_order=2,
        run_multishell_step=True,
        n_shells=3,
        multishell_min_voxels=8,
        run_axis_pdb_step=True,
        write_score_plot=True,
    )
    assert res.score_plot_png is not None
    assert Path(res.score_plot_png).is_file()
    assert Path(res.symmetry_find_json).is_file()
