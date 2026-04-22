"""Tests for symmetry find auto-family mode."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from cryomodel.io.mrc import MapVolume, write_map
from cryomodel.symmetry.pipeline_find import run_symmetry_find_auto


def _d2_four_blobs_zyx(nz: int = 32, ny: int = 32, nx: int = 32) -> np.ndarray:
    zz, yy, xx = np.meshgrid(
        np.arange(nz, dtype=np.float64),
        np.arange(ny, dtype=np.float64),
        np.arange(nx, dtype=np.float64),
        indexing="ij",
    )
    cz = (nz - 1) / 2.0
    cy = (ny - 1) / 2.0
    cx = (nx - 1) / 2.0
    dx = 6.0
    dy = 4.0
    sig2 = 2.0**2
    dens = np.zeros((nz, ny, nx), dtype=np.float64)
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            dens += np.exp(-0.5 * (((xx - (cx + sx * dx)) ** 2) + ((yy - (cy + sy * dy)) ** 2) + ((zz - cz) ** 2)) / sig2)
    return dens.astype(np.float32)


def test_run_symmetry_find_auto_writes_combined_summary(tmp_path: Path):
    dens = _d2_four_blobs_zyx()
    mv = MapVolume(
        data_zyx=dens,
        apix=1.0,
        origin_xyzA=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        halfmaps=None,
        grid=None,
        _ccp4=None,
    )
    mpath = tmp_path / "in.mrc"
    write_map(mpath, mv, dens)
    out_dir = tmp_path / "auto"
    res = run_symmetry_find_auto(
        mpath,
        out_dir,
        downsample_factor=1,
        density_percentile=80.0,
        max_voxels_pca=50_000,
        tilt_degrees=(0.0,),
        include_diagonals=False,
        mode="guided",
        guided_order=2,
        run_multishell_step=False,
        run_axis_pdb_step=False,
    )
    assert Path(res.auto_summary_json).is_file()
    assert Path(res.c_result.symmetry_find_json).is_file()
    assert Path(res.d_result.symmetry_find_json).is_file()
    assert res.winner_family in ("c", "d")
    # Guided mode constrains orders to one value for both families.
    assert res.c_result.phase2.orders == [2]
    assert res.d_result.phase2.orders == [2]
