"""Tests for helical symmetry finder."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from cryomodel.helical.finder import run_helical_find
from cryomodel.io.mrc import MapVolume, write_map


def _synthetic_helix_zyx(
    nz: int = 80,
    ny: int = 64,
    nx: int = 64,
    rise_A: float = 3.0,
    twist_deg: float = 30.0,
    apix: float = 1.0,
) -> np.ndarray:
    zz, yy, xx = np.meshgrid(
        np.arange(nz, dtype=np.float64),
        np.arange(ny, dtype=np.float64),
        np.arange(nx, dtype=np.float64),
        indexing="ij",
    )
    cz = (nz - 1) / 2.0
    cy = (ny - 1) / 2.0
    cx = (nx - 1) / 2.0
    radius = 9.0
    sig2 = 1.8**2
    dens = np.zeros((nz, ny, nx), dtype=np.float64)
    nsteps = int((nz * apix) / rise_A) + 4
    for k in range(-2, nsteps):
        zA = k * rise_A
        z = cz + zA / apix
        phi = np.deg2rad(k * twist_deg)
        x = cx + radius * np.cos(phi)
        y = cy + radius * np.sin(phi)
        dens += np.exp(-0.5 * (((xx - x) ** 2) + ((yy - y) ** 2) + ((zz - z) ** 2)) / sig2)
    return dens.astype(np.float32)


def test_helical_find_recovers_rise_twist_near_truth(tmp_path: Path):
    rise_true = 3.0
    twist_true = 30.0
    data = _synthetic_helix_zyx(rise_A=rise_true, twist_deg=twist_true)
    mv = MapVolume(
        data_zyx=data,
        apix=1.0,
        origin_xyzA=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        halfmaps=None,
        grid=None,
        _ccp4=None,
    )
    mpath = tmp_path / "helix.mrc"
    write_map(mpath, mv, data)
    out_dir = tmp_path / "out"
    res = run_helical_find(
        mpath,
        out_dir,
        density_percentile=88.0,
        axis_mode="cardinal",
        twist_min_deg=20.0,
        twist_max_deg=40.0,
        twist_step_deg=1.0,
        rise_min_A=2.0,
        rise_max_A=4.5,
        rise_step_A=0.2,
        max_voxels_score=120_000,
        refine_iters=2,
        write_heatmap=True,
    )
    assert Path(res.output_json).is_file()
    assert res.heatmap_png is not None and Path(res.heatmap_png).is_file()
    assert abs(res.best_rise_A - rise_true) < 0.7
    assert abs(abs(res.best_twist_deg) - abs(twist_true)) < 6.0
    # axis expected along z for this synthetic setup
    assert abs(float(res.axis_xyz[2])) > 0.8

