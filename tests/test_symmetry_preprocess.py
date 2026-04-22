"""Tests for symmetry phase-0 preprocessing."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from cryomodel.io.mrc import MapVolume, write_map
from cryomodel.symmetry.preprocess import downsample_block_mean, run_phase0_preprocess


def test_downsample_block_mean_identity():
    x = np.arange(64, dtype=np.float32).reshape(4, 4, 4)
    y = downsample_block_mean(x, 1)
    np.testing.assert_array_almost_equal(y, x)


def test_downsample_block_mean_factor2():
    x = np.ones((8, 8, 8), dtype=np.float32)
    y = downsample_block_mean(x, 2)
    assert y.shape == (4, 4, 4)
    np.testing.assert_allclose(y, 1.0)


def test_phase0_principal_axis_elongated_along_z(tmp_path: Path):
    """Gaussian ellipsoid σz >> σx,σy → primary axis ≈ z."""
    nz, ny, nx = 48, 32, 32
    z = np.linspace(-1, 1, nz)[:, None, None]
    y = np.linspace(-1, 1, ny)[None, :, None]
    x = np.linspace(-1, 1, nx)[None, None, :]
    # Elongated along z (narrow in x,y)
    dens = np.exp(-8.0 * (x * x + y * y) - 0.5 * (z * z))
    dens = dens.astype(np.float32)

    apix = 1.5
    mv = MapVolume(
        data_zyx=dens,
        apix=apix,
        origin_xyzA=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        halfmaps=None,
        grid=None,
        _ccp4=None,
    )
    mpath = tmp_path / "ellipsoid.mrc"
    write_map(mpath, mv, dens)

    out_dir = tmp_path / "out"
    res = run_phase0_preprocess(
        mpath,
        out_dir=out_dir,
        downsample_factor=2,
        density_percentile=50.0,
        max_voxels_pca=50_000,
    )

    assert Path(res.output_map).is_file()
    assert Path(res.output_json).is_file()
    primary = np.array(res.principal_axes_xyz[0], dtype=np.float64)
    n = np.linalg.norm(primary)
    assert n > 0
    primary /= n
    # Expect dominant direction along z (0,0,1) in x,y,z column order
    assert abs(primary[2]) > 0.85
    assert abs(primary[0]) < 0.35
    assert abs(primary[1]) < 0.35

    with open(res.output_json, encoding="utf-8") as fh:
        payload = json.load(fh)
    assert tuple(payload["shape_out"]) == downsample_block_mean(dens, 2).shape
    assert "principal_axes_xyz" in payload
