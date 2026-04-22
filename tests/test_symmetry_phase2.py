"""Tests for symmetry phase-2 Cₙ rotational self-correlation."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from cryomodel.io.mrc import MapVolume, write_map
from cryomodel.symmetry.axis_candidates import run_phase1_candidates
from cryomodel.symmetry.phase2_cn import cn_rotation_correlation, run_phase2_cn_scores
from cryomodel.symmetry.preprocess import run_phase0_preprocess


def _c2_two_blobs_zyx(nz: int = 40, ny: int = 40, nx: int = 40) -> np.ndarray:
    """Two Gaussians on ±x from center; invariant under 180° about z through COM."""
    zz, yy, xx = np.meshgrid(
        np.arange(nz, dtype=np.float64),
        np.arange(ny, dtype=np.float64),
        np.arange(nx, dtype=np.float64),
        indexing="ij",
    )
    cz, cy = (nz - 1) / 2.0, (ny - 1) / 2.0
    cx1 = cx2 = (nx - 1) / 2.0
    dx = 8.0
    sig2 = 3.0**2
    g1 = np.exp(-0.5 * ((xx - (cx1 - dx)) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2) / sig2)
    g2 = np.exp(-0.5 * ((xx - (cx2 + dx)) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2) / sig2)
    return (g1 + g2).astype(np.float32)


def test_cn_rotation_correlation_c2_about_z_high():
    data = _c2_two_blobs_zyx()
    nz, ny, nx = data.shape
    apix = 1.0
    origin = np.zeros(3, dtype=np.float64)
    thr = float(np.percentile(data, 85.0))
    sel = data > thr
    iz, iy, ix = np.nonzero(sel)
    assert iz.size > 100
    w = data[sel].astype(np.float64)
    com = np.array(
        [
            (w * (origin[0] + (ix + 0.5) * apix)).sum() / w.sum(),
            (w * (origin[1] + (iy + 0.5) * apix)).sum() / w.sum(),
            (w * (origin[2] + (iz + 0.5) * apix)).sum() / w.sum(),
        ]
    )
    axis_z = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    r2 = cn_rotation_correlation(data, iz, iy, ix, origin, apix, com, axis_z, 2)
    r3 = cn_rotation_correlation(data, iz, iy, ix, origin, apix, com, axis_z, 3)
    assert r2 > 0.92
    assert r3 < r2


def test_phase2_requires_phase_artifacts(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        run_phase2_cn_scores(tmp_path)


def test_phase2_pipeline_finds_c2_on_synthetic(tmp_path: Path):
    dens = _c2_two_blobs_zyx()
    mv = MapVolume(
        data_zyx=dens,
        apix=1.0,
        origin_xyzA=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        halfmaps=None,
        grid=None,
        _ccp4=None,
    )
    mpath = tmp_path / "c2.mrc"
    write_map(mpath, mv, dens)
    out_dir = tmp_path / "sym"
    run_phase0_preprocess(
        mpath,
        out_dir=out_dir,
        downsample_factor=1,
        density_percentile=80.0,
        max_voxels_pca=50_000,
    )
    run_phase1_candidates(out_dir, tilt_degrees=(0.0,), include_diagonals=False)
    res = run_phase2_cn_scores(out_dir, orders=(2, 3, 4, 5, 6))
    assert Path(res.output_json).is_file()
    with open(res.output_json, encoding="utf-8") as fh:
        payload = json.load(fh)
    gb = payload["global_best"]
    assert gb["n"] == 2
    assert gb["score"] > 0.85
