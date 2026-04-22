"""Tests for symmetry phase-2D (Dₙ) scoring."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from cryomodel.io.mrc import MapVolume, write_map
from cryomodel.symmetry.axis_candidates import run_phase1_candidates
from cryomodel.symmetry.phase2_dn import dn_rotation_correlation, run_phase2_dn_scores
from cryomodel.symmetry.preprocess import run_phase0_preprocess


def _d2_four_blobs_zyx(nz: int = 40, ny: int = 40, nx: int = 40) -> np.ndarray:
    """Four blobs at (±x, ±y, z=0), consistent with D2 about z + perpendicular C2 axes."""
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


def test_dn_rotation_correlation_d2_about_z_high():
    data = _d2_four_blobs_zyx()
    origin = np.zeros(3, dtype=np.float64)
    apix = 1.0
    thr = float(np.percentile(data, 85.0))
    sel = data > thr
    iz, iy, ix = np.nonzero(sel)
    w = data[sel].astype(np.float64)
    pivot = np.array(
        [
            (w * (origin[0] + (ix + 0.5) * apix)).sum() / w.sum(),
            (w * (origin[1] + (iy + 0.5) * apix)).sum() / w.sum(),
            (w * (origin[2] + (iz + 0.5) * apix)).sum() / w.sum(),
        ]
    )
    axis_z = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    d2 = dn_rotation_correlation(data, iz, iy, ix, origin, apix, pivot, axis_z, 2, inplane_samples=36)
    assert d2["dn_score"] > 0.85
    assert d2["cn_component"] > 0.9
    assert d2["c2_perp_component"] > 0.8


def test_phase2d_pipeline_writes_json(tmp_path: Path):
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
    run_phase0_preprocess(
        mpath,
        out_dir=out_dir,
        downsample_factor=1,
        density_percentile=80.0,
        max_voxels_pca=50_000,
    )
    run_phase1_candidates(out_dir, tilt_degrees=(0.0,), include_diagonals=False)
    res = run_phase2_dn_scores(out_dir, orders=(2, 3, 4), inplane_samples=24)
    assert Path(res.output_json).is_file()
    with open(res.output_json, encoding="utf-8") as fh:
        payload = json.load(fh)
    gb = payload["global_best"]
    assert gb["n"] == 2
    assert gb["score"] > 0.7

