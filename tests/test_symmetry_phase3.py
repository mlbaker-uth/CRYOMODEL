"""Tests for symmetry phase-3 Cₙ axis/pivot refinement."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from cryomodel.io.mrc import MapVolume, write_map
from cryomodel.symmetry.axis_candidates import run_phase1_candidates
from cryomodel.symmetry.phase2_cn import run_phase2_cn_scores
from cryomodel.symmetry.phase3_refine import refine_cn_axis_pivot, run_phase3_refine
from cryomodel.symmetry.preprocess import run_phase0_preprocess


def _c2_two_blobs_zyx(nz: int = 40, ny: int = 40, nx: int = 40) -> np.ndarray:
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


def _mask_indices_com(data: np.ndarray, origin: np.ndarray, apix: float, thr_pct: float = 85.0):
    thr = float(np.percentile(data, thr_pct))
    sel = data > thr
    iz, iy, ix = np.nonzero(sel)
    w = data[sel].astype(np.float64)
    com = np.array(
        [
            (w * (origin[0] + (ix + 0.5) * apix)).sum() / w.sum(),
            (w * (origin[1] + (iy + 0.5) * apix)).sum() / w.sum(),
            (w * (origin[2] + (iz + 0.5) * apix)).sum() / w.sum(),
        ]
    )
    return iz, iy, ix, com, thr


def test_refine_cn_improves_misaligned_axis():
    data = _c2_two_blobs_zyx()
    apix = 1.0
    origin = np.zeros(3, dtype=np.float64)
    iz, iy, ix, com, _thr = _mask_indices_com(data, origin, apix)
    deg = 4.0
    rad = np.deg2rad(deg)
    u_tilt = np.array([np.sin(rad), 0.0, np.cos(rad)], dtype=np.float64)
    u_tilt /= np.linalg.norm(u_tilt)
    out = refine_cn_axis_pivot(
        data,
        iz,
        iy,
        ix,
        origin,
        apix,
        u_tilt,
        com,
        2,
        max_tilt_deg=8.0,
        max_shift_along_axis_A=6.0,
        max_shift_perp_A=4.0,
        maxiter=100,
    )
    assert out["refined_score"] >= out["phase2_score"] - 1e-9
    assert out["refined_score"] > out["phase2_score"] + 0.005
    u_ref = np.array(out["refined_axis_xyz"], dtype=np.float64)
    u_ref /= np.linalg.norm(u_ref)
    uz = abs(float(np.dot(u_ref, [0.0, 0.0, 1.0])))
    assert uz > abs(float(np.dot(u_tilt, [0.0, 0.0, 1.0])))


def test_phase3_requires_phase2_json(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        run_phase3_refine(tmp_path)


def test_phase3_pipeline_writes_json(tmp_path: Path):
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
    run_phase2_cn_scores(out_dir, orders=(2, 3, 4, 5, 6))
    res = run_phase3_refine(out_dir, top_hypotheses=2, maxiter=60)
    assert Path(res.output_json).is_file()
    with open(res.output_json, encoding="utf-8") as fh:
        payload = json.load(fh)
    assert len(payload["refinements"]) <= 2
    for row in payload["refinements"]:
        assert "refined_score" in row
        assert row["n"] >= 2
