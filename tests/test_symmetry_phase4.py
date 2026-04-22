"""Tests for symmetry phase-4 axis PDB export."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from cryomodel.io.mrc import MapVolume, write_map
from cryomodel.symmetry.axis_candidates import run_phase1_candidates
from cryomodel.symmetry.phase2_cn import run_phase2_cn_scores
from cryomodel.symmetry.phase4_axis_pdb import (
    ray_aabb_intersect_t,
    run_phase4_axis_pdb,
    sample_axis_parameters,
)


def test_ray_aabb_hits_unit_box():
    bmin = np.zeros(3)
    bmax = np.array([10.0, 10.0, 10.0])
    o = np.array([5.0, 5.0, -5.0])
    d = np.array([0.0, 0.0, 1.0])
    hit = ray_aabb_intersect_t(o, d, bmin, bmax)
    assert hit is not None
    t0, t1 = hit
    assert t0 < t1
    assert abs(t0 - 5.0) < 1e-6 and abs(t1 - 15.0) < 1e-6


def test_sample_axis_parameters_includes_ends():
    ts = sample_axis_parameters(0.0, 100.0, step_along_axis_A=10.0)
    assert ts[0] == 0.0
    assert abs(ts[-1] - 100.0) < 1e-3
    assert len(ts) >= 11


def _tiny_map(tmp_path: Path) -> Path:
    dens = np.ones((20, 20, 20), dtype=np.float32) * 0.1
    dens[8:12, 9:11, 9:11] = 2.0
    mv = MapVolume(
        data_zyx=dens,
        apix=1.0,
        origin_xyzA=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        halfmaps=None,
        grid=None,
        _ccp4=None,
    )
    p = tmp_path / "in.mrc"
    write_map(p, mv, dens)
    return p


def test_phase4_writes_pdb_and_json(tmp_path: Path):
    from cryomodel.symmetry.preprocess import run_phase0_preprocess

    mpath = _tiny_map(tmp_path)
    out_dir = tmp_path / "sym"
    run_phase0_preprocess(
        mpath,
        out_dir=out_dir,
        downsample_factor=1,
        density_percentile=50.0,
        max_voxels_pca=10_000,
    )
    run_phase1_candidates(out_dir, tilt_degrees=(0.0,), include_diagonals=False)
    run_phase2_cn_scores(out_dir, orders=(2, 3))

    res = run_phase4_axis_pdb(out_dir, slice_step_voxels=5.0, prefer_phase3=False)
    assert Path(res.output_pdb).is_file()
    assert Path(res.output_json).is_file()
    text = Path(res.output_pdb).read_text(encoding="utf-8")
    assert "ATOM" in text
    assert "REMARK" in text
    assert text.count("ATOM") == res.n_points
    with open(res.output_json, encoding="utf-8") as fh:
        meta = json.load(fh)
    assert meta["symmetry_family"] == "C"
    assert meta["n_fold"] >= 2
