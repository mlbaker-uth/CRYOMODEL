"""Tests for symmetry multishell scoring and ``find`` pipeline."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from cryomodel.io.mrc import MapVolume, write_map
from cryomodel.symmetry.axis_candidates import run_phase1_candidates
from cryomodel.symmetry.multishell_cn import run_multishell_cn_scores
from cryomodel.symmetry.phase2_cn import run_phase2_cn_scores
from cryomodel.symmetry.pipeline_find import run_symmetry_find
from cryomodel.symmetry.preprocess import run_phase0_preprocess


def _tiny_map(tmp_path) -> Path:
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


def test_multishell_writes_json(tmp_path):
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
    run_phase2_cn_scores(out_dir, orders=(2, 3, 4))
    res = run_multishell_cn_scores(
        out_dir,
        orders=(2, 3, 4),
        n_shells=4,
        prefer_phase3=False,
        min_voxels_per_shell=8,
    )
    assert Path(res.output_json).is_file()
    with open(res.output_json, encoding="utf-8") as fh:
        data = json.load(fh)
    assert len(data["shells"]) == 4
    assert any(not sh.get("skipped") for sh in data["shells"])


def test_find_writes_summary_and_artifacts(tmp_path):
    mpath = _tiny_map(tmp_path)
    out_dir = tmp_path / "find_out"
    res = run_symmetry_find(
        mpath,
        out_dir,
        downsample_factor=1,
        density_percentile=50.0,
        max_voxels_pca=10_000,
        tilt_degrees=(0.0,),
        include_diagonals=False,
        orders=(2, 3, 4, 5),
        run_phase3_step=False,
        run_multishell_step=True,
        n_shells=3,
        multishell_min_voxels=8,
        run_axis_pdb_step=True,
        axis_slice_step_voxels=5.0,
        prefer_phase3_geometry=False,
    )
    assert Path(res.symmetry_find_json).is_file()
    assert Path(res.phase0.output_json).is_file()
    assert Path(res.phase2.output_json).is_file()
    assert res.multishell is not None
    assert res.phase4 is not None
    with open(res.symmetry_find_json, encoding="utf-8") as fh:
        summary = json.load(fh)
    assert summary["multishell_json"]
    assert summary["phase4_pdb"]
    assert "multishell_shell_summary" in summary
