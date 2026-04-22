"""Tests for local helical boundary refinement."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from cryomodel.helical.finder import run_helical_find
from cryomodel.helical.local_refine import run_helical_refine_local
from cryomodel.helical.segmenter import run_helical_segment
from cryomodel.io.mrc import MapVolume, read_map, write_map


def _synthetic_helix_zyx(
    nz: int = 72,
    ny: int = 56,
    nx: int = 56,
    rise_A: float = 3.0,
    twist_deg: float = 24.0,
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
    radius = 8.0
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


def test_segment_json_includes_repeat_center_t_A(tmp_path: Path):
    data = _synthetic_helix_zyx()
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
    out_find = tmp_path / "find"
    fres = run_helical_find(
        mpath,
        out_find,
        density_percentile=88.0,
        axis_mode="cardinal",
        twist_min_deg=12.0,
        twist_max_deg=36.0,
        twist_step_deg=1.0,
        rise_min_A=2.0,
        rise_max_A=4.5,
        rise_step_A=0.2,
        write_heatmap=False,
    )
    out_seg = tmp_path / "seg"
    run_helical_segment(
        mpath,
        Path(fres.output_json),
        out_seg,
        mode="phase_peaks",
        write_average=False,
        write_qc_png=False,
        max_norm_cost=12.0,
        min_cost_margin=0.0,
    )
    with open(out_seg / "helical_segment.json", encoding="utf-8") as fh:
        meta = json.load(fh)
    pp = meta.get("phase_peaks") or {}
    assert "repeat_center_t_A" in pp
    assert len(pp["repeat_center_t_A"]) == meta["n_subunits"]


def test_refine_local_runs(tmp_path: Path):
    data = _synthetic_helix_zyx()
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
    out_find = tmp_path / "find"
    fres = run_helical_find(
        mpath,
        out_find,
        density_percentile=88.0,
        axis_mode="cardinal",
        twist_min_deg=12.0,
        twist_max_deg=36.0,
        twist_step_deg=1.0,
        rise_min_A=2.0,
        rise_max_A=4.5,
        rise_step_A=0.2,
        write_heatmap=False,
    )
    out_seg = tmp_path / "seg"
    sres = run_helical_segment(
        mpath,
        Path(fres.output_json),
        out_seg,
        mode="phase_peaks",
        write_average=False,
        write_qc_png=False,
        max_norm_cost=12.0,
        min_cost_margin=0.0,
    )
    out_ref = tmp_path / "ref"
    rres = run_helical_refine_local(
        mpath,
        Path(sres.labels_map),
        Path(sres.output_json),
        out_ref,
        neighbor_layers=2,
        pad_voxels=6,
    )
    assert Path(rres.labels_map).is_file()
    assert Path(rres.representative_map).is_file()
    assert Path(rres.representative_mask_map).is_file()
    lab = read_map(rres.labels_map).data_zyx
    assert float(np.max(lab)) >= 1.0
    mask = read_map(rres.representative_mask_map).data_zyx
    assert np.any(mask > 0.5)
