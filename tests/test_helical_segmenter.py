"""Tests for helical segmenter outputs."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from cryomodel.helical.finder import rotation_matrix_axis_angle, run_helical_find
from cryomodel.helical.segmenter import (
    _prune_largest_cc_per_label_sparse,
    _rotation_matrix,
    run_helical_segment,
)
from cryomodel.io.mrc import MapVolume, read_map, write_map


def test_average_screw_uses_same_R_as_helical_finder():
    """Average inverse-sample extends screw_correlation with R(+dk*twist), same R as one-step finder."""
    u = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    th = 0.31
    assert np.max(np.abs(rotation_matrix_axis_angle(u, th) - _rotation_matrix(u, th))) < 1e-14
    p0 = np.zeros((1, 3), dtype=np.float64)
    P = np.array([[3.0, -1.0, 12.0]], dtype=np.float64)
    rise = 4.5
    shift = rise * u.reshape(1, 3)
    R = rotation_matrix_axis_angle(u, th)
    Q = p0 + ((P - p0 - shift) @ R)
    assert np.all(np.isfinite(Q))


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


def test_helical_segment_outputs_label_and_rep_maps(tmp_path: Path):
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
        write_average=True,
        max_norm_cost=12.0,
        min_cost_margin=0.0,
    )
    assert Path(sres.output_json).is_file()
    assert Path(sres.labels_map).is_file()
    assert Path(sres.representative_map).is_file()
    assert sres.qc_png is not None and Path(sres.qc_png).is_file()
    assert sres.average_map is not None and Path(sres.average_map).is_file()
    assert sres.n_subunits >= 2

    lab = read_map(sres.labels_map).data_zyx
    assert float(np.max(lab)) >= 2.0
    assert float(np.min(lab)) == 0.0

    with open(sres.output_json, encoding="utf-8") as fh:
        meta = json.load(fh)
    pp = meta.get("phase_peaks") or {}
    assert pp.get("shear_piecewise_dz") is True
    assert "shear_alpha_pos_rad_per_A" in pp and "shear_alpha_neg_rad_per_A" in pp


def test_helical_segment_explicit_two_slope(tmp_path: Path):
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
        max_norm_cost=12.0,
        min_cost_margin=0.0,
        shear_alpha_pos_rad_per_A=0.01,
        shear_alpha_neg_rad_per_A=-0.01,
    )
    with open(sres.output_json, encoding="utf-8") as fh:
        meta = json.load(fh)
    pp = meta["phase_peaks"]
    assert pp["shear_piecewise_dz"] is True
    assert abs(pp["shear_alpha_pos_rad_per_A"] - 0.01) < 1e-6
    assert abs(pp["shear_alpha_neg_rad_per_A"] - (-0.01)) < 1e-6


def test_prune_largest_cc_per_label_drops_small_blob():
    shape = (10, 10, 10)
    vol = np.zeros(shape, dtype=np.int32)
    vol[1:4, 4:7, 4:7] = 1
    vol[6:8, 4:6, 4:6] = 1
    iz, iy, ix = np.nonzero(vol)
    lab1d = vol[iz, iy, ix]
    out = _prune_largest_cc_per_label_sparse(lab1d, iz, iy, ix, shape)
    assert np.count_nonzero(out) < np.count_nonzero(lab1d)
    assert np.count_nonzero(out) == 27


def test_helical_segment_prune_and_watershed_gate_flags(tmp_path: Path):
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
        prune_labels_largest_component=True,
        watershed_max_norm_cost=25.0,
    )
    with open(sres.output_json, encoding="utf-8") as fh:
        meta = json.load(fh)
    assert meta.get("prune_labels_largest_component") is True
    assert meta.get("watershed_max_norm_cost") == 25.0

