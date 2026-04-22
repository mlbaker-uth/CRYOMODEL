"""Tests for resampling maps onto a primary grid."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cryomodel.io.map_resample import maps_grid_compatible, resample_map_volume
from cryomodel.io.mrc import MapVolume, read_map, write_map


def test_resample_identity_roundtrip(tmp_path: Path) -> None:
    nz, ny, nx = 12, 10, 14
    data = np.random.RandomState(0).rand(nz, ny, nx).astype(np.float32)
    origin = np.array([-1.0, 2.0, 3.5], dtype=np.float32)
    apix = 1.25
    mv0 = MapVolume(data_zyx=data, apix=apix, origin_xyzA=origin)
    path = tmp_path / "m.mrc"
    write_map(path, mv0, data)
    a = read_map(path)
    b = read_map(path)
    assert maps_grid_compatible(a, b)
    r = resample_map_volume(b, a)
    assert r is b
    assert np.allclose(a.data_zyx, b.data_zyx)


def test_resample_coarser_target_matches_integral_mean(tmp_path: Path) -> None:
    """Fine constant source (apix 1); coarse target (apix 2) same physical frame — mean preserved."""
    fine = np.ones((20, 20, 20), dtype=np.float32) * 2.5
    origin = np.zeros(3, dtype=np.float32)
    mv_f = MapVolume(data_zyx=fine, apix=1.0, origin_xyzA=origin)
    p_f = tmp_path / "fine.mrc"
    write_map(p_f, mv_f, fine)
    src = read_map(p_f)

    coarse = np.zeros((10, 10, 10), dtype=np.float32)
    mv_c = MapVolume(data_zyx=coarse, apix=2.0, origin_xyzA=origin)
    p_c = tmp_path / "coarse.mrc"
    write_map(p_c, mv_c, coarse)
    tgt = read_map(p_c)

    assert not maps_grid_compatible(src, tgt)
    out = resample_map_volume(src, tgt)
    assert out.data_zyx.shape == tgt.data_zyx.shape
    assert float(np.mean(out.data_zyx)) == pytest.approx(2.5, abs=0.15)
