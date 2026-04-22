"""Map sampling: gemmi grid path vs legacy numpy (synthetic volumes)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cryomodel.io.mrc import MapVolume, read_map, write_map
from cryomodel.validation.ringer_lite import sample_density_at_position


def test_read_map_grid_aligns_with_data_zyx_corner(tmp_path: Path):
    """Gemmi grid frame must match the numpy slab we keep in MapVolume.data_zyx."""
    z, y, x = 12, 10, 14
    data = np.arange(z * y * x, dtype=np.float32).reshape(z, y, x) * 0.01
    origin = np.array([-5.5, 12.0, 3.25], dtype=np.float32)
    apix = 1.375
    mv0 = MapVolume(data_zyx=data, apix=apix, origin_xyzA=origin)
    path = tmp_path / "t.mrc"
    write_map(path, mv0, data)
    mv = read_map(path)
    assert mv.grid is not None
    p0 = mv.grid.get_position(0, 0, 0)
    pos = np.array([p0.x, p0.y, p0.z], dtype=np.float64)
    v = sample_density_at_position(mv, pos, density_threshold=0.0)
    assert v == pytest.approx(float(mv.data_zyx[0, 0, 0]), rel=1e-4, abs=1e-5)


def test_sample_density_threshold_with_grid(tmp_path: Path):
    data = np.full((4, 4, 4), 2.5, dtype=np.float32)
    origin = np.zeros(3, dtype=np.float32)
    mv0 = MapVolume(data_zyx=data, apix=1.0, origin_xyzA=origin)
    path = tmp_path / "u.mrc"
    write_map(path, mv0, data)
    mv1 = read_map(path)
    p = np.array([0.5, 0.5, 0.5])
    assert sample_density_at_position(mv1, p, density_threshold=1.0) == pytest.approx(1.5)
