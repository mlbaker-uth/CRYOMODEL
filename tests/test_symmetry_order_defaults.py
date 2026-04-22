"""Default order-range behavior for symmetry find."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from cryomodel.io.mrc import MapVolume, write_map
from cryomodel.symmetry.pipeline_find import run_symmetry_find


def _tiny_map(tmp_path: Path) -> Path:
    dens = np.ones((16, 16, 16), dtype=np.float32) * 0.1
    dens[6:10, 7:9, 7:9] = 2.0
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


def test_find_default_orders_are_family_specific(tmp_path: Path):
    mpath = _tiny_map(tmp_path)
    c = run_symmetry_find(
        mpath,
        tmp_path / "c",
        family="c",
        downsample_factor=1,
        density_percentile=50.0,
        max_voxels_pca=10_000,
        tilt_degrees=(0.0,),
        include_diagonals=False,
        run_phase3_step=False,
        run_multishell_step=False,
        run_axis_pdb_step=False,
    )
    d = run_symmetry_find(
        mpath,
        tmp_path / "d",
        family="d",
        downsample_factor=1,
        density_percentile=50.0,
        max_voxels_pca=10_000,
        tilt_degrees=(0.0,),
        include_diagonals=False,
        run_phase3_step=False,
        run_multishell_step=False,
        run_axis_pdb_step=False,
    )
    assert max(c.phase2.orders) >= 20
    assert max(d.phase2.orders) <= 12
