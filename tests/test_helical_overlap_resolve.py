"""Tests for helical mask overlap resolution."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from cryomodel.helical.overlap_resolve import resolve_binary_masks_to_labels, run_helical_resolve_overlaps
from cryomodel.io.mrc import MapVolume, read_map, write_map


def test_resolve_binary_masks_prefers_higher_density_in_overlap():
    z, y, x = 6, 6, 6
    dens = np.zeros((z, y, x), dtype=np.float32)
    dens[2, 2, 2] = 1.0
    dens[2, 2, 3] = 0.5
    m0 = np.zeros_like(dens, dtype=np.float32)
    m1 = np.zeros_like(dens, dtype=np.float32)
    m0[2, 2, 2] = 1.0
    m1[2, 2, 2] = 1.0
    m1[2, 2, 3] = 1.0
    labels, meta = resolve_binary_masks_to_labels(dens, [m0, m1], tie_break="density")
    assert int(labels[2, 2, 2]) == 1
    assert int(labels[2, 2, 3]) == 2
    assert meta["n_overlap_voxels"] == 1


def test_run_helical_resolve_overlaps_writes_maps(tmp_path: Path):
    z, y, x = 8, 8, 8
    dens = np.random.default_rng(0).random((z, y, x), dtype=np.float32) * 0.1 + 0.5
    m0 = np.zeros_like(dens)
    m1 = np.zeros_like(dens)
    m0[2:5, 3:6, 3:6] = 1.0
    m1[3:6, 3:6, 3:6] = 1.0
    dens[4, 4, 4] = 0.99
    mv = MapVolume(
        data_zyx=dens,
        apix=1.0,
        origin_xyzA=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        halfmaps=None,
        grid=None,
        _ccp4=None,
    )
    mp = tmp_path / "map.mrc"
    write_map(mp, mv, dens)
    m0p = tmp_path / "m0.mrc"
    m1p = tmp_path / "m1.mrc"
    write_map(m0p, mv, m0.astype(np.float32))
    write_map(m1p, mv, m1.astype(np.float32))

    out = tmp_path / "out"
    res = run_helical_resolve_overlaps(
        mp,
        [m0p, m1p],
        out,
        tie_break="density",
        write_representative=True,
        representative_largest_component=False,
    )
    assert Path(res.labels_map).is_file()
    assert Path(res.output_json).is_file()
    assert res.representative_map is not None and Path(res.representative_map).is_file()
    lab = read_map(res.labels_map).data_zyx
    assert float(np.max(lab)) <= 2.0
    assert res.n_overlap_voxels >= 1
