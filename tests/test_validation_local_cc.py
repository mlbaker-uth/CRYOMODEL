"""Tests for resolution-aware local map–model CC (validation)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cryomodel.io.mrc import MapVolume, read_map, write_map
from cryomodel.maps.synthetic_from_model import synthetic_density_zyx_from_weighted_positions
from cryomodel.validation.feature_extractor import _get_local_resolution
from cryomodel.validation.local_cc import compute_local_cc_variants


def _fill_map_from_atoms(
    data: np.ndarray,
    atom_positions: np.ndarray,
    origin: np.ndarray,
    apix: float,
    resolution_A: float,
) -> None:
    """Ground truth: full-grid model2map synthesis (same as local CC uses per patch)."""
    nz, ny, nx = data.shape
    w = [(atom_positions[i], 1.0) for i in range(len(atom_positions))]
    syn = synthetic_density_zyx_from_weighted_positions(
        w, (nz, ny, nx), apix, origin, resolution_A, normalize_max=True
    )
    data[:] = syn


def test_cc_mask_near_one_when_map_matches_model() -> None:
    nz = ny = nx = 28
    origin = np.zeros(3, dtype=np.float32)
    apix = 1.0
    atoms = np.array([[14.0, 14.0, 14.0]], dtype=np.float64)
    res_a = 3.0
    mask_r = 4.0

    data = np.zeros((nz, ny, nx), dtype=np.float32)
    _fill_map_from_atoms(data, atoms, origin, apix, res_a)

    mv = MapVolume(data_zyx=data, apix=apix, origin_xyzA=origin, grid=None)
    out = compute_local_cc_variants(
        atoms, mv, mask_radius=mask_r, box_size=6.0, model_resolution_A=res_a
    )
    assert out["CC_mask"] == pytest.approx(1.0, abs=0.06)
    assert out["ZNCC"] == pytest.approx(1.0, abs=0.06)
    assert out["CC_box"] > 0.5


def test_cc_mask_drops_when_atom_misaligned() -> None:
    nz = ny = nx = 32
    origin = np.zeros(3, dtype=np.float32)
    apix = 1.0
    atoms_map = np.array([[16.0, 16.0, 16.0]], dtype=np.float64)
    atoms_model = np.array([[23.0, 16.0, 16.0]], dtype=np.float64)
    res_a = 3.0
    mask_r = 4.0

    data = np.zeros((nz, ny, nx), dtype=np.float32)
    _fill_map_from_atoms(data, atoms_map, origin, apix, res_a)

    mv = MapVolume(data_zyx=data, apix=apix, origin_xyzA=origin, grid=None)
    good = compute_local_cc_variants(atoms_map, mv, mask_radius=mask_r, model_resolution_A=res_a)
    bad = compute_local_cc_variants(atoms_model, mv, mask_radius=mask_r, model_resolution_A=res_a)
    assert good["CC_mask"] > 0.9
    assert bad["CC_mask"] < good["CC_mask"] - 0.1


def test_half_map_cc_shape_mismatch_returns_zero() -> None:
    data = np.zeros((10, 10, 10), dtype=np.float32)
    data[5, 5, 5] = 1.0
    mv = MapVolume(
        data_zyx=data,
        apix=1.0,
        origin_xyzA=np.zeros(3, dtype=np.float32),
        grid=None,
    )
    half = MapVolume(
        data_zyx=np.zeros((8, 8, 8), dtype=np.float32),
        apix=1.0,
        origin_xyzA=np.zeros(3, dtype=np.float32),
        grid=None,
    )
    atoms = np.array([[5.0, 5.0, 5.0]], dtype=np.float64)
    out = compute_local_cc_variants(atoms, mv, half1_vol=half, mask_radius=2.0)
    assert out["CC_half1"] == 0.0


def test_get_local_resolution_interpolates_with_gemmi_grid(tmp_path: Path) -> None:
    z, y, x = 8, 8, 8
    data = np.full((z, y, x), 2.5, dtype=np.float32)
    origin = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    mv0 = MapVolume(data_zyx=data, apix=1.0, origin_xyzA=origin)
    path = tmp_path / "locres.mrc"
    write_map(path, mv0, data)
    mv = read_map(path)
    assert mv.grid is not None
    pos = np.array([1.5, 2.5, 3.5], dtype=np.float64)
    v = _get_local_resolution(mv, pos)
    assert v == pytest.approx(2.5, rel=1e-4)
