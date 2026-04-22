"""Resample MRC/CCP4 volumes onto another map's voxel grid (Gemmi interpolation)."""
from __future__ import annotations

import numpy as np
import gemmi

from .mrc import MapVolume, _grid_from_array_zyx


def maps_grid_compatible(a: MapVolume, b: MapVolume) -> bool:
    """True if shapes, origin (Å), and isotropic apix match (no resampling needed)."""
    if a.data_zyx.shape != b.data_zyx.shape:
        return False
    if not np.allclose(a.origin_xyzA, b.origin_xyzA, atol=1e-2, rtol=0):
        return False
    if abs(float(a.apix) - float(b.apix)) > 1e-2:
        return False
    return True


def _float_grid_for_volume(vol: MapVolume) -> gemmi.FloatGrid:
    if vol.grid is not None:
        return vol.grid
    return _grid_from_array_zyx(vol.data_zyx, float(vol.apix))


def _target_grid_template(vol: MapVolume) -> gemmi.FloatGrid:
    """Grid whose indices match ``vol.data_zyx[iz,iy,ix]`` ↔ ``get_position(ix,iy,iz)``."""
    if vol.grid is not None:
        return vol.grid
    return _grid_from_array_zyx(vol.data_zyx, float(vol.apix))


def _volume_with_grid_from_data(template_grid: gemmi.FloatGrid, data_zyx: np.ndarray, vol: MapVolume) -> MapVolume:
    """New map: same cell/frame as ``template_grid``, array from ``data_zyx`` (z,y,x)."""
    nz, ny, nx = data_zyx.shape
    if template_grid.nu != nx or template_grid.nv != ny or template_grid.nw != nz:
        raise ValueError("data_zyx shape does not match template grid dimensions")
    uc = template_grid.unit_cell
    g = gemmi.FloatGrid(template_grid.nu, template_grid.nv, template_grid.nw)
    g.set_unit_cell(gemmi.UnitCell(uc.a, uc.b, uc.c, uc.alpha, uc.beta, uc.gamma))
    g.spacegroup = template_grid.spacegroup
    arr = np.asarray(g, dtype=np.float32)
    arr[:] = np.transpose(np.asarray(data_zyx, dtype=np.float32), (2, 1, 0))
    return MapVolume(
        data_zyx=np.asarray(data_zyx, dtype=np.float32),
        apix=float(vol.apix),
        origin_xyzA=np.asarray(vol.origin_xyzA, dtype=np.float32).copy(),
        halfmaps=None,
        grid=g,
        _ccp4=None,
    )


def resample_map_volume(source: MapVolume, target: MapVolume) -> MapVolume:
    """Trilinearly sample ``source`` at every voxel center of ``target``'s grid.

    Returns a new :class:`MapVolume` with ``target``'s shape, origin, apix, and a Gemmi
    grid suitable for :func:`cryomodel.validation.ringer_lite.sample_density_at_position`.
    """
    if maps_grid_compatible(source, target):
        return source

    src_g = _float_grid_for_volume(source)
    tgt_t = _target_grid_template(target)
    nz, ny, nx = target.data_zyx.shape
    if tgt_t.nu != nx or tgt_t.nv != ny or tgt_t.nw != nz:
        raise ValueError("target grid dimensions do not match target.data_zyx shape")

    out = np.zeros((nz, ny, nx), dtype=np.float32)
    for iz in range(nz):
        pos = np.empty((ny * nx, 3), dtype=np.float64)
        t = 0
        for iy in range(ny):
            for ix in range(nx):
                p = tgt_t.get_position(ix, iy, iz)
                pos[t, 0] = p.x
                pos[t, 1] = p.y
                pos[t, 2] = p.z
                t += 1
        out[iz] = src_g.interpolate_position_array(pos, order=1).reshape(ny, nx)

    return _volume_with_grid_from_data(tgt_t, out, target)
