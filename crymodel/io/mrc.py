# crymodel/io/mrc.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import gemmi

@dataclass
class MapVolume:
    data_zyx: np.ndarray  # numpy array in (z, y, x)
    apix: float           # Å/voxel (isotropic)
    origin_xyzA: np.ndarray  # np.float32 shape (3,), Å
    halfmaps: tuple[np.ndarray, np.ndarray] | None = None  # optional (z,y,x)

def _grid_from_array_zyx(arr_zyx: np.ndarray, apix: float) -> gemmi.FloatGrid:
    z, y, x = arr_zyx.shape
    grid = gemmi.FloatGrid(x, y, z)  # gemmi is (x, y, z)
    # Put the numpy data into gemmi grid with the correct transpose
    # arr_zyx -> arr_xyz for gemmi buffer
    grid_array = np.asarray(grid, dtype=np.float32)
    grid_array[...] = np.transpose(arr_zyx, (2,1,0))
    grid.unit_cell = gemmi.UnitCell(apix*x, apix*y, apix*z, 90.0, 90.0, 90.0)
    grid.spacegroup = gemmi.SpaceGroup(1)  # P1
    return grid

def _array_zyx_from_grid(grid: gemmi.FloatGrid) -> np.ndarray:
    # gemmi array is (x, y, z); convert back to (z, y, x)
    a = np.asarray(grid, dtype=np.float32)  # (x, y, z)
    return np.transpose(a, (2,1,0)).copy()

def read_map(path: str | Path) -> MapVolume:
    """Read CCP4/MRC map and return MapVolume with data in (z,y,x), apix, origin(x,y,z).
    Uses Gemmi's high-level reader and handles origin if present.
    """
    m = gemmi.read_ccp4_map(str(path))
    # Ensure grid is set up (older gemmi requires setup(), newer allows setup(default))
    try:
        m.setup(0.0)
    except TypeError:
        m.setup()
    g = m.grid

    # Å/voxel for isotropic grids
    apix = float(g.unit_cell.a / g.nu)

    # Extract origin in Å; prefer grid.origin if available, else header.origin, else zeros
    origin_xyzA: np.ndarray
    try:
        origin_xyzA = np.array([float(g.origin.x), float(g.origin.y), float(g.origin.z)], dtype=np.float32)
    except AttributeError:
        try:
            origin_xyzA = np.array([float(m.header.origin.x), float(m.header.origin.y), float(m.header.origin.z)], dtype=np.float32)
        except Exception:
            origin_xyzA = np.zeros(3, np.float32)

    data_zyx = _array_zyx_from_grid(g)
    return MapVolume(data_zyx=data_zyx, apix=apix, origin_xyzA=origin_xyzA, halfmaps=None)

def read_map_with_halves(map_path: str | Path,
                         half1_path: str | None = None,
                         half2_path: str | None = None) -> MapVolume:
    mv = read_map(map_path)
    if half1_path and half2_path:
        h1 = read_map(half1_path).data_zyx
        h2 = read_map(half2_path).data_zyx
        mv.halfmaps = (h1, h2)
    return mv

def write_map(path: str | Path, mv: MapVolume, data_zyx: np.ndarray) -> None:
    grid = _grid_from_array_zyx(np.asarray(data_zyx, dtype=np.float32), mv.apix)
    ccp4 = gemmi.Ccp4Map()
    ccp4.grid = grid
    # store origin if available
    try:
        ccp4.header.origin = gemmi.Vec3(*[float(v) for v in mv.origin_xyzA])
    except Exception:
        pass
    ccp4.update_ccp4_header()
    ccp4.write_ccp4_map(str(path))  # writes .mrc just fine
