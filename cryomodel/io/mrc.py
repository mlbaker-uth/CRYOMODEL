# cryomodel/io/mrc.py
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
import numpy as np
import gemmi

@dataclass
class MapVolume:
    data_zyx: np.ndarray  # numpy array in (z, y, x)
    apix: float           # Å/voxel (isotropic heuristic from unit cell; see grid for exact frame)
    origin_xyzA: np.ndarray  # np.float32 shape (3,), Å — MRC words 50–52; may not match gemmi grid frame
    halfmaps: tuple[np.ndarray, np.ndarray] | None = None  # optional (z,y,x)
    #: When set (maps from :func:`read_map`), density sampling should use this grid so the frame matches gemmi/ChimeraX.
    grid: Optional[gemmi.FloatGrid] = None
    #: Keeps the owning :class:`gemmi.Ccp4Map` alive so ``grid`` stays valid.
    _ccp4: Any = field(default=None, repr=False, compare=False)

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

    # Extract origin in Å: MRC words 50–52 (gemmi header_float); FloatGrid may not expose origin in 0.7.x
    origin_xyzA = np.array(
        [float(m.header_float(50)), float(m.header_float(51)), float(m.header_float(52))],
        dtype=np.float32,
    )

    data_zyx = _array_zyx_from_grid(g)
    return MapVolume(
        data_zyx=data_zyx,
        apix=apix,
        origin_xyzA=origin_xyzA,
        halfmaps=None,
        grid=g,
        _ccp4=m,
    )

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
    out = Path(path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    grid = _grid_from_array_zyx(np.asarray(data_zyx, dtype=np.float32), mv.apix)
    ccp4 = gemmi.Ccp4Map()
    ccp4.grid = grid
    ccp4.update_ccp4_header()
    # MRC/CCP4: origin (Å) for the first voxel in x, y, z is in header words 50–52
    # (gemmi 1-based word indices; see Ccp4Base::get_origin() in gemmi). The old
    # `ccp4.header.origin` attribute is not present on gemmi 0.7.x Python bindings,
    # so origins were previously never written.
    ox, oy, oz = (float(v) for v in mv.origin_xyzA)
    ccp4.set_header_float(50, ox)
    ccp4.set_header_float(51, oy)
    ccp4.set_header_float(52, oz)
    ccp4.write_ccp4_map(str(out))  # writes .mrc just fine
