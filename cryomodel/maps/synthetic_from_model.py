"""Build synthetic density on a fixed (z,y,x) grid — same physics as model2map (trilinear + Gaussian)."""
from __future__ import annotations

import math
from typing import Optional

import gemmi
import numpy as np
from scipy.ndimage import gaussian_filter


def _trilinear_add(grid_zyx: np.ndarray, xyz_vox: np.ndarray, value: float) -> None:
    x, y, z = float(xyz_vox[0]), float(xyz_vox[1]), float(xyz_vox[2])
    x0 = int(math.floor(x))
    y0 = int(math.floor(y))
    z0 = int(math.floor(z))
    dx = x - x0
    dy = y - y0
    dz = z - z0
    nz, ny, nx = grid_zyx.shape
    for oz in (0, 1):
        for oy in (0, 1):
            for ox in (0, 1):
                ix = x0 + ox
                iy = y0 + oy
                iz = z0 + oz
                if not (0 <= ix < nx and 0 <= iy < ny and 0 <= iz < nz):
                    continue
                wx = (1.0 - dx) if ox == 0 else dx
                wy = (1.0 - dy) if oy == 0 else dy
                wz = (1.0 - dz) if oz == 0 else dz
                grid_zyx[iz, iy, ix] += float(value) * wx * wy * wz


def _plddt_from_atom(atom: gemmi.Atom) -> float:
    b_iso = float(getattr(atom, "b_iso", 0.0) or 0.0)
    if b_iso > 0:
        return min(1.0, max(0.0, b_iso / 100.0))
    return 1.0


def _is_hydrogen(atom: gemmi.Atom) -> bool:
    element_name = atom.element.name if atom.element else ""
    if element_name:
        return element_name.upper() == "H"
    name = (atom.name or "").strip()
    return len(name) > 0 and name[0].upper() == "H"


def collect_atoms_weighted(
    structure: gemmi.Structure,
    *,
    skip_hydrogen: bool = True,
    plddt_min: Optional[float] = None,
    weight_by_plddt: bool = False,
    scale_occupancy: bool = False,
    scale_bfactor: bool = False,
    resolution_A: float = 3.0,
) -> list[tuple[np.ndarray, float]]:
    """Return (position_xyz, weight) for each atom. plddt_min None = no pLDDT filtering."""
    atoms: list[tuple[np.ndarray, float]] = []
    bfactor_scale_const = 1.0 / (4.0 * float(resolution_A) * float(resolution_A))
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    if skip_hydrogen and _is_hydrogen(atom):
                        continue
                    plddt = _plddt_from_atom(atom)
                    if plddt_min is not None and plddt < float(plddt_min):
                        continue
                    pos = np.array([float(atom.pos.x), float(atom.pos.y), float(atom.pos.z)], dtype=np.float32)
                    weight = 1.0
                    if weight_by_plddt:
                        weight *= plddt
                    if scale_occupancy:
                        occ = float(getattr(atom, "occ", 1.0) or 1.0)
                        weight *= occ if occ > 0 else 1.0
                    if scale_bfactor:
                        b_iso = float(getattr(atom, "b_iso", 0.0) or 0.0)
                        weight *= math.exp(-max(0.0, b_iso) * bfactor_scale_const)
                    atoms.append((pos, float(weight)))
    return atoms


def synthetic_density_zyx_on_grid(
    structure: gemmi.Structure,
    shape_zyx: tuple[int, int, int],
    apix: float,
    origin_corner_xyz: np.ndarray,
    resolution_A: float,
    *,
    skip_hydrogen: bool = True,
    plddt_min: Optional[float] = None,
    weight_by_plddt: bool = False,
    scale_occupancy: bool = False,
    scale_bfactor: bool = False,
    normalize_max: bool = True,
) -> np.ndarray:
    """
    Fill a (nz, ny, nx) volume with the same pipeline as ``cryomodel model2map``:
    trilinear impulse deposit + isotropic Gaussian blur to resolution_A (FWHM).

    ``origin_corner_xyz`` is the physical (x,y,z) position of voxel index (0,0,0)
    (corner convention, matches MapVolume / MRC words 50–52).
    """
    nz, ny, nx = int(shape_zyx[0]), int(shape_zyx[1]), int(shape_zyx[2])
    if nz < 1 or ny < 1 or nx < 1 or apix <= 0 or resolution_A <= 0:
        raise ValueError("Invalid shape, apix, or resolution")
    origin = np.asarray(origin_corner_xyz, dtype=np.float32).reshape(3)

    weighted = collect_atoms_weighted(
        structure,
        skip_hydrogen=skip_hydrogen,
        plddt_min=plddt_min,
        weight_by_plddt=weight_by_plddt,
        scale_occupancy=scale_occupancy,
        scale_bfactor=scale_bfactor,
        resolution_A=resolution_A,
    )
    grid_zyx = np.zeros((nz, ny, nx), dtype=np.float32)
    for pos, w in weighted:
        if w <= 0:
            continue
        vox = (pos - origin) / float(apix)
        _trilinear_add(grid_zyx, vox, w)

    sigma_vox = float(resolution_A) / (2.355 * float(apix))
    if sigma_vox > 0:
        grid_zyx = gaussian_filter(grid_zyx, sigma=sigma_vox, mode="constant", cval=0.0).astype(np.float32)

    if normalize_max:
        vmax = float(np.max(grid_zyx))
        if vmax > 0:
            grid_zyx /= vmax
    return grid_zyx


def synthetic_density_zyx_from_weighted_positions(
    weighted_positions: list[tuple[np.ndarray, float]],
    shape_zyx: tuple[int, int, int],
    apix: float,
    origin_corner_xyz: np.ndarray,
    resolution_A: float,
    *,
    normalize_max: bool = True,
) -> np.ndarray:
    """
    Same pipeline as :func:`synthetic_density_zyx_on_grid` (trilinear deposit + Gaussian blur
    to ``resolution_A``), but from explicit (xyz, weight) pairs (e.g. one residue's atoms).
    """
    nz, ny, nx = int(shape_zyx[0]), int(shape_zyx[1]), int(shape_zyx[2])
    if nz < 1 or ny < 1 or nx < 1 or apix <= 0 or resolution_A <= 0:
        raise ValueError("Invalid shape, apix, or resolution")
    origin = np.asarray(origin_corner_xyz, dtype=np.float32).reshape(3)

    grid_zyx = np.zeros((nz, ny, nx), dtype=np.float32)
    for pos, w in weighted_positions:
        w = float(w)
        if w <= 0:
            continue
        p = np.asarray(pos, dtype=np.float32).reshape(3)
        vox = (p - origin) / float(apix)
        _trilinear_add(grid_zyx, vox, w)

    sigma_vox = float(resolution_A) / (2.355 * float(apix))
    if sigma_vox > 0:
        grid_zyx = gaussian_filter(grid_zyx, sigma=sigma_vox, mode="constant", cval=0.0).astype(np.float32)

    if normalize_max:
        vmax = float(np.max(grid_zyx))
        if vmax > 0:
            grid_zyx /= vmax
    return grid_zyx
