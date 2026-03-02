# crymodel/finders/pipeline.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi

from ..io.mrc import MapVolume, read_map_with_halves, write_map
from ..io.pdb import read_model_xyz   # returns (N,3) xyz Å numpy
from ..io.site_export import AssignmentSet, write_json, write_water_pdb

# ---------- helpers: coordinate transforms ----------
def zyx_to_xyzA(zyx_idx: np.ndarray, apix: float, origin_xyzA: np.ndarray) -> np.ndarray:
    """(…,3) z,y,x integer indices -> (…,3) x,y,z in Å"""
    zyx = np.asarray(zyx_idx, dtype=np.float32)
    x = zyx[..., 2] * apix + origin_xyzA[0]
    y = zyx[..., 1] * apix + origin_xyzA[1]
    z = zyx[..., 0] * apix + origin_xyzA[2]
    return np.stack([x, y, z], axis=-1)

def xyzA_to_zyx(xyzA: np.ndarray, apix: float, origin_xyzA: np.ndarray) -> np.ndarray:
    """(…,3) x,y,z in Å -> (…,3) z,y,x integer indices"""
    xyz = np.asarray(xyzA, dtype=np.float32)
    ix = np.rint((xyz[...,0] - origin_xyzA[0]) / apix).astype(np.int64)
    iy = np.rint((xyz[...,1] - origin_xyzA[1]) / apix).astype(np.int64)
    iz = np.rint((xyz[...,2] - origin_xyzA[2]) / apix).astype(np.int64)
    return np.stack([iz, iy, ix], axis=-1)

# ---------- Phase 1: mask + threshold ----------
def mask_and_threshold(vol: MapVolume,
                       model_xyzA: np.ndarray,
                       thresh: float,
                       mask_radius_A: float) -> tuple[np.ndarray, np.ndarray]:
    data = vol.data_zyx.copy()
    apix = vol.apix
    # distance mask: zero any voxel within mask_radius of any model atom
    # build a binary mask via distance transform from a rasterized atom map
    z, y, x = data.shape
    atom_zyx = xyzA_to_zyx(model_xyzA, apix, vol.origin_xyzA)
    # keep points inside volume
    m = ((atom_zyx[:,0] >= 0) & (atom_zyx[:,0] < z) &
         (atom_zyx[:,1] >= 0) & (atom_zyx[:,1] < y) &
         (atom_zyx[:,2] >= 0) & (atom_zyx[:,2] < x))
    pts = atom_zyx[m]
    seeds = np.zeros((z,y,x), np.uint8)
    seeds[pts[:,0], pts[:,1], pts[:,2]] = 1
    dist = ndi.distance_transform_edt(1 - seeds) * apix
    masked = data.copy()
    masked[dist <= mask_radius_A] = 0.0

    masked_thr = masked.copy()
    masked_thr[masked_thr < float(thresh)] = 0.0
    return masked, masked_thr

# ---------- Phase 2: split into water/ligand maps by component size ----------
def split_micro_vs_ligand(masked_thr_zyx: np.ndarray,
                          micro_vvox_min: int,
                          micro_vvox_max: int) -> tuple[np.ndarray, np.ndarray]:
    # label connected components in 3D
    lab, nlab = ndi.label(masked_thr_zyx > 0.0, structure=np.ones((3,3,3), np.uint8))
    counts = np.bincount(lab.ravel())
    # 0-label is background
    keep_micro = np.isin(lab, np.where((counts >= micro_vvox_min) & (counts <= micro_vvox_max))[0])
    keep_lig   = np.isin(lab, np.where(counts >  micro_vvox_max)[0])

    waters_map = np.where(keep_micro, masked_thr_zyx, 0.0).astype(np.float32)
    lig_map    = np.where(keep_lig,   masked_thr_zyx, 0.0).astype(np.float32)
    return waters_map, lig_map

# ---------- Phase 3: greedy water picking in voxel space ----------
def greedy_peaks_from_map(waters_zyx: np.ndarray,
                          apix: float,
                          origin_xyzA: np.ndarray,
                          zero_radius_A: float = 2.0) -> np.ndarray:
    """Return centers in Å (x,y,z) from a greedy max-pooling on water map."""
    vol = waters_zyx.copy()
    r_vox = max(1, int(np.ceil(zero_radius_A / apix)))
    centers_xyz = []

    while True:
        idx = np.unravel_index(np.argmax(vol), vol.shape)
        vmax = float(vol[idx])
        if vmax <= 0.0:
            break
        zc, yc, xc = idx
        centers_xyz.append(zyx_to_xyzA(np.array([[zc, yc, xc]], np.int64), apix, origin_xyzA)[0])
        # zero-out a ball of radius r_vox around (zc, yc, xc)
        z, y, x = np.ogrid[:vol.shape[0], :vol.shape[1], :vol.shape[2]]
        mask = (z - zc)**2 + (y - yc)**2 + (x - xc)**2 <= r_vox**2
        vol[mask] = 0.0

    if len(centers_xyz) == 0:
        return np.zeros((0,3), np.float32)
    return np.asarray(centers_xyz, dtype=np.float32)

# ---------- Public: Phase 1–3 runner and writer ----------
def run_phase_1_to_3(map_path: str,
                     model_path: str,
                     thresh: float,
                     mask_radius_A: float,
                     micro_vvox_min: int,
                     micro_vvox_max: int,
                     zero_radius_A: float,
                     half1_path: Optional[str] = None,
                     half2_path: Optional[str] = None,
                     remove_hydrogens: bool = True):
    vol = read_map_with_halves(map_path, half1_path=half1_path, half2_path=half2_path)
    model_xyzA = read_model_xyz(model_path, remove_hydrogens=remove_hydrogens)

    masked, masked_thr = mask_and_threshold(vol, model_xyzA, thresh, mask_radius_A)
    waters_map, lig_map = split_micro_vs_ligand(masked_thr, micro_vvox_min, micro_vvox_max)
    water_centers_xyzA = greedy_peaks_from_map(waters_map, vol.apix, vol.origin_xyzA, zero_radius_A)

    assigns = AssignmentSet(centers_xyzA=water_centers_xyzA,
                            meta=dict(
                                apix=vol.apix,
                                origin_xyzA=vol.origin_xyzA.tolist(),
                                thresh=float(thresh),
                                mask_radius_A=float(mask_radius_A),
                                micro_vvox_min=int(micro_vvox_min),
                                micro_vvox_max=int(micro_vvox_max),
                                zero_radius_A=float(zero_radius_A),
                                n_waters=int(water_centers_xyzA.shape[0]),
                                shape_zyx=list(masked.shape),
                            ))
    return vol, masked, masked_thr, waters_map, lig_map, assigns

def write_phase_outputs(vol: MapVolume,
                        masked: np.ndarray,
                        masked_thr: np.ndarray,
                        waters_map: np.ndarray,
                        lig_map: np.ndarray,
                        assigns: AssignmentSet,
                        out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    write_map(out_dir / "masked.mrc", vol, masked)
    write_map(out_dir / "masked_thr.mrc", vol, masked_thr)
    write_map(out_dir / "waters.mrc", vol, waters_map)
    write_map(out_dir / "ligands.mrc", vol, lig_map)
    write_water_pdb(assigns, out_dir / "water")
    write_json(assigns, out_dir / "assigns.json")
