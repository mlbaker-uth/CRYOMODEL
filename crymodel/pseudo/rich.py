# crymodel/pseudo/rich.py
from __future__ import annotations
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.ndimage import maximum_filter
from scipy.ndimage import convolve

from ..core.types import MapVolume

def _ang_to_vox(pts_ang: np.ndarray, mv: MapVolume) -> np.ndarray:
    if pts_ang.size == 0:
        return np.zeros((0,3), dtype=int)
    p = (pts_ang - np.asarray(mv.origin, np.float32)) / float(mv.apix)
    p = np.round(p).astype(int)
    Z, Y, X = mv.data.shape
    p[:,0] = np.clip(p[:,0], 0, Z-1)
    p[:,1] = np.clip(p[:,1], 0, Y-1)
    p[:,2] = np.clip(p[:,2], 0, X-1)
    return p

def _vox_to_ang(pts_vox: np.ndarray, mv: MapVolume) -> np.ndarray:
    if pts_vox.size == 0:
        return np.zeros((0,3), dtype=np.float32)
    pts = pts_vox.astype(np.float32) * float(mv.apix) + np.asarray(mv.origin, np.float32)
    return pts.astype(np.float32)

def generate_pseudoatoms_rich(
    vol: MapVolume,
    apix: float,
    thresh: float,
    *,
    mask_bool: np.ndarray,
    sigma_vox: float = 1.0,
    min_sep_A: float = 2.2,
    ascent_steps: int = 8,
) -> np.ndarray:
    """
    'Rich' picker: LoG-esque blob scoring + non-max suppression + mean-shift ascent.
    Designed for *micro* blobs (waters/ions). Obeys mask_bool strictly.
    Returns centers in Å (N,3).
    """
    data = vol.data.astype(np.float32)
    assert mask_bool.shape == data.shape
    # Zero outside mask to prevent wandering and off-mask picks
    work = np.where(mask_bool, data, -np.inf).copy()

    # Gentle smoothing to tame voxel noise
    sm = gaussian_filter(np.where(np.isfinite(work), work, 0.0), sigma=sigma_vox, mode="nearest")

    # Simple LoG-ish blobness: Laplacian of Gaussian approximation via 3D kernel
    #  3D Laplacian kernel
    kern = np.zeros((3,3,3), np.float32)
    kern[1,1,1] = -6.0
    kern[0,1,1] = kern[2,1,1] = kern[1,0,1] = kern[1,2,1] = kern[1,1,0] = kern[1,1,2] = 1.0
    lap = convolve(sm, kern, mode="nearest")
    blobness = -(lap)  # negative Laplacian peaks inside bright blobs

    # Non-max suppression (voxel grid)
    # Window radius from desired minimum separation
    r_vox = max(1, int(round(min_sep_A / float(apix))))
    max_f = maximum_filter(blobness, size=(2*r_vox+1,)*3, mode="nearest")
    peaks = (blobness == max_f) & mask_bool & np.isfinite(work) & (work >= float(thresh))

    idx = np.argwhere(peaks)
    if idx.size == 0:
        return np.zeros((0,3), dtype=np.float32)

    # Rank peaks by (map * blobness)
    score = blobness[peaks] * sm[peaks]
    order = np.argsort(-score)
    idx = idx[order]

    # Gradient-ascent (mean-shift-ish): small steps uphill inside the mask
    centers = idx.astype(np.float32)
    step_vox = 0.25 * (1.0 / float(apix))  # ~0.25 Å steps
    for _ in range(int(ascent_steps)):
        # finite differences gradient
        # (safe pad)
        g = np.stack(np.gradient(sm), axis=-1)  # (z,y,x,3)
        zyx = centers.astype(int)
        zyx[:,0] = np.clip(zyx[:,0], 0, sm.shape[0]-1)
        zyx[:,1] = np.clip(zyx[:,1], 0, sm.shape[1]-1)
        zyx[:,2] = np.clip(zyx[:,2], 0, sm.shape[2]-1)
        dz = g[zyx[:,0], zyx[:,1], zyx[:,2], 0]
        dy = g[zyx[:,0], zyx[:,1], zyx[:,2], 1]
        dx = g[zyx[:,0], zyx[:,1], zyx[:,2], 2]
        move = np.stack([dz, dy, dx], axis=1)
        centers = centers + step_vox * np.sign(move)
        # clamp to mask bounds
        z = np.clip(np.round(centers[:,0]).astype(int), 0, sm.shape[0]-1)
        y = np.clip(np.round(centers[:,1]).astype(int), 0, sm.shape[1]-1)
        x = np.clip(np.round(centers[:,2]).astype(int), 0, sm.shape[2]-1)
        ok = mask_bool[z,y,x]
        centers = centers[ok]
        idx = idx[ok]
        if centers.size == 0:
            break

    # Enforce minimum separation on the refined set (simple voxel grid NMS)
    if centers.size:
        z = np.clip(np.round(centers[:,0]).astype(int), 0, sm.shape[0]-1)
        y = np.clip(np.round(centers[:,1]).astype(int), 0, sm.shape[1]-1)
        x = np.clip(np.round(centers[:,2]).astype(int), 0, sm.shape[2]-1)
        chosen = []
        taken = np.zeros(sm.shape, dtype=bool)
        rad = r_vox
        for i,(zz,yy,xx) in enumerate(zip(z,y,x)):
            if taken[zz,yy,xx]: 
                continue
            chosen.append(i)
            zz0,zz1 = max(0,zz-rad), min(sm.shape[0],zz+rad+1)
            yy0,yy1 = max(0,yy-rad), min(sm.shape[1],yy+rad+1)
            xx0,xx1 = max(0,xx-rad), min(sm.shape[2],xx+rad+1)
            taken[zz0:zz1, yy0:yy1, xx0:xx1] = True
        centers = centers[chosen]

    # Convert to Å
    pts_vox = np.round(centers).astype(int)
    pts_ang = _vox_to_ang(pts_vox, vol)
    return pts_ang.astype(np.float32)
