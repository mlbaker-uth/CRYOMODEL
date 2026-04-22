# cryomodel/validation/local_cc.py
"""Local CC variants: CC_mask, CC_box, ZNCC (model vs map in local neighborhoods)."""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..io.mrc import MapVolume
from ..maps.synthetic_from_model import synthetic_density_zyx_from_weighted_positions


def _margin_voxels(resolution_A: float, apix: float) -> int:
    """Extra voxels around a crop so Gaussian blur (model2map) stays inside the patch."""
    sigma_vox = float(resolution_A) / (2.355 * max(float(apix), 1e-6))
    return max(2, int(math.ceil(4.0 * sigma_vox)))


def _weighted_positions(atom_positions: np.ndarray, atom_weights: Optional[np.ndarray] = None) -> List[Tuple[np.ndarray, float]]:
    n = len(atom_positions)
    if n == 0:
        return []
    if atom_weights is None:
        return [(np.asarray(atom_positions[i], dtype=np.float32), 1.0) for i in range(n)]
    w = np.asarray(atom_weights, dtype=np.float64)
    return [(np.asarray(atom_positions[i], dtype=np.float32), float(w[i])) for i in range(n)]


def _crop_bounds_from_mask(
    mask: np.ndarray,
    shape_zyx: Tuple[int, int, int],
    margin_vox: int,
) -> Tuple[int, int, int, int, int, int]:
    idx = np.argwhere(mask)
    if idx.size == 0:
        nz, ny, nx = shape_zyx
        return 0, nz, 0, ny, 0, nx
    iz0 = max(0, int(idx[:, 0].min()) - margin_vox)
    iz1 = min(shape_zyx[0], int(idx[:, 0].max()) + margin_vox + 1)
    iy0 = max(0, int(idx[:, 1].min()) - margin_vox)
    iy1 = min(shape_zyx[1], int(idx[:, 1].max()) + margin_vox + 1)
    ix0 = max(0, int(idx[:, 2].min()) - margin_vox)
    ix1 = min(shape_zyx[2], int(idx[:, 2].max()) + margin_vox + 1)
    return iz0, iz1, iy0, iy1, ix0, ix1


def _synthetic_patch_for_mask(
    atom_positions: np.ndarray,
    mask: np.ndarray,
    shape_zyx: Tuple[int, int, int],
    origin_xyz: np.ndarray,
    apix: float,
    model_resolution_A: float,
    atom_weights: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int, int, int, int]]:
    """
    Build model2map-equivalent synthetic density on a crop covering ``mask`` + blur margin.

    Returns:
        pred: 1D array aligned with ``map_data[mask].ravel()``
        idx: (K,3) zyx indices for mask voxels
        bounds: (iz0, iz1, iy0, iy1, ix0, ix1) crop in full volume
    """
    nz, ny, nx = shape_zyx
    margin = _margin_voxels(model_resolution_A, apix)
    iz0, iz1, iy0, iy1, ix0, ix1 = _crop_bounds_from_mask(mask, shape_zyx, margin)
    cz, cy, cx = iz1 - iz0, iy1 - iy0, ix1 - ix0
    origin = np.asarray(origin_xyz, dtype=np.float64).reshape(3)
    origin_crop = origin + float(apix) * np.array([ix0, iy0, iz0], dtype=np.float64)

    weighted = _weighted_positions(atom_positions, atom_weights)
    if not weighted:
        idx = np.argwhere(mask)
        return np.zeros(idx.shape[0], dtype=np.float64), idx, (iz0, iz1, iy0, iy1, ix0, ix1)

    synth = synthetic_density_zyx_from_weighted_positions(
        weighted,
        (cz, cy, cx),
        float(apix),
        origin_crop.astype(np.float32),
        float(model_resolution_A),
        normalize_max=True,
    )

    idx = np.argwhere(mask)
    pred = np.empty(idx.shape[0], dtype=np.float64)
    for k in range(idx.shape[0]):
        iz, iy, ix = int(idx[k, 0]), int(idx[k, 1]), int(idx[k, 2])
        pred[k] = float(synth[iz - iz0, iy - iy0, ix - ix0])

    return pred, idx, (iz0, iz1, iy0, iy1, ix0, ix1)


def _pearson_r(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.size < 2 or b.size != a.size:
        return 0.0
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    r = np.corrcoef(a, b)[0, 1]
    return float(r) if np.isfinite(r) else 0.0


def _zncc(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel() - np.mean(a)
    b = np.asarray(b, dtype=np.float64).ravel() - np.mean(b)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-18:
        return 0.0
    return float(np.dot(a, b) / denom)


def _spherical_mask_union(
    atom_positions: np.ndarray,
    shape_zyx: Tuple[int, int, int],
    origin: np.ndarray,
    apix: float,
    radius: float,
) -> np.ndarray:
    nz, ny, nx = shape_zyx
    mask = np.zeros((nz, ny, nx), dtype=bool)
    radius_vox = radius / apix
    for atom_pos in atom_positions:
        vox = (atom_pos - origin) / apix
        z, y, x = int(round(vox[2])), int(round(vox[1])), int(round(vox[0]))
        z_min = max(0, int(z - radius_vox))
        z_max = min(nz, int(z + radius_vox) + 1)
        y_min = max(0, int(y - radius_vox))
        y_max = min(ny, int(y + radius_vox) + 1)
        x_min = max(0, int(x - radius_vox))
        x_max = min(nx, int(x + radius_vox) + 1)
        for iz in range(z_min, z_max):
            for iy in range(y_min, y_max):
                for ix in range(x_min, x_max):
                    dz, dy, dx = iz - z, iy - y, ix - x
                    dist = math.sqrt(dz * dz + dy * dy + dx * dx) * apix
                    if dist <= radius:
                        mask[iz, iy, ix] = True
    return mask


def compute_local_cc_variants(
    atom_positions: np.ndarray,
    map_vol: MapVolume,
    half1_vol: Optional[MapVolume] = None,
    half2_vol: Optional[MapVolume] = None,
    mask_radius: float = 2.0,
    box_size: float = 4.0,
    model_resolution_A: float = 3.0,
) -> Dict[str, float]:
    """Local map–model agreement using the same trilinear + Gaussian blur as ``model2map`` (cropped)."""
    if len(atom_positions) == 0:
        return {
            "CC_mask": 0.0,
            "CC_box": 0.0,
            "ZNCC": 0.0,
            "CC_half1": 0.0,
            "CC_half2": 0.0,
            "CC_delta": 0.0,
        }

    origin = map_vol.origin_xyzA
    apix = map_vol.apix
    data = map_vol.data_zyx
    shape_zyx = data.shape
    res_a = float(model_resolution_A) if model_resolution_A > 0 else 3.0

    mask = _spherical_mask_union(atom_positions, shape_zyx, origin, apix, mask_radius)
    if not np.any(mask):
        return {
            "CC_mask": 0.0,
            "CC_box": 0.0,
            "ZNCC": 0.0,
            "CC_half1": 0.0,
            "CC_half2": 0.0,
            "CC_delta": 0.0,
        }

    pred, _, _ = _synthetic_patch_for_mask(atom_positions, mask, shape_zyx, origin, apix, res_a)
    obs_full = data[mask].astype(np.float64, copy=False)
    cc_mask = _pearson_r(obs_full, pred)
    zncc = _zncc(obs_full, pred)

    cc_box = _cc_in_box_model2map(
        atom_positions, data, shape_zyx, origin, apix, box_size, res_a
    )

    cc_half1 = 0.0
    cc_half2 = 0.0
    if half1_vol is not None and half1_vol.data_zyx.shape == shape_zyx:
        obs_h1 = half1_vol.data_zyx[mask].astype(np.float64, copy=False)
        cc_half1 = _pearson_r(obs_h1, pred)
    if half2_vol is not None and half2_vol.data_zyx.shape == shape_zyx:
        obs_h2 = half2_vol.data_zyx[mask].astype(np.float64, copy=False)
        cc_half2 = _pearson_r(obs_h2, pred)

    cc_delta = cc_mask - max(cc_half1, cc_half2) if (half1_vol is not None or half2_vol is not None) else 0.0

    return {
        "CC_mask": float(cc_mask),
        "CC_box": float(cc_box),
        "ZNCC": float(zncc),
        "CC_half1": float(cc_half1),
        "CC_half2": float(cc_half2),
        "CC_delta": float(cc_delta),
    }


def _cc_in_box_model2map(
    atom_positions: np.ndarray,
    map_data_zyx: np.ndarray,
    shape_zyx: Tuple[int, int, int],
    origin: np.ndarray,
    apix: float,
    box_size: float,
    model_resolution_A: float,
) -> float:
    centroid = np.mean(atom_positions, axis=0)
    nz, ny, nx = shape_zyx
    vox = (centroid - origin) / apix
    z, y, x = int(round(vox[2])), int(round(vox[1])), int(round(vox[0]))
    box_vox = max(1, int(box_size / apix))
    half = box_vox // 2
    z_min = max(0, z - half)
    z_max = min(nz, z + half + 1)
    y_min = max(0, y - half)
    y_max = min(ny, y + half + 1)
    x_min = max(0, x - half)
    x_max = min(nx, x + half + 1)

    box_mask = np.zeros((nz, ny, nx), dtype=bool)
    box_mask[z_min:z_max, y_min:y_max, x_min:x_max] = True

    pred, _, _ = _synthetic_patch_for_mask(
        atom_positions, box_mask, shape_zyx, origin, apix, model_resolution_A
    )
    obs = map_data_zyx[box_mask].astype(np.float64, copy=False)
    return _pearson_r(obs, pred)
