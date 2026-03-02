import numpy as np
from scipy.ndimage import binary_dilation, generate_binary_structure
from ..maps.ops import vox_to_ang

def generate_pseudoatoms_greedy(unmodeled_map,
                                apix: float,
                                thresh: float,
                                zero_radius_A: float = 0.5,
                                max_points: int = 50000,
                                mask_bool: np.ndarray | None = None):   # NEW
    data = unmodeled_map.data.copy().astype(np.float32)
    base_mask = (data >= float(thresh)) if mask_bool is None else mask_bool.astype(bool)
    if not base_mask.any():
        return np.zeros((0,3), np.float32)

    # ensure we never pick outside the mask
    data[~base_mask] = -np.inf

    r_vox = max(1.0, float(zero_radius_A) / float(apix))
    rad = int(np.ceil(r_vox))
    zyx = np.arange(-rad, rad+1)
    Z,Y,X = np.meshgrid(zyx, zyx, zyx, indexing='ij')
    se = ((Z*Z + Y*Y + X*X) <= (r_vox*r_vox))

    kept = []
    taken = np.zeros_like(base_mask, dtype=bool)
    count = 0
    while True:
        rem = base_mask & (~taken)
        if not rem.any():
            break
        idx = np.argmax(np.where(rem, data, -np.inf))
        if not np.isfinite(data.ravel()[idx]):
            break
        z = idx // (data.shape[1]*data.shape[2])
        y = (idx // data.shape[2]) % data.shape[1]
        x = idx % data.shape[2]
        kept.append([z, y, x])
        count += 1
        if count >= max_points:
            break

        slab = (slice(max(0,z-rad), min(data.shape[0], z+rad+1)),
                slice(max(0,y-rad), min(data.shape[1], y+rad+1)),
                slice(max(0,x-rad), min(data.shape[2], x+rad+1)))
        # apply ball
        taken[slab] |= se[
            slice(rad-(z-slab[0].start), rad+(slab[0].stop-1-z)+1),
            slice(rad-(y-slab[1].start), rad+(slab[1].stop-1-y)+1),
            slice(rad-(x-slab[2].start), rad+(slab[2].stop-1-x)+1),
        ]

    if not kept:
        return np.zeros((0,3), np.float32)
    peaks_zyx = np.asarray(kept, dtype=np.float32)
    return vox_to_ang(peaks_zyx, unmodeled_map).astype(np.float32)

