import numpy as np
from ..maps.ops import vox_to_ang

def generate_pseudoatoms_uniform(unmodeled_map,
                                 apix: float,
                                 thresh: float,
                                 spacing_A: float = 0.45,
                                 stride_vox: int = 1,
                                 bias_power: float = 1.0,
                                 max_points: int = 100000,
                                 mask_bool: np.ndarray | None = None):   # NEW
    data = unmodeled_map.data.astype(np.float32)

    # use provided mask exactly; fall back to local threshold if None
    if mask_bool is None:
        mask = data >= float(thresh)
    else:
        mask = mask_bool.astype(bool)

    if not mask.any():
        return np.zeros((0,3), np.float32)

    zyx = np.argwhere(mask)
    if stride_vox > 1:
        zyx = zyx[::stride_vox]
    if len(zyx) == 0:
        return np.zeros((0,3), np.float32)

    # weights from *original* density but only at mask voxels
    vals = data[mask].astype(np.float64)
    if stride_vox > 1:
        m_flat = np.flatnonzero(mask)
        vals = data.ravel()[m_flat][::stride_vox].astype(np.float64)

    w = np.maximum(vals, 0.0) ** float(bias_power)
    w_sum = w.sum()
    w = (np.ones_like(w)/len(w)) if w_sum <= 0 else (w / w_sum)

    r_vox = float(spacing_A) / float(apix)
    r2 = r_vox * r_vox
    kept = []
    cell = max(1, int(np.floor(r_vox)))
    buckets = {}
    def bkey(p): return tuple((p // cell).astype(int))

    order = np.random.choice(len(zyx), size=len(zyx), replace=False, p=w)
    for idx in order:
        p = zyx[idx].astype(np.float32)
        key = bkey(p)
        ok = True
        for dz in (-1,0,1):
            for dy in (-1,0,1):
                for dx in (-1,0,1):
                    for j in buckets.get((key[0]+dz, key[1]+dy, key[2]+dx), []):
                        q = kept[j]
                        if ((p-q)**2).sum() < r2:
                            ok = False; break
                    if not ok: break
                if not ok: break
            if not ok: break
        if ok:
            buckets.setdefault(key, []).append(len(kept))
            kept.append(p)
            if len(kept) >= max_points:
                break

    if not kept:
        return np.zeros((0,3), np.float32)
    peaks_zyx = np.vstack(kept).astype(np.float32)
    return vox_to_ang(peaks_zyx, unmodeled_map).astype(np.float32)

