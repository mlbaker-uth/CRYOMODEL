# crymodel/finders/microblobs.py
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from scipy.ndimage import label, gaussian_laplace
from sklearn.neighbors import KDTree

from ..core.types import MapVolume, ModelAtoms, Assignment
from ..maps.ops import vox_to_ang

@dataclass
class MicroPassResult:
    remaining_mask: np.ndarray         # bool mask (same shape as map)
    centers_ang: np.ndarray            # (N,3) Å
    assignments: list[Assignment]      # locked "water" (or "ion-like") assignments

def _sphericity_proxy(coords_zyx: np.ndarray) -> float:
    """Cheap compactness proxy from covariance eigenvalues (1=perfect sphere)."""
    if len(coords_zyx) < 3:
        return 1.0
    c = coords_zyx.astype(np.float32)
    c -= c.mean(0, keepdims=True)
    cov = (c.T @ c) / max(1, len(c) - 1)
    w = np.linalg.eigvalsh(cov + 1e-6*np.eye(3, dtype=np.float32))
    w = np.clip(w, 1e-6, None)
    ratio = (w.min() / w.max())
    return float(np.sqrt(ratio))

def micro_water_pass(
    unmodeled: MapVolume,
    thr_mask: np.ndarray,             # bool mask where map >= thresh
    apix: float,
    model: ModelAtoms,
    vvox_max: int = 8,                # max voxels for micro-blob (≈ ≤2–3 Å^3 at ~0.7 Å/vox)
    phi_min: float = 0.6,             # sphericity proxy threshold
    nearest_ONS_max: float = 3.2,     # Å (water coordination)
) -> MicroPassResult:
    """Detect tiny, round components and collapse them to 1 pseudoatom each."""
    assert thr_mask.dtype == bool
    data = unmodeled.data
    Z, Y, X = data.shape

    # protein O/N/S kd-tree
    ons_mask = np.isin(model.element.astype(str), ["O", "N", "S"])
    ons_xyz = model.xyz[ons_mask].astype(np.float32)
    tree = KDTree(ons_xyz) if len(ons_xyz) else None

    labeled, ncomp = label(thr_mask.astype(np.uint8))
    keep_mask = thr_mask.copy()
    centers = []
    assigns: list[Assignment] = []

    if ncomp == 0:
        return MicroPassResult(remaining_mask=keep_mask, centers_ang=np.zeros((0,3), np.float32), assignments=[])

    # precompute small LoG to pick a single peak per micro-blob (stable)
    log_map = -gaussian_laplace(data, sigma=0.8)  # maxima of -LoG ≈ blob centers

    for cid in range(1, ncomp + 1):
        comp = (labeled == cid)
        vvox = int(comp.sum())
        if vvox == 0:
            continue

        # very small components only
        if vvox > int(vvox_max):
            continue

        # sphericity proxy
        zyx_pts = np.argwhere(comp)  # (n,3)
        phi = _sphericity_proxy(zyx_pts)
        if phi < float(phi_min):
            continue

        # nearest O/N/S distance
        # place a tentative peak at LoG maximum within the component
        idx_max = np.argmax(log_map[comp])
        flat = np.flatnonzero(comp)[idx_max]
        z = flat // (Y * X)
        y = (flat // X) % Y
        x = flat % X
        peak_vox = np.array([[z, y, x]], dtype=np.float32)

        nearest = 9.99
        n28 = 0
        if tree is not None and len(ons_xyz):
            d, _ = tree.query(peak_vox.astype(np.float32), k=min(8, len(ons_xyz)))
            d = d.reshape(-1)
            nearest = float(d.min()) if len(d) else 9.99
            n28 = int((d <= 2.8).sum())

        if nearest > float(nearest_ONS_max):
            # no reasonable H-bond/coordination → let Stage 2 handle it
            continue

        # accept as water micro-blob: add 1 center at the LoG peak
        keep_mask[comp] = False  # remove from remaining mask
        centers.append(peak_vox[0])
        assigns.append(
            Assignment(
                index=-1,                  # will be filled after merge
                cluster_id=-1,
                probs={"water": 0.95, "unknown": 0.05},
                top="water",
                explain={
                    "micro_blob": True,
                    "Vvox": vvox,
                    "phi": float(phi),
                    "nearest_ONS": float(nearest),
                    "n_ONS_2p8": int(n28),
                },
            )
        )

    if len(centers) == 0:
        return MicroPassResult(remaining_mask=keep_mask, centers_ang=np.zeros((0,3), np.float32), assignments=[])

    centers_zyx = np.vstack(centers).astype(np.float32)
    centers_ang = vox_to_ang(centers_zyx, unmodeled).astype(np.float32)
    return MicroPassResult(remaining_mask=keep_mask, centers_ang=centers_ang, assignments=assigns)
