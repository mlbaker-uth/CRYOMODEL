"""Local boundary refinement around a representative subunit using one label map + density."""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
from scipy import ndimage

from cryomodel.helical.segmenter import _perp_frame, _unit, _wrap_to_pi
from cryomodel.io.mrc import read_map, write_map
from cryomodel.symmetry.preprocess import _voxel_centers_xyz


@dataclass
class HelicalLocalRefineResult:
    output_json: str
    labels_map: str
    representative_map: str
    representative_mask_map: str
    representative_label_id: int
    n_voxels_refined: int
    crop_zyx_slices: list[list[int]]

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


def _infer_repeat_centers_t_A(
    labels_zyx: np.ndarray,
    pivot: np.ndarray,
    axis: np.ndarray,
    origin: np.ndarray,
    apix: float,
    n_sub: int,
) -> np.ndarray:
    """Median axial coordinate t (Å) per label 1..n_sub; NaN if label missing."""
    centers = np.full(n_sub, np.nan, dtype=np.float64)
    axis_u = _unit(axis)
    for lab in range(1, n_sub + 1):
        iz, iy, ix = np.nonzero(labels_zyx == lab)
        if iz.size == 0:
            continue
        P = _voxel_centers_xyz(iz, iy, ix, origin, apix).astype(np.float64)
        rel = P - pivot.reshape(1, 3)
        with np.errstate(all="ignore"):
            t = rel @ axis_u
        centers[lab - 1] = float(np.median(t))
    return centers


def _active_label_ids(rep_id: int, n_sub: int, neighbor_layers: int) -> list[int]:
    lo = max(1, rep_id - neighbor_layers)
    hi = min(n_sub, rep_id + neighbor_layers)
    return list(range(lo, hi + 1))


def run_helical_refine_local(
    map_path: Path,
    labels_path: Path,
    segment_json: Path,
    out_dir: Path,
    *,
    neighbor_layers: int = 2,
    pad_voxels: int = 8,
    density_threshold: Optional[float] = None,
    representative_label: Optional[int] = None,
    representative_largest_component: bool = False,
) -> HelicalLocalRefineResult:
    """
    Re-assign voxels in a bounding box around the representative label and ±neighbor_layers
    using the same helical phase+shear cost as phase_peaks (parameters from ``helical_segment.json``).

    Does not require per-subunit mask files. End subunits naturally have fewer neighbors when
    ``rep_id`` is near 1 or n_sub.
    """
    map_path = Path(map_path).expanduser().resolve()
    labels_path = Path(labels_path).expanduser().resolve()
    segment_json = Path(segment_json).expanduser().resolve()
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(segment_json, encoding="utf-8") as fh:
        meta = json.load(fh)

    mv = read_map(map_path)
    data = mv.data_zyx.astype(np.float32)
    apix = float(mv.apix)
    origin = np.asarray(mv.origin_xyzA, dtype=np.float64)

    mv_lab = read_map(labels_path)
    if tuple(mv_lab.data_zyx.shape) != tuple(data.shape):
        raise ValueError(f"Labels shape {mv_lab.data_zyx.shape} != map shape {data.shape}")
    labels_full = np.rint(mv_lab.data_zyx).astype(np.int32)

    thr = float(density_threshold) if density_threshold is not None else float(meta.get("threshold", 0.0))
    rise = float(meta["rise_A"])
    twist_deg = float(meta["twist_deg"])
    twist_rad = math.radians(twist_deg)
    axis = _unit(np.asarray(meta["axis_xyz"], dtype=np.float64))
    pivot = np.asarray(meta["pivot_xyz"], dtype=np.float64)
    t0 = float(meta.get("t0_A", 0.0))
    phi0 = math.radians(float(meta.get("phi0_deg", 0.0)))
    sig_t = float(meta.get("sigma_t_A", max(0.5, 0.5 * abs(rise))))
    sig_phi = math.radians(float(meta.get("sigma_phi_deg", max(3.0, 0.35 * abs(twist_deg)))))

    n_sub = int(np.max(labels_full))
    if n_sub < 1:
        raise ValueError("No positive labels in label map.")

    pp = meta.get("phase_peaks") or {}
    rc = pp.get("repeat_center_t_A")
    centers_all = np.full(n_sub, np.nan, dtype=np.float64)
    if isinstance(rc, list) and len(rc) > 0:
        nc = min(len(rc), n_sub)
        centers_all[:nc] = np.asarray(rc[:nc], dtype=np.float64)
    inferred = _infer_repeat_centers_t_A(labels_full, pivot, axis, origin, apix, n_sub)
    for i in range(n_sub):
        if not np.isfinite(centers_all[i]):
            centers_all[i] = inferred[i]
    if not np.all(np.isfinite(centers_all)):
        for i in range(n_sub):
            if not np.isfinite(centers_all[i]):
                centers_all[i] = float(t0) + float(i) * float(rise)

    piecewise = bool(pp.get("shear_piecewise_dz", True))
    alpha_pos = float(pp.get("shear_alpha_pos_rad_per_A", 0.0))
    alpha_neg = float(pp.get("shear_alpha_neg_rad_per_A", 0.0))
    alpha_single = pp.get("shear_alpha_rad_per_A")
    if not piecewise and alpha_single is not None:
        alpha_pos = alpha_neg = float(alpha_single)

    if representative_label is not None:
        rep_id = int(representative_label)
    else:
        rep_id = int(meta.get("representative_label_id", 0))
        if rep_id < 1 or rep_id > n_sub:
            lbls = labels_full[labels_full > 0].ravel()
            if lbls.size == 0:
                raise ValueError("No positive labels.")
            mx = int(np.max(lbls))
            cnt = np.bincount(lbls.astype(np.int64), minlength=mx + 1)
            rep_id = int(np.argmax(cnt[1:]) + 1)
    rep_id = int(np.clip(rep_id, 1, n_sub))

    active = _active_label_ids(rep_id, n_sub, neighbor_layers)
    if not active:
        raise ValueError("No active labels (check neighbor_layers and n_sub).")

    mask_sub = np.isin(labels_full, np.asarray(active, dtype=np.int32))
    if not np.any(mask_sub):
        raise ValueError(f"No voxels with labels in active range {active}.")

    struct = np.ones((3, 3, 3), dtype=bool)
    mask_bb = ndimage.binary_dilation(mask_sub, structure=struct, iterations=1)
    coords = np.argwhere(mask_bb)
    z0, y0, x0 = (int(coords[:, 0].min()), int(coords[:, 1].min()), int(coords[:, 2].min()))
    z1, y1, x1 = (int(coords[:, 0].max()) + 1, int(coords[:, 1].max()) + 1, int(coords[:, 2].max()) + 1)
    pad = max(0, int(pad_voxels))
    Z, Y, X = data.shape
    z0 = max(0, z0 - pad)
    y0 = max(0, y0 - pad)
    x0 = max(0, x0 - pad)
    z1 = min(Z, z1 + pad)
    y1 = min(Y, y1 + pad)
    x1 = min(X, x1 + pad)

    slz = slice(z0, z1)
    sly = slice(y0, y1)
    slx = slice(x0, x1)
    sub_data = data[slz, sly, slx]
    sub_lab = labels_full[slz, sly, slx].copy()

    iz, iy, ix = np.nonzero(sub_data > thr)
    if iz.size == 0:
        raise ValueError("No voxels above threshold inside crop; adjust threshold.")

    P = _voxel_centers_xyz(iz + z0, iy + y0, ix + x0, origin, apix).astype(np.float64)
    rel = P - pivot.reshape(1, 3)
    a, b = _perp_frame(axis)
    with np.errstate(all="ignore"):
        t = rel @ axis
        xp = rel @ a
        yp = rel @ b
    phi = np.arctan2(yp, xp)

    center_t = np.asarray([centers_all[i - 1] for i in active], dtype=np.float64)
    k_cent = np.rint((center_t - t0) / rise).astype(np.float64)
    theta_cent = phi0 + k_cent * twist_rad

    dtm = t.reshape(-1, 1) - center_t.reshape(1, -1)
    dph0 = _wrap_to_pi(phi.reshape(-1, 1) - theta_cent.reshape(1, -1))

    if piecewise:
        alpha_eff = np.where(dtm >= 0.0, alpha_pos, alpha_neg)
    else:
        alpha_eff = alpha_pos
    with np.errstate(all="ignore"):
        c1 = (dtm / max(sig_t, 1e-6)) ** 2 + (
            (_wrap_to_pi(dph0 - alpha_eff * dtm)) / max(sig_phi, 1e-6)
        ) ** 2
    j_best = np.argmin(c1, axis=1).astype(np.int32)
    new_ids = np.asarray(active, dtype=np.int32)[j_best]

    old_flat = sub_lab[iz, iy, ix]
    refine_mask = np.isin(old_flat, np.asarray(active, dtype=np.int32)) | (old_flat == 0)
    n_ref = int(np.count_nonzero(refine_mask))
    sub_new = sub_lab.copy()
    sub_new[iz[refine_mask], iy[refine_mask], ix[refine_mask]] = new_ids[refine_mask]

    labels_out = labels_full.copy()
    labels_out[slz, sly, slx] = sub_new

    labels_path_out = out_dir / "helical_subunit_labels_refined.mrc"
    write_map(labels_path_out, mv, labels_out.astype(np.float32))

    rep_vol = np.zeros_like(data, dtype=np.float32)
    rep_m = labels_out == rep_id
    if representative_largest_component:
        cc, ncc = ndimage.label(rep_m, structure=struct)
        if ncc > 1:
            sizes = np.bincount(cc.ravel())
            sizes[0] = 0
            keep = int(np.argmax(sizes))
            rep_m = cc == keep
    rep_vol[rep_m] = data[rep_m].astype(np.float32)
    rep_path = out_dir / "helical_subunit_representative_refined.mrc"
    write_map(rep_path, mv, rep_vol)

    mask_vol = np.zeros_like(data, dtype=np.float32)
    mask_vol[rep_m] = 1.0
    mask_path = out_dir / "helical_representative_mask.mrc"
    write_map(mask_path, mv, mask_vol)

    out_json = out_dir / "helical_refine_local.json"
    crop_spec = [[z0, z1], [y0, y1], [x0, x1]]
    result = HelicalLocalRefineResult(
        output_json=str(out_json),
        labels_map=str(labels_path_out),
        representative_map=str(rep_path),
        representative_mask_map=str(mask_path),
        representative_label_id=rep_id,
        n_voxels_refined=n_ref,
        crop_zyx_slices=crop_spec,
    )
    payload = result.to_json_dict()
    payload["input_map"] = str(map_path)
    payload["input_labels"] = str(labels_path)
    payload["segment_json"] = str(segment_json)
    payload["neighbor_layers"] = int(neighbor_layers)
    payload["pad_voxels"] = int(pad_voxels)
    payload["active_label_ids"] = active
    payload["threshold"] = float(thr)
    payload["repeat_centers_source"] = (
        "segment_json+fill" if isinstance(rc, list) and len(rc) > 0 else "inferred"
    )
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return result
