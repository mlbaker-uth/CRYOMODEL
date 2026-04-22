"""Helical subunit segmentation using rise/twist/axis parameters."""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
from scipy import ndimage
from scipy.ndimage import map_coordinates
from scipy.signal import find_peaks
from skimage.segmentation import watershed

from cryomodel.io.mrc import MapVolume, read_map, write_map
from cryomodel.symmetry.preprocess import _voxel_centers_xyz


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        raise ValueError("Zero-length vector.")
    return v / n


def _perp_frame(u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    u = _unit(u)
    tmp = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(u, tmp))) > 0.9:
        tmp = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    a = np.cross(u, tmp)
    a = a / np.linalg.norm(a)
    b = np.cross(u, a)
    return a, b


def _wrap_to_pi(x: np.ndarray) -> np.ndarray:
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def _clip_shear_alpha(alpha: float, rise_A: float) -> float:
    lim = 2.0 / max(abs(rise_A), 1e-6)
    return float(np.clip(alpha, -lim, lim))


def _prune_largest_cc_per_label_sparse(
    labels_1d: np.ndarray,
    iz: np.ndarray,
    iy: np.ndarray,
    ix: np.ndarray,
    shape: tuple[int, int, int],
) -> np.ndarray:
    """For each positive label id, keep only the largest 26-connected component."""
    labels_1d = np.asarray(labels_1d, dtype=np.int32).reshape(-1)
    vol = np.zeros(shape, dtype=np.int32)
    vol[iz, iy, ix] = labels_1d
    struct = np.ones((3, 3, 3), dtype=bool)
    max_lab = int(np.max(vol))
    if max_lab <= 0:
        return labels_1d
    for lab in range(1, max_lab + 1):
        m = vol == lab
        if not np.any(m):
            continue
        cc, ncc = ndimage.label(m, structure=struct)
        if ncc <= 1:
            continue
        sizes = np.bincount(cc.ravel())
        sizes[0] = 0
        keep_id = int(np.argmax(sizes))
        drop = m & (cc != keep_id)
        vol[drop] = 0
    return vol[iz, iy, ix].astype(np.int32)


def _fit_shear_slope(dz: np.ndarray, dtheta: np.ndarray, rise_A: float) -> float:
    dz = np.asarray(dz, dtype=np.float64).reshape(-1)
    dtheta = np.asarray(dtheta, dtype=np.float64).reshape(-1)
    if dz.size < 8:
        return 0.0
    den = float(np.dot(dz, dz))
    if den < 1e-12:
        return 0.0
    return _clip_shear_alpha(float(np.dot(dz, dtheta) / den), rise_A)


def _relabel_sequential_along_axis(
    labels_1d: np.ndarray,
    iz: np.ndarray,
    iy: np.ndarray,
    ix: np.ndarray,
    shape: tuple[int, int, int],
    origin: np.ndarray,
    apix: float,
    pivot: np.ndarray,
    axis_u: np.ndarray,
) -> tuple[np.ndarray, dict[int, int]]:
    """
    Remap positive label IDs to 1..K where K is the number of distinct labels and
    order follows increasing median axial coordinate t (tie-break: smaller old id).

    Ensures one distinct integer per physical subunit along the helix for mask/label maps.
    """
    labels_full = np.zeros(shape, dtype=np.int32)
    labels_full[iz, iy, ix] = labels_1d.astype(np.int32)
    t_med = _median_axial_t_per_label(labels_full, origin, apix, pivot, axis_u)
    if not t_med:
        return labels_1d.astype(np.int32), {}
    old_ordered = sorted(t_med.keys(), key=lambda L: (t_med[L], int(L)))
    old_to_new = {int(o): i + 1 for i, o in enumerate(old_ordered)}
    out = np.zeros_like(labels_1d, dtype=np.int32)
    for i in range(labels_1d.size):
        lo = int(labels_1d[i])
        out[i] = old_to_new[lo] if lo > 0 and lo in old_to_new else 0
    return out, old_to_new


def _median_axial_t_per_label(
    labels_full_zyx: np.ndarray,
    origin: np.ndarray,
    apix: float,
    pivot: np.ndarray,
    axis_u: np.ndarray,
) -> dict[int, float]:
    """Median axial coordinate t (Å) for each positive label id."""
    out: dict[int, float] = {}
    labs = np.unique(labels_full_zyx[labels_full_zyx > 0])
    for lab in labs:
        iz, iy, ix = np.nonzero(labels_full_zyx == int(lab))
        if iz.size == 0:
            continue
        P = _voxel_centers_xyz(iz, iy, ix, origin, apix).astype(np.float64)
        rel = P - pivot.reshape(1, 3)
        with np.errstate(all="ignore"):
            t = rel @ axis_u
        out[int(lab)] = float(np.median(t))
    return out


def _helical_average_from_representative(
    rep_vol_zyx: np.ndarray,
    *,
    origin: np.ndarray,
    apix: float,
    pivot: np.ndarray,
    axis: np.ndarray,
    twist_rad: float,
    rise_A: float,
    label_ids: list[int],
    helical_step_per_label: dict[int, int],
) -> np.ndarray:
    """
    Average subunit map: for each present label L, add a trilinear sample of ``rep_vol``
    at inverse-screw positions so all copies align to the representative frame.

    Uses the **same** row-vector screw as ``helical.finder.screw_correlation`` for one step:
    ``Q = pivot + ((P - pivot - dk*rise*u) @ R(+dk*twist))``. A minus twist here caused
    cumulative pitch drift vs the map the finder scored.

    helical_step_per_label[L] is the integer helical step of L relative to rep_id
    (0 for the representative label).
    """
    data = np.asarray(rep_vol_zyx, dtype=np.float64)
    Z, Y, X = data.shape
    axis_u = _unit(axis)
    p0 = np.asarray(pivot, dtype=np.float64).reshape(1, 3)
    ox, oy, oz = float(origin[0]), float(origin[1]), float(origin[2])
    acc = np.zeros_like(data, dtype=np.float64)
    n_terms = 0
    for lab in label_ids:
        dk = int(helical_step_per_label.get(int(lab), 0))
        theta = float(dk) * twist_rad
        R = _rotation_matrix(axis_u, theta)
        shift = float(dk) * float(rise_A) * axis_u.reshape(1, 3)
        for z in range(Z):
            iy, ix = np.meshgrid(np.arange(Y, dtype=np.float64), np.arange(X, dtype=np.float64), indexing="ij")
            iz = np.full_like(iy, float(z), dtype=np.float64)
            iz_f = iz.ravel().astype(np.int64)
            iy_f = iy.ravel().astype(np.int64)
            ix_f = ix.ravel().astype(np.int64)
            P = _voxel_centers_xyz(iz_f, iy_f, ix_f, origin, apix).astype(np.float64)
            with np.errstate(all="ignore"):
                Q = p0 + ((P - p0 - shift) @ R)
            ix_s = (Q[:, 0] - ox) / apix - 0.5
            iy_s = (Q[:, 1] - oy) / apix - 0.5
            iz_s = (Q[:, 2] - oz) / apix - 0.5
            coords = np.stack([iz_s, iy_s, ix_s], axis=0).reshape(3, Y, X)
            with np.errstate(all="ignore"):
                layer = map_coordinates(data, coords, order=1, mode="constant", cval=0.0)
            acc[z] += layer
        n_terms += 1
    if n_terms < 1:
        return np.zeros_like(rep_vol_zyx, dtype=np.float32)
    with np.errstate(all="ignore"):
        return (acc / float(n_terms)).astype(np.float32)


def _rotation_matrix(axis: np.ndarray, theta: float) -> np.ndarray:
    u = _unit(axis)
    x, y, z = (float(u[0]), float(u[1]), float(u[2]))
    c, s = math.cos(theta), math.sin(theta)
    C = 1.0 - c
    return np.array(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ],
        dtype=np.float64,
    )


@dataclass
class HelicalSegmentResult:
    input_map: str
    helical_find_json: str
    threshold: float
    n_subunits: int
    rise_A: float
    twist_deg: float
    axis_xyz: list[float]
    pivot_xyz: list[float]
    labels_map: str
    representative_map: str
    average_map: Optional[str]
    qc_png: Optional[str]
    output_json: str

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_helical_segment(
    map_path: Path,
    helical_find_json: Path,
    out_dir: Path,
    *,
    density_threshold: Optional[float] = None,
    k_window: int = 3,
    sigma_t_A: Optional[float] = None,
    sigma_phi_deg: Optional[float] = None,
    max_norm_cost: float = 12.0,
    min_cost_margin: float = 0.05,
    mode: str = "phase_peaks",
    radial_band_center_A: Optional[float] = None,
    radial_band_halfwidth_A: float = 2.5,
    axial_window_halfwidth_A: Optional[float] = None,
    peak_min_prominence: float = 0.0,
    shear_alpha_rad_per_A: Optional[float] = None,
    shear_alpha_pos_rad_per_A: Optional[float] = None,
    shear_alpha_neg_rad_per_A: Optional[float] = None,
    representative_largest_component: bool = False,
    prune_labels_largest_component: bool = False,
    watershed_max_norm_cost: Optional[float] = None,
    sequential_helical_labels: bool = True,
    write_qc_png: bool = True,
    write_average: bool = True,
) -> HelicalSegmentResult:
    map_path = Path(map_path).expanduser().resolve()
    helical_find_json = Path(helical_find_json).expanduser().resolve()
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    mv = read_map(map_path)
    data = mv.data_zyx.astype(np.float32)
    apix = float(mv.apix)
    origin = np.asarray(mv.origin_xyzA, dtype=np.float64)

    with open(helical_find_json, encoding="utf-8") as fh:
        hf = json.load(fh)

    thr = float(density_threshold) if density_threshold is not None else float(hf.get("threshold", np.percentile(data, 90.0)))
    rise = float(hf["best_rise_A"])
    twist_deg = float(hf["best_twist_deg"])
    twist_rad = math.radians(twist_deg)
    axis = _unit(np.asarray(hf["axis_xyz"], dtype=np.float64))
    pivot = np.asarray(hf["pivot_xyz"], dtype=np.float64)
    a, b = _perp_frame(axis)

    iz, iy, ix = np.nonzero(data > thr)
    if iz.size == 0:
        raise ValueError("No voxels above threshold; increase sensitivity or lower threshold.")

    P = _voxel_centers_xyz(iz, iy, ix, origin, apix).astype(np.float64)
    rel = P - pivot.reshape(1, 3)
    with np.errstate(all="ignore"):
        t = rel @ axis
        x = rel @ a
        y = rel @ b
    phi = np.arctan2(y, x)

    sig_t = float(sigma_t_A) if sigma_t_A is not None else max(0.5, 0.5 * abs(rise))
    sig_phi = math.radians(float(sigma_phi_deg) if sigma_phi_deg is not None else max(3.0, 0.35 * abs(twist_deg)))

    # Estimate global phase offsets so k-indexing is less biased by arbitrary frame origin.
    k_guess = np.rint(t / rise).astype(np.int64)
    t0 = float(np.median(t - k_guess.astype(np.float64) * rise))
    ph_res = _wrap_to_pi(phi - k_guess.astype(np.float64) * twist_rad)
    phi0 = float(np.angle(np.mean(np.exp(1j * ph_res))))

    k0 = np.rint((t - t0) / rise).astype(np.int64)
    best_k = k0.copy()
    best_cost = np.full(k0.shape, np.inf, dtype=np.float64)
    second_cost = np.full(k0.shape, np.inf, dtype=np.float64)
    kw = max(1, int(k_window))
    for dk in range(-kw, kw + 1):
        kk = k0 + dk
        dt = t - (kk.astype(np.float64) * rise + t0)
        dphi = _wrap_to_pi(phi - (kk.astype(np.float64) * twist_rad + phi0))
        cost = (dt / sig_t) ** 2 + (dphi / sig_phi) ** 2
        upd2 = cost < second_cost
        second_cost[upd2] = cost[upd2]
        take = cost < best_cost
        second_cost[take] = best_cost[take]
        best_cost[take] = cost[take]
        best_k[take] = kk[take]

    margin = second_cost - best_cost
    confident = (best_cost <= float(max_norm_cost)) & (margin >= float(min_cost_margin))

    # Relabel to compact positive IDs in ascending helical order.
    uniq = np.unique(best_k[confident])
    id_map = {int(k): i + 1 for i, k in enumerate(sorted(int(x) for x in uniq))}
    labels = np.zeros(best_k.shape, dtype=np.int32)
    for idx, k in enumerate(best_k):
        if confident[idx]:
            labels[idx] = id_map[int(k)]
    n_sub = int(len(id_map))

    phase_peaks_meta: dict[str, Any] = {}
    if mode == "phase_peaks" and n_sub > 0:
        # 1) radial band selection for repeat profile
        rho = np.sqrt(x * x + y * y)
        if radial_band_center_A is None:
            rbins = np.linspace(float(np.percentile(rho, 2.0)), float(np.percentile(rho, 98.0)), 120)
            if np.unique(rbins).size < 4:
                r0 = float(np.median(rho))
            else:
                hist, e = np.histogram(rho, bins=rbins, weights=data[iz, iy, ix].astype(np.float64))
                j = int(np.argmax(hist))
                r0 = float(0.5 * (e[j] + e[j + 1]))
        else:
            r0 = float(radial_band_center_A)
        rw = max(0.5, float(radial_band_halfwidth_A))
        in_band = np.abs(rho - r0) <= rw
        if np.count_nonzero(in_band) > 32:
            t_band = t[in_band]
            v_band = data[iz[in_band], iy[in_band], ix[in_band]].astype(np.float64)
            # 2) axial profile + peak detection
            tmin, tmax = float(np.min(t_band)), float(np.max(t_band))
            dz = max(0.3, min(apix, abs(rise) / 8.0))
            nb = max(32, int(np.ceil((tmax - tmin) / dz)))
            edges = np.linspace(tmin, tmax, nb + 1)
            prof, _ = np.histogram(t_band, bins=edges, weights=v_band)
            prof = ndimage.gaussian_filter1d(prof.astype(np.float64), sigma=1.2)
            dist = max(1, int(round(abs(rise) / dz * 0.6)))
            prom = float(peak_min_prominence) if peak_min_prominence > 0 else max(0.0, 0.05 * float(np.max(prof)))
            pk, props = find_peaks(prof, distance=dist, prominence=prom)
            if pk.size > 0:
                centers = 0.5 * (edges[pk] + edges[pk + 1])
                # 3) local center refinement in ±rise/3 window
                hw_ref = max(0.4, abs(rise) / 3.0)
                cref = []
                for c in centers:
                    m = np.abs(t - c) <= hw_ref
                    if np.any(m):
                        wloc = np.maximum(data[iz[m], iy[m], ix[m]].astype(np.float64), 0.0)
                        tt = t[m]
                        s = float(np.sum(wloc))
                        cref.append(float(np.sum(tt * wloc) / s) if s > 1e-12 else float(c))
                    else:
                        cref.append(float(c))
                centers = np.asarray(sorted(cref), dtype=np.float64)
                # 3b) Lattice regularization: enforce one center per rise period.
                if centers.size >= 2:
                    zref = float(np.median(centers))
                    kk = np.rint((centers - zref) / rise).astype(np.int64)
                    z0 = float(np.median(centers - kk.astype(np.float64) * rise))
                    kmin = int(np.floor((float(np.min(t)) - z0) / rise)) - 1
                    kmax = int(np.ceil((float(np.max(t)) - z0) / rise)) + 1
                    centers_reg = z0 + np.arange(kmin, kmax + 1, dtype=np.float64) * rise
                    # keep centers within observed range with small halo
                    halo = 0.6 * abs(rise)
                    centers = centers_reg[(centers_reg >= float(np.min(t)) - halo) & (centers_reg <= float(np.max(t)) + halo)]
                # 4) bounded axial-window assignment
                hw_assign = float(axial_window_halfwidth_A) if axial_window_halfwidth_A is not None else max(0.6, 0.55 * abs(rise))
                if centers.size > 0:
                    # Helical phase-consistent assignment with shear coupling:
                    # cost(j) = (dz/sig_t)^2 + ((dtheta - alpha*dz)/sig_phi)^2
                    k_cent = np.rint((centers - t0) / rise).astype(np.int64)
                    theta_cent = phi0 + k_cent.astype(np.float64) * twist_rad
                    dtm = t.reshape(-1, 1) - centers.reshape(1, -1)
                    dph0 = _wrap_to_pi(phi.reshape(-1, 1) - theta_cent.reshape(1, -1))

                    # pass 1 (alpha=0) for provisional pairing
                    with np.errstate(all="ignore"):
                        c0 = (dtm / max(sig_t, 1e-6)) ** 2 + (dph0 / max(sig_phi, 1e-6)) ** 2
                    j0 = np.argmin(c0, axis=1)
                    dz0 = dtm[np.arange(dtm.shape[0]), j0]
                    dth0 = dph0[np.arange(dtm.shape[0]), j0]

                    # Shear: single slope (--shear-alpha-rad-per-A) or piecewise dz>=0 vs dz<0
                    # (manifold / bent fibril: different effective twist-per-Å above vs below each center).
                    den_all = float(np.dot(dz0, dz0))
                    alpha_global = (
                        _clip_shear_alpha(float(np.dot(dz0, dth0) / den_all), rise) if den_all > 1e-12 else 0.0
                    )
                    if shear_alpha_rad_per_A is not None:
                        alpha_pos = alpha_neg = float(shear_alpha_rad_per_A)
                    elif shear_alpha_pos_rad_per_A is not None and shear_alpha_neg_rad_per_A is not None:
                        alpha_pos = float(shear_alpha_pos_rad_per_A)
                        alpha_neg = float(shear_alpha_neg_rad_per_A)
                    else:
                        pos_m = dz0 > 0.0
                        neg_m = dz0 < 0.0
                        alpha_pos = (
                            _fit_shear_slope(dz0[pos_m], dth0[pos_m], rise)
                            if np.count_nonzero(pos_m) >= 8
                            else alpha_global
                        )
                        alpha_neg = (
                            _fit_shear_slope(dz0[neg_m], dth0[neg_m], rise)
                            if np.count_nonzero(neg_m) >= 8
                            else alpha_global
                        )

                    piecewise_shear = shear_alpha_rad_per_A is None
                    if piecewise_shear:
                        alpha_eff = np.where(dtm >= 0.0, alpha_pos, alpha_neg)
                    else:
                        alpha_eff = alpha_pos

                    with np.errstate(all="ignore"):
                        c1 = (dtm / max(sig_t, 1e-6)) ** 2 + (
                            (_wrap_to_pi(dph0 - alpha_eff * dtm)) / max(sig_phi, 1e-6)
                        ) ** 2
                    jmin = np.argmin(c1, axis=1)
                    cmin = c1[np.arange(c1.shape[0]), jmin]
                    # second-best margin from full cost matrix
                    csort = np.sort(c1, axis=1)
                    margin2 = csort[:, 1] - csort[:, 0] if csort.shape[1] > 1 else np.full(cmin.shape, np.inf, dtype=np.float64)
                    zmin = np.abs(dtm[np.arange(dtm.shape[0]), jmin])
                    core = (cmin <= float(max_norm_cost)) & (margin2 >= float(min_cost_margin)) & (zmin <= hw_assign)
                    labels_soft = (jmin + 1).astype(np.int32)
                    labels = np.where(core, labels_soft, 0).astype(np.int32)

                    # Fallback if too strict: keep low-cost nearest labels (no margin gate).
                    if int(np.count_nonzero(labels)) < max(256, int(0.01 * labels.size)):
                        loose = (cmin <= float(max_norm_cost) * 1.8) & (zmin <= hw_assign * 1.2)
                        labels = np.where(loose, labels_soft, 0).astype(np.int32)
                        core = loose

                    # Connectivity-aware fill from high-confidence cores.
                    marker_vol = np.zeros(data.shape, dtype=np.int32)
                    if np.any(core):
                        marker_vol[iz[core], iy[core], ix[core]] = labels_soft[core]
                        ws = watershed((-data).astype(np.float32), marker_vol, mask=(data > thr))
                        labels = ws[iz, iy, ix].astype(np.int32)
                        # Strip ambiguous watershed fill only: expanded voxels often have higher cmin than
                        # seeds; applying the same threshold to everyone removes most of the fill (~15% left).
                        # Always keep high-confidence core seeds (same mask that built marker_vol).
                        if watershed_max_norm_cost is not None:
                            wmc = float(watershed_max_norm_cost)
                            keep = core | ((cmin <= wmc) & (labels > 0))
                            labels = np.where(keep, labels, 0).astype(np.int32)

                    n_sub = int(centers.size)
                    _cost_desc = (
                        "(dz/sig_t)^2 + (wrap(dtheta - alpha(dz)*dz)/sig_phi)^2, alpha(dz)=alpha_pos if dz>=0 else alpha_neg"
                        if piecewise_shear
                        else "(dz/sig_t)^2 + (wrap(dtheta - alpha*dz)/sig_phi)^2"
                    )
                    phase_peaks_meta = {
                        "radial_band_center_A": float(r0),
                        "radial_band_halfwidth_A": float(rw),
                        "axial_bin_dz_A": float(dz),
                        "n_profile_peaks": int(pk.size),
                        "n_regularized_centers": int(centers.size),
                        "peak_prominence_threshold": float(prom),
                        "axial_window_halfwidth_A": float(hw_assign),
                        "shear_alpha_pos_rad_per_A": float(alpha_pos),
                        "shear_alpha_neg_rad_per_A": float(alpha_neg),
                        "shear_piecewise_dz": bool(piecewise_shear),
                        "shear_alpha_rad_per_A": float(shear_alpha_rad_per_A)
                        if shear_alpha_rad_per_A is not None
                        else float(0.5 * (alpha_pos + alpha_neg)),
                        "assignment_cost": _cost_desc,
                        "n_core_voxels": int(np.count_nonzero(core)),
                        "watershed_max_norm_cost": float(watershed_max_norm_cost)
                        if watershed_max_norm_cost is not None
                        else None,
                        "watershed_gate_preserves_core": bool(watershed_max_norm_cost is not None),
                        "repeat_center_t_A": [float(c) for c in centers],
                    }

    # Optional seeded-watershed reassignment: use one canonical seed feature and propagate by screw.
    if mode == "seeded_watershed" and n_sub > 0:
        # Build canonical average map using confident assignments.
        avg0 = np.zeros_like(data, dtype=np.float64)
        wgt0 = np.zeros_like(data, dtype=np.float64)
        vals = data[iz, iy, ix].astype(np.float64)
        for k_val, lab in id_map.items():
            m = labels == lab
            if not np.any(m):
                continue
            Pm = P[m]
            Vm = vals[m]
            theta = -k_val * twist_rad
            R = _rotation_matrix(axis, theta)
            shift = k_val * rise * axis.reshape(1, 3)
            with np.errstate(all="ignore"):
                Q = pivot.reshape(1, 3) + ((Pm - pivot.reshape(1, 3) - shift) @ R)
            ix_q = np.rint((Q[:, 0] - origin[0]) / apix - 0.5).astype(np.int64)
            iy_q = np.rint((Q[:, 1] - origin[1]) / apix - 0.5).astype(np.int64)
            iz_q = np.rint((Q[:, 2] - origin[2]) / apix - 0.5).astype(np.int64)
            ok = (
                (iz_q >= 0)
                & (iz_q < data.shape[0])
                & (iy_q >= 0)
                & (iy_q < data.shape[1])
                & (ix_q >= 0)
                & (ix_q < data.shape[2])
            )
            iz2, iy2, ix2, vv = iz_q[ok], iy_q[ok], ix_q[ok], Vm[ok]
            avg0[iz2, iy2, ix2] += vv
            wgt0[iz2, iy2, ix2] += 1.0
        with np.errstate(all="ignore"):
            avg0 = np.where(wgt0 > 0, avg0 / np.maximum(wgt0, 1e-9), 0.0)
        # Canonical key feature = highest-density voxel in canonical average.
        seed_flat = int(np.argmax(avg0))
        seed_iz, seed_iy, seed_ix = np.unravel_index(seed_flat, avg0.shape)
        seed_xyz = np.array(
            [
                origin[0] + (seed_ix + 0.5) * apix,
                origin[1] + (seed_iy + 0.5) * apix,
                origin[2] + (seed_iz + 0.5) * apix,
            ],
            dtype=np.float64,
        )

        # Populate seed markers at all subunit-related positions.
        marker_vol = np.zeros(data.shape, dtype=np.int32)
        sid = 1
        for k_val in sorted(id_map.keys()):
            theta = k_val * twist_rad
            Rt = _rotation_matrix(axis, theta).T
            shift = k_val * rise * axis.reshape(1, 3)
            with np.errstate(all="ignore"):
                Pw = pivot.reshape(1, 3) + shift + ((seed_xyz.reshape(1, 3) - pivot.reshape(1, 3)) @ Rt)
            px, py, pz = float(Pw[0, 0]), float(Pw[0, 1]), float(Pw[0, 2])
            ixw = int(round((px - origin[0]) / apix - 0.5))
            iyw = int(round((py - origin[1]) / apix - 0.5))
            izw = int(round((pz - origin[2]) / apix - 0.5))
            if 0 <= izw < data.shape[0] and 0 <= iyw < data.shape[1] and 0 <= ixw < data.shape[2]:
                if data[izw, iyw, ixw] > thr:
                    marker_vol[izw, iyw, ixw] = sid
                    sid += 1
        if sid > 1:
            # Watershed over inverted density inside threshold mask.
            elev = (-data).astype(np.float32)
            ws = watershed(elev, marker_vol, mask=(data > thr))
            # keep only confident-assigned region support to reduce flooding artifacts
            ws = ws.astype(np.int32)
            labels = ws[iz, iy, ix]
            n_sub = int(np.max(ws))

    if prune_labels_largest_component and n_sub > 0:
        labels = _prune_largest_cc_per_label_sparse(labels, iz, iy, ix, data.shape)

    label_remap_meta: dict[str, Any] = {}
    remap_old_to_new: dict[int, int] = {}
    if sequential_helical_labels and np.any(labels > 0):
        mx0 = int(np.max(labels))
        cnt0 = np.bincount(labels[labels > 0], minlength=mx0 + 1)
        rep_old = int(np.argmax(cnt0[1:]) + 1)
        axis_u_pre = _unit(axis)
        labels, old_to_new = _relabel_sequential_along_axis(
            labels, iz, iy, ix, data.shape, origin, apix, pivot, axis_u_pre
        )
        if old_to_new:
            remap_old_to_new = old_to_new
            n_sub = int(np.max(labels))
            rep_id = int(old_to_new.get(rep_old, 1))
            label_remap_meta = {
                "sequential_helical_labels": True,
                "old_label_to_new": {str(k): int(v) for k, v in old_to_new.items()},
                "representative_label_before_remap": int(rep_old),
            }
        else:
            cnt = np.bincount(labels[labels > 0], minlength=n_sub + 1) if n_sub > 0 else np.array([0], dtype=np.int64)
            if n_sub == 0:
                raise ValueError("No confident subunit assignments; lower threshold or loosen tolerances.")
            rep_id = int(np.argmax(cnt[1:]) + 1)
    else:
        cnt = np.bincount(labels[labels > 0], minlength=n_sub + 1) if n_sub > 0 else np.array([0], dtype=np.int64)
        if n_sub == 0:
            raise ValueError("No confident subunit assignments; lower threshold or loosen tolerances.")
        rep_id = int(np.argmax(cnt[1:]) + 1)

    if np.any(labels > 0):
        n_sub = int(np.max(labels))

    # Integer-valued label map (one distinct id per subunit along axis after remap).
    label_vol = np.zeros_like(data, dtype=np.float32)
    label_vol[iz, iy, ix] = np.rint(labels.astype(np.float64)).astype(np.float32)
    labels_path = out_dir / "helical_subunit_labels.mrc"
    write_map(labels_path, mv, label_vol)

    # Representative: same physical subunit as before axial remap (rep_old → rep_id).
    if n_sub == 0:
        raise ValueError("No confident subunit assignments; lower threshold or loosen tolerances.")
    rep_vol = np.zeros_like(data, dtype=np.float32)
    rep_mask = labels == rep_id
    rep_tmp = np.zeros_like(data, dtype=bool)
    rep_tmp[iz[rep_mask], iy[rep_mask], ix[rep_mask]] = True
    if representative_largest_component:
        cc, ncc = ndimage.label(rep_tmp)
        if ncc > 1:
            sizes = np.bincount(cc.ravel())
            sizes[0] = 0
            keep = int(np.argmax(sizes))
            rep_tmp = cc == keep
    rep_vol[rep_tmp] = data[rep_tmp]
    rep_path = out_dir / "helical_subunit_representative.mrc"
    write_map(rep_path, mv, rep_vol)

    avg_path: Optional[Path] = None
    average_meta: dict[str, Any] = {}
    if write_average:
        labels_full = np.zeros(data.shape, dtype=np.int32)
        labels_full[iz, iy, ix] = labels.astype(np.int32)
        labs_present = sorted(
            int(x) for x in np.unique(labels_full[labels_full > 0].ravel()) if int(x) > 0
        )
        if len(labs_present) == 0:
            raise ValueError("No labels for average map.")
        if sequential_helical_labels:
            # Integer Δk = label - rep_id (labels are 1..K along axis after remap); matches lattice pitch.
            helical_step = {lab: int(lab) - int(rep_id) for lab in labs_present}
            step_basis = "label_minus_representative_after_axial_remap"
        else:
            axis_u = _unit(axis)
            t_med = _median_axial_t_per_label(labels_full, origin, apix, pivot, axis_u)
            t_rep = t_med.get(rep_id)
            if t_rep is None:
                t_rep = t_med[labs_present[len(labs_present) // 2]]
            rise_eff = float(rise) if abs(float(rise)) > 1e-9 else 1.0
            helical_step = {
                lab: int(np.round((t_med[lab] - t_rep) / rise_eff)) for lab in labs_present
            }
            step_basis = "median_axial_t_over_rise"
        avg_map = _helical_average_from_representative(
            rep_vol,
            origin=origin,
            apix=apix,
            pivot=pivot,
            axis=axis,
            twist_rad=twist_rad,
            rise_A=float(rise),
            label_ids=labs_present,
            helical_step_per_label=helical_step,
        )
        avg_path = out_dir / "helical_subunit_average.mrc"
        write_map(avg_path, mv, avg_map)
        average_meta = {
            "average_method": "representative_screw_stack",
            "helical_step_relative_to_rep": {str(k): v for k, v in helical_step.items()},
            "helical_step_basis": step_basis,
            "screw_twist_sign": "finder_correlation",
            "screw_formula": "Q = pivot + ((P - pivot - dk*rise*u) @ R(+dk*twist))",
        }

    qc_png: Optional[str] = None
    if write_qc_png:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        qc_path = out_dir / "helical_segment_qc.png"
        fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.2), dpi=140)
        ax0, ax1 = axs
        # Left: cost margin diagnostics
        sel = np.isfinite(best_cost) & np.isfinite(margin)
        if np.any(sel):
            xplot = np.clip(best_cost[sel], 0.0, 25.0)
            yplot = np.clip(margin[sel], -5.0, 25.0)
            ax0.scatter(xplot, yplot, s=2, alpha=0.15, c="tab:blue")
        ax0.axvline(float(max_norm_cost), color="tab:red", ls="--", lw=1.1, label="max_norm_cost")
        ax0.axhline(float(min_cost_margin), color="tab:orange", ls="--", lw=1.1, label="min_cost_margin")
        ax0.set_xlabel("Best normalized cost")
        ax0.set_ylabel("2nd-best - best cost")
        ax0.set_title("Assignment confidence")
        ax0.legend(loc="best", fontsize=7)
        ax0.grid(True, alpha=0.25)

        # Right: label occupancy (top bins)
        if n_sub > 0:
            c = np.bincount(labels[labels > 0], minlength=n_sub + 1)[1:]
            order = np.argsort(c)[::-1]
            top = min(15, order.size)
            xs = np.arange(top)
            ax1.bar(xs, c[order[:top]], color="tab:green", alpha=0.8)
            ax1.set_xticks(xs)
            ax1.set_xticklabels([str(int(i + 1)) for i in order[:top]], rotation=60, fontsize=7)
        ax1.set_xlabel("Subunit label ID (top by size)")
        ax1.set_ylabel("Voxel count")
        ax1.set_title("Label occupancy")
        ax1.grid(True, axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(qc_path)
        plt.close(fig)
        qc_png = str(qc_path)

    if phase_peaks_meta and remap_old_to_new:
        rc = phase_peaks_meta.get("repeat_center_t_A")
        if isinstance(rc, list) and len(rc) > 0 and n_sub > 0:
            n2o = {int(v): int(k) for k, v in remap_old_to_new.items()}
            rc_new: list[float] = []
            for j in range(1, n_sub + 1):
                old = n2o.get(j)
                if old is not None and 1 <= old <= len(rc):
                    rc_new.append(float(rc[old - 1]))
            if len(rc_new) == n_sub:
                phase_peaks_meta["repeat_center_t_A"] = rc_new

    out_json = out_dir / "helical_segment.json"
    result = HelicalSegmentResult(
        input_map=str(map_path),
        helical_find_json=str(helical_find_json),
        threshold=float(thr),
        n_subunits=n_sub,
        rise_A=rise,
        twist_deg=twist_deg,
        axis_xyz=[float(x) for x in axis],
        pivot_xyz=[float(x) for x in pivot],
        labels_map=str(labels_path),
        representative_map=str(rep_path),
        average_map=str(avg_path) if avg_path else None,
        qc_png=qc_png,
        output_json=str(out_json),
    )
    payload = result.to_json_dict()
    payload["k_window"] = int(kw)
    payload["sigma_t_A"] = float(sig_t)
    payload["sigma_phi_deg"] = math.degrees(sig_phi)
    payload["t0_A"] = float(t0)
    payload["phi0_deg"] = math.degrees(phi0)
    payload["max_norm_cost"] = float(max_norm_cost)
    payload["min_cost_margin"] = float(min_cost_margin)
    payload["representative_largest_component"] = bool(representative_largest_component)
    payload["prune_labels_largest_component"] = bool(prune_labels_largest_component)
    payload["watershed_max_norm_cost"] = (
        float(watershed_max_norm_cost) if watershed_max_norm_cost is not None else None
    )
    payload["mode"] = str(mode)
    if average_meta:
        payload["average"] = average_meta
    if label_remap_meta:
        payload["label_remap"] = label_remap_meta
    if phase_peaks_meta:
        payload["phase_peaks"] = phase_peaks_meta
    payload["qc_png"] = qc_png
    payload["n_confident_voxels"] = int(np.count_nonzero(labels > 0))
    payload["representative_label_id"] = rep_id
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return result

