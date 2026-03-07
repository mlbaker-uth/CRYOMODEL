from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import math

import numpy as np
import gemmi
from scipy import ndimage as ndi
from scipy.interpolate import splprep, splev
from skimage.graph import route_through_array

from ..io.mrc import read_map, write_map, MapVolume


@dataclass
class AxisResult:
    points_zyx: np.ndarray  # (N, 3) voxel coordinates (z,y,x)
    points_xyzA: np.ndarray  # (N, 3) Å coordinates (x,y,z)


def _voxel_to_xyz(ijk: np.ndarray, apix: float, origin_xyzA: np.ndarray) -> np.ndarray:
    ijk = np.asarray(ijk, dtype=float)
    xyz = np.empty_like(ijk, dtype=float)
    xyz[:, 0] = origin_xyzA[0] + (ijk[:, 2] + 0.5) * apix
    xyz[:, 1] = origin_xyzA[1] + (ijk[:, 1] + 0.5) * apix
    xyz[:, 2] = origin_xyzA[2] + (ijk[:, 0] + 0.5) * apix
    return xyz


def _xyz_to_voxel(xyz: np.ndarray, apix: float, origin_xyzA: np.ndarray) -> np.ndarray:
    xyz = np.asarray(xyz, dtype=float)
    out = np.empty_like(xyz, dtype=int)
    out[:, 2] = np.floor((xyz[:, 0] - origin_xyzA[0]) / apix).astype(int)
    out[:, 1] = np.floor((xyz[:, 1] - origin_xyzA[1]) / apix).astype(int)
    out[:, 0] = np.floor((xyz[:, 2] - origin_xyzA[2]) / apix).astype(int)
    return out


def _largest_component(mask: np.ndarray) -> np.ndarray:
    lab, n = ndi.label(mask)
    if n == 0:
        raise ValueError("No connected component found at the chosen threshold.")
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    return lab == int(np.argmax(sizes))


def _cleanup_mask(mask: np.ndarray, close_iters: int = 0) -> np.ndarray:
    out = _largest_component(mask)
    if close_iters > 0:
        out = ndi.binary_closing(out, structure=np.ones((3, 3, 3), dtype=bool), iterations=close_iters)
        out = ndi.binary_fill_holes(out)
        out = _largest_component(out)
    return out


def _read_pdb_points(path: Path) -> np.ndarray:
    pts = []
    for line in Path(path).read_text().splitlines():
        rec = line[:6].strip()
        if rec in {"ATOM", "HETATM"}:
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                pts.append([x, y, z])
            except ValueError:
                continue
    if len(pts) < 2:
        raise ValueError("Guide PDB must contain at least two ATOM/HETATM records.")
    return np.asarray(pts, dtype=float)


def _estimate_endpoints(mask: np.ndarray, apix: float, origin_xyzA: np.ndarray, dt: np.ndarray, cap_fraction: float = 0.03):
    ijk = np.argwhere(mask)
    if len(ijk) < 10:
        raise ValueError("Mask is too small to estimate endpoints.")

    xyz = _voxel_to_xyz(ijk.astype(float), apix, origin_xyzA)
    cen = xyz.mean(axis=0)
    centered = xyz - cen
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    axis = vh[0]
    proj = centered @ axis
    lo = np.quantile(proj, cap_fraction)
    hi = np.quantile(proj, 1.0 - cap_fraction)
    cap1 = ijk[proj <= lo]
    cap2 = ijk[proj >= hi]
    if len(cap1) == 0 or len(cap2) == 0:
        raise ValueError("Failed to identify terminal caps. Try lowering cap_fraction.")
    ep1 = cap1[np.argmax(dt[cap1[:, 0], cap1[:, 1], cap1[:, 2]])]
    ep2 = cap2[np.argmax(dt[cap2[:, 0], cap2[:, 1], cap2[:, 2]])]
    return np.array(ep1, dtype=int), np.array(ep2, dtype=int), axis, cen


def _anchor_point_to_mask(
    mask: np.ndarray,
    dt: np.ndarray,
    apix: float,
    origin_xyzA: np.ndarray,
    point_xyz: np.ndarray,
    search_radius_A: float = 8.0,
    dt_weight: float = 2.0,
) -> np.ndarray:
    ijk_all = np.argwhere(mask)
    xyz_all = _voxel_to_xyz(ijk_all.astype(float), apix, origin_xyzA)
    d = np.linalg.norm(xyz_all - point_xyz[None, :], axis=1)
    within = d <= search_radius_A
    if np.any(within):
        cand_ijk = ijk_all[within]
        cand_d = d[within]
    else:
        order = np.argsort(d)[: min(1000, len(d))]
        cand_ijk = ijk_all[order]
        cand_d = d[order]
    cand_dt = dt[cand_ijk[:, 0], cand_ijk[:, 1], cand_ijk[:, 2]]
    score = dt_weight * cand_dt - cand_d
    return np.asarray(cand_ijk[np.argmax(score)], dtype=int)


def _polyline_length(xyz: np.ndarray) -> float:
    if len(xyz) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum())


def _path_segment_lengths(xyz: np.ndarray) -> np.ndarray:
    if len(xyz) < 2:
        return np.zeros(0, dtype=float)
    return np.linalg.norm(np.diff(xyz, axis=0), axis=1)


def _cumulative_arc(xyz: np.ndarray) -> np.ndarray:
    seg = _path_segment_lengths(xyz)
    return np.concatenate([[0.0], np.cumsum(seg)])


def _interpolate_along_polyline(xyz: np.ndarray, s_query: np.ndarray) -> np.ndarray:
    s = _cumulative_arc(xyz)
    if s[-1] == 0:
        return np.repeat(xyz[:1], len(s_query), axis=0)
    s_query = np.clip(np.asarray(s_query, dtype=float), 0.0, s[-1])
    out = np.empty((len(s_query), 3), dtype=float)
    idx = np.searchsorted(s, s_query, side="right") - 1
    idx = np.clip(idx, 0, len(s) - 2)
    s0 = s[idx]
    s1 = s[idx + 1]
    denom = np.maximum(1e-12, s1 - s0)
    t = ((s_query - s0) / denom)[:, None]
    out = xyz[idx] * (1.0 - t) + xyz[idx + 1] * t
    return out


def _crop_to_mask_bbox(mask: np.ndarray, *arrays, pad_vox: int = 2):
    ijk = np.argwhere(mask)
    if len(ijk) == 0:
        raise ValueError("Mask is empty; cannot crop.")
    lo = np.maximum(ijk.min(axis=0) - pad_vox, 0)
    hi = np.minimum(ijk.max(axis=0) + pad_vox + 1, np.array(mask.shape))
    slices = tuple(slice(int(lo[d]), int(hi[d])) for d in range(3))
    cropped = [mask[slices]]
    for arr in arrays:
        cropped.append(arr[slices])
    return cropped, lo, hi, slices


def _resample_polyline_uniform(xyz: np.ndarray, n_points: int) -> np.ndarray:
    if len(xyz) < 2:
        return xyz.copy()
    s = _cumulative_arc(xyz)
    samples = np.linspace(0.0, s[-1], max(2, n_points))
    out = _interpolate_along_polyline(xyz, samples)
    out[0] = xyz[0]
    out[-1] = xyz[-1]
    return out


def _densify_polyline(xyz: np.ndarray, step_A: float = 0.25) -> tuple[np.ndarray, np.ndarray]:
    s = _cumulative_arc(xyz)
    if s[-1] == 0:
        return xyz.copy(), s
    n = max(2, int(math.ceil(s[-1] / step_A)) + 1)
    s_dense = np.linspace(0.0, s[-1], n)
    xyz_dense = _interpolate_along_polyline(xyz, s_dense)
    return xyz_dense, s_dense


def _trace_path_between_voxels(cost: np.ndarray, start_ijk: np.ndarray, end_ijk: np.ndarray, apix: float, origin_xyzA: np.ndarray):
    path, total_cost = route_through_array(
        cost,
        start=tuple(int(x) for x in start_ijk),
        end=tuple(int(x) for x in end_ijk),
        fully_connected=True,
        geometric=True,
    )
    path = np.asarray(path, dtype=float)
    xyz = _voxel_to_xyz(path, apix, origin_xyzA)
    return xyz, float(total_cost)


def _gaussian_smooth_curve(xyz: np.ndarray, sigma=1.0, n_iter=2, pin_start=None, pin_end=None) -> np.ndarray:
    if len(xyz) < 3 or sigma <= 0 or n_iter <= 0:
        out = xyz.copy()
        if pin_start is not None:
            out[0] = pin_start
        if pin_end is not None:
            out[-1] = pin_end
        return out
    out = xyz.copy()
    for _ in range(n_iter):
        tmp = out.copy()
        for j in range(3):
            tmp[:, j] = ndi.gaussian_filter1d(out[:, j], sigma=sigma, mode="nearest")
        out = tmp
        if pin_start is not None:
            out[0] = pin_start
        if pin_end is not None:
            out[-1] = pin_end
    return out


def _smooth_segment_geometry(xyz, smooth, pre_smooth_sigma, pre_smooth_iters, pin_start, pin_end, min_points=25):
    work = _gaussian_smooth_curve(xyz, sigma=pre_smooth_sigma, n_iter=pre_smooth_iters, pin_start=pin_start, pin_end=pin_end)
    seg = _path_segment_lengths(work)
    keep = np.concatenate([[True], seg > 1e-6]) if len(work) > 1 else np.array([True])
    work = work[keep]
    if len(work) < 4:
        out = np.vstack([pin_start[None, :], work[1:-1], pin_end[None, :]]) if len(work) > 2 else np.vstack([pin_start, pin_end])
        return _resample_polyline_uniform(out, max(min_points, len(out)))
    u = _cumulative_arc(work)
    u = u / max(u[-1], 1e-12)
    k = min(3, len(work) - 1)
    tck, _ = splprep([work[:, 0], work[:, 1], work[:, 2]], u=u, s=smooth, k=k)
    uu = np.linspace(0.0, 1.0, max(min_points, len(work)))
    xs, ys, zs = splev(uu, tck)
    out = np.vstack([xs, ys, zs]).T
    out[0] = pin_start
    out[-1] = pin_end
    return out


def _sample_dt_along_xyz(xyz: np.ndarray, dt: np.ndarray, apix: float, origin_xyzA: np.ndarray) -> np.ndarray:
    ijk = _xyz_to_voxel(xyz, apix, origin_xyzA)
    ijk = np.clip(ijk, [0, 0, 0], np.array(dt.shape) - 1)
    return dt[ijk[:, 0], ijk[:, 1], ijk[:, 2]]


def _refine_uniform_points_along_path(
    xyz: np.ndarray,
    n_points: int,
    dt: np.ndarray,
    apix: float,
    origin_xyzA: np.ndarray,
    search_window_A: float = 0.8,
    dense_step_A: float = 0.2,
    dt_weight: float = 0.35,
) -> tuple[np.ndarray, list[float]]:
    xyz_dense, s_dense = _densify_polyline(xyz, step_A=dense_step_A)
    dt_dense = _sample_dt_along_xyz(xyz_dense, dt, apix, origin_xyzA)
    length = s_dense[-1]
    targets = np.linspace(0.0, length, max(2, n_points))
    selected_s = [0.0]
    min_sep = max(0.5, 0.45 * (length / max(1, n_points - 1)))
    for k in range(1, n_points - 1):
        t = targets[k]
        lo = max(selected_s[-1] + min_sep, t - search_window_A)
        hi = min(length, t + search_window_A)
        if hi <= lo:
            selected_s.append(t)
            continue
        valid = np.where((s_dense >= lo) & (s_dense <= hi))[0]
        if len(valid) == 0:
            selected_s.append(t)
            continue
        penalty = ((s_dense[valid] - t) / max(search_window_A, 1e-6)) ** 2
        score = dt_dense[valid] - dt_weight * penalty
        best = valid[int(np.argmax(score))]
        selected_s.append(float(s_dense[best]))
    selected_s.append(length)
    selected_s = np.asarray(selected_s, dtype=float)
    selected_s = np.maximum.accumulate(selected_s)
    refined = _interpolate_along_polyline(xyz, selected_s)
    refined[0] = xyz[0]
    refined[-1] = xyz[-1]
    spacings = np.diff(selected_s).tolist()
    return refined, spacings


def _local_recenter_points(
    xyz: np.ndarray,
    mask: np.ndarray,
    dt: np.ndarray,
    apix: float,
    origin_xyzA: np.ndarray,
    radius_A: float = 1.2,
    dt_weight: float = 1.0,
    disp_weight: float = 0.35,
    n_iter: int = 1,
    pin_first: bool = True,
    pin_last: bool = True,
) -> np.ndarray:
    if len(xyz) < 3 or radius_A <= 0 or n_iter <= 0:
        return xyz.copy()

    out = xyz.copy()
    grid_radius = max(1, int(math.ceil(radius_A / apix)))
    shape = np.array(mask.shape)
    for _ in range(n_iter):
        new = out.copy()
        start_idx = 1 if pin_first else 0
        stop_idx = len(out) - 1 if pin_last else len(out)
        for i in range(start_idx, stop_idx):
            center_ijk = _xyz_to_voxel(out[i:i + 1], apix, origin_xyzA)[0]
            lo = np.maximum(center_ijk - grid_radius, 0)
            hi = np.minimum(center_ijk + grid_radius + 1, shape)
            cand = np.argwhere(mask[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]])
            if len(cand) == 0:
                continue
            cand = cand + lo
            cand_xyz = _voxel_to_xyz(cand.astype(float), apix, origin_xyzA)
            disp = np.linalg.norm(cand_xyz - out[i][None, :], axis=1)
            within = disp <= radius_A
            if np.any(within):
                cand = cand[within]
                cand_xyz = cand_xyz[within]
                disp = disp[within]
            cand_dt = dt[cand[:, 0], cand[:, 1], cand[:, 2]]
            score = dt_weight * cand_dt - disp_weight * disp
            best = int(np.argmax(score))
            new[i] = cand_xyz[best]
        if pin_first:
            new[0] = out[0]
        if pin_last:
            new[-1] = out[-1]
        out = new
    return out


def _write_pdb_polyline(path: Path, xyz: np.ndarray, chain_id: str = "A"):
    lines = []
    serial = 1
    resseq = 1
    prev_serial = None
    for p in xyz:
        x, y, z = p
        lines.append(
            f"ATOM  {serial:5d}  CA  GLY {chain_id}{resseq:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}{1.00:6.2f}{0.00:6.2f}           C"
        )
        if prev_serial is not None:
            lines.append(f"CONECT{prev_serial:5d}{serial:5d}")
            lines.append(f"CONECT{serial:5d}{prev_serial:5d}")
        prev_serial = serial
        serial += 1
        resseq += 1
    lines.append("END")
    Path(path).write_text("\n".join(lines) + "\n")


def _save_report(path: Path, report: dict):
    lines = []
    for k, v in report.items():
        if isinstance(v, (dict, list)):
            lines.append(f"{k}: {json.dumps(v)}")
        else:
            lines.append(f"{k}: {v}")
    Path(path).write_text("\n".join(lines) + "\n")


def _choose_point_count(total_length_A: float, rise_per_bp_A: float, mode: str = "round", minimum_points: int = 2):
    if rise_per_bp_A <= 0:
        raise ValueError("rise_per_bp_A must be > 0")
    x = total_length_A / rise_per_bp_A
    if mode == "ceil":
        intervals = max(1, int(math.ceil(x)))
    elif mode == "floor":
        intervals = max(1, int(math.floor(x)))
    else:
        intervals = max(1, int(round(x)))
    n_points = max(minimum_points, intervals + 1)
    realized_spacing = total_length_A / max(1, n_points - 1)
    return n_points, realized_spacing, x


def _allocate_points_by_length(lengths: list[float], n_total: int) -> list[int]:
    nseg = len(lengths)
    if nseg == 0:
        return []
    if nseg == 1:
        return [max(2, n_total)]
    total_len = sum(max(1e-6, x) for x in lengths)
    interior_total = max(0, n_total - (nseg + 1))
    raw = [interior_total * max(1e-6, L) / total_len for L in lengths]
    interior = [max(0, int(math.floor(x))) for x in raw]
    rem = interior_total - sum(interior)
    order = np.argsort([x - math.floor(x) for x in raw])[::-1]
    for idx in order[:rem]:
        interior[idx] += 1
    return [n + 2 for n in interior]


def _concatenate_segments(segments: list[np.ndarray]) -> np.ndarray:
    if not segments:
        return np.zeros((0, 3), dtype=float)
    out = [segments[0]]
    for seg in segments[1:]:
        out.append(seg[1:])
    return np.vstack(out)


def _draw_polyline(volume: np.ndarray, points: np.ndarray) -> np.ndarray:
    if points.shape[0] == 0:
        return volume
    for i in range(len(points) - 1):
        p0 = points[i]
        p1 = points[i + 1]
        dist = np.linalg.norm(p1 - p0)
        steps = max(2, int(np.ceil(dist * 2)))
        for t in np.linspace(0, 1, steps):
            p = p0 * (1 - t) + p1 * t
            z, y, x = np.round(p).astype(int)
            if 0 <= z < volume.shape[0] and 0 <= y < volume.shape[1] and 0 <= x < volume.shape[2]:
                volume[z, y, x] = 1.0
    return volume


def extract_medial_axis(
    map_path: Path,
    threshold: float,
    n_points: int | None,
    out_mrc: Path,
    out_pdb: Path,
    guides_pdb: Path | None = None,
    endpoints_pdb: Path | None = None,
    power: float = 3.0,
    eps: float = 1e-3,
    cap_fraction: float = 0.03,
    close_iters: int = 0,
    smooth: float = 0.4,
    pre_smooth_sigma: float = 1.0,
    pre_smooth_iters: int = 2,
    guide_search_radius_A: float = 8.0,
    guide_dt_weight: float = 2.0,
    max_guide_span_A: float = 60.0,
    point_refine_window_A: float = 0.8,
    point_refine_dt_weight: float = 0.35,
    recenter_radius_A: float = 1.2,
    recenter_dt_weight: float = 1.0,
    recenter_disp_weight: float = 0.35,
    recenter_iters: int = 1,
    rise_per_bp_A: float = 3.4,
    target_spacing_A: float | None = None,
    count_mode: str = "round",
    report_path: Path | None = None,
) -> AxisResult:
    mv = read_map(map_path)
    apix = mv.apix
    origin = np.array(mv.origin_xyzA, dtype=float)
    vol = mv.data_zyx

    mask_full = _cleanup_mask(vol >= threshold, close_iters=close_iters)
    dt_full = ndi.distance_transform_edt(mask_full, sampling=[apix, apix, apix])
    auto_ep1_full, auto_ep2_full, axis, cen = _estimate_endpoints(mask_full, apix, origin, dt_full, cap_fraction=cap_fraction)

    (cropped, crop_lo, crop_hi, crop_slices) = _crop_to_mask_bbox(mask_full, vol, dt_full, pad_vox=3)
    mask, vol_crop, dt = cropped
    auto_ep1 = auto_ep1_full - crop_lo
    auto_ep2 = auto_ep2_full - crop_lo

    guide_pdb = guides_pdb if guides_pdb is not None else endpoints_pdb
    warnings = []
    if guide_pdb:
        guide_xyz = _read_pdb_points(guide_pdb)
        origin_crop = origin + crop_lo[[2, 1, 0]] * apix
        anchored_guides_ijk = np.array(
            [
                _anchor_point_to_mask(
                    mask,
                    dt,
                    apix,
                    origin_crop,
                    g,
                    search_radius_A=guide_search_radius_A,
                    dt_weight=guide_dt_weight,
                )
                for g in guide_xyz
            ],
            dtype=int,
        )
    else:
        anchored_guides_ijk = np.array([auto_ep1, auto_ep2], dtype=int)
        guide_xyz = _voxel_to_xyz((anchored_guides_ijk + crop_lo).astype(float), apix, origin)

    guide_straight = []
    for i in range(len(guide_xyz) - 1):
        d = float(np.linalg.norm(guide_xyz[i + 1] - guide_xyz[i]))
        guide_straight.append(d)
        if d > max_guide_span_A:
            warnings.append(
                f"Guide segment {i + 1} spans {d:.2f} A straight-line distance, above {max_guide_span_A:.2f} A. Consider adding intermediate guide points."
            )

    cost = np.full(mask.shape, np.inf, dtype=np.float32)
    cost[mask] = 1.0 / np.power(dt[mask] + eps, power)

    raw_segments = []
    raw_segment_lengths = []
    raw_segment_costs = []
    for i in range(len(anchored_guides_ijk) - 1):
        xyz_seg_local, seg_cost = _trace_path_between_voxels(cost, anchored_guides_ijk[i], anchored_guides_ijk[i + 1], apix, origin_xyzA=np.zeros(3))
        xyz_seg = xyz_seg_local + origin + crop_lo[[2, 1, 0]] * apix
        raw_segments.append(xyz_seg)
        raw_segment_lengths.append(_polyline_length(xyz_seg))
        raw_segment_costs.append(seg_cost)

    smoothed_geometry_segments = []
    smoothed_geometry_lengths = []
    for i, xyz_seg in enumerate(raw_segments):
        seg = _smooth_segment_geometry(
            xyz_seg,
            smooth=smooth,
            pre_smooth_sigma=pre_smooth_sigma,
            pre_smooth_iters=pre_smooth_iters,
            pin_start=guide_xyz[i],
            pin_end=guide_xyz[i + 1],
            min_points=max(25, len(xyz_seg)),
        )
        smoothed_geometry_segments.append(seg)
        smoothed_geometry_lengths.append(_polyline_length(seg))

    smoothed_geometry = _concatenate_segments(smoothed_geometry_segments)
    smoothed_total_length = _polyline_length(smoothed_geometry)

    effective_spacing = target_spacing_A if target_spacing_A is not None else rise_per_bp_A
    if n_points is None:
        n_points_final, target_spacing_A_eff, bp_count_est = _choose_point_count(smoothed_total_length, effective_spacing, mode=count_mode)
        point_count_mode = f"auto_from_smoothed_length_{count_mode}"
    else:
        n_points_final = max(2, int(n_points))
        target_spacing_A_eff = smoothed_total_length / max(1, n_points_final - 1)
        bp_count_est = smoothed_total_length / effective_spacing if effective_spacing > 0 else float("nan")
        point_count_mode = "user_supplied"

    if n_points is None:
        frac = abs(bp_count_est - round(bp_count_est))
        if frac < 0.18:
            warnings.append(
                f"Estimated interval count is near an integer boundary ({bp_count_est:.2f}). A +/-1 pseudoatom ambiguity is plausible; inspect the result visually."
            )

    final_segments = []
    per_segment_points = _allocate_points_by_length(smoothed_geometry_lengths, n_points_final)
    refined_spacings = []
    for i, seg in enumerate(smoothed_geometry_segments):
        provisional = _resample_polyline_uniform(seg, per_segment_points[i])
        refined, spacings = _refine_uniform_points_along_path(
            provisional,
            n_points=per_segment_points[i],
            dt=dt_full,
            apix=apix,
            origin_xyzA=origin,
            search_window_A=point_refine_window_A,
            dense_step_A=0.2,
            dt_weight=point_refine_dt_weight,
        )
        refined[0] = guide_xyz[i]
        refined[-1] = guide_xyz[i + 1]
        recentered = _local_recenter_points(
            refined,
            mask=mask_full,
            dt=dt_full,
            apix=apix,
            origin_xyzA=origin,
            radius_A=recenter_radius_A,
            dt_weight=recenter_dt_weight,
            disp_weight=recenter_disp_weight,
            n_iter=recenter_iters,
            pin_first=True,
            pin_last=True,
        )
        recentered[0] = guide_xyz[i]
        recentered[-1] = guide_xyz[i + 1]
        final_segments.append(recentered)
        refined_spacings.append(spacings)

    xyz = _concatenate_segments(final_segments)
    _write_pdb_polyline(out_pdb, xyz)

    final_length = _polyline_length(xyz)
    realized_spacing_A = final_length / max(1, len(xyz) - 1)
    all_spacings = _path_segment_lengths(xyz)
    dt_path = _sample_dt_along_xyz(xyz, dt_full, apix, origin)
    mask_volume = float(mask_full.sum() * (apix**3))
    r_eff = float(np.median(dt_path)) if len(dt_path) else float("nan")
    length_est = mask_volume / (math.pi * r_eff * r_eff) if r_eff > 0 else float("nan")

    report = {
        "input_mrc": str(map_path),
        "output_pdb": str(out_pdb),
        "threshold": threshold,
        "guide_pdb": str(guide_pdb) if guide_pdb else None,
        "voxel_size_A": [apix, apix, apix],
        "mask_voxels": int(mask_full.sum()),
        "mask_volume_A3": mask_volume,
        "auto_endpoint1_ijk": auto_ep1_full.tolist(),
        "auto_endpoint2_ijk": auto_ep2_full.tolist(),
        "guide_count": int(len(guide_xyz)),
        "guide_xyz_A": guide_xyz.tolist(),
        "anchored_guides_ijk": (anchored_guides_ijk + crop_lo).tolist(),
        "guide_straight_segment_lengths_A": guide_straight,
        "raw_segment_lengths_A": raw_segment_lengths,
        "smoothed_geometry_segment_lengths_A": smoothed_geometry_lengths,
        "raw_segment_costs": raw_segment_costs,
        "point_count_mode": point_count_mode,
        "requested_n_points": n_points,
        "rise_per_bp_A": rise_per_bp_A,
        "target_spacing_A_requested": target_spacing_A,
        "effective_target_spacing_A": effective_spacing,
        "estimated_bp_intervals_from_smoothed_length": bp_count_est,
        "target_spacing_A": target_spacing_A_eff,
        "allocated_points_per_segment": per_segment_points,
        "final_path_points": int(len(xyz)),
        "total_path_length_A": final_length,
        "realized_spacing_A": realized_spacing_A,
        "all_segment_spacings_A": all_spacings.tolist(),
        "refined_segment_spacings_A": refined_spacings,
        "median_radius_A": r_eff,
        "volume_based_length_estimate_A": length_est,
        "crop_bbox_min_ijk": crop_lo.tolist(),
        "crop_bbox_max_ijk": (crop_hi - 1).tolist(),
        "pc1_axis": axis.tolist(),
        "pca_center_xyz_A": cen.tolist(),
        "recenter_radius_A": recenter_radius_A,
        "recenter_dt_weight": recenter_dt_weight,
        "recenter_disp_weight": recenter_disp_weight,
        "recenter_iters": recenter_iters,
        "warnings": warnings,
    }
    if report_path:
        _save_report(report_path, report)

    axis_vol = np.zeros_like(mv.data_zyx, dtype=np.float32)
    axis_vol = _draw_polyline(axis_vol, _xyz_to_voxel(xyz, apix, origin))
    write_map(out_mrc, mv, axis_vol)

    points_xyzA = xyz.astype(np.float32)
    points_zyx = _xyz_to_voxel(points_xyzA, apix, origin).astype(np.float32)
    return AxisResult(points_zyx=points_zyx, points_xyzA=points_xyzA)
