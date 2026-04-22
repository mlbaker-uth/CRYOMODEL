"""Helical symmetry finder: estimate axis, rise, and twist from map self-correlation."""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
from scipy.ndimage import map_coordinates

from cryomodel.io.mrc import read_map
from cryomodel.symmetry.preprocess import _voxel_centers_xyz, weighted_principal_axes


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        raise ValueError("Zero-length axis.")
    return v / n


def rotation_matrix_axis_angle(axis: np.ndarray, theta: float) -> np.ndarray:
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


def pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.size == 0 or b.size == 0:
        return 0.0
    a = a - np.mean(a)
    b = b - np.mean(b)
    da, db = np.linalg.norm(a), np.linalg.norm(b)
    if da < 1e-15 or db < 1e-15:
        return 0.0
    return float(np.dot(a, b) / (da * db))


def screw_correlation(
    data_zyx: np.ndarray,
    iz: np.ndarray,
    iy: np.ndarray,
    ix: np.ndarray,
    origin_xyzA: np.ndarray,
    apix: float,
    pivot_xyz: np.ndarray,
    axis_unit: np.ndarray,
    twist_deg: float,
    rise_A: float,
) -> float:
    """
    Correlation for one helical step (twist + rise along axis):
      V_step(P) = V_orig( pivot + R^{-1}(P - pivot - rise*u) )
    Row-vector form: Q = pivot + ((P - pivot - rise*u) @ R), where R is +twist.
    """
    u = _unit(axis_unit)
    theta = math.radians(float(twist_deg))
    R = rotation_matrix_axis_angle(u, theta)
    P = _voxel_centers_xyz(iz, iy, ix, origin_xyzA, apix).astype(np.float64)
    p0 = np.asarray(pivot_xyz, dtype=np.float64).reshape(1, 3)
    shift = float(rise_A) * u.reshape(1, 3)
    with np.errstate(all="ignore"):
        Q = p0 + ((P - p0 - shift) @ R)
    ox, oy, oz = (float(origin_xyzA[0]), float(origin_xyzA[1]), float(origin_xyzA[2]))
    ix_s = (Q[:, 0] - ox) / apix - 0.5
    iy_s = (Q[:, 1] - oy) / apix - 0.5
    iz_s = (Q[:, 2] - oz) / apix - 0.5
    coords = np.stack([iz_s, iy_s, ix_s], axis=0)
    with np.errstate(all="ignore"):
        rot = map_coordinates(data_zyx, coords, order=1, mode="constant", cval=0.0).astype(np.float64)
    orig = data_zyx[iz, iy, ix].astype(np.float64)
    return pearson_corr(orig, rot)


def _evaluate_grid_for_axis(
    data: np.ndarray,
    iz: np.ndarray,
    iy: np.ndarray,
    ix: np.ndarray,
    origin: np.ndarray,
    apix: float,
    pivot: np.ndarray,
    axis_u: np.ndarray,
    twist_vals: np.ndarray,
    rise_vals: np.ndarray,
) -> np.ndarray:
    scores = np.zeros((rise_vals.size, twist_vals.size), dtype=np.float64)
    for i, ri in enumerate(rise_vals):
        for j, tw in enumerate(twist_vals):
            if abs(float(tw)) < 1e-9:
                scores[i, j] = -2.0
                continue
            scores[i, j] = screw_correlation(data, iz, iy, ix, origin, apix, pivot, axis_u, float(tw), float(ri))
    return scores


def _write_helical_heatmap_png(
    out_png: Path,
    scores_rise_twist: np.ndarray,
    twist_vals: np.ndarray,
    rise_vals: np.ndarray,
    *,
    best_twist_deg: float,
    best_rise_A: float,
    axis_source: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_png = Path(out_png).expanduser().resolve()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 5.0), dpi=150)
    extent = [float(twist_vals.min()), float(twist_vals.max()), float(rise_vals.min()), float(rise_vals.max())]
    im = ax.imshow(scores_rise_twist, origin="lower", aspect="auto", extent=extent, cmap="viridis")
    ax.scatter([best_twist_deg], [best_rise_A], marker="x", s=70, c="red", linewidths=1.8, label="best")
    ax.set_xlabel("Twist (deg/subunit)")
    ax.set_ylabel("Rise (Å/subunit)")
    ax.set_title(f"Helical score heatmap ({axis_source})")
    ax.legend(loc="best")
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("Pearson self-correlation")
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


@dataclass
class HelicalFindResult:
    input_map: str
    output_json: str
    apix: float
    threshold: float
    pivot_xyz: list[float]
    axis_xyz: list[float]
    best_twist_deg: float
    best_rise_A: float
    best_score: float
    heatmap_png: Optional[str]
    candidates: list[dict[str, Any]]

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_helical_find(
    map_path: Path,
    out_dir: Path,
    *,
    density_threshold: Optional[float] = None,
    density_percentile: float = 90.0,
    axis_mode: str = "cardinal_pca",
    twist_min_deg: float = -20.0,
    twist_max_deg: float = 20.0,
    twist_step_deg: float = 0.5,
    rise_min_A: float = 2.0,
    rise_max_A: float = 8.0,
    rise_step_A: float = 0.2,
    max_voxels_score: int = 200_000,
    seed: int = 0,
    refine: bool = True,
    refine_iters: int = 2,
    write_heatmap: bool = True,
) -> HelicalFindResult:
    map_path = Path(map_path).expanduser().resolve()
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    mv = read_map(map_path)
    data = mv.data_zyx.astype(np.float32)
    apix = float(mv.apix)
    origin = np.asarray(mv.origin_xyzA, dtype=np.float64)

    flat = data.ravel()
    pos = flat[flat > 0]
    if pos.size == 0:
        pos = flat
    thr = float(density_threshold) if density_threshold is not None else float(np.percentile(pos, float(density_percentile)))
    iz, iy, ix = np.nonzero(data > thr)
    if iz.size == 0:
        raise ValueError("No voxels above threshold.")

    if iz.size > int(max_voxels_score):
        rng = np.random.default_rng(int(seed))
        idx = rng.choice(iz.size, size=int(max_voxels_score), replace=False)
        iz, iy, ix = iz[idx], iy[idx], ix[idx]

    w = np.maximum(data[iz, iy, ix].astype(np.float64), 0.0)
    coords = _voxel_centers_xyz(iz, iy, ix, origin, apix).astype(np.float64)
    com, _evals, axes = weighted_principal_axes(coords, w)
    pca_axis = _unit(np.array(axes[0], dtype=np.float64))

    candidates: list[tuple[np.ndarray, str]] = []
    if axis_mode in ("cardinal", "cardinal_pca"):
        candidates.extend(
            [
                (np.array([1.0, 0.0, 0.0], dtype=np.float64), "cardinal_x"),
                (np.array([0.0, 1.0, 0.0], dtype=np.float64), "cardinal_y"),
                (np.array([0.0, 0.0, 1.0], dtype=np.float64), "cardinal_z"),
            ]
        )
    if axis_mode in ("pca", "cardinal_pca"):
        candidates.append((pca_axis, "pca_primary"))

    twist_vals = np.arange(float(twist_min_deg), float(twist_max_deg) + 1e-9, float(twist_step_deg), dtype=np.float64)
    rise_vals = np.arange(float(rise_min_A), float(rise_max_A) + 1e-9, float(rise_step_A), dtype=np.float64)
    if twist_vals.size == 0 or rise_vals.size == 0:
        raise ValueError("Empty rise/twist grid.")

    rows: list[dict[str, Any]] = []
    best = {"score": -2.0, "axis": [0.0, 0.0, 1.0], "twist": 0.0, "rise": 0.0, "source": ""}
    best_axis_source = ""
    best_axis_u = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    for u, src in candidates:
        u = _unit(u)
        coarse = _evaluate_grid_for_axis(data, iz, iy, ix, origin, apix, com, u, twist_vals, rise_vals)
        k = int(np.argmax(coarse))
        i, j = divmod(k, coarse.shape[1])
        best_local = {"score": float(coarse[i, j]), "twist": float(twist_vals[j]), "rise": float(rise_vals[i])}

        if refine:
            tw0, ri0 = best_local["twist"], best_local["rise"]
            tw_fine = np.arange(tw0 - twist_step_deg, tw0 + twist_step_deg + 1e-9, max(0.1, twist_step_deg / 5.0))
            ri_fine = np.arange(ri0 - rise_step_A, ri0 + rise_step_A + 1e-9, max(0.05, rise_step_A / 4.0))
            fine = _evaluate_grid_for_axis(data, iz, iy, ix, origin, apix, com, u, tw_fine, ri_fine)
            kf = int(np.argmax(fine))
            fi, fj = divmod(kf, fine.shape[1])
            best_local = {"score": float(fine[fi, fj]), "twist": float(tw_fine[fj]), "rise": float(ri_fine[fi])}

        row = {
            "axis_source": src,
            "axis_xyz": [float(x) for x in u],
            "best_twist_deg": float(best_local["twist"]),
            "best_rise_A": float(best_local["rise"]),
            "best_score": float(best_local["score"]),
        }
        rows.append(row)
        if best_local["score"] > best["score"]:
            best = {
                "score": float(best_local["score"]),
                "axis": [float(x) for x in u],
                "twist": float(best_local["twist"]),
                "rise": float(best_local["rise"]),
                "source": src,
            }
            best_axis_source = src
            best_axis_u = u.copy()

    refine_history: list[dict[str, float]] = []
    if refine and refine_iters > 0:
        tw_c = float(best["twist"])
        ri_c = float(best["rise"])
        tw_step = max(0.05, float(twist_step_deg) / 4.0)
        ri_step = max(0.02, float(rise_step_A) / 4.0)
        tw_half = max(float(twist_step_deg), 0.2)
        ri_half = max(float(rise_step_A), 0.08)
        for _ in range(int(refine_iters)):
            tw_vals = np.arange(tw_c - tw_half, tw_c + tw_half + 1e-12, tw_step, dtype=np.float64)
            ri_vals = np.arange(max(0.05, ri_c - ri_half), ri_c + ri_half + 1e-12, ri_step, dtype=np.float64)
            if tw_vals.size == 0 or ri_vals.size == 0:
                break
            grid = _evaluate_grid_for_axis(data, iz, iy, ix, origin, apix, com, best_axis_u, tw_vals, ri_vals)
            k = int(np.argmax(grid))
            i, j = divmod(k, grid.shape[1])
            sc = float(grid[i, j])
            tw_c = float(tw_vals[j])
            ri_c = float(ri_vals[i])
            refine_history.append({"twist_deg": tw_c, "rise_A": ri_c, "score": sc})
            if sc > float(best["score"]):
                best["score"] = sc
                best["twist"] = tw_c
                best["rise"] = ri_c
            tw_half *= 0.5
            ri_half *= 0.5

    heatmap_png: Optional[str] = None
    if write_heatmap:
        hm = _evaluate_grid_for_axis(data, iz, iy, ix, origin, apix, com, best_axis_u, twist_vals, rise_vals)
        hm_path = out_dir / "helical_score_heatmap.png"
        _write_helical_heatmap_png(
            hm_path,
            hm,
            twist_vals,
            rise_vals,
            best_twist_deg=float(best["twist"]),
            best_rise_A=float(best["rise"]),
            axis_source=best_axis_source or str(best.get("source", "")),
        )
        heatmap_png = str(hm_path)

    out_json = out_dir / "helical_find.json"
    result = HelicalFindResult(
        input_map=str(map_path),
        output_json=str(out_json),
        apix=apix,
        threshold=thr,
        pivot_xyz=[float(x) for x in com],
        axis_xyz=[float(x) for x in best["axis"]],
        best_twist_deg=float(best["twist"]),
        best_rise_A=float(best["rise"]),
        best_score=float(best["score"]),
        heatmap_png=heatmap_png,
        candidates=rows,
    )
    payload = result.to_json_dict()
    payload["best_axis_source"] = str(best["source"])
    payload["grid"] = {
        "twist_min_deg": float(twist_min_deg),
        "twist_max_deg": float(twist_max_deg),
        "twist_step_deg": float(twist_step_deg),
        "rise_min_A": float(rise_min_A),
        "rise_max_A": float(rise_max_A),
        "rise_step_A": float(rise_step_A),
    }
    payload["refine_history"] = refine_history
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return result

