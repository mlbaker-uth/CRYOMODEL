"""Phase 2: Cₙ rotational self-correlation scores on the phase-0 grid (per phase-1 axis candidates)."""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
from scipy.ndimage import map_coordinates

from cryomodel.io.mrc import read_map


def rotation_matrix_axis_angle(axis: np.ndarray, theta: float) -> np.ndarray:
    """Right-handed rotation (3×3) by angle ``theta`` (rad) about unit axis through origin."""
    u = axis.astype(np.float64).reshape(3)
    n = float(np.linalg.norm(u))
    if n < 1e-15:
        raise ValueError("Zero-length rotation axis")
    x, y, z = u / n
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


def _xyz_centers_from_indices(
    iz: np.ndarray, iy: np.ndarray, ix: np.ndarray, origin_xyzA: np.ndarray, apix: float
) -> np.ndarray:
    """Shape (N, 3) rows (x, y, z) in Å."""
    ox, oy, oz = float(origin_xyzA[0]), float(origin_xyzA[1]), float(origin_xyzA[2])
    x = ox + (ix.astype(np.float64) + 0.5) * apix
    y = oy + (iy.astype(np.float64) + 0.5) * apix
    z = oz + (iz.astype(np.float64) + 0.5) * apix
    return np.stack([x, y, z], axis=1)


def _sample_after_inverse_rotation(
    data_zyx: np.ndarray,
    iz: np.ndarray,
    iy: np.ndarray,
    ix: np.ndarray,
    origin_xyzA: np.ndarray,
    apix: float,
    com: np.ndarray,
    axis_unit: np.ndarray,
    theta: float,
) -> np.ndarray:
    """
    For voxel centers P, sample original map at Q = com + (P - com) @ R
    where R rotates by +theta about ``axis_unit`` (same convention as rotation_matrix_axis_angle).

    A density rigidly rotated by +theta about com has V_rot(P) = V_orig(Q) with Q = com + R^{-1}(P-com).
    Here R is the forward rotation matrix, so R^{-1} = R^T and Q = com + (P-com) @ R.
    """
    R = rotation_matrix_axis_angle(axis_unit, theta)
    P = _xyz_centers_from_indices(iz, iy, ix, origin_xyzA, apix)
    com = com.reshape(1, 3).astype(np.float64)
    with np.errstate(all="ignore"):
        Q = com + (P - com).astype(np.float64) @ R

    ox, oy, oz = float(origin_xyzA[0]), float(origin_xyzA[1]), float(origin_xyzA[2])
    ix_s = (Q[:, 0] - ox) / apix - 0.5
    iy_s = (Q[:, 1] - oy) / apix - 0.5
    iz_s = (Q[:, 2] - oz) / apix - 0.5

    coords = np.stack([iz_s, iy_s, ix_s], axis=0)
    with np.errstate(all="ignore"):
        out = map_coordinates(data_zyx, coords, order=1, mode="constant", cval=0.0)
    return out.astype(np.float64)


def pearson_correlation(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.size == 0 or b.size == 0:
        return 0.0
    sa, sb = float(np.std(a)), float(np.std(b))
    if sa < 1e-15 and sb < 1e-15:
        return 1.0 if np.allclose(np.mean(a), np.mean(b), rtol=1e-6, atol=1e-6) else 0.0
    a = a - np.mean(a)
    b = b - np.mean(b)
    da, db = np.linalg.norm(a), np.linalg.norm(b)
    if da < 1e-15 or db < 1e-15:
        return 0.0
    return float(np.dot(a, b) / (da * db))


def cn_rotation_correlation(
    data_zyx: np.ndarray,
    iz: np.ndarray,
    iy: np.ndarray,
    ix: np.ndarray,
    origin_xyzA: np.ndarray,
    apix: float,
    com: np.ndarray,
    axis_unit: np.ndarray,
    n_fold: int,
) -> float:
    """Pearson r between values at mask voxels and values sampled after rotation by 2π/n about axis through COM."""
    if n_fold < 2:
        raise ValueError("n_fold must be >= 2")
    theta = 2.0 * math.pi / float(n_fold)
    u = axis_unit.astype(np.float64).reshape(3)
    u = u / np.linalg.norm(u)
    orig = data_zyx[iz, iy, ix].astype(np.float64)
    rot = _sample_after_inverse_rotation(data_zyx, iz, iy, ix, origin_xyzA, apix, com, u, theta)
    return pearson_correlation(orig, rot)


@dataclass
class Phase2Result:
    phase0_json: str
    phase1_json: str
    input_downsample_map: str
    orders: list[int]
    candidates: list[dict[str, Any]]
    global_best: dict[str, Any]
    output_json: str

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_phase2_cn_scores(
    phase0_dir: Path,
    *,
    orders: tuple[int, ...] = (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12),
    max_candidates: Optional[int] = None,
) -> Phase2Result:
    """
    Read phase-0 map + JSON and phase-1 candidate list; for each candidate axis, score each Cₙ
    by rotational self-correlation on voxels above the phase-0 threshold.
    """
    phase0_dir = Path(phase0_dir).expanduser().resolve()
    p0_json = phase0_dir / "symmetry_phase0.json"
    p1_json = phase0_dir / "symmetry_phase1.json"
    p0_map = phase0_dir / "symmetry_phase0_downsample.mrc"
    for p in (p0_json, p1_json, p0_map):
        if not p.is_file():
            raise FileNotFoundError(f"Missing {p}")

    with open(p0_json, encoding="utf-8") as fh:
        p0 = json.load(fh)
    with open(p1_json, encoding="utf-8") as fh:
        p1 = json.load(fh)

    mv = read_map(p0_map)
    data = mv.data_zyx.astype(np.float32)
    apix = float(mv.apix)
    origin = np.asarray(p0["origin_xyzA"], dtype=np.float64)
    thr = float(p0["density_threshold"])
    com = np.array(p0["center_of_mass_angstrom_xyz"], dtype=np.float64)

    sel = data > thr
    iz, iy, ix = np.nonzero(sel)
    if iz.size == 0:
        raise ValueError("No voxels above threshold.")

    cand_list = p1.get("candidates") or []
    if max_candidates is not None:
        cand_list = cand_list[: int(max_candidates)]

    ord_list = [int(n) for n in orders if int(n) >= 2]
    out_cands: list[dict[str, Any]] = []
    best_global = {"candidate_id": -1, "n": 0, "score": -2.0, "source": ""}

    for c in cand_list:
        cid = int(c.get("id", -1))
        u = np.array(c["direction_xyz"], dtype=np.float64)
        src = str(c.get("source", ""))
        scores: dict[str, float] = {}
        best_n, best_s = 2, -2.0
        for n in ord_list:
            try:
                r = cn_rotation_correlation(data, iz, iy, ix, origin, apix, com, u, n)
            except Exception:
                r = -2.0
            scores[str(n)] = float(r)
            if r > best_s:
                best_s, best_n = r, n
        row = {
            "id": cid,
            "source": src,
            "direction_xyz": [float(x) for x in u],
            "cn_scores": scores,
            "best_n": int(best_n),
            "best_score": float(best_s),
        }
        out_cands.append(row)
        if best_s > best_global["score"]:
            best_global = {
                "candidate_id": cid,
                "n": int(best_n),
                "score": float(best_s),
                "source": src,
            }

    out_path = phase0_dir / "symmetry_phase2.json"
    result = Phase2Result(
        phase0_json=str(p0_json),
        phase1_json=str(p1_json),
        input_downsample_map=str(p0_map),
        orders=ord_list,
        candidates=out_cands,
        global_best=best_global,
        output_json=str(out_path),
    )
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(result.to_json_dict(), fh, indent=2)
    return result
