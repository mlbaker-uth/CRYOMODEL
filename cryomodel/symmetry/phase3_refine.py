"""Phase 3: local refinement of Cₙ axis and pivot for top phase-2 hypotheses."""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
from scipy.optimize import minimize

from cryomodel.io.mrc import read_map

from .phase2_cn import cn_rotation_correlation


def _orthonormal_frame_perp_to_u(u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(a, b)`` orthonormal, both ⟂ unit vector ``u``."""
    u = np.asarray(u, dtype=np.float64).reshape(3)
    u = u / np.linalg.norm(u)
    tmp = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(u, tmp))) > 0.9:
        tmp = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    a = np.cross(u, tmp)
    a = a / np.linalg.norm(a)
    b = np.cross(u, a)
    return a, b


def _decode_axis_pivot(
    x: np.ndarray, u0: np.ndarray, c0: np.ndarray, a0: np.ndarray, b0: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Five parameters: tilt of axis in the u0 tangent plane (α, β), shift along refined axis (s),
    and perpendicular offsets (d1, d2) in the plane ⟂ refined axis.
    """
    alpha, beta, s, d1, d2 = (float(v) for v in x)
    u_tilt = u0 + alpha * a0 + beta * b0
    nu = float(np.linalg.norm(u_tilt))
    if nu < 1e-12:
        u = u0.copy()
    else:
        u = u_tilt / nu
    a1 = a0 - np.dot(a0, u) * u
    na1 = float(np.linalg.norm(a1))
    if na1 < 1e-9:
        tmp = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(float(np.dot(u, tmp))) > 0.9:
            tmp = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        a1 = np.cross(u, tmp)
        na1 = float(np.linalg.norm(a1))
    a1 = a1 / na1
    b1 = np.cross(u, a1)
    c = c0 + s * u + d1 * a1 + d2 * b1
    return u.astype(np.float64), c.astype(np.float64)


def refine_cn_axis_pivot(
    data_zyx: np.ndarray,
    iz: np.ndarray,
    iy: np.ndarray,
    ix: np.ndarray,
    origin_xyzA: np.ndarray,
    apix: float,
    u0: np.ndarray,
    c0: np.ndarray,
    n_fold: int,
    *,
    max_tilt_deg: float = 5.0,
    max_shift_along_axis_A: float = 10.0,
    max_shift_perp_A: float = 6.0,
    maxiter: int = 80,
) -> dict[str, Any]:
    """
    Maximize Pearson rotational self-correlation over a small neighborhood of axis direction
    and pivot point. Returns dict with refined geometry, scores, and optimizer metadata.
    """
    u0 = np.asarray(u0, dtype=np.float64).reshape(3)
    u0 = u0 / np.linalg.norm(u0)
    c0 = np.asarray(c0, dtype=np.float64).reshape(3)
    a0, b0 = _orthonormal_frame_perp_to_u(u0)

    r0 = cn_rotation_correlation(data_zyx, iz, iy, ix, origin_xyzA, apix, c0, u0, n_fold)

    tan_b = math.tan(math.radians(max_tilt_deg))
    b_tilt = tan_b
    bounds = [
        (-b_tilt, b_tilt),
        (-b_tilt, b_tilt),
        (-max_shift_along_axis_A, max_shift_along_axis_A),
        (-max_shift_perp_A, max_shift_perp_A),
        (-max_shift_perp_A, max_shift_perp_A),
    ]

    def objective(vec: np.ndarray) -> float:
        u, c = _decode_axis_pivot(vec, u0, c0, a0, b0)
        r = cn_rotation_correlation(data_zyx, iz, iy, ix, origin_xyzA, apix, c, u, n_fold)
        return float(-r)

    x0 = np.zeros(5, dtype=np.float64)
    res = minimize(
        objective,
        x0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": int(maxiter), "ftol": 1e-10},
    )
    u_opt, c_opt = _decode_axis_pivot(res.x, u0, c0, a0, b0)
    r_opt = cn_rotation_correlation(data_zyx, iz, iy, ix, origin_xyzA, apix, c_opt, u_opt, n_fold)

    return {
        "phase2_axis_xyz": [float(v) for v in u0],
        "phase2_pivot_xyz": [float(v) for v in c0],
        "phase2_score": float(r0),
        "refined_axis_xyz": [float(v) for v in u_opt],
        "refined_pivot_xyz": [float(v) for v in c_opt],
        "refined_score": float(r_opt),
        "score_delta": float(r_opt - r0),
        "optimizer_success": bool(res.success),
        "optimizer_message": str(res.message),
        "optimizer_nit": int(res.nit),
        "param_vector": [float(v) for v in res.x],
    }


@dataclass
class Phase3Result:
    phase0_json: str
    phase2_json: str
    input_downsample_map: str
    top_hypotheses: int
    refinements: list[dict[str, Any]]
    output_json: str

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_phase3_refine(
    phase0_dir: Path,
    *,
    top_hypotheses: int = 3,
    max_tilt_deg: float = 5.0,
    max_shift_along_axis_A: float = 10.0,
    max_shift_perp_A: float = 6.0,
    maxiter: int = 80,
) -> Phase3Result:
    """
    Refine the top ``top_hypotheses`` phase-2 candidates (by ``best_score``) with local
    optimization of axis direction and pivot. Requires phase 0–2 artifacts in ``phase0_dir``.
    """
    phase0_dir = Path(phase0_dir).expanduser().resolve()
    p0_json = phase0_dir / "symmetry_phase0.json"
    p2_json = phase0_dir / "symmetry_phase2.json"
    p0_map = phase0_dir / "symmetry_phase0_downsample.mrc"
    for p in (p0_json, p2_json, p0_map):
        if not p.is_file():
            raise FileNotFoundError(f"Missing {p}")

    with open(p0_json, encoding="utf-8") as fh:
        p0 = json.load(fh)
    with open(p2_json, encoding="utf-8") as fh:
        p2 = json.load(fh)

    mv = read_map(p0_map)
    data = mv.data_zyx.astype(np.float32)
    apix = float(mv.apix)
    origin = np.asarray(p0["origin_xyzA"], dtype=np.float64)
    thr = float(p0["density_threshold"])
    com0 = np.array(p0["center_of_mass_angstrom_xyz"], dtype=np.float64)

    sel = data > thr
    iz, iy, ix = np.nonzero(sel)
    if iz.size == 0:
        raise ValueError("No voxels above threshold.")

    cands = list(p2.get("candidates") or [])
    cands.sort(key=lambda c: float(c.get("best_score", -99.0)), reverse=True)
    k = max(1, int(top_hypotheses))
    picked = cands[:k]

    refinements: list[dict[str, Any]] = []
    for c in picked:
        cid = int(c.get("id", -1))
        src = str(c.get("source", ""))
        u = np.array(c["direction_xyz"], dtype=np.float64)
        n_fold = int(c.get("best_n", 2))
        base_score = float(c.get("best_score", -2.0))
        row: dict[str, Any] = {
            "phase2_candidate_id": cid,
            "source": src,
            "n": n_fold,
            "phase2_best_score": base_score,
        }
        try:
            ref = refine_cn_axis_pivot(
                data,
                iz,
                iy,
                ix,
                origin,
                apix,
                u,
                com0,
                n_fold,
                max_tilt_deg=max_tilt_deg,
                max_shift_along_axis_A=max_shift_along_axis_A,
                max_shift_perp_A=max_shift_perp_A,
                maxiter=maxiter,
            )
            row.update(ref)
        except Exception as exc:
            row["error"] = str(exc)
            row["refined_score"] = base_score
            row["score_delta"] = 0.0
        refinements.append(row)

    refinements.sort(key=lambda r: float(r.get("refined_score", r.get("phase2_best_score", -99.0))), reverse=True)
    out_path = phase0_dir / "symmetry_phase3.json"
    result = Phase3Result(
        phase0_json=str(p0_json),
        phase2_json=str(p2_json),
        input_downsample_map=str(p0_map),
        top_hypotheses=k,
        refinements=refinements,
        output_json=str(out_path),
    )
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(result.to_json_dict(), fh, indent=2)
    return result
