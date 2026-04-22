"""Phase 3D: local refinement for Dₙ hypotheses."""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize

from cryomodel.io.mrc import read_map

from .phase2_dn import dn_rotation_correlation
from .phase3_refine import _decode_axis_pivot, _orthonormal_frame_perp_to_u


def refine_dn_axis_pivot(
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
    inplane_samples: int = 36,
    max_tilt_deg: float = 5.0,
    max_shift_along_axis_A: float = 10.0,
    max_shift_perp_A: float = 6.0,
    maxiter: int = 80,
) -> dict[str, Any]:
    u0 = np.asarray(u0, dtype=np.float64).reshape(3)
    u0 = u0 / np.linalg.norm(u0)
    c0 = np.asarray(c0, dtype=np.float64).reshape(3)
    a0, b0 = _orthonormal_frame_perp_to_u(u0)

    d0 = dn_rotation_correlation(
        data_zyx, iz, iy, ix, origin_xyzA, apix, c0, u0, n_fold, inplane_samples=inplane_samples
    )
    s0 = float(d0["dn_score"])

    tan_b = math.tan(math.radians(max_tilt_deg))
    bounds = [
        (-tan_b, tan_b),
        (-tan_b, tan_b),
        (-max_shift_along_axis_A, max_shift_along_axis_A),
        (-max_shift_perp_A, max_shift_perp_A),
        (-max_shift_perp_A, max_shift_perp_A),
    ]

    def objective(vec: np.ndarray) -> float:
        u, c = _decode_axis_pivot(vec, u0, c0, a0, b0)
        d = dn_rotation_correlation(
            data_zyx, iz, iy, ix, origin_xyzA, apix, c, u, n_fold, inplane_samples=inplane_samples
        )
        return float(-d["dn_score"])

    x0 = np.zeros(5, dtype=np.float64)
    res = minimize(objective, x0, method="L-BFGS-B", bounds=bounds, options={"maxiter": int(maxiter), "ftol": 1e-10})
    u_opt, c_opt = _decode_axis_pivot(res.x, u0, c0, a0, b0)
    d_opt = dn_rotation_correlation(
        data_zyx, iz, iy, ix, origin_xyzA, apix, c_opt, u_opt, n_fold, inplane_samples=inplane_samples
    )
    s_opt = float(d_opt["dn_score"])

    return {
        "phase2d_axis_xyz": [float(v) for v in u0],
        "phase2d_pivot_xyz": [float(v) for v in c0],
        "phase2d_score": s0,
        "phase2d_cn_component": float(d0["cn_component"]),
        "phase2d_c2_perp_component": float(d0["c2_perp_component"]),
        "refined_axis_xyz": [float(v) for v in u_opt],
        "refined_pivot_xyz": [float(v) for v in c_opt],
        "refined_score": s_opt,
        "refined_cn_component": float(d_opt["cn_component"]),
        "refined_c2_perp_component": float(d_opt["c2_perp_component"]),
        "score_delta": float(s_opt - s0),
        "optimizer_success": bool(res.success),
        "optimizer_message": str(res.message),
        "optimizer_nit": int(res.nit),
        "param_vector": [float(v) for v in res.x],
    }


@dataclass
class Phase3DResult:
    phase0_json: str
    phase2d_json: str
    input_downsample_map: str
    top_hypotheses: int
    inplane_samples: int
    refinements: list[dict[str, Any]]
    output_json: str

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_phase3d_refine(
    phase0_dir: Path,
    *,
    top_hypotheses: int = 3,
    inplane_samples: int = 36,
    max_tilt_deg: float = 5.0,
    max_shift_along_axis_A: float = 10.0,
    max_shift_perp_A: float = 6.0,
    maxiter: int = 80,
) -> Phase3DResult:
    phase0_dir = Path(phase0_dir).expanduser().resolve()
    p0_json = phase0_dir / "symmetry_phase0.json"
    p2d_json = phase0_dir / "symmetry_phase2d.json"
    p0_map = phase0_dir / "symmetry_phase0_downsample.mrc"
    for p in (p0_json, p2d_json, p0_map):
        if not p.is_file():
            raise FileNotFoundError(f"Missing {p}")

    with open(p0_json, encoding="utf-8") as fh:
        p0 = json.load(fh)
    with open(p2d_json, encoding="utf-8") as fh:
        p2d = json.load(fh)

    mv = read_map(p0_map)
    data = mv.data_zyx.astype(np.float32)
    apix = float(mv.apix)
    origin = np.asarray(p0["origin_xyzA"], dtype=np.float64)
    thr = float(p0["density_threshold"])
    com0 = np.array(p0["center_of_mass_angstrom_xyz"], dtype=np.float64)
    iz, iy, ix = np.nonzero(data > thr)
    if iz.size == 0:
        raise ValueError("No voxels above threshold.")

    cands = list(p2d.get("candidates") or [])
    cands.sort(key=lambda c: float(c.get("best_score", -99.0)), reverse=True)
    picked = cands[: max(1, int(top_hypotheses))]

    refinements: list[dict[str, Any]] = []
    for c in picked:
        cid = int(c.get("id", -1))
        u = np.array(c["direction_xyz"], dtype=np.float64)
        n_fold = int(c.get("best_n", 2))
        row: dict[str, Any] = {
            "phase2d_candidate_id": cid,
            "source": str(c.get("source", "")),
            "n": n_fold,
            "phase2d_best_score": float(c.get("best_score", -2.0)),
        }
        try:
            row.update(
                refine_dn_axis_pivot(
                    data,
                    iz,
                    iy,
                    ix,
                    origin,
                    apix,
                    u,
                    com0,
                    n_fold,
                    inplane_samples=inplane_samples,
                    max_tilt_deg=max_tilt_deg,
                    max_shift_along_axis_A=max_shift_along_axis_A,
                    max_shift_perp_A=max_shift_perp_A,
                    maxiter=maxiter,
                )
            )
        except Exception as exc:
            row["error"] = str(exc)
            row["refined_score"] = row["phase2d_best_score"]
            row["score_delta"] = 0.0
        refinements.append(row)

    refinements.sort(key=lambda r: float(r.get("refined_score", -99.0)), reverse=True)
    out_path = phase0_dir / "symmetry_phase3d.json"
    result = Phase3DResult(
        phase0_json=str(p0_json),
        phase2d_json=str(p2d_json),
        input_downsample_map=str(p0_map),
        top_hypotheses=max(1, int(top_hypotheses)),
        inplane_samples=int(inplane_samples),
        refinements=refinements,
        output_json=str(out_path),
    )
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(result.to_json_dict(), fh, indent=2)
    return result

