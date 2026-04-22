"""Phase 2D: Dₙ rotational + perpendicular C2 self-correlation scores."""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from cryomodel.io.mrc import read_map

from .phase2_cn import _sample_after_inverse_rotation, cn_rotation_correlation, pearson_correlation


def _orthonormal_perp_frame(u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    u = np.asarray(u, dtype=np.float64).reshape(3)
    u = u / np.linalg.norm(u)
    tmp = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(u, tmp))) > 0.9:
        tmp = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    a = np.cross(u, tmp)
    a = a / np.linalg.norm(a)
    b = np.cross(u, a)
    return a, b


def c2_perpendicular_best_correlation(
    data_zyx: np.ndarray,
    iz: np.ndarray,
    iy: np.ndarray,
    ix: np.ndarray,
    origin_xyzA: np.ndarray,
    apix: float,
    pivot: np.ndarray,
    axis_unit: np.ndarray,
    *,
    inplane_samples: int = 36,
) -> tuple[float, float]:
    """
    Best C2 correlation over axes perpendicular to ``axis_unit``.
    Returns ``(best_score, best_phi_deg)``.
    """
    u = np.asarray(axis_unit, dtype=np.float64).reshape(3)
    u = u / np.linalg.norm(u)
    a, b = _orthonormal_perp_frame(u)
    orig = data_zyx[iz, iy, ix].astype(np.float64)
    n = max(4, int(inplane_samples))
    best_r = -2.0
    best_phi = 0.0
    for k in range(n):
        phi = (2.0 * math.pi * k) / float(n)
        v = math.cos(phi) * a + math.sin(phi) * b
        rot = _sample_after_inverse_rotation(data_zyx, iz, iy, ix, origin_xyzA, apix, pivot, v, math.pi)
        r = pearson_correlation(orig, rot)
        if r > best_r:
            best_r = float(r)
            best_phi = math.degrees(phi)
    return best_r, best_phi


def dn_rotation_correlation(
    data_zyx: np.ndarray,
    iz: np.ndarray,
    iy: np.ndarray,
    ix: np.ndarray,
    origin_xyzA: np.ndarray,
    apix: float,
    pivot: np.ndarray,
    axis_unit: np.ndarray,
    n_fold: int,
    *,
    inplane_samples: int = 36,
) -> dict[str, float]:
    """
    Dₙ score from:
    - Cₙ score around the principal axis
    - best C2 score around any axis perpendicular to the principal axis
    Combined as geometric mean to penalize lopsided evidence.
    """
    cn = cn_rotation_correlation(data_zyx, iz, iy, ix, origin_xyzA, apix, pivot, axis_unit, n_fold)
    c2, phi_deg = c2_perpendicular_best_correlation(
        data_zyx,
        iz,
        iy,
        ix,
        origin_xyzA,
        apix,
        pivot,
        axis_unit,
        inplane_samples=inplane_samples,
    )
    cn_c = max(-1.0, min(1.0, float(cn)))
    c2_c = max(-1.0, min(1.0, float(c2)))
    # map [-1,1] -> [0,1], combine, map back
    dn = 2.0 * math.sqrt(max(0.0, 0.5 * (cn_c + 1.0)) * max(0.0, 0.5 * (c2_c + 1.0))) - 1.0
    return {"dn_score": float(dn), "cn_component": float(cn), "c2_perp_component": float(c2), "best_phi_deg": float(phi_deg)}


@dataclass
class Phase2DResult:
    phase0_json: str
    phase1_json: str
    input_downsample_map: str
    orders: list[int]
    inplane_samples: int
    candidates: list[dict[str, Any]]
    global_best: dict[str, Any]
    output_json: str

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_phase2_dn_scores(
    phase0_dir: Path,
    *,
    orders: tuple[int, ...] = (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12),
    max_candidates: Optional[int] = None,
    inplane_samples: int = 36,
) -> Phase2DResult:
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
    pivot = np.array(p0["center_of_mass_angstrom_xyz"], dtype=np.float64)

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
        dn_scores: dict[str, float] = {}
        cn_comp: dict[str, float] = {}
        c2_comp: dict[str, float] = {}
        phi_best: dict[str, float] = {}
        best_n, best_s = 2, -2.0
        for n in ord_list:
            try:
                d = dn_rotation_correlation(
                    data,
                    iz,
                    iy,
                    ix,
                    origin,
                    apix,
                    pivot,
                    u,
                    n,
                    inplane_samples=inplane_samples,
                )
                r = float(d["dn_score"])
                dn_scores[str(n)] = r
                cn_comp[str(n)] = float(d["cn_component"])
                c2_comp[str(n)] = float(d["c2_perp_component"])
                phi_best[str(n)] = float(d["best_phi_deg"])
            except Exception:
                r = -2.0
                dn_scores[str(n)] = r
                cn_comp[str(n)] = -2.0
                c2_comp[str(n)] = -2.0
                phi_best[str(n)] = 0.0
            if r > best_s:
                best_s, best_n = r, n
        row = {
            "id": cid,
            "source": src,
            "direction_xyz": [float(x) for x in u],
            "dn_scores": dn_scores,
            "cn_component_scores": cn_comp,
            "c2_perp_component_scores": c2_comp,
            "best_phi_deg_by_n": phi_best,
            "best_n": int(best_n),
            "best_score": float(best_s),
        }
        out_cands.append(row)
        if best_s > best_global["score"]:
            best_global = {"candidate_id": cid, "n": int(best_n), "score": float(best_s), "source": src}

    out_path = phase0_dir / "symmetry_phase2d.json"
    result = Phase2DResult(
        phase0_json=str(p0_json),
        phase1_json=str(p1_json),
        input_downsample_map=str(p0_map),
        orders=ord_list,
        inplane_samples=int(inplane_samples),
        candidates=out_cands,
        global_best=best_global,
        output_json=str(out_path),
    )
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(result.to_json_dict(), fh, indent=2)
    return result

