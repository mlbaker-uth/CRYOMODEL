"""Cₙ rotational self-correlation in cylindrical shells about the chosen symmetry axis."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np

from cryomodel.io.mrc import read_map

from .phase2_cn import _xyz_centers_from_indices, cn_rotation_correlation
from .phase2_dn import dn_rotation_correlation
from .phase4_axis_pdb import load_symmetry_axis_geometry


@dataclass
class MultishellResult:
    phase0_json: str
    axis_source: str
    pivot_xyz: list[float]
    axis_xyz: list[float]
    orders: list[int]
    n_shells: int
    radius_cap_A: float
    shells: list[dict[str, Any]]
    output_json: str

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_multishell_cn_scores(
    phase0_dir: Path,
    *,
    family: Literal["c", "d"] = "c",
    orders: tuple[int, ...] = (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12),
    n_shells: int = 8,
    max_radius_A: Optional[float] = None,
    radius_percentile_cap: float = 99.5,
    min_voxels_per_shell: int = 64,
    refinement_index: int = 0,
    prefer_phase3: bool = True,
) -> MultishellResult:
    """
    For voxels above the phase-0 threshold on the downsampled map, compute cylindrical radius ρ
    from the symmetry axis through ``pivot`` with direction ``axis``. Split ``ρ ∈ [0, r_cap]`` into
    ``n_shells`` equal annuli and score each Cₙ order per shell via the same Pearson correlation
    as phase 2.

    Axis and pivot default to phase-3 refinement row ``refinement_index`` when
    ``symmetry_phase3.json`` exists; otherwise phase-2 global best with phase-0 COM.

    ``r_cap`` is ``max_radius_A`` if set, else the given percentile of ρ among masked voxels.
    """
    phase0_dir = Path(phase0_dir).expanduser().resolve()
    p0_json = phase0_dir / "symmetry_phase0.json"
    p0_map = phase0_dir / "symmetry_phase0_downsample.mrc"
    if not p0_json.is_file():
        raise FileNotFoundError(f"Missing {p0_json}")
    if not p0_map.is_file():
        raise FileNotFoundError(f"Missing {p0_map}")

    with open(p0_json, encoding="utf-8") as fh:
        p0 = json.load(fh)

    mv = read_map(p0_map)
    data = mv.data_zyx.astype(np.float32)
    apix = float(mv.apix)
    origin = np.asarray(p0["origin_xyzA"], dtype=np.float64)
    thr = float(p0["density_threshold"])

    sel = data > thr
    iz, iy, ix = np.nonzero(sel)
    if iz.size == 0:
        raise ValueError("No voxels above threshold.")

    u, pivot, n_hint, prov, extra = load_symmetry_axis_geometry(
        phase0_dir,
        family=family,
        refinement_index=refinement_index,
        prefer_phase3=prefer_phase3,
    )
    u = u.astype(np.float64)
    nu = float(np.linalg.norm(u))
    if (not np.isfinite(nu)) or nu < 1e-12:
        raise ValueError("Invalid symmetry axis direction from phase2/phase3.")
    u = u / nu
    pivot = pivot.astype(np.float64)

    P = _xyz_centers_from_indices(iz, iy, ix, origin, apix).astype(np.float64)
    v = P - pivot.reshape(1, 3)
    # Some BLAS builds emit spurious matmul warnings on large arrays; values are finite here.
    with np.errstate(all="ignore"):
        s_along = (v @ u).reshape(-1, 1)
    s_proj = s_along * u.reshape(1, 3)
    rho = np.linalg.norm(v - s_proj, axis=1)
    valid = np.isfinite(rho)
    if not np.all(valid):
        iz = iz[valid]
        iy = iy[valid]
        ix = ix[valid]
        rho = rho[valid]
    if rho.size == 0:
        raise ValueError("No finite cylindrical radii for multishell scoring.")

    if max_radius_A is not None:
        r_cap = float(max_radius_A)
    else:
        r_cap = float(np.percentile(rho, float(radius_percentile_cap)))
    r_cap = max(r_cap, float(apix) * 0.5)

    n_shells = max(1, int(n_shells))
    edges = np.linspace(0.0, r_cap, n_shells + 1, dtype=np.float64)
    ord_list = [int(n) for n in orders if int(n) >= 2]

    shells: list[dict[str, Any]] = []
    for k in range(n_shells):
        r0, r1 = float(edges[k]), float(edges[k + 1])
        if k == n_shells - 1:
            mask = (rho >= r0) & (rho <= r1 + 1e-9)
        else:
            mask = (rho >= r0) & (rho < r1)
        idx = np.nonzero(mask)[0]
        row: dict[str, Any] = {
            "shell_index": k,
            "r_inner_A": r0,
            "r_outer_A": r1,
            "n_voxels": int(idx.size),
        }
        if idx.size < int(min_voxels_per_shell):
            row["skipped"] = True
            row["skip_reason"] = "too_few_voxels"
            shells.append(row)
            continue

        iz_s = iz[idx]
        iy_s = iy[idx]
        ix_s = ix[idx]
        scores: dict[str, float] = {}
        best_n, best_r = 2, -2.0
        for n in ord_list:
            try:
                if family == "d":
                    r = float(
                        dn_rotation_correlation(
                            data, iz_s, iy_s, ix_s, origin, apix, pivot, u, n, inplane_samples=36
                        )["dn_score"]
                    )
                else:
                    r = cn_rotation_correlation(data, iz_s, iy_s, ix_s, origin, apix, pivot, u, n)
            except Exception:
                r = -2.0
            scores[str(n)] = float(r)
            if r > best_r:
                best_r, best_n = r, n
        row["cn_scores"] = scores
        row["best_n"] = int(best_n)
        row["best_score"] = float(best_r)
        row["skipped"] = False
        shells.append(row)

    out_path = phase0_dir / "symmetry_multishell.json"
    result = MultishellResult(
        phase0_json=str(p0_json),
        axis_source=str(prov),
        pivot_xyz=[float(x) for x in pivot],
        axis_xyz=[float(x) for x in u],
        orders=ord_list,
        n_shells=n_shells,
        radius_cap_A=float(r_cap),
        shells=shells,
        output_json=str(out_path),
    )
    payload = result.to_json_dict()
    payload["global_n_hint"] = int(n_hint)
    payload["extra"] = {**extra, "radius_percentile_cap": float(radius_percentile_cap)}
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return result
