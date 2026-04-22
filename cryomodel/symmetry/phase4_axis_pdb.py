"""Phase 4: write a CA trace PDB for the Cₙ symmetry axis in the full-map coordinate frame."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np

from cryomodel.io.mrc import read_map


def ray_aabb_intersect_t(
    origin: np.ndarray,
    direction: np.ndarray,
    bmin: np.ndarray,
    bmax: np.ndarray,
) -> Optional[tuple[float, float]]:
    """
    Intersect infinite ray ``origin + t * direction`` with axis-aligned box [bmin, bmax].
    Returns ``(t_min, t_max)`` along the ray inside the box, or ``None`` if miss.
    """
    o = np.asarray(origin, dtype=np.float64).reshape(3)
    d = np.asarray(direction, dtype=np.float64).reshape(3)
    dn = float(np.linalg.norm(d))
    if dn < 1e-15:
        return None
    d = d / dn
    bmin = np.asarray(bmin, dtype=np.float64).reshape(3)
    bmax = np.asarray(bmax, dtype=np.float64).reshape(3)
    t_near = -np.inf
    t_far = np.inf
    for i in range(3):
        if abs(d[i]) < 1e-14:
            if o[i] < bmin[i] - 1e-9 or o[i] > bmax[i] + 1e-9:
                return None
            continue
        inv = 1.0 / d[i]
        t1 = (bmin[i] - o[i]) * inv
        t2 = (bmax[i] - o[i]) * inv
        lo, hi = (t1, t2) if t1 <= t2 else (t2, t1)
        t_near = max(t_near, lo)
        t_far = min(t_far, hi)
        if t_near > t_far + 1e-12:
            return None
    if t_far < t_near:
        return None
    return (float(t_near), float(t_far))


def sample_axis_parameters(
    t_near: float,
    t_far: float,
    *,
    step_along_axis_A: float,
) -> np.ndarray:
    """Monotonic ``t`` values from ``t_near`` to ``t_far`` spaced by ~``step_along_axis_A`` (Å)."""
    if t_far < t_near:
        t_near, t_far = t_far, t_near
    span = t_far - t_near
    if span < 1e-9:
        return np.array([0.5 * (t_near + t_far)], dtype=np.float64)
    step = max(float(step_along_axis_A), 1e-6)
    ts = [t_near]
    t = t_near + step
    eps = 1e-4 * max(1.0, abs(t_far))
    while t < t_far - eps:
        ts.append(t)
        t += step
    if abs(ts[-1] - t_far) > eps:
        ts.append(t_far)
    return np.asarray(ts, dtype=np.float64)


def write_axis_trace_pdb(
    path: Path,
    xyz: np.ndarray,
    remarks: Optional[list[str]] = None,
    chain_id: str = "A",
) -> None:
    """Write CA ``GLY`` atoms with ``CONECT`` bonds along the polyline (Å, Cartesian x,y,z)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    if remarks:
        for r in remarks:
            for chunk in (r[i : i + 70] for i in range(0, len(r), 70)):
                lines.append(f"REMARK {chunk}")
    serial = 1
    resseq = 1
    prev_serial: Optional[int] = None
    for p in np.asarray(xyz, dtype=np.float64).reshape(-1, 3):
        x, y, z = float(p[0]), float(p[1]), float(p[2])
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
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_symmetry_axis_geometry(
    phase0_dir: Path,
    *,
    family: Literal["c", "d"] = "c",
    refinement_index: int = 0,
    prefer_phase3: bool = True,
) -> tuple[np.ndarray, np.ndarray, int, str, dict[str, Any]]:
    """
    Return ``(axis_unit_xyz, pivot_xyz, n_fold, provenance, extra)`` using phase 3 if present,
    else phase 2 global best (pivot = phase-0 COM).

    ``n_fold`` is the best order from phase 2/3 metadata (hint for reporting); multishell scores
    all ``orders`` separately per shell.
    """
    phase0_dir = Path(phase0_dir).expanduser().resolve()
    p0_path = phase0_dir / "symmetry_phase0.json"
    p2_path = phase0_dir / ("symmetry_phase2d.json" if family == "d" else "symmetry_phase2.json")
    if not p0_path.is_file():
        raise FileNotFoundError(f"Missing {p0_path}")
    if not p2_path.is_file():
        raise FileNotFoundError(f"Missing {p2_path}")

    with open(p0_path, encoding="utf-8") as fh:
        p0 = json.load(fh)
    with open(p2_path, encoding="utf-8") as fh:
        p2 = json.load(fh)

    com0 = np.array(p0["center_of_mass_angstrom_xyz"], dtype=np.float64)
    p3_path = phase0_dir / ("symmetry_phase3d.json" if family == "d" else "symmetry_phase3.json")
    extra: dict[str, Any] = {}

    if prefer_phase3 and p3_path.is_file():
        with open(p3_path, encoding="utf-8") as fh:
            p3 = json.load(fh)
        refs = list(p3.get("refinements") or [])
        idx = int(refinement_index)
        if 0 <= idx < len(refs):
            r = refs[idx]
            if "refined_axis_xyz" in r and "refined_pivot_xyz" in r:
                u = np.array(r["refined_axis_xyz"], dtype=np.float64)
                p = np.array(r["refined_pivot_xyz"], dtype=np.float64)
                n = int(r.get("n", 2))
                u = u / np.linalg.norm(u)
                extra["phase3_refinement_index"] = idx
                key = "phase2d_candidate_id" if family == "d" else "phase2_candidate_id"
                extra[key] = int(r.get(key, -1))
                return u, p, n, "phase3_refined", extra

    gb = p2.get("global_best") or {}
    cid = int(gb.get("candidate_id", -1))
    n = int(gb.get("n", 2))
    cand_list = p2.get("candidates") or []
    cand = next((c for c in cand_list if int(c.get("id", -2)) == cid), None)
    if cand is None and cand_list:
        cand = max(cand_list, key=lambda c: float(c.get("best_score", -99.0)))
        cid = int(cand.get("id", -1))
        n = int(cand.get("best_n", n))
    if cand is None:
        raise ValueError("No phase-2 candidates available for axis geometry.")
    u = np.array(cand["direction_xyz"], dtype=np.float64)
    u = u / np.linalg.norm(u)
    extra["phase2d_candidate_id" if family == "d" else "phase2_candidate_id"] = cid
    return u, com0, n, "phase2_global_com", extra


@dataclass
class Phase4Result:
    output_pdb: str
    output_json: str
    reference_map: str
    n_points: int
    symmetry_family: str
    n_fold: int
    axis_source: str
    slice_step_voxels: float
    apix_reference: float

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_phase4_axis_pdb(
    phase0_dir: Path,
    *,
    family: Literal["c", "d"] = "c",
    out_pdb: Optional[Path] = None,
    reference_map_path: Optional[Path] = None,
    slice_step_voxels: float = 10.0,
    refinement_index: int = 0,
    prefer_phase3: bool = True,
) -> Phase4Result:
    """
    Write a polyline of CA atoms along the Cₙ axis, clipped to the reference map bounding box.

    Coordinates are in Å (MRC header origin + orthorhombic cell), matching the map opened in the
    viewer. Spacing along the axis is ``slice_step_voxels * apix`` (Å), i.e. one pseudoatom every
    *slice_step_voxels* full-map voxels along the axis direction.

    Default reference map is ``input_map`` from ``symmetry_phase0.json``; override with the
    unfiltered twin **only if** it shares the same origin, apix, and dimensions as that input.
    """
    phase0_dir = Path(phase0_dir).expanduser().resolve()
    p0_path = phase0_dir / "symmetry_phase0.json"
    if not p0_path.is_file():
        raise FileNotFoundError(f"Missing {p0_path}")

    with open(p0_path, encoding="utf-8") as fh:
        p0 = json.load(fh)

    ref_path = Path(reference_map_path).expanduser().resolve() if reference_map_path else Path(p0["input_map"]).expanduser().resolve()
    if not ref_path.is_file():
        raise FileNotFoundError(f"Reference map not found: {ref_path}")

    mv = read_map(ref_path)
    nz, ny, nx = (int(s) for s in mv.data_zyx.shape)
    apix_ref = float(mv.apix)
    origin = np.asarray(mv.origin_xyzA, dtype=np.float64).reshape(3)
    bmin = origin.copy()
    bmax = origin + np.array([float(nx), float(ny), float(nz)], dtype=np.float64) * apix_ref

    u, pivot, n_fold, prov, extra = load_symmetry_axis_geometry(
        phase0_dir,
        family=family,
        refinement_index=refinement_index,
        prefer_phase3=prefer_phase3,
    )

    hit = ray_aabb_intersect_t(pivot, u, bmin, bmax)
    if hit is None:
        hit = ray_aabb_intersect_t(pivot, -u, bmin, bmax)
        if hit is None:
            raise RuntimeError("Symmetry axis does not intersect the reference map bounding box.")
        u = -u
    t_near, t_far = hit

    step_A = float(slice_step_voxels) * apix_ref
    ts = sample_axis_parameters(t_near, t_far, step_along_axis_A=step_A)
    points = pivot.reshape(1, 3) + ts.reshape(-1, 1) * u.reshape(1, 3)

    out = Path(out_pdb).expanduser().resolve() if out_pdb else phase0_dir / "symmetry_axis_ca.pdb"
    remarks = [
        "CryoModel symmetry axis trace (C_n, map frame).",
        f"Reference map: {ref_path}",
        f"C_n order n={n_fold}; axis source: {prov}",
        f"Spacing along axis ~ {slice_step_voxels:g} voxels * apix ({apix_ref:g} A) = {step_A:g} A.",
        f"Phase0 dir: {phase0_dir}",
    ]
    write_axis_trace_pdb(out, points, remarks=remarks)

    meta = Phase4Result(
        output_pdb=str(out),
        output_json=str(phase0_dir / "symmetry_phase4.json"),
        reference_map=str(ref_path),
        n_points=int(points.shape[0]),
        symmetry_family="D" if family == "d" else "C",
        n_fold=int(n_fold),
        axis_source=str(prov),
        slice_step_voxels=float(slice_step_voxels),
        apix_reference=float(apix_ref),
    )
    payload = meta.to_json_dict()
    payload["axis_xyz"] = [float(v) for v in u]
    payload["pivot_xyz"] = [float(v) for v in pivot]
    payload["t_range"] = [float(t_near), float(t_far)]
    payload["extra"] = extra
    with open(meta.output_json, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return meta
