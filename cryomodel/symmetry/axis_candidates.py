"""Phase 1: discrete symmetry-axis candidates + 1D axial / radial summaries on the phase-0 grid."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from cryomodel.io.mrc import read_map

from .preprocess import _voxel_centers_xyz


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-15:
        raise ValueError("Zero-length direction")
    return (v / n).astype(np.float64)


def cardinal_and_tilt_directions(tilt_degrees: tuple[float, ...]) -> list[np.ndarray]:
    """
    Near-cardinal directions: ±X, ±Y, ±Z plus small tilts (rotate each toward the other two axes).
    tilt_degrees should include 0 for pure cardinals.
    """
    out: list[np.ndarray] = []
    ex = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    ey = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    ez = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    def add_tilted_z():
        for deg in tilt_degrees:
            t = np.deg2rad(float(deg))
            if deg == 0.0:
                out.append(ez)
                out.append(-ez)
                continue
            # Small-circle tilts from +Z toward ±X and ±Y
            out.append(_unit(np.array([np.sin(t), 0.0, np.cos(t)])))
            out.append(_unit(np.array([-np.sin(t), 0.0, np.cos(t)])))
            out.append(_unit(np.array([0.0, np.sin(t), np.cos(t)])))
            out.append(_unit(np.array([0.0, -np.sin(t), np.cos(t)])))

    def add_tilted_x():
        for deg in tilt_degrees:
            t = np.deg2rad(float(deg))
            if deg == 0.0:
                out.append(ex)
                out.append(-ex)
                continue
            out.append(_unit(np.array([np.cos(t), 0.0, np.sin(t)])))
            out.append(_unit(np.array([np.cos(t), 0.0, -np.sin(t)])))
            out.append(_unit(np.array([np.cos(t), np.sin(t), 0.0])))
            out.append(_unit(np.array([np.cos(t), -np.sin(t), 0.0])))

    def add_tilted_y():
        for deg in tilt_degrees:
            t = np.deg2rad(float(deg))
            if deg == 0.0:
                out.append(ey)
                out.append(-ey)
                continue
            out.append(_unit(np.array([0.0, np.cos(t), np.sin(t)])))
            out.append(_unit(np.array([0.0, np.cos(t), -np.sin(t)])))
            out.append(_unit(np.array([np.sin(t), np.cos(t), 0.0])))
            out.append(_unit(np.array([-np.sin(t), np.cos(t), 0.0])))

    add_tilted_x()
    add_tilted_y()
    add_tilted_z()
    return out


def diagonal_directions() -> list[np.ndarray]:
    """Body diagonals of the cube (±1,±1,±1) normalized — eight directions, four unique lines."""
    out = []
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                out.append(_unit(np.array([sx, sy, sz], dtype=np.float64)))
    return out


def canonical_axis_key(u: np.ndarray) -> tuple[float, float, float]:
    """Map u and -u to the same key (hemisphere: flip so largest-magnitude component is positive)."""
    v = _unit(u)
    j = int(np.argmax(np.abs(v)))
    if v[j] < 0:
        v = -v
    return (float(v[0]), float(v[1]), float(v[2]))


def merge_candidate_sources(
    tilt_degrees: tuple[float, ...],
    *,
    include_diagonals: bool,
    pca_axes: Optional[list[list[float]]],
) -> list[tuple[np.ndarray, str]]:
    """Return (direction, source_label) before dedupe; then dedupe preserving first source."""
    items: list[tuple[np.ndarray, str]] = []

    for u in cardinal_and_tilt_directions(tilt_degrees):
        items.append((u, "cardinal_tilt"))

    if include_diagonals:
        for u in diagonal_directions():
            items.append((u, "diagonal"))

    if pca_axes:
        for i, row in enumerate(pca_axes):
            u = np.array(row, dtype=np.float64)
            items.append((u, f"pca_axis_{i}"))
            items.append((-u, f"pca_axis_{i}_neg"))

    # Dedupe lines
    seen: set[tuple[float, float, float]] = set()
    out: list[tuple[np.ndarray, str]] = []
    for u, src in items:
        key = canonical_axis_key(u)
        if key in seen:
            continue
        seen.add(key)
        v = _unit(u)
        # Store canonical direction for consistent axial coordinate sign
        if v[np.argmax(np.abs(v))] < 0:
            v = -v
        out.append((v, src))
    return out


def _summarize_one_candidate(
    data_zyx: np.ndarray,
    iz: np.ndarray,
    iy: np.ndarray,
    ix: np.ndarray,
    origin_xyzA: np.ndarray,
    apix: float,
    com: np.ndarray,
    u: np.ndarray,
    *,
    n_axial_bins: int,
) -> dict[str, Any]:
    pos = _voxel_centers_xyz(iz, iy, ix, origin_xyzA, apix).astype(np.float64)
    rel = pos - com.reshape(1, 3)
    w = data_zyx[iz, iy, ix].astype(np.float64)
    w = np.maximum(w, 0.0)
    sw = float(w.sum())
    if sw <= 0:
        return {
            "axial_bin_edges_A": [],
            "axial_mass": [],
            "s_min_A": 0.0,
            "s_max_A": 0.0,
            "mean_radial_A": 0.0,
            "rms_radial_A": 0.0,
            "integrated_mass": 0.0,
            "n_voxels": int(iz.size),
        }

    u = u.astype(np.float64, copy=False)
    with np.errstate(all="ignore"):
        s = rel @ u
    s_proj = (s[:, np.newaxis]) * u.reshape(1, 3)
    rho = np.linalg.norm(rel - s_proj, axis=1)
    mean_rho = float((rho * w).sum() / sw)
    rms_rho = float(np.sqrt((rho * rho * w).sum() / sw))

    p_lo, p_hi = np.percentile(s, [0.5, 99.5])
    if p_hi <= p_lo:
        p_lo, p_hi = float(s.min()), float(s.max())
    if p_hi <= p_lo:
        p_hi = p_lo + 1e-6
    edges = np.linspace(p_lo, p_hi, int(n_axial_bins) + 1)
    mass, _ = np.histogram(s, bins=edges, weights=w)
    return {
        "axial_bin_edges_A": [float(x) for x in edges],
        "axial_mass": [float(x) for x in mass],
        "s_min_A": float(p_lo),
        "s_max_A": float(p_hi),
        "mean_radial_A": mean_rho,
        "rms_radial_A": rms_rho,
        "integrated_mass": float(sw),
        "n_voxels": int(iz.size),
    }


@dataclass
class Phase1Result:
    phase0_json: str
    input_downsample_map: str
    n_candidates: int
    tilt_degrees: list[float]
    include_diagonals: bool
    candidates: list[dict[str, Any]]
    output_json: str

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_phase1_candidates(
    phase0_dir: Path,
    *,
    tilt_degrees: tuple[float, ...] = (0.0, 5.0, 10.0, 15.0),
    include_diagonals: bool = True,
    n_axial_bins: int = 64,
) -> Phase1Result:
    """
    Read phase-0 outputs from ``phase0_dir`` (symmetry_phase0.json + symmetry_phase0_downsample.mrc),
    build discrete axis candidates, and write 1D axial mass profiles + radial stats for each.
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
    com = np.array(p0["center_of_mass_angstrom_xyz"], dtype=np.float64)

    sel = data > thr
    iz, iy, ix = np.nonzero(sel)
    if iz.size == 0:
        raise ValueError("No voxels above phase-0 density threshold on downsampled map.")

    pca_axes = p0.get("principal_axes_xyz")
    items = merge_candidate_sources(
        tilt_degrees,
        include_diagonals=include_diagonals,
        pca_axes=pca_axes if isinstance(pca_axes, list) else None,
    )

    candidates: list[dict[str, Any]] = []
    for i, (u, src) in enumerate(items):
        summ = _summarize_one_candidate(
            data, iz, iy, ix, origin, apix, com, u, n_axial_bins=n_axial_bins
        )
        candidates.append(
            {
                "id": i,
                "source": src,
                "direction_xyz": [float(x) for x in u],
                **summ,
            }
        )

    out_path = phase0_dir / "symmetry_phase1.json"
    result = Phase1Result(
        phase0_json=str(p0_json),
        input_downsample_map=str(p0_map),
        n_candidates=len(candidates),
        tilt_degrees=[float(x) for x in tilt_degrees],
        include_diagonals=bool(include_diagonals),
        candidates=candidates,
        output_json=str(out_path),
    )
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(result.to_json_dict(), fh, indent=2)
    return result
