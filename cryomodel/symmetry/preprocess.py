"""Phase 0: map preprocess for symmetry search — mask, optional filters, downsample, inertia/PCA axes."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np

from cryomodel.io.mrc import MapVolume, read_map, write_map
from cryomodel.maps import filters

EdgeMode = Literal["none", "laplacian", "laplacian_sharpen"]


@dataclass
class Phase0Result:
    """Summary of preprocessing and a first-guess principal frame from high-density voxels."""

    input_map: str
    mask_path: Optional[str]
    shape_in: tuple[int, int, int]
    shape_out: tuple[int, int, int]
    apix_in: float
    apix_out: float
    origin_xyzA: list[float]
    downsample_factor: int
    bandpass_low_res_A: Optional[float]
    bandpass_high_res_A: Optional[float]
    edge_emphasis: str
    laplacian_sharpen_strength: float
    density_threshold: float
    density_percentile: Optional[float]
    n_voxels_above_threshold: int
    n_voxels_in_pca: int
    center_of_mass_angstrom_xyz: list[float]
    inertia_eigenvalues: list[float]
    principal_axes_xyz: list[list[float]]  # rows = eigenvectors (3), primary axis = row 0
    eigenvalue_fractions: list[float]
    output_map: str
    output_json: str

    def to_json_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


def _resample_mask_to_shape(mask_zyx: np.ndarray, target_shape: tuple[int, int, int]) -> np.ndarray:
    from scipy import ndimage

    mz, my, mx = mask_zyx.shape
    tz, ty, tx = target_shape
    if (mz, my, mx) == (tz, ty, tx):
        return mask_zyx.astype(np.float32, copy=False)
    zoom = (tz / mz, ty / my, tx / mx)
    out = ndimage.zoom(mask_zyx.astype(np.float64), zoom, order=1)
    return out.astype(np.float32)


def downsample_block_mean(data_zyx: np.ndarray, factor: int) -> np.ndarray:
    """Average-pool by integer factor along z,y,x. Trims trailing partial blocks."""
    if factor <= 1:
        return np.asarray(data_zyx, dtype=np.float32)
    arr = np.asarray(data_zyx, dtype=np.float64)
    nz, ny, nx = arr.shape
    nz2 = (nz // factor) * factor
    ny2 = (ny // factor) * factor
    nx2 = (nx // factor) * factor
    if nz2 == 0 or ny2 == 0 or nx2 == 0:
        raise ValueError("Map too small for requested downsample factor.")
    t = arr[:nz2, :ny2, :nx2]
    sh = (nz2 // factor, factor, ny2 // factor, factor, nx2 // factor, factor)
    out = t.reshape(sh).mean(axis=(1, 3, 5))
    return out.astype(np.float32)


def _voxel_centers_xyz(
    iz: np.ndarray, iy: np.ndarray, ix: np.ndarray, origin_xyzA: np.ndarray, apix: float
) -> np.ndarray:
    """Voxel center coordinates in Å, shape (N, 3) columns x,y,z."""
    ox, oy, oz = (float(origin_xyzA[0]), float(origin_xyzA[1]), float(origin_xyzA[2]))
    x = ox + (ix.astype(np.float64) + 0.5) * apix
    y = oy + (iy.astype(np.float64) + 0.5) * apix
    z = oz + (iz.astype(np.float64) + 0.5) * apix
    return np.stack([x, y, z], axis=1)


def weighted_principal_axes(coords_xyz: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (center_of_mass_xyz, eigenvalues_descending, eigenvectors_rows_descending)
    eigenvectors_rows: 3x3, row i = i-th principal direction (same order as eigenvalues).
    """
    w = np.maximum(weights.astype(np.float64), 0.0)
    sw = float(w.sum())
    if sw <= 0:
        raise ValueError("No positive weights for principal-axis estimate.")
    w /= sw
    com = (coords_xyz.astype(np.float64) * w[:, None]).sum(axis=0)
    x = coords_xyz.astype(np.float64) - com
    swx = np.sqrt(np.maximum(w, 0.0))
    # Some OpenBLAS builds emit spurious matmul warnings on (N,3)@(3,N) for large N;
    # the result is finite for well-behaved cryo maps.
    with np.errstate(all="ignore"):
        cov = (swx[:, None] * x).T @ (swx[:, None] * x)
    # Tiny ridge for numerical stability with near-degenerate point clouds
    cov = cov + np.eye(3, dtype=np.float64) * (1e-12 * float(np.trace(cov)) + 1e-18)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order].T
    return com.astype(np.float64), evals.astype(np.float64), evecs.astype(np.float64)


def run_phase0_preprocess(
    map_path: Path,
    *,
    out_dir: Path,
    mask_path: Optional[Path] = None,
    downsample_factor: int = 4,
    bandpass_low_res_A: Optional[float] = None,
    bandpass_high_res_A: Optional[float] = None,
    edge_emphasis: EdgeMode = "none",
    laplacian_sharpen_strength: float = 1.0,
    density_threshold: Optional[float] = None,
    density_percentile: Optional[float] = 90.0,
    max_voxels_pca: int = 400_000,
    random_seed: int = 0,
) -> Phase0Result:
    """
    Load map, optionally apply mask and band-pass / edge emphasis, downsample, then estimate
    principal axes from voxels above a density cutoff (percentile or absolute threshold).
    """
    map_path = Path(map_path).expanduser().resolve()
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    mv = read_map(map_path)
    data = mv.data_zyx.astype(np.float32, copy=True)
    shape_in = tuple(int(x) for x in data.shape)

    if mask_path is not None:
        mm = read_map(Path(mask_path).expanduser().resolve())
        mdat = mm.data_zyx.astype(np.float32)
        if mdat.shape != data.shape:
            mdat = _resample_mask_to_shape(mdat, data.shape)
        data *= mdat

    if bandpass_low_res_A is not None and bandpass_high_res_A is not None:
        data = filters.bandpass(data, mv.apix, bandpass_low_res_A, bandpass_high_res_A, order=None)

    if edge_emphasis == "laplacian":
        data = filters.laplacian_filter(data).astype(np.float32)
    elif edge_emphasis == "laplacian_sharpen":
        data = filters.laplacian_sharpen(data, strength=float(laplacian_sharpen_strength)).astype(np.float32)
    elif edge_emphasis != "none":
        raise ValueError(f"Unknown edge_emphasis: {edge_emphasis}")

    dsf = max(1, int(downsample_factor))
    down = downsample_block_mean(data, dsf)
    apix_out = float(mv.apix) * dsf
    shape_out = tuple(int(x) for x in down.shape)

    flat = down.ravel()
    pos = flat[flat > 0]
    if pos.size == 0:
        pos = flat

    if density_threshold is not None:
        thr = float(density_threshold)
        used_percentile: Optional[float] = None
    else:
        p = 90.0 if density_percentile is None else float(density_percentile)
        p = min(100.0, max(0.0, p))
        thr = float(np.percentile(pos, p))
        used_percentile = p

    sel = down > thr
    iz, iy, ix = np.nonzero(sel)
    w = down[sel].astype(np.float64)
    n_all = int(w.size)
    if n_all == 0:
        raise ValueError("No voxels above density threshold after preprocessing.")

    rng = np.random.default_rng(random_seed)
    if n_all > max_voxels_pca:
        idx = rng.choice(n_all, size=int(max_voxels_pca), replace=False)
        iz, iy, ix = iz[idx], iy[idx], ix[idx]
        w = w[idx]
    n_used = int(w.size)

    coords = _voxel_centers_xyz(iz, iy, ix, mv.origin_xyzA, apix_out)
    com, evals, axes = weighted_principal_axes(coords, w)
    total_var = float(evals.sum()) if float(evals.sum()) > 0 else 1.0
    frac = [float(e / total_var) for e in evals]

    out_map = out_dir / "symmetry_phase0_downsample.mrc"
    mv_out = MapVolume(
        data_zyx=down,
        apix=apix_out,
        origin_xyzA=np.asarray(mv.origin_xyzA, dtype=np.float32).copy(),
        halfmaps=None,
        grid=None,
        _ccp4=None,
    )
    write_map(out_map, mv_out, down)

    result = Phase0Result(
        input_map=str(map_path),
        mask_path=str(mask_path) if mask_path else None,
        shape_in=shape_in,
        shape_out=shape_out,
        apix_in=float(mv.apix),
        apix_out=apix_out,
        origin_xyzA=[float(x) for x in mv.origin_xyzA],
        downsample_factor=dsf,
        bandpass_low_res_A=bandpass_low_res_A,
        bandpass_high_res_A=bandpass_high_res_A,
        edge_emphasis=str(edge_emphasis),
        laplacian_sharpen_strength=float(laplacian_sharpen_strength),
        density_threshold=float(thr),
        density_percentile=used_percentile,
        n_voxels_above_threshold=n_all,
        n_voxels_in_pca=n_used,
        center_of_mass_angstrom_xyz=[float(x) for x in com],
        inertia_eigenvalues=[float(x) for x in evals],
        principal_axes_xyz=[[float(v) for v in row] for row in axes],
        eigenvalue_fractions=frac,
        output_map=str(out_map),
        output_json=str(out_dir / "symmetry_phase0.json"),
    )

    with open(result.output_json, "w", encoding="utf-8") as fh:
        json.dump(result.to_json_dict(), fh, indent=2)

    return result
