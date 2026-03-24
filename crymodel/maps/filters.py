# crymodel/maps/filters.py
"""Common cryo-EM map filters: low/high/bandpass, Gaussian, threshold, binary, bilateral, Laplacian, median, Butterworth."""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from typing import Optional

# Optional scikit-image for 2D bilateral; we provide a simple 3D fallback
try:
    from skimage.restoration import denoise_bilateral as _denoise_bilateral_2d
except ImportError:
    _denoise_bilateral_2d = None


def _freq_grid_zyx(shape_zyx: tuple, apix: float) -> np.ndarray:
    """Return |k| in 1/Å for each voxel, shape (z,y,x)."""
    nz, ny, nx = shape_zyx
    kz = np.fft.fftfreq(nz, d=apix)
    ky = np.fft.fftfreq(ny, d=apix)
    kx = np.fft.fftfreq(nx, d=apix)
    KZ, KY, KX = np.meshgrid(kz, ky, kx, indexing="ij")
    return np.sqrt(KZ**2 + KY**2 + KX**2).astype(np.float32)


def lowpass(data_zyx: np.ndarray, apix: float, resolution_A: float, gaussian: bool = True) -> np.ndarray:
    """Low-pass filter: attenuate frequencies beyond 1/resolution_A. If gaussian=True use Gaussian rolloff, else sharp cutoff."""
    k = _freq_grid_zyx(data_zyx.shape, apix)
    k_cut = 1.0 / resolution_A
    F = np.fft.fftn(data_zyx.astype(np.float64))
    if gaussian:
        sigma = k_cut / np.sqrt(8 * np.log(2))
        H = np.exp(-(k**2) / (2 * sigma**2))
    else:
        H = (k <= k_cut).astype(np.float64)
    out = np.fft.ifftn(F * H).real.astype(np.float32)
    return out


def highpass(data_zyx: np.ndarray, apix: float, resolution_A: float, gaussian: bool = True) -> np.ndarray:
    """High-pass filter: remove frequencies below 1/resolution_A (e.g. subtract low-pass from data)."""
    k = _freq_grid_zyx(data_zyx.shape, apix)
    k_cut = 1.0 / resolution_A
    F = np.fft.fftn(data_zyx.astype(np.float64))
    if gaussian:
        sigma = k_cut / np.sqrt(8 * np.log(2))
        H_low = np.exp(-(k**2) / (2 * sigma**2))
    else:
        H_low = (k <= k_cut).astype(np.float64)
    H_high = 1.0 - H_low
    out = np.fft.ifftn(F * H_high).real.astype(np.float32)
    return out


def bandpass(
    data_zyx: np.ndarray,
    apix: float,
    low_res_A: float,
    high_res_A: float,
    order: Optional[int] = None,
) -> np.ndarray:
    """Bandpass: keep frequencies between 1/high_res_A and 1/low_res_A. Optional Butterworth order (smooth rolloff)."""
    k = _freq_grid_zyx(data_zyx.shape, apix)
    k_high = 1.0 / high_res_A  # pass above this (high freq)
    k_low = 1.0 / low_res_A    # pass below this (low freq)
    F = np.fft.fftn(data_zyx.astype(np.float64))
    if order is not None and order >= 1:
        # Butterworth bandpass: 1 / (1 + (k_low/k)^(2n)) * 1 / (1 + (k/k_high)^(2n))
        k_safe = np.maximum(k, 1e-12)
        L = 1.0 / (1.0 + (k_low / k_safe) ** (2 * order))
        H = 1.0 / (1.0 + (k_safe / k_high) ** (2 * order))
        H_band = L * H
    else:
        H_band = ((k >= k_high) & (k <= k_low)).astype(np.float64)
    out = np.fft.ifftn(F * H_band).real.astype(np.float32)
    return out


def gaussian_filter(data_zyx: np.ndarray, sigma_vox: float) -> np.ndarray:
    """Spatial Gaussian blur. sigma_vox: standard deviation in voxels (isotropic)."""
    return ndimage.gaussian_filter(data_zyx.astype(np.float64), sigma=sigma_vox, mode="nearest").astype(np.float32)


def threshold_filter(
    data_zyx: np.ndarray,
    threshold: float,
    below_value: float = 0.0,
    above_scale: Optional[float] = None,
) -> np.ndarray:
    """Set values below threshold to below_value; optionally scale values above (above_scale) or leave as-is."""
    out = np.where(data_zyx >= threshold, data_zyx.astype(np.float64), below_value)
    if above_scale is not None:
        out = np.where(data_zyx >= threshold, (data_zyx - threshold) * above_scale + below_value, below_value)
    return out.astype(np.float32)


def binary_filter(data_zyx: np.ndarray, threshold: float, one_above: bool = True) -> np.ndarray:
    """Binary mask: 1 where density >= threshold (if one_above), else 0."""
    if one_above:
        out = (data_zyx >= threshold).astype(np.float32)
    else:
        out = (data_zyx < threshold).astype(np.float32)
    return out


def laplacian_filter(data_zyx: np.ndarray, mode: str = "nearest") -> np.ndarray:
    """Laplacian (edge enhancement). Output can be added to original for sharpening."""
    out = ndimage.laplace(data_zyx.astype(np.float64), mode=mode).astype(np.float32)
    return out


def laplacian_sharpen(data_zyx: np.ndarray, strength: float = 1.0, mode: str = "nearest") -> np.ndarray:
    """Sharpen by subtracting scaled Laplacian: data - strength * laplace(data)."""
    lap = ndimage.laplace(data_zyx.astype(np.float64), mode=mode)
    out = (data_zyx.astype(np.float64) - strength * lap).astype(np.float32)
    return out


def median_filter(data_zyx: np.ndarray, size: int) -> np.ndarray:
    """Median filter (noise reduction, preserves edges). size: cubic half-width (total 2*size+1)."""
    return ndimage.median_filter(data_zyx.astype(np.float64), size=size, mode="nearest").astype(np.float32)


def bilateral_filter_3d(
    data_zyx: np.ndarray,
    sigma_spatial: float = 1.0,
    sigma_range: Optional[float] = None,
) -> np.ndarray:
    """Simple 3D bilateral filter (edge-preserving). sigma_range: intensity (default from data std)."""
    if sigma_range is None:
        sigma_range = float(np.std(data_zyx)) * 0.1 + 1e-6
    size = max(2, int(round(3 * sigma_spatial)))
    data = data_zyx.astype(np.float64)
    out = np.empty_like(data)
    nz, ny, nx = data.shape
    for z in range(nz):
        z0 = max(0, z - size)
        z1 = min(nz, z + size + 1)
        for y in range(ny):
            y0 = max(0, y - size)
            y1 = min(ny, y + size + 1)
            for x in range(nx):
                x0 = max(0, x - size)
                x1 = min(nx, x + size + 1)
                patch = data[z0:z1, y0:y1, x0:x1]
                d = (np.arange(z0, z1) - z) ** 2 + (np.arange(y0, y1).reshape(-1, 1) - y) ** 2 + (np.arange(x0, x1).reshape(1, -1) - x) ** 2
                spatial_w = np.exp(-0.5 * d / (sigma_spatial**2))
                range_w = np.exp(-0.5 * ((patch - data[z, y, x]) / sigma_range) ** 2)
                w = spatial_w * range_w
                out[z, y, x] = np.average(patch, weights=w)
    return out.astype(np.float32)


def butterworth_lowpass(
    data_zyx: np.ndarray,
    apix: float,
    resolution_A: float,
    order: int = 4,
) -> np.ndarray:
    """Butterworth low-pass: smooth rolloff at resolution_A. order: steepness (default 4)."""
    k = _freq_grid_zyx(data_zyx.shape, apix)
    k_cut = 1.0 / resolution_A
    k_safe = np.maximum(k, 1e-12)
    H = 1.0 / (1.0 + (k_safe / k_cut) ** (2 * order))
    F = np.fft.fftn(data_zyx.astype(np.float64))
    out = np.fft.ifftn(F * H).real.astype(np.float32)
    return out


def butterworth_highpass(
    data_zyx: np.ndarray,
    apix: float,
    resolution_A: float,
    order: int = 4,
) -> np.ndarray:
    """Butterworth high-pass: smooth rolloff below resolution_A."""
    k = _freq_grid_zyx(data_zyx.shape, apix)
    k_cut = 1.0 / resolution_A
    k_safe = np.maximum(k, 1e-12)
    H_low = 1.0 / (1.0 + (k_safe / k_cut) ** (2 * order))
    H = 1.0 - H_low
    F = np.fft.fftn(data_zyx.astype(np.float64))
    out = np.fft.ifftn(F * H).real.astype(np.float32)
    return out


def normalize_zero_mean_unit_variance(data_zyx: np.ndarray) -> np.ndarray:
    """Normalize map to zero mean and unit variance (common preprocessing)."""
    d = data_zyx.astype(np.float64)
    mean, std = d.mean(), d.std()
    if std < 1e-12:
        return data_zyx.astype(np.float32)
    return ((d - mean) / std).astype(np.float32)


FILTER_REGISTRY = {
    "lowpass": (lowpass, "resolution_A", "Low-pass Fourier filter (Gaussian or sharp cutoff)"),
    "highpass": (highpass, "resolution_A", "High-pass Fourier filter"),
    "bandpass": (bandpass, "low_res_A, high_res_A [, order]", "Bandpass Fourier (optional Butterworth)"),
    "gaussian": (gaussian_filter, "sigma_vox", "Spatial Gaussian blur"),
    "threshold": (threshold_filter, "threshold [, below_value]", "Set values below threshold"),
    "binary": (binary_filter, "threshold", "Binary mask (0/1) by threshold"),
    "laplacian": (laplacian_filter, "", "Laplacian edge enhancement"),
    "laplacian-sharpen": (laplacian_sharpen, "strength", "Sharpen via Laplacian subtraction"),
    "median": (median_filter, "size", "Median filter (noise reduction)"),
    "bilateral": (bilateral_filter_3d, "sigma_spatial [, sigma_range]", "Edge-preserving bilateral filter"),
    "butterworth-lowpass": (butterworth_lowpass, "resolution_A [, order]", "Butterworth low-pass"),
    "butterworth-highpass": (butterworth_highpass, "resolution_A [, order]", "Butterworth high-pass"),
    "normalize": (normalize_zero_mean_unit_variance, "", "Zero mean, unit variance"),
}
