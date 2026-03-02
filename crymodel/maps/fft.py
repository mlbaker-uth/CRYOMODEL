import numpy as np

def fourier_gaussian_lowpass(vol: np.ndarray, apix: float, resA: float) -> np.ndarray:
    F = np.fft.fftn(vol)
    nz, ny, nx = vol.shape
    kz = np.fft.fftfreq(nz, d=apix); ky = np.fft.fftfreq(ny, d=apix); kx = np.fft.fftfreq(nx, d=apix)
    KZ, KY, KX = np.meshgrid(kz, ky, kx, indexing='ij')
    k = np.sqrt(KZ**2 + KY**2 + KX**2)
    sigma = 1.0 / (resA * np.sqrt(8*np.log(2)))
    H = np.exp(-(k**2) / (2*sigma**2))
    return np.fft.ifftn(F * H).real.astype(np.float32)

def add_colored_noise(vol: np.ndarray, snr: float = 5.0) -> np.ndarray:
    rng = np.random.default_rng()
    noise = rng.standard_normal(vol.shape).astype(np.float32)
    noise = fourier_gaussian_lowpass(noise, apix=1.0, resA=8.0)
    scale = np.std(vol) / max(snr, 1e-6)
    return (vol + noise * scale).astype(np.float32)
