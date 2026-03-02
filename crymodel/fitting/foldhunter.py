# crymodel/fitting/foldhunter.py
"""FoldHunter: exhaustive cross-correlation search for fitting AlphaFold models to cryo-EM maps."""
from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import gemmi
from scipy.ndimage import gaussian_filter
from scipy.spatial.transform import Rotation

from ..io.mrc import MapVolume, read_map, write_map
from ..io.pdb import read_model_xyz


@dataclass
class FoldHunterResult:
    """Result from FoldHunter search."""
    translation: np.ndarray  # (3,) translation in Å
    rotation: np.ndarray  # (4,) quaternion (w, x, y, z)
    correlation: float
    atom_inclusion_score: float
    n_atoms_in_density: int
    n_atoms_total: int
    out_of_bounds_penalty: float


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Convert quaternion (w, x, y, z) to 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
    ], dtype=np.float32)


def sample_quaternions_uniform(n_samples: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Sample quaternions uniformly on SO(3) using Marsaglia method.
    
    Returns:
        Array of shape (n_samples, 4) with quaternions (w, x, y, z)
    """
    if rng is None:
        rng = np.random.default_rng()
    
    u1 = rng.uniform(0, 1, n_samples)
    u2 = rng.uniform(0, 1, n_samples)
    u3 = rng.uniform(0, 1, n_samples)
    
    w = np.sqrt(1 - u1) * np.sin(2 * np.pi * u2)
    x = np.sqrt(1 - u1) * np.cos(2 * np.pi * u2)
    y = np.sqrt(u1) * np.sin(2 * np.pi * u3)
    z = np.sqrt(u1) * np.cos(2 * np.pi * u3)
    
    return np.stack([w, x, y, z], axis=1).astype(np.float32)


def pdb_to_density_map(
    pdb_path: Path,
    target_apix: float,
    target_shape: Tuple[int, int, int],
    target_origin: np.ndarray,
    resolution_A: float,
    plddt_threshold: float = 0.5,
    remove_hydrogens: bool = True,
    atom_radius_A: float = 1.5,
) -> np.ndarray:
    """Convert PDB structure to density map at specified resolution.
    
    Args:
        pdb_path: Path to PDB file
        target_apix: Voxel size in Å
        target_shape: Shape of target map (z, y, x)
        target_origin: Origin of target map in Å (x, y, z)
        resolution_A: Resolution for Gaussian blur
        plddt_threshold: Minimum pLDDT (from B-factor) to include atom
        remove_hydrogens: If True, filter out hydrogen atoms
        
    Returns:
        Density map array in (z, y, x) format
    """
    st = gemmi.read_structure(str(pdb_path))
    
    # Collect atom positions and weights (based on pLDDT)
    atoms_xyz = []
    weights = []
    
    for model in st:
        for chain in model:
            for res in chain:
                for atom in res:
                    # Skip hydrogens if requested
                    if remove_hydrogens:
                        element_name = atom.element.name if atom.element else atom.name.strip()[0] if atom.name.strip() else "C"
                        if element_name.upper() == "H":
                            continue
                    
                    # Get pLDDT from B-factor (normalized 0-1)
                    plddt = atom.b_iso / 100.0 if atom.b_iso > 0 else 1.0
                    
                    # Filter by pLDDT threshold
                    if plddt < plddt_threshold:
                        continue
                    
                    pos = atom.pos
                    atoms_xyz.append([float(pos.x), float(pos.y), float(pos.z)])
                    # Weight by pLDDT (higher confidence = higher weight)
                    weights.append(plddt)
    
    if not atoms_xyz:
        return np.zeros(target_shape, dtype=np.float32)
    
    atoms_xyz = np.array(atoms_xyz, dtype=np.float32)
    weights = np.array(weights, dtype=np.float32)
    
    # Create empty map
    density = np.zeros(target_shape, dtype=np.float32)
    
    # Convert atom positions to voxel coordinates
    # Map uses (z, y, x) indexing, but origin is (x, y, z)
    z_shape, y_shape, x_shape = target_shape
    
    # Place atoms as Gaussian spheres
    radius_vox = atom_radius_A / target_apix
    sigma_vox = radius_vox / 2.0  # Gaussian width
    
    # Create a small Gaussian kernel for each atom
    kernel_radius = int(np.ceil(3 * sigma_vox))
    if kernel_radius > 0:
        k_range = np.arange(-kernel_radius, kernel_radius + 1, dtype=np.float32)
        KZ, KY, KX = np.meshgrid(k_range, k_range, k_range, indexing='ij')
        kernel = np.exp(-(KZ**2 + KY**2 + KX**2) / (2 * sigma_vox**2))
        kernel /= np.sum(kernel)  # Normalize
    
    # Place atoms
    for i, (x, y, z) in enumerate(atoms_xyz):
        # Convert to voxel coordinates (Z, Y, X)
        vx = (x - target_origin[0]) / target_apix
        vy = (y - target_origin[1]) / target_apix
        vz = (z - target_origin[2]) / target_apix
        
        # Get integer position
        vx_int = int(np.round(vx))
        vy_int = int(np.round(vy))
        vz_int = int(np.round(vz))
        
        # Place kernel
        z0 = max(0, vz_int - kernel_radius)
        z1 = min(z_shape, vz_int + kernel_radius + 1)
        y0 = max(0, vy_int - kernel_radius)
        y1 = min(y_shape, vy_int + kernel_radius + 1)
        x0 = max(0, vx_int - kernel_radius)
        x1 = min(x_shape, vx_int + kernel_radius + 1)
        
        kz0 = kernel_radius - (vz_int - z0)
        kz1 = kernel_radius + (z1 - vz_int)
        ky0 = kernel_radius - (vy_int - y0)
        ky1 = kernel_radius + (y1 - vy_int)
        kx0 = kernel_radius - (vx_int - x0)
        kx1 = kernel_radius + (x1 - vx_int)
        
        if z0 < z1 and y0 < y1 and x0 < x1:
            density[z0:z1, y0:y1, x0:x1] += weights[i] * kernel[kz0:kz1, ky0:ky1, kx0:kx1]
    
    # Apply additional Gaussian blur to simulate resolution
    # FWHM = 0.939 * resolution, sigma = FWHM / 2.355
    resolution_sigma_vox = (0.939 * resolution_A) / (2.355 * target_apix)
    if resolution_sigma_vox > 0.5:  # Only blur if significant
        density = gaussian_filter(density, sigma=resolution_sigma_vox, mode='constant')
    
    return density.astype(np.float32)


def compute_cross_correlation_fft(
    probe_map: np.ndarray,
    target_map: np.ndarray,
) -> np.ndarray:
    """Compute normalized cross-correlation using FFT.
    
    Args:
        probe_map: Probe density map (z, y, x)
        target_map: Target density map (z, y, x)
        
    Returns:
        Correlation map (z, y, x) with values in [-1, 1]
    """
    # Ensure same shape (pad if needed)
    if probe_map.shape != target_map.shape:
        # Pad probe to match target
        pad_z = target_map.shape[0] - probe_map.shape[0]
        pad_y = target_map.shape[1] - probe_map.shape[1]
        pad_x = target_map.shape[2] - probe_map.shape[2]
        
        probe_padded = np.pad(
            probe_map,
            ((0, max(0, pad_z)), (0, max(0, pad_y)), (0, max(0, pad_x))),
            mode='constant'
        )
    else:
        probe_padded = probe_map.copy()
    
    # Normalize
    probe_norm = probe_padded - np.mean(probe_padded)
    probe_std = np.std(probe_norm)
    if probe_std > 1e-6:
        probe_norm /= probe_std
    
    target_norm = target_map - np.mean(target_map)
    target_std = np.std(target_norm)
    if target_std > 1e-6:
        target_norm /= target_std
    
    # Cross-correlation via FFT
    probe_fft = np.fft.fftn(probe_norm)
    target_fft = np.fft.fftn(target_norm)
    
    # Cross-correlation is IFFT of conjugate product
    corr_fft = np.conj(probe_fft) * target_fft
    corr = np.fft.ifftn(corr_fft).real
    
    # Normalize by number of voxels
    corr /= np.sqrt(np.sum(probe_norm**2) * np.sum(target_norm**2) + 1e-10)
    
    return corr.astype(np.float32)


def find_peaks_3d(
    corr_map: np.ndarray,
    min_distance_vox: int = 5,
    threshold: Optional[float] = None,
    top_n: int = 10,
) -> List[Tuple[int, int, int, float]]:
    """Find peaks in 3D correlation map.
    
    Args:
        corr_map: Correlation map (z, y, x)
        min_distance_vox: Minimum distance between peaks in voxels
        threshold: Minimum correlation value (if None, use top_n)
        top_n: Number of top peaks to return
        
    Returns:
        List of (z, y, x, correlation) tuples
    """
    from scipy.ndimage import maximum_filter
    
    # Apply maximum filter for non-maximum suppression
    max_filtered = maximum_filter(corr_map, size=2*min_distance_vox + 1)
    peaks_mask = (corr_map == max_filtered) & (corr_map > (threshold if threshold is not None else -np.inf))
    
    peak_indices = np.argwhere(peaks_mask)
    peak_values = corr_map[peaks_mask]
    
    # Sort by correlation value
    sorted_idx = np.argsort(-peak_values)
    
    # Take top N, ensuring minimum distance
    selected_peaks = []
    for idx in sorted_idx:
        z, y, x = peak_indices[idx]
        corr_val = peak_values[idx]
        
        # Check distance to already selected peaks
        too_close = False
        for sz, sy, sx, _ in selected_peaks:
            dist = np.sqrt((z - sz)**2 + (y - sy)**2 + (x - sx)**2)
            if dist < min_distance_vox:
                too_close = True
                break
        
        if not too_close:
            selected_peaks.append((int(z), int(y), int(x), float(corr_val)))
            if len(selected_peaks) >= top_n:
                break
    
    return selected_peaks


def compute_atom_inclusion_score(
    atoms_xyz: np.ndarray,
    density_map: MapVolume,
    threshold: Optional[float] = None,
) -> Tuple[float, int, int]:
    """Compute atom inclusion score based on density map.
    
    Args:
        atoms_xyz: Atom coordinates in Å (N, 3) with x, y, z
        density_map: Target density map
        threshold: Density threshold (if None, use mean + 1*std)
        
    Returns:
        (inclusion_score, n_atoms_in_density, n_atoms_total)
        inclusion_score: fraction of atoms in density (0-1)
    """
    if len(atoms_xyz) == 0:
        return 0.0, 0, 0
    
    # Determine threshold
    if threshold is None:
        # Use mean + 1*std as dynamic threshold
        data = density_map.data_zyx
        threshold = float(np.mean(data) + np.std(data))
    
    # Convert atom positions to voxel coordinates
    apix = density_map.apix
    origin = density_map.origin_xyzA
    
    n_in_density = 0
    n_out_of_bounds = 0
    
    for x, y, z in atoms_xyz:
        # Convert to voxel (Z, Y, X)
        vx = (x - origin[0]) / apix
        vy = (y - origin[1]) / apix
        vz = (z - origin[2]) / apix
        
        # Check bounds
        z_shape, y_shape, x_shape = density_map.data_zyx.shape
        if not (0 <= vx < x_shape and 0 <= vy < y_shape and 0 <= vz < z_shape):
            n_out_of_bounds += 1
            continue
        
        # Sample density at atom position (nearest neighbor for speed)
        vz_int = int(np.round(vz))
        vy_int = int(np.round(vy))
        vx_int = int(np.round(vx))
        
        if 0 <= vz_int < z_shape and 0 <= vy_int < y_shape and 0 <= vx_int < x_shape:
            density_val = density_map.data_zyx[vz_int, vy_int, vx_int]
            if density_val >= threshold:
                n_in_density += 1
    
    n_total = len(atoms_xyz)
    inclusion_score = n_in_density / n_total if n_total > 0 else 0.0
    
    return inclusion_score, n_in_density, n_total


def apply_transformation(
    atoms_xyz: np.ndarray,
    rotation: np.ndarray,  # quaternion (w, x, y, z)
    translation: np.ndarray,  # (3,) in Å
) -> np.ndarray:
    """Apply rotation and translation to atom coordinates.
    
    Args:
        atoms_xyz: Atom coordinates (N, 3) in Å
        rotation: Quaternion (w, x, y, z)
        translation: Translation vector (3,) in Å
        
    Returns:
        Transformed coordinates (N, 3)
    """
    R = quaternion_to_rotation_matrix(rotation)
    rotated = atoms_xyz @ R.T
    translated = rotated + translation
    return translated.astype(np.float32)


def rotate_map_3d(
    map_data: np.ndarray,
    quaternion: np.ndarray,
    center: Optional[np.ndarray] = None,
    order: int = 1,
) -> np.ndarray:
    """Rotate 3D map using quaternion via scipy.ndimage.rotate.
    
    Args:
        map_data: Map data (z, y, x)
        quaternion: Quaternion (w, x, y, z)
        center: Rotation center in voxels (z, y, x). If None, use map center.
        order: Interpolation order (0=nearest, 1=linear, 3=cubic)
        
    Returns:
        Rotated map (z, y, x)
    """
    from scipy.ndimage import rotate
    from scipy.spatial.transform import Rotation as R
    
    # Convert quaternion to Euler angles (scipy uses (x, y, z, w) order)
    rot = R.from_quat([quaternion[1], quaternion[2], quaternion[3], quaternion[0]])
    euler = rot.as_euler('ZYX', degrees=True)  # Z, Y, X rotations
    
    # Default center to map center
    if center is None:
        center = np.array([s / 2.0 for s in map_data.shape], dtype=np.float32)
    
    # Rotate around Z axis (first)
    rotated = rotate(map_data, euler[0], axes=(1, 2), reshape=False, order=order, mode='constant', cval=0.0)
    # Rotate around Y axis (second)
    rotated = rotate(rotated, euler[1], axes=(0, 2), reshape=False, order=order, mode='constant', cval=0.0)
    # Rotate around X axis (third)
    rotated = rotate(rotated, euler[2], axes=(0, 1), reshape=False, order=order, mode='constant', cval=0.0)
    
    return rotated.astype(np.float32)


def downsample_to_apix(
    map_data: np.ndarray,
    current_apix: float,
    target_apix: float,
) -> Tuple[np.ndarray, float]:
    """Downsample map to target Å/pixel using averaging.
    
    Args:
        map_data: Map data (z, y, x)
        current_apix: Current voxel size in Å
        target_apix: Target voxel size in Å
        
    Returns:
        (downsampled_map, actual_apix) where actual_apix may differ slightly
    """
    from scipy.ndimage import zoom
    
    if target_apix >= current_apix:
        # Upsampling not supported, return original
        return map_data.copy(), current_apix
    
    # Compute zoom factor
    zoom_factor = current_apix / target_apix
    
    # Downsample using zoom (averaging)
    downsampled = zoom(map_data, zoom_factor, order=1, mode='constant', cval=0.0)
    
    # Actual apix may differ slightly due to rounding
    actual_apix = current_apix / zoom_factor
    
    return downsampled.astype(np.float32), actual_apix


def compute_cross_correlation_fft_translation(
    probe_map: np.ndarray,
    target_map: np.ndarray,
    translation_vox: np.ndarray,
) -> float:
    """Compute cross-correlation at specific translation using FFT phase shift.
    
    Args:
        probe_map: Probe density map (z, y, x)
        target_map: Target density map (z, y, x)
        translation_vox: Translation in voxels (z, y, x)
        
    Returns:
        Correlation value at this translation
    """
    # Normalize
    probe_norm = probe_map - np.mean(probe_map)
    probe_std = np.std(probe_norm)
    if probe_std > 1e-6:
        probe_norm /= probe_std
    
    target_norm = target_map - np.mean(target_map)
    target_std = np.std(target_norm)
    if target_std > 1e-6:
        target_norm /= target_std
    
    # FFT
    probe_fft = np.fft.fftn(probe_norm)
    target_fft = np.fft.fftn(target_norm)
    
    # Apply phase shift for translation
    z, y, x = probe_map.shape
    kz, ky, kx = np.meshgrid(
        np.fft.fftfreq(z),
        np.fft.fftfreq(y),
        np.fft.fftfreq(x),
        indexing='ij'
    )
    
    phase = np.exp(-2j * np.pi * (
        kz * translation_vox[0] +
        ky * translation_vox[1] +
        kx * translation_vox[2]
    ))
    
    probe_fft_shifted = probe_fft * phase
    
    # Cross-correlation
    corr_fft = np.conj(probe_fft_shifted) * target_fft
    corr = np.fft.ifftn(corr_fft).real
    
    # Get correlation at origin (after translation)
    corr_val = corr[0, 0, 0] if corr.size > 0 else 0.0
    
    # Normalize
    norm = np.sqrt(np.sum(probe_norm**2) * np.sum(target_norm**2) + 1e-10)
    if norm > 1e-6:
        corr_val /= norm
    
    return float(corr_val)


def compute_optimal_angular_step(
    map_size_vox: int,
    apix: float,
    resolution_A: float,
    probe_size_A: float = 50.0,  # Approximate probe size in Å
) -> float:
    """
    Compute optimal angular step based on map size, resolution, and Å/pixel.
    
    The angular step should be small enough to avoid missing peaks but large enough
    to be computationally feasible. Based on the Nyquist criterion for rotation space.
    
    Args:
        map_size_vox: Size of map in voxels (max dimension)
        apix: Voxel size in Å
        resolution_A: Map resolution in Å
        probe_size_A: Approximate size of probe in Å
        
    Returns:
        Optimal angular step in degrees
    """
    # Angular resolution needed: at resolution R, we need to sample rotations
    # such that the probe doesn't move more than ~resolution/2 between samples
    # At distance D from rotation center, angular step θ should satisfy:
    # D * sin(θ) ≈ resolution/2
    
    # Use probe size as characteristic distance
    D = probe_size_A / 2.0  # Half the probe size
    
    # Angular step needed: θ ≈ arcsin(resolution / (2*D))
    # But we also need to consider map sampling
    # If map is sampled at apix, we need angular step such that:
    # probe_size * sin(θ) ≈ apix (to avoid aliasing)
    
    # Use the more restrictive criterion
    angular_resolution_1 = np.arcsin(resolution_A / (2.0 * D)) * 180.0 / np.pi
    angular_resolution_2 = np.arcsin(apix / D) * 180.0 / np.pi
    
    # Take the smaller (more restrictive) step
    optimal_step = min(angular_resolution_1, angular_resolution_2)
    
    # Clamp to reasonable range (5° to 30°)
    optimal_step = max(5.0, min(30.0, optimal_step))
    
    # For very large maps, we can use larger steps (less detail needed)
    if map_size_vox > 200:
        optimal_step *= 1.5
    if map_size_vox > 400:
        optimal_step *= 1.5
    
    return float(optimal_step)


def find_broad_peaks(
    candidates: List[Dict],
    peak_width_deg: float = 10.0,
    min_peak_correlation: float = 0.3,
    max_peaks: int = 20,
) -> List[Dict]:
    """
    Find broad peaks in candidate results by clustering nearby rotations.
    
    Groups candidates that are within peak_width_deg of each other and returns
    the best candidate from each cluster.
    
    Args:
        candidates: List of candidate dicts with 'rotation', 'correlation', etc.
        peak_width_deg: Angular width for peak clustering (degrees)
        min_peak_correlation: Minimum correlation to consider
        max_peaks: Maximum number of peaks to return
        
    Returns:
        List of peak candidates (one per cluster)
    """
    from scipy.spatial.transform import Rotation as R
    
    if not candidates:
        return []
    
    # Filter by minimum correlation
    filtered = [c for c in candidates if c['correlation'] >= min_peak_correlation]
    if not filtered:
        return []
    
    # Sort by correlation
    filtered.sort(key=lambda x: x['correlation'], reverse=True)
    
    # Cluster by rotation similarity
    peaks = []
    used = set()
    
    for i, cand in enumerate(filtered):
        if i in used:
            continue
        
        # Start a new peak cluster
        cluster = [cand]
        used.add(i)
        
        # Find all candidates within peak_width_deg
        q1 = cand['rotation']
        r1 = R.from_quat([q1[1], q1[2], q1[3], q1[0]])
        
        for j, other in enumerate(filtered):
            if j <= i or j in used:
                continue
            
            q2 = other['rotation']
            r2 = R.from_quat([q2[1], q2[2], q2[3], q2[0]])
            
            # Compute angular distance
            r_diff = r1.inv() * r2
            angle_deg = np.abs(r_diff.as_rotvec()) * 180.0 / np.pi
            max_angle = np.max(angle_deg)
            
            if max_angle <= peak_width_deg:
                cluster.append(other)
                used.add(j)
        
        # Take best candidate from cluster
        best_in_cluster = max(cluster, key=lambda x: x['correlation'])
        peaks.append(best_in_cluster)
        
        if len(peaks) >= max_peaks:
            break
    
    return peaks


def sample_rotations_around(
    center_rotation: np.ndarray,
    angle_range_deg: float,
    angle_step_deg: float,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Sample rotations within angle_range_deg of center_rotation.
    
    Args:
        center_rotation: Center quaternion (w, x, y, z)
        angle_range_deg: Angular range to sample (degrees)
        angle_step_deg: Angular step size (degrees)
        rng: Random number generator
        
    Returns:
        Array of quaternions (n_samples, 4)
    """
    from scipy.spatial.transform import Rotation as R
    
    if rng is None:
        rng = np.random.default_rng()
    
    # Convert center to Rotation object
    center_rot = R.from_quat([center_rotation[1], center_rotation[2], center_rotation[3], center_rotation[0]])
    
    # Number of samples based on angle range and step
    n_samples = int(np.ceil((angle_range_deg * 2) / angle_step_deg))
    n_samples = max(10, min(500, n_samples))  # Clamp to reasonable range
    
    quaternions = []
    for _ in range(n_samples):
        # Random axis
        axis = rng.uniform(-1, 1, 3)
        axis = axis / (np.linalg.norm(axis) + 1e-10)
        
        # Random angle within range
        angle_deg = rng.uniform(-angle_range_deg, angle_range_deg)
        angle_rad = np.deg2rad(angle_deg)
        
        # Create rotation around axis
        delta_rot = R.from_rotvec(axis * angle_rad)
        
        # Apply to center
        combined_rot = center_rot * delta_rot
        q = combined_rot.as_quat()
        quaternions.append(np.array([q[3], q[0], q[1], q[2]], dtype=np.float32))
    
    return np.array(quaternions)


def search_single_rotation(
    probe_rotated: np.ndarray,
    target_map: np.ndarray,
    translation_step_A: float,
    apix: float,
) -> List[Tuple[np.ndarray, float]]:
    """Search translations for a single rotated probe.
    
    Args:
        probe_rotated: Rotated probe map (z, y, x)
        target_map: Target map (z, y, x)
        translation_step_A: Translation step in Å
        apix: Voxel size in Å
        
    Returns:
        List of (translation_vox, correlation) tuples
        translation_vox is in voxels (z, y, x) relative to map origin
    """
    # Compute full correlation map using FFT
    corr_map = compute_cross_correlation_fft(probe_rotated, target_map)
    
    # Find peaks
    translation_step_vox = translation_step_A / apix
    peaks = find_peaks_3d(
        corr_map,
        min_distance_vox=max(1, int(translation_step_vox)),
        top_n=50,  # Get more candidates for refinement
    )
    
    results = []
    for z, y, x, corr_val in peaks:
        # Translation in voxels (z, y, x) - these are already relative to map grid
        translation_vox = np.array([z, y, x], dtype=np.float32)
        results.append((translation_vox, float(corr_val)))
    
    return results


def foldhunter_search(
    probe_pdb: Optional[Path],
    probe_map: Optional[Path],
    target_map: Path,
    resolution_A: float,
    plddt_threshold: float = 0.5,
    density_threshold: Optional[float] = None,
    coarse_angle_step: Optional[float] = None,  # Auto-compute if None
    fine_angle_step: float = 1.0,  # degrees
    coarse_translation_step: float = 5.0,  # Å
    fine_translation_step: float = 1.0,  # Å
    n_coarse_rotations: Optional[int] = None,  # Auto-compute if None
    symmetry: Optional[int] = None,  # rotational symmetry (e.g., 3 for C3)
    out_of_bounds_penalty_weight: float = 0.1,
    top_n_candidates: int = 10,
    peak_width_deg: float = 10.0,  # Angular width for peak clustering
    max_peaks_coarse: int = 20,  # Max peaks from coarse round
) -> List[FoldHunterResult]:
    """
    Run FoldHunter exhaustive search with progressive subsampling.
    
    Implements a 3-stage progressive subsampling strategy:
    1. Coarse: Downsample to 3-4 Å/pixel, exhaustive search with smart angular step
    2. Medium: Scale to 2-3 Å/pixel, search +/-30° around top candidates from round 1
    3. Fine: Original map, search +/-5° with 1° steps around top candidates from round 2
    
    Args:
        probe_pdb: Path to probe PDB file (AlphaFold model)
        probe_map: Path to probe MRC file (alternative to PDB)
        target_map: Path to target density map
        resolution_A: Resolution for probe map generation
        plddt_threshold: Minimum pLDDT to include atoms (default 0.5)
        density_threshold: Density threshold for inclusion scoring (None = dynamic)
        coarse_angle_step: Coarse rotation sampling step (degrees, auto if None)
        fine_angle_step: Fine rotation sampling step (degrees, default 1.0)
        coarse_translation_step: Coarse translation step (Å)
        fine_translation_step: Fine translation step (Å)
        n_coarse_rotations: Number of rotations for coarse search (auto if None)
        symmetry: Rotational symmetry (e.g., 3 for C3 symmetry)
        out_of_bounds_penalty_weight: Weight for out-of-bounds penalty
        top_n_candidates: Number of top candidates to return
        peak_width_deg: Angular width for peak clustering (degrees)
        max_peaks_coarse: Maximum peaks from coarse round
        
    Returns:
        List of FoldHunterResult objects sorted by correlation
    """
    # Load target map
    target_mv = read_map(target_map)
    target_shape = target_mv.data_zyx.shape
    target_origin = target_mv.origin_xyzA
    target_apix = target_mv.apix
    
    print(f"Target map shape: {target_shape}, apix: {target_apix:.3f} Å/voxel, resolution: {resolution_A:.2f} Å")
    
    # Generate or load probe map
    if probe_pdb is not None:
        probe_map_data = pdb_to_density_map(
            probe_pdb,
            target_apix,
            target_shape,
            target_origin,
            resolution_A,
            plddt_threshold,
        )
        # Load original PDB for atom inclusion scoring
        atoms_xyz_original = read_model_xyz(str(probe_pdb), remove_hydrogens=True)
    elif probe_map is not None:
        probe_mv = read_map(probe_map)
        probe_map_data = probe_mv.data_zyx
        atoms_xyz_original = None
    else:
        raise ValueError("Must provide either probe_pdb or probe_map")
    
    rng = np.random.default_rng(42)  # Deterministic seed
    
    # ========================================================================
    # ROUND 1: COARSE SEARCH (3-4 Å/pixel)
    # ========================================================================
    print("\n" + "="*70)
    print("ROUND 1: COARSE SEARCH (3-4 Å/pixel)")
    print("="*70)
    
    target_apix_coarse = 3.5  # Target 3.5 Å/pixel for coarse search
    target_coarse, actual_apix_coarse = downsample_to_apix(target_mv.data_zyx, target_apix, target_apix_coarse)
    probe_coarse, _ = downsample_to_apix(probe_map_data, target_apix, target_apix_coarse)
    
    print(f"Coarse maps: shape={target_coarse.shape}, apix={actual_apix_coarse:.3f} Å/voxel")
    
    # Compute optimal angular step for coarse search
    max_dim_coarse = max(target_coarse.shape)
    if coarse_angle_step is None:
        coarse_angle_step = compute_optimal_angular_step(
            max_dim_coarse, actual_apix_coarse, resolution_A
        )
    print(f"Optimal coarse angular step: {coarse_angle_step:.2f}°")
    
    # Compute number of rotations needed for exhaustive search
    if n_coarse_rotations is None:
        # Estimate rotations needed to cover SO(3) with this step
        # Rough estimate: ~(360/step)^3 rotations for full coverage
        n_coarse_rotations = int((360.0 / coarse_angle_step) ** 2.5)  # Slightly less than cubic
        n_coarse_rotations = max(500, min(5000, n_coarse_rotations))  # Clamp to reasonable range
    
    # Adjust for symmetry
    n_rotations = n_coarse_rotations
    if symmetry is not None:
        n_rotations = n_coarse_rotations // symmetry
        print(f"Using symmetry {symmetry}: searching {360/symmetry:.1f}° rotation space")
    
    print(f"Sampling {n_rotations} rotations for exhaustive coarse search...")
    
    coarse_quaternions = sample_quaternions_uniform(n_rotations, rng)
    
    # If symmetry, generate symmetric rotations
    if symmetry is not None:
        from scipy.spatial.transform import Rotation as R
        symmetric_quaternions = []
        for q in coarse_quaternions:
            rot = R.from_quat([q[1], q[2], q[3], q[0]])
            for sym_i in range(symmetry):
                sym_angle = 360.0 * sym_i / symmetry
                sym_rot = R.from_euler('z', sym_angle, degrees=True)
                combined = rot * sym_rot
                q_sym = combined.as_quat()
                symmetric_quaternions.append(np.array([q_sym[3], q_sym[0], q_sym[1], q_sym[2]], dtype=np.float32))
        coarse_quaternions = np.array(symmetric_quaternions)
        print(f"Generated {len(coarse_quaternions)} rotations with symmetry")
    
    # Search each rotation
    coarse_candidates = []
    for i, q in enumerate(coarse_quaternions):
        if (i + 1) % max(1, len(coarse_quaternions) // 10) == 0:
            print(f"  Processing rotation {i+1}/{len(coarse_quaternions)}...")
        
        # Rotate probe map
        probe_rotated = rotate_map_3d(probe_coarse, q, order=1)
        
        # Search translations
        translation_results = search_single_rotation(
            probe_rotated,
            target_coarse,
            coarse_translation_step,
            actual_apix_coarse,
        )
        
        # Store candidates (scale translation back to original apix)
        for translation_vox_coarse, corr_val in translation_results:
            # Scale translation to original apix
            scale_factor = target_apix / actual_apix_coarse
            translation_vox_full = translation_vox_coarse * scale_factor
            
            coarse_candidates.append({
                'rotation': q,
                'translation_vox': translation_vox_full,
                'correlation': corr_val,
            })
    
    print(f"Found {len(coarse_candidates)} coarse candidates")
    
    # Find broad peaks in coarse results
    coarse_peaks = find_broad_peaks(
        coarse_candidates,
        peak_width_deg=peak_width_deg,
        min_peak_correlation=0.2,  # Lower threshold for coarse
        max_peaks=max_peaks_coarse,
    )
    print(f"Identified {len(coarse_peaks)} broad peaks from coarse search")
    
    # ========================================================================
    # ROUND 2: MEDIUM SEARCH (2-3 Å/pixel, +/-30° around coarse peaks)
    # ========================================================================
    print("\n" + "="*70)
    print("ROUND 2: MEDIUM SEARCH (2-3 Å/pixel, +/-30° around coarse peaks)")
    print("="*70)
    
    target_apix_medium = 2.5  # Target 2.5 Å/pixel for medium search
    target_medium, actual_apix_medium = downsample_to_apix(target_mv.data_zyx, target_apix, target_apix_medium)
    probe_medium, _ = downsample_to_apix(probe_map_data, target_apix, target_apix_medium)
    
    print(f"Medium maps: shape={target_medium.shape}, apix={actual_apix_medium:.3f} Å/voxel")
    
    medium_angle_range = 30.0  # +/-30 degrees
    medium_angle_step = 3.0  # 3° steps for medium search
    
    medium_candidates = []
    for i, coarse_peak in enumerate(coarse_peaks):
        print(f"  Searching around coarse peak {i+1}/{len(coarse_peaks)}...")
        
        # Sample rotations around this peak
        medium_quaternions = sample_rotations_around(
            coarse_peak['rotation'],
            medium_angle_range,
            medium_angle_step,
            rng,
        )
        
        # Scale translation to medium apix
        scale_factor = actual_apix_medium / target_apix
        translation_vox_medium = coarse_peak['translation_vox'] * scale_factor
        
        for q in medium_quaternions:
            # Rotate probe map
            probe_rotated = rotate_map_3d(probe_medium, q, order=1)
            
            # Search translations around initial
            translation_results = search_single_rotation(
                probe_rotated,
                target_medium,
                coarse_translation_step * 0.7,  # Slightly finer translation step
                actual_apix_medium,
            )
            
            for translation_vox_m, corr_val in translation_results:
                # Scale translation back to original apix
                scale_factor_back = target_apix / actual_apix_medium
                translation_vox_full = translation_vox_m * scale_factor_back
                
                medium_candidates.append({
                    'rotation': q,
                    'translation_vox': translation_vox_full,
                    'correlation': corr_val,
                })
    
    print(f"Found {len(medium_candidates)} medium candidates")
    
    # Find broad peaks in medium results
    medium_peaks = find_broad_peaks(
        medium_candidates,
        peak_width_deg=5.0,  # Tighter clustering for medium
        min_peak_correlation=0.3,
        max_peaks=min(20, len(coarse_peaks) * 2),
    )
    print(f"Identified {len(medium_peaks)} peaks from medium search")
    
    # ========================================================================
    # ROUND 3: FINE SEARCH (original map, +/-5° with 1° steps)
    # ========================================================================
    print("\n" + "="*70)
    print("ROUND 3: FINE SEARCH (original map, +/-5° with 1° steps)")
    print("="*70)
    
    fine_angle_range = 5.0  # +/-5 degrees
    fine_angle_step = fine_angle_step  # Use provided fine step (default 1°)
    
    final_results = []
    for i, medium_peak in enumerate(medium_peaks):
        print(f"  Refining peak {i+1}/{len(medium_peaks)}...")
        
        # Sample rotations around this peak
        fine_quaternions = sample_rotations_around(
            medium_peak['rotation'],
            fine_angle_range,
            fine_angle_step,
            rng,
        )
        
        for q in fine_quaternions:
            # Rotate probe map (full resolution)
            probe_rotated = rotate_map_3d(probe_map_data, q, order=1)
            
            # Search translations
            translation_results = search_single_rotation(
                probe_rotated,
                target_mv.data_zyx,
                fine_translation_step,
                target_apix,
            )
            
            for translation_vox, corr_val in translation_results:
                # Convert translation to Å
                translation_A = np.array([
                    translation_vox[2] * target_apix + target_origin[0],
                    translation_vox[1] * target_apix + target_origin[1],
                    translation_vox[0] * target_apix + target_origin[2],
                ], dtype=np.float32)
                
                # Compute atom inclusion score if available
                if atoms_xyz_original is not None:
                    transformed_atoms = apply_transformation(atoms_xyz_original, q, translation_A)
                    inclusion_score, n_in, n_total = compute_atom_inclusion_score(
                        transformed_atoms, target_mv, density_threshold
                    )
                    n_out = n_total - n_in
                    out_of_bounds_penalty = out_of_bounds_penalty_weight * (n_out / n_total) if n_total > 0 else 0.0
                else:
                    inclusion_score = 0.0
                    n_in = 0
                    n_total = 0
                    out_of_bounds_penalty = 0.0
                
                final_results.append(FoldHunterResult(
                    translation=translation_A,
                    rotation=q,
                    correlation=float(corr_val),
                    atom_inclusion_score=float(inclusion_score),
                    n_atoms_in_density=n_in,
                    n_atoms_total=n_total,
                    out_of_bounds_penalty=float(out_of_bounds_penalty),
                ))
    
    # Sort by combined score
    final_results.sort(key=lambda r: r.correlation - r.out_of_bounds_penalty, reverse=True)
    
    print(f"\nFinal results: {len(final_results)} candidates")
    print(f"Top {min(top_n_candidates, len(final_results))} candidates:")
    for i, result in enumerate(final_results[:top_n_candidates], 1):
        print(f"  {i}. Correlation: {result.correlation:.4f}, "
              f"Inclusion: {result.atom_inclusion_score:.2%}, "
              f"Score: {result.correlation - result.out_of_bounds_penalty:.4f}")
    
    return final_results[:top_n_candidates]
