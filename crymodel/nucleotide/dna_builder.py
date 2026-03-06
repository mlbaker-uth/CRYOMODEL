from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import gemmi
from scipy.spatial.transform import Rotation

from ..io.mrc import read_map


@dataclass
class TemplatePair:
    atoms_a: List[Tuple[str, np.ndarray]]  # (atom_name, coords)
    atoms_b: List[Tuple[str, np.ndarray]]
    axis: np.ndarray
    centroid: np.ndarray


def _resname_for_base(base: str) -> str:
    base = base.upper()
    if base == "A":
        return "DA"
    if base == "T":
        return "DT"
    if base == "G":
        return "DG"
    if base == "C":
        return "DC"
    return "DA"


def _load_template_pair(template_pdb: Path) -> TemplatePair:
    st = gemmi.read_structure(str(template_pdb))
    model = st[0]
    chain_a = model[0] if model else None
    if chain_a is None:
        raise ValueError("Template PDB missing chains")

    # Expect two chains; pick first two
    chain_ids = [chain.name for chain in model]
    if len(chain_ids) < 2:
        raise ValueError("Template PDB must contain two chains")

    chain_a = model[chain_ids[0]]
    chain_b = model[chain_ids[1]]

    # Use first residue from chain A and last from chain B (antiparallel)
    res_a = next((r for r in chain_a if r), None)
    res_b = next((r for r in reversed(chain_b) if r), None)
    if res_a is None or res_b is None:
        raise ValueError("Template PDB missing residues for base pair")

    atoms_a = [(atom.name, np.array([atom.pos.x, atom.pos.y, atom.pos.z], dtype=np.float32)) for atom in res_a]
    atoms_b = [(atom.name, np.array([atom.pos.x, atom.pos.y, atom.pos.z], dtype=np.float32)) for atom in res_b]

    all_coords = np.array([a[1] for a in atoms_a + atoms_b], dtype=np.float32)
    centroid = np.mean(all_coords, axis=0)
    centered = all_coords - centroid
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    axis = eigvecs[:, np.argmax(eigvals)]
    axis = axis / (np.linalg.norm(axis) + 1e-8)

    # Recenter atoms for placement
    atoms_a_centered = [(name, coords - centroid) for name, coords in atoms_a]
    atoms_b_centered = [(name, coords - centroid) for name, coords in atoms_b]

    return TemplatePair(atoms_a_centered, atoms_b_centered, axis=axis, centroid=centroid)


def _pdb_density_from_atoms(
    atoms_xyz: np.ndarray,
    target_shape: Tuple[int, int, int],
    apix: float,
    resolution_A: float,
) -> np.ndarray:
    density = np.zeros(target_shape, dtype=np.float32)
    z_shape, y_shape, x_shape = target_shape
    center = np.array([(x_shape - 1) / 2.0, (y_shape - 1) / 2.0, (z_shape - 1) / 2.0], dtype=np.float32)

    atom_radius_A = 1.5
    radius_vox = atom_radius_A / apix
    sigma_vox = radius_vox / 2.0
    kernel_radius = int(np.ceil(3 * sigma_vox))
    if kernel_radius > 0:
        k_range = np.arange(-kernel_radius, kernel_radius + 1, dtype=np.float32)
        KZ, KY, KX = np.meshgrid(k_range, k_range, k_range, indexing="ij")
        kernel = np.exp(-(KZ**2 + KY**2 + KX**2) / (2 * sigma_vox**2))
        kernel /= np.sum(kernel)

    for x, y, z in atoms_xyz:
        vx = center[0] + x / apix
        vy = center[1] + y / apix
        vz = center[2] + z / apix

        vx_int = int(np.round(vx))
        vy_int = int(np.round(vy))
        vz_int = int(np.round(vz))

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
            density[z0:z1, y0:y1, x0:x1] += kernel[kz0:kz1, ky0:ky1, kx0:kx1]

    # Apply Gaussian blur to simulate resolution
    resolution_sigma_vox = (0.939 * resolution_A) / (2.355 * apix)
    if resolution_sigma_vox > 0.5:
        from scipy.ndimage import gaussian_filter
        density = gaussian_filter(density, sigma=resolution_sigma_vox, mode="constant")

    return density.astype(np.float32)


def _find_peak_positions(
    volume: np.ndarray,
    template: np.ndarray,
    n_peaks: int,
    min_distance_vox: int,
) -> List[Tuple[int, int, int]]:
    vol_fft = np.fft.fftn(volume)
    temp_fft = np.fft.fftn(template)
    corr = np.fft.ifftn(np.conj(vol_fft) * temp_fft).real

    peaks = []
    corr_work = corr.copy()
    for _ in range(n_peaks):
        idx = np.unravel_index(np.argmax(corr_work), corr_work.shape)
        peaks.append(idx)
        z0 = max(0, idx[0] - min_distance_vox)
        z1 = min(corr_work.shape[0], idx[0] + min_distance_vox + 1)
        y0 = max(0, idx[1] - min_distance_vox)
        y1 = min(corr_work.shape[1], idx[1] + min_distance_vox + 1)
        x0 = max(0, idx[2] - min_distance_vox)
        x1 = min(corr_work.shape[2], idx[2] + min_distance_vox + 1)
        corr_work[z0:z1, y0:y1, x0:x1] = -np.inf

    return peaks


def _order_positions(positions_xyz: List[np.ndarray]) -> List[np.ndarray]:
    if len(positions_xyz) <= 2:
        return positions_xyz
    coords = np.vstack(positions_xyz)
    # Farthest pair as endpoints
    dists = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
    i, j = np.unravel_index(np.argmax(dists), dists.shape)
    start = i
    remaining = set(range(len(positions_xyz)))
    ordered = [coords[start]]
    remaining.remove(start)
    current = coords[start]
    while remaining:
        next_idx = min(remaining, key=lambda k: np.linalg.norm(coords[k] - current))
        ordered.append(coords[next_idx])
        current = coords[next_idx]
        remaining.remove(next_idx)
    return [c for c in ordered]


def build_poly_at_dna(
    map_path: Path,
    n_basepairs: int,
    threshold: float,
    out_pdb: Path,
    template_pdb: Path,
    sequence_a: Optional[str] = None,
    sequence_b: Optional[str] = None,
    min_distance_vox: int = 4,
    resolution_A: float = 3.0,
    output_swapped: bool = False,
    out_pdb_swapped: Optional[Path] = None,
) -> List[Path]:
    mv = read_map(str(map_path))
    data = mv.data_zyx
    apix = mv.apix

    data_thr = np.where(data >= threshold, data, 0.0).astype(np.float32)

    template_pair = _load_template_pair(template_pdb)
    template_atoms = np.array([a[1] for a in template_pair.atoms_a + template_pair.atoms_b], dtype=np.float32)
    template_density = _pdb_density_from_atoms(template_atoms, data.shape, apix, resolution_A)

    peaks = _find_peak_positions(data_thr, template_density, n_basepairs, min_distance_vox=min_distance_vox)
    positions_xyz = []
    for z, y, x in peaks:
        pos = np.array([x * apix, y * apix, z * apix], dtype=np.float32) + mv.origin_xyzA
        positions_xyz.append(pos)

    ordered = _order_positions(positions_xyz)

    if sequence_a is None:
        sequence_a = "A" * n_basepairs
    if sequence_b is None:
        sequence_b = "T" * n_basepairs

    def _build_model(seq_a: str, seq_b: str, out_path: Path) -> Path:
        st = gemmi.Structure()
        st.cell = gemmi.UnitCell(1, 1, 1, 90, 90, 90)
        model = gemmi.Model("1")
        chainA = gemmi.Chain("A")
        chainB = gemmi.Chain("B")

        for idx, pos in enumerate(ordered, start=1):
            tangent = None
            if len(ordered) > 1:
                if idx == 1:
                    tangent = ordered[1] - ordered[0]
                elif idx == len(ordered):
                    tangent = ordered[-1] - ordered[-2]
                else:
                    tangent = ordered[idx] - ordered[idx - 2]
            if tangent is None or np.linalg.norm(tangent) < 1e-6:
                tangent = template_pair.axis
            else:
                tangent = tangent / (np.linalg.norm(tangent) + 1e-8)

            rot = Rotation.align_vectors([tangent], [template_pair.axis])[0]

            resA = gemmi.Residue()
            resA.name = _resname_for_base(seq_a[idx - 1])
            resA.seqid = gemmi.SeqId(idx)
            for name, coords in template_pair.atoms_a:
                rotated = rot.apply(coords)
                atom = gemmi.Atom()
                atom.name = name
                atom.pos = gemmi.Position(
                    float(pos[0] + rotated[0]),
                    float(pos[1] + rotated[1]),
                    float(pos[2] + rotated[2]),
                )
                resA.add_atom(atom)
            chainA.add_residue(resA)

            resB = gemmi.Residue()
            resB.name = _resname_for_base(seq_b[idx - 1])
            resB.seqid = gemmi.SeqId(idx)
            for name, coords in template_pair.atoms_b:
                rotated = rot.apply(coords)
                atom = gemmi.Atom()
                atom.name = name
                atom.pos = gemmi.Position(
                    float(pos[0] + rotated[0]),
                    float(pos[1] + rotated[1]),
                    float(pos[2] + rotated[2]),
                )
                resB.add_atom(atom)
            chainB.add_residue(resB)

        model.add_chain(chainA)
        model.add_chain(chainB)
        st.add_model(model)
        st.write_pdb(str(out_path))
        return out_path

    outputs = [_build_model(sequence_a, sequence_b, out_pdb)]
    if output_swapped:
        if out_pdb_swapped is None:
            out_pdb_swapped = out_pdb.with_name(out_pdb.stem + "_swapped.pdb")
        outputs.append(_build_model(sequence_b, sequence_a, out_pdb_swapped))

    return outputs
