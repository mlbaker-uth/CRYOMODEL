from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import math
import re

import numpy as np
import gemmi
from scipy.spatial.transform import Rotation
from scipy.interpolate import RegularGridInterpolator

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

    # Use first residue from chain A and first from chain B
    res_a = next((r for r in chain_a if r), None)
    res_b = next((r for r in chain_b if r), None)
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


def _find_component_centroids(
    volume: np.ndarray,
    n_targets: int,
) -> List[Tuple[int, int, int]]:
    from scipy.ndimage import label

    if np.all(volume == 0):
        return []

    labeled, n_components = label(volume > 0)
    if n_components == 0:
        return []

    centroids = []
    for i in range(1, n_components + 1):
        mask = labeled == i
        if not np.any(mask):
            continue
        weights = volume[mask]
        idx = np.argwhere(mask)
        # density-weighted centroid in z,y,x
        wsum = np.sum(weights)
        if wsum <= 0:
            continue
        centroid = np.sum(idx * weights[:, None], axis=0) / wsum
        # Use component total density as score
        score = float(wsum)
        centroids.append((score, centroid))

    centroids.sort(key=lambda c: c[0], reverse=True)
    centroids = centroids[:n_targets]
    return [tuple(int(round(v)) for v in c[1]) for c in centroids]

def _build_mst(coords: np.ndarray) -> List[List[int]]:
    n = coords.shape[0]
    if n == 0:
        return []
    in_tree = [False] * n
    in_tree[0] = True
    edges = [[] for _ in range(n)]
    dist = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
    for _ in range(n - 1):
        best = None
        best_d = float("inf")
        for i in range(n):
            if not in_tree[i]:
                continue
            for j in range(n):
                if in_tree[j] or i == j:
                    continue
                if dist[i, j] < best_d:
                    best_d = dist[i, j]
                    best = (i, j)
        if best is None:
            break
        i, j = best
        in_tree[j] = True
        edges[i].append(j)
        edges[j].append(i)
    return edges


def _order_positions(positions_xyz: List[np.ndarray]) -> List[np.ndarray]:
    if len(positions_xyz) <= 2:
        return positions_xyz
    coords = np.vstack(positions_xyz)
    edges = _build_mst(coords)
    # Start from one end of the tree (node with smallest degree)
    degrees = [len(e) for e in edges]
    start = int(np.argmin(degrees))
    ordered = []
    visited = set()
    current = start
    while len(visited) < len(coords):
        ordered.append(coords[current])
        visited.add(current)
        # Prefer unvisited neighbors, else jump to nearest unvisited node
        unvisited_neighbors = [n for n in edges[current] if n not in visited]
        if unvisited_neighbors:
            next_idx = min(unvisited_neighbors, key=lambda k: np.linalg.norm(coords[k] - coords[current]))
            current = next_idx
            continue
        remaining = [i for i in range(len(coords)) if i not in visited]
        if not remaining:
            break
        current = min(remaining, key=lambda k: np.linalg.norm(coords[k] - coords[current]))
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

    peaks = _find_component_centroids(data_thr, n_basepairs)
    if len(peaks) < n_basepairs:
        extra = _find_peak_positions(
            data_thr,
            template_density,
            n_basepairs - len(peaks),
            min_distance_vox=min_distance_vox,
        )
        peaks = peaks + extra
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
            resA.seqid = gemmi.SeqId(str(idx))
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
            resB.seqid = gemmi.SeqId(str(idx))
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


SUGAR_PHOS = {"P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'"}
BB_SCORE = {"P", "OP1", "OP2", "O5'", "C5'", "C4'", "C3'", "O3'"}


def _2bp_read_pdb_atoms(path: Path):
    atoms = []
    for line in Path(path).read_text().splitlines():
        if line.startswith(("ATOM", "HETATM")):
            name = line[12:16].strip()
            element = line[76:78].strip() or re.sub(r"[^A-Za-z]", "", name)[:1].upper() or "C"
            atoms.append(
                {
                    "record": line[:6].strip(),
                    "serial": int(line[6:11]),
                    "name": name,
                    "resname": line[17:20].strip(),
                    "chain": (line[21].strip() or "A"),
                    "resseq": int(line[22:26]),
                    "icode": line[26],
                    "coord": np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])], dtype=float),
                    "occ": float(line[54:60]) if line[54:60].strip() else 1.0,
                    "bfac": float(line[60:66]) if line[60:66].strip() else 0.0,
                    "element": element.upper(),
                }
            )
    if not atoms:
        raise ValueError(f"No ATOM/HETATM records found in {path}")
    return atoms


def _2bp_read_polyline_pdb(path: Path) -> np.ndarray:
    pts = []
    for line in Path(path).read_text().splitlines():
        if line.startswith(("ATOM", "HETATM")):
            pts.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    pts = np.asarray(pts, dtype=float)
    if len(pts) < 2:
        raise ValueError("Centerline PDB must contain at least 2 points.")
    return pts


def _2bp_load_map(path: Path):
    ccp4 = gemmi.read_ccp4_map(str(path))
    ccp4.setup(0.0)
    arr = np.array(ccp4.grid, copy=False)
    vx = ccp4.grid.unit_cell.a / arr.shape[0]
    vy = ccp4.grid.unit_cell.b / arr.shape[1]
    vz = ccp4.grid.unit_cell.c / arr.shape[2]
    x = np.arange(arr.shape[0]) * vx
    y = np.arange(arr.shape[1]) * vy
    z = np.arange(arr.shape[2]) * vz
    interp = RegularGridInterpolator((x, y, z), arr, bounds_error=False, fill_value=float(arr.min()))
    return arr, np.array([vx, vy, vz], dtype=float), interp


def _2bp_polyline_arclength(points: np.ndarray) -> np.ndarray:
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)])


def _2bp_interp_polyline(points: np.ndarray, s_query: np.ndarray, s=None) -> np.ndarray:
    if s is None:
        s = _2bp_polyline_arclength(points)
    s_query = np.asarray(s_query, dtype=float)
    out = np.zeros((len(s_query), 3), dtype=float)
    idxs = np.searchsorted(s, s_query, side="right") - 1
    idxs = np.clip(idxs, 0, len(points) - 2)
    ds = s[idxs + 1] - s[idxs]
    frac = np.divide(s_query - s[idxs], ds, out=np.zeros_like(s_query), where=ds > 0)
    out[:] = points[idxs] * (1.0 - frac[:, None]) + points[idxs + 1] * frac[:, None]
    return out


def _2bp_resample_centerline(points, n_points=None, spacing=3.4, trim_start_A=0.0, trim_end_A=0.0):
    s = _2bp_polyline_arclength(points)
    total = s[-1]
    s0 = trim_start_A
    s1 = total - trim_end_A
    if s1 <= s0:
        raise ValueError("Trim too large for centerline length.")
    if n_points is None:
        n_points = max(2, int(round((s1 - s0) / spacing)) + 1)
    target_s = np.linspace(s0, s1, n_points)
    return _2bp_interp_polyline(points, target_s, s=s), target_s, total


def _2bp_tangents(points):
    t = np.zeros_like(points)
    t[0] = points[1] - points[0]
    t[-1] = points[-1] - points[-2]
    t[1:-1] = points[2:] - points[:-2]
    nrm = np.linalg.norm(t, axis=1)
    nrm[nrm == 0] = 1.0
    return t / nrm[:, None]


def _2bp_rotate_about_axis(v, axis, angle_rad):
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return v * c + np.cross(axis, v) * s + axis * np.dot(axis, v) * (1.0 - c)


def _2bp_parallel_transport_frames(points, initial_normal=None):
    t = _2bp_tangents(points)
    n = np.zeros_like(points)
    b = np.zeros_like(points)
    if initial_normal is None:
        ref = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(ref, t[0])) > 0.9:
            ref = np.array([0.0, 1.0, 0.0])
        n0 = ref - t[0] * np.dot(ref, t[0])
        n0 /= np.linalg.norm(n0)
    else:
        n0 = initial_normal - t[0] * np.dot(initial_normal, t[0])
        n0 /= np.linalg.norm(n0)
    n[0] = n0
    b[0] = np.cross(t[0], n0)
    b[0] /= np.linalg.norm(b[0])
    for i in range(1, len(points)):
        prev_t, cur_t = t[i - 1], t[i]
        axis = np.cross(prev_t, cur_t)
        axis_norm = np.linalg.norm(axis)
        cur_n = n[i - 1].copy()
        if axis_norm > 1e-8:
            axis /= axis_norm
            angle = math.acos(np.clip(np.dot(prev_t, cur_t), -1.0, 1.0))
            cur_n = _2bp_rotate_about_axis(cur_n, axis, angle)
        cur_n = cur_n - cur_t * np.dot(cur_n, cur_t)
        cur_n /= np.linalg.norm(cur_n)
        cur_b = np.cross(cur_t, cur_n)
        cur_b /= np.linalg.norm(cur_b)
        n[i] = cur_n
        b[i] = cur_b
    return t, n, b


def _2bp_group_residues(atoms):
    groups = defaultdict(list)
    for a in atoms:
        groups[(a["chain"], a["resseq"], a["resname"])].append(a)
    return groups


def _2bp_residue_base_center(res_atoms):
    pts = [a["coord"] for a in res_atoms if a["name"] not in SUGAR_PHOS]
    if not pts:
        pts = [a["coord"] for a in res_atoms]
    return np.mean(np.asarray(pts), axis=0)


def _2bp_basepair_frame(res1_atoms, res2_atoms):
    base_atoms = [a for a in res1_atoms + res2_atoms if a["name"] not in SUGAR_PHOS]
    base_coords = np.array([a["coord"] for a in base_atoms], dtype=float)
    origin = base_coords.mean(axis=0)
    xmat = base_coords - origin
    _, _, vt = np.linalg.svd(xmat, full_matrices=False)
    z_axis = vt[-1]
    z_axis /= np.linalg.norm(z_axis)
    c1 = _2bp_residue_base_center(res1_atoms)
    c2 = _2bp_residue_base_center(res2_atoms)
    x_axis = c2 - c1
    x_axis = x_axis - z_axis * np.dot(x_axis, z_axis)
    if np.linalg.norm(x_axis) < 1e-6:
        x_axis = vt[0]
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    return origin, np.column_stack([x_axis, y_axis, z_axis])


def _2bp_derive_template(template_atoms):
    groups = _2bp_group_residues(template_atoms)
    chains = sorted({k[0] for k in groups})
    if len(chains) != 2:
        raise ValueError("2-bp template must contain exactly two chains.")
    chainA, chainB = chains
    resA = sorted([k for k in groups if k[0] == chainA], key=lambda x: x[1])
    resB = sorted([k for k in groups if k[0] == chainB], key=lambda x: x[1])
    if len(resA) != 2 or len(resB) != 2:
        raise ValueError("2-bp template must contain exactly 2 residues per chain.")
    pair0 = (resA[0], resB[-1])
    pair1 = (resA[1], resB[0])
    o0, f0 = _2bp_basepair_frame(groups[pair0[0]], groups[pair0[1]])
    o1, f1 = _2bp_basepair_frame(groups[pair1[0]], groups[pair1[1]])
    step_origin = 0.5 * (o0 + o1)
    z_axis = o1 - o0
    z_axis /= np.linalg.norm(z_axis)
    x_axis = f0[:, 0] + f1[:, 0]
    x_axis = x_axis - z_axis * np.dot(x_axis, z_axis)
    if np.linalg.norm(x_axis) < 1e-6:
        x_axis = f0[:, 0] - z_axis * np.dot(f0[:, 0], z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    step_basis = np.column_stack([x_axis, y_axis, z_axis])
    return {
        "chainA": chainA,
        "chainB": chainB,
        "resA": resA,
        "resB": resB,
        "pair0": pair0,
        "pair1": pair1,
        "bp0_origin": o0,
        "bp1_origin": o1,
        "bp0_basis": f0,
        "bp1_basis": f1,
        "step_origin": step_origin,
        "step_basis": step_basis,
    }


def _2bp_transform_atoms(atoms, origin_src, basis_src, origin_tgt, basis_tgt):
    rot = basis_tgt @ basis_src.T
    out = []
    for atom in atoms:
        aa = atom.copy()
        aa["coord"] = rot @ (atom["coord"] - origin_src) + origin_tgt
        out.append(aa)
    return out


def _2bp_score_atoms_density(atoms, interp, use_backbone_only=False):
    if interp is None:
        return 0.0
    coords = []
    for a in atoms:
        if use_backbone_only and a["name"] not in BB_SCORE:
            continue
        coords.append(a["coord"])
    if not coords:
        return 0.0
    return float(np.mean(interp(np.asarray(coords))))


def _2bp_target_bp_frames(points, nvec, bvec, twist_deg, phase_deg):
    t = _2bp_tangents(points)
    bases = []
    for i, p in enumerate(points):
        phi = math.radians(phase_deg + i * twist_deg)
        xi = math.cos(phi) * nvec[i] + math.sin(phi) * bvec[i]
        yi = -math.sin(phi) * nvec[i] + math.cos(phi) * bvec[i]
        zi = t[i]
        bases.append((p, np.column_stack([xi, yi, zi])))
    return bases


def _2bp_step_target_basis(bp_basis_i, bp_basis_j, p_i, p_j):
    z_axis = p_j - p_i
    z_axis /= np.linalg.norm(z_axis)
    x_axis = bp_basis_i[:, 0] + bp_basis_j[:, 0]
    x_axis = x_axis - z_axis * np.dot(x_axis, z_axis)
    if np.linalg.norm(x_axis) < 1e-6:
        x_axis = bp_basis_i[:, 0] - z_axis * np.dot(bp_basis_i[:, 0], z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    return 0.5 * (p_i + p_j), np.column_stack([x_axis, y_axis, z_axis])


def _2bp_place_step(template_atoms, templ, p_i, basis_i, p_j, basis_j, n_bp, i_step):
    origin_tgt, basis_tgt = _2bp_step_target_basis(basis_i, basis_j, p_i, p_j)
    atoms = _2bp_transform_atoms(template_atoms, templ["step_origin"], templ["step_basis"], origin_tgt, basis_tgt)
    keyA0, keyA1 = templ["resA"][0], templ["resA"][1]
    keyB0, keyB1 = templ["resB"][0], templ["resB"][1]
    out = []
    for a in atoms:
        aa = a.copy()
        k = (a["chain"], a["resseq"], a["resname"])
        if k == keyA0:
            aa["chain"] = "A"
            aa["resseq"] = i_step + 1
        elif k == keyA1:
            aa["chain"] = "A"
            aa["resseq"] = i_step + 2
        elif k == keyB1:
            aa["chain"] = "B"
            aa["resseq"] = n_bp - i_step
        elif k == keyB0:
            aa["chain"] = "B"
            aa["resseq"] = n_bp - i_step - 1
        else:
            raise RuntimeError(f"Unexpected residue mapping for {k}")
        out.append(aa)
    return out


def _2bp_choose_global_phase(points, template_atoms, templ, interp, twist_deg, phase_step_deg, use_backbone_only=False):
    if interp is None:
        return 0.0, 0.0
    _, nvec, bvec = _2bp_parallel_transport_frames(points)
    best = None
    for phase in np.arange(0.0, 360.0, phase_step_deg):
        bp_frames = _2bp_target_bp_frames(points, nvec, bvec, twist_deg, phase)
        score = 0.0
        for i in range(len(points) - 1):
            p_i, basis_i = bp_frames[i]
            p_j, basis_j = bp_frames[i + 1]
            atoms = _2bp_place_step(template_atoms, templ, p_i, basis_i, p_j, basis_j, len(points), i)
            score += _2bp_score_atoms_density(atoms, interp, use_backbone_only)
        if best is None or score > best[1]:
            best = (phase, score)
    return best


def _2bp_local_refine_step(
    atoms,
    origin,
    tangent,
    normal,
    binormal,
    interp,
    max_shift_A=1.0,
    shift_step_A=0.5,
    max_twist_deg=10.0,
    twist_step_deg=2.5,
    shift_penalty=0.10,
    twist_penalty=0.01,
    use_backbone_only=False,
):
    if interp is None:
        return atoms, {"shift": (0.0, 0.0), "twist_deg": 0.0, "score": _2bp_score_atoms_density(atoms, interp, use_backbone_only)}
    best_atoms = atoms
    base_score = _2bp_score_atoms_density(atoms, interp, use_backbone_only)
    best = (base_score, 0.0, 0.0, 0.0)
    shifts = np.arange(-max_shift_A, max_shift_A + 1e-6, shift_step_A)
    twists = np.arange(-max_twist_deg, max_twist_deg + 1e-6, twist_step_deg)
    for du in shifts:
        for dv in shifts:
            shift_vec = du * normal + dv * binormal
            for dphi in twists:
                trial = []
                for a in atoms:
                    rel = a["coord"] - origin
                    aa = a.copy()
                    aa["coord"] = origin + shift_vec + _2bp_rotate_about_axis(rel, tangent, math.radians(dphi))
                    trial.append(aa)
                dens = _2bp_score_atoms_density(trial, interp, use_backbone_only)
                score = dens - shift_penalty * (du * du + dv * dv) - twist_penalty * (dphi / 10.0) ** 2
                if score > best[0]:
                    best = (score, du, dv, dphi)
                    best_atoms = trial
    return best_atoms, {"shift": (best[1], best[2]), "twist_deg": best[3], "score": best[0]}


def _2bp_merge_atoms(step_atoms_list):
    accum = {}
    meta = {}
    for atoms in step_atoms_list:
        for a in atoms:
            key = (a["chain"], a["resseq"], a["name"])
            if key not in accum:
                accum[key] = []
                meta[key] = a.copy()
            accum[key].append(a["coord"])
    out = []
    for key in sorted(accum, key=lambda x: (x[0], x[1], x[2])):
        aa = meta[key].copy()
        aa["coord"] = np.mean(np.vstack(accum[key]), axis=0)
        out.append(aa)
    return out


def _2bp_linkage_stats(all_atoms, n_bp):
    by = defaultdict(dict)
    for a in all_atoms:
        by[(a["chain"], a["resseq"])][a["name"]] = a["coord"]
    a_d = []
    b_d = []
    for i in range(1, n_bp):
        if ("A", i) in by and ("A", i + 1) in by and "O3'" in by[("A", i)] and "P" in by[("A", i + 1)]:
            a_d.append(float(np.linalg.norm(by[("A", i)]["O3'"] - by[("A", i + 1)]["P"])))
        if ("B", i) in by and ("B", i + 1) in by and "P" in by[("B", i)] and "O3'" in by[("B", i + 1)]:
            b_d.append(float(np.linalg.norm(by[("B", i)]["P"] - by[("B", i + 1)]["O3'"])))
    return a_d, b_d


def _2bp_write_pdb(atoms, path: Path):
    with Path(path).open("w") as out:
        serial = 1
        prev_chain = None
        for atom in atoms:
            chain, resseq = atom["chain"], atom["resseq"]
            if prev_chain is not None and chain != prev_chain:
                out.write("TER\n")
            prev_chain = chain
            x, y, z = atom["coord"]
            out.write(
                f"ATOM  {serial:5d} {atom['name']:<4s} {atom['resname']:>3s} {chain:1s}{resseq:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}{atom['occ']:6.2f}{atom['bfac']:6.2f}          {atom['element']:>2s}\n"
            )
            serial += 1
        out.write("TER\nEND\n")


def _2bp_write_report(path: Path, report_dict: dict):
    with Path(path).open("w") as out:
        for k, v in report_dict.items():
            out.write(f"{k}: {v}\n")


def build_poly_at_from_2bp_centerline(
    centerline_pdb: Path,
    template_2bp_pdb: Path,
    out_pdb: Path,
    report_path: Path | None = None,
    map_path: Path | None = None,
    threshold: float | None = None,
    n_bp: int | None = None,
    target_spacing: float = 3.4,
    twist_deg: float = 35.0,
    trim_start_bp: int = 0,
    trim_end_bp: int = 0,
    trim_start_A: float = 0.0,
    trim_end_A: float = 0.0,
    global_phase_step_deg: float = 5.0,
    no_global_phase_opt: bool = False,
    local_refine: bool = False,
    local_shift_A: float = 1.0,
    local_shift_step_A: float = 0.5,
    local_twist_deg: float = 10.0,
    local_twist_step_deg: float = 2.5,
    backbone_only_score: bool = False,
) -> Tuple[Path, Optional[Path]]:
    centerline = _2bp_read_polyline_pdb(centerline_pdb)
    template_atoms = _2bp_read_pdb_atoms(template_2bp_pdb)
    templ = _2bp_derive_template(template_atoms)

    trim_start_A = trim_start_A + trim_start_bp * target_spacing
    trim_end_A = trim_end_A + trim_end_bp * target_spacing
    points, target_s, total_len = _2bp_resample_centerline(
        centerline,
        n_points=n_bp,
        spacing=target_spacing,
        trim_start_A=trim_start_A,
        trim_end_A=trim_end_A,
    )
    n_bp = len(points)

    interp = None
    phase_score = 0.0
    if map_path is not None:
        _, _, interp = _2bp_load_map(map_path)
    _, nvec, bvec = _2bp_parallel_transport_frames(points)
    if no_global_phase_opt or interp is None:
        global_phase_deg = 0.0
    else:
        global_phase_deg, phase_score = _2bp_choose_global_phase(
            points,
            template_atoms,
            templ,
            interp,
            twist_deg,
            global_phase_step_deg,
            backbone_only_score,
        )
    bp_frames = _2bp_target_bp_frames(points, nvec, bvec, twist_deg, global_phase_deg)

    step_models = []
    refine_log = []
    for i in range(n_bp - 1):
        p_i, basis_i = bp_frames[i]
        p_j, basis_j = bp_frames[i + 1]
        atoms = _2bp_place_step(template_atoms, templ, p_i, basis_i, p_j, basis_j, n_bp, i)
        if local_refine and interp is not None:
            origin, basis = _2bp_step_target_basis(basis_i, basis_j, p_i, p_j)
            atoms, info = _2bp_local_refine_step(
                atoms,
                origin,
                basis[:, 2],
                basis[:, 0],
                basis[:, 1],
                interp,
                max_shift_A=local_shift_A,
                shift_step_A=local_shift_step_A,
                max_twist_deg=local_twist_deg,
                twist_step_deg=local_twist_step_deg,
                use_backbone_only=backbone_only_score,
            )
            refine_log.append((i + 1, info))
        step_models.append(atoms)

    all_atoms = _2bp_merge_atoms(step_models)
    out_pdb.parent.mkdir(parents=True, exist_ok=True)
    _2bp_write_pdb(all_atoms, out_pdb)

    spacings = np.linalg.norm(np.diff(points, axis=0), axis=1)
    a_d, b_d = _2bp_linkage_stats(all_atoms, n_bp)
    report = {
        "centerline_pdb": str(centerline_pdb.resolve()),
        "template_2bp_pdb": str(template_2bp_pdb.resolve()),
        "map": str(map_path.resolve()) if map_path else "None",
        "threshold_context": threshold,
        "input_centerline_points": len(centerline),
        "trim_start_A": trim_start_A,
        "trim_end_A": trim_end_A,
        "usable_path_length_A": float(target_s[-1] - target_s[0]),
        "n_bp": n_bp,
        "realized_center_spacing_A_mean": float(np.mean(spacings)) if len(spacings) else 0.0,
        "realized_center_spacing_A_min": float(np.min(spacings)) if len(spacings) else 0.0,
        "realized_center_spacing_A_max": float(np.max(spacings)) if len(spacings) else 0.0,
        "twist_deg_per_bp": twist_deg,
        "global_phase_deg": float(global_phase_deg),
        "global_phase_score": float(phase_score),
        "template_pairing": f"{templ['pair0'][0]}<->{templ['pair0'][1]}; {templ['pair1'][0]}<->{templ['pair1'][1]}",
        "chain_A_link_O3_to_next_P_mean_A": float(np.mean(a_d)) if a_d else None,
        "chain_A_link_O3_to_next_P_min_A": float(np.min(a_d)) if a_d else None,
        "chain_A_link_O3_to_next_P_max_A": float(np.max(a_d)) if a_d else None,
        "chain_B_link_P_to_next_O3_mean_A": float(np.mean(b_d)) if b_d else None,
        "chain_B_link_P_to_next_O3_min_A": float(np.min(b_d)) if b_d else None,
        "chain_B_link_P_to_next_O3_max_A": float(np.max(b_d)) if b_d else None,
        "local_refine_steps": len(refine_log),
        "note": "Rigid 2-bp step starting model. Overlapping residues from adjacent steps are averaged; real-space/stereochemical refinement is still recommended.",
    }
    if report_path:
        _2bp_write_report(report_path, report)

    return out_pdb, report_path
