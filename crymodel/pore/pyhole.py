# crymodel/pore/pyhole.py
"""HOLE-like pore profile calculation with straight/curved centerline options."""
from __future__ import annotations
import re
import math
import json
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np

from ..io.pdb import read_model_xyz


# Constants
ONE_LETTER = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', 'GLY': 'G',
    'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N',
    'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S', 'THR': 'T', 'VAL': 'V',
    'TRP': 'W', 'TYR': 'Y', 'SEC': 'U', 'PYL': 'O'
}

VDW_DEFAULT = {
    'H': 1.20, 'C': 1.70, 'N': 1.55, 'O': 1.52, 'F': 1.47, 'P': 1.80,
    'S': 1.80, 'CL': 1.75, 'BR': 1.85, 'I': 1.98, 'NA': 2.27, 'MG': 1.73,
    'K': 2.75, 'CA': 2.31, 'ZN': 1.39, 'FE': 1.56, 'CU': 1.40, 'MN': 1.61
}

KD_HYDRO = {
    'ILE': 4.5, 'VAL': 4.2, 'LEU': 3.8, 'PHE': 2.8, 'CYS': 2.5, 'MET': 1.9,
    'ALA': 1.8, 'GLY': -0.4, 'THR': -0.7, 'SER': -0.8, 'TRP': -0.9,
    'TYR': -1.3, 'PRO': -1.6, 'HIS': -3.2, 'GLU': -3.5, 'GLN': -3.5,
    'ASP': -3.5, 'ASN': -3.5, 'LYS': -3.9, 'ARG': -4.5
}

KD_MIN = min(KD_HYDRO.values())
KD_MAX = max(KD_HYDRO.values())

CHARGE_MAP = {'ARG': +1.0, 'LYS': +1.0, 'HIS': +0.1, 'ASP': -1.0, 'GLU': -1.0}
ELEC_MIN, ELEC_MAX = -1.0, +1.0


class Atom:
    """Simple atom representation."""
    __slots__ = ('x', 'y', 'z', 'name', 'resname', 'chain', 'resi', 'element', 'occ')
    
    def __init__(self, x, y, z, name, resname, chain, resi, element, occ):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.name = name.strip()
        self.resname = resname
        self.chain = chain
        self.resi = int(resi)
        self.element = element
        self.occ = occ


def guess_element(atom_name: str, element_field: str) -> str:
    """Guess element from atom name and element field."""
    if element_field and element_field.strip():
        return element_field.strip().upper()
    an = atom_name.strip()
    if not an:
        return 'C'
    if an[0].isdigit() and len(an) >= 2:
        c = an[1]
        if len(an) >= 3 and an[2].isalpha():
            return (c + an[2]).upper()
        return c.upper()
    c = an[0]
    if len(an) >= 2 and an[1].isalpha() and an[1].islower():
        return (c + an[1]).upper()
    return c.upper()


def vdw_radius(element: str, custom: Dict[str, float]) -> float:
    """Get van der Waals radius for element."""
    e = element.upper()
    return custom.get(e, VDW_DEFAULT.get(e, 1.7))


def parse_residue_tokens(s: str) -> List[Tuple[str, int]]:
    """Parse residue selection string.
    
    Accepts formats like:
    - "A:123" or "A 123" -> (chain, residue)
    - "123" -> ('*', residue)
    - "A:123,B:456" -> multiple residues
    
    Returns list of (chain, residue_number) tuples.
    """
    tokens = []
    if not s:
        return tokens
    for t in s.split(','):
        t = t.strip()
        if not t:
            continue
        # Match "RESNAME123/CHAIN" format
        m = re.match(r'^[A-Za-z]{0,3}?(\d+)\s*/\s*([A-Za-z0-9])$', t)
        if m:
            tokens.append((m.group(2), int(m.group(1))))
            continue
        # Match "CHAIN:123" or "CHAIN 123" format
        m = re.match(r'^([A-Za-z0-9])\s*[:\s]\s*(\d+)$', t)
        if m:
            tokens.append((m.group(1), int(m.group(2))))
            continue
        # Match just number "123"
        m = re.match(r'^\s*(\d+)\s*$', t)
        if m:
            tokens.append(('*', int(m.group(1))))
            continue
        raise ValueError(f"Could not parse residue token: '{t}'")
    return tokens


def load_pdb_atoms(path: str, include_h: bool = True) -> List[Atom]:
    """Load atoms from PDB file."""
    atoms = []
    with open(path, 'r') as f:
        for line in f:
            if line[:6] not in ('ATOM  ', 'HETATM'):
                continue
            name = line[12:16]
            resname = line[17:20].strip()
            chain = line[21].strip() or 'A'
            resi_str = line[22:26].strip() or '0'
            try:
                resi = int(resi_str)
            except ValueError:
                continue
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            occ = float(line[54:60]) if line[54:60].strip() else 1.0
            elem_field = line[76:78] if len(line) >= 78 else ''
            elem = guess_element(name, elem_field)
            if not include_h and elem.upper() == 'H':
                continue
            atoms.append(Atom(x, y, z, name, resname, chain, resi, elem, occ))
    return atoms


def ca_positions_for(atoms: List[Atom], sel: List[Tuple[str, int]]) -> np.ndarray:
    """Get C-alpha positions for selected residues."""
    out = []
    for (ch, rnum) in sel:
        for a in atoms:
            if a.name == 'CA' and a.resi == rnum and (ch == '*' or a.chain == ch):
                out.append([a.x, a.y, a.z])
    return np.array(out, dtype=float) if out else np.zeros((0, 3), dtype=float)


def orthonormal_basis_from_axis(axis: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Generate orthonormal basis vectors perpendicular to axis."""
    u = axis / (np.linalg.norm(axis) + 1e-12)
    a = np.array([1.0, 0.0, 0.0]) if abs(u[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    v = np.cross(u, a)
    n = np.linalg.norm(v)
    if n < 1e-8:
        a = np.array([0.0, 0.0, 1.0])
        v = np.cross(u, a)
        n = np.linalg.norm(v)
    v /= (n + 1e-12)
    w = np.cross(u, v)
    w /= (np.linalg.norm(w) + 1e-12)
    return v, w


def _clearance_at_point(atom_xyz: np.ndarray, atom_r: np.ndarray, c: np.ndarray) -> float:
    """Calculate minimum clearance at point c."""
    dv = atom_xyz - c
    d = np.sqrt((dv * dv).sum(axis=1)) - atom_r
    return float(np.min(d))


def _evaluate_slice(
    atom_xyz: np.ndarray,
    atom_r: np.ndarray,
    atom_meta: List[Tuple[str, str, int]],
    c: np.ndarray,
    contact_eps: float,
    hydro_scale: str,
    electro_scale: str,
) -> Tuple[float, List[str], float, float]:
    """Evaluate pore properties at a single point."""
    dv = atom_xyz - c
    d = np.sqrt((dv * dv).sum(axis=1)) - atom_r
    rmin = float(np.min(d))
    mask = d <= (rmin + contact_eps)
    
    seen = set()
    tags = []
    hyd = []
    chg = []
    
    for i, ok in enumerate(mask):
        if not ok:
            continue
        chain, resname, resi = atom_meta[i]
        key = (chain, resi, resname)
        if key in seen:
            continue
        seen.add(key)
        tags.append(f"{ONE_LETTER.get(resname.upper(), '?')}{resi}/{chain}")
        hyd.append(KD_HYDRO.get(resname.upper(), 0.0))
        chg.append(CHARGE_MAP.get(resname.upper(), 0.0))
    
    hydro = float(np.mean(hyd)) if hyd else 0.0
    electro = float(np.mean(chg)) if chg else 0.0
    
    if hydro_scale == '01':
        hydro = (hydro - KD_MIN) / (KD_MAX - KD_MIN) if KD_MAX > KD_MIN else 0.0
    if electro_scale == '01':
        electro = (electro - ELEC_MIN) / (ELEC_MAX - ELEC_MIN)
    
    return rmin, tags, hydro, electro


def profile_along_axis(
    atom_xyz: np.ndarray,
    atom_r: np.ndarray,
    c0: np.ndarray,
    c1: np.ndarray,
    step: float,
    eps: float,
    atom_meta: List[Tuple[str, str, int]],
    adaptive: bool = False,
    slope_thresh: float = 0.5,
    max_refine: int = 3,
    hydro_scale: str = 'raw',
    electro_scale: str = 'raw',
    occupancy_metric: str = 'hydro',
) -> Tuple[List[Dict], np.ndarray, float]:
    """Calculate pore profile along straight axis."""
    axis = c1 - c0
    L = np.linalg.norm(axis)
    if L < 1e-6:
        raise ValueError("Top and bottom centers are too close.")
    u = axis / L
    
    def ctr(s):
        return c0 + u * s
    
    svals = list(np.linspace(0.0, L, max(1, int(round(L / step)) + 1)))
    
    def eval_rows(vals):
        out = []
        for s in vals:
            c = ctr(s)
            rmin, tags, hyd, elec = _evaluate_slice(atom_xyz, atom_r, atom_meta, c, eps, hydro_scale, electro_scale)
            out.append({
                's_A': float(s),
                'x': float(c[0]),
                'y': float(c[1]),
                'z': float(c[2]),
                'radius_A': float(rmin),
                'hydro_index': float(hyd),
                'electro_index': float(elec),
                'contributors': ';'.join(tags[:50])
            })
        return out
    
    rows = eval_rows(svals)
    
    if adaptive:
        for _ in range(max_refine):
            rows.sort(key=lambda r: r['s_A'])
            ns = []
            for i in range(len(rows) - 1):
                s0, r0 = rows[i]['s_A'], rows[i]['radius_A']
                s1, r1 = rows[i + 1]['s_A'], rows[i + 1]['radius_A']
                ds = s1 - s0
                if ds <= step / 2:
                    continue
                slope = abs((r1 - r0) / ds) if ds > 1e-9 else 0.0
                if slope > slope_thresh:
                    ns.append(0.5 * (s0 + s1))
            if not ns:
                break
            rows += eval_rows(ns)
    
    rows.sort(key=lambda r: r['s_A'])
    for r in rows:
        r['occ_value'] = float(r['hydro_index'] if occupancy_metric == 'hydro' else r['electro_index'])
    
    return rows, u, L


def construct_centers_curved(
    atom_xyz: np.ndarray,
    atom_r: np.ndarray,
    c0: np.ndarray,
    c1: np.ndarray,
    step: float,
    curve_radius: float,
    curve_iters: int,
) -> Tuple[List[np.ndarray], np.ndarray, float]:
    """Construct curved centerline through pore."""
    axis = c1 - c0
    L = float(np.linalg.norm(axis))
    if L < 1e-6:
        raise ValueError("Top and bottom centers are too close.")
    u = axis / L
    v, w = orthonormal_basis_from_axis(u)
    
    svals = list(np.linspace(0.0, L, max(1, int(round(L / step)) + 1)))
    centers = []
    c_prev = c0.copy()
    
    for idx, s in enumerate(svals):
        if s <= 1e-9:
            c = c0.copy()
        elif abs(s - L) <= 1e-9:
            c = c1.copy()
        else:
            ds = svals[idx] - svals[idx - 1]
            c = c_prev + u * ds
            r = float(curve_radius)
            for _ in range(int(curve_iters)):
                best_c = c
                best_cl = _clearance_at_point(atom_xyz, atom_r, c)
                for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (1, -1), (-1, -1), (0, 0)]:
                    cand = c + (dx * r) * v + (dy * r) * w
                    cl = _clearance_at_point(atom_xyz, atom_r, cand)
                    if cl > best_cl:
                        best_cl = cl
                        best_c = cand
                c = best_c
                r *= 0.5
        centers.append(c)
        c_prev = c
    
    return centers, u, L


def profile_along_centers(
    atom_xyz: np.ndarray,
    atom_r: np.ndarray,
    centers: List[np.ndarray],
    eps: float,
    atom_meta: List[Tuple[str, str, int]],
    hydro_scale: str,
    electro_scale: str,
    occupancy_metric: str,
) -> List[Dict]:
    """Calculate pore profile along curved centerline."""
    rows = []
    s_acc = 0.0
    
    for i, c in enumerate(centers):
        if i > 0:
            s_acc += float(np.linalg.norm(centers[i] - centers[i - 1]))
        rmin, tags, hyd, elec = _evaluate_slice(atom_xyz, atom_r, atom_meta, c, eps, hydro_scale, electro_scale)
        rows.append({
            's_A': float(s_acc),
            'x': float(c[0]),
            'y': float(c[1]),
            'z': float(c[2]),
            'radius_A': float(rmin),
            'hydro_index': float(hyd),
            'electro_index': float(elec),
            'contributors': ';'.join(tags[:50])
        })
    
    # Calculate tangent vectors
    for i in range(len(rows)):
        if i == 0:
            t = np.array([
                rows[1]['x'] - rows[0]['x'],
                rows[1]['y'] - rows[0]['y'],
                rows[1]['z'] - rows[0]['z']
            ], dtype=float)
        elif i == len(rows) - 1:
            t = np.array([
                rows[i]['x'] - rows[i - 1]['x'],
                rows[i]['y'] - rows[i - 1]['y'],
                rows[i]['z'] - rows[i - 1]['z']
            ], dtype=float)
        else:
            t = np.array([
                rows[i + 1]['x'] - rows[i - 1]['x'],
                rows[i + 1]['y'] - rows[i - 1]['y'],
                rows[i + 1]['z'] - rows[i - 1]['z']
            ], dtype=float)
        n = np.linalg.norm(t)
        t = np.array([0.0, 0.0, 1.0]) if n < 1e-9 else (t / n)
        rows[i]['tx'] = float(t[0])
        rows[i]['ty'] = float(t[1])
        rows[i]['tz'] = float(t[2])
        rows[i]['occ_value'] = float(rows[i]['hydro_index'] if occupancy_metric == 'hydro' else rows[i]['electro_index'])
    
    return rows


def write_csv(path: Path, rows: List[Dict]) -> None:
    """Write profile data to CSV."""
    if not rows:
        return
    with open(path, 'w', newline='') as f:
        cols = list(rows[0].keys())
        if 'occ_value' not in cols:
            cols.append('occ_value')
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_centerline_pdb(path: Path, rows: List[Dict]) -> None:
    """Write centerline as PDB file."""
    lines = []
    serial = 1
    for i, r in enumerate(rows, start=1):
        x, y, z = r['x'], r['y'], r['z']
        b = r['radius_A']
        occ = float(r.get('occ_value', 0.0))
        lines.append(
            f"HETATM{serial:5d}  O  PORE Z{i:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}{occ:6.2f}{b:6.2f}          O \n"
        )
        serial += 1
    with open(path, 'w', newline='') as f:
        f.writelines(lines)


def _format_pdb_atom_line(
    serial: int,
    name: str,
    resName: str,
    chainID: str,
    resSeq: int,
    x: float, y: float, z: float,
    occupancy: float,
    bfactor: float,
    element: str,
    altLoc: str = ' ',
    iCode: str = ' '
) -> str:
    """Format a PDB ATOM line."""
    return (
        f"ATOM  {serial:5d} {name:^4}{altLoc}{resName:>3} {chainID}{resSeq:>4}{iCode}   "
        f"{x:8.3f}{y:8.3f}{z:8.3f}{occupancy:6.2f}{bfactor:6.2f}          {element:>2}  \n"
    )


def write_mesh_pdb(path: Path, rows: List[Dict], axis_u: np.ndarray, rings: int = 24) -> None:
    """Write pore mesh as PDB with CONECT records."""
    use_local = ('tx' in rows[0])
    if not use_local:
        u = axis_u / (np.linalg.norm(axis_u) + 1e-12)
        base_v, base_w = orthonormal_basis_from_axis(u)
    
    coords = []
    bvals = []
    occs = []
    
    for r in rows:
        c = np.array([r['x'], r['y'], r['z']], dtype=float)
        rad = max(0.0, float(r['radius_A']))
        occ = float(r.get('occ_value', 0.0))
        if use_local:
            u_loc = np.array([r['tx'], r['ty'], r['tz']], dtype=float)
            v, w = orthonormal_basis_from_axis(u_loc)
        else:
            v, w = base_v, base_w
        for k in range(rings):
            ang = 2 * np.pi * (k / rings)
            p = c + rad * (np.cos(ang) * v + np.sin(ang) * w)
            coords.append(p)
            bvals.append(rad)
            occs.append(occ)
    
    lines = []
    serial_start = 1
    for i, pnt in enumerate(coords, start=serial_start):
        x, y, z = pnt
        b = bvals[i - serial_start]
        occ = occs[i - serial_start]
        lines.append(_format_pdb_atom_line(i, 'C', 'ALA', 'M', 1, x, y, z, occ, b, 'C'))
    
    n = len(rows)
    R = rings
    
    def idx(step, k):
        return serial_start + step * R + k
    
    for step in range(n):
        for k in range(R):
            lines.append(f"CONECT{idx(step, k):5d}{idx(step, (k + 1) % R):5d}\n")
    for step in range(n - 1):
        for k in range(R):
            lines.append(f"CONECT{idx(step, k):5d}{idx(step + 1, k):5d}\n")
    
    with open(path, 'w', newline='') as f:
        f.writelines(lines)


def calculate_pore_statistics(
    rows: List[Dict],
    probe: float = 0.0,
    conductivity: float = 1.5,
    pass_radii: Optional[Dict[str, float]] = None,
) -> Dict:
    """Calculate pore statistics: volume, resistance, conductance, passability."""
    if not rows:
        return {}
    
    # Volume calculation
    def trap_vol(rows, probe):
        if len(rows) < 2:
            return 0.0
        vol = 0.0
        for i in range(len(rows) - 1):
            ds = float(rows[i + 1]['s_A'] - rows[i]['s_A'])
            r0 = max(0.0, float(rows[i]['radius_A']) - probe)
            r1 = max(0.0, float(rows[i + 1]['radius_A']) - probe)
            vol += 0.5 * (math.pi * r0 * r0 + math.pi * r1 * r1) * ds
        return vol
    
    volume_geom = trap_vol(rows, 0.0)
    volume_access = trap_vol(rows, float(probe))
    
    # Resistance calculation
    kappa = float(conductivity)
    rho = 1.0 / max(kappa, 1e-12)
    R_pore = 0.0
    blocked = False
    
    for i in range(len(rows) - 1):
        ds_m = float(rows[i + 1]['s_A'] - rows[i]['s_A']) * 1e-10
        r0 = max(1e-6, float(rows[i]['radius_A'])) * 1e-10
        r1 = max(1e-6, float(rows[i + 1]['radius_A'])) * 1e-10
        if r0 <= 0.0 or r1 <= 0.0:
            blocked = True
        A0 = math.pi * r0 * r0
        A1 = math.pi * r1 * r1
        R_pore += 0.5 * ((1.0 / max(A0, 1e-30)) + (1.0 / max(A1, 1e-30))) * ds_m * rho
    
    rmin_A = max(1e-6, float(min([r['radius_A'] for r in rows])))
    rmin_m = rmin_A * 1e-10
    R_access = (1.0 / (2.0 * kappa * max(rmin_m, 1e-12)))
    R_total = float('inf') if blocked else (R_pore + R_access)
    G_nS = 0.0 if not math.isfinite(R_total) else (1.0 / R_total) * 1e9
    
    # Passability analysis
    if pass_radii is None:
        pass_radii = {'water': 1.4, 'na': 1.02, 'k': 1.38, 'ca': 1.00}
    
    pass_report = {}
    for sp, radA in pass_radii.items():
        radA = float(radA)
        spans = []
        start = None
        local_min = 1e9
        local_idx = None
        
        for i, row in enumerate(rows):
            rA = float(row['radius_A'])
            if rA < radA:
                if start is None:
                    start = row['s_A']
                    local_min = rA
                    local_idx = i
                else:
                    if rA < local_min:
                        local_min = rA
                        local_idx = i
            else:
                if start is not None:
                    end = rows[i]['s_A']
                    contrib = rows[local_idx]['contributors'] if local_idx is not None else ''
                    spans.append({
                        'start_s_A': float(start),
                        'end_s_A': float(end),
                        'min_radius_A': float(local_min),
                        'min_contributors': contrib
                    })
                    start = None
                    local_min = 1e9
                    local_idx = None
        
        if start is not None:
            end = rows[-1]['s_A']
            contrib = rows[local_idx]['contributors'] if local_idx is not None else ''
            spans.append({
                'start_s_A': float(start),
                'end_s_A': float(end),
                'min_radius_A': float(local_min),
                'min_contributors': contrib
            })
        
        pass_report[sp] = {'is_passable': len(spans) == 0, 'blocked_spans': spans}
    
    return {
        'volume_geometric_A3': float(volume_geom),
        'volume_accessible_A3': float(volume_access),
        'R_pore_ohm': float(R_pore),
        'R_access_ohm': float(R_access),
        'R_total_ohm': float(R_total),
        'G_nS': float(G_nS),
        'min_radius_A': float(rmin_A),
        'passability': pass_report,
    }

