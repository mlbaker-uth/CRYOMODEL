# cryomodel/validation/geometry_priors.py
"""MolProbity-style geometry metrics (lite): Ramachandran, ω, steric clashes, Cβ, χ1 rotamer."""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import gemmi
import numpy as np

from ..gemmi_atoms import sole_atom
from ..mutate.chi import CHI1_PRIOR_TRIPLET, CHI1_TRIAL_ANGLES, chi1_dihedral_deg, chi1_quadruple
from ..zonal.ramachandran import RamaClass, classify_phi_psi_general, phi_psi_deg, rama_penalty

# MolProbity-style probe radius (Å) for van der Waals overlap (heavy atoms).
_PROBE_RADIUS_A = 0.25
# Distance < r_cov_i + r_cov_j + pad ⇒ inferred covalent bond for connectivity.
_BOND_PAD_A = 0.5
# MolProbity all-atom clash: serious overlap (vdW penetration) ≥ this value (Å).
# https://molprobity.biochem.duke.edu — full MolProbity uses explicit H; we use heavy atoms only.
_MOLPROBITY_OVERLAP_MIN_A = 0.4
# Skip bond-inference tests for pairs farther than this (Å); no covalent bond is longer.
_MAX_SINGLE_BOND_DISTANCE_A = 3.2
# Trans-peptide C–N when Gemmi MonLib lacks explicit inter-residue links (prepare_topology).
_MAX_PEPTIDE_CN_DISTANCE_A = 2.2


def _bundled_monomer_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "data" / "monomers"


def _monomer_search_roots(explicit_dir: Optional[str]) -> List[Path]:
    roots: List[Path] = []
    if explicit_dir:
        roots.append(Path(explicit_dir))
    for key in ("CRYOMODEL_MONOMER_LIB", "CLIBD_MON", "PHENIX_CIF_MON_LIB_PATH"):
        raw = os.environ.get(key)
        if not raw:
            continue
        for part in raw.split(os.pathsep):
            part = part.strip()
            if part:
                roots.append(Path(part))
    roots.append(_bundled_monomer_dir())
    return roots


def _normalize_residue_name(residue: gemmi.Residue) -> str:
    name = residue.name.strip().upper().split()[0]
    return name[:3] if len(name) >= 3 else name


def _collect_structure_resnames(structure: gemmi.Structure) -> List[str]:
    seen: Set[str] = set()
    for model in structure:
        for chain in model:
            for res in chain:
                rn = _normalize_residue_name(res)
                if rn:
                    seen.add(rn)
    return sorted(seen)


def _try_load_monlib(resnames: Sequence[str], roots: List[Path]) -> Optional[gemmi.MonLib]:
    """Return a :class:`gemmi.MonLib` or ``None`` if nothing could be loaded."""
    if not resnames:
        return None
    for root in roots:
        if not root.is_dir():
            continue
        list_cif = root / "list" / "mon_lib_list.cif"
        if list_cif.is_file():
            try:
                return gemmi.read_monomer_lib(str(root), list(resnames), ignore_missing=True)
            except OSError:
                continue
    ml = gemmi.MonLib()
    n_ok = 0
    for rn in resnames:
        got = False
        for root in roots:
            if not root.is_dir():
                continue
            cands = [root / f"{rn}.cif"]
            if rn:
                letter = rn[0]
                if letter.isalpha():
                    cands.append(root / letter.lower() / f"{rn}.cif")
            for p in cands:
                if not p.is_file():
                    continue
                try:
                    ml.read_monomer_cif(str(p))
                except Exception:
                    continue
                n_ok += 1
                got = True
                break
            if got:
                break
    return ml if n_ok > 0 else None


def _undirected_graph_edge(adj: List[List[int]], i: int, j: int) -> None:
    if i == j:
        return
    if j not in adj[i]:
        adj[i].append(j)
        adj[j].append(i)


def _adjacency_from_gemmi_topology(
    structure: gemmi.Structure,
    monlib: gemmi.MonLib,
    idx_by_atom_id: Dict[int, int],
    n_atoms: int,
) -> List[List[int]]:
    """Bond edges from :func:`gemmi.prepare_topology` (intra-residue) plus standard peptide C–N."""
    topo = gemmi.prepare_topology(structure, monlib, model_index=0)
    adj: List[List[int]] = [[] for _ in range(n_atoms)]
    for bond in topo.bonds:
        atoms = bond.atoms
        if len(atoms) != 2:
            continue
        i0 = idx_by_atom_id.get(id(atoms[0]))
        i1 = idx_by_atom_id.get(id(atoms[1]))
        if i0 is None or i1 is None:
            continue
        _undirected_graph_edge(adj, i0, i1)
    _append_peptide_cn_edges(structure, idx_by_atom_id, adj)
    return adj


def _append_peptide_cn_edges(
    structure: gemmi.Structure,
    idx_by_atom_id: Dict[int, int],
    adj: List[List[int]],
) -> None:
    """Add trans-peptide C(i)–N(i+1) edges when atoms are clash-scoped and C–N is short enough."""
    for model in structure:
        for chain in model:
            prev: Optional[gemmi.Residue] = None
            for res in chain:
                if prev is not None:
                    c_atom: Optional[gemmi.Atom] = None
                    n_atom: Optional[gemmi.Atom] = None
                    for a in prev:
                        if a.is_hydrogen():
                            continue
                        if not _altloc_included(a):
                            continue
                        if a.name.strip() == "C":
                            c_atom = a
                            break
                    for a in res:
                        if a.is_hydrogen():
                            continue
                        if not _altloc_included(a):
                            continue
                        if a.name.strip() == "N":
                            n_atom = a
                            break
                    if c_atom is not None and n_atom is not None:
                        if c_atom.pos.dist(n_atom.pos) <= _MAX_PEPTIDE_CN_DISTANCE_A:
                            ic = idx_by_atom_id.get(id(c_atom))
                            jn = idx_by_atom_id.get(id(n_atom))
                            if ic is not None and jn is not None:
                                _undirected_graph_edge(adj, ic, jn)
                prev = res


def _elem_vdw_cov(atom: gemmi.Atom) -> Tuple[float, float]:
    el = atom.element.name.strip() or "C"
    try:
        e = gemmi.Element(el)
        return float(e.vdw_r), float(e.covalent_r)
    except Exception:
        return 1.7, 0.77


def _atom_xyz(atom: gemmi.Atom) -> np.ndarray:
    return np.array([atom.pos.x, atom.pos.y, atom.pos.z], dtype=np.float64)


def _altloc_included(atom: gemmi.Atom) -> bool:
    """Primary altloc / blank / unset (Gemmi may use null for default)."""
    al = atom.altloc
    if al is None:
        return True
    s = str(al).strip()
    if s in ("", "\x00", "\0"):
        return True
    return s in ("A", "1")


def _collect_clash_scope_atoms(structure: gemmi.Structure) -> List[Tuple[gemmi.Atom, str, str]]:
    atoms: List[Tuple[gemmi.Atom, str, str]] = []
    for model in structure:
        for chain in model:
            cid = chain.name
            for res in chain:
                rid = str(res.seqid)
                for atom in res:
                    if atom.is_hydrogen():
                        continue
                    if not _altloc_included(atom):
                        continue
                    atoms.append((atom, cid, rid))
    return atoms


def _cov_bond_adjacency(
    positions: np.ndarray,
    cov_radii: np.ndarray,
) -> List[List[int]]:
    """Undirected graph: edge if interatomic distance suggests a covalent bond."""
    n = int(positions.shape[0])
    adj: List[List[int]] = [[] for _ in range(n)]
    for i in range(n):
        pi = positions[i]
        ci = float(cov_radii[i])
        for j in range(i + 1, n):
            d = float(np.linalg.norm(pi - positions[j]))
            if d > _MAX_SINGLE_BOND_DISTANCE_A:
                continue
            if d < ci + float(cov_radii[j]) + _BOND_PAD_A:
                _undirected_graph_edge(adj, i, j)
    return adj


def _nodes_within_k_graph_edges(adj: List[List[int]], start: int, k: int) -> set:
    """All atoms reachable from ``start`` along ≤ ``k`` inferred covalent edges."""
    seen = {start}
    frontier = [start]
    for _ in range(k):
        nxt: List[int] = []
        for u in frontier:
            for v in adj[u]:
                if v not in seen:
                    seen.add(v)
                    nxt.append(v)
        frontier = nxt
    return seen


def _within_k_bond_graph_edges(adj: List[List[int]], n: int, k: int = 3) -> List[set]:
    """For each atom, the set of atoms within ≤k bond steps (excludes 1–4 when k=3)."""
    return [_nodes_within_k_graph_edges(adj, s, k) for s in range(n)]


def _pair_excluded_as_bonded_1_4(
    within3: List[set],
    i: int,
    j: int,
) -> bool:
    """Exclude 1–2, 1–3, and 1–4 (Probe/MolProbity non-bonded contact shell)."""
    return j in within3[i]


@dataclass(frozen=True)
class HeavyAtomClashContext:
    """Precomputed heavy-atom lists, coordinates, radii, and covalent adjacency for clash shells."""

    atoms: List[Tuple[gemmi.Atom, str, str]]
    positions: np.ndarray
    vdw_radii: np.ndarray
    cov_radii: np.ndarray
    adj: List[List[int]]
    within3_bonds: List[set]
    bond_topology: str = "distance"


def build_heavy_atom_clash_context(
    structure: gemmi.Structure,
    monomer_lib_dir: Optional[str] = None,
) -> HeavyAtomClashContext:
    """Collect clash-scope atoms once and build a bond graph for 1–4 exclusion.

    When a CCP4-style monomer directory (with ``list/mon_lib_list.cif``) or per-residue
    ``*.cif`` files are found—via ``monomer_lib_dir``, ``CRYOMODEL_MONOMER_LIB``,
    ``CLIBD_MON``, ``PHENIX_CIF_MON_LIB_PATH``, or bundled ``data/monomers``—bonds come
    from :func:`gemmi.prepare_topology` plus standard peptide C–N links. Otherwise, or if
    that graph has no edges, connectivity falls back to distance vs covalent radii (previous
    behavior).
    """
    atoms = _collect_clash_scope_atoms(structure)
    if not atoms:
        return HeavyAtomClashContext(
            atoms=[],
            positions=np.zeros((0, 3), dtype=np.float64),
            vdw_radii=np.zeros(0, dtype=np.float64),
            cov_radii=np.zeros(0, dtype=np.float64),
            adj=[],
            within3_bonds=[],
            bond_topology="distance",
        )
    positions = np.stack([_atom_xyz(a) for a, _, _ in atoms], axis=0)
    vdw = np.array([_elem_vdw_cov(a)[0] for a, _, _ in atoms], dtype=np.float64)
    cov = np.array([_elem_vdw_cov(a)[1] for a, _, _ in atoms], dtype=np.float64)
    n = len(atoms)
    idx_by_atom_id = {id(a): i for i, (a, _, _) in enumerate(atoms)}

    adj: List[List[int]]
    bond_topology = "distance"
    roots = _monomer_search_roots(monomer_lib_dir)
    resnames = _collect_structure_resnames(structure)
    monlib = _try_load_monlib(resnames, roots)
    if monlib is not None:
        try:
            adj = _adjacency_from_gemmi_topology(structure, monlib, idx_by_atom_id, n)
        except Exception:
            adj = _cov_bond_adjacency(positions, cov)
        else:
            n_edges = sum(len(x) for x in adj) // 2
            if n_edges > 0:
                bond_topology = "gemmi"
            else:
                adj = _cov_bond_adjacency(positions, cov)
    else:
        adj = _cov_bond_adjacency(positions, cov)

    w3 = _within_k_bond_graph_edges(adj, n, k=3)
    return HeavyAtomClashContext(
        atoms=atoms,
        positions=positions,
        vdw_radii=vdw,
        cov_radii=cov,
        adj=adj,
        within3_bonds=w3,
        bond_topology=bond_topology,
    )


def steric_clash_counts_from_context(ctx: HeavyAtomClashContext) -> Dict[Tuple[str, str], int]:
    """Like :func:`steric_clash_counts_by_residue` using a pre-built context."""
    atoms = ctx.atoms
    n = len(atoms)
    counts: Dict[Tuple[str, str], int] = {}
    if n == 0:
        return counts
    pos = ctx.positions
    vdw = ctx.vdw_radii
    w3 = ctx.within3_bonds

    def bump(key: Tuple[str, str]) -> None:
        counts[key] = counts.get(key, 0) + 1

    for i in range(n):
        _, c1, r1 = atoms[i]
        p1 = pos[i]
        vdw1 = float(vdw[i])
        for j in range(i + 1, n):
            if _pair_excluded_as_bonded_1_4(w3, i, j):
                continue
            d = float(np.linalg.norm(p1 - pos[j]))
            vdw2 = float(vdw[j])
            if d < vdw1 + vdw2 + _PROBE_RADIUS_A:
                _, c2, r2 = atoms[j]
                bump((c1, r1))
                bump((c2, r2))
    return counts


def molprobity_like_clashscore_from_context(ctx: HeavyAtomClashContext) -> Tuple[float, int, int]:
    """Like :func:`molprobity_like_clashscore_heavy` using a pre-built context."""
    atoms = ctx.atoms
    n = len(atoms)
    if n == 0:
        return 0.0, 0, 0
    pos = ctx.positions
    vdw = ctx.vdw_radii
    w3 = ctx.within3_bonds
    n_pairs = 0
    for i in range(n):
        p1 = pos[i]
        vdw1 = float(vdw[i])
        for j in range(i + 1, n):
            if _pair_excluded_as_bonded_1_4(w3, i, j):
                continue
            d = float(np.linalg.norm(p1 - pos[j]))
            vdw2 = float(vdw[j])
            overlap = vdw1 + vdw2 - d
            if overlap >= _MOLPROBITY_OVERLAP_MIN_A - 1e-9:
                n_pairs += 1
    score = 1000.0 * float(n_pairs) / float(n)
    return float(score), int(n_pairs), int(n)


def steric_clash_counts_by_residue(structure: gemmi.Structure) -> Dict[Tuple[str, str], int]:
    """Heavy-atom steric overlaps (vdW + 0.25 Å probe).

    Excludes 1–2, 1–3, and 1–4 pairs using the same bond graph as
    :func:`build_heavy_atom_clash_context` (Gemmi MonLib when available, else distance-based).

    Returns counts keyed by ``(chain_name, str(residue.seqid))`` for any residue that has
    at least one atom in a clashing pair (each unordered pair counted once; attributed to both residues).
    """
    return steric_clash_counts_from_context(build_heavy_atom_clash_context(structure))


def count_clash_scoped_heavy_atoms(structure: gemmi.Structure) -> int:
    """Heavy atoms included in :func:`steric_clash_counts_by_residue` (altloc A/blank/1 only)."""
    return len(_collect_clash_scope_atoms(structure))


def steric_clash_pair_count(counts: Dict[Tuple[str, str], int]) -> int:
    """Number of unique heavy-atom clash pairs (each pair increments two residue tallies)."""
    return int(sum(counts.values()) // 2)


def molprobity_like_clashscore_heavy(
    structure: gemmi.Structure,
) -> Tuple[float, int, int]:
    """MolProbity-style clashscore: serious overlaps per 1000 **heavy** atoms.

    Counts each unordered atom pair once where vdW penetration
    ``(vdw_i + vdw_j) - distance`` is ≥ 0.4 Å. Pairs within 1–4 of the covalent graph are
    excluded (Gemmi ``prepare_topology`` + peptide C–N when MonLib is available, else
    distance-based inference), matching Probe/MolProbity non-bonded contact logic.

    Published MolProbity clashscores use all atoms including hydrogens (Reduce); without
    explicit H, the value is **not** numerically identical to PDB validation reports but
    should be in the same ballpark as ModBench ``clashscore_*`` for heavy-atom models.
    """
    return molprobity_like_clashscore_from_context(build_heavy_atom_clash_context(structure))


def _rama_score_from_class(class_: Optional[RamaClass]) -> float:
    """Continuous score in ~[0,1] for priors (higher = better backbone)."""
    if class_ is None:
        return 0.5
    if class_ == "favored":
        return 0.95
    if class_ == "allowed":
        return 0.65
    return 0.2


def _classify_rama_for_residue(
    prev_res: Optional[gemmi.Residue],
    residue: gemmi.Residue,
    next_res: Optional[gemmi.Residue],
) -> Tuple[float, Optional[RamaClass]]:
    rn = residue.name.strip().upper().split()[0][:3]
    if rn in ("GLY", "PRO"):
        return 0.5, None
    pp = phi_psi_deg(prev_res, residue, next_res)
    if pp is None:
        return 0.5, None
    cls = classify_phi_psi_general(pp[0], pp[1])
    return _rama_score_from_class(cls), cls


def _omega_deviation_rad(residue: gemmi.Residue, next_res: Optional[gemmi.Residue]) -> Optional[float]:
    """|ω − π| for trans peptide (radians); None if undefined."""
    if next_res is None:
        return None
    try:
        om = float(gemmi.calculate_omega(residue, next_res))
    except Exception:
        return None
    if math.isnan(om):
        return None
    # trans ~ π; cis ~ 0
    d_trans = abs(abs(om) - math.pi)
    d_cis = abs(om)
    return float(min(d_trans, d_cis))


def _cb_deviation_A(residue: gemmi.Residue) -> Optional[float]:
    """Distance of Cβ from an ideal tetrahedral placement (Å); None for GLY/ALA or missing atoms."""
    rn = residue.name.strip().upper().split()[0][:3]
    if rn in ("GLY", "ALA"):
        return None
    n = sole_atom(residue, "N")
    ca = sole_atom(residue, "CA")
    c = sole_atom(residue, "C")
    cb = sole_atom(residue, "CB")
    if not n or not ca or not c or not cb:
        return None
    a = np.array([n.pos.x - ca.pos.x, n.pos.y - ca.pos.y, n.pos.z - ca.pos.z], dtype=np.float64)
    b = np.array([c.pos.x - ca.pos.x, c.pos.y - ca.pos.y, c.pos.z - ca.pos.z], dtype=np.float64)
    la = np.linalg.norm(a)
    lb = np.linalg.norm(b)
    if la < 1e-6 or lb < 1e-6:
        return None
    a /= la
    b /= lb
    bis = -(a + b)
    lbis = np.linalg.norm(bis)
    if lbis < 1e-6:
        return None
    bis /= lbis
    # L-amino: Cβ along bisector, ~1.53 Å (Engh & Huber ballpark).
    ideal = np.array([ca.pos.x, ca.pos.y, ca.pos.z], dtype=np.float64) + 1.53 * bis
    actual = np.array([cb.pos.x, cb.pos.y, cb.pos.z], dtype=np.float64)
    return float(np.linalg.norm(actual - ideal))


def _rotamer_features(residue: gemmi.Residue) -> Tuple[float, float]:
    """Approximate rotamer quality: prior of nearest χ1 bin and angular deviation (deg)."""
    rn = residue.name.strip().upper().split()[0][:3]
    quad = chi1_quadruple(rn, residue)
    if quad is None:
        return 0.5, 0.0
    try:
        chi = chi1_dihedral_deg(residue, quad)
    except Exception:
        return 0.5, 0.0
    trials = CHI1_TRIAL_ANGLES
    priors = CHI1_PRIOR_TRIPLET.get(rn, (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0))
    best_i = 0
    best_d = 1e9
    for i, t in enumerate(trials):
        d = abs((chi - t + 180.0) % 360.0 - 180.0)
        if d < best_d:
            best_d = d
            best_i = i
    prior = float(priors[best_i])
    # Down-weight large deviations from the nearest canonical well.
    w = math.exp(-((best_d / 25.0) ** 2))
    score = min(1.0, prior * (0.35 + 0.65 * w))
    return score, float(best_d)


def _clash_z_for_residue(n_clashes: int, n_heavy: int, global_mean: float, global_std: float) -> float:
    """Z-score of this residue's clash *density* vs the whole structure.

    If every residue has the same density (including all zeros), there is no
    cross-residue spread: return NaN instead of 0 so it is not mistaken for
    "good on an absolute scale".
    """
    if n_heavy <= 0:
        return 0.0
    if global_std is None or not math.isfinite(global_std) or global_std < 1e-12:
        return float("nan")
    density = n_clashes / n_heavy
    return float((density - global_mean) / global_std)


def compute_global_clash_z_stats(
    structure: gemmi.Structure,
    clash_counts: Optional[Dict[Tuple[str, str], int]] = None,
) -> Tuple[float, float]:
    """Mean and population std of per-residue clash density (clashes / n_heavy).

    Std is NaN when all residues share the same density (e.g. all zero clashes),
    so downstream ``clashscore_z`` is NaN rather than a column of misleading zeros.

    Pass ``clash_counts`` from :func:`steric_clash_counts_from_context` to avoid a second clash scan.
    """
    if clash_counts is None:
        clash_counts = steric_clash_counts_by_residue(structure)
    densities = []
    for model in structure:
        for chain in model:
            for res in chain:
                n_h = sum(1 for a in res if not a.is_hydrogen())
                if n_h == 0:
                    continue
                key = (chain.name, str(res.seqid))
                c = clash_counts.get(key, 0)
                densities.append(c / n_h)
    if not densities:
        return 0.0, float("nan")
    arr = np.array(densities, dtype=np.float64)
    mean = float(arr.mean())
    std = float(arr.std(ddof=0))
    if not math.isfinite(std) or std < 1e-12:
        return mean, float("nan")
    return mean, std


def compute_geometry_features(
    residue: gemmi.Residue,
    chain_residues: List[gemmi.Residue],
    residue_index: int,
    chain_name: str,
    clash_counts: Dict[Tuple[str, str], int],
    clash_mean_density: float,
    clash_std_density: float,
) -> Dict[str, float]:
    """Per-residue geometry features (MolProbity-class lite)."""
    prev_r = chain_residues[residue_index - 1] if residue_index > 0 else None
    next_r = chain_residues[residue_index + 1] if residue_index + 1 < len(chain_residues) else None

    rama_prob, rama_class = _classify_rama_for_residue(prev_r, residue, next_r)
    rama_outlier = 1.0 if rama_class == "outlier" else 0.0
    rama_pen = rama_penalty(rama_class) if rama_class is not None else 0.0

    omega_dev = _omega_deviation_rad(residue, next_r)
    if omega_dev is None:
        omega_dev_deg = float("nan")
        peptide_twist_score = 0.5
    else:
        omega_dev_deg = float(np.rad2deg(omega_dev))
        # 1.0 if trans-like (within ~15°), falling off toward cis/strained
        peptide_twist_score = float(math.exp(-((omega_dev / 0.35) ** 2)))

    cb_dev = _cb_deviation_A(residue)
    cb_deviation_A = float(cb_dev) if cb_dev is not None else float("nan")

    rotamer_prob, rotamer_chi1_nearest_deg = _rotamer_features(residue)

    seq_key = (chain_name, str(residue.seqid))
    n_clash = clash_counts.get(seq_key, 0)
    n_heavy = sum(1 for a in residue if not a.is_hydrogen())
    clashscore_z = _clash_z_for_residue(n_clash, n_heavy, clash_mean_density, clash_std_density)

    # Legacy scalar for YAML priors (higher = fewer / less severe clashes in a loose sense)
    clashscore_severity_approx = float(n_clash) / float(max(n_heavy, 1))

    return {
        "ramachandran_prob": float(rama_prob),
        "rama_outlier": float(rama_outlier),
        "rama_penalty": float(rama_pen),
        "omega_dev_deg": omega_dev_deg,
        "peptide_twist_score": float(peptide_twist_score),
        "cb_deviation_A": cb_deviation_A,
        "rotamer_prob": float(rotamer_prob),
        "rotamer_chi1_nearest_deg": float(rotamer_chi1_nearest_deg),
        "steric_clashes": float(n_clash),
        "clashscore_z": float(clashscore_z),
        "clashscore_severity_approx": float(clashscore_severity_approx),
        # Back-compat keys used in older CSV / priors (map to new semantics)
        "cablam_flag": float(rama_outlier),
        "peptide_planarity_z": float(omega_dev_deg) if math.isfinite(omega_dev_deg) else 0.0,
        "cb_deviation_z": float(cb_deviation_A) if math.isfinite(cb_deviation_A) else 0.0,
    }
