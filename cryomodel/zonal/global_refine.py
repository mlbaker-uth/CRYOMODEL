"""Global zonal refinement: overlapping GMM regions + local χ1 / Rama on NCS chains in-zone, with propagation to off-sphere copies."""
from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import gemmi
import numpy as np
from sklearn.mixture import GaussianMixture

from ..domains.domain_identifier import parse_sse_from_pdb_header
from ..io.mrc import MapVolume
from ..mutate.chi import chi1_dihedral_deg, chi1_quadruple, rotate_sidechain_chi1
from .backbone import rotate_phi, rotate_psi
from .refine import run_zonal_chi_refine
from .zone import residues_in_sphere


def parse_ncs_chains(ncs: str) -> Tuple[str, List[str]]:
    """
    Parse ``--ncs`` string: first chain is **master**, remaining are **copies**.

    ``"A"`` → master ``A``, no copies. ``"A,B,C"`` → master ``A``, copies ``[B, C]``.
    """
    parts = [p.strip() for p in str(ncs).split(",") if p.strip()]
    if not parts:
        raise ValueError("NCS string is empty (expected e.g. 'A' or 'A,B,C').")
    master = parts[0]
    copies = parts[1:]
    return master, copies


def _find_chain(st: gemmi.Structure, chain_id: str) -> Optional[gemmi.Chain]:
    for model in st:
        ch = model.find_chain(chain_id)
        if ch:
            return ch
    return None


def _resname_3(res: gemmi.Residue) -> str:
    return res.name.strip().upper().split()[0][:3]


def _residue_seqnum(res: gemmi.Residue) -> int:
    return int(res.seqid.num)


def _wrap_delta_deg(delta: float) -> float:
    return (delta + 180.0) % 360.0 - 180.0


def _residue_index_in_chain(ch: gemmi.Chain, res: gemmi.Residue) -> int:
    for i, r in enumerate(ch):
        if r.seqid == res.seqid and r.name.strip() == res.name.strip():
            return i
    return -1


def _residue_heavy_in_sphere(
    res: gemmi.Residue,
    center_xyz: np.ndarray,
    radius: float,
) -> bool:
    """True if any non-hydrogen atom of ``res`` lies within ``radius`` Å of ``center_xyz``."""
    c = np.asarray(center_xyz, dtype=np.float64).reshape(3)
    r2 = float(radius) * float(radius)
    for atom in res:
        if atom.element.name == "H":
            continue
        dx = atom.pos.x - c[0]
        dy = atom.pos.y - c[1]
        dz = atom.pos.z - c[2]
        if dx * dx + dy * dy + dz * dz <= r2:
            return True
    return False


def _find_residue_on_chain(st: gemmi.Structure, chain_id: str, seqid_str: str, resname: str) -> Optional[gemmi.Residue]:
    ch = _find_chain(st, chain_id)
    if ch is None:
        return None
    want_name = resname.strip().upper()
    for res in ch:
        if str(res.seqid) == seqid_str and res.name.strip().upper() == want_name:
            return res
    return None


def collect_master_ca(
    st: gemmi.Structure,
    master_chain: str,
) -> Tuple[List[gemmi.Residue], np.ndarray]:
    """Ordered polymer residues with a CA atom, and ``(n, 3)`` Cα coordinates."""
    ch = _find_chain(st, master_chain)
    if ch is None:
        raise ValueError(f"Master chain {master_chain!r} not found.")
    residues: List[gemmi.Residue] = []
    for res in ch:
        try:
            res.sole_atom("CA")
        except Exception:
            continue
        residues.append(res)
    if not residues:
        raise ValueError(f"No residues with CA on chain {master_chain!r}.")
    xyz = np.array([[res.sole_atom("CA").pos.x, res.sole_atom("CA").pos.y, res.sole_atom("CA").pos.z] for res in residues], dtype=np.float64)
    return residues, xyz


def snapshot_chain_ca(st: gemmi.Structure, chain_id: str) -> np.ndarray:
    ch = _find_chain(st, chain_id)
    if ch is None:
        return np.zeros((0, 3), dtype=np.float64)
    pts: List[Tuple[float, float, float]] = []
    for res in ch:
        try:
            a = res.sole_atom("CA")
        except Exception:
            continue
        pts.append((a.pos.x, a.pos.y, a.pos.z))
    return np.array(pts, dtype=np.float64) if pts else np.zeros((0, 3), dtype=np.float64)


def rmsd_xyz(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0 or a.shape != b.shape:
        return float("nan")
    d = a - b
    return float(np.sqrt(np.mean(np.sum(d * d, axis=1))))


def build_overlapping_gmm_regions(
    X: np.ndarray,
    *,
    target_residues_per_region: int,
    n_components: Optional[int] = None,
    soft_resp_floor: float,
    random_state: int,
    reg_covar: float,
) -> List[Set[int]]:
    """
    Fit a 3D GMM and build **overlapping** region index sets (master residue indices).

    Residue *i* belongs to region *k* if posterior ``resp[i, k] >= soft_resp_floor``.
    Ensures every residue appears in at least one region.

    If ``n_components`` is set, use that many mixture components (clamped to ``1 … N``).
    Otherwise set ``K ≈ N / target_residues_per_region``.
    """
    n, dim = X.shape
    if dim != 3:
        raise ValueError("Expected Nx3 Cα coordinates.")
    if n_components is not None:
        n_comp = max(1, min(int(n_components), n))
    else:
        n_comp = max(1, min(n, n // max(target_residues_per_region, 1)))
    rs = np.random.RandomState(random_state)
    gm = GaussianMixture(
        n_components=n_comp,
        covariance_type="full",
        reg_covar=max(reg_covar, 1e-6),
        random_state=rs,
        max_iter=200,
        n_init=2,
    )
    gm.fit(X)
    resp = gm.predict_proba(X)
    floor = float(np.clip(soft_resp_floor, 1e-6, 1.0))
    regions: List[Set[int]] = []
    for k in range(n_comp):
        idx = [i for i in range(n) if resp[i, k] >= floor]
        if not idx:
            idx = [int(np.argmax(resp[:, k]))]
        regions.append(set(idx))
    for i in range(n):
        if not any(i in s for s in regions):
            k_best = int(np.argmax(resp[i]))
            regions[k_best].add(i)
    return regions


def expand_regions_for_sse(
    master_residues: Sequence[gemmi.Residue],
    regions: List[Set[int]],
    sse_resnums: Set[int],
) -> None:
    """
    In-place: for each contiguous SSE run (by sequence number on the master chain),
    if any residue in the run appears in a region, add **all** residues of that run
    to every region that touches the run (avoids splitting HELIX/SHEET from PDB header).
    """
    n = len(master_residues)
    if n == 0 or not sse_resnums:
        return
    seqnums = [_residue_seqnum(r) for r in master_residues]
    i = 0
    while i < n:
        if seqnums[i] not in sse_resnums:
            i += 1
            continue
        j = i
        while j + 1 < n and seqnums[j + 1] in sse_resnums:
            j += 1
        seg = set(range(i, j + 1))
        touched = [ri for ri, s in enumerate(regions) if seg & s]
        if not touched:
            i = j + 1
            continue
        for ri in touched:
            regions[ri].update(seg)
        i = j + 1


def region_center_radius(
    X: np.ndarray,
    indices: Set[int],
    pad: float,
) -> Tuple[np.ndarray, float]:
    """Centroid and radius (max CA distance + pad)."""
    idx = sorted(indices)
    pts = X[idx]
    c = np.mean(pts, axis=0)
    dist = np.linalg.norm(pts - c.reshape(1, 3), axis=1)
    r = float(np.max(dist)) + float(pad)
    r = max(r, 2.0)
    return c, r


def copy_chain_region_sphere(
    st: gemmi.Structure,
    copy_chain: str,
    idx_set: Set[int],
    master_ca_residues: Sequence[gemmi.Residue],
    radius_pad: float,
) -> Tuple[Optional[np.ndarray], float]:
    """
    Same logical region as ``idx_set`` on the master (indices into ``master_ca_residues``),
    but a sphere built from **copy** chain Cα positions so a distant NCS partner gets its own
    in-map local refinement.
    """
    pts: List[List[float]] = []
    for j in sorted(idx_set):
        if j < 0 or j >= len(master_ca_residues):
            continue
        mres = master_ca_residues[j]
        cres = _find_residue_on_chain(st, copy_chain, str(mres.seqid), mres.name.strip())
        if cres is None:
            continue
        try:
            ca = cres.sole_atom("CA")
        except Exception:
            continue
        pts.append([ca.pos.x, ca.pos.y, ca.pos.z])
    if len(pts) < 1:
        return None, 0.0
    X = np.array(pts, dtype=np.float64)
    c = np.mean(X, axis=0)
    dist = np.linalg.norm(X - c.reshape(1, 3), axis=1)
    r = float(np.max(dist)) + float(radius_pad)
    r = max(r, 2.0)
    return c, r


def propagate_chi1_ncs(
    st: gemmi.Structure,
    *,
    master_chain: str,
    copy_chains: Sequence[str],
    master_residues_in_zone: Sequence[gemmi.Residue],
    center_xyz: Optional[np.ndarray] = None,
    zone_radius: Optional[float] = None,
) -> int:
    """
    For each master residue in the zone with χ1, set copy-chain χ1 to match master (torsion space).

    If ``center_xyz`` and ``zone_radius`` are set, skip a copy residue when **any** of its heavy
    atoms already lie in the local sphere (that copy was refined in-place by the joint local run).
    """
    if not copy_chains:
        return 0
    use_sphere = center_xyz is not None and zone_radius is not None and float(zone_radius) > 0
    n_updates = 0
    for mres in master_residues_in_zone:
        rn = _resname_3(mres)
        quad_m = chi1_quadruple(rn, mres)
        if quad_m is None:
            continue
        target_deg = chi1_dihedral_deg(mres, quad_m)
        m_seq = str(mres.seqid)
        m_name = mres.name.strip()
        for cc in copy_chains:
            cres = _find_residue_on_chain(st, cc, m_seq, m_name)
            if cres is None:
                continue
            if use_sphere and _residue_heavy_in_sphere(cres, center_xyz, float(zone_radius)):
                continue
            quad_c = chi1_quadruple(rn, cres)
            if quad_c is None:
                continue
            cur = chi1_dihedral_deg(cres, quad_c)
            delta = _wrap_delta_deg(target_deg - cur)
            if abs(delta) > 1e-7:
                rotate_sidechain_chi1(cres, quad_c, delta)
                n_updates += 1
    return n_updates


def propagate_rama_deltas_ncs(
    st: gemmi.Structure,
    *,
    master_chain: str,
    copy_chains: Sequence[str],
    residue_log: Sequence[Dict[str, Any]],
    center_xyz: np.ndarray,
    zone_radius: float,
) -> int:
    """
    Apply master's accepted Rama Δφ, Δψ (from ``residue_log``) to NCS copy residues that were
    **outside** the local sphere (copies inside the sphere were already updated by the joint local run).
    """
    if not copy_chains:
        return 0
    n = 0
    for entry in residue_log:
        if entry.get("stage") != "rama":
            continue
        if entry.get("chain") != master_chain:
            continue
        dphi = float(entry.get("dphi_deg", 0.0))
        dpsi = float(entry.get("dpsi_deg", 0.0))
        if abs(dphi) < 1e-9 and abs(dpsi) < 1e-9:
            continue
        seqid_str = str(entry.get("seqid", ""))
        resname = str(entry.get("resname", "ALA"))
        rn_u = resname.strip().upper()[:3]
        if rn_u in ("GLY", "PRO"):
            continue
        for cc in copy_chains:
            cres = _find_residue_on_chain(st, cc, seqid_str, resname)
            if cres is None:
                continue
            if _residue_heavy_in_sphere(cres, center_xyz, zone_radius):
                continue
            ch = _find_chain(st, cc)
            if ch is None:
                continue
            ri = _residue_index_in_chain(ch, cres)
            if ri < 0:
                continue
            if dphi != 0.0:
                rotate_phi(ch, ri, dphi)
            if dpsi != 0.0:
                rotate_psi(ch, ri, dpsi)
            n += 1
    return n


def _master_residues_in_hard_zone(
    st: gemmi.Structure,
    center: np.ndarray,
    radius: float,
    master_chain: str,
) -> List[gemmi.Residue]:
    zone = residues_in_sphere(st, center, radius, chain_filter={master_chain})
    return [r for cid, r in zone if cid == master_chain]


@dataclass
class GlobalZonalResult:
    """Summary of a global zonal refinement run."""

    rounds_done: int = 0
    stopped_reason: str = ""
    region_count: int = 0
    final_ca_rmsd_vs_initial: float = float("nan")
    round_log: List[Dict[str, Any]] = field(default_factory=list)
    elapsed_sec: float = 0.0
    meta: Dict[str, Any] = field(default_factory=dict)


def run_global_zonal_refine(
    st: gemmi.Structure,
    map_vol: MapVolume,
    *,
    pdb_path: Optional[Path],
    master_chain: str,
    copy_chains: Optional[Sequence[str]] = None,
    target_residues_per_region: int = 30,
    gmm_components: Optional[int] = None,
    soft_resp_floor: float = 0.12,
    radius_pad: float = 4.0,
    max_rounds: int = 7,
    converge_rmsd_eps: float = 0.03,
    converge_patience: int = 2,
    random_seed: Optional[int] = 0,
    sse_from_pdb_header: bool = True,
    gmm_reg_covar: float = 1e-4,
    ncs_mirror_zones: bool = True,
    progress: Optional[Callable[[str], None]] = None,
    **local_kw: Any,
) -> GlobalZonalResult:
    """
    Overlapping GMM regions on the master chain Cα cloud; each macro-round shuffles regions and runs
    local refinement on the **master** in a sphere around that patch, then (when ``ncs_mirror_zones``
    and copy chains exist) runs **separate** local passes on each copy using a sphere built from the
    **same** residue mapping in the copy's coordinates—so homomers separated in space still feel map+clash.

    χ1 / Rama on the master are **propagated** to copy residues that lie **outside** the master's sphere
    (unchanged when mirror passes already covered those sites).

    Regions are fit **once** from the initial Cα layout (GMM membership fixed; sphere centers follow current Cα).

    ``progress`` is an optional ``callable(str)`` (e.g. log to stderr) for coarse-grained status lines.
    """
    t0 = time.perf_counter()
    copies = list(copy_chains or [])
    for cc in copies:
        if _find_chain(st, cc) is None:
            raise ValueError(f"NCS copy chain {cc!r} not found.")

    ncs_chain_filter: Set[str] = {master_chain} | set(copies)

    master_res, X0 = collect_master_ca(st, master_chain)
    X_work = np.array(X0, copy=True)

    sse_resnums: Set[int] = set()
    if sse_from_pdb_header and pdb_path is not None:
        suf = pdb_path.suffix.lower()
        if suf in (".pdb", ".ent"):
            try:
                sse_resnums = parse_sse_from_pdb_header(Path(pdb_path), master_chain)
            except OSError:
                sse_resnums = set()
        else:
            sse_resnums = set()

    rs = int(random_seed) if random_seed is not None else int(time.time()) % (2**31)
    regions = build_overlapping_gmm_regions(
        X_work,
        target_residues_per_region=max(5, int(target_residues_per_region)),
        n_components=gmm_components,
        soft_resp_floor=float(soft_resp_floor),
        random_state=rs,
        reg_covar=float(gmm_reg_covar),
    )
    expand_regions_for_sse(master_res, regions, sse_resnums)

    initial_ca = snapshot_chain_ca(st, master_chain)
    out = GlobalZonalResult()
    out.region_count = len(regions)
    out.meta = {
        "master_chain": master_chain,
        "copy_chains": list(copies),
        "target_residues_per_region": int(target_residues_per_region),
        "gmm_components_requested": int(gmm_components) if gmm_components is not None else None,
        "gmm_components_used": len(regions),
        "soft_resp_floor": float(soft_resp_floor),
        "radius_pad": float(radius_pad),
        "max_rounds": int(max_rounds),
        "converge_rmsd_eps": float(converge_rmsd_eps),
        "converge_patience": int(converge_patience),
        "random_seed": rs,
        "sse_from_header": bool(sse_resnums),
        "n_master_residues": len(master_res),
        "gmm_regions": [sorted(s) for s in regions],
        "ncs_chain_filter": sorted(ncs_chain_filter),
        "ncs_mirror_zones": bool(ncs_mirror_zones and copies),
    }

    if progress:
        progress(
            f"zonal-refine global: {len(regions)} overlapping regions, "
            f"{len(master_res)} master Cα, ≤{max_rounds} macro-round(s); "
            f"joint={'master' if (copies and ncs_mirror_zones) else 'all'} {sorted(ncs_chain_filter)}; "
            f"mirror={bool(ncs_mirror_zones and copies)}"
        )

    plateau_count = 0
    rng = random.Random(rs)

    for round_i in range(max_rounds):
        if progress:
            progress(f"— macro-round {round_i + 1}/{max_rounds} —")
        ca_before_round = snapshot_chain_ca(st, master_chain)
        order = list(range(len(regions)))
        rng.shuffle(order)
        round_entries: List[Dict[str, Any]] = []

        for j, rk in enumerate(order, start=1):
            _, X_current = collect_master_ca(st, master_chain)
            idx_set = regions[rk]
            center, radius = region_center_radius(X_current, idx_set, radius_pad)
            joint_filter: Set[str] = (
                {master_chain} if (copies and ncs_mirror_zones) else ncs_chain_filter
            )
            local_result = run_zonal_chi_refine(
                st,
                map_vol,
                center,
                radius,
                chain_filter=joint_filter,
                **local_kw,
            )
            hard_master = _master_residues_in_hard_zone(st, center, radius, master_chain)
            n_rama_prop = propagate_rama_deltas_ncs(
                st,
                master_chain=master_chain,
                copy_chains=copies,
                residue_log=local_result.residue_log,
                center_xyz=center,
                zone_radius=radius,
            )
            n_prop = propagate_chi1_ncs(
                st,
                master_chain=master_chain,
                copy_chains=copies,
                master_residues_in_zone=hard_master,
                center_xyz=center,
                zone_radius=radius,
            )
            mirror_locals: List[Dict[str, Any]] = []
            if copies and ncs_mirror_zones:
                master_res_now, _ = collect_master_ca(st, master_chain)
                for cc in copies:
                    c_c, r_c = copy_chain_region_sphere(st, cc, idx_set, master_res_now, radius_pad)
                    if c_c is None:
                        mirror_locals.append({"chain": cc, "skipped": True, "reason": "no_matching_ca"})
                        continue
                    loc_c = run_zonal_chi_refine(
                        st,
                        map_vol,
                        c_c,
                        r_c,
                        chain_filter={cc},
                        **local_kw,
                    )
                    mirror_locals.append(
                        {
                            "chain": cc,
                            "center": [float(x) for x in c_c],
                            "radius": float(r_c),
                            "residues_in_zone": loc_c.residues_in_zone,
                            "improvements_last_pass": loc_c.improvements_in_last_pass,
                            "rama_improvements": loc_c.rama_improvements,
                            "elapsed_sec": round(loc_c.elapsed_sec, 4),
                        }
                    )
            if progress:
                mir_s = ""
                if mirror_locals:
                    parts = [f"{m['chain']}:{m.get('residues_in_zone', 0)}" for m in mirror_locals]
                    mir_s = "  mirror[" + ",".join(parts) + "]"
                progress(
                    f"  zone {j}/{len(order)}  reg#{rk}  R={radius:.1f}Å  "
                    f"hard_res={local_result.residues_in_zone}  "
                    f"χ1_last+={local_result.improvements_in_last_pass}  "
                    f"rama+={local_result.rama_improvements}  "
                    f"local_s={local_result.elapsed_sec:.2f}  "
                    f"ncs_χ_prop={n_prop}  ncs_rama_prop={n_rama_prop}"
                    f"{mir_s}"
                )
            round_entries.append(
                {
                    "region_index": rk,
                    "center": [float(x) for x in center],
                    "radius": float(radius),
                    "joint_chain_filter": sorted(joint_filter),
                    "local": {
                        "residues_in_zone": local_result.residues_in_zone,
                        "improvements_last_pass": local_result.improvements_in_last_pass,
                        "rama_improvements": local_result.rama_improvements,
                        "elapsed_sec": round(local_result.elapsed_sec, 4),
                    },
                    "ncs_chi_updates": n_prop,
                    "ncs_rama_propagations": n_rama_prop,
                    "ncs_mirror_locals": mirror_locals,
                }
            )

        ca_after_round = snapshot_chain_ca(st, master_chain)
        delta = rmsd_xyz(ca_before_round, ca_after_round)
        out.round_log.append(
            {
                "round": round_i,
                "master_ca_rmsd_start_end": round(delta, 6) if math.isfinite(delta) else None,
                "regions": round_entries,
            }
        )
        out.rounds_done = round_i + 1

        if math.isfinite(delta) and delta < converge_rmsd_eps:
            plateau_count += 1
            if plateau_count >= converge_patience:
                out.stopped_reason = "converged_rmsd"
                break
        else:
            plateau_count = 0
    else:
        out.stopped_reason = "max_rounds"

    out.final_ca_rmsd_vs_initial = rmsd_xyz(initial_ca, snapshot_chain_ca(st, master_chain))
    out.elapsed_sec = time.perf_counter() - t0
    if progress:
        reason = out.stopped_reason or "done"
        rms = out.final_ca_rmsd_vs_initial
        rms_s = f"{rms:.4f}" if math.isfinite(rms) else "nan"
        progress(
            f"done: {out.rounds_done} macro-round(s), {reason}, "
            f"master Cα RMSD vs start {rms_s} Å ({out.elapsed_sec:.1f}s total)"
        )
    return out


def write_global_result_json(path: Path, result: GlobalZonalResult) -> None:
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "rounds_done": result.rounds_done,
        "stopped_reason": result.stopped_reason,
        "region_count": result.region_count,
        "final_ca_rmsd_vs_initial": round(result.final_ca_rmsd_vs_initial, 6)
        if math.isfinite(result.final_ca_rmsd_vs_initial)
        else None,
        "elapsed_sec": round(result.elapsed_sec, 4),
        "meta": result.meta,
        "round_log": result.round_log,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
