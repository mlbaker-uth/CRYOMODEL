"""Local χ1 zonal refinement in a spherical mask (map + clash + rotamer prior)."""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import gemmi
import numpy as np

from ..io.mrc import MapVolume
from ..mutate.chi import (
    chi1_quadruple,
    map_fit_anchor_term,
    mean_sidechain_map_value,
    pick_best_chi1,
)
from ..mutate.guide_metrics import total_clash_for_residue
from ..validation.ringer_lite import sample_density_at_position
from .backbone import rotate_phi, rotate_psi
from .ramachandran import classify_residue_backbone, rama_penalty
from .zone import partition_hard_soft_spherical


@dataclass
class ZonalRefineResult:
    """Summary of a zonal χ1 refinement run."""

    residues_in_zone: int = 0  # hard zone (same meaning as A0)
    residues_soft_zone: int = 0
    residues_with_chi1: int = 0  # χ1-capable in hard zone
    residues_soft_with_chi1: int = 0
    passes_done: int = 0
    improvements_in_last_pass: int = 0
    soft_passes_done: int = 0
    improvements_soft_last_pass: int = 0
    rama_residues_tried: int = 0
    rama_improvements: int = 0
    elapsed_sec: float = 0.0
    residue_log: List[Dict[str, Any]] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)


def _resname_3(res: gemmi.Residue) -> str:
    return res.name.strip().upper().split()[0][:3]


def _simple_objective(
    st: gemmi.Structure,
    chain_id: str,
    res: gemmi.Residue,
    map_vol: MapVolume,
    weight_map: float,
    map_density_threshold: float = 0.0,
) -> float:
    """Clash − weight_map × mean(side-chain map); rotamer handled inside pick_best_chi1."""
    c = float(total_clash_for_residue(st, chain_id, res))
    if weight_map <= 0:
        return c
    m = float(mean_sidechain_map_value(res, map_vol, density_threshold=map_density_threshold))
    return c - weight_map * m


def _build_movable(
    zoned: List[Tuple[str, gemmi.Residue]],
) -> List[Tuple[str, gemmi.Residue, str, Tuple[str, str, str, str]]]:
    out: List[Tuple[str, gemmi.Residue, str, Tuple[str, str, str, str]]] = []
    for chain_id, res in zoned:
        rn = _resname_3(res)
        quad = chi1_quadruple(rn, res)
        if quad is None:
            continue
        out.append((chain_id, res, rn, quad))
    return out


def _res_key(chain_id: str, res: gemmi.Residue) -> Tuple[str, str]:
    return (chain_id, str(res.seqid))


def _find_chain(st: gemmi.Structure, chain_id: str) -> Optional[gemmi.Chain]:
    for model in st:
        ch = model.find_chain(chain_id)
        if ch:
            return ch
    return None


def _residue_index_in_chain(ch: gemmi.Chain, res: gemmi.Residue) -> int:
    for i, r in enumerate(ch):
        if r.seqid == res.seqid and r.name.strip() == res.name.strip():
            return i
    return -1


def _mean_heavy_atom_map(
    res: gemmi.Residue,
    map_vol: MapVolume,
    density_threshold: float = 0.0,
) -> float:
    vals: List[float] = []
    for atom in res:
        if atom.element.name == "H":
            continue
        p = np.array([atom.pos.x, atom.pos.y, atom.pos.z], dtype=np.float64)
        vals.append(sample_density_at_position(map_vol, p, density_threshold=density_threshold))
    return float(sum(vals) / max(len(vals), 1))


def _snapshot_all_positions(st: gemmi.Structure) -> List[Tuple[float, float, float]]:
    out: List[Tuple[float, float, float]] = []
    for model in st:
        for chain in model:
            for res in chain:
                for atom in res:
                    out.append((atom.pos.x, atom.pos.y, atom.pos.z))
    return out


def _restore_all_positions(st: gemmi.Structure, snap: List[Tuple[float, float, float]]) -> None:
    i = 0
    for model in st:
        for chain in model:
            for res in chain:
                for atom in res:
                    x, y, z = snap[i]
                    atom.pos = gemmi.Position(x, y, z)
                    i += 1


def _rama_objective(
    st: gemmi.Structure,
    chain_id: str,
    res: gemmi.Residue,
    map_vol: MapVolume,
    weight_map: float,
    prev: Optional[gemmi.Residue],
    next_r: Optional[gemmi.Residue],
    weight_rama: float,
    dphi: float,
    dpsi: float,
    weight_bb_move: float,
    map_density_threshold: float = 0.0,
    m_density_anchor: Optional[float] = None,
    weight_density_anchor: float = 0.0,
    weight_density_gain: float = 0.0,
    map_anchor_eps: float = 1e-5,
) -> float:
    m_now = _mean_heavy_atom_map(res, map_vol, density_threshold=map_density_threshold)
    c = float(total_clash_for_residue(st, chain_id, res))
    if weight_map > 0:
        c -= weight_map * m_now
    cls = classify_residue_backbone(prev, res, next_r)
    rp = rama_penalty(cls) if cls else 0.0
    move = dphi * dphi + dpsi * dpsi
    anchor_term = 0.0
    if m_density_anchor is not None:
        anchor_term = map_fit_anchor_term(
            m_now,
            m_density_anchor,
            weight_anchor=weight_density_anchor,
            weight_gain=weight_density_gain,
            eps=map_anchor_eps,
        )
    return c + weight_rama * rp + weight_bb_move * move + anchor_term


def _grid_angles(step_deg: float, max_abs_deg: float) -> List[float]:
    if max_abs_deg <= 0 or step_deg <= 0:
        return [0.0]
    n = int(math.floor(max_abs_deg / step_deg + 1e-9))
    return [k * step_deg for k in range(-n, n + 1)]


def _rama_fragment_indices_in_zone(
    chain: gemmi.Chain,
    res_index: int,
    zone_indices: Set[int],
) -> Tuple[bool, bool]:
    """
    Whether a φ and/or ψ move at ``res_index`` affects **only** residues whose indices lie in
    ``zone_indices``.

    ``rotate_phi`` moves all residues with index ``< res_index``; ``rotate_psi`` moves all with
    index ``> res_index``. Without this check, a single "local" Rama step can rigidly translate
    half the chain — catastrophic for overlapping global refinement.
    """
    n = len(chain)
    phi_ok = all(k in zone_indices for k in range(0, res_index))
    psi_ok = all(k in zone_indices for k in range(res_index + 1, n))
    return phi_ok, psi_ok


def _rama_backbone_micro(
    st: gemmi.Structure,
    map_vol: MapVolume,
    zone: List[Tuple[str, gemmi.Residue]],
    *,
    weight_map: float,
    weight_rama: float,
    weight_bb_move: float,
    rama_step_deg: float,
    rama_max_shift_deg: float,
    rama_nudge_favored: bool,
    map_density_threshold: float,
    weight_density_anchor: float,
    weight_density_gain: float,
    map_anchor_eps: float,
    result: ZonalRefineResult,
) -> None:
    """Small φ/ψ grid search on non-GLY/PRO residues in zone (mutates ``st``)."""
    zone_by_chain: Dict[str, Set[int]] = {}
    for cid, r in zone:
        ch = _find_chain(st, cid)
        if ch is None:
            continue
        ri = _residue_index_in_chain(ch, r)
        if ri < 0:
            continue
        zone_by_chain.setdefault(cid, set()).add(ri)

    phis = _grid_angles(rama_step_deg, rama_max_shift_deg)
    psis = phis
    for chain_id, res in zone:
        rn = _resname_3(res)
        if rn in ("GLY", "PRO"):
            continue
        ch = _find_chain(st, chain_id)
        if ch is None:
            continue
        ri = _residue_index_in_chain(ch, res)
        if ri < 0:
            continue
        zidx = zone_by_chain.get(chain_id, set())
        phi_ok, psi_ok = _rama_fragment_indices_in_zone(ch, ri, zidx)
        if not phi_ok and not psi_ok:
            continue
        prev = ch[ri - 1] if ri > 0 else None
        nxt = ch[ri + 1] if ri + 1 < len(ch) else None
        cls0 = classify_residue_backbone(prev, res, nxt)
        if cls0 is None:
            continue
        if cls0 == "favored" and not rama_nudge_favored:
            continue
        result.rama_residues_tried += 1
        snap0 = _snapshot_all_positions(st)
        use_fit_anchor = weight_density_anchor > 0.0 or weight_density_gain > 0.0
        m_density_anchor = (
            _mean_heavy_atom_map(res, map_vol, density_threshold=map_density_threshold)
            if use_fit_anchor
            else None
        )
        score0 = _rama_objective(
            st,
            chain_id,
            res,
            map_vol,
            weight_map,
            prev,
            nxt,
            weight_rama,
            0.0,
            0.0,
            weight_bb_move,
            map_density_threshold=map_density_threshold,
            m_density_anchor=m_density_anchor,
            weight_density_anchor=weight_density_anchor,
            weight_density_gain=weight_density_gain,
            map_anchor_eps=map_anchor_eps,
        )
        best_dphi = 0.0
        best_dpsi = 0.0
        best_score = score0
        for dphi in phis:
            for dpsi in psis:
                if abs(dphi) < 1e-12 and abs(dpsi) < 1e-12:
                    continue
                if dphi != 0.0 and not phi_ok:
                    continue
                if dpsi != 0.0 and not psi_ok:
                    continue
                _restore_all_positions(st, snap0)
                if dphi != 0.0:
                    rotate_phi(ch, ri, dphi)
                if dpsi != 0.0:
                    rotate_psi(ch, ri, dpsi)
                sc = _rama_objective(
                    st,
                    chain_id,
                    res,
                    map_vol,
                    weight_map,
                    prev,
                    nxt,
                    weight_rama,
                    dphi,
                    dpsi,
                    weight_bb_move,
                    map_density_threshold=map_density_threshold,
                    m_density_anchor=m_density_anchor,
                    weight_density_anchor=weight_density_anchor,
                    weight_density_gain=weight_density_gain,
                    map_anchor_eps=map_anchor_eps,
                )
                cls1 = classify_residue_backbone(prev, res, nxt)
                if cls1 is None:
                    continue
                if cls0 != "outlier" and cls1 == "outlier":
                    continue
                if sc < best_score - 1e-7:
                    best_score = sc
                    best_dphi = dphi
                    best_dpsi = dpsi
        _restore_all_positions(st, snap0)
        if best_score < score0 - 1e-7 and (abs(best_dphi) > 1e-12 or abs(best_dpsi) > 1e-12):
            if best_dphi != 0.0 and phi_ok:
                rotate_phi(ch, ri, best_dphi)
            if best_dpsi != 0.0 and psi_ok:
                rotate_psi(ch, ri, best_dpsi)
            result.rama_improvements += 1
            cls_after = classify_residue_backbone(
                ch[ri - 1] if ri > 0 else None, ch[ri], ch[ri + 1] if ri + 1 < len(ch) else None
            )
            result.residue_log.append(
                {
                    "stage": "rama",
                    "chain": chain_id,
                    "seqid": str(res.seqid),
                    "resname": rn,
                    "score_before": round(score0, 6),
                    "score_after": round(best_score, 6),
                    "dphi_deg": round(best_dphi, 4),
                    "dpsi_deg": round(best_dpsi, 4),
                    "rama_before": cls0,
                    "rama_after": cls_after,
                }
            )
        else:
            _restore_all_positions(st, snap0)


def run_zonal_chi_refine(
    st: gemmi.Structure,
    map_vol: MapVolume,
    center_xyz: np.ndarray,
    radius: float,
    *,
    chain_filter: Optional[Set[str]] = None,
    passes: int = 3,
    weight_map: float = 0.65,
    map_density_threshold: float = 0.0,
    weight_rot: float = 0.15,
    soft_buffer: float = 0.0,
    soft_passes: int = 2,
    soft_min_clash: float = 1.0,
    soft_only_if_worsened: bool = True,
    rama_backbone: bool = False,
    rama_step_deg: float = 3.0,
    rama_max_shift_deg: float = 9.0,
    weight_rama: float = 0.08,
    weight_bb_move: float = 0.015,
    rama_include_soft: bool = False,
    rama_nudge_favored: bool = False,
    weight_density_anchor: float = 0.0,
    weight_density_gain: float = 0.0,
    map_anchor_eps: float = 1e-5,
) -> ZonalRefineResult:
    """
    Greedy χ1 rotamer trials for residues in a hard sphere; optional **soft shell** stage.

    **Stage 1 (hard):** same as A0 — multiple passes until no improvement.

    **Stage 2 (soft):** if ``soft_buffer > 0``, residues in the outer ball (radius
    ``radius + soft_buffer``) but not in the hard set may be adjusted only when
    ``total_clash`` exceeds ``soft_min_clash`` and, if ``soft_only_if_worsened``,
    exceeds the clash snapshot for that residue taken **before** any refinement.

    **Stage 3 (optional, A2):** if ``rama_backbone`` is True, small φ/ψ grid search
    on non-Gly/Pro residues in the zone (hard, optionally soft) to reduce Ramachandran
    outliers while penalizing |Δφ|²+|Δψ|² so backbone moves stay small.

    **Map density:** if ``map_density_threshold`` > 0, each trilinear sample uses
    ``max(0, raw - map_density_threshold)`` so low continuous background does not
    reward drifting into unoccupied regions.

    **Map-fit anchor:** if ``weight_density_anchor`` / ``weight_density_gain`` > 0,
    χ1 and rama trials compare mean thresholded map at side-chain / heavy atoms to
    the value **before** that trial block: losing fit is penalized when the anchor
    was above ``map_anchor_eps``; when the anchor was weak, only **gains** are
    rewarded (no extra penalty for staying weak).
    """
    t0 = time.perf_counter()
    result = ZonalRefineResult()
    result.meta = {
        "center": [float(x) for x in np.asarray(center_xyz).reshape(3)],
        "radius": float(radius),
        "chain_filter": sorted(chain_filter) if chain_filter else None,
        "passes_requested": int(passes),
        "weight_map": float(weight_map),
        "map_density_threshold": float(map_density_threshold),
        "weight_rot": float(weight_rot),
        "soft_buffer": float(soft_buffer),
        "soft_passes_max": int(soft_passes),
        "soft_min_clash": float(soft_min_clash),
        "soft_only_if_worsened": bool(soft_only_if_worsened),
        "rama_backbone": bool(rama_backbone),
        "rama_step_deg": float(rama_step_deg),
        "rama_max_shift_deg": float(rama_max_shift_deg),
        "weight_rama": float(weight_rama),
        "weight_bb_move": float(weight_bb_move),
        "rama_include_soft": bool(rama_include_soft),
        "rama_nudge_favored": bool(rama_nudge_favored),
        "weight_density_anchor": float(weight_density_anchor),
        "weight_density_gain": float(weight_density_gain),
        "map_anchor_eps": float(map_anchor_eps),
    }

    hard_zone, soft_zone = partition_hard_soft_spherical(
        st, center_xyz, radius, soft_buffer, chain_filter=chain_filter
    )
    result.residues_in_zone = len(hard_zone)
    result.residues_soft_zone = len(soft_zone)

    movable_hard = _build_movable(hard_zone)
    result.residues_with_chi1 = len(movable_hard)

    movable_soft = _build_movable(soft_zone)
    result.residues_soft_with_chi1 = len(movable_soft)

    baseline_clash: Dict[Tuple[str, str], float] = {}
    for chain_id, res, _, _ in movable_soft:
        k = _res_key(chain_id, res)
        baseline_clash[k] = float(total_clash_for_residue(st, chain_id, res))

    has_chi = bool(movable_hard or (soft_buffer > 0 and movable_soft))
    want_rama = bool(rama_backbone and rama_max_shift_deg > 0 and rama_step_deg > 0)
    if not has_chi and not want_rama:
        result.elapsed_sec = time.perf_counter() - t0
        return result

    # --- Stage 1: hard zone ---
    for p in range(passes):
        if not movable_hard:
            break
        result.improvements_in_last_pass = 0
        for chain_id, res, rn, quad in movable_hard:

            def clash_fn(r: gemmi.Residue) -> float:
                return float(total_clash_for_residue(st, chain_id, r))

            def map_fn(r: gemmi.Residue) -> float:
                return float(mean_sidechain_map_value(r, map_vol, density_threshold=map_density_threshold))

            score_before = _simple_objective(
                st, chain_id, res, map_vol, weight_map, map_density_threshold=map_density_threshold
            )

            pick_best_chi1(
                rn,
                res,
                quad,
                clash_fn,
                map_fn,
                weight_rot=weight_rot,
                weight_map=weight_map,
                weight_density_anchor=weight_density_anchor,
                weight_density_gain=weight_density_gain,
                map_anchor_eps=map_anchor_eps,
            )

            score_after = _simple_objective(
                st, chain_id, res, map_vol, weight_map, map_density_threshold=map_density_threshold
            )

            if score_after < score_before - 1e-8:
                result.improvements_in_last_pass += 1
                result.residue_log.append(
                    {
                        "stage": "hard",
                        "chain": chain_id,
                        "seqid": str(res.seqid),
                        "resname": rn,
                        "pass": p + 1,
                        "score_before": round(score_before, 6),
                        "score_after": round(score_after, 6),
                    }
                )

        result.passes_done = p + 1
        if not movable_hard or result.improvements_in_last_pass == 0:
            break

    # --- Stage 2: soft shell (clash-triggered) ---
    if soft_buffer > 0 and movable_soft:
        for sp in range(soft_passes):
            result.improvements_soft_last_pass = 0
            for chain_id, res, rn, quad in movable_soft:
                k = _res_key(chain_id, res)
                clash_now = float(total_clash_for_residue(st, chain_id, res))
                base = baseline_clash.get(k, 0.0)
                worsened = clash_now > base + 1e-6
                if clash_now < soft_min_clash:
                    continue
                if soft_only_if_worsened and not worsened:
                    continue

                def clash_fn(r: gemmi.Residue) -> float:
                    return float(total_clash_for_residue(st, chain_id, r))

                def map_fn(r: gemmi.Residue) -> float:
                    return float(mean_sidechain_map_value(r, map_vol, density_threshold=map_density_threshold))

                score_before = _simple_objective(
                    st, chain_id, res, map_vol, weight_map, map_density_threshold=map_density_threshold
                )

                pick_best_chi1(
                    rn,
                    res,
                    quad,
                    clash_fn,
                    map_fn,
                    weight_rot=weight_rot,
                    weight_map=weight_map,
                    weight_density_anchor=weight_density_anchor,
                    weight_density_gain=weight_density_gain,
                    map_anchor_eps=map_anchor_eps,
                )

                score_after = _simple_objective(
                    st, chain_id, res, map_vol, weight_map, map_density_threshold=map_density_threshold
                )

                if score_after < score_before - 1e-8:
                    result.improvements_soft_last_pass += 1
                    result.residue_log.append(
                        {
                            "stage": "soft",
                            "chain": chain_id,
                            "seqid": str(res.seqid),
                            "resname": rn,
                            "pass": sp + 1,
                            "score_before": round(score_before, 6),
                            "score_after": round(score_after, 6),
                            "clash_before": round(clash_now, 6),
                            "baseline_clash": round(base, 6),
                        }
                    )

            result.soft_passes_done = sp + 1
            if result.improvements_soft_last_pass == 0:
                break

    # --- Stage 3: Ramachandran micro-refine (optional) ---
    if want_rama:
        rama_zone = list(hard_zone)
        if rama_include_soft:
            rama_zone.extend(soft_zone)
        _rama_backbone_micro(
            st,
            map_vol,
            rama_zone,
            weight_map=weight_map,
            weight_rama=weight_rama,
            weight_bb_move=weight_bb_move,
            rama_step_deg=rama_step_deg,
            rama_max_shift_deg=rama_max_shift_deg,
            rama_nudge_favored=rama_nudge_favored,
            map_density_threshold=map_density_threshold,
            weight_density_anchor=weight_density_anchor,
            weight_density_gain=weight_density_gain,
            map_anchor_eps=map_anchor_eps,
            result=result,
        )

    result.elapsed_sec = time.perf_counter() - t0
    return result


def write_result_json(path: Path, result: ZonalRefineResult) -> None:
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "residues_in_zone": result.residues_in_zone,
        "residues_soft_zone": result.residues_soft_zone,
        "residues_with_chi1": result.residues_with_chi1,
        "residues_soft_with_chi1": result.residues_soft_with_chi1,
        "passes_done": result.passes_done,
        "improvements_last_pass": result.improvements_in_last_pass,
        "soft_passes_done": result.soft_passes_done,
        "improvements_soft_last_pass": result.improvements_soft_last_pass,
        "rama_residues_tried": result.rama_residues_tried,
        "rama_improvements": result.rama_improvements,
        "elapsed_sec": round(result.elapsed_sec, 4),
        "meta": result.meta,
        "residue_log": result.residue_log,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
