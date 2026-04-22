"""Tier-3: 3D graph diffusion of conservation variability across selected chains.

Treats Cα atoms within ``contact_radius`` Å as a graph with soft distance falloff, diffuses
a seed signal (high-variability sites) to highlight spatial clusters of variation in the
oligomer. Optional ``nearest_peak`` basins approximate a lightweight watershed-style grouping.
"""
from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import gemmi
import numpy as np

from .conservation import build_conservation_rows

# Primitive columns from build_conservation_rows; composites are derived (no extra MSA pass).
SEED_PRIMITIVES = frozenset(
    {"p_nonref", "n_aa_types", "entropy", "frac_nonconservative", "p_gap", "mean_penalty"}
)
# Composites: emphasize “how much change” × “how harsh” (mean_penalty / frac_nonconservative).
SEED_COMPOSITES = frozenset(
    {
        "composite_nonref_penalty",  # p_nonref * mean_penalty — harsh substitutions weighted by frequency
        "composite_entropy_noncons",  # entropy * frac_nonconservative — diversity × chemically harsh fraction
        "composite_diversity_penalty",  # ((n_aa_types-1)/19) * mean_penalty — type count × harshness
    }
)
SEED_METRICS = SEED_PRIMITIVES | SEED_COMPOSITES


def _seed_raw_array(rows: Sequence[Dict[str, object]], seed_metric: str) -> np.ndarray:
    n = len(rows)
    out = np.zeros(n, dtype=np.float64)
    if seed_metric in SEED_PRIMITIVES:
        for i, row in enumerate(rows):
            out[i] = float(row[seed_metric])
        return out
    if seed_metric == "composite_nonref_penalty":
        for i, row in enumerate(rows):
            out[i] = float(row["p_nonref"]) * float(row["mean_penalty"])
        return out
    if seed_metric == "composite_entropy_noncons":
        for i, row in enumerate(rows):
            out[i] = float(row["entropy"]) * float(row["frac_nonconservative"])
        return out
    if seed_metric == "composite_diversity_penalty":
        for i, row in enumerate(rows):
            nt = float(row["n_aa_types"])
            out[i] = max(0.0, nt - 1.0) / 19.0 * float(row["mean_penalty"])
        return out
    raise ValueError(f"Unhandled seed_metric {seed_metric!r}")


def _ca_xyz(st: gemmi.Structure, chain_id: str, seqid: int, icode: str) -> np.ndarray:
    ch = st[0][chain_id]
    ic = icode.strip() if icode else ""
    for res in ch:
        sid = int(res.seqid.num)
        ric = res.seqid.icode.strip() if res.seqid.icode else ""
        if sid == seqid and ric == ic:
            for atom in res:
                if atom.name == "CA":
                    p = atom.pos
                    return np.array([p.x, p.y, p.z], dtype=np.float64)
            raise ValueError(f"No CA for {chain_id} {seqid}{ic or ''} ({res.name})")
    raise KeyError(f"Residue {chain_id} {seqid}{ic or ''} not found")


def _build_ca_graph(
    rows: Sequence[Dict[str, object]],
    st: gemmi.Structure,
    contact_radius: float,
    falloff_angstrom: float,
) -> Tuple[np.ndarray, List[List[int]], List[List[float]]]:
    """Positions (N,3), neighbor indices, neighbor weights (exponential falloff)."""
    n = len(rows)
    pos = np.zeros((n, 3), dtype=np.float64)
    for i, row in enumerate(rows):
        pos[i] = _ca_xyz(st, str(row["chain"]), int(row["seqid"]), str(row["icode"]))

    d0 = max(falloff_angstrom, 1e-3)
    neigh: List[List[int]] = [[] for _ in range(n)]
    wght: List[List[float]] = [[] for _ in range(n)]
    r2_max = contact_radius * contact_radius
    for i in range(n):
        for j in range(i + 1, n):
            d2 = float(np.sum((pos[i] - pos[j]) ** 2))
            if d2 > r2_max or d2 < 1e-12:
                continue
            d = math.sqrt(d2)
            w = math.exp(-d / d0)
            neigh[i].append(j)
            wght[i].append(w)
            neigh[j].append(i)
            wght[j].append(w)
    return pos, neigh, wght


def _diffuse(
    u0: np.ndarray,
    neigh: List[List[int]],
    wght: List[List[float]],
    steps: int,
    mix: float,
) -> np.ndarray:
    """
    Each step: ``u <- (1-mix)*u + mix*weighted_mean(neighbors)``.

    Isolated nodes (no edges) keep their value.
    """
    u = u0.astype(np.float64).copy()
    lam = float(np.clip(mix, 0.0, 1.0))
    for _ in range(int(steps)):
        u_next = np.zeros_like(u)
        for i in range(len(u)):
            js, ws = neigh[i], wght[i]
            if not js:
                u_next[i] = u[i]
                continue
            sw = sum(ws)
            if sw <= 0:
                u_next[i] = u[i]
                continue
            nbr_mean = sum(w * u[j] for j, w in zip(js, ws)) / sw
            u_next[i] = (1.0 - lam) * u[i] + lam * nbr_mean
        u = u_next
    return u


def _local_maxima(u: np.ndarray, neigh: List[List[int]], min_height: float) -> List[int]:
    peaks: List[int] = []
    for i in range(len(u)):
        if u[i] < min_height:
            continue
        ok = True
        for j in neigh[i]:
            if u[j] > u[i] + 1e-9:
                ok = False
                break
        if ok:
            peaks.append(i)
    return peaks


def _assign_basins_nearest_peak(
    pos: np.ndarray,
    u: np.ndarray,
    peaks: List[int],
    peak_weight_gamma: float,
) -> np.ndarray:
    """basin_id[i] = index into ``peaks`` (0..K-1), or -1 if no peaks."""
    n = len(u)
    out = np.full(n, -1, dtype=np.int32)
    if not peaks:
        return out
    gam = float(max(peak_weight_gamma, 1e-6))
    for i in range(n):
        best_k = 0
        best = float("inf")
        for k, p in enumerate(peaks):
            d = float(np.linalg.norm(pos[i] - pos[p]))
            score = d / (float(u[p]) ** gam + 1e-6)
            if score < best:
                best = score
                best_k = k
        out[i] = best_k
    return out


@dataclass
class ConservationDiffuseResult:
    rows: List[Dict[str, object]]
    out_csv: Path
    out_json: Optional[Path]
    out_pdb: Optional[Path]


def run_conservation_diffusion(
    pdb_path: Path,
    chains: str,
    alignment_fasta: Path,
    *,
    out_csv: Path,
    out_json: Optional[Path] = None,
    out_pdb: Optional[Path] = None,
    include_reference_in_stats: bool = False,
    seed_metric: str = "p_nonref",
    seed_threshold: float = 0.0,
    contact_radius: float = 10.0,
    falloff_angstrom: float = 3.0,
    diffusion_steps: int = 24,
    mix: float = 0.4,
    peak_min: float = 0.02,
    basin_mode: str = "nearest_peak",
    peak_weight_gamma: float = 0.5,
    bfactor_writes: str = "diffused_score",
) -> ConservationDiffuseResult:
    """
    Build conservation table, diffuse ``seed_metric`` on a Cα graph over **all** listed chains.

    ``bfactor_writes`` selects which scalar column is written to output PDB B-factors
    (``diffused_score`` or ``seed_signal``).

    **Composite seeds** combine primitives so conservative columns (low ``mean_penalty``) contribute
    less than columns with frequent or chemically strong mismatches. Tune ``--seed-threshold`` —
    composites are often smaller in magnitude than ``p_nonref`` alone.
    """
    if seed_metric not in SEED_METRICS:
        raise ValueError(
            f"Unknown seed_metric {seed_metric!r}; choose from {sorted(SEED_METRICS)}"
        )
    if basin_mode not in ("none", "nearest_peak"):
        raise ValueError("basin_mode must be 'none' or 'nearest_peak'")
    if bfactor_writes not in ("diffused_score", "seed_signal"):
        raise ValueError("bfactor_writes must be 'diffused_score' or 'seed_signal'")

    rows, chain_list, ref_id, st = build_conservation_rows(
        pdb_path,
        chains,
        alignment_fasta,
        include_reference_in_stats=include_reference_in_stats,
    )
    if not rows:
        raise ValueError("No residues mapped; cannot run diffusion.")

    pos, neigh, wght = _build_ca_graph(rows, st, contact_radius, falloff_angstrom)
    n = len(rows)
    raw = _seed_raw_array(rows, seed_metric)
    u0 = np.maximum(0.0, raw - float(seed_threshold))
    u_diff = _diffuse(u0, neigh, wght, diffusion_steps, mix)

    peaks = _local_maxima(u_diff, neigh, peak_min) if basin_mode == "nearest_peak" else []
    if basin_mode == "nearest_peak" and not peaks:
        peaks = [int(np.argmax(u_diff))]

    basins = (
        _assign_basins_nearest_peak(pos, u_diff, peaks, peak_weight_gamma)
        if basin_mode == "nearest_peak"
        else np.full(n, -1, dtype=np.int32)
    )

    out_rows: List[Dict[str, object]] = []
    for i, row in enumerate(rows):
        od = dict(row)
        od["seed_raw"] = round(float(raw[i]), 6)
        od["seed_signal"] = round(float(u0[i]), 6)
        od["diffused_score"] = round(float(u_diff[i]), 6)
        od["basin_id"] = int(basins[i])
        od["is_diffusion_peak"] = int(i in peaks) if peaks else 0
        out_rows.append(od)

    out_csv = out_csv.expanduser()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(out_rows[0].keys())
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in out_rows:
            w.writerow(r)

    out_json_path: Optional[Path] = None
    if out_json is not None:
        out_json_path = out_json.expanduser()
        out_json_path.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "tier": 3,
            "method": "ca_graph_diffusion",
            "pdb": str(pdb_path),
            "chains": chain_list,
            "alignment_fasta": str(alignment_fasta),
            "reference_sequence_id": ref_id,
            "seed_metric": seed_metric,
            "seed_metric_composite": seed_metric in SEED_COMPOSITES,
            "seed_threshold": seed_threshold,
            "contact_radius": contact_radius,
            "falloff_angstrom": falloff_angstrom,
            "diffusion_steps": diffusion_steps,
            "mix": mix,
            "peak_min": peak_min,
            "basin_mode": basin_mode,
            "peak_weight_gamma": peak_weight_gamma,
            "n_graph_nodes": n,
            "n_peaks": len(peaks),
            "peak_node_indices": peaks,
        }
        out_json_path.write_text(
            json.dumps({"meta": meta, "rows": out_rows}, indent=2),
            encoding="utf-8",
        )

    out_pdb_path: Optional[Path] = None
    if out_pdb is not None:
        chain_set = set(chain_list)
        metric_key = bfactor_writes
        val_by_key = {
            (str(r["chain"]), int(r["seqid"]), str(r["icode"])): float(r[metric_key])
            for r in out_rows
        }
        for model in st:
            for chn in model:
                if chn.name not in chain_set:
                    continue
                for res in chn:
                    key3 = (
                        chn.name,
                        int(res.seqid.num),
                        res.seqid.icode.strip() if res.seqid.icode else "",
                    )
                    if key3 not in val_by_key:
                        continue
                    v = val_by_key[key3]
                    for atom in res:
                        atom.b_iso = float(v)
        out_pdb_path = out_pdb.expanduser()
        out_pdb_path.parent.mkdir(parents=True, exist_ok=True)
        st.write_pdb(str(out_pdb_path))

    return ConservationDiffuseResult(rows=out_rows, out_csv=out_csv, out_json=out_json_path, out_pdb=out_pdb_path)
