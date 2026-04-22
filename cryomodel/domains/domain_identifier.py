"""Domain identification from PDB coordinates."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
import shutil
import subprocess
import tempfile

import json
import numpy as np
import gemmi
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist


@dataclass
class ResidueRecord:
    chain_id: str
    resnum: int
    icode: str
    ca: np.ndarray


def _collect_residues(structure: gemmi.Structure, chain_id: Optional[str]) -> List[ResidueRecord]:
    model = structure[0]
    chains = [c for c in model if (chain_id is None or c.name == chain_id)]
    if not chains:
        raise ValueError(f"Chain '{chain_id}' not found.")
    records: List[ResidueRecord] = []
    for chain in chains:
        for residue in chain:
            if "CA" not in residue:
                continue
            atom = residue["CA"][0]
            records.append(
                ResidueRecord(
                    chain_id=chain.name,
                    resnum=residue.seqid.num,
                    icode=residue.seqid.icode or "",
                    ca=np.array([atom.pos.x, atom.pos.y, atom.pos.z], dtype=np.float32),
                )
            )
    if not records:
        raise ValueError("No CA atoms found in selected chain(s).")
    return records


def _build_seeds(records: List[ResidueRecord], seed_size: int) -> Tuple[np.ndarray, List[List[int]]]:
    seed_indices: List[List[int]] = []
    for start in range(0, len(records), seed_size):
        seed_indices.append(list(range(start, min(start + seed_size, len(records)))))
    centroids = []
    for indices in seed_indices:
        points = np.stack([records[i].ca for i in indices], axis=0)
        centroids.append(points.mean(axis=0))
    return np.stack(centroids, axis=0), seed_indices


def _cluster_seeds(
    centroids: np.ndarray,
    n_domains: Optional[int],
    merge_distance: Optional[float],
) -> np.ndarray:
    if len(centroids) == 1:
        return np.array([1], dtype=int)

    dists = pdist(centroids, metric="euclidean")
    linkage_matrix = linkage(dists, method="average")

    if n_domains is not None and n_domains > 0:
        labels = fcluster(linkage_matrix, t=n_domains, criterion="maxclust")
    else:
        if merge_distance is None:
            merge_distance = 25.0
        labels = fcluster(linkage_matrix, t=float(merge_distance), criterion="distance")
    return labels.astype(int)


def _merge_small_domains(
    domain_ids: np.ndarray,
    seed_indices: List[List[int]],
    records: List[ResidueRecord],
    min_domain_residues: int,
) -> np.ndarray:
    if min_domain_residues <= 0:
        return domain_ids

    def domain_centroid(domain: int) -> np.ndarray:
        indices = []
        for seed_idx, d in enumerate(domain_ids):
            if d == domain:
                indices.extend(seed_indices[seed_idx])
        if not indices:
            return None
        points = np.stack([records[i].ca for i in indices], axis=0)
        return points.mean(axis=0)

    while True:
        domain_counts: Dict[int, int] = {}
        for seed_idx, d in enumerate(domain_ids):
            domain_counts.setdefault(d, 0)
            domain_counts[d] += len(seed_indices[seed_idx])
        small_domains = [d for d, count in domain_counts.items() if count < min_domain_residues]
        if not small_domains or len(domain_counts) <= 1:
            break
        for small in small_domains:
            small_centroid = domain_centroid(small)
            if small_centroid is None:
                continue
            other_domains = [d for d in domain_counts if d != small]
            if not other_domains:
                continue
            distances = []
            for d in other_domains:
                centroid = domain_centroid(d)
                if centroid is None:
                    continue
                distances.append((d, np.linalg.norm(centroid - small_centroid)))
            if not distances:
                continue
            target = min(distances, key=lambda item: item[1])[0]
            domain_ids = np.array([target if d == small else d for d in domain_ids], dtype=int)
    return domain_ids


def _gap_positions(records: List[ResidueRecord]) -> List[int]:
    gaps = []
    for idx in range(len(records) - 1):
        current = records[idx]
        nxt = records[idx + 1]
        if current.chain_id != nxt.chain_id:
            gaps.append(idx)
            continue
        if nxt.resnum > current.resnum + 1:
            gaps.append(idx)
    return gaps


def parse_sse_from_pdb_header(pdb_path: Path, chain_id: str) -> Set[int]:
    sse_resnums: Set[int] = set()
    with pdb_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                break
            if line.startswith("HELIX "):
                if len(line) < 38:
                    continue
                init_chain = line[19].strip()
                end_chain = line[31].strip()
                if init_chain != chain_id or end_chain != chain_id:
                    continue
                try:
                    start = int(line[21:25].strip())
                    end = int(line[33:37].strip())
                except ValueError:
                    continue
                for resnum in range(start, end + 1):
                    sse_resnums.add(resnum)
            elif line.startswith("SHEET "):
                if len(line) < 38:
                    continue
                init_chain = line[21].strip()
                end_chain = line[32].strip()
                if init_chain != chain_id or end_chain != chain_id:
                    continue
                try:
                    start = int(line[22:26].strip())
                    end = int(line[33:37].strip())
                except ValueError:
                    continue
                for resnum in range(start, end + 1):
                    sse_resnums.add(resnum)
    return sse_resnums


def parse_sse_with_dssp(pdb_path: Path, chain_id: str) -> Set[int]:
    dssp = shutil.which("mkdssp") or shutil.which("dssp")
    if not dssp:
        return set()
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "out.dssp"
        result = subprocess.run(
            [dssp, "-i", str(pdb_path), "-o", str(out_path)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0 or not out_path.exists():
            return set()
        sse_resnums: Set[int] = set()
        started = False
        for line in out_path.read_text(errors="ignore").splitlines():
            if not started:
                if line.strip().startswith("#"):
                    started = True
                continue
            if len(line) < 18:
                continue
            try:
                resnum = int(line[5:10].strip())
            except ValueError:
                continue
            chain = line[11].strip()
            if chain != chain_id:
                continue
            ss = line[16].strip()
            if ss in {"H", "G", "I", "E", "B"}:
                sse_resnums.add(resnum)
        return sse_resnums


def _prefer_non_sse_boundaries(
    records: List[ResidueRecord],
    residue_domain: List[int],
    sse_resnums: Set[int],
    sse_window: int,
) -> List[int]:
    if not sse_resnums or sse_window <= 0:
        return residue_domain

    for idx in range(len(residue_domain) - 1):
        if residue_domain[idx] == residue_domain[idx + 1]:
            continue
        left_res = records[idx].resnum
        right_res = records[idx + 1].resnum
        if left_res not in sse_resnums and right_res not in sse_resnums:
            continue

        # Search for nearest boundary between non-SSE residues.
        best_idx = None
        best_dist = None
        for offset in range(1, sse_window + 1):
            left = idx - offset
            right = idx + offset
            if left >= 0:
                lr = records[left].resnum
                rr = records[left + 1].resnum
                if lr not in sse_resnums and rr not in sse_resnums:
                    best_idx = left
                    best_dist = offset
                    break
            if right < len(residue_domain) - 1:
                lr = records[right].resnum
                rr = records[right + 1].resnum
                if lr not in sse_resnums and rr not in sse_resnums:
                    best_idx = right
                    best_dist = offset
                    break
        if best_idx is None or best_dist is None:
            continue
        if best_idx > idx:
            for j in range(idx + 1, best_idx + 1):
                residue_domain[j] = residue_domain[idx]
        elif best_idx < idx:
            for j in range(best_idx + 1, idx + 1):
                residue_domain[j] = residue_domain[idx + 1]
    return residue_domain


def _prefer_gap_boundaries(
    records: List[ResidueRecord],
    residue_domain: List[int],
    gap_window: int,
    gaps_only: bool,
) -> List[int]:
    if gaps_only:
        gap_window = 0
    if gap_window <= 0:
        gaps = _gap_positions(records)
        if not gaps:
            return residue_domain
        gaps_set = set(gaps)
        for idx in range(len(residue_domain) - 1):
            if residue_domain[idx] == residue_domain[idx + 1]:
                continue
            if idx in gaps_set:
                continue
            residue_domain[idx + 1] = residue_domain[idx]
        return residue_domain
    gaps = _gap_positions(records)
    if not gaps:
        return residue_domain
    gaps_set = set(gaps)
    gap_list = gaps

    for idx in range(len(residue_domain) - 1):
        if residue_domain[idx] == residue_domain[idx + 1]:
            continue
        if idx in gaps_set:
            continue
        # find nearest gap within window
        nearest = None
        nearest_dist = None
        for g in gap_list:
            dist = abs(g - idx)
            if dist > gap_window:
                continue
            if nearest is None or dist < nearest_dist:
                nearest = g
                nearest_dist = dist
        if nearest is None or nearest == idx:
            continue
        if nearest > idx:
            # move boundary right to nearest gap
            for j in range(idx + 1, nearest + 1):
                residue_domain[j] = residue_domain[idx]
        else:
            # move boundary left to nearest gap
            for j in range(nearest + 1, idx + 1):
                residue_domain[j] = residue_domain[idx + 1]
    return residue_domain


def identify_domains(
    structure: gemmi.Structure,
    chain_id: Optional[str] = None,
    seed_size: int = 20,
    n_domains: Optional[int] = None,
    merge_distance: Optional[float] = 25.0,
    min_domain_residues: int = 50,
    prefer_gaps: bool = True,
    gap_window: int = 10,
    gaps_only: bool = False,
    sse_resnums: Optional[Set[int]] = None,
    sse_window: int = 10,
) -> Tuple[List[ResidueRecord], Dict[str, List[Tuple[int, int]]]]:
    records = _collect_residues(structure, chain_id)
    centroids, seed_indices = _build_seeds(records, seed_size)
    domain_ids = _cluster_seeds(centroids, n_domains, merge_distance)
    domain_ids = _merge_small_domains(domain_ids, seed_indices, records, min_domain_residues)

    # Map residues to domain IDs
    residue_domain_list: List[int] = [0] * len(records)
    for seed_idx, indices in enumerate(seed_indices):
        for idx in indices:
            residue_domain_list[idx] = int(domain_ids[seed_idx])

    if prefer_gaps or gaps_only:
        residue_domain_list = _prefer_gap_boundaries(records, residue_domain_list, gap_window, gaps_only)
    if sse_resnums:
        residue_domain_list = _prefer_non_sse_boundaries(records, residue_domain_list, sse_resnums, sse_window)

    residue_domain: Dict[int, int] = {idx: residue_domain_list[idx] for idx in range(len(records))}

    # Build contiguous ranges per domain
    domain_ranges: Dict[str, List[Tuple[int, int]]] = {}
    for idx, record in enumerate(records):
        domain_label = f"D{residue_domain[idx]}"
        domain_ranges.setdefault(domain_label, [])
        domain_ranges[domain_label].append(record.resnum)

    ranges_by_domain: Dict[str, List[Tuple[int, int]]] = {}
    for domain_label, resnums in domain_ranges.items():
        resnums_sorted = sorted(set(resnums))
        ranges: List[Tuple[int, int]] = []
        start = prev = resnums_sorted[0]
        for r in resnums_sorted[1:]:
            if r == prev + 1:
                prev = r
                continue
            ranges.append((start, prev))
            start = prev = r
        ranges.append((start, prev))
        ranges_by_domain[domain_label] = ranges

    return records, ranges_by_domain


def write_domain_spec(
    out_path: Path,
    chain_id: str,
    ranges_by_domain: Dict[str, List[Tuple[int, int]]],
) -> None:
    payload: Dict[str, Dict[str, List[str]]] = {}
    for domain_label, ranges in ranges_by_domain.items():
        payload[domain_label] = {
            chain_id: [f"{start}-{end}" if start != end else f"{start}" for start, end in ranges]
        }
    out_path.write_text(json.dumps(payload, indent=2) + "\n")


def write_domain_csv(
    out_path: Path,
    records: List[ResidueRecord],
    ranges_by_domain: Dict[str, List[Tuple[int, int]]],
) -> None:
    # Map residue number to domain
    domain_by_res: Dict[int, str] = {}
    for domain_label, ranges in ranges_by_domain.items():
        for start, end in ranges:
            for resnum in range(start, end + 1):
                domain_by_res[resnum] = domain_label
    lines = ["chain,resnum,domain\n"]
    for record in records:
        domain = domain_by_res.get(record.resnum, "NA")
        lines.append(f"{record.chain_id},{record.resnum},{domain}\n")
    out_path.write_text("".join(lines))


def write_domain_pdb(
    structure: gemmi.Structure,
    chain_id: str,
    ranges_by_domain: Dict[str, List[Tuple[int, int]]],
    out_path: Path,
) -> None:
    domain_by_res: Dict[int, int] = {}
    for domain_label, ranges in ranges_by_domain.items():
        domain_num = int(domain_label.replace("D", "")) if domain_label.startswith("D") else 0
        for start, end in ranges:
            for resnum in range(start, end + 1):
                domain_by_res[resnum] = domain_num

    st = structure.clone()
    for model in st:
        for chain in model:
            if chain.name != chain_id:
                continue
            for residue in chain:
                resnum = residue.seqid.num
                domain_num = domain_by_res.get(resnum, 0)
                for atom in residue:
                    atom.b_iso = float(domain_num)
    st.write_pdb(str(out_path))
