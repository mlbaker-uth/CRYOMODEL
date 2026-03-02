from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import json
import math
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN

from ..io.mrc import MapVolume, read_map, write_map


@dataclass
class Pathwalker2Config:
    map_path: Path
    threshold: float
    n_residues: int
    out_dir: Path
    seed_method: str = "ridge"
    grid_step: float = 1.0
    map_prep: str = "locscale"  # locscale | gaussian | none
    gaussian_sigma: float = 0.6
    k_neighbors: int = 12
    max_edge_length: float = 5.5
    fragments_per_partition: int = 10
    random_state: Optional[int] = 0
    weights: Dict[str, float] = None  # updated in __post_init__

    def __post_init__(self) -> None:
        if self.weights is None:
            self.weights = {
                "w_d": 1.0,
                "sigma_d": 0.20,
                "w_m": 2.0,
                "w_g": 1.0,
                "w_t": 0.5,
                "w_c": 0.25,
                "w_ang": 2.0,
                "w_backtrack": 0.5,
                "w_cross": 3.0,
                "w_clash": 2.0,
                "coverage_boost": 5.0,
                "ss_consistency": 2.0,
                "pen_cross": 4.0,
                "pen_clash": 2.0,
            }
        self.out_dir = Path(self.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        if isinstance(self.map_path, str):
            self.map_path = Path(self.map_path)


@dataclass
class CandidateNode:
    idx: int
    voxel_zyx: np.ndarray
    position: np.ndarray  # xyz in Å
    density: float
    z_score: float
    ridgeness: float
    gradient: np.ndarray
    thickness: float
    component: int = -1
    partition: int = -1
    is_rail: bool = False
    rail_id: Optional[int] = None
    ss_tag: Optional[str] = None  # helix | sheet | loop


@dataclass
class Fragment:
    node_indices: List[int]
    cost: float
    coverage: float
    ss_consistency: float
    crossovers: int
    clashes: int


def run_pathwalker2(config: Pathwalker2Config) -> Dict[str, Path]:
    """Run Pathwalker2 automatic trace discovery (Step 1)."""
    rng = np.random.default_rng(config.random_state)

    map_vol = read_map(config.map_path)
    prepped_map, prep_report = _prepare_map(map_vol, config)

    # Create mask from prepped map (already thresholded, so just check for non-zero)
    # For locscale, use > 0; for others, use >= threshold
    if config.map_prep == "locscale":
        mask = prepped_map > 0.0
    else:
        mask = prepped_map >= config.threshold
    if mask.sum() == 0:
        raise RuntimeError("Threshold produced empty mask; try lowering --threshold.")
    
    # Debug info
    print(f"Map shape: {prepped_map.shape}")
    print(f"Mask voxels: {mask.sum()} / {mask.size} ({100.0 * mask.sum() / mask.size:.2f}%)")
    print(f"Prepped map range: [{prepped_map[mask].min():.4f}, {prepped_map[mask].max():.4f}]")

    features = _compute_voxel_features(prepped_map, mask)

    candidates, target = _generate_candidates(
        map_vol,
        prepped_map,
        mask,
        features,
        config,
        rng,
    )

    if not candidates:
        raise RuntimeError("Pseudoatom seeding produced zero candidates.")
    
    print(f"Generated {len(candidates)} candidate pseudoatoms (target was {target})")
    if candidates:
        positions = np.array([c.position for c in candidates])
        print(f"Candidate positions range: X[{positions[:, 0].min():.1f}, {positions[:, 0].max():.1f}], "
              f"Y[{positions[:, 1].min():.1f}, {positions[:, 1].max():.1f}], "
              f"Z[{positions[:, 2].min():.1f}, {positions[:, 2].max():.1f}]")

    partitions = _assign_partitions(candidates, mask, features["components"])

    graph = _build_graph(candidates, config)

    fragments = _route_and_score_partitions(
        candidates, graph, partitions, config, rng
    )

    outputs = _write_outputs(
        config,
        map_vol,
        prepped_map,
        candidates,
        fragments,
        prep_report,
    )
    return outputs


# ---------------------------------------------------------------------------
# Map preparation


def _prepare_map(
    map_vol: MapVolume,
    config: Pathwalker2Config,
) -> Tuple[np.ndarray, Dict[str, float]]:
    data = map_vol.data_zyx.astype(np.float32, copy=True)
    report: Dict[str, float] = {}

    # Step 1: Threshold FIRST (before normalization) to exclude noise
    # This ensures normalization statistics are only computed from signal
    initial_mask = data >= config.threshold
    data[~initial_mask] = 0.0
    report["voxels_above_threshold"] = float(initial_mask.sum())
    report["voxels_total"] = float(data.size)

    # Step 2: Apply normalization (only affects non-zero regions)
    if config.map_prep == "gaussian":
        sigma = max(config.gaussian_sigma, 0.1)
        data = ndimage.gaussian_filter(data, sigma=sigma)
        report["gaussian_sigma"] = float(sigma)
    elif config.map_prep == "locscale":
        # local z-normalization using a 6 Å window (approx voxels)
        # Note: zeros from thresholding will affect local stats, but that's acceptable
        window = max(int(round(6.0 / map_vol.apix)), 1)
        footprint = np.ones((window, window, window), dtype=np.float32)
        local_mean = ndimage.uniform_filter(data, size=window, mode="nearest")
        local_sq_mean = ndimage.uniform_filter(data * data, size=window, mode="nearest")
        local_var = np.maximum(local_sq_mean - local_mean**2, 1e-6)
        data = (data - local_mean) / np.sqrt(local_var)
        report["locscale_window_vox"] = window
    # else: none

    # Step 3: Re-threshold to ensure nothing below threshold remains
    # For normalized data (locscale), use threshold >= 0 to keep positive z-scores
    # For non-normalized data, use original threshold
    if config.map_prep == "locscale":
        # After locscale, data is z-normalized, so threshold at 0 (or small positive)
        final_mask = data > 0.0
    else:
        # For gaussian or none, use original threshold
        final_mask = data >= config.threshold
    data[~final_mask] = 0.0
    report["voxels_after_normalization"] = float(final_mask.sum())

    return data, report


# ---------------------------------------------------------------------------
# Feature computation and seeding


def _compute_voxel_features(
    data_zyx: np.ndarray,
    mask: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Return dictionary of voxel-wise features needed downstream."""
    # Ridgeness: negative Laplacian-of-Gaussian (higher is better ridge)
    log_resp = -ndimage.gaussian_laplace(data_zyx, sigma=1.2)

    # Gradient direction
    grad_z, grad_y, grad_x = np.gradient(data_zyx)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2) + 1e-6
    grad_dir = np.stack([grad_x / grad_mag, grad_y / grad_mag, grad_z / grad_mag], axis=-1)

    # Thickness via distance transform on mask
    thickness = ndimage.distance_transform_edt(mask).astype(np.float32)

    # Connected components (for partitions)
    components, _ = ndimage.label(mask)

    # Z-score at each voxel (already zero-mean if locscale)
    mean = np.mean(data_zyx[mask])
    std = np.std(data_zyx[mask]) + 1e-6
    z_map = (data_zyx - mean) / std

    return {
        "ridgeness": log_resp,
        "gradient": grad_dir,
        "thickness": thickness,
        "components": components,
        "z_map": z_map,
    }


def _voxel_to_xyz(map_vol: MapVolume, voxel_zyx: np.ndarray) -> np.ndarray:
    x = voxel_zyx[2]
    y = voxel_zyx[1]
    z = voxel_zyx[0]
    return map_vol.origin_xyzA + map_vol.apix * np.array([x, y, z], dtype=np.float32)


def _generate_candidates(
    map_vol: MapVolume,
    data_zyx: np.ndarray,
    mask: np.ndarray,
    features: Dict[str, np.ndarray],
    config: Pathwalker2Config,
    rng: np.random.Generator,
) -> Tuple[List[CandidateNode], int]:
    """Select pseudoatom candidates."""
    ridgeness = features["ridgeness"]
    thickness = features["thickness"]
    z_map = features["z_map"]
    grad_dir = features["gradient"]

    voxel_indices = np.argwhere(mask)
    if len(voxel_indices) == 0:
        return [], 0

    ridges = ridgeness[mask]
    z_values = z_map[mask]
    thickness_vals = thickness[mask]
    gradients = grad_dir[mask]

    # Score each voxel: balance ridgeness, density, and penalize thickness
    # Thickness penalty helps avoid side chains (which are thicker than backbone)
    normalized_ridges = (ridges - np.min(ridges)) / (np.max(ridges) - np.min(ridges) + 1e-6)
    normalized_z = np.clip((z_values - np.min(z_values)) / (np.max(z_values) - np.min(z_values) + 1e-6), 0, 1)
    # Normalize thickness (inverse: thinner is better for backbone)
    # Use a stronger penalty for thickness to better favor backbone
    thickness_normalized = (thickness_vals - np.min(thickness_vals)) / (np.max(thickness_vals) - np.min(thickness_vals) + 1e-6)
    # Strongly penalize thick regions (side chains) - use exponential to make it more aggressive
    normalized_thickness = np.exp(-3.0 * thickness_normalized)  # Exponential penalty for thickness
    
    # Combine: favor ridgeness and density, strongly penalize thickness
    # Increased weight on thickness penalty to better avoid side chains
    scores = 0.35 * normalized_ridges + 0.35 * normalized_z + 0.30 * normalized_thickness

    # Target number of seeds based on n_residues
    n_res = config.n_residues
    n_vox = voxel_indices.shape[0]
    # Target should be close to n_residues (allow slight over-sampling for better coverage)
    # We want ~171 Ca atoms, so target should be at least n_residues
    target = int(min(1.1 * n_res, n_vox))  # Allow 10% over-sampling
    target = max(target, n_res)  # At least n_residues
    print(f"Target candidates: {target} (n_residues={n_res}, n_vox={n_vox})")

    # Select candidates with spatial diversity using farthest-point sampling
    # First, get top candidates by score - use larger pool for better selection
    n_candidates_pool = min(target * 5, len(scores))  # Larger pool (5x instead of 3x)
    top_scores_idx = np.argpartition(-scores, min(n_candidates_pool - 1, len(scores) - 1))[:n_candidates_pool]
    candidate_pool_voxels = voxel_indices[top_scores_idx]
    candidate_pool_scores = scores[top_scores_idx]
    candidate_pool_positions = np.array([_voxel_to_xyz(map_vol, v) for v in candidate_pool_voxels])
    
    # Farthest-point sampling for spatial diversity
    selected_voxels = []
    selected_indices = []
    if len(candidate_pool_voxels) > 0:
        # Start with highest-scoring candidate
        first_idx = np.argmax(candidate_pool_scores)
        selected_voxels.append(candidate_pool_voxels[first_idx])
        selected_indices.append(top_scores_idx[first_idx])
        remaining_indices = set(range(len(candidate_pool_voxels)))
        remaining_indices.remove(first_idx)
        
        # Iteratively add farthest point from already selected
        # Use a smaller minimum distance to allow more candidates
        # Ca-Ca distances are ~3.8 Å, so we can allow candidates closer than that
        # This allows over-sampling which is needed for good coverage
        min_distance = 1.5  # Minimum spacing between candidates (Å) - allows more candidates
        while len(selected_voxels) < target and remaining_indices:
            selected_positions = np.array([_voxel_to_xyz(map_vol, v) for v in selected_voxels])
            remaining_positions = candidate_pool_positions[list(remaining_indices)]
            
            # For each remaining candidate, find min distance to selected
            distances = cdist(remaining_positions, selected_positions)
            min_dists = np.min(distances, axis=1)
            
            # Filter candidates that are too close
            far_enough = min_dists >= min_distance
            if not np.any(far_enough):
                # If all remaining are too close, relax constraint slightly
                far_enough = min_dists >= (min_distance * 0.7)
            
            if not np.any(far_enough):
                # If we can't add more with current distance, try even smaller distance
                if min_distance > 1.0:
                    min_distance = 1.0
                    continue
                break  # Can't add more without getting too close
            
            # Among far-enough candidates, pick highest score
            # far_enough is a boolean array over remaining_positions, which corresponds to remaining_indices
            remaining_indices_list = list(remaining_indices)
            far_indices_list = [remaining_indices_list[j] for j in range(len(remaining_indices_list)) if far_enough[j]]
            
            if not far_indices_list:
                break
            
            # Get scores for far-enough candidates (using indices into candidate_pool_scores)
            far_scores = candidate_pool_scores[far_indices_list]
            best_far_idx = far_indices_list[np.argmax(far_scores)]
            
            selected_voxels.append(candidate_pool_voxels[best_far_idx])
            selected_indices.append(top_scores_idx[best_far_idx])
            remaining_indices.remove(best_far_idx)
            
            # Progress update every 20 candidates
            if len(selected_voxels) % 20 == 0:
                print(f"  Selected {len(selected_voxels)}/{target} candidates...")
    
    selected_voxels = np.array(selected_voxels)
    print(f"Final candidate count: {len(selected_voxels)} (target was {target})")

    candidates: List[CandidateNode] = []
    # selected_indices already contains the correct indices into the masked arrays
    # (they were stored as top_scores_idx[best_far_idx] during farthest-point sampling)
    for i, vox in enumerate(selected_voxels):
        idx = i
        position = _voxel_to_xyz(map_vol, vox)
        vox_tuple = tuple(vox)
        
        if i < len(selected_indices):
            # selected_indices[i] is already the index into z_values, ridges, etc.
            mask_idx = selected_indices[i]
            
            # Safety check: ensure mask_idx is within bounds
            if mask_idx < len(z_values) and mask_idx < len(ridges) and mask_idx < len(thickness_vals) and mask_idx < len(gradients):
                candidates.append(
                    CandidateNode(
                        idx=idx,
                        voxel_zyx=vox.astype(np.int32),
                        position=position,
                        density=float(data_zyx[vox_tuple]),
                        z_score=float(z_values[mask_idx]),
                        ridgeness=float(ridges[mask_idx]),
                        gradient=gradients[mask_idx],
                        thickness=float(thickness_vals[mask_idx]),
                    )
                )
    return candidates, target


# ---------------------------------------------------------------------------
# Partitioning & Graph building


def _assign_partitions(
    candidates: List[CandidateNode],
    mask: np.ndarray,
    component_labels: np.ndarray,
) -> Dict[int, List[int]]:
    """Group candidates into partitions using connected components + DBSCAN refinement."""
    # Initial component assignment from binary mask
    partitions: Dict[int, List[int]] = {}
    for cand in candidates:
        label = int(component_labels[tuple(cand.voxel_zyx)])
        cand.component = label
        partitions.setdefault(label, []).append(cand.idx)

    # For large components, further partition using spatial clustering
    for label, idxs in list(partitions.items()):
        if label == 0 or len(idxs) < 50:
            for idx in idxs:
                candidates[idx].partition = label
            continue

        coords = np.array([candidates[idx].position for idx in idxs])
        clustering = DBSCAN(eps=6.0, min_samples=10).fit(coords)
        sublabels = clustering.labels_
        unique_sublabels = np.unique(sublabels)
        partitions.pop(label)
        for sub in unique_sublabels:
            mask_sub = sublabels == sub
            sub_indices = [idxs[i] for i, flag in enumerate(mask_sub) if flag]
            if not sub_indices:
                continue
            new_label = (label << 8) + int(sub)
            partitions[new_label] = sub_indices
            for idx in sub_indices:
                candidates[idx].partition = new_label

    return partitions


def _build_graph(
    candidates: List[CandidateNode],
    config: Pathwalker2Config,
) -> Dict[int, List[Tuple[int, float]]]:
    """Return adjacency list with edge distances for neighbors within radius."""
    positions = np.array([cand.position for cand in candidates])
    tree = cKDTree(positions)
    radius = config.max_edge_length
    adjacency: Dict[int, List[Tuple[int, float]]] = {cand.idx: [] for cand in candidates}

    for cand in candidates:
        idx = cand.idx
        neighbors = tree.query_ball_point(cand.position, r=radius)
        for neighbor_idx in neighbors:
            if neighbor_idx == idx:
                continue
            dist = np.linalg.norm(positions[neighbor_idx] - cand.position)
            adjacency[idx].append((neighbor_idx, float(dist)))

    return adjacency


# ---------------------------------------------------------------------------
# Routing


def _edge_cost(
    candidates: Sequence[CandidateNode],
    prev_idx: Optional[int],
    i: int,
    j: int,
    distance: float,
    config: Pathwalker2Config,
) -> float:
    """Compute heuristic edge cost using geometry & map cues."""
    weights = config.weights
    node_i = candidates[i]
    node_j = candidates[j]

    # Distance penalty around 3.8 Å
    sigma_d = weights["sigma_d"]
    c_distance = weights["w_d"] * ((distance - 3.8) / sigma_d) ** 2

    # Density support: use smaller of the two densities
    min_density = min(node_i.density, node_j.density)
    c_density = weights["w_m"] / (1e-3 + np.clip(min_density, 1e-3, None))

    # Gradient alignment (prefer moving along ridges)
    direction = node_j.position - node_i.position
    direction_norm = np.linalg.norm(direction) + 1e-6
    direction_unit = direction / direction_norm
    c_grad = weights["w_g"] * (1.0 - abs(float(np.dot(node_i.gradient, direction_unit))))

    # Thickness (penalize thick blobs)
    c_thickness = weights["w_t"] * node_i.thickness

    # Curvature term using predecessor
    c_curv = 0.0
    if prev_idx is not None:
        prev_node = candidates[prev_idx]
        v1 = node_i.position - prev_node.position
        v2 = node_j.position - node_i.position
        v1 /= np.linalg.norm(v1) + 1e-6
        v2 /= np.linalg.norm(v2) + 1e-6
        angle = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))
        c_curv = weights["w_c"] * (angle**2)
        if np.dot(v1, v2) < 0:
            c_curv += weights["w_backtrack"]

    # Simple pseudo-Rama: prefer angles near 110 degrees
    c_rama = 0.0
    if prev_idx is not None:
        preferred = np.deg2rad(110.0)
        angle = np.arccos(
            np.clip(
                np.dot(
                    (candidates[prev_idx].position - node_i.position)
                    / (np.linalg.norm(candidates[prev_idx].position - node_i.position) + 1e-6),
                    direction_unit,
                ),
                -1.0,
                1.0,
            )
        )
        diff = angle - preferred
        c_rama = weights["w_ang"] * (diff**2)

    return c_distance + c_density + c_grad + c_thickness + c_curv + c_rama


def _route_partition(
    partition_indices: Sequence[int],
    candidates: List[CandidateNode],
    adjacency: Dict[int, List[Tuple[int, float]]],
    config: Pathwalker2Config,
    rng: np.random.Generator,
) -> List[List[int]]:
    """Greedy routing that builds multiple fragments per partition."""
    unused = set(int(idx) for idx in partition_indices)
    if not unused:
        return []

    # Sort by ridgeness to choose start nodes
    sorted_nodes = sorted(unused, key=lambda idx: candidates[idx].ridgeness, reverse=True)
    fragments: List[List[int]] = []

    while unused:
        # Start fragment at highest remaining ridgeness
        current = next(idx for idx in sorted_nodes if idx in unused)
        fragment = [current]
        unused.remove(current)
        prev_idx = None

        while True:
            neighbors = [
                (nbr, dist)
                for nbr, dist in adjacency[current]
                if nbr in unused and dist <= config.max_edge_length
            ]
            if not neighbors:
                break

            # Evaluate heuristic cost for each neighbor
            costs = [
                (_edge_cost(candidates, prev_idx, current, nbr, dist, config), nbr, dist)
                for nbr, dist in neighbors
            ]
            costs.sort(key=lambda tup: tup[0])
            best_cost, best_neighbor, _ = costs[0]

            # Allow optional skip: if cost too high, terminate fragment
            if best_cost > 15.0:  # heuristic threshold
                break

            fragment.append(best_neighbor)
            unused.remove(best_neighbor)
            prev_idx = current
            current = best_neighbor

        fragments.append(fragment)

    return fragments


def _route_and_score_partitions(
    candidates: List[CandidateNode],
    adjacency: Dict[int, List[Tuple[int, float]]],
    partitions: Dict[int, List[int]],
    config: Pathwalker2Config,
    rng: np.random.Generator,
) -> Dict[int, List[Fragment]]:
    """Route each partition and score fragments."""
    results: Dict[int, List[Fragment]] = {}
    total_nodes = len(candidates)

    for partition_id, indices in partitions.items():
        fragments_idx = _route_partition(indices, candidates, adjacency, config, rng)
        scored: List[Fragment] = []
        for frag_idx in fragments_idx:
            if len(frag_idx) < 2:
                continue
            # compute coverage, cost
            cost = 0.0
            crossovers = 0
            clashes = 0
            visited_positions = []
            for n1, n2 in zip(frag_idx[:-1], frag_idx[1:]):
                entry = next((dist for (nbr, dist) in adjacency[n1] if nbr == n2), None)
                if entry is None:
                    entry = np.linalg.norm(
                        candidates[n2].position - candidates[n1].position
                    )
                cost += _edge_cost(candidates, None, n1, n2, entry, config)

                # clash / crossover detection (simple)
                midpoint = 0.5 * (candidates[n1].position + candidates[n2].position)
                for pos in visited_positions:
                    if np.linalg.norm(midpoint - pos) < 2.5:
                        crossovers += 1
                visited_positions.append(midpoint)
                if entry < 3.0 or entry > 4.8:
                    clashes += 1

            coverage = len(frag_idx) / max(len(indices), 1)
            ss_consistency = np.mean(
                [1.0 if candidates[idx].ss_tag == "helix" else 0.5 for idx in frag_idx]
            )
            scored.append(
                Fragment(
                    node_indices=list(frag_idx),
                    cost=cost,
                    coverage=float(coverage),
                    ss_consistency=float(ss_consistency),
                    crossovers=crossovers,
                    clashes=clashes,
                )
            )

        # Rank fragments
        scored.sort(key=lambda frag: frag.cost - config.weights["coverage_boost"] * frag.coverage)
        results[partition_id] = scored[: config.fragments_per_partition]

    return results


# ---------------------------------------------------------------------------
# Output helpers


def _write_outputs(
    config: Pathwalker2Config,
    map_vol: MapVolume,
    prepped_map: np.ndarray,
    candidates: List[CandidateNode],
    fragments: Dict[int, List[Fragment]],
    prep_report: Dict[str, float],
) -> Dict[str, Path]:
    out_dir = config.out_dir

    # Write prepped map for reference (already thresholded in _prepare_map)
    prepped_map_path = out_dir / "pathwalker2_prepped_map.mrc"
    write_map(prepped_map_path, map_vol, prepped_map)

    # Write fragments pdb
    fragments_pdb = out_dir / "pathwalker2_fragments.pdb"
    _write_fragments_pdb(fragments_pdb, candidates, fragments)

    # Metadata json
    meta = {
        "config": asdict(config),
        "map_prep": prep_report,
        "num_candidates": len(candidates),
        "num_partitions": len(fragments),
        "fragments": {
            str(pid): [
                {
                    "nodes": frag.node_indices,
                    "cost": frag.cost,
                    "coverage": frag.coverage,
                    "ss_consistency": frag.ss_consistency,
                    "crossovers": frag.crossovers,
                    "clashes": frag.clashes,
                }
                for frag in frags
            ]
            for pid, frags in fragments.items()
        },
    }
    meta_path = out_dir / "pathwalker2_meta.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=_json_default)

    return {
        "prepped_map": prepped_map_path,
        "fragments_pdb": fragments_pdb,
        "metadata": meta_path,
    }


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _write_fragments_pdb(
    path: Path,
    candidates: List[CandidateNode],
    fragments: Dict[int, List[Fragment]],
) -> None:
    """Write fragments as PDB models."""
    with path.open("w", encoding="utf-8") as f:
        model_id = 1
        for partition_id, frag_list in fragments.items():
            for frag in frag_list:
                f.write(f"MODEL     {model_id:4d}\n")
                serial = 1
                for idx in frag.node_indices:
                    node = candidates[idx]
                    x, y, z = node.position
                    bfactor = max(5.0, min(100.0, 50.0 * node.ridgeness))
                    chain_id = chr(65 + (partition_id % 26))
                    f.write(
                        f"HETATM{serial:5d}  CA  TRC {chain_id:1s}{serial:4d}    "
                        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00{bfactor:6.2f}           C  \n"
                    )
                    serial += 1
                f.write("ENDMDL\n")
                model_id += 1


# ---------------------------------------------------------------------------
# Public helper for CLI usage


def load_config_from_cli_kwargs(**kwargs) -> Pathwalker2Config:
    """Utility for CLI to build config object from parsed keywords."""
    return Pathwalker2Config(
        map_path=Path(kwargs["map"]),
        threshold=kwargs["threshold"],
        n_residues=kwargs["n_residues"],
        out_dir=Path(kwargs["out_dir"]),
        seed_method=kwargs.get("seed_method", "ridge"),
        grid_step=kwargs.get("grid_step", 1.0),
        map_prep=kwargs.get("map_prep", "locscale"),
        gaussian_sigma=kwargs.get("gaussian_sigma", 0.6),
        k_neighbors=kwargs.get("k_neighbors", 12),
        max_edge_length=kwargs.get("max_edge_length", 5.5),
        fragments_per_partition=kwargs.get("fragments_per_partition", 10),
        random_state=kwargs.get("random_state"),
    )


