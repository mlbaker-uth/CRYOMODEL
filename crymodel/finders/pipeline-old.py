# crymodel/finders/pipeline.py
from __future__ import annotations
import numpy as np
from dataclasses import asdict
from scipy.ndimage import label

from sklearn.cluster import DBSCAN
from sklearn.neighbors import KDTree

from ..core.types import MapVolume, ModelAtoms, Assignment, AssignmentSet
from ..maps.ops import (mask_by_model, apply_mask, vox_to_ang, within_distance_window, density_at_points)
from ..pseudo.core import merge_by_distance
from .components import split_by_component_size
from ..pseudo.uniform import generate_pseudoatoms_uniform
from ..pseudo.greedy import generate_pseudoatoms_greedy
#from ..pseudo.rich import generate_pseudoatoms_rich
from .microblobs import micro_water_pass
from ..io.mrc import read_map_with_halves, write_map, as_volume_like

# ---------------------------- quick classifiers ----------------------------

def _class_from_cluster_size(size: int) -> str:
    # simple heuristic until ML head is added
    if size <= 1:
        return "water"
    if size <= 3:
        return "M2+"
    return "ligand"

def _ons_features(center_ang: np.ndarray, ons_xyz: np.ndarray) -> tuple[float, int]:
    """Return (min_distance_to_ONS, n_neighbors_within_2.8A)."""
    if ons_xyz.size == 0:
        return 9.99, 0
    tree = KDTree(ons_xyz)
    d, _ = tree.query(center_ang.reshape(1, -1).astype(np.float32), k=min(8, len(ons_xyz)))
    d = d.reshape(-1)
    return float(d.min()) if len(d) else 9.99, int((d <= 2.8).sum())


# ------------------------------ main pipeline ------------------------------

def run_pipeline(
    vol: MapVolume,
    model: ModelAtoms,
    thresh: float = 0.5,
    dmin: float = 2.0, dmax: float = 6.0, mask_radius: float = 2.0,
    mode: str = "uniform",
    # uniform
    spacing_A: float = 0.45, stride_vox: int = 1, bias_power: float = 1.1,
    # greedy
    zero_radius_A: float = 0.45,
    # rich
    rich_step_vox: int = 1, rich_sigma_vox: float = 0.3, rich_ascent_steps: int = 0, nms_radius_A: float = 0.45,
    # micro-blob (water-first)
    micro_vvox_max: int = 8, micro_phi_min: float = 0.6, micro_near_ONS: float = 3.2,
    # half-maps / clustering
    bandwidth: float = 1.2, merge_radius: float = 2.0, half_thr_scale: float = 0.5,
) -> AssignmentSet:
    """
    Returns:
      AssignmentSet with centers in Å (assigns.centers) and a list of Assignment objects.
      assigns.meta includes debug counters and (optionally) seed list, etc.
    """

    # 1) Mask modeled region out (keep only unmodeled)
    mdl_mask = mask_by_model(vol, model, radius_A=mask_radius)
    unmodeled = apply_mask(vol, mdl_mask, invert=True)

    # binary mask at thresh for debug
    thr_mask = (unmodeled.data >= float(thresh))
    meta = {
        "mode": mode,
        "n_vox_thr": int(thr_mask.sum()),
        "mask_radius_A": float(mask_radius),
        "apix": float(vol.apix),
    }
        
    if meta["n_vox_thr"] == 0:
        return AssignmentSet(centers=np.zeros((0,3), np.float32), assignments=[], meta=meta)


    # --- Stage 1: micro-blob (water-first) ---
    mp = micro_water_pass(
        unmodeled=unmodeled,
        thr_mask=thr_mask,
        apix=float(vol.apix),
        model=model,
        vvox_max=int(micro_vvox_max),
        phi_min=float(micro_phi_min),
        nearest_ONS_max=float(micro_near_ONS),
    )
    meta.update({
        "micro_N": int(len(mp.centers_ang)),
        "micro_vvox_max": int(micro_vvox_max),
        "micro_phi_min": float(micro_phi_min),
        "micro_near_ONS": float(micro_near_ONS),
    })
    # Mask out micro-blob voxels for the ligand/lipid stage
    thr_mask_remaining = mp.remaining_mask
    #masked_rest = as_volume_like(unmodeled.data * thr_mask_remaining.astype(np.float32), unmodeled)
    masked_rest = as_volume_like(unmodeled.data, unmodeled)  # use original values
    
    # 2) Generate pseudoatoms (Å)
    mode_lc = str(mode).lower()
    if mode_lc == "uniform":
        centers_ang_rest = generate_pseudoatoms_uniform(
            masked_rest, float(vol.apix), float(thresh),
            spacing_A=float(spacing_A), stride_vox=int(stride_vox), bias_power=float(bias_power),
            mask_bool=thr_mask_remaining,     # <— CRITICAL
        )
    elif mode_lc == "greedy":
        centers_ang_rest = generate_pseudoatoms_greedy(
            masked_rest, float(vol.apix), float(thresh),
            zero_radius_A=float(zero_radius_A),
            mask_bool=thr_mask_remaining,     # <— CRITICAL
        )
    elif mode_lc == "rich":
        centers_ang_rest = generate_pseudoatoms_rich(
            masked_rest, float(vol.apix), float(thresh),
            smooth_sigma_vox=float(rich_sigma_vox),
            step_vox=int(rich_step_vox),
            ascent_steps=int(rich_ascent_steps),
            step_scale=0.6,
            nms_radius_A=float(nms_radius_A),
            mask_bool=thr_mask_remaining,     # add if you patch rich similarly
        )
    
    # combine centers (waters first to make indexing easy)
    if len(mp.centers_ang):
        centers_all = np.vstack([mp.centers_ang, centers_ang_rest]).astype(np.float32)
    else:
        centers_all = centers_ang_rest.astype(np.float32)
        
    # --- Final guard: keep only centers that land on the remaining threshold mask ---
    def _ang_to_vox(pts_ang: np.ndarray, mv: MapVolume) -> np.ndarray:
        # convert from Å to voxel indices, respecting origin
        p = (pts_ang - np.asarray(mv.origin, dtype=np.float32)) / float(mv.apix)
        p = np.round(p).astype(int)
        Z, Y, X = mv.data.shape
        p[:, 0] = np.clip(p[:, 0], 0, Z - 1)
        p[:, 1] = np.clip(p[:, 1], 0, Y - 1)
        p[:, 2] = np.clip(p[:, 2], 0, X - 1)
        return p

    if len(centers_all):
        vox_all = _ang_to_vox(centers_all, masked_rest)
        keep_thr = thr_mask_remaining[vox_all[:, 0], vox_all[:, 1], vox_all[:, 2]]
        centers_all = centers_all[keep_thr]
        meta["n_centers_after_thr_mask"] = int(len(centers_all))

    if len(centers_all) == 0:
        return AssignmentSet(centers=np.zeros((0,3), np.float32), assignments=[], meta=meta)
        
    # Distance window (Å) relative to protein
    if len(centers_all):
        keep = within_distance_window(centers_all, model, float(dmin), float(dmax))
        centers_all = centers_all[keep]
    meta.update({"dmin": float(dmin), "dmax": float(dmax), "n_centers_after_dist": int(len(centers_all))})
    if len(centers_all) == 0:
        return AssignmentSet(centers=np.zeros((0,3), np.float32), assignments=[], meta=meta)

    # Optional half-map gating
    if vol.halfmaps is not None:
        h1, h2 = vol.halfmaps
        mv1 = MapVolume(data=h1, apix=vol.apix, origin=vol.origin)
        mv2 = MapVolume(data=h2, apix=vol.apix, origin=vol.origin)
        d1 = density_at_points(mv1, centers_all)
        d2 = density_at_points(mv2, centers_all)
        keep = (d1 > 0.0) & (d2 > 0.0) & ((d1 + d2) >= float(half_thr_scale) * float(thresh))
        centers_all = centers_all[keep]
        meta["n_centers_after_halfmaps"] = int(len(centers_all))
        meta["had_halfmaps"] = True
    else:
        meta["had_halfmaps"] = False

    # unify name going forward
    centers = centers_all
    if len(centers) == 0:
        return AssignmentSet(centers=np.zeros((0,3), np.float32), assignments=[], meta=meta)
    
    # 5) Build assignments
    assignments: list[Assignment] = []

    # micro-waters first (locked)
    n_micro = len(mp.centers_ang)
    for i, a in enumerate(mp.assignments):
        a.index = int(i)
        a.cluster_id = -1
        assignments.append(a)

    # remainder to cluster (ligand/lipid geometry signal)
    if len(centers) > n_micro:
        rem = centers[n_micro:].astype(np.float32)

        # cluster remainder only
        clustering = DBSCAN(eps=1.4, min_samples=1).fit(rem)
        labels = clustering.labels_
        meta["n_clusters"] = int(labels.max() + 1) if labels.size else 0

        # sizes per cluster
        sizes = {int(lbl): int((labels == lbl).sum()) for lbl in np.unique(labels)}

        # O/N/S neighbors (build KDTree once)
        ons_mask = np.isin(model.element.astype(str), ["O", "N", "S"])
        ons_xyz = model.xyz[ons_mask]
        tree = KDTree(ons_xyz) if ons_xyz.size else None

        def _ons_features(center_ang: np.ndarray) -> tuple[float,int]:
            if tree is None:
                return 9.99, 0
            d, _ = tree.query(center_ang.reshape(1, -1).astype(np.float32), k=min(8, len(ons_xyz)))
            d = d.reshape(-1)
            return (float(d.min()) if len(d) else 9.99), int((d <= 2.8).sum())

        # build assignments for remainder
        for j, c in enumerate(rem):
            lbl = int(labels[j])
            size = sizes.get(lbl, 1)
            nearest, n28 = _ons_features(c)

            if size == 1 and nearest <= 3.2:
                top = "water";  probs = {"water": 0.9, "unknown": 0.1}
            elif size <= 3 and n28 >= 2:
                top = "M2+";    probs = {"M2+": 0.6, "unknown": 0.4}
            elif size >= 4:
                top = "ligand"; probs = {"ligand": 0.8, "unknown": 0.2}
            else:
                top = "unknown"; probs = {"unknown": 1.0}

            assignments.append(
                Assignment(
                    index=int(n_micro + j),
                    cluster_id=lbl,
                    probs=probs,
                    top=top,
                    explain={"cluster_size": int(size), "nearest_ONS": float(nearest), "n_ONS_2p8": int(n28)},
                )
            )
    else:
        meta["n_clusters"] = 0
    

    # protein O/N/S lookup for coordination features (optional but handy)
    #ons_mask = np.isin(model.element.astype(str), ["O", "N", "S"])
    #ons_xyz = model.xyz[ons_mask]

    assignments: list[Assignment] = []
    #centers = centers_all  # final centers array

    n_micro = len(mp.centers_ang)
    # 1) micro-water assignments, update their indices
    for i, a in enumerate(mp.assignments):
        a.index = int(i)
        a.cluster_id = -1
        assignments.append(a)

    # 2) cluster the remainder (ligand/lipid geometry signal)
    if len(centers) > n_micro:
        #from sklearn.cluster import DBSCAN
        rem = centers[n_micro:].astype(np.float32)
        clustering = DBSCAN(eps=1.4, min_samples=1).fit(rem)
        labels = clustering.labels_
        # sizes per cluster
        sizes = {int(lbl): int((labels == lbl).sum()) for lbl in np.unique(labels)}

        # protein O/N/S for features
        ons_mask = np.isin(model.element.astype(str), ["O", "N", "S"])
        ons_xyz = model.xyz[ons_mask]
        tree = KDTree(ons_xyz) if len(ons_xyz) else None

        def _ons_features(center_ang):
            if tree is None or len(ons_xyz) == 0:
                return 9.99, 0
            d, _ = tree.query(center_ang.reshape(1, -1).astype(np.float32), k=min(8, len(ons_xyz)))
            d = d.reshape(-1)
            return (float(d.min()) if len(d) else 9.99), int((d <= 2.8).sum())

        # build assignments for remainder
        for j, c in enumerate(rem):
            lbl = int(labels[j])
            size = sizes.get(lbl, 1)
            nearest, n28 = _ons_features(c)

            # quick rule: size==1 & nearest<=3.2 Å ⇒ likely water (but not micro)
            if size == 1 and nearest <= 3.2:
                top = "water"
                probs = {"water": 0.9, "unknown": 0.1}
            elif size <= 3 and n28 >= 2:
                top = "M2+"
                probs = {"M2+": 0.6, "unknown": 0.4}
            elif size >= 4:
                top = "ligand"
                probs = {"ligand": 0.8, "unknown": 0.2}
            else:
                top = "unknown"
                probs = {"unknown": 1.0}

            assignments.append(
                Assignment(
                    index=int(n_micro + j),
                    cluster_id=int(lbl),
                    probs=probs,
                    top=top,
                    explain={"cluster_size": int(size), "nearest_ONS": float(nearest), "n_ONS_2p8": int(n28)},
                )
            )
    else:
        labels = np.array([], dtype=int)

    # classes list (unchanged)
    classes = ["water","Na+","K+","Mg2+","Ca2+","Cl-","M2+",
               "lipid_head","lipid_tail","sterol","detergent","carbohydrate","ligand","unknown"]
    meta["classes"] = classes
    meta["n_assignments"] = int(len(assignments))
    return AssignmentSet(centers=centers.astype(np.float32), assignments=assignments, meta=meta)
    

