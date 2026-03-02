# crymodel/finders/phase45.py
from __future__ import annotations
from typing import List

import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import label

from ..io.mrc import MapVolume


def gate_points_by_distance(points_xyzA: np.ndarray,
                            model_xyzA: np.ndarray,
                            dmin_A: float,
                            dmax_A: float) -> np.ndarray:
    if points_xyzA.size == 0 or model_xyzA.size == 0:
        return np.zeros((0,3), dtype=np.float32)
    tree = cKDTree(model_xyzA.astype(np.float32))
    dists, _ = tree.query(points_xyzA.astype(np.float32), k=1, workers=-1)
    keep = (dists >= float(dmin_A)) & (dists <= float(dmax_A))
    return points_xyzA[keep]


def cluster_single_linkage(points_xyzA: np.ndarray, link_radius_A: float) -> List[np.ndarray]:
    if points_xyzA.size == 0:
        return []
    tree = cKDTree(points_xyzA.astype(np.float32))
    visited = np.zeros(points_xyzA.shape[0], dtype=bool)
    clusters: List[np.ndarray] = []
    for i in range(points_xyzA.shape[0]):
        if visited[i]:
            continue
        stack = [i]
        group = []
        visited[i] = True
        while stack:
            j = stack.pop()
            group.append(j)
            idxs = tree.query_ball_point(points_xyzA[j], r=float(link_radius_A))
            for k in idxs:
                if not visited[k]:
                    visited[k] = True
                    stack.append(k)
        clusters.append(points_xyzA[np.array(group)])
    return clusters


def cluster_mean_centers(points_xyzA: np.ndarray, link_radius_A: float) -> np.ndarray:
    clusters = cluster_single_linkage(points_xyzA, link_radius_A)
    if not clusters:
        return np.zeros((0,3), dtype=np.float32)
    centers = [c.mean(axis=0) for c in clusters]
    return np.asarray(centers, dtype=np.float32)


def _zyx_to_xyzA(zyx_idx: np.ndarray, vol: MapVolume) -> np.ndarray:
    if zyx_idx.size == 0:
        return np.zeros((0,3), dtype=np.float32)
    apix = float(vol.apix)
    ox, oy, oz = [float(v) for v in vol.origin_xyzA]
    z = zyx_idx[:, 0].astype(float)
    y = zyx_idx[:, 1].astype(float)
    x = zyx_idx[:, 2].astype(float)
    xA = ox + x * apix
    yA = oy + y * apix
    zA = oz + z * apix
    return np.stack([xA, yA, zA], axis=1).astype(np.float32)


def greedy_pseudoatoms_per_ligand_component(ligands_map: MapVolume,
                                            zero_radius_A: float) -> List[np.ndarray]:
    """Return list of pseudoatom xyzA arrays, one per ligand component, via greedy maxima.
    Oversamples shape within each component by zeroing a small spherical radius per pick.
    """
    data = ligands_map.data_zyx
    labels, nlab = label(data > 0.0)
    if nlab == 0:
        return []

    apix = float(ligands_map.apix)
    r_vox = max(1.0, float(zero_radius_A) / apix)
    r = int(np.floor(r_vox))
    r2 = r_vox * r_vox
    dz = np.arange(-r, r+1)
    dy = np.arange(-r, r+1)
    dx = np.arange(-r, r+1)
    offsets = np.array([(zz,yy,xx) for zz in dz for yy in dy for xx in dx
                        if (zz*zz + yy*yy + xx*xx) <= r2], dtype=int)

    Z, Y, X = data.shape
    out: List[np.ndarray] = []
    for cc_id in range(1, nlab+1):
        comp = np.where(labels == cc_id, data, 0.0).copy()
        centers = []
        while True:
            flat_idx = int(np.argmax(comp))
            vmax = float(comp.ravel()[flat_idx])
            if vmax <= 0.0:
                break
            z, y, x = np.unravel_index(flat_idx, comp.shape)
            centers.append([z, y, x])
            for zz, yy, xx in offsets:
                z2 = z + zz; y2 = y + yy; x2 = x + xx
                if 0 <= z2 < Z and 0 <= y2 < Y and 0 <= x2 < X:
                    comp[z2, y2, x2] = 0.0
        centers = np.asarray(centers, dtype=int)
        out.append(_zyx_to_xyzA(centers, ligands_map))
    return out


def gate_ligand_components_by_distance(components_xyzA: List[np.ndarray],
                                        model_xyzA: np.ndarray,
                                        dmin_A: float,
                                        dmax_A: float) -> List[np.ndarray]:
    if not components_xyzA or model_xyzA.size == 0:
        return []
    tree = cKDTree(model_xyzA.astype(np.float32))
    kept: List[np.ndarray] = []
    for comp in components_xyzA:
        if comp.size == 0:
            continue
        dists, _ = tree.query(comp.astype(np.float32), k=1, workers=-1)
        if np.any((dists >= float(dmin_A)) & (dists <= float(dmax_A))):
            kept.append(comp)
    return kept


