# features.py
from typing import Dict, Any
import numpy as np
from sklearn.neighbors import KDTree
from ..core.types import MapVolume, ModelAtoms

def extract_patch(vol: MapVolume, center_vox: np.ndarray, box: int = 16) -> np.ndarray:
    zyx = np.round(center_vox).astype(int)
    z0,y0,x0 = np.maximum(zyx - box//2, 0)
    z1,y1,x1 = np.minimum(zyx + box//2, np.array(vol.data.shape)-1)
    return vol.data[z0:z1, y0:y1, x0:x1].copy().astype(np.float32)

def neighbor_graph(model: ModelAtoms, center_xyz: np.ndarray, cutoff: float = 8.0) -> Dict[str, np.ndarray]:
    tree = KDTree(model.xyz)
    idx = tree.query_radius(center_xyz[None,:], r=cutoff)[0]
    return {"idx": idx.astype(int),
            "rel_xyz": (model.xyz[idx] - center_xyz).astype(np.float32),
            "element": model.element[idx].astype(object),
            "resname": model.resname[idx].astype(object)}

def coordination_features(rel_xyz: np.ndarray, element: np.ndarray) -> Dict[str, Any]:
    d = np.linalg.norm(rel_xyz, axis=1)
    ons = np.isin(element, ["O","N","S"])
    shells = [2.3, 2.8, 3.2, 4.0, 6.0]
    counts = [int(((d<=r) & ons).sum()) for r in shells]
    return {"shell_counts": counts, "min_ONS": float(np.min(d[ons])) if np.any(ons) else 9.99}
