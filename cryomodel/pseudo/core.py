from typing import Tuple
import numpy as np
try:
    from sklearn.cluster import KMeans, AgglomerativeClustering, MeanShift, Birch, SpectralClustering
except Exception:
    KMeans = AgglomerativeClustering = MeanShift = Birch = SpectralClustering = None

from ..core.types import PseudoAtoms

def merge_by_distance(xyz: np.ndarray, cutoff: float = 2.5) -> Tuple[np.ndarray, np.ndarray]:
    if len(xyz) == 0: return xyz, np.empty((0,), int)
    used = np.zeros(len(xyz), bool); centers=[]; labels=-np.ones(len(xyz), int); cid=0
    for i in range(len(xyz)):
        if used[i]: continue
        d = np.linalg.norm(xyz - xyz[i], axis=1)
        members = np.where(d <= cutoff)[0]
        used[members] = True
        centers.append(xyz[members].mean(axis=0))
        labels[members] = cid; cid += 1
    return np.vstack(centers), labels

def cluster_points(xyz: np.ndarray, method: str = "meanshift", **kw) -> PseudoAtoms:
    if len(xyz) == 0: return PseudoAtoms(xyz=np.empty((0,3)))
    m = (method or "meanshift").lower()
    if m == "kmeans" and KMeans is not None:
        n = kw.get("n_clusters", 100)
        labels = KMeans(n_clusters=n, random_state=0).fit_predict(xyz)
        centers = np.vstack([xyz[labels==i].mean(axis=0) for i in range(n)])
    elif m == "agglomerative" and AgglomerativeClustering is not None:
        n = kw.get("n_clusters", 100)
        labels = AgglomerativeClustering(n_clusters=n).fit_predict(xyz)
        centers = np.vstack([xyz[labels==i].mean(axis=0) for i in range(n)])
    elif m == "birch" and Birch is not None:
        n = kw.get("n_clusters", 100)
        labels = Birch(n_clusters=n).fit_predict(xyz)
        centers = np.vstack([xyz[labels==i].mean(axis=0) for i in range(n)])
    else:
        if MeanShift is None: raise ImportError("scikit-learn not available for clustering.")
        ms = MeanShift(bandwidth=kw.get("bandwidth", None), bin_seeding=True)
        labels = ms.fit_predict(xyz); centers = ms.cluster_centers_
    return PseudoAtoms(xyz=centers.astype('float32'), cluster_id=labels.astype(int))
