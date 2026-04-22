from typing import Optional, Callable, Tuple
import numpy as np

def build_distance_matrix(points: np.ndarray, weight: Optional[Callable[[np.ndarray, np.ndarray], float]] = None) -> np.ndarray:
    N = len(points); D = np.zeros((N,N), np.float32)
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(points[i]-points[j])
            if weight is not None: d = weight(points[i], points[j])
            D[i,j] = D[j,i] = d
    return D

def solve_tsp_greedy(D: np.ndarray) -> np.ndarray:
    N = D.shape[0]; visited=[0]; remaining=set(range(1,N))
    while remaining:
        i = visited[-1]; j = min(remaining, key=lambda k: D[i,k])
        visited.append(j); remaining.remove(j)
    return np.array(visited, int)

def build_path(points: np.ndarray, anchors: Optional[Tuple[int,int]] = None) -> np.ndarray:
    D = build_distance_matrix(points); order = solve_tsp_greedy(D)
    if anchors is not None:
        a, b = anchors
        ia = np.argmin(np.linalg.norm(points - points[a], axis=1))
        ia_pos = int(np.where(order==ia)[0][0]); order = np.roll(order, -ia_pos)
    return points[order]
