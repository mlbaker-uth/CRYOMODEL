import os
import shutil
import subprocess
from pathlib import Path

import numpy as np

# Optional SciPy imports. ChimeraX installations may not include SciPy.
try:
    import scipy.spatial.distance as scipydist
except Exception:
    scipydist = None

try:
    from scipy.optimize import minimize
except Exception:
    minimize = None

# Optional scikit-learn import. ChimeraX installations may not include sklearn.
try:
    from sklearn.cluster import KMeans
except Exception:
    KMeans = None


def _cdist(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if scipydist is not None:
        return scipydist.cdist(a, b)
    diff = a[:, None, :] - b[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2))


def _simple_kmeans(points, weights, n_clusters, max_iter=20):
    """NumPy-only weighted k-means fallback."""
    pts = np.asarray(points, dtype=float)
    wts = np.asarray(weights, dtype=float)
    n_pts = len(pts)
    if n_pts == 0:
        raise ValueError('No points provided for clustering.')
    n_clusters = max(1, min(int(n_clusters), n_pts))

    # Initialize centers by weighted quantiles over a stable ordering.
    order = np.argsort(np.sum(pts, axis=1))
    pts_ord = pts[order]
    wts_ord = wts[order]
    cum = np.cumsum(wts_ord)
    total = cum[-1] if len(cum) else 1.0
    targets = np.linspace(0, total, n_clusters + 2)[1:-1]
    idxs = [np.searchsorted(cum, t, side='left') for t in targets]
    centers = pts_ord[np.clip(idxs, 0, n_pts - 1)].copy()

    labels = np.zeros(n_pts, dtype=int)
    for _ in range(max_iter):
        d2 = np.sum((pts[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        new_labels = np.argmin(d2, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        new_centers = []
        for k in range(n_clusters):
            mask = labels == k
            if not np.any(mask):
                new_centers.append(centers[k])
                continue
            ww = wts[mask]
            pp = pts[mask]
            sw = np.sum(ww)
            if sw <= 0:
                new_centers.append(np.mean(pp, axis=0))
            else:
                new_centers.append(np.sum(pp * ww[:, None], axis=0) / sw)
        centers = np.asarray(new_centers, dtype=float)
    return centers


def _volume_matrix(volume):
    if hasattr(volume, 'matrix'):
        data = np.array(volume.matrix())
    elif hasattr(volume, 'full_matrix'):
        data = np.array(volume.full_matrix())
    else:
        raise AttributeError('Volume model has neither matrix() nor full_matrix().')
    return np.array(data).T


def _surface_level(volume):
    # Common direct attributes.
    for attr in ('surface_levels', 'levels', 'image_levels'):
        if hasattr(volume, attr):
            try:
                val = getattr(volume, attr)
                if len(val) > 0:
                    first = val[0]
                    if isinstance(first, (tuple, list)):
                        return float(first[0])
                    return float(first)
            except Exception:
                pass

    # Some volume objects expose contour levels on surface models.
    if hasattr(volume, 'surfaces'):
        try:
            surfs = getattr(volume, 'surfaces')
            if surfs and len(surfs) > 0:
                s0 = surfs[0]
                for sattr in ('level', 'contour_level'):
                    if hasattr(s0, sattr):
                        return float(getattr(s0, sattr))
        except Exception:
            pass

    # Conservative fallback from map values.
    data = _volume_matrix(volume)
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        raise AttributeError('Could not determine map threshold from the volume model.')
    return float(np.percentile(finite, 90.0))


def _step_origin(volume):
    data = volume.data
    step = np.array(getattr(data, 'step', (1.0, 1.0, 1.0)), dtype=float)
    origin = np.array(getattr(data, 'origin', (0.0, 0.0, 0.0)), dtype=float)
    return step, origin


def seed_from_volume(volume, nres, threshold=None):
    data = _volume_matrix(volume)
    thr = _surface_level(volume) if threshold is None else float(threshold)
    step, origin = _step_origin(volume)
    pts = np.array(np.where(data > thr), dtype=float).T
    if len(pts) == 0:
        raise ValueError(f'No voxels above threshold {thr:.4g}.')
    wts = data[data > thr]
    pts = pts * step

    if KMeans is not None:
        km = KMeans(n_clusters=nres, max_iter=20, n_init=10)
        km.fit(pts, sample_weight=wts)
        centers = km.cluster_centers_
    else:
        centers = _simple_kmeans(pts, wts, nres, max_iter=20)

    return centers + origin


def write_tsplib(filename, dstmat, fixededges=None):
    fixededges = fixededges or []
    with open(filename, 'w') as fout:
        fout.write('\n'.join([
            f'NAME: {filename}',
            'TYPE: TSP',
            f'COMMENT: {filename}',
            f'DIMENSION: {len(dstmat):d}',
            'EDGE_WEIGHT_TYPE: EXPLICIT',
            'EDGE_WEIGHT_FORMAT: FULL_MATRIX',
            ''
        ]))
        if fixededges:
            fout.write('FIXED_EDGES_SECTION\n')
            for i, j in fixededges:
                fout.write(f'{i + 1:d} {j + 1:d}\n')
            fout.write('-1\n')
        fout.write('EDGE_WEIGHT_SECTION\n')
        for dst in dstmat:
            fout.write(' '.join([f'{int(d * 10):d}' for d in dst]) + '\n')
        fout.write('EOF')


def _weighted_distance(points, map_model=None, map_weight=0.0, distance_cutoff=15.0, sample_points=11):
    dst = _cdist(points, points)
    if map_model is None or map_weight <= 0:
        return dst
    data = _volume_matrix(map_model).copy()
    data[data < 0] = 0
    step, origin = _step_origin(map_model)
    pts = (points - origin) / step
    wtdst = np.zeros_like(dst)
    nx, ny, nz = data.shape
    for ix in range(len(points)):
        px = pts[ix]
        smp = np.round(px).astype(int)
        if 0 <= smp[0] < nx and 0 <= smp[1] < ny and 0 <= smp[2] < nz:
            wtdst[ix, ix] = data[smp[0], smp[1], smp[2]]
        for iy in range(ix):
            if dst[ix, iy] > distance_cutoff:
                continue
            py = pts[iy]
            dt = (py - px) / sample_points
            smp = px + np.arange(1, sample_points)[:, None] * dt
            smp = np.round(smp).astype(int)
            mask = (
                (smp[:, 0] >= 0) & (smp[:, 0] < nx) &
                (smp[:, 1] >= 0) & (smp[:, 1] < ny) &
                (smp[:, 2] >= 0) & (smp[:, 2] < nz)
            )
            if not np.any(mask):
                continue
            ss = smp[mask]
            val = np.mean(data[ss[:, 0], ss[:, 1], ss[:, 2]])
            wtdst[ix, iy] = wtdst[iy, ix] = val
    if np.max(wtdst) > 0:
        wtdst /= np.max(wtdst)
        wtdst = 1 - wtdst
        dst = dst + wtdst * map_weight
    return dst


def _run_lkh(work_dir, lkh_executable):
    work_dir = Path(work_dir)
    par = work_dir / 'lkh_run.txt'
    out = work_dir / 'pts.out'
    with open(par, 'w') as f:
        f.write('PROBLEM_FILE = pts.tsp\nOUTPUT_TOUR_FILE = pts.out\nPRECISION = 100\nRUNS = 5')
    if shutil.which(lkh_executable) is None and not Path(lkh_executable).exists():
        raise FileNotFoundError(f'LKH executable not found: {lkh_executable}')
    proc = subprocess.run([lkh_executable, str(par)], cwd=str(work_dir), capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr or proc.stdout or 'LKH failed')
    if not out.exists():
        raise RuntimeError('LKH did not produce pts.out')
    tour = []
    start = False
    for line in out.read_text().splitlines():
        if line.startswith('TOUR_SECTION'):
            start = True
            continue
        if start:
            idx = int(line.strip())
            if idx > 0:
                tour.append(idx)
            else:
                break
    return np.array(tour, dtype=int)


def trace_path(points, fixed_edges=None, map_model=None, map_weight=0.0,
               distance_cutoff=15.0, sample_points=11, lkh_executable='LKH',
               work_dir=None, keep_temp=False):
    fixed_edges = fixed_edges or []
    work_dir = Path(work_dir or Path.cwd())
    work_dir.mkdir(parents=True, exist_ok=True)

    dst = _weighted_distance(points, map_model=map_model, map_weight=map_weight,
                             distance_cutoff=distance_cutoff, sample_points=sample_points)
    ngap = 1
    dst = np.vstack([np.zeros((ngap, dst.shape[1])), dst])
    dst = np.hstack([np.zeros((dst.shape[0], ngap)), dst])
    fx = np.array(fixed_edges, dtype=int) if fixed_edges else np.zeros((0, 2), dtype=int)
    write_tsplib(work_dir / 'pts.tsp', dst, fx + ngap)
    tour = _run_lkh(work_dir, lkh_executable)
    if len(tour) == 0:
        raise RuntimeError('Empty LKH tour returned')
    if np.any(tour == 1):
        gap_idx = int(np.where(tour == 1)[0][0])
        tour = np.concatenate([tour[gap_idx + 1:], tour[:gap_idx]])
    tour = tour - 2
    if np.any((tour < 0) | (tour >= len(points))):
        raise RuntimeError('Invalid tour returned by LKH after removing gap node')
    ordered_points = points[tour]

    inv = {old: new for new, old in enumerate(tour.tolist())}
    fixed_reindexed = []
    for i, j in fixed_edges:
        if i in inv and j in inv:
            fixed_reindexed.append((inv[i], inv[j]))

    if not keep_temp:
        for fn in ('pts.tsp', 'pts.out', 'lkh_run.txt'):
            p = work_dir / fn
            if p.exists():
                try:
                    p.unlink()
                except Exception:
                    pass

    return {
        'order': tour.tolist(),
        'ordered_points': ordered_points,
        'fixed_edges_reindexed': fixed_reindexed,
    }


def rot_a_to_b(a, b):
    a = np.array(a, dtype=float) / np.linalg.norm(a)
    b = np.array(b, dtype=float) / np.linalg.norm(b)
    v = np.cross(a, b).astype(float)
    nv = np.linalg.norm(v)
    if nv < 1e-8:
        return np.eye(3)
    s = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + s + np.dot(s, s) * ((1 - np.dot(a, b)) / (nv ** 2))


def make_hlx(n, pms):
    phs = pms[0]
    v = [pms[1], pms[2], pms[3]]
    dz = 1.5
    dtheta = 100.0 / 180 * np.pi
    rad = 2.3
    idx = np.arange(n, dtype=float)
    hlx = np.zeros((n, 3), dtype=float)
    hlx[:, 0] = rad * np.cos(phs + idx * dtheta)
    hlx[:, 1] = rad * np.sin(phs + idx * dtheta)
    hlx[:, 2] = idx * dz
    hlx -= np.mean(hlx, 0)
    rot = rot_a_to_b([0, 0, 1], v)
    return np.dot(rot, hlx.T).T


def refine_hlx(pms, n, pts):
    hlx = make_hlx(n, pms)
    hlx += np.mean(pts, 0)
    dst = _cdist(pts, hlx)
    return np.mean(np.min(dst, axis=0))


def remap_fixed_edges_after_insert(fixed_edges, insert_at):
    out = []
    for i, j in fixed_edges:
        ii = i + 1 if i >= insert_at else i
        jj = j + 1 if j >= insert_at else j
        out.append(tuple(sorted((ii, jj))))
    return sorted(set(out))


def remap_fixed_edges_after_delete(fixed_edges, delete_indices):
    delete_set = set(delete_indices)
    mapping = {}
    shift = 0
    max_idx = max([x for e in fixed_edges for x in e], default=-1)
    for i in range(max_idx + len(delete_indices) + 2):
        if i in delete_set:
            shift += 1
        else:
            mapping[i] = i - shift
    out = []
    for i, j in fixed_edges:
        if i in delete_set or j in delete_set:
            continue
        if i in mapping and j in mapping:
            out.append(tuple(sorted((mapping[i], mapping[j]))))
    return sorted(set(out))


def replace_segment_with_helix(points, i0, i1, fixed_edges):
    sel = points[i0:i1 + 1]
    pts = np.array(sel, dtype=float)
    dz = 1.5
    l = np.sqrt(np.sum((pts[-1] - pts[0]) ** 2))
    n = max(2, int(np.round(l / dz)))
    v = (pts[-1] - pts[0]) / l
    pms = [0, v[0], v[1], v[2]]

    if minimize is not None:
        res = minimize(refine_hlx, pms, (n, pts), method='Nelder-Mead', tol=1e-6)
        best = res.x
    else:
        best = list(pms)
        best_score = refine_hlx(best, n, pts)
        for phs in np.linspace(0, 2 * np.pi, 36, endpoint=False):
            cand = [float(phs), v[0], v[1], v[2]]
            score = refine_hlx(cand, n, pts)
            if score < best_score:
                best = cand
                best_score = score

    hlx = make_hlx(n, best)
    hlx += np.mean(pts, 0)
    if np.linalg.norm(hlx[0] - pts[0]) > np.linalg.norm(hlx[0] - pts[-1]):
        hlx = hlx[::-1]
    new_pts = np.vstack([points[:i0], hlx, points[i1 + 1:]])

    removed = set(range(i0, i1 + 1))
    shift = len(hlx) - len(sel)
    new_fixed = []
    for a, b in fixed_edges:
        if a in removed or b in removed:
            continue
        na = a if a < i0 else a + shift
        nb = b if b < i0 else b + shift
        new_fixed.append(tuple(sorted((na, nb))))
    for k in range(len(hlx) - 1):
        new_fixed.append((i0 + k, i0 + k + 1))
    return new_pts, sorted(set(new_fixed))
