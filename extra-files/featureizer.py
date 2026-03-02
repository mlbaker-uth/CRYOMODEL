import numpy as np
import pandas as pd
from collections import Counter
from scipy.spatial import cKDTree

AA_ACID_NAMES = {"ASP","GLU"}
AA_POLAR_NAMES = {"ASN","GLN","HIS"}
BACKBONE_O_NAMES = {"O","OXT"}
HIS_N_NAMES = {"ND1","NE2"}

def _angle(u, v):
    nu = np.linalg.norm(u); nv = np.linalg.norm(v)
    if nu < 1e-6 or nv < 1e-6: return np.nan
    c = np.clip(np.dot(u, v) / (nu*nv), -1.0, 1.0)
    return np.degrees(np.arccos(c))

def build_kdtree(atom_df: pd.DataFrame):
    coords = atom_df[["x","y","z"]].to_numpy(np.float32)
    return cKDTree(coords), coords

def element_mask(atom_df, elems):
    return atom_df["element"].isin(elems).to_numpy()

def name_mask(atom_df, names):
    return atom_df["name"].isin(names).to_numpy()

def resname_mask(atom_df, resnames):
    return atom_df["resname"].isin(resnames).to_numpy()

def nearest_dists(point, coords, mask, k=3):
    idx = np.where(mask)[0]
    if idx.size == 0:
        return [np.inf]*k
    sub = coords[idx]
    if sub.shape[0] < k:
        k = sub.shape[0]
    d = np.linalg.norm(sub - point, axis=1)
    d_sorted = np.sort(d)[:k].tolist()
    d_sorted += [np.inf] * (3 - len(d_sorted))
    return d_sorted

def counts_in_shells(point, coords, mask, edges=(2.1,2.5,3.0,3.4)):
    idx = np.where(mask)[0]
    if idx.size == 0:
        return (0,0,0)
    d = np.linalg.norm(coords[idx]-point, axis=1)
    a,b,c,dmax = edges
    return (int(np.sum((d>=a)&(d<=b))),
            int(np.sum((d>b)&(d<=c))),
            int(np.sum((d>c)&(d<=dmax))))

def angle_variance(point, coords, mask, k=6):
    idx = np.where(mask)[0]
    if idx.size < 3:
        return np.nan
    sub = coords[idx]
    # take nearest K neighbors
    d = np.linalg.norm(sub - point, axis=1)
    pick = np.argsort(d)[:min(k, sub.shape[0])]
    vecs = sub[pick] - point
    angs = []
    for i in range(len(vecs)):
        for j in range(i+1, len(vecs)):
            a = _angle(vecs[i], vecs[j])
            if not np.isnan(a):
                angs.append(a)
    return float(np.var(angs)) if angs else np.nan

def residue_tallies(point, atom_df, coords, radius=4.0):
    d = np.linalg.norm(coords - point, axis=1)
    within = atom_df.loc[d<=radius]
    tallies = Counter(within["resname"])
    acids = sum(tallies[r] for r in AA_ACID_NAMES if r in tallies)
    polars = sum(tallies[r] for r in AA_POLAR_NAMES if r in tallies)
    return acids, polars, int(len(within))

def carboxylate_features(point, atom_df, coords):
    is_acid = resname_mask(atom_df, AA_ACID_NAMES)
    is_oxy = element_mask(atom_df, {"O"})
    idx = np.where(is_acid & is_oxy)[0]
    if idx.size == 0:
        return 0, np.inf
    sub = coords[idx]
    d = np.linalg.norm(sub - point, axis=1)
    count_24_26 = int(np.sum((d>=2.4)&(d<=2.6)))
    return count_24_26, float(np.min(d))

def his_counts(point, atom_df, coords, rmax=3.4):
    is_his = resname_mask(atom_df, {"HIS"})
    is_hisN = name_mask(atom_df, HIS_N_NAMES)
    idx = np.where(is_his & is_hisN)[0]
    if idx.size == 0: return 0
    d = np.linalg.norm(coords[idx]-point, axis=1)
    return int(np.sum(d <= rmax))

def backbone_O_count(point, atom_df, coords, rmax=3.2):
    is_O = name_mask(atom_df, BACKBONE_O_NAMES)
    idx = np.where(is_O)[0]
    if idx.size == 0: return 0
    d = np.linalg.norm(coords[idx]-point, axis=1)
    return int(np.sum(d<=rmax))

def build_feature_row(row, atom_df, kdtree, coords, extra_cols=None, entry_resolution=None):
    px = np.array([row["x"], row["y"], row["z"]], dtype=np.float32)

    is_O = element_mask(atom_df, {"O"})
    is_N = element_mask(atom_df, {"N"})
    is_S = element_mask(atom_df, {"S"})
    # nearest distances
    dO1,dO2,dO3 = nearest_dists(px, coords, is_O, k=3)
    dN1, = nearest_dists(px, coords, is_N, k=1)[:1]
    dS1, = nearest_dists(px, coords, is_S, k=1)[:1]

    # shells
    O_21_25, O_25_30, O_30_34 = counts_in_shells(px, coords, is_O)
    N_25_34 = sum(counts_in_shells(px, coords, is_N, edges=(2.5,3.0,3.4,3.4)))

    # geometry/chemistry
    ang_var = angle_variance(px, coords, (is_O|is_N))
    carb_cnt, carb_min = carboxylate_features(px, atom_df, coords)
    hisN_cnt = his_counts(px, atom_df, coords)
    bbO_cnt = backbone_O_count(px, atom_df, coords)

    acid_tally, polar_tally, neighbors_total = residue_tallies(px, atom_df, coords)

    feats = {
        "dO1": dO1, "dO2": dO2, "dO3": dO3,
        "dN1": dN1, "dS1": dS1,
        "O_21_25": O_21_25, "O_25_30": O_25_30, "O_30_34": O_30_34,
        "N_25_34": N_25_34,
        "ang_var": ang_var,
        "carbox_cnt_24_26": carb_cnt, "carbox_min": carb_min,
        "hisN_cnt": hisN_cnt, "bbO_cnt": bbO_cnt,
        "acid_tally": acid_tally, "polar_tally": polar_tally,
        "neighbors_total": neighbors_total,
        "entry_res": float(entry_resolution) if entry_resolution is not None else np.nan,
    }
    # include your precomputed shape stats if present
    if extra_cols:
        for c in extra_cols:
            feats[c] = row.get(c, np.nan)
    return feats
