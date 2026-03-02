# crymodel/ml/featureizer.py
"""Feature extraction for water/ion classification."""
from __future__ import annotations
import numpy as np
import pandas as pd
from collections import Counter
from scipy.spatial import cKDTree
from typing import Optional

from .coordination import coordination_number, coordination_geometry

AA_ACID_NAMES = {"ASP", "GLU"}
AA_POLAR_NAMES = {"ASN", "GLN", "HIS"}
BACKBONE_O_NAMES = {"O", "OXT"}
HIS_N_NAMES = {"ND1", "NE2"}


def _angle(u: np.ndarray, v: np.ndarray) -> float:
    """Compute angle between two vectors in degrees."""
    nu = np.linalg.norm(u)
    nv = np.linalg.norm(v)
    if nu < 1e-6 or nv < 1e-6:
        return np.nan
    c = np.clip(np.dot(u, v) / (nu * nv), -1.0, 1.0)
    return np.degrees(np.arccos(c))


def build_kdtree(atom_df: pd.DataFrame) -> tuple[cKDTree, np.ndarray]:
    """Build KD-tree and return coordinates array from atom DataFrame."""
    coords = atom_df[["x", "y", "z"]].to_numpy(np.float32)
    return cKDTree(coords), coords


def element_mask(atom_df: pd.DataFrame, elems: set[str]) -> np.ndarray:
    """Return boolean mask for atoms matching element names."""
    return atom_df["element"].isin(elems).to_numpy()


def name_mask(atom_df: pd.DataFrame, names: set[str]) -> np.ndarray:
    """Return boolean mask for atoms matching atom names."""
    return atom_df["name"].isin(names).to_numpy()


def resname_mask(atom_df: pd.DataFrame, resnames: set[str]) -> np.ndarray:
    """Return boolean mask for atoms matching residue names."""
    return atom_df["resname"].isin(resnames).to_numpy()


def nearest_dists(point: np.ndarray, coords: np.ndarray, mask: np.ndarray, k: int = 3) -> list[float]:
    """Return k nearest distances from point to masked coordinates."""
    idx = np.where(mask)[0]
    if idx.size == 0:
        return [np.inf] * k
    sub = coords[idx]
    if sub.shape[0] < k:
        k = sub.shape[0]
    d = np.linalg.norm(sub - point, axis=1)
    d_sorted = np.sort(d)[:k].tolist()
    d_sorted += [np.inf] * (3 - len(d_sorted))
    return d_sorted


def counts_in_shells(
    point: np.ndarray, coords: np.ndarray, mask: np.ndarray, edges: tuple[float, float, float, float] = (2.1, 2.5, 3.0, 3.4)
) -> tuple[int, int, int]:
    """Count atoms in distance shells: [a,b), (b,c], (c,dmax]."""
    idx = np.where(mask)[0]
    if idx.size == 0:
        return (0, 0, 0)
    d = np.linalg.norm(coords[idx] - point, axis=1)
    a, b, c, dmax = edges
    return (
        int(np.sum((d >= a) & (d <= b))),
        int(np.sum((d > b) & (d <= c))),
        int(np.sum((d > c) & (d <= dmax))),
    )


def angle_variance(point: np.ndarray, coords: np.ndarray, mask: np.ndarray, k: int = 6) -> float:
    """Compute variance of angles between vectors to k nearest neighbors."""
    idx = np.where(mask)[0]
    if idx.size < 3:
        return np.nan
    sub = coords[idx]
    # take nearest K neighbors
    d = np.linalg.norm(sub - point, axis=1)
    pick = np.argsort(d)[: min(k, sub.shape[0])]
    vecs = sub[pick] - point
    angs = []
    for i in range(len(vecs)):
        for j in range(i + 1, len(vecs)):
            a = _angle(vecs[i], vecs[j])
            if not np.isnan(a):
                angs.append(a)
    return float(np.var(angs)) if angs else np.nan


def residue_tallies(point: np.ndarray, atom_df: pd.DataFrame, coords: np.ndarray, radius: float = 4.0) -> tuple[int, int, int]:
    """Count acidic, polar residues, and total atoms within radius."""
    d = np.linalg.norm(coords - point, axis=1)
    within = atom_df.loc[d <= radius]
    tallies = Counter(within["resname"])
    acids = sum(tallies[r] for r in AA_ACID_NAMES if r in tallies)
    polars = sum(tallies[r] for r in AA_POLAR_NAMES if r in tallies)
    return acids, polars, int(len(within))


def carboxylate_features(point: np.ndarray, atom_df: pd.DataFrame, coords: np.ndarray) -> tuple[int, float]:
    """Count carboxylate O atoms in 2.4-2.6Å shell and minimum distance."""
    is_acid = resname_mask(atom_df, AA_ACID_NAMES)
    is_oxy = element_mask(atom_df, {"O"})
    idx = np.where(is_acid & is_oxy)[0]
    if idx.size == 0:
        return 0, np.inf
    sub = coords[idx]
    d = np.linalg.norm(sub - point, axis=1)
    count_24_26 = int(np.sum((d >= 2.4) & (d <= 2.6)))
    return count_24_26, float(np.min(d))


def his_counts(point: np.ndarray, atom_df: pd.DataFrame, coords: np.ndarray, rmax: float = 3.4) -> int:
    """Count histidine N atoms within rmax."""
    is_his = resname_mask(atom_df, {"HIS"})
    is_hisN = name_mask(atom_df, HIS_N_NAMES)
    idx = np.where(is_his & is_hisN)[0]
    if idx.size == 0:
        return 0
    d = np.linalg.norm(coords[idx] - point, axis=1)
    return int(np.sum(d <= rmax))


def backbone_O_count(point: np.ndarray, atom_df: pd.DataFrame, coords: np.ndarray, rmax: float = 3.2) -> int:
    """Count backbone O atoms within rmax."""
    is_O = name_mask(atom_df, BACKBONE_O_NAMES)
    idx = np.where(is_O)[0]
    if idx.size == 0:
        return 0
    d = np.linalg.norm(coords[idx] - point, axis=1)
    return int(np.sum(d <= rmax))


def build_feature_row(
    row: pd.Series | dict,
    atom_df: pd.DataFrame,
    kdtree: cKDTree,
    coords: np.ndarray,
    extra_cols: Optional[list[str]] = None,
    entry_resolution: Optional[float] = None,
) -> dict:
    """Build a feature dictionary for a single candidate point."""
    px = np.array([row["center_x"], row["center_y"], row["center_z"]], dtype=np.float32)

    is_O = element_mask(atom_df, {"O"})
    is_N = element_mask(atom_df, {"N"})
    is_S = element_mask(atom_df, {"S"})
    # nearest distances
    dO1, dO2, dO3 = nearest_dists(px, coords, is_O, k=3)
    dN1 = nearest_dists(px, coords, is_N, k=1)[0]
    dS_dists = nearest_dists(px, coords, is_S, k=2)
    dS1, dS2 = dS_dists[0], dS_dists[1]  # Take first 2 values (function always returns 3)

    # shells
    O_21_25, O_25_30, O_30_34 = counts_in_shells(px, coords, is_O)
    N_25_34 = sum(counts_in_shells(px, coords, is_N, edges=(2.5, 3.0, 3.4, 3.4)))

    # geometry/chemistry
    ang_var = angle_variance(px, coords, (is_O | is_N))
    carb_cnt, carb_min = carboxylate_features(px, atom_df, coords)
    hisN_cnt = his_counts(px, atom_df, coords)
    bbO_cnt = backbone_O_count(px, atom_df, coords)

    acid_tally, polar_tally, neighbors_total = residue_tallies(px, atom_df, coords)

    # Coordination geometry features (important for ion classification)
    is_O_or_N = is_O | is_N
    coord_feats = coordination_geometry(px, coords, is_O_or_N, cutoff_A=3.5, k=6)
    
    # Coordination number with just O atoms (most common ligand)
    coord_O = coordination_number(px, coords, is_O, cutoff_A=3.5)
    
    # Coordination number with S atoms (important for Zn, Fe)
    coord_S = coordination_number(px, coords, is_S, cutoff_A=3.5)

    # Ratio features (help distinguish similar ions)
    dO1_dO2_ratio = dO1 / max(dO2, 0.1) if dO2 > 0 and not np.isinf(dO2) else np.nan
    coord_O_ratio = coord_O / max(coord_feats["coord_number"], 1) if coord_feats["coord_number"] > 0 else 0.0
    coord_S_ratio = coord_S / max(coord_feats["coord_number"], 1) if coord_feats["coord_number"] > 0 else 0.0
    
    feats = {
        "dO1": dO1,
        "dO2": dO2,
        "dO3": dO3,
        "dN1": dN1,
        "dS1": dS1,
        "dS2": dS2,
        "O_21_25": O_21_25,
        "O_25_30": O_25_30,
        "O_30_34": O_30_34,
        "N_25_34": N_25_34,
        "ang_var": ang_var,
        "carbox_cnt_24_26": carb_cnt,
        "carbox_min": carb_min,
        "hisN_cnt": hisN_cnt,
        "bbO_cnt": bbO_cnt,
        "acid_tally": acid_tally,
        "polar_tally": polar_tally,
        "neighbors_total": neighbors_total,
        "entry_res": float(entry_resolution) if entry_resolution is not None else np.nan,
        # Coordination features
        "coord_number": coord_feats["coord_number"],
        "coord_O": coord_O,
        "coord_S": coord_S,
        "coord_mean_dist": coord_feats["coord_mean_dist"],
        "coord_angle_mean": coord_feats["coord_angle_mean"],
        "coord_tetrahedral_score": coord_feats["coord_tetrahedral_score"],
        "coord_octahedral_score": coord_feats["coord_octahedral_score"],
        # Ratio features (help distinguish similar ions)
        "dO1_dO2_ratio": dO1_dO2_ratio,
        "coord_O_ratio": coord_O_ratio,
        "coord_S_ratio": coord_S_ratio,
    }
    # include precomputed shape stats if present
    if extra_cols:
        for c in extra_cols:
            feats[c] = row.get(c, np.nan)
    return feats

