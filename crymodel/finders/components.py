# crymodel/finders/components.py
from __future__ import annotations
import numpy as np
from scipy.ndimage import label

def split_by_component_size(mask_bool: np.ndarray, vvox_max_micro: int):
    """
    Split a boolean mask into (micro_mask, ligand_mask) by connected-component size.
    micro_mask contains components with voxel-count <= vvox_max_micro.
    ligand_mask is the remainder.
    """
    assert mask_bool.dtype == bool
    if not mask_bool.any():
        return np.zeros_like(mask_bool), np.zeros_like(mask_bool)

    lab, nlab = label(mask_bool.astype(np.uint8))
    if nlab == 0:
        return np.zeros_like(mask_bool), np.zeros_like(mask_bool)

    counts = np.bincount(lab.ravel())
    micro_ids = set([i for i, c in enumerate(counts) if (i != 0 and c <= int(vvox_max_micro))])
    if not micro_ids:
        return np.zeros_like(mask_bool), mask_bool.copy()

    micro_mask = np.isin(lab, list(micro_ids))
    ligand_mask = mask_bool & (~micro_mask)
    return micro_mask, ligand_mask
