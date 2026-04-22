"""Resolve overlapping subunit masks into a single label volume (one ID per voxel)."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np
from scipy import ndimage

from cryomodel.io.mrc import MapVolume, read_map, write_map


@dataclass
class HelicalOverlapResolveResult:
    output_json: str
    labels_map: str
    representative_map: Optional[str]
    n_masks: int
    n_overlap_voxels: int
    n_labels_assigned: int
    tie_break: str

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_binary_masks_to_labels(
    density_zyx: np.ndarray,
    masks_zyx: list[np.ndarray],
    *,
    tie_break: Literal["density", "mask_order"] = "density",
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Build an integer label map (1..N) from N binary masks that may overlap.

    Unambiguous voxels (exactly one mask positive) get that mask's label.
    Overlaps are resolved by highest density at the voxel among claiming masks;
    if ``tie_break`` is ``mask_order``, earlier masks win on exact density ties.
    """
    if len(masks_zyx) < 1:
        raise ValueError("At least one mask is required.")
    shape = density_zyx.shape
    for i, m in enumerate(masks_zyx):
        if tuple(m.shape) != tuple(shape):
            raise ValueError(f"Mask {i} shape {m.shape} != density shape {shape}")

    dens = np.asarray(density_zyx, dtype=np.float64)
    n = len(masks_zyx)
    stack = np.stack([np.asarray(m, dtype=np.float64) > 0.0 for m in masks_zyx], axis=0)
    count = stack.sum(axis=0).astype(np.int32)
    n_overlap = int(np.count_nonzero(count > 1))

    out = np.zeros(shape, dtype=np.int32)
    one = count == 1
    for i in range(n):
        out = np.where(one & stack[i], np.int32(i + 1), out)

    multi = count > 1
    if np.any(multi):
        if tie_break == "density":
            scores = np.stack([np.where(stack[i], dens, -np.inf) for i in range(n)], axis=0)
        else:
            eps = np.finfo(np.float64).eps
            scores = np.stack(
                [np.where(stack[i], dens + eps * (n - i), -np.inf) for i in range(n)],
                axis=0,
            )
        win = np.argmax(scores, axis=0).astype(np.int32)
        out = np.where(multi, win + 1, out)

    n_assigned = int(np.count_nonzero(out > 0))
    meta = {
        "n_overlap_voxels": n_overlap,
        "n_voxels_assigned": n_assigned,
        "tie_break": tie_break,
    }
    return out, meta


def _representative_from_labels(
    data_zyx: np.ndarray,
    labels_zyx: np.ndarray,
    *,
    label_id: int,
    largest_component: bool,
) -> np.ndarray:
    rep = np.zeros_like(data_zyx, dtype=np.float32)
    m = labels_zyx == int(label_id)
    if not np.any(m):
        return rep
    if largest_component:
        cc, ncc = ndimage.label(m, structure=np.ones((3, 3, 3), dtype=bool))
        if ncc > 1:
            sizes = np.bincount(cc.ravel())
            sizes[0] = 0
            keep = int(np.argmax(sizes))
            m = cc == keep
    rep[m] = data_zyx[m].astype(np.float32)
    return rep


def run_helical_resolve_overlaps(
    map_path: Path,
    mask_paths: list[Path],
    out_dir: Path,
    *,
    tie_break: Literal["density", "mask_order"] = "density",
    write_representative: bool = False,
    representative_label: Optional[int] = None,
    representative_largest_component: bool = False,
) -> HelicalOverlapResolveResult:
    """
    Read a density map and N mask maps (positive = claimed), write a single label MRC
    with one positive integer per voxel (0 = background).

    Overlapping claims are resolved so each voxel has at most one label.
    """
    map_path = Path(map_path).expanduser().resolve()
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if len(mask_paths) < 1:
        raise ValueError("Provide at least one mask MRC path.")

    mv = read_map(map_path)
    dens = mv.data_zyx.astype(np.float32)
    masks: list[np.ndarray] = []
    for p in mask_paths:
        mp = Path(p).expanduser().resolve()
        mv_m = read_map(mp)
        if tuple(mv_m.data_zyx.shape) != tuple(dens.shape):
            raise ValueError(f"Mask {mp} shape {mv_m.data_zyx.shape} != map {dens.shape}")
        if abs(float(mv_m.apix) - float(mv.apix)) > 1e-3:
            raise ValueError(f"Mask {mp} apix {mv_m.apix} != map apix {mv.apix}")
        masks.append(mv_m.data_zyx)

    labels, meta = resolve_binary_masks_to_labels(dens, masks, tie_break=tie_break)
    labels_path = out_dir / "helical_overlap_labels.mrc"
    write_map(labels_path, mv, labels.astype(np.float32))

    rep_path: Optional[str] = None
    rep_lid: Optional[int] = None
    if write_representative:
        if representative_label is None:
            if int(np.max(labels)) <= 0:
                raise ValueError("No positive labels; cannot pick representative.")
            lbls = labels[labels > 0].ravel().astype(np.int64)
            mx = int(np.max(lbls))
            cnt = np.bincount(lbls, minlength=mx + 1)
            rep_lid = int(np.argmax(cnt[1:]) + 1)
        else:
            rep_lid = int(representative_label)
        rep = _representative_from_labels(
            dens,
            labels,
            label_id=rep_lid,
            largest_component=representative_largest_component,
        )
        rp = out_dir / "helical_overlap_representative.mrc"
        write_map(rp, mv, rep)
        rep_path = str(rp)

    out_json = out_dir / "helical_overlap_resolve.json"
    result = HelicalOverlapResolveResult(
        output_json=str(out_json),
        labels_map=str(labels_path),
        representative_map=rep_path,
        n_masks=len(mask_paths),
        n_overlap_voxels=int(meta["n_overlap_voxels"]),
        n_labels_assigned=int(meta["n_voxels_assigned"]),
        tie_break=str(tie_break),
    )
    payload = result.to_json_dict()
    payload["input_map"] = str(map_path)
    payload["mask_paths"] = [str(Path(p).expanduser().resolve()) for p in mask_paths]
    payload["meta"] = meta
    payload["representative_label_id"] = rep_lid
    payload["representative_largest_component"] = bool(representative_largest_component)
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return result
