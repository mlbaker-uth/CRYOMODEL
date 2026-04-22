"""Tests for zonal χ1 refinement."""
from __future__ import annotations

from pathlib import Path

import gemmi
import numpy as np
import pytest
from cryomodel.io.mrc import MapVolume
from cryomodel.zonal import (
    parse_center_xyz,
    partition_hard_soft_spherical,
    residues_in_sphere,
    run_zonal_chi_refine,
)
from cryomodel.mutate.chi import map_fit_anchor_term
from cryomodel.validation.ringer_lite import sample_density_at_position
from cryomodel.zonal.ramachandran import classify_phi_psi_general

FIX = Path(__file__).resolve().parent / "fixtures"


def test_parse_center_xyz():
    c = parse_center_xyz("10, 20 , 30.5")
    assert c.shape == (3,)
    assert list(c) == [10.0, 20.0, 30.5]


def test_residues_in_sphere_tiny_peptide():
    st = gemmi.read_structure(str(FIX / "tiny_peptide.pdb"))
    center = np.array([13.0, 19.0, 25.0], dtype=np.float64)
    z = residues_in_sphere(st, center, 10.0)
    assert len(z) == 3
    chains = {cid for cid, _ in z}
    assert chains == {"A"}


def test_run_zonal_chi_refine_smoke():
    st = gemmi.read_structure(str(FIX / "tiny_peptide.pdb"))
    # Uniform map covering the peptide (origin 0, apix 1 Å)
    mv = MapVolume(
        data_zyx=np.ones((40, 40, 40), dtype=np.float32),
        apix=1.0,
        origin_xyzA=np.array([0.0, 0.0, 0.0], dtype=np.float32),
    )
    center = np.array([13.0, 19.0, 25.0], dtype=np.float64)
    r = run_zonal_chi_refine(
        st,
        mv,
        center,
        12.0,
        passes=2,
        weight_map=0.5,
        weight_rot=0.15,
    )
    # ALA/GLY have no χ1 in our rotamer model; SER does
    assert r.residues_in_zone == 3
    assert r.residues_with_chi1 == 1
    assert r.passes_done >= 1
    assert r.elapsed_sec >= 0


def test_partition_hard_soft_tiny_peptide():
    """Tight hard sphere around ALA only; GLY+SER in soft shell (outer minus hard)."""
    st = gemmi.read_structure(str(FIX / "tiny_peptide.pdb"))
    center = np.array([19.03, 17.825, 21.748], dtype=np.float64)
    # 1.5 Å: only ALA atoms; 3 Å would include GLY (backbone N ~2.4 Å from ALA CA).
    hard, soft = partition_hard_soft_spherical(st, center, 1.5, 20.0)
    assert len(hard) == 1
    assert len(soft) == 2


def test_run_zonal_soft_buffer_smoke():
    st = gemmi.read_structure(str(FIX / "tiny_peptide.pdb"))
    mv = MapVolume(
        data_zyx=np.ones((40, 40, 40), dtype=np.float32),
        apix=1.0,
        origin_xyzA=np.array([0.0, 0.0, 0.0], dtype=np.float32),
    )
    center = np.array([13.0, 19.0, 25.0], dtype=np.float64)
    r = run_zonal_chi_refine(
        st,
        mv,
        center,
        12.0,
        passes=2,
        weight_map=0.5,
        weight_rot=0.15,
        soft_buffer=5.0,
        soft_passes=1,
        soft_min_clash=0.0,
        soft_only_if_worsened=False,
    )
    assert r.residues_soft_zone >= 0
    assert r.meta["soft_buffer"] == 5.0


def test_map_fit_anchor_term_penalty_and_bonus():
    # Was "in" density: penalize drop
    t = map_fit_anchor_term(0.5, 1.0, weight_anchor=2.0, weight_gain=1.0, eps=0.01)
    assert t == pytest.approx(2.0 * 0.5)
    # No penalty if improved
    assert map_fit_anchor_term(1.2, 1.0, weight_anchor=2.0, weight_gain=1.0, eps=0.01) == 0.0
    # Was weak: bonus for gain, no penalty for loss
    assert map_fit_anchor_term(0.3, 0.0, weight_anchor=2.0, weight_gain=1.0, eps=0.01) == pytest.approx(-0.3)
    assert map_fit_anchor_term(0.0, 0.0, weight_anchor=2.0, weight_gain=1.0, eps=0.01) == 0.0
    assert map_fit_anchor_term(0.0, 0.0, weight_anchor=0.0, weight_gain=0.0, eps=0.01) == 0.0


def test_map_density_threshold_reduces_sample():
    mv = MapVolume(
        data_zyx=np.ones((40, 40, 40), dtype=np.float32),
        apix=1.0,
        origin_xyzA=np.array([0.0, 0.0, 0.0], dtype=np.float32),
    )
    p = np.array([5.0, 5.0, 5.0], dtype=np.float64)
    assert sample_density_at_position(mv, p) == 1.0
    assert sample_density_at_position(mv, p, density_threshold=0.3) == pytest.approx(0.7)
    assert sample_density_at_position(mv, p, density_threshold=1.0) == 0.0


def test_rama_classify_helix_like_favored():
    assert classify_phi_psi_general(-60.0, -45.0) == "favored"


def test_run_zonal_rama_backbone_smoke():
    st = gemmi.read_structure(str(FIX / "tiny_peptide.pdb"))
    mv = MapVolume(
        data_zyx=np.ones((40, 40, 40), dtype=np.float32),
        apix=1.0,
        origin_xyzA=np.array([0.0, 0.0, 0.0], dtype=np.float32),
    )
    center = np.array([13.0, 19.0, 25.0], dtype=np.float64)
    r = run_zonal_chi_refine(
        st,
        mv,
        center,
        12.0,
        passes=1,
        weight_map=0.5,
        weight_rot=0.15,
        rama_backbone=True,
        rama_step_deg=6.0,
        rama_max_shift_deg=6.0,
        weight_rama=0.05,
        weight_bb_move=0.02,
    )
    assert r.meta["rama_backbone"] is True
    assert r.rama_residues_tried >= 0


def test_chain_filter_excludes():
    st = gemmi.read_structure(str(FIX / "tiny_peptide.pdb"))
    center = np.array([13.0, 19.0, 25.0], dtype=np.float64)
    z = residues_in_sphere(st, center, 10.0, chain_filter={"B"})
    assert len(z) == 0
