"""Tests for global zonal refinement (GMM zones + NCS χ1 propagation)."""
from __future__ import annotations

from pathlib import Path

import gemmi
import numpy as np
import pytest

from cryomodel.io.mrc import MapVolume
from cryomodel.mutate.chi import chi1_dihedral_deg, chi1_quadruple, rotate_sidechain_chi1
from cryomodel.zonal.global_refine import (
    build_overlapping_gmm_regions,
    parse_ncs_chains,
    propagate_chi1_ncs,
    run_global_zonal_refine,
)

FIX = Path(__file__).resolve().parent / "fixtures"


def test_parse_ncs_chains():
    assert parse_ncs_chains("A") == ("A", [])
    assert parse_ncs_chains(" A , B , C ") == ("A", ["B", "C"])
    with pytest.raises(ValueError):
        parse_ncs_chains("")


def test_build_overlapping_gmm_regions_covers_all():
    rng = np.random.RandomState(42)
    X = rng.randn(24, 3) * 5.0 + np.array([10.0, 10.0, 10.0])
    regions = build_overlapping_gmm_regions(
        X,
        target_residues_per_region=8,
        soft_resp_floor=0.08,
        random_state=0,
        reg_covar=1e-3,
    )
    assert regions
    covered = set()
    for s in regions:
        covered |= s
    assert covered == set(range(24))


def test_build_overlapping_gmm_regions_explicit_component_count():
    rng = np.random.RandomState(1)
    X = rng.randn(30, 3) * 8.0
    regions = build_overlapping_gmm_regions(
        X,
        target_residues_per_region=99,
        n_components=5,
        soft_resp_floor=0.1,
        random_state=2,
        reg_covar=1e-3,
    )
    assert len(regions) == 5


def _homodimer_shifted(tmp_path: Path, dx: float = 45.0) -> Path:
    st = gemmi.read_structure(str(FIX / "tiny_peptide.pdb"))
    ch_b = gemmi.Chain("B")
    for res in st[0]["A"]:
        r = res.clone()
        for atom in r:
            p = atom.pos
            atom.pos = gemmi.Position(p.x + dx, p.y, p.z)
        ch_b.add_residue(r)
    st[0].add_chain(ch_b)
    out = tmp_path / "dimer.pdb"
    st.write_pdb(str(out))
    return out


def test_run_global_zonal_smoke_single_chain(tmp_path: Path):
    st = gemmi.read_structure(str(FIX / "tiny_peptide.pdb"))
    mv = MapVolume(
        data_zyx=np.ones((40, 40, 40), dtype=np.float32),
        apix=1.0,
        origin_xyzA=np.array([0.0, 0.0, 0.0], dtype=np.float32),
    )
    r = run_global_zonal_refine(
        st,
        mv,
        pdb_path=FIX / "tiny_peptide.pdb",
        master_chain="A",
        copy_chains=[],
        max_rounds=1,
        converge_rmsd_eps=1e-9,
        converge_patience=99,
        random_seed=0,
        passes=1,
        weight_map=0.5,
        weight_rot=0.15,
    )
    assert r.rounds_done == 1
    assert r.region_count >= 1
    assert r.stopped_reason == "max_rounds"


def test_propagate_chi1_ncs_matches_master(tmp_path: Path):
    pdb = _homodimer_shifted(tmp_path)
    st = gemmi.read_structure(str(pdb))
    ser_a = next(r for r in st[0]["A"] if r.name == "SER")
    ser_b = next(r for r in st[0]["B"] if r.name == "SER")
    quad = chi1_quadruple("SER", ser_a)
    assert quad is not None
    rotate_sidechain_chi1(ser_b, quad, 55.0)

    before_m = chi1_dihedral_deg(ser_a, quad)
    before_b = chi1_dihedral_deg(ser_b, quad)
    assert abs(before_m - before_b) > 1.0

    propagate_chi1_ncs(
        st,
        master_chain="A",
        copy_chains=["B"],
        master_residues_in_zone=[ser_a],
    )
    quad_b = chi1_quadruple("SER", ser_b)
    assert quad_b is not None
    after_b = chi1_dihedral_deg(ser_b, quad_b)
    assert after_b == pytest.approx(before_m, abs=0.2)


def test_run_global_zonal_with_ncs_copy(tmp_path: Path):
    pdb = _homodimer_shifted(tmp_path)
    st = gemmi.read_structure(str(pdb))
    ser_b = next(r for r in st[0]["B"] if r.name == "SER")
    quad = chi1_quadruple("SER", ser_b)
    assert quad is not None
    rotate_sidechain_chi1(ser_b, quad, 40.0)

    mv = MapVolume(
        data_zyx=np.ones((120, 120, 120), dtype=np.float32),
        apix=1.0,
        origin_xyzA=np.array([0.0, 0.0, 0.0], dtype=np.float32),
    )
    r = run_global_zonal_refine(
        st,
        mv,
        pdb_path=pdb,
        master_chain="A",
        copy_chains=["B"],
        max_rounds=1,
        converge_rmsd_eps=1e-9,
        converge_patience=99,
        random_seed=1,
        target_residues_per_region=10,
        passes=1,
        weight_map=0.3,
        weight_rot=0.1,
    )
    assert r.rounds_done == 1
    ser_a = next(r for r in st[0]["A"] if r.name == "SER")
    quad_a = chi1_quadruple("SER", ser_a)
    quad_b = chi1_quadruple("SER", ser_b)
    assert quad_a and quad_b
    assert chi1_dihedral_deg(ser_b, quad_b) == pytest.approx(chi1_dihedral_deg(ser_a, quad_a), abs=0.5)
