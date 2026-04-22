"""Tests for MolProbity-class lite geometry (validation)."""
from __future__ import annotations

import math
from pathlib import Path

import gemmi
import numpy as np
import pytest

from cryomodel.validation.feature_extractor import extract_residue_features
from cryomodel.validation.geometry_priors import (
    build_heavy_atom_clash_context,
    compute_geometry_features,
    compute_global_clash_z_stats,
    steric_clash_counts_by_residue,
)
from cryomodel.io.mrc import MapVolume


FIX = Path(__file__).resolve().parent / "fixtures"


def test_heavy_atom_clash_context_uses_gemmi_on_tiny_peptide() -> None:
    st = gemmi.read_structure(str(FIX / "tiny_peptide.pdb"))
    ctx = build_heavy_atom_clash_context(st)
    assert ctx.bond_topology == "gemmi"
    n_edges = sum(len(nbrs) for nbrs in ctx.adj) // 2
    assert n_edges >= 12


def test_steric_clash_counts_runs_on_tiny_peptide() -> None:
    st = gemmi.read_structure(str(FIX / "tiny_peptide.pdb"))
    c = steric_clash_counts_by_residue(st)
    mu, sd = compute_global_clash_z_stats(st)
    assert isinstance(c, dict)
    assert math.isfinite(mu)
    assert math.isnan(sd) or sd > 0


def test_geometry_features_use_phi_psi_rama() -> None:
    st = gemmi.read_structure(str(FIX / "tiny_peptide.pdb"))
    chain = st[0][0]
    residues = list(chain)
    c = steric_clash_counts_by_residue(st)
    mu, sd = compute_global_clash_z_stats(st)
    for i, res in enumerate(residues):
        if not any(a.name == "CA" for a in res):
            continue
        g = compute_geometry_features(res, residues, i, chain.name, c, mu, sd)
        assert "ramachandran_prob" in g
        assert 0.0 <= g["ramachandran_prob"] <= 1.0
        assert "clashscore_z" in g
        assert math.isfinite(g["clashscore_z"]) or math.isnan(g["clashscore_z"])
        assert "rotamer_prob" in g
        assert "omega_dev_deg" in g


def test_extract_residue_features_smoke_with_map() -> None:
    st = gemmi.read_structure(str(FIX / "tiny_peptide.pdb"))
    nz = ny = nx = 32
    data = np.zeros((nz, ny, nx), dtype=np.float32)
    data[16, 16, 16] = 1.0
    mv = MapVolume(
        data_zyx=data,
        apix=1.0,
        origin_xyzA=np.zeros(3, dtype=np.float32),
        grid=None,
    )
    df = extract_residue_features(st, mv)
    assert len(df) >= 1
    assert "rama_outlier" in df.columns
    assert "steric_clashes" in df.columns
