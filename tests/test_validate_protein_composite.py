"""Protein-only filtering and composite B-factor export for validate."""
from __future__ import annotations

from pathlib import Path

import gemmi
import numpy as np
import pytest

from cryomodel.io.structure_filter import filter_protein_only, is_tabulated_amino_acid_residue
from cryomodel.validation.composite_score import (
    add_composite_quality_columns,
    resolve_bfactor_column,
    write_structure_with_composite_bfactors,
)
from cryomodel.validation.feature_extractor import extract_residue_features
from cryomodel.io.mrc import MapVolume

FIX = Path(__file__).resolve().parent / "fixtures"


def _tiny_with_water() -> gemmi.Structure:
    st = gemmi.read_structure(str(FIX / "tiny_peptide.pdb"))
    w = gemmi.Residue()
    w.name = "HOH"
    w.seqid = gemmi.SeqId("100")
    o = gemmi.Atom()
    o.name = "O"
    o.element = gemmi.Element("O")
    o.pos = gemmi.Position(1.0, 1.0, 1.0)
    w.add_atom(o)
    st[0][0].add_residue(w)
    return st


def test_is_tabulated_amino_acid_residue() -> None:
    st = gemmi.read_structure(str(FIX / "tiny_peptide.pdb"))
    ala = list(st[0][0])[0]
    assert is_tabulated_amino_acid_residue(ala)
    w = gemmi.Residue()
    w.name = "HOH"
    w.seqid = gemmi.SeqId("1")
    assert not is_tabulated_amino_acid_residue(w)


def test_filter_protein_only_drops_water() -> None:
    st = _tiny_with_water()
    assert len(list(st[0][0])) == 4
    st2 = filter_protein_only(st)
    assert len(list(st2[0][0])) == 3


def test_composite_columns_and_bfactors_pdb(tmp_path: Path) -> None:
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
    assert "seqid" in df.columns
    assert "molprobity_clashscore" in df.columns
    assert "molprobity_clash_pairs" in df.columns
    df2 = add_composite_quality_columns(df)
    assert "composite_quality_0_100" in df2.columns
    assert df2["composite_quality_0_100"].between(0.0, 100.0).all()
    assert "composite_badness_0_100" in df2.columns
    assert (df2["composite_quality_0_100"] + df2["composite_badness_0_100"]).between(99.99, 100.01).all()
    assert df2["composite_band_deviation_0_100"].isna().all()
    assert resolve_bfactor_column(df2, "auto") == "composite_badness_0_100"

    out = tmp_path / "colored.pdb"
    write_structure_with_composite_bfactors(
        st, df2, out, column="composite_badness_0_100", threshold=50.0, higher_is_worse=True
    )
    st2 = gemmi.read_structure(str(out))
    ca0 = next(a for a in list(st2[0][0])[0] if a.name == "CA")
    v_bad = float(
        df2.loc[(df2["chain"] == st2[0][0].name) & (df2["seqid"] == "1"), "composite_badness_0_100"].iloc[0]
    )
    assert abs(float(ca0.b_iso) - v_bad) < 0.05
    assert float(ca0.occ) == (1.0 if v_bad <= 50.0 else 0.35)
