"""Per-residue validate report for Coot / manual review."""
from __future__ import annotations

from pathlib import Path

import gemmi
import numpy as np
import pytest

from cryomodel.io.mrc import MapVolume
from cryomodel.validation.composite_score import add_composite_quality_columns
from cryomodel.validation.feature_extractor import extract_residue_features
from cryomodel.validation.residue_report import (
    build_residue_report_table,
    format_residue_report_line,
    write_residue_report,
)

FIX = Path(__file__).resolve().parent / "fixtures"


def test_residue_report_txt_and_csv(tmp_path: Path) -> None:
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
    df = add_composite_quality_columns(extract_residue_features(st, mv))
    csv_p, txt_p = write_residue_report(df, tmp_path)
    assert csv_p.is_file()
    assert txt_p.is_file()
    text = txt_p.read_text(encoding="utf-8")
    assert "validate score:" in text
    assert "rama:" in text
    assert "rotamer:" in text
    assert "Q-score:" in text
    table = build_residue_report_table(df)
    assert "validate_score" in table.columns
    assert len(table) == len(df)
    line = format_residue_report_line(table.iloc[0])
    assert "validate score:" in line


def test_residue_report_rama_class_column() -> None:
    st = gemmi.read_structure(str(FIX / "tiny_peptide.pdb"))
    nz = ny = nx = 32
    data = np.zeros((nz, ny, nx), dtype=np.float32)
    mv = MapVolume(
        data_zyx=data,
        apix=1.0,
        origin_xyzA=np.zeros(3, dtype=np.float32),
        grid=None,
    )
    df = extract_residue_features(st, mv)
    assert "rama_class" in df.columns
