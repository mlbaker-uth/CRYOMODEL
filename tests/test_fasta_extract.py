"""Tests for cryomodel fasta-extract."""
from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from cryomodel.cli.fasta_extract import app

FIX = Path(__file__).resolve().parent / "fixtures"


def test_fasta_extract_row_writes_gap_stripped_sequence():
    out = FIX / "_tmp_extract.fasta"
    try:
        r = CliRunner().invoke(
            app,
            [str(FIX / "mutate_align_three_records.fasta"), str(out), "--row", "2"],
        )
        assert r.exit_code == 0, r.output
        text = out.read_text()
        assert ">target" in text or text.startswith(">")
        assert "XXMV" in text.replace("\n", "")
        assert "-" not in text
    finally:
        if out.exists():
            out.unlink()
