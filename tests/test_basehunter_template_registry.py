from __future__ import annotations

from pathlib import Path

from cryomodel.nucleotide.template_registry import parse_templates_txt, validate_template_pack


def test_parse_templates_txt_extracts_thresholds(tmp_path: Path) -> None:
    txt = tmp_path / "templates.txt"
    txt.write_text(
        "\n".join(
            [
                "templateBP-purine-3.mrc: averaged density map (64x64x64, ~2.0 for threshold)",
                "templateBP-A2-base.pdb: aligned model of A (base only)",
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "templateBP-purine-3.mrc").write_text("fake", encoding="utf-8")
    (tmp_path / "templateBP-A2-base.pdb").write_text("fake", encoding="utf-8")

    entries = parse_templates_txt(txt)
    assert len(entries) == 2
    assert entries[0].filename == "templateBP-purine-3.mrc"
    assert entries[0].threshold == 2.0
    assert entries[1].threshold is None


def test_validate_template_pack_reports_missing() -> None:
    root = Path("/Users/mbaker-local/Downloads/CRYOMODEL_LOCAL/NEW-DNA-TEMPLATES")
    res = validate_template_pack(root)
    assert len(res.entries) > 0
    assert "templateBP-purine-3.mrc" in {e.filename for e in res.entries}
    # This pack currently has metadata listing files not present on disk.
    assert len(res.missing_files) > 0
    assert "templateBP-pyrimidine.mrc" in res.missing_files

