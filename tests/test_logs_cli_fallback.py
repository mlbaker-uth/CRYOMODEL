from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from cryomodel.cli.logs import app


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(json.dumps(r) for r in rows) + "\n"
    path.write_text(text, encoding="utf-8")


def test_log_show_prefers_activity_over_legacy(tmp_path: Path, monkeypatch) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    _write_jsonl(
        project / ".cryomodel" / "activity.jsonl",
        [
            {
                "timestamp": "2026-04-06T10:00:00",
                "source": "workflow_ui",
                "card_id": "validate_card",
                "command": "cryomodel validate --map m.mrc --model x.pdb",
                "status": "success",
                "return_code": 0,
                "duration_s": 1.2,
            }
        ],
    )
    _write_jsonl(
        project / ".cryomodel_history.jsonl",
        [
            {
                "timestamp": "2026-04-06T09:59:00",
                "tool": "legacy_tool",
                "command": "cryomodel legacy",
                "status": "error",
                "duration_s": 0.1,
            }
        ],
    )
    monkeypatch.chdir(tmp_path)
    r = CliRunner().invoke(app, ["show", "--cwd", str(project), "--limit", "10"])
    assert r.exit_code == 0, r.output
    assert "validate_card" in r.output
    assert "legacy_tool" not in r.output


def test_log_tail_falls_back_to_legacy_when_no_activity(tmp_path: Path, monkeypatch) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    _write_jsonl(
        project / ".cryomodel_history.jsonl",
        [
            {
                "timestamp": "2026-04-06T09:58:00",
                "tool": "fitprep",
                "command": "cryomodel fitprep check --map m.mrc",
                "status": "success",
                "duration_s": 0.4,
            }
        ],
    )
    monkeypatch.chdir(tmp_path)
    r = CliRunner().invoke(app, ["tail", "--cwd", str(project), "--lines", "1"])
    assert r.exit_code == 0, r.output
    assert "fitprep" in r.output


def test_log_stats_reads_activity_schema(tmp_path: Path, monkeypatch) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    _write_jsonl(
        project / ".cryomodel" / "activity.jsonl",
        [
            {
                "timestamp": "2026-04-06T10:01:00",
                "source": "workflow_ui",
                "card_id": "foldhunter_card",
                "command": "cryomodel foldhunter --map m.mrc --model x.pdb",
                "status": "success",
                "return_code": 0,
                "duration_s": 2.0,
            },
            {
                "timestamp": "2026-04-06T10:02:00",
                "source": "workflow_ui",
                "card_id": "foldhunter_card",
                "command": "cryomodel foldhunter --map m2.mrc --model x.pdb",
                "status": "error",
                "return_code": 1,
                "duration_s": 0.8,
            },
        ],
    )
    monkeypatch.chdir(tmp_path)
    r = CliRunner().invoke(app, ["stats", "--cwd", str(project)])
    assert r.exit_code == 0, r.output
    assert "foldhunter_card" in r.output
    assert "success: 1" in r.output
    assert "error: 1" in r.output
