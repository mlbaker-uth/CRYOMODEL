import json

import pytest

from cryomodel.workflow.activity_log import (
    append_activity_line,
    persist_workflow_ui_run,
    activity_jsonl_path,
    run_log_path,
)


def test_persist_workflow_ui_run_creates_files(tmp_path):
    d = tmp_path / "proj"
    d.mkdir()
    persist_workflow_ui_run(
        str(d),
        run_id="run_abc123",
        card_id="card_1",
        command="cryomodel validate --model x.pdb --map m.mrc",
        status="success",
        return_code=0,
        log_text="line1\nline2\n",
        started_perf=__import__("time").perf_counter() - 1.0,
    )
    act = activity_jsonl_path(d)
    assert act.is_file()
    lines = act.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["source"] == "workflow_ui"
    assert row["run_id"] == "run_abc123"
    assert row["card_id"] == "card_1"
    assert row["status"] == "success"
    assert row["return_code"] == 0
    assert row["command"].startswith("cryomodel validate")
    assert row["output_log"] == ".cryomodel/runs/run_abc123.log"
    assert "duration_s" in row and row["duration_s"] >= 0

    rlog = run_log_path(d, "run_abc123")
    assert rlog.is_file()
    assert rlog.read_text(encoding="utf-8") == "line1\nline2\n"


def test_append_activity_line_merge(tmp_path):
    d = tmp_path / "p"
    d.mkdir()
    append_activity_line(d, {"a": 1})
    append_activity_line(d, {"b": 2})
    text = activity_jsonl_path(d).read_text(encoding="utf-8").strip().splitlines()
    assert json.loads(text[0]) == {"a": 1}
    assert json.loads(text[1]) == {"b": 2}


def test_persist_requires_directory(tmp_path):
    with pytest.raises(FileNotFoundError):
        persist_workflow_ui_run(
            str(tmp_path / "nope"),
            run_id="r",
            card_id="c",
            command="x",
            status="error",
            return_code=1,
            log_text="",
            started_perf=0.0,
        )


def test_run_id_sanitized_path_separators(tmp_path):
    d = tmp_path / "p"
    d.mkdir()
    p = run_log_path(d, "../evil")
    assert ".." not in p.name and "/" not in p.name
