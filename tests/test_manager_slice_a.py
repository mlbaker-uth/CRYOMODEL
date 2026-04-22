from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from typer.testing import CliRunner

import cryomodel.cli.manager as manager_mod
from cryomodel.cli.manager import manager_app


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _wire_registry(monkeypatch, tmp_path: Path) -> tuple[Path, Path]:
    reg = tmp_path / ".cryomodel"
    projects = reg / "projects.json"
    sessions = reg / "sessions.json"
    monkeypatch.setattr(manager_mod, "REGISTRY_DIR", reg)
    monkeypatch.setattr(manager_mod, "PROJECTS_FILE", projects)
    monkeypatch.setattr(manager_mod, "SESSIONS_FILE", sessions)
    return projects, sessions


def test_manager_save_writes_rich_project_record(tmp_path: Path, monkeypatch) -> None:
    projects_file, _ = _wire_registry(monkeypatch, tmp_path)
    project_root = tmp_path / "proj_a"
    project_root.mkdir()
    r = CliRunner().invoke(
        manager_app,
        [
            "save",
            "--project",
            str(project_root),
            "--name",
            "DNA Modeling",
            "--description",
            "test project",
            "--api-host",
            "127.0.0.1",
            "--api-port",
            "8123",
            "--chimerax-app",
            "ChimeraX",
            "--manifest-path",
            str(project_root / "manifest.json"),
            "--auto-load-last",
            "--start-server-on-launch",
        ],
    )
    assert r.exit_code == 0, r.output
    payload = _read_json(projects_file)
    assert len(payload) == 1
    rec = payload[0]
    assert rec["name"] == "DNA Modeling"
    assert rec["project_root"] == str(project_root.resolve())
    assert rec["description"] == "test project"
    assert rec["api_base"] == "http://127.0.0.1:8123"
    assert rec["auto_load_last"] is True
    assert rec["start_server_on_launch"] is True


def test_manager_delete_removes_entry_not_data(tmp_path: Path, monkeypatch) -> None:
    projects_file, sessions_file = _wire_registry(monkeypatch, tmp_path)
    project_root = tmp_path / "proj_b"
    project_root.mkdir()
    key = str(project_root.resolve())
    projects_file.parent.mkdir(parents=True, exist_ok=True)
    projects_file.write_text(
        json.dumps([{"id": "p1", "name": "P", "project_root": key}], indent=2),
        encoding="utf-8",
    )
    sessions_file.write_text(
        json.dumps({"projects": {key: {"pid": 123}}, "meta": {"last_project": key}}, indent=2),
        encoding="utf-8",
    )

    r_no = CliRunner().invoke(manager_app, ["delete", "--project", str(project_root)])
    assert r_no.exit_code != 0
    assert project_root.is_dir()

    r_yes = CliRunner().invoke(manager_app, ["delete", "--project", str(project_root), "--yes"])
    assert r_yes.exit_code == 0, r_yes.output
    assert project_root.is_dir()
    assert _read_json(projects_file) == []
    sess = _read_json(sessions_file)
    assert sess["projects"] == {}
    assert sess["meta"].get("last_project") is None


def test_manager_open_uses_saved_defaults_and_updates_last_project(tmp_path: Path, monkeypatch) -> None:
    projects_file, sessions_file = _wire_registry(monkeypatch, tmp_path)
    project_root = tmp_path / "proj_c"
    project_root.mkdir()
    key = str(project_root.resolve())
    projects_file.parent.mkdir(parents=True, exist_ok=True)
    projects_file.write_text(
        json.dumps(
            [
                {
                    "id": "p1",
                    "name": "Proj C",
                    "project_root": key,
                    "api_host": "127.0.0.1",
                    "api_port": 8555,
                    "start_server_on_launch": False,
                }
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(manager_mod, "_is_port_open", lambda *_args, **_kw: True)
    monkeypatch.setattr(manager_mod, "_open_browser", lambda *_args, **_kw: None)

    r = CliRunner().invoke(manager_app, ["open", "--project", str(project_root), "--no-open-ui"])
    assert r.exit_code == 0, r.output
    assert "API: http://127.0.0.1:8555" in r.output
    assert "Start server on launch: False" in r.output

    sessions = _read_json(sessions_file)
    assert sessions["meta"]["last_project"] == key
    assert key in sessions["projects"]
    assert sessions["projects"][key]["api_base"] == "http://127.0.0.1:8555"


def test_manager_ui_url_includes_default_context() -> None:
    url = manager_mod._manager_ui_url(
        "127.0.0.1",
        8011,
        default_project_root=Path("/tmp/project"),
        default_api_host="127.0.0.1",
        default_api_port=8010,
    )
    qs = parse_qs(urlparse(url).query)
    assert qs["api"][0] == "http://127.0.0.1:8011"
    assert qs["default_project_root"][0] == "/tmp/project"
    assert qs["default_api_host"][0] == "127.0.0.1"
    assert qs["default_api_port"][0] == "8010"
    assert "home_dir" in qs


def test_workflow_ui_url_uses_cryomodel_html_by_default(monkeypatch) -> None:
    monkeypatch.delenv("CRYOMODEL_WORKFLOW_HTML", raising=False)
    url = manager_mod._workflow_ui_url(Path("/tmp/p"), "127.0.0.1", 8010)
    assert "cryomodel.html" in url
    assert "dna_workflow_ui_demo.html" not in url


def test_load_projects_accepts_wrapped_projects_json(tmp_path: Path, monkeypatch) -> None:
    """Legacy {\"projects\": [...]} must not be treated as empty (would wipe registry on save)."""
    projects_file, _ = _wire_registry(monkeypatch, tmp_path)
    projects_file.parent.mkdir(parents=True, exist_ok=True)
    project_root = tmp_path / "wrapped_proj"
    project_root.mkdir()
    legacy = {"projects": [{"project_root": str(project_root), "name": "Legacy Wrap"}]}
    projects_file.write_text(json.dumps(legacy), encoding="utf-8")
    items = manager_mod._load_projects()
    assert len(items) == 1
    assert items[0]["name"] == "Legacy Wrap"
    saved = json.loads(projects_file.read_text(encoding="utf-8"))
    assert isinstance(saved, list)
    assert saved[0]["project_root"] == str(project_root.resolve())


def test_workflow_ui_url_includes_registry_query_params() -> None:
    url = manager_mod._workflow_ui_url(
        Path("/tmp/myproj"),
        "127.0.0.1",
        8010,
        project_name="DNA run",
        manifest_path="/tmp/myproj/manifest.json",
        chimerax_app="/Applications/ChimeraX.app",
    )
    qs = parse_qs(urlparse(url).query)
    assert qs["cwd"][0] == "/tmp/myproj"
    assert qs["project"][0] == "DNA run"
    assert qs["manifest"][0] == "/tmp/myproj/manifest.json"
    assert qs["chimerax"][0] == "/Applications/ChimeraX.app"


def test_workflow_ui_url_always_passes_manifest_chimerax_for_localstorage_reset() -> None:
    url = manager_mod._workflow_ui_url(
        Path("/tmp/empty"),
        "127.0.0.1",
        8010,
        project_name="",
        manifest_path="",
        chimerax_app="",
    )
    qstr = urlparse(url).query
    assert "manifest=" in qstr and "chimerax=" in qstr
    qs = parse_qs(qstr, keep_blank_values=True)
    assert qs.get("manifest") == [""]
    assert qs.get("chimerax") == [""]
