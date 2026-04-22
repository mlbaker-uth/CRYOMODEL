from __future__ import annotations

from pathlib import Path

import cryomodel.cli.manager as manager_mod
from cryomodel.workflow import manager_api


def _wire_registry(monkeypatch, tmp_path: Path):
    reg = tmp_path / ".cryomodel"
    monkeypatch.setattr(manager_mod, "REGISTRY_DIR", reg)
    monkeypatch.setattr(manager_mod, "PROJECTS_FILE", reg / "projects.json")
    monkeypatch.setattr(manager_mod, "SESSIONS_FILE", reg / "sessions.json")


def test_manager_api_save_list_delete(tmp_path: Path, monkeypatch) -> None:
    _wire_registry(monkeypatch, tmp_path)
    project = tmp_path / "proj"
    project.mkdir()
    r_save = manager_api.save_project(
        manager_api.SaveProjectRequest(
            project_root=str(project),
            name="Proj API",
            api_host="127.0.0.1",
            api_port=8550,
        )
    )
    assert r_save["project"]["name"] == "Proj API"

    r_list = manager_api.list_projects()
    assert len(r_list["projects"]) == 1

    r_del = manager_api.delete_project(manager_api.DeleteProjectRequest(project_root=str(project), yes=True))
    assert r_del["deleted"] is True


def test_manager_api_launch_no_ui(tmp_path: Path, monkeypatch) -> None:
    _wire_registry(monkeypatch, tmp_path)
    project = tmp_path / "proj_launch"
    project.mkdir()
    monkeypatch.setattr(manager_mod, "_is_port_open", lambda *_a, **_k: True)
    manager_api.save_project(manager_api.SaveProjectRequest(project_root=str(project), name="L"))
    payload = manager_api.launch_project(manager_api.LaunchProjectRequest(project_root=str(project), open_ui=False))
    assert payload["ok"] is True
    assert payload["opened_ui"] is False
    assert payload["api_url"].startswith("http://")


def test_manager_api_list_sorts_by_last_opened(tmp_path: Path, monkeypatch) -> None:
    _wire_registry(monkeypatch, tmp_path)
    a = tmp_path / "proj_a"
    b = tmp_path / "proj_b"
    a.mkdir()
    b.mkdir()
    manager_api.save_project(
        manager_api.SaveProjectRequest(
            project_root=str(a),
            name="A",
            api_host="127.0.0.1",
            api_port=8010,
        )
    )
    manager_api.save_project(
        manager_api.SaveProjectRequest(
            project_root=str(b),
            name="B",
            api_host="127.0.0.1",
            api_port=8010,
        )
    )
    r_list = manager_api.list_projects()
    items = r_list["projects"]
    assert len(items) == 2
    keys = [(str(p.get("last_opened") or ""), str(p.get("updated_at") or "")) for p in items]
    assert keys == sorted(keys, reverse=True)
    assert {p["project_root"] for p in items} == {str(a.resolve()), str(b.resolve())}


def test_manager_api_match_by_path(tmp_path: Path, monkeypatch) -> None:
    _wire_registry(monkeypatch, tmp_path)
    a = tmp_path / "proj_match"
    a.mkdir()
    manager_api.save_project(
        manager_api.SaveProjectRequest(
            project_root=str(a),
            name="MatchMe",
            api_host="127.0.0.1",
            api_port=8010,
        )
    )
    hit = manager_api.match_project(path=str(a))
    assert hit["project"] and hit["project"]["name"] == "MatchMe"
    miss = manager_api.match_project(path=str(tmp_path / "missing"))
    assert miss["project"] is None
