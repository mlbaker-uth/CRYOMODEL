"""Tests for CryoModel UI server cleanup helpers."""
from __future__ import annotations

import json
from pathlib import Path

from cryomodel.cli import server_cleanup as sc


def test_classify_workflow_and_manager() -> None:
    wf_cmd = "/usr/bin/python -m cryomodel.cli workflow-ui serve --port 8010"
    mgr_cmd = "python -m cryomodel.cli manager serve --port 8011"
    s1, c1 = sc._classify_service(wf_cmd, 8010)
    s2, c2 = sc._classify_service(mgr_cmd, 8011)
    assert c1 and s1 == "workflow-ui"
    assert c2 and s2 == "manager-api"


def test_collect_registry_ports_includes_defaults_and_project(tmp_path: Path) -> None:
    reg = tmp_path / ".cryomodel"
    reg.mkdir()
    projects = reg / "projects.json"
    projects.write_text(
        json.dumps([{"project_root": "/tmp/x", "api_port": 8123}], indent=2),
        encoding="utf-8",
    )
    ports = sc.collect_registry_ports(projects_file=projects, sessions_file=reg / "missing.json")
    assert sc.DEFAULT_WORKFLOW_PORT in ports
    assert sc.DEFAULT_MANAGER_PORT in ports
    assert 8123 in ports


def test_format_kill_help_other_user() -> None:
    item = sc.PortListener(
        port=8011,
        pid=999,
        user="otheruser",
        command="python -m cryomodel.cli manager serve",
        service="manager-api",
        cryomodel=True,
    )
    text = sc.format_kill_help([item], me="me")
    assert "otheruser" in text
    assert "sudo kill 999" in text
    assert "cryomodel manager cleanup" in text


def test_clear_stale_session_pids(tmp_path: Path) -> None:
    sf = tmp_path / "sessions.json"
    sf.write_text(
        json.dumps(
            {
                "projects": {
                    "/tmp/p": {"pid": 42, "running": True, "port": 8010},
                    "/tmp/q": {"pid": 99, "running": True, "port": 8010},
                },
                "meta": {},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    n = sc.clear_stale_session_pids({42}, sessions_file=sf)
    assert n == 1
    data = json.loads(sf.read_text(encoding="utf-8"))
    assert data["projects"]["/tmp/p"]["pid"] is None
    assert data["projects"]["/tmp/p"]["running"] is False
    assert data["projects"]["/tmp/q"]["pid"] == 99


def test_kill_listeners_skips_other_users(monkeypatch) -> None:
    item = sc.PortListener(
        port=8010,
        pid=123,
        user="other",
        command="cryomodel workflow-ui serve",
        service="workflow-ui",
        cryomodel=True,
    )
    results = sc.kill_listeners([item], current_user_only=True, me="me")
    assert len(results) == 1
    assert not results[0].ok
    assert "other" in results[0].message


def test_run_cleanup_dry_run(monkeypatch) -> None:
    fake = [
        sc.PortListener(
            port=8010,
            pid=1,
            user="me",
            command="cryomodel workflow-ui serve",
            service="workflow-ui",
            cryomodel=True,
        )
    ]
    monkeypatch.setattr(sc, "scan_listeners", lambda _ports: fake)
    monkeypatch.setattr(sc, "_current_user", lambda: "me")
    monkeypatch.setattr(sc.shutil, "which", lambda _name: "/usr/sbin/lsof")
    out = sc.run_cleanup(ports=[8010], kill=False)
    assert "8010" in out["report"]
    assert out["killed"] == []
