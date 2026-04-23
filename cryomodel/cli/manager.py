"""Project organizer / launcher for workflow UI sessions (Phase P2 Slice A)."""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode
from uuid import uuid4

import typer

from .command_log import log_command

manager_app = typer.Typer(
    help="Project organizer and workflow-ui launcher.",
    invoke_without_command=True,
)

REGISTRY_DIR = Path.home() / ".cryomodel"
PROJECTS_FILE = REGISTRY_DIR / "projects.json"
SESSIONS_FILE = REGISTRY_DIR / "sessions.json"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8010
DEFAULT_MANAGER_HOST = "127.0.0.1"
DEFAULT_MANAGER_PORT = 8011

# Repo / install root: parent of the `cryomodel` package (…/cryomodel/cli/manager.py → parents[2]).
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _now_local() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _ensure_registry_dir() -> None:
    REGISTRY_DIR.mkdir(parents=True, exist_ok=True)


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _write_json(path: Path, payload: Any) -> None:
    _ensure_registry_dir()
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _project_key(path: Path) -> str:
    return str(path.expanduser().resolve())


def _coerce_projects_raw(data: Any) -> List[Any]:
    """Accept legacy shapes: bare list, or {\"projects\": [...] }."""
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        inner = data.get("projects")
        if isinstance(inner, list):
            return inner
        if data.get("project_root") or data.get("path"):
            return [data]
    return []


def _load_projects() -> List[Dict[str, Any]]:
    data = _read_json(PROJECTS_FILE, [])
    items = _coerce_projects_raw(data)
    norm: List[Dict[str, Any]] = []
    changed = False
    format_migrated = isinstance(data, dict) and isinstance(data.get("projects"), list)
    for p in items:
        q = _normalize_project_record(p if isinstance(p, dict) else {})
        if q:
            norm.append(q)
            if q != p:
                changed = True
    if format_migrated:
        changed = True
    if changed:
        _save_projects(norm)
    return norm


def _save_projects(items: List[Dict[str, Any]]) -> None:
    if PROJECTS_FILE.exists() and PROJECTS_FILE.stat().st_size > 0:
        try:
            shutil.copy2(PROJECTS_FILE, PROJECTS_FILE.with_suffix(".json.bak"))
        except OSError:
            pass
    _write_json(PROJECTS_FILE, items)


def _load_sessions() -> Dict[str, Any]:
    """
    sessions.json schema:
    {
      "projects": { "<project_root>": {...session...} },
      "meta": { "last_project": "<project_root>" }
    }

    Backward compatibility: old flat map { "<project_root>": {...} }.
    """
    data = _read_json(SESSIONS_FILE, {})
    if not isinstance(data, dict):
        return {"projects": {}, "meta": {}}
    if "projects" in data and isinstance(data.get("projects"), dict):
        data.setdefault("meta", {})
        if not isinstance(data["meta"], dict):
            data["meta"] = {}
        return data
    flat_projects = {k: v for k, v in data.items() if isinstance(v, dict)}
    return {"projects": flat_projects, "meta": {}}


def _save_sessions(items: Dict[str, Any]) -> None:
    _write_json(SESSIONS_FILE, items)


def _normalize_project_record(item: Dict[str, Any]) -> Dict[str, Any]:
    path_s = str(item.get("project_root") or item.get("path") or "").strip()
    if not path_s:
        return {}
    root = Path(path_s).expanduser().resolve()
    now = _now_local()
    api_host = str(item.get("api_host") or DEFAULT_HOST)
    api_port = int(item.get("api_port") or DEFAULT_PORT)
    out = {
        "id": str(item.get("id") or f"prj_{uuid4().hex[:10]}"),
        "name": str(item.get("name") or root.name),
        "project_root": str(root),
        "description": str(item.get("description") or ""),
        "api_host": api_host,
        "api_port": api_port,
        "api_base": str(item.get("api_base") or f"http://{api_host}:{api_port}"),
        "chimerax_app": str(item.get("chimerax_app") or "ChimeraX"),
        "manifest_path": str(item.get("manifest_path") or ""),
        "auto_load_last": bool(item.get("auto_load_last", True)),
        "start_server_on_launch": bool(item.get("start_server_on_launch", True)),
        "created_at": str(item.get("created_at") or now),
        "updated_at": str(item.get("updated_at") or now),
        "last_opened": str(item.get("last_opened") or item.get("updated_at") or now),
    }
    return out


def _find_project(projects: List[Dict[str, Any]], project_root: Path) -> Optional[Dict[str, Any]]:
    key = _project_key(project_root)
    for p in projects:
        if p.get("project_root") == key:
            return p
    return None


def _upsert_project(
    project_root: Path,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
    api_host: Optional[str] = None,
    api_port: Optional[int] = None,
    chimerax_app: Optional[str] = None,
    manifest_path: Optional[str] = None,
    auto_load_last: Optional[bool] = None,
    start_server_on_launch: Optional[bool] = None,
    touch_last_opened: bool = False,
) -> Dict[str, Any]:
    root = project_root.expanduser().resolve()
    projects = _load_projects()
    p = _find_project(projects, root)
    now = _now_local()
    if p is None:
        p = _normalize_project_record({"project_root": str(root), "name": root.name})
        projects.append(p)
    if name is not None:
        p["name"] = str(name)
    if description is not None:
        p["description"] = str(description)
    if api_host is not None:
        p["api_host"] = str(api_host)
    if api_port is not None:
        p["api_port"] = int(api_port)
    p["api_base"] = f"http://{p.get('api_host', DEFAULT_HOST)}:{int(p.get('api_port', DEFAULT_PORT))}"
    if chimerax_app is not None:
        p["chimerax_app"] = str(chimerax_app)
    if manifest_path is not None:
        p["manifest_path"] = str(manifest_path)
    if auto_load_last is not None:
        p["auto_load_last"] = bool(auto_load_last)
    if start_server_on_launch is not None:
        p["start_server_on_launch"] = bool(start_server_on_launch)
    p["updated_at"] = now
    if touch_last_opened:
        p["last_opened"] = now
    _save_projects(projects)
    return p


def _resolve_project_for_open(project: Optional[Path]) -> Path:
    if project is not None:
        return project.expanduser().resolve()
    sessions = _load_sessions()
    meta = sessions.get("meta", {})
    if isinstance(meta, dict):
        last = str(meta.get("last_project") or "").strip()
        if last:
            return Path(last).expanduser().resolve()
    return Path.cwd().resolve()


def _is_port_open(host: str, port: int, timeout: float = 0.25) -> bool:
    try:
        with socket.create_connection((host, int(port)), timeout=timeout):
            return True
    except OSError:
        return False


def _pid_running(pid: Optional[int]) -> bool:
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
        return True
    except OSError:
        return False


def _browse_tk(mode: str, initial_dir: Optional[str], title: Optional[str]) -> str:
    """Fallback file/folder picker (may fail when the server has no GUI session)."""
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception:
        return ""
    root = tk.Tk()
    root.withdraw()
    try:
        root.attributes("-topmost", True)
    except Exception:
        pass
    idir = str(Path(initial_dir).expanduser().resolve()) if initial_dir else str(Path.home())
    ttl = title or ("Select directory" if mode == "dir" else "Select file")
    try:
        if mode == "dir":
            out = filedialog.askdirectory(initialdir=idir, title=ttl, mustexist=True)
        else:
            out = filedialog.askopenfilename(initialdir=idir, title=ttl)
    finally:
        root.destroy()
    return str(out or "")


def _browse_macos_folder(title: str, initial_dir: str) -> str:
    try:
        idir_res = Path(initial_dir).expanduser().resolve()
    except OSError:
        idir_res = Path.home()
    if not idir_res.is_dir():
        idir_res = Path.home()
    pstr = str(idir_res).replace("\\", "\\\\").replace('"', '\\"')
    esc_t = title.replace("\\", "\\\\").replace('"', '\\"')
    script = (
        f"try\n"
        f"\tset defAlias to POSIX file \"{pstr}\"\n"
        f"\tset out to POSIX path of (choose folder with prompt \"{esc_t}\" default location defAlias)\n"
        f"on error\n"
        f"\tset out to POSIX path of (choose folder with prompt \"{esc_t}\")\n"
        f"end try\n"
        f"return out"
    )
    r = subprocess.run(["osascript"], input=script, capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        return ""
    return (r.stdout or "").strip()


def _browse_macos_file(title: str, initial_dir: str) -> str:
    try:
        idir_res = Path(initial_dir).expanduser().resolve()
    except OSError:
        idir_res = Path.home()
    if not idir_res.is_dir():
        idir_res = Path.home()
    pstr = str(idir_res).replace("\\", "\\\\").replace('"', '\\"')
    esc_t = title.replace("\\", "\\\\").replace('"', '\\"')
    script = (
        f"try\n"
        f"\tset defAlias to POSIX file \"{pstr}\"\n"
        f"\tset out to POSIX path of (choose file with prompt \"{esc_t}\" default location defAlias)\n"
        f"on error\n"
        f"\tset out to POSIX path of (choose file with prompt \"{esc_t}\")\n"
        f"end try\n"
        f"return out"
    )
    r = subprocess.run(["osascript"], input=script, capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        return ""
    return (r.stdout or "").strip()


def _browse_linux_zenity(mode: str, title: str, initial_dir: str) -> str:
    zen = shutil.which("zenity")
    if not zen:
        return ""
    idir = str(Path(initial_dir).expanduser().resolve()) if initial_dir else str(Path.home())
    if not Path(idir).is_dir():
        idir = str(Path.home())
    if mode == "dir":
        cmd = [zen, "--file-selection", "--directory", f"--title={title}", f"--filename={idir}/"]
    else:
        cmd = [zen, "--file-selection", f"--title={title}", f"--filename={idir}"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        return ""
    line = (r.stdout or "").strip().split("\n", 1)[0]
    return line


def browse_directory_for_ui(initial_dir: Optional[str] = None, title: Optional[str] = None) -> str:
    """Native folder picker for the manager API (osascript on macOS; zenity on Linux; Tk elsewhere)."""
    ttl = title or "Select project directory"
    idir = (initial_dir or "").strip() or str(Path.home())
    if sys.platform == "darwin" and shutil.which("osascript"):
        return _browse_macos_folder(ttl, idir) or ""
    if sys.platform.startswith("linux") and shutil.which("zenity"):
        return _browse_linux_zenity("dir", ttl, idir) or ""
    return _browse_tk("dir", initial_dir, title)


def browse_file_for_ui(initial_dir: Optional[str] = None, title: Optional[str] = None) -> str:
    """Native file picker for the manager API."""
    ttl = title or "Select file"
    idir = (initial_dir or "").strip() or str(Path.home())
    if sys.platform == "darwin" and shutil.which("osascript"):
        return _browse_macos_file(ttl, idir) or ""
    if sys.platform.startswith("linux") and shutil.which("zenity"):
        return _browse_linux_zenity("file", ttl, idir) or ""
    return _browse_tk("file", initial_dir, title)


def _workflow_ui_html_path() -> Path:
    """Primary workflow HTML (V2). Set CRYOMODEL_WORKFLOW_HTML to override (absolute or repo-relative)."""
    override = os.environ.get("CRYOMODEL_WORKFLOW_HTML", "").strip()
    if override:
        p = Path(override).expanduser()
        return p.resolve() if p.is_absolute() else (_REPO_ROOT / p).resolve()
    primary = (_REPO_ROOT / "cryomodel.html").resolve()
    if primary.is_file():
        return primary
    raise FileNotFoundError(
        f"cryomodel.html not found at {primary}. "
        "Install from source with repo root on disk, or set CRYOMODEL_WORKFLOW_HTML to the full path."
    )


def _workflow_ui_url(
    project_root: Path,
    host: str,
    port: int,
    *,
    project_name: Optional[str] = None,
    manifest_path: Optional[str] = None,
    chimerax_app: Optional[str] = None,
) -> str:
    """Build file:// URL for workflow UI; query mirrors saved project fields when present."""
    html = _workflow_ui_html_path()
    api = f"http://{host}:{port}"
    params: Dict[str, Any] = {
        "cwd": str(project_root),
        "api": api,
        "manifest": (manifest_path or "").strip(),
        "chimerax": (chimerax_app or "").strip(),
    }
    if project_name and str(project_name).strip():
        params["project"] = str(project_name).strip()
    q = urlencode(params)
    return f"file://{html}?{q}"


def _manager_ui_url(
    host: str,
    port: int,
    *,
    default_project_root: Path,
    default_api_host: str,
    default_api_port: int,
) -> str:
    html = _REPO_ROOT / "cryomodel_manager.html"
    api = f"http://{host}:{port}"
    q = urlencode(
        {
            "api": api,
            "default_project_root": str(default_project_root),
            "default_api_host": str(default_api_host),
            "default_api_port": int(default_api_port),
            "home_dir": str(Path.home().resolve()),
        }
    )
    return f"file://{html}?{q}"


def _open_browser(url: str, browser_cmd: Optional[str] = None) -> bool:
    """
    Best-effort browser launch.

    On some Linux systems, ``xdg-open file://...?...`` may route to ``gio`` and print
    ``Operation not supported`` for file-URLs with query parameters even though the file exists.
    Use a resilient sequence and suppress noisy launcher stderr.
    """
    # Explicit browser command override (e.g., "firefox", "google-chrome").
    if browser_cmd:
        parts = [p for p in str(browser_cmd).strip().split() if p]
        if parts:
            try:
                subprocess.Popen(parts + [url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return True
            except Exception:
                pass

    # First try Python's browser dispatch.
    try:
        if webbrowser.open(url, new=2):
            return True
    except Exception:
        pass

    # OS-specific launchers.
    if sys.platform == "darwin":
        try:
            subprocess.Popen(["open", url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        except Exception:
            return False

    if sys.platform.startswith("linux"):
        # Try common launchers; swallow stderr noise from gio/xdg edge cases.
        for cmd in ("xdg-open", "gio"):
            exe = shutil.which(cmd)
            if not exe:
                continue
            args = [exe, "open", url] if cmd == "gio" else [exe, url]
            try:
                subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return True
            except Exception:
                continue
        return False

    if os.name == "nt":
        try:
            os.startfile(url)  # type: ignore[attr-defined]
            return True
        except Exception:
            return False

    return False


def _start_server(project_root: Path, host: str, port: int) -> int:
    args = [
        sys.executable,
        "-m",
        "cryomodel.cli",
        "workflow-ui",
        "serve",
        "--host",
        host,
        "--port",
        str(port),
    ]
    proc = subprocess.Popen(
        args,
        cwd=str(project_root),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    return int(proc.pid)


def _start_manager_api_server(host: str, port: int) -> int:
    args = [
        sys.executable,
        "-m",
        "cryomodel.cli",
        "manager",
        "serve",
        "--host",
        host,
        "--port",
        str(port),
    ]
    proc = subprocess.Popen(
        args,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    return int(proc.pid)


def _open_project_impl(
    *,
    project: Optional[Path],
    host: Optional[str],
    port: Optional[int],
    open_ui: bool,
    start_server: Optional[bool],
    browser_cmd: Optional[str] = None,
) -> Dict[str, Any]:
    project_root = _resolve_project_for_open(project)
    if not project_root.is_dir():
        raise ValueError(f"Project path is not a directory: {project_root}")
    existing = _find_project(_load_projects(), project_root)
    effective_host = host if host is not None else (existing.get("api_host") if existing else DEFAULT_HOST)
    effective_port = int(port if port is not None else (existing.get("api_port") if existing else DEFAULT_PORT))
    effective_start_server = bool(
        start_server if start_server is not None else (existing.get("start_server_on_launch", True) if existing else True)
    )

    project_record = _upsert_project(
        project_root,
        api_host=effective_host,
        api_port=effective_port,
        touch_last_opened=True,
    )
    key = _project_key(project_root)
    sessions = _load_sessions()
    project_sessions = sessions.get("projects", {})
    if not isinstance(project_sessions, dict):
        project_sessions = {}
    sess = project_sessions.get(key, {})
    running = _is_port_open(effective_host, effective_port)

    pid = sess.get("pid")
    if effective_start_server and not running:
        pid = _start_server(project_root, effective_host, effective_port)
        running = True

    project_sessions[key] = {
        "project_id": project_record.get("id"),
        "project_root": str(project_root),
        "host": effective_host,
        "port": int(effective_port),
        "api_base": f"http://{effective_host}:{effective_port}",
        "pid": int(pid) if pid else None,
        "running": bool(running),
        "updated_at": _now_local(),
    }
    sessions["projects"] = project_sessions
    meta = sessions.get("meta", {})
    if not isinstance(meta, dict):
        meta = {}
    meta["last_project"] = key
    sessions["meta"] = meta
    _save_sessions(sessions)

    workflow_url = _workflow_ui_url(
        project_root,
        effective_host,
        effective_port,
        project_name=str(project_record.get("name") or ""),
        manifest_path=str(project_record.get("manifest_path") or ""),
        chimerax_app=str(project_record.get("chimerax_app") or ""),
    )
    if open_ui:
        _open_browser(workflow_url, browser_cmd=browser_cmd)
    return {
        "project_root": str(project_root),
        "api_url": f"http://{effective_host}:{effective_port}",
        "start_server_on_launch": bool(effective_start_server),
        "pid": int(pid) if pid else None,
        "opened_ui": bool(open_ui),
        "workflow_url": workflow_url,
    }


@manager_app.callback()
def _manager(
    ctx: typer.Context,
    ui: bool = typer.Option(True, "--ui/--no-ui", help="Open startup manager window when no subcommand is given."),
    host: str = typer.Option(DEFAULT_MANAGER_HOST, "--host", help="Manager API host for startup window."),
    port: int = typer.Option(DEFAULT_MANAGER_PORT, "--port", help="Manager API port for startup window."),
    browser: Optional[str] = typer.Option(
        None,
        "--browser",
        help="Browser command override (e.g. 'firefox', 'google-chrome', '/usr/bin/open -a Safari').",
    ),
) -> None:
    if ctx.invoked_subcommand is not None:
        return
    if ui:
        if _is_port_open(host, port):
            typer.echo(
                f"Note: port {port} is already in use (manager API may be an older process). "
                "If Launch Application opens the legacy workflow UI, stop that process and run "
                "`cryomodel manager` again, or restart with: `cryomodel manager serve` on a free port.",
                err=True,
            )
        else:
            _start_manager_api_server(host, port)
        cwd = Path.cwd().resolve()
        default_project_root = cwd if str(cwd).strip() not in {"", "/"} else Path.home().resolve()
        url = _manager_ui_url(
            host,
            port,
            default_project_root=default_project_root,
            default_api_host=DEFAULT_HOST,
            default_api_port=DEFAULT_PORT,
        )
        opened = _open_browser(url, browser_cmd=browser)
        if opened:
            typer.echo(f"Opened manager UI: {url}")
        else:
            typer.echo(f"Manager UI URL: {url}")
            typer.echo("Could not auto-open browser on this system. Copy URL into a browser manually.", err=True)
        raise typer.Exit(0)
    typer.echo("Use `cryomodel manager --ui` to open startup window, or run a manager subcommand.")


@manager_app.command("list")
@log_command("manager list")
def list_projects() -> None:
    """List known projects from ~/.cryomodel/projects.json."""
    projects = _load_projects()
    if not projects:
        typer.echo("No projects registered yet.")
        raise typer.Exit(0)
    for p in projects:
        typer.echo(
            f"{p.get('name','(unnamed)')} | {p.get('project_root','?')} | "
            f"api={p.get('api_base','n/a')} | last_opened={p.get('last_opened','n/a')}"
        )


@manager_app.command("save")
@log_command("manager save")
def save_project(
    project: Path = typer.Option(..., "--project", help="Project directory"),
    name: Optional[str] = typer.Option(None, "--name", help="Display name"),
    description: Optional[str] = typer.Option(None, "--description", help="Project description"),
    api_host: Optional[str] = typer.Option(None, "--api-host", help="Workflow API host"),
    api_port: Optional[int] = typer.Option(None, "--api-port", help="Workflow API port"),
    chimerax_app: Optional[str] = typer.Option(None, "--chimerax-app", help="ChimeraX app name/path"),
    manifest_path: Optional[str] = typer.Option(None, "--manifest-path", help="Optional manifest path"),
    auto_load_last: Optional[bool] = typer.Option(
        None, "--auto-load-last/--no-auto-load-last", help="Whether manager should remember this project as default"
    ),
    start_server_on_launch: Optional[bool] = typer.Option(
        None, "--start-server-on-launch/--no-start-server-on-launch", help="Default launch behavior for workflow API"
    ),
) -> None:
    """Create or update a project entry (settings-only; does not start UI)."""
    root = project.expanduser().resolve()
    if not root.is_dir():
        raise typer.BadParameter(f"Project path is not a directory: {root}")
    p = _upsert_project(
        root,
        name=name,
        description=description,
        api_host=api_host,
        api_port=api_port,
        chimerax_app=chimerax_app,
        manifest_path=manifest_path,
        auto_load_last=auto_load_last,
        start_server_on_launch=start_server_on_launch,
    )
    typer.echo(f"Saved project: {p['name']} | {p['project_root']}")
    typer.echo(f"API: {p['api_base']}")


@manager_app.command("delete")
@log_command("manager delete")
def delete_project(
    project: Path = typer.Option(..., "--project", help="Project directory"),
    yes: bool = typer.Option(False, "--yes", help="Confirm deleting project entry from registry"),
) -> None:
    """Delete project entry and manager session metadata (never deletes project files)."""
    root = project.expanduser().resolve()
    key = _project_key(root)
    projects = _load_projects()
    kept = [p for p in projects if p.get("project_root") != key]
    if len(kept) == len(projects):
        typer.echo(f"No project entry found for: {root}")
        raise typer.Exit(0)
    if not yes:
        typer.echo("Refusing to delete without --yes (entry only; data is untouched).")
        raise typer.Exit(1)
    _save_projects(kept)
    sessions = _load_sessions()
    proj_sessions = sessions.get("projects", {})
    if isinstance(proj_sessions, dict):
        proj_sessions.pop(key, None)
        sessions["projects"] = proj_sessions
    meta = sessions.get("meta", {})
    if isinstance(meta, dict) and meta.get("last_project") == key:
        meta.pop("last_project", None)
        sessions["meta"] = meta
    _save_sessions(sessions)
    typer.echo(f"Deleted project entry: {root}")


@manager_app.command("open")
@log_command("manager open")
def open_project(
    project: Optional[Path] = typer.Option(None, "--project", help="Project directory (default: cwd)"),
    host: Optional[str] = typer.Option(None, "--host", help="Workflow API host (overrides project setting)"),
    port: Optional[int] = typer.Option(None, "--port", help="Workflow API port (overrides project setting)"),
    open_ui: bool = typer.Option(True, "--open-ui/--no-open-ui", help="Open workflow HTML in browser"),
    browser: Optional[str] = typer.Option(
        None,
        "--browser",
        help="Browser command override (e.g. 'firefox', 'google-chrome').",
    ),
    start_server: Optional[bool] = typer.Option(
        None,
        "--start-server/--no-start-server",
        help="Whether to start workflow API if not running (default from project settings)",
    ),
) -> None:
    """Open a project using saved settings, optionally starting workflow API and opening UI."""
    try:
        result = _open_project_impl(
            project=project,
            host=host,
            port=port,
            open_ui=open_ui,
            start_server=start_server,
            browser_cmd=browser,
        )
    except ValueError as e:
        raise typer.BadParameter(str(e))

    typer.echo(f"Project: {result['project_root']}")
    typer.echo(f"API: {result['api_url']}")
    typer.echo(f"Start server on launch: {result['start_server_on_launch']}")
    if result.get("pid"):
        typer.echo(f"PID: {result['pid']}")
    if result["opened_ui"]:
        typer.echo(f"Opened UI: {result['workflow_url']}")


@manager_app.command("status")
@log_command("manager status")
def status(
    project: Optional[Path] = typer.Option(None, "--project", help="Project directory (default: cwd)"),
) -> None:
    """Show manager session status for the project."""
    project_root = _resolve_project_for_open(project)
    key = _project_key(project_root)
    sessions = _load_sessions()
    proj_sessions = sessions.get("projects", {})
    sess = proj_sessions.get(key) if isinstance(proj_sessions, dict) else None
    if not sess:
        typer.echo(f"No manager session recorded for: {project_root}")
        raise typer.Exit(0)
    host = sess.get("host", DEFAULT_HOST)
    port = int(sess.get("port", DEFAULT_PORT))
    pid = sess.get("pid")
    typer.echo(f"Project: {project_root}")
    typer.echo(f"API: http://{host}:{port}")
    typer.echo(f"Port open: {_is_port_open(host, port)}")
    typer.echo(f"PID alive: {_pid_running(pid)}")
    typer.echo(f"Updated: {sess.get('updated_at', 'n/a')}")


@manager_app.command("stop")
@log_command("manager stop")
def stop(
    project: Optional[Path] = typer.Option(None, "--project", help="Project directory (default: cwd)"),
) -> None:
    """Stop managed workflow-ui server for the project if PID is known."""
    project_root = _resolve_project_for_open(project)
    key = _project_key(project_root)
    sessions = _load_sessions()
    proj_sessions = sessions.get("projects", {})
    sess = proj_sessions.get(key) if isinstance(proj_sessions, dict) else None
    if not sess:
        typer.echo(f"No manager session recorded for: {project_root}")
        raise typer.Exit(0)
    pid = sess.get("pid")
    if not pid:
        typer.echo("No PID recorded; nothing to stop.")
        raise typer.Exit(0)
    try:
        os.kill(int(pid), 15)
        typer.echo(f"Sent SIGTERM to PID {pid}")
    except OSError as e:
        typer.echo(f"Failed to stop PID {pid}: {e}")
    sess["running"] = False
    sess["updated_at"] = _now_local()
    if isinstance(proj_sessions, dict):
        proj_sessions[key] = sess
        sessions["projects"] = proj_sessions
    _save_sessions(sessions)


@manager_app.command("serve")
@log_command("manager serve")
def serve(
    host: str = typer.Option(DEFAULT_MANAGER_HOST, "--host", help="Host to bind manager API"),
    port: int = typer.Option(DEFAULT_MANAGER_PORT, "--port", help="Port to bind manager API"),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload"),
) -> None:
    """Serve the startup manager API used by the manager window."""
    try:
        import uvicorn
    except Exception as e:
        typer.echo(
            "uvicorn is required for manager serve.\n"
            "Install with: pip install uvicorn fastapi\n"
            f"Import error: {e}",
            err=True,
        )
        raise typer.Exit(1)
    try:
        typer.echo(f"Launch Application will open workflow UI: {_workflow_ui_html_path()}")
    except OSError as e:
        typer.echo(f"Warning: workflow UI path: {e}", err=True)
    uvicorn.run("cryomodel.workflow.manager_api:app", host=host, port=port, reload=reload)
