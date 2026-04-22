"""HTTP API for startup project manager UI (Phase P2 Slice B)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from cryomodel.cli import manager as manager_cli


class SaveProjectRequest(BaseModel):
    project_root: str
    name: Optional[str] = None
    description: Optional[str] = None
    api_host: Optional[str] = None
    api_port: Optional[int] = None
    chimerax_app: Optional[str] = None
    manifest_path: Optional[str] = None
    auto_load_last: Optional[bool] = None
    start_server_on_launch: Optional[bool] = None


class DeleteProjectRequest(BaseModel):
    project_root: str
    yes: bool = False


class LaunchProjectRequest(BaseModel):
    project_root: Optional[str] = None
    host: Optional[str] = None
    port: Optional[int] = None
    open_ui: bool = True
    start_server: Optional[bool] = None


class BrowseRequest(BaseModel):
    initial_dir: Optional[str] = None
    title: Optional[str] = None


app = FastAPI(title="CryoModel Manager API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> Dict[str, Any]:
    out: Dict[str, Any] = {"status": "ok"}
    try:
        out["workflow_ui_html"] = str(manager_cli._workflow_ui_html_path())
    except OSError as e:
        out["workflow_ui_html_error"] = str(e)
    return out


@app.get("/manager/projects/match")
def match_project(path: str) -> Dict[str, Any]:
    """Return registry entry whose project_root resolves to the same path as ``path``."""
    raw = (path or "").strip()
    if not raw:
        return {"ok": True, "project": None}
    try:
        root = Path(raw).expanduser().resolve()
    except OSError:
        raise HTTPException(status_code=400, detail="invalid path")
    if not root.is_dir():
        return {"ok": True, "project": None}
    for p in manager_cli._load_projects():
        pr_s = p.get("project_root")
        if not pr_s:
            continue
        try:
            pr = Path(str(pr_s)).expanduser().resolve()
        except OSError:
            continue
        if pr == root:
            return {"ok": True, "project": p}
    return {"ok": True, "project": None}


@app.get("/manager/projects")
def list_projects() -> Dict[str, Any]:
    sessions = manager_cli._load_sessions()
    meta = sessions.get("meta", {}) if isinstance(sessions, dict) else {}
    items = list(manager_cli._load_projects())
    items.sort(
        key=lambda p: (str(p.get("last_opened") or ""), str(p.get("updated_at") or "")),
        reverse=True,
    )
    return {
        "projects": items,
        "last_project": meta.get("last_project") if isinstance(meta, dict) else None,
    }


@app.post("/manager/projects/save")
def save_project(req: SaveProjectRequest) -> Dict[str, Any]:
    root = Path(req.project_root).expanduser().resolve()
    if not root.is_dir():
        raise HTTPException(status_code=400, detail=f"project_root is not a directory: {root}")
    p = manager_cli._upsert_project(
        root,
        name=req.name,
        description=req.description,
        api_host=req.api_host,
        api_port=req.api_port,
        chimerax_app=req.chimerax_app,
        manifest_path=req.manifest_path,
        auto_load_last=req.auto_load_last,
        start_server_on_launch=req.start_server_on_launch,
        touch_last_opened=True,
    )
    return {"ok": True, "project": p}


@app.post("/manager/projects/delete")
def delete_project(req: DeleteProjectRequest) -> Dict[str, Any]:
    root = Path(req.project_root).expanduser().resolve()
    key = manager_cli._project_key(root)
    projects = manager_cli._load_projects()
    kept = [p for p in projects if p.get("project_root") != key]
    if len(kept) == len(projects):
        return {"ok": True, "deleted": False}
    if not req.yes:
        raise HTTPException(status_code=400, detail="Missing confirmation; send yes=true to delete entry.")
    manager_cli._save_projects(kept)
    sessions = manager_cli._load_sessions()
    proj_sessions = sessions.get("projects", {})
    if isinstance(proj_sessions, dict):
        proj_sessions.pop(key, None)
        sessions["projects"] = proj_sessions
    meta = sessions.get("meta", {})
    if isinstance(meta, dict) and meta.get("last_project") == key:
        meta.pop("last_project", None)
        sessions["meta"] = meta
    manager_cli._save_sessions(sessions)
    return {"ok": True, "deleted": True}


@app.post("/manager/projects/launch")
def launch_project(req: LaunchProjectRequest) -> Dict[str, Any]:
    try:
        result = manager_cli._open_project_impl(
            project=Path(req.project_root).expanduser().resolve() if req.project_root else None,
            host=req.host,
            port=req.port,
            open_ui=req.open_ui,
            start_server=req.start_server,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"ok": True, **result}


@app.post("/manager/browse/directory")
def browse_directory(req: BrowseRequest) -> Dict[str, Any]:
    path = manager_cli.browse_directory_for_ui(initial_dir=req.initial_dir, title=req.title)
    return {"ok": True, "path": path}


@app.post("/manager/browse/file")
def browse_file(req: BrowseRequest) -> Dict[str, Any]:
    path = manager_cli.browse_file_for_ui(initial_dir=req.initial_dir, title=req.title)
    return {"ok": True, "path": path}
