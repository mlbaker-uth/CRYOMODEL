"""Lightweight API for UI-driven workflow job execution."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import json
import os
from pathlib import Path
import socket
import signal
from threading import Lock, Thread
from typing import Any, Dict, List, Optional
import shlex
import subprocess
import sys
import time
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


class RunJobRequest(BaseModel):
    card_id: str
    command: str
    cwd: Optional[str] = None


class RunJobResponse(BaseModel):
    run_id: str
    status: str


class RunStatusResponse(BaseModel):
    run_id: str
    card_id: str
    status: str
    return_code: Optional[int] = None
    started_at: str
    ended_at: Optional[str] = None


class RunLogResponse(BaseModel):
    run_id: str
    log: str


class ChimeraManifestLoadRequest(BaseModel):
    """Path to manifest JSON written by ChimeraX (command: cryomodel_manifest)."""

    path: str


class OpenChimeraXRequest(BaseModel):
    """Resolve file paths against cwd and open them with macOS `open -a <ChimeraX>`."""

    paths: List[str]
    cwd: Optional[str] = None
    """Application name as used by `open -a` (default bundle name on macOS)."""
    app_name: Optional[str] = None


class AssistantRequest(BaseModel):
    """Assistant request from workflow UI."""

    mode: str
    prompt: str
    cwd: Optional[str] = None
    tool: Optional[str] = None
    resolution: Optional[float] = None


class WorkflowYamlExportRequest(BaseModel):
    """Workflow JSON document to serialize as YAML."""

    workflow: Dict[str, Any]


class PathMeasureStartRequest(BaseModel):
    """Request to start PathMeasure server and optionally open browser."""

    host: str = "127.0.0.1"
    port: int = 8008
    cwd: Optional[str] = None
    open_browser: bool = True


class PathMeasureControlRequest(BaseModel):
    """Request for checking/stopping PathMeasure."""

    host: str = "127.0.0.1"
    port: int = 8008


@dataclass
class RunRecord:
    run_id: str
    card_id: str
    command: str
    cwd: str
    status: str = "running"
    return_code: Optional[int] = None
    started_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    ended_at: Optional[str] = None
    log: str = ""
    started_perf: float = field(default_factory=time.perf_counter)


RUNS: Dict[str, RunRecord] = {}
RUNS_LOCK = Lock()
PATHMEASURE_PROCS: Dict[str, int] = {}
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PATHMEASURE_DEMO_HTML = Path(
    os.environ.get("CRYOMODEL_PATHMEASURE_DEMO_HTML", str(PROJECT_ROOT / "pathmeasure_demo.html"))
).expanduser()

app = FastAPI(title="CryoModel Workflow UI API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _append_log(run_id: str, text: str) -> None:
    with RUNS_LOCK:
        rec = RUNS.get(run_id)
        if rec is not None:
            rec.log += text


def _set_done(run_id: str, return_code: int) -> None:
    snap: Optional[tuple[Any, ...]] = None
    with RUNS_LOCK:
        rec = RUNS.get(run_id)
        if rec is None:
            return
        rec.return_code = int(return_code)
        rec.status = "success" if return_code == 0 else "error"
        rec.ended_at = datetime.now().isoformat(timespec="seconds")
        snap = (
            rec.cwd,
            rec.run_id,
            rec.card_id,
            rec.command,
            rec.status,
            rec.return_code,
            rec.log,
            rec.started_perf,
        )
    if snap:
        cwd, rid, cid, cmd, st, rc, log_text, t0 = snap
        try:
            from cryomodel.workflow.activity_log import persist_workflow_ui_run

            persist_workflow_ui_run(
                cwd,
                run_id=rid,
                card_id=cid,
                command=cmd,
                status=st,
                return_code=rc,
                log_text=log_text,
                started_perf=t0,
            )
        except OSError as e:
            print(f"[cryomodel.workflow.ui_api] activity persist failed: {e}", file=sys.stderr)
        except Exception as e:  # pragma: no cover
            print(f"[cryomodel.workflow.ui_api] activity persist failed: {e}", file=sys.stderr)


def _worker(run_id: str) -> None:
    with RUNS_LOCK:
        rec = RUNS.get(run_id)
    if rec is None:
        return
    try:
        args = shlex.split(rec.command)
        # Line-iterating stdout blocks until a newline; Typer/Rich often emit progress
        # without newlines, and many CLIs block-buffer when not a TTY. Chunk reads +
        # PYTHONUNBUFFERED keep the UI log updating during long runs.
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        proc = subprocess.Popen(
            args,
            cwd=rec.cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=0,
            env=env,
        )
        assert proc.stdout is not None
        while True:
            chunk = proc.stdout.read(4096)
            if not chunk:
                break
            _append_log(run_id, chunk.decode("utf-8", errors="replace"))
        rc = proc.wait()
        _set_done(run_id, rc)
    except Exception as e:
        _append_log(run_id, f"\n[ui_api] failed to run command: {e}\n")
        _set_done(run_id, 1)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ui/home-dir")
def ui_home_dir() -> Dict[str, str]:
    """Home directory for workflow UI (manager defaults, path hints)."""
    return {"home_dir": str(Path.home().resolve())}


def _is_port_open(host: str, port: int, timeout: float = 0.25) -> bool:
    try:
        with socket.create_connection((host, int(port)), timeout=timeout):
            return True
    except OSError:
        return False


def _pid_running(pid: int) -> bool:
    try:
        os.kill(int(pid), 0)
        return True
    except OSError:
        return False


@app.post("/ui/run", response_model=RunJobResponse)
def run_job(req: RunJobRequest) -> RunJobResponse:
    cwd = str(Path(req.cwd).expanduser().resolve()) if req.cwd else str(Path.cwd())
    if not Path(cwd).exists():
        raise HTTPException(status_code=400, detail=f"cwd does not exist: {cwd}")
    run_id = f"run_{uuid.uuid4().hex[:10]}"
    rec = RunRecord(
        run_id=run_id,
        card_id=req.card_id,
        command=req.command,
        cwd=cwd,
        log=f"[{datetime.now().isoformat(timespec='seconds')}] Running in {cwd}\n$ {req.command}\n",
    )
    with RUNS_LOCK:
        RUNS[run_id] = rec
    t = Thread(target=_worker, args=(run_id,), daemon=True)
    t.start()
    return RunJobResponse(run_id=run_id, status="started")


@app.get("/ui/status/{run_id}", response_model=RunStatusResponse)
def run_status(run_id: str) -> RunStatusResponse:
    with RUNS_LOCK:
        rec = RUNS.get(run_id)
    if rec is None:
        raise HTTPException(status_code=404, detail=f"run_id not found: {run_id}")
    return RunStatusResponse(
        run_id=rec.run_id,
        card_id=rec.card_id,
        status=rec.status,
        return_code=rec.return_code,
        started_at=rec.started_at,
        ended_at=rec.ended_at,
    )


@app.get("/ui/log/{run_id}", response_model=RunLogResponse)
def run_log(run_id: str) -> RunLogResponse:
    with RUNS_LOCK:
        rec = RUNS.get(run_id)
    if rec is None:
        raise HTTPException(status_code=404, detail=f"run_id not found: {run_id}")
    return RunLogResponse(run_id=run_id, log=rec.log)


@app.post("/ui/chimerax-manifest")
def load_chimerax_manifest(req: ChimeraManifestLoadRequest) -> dict:
    """Load and validate a ChimeraX manifest JSON (phase 1: read-only)."""
    raw = (req.path or "").strip()
    if not raw or "\x00" in raw:
        raise HTTPException(status_code=400, detail="path is required")
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = Path.cwd() / p
    try:
        p = p.resolve()
    except OSError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if not p.is_file():
        raise HTTPException(status_code=400, detail=f"Manifest not found: {p}")
    max_bytes = 4 * 1024 * 1024
    if p.stat().st_size > max_bytes:
        raise HTTPException(status_code=400, detail="Manifest file too large")
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")
    if not isinstance(data, dict):
        raise HTTPException(status_code=400, detail="Manifest must be a JSON object")
    if data.get("schema_version") != 1:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported schema_version (expected 1): {data.get('schema_version')!r}",
        )
    entries = data.get("entries")
    if not isinstance(entries, list):
        raise HTTPException(status_code=400, detail="Manifest missing 'entries' array")
    cleaned: List[Dict[str, Any]] = []
    for i, ent in enumerate(entries):
        if not isinstance(ent, dict):
            continue
        cleaned.append(
            {
                "id": str(ent.get("id", i)),
                "label": str(ent.get("label", "")),
                "kind": str(ent.get("kind", "other")),
                "format_hint": str(ent.get("format_hint", "")),
                "path": ent.get("path"),
                "source": str(ent.get("source", "session")),
                "artifact_type": str(ent.get("artifact_type", "other")),
                "reason_no_path": ent.get("reason_no_path"),
            }
        )
    return {
        "schema_version": 1,
        "created_utc": data.get("created_utc"),
        "manifest_path": str(p),
        "entries": cleaned,
    }


@app.post("/ui/open-chimerax")
def open_chimera_x(req: OpenChimeraXRequest) -> dict:
    """Launch ChimeraX (or compatible app) with the given files. macOS only for this prototype."""
    if sys.platform != "darwin":
        raise HTTPException(
            status_code=501,
            detail="ChimeraX launch via `open -a` is only implemented for macOS in this prototype.",
        )
    cwd = str(Path(req.cwd).expanduser().resolve()) if req.cwd else str(Path.cwd())
    if not Path(cwd).exists():
        raise HTTPException(status_code=400, detail=f"cwd does not exist: {cwd}")

    app = (req.app_name or "ChimeraX").strip() or "ChimeraX"
    abs_paths: List[Path] = []
    seen = set()
    for raw in req.paths:
        s = (raw or "").strip()
        if not s or "\x00" in s:
            continue
        p = Path(s).expanduser()
        if not p.is_absolute():
            p = Path(cwd) / p
        try:
            p = p.resolve()
        except OSError:
            continue
        if not p.is_file():
            continue
        key = str(p)
        if key in seen:
            continue
        seen.add(key)
        abs_paths.append(p)

    if not abs_paths:
        raise HTTPException(
            status_code=400,
            detail="No existing files to open. Check workspace output paths and CWD, or run the pipeline first.",
        )

    cmd = ["open", "-a", app, *[str(p) for p in abs_paths]]
    try:
        subprocess.Popen(cmd, start_new_session=True)  # type: ignore[call-arg]
    except TypeError:
        subprocess.Popen(cmd)
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="`/usr/bin/open` not found (unexpected on macOS).")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "ok": True,
        "opened": [str(p) for p in abs_paths],
        "app": app,
        "command": cmd,
    }


@app.post("/ui/assistant")
def run_assistant(req: AssistantRequest) -> dict:
    """Run `cryomodel assistant <mode>` and return captured output."""
    cwd = str(Path(req.cwd).expanduser().resolve()) if req.cwd else str(Path.cwd())
    if not Path(cwd).exists():
        raise HTTPException(status_code=400, detail=f"cwd does not exist: {cwd}")

    mode = (req.mode or "").strip().lower()
    prompt = (req.prompt or "").strip()
    tool = (req.tool or "").strip()

    # UI-facing modes mapped to assistant subcommands.
    mode_to_subcommand = {
        "ask": "ask",
        "suggest": "suggest",
        "workflow": "suggest",
        "parameter": "ask",
        "program": "explain",
        "explain": "explain",
        "troubleshoot": "troubleshoot",
        "resolution": "resolution",
    }
    subcmd = mode_to_subcommand.get(mode)
    if subcmd is None:
        raise HTTPException(status_code=400, detail=f"Unknown assistant mode: {mode!r}")

    args: List[str] = ["cryomodel", "assistant", subcmd]

    # Build subcommand args
    if subcmd in ("ask", "suggest", "troubleshoot"):
        if not prompt:
            raise HTTPException(status_code=400, detail="prompt is required for this mode")
        args.append(prompt)
        if tool and subcmd in ("ask", "troubleshoot"):
            args.extend(["--tool", tool])
        if req.resolution is not None and subcmd == "ask":
            args.extend(["--resolution", str(req.resolution)])
    elif subcmd == "explain":
        if not prompt:
            raise HTTPException(status_code=400, detail="prompt is required for program/explain mode")
        # prompt is the tool name
        args.append(prompt)
    elif subcmd == "resolution":
        value: Optional[float] = req.resolution
        if value is None and prompt:
            try:
                value = float(prompt)
            except ValueError:
                value = None
        if value is None:
            raise HTTPException(status_code=400, detail="resolution mode needs a numeric prompt or resolution field")
        args.append(str(value))

    try:
        proc = subprocess.run(
            args,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="`cryomodel` executable not found in PATH")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "ok": proc.returncode == 0,
        "return_code": int(proc.returncode),
        "command": args,
        "stdout": proc.stdout or "",
        "stderr": proc.stderr or "",
        "mode": mode,
        "subcommand": subcmd,
    }


@app.post("/ui/workflow-export-yaml")
def workflow_export_yaml(req: WorkflowYamlExportRequest) -> dict:
    """Convert a workflow JSON object to YAML text."""
    wf = req.workflow
    if not isinstance(wf, dict):
        raise HTTPException(status_code=400, detail="workflow must be an object")
    if "steps" not in wf or not isinstance(wf.get("steps"), list):
        raise HTTPException(status_code=400, detail="workflow.steps must be an array")
    try:
        import yaml

        text = yaml.safe_dump(wf, default_flow_style=False, sort_keys=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"ok": True, "yaml": text}


@app.post("/ui/pathmeasure-start")
def start_pathmeasure(req: PathMeasureStartRequest) -> dict:
    """Start PathMeasure server in background and return URL."""
    host = (req.host or "127.0.0.1").strip() or "127.0.0.1"
    port = int(req.port)
    if port < 1 or port > 65535:
        raise HTTPException(status_code=400, detail="port must be between 1 and 65535")
    cwd = str(Path(req.cwd).expanduser().resolve()) if req.cwd else str(Path.cwd())
    if not Path(cwd).exists():
        raise HTTPException(status_code=400, detail=f"cwd does not exist: {cwd}")

    url = f"http://{host}:{port}"
    demo_path = PATHMEASURE_DEMO_HTML.resolve()
    if not demo_path.is_file():
        raise HTTPException(status_code=500, detail=f"PathMeasure demo HTML not found: {demo_path}")
    ui_url = f"file://{demo_path}?apiBase={url}"
    key = f"{host}:{port}"
    if _is_port_open(host, port):
        if req.open_browser and sys.platform == "darwin":
            try:
                subprocess.Popen(["open", ui_url], start_new_session=True)  # type: ignore[call-arg]
            except Exception:
                pass
        return {"ok": True, "already_running": True, "url": url, "ui_url": ui_url, "host": host, "port": port}

    args = ["cryomodel", "pathmeasure", "serve", "--host", host, "--port", str(port)]
    try:
        proc = subprocess.Popen(
            args,
            cwd=cwd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,  # type: ignore[call-arg]
            env={**os.environ},
        )
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="`cryomodel` executable not found in PATH")
    except TypeError:
        proc = subprocess.Popen(args, cwd=cwd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    PATHMEASURE_PROCS[key] = int(proc.pid)

    # Poll briefly so the UI can open only when service is up.
    for _ in range(20):
        if _is_port_open(host, port):
            break
        import time

        time.sleep(0.2)

    running = _is_port_open(host, port)
    if req.open_browser and running and sys.platform == "darwin":
        try:
            subprocess.Popen(["open", ui_url], start_new_session=True)  # type: ignore[call-arg]
        except Exception:
            pass

    return {
        "ok": running,
        "already_running": False,
        "started": True,
        "running": running,
        "pid": int(proc.pid),
        "url": url,
        "ui_url": ui_url,
        "host": host,
        "port": port,
    }


@app.post("/ui/pathmeasure-status")
def pathmeasure_status(req: PathMeasureControlRequest) -> dict:
    host = (req.host or "127.0.0.1").strip() or "127.0.0.1"
    port = int(req.port)
    key = f"{host}:{port}"
    pid = PATHMEASURE_PROCS.get(key)
    running = _is_port_open(host, port)
    pid_alive = _pid_running(pid) if pid else False
    return {
        "ok": True,
        "host": host,
        "port": port,
        "url": f"http://{host}:{port}",
        "running": running,
        "pid": pid,
        "pid_alive": pid_alive,
    }


@app.post("/ui/pathmeasure-stop")
def stop_pathmeasure(req: PathMeasureControlRequest) -> dict:
    host = (req.host or "127.0.0.1").strip() or "127.0.0.1"
    port = int(req.port)
    key = f"{host}:{port}"
    pid = PATHMEASURE_PROCS.get(key)

    if not _is_port_open(host, port):
        PATHMEASURE_PROCS.pop(key, None)
        return {"ok": True, "stopped": False, "running": False, "message": "PathMeasure is not running."}

    if not pid:
        # Service may have been started outside this API process.
        return {
            "ok": False,
            "stopped": False,
            "running": True,
            "message": "PathMeasure is running but PID is unknown to this API process.",
        }

    try:
        os.kill(int(pid), signal.SIGTERM)
    except OSError as e:
        return {"ok": False, "stopped": False, "running": True, "message": f"Failed to stop PID {pid}: {e}"}

    import time

    for _ in range(20):
        if not _is_port_open(host, port):
            break
        time.sleep(0.1)

    running = _is_port_open(host, port)
    if not running:
        PATHMEASURE_PROCS.pop(key, None)
    return {
        "ok": not running,
        "stopped": not running,
        "running": running,
        "host": host,
        "port": port,
        "pid": pid,
        "url": f"http://{host}:{port}",
    }

