"""Lightweight API for UI-driven workflow job execution."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from threading import Lock, Thread
from typing import Dict, Optional
import shlex
import subprocess
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


RUNS: Dict[str, RunRecord] = {}
RUNS_LOCK = Lock()

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
    with RUNS_LOCK:
        rec = RUNS.get(run_id)
        if rec is None:
            return
        rec.return_code = int(return_code)
        rec.status = "success" if return_code == 0 else "error"
        rec.ended_at = datetime.now().isoformat(timespec="seconds")


def _worker(run_id: str) -> None:
    with RUNS_LOCK:
        rec = RUNS.get(run_id)
    if rec is None:
        return
    try:
        args = shlex.split(rec.command)
        proc = subprocess.Popen(
            args,
            cwd=rec.cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            _append_log(run_id, line)
        rc = proc.wait()
        _set_done(run_id, rc)
    except Exception as e:
        _append_log(run_id, f"\n[ui_api] failed to run command: {e}\n")
        _set_done(run_id, 1)


@app.get("/health")
def health():
    return {"status": "ok"}


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

