"""Append-only project-local activity log for workflow UI runs (Phase P0)."""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore


ACTIVITY_FILENAME = "activity.jsonl"
RUNS_SUBDIR = "runs"
DOT_DIR = ".cryomodel"


def _local_timestamp() -> str:
    """ISO-8601 local time (seconds precision), no TZ offset — per SESSION_LAUNCHER_INFRA_PLAN."""
    return datetime.now().isoformat(timespec="seconds")


def activity_jsonl_path(project_root: Path) -> Path:
    return project_root.expanduser().resolve() / DOT_DIR / ACTIVITY_FILENAME


def _safe_run_id_for_filename(run_id: str) -> str:
    """Avoid path components / odd filenames in run_id."""
    cleaned = "".join(c if c.isalnum() or c in "._-" else "_" for c in (run_id or "").strip())
    cleaned = cleaned.replace("..", "_").strip("._") or "unknown"
    return cleaned[:200]


def run_log_path(project_root: Path, run_id: str) -> Path:
    return project_root.expanduser().resolve() / DOT_DIR / RUNS_SUBDIR / f"{_safe_run_id_for_filename(run_id)}.log"


def append_activity_line(project_root: Path, record: Dict[str, Any]) -> None:
    """Append one JSON line to project_root/.cryomodel/activity.jsonl (creates parents)."""
    path = activity_jsonl_path(project_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False) + "\n"
    with open(path, "a", encoding="utf-8") as f:
        if fcntl is not None:
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            except OSError:
                pass
        try:
            f.write(line)
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                pass
        finally:
            if fcntl is not None:
                try:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                except OSError:
                    pass


def write_run_text_log(project_root: Path, run_id: str, text: str) -> Path:
    """Write full stdout/stderr capture for one UI run."""
    path = run_log_path(project_root, run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def persist_workflow_ui_run(
    cwd: str,
    *,
    run_id: str,
    card_id: str,
    command: str,
    status: str,
    return_code: Optional[int],
    log_text: str,
    started_perf: float,
) -> None:
    """
    Write .cryomodel/runs/<run_id>.log and append a summary line to activity.jsonl.

    Uses cwd as project root (must exist). Errors are propagated to caller.
    """
    project_root = Path(cwd).expanduser().resolve()
    if not project_root.is_dir():
        raise FileNotFoundError(f"project cwd is not a directory: {project_root}")

    run_file = write_run_text_log(project_root, run_id, log_text)
    duration_s = round(time.perf_counter() - started_perf, 4)
    rel_run_log = str(run_file.relative_to(project_root))

    record: Dict[str, Any] = {
        "timestamp": _local_timestamp(),
        "source": "workflow_ui",
        "run_id": run_id,
        "card_id": card_id,
        "command": command,
        "cwd": str(project_root),
        "status": status,
        "return_code": return_code,
        "duration_s": duration_s,
        "output_log": rel_run_log,
    }
    append_activity_line(project_root, record)
