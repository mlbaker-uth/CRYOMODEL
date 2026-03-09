"""Command logging utilities for CryoModel CLI."""
from __future__ import annotations

import json
import os
import shlex
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional, Any

LOG_FILENAME = ".crymodel_history.jsonl"
LOG_OUTPUT_DIR = ".crymodel_logs"


class _Tee:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


def _safe_timestamp() -> str:
    # Use local time for human readability and make it filesystem-safe.
    return datetime.now().isoformat(timespec="seconds").replace(":", "-")


def _command_string(argv: list[str]) -> str:
    return " ".join(shlex.quote(arg) for arg in argv)


def log_command(command_name: Optional[str] = None) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator to log command execution, stdout/stderr, and status."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(*args, **kwargs):
            start = time.time()
            cwd = os.getcwd()
            argv = sys.argv[:]
            command = _command_string(argv)
            tool = command_name or func.__name__

            output_dir = Path(cwd) / LOG_OUTPUT_DIR
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{_safe_timestamp()}_{tool.replace(' ', '_')}.log"

            status = "success"
            exit_code = None
            error = None
            tb = None

            original_stdout = sys.stdout
            original_stderr = sys.stderr
            out_file = open(output_path, "w", encoding="utf-8")
            try:
                sys.stdout = _Tee(original_stdout, out_file)
                sys.stderr = _Tee(original_stderr, out_file)
                return func(*args, **kwargs)
            except SystemExit as exc:
                status = "exit"
                exit_code = exc.code
                raise
            except Exception as exc:
                status = "error"
                error = f"{exc.__class__.__name__}: {exc}"
                tb = traceback.format_exc()
                raise
            finally:
                duration_s = round(time.time() - start, 4)
                sys.stdout = original_stdout
                sys.stderr = original_stderr
                out_file.close()

                record = {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "cwd": cwd,
                    "command": command,
                    "argv": argv,
                    "tool": tool,
                    "status": status,
                    "exit_code": exit_code,
                    "duration_s": duration_s,
                    "output_log": str(output_path),
                }
                if error:
                    record["error"] = error
                if tb:
                    record["traceback"] = tb

                log_path = Path(cwd) / LOG_FILENAME
                with open(log_path, "a", encoding="utf-8") as log_file:
                    log_file.write(json.dumps(record) + "\n")

        return wrapper

    return decorator
