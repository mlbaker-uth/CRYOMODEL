"""Detect and optionally stop CryoModel workflow / manager API listeners."""

from __future__ import annotations

import getpass
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

# Default CryoModel UI ports (workflow-ui / manager API).
DEFAULT_WORKFLOW_PORT = 8010
DEFAULT_MANAGER_PORT = 8011

_CRYOMODEL_CMD_MARKERS = (
    "cryomodel.cli",
    "cryomodel workflow-ui",
    "cryomodel manager",
    "cryomodel pathmeasure",
    "cryomodel.workflow.ui_api",
    "cryomodel.workflow.manager_api",
    "cryomodel.pathmeasure.api",
)


@dataclass(frozen=True)
class PortListener:
    port: int
    pid: int
    user: str
    command: str
    service: str
    cryomodel: bool
    address: str = ""


@dataclass
class KillResult:
    pid: int
    port: int
    ok: bool
    message: str


def _current_user() -> str:
    try:
        return getpass.getuser()
    except Exception:
        return os.environ.get("USER") or os.environ.get("USERNAME") or ""


def _classify_service(command: str, port: int) -> Tuple[str, bool]:
    cmd = (command or "").lower()
    if any(m in cmd for m in _CRYOMODEL_CMD_MARKERS):
        if "manager_api" in cmd or "manager serve" in cmd or (
            port == DEFAULT_MANAGER_PORT and "manager" in cmd
        ):
            return "manager-api", True
        if "pathmeasure" in cmd:
            return "pathmeasure", True
        if "ui_api" in cmd or "workflow-ui" in cmd or port == DEFAULT_WORKFLOW_PORT:
            return "workflow-ui", True
        return "cryomodel", True
    if "uvicorn" in cmd and "cryomodel" in cmd:
        if "manager_api" in cmd:
            return "manager-api", True
        if "ui_api" in cmd:
            return "workflow-ui", True
        return "cryomodel", True
    if port == DEFAULT_MANAGER_PORT:
        return "unknown (8011)", False
    if port == DEFAULT_WORKFLOW_PORT:
        return "unknown (8010)", False
    return "unknown", False


def _ps_command(pid: int) -> Tuple[str, str]:
    """Return (user, full command line) for pid, or ('', '') if unavailable."""
    if sys.platform == "win32":
        r = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        line = (r.stdout or "").strip().splitlines()
        if not line or "No tasks" in line[0]:
            return "", ""
        parts = line[0].split(",")
        if len(parts) >= 2:
            name = parts[0].strip('"')
            return _current_user(), name
        return "", ""

    col = "command=" if sys.platform == "darwin" else "args="
    r = subprocess.run(
        ["ps", "-p", str(pid), "-o", f"user=,{col}"],
        capture_output=True,
        text=True,
        timeout=5,
    )
    line = (r.stdout or "").strip()
    if not line:
        return "", ""
    parts = line.split(None, 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    if len(parts) == 1:
        return parts[0], ""
    return "", ""


def _lsof_listeners(port: int) -> List[Tuple[int, str, str]]:
    """Return [(pid, user, address), ...] for TCP listeners on port."""
    lsof = shutil.which("lsof")
    if not lsof:
        return []
    r = subprocess.run(
        [lsof, "-nP", f"-iTCP:{port}", "-sTCP:LISTEN"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    if r.returncode != 0:
        return []
    out: List[Tuple[int, str, str]] = []
    for line in (r.stdout or "").splitlines()[1:]:
        parts = line.split()
        if len(parts) < 9:
            continue
        try:
            pid = int(parts[1])
        except ValueError:
            continue
        user = parts[2]
        addr = parts[-2] if parts[-1] == "(LISTEN)" else parts[-1]
        out.append((pid, user, addr))
    return out


def _netstat_listeners(port: int) -> List[Tuple[int, str, str]]:
    """Windows fallback: [(pid, user, address), ...]."""
    r = subprocess.run(
        ["netstat", "-ano"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    if r.returncode != 0:
        return []
    out: List[Tuple[int, str, str]] = []
    pat = re.compile(rf":{port}\s")
    for line in (r.stdout or "").splitlines():
        if "LISTENING" not in line.upper():
            continue
        if not pat.search(line):
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            pid = int(parts[-1])
        except ValueError:
            continue
        addr = parts[1] if len(parts) > 1 else ""
        out.append((pid, _current_user(), addr))
    return out


def find_port_listeners(port: int) -> List[PortListener]:
    """All TCP listeners on ``port`` with CryoModel classification."""
    raw = _lsof_listeners(port)
    if not raw and sys.platform == "win32":
        raw = _netstat_listeners(port)

    seen: Set[int] = set()
    listeners: List[PortListener] = []
    for pid, lsof_user, addr in raw:
        if pid in seen:
            continue
        seen.add(pid)
        ps_user, cmd = _ps_command(pid)
        user = ps_user or lsof_user
        command = cmd or "(command unavailable)"
        service, cryo = _classify_service(command, port)
        listeners.append(
            PortListener(
                port=port,
                pid=pid,
                user=user,
                command=command,
                service=service,
                cryomodel=cryo,
                address=addr,
            )
        )
    return listeners


def collect_registry_ports(
    *,
    projects_file=None,
    sessions_file=None,
    default_workflow: int = DEFAULT_WORKFLOW_PORT,
    default_manager: int = DEFAULT_MANAGER_PORT,
) -> Set[int]:
    """Ports referenced in ~/.cryomodel projects/sessions plus defaults."""
    from pathlib import Path

    import json

    ports: Set[int] = {int(default_workflow), int(default_manager)}
    reg = Path.home() / ".cryomodel"
    pf = Path(projects_file) if projects_file else reg / "projects.json"
    sf = Path(sessions_file) if sessions_file else reg / "sessions.json"

    def _load(path: Path):
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    projects = _load(pf)
    if isinstance(projects, list):
        for p in projects:
            if isinstance(p, dict) and p.get("api_port") is not None:
                try:
                    ports.add(int(p["api_port"]))
                except (TypeError, ValueError):
                    pass
    elif isinstance(projects, dict):
        inner = projects.get("projects")
        if isinstance(inner, list):
            for p in inner:
                if isinstance(p, dict) and p.get("api_port") is not None:
                    try:
                        ports.add(int(p["api_port"]))
                    except (TypeError, ValueError):
                        pass

    sessions = _load(sf)
    if isinstance(sessions, dict):
        proj = sessions.get("projects")
        if isinstance(proj, dict):
            for s in proj.values():
                if isinstance(s, dict) and s.get("port") is not None:
                    try:
                        ports.add(int(s["port"]))
                    except (TypeError, ValueError):
                        pass
    return ports


def scan_listeners(ports: Iterable[int]) -> List[PortListener]:
    """Scan ports and return listeners sorted by port, pid."""
    all_listeners: List[PortListener] = []
    for port in sorted({int(p) for p in ports}):
        all_listeners.extend(find_port_listeners(port))
    return sorted(all_listeners, key=lambda x: (x.port, x.pid))


def format_listener_line(item: PortListener, *, me: str) -> str:
    owner = "you" if item.user == me else f"user {item.user!r}"
    kind = item.service if item.cryomodel else f"non-CryoModel ({item.service})"
    cmd = item.command if len(item.command) <= 120 else item.command[:117] + "..."
    addr = f" @ {item.address}" if item.address else ""
    return (
        f"  port {item.port}{addr}: PID {item.pid} ({owner}) — {kind}\n"
        f"    {cmd}"
    )


def kill_instructions_for_listener(item: PortListener, *, me: str) -> List[str]:
    lines: List[str] = []
    if item.user == me:
        lines.append(f"kill {item.pid}          # graceful (SIGTERM)")
        lines.append(f"kill -9 {item.pid}       # force if still listening")
        if sys.platform == "darwin":
            lines.append(f"kill -TERM {item.pid}   # same as kill {item.pid}")
    else:
        lines.append(
            f"# Port {item.port} is held by another macOS/Linux user ({item.user!r}). "
            "Switch to that account and run:"
        )
        lines.append(f"  cryomodel manager cleanup --kill --yes")
        lines.append("  # or from any account with permission:")
        lines.append(f"  sudo kill {item.pid}")
    if shutil.which("lsof"):
        lines.append(f"lsof -nP -iTCP:{item.port} -sTCP:LISTEN")
    return lines


def format_cleanup_report(
    listeners: Sequence[PortListener],
    *,
    ports_scanned: Sequence[int],
    me: Optional[str] = None,
) -> str:
    me = me or _current_user()
    if not shutil.which("lsof") and sys.platform != "win32":
        return (
            "Cannot inspect listening ports: `lsof` not found on PATH.\n"
            "Install lsof or run manually, e.g.:\n"
            f"  lsof -nP -iTCP:{DEFAULT_WORKFLOW_PORT} -sTCP:LISTEN\n"
            f"  lsof -nP -iTCP:{DEFAULT_MANAGER_PORT} -sTCP:LISTEN"
        )

    lines = [f"Scanned ports: {', '.join(str(p) for p in sorted(set(ports_scanned)))}"]
    if not listeners:
        lines.append("No TCP listeners found on those ports.")
        return "\n".join(lines)

    lines.append(f"Found {len(listeners)} listener(s):")
    for item in listeners:
        lines.append(format_listener_line(item, me=me))
    return "\n".join(lines)


def format_kill_help(listeners: Sequence[PortListener], *, me: Optional[str] = None) -> str:
    me = me or _current_user()
    if not listeners:
        return ""
    blocks: List[str] = ["To free these ports manually:"]
    for item in listeners:
        blocks.append(f"\nPort {item.port}, PID {item.pid}:")
        blocks.extend(f"  {ln}" for ln in kill_instructions_for_listener(item, me=me))
    blocks.append("\nOr run: cryomodel manager cleanup --kill --yes")
    return "\n".join(blocks)


def _pid_listening_on_port(pid: int, port: int) -> bool:
    return any(x.pid == pid for x in find_port_listeners(port))


def kill_listeners(
    listeners: Sequence[PortListener],
    *,
    current_user_only: bool = True,
    cryomodel_only: bool = True,
    me: Optional[str] = None,
    term_timeout_s: float = 2.5,
) -> List[KillResult]:
    """Send SIGTERM (then SIGKILL if needed). Returns one result per attempted pid."""
    me = me or _current_user()
    results: List[KillResult] = []
    for item in listeners:
        if cryomodel_only and not item.cryomodel:
            results.append(
                KillResult(
                    pid=item.pid,
                    port=item.port,
                    ok=False,
                    message="skipped (not classified as CryoModel)",
                )
            )
            continue
        if current_user_only and item.user != me:
            results.append(
                KillResult(
                    pid=item.pid,
                    port=item.port,
                    ok=False,
                    message=f"skipped (owned by {item.user!r}, not {me!r})",
                )
            )
            continue
        try:
            os.kill(item.pid, signal.SIGTERM)
        except ProcessLookupError:
            results.append(KillResult(item.pid, item.port, True, "already exited"))
            continue
        except PermissionError:
            results.append(
                KillResult(
                    item.pid,
                    item.port,
                    False,
                    f"permission denied (try: sudo kill {item.pid})",
                )
            )
            continue

        deadline = time.monotonic() + term_timeout_s
        while time.monotonic() < deadline:
            if not _pid_listening_on_port(item.pid, item.port):
                results.append(KillResult(item.pid, item.port, True, "stopped (SIGTERM)"))
                break
            time.sleep(0.15)
        else:
            try:
                os.kill(item.pid, signal.SIGKILL)
                results.append(KillResult(item.pid, item.port, True, "stopped (SIGKILL)"))
            except ProcessLookupError:
                results.append(KillResult(item.pid, item.port, True, "already exited"))
            except PermissionError:
                results.append(
                    KillResult(
                        item.pid,
                        item.port,
                        False,
                        f"SIGTERM sent but still listening; permission denied for SIGKILL",
                    )
                )
    return results


def clear_stale_session_pids(killed_pids: Set[int], *, sessions_file=None) -> int:
    """Mark sessions dead when their recorded workflow PID was killed."""
    if not killed_pids:
        return 0
    from pathlib import Path

    import json

    sf = Path(sessions_file) if sessions_file else Path.home() / ".cryomodel" / "sessions.json"
    if not sf.is_file():
        return 0
    try:
        data = json.loads(sf.read_text(encoding="utf-8"))
    except Exception:
        return 0
    if not isinstance(data, dict):
        return 0
    projects = data.get("projects")
    if not isinstance(projects, dict):
        return 0
    changed = 0
    for sess in projects.values():
        if not isinstance(sess, dict):
            continue
        pid = sess.get("pid")
        if pid is not None and int(pid) in killed_pids:
            sess["running"] = False
            sess["pid"] = None
            changed += 1
    if changed:
        sf.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return changed


def run_cleanup(
    *,
    ports: Optional[Sequence[int]] = None,
    kill: bool = False,
    yes: bool = False,
    all_listeners: bool = False,
    all_users: bool = False,
    projects_file=None,
    sessions_file=None,
) -> Dict[str, object]:
    """
    Scan ports, print report, optionally kill CryoModel listeners.

    Returns summary dict for tests / CLI.
    """
    if ports is None:
        port_set = collect_registry_ports(
            projects_file=projects_file,
            sessions_file=sessions_file,
        )
    else:
        port_set = set(int(p) for p in ports)

    listeners = scan_listeners(port_set)
    me = _current_user()
    report = format_cleanup_report(listeners, ports_scanned=sorted(port_set), me=me)
    help_text = format_kill_help(listeners, me=me) if listeners else ""

    summary: Dict[str, object] = {
        "ports": sorted(port_set),
        "listeners": listeners,
        "report": report,
        "kill_help": help_text,
        "killed": [],
    }

    if not kill:
        return summary

    to_kill = listeners if all_listeners else [x for x in listeners if x.cryomodel]
    if not to_kill:
        summary["message"] = "Nothing to kill."
        return summary

    if not yes:
        summary["message"] = "Confirmation required (--yes to kill without prompt)."
        summary["needs_confirmation"] = True
        return summary

    results = kill_listeners(
        to_kill,
        current_user_only=not all_users,
        cryomodel_only=not all_listeners,
    )
    summary["killed"] = results
    ok_pids = {r.pid for r in results if r.ok}
    summary["sessions_cleared"] = clear_stale_session_pids(ok_pids, sessions_file=sessions_file)
    return summary
