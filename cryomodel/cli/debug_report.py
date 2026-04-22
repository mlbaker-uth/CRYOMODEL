"""Collect a local diagnostic bundle for support and troubleshooting."""
from __future__ import annotations

import importlib
import importlib.metadata as im
import json
import os
import platform
import shutil
import site
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import typer

SENSITIVE_ENV_TOKENS = (
    "TOKEN",
    "SECRET",
    "PASSWORD",
    "PASS",
    "KEY",
    "CREDENTIAL",
    "AUTH",
    "COOKIE",
)


def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _ok(name: str, details: str = "") -> Dict[str, Any]:
    return {"name": name, "status": "ok", "details": details}


def _warn(name: str, details: str = "") -> Dict[str, Any]:
    return {"name": name, "status": "warn", "details": details}


def _fail(name: str, details: str = "") -> Dict[str, Any]:
    return {"name": name, "status": "fail", "details": details}


def _sanitize_env(env: Dict[str, str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for k, v in env.items():
        upper = k.upper()
        if any(tok in upper for tok in SENSITIVE_ENV_TOKENS):
            out[k] = "<redacted>"
        else:
            out[k] = str(v)
    return out


def _safe_run(cmd: List[str], timeout_s: int = 8) -> Dict[str, Any]:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s, check=False)
    except Exception as e:
        return {"ok": False, "error": str(e), "cmd": cmd}
    return {
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "stdout": (proc.stdout or "").strip(),
        "stderr": (proc.stderr or "").strip(),
        "cmd": cmd,
    }


def _dep_checks() -> List[Dict[str, Any]]:
    checks = []
    for pkg in ["numpy", "scipy", "mrcfile", "gemmi", "sklearn", "skimage", "yaml", "typer"]:
        try:
            importlib.import_module(pkg)
            checks.append(_ok(f"import:{pkg}"))
        except Exception as e:
            checks.append(_fail(f"import:{pkg}", str(e)))

    for pkg in ["torch", "ortools"]:
        try:
            importlib.import_module(pkg)
            checks.append(_ok(f"optional:{pkg}"))
        except Exception as e:
            checks.append(_warn(f"optional:{pkg}", str(e)))
    return checks


def _tool_checks() -> List[Dict[str, Any]]:
    checks = []
    py = sys.executable
    probes = [
        ("python_version", [py, "--version"]),
        ("cryomodel_help", [py, "-m", "cryomodel.cli", "--help"]),
        ("cryomodel_version", [py, "-m", "cryomodel.cli", "version"]),
        ("workflow_validate_help", [py, "-m", "cryomodel.cli", "workflow-validate", "--help"]),
        ("validate_help", [py, "-m", "cryomodel.cli", "validate", "--help"]),
    ]
    for name, cmd in probes:
        res = _safe_run(cmd)
        if res.get("ok"):
            excerpt = (res.get("stdout") or res.get("stderr") or "").splitlines()
            checks.append(_ok(name, excerpt[0] if excerpt else "ok"))
        else:
            err = res.get("stderr") or res.get("error") or f"returncode={res.get('returncode')}"
            checks.append(_fail(name, str(err)))
    return checks


def _project_checks(project_root: Optional[Path], manifest_path: Optional[Path]) -> List[Dict[str, Any]]:
    checks = []
    if project_root is None:
        checks.append(_warn("project_root", "not provided"))
    else:
        p = project_root.expanduser().resolve()
        if p.is_dir():
            checks.append(_ok("project_root_exists", str(p)))
            try:
                test = p / ".cryomodel_write_test.tmp"
                test.write_text("ok", encoding="utf-8")
                test.unlink(missing_ok=True)
                checks.append(_ok("project_root_writable"))
            except Exception as e:
                checks.append(_warn("project_root_writable", str(e)))
        else:
            checks.append(_fail("project_root_exists", str(p)))

    if manifest_path is None:
        checks.append(_warn("manifest_path", "not provided"))
    else:
        m = manifest_path.expanduser().resolve()
        checks.append(_ok("manifest_exists", str(m)) if m.is_file() else _warn("manifest_exists", str(m)))
    return checks


def _system_info() -> Dict[str, Any]:
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "python": {
            "executable": sys.executable,
            "version": sys.version.replace("\n", " "),
            "prefix": sys.prefix,
            "base_prefix": getattr(sys, "base_prefix", ""),
            "venv_active": bool(os.environ.get("VIRTUAL_ENV")),
            "site_packages": site.getsitepackages() if hasattr(site, "getsitepackages") else [],
        },
        "cryomodel_version": im.version("cryomodel") if _dist_installed("cryomodel") else "unknown",
        "paths": {
            "cwd": str(Path.cwd()),
            "home": str(Path.home()),
            "path_entries": os.environ.get("PATH", "").split(os.pathsep),
        },
    }


def _dist_installed(name: str) -> bool:
    try:
        im.version(name)
        return True
    except Exception:
        return False


def _write_text_report(path: Path, payload: Dict[str, Any]) -> None:
    lines: List[str] = []
    lines.append("CryoModel Debug Report")
    lines.append("=" * 22)
    lines.append(f"Generated: {payload.get('system', {}).get('timestamp', 'unknown')}")
    lines.append("")
    lines.append("Summary")
    lines.append("-" * 7)
    summary = payload.get("summary", {})
    lines.append(f"ok={summary.get('ok', 0)} warn={summary.get('warn', 0)} fail={summary.get('fail', 0)}")
    lines.append("")
    lines.append("Checks")
    lines.append("-" * 6)
    for section_name in ["dependency_checks", "tool_checks", "project_checks"]:
        lines.append(f"[{section_name}]")
        for item in payload.get(section_name, []):
            status = item.get("status", "unknown").upper()
            detail = item.get("details", "")
            lines.append(f"- {status:4} {item.get('name', 'unknown')}{(': ' + detail) if detail else ''}")
        lines.append("")
    lines.append("Artifacts")
    lines.append("-" * 9)
    for k, v in payload.get("artifacts", {}).items():
        lines.append(f"- {k}: {v}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _summarize(checks: List[Dict[str, Any]]) -> Dict[str, int]:
    out = {"ok": 0, "warn": 0, "fail": 0}
    for c in checks:
        s = str(c.get("status", "warn"))
        if s not in out:
            s = "warn"
        out[s] += 1
    return out


def generate(
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir",
        help="Directory where report bundle will be written (default: ./debug_reports).",
    ),
    project_root: Optional[Path] = typer.Option(None, "--project-root", help="Optional project path to validate."),
    manifest_path: Optional[Path] = typer.Option(None, "--manifest-path", help="Optional manifest path to validate."),
    include_freeze: bool = typer.Option(True, "--include-freeze/--no-freeze", help="Include pip freeze snapshot."),
) -> None:
    """Create a diagnostics bundle for troubleshooting and support."""
    base = (output_dir or (Path.cwd() / "debug_reports")).expanduser().resolve()
    stamp = _now_stamp()
    report_dir = base / f"cryomodel_debug_{stamp}"
    report_dir.mkdir(parents=True, exist_ok=True)

    system = _system_info()
    dep_checks = _dep_checks()
    tool_checks = _tool_checks()
    proj_checks = _project_checks(project_root, manifest_path)
    all_checks = dep_checks + tool_checks + proj_checks
    summary = _summarize(all_checks)

    artifacts: Dict[str, str] = {}

    env_path = report_dir / "env.json"
    env_payload = _sanitize_env(dict(os.environ))
    env_path.write_text(json.dumps(env_payload, indent=2, sort_keys=True), encoding="utf-8")
    artifacts["env_json"] = str(env_path)

    if include_freeze:
        freeze_path = report_dir / "pip_freeze.txt"
        pip_cmd = shutil.which("pip") or sys.executable
        if pip_cmd == sys.executable:
            res = _safe_run([sys.executable, "-m", "pip", "freeze"], timeout_s=20)
        else:
            res = _safe_run([pip_cmd, "freeze"], timeout_s=20)
        if res.get("ok"):
            freeze_path.write_text((res.get("stdout") or "") + "\n", encoding="utf-8")
            artifacts["pip_freeze"] = str(freeze_path)
        else:
            artifacts["pip_freeze"] = f"failed: {res.get('stderr') or res.get('error') or res.get('returncode')}"

    payload = {
        "system": system,
        "dependency_checks": dep_checks,
        "tool_checks": tool_checks,
        "project_checks": proj_checks,
        "summary": summary,
        "artifacts": artifacts,
    }

    json_path = report_dir / "report.json"
    txt_path = report_dir / "debug_report.txt"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    _write_text_report(txt_path, payload)

    typer.echo(f"Debug report written to: {report_dir}")
    typer.echo(f"- Summary: ok={summary['ok']} warn={summary['warn']} fail={summary['fail']}")
    typer.echo(f"- Text report: {txt_path}")
    typer.echo(f"- JSON report: {json_path}")
