"""CLI tools for inspecting CryoModel command history."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Iterable

import typer

from .command_log import LOG_FILENAME
from .command_log import log_command

app = typer.Typer(no_args_is_help=True, help="Inspect CryoModel command history")


def _load_history(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def _parse_since(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        raise typer.BadParameter("Invalid --since format. Use ISO-8601, e.g. 2026-03-06T14:00:00")


def _filter_records(
    records: Iterable[dict],
    tool: Optional[str],
    status: Optional[str],
    since: Optional[datetime],
    command_contains: Optional[str],
    arg_contains: Optional[str],
    file_contains: Optional[str],
) -> list[dict]:
    filtered = []
    for record in records:
        if tool and record.get("tool") != tool:
            continue
        if status and record.get("status") != status:
            continue
        if command_contains:
            command = record.get("command", "")
            if command_contains not in command:
                continue
        if arg_contains:
            argv = record.get("argv") or []
            if not any(arg_contains in str(arg) for arg in argv):
                continue
        if file_contains:
            argv = record.get("argv") or []
            found = False
            for arg in argv:
                arg_str = str(arg)
                if "/" in arg_str or "." in arg_str:
                    if file_contains in arg_str:
                        found = True
                        break
            if not found:
                continue
        if since:
            ts = record.get("timestamp")
            if not ts:
                continue
            try:
                if datetime.fromisoformat(ts) < since:
                    continue
            except ValueError:
                continue
        filtered.append(record)
    return filtered


def _format_record(record: dict) -> str:
    parts = [
        f"{record.get('timestamp', 'unknown')}",
        f"{record.get('tool', 'unknown')}",
        f"{record.get('status', 'unknown')}",
        f"{record.get('duration_s', 'n/a')}s",
        record.get("command", ""),
    ]
    return " | ".join(part for part in parts if part)


def _read_output_tail(path: Path, max_lines: int) -> str:
    if max_lines <= 0 or not path.exists():
        return ""
    with open(path, "r", encoding="utf-8") as handle:
        lines = handle.readlines()
    tail = lines[-max_lines:] if len(lines) > max_lines else lines
    return "".join(tail).rstrip()


@app.command()
@log_command("log show")
def show(
    cwd: Optional[Path] = typer.Option(None, "--cwd", help="Directory containing history file"),
    tool: Optional[str] = typer.Option(None, "--tool", help="Filter by tool name"),
    status: Optional[str] = typer.Option(None, "--status", help="Filter by status (success, error, exit)"),
    since: Optional[str] = typer.Option(None, "--since", help="ISO timestamp filter"),
    contains: Optional[str] = typer.Option(None, "--contains", help="Substring match in full command"),
    arg: Optional[str] = typer.Option(None, "--arg", help="Substring match in any argv entry"),
    file: Optional[str] = typer.Option(None, "--file", help="Substring match in any file-like argv"),
    limit: int = typer.Option(20, "--limit", help="Max records to show"),
    include_output: bool = typer.Option(False, "--include-output", help="Include tail of stdout/stderr"),
    output_lines: int = typer.Option(40, "--output-lines", help="Lines of output to show"),
    json_output: bool = typer.Option(False, "--json", help="Output JSON"),
):
    """Show recent CryoModel command history."""
    base = cwd or Path.cwd()
    history_path = base / LOG_FILENAME
    records = _load_history(history_path)
    if not records:
        typer.echo(f"No history found in {history_path}")
        raise typer.Exit(0)

    since_dt = _parse_since(since)
    filtered = _filter_records(records, tool, status, since_dt, contains, arg, file)
    if not filtered:
        typer.echo("No matching records.")
        raise typer.Exit(0)

    filtered = filtered[-limit:] if limit > 0 else filtered
    if json_output:
        if include_output:
            for record in filtered:
                output_path = Path(record["output_log"]) if record.get("output_log") else None
                if output_path:
                    record["output_tail"] = _read_output_tail(output_path, output_lines)
        typer.echo(json.dumps(filtered, indent=2))
        raise typer.Exit(0)
    for record in filtered:
        typer.echo(_format_record(record))
        if include_output and record.get("output_log"):
            output_path = Path(record["output_log"])
            tail = _read_output_tail(output_path, output_lines)
            if tail:
                typer.echo("---- output (tail) ----")
                typer.echo(tail)
                typer.echo("-----------------------")


@app.command()
@log_command("log tail")
def tail(
    cwd: Optional[Path] = typer.Option(None, "--cwd", help="Directory containing history file"),
    tool: Optional[str] = typer.Option(None, "--tool", help="Filter by tool name"),
    status: Optional[str] = typer.Option(None, "--status", help="Filter by status (success, error, exit)"),
    since: Optional[str] = typer.Option(None, "--since", help="ISO timestamp filter"),
    contains: Optional[str] = typer.Option(None, "--contains", help="Substring match in full command"),
    arg: Optional[str] = typer.Option(None, "--arg", help="Substring match in any argv entry"),
    file: Optional[str] = typer.Option(None, "--file", help="Substring match in any file-like argv"),
    lines: int = typer.Option(10, "--lines", help="Number of records to show"),
    json_output: bool = typer.Option(False, "--json", help="Output JSON"),
):
    """Show the most recent history entries."""
    base = cwd or Path.cwd()
    history_path = base / LOG_FILENAME
    records = _load_history(history_path)
    if not records:
        typer.echo(f"No history found in {history_path}")
        raise typer.Exit(0)
    since_dt = _parse_since(since)
    records = _filter_records(records, tool, status, since_dt, contains, arg, file)
    if not records:
        typer.echo("No matching records.")
        raise typer.Exit(0)
    records = records[-lines:] if lines > 0 else records
    if json_output:
        typer.echo(json.dumps(records, indent=2))
        raise typer.Exit(0)
    for record in records:
        typer.echo(_format_record(record))


@app.command()
@log_command("log stats")
def stats(
    cwd: Optional[Path] = typer.Option(None, "--cwd", help="Directory containing history file"),
    tool: Optional[str] = typer.Option(None, "--tool", help="Filter by tool name"),
    status: Optional[str] = typer.Option(None, "--status", help="Filter by status (success, error, exit)"),
    since: Optional[str] = typer.Option(None, "--since", help="ISO timestamp filter"),
    contains: Optional[str] = typer.Option(None, "--contains", help="Substring match in full command"),
    arg: Optional[str] = typer.Option(None, "--arg", help="Substring match in any argv entry"),
    file: Optional[str] = typer.Option(None, "--file", help="Substring match in any file-like argv"),
    json_output: bool = typer.Option(False, "--json", help="Output JSON"),
):
    """Summarize history by tool and status."""
    base = cwd or Path.cwd()
    history_path = base / LOG_FILENAME
    records = _load_history(history_path)
    if not records:
        typer.echo(f"No history found in {history_path}")
        raise typer.Exit(0)
    since_dt = _parse_since(since)
    records = _filter_records(records, tool, status, since_dt, contains, arg, file)
    if not records:
        typer.echo("No matching records.")
        raise typer.Exit(0)

    status_counts = Counter(r.get("status", "unknown") for r in records)
    tool_counts = Counter(r.get("tool", "unknown") for r in records)
    tool_status = defaultdict(Counter)
    for record in records:
        tool_status[record.get("tool", "unknown")][record.get("status", "unknown")] += 1

    if json_output:
        payload = {
            "status_counts": dict(status_counts),
            "tool_counts": dict(tool_counts),
            "tool_status": {tool: dict(counts) for tool, counts in tool_status.items()},
        }
        typer.echo(json.dumps(payload, indent=2))
        raise typer.Exit(0)

    typer.echo("Status counts:")
    for key, count in status_counts.most_common():
        typer.echo(f"  {key}: {count}")

    typer.echo("\nTool counts:")
    for key, count in tool_counts.most_common():
        typer.echo(f"  {key}: {count}")

    typer.echo("\nTool × Status:")
    for tool_name, counts in tool_status.items():
        detail = ", ".join(f"{status}={count}" for status, count in counts.items())
        typer.echo(f"  {tool_name}: {detail}")
