"""Build CryoModel workflow manifest JSON from open ChimeraX models (phase 1: disk paths preferred)."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

SCHEMA_VERSION = 1


def _disk_path_for_model(m) -> Optional[str]:
    """Return absolute filesystem path if the model was opened from a local file."""
    def _check_path(candidate: Any) -> Optional[str]:
        if not candidate:
            return None
        try:
            s = str(candidate)
        except Exception:
            return None
        if os.path.isfile(s):
            return os.path.abspath(s)
        return None

    # ChimeraX map/volume objects often keep their original file under a variety
    # of attributes depending on how the volume was opened.
    oa = getattr(m, "opened_data", None)
    if oa is not None:
        for attr in ("path", "filename", "file_name"):
            v = getattr(oa, attr, None)
            p = _check_path(v)
            if p:
                return p

    oa = getattr(m, "openedAs", None) or getattr(m, "opened_as", None)
    if oa is not None:
        # Some ChimeraX versions store openedAs as (path, ...) or similar.
        if isinstance(oa, (tuple, list)):
            for cand in oa:
                p = _check_path(cand)
                if p:
                    return p
        else:
            p = _check_path(oa)
            if p:
                return p

    # Many Volume-like models expose a `.data` object carrying file metadata.
    data = getattr(m, "data", None)
    if data is not None:
        for attr in ("path", "filename", "file_name"):
            p = _check_path(getattr(data, attr, None))
            if p:
                return p

    # Finally try common direct attributes on the model itself.
    for attr in (
        "filename",
        "file_name",
        "fileName",
        "path",
        "file",
        "data_file",
        "source_path",
    ):
        p = _check_path(getattr(m, attr, None))
        if p:
            return p

    return None


def _kind_for_model(m) -> str:
    try:
        from chimerax.map import Volume

        if isinstance(m, Volume):
            return "map"
    except Exception:
        pass
    try:
        from chimerax.atomic import AtomicStructure

        if isinstance(m, AtomicStructure):
            return "structure"
    except Exception:
        pass
    cls = getattr(m, "__class__", type(m)).__name__
    if "Volume" in cls:
        return "map"
    return "other"


def _format_hint(path: Optional[str]) -> str:
    if not path:
        return ""
    suf = Path(path).suffix.lower().lstrip(".")
    return suf or ""


def _artifact_type(kind: str) -> str:
    if kind == "map":
        return "map.mrc"
    if kind == "structure":
        return "model.structure"
    return "other"


def build_manifest_entries(session) -> List[Dict[str, Any]]:
    """Inspect open models and return manifest entry dicts."""
    entries: List[Dict[str, Any]] = []
    models = []
    try:
        for m in session.models:
            models.append(m)
    except Exception:
        try:
            models = list(session.models.list())
        except Exception:
            models = []

    for m in models:
        label = getattr(m, "name", None) or getattr(m, "__class__", type(m)).__name__
        mid = str(getattr(m, "id_string", None) or getattr(m, "id", None) or label)
        kind = _kind_for_model(m)
        path = _disk_path_for_model(m)
        source = "disk" if path else "session"
        ent: Dict[str, Any] = {
            "id": mid,
            "label": str(label),
            "kind": kind,
            "format_hint": _format_hint(path),
            "path": path,
            "source": source,
            "artifact_type": _artifact_type(kind),
        }
        if not path:
            ent["reason_no_path"] = "not_saved_or_not_a_local_file"
        entries.append(ent)
    return entries


def write_manifest(session, output_path: Path) -> Path:
    """Write manifest JSON; returns resolved path."""
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "entries": build_manifest_entries(session),
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def log_manifest_summary(session, entries: List[Dict[str, Any]], output_path: Optional[Path] = None) -> None:
    """Write a concise per-entry manifest summary to the ChimeraX log."""
    prefix = "[CryoModel Manifest]"
    if output_path is not None:
        session.logger.info(f"{prefix} Output: {Path(output_path).expanduser().resolve()}")
    if not entries:
        session.logger.info(f"{prefix} No open models found.")
        return
    session.logger.info(f"{prefix} {len(entries)} open model(s) detected:")
    for ent in entries:
        model_id = ent.get("id", "?")
        label = ent.get("label", "unknown")
        kind = ent.get("kind", "other")
        src = ent.get("source", "session")
        p = ent.get("path")
        if p:
            session.logger.info(f"{prefix}  #{model_id} {label} ({kind}, {src}) -> {p}")
        else:
            reason = ent.get("reason_no_path", "no_path")
            session.logger.info(f"{prefix}  #{model_id} {label} ({kind}, {src}) -> <no path: {reason}>")
