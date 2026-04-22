"""Paths to assets shipped inside the ``cryomodel`` package tree."""

from __future__ import annotations

from pathlib import Path


def basehunter_template_pack_dir() -> Path:
    """Directory containing the bundled BaseHunter DNA template pack (``NEW-DNA-TEMPLATES``)."""
    import cryomodel  # noqa: PLC0415 — resolve after package layout exists

    return Path(cryomodel.__file__).resolve().parent / "data" / "basehunter" / "NEW-DNA-TEMPLATES"
