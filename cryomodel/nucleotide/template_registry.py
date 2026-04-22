"""Template registry and validation for interactive BaseHunter template packs."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class TemplateEntry:
    """A single template file declaration parsed from ``templates.txt``."""

    filename: str
    path: Path
    threshold: Optional[float]
    description: str


@dataclass(frozen=True)
class TemplateValidationResult:
    """Structured template-pack validation status."""

    root: Path
    metadata_file: Optional[Path]
    entries: Tuple[TemplateEntry, ...]
    missing_files: Tuple[str, ...]
    warnings: Tuple[str, ...]

    @property
    def is_valid(self) -> bool:
        return len(self.entries) > 0 and len(self.missing_files) == 0

    def suggested_thresholds(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for ent in self.entries:
            if ent.threshold is not None:
                out[ent.filename] = ent.threshold
        return out


_LINE_RE = re.compile(r"^\s*([A-Za-z0-9._-]+\.(?:mrc|map|ccp4|pdb|cif))\s*:\s*(.+?)\s*$")
_THRESH_RE = re.compile(r"(?:~|approx(?:\.|imately)?\s*)\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)


def _parse_threshold(description: str) -> Optional[float]:
    m = _THRESH_RE.search(description)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def parse_templates_txt(metadata_path: Path, root: Optional[Path] = None) -> List[TemplateEntry]:
    """Parse template file declarations from ``templates.txt`` style metadata."""
    root_dir = (root or metadata_path.parent).resolve()
    entries: List[TemplateEntry] = []
    for raw in metadata_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = _LINE_RE.match(line)
        if not m:
            continue
        filename = m.group(1).strip()
        desc = m.group(2).strip()
        entries.append(
            TemplateEntry(
                filename=filename,
                path=root_dir / filename,
                threshold=_parse_threshold(desc),
                description=desc,
            )
        )
    return entries


def validate_template_pack(
    root: Path,
    required_files: Optional[Iterable[str]] = None,
) -> TemplateValidationResult:
    """Validate a template pack directory and report missing assets/warnings."""
    root = root.resolve()
    metadata = root / "templates.txt"
    warnings: List[str] = []
    entries: List[TemplateEntry] = []
    missing: List[str] = []

    if metadata.is_file():
        entries = parse_templates_txt(metadata, root=root)
        for ent in entries:
            if not ent.path.exists():
                missing.append(ent.filename)
    else:
        warnings.append("templates.txt not found; using directory scan only.")
        for p in sorted(root.glob("*")):
            if p.suffix.lower() in (".mrc", ".map", ".ccp4", ".pdb", ".cif"):
                entries.append(
                    TemplateEntry(
                        filename=p.name,
                        path=p,
                        threshold=None,
                        description="Discovered by extension without metadata.",
                    )
                )

    if required_files:
        seen = {e.filename for e in entries}
        for name in required_files:
            if name not in seen:
                missing.append(str(name))

    if not entries:
        warnings.append("No template entries discovered.")

    # Preserve order and remove duplicates in missing list.
    dedup_missing: List[str] = []
    for name in missing:
        if name not in dedup_missing:
            dedup_missing.append(name)

    return TemplateValidationResult(
        root=root,
        metadata_file=metadata if metadata.is_file() else None,
        entries=tuple(entries),
        missing_files=tuple(dedup_missing),
        warnings=tuple(warnings),
    )

