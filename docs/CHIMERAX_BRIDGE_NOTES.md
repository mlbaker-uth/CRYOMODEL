# ChimeraX ↔ CryoModel bridge (internal spec)

Design notes for connecting the workflow UI / CryoModel CLI to structures and maps that may already be open in ChimeraX—without unnecessary duplication of large density volumes.

## Goals

- Let users **“import from ChimeraX”** so inherited/manual paths in workflow cards can point at **real files** CryoModel can read.
- **Prefer existing filesystem paths** when ChimeraX opened a file from disk (no extra copy, critical for multi‑GB maps).
- When an open model/map **has no saved path** (built in session, fetched, etc.), **export via ChimeraX** into a **staging directory** and reference those paths in a manifest.
- **Prompt the user** before exporting unsaved items when they are required as inputs—especially large maps—so model-only steps never silently trigger a huge write.
- Keep the first implementation **local and explicit** (manifest file + staging dir); a full bidirectional live API can come later.

## Non-goals (v1)

- Streaming map data over a socket without writing files.
- Automatic sync of ChimeraX scene state into CryoModel on every edit.
- Cross-machine access (assume same workstation; localhost only if any server is used).

## Core artifact: manifest

A JSON file (name/location TBD, e.g. under project dir or `~/.cryomodel/chimerax_bridge/`) produced by **code running inside ChimeraX** (bundle command or script), not by guessing from outside.

Suggested fields per entry (conceptual):

| Field | Purpose |
|--------|--------|
| `id` | Stable id within manifest (e.g. ChimeraX model id or bundle-assigned) |
| `kind` | `map` \| `structure` \| other as needed |
| `format_hint` | e.g. mrc, cif, pdb |
| `path` | **If known**: absolute path to the file ChimeraX used to open it |
| `source` | `disk` \| `session` (or similar)—whether `path` is authoritative or export required |
| `export_path` | After optional export: path CryoModel should use |
| `size_bytes` | Optional; for prompting on large exports |
| `label` | Short display name for the UI |

The workflow UI loads this manifest and offers entries alongside “inherit from prior card” when wiring inputs.

## Resolution policy (preferential use of paths)

1. **Disk-backed open** (`source: disk`, valid `path`): CryoModel uses **`path`**. No copy into staging.
2. **Session-only** (`source: session` or missing path): **Do not** export until a CryoModel step **requires** that object *and* the user confirms (or pre‑opts in). Then ChimeraX exports to **staging**; manifest is updated or a one-off response lists `export_path`.
3. **Job needs model only**: manifest entries for maps can be ignored; no map export.
4. **Job needs map**: if only session-only map exists, show **size-aware prompt** before export to staging.

Staging directory should be configurable later; v1 can use a temp or project subfolder with a clear prefix (e.g. `chimerax_staging_<session_id>/`).

## ChimeraX side (implementation sketch)

- **In-process** Python/commands are authoritative: list open models, query filename/path when available, run `save` / export for session-only data.
- Expose a **single user action** in a ChimeraX bundle: **“Write manifest + export session-only as needed”** vs **“Write manifest only”** (lightweight).
- Optional later: localhost HTTP server started from the bundle for on-demand export; v1 can stay **file-based** (write manifest, CryoModel polls or user clicks “refresh import”).

## CryoModel / workflow UI side

- **Import from ChimeraX** button: read manifest (path TBD), populate a small **imported sources** list; inputs can **inherit** those paths like outputs from prior cards.
- **Card specs** already declare input types (`map.mrc`, `model.structure`); the UI only offers compatible manifest entries and **warns** when a required input would trigger export of unsaved large maps.
- CLI continues to take **paths only**; bridge responsibility ends at providing valid paths.

## Large maps / annoyance mitigation

- Default import flows toward **reference original path** when present.
- Copy/export only when necessary; **user prompt** for unsaved maps above a size threshold (threshold configurable).
- Optional future: **hard link** on same volume when a duplicate file is unavoidable and policy allows (advanced).

## Staging cleanup

- Document policy in implementation: delete on session end vs keep until user clears vs TTL. v1: **manual clear** or **per-run subdir** is acceptable.

## Security

- Any listener or helper that can trigger export should be **localhost-only** and avoid executing arbitrary paths from untrusted manifest sources. Manifest should be written only by trusted ChimeraX-side code.

## Phased rollout (suggested)

1. **Manifest-only, disk paths** *(implemented)*:
   - **ChimeraX:** command `cryomodel_manifest` / `cryomodel_manifest /path/to/out.json` (default `~/cryomodel_chimerax_manifest.json`). Writes `schema_version: 1` with `entries` (disk `path` when known; `source: disk|session`).
   - **Workflow API:** `POST /ui/chimerax-manifest` with `{ "path": "<manifest.json>" }` loads and validates JSON for the browser.
   - **Workflow UI (`dna_workflow_ui_demo.html`):** top bar **Manifest** path + **Load manifest**; each input can use **ChimeraX manifest** mode and pick a compatible entry (filtered by `artifact_type`). No auto-export yet.
2. **Selective export**: session-only structures/maps to staging with confirmation; update manifest or return paths to UI.
3. **Polish**: size prompts, cleanup policy, optional hard links, optional small HTTP bridge.

---

*This doc is internal planning; update it as implementation choices land.*
