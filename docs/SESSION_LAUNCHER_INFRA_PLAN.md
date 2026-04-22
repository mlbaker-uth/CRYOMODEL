# Session / project launcher & logging — infrastructure plan

**Status:** Phase **P0** implemented — workflow UI runs persist under `<project>/.cryomodel/`. Registry + `cryomodel manager` UI = Phase P2.  
**Context:** Phase 2 backlog (project organizer, activity log, workspace UX) splits **global project registry** vs **project-local scientific record**.

---

## 1. Decided product model (locked)

**Phenix-style project organizer + project-local logs + optional CLI anywhere.**

| Piece | Role |
|--------|------|
| **Global “organizer”** | Small store under **`~/.cryomodel/`** — list of known projects (id, display name, **absolute path**, last opened, API prefs, notes). Think Phenix’s project list, not CryoSPARC’s job database. |
| **Project directory** | **Source of truth for work:** maps, outputs, and **append-only CryoModel logs** live next to the data under **`<project_root>/.cryomodel/`**. |
| **CLI anywhere** | Running `cryomodel …` from **any** current working directory writes logs under **that** directory’s `.cryomodel*` / legacy paths — no requirement to register the folder first. The organizer **may** offer “register this path” or pick up projects from recent use. |
| **Unified operator experience** | **`cryomodel manager`** (Phase P2) will open the **project manager** UI, then the workflow UI with **cwd + API** set from the chosen project. Direct CLI use remains fully supported and does not go through the registry. |

**Timestamps:** Use **local time** in human-facing JSONL and registry fields (document as local ISO-8601 without forcing UTC). Avoids day-to-day friction on a single workstation; optional `utc` field can be added later for sync/multi-host scenarios.

**Non-goals for v1:** CryoSPARC-style central job database; requiring every command to touch the global registry.

---

## 2. Entry command: project manager (`cryomodel manager`)

**Initial phase:** expose the startup / project organizer from a **single, memorable CLI** — no separate install step.

| Name | Purpose |
|------|---------|
| **`cryomodel manager`** | Opens the project organizer (Phase P2). **Today:** prints status and where workflow runs are logged. |

**Optional alias (packaging):** console script **`cryomodel-manager`** (same entry point) — only if we want a hyphenated binary-style name.

**Relationship to existing commands:**

- **`cryomodel workflow-ui serve`** — raw API server (used by `manager` in P2 or by power users).
- **`cryomodel manager`** — opinionated “launch CryoModel for a project” path (organizer → workflow), **after P2**.

---

## 3. Current behavior (facts from the repo)

### 3.1 Workflow UI + API

- **UI:** **`cryomodel.html`** is the default workflow surface (V2). **`dna_workflow_ui_demo.html`** remains as **legacy** (still openable from Advanced → “Legacy workflow”). **`cryomodel manager open`** / **Launch Application** open `cryomodel.html` via `file://` with `cwd`, `api`, `manifest`, `chimerax`, `project` query params. Override the HTML path with env **`CRYOMODEL_WORKFLOW_HTML`** (absolute path or path relative to repo root) if needed.
- **Stale manager API:** `cryomodel manager` only starts the manager API if the default port (**8011**) is free. If an **old** `cryomodel manager serve` is still running, **Launch Application** keeps using that process and may still open the **legacy** HTML. Stop the old server (e.g. macOS/Linux: `lsof -ti :8011 | xargs kill`, then run `cryomodel manager` again), or check `GET http://127.0.0.1:8011/health` — field **`workflow_ui_html`** must end with **`cryomodel.html`**.
- **API:** `cryomodel workflow-ui serve` → FastAPI in `cryomodel/workflow/ui_api.py`.
- **Runs:** `POST /ui/run` accepts `{ card_id, command, cwd }`, spawns `subprocess.Popen(..., cwd=rec.cwd)` and streams stdout/stderr into in-memory `RUNS[run_id].log`. **P0:** on completion, also writes **`<cwd>/.cryomodel/runs/<run_id>.log`** and appends **`<cwd>/.cryomodel/activity.jsonl`** via `cryomodel.workflow.activity_log`.

### 3.2 CLI command logging (`cryomodel/cli/command_log.py`)

- **Per-project append log:** `<cwd>/.cryomodel_history.jsonl` (`LOG_FILENAME`).
- **Per-run stdout/stderr capture:** `<cwd>/.cryomodel_logs/<timestamp>_<tool>.log` (`LOG_OUTPUT_DIR`).
- **Scope:** Only commands that go through the `@log_command` decorator write these files, using **`os.getcwd()`** as the project root at the time the CLI runs.
- **Implication:** When the workflow UI runs `cryomodel ...` with `cwd` set to the project folder, the child process’s working directory is correct, so **decorated** CLI paths should write history under that project — *if* every card command is actually logged (worth auditing). Direct `python -m` or non-logged entry points would not get this behavior.

### 3.3 “Logs” CLI (`cryomodel log show|tail|stats`)

- Reads **`--cwd` / current directory** → `<cwd>/.cryomodel_history.jsonl` only. **P1:** extend to `.cryomodel/activity.jsonl` (+ legacy fallback).

### 3.4 Gap (remaining)

| Concern | Today |
|--------|--------|
| Project organizer / registry | Not implemented (**P2**) |
| Persistence of UI/API config | Not persisted |
| Single “activity log” story | Operator reconciles **`.cryomodel/activity.jsonl`** (UI runs), **`.cryomodel_history.jsonl`** (decorated CLI), and **`.cryomodel_logs/`** until **P1** merges reading |

---

## 3.5 Project registry file (`~/.cryomodel/projects.json`)

- **On disk:** a **JSON array** of project records (normalized fields include `project_root`, `name`, `api_host`, `api_port`, etc.).
- **Legacy:** an object shaped like `{ "projects": [ ... ] }` is still **read** correctly; the next save rewrites a flat array (migration on load).
- **Backup:** each save copies the previous file to **`projects.json.bak`** next to it.
- **Browse (manager UI):** folder/file pickers use **native dialogs** on macOS (`osascript`) and Linux (`zenity` when installed); **Tk** is only a fallback and often fails when the manager API runs as a background process—use the native path above.

## 4. Architecture: registry vs project-local

### 4.1 Global registry (`~/.cryomodel/`)

- **Purpose:** “Hidden” project organizer — **bookmarks + metadata**, not a second copy of runs.
- **Contents (sketch):** `projects.json` or `sessions/<id>.json` listing `{ id, name, project_root, api_base, chimerax_app, default_manifest_path, notes, updated_at (local) }`.
- **Mitigations:** On open, **validate** `project_root` exists; allow “repair path” if the user moved a folder.

### 4.2 Project-local tree (`<project_root>/.cryomodel/`)

- **Purpose:** **Portable** record next to data — backup, NFS, gitignore one directory.
- **P0 files:** `activity.jsonl`, `runs/<sanitized_run_id>.log`

### 4.3 Optional CLI without registry

- Any `cryomodel` command run from a directory writes project-local logs for **that** cwd only. Registry is **optional** for discoverability; it does not block execution.

---

## 5. Logging — target design

### 5.1 Principles

- **Project-local:** Primary log store under **`<project_root>/.cryomodel/`**.
- **Append-only event stream:** `activity.jsonl` for **workflow UI** runs (P0); CLI unified later if desired.
- **Per-run detail:** `.cryomodel/runs/*.log` holds full stdout/stderr for UI jobs.

### 5.2 On-disk layout (under project root)

```
<project_root>/
  .cryomodel/
    activity.jsonl        # workflow UI summaries (P0; local timestamps)
    project.json          # optional later: name, description, schema_version
    history.jsonl         # optional migrate from .cryomodel_history.jsonl
    logs/                 # legacy CLI capture from .cryomodel_logs (transition)
    runs/
      <sanitized_run_id>.log
```

Legacy files **at project root** (`.cryomodel_history.jsonl`, `.cryomodel_logs/`) remain until migration.

### 5.3 Activity line schema (workflow UI)

Each line in `activity.jsonl` — JSON with **local** `timestamp`:

- `source`: `"workflow_ui"`
- `run_id`, `card_id`, `command`, `cwd`, `status`, `return_code`, `duration_s`, `output_log` (relative path under project)

### 5.4 Implementation (`cryomodel/workflow/activity_log.py`)

- **`persist_workflow_ui_run(...)`** — mkdir `.cryomodel`, write run log, append one JSON line with optional **fcntl** lock on POSIX.

### 5.5 `cryomodel log` CLI (**P1**)

- Read **`--cwd` / `.cryomodel/activity.jsonl`**, fall back to `.cryomodel_history.jsonl`.

---

## 6. `cryomodel manager` — behavior

**Now:** callback prints Phase P2 notice and log file locations.

**P2:**

1. Ensure **`~/.cryomodel/`** exists; load/save project registry.
2. Present **project organizer** (browser-first): list projects, **add**, **open**.
3. **Open project:** start `workflow-ui serve` if needed, open workflow HTML with `cwd` + `apiBase`.
4. **Optional:** EMAN2-style terminal hints.

---

## 7. Workflow demo HTML

- **Bootstrap:** Read `URLSearchParams` / hash for `cwd`, `api`, etc.; banner if missing.
- **Secrets:** Prefer **session id** resolving to `~/.cryomodel` over huge URLs.

---

## 8. Implementation phases

| Phase | Scope | Outcome |
|-------|--------|---------|
| **P0** | `/ui/run` → **`activity.jsonl` + `runs/<id>.log`**; **`cryomodel manager`** placeholder | **Done** |
| **P1** | `cryomodel log` reads `activity.jsonl` (+ legacy fallback) | Terminal-friendly tail |
| **P2** | **`cryomodel manager`** + registry + organizer UI | Phenix-style entry |
| **P3** | Optional `project.json`; migrate `history.jsonl` | Cleaner tree |
| **P4** | Card reorder/clone + richer activity UI | Backlog polish |

---

## 9. Risks and remaining choices

1. **Stale paths in registry** — mitigate with validate-on-open (P2).
2. **Naming** — locked: **`manager`** (not `pm`).
3. **Windows** — defer; macOS/Linux first per backlog.
4. **Electron/Tauri** — optional later; same data model.

---

## 10. What we are *not* doing in the first slice

- CryoSPARC-style central job database as the primary store.
- Full Electron/Tauri (unless prioritized separately).
- Full frontend framework rewrite (React/Vite) — optional.
- ChimeraX bidirectional session API (Phase 2 non-goal per `PHASES.md`).

---

*Next: **P1** (`cryomodel log` + `activity.jsonl`), then **P2** (manager UI + registry).*

---

## Checkpoint — resume (local)

- **Sequence conservation workflow card:** Done (metric dropdowns for B-factor / occupancy, short ChimeraX-oriented description on the card). Optional follow-up: mirror new CLI flags in `TOOLS_PORT.md` or the user guide if you maintain a CLI changelog there.
- **Tier 3 diffusion (sequence conservation):** Implemented as `cryomodel seqconservation-diffuse` — 3D Cα graph over all selected chains, soft falloff edges, iterative mixing from variability seeds; optional `nearest_peak` basins; workflow card `seqconservation_diffuse_run`. Further ideas: geodesic / multi-scale, explicit watershed, learned kernels.
