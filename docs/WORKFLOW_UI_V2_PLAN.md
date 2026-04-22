# CryoModel Workflow UI v2 — implementation plan

**Status:** Planning (no implementation commitment until this doc is reviewed).  
**Goal:** A polished, tester-ready workflow surface that matches the agreed mockup and reuses the **cryomodel_ui_v2_full** visual language (see `cryomodel_ui_v2_full.zip`: `index.html`, `styles.css`, `app.js` scaffold), while preserving existing behavior: workflow API (`cryomodel workflow-ui serve`), project manager registry, ChimeraX manifest, import/export JSON/YAML.

**Reference mockup:** high-fidelity layout (header with project + actions, three-column body, assistant footer) — align structure and hierarchy with that design, not necessarily pixel-perfect on pass one.

---

## 1. Problems we are solving

| Audience | Need |
|----------|------|
| **External testers** | Clear project context, obvious next steps, fewer raw paths in the primary surface. |
| **Developers** | Still reachable **Advanced** settings (CWD, manifest path, API base, manager API) without blocking normal use. |
| **Product** | One coherent story: **Project** ↔ **Workflow** (workflow = ordered tool graph; project = CryoModel-owned context + registry). |

---

## 2. Design principles

1. **Progressive disclosure:** Default top bar shows identity, project, and primary actions; **Advanced** holds CWD, manifest, API URLs, and other footguns.
2. **Same engine:** No change to the core execution contract (`POST /ui/run`, cwd, command strings) unless strictly necessary; UI is a better shell over the same API.
3. **Incremental delivery:** Ship vertical slices that are demoable after each phase (testers can use a subset early).
4. **Accessibility baseline:** Keyboard alternatives for drag-and-drop (move up/down, add from library); visible focus states.

---

## 3. Target layout (information architecture)

### 3.1 Header (top bar)

| Element | Behavior |
|---------|----------|
| **CryoModel branding** | Name + logo mark (simple SVG or CSS mark; asset can be finalized later). |
| **Current project** | Clickable label (e.g. `Project: DNA Modeling`). Opens **project manager** (`cryomodel_manager.html` / manager API) in a new tab or same window per policy — same registry as today. |
| **Primary actions** | Run All (pipeline), Import workflow, Export workflow (JSON/YAML parity with current UI). |
| **ChimeraX** | Single clear action: open outputs / session (reuse existing API behavior). |
| **Advanced** | Dropdown or slide-out: **CWD**, **Workflow API base**, **Manifest path**, **Mgr API**, **Sync from registry**, optional **Save env to prefs** — mirrors fields we already wire but hidden from the default view. |

**Non-goals for v2 header:** Exposing every control in the first row; duplicating full manager UI inside the workflow page.

### 3.2 Main body — three columns

| Column | Role |
|--------|------|
| **Card library (left)** | Searchable, filterable list of **card types** from existing `SPECS` / catalog. **Tags** (e.g. DNA, Protein, Utility, Experimental, Map, PDB, …) are **data labels** on each spec; filters are toggles / multi-select. Short **3–5 word** blurbs per card. |
| **Workspace (center)** | Ordered execution list. **Run All** runs cards **top → bottom**. Each row: step number, name, **status** (stoplight), compact I/O + key options, actions: reorder (drag or ↑↓), duplicate, delete. |
| **Inspector (right)** | **Tabs:** Parameters \| Command \| Run log — same content as today’s options panel + command preview + log, with more vertical space for Run log. |

### 3.3 Footer — Assistant

| Addition | Notes |
|----------|------|
| **Links** | CryoModel website, Manual / FAQ (URLs configurable or env; placeholder `#` until content exists). |
| **Cite** | Deferred milestone; button placeholder or “Coming soon” to avoid scope creep. |

---

## 4. Functional requirements (by area)

### 4.1 Card library

- **Search:** Filter catalog by display name and tags (client-side).
- **Filters:** Tag chips (All + tag groups). “All” clears other filters or shows union; exact semantics to be decided (see **Open decisions** below).
- **Add to workspace:** Drag from library → insert at drop position **or** append at end; **Add** button fallback for no-drag.
- **Descriptions:** Short line in catalog (≤ 5 words); full description remains in inspector or tooltip.

### 4.2 Workspace

- **Order = execution order** for Run All and for dependency defaults when wiring artifacts.
- **Status model (stoplight):**  
  - **Grey** — waiting / not run / needs input  
  - **Yellow** — running  
  - **Green** — success  
  - **Red** — error (surface last error line or short message)  
  Persist mapping from current `badge` / run state to these four.
- **Row content:** Number, title, one-line inputs/outputs (best-effort from card + artifact graph), key params; align with mockup density.
- **Reorder:** Drag-and-drop within list + **duplicate** + **delete** (same semantics as today, better UX).

### 4.3 Inspector

- **Tabs** for Parameters / Command / Run log (single visible panel at a time).
- **Run Card** remains primary action inside Parameters when a card is selected.
- **Command** tab: copy-friendly command preview (existing).

### 4.4 Project + manager integration

- **Project title** in header comes from registry / URL / `sync` — already partially implemented; v2 header must show it **without** requiring raw path in the default view.
- **Click project** → manager (or deep-link to manager with same query params as **Project** button today).

### 4.5 Advanced (developer) panel

- All current top-bar fields that testers should not need daily: CWD, API, manifest, Mgr API, sync.

---

## 5. Technical approach

### 5.1 Files and migration strategy

**HTML entry points (repo root):**

| File | Role |
|------|------|
| **`dna_workflow_ui_demo.html`** | **Legacy** — first UI experiment; **keep working** until v2 is ready for default launch. |
| **`cryomodel.html`** | **New primary UI** — target filename for the polished workflow (v2 layout + behavior). Build here while the legacy page stays available for developers and rollback. |

**Implementation options:**

| Option | Pros | Cons |
|--------|------|------|
| **A. Evolve `dna_workflow_ui_demo.html` in place** | Single file, gradual diff | File already large; name no longer matches product |
| **B. New `cryomodel.html` + shared JS module** | Clear cut; legacy remains for parallel testing | Shared logic until one module owns state |

**Recommendation:** **B** — implement v2 in **`cryomodel.html`**, optionally extract shared logic into `static/workflow/` (or similar) as **`workflow-core.js`** imported by both pages during transition. When v2 reaches parity and testers switch over:

1. Point **`cryomodel/cli/manager.py`** `_workflow_ui_url` from `dna_workflow_ui_demo.html` → **`cryomodel.html`** (and align any docs).
2. Treat **`dna_workflow_ui_demo.html`** as **deprecated** / dev-only unless still needed for A/B.

Until that cutover, **`cryomodel manager open`** continues to open the **legacy** HTML so nothing breaks mid-build.

### 5.2 Styling

- Base **tokens** (colors, spacing, radii, shadows) from **cryomodel_ui_v2_full** `styles.css`.
- Align **component** class names (`.topbar`, `.library-item`, `.workspace-card`, `.tabs`, `.assistant-shell`) with that file where possible; extend only where CryoModel-specific.

### 5.3 Drag-and-drop

- Use **HTML5 DnD** for workspace reorder and library→workspace; **fallback** buttons (↑↓, Add) for reliability.
- Optional later: touch-friendly polyfill if needed.

### 5.4 Tags and filters

- Add optional `tags: string[]` (or `tags: string`) to each entry in `SPECS` / catalog metadata; **default** tag for untagged cards (e.g. `Utility`).
- Filter logic: **intersection** of selected tag chips with card tags (document in UI).

---

## 6. Phased delivery (concrete milestones)

| Phase | Scope | Tester value |
|-------|--------|----------------|
| **V2.0 Shell** | New layout (header + 3 columns + empty states + Advanced drawer); project title + links to manager; Assistant footer + links (stub URLs). | Looks “real”; can navigate project. |
| **V2.1 Library** | Search + tag filters + short descriptions; add from library (button first, then drag). | Can build workflows without editing raw HTML. |
| **V2.2 Workspace** | Numbered rows, stoplight, I/O summary, DnD reorder, duplicate/delete, Run All wired to existing engine. | End-to-end pipeline UX. |
| **V2.3 Inspector** | Tabbed Parameters / Command / Run log; polish. | Usable review and debugging. |
| **V2.4 Polish** | Loading states, empty/error copy, keyboard shortcuts, cite button placeholder, docs links final. | Academic-ready polish. |

**Parallel:** Keep **manager** and **launch URL** query params in sync (see `SESSION_LAUNCHER_INFRA_PLAN.md`).

---

## 7. Risks and mitigations

| Risk | Mitigation |
|------|------------|
| **`file://` vs `http`** for UI | Prefer documenting **open via manager** or local static server; test both. |
| **Large monolith JS** | Split modules before adding DnD complexity. |
| **Tag taxonomy churn** | Ship with **minimal** tag set; extend `SPECS` incrementally. |
| **Tester confusion (project vs workflow)** | Short copy in header + one link to “What is a project?” in FAQ. |

---

## 8. Open decisions (to lock before build)

1. **Filter semantics:** OR vs AND across multiple tag chips; whether “All” clears others.
2. **Project click:** open manager in **new tab** (recommended) vs same window.
3. **Run All** on partial validation: **stop on first error** vs **continue** (recommend default: stop on first failure).
4. **Single HTML file vs bundled assets** for distribution (pip package may ship `static/` directory).

---

## 9. Relationship to existing docs

- **Session / launcher:** `docs/SESSION_LAUNCHER_INFRA_PLAN.md` — registry, manager, activity logs; **this** doc focuses on **workflow UI** only.
- **Phase backlog:** `PHASE2_CARD_BACKLOG.md` — card coverage; v2 UI is **presentation** for the same tools.

---

## 10. Next step

**Default launch:** **`cryomodel/cli/manager.py`** opens **`cryomodel.html`** (set **`CRYOMODEL_WORKFLOW_HTML`** to override). Legacy demo remains for developers (**Legacy workflow** in Advanced).
