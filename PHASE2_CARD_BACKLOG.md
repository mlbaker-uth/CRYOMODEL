# Phase 2 Card Backlog

Practical implementation backlog for **CryoModel workflow UI** card integration.

**ChimeraX** is separate molecular visualization software. Tools you build *inside* ChimeraX are **out of scope** here — they do not need workflow cards.

Legend:
- **Type**: `pipeline` (chainable job card) or `utility` (launcher/helper)
- **Effort**: `S` (small), `M` (medium), `L` (large)
- **Priority**: `P0` highest, then `P1`, `P2`

## Phase 2 manual verification (operator sign-off)

| Card / tool | Status | Notes |
|---|---|---|
| DNA Axis (`dnaaxis`) | **Complete** | Passed operator tests |
| DNA Build (`dnabuild`) | **Not verified** | Card present; re-test with `dnaaxis` → `dnabuild` chain and manifest wiring |
| Map Filter (`mapfilter`) | **Complete** | Passed operator tests |
| Affilter (`affilter`) | **Complete** | Passed operator tests |
| PathMeasure (utility launcher) | **Complete** | Passed operator tests |
| Foldhunter (`foldhunter`) | **Not verified** | Did not pass tests; suspected tie-in to map/model alignment (e.g. `model2map` quality); re-test after map pipeline is stable |
| `model2map` | **Complete** | Operator verified; MRC origin words 50–52 + zyx write path |
| Zonal refine (`zonal-refine run`) | **Not verified** | Card `zonal_refine_run`; local χ1 + map (+ optional Rama); defer A3-style domain UI until global refiner shares the same engine |

## Phase 2 end bucket (session launcher + workspace UX)

End-of–Phase 2 items (PHENIX-style operator workflow, not necessarily new science tools):

| Item | Scope | Notes |
|---|---|---|
| **Session / project launcher** | First step before opening the workflow UI | **Select / create / delete** sessions; **project name, CWD, API base URL, ChimeraX app name/path, manifest path**; **free-text description** (“what is this project for?”) to avoid hunting the right folder like in PHENIX. **Launch** starts workflow UI + API with that context. **Platform:** **macOS + Linux** first; Windows optional/low priority. Implementation options: small **Electron / Tauri / neutralino** shell, or **Node**-driven wrapper (still needs a window if you want native pickers); plain **Python + Qt** is viable but heavier to ship. |
| **Session persistence** | JSON or YAML on disk (e.g. `~/.cryomodel/sessions/` or per-project `.cryomodel/session.json`) | Browsers cannot set shell `CWD`; launcher + API own the truth and pass config into the UI. |
| **CryoModel activity log** | **Not** a full recursive dump of the project tree | Append-only **CryoModel-only** record: **tool/card name, timestamp** (and optionally command summary); sourced from workflow API job metadata and/or a single project-local log file under the session dir (and allowed subdirs). **No** requirement to ingest arbitrary non-CryoModel files. Goal: answer “what did I run here, when?” without combing the disk. |
| **Workspace card actions** | Per-card **move up**, **move down**, **delete**, **clone** | **Clone** deep-copies spec + params + wiring; new card id. |

**Launcher note:** A cross-platform “app” usually means **Electron/Tauri** (HTML UI + Node/Rust host) or **PyInstaller + Qt**; “Node only” still needs one of those shells if you want a real window and file dialogs without opening a terminal.

## Phase 2 baseline pipeline cards (integrated; see sign-off table)

These tools already have workflow UI coverage and/or operator sign-off but were not listed in the priority tables below.

| Tool | Type | Primary command | Required inputs | Key outputs | Effort | Notes |
|---|---|---|---|---|---|---|
| `mapfilter` | pipeline | `cryomodel mapfilter apply` | `map.mrc`, filter params | filtered MRC | S | Operator verified |
| `dnaaxis` | pipeline | `cryomodel dnaaxis extract` | `map.mrc`, threshold, optional guides PDB | centerline PDB, optional axis MRC | M | Operator verified; feeds `dnabuild` |
| `dnabuild` | pipeline | `cryomodel dnabuild build`, `build-2bp` | map + threshold **or** centerline PDB + 2-bp template | DNA PDB | M | Typical chain: `dnaaxis` → `dnabuild`; sign-off pending |
| `model2map` | pipeline | `cryomodel model2map` | `model.structure` | synthetic MRC | S | Alias `pdb2mrc`; operator verified; FoldHunter probe / synthetic density |

## P0: Core pipeline expansion

| Tool | Type | Primary command | Required inputs | Key outputs | Effort | Notes |
|---|---|---|---|---|---|---|
| `affilter` | pipeline | `cryomodel affilter` | `model.structure` (AlphaFold/model PDB) | filtered model PDB, optional region report CSV | M | Foundation for foldhunter and downstream fitting |
| `foldhunter` | pipeline | `cryomodel foldhunter` | `map.mrc`, `model.structure` | best-fit model PDB, fit report/log | M | Strong candidate for template card + preset defaults |
| `findligands` | pipeline | `cryomodel findligands` | `map.mrc`, `model.structure` | ligand PDB, ligand map, reports | M | Important with manifest-imported map/model paths |
| `predictligands` | pipeline | `cryomodel predictligands` | ligand candidates + map/model context | prediction table CSV/JSON | M | Chain immediately after `findligands` |
| `validate` | pipeline | `cryomodel validate` | `map.mrc`, `model.structure` | validation CSV/JSON | S | High-value QC endpoint card |
| `fitprep` | pipeline | `cryomodel fitprep check` | `map.mrc`, `model.structure` | prep/QC report | S | Pre-flight card before expensive jobs |

## P1: Model-building and comparative analysis

| Tool | Type | Primary command | Required inputs | Key outputs | Effort | Notes |
|---|---|---|---|---|---|---|
| `pathwalker2` | pipeline | `cryomodel pathwalker2 discover` | `map.mrc` (+ optional residue count/settings) | traced model/fragments PDB | M | Add guided defaults for low/med/high resolution |
| `loopcloud` | pipeline | `cryomodel loopcloud generate` | map/model or model gaps context | rebuilt/refined model | M | Candidate for advanced-params collapsed section |
| `fitcompare` | pipeline | `cryomodel fitcompare compare` | map + multiple candidate models | comparison table/report | M | Needs multi-model input UX |
| `pdbdomain` | pipeline | `cryomodel pdbdomain identify` | `model.structure` | domain assignment outputs | M | Pairs with ChimeraX domain UI/visualization |
| `pdbcom` | pipeline | `cryomodel pdbcom compute` | domain/model inputs | COM table + model | S | Great for quick structural analytics |
| `basehunter` polish | pipeline | `cryomodel basehunter compare` | map + model | scores CSV + summary JSON | S | Already present; refine UX, defaults, and docs |

## P2: Specialized and expert workflows

| Tool | Type | Primary command | Required inputs | Key outputs | Effort | Notes |
|---|---|---|---|---|---|---|
| pathwalker (legacy) | pipeline | `cryomodel pathwalker` | map (legacy pathway) | traced path/model PDB | M | Aliases: `cryomodel pathwalk`; engine under `cryomodel/pathalker/`; prefer `pathwalker2` for new work |
| `pathwalker-average` | utility | `cryomodel pathwalker-average` | comma-separated path PDBs | averaged PDB | S | Alias: `pathwalk-average`; pairs with legacy pathwalker |
| `train-ml` | pipeline/expert | `cryomodel train-ml` | training features/datasets | trained model artifacts | L | Workflow card added (`train_ml_run`); requires ML extras |
| `train-ensemble` | pipeline/expert | `cryomodel train-ensemble` | datasets/models | ensemble artifacts | L | Workflow card (`train_ensemble_run`) |
| `extract-features` | pipeline/expert | `cryomodel extract-features` | input datasets | feature matrix outputs | M | Workflow card (`extract_features_run`) |

## Utility cards (keep separate from pipeline cards)

These are part of UX but not regular dataflow steps:

| Tool | Type | Current status | Next step |
|---|---|---|---|
| `pathmeasure serve` | utility | **Verified** (utility card launcher) | Optional: quick help link for students |
| `assistant` | utility | Implemented in assistant panel | Improve context collection + mode presets |
| `chimerax manifest` | utility | Command + ChimeraX widget done | Optional in-app “refresh manifest” helper |
| `workflow export/import` | utility | JSON/YAML export + JSON import present | Add schema validation + import warnings |
| `logs show/tail/stats` | utility | CLI available | Optional run-history panel in UI |
| `workflow` | utility | CLI available | `cryomodel workflow`, `cryomodel workflow-validate` — headless runner; complements UI and export/import |
| `workflow-ui serve` | utility | Required with UI | `cryomodel workflow-ui serve` — HTTP API for workflow UI; operator or future session launcher starts this |
| `pyhole` | utility | CLI + workflow card (`pyhole_analyze`) | `cryomodel pyhole analyze` |
| `pyhole-plot` | utility | CLI + workflow card (`pyhole_plot_run`) | `cryomodel pyhole-plot plot` |

## Exploratory / not Phase 2 exit criteria

Ideas captured for later; do not block Phase 2 completion.

| Topic | Notes |
|---|---|
| Hybrid GMM + local NCC rigid refinement | Exploratory design only — see `docs/GMM_LOCAL_NCC_FITTING_NOTE.md`. Revisit after FoldHunter and `model2map` are stable. |

## Implementation order recommendation

1. Baseline already integrated: `mapfilter`, `dnaaxis`, `model2map`, `affilter`; finish `dnabuild` operator sign-off  
2. `foldhunter` -> `fitprep` -> `validate`  
3. `findligands` -> `predictligands`  
4. `pathwalker2` -> `loopcloud` -> `fitcompare`  
5. `pdbdomain` -> `pdbcom`  
6. P2: legacy pathwalker / pathwalker-average / pyhole / ML cards (in demo UI catalog)

## Definition of done (per card)

- Command template with validated required params.
- Input source modes: manual / inherited / ChimeraX manifest (where applicable).
- Typed outputs added to artifact graph.
- At least 1 happy-path test and 1 failure-path validation test.
- Example usage documented in UI help text or docs.
