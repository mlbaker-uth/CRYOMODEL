# Workflow cards vs CLI inventory

**Purpose:** Snapshot of what exists in `cryomodel` CLI vs what is exposed as **workflow job types** in `dna_workflow_ui_demo.html`, plus gaps. Use this before session/launcher design so new UX lines up with real commands.

**Sources of truth**

- **CLI:** `cryomodel/cli/__init__.py` (`_register` / `add_typer`).
- **Workflow UI specs:** `dna_workflow_ui_demo.html` — `SPECS`, `ORDER`, `SPECS_DEV_STATUS`, `JOB_TOOL_COMMAND`.
- **Resolver tests (partial):** `tests/test_job_resolver.py` — `dnaaxis_extract`, `dnabuild_build`, `basehunter_run` only (not a full registry).

---

## 1. Top-level CLI commands (`cryomodel …`)

| Command | Kind | Notes |
|---|---|---|
| `findligands` | pipeline | |
| `predictligands` | pipeline | |
| `pathwalker` | pipeline | Legacy pathwalking; alias `pathwalk` |
| `pathwalker-average` | utility | Averaging; alias `pathwalk-average` |
| `pyhole` | pipeline | |
| `pyhole-plot` | utility | |
| `basehunter` | pipeline | |
| `validate` | pipeline | |
| `pdbcom` | pipeline | |
| `pdbdomain` | pipeline | |
| `fitcompare` | pipeline | |
| `fitprep` | pipeline | |
| `loopcloud` | pipeline | |
| `pathwalker2` | pipeline | |
| `version` | meta | |
| `train-ml` | expert | Lazy import |
| `train-ensemble` | expert | Lazy import |
| `extract-features` | expert | Lazy import |
| `foldhunter` | pipeline | |
| `affilter` | pipeline | |
| `workflow` | utility | YAML/JSON runner |
| `workflow-validate` | utility | |
| `assistant` | Typer app | Multi-subcommand |
| `dnabuild` | Typer app | `build`, `build-2bp` |
| `dnaaxis` | Typer app | `extract`, … |
| `log` | Typer app | `show`, `tail`, `stats` (`cryomodel log …`) |
| `mapfilter` | Typer app | `apply`, `list` |
| `pdb-mutate` | Typer app | `run` — alignment-driven sequence / side-chain update |
| `seqconservation` | pipeline | MSA → per-residue metrics; optional PDB B-factor / occupancy |
| `seqconservation-diffuse` | pipeline | Same table + 3D Cα graph diffusion on raw or composite seeds |
| `fasta-extract` | Typer app | `row` — extract one gap-stripped record for `pdb-mutate --target-fasta` |
| `zonal-refine` | Typer app | `run` / `global` — local and global χ1 (+ optional Rama); `check-map` — Cα map sampling sanity check; see `docs/ZONAL_GLOBAL_OVERLAP_AND_GMM.md` |
| `pathmeasure` | Typer app | `serve` |
| `workflow-ui` | Typer app | `serve` |
| `chimerax` | Typer app | e.g. `manifest` |
| `chimerax-manifest` | help | |
| `model2map` | pipeline | |
| `pdb2mrc` | pipeline | Alias of `model2map` |

---

## 2. Workflow UI job types (cards) in `dna_workflow_ui_demo.html`

Each row is one `job_type` key in `SPECS`.

| `job_type` | Maps to CLI (approx.) | `SPECS_DEV_STATUS` |
|---|---|---|
| `mapfilter_apply` | `mapfilter apply` | production |
| `model2map_convert` | `model2map` | production |
| `affilter_run` | `affilter` | production |
| `foldhunter_search` | `foldhunter` | testing |
| `findligands_run` | `findligands` | untested |
| `predictligands_run` | `predictligands` | untested |
| `fitprep_check` | `fitprep` | untested |
| `validate_run` | `validate` | untested |
| `pathwalker2_discover` | `pathwalker2` | untested |
| `pathwalker_run` | `pathwalker` | untested |
| `pathwalker_average_run` | `pathwalker-average` | untested |
| `pyhole_analyze` | `pyhole analyze` | untested |
| `pyhole_plot_run` | `pyhole-plot plot` | untested |
| `train_ml_run` | `train-ml` | experimental |
| `train_ensemble_run` | `train-ensemble` | experimental |
| `extract_features_run` | `extract-features` | experimental |
| `loopcloud_generate` | `loopcloud` | untested |
| `fitcompare_run` | `fitcompare` | untested |
| `pdbdomain_identify` | `pdbdomain` | untested |
| `pdbcom_compute` | `pdbcom` | untested |
| `dnaaxis_extract` | `dnaaxis extract` | production |
| `dnabuild_build` | `dnabuild build-2bp` | production |
| `basehunter_run` | `basehunter` | untested |
| `pdb_mutate_run` | `pdb-mutate run` | testing |
| `zonal_refine_run` | `zonal-refine run` | testing |
| `alignment_sequence_pick_run` | `fasta-extract row` | testing |
| `seqconservation_run` | `seqconservation` | testing |
| `seqconservation_diffuse_run` | `seqconservation-diffuse` | experimental |
| `pathmeasure_launcher` | utility (starts PathMeasure server) | production |

**Chaining:** `seqconservation_run` and `seqconservation_diffuse_run` expose **`msa_fasta`** (passthrough of the alignment input path, not versioned on re-run) so a downstream card can **inherit** the same FASTA as `sequence.fasta`.

**`ORDER` (Run all):** pipeline cards through `alignment_sequence_pick_run` (after sequence conservation / diffusion); includes `zonal_refine_run` after `pdb_mutate_run`. Legacy pathwalker / pyhole cards are listed but **not** the ML trio (`train_ml_run`, …), **`pathmeasure_launcher`**, or **`train_*` / `extract_features_run`** so “Run all” does not start servers or long training jobs.

**Utility / special:** `pathmeasure_launcher` (no `cryomodel` string — API starts PathMeasure). `pathwalker_average_run` and `pyhole_plot_run` are tagged **Utility** in the catalog; ML cards tagged **Expert / ML**.

---

## 3. Gap analysis

### 3.1 CLI commands with **no** workflow card (today)

Likely intentional as **utilities** or **out of scope** for the graph:

| CLI | Suggested classification |
|---|---|
| `workflow`, `workflow-validate` | Headless orchestration; may stay CLI-only or get a thin “export/run” utility later |
| `assistant` | Covered by Assistant bar in demo UI, not a job card |
| `log` | Operator tooling; optional future “run history” panel |
| `workflow-ui serve` | Server process; session launcher concern |
| `chimerax` / `chimerax-manifest` | Bridge / manifest; separate from pipeline cards |
| `version` | Meta |

**Pipeline CLIs without a card**

| CLI | Notes |
|---|---|
| `pdb2mrc` | Redundant with `model2map` card |

### 3.2 **Subcommand / mode** gaps (CLI richer than UI)

| Area | UI today | CLI also has |
|---|---|---|
| `dnabuild` | `build-2bp` only | `dnabuild build` (map-based poly-AT build) |
| `mapfilter` | `apply` only | `mapfilter list` (discovery/help) |
| `pathwalker` | core options on card | Extra flags (`--map-weighted`, `noise`, …) available on CLI |
| `pyhole` | non-interactive card | Full CLI has many more flags; interactive mode not exposed on card |
| ML trio | simplified params | Full CLI (focal loss, class weights, etc.) |

### 3.3 **Tests vs UI**

- `tests/test_job_resolver.py` only embeds specs for **`dnaaxis_extract`**, **`dnabuild_build`**, **`basehunter_run`**. Other job types are **not** covered by the same resolver JSON in tests (the demo HTML holds the full set).

### 3.4 Operator sign-off vs `SPECS_DEV_STATUS`

`PHASE2_CARD_BACKLOG.md` tracks manual verification (e.g. FoldHunter **not verified**). The HTML `SPECS_DEV_STATUS` uses overlapping labels (`foldhunter_search`: **testing**). Reconcile these when you formalize session/state so “production” in the UI matches operator sign-off.

---

## 4. Summary counts

| Bucket | Count |
|---|---|
| Top-level `cryomodel` commands (excluding Typer sub-apps’ *inner* subcommands) | ~39 distinct entry styles (see §1) |
| Workflow `job_type` cards in demo HTML | **28** (incl. `pathmeasure_launcher`, ML trio, pathwalker / pyhole, `pdb_mutate_run`, `alignment_sequence_pick_run`, `seqconservation_run`, `seqconservation_diffuse_run`) |
| `ORDER` length (“Run all” chain) | **24** job types |
| CLI pipeline-ish commands with **no** card | **Meta/utility only** (`workflow`, `log`, `version`, …) plus `pdb2mrc` alias |

---

*Generated for planning; update when `SPECS` or `cryomodel/cli/__init__.py` changes.*
