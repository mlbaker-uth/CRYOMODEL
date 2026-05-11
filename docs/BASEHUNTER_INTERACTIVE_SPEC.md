# BaseHunter Interactive (ChimeraX): Implementation Specification

This document specifies an **interactive, decision-support** workflow for dsDNA (primary) in cryo-EM maps: users anchor **base pairs** with markers; the backend scores **purine vs pyrimidine** (and optional A/T/G/C), applies **Watson–Crick (WC)** and **anti-parallel** priors, optional **clash** penalties, and can **emit a built PDB** (default **poly-AT**). It complements the batch pipeline described in `BASEHUNTER_IMPLEMENTATION.md` and reuses template/NCC/EMD ideas in `cryomodel/nucleotide/`.

---

## 1. Goals and non-goals

### 1.1 Goals

- **Local inference** at user-defined loci: no requirement for full segmentation or a perfect global helical model.
- **Fast enough** for interactive use: **on the order of a few seconds per base pair** on a typical workstation (target: **≤5 s** per pair for default search grids; configurable “quality” mode may be slower).
- **Quantify** ambiguity (likelihoods, bootstrap optional) and **supplement** manual building rather than replace it.
- **Default chemistry**: **dsDNA**, **WC** base pairing, **anti-parallel** strands; user can relax constraints for special cases (later).
- **Output**: not only scores—**write a DNA coordinate model (PDB/mmCIF)** assembled from chosen bases, with sensible defaults (**poly-AT**).
- **Geometry QA**: **planarity** and **marker-placement sensitivity** as diagnostics.
- **Clash awareness**: penalize poses that overlap existing atoms (same model or user-selected context).

### 1.2 Non-goals (initial phases)

- Automatic tracing of full duplex without user markers.
- Guaranteed correct **register** (which pair index along the helix); user supplies pair sites.
- Full **4-letter** discrimination at equal reliability to purine/pyrimidine at all resolutions (optional phase).
- RNA-specific workflows (may share machinery later).

---

## 2. High-level architecture


| Layer                  | Responsibility                                                                                                                                                                                                                                              |
| ---------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **ChimeraX bundle**    | Map display; **marker placement** (CMM-style or adapted tool); optional **base-region** selection; **buttons** for “score pair”, “build pair”, **purine/pyrimidine overrides** (default poly-AT); export **subvolumes + transforms** (JSON/NPZ) to backend. |
| **cryomodel** (Python) | Pair frame construction; **DOF-limited search**; **PDB-template** fast fit; **masked map** NCC/EMD; **WC joint** scoring; **clash** term; **PDB/mmCIF** writer for duplex segment.                                                                          |


**IPC**: subprocess CLI (stable contract) or in-process import if environments align; v1 can be **files on disk** + JSON args for reproducibility.

---

## 3. User workflow (conceptual)

1. User loads map (and optionally an **in-progress** DNA model for clashes).
2. For each intended pair, user places **two markers** at the **base** regions (one per strand), **not** on backbone (see §4).
3. User triggers **Score** (or auto-score on marker commit).
4. Tool shows: per-side **purine vs pyrimidine** likelihoods, **joint WC-consistent** pair likelihood, **planarity** metric, **clash** flag/penalty, optional **sensitivity** to small marker shift.
5. User may **override** identity (e.g. force **A** and **T** for this pair) via UI.
6. User triggers **Build/Append**: system places **idealized** WC base pair geometry (from PDB templates) in the **fitted** frame and appends to the working model (or writes PDB).

---

## 4. Input contract (ChimeraX → backend)

### 4.1 Markers and frames

- **Minimum**: two 3D points **P₁**, **P₂** (Å, same coordinate system as map), one per strand, intended **near base centers** (user error expected).
- **Helix axis / strand direction** (strongly recommended for dsDNA):
  - Either **two additional** points per strand (e.g. backbone clicks) to define **local tangent**, or
  - A single **helix axis vector** + handedness bit, or
  - **Inferred** from last built segment (if building sequentially).

Used to build an orthonormal **pair frame** (§6): midpoints, **hydrogen-bond** in-plane directions, and **anti-parallel** flip for the second strand.

### 4.2 Map crops

- For each side **s ∈ {1,2}**: axis-aligned or **oriented** subvolume (recommended: **oriented** box aligned to pair frame) large enough to contain **base + minimal clash context**.
- **Voxel size**, **origin**, **dimensions**; same grid as primary map or **resampled** to a working apix (cached).

### 4.3 Base masking

- Experimental density for scoring should **de-emphasize or exclude** sugar/phosphate where possible:
  - **User mask** (optional): selected zone around bases only, or
  - **Template-derived mask**: base atoms only, as in existing backbone-mask concepts (`BASEHUNTER_IMPLEMENTATION.md`).

### 4.4 Optional context for clashes

- Path to **PDB/mmCIF** of current model (or ChimeraX atom list) + **cutoff radius** (e.g. 5–8 Å) around the pair crop for **steric** checks.

---

## 5. Degrees of freedom (search space)

### 5.1 Assumptions (dsDNA, WC, anti-parallel)

- Each strand’s base is refined in a **local frame** attached to the pair.
- **Opposite strand** pose is related by a **known flip** (180° about an axis in the pair plane + anti-parallel translation along helix) once **P₁**, **P₂**, and **axis** are fixed—**not** two independent 6D searches.

### 5.2 Parameterization (recommended v1)


| Parameter    | Meaning                                                                                                                | Typical search range                                                                                                       |
| ------------ | ---------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| **θ₁, θ₂**   | Two **in-plane** rotations of the base template (e.g. tip and twist about normal to pair plane, exact Euler split TBD) | Coarse grid (e.g. 5–10° steps) + optional local refine                                                                     |
| **flip**     | Discrete **anti-parallel** partner transform (fixed from geometry, not freely searched)                                | 1–2 states if ambiguity                                                                                                    |
| **t∥₁, t∥₂** | **In-plane** translations (two components) relative to marker                                                          | e.g. **±2–4 Å** or **±2–3 voxels** scaled by apix                                                                          |
| **t⊥**       | **Out-of-plane** translation (along helix step direction between successive pairs)                                     | **Minimal**: **±1–2 voxels** (or ±0.3–0.6 Å at typical apix) as **prior**, not absent—users mis-click along helix slightly |


**Rationale**: You **only need two rotations** per base (plus the **strand flip** for the partner); **three translations** matter, but **t⊥** should stay **tight** if markers truly sit in one **pair plane**. If data pushes optimum outside that window, **planarity diagnostic** (§9) should warn.

### 5.3 Optional reduction

- If **Stage A** (PDB fit, §7) is very peaked, **Stage B** can **refine** only **t∥** and **θ** with **t⊥** fixed.

---

## 6. Pair frame construction

1. **Midpoint** **M** = (P₁ + P₂)/2.
2. **Pair normal** **n̂** ≈ unit(P₁ − P₂) × **â** (or from explicit H-bond direction), normalized; sign convention fixed per UI.
3. **In-plane** axes **û**, **v̂** spanning the pair plane.
4. **Helix rise direction** **ĥ** (along duplex): from user-provided axis or from adjacent markers in **build-next** mode.

**Anti-parallel rule**: second-strand template orientation = **R_flip · R_first** with **R_flip** derived from WC geometry and **ĥ** (document explicitly in code from one canonical B-DNA reference pair).

---

## 7. Two-stage scoring (fast PDB → masked map)

### 7.1 Stage A — Real-space fit using **PDB-derived** density (fast)

- For each candidate **(purine vs pyrimidine)** template **PDB** (base atoms only), generate **synthetic density** on the **same grid** as the crop (Gaussian or atom-mask stamps at resolution σ ≈ target resolution).
- **Score** alignment vs **target** that is also **base-heavy** (masked experimental or synthetic-from-mask).
- Output: **best rigid transform** **T** (rotation + translation) per class, within the **DOF grid** of §5.

**Purpose**: Rapid, smooth correlation; reduces wasted search on pure experimental noise.

### 7.2 Stage B — Same **T** applied to **experimental** map

- Apply **T** to **masked experimental** subvolume (or apply **T⁻¹** to template map in map space—implementation choice; keep **one** convention).
- Compute **NCC** and optional **EMD** vs class templates (`classification.py` / existing EMD hooks).
- **Joint WC pair score**: combine side-1 and side-2 posteriors with **hard** WC (purine+pyrimidine) or **soft** penalty for violation (existing `pair_mismatch_penalty` pattern).

### 7.3 Micro-refine (optional)

- Small **local** optimization of **T** on **Stage B** objective only (few iterations) if Stage A/B disagree—keep bounded for latency.

### 7.4 Three-phase classification (design reference)

This subsection captures the **intended** staged inference for purine vs pyrimidine (and chemistry), aligned with handwritten design notes. The ChimeraX tool may implement subsets incrementally; **atom names below** are the canonical targets for **base-heavy** map overlap / molmap scoring.

#### 7.4.0 v1 scope and implementation status (ChimeraX)

- **Reference files (unchanged)**: the tool still loads `**referencePDB-purine.pdb`** and `**referencePDB-pyrimidine.pdb`**. **Internal convention only**: those files are **adenine (A)** and **cytosine (C)** geometry respectively—the **smallest** bases per class (fewest base atoms) for v1; filenames are not switched to `referencePDB-A.pdb` / `referencePDB-C.pdb`.
- **Phases 1–3**: the **pipeline slots** for all three phases can run in a loose sense, but **answers often coincide** until the following are fully implemented: Phase **2** correlation / inclusion / exclusion (and related map–template contrasts), and Phase **3** **EMD** and **molmap**-style synthetic-vs-experimental tests.
- **EMD / Phase 3**: **not wired** in the interactive tool yet; §7.4.4 describes the **target** behavior only.

#### 7.4.1 Local orientation (roll, pitch, yaw about **C1′**)

- Treat **C1′** as the anchor for small **base-only** adjustments: **roll** (twist about the glycosidic / N9–C1′ or N1–C1′ axis), **pitch** and **yaw** (tilt of the base plane relative to the sugar).
- Full duplex placement still uses the **pair frame** (§6) and WC geometry; C1′-centric Euler-like moves are a **refinement** layer on top of that frame, not a replacement for strand anti-parallel registration.

#### 7.4.2 Phase 1 — **Fast** heuristics

- Operate in a **marker-centered zone** (crop).
- Features: **shape / size** cues; count **N voxels above threshold** in the zone.
- Geometry intuition: **purine** density footprint tends to be more **rectangular** (two-ring); **pyrimidine** more **triangular** (single-ring). Use as a **weak prior** or tie-breaker, not a hard classifier at low resolution.

#### 7.4.3 Phase 2 — **Balanced** (map + template fit)

- Combine Phase 1 signals with **correlation / average map value** (or overlap-style metrics) after placing **reference** purine and pyrimidine templates (base-heavy).
- **Differentiation**: a pyrimidine template can sit in a purine-sized lobe but **under-fills** it (unoccupied density remains); prefer **N voxels (purine template) > N voxels (pyrimidine template)** when the geometry cue agrees (rectangle vs triangle).
- **Atom sets for map-driven fitting** (base + **C1′** only; no sugar/phosphate beyond C1′):


| Class          | Atom names                                  |
| -------------- | ------------------------------------------- |
| **Pyrimidine** | C1′, N1, C2, O2, N3, C4, N4, C5, C6         |
| **Purine**     | C1′, N9, C4, N3, C2, N1, C6, N6, C5, N7, C8 |


- **Thymine**: the ring carbonyl is **O4** (not O2); implementations should include **O4** in the pyrimidine set when the template is T/U-like, even though the shorthand list above is cytosine-centric.

#### 7.4.4 Phase 3 — **Thorough** (synthetic density vs EMD) — *design only; not implemented in tool yet*

- **Target behavior** (future): add **experimental map** (or full **EMD**) evidence on top of Phases 1–2.
- **Synthetic map from the built/fitted model** (ChimeraX `**molmap`**): e.g. `**molmap #8/A 2.5`** where `**#8**` is the atomic model, `**A**` is the chain ID for that nucleotide, and `**2.5**` is the **simulated resolution in Å**. After `molmap`, use a displayed isosurface threshold (example from notes: **0.8** on the generated volume) for **inside / outside** or overlap tests.
- **Discrimination test**: score **purine-in-purine-density** vs **purine-in-pyrimidine-density** (and symmetrically for pyrimidine). A **correct** class match should produce a **small** residual change when swapping labels; a **wrong** class (**Pur→Pyr** or **Pyr→Pur**) should produce a **large** change (poor overlap / high EMD).

---

## 8. Clash penalties

### 8.1 Definition

- After choosing pose **T**, place **base atoms** (and optionally **sugar** for clash only) in world coordinates.
- **Clash score**: pairwise **vdW overlap** (same form as cryomodel validation clash terms: heavy atoms, **probe** radius small) against:
  - **Rest of user model** within radius **R_clash**, and/or
  - **Symmetry mates** if map symmetry provided (phase 2).

### 8.2 Use in ranking

- **Total score** = **w_map · S_map** + **w_pair · S_WC_joint** − **w_clash · S_clash** (weights configurable; defaults conservative so clashes don’t dominate map evidence).

### 8.3 Output

- Per-pair: **clash count** or **max overlap**; **flag** if above threshold (user may move marker or model).

---

## 9. Planarity and marker sensitivity (QA)

### 9.1 Planarity

- Fit plane to **base atoms** of both templates in final pose; report **rms** deviation of atoms from plane.
- **Interpretation**: large deviation ⇒ possible **wrong plane**, **neighbor** density, **flexible** region, or **bad** marker—**advisory**, not hard fail unless calibrated.

### 9.2 Sensitivity

- Re-score with **P₁**, **P₂** jittered ±1 voxel / ±0.5 Å in plane; report **stability** of class label (e.g. fraction of jitter samples agreeing).

---

## 10. Building the DNA model (PDB/mmCIF output)

### 10.1 Default identity

- **Poly-AT**: strand 1 **A**, strand 2 **T** (or consistent with strand naming); user-visible **default**.

### 10.2 User overrides (UI)

- Per pair or global session default:
  - **Purine**: A or G  
  - **Pyrimidine**: C or T
- Enforce **WC** when **WC mode** on: (A,T), (G,C); reject or warn on mismatch.

### 10.3 Geometry source

- **Idealized** coordinates from same **PDB templates** used for scoring, transformed by final **T** and **R_flip**.
- **Connectivity**: single pair = **two residues** + **H-bond** distance restraints optional; multi-pair = append along **ĥ** with **rise/twist** defaults (B-form) or from previous pair.

### 10.4 File output

- **Append** or **write** `model_dna.pdb` / `.cif` with **CRYST1** if cell known; **REMARK** records for per-pair scores and flags.

---

## 11. Performance budget

- **Target**: **≤5 s** per pair on a typical machine for default grids (§5) and **moderate** crop sizes (e.g. 32³–48³ at ~0.8–1.2 Å apix).
- **Tactics**: precompute **rotated** synthetic templates; **FFT** for translation where applicable; **parallel** over (θ₁, θ₂) grids; reduce bootstrap in interactive mode (e.g. **20** draws or **off**).

---

## 12. ChimeraX UI: wireframe-level specification

This section defines panel layout, control names, defaults, and interaction logic for v1.

### 12.1 Panel layout

Single-tool panel with stacked groups:

1. **Data**
2. **Threshold**
3. **Pair markers**
4. **Compute**
5. **Results**
6. **Build**
7. **Advanced** (collapsed by default)
8. **Session / export**

### 12.2 Data group

Controls:

- **Map** (`QComboBox`): list open volume models from ChimeraX Model Panel.
  - Default: currently active volume if exactly one; otherwise blank with warning badge.
- **Working model (optional)** (`QComboBox`): open atomic models for clash context / append target.
  - Default: blank.
- **Template set** (`QComboBox`): `auto`, `purine-pyrimidine`, `AGCT` (phase 2).
  - Default: `purine-pyrimidine`.

Validation:

- Disable compute/build if no map selected.
- If map changes, mark all cached pair results as **stale**.

### 12.3 Threshold group

Controls:

- **Use Volume Viewer threshold** (`QCheckBox`, default ON).
- **Threshold value** (`QDoubleSpinBox`, disabled when checkbox ON).
- **Sync from viewer now** (`QPushButton`).
- **Apply to all pairs** (`QCheckBox`, default ON).

Behavior:

- When inherit ON, threshold displayed read-only and pulled from current map's Volume Viewer surface level.
- If viewer threshold unavailable, fallback to percentile heuristic and show info note.

### 12.4 Pair markers group

Controls:

- **Placement mode** (`QButtonGroup` with toggles):
  - `Place marker`
  - `Move marker`
  - `Select marker`
  - `Delete marker`
- **Pair controls**:
  - `New pair`
  - `Auto-pair consecutive clicks` (`QCheckBox`, default ON)
  - `Clear selected pair`
  - `Clear all pairs`
- **Pair table** (`QTableView`), columns:
  - `Pair ID`
  - `A marker` (set/missing)
  - `B marker` (set/missing)
  - `Status` (`new`, `ready`, `computed`, `stale`, `error`)
  - `Label` (user-editable optional text)

Placement rules:

- A pair is **ready** when two markers are present.
- Click order under auto-pair: odd click -> side A, even click -> side B.
- Moving or deleting any marker sets pair status to **stale**.
- Marker glyph colors:
  - Side A: cyan
  - Side B: magenta
  - Selected marker: yellow outline

### 12.5 Compute group

Controls:

- `Compute selected pair`
- `Compute all ready pairs`
- `Cancel`
- `Auto-compute when pair becomes ready` (`QCheckBox`, default OFF)
- `Quality` (`QComboBox`: `Fast`, `Balanced`, `Thorough`; default `Balanced`)

Progress + status:

- Progress bar with current pair ID.
- Log line area (last 3 messages): alignment score, WC penalty, clash warning.

Runtime presets:

- **Fast**: coarse rotations/translations, no bootstrap, no micro-refine.
- **Balanced**: default DOF grid, bootstrap off, micro-refine on.
- **Thorough**: finer grid, optional bootstrap, micro-refine on.

### 12.6 Results group

Views:

- **Selected pair detail** card:
  - Side A: `P(purine)`, `P(pyrimidine)`, confidence
  - Side B: `P(purine)`, `P(pyrimidine)`, confidence
  - Joint: WC-consistent posterior
  - Planarity RMS
  - Clash metric (count or max overlap)
  - Stability under jitter (if enabled)
- **Results table** for all computed pairs:
  - `Pair ID`, `Call`, `Confidence`, `WC`, `Planarity`, `Clash`, `State`

Visual overlays:

- Pair coloring in viewport by confidence:
  - green >= 0.8
  - yellow 0.6-0.8
  - red < 0.6 or clash-flagged
- Optional toggle: `Show fitted template ghosts`.

### 12.7 Build group

Controls:

- `Build selected pair`
- `Build all computed pairs`
- `Append to working model` (`QCheckBox`, default ON if working model selected)
- `Create new model` (`QCheckBox`, default ON if no working model)
- **Identity mode**:
  - `Use best-scoring class` (default)
  - `Default poly-AT`
  - `Manual override`
- **Manual override widgets** (enabled when selected):
  - Side A (`QComboBox`): `Purine(A/G)` then concrete `A`/`G`
  - Side B (`QComboBox`): `Pyrimidine(C/T)` then concrete `C`/`T`
- `Enforce WC` (`QCheckBox`, default ON)

Build rules:

- If `Enforce WC` and manual override conflicts, show blocking warning and do not build.
- Built residues get deterministic chain IDs/pair index labels for reproducibility.
- Post-build, optionally auto-advance selection to next unbuilt pair.

### 12.8 Advanced group (collapsed by default)

Controls:

- **DOF bounds**
  - In-plane translation range
  - Out-of-plane translation range (default tight ±1-2 voxels)
  - Rotation step size
- **Scoring weights**
  - `w_map`, `w_pair`, `w_clash`
- **Compute options**
  - `Enable micro-refine` (default ON)
  - `Enable bootstrap` (default OFF)
  - `Bootstrap n` (default 20 when enabled)
  - `Enable jitter sensitivity` (default ON for selected pair only)
- **Clash options**
  - Clash radius cutoff
  - Ignore hydrogens
  - Include symmetry mates (phase 2 toggle disabled in v1)

### 12.9 Session / export group

Controls:

- `Save session` (JSON with map id, marker coordinates, pair states, overrides, results).
- `Load session`.
- `Export results CSV`.
- `Export built model` (`.pdb`/`.cif`).
- `Reset tool` (clear UI state, keep models loaded).

### 12.10 Enable/disable state logic

- **Compute selected** enabled when selected pair is `ready` or `stale`.
- **Compute all** enabled when at least one pair is `ready`/`stale`.
- **Build selected** enabled when selected pair has `computed` result and no blocking error.
- **Build all** enabled when at least one computed pair exists.
- While compute job running:
  - disable marker editing controls for affected pairs;
  - keep selection/read-only browsing enabled;
  - `Cancel` enabled.

### 12.11 Error and warning messages

Standardized user-facing messages:

- `No map selected.`
- `Selected pair requires two markers.`
- `Threshold unavailable from Volume Viewer; using fallback value X.`
- `Pair N result stale: marker moved after last compute.`
- `Build blocked: manual override violates Watson-Crick mode.`
- `High clash penalty for pair N; consider moving marker or changing identity.`
- `Low confidence for pair N (<0.6); check base-only marker placement.`

---

## 13. Phasing


| Phase  | Scope                                                                                           |
| ------ | ----------------------------------------------------------------------------------------------- |
| **M1** | Two markers + crops + **purine/pyrimidine** + WC joint + **poly-AT** PDB + clashes + planarity. |
| **M2** | **A/G/C/T** buttons, **4-template** map scoring, micro-refine.                                  |
| **M3** | Sequential build along **ĥ**, **neighbor** priors, symmetry.                                    |


---

## 14. Open technical decisions

1. **Exact** split of **two rotations** (Euler convention vs tip/pitch about **n̂** and **ĥ**).
2. **Stage A** vs **Stage B** if optimum **T** differs: single **joint** refine vs trust Stage A only for speed.
3. **Clash** weights relative to map score (may require **one** calibration set).
4. **mmCIF** vs **PDB** as primary output for ChimeraX.

---

## 15. References in repo

- `BASEHUNTER_IMPLEMENTATION.md` — batch templates, masks, bootstrap.
- `cryomodel/nucleotide/basehunter_enhanced.py` — `classify_base_pairs`, alignment flags.
- `cryomodel/nucleotide/classification.py` — NCC, alignment, EMD hooks.
- `cryomodel/validation/geometry_priors.py` — clash overlap patterns (reuse for §8).

---

## 16. Implementation roadmap (phased, with gates)

This roadmap is intentionally phase-gated to reduce ChimeraX integration risk and avoid UI/compute coupling failures.

### 16.1 Phase 0 — Bundle scaffold and smoke stability

**Objective:** Create a minimal, stable ChimeraX bundle skeleton for BaseHunter Interactive using the known-good PathWalker layout pattern.

**Tasks:**

- Mirror PathWalker-style bundle anatomy:
  - `bundle_info.xml` with correct tool classifier/dependencies.
  - `src/__init__.py` implementing `BundleAPI` (`start_tool`, `get_class`).
  - optional shim `src/tool.py` forwarding to canonical tool module.
- Add minimal `BaseHunterTool` that:
  - opens as a dockable panel,
  - displays one static label and one test button,
  - writes a log message on button click.
- Verify install/reload/session restore workflow.

**Acceptance criteria (gate):**

- Bundle installs and appears in Tools menu.
- Tool opens/closes repeatedly without errors.
- Tool restores in session reload (if session-enduring enabled).
- No import path / classifier / package-name mismatches.

### 16.2 Phase 1 — Template and data contract stabilization

**Objective:** Lock template discovery/validation and threshold defaults before any real compute.

**Tasks:**

- Implement `TemplateRegistry`:
  - reads directory + `templates.txt`,
  - normalizes internal IDs (purine/pyrimidine, A/G/C/T, base-only vs full).
- Validate required assets for selected mode:
  - p/p scoring minimum set,
  - optional AGCT set.
- Parse suggested thresholds from template metadata.
- Expose explicit warnings for missing files and unsupported combinations.
- Add option to use external template root (including `NEW-DNA-TEMPLATES`).

**Acceptance criteria (gate):**

- Startup validation produces deterministic pass/fail report.
- Required files for v1 mode resolve without manual guessing.
- Threshold defaults are loaded and visible in UI.

### 16.3 Phase 2 — UI v1 (full interaction, mock backend)

**Objective:** Deliver the complete panel behavior with stable interaction flow before integrating expensive scoring.

**Tasks:**

- Implement all groups from §12:
  - Data, Threshold, Pair markers, Compute, Results, Build, Advanced, Session/Export.
- Implement marker interaction modes:
  - place, move, delete, select.
- Implement pair table state machine:
  - `new -> ready -> computed -> stale -> error`.
- Implement map selector and threshold inheritance from Volume Viewer.
- Wire compute/build buttons to mock responses and synthetic results.
- Implement save/load session JSON for markers and pair states.

**Acceptance criteria (gate):**

- User can complete an end-to-end mock flow:
  - select map -> place markers -> compute (mock) -> build (mock).
- No UI freezes during marker operations.
- Pair state transitions are deterministic and recoverable after reload.

### 16.4 Phase 3 — Compute engine integration (purine/pyrimidine)

**Objective:** Replace mocked compute with real p/p scoring using the two-stage method (§7), while preserving responsive UI.

**Tasks:**

- Add background worker + cancellation token for compute jobs.
- Implement Stage A (fast fit, PDB-derived representation).
- Implement Stage B (masked map NCC/EMD scoring).
- Add WC/anti-parallel pair coupling and score aggregation.
- Map quality presets (`Fast/Balanced/Thorough`) to DOF grids and options.
- Return per-side and joint posterior metrics to results table.

**Acceptance criteria (gate):**

- Compute runs asynchronously and can be canceled cleanly.
- Typical per-pair runtime meets interactive target envelope.
- Repeated runs on fixed input yield stable scores within tolerance.

### 16.5 Phase 4 — Build engine + clash/planarity diagnostics

**Objective:** Enable real model output and geometry-aware decision support.

**Tasks:**

- Build selected/all pairs into PDB/mmCIF:
  - default poly-AT,
  - best-scoring assignment mode,
  - manual overrides with WC enforcement.
- Append/new-model output modes and deterministic residue labeling.
- Integrate clash penalty computation against selected context model.
- Compute and display planarity RMS and warning flags.
- Add viewport overlays (confidence color, optional template ghost).

**Acceptance criteria (gate):**

- User can compute and build model coordinates from the same panel.
- Clash and planarity metrics are displayed and affect rank/flags as configured.
- Built output is loadable in ChimeraX and round-trippable.

### 16.6 Phase 5 — Hardening, packaging, and documentation

**Objective:** Make the tool robust for repeated use and distribution.

**Tasks:**

- Add regression tests for:
  - template validation,
  - pair state transitions,
  - compute payload contract,
  - output writing.
- Add compatibility checks for supported ChimeraX versions.
- Improve failure messaging and user guidance.
- Publish concise usage doc + troubleshooting notes.

**Acceptance criteria (gate):**

- Fresh install + first run succeeds on clean environment.
- Common user errors show actionable diagnostics.
- Tool passes regression suite and smoke checks.

### 16.7 Cross-phase risk controls

**High-risk items and mitigations:**

- **Bundle fragility (structure/metadata):**
  - Freeze PathWalker-like layout during early phases.
  - Avoid package/module renames until Phase 5.
- **Template naming drift:**
  - Enforce registry normalization + strict validation in Phase 1.
- **UI thread blocking:**
  - All compute in worker thread/process from Phase 3 onward.
- **Scope inflation in v1:**
  - Keep v1 centered on p/p + WC + poly-AT build.
  - Defer AGCT full scoring and advanced neighbor priors to follow-on milestone.

### 16.8 Milestone task board (checklist)

- **M0:** Bundle scaffold stable and session-safe.
- **M1:** Template registry and threshold metadata validated.
- **M2:** Full UI behavior complete with mocked backend.
- **M3:** Real p/p compute integrated and responsive.
- **M4:** Build output + clash/planarity integrated.
- **M5:** Packaging hardening, tests, and docs complete.

