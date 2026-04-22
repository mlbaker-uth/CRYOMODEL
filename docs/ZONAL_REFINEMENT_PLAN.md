# Zonal refinement: local optimizer → global meta-optimizer

**Status:** **Phases A0–A2 (local core) wrapped** — χ1 hard + optional soft shell + optional **small φ/ψ** Ramachandran micro-moves (`--rama-backbone`). **Next:** A3 hardening / reporting polish; Part **B** (global meta-optimizer) after local A is mature enough for orchestration.

**Purpose:** Sequence work so a **fast local** refinement engine exists first; a **global** strategy (overlapping passes + convergence) reuses that engine as a black box.

**Positioning:** Complement **Phenix real-space refinement** and **Coot** — lightweight, scriptable, local-to-map optimization for cryo-EM. Not a drop-in replacement for full-pipeline crystallographic refinement.

---

## 1. Design principles

| Principle | Implication |
|-----------|-------------|
| **Local first** | Ship a single-zone optimizer that is correct, bounded, and &lt;~1 min for helix-sized regions before any orchestration layer. |
| **Bounded degrees of freedom** | Prefer χ rotamers (+ optional small backbone moves later) over free torsion MD. |
| **Scalarized multi-objective** | One tunable score: map fit + clash + rotamer + Ramachandran penalties with documented weights / schedules. |
| **Explicit masks** | **Hard zone** (residues with any atom in user geometry) vs **soft zone** (neighbors movable only to relieve clashes / bad geometry caused by hard moves). |
| **Global = scheduler** | Phase B does not replace the local objective — it only decides *where* and *when* to call Phase A. |

---

## 2. Part A — Local zonal optimizer (build order)

Deliver a CLI with a stable JSON summary of scores and moved atoms. **Current:** `cryomodel zonal-refine run PDB MAP OUT --radius R` and either `--center x,y,z` (**commas**, one shell token) or `--cx X --cy Y --cz Z` (optional `--chains`, `--passes`, `--weight-map`, `--map-density-threshold`, `--weight-density-anchor`, `--weight-density-gain`, `--map-anchor-eps` (optional map-fit anchoring vs pre-trial density), `--weight-rotamer`, `--json-log`, soft-shell flags, and Ramachandran flags `--rama-backbone`, `--rama-step-deg`, `--rama-max-shift-deg`, `--weight-rama`, `--weight-backbone-move`, `--rama-include-soft`, `--rama-nudge-favored`). Use ASCII `--` for flags, not an em dash.

### A0 — MVP (prove the loop)

- **Geometry:** User supplies **center + radius** (or equivalent sphere) in model coordinates; map on the same frame as the PDB (existing map I/O).
- **Mask:** **Hard** atoms/residues inside radius; include **whole residue** if any atom qualifies.
- **Motion:** **Side-chain χ** search only (reuse / extend rotamer + clash machinery aligned with `pdb-mutate` patterns); **backbone fixed**.
- **Objectives (v0):** Map correlation or sum-of-squared residuals in a masked region + van der Waals–style clash penalty. Ramachandran / rotamer as **optional** light penalties or deferred to A1 if schedule risk.
- **Output:** Refined PDB + log (per-residue delta scores, timings).
- **Budget:** Target **&lt; 1 minute** for ~10–20 residues on a typical laptop CPU (heuristic, to be benchmarked).

**Exit criteria:** Reproducible improvement on 2–3 internal test systems; no silent corruption of chemistry (bonds, seqids).

**Phase wrap (A0):** CLI shipped; center as `--center x,y,z` or `--cx/--cy/--cz`; `--chains`; `--json-log`; tests in `tests/test_zonal.py`. Known limits: hard mask only; no Ramachandran term; no workflow UI card yet.

### A1 — Soft shell + neighbor logic **(wrapped)**

- **Soft zone:** Residues in the sphere at `radius + soft_buffer` that are not in the hard set.
- **Staged optimization:** hard χ1 passes, then optional soft-shell χ1 under clash triggers.

### A2 — Ramachandran + rotamer priors **(wrapped, lite)**

- **Rotamer:** unchanged χ1 priors from A0 (Dunbrack-style triplets in `pick_best_chi1`).
- **Ramachandran:** general-case (non-Gly/Pro) **favored / allowed / outlier** via elliptical regions in φ–ψ space; **optional** stage 3 applies a **small** grid of Δφ, Δψ (defaults ±9° in 3° steps) only on residues in the zone, scoring clash + map + weighted rama prior + **penalty on Δφ²+Δψ²** so backbone motion stays minimal. Does not move Gly/Pro backbones; terminal residues skipped when φ or ψ is undefined.
- **Fragment safety:** standard φ/ψ moves rotate the **entire** N-terminal or C-terminal fragment of the chain. Stage 3 **only applies** Δφ if every residue **before** *i* lies in the hard∪soft zone, and Δψ only if every residue **after** *i* lies in the zone—otherwise that component is skipped. This prevents “local” Rama from translating the whole model out of the map during **global** multi-pass runs.
- **Reporting:** JSON log entries with `stage: "rama"`, `rama_before` / `rama_after`, and `dphi_deg` / `dpsi_deg`.

### A3 — Hardening

- Alternative zone spec: **multiple spheres** or **distance-from-atom-set** mask (same residue inclusion rule).
- **Tests:** Unit tests on masks, golden-ish regression on tiny PDB+map fixtures.
- **Docs:** Operator-facing page — when to use vs Phenix/Coot, weight tuning, failure modes.

---

## 3. Part B — Global meta-optimizer (after Part A is solid)

Part B is an **orchestration layer**: it proposes regions, schedules local runs, and monitors whole-model metrics. It **calls Part A** repeatedly; it does not implement a second refinement core.

### B1 — Region proposal (GMM or simpler)

- **Input:** Current model (Cα or heavy atoms) + optional map-derived weights later.
- **Method:** Fit a **Gaussian mixture** in 3D to obtain **K** overlapping ellipsoid-like regions; assign residues to one or more components (threshold / soft assignment) so patches **overlap**.
- **Simpler fallback:** Grid or random overlapping spheres for ablation before GMM is required.

**Note:** This use of GMM is **spatial clustering for zoning**, distinct from the exploratory **GMM + local NCC** rigid-body idea in `docs/GMM_LOCAL_NCC_FITTING_NOTE.md`.

### B2 — Schedule

- Optimize regions in **randomized order** each **macro-cycle**.
- **Overlap** reduces boundary artifacts (Schwarz / overlapping block idea); exact overlap policy is an implementation choice with benchmarks.

### B3 — Convergence

- Stop when **whole-model** metrics plateau (e.g. map correlation, aggregate clash / geometry proxies) or **no improvement for N** macro-cycles, plus **max cycles** cap.
- Frame as **empirical stabilization**, not guaranteed global optimum.

### B4 — Scope guardrails

- **Performance:** Total runtime scales with `(#regions × #cycles × local runtime)` — keep local A fast and cap cycles for interactive use.
- **Symmetry / NCS:** Orchestration-level design (master chain, GMM on master, χ propagation to copies) is specified in **`docs/ZONAL_GLOBAL_OVERLAP_AND_GMM.md`**.

---

## 4. Engineering dependencies (CryoModel)

- **Maps:** Existing MRC/CCP4 read, interpolation (trilinear), optional local normalization.
- **Structure:** `gemmi` for coordinates, residue sets, writing PDB/mmCIF.
- **Energies / χ:** Reuse patterns from **mutate** (clash, rotamer pick, optional map guide) — extend with zonal masks and global score aggregation.
- **CLI + tests:** Typer entry point, pytest fixtures with tiny map + fragment.

---

## 5. Risks & open questions

| Risk | Mitigation |
|------|------------|
| Map vs geometry weight imbalance | Expose weights; document defaults; optional short annealing schedule |
| Mask boundary bias | Padding, soft-edged map masks, overlap in Part B |
| Local minima | Part B random order + multiple cycles; user expectations set in docs |
| GMM “wrong K” | Make K user-tunable; provide dumb overlapping-sphere baseline |

**Open questions:** Minimum backbone motion (peptide plane?) for helices; explicit solvent in clash model; mmCIF output priority.

---

## 6. Suggested roadmap checkpoint

| Checkpoint | Definition |
|------------|------------|
| **A0 done** | ~~CLI + masks + χ local refine + benchmarks on 2–3 systems~~ **(wrapped)** |
| **A1 done** | ~~Soft shell + staged neighbor χ + CLI flags + tests~~ **(wrapped)** |
| **A2 done** | ~~Rama micro-moves + rotamer (χ1) + CLI + tests~~ **(wrapped)** |
| **B pilot** | GMM (or fallback) + 2 macro-cycles + convergence log on one full model |

---

## 7. Related documents

- `docs/ZONAL_GLOBAL_OVERLAP_AND_GMM.md` — overlapping GMM zoning, Schwarz-style overlap, NCS, SSE guardrails, convergence, and pitfalls (Part B detail).
- `docs/GMM_LOCAL_NCC_FITTING_NOTE.md` — different feature (rigid pose / NCC), not the same GMM role as Part B zoning.
- `PHASES.md` — “Future / advanced tooling” notes zonal A0–A2 shipped; A3+ and Part B remain roadmap items.
