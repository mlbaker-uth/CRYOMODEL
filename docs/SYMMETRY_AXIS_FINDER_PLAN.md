# Symmetry / symmetry axis finder — design plan

This document scopes a **symmetry and symmetry-axis discovery** tool for CryoModel: fast, robust enough for lab use, and able to express **multiple symmetry hypotheses** (e.g. different orders along a shared axis). The output should feed **segmentation** and **multi-body fitting** later.

---

## 1. Goals

### Primary outputs

1. **Symmetry identifier(s)** — one or more ranked hypotheses: family (C, D, helical, icosahedral), order / parameters, and confidence.
2. **Symmetry axis (or axes)** — a simple geometric representation, e.g. a **line in PDB/mmCIF** (pseudo-Cα or minimal backbone stub) plus a **JSON** sidecar with numeric detail (direction, point on axis, optional operators).

### Non-goals (initial phases)

- Full atomic symmetry enforcement or refinement (downstream tools).
- Replacing established SPA packages for global particle alignment; this targets **map-centric** axis discovery in the **reconstruction frame**.

---

## 2. Inputs


| Input                          | Role                                                                                                     |
| ------------------------------ | -------------------------------------------------------------------------------------------------------- |
| **Map** (MRC/CCP4)             | Primary signal; use same grid/interpolator conventions as the rest of CryoModel (e.g. gemmi `read_map`). |
| **Optional mask**              | Strongly recommended — focus density on the complex, suppress solvent/ice.                               |
| **Optional model** (PDB/mmCIF) | Optional constraint or reporting: axis relative to model vs map frame.                                   |


---

## 3. Modes

### Mode A — User-specified (“guided”)

- User selects **family**: C, D, helical, icosahedral (and optionally order, e.g. C11, D5, helical priors, icosahedral setting).
- Tool **estimates and refines** the **axis** (direction + origin along axis) and **family-specific parameters** within that hypothesis.
- Use when symmetry class is known but **orientation in the map** is not.

### Mode B — Complete search (“discovery”)

- **Coarse** search over a **discrete** set of candidate axes and a **limited** set of families/orders (or a user-supplied shortlist for speed).
- **Multi-hypothesis output**: ranked list of candidates, not only the single best peak.

### Augmentation: general families

- **Cₙ / Dₙ** — search over **n** in a user range (e.g. 2–16) or explicit list.
- **Helical** — search **(rise, twist)** on **multi-scale grids** (large vs small step) to cover actin-like vs amyloid-like regimes.
- **Icosahedral** — fix **5-3-2** relationships but **search orientation** in the map frame (rotation of the symmetry frame + optional translation of the symmetry center).

---

## 4. Multiple symmetries in one complex

Biology often has **different orders** in different radial or axial zones while sharing a **common axis** (e.g. C5 vs C11 in motor regions).

**Recommended decomposition:**

1. **Global axis finder** — dominant **rotation axis** from the map (or masked density), largely independent of exact *n*.
2. **Local order estimation** — along that axis, use **cylindrical shells** or **1D profiles vs radius** (or angular sectors) to find where different **n** give consistent rotational correlation / self-correlation peaks.
3. **Structured output** — JSON listing `axes[]` and `hypotheses[]` with optional `radial_range_A` / `z_along_axis_range` so downstream steps can apply **C5 in shell A, C11 in shell B** without forcing one global label.

**Status (multishell):** `cryomodel symmetry multishell PHASE0_DIR` (after phase 2+) writes `symmetry_multishell.json`: equal-width annuli in cylindrical radius ρ about the chosen axis (phase 3 refinement if present, else phase 2 global + COM), each shell with per-order Cₙ Pearson **r** and `best_n`. Cap ρ with `--max-radius-A` or `--radius-percentile`. See `cryomodel/symmetry/multishell_cn.py`.

---

## 5. Algorithm pipeline (fast and robust)

### Phase 0 — Preprocess

- **Downsample** map for search; optional band-pass / edge emphasis.
- Apply **mask**.
- **PCA / inertia** of high-density voxels for an initial elongation / principal-direction guess (often near a true symmetry axis for elongated assemblies).

**Status (implemented):** `cryomodel symmetry phase0 MAP.mrc OUT_DIR` writes `symmetry_phase0_downsample.mrc` and `symmetry_phase0.json` (principal axes, COM, voxel counts). See `cryomodel/symmetry/preprocess.py`.

**Gold regression data (lab):** EMD-22898 map + 7KJR model (C2), absolute threshold **0.5** in raw map units:

- Default paths (skipped in CI if missing):  
`CRYOMODEL_LOCAL/examples/emd_22898.map`  
`CRYOMODEL_LOCAL/examples/7kjr-no-het.pdb`
- Override with `CRYOMODEL_SYMMETRY_TEST_MAP`, `CRYOMODEL_SYMMETRY_TEST_PDB`, optional `CRYOMODEL_SYMMETRY_TEST_THRESHOLD`.
- Tests: `tests/test_symmetry_emd_22898.py` (phase-0 output + geometric check vs A/B CA COM direction).

### Phase 1 — Axis candidates (discrete)

Combine:

- **Near-cardinal** directions with a **coarse** tilt grid (Z common but not guaranteed; include X, Y).
- **PCA / inertia** directions.
- Optional: peaks from **rotationally averaged power spectrum** (familiar from SPA).

For each candidate direction, define **cylinder or slab** geometry and reduce to **1D/2D** summaries along the axis.

**Status (implemented):** After phase 0, run `cryomodel symmetry phase1 PHASE0_DIR` (same directory as `symmetry_phase0.json` / `symmetry_phase0_downsample.mrc`). Writes `symmetry_phase1.json` with deduplicated unit directions (`cardinal_tilt`, optional `diagonal`, `pca_axis_`*) and per-candidate **axial mass histograms**, **mean/rms radial distance** (Å), and integrated mass on voxels above the phase-0 threshold. See `cryomodel/symmetry/axis_candidates.py`. Options: `--tilt-deg`, `--no-diagonals`, `--axial-bins`.

### Phase 2 — Scoring (shared infrastructure, family-specific kernels)


| Family          | Idea                                                                                                                                     |
| --------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| **Cₙ**          | Rotational self-correlation of cylindrical slice / annulus; peak at angle 2\pi/n.                                                        |
| **Dₙ**          | Cₙ-style score plus **mirror** (reflection) consistency.                                                                                 |
| **Helical**     | Correlation under **joint rotation + translation** along axis; coarse (twist, rise) grid then refine.                                    |
| **Icosahedral** | Correlation under a **subset of icosahedral rotations** or spherical representation; **early reject** if the map is not globular enough. |


**Status (Cₙ implemented):** After phases 0 and 1, run `cryomodel symmetry phase2 PHASE0_DIR` (same directory). Writes `symmetry_phase2.json` with per-candidate Pearson **r** vs rotation by 2\pi/n about each axis through the phase-0 COM, on voxels above the phase-0 threshold. Options: `--orders` (comma-separated list, default 2–12), `--max-candidates`. See `cryomodel/symmetry/phase2_cn.py`.

**Status (Dₙ scoring implemented):** `cryomodel symmetry phase2d PHASE0_DIR` writes `symmetry_phase2d.json`. For each axis candidate and order *n*, score combines (i) Cₙ correlation around the candidate axis and (ii) best C2 correlation around a perpendicular axis, searched over in-plane angle (`--inplane-samples`). See `cryomodel/symmetry/phase2_dn.py`.

### Phase 3 — Refinement

- Sub-degree / sub-voxel refinement for the **top few** hypotheses only.
- Optional joint optimization of **axis, in-plane rotation, and origin** in a small bounded neighborhood.

**Status (Cₙ axis + pivot implemented):** After phase 2, run `cryomodel symmetry phase3 PHASE0_DIR`. Refines the top `--top` phase-2 candidates (by `best_score`) with **L-BFGS-B** on five parameters: small tilts of the axis in the tangent plane, shift of the pivot along the refined axis, and perpendicular pivot offsets. Phase 2 used the phase-0 COM as pivot; phase 3 allows that point to move within the configured bounds. Options: `--top`, `--max-tilt-deg`, `--max-shift-along-A`, `--max-shift-perp-A`, `--maxiter`. Writes `symmetry_phase3.json`. See `cryomodel/symmetry/phase3_refine.py`.

**Status (Dₙ axis + pivot refinement):** After phase 2D, run `cryomodel symmetry phase3d PHASE0_DIR`. Uses the same bounded local axis/pivot parameterization, but optimizes the Dₙ objective (combined Cₙ + perpendicular C2 evidence). Writes `symmetry_phase3d.json`. See `cryomodel/symmetry/phase3_dn_refine.py`.

### Phase 4 — Artifacts

- **Axis PDB/mmCIF**: e.g. two atoms at `p0` and `p0 + L * û` (Å), with `REMARK` pointing to JSON or embedding a short symmetry ID.
- **JSON**: `axes[]`, `hypotheses[]`, scores, and optional shell metadata for multi-order cases.

**Status (Cₙ CA trace PDB):** `cryomodel symmetry phase4 PHASE0_DIR` writes `symmetry_axis_ca.pdb` (default path) plus `symmetry_phase4.json`. The polyline uses **GLY CA** atoms every `--slice-step` **full-map voxels** along the axis (spacing Å = step × reference apix), clipped to the reference map’s orthorhombic box. The default reference map is `input_map` from phase 0; use `--map` only for an unfiltered twin that shares the **same** origin, apix, and dimensions. Axis geometry comes from `symmetry_phase3.json` when present (`--refinement-index`), else phase-2 global best with phase-0 COM as pivot. See `cryomodel/symmetry/phase4_axis_pdb.py`. Public entry point `load_symmetry_axis_geometry` shares axis/pivot resolution with multishell.

### One-shot pipeline

**Status:** `cryomodel symmetry find MAP.mrc OUT_DIR` runs phase 0 → 1 → phase2/phase2d → phase3/phase3d → multishell → axis PDB (each step optional via `--no-phase3`, `--no-multishell`, `--no-axis-pdb`). Supports both use cases: complete search (`--mode search`) and user-guided hypothesis (`--mode guided --guided-order N`), with family choice (`--family c|d`). Writes all intermediate artifacts plus `symmetry_find.json` and a family quality plot (`symmetry_scores_Cn.png` or `symmetry_scores_Dn.png`) to visualize score-vs-*n* within family. See `cryomodel/symmetry/pipeline_find.py` and `cryomodel/symmetry/score_plot.py`.

---

## 6. Performance and robustness

- **Fast**: coarse search only on **binned** maps; full resolution only for **final verification** of 1–3 hypotheses.
- **Robust**: always report **score margin** vs second place and qualitative **peak width** where possible.
- **Honest UX**: distinguish **dominant global axis** from **local order** so users do not over-interpret a single integer *n*.

---

## 7. CLI / CryoModel integration

**Implemented (Cₙ discovery):**

```text
cryomodel symmetry find MAP.mrc OUT_DIR [--mask MASK.mrc] [--downsample N] \
  [--density-threshold T | --density-percentile P] [--orders 2,3,...] \
  [--no-phase3] [--no-multishell] [--no-axis-pdb] [--n-shells 8] [--out-pdb PATH]
```

Individual steps remain available as `symmetry phase0` … `phase4` and `symmetry multishell`.

**Future:** `guided|search` modes, additional families (D/H/I), workflow-UI cards — follow the same UX rule as other tools (split when modes confuse users).

---

## 8. Validation

- **Synthetic phantoms**: known Cₙ, Dₙ, helical, icosahedral axes; add noise and optional missing-wedge stress tests.
- **Public EMDB** entries with documented symmetry (capsids, helical filaments).
- **Regression**: golden **JSON** outputs (axis direction within ε°, origin along axis within tolerance).

---

## 9. Phased delivery


| Phase   | Scope                                                                                                        |
| ------- | ------------------------------------------------------------------------------------------------------------ |
| **MVP** | Masked map → **Cₙ** only; guided + small-order search; axis PDB + JSON; binned-map search.                   |
| **V2**  | **Dₙ**; multi-hypothesis ranking; cylindrical shells (**Cₙ per shell implemented**; Dₙ mirror term not yet). |
| **V3**  | **Helical** coarse-to-fine grid + refinement.                                                                |
| **V4**  | **Icosahedral** orientation search; tighter integration with segmentation / multi-body fitting.              |


---

## 10. Open decisions (lock early)

- Default **map-only** vs optional **map + model** for difficult low-SNR cases.
- Whether “full” icosahedral search is **full 6D orientation** from the start vs **guided icosahedral + local perturbation** for speed.
- **Downstream contract**: minimal JSON (`direction` + `point`) vs richer `operators[]` for rigid transforms.

---

## 11. Recommended first lab-usable slice

For the first drop that matches common motor/filament narratives without boiling the ocean:

**MVP Cₙ + dominant axis + multi-shell order hints on that axis** — delivers a clear axis artifact and hints at **where** *n* changes, before full helical/icosa search.