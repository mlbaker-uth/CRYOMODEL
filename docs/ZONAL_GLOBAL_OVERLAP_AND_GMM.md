# Global zonal refinement: overlapping GMM regions and orchestration

**Status:** Design spec for **Part B** (global meta-optimizer) on top of the **local** engine `cryomodel zonal-refine run`. Not yet fully implemented; use this document when implementing `zonal-refine global` (or equivalent).

**See also:** `docs/ZONAL_REFINEMENT_PLAN.md` (Parts A/B roadmap), `docs/GMM_LOCAL_NCC_FITTING_NOTE.md` (different use of GMM — rigid density fitting, not zoning).

---

## 1. Purpose

Build a **full-model** refinement workflow by **stitching many overlapping local refinements** into repeated **macro-cycles**. The local objective (map + clash + rotamer + optional Ramachandran micro-moves) stays in `run_zonal_chi_refine`; the global layer only decides **where** and **when** to call it and how to **stop**.

This is deliberately **not** positioned as a provably global optimizer. It is **stochastic multi-pass local refinement** with empirical stopping rules, complementary to tools like Phenix real-space refinement and Coot.

---

## 2. Optimization framing: overlapping blocks

Overlapping local refinements combined with iteration match a standard pattern in large-scale optimization:

- **Block coordinate descent / alternating minimization** over subsets of degrees of freedom (here, side chains — and optionally small backbone moves — inside a zone).
- **Overlap between blocks** (**Schwarz domain decomposition**, overlapping subdomains) so boundary residues are updated in more than one local solve, reducing **boundary artifacts** that appear when disjoint patches are optimized independently.

**Architecture:** many **small, fast** solves; **shared information** through **spatial overlap** and **repeated macro-rounds** (with **shuffled region order** per round). The local engine’s **soft shell** remains the **fine-scale** boundary treatment inside each spherical (or sphere-bounding) call; overlap is the **coarse-scale** stitching mechanism.

---

## 3. GMMs for region proposal (spatial clustering)

**Role:** A **Gaussian mixture model (GMM)** in **3D** is a plausible way to propose **zone centers and shapes** from **atomic positions** (typically **Cα** per residue on a chosen chain; later optional **heavy-atom** features or **map-weighted** points).

- Each mixture component is an **ellipsoid-like** neighborhood in space (mean + covariance).
- **Soft assignment** of residues to components (e.g. posterior responsibilities, or “top-*m*” components, or threshold \(p_k \geq p_\text{min}\)) yields **overlapping patches**: one residue may belong to **multiple** regions for that macro-cycle’s scheduling.

**Tuning knobs (product / research):**

| Knob | Intent |
|------|--------|
| **K** (number of components) | Too small → one huge blob; too large → many tiny expensive zones. **CLI:** ``--gmm-components K`` sets K explicitly; otherwise ``K ≈ N_Cα / --target-residues-per-region`` (indirect knob). |
| **Covariance** | Spherical (simple) vs full / tied — **elongated** components can align with **helices** once a baseline exists. |
| **Soft-assignment policy** | Controls **how much overlap** (e.g. responsibility floor, top-2 regions). |
| **Minimum overlap** | Can be operationalized (e.g. average number of regions per residue, or pairwise co-membership for neighbors). |
| **Fallback** | Grid or **random overlapping spheres** for ablation if GMM is unstable (see pitfalls below). |

**Important:** This GMM is **zoning geometry only**. It does not by itself encode **secondary structure** or **map quality** unless extended (e.g. per-residue feature vectors, map-derived weights). A separate guardrail is **SSE-aware boundaries** (below).

**Not the same as:** `docs/GMM_LOCAL_NCC_FITTING_NOTE.md`, which sketches **GMM on map density** for **rigid** pose / NCC-style fitting — orthogonal role.

---

## 4. Objective coupling and global checks

**Coupling:** Optimizing region **A** changes clash and map scores for residues in region **B**. Overlap and iteration **mitigate** boundary bias but do not remove cross-talk.

**Practical mitigations (in order of expected V1 → V2):**

1. **Overlap + multiple macro-rounds** (primary).
2. **Periodic whole-model metrics** (cheap aggregates), e.g. each round or every *k* rounds: map correlation / mean sampled density / clash proxy on the **entire model** — used for **logging, plateau detection, and stopping**, not as a separate global solve.
3. **Later (if needed):** scheduling tricks — e.g. anneal weights, or damp repeated updates on **high-overlap** residues if oscillation appears in practice.

---

## 5. Convergence and stopping (honest framing)

There is **no** guarantee of convergence to a single global optimum. Acceptable **heuristic** stopping:

- **Max macro-rounds** (e.g. 5–10 for interactive use).
- **Plateau** in a **whole-model** metric (or master-chain-only variant under NCS).
- **No meaningful improvement** for **N** consecutive rounds (e.g. Cα RMSD change vs start-of-round snapshot below ε, or score delta below a threshold).

Document this as **empirical stabilization**, not optimality.

---

## 6. Performance at “full model” scale

Cost scales roughly as:

\[
(\text{\# region invocations per round}) \times (\text{\# macro-rounds}) \times (\text{local runtime per call}).
\]

Overlap **increases** the number of local calls per round. **V4-scale** workflows stay practical only if each **local** step remains **small and fast** (the existing zonal design). Caps on **K**, **max rounds**, and **maximum zone size** are first-class **product controls**.

---

## 7. GMM pitfalls

| Pitfall | Mitigation |
|---------|------------|
| **Wrong K** | User-tunable K; merge/split by assigned residue count; overlapping-sphere fallback. |
| **Boundary residues** | **Soft assignment** + local **soft-shell** behavior in `zonal-refine run`. |
| **Ill-conditioned covariances** | Prefer spherical or strongly regularized covariances in V1. |
| **Homomers / NCS** | **Symmetry-aware orchestration** (below), not a single GMM on superposed identical chains without thought. |

---

## 8. NCS and homomers (orchestration-level symmetry)

**Intended pattern (design target):**

- User specifies related chains, e.g. `--ncs A,B,C,D` with **first chain = master**.
- **Region proposal** (GMM, optional SSE constraints) runs on the **master chain** only (e.g. Cα positions along **A**).
- **Local refine** runs with **`chain_filter` = master ∪ copies** so every NCS chain can move **inside** the local sphere (χ1 and optional Rama), matching a multi-chain ``zonal-refine run --chains …``.
- **Rama:** φ/ψ moves are **fragment-limited** to the current hard∪soft residue set (see `ZONAL_REFINEMENT_PLAN.md` A2). Without that, standard peptide kinematics would drag the whole chain outside a small zone — the failure mode you see as the refined ribbon leaving the density during **global** runs.
- After each local call, for copy residues **outside** the sphere (e.g. homomer subunits far apart in space), **propagate** the master’s χ1 and any **accepted Rama Δφ, Δψ** from the run log to the matching residue on each copy (torsion-level sync; no crystallographic symmetry operator in Cartesian space).

**Validation:** warn or fail if copy chains lack expected residues relative to master.

---

## 9. Secondary structure: do not split helices and strands (v1)

**Goal:** Avoid cutting **HELIX** / **SHEET** segments in the middle when forming regions.

**v1 approach:** Use **PDB header** records (`HELIX` / `SHEET`) for the master chain — reuse patterns such as `parse_sse_from_pdb_header` in `cryomodel/domains/domain_identifier.py`. Treat each contiguous SSE run as **atomic** for boundary placement; allow cuts in **loop** or between SSE elements. **mmCIF** may need a parallel path (`_struct_conf` or equivalent) if header records are absent.

If no SSE records exist, fall back to **pure geometric** zoning with a clear **warning** in the log.

---

## 10. Implementation pointers (CryoModel)

- **Local engine:** `cryomodel.zonal.refine.run_zonal_chi_refine` — unchanged contract; global code passes `center_xyz`, `radius`, `chain_filter`, and existing weights / soft shell / Rama flags.
- **Global orchestration:** `cryomodel.zonal.global_refine.run_global_zonal_refine` (GMM once on initial master Cα layout; centers/radii updated each local call from current coordinates; SSE expansion via PDB header).
- **CLI:** `cryomodel zonal-refine global PDB MAP OUT --ncs A[,B,C,…]` — pass-through options for local χ1 / soft shell / Rama; `--json-log` writes a **global** summary (`write_global_result_json`).
- **Logging:** round- and region-level JSON for debugging overlap and plateau behavior.

---

## 11. Related documents

| Document | Relation |
|----------|----------|
| `docs/ZONAL_REFINEMENT_PLAN.md` | Part A (local) wrapped; Part B overview. |
| `docs/GMM_LOCAL_NCC_FITTING_NOTE.md` | Different GMM application (map / rigid fit). |
| `PHASES.md` | High-level roadmap for zonal + meta refinement. |
