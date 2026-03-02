# Pathwalker2 Design Specification

**Status**: Draft  
**Author**: CryoModel team  
**Date**: 2024-11-11

---

## 1. Goals

Pathwalker2 is the next-generation backbone tracing engine for CryoModel. It targets 3–4 Å cryo-EM maps and improves upon the existing `pathwalk` pipeline by:

- Generating a **backbone-focused pseudoatom set** that suppresses bulky sidechains
- Allowing **optional visits** (skip junk pseudoatoms) and **multiple fragments**
- Enforcing **protein geometry constraints** during routing
- Anchoring **secondary-structure rails** before pathing
- Producing **ranked path fragments** with confidence metrics
- Providing a future path for **interactive curation and sequence threading**

The design is split into three steps:

1. **Trace discovery (automatic)**
2. **User-guided stitching and threading** (CLI first, GUI later)
3. **Refinement** (external tools today, “lite” RSR in the future)

The initial implementation will focus on Step 1, with CLI scaffolding for Steps 2–3.

---

## 2. High-Level Pipeline

### Step 1 – Automatic Trace Discovery

1. **Map preparation & masking**  
   - Optional LocScale-style local Z-normalization (6–8 Å window)  
   - Non-linear smoothing (Perona–Malik / bilateral) to preserve ridges  
   - Threshold to binary mask Ω

2. **Skeleton & ridge scoring**  
   - Distance transform (DT) on Ω (local thickness)  
   - Flux-based or thinning skeleton S  
   - Multi-scale Laplacian-of-Gaussian (σ≈1.2–2.0 Å) ridgeness score R  
   - Gradient direction ĝ per voxel

3. **Overcomplete seeding & pruning**  
   - Poisson-disk (1 Å) or dense grid sampling  
   - Keep candidates near skeleton (≤1.2 Å), high ridgeness, thickness within [0.8, 2.3] Å  
   - Farthest-point sampling to target pseudoatom count `N_seed` such that  
     `0.8 * N_res ≤ N_seed ≤ min(map_voxels, max(0.6 * N_res, 1500))`

4. **Secondary structure (SS) detection**  
   - Helix template correlation (length 10–18 Cα) + axis refinement  
   - Create “rails” of evenly spaced nodes (1.5–1.6 Å) collapsed to 3.8 Å  
   - Rails are must-visit ordered nodes with handedness metadata

5. **Partition discovery**  
   - Connected components in Ω → coarse partitions  
   - kNN graph (k≈10–12, r_max≈5.5 Å) on pseudoatoms  
   - Community detection (Louvain/HDBSCAN) within components → final partitions `P_j`

6. **Routing graph construction**  
   - Nodes: pseudoatoms (including rail nodes)  
   - Edges: connect `i→j` if distance ∈ [3.0, 4.8] Å and interpolated segment stays within Ω  
   - Pre-compute segment samples: min density, gradient alignment, thickness, curvature, pseudo-Rama likelihood

7. **Edge cost function**  
   ```
   C(i→j | k) =
       w_d * ((‖x_j - x_i‖ - 3.8) / σ_d)^2
     + w_m * 1 / (ε + minρ(i,j))
     + w_g * (1 - |ĝ_i · û_ij|)
     + w_t * thickness(i)
     + w_c * angle(k,i,j)^2
     - w_ang * log p*(θ_i[, τ_i])
     + w_b  (if backtracking)
     + w_x  (if segment intersects earlier path)
     + w_clash (if Cα–Cα < 3.2 Å vs earlier nodes)
   ```
   - Default weights (tunable via CLI):  
     `σ_d=0.20, w_d=1.0, w_m=2.0, w_g=1.0, w_t=0.5, w_c=0.25, w_ang=2.0, w_b=0.5, w_x=3.0, w_clash=2.0`
   - Pseudo-Rama `p*(θ[,τ])`: KDE per SS class (helix/strand/loop) blended with SS likelihood

8. **Optional-node routing**  
   - Construct VRP with disjunctions using OR-Tools  
   - One “vehicle” per partition; precedence constraints for rails  
   - Node penalties `pen_i = λ * f(map_z, ridgeness, SS proximity)`  
   - Paths (open) rather than cycles via dummy start/end nodes  
   - Optional fallback: beam search/A* per partition (state includes predecessor)

9. **Fragment scoring & selection**  
   ```
   Score(F) = Σ edges C(i→j)  + α * coverage% + β * SS consistency
              - γ * crossovers  - δ * clashes
   ```
   - Defaults: α=5.0, β=2.0, γ=4.0, δ=2.0  
   - Keep top-K fragments per partition (e.g., K=10)  
   - Export confidences (map Z, pseudo-Rama p*, edge support) to B-factors

**Outputs** (Step 1):
- `pathwalker2_fragments.pdb`: Cα positions, chain=partition, MODEL/altLoc=fragment ID
- `pathwalker2_meta.json`: threshold, filters, weights, partitions, rails, fragment stats, per-edge costs
- `pathwalker2_marks.cif`: pseudoatoms, rails, skeleton, partitions (for visualization)

### Step 2 – Stitching & Threading (future CLI/GUI)

1. **Fragment curation**  
   - Toggle fragments per partition  
   - Reverse direction (respect rail handedness)  
   - Highlight join suggestions (termini within 7–10 Å with compatible tangents)  
   - Split fragments at low-confidence spans (minρ or pseudo-Rama dips)

2. **Sequence threading (HMM/Viterbi)**  
   - States: residues in sequence  
   - Observations: fragment positions with emission scores mixing density, SS, side-chain likelihoods  
   - Transitions: i→i+1 with penalties for skips/gaps  
   - Output: Cα-only model with per-residue confidence (B-factor) and flagged low-confidence segments

3. **Feature registration**  
   - Adjust helical register (±1–2 residues) to align aromatics / Pro / Gly motifs with density  
   - Support multi-subunit mapping (assign chain IDs, symmetry hints)

**Outputs** (Step 2):
- `pathwalker2_selected.pdb`: curated Cα traces, threaded if sequence provided  
- `pathwalker2_selected.cif`: same + annotations  
- `pathwalker2_threading.json`: Viterbi scores, per-residue confidence, anchors used

### Step 3 – Refinement (future work)

- Export to external RSR tools (Phenix, ISOLDE, CCP-EM)  
- Plan for “lite” RSR: L-BFGS on map Z, virtual geometry, pseudo-Rama, clash avoidance  
- Optional Cβ prediction for better placement prior to full refinement

---

## 3. Proposed Package Layout

```
crymodel/pathwalker2/
  __init__.py
  map_prep.py           # smoothing, normalization, thresholding
  skeleton.py           # skeletonization, ridgeness, gradient
  seeding.py            # overcomplete sampling, pruning
  ss_detection.py       # helix detection & rail construction
  partition.py          # components, clustering
  graph.py              # node/edge structures, precomputed samples
  costs.py              # edge & fragment cost evaluations
  routing.py            # VRP/beam search wrappers
  fragments.py          # scoring, ranking, export utilities
  threading.py          # HMM/Viterbi sequence threading (Step 2)
  io.py                 # PDB/CIF/JSON helpers

crymodel/cli/pathwalker2.py          # Step 1 command
crymodel/cli/pathwalker2_thread.py   # Step 2 command (future)
```

---

## 4. Interfaces & CLI Sketch

### 4.1 Automatic Trace Discovery
```bash
crymodel pathwalker2 \
  --map map.mrc \
  --thresh 0.05 \
  --residues 840 \
  --seed ridge \
  --grid 1.0 \
  --ss helix:on sheet:off \
  --rails-lock \
  --partition auto \
  --routing vpr --optional-nodes \
  --k 12 --rmax 5.5 \
  --weights w_d=1.0,w_m=2.0,w_g=1.0,w_t=0.5,w_c=0.25,w_ang=2.0,w_b=0.5,w_x=3.0,w_clash=2.0 \
  --fragments K=10 --coverage-boost 5.0 \
  --out ./pw2_job
```

Key flags (initial implementation subset):
- `--seed {grid,poisson,ridge}`  
- `--map-prep {none,locscale,bilateral}`  
- `--ss helix:on/off sheet:on/off`  
- `--routing {vpr,beam}`  
- `--optional-nodes` (boolean)  
- `--weights w_d=...,w_m=...,...` (comma-separated overrides)  
- `--fragments K` (number of fragments per partition to keep)  
- `--out <dir>`

### 4.2 Threading CLI (Phase 2)
```bash
crymodel pathwalker2-thread \
  --fragments ./pw2_job/pathwalker2_fragments.pdb \
  --metadata ./pw2_job/pathwalker2_meta.json \
  --sequence subunitA.fasta \
  --chain A \
  --anchors helix.txt \
  --out ./pw2_job
```

Options:
- `--fragments`: Step-1 output PDB  
- `--metadata`: Step-1 JSON (weights, scores, SS info)  
- `--sequence`: FASTA or simple sequence string  
- `--chain`: Chain ID to assign  
- `--anchors`: File with SS annotations / anchor residues (optional)  
- `--out`: Output directory

---

## 5. Data Structures

- **Pseudoatom node**:  
  ```
  Node = {
    id, position (x,y,z),
    map_density, map_z,
    ridgeness, gradient_dir, thickness,
    ss_tag (optional), rail_id (optional)
  }
  ```

- **Edge** (`i→j`):  
  ```
  Edge = {
    source, target,
    distance, min_density, grad_alignment,
    thickness_penalty, curvature (requires predecessor),
    pseudo_rama_theta, tau
  }
  ```

- **Partition**: subset of nodes with metadata (component id, rail membership)

- **Fragment**: ordered list of node IDs + scores (Σ costs, coverage, SS consistency, etc.)

- **Metadata JSON**: captures all configuration parameters, weights, scoring components, per-node and per-edge annotations (for reproducibility)

---

## 6. Implementation Roadmap

### Phase 1 – Minimum Viable Pathwalker2 (Step 1)
1. Map prep module (LocScale-style Z-normalization, optional smoothing)
2. Skeleton + ridgeness computation
3. Seeding (ridge-aware pruning)
4. Helix detection & rail construction
5. Partition discovery (components + community detection)
6. Graph construction with edge sampling caches
7. Edge cost module (distance, density, gradient, thickness, pseudo-Rama)
8. Optional-node routing (OR-Tools VRP)  
   - fallback beam-search if OR-Tools absent
9. Fragment scoring & export
10. CLI `pathwalker2` with configuration file support

### Phase 2 – Stitching & Threading
1. Fragment curation CLI (selection, reversal, join suggestions)
2. Viterbi-based threading with emission & transition scores
3. Per-residue confidence export (B-factors)
4. Sequence/anchor IO helpers
5. Optional simple GUI (future milestone)

### Phase 3 – Refinement & Extras
1. Lite RSR (coordinate optimization against map Z & pseudo-Rama)
2. ML node classifier (optional feature)
3. Integration with external refinement pipelines
4. GUI enhancements and scripting hooks

---

## 7. Dependencies

- Required: `numpy`, `scipy`, `scikit-learn`, `gemmi`, `networkx`
- Optional: `ortools` (VRP solver), `pyvista`/`vedo` for future visualization
- Future ML assist: `torch`/`lightgbm` (optional)

---

## 8. Testing Strategy

1. **Unit tests** for each module (map prep, seeding, SS detection, graph costs)
2. **Integration tests** on benchmark maps with known backbone traces
3. **Regression tests** comparing Step-1 fragments with baseline pathwalker outputs
4. **Performance benchmarks** (runtime vs. map size, pseudoatom count)
5. **User validation test plan** for Step 2 CLI once implemented

---

## 9. Open Questions / Future Considerations

- Best default parameters for ridge detection across resolutions?
- Heuristic vs. exact handling of predecessor-dependent turn costs in OR-Tools?
- How to persist partition/fragment scoring for interactive GUI?
- Sequence threading for maps with multiple subunits (symmetry awareness)?
- Automated selection of thresholds (`--thresh`) and grid spacing?

---

## 10. Summary

Pathwalker2 replaces the monolithic “all pseudoatoms, single TSP” model with a modular pipeline:

1. **Ridge-first seeding** that focuses on backbone signal
2. **Secondary-structure anchoring** to lock confident rails
3. **Optional-node VRP routing** with protein-aware geometry costs
4. **Fragment ranking & confidence scoring**
5. A roadmap to **interactive stitching and threading** plus future refinement

Phase 1 can be delivered as a pure CLI workflow, producing higher-quality Cα fragments ready for user review. Later phases will improve user experience and integration with downstream refinement.

---

