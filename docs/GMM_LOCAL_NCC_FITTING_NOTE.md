# Hybrid GMM + local NCC rigid fitting (exploratory design note)

**Status:** Exploratory — not on the active roadmap. Revisit after more prototyping and after the current FoldHunter / `model2map` path is stable.

**Purpose:** Capture a coherent coarse-to-fine idea (from internal brainstorming) so it is not lost, without committing CryoModel to implementation yet.

---

## One-line summary

Use a **compact probe representation** (Gaussian mixture on density-weighted samples from the blurred model map) as a **cheap geometric prescreen**, then run **masked, local normalized cross-correlation** only on a **small set of pose candidates**. Optional **Markov / MCMC** exploration is explicitly **out of scope for a first version**.

## Why it might be worth doing

- Exhaustive global 6D correlation is expensive; **most of pose space is obviously wrong**. A fast coarse score can shrink work before FFT-heavy refinement.
- A GMM summarizes **mass distribution** (centers + weights + optional covariances) with far fewer parameters than the full voxel grid.
- **Masked local NCC** near good candidates is a credible **final arbiter**, especially for **subunit-in-assembly** maps where neighbors and solvent matter.
- Fits a **Python-first** stack if heavy steps stay in NumPy/SciPy/FFT (optional pyFFTW later).

## Suggested operating modes (when/if implemented)

| Mode | Use case | Coarse stage emphasis |
|------|----------|------------------------|
| **A — Subunit → subunit** | Two similarly sized maps | Rotation bank + translation scoring; simpler geometry |
| **B — Subunit → assembly** | Small probe, large target | Regional candidates, stronger peak clustering, symmetry duplicates expected |

Both modes share the same **high-fidelity refiner** (local masked NCC); they differ in **how candidates are generated**.

## Pipeline sketch (v1, deterministic)

1. **Preprocess probe map** (from existing PDB→map path): normalize, **mask** / crop to support, `float32`.
2. **Preprocess target**: normalize, optional smooth \(A_\sigma\), **3D integral images** for fast local mean/variance in refinement.
3. **Fit GMM to probe**: sample **above threshold**, **density-weighted**; store \(w_k, \mu_k, \Sigma_k\) (isotropic option for speed).
4. **Coarse poses**: for each rotation in a **coarse bank**, score translations — see **open questions** below — then **NMS / clustering**, keep top \(N\).
5. **Refine**: extract local subvolume(s); **FFT CC numerator** + **masked local NCC** denominator; small local rotation/translation refinement.
6. **Post**: rank, merge near-duplicate poses; optional symmetry-aware clustering if user supplies operators.

## Relationship to current CryoModel

- **`model2map` / `synthetic_from_model`**: supplies a **consistent** probe map on the target grid; this design assumes that (or equivalent) exists.
- **FoldHunter**: already does **coarse-to-fine FFT correlation** with rotation banks. A GMM stage would be an **alternative or additional coarse filter**, not a replacement for the final correlation step unless proven equivalent.

## Open questions / risks (read before coding)

- **Coarse translation must be defined precisely.** A score \(\sum_k w_k\, A_\sigma(R\mu_k + t)\) is a scalar per \((R,t)\). Scanning **all** translations requires either a **3D grid** over \(t\) (cost \(\propto N_t \times K\)) or reformulating as **convolution / FFT** against a rasterized sparse probe. The whiteboard version often glosses over this — close the gap in the first prototype spec.
- **Interpolation:** point samples of \(A_\sigma\) at \(R\mu_k + t\) should use **trilinear (or better) interpolation**, not naive nearest-neighbor, once smoothing is tight relative to voxel size.
- **Component count \(K\) vs cost:** large \(K\) (e.g. dozens–100+) × many rotations still adds up; start small and increase only if the coarse stage misses solutions.
- **Spherical / symmetric probes:** coarse orientation discrimination is weak; expect **more rotations** or heavier reliance on local NCC.
- **Conformational mismatch:** rigid GMM + rigid CC will **not** fix a wrong probe conformation; retain **more candidates** or integrate flexible tooling separately.
- **Markov / MCMC:** reserve for **multi-copy**, **clash / occupancy**, or **strong ambiguity**; not the default for single rigid-body v1.

## Possible module layout (future)

Not prescriptive — only a sketch: `density` / mask / normalize, `gmm` (fit + transform centers), `rotations`, `candidates` (peaks + clustering), `integral` (3D summed tables), `ncc` (masked local), `fit_single` / `fit_assembly` entrypoints.

## Next steps (when you pick this up again)

1. Write a **half-page math note** for one rotation \(R\): exact formula for the **coarse translation map** (grid vs FFT) and complexity.
2. **Prototype** GMM fit + coarse score + **one** local NCC refine on a toy map pair; measure miss rate vs FoldHunter on 1–2 real cases.
3. Decide **integration**: new CLI tool vs optional `foldhunter` coarse backend vs separate card.

---

*Derived from an internal design discussion (March 2026). Treat as RFC, not a commitment.*
