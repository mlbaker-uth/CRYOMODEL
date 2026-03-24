# CRYOMODEL Codebase Summary

*Generated as of the latest update (map filtering, BaseHunter, ChimeraX bundle, user guide).*

---

## 1. CLI / Tools Overview

### Top-level commands (22)

| Command | Description |
|---------|-------------|
| `findligands` | Identify and cluster unmodeled density (waters/ligands) |
| `predictligands` | ML-based ligand prediction |
| `pathwalk` | Pathwalking (trace backbone in density) |
| `pathwalk-average` | Average pathwalk results |
| `pyhole` | Pore analysis (pyHole) |
| `pyhole-plot` | Pore plotting |
| `basehunter` | Nucleotide base classification (purine/pyrimidine) |
| `validate` | Map/model validation |
| `pdbcom` | Domain center-of-mass from PDB |
| `pdbdomain` | PDB domain identification |
| `fitcompare` | Fit comparison |
| `fitprep` | Preflight check (map/model) |
| `loopcloud` | Loop modeling in weak density |
| `pathwalker2` | New pathwalker (discover) |
| `version` | Print version |
| `train-ml` | Train ML model (lazy) |
| `train-ensemble` | Train ensemble (lazy) |
| `extract-features` | Extract features for ML (lazy) |
| `foldhunter` | Fold/structure search |
| `affilter` | AlphaFold model filtering |
| `workflow` | Run YAML/JSON workflow |
| `workflow-validate` | Validate workflow file |

### Command groups (subcommands)

| Group | Subcommands | Description |
|-------|-------------|-------------|
| **mapfilter** | `apply`, `list` | Map filtering (lowpass, highpass, gaussian, threshold, binary, etc.) |
| **dnabuild** | `build`, `build-2bp` | Build DNA from map / centerline |
| **dnaaxis** | `extract` | Extract DNA axis/centerline |
| **assistant** | `ask`, `suggest`, `explain`, `troubleshoot`, `resolution`, `diagnose` | Guided help and workflows |
| **log** | 3 commands | Command log view/export |

**Total distinct CLI entry points: 36** (22 standalone + 14 subcommands)

---

## 2. Support / Library Modules

| Package | Approx. lines | Role |
|---------|----------------|------|
| **cli** | 3,767 | Command-line interfaces (Typer) |
| **nucleotide** | 3,406 | BaseHunter, DNA axis, DNA builder, classification, templates |
| **ml** | 1,721 | ML training, prediction, features, ensemble |
| **finders** | 1,712 | Ligand/water finder pipeline, components, features |
| **validation** | 1,361 | Q-score, Ringer, local CC, geometry, resolution priors |
| **assistant** | 1,085 | Guided assistant, knowledge base |
| **fitting** | 1,026 | FoldHunter (structure search) |
| **pathalker** | 997 | Legacy pathwalking (pseudoatoms, TSP, averaging) |
| **pore** | 891 | pyHole pore analysis and plotting |
| **pathwalker2** | 745 | New pathwalker core |
| **domains** | 648 | Domain identification, PDB COM |
| **alphafold** | 439 | AlphaFold model filtering (affilter) |
| **workflow** | 403 | YAML/JSON workflow engine |
| **maps** | 323 | Map ops, FFT, **filters** (lowpass, gaussian, etc.) |
| **pseudo** | 289 | Pseudoatom generation (greedy, uniform, rich) |
| **io** | 287 | MRC/CCP4, PDB, features, export |
| **loops** | 285 | Loop cloud generation |
| **compare** | 270 | Fit comparison logic |
| **prep** | 215 | Fit prep (map/model) |
| **backends** | 148 | I/O backends (lite) |
| **core** | 39 | Types (MapVolume, ModelAtoms, etc.) |
| **datasets** | 27 | Simulation/data helpers |

**Total library/support Python (excl. CLI): 86 files, ~16,317 lines**

---

## 3. Lines of Code

| Scope | Python files | Lines |
|-------|--------------|-------|
| **crymodel/** (all) | 113 | **20,084** |
| └─ cli | 27 | 3,767 |
| └─ library (rest) | 86 | 16,317 |
| **chimerax-bundles/crymodel** | 5 | 1,096 |
| **Documentation (*.md)** | 22 | 3,563 |

**Total Python in repo (crymodel + ChimeraX bundle): ~21,180 lines**  
**Total docs (Markdown): 3,563 lines**

---

## 4. Map Filtering (New in This Update)

- **CLI**: `crymodel mapfilter apply`, `crymodel mapfilter list`
- **Filters**: lowpass, highpass, bandpass, gaussian, threshold, binary, laplacian, laplacian-sharpen, median, bilateral, butterworth-lowpass, butterworth-highpass, normalize (13 total)
- **Module**: `crymodel.maps.filters` (+ `crymodel.maps` __init__)
- **User guide**: New §4 “Map Filtering” in `CRYOMODEL_USER_GUIDE.md`

---

## 5. One-Line Summary

**CRYOMODEL** has **36 CLI entry points** (22 top-level commands + 14 subcommands across mapfilter, dnabuild, dnaaxis, assistant, log), **~20k lines** of Python in the main package, **~1.1k lines** in the ChimeraX bundle, and **~3.6k lines** of Markdown documentation.
