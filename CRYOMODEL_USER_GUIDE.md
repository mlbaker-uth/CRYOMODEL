# CryoModel User Guide

**CryoModel: Unified cryo-EM modeling toolkit**

Version 0.1.0

---

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Core Tools](#core-tools)
   - [findligands](#findligands)
   - [predictligands](#predictligands)
   - [validate](#validate)
4. [Map Filtering](#map-filtering)
   - [mapfilter](#mapfilter)
5. [Pathwalking Tools](#pathwalking-tools)
   - [pathwalk](#pathwalk)
   - [pathwalk-average](#pathwalk-average)
6. [Pore Analysis Tools](#pore-analysis-tools)
   - [pyhole](#pyhole)
   - [pyhole-plot](#pyhole-plot)
7. [Nucleotide Analysis](#nucleotide-analysis)
   - [basehunter](#basehunter)
8. [Domain Analysis](#domain-analysis)
   - [pdbcom](#pdbcom)
   - [pdbdomain](#pdbdomain)
9. [Model Comparison](#model-comparison)
   - [fitcompare](#fitcompare)
10. [Preflight Tools](#preflight-tools)
    - [fitprep](#fitprep)
11. [Loop Modeling](#loop-modeling)
    - [loopcloud](#loopcloud)
12. [Machine Learning Tools](#machine-learning-tools)
    - [extract-features](#extract-features)
    - [train-ml](#train-ml)
    - [train-ensemble](#train-ensemble)
13. [Utility Commands](#utility-commands)
    - [version](#version)

---

## Overview

CryoModel is a comprehensive toolkit for cryo-EM structure modeling, validation, and analysis. It provides tools for:

- **Ligand and water identification** with ML-based classification
- **Model validation** with resolution-aware metrics
- **Map filtering** (lowpass, highpass, bandpass, Gaussian, threshold, binary, Laplacian, median, bilateral, Butterworth, normalize)
- **Pathwalking** for protein backbone tracing
- **Pore analysis** for transmembrane channels
- **Nucleotide density comparison**
- **Domain analysis** and model comparison
- **Loop modeling** in weak density
- **Preflight checking** for map/model alignment

---

## Installation

```bash
# Install from source
pip install -e .

# Or with optional dependencies
pip install -e ".[ml,pathwalk]"
```

---

## Core Tools

### findligands

**Purpose**: Identify, cluster, and annotate unmodeled density as water/ions and ligands.

**Usage**:
```bash
crymodel findligands \
  --map <map.mrc> \
  --model <model.pdb> \
  --thresh <threshold> \
  [OPTIONS]
```

**Required Arguments**:
- `--map`: Input density map (.mrc)
- `--model`: Input model PDB/mmCIF
- `--thresh`: Density threshold

**Options**:
- `--mask-radius <float>`: Mask radius around atoms (default: 2.0 Å)
- `--micro-vvox-min <int>`: Minimum voxels for water/ion blobs (default: 1)
- `--micro-vvox-max <int>`: Maximum voxels for water/ion blobs (default: 12)
- `--zero-radius <float>`: Zero-out radius for greedy picking (default: 2.0 Å)
- `--water-dist-min <float>`: Minimum distance to protein for waters (default: 2.0 Å)
- `--water-dist-max <float>`: Maximum distance to protein for waters (default: 6.0 Å)
- `--ligand-dist-min <float>`: Minimum distance to protein for ligands (default: 2.0 Å)
- `--ligand-dist-max <float>`: Maximum distance to protein for ligands (default: 10.0 Å)
- `--half1 <path>`: Half-map 1 (.mrc) for half-map statistics
- `--half2 <path>`: Half-map 2 (.mrc) for half-map statistics
- `--ml-model <path>`: ML model for water/ion classification
- `--entry-resolution <float>`: Map resolution (Å) for resolution-based filtering
- `--keep-hydrogens`: Keep hydrogen atoms (default: False)
- `--out-dir <path>`: Output directory (default: "outputs")

**Outputs**:
- `masked_unmodeled.mrc`: Masked and thresholded unmodeled density
- `waters_map.mrc`: Water/ion density map
- `ligands_map.mrc`: Ligand density map
- `water.pdb`: Water pseudoatoms
- `candidate-waters.pdb`: Filtered and clustered water candidates
- `ligands.pdb`: Ligand pseudoatoms grouped by component
- `water-predictions.csv`: ML predictions for water candidates (if `--ml-model` provided)
- `sites.csv`: Feature table with all computed metrics
- `assigns.json`: Program parameters and debug info

**Example**:
```bash
crymodel findligands \
  --map examples/emd_22898.map \
  --model examples/7kjr-no-het.pdb \
  --thresh 0.5 \
  --mask-radius 2.0 \
  --micro-vvox-min 1 \
  --micro-vvox-max 12 \
  --half1 examples/emd_22898_half1.map \
  --half2 examples/emd_22898_half2.map \
  --ml-model TRAINING/models/ensemble \
  --entry-resolution 2.2 \
  --out-dir outputs
```

---

### predictligands

**Purpose**: Classify ligand components using rule-based and template matching.

**Usage**:
```bash
crymodel predictligands \
  --ligands-pdb <ligands.pdb> \
  --ligand-map <ligands_map.mrc> \
  --model <model.pdb> \
  [OPTIONS]
```

**Required Arguments**:
- `--ligands-pdb`: Ligand pseudoatoms PDB from findligands
- `--ligand-map`: Ligand density map (.mrc)
- `--model`: Protein model PDB/mmCIF

**Options**:
- `--sites-csv <path>`: Sites CSV from findligands (optional)
- `--min-pseudoatoms <int>`: Minimum pseudoatoms per component (default: 3)
- `--max-pseudoatoms <int>`: Maximum pseudoatoms per grouped component (default: 80)
- `--threshold-original <float>`: Original threshold for component extraction (default: 0.5)
- `--threshold-lower <float>`: Lower threshold for connectivity (default: 0.3)
- `--distance-threshold <float>`: Distance threshold for grouping (default: 3.0 Å)
- `--out-dir <path>`: Output directory (default: "outputs")

**Outputs**:
- `ligand-predictions.csv`: Class predictions with features and scores

**Example**:
```bash
crymodel predictligands \
  --ligands-pdb outputs/ligands.pdb \
  --ligand-map outputs/ligands_map.mrc \
  --model examples/7kjr-no-het.pdb \
  --sites-csv outputs/sites.csv \
  --min-pseudoatoms 3 \
  --max-pseudoatoms 80 \
  --out-dir outputs
```

---

### validate

**Purpose**: Resolution-aware model validation using multiple validation metrics.

**Usage**:
```bash
crymodel validate \
  --model <model.pdb> \
  --map <map.mrc> \
  [OPTIONS]
```

**Required Arguments**:
- `--model`: Input model PDB/mmCIF
- `--map`: Full map (.mrc)

**Options**:
- `--half1 <path>`: Half-map 1 (.mrc)
- `--half2 <path>`: Half-map 2 (.mrc)
- `--localres <path>`: Local resolution map (.mrc)
- `--priors <path>`: Resolution priors YAML file
- `--fit-priors`: Fit priors from this data
- `--weights <path>`: Model weights file (for future ML)
- `--out-dir <path>`: Output directory (default: "outputs")

**Outputs**:
- `features.csv`: Per-residue features with all validation metrics
- `priors.yaml`: Resolution-aware priors (if `--fit-priors` used)

**Example**:
```bash
crymodel validate \
  --model structure.pdb \
  --map full_map.mrc \
  --half1 half1.mrc \
  --half2 half2.mrc \
  --localres local_resolution.mrc \
  --fit-priors \
  --out-dir outputs
```

---

## Map Filtering

### mapfilter

**Purpose**: Apply common cryo-EM map filters. Takes an input map, a filter type, and options; writes a filtered map with the given output name. Useful for preprocessing (lowpass, normalize), masking (threshold, binary), denoising (median, bilateral, gaussian), or enhancement (laplacian, bandpass).

**Commands**:
- `crymodel mapfilter apply <input.mrc> <output.mrc> --filter <type> [options]` — apply a filter and write the result
- `crymodel mapfilter list` — list all available filters and their options

**Usage**:
```bash
crymodel mapfilter apply <input_map> <output_map> --filter <filter_type> [OPTIONS]
crymodel mapfilter list
```

**Arguments**:
- *input_map*: Input MRC/CCP4 map path
- *output_map*: Output filtered map path

**Common options** (depend on filter):
- `--filter`, `-f`: Filter type (required for apply)
- `--resolution`, `-r`: Resolution in Å (lowpass, highpass, butterworth-*)
- `--low-res`, `--high-res`: Bandpass cutoffs in Å
- `--sigma-vox`: Gaussian blur sigma or median size in voxels
- `--median-size`: Median filter half-size (cubic 2×size+1)
- `--threshold`, `-t`: Threshold value (threshold, binary)
- `--below-value`: Value to set below threshold (default: 0)
- `--sharpen-strength`: Strength for laplacian-sharpen (default: 1)
- `--sigma-spatial`, `--sigma-range`: Bilateral filter parameters
- `--butterworth-order`: Order for Butterworth/bandpass (default: 4)
- `--gaussian-rolloff` / `--sharp-cutoff`: Use Gaussian or sharp cutoff for lowpass/highpass (default: Gaussian)

**Included filters**

| Filter | Description | Key options |
|--------|-------------|-------------|
| **lowpass** | Low-pass Fourier filter; attenuate frequencies beyond cutoff | `--resolution` (Å), `--gaussian-rolloff` / `--sharp-cutoff` |
| **highpass** | High-pass Fourier filter; remove low frequencies | `--resolution` (Å) |
| **bandpass** | Keep frequencies between two cutoffs (optional Butterworth) | `--low-res`, `--high-res` (Å), `--butterworth-order` |
| **gaussian** | Spatial Gaussian blur | `--sigma-vox` |
| **threshold** | Set values below threshold to a given value (e.g. 0) | `--threshold`, `--below-value` |
| **binary** | Binary mask: 1 where density ≥ threshold, else 0 | `--threshold` |
| **laplacian** | Laplacian (edge enhancement) | — |
| **laplacian-sharpen** | Sharpen by subtracting scaled Laplacian | `--sharpen-strength` |
| **median** | Median filter (noise reduction, edge-preserving) | `--median-size` or `--sigma-vox` |
| **bilateral** | Edge-preserving bilateral filter (3D) | `--sigma-spatial`, `--sigma-range` |
| **butterworth-lowpass** | Butterworth low-pass (smooth rolloff) | `--resolution`, `--butterworth-order` |
| **butterworth-highpass** | Butterworth high-pass | `--resolution`, `--butterworth-order` |
| **normalize** | Zero mean, unit variance (preprocessing) | — |

**Examples**:
```bash
# Low-pass filter at 4 Å resolution
crymodel mapfilter apply map.mrc lowpass_4A.mrc -f lowpass -r 4

# Gaussian blur (sigma = 2 voxels)
crymodel mapfilter apply map.mrc blurred.mrc -f gaussian --sigma-vox 2

# Threshold: set density below 0.5 to 0
crymodel mapfilter apply map.mrc masked.mrc -f threshold -t 0.5

# Binary mask (density ≥ 0.3 → 1, else 0)
crymodel mapfilter apply map.mrc binary.mrc -f binary -t 0.3

# Bandpass 3–10 Å with Butterworth order 4
crymodel mapfilter apply map.mrc band.mrc -f bandpass --low-res 10 --high-res 3 --butterworth-order 4

# Laplacian sharpening
crymodel mapfilter apply map.mrc sharp.mrc -f laplacian-sharpen --sharpen-strength 0.5

# Median filter for noise reduction
crymodel mapfilter apply map.mrc denoised.mrc -f median --median-size 2

# Normalize to zero mean, unit variance
crymodel mapfilter apply map.mrc normalized.mrc -f normalize

# List all filters and options
crymodel mapfilter list
```

**Output**: A single MRC/CCP4 map with the same grid and origin as the input, containing the filtered density.

---

## Pathwalking Tools

### pathwalk

**Purpose**: Trace protein backbone through density using TSP-based pathfinding.

**Usage**:
```bash
crymodel pathwalk \
  --map <map.mrc> \
  --model <model.pdb> \
  [OPTIONS]
```

**Required Arguments**:
- `--map`: Input density map (.mrc)
- `--model`: Reference model PDB/mmCIF (for Cα count)

**Options**:
- `--thresh <float>`: Density threshold
- `--method <str>`: Clustering method: kmeans, spectral, agglomerative, meanshift, gmm, birch (default: kmeans)
- `--tsp-solver <str>`: TSP solver: ortools or lkh (default: ortools)
- `--iterations <int>`: Number of optimization iterations (default: 1)
- `--out-dir <path>`: Output directory (default: "outputs")

**Outputs**:
- `path_*.pdb`: Pathwalking results for each iteration

**Example**:
```bash
crymodel pathwalk \
  --map density.mrc \
  --model reference.pdb \
  --thresh 0.5 \
  --method kmeans \
  --tsp-solver ortools \
  --iterations 3 \
  --out-dir outputs
```

---

### pathwalk-average

**Purpose**: Average multiple pathwalking runs and compute probabilities.

**Usage**:
```bash
crymodel pathwalk-average \
  --paths <path1.pdb,path2.pdb,...> \
  --out <output.pdb>
```

**Required Arguments**:
- `--paths`: Comma-separated list of path PDB files
- `--out`: Output averaged path PDB

**Options**:
- `--probabilities`: Include probability scores in output

**Example**:
```bash
crymodel pathwalk-average \
  --paths outputs/path_1.pdb,outputs/path_2.pdb,outputs/path_3.pdb \
  --out outputs/path_averaged.pdb \
  --probabilities
```

---

## Pore Analysis Tools

### pyhole

**Purpose**: Calculate pore profiles through transmembrane channels/tunnels.

**Usage**:
```bash
crymodel pyhole analyze \
  --pdb <structure.pdb> \
  --top <selection> \
  --bottom <selection> \
  [OPTIONS]
```

**Required Arguments**:
- `--pdb`: Input structure PDB/mmCIF
- `--top`: Top residue selection (e.g., "A:123" or "A:123,B:456")
- `--bottom`: Bottom residue selection

**Options**:
- `--step <float>`: Step size along axis (default: 1.0 Å)
- `--eps <float>`: Contact epsilon (default: 0.25 Å)
- `--no-h`: Exclude hydrogen atoms
- `--vdw-json <path>`: Custom VDW radii JSON file
- `--rings <int>`: Number of rings for mesh PDB (default: 24)
- `--out-prefix <str>`: Output file prefix (default: "pyhole_out")
- `--probe <float>`: Probe radius for accessible volume (default: 0.0 Å)
- `--conductivity <float>`: Conductivity (default: 1.5 S/m)
- `--occupancy <str>`: Occupancy metric: 'hydro' or 'electro' (default: 'hydro')
- `--hydro-scale <str>`: Hydrophobicity scale: 'raw' or '01' (default: 'raw')
- `--electro-scale <str>`: Electrostatics scale: 'raw' or '01' (default: 'raw')
- `--passable-json <path>`: Passability radii JSON file
- `--centerline <str>`: Centerline type: 'straight' or 'curved' (default: 'straight')
- `--adaptive`: Use adaptive sampling
- `--slope-thresh <float>`: Slope threshold for adaptive sampling (default: 0.5)
- `--max-refine <int>`: Maximum refinement iterations (default: 3)
- `--curve-radius <float>`: Curve radius for curved centerline (default: 2.0 Å)
- `--curve-iters <int>`: Curve iteration count (default: 3)
- `--interactive`: Interactive mode for top/bottom selection

**Outputs**:
- `*_centerline.pdb`: Pore centerline
- `*_mesh.pdb`: Pore mesh with CONECT records
- `*.csv`: Profile data with radius, hydrophobicity, electrostatics
- `*_summary.json`: Summary statistics (volume, resistance, conductance, passability)

**Example**:
```bash
crymodel pyhole analyze \
  --pdb channel.pdb \
  --top "A:123" \
  --bottom "A:456" \
  --step 1.0 \
  --centerline curved \
  --adaptive \
  --out-prefix pore_analysis
```

---

### pyhole-plot

**Purpose**: Create publication-quality plots of pyHole pore profiles.

**Usage**:
```bash
crymodel pyhole-plot plot \
  <input1,input2,...> \
  [OPTIONS]
```

**Required Arguments**:
- Inputs: Comma-separated list of pyHole output prefixes or CSV/JSON files

**Options**:
- `--out <str>`: Output basename (default: "pyhole_plot")
- `--overlay`: Overlay all inputs in one plot
- `--grid <str>`: Grid layout (e.g., "1x5", "2x3")
- `--labels <str>`: Comma-separated labels matching inputs
- `--titles <str>`: Comma-separated titles (grid mode)
- `--ylim <str>`: Radius range "lo,hi" (default: "0.5,8.0")
- `--species <str>`: Passability species to shade (default: "water")
- `--hlines <str>`: Reference lines "y[:label],y[:label],..." (default: "1.4:water")
- `--pdf`: Also save PDF alongside PNG
- `--style-paper`: Apply publication-quality styling
- `--swap-axes`: Swap axes (radius on X, s_A on Y)
- `--secondary <str>`: Secondary curve: 'hydro', 'electro', or 'occ'
- `--sec-ylim <str>`: Limits for secondary axis "lo,hi"
- `--sec-label <str>`: Override label for secondary axis
- `--sec-order <str>`: Secondary axis direction: 'asc' or 'desc' (default: 'asc')
- `--primary-color <str>`: Primary curve color(s), comma-separated
- `--secondary-color <str>`: Secondary curve color

**Outputs**:
- `*.png`: Plot image
- `*.pdf`: Plot PDF (if `--pdf` used)

**Example**:
```bash
# Single plot
crymodel pyhole-plot plot outputs/pore_analysis \
  --out fig1C \
  --ylim 0.5,8.0 \
  --species water

# Overlay multiple profiles
crymodel pyhole-plot plot stateA,stateB \
  --overlay \
  --labels "A,B" \
  --primary-color "black,orange" \
  --out overlay

# Grid layout
crymodel pyhole-plot plot P1,P2,P3,P4,P5 \
  --grid 1x5 \
  --out fig2 \
  --titles "Prot1,Prot2,Prot3,Prot4,Prot5"
```

---

## Nucleotide Analysis

### basehunter

**Purpose**: Compare and sort nucleotide density at near-atomic resolutions.

**Usage**:
```bash
crymodel basehunter compare \
  --input-file <pairs.txt> \
  --threshold <float> \
  [OPTIONS]
```

**Required Arguments**:
- `--input-file`: File with volume directory and pairs
- `--threshold`: Density threshold for volumes (unless per-pair thresholds are provided)

**Options**:
- `--template-dir <path>`: Template directory (defaults to `data/DNA-TEMPLATES` if present)
- `--out-dir <path>`: Output directory (default: "outputs")
- `--max-iterations <int>`: Maximum Monte Carlo iterations (default: 1000)
- `--min-stability <int>`: Minimum stability for convergence (default: 100000)
- `--min-improvement <float>`: Minimum improvement for convergence (default: 1e-4)
- `--exploration-chance <float>`: Exploration probability (default: 0.1)
- `--force-iterations <int>`: Force minimum iterations (default: 250)

**Input File Format**:
```
/path/to/volume/directory
volume1.mrc volume2.mrc
volume3.mrc volume4.mrc
...
```

**Per-pair thresholds**:
```
/path/to/volume/directory
volume1.mrc volume2.mrc 0.45
volume3.mrc volume4.mrc 0.55
```

**Outputs**:
- `group1.txt`, `group2.txt`: Group assignments
- `group1_with_ncc.txt`, `group2_with_ncc.txt`: Group assignments with NCC statistics
- `avg_group1.mrc`, `avg_group2.mrc`: Average volumes for each group

**Example**:
```bash
crymodel basehunter compare \
  --input-file volume_pairs.txt \
  --threshold 0.5 \
  --out-dir outputs
```

### dnaaxis

**Purpose**: Trace a medial-axis-like centerline through a dsDNA density map and output a PDB polyline and MRC axis map.

**Usage**:
```bash
crymodel dnaaxis extract \
  --map <map.mrc> \
  --threshold <threshold> \
  [OPTIONS]
```

**Required Arguments**:
- `--map`: Input density map (.mrc) containing dsDNA
- `--threshold`: Density threshold for the binary mask

**Options**:
- `--n-points <int>`: Total number of points in final polyline (default: auto from length)
- `--guides-pdb <path>`: Ordered guide PDB (all ATOM/HETATM records used in file order)
- `--endpoints-pdb <path>`: Alias for a 2-point guide PDB
- `--power <float>`: Cost exponent for inverse distance-transform weighting (default: 3.0)
- `--eps <float>`: Stabilizer added to distance transform (default: 1e-3)
- `--cap-fraction <float>`: Fraction of voxels at each PC1 extreme for automatic termini (default: 0.03)
- `--close-iters <int>`: Binary closing iterations (default: 0)
- `--smooth <float>`: Spline smoothing factor per segment (default: 0.4)
- `--pre-smooth-sigma <float>`: Gaussian sigma for segment smoothing (default: 1.0)
- `--pre-smooth-iters <int>`: Pre-smoothing passes per segment (default: 2)
- `--guide-search-radius <float>`: Search radius (Angstrom) around each guide point (default: 8.0)
- `--guide-dt-weight <float>`: Weight on distance transform when anchoring guide points (default: 2.0)
- `--max-guide-span-A <float>`: Warn if guide span exceeds this distance (default: 60.0)
- `--point-refine-window-A <float>`: Arc-length search window for point refinement (default: 0.8)
- `--point-refine-dt-weight <float>`: DT weight for point refinement (default: 0.35)
- `--recenter-radius-A <float>`: Local recenter search radius (default: 1.2)
- `--recenter-dt-weight <float>`: DT weight for recentering (default: 1.0)
- `--recenter-disp-weight <float>`: Displacement penalty for recentering (default: 0.35)
- `--recenter-iters <int>`: Recenter iterations (default: 1)
- `--rise-per-bp <float>`: Rise per base pair in Angstroms (default: 3.4)
- `--target-spacing <float>`: Override spacing between points (Angstroms)
- `--count-mode <round|ceil|floor>`: Auto point count mode when `--n-points` omitted
- `--report <path>`: Optional text report path
- `--out-pdb <path>`: Output PDB path (default: `dna_axis.pdb`)
- `--out-mrc <path>`: Output axis MRC path (default: `dna_axis.mrc`)

**Notes**:
- If `--guides-pdb` or `--endpoints-pdb` is provided, the centerline is traced segment-by-segment between consecutive guide points.
- If no guides are provided, the tool estimates endpoints from the binary mask and traces a single segment.

**Example**:
```bash
crymodel dnaaxis extract \
  --map 1BNA.mrc \
  --threshold 0.26 \
  --endpoints-pdb endpoints.pdb \
  --out-pdb 1BNA_centerline.pdb \
  --out-mrc 1BNA_centerline.mrc \
  --report 1BNA_centerline_report.txt
```

### dnabuild

**Purpose**: Build an initial poly-AT dsDNA model from a dsDNA-only density map.

**Usage**:
```bash
crymodel dnabuild build \
  --map <map.mrc> \
  --n-bp <num_basepairs> \
  --threshold <threshold> \
  [OPTIONS]
```

**Required Arguments**:
- `--map`: Input density map (.mrc) containing dsDNA
- `--n-bp`: Number of base pairs to build
- `--threshold`: Density threshold for peak detection

**Options**:
- `--out-pdb <path>`: Output PDB path (default: `dna_model.pdb`)
- `--sequence-file <path>`: Optional file with two sequences (one per strand)
- `--template-pdb <path>`: Template PDB for base-pair extraction (default: `data/DNA-TEMPLATES/AT-template.pdb`)
- `--min-distance-vox <int>`: Minimum spacing between base pairs (voxels)
- `--resolution <float>`: Template density resolution (Å)
- `--output-swapped`: Also output swapped-strand model when sequences provided

**Sequence File Format**:
```
ACGTA...
TGCAT...
```

**Example**:
```bash
crymodel dnabuild build \
  --map dna_only.mrc \
  --n-bp 12 \
  --threshold 0.45 \
  --out-pdb dna_init.pdb
```

**2-bp template builder**:

**Purpose**: Build a poly-AT dsDNA model using a rigid 2-bp template placed along a centerline.

**Usage**:
```bash
crymodel dnabuild build-2bp \
  --centerline-pdb <centerline.pdb> \
  --template-2bp-pdb <2AT-template.pdb> \
  --out-pdb <model.pdb> \
  [OPTIONS]
```

**Options**:
- `--map <map.mrc>`: Optional map for global phase optimization
- `--n-bp <int>`: Override base-pair count (default: inferred from centerline length)
- `--target-spacing <float>`: Target spacing between base pairs (default: 3.4)
- `--twist-deg <float>`: Twist per base pair (default: 35.0)
- `--trim-start-bp/--trim-end-bp <int>`: Trim base pairs from start/end
- `--trim-start-A/--trim-end-A <float>`: Trim distance (Å) from start/end
- `--global-phase-step-deg <float>`: Phase search step (default: 5.0)
- `--no-global-phase-opt`: Disable global phase optimization
- `--local-refine`: Enable per-step local refinement
- `--local-shift-A <float>`: Max local shift (Å)
- `--local-shift-step-A <float>`: Local shift step (Å)
- `--local-twist-deg <float>`: Max local twist (deg)
- `--local-twist-step-deg <float>`: Local twist step (deg)
- `--backbone-only-score`: Use backbone atoms only when scoring
- `--report <path>`: Optional report path

**Example**:
```bash
crymodel dnabuild build-2bp \
  --centerline-pdb 906_centerline_v7.pdb \
  --template-2bp-pdb data/DNA-TEMPLATES/2AT-template.pdb \
  --out-pdb 906_polyAT_2bp_model.pdb \
  --map 906-DNA-map.mrc \
  --report 906_polyAT_2bp_model_report.txt
```

---

## Domain Analysis

### pdbcom

**Purpose**: Compute domain centers of mass and output as PDB.

**Usage**:
```bash
crymodel pdbcom \
  --model <model.pdb> \
  --domains <domains.json> \
  [OPTIONS]
```

**Required Arguments**:
- `--model`: Input model PDB/mmCIF
- `--domains`: Domain specification JSON file

**Options**:
- `--out-prefix <str>`: Output file prefix (default: "domains_com")
- `--mass-weighted/--no-mass-weighted`: Use mass-weighted COM (default: True)
- `--atoms <str>`: Atom filter: 'all', 'backbone', or 'CA' (default: 'all')

**Domain JSON Format**:
```json
{
  "LBD-S1": {"A": ["45-125"]},
  "LBD-S2": {"A": ["260-330"]},
  "TM": {"A": "350-600"},
  "CTD": {"A": "601-720"}
}
```

**Outputs**:
- `*.pdb`: PDB with one pseudo-atom per domain
- `*.csv`: Domain COMs with coordinates, atom counts, masses

**Example**:
```bash
crymodel pdbcom \
  --model structure.pdb \
  --domains domains.json \
  --mass-weighted \
  --atoms all \
  --out-prefix domains_com
```

---

### pdbdomain

**Purpose**: Automatically identify structural domains from a PDB/mmCIF model.

**Usage**:
```bash
crymodel pdbdomain \
  --model <model.pdb> \
  [OPTIONS]
```

**Required Arguments**:
- `--model`: Input model PDB/mmCIF

**Options**:
- `--chain <id>`: Chain ID (default: first chain)
- `--n-domains <int>`: Force number of domains (optional)
- `--merge-distance <float>`: Merge distance in Å (default: 25.0)
- `--seed-size <int>`: Residues per seed segment (default: 20)
- `--min-domain-residues <int>`: Merge smaller domains into nearest (default: 50)
- `--prefer-gaps/--no-prefer-gaps`: Prefer domain breaks at sequence gaps (default: True)
- `--gap-window <int>`: Residues to search for nearest gap (default: 10)
- `--gaps-only/--no-gaps-only`: Only allow domain breaks at sequence gaps (default: False)
- `--sse-source <str>`: SSE source: `header`, `dssp`, `auto`, or `none` (default: header)
- `--sse-window <int>`: Residues to search for a non‑SSE boundary (default: 10)
- `--out-prefix <str>`: Output prefix (default: "domains")
- `--write-pdb/--no-write-pdb`: Write PDB with domain IDs in B-factor (default: True)

**Outputs**:
- `*.json`: Domain specification compatible with `pdbcom`
- `*.csv`: Per-residue domain assignments
- `*.pdb`: PDB with domain IDs in B-factor (optional)

**Example**:
```bash
crymodel pdbdomain \
  --model structure.pdb \
  --chain A \
  --merge-distance 22 \
  --seed-size 20 \
  --out-prefix domains
```

---

## Model Comparison

### fitcompare

**Purpose**: Align and compare models across conformational states.

**Usage**:
```bash
crymodel fitcompare compare \
  --model-a <modelA.pdb> \
  --model-b <modelB.pdb> \
  [OPTIONS]
```

**Required Arguments**:
- `--model-a`: Reference model PDB/mmCIF
- `--model-b`: Model to align PDB/mmCIF

**Options**:
- `--out-dir <path>`: Output directory (default: "outputs")
- `--anchors <str>`: Anchor selection (e.g., "A:100-160,B:20-45")
- `--domains <path>`: Domain specification JSON (for domain summaries)

**Outputs**:
- `fitcompare_superposed.pdb`: Aligned structure B in A's frame
- `per_residue_deltas.csv`: Per-residue RMSD (Cα, backbone, sidechain)

**Example**:
```bash
crymodel fitcompare compare \
  --model-a stateA.pdb \
  --model-b stateB.pdb \
  --anchors "A:100-160,B:20-45" \
  --out-dir outputs
```

---

## Preflight Tools

### fitprep

**Purpose**: Preflight checker for maps & models before fitting/validation.

**Usage**:
```bash
crymodel fitprep check \
  --model <model.pdb> \
  --map <map.mrc> \
  [OPTIONS]
```

**Required Arguments**:
- `--model`: Input model PDB/mmCIF
- `--map`: Full map (.mrc)

**Options**:
- `--half1 <path>`: Half-map 1 (.mrc)
- `--half2 <path>`: Half-map 2 (.mrc)
- `--out-dir <path>`: Output directory (default: "outputs")
- `--apply`: Apply suggested fixes

**Outputs**:
- `fitprep_report.json`: Detailed check report
- `map_fixed.mrc`: Fixed map (if `--apply` used)
- `model_shifted.pdb`: Shifted model (if `--apply` used)

**Example**:
```bash
crymodel fitprep check \
  --model structure.pdb \
  --map full_map.mrc \
  --half1 half1.mrc \
  --half2 half2.mrc \
  --apply \
  --out-dir outputs
```

---

## Loop Modeling

### loopcloud

**Purpose**: Generate clash-free loop completions in weak density.

**Usage**:
```bash
crymodel loopcloud generate \
  --model <model.pdb> \
  --anchors <spec> \
  --sequence <seq> \
  [OPTIONS]
```

**Required Arguments**:
- `--model`: Input model PDB/mmCIF
- `--anchors`: Anchor specification (e.g., "chainA:res123 -> chainA:res140")
- `--sequence`: Sequence for missing residues

**Options**:
- `--out-dir <path>`: Output directory (default: "outputs")
- `--map <path>`: Full map (.mrc) for density scoring
- `--half1 <path>`: Half-map 1 (.mrc)
- `--half2 <path>`: Half-map 2 (.mrc)
- `--num-candidates <int>`: Number of loop candidates to generate (default: 50)
- `--top-n <int>`: Number of top candidates to output (default: 10)
- `--ss-type <str>`: Secondary structure type: 'helix', 'sheet', or 'loop' (default: 'loop')

**Outputs**:
- `scores.csv`: Scores for all candidates
- `loopcloud_*.pdb`: Top-N loop candidates as PDB files

**Example**:
```bash
crymodel loopcloud generate \
  --model structure.pdb \
  --anchors "chainA:res123 -> chainA:res140" \
  --sequence "GAVLIS" \
  --map full_map.mrc \
  --num-candidates 50 \
  --top-n 10 \
  --ss-type loop \
  --out-dir outputs
```

---

## Machine Learning Tools

### extract-features

**Purpose**: Extract features from PDB files for ML training.

**Usage**:
```bash
crymodel extract-features \
  --pdb-dir <dir> \
  --output <features.csv> \
  [OPTIONS]
```

**Required Arguments**:
- `--pdb-dir`: Directory containing PDB files
- `--output`: Output CSV file

**Options**:
- `--training-set <path>`: Training set CSV with resolution info
- `--keep-hydrogens`: Keep hydrogen atoms (default: False)
- `--batch-size <int>`: Batch size for processing (default: 100)

**Example**:
```bash
crymodel extract-features \
  --pdb-dir TRAINING/PDBs \
  --output TRAINING/all_features.csv \
  --training-set TRAINING/training_set.csv
```

---

### train-ml

**Purpose**: Train ML model for water/ion classification.

**Usage**:
```bash
crymodel train-ml \
  --features <features.csv> \
  --output <model_dir> \
  [OPTIONS]
```

**Required Arguments**:
- `--features`: Input features CSV
- `--output`: Output directory for model

**Options**:
- `--epochs <int>`: Number of training epochs (default: 100)
- `--batch-size <int>`: Batch size (default: 64)
- `--learning-rate <float>`: Learning rate (default: 0.001)
- `--test-split <float>`: Test set fraction (default: 0.2)
- `--use-focal-loss`: Use focal loss instead of cross-entropy

**Example**:
```bash
crymodel train-ml \
  --features TRAINING/all_features.csv \
  --output TRAINING/models/single \
  --epochs 100 \
  --use-focal-loss
```

---

### train-ensemble

**Purpose**: Train ensemble of ML models with different random seeds.

**Usage**:
```bash
crymodel train-ensemble \
  --features <features.csv> \
  --output <ensemble_dir> \
  --n-models <int> \
  [OPTIONS]
```

**Required Arguments**:
- `--features`: Input features CSV
- `--output`: Output directory for ensemble
- `--n-models`: Number of models in ensemble

**Options**:
- `--epochs <int>`: Number of training epochs per model (default: 100)
- `--batch-size <int>`: Batch size (default: 64)
- `--learning-rate <float>`: Learning rate (default: 0.001)
- `--test-split <float>`: Test set fraction (default: 0.2)
- `--use-focal-loss`: Use focal loss instead of cross-entropy

**Example**:
```bash
crymodel train-ensemble \
  --features TRAINING/all_features.csv \
  --output TRAINING/models/ensemble \
  --n-models 5 \
  --epochs 100 \
  --use-focal-loss
```

---

## Utility Commands

### version

**Purpose**: Display version information.

**Usage**:
```bash
crymodel version
```

**Example**:
```bash
$ crymodel version
CryoModel version 0.1.0
```

---

## Common Options

Many commands share common options:

- `--out-dir <path>`: Output directory (default: "outputs")
- `--help` or `-h`: Show help message
- `--version`: Show version (when available)

---

## File Formats

### MRC/CCP4 Maps
- Standard cryo-EM density map format
- Supports half-maps for validation
- Local resolution maps for resolution-aware analysis

### PDB/mmCIF
- Standard structure format
- Supports multi-model structures
- Chain IDs and residue numbering preserved

### JSON
- Domain specifications
- Configuration files
- Summary reports

### CSV
- Feature tables
- Per-residue metrics
- Scores and predictions

---

## Tips and Best Practices

1. **Always run fitprep first** before major analysis to check map/model alignment
2. **Use half-maps** when available for better validation metrics
3. **Provide local resolution maps** for resolution-aware validation
4. **Start with default parameters** and adjust based on results
5. **Check output JSON/CSV files** for detailed statistics and warnings

---

## Troubleshooting

### Common Issues

1. **Empty output files**: Check that input files exist and are readable
2. **Low CC scores**: Verify map/model alignment with fitprep
3. **Memory errors**: Reduce batch sizes or process in chunks
4. **Import errors**: Ensure all dependencies are installed

### Getting Help

- Check command help: `crymodel <command> --help`
- Review output JSON files for detailed error messages
- Verify input file formats and paths

---

## Citation

If you use CryoModel in your research, please cite:

```
CryoModel: Unified cryo-EM modeling toolkit
Version 0.1.0
```

---

## License

[Add license information here]

---

**Last Updated**: 2024

