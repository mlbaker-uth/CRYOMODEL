# Tools Port Summary

Three additional tools have been successfully ported into the cryomodel framework:

## 1. pyHole - Pore/Tunnel Analysis

### Location
- **Core module**: `cryomodel/pore/pyhole.py`
- **CLI command**: `cryomodel pyhole analyze`

### Description
Calculates and describes transmembrane pores/tunnels using HOLE-like methodology. Supports both straight and curved centerlines.

### Features
- Straight or curved centerline calculation
- Adaptive sampling for better resolution
- Hydrophobicity and electrostatics analysis
- Passability analysis for different species (water, ions)
- Volume, resistance, and conductance calculations
- Outputs: CSV profile, PDB centerline, PDB mesh, summary JSON

### Usage
```bash
cryomodel pyhole analyze \
  --pdb structure.pdb \
  --top "A:123" \
  --bottom "A:456" \
  --step 1.0 \
  --out-prefix pore_analysis
```

### Key Parameters
- `--top`, `--bottom`: Residue selections for pore endpoints
- `--centerline`: 'straight' or 'curved'
- `--step`: Sampling step size (Å)
- `--adaptive`: Enable adaptive sampling
- `--occupancy`: 'hydro' or 'electro' for occupancy metric
- `--probe`: Probe radius for accessible volume (Å)
- `--conductivity`: Conductivity for resistance calculation (S/m)

## 2. pyHole Plotter - Pore Profile Visualization

### Location
- **Core module**: `cryomodel/pore/plotter.py`
- **CLI command**: `cryomodel pyhole-plot plot`

### Description
Creates publication-quality plots of pyHole pore profiles with support for overlays, grids, and secondary axes.

### Features
- Single plot, overlay, or grid layouts
- Blocked span shading from passability analysis
- Secondary axis for hydrophobicity/electrostatics/occupancy
- Axis swapping (vertical profiles)
- Publication-quality styling
- PNG and PDF output

### Usage
```bash
# Single plot
cryomodel pyhole-plot plot outputs/pore_analysis --out fig1C --ylim 0.5,8.0

# Overlay multiple profiles
cryomodel pyhole-plot plot stateA,stateB --overlay --labels "A,B" --out overlay

# Grid layout
cryomodel pyhole-plot plot P1,P2,P3,P4,P5 --grid 1x5 --out fig2
```

### Key Parameters
- `--overlay`: Overlay all inputs in one plot
- `--grid`: Grid layout (e.g., '1x5', '2x3')
- `--ylim`: Radius range (lo,hi)
- `--species`: Species for passability shading
- `--secondary`: Secondary axis ('hydro', 'electro', 'occ')
- `--swap-axes`: Swap axes for vertical profile
- `--style-paper`: Apply publication-quality styling

## 3. BaseHunter - Nucleotide Density Comparison

### Location
- **Core module**: `cryomodel/nucleotide/basehunter.py`
- **CLI command**: `cryomodel basehunter compare`

### Description
Compares and sorts nucleotide density at near-atomic resolutions using Earth Mover's Distance (EMD) and Normalized Cross-Correlation (NCC).

### Features
- Point cloud generation from thresholded volumes
- EMD-based similarity comparison
- Monte Carlo refinement for optimal grouping
- NCC calculation for group consistency
- Average volume computation
- Group assignment output with statistics

### Usage
```bash
cryomodel basehunter compare \
  --input-file volume_pairs.txt \
  --threshold 0.5 \
  --out-dir outputs
```

### Input File Format
The input file should contain:
```
/path/to/volume/directory
volume1.mrc volume2.mrc
volume3.mrc volume4.mrc
...
```

### Key Parameters
- `--input-file`: File with volume directory and pairs
- `--threshold`: Density threshold for point cloud generation
- `--max-iterations`: Maximum Monte Carlo iterations
- `--min-stability`: Minimum stability for convergence
- `--min-improvement`: Minimum improvement for convergence
- `--exploration-chance`: Exploration probability

### Outputs
- `group1.txt`, `group2.txt`: Group assignments
- `group1_with_ncc.txt`, `group2_with_ncc.txt`: Group assignments with NCC statistics
- `avg_group1.mrc`, `avg_group2.mrc`: Average volumes for each group

## 4. Sequence conservation (MSA → PDB)

### Location
- **Core module**: `cryomodel/conservation.py` (`build_conservation_rows`, `compute_conservation`)
- **CLI command**: `cryomodel seqconservation`

### Description
Maps a multi-sequence FASTA alignment onto one or more PDB/mmCIF chains. The **first** FASTA record is the reference (may be longer than the coordinate chain; extra residues are skipped until the model sequence matches in order). Homomultimers: comma-separated `--chain` / `--chains` with **identical** polymer sequences.

### Outputs
- **CSV / JSON**: per-residue metrics (`n_aa_types`, `p_nonref`, `entropy`, `mean_penalty`, `frac_nonconservative`, …).
- **Optional PDB**: B-factor and optional occupancy columns from chosen metrics (e.g. ChimeraX coloring).

### Usage
```bash
cryomodel seqconservation structure.pdb alignment.fasta \
  --chain A,B,C,D \
  --out-csv conservation.csv \
  --out-json conservation.json \
  --out-pdb conservation_bfac.pdb \
  --bfactor-metric n_aa_types \
  --occupancy-metric p_nonref
```

### Key parameters
- `--chain` / `--chains`: comma-separated chain IDs (homomultimer).
- `--bfactor-metric`, `--occupancy-metric`: which column to write to PDB fields.
- `--include-reference-in-stats`: include first MSA row in frequency counts (default excludes it).

---

## 5. Sequence conservation — 3D diffusion (Tier 3)

### Location
- **Core module**: `cryomodel/conservation_diffusion.py` (`run_conservation_diffusion`)
- **CLI command**: `cryomodel seqconservation-diffuse`

### Description
Builds the same per-residue conservation table, then places **all** selected chains on one **Cα proximity graph** (edges within and between subunits within `--contact-radius`). A **seed** field (raw column or **composite** such as `p_nonref × mean_penalty`) is diffused with soft distance falloff; optional **basin** labels group residues toward local maxima of the diffused field in 3D.

### Outputs
- **CSV**: conservation columns plus `seed_raw`, `seed_signal`, `diffused_score`, `basin_id`, `is_diffusion_peak`.
- **JSON**: `meta` (parameters, peak indices) + `rows`.
- **Optional PDB**: B-factor from `diffused_score` or `seed_signal`.

### Usage
```bash
cryomodel seqconservation-diffuse structure.pdb alignment.fasta \
  --chains A,B,C,D \
  --out-csv diffusion.csv \
  --seed-metric composite_nonref_penalty \
  --seed-threshold 0.02 \
  --contact-radius 10 --falloff-angstrom 3 \
  --diffusion-steps 24 --mix 0.4 \
  --peak-min 0.02 --basin-mode nearest_peak \
  --out-pdb diffusion_bfac.pdb --bfactor-writes diffused_score
```

### Composite seed metrics
- `composite_nonref_penalty` — `p_nonref * mean_penalty`
- `composite_entropy_noncons` — `entropy * frac_nonconservative`
- `composite_diversity_penalty` — `((n_aa_types - 1) / 19) * mean_penalty`

### Key parameters
- `--seed-metric`: primitive or composite (see `--help`).
- `--seed-threshold`: subtracted from seed before diffusion; composites often need smaller thresholds than `p_nonref` alone.
- `--contact-radius`, `--falloff-angstrom`: graph connectivity and `exp(-d/d0)` weights.
- `--diffusion-steps`, `--mix`: relaxation toward neighbor means.
- `--basin-mode none | nearest_peak`, `--peak-min`, `--peak-weight-gamma`.

---

## Integration Status

✅ **All tools ported and integrated**
✅ **CLI commands registered**
✅ **No conflicts with existing code**
✅ **Isolated modules (no modifications to existing code)**

## Dependencies

All tools use existing dependencies:
- `numpy`, `scipy` (already in dependencies)
- `matplotlib`, `pandas` (already in dependencies)
- `mrcfile` (already in dependencies)
- `gemmi` (already in dependencies)

No new dependencies required.

## Future Enhancements

### BaseHunter
- Machine learning integration for classification
- Probability analysis
- Segmentation/alignment (currently handled offline)
- Improved convergence criteria

### pyHole
- Additional centerline algorithms
- Improved passability analysis
- Integration with density maps

### pyHole Plotter
- Additional plot types
- Interactive visualization
- Export to other formats

## Notes

- All tools maintain backward compatibility with original functionality
- Input/output formats remain compatible with original tools
- CLI interfaces provide user-friendly access to all features
- Code follows cryomodel conventions and structure

