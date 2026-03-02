# Tools Port Summary

Three additional tools have been successfully ported into the crymodel framework:

## 1. pyHole - Pore/Tunnel Analysis

### Location
- **Core module**: `crymodel/pore/pyhole.py`
- **CLI command**: `crymodel pyhole analyze`

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
crymodel pyhole analyze \
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
- **Core module**: `crymodel/pore/plotter.py`
- **CLI command**: `crymodel pyhole-plot plot`

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
crymodel pyhole-plot plot outputs/pore_analysis --out fig1C --ylim 0.5,8.0

# Overlay multiple profiles
crymodel pyhole-plot plot stateA,stateB --overlay --labels "A,B" --out overlay

# Grid layout
crymodel pyhole-plot plot P1,P2,P3,P4,P5 --grid 1x5 --out fig2
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
- **Core module**: `crymodel/nucleotide/basehunter.py`
- **CLI command**: `crymodel basehunter compare`

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
crymodel basehunter compare \
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
- Code follows crymodel conventions and structure

