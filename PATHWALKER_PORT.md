# Pathwalker Port Summary

The pathwalking program has been successfully ported from CCTBX/Phenix dependencies to the crymodel codebase.

## What Was Ported

### Core Modules

1. **`crymodel/pathalker/pseudoatoms.py`**
   - Pseudoatom generation using clustering methods (KMeans, Spectral Clustering, Agglomerative Clustering, Mean Shift, GMM, Birch)
   - Noise addition for iterative refinement
   - Integrated with existing `MapVolume` class

2. **`crymodel/pathalker/tsp_solver.py`**
   - OR-Tools TSP solver (primary)
   - LKH TSP solver support (optional, requires external executable)
   - TSPLib file format support for LKH

3. **`crymodel/pathalker/distances.py`**
   - Euclidean distance matrix computation
   - Map-weighted distance matrix (penalizes paths through low-density regions)
   - TSP distance matrix preparation (adds depot, scales to integers)

4. **`crymodel/pathalker/path_evaluation.py`**
   - C-alpha to C-alpha distance evaluation
   - Path geometry statistics
   - Identifies problematic distances (too short/long)

5. **`crymodel/pathalker/averaging.py`**
   - Averaging multiple pathwalking runs
   - Probability computation for path positions
   - Path alignment and matching

6. **`crymodel/pathalker/pathwalker.py`**
   - Main pathwalking engine
   - PDB output functions (with optional probability B-factors)
   - Integrated workflow from map to path

7. **`crymodel/cli/pathwalk.py`**
   - CLI commands: `pathwalk` and `pathwalk-average`
   - Full parameter support
   - Integration with existing CLI infrastructure

## Key Changes from Original

1. **Removed CCTBX/Phenix Dependencies**
   - No longer uses `iotbx`, `cctbx`, `phenix` modules
   - Uses existing `MapVolume` and PDB I/O from codebase
   - Uses NumPy/NumPy-compatible arrays throughout

2. **Integration with Existing Codebase**
   - Uses `crymodel.io.mrc.MapVolume` for map handling
   - Uses `crymodel.io.site_export._pdb_atom_line` for PDB writing
   - Follows codebase conventions and structure

3. **Improved Error Handling**
   - Better handling of route indices and depot
   - Graceful handling of incomplete routes
   - Clear error messages

4. **Modular Design**
   - Each component is independent and testable
   - Easy to extend with future features

## Dependencies

### Required
- `numpy`
- `scipy`
- `scikit-learn`
- `gemmi` (for map I/O, already in codebase)

### Optional
- `ortools` (for TSP solving) - **needs to be added to `pyproject.toml`**
- LKH executable (external, for alternative TSP solver)

## Usage

### Basic Pathwalking

```bash
crymodel pathwalk \
  --map examples/emd_22898.map \
  --threshold 0.5 \
  --n-residues 200 \
  --pseudoatom-method kmeans \
  --output-pdb pathwalk.pdb \
  --out-dir outputs
```

### Map-Weighted Pathwalking

```bash
crymodel pathwalk \
  --map examples/emd_22898.map \
  --threshold 0.5 \
  --n-residues 200 \
  --map-weighted \
  --output-pdb pathwalk_weighted.pdb \
  --out-dir outputs
```

### Averaging Multiple Runs

```bash
crymodel pathwalk-average \
  --path-files path1.pdb,path2.pdb,path3.pdb \
  --output-pdb pathwalk_averaged.pdb \
  --probabilistic \
  --out-dir outputs
```

## Parameters

### Pathwalking Parameters

- `--map`: Input density map (.mrc/.map)
- `--threshold`: Density threshold for pseudoatom generation
- `--n-residues`: Number of residues (C-alpha atoms) - should match protein length
- `--pseudoatom-method`: Clustering method (kmeans, sc, ac, ms, gmm, birch)
- `--map-weighted`: Use map-weighted distances (penalizes low-density paths)
- `--tsp-solver`: TSP solver (ortools or lkh)
- `--time-limit`: Time limit for TSP solver (seconds)
- `--noise-level`: Noise level to add to pseudoatoms (Å) for iterative refinement
- `--random-state`: Random seed for reproducibility

### Averaging Parameters

- `--path-files`: Comma-separated list of path PDB files
- `--output-pdb`: Output averaged PDB file
- `--probabilistic`: Write probabilities to B-factor column
- `--out-dir`: Output directory

## Future Enhancements

The following features from the original code are planned for future implementation:

1. **Pseudoatom Smoothing**: Smooth pseudoatom positions to improve path quality
2. **Overpopulating Pseudoatoms**: Generate more pseudoatoms than residues for better coverage
3. **Path Filtering**: Filter paths based on geometry constraints
4. **C-alpha Ramachandran Angles**: Validate paths using Ramachandran plots
5. **Iterative Refinement**: Multiple iterations with noise to improve path quality
6. **Bracket Search**: Test multiple thresholds to find optimal path

## Notes

- The pathwalking algorithm assumes that the number of pseudoatoms matches the number of C-alpha atoms in the protein
- Target C-alpha to C-alpha distance is 3.8 Å (with acceptable range 2.8-4.8 Å)
- The TSP solver finds the optimal path that minimizes total distance while visiting all pseudoatoms
- Map-weighted distances help avoid paths through low-density regions
- Averaging multiple runs can improve path quality and identify highly confident regions

## Testing

To test the pathwalker:

1. Ensure `ortools` is installed: `pip install ortools`
2. Run with a test map and known number of residues
3. Check output PDB file for reasonable C-alpha distances
4. Evaluate path geometry using the built-in evaluation functions

## Integration Status

✅ Core pathwalking functionality ported
✅ CLI commands integrated
✅ PDB I/O integrated
✅ Map I/O integrated
✅ Path evaluation integrated
✅ Averaging functionality ported
⚠️  OR-Tools dependency needs to be added to `pyproject.toml`
⚠️  LKH solver support (optional, requires external executable)
⏳ Future enhancements (smoothing, filtering, Ramachandran angles) - planned

