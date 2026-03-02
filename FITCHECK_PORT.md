# FitCheck Port Summary

The FitCheck (fitcheck.py) tool has been successfully ported into the crymodel framework as a resolution-aware cryoEM model validation tool.

## Overview

FitCheck combines and re-implements "lite" versions of common cryoEM validation tools into a single analysis package that synthesizes scores and indicates how accurate/overfit regions of the model are relative to the resolution of the map (both reported and local resolution).

## Location

- **Core modules**: `crymodel/validation/`
- **CLI command**: `crymodel validate`

## Implemented Features

### 1. Ringer-Lite (χ1-scan side-chain density score)
**Module**: `crymodel/validation/ringer_lite.py`

- Scans χ1 torsion angle in 10° steps (-180° to 180°)
- Rotates side-chain atoms virtually (no coordinate editing)
- Samples density at Cγ (or equivalent) position
- Computes:
  - Peak Z-score: (max(D) - μ)/σ
  - Rotamer alignment: distance to nearest canonical rotamer
  - Half-map stability: Δpeak between half1 and half2
  - Local resolution penalty for low-resolution regions

### 2. Q-Lite (Atom resolvability index)
**Module**: `crymodel/validation/q_lite.py`

- Samples radial density profile around each atom (0-2.0 Å)
- Constructs expected profile from local resolution
- Computes normalized correlation (Q-score)
- Includes half-map versions (Q_half1, Q_half2, Q_delta)
- Amplitude ratio (best fit scale)

### 3. Cα-Tube (Backbone trace continuity)
**Module**: `crymodel/validation/ca_tube.py`

- Builds spline through Cα positions
- Samples density in cylindrical tube (radius ~1.25 Å) along backbone
- Computes continuity score: fraction of points above threshold
- Half-map stability metric

### 4. Local CC Variants
**Module**: `crymodel/validation/local_cc.py`

- **CC_mask**: 2 Å mask around atoms
- **CC_box**: Fixed 4 Å cube
- **ZNCC**: Zero-mean normalized cross-correlation
- Full-map and half-map versions
- Delta CC: full vs best half-map

### 5. Geometry-vs-Resolution Priors
**Module**: `crymodel/validation/geometry_priors.py`

- Ramachandran probability (simplified)
- Clashscore Z-score
- Placeholders for:
  - Rotamer probability
  - CaBLAM flags
  - Peptide planarity
  - Cβ deviations

### 6. Resolution-Aware Priors
**Module**: `crymodel/validation/resolution_priors.py`

- Bins residues by local resolution (0.2 Å bins)
- Fits robust statistics (median + MAD) per bin
- Converts raw metrics to Z-residuals
- Prevents "why is Q lower at 3.9 Å" problem
- Flags too-good geometry as potential overfit

### 7. Feature Extractor
**Module**: `crymodel/validation/feature_extractor.py`

- Orchestrates all feature extraction
- Processes structure residue-by-residue
- Aggregates atom-level features to residues
- Outputs DataFrame with all features

## Usage

### Basic Validation

```bash
crymodel validate \
  --model structure.pdb \
  --map full_map.mrc \
  --half1 half1.mrc \
  --half2 half2.mrc \
  --localres local_resolution.mrc \
  --out-dir outputs
```

### Fit Priors from Data

```bash
crymodel validate \
  --model structure.pdb \
  --map full_map.mrc \
  --half1 half1.mrc \
  --half2 half2.mrc \
  --fit-priors \
  --out-dir outputs
```

### Use Pre-computed Priors

```bash
crymodel validate \
  --model structure.pdb \
  --map full_map.mrc \
  --half1 half1.mrc \
  --half2 half2.mrc \
  --priors priors.yaml \
  --out-dir outputs
```

## Outputs

- **`features.csv`**: Per-residue features with all computed metrics
- **`priors.yaml`**: Resolution-aware priors (if `--fit-priors` used)

## Features Extracted Per Residue

### Map-Model Evidence
- `ringer_Z`: Ringer-Lite peak Z-score
- `ringer_peak_deg`: Peak angle (degrees)
- `ringer_to_rotamer_deg`: Distance to canonical rotamer
- `ringer_half_drop`: Half-map stability
- `Q_mean`, `Q_min`: Q-Lite scores (averaged over atoms)
- `CC_mask`, `CC_box`, `ZNCC`: Local CC variants
- `CC_half1`, `CC_half2`, `CC_delta`: Half-map CCs
- `continuity_score`, `continuity_mean`, `continuity_std`: Backbone continuity

### Geometry Evidence
- `ramachandran_prob`: Ramachandran probability
- `clashscore_z`: Clashscore Z-score
- `rotamer_prob`: Rotamer probability (placeholder)
- `cablam_flag`: CaBLAM flag (placeholder)
- `peptide_planarity_z`: Peptide planarity Z-score (placeholder)
- `cb_deviation_z`: Cβ deviation Z-score (placeholder)

### Context
- `chain`: Chain ID
- `resi`: Residue number
- `resname`: Residue name
- `local_res`: Local resolution (Å)

### Z-Residuals (if priors available)
- `ringer_Z_z_residual`: Z-residual for Ringer Z-score
- `Q_mean_z_residual`: Z-residual for Q-score
- `CC_mask_z_residual`: Z-residual for CC_mask
- etc.

## Future Enhancements

### Planned Features
1. **3D CNN**: Local map patch analysis (16-24 Å cubic patches)
2. **Residue Graph Neural Network**: Graph-based residue analysis
3. **ML Classifier**: XGBoost baseline → Deep learning models
4. **Overfitting Detector**: 
   - ΔCC_full−half analysis
   - Geometry improbability detection
   - Side-chain exuberance detection
5. **Per-residue p(correct)**: pLDDT-style confidence scores
6. **Uncertainty Map**: Visualization of uncertain regions
7. **Publication-ready Outputs**: 
   - Residue table with sortable columns
   - Calibration plots
   - Violin plots of observed-vs-expected metrics

### Integration Points
- Full MolProbity integration (currently simplified)
- CaBLAM integration
- ResMap/DeepRes integration for local resolution
- Directional FSC for anisotropy analysis

## Dependencies

All dependencies already in codebase:
- `numpy`, `scipy` (already in dependencies)
- `pandas`, `pyyaml` (added to dependencies)
- `gemmi` (already in dependencies)
- `matplotlib` (already in dependencies)

## Status

✅ **Core metric calculators implemented**
✅ **Feature extraction pipeline complete**
✅ **Resolution-aware priors framework ready**
✅ **CLI command integrated**
⏳ **ML training/inference** (ready for training data)
⏳ **Overfitting detector** (framework ready)
⏳ **3D CNN/GNN** (future enhancement)

## Notes

- All "lite" metrics are simplified but functional versions of full tools
- Geometry features are simplified (full MolProbity integration would require additional dependencies)
- Ready for training data collection and ML model development
- Framework supports future deep learning enhancements

