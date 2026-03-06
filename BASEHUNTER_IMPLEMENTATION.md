# BaseHunter Enhanced Implementation

## Overview

BaseHunter has been fully enhanced with template-based classification, constraint enforcement, and statistical analysis. The system now:

1. **Template-based classification** - Uses purine/pyrimidine templates for accurate classification
2. **Three-class output** - Classifies as purine, pyrimidine, or unclassified
3. **Base pair constraints** - Enforces 1 purine + 1 pyrimidine per base pair
4. **Inter/intra-class statistics** - Computes NCC statistics within and between classes
5. **Bootstrap likelihoods** - Estimates classification probabilities via bootstrap resampling
6. **Feature extraction** - Uses geometric and density features for better discrimination

## Template System

### Template Location
Default packaged templates are located at: `data/DNA-TEMPLATES/`

### Template Files
- `template-a-35.mrc` - Adenine (purine) at 3.5 Å
- `template-g-35.mrc` - Guanine (purine) at 3.5 Å
- `template-c-35.mrc` - Cytosine (pyrimidine) at 3.5 Å
- `template-t-35.mrc` - Thymine (pyrimidine) at 3.5 Å

### Naming Convention
The system automatically detects templates using the naming pattern:
- `template-{base}-{resolution}.mrc`
- Base: `a`, `g` (purine) or `c`, `t` (pyrimidine)
- Resolution: x10 format (e.g., `35` = 3.5 Å)

## Usage

### Basic Command

```bash
crymodel basehunter compare \
    --input-file input_pairs.txt \
    --threshold 0.3 \
    --out-dir basehunter_outputs
```

### Input File Format

The input file should contain:
- Line 1: Directory path containing volume files
- Subsequent lines: Pairs of volume filenames (space-separated)
- Optional third value: per-pair threshold override

Example `input_pairs.txt`:
```
/path/to/volumes
pair1_vol1.mrc pair1_vol2.mrc
pair2_vol1.mrc pair2_vol2.mrc
pair3_vol1.mrc pair3_vol2.mrc
```

Example with per-pair thresholds:
```
/path/to/volumes
pair1_vol1.mrc pair1_vol2.mrc 0.45
pair2_vol1.mrc pair2_vol2.mrc 0.55
pair3_vol1.mrc pair3_vol2.mrc 0.40
```

### Options

- `--input-file`: Input file with volume pairs
- `--template-dir`: Directory containing templates (required)
- `--threshold`: Density threshold for volumes (required)
- `--template-threshold`: Threshold for template maps (zero below)
- `--backbone-mask-mrc`: Backbone mask map to exclude sugar/phosphate
- `--backbone-mask-threshold`: Threshold for backbone mask map
- `--resolution`: Target resolution in Å (auto-detect if not provided)
- `--alignment-threshold`: Minimum correlation for classification (default: 0.3)
- `--bootstrap/--no-bootstrap`: Enable/disable bootstrap analysis (default: enabled)
- `--n-bootstrap`: Number of bootstrap iterations (default: 100)
- `--out-dir`: Output directory (default: `basehunter_outputs`)

## Output Files

### `basehunter_classifications.csv`
CSV file with detailed classification results:
- `volume1`, `volume2`: Volume filenames
- `volume1_class`, `volume2_class`: Classification (purine/pyrimidine)
- `volume1_purine_score`, `volume1_pyrimidine_score`: Template alignment scores
- `volume1_confidence`: Classification confidence
- `volume1_purine_likelihood`, `volume1_pyrimidine_likelihood`: Bootstrap probabilities
- `base_pair_valid`: Whether pair satisfies constraint (1 purine + 1 pyrimidine)

### `basehunter_statistics.txt`
Text file with summary statistics:
- Total volumes and base pairs
- Assignment counts (purine, pyrimidine, unclassified)
- Inter/intra-class NCC statistics
- Separation score (higher = better class separation)

## Implementation Details

### Classification Pipeline

1. **Template Loading**
   - Loads all templates from directory
   - Parses resolution from filename (x10 format)
   - Averages multiple templates per class (A+G for purine, C+T for pyrimidine)

2. **Volume Alignment**
   - Aligns each volume to both templates using FFT-based cross-correlation
   - Finds optimal translation for best match

3. **Feature Extraction**
   - Extracts geometric features (size, compactness, component count)
   - Extracts density features (mean, max, skewness, kurtosis)
   - Computes feature similarity to templates

4. **Classification**
   - Combined score: 60% correlation + 40% feature similarity
   - Three-class output: purine, pyrimidine, or unclassified
   - Confidence based on score difference

5. **Constraint Enforcement**
   - Enforces 1 purine + 1 pyrimidine per base pair
   - Resolves conflicts by confidence
   - Assigns unclassified based on partner or scores

6. **Statistical Analysis**
   - Computes intra-class NCC (within purine, within pyrimidine)
   - Computes inter-class NCC (between purine and pyrimidine)
   - Calculates separation score

7. **Bootstrap Analysis**
   - Adds noise to volumes and reclassifies
   - Estimates likelihoods for each class
   - Provides confidence intervals

## Key Features

### Template-Based Classification
- Uses actual purine/pyrimidine density templates
- Handles alignment automatically
- Supports multiple templates per class (averaged)

### Constraint Enforcement
- Automatically enforces base pair constraint
- Resolves unclassified assignments
- Handles conflicts intelligently

### Statistical Validation
- Inter/intra-class comparison
- Separation score for quality assessment
- Bootstrap likelihoods for confidence

### Feature-Based Discrimination
- Geometric features (size, compactness)
- Density features (distribution statistics)
- Combined with template correlation

## Example Output

```
Loading 20 volumes...
Templates loaded: purine shape=(20, 20, 20), pyrimidine shape=(20, 20, 20)

Classifying volumes...
  pair1_vol1.mrc: purine (purine=0.723, pyrimidine=0.412, conf=0.430)
  pair1_vol2.mrc: pyrimidine (purine=0.398, pyrimidine=0.689, conf=0.422)
  ...

Enforcing base pair constraints...
  Resolved 2 unclassified/conflicting assignments

Computing class statistics...
  Purine intra-class NCC: 0.7234 ± 0.0823 (n=45)
  Pyrimidine intra-class NCC: 0.6891 ± 0.0956 (n=45)
  Inter-class NCC: 0.4123 ± 0.1234 (n=100)
  Separation score: 0.5879

Performing bootstrap analysis (100 iterations)...
  Bootstrap analysis complete

✓ Classification complete!
  Purine assignments: 10
  Pyrimidine assignments: 10
  Unclassified: 0
  Valid base pairs: 10/10
```

## Notes

- Templates are automatically averaged if multiple templates per class are provided
- Resolution is auto-detected from first volume if not specified
- Bootstrap analysis can be disabled for faster processing
- All volumes must have the same shape (or will be padded)


