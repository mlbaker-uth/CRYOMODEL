# BaseHunter Template Guide

## Recommended Approach

**Best option: Provide density maps (MRC files) at multiple resolutions**

### Why Density Maps?
- Directly usable without conversion
- Can match resolution to your data (2-4 Å range)
- Faster processing (no PDB→density conversion)
- More accurate representation of what you're comparing

### Why Multiple Resolutions?
- BaseHunter will auto-select the closest match to your data resolution
- More robust across different datasets
- Better handling of resolution-dependent features

### Why Multiple Templates?
- More robust classification (averages out noise/variation)
- Handles different base types (A/G for purine, T/C/U for pyrimidine)
- Better discrimination at 2-5 Å resolution

## Template Options

### Option 1: Directory Structure (Recommended)

Create a directory structure like this:

```
templates/
├── purine/
│   ├── purine_2.0A.mrc
│   ├── purine_2.5A.mrc
│   ├── purine_3.0A.mrc
│   ├── purine_3.5A.mrc
│   └── purine_4.0A.mrc
└── pyrimidine/
    ├── pyrimidine_2.0A.mrc
    ├── pyrimidine_2.5A.mrc
    ├── pyrimidine_3.0A.mrc
    ├── pyrimidine_3.5A.mrc
    └── pyrimidine_4.0A.mrc
```

**Advantages:**
- Clean organization
- Easy to add/remove templates
- BaseHunter auto-discovers templates

**Usage:**
```python
from crymodel.nucleotide.templates import TemplateLibrary

templates = TemplateLibrary(template_dir="templates/")
```

### Option 2: Multiple Templates Per Class

You can provide multiple templates per class (e.g., different base types):

```
templates/
├── purine/
│   ├── adenine_2.5A.mrc
│   ├── adenine_3.0A.mrc
│   ├── guanine_2.5A.mrc
│   └── guanine_3.0A.mrc
└── pyrimidine/
    ├── thymine_2.5A.mrc
    ├── thymine_3.0A.mrc
    ├── cytosine_2.5A.mrc
    └── cytosine_3.0A.mrc
```

BaseHunter will average all purine templates and all pyrimidine templates.

### Option 3: PDB Models (Alternative)

If you prefer PDB models, BaseHunter will auto-convert them:

```
templates/
├── purine/
│   ├── adenine.pdb
│   └── guanine.pdb
└── pyrimidine/
    ├── thymine.pdb
    └── cytosine.pdb
```

**Note:** PDB templates will be converted to density at your target resolution, which is slower but works fine.

### Option 4: Mixed Format

You can mix PDB and MRC files:

```
templates/
├── purine/
│   ├── adenine.pdb
│   └── guanine_3.0A.mrc
└── pyrimidine/
    ├── thymine.pdb
    └── cytosine_3.0A.mrc
```

## Template Requirements

### For Density Maps (MRC):
- **Format**: MRC/CCP4/MAP format
- **Resolution**: 2.0-4.0 Å (match your data range)
- **Size**: Should be similar to your base pair volumes (typically 10-20 Å cube)
- **Content**: Just the base itself (not the full nucleotide with sugar-phosphate)

### For PDB Models:
- **Format**: PDB or mmCIF
- **Content**: Just the base atoms (not full nucleotide)
- **Resolution**: Will be converted at your target resolution

## Recommendations

1. **Start with 1 template per class at 2.5-3.0 Å** (most common resolution)
   - This will work for most cases
   - Fastest to set up

2. **Add more resolutions if you have varied data** (2.0, 2.5, 3.0, 3.5, 4.0 Å)
   - Better matching across different datasets
   - Slightly slower but more robust

3. **Add multiple base types if available** (A, G for purine; T, C, U for pyrimidine)
   - More robust classification
   - Handles variation in base types

4. **Use density maps if possible** (faster than PDB conversion)

## Template Generation

If you have PDB models and want to create density maps:

```python
# Example: Convert PDB to density at 3.0 Å resolution
from crymodel.fitting.foldhunter import pdb_to_density_map
import numpy as np

# Define target map parameters
target_apix = 1.0  # Å per voxel
target_shape = (20, 20, 20)  # 20 Å cube
target_origin = np.array([0.0, 0.0, 0.0])  # Center at origin
resolution = 3.0  # Å

# Convert
density = pdb_to_density_map(
    pdb_path="adenine.pdb",
    target_apix=target_apix,
    target_shape=target_shape,
    target_origin=target_origin,
    resolution_A=resolution,
)

# Save
from crymodel.io.mrc import write_map, MapVolume
mv = MapVolume(
    data_zyx=density,
    apix=target_apix,
    origin_xyzA=target_origin,
    halfmaps=None,
)
write_map("adenine_3.0A.mrc", mv)
```

## Minimal Setup

**Absolute minimum:** 1 purine template + 1 pyrimidine template

```
templates/
├── purine/
│   └── template.mrc  (or .pdb)
└── pyrimidine/
    └── template.mrc  (or .pdb)
```

This will work, but may be less robust than multiple templates/resolutions.


