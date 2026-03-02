# CryoModel Quick Reference

## Core Workflow

```bash
# 1. Preflight check
crymodel fitprep check --model model.pdb --map map.mrc --apply

# 2. Find ligands and waters
crymodel findligands --map map.mrc --model model.pdb --thresh 0.5 \
  --half1 half1.mrc --half2 half2.mrc --ml-model models/ensemble \
  --entry-resolution 2.2

# 3. Predict ligand identities
crymodel predictligands --ligands-pdb outputs/ligands.pdb \
  --ligand-map outputs/ligands_map.mrc --model model.pdb

# 4. Validate model
crymodel validate --model model.pdb --map map.mrc \
  --half1 half1.mrc --half2 half2.mrc --localres localres.mrc
```

## All Commands

| Command | Purpose | Key Options |
|---------|---------|-------------|
| `findligands` | Find unmodeled density | `--map`, `--model`, `--thresh`, `--ml-model` |
| `predictligands` | Classify ligands | `--ligands-pdb`, `--ligand-map`, `--model` |
| `validate` | Model validation | `--model`, `--map`, `--half1`, `--half2`, `--localres` |
| `pathwalk` | Backbone tracing | `--map`, `--model`, `--thresh`, `--method` |
| `pathwalk-average` | Average paths | `--paths`, `--out` |
| `pyhole` | Pore analysis | `--pdb`, `--top`, `--bottom`, `--centerline` |
| `pyhole-plot` | Plot pore profiles | Input files, `--overlay`, `--grid` |
| `basehunter` | Nucleotide comparison | `--input-file`, `--threshold` |
| `pdbcom` | Domain COMs | `--model`, `--domains`, `--mass-weighted` |
| `fitcompare` | Compare models | `--model-a`, `--model-b`, `--anchors` |
| `fitprep` | Preflight check | `--model`, `--map`, `--apply` |
| `loopcloud` | Loop modeling | `--model`, `--anchors`, `--sequence`, `--map` |
| `extract-features` | ML feature extraction | `--pdb-dir`, `--output` |
| `train-ml` | Train single model | `--features`, `--output`, `--epochs` |
| `train-ensemble` | Train ensemble | `--features`, `--output`, `--n-models` |
| `version` | Show version | (no options) |

## Common File Types

- **Maps**: `.mrc`, `.map` (MRC/CCP4 format)
- **Models**: `.pdb`, `.cif` (PDB/mmCIF format)
- **Data**: `.csv`, `.json`, `.yaml`

## Quick Examples

### Find ligands with ML
```bash
crymodel findligands --map map.mrc --model model.pdb --thresh 0.5 \
  --ml-model models/ensemble --entry-resolution 2.2
```

### Validate model
```bash
crymodel validate --model model.pdb --map map.mrc \
  --half1 half1.mrc --half2 half2.mrc --fit-priors
```

### Compare two states
```bash
crymodel fitcompare compare --model-a stateA.pdb --model-b stateB.pdb \
  --anchors "A:100-160"
```

### Analyze pore
```bash
crymodel pyhole analyze --pdb channel.pdb \
  --top "A:123" --bottom "A:456" --adaptive
```

### Generate loops
```bash
crymodel loopcloud generate --model model.pdb \
  --anchors "A:123 -> A:140" --sequence "GAVLIS" --map map.mrc
```

## Help

Get help for any command:
```bash
crymodel <command> --help
```

