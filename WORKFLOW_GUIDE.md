# CryoModel Workflow System

The workflow system allows you to chain multiple CryoModel tools together into automated pipelines. Workflows are defined in YAML or JSON files and can reference files, variables, and outputs from previous steps.

## Basic Usage

```bash
# Run a workflow
crymodel workflow run workflow.yaml

# Validate a workflow without running it
crymodel workflow-validate workflow.yaml

# Dry run (print commands without executing)
crymodel workflow run workflow.yaml --dry-run

# Run specific steps only
crymodel workflow run workflow.yaml --steps step1,step2
```

## Workflow File Format

Workflow files use YAML or JSON format with the following structure:

```yaml
name: workflow_name
description: Description of what this workflow does
version: 1.0

# Global variables (can be referenced with ${variable_name})
variables:
  resolution: 3.0
  plddt_threshold: 0.5

# File references (paths relative to workflow file or absolute)
files:
  input_pdb: models/alphafold_model.pdb
  target_map: maps/target.mrc

# Global parameters (nested dictionaries)
parameters:
  foldhunter:
    n_coarse_rotations: 1000
    n_fine_rotations: 500

# Workflow steps
steps:
  - name: step1_name
    tool: tool_name          # e.g., "affilter", "foldhunter"
    command: command_name    # e.g., "filter", "search"
    inputs:
      input_pdb: ${input_pdb}  # Reference file from files section
      threshold: ${plddt_threshold}  # Reference variable
    outputs:
      output_pdb: outputs/filtered.pdb
    depends_on: []  # List of step names that must complete first
    
  - name: step2_name
    tool: foldhunter
    command: search
    depends_on: [step1_name]  # Wait for step1 to complete
    inputs:
      target_map: ${target_map}
      probe_pdb: ${step1_name.output_pdb}  # Reference output from step1
    outputs:
      best_fit: outputs/fitted.pdb
```

## Variable Substitution

Variables can be referenced using `${variable_name}` syntax:

- **Global variables**: `${resolution}`, `${plddt_threshold}`
- **File references**: `${input_pdb}`, `${target_map}`
- **Step outputs**: `${step_name.output_key}` (e.g., `${filter_alphafold.filtered_pdb}`)
- **Nested parameters**: `${parameters.foldhunter.n_coarse_rotations}`

## Step Dependencies

Steps can depend on other steps using the `depends_on` field:

```yaml
steps:
  - name: filter
    tool: affilter
    command: filter
    depends_on: []  # No dependencies
    
  - name: fit
    tool: foldhunter
    command: search
    depends_on: [filter]  # Wait for filter step
```

The workflow engine automatically resolves dependencies and executes steps in the correct order.

## Conditional Steps

Steps can include conditions that must be met before execution:

```yaml
- name: rebuild_loops
  tool: loopcloud
  command: generate
  condition: "len(step_outputs.get('filter_alphafold', {})) > 0"
  depends_on: [filter_alphafold]
```

Conditions are Python expressions evaluated in a safe context with access to:
- `step_outputs`: Dictionary of outputs from previous steps
- `workflow`: The workflow object
- Basic functions: `len`, `bool`, `str`, `int`, `float`

## Example Workflows

### AlphaFold Fitting Workflow

```yaml
name: alphafold_fitting
description: Filter AlphaFold model and fit to density map

variables:
  resolution: 3.0
  plddt_threshold: 0.5

files:
  alphafold_pdb: models/af_model.pdb
  target_map: maps/target.mrc

steps:
  - name: filter_alphafold
    tool: affilter
    command: filter
    inputs:
      input_pdb: ${alphafold_pdb}
      plddt_threshold: ${plddt_threshold}
      filter_loops: true
      filter_connectivity: true
      out_dir: outputs/affilter
    outputs:
      filtered_pdb: outputs/affilter/alphafold_filtered.pdb
      low_plddt_csv: outputs/affilter/affilter_low_plddt_regions.csv

  - name: fit_to_map
    tool: foldhunter
    command: search
    depends_on: [filter_alphafold]
    inputs:
      target_map: ${target_map}
      probe_pdb: ${filter_alphafold.filtered_pdb}
      resolution: ${resolution}
      plddt_threshold: ${plddt_threshold}
      out_dir: outputs/foldhunter
    outputs:
      best_fit_pdb: outputs/foldhunter/foldhunter_top_fit.pdb
```

### Complete Pipeline (Filter → Fit → Rebuild Loops)

See `examples/workflow_example.yaml` for a complete example that:
1. Filters an AlphaFold model
2. Fits the filtered model to a density map
3. Rebuilds low pLDDT regions using loop modeling

## Integration Points

The workflow system enables integration between tools:

- **affilter → foldhunter**: Use filtered PDB as probe
- **affilter → loopcloud**: Use low pLDDT regions CSV to identify loops to rebuild
- **foldhunter → loopcloud**: Use fitted model as base for loop modeling
- **findligands → predictligands**: Use ligand outputs for classification

## Tips

1. **Use relative paths**: File paths are resolved relative to the workflow file directory
2. **Organize outputs**: Use consistent output directories (e.g., `outputs/tool_name/`)
3. **Test with dry-run**: Use `--dry-run` to verify commands before execution
4. **Validate first**: Use `workflow-validate` to check syntax and dependencies
5. **Reference step outputs**: Use `${step_name.output_key}` to chain steps together

## Advanced Features

### Nested Parameter Access

Access nested parameters using dot notation:

```yaml
parameters:
  foldhunter:
    search:
      n_coarse: 1000
      n_fine: 500

steps:
  - name: fit
    inputs:
      n_coarse_rotations: ${parameters.foldhunter.search.n_coarse}
```

### Multiple Outputs

Steps can define multiple outputs:

```yaml
outputs:
  filtered_pdb: outputs/filtered.pdb
  stats_txt: outputs/stats.txt
  domains_csv: outputs/domains.csv
```

All outputs are available to subsequent steps via `${step_name.output_key}`.

