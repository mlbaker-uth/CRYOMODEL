# crymodel/cli/workflow.py
"""CLI for workflow execution."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, List
import typer

from ..workflow.workflow import execute_workflow

app = typer.Typer(no_args_is_help=True)


@app.command()
def run(
    workflow_file: Path = typer.Argument(..., help="Path to workflow file (.yaml or .json)"),
    base_dir: Optional[Path] = typer.Option(None, "--base-dir", help="Base directory for paths (default: workflow file directory)"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print commands without executing"),
    steps: Optional[str] = typer.Option(None, "--steps", help="Comma-separated list of step names to execute (default: all)"),
):
    """Execute a CryoModel workflow.
    
    Workflow files define a series of tool invocations with dependencies,
    file references, and parameter passing. Supports YAML and JSON formats.
    
    Example workflow structure:
    
    \b
    name: alphafold_fitting
    description: Filter AlphaFold model and fit to density map
    variables:
      resolution: 3.0
      plddt_threshold: 0.5
    files:
      alphafold_pdb: models/af_model.pdb
      target_map: maps/target.mrc
    steps:
      - name: filter_model
        tool: affilter
        command: filter
        inputs:
          input_pdb: ${alphafold_pdb}
          plddt_threshold: ${plddt_threshold}
        outputs:
          output_pdb: filtered_model.pdb
      - name: fit_to_map
        tool: foldhunter
        command: search
        depends_on: [filter_model]
        inputs:
          target_map: ${target_map}
          probe_pdb: ${filtered_model.pdb}
          resolution: ${resolution}
    """
    step_names = None
    if steps:
        step_names = [s.strip() for s in steps.split(',')]
    
    try:
        execute_workflow(
            workflow_path=workflow_file,
            base_dir=base_dir,
            dry_run=dry_run,
            step_names=step_names,
        )
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def validate(
    workflow_file: Path = typer.Argument(..., help="Path to workflow file (.yaml or .json)"),
):
    """Validate a workflow file without executing it.
    
    Checks syntax, dependencies, and file references.
    """
    from ..workflow.workflow import load_workflow
    
    try:
        workflow = load_workflow(workflow_file)
        typer.echo(f"✓ Workflow loaded successfully")
        typer.echo(f"  Name: {workflow.name}")
        typer.echo(f"  Description: {workflow.description}")
        typer.echo(f"  Version: {workflow.version}")
        typer.echo(f"  Steps: {len(workflow.steps)}")
        
        # Check dependencies
        step_names = {step.name for step in workflow.steps}
        for step in workflow.steps:
            for dep in step.depends_on:
                if dep not in step_names:
                    typer.echo(f"  ⚠ Warning: Step '{step.name}' depends on '{dep}' which doesn't exist", err=True)
        
        typer.echo(f"\n✓ Workflow is valid")
    except Exception as e:
        typer.echo(f"✗ Workflow validation failed: {e}", err=True)
        raise typer.Exit(1)


if __name__ == "__main__":
    app()

