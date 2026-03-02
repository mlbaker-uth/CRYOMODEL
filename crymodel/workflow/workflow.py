# crymodel/workflow/workflow.py
"""Workflow execution engine for CryoModel."""
from __future__ import annotations

import yaml
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import subprocess
import sys
from dataclasses import dataclass, field


@dataclass
class WorkflowStep:
    """Represents a single step in a workflow."""
    name: str
    tool: str
    command: str
    inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, str] = field(default_factory=dict)
    depends_on: List[str] = field(default_factory=list)
    condition: Optional[str] = None  # Optional condition to run step


@dataclass
class Workflow:
    """Represents a complete workflow."""
    name: str
    description: str
    version: str = "1.0"
    variables: Dict[str, Any] = field(default_factory=dict)
    files: Dict[str, str] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)
    steps: List[WorkflowStep] = field(default_factory=list)


def resolve_path(path: str, base_dir: Path, variables: Dict[str, Any]) -> Path:
    """Resolve a path with variable substitution.
    
    Args:
        path: Path string (may contain ${var} substitutions)
        base_dir: Base directory for relative paths
        variables: Variable dictionary for substitution
        
    Returns:
        Resolved Path object
    """
    # Substitute variables
    for key, value in variables.items():
        path = path.replace(f"${{{key}}}", str(value))
    
    # Resolve relative to base_dir
    if not Path(path).is_absolute():
        path = str(base_dir / path)
    
    return Path(path)


def resolve_value(value: Any, base_dir: Path, variables: Dict[str, Any], step_outputs: Optional[Dict] = None) -> Any:
    """Resolve a value (recursively handle dicts/lists).
    
    Args:
        value: Value to resolve (may contain ${var} substitutions)
        base_dir: Base directory for relative paths
        variables: Variable dictionary for substitution
        step_outputs: Dictionary of step outputs for reference
        
    Returns:
        Resolved value
    """
    if isinstance(value, str):
        # First, try to resolve step outputs (e.g., ${step_name.output_key})
        if step_outputs:
            for step_name, outputs in step_outputs.items():
                for output_key, output_value in outputs.items():
                    pattern = f"${{{step_name}.{output_key}}}"
                    if pattern in value:
                        value = value.replace(pattern, str(output_value))
        
        # Then substitute variables
        for key, val in variables.items():
            # Handle nested dict access (e.g., ${parameters.affilter.filter_loops})
            if '.' in key:
                parts = key.split('.')
                current = variables
                try:
                    for part in parts:
                        current = current[part]
                    value = value.replace(f"${{{key}}}", str(current))
                except (KeyError, TypeError):
                    pass
            else:
                value = value.replace(f"${{{key}}}", str(val))
        
        return value
    elif isinstance(value, dict):
        return {k: resolve_value(v, base_dir, variables, step_outputs) for k, v in value.items()}
    elif isinstance(value, list):
        return [resolve_value(item, base_dir, variables, step_outputs) for item in value]
    else:
        return value


def load_workflow(workflow_path: Path) -> Workflow:
    """Load workflow from YAML or JSON file.
    
    Args:
        workflow_path: Path to workflow file
        
    Returns:
        Workflow object
    """
    workflow_path = Path(workflow_path)
    
    if workflow_path.suffix in ['.yaml', '.yml']:
        with open(workflow_path, 'r') as f:
            data = yaml.safe_load(f)
    elif workflow_path.suffix == '.json':
        with open(workflow_path, 'r') as f:
            data = json.load(f)
    else:
        raise ValueError(f"Unknown workflow file format: {workflow_path.suffix}")
    
    # Parse workflow
    name = data.get('name', 'unnamed_workflow')
    description = data.get('description', '')
    version = data.get('version', '1.0')
    variables = data.get('variables', {})
    files = data.get('files', {})
    parameters = data.get('parameters', {})
    steps_data = data.get('steps', [])
    
    # Parse steps
    steps = []
    for step_data in steps_data:
        step = WorkflowStep(
            name=step_data['name'],
            tool=step_data['tool'],
            command=step_data.get('command', ''),
            inputs=step_data.get('inputs', {}),
            outputs=step_data.get('outputs', {}),
            depends_on=step_data.get('depends_on', []),
            condition=step_data.get('condition'),
        )
        steps.append(step)
    
    return Workflow(
        name=name,
        description=description,
        version=version,
        variables=variables,
        files=files,
        parameters=parameters,
        steps=steps,
    )


def execute_step(
    step: WorkflowStep,
    workflow: Workflow,
    base_dir: Path,
    step_outputs: Dict[str, Dict[str, Any]],
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Execute a single workflow step.
    
    Args:
        step: WorkflowStep to execute
        workflow: Parent workflow
        base_dir: Base directory for paths
        step_outputs: Dictionary of outputs from previous steps
        dry_run: If True, only print command without executing
        
    Returns:
        Dictionary of step outputs
    """
    print(f"\n{'='*60}")
    print(f"Step: {step.name} ({step.tool})")
    print(f"{'='*60}")
    
    # Check condition
    if step.condition:
        # Safe condition evaluation
        try:
            # Only allow safe operations
            safe_dict = {
                "step_outputs": step_outputs,
                "workflow": workflow,
                "len": len,
                "bool": bool,
                "str": str,
                "int": int,
                "float": float,
            }
            condition_met = eval(step.condition, {"__builtins__": {}}, safe_dict)
            if not condition_met:
                print(f"  Condition not met: {step.condition}")
                print(f"  Skipping step.")
                return {}
        except Exception as e:
            print(f"  ⚠ Warning: Could not evaluate condition '{step.condition}': {e}")
            print(f"  Proceeding with step execution.")
    
    # Resolve inputs (can reference previous step outputs)
    resolved_inputs = {}
    for key, value in step.inputs.items():
        resolved_value = resolve_value(value, base_dir, workflow.variables, step_outputs)
        resolved_inputs[key] = resolved_value
    
    # Build command
    cmd_parts = ["crymodel", step.tool, step.command]
    
    # Define which inputs are positional arguments (not options)
    # These vary by tool - need to handle each tool's CLI structure
    # Format: {tool: {command: [ordered_positional_args]}}
    positional_args_map = {
        "affilter": {
            "filter": ["input_pdb"],  # affilter filter <input_pdb> [options]
        },
        "foldhunter": {
            "search": ["target_map"],  # foldhunter search <target_map> [options]
        },
        "findligands": {
            "findligands": [],  # findligands --map <map> --model <model> (all are options)
        },
        "predictligands": {
            "predict": [],  # All are options
        },
        "pathwalker2": {
            "discover": ["map"],  # pathwalker2 discover <map> [options]
        },
        "validate": {
            "validate": [],  # validate --model <model> --map <map> (all are options)
        },
    }
    
    # Get positional args for this tool+command
    tool_positional = []
    if step.tool in positional_args_map:
        tool_cmds = positional_args_map[step.tool]
        if step.command in tool_cmds:
            tool_positional = tool_cmds[step.command]
    
    # Add positional arguments first (in order)
    for pos_arg in tool_positional:
        if pos_arg in resolved_inputs:
            cmd_parts.append(str(resolved_inputs[pos_arg]))
    
    # Add inputs as command-line options (skip positional args already added)
    for key, value in resolved_inputs.items():
        if key in tool_positional:
            continue  # Already added as positional
        if value is None:
            continue
        elif isinstance(value, bool):
            if value:
                cmd_parts.append(f"--{key.replace('_', '-')}")
        elif isinstance(value, (list, tuple)):
            for item in value:
                cmd_parts.append(f"--{key.replace('_', '-')}")
                cmd_parts.append(str(item))
        else:
            cmd_parts.append(f"--{key.replace('_', '-')}")
            cmd_parts.append(str(value))
    
    # Add outputs (resolve paths and add to command)
    resolved_outputs = {}
    for key, value in step.outputs.items():
        # Resolve value (may contain variables)
        resolved_value = resolve_value(value, base_dir, workflow.variables, step_outputs)
        resolved_outputs[key] = resolved_value
        
        # Add output to command if it's a file path parameter
        # Map common output keys to CLI parameter names
        output_param_map = {
            'output_pdb': 'output',
            'filtered_pdb': 'output',
            'best_fit_pdb': 'output',
            'out_dir': 'out-dir',
        }
        param_name = output_param_map.get(key, key.replace('_', '-'))
        if param_name in ['output', 'out-dir']:
            cmd_parts.append(f"--{param_name}")
            cmd_parts.append(str(resolved_value))
    
    # Print command
    cmd_str = " ".join(cmd_parts)
    print(f"  Command: {cmd_str}")
    
    if dry_run:
        print(f"  [DRY RUN - not executing]")
        return resolved_outputs
    
    # Execute command
    try:
        result = subprocess.run(
            cmd_parts,
            check=True,
            capture_output=True,
            text=True,
        )
        print(f"  ✓ Step completed successfully")
        if result.stdout:
            print(f"  Output:\n{result.stdout}")
        return resolved_outputs
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Step failed with error:")
        print(f"  {e.stderr}")
        raise


def execute_workflow(
    workflow_path: Path,
    base_dir: Optional[Path] = None,
    dry_run: bool = False,
    step_names: Optional[List[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Execute a complete workflow.
    
    Args:
        workflow_path: Path to workflow file
        base_dir: Base directory (default: workflow file directory)
        dry_run: If True, only print commands without executing
        step_names: Optional list of step names to execute (default: all)
        
    Returns:
        Dictionary mapping step names to their outputs
    """
    workflow_path = Path(workflow_path)
    if base_dir is None:
        base_dir = workflow_path.parent
    
    # Load workflow
    workflow = load_workflow(workflow_path)
    
    print(f"Workflow: {workflow.name}")
    print(f"Description: {workflow.description}")
    print(f"Version: {workflow.version}")
    print(f"Base directory: {base_dir}")
    
    # Resolve file paths
    resolved_files = {}
    for key, path in workflow.files.items():
        resolved_files[key] = str(resolve_path(path, base_dir, workflow.variables))
    
    # Update variables with resolved files
    workflow.variables.update(resolved_files)
    
    # Determine execution order (topological sort based on dependencies)
    executed = set()
    step_outputs = {}
    
    # Build step lookup
    step_lookup = {step.name: step for step in workflow.steps}
    
    # Filter steps if step_names specified
    if step_names:
        steps_to_execute = [step for step in workflow.steps if step.name in step_names]
        # Also include dependencies
        all_needed = set(step_names)
        for step_name in step_names:
            if step_name in step_lookup:
                all_needed.update(step_lookup[step_name].depends_on)
        steps_to_execute = [step for step in workflow.steps if step.name in all_needed]
    else:
        steps_to_execute = workflow.steps
    
    # Simple dependency resolution (assumes DAG)
    remaining_steps = {step.name: step for step in steps_to_execute}
    
    while remaining_steps:
        # Find steps with no unmet dependencies
        ready_steps = []
        for step_name, step in remaining_steps.items():
            if all(dep in executed for dep in step.depends_on):
                ready_steps.append(step)
        
        if not ready_steps:
            # Circular dependency or missing dependency
            remaining = list(remaining_steps.keys())
            raise ValueError(f"Cannot resolve dependencies. Remaining steps: {remaining}")
        
        # Execute ready steps
        for step in ready_steps:
            try:
                outputs = execute_step(step, workflow, base_dir, step_outputs, dry_run=dry_run)
                step_outputs[step.name] = outputs
                executed.add(step.name)
                del remaining_steps[step.name]
            except Exception as e:
                print(f"\n✗ Workflow failed at step: {step.name}")
                raise
    
    print(f"\n{'='*60}")
    print(f"Workflow completed successfully!")
    print(f"{'='*60}")
    
    return step_outputs

