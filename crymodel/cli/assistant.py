# crymodel/cli/assistant.py
"""CLI for CryoModel AI assistant."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Any, Dict
import typer
import json
import yaml

from ..assistant.assistant import CryoModelAssistant

app = typer.Typer(no_args_is_help=True)


@app.command()
def ask(
    question: str = typer.Argument(..., help="Your question about CryoModel"),
    context_file: Optional[Path] = typer.Option(None, "--context", help="JSON file with context (files, errors, tool_name, etc.)"),
    tool: Optional[str] = typer.Option(None, "--tool", help="Tool name for context"),
    resolution: Optional[float] = typer.Option(None, "--resolution", help="Map resolution (Å) for context"),
):
    """Ask the AI assistant a question about CryoModel.
    
    Examples:
        crymodel assistant ask "How do I build a model from my sequence?"
        crymodel assistant ask "foldhunter failed with low correlation" --tool foldhunter
        crymodel assistant ask "What does the threshold parameter do?" --tool findligands
    """
    assistant = CryoModelAssistant()
    
    # Load context
    context = {}
    if context_file and context_file.exists():
        context = json.load(open(context_file))
    
    if tool:
        context["tool_name"] = tool
    
    if resolution:
        context["resolution"] = resolution
    
    # Get answer
    answer = assistant.answer_question(question, context)
    
    # Print formatted answer
    typer.echo(answer)


@app.command()
def suggest(
    goal: str = typer.Argument(..., help="What you want to accomplish"),
    files: Optional[str] = typer.Option(None, "--files", help="Comma-separated list of files you have"),
    generate: Optional[Path] = typer.Option(None, "--generate", help="Generate workflow.yaml file at this path"),
    resolution: Optional[float] = typer.Option(None, "--resolution", help="Map resolution (Å) for parameter defaults"),
    interactive: bool = typer.Option(True, "--interactive/--no-interactive", help="Prompt for additional parameters"),
):
    """Get workflow suggestions and optionally generate a workflow.yaml file.
    
    Examples:
        crymodel assistant suggest "build a model from sequence" --files sequence.fasta,map.mrc
        crymodel assistant suggest "find ligands" --files model.pdb,map.mrc --generate workflow.yaml
    """
    assistant = CryoModelAssistant()
    
    file_list = []
    if files:
        file_list = [f.strip() for f in files.split(",")]
    
    suggestion = assistant.suggest_workflow(goal, file_list)
    
    if not suggestion["workflow"]:
        typer.echo("I couldn't match your goal to a known workflow. Try being more specific:")
        typer.echo("  - 'build a model from sequence'")
        typer.echo("  - 'find and classify ligands'")
        typer.echo("  - 'trace protein backbone'")
        raise typer.Exit(1)
    
    workflow = suggestion["workflow"]
    typer.echo(f"**Suggested Workflow: {workflow['description']}**\n")
    
    if suggestion["missing_prerequisites"]:
        typer.echo("⚠️  **Missing prerequisites:**")
        for missing in suggestion["missing_prerequisites"]:
            typer.echo(f"  - {missing}")
        typer.echo()
    
    # Map files to workflow variables
    file_mapping = _map_files_to_workflow(file_list, workflow)
    
    # Show file mapping
    if file_mapping:
        typer.echo("**File Mapping:**")
        for key, value in file_mapping.items():
            typer.echo(f"  {key}: {value}")
        typer.echo()
    
    # Collect parameters interactively if requested
    parameters = {}
    if generate:
        typer.echo("**Workflow Steps:**")
        for step_info in workflow["steps"]:
            typer.echo(f"\n{step_info['step']}. {step_info['action']}")
            typer.echo(f"   Tool: {step_info['tool']}")
        
        typer.echo("\n" + "="*60)
        typer.echo("**Parameter Configuration**")
        typer.echo("="*60 + "\n")
        
        # Get resolution if not provided
        if not resolution:
            resolution_str = typer.prompt("Map resolution (Å)", default="3.0")
            try:
                resolution = float(resolution_str)
            except ValueError:
                resolution = 3.0
                typer.echo("⚠️  Invalid resolution, using default 3.0")
        
        parameters["resolution"] = resolution
        
        # Get tool-specific parameters
        for step_info in workflow["steps"]:
            tool_name = step_info["tool"]
            if tool_name in assistant.tool_descriptions:
                tool_info = assistant.tool_descriptions[tool_name]
                if "parameter_guidance" in tool_info:
                    typer.echo(f"\n**Parameters for {tool_name}:**")
                    for param_name, param_guidance in tool_info["parameter_guidance"].items():
                        # Extract default from guidance or use common defaults
                        default = _get_default_parameter(param_name, tool_name, resolution, assistant)
                        if default is None:
                            default = 0.5  # Last resort fallback
                        
                        if interactive:
                            # Shorten guidance for prompt
                            guidance_short = param_guidance.split('.')[0][:50] if len(param_guidance) > 50 else param_guidance.split('.')[0]
                            prompt_text = f"  {param_name}"
                            if guidance_short:
                                prompt_text += f" [{guidance_short}]"
                            value_str = typer.prompt(prompt_text, default=str(default))
                            try:
                                # Try to convert to appropriate type
                                if "." in value_str or "e" in value_str.lower():
                                    parameters[f"{tool_name}_{param_name}"] = float(value_str)
                                else:
                                    parameters[f"{tool_name}_{param_name}"] = int(value_str)
                            except ValueError:
                                # Keep as string (for boolean-like strings)
                                if value_str.lower() in ["true", "false"]:
                                    parameters[f"{tool_name}_{param_name}"] = value_str.lower() == "true"
                                else:
                                    parameters[f"{tool_name}_{param_name}"] = value_str
                        else:
                            # Use defaults without prompting
                            if default is None:
                                default = 0.5  # Last resort fallback
                            parameters[f"{tool_name}_{param_name}"] = default
                            typer.echo(f"  {param_name}: {default} (default)")
        
        # Show summary
        typer.echo("\n" + "="*60)
        typer.echo("**Workflow Summary**")
        typer.echo("="*60)
        typer.echo(f"Resolution: {resolution} Å")
        typer.echo(f"Files: {', '.join(file_list) if file_list else 'None specified'}")
        typer.echo(f"Parameters: {len(parameters)} configured")
    
    # Generate workflow.yaml if requested
    if generate:
        workflow_yaml = _generate_workflow_yaml(
            workflow, file_mapping, parameters, resolution, assistant
        )
        
        generate_path = Path(generate)
        with open(generate_path, 'w') as f:
            yaml.dump(workflow_yaml, f, default_flow_style=False, sort_keys=False)
        
        typer.echo(f"\n✓ Workflow file generated: {generate_path}")
        typer.echo(f"\nRun with: crymodel workflow run {generate_path}")
    else:
        # Just show the workflow
        typer.echo("\n**Steps:**")
        for step_info in workflow["steps"]:
            typer.echo(f"\n{step_info['step']}. {step_info['action']}")
            typer.echo(f"   {step_info['command']}")
        
        if "notes" in workflow:
            typer.echo("\n**Notes:**")
            for note in workflow["notes"]:
                typer.echo(f"  - {note}")
        
        typer.echo("\n💡 Tip: Use --generate workflow.yaml to create a ready-to-run workflow file!")


def _map_files_to_workflow(file_list: List[str], workflow: Dict) -> Dict[str, str]:
    """Map user-provided files to workflow file variables."""
    mapping = {}
    
    # Simple heuristics to match files
    pdb_files = []
    for file_path in file_list:
        file_lower = file_path.lower()
        if "alphafold" in file_lower or "af" in file_lower:
            mapping["alphafold_pdb"] = file_path
        elif "map" in file_lower or file_path.endswith(".mrc") or file_path.endswith(".map"):
            mapping["target_map"] = file_path
        elif "sequence" in file_lower or file_path.endswith(".fasta") or file_path.endswith(".fa"):
            mapping["sequence"] = file_path
        elif file_path.endswith(".pdb"):
            pdb_files.append(file_path)
    
    # Assign PDB files if not already assigned
    if "alphafold_pdb" not in mapping and pdb_files:
        mapping["alphafold_pdb"] = pdb_files[0]
    if "model_pdb" not in mapping and len(pdb_files) > 1:
        mapping["model_pdb"] = pdb_files[1]
    elif "model_pdb" not in mapping and pdb_files:
        mapping["model_pdb"] = pdb_files[0]
    
    return mapping


def _get_default_parameter(param_name: str, tool_name: str, resolution: float, assistant: CryoModelAssistant) -> Any:
    """Get default parameter value based on name, tool, and resolution."""
    # Tool-specific defaults (matching actual CLI defaults)
    tool_defaults = {
        "affilter": {
            "plddt_threshold": 0.5,
            "filter_loops": True,
            "filter_connectivity": True,
            "max_ca_distance": 4.5,
            "min_loop_length": 10,
            "max_loop_length": 50,
            "connectivity_threshold": 6.0,
            "min_neighbors": 2,
            "clustering_method": "dbscan",
            "clustering_eps": 15.0,
            "clustering_min_samples": 10,
        },
        "foldhunter": {
            "resolution": resolution,  # Use provided resolution
            "plddt_threshold": 0.5,
            "n_coarse_rotations": 1000,
            "n_fine_rotations": 500,
            "coarse_angle_step": 15.0,
            "fine_angle_step": 5.0,
            "coarse_translation_step": 5.0,
            "fine_translation_step": 1.0,
        },
        "findligands": {
            "thresh": 0.5 if resolution <= 2.5 else 0.4 if resolution <= 3.5 else 0.3,
            "mask_radius": 2.0,
            "micro_vvox_min": 2,  # Actual CLI default
            "micro_vvox_max": 12,
            "zero_radius": 2.0,
            "water_gate_min": 2.0,
            "water_gate_max": 6.0,
            "water_cluster_radius": 2.4,
            "ligand_zero_radius": 1.5,
            "ligand_gate_min": 2.0,
            "ligand_gate_max": 10.0,
        },
        "pathwalker2": {
            "threshold": 0.013 if resolution <= 2.5 else 0.02 if resolution <= 3.5 else 0.05,
            "map_prep": "locscale",
        },
    }
    
    # Get tool-specific defaults
    if tool_name in tool_defaults:
        if param_name in tool_defaults[tool_name]:
            return tool_defaults[tool_name][param_name]
    
    # Get resolution-specific guidance as fallback
    guidance = assistant.get_resolution_guidance(resolution)
    
    # Check if parameter has resolution-specific guidance
    if tool_name in guidance and param_name in guidance[tool_name]:
        value = guidance[tool_name][param_name]
        # Handle range strings like "0.3-0.5"
        if isinstance(value, str) and "-" in value:
            try:
                parts = value.split("-")
                return (float(parts[0]) + float(parts[1])) / 2
            except:
                pass
        return value
    
    # Generic fallback (should rarely be used)
    generic_defaults = {
        "threshold": 0.5,
        "plddt_threshold": 0.5,
        "mask_radius": 2.0,
        "n_coarse_rotations": 1000,
        "n_fine_rotations": 500,
    }
    
    return generic_defaults.get(param_name, None)


def _generate_workflow_yaml(
    workflow_template: Dict,
    file_mapping: Dict[str, str],
    parameters: Dict[str, Any],
    resolution: float,
    assistant: CryoModelAssistant,
) -> Dict:
    """Generate a workflow.yaml structure from template and user inputs."""
    
    # Determine workflow name from template
    workflow_name = workflow_template.get("description", "workflow").lower().replace(" ", "_")
    
    # Build workflow structure
    workflow_yaml = {
        "name": workflow_name,
        "description": workflow_template["description"],
        "version": "1.0",
        "variables": {
            "resolution": resolution,
            "plddt_threshold": parameters.get("affilter_plddt_threshold", 0.5),
        },
        "files": file_mapping,
        "parameters": {},
        "steps": [],
    }
    
    # Add tool-specific parameters
    tool_params = {}
    for key, value in parameters.items():
        if "_" in key:
            tool, param = key.split("_", 1)
            if tool not in tool_params:
                tool_params[tool] = {}
            tool_params[tool][param] = value
    
    workflow_yaml["parameters"] = tool_params
    
    # Generate steps
    step_dependencies = []
    for step_info in workflow_template["steps"]:
        step_num = step_info["step"]
        tool_name = step_info["tool"]
        command = step_info.get("command", "")
        
        # Build step
        step = {
            "name": f"step_{step_num}_{tool_name}",
            "tool": tool_name,
            "command": command.split()[1] if len(command.split()) > 1 else command,
            "inputs": {},
            "outputs": {},
            "depends_on": step_dependencies.copy(),
        }
        
        # Add inputs based on tool
        # Note: Positional arguments should be listed first in inputs dict
        if tool_name == "affilter":
            # input_pdb is positional argument (first), not an option
            step["inputs"]["input_pdb"] = "${alphafold_pdb}" if "alphafold_pdb" in file_mapping else "alphafold_model.pdb"
            step["inputs"]["plddt_threshold"] = "${plddt_threshold}"
            if f"affilter_filter_loops" in parameters:
                step["inputs"]["filter_loops"] = parameters.get(f"affilter_filter_loops", True)
            if f"affilter_filter_connectivity" in parameters:
                step["inputs"]["filter_connectivity"] = parameters.get(f"affilter_filter_connectivity", True)
            step["inputs"]["output_pdb"] = f"outputs/{tool_name}/alphafold_filtered.pdb"
            step["inputs"]["out_dir"] = f"outputs/{tool_name}"
            step["outputs"]["filtered_pdb"] = f"outputs/{tool_name}/alphafold_filtered.pdb"
            step["outputs"]["low_plddt_csv"] = f"outputs/{tool_name}/affilter_low_plddt_regions.csv"
            
        elif tool_name == "foldhunter":
            step["inputs"]["target_map"] = "${target_map}"
            # Find previous affilter step
            prev_step_name = None
            for prev_step in workflow_yaml["steps"]:
                if prev_step["tool"] == "affilter":
                    prev_step_name = prev_step["name"]
                    break
            if prev_step_name:
                step["inputs"]["probe_pdb"] = f"${{{prev_step_name}.filtered_pdb}}"
            else:
                step["inputs"]["probe_pdb"] = "${alphafold_pdb}"
            step["inputs"]["resolution"] = "${resolution}"
            step["inputs"]["plddt_threshold"] = "${plddt_threshold}"
            if f"foldhunter_n_coarse_rotations" in parameters:
                step["inputs"]["n_coarse_rotations"] = parameters[f"foldhunter_n_coarse_rotations"]
            if f"foldhunter_n_fine_rotations" in parameters:
                step["inputs"]["n_fine_rotations"] = parameters[f"foldhunter_n_fine_rotations"]
            step["inputs"]["out_dir"] = f"outputs/{tool_name}"
            step["outputs"]["best_fit_pdb"] = f"outputs/{tool_name}/foldhunter_top_fit.pdb"
            
        elif tool_name == "findligands":
            step["inputs"]["map"] = "${target_map}" if "target_map" in file_mapping else "target_map.mrc"
            step["inputs"]["model"] = "${model_pdb}" if "model_pdb" in file_mapping else "model.pdb"
            if f"findligands_thresh" in parameters:
                step["inputs"]["thresh"] = parameters[f"findligands_thresh"]
            elif f"findligands_threshold" in parameters:
                step["inputs"]["thresh"] = parameters[f"findligands_threshold"]
            step["inputs"]["entry_resolution"] = "${resolution}"
            step["inputs"]["out_dir"] = f"outputs/{tool_name}"
            step["outputs"]["ligands_pdb"] = f"outputs/{tool_name}/ligands.pdb"
            step["outputs"]["ligand_map"] = f"outputs/{tool_name}/ligands_map.mrc"
            
        elif tool_name == "predictligands":
            # Find previous findligands step
            prev_step_name = None
            for prev_step in workflow_yaml["steps"]:
                if prev_step["tool"] == "findligands":
                    prev_step_name = prev_step["name"]
                    break
            if prev_step_name:
                step["inputs"]["ligands_pdb"] = f"${{{prev_step_name}.ligands_pdb}}"
                step["inputs"]["ligand_map"] = f"${{{prev_step_name}.ligand_map}}"
            else:
                step["inputs"]["ligands_pdb"] = "outputs/findligands/ligands.pdb"
                step["inputs"]["ligand_map"] = "outputs/findligands/ligands_map.mrc"
            step["inputs"]["model"] = "${model_pdb}"
            step["inputs"]["out_dir"] = f"outputs/{tool_name}"
            step["outputs"]["predictions_csv"] = f"outputs/{tool_name}/ligand-predictions.csv"
            
        elif tool_name == "pathwalker2":
            step["inputs"]["map"] = "${target_map}"
            if "pathwalker2_threshold" in parameters:
                step["inputs"]["threshold"] = parameters["pathwalker2_threshold"]
            # n_residues would need to be prompted
            step["inputs"]["out_dir"] = f"outputs/{tool_name}"
            step["outputs"]["fragments_pdb"] = f"outputs/{tool_name}/pathwalker2_fragments.pdb"
            
        elif tool_name == "validate":
            # Find previous step that produces a model
            prev_step_name = None
            for prev_step in reversed(workflow_yaml["steps"]):
                if prev_step["tool"] in ["foldhunter", "loopcloud", "pathwalker2"]:
                    prev_step_name = prev_step["name"]
                    break
            if prev_step_name:
                if prev_step["tool"] == "foldhunter":
                    step["inputs"]["model"] = f"${{{prev_step_name}.best_fit_pdb}}"
                elif prev_step["tool"] == "loopcloud":
                    step["inputs"]["model"] = f"${{{prev_step_name}.best_loop_pdb}}"
                elif prev_step["tool"] == "pathwalker2":
                    step["inputs"]["model"] = f"${{{prev_step_name}.fragments_pdb}}"
            else:
                step["inputs"]["model"] = "${model_pdb}"
            step["inputs"]["map"] = "${target_map}"
            step["inputs"]["out_dir"] = f"outputs/{tool_name}"
            step["outputs"]["validation_csv"] = f"outputs/{tool_name}/fitcheck_per_residue.csv"
        
        workflow_yaml["steps"].append(step)
        step_dependencies.append(step["name"])
    
    return workflow_yaml


@app.command()
def explain(
    tool_name: str = typer.Argument(..., help="Tool name to explain"),
    parameter: Optional[str] = typer.Option(None, "--parameter", help="Specific parameter to explain"),
):
    """Explain what a tool does or explain a specific parameter.
    
    Examples:
        crymodel assistant explain findligands
        crymodel assistant explain foldhunter --parameter plddt_threshold
    """
    assistant = CryoModelAssistant()
    
    context = {"tool_name": tool_name}
    if parameter:
        context["parameter"] = parameter
        question = f"What does the {parameter} parameter do in {tool_name}?"
    else:
        question = f"What does {tool_name} do?"
    
    answer = assistant.answer_question(question, context)
    typer.echo(answer)


@app.command()
def troubleshoot(
    error_message: str = typer.Argument(..., help="Error message or description of problem"),
    tool_name: Optional[str] = typer.Option(None, "--tool", help="Tool that produced the error"),
    context_file: Optional[Path] = typer.Option(None, "--context", help="JSON file with additional context"),
):
    """Get troubleshooting help for an error or problem.
    
    Examples:
        crymodel assistant troubleshoot "empty output files" --tool findligands
        crymodel assistant troubleshoot "low correlation" --tool foldhunter
    """
    assistant = CryoModelAssistant()
    
    context = {"error_message": error_message}
    if tool_name:
        context["tool_name"] = tool_name
    
    if context_file and context_file.exists():
        additional_context = json.load(open(context_file))
        context.update(additional_context)
    
    answer = assistant.answer_question(f"Error: {error_message}", context)
    typer.echo(answer)


@app.command()
def resolution(
    resolution_value: float = typer.Argument(..., help="Map resolution in Å"),
):
    """Get parameter recommendations based on map resolution.
    
    Examples:
        crymodel assistant resolution 2.8
        crymodel assistant resolution 3.5
    """
    assistant = CryoModelAssistant()
    
    guidance = assistant.get_resolution_guidance(resolution_value)
    
    typer.echo(f"**Resolution Guidance for {resolution_value} Å**\n")
    typer.echo(f"Resolution category: {guidance['range']}\n")
    
    for tool, params in guidance.items():
        if tool != "range":
            typer.echo(f"**{tool}:**")
            for param, value in params.items():
                typer.echo(f"  {param}: {value}")
            typer.echo()


if __name__ == "__main__":
    app()

