# crymodel/assistant/assistant.py
"""AI assistant for CryoModel guidance."""
from __future__ import annotations

import json
import re
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

from .knowledge_base import (
    TOOL_DESCRIPTIONS,
    WORKFLOW_TEMPLATES,
    ERROR_PATTERNS,
    RESOLUTION_GUIDANCE,
)


class CryoModelAssistant:
    """AI assistant for guiding CryoModel usage."""
    
    def __init__(self):
        self.tool_descriptions = TOOL_DESCRIPTIONS
        self.workflow_templates = WORKFLOW_TEMPLATES
        self.error_patterns = ERROR_PATTERNS
        self.resolution_guidance = RESOLUTION_GUIDANCE
    
    def answer_question(self, question: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Answer user question with context.
        
        Args:
            question: User's question
            context: Optional context (files, errors, tool outputs, etc.)
            
        Returns:
            Formatted answer string
        """
        question_lower = question.lower()
        context = context or {}
        context = self._augment_context_with_history(context)
        
        # Route to appropriate handler
        if any(phrase in question_lower for phrase in ["how do i", "how can i", "how to", "what steps"]):
            return self._suggest_workflow(question, context)
        elif any(phrase in question_lower for phrase in ["error", "failed", "problem", "issue", "not working", "broken"]):
            return self._troubleshoot_error(question, context)
        elif any(phrase in question_lower for phrase in ["parameter", "option", "flag", "what does", "explain"]):
            return self._explain_parameter(question, context)
        elif any(phrase in question_lower for phrase in ["bad results", "poor", "low quality", "not good"]):
            return self._diagnose_quality_issue(question, context)
        elif any(phrase in question_lower for phrase in ["what is", "what does", "describe", "tell me about"]):
            return self._explain_tool(question, context)
        else:
            return self._general_guidance(question, context)

    def _augment_context_with_history(self, context: Dict[str, Any]) -> Dict[str, Any]:
        if context.get("disable_history"):
            return context
        if context.get("history"):
            return context
        cwd = context.get("cwd")
        if cwd:
            base = Path(cwd)
        else:
            base = Path.cwd()
        history_path = base / ".crymodel_history.jsonl"
        records = self._load_history(history_path, limit=100)
        if not records:
            return context
        history_context = self._summarize_history(records)
        context.update(history_context)
        return context

    def _load_history(self, path: Path, limit: int = 100) -> List[Dict[str, Any]]:
        if not path.exists():
            return []
        records = []
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return records[-limit:] if limit > 0 else records

    def _summarize_history(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        recent_tools: List[str] = []
        recent_files: List[str] = []
        last_error = None
        last_error_output = None

        for record in reversed(records):
            tool = record.get("tool")
            if tool and tool not in recent_tools:
                recent_tools.append(tool)
            if record.get("status") == "error" and last_error is None:
                last_error = record
            if record.get("status") == "error" and not last_error_output:
                last_error_output = record.get("output_log")
            argv = record.get("argv") or []
            for arg in argv:
                arg_str = str(arg)
                if "/" in arg_str or any(arg_str.lower().endswith(ext) for ext in [".mrc", ".map", ".pdb", ".cif", ".csv", ".json", ".yaml", ".yml"]):
                    if arg_str not in recent_files:
                        recent_files.append(arg_str)

        return {
            "history": records,
            "recent_tools": recent_tools[:10],
            "recent_files": recent_files[:10],
            "last_error_record": last_error,
            "last_error_output_log": last_error_output,
        }
    
    def _suggest_workflow(self, question: str, context: Dict[str, Any]) -> str:
        """Suggest a workflow based on user goal."""
        question_lower = question.lower()
        
        # Detect goal from question
        goal = None
        if "build model" in question_lower or "model from sequence" in question_lower:
            goal = "build_model_from_sequence"
        elif "ligand" in question_lower or "find ligand" in question_lower:
            goal = "find_and_classify_ligands"
        elif "trace" in question_lower or "backbone" in question_lower:
            goal = "trace_backbone"
        elif "dna" in question_lower or "centerline" in question_lower:
            goal = "trace_dna_centerline"
        elif "basehunter" in question_lower or "purine" in question_lower or "pyrimidine" in question_lower:
            goal = "basehunter_classify"
        if "ligand" in question_lower and "validate" in question_lower:
            goal = "ligand_qc"
        if "dna" in question_lower and "basehunter" in question_lower and "build" in question_lower:
            goal = "dna_basehunter_build"
        
        if goal and goal in self.workflow_templates:
            template = self.workflow_templates[goal]
            response = f"**Suggested Workflow: {template['description']}**\n\n"
            
            if "prerequisites" in template:
                response += f"**Prerequisites:** {', '.join(template['prerequisites'])}\n\n"
            
            response += "**Steps:**\n"
            for step_info in template["steps"]:
                response += f"\n{step_info['step']}. **{step_info['action']}**\n"
                response += f"   ```bash\n   {step_info['command']}\n   ```\n"
            
            if "notes" in template:
                response += "\n**Notes:**\n"
                for note in template["notes"]:
                    response += f"- {note}\n"

            recent_files = context.get("recent_files", [])
            recent_tools = context.get("recent_tools", [])
            if recent_files or recent_tools:
                response += "\n**Context from recent runs in this folder:**\n"
                if recent_tools:
                    response += f"- Recent tools: {', '.join(recent_tools)}\n"
                if recent_files:
                    response += f"- Recent files: {', '.join(recent_files)}\n"
            
            return response
        
        # Generic workflow suggestion
        return (
            "I can help you with several common workflows:\n\n"
            "1. **Build model from sequence**: affilter → foldhunter → loopcloud → validate\n"
            "2. **Find and classify ligands**: findligands → predictligands\n"
            "3. **Trace backbone**: pathwalker2 → validate\n\n"
            "Could you provide more details about what you're trying to accomplish?\n"
            "For example: 'I have a sequence and density map, how do I build a model?'"
        )
    
    def _troubleshoot_error(self, question: str, context: Dict[str, Any]) -> str:
        """Provide troubleshooting steps for an error."""
        error_message = context.get("error_message", question)
        tool_name = context.get("tool_name", "")
        
        error_lower = error_message.lower()
        
        # Match error to patterns
        matched_patterns = []
        for pattern_name, pattern_info in self.error_patterns.items():
            for pattern in pattern_info["patterns"]:
                if pattern.lower() in error_lower:
                    matched_patterns.append((pattern_name, pattern_info))
                    break
        
        if matched_patterns:
            response = "**Troubleshooting Steps:**\n\n"
            for pattern_name, pattern_info in matched_patterns:
                response += f"**Issue: {pattern_name.replace('_', ' ').title()}**\n\n"
                response += "**Solutions:**\n"
                for i, solution in enumerate(pattern_info["solutions"], 1):
                    response += f"{i}. {solution}\n"
                
                if "prevention" in pattern_info:
                    response += f"\n**Prevention:** {pattern_info['prevention']}\n"
                response += "\n"
            
            return response
        
        # Check tool-specific errors
        if tool_name and tool_name in self.tool_descriptions:
            tool_info = self.tool_descriptions[tool_name]
            if "common_errors" in tool_info:
                response = f"**Tool-specific troubleshooting for {tool_name}:**\n\n"
                for error_name, error_info in tool_info["common_errors"].items():
                    if any(symptom.lower() in error_lower for symptom in error_info.get("symptoms", [])):
                        response += f"**{error_name.replace('_', ' ').title()}**\n\n"
                        response += "**Solutions:**\n"
                        for i, solution in enumerate(error_info["solutions"], 1):
                            response += f"{i}. {solution}\n"
                        response += "\n"
                
                if response != f"**Tool-specific troubleshooting for {tool_name}:**\n\n":
                    return response

        last_error = context.get("last_error_record")
        if last_error:
            response = "**Recent error in this folder:**\n\n"
            response += f"- Tool: {last_error.get('tool', 'unknown')}\n"
            response += f"- Command: {last_error.get('command', 'unknown')}\n"
            response += f"- Status: {last_error.get('status', 'unknown')}\n"
            if last_error.get("output_log"):
                response += f"- Output log: {last_error.get('output_log')}\n"
            response += "\nIf this is the same issue, share the relevant lines from the output log."
            return response
        
        return (
            "I couldn't find a specific match for this error. Here are general troubleshooting steps:\n\n"
            "1. Check that all input files exist and are readable\n"
            "2. Verify file formats are correct (.mrc for maps, .pdb for models)\n"
            "3. Check that parameters are within reasonable ranges\n"
            "4. Try running with --dry-run or validate inputs first\n"
            "5. Check the tool's documentation for parameter guidance\n\n"
            "If you can share the full error message and which tool you're using, I can provide more specific help."
        )
    
    def _explain_parameter(self, question: str, context: Dict[str, Any]) -> str:
        """Explain a parameter or option."""
        question_lower = question.lower()
        tool_name = context.get("tool_name", "")
        
        # Extract parameter name from question
        param_match = re.search(r'--?([\w-]+)|parameter\s+([\w-]+)', question_lower)
        if param_match:
            param_name = param_match.group(1) or param_match.group(2)
            param_name = param_name.replace("-", "_")
        else:
            param_name = None
        
        if tool_name and tool_name in self.tool_descriptions:
            tool_info = self.tool_descriptions[tool_name]
            if "parameter_guidance" in tool_info:
                if param_name and param_name in tool_info["parameter_guidance"]:
                    return f"**Parameter: {param_name}**\n\n{tool_info['parameter_guidance'][param_name]}"
                else:
                    # List all parameters
                    response = f"**Parameters for {tool_name}:**\n\n"
                    for param, guidance in tool_info["parameter_guidance"].items():
                        response += f"**{param}**: {guidance}\n\n"
                    return response
        
        return (
            "I can explain parameters for specific tools. Please specify:\n"
            "- Which tool you're asking about\n"
            "- Which parameter (e.g., 'What does the threshold parameter do in findligands?')"
        )
    
    def _diagnose_quality_issue(self, question: str, context: Dict[str, Any]) -> str:
        """Diagnose quality issues with results."""
        tool_name = context.get("tool_name", "")
        resolution = context.get("resolution")
        
        response = "**Quality Issue Diagnosis:**\n\n"
        
        if tool_name == "foldhunter" and resolution:
            if resolution <= 2.5:
                guidance = self.resolution_guidance["high_resolution"]["foldhunter"]
                response += f"At high resolution (≤2.5 Å), you should expect:\n"
                response += f"- Correlation: {guidance['expect_correlation']}\n"
            elif resolution <= 3.5:
                guidance = self.resolution_guidance["medium_resolution"]["foldhunter"]
                response += f"At medium resolution (2.5-3.5 Å), you should expect:\n"
                response += f"- Correlation: {guidance['expect_correlation']}\n"
            else:
                guidance = self.resolution_guidance["low_resolution"]["foldhunter"]
                response += f"At low resolution (>3.5 Å), you should expect:\n"
                response += f"- Correlation: {guidance['expect_correlation']}\n"
        
        if tool_name and tool_name in self.tool_descriptions:
            tool_info = self.tool_descriptions[tool_name]
            if "common_errors" in tool_info:
                response += "\n**Common issues and solutions:**\n\n"
                for error_name, error_info in tool_info["common_errors"].items():
                    if "low" in error_name or "poor" in error_name or "bad" in error_name:
                        response += f"**{error_name.replace('_', ' ').title()}**\n"
                        for solution in error_info["solutions"]:
                            response += f"- {solution}\n"
                        response += "\n"
        
        response += "\n**General recommendations:**\n"
        response += "1. Validate inputs with fitprep before running tools\n"
        response += "2. Check map resolution matches your expectations\n"
        response += "3. Try different parameter values (threshold, etc.)\n"
        response += "4. Use validate tool to assess result quality\n"
        
        return response
    
    def _explain_tool(self, question: str, context: Dict[str, Any]) -> str:
        """Explain what a tool does."""
        question_lower = question.lower()
        
        # Find tool mentioned in question
        for tool_name, tool_info in self.tool_descriptions.items():
            if tool_name in question_lower:
                response = f"**{tool_name}**\n\n"
                response += f"**Purpose:** {tool_info['purpose']}\n\n"
                response += f"**Description:** {tool_info.get('description', '')}\n\n"
                
                if "common_use_cases" in tool_info:
                    response += "**Common use cases:**\n"
                    for use_case in tool_info["common_use_cases"]:
                        response += f"- {use_case}\n"
                    response += "\n"
                
                if "typical_workflow" in tool_info:
                    response += f"**Typical workflow:** {tool_info['typical_workflow']}\n\n"
                
                return response
        
        return (
            "I can explain any CryoModel tool. Please specify which tool, for example:\n"
            "- 'What does findligands do?'\n"
            "- 'Tell me about foldhunter'\n"
            "- 'Explain pathwalker2'"
        )
    
    def _general_guidance(self, question: str, context: Dict[str, Any]) -> str:
        """Provide general guidance."""
        response = (
            "I'm here to help you use CryoModel effectively! I can:\n\n"
            "1. **Suggest workflows** - Tell me what you want to accomplish\n"
            "2. **Troubleshoot errors** - Share error messages and I'll help fix them\n"
            "3. **Explain parameters** - Ask about specific tool options\n"
            "4. **Explain tools** - Learn what each tool does\n"
            "5. **Diagnose issues** - Help with poor results or quality problems\n\n"
            "Try asking:\n"
            "- 'How do I build a model from my sequence?'\n"
            "- 'What does the threshold parameter do?'\n"
            "- 'foldhunter gave me low correlation, what should I do?'\n"
            "- 'What is findligands used for?'"
        )
        recent_tools = context.get("recent_tools", [])
        recent_files = context.get("recent_files", [])
        if recent_tools or recent_files:
            response += "\n\nRecent activity detected in this folder:\n"
            if recent_tools:
                response += f"- Tools: {', '.join(recent_tools)}\n"
            if recent_files:
                response += f"- Files: {', '.join(recent_files)}\n"
        return response
    
    def suggest_workflow(self, goal: str, available_files: List[str]) -> Dict[str, Any]:
        """Suggest a workflow based on goal and available files.
        
        Args:
            goal: What the user wants to accomplish
            available_files: List of file paths user has
            
        Returns:
            Workflow suggestion dictionary
        """
        goal_lower = goal.lower()
        
        # Match goal to template
        template_key = None
        if "build model" in goal_lower or "sequence" in goal_lower:
            template_key = "build_model_from_sequence"
        elif "ligand" in goal_lower:
            template_key = "find_and_classify_ligands"
        elif "trace" in goal_lower or "backbone" in goal_lower:
            template_key = "trace_backbone"
        elif "dna" in goal_lower or "centerline" in goal_lower:
            template_key = "trace_dna_centerline"
        elif "basehunter" in goal_lower or "purine" in goal_lower or "pyrimidine" in goal_lower:
            template_key = "basehunter_classify"
        if "ligand" in goal_lower and "validate" in goal_lower:
            template_key = "ligand_qc"
        if "dna" in goal_lower and "basehunter" in goal_lower and "build" in goal_lower:
            template_key = "dna_basehunter_build"
        
        if template_key and template_key in self.workflow_templates:
            template = self.workflow_templates[template_key]
            
            # Check prerequisites
            missing = []
            for prereq in template.get("prerequisites", []):
                # Simple check - could be more sophisticated
                if not any(prereq.lower() in f.lower() for f in available_files):
                    missing.append(prereq)
            
            return {
                "workflow": template,
                "missing_prerequisites": missing,
                "ready": len(missing) == 0,
            }
        
        return {"workflow": None, "missing_prerequisites": [], "ready": False}
    
    def get_resolution_guidance(self, resolution: float) -> Dict[str, Any]:
        """Get parameter guidance based on map resolution.
        
        Args:
            resolution: Map resolution in Å
            
        Returns:
            Dictionary with resolution-specific guidance
        """
        if resolution <= 2.5:
            return self.resolution_guidance["high_resolution"]
        elif resolution <= 3.5:
            return self.resolution_guidance["medium_resolution"]
        else:
            return self.resolution_guidance["low_resolution"]

