# CryoModel AI Assistant Guide

The CryoModel AI Assistant is a rule-based guidance system that helps users navigate the CryoModel toolkit effectively. It provides workflow suggestions, troubleshooting help, parameter explanations, and resolution-based guidance.

## Features

1. **Workflow Suggestions** - Get step-by-step workflows for common tasks
2. **Error Troubleshooting** - Diagnose and fix common errors
3. **Parameter Guidance** - Understand tool parameters and get recommendations
4. **Tool Explanations** - Learn what each tool does and when to use it
5. **Resolution-Based Advice** - Get parameter recommendations based on map resolution

## Usage

### Basic Commands

```bash
# Ask a question
crymodel assistant ask "How do I build a model from my sequence?"

# Get workflow suggestion
crymodel assistant suggest "find ligands" --files model.pdb,map.mrc

# Explain a tool
crymodel assistant explain findligands

# Explain a parameter
crymodel assistant explain foldhunter --parameter plddt_threshold

# Troubleshoot an error
crymodel assistant troubleshoot "empty output files" --tool findligands

# Get resolution-based guidance
crymodel assistant resolution 2.8
```

## Knowledge Base

The assistant uses a structured knowledge base containing:

### Tool Descriptions
- Purpose and description
- Input/output specifications
- Common use cases
- Typical workflows
- Parameter guidance
- Common errors and solutions

### Workflow Templates
- Build model from sequence
- Find and classify ligands
- Trace backbone

### Error Patterns
- Import errors
- Empty outputs
- Memory errors
- File not found
- Coordinate mismatches
- Low quality results

### Resolution Guidance
- High resolution (≤2.5 Å)
- Medium resolution (2.5-3.5 Å)
- Low resolution (>3.5 Å)

## Example Interactions

### Question: "How do I build a model?"
The assistant will:
1. Identify the goal (build model from sequence)
2. Match to workflow template
3. Provide step-by-step commands
4. List prerequisites
5. Include notes and tips

### Question: "foldhunter gave me low correlation"
The assistant will:
1. Identify the tool (foldhunter)
2. Match error pattern (low correlation)
3. Provide specific solutions
4. Consider resolution context if provided
5. Give parameter recommendations

### Question: "What does the threshold parameter do?"
The assistant will:
1. Identify the tool and parameter
2. Explain the parameter's purpose
3. Provide typical value ranges
4. Give resolution-specific guidance

## Architecture

```
crymodel/assistant/
├── __init__.py
├── knowledge_base.py    # Structured knowledge (tools, workflows, errors)
└── assistant.py          # Rule-based assistant logic

crymodel/cli/
└── assistant.py          # CLI interface
```

## Extending the Knowledge Base

To add new information:

1. **Add tool description** in `knowledge_base.py`:
```python
"newtool": {
    "purpose": "...",
    "description": "...",
    "inputs": {...},
    "outputs": {...},
    "common_use_cases": [...],
    "parameter_guidance": {...},
    "common_errors": {...}
}
```

2. **Add workflow template**:
```python
"new_workflow": {
    "description": "...",
    "prerequisites": [...],
    "steps": [...]
}
```

3. **Add error pattern**:
```python
"new_error": {
    "patterns": [...],
    "solutions": [...],
    "prevention": "..."
}
```

## Future Enhancements

The current implementation is rule-based. Future enhancements could include:

1. **LLM Integration** - Connect to OpenAI/Anthropic for more natural language understanding
2. **RAG System** - Retrieve relevant documentation and examples
3. **Learning from Interactions** - Improve responses based on user feedback
4. **Interactive Mode** - Chat-like interface for extended conversations
5. **Automatic Workflow Generation** - Create custom workflows based on user files and goals
6. **Quality Assessment** - Automatically assess results and suggest improvements

## Integration with Workflows

The assistant can help users create workflow files:

```bash
# Get workflow suggestion
crymodel assistant suggest "build model" --files sequence.fasta,map.mrc

# User creates workflow.yaml based on suggestion
# Execute workflow
crymodel workflow run workflow.yaml
```

## Best Practices

1. **Be specific** - More specific questions get better answers
2. **Provide context** - Use --tool, --resolution flags when relevant
3. **Check prerequisites** - Use `suggest` command to verify you have required files
4. **Follow workflows** - Use suggested workflows as starting points
5. **Validate inputs** - Always check inputs with fitprep before running tools

## Troubleshooting the Assistant

If the assistant doesn't understand your question:

1. Try rephrasing with keywords like "how do I", "error", "explain"
2. Specify the tool name with --tool flag
3. Provide context file with --context flag
4. Use more specific commands (explain, troubleshoot, suggest)

## Contributing

To improve the assistant:

1. Add common error patterns you encounter
2. Document tool-specific solutions
3. Add workflow templates for new use cases
4. Improve parameter guidance based on experience
5. Add resolution-specific recommendations

