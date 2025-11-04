# Agent Orchestration Framework

Orchestrate complex agent workflows and execution patterns.

## Features

- Workflow definition (sequential, parallel, conditional)
- Agent execution management
- State management across workflow steps
- Error handling and automatic retries
- Workflow monitoring and history
- Conditional execution based on context

## Usage

```python
from agent_orchestrator import AgentOrchestrator, WorkflowStep, ExecutionMode

orchestrator = AgentOrchestrator()

# Create workflow
workflow = orchestrator.create_workflow(
    "workflow_1",
    "Data Pipeline",
    ExecutionMode.SEQUENTIAL
)

# Add steps
workflow.add_step(WorkflowStep("analyze", "agent_1", analyze_function))
workflow.add_step(WorkflowStep("report", "agent_2", report_function))

# Execute workflow
result = orchestrator.execute_workflow("workflow_1")

# Check status
status = orchestrator.get_workflow_status("workflow_1")
```

## Execution Modes

- **Sequential**: Steps execute one after another
- **Parallel**: Steps execute simultaneously
- **Conditional**: Steps execute based on conditions

## Demo

```bash
python agent_orchestrator.py
```
