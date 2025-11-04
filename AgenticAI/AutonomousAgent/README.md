# Autonomous Agent Framework

Self-directed agent with planning, reasoning, and execution capabilities.

## Features

- Goal-oriented planning
- Dynamic task decomposition
- Action execution with tool integration
- Self-reflection and learning
- Error recovery and retry mechanisms
- Execution history tracking

## Usage

```python
from autonomous_agent import AutonomousAgent

# Create agent
agent = AutonomousAgent(name="AutoBot")

# Register tools
agent.add_tool("analyze_data", analyze_function, "Analyze dataset")

# Run autonomous execution
result = agent.run(
    goal="Analyze customer data and deliver insights",
    context={"dataset": "customers.csv"}
)

# Check status
status = agent.get_status()
```

## Demo

```bash
python autonomous_agent.py
```
