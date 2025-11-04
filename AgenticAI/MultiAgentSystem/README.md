# Multi-Agent System

Coordinated multi-agent collaboration framework.

## Features

- Agent communication protocols
- Task distribution and load balancing
- Collaborative problem solving
- Consensus mechanisms
- Role-based agent coordination
- Message routing and broadcasting

## Usage

```python
from multi_agent_system import MultiAgentSystem, AgentRole

# Create system
mas = MultiAgentSystem(name="CollaborativeAI")

# Create agents
coordinator = mas.create_agent("coord_1", AgentRole.COORDINATOR, ["coordination"])
worker = mas.create_agent("worker_1", AgentRole.WORKER, ["data_processing"])

# Send messages
mas.send_message("coord_1", "worker_1", {"request": "status"})

# Distribute tasks
tasks = [
    {"id": 1, "type": "data_processing", "description": "Process dataset"}
]
mas.distribute_tasks(tasks)

# Reach consensus
result = mas.consensus("Select strategy", ["option_a", "option_b"])
```

## Demo

```bash
python multi_agent_system.py
```
