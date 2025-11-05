# Planning Agents

Advanced planning agents with hierarchical task networks and goal-oriented reasoning.

## Features

- **Hierarchical Planning**: Decompose complex plans
- **Goal Decomposition**: Break down high-level objectives
- **Plan Optimization**: Find efficient action sequences
- **Constraint Satisfaction**: Honor planning constraints
- **Execution Monitoring**: Track plan progress
- **Replanning**: Adapt when plans fail

## Quick Start

```python
from planning_agents import PlanningAgent, Goal, Action

# Create agent
agent = PlanningAgent()

# Define goal
goal = Goal(
    description="deliver_package",
    conditions={"package_at": "destination"}
)

# Available actions
actions = [
    Action("pickup", precond={"at": "warehouse"}, effect={"holding": "package"}),
    Action("drive", precond={"holding": "package"}, effect={"at": "destination"}),
    Action("deliver", precond={"at": "destination", "holding": "package"}, 
           effect={"package_at": "destination"})
]

# Generate plan
plan = agent.plan(
    initial_state={"at": "warehouse"},
    goal=goal,
    actions=actions
)

# Execute plan
for action in plan:
    agent.execute(action)
```

## Use Cases

- **Autonomous Robots**: Navigate and manipulate objects
- **Logistics**: Route and delivery planning
- **Manufacturing**: Assembly sequence planning

## Author

Brill Consulting
