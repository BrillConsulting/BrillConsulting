# Reactive Agents

Fast, event-driven agents with condition-action rules and behavior trees for real-time decision making.

## Features

- **Condition-Action Rules**: Immediate stimulus-response
- **Sensor Processing**: Real-time environment perception
- **Behavior Trees**: Hierarchical behavior modeling
- **Subsumption Architecture**: Layered reactive behaviors
- **Priority-based Actions**: Handle multiple stimuli
- **Fast Response**: Minimal deliberation overhead

## Quick Start

```python
from reactive_agents import ReactiveAgent, Rule

# Create agent
agent = ReactiveAgent()

# Define rules
agent.add_rule(Rule(
    condition=lambda state: state["obstacle_detected"],
    action=lambda: agent.stop_and_turn()
))

agent.add_rule(Rule(
    condition=lambda state: state["target_visible"],
    action=lambda: agent.move_toward_target(),
    priority=10
))

# Run agent loop
while True:
    state = agent.perceive_environment()
    agent.react(state)
```

## Use Cases

- **Robot Control**: Real-time obstacle avoidance
- **Game AI**: Fast NPC reactions
- **Industrial Control**: Process monitoring

## Author

Brill Consulting
