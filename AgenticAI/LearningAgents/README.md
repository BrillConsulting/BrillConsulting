# Learning Agents

Reinforcement learning agents that improve performance through experience and reward optimization.

## Features

- **Q-Learning**: Value-based reinforcement learning
- **Experience Replay**: Store and reuse past experiences
- **Reward Optimization**: Maximize cumulative rewards
- **Policy Improvement**: Iterative policy enhancement
- **Transfer Learning**: Apply knowledge to new tasks
- **Exploration vs Exploitation**: Balance learning strategies

## Quick Start

```python
from learning_agents import QLearningAgent, Environment

# Create environment
env = Environment(state_space=10, action_space=4)

# Create agent
agent = QLearningAgent(
    state_space=10,
    action_space=4,
    learning_rate=0.1,
    discount_factor=0.95,
    epsilon=0.1
)

# Training loop
for episode in range(1000):
    state = env.reset()
    done = False
    
    while not done:
        action = agent.select_action(state)
        next_state, reward, done = env.step(action)
        agent.update(state, action, reward, next_state)
        state = next_state

# Use learned policy
state = env.reset()
action = agent.select_action(state, greedy=True)
```

## Use Cases

- **Game Playing**: Learn optimal strategies
- **Robot Control**: Adaptive behavior learning
- **Resource Management**: Optimize allocation decisions

## Author

Brill Consulting
