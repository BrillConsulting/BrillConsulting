# 🎮 Reinforcement Learning

Advanced reinforcement learning algorithms for sequential decision-making and control tasks.

## 🌟 Algorithms

### Value-Based Methods
1. **Q-Learning** - Off-policy TD control algorithm
2. **Deep Q-Network (DQN)** - Neural network-based Q-learning
3. **Double DQN** - Reduces overestimation bias
4. **Dueling DQN** - Separate value and advantage streams

### Policy-Based Methods
5. **REINFORCE** - Monte Carlo policy gradient
6. **Actor-Critic** - Combines value and policy methods
7. **A2C/A3C** - Advantage Actor-Critic (sync/async)
8. **PPO** - Proximal Policy Optimization

### Model-Based Methods
9. **Dyna-Q** - Combines planning and learning
10. **MCTS** - Monte Carlo Tree Search

## ✨ Key Features

- **Multiple Algorithm Types** - Value, policy, and model-based methods
- **Environment Support** - OpenAI Gym integration
- **Experience Replay** - Improves sample efficiency
- **Target Networks** - Stabilizes learning
- **Epsilon-Greedy** - Exploration strategy
- **Reward Tracking** - Performance visualization
- **Model Persistence** - Save/load trained agents

## 🚀 Quick Start

### Basic Q-Learning

```bash
python reinforcementlearning.py --env CartPole-v1 --algorithm q-learning --episodes 1000
```

### Deep Q-Network (DQN)

```bash
python reinforcementlearning.py --env CartPole-v1 --algorithm dqn --episodes 500 --render
```

## 📊 Example Code

```python
from reinforcementlearning import RLAgent
import gym

# Create environment
env = gym.make('CartPole-v1')

# Initialize Q-Learning agent
agent = RLAgent(
    env=env,
    algorithm='q-learning',
    learning_rate=0.1,
    discount_factor=0.99,
    epsilon=1.0,
    epsilon_decay=0.995
)

# Train agent
rewards = agent.train(n_episodes=1000)

# Evaluate
mean_reward = agent.evaluate(n_episodes=100)
print(f"Mean reward: {mean_reward:.2f}")

# Save model
agent.save('cartpole_agent.pkl')
```

## 🎯 Use Cases

### 🤖 Robotics
- **Robot Control**: Arm manipulation, locomotion
- **Navigation**: Path planning, obstacle avoidance
- **Grasping**: Object manipulation strategies

### 🎮 Gaming
- **Game AI**: Board games (Chess, Go), video games
- **Player Modeling**: Adaptive difficulty
- **NPC Behavior**: Intelligent opponents

### 🏭 Industrial Control
- **Process Optimization**: Manufacturing efficiency
- **Resource Allocation**: Scheduling, planning
- **Energy Management**: Grid optimization

### 💼 Business Applications
- **Trading**: Algorithmic trading strategies
- **Recommendation**: Personalized content
- **Bidding**: Ad placement optimization

### 🚗 Autonomous Systems
- **Self-Driving**: Vehicle control policies
- **Drones**: Flight path optimization
- **Traffic Control**: Signal optimization

## 📚 Algorithm Details

### Q-Learning
**Type**: Off-policy, value-based
**Best for**: Discrete state/action spaces
**Pros**: Simple, guaranteed convergence
**Cons**: Doesn't scale to large state spaces

```python
agent = RLAgent(env, algorithm='q-learning')
agent.train(n_episodes=1000)
```

### Deep Q-Network (DQN)
**Type**: Off-policy, deep RL
**Best for**: High-dimensional state spaces
**Pros**: Handles complex environments, experience replay
**Cons**: Sample inefficient, can be unstable

**Key innovations**:
- Experience replay buffer
- Target network for stability
- Epsilon-greedy exploration

```python
agent = RLAgent(
    env,
    algorithm='dqn',
    buffer_size=10000,
    batch_size=32,
    target_update_freq=100
)
```

### Policy Gradient (REINFORCE)
**Type**: On-policy, policy-based
**Best for**: Continuous actions, stochastic policies
**Pros**: Can learn stochastic policies, works in continuous spaces
**Cons**: High variance, sample inefficient

```python
agent = RLAgent(env, algorithm='reinforce')
```

### Actor-Critic
**Type**: On-policy, hybrid
**Best for**: Balancing variance and bias
**Pros**: Lower variance than policy gradient
**Cons**: More complex, requires tuning

### Proximal Policy Optimization (PPO)
**Type**: On-policy, policy-based
**Best for**: Continuous control, stability
**Pros**: Stable, sample efficient, easy to tune
**Cons**: Requires multiple epochs per update

## 🔧 Configuration

### Learning Parameters

```python
agent = RLAgent(
    env=env,
    algorithm='dqn',
    learning_rate=0.001,      # Step size for updates
    discount_factor=0.99,      # Future reward importance (gamma)
    epsilon=1.0,               # Initial exploration rate
    epsilon_decay=0.995,       # Exploration decay
    epsilon_min=0.01           # Minimum exploration
)
```

### Network Architecture (DQN)

```python
network_config = {
    'hidden_layers': [128, 128],  # Two hidden layers
    'activation': 'relu',
    'optimizer': 'adam'
}
agent = RLAgent(env, algorithm='dqn', network_config=network_config)
```

### Experience Replay

```python
replay_config = {
    'buffer_size': 100000,
    'batch_size': 64,
    'min_buffer_size': 1000
}
agent = RLAgent(env, algorithm='dqn', replay_config=replay_config)
```

## 📊 Training & Evaluation

### Training Loop

```python
# Train with progress tracking
rewards = agent.train(
    n_episodes=1000,
    render=False,
    save_freq=100,        # Save every 100 episodes
    eval_freq=50,         # Evaluate every 50 episodes
    verbose=True
)

# Plot learning curve
agent.plot_learning_curve(rewards)
```

### Evaluation

```python
# Evaluate trained agent
mean_reward, std_reward = agent.evaluate(
    n_episodes=100,
    render=True
)
print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
```

### Visualization

```python
# Visualize agent performance
agent.visualize_episode(render=True)

# Plot Q-values (for DQN)
agent.plot_q_values()

# Plot policy (for policy gradient)
agent.plot_policy()
```

## 💡 Best Practices

### 1. **Start Simple**
- Begin with Q-learning for discrete environments
- Move to DQN for complex state spaces
- Use PPO for continuous control

### 2. **Hyperparameter Tuning**
- Learning rate: Start with 0.001 for neural networks
- Discount factor: 0.99 for long-term rewards, 0.9 for short-term
- Exploration: High initial epsilon (1.0), slow decay

### 3. **Stabilize Training**
- Use experience replay (DQN)
- Implement target networks
- Clip gradients
- Normalize rewards

### 4. **Monitor Progress**
- Track episode rewards
- Log Q-values (value-based)
- Monitor policy entropy (policy-based)
- Evaluate periodically

### 5. **Environment Design**
- Shape rewards carefully
- Normalize observations
- Clip actions if continuous
- Use curriculum learning for complex tasks

## 🐛 Troubleshooting

**Agent not learning?**
- Check learning rate (too high/low)
- Verify reward function is informative
- Increase exploration (epsilon)
- Check environment is solvable

**Training unstable?**
- Reduce learning rate
- Use target networks (DQN)
- Normalize inputs/rewards
- Clip gradients
- Increase batch size

**Poor generalization?**
- Add experience replay
- Increase network capacity
- Add regularization (dropout, L2)
- Train longer

**Slow convergence?**
- Increase learning rate carefully
- Use better optimizer (Adam)
- Tune discount factor
- Improve reward shaping

**Overestimation (DQN)?**
- Use Double DQN
- Reduce target network update frequency
- Use softer target updates

## 📈 Algorithm Comparison

| Algorithm | Type | Sample Efficiency | Stability | Continuous Actions | Best For |
|-----------|------|-------------------|-----------|-------------------|----------|
| **Q-Learning** | Value | Low | High | ❌ | Simple discrete |
| **DQN** | Value | Medium | Medium | ❌ | Complex discrete |
| **REINFORCE** | Policy | Low | Low | ✅ | Stochastic policies |
| **Actor-Critic** | Hybrid | Medium | Medium | ✅ | General purpose |
| **PPO** | Policy | High | High | ✅ | Continuous control |
| **A3C** | Hybrid | High | High | ✅ | Parallel training |
| **DDPG** | Value | High | Medium | ✅ | Continuous control |
| **SAC** | Value | Very High | High | ✅ | Sample efficiency |

## 🎓 Key Concepts

### Markov Decision Process (MDP)
- **States (S)**: Environment configurations
- **Actions (A)**: Agent's choices
- **Rewards (R)**: Feedback signal
- **Transitions (P)**: State dynamics
- **Policy (π)**: Action selection strategy

### Value Functions
- **V(s)**: Expected return from state s
- **Q(s,a)**: Expected return from state s, action a
- **Advantage A(s,a)**: Q(s,a) - V(s)

### Bellman Equation
```
Q(s,a) = R(s,a) + γ * max_a' Q(s',a')
```

### Exploration vs Exploitation
- **Epsilon-greedy**: Random action with probability ε
- **Boltzmann**: Softmax action selection
- **UCB**: Upper confidence bound

## 🛠️ Environments

### OpenAI Gym Classics
- **CartPole-v1**: Balance pole on cart
- **MountainCar-v0**: Reach goal on hill
- **LunarLander-v2**: Land spacecraft
- **Acrobot-v1**: Swing up acrobat

### Atari Games
- **Breakout**: Brick breaking
- **Pong**: Table tennis
- **Space Invaders**: Arcade shooter

### Robotics (Mujoco)
- **HalfCheetah**: Locomotion
- **Walker2d**: Bipedal walking
- **Ant**: Quadruped locomotion

## 📄 Dependencies

```bash
pip install gym numpy matplotlib torch

# Optional: Atari environments
pip install gym[atari]

# Optional: Mujoco physics
pip install gym[mujoco]

# Optional: Stable-Baselines3
pip install stable-baselines3
```

## 🏆 Status

**Version:** 1.0
**Status:** Research/Educational

**Features:**
- ✅ Q-Learning Implementation
- ✅ Deep Q-Network (DQN)
- ✅ Policy Gradient Methods
- ✅ Experience Replay
- ✅ Target Networks
- ✅ Gym Environment Support
- ✅ Visualization Tools
- ⚠️ Advanced algorithms (PPO, SAC) - Planned

## 📞 Support

**Author**: BrillConsulting | AI Consultant & Data Scientist
**Email**: clientbrill@gmail.com
**LinkedIn**: [BrillConsulting](https://www.linkedin.com/in/brillconsulting)

---

**⭐ Star this repository if you find it useful!**

*Made with ❤️ by BrillConsulting*
