"""
Learning Agents - Reinforcement Learning
========================================

Q-Learning and Deep Q-Network agents that learn optimal policies through
experience and reward optimization.

Author: Brill Consulting
"""

import numpy as np
from typing import Dict, Tuple, List, Optional
from collections import defaultdict, deque
import random


class QLearningAgent:
    """Tabular Q-Learning agent."""

    def __init__(self,
                 state_space_size: int,
                 action_space_size: int,
                 learning_rate: float = 0.1,
                 discount_factor: float = 0.95,
                 epsilon: float = 0.1,
                 epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01):
        """
        Initialize Q-Learning agent.

        Args:
            state_space_size: Number of possible states
            action_space_size: Number of possible actions
            learning_rate: Alpha parameter (0-1)
            discount_factor: Gamma parameter (0-1)
            epsilon: Exploration rate
            epsilon_decay: Decay rate for epsilon
            epsilon_min: Minimum epsilon value
        """
        self.state_space = state_space_size
        self.action_space = action_space_size
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Q-table: Q(s, a)
        self.q_table = defaultdict(lambda: np.zeros(action_space_size))

        # Statistics
        self.episode_rewards = []
        self.total_steps = 0

    def select_action(self, state: int, greedy: bool = False) -> int:
        """
        Select action using epsilon-greedy policy.

        Args:
            state: Current state
            greedy: If True, always choose best action

        Returns:
            Selected action
        """
        if not greedy and random.random() < self.epsilon:
            # Explore: random action
            return random.randint(0, self.action_space - 1)
        else:
            # Exploit: best known action
            return int(np.argmax(self.q_table[state]))

    def update(self, state: int, action: int, reward: float,
               next_state: int, done: bool):
        """
        Update Q-value using Q-learning rule.

        Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
        """
        current_q = self.q_table[state][action]

        if done:
            target_q = reward
        else:
            max_next_q = np.max(self.q_table[next_state])
            target_q = reward + self.gamma * max_next_q

        # Update Q-value
        self.q_table[state][action] = current_q + self.alpha * (target_q - current_q)

        self.total_steps += 1

    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_policy(self) -> Dict[int, int]:
        """Get current policy (best action for each state)."""
        policy = {}
        for state in self.q_table.keys():
            policy[state] = int(np.argmax(self.q_table[state]))
        return policy


class ExperienceReplay:
    """Experience replay buffer for DQN."""

    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)

    def add(self, state: np.ndarray, action: int, reward: float,
            next_state: np.ndarray, done: bool):
        """Add experience to buffer."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> List[Tuple]:
        """Sample random batch of experiences."""
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def size(self) -> int:
        """Get current buffer size."""
        return len(self.buffer)


class SARSAAgent:
    """SARSA (State-Action-Reward-State-Action) agent."""

    def __init__(self, state_space_size: int, action_space_size: int,
                 learning_rate: float = 0.1, discount_factor: float = 0.95,
                 epsilon: float = 0.1):
        self.state_space = state_space_size
        self.action_space = action_space_size
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.q_table = defaultdict(lambda: np.zeros(action_space_size))

    def select_action(self, state: int) -> int:
        """Select action using epsilon-greedy."""
        if random.random() < self.epsilon:
            return random.randint(0, self.action_space - 1)
        return int(np.argmax(self.q_table[state]))

    def update(self, state: int, action: int, reward: float,
               next_state: int, next_action: int, done: bool):
        """
        Update using SARSA rule.

        Q(s,a) ← Q(s,a) + α[r + γQ(s',a') - Q(s,a)]
        """
        current_q = self.q_table[state][action]

        if done:
            target_q = reward
        else:
            next_q = self.q_table[next_state][next_action]
            target_q = reward + self.gamma * next_q

        self.q_table[state][action] = current_q + self.alpha * (target_q - current_q)


class Environment:
    """Simple grid world environment for testing."""

    def __init__(self, size: int = 5):
        self.size = size
        self.state_space = size * size
        self.action_space = 4  # up, down, left, right
        self.goal = (size - 1, size - 1)
        self.current_pos = (0, 0)

    def reset(self) -> int:
        """Reset environment to start state."""
        self.current_pos = (0, 0)
        return self._pos_to_state(self.current_pos)

    def step(self, action: int) -> Tuple[int, float, bool]:
        """
        Take action in environment.

        Returns:
            (next_state, reward, done)
        """
        x, y = self.current_pos

        # Actions: 0=up, 1=down, 2=left, 3=right
        if action == 0:  # up
            y = max(0, y - 1)
        elif action == 1:  # down
            y = min(self.size - 1, y + 1)
        elif action == 2:  # left
            x = max(0, x - 1)
        elif action == 3:  # right
            x = min(self.size - 1, x + 1)

        self.current_pos = (x, y)

        # Reward structure
        if self.current_pos == self.goal:
            reward = 10.0
            done = True
        else:
            reward = -0.1  # Small penalty for each step
            done = False

        next_state = self._pos_to_state(self.current_pos)
        return next_state, reward, done

    def _pos_to_state(self, pos: Tuple[int, int]) -> int:
        """Convert position to state number."""
        x, y = pos
        return y * self.size + x


def demo():
    """Demonstration of Learning Agents."""
    print("Learning Agents - Reinforcement Learning Demo")
    print("=" * 70)

    # Create environment
    env = Environment(size=5)

    # Create Q-Learning agent
    agent = QLearningAgent(
        state_space_size=env.state_space,
        action_space_size=env.action_space,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=0.2
    )

    print("\n1️⃣  Training Q-Learning Agent")
    print("-" * 70)

    # Training loop
    num_episodes = 500
    episode_rewards = []

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        steps = 0

        while not done and steps < 100:
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state, done)

            episode_reward += reward
            state = next_state
            steps += 1

        episode_rewards.append(episode_reward)
        agent.decay_epsilon()

        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episode {episode + 1:3d} | "
                  f"Avg Reward: {avg_reward:6.2f} | "
                  f"Epsilon: {agent.epsilon:.3f}")

    print("\n2️⃣  Testing Learned Policy")
    print("-" * 70)

    # Test learned policy
    state = env.reset()
    path = [(0, 0)]
    done = False
    total_reward = 0
    steps = 0

    while not done and steps < 50:
        action = agent.select_action(state, greedy=True)
        next_state, reward, done = env.step(action)
        path.append(env.current_pos)
        total_reward += reward
        state = next_state
        steps += 1

    print(f"Path taken: {' → '.join([f'({x},{y})' for x, y in path[:10]])}")
    print(f"Total steps: {steps}")
    print(f"Total reward: {total_reward:.2f}")

    print("\n3️⃣  Learned Q-Values (sample)")
    print("-" * 70)

    policy = agent.get_policy()
    action_names = ["↑", "↓", "←", "→"]

    print("State | Best Action | Q-Values")
    print("-" * 40)
    for state in sorted(list(policy.keys()))[:5]:
        best_action = policy[state]
        q_values = agent.q_table[state]
        print(f"  {state:2d}  |      {action_names[best_action]}      | "
              f"{' '.join([f'{q:5.2f}' for q in q_values])}")

    print("\n4️⃣  Training Statistics")
    print("-" * 70)

    print(f"Total steps taken: {agent.total_steps:,}")
    print(f"Final epsilon: {agent.epsilon:.4f}")
    print(f"States explored: {len(agent.q_table)}")
    print(f"Average reward (last 100 episodes): {np.mean(episode_rewards[-100:]):.2f}")

    print("\n" + "=" * 70)
    print("✓ Demo Complete!")


if __name__ == '__main__':
    demo()
