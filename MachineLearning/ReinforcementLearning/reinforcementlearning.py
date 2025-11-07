"""
ReinforcementLearning v2.0
Author: BrillConsulting
Description: Advanced Reinforcement Learning with Q-Learning, DQN, Policy Gradients, and Actor-Critic
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from collections import deque, defaultdict
import pickle
import warnings
warnings.filterwarnings('ignore')

class RLEnvironment:
    """Simple Grid World Environment for RL algorithms"""

    def __init__(self, grid_size: int = 5, obstacles: Optional[List[Tuple[int, int]]] = None):
        self.grid_size = grid_size
        self.obstacles = obstacles or [(1, 1), (2, 2), (3, 1)]
        self.goal = (grid_size - 1, grid_size - 1)
        self.state = (0, 0)
        self.num_states = grid_size * grid_size
        self.num_actions = 4  # up, down, left, right

    def reset(self) -> int:
        """Reset environment to start state"""
        self.state = (0, 0)
        return self._state_to_index(self.state)

    def _state_to_index(self, state: Tuple[int, int]) -> int:
        """Convert (row, col) to state index"""
        return state[0] * self.grid_size + state[1]

    def _index_to_state(self, index: int) -> Tuple[int, int]:
        """Convert state index to (row, col)"""
        return (index // self.grid_size, index % self.grid_size)

    def step(self, action: int) -> Tuple[int, float, bool, Dict]:
        """
        Execute action and return (next_state, reward, done, info)
        Actions: 0=up, 1=down, 2=left, 3=right
        """
        row, col = self.state

        # Apply action
        if action == 0:  # up
            row = max(0, row - 1)
        elif action == 1:  # down
            row = min(self.grid_size - 1, row + 1)
        elif action == 2:  # left
            col = max(0, col - 1)
        elif action == 3:  # right
            col = min(self.grid_size - 1, col + 1)

        new_state = (row, col)

        # Check for obstacles
        if new_state in self.obstacles:
            new_state = self.state  # Stay in place
            reward = -10.0
        elif new_state == self.goal:
            reward = 100.0
        else:
            reward = -1.0  # Small penalty for each step

        self.state = new_state
        done = (new_state == self.goal)

        return self._state_to_index(new_state), reward, done, {}

    def render(self):
        """Visualize current state"""
        grid = np.zeros((self.grid_size, self.grid_size))
        for obs in self.obstacles:
            grid[obs] = -1
        grid[self.goal] = 2
        grid[self.state] = 1
        return grid


class QLearningAgent:
    """Q-Learning Agent with epsilon-greedy exploration"""

    def __init__(self, num_states: int, num_actions: int,
                 learning_rate: float = 0.1, gamma: float = 0.99,
                 epsilon: float = 1.0, epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01):
        self.num_states = num_states
        self.num_actions = num_actions
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Initialize Q-table
        self.q_table = np.zeros((num_states, num_actions))

    def get_action(self, state: int, training: bool = True) -> int:
        """Epsilon-greedy action selection"""
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.num_actions)
        return np.argmax(self.q_table[state])

    def update(self, state: int, action: int, reward: float,
               next_state: int, done: bool):
        """Q-Learning update rule"""
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q_table[next_state])

        self.q_table[state, action] += self.lr * (target - self.q_table[state, action])

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


class DQNAgent:
    """Deep Q-Network Agent with experience replay (simplified version)"""

    def __init__(self, num_states: int, num_actions: int,
                 learning_rate: float = 0.001, gamma: float = 0.99,
                 epsilon: float = 1.0, epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01, memory_size: int = 2000):
        self.num_states = num_states
        self.num_actions = num_actions
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Experience replay memory
        self.memory = deque(maxlen=memory_size)

        # Simplified neural network (using Q-table for demonstration)
        self.q_network = np.random.randn(num_states, num_actions) * 0.01
        self.target_network = self.q_network.copy()
        self.update_counter = 0
        self.target_update_freq = 10

    def get_action(self, state: int, training: bool = True) -> int:
        """Epsilon-greedy action selection"""
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.num_actions)
        return np.argmax(self.q_network[state])

    def remember(self, state: int, action: int, reward: float,
                 next_state: int, done: bool):
        """Store experience in replay memory"""
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size: int = 32):
        """Train on batch of experiences"""
        if len(self.memory) < batch_size:
            return

        # Sample batch
        indices = np.random.choice(len(self.memory), batch_size, replace=False)
        batch = [self.memory[i] for i in indices]

        for state, action, reward, next_state, done in batch:
            if done:
                target = reward
            else:
                target = reward + self.gamma * np.max(self.target_network[next_state])

            # Update Q-network
            self.q_network[state, action] += self.lr * (target - self.q_network[state, action])

        # Update target network periodically
        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.target_network = self.q_network.copy()

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


class PolicyGradientAgent:
    """REINFORCE Policy Gradient Agent (simplified)"""

    def __init__(self, num_states: int, num_actions: int,
                 learning_rate: float = 0.01, gamma: float = 0.99):
        self.num_states = num_states
        self.num_actions = num_actions
        self.lr = learning_rate
        self.gamma = gamma

        # Policy parameters (softmax over actions)
        self.theta = np.random.randn(num_states, num_actions) * 0.01

        # Episode memory
        self.states = []
        self.actions = []
        self.rewards = []

    def get_action(self, state: int, training: bool = True) -> int:
        """Sample action from policy"""
        logits = self.theta[state]
        probs = self._softmax(logits)
        action = np.random.choice(self.num_actions, p=probs)

        if training:
            self.states.append(state)
            self.actions.append(action)

        return action

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Compute softmax probabilities"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()

    def store_reward(self, reward: float):
        """Store reward for current step"""
        self.rewards.append(reward)

    def update(self):
        """Update policy using REINFORCE algorithm"""
        if len(self.rewards) == 0:
            return

        # Compute discounted returns
        returns = []
        G = 0
        for r in reversed(self.rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = np.array(returns)

        # Normalize returns
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Update policy parameters
        for state, action, G in zip(self.states, self.actions, returns):
            probs = self._softmax(self.theta[state])

            # Gradient ascent
            for a in range(self.num_actions):
                if a == action:
                    self.theta[state, a] += self.lr * G * (1 - probs[a])
                else:
                    self.theta[state, a] -= self.lr * G * probs[a]

        # Clear episode memory
        self.states = []
        self.actions = []
        self.rewards = []


class ActorCriticAgent:
    """Actor-Critic Agent (simplified)"""

    def __init__(self, num_states: int, num_actions: int,
                 actor_lr: float = 0.01, critic_lr: float = 0.1,
                 gamma: float = 0.99):
        self.num_states = num_states
        self.num_actions = num_actions
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.gamma = gamma

        # Actor: policy parameters
        self.actor = np.random.randn(num_states, num_actions) * 0.01

        # Critic: value function
        self.critic = np.zeros(num_states)

    def get_action(self, state: int, training: bool = True) -> int:
        """Sample action from policy"""
        logits = self.actor[state]
        probs = self._softmax(logits)
        return np.random.choice(self.num_actions, p=probs)

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Compute softmax probabilities"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()

    def update(self, state: int, action: int, reward: float,
               next_state: int, done: bool):
        """Update actor and critic"""
        # TD error (advantage)
        if done:
            td_target = reward
        else:
            td_target = reward + self.gamma * self.critic[next_state]

        td_error = td_target - self.critic[state]

        # Update critic (value function)
        self.critic[state] += self.critic_lr * td_error

        # Update actor (policy)
        probs = self._softmax(self.actor[state])
        for a in range(self.num_actions):
            if a == action:
                self.actor[state, a] += self.actor_lr * td_error * (1 - probs[a])
            else:
                self.actor[state, a] -= self.actor_lr * td_error * probs[a]


class ReinforcementLearningManager:
    """
    Advanced Reinforcement Learning Manager

    Features:
    - Q-Learning with epsilon-greedy exploration
    - Deep Q-Network (DQN) with experience replay
    - Policy Gradient (REINFORCE)
    - Actor-Critic
    - Grid world environment
    - Training visualization
    - Model persistence
    """

    def __init__(self, grid_size: int = 5):
        self.grid_size = grid_size
        self.env = RLEnvironment(grid_size=grid_size)
        self.agents = {}
        self.training_history = defaultdict(list)

        print(f"🤖 ReinforcementLearning Manager v2.0 initialized")
        print(f"   Grid size: {grid_size}x{grid_size}")
        print(f"   Goal: {self.env.goal}")
        print(f"   Obstacles: {self.env.obstacles}")

    def train_qlearning(self, num_episodes: int = 500, max_steps: int = 100,
                        learning_rate: float = 0.1, gamma: float = 0.99) -> Dict[str, Any]:
        """Train Q-Learning agent"""
        print(f"\n🎯 Training Q-Learning Agent...")
        print(f"   Episodes: {num_episodes}, Max steps: {max_steps}")
        print(f"   Learning rate: {learning_rate}, Gamma: {gamma}")

        agent = QLearningAgent(
            num_states=self.env.num_states,
            num_actions=self.env.num_actions,
            learning_rate=learning_rate,
            gamma=gamma
        )

        episode_rewards = []
        episode_lengths = []

        for episode in range(num_episodes):
            state = self.env.reset()
            total_reward = 0
            steps = 0

            for step in range(max_steps):
                action = agent.get_action(state, training=True)
                next_state, reward, done, _ = self.env.step(action)

                agent.update(state, action, reward, next_state, done)

                total_reward += reward
                state = next_state
                steps += 1

                if done:
                    break

            episode_rewards.append(total_reward)
            episode_lengths.append(steps)

            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                print(f"   Episode {episode + 1}: Avg reward = {avg_reward:.2f}, Epsilon = {agent.epsilon:.4f}")

        self.agents['qlearning'] = agent
        self.training_history['qlearning'] = {
            'rewards': episode_rewards,
            'lengths': episode_lengths,
            'algorithm': 'Q-Learning'
        }

        print(f"✓ Q-Learning training complete!")
        return {
            'final_avg_reward': np.mean(episode_rewards[-100:]),
            'final_epsilon': agent.epsilon,
            'total_episodes': num_episodes
        }

    def train_dqn(self, num_episodes: int = 500, max_steps: int = 100,
                  batch_size: int = 32) -> Dict[str, Any]:
        """Train DQN agent"""
        print(f"\n🎯 Training DQN Agent...")
        print(f"   Episodes: {num_episodes}, Batch size: {batch_size}")

        agent = DQNAgent(
            num_states=self.env.num_states,
            num_actions=self.env.num_actions
        )

        episode_rewards = []
        episode_lengths = []

        for episode in range(num_episodes):
            state = self.env.reset()
            total_reward = 0
            steps = 0

            for step in range(max_steps):
                action = agent.get_action(state, training=True)
                next_state, reward, done, _ = self.env.step(action)

                agent.remember(state, action, reward, next_state, done)
                agent.replay(batch_size=batch_size)

                total_reward += reward
                state = next_state
                steps += 1

                if done:
                    break

            episode_rewards.append(total_reward)
            episode_lengths.append(steps)

            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                print(f"   Episode {episode + 1}: Avg reward = {avg_reward:.2f}, Epsilon = {agent.epsilon:.4f}")

        self.agents['dqn'] = agent
        self.training_history['dqn'] = {
            'rewards': episode_rewards,
            'lengths': episode_lengths,
            'algorithm': 'DQN'
        }

        print(f"✓ DQN training complete!")
        return {
            'final_avg_reward': np.mean(episode_rewards[-100:]),
            'memory_size': len(agent.memory),
            'total_episodes': num_episodes
        }

    def train_policy_gradient(self, num_episodes: int = 500, max_steps: int = 100) -> Dict[str, Any]:
        """Train Policy Gradient agent"""
        print(f"\n🎯 Training Policy Gradient Agent...")
        print(f"   Episodes: {num_episodes}")

        agent = PolicyGradientAgent(
            num_states=self.env.num_states,
            num_actions=self.env.num_actions
        )

        episode_rewards = []
        episode_lengths = []

        for episode in range(num_episodes):
            state = self.env.reset()
            total_reward = 0
            steps = 0

            for step in range(max_steps):
                action = agent.get_action(state, training=True)
                next_state, reward, done, _ = self.env.step(action)

                agent.store_reward(reward)
                total_reward += reward
                state = next_state
                steps += 1

                if done:
                    break

            agent.update()
            episode_rewards.append(total_reward)
            episode_lengths.append(steps)

            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                print(f"   Episode {episode + 1}: Avg reward = {avg_reward:.2f}")

        self.agents['policy_gradient'] = agent
        self.training_history['policy_gradient'] = {
            'rewards': episode_rewards,
            'lengths': episode_lengths,
            'algorithm': 'Policy Gradient'
        }

        print(f"✓ Policy Gradient training complete!")
        return {
            'final_avg_reward': np.mean(episode_rewards[-100:]),
            'total_episodes': num_episodes
        }

    def train_actor_critic(self, num_episodes: int = 500, max_steps: int = 100) -> Dict[str, Any]:
        """Train Actor-Critic agent"""
        print(f"\n🎯 Training Actor-Critic Agent...")
        print(f"   Episodes: {num_episodes}")

        agent = ActorCriticAgent(
            num_states=self.env.num_states,
            num_actions=self.env.num_actions
        )

        episode_rewards = []
        episode_lengths = []

        for episode in range(num_episodes):
            state = self.env.reset()
            total_reward = 0
            steps = 0

            for step in range(max_steps):
                action = agent.get_action(state, training=True)
                next_state, reward, done, _ = self.env.step(action)

                agent.update(state, action, reward, next_state, done)

                total_reward += reward
                state = next_state
                steps += 1

                if done:
                    break

            episode_rewards.append(total_reward)
            episode_lengths.append(steps)

            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                print(f"   Episode {episode + 1}: Avg reward = {avg_reward:.2f}")

        self.agents['actor_critic'] = agent
        self.training_history['actor_critic'] = {
            'rewards': episode_rewards,
            'lengths': episode_lengths,
            'algorithm': 'Actor-Critic'
        }

        print(f"✓ Actor-Critic training complete!")
        return {
            'final_avg_reward': np.mean(episode_rewards[-100:]),
            'total_episodes': num_episodes
        }

    def train_all(self, num_episodes: int = 500):
        """Train all RL algorithms"""
        print(f"\n🚀 Training all RL algorithms...")

        self.train_qlearning(num_episodes=num_episodes)
        self.train_dqn(num_episodes=num_episodes)
        self.train_policy_gradient(num_episodes=num_episodes)
        self.train_actor_critic(num_episodes=num_episodes)

        print(f"\n✓ All algorithms trained!")

    def evaluate_agent(self, agent_name: str, num_episodes: int = 100,
                       max_steps: int = 100) -> Dict[str, Any]:
        """Evaluate trained agent"""
        if agent_name not in self.agents:
            raise ValueError(f"Agent '{agent_name}' not found. Train it first.")

        agent = self.agents[agent_name]
        rewards = []
        lengths = []
        success_count = 0

        for _ in range(num_episodes):
            state = self.env.reset()
            total_reward = 0
            steps = 0

            for step in range(max_steps):
                action = agent.get_action(state, training=False)
                next_state, reward, done, _ = self.env.step(action)

                total_reward += reward
                state = next_state
                steps += 1

                if done:
                    success_count += 1
                    break

            rewards.append(total_reward)
            lengths.append(steps)

        return {
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'mean_length': np.mean(lengths),
            'success_rate': success_count / num_episodes
        }

    def visualize_training(self, figsize: Tuple[int, int] = (15, 10)):
        """Visualize training progress for all algorithms"""
        if not self.training_history:
            print("⚠ No training history available. Train agents first.")
            return

        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('🎮 Reinforcement Learning Training Progress', fontsize=16, fontweight='bold')

        colors = ['blue', 'green', 'red', 'purple']

        # Plot 1: Rewards over time
        ax = axes[0, 0]
        for (name, history), color in zip(self.training_history.items(), colors):
            rewards = history['rewards']
            # Smooth with moving average
            window = 50
            smoothed = pd.Series(rewards).rolling(window=window, min_periods=1).mean()
            ax.plot(smoothed, label=history['algorithm'], color=color, linewidth=2)

        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward (smoothed)')
        ax.set_title('Training Rewards')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: Episode lengths
        ax = axes[0, 1]
        for (name, history), color in zip(self.training_history.items(), colors):
            lengths = history['lengths']
            window = 50
            smoothed = pd.Series(lengths).rolling(window=window, min_periods=1).mean()
            ax.plot(smoothed, label=history['algorithm'], color=color, linewidth=2)

        ax.set_xlabel('Episode')
        ax.set_ylabel('Steps (smoothed)')
        ax.set_title('Episode Lengths')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 3: Final performance comparison
        ax = axes[1, 0]
        algorithms = []
        final_rewards = []

        for name, history in self.training_history.items():
            algorithms.append(history['algorithm'])
            final_rewards.append(np.mean(history['rewards'][-100:]))

        bars = ax.bar(algorithms, final_rewards, color=colors[:len(algorithms)])
        ax.set_ylabel('Average Reward (last 100 episodes)')
        ax.set_title('Final Performance Comparison')
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}',
                   ha='center', va='bottom')

        # Plot 4: Convergence speed
        ax = axes[1, 1]
        for (name, history), color in zip(self.training_history.items(), colors):
            rewards = history['rewards']
            # Calculate cumulative mean
            cumulative_mean = pd.Series(rewards).expanding().mean()
            ax.plot(cumulative_mean, label=history['algorithm'], color=color, linewidth=2)

        ax.set_xlabel('Episode')
        ax.set_ylabel('Cumulative Mean Reward')
        ax.set_title('Convergence Speed')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('rl_training_progress.png', dpi=300, bbox_inches='tight')
        print(f"✓ Training visualization saved: rl_training_progress.png")
        plt.close()

    def visualize_policy(self, agent_name: str = 'qlearning'):
        """Visualize learned policy on grid"""
        if agent_name not in self.agents:
            raise ValueError(f"Agent '{agent_name}' not found")

        agent = self.agents[agent_name]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Plot 1: Value function
        if hasattr(agent, 'q_table'):
            values = np.max(agent.q_table, axis=1).reshape(self.grid_size, self.grid_size)
        elif hasattr(agent, 'critic'):
            values = agent.critic.reshape(self.grid_size, self.grid_size)
        else:
            values = np.zeros((self.grid_size, self.grid_size))

        im1 = ax1.imshow(values, cmap='RdYlGn', interpolation='nearest')
        ax1.set_title(f'State Values - {agent_name.upper()}')

        # Mark special states
        for obs in self.env.obstacles:
            ax1.plot(obs[1], obs[0], 'kx', markersize=20, markeredgewidth=3)
        ax1.plot(self.env.goal[1], self.env.goal[0], 'g*', markersize=20)
        ax1.plot(0, 0, 'ro', markersize=15)

        plt.colorbar(im1, ax=ax1)

        # Plot 2: Policy arrows
        ax2.set_xlim(-0.5, self.grid_size - 0.5)
        ax2.set_ylim(-0.5, self.grid_size - 0.5)
        ax2.set_aspect('equal')
        ax2.invert_yaxis()

        arrow_map = {0: (0, -0.3), 1: (0, 0.3), 2: (-0.3, 0), 3: (0.3, 0)}

        for row in range(self.grid_size):
            for col in range(self.grid_size):
                if (row, col) in self.env.obstacles:
                    ax2.add_patch(plt.Rectangle((col-0.4, row-0.4), 0.8, 0.8,
                                                color='black', alpha=0.3))
                    continue

                if (row, col) == self.env.goal:
                    ax2.plot(col, row, 'g*', markersize=30)
                    continue

                state_idx = row * self.grid_size + col
                action = agent.get_action(state_idx, training=False)
                dx, dy = arrow_map[action]

                ax2.arrow(col, row, dx, dy, head_width=0.2, head_length=0.15,
                         fc='blue', ec='blue', alpha=0.7)

        ax2.set_title(f'Learned Policy - {agent_name.upper()}')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'rl_policy_{agent_name}.png', dpi=300, bbox_inches='tight')
        print(f"✓ Policy visualization saved: rl_policy_{agent_name}.png")
        plt.close()

    def save_agents(self, filepath: str = 'rl_agents.pkl'):
        """Save all trained agents"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'agents': self.agents,
                'training_history': dict(self.training_history),
                'grid_size': self.grid_size
            }, f)
        print(f"✓ Agents saved to {filepath}")

    def load_agents(self, filepath: str = 'rl_agents.pkl'):
        """Load trained agents"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        self.agents = data['agents']
        self.training_history = defaultdict(list, data['training_history'])
        self.grid_size = data['grid_size']
        self.env = RLEnvironment(grid_size=self.grid_size)

        print(f"✓ Agents loaded from {filepath}")
        print(f"   Loaded algorithms: {list(self.agents.keys())}")

    def get_summary(self) -> pd.DataFrame:
        """Get summary of all trained algorithms"""
        if not self.training_history:
            print("⚠ No training history available")
            return pd.DataFrame()

        data = []
        for name, history in self.training_history.items():
            rewards = history['rewards']
            lengths = history['lengths']

            data.append({
                'Algorithm': history['algorithm'],
                'Final Avg Reward': np.mean(rewards[-100:]),
                'Best Reward': np.max(rewards),
                'Avg Episode Length': np.mean(lengths[-100:]),
                'Total Episodes': len(rewards)
            })

        df = pd.DataFrame(data)
        return df


def main():
    """Example usage"""
    print("=" * 60)
    print("🎮 Reinforcement Learning v2.0 - Demo")
    print("=" * 60)

    # Create manager
    manager = ReinforcementLearningManager(grid_size=5)

    # Train all algorithms
    manager.train_all(num_episodes=300)

    # Visualize results
    manager.visualize_training()

    # Evaluate agents
    print("\n📊 Evaluation Results:")
    for agent_name in manager.agents.keys():
        results = manager.evaluate_agent(agent_name, num_episodes=100)
        print(f"\n{agent_name.upper()}:")
        print(f"   Mean reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
        print(f"   Success rate: {results['success_rate']:.2%}")
        print(f"   Avg steps: {results['mean_length']:.1f}")

    # Visualize policies
    for agent_name in ['qlearning', 'dqn']:
        manager.visualize_policy(agent_name)

    # Get summary
    print("\n📋 Training Summary:")
    print(manager.get_summary().to_string(index=False))

    # Save agents
    manager.save_agents()

    print("\n✓ Demo complete!")


if __name__ == "__main__":
    main()
