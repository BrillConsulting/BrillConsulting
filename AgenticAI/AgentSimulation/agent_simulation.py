"""
Agent Simulation - Multi-Agent Modeling Framework
==================================================

Agent-based modeling framework for simulating complex systems with
discrete event simulation and batch execution.

Author: Brill Consulting
"""

from typing import List, Dict, Optional, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import random


class Agent(ABC):
    """Base agent for simulations."""

    def __init__(self, unique_id: int, model: 'Model'):
        self.unique_id = unique_id
        self.model = model

    @abstractmethod
    def step(self):
        """Execute one step of agent behavior."""
        pass


class Model:
    """Base model for agent simulations."""

    def __init__(self, num_agents: int = 10):
        self.num_agents = num_agents
        self.agents: List[Agent] = []
        self.schedule = []
        self.current_step = 0
        self.data_collector = DataCollector()

    def add_agent(self, agent: Agent):
        """Add agent to model."""
        self.agents.append(agent)
        self.schedule.append(agent)

    def step(self):
        """Execute one time step."""
        self.current_step += 1

        # Execute all agents
        for agent in self.schedule:
            agent.step()

        # Collect data
        self.data_collector.collect(self)

    def run(self, steps: int):
        """Run simulation for N steps."""
        for _ in range(steps):
            self.step()


@dataclass
class DataCollector:
    """Collects data during simulation."""
    model_data: Dict[int, Dict] = field(default_factory=dict)
    agent_data: Dict[int, List[Dict]] = field(default_factory=dict)

    def collect(self, model: Model):
        """Collect data at current step."""
        step = model.current_step

        # Model-level data
        self.model_data[step] = {
            'num_agents': len(model.agents),
            'step': step
        }

        # Agent-level data
        self.agent_data[step] = []
        for agent in model.agents:
            if hasattr(agent, 'wealth'):
                self.agent_data[step].append({
                    'agent_id': agent.unique_id,
                    'wealth': agent.wealth
                })

    def get_model_vars(self) -> Dict:
        """Get collected model data."""
        return self.model_data

    def get_agent_vars(self) -> Dict:
        """Get collected agent data."""
        return self.agent_data


# Example: Wealth Distribution Model


class WealthAgent(Agent):
    """Agent with wealth that can trade."""

    def __init__(self, unique_id: int, model: Model, initial_wealth: int = 1):
        super().__init__(unique_id, model)
        self.wealth = initial_wealth

    def step(self):
        """Give money to random agent if have wealth."""
        if self.wealth > 0:
            other_agent = random.choice(self.model.agents)
            if other_agent != self:
                other_agent.wealth += 1
                self.wealth -= 1


class WealthModel(Model):
    """Model of wealth redistribution."""

    def __init__(self, num_agents: int = 100, initial_wealth: int = 1):
        super().__init__(num_agents)
        self.initial_wealth = initial_wealth

        # Create agents
        for i in range(num_agents):
            agent = WealthAgent(i, self, initial_wealth)
            self.add_agent(agent)

    def get_statistics(self) -> Dict:
        """Get wealth statistics."""
        wealth_values = [a.wealth for a in self.agents]
        return {
            'total_wealth': sum(wealth_values),
            'mean_wealth': sum(wealth_values) / len(wealth_values) if wealth_values else 0,
            'max_wealth': max(wealth_values) if wealth_values else 0,
            'min_wealth': min(wealth_values) if wealth_values else 0
        }


# Example: Segregation Model


class SchellingAgent(Agent):
    """Agent for Schelling segregation model."""

    def __init__(self, unique_id: int, model: 'SchellingModel',
                 agent_type: int, x: int, y: int):
        super().__init__(unique_id, model)
        self.type = agent_type
        self.x = x
        self.y = y
        self.happy = False

    def step(self):
        """Move if not happy with neighborhood."""
        similar = 0
        total = 0

        # Check neighbors
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue

                nx, ny = self.x + dx, self.y + dy

                # Check bounds
                if 0 <= nx < self.model.width and 0 <= ny < self.model.height:
                    neighbor = self.model.grid[nx][ny]
                    if neighbor is not None:
                        total += 1
                        if neighbor.type == self.type:
                            similar += 1

        # Determine happiness
        if total > 0:
            self.happy = (similar / total) >= self.model.homophily
        else:
            self.happy = True

        # Move if unhappy
        if not self.happy:
            self.model.move_agent(self)


class SchellingModel(Model):
    """Schelling segregation model."""

    def __init__(self, width: int = 20, height: int = 20,
                 density: float = 0.8, minority_pc: float = 0.5,
                 homophily: float = 0.3):
        super().__init__()
        self.width = width
        self.height = height
        self.density = density
        self.homophily = homophily

        # Create grid
        self.grid = [[None for _ in range(height)] for _ in range(width)]
        self.empty_cells = []

        # Place agents
        agent_id = 0
        for x in range(width):
            for y in range(height):
                if random.random() < density:
                    agent_type = 1 if random.random() < minority_pc else 0
                    agent = SchellingAgent(agent_id, self, agent_type, x, y)
                    self.grid[x][y] = agent
                    self.add_agent(agent)
                    agent_id += 1
                else:
                    self.empty_cells.append((x, y))

    def move_agent(self, agent: SchellingAgent):
        """Move agent to random empty cell."""
        if not self.empty_cells:
            return

        # Free current cell
        self.grid[agent.x][agent.y] = None
        self.empty_cells.append((agent.x, agent.y))

        # Move to new cell
        new_x, new_y = random.choice(self.empty_cells)
        self.empty_cells.remove((new_x, new_y))
        agent.x = new_x
        agent.y = new_y
        self.grid[new_x][new_y] = agent

    def get_segregation(self) -> float:
        """Calculate segregation level."""
        happy_count = sum(1 for agent in self.agents if agent.happy)
        return happy_count / len(self.agents) if self.agents else 0


def demo():
    """Demonstration of Agent Simulation."""
    print("Agent Simulation - Multi-Agent Modeling Demo")
    print("=" * 70)

    # Demo 1: Wealth Model
    print("\n1️⃣  Wealth Distribution Model")
    print("-" * 70)

    wealth_model = WealthModel(num_agents=10, initial_wealth=1)

    print(f"Initial state:")
    stats = wealth_model.get_statistics()
    print(f"  Total wealth: {stats['total_wealth']}")
    print(f"  Mean wealth: {stats['mean_wealth']:.2f}")

    # Run simulation
    wealth_model.run(steps=100)

    print(f"\nAfter 100 steps:")
    stats = wealth_model.get_statistics()
    print(f"  Total wealth: {stats['total_wealth']}")
    print(f"  Mean wealth: {stats['mean_wealth']:.2f}")
    print(f"  Max wealth: {stats['max_wealth']}")
    print(f"  Min wealth: {stats['min_wealth']}")

    # Show distribution
    print(f"\nWealth distribution (first 5 agents):")
    for agent in wealth_model.agents[:5]:
        print(f"  Agent {agent.unique_id}: ${agent.wealth}")

    # Demo 2: Schelling Segregation Model
    print("\n2️⃣  Schelling Segregation Model")
    print("-" * 70)

    schelling_model = SchellingModel(
        width=10,
        height=10,
        density=0.8,
        minority_pc=0.5,
        homophily=0.4
    )

    print(f"Grid size: {schelling_model.width}x{schelling_model.height}")
    print(f"Number of agents: {len(schelling_model.agents)}")
    print(f"Homophily threshold: {schelling_model.homophily}")

    # Initial segregation
    seg_initial = schelling_model.get_segregation()
    print(f"\nInitial happiness: {seg_initial:.2%}")

    # Run simulation
    schelling_model.run(steps=50)

    # Final segregation
    seg_final = schelling_model.get_segregation()
    print(f"Final happiness (after 50 steps): {seg_final:.2%}")
    print(f"Improvement: {(seg_final - seg_initial):.2%}")

    # Datacollector
    print("\n3️⃣  Data Collection")
    print("-" * 70)

    model_data = wealth_model.data_collector.get_model_vars()
    print(f"Steps recorded: {len(model_data)}")
    print(f"Sample data points (steps 0, 50, 100):")
    for step in [0, 50, 100]:
        if step in model_data:
            print(f"  Step {step}: {model_data[step]}")

    # Statistics
    print("\n4️⃣  Simulation Statistics")
    print("-" * 70)
    print(f"Wealth Model:")
    print(f"  Agents: {len(wealth_model.agents)}")
    print(f"  Steps: {wealth_model.current_step}")

    print(f"\nSchelling Model:")
    print(f"  Agents: {len(schelling_model.agents)}")
    print(f"  Steps: {schelling_model.current_step}")
    print(f"  Grid cells: {schelling_model.width * schelling_model.height}")
    print(f"  Empty cells: {len(schelling_model.empty_cells)}")

    print("\n" + "=" * 70)
    print("✓ Demo Complete!")


if __name__ == '__main__':
    demo()
