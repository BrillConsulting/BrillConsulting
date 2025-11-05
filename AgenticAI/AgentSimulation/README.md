# Agent Simulation

Multi-agent simulation framework using Mesa for agent-based modeling and emergent behavior analysis.

## Features

- **Mesa Integration**: Built on Mesa agent-based modeling framework
- **Spatial Environments**: 2D grids and continuous spaces
- **Agent Scheduling**: Flexible agent activation patterns
- **Data Collection**: Automated metrics gathering during simulation
- **Visualization**: Real-time and post-simulation visualization
- **Batch Running**: Parameter sweeps and statistical analysis
- **Network Topologies**: Graph-based agent interactions
- **Custom Behaviors**: Extensible agent behavior models

## Quick Start

```python
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
import matplotlib.pyplot as plt

# Define agent class
class WealthAgent(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.wealth = 1

    def step(self):
        # Move randomly
        possible_steps = self.model.grid.get_neighborhood(
            self.pos, moore=True, include_center=False
        )
        new_position = self.random.choice(possible_steps)
        self.model.grid.move_agent(self, new_position)

        # Share wealth with nearby agents
        cellmates = self.model.grid.get_cell_list_contents([self.pos])
        if len(cellmates) > 1:
            other = self.random.choice(cellmates)
            other.wealth += 1
            self.wealth -= 1

# Define model
class WealthModel(Model):
    def __init__(self, n_agents, width, height):
        super().__init__()
        self.num_agents = n_agents
        self.grid = MultiGrid(width, height, True)
        self.schedule = RandomActivation(self)

        # Create agents
        for i in range(self.num_agents):
            agent = WealthAgent(i, self)
            self.schedule.add(agent)

            # Place agent
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

        # Data collection
        self.datacollector = DataCollector(
            model_reporters={"Gini": compute_gini},
            agent_reporters={"Wealth": "wealth"}
        )

    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()

# Run simulation
model = WealthModel(n_agents=50, width=10, height=10)

for i in range(100):
    model.step()

# Analyze results
agent_wealth = model.datacollector.get_agent_vars_dataframe()
model_data = model.datacollector.get_model_vars_dataframe()

plt.plot(model_data['Gini'])
plt.xlabel('Step')
plt.ylabel('Gini Coefficient')
plt.show()
```

## Use Cases

- **Economics**: Market simulations, wealth distribution models
- **Epidemiology**: Disease spread and vaccination strategies
- **Ecology**: Predator-prey dynamics, ecosystem modeling
- **Social Sciences**: Opinion dynamics, segregation models
- **Urban Planning**: Traffic flow, crowd behavior
- **Emergency Response**: Evacuation simulations

## Mesa Components

### Agent
Individual entities with autonomous behavior:
```python
class MyAgent(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.state = "initial"

    def step(self):
        # Agent behavior logic
        pass
```

### Model
Simulation environment and orchestration:
```python
class MyModel(Model):
    def __init__(self):
        super().__init__()
        self.schedule = RandomActivation(self)
        self.grid = MultiGrid(width, height, torus=True)

    def step(self):
        self.schedule.step()
```

### Space Types
- **MultiGrid**: Multiple agents per cell
- **SingleGrid**: One agent per cell
- **ContinuousSpace**: Continuous 2D coordinates
- **NetworkGrid**: Graph-based topology

### Schedulers
- **RandomActivation**: Random agent order each step
- **SimultaneousActivation**: All agents act, then update
- **StagedActivation**: Multi-phase agent activation
- **BaseScheduler**: Custom scheduling logic

## Advanced Features

### Batch Running
```python
from mesa.batchrunner import BatchRunner

# Define parameter ranges
params = {
    "n_agents": range(10, 100, 10),
    "width": 20,
    "height": 20
}

batch_run = BatchRunner(
    WealthModel,
    variable_parameters=params,
    iterations=5,
    max_steps=100,
    model_reporters={"Gini": compute_gini}
)

batch_run.run_all()
results = batch_run.get_model_vars_dataframe()
```

### Interactive Visualization
```python
from mesa.visualization.modules import CanvasGrid
from mesa.visualization.ModularVisualization import ModularServer

def agent_portrayal(agent):
    return {
        "Shape": "circle",
        "Filled": "true",
        "Color": "red" if agent.wealth > 5 else "blue",
        "r": 0.8
    }

grid = CanvasGrid(agent_portrayal, 10, 10, 500, 500)

server = ModularServer(
    WealthModel,
    [grid],
    "Wealth Model",
    {"n_agents": 50, "width": 10, "height": 10}
)

server.launch()
```

### Network Models
```python
import networkx as nx
from mesa.space import NetworkGrid

# Create network topology
G = nx.watts_strogatz_graph(n=100, k=4, p=0.1)
network = NetworkGrid(G)

# Place agents on network
for node in G.nodes():
    agent = MyAgent(node, model)
    network.place_agent(agent, node)
```

## Data Collection

```python
def compute_avg_wealth(model):
    agent_wealths = [agent.wealth for agent in model.schedule.agents]
    return sum(agent_wealths) / len(agent_wealths)

datacollector = DataCollector(
    model_reporters={
        "Average Wealth": compute_avg_wealth,
        "Total Wealth": lambda m: sum(a.wealth for a in m.schedule.agents)
    },
    agent_reporters={
        "Wealth": "wealth",
        "Position": lambda a: a.pos
    }
)
```

## Best Practices

- Start with simple agent rules
- Use appropriate space topology for your domain
- Collect relevant metrics from the start
- Run multiple iterations for statistical significance
- Validate against real-world data when possible
- Profile performance for large simulations
- Document agent behavior clearly
- Use version control for model evolution

## Author

Brill Consulting
