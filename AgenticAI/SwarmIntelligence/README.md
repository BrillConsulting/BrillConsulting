# Swarm Intelligence

Collective behavior algorithms including particle swarm optimization and ant colony systems.

## Features

- **Particle Swarm Optimization**: Global optimization through particle movement
- **Ant Colony**: Pheromone-based pathfinding
- **Flocking Behavior**: Coordinated group movement
- **Collective Decision Making**: Swarm consensus mechanisms
- **Distributed Problem Solving**: No central control
- **Emergence Patterns**: Complex behavior from simple rules

## Quick Start

```python
from swarm_intelligence import ParticleSwarm, AntColony

# Particle Swarm Optimization
pso = ParticleSwarm(
    n_particles=30,
    dimensions=10,
    bounds=(-10, 10)
)

best_position = pso.optimize(
    fitness_function=lambda x: sum(x**2),  # Example: sphere function
    max_iterations=100
)

# Ant Colony Optimization
aco = AntColony(
    n_ants=20,
    n_nodes=10,
    evaporation_rate=0.1
)

best_path = aco.find_path(
    start=0,
    end=9,
    iterations=100
)
```

## Use Cases

- **Optimization**: Find global optima in complex spaces
- **Routing**: Efficient path planning
- **Scheduling**: Resource allocation problems

## Author

Brill Consulting
