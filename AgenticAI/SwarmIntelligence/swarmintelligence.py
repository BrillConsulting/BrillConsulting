"""
Swarm Intelligence - PSO, ACO, and Flocking
============================================

Collective behavior and emergent intelligence using Particle Swarm Optimization,
Ant Colony Optimization, and flocking algorithms.

Author: Brill Consulting
"""

from typing import List, Tuple, Callable, Optional
from dataclasses import dataclass, field
import numpy as np
import random
from abc import ABC, abstractmethod


# Particle Swarm Optimization


@dataclass
class Particle:
    """Particle in PSO with position and velocity."""
    position: np.ndarray
    velocity: np.ndarray
    best_position: np.ndarray = None
    best_fitness: float = float('inf')
    fitness: float = float('inf')

    def __post_init__(self):
        if self.best_position is None:
            self.best_position = self.position.copy()


class ParticleSwarm:
    """Particle Swarm Optimization algorithm."""

    def __init__(self,
                 objective_function: Callable[[np.ndarray], float],
                 dimensions: int,
                 num_particles: int = 30,
                 bounds: Tuple[float, float] = (-100, 100),
                 inertia: float = 0.7,
                 cognitive_weight: float = 1.5,
                 social_weight: float = 1.5):
        """
        Initialize PSO.

        Args:
            objective_function: Function to minimize
            dimensions: Problem dimensionality
            num_particles: Number of particles in swarm
            bounds: Search space bounds (min, max)
            inertia: Inertia weight (w)
            cognitive_weight: Cognitive coefficient (c1)
            social_weight: Social coefficient (c2)
        """
        self.objective_function = objective_function
        self.dimensions = dimensions
        self.num_particles = num_particles
        self.bounds = bounds
        self.w = inertia
        self.c1 = cognitive_weight
        self.c2 = social_weight

        # Initialize swarm
        self.particles: List[Particle] = []
        self.global_best_position = None
        self.global_best_fitness = float('inf')
        self.iteration = 0

        self._initialize_swarm()

    def _initialize_swarm(self):
        """Create initial particles with random positions."""
        for _ in range(self.num_particles):
            position = np.random.uniform(
                self.bounds[0], self.bounds[1], self.dimensions
            )
            velocity = np.random.uniform(
                -abs(self.bounds[1] - self.bounds[0]) * 0.1,
                abs(self.bounds[1] - self.bounds[0]) * 0.1,
                self.dimensions
            )

            particle = Particle(position=position, velocity=velocity)
            particle.fitness = self.objective_function(particle.position)
            particle.best_fitness = particle.fitness
            particle.best_position = particle.position.copy()

            # Update global best
            if particle.fitness < self.global_best_fitness:
                self.global_best_fitness = particle.fitness
                self.global_best_position = particle.position.copy()

            self.particles.append(particle)

    def optimize(self, max_iterations: int = 100) -> Tuple[np.ndarray, float]:
        """
        Run PSO optimization.

        Returns:
            (best_position, best_fitness)
        """
        for iteration in range(max_iterations):
            self.iteration = iteration

            for particle in self.particles:
                # Update velocity
                r1 = np.random.random(self.dimensions)
                r2 = np.random.random(self.dimensions)

                cognitive = self.c1 * r1 * (particle.best_position - particle.position)
                social = self.c2 * r2 * (self.global_best_position - particle.position)

                particle.velocity = self.w * particle.velocity + cognitive + social

                # Update position
                particle.position = particle.position + particle.velocity

                # Enforce bounds
                particle.position = np.clip(
                    particle.position, self.bounds[0], self.bounds[1]
                )

                # Evaluate fitness
                particle.fitness = self.objective_function(particle.position)

                # Update personal best
                if particle.fitness < particle.best_fitness:
                    particle.best_fitness = particle.fitness
                    particle.best_position = particle.position.copy()

                # Update global best
                if particle.fitness < self.global_best_fitness:
                    self.global_best_fitness = particle.fitness
                    self.global_best_position = particle.position.copy()

        return self.global_best_position, self.global_best_fitness


# Ant Colony Optimization


@dataclass
class Ant:
    """Ant agent for ACO."""
    current_node: int
    visited: List[int] = field(default_factory=list)
    path_cost: float = 0.0

    def visit(self, node: int, cost: float):
        """Visit a node."""
        self.visited.append(node)
        self.current_node = node
        self.path_cost += cost

    def has_visited(self, node: int) -> bool:
        """Check if node was visited."""
        return node in self.visited


class AntColony:
    """Ant Colony Optimization for TSP."""

    def __init__(self,
                 distance_matrix: np.ndarray,
                 num_ants: int = 20,
                 evaporation_rate: float = 0.5,
                 alpha: float = 1.0,
                 beta: float = 2.0,
                 q: float = 100.0):
        """
        Initialize ACO.

        Args:
            distance_matrix: NxN matrix of distances between nodes
            num_ants: Number of ants
            evaporation_rate: Pheromone evaporation rate (ρ)
            alpha: Pheromone importance
            beta: Distance importance
            q: Pheromone deposit factor
        """
        self.distances = distance_matrix
        self.num_nodes = len(distance_matrix)
        self.num_ants = num_ants
        self.rho = evaporation_rate
        self.alpha = alpha
        self.beta = beta
        self.q = q

        # Initialize pheromones
        self.pheromones = np.ones((self.num_nodes, self.num_nodes))
        self.best_path = None
        self.best_cost = float('inf')
        self.iteration = 0

    def _select_next_node(self, ant: Ant) -> int:
        """Select next node using probability rule."""
        current = ant.current_node
        unvisited = [i for i in range(self.num_nodes) if not ant.has_visited(i)]

        if not unvisited:
            return ant.visited[0]  # Return to start

        # Calculate probabilities
        probabilities = []
        for node in unvisited:
            pheromone = self.pheromones[current][node] ** self.alpha
            visibility = (1.0 / self.distances[current][node]) ** self.beta
            probabilities.append(pheromone * visibility)

        # Normalize
        total = sum(probabilities)
        probabilities = [p / total for p in probabilities]

        # Select node
        return np.random.choice(unvisited, p=probabilities)

    def _construct_solution(self, ant: Ant):
        """Construct solution for one ant."""
        # Start at random node
        start_node = random.randint(0, self.num_nodes - 1)
        ant.current_node = start_node
        ant.visited = [start_node]
        ant.path_cost = 0.0

        # Visit all nodes
        for _ in range(self.num_nodes - 1):
            next_node = self._select_next_node(ant)
            cost = self.distances[ant.current_node][next_node]
            ant.visit(next_node, cost)

        # Return to start
        cost = self.distances[ant.current_node][start_node]
        ant.path_cost += cost

    def _update_pheromones(self, ants: List[Ant]):
        """Update pheromone trails."""
        # Evaporation
        self.pheromones *= (1 - self.rho)

        # Deposit pheromones
        for ant in ants:
            deposit = self.q / ant.path_cost
            for i in range(len(ant.visited)):
                from_node = ant.visited[i]
                to_node = ant.visited[(i + 1) % len(ant.visited)]
                self.pheromones[from_node][to_node] += deposit
                self.pheromones[to_node][from_node] += deposit

    def optimize(self, max_iterations: int = 100) -> Tuple[List[int], float]:
        """
        Run ACO optimization.

        Returns:
            (best_path, best_cost)
        """
        for iteration in range(max_iterations):
            self.iteration = iteration
            ants = [Ant(current_node=0) for _ in range(self.num_ants)]

            # Construct solutions
            for ant in ants:
                self._construct_solution(ant)

                # Update best solution
                if ant.path_cost < self.best_cost:
                    self.best_cost = ant.path_cost
                    self.best_path = ant.visited.copy()

            # Update pheromones
            self._update_pheromones(ants)

        return self.best_path, self.best_cost


# Flocking Behavior


@dataclass
class Boid:
    """Individual agent in flocking simulation."""
    position: np.ndarray
    velocity: np.ndarray
    id: int = 0

    def update(self, acceleration: np.ndarray, dt: float = 1.0,
               max_speed: float = 5.0):
        """Update boid position and velocity."""
        self.velocity += acceleration * dt

        # Limit speed
        speed = np.linalg.norm(self.velocity)
        if speed > max_speed:
            self.velocity = (self.velocity / speed) * max_speed

        self.position += self.velocity * dt


class FlockingSimulation:
    """Flocking simulation with Reynolds' rules."""

    def __init__(self,
                 num_boids: int = 50,
                 bounds: Tuple[float, float, float, float] = (0, 100, 0, 100),
                 separation_weight: float = 1.5,
                 alignment_weight: float = 1.0,
                 cohesion_weight: float = 1.0,
                 perception_radius: float = 10.0):
        """
        Initialize flocking simulation.

        Args:
            num_boids: Number of boids
            bounds: (min_x, max_x, min_y, max_y)
            separation_weight: Weight for separation force
            alignment_weight: Weight for alignment force
            cohesion_weight: Weight for cohesion force
            perception_radius: Perception range for neighbors
        """
        self.bounds = bounds
        self.w_sep = separation_weight
        self.w_align = alignment_weight
        self.w_coh = cohesion_weight
        self.perception_radius = perception_radius

        # Initialize boids
        self.boids: List[Boid] = []
        for i in range(num_boids):
            position = np.array([
                random.uniform(bounds[0], bounds[1]),
                random.uniform(bounds[2], bounds[3])
            ])
            velocity = np.random.uniform(-2, 2, 2)
            self.boids.append(Boid(position=position, velocity=velocity, id=i))

        self.time_step = 0

    def _get_neighbors(self, boid: Boid) -> List[Boid]:
        """Get nearby boids within perception radius."""
        neighbors = []
        for other in self.boids:
            if other.id != boid.id:
                distance = np.linalg.norm(boid.position - other.position)
                if distance < self.perception_radius:
                    neighbors.append(other)
        return neighbors

    def _separation(self, boid: Boid, neighbors: List[Boid]) -> np.ndarray:
        """Separation: steer away from neighbors."""
        if not neighbors:
            return np.zeros(2)

        force = np.zeros(2)
        for neighbor in neighbors:
            diff = boid.position - neighbor.position
            distance = np.linalg.norm(diff)
            if distance > 0:
                force += diff / (distance ** 2)

        return force

    def _alignment(self, boid: Boid, neighbors: List[Boid]) -> np.ndarray:
        """Alignment: steer toward average heading."""
        if not neighbors:
            return np.zeros(2)

        avg_velocity = np.mean([n.velocity for n in neighbors], axis=0)
        return avg_velocity - boid.velocity

    def _cohesion(self, boid: Boid, neighbors: List[Boid]) -> np.ndarray:
        """Cohesion: steer toward average position."""
        if not neighbors:
            return np.zeros(2)

        avg_position = np.mean([n.position for n in neighbors], axis=0)
        return avg_position - boid.position

    def _enforce_bounds(self, boid: Boid) -> np.ndarray:
        """Keep boids within bounds."""
        force = np.zeros(2)
        margin = 5.0

        if boid.position[0] < self.bounds[0] + margin:
            force[0] += 1.0
        elif boid.position[0] > self.bounds[1] - margin:
            force[0] -= 1.0

        if boid.position[1] < self.bounds[2] + margin:
            force[1] += 1.0
        elif boid.position[1] > self.bounds[3] - margin:
            force[1] -= 1.0

        return force * 10

    def step(self, dt: float = 0.1):
        """Execute one simulation step."""
        self.time_step += 1

        for boid in self.boids:
            neighbors = self._get_neighbors(boid)

            # Calculate forces
            separation = self._separation(boid, neighbors) * self.w_sep
            alignment = self._alignment(boid, neighbors) * self.w_align
            cohesion = self._cohesion(boid, neighbors) * self.w_coh
            bounds = self._enforce_bounds(boid)

            # Total acceleration
            acceleration = separation + alignment + cohesion + bounds

            # Update boid
            boid.update(acceleration, dt)

    def get_statistics(self) -> dict:
        """Get swarm statistics."""
        positions = np.array([b.position for b in self.boids])
        velocities = np.array([b.velocity for b in self.boids])

        center = np.mean(positions, axis=0)
        spread = np.std(positions, axis=0)
        avg_speed = np.mean([np.linalg.norm(v) for v in velocities])

        return {
            'center': center,
            'spread': spread,
            'avg_speed': avg_speed,
            'num_boids': len(self.boids)
        }


def demo():
    """Demonstration of Swarm Intelligence."""
    print("Swarm Intelligence - PSO, ACO, and Flocking Demo")
    print("=" * 70)

    # Demo 1: Particle Swarm Optimization
    print("\n1️⃣  Particle Swarm Optimization (PSO)")
    print("-" * 70)

    # Sphere function: f(x) = sum(x_i^2)
    def sphere_function(x: np.ndarray) -> float:
        return np.sum(x ** 2)

    # Rastrigin function (harder)
    def rastrigin_function(x: np.ndarray) -> float:
        A = 10
        return A * len(x) + np.sum(x**2 - A * np.cos(2 * np.pi * x))

    pso = ParticleSwarm(
        objective_function=sphere_function,
        dimensions=5,
        num_particles=30,
        bounds=(-10, 10)
    )

    print(f"Optimizing sphere function in {pso.dimensions}D space")
    print(f"Swarm size: {pso.num_particles} particles")

    best_pos, best_fitness = pso.optimize(max_iterations=50)

    print(f"\n✓ Optimization complete!")
    print(f"Best position: {best_pos}")
    print(f"Best fitness: {best_fitness:.6f}")
    print(f"Expected optimum: 0.0 at [0, 0, 0, 0, 0]")

    # Demo 2: Ant Colony Optimization
    print("\n2️⃣  Ant Colony Optimization (ACO)")
    print("-" * 70)

    # Create distance matrix for 5 cities
    np.random.seed(42)
    num_cities = 5
    cities = np.random.rand(num_cities, 2) * 100

    # Calculate distance matrix
    distances = np.zeros((num_cities, num_cities))
    for i in range(num_cities):
        for j in range(num_cities):
            if i != j:
                distances[i][j] = np.linalg.norm(cities[i] - cities[j])

    print(f"Solving TSP for {num_cities} cities")
    print("Distance matrix:")
    print(distances.astype(int))

    aco = AntColony(
        distance_matrix=distances,
        num_ants=20,
        evaporation_rate=0.5,
        alpha=1.0,
        beta=2.0
    )

    best_path, best_cost = aco.optimize(max_iterations=50)

    print(f"\n✓ Optimization complete!")
    print(f"Best path: {best_path}")
    print(f"Best cost: {best_cost:.2f}")
    print(f"Path visits all cities and returns to start")

    # Demo 3: Flocking Simulation
    print("\n3️⃣  Flocking Simulation (Boids)")
    print("-" * 70)

    flock = FlockingSimulation(
        num_boids=30,
        bounds=(0, 100, 0, 100),
        separation_weight=1.5,
        alignment_weight=1.0,
        cohesion_weight=1.0,
        perception_radius=15.0
    )

    print(f"Simulating {len(flock.boids)} boids")
    print(f"Perception radius: {flock.perception_radius}")

    # Initial statistics
    stats = flock.get_statistics()
    print(f"\nInitial state:")
    print(f"  Center: ({stats['center'][0]:.2f}, {stats['center'][1]:.2f})")
    print(f"  Spread: ({stats['spread'][0]:.2f}, {stats['spread'][1]:.2f})")
    print(f"  Avg speed: {stats['avg_speed']:.2f}")

    # Run simulation
    print("\nRunning simulation...")
    for step in range(100):
        flock.step(dt=0.1)

    # Final statistics
    stats = flock.get_statistics()
    print(f"\nFinal state (after 100 steps):")
    print(f"  Center: ({stats['center'][0]:.2f}, {stats['center'][1]:.2f})")
    print(f"  Spread: ({stats['spread'][0]:.2f}, {stats['spread'][1]:.2f})")
    print(f"  Avg speed: {stats['avg_speed']:.2f}")

    # Sample boid positions
    print("\nSample boid positions (first 5):")
    for i in range(min(5, len(flock.boids))):
        boid = flock.boids[i]
        print(f"  Boid {i}: ({boid.position[0]:.2f}, {boid.position[1]:.2f})")

    # Performance summary
    print("\n4️⃣  Swarm Intelligence Summary")
    print("-" * 70)
    print(f"PSO: Converged to fitness {best_fitness:.6f}")
    print(f"ACO: Found TSP solution with cost {best_cost:.2f}")
    print(f"Flocking: {len(flock.boids)} boids coordinated over {flock.time_step} steps")
    print("\n✓ All swarm algorithms demonstrated emergent intelligence!")

    print("\n" + "=" * 70)
    print("✓ Demo Complete!")


if __name__ == '__main__':
    demo()
