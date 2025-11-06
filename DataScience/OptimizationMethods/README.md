# Optimization Methods Toolkit

A comprehensive toolkit implementing multiple optimization algorithms for solving complex optimization problems in machine learning and data science.

## Description

The Optimization Methods Toolkit provides a unified interface for various optimization algorithms. It includes gradient-based methods, evolutionary algorithms, and metaheuristic approaches for finding optimal or near-optimal solutions to continuous and discrete optimization problems.

## Key Features

- **Gradient-Based Methods**
  - Gradient Descent (Batch, Mini-batch)
  - Stochastic Gradient Descent (SGD)
  - Momentum optimization
  - Nesterov Accelerated Gradient (NAG)
  - AdaGrad, RMSprop, Adam optimizers

- **Evolutionary Algorithms**
  - Genetic Algorithms (GA)
  - Differential Evolution (DE)
  - Evolution Strategies (ES)
  - Crossover and mutation operators
  - Tournament and roulette selection

- **Metaheuristic Methods**
  - Simulated Annealing
  - Particle Swarm Optimization (PSO)
  - Ant Colony Optimization (ACO)
  - Tabu Search

- **Analysis Tools**
  - Convergence tracking
  - Objective function evaluation
  - Parameter sensitivity analysis
  - Performance comparison across methods

- **Visualization**
  - Convergence curves
  - Solution space exploration
  - Population diversity plots
  - Optimization trajectory visualization

## Technologies Used

- **Python 3.x**
- **NumPy** - Numerical computing
- **SciPy** - Scientific computing and optimization
- **Matplotlib/Seaborn** - Visualization
- **Pandas** - Data manipulation

## Installation

```bash
# Clone the repository
cd /home/user/BrillConsulting/DataScience/OptimizationMethods

# Install required packages
pip install numpy scipy matplotlib seaborn pandas
```

## Usage Examples

### Gradient Descent Optimization

```python
from optimization_methods import OptimizationToolkit
import numpy as np

# Define objective function (Rosenbrock function)
def rosenbrock(x):
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

def rosenbrock_gradient(x):
    dx0 = -2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0]**2)
    dx1 = 200 * (x[1] - x[0]**2)
    return np.array([dx0, dx1])

# Initialize toolkit
optimizer = OptimizationToolkit()

# Apply gradient descent
gd_result = optimizer.gradient_descent(
    objective_func=rosenbrock,
    gradient_func=rosenbrock_gradient,
    x0=np.array([-1.0, 1.0]),
    learning_rate=0.001,
    max_iterations=10000,
    tolerance=1e-6
)

print(f"Optimal solution: {gd_result['x_optimal']}")
print(f"Optimal value: {gd_result['f_optimal']:.6f}")
print(f"Iterations: {gd_result['n_iterations']}")
print(f"Convergence achieved: {gd_result['converged']}")
```

### Adam Optimizer

```python
# Apply Adam optimizer
adam_result = optimizer.adam(
    objective_func=rosenbrock,
    gradient_func=rosenbrock_gradient,
    x0=np.array([-1.0, 1.0]),
    learning_rate=0.01,
    beta1=0.9,
    beta2=0.999,
    max_iterations=5000
)

print(f"Adam optimizer results:")
print(f"Optimal solution: {adam_result['x_optimal']}")
print(f"Optimal value: {adam_result['f_optimal']:.6f}")
print(f"Iterations: {adam_result['n_iterations']}")

# Visualize convergence
fig = optimizer.plot_convergence(adam_result['history'])
fig.savefig('adam_convergence.png', dpi=300, bbox_inches='tight')
```

### Genetic Algorithm

```python
# Define bounds for optimization problem
bounds = [(-2.0, 2.0), (-2.0, 2.0)]

# Apply genetic algorithm
ga_result = optimizer.genetic_algorithm(
    objective_func=rosenbrock,
    bounds=bounds,
    population_size=100,
    n_generations=200,
    crossover_rate=0.8,
    mutation_rate=0.1,
    tournament_size=5
)

print(f"Genetic Algorithm results:")
print(f"Best solution: {ga_result['x_best']}")
print(f"Best fitness: {ga_result['f_best']:.6f}")
print(f"Generations: {ga_result['n_generations']}")
print(f"Final population diversity: {ga_result['diversity']:.3f}")
```

### Simulated Annealing

```python
# Apply simulated annealing
sa_result = optimizer.simulated_annealing(
    objective_func=rosenbrock,
    x0=np.array([0.0, 0.0]),
    bounds=bounds,
    initial_temp=100.0,
    cooling_rate=0.95,
    max_iterations=10000
)

print(f"Simulated Annealing results:")
print(f"Best solution: {sa_result['x_best']}")
print(f"Best value: {sa_result['f_best']:.6f}")
print(f"Acceptance rate: {sa_result['acceptance_rate']:.3f}")
print(f"Final temperature: {sa_result['final_temp']:.3f}")
```

### Particle Swarm Optimization

```python
# Apply PSO
pso_result = optimizer.particle_swarm_optimization(
    objective_func=rosenbrock,
    bounds=bounds,
    n_particles=50,
    max_iterations=200,
    w=0.7,  # Inertia weight
    c1=1.5,  # Cognitive parameter
    c2=1.5   # Social parameter
)

print(f"PSO results:")
print(f"Best position: {pso_result['x_best']}")
print(f"Best value: {pso_result['f_best']:.6f}")
print(f"Iterations: {pso_result['n_iterations']}")
print(f"Swarm diversity: {pso_result['swarm_diversity']:.3f}")

# Visualize particle trajectories
fig = optimizer.plot_swarm_trajectory(pso_result['history'])
fig.savefig('pso_trajectory.png', dpi=300, bbox_inches='tight')
```

### Differential Evolution

```python
# Apply differential evolution
de_result = optimizer.differential_evolution(
    objective_func=rosenbrock,
    bounds=bounds,
    population_size=50,
    max_iterations=200,
    mutation_factor=0.8,
    crossover_prob=0.7,
    strategy='best1bin'
)

print(f"Differential Evolution results:")
print(f"Best solution: {de_result['x_best']}")
print(f"Best value: {de_result['f_best']:.6f}")
print(f"Function evaluations: {de_result['n_evaluations']}")
```

### Method Comparison

```python
# Compare multiple optimization methods
comparison = optimizer.compare_methods(
    objective_func=rosenbrock,
    gradient_func=rosenbrock_gradient,
    x0=np.array([0.0, 0.0]),
    bounds=bounds,
    methods=['gradient_descent', 'adam', 'genetic_algorithm',
             'simulated_annealing', 'pso', 'differential_evolution']
)

print("\nMethod Comparison:")
print(f"{'Method':<25} {'Best Value':<15} {'Iterations':<12} {'Time (s)'}")
print("-" * 70)
for method, result in comparison.items():
    print(f"{method:<25} {result['f_best']:<15.6f} "
          f"{result['n_iterations']:<12} {result['time']:.3f}")

# Visualize comparison
fig = optimizer.plot_method_comparison(comparison)
fig.savefig('method_comparison.png', dpi=300, bbox_inches='tight')
```

## Demo Instructions

Run the comprehensive demo to see all features in action:

```bash
python optimization_methods.py
```

The demo will:
1. Define standard test functions (Rosenbrock, Rastrigin, Ackley, Sphere)
2. Apply all optimization methods to each function
3. Track convergence and performance metrics
4. Compare methods across different problem complexities
5. Generate visualizations (convergence curves, solution landscapes)
6. Display comprehensive performance comparison

## Output Examples

**Console Output:**
```
Optimization Methods Toolkit Demo
======================================================================

Test Function: Rosenbrock
Known minimum: f(1, 1) = 0
Dimension: 2

1. Gradient Descent
----------------------------------------------------------------------
Initial position: [0.0, 0.0]
Learning rate: 0.001
Optimal solution: [0.9998, 0.9996]
Optimal value: 0.000004
Iterations: 8,543
Convergence time: 0.234s
Converged: True

2. Adam Optimizer
----------------------------------------------------------------------
Learning rate: 0.01
Beta1: 0.9, Beta2: 0.999
Optimal solution: [1.0001, 1.0002]
Optimal value: 0.000001
Iterations: 3,421
Convergence time: 0.145s

3. Genetic Algorithm
----------------------------------------------------------------------
Population size: 100
Generations: 200
Best solution: [0.9995, 0.9989]
Best fitness: 0.000026
Final diversity: 0.123
Time: 2.345s

4. Simulated Annealing
----------------------------------------------------------------------
Initial temperature: 100.0
Cooling rate: 0.95
Best solution: [0.9992, 0.9984]
Best value: 0.000067
Acceptance rate: 0.432
Time: 0.876s

5. Particle Swarm Optimization
----------------------------------------------------------------------
Number of particles: 50
Iterations: 200
Best position: [1.0003, 1.0006]
Best value: 0.000018
Swarm diversity: 0.089
Time: 1.234s

6. Differential Evolution
----------------------------------------------------------------------
Population size: 50
Strategy: best1bin
Best solution: [0.9997, 0.9994]
Best value: 0.000011
Function evaluations: 10,000
Time: 1.567s

Performance Summary (Rosenbrock Function)
----------------------------------------------------------------------
Method                    Best Value      Iterations    Time (s)
----------------------------------------------------------------------
Gradient Descent          0.000004        8,543         0.234
Adam                      0.000001        3,421         0.145
Genetic Algorithm         0.000026        20,000        2.345
Simulated Annealing       0.000067        10,000        0.876
PSO                       0.000018        10,000        1.234
Differential Evolution    0.000011        10,000        1.567
----------------------------------------------------------------------

Winner: Adam (fastest convergence, best solution)
```

**Generated Visualizations:**
- `convergence_curves.png` - Convergence comparison across methods
- `solution_landscape.png` - 2D visualization of optimization trajectory
- `population_evolution.png` - Evolution of GA/PSO populations
- `method_comparison.png` - Performance metrics comparison
- `sensitivity_analysis.png` - Parameter sensitivity plots

## Author

**Brill Consulting**

---

For more information about the algorithms and methodologies, see the inline documentation in `optimization_methods.py`.
