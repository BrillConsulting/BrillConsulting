"""
Optimization Methods Toolkit

A comprehensive toolkit for various optimization algorithms including
linear programming, genetic algorithms, simulated annealing, particle swarm
optimization, and gradient descent variants.

Author: Brill Consulting
Date: 2025-11-06
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Callable, Optional, Tuple, List, Dict, Any, Union
from scipy.optimize import linprog, minimize, differential_evolution
from dataclasses import dataclass
import warnings

warnings.filterwarnings('ignore')


@dataclass
class OptimizationResult:
    """Container for optimization results."""
    x: np.ndarray
    fun: float
    nit: int
    success: bool
    message: str
    history: Optional[List[float]] = None


class OptimizationToolkit:
    """
    Comprehensive optimization toolkit.

    Provides various optimization algorithms for solving different types
    of optimization problems including linear programming, non-linear
    optimization, and metaheuristic methods.

    Attributes:
        random_state: Random seed for reproducibility
        verbose: Whether to print progress messages

    Example:
        >>> opt = OptimizationToolkit(random_state=42)
        >>> result = opt.genetic_algorithm(objective_func, bounds)
        >>> print(f"Optimal value: {result.fun}")
    """

    def __init__(self, random_state: Optional[int] = None, verbose: bool = True):
        """
        Initialize the OptimizationToolkit.

        Args:
            random_state: Random seed for reproducibility
            verbose: Whether to print progress messages
        """
        self.random_state = random_state
        self.verbose = verbose
        if random_state is not None:
            np.random.seed(random_state)

    def linear_programming(
        self,
        c: np.ndarray,
        A_ub: Optional[np.ndarray] = None,
        b_ub: Optional[np.ndarray] = None,
        A_eq: Optional[np.ndarray] = None,
        b_eq: Optional[np.ndarray] = None,
        bounds: Optional[List[Tuple[float, float]]] = None,
        method: str = 'highs'
    ) -> OptimizationResult:
        """
        Solve linear programming problem using scipy.

        Minimize: c^T * x
        Subject to: A_ub @ x <= b_ub
                   A_eq @ x == b_eq
                   bounds

        Args:
            c: Coefficients of the linear objective function
            A_ub: Inequality constraint matrix
            b_ub: Inequality constraint vector
            A_eq: Equality constraint matrix
            b_eq: Equality constraint vector
            bounds: Bounds for variables
            method: Solution method

        Returns:
            OptimizationResult containing solution
        """
        if self.verbose:
            print("Solving linear programming problem...")

        result = linprog(
            c=c,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds,
            method=method
        )

        if self.verbose:
            print(f"Status: {result.message}")
            print(f"Optimal value: {result.fun:.6f}")

        return OptimizationResult(
            x=result.x,
            fun=result.fun,
            nit=result.nit,
            success=result.success,
            message=result.message
        )

    def genetic_algorithm(
        self,
        objective_func: Callable,
        bounds: List[Tuple[float, float]],
        population_size: int = 50,
        generations: int = 100,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.8,
        elitism: float = 0.1
    ) -> OptimizationResult:
        """
        Genetic Algorithm optimization.

        Args:
            objective_func: Function to minimize
            bounds: List of (min, max) tuples for each dimension
            population_size: Size of population
            generations: Number of generations
            mutation_rate: Probability of mutation
            crossover_rate: Probability of crossover
            elitism: Fraction of best individuals to preserve

        Returns:
            OptimizationResult containing best solution
        """
        n_dims = len(bounds)
        bounds_array = np.array(bounds)

        # Initialize population
        population = np.random.uniform(
            bounds_array[:, 0],
            bounds_array[:, 1],
            size=(population_size, n_dims)
        )

        best_fitness_history = []
        n_elite = max(1, int(population_size * elitism))

        for gen in range(generations):
            # Evaluate fitness
            fitness = np.array([objective_func(ind) for ind in population])

            # Track best
            best_idx = np.argmin(fitness)
            best_fitness_history.append(fitness[best_idx])

            if self.verbose and (gen + 1) % 20 == 0:
                print(f"Generation {gen + 1}/{generations}: Best = {fitness[best_idx]:.6f}")

            # Elite preservation
            elite_indices = np.argsort(fitness)[:n_elite]
            new_population = [population[i].copy() for i in elite_indices]

            # Selection (tournament selection)
            n_offspring = population_size - n_elite
            selected_indices = []
            for _ in range(n_offspring):
                tournament = np.random.choice(population_size, size=3, replace=False)
                winner = tournament[np.argmin(fitness[tournament])]
                selected_indices.append(winner)

            # Crossover and mutation
            for i in range(0, n_offspring - 1, 2):
                parent1 = population[selected_indices[i]].copy()
                parent2 = population[selected_indices[i + 1]].copy()

                # Crossover
                if np.random.random() < crossover_rate:
                    alpha = np.random.random()
                    child1 = alpha * parent1 + (1 - alpha) * parent2
                    child2 = (1 - alpha) * parent1 + alpha * parent2
                else:
                    child1, child2 = parent1.copy(), parent2.copy()

                # Mutation
                for child in [child1, child2]:
                    if np.random.random() < mutation_rate:
                        mutation_idx = np.random.randint(n_dims)
                        child[mutation_idx] = np.random.uniform(
                            bounds_array[mutation_idx, 0],
                            bounds_array[mutation_idx, 1]
                        )

                    # Apply bounds
                    child = np.clip(child, bounds_array[:, 0], bounds_array[:, 1])
                    new_population.append(child)

            # Handle odd number of offspring
            if len(new_population) < population_size:
                parent = population[selected_indices[-1]].copy()
                if np.random.random() < mutation_rate:
                    mutation_idx = np.random.randint(n_dims)
                    parent[mutation_idx] = np.random.uniform(
                        bounds_array[mutation_idx, 0],
                        bounds_array[mutation_idx, 1]
                    )
                parent = np.clip(parent, bounds_array[:, 0], bounds_array[:, 1])
                new_population.append(parent)

            population = np.array(new_population[:population_size])

        # Final evaluation
        fitness = np.array([objective_func(ind) for ind in population])
        best_idx = np.argmin(fitness)

        return OptimizationResult(
            x=population[best_idx],
            fun=fitness[best_idx],
            nit=generations,
            success=True,
            message="Optimization completed",
            history=best_fitness_history
        )

    def simulated_annealing(
        self,
        objective_func: Callable,
        x0: np.ndarray,
        bounds: List[Tuple[float, float]],
        initial_temp: float = 100.0,
        cooling_rate: float = 0.95,
        iterations: int = 1000,
        step_size: float = 0.1
    ) -> OptimizationResult:
        """
        Simulated Annealing optimization.

        Args:
            objective_func: Function to minimize
            x0: Initial solution
            bounds: List of (min, max) tuples
            initial_temp: Initial temperature
            cooling_rate: Temperature reduction factor (0 < rate < 1)
            iterations: Number of iterations
            step_size: Step size for neighbor generation

        Returns:
            OptimizationResult containing best solution
        """
        x_current = np.array(x0).copy()
        f_current = objective_func(x_current)

        x_best = x_current.copy()
        f_best = f_current

        temp = initial_temp
        history = [f_best]
        bounds_array = np.array(bounds)

        for iteration in range(iterations):
            # Generate neighbor
            x_neighbor = x_current + np.random.normal(0, step_size, size=len(x_current))
            x_neighbor = np.clip(x_neighbor, bounds_array[:, 0], bounds_array[:, 1])

            f_neighbor = objective_func(x_neighbor)

            # Accept or reject
            delta = f_neighbor - f_current

            if delta < 0 or np.random.random() < np.exp(-delta / temp):
                x_current = x_neighbor
                f_current = f_neighbor

                if f_current < f_best:
                    x_best = x_current.copy()
                    f_best = f_current

            # Cool down
            temp *= cooling_rate
            history.append(f_best)

            if self.verbose and (iteration + 1) % 200 == 0:
                print(f"Iteration {iteration + 1}/{iterations}: "
                      f"Best = {f_best:.6f}, Temp = {temp:.4f}")

        return OptimizationResult(
            x=x_best,
            fun=f_best,
            nit=iterations,
            success=True,
            message="Optimization completed",
            history=history
        )

    def particle_swarm_optimization(
        self,
        objective_func: Callable,
        bounds: List[Tuple[float, float]],
        n_particles: int = 30,
        iterations: int = 100,
        w: float = 0.7,
        c1: float = 1.5,
        c2: float = 1.5
    ) -> OptimizationResult:
        """
        Particle Swarm Optimization.

        Args:
            objective_func: Function to minimize
            bounds: List of (min, max) tuples
            n_particles: Number of particles
            iterations: Number of iterations
            w: Inertia weight
            c1: Cognitive parameter
            c2: Social parameter

        Returns:
            OptimizationResult containing best solution
        """
        n_dims = len(bounds)
        bounds_array = np.array(bounds)

        # Initialize particles
        particles = np.random.uniform(
            bounds_array[:, 0],
            bounds_array[:, 1],
            size=(n_particles, n_dims)
        )

        velocities = np.random.uniform(
            -1, 1, size=(n_particles, n_dims)
        )

        # Personal best
        p_best = particles.copy()
        p_best_fitness = np.array([objective_func(p) for p in particles])

        # Global best
        g_best_idx = np.argmin(p_best_fitness)
        g_best = p_best[g_best_idx].copy()
        g_best_fitness = p_best_fitness[g_best_idx]

        history = [g_best_fitness]

        for iteration in range(iterations):
            for i in range(n_particles):
                # Update velocity
                r1, r2 = np.random.random(2)

                velocities[i] = (
                    w * velocities[i] +
                    c1 * r1 * (p_best[i] - particles[i]) +
                    c2 * r2 * (g_best - particles[i])
                )

                # Update position
                particles[i] += velocities[i]

                # Apply bounds
                particles[i] = np.clip(particles[i], bounds_array[:, 0], bounds_array[:, 1])

                # Evaluate fitness
                fitness = objective_func(particles[i])

                # Update personal best
                if fitness < p_best_fitness[i]:
                    p_best[i] = particles[i].copy()
                    p_best_fitness[i] = fitness

                    # Update global best
                    if fitness < g_best_fitness:
                        g_best = particles[i].copy()
                        g_best_fitness = fitness

            history.append(g_best_fitness)

            if self.verbose and (iteration + 1) % 20 == 0:
                print(f"Iteration {iteration + 1}/{iterations}: Best = {g_best_fitness:.6f}")

        return OptimizationResult(
            x=g_best,
            fun=g_best_fitness,
            nit=iterations,
            success=True,
            message="Optimization completed",
            history=history
        )

    def gradient_descent(
        self,
        objective_func: Callable,
        gradient_func: Callable,
        x0: np.ndarray,
        learning_rate: float = 0.01,
        iterations: int = 1000,
        method: str = 'standard',
        momentum: float = 0.9,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8
    ) -> OptimizationResult:
        """
        Gradient descent with various variants.

        Args:
            objective_func: Function to minimize
            gradient_func: Function computing gradient
            x0: Initial point
            learning_rate: Learning rate
            iterations: Number of iterations
            method: 'standard', 'momentum', 'adam'
            momentum: Momentum parameter (for momentum method)
            beta1: First moment decay (for Adam)
            beta2: Second moment decay (for Adam)
            epsilon: Small constant for numerical stability

        Returns:
            OptimizationResult containing solution
        """
        x = np.array(x0).copy()
        history = [objective_func(x)]

        if method == 'momentum':
            velocity = np.zeros_like(x)
        elif method == 'adam':
            m = np.zeros_like(x)  # First moment
            v = np.zeros_like(x)  # Second moment

        for t in range(iterations):
            grad = gradient_func(x)

            if method == 'standard':
                x -= learning_rate * grad

            elif method == 'momentum':
                velocity = momentum * velocity - learning_rate * grad
                x += velocity

            elif method == 'adam':
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * (grad ** 2)

                m_hat = m / (1 - beta1 ** (t + 1))
                v_hat = v / (1 - beta2 ** (t + 1))

                x -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

            else:
                raise ValueError(f"Unknown method: {method}")

            f_val = objective_func(x)
            history.append(f_val)

            if self.verbose and (t + 1) % 200 == 0:
                print(f"Iteration {t + 1}/{iterations}: f(x) = {f_val:.6f}")

        return OptimizationResult(
            x=x,
            fun=objective_func(x),
            nit=iterations,
            success=True,
            message="Optimization completed",
            history=history
        )

    def multi_objective_optimization(
        self,
        objective_funcs: List[Callable],
        bounds: List[Tuple[float, float]],
        n_points: int = 100,
        method: str = 'genetic'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Multi-objective optimization to find Pareto front.

        Args:
            objective_funcs: List of objective functions
            bounds: List of (min, max) tuples
            n_points: Number of Pareto optimal points to find
            method: Optimization method

        Returns:
            Tuple of (pareto_front_x, pareto_front_y)
        """
        if self.verbose:
            print(f"Finding Pareto front with {n_points} points...")

        n_objectives = len(objective_funcs)
        solutions = []
        objective_values = []

        # Generate multiple solutions with different weight combinations
        for _ in range(n_points):
            # Random weights
            weights = np.random.random(n_objectives)
            weights /= weights.sum()

            # Weighted sum objective
            def weighted_objective(x):
                return sum(w * f(x) for w, f in zip(weights, objective_funcs))

            # Optimize
            if method == 'genetic':
                result = self.genetic_algorithm(
                    weighted_objective,
                    bounds,
                    population_size=30,
                    generations=50
                )
            else:
                result = self.particle_swarm_optimization(
                    weighted_objective,
                    bounds,
                    n_particles=20,
                    iterations=50
                )

            solutions.append(result.x)
            objective_values.append([f(result.x) for f in objective_funcs])

        solutions = np.array(solutions)
        objective_values = np.array(objective_values)

        # Filter for Pareto optimal solutions
        pareto_mask = self._is_pareto_efficient(objective_values)
        pareto_solutions = solutions[pareto_mask]
        pareto_objectives = objective_values[pareto_mask]

        if self.verbose:
            print(f"Found {len(pareto_solutions)} Pareto optimal solutions")

        return pareto_solutions, pareto_objectives

    def _is_pareto_efficient(self, costs: np.ndarray) -> np.ndarray:
        """
        Find Pareto efficient points.

        Args:
            costs: Array of objective values (n_points, n_objectives)

        Returns:
            Boolean array indicating Pareto efficient points
        """
        is_efficient = np.ones(len(costs), dtype=bool)

        for i, c in enumerate(costs):
            if is_efficient[i]:
                # Remove dominated points
                is_efficient[is_efficient] = np.any(costs[is_efficient] < c, axis=1)
                is_efficient[i] = True

        return is_efficient

    def portfolio_optimization(
        self,
        returns: np.ndarray,
        risk_free_rate: float = 0.02,
        target_return: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Portfolio optimization (Markowitz).

        Args:
            returns: Historical returns (n_samples, n_assets)
            risk_free_rate: Risk-free rate
            target_return: Target portfolio return (optional)

        Returns:
            Dictionary with optimal weights and metrics
        """
        returns = np.array(returns)
        n_assets = returns.shape[1]

        # Calculate expected returns and covariance
        expected_returns = np.mean(returns, axis=0)
        cov_matrix = np.cov(returns.T)

        if self.verbose:
            print("Portfolio Optimization:")
            print(f"  Assets: {n_assets}")
            print(f"  Expected returns: {expected_returns}")

        # Minimize variance
        def portfolio_variance(weights):
            return weights @ cov_matrix @ weights

        # Constraints: weights sum to 1
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]

        # Optional: target return constraint
        if target_return is not None:
            constraints.append({
                'type': 'eq',
                'fun': lambda w: w @ expected_returns - target_return
            })

        # Bounds: weights between 0 and 1 (long only)
        bounds = [(0, 1) for _ in range(n_assets)]

        # Initial guess
        x0 = np.ones(n_assets) / n_assets

        # Optimize
        result = minimize(
            portfolio_variance,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        optimal_weights = result.x
        optimal_return = optimal_weights @ expected_returns
        optimal_risk = np.sqrt(portfolio_variance(optimal_weights))
        sharpe_ratio = (optimal_return - risk_free_rate) / optimal_risk

        if self.verbose:
            print(f"  Optimal return: {optimal_return:.4f}")
            print(f"  Optimal risk: {optimal_risk:.4f}")
            print(f"  Sharpe ratio: {sharpe_ratio:.4f}")

        return {
            'weights': optimal_weights,
            'return': optimal_return,
            'risk': optimal_risk,
            'sharpe_ratio': sharpe_ratio,
            'success': result.success
        }

    def resource_allocation(
        self,
        values: np.ndarray,
        costs: np.ndarray,
        budget: float,
        method: str = 'knapsack'
    ) -> Dict[str, Any]:
        """
        Optimize resource allocation.

        Args:
            values: Value of each item
            costs: Cost of each item
            budget: Total budget
            method: 'knapsack' or 'fractional'

        Returns:
            Dictionary with allocation results
        """
        n_items = len(values)

        if method == 'fractional':
            # Fractional knapsack (greedy)
            ratios = values / costs
            sorted_indices = np.argsort(ratios)[::-1]

            allocation = np.zeros(n_items)
            remaining_budget = budget

            for idx in sorted_indices:
                if costs[idx] <= remaining_budget:
                    allocation[idx] = 1.0
                    remaining_budget -= costs[idx]
                else:
                    allocation[idx] = remaining_budget / costs[idx]
                    break

        elif method == 'knapsack':
            # 0-1 knapsack (dynamic programming approximation via linear programming)
            # Relaxation: allow fractional selection, then round
            c = -values  # Negative because linprog minimizes

            result = linprog(
                c=c,
                A_ub=costs.reshape(1, -1),
                b_ub=[budget],
                bounds=[(0, 1) for _ in range(n_items)],
                method='highs'
            )

            allocation = np.round(result.x).astype(int)

            # Ensure budget constraint
            while np.sum(allocation * costs) > budget:
                # Remove least valuable item
                selected = np.where(allocation > 0)[0]
                if len(selected) == 0:
                    break
                ratios = values[selected] / costs[selected]
                remove_idx = selected[np.argmin(ratios)]
                allocation[remove_idx] = 0

        else:
            raise ValueError(f"Unknown method: {method}")

        total_value = np.sum(allocation * values)
        total_cost = np.sum(allocation * costs)

        if self.verbose:
            print(f"Resource Allocation ({method}):")
            print(f"  Total value: {total_value:.2f}")
            print(f"  Total cost: {total_cost:.2f}")
            print(f"  Budget: {budget:.2f}")
            print(f"  Items selected: {np.sum(allocation > 0)}")

        return {
            'allocation': allocation,
            'total_value': total_value,
            'total_cost': total_cost,
            'budget': budget,
            'selected_items': np.where(allocation > 0)[0].tolist()
        }

    def plot_convergence(
        self,
        results: List[OptimizationResult],
        labels: List[str],
        figsize: Tuple[int, int] = (10, 6)
    ) -> plt.Figure:
        """
        Plot convergence history of multiple optimization runs.

        Args:
            results: List of OptimizationResults
            labels: Labels for each result
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)

        for result, label in zip(results, labels):
            if result.history:
                ax.plot(result.history, linewidth=2, label=label, alpha=0.8)

        ax.set_xlabel('Iteration')
        ax.set_ylabel('Objective Value')
        ax.set_title('Optimization Convergence')
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_yscale('log')

        plt.tight_layout()
        return fig

    def plot_pareto_front(
        self,
        pareto_objectives: np.ndarray,
        labels: List[str],
        figsize: Tuple[int, int] = (10, 6)
    ) -> plt.Figure:
        """
        Plot Pareto front for bi-objective optimization.

        Args:
            pareto_objectives: Pareto optimal objective values (n_points, 2)
            labels: Labels for objectives
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Sort by first objective
        sorted_indices = np.argsort(pareto_objectives[:, 0])
        sorted_front = pareto_objectives[sorted_indices]

        ax.scatter(sorted_front[:, 0], sorted_front[:, 1],
                  s=100, alpha=0.6, edgecolors='black', linewidths=2)
        ax.plot(sorted_front[:, 0], sorted_front[:, 1],
               'r--', alpha=0.5, linewidth=2, label='Pareto Front')

        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        ax.set_title('Pareto Front')
        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout()
        return fig


def demo():
    """
    Demonstrate the OptimizationToolkit capabilities.
    """
    print("=" * 80)
    print("OPTIMIZATION METHODS TOOLKIT DEMO")
    print("=" * 80)

    opt = OptimizationToolkit(random_state=42, verbose=True)

    # 1. Linear Programming
    print("\n1. LINEAR PROGRAMMING")
    print("-" * 80)
    print("Maximize: 3x + 4y")
    print("Subject to: x + 2y <= 14")
    print("           3x + y <= 18")
    print("           x, y >= 0")

    # Convert to minimization (negate coefficients)
    c = np.array([-3, -4])
    A_ub = np.array([[1, 2], [3, 1]])
    b_ub = np.array([14, 18])
    bounds = [(0, None), (0, None)]

    lp_result = opt.linear_programming(c, A_ub, b_ub, bounds=bounds)
    print(f"Optimal solution: x = {lp_result.x[0]:.4f}, y = {lp_result.x[1]:.4f}")
    print(f"Maximum value: {-lp_result.fun:.4f}")

    # 2. Test functions for metaheuristics
    print("\n2. OPTIMIZATION TEST FUNCTIONS")
    print("-" * 80)

    # Rastrigin function (has many local minima)
    def rastrigin(x):
        n = len(x)
        return 10 * n + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

    # Rosenbrock function (narrow curved valley)
    def rosenbrock(x):
        return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

    # Sphere function (simple convex)
    def sphere(x):
        return np.sum(x**2)

    bounds_2d = [(-5, 5), (-5, 5)]
    bounds_5d = [(-5, 5)] * 5

    # 3. Genetic Algorithm
    print("\n3. GENETIC ALGORITHM (Rastrigin function)")
    print("-" * 80)
    ga_result = opt.genetic_algorithm(
        rastrigin,
        bounds_2d,
        population_size=50,
        generations=100,
        mutation_rate=0.1,
        crossover_rate=0.8
    )
    print(f"Solution: {ga_result.x}")
    print(f"Objective value: {ga_result.fun:.6f}")
    print(f"Global minimum: [0, 0] with value 0")

    # 4. Simulated Annealing
    print("\n4. SIMULATED ANNEALING (Rosenbrock function)")
    print("-" * 80)
    x0 = np.array([-2.0, -2.0])
    sa_result = opt.simulated_annealing(
        rosenbrock,
        x0,
        bounds_2d,
        initial_temp=100.0,
        cooling_rate=0.95,
        iterations=1000
    )
    print(f"Solution: {sa_result.x}")
    print(f"Objective value: {sa_result.fun:.6f}")
    print(f"Global minimum: [1, 1] with value 0")

    # 5. Particle Swarm Optimization
    print("\n5. PARTICLE SWARM OPTIMIZATION (Sphere function)")
    print("-" * 80)
    pso_result = opt.particle_swarm_optimization(
        sphere,
        bounds_5d,
        n_particles=30,
        iterations=100
    )
    print(f"Solution: {pso_result.x}")
    print(f"Objective value: {pso_result.fun:.6f}")
    print(f"Global minimum: [0, 0, 0, 0, 0] with value 0")

    # 6. Gradient Descent variants
    print("\n6. GRADIENT DESCENT VARIANTS (Sphere function)")
    print("-" * 80)

    def sphere_gradient(x):
        return 2 * x

    x0 = np.array([3.0, 3.0, 3.0])
    bounds_3d = [(-5, 5)] * 3

    print("\n  Standard Gradient Descent:")
    gd_result = opt.gradient_descent(
        sphere,
        sphere_gradient,
        x0,
        learning_rate=0.1,
        iterations=100,
        method='standard'
    )
    print(f"  Solution: {gd_result.x}")
    print(f"  Objective value: {gd_result.fun:.6f}")

    print("\n  Momentum:")
    momentum_result = opt.gradient_descent(
        sphere,
        sphere_gradient,
        x0,
        learning_rate=0.1,
        iterations=100,
        method='momentum',
        momentum=0.9
    )
    print(f"  Solution: {momentum_result.x}")
    print(f"  Objective value: {momentum_result.fun:.6f}")

    print("\n  Adam:")
    adam_result = opt.gradient_descent(
        sphere,
        sphere_gradient,
        x0,
        learning_rate=0.1,
        iterations=100,
        method='adam'
    )
    print(f"  Solution: {adam_result.x}")
    print(f"  Objective value: {adam_result.fun:.6f}")

    # 7. Multi-objective Optimization
    print("\n7. MULTI-OBJECTIVE OPTIMIZATION (Pareto Front)")
    print("-" * 80)
    print("Objectives: minimize f1(x) = x^2, minimize f2(x) = (x-2)^2")

    def obj1(x):
        return x[0]**2

    def obj2(x):
        return (x[0] - 2)**2

    pareto_x, pareto_y = opt.multi_objective_optimization(
        [obj1, obj2],
        [(-3, 3)],
        n_points=50,
        method='genetic'
    )
    print(f"Pareto front contains {len(pareto_x)} solutions")
    print(f"Objective ranges: f1=[{pareto_y[:, 0].min():.4f}, {pareto_y[:, 0].max():.4f}], "
          f"f2=[{pareto_y[:, 1].min():.4f}, {pareto_y[:, 1].max():.4f}]")

    # 8. Portfolio Optimization
    print("\n8. PORTFOLIO OPTIMIZATION")
    print("-" * 80)

    # Generate synthetic returns
    np.random.seed(42)
    n_days = 252
    n_assets = 4

    # Expected returns and volatilities
    expected_returns = np.array([0.08, 0.12, 0.15, 0.10])
    volatilities = np.array([0.15, 0.25, 0.30, 0.20])

    # Generate correlated returns
    correlation = np.array([
        [1.0, 0.5, 0.3, 0.4],
        [0.5, 1.0, 0.6, 0.5],
        [0.3, 0.6, 1.0, 0.7],
        [0.4, 0.5, 0.7, 1.0]
    ])

    cov_matrix = np.outer(volatilities, volatilities) * correlation
    returns = np.random.multivariate_normal(expected_returns / 252, cov_matrix / 252, n_days)

    portfolio_result = opt.portfolio_optimization(returns, risk_free_rate=0.02)
    print(f"Optimal weights: {portfolio_result['weights']}")
    print(f"Expected return: {portfolio_result['return']:.4f}")
    print(f"Risk (std): {portfolio_result['risk']:.4f}")
    print(f"Sharpe ratio: {portfolio_result['sharpe_ratio']:.4f}")

    # 9. Resource Allocation
    print("\n9. RESOURCE ALLOCATION (Knapsack Problem)")
    print("-" * 80)

    values = np.array([60, 100, 120, 80, 90])
    costs = np.array([10, 20, 30, 15, 25])
    budget = 50

    print(f"Items: {len(values)}")
    print(f"Values: {values}")
    print(f"Costs: {costs}")
    print(f"Budget: {budget}")

    alloc_result = opt.resource_allocation(values, costs, budget, method='knapsack')
    print(f"Selected items: {alloc_result['selected_items']}")
    print(f"Total value: {alloc_result['total_value']:.2f}")
    print(f"Total cost: {alloc_result['total_cost']:.2f}")

    # Visualizations
    print("\n10. CREATING VISUALIZATIONS")
    print("-" * 80)

    # Convergence comparison
    fig1 = opt.plot_convergence(
        [ga_result, sa_result, pso_result],
        ['Genetic Algorithm', 'Simulated Annealing', 'Particle Swarm']
    )
    plt.savefig('/tmp/optimization_convergence.png', dpi=150, bbox_inches='tight')
    print("Saved: /tmp/optimization_convergence.png")
    plt.close()

    # Pareto front
    fig2 = opt.plot_pareto_front(pareto_y, ['$f_1(x) = x^2$', '$f_2(x) = (x-2)^2$'])
    plt.savefig('/tmp/pareto_front.png', dpi=150, bbox_inches='tight')
    print("Saved: /tmp/pareto_front.png")
    plt.close()

    # Gradient descent comparison
    fig3, ax = plt.subplots(figsize=(10, 6))
    ax.plot(gd_result.history, label='Standard GD', linewidth=2)
    ax.plot(momentum_result.history, label='Momentum', linewidth=2)
    ax.plot(adam_result.history, label='Adam', linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Objective Value')
    ax.set_title('Gradient Descent Variants Comparison')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('/tmp/gradient_descent_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved: /tmp/gradient_descent_comparison.png")
    plt.close()

    print("\n" + "=" * 80)
    print("DEMO COMPLETE!")
    print("=" * 80)
    print("\nKey Insights:")
    print("1. Linear programming efficiently solves convex optimization problems")
    print("2. Genetic algorithms excel at avoiding local minima")
    print("3. Simulated annealing balances exploration and exploitation")
    print("4. PSO leverages swarm intelligence for global optimization")
    print("5. Gradient descent variants (Adam, Momentum) improve convergence")
    print("6. Multi-objective optimization finds trade-off solutions (Pareto front)")
    print("7. Portfolio optimization balances risk and return")
    print("8. Resource allocation maximizes value under constraints")
    print("\nAll visualizations saved to /tmp/")


if __name__ == "__main__":
    demo()
