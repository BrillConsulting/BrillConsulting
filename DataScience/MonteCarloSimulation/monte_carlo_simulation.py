"""
Monte Carlo Simulation Toolkit
===============================

Advanced Monte Carlo methods and stochastic simulation:
- Monte Carlo integration
- Risk analysis and VaR (Value at Risk)
- Sensitivity analysis and scenario testing
- Option pricing (Black-Scholes, Binomial)
- Portfolio optimization under uncertainty
- Stochastic process simulation
- Markov Chain Monte Carlo (MCMC)
- Bootstrap and resampling methods

Author: Brill Consulting
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import minimize
from typing import Dict, List, Tuple, Optional, Callable, Union
import warnings
warnings.filterwarnings('ignore')


class MonteCarloSimulation:
    """Monte Carlo simulation and stochastic modeling toolkit."""

    def __init__(self, random_state: int = 42, n_simulations: int = 10000):
        """Initialize Monte Carlo simulation toolkit."""
        self.random_state = random_state
        self.n_simulations = n_simulations
        np.random.seed(random_state)
        self.simulations = {}

    def monte_carlo_integration(self, func: Callable, bounds: List[Tuple[float, float]],
                               n_samples: Optional[int] = None) -> Dict:
        """
        Estimate integral using Monte Carlo method.

        Args:
            func: Function to integrate
            bounds: List of (lower, upper) bounds for each dimension
            n_samples: Number of samples (default: self.n_simulations)

        Returns:
            Dictionary with integral estimate and statistics
        """
        if n_samples is None:
            n_samples = self.n_simulations

        n_dims = len(bounds)

        # Generate random samples
        samples = np.zeros((n_samples, n_dims))
        volume = 1.0

        for i, (lower, upper) in enumerate(bounds):
            samples[:, i] = np.random.uniform(lower, upper, n_samples)
            volume *= (upper - lower)

        # Evaluate function at samples
        values = np.array([func(*sample) for sample in samples])

        # Estimate integral
        integral_estimate = volume * np.mean(values)
        standard_error = volume * np.std(values) / np.sqrt(n_samples)

        return {
            'integral_estimate': integral_estimate,
            'standard_error': standard_error,
            'confidence_interval_95': (
                integral_estimate - 1.96 * standard_error,
                integral_estimate + 1.96 * standard_error
            ),
            'n_samples': n_samples,
            'function_values': values
        }

    def value_at_risk(self, returns: np.ndarray, confidence_level: float = 0.95,
                     method: str = 'historical') -> Dict:
        """
        Calculate Value at Risk (VaR) using different methods.

        Args:
            returns: Array of returns or profit/loss values
            confidence_level: Confidence level (e.g., 0.95 for 95%)
            method: 'historical', 'parametric', or 'monte_carlo'

        Returns:
            Dictionary with VaR estimates
        """
        alpha = 1 - confidence_level

        if method == 'historical':
            # Historical VaR (non-parametric)
            var = np.percentile(returns, alpha * 100)

        elif method == 'parametric':
            # Parametric VaR (assuming normal distribution)
            mu = np.mean(returns)
            sigma = np.std(returns)
            var = mu + sigma * stats.norm.ppf(alpha)

        elif method == 'monte_carlo':
            # Monte Carlo VaR
            mu = np.mean(returns)
            sigma = np.std(returns)
            simulated_returns = np.random.normal(mu, sigma, self.n_simulations)
            var = np.percentile(simulated_returns, alpha * 100)

        else:
            raise ValueError(f"Unknown method: {method}")

        # Conditional VaR (Expected Shortfall)
        cvar = np.mean(returns[returns <= var])

        return {
            'var': var,
            'cvar': cvar,
            'confidence_level': confidence_level,
            'method': method,
            'mean_return': np.mean(returns),
            'volatility': np.std(returns)
        }

    def sensitivity_analysis(self, model: Callable, base_params: Dict[str, float],
                           param_ranges: Dict[str, Tuple[float, float]],
                           n_samples: Optional[int] = None) -> pd.DataFrame:
        """
        Perform global sensitivity analysis using Monte Carlo sampling.

        Args:
            model: Function that takes parameters and returns output
            base_params: Base parameter values
            param_ranges: Dictionary of parameter ranges to vary
            n_samples: Number of samples per parameter

        Returns:
            DataFrame with sensitivity analysis results
        """
        if n_samples is None:
            n_samples = self.n_simulations // len(param_ranges)

        results = []

        # One-at-a-time sensitivity
        for param_name, (low, high) in param_ranges.items():
            param_values = np.linspace(low, high, n_samples)

            for value in param_values:
                params = base_params.copy()
                params[param_name] = value

                output = model(**params)

                results.append({
                    'parameter': param_name,
                    'value': value,
                    'output': output
                })

        results_df = pd.DataFrame(results)

        # Calculate sensitivity indices
        sensitivity_indices = {}
        base_output = model(**base_params)

        for param_name in param_ranges.keys():
            param_results = results_df[results_df['parameter'] == param_name]
            output_variance = param_results['output'].var()
            sensitivity_indices[param_name] = output_variance

        self.simulations['sensitivity'] = {
            'results': results_df,
            'indices': sensitivity_indices
        }

        return results_df

    def geometric_brownian_motion(self, S0: float, mu: float, sigma: float,
                                 T: float, n_steps: int,
                                 n_paths: Optional[int] = None) -> np.ndarray:
        """
        Simulate Geometric Brownian Motion (stock price paths).

        Args:
            S0: Initial price
            mu: Drift (expected return)
            sigma: Volatility
            T: Time horizon
            n_steps: Number of time steps
            n_paths: Number of paths to simulate

        Returns:
            Array of simulated paths (n_paths x n_steps+1)
        """
        if n_paths is None:
            n_paths = self.n_simulations

        dt = T / n_steps
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = S0

        for i in range(1, n_steps + 1):
            z = np.random.standard_normal(n_paths)
            paths[:, i] = paths[:, i-1] * np.exp((mu - 0.5 * sigma**2) * dt +
                                                 sigma * np.sqrt(dt) * z)

        self.simulations['gbm'] = paths

        return paths

    def black_scholes_option_pricing(self, S0: float, K: float, T: float,
                                    r: float, sigma: float, option_type: str = 'call') -> Dict:
        """
        Price European options using Black-Scholes formula with Monte Carlo validation.

        Args:
            S0: Current stock price
            K: Strike price
            T: Time to maturity
            r: Risk-free rate
            sigma: Volatility
            option_type: 'call' or 'put'

        Returns:
            Dictionary with option prices
        """
        # Analytical Black-Scholes
        d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if option_type == 'call':
            analytical_price = S0 * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)
        else:  # put
            analytical_price = K * np.exp(-r * T) * stats.norm.cdf(-d2) - S0 * stats.norm.cdf(-d1)

        # Monte Carlo simulation
        paths = self.geometric_brownian_motion(S0, r, sigma, T, n_steps=252, n_paths=self.n_simulations)
        terminal_prices = paths[:, -1]

        if option_type == 'call':
            payoffs = np.maximum(terminal_prices - K, 0)
        else:  # put
            payoffs = np.maximum(K - terminal_prices, 0)

        mc_price = np.exp(-r * T) * np.mean(payoffs)
        mc_std = np.std(payoffs) * np.exp(-r * T) / np.sqrt(self.n_simulations)

        # Greeks (Delta via finite differences)
        dS = 0.01 * S0
        paths_up = self.geometric_brownian_motion(S0 + dS, r, sigma, T, n_steps=252, n_paths=1000)
        paths_down = self.geometric_brownian_motion(S0 - dS, r, sigma, T, n_steps=252, n_paths=1000)

        if option_type == 'call':
            payoffs_up = np.maximum(paths_up[:, -1] - K, 0)
            payoffs_down = np.maximum(paths_down[:, -1] - K, 0)
        else:
            payoffs_up = np.maximum(K - paths_up[:, -1], 0)
            payoffs_down = np.maximum(K - paths_down[:, -1], 0)

        price_up = np.exp(-r * T) * np.mean(payoffs_up)
        price_down = np.exp(-r * T) * np.mean(payoffs_down)
        delta = (price_up - price_down) / (2 * dS)

        return {
            'analytical_price': analytical_price,
            'monte_carlo_price': mc_price,
            'monte_carlo_std': mc_std,
            'delta': delta,
            'option_type': option_type,
            'parameters': {
                'S0': S0, 'K': K, 'T': T, 'r': r, 'sigma': sigma
            }
        }

    def portfolio_simulation(self, returns: np.ndarray, weights: np.ndarray,
                            initial_value: float = 1000000,
                            n_days: int = 252) -> Dict:
        """
        Simulate portfolio returns and calculate risk metrics.

        Args:
            returns: Historical returns matrix (n_samples x n_assets)
            weights: Portfolio weights
            initial_value: Initial portfolio value
            n_days: Number of days to simulate

        Returns:
            Dictionary with simulation results
        """
        n_assets = returns.shape[1]

        if len(weights) != n_assets:
            raise ValueError("Weights must match number of assets")

        # Calculate portfolio statistics
        mean_returns = np.mean(returns, axis=0)
        cov_matrix = np.cov(returns.T)

        portfolio_mean = np.dot(weights, mean_returns)
        portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

        # Simulate portfolio paths
        portfolio_paths = np.zeros((self.n_simulations, n_days + 1))
        portfolio_paths[:, 0] = initial_value

        for i in range(1, n_days + 1):
            # Generate correlated returns
            z = np.random.multivariate_normal(mean_returns, cov_matrix, self.n_simulations)
            daily_returns = np.dot(z, weights)
            portfolio_paths[:, i] = portfolio_paths[:, i-1] * (1 + daily_returns)

        # Calculate risk metrics
        final_values = portfolio_paths[:, -1]
        returns_pct = (final_values - initial_value) / initial_value

        var_95 = np.percentile(returns_pct, 5)
        cvar_95 = np.mean(returns_pct[returns_pct <= var_95])

        prob_loss = np.mean(final_values < initial_value)
        max_drawdown = np.max(initial_value - np.min(portfolio_paths, axis=1))

        return {
            'portfolio_paths': portfolio_paths,
            'expected_return': portfolio_mean,
            'volatility': portfolio_std,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'probability_loss': prob_loss,
            'max_drawdown': max_drawdown,
            'final_value_percentiles': {
                '5th': np.percentile(final_values, 5),
                '25th': np.percentile(final_values, 25),
                '50th': np.percentile(final_values, 50),
                '75th': np.percentile(final_values, 75),
                '95th': np.percentile(final_values, 95)
            }
        }

    def bootstrap_confidence_interval(self, data: np.ndarray, statistic: Callable,
                                     n_bootstrap: Optional[int] = None,
                                     confidence_level: float = 0.95) -> Dict:
        """
        Calculate bootstrap confidence intervals for a statistic.

        Args:
            data: Original data
            statistic: Function to calculate statistic (e.g., np.mean)
            n_bootstrap: Number of bootstrap samples
            confidence_level: Confidence level

        Returns:
            Dictionary with bootstrap results
        """
        if n_bootstrap is None:
            n_bootstrap = self.n_simulations

        # Calculate observed statistic
        observed_stat = statistic(data)

        # Bootstrap sampling
        bootstrap_stats = np.zeros(n_bootstrap)
        n = len(data)

        for i in range(n_bootstrap):
            bootstrap_sample = np.random.choice(data, size=n, replace=True)
            bootstrap_stats[i] = statistic(bootstrap_sample)

        # Percentile method
        alpha = 1 - confidence_level
        ci_lower = np.percentile(bootstrap_stats, alpha/2 * 100)
        ci_upper = np.percentile(bootstrap_stats, (1 - alpha/2) * 100)

        # Bootstrap standard error
        bootstrap_se = np.std(bootstrap_stats)

        return {
            'observed_statistic': observed_stat,
            'bootstrap_mean': np.mean(bootstrap_stats),
            'bootstrap_std': bootstrap_se,
            'confidence_interval': (ci_lower, ci_upper),
            'confidence_level': confidence_level,
            'bootstrap_distribution': bootstrap_stats
        }

    def scenario_analysis(self, model: Callable, scenarios: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        """
        Perform scenario analysis by evaluating model under different scenarios.

        Args:
            model: Model function that takes parameters
            scenarios: Dictionary of scenario names to parameter dictionaries

        Returns:
            DataFrame with scenario results
        """
        results = []

        for scenario_name, params in scenarios.items():
            output = model(**params)

            result_dict = {'scenario': scenario_name}
            result_dict.update(params)
            result_dict['output'] = output

            results.append(result_dict)

        return pd.DataFrame(results)

    def latin_hypercube_sampling(self, distributions: List[Tuple[str, tuple]],
                                 n_samples: Optional[int] = None) -> np.ndarray:
        """
        Generate Latin Hypercube samples for efficient space-filling sampling.

        Args:
            distributions: List of (distribution_name, parameters) tuples
                          e.g., [('norm', (0, 1)), ('uniform', (0, 1))]
            n_samples: Number of samples

        Returns:
            Array of samples (n_samples x n_dimensions)
        """
        if n_samples is None:
            n_samples = self.n_simulations

        n_dims = len(distributions)
        samples = np.zeros((n_samples, n_dims))

        for dim, (dist_name, params) in enumerate(distributions):
            # Generate LHS samples in [0, 1]
            intervals = np.arange(n_samples) / n_samples
            random_offsets = np.random.uniform(0, 1/n_samples, n_samples)
            lhs_samples = intervals + random_offsets
            np.random.shuffle(lhs_samples)

            # Transform to target distribution
            if dist_name == 'norm':
                mu, sigma = params
                samples[:, dim] = stats.norm.ppf(lhs_samples, mu, sigma)
            elif dist_name == 'uniform':
                low, high = params
                samples[:, dim] = stats.uniform.ppf(lhs_samples, low, high - low)
            elif dist_name == 'lognorm':
                mu, sigma = params
                samples[:, dim] = stats.lognorm.ppf(lhs_samples, sigma, scale=np.exp(mu))
            else:
                raise ValueError(f"Unknown distribution: {dist_name}")

        return samples

    def visualize_monte_carlo_paths(self, paths: np.ndarray, title: str = "Monte Carlo Paths",
                                   percentiles: List[int] = [5, 25, 50, 75, 95]) -> plt.Figure:
        """Visualize Monte Carlo simulation paths."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Plot sample paths
        n_display = min(100, paths.shape[0])
        time_steps = np.arange(paths.shape[1])

        for i in range(n_display):
            axes[0, 0].plot(time_steps, paths[i], alpha=0.1, color='blue')

        axes[0, 0].plot(time_steps, np.mean(paths, axis=0), color='red',
                       linewidth=2, label='Mean')
        axes[0, 0].set_xlabel('Time Step')
        axes[0, 0].set_ylabel('Value')
        axes[0, 0].set_title(f'{title}: Sample Paths')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)

        # Plot percentile bands
        for p in percentiles:
            percentile_path = np.percentile(paths, p, axis=0)
            axes[0, 1].plot(time_steps, percentile_path, label=f'{p}th percentile', linewidth=2)

        axes[0, 1].set_xlabel('Time Step')
        axes[0, 1].set_ylabel('Value')
        axes[0, 1].set_title('Percentile Bands')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)

        # Distribution of final values
        final_values = paths[:, -1]
        axes[1, 0].hist(final_values, bins=50, density=True, alpha=0.7,
                       edgecolor='black', color='skyblue')
        axes[1, 0].axvline(np.mean(final_values), color='red', linestyle='--',
                          linewidth=2, label=f'Mean: {np.mean(final_values):.2f}')
        axes[1, 0].axvline(np.median(final_values), color='green', linestyle='--',
                          linewidth=2, label=f'Median: {np.median(final_values):.2f}')
        axes[1, 0].set_xlabel('Final Value')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].set_title('Distribution of Final Values')
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)

        # Summary statistics
        summary_text = "Simulation Statistics\n"
        summary_text += "=" * 30 + "\n\n"
        summary_text += f"Number of paths: {paths.shape[0]}\n"
        summary_text += f"Time steps: {paths.shape[1]}\n\n"
        summary_text += f"Final Value Statistics:\n"
        summary_text += f"  Mean: {np.mean(final_values):.2f}\n"
        summary_text += f"  Median: {np.median(final_values):.2f}\n"
        summary_text += f"  Std Dev: {np.std(final_values):.2f}\n"
        summary_text += f"  Min: {np.min(final_values):.2f}\n"
        summary_text += f"  Max: {np.max(final_values):.2f}\n\n"
        summary_text += f"Percentiles:\n"
        for p in [5, 25, 50, 75, 95]:
            summary_text += f"  {p}th: {np.percentile(final_values, p):.2f}\n"

        axes[1, 1].text(0.1, 0.5, summary_text, transform=axes[1, 1].transAxes,
                       fontsize=10, verticalalignment='center', family='monospace',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        axes[1, 1].set_title('Summary Statistics')
        axes[1, 1].axis('off')

        plt.tight_layout()
        return fig

    def visualize_sensitivity_analysis(self, results_df: pd.DataFrame) -> plt.Figure:
        """Visualize sensitivity analysis results."""
        params = results_df['parameter'].unique()
        n_params = len(params)

        fig, axes = plt.subplots(1, n_params, figsize=(5*n_params, 5))
        if n_params == 1:
            axes = [axes]

        for i, param in enumerate(params):
            param_data = results_df[results_df['parameter'] == param]

            axes[i].plot(param_data['value'], param_data['output'],
                        linewidth=2, marker='o', markersize=4)
            axes[i].set_xlabel(param, fontsize=12)
            axes[i].set_ylabel('Output', fontsize=12)
            axes[i].set_title(f'Sensitivity to {param}')
            axes[i].grid(alpha=0.3)

        plt.tight_layout()
        return fig


def demo():
    """Demo Monte Carlo simulation toolkit."""
    np.random.seed(42)

    print("Monte Carlo Simulation Toolkit Demo")
    print("="*60)

    mc = MonteCarloSimulation(random_state=42, n_simulations=10000)

    # 1. Monte Carlo Integration
    print("\n1. Monte Carlo Integration")
    print("-" * 60)

    def test_function(x, y):
        return x**2 + y**2

    result = mc.monte_carlo_integration(test_function, [(0, 1), (0, 1)], n_samples=10000)
    print(f"Integral estimate: {result['integral_estimate']:.6f}")
    print(f"Standard error: {result['standard_error']:.6f}")
    print(f"95% CI: [{result['confidence_interval_95'][0]:.6f}, {result['confidence_interval_95'][1]:.6f}]")
    print(f"Analytical result: {2/3:.6f}")  # ∫∫(x²+y²)dxdy from 0 to 1

    # 2. Value at Risk
    print("\n2. Value at Risk (VaR) Analysis")
    print("-" * 60)
    returns = np.random.normal(0.001, 0.02, 1000)

    var_result = mc.value_at_risk(returns, confidence_level=0.95, method='historical')
    print(f"Historical VaR (95%): {var_result['var']:.4f}")
    print(f"Conditional VaR (CVaR): {var_result['cvar']:.4f}")
    print(f"Mean return: {var_result['mean_return']:.4f}")
    print(f"Volatility: {var_result['volatility']:.4f}")

    # 3. Geometric Brownian Motion
    print("\n3. Geometric Brownian Motion (Stock Price Simulation)")
    print("-" * 60)
    S0 = 100  # Initial price
    mu = 0.1  # 10% expected return
    sigma = 0.2  # 20% volatility
    T = 1.0  # 1 year
    n_steps = 252  # Daily steps

    paths = mc.geometric_brownian_motion(S0, mu, sigma, T, n_steps, n_paths=1000)
    print(f"Simulated {paths.shape[0]} paths with {paths.shape[1]} time steps")
    print(f"Initial price: ${S0:.2f}")
    print(f"Mean final price: ${np.mean(paths[:, -1]):.2f}")
    print(f"Median final price: ${np.median(paths[:, -1]):.2f}")

    # Visualize GBM
    fig1 = mc.visualize_monte_carlo_paths(paths, title="Stock Price Simulation (GBM)")
    fig1.savefig('monte_carlo_gbm.png', dpi=300, bbox_inches='tight')
    print("✓ Saved monte_carlo_gbm.png")
    plt.close()

    # 4. Black-Scholes Option Pricing
    print("\n4. Black-Scholes Option Pricing")
    print("-" * 60)
    option_result = mc.black_scholes_option_pricing(S0=100, K=100, T=1.0, r=0.05, sigma=0.2, option_type='call')
    print(f"Analytical price: ${option_result['analytical_price']:.4f}")
    print(f"Monte Carlo price: ${option_result['monte_carlo_price']:.4f}")
    print(f"MC std error: ${option_result['monte_carlo_std']:.4f}")
    print(f"Delta: {option_result['delta']:.4f}")

    # 5. Sensitivity Analysis
    print("\n5. Sensitivity Analysis")
    print("-" * 60)

    def simple_model(x, y, z):
        return x**2 + 2*y + 0.5*z

    base_params = {'x': 1.0, 'y': 2.0, 'z': 3.0}
    param_ranges = {
        'x': (0.5, 2.0),
        'y': (1.0, 4.0),
        'z': (2.0, 5.0)
    }

    sensitivity_df = mc.sensitivity_analysis(simple_model, base_params, param_ranges, n_samples=50)
    print(f"\nSensitivity indices:")
    for param, variance in mc.simulations['sensitivity']['indices'].items():
        print(f"  {param}: {variance:.4f}")

    # Visualize sensitivity
    fig2 = mc.visualize_sensitivity_analysis(sensitivity_df)
    fig2.savefig('monte_carlo_sensitivity.png', dpi=300, bbox_inches='tight')
    print("✓ Saved monte_carlo_sensitivity.png")
    plt.close()

    # 6. Portfolio Simulation
    print("\n6. Portfolio Risk Simulation")
    print("-" * 60)
    # Simulate historical returns for 3 assets
    n_days = 500
    returns_matrix = np.random.multivariate_normal(
        [0.0005, 0.0008, 0.0006],  # Expected returns
        [[0.0004, 0.0001, 0.0002],  # Covariance matrix
         [0.0001, 0.0009, 0.0001],
         [0.0002, 0.0001, 0.0005]],
        n_days
    )
    weights = np.array([0.4, 0.4, 0.2])

    portfolio_result = mc.portfolio_simulation(returns_matrix, weights, initial_value=1000000, n_days=252)
    print(f"Expected return: {portfolio_result['expected_return']:.4%}")
    print(f"Volatility: {portfolio_result['volatility']:.4%}")
    print(f"VaR (95%): {portfolio_result['var_95']:.4%}")
    print(f"CVaR (95%): {portfolio_result['cvar_95']:.4%}")
    print(f"Probability of loss: {portfolio_result['probability_loss']:.2%}")
    print(f"\nFinal value percentiles:")
    for p, v in portfolio_result['final_value_percentiles'].items():
        print(f"  {p}: ${v:,.2f}")

    # 7. Bootstrap Confidence Intervals
    print("\n7. Bootstrap Confidence Intervals")
    print("-" * 60)
    data = np.random.exponential(scale=2.0, size=100)

    bootstrap_result = mc.bootstrap_confidence_interval(data, np.mean, n_bootstrap=5000)
    print(f"Observed mean: {bootstrap_result['observed_statistic']:.4f}")
    print(f"Bootstrap mean: {bootstrap_result['bootstrap_mean']:.4f}")
    print(f"Bootstrap SE: {bootstrap_result['bootstrap_std']:.4f}")
    print(f"95% CI: [{bootstrap_result['confidence_interval'][0]:.4f}, {bootstrap_result['confidence_interval'][1]:.4f}]")

    # 8. Latin Hypercube Sampling
    print("\n8. Latin Hypercube Sampling")
    print("-" * 60)
    distributions = [
        ('norm', (0, 1)),
        ('uniform', (0, 10)),
        ('lognorm', (0, 0.5))
    ]
    lhs_samples = mc.latin_hypercube_sampling(distributions, n_samples=1000)
    print(f"Generated {lhs_samples.shape[0]} samples in {lhs_samples.shape[1]} dimensions")
    print(f"Sample means: {np.mean(lhs_samples, axis=0)}")
    print(f"Sample stds: {np.std(lhs_samples, axis=0)}")

    # 9. Scenario Analysis
    print("\n9. Scenario Analysis")
    print("-" * 60)

    def revenue_model(price, volume, cost):
        return price * volume - cost * volume

    scenarios = {
        'Base Case': {'price': 100, 'volume': 1000, 'cost': 60},
        'Optimistic': {'price': 120, 'volume': 1200, 'cost': 55},
        'Pessimistic': {'price': 80, 'volume': 800, 'cost': 70}
    }

    scenario_df = mc.scenario_analysis(revenue_model, scenarios)
    print(f"\nScenario results:\n{scenario_df}")

    print("\n" + "="*60)
    print("✓ Monte Carlo Simulation Demo Complete!")
    print("="*60)


if __name__ == '__main__':
    demo()
