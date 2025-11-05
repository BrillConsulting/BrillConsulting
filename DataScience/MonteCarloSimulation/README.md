# Monte Carlo Simulation Toolkit

Advanced Monte Carlo methods and stochastic simulation for risk analysis, option pricing, and uncertainty quantification.

## Overview

The Monte Carlo Simulation Toolkit provides comprehensive methods for stochastic modeling and simulation. It implements Monte Carlo integration, risk analysis (VaR), sensitivity analysis, option pricing, portfolio simulation, and bootstrap methods.

## Key Features

- **Monte Carlo Integration**: Estimate complex integrals using random sampling
- **Value at Risk (VaR)**: Calculate VaR and CVaR for risk assessment
- **Sensitivity Analysis**: Global sensitivity analysis with parameter variation
- **Geometric Brownian Motion**: Simulate stock price paths
- **Black-Scholes Option Pricing**: Monte Carlo option pricing with Greeks
- **Portfolio Simulation**: Risk metrics for multi-asset portfolios
- **Bootstrap Methods**: Confidence intervals via resampling
- **Latin Hypercube Sampling**: Efficient space-filling sampling
- **Scenario Analysis**: Evaluate outcomes under different scenarios

## Technologies Used

- **NumPy**: Numerical computing and random number generation
- **Pandas**: Data manipulation
- **SciPy**: Statistical distributions and optimization
- **Matplotlib & Seaborn**: Visualization

## Installation

```bash
cd MonteCarloSimulation/
pip install numpy pandas scipy matplotlib seaborn
```

## Usage Examples

### Value at Risk Analysis

```python
from monte_carlo_simulation import MonteCarloSimulation

mc = MonteCarloSimulation(random_state=42, n_simulations=10000)

# Calculate VaR
returns = np.random.normal(0.001, 0.02, 1000)
var_result = mc.value_at_risk(returns, confidence_level=0.95, method='historical')

print(f"VaR (95%): {var_result['var']:.4f}")
print(f"CVaR: {var_result['cvar']:.4f}")
```

### Stock Price Simulation

```python
# Simulate stock prices using GBM
paths = mc.geometric_brownian_motion(
    S0=100,      # Initial price
    mu=0.1,      # Expected return
    sigma=0.2,   # Volatility
    T=1.0,       # Time horizon (1 year)
    n_steps=252, # Daily steps
    n_paths=1000
)

# Visualize
fig = mc.visualize_monte_carlo_paths(paths, title="Stock Price Simulation")
```

### Option Pricing

```python
# Price European call option
option_result = mc.black_scholes_option_pricing(
    S0=100, K=100, T=1.0, r=0.05, sigma=0.2, option_type='call'
)

print(f"Analytical price: ${option_result['analytical_price']:.4f}")
print(f"Monte Carlo price: ${option_result['monte_carlo_price']:.4f}")
print(f"Delta: {option_result['delta']:.4f}")
```

### Portfolio Risk Analysis

```python
# Simulate portfolio returns
portfolio_result = mc.portfolio_simulation(
    returns=historical_returns,
    weights=np.array([0.4, 0.4, 0.2]),
    initial_value=1000000,
    n_days=252
)

print(f"Expected return: {portfolio_result['expected_return']:.4%}")
print(f"VaR (95%): {portfolio_result['var_95']:.4%}")
print(f"Probability of loss: {portfolio_result['probability_loss']:.2%}")
```

## Demo

```bash
python monte_carlo_simulation.py
```

The demo includes:
- Monte Carlo integration
- Value at Risk calculation
- Geometric Brownian Motion simulation
- Black-Scholes option pricing
- Sensitivity analysis
- Portfolio risk simulation
- Bootstrap confidence intervals
- Latin hypercube sampling
- Scenario analysis

## Output Examples

- `monte_carlo_gbm.png`: Simulated stock price paths with statistics
- `monte_carlo_sensitivity.png`: Sensitivity analysis results
- Console output with risk metrics and simulation statistics

## Author

**Brill Consulting**
- Email: clientbrill@gmail.com
- LinkedIn: [brillconsulting](https://www.linkedin.com/in/brillconsulting)
