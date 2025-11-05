# Bayesian Inference Toolkit

Advanced Bayesian statistics and probabilistic modeling for parameter estimation, hypothesis testing, and model comparison.

## Overview

The Bayesian Inference Toolkit provides a comprehensive suite of methods for Bayesian statistical analysis. It implements conjugate priors, MCMC sampling, credible intervals, Bayes factors, and hierarchical models, enabling robust probabilistic inference from data.

## Key Features

- **Beta-Binomial Inference**: Bayesian parameter estimation for binomial proportions with Beta priors
- **Normal Inference**: Posterior distributions for normal means with conjugate priors
- **Bayes Factors**: Model comparison and hypothesis testing using Bayes factors
- **MCMC Sampling**: Metropolis-Hastings algorithm with convergence diagnostics
- **Bayesian Linear Regression**: Conjugate Bayesian regression with predictive distributions
- **HPD Intervals**: Highest Posterior Density credible intervals
- **Posterior Predictive Checks**: Model validation and goodness-of-fit assessment
- **Visualization**: Interactive plots for prior/posterior distributions and MCMC diagnostics

## Technologies Used

- **NumPy**: Numerical computing and array operations
- **Pandas**: Data manipulation and analysis
- **SciPy**: Statistical distributions and special functions
- **Matplotlib**: Comprehensive visualization
- **Seaborn**: Statistical data visualization

## Installation

```bash
cd BayesianInference/
pip install numpy pandas scipy matplotlib seaborn
```

## Usage Examples

### Beta-Binomial Inference

```python
from bayesianinference import BayesianInference

# Initialize toolkit
bi = BayesianInference(random_state=42)

# Analyze conversion rate with 85 successes out of 100 trials
result = bi.beta_binomial_inference(
    successes=85,
    trials=100,
    prior_alpha=1,  # Uniform prior
    prior_beta=1
)

print(f"Posterior Mean: {result['posterior_mean']:.4f}")
print(f"95% Credible Interval: {result['credible_interval_95']}")
print(f"P(rate > 0.5): {result['probability_gt_0.5']:.4f}")

# Visualize prior and posterior
fig = bi.visualize_beta_inference(85, 100, 1, 1)
fig.savefig('beta_inference.png')
```

### Bayesian Hypothesis Testing

```python
# Test if mean is different from null value
data = np.random.normal(0.5, 1, 50)
result = bi.bayes_factor(data, null_value=0)

print(f"Bayes Factor (BF10): {result['bayes_factor_10']:.4f}")
print(f"Interpretation: {result['interpretation']}")
```

### MCMC Sampling

```python
# Define log posterior function
def log_posterior(params):
    mu, log_sigma = params
    sigma = np.exp(log_sigma)
    # Log likelihood + log prior
    ll = np.sum(stats.norm.logpdf(data, mu, sigma))
    lp = stats.norm.logpdf(mu, 0, 10) + stats.norm.logpdf(log_sigma, 0, 2)
    return ll + lp

# Run MCMC
result = bi.mcmc_metropolis(
    log_posterior,
    initial_params=np.array([0, 0]),
    n_iterations=5000
)

print(f"Acceptance Rate: {result['acceptance_rate']:.4f}")
print(f"Effective Sample Size: {result['effective_sample_size']}")

# Visualize MCMC diagnostics
fig = bi.visualize_mcmc_diagnostics(
    result['samples'],
    param_names=['mu', 'log(sigma)']
)
fig.savefig('mcmc_diagnostics.png')
```

### Bayesian Linear Regression

```python
# Generate data
X = np.random.randn(100, 2)
y = X @ np.array([2, -1]) + np.random.randn(100) * 0.5

# Fit Bayesian regression
result = bi.bayesian_linear_regression(X, y, alpha=1.0, beta=4.0)

print(f"Coefficients: {result['coefficients']}")
print(f"Posterior Covariance:\n{result['posterior_cov']}")
print(f"Predictive Std: {result['predictive_std'][:5]}")
```

## Demo

Run the comprehensive demo to see all features in action:

```bash
python bayesianinference.py
```

The demo includes:
- Beta-Binomial inference with visualization
- Normal distribution parameter estimation
- Bayes factor hypothesis testing
- MCMC sampling with diagnostics
- Bayesian linear regression
- HPD interval calculation
- Posterior predictive checks

## Output Examples

The demo generates:
- `bayesian_beta_inference.png`: Prior and posterior distributions
- `bayesian_mcmc_diagnostics.png`: Trace plots and posterior distributions
- Comprehensive console output with parameter estimates and test results

## Key Concepts

**Bayesian Inference**: Update beliefs about parameters using data through Bayes' theorem: P(θ|data) ∝ P(data|θ) × P(θ)

**Conjugate Priors**: Prior distributions that yield posterior distributions in the same family, enabling closed-form solutions

**Credible Intervals**: Bayesian equivalent of confidence intervals, representing the probability that the parameter lies within the interval

**Bayes Factors**: Ratio of marginal likelihoods comparing evidence for different hypotheses

**MCMC**: Markov Chain Monte Carlo sampling for approximating posterior distributions when analytical solutions are intractable

## Applications

- Clinical trials and medical research
- A/B testing and conversion rate analysis
- Quality control and reliability analysis
- Risk assessment and decision making
- Parameter estimation with uncertainty quantification
- Model selection and comparison

## Author

**Brill Consulting**
- Email: clientbrill@gmail.com
- LinkedIn: [brillconsulting](https://www.linkedin.com/in/brillconsulting)
