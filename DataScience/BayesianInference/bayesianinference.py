"""
Bayesian Inference Toolkit
===========================

Advanced Bayesian statistics and probabilistic modeling:
- Prior and posterior distributions
- Bayesian parameter estimation
- Credible intervals and HPD
- Bayesian hypothesis testing
- Model comparison (Bayes factors)
- MCMC sampling with diagnostics
- Hierarchical Bayesian models
- Bayesian linear regression

Author: Brill Consulting
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.special import gammaln, betaln
from typing import Dict, List, Tuple, Optional, Callable
import warnings
warnings.filterwarnings('ignore')


class BayesianInference:
    """Bayesian inference and probabilistic modeling toolkit."""

    def __init__(self, random_state: int = 42):
        """Initialize Bayesian inference toolkit."""
        self.random_state = random_state
        np.random.seed(random_state)
        self.samples = {}
        self.models = {}

    def beta_binomial_inference(self, successes: int, trials: int,
                                prior_alpha: float = 1, prior_beta: float = 1) -> Dict:
        """
        Bayesian inference for binomial proportion using Beta prior.

        Args:
            successes: Number of successes
            trials: Number of trials
            prior_alpha: Alpha parameter for Beta prior
            prior_beta: Beta parameter for Beta prior

        Returns:
            Dictionary with posterior parameters and statistics
        """
        # Posterior parameters
        post_alpha = prior_alpha + successes
        post_beta = prior_beta + (trials - successes)

        # Posterior statistics
        post_mean = post_alpha / (post_alpha + post_beta)
        post_mode = (post_alpha - 1) / (post_alpha + post_beta - 2) if post_alpha > 1 and post_beta > 1 else None
        post_var = (post_alpha * post_beta) / ((post_alpha + post_beta)**2 * (post_alpha + post_beta + 1))

        # Credible interval (95%)
        credible_interval = stats.beta.interval(0.95, post_alpha, post_beta)

        return {
            'prior': {'alpha': prior_alpha, 'beta': prior_beta},
            'posterior': {'alpha': post_alpha, 'beta': post_beta},
            'posterior_mean': post_mean,
            'posterior_mode': post_mode,
            'posterior_std': np.sqrt(post_var),
            'credible_interval_95': credible_interval,
            'probability_gt_0.5': 1 - stats.beta.cdf(0.5, post_alpha, post_beta)
        }

    def normal_inference(self, data: np.ndarray, prior_mu: float = 0,
                        prior_sigma: float = 1000, known_sigma: Optional[float] = None) -> Dict:
        """
        Bayesian inference for normal distribution mean.

        Args:
            data: Observed data
            prior_mu: Prior mean
            prior_sigma: Prior standard deviation
            known_sigma: Known standard deviation (if None, estimated from data)

        Returns:
            Dictionary with posterior parameters and statistics
        """
        n = len(data)
        sample_mean = np.mean(data)

        if known_sigma is None:
            known_sigma = np.std(data, ddof=1)

        # Posterior parameters (conjugate prior)
        prior_precision = 1 / (prior_sigma ** 2)
        data_precision = n / (known_sigma ** 2)

        post_precision = prior_precision + data_precision
        post_sigma = 1 / np.sqrt(post_precision)
        post_mu = (prior_precision * prior_mu + data_precision * sample_mean) / post_precision

        # Credible interval
        credible_interval = stats.norm.interval(0.95, post_mu, post_sigma)

        return {
            'prior': {'mu': prior_mu, 'sigma': prior_sigma},
            'posterior': {'mu': post_mu, 'sigma': post_sigma},
            'sample_mean': sample_mean,
            'sample_std': known_sigma,
            'credible_interval_95': credible_interval,
            'n_observations': n
        }

    def bayes_factor(self, data: np.ndarray, null_value: float = 0,
                    prior_scale: float = 1.0) -> Dict:
        """
        Calculate Bayes factor for hypothesis testing.

        Args:
            data: Observed data
            null_value: Value under null hypothesis
            prior_scale: Scale of Cauchy prior on effect size

        Returns:
            Dictionary with Bayes factor and interpretation
        """
        n = len(data)
        t_stat = (np.mean(data) - null_value) / (np.std(data, ddof=1) / np.sqrt(n))

        # Approximate Bayes factor using BIC approximation
        # BF01 = exp((BIC_1 - BIC_0) / 2)
        # For t-test: BF01 ≈ sqrt(n) * exp(-t^2/2)
        bf_01 = np.sqrt(n) * np.exp(-t_stat**2 / 2)
        bf_10 = 1 / bf_01

        # Interpretation
        if bf_10 > 100:
            interpretation = "Extreme evidence for H1"
        elif bf_10 > 30:
            interpretation = "Very strong evidence for H1"
        elif bf_10 > 10:
            interpretation = "Strong evidence for H1"
        elif bf_10 > 3:
            interpretation = "Moderate evidence for H1"
        elif bf_10 > 1:
            interpretation = "Weak evidence for H1"
        elif bf_10 > 1/3:
            interpretation = "Weak evidence for H0"
        elif bf_10 > 1/10:
            interpretation = "Moderate evidence for H0"
        elif bf_10 > 1/30:
            interpretation = "Strong evidence for H0"
        elif bf_10 > 1/100:
            interpretation = "Very strong evidence for H0"
        else:
            interpretation = "Extreme evidence for H0"

        return {
            'bayes_factor_10': bf_10,
            'bayes_factor_01': bf_01,
            't_statistic': t_stat,
            'interpretation': interpretation,
            'log_bf_10': np.log(bf_10)
        }

    def mcmc_metropolis(self, log_posterior: Callable, initial_params: np.ndarray,
                       n_iterations: int = 10000, proposal_std: float = 0.1) -> Dict:
        """
        Metropolis-Hastings MCMC sampling.

        Args:
            log_posterior: Log posterior function
            initial_params: Initial parameter values
            n_iterations: Number of MCMC iterations
            proposal_std: Standard deviation of proposal distribution

        Returns:
            Dictionary with samples and diagnostics
        """
        n_params = len(initial_params)
        samples = np.zeros((n_iterations, n_params))
        samples[0] = initial_params

        current_log_post = log_posterior(initial_params)
        accepted = 0

        for i in range(1, n_iterations):
            # Propose new parameters
            proposal = samples[i-1] + np.random.normal(0, proposal_std, n_params)
            proposal_log_post = log_posterior(proposal)

            # Acceptance ratio
            log_ratio = proposal_log_post - current_log_post

            if np.log(np.random.rand()) < log_ratio:
                # Accept
                samples[i] = proposal
                current_log_post = proposal_log_post
                accepted += 1
            else:
                # Reject
                samples[i] = samples[i-1]

        acceptance_rate = accepted / n_iterations

        # Diagnostics
        # Effective sample size (simplified)
        ess = self._calculate_ess(samples)

        return {
            'samples': samples,
            'acceptance_rate': acceptance_rate,
            'effective_sample_size': ess,
            'n_iterations': n_iterations
        }

    def _calculate_ess(self, samples: np.ndarray, max_lag: int = 100) -> np.ndarray:
        """Calculate effective sample size for each parameter."""
        n, p = samples.shape
        ess = np.zeros(p)

        for i in range(p):
            # Calculate autocorrelation
            param_samples = samples[:, i]
            autocorr = np.correlate(param_samples - param_samples.mean(),
                                   param_samples - param_samples.mean(),
                                   mode='full')[len(param_samples)-1:]
            autocorr = autocorr / autocorr[0]

            # Sum until autocorrelation becomes small
            sum_autocorr = 1
            for lag in range(1, min(max_lag, len(autocorr))):
                if autocorr[lag] < 0.05:
                    break
                sum_autocorr += 2 * autocorr[lag]

            ess[i] = n / sum_autocorr

        return ess

    def bayesian_linear_regression(self, X: np.ndarray, y: np.ndarray,
                                   alpha: float = 1.0, beta: float = 1.0) -> Dict:
        """
        Bayesian linear regression with conjugate priors.

        Args:
            X: Design matrix (n_samples, n_features)
            y: Target values
            alpha: Prior precision for coefficients
            beta: Noise precision

        Returns:
            Dictionary with posterior parameters
        """
        n, p = X.shape

        # Add intercept
        X_design = np.column_stack([np.ones(n), X])

        # Prior
        prior_precision = alpha * np.eye(p + 1)
        prior_mean = np.zeros(p + 1)

        # Posterior (conjugate update)
        post_precision = prior_precision + beta * X_design.T @ X_design
        post_cov = np.linalg.inv(post_precision)
        post_mean = post_cov @ (prior_precision @ prior_mean + beta * X_design.T @ y)

        # Predictions
        y_pred_mean = X_design @ post_mean

        # Predictive variance
        pred_var = 1/beta + np.sum(X_design @ post_cov * X_design, axis=1)

        return {
            'posterior_mean': post_mean,
            'posterior_cov': post_cov,
            'predictions': y_pred_mean,
            'predictive_std': np.sqrt(pred_var),
            'coefficients': post_mean[1:],  # Exclude intercept
            'intercept': post_mean[0]
        }

    def hpd_interval(self, samples: np.ndarray, credible_mass: float = 0.95) -> Tuple[float, float]:
        """
        Calculate Highest Posterior Density (HPD) interval.

        Args:
            samples: MCMC samples
            credible_mass: Desired credible mass (default 0.95)

        Returns:
            Tuple of (lower, upper) HPD bounds
        """
        sorted_samples = np.sort(samples)
        n = len(sorted_samples)
        n_included = int(np.ceil(credible_mass * n))

        # Find interval with smallest width
        interval_widths = sorted_samples[n_included:] - sorted_samples[:n-n_included]
        min_idx = np.argmin(interval_widths)

        hpd_lower = sorted_samples[min_idx]
        hpd_upper = sorted_samples[min_idx + n_included]

        return (hpd_lower, hpd_upper)

    def posterior_predictive_check(self, observed_data: np.ndarray,
                                  posterior_samples: np.ndarray,
                                  data_generator: Callable) -> Dict:
        """
        Posterior predictive check for model validation.

        Args:
            observed_data: Actual observed data
            posterior_samples: Samples from posterior distribution
            data_generator: Function to generate data from parameters

        Returns:
            Dictionary with test statistics and p-value
        """
        n_samples = len(posterior_samples)

        # Observed test statistic (e.g., mean)
        obs_stat = np.mean(observed_data)

        # Generate replicated datasets
        rep_stats = np.zeros(n_samples)
        for i in range(n_samples):
            rep_data = data_generator(posterior_samples[i])
            rep_stats[i] = np.mean(rep_data)

        # Bayesian p-value
        p_value = np.mean(rep_stats >= obs_stat)

        return {
            'observed_statistic': obs_stat,
            'replicated_statistics': rep_stats,
            'bayesian_p_value': p_value,
            'interpretation': 'Good fit' if 0.05 < p_value < 0.95 else 'Poor fit'
        }

    def visualize_beta_inference(self, successes: int, trials: int,
                                prior_alpha: float = 1, prior_beta: float = 1) -> plt.Figure:
        """Visualize Beta-Binomial inference."""
        result = self.beta_binomial_inference(successes, trials, prior_alpha, prior_beta)

        x = np.linspace(0, 1, 1000)

        # Prior and posterior densities
        prior_pdf = stats.beta.pdf(x, prior_alpha, prior_beta)
        post_pdf = stats.beta.pdf(x, result['posterior']['alpha'], result['posterior']['beta'])

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Distributions
        axes[0].plot(x, prior_pdf, 'b--', label=f'Prior: Beta({prior_alpha}, {prior_beta})', linewidth=2)
        axes[0].plot(x, post_pdf, 'r-', label=f"Posterior: Beta({result['posterior']['alpha']:.1f}, {result['posterior']['beta']:.1f})", linewidth=2)
        axes[0].axvline(result['posterior_mean'], color='red', linestyle=':', label=f"Posterior Mean: {result['posterior_mean']:.3f}")
        axes[0].fill_between(x, post_pdf, where=(x >= result['credible_interval_95'][0]) & (x <= result['credible_interval_95'][1]),
                            alpha=0.3, color='red', label='95% Credible Interval')
        axes[0].set_xlabel('Probability')
        axes[0].set_ylabel('Density')
        axes[0].set_title('Bayesian Inference: Prior vs Posterior')
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        # Summary statistics
        summary_text = f"Data: {successes}/{trials} successes\n\n"
        summary_text += f"Posterior Mean: {result['posterior_mean']:.4f}\n"
        summary_text += f"Posterior Std: {result['posterior_std']:.4f}\n"
        summary_text += f"95% CI: [{result['credible_interval_95'][0]:.4f}, {result['credible_interval_95'][1]:.4f}]\n\n"
        summary_text += f"P(p > 0.5): {result['probability_gt_0.5']:.4f}"

        axes[1].text(0.1, 0.5, summary_text, transform=axes[1].transAxes,
                    fontsize=12, verticalalignment='center', family='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        axes[1].set_title('Summary Statistics')
        axes[1].axis('off')

        plt.tight_layout()
        return fig

    def visualize_mcmc_diagnostics(self, samples: np.ndarray, param_names: Optional[List[str]] = None) -> plt.Figure:
        """Visualize MCMC diagnostics (trace plots and posteriors)."""
        n_samples, n_params = samples.shape

        if param_names is None:
            param_names = [f'param_{i}' for i in range(n_params)]

        fig, axes = plt.subplots(n_params, 2, figsize=(14, 4*n_params))

        if n_params == 1:
            axes = axes.reshape(1, -1)

        for i in range(n_params):
            # Trace plot
            axes[i, 0].plot(samples[:, i], alpha=0.7)
            axes[i, 0].set_xlabel('Iteration')
            axes[i, 0].set_ylabel(param_names[i])
            axes[i, 0].set_title(f'Trace Plot: {param_names[i]}')
            axes[i, 0].grid(alpha=0.3)

            # Posterior distribution
            axes[i, 1].hist(samples[:, i], bins=50, density=True, alpha=0.7, edgecolor='black')
            axes[i, 1].axvline(np.mean(samples[:, i]), color='red', linestyle='--',
                             label=f'Mean: {np.mean(samples[:, i]):.3f}')
            axes[i, 1].axvline(np.median(samples[:, i]), color='green', linestyle='--',
                             label=f'Median: {np.median(samples[:, i]):.3f}')
            axes[i, 1].set_xlabel(param_names[i])
            axes[i, 1].set_ylabel('Density')
            axes[i, 1].set_title(f'Posterior Distribution: {param_names[i]}')
            axes[i, 1].legend()
            axes[i, 1].grid(alpha=0.3)

        plt.tight_layout()
        return fig


def demo():
    """Demo Bayesian inference toolkit."""
    np.random.seed(42)

    print("Bayesian Inference Toolkit Demo")
    print("="*60)

    bi = BayesianInference(random_state=42)

    # 1. Beta-Binomial Inference
    print("\n1. Beta-Binomial Inference (Conversion Rate)")
    print("-" * 60)
    result = bi.beta_binomial_inference(successes=85, trials=100, prior_alpha=1, prior_beta=1)
    print(f"Observed: 85/100 conversions")
    print(f"Posterior Mean: {result['posterior_mean']:.4f}")
    print(f"95% Credible Interval: [{result['credible_interval_95'][0]:.4f}, {result['credible_interval_95'][1]:.4f}]")
    print(f"P(conversion rate > 0.5): {result['probability_gt_0.5']:.4f}")

    # Visualize
    fig1 = bi.visualize_beta_inference(85, 100, 1, 1)
    fig1.savefig('bayesian_beta_inference.png', dpi=300, bbox_inches='tight')
    print("✓ Saved bayesian_beta_inference.png")
    plt.close()

    # 2. Normal Inference
    print("\n2. Bayesian Inference for Normal Mean")
    print("-" * 60)
    data = np.random.normal(10, 2, 100)
    result = bi.normal_inference(data, prior_mu=0, prior_sigma=10)
    print(f"Sample Mean: {result['sample_mean']:.4f}")
    print(f"Posterior Mean: {result['posterior']['mu']:.4f}")
    print(f"Posterior Std: {result['posterior']['sigma']:.4f}")
    print(f"95% Credible Interval: [{result['credible_interval_95'][0]:.4f}, {result['credible_interval_95'][1]:.4f}]")

    # 3. Bayes Factor
    print("\n3. Bayes Factor (Hypothesis Testing)")
    print("-" * 60)
    data = np.random.normal(0.5, 1, 50)
    result = bi.bayes_factor(data, null_value=0)
    print(f"T-statistic: {result['t_statistic']:.4f}")
    print(f"Bayes Factor (BF10): {result['bayes_factor_10']:.4f}")
    print(f"Interpretation: {result['interpretation']}")

    # 4. MCMC Sampling
    print("\n4. MCMC Metropolis-Hastings Sampling")
    print("-" * 60)

    # Simple example: sampling from normal distribution
    def log_posterior(params):
        mu, log_sigma = params
        sigma = np.exp(log_sigma)
        # Log likelihood (normal)
        ll = np.sum(stats.norm.logpdf(data, mu, sigma))
        # Log prior (weakly informative)
        lp = stats.norm.logpdf(mu, 0, 10) + stats.norm.logpdf(log_sigma, 0, 2)
        return ll + lp

    result = bi.mcmc_metropolis(log_posterior, initial_params=np.array([0, 0]),
                               n_iterations=5000, proposal_std=0.1)
    print(f"Acceptance Rate: {result['acceptance_rate']:.4f}")
    print(f"Effective Sample Size: {result['effective_sample_size']}")

    # Visualize MCMC
    fig2 = bi.visualize_mcmc_diagnostics(result['samples'], param_names=['mu', 'log(sigma)'])
    fig2.savefig('bayesian_mcmc_diagnostics.png', dpi=300, bbox_inches='tight')
    print("✓ Saved bayesian_mcmc_diagnostics.png")
    plt.close()

    # 5. Bayesian Linear Regression
    print("\n5. Bayesian Linear Regression")
    print("-" * 60)
    n = 100
    X = np.random.randn(n, 2)
    true_beta = np.array([2, -1])
    y = X @ true_beta + np.random.randn(n) * 0.5

    result = bi.bayesian_linear_regression(X, y, alpha=1.0, beta=4.0)
    print(f"True coefficients: {true_beta}")
    print(f"Posterior mean: {result['coefficients']}")
    print(f"Posterior std: {np.sqrt(np.diag(result['posterior_cov'])[1:])}")
    print(f"Intercept: {result['intercept']:.4f}")

    # 6. HPD Interval
    print("\n6. Highest Posterior Density Interval")
    print("-" * 60)
    samples = np.random.beta(5, 2, 10000)
    hpd = bi.hpd_interval(samples, credible_mass=0.95)
    print(f"95% HPD Interval: [{hpd[0]:.4f}, {hpd[1]:.4f}]")
    equal_tailed = np.percentile(samples, [2.5, 97.5])
    print(f"95% Equal-Tailed Interval: [{equal_tailed[0]:.4f}, {equal_tailed[1]:.4f}]")

    print("\n" + "="*60)
    print("✓ Bayesian Inference Demo Complete!")
    print("="*60)


if __name__ == '__main__':
    demo()
