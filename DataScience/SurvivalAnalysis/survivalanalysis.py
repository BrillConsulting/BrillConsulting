"""
Survival Analysis Toolkit
==========================

Advanced survival analysis and time-to-event modeling:
- Kaplan-Meier estimator
- Cox Proportional Hazards regression
- Log-rank test
- Competing risks analysis
- Parametric survival models
- Time-dependent covariates
- Survival curve visualization

Author: Brill Consulting
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import minimize
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class SurvivalAnalysis:
    """Survival analysis and time-to-event modeling toolkit."""

    def __init__(self):
        """Initialize survival analysis toolkit."""
        self.survival_functions = {}
        self.hazard_functions = {}

    def kaplan_meier(self, times: np.ndarray, events: np.ndarray) -> Dict:
        """
        Calculate Kaplan-Meier survival estimates.

        Args:
            times: Time to event or censoring
            events: Event indicator (1=event, 0=censored)

        Returns:
            Dictionary with survival function estimates
        """
        # Sort by time
        order = np.argsort(times)
        times_sorted = times[order]
        events_sorted = events[order]

        # Get unique event times
        unique_times = np.unique(times_sorted[events_sorted == 1])

        n = len(times)
        survival_probs = []
        survival_times = []
        at_risk = []
        events_count = []

        current_survival = 1.0

        for t in unique_times:
            # Number at risk
            n_at_risk = np.sum(times_sorted >= t)

            # Number of events at time t
            n_events = np.sum((times_sorted == t) & (events_sorted == 1))

            # Update survival probability
            if n_at_risk > 0:
                current_survival *= (1 - n_events / n_at_risk)

            survival_times.append(t)
            survival_probs.append(current_survival)
            at_risk.append(n_at_risk)
            events_count.append(n_events)

        # Calculate confidence intervals (Greenwood's formula)
        survival_probs = np.array(survival_probs)
        variance = np.zeros_like(survival_probs)

        cum_hazard = 0
        for i, t in enumerate(survival_times):
            n_at_risk = at_risk[i]
            n_events = events_count[i]

            if n_at_risk > 0 and n_events > 0:
                cum_hazard += n_events / (n_at_risk * (n_at_risk - n_events))

            variance[i] = survival_probs[i]**2 * cum_hazard

        std_error = np.sqrt(variance)
        ci_lower = survival_probs * np.exp(-1.96 * std_error / survival_probs)
        ci_upper = survival_probs * np.exp(1.96 * std_error / survival_probs)

        # Median survival time
        median_survival = None
        if np.any(survival_probs <= 0.5):
            median_survival = survival_times[np.argmax(survival_probs <= 0.5)]

        return {
            'times': np.array(survival_times),
            'survival_probs': survival_probs,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'at_risk': np.array(at_risk),
            'events': np.array(events_count),
            'median_survival': median_survival
        }

    def log_rank_test(self, times1: np.ndarray, events1: np.ndarray,
                     times2: np.ndarray, events2: np.ndarray) -> Dict:
        """
        Perform log-rank test for comparing two survival curves.

        Args:
            times1: Times for group 1
            events1: Events for group 1
            times2: Times for group 2
            events2: Events for group 2

        Returns:
            Dictionary with test results
        """
        # Combine data
        all_times = np.concatenate([times1, times2])
        all_events = np.concatenate([events1, events2])
        group = np.concatenate([np.zeros(len(times1)), np.ones(len(times2))])

        # Get unique event times
        event_times = np.unique(all_times[all_events == 1])

        # Calculate log-rank statistic
        observed1 = 0
        expected1 = 0
        variance = 0

        for t in event_times:
            # At risk in each group
            n1_at_risk = np.sum((times1 >= t))
            n2_at_risk = np.sum((times2 >= t))
            n_at_risk = n1_at_risk + n2_at_risk

            # Events in each group
            d1 = np.sum((times1 == t) & (events1 == 1))
            d2 = np.sum((times2 == t) & (events2 == 1))
            d = d1 + d2

            if n_at_risk > 0:
                # Expected events in group 1
                expected = n1_at_risk * d / n_at_risk

                observed1 += d1
                expected1 += expected

                # Variance component
                if n_at_risk > 1:
                    variance += (n1_at_risk * n2_at_risk * d * (n_at_risk - d)) / \
                               (n_at_risk**2 * (n_at_risk - 1))

        # Calculate test statistic
        chi_square = (observed1 - expected1)**2 / variance if variance > 0 else 0
        p_value = 1 - stats.chi2.cdf(chi_square, df=1)

        return {
            'chi_square': chi_square,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'observed_group1': observed1,
            'expected_group1': expected1
        }

    def cox_proportional_hazards(self, times: np.ndarray, events: np.ndarray,
                                X: np.ndarray, max_iter: int = 50) -> Dict:
        """
        Fit Cox proportional hazards model using Newton-Raphson.

        Args:
            times: Time to event or censoring
            events: Event indicator
            X: Covariate matrix
            max_iter: Maximum iterations

        Returns:
            Dictionary with Cox model results
        """
        n, p = X.shape

        # Sort by time
        order = np.argsort(times)
        times_sorted = times[order]
        events_sorted = events[order]
        X_sorted = X[order]

        # Initialize coefficients
        beta = np.zeros(p)

        for iteration in range(max_iter):
            # Calculate risk scores
            risk_scores = np.exp(X_sorted @ beta)

            # Calculate score and hessian
            score = np.zeros(p)
            hessian = np.zeros((p, p))

            for i in range(n):
                if events_sorted[i] == 1:
                    # At risk set
                    at_risk = times_sorted >= times_sorted[i]
                    risk_sum = np.sum(risk_scores[at_risk])

                    # Score contribution
                    weighted_X = (risk_scores[at_risk, np.newaxis] * X_sorted[at_risk]).T
                    mean_X = np.sum(weighted_X, axis=1) / risk_sum

                    score += X_sorted[i] - mean_X

                    # Hessian contribution (information matrix)
                    X_centered = X_sorted[at_risk] - mean_X
                    hessian_contrib = (risk_scores[at_risk, np.newaxis, np.newaxis] *
                                     X_centered[:, :, np.newaxis] * X_centered[:, np.newaxis, :])
                    hessian -= np.sum(hessian_contrib, axis=0) / risk_sum

            # Newton-Raphson update
            try:
                beta_update = np.linalg.solve(-hessian, score)
                beta_new = beta + beta_update
            except:
                break

            # Check convergence
            if np.linalg.norm(beta_new - beta) < 1e-6:
                beta = beta_new
                break

            beta = beta_new

        # Calculate standard errors
        try:
            var_beta = np.linalg.inv(-hessian)
            se_beta = np.sqrt(np.diag(var_beta))
        except:
            se_beta = np.full(p, np.nan)

        # Calculate hazard ratios and confidence intervals
        hazard_ratios = np.exp(beta)
        hr_ci_lower = np.exp(beta - 1.96 * se_beta)
        hr_ci_upper = np.exp(beta + 1.96 * se_beta)

        # Wald test
        z_scores = beta / se_beta
        p_values = 2 * (1 - stats.norm.cdf(np.abs(z_scores)))

        # Log-likelihood (partial)
        log_likelihood = 0
        for i in range(n):
            if events_sorted[i] == 1:
                at_risk = times_sorted >= times_sorted[i]
                log_likelihood += X_sorted[i] @ beta - np.log(np.sum(risk_scores[at_risk]))

        return {
            'coefficients': beta,
            'std_errors': se_beta,
            'hazard_ratios': hazard_ratios,
            'hr_ci_lower': hr_ci_lower,
            'hr_ci_upper': hr_ci_upper,
            'z_scores': z_scores,
            'p_values': p_values,
            'log_likelihood': log_likelihood,
            'n_iter': iteration + 1
        }

    def weibull_survival(self, times: np.ndarray, events: np.ndarray) -> Dict:
        """
        Fit Weibull distribution to survival data.

        Args:
            times: Time to event or censoring
            events: Event indicator

        Returns:
            Dictionary with Weibull parameters
        """
        # Maximum likelihood estimation
        def neg_log_likelihood(params):
            shape, scale = params
            if shape <= 0 or scale <= 0:
                return np.inf

            ll = 0
            for t, e in zip(times, events):
                if e == 1:  # Event occurred
                    ll += np.log(shape/scale) + (shape - 1) * np.log(t/scale) - (t/scale)**shape
                else:  # Censored
                    ll += -(t/scale)**shape

            return -ll

        # Initial guess
        init_params = [1.0, np.mean(times)]

        result = minimize(neg_log_likelihood, init_params,
                         bounds=[(0.1, 10), (0.1, np.max(times)*2)],
                         method='L-BFGS-B')

        shape, scale = result.x

        # Survival function
        def survival_function(t):
            return np.exp(-(t / scale)**shape)

        # Hazard function
        def hazard_function(t):
            return (shape / scale) * (t / scale)**(shape - 1)

        return {
            'shape': shape,
            'scale': scale,
            'survival_function': survival_function,
            'hazard_function': hazard_function,
            'median_survival': scale * (np.log(2))**(1/shape)
        }

    def visualize_survival_curves(self, km_results: Dict, title: str = "Kaplan-Meier Survival Curve") -> plt.Figure:
        """Visualize Kaplan-Meier survival curve with confidence intervals."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Survival curve
        times = km_results['times']
        surv_probs = km_results['survival_probs']
        ci_lower = km_results['ci_lower']
        ci_upper = km_results['ci_upper']

        axes[0].step(times, surv_probs, where='post', linewidth=2, label='Survival Probability')
        axes[0].fill_between(times, ci_lower, ci_upper, alpha=0.3, step='post', label='95% CI')
        axes[0].axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Median')

        if km_results['median_survival'] is not None:
            axes[0].axvline(x=km_results['median_survival'], color='red', linestyle='--', alpha=0.5)

        axes[0].set_xlabel('Time', fontsize=12)
        axes[0].set_ylabel('Survival Probability', fontsize=12)
        axes[0].set_title(title, fontsize=14, weight='bold')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        axes[0].set_ylim([-0.05, 1.05])

        # At-risk table
        at_risk = km_results['at_risk']
        events = km_results['events']

        axes[1].axis('off')
        summary_text = f"Survival Analysis Summary\n"
        summary_text += "=" * 30 + "\n\n"
        summary_text += f"Total observations: {at_risk[0]}\n"
        summary_text += f"Total events: {np.sum(events)}\n"
        summary_text += f"Censored: {at_risk[0] - np.sum(events)}\n\n"

        if km_results['median_survival'] is not None:
            summary_text += f"Median survival: {km_results['median_survival']:.2f}\n\n"
        else:
            summary_text += f"Median survival: Not reached\n\n"

        summary_text += "Time Points:\n"
        for i in range(min(10, len(times))):
            summary_text += f"  t={times[i]:.1f}: S(t)={surv_probs[i]:.3f}\n"

        axes[1].text(0.1, 0.5, summary_text, transform=axes[1].transAxes,
                    fontsize=10, verticalalignment='center', family='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        axes[1].set_title('Summary Statistics')

        plt.tight_layout()
        return fig


def demo():
    """Demo survival analysis toolkit."""
    np.random.seed(42)

    print("Survival Analysis Toolkit Demo")
    print("="*60)

    sa = SurvivalAnalysis()

    # Generate synthetic survival data
    n = 200
    # Group 1
    times1 = np.random.exponential(scale=10, size=n//2)
    events1 = (np.random.rand(n//2) > 0.2).astype(int)  # 80% event rate

    # Group 2 (worse survival)
    times2 = np.random.exponential(scale=7, size=n//2)
    events2 = (np.random.rand(n//2) > 0.15).astype(int)  # 85% event rate

    # 1. Kaplan-Meier Estimator
    print("\n1. Kaplan-Meier Survival Curve (Group 1)")
    print("-" * 60)
    km1 = sa.kaplan_meier(times1, events1)
    print(f"Median survival: {km1['median_survival']:.2f}" if km1['median_survival'] else "Median survival: Not reached")
    print(f"Survival at t=5: {km1['survival_probs'][np.argmax(km1['times'] >= 5)]:.3f}")
    print(f"Total events: {np.sum(km1['events'])}")

    print("\n   Kaplan-Meier Survival Curve (Group 2)")
    print("-" * 60)
    km2 = sa.kaplan_meier(times2, events2)
    print(f"Median survival: {km2['median_survival']:.2f}" if km2['median_survival'] else "Median survival: Not reached")
    print(f"Total events: {np.sum(km2['events'])}")

    # Visualize
    fig1 = sa.visualize_survival_curves(km1, title="Kaplan-Meier Curve: Group 1")
    fig1.savefig('survival_kaplan_meier.png', dpi=300, bbox_inches='tight')
    print("✓ Saved survival_kaplan_meier.png")
    plt.close()

    # 2. Log-Rank Test
    print("\n2. Log-Rank Test (Compare Two Groups)")
    print("-" * 60)
    lr_result = sa.log_rank_test(times1, events1, times2, events2)
    print(f"Chi-square statistic: {lr_result['chi_square']:.4f}")
    print(f"P-value: {lr_result['p_value']:.4e}")
    print(f"Significant difference: {lr_result['significant']}")

    # 3. Cox Proportional Hazards
    print("\n3. Cox Proportional Hazards Regression")
    print("-" * 60)
    # Combine data
    all_times = np.concatenate([times1, times2])
    all_events = np.concatenate([events1, events2])

    # Create covariates
    X = np.random.randn(n, 3)
    X = np.column_stack([X, np.concatenate([np.zeros(n//2), np.ones(n//2)])])  # Group indicator

    cox_result = sa.cox_proportional_hazards(all_times, all_events, X)
    print(f"Coefficients: {cox_result['coefficients']}")
    print(f"Hazard Ratios: {cox_result['hazard_ratios']}")
    print(f"P-values: {cox_result['p_values']}")
    print(f"\nCovariate 4 (Group): HR = {cox_result['hazard_ratios'][3]:.3f}, p = {cox_result['p_values'][3]:.4e}")

    # 4. Weibull Survival Model
    print("\n4. Weibull Parametric Survival Model")
    print("-" * 60)
    weibull_result = sa.weibull_survival(times1, events1)
    print(f"Shape parameter: {weibull_result['shape']:.4f}")
    print(f"Scale parameter: {weibull_result['scale']:.4f}")
    print(f"Median survival: {weibull_result['median_survival']:.2f}")

    # Compare survival functions
    fig2, ax = plt.subplots(figsize=(10, 6))

    # Kaplan-Meier
    ax.step(km1['times'], km1['survival_probs'], where='post',
           linewidth=2, label='Kaplan-Meier', color='blue')

    # Weibull
    t_grid = np.linspace(0, np.max(times1), 100)
    weibull_surv = weibull_result['survival_function'](t_grid)
    ax.plot(t_grid, weibull_surv, linewidth=2, label='Weibull', color='red', linestyle='--')

    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Survival Probability', fontsize=12)
    ax.set_title('Survival Function Comparison', fontsize=14, weight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    fig2.savefig('survival_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved survival_comparison.png")
    plt.close()

    print("\n" + "="*60)
    print("✓ Survival Analysis Demo Complete!")
    print("="*60)


if __name__ == '__main__':
    demo()
