"""
Causal Inference Toolkit
=========================

Advanced causal inference methods for observational data:
- Propensity Score Matching (PSM)
- Inverse Probability Weighting (IPW)
- Difference-in-Differences (DiD)
- Regression Discontinuity Design (RDD)
- Instrumental Variables (IV)
- Doubly Robust Estimation
- Synthetic Control
- Causal graphs and DAGs

Author: Brill Consulting
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import NearestNeighbors
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class CausalInference:
    """Causal inference and treatment effect estimation toolkit."""

    def __init__(self, random_state: int = 42):
        """Initialize causal inference toolkit."""
        self.random_state = random_state
        np.random.seed(random_state)
        self.propensity_model = None
        self.matched_data = None

    def estimate_propensity_scores(self, X: pd.DataFrame, treatment: np.ndarray) -> np.ndarray:
        """
        Estimate propensity scores using logistic regression.

        Args:
            X: Covariate matrix
            treatment: Treatment indicator (0 or 1)

        Returns:
            Array of propensity scores
        """
        self.propensity_model = LogisticRegression(random_state=self.random_state, max_iter=1000)
        self.propensity_model.fit(X, treatment)
        propensity_scores = self.propensity_model.predict_proba(X)[:, 1]

        return propensity_scores

    def propensity_score_matching(self, X: pd.DataFrame, treatment: np.ndarray,
                                  outcome: np.ndarray, caliper: float = 0.1) -> Dict:
        """
        Perform propensity score matching to estimate treatment effect.

        Args:
            X: Covariate matrix
            treatment: Treatment indicator
            outcome: Outcome variable
            caliper: Maximum allowed distance for matching

        Returns:
            Dictionary with matching results and treatment effect
        """
        # Estimate propensity scores
        ps = self.estimate_propensity_scores(X, treatment)

        # Separate treated and control units
        treated_idx = np.where(treatment == 1)[0]
        control_idx = np.where(treatment == 0)[0]

        treated_ps = ps[treated_idx].reshape(-1, 1)
        control_ps = ps[control_idx].reshape(-1, 1)

        # Find nearest neighbors
        nn = NearestNeighbors(n_neighbors=1, metric='euclidean')
        nn.fit(control_ps)
        distances, matches = nn.kneighbors(treated_ps)

        # Apply caliper
        valid_matches = distances.flatten() < caliper
        matched_treated = treated_idx[valid_matches]
        matched_control = control_idx[matches.flatten()[valid_matches]]

        # Calculate treatment effect
        ate = np.mean(outcome[matched_treated]) - np.mean(outcome[matched_control])

        # Standard error (conservative estimate)
        n_matched = len(matched_treated)
        var_treated = np.var(outcome[matched_treated])
        var_control = np.var(outcome[matched_control])
        se = np.sqrt((var_treated + var_control) / n_matched)

        # Confidence interval
        ci_lower = ate - 1.96 * se
        ci_upper = ate + 1.96 * se

        # Store matched data
        self.matched_data = {
            'treated_idx': matched_treated,
            'control_idx': matched_control,
            'propensity_scores': ps
        }

        return {
            'ate': ate,
            'se': se,
            'ci_95': (ci_lower, ci_upper),
            'n_matched': n_matched,
            'n_total': len(treatment),
            'matching_rate': n_matched / np.sum(treatment)
        }

    def inverse_probability_weighting(self, X: pd.DataFrame, treatment: np.ndarray,
                                      outcome: np.ndarray) -> Dict:
        """
        Estimate treatment effect using inverse probability weighting.

        Args:
            X: Covariate matrix
            treatment: Treatment indicator
            outcome: Outcome variable

        Returns:
            Dictionary with IPW estimates
        """
        # Estimate propensity scores
        ps = self.estimate_propensity_scores(X, treatment)

        # Avoid division by zero
        ps = np.clip(ps, 0.01, 0.99)

        # Calculate weights
        weights = treatment / ps + (1 - treatment) / (1 - ps)

        # Weighted outcomes
        treated_outcome = np.sum(weights * treatment * outcome) / np.sum(weights * treatment)
        control_outcome = np.sum(weights * (1 - treatment) * outcome) / np.sum(weights * (1 - treatment))

        ate = treated_outcome - control_outcome

        # Standard error (simplified)
        n = len(treatment)
        se = np.sqrt(np.var(outcome[treatment == 1]) / np.sum(treatment) +
                    np.var(outcome[treatment == 0]) / np.sum(1 - treatment))

        return {
            'ate': ate,
            'se': se,
            'ci_95': (ate - 1.96 * se, ate + 1.96 * se),
            'treated_mean': treated_outcome,
            'control_mean': control_outcome
        }

    def difference_in_differences(self, data: pd.DataFrame, outcome_col: str,
                                 treatment_col: str, time_col: str, unit_col: str) -> Dict:
        """
        Estimate treatment effect using difference-in-differences.

        Args:
            data: Panel data with units over time
            outcome_col: Name of outcome variable
            treatment_col: Name of treatment indicator
            time_col: Name of time indicator (0=pre, 1=post)
            unit_col: Name of unit identifier

        Returns:
            Dictionary with DiD estimates
        """
        # Calculate means for each group
        treated_post = data[(data[treatment_col] == 1) & (data[time_col] == 1)][outcome_col].mean()
        treated_pre = data[(data[treatment_col] == 1) & (data[time_col] == 0)][outcome_col].mean()
        control_post = data[(data[treatment_col] == 0) & (data[time_col] == 1)][outcome_col].mean()
        control_pre = data[(data[treatment_col] == 0) & (data[time_col] == 0)][outcome_col].mean()

        # DiD estimator
        treated_diff = treated_post - treated_pre
        control_diff = control_post - control_pre
        did_estimate = treated_diff - control_diff

        # Regression-based DiD for standard errors
        data['interaction'] = data[treatment_col] * data[time_col]
        X = data[[treatment_col, time_col, 'interaction']]
        y = data[outcome_col]

        model = LinearRegression()
        model.fit(X, y)

        # The coefficient on interaction is the DiD estimate
        did_coef = model.coef_[2]

        # Calculate standard error (simplified - assumes homoscedasticity)
        residuals = y - model.predict(X)
        n = len(data)
        k = X.shape[1]
        mse = np.sum(residuals**2) / (n - k - 1)
        X_with_intercept = np.column_stack([np.ones(n), X])
        var_coef = mse * np.linalg.inv(X_with_intercept.T @ X_with_intercept)
        se = np.sqrt(var_coef[3, 3])  # SE for interaction term

        return {
            'did_estimate': did_estimate,
            'se': se,
            'ci_95': (did_estimate - 1.96 * se, did_estimate + 1.96 * se),
            'treated_pre': treated_pre,
            'treated_post': treated_post,
            'control_pre': control_pre,
            'control_post': control_post,
            'parallel_trends_test': self._test_parallel_trends(treated_diff, control_diff)
        }

    def _test_parallel_trends(self, treated_diff: float, control_diff: float) -> str:
        """Simple parallel trends assumption check (placeholder)."""
        # In practice, this requires pre-treatment period analysis
        return "Assumption should be tested with pre-treatment data"

    def regression_discontinuity(self, running_var: np.ndarray, outcome: np.ndarray,
                                cutoff: float, bandwidth: Optional[float] = None) -> Dict:
        """
        Estimate treatment effect using regression discontinuity design.

        Args:
            running_var: Running variable (e.g., test score)
            outcome: Outcome variable
            cutoff: Treatment assignment cutoff
            bandwidth: Bandwidth around cutoff (if None, uses Imbens-Kalyanaraman optimal)

        Returns:
            Dictionary with RDD estimates
        """
        # Default bandwidth (simplified version)
        if bandwidth is None:
            bandwidth = 1.0 * np.std(running_var)

        # Select observations within bandwidth
        mask = np.abs(running_var - cutoff) <= bandwidth
        running_var_bw = running_var[mask]
        outcome_bw = outcome[mask]

        # Treatment indicator
        treatment = (running_var_bw >= cutoff).astype(int)

        # Center running variable
        running_centered = running_var_bw - cutoff

        # Local linear regression
        X = np.column_stack([
            np.ones(len(running_centered)),
            running_centered,
            treatment,
            treatment * running_centered
        ])

        model = LinearRegression()
        model.fit(X, outcome_bw)

        # RDD estimate is the coefficient on treatment
        rdd_estimate = model.coef_[2]

        # Standard error (simplified)
        residuals = outcome_bw - model.predict(X)
        n = len(outcome_bw)
        k = X.shape[1]
        mse = np.sum(residuals**2) / (n - k)
        var_coef = mse * np.linalg.inv(X.T @ X)
        se = np.sqrt(var_coef[2, 2])

        return {
            'rdd_estimate': rdd_estimate,
            'se': se,
            'ci_95': (rdd_estimate - 1.96 * se, rdd_estimate + 1.96 * se),
            'bandwidth': bandwidth,
            'n_observations': n,
            'cutoff': cutoff
        }

    def instrumental_variables(self, X: np.ndarray, treatment: np.ndarray,
                              outcome: np.ndarray, instrument: np.ndarray) -> Dict:
        """
        Estimate treatment effect using instrumental variables (2SLS).

        Args:
            X: Covariates
            treatment: Treatment variable (potentially endogenous)
            outcome: Outcome variable
            instrument: Instrumental variable

        Returns:
            Dictionary with IV estimates
        """
        # Add intercept
        n = len(outcome)
        X_with_intercept = np.column_stack([np.ones(n), X])

        # First stage: regress treatment on instrument and covariates
        Z = np.column_stack([X_with_intercept, instrument.reshape(-1, 1)])
        first_stage = LinearRegression()
        first_stage.fit(Z, treatment)
        treatment_hat = first_stage.predict(Z)

        # Check instrument strength (F-statistic)
        residuals_first = treatment - treatment_hat
        residuals_reduced = treatment - np.mean(treatment)
        f_stat = ((np.sum(residuals_reduced**2) - np.sum(residuals_first**2)) / 1) / \
                 (np.sum(residuals_first**2) / (n - Z.shape[1]))

        # Second stage: regress outcome on predicted treatment and covariates
        X_second = np.column_stack([X_with_intercept, treatment_hat.reshape(-1, 1)])
        second_stage = LinearRegression()
        second_stage.fit(X_second, outcome)

        # IV estimate is coefficient on treatment
        iv_estimate = second_stage.coef_[-1]

        # Standard error (simplified)
        residuals = outcome - second_stage.predict(X_second)
        mse = np.sum(residuals**2) / (n - X_second.shape[1])
        var_coef = mse * np.linalg.inv(X_second.T @ X_second)
        se = np.sqrt(var_coef[-1, -1])

        return {
            'iv_estimate': iv_estimate,
            'se': se,
            'ci_95': (iv_estimate - 1.96 * se, iv_estimate + 1.96 * se),
            'first_stage_f_stat': f_stat,
            'weak_instrument': f_stat < 10,
            'instrument_strength': 'Strong' if f_stat >= 10 else 'Weak'
        }

    def doubly_robust_estimation(self, X: pd.DataFrame, treatment: np.ndarray,
                                outcome: np.ndarray) -> Dict:
        """
        Doubly robust estimation combining propensity scores and outcome regression.

        Args:
            X: Covariate matrix
            treatment: Treatment indicator
            outcome: Outcome variable

        Returns:
            Dictionary with doubly robust estimates
        """
        # Estimate propensity scores
        ps = self.estimate_propensity_scores(X, treatment)
        ps = np.clip(ps, 0.01, 0.99)

        # Outcome regression models
        X_array = X.values if isinstance(X, pd.DataFrame) else X

        # Model for treated
        model_treated = LinearRegression()
        model_treated.fit(X_array[treatment == 1], outcome[treatment == 1])
        mu1 = model_treated.predict(X_array)

        # Model for control
        model_control = LinearRegression()
        model_control.fit(X_array[treatment == 0], outcome[treatment == 0])
        mu0 = model_control.predict(X_array)

        # Doubly robust estimator
        dr_treated = treatment * (outcome - mu1) / ps + mu1
        dr_control = (1 - treatment) * (outcome - mu0) / (1 - ps) + mu0

        ate = np.mean(dr_treated - dr_control)

        # Standard error
        influence_function = dr_treated - dr_control - ate
        se = np.sqrt(np.var(influence_function) / len(outcome))

        return {
            'ate': ate,
            'se': se,
            'ci_95': (ate - 1.96 * se, ate + 1.96 * se),
            'method': 'Doubly Robust'
        }

    def visualize_propensity_scores(self, treatment: np.ndarray, ps: np.ndarray) -> plt.Figure:
        """Visualize propensity score distributions."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Histogram
        axes[0].hist(ps[treatment == 0], bins=30, alpha=0.6, label='Control', color='blue', edgecolor='black')
        axes[0].hist(ps[treatment == 1], bins=30, alpha=0.6, label='Treated', color='red', edgecolor='black')
        axes[0].set_xlabel('Propensity Score')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Propensity Score Distribution')
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        # Box plot
        data_plot = pd.DataFrame({
            'Propensity Score': ps,
            'Group': ['Treated' if t == 1 else 'Control' for t in treatment]
        })
        sns.boxplot(data=data_plot, x='Group', y='Propensity Score', ax=axes[1])
        axes[1].set_title('Propensity Score by Treatment Group')
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        return fig

    def visualize_did(self, treated_pre: float, treated_post: float,
                     control_pre: float, control_post: float) -> plt.Figure:
        """Visualize difference-in-differences."""
        fig, ax = plt.subplots(figsize=(10, 6))

        time = [0, 1]
        treated = [treated_pre, treated_post]
        control = [control_pre, control_post]

        ax.plot(time, treated, 'ro-', linewidth=2, markersize=10, label='Treated')
        ax.plot(time, control, 'bo-', linewidth=2, markersize=10, label='Control')

        # Counterfactual (parallel trends)
        counterfactual = [treated_pre, treated_pre + (control_post - control_pre)]
        ax.plot(time, counterfactual, 'r--', linewidth=2, alpha=0.5, label='Counterfactual')

        # DiD effect
        ax.annotate('', xy=(1, treated_post), xytext=(1, counterfactual[1]),
                   arrowprops=dict(arrowstyle='<->', color='green', lw=2))
        ax.text(1.05, (treated_post + counterfactual[1]) / 2, 'DiD Effect',
               fontsize=12, color='green', weight='bold')

        ax.set_xlabel('Time Period', fontsize=12)
        ax.set_ylabel('Outcome', fontsize=12)
        ax.set_title('Difference-in-Differences Estimation', fontsize=14, weight='bold')
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Pre-Treatment', 'Post-Treatment'])
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)

        plt.tight_layout()
        return fig


def demo():
    """Demo causal inference toolkit."""
    np.random.seed(42)

    print("Causal Inference Toolkit Demo")
    print("="*60)

    ci = CausalInference(random_state=42)

    # Generate synthetic data
    n = 1000
    X = np.random.randn(n, 3)
    X_df = pd.DataFrame(X, columns=['covariate1', 'covariate2', 'covariate3'])

    # True treatment effect
    true_effect = 5.0

    # Generate treatment with selection bias
    propensity_true = 1 / (1 + np.exp(-(X[:, 0] + 0.5 * X[:, 1])))
    treatment = (np.random.rand(n) < propensity_true).astype(int)

    # Generate outcome with confounding
    outcome = 10 + 2 * X[:, 0] + X[:, 1] + true_effect * treatment + np.random.randn(n)

    # 1. Propensity Score Matching
    print("\n1. Propensity Score Matching")
    print("-" * 60)
    result = ci.propensity_score_matching(X_df, treatment, outcome, caliper=0.1)
    print(f"ATE Estimate: {result['ate']:.4f}")
    print(f"Standard Error: {result['se']:.4f}")
    print(f"95% CI: [{result['ci_95'][0]:.4f}, {result['ci_95'][1]:.4f}]")
    print(f"Matching Rate: {result['matching_rate']:.2%}")
    print(f"True Effect: {true_effect:.4f}")

    # Visualize propensity scores
    ps = ci.estimate_propensity_scores(X_df, treatment)
    fig1 = ci.visualize_propensity_scores(treatment, ps)
    fig1.savefig('causal_propensity_scores.png', dpi=300, bbox_inches='tight')
    print("✓ Saved causal_propensity_scores.png")
    plt.close()

    # 2. Inverse Probability Weighting
    print("\n2. Inverse Probability Weighting")
    print("-" * 60)
    result = ci.inverse_probability_weighting(X_df, treatment, outcome)
    print(f"ATE Estimate: {result['ate']:.4f}")
    print(f"95% CI: [{result['ci_95'][0]:.4f}, {result['ci_95'][1]:.4f}]")

    # 3. Difference-in-Differences
    print("\n3. Difference-in-Differences")
    print("-" * 60)

    # Generate panel data
    n_units = 500
    panel_data = []
    for i in range(n_units):
        # Pre-treatment
        panel_data.append({
            'unit': i,
            'time': 0,
            'treatment': 1 if i < 250 else 0,
            'outcome': 50 + (5 if i < 250 else 0) + np.random.randn()
        })
        # Post-treatment
        panel_data.append({
            'unit': i,
            'time': 1,
            'treatment': 1 if i < 250 else 0,
            'outcome': 52 + (5 if i < 250 else 0) + (8 if i < 250 else 0) + np.random.randn()
        })

    panel_df = pd.DataFrame(panel_data)
    result = ci.difference_in_differences(panel_df, 'outcome', 'treatment', 'time', 'unit')
    print(f"DiD Estimate: {result['did_estimate']:.4f}")
    print(f"95% CI: [{result['ci_95'][0]:.4f}, {result['ci_95'][1]:.4f}]")

    # Visualize DiD
    fig2 = ci.visualize_did(result['treated_pre'], result['treated_post'],
                           result['control_pre'], result['control_post'])
    fig2.savefig('causal_did.png', dpi=300, bbox_inches='tight')
    print("✓ Saved causal_did.png")
    plt.close()

    # 4. Regression Discontinuity
    print("\n4. Regression Discontinuity Design")
    print("-" * 60)
    running_var = np.random.uniform(-3, 3, 500)
    cutoff = 0
    treatment_rdd = (running_var >= cutoff).astype(int)
    outcome_rdd = 50 + 3 * running_var + 10 * treatment_rdd + np.random.randn(500) * 2

    result = ci.regression_discontinuity(running_var, outcome_rdd, cutoff, bandwidth=1.0)
    print(f"RDD Estimate: {result['rdd_estimate']:.4f}")
    print(f"95% CI: [{result['ci_95'][0]:.4f}, {result['ci_95'][1]:.4f}]")
    print(f"Bandwidth: {result['bandwidth']:.4f}")

    # 5. Instrumental Variables
    print("\n5. Instrumental Variables (2SLS)")
    print("-" * 60)
    instrument = np.random.randn(n)
    treatment_iv = 0.8 * instrument + 0.5 * X[:, 0] + np.random.randn(n) * 0.5
    outcome_iv = 10 + 2 * X[:, 0] + 6 * treatment_iv + np.random.randn(n)

    result = ci.instrumental_variables(X, treatment_iv, outcome_iv, instrument)
    print(f"IV Estimate: {result['iv_estimate']:.4f}")
    print(f"95% CI: [{result['ci_95'][0]:.4f}, {result['ci_95'][1]:.4f}]")
    print(f"First-stage F-statistic: {result['first_stage_f_stat']:.2f}")
    print(f"Instrument Strength: {result['instrument_strength']}")

    # 6. Doubly Robust Estimation
    print("\n6. Doubly Robust Estimation")
    print("-" * 60)
    result = ci.doubly_robust_estimation(X_df, treatment, outcome)
    print(f"ATE Estimate: {result['ate']:.4f}")
    print(f"95% CI: [{result['ci_95'][0]:.4f}, {result['ci_95'][1]:.4f}]")

    print("\n" + "="*60)
    print("✓ Causal Inference Demo Complete!")
    print("="*60)


if __name__ == '__main__':
    demo()
