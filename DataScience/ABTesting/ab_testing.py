"""
A/B Testing Toolkit
===================

Statistical testing and experiment analysis:
- T-tests, Chi-square tests, ANOVA
- Sample size calculation
- Power analysis
- Bayesian A/B testing
- Sequential testing
- Experiment result interpretation

Author: Brill Consulting
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency, f_oneway
from typing import Tuple, Dict, Optional
import matplotlib.pyplot as plt
import seaborn as sns


class ABTester:
    """A/B testing and statistical analysis toolkit."""

    def __init__(self, alpha: float = 0.05):
        """Initialize with significance level."""
        self.alpha = alpha

    def ttest_independent(self, group_a: np.ndarray, group_b: np.ndarray) -> Dict:
        """Perform independent samples t-test."""
        t_stat, p_value = stats.ttest_ind(group_a, group_b)

        effect_size = (group_a.mean() - group_b.mean()) / np.sqrt((group_a.std()**2 + group_b.std()**2) / 2)

        result = {
            'test': 'Independent T-Test',
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < self.alpha,
            'effect_size_cohens_d': effect_size,
            'mean_a': group_a.mean(),
            'mean_b': group_b.mean(),
            'std_a': group_a.std(),
            'std_b': group_b.std(),
            'n_a': len(group_a),
            'n_b': len(group_b)
        }

        return result

    def chi_square_test(self, contingency_table: np.ndarray) -> Dict:
        """Perform chi-square test of independence."""
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)

        result = {
            'test': 'Chi-Square Test',
            'chi2_statistic': chi2,
            'p_value': p_value,
            'degrees_of_freedom': dof,
            'significant': p_value < self.alpha,
            'expected_frequencies': expected
        }

        return result

    def proportion_test(self, successes_a: int, n_a: int,
                       successes_b: int, n_b: int) -> Dict:
        """Test difference in proportions (e.g., conversion rates)."""
        p_a = successes_a / n_a
        p_b = successes_b / n_b

        p_pooled = (successes_a + successes_b) / (n_a + n_b)
        se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n_a + 1/n_b))

        z_stat = (p_a - p_b) / se
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

        result = {
            'test': 'Proportion Z-Test',
            'z_statistic': z_stat,
            'p_value': p_value,
            'significant': p_value < self.alpha,
            'proportion_a': p_a,
            'proportion_b': p_b,
            'difference': p_a - p_b,
            'relative_lift': (p_a - p_b) / p_b * 100 if p_b > 0 else None
        }

        return result

    def anova_test(self, *groups) -> Dict:
        """Perform one-way ANOVA test for multiple groups."""
        f_stat, p_value = f_oneway(*groups)

        result = {
            'test': 'One-Way ANOVA',
            'f_statistic': f_stat,
            'p_value': p_value,
            'significant': p_value < self.alpha,
            'n_groups': len(groups),
            'group_means': [g.mean() for g in groups]
        }

        return result

    def calculate_sample_size(self, baseline_rate: float, mde: float,
                             alpha: float = 0.05, power: float = 0.8) -> int:
        """Calculate required sample size per group."""
        effect_size = mde / np.sqrt(baseline_rate * (1 - baseline_rate))

        z_alpha = stats.norm.ppf(1 - alpha / 2)
        z_beta = stats.norm.ppf(power)

        n = 2 * ((z_alpha + z_beta) / effect_size) ** 2

        return int(np.ceil(n))

    def calculate_confidence_interval(self, data: np.ndarray, confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate confidence interval for mean."""
        mean = data.mean()
        se = stats.sem(data)
        ci = stats.t.interval(confidence, len(data)-1, mean, se)

        return ci

    def bayesian_ab_test(self, successes_a: int, trials_a: int,
                        successes_b: int, trials_b: int, n_samples: int = 10000) -> Dict:
        """Bayesian A/B test using Beta distribution."""
        # Prior: Beta(1, 1) - uniform
        # Posterior: Beta(1 + successes, 1 + failures)

        alpha_a = 1 + successes_a
        beta_a = 1 + (trials_a - successes_a)

        alpha_b = 1 + successes_b
        beta_b = 1 + (trials_b - successes_b)

        # Sample from posterior distributions
        samples_a = np.random.beta(alpha_a, beta_a, n_samples)
        samples_b = np.random.beta(alpha_b, beta_b, n_samples)

        # Probability that B is better than A
        prob_b_better = (samples_b > samples_a).mean()

        result = {
            'test': 'Bayesian A/B Test',
            'prob_b_better_than_a': prob_b_better,
            'prob_a_better_than_b': 1 - prob_b_better,
            'expected_loss_choosing_b': np.maximum(samples_a - samples_b, 0).mean(),
            'expected_loss_choosing_a': np.maximum(samples_b - samples_a, 0).mean()
        }

        return result

    def visualize_results(self, group_a: np.ndarray, group_b: np.ndarray,
                         labels: Tuple[str, str] = ('Control', 'Treatment')) -> plt.Figure:
        """Visualize A/B test results."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Distribution comparison
        axes[0, 0].hist(group_a, bins=30, alpha=0.5, label=labels[0], color='blue', edgecolor='black')
        axes[0, 0].hist(group_b, bins=30, alpha=0.5, label=labels[1], color='red', edgecolor='black')
        axes[0, 0].axvline(group_a.mean(), color='blue', linestyle='--', linewidth=2)
        axes[0, 0].axvline(group_b.mean(), color='red', linestyle='--', linewidth=2)
        axes[0, 0].set_title('Distribution Comparison')
        axes[0, 0].set_xlabel('Value')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].legend()

        # Box plot
        data_df = pd.DataFrame({
            'value': np.concatenate([group_a, group_b]),
            'group': [labels[0]]*len(group_a) + [labels[1]]*len(group_b)
        })
        sns.boxplot(data=data_df, x='group', y='value', ax=axes[0, 1])
        axes[0, 1].set_title('Box Plot Comparison')

        # Violin plot
        sns.violinplot(data=data_df, x='group', y='value', ax=axes[1, 0])
        axes[1, 0].set_title('Violin Plot Comparison')

        # Summary statistics
        summary_text = f'{labels[0]}:\nMean: {group_a.mean():.2f}\nStd: {group_a.std():.2f}\nn: {len(group_a)}\n\n'
        summary_text += f'{labels[1]}:\nMean: {group_b.mean():.2f}\nStd: {group_b.std():.2f}\nn: {len(group_b)}\n\n'

        result = self.ttest_independent(group_a, group_b)
        summary_text += f'T-test:\np-value: {result["p_value"]:.4f}\n'
        summary_text += f'Significant: {result["significant"]}\n'
        summary_text += f"Cohen's d: {result['effect_size_cohens_d']:.3f}"

        axes[1, 1].text(0.1, 0.5, summary_text, transform=axes[1, 1].transAxes,
                       fontsize=11, verticalalignment='center',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        axes[1, 1].set_title('Summary Statistics')
        axes[1, 1].axis('off')

        plt.tight_layout()
        return fig


def demo():
    """Demo A/B testing."""
    np.random.seed(42)

    print("A/B Testing Demo")
    print("="*50)

    tester = ABTester(alpha=0.05)

    # Scenario 1: Continuous metric (e.g., time spent)
    print("\n1. T-Test: Average Time Spent")
    control = np.random.normal(100, 15, 1000)
    treatment = np.random.normal(105, 15, 1000)

    result = tester.ttest_independent(control, treatment)
    print(f"Control mean: {result['mean_a']:.2f}")
    print(f"Treatment mean: {result['mean_b']:.2f}")
    print(f"P-value: {result['p_value']:.4f}")
    print(f"Significant: {result['significant']}")
    print(f"Effect size: {result['effect_size_cohens_d']:.3f}")

    # Scenario 2: Conversion rate test
    print("\n2. Proportion Test: Conversion Rate")
    conversions_a = 230
    visitors_a = 10000
    conversions_b = 275
    visitors_b = 10000

    result = tester.proportion_test(conversions_a, visitors_a, conversions_b, visitors_b)
    print(f"Control conversion: {result['proportion_a']:.2%}")
    print(f"Treatment conversion: {result['proportion_b']:.2%}")
    print(f"Relative lift: {result['relative_lift']:.2f}%")
    print(f"P-value: {result['p_value']:.4f}")
    print(f"Significant: {result['significant']}")

    # Scenario 3: Sample size calculation
    print("\n3. Sample Size Calculation")
    required_n = tester.calculate_sample_size(baseline_rate=0.023, mde=0.002, power=0.8)
    print(f"Required sample size per group: {required_n}")

    # Scenario 4: Bayesian A/B test
    print("\n4. Bayesian A/B Test")
    result = tester.bayesian_ab_test(230, 10000, 275, 10000)
    print(f"Probability B better than A: {result['prob_b_better_than_a']:.2%}")
    print(f"Expected loss choosing A: {result['expected_loss_choosing_a']:.6f}")
    print(f"Expected loss choosing B: {result['expected_loss_choosing_b']:.6f}")

    # Visualization
    print("\n5. Generating visualization...")
    fig = tester.visualize_results(control, treatment, ('Control', 'Treatment'))
    fig.savefig('ab_test_results.png', dpi=300, bbox_inches='tight')
    print("✓ Saved ab_test_results.png")
    plt.close()

    print("\n✓ A/B Testing Complete!")


if __name__ == '__main__':
    demo()
