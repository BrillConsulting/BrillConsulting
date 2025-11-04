# A/B Testing Toolkit

Statistical experimentation and hypothesis testing toolkit for data-driven decision making.

## Features

- **T-Tests**: Independent and paired samples
- **Proportion Tests**: Conversion rate comparisons
- **Chi-Square Tests**: Categorical data analysis
- **ANOVA**: Multiple group comparisons
- **Sample Size Calculation**: Power analysis
- **Bayesian A/B Testing**: Probabilistic approach
- **Effect Size Metrics**: Cohen's d, relative lift
- **Visualization**: Comprehensive result plots

## Technologies

- SciPy: Statistical tests
- NumPy: Numerical computations
- Matplotlib, Seaborn: Visualization

## Usage

```python
from ab_testing import ABTester

tester = ABTester(alpha=0.05)

# T-test
result = tester.ttest_independent(control_group, treatment_group)

# Proportion test
result = tester.proportion_test(conv_a, n_a, conv_b, n_b)

# Sample size
n = tester.calculate_sample_size(baseline_rate=0.02, mde=0.002)

# Bayesian test
result = tester.bayesian_ab_test(success_a, trials_a, success_b, trials_b)
```

## Demo

```bash
python ab_testing.py
```
