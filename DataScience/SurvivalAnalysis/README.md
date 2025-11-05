# Survival Analysis Toolkit

Advanced survival analysis and time-to-event modeling with Kaplan-Meier curves, Cox proportional hazards, and parametric survival models.

## Overview

The Survival Analysis Toolkit provides comprehensive methods for analyzing time-to-event data. It implements Kaplan-Meier estimation, log-rank tests, Cox proportional hazards regression, and parametric survival models.

## Key Features

- **Kaplan-Meier Estimator**: Non-parametric survival function estimation
- **Cox Proportional Hazards**: Regression modeling for survival data
- **Log-Rank Test**: Compare survival curves between groups
- **Weibull Survival Model**: Parametric survival modeling
- **Confidence Intervals**: Greenwood's formula for standard errors
- **Hazard Ratios**: Effect size measures for covariates
- **Censored Data Handling**: Proper treatment of right-censored observations
- **Visualization**: Survival curves with confidence bands

## Technologies Used

- **NumPy**: Numerical computing
- **Pandas**: Data manipulation
- **SciPy**: Statistical analysis and optimization
- **Matplotlib & Seaborn**: Visualization

## Installation

```bash
cd SurvivalAnalysis/
pip install numpy pandas scipy matplotlib seaborn
```

## Usage Examples

### Kaplan-Meier Estimation

```python
from survivalanalysis import SurvivalAnalysis

sa = SurvivalAnalysis()

# Estimate survival function
km_result = sa.kaplan_meier(times, events)

print(f"Median survival: {km_result['median_survival']:.2f}")
print(f"Survival at t=10: {km_result['survival_probs'][10]:.3f}")

# Visualize
fig = sa.visualize_survival_curves(km_result, title="Kaplan-Meier Curve")
fig.savefig('km_curve.png')
```

### Log-Rank Test

```python
# Compare two groups
lr_result = sa.log_rank_test(times1, events1, times2, events2)

print(f"Chi-square: {lr_result['chi_square']:.4f}")
print(f"P-value: {lr_result['p_value']:.4e}")
print(f"Significant difference: {lr_result['significant']}")
```

### Cox Proportional Hazards

```python
# Fit Cox regression
cox_result = sa.cox_proportional_hazards(times, events, X)

print(f"Hazard Ratios: {cox_result['hazard_ratios']}")
print(f"95% CI: [{cox_result['hr_ci_lower']}, {cox_result['hr_ci_upper']}]")
print(f"P-values: {cox_result['p_values']}")
```

### Weibull Survival Model

```python
# Fit parametric model
weibull_result = sa.weibull_survival(times, events)

print(f"Shape: {weibull_result['shape']:.4f}")
print(f"Scale: {weibull_result['scale']:.4f}")
print(f"Median survival: {weibull_result['median_survival']:.2f}")
```

## Demo

```bash
python survivalanalysis.py
```

The demo includes:
- Kaplan-Meier survival curves for two groups
- Log-rank test for comparing groups
- Cox proportional hazards regression
- Weibull parametric survival model
- Survival curve visualization
- Comprehensive survival statistics

## Output Examples

- `survival_kaplan_meier.png`: Kaplan-Meier curve with confidence intervals
- `survival_comparison.png`: Comparison of Kaplan-Meier and Weibull fits
- Console output with median survival times and test statistics

## Key Concepts

**Survival Function**: S(t) = Probability of surviving beyond time t

**Hazard Function**: Instantaneous risk of event at time t

**Censoring**: Observations where event has not occurred by end of study

**Hazard Ratio**: Relative risk of event for one group vs another

## Applications

- Clinical trials and medical research
- Customer churn analysis
- Equipment failure prediction
- Employee retention studies
- Warranty analysis
- Loan default modeling

## Author

**Brill Consulting**
- Email: clientbrill@gmail.com
- LinkedIn: [brillconsulting](https://www.linkedin.com/in/brillconsulting)
