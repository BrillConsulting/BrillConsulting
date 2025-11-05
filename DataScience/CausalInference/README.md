# Causal Inference Toolkit

Advanced causal inference methods for establishing causality from observational data using propensity scores, difference-in-differences, and instrumental variables.

## Overview

The Causal Inference Toolkit provides state-of-the-art methods for estimating causal effects from observational data. It implements propensity score matching, inverse probability weighting, difference-in-differences, regression discontinuity, instrumental variables, and doubly robust estimation.

## Key Features

- **Propensity Score Matching (PSM)**: Match treated and control units based on propensity scores
- **Inverse Probability Weighting (IPW)**: Weight observations by inverse propensity scores
- **Difference-in-Differences (DiD)**: Estimate causal effects using panel data
- **Regression Discontinuity Design (RDD)**: Exploit discontinuities in treatment assignment
- **Instrumental Variables (2SLS)**: Address endogeneity using instrumental variables
- **Doubly Robust Estimation**: Combine propensity scores and outcome regression
- **Synthetic Control**: Create synthetic comparison groups
- **Visualization**: Comprehensive plots for propensity scores and DiD analysis

## Technologies Used

- **NumPy**: Numerical computing
- **Pandas**: Data manipulation
- **SciPy**: Statistical analysis and optimization
- **Scikit-learn**: Machine learning models for propensity scores
- **Matplotlib & Seaborn**: Visualization

## Installation

```bash
cd CausalInference/
pip install numpy pandas scipy scikit-learn matplotlib seaborn
```

## Usage Examples

### Propensity Score Matching

```python
from causalinference import CausalInference
import pandas as pd

ci = CausalInference(random_state=42)

# Estimate treatment effect using PSM
result = ci.propensity_score_matching(
    X=covariates_df,
    treatment=treatment_array,
    outcome=outcome_array,
    caliper=0.1
)

print(f"ATE: {result['ate']:.4f}")
print(f"95% CI: [{result['ci_95'][0]:.4f}, {result['ci_95'][1]:.4f}]")
print(f"Matching rate: {result['matching_rate']:.2%}")

# Visualize propensity scores
ps = ci.estimate_propensity_scores(X, treatment)
fig = ci.visualize_propensity_scores(treatment, ps)
```

### Difference-in-Differences

```python
# Analyze panel data with DiD
result = ci.difference_in_differences(
    data=panel_df,
    outcome_col='revenue',
    treatment_col='treated',
    time_col='post_treatment',
    unit_col='store_id'
)

print(f"DiD Estimate: {result['did_estimate']:.4f}")
print(f"95% CI: [{result['ci_95'][0]:.4f}, {result['ci_95'][1]:.4f}]")

# Visualize DiD
fig = ci.visualize_did(
    result['treated_pre'],
    result['treated_post'],
    result['control_pre'],
    result['control_post']
)
```

### Instrumental Variables

```python
# 2SLS estimation
result = ci.instrumental_variables(
    X=covariates,
    treatment=endogenous_treatment,
    outcome=outcome,
    instrument=instrument_variable
)

print(f"IV Estimate: {result['iv_estimate']:.4f}")
print(f"First-stage F-stat: {result['first_stage_f_stat']:.2f}")
print(f"Instrument strength: {result['instrument_strength']}")
```

## Demo

Run the comprehensive demo:

```bash
python causalinference.py
```

The demo includes:
- Propensity score matching with visualization
- Inverse probability weighting
- Difference-in-differences analysis
- Regression discontinuity design
- Instrumental variables estimation
- Doubly robust estimation

## Output Examples

The demo generates:
- `causal_propensity_scores.png`: Propensity score distributions by group
- `causal_did.png`: Difference-in-differences visualization
- Console output with treatment effect estimates and confidence intervals

## Key Concepts

**Average Treatment Effect (ATE)**: The average causal effect of treatment across the population

**Propensity Score**: Probability of receiving treatment given observed covariates

**Parallel Trends**: Key assumption for DiD that treatment and control groups would follow parallel trends absent treatment

**Instrument**: Variable that affects treatment but only affects outcome through treatment

## Applications

- Policy evaluation and impact assessment
- Marketing campaign effectiveness
- Healthcare intervention analysis
- Economic policy analysis
- Program evaluation
- Treatment effect heterogeneity

## Author

**Brill Consulting**
- Email: clientbrill@gmail.com
- LinkedIn: [brillconsulting](https://www.linkedin.com/in/brillconsulting)
