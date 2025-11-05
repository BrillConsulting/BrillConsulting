# Experiment Design Toolkit

Advanced experimental design and analysis methods for scientific experiments, factorial designs, and ANOVA.

## Overview

The Experiment Design Toolkit provides comprehensive methods for designing and analyzing scientific experiments. It implements factorial designs, ANOVA, power analysis, response surface methodology, and various blocking strategies.

## Key Features

- **Factorial Design**: Full and fractional factorial designs (2^k, 2^(k-p))
- **ANOVA**: One-way, two-way, and repeated measures ANOVA
- **Power Analysis**: Sample size and power calculations
- **Randomization Strategies**: Randomized complete block design (RCBD)
- **Latin Square Design**: Efficient experimental layouts
- **Response Surface Methodology (RSM)**: Optimize multi-factor processes
- **Tukey HSD**: Post-hoc pairwise comparisons
- **Visualization**: Interaction plots, factorial design matrices, ANOVA diagnostics

## Technologies Used

- **NumPy**: Numerical computing
- **Pandas**: Data manipulation
- **SciPy**: Statistical analysis
- **Matplotlib & Seaborn**: Visualization

## Installation

```bash
cd ExperimentDesign/
pip install numpy pandas scipy matplotlib seaborn
```

## Usage Examples

### Full Factorial Design

```python
from experimentdesign import ExperimentDesign

ed = ExperimentDesign(random_state=42)

# Create 2^3 factorial design
factors = {
    'Temperature': [20, 30],
    'Pressure': [1, 2],
    'Catalyst': ['A', 'B']
}
design = ed.full_factorial_design(factors, replicates=2)
print(design)
```

### ANOVA Analysis

```python
# One-way ANOVA
result = ed.one_way_anova(data, group_col='treatment', value_col='response')
print(f"F-statistic: {result['f_statistic']:.4f}")
print(f"P-value: {result['p_value']:.4e}")
print(f"Effect size (η²): {result['eta_squared']:.4f}")

# Two-way ANOVA with interaction
result = ed.two_way_anova(data, 'factor_a', 'factor_b', 'response')
print(f"Factor A significant: {result['factor_a']['significant']}")
print(f"Interaction significant: {result['interaction']['significant']}")
```

### Power Analysis

```python
# Calculate required sample size
power_result = ed.power_analysis(
    effect_size=0.5,
    alpha=0.05,
    power=0.8,
    n_groups=3
)
print(f"Required n per group: {power_result['required_n_per_group']}")
```

## Demo

```bash
python experimentdesign.py
```

The demo includes:
- Full factorial design creation
- One-way and two-way ANOVA
- Power analysis and sample size calculation
- Fractional factorial design
- Latin square design
- Randomized complete block design
- Tukey HSD post-hoc test
- Response surface methodology

## Output Examples

- `experiment_factorial_design.png`: Factorial design layout with responses
- `experiment_anova_results.png`: ANOVA diagnostic plots
- Console output with F-statistics, p-values, and effect sizes

## Author

**Brill Consulting**
- Email: clientbrill@gmail.com
- LinkedIn: [brillconsulting](https://www.linkedin.com/in/brillconsulting)
