# Statistical Modeling Toolkit

Advanced statistical modeling with generalized linear models, robust regression, and model diagnostics.

## Overview

The Statistical Modeling Toolkit provides comprehensive regression methods including linear, logistic, Poisson, robust, and ridge regression. It features model selection, diagnostics, and comprehensive visualization.

## Key Features

- **Linear Regression**: OLS with full inference and diagnostics
- **Generalized Linear Models (GLM)**: Poisson and logistic regression
- **Robust Regression**: M-estimators (Huber and Bisquare)
- **Polynomial Regression**: Fit nonlinear relationships
- **Ridge Regression**: L2 regularization for high-dimensional data
- **Weighted Least Squares**: Accommodate heteroscedasticity
- **Stepwise Selection**: Forward, backward, and bidirectional feature selection
- **Model Diagnostics**: Residual plots, Q-Q plots, and goodness-of-fit tests

## Technologies Used

- **NumPy**: Numerical computing
- **Pandas**: Data manipulation
- **SciPy**: Statistical analysis and optimization
- **Matplotlib & Seaborn**: Visualization

## Installation

```bash
cd StatisticalModeling/
pip install numpy pandas scipy matplotlib seaborn
```

## Usage Examples

### Linear Regression

```python
from statisticalmodeling import StatisticalModeling

sm = StatisticalModeling()

# Fit OLS regression
result = sm.linear_regression(X, y, fit_intercept=True)
print(f"Coefficients: {result['coefficients']}")
print(f"R²: {result['r_squared']:.4f}")
print(f"Adjusted R²: {result['adj_r_squared']:.4f}")
print(f"F-statistic: {result['f_statistic']:.4f}")
print(f"P-values: {result['p_values']}")
```

### Poisson Regression

```python
# Fit Poisson GLM for count data
pois_result = sm.poisson_regression(X, y_count, max_iter=100)
print(f"Coefficients: {pois_result['coefficients']}")
print(f"Deviance: {pois_result['deviance']:.4f}")
```

### Robust Regression

```python
# Fit robust regression (resistant to outliers)
robust_result = sm.robust_regression(X, y_with_outliers, method='huber')
print(f"Robust coefficients: {robust_result['coefficients']}")
print(f"Method: {robust_result['method']}")
```

### Model Diagnostics

```python
# Generate diagnostic plots
fig = sm.diagnostic_plots(y, result['predictions'], result['residuals'])
fig.savefig('diagnostics.png')
```

## Demo

```bash
python statisticalmodeling.py
```

The demo includes:
- Linear regression (OLS)
- Poisson regression
- Logistic regression
- Robust regression (Huber)
- Polynomial regression
- Ridge regression
- Stepwise feature selection
- Comprehensive model diagnostics

## Output Examples

- `statistical_modeling_diagnostics.png`: Residual plots, Q-Q plot, scale-location, and residual distribution
- Console output with coefficients, significance tests, and model fit statistics

## Author

**Brill Consulting**
- Email: clientbrill@gmail.com
- LinkedIn: [brillconsulting](https://www.linkedin.com/in/brillconsulting)
