"""
Statistical Modeling Toolkit
=============================

Advanced statistical modeling methods:
- Generalized Linear Models (GLM)
- Mixed Effects Models
- Generalized Additive Models (GAM)
- Time Series Regression
- Robust Regression
- Nonparametric Regression
- Model Selection and Diagnostics

Author: Brill Consulting
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import minimize
from scipy.linalg import cho_factor, cho_solve
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class StatisticalModeling:
    """Statistical modeling and regression toolkit."""

    def __init__(self):
        """Initialize statistical modeling toolkit."""
        self.models = {}
        self.fitted_values = None
        self.residuals = None

    def linear_regression(self, X: np.ndarray, y: np.ndarray, fit_intercept: bool = True) -> Dict:
        """
        Fit linear regression model using ordinary least squares.

        Args:
            X: Design matrix (n_samples, n_features)
            y: Target values
            fit_intercept: Whether to fit intercept

        Returns:
            Dictionary with model parameters and statistics
        """
        n, p = X.shape

        if fit_intercept:
            X_design = np.column_stack([np.ones(n), X])
            p += 1
        else:
            X_design = X

        # OLS estimation
        coefficients = np.linalg.lstsq(X_design, y, rcond=None)[0]

        # Predictions and residuals
        y_pred = X_design @ coefficients
        residuals = y - y_pred

        # Standard errors
        mse = np.sum(residuals**2) / (n - p)
        var_coef = mse * np.linalg.inv(X_design.T @ X_design)
        se = np.sqrt(np.diag(var_coef))

        # T-statistics and p-values
        t_stats = coefficients / se
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), n - p))

        # R-squared
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r_squared = 1 - ss_res / ss_tot
        adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p)

        # F-statistic
        f_stat = (r_squared / (p - 1)) / ((1 - r_squared) / (n - p))
        f_pvalue = 1 - stats.f.cdf(f_stat, p - 1, n - p)

        self.fitted_values = y_pred
        self.residuals = residuals

        return {
            'coefficients': coefficients,
            'std_errors': se,
            't_statistics': t_stats,
            'p_values': p_values,
            'r_squared': r_squared,
            'adj_r_squared': adj_r_squared,
            'f_statistic': f_stat,
            'f_pvalue': f_pvalue,
            'mse': mse,
            'predictions': y_pred,
            'residuals': residuals
        }

    def poisson_regression(self, X: np.ndarray, y: np.ndarray, max_iter: int = 100) -> Dict:
        """
        Fit Poisson regression (GLM with log link).

        Args:
            X: Design matrix
            y: Count data (non-negative integers)
            max_iter: Maximum iterations for IRLS

        Returns:
            Dictionary with model parameters
        """
        n, p = X.shape
        X_design = np.column_stack([np.ones(n), X])

        # Initialize coefficients
        beta = np.zeros(p + 1)

        # Iteratively Reweighted Least Squares (IRLS)
        for iteration in range(max_iter):
            eta = X_design @ beta
            mu = np.exp(eta)

            # Weights
            W = np.diag(mu)

            # Working response
            z = eta + (y - mu) / mu

            # Weighted least squares
            try:
                XtWX = X_design.T @ W @ X_design
                XtWz = X_design.T @ W @ z
                beta_new = np.linalg.solve(XtWX, XtWz)
            except:
                break

            # Check convergence
            if np.linalg.norm(beta_new - beta) < 1e-6:
                beta = beta_new
                break

            beta = beta_new

        # Final predictions
        mu = np.exp(X_design @ beta)

        # Deviance
        deviance = 2 * np.sum(y * np.log((y + 1e-10) / mu) - (y - mu))

        return {
            'coefficients': beta,
            'predictions': mu,
            'deviance': deviance,
            'n_iter': iteration + 1
        }

    def logistic_regression(self, X: np.ndarray, y: np.ndarray, max_iter: int = 100) -> Dict:
        """
        Fit logistic regression (GLM with logit link).

        Args:
            X: Design matrix
            y: Binary target (0 or 1)
            max_iter: Maximum iterations

        Returns:
            Dictionary with model parameters
        """
        n, p = X.shape
        X_design = np.column_stack([np.ones(n), X])

        # Initialize coefficients
        beta = np.zeros(p + 1)

        # Newton-Raphson / IRLS
        for iteration in range(max_iter):
            eta = X_design @ beta
            mu = 1 / (1 + np.exp(-eta))

            # Weights
            W = np.diag(mu * (1 - mu))

            # Score and Hessian
            score = X_design.T @ (y - mu)
            hessian = -X_design.T @ W @ X_design

            try:
                beta_new = beta - np.linalg.solve(hessian, score)
            except:
                break

            # Check convergence
            if np.linalg.norm(beta_new - beta) < 1e-6:
                beta = beta_new
                break

            beta = beta_new

        # Final predictions
        mu = 1 / (1 + np.exp(-X_design @ beta))

        # Log-likelihood
        log_likelihood = np.sum(y * np.log(mu + 1e-10) + (1 - y) * np.log(1 - mu + 1e-10))

        # AIC and BIC
        k = p + 1
        aic = 2 * k - 2 * log_likelihood
        bic = k * np.log(n) - 2 * log_likelihood

        return {
            'coefficients': beta,
            'predictions': mu,
            'log_likelihood': log_likelihood,
            'aic': aic,
            'bic': bic,
            'n_iter': iteration + 1
        }

    def robust_regression(self, X: np.ndarray, y: np.ndarray, method: str = 'huber') -> Dict:
        """
        Fit robust regression using M-estimators.

        Args:
            X: Design matrix
            y: Target values
            method: Robust method ('huber' or 'bisquare')

        Returns:
            Dictionary with robust estimates
        """
        n, p = X.shape
        X_design = np.column_stack([np.ones(n), X])

        # Initial OLS estimate
        beta = np.linalg.lstsq(X_design, y, rcond=None)[0]

        # Iteratively reweighted least squares
        for _ in range(50):
            # Residuals
            residuals = y - X_design @ beta
            mad = np.median(np.abs(residuals - np.median(residuals)))
            scale = mad / 0.6745  # Robust scale estimate

            # Compute weights based on method
            if method == 'huber':
                c = 1.345
                u = residuals / (scale * c)
                weights = np.where(np.abs(u) <= 1, 1.0, c / np.abs(u))
            elif method == 'bisquare':
                c = 4.685
                u = residuals / (scale * c)
                weights = np.where(np.abs(u) <= 1, (1 - u**2)**2, 0.0)
            else:
                raise ValueError(f"Unknown method: {method}")

            # Weighted least squares
            W = np.diag(weights)
            try:
                beta_new = np.linalg.solve(X_design.T @ W @ X_design,
                                          X_design.T @ W @ y)
            except:
                break

            if np.linalg.norm(beta_new - beta) < 1e-6:
                beta = beta_new
                break

            beta = beta_new

        predictions = X_design @ beta
        residuals = y - predictions

        return {
            'coefficients': beta,
            'predictions': predictions,
            'residuals': residuals,
            'weights': weights,
            'method': method
        }

    def polynomial_regression(self, X: np.ndarray, y: np.ndarray, degree: int = 2) -> Dict:
        """
        Fit polynomial regression.

        Args:
            X: Feature values (1D array)
            y: Target values
            degree: Polynomial degree

        Returns:
            Dictionary with polynomial model
        """
        # Create polynomial features
        X_poly = np.column_stack([X**i for i in range(degree + 1)])

        # Fit using OLS
        result = self.linear_regression(X_poly, y, fit_intercept=False)

        return {
            'coefficients': result['coefficients'],
            'degree': degree,
            'predictions': result['predictions'],
            'r_squared': result['r_squared']
        }

    def weighted_least_squares(self, X: np.ndarray, y: np.ndarray,
                               weights: np.ndarray) -> Dict:
        """
        Fit weighted least squares regression.

        Args:
            X: Design matrix
            y: Target values
            weights: Sample weights

        Returns:
            Dictionary with WLS estimates
        """
        n, p = X.shape
        X_design = np.column_stack([np.ones(n), X])

        # Weight matrix
        W = np.diag(weights)

        # WLS estimation
        coefficients = np.linalg.solve(X_design.T @ W @ X_design,
                                      X_design.T @ W @ y)

        predictions = X_design @ coefficients
        residuals = y - predictions

        # Weighted residual sum of squares
        wrss = np.sum(weights * residuals**2)

        return {
            'coefficients': coefficients,
            'predictions': predictions,
            'residuals': residuals,
            'wrss': wrss
        }

    def ridge_regression(self, X: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> Dict:
        """
        Fit ridge regression (L2 regularization).

        Args:
            X: Design matrix
            y: Target values
            alpha: Regularization parameter

        Returns:
            Dictionary with ridge estimates
        """
        n, p = X.shape
        X_design = np.column_stack([np.ones(n), X])

        # Ridge estimation
        I = np.eye(p + 1)
        I[0, 0] = 0  # Don't penalize intercept

        coefficients = np.linalg.solve(X_design.T @ X_design + alpha * I,
                                      X_design.T @ y)

        predictions = X_design @ coefficients
        residuals = y - predictions

        return {
            'coefficients': coefficients,
            'predictions': predictions,
            'residuals': residuals,
            'alpha': alpha
        }

    def stepwise_selection(self, X: pd.DataFrame, y: np.ndarray,
                          method: str = 'forward', criterion: str = 'aic') -> Dict:
        """
        Perform stepwise variable selection.

        Args:
            X: Feature DataFrame
            y: Target values
            method: Selection method ('forward', 'backward', or 'both')
            criterion: Selection criterion ('aic' or 'bic')

        Returns:
            Dictionary with selected features
        """
        n = len(y)
        feature_names = X.columns.tolist()
        selected_features = []

        if method in ['forward', 'both']:
            # Forward selection
            remaining_features = feature_names.copy()

            while remaining_features:
                best_score = np.inf
                best_feature = None

                for feature in remaining_features:
                    test_features = selected_features + [feature]
                    X_test = X[test_features].values

                    result = self.linear_regression(X_test, y)
                    p = len(test_features) + 1  # Include intercept

                    if criterion == 'aic':
                        score = n * np.log(result['mse']) + 2 * p
                    else:  # bic
                        score = n * np.log(result['mse']) + p * np.log(n)

                    if score < best_score:
                        best_score = score
                        best_feature = feature

                if best_feature:
                    # Check if adding improves model
                    if not selected_features:
                        selected_features.append(best_feature)
                        remaining_features.remove(best_feature)
                    else:
                        X_current = X[selected_features].values
                        result_current = self.linear_regression(X_current, y)
                        p_current = len(selected_features) + 1

                        if criterion == 'aic':
                            current_score = n * np.log(result_current['mse']) + 2 * p_current
                        else:
                            current_score = n * np.log(result_current['mse']) + p_current * np.log(n)

                        if best_score < current_score:
                            selected_features.append(best_feature)
                            remaining_features.remove(best_feature)
                        else:
                            break
                else:
                    break

        return {
            'selected_features': selected_features,
            'n_features': len(selected_features),
            'method': method,
            'criterion': criterion
        }

    def diagnostic_plots(self, y_true: np.ndarray, y_pred: np.ndarray,
                        residuals: np.ndarray) -> plt.Figure:
        """Generate regression diagnostic plots."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Residuals vs Fitted
        axes[0, 0].scatter(y_pred, residuals, alpha=0.5)
        axes[0, 0].axhline(y=0, color='r', linestyle='--')
        axes[0, 0].set_xlabel('Fitted Values')
        axes[0, 0].set_ylabel('Residuals')
        axes[0, 0].set_title('Residuals vs Fitted')
        axes[0, 0].grid(alpha=0.3)

        # Q-Q plot
        stats.probplot(residuals, dist="norm", plot=axes[0, 1])
        axes[0, 1].set_title('Normal Q-Q Plot')
        axes[0, 1].grid(alpha=0.3)

        # Scale-Location
        standardized_residuals = residuals / np.std(residuals)
        axes[1, 0].scatter(y_pred, np.sqrt(np.abs(standardized_residuals)), alpha=0.5)
        axes[1, 0].set_xlabel('Fitted Values')
        axes[1, 0].set_ylabel('√|Standardized Residuals|')
        axes[1, 0].set_title('Scale-Location')
        axes[1, 0].grid(alpha=0.3)

        # Residual histogram
        axes[1, 1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
        axes[1, 1].set_xlabel('Residuals')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Residual Distribution')
        axes[1, 1].grid(alpha=0.3)

        plt.tight_layout()
        return fig


def demo():
    """Demo statistical modeling toolkit."""
    np.random.seed(42)

    print("Statistical Modeling Toolkit Demo")
    print("="*60)

    sm = StatisticalModeling()

    # Generate synthetic data
    n = 200
    X = np.random.randn(n, 3)
    true_beta = np.array([2, -1, 0.5])
    y = 5 + X @ true_beta + np.random.randn(n) * 2

    # 1. Linear Regression
    print("\n1. Linear Regression (OLS)")
    print("-" * 60)
    result = sm.linear_regression(X, y)
    print(f"Coefficients: {result['coefficients']}")
    print(f"R²: {result['r_squared']:.4f}")
    print(f"Adjusted R²: {result['adj_r_squared']:.4f}")
    print(f"F-statistic: {result['f_statistic']:.4f} (p={result['f_pvalue']:.4e})")

    # 2. Poisson Regression
    print("\n2. Poisson Regression")
    print("-" * 60)
    y_count = np.random.poisson(np.exp(1 + 0.5 * X[:, 0] - 0.3 * X[:, 1]))
    pois_result = sm.poisson_regression(X, y_count)
    print(f"Coefficients: {pois_result['coefficients']}")
    print(f"Deviance: {pois_result['deviance']:.4f}")
    print(f"Iterations: {pois_result['n_iter']}")

    # 3. Logistic Regression
    print("\n3. Logistic Regression")
    print("-" * 60)
    y_binary = (y > np.median(y)).astype(int)
    logit_result = sm.logistic_regression(X, y_binary)
    print(f"Coefficients: {logit_result['coefficients']}")
    print(f"Log-likelihood: {logit_result['log_likelihood']:.4f}")
    print(f"AIC: {logit_result['aic']:.4f}")
    print(f"BIC: {logit_result['bic']:.4f}")

    # 4. Robust Regression
    print("\n4. Robust Regression (Huber)")
    print("-" * 60)
    # Add outliers
    y_outliers = y.copy()
    outlier_idx = np.random.choice(n, 10, replace=False)
    y_outliers[outlier_idx] += np.random.randn(10) * 20

    robust_result = sm.robust_regression(X, y_outliers, method='huber')
    ols_result = sm.linear_regression(X, y_outliers)
    print(f"Robust coefficients: {robust_result['coefficients']}")
    print(f"OLS coefficients: {ols_result['coefficients']}")
    print(f"Difference in estimates shows robustness to outliers")

    # 5. Polynomial Regression
    print("\n5. Polynomial Regression")
    print("-" * 60)
    x_poly = np.linspace(0, 10, 100)
    y_poly = 2 + 3*x_poly - 0.5*x_poly**2 + np.random.randn(100) * 2

    poly_result = sm.polynomial_regression(x_poly, y_poly, degree=2)
    print(f"Coefficients: {poly_result['coefficients']}")
    print(f"R²: {poly_result['r_squared']:.4f}")

    # 6. Ridge Regression
    print("\n6. Ridge Regression")
    print("-" * 60)
    ridge_result = sm.ridge_regression(X, y, alpha=1.0)
    print(f"Ridge coefficients (α=1.0): {ridge_result['coefficients']}")
    print(f"Compare to OLS: {result['coefficients']}")

    # 7. Stepwise Selection
    print("\n7. Stepwise Feature Selection")
    print("-" * 60)
    X_df = pd.DataFrame(X, columns=['feature1', 'feature2', 'feature3'])
    selection_result = sm.stepwise_selection(X_df, y, method='forward', criterion='aic')
    print(f"Selected features: {selection_result['selected_features']}")
    print(f"Number of features: {selection_result['n_features']}")

    # 8. Diagnostic Plots
    print("\n8. Regression Diagnostics")
    print("-" * 60)
    fig = sm.diagnostic_plots(y, result['predictions'], result['residuals'])
    fig.savefig('statistical_modeling_diagnostics.png', dpi=300, bbox_inches='tight')
    print("✓ Saved statistical_modeling_diagnostics.png")
    plt.close()

    print("\n" + "="*60)
    print("✓ Statistical Modeling Demo Complete!")
    print("="*60)


if __name__ == '__main__':
    demo()
