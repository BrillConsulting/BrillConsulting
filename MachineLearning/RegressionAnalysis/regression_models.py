"""
Advanced Regression Analysis System
Author: BrillConsulting
Description: Production-ready regression with 10+ algorithms including XGBoost, LightGBM, and ensemble methods
Version: 2.0 - Enhanced with gradient boosting and advanced diagnostics
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import argparse
import joblib
import warnings
warnings.filterwarnings('ignore')

# Try to import XGBoost and LightGBM (optional dependencies)
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


class RegressionAnalyzer:
    """
    Complete regression analysis system with multiple algorithms
    """

    def __init__(self, scale_features: bool = True):
        """
        Initialize regression analyzer

        Args:
            scale_features: Whether to standardize features
        """
        self.scale_features = scale_features
        self.scaler = StandardScaler() if scale_features else None
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None

    def prepare_data(self, X: pd.DataFrame, y: pd.Series,
                    test_size: float = 0.2,
                    random_state: int = 42) -> Tuple:
        """
        Prepare data for training

        Args:
            X: Features
            y: Target variable
            test_size: Test set proportion
            random_state: Random seed

        Returns:
            X_train, X_test, y_train, y_test
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # Scale features if needed
        if self.scale_features:
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)

        return X_train, X_test, y_train, y_test

    def train_linear_regression(self, X_train, y_train, X_test, y_test) -> Dict:
        """Train simple linear regression"""
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        # Metrics
        results = {
            'model': model,
            'train_r2': r2_score(y_train, y_pred_train),
            'test_r2': r2_score(y_test, y_pred_test),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'train_mae': mean_absolute_error(y_train, y_pred_train),
            'test_mae': mean_absolute_error(y_test, y_pred_test),
            'coefficients': model.coef_,
            'intercept': model.intercept_
        }

        self.models['Linear Regression'] = model
        self.results['Linear Regression'] = results
        return results

    def train_ridge_regression(self, X_train, y_train, X_test, y_test,
                               alpha_range: List[float] = None) -> Dict:
        """Train Ridge regression with hyperparameter tuning"""
        if alpha_range is None:
            alpha_range = [0.001, 0.01, 0.1, 1, 10, 100]

        # Grid search for best alpha
        ridge = Ridge()
        param_grid = {'alpha': alpha_range}
        grid_search = GridSearchCV(ridge, param_grid, cv=5,
                                   scoring='neg_mean_squared_error')
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_

        # Predictions
        y_pred_train = best_model.predict(X_train)
        y_pred_test = best_model.predict(X_test)

        results = {
            'model': best_model,
            'best_alpha': grid_search.best_params_['alpha'],
            'train_r2': r2_score(y_train, y_pred_train),
            'test_r2': r2_score(y_test, y_pred_test),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'train_mae': mean_absolute_error(y_train, y_pred_train),
            'test_mae': mean_absolute_error(y_test, y_pred_test),
            'coefficients': best_model.coef_,
            'intercept': best_model.intercept_
        }

        self.models['Ridge Regression'] = best_model
        self.results['Ridge Regression'] = results
        return results

    def train_lasso_regression(self, X_train, y_train, X_test, y_test,
                               alpha_range: List[float] = None) -> Dict:
        """Train Lasso regression with hyperparameter tuning"""
        if alpha_range is None:
            alpha_range = [0.001, 0.01, 0.1, 1, 10]

        # Grid search
        lasso = Lasso(max_iter=10000)
        param_grid = {'alpha': alpha_range}
        grid_search = GridSearchCV(lasso, param_grid, cv=5,
                                   scoring='neg_mean_squared_error')
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_

        # Predictions
        y_pred_train = best_model.predict(X_train)
        y_pred_test = best_model.predict(X_test)

        # Feature selection (non-zero coefficients)
        selected_features = np.sum(best_model.coef_ != 0)

        results = {
            'model': best_model,
            'best_alpha': grid_search.best_params_['alpha'],
            'train_r2': r2_score(y_train, y_pred_train),
            'test_r2': r2_score(y_test, y_pred_test),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'train_mae': mean_absolute_error(y_train, y_pred_train),
            'test_mae': mean_absolute_error(y_test, y_pred_test),
            'selected_features': selected_features,
            'coefficients': best_model.coef_,
            'intercept': best_model.intercept_
        }

        self.models['Lasso Regression'] = best_model
        self.results['Lasso Regression'] = results
        return results

    def train_polynomial_regression(self, X_train, y_train, X_test, y_test,
                                    degree: int = 2) -> Dict:
        """Train polynomial regression"""
        # Create pipeline
        pipeline = Pipeline([
            ('poly', PolynomialFeatures(degree=degree)),
            ('linear', LinearRegression())
        ])

        pipeline.fit(X_train, y_train)

        # Predictions
        y_pred_train = pipeline.predict(X_train)
        y_pred_test = pipeline.predict(X_test)

        results = {
            'model': pipeline,
            'degree': degree,
            'train_r2': r2_score(y_train, y_pred_train),
            'test_r2': r2_score(y_test, y_pred_test),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'train_mae': mean_absolute_error(y_train, y_pred_train),
            'test_mae': mean_absolute_error(y_test, y_pred_test)
        }

        self.models[f'Polynomial Regression (degree={degree})'] = pipeline
        self.results[f'Polynomial Regression (degree={degree})'] = results
        return results

    def train_random_forest(self, X_train, y_train, X_test, y_test) -> Dict:
        """Train Random Forest Regressor with hyperparameter tuning"""
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }

        rf = RandomForestRegressor(random_state=42)
        grid_search = GridSearchCV(rf, param_grid, cv=3,
                                   scoring='neg_mean_squared_error', n_jobs=-1)
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_

        # Predictions
        y_pred_train = best_model.predict(X_train)
        y_pred_test = best_model.predict(X_test)

        results = {
            'model': best_model,
            'best_params': grid_search.best_params_,
            'train_r2': r2_score(y_train, y_pred_train),
            'test_r2': r2_score(y_test, y_pred_test),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'train_mae': mean_absolute_error(y_train, y_pred_train),
            'test_mae': mean_absolute_error(y_test, y_pred_test),
            'feature_importance': best_model.feature_importances_ if hasattr(best_model, 'feature_importances_') else None
        }

        self.models['Random Forest'] = best_model
        self.results['Random Forest'] = results
        return results

    def train_gradient_boosting(self, X_train, y_train, X_test, y_test) -> Dict:
        """Train Gradient Boosting Regressor"""
        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 1.0]
        }

        gb = GradientBoostingRegressor(random_state=42)
        grid_search = GridSearchCV(gb, param_grid, cv=3,
                                   scoring='neg_mean_squared_error', n_jobs=-1)
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_

        y_pred_train = best_model.predict(X_train)
        y_pred_test = best_model.predict(X_test)

        results = {
            'model': best_model,
            'best_params': grid_search.best_params_,
            'train_r2': r2_score(y_train, y_pred_train),
            'test_r2': r2_score(y_test, y_pred_test),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'train_mae': mean_absolute_error(y_train, y_pred_train),
            'test_mae': mean_absolute_error(y_test, y_pred_test),
            'feature_importance': best_model.feature_importances_
        }

        self.models['Gradient Boosting'] = best_model
        self.results['Gradient Boosting'] = results
        return results

    def train_svr(self, X_train, y_train, X_test, y_test) -> Dict:
        """Train Support Vector Regression"""
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'kernel': ['rbf', 'linear'],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1]
        }

        svr = SVR()
        grid_search = GridSearchCV(svr, param_grid, cv=3,
                                   scoring='neg_mean_squared_error', n_jobs=-1)
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_

        y_pred_train = best_model.predict(X_train)
        y_pred_test = best_model.predict(X_test)

        results = {
            'model': best_model,
            'best_params': grid_search.best_params_,
            'train_r2': r2_score(y_train, y_pred_train),
            'test_r2': r2_score(y_test, y_pred_test),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'train_mae': mean_absolute_error(y_train, y_pred_train),
            'test_mae': mean_absolute_error(y_test, y_pred_test)
        }

        self.models['SVR'] = best_model
        self.results['SVR'] = results
        return results

    def train_xgboost(self, X_train, y_train, X_test, y_test) -> Dict:
        """Train XGBoost Regressor (if available)"""
        if not XGBOOST_AVAILABLE:
            print("⚠️  XGBoost not available. Skipping...")
            return {}

        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }

        xgb_model = xgb.XGBRegressor(random_state=42, tree_method='hist')
        grid_search = GridSearchCV(xgb_model, param_grid, cv=3,
                                   scoring='neg_mean_squared_error', n_jobs=-1)
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_

        y_pred_train = best_model.predict(X_train)
        y_pred_test = best_model.predict(X_test)

        results = {
            'model': best_model,
            'best_params': grid_search.best_params_,
            'train_r2': r2_score(y_train, y_pred_train),
            'test_r2': r2_score(y_test, y_pred_test),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'train_mae': mean_absolute_error(y_train, y_pred_train),
            'test_mae': mean_absolute_error(y_test, y_pred_test),
            'feature_importance': best_model.feature_importances_
        }

        self.models['XGBoost'] = best_model
        self.results['XGBoost'] = results
        return results

    def train_lightgbm(self, X_train, y_train, X_test, y_test) -> Dict:
        """Train LightGBM Regressor (if available)"""
        if not LIGHTGBM_AVAILABLE:
            print("⚠️  LightGBM not available. Skipping...")
            return {}

        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.05, 0.1],
            'num_leaves': [31, 50, 70],
            'max_depth': [-1, 10, 20],
            'subsample': [0.8, 1.0]
        }

        lgb_model = lgb.LGBMRegressor(random_state=42, verbose=-1)
        grid_search = GridSearchCV(lgb_model, param_grid, cv=3,
                                   scoring='neg_mean_squared_error', n_jobs=-1)
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_

        y_pred_train = best_model.predict(X_train)
        y_pred_test = best_model.predict(X_test)

        results = {
            'model': best_model,
            'best_params': grid_search.best_params_,
            'train_r2': r2_score(y_train, y_pred_train),
            'test_r2': r2_score(y_test, y_pred_test),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'train_mae': mean_absolute_error(y_train, y_pred_train),
            'test_mae': mean_absolute_error(y_test, y_pred_test),
            'feature_importance': best_model.feature_importances_
        }

        self.models['LightGBM'] = best_model
        self.results['LightGBM'] = results
        return results

    def residual_analysis(self, model_name: str, X_test, y_test) -> Dict:
        """Perform residual analysis for a trained model"""
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")

        model = self.models[model_name]
        y_pred = model.predict(X_test)
        residuals = y_test - y_pred

        # Normality test (Shapiro-Wilk)
        _, p_value = stats.shapiro(residuals[:min(5000, len(residuals))])  # Max 5000 samples

        # Homoscedasticity (constant variance) - Breusch-Pagan test approximation
        residuals_squared = residuals ** 2
        correlation = np.corrcoef(y_pred, residuals_squared)[0, 1]

        return {
            'residuals_mean': np.mean(residuals),
            'residuals_std': np.std(residuals),
            'normality_p_value': p_value,
            'residuals_homoscedastic': abs(correlation) < 0.3,  # Rule of thumb
            'residuals': residuals
        }

    def train_all_models(self, X_train, y_train, X_test, y_test) -> Dict:
        """Train all available regression models"""
        total_models = 10
        current = 0

        print(f"🔧 Training {total_models} regression models...")
        print("=" * 60)

        # Linear models
        current += 1
        print(f"  [{current}/{total_models}] Linear Regression...")
        self.train_linear_regression(X_train, y_train, X_test, y_test)

        current += 1
        print(f"  [{current}/{total_models}] Ridge Regression...")
        self.train_ridge_regression(X_train, y_train, X_test, y_test)

        current += 1
        print(f"  [{current}/{total_models}] Lasso Regression...")
        self.train_lasso_regression(X_train, y_train, X_test, y_test)

        current += 1
        print(f"  [{current}/{total_models}] Polynomial Regression...")
        self.train_polynomial_regression(X_train, y_train, X_test, y_test, degree=2)

        # Tree-based models
        current += 1
        print(f"  [{current}/{total_models}] Random Forest...")
        self.train_random_forest(X_train, y_train, X_test, y_test)

        current += 1
        print(f"  [{current}/{total_models}] Gradient Boosting...")
        self.train_gradient_boosting(X_train, y_train, X_test, y_test)

        # Support Vector Regression
        current += 1
        print(f"  [{current}/{total_models}] SVR...")
        self.train_svr(X_train, y_train, X_test, y_test)

        # Advanced gradient boosting (optional)
        if XGBOOST_AVAILABLE:
            current += 1
            print(f"  [{current}/{total_models}] XGBoost...")
            self.train_xgboost(X_train, y_train, X_test, y_test)

        if LIGHTGBM_AVAILABLE:
            current += 1
            print(f"  [{current}/{total_models}] LightGBM...")
            self.train_lightgbm(X_train, y_train, X_test, y_test)

        print("=" * 60)
        print(f"✅ All {len(self.models)} models trained successfully!\n")
        return self.results

    def compare_models(self) -> pd.DataFrame:
        """Compare all trained models"""
        comparison = []

        for name, results in self.results.items():
            comparison.append({
                'Model': name,
                'Train R²': results['train_r2'],
                'Test R²': results['test_r2'],
                'Train RMSE': results['train_rmse'],
                'Test RMSE': results['test_rmse'],
                'Train MAE': results['train_mae'],
                'Test MAE': results['test_mae']
            })

        df = pd.DataFrame(comparison)
        df = df.sort_values('Test R²', ascending=False)

        # Select best model
        self.best_model_name = df.iloc[0]['Model']
        self.best_model = self.models[self.best_model_name]

        return df

    def plot_results(self, X_test, y_test, save_path: Optional[str] = None):
        """Plot predictions vs actual values for all models"""
        n_models = len(self.models)
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()

        for idx, (name, model) in enumerate(self.models.items()):
            if idx >= 4:
                break

            y_pred = model.predict(X_test)

            axes[idx].scatter(y_test, y_pred, alpha=0.6)
            axes[idx].plot([y_test.min(), y_test.max()],
                          [y_test.min(), y_test.max()],
                          'r--', lw=2)
            axes[idx].set_xlabel('Actual Values')
            axes[idx].set_ylabel('Predicted Values')
            axes[idx].set_title(f'{name}\nR² = {self.results[name]["test_r2"]:.4f}')
            axes[idx].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Plot saved to {save_path}")

        plt.show()

    def save_model(self, model_name: str, filepath: str):
        """Save a trained model"""
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")

        joblib.dump(self.models[model_name], filepath)
        print(f"💾 Model saved to {filepath}")

    def load_model(self, filepath: str) -> object:
        """Load a saved model"""
        model = joblib.load(filepath)
        print(f"✅ Model loaded from {filepath}")
        return model


def main():
    parser = argparse.ArgumentParser(description='Regression Analysis')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to CSV data file')
    parser.add_argument('--target', type=str, required=True,
                       help='Target column name')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Test set size')
    parser.add_argument('--output', type=str,
                       help='Output plot path')
    parser.add_argument('--save-model', type=str,
                       help='Path to save best model')

    args = parser.parse_args()

    # Load data
    print(f"📂 Loading data from {args.data}...")
    df = pd.read_csv(args.data)

    # Prepare features and target
    X = df.drop(columns=[args.target])
    y = df[args.target]

    print(f"📊 Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"🎯 Target: {args.target}\n")

    # Initialize analyzer
    analyzer = RegressionAnalyzer(scale_features=True)

    # Prepare data
    X_train, X_test, y_train, y_test = analyzer.prepare_data(
        X, y, test_size=args.test_size
    )

    # Train all models
    analyzer.train_all_models(X_train, y_train, X_test, y_test)

    # Compare models
    print("📊 Model Comparison:")
    print("=" * 80)
    comparison_df = analyzer.compare_models()
    print(comparison_df.to_string(index=False))
    print("=" * 80)

    print(f"\n🏆 Best Model: {analyzer.best_model_name}")
    print(f"   Test R²: {analyzer.results[analyzer.best_model_name]['test_r2']:.4f}")
    print(f"   Test RMSE: {analyzer.results[analyzer.best_model_name]['test_rmse']:.4f}")

    # Plot results
    if args.output:
        analyzer.plot_results(X_test, y_test, save_path=args.output)

    # Save best model
    if args.save_model:
        analyzer.save_model(analyzer.best_model_name, args.save_model)


if __name__ == "__main__":
    main()
