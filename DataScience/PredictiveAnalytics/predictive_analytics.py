"""
Predictive Analytics Toolkit
=============================

End-to-end predictive modeling pipeline:
- Data preparation and splitting
- Model training and evaluation
- Hyperparameter tuning
- Model comparison
- Prediction generation
- Model persistence

Author: Brill Consulting
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                            roc_auc_score, mean_squared_error, r2_score, classification_report)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import joblib
import warnings
warnings.filterwarnings('ignore')


class PredictiveAnalytics:
    """Predictive modeling and analytics toolkit."""

    def __init__(self, task: str = 'classification'):
        """Initialize with task type (classification or regression)."""
        self.task = task
        self.models = {}
        self.results = {}
        self.best_model = None

    def prepare_data(self, X: pd.DataFrame, y: pd.Series,
                    test_size: float = 0.2, random_state: int = 42) -> Tuple:
        """Split data into train and test sets."""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y if self.task == 'classification' else None
        )

        return X_train, X_test, y_train, y_test

    def train_multiple_models(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict:
        """Train multiple models and compare."""
        if self.task == 'classification':
            self.models = {
                'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
                'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
                'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
                'SVM': SVC(probability=True, random_state=42)
            }
        else:
            self.models = {
                'Linear Regression': LinearRegression(),
                'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
            }

        # Train all models
        for name, model in self.models.items():
            model.fit(X_train, y_train)
            print(f"✓ Trained {name}")

        return self.models

    def evaluate_models(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """Evaluate all trained models."""
        self.results = {}

        for name, model in self.models.items():
            y_pred = model.predict(X_test)

            if self.task == 'classification':
                y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

                self.results[name] = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred, average='binary'),
                    'recall': recall_score(y_test, y_pred, average='binary'),
                    'f1': f1_score(y_test, y_pred, average='binary'),
                    'roc_auc': roc_auc_score(y_test, y_proba) if y_proba is not None else None
                }

            else:  # regression
                self.results[name] = {
                    'mse': mean_squared_error(y_test, y_pred),
                    'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                    'r2': r2_score(y_test, y_pred)
                }

        return self.results

    def cross_validate(self, X: pd.DataFrame, y: pd.Series, cv: int = 5) -> Dict:
        """Perform cross-validation for all models."""
        cv_results = {}

        scoring = 'accuracy' if self.task == 'classification' else 'r2'

        for name, model in self.models.items():
            scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
            cv_results[name] = {
                'mean_score': scores.mean(),
                'std_score': scores.std(),
                'scores': scores.tolist()
            }

        return cv_results

    def tune_hyperparameters(self, X_train: pd.DataFrame, y_train: pd.Series,
                            model_name: str = 'Random Forest') -> Dict:
        """Tune hyperparameters using GridSearchCV."""
        if model_name == 'Random Forest' and self.task == 'classification':
            model = RandomForestClassifier(random_state=42)
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10]
            }
        elif model_name == 'Random Forest' and self.task == 'regression':
            model = RandomForestRegressor(random_state=42)
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10]
            }
        else:
            return {}

        scoring = 'accuracy' if self.task == 'classification' else 'r2'
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring=scoring, n_jobs=-1)
        grid_search.fit(X_train, y_train)

        result = {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'best_model': grid_search.best_estimator_
        }

        self.best_model = grid_search.best_estimator_

        return result

    def get_feature_importance(self, model_name: str = 'Random Forest',
                              feature_names: Optional[List[str]] = None) -> pd.DataFrame:
        """Get feature importance from tree-based models."""
        if model_name not in self.models:
            return pd.DataFrame()

        model = self.models[model_name]

        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': feature_names if feature_names else range(len(model.feature_importances_)),
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)

            return importance_df

        return pd.DataFrame()

    def visualize_model_comparison(self) -> plt.Figure:
        """Visualize model performance comparison."""
        if not self.results:
            print("No results to visualize")
            return None

        results_df = pd.DataFrame(self.results).T

        if self.task == 'classification':
            metrics = ['accuracy', 'precision', 'recall', 'f1']
            n_metrics = len(metrics)

            fig, axes = plt.subplots(1, n_metrics, figsize=(16, 4))

            for idx, metric in enumerate(metrics):
                if metric in results_df.columns:
                    results_df[metric].plot(kind='bar', ax=axes[idx], color='steelblue')
                    axes[idx].set_title(f'{metric.capitalize()}')
                    axes[idx].set_ylabel('Score')
                    axes[idx].set_ylim(0, 1)
                    axes[idx].tick_params(axis='x', rotation=45)

        else:  # regression
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))

            results_df['rmse'].plot(kind='bar', ax=axes[0], color='coral')
            axes[0].set_title('RMSE (Lower is Better)')
            axes[0].set_ylabel('RMSE')
            axes[0].tick_params(axis='x', rotation=45)

            results_df['r2'].plot(kind='bar', ax=axes[1], color='green')
            axes[1].set_title('R² Score (Higher is Better)')
            axes[1].set_ylabel('R² Score')
            axes[1].set_ylim(0, 1)
            axes[1].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        return fig

    def generate_predictions(self, X_new: pd.DataFrame, model_name: Optional[str] = None) -> np.ndarray:
        """Generate predictions for new data."""
        if model_name and model_name in self.models:
            model = self.models[model_name]
        elif self.best_model:
            model = self.best_model
        else:
            # Use first available model
            model = list(self.models.values())[0]

        predictions = model.predict(X_new)

        return predictions

    def save_model(self, model_name: str, filepath: str):
        """Save trained model to disk."""
        if model_name in self.models:
            joblib.dump(self.models[model_name], filepath)
            print(f"✓ Saved {model_name} to {filepath}")

    def load_model(self, filepath: str, model_name: str):
        """Load model from disk."""
        model = joblib.load(filepath)
        self.models[model_name] = model
        print(f"✓ Loaded {model_name} from {filepath}")


def demo():
    """Demo predictive analytics."""
    np.random.seed(42)

    print("Predictive Analytics Demo")
    print("="*50)

    # Generate sample classification data
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=1000, n_features=10, n_informative=5,
                               n_redundant=2, random_state=42)

    X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    y_series = pd.Series(y, name='target')

    # Initialize
    pa = PredictiveAnalytics(task='classification')

    # Prepare data
    print("\n1. Preparing data...")
    X_train, X_test, y_train, y_test = pa.prepare_data(X_df, y_series)
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    # Train models
    print("\n2. Training multiple models...")
    pa.train_multiple_models(X_train, y_train)

    # Evaluate models
    print("\n3. Evaluating models...")
    results = pa.evaluate_models(X_test, y_test)
    for model, metrics in results.items():
        print(f"\n{model}:")
        for metric, value in metrics.items():
            if value is not None:
                print(f"  {metric}: {value:.4f}")

    # Cross-validation
    print("\n4. Cross-validation...")
    cv_results = pa.cross_validate(X_train, y_train, cv=5)
    for model, scores in cv_results.items():
        print(f"{model}: {scores['mean_score']:.4f} (+/- {scores['std_score']:.4f})")

    # Hyperparameter tuning
    print("\n5. Tuning hyperparameters...")
    tuning_result = pa.tune_hyperparameters(X_train, y_train, 'Random Forest')
    print(f"Best params: {tuning_result['best_params']}")
    print(f"Best CV score: {tuning_result['best_score']:.4f}")

    # Feature importance
    print("\n6. Feature importance...")
    importance = pa.get_feature_importance('Random Forest', feature_names=X_df.columns.tolist())
    print(importance.head(10))

    # Visualization
    print("\n7. Generating comparison plot...")
    fig = pa.visualize_model_comparison()
    fig.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved model_comparison.png")
    plt.close()

    # Predictions
    print("\n8. Generating predictions...")
    predictions = pa.generate_predictions(X_test.head(10))
    print(f"First 10 predictions: {predictions}")

    # Save model
    print("\n9. Saving best model...")
    pa.save_model('Random Forest', 'random_forest_model.joblib')

    print("\n✓ Predictive Analytics Complete!")


if __name__ == '__main__':
    demo()
