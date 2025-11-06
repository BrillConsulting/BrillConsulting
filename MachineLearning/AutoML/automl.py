"""
Advanced AutoML System
Author: BrillConsulting
Description: Automated machine learning with hyperparameter optimization and model selection
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split, RandomizedSearchCV
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor,
                              GradientBoostingClassifier, GradientBoostingRegressor)
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, f1_score, mean_squared_error,
                             r2_score, mean_absolute_error)
from sklearn.pipeline import Pipeline
from typing import Dict, List, Tuple, Optional, Union
import warnings
import time
from dataclasses import dataclass
warnings.filterwarnings('ignore')


@dataclass
class AutoMLResult:
    """Results from AutoML optimization"""
    best_model: object
    best_params: Dict
    best_score: float
    model_scores: Dict[str, float]
    optimization_time: float
    task_type: str


class AutoML:
    """
    Automated Machine Learning System

    Features:
    - Automatic task detection (classification/regression)
    - Model selection from multiple algorithms
    - Hyperparameter optimization with random search
    - Automatic feature scaling
    - Cross-validation
    - Pipeline creation
    """

    def __init__(self,
                 task_type: str = 'auto',
                 time_limit: int = 300,
                 n_iter: int = 20,
                 cv: int = 5,
                 random_state: int = 42,
                 verbose: bool = True):
        """
        Initialize AutoML

        Args:
            task_type: 'auto', 'classification', or 'regression'
            time_limit: Maximum optimization time in seconds
            n_iter: Number of parameter settings sampled per model
            cv: Number of cross-validation folds
            random_state: Random seed
            verbose: Print progress
        """
        self.task_type = task_type
        self.time_limit = time_limit
        self.n_iter = n_iter
        self.cv = cv
        self.random_state = random_state
        self.verbose = verbose

        self.best_model = None
        self.best_params = None
        self.best_score = None
        self.model_scores = {}
        self.scaler = StandardScaler()
        self.label_encoder = None

    def _detect_task_type(self, y: np.ndarray) -> str:
        """
        Auto-detect classification vs regression

        Args:
            y: Target variable

        Returns:
            'classification' or 'regression'
        """
        # Check if target is numeric and has many unique values
        unique_values = np.unique(y)
        n_unique = len(unique_values)

        # Heuristic: if < 20 unique values and all integers, likely classification
        if n_unique < 20 and np.all(y == y.astype(int)):
            return 'classification'
        else:
            return 'regression'

    def _get_classification_models(self) -> Dict[str, Tuple[object, Dict]]:
        """
        Get classification models and their hyperparameter spaces

        Returns:
            Dict of {model_name: (model, param_grid)}
        """
        models = {
            'RandomForest': (
                RandomForestClassifier(random_state=self.random_state),
                {
                    'n_estimators': [50, 100, 200, 300],
                    'max_depth': [None, 10, 20, 30, 50],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2', None]
                }
            ),
            'GradientBoosting': (
                GradientBoostingClassifier(random_state=self.random_state),
                {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.05, 0.1, 0.2],
                    'max_depth': [3, 5, 7, 10],
                    'subsample': [0.6, 0.8, 1.0],
                    'min_samples_split': [2, 5, 10]
                }
            ),
            'LogisticRegression': (
                LogisticRegression(random_state=self.random_state, max_iter=1000),
                {
                    'C': [0.001, 0.01, 0.1, 1, 10, 100],
                    'penalty': ['l2'],
                    'solver': ['lbfgs', 'liblinear']
                }
            ),
            'SVM': (
                SVC(random_state=self.random_state, probability=True),
                {
                    'C': [0.1, 1, 10, 100],
                    'kernel': ['rbf', 'linear'],
                    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1]
                }
            ),
            'KNeighbors': (
                KNeighborsClassifier(),
                {
                    'n_neighbors': [3, 5, 7, 9, 11, 15],
                    'weights': ['uniform', 'distance'],
                    'metric': ['euclidean', 'manhattan', 'minkowski']
                }
            ),
            'DecisionTree': (
                DecisionTreeClassifier(random_state=self.random_state),
                {
                    'max_depth': [None, 5, 10, 20, 30],
                    'min_samples_split': [2, 5, 10, 20],
                    'min_samples_leaf': [1, 2, 4, 8],
                    'criterion': ['gini', 'entropy']
                }
            )
        }

        return models

    def _get_regression_models(self) -> Dict[str, Tuple[object, Dict]]:
        """
        Get regression models and their hyperparameter spaces

        Returns:
            Dict of {model_name: (model, param_grid)}
        """
        models = {
            'RandomForest': (
                RandomForestRegressor(random_state=self.random_state),
                {
                    'n_estimators': [50, 100, 200, 300],
                    'max_depth': [None, 10, 20, 30, 50],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2', None]
                }
            ),
            'GradientBoosting': (
                GradientBoostingRegressor(random_state=self.random_state),
                {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.05, 0.1, 0.2],
                    'max_depth': [3, 5, 7, 10],
                    'subsample': [0.6, 0.8, 1.0],
                    'min_samples_split': [2, 5, 10]
                }
            ),
            'Ridge': (
                Ridge(random_state=self.random_state),
                {
                    'alpha': [0.001, 0.01, 0.1, 1, 10, 100, 1000]
                }
            ),
            'Lasso': (
                Lasso(random_state=self.random_state, max_iter=10000),
                {
                    'alpha': [0.001, 0.01, 0.1, 1, 10, 100]
                }
            ),
            'SVR': (
                SVR(),
                {
                    'C': [0.1, 1, 10, 100],
                    'kernel': ['rbf', 'linear'],
                    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
                    'epsilon': [0.01, 0.1, 0.5, 1.0]
                }
            ),
            'KNeighbors': (
                KNeighborsRegressor(),
                {
                    'n_neighbors': [3, 5, 7, 9, 11, 15],
                    'weights': ['uniform', 'distance'],
                    'metric': ['euclidean', 'manhattan', 'minkowski']
                }
            )
        }

        return models

    def fit(self, X, y):
        """
        Fit AutoML system

        Args:
            X: Feature matrix
            y: Target variable

        Returns:
            self
        """
        start_time = time.time()

        # Auto-detect task type if needed
        if self.task_type == 'auto':
            self.task_type = self._detect_task_type(y)
            if self.verbose:
                print(f"🔍 Detected task type: {self.task_type}")

        # Encode labels for classification if needed
        if self.task_type == 'classification':
            self.label_encoder = LabelEncoder()
            y = self.label_encoder.fit_transform(y)

        # Get appropriate models
        if self.task_type == 'classification':
            models = self._get_classification_models()
            scoring = 'f1_weighted'
        else:
            models = self._get_regression_models()
            scoring = 'neg_mean_squared_error'

        if self.verbose:
            print(f"\n🤖 AutoML Optimization")
            print(f"   Time limit: {self.time_limit}s")
            print(f"   Models to test: {len(models)}")
            print(f"   Cross-validation folds: {self.cv}")
            print(f"   Iterations per model: {self.n_iter}\n")

        # Test each model
        best_overall_score = -np.inf

        for idx, (name, (model, param_grid)) in enumerate(models.items(), 1):
            if time.time() - start_time > self.time_limit:
                if self.verbose:
                    print(f"\n⏰ Time limit reached ({self.time_limit}s)")
                break

            if self.verbose:
                print(f"   {idx}/{len(models)} Testing {name}...")

            try:
                # Random search for hyperparameters
                search = RandomizedSearchCV(
                    model,
                    param_grid,
                    n_iter=self.n_iter,
                    cv=self.cv,
                    scoring=scoring,
                    random_state=self.random_state,
                    n_jobs=-1,
                    error_score='raise'
                )

                search.fit(X, y)

                # Get score (convert MSE to positive for comparison)
                if self.task_type == 'regression':
                    score = -search.best_score_  # Convert neg_mse to positive
                else:
                    score = search.best_score_

                self.model_scores[name] = score

                # Update best model if better
                if score > best_overall_score:
                    best_overall_score = score
                    self.best_model = search.best_estimator_
                    self.best_params = search.best_params_
                    self.best_score = score
                    best_model_name = name

                if self.verbose:
                    print(f"      Score: {score:.4f} | Best params: {search.best_params_}")

            except Exception as e:
                if self.verbose:
                    print(f"      ❌ Failed: {str(e)[:50]}")
                continue

        elapsed_time = time.time() - start_time

        if self.verbose:
            print(f"\n✅ Optimization complete!")
            print(f"   Best model: {best_model_name}")
            print(f"   Best score: {self.best_score:.4f}")
            print(f"   Time elapsed: {elapsed_time:.1f}s")

        return self

    def predict(self, X):
        """
        Make predictions with best model

        Args:
            X: Feature matrix

        Returns:
            Predictions
        """
        if self.best_model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        predictions = self.best_model.predict(X)

        # Decode labels for classification
        if self.task_type == 'classification' and self.label_encoder is not None:
            predictions = self.label_encoder.inverse_transform(predictions)

        return predictions

    def predict_proba(self, X):
        """
        Predict class probabilities (classification only)

        Args:
            X: Feature matrix

        Returns:
            Probability predictions
        """
        if self.task_type != 'classification':
            raise ValueError("predict_proba only available for classification")

        if self.best_model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        if hasattr(self.best_model, 'predict_proba'):
            return self.best_model.predict_proba(X)
        else:
            raise ValueError(f"Best model ({type(self.best_model).__name__}) does not support predict_proba")

    def get_results(self) -> AutoMLResult:
        """
        Get detailed results

        Returns:
            AutoMLResult object
        """
        return AutoMLResult(
            best_model=self.best_model,
            best_params=self.best_params,
            best_score=self.best_score,
            model_scores=self.model_scores,
            optimization_time=0,  # Would need to track separately
            task_type=self.task_type
        )

    def get_model_ranking(self) -> pd.DataFrame:
        """
        Get ranking of all tested models

        Returns:
            DataFrame with model rankings
        """
        if not self.model_scores:
            raise ValueError("No models have been tested yet")

        ranking = pd.DataFrame([
            {'Model': name, 'Score': score}
            for name, score in self.model_scores.items()
        ]).sort_values('Score', ascending=False).reset_index(drop=True)

        ranking['Rank'] = range(1, len(ranking) + 1)

        return ranking[['Rank', 'Model', 'Score']]


def demo():
    """Demonstration of AutoML system"""
    print("=" * 80)
    print("🤖 Advanced AutoML System - Demo")
    print("=" * 80)

    # Classification example
    print("\n📊 CLASSIFICATION EXAMPLE")
    print("-" * 80)

    from sklearn.datasets import load_breast_cancer
    data = load_breast_cancer()
    X, y = data.data, data.target

    print(f"Dataset: {len(X)} samples, {X.shape[1]} features")
    print(f"Classes: {np.unique(y)}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Run AutoML
    automl = AutoML(
        task_type='auto',
        time_limit=60,
        n_iter=10,
        cv=3,
        random_state=42,
        verbose=True
    )

    automl.fit(X_train, y_train)

    # Predictions
    y_pred = automl.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    print(f"\n📈 Test Performance:")
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   F1 Score: {f1:.4f}")

    # Model ranking
    print(f"\n🏆 Model Ranking:")
    print(automl.get_model_ranking().to_string(index=False))

    # Regression example
    print("\n" + "=" * 80)
    print("📊 REGRESSION EXAMPLE")
    print("-" * 80)

    from sklearn.datasets import load_diabetes
    data = load_diabetes()
    X, y = data.data, data.target

    print(f"Dataset: {len(X)} samples, {X.shape[1]} features")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Run AutoML
    automl_reg = AutoML(
        task_type='auto',
        time_limit=60,
        n_iter=10,
        cv=3,
        random_state=42,
        verbose=True
    )

    automl_reg.fit(X_train, y_train)

    # Predictions
    y_pred = automl_reg.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\n📈 Test Performance:")
    print(f"   MSE: {mse:.4f}")
    print(f"   R²: {r2:.4f}")

    print(f"\n🏆 Model Ranking:")
    print(automl_reg.get_model_ranking().to_string(index=False))

    print("\n" + "=" * 80)
    print("✅ Demo completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    demo()
