"""
Model Interpretability Toolkit

A comprehensive toolkit for interpreting machine learning models with various
explanation methods including SHAP-like analysis, LIME-style local explanations,
PDP, ICE plots, and permutation importance.

Author: Brill Consulting
Date: 2025-11-06
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Callable, Optional, Union, List, Tuple, Dict, Any
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
from dataclasses import dataclass

warnings.filterwarnings('ignore')


@dataclass
class ExplanationResult:
    """Container for explanation results."""
    feature_names: List[str]
    values: np.ndarray
    base_value: float
    prediction: float


class ModelInterpreter:
    """
    Comprehensive model interpretability toolkit.

    Provides various methods for explaining machine learning model predictions
    including global and local interpretability techniques.

    Attributes:
        model: The trained model to interpret
        X_train: Training data used for explanations
        feature_names: Names of features
        n_features: Number of features

    Example:
        >>> interpreter = ModelInterpreter(model, X_train, feature_names)
        >>> shap_values = interpreter.calculate_shap_values(X_test)
        >>> interpreter.plot_feature_importance()
    """

    def __init__(
        self,
        model: BaseEstimator,
        X_train: np.ndarray,
        feature_names: Optional[List[str]] = None
    ):
        """
        Initialize the ModelInterpreter.

        Args:
            model: Trained sklearn-compatible model
            X_train: Training data (numpy array or pandas DataFrame)
            feature_names: Optional list of feature names
        """
        self.model = model

        if isinstance(X_train, pd.DataFrame):
            self.feature_names = X_train.columns.tolist()
            self.X_train = X_train.values
        else:
            self.X_train = np.array(X_train)
            self.feature_names = feature_names or [f'Feature_{i}' for i in range(X_train.shape[1])]

        self.n_features = self.X_train.shape[1]
        self._feature_importance_cache = None

    def calculate_shap_values(
        self,
        X: np.ndarray,
        n_samples: int = 100,
        method: str = 'kernel'
    ) -> np.ndarray:
        """
        Calculate SHAP-like values using kernel SHAP approximation.

        This is a simplified implementation that approximates SHAP values
        using a kernel-based approach similar to the original SHAP paper.

        Args:
            X: Data to explain (n_samples, n_features)
            n_samples: Number of coalition samples for approximation
            method: Method to use ('kernel' or 'sampling')

        Returns:
            SHAP values array of shape (n_samples, n_features)
        """
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        shap_values = np.zeros((X.shape[0], self.n_features))

        # Get base prediction (average over training data)
        base_predictions = self.model.predict(self.X_train)
        if hasattr(self.model, 'predict_proba'):
            base_value = np.mean(self.model.predict_proba(self.X_train)[:, 1])
        else:
            base_value = np.mean(base_predictions)

        for i, instance in enumerate(X):
            shap_values[i] = self._explain_instance_shap(
                instance, base_value, n_samples
            )

        return shap_values

    def _explain_instance_shap(
        self,
        instance: np.ndarray,
        base_value: float,
        n_samples: int
    ) -> np.ndarray:
        """
        Explain a single instance using SHAP-like approximation.

        Args:
            instance: Single instance to explain
            base_value: Base prediction value
            n_samples: Number of samples for approximation

        Returns:
            SHAP values for the instance
        """
        # Use coalitional sampling
        shap_vals = np.zeros(self.n_features)

        for feature_idx in range(self.n_features):
            # Sample coalitions with and without this feature
            marginal_contributions = []

            for _ in range(n_samples // self.n_features):
                # Random coalition
                coalition = np.random.rand(self.n_features) > 0.5

                # Prediction with feature
                x_with = instance.copy()
                for j, include in enumerate(coalition):
                    if not include and j != feature_idx:
                        # Replace with random training sample value
                        x_with[j] = self.X_train[np.random.randint(len(self.X_train)), j]

                # Prediction without feature
                x_without = x_with.copy()
                x_without[feature_idx] = self.X_train[np.random.randint(len(self.X_train)), feature_idx]

                # Get predictions
                if hasattr(self.model, 'predict_proba'):
                    pred_with = self.model.predict_proba(x_with.reshape(1, -1))[0, 1]
                    pred_without = self.model.predict_proba(x_without.reshape(1, -1))[0, 1]
                else:
                    pred_with = self.model.predict(x_with.reshape(1, -1))[0]
                    pred_without = self.model.predict(x_without.reshape(1, -1))[0]

                marginal_contributions.append(pred_with - pred_without)

            shap_vals[feature_idx] = np.mean(marginal_contributions)

        return shap_vals

    def explain_local_lime(
        self,
        instance: np.ndarray,
        n_samples: int = 5000,
        n_features: int = 10
    ) -> ExplanationResult:
        """
        Create LIME-style local explanation for a single instance.

        Args:
            instance: Instance to explain
            n_samples: Number of perturbed samples to generate
            n_features: Number of top features to include in explanation

        Returns:
            ExplanationResult containing feature contributions
        """
        instance = np.array(instance).flatten()

        # Generate perturbed samples around the instance
        perturbations = np.random.normal(
            loc=instance,
            scale=np.std(self.X_train, axis=0) * 0.25,
            size=(n_samples, self.n_features)
        )

        # Get predictions for perturbations
        if hasattr(self.model, 'predict_proba'):
            predictions = self.model.predict_proba(perturbations)[:, 1]
            base_pred = self.model.predict_proba(instance.reshape(1, -1))[0, 1]
        else:
            predictions = self.model.predict(perturbations)
            base_pred = self.model.predict(instance.reshape(1, -1))[0]

        # Calculate distances from instance
        distances = np.sum((perturbations - instance) ** 2, axis=1)
        weights = np.exp(-distances / (2 * np.std(distances) ** 2))

        # Fit linear model
        from sklearn.linear_model import Ridge
        linear_model = Ridge(alpha=1.0)
        linear_model.fit(perturbations, predictions, sample_weight=weights)

        # Get feature importance
        coefficients = linear_model.coef_

        # Select top features
        top_indices = np.argsort(np.abs(coefficients))[-n_features:][::-1]

        return ExplanationResult(
            feature_names=[self.feature_names[i] for i in top_indices],
            values=coefficients[top_indices],
            base_value=linear_model.intercept_,
            prediction=base_pred
        )

    def calculate_pdp(
        self,
        feature_idx: int,
        n_points: int = 50,
        percentile_range: Tuple[float, float] = (0.05, 0.95)
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate Partial Dependence Plot values for a feature.

        Args:
            feature_idx: Index of feature to analyze
            n_points: Number of points to evaluate
            percentile_range: Range of feature values to consider

        Returns:
            Tuple of (grid_values, pdp_values)
        """
        # Get feature range
        feature_values = self.X_train[:, feature_idx]
        lower = np.percentile(feature_values, percentile_range[0] * 100)
        upper = np.percentile(feature_values, percentile_range[1] * 100)
        grid_values = np.linspace(lower, upper, n_points)

        pdp_values = np.zeros(n_points)

        for i, value in enumerate(grid_values):
            # Create copies of training data with feature set to value
            X_modified = self.X_train.copy()
            X_modified[:, feature_idx] = value

            # Get predictions
            if hasattr(self.model, 'predict_proba'):
                predictions = self.model.predict_proba(X_modified)[:, 1]
            else:
                predictions = self.model.predict(X_modified)

            pdp_values[i] = np.mean(predictions)

        return grid_values, pdp_values

    def calculate_ice(
        self,
        feature_idx: int,
        X: Optional[np.ndarray] = None,
        n_samples: int = 50,
        n_points: int = 50,
        percentile_range: Tuple[float, float] = (0.05, 0.95)
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate Individual Conditional Expectation curves.

        Args:
            feature_idx: Index of feature to analyze
            X: Data samples to use (uses training data if None)
            n_samples: Number of samples to plot
            n_points: Number of points per curve
            percentile_range: Range of feature values to consider

        Returns:
            Tuple of (grid_values, ice_curves)
        """
        if X is None:
            X = self.X_train

        X = np.array(X)
        if len(X) > n_samples:
            indices = np.random.choice(len(X), n_samples, replace=False)
            X = X[indices]

        # Get feature range
        feature_values = self.X_train[:, feature_idx]
        lower = np.percentile(feature_values, percentile_range[0] * 100)
        upper = np.percentile(feature_values, percentile_range[1] * 100)
        grid_values = np.linspace(lower, upper, n_points)

        ice_curves = np.zeros((len(X), n_points))

        for i, instance in enumerate(X):
            for j, value in enumerate(grid_values):
                # Modify feature value
                instance_modified = instance.copy()
                instance_modified[feature_idx] = value

                # Get prediction
                if hasattr(self.model, 'predict_proba'):
                    pred = self.model.predict_proba(instance_modified.reshape(1, -1))[0, 1]
                else:
                    pred = self.model.predict(instance_modified.reshape(1, -1))[0]

                ice_curves[i, j] = pred

        return grid_values, ice_curves

    def permutation_importance(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_repeats: int = 10,
        random_state: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        """
        Calculate permutation feature importance.

        Args:
            X: Test data
            y: True labels
            n_repeats: Number of times to permute each feature
            random_state: Random seed for reproducibility

        Returns:
            Dictionary with 'importances_mean' and 'importances_std'
        """
        if random_state is not None:
            np.random.seed(random_state)

        X = np.array(X)
        y = np.array(y)

        # Baseline score
        if hasattr(self.model, 'predict_proba'):
            baseline_preds = self.model.predict_proba(X)[:, 1]
            baseline_score = -np.mean((baseline_preds - y) ** 2)
        else:
            baseline_preds = self.model.predict(X)
            baseline_score = accuracy_score(y, baseline_preds)

        importances = np.zeros((self.n_features, n_repeats))

        for feature_idx in range(self.n_features):
            for repeat in range(n_repeats):
                # Permute feature
                X_permuted = X.copy()
                X_permuted[:, feature_idx] = np.random.permutation(X_permuted[:, feature_idx])

                # Get new score
                if hasattr(self.model, 'predict_proba'):
                    preds = self.model.predict_proba(X_permuted)[:, 1]
                    score = -np.mean((preds - y) ** 2)
                else:
                    preds = self.model.predict(X_permuted)
                    score = accuracy_score(y, preds)

                # Importance is decrease in score
                importances[feature_idx, repeat] = baseline_score - score

        return {
            'importances_mean': np.mean(importances, axis=1),
            'importances_std': np.std(importances, axis=1),
            'importances': importances
        }

    def plot_feature_importance(
        self,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        method: str = 'permutation',
        top_n: int = 10,
        figsize: Tuple[int, int] = (10, 6)
    ) -> plt.Figure:
        """
        Plot feature importance.

        Args:
            X: Test data (required for permutation method)
            y: True labels (required for permutation method)
            method: Method to use ('permutation', 'model', or 'shap')
            top_n: Number of top features to show
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)

        if method == 'permutation':
            if X is None or y is None:
                raise ValueError("X and y required for permutation importance")
            result = self.permutation_importance(X, y)
            importances = result['importances_mean']
            std = result['importances_std']

        elif method == 'model':
            if hasattr(self.model, 'feature_importances_'):
                importances = self.model.feature_importances_
                std = None
            else:
                raise ValueError("Model does not have feature_importances_ attribute")

        elif method == 'shap':
            if X is None:
                raise ValueError("X required for SHAP importance")
            shap_values = self.calculate_shap_values(X)
            importances = np.mean(np.abs(shap_values), axis=0)
            std = np.std(np.abs(shap_values), axis=0)
        else:
            raise ValueError(f"Unknown method: {method}")

        # Sort and select top features
        indices = np.argsort(importances)[-top_n:]

        # Plot
        y_pos = np.arange(len(indices))
        if std is not None:
            ax.barh(y_pos, importances[indices], xerr=std[indices], alpha=0.7)
        else:
            ax.barh(y_pos, importances[indices], alpha=0.7)

        ax.set_yticks(y_pos)
        ax.set_yticklabels([self.feature_names[i] for i in indices])
        ax.set_xlabel('Importance')
        ax.set_title(f'Feature Importance ({method})')
        ax.grid(axis='x', alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_pdp(
        self,
        feature_idx: int,
        n_points: int = 50,
        figsize: Tuple[int, int] = (10, 6)
    ) -> plt.Figure:
        """
        Plot Partial Dependence Plot for a feature.

        Args:
            feature_idx: Index of feature to plot
            n_points: Number of points to evaluate
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        grid_values, pdp_values = self.calculate_pdp(feature_idx, n_points)

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(grid_values, pdp_values, linewidth=2, color='blue')
        ax.set_xlabel(self.feature_names[feature_idx])
        ax.set_ylabel('Partial Dependence')
        ax.set_title(f'Partial Dependence Plot: {self.feature_names[feature_idx]}')
        ax.grid(alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_ice(
        self,
        feature_idx: int,
        X: Optional[np.ndarray] = None,
        n_samples: int = 50,
        n_points: int = 50,
        show_pdp: bool = True,
        figsize: Tuple[int, int] = (10, 6)
    ) -> plt.Figure:
        """
        Plot Individual Conditional Expectation curves.

        Args:
            feature_idx: Index of feature to plot
            X: Data samples to use
            n_samples: Number of samples to plot
            n_points: Number of points per curve
            show_pdp: Whether to overlay PDP
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        grid_values, ice_curves = self.calculate_ice(
            feature_idx, X, n_samples, n_points
        )

        fig, ax = plt.subplots(figsize=figsize)

        # Plot ICE curves
        for curve in ice_curves:
            ax.plot(grid_values, curve, alpha=0.3, color='gray', linewidth=1)

        # Overlay PDP if requested
        if show_pdp:
            pdp_values = np.mean(ice_curves, axis=0)
            ax.plot(grid_values, pdp_values, linewidth=3, color='blue', label='PDP (mean)')
            ax.legend()

        ax.set_xlabel(self.feature_names[feature_idx])
        ax.set_ylabel('Prediction')
        ax.set_title(f'ICE Plot: {self.feature_names[feature_idx]}')
        ax.grid(alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_shap_summary(
        self,
        X: np.ndarray,
        n_samples: int = 100,
        max_display: int = 10,
        figsize: Tuple[int, int] = (10, 8)
    ) -> plt.Figure:
        """
        Create SHAP summary plot showing feature importance and effects.

        Args:
            X: Data to explain
            n_samples: Number of samples for SHAP calculation
            max_display: Maximum number of features to display
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        X = np.array(X)
        shap_values = self.calculate_shap_values(X, n_samples)

        # Calculate mean absolute SHAP values
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
        feature_order = np.argsort(mean_abs_shap)[-max_display:]

        fig, ax = plt.subplots(figsize=figsize)

        # Create summary plot
        for i, feature_idx in enumerate(feature_order):
            # Color points by feature value
            feature_values = X[:, feature_idx]
            normalized_values = (feature_values - feature_values.min()) / (feature_values.max() - feature_values.min() + 1e-10)

            y_positions = np.ones(len(shap_values)) * i
            y_positions += np.random.normal(0, 0.1, len(shap_values))

            scatter = ax.scatter(
                shap_values[:, feature_idx],
                y_positions,
                c=normalized_values,
                cmap='coolwarm',
                alpha=0.6,
                s=20
            )

        ax.set_yticks(range(len(feature_order)))
        ax.set_yticklabels([self.feature_names[i] for i in feature_order])
        ax.set_xlabel('SHAP value (impact on model output)')
        ax.set_title('SHAP Summary Plot')
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        ax.grid(axis='x', alpha=0.3)

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Feature value', rotation=270, labelpad=20)

        plt.tight_layout()
        return fig

    def plot_force_plot(
        self,
        instance: np.ndarray,
        n_samples: int = 100,
        max_display: int = 10,
        figsize: Tuple[int, int] = (12, 3)
    ) -> plt.Figure:
        """
        Create a force plot showing feature contributions for a single prediction.

        Args:
            instance: Single instance to explain
            n_samples: Number of samples for SHAP calculation
            max_display: Maximum number of features to display
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        instance = np.array(instance).flatten()
        shap_values = self.calculate_shap_values(instance.reshape(1, -1), n_samples)[0]

        # Get base value
        if hasattr(self.model, 'predict_proba'):
            base_value = np.mean(self.model.predict_proba(self.X_train)[:, 1])
            prediction = self.model.predict_proba(instance.reshape(1, -1))[0, 1]
        else:
            base_value = np.mean(self.model.predict(self.X_train))
            prediction = self.model.predict(instance.reshape(1, -1))[0]

        # Sort by absolute SHAP value
        sorted_indices = np.argsort(np.abs(shap_values))[::-1][:max_display]

        fig, ax = plt.subplots(figsize=figsize)

        # Calculate cumulative sum
        cumsum = base_value
        positions = [cumsum]

        for idx in sorted_indices:
            cumsum += shap_values[idx]
            positions.append(cumsum)

        # Plot
        colors = ['red' if sv < 0 else 'blue' for sv in shap_values[sorted_indices]]

        for i, idx in enumerate(sorted_indices):
            start = positions[i]
            end = positions[i + 1]
            width = abs(end - start)

            ax.barh(0, width, left=min(start, end), height=0.5,
                   color=colors[i], alpha=0.7,
                   label=f'{self.feature_names[idx]}={instance[idx]:.2f}')

        ax.axvline(x=base_value, color='gray', linestyle='--', linewidth=2,
                  label=f'Base value: {base_value:.3f}')
        ax.axvline(x=prediction, color='black', linewidth=3,
                  label=f'Prediction: {prediction:.3f}')

        ax.set_yticks([])
        ax.set_xlabel('Model output value')
        ax.set_title('Force Plot: Feature Contributions')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax.grid(axis='x', alpha=0.3)

        plt.tight_layout()
        return fig

    def create_explanation_report(
        self,
        instance: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        output_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a comprehensive explanation report for an instance.

        Args:
            instance: Instance to explain
            X_test: Test data for global analysis
            y_test: Test labels
            output_file: Optional file path to save plots

        Returns:
            Dictionary containing all explanation results
        """
        instance = np.array(instance).flatten()

        # Local explanations
        lime_result = self.explain_local_lime(instance)
        shap_values = self.calculate_shap_values(instance.reshape(1, -1))[0]

        # Global explanations
        perm_importance = self.permutation_importance(X_test, y_test)

        # Prediction
        if hasattr(self.model, 'predict_proba'):
            prediction = self.model.predict_proba(instance.reshape(1, -1))[0, 1]
        else:
            prediction = self.model.predict(instance.reshape(1, -1))[0]

        report = {
            'prediction': prediction,
            'lime_explanation': {
                'top_features': lime_result.feature_names,
                'contributions': lime_result.values.tolist(),
                'base_value': lime_result.base_value
            },
            'shap_values': {
                'feature_names': self.feature_names,
                'values': shap_values.tolist()
            },
            'permutation_importance': {
                'feature_names': self.feature_names,
                'importances': perm_importance['importances_mean'].tolist()
            },
            'instance_values': instance.tolist()
        }

        if output_file:
            # Create visualizations
            fig = plt.figure(figsize=(16, 12))

            # 1. Force plot
            plt.subplot(3, 2, 1)
            self.plot_force_plot(instance)

            # 2. SHAP summary
            plt.subplot(3, 2, 2)
            self.plot_shap_summary(X_test[:100])

            # 3-4. PDP for top 2 features
            top_features = np.argsort(perm_importance['importances_mean'])[-2:]
            for i, feat_idx in enumerate(top_features):
                plt.subplot(3, 2, 3 + i)
                self.plot_pdp(feat_idx)

            # 5-6. ICE for top 2 features
            for i, feat_idx in enumerate(top_features):
                plt.subplot(3, 2, 5 + i)
                self.plot_ice(feat_idx, X_test[:50])

            plt.tight_layout()
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()

        return report


def demo():
    """
    Demonstrate the ModelInterpreter capabilities with comprehensive examples.
    """
    print("=" * 80)
    print("MODEL INTERPRETABILITY TOOLKIT DEMO")
    print("=" * 80)

    # Generate synthetic dataset
    print("\n1. Generating synthetic dataset...")
    np.random.seed(42)
    n_samples = 1000
    n_features = 8

    # Create features with different importance levels
    X = np.random.randn(n_samples, n_features)
    feature_names = [f'Feature_{i}' for i in range(n_features)]

    # Create target with known relationships
    y = (
        2.0 * X[:, 0] +          # Strong positive
        -1.5 * X[:, 1] +         # Strong negative
        0.5 * X[:, 2] * X[:, 3] + # Interaction
        0.3 * X[:, 4] ** 2 +     # Non-linear
        np.random.randn(n_samples) * 0.5
    )
    y = (y > np.median(y)).astype(int)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    print(f"   Features: {n_features}")

    # Train model
    print("\n2. Training Random Forest model...")
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    print(f"   Training accuracy: {train_acc:.4f}")
    print(f"   Test accuracy: {test_acc:.4f}")

    # Initialize interpreter
    print("\n3. Initializing ModelInterpreter...")
    interpreter = ModelInterpreter(model, X_train, feature_names)

    # SHAP values
    print("\n4. Calculating SHAP values...")
    shap_values = interpreter.calculate_shap_values(X_test[:10], n_samples=50)
    print(f"   SHAP values shape: {shap_values.shape}")
    print(f"   Mean absolute SHAP values:")
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    for i, name in enumerate(feature_names):
        print(f"      {name}: {mean_abs_shap[i]:.4f}")

    # LIME explanation
    print("\n5. Creating LIME explanation for first test instance...")
    test_instance = X_test[0]
    lime_result = interpreter.explain_local_lime(test_instance, n_samples=1000)
    print(f"   Prediction: {lime_result.prediction:.4f}")
    print(f"   Base value: {lime_result.base_value:.4f}")
    print(f"   Top contributing features:")
    for name, value in zip(lime_result.feature_names[:5], lime_result.values[:5]):
        print(f"      {name}: {value:+.4f}")

    # Permutation importance
    print("\n6. Calculating permutation importance...")
    perm_importance = interpreter.permutation_importance(X_test, y_test, n_repeats=10)
    print(f"   Feature importances:")
    sorted_idx = np.argsort(perm_importance['importances_mean'])[::-1]
    for idx in sorted_idx[:5]:
        mean = perm_importance['importances_mean'][idx]
        std = perm_importance['importances_std'][idx]
        print(f"      {feature_names[idx]}: {mean:.4f} (+/- {std:.4f})")

    # Partial Dependence
    print("\n7. Calculating Partial Dependence for top feature...")
    top_feature_idx = sorted_idx[0]
    grid_values, pdp_values = interpreter.calculate_pdp(top_feature_idx, n_points=30)
    print(f"   Feature: {feature_names[top_feature_idx]}")
    print(f"   PDP range: [{pdp_values.min():.4f}, {pdp_values.max():.4f}]")
    print(f"   PDP mean: {pdp_values.mean():.4f}")

    # ICE curves
    print("\n8. Calculating ICE curves...")
    grid_values, ice_curves = interpreter.calculate_ice(
        top_feature_idx, X_test[:20], n_samples=20, n_points=30
    )
    print(f"   ICE curves shape: {ice_curves.shape}")
    print(f"   Individual curve variance: {np.var(ice_curves, axis=1).mean():.4f}")

    # Visualizations
    print("\n9. Creating visualizations...")

    # Feature importance comparison
    fig1 = interpreter.plot_feature_importance(
        X_test, y_test, method='permutation', top_n=8
    )
    plt.savefig('/tmp/feature_importance.png', dpi=150, bbox_inches='tight')
    print("   Saved: /tmp/feature_importance.png")
    plt.close()

    # PDP plot
    fig2 = interpreter.plot_pdp(top_feature_idx, n_points=50)
    plt.savefig('/tmp/pdp_plot.png', dpi=150, bbox_inches='tight')
    print("   Saved: /tmp/pdp_plot.png")
    plt.close()

    # ICE plot
    fig3 = interpreter.plot_ice(top_feature_idx, X_test[:30], n_samples=30, show_pdp=True)
    plt.savefig('/tmp/ice_plot.png', dpi=150, bbox_inches='tight')
    print("   Saved: /tmp/ice_plot.png")
    plt.close()

    # SHAP summary plot
    fig4 = interpreter.plot_shap_summary(X_test[:100], n_samples=50, max_display=8)
    plt.savefig('/tmp/shap_summary.png', dpi=150, bbox_inches='tight')
    print("   Saved: /tmp/shap_summary.png")
    plt.close()

    # Force plot
    fig5 = interpreter.plot_force_plot(test_instance, n_samples=50, max_display=6)
    plt.savefig('/tmp/force_plot.png', dpi=150, bbox_inches='tight')
    print("   Saved: /tmp/force_plot.png")
    plt.close()

    # Comprehensive report
    print("\n10. Creating comprehensive explanation report...")
    report = interpreter.create_explanation_report(
        test_instance, X_test[:100], y_test[:100]
    )
    print(f"   Prediction: {report['prediction']:.4f}")
    print(f"   Top LIME features: {report['lime_explanation']['top_features'][:3]}")
    print(f"   Report contains {len(report)} sections")

    # Global vs Local comparison
    print("\n11. Global vs Local Interpretability Comparison:")
    print("   " + "-" * 76)
    print(f"   {'Feature':<15} {'Global (Perm)':<18} {'Local (SHAP)':<18} {'Local (LIME)':<18}")
    print("   " + "-" * 76)

    global_imp = perm_importance['importances_mean']
    local_shap = np.abs(shap_values[0])
    lime_dict = dict(zip(lime_result.feature_names, lime_result.values))

    for i in sorted_idx[:5]:
        fname = feature_names[i]
        global_val = global_imp[i]
        local_shap_val = local_shap[i]
        local_lime_val = abs(lime_dict.get(fname, 0.0))
        print(f"   {fname:<15} {global_val:>17.4f} {local_shap_val:>17.4f} {local_lime_val:>17.4f}")

    print("   " + "-" * 76)

    print("\n" + "=" * 80)
    print("DEMO COMPLETE!")
    print("=" * 80)
    print("\nKey Insights:")
    print("1. SHAP values provide consistent local explanations")
    print("2. LIME offers interpretable linear approximations")
    print("3. Permutation importance reveals global feature relevance")
    print("4. PDP shows average feature effects")
    print("5. ICE curves reveal heterogeneous effects across instances")
    print("\nAll visualizations saved to /tmp/")


if __name__ == "__main__":
    demo()
