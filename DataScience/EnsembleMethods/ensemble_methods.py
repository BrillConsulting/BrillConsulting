"""
Ensemble Methods Toolkit
=========================

Comprehensive ensemble learning methods:
- Bagging (Bootstrap Aggregating)
- Boosting (AdaBoost, Gradient Boosting)
- Stacking (Meta-learning)
- Voting (Hard and Soft voting)
- Blending
- Feature importance aggregation
- Ensemble diversity metrics
- Model combination strategies

Author: Brill Consulting
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.base import clone
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')


class EnsembleMethodsToolkit:
    """Comprehensive ensemble methods toolkit with multiple strategies."""

    def __init__(self, random_state: int = 42):
        """
        Initialize ensemble methods toolkit.

        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        np.random.seed(random_state)
        self.base_models = []
        self.meta_model = None
        self.feature_importances = {}

    def bagging(self, X: np.ndarray, y: np.ndarray,
                base_estimator=None, n_estimators: int = 10,
                max_samples: float = 1.0, task: str = 'classification') -> Dict:
        """
        Implement Bootstrap Aggregating (Bagging).

        Args:
            X: Training features (n_samples, n_features)
            y: Training labels
            base_estimator: Base estimator to use (None = DecisionTree)
            n_estimators: Number of bootstrap samples
            max_samples: Fraction of samples to use for each bootstrap
            task: 'classification' or 'regression'

        Returns:
            Dictionary with trained models and predictions
        """
        if base_estimator is None:
            if task == 'classification':
                base_estimator = DecisionTreeClassifier(random_state=self.random_state)
            else:
                base_estimator = DecisionTreeRegressor(random_state=self.random_state)

        models = []
        n_samples = int(max_samples * len(X))

        # Train models on bootstrap samples
        for i in range(n_estimators):
            # Bootstrap sampling
            indices = np.random.choice(len(X), size=n_samples, replace=True)
            X_bootstrap = X[indices]
            y_bootstrap = y[indices]

            # Train model
            model = clone(base_estimator)
            model.fit(X_bootstrap, y_bootstrap)
            models.append(model)

        self.base_models = models

        return {
            'models': models,
            'n_estimators': n_estimators,
            'task': task,
            'method': 'Bagging'
        }

    def predict_bagging(self, X: np.ndarray, models: List,
                       task: str = 'classification') -> np.ndarray:
        """
        Make predictions using bagging ensemble.

        Args:
            X: Test features
            models: List of trained models
            task: 'classification' or 'regression'

        Returns:
            Ensemble predictions
        """
        predictions = np.array([model.predict(X) for model in models])

        if task == 'classification':
            # Majority voting
            from scipy.stats import mode
            ensemble_pred, _ = mode(predictions, axis=0)
            return ensemble_pred.flatten()
        else:
            # Average predictions
            return np.mean(predictions, axis=0)

    def adaboost(self, X: np.ndarray, y: np.ndarray,
                 n_estimators: int = 50, learning_rate: float = 1.0) -> Dict:
        """
        Implement AdaBoost (Adaptive Boosting).

        Args:
            X: Training features (n_samples, n_features)
            y: Training labels (binary: 0 or 1)
            n_estimators: Number of boosting iterations
            learning_rate: Learning rate (shrinkage)

        Returns:
            Dictionary with trained models and weights
        """
        n_samples = len(X)
        # Initialize sample weights uniformly
        sample_weights = np.ones(n_samples) / n_samples

        models = []
        model_weights = []

        for i in range(n_estimators):
            # Train weak classifier
            model = DecisionTreeClassifier(max_depth=1, random_state=self.random_state + i)
            model.fit(X, y, sample_weight=sample_weights)

            # Make predictions
            predictions = model.predict(X)

            # Calculate error
            incorrect = predictions != y
            error = np.sum(sample_weights * incorrect) / np.sum(sample_weights)

            # Avoid division by zero and error >= 0.5
            if error >= 0.5 or error == 0:
                if i == 0:
                    # First iteration, keep the model
                    models.append(model)
                    model_weights.append(1.0)
                break

            # Calculate model weight
            alpha = learning_rate * 0.5 * np.log((1 - error) / (error + 1e-10))

            # Update sample weights
            sample_weights *= np.exp(alpha * (2 * incorrect - 1))
            sample_weights /= np.sum(sample_weights)

            models.append(model)
            model_weights.append(alpha)

        self.base_models = models
        self.model_weights = model_weights

        return {
            'models': models,
            'model_weights': model_weights,
            'n_estimators': len(models),
            'method': 'AdaBoost'
        }

    def predict_adaboost(self, X: np.ndarray, models: List,
                        model_weights: List[float]) -> np.ndarray:
        """
        Make predictions using AdaBoost ensemble.

        Args:
            X: Test features
            models: List of trained models
            model_weights: Weights for each model

        Returns:
            Ensemble predictions
        """
        # Weighted voting
        predictions = np.zeros(len(X))

        for model, weight in zip(models, model_weights):
            pred = model.predict(X)
            # Convert to -1, 1 for weighted sum
            pred = 2 * pred - 1
            predictions += weight * pred

        # Convert back to 0, 1
        return (predictions > 0).astype(int)

    def gradient_boosting(self, X: np.ndarray, y: np.ndarray,
                         n_estimators: int = 100, learning_rate: float = 0.1,
                         max_depth: int = 3) -> Dict:
        """
        Implement Gradient Boosting (for regression or classification).

        Uses sklearn's GradientBoostingClassifier for demonstration.

        Args:
            X: Training features
            y: Training labels
            n_estimators: Number of boosting stages
            learning_rate: Learning rate
            max_depth: Maximum depth of trees

        Returns:
            Dictionary with trained model
        """
        model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=self.random_state
        )

        model.fit(X, y)
        self.gradient_boosting_model = model

        return {
            'model': model,
            'n_estimators': n_estimators,
            'learning_rate': learning_rate,
            'feature_importances': model.feature_importances_,
            'method': 'Gradient Boosting'
        }

    def stacking(self, X_train: np.ndarray, y_train: np.ndarray,
                X_val: np.ndarray, y_val: np.ndarray,
                base_models: Optional[List] = None,
                meta_model=None, task: str = 'classification') -> Dict:
        """
        Implement Stacking (Meta-learning).

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            base_models: List of base models (None = default set)
            meta_model: Meta-learner model (None = LogisticRegression)
            task: 'classification' or 'regression'

        Returns:
            Dictionary with trained models
        """
        if base_models is None:
            if task == 'classification':
                base_models = [
                    DecisionTreeClassifier(max_depth=5, random_state=self.random_state),
                    RandomForestClassifier(n_estimators=50, random_state=self.random_state),
                    KNeighborsClassifier(n_neighbors=5),
                    SVC(kernel='rbf', probability=True, random_state=self.random_state)
                ]
            else:
                base_models = [
                    DecisionTreeRegressor(max_depth=5, random_state=self.random_state),
                    LinearRegression(),
                ]

        if meta_model is None:
            if task == 'classification':
                meta_model = LogisticRegression(random_state=self.random_state, max_iter=1000)
            else:
                meta_model = LinearRegression()

        # Train base models
        trained_base_models = []
        meta_features_train = []
        meta_features_val = []

        for model in base_models:
            # Train base model
            model_clone = clone(model)
            model_clone.fit(X_train, y_train)
            trained_base_models.append(model_clone)

            # Generate meta-features
            if task == 'classification' and hasattr(model_clone, 'predict_proba'):
                train_pred = model_clone.predict_proba(X_train)
                val_pred = model_clone.predict_proba(X_val)
            else:
                train_pred = model_clone.predict(X_train).reshape(-1, 1)
                val_pred = model_clone.predict(X_val).reshape(-1, 1)

            meta_features_train.append(train_pred)
            meta_features_val.append(val_pred)

        # Combine meta-features
        X_meta_train = np.hstack(meta_features_train)
        X_meta_val = np.hstack(meta_features_val)

        # Train meta-model
        meta_model.fit(X_meta_train, y_train)

        self.base_models = trained_base_models
        self.meta_model = meta_model

        # Evaluate
        val_predictions = meta_model.predict(X_meta_val)

        if task == 'classification':
            score = accuracy_score(y_val, val_predictions)
            metric = 'accuracy'
        else:
            score = r2_score(y_val, val_predictions)
            metric = 'r2_score'

        return {
            'base_models': trained_base_models,
            'meta_model': meta_model,
            'n_base_models': len(trained_base_models),
            f'{metric}': score,
            'method': 'Stacking'
        }

    def predict_stacking(self, X: np.ndarray, base_models: List,
                        meta_model, task: str = 'classification') -> np.ndarray:
        """
        Make predictions using stacking ensemble.

        Args:
            X: Test features
            base_models: List of trained base models
            meta_model: Trained meta-model
            task: 'classification' or 'regression'

        Returns:
            Ensemble predictions
        """
        # Generate meta-features
        meta_features = []

        for model in base_models:
            if task == 'classification' and hasattr(model, 'predict_proba'):
                pred = model.predict_proba(X)
            else:
                pred = model.predict(X).reshape(-1, 1)
            meta_features.append(pred)

        X_meta = np.hstack(meta_features)

        # Meta-model prediction
        return meta_model.predict(X_meta)

    def voting_classifier(self, X: np.ndarray, y: np.ndarray,
                         models: Optional[List] = None,
                         voting: str = 'hard') -> Dict:
        """
        Implement Voting Classifier.

        Args:
            X: Training features
            y: Training labels
            models: List of models (None = default set)
            voting: 'hard' (majority) or 'soft' (weighted probabilities)

        Returns:
            Dictionary with trained models
        """
        if models is None:
            models = [
                DecisionTreeClassifier(max_depth=5, random_state=self.random_state),
                RandomForestClassifier(n_estimators=50, random_state=self.random_state),
                LogisticRegression(random_state=self.random_state, max_iter=1000)
            ]

        # Train all models
        trained_models = []
        for model in models:
            model_clone = clone(model)
            model_clone.fit(X, y)
            trained_models.append(model_clone)

        self.base_models = trained_models

        return {
            'models': trained_models,
            'voting': voting,
            'n_models': len(trained_models),
            'method': f'Voting ({voting})'
        }

    def predict_voting(self, X: np.ndarray, models: List,
                      voting: str = 'hard') -> np.ndarray:
        """
        Make predictions using voting ensemble.

        Args:
            X: Test features
            models: List of trained models
            voting: 'hard' or 'soft'

        Returns:
            Ensemble predictions
        """
        if voting == 'hard':
            # Majority voting
            predictions = np.array([model.predict(X) for model in models])
            from scipy.stats import mode
            ensemble_pred, _ = mode(predictions, axis=0)
            return ensemble_pred.flatten()
        else:
            # Soft voting: average probabilities
            probabilities = np.array([model.predict_proba(X) for model in models])
            avg_probabilities = np.mean(probabilities, axis=0)
            return np.argmax(avg_probabilities, axis=1)

    def blending(self, X_train: np.ndarray, y_train: np.ndarray,
                X_blend: np.ndarray, y_blend: np.ndarray,
                base_models: Optional[List] = None,
                blender_model=None, task: str = 'classification') -> Dict:
        """
        Implement Blending.

        Similar to stacking but uses a hold-out set instead of cross-validation.

        Args:
            X_train: Training features
            y_train: Training labels
            X_blend: Blending set features
            y_blend: Blending set labels
            base_models: List of base models
            blender_model: Blender model
            task: 'classification' or 'regression'

        Returns:
            Dictionary with trained models
        """
        if base_models is None:
            if task == 'classification':
                base_models = [
                    DecisionTreeClassifier(max_depth=5, random_state=self.random_state),
                    RandomForestClassifier(n_estimators=50, random_state=self.random_state),
                    LogisticRegression(random_state=self.random_state, max_iter=1000)
                ]
            else:
                base_models = [
                    DecisionTreeRegressor(max_depth=5, random_state=self.random_state),
                    LinearRegression()
                ]

        if blender_model is None:
            if task == 'classification':
                blender_model = LogisticRegression(random_state=self.random_state, max_iter=1000)
            else:
                blender_model = LinearRegression()

        # Train base models on training data
        trained_base_models = []
        blend_features = []

        for model in base_models:
            model_clone = clone(model)
            model_clone.fit(X_train, y_train)
            trained_base_models.append(model_clone)

            # Generate predictions on blending set
            if task == 'classification' and hasattr(model_clone, 'predict_proba'):
                pred = model_clone.predict_proba(X_blend)
            else:
                pred = model_clone.predict(X_blend).reshape(-1, 1)

            blend_features.append(pred)

        # Combine blend features
        X_blend_features = np.hstack(blend_features)

        # Train blender
        blender_model.fit(X_blend_features, y_blend)

        self.base_models = trained_base_models
        self.meta_model = blender_model

        return {
            'base_models': trained_base_models,
            'blender_model': blender_model,
            'n_base_models': len(trained_base_models),
            'method': 'Blending'
        }

    def aggregate_feature_importance(self, models: List,
                                    feature_names: Optional[List[str]] = None) -> Dict:
        """
        Aggregate feature importances from multiple models.

        Args:
            models: List of trained models with feature_importances_
            feature_names: Names of features

        Returns:
            Dictionary with aggregated feature importances
        """
        importances_list = []

        for model in models:
            if hasattr(model, 'feature_importances_'):
                importances_list.append(model.feature_importances_)
            elif hasattr(model, 'coef_'):
                # For linear models
                importances_list.append(np.abs(model.coef_).flatten())

        if not importances_list:
            return {'error': 'No models with feature importances found'}

        # Average importances
        avg_importances = np.mean(importances_list, axis=0)
        std_importances = np.std(importances_list, axis=0)

        if feature_names is None:
            feature_names = [f'Feature {i}' for i in range(len(avg_importances))]

        self.feature_importances = {
            'features': feature_names,
            'mean_importance': avg_importances,
            'std_importance': std_importances
        }

        # Sort by importance
        sorted_indices = np.argsort(avg_importances)[::-1]

        return {
            'features': [feature_names[i] for i in sorted_indices],
            'mean_importance': avg_importances[sorted_indices],
            'std_importance': std_importances[sorted_indices],
            'n_models': len(importances_list)
        }

    def calculate_diversity(self, models: List, X: np.ndarray,
                          y: np.ndarray) -> Dict:
        """
        Calculate ensemble diversity metrics.

        Args:
            models: List of trained models
            X: Features
            y: True labels

        Returns:
            Dictionary with diversity metrics
        """
        # Get predictions from all models
        predictions = np.array([model.predict(X) for model in models])

        # 1. Disagreement measure
        n_models = len(models)
        disagreements = []

        for i in range(n_models):
            for j in range(i + 1, n_models):
                disagreement = np.mean(predictions[i] != predictions[j])
                disagreements.append(disagreement)

        avg_disagreement = np.mean(disagreements) if disagreements else 0.0

        # 2. Q-statistic (for pairs of classifiers)
        q_statistics = []

        for i in range(n_models):
            for j in range(i + 1, n_models):
                # Confusion matrix elements
                n11 = np.sum((predictions[i] == y) & (predictions[j] == y))
                n00 = np.sum((predictions[i] != y) & (predictions[j] != y))
                n10 = np.sum((predictions[i] == y) & (predictions[j] != y))
                n01 = np.sum((predictions[i] != y) & (predictions[j] == y))

                # Q-statistic
                denominator = (n11 * n00 + n01 * n10)
                if denominator > 0:
                    q_stat = (n11 * n00 - n01 * n10) / denominator
                    q_statistics.append(q_stat)

        avg_q_statistic = np.mean(q_statistics) if q_statistics else 0.0

        # 3. Correlation coefficient
        correlations = []
        for i in range(n_models):
            for j in range(i + 1, n_models):
                corr = np.corrcoef(predictions[i], predictions[j])[0, 1]
                correlations.append(corr)

        avg_correlation = np.mean(correlations) if correlations else 0.0

        # 4. Individual accuracies
        accuracies = [accuracy_score(y, pred) for pred in predictions]

        return {
            'disagreement': float(avg_disagreement),
            'q_statistic': float(avg_q_statistic),
            'correlation': float(avg_correlation),
            'individual_accuracies': accuracies,
            'avg_accuracy': float(np.mean(accuracies)),
            'std_accuracy': float(np.std(accuracies)),
            'n_models': n_models
        }

    def visualize_feature_importance(self, importance_dict: Dict) -> plt.Figure:
        """
        Visualize aggregated feature importances.

        Args:
            importance_dict: Dictionary from aggregate_feature_importance

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        features = importance_dict['features'][:10]  # Top 10
        mean_imp = importance_dict['mean_importance'][:10]
        std_imp = importance_dict['std_importance'][:10]

        y_pos = np.arange(len(features))

        ax.barh(y_pos, mean_imp, xerr=std_imp, alpha=0.7,
               color='steelblue', edgecolor='black', capsize=5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        ax.invert_yaxis()
        ax.set_xlabel('Importance', fontsize=12)
        ax.set_title('Top 10 Feature Importances (Ensemble)', fontsize=13, fontweight='bold')
        ax.grid(alpha=0.3, axis='x')

        plt.tight_layout()
        return fig

    def visualize_model_comparison(self, results: Dict[str, Dict],
                                   X_test: np.ndarray,
                                   y_test: np.ndarray) -> plt.Figure:
        """
        Compare performance of different ensemble methods.

        Args:
            results: Dictionary of ensemble results
            X_test: Test features
            y_test: Test labels

        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        method_names = []
        accuracies = []
        n_models_list = []

        for method_name, result in results.items():
            method_names.append(method_name)

            # Get predictions based on method
            if 'models' in result and result['method'] == 'Bagging':
                y_pred = self.predict_bagging(X_test, result['models'])
            elif 'models' in result and 'model_weights' in result:
                y_pred = self.predict_adaboost(X_test, result['models'], result['model_weights'])
            elif 'model' in result:
                y_pred = result['model'].predict(X_test)
            elif 'base_models' in result and 'meta_model' in result:
                y_pred = self.predict_stacking(X_test, result['base_models'], result['meta_model'])
            elif 'models' in result and 'voting' in result:
                y_pred = self.predict_voting(X_test, result['models'], result['voting'])
            else:
                continue

            accuracy = accuracy_score(y_test, y_pred)
            accuracies.append(accuracy)
            n_models_list.append(result.get('n_estimators', result.get('n_models', result.get('n_base_models', 1))))

        # Accuracy comparison
        colors = plt.cm.viridis(np.linspace(0, 1, len(method_names)))
        axes[0].bar(method_names, accuracies, color=colors, alpha=0.7, edgecolor='black')
        axes[0].set_ylabel('Accuracy', fontsize=12)
        axes[0].set_title('Ensemble Method Comparison', fontsize=13, fontweight='bold')
        axes[0].set_ylim([0, 1])
        axes[0].grid(alpha=0.3, axis='y')
        axes[0].tick_params(axis='x', rotation=45)

        # Number of models
        axes[1].bar(method_names, n_models_list, color=colors, alpha=0.7, edgecolor='black')
        axes[1].set_ylabel('Number of Base Models', fontsize=12)
        axes[1].set_title('Ensemble Size', fontsize=13, fontweight='bold')
        axes[1].grid(alpha=0.3, axis='y')
        axes[1].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        return fig


def demo():
    """Demonstrate ensemble methods toolkit."""
    np.random.seed(42)

    print("Ensemble Methods Toolkit Demo")
    print("=" * 80)

    # 1. Generate synthetic data
    print("\n1. Generating Synthetic Classification Data")
    print("-" * 80)

    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    X, y = make_classification(
        n_samples=1000, n_features=20, n_informative=15,
        n_redundant=5, n_classes=2, random_state=42
    )

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")

    toolkit = EnsembleMethodsToolkit(random_state=42)

    # 2. Bagging
    print("\n2. Bootstrap Aggregating (Bagging)")
    print("-" * 80)
    bagging_result = toolkit.bagging(X_train, y_train, n_estimators=20)
    y_pred_bagging = toolkit.predict_bagging(X_test, bagging_result['models'])
    accuracy_bagging = accuracy_score(y_test, y_pred_bagging)
    print(f"Number of estimators: {bagging_result['n_estimators']}")
    print(f"Test Accuracy: {accuracy_bagging:.4f}")

    # 3. AdaBoost
    print("\n3. AdaBoost")
    print("-" * 80)
    adaboost_result = toolkit.adaboost(X_train, y_train, n_estimators=50, learning_rate=1.0)
    y_pred_adaboost = toolkit.predict_adaboost(X_test, adaboost_result['models'],
                                               adaboost_result['model_weights'])
    accuracy_adaboost = accuracy_score(y_test, y_pred_adaboost)
    print(f"Number of estimators: {adaboost_result['n_estimators']}")
    print(f"Test Accuracy: {accuracy_adaboost:.4f}")

    # 4. Gradient Boosting
    print("\n4. Gradient Boosting")
    print("-" * 80)
    gb_result = toolkit.gradient_boosting(X_train, y_train, n_estimators=100,
                                         learning_rate=0.1, max_depth=3)
    y_pred_gb = gb_result['model'].predict(X_test)
    accuracy_gb = accuracy_score(y_test, y_pred_gb)
    print(f"Number of estimators: {gb_result['n_estimators']}")
    print(f"Learning rate: {gb_result['learning_rate']}")
    print(f"Test Accuracy: {accuracy_gb:.4f}")

    # 5. Stacking
    print("\n5. Stacking (Meta-Learning)")
    print("-" * 80)
    stacking_result = toolkit.stacking(X_train, y_train, X_val, y_val)
    y_pred_stacking = toolkit.predict_stacking(X_test, stacking_result['base_models'],
                                              stacking_result['meta_model'])
    accuracy_stacking = accuracy_score(y_test, y_pred_stacking)
    print(f"Number of base models: {stacking_result['n_base_models']}")
    print(f"Validation Accuracy: {stacking_result['accuracy']:.4f}")
    print(f"Test Accuracy: {accuracy_stacking:.4f}")

    # 6. Voting Classifier
    print("\n6. Voting Classifier")
    print("-" * 80)

    # Hard voting
    voting_hard_result = toolkit.voting_classifier(X_train, y_train, voting='hard')
    y_pred_voting_hard = toolkit.predict_voting(X_test, voting_hard_result['models'], voting='hard')
    accuracy_voting_hard = accuracy_score(y_test, y_pred_voting_hard)
    print(f"Hard Voting - Number of models: {voting_hard_result['n_models']}")
    print(f"Hard Voting - Test Accuracy: {accuracy_voting_hard:.4f}")

    # Soft voting
    voting_soft_result = toolkit.voting_classifier(X_train, y_train, voting='soft')
    y_pred_voting_soft = toolkit.predict_voting(X_test, voting_soft_result['models'], voting='soft')
    accuracy_voting_soft = accuracy_score(y_test, y_pred_voting_soft)
    print(f"Soft Voting - Test Accuracy: {accuracy_voting_soft:.4f}")

    # 7. Blending
    print("\n7. Blending")
    print("-" * 80)
    blending_result = toolkit.blending(X_train, y_train, X_val, y_val)
    y_pred_blending = toolkit.predict_stacking(X_test, blending_result['base_models'],
                                              blending_result['blender_model'])
    accuracy_blending = accuracy_score(y_test, y_pred_blending)
    print(f"Number of base models: {blending_result['n_base_models']}")
    print(f"Test Accuracy: {accuracy_blending:.4f}")

    # 8. Feature Importance Aggregation
    print("\n8. Feature Importance Aggregation")
    print("-" * 80)
    # Use bagging models
    importance_result = toolkit.aggregate_feature_importance(
        bagging_result['models'],
        feature_names=[f'Feature_{i}' for i in range(X.shape[1])]
    )
    print("Top 5 Most Important Features:")
    for i in range(5):
        print(f"  {importance_result['features'][i]}: "
              f"{importance_result['mean_importance'][i]:.4f} ± "
              f"{importance_result['std_importance'][i]:.4f}")

    # 9. Ensemble Diversity
    print("\n9. Ensemble Diversity Metrics")
    print("-" * 80)
    diversity = toolkit.calculate_diversity(bagging_result['models'], X_test, y_test)
    print(f"Average Disagreement: {diversity['disagreement']:.4f}")
    print(f"Average Q-Statistic: {diversity['q_statistic']:.4f}")
    print(f"Average Correlation: {diversity['correlation']:.4f}")
    print(f"Average Individual Accuracy: {diversity['avg_accuracy']:.4f} ± {diversity['std_accuracy']:.4f}")

    # 10. Performance Summary
    print("\n10. Performance Summary")
    print("-" * 80)
    print(f"{'Method':<25} {'Test Accuracy':<15}")
    print("-" * 80)
    print(f"{'Bagging':<25} {accuracy_bagging:<15.4f}")
    print(f"{'AdaBoost':<25} {accuracy_adaboost:<15.4f}")
    print(f"{'Gradient Boosting':<25} {accuracy_gb:<15.4f}")
    print(f"{'Stacking':<25} {accuracy_stacking:<15.4f}")
    print(f"{'Voting (Hard)':<25} {accuracy_voting_hard:<15.4f}")
    print(f"{'Voting (Soft)':<25} {accuracy_voting_soft:<15.4f}")
    print(f"{'Blending':<25} {accuracy_blending:<15.4f}")

    # 11. Visualizations
    print("\n11. Generating Visualizations")
    print("-" * 80)

    # Feature importance
    fig1 = toolkit.visualize_feature_importance(importance_result)
    fig1.savefig('ensemble_feature_importance.png', dpi=300, bbox_inches='tight')
    print("✓ Saved ensemble_feature_importance.png")
    plt.close()

    # Method comparison
    results_dict = {
        'Bagging': bagging_result,
        'AdaBoost': adaboost_result,
        'Gradient Boosting': gb_result,
        'Stacking': stacking_result,
        'Voting (Hard)': voting_hard_result,
        'Blending': blending_result
    }

    fig2 = toolkit.visualize_model_comparison(results_dict, X_test, y_test)
    fig2.savefig('ensemble_methods_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved ensemble_methods_comparison.png")
    plt.close()

    # Diversity visualization
    fig3, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Individual model accuracies
    models_idx = range(len(diversity['individual_accuracies']))
    axes[0].bar(models_idx, diversity['individual_accuracies'],
               alpha=0.7, color='steelblue', edgecolor='black')
    axes[0].axhline(y=diversity['avg_accuracy'], color='red', linestyle='--',
                   linewidth=2, label=f'Average: {diversity["avg_accuracy"]:.3f}')
    axes[0].set_xlabel('Model Index', fontsize=12)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].set_title('Individual Model Accuracies', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(alpha=0.3, axis='y')

    # Diversity metrics
    metrics = ['Disagreement', 'Q-Statistic', 'Correlation']
    values = [diversity['disagreement'], diversity['q_statistic'], diversity['correlation']]
    colors_div = ['orange', 'green', 'purple']

    axes[1].bar(metrics, values, alpha=0.7, color=colors_div, edgecolor='black')
    axes[1].set_ylabel('Value', fontsize=12)
    axes[1].set_title('Ensemble Diversity Metrics', fontsize=13, fontweight='bold')
    axes[1].grid(alpha=0.3, axis='y')

    plt.tight_layout()
    fig3.savefig('ensemble_diversity.png', dpi=300, bbox_inches='tight')
    print("✓ Saved ensemble_diversity.png")
    plt.close()

    print("\n" + "=" * 80)
    print("✓ Ensemble Methods Demo Complete!")
    print("=" * 80)


if __name__ == '__main__':
    demo()
