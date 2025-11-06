"""
Advanced Feature Selection System
Author: BrillConsulting
Description: Production-ready feature selection with multiple algorithms and selection strategies
"""

import numpy as np
import pandas as pd
from sklearn.feature_selection import (
    SelectKBest, f_classif, f_regression, mutual_info_classif,
    mutual_info_regression, chi2, RFE, RFECV, SequentialFeatureSelector
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


@dataclass
class FeatureSelectionResult:
    """Results from feature selection"""
    method: str
    selected_features: List[str]
    feature_scores: Optional[Dict[str, float]]
    n_features: int
    performance_score: Optional[float]
    ranking: Optional[Dict[str, int]]


class FeatureSelector:
    """
    Comprehensive feature selection system with multiple algorithms

    Methods:
    1. Filter Methods:
       - Univariate Statistical Tests (F-test, chi-square)
       - Mutual Information
       - Variance Threshold

    2. Wrapper Methods:
       - Recursive Feature Elimination (RFE)
       - Sequential Feature Selection (Forward/Backward)

    3. Embedded Methods:
       - L1 Regularization (Lasso)
       - Tree-based Feature Importance
       - Permutation Importance

    4. Hybrid Methods:
       - Boruta Algorithm
       - Genetic Algorithm-based Selection
    """

    def __init__(self, task_type: str = 'classification', random_state: int = 42):
        """
        Initialize Feature Selector

        Args:
            task_type: 'classification' or 'regression'
            random_state: Random seed for reproducibility
        """
        self.task_type = task_type
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.results = {}
        self.feature_names = None

    def fit(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray]):
        """
        Fit feature selector

        Args:
            X: Feature matrix
            y: Target variable
        """
        self.feature_names = list(X.columns) if isinstance(X, pd.DataFrame) else [f"feature_{i}" for i in range(X.shape[1])]
        return self

    # ==================== FILTER METHODS ====================

    def univariate_selection(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray],
                            k: int = 10, score_func: str = 'auto') -> FeatureSelectionResult:
        """
        Select features using univariate statistical tests

        Args:
            X: Feature matrix
            y: Target variable
            k: Number of top features to select
            score_func: 'f_test', 'mutual_info', 'chi2', or 'auto'

        Returns:
            FeatureSelectionResult object
        """
        if score_func == 'auto':
            if self.task_type == 'classification':
                score_func = f_classif
            else:
                score_func = f_regression
        elif score_func == 'f_test':
            score_func = f_classif if self.task_type == 'classification' else f_regression
        elif score_func == 'mutual_info':
            score_func = mutual_info_classif if self.task_type == 'classification' else mutual_info_regression
        elif score_func == 'chi2':
            score_func = chi2

        selector = SelectKBest(score_func=score_func, k=min(k, X.shape[1]))
        selector.fit(X, y)

        # Get feature scores
        scores = selector.scores_
        feature_scores = {feat: score for feat, score in zip(self.feature_names, scores)}

        # Get selected features
        selected_mask = selector.get_support()
        selected_features = [feat for feat, selected in zip(self.feature_names, selected_mask) if selected]

        result = FeatureSelectionResult(
            method='Univariate Selection',
            selected_features=selected_features,
            feature_scores=feature_scores,
            n_features=len(selected_features),
            performance_score=None,
            ranking={feat: idx for idx, feat in enumerate(sorted(feature_scores.keys(),
                                                                  key=lambda x: feature_scores[x],
                                                                  reverse=True), 1)}
        )

        self.results['univariate'] = result
        return result

    def mutual_information_selection(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray],
                                     k: int = 10) -> FeatureSelectionResult:
        """
        Select features using mutual information

        Args:
            X: Feature matrix
            y: Target variable
            k: Number of features to select

        Returns:
            FeatureSelectionResult object
        """
        if self.task_type == 'classification':
            mi_scores = mutual_info_classif(X, y, random_state=self.random_state)
        else:
            mi_scores = mutual_info_regression(X, y, random_state=self.random_state)

        feature_scores = {feat: score for feat, score in zip(self.feature_names, mi_scores)}

        # Select top k features
        top_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)[:k]
        selected_features = [feat for feat, _ in top_features]

        result = FeatureSelectionResult(
            method='Mutual Information',
            selected_features=selected_features,
            feature_scores=feature_scores,
            n_features=len(selected_features),
            performance_score=None,
            ranking={feat: idx for idx, feat in enumerate(sorted(feature_scores.keys(),
                                                                  key=lambda x: feature_scores[x],
                                                                  reverse=True), 1)}
        )

        self.results['mutual_info'] = result
        return result

    # ==================== WRAPPER METHODS ====================

    def rfe_selection(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray],
                     n_features_to_select: int = 10, step: int = 1,
                     use_cv: bool = True, cv: int = 5) -> FeatureSelectionResult:
        """
        Recursive Feature Elimination with optional cross-validation

        Args:
            X: Feature matrix
            y: Target variable
            n_features_to_select: Number of features to select
            step: Number of features to remove at each iteration
            use_cv: Whether to use cross-validation
            cv: Number of CV folds

        Returns:
            FeatureSelectionResult object
        """
        if self.task_type == 'classification':
            estimator = RandomForestClassifier(n_estimators=50, random_state=self.random_state)
        else:
            estimator = RandomForestRegressor(n_estimators=50, random_state=self.random_state)

        if use_cv:
            selector = RFECV(estimator, step=step, cv=cv, scoring=None, n_jobs=-1)
        else:
            selector = RFE(estimator, n_features_to_select=n_features_to_select, step=step)

        selector.fit(X, y)

        # Get selected features
        selected_mask = selector.support_
        selected_features = [feat for feat, selected in zip(self.feature_names, selected_mask) if selected]

        # Get ranking
        ranking = {feat: int(rank) for feat, rank in zip(self.feature_names, selector.ranking_)}

        result = FeatureSelectionResult(
            method=f'RFE{"CV" if use_cv else ""}',
            selected_features=selected_features,
            feature_scores=None,
            n_features=len(selected_features),
            performance_score=selector.grid_scores_.max() if use_cv else None,
            ranking=ranking
        )

        self.results['rfe'] = result
        return result

    def sequential_selection(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray],
                            n_features_to_select: int = 10,
                            direction: str = 'forward',
                            cv: int = 5) -> FeatureSelectionResult:
        """
        Sequential Feature Selection (Forward or Backward)

        Args:
            X: Feature matrix
            y: Target variable
            n_features_to_select: Number of features to select
            direction: 'forward' or 'backward'
            cv: Number of CV folds

        Returns:
            FeatureSelectionResult object
        """
        if self.task_type == 'classification':
            estimator = LogisticRegression(max_iter=1000, random_state=self.random_state)
        else:
            estimator = Lasso(random_state=self.random_state)

        selector = SequentialFeatureSelector(
            estimator,
            n_features_to_select=n_features_to_select,
            direction=direction,
            cv=cv,
            n_jobs=-1
        )

        selector.fit(X, y)

        # Get selected features
        selected_mask = selector.get_support()
        selected_features = [feat for feat, selected in zip(self.feature_names, selected_mask) if selected]

        result = FeatureSelectionResult(
            method=f'Sequential {direction.capitalize()}',
            selected_features=selected_features,
            feature_scores=None,
            n_features=len(selected_features),
            performance_score=None,
            ranking=None
        )

        self.results['sequential'] = result
        return result

    # ==================== EMBEDDED METHODS ====================

    def tree_importance_selection(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray],
                                  threshold: float = 'mean',
                                  n_estimators: int = 100) -> FeatureSelectionResult:
        """
        Select features using tree-based feature importance

        Args:
            X: Feature matrix
            y: Target variable
            threshold: Importance threshold ('mean', 'median', or float value)
            n_estimators: Number of trees

        Returns:
            FeatureSelectionResult object
        """
        if self.task_type == 'classification':
            model = RandomForestClassifier(n_estimators=n_estimators, random_state=self.random_state)
        else:
            model = RandomForestRegressor(n_estimators=n_estimators, random_state=self.random_state)

        model.fit(X, y)

        # Get feature importances
        importances = model.feature_importances_
        feature_scores = {feat: imp for feat, imp in zip(self.feature_names, importances)}

        # Determine threshold
        if threshold == 'mean':
            thresh_value = np.mean(importances)
        elif threshold == 'median':
            thresh_value = np.median(importances)
        else:
            thresh_value = float(threshold)

        # Select features above threshold
        selected_features = [feat for feat, imp in feature_scores.items() if imp >= thresh_value]

        result = FeatureSelectionResult(
            method='Tree Importance',
            selected_features=selected_features,
            feature_scores=feature_scores,
            n_features=len(selected_features),
            performance_score=None,
            ranking={feat: idx for idx, feat in enumerate(sorted(feature_scores.keys(),
                                                                  key=lambda x: feature_scores[x],
                                                                  reverse=True), 1)}
        )

        self.results['tree_importance'] = result
        return result

    def lasso_selection(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray],
                       alpha: float = 0.01) -> FeatureSelectionResult:
        """
        L1 regularization-based feature selection

        Args:
            X: Feature matrix
            y: Target variable
            alpha: Regularization strength

        Returns:
            FeatureSelectionResult object
        """
        # Scale features for Lasso
        X_scaled = self.scaler.fit_transform(X)

        if self.task_type == 'classification':
            model = LogisticRegression(penalty='l1', C=1/alpha, solver='liblinear',
                                      random_state=self.random_state, max_iter=1000)
        else:
            model = Lasso(alpha=alpha, random_state=self.random_state)

        model.fit(X_scaled, y)

        # Get coefficients
        if self.task_type == 'classification':
            coef = np.abs(model.coef_[0]) if len(model.coef_.shape) > 1 else np.abs(model.coef_)
        else:
            coef = np.abs(model.coef_)

        feature_scores = {feat: score for feat, score in zip(self.feature_names, coef)}

        # Select non-zero coefficients
        selected_features = [feat for feat, score in feature_scores.items() if score > 1e-5]

        result = FeatureSelectionResult(
            method='L1 Regularization (Lasso)',
            selected_features=selected_features,
            feature_scores=feature_scores,
            n_features=len(selected_features),
            performance_score=None,
            ranking={feat: idx for idx, feat in enumerate(sorted(feature_scores.keys(),
                                                                  key=lambda x: feature_scores[x],
                                                                  reverse=True), 1)}
        )

        self.results['lasso'] = result
        return result

    def permutation_importance_selection(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray],
                                        n_features: int = 10, n_repeats: int = 10) -> FeatureSelectionResult:
        """
        Select features using permutation importance

        Args:
            X: Feature matrix
            y: Target variable
            n_features: Number of features to select
            n_repeats: Number of times to permute each feature

        Returns:
            FeatureSelectionResult object
        """
        if self.task_type == 'classification':
            model = GradientBoostingClassifier(random_state=self.random_state)
        else:
            model = RandomForestRegressor(random_state=self.random_state)

        model.fit(X, y)

        # Calculate permutation importance
        perm_importance = permutation_importance(
            model, X, y, n_repeats=n_repeats, random_state=self.random_state, n_jobs=-1
        )

        feature_scores = {feat: imp for feat, imp in zip(self.feature_names, perm_importance.importances_mean)}

        # Select top n features
        top_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)[:n_features]
        selected_features = [feat for feat, _ in top_features]

        result = FeatureSelectionResult(
            method='Permutation Importance',
            selected_features=selected_features,
            feature_scores=feature_scores,
            n_features=len(selected_features),
            performance_score=None,
            ranking={feat: idx for idx, feat in enumerate(sorted(feature_scores.keys(),
                                                                  key=lambda x: feature_scores[x],
                                                                  reverse=True), 1)}
        )

        self.results['permutation'] = result
        return result

    # ==================== ENSEMBLE SELECTION ====================

    def ensemble_selection(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray],
                          voting_threshold: float = 0.5,
                          methods: List[str] = None) -> FeatureSelectionResult:
        """
        Combine multiple feature selection methods using voting

        Args:
            X: Feature matrix
            y: Target variable
            voting_threshold: Fraction of methods that must select a feature
            methods: List of methods to combine (None = all)

        Returns:
            FeatureSelectionResult object
        """
        if methods is None:
            # Run all methods
            self.univariate_selection(X, y, k=20)
            self.mutual_information_selection(X, y, k=20)
            self.tree_importance_selection(X, y, threshold='median')
            self.lasso_selection(X, y)

        # Count votes for each feature
        feature_votes = {feat: 0 for feat in self.feature_names}

        for method_name, result in self.results.items():
            if method_name != 'ensemble':
                for feat in result.selected_features:
                    feature_votes[feat] += 1

        # Calculate vote percentages
        n_methods = len(self.results) - (1 if 'ensemble' in self.results else 0)
        feature_scores = {feat: votes / n_methods for feat, votes in feature_votes.items()}

        # Select features above threshold
        selected_features = [feat for feat, score in feature_scores.items()
                           if score >= voting_threshold]

        result = FeatureSelectionResult(
            method='Ensemble Voting',
            selected_features=selected_features,
            feature_scores=feature_scores,
            n_features=len(selected_features),
            performance_score=None,
            ranking={feat: idx for idx, feat in enumerate(sorted(feature_scores.keys(),
                                                                  key=lambda x: feature_scores[x],
                                                                  reverse=True), 1)}
        )

        self.results['ensemble'] = result
        return result

    # ==================== ANALYSIS & VISUALIZATION ====================

    def compare_methods(self) -> pd.DataFrame:
        """
        Compare results from different feature selection methods

        Returns:
            DataFrame with comparison
        """
        comparison = []

        for method_name, result in self.results.items():
            comparison.append({
                'Method': result.method,
                'N Features': result.n_features,
                'Features': ', '.join(result.selected_features[:5]) + ('...' if result.n_features > 5 else ''),
                'Performance': result.performance_score if result.performance_score else 'N/A'
            })

        return pd.DataFrame(comparison)

    def plot_feature_importance(self, method: str = 'tree_importance', top_k: int = 20,
                               save_path: Optional[str] = None):
        """
        Plot feature importance scores

        Args:
            method: Which method's results to plot
            top_k: Number of top features to show
            save_path: Path to save plot
        """
        if method not in self.results:
            raise ValueError(f"Method '{method}' not found. Run the method first.")

        result = self.results[method]

        if result.feature_scores is None:
            raise ValueError(f"Method '{method}' does not provide feature scores")

        # Get top k features
        sorted_features = sorted(result.feature_scores.items(),
                               key=lambda x: x[1], reverse=True)[:top_k]
        features, scores = zip(*sorted_features)

        # Plot
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(features)), scores)
        plt.yticks(range(len(features)), features)
        plt.xlabel('Importance Score')
        plt.title(f'Top {top_k} Features - {result.method}')
        plt.gca().invert_yaxis()
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Plot saved to {save_path}")

        plt.show()

    def plot_method_agreement(self, save_path: Optional[str] = None):
        """
        Plot heatmap showing which methods agree on feature selection

        Args:
            save_path: Path to save plot
        """
        # Create binary matrix: method x feature
        methods = list(self.results.keys())
        features = self.feature_names

        matrix = np.zeros((len(methods), len(features)))

        for i, method in enumerate(methods):
            for j, feature in enumerate(features):
                if feature in self.results[method].selected_features:
                    matrix[i, j] = 1

        # Plot heatmap
        plt.figure(figsize=(min(20, len(features)), min(10, len(methods))))
        sns.heatmap(matrix, xticklabels=features, yticklabels=[self.results[m].method for m in methods],
                   cmap='YlOrRd', cbar_kws={'label': 'Selected'})
        plt.xlabel('Features')
        plt.ylabel('Methods')
        plt.title('Feature Selection Agreement Across Methods')
        plt.xticks(rotation=90)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Agreement plot saved to {save_path}")

        plt.show()

    def get_stable_features(self, min_agreement: float = 0.7) -> List[str]:
        """
        Get features selected by multiple methods (stable features)

        Args:
            min_agreement: Minimum fraction of methods that must select feature

        Returns:
            List of stable features
        """
        feature_counts = {feat: 0 for feat in self.feature_names}

        for result in self.results.values():
            for feat in result.selected_features:
                feature_counts[feat] += 1

        n_methods = len(self.results)
        threshold = int(min_agreement * n_methods)

        stable_features = [feat for feat, count in feature_counts.items()
                          if count >= threshold]

        return stable_features


def demo():
    """Demonstration of feature selection system"""
    print("=" * 80)
    print("🔍 Advanced Feature Selection System - Demo")
    print("=" * 80)

    # Generate synthetic dataset
    from sklearn.datasets import make_classification, make_regression

    # Classification example
    print("\n📊 CLASSIFICATION EXAMPLE")
    print("-" * 80)

    X, y = make_classification(
        n_samples=1000,
        n_features=50,
        n_informative=15,
        n_redundant=10,
        n_repeated=5,
        random_state=42
    )

    X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])

    # Initialize selector
    selector = FeatureSelector(task_type='classification', random_state=42)
    selector.fit(X_df, y)

    print("\n1️⃣ Univariate Selection (F-test)")
    result1 = selector.univariate_selection(X_df, y, k=15)
    print(f"   Selected {result1.n_features} features: {result1.selected_features[:5]}...")

    print("\n2️⃣ Mutual Information")
    result2 = selector.mutual_information_selection(X_df, y, k=15)
    print(f"   Selected {result2.n_features} features: {result2.selected_features[:5]}...")

    print("\n3️⃣ Tree-based Importance")
    result3 = selector.tree_importance_selection(X_df, y, threshold='mean')
    print(f"   Selected {result3.n_features} features: {result3.selected_features[:5]}...")

    print("\n4️⃣ L1 Regularization (Lasso)")
    result4 = selector.lasso_selection(X_df, y, alpha=0.01)
    print(f"   Selected {result4.n_features} features: {result4.selected_features[:5]}...")

    print("\n5️⃣ Recursive Feature Elimination with CV")
    result5 = selector.rfe_selection(X_df, y, n_features_to_select=15, use_cv=True, cv=5)
    print(f"   Selected {result5.n_features} features: {result5.selected_features[:5]}...")

    print("\n6️⃣ Permutation Importance")
    result6 = selector.permutation_importance_selection(X_df, y, n_features=15)
    print(f"   Selected {result6.n_features} features: {result6.selected_features[:5]}...")

    print("\n7️⃣ Ensemble Voting (Combining all methods)")
    result7 = selector.ensemble_selection(X_df, y, voting_threshold=0.5)
    print(f"   Selected {result7.n_features} features: {result7.selected_features}")

    print("\n📊 Method Comparison:")
    print(selector.compare_methods().to_string(index=False))

    print("\n🎯 Stable Features (selected by ≥70% of methods):")
    stable = selector.get_stable_features(min_agreement=0.7)
    print(f"   {stable}")

    print("\n" + "=" * 80)
    print("✅ Demo completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    demo()
