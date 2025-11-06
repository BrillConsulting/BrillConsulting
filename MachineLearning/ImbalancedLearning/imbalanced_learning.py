"""
Advanced Imbalanced Learning System
Author: BrillConsulting
Description: Production-ready techniques for handling imbalanced datasets with resampling and ensemble methods
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_X_y, check_array
from sklearn.utils.validation import check_is_fitted
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score,
                             precision_recall_curve, f1_score, balanced_accuracy_score)
from collections import Counter
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


@dataclass
class ResamplingResult:
    """Results from resampling operation"""
    method: str
    original_distribution: Dict[int, int]
    new_distribution: Dict[int, int]
    X_resampled: np.ndarray
    y_resampled: np.ndarray
    sampling_ratio: float


class SMOTE:
    """
    Synthetic Minority Over-sampling Technique (SMOTE)

    Creates synthetic samples from the minority class using k-nearest neighbors
    """

    def __init__(self, sampling_ratio: float = 1.0, k_neighbors: int = 5, random_state: int = 42):
        """
        Initialize SMOTE

        Args:
            sampling_ratio: Target ratio of minority to majority (1.0 = balanced)
            k_neighbors: Number of nearest neighbors for synthesis
            random_state: Random seed
        """
        self.sampling_ratio = sampling_ratio
        self.k_neighbors = k_neighbors
        self.random_state = random_state
        np.random.seed(random_state)

    def fit_resample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit and resample the dataset

        Args:
            X: Feature matrix
            y: Labels

        Returns:
            X_resampled, y_resampled
        """
        X, y = check_X_y(X, y)

        # Find minority and majority classes
        class_counts = Counter(y)
        minority_class = min(class_counts, key=class_counts.get)
        majority_class = max(class_counts, key=class_counts.get)

        # Calculate number of synthetic samples to generate
        n_minority = class_counts[minority_class]
        n_majority = class_counts[majority_class]
        n_synthetic = int(n_majority * self.sampling_ratio) - n_minority

        if n_synthetic <= 0:
            return X, y

        # Get minority class samples
        minority_indices = np.where(y == minority_class)[0]
        X_minority = X[minority_indices]

        # Fit k-NN on minority samples
        nn = NearestNeighbors(n_neighbors=self.k_neighbors + 1)
        nn.fit(X_minority)

        # Generate synthetic samples
        synthetic_samples = []

        for _ in range(n_synthetic):
            # Randomly select a minority sample
            idx = np.random.randint(0, len(X_minority))
            sample = X_minority[idx].reshape(1, -1)

            # Find k nearest neighbors
            neighbors_indices = nn.kneighbors(sample, return_distance=False)[0][1:]

            # Randomly select one neighbor
            neighbor_idx = np.random.choice(neighbors_indices)
            neighbor = X_minority[neighbor_idx]

            # Generate synthetic sample (random point on line between sample and neighbor)
            alpha = np.random.random()
            synthetic = sample + alpha * (neighbor - sample)
            synthetic_samples.append(synthetic.ravel())

        # Combine original and synthetic samples
        X_resampled = np.vstack([X, np.array(synthetic_samples)])
        y_resampled = np.hstack([y, np.full(n_synthetic, minority_class)])

        return X_resampled, y_resampled


class ADASYN:
    """
    Adaptive Synthetic Sampling (ADASYN)

    Focuses on generating samples in regions where the minority class is sparse
    """

    def __init__(self, sampling_ratio: float = 1.0, k_neighbors: int = 5, random_state: int = 42):
        """
        Initialize ADASYN

        Args:
            sampling_ratio: Target ratio of minority to majority
            k_neighbors: Number of nearest neighbors
            random_state: Random seed
        """
        self.sampling_ratio = sampling_ratio
        self.k_neighbors = k_neighbors
        self.random_state = random_state
        np.random.seed(random_state)

    def fit_resample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit and resample using ADASYN

        Args:
            X: Feature matrix
            y: Labels

        Returns:
            X_resampled, y_resampled
        """
        X, y = check_X_y(X, y)

        # Find minority and majority classes
        class_counts = Counter(y)
        minority_class = min(class_counts, key=class_counts.get)
        majority_class = max(class_counts, key=class_counts.get)

        # Calculate number of synthetic samples
        n_minority = class_counts[minority_class]
        n_majority = class_counts[majority_class]
        n_synthetic = int(n_majority * self.sampling_ratio) - n_minority

        if n_synthetic <= 0:
            return X, y

        # Get minority samples
        minority_indices = np.where(y == minority_class)[0]
        X_minority = X[minority_indices]

        # Fit k-NN on entire dataset
        nn = NearestNeighbors(n_neighbors=self.k_neighbors + 1)
        nn.fit(X)

        # Calculate density distribution (ratio of majority neighbors)
        density = []
        for sample in X_minority:
            neighbors_indices = nn.kneighbors(sample.reshape(1, -1), return_distance=False)[0]
            neighbors_labels = y[neighbors_indices]
            ratio = np.sum(neighbors_labels == majority_class) / self.k_neighbors
            density.append(ratio)

        density = np.array(density)

        # Normalize density to get sampling distribution
        if density.sum() == 0:
            density = np.ones(len(density))
        density = density / density.sum()

        # Generate synthetic samples based on density
        synthetic_samples = []
        n_samples_per_point = np.random.multinomial(n_synthetic, density)

        nn_minority = NearestNeighbors(n_neighbors=self.k_neighbors + 1)
        nn_minority.fit(X_minority)

        for idx, n_samples in enumerate(n_samples_per_point):
            if n_samples == 0:
                continue

            sample = X_minority[idx].reshape(1, -1)
            neighbors_indices = nn_minority.kneighbors(sample, return_distance=False)[0][1:]

            for _ in range(n_samples):
                neighbor_idx = np.random.choice(neighbors_indices)
                neighbor = X_minority[neighbor_idx]

                alpha = np.random.random()
                synthetic = sample + alpha * (neighbor - sample)
                synthetic_samples.append(synthetic.ravel())

        # Combine
        X_resampled = np.vstack([X, np.array(synthetic_samples)])
        y_resampled = np.hstack([y, np.full(len(synthetic_samples), minority_class)])

        return X_resampled, y_resampled


class RandomUnderSampler:
    """
    Random under-sampling of majority class
    """

    def __init__(self, sampling_ratio: float = 1.0, random_state: int = 42):
        """
        Initialize under-sampler

        Args:
            sampling_ratio: Target ratio of majority to minority
            random_state: Random seed
        """
        self.sampling_ratio = sampling_ratio
        self.random_state = random_state
        np.random.seed(random_state)

    def fit_resample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit and resample

        Args:
            X: Feature matrix
            y: Labels

        Returns:
            X_resampled, y_resampled
        """
        X, y = check_X_y(X, y)

        class_counts = Counter(y)
        minority_class = min(class_counts, key=class_counts.get)
        majority_class = max(class_counts, key=class_counts.get)

        n_minority = class_counts[minority_class]
        n_majority_target = int(n_minority / self.sampling_ratio)

        # Get indices
        minority_indices = np.where(y == minority_class)[0]
        majority_indices = np.where(y == majority_class)[0]

        # Randomly sample majority class
        majority_indices_sampled = np.random.choice(
            majority_indices,
            size=min(n_majority_target, len(majority_indices)),
            replace=False
        )

        # Combine
        selected_indices = np.hstack([minority_indices, majority_indices_sampled])
        np.random.shuffle(selected_indices)

        return X[selected_indices], y[selected_indices]


class TomekLinks:
    """
    Remove Tomek links (pairs of opposite class samples that are nearest neighbors)
    """

    def fit_resample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Remove Tomek links

        Args:
            X: Feature matrix
            y: Labels

        Returns:
            X_resampled, y_resampled
        """
        X, y = check_X_y(X, y)

        # Find nearest neighbors
        nn = NearestNeighbors(n_neighbors=2)
        nn.fit(X)
        neighbors_indices = nn.kneighbors(X, return_distance=False)

        # Identify Tomek links
        tomek_indices = set()
        for i in range(len(X)):
            j = neighbors_indices[i, 1]
            # Check if they are mutual nearest neighbors and opposite classes
            if neighbors_indices[j, 1] == i and y[i] != y[j]:
                # Remove majority class sample
                class_counts = Counter(y)
                majority_class = max(class_counts, key=class_counts.get)
                if y[i] == majority_class:
                    tomek_indices.add(i)
                else:
                    tomek_indices.add(j)

        # Keep samples not in Tomek links
        keep_indices = [i for i in range(len(X)) if i not in tomek_indices]

        return X[keep_indices], y[keep_indices]


class ImbalancedClassifier(BaseEstimator, ClassifierMixin):
    """
    Classifier wrapper for imbalanced datasets

    Combines resampling with classification
    """

    def __init__(self,
                 base_estimator=None,
                 sampling_strategy: str = 'smote',
                 sampling_ratio: float = 1.0,
                 k_neighbors: int = 5,
                 random_state: int = 42):
        """
        Initialize imbalanced classifier

        Args:
            base_estimator: Base classifier (default: RandomForest)
            sampling_strategy: 'smote', 'adasyn', 'undersample', 'smote_tomek'
            sampling_ratio: Sampling ratio
            k_neighbors: Number of neighbors for SMOTE/ADASYN
            random_state: Random seed
        """
        self.base_estimator = base_estimator
        self.sampling_strategy = sampling_strategy
        self.sampling_ratio = sampling_ratio
        self.k_neighbors = k_neighbors
        self.random_state = random_state

        if base_estimator is None:
            self.base_estimator = RandomForestClassifier(random_state=random_state)

    def fit(self, X, y):
        """
        Fit classifier on resampled data

        Args:
            X: Feature matrix
            y: Labels

        Returns:
            self
        """
        X, y = check_X_y(X, y)

        # Apply resampling
        if self.sampling_strategy == 'smote':
            sampler = SMOTE(self.sampling_ratio, self.k_neighbors, self.random_state)
            X_resampled, y_resampled = sampler.fit_resample(X, y)
        elif self.sampling_strategy == 'adasyn':
            sampler = ADASYN(self.sampling_ratio, self.k_neighbors, self.random_state)
            X_resampled, y_resampled = sampler.fit_resample(X, y)
        elif self.sampling_strategy == 'undersample':
            sampler = RandomUnderSampler(self.sampling_ratio, self.random_state)
            X_resampled, y_resampled = sampler.fit_resample(X, y)
        elif self.sampling_strategy == 'smote_tomek':
            # SMOTE followed by Tomek links removal
            smote = SMOTE(self.sampling_ratio, self.k_neighbors, self.random_state)
            X_temp, y_temp = smote.fit_resample(X, y)
            tomek = TomekLinks()
            X_resampled, y_resampled = tomek.fit_resample(X_temp, y_temp)
        else:
            X_resampled, y_resampled = X, y

        # Fit base estimator
        self.estimator_ = self.base_estimator
        self.estimator_.fit(X_resampled, y_resampled)
        self.classes_ = np.unique(y)

        return self

    def predict(self, X):
        """Predict class labels"""
        check_is_fitted(self)
        X = check_array(X)
        return self.estimator_.predict(X)

    def predict_proba(self, X):
        """Predict class probabilities"""
        check_is_fitted(self)
        X = check_array(X)
        return self.estimator_.predict_proba(X)


class ImbalancedLearningAnalyzer:
    """
    Comprehensive analysis and comparison of imbalanced learning techniques
    """

    def __init__(self, random_state: int = 42):
        """
        Initialize analyzer

        Args:
            random_state: Random seed
        """
        self.random_state = random_state
        self.results = {}

    def compare_sampling_strategies(self, X, y, base_estimator=None, cv: int = 5) -> pd.DataFrame:
        """
        Compare different sampling strategies

        Args:
            X: Feature matrix
            y: Labels
            base_estimator: Classifier to use
            cv: Number of CV folds

        Returns:
            DataFrame with results
        """
        if base_estimator is None:
            base_estimator = RandomForestClassifier(n_estimators=100, random_state=self.random_state)

        strategies = {
            'No Sampling': None,
            'SMOTE': 'smote',
            'ADASYN': 'adasyn',
            'Random Undersampling': 'undersample',
            'SMOTE + Tomek': 'smote_tomek'
        }

        results = []

        for name, strategy in strategies.items():
            if strategy is None:
                clf = base_estimator
            else:
                clf = ImbalancedClassifier(
                    base_estimator=base_estimator,
                    sampling_strategy=strategy,
                    random_state=self.random_state
                )

            # Cross-validation
            skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)

            # Multiple metrics
            accuracy = cross_val_score(clf, X, y, cv=skf, scoring='accuracy').mean()
            balanced_acc = cross_val_score(clf, X, y, cv=skf, scoring='balanced_accuracy').mean()
            f1 = cross_val_score(clf, X, y, cv=skf, scoring='f1_weighted').mean()

            results.append({
                'Strategy': name,
                'Accuracy': accuracy,
                'Balanced Accuracy': balanced_acc,
                'F1 Score': f1
            })

            self.results[name] = {
                'accuracy': accuracy,
                'balanced_accuracy': balanced_acc,
                'f1': f1
            }

        return pd.DataFrame(results).sort_values('F1 Score', ascending=False)

    def plot_class_distribution(self, y_original, y_resampled, save_path: Optional[str] = None):
        """
        Plot class distribution before and after resampling

        Args:
            y_original: Original labels
            y_resampled: Resampled labels
            save_path: Path to save plot
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Original distribution
        original_counts = Counter(y_original)
        axes[0].bar(original_counts.keys(), original_counts.values(), color='steelblue')
        axes[0].set_title('Original Class Distribution')
        axes[0].set_xlabel('Class')
        axes[0].set_ylabel('Count')
        axes[0].set_xticks(list(original_counts.keys()))

        # Resampled distribution
        resampled_counts = Counter(y_resampled)
        axes[1].bar(resampled_counts.keys(), resampled_counts.values(), color='coral')
        axes[1].set_title('Resampled Class Distribution')
        axes[1].set_xlabel('Class')
        axes[1].set_ylabel('Count')
        axes[1].set_xticks(list(resampled_counts.keys()))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Plot saved to {save_path}")

        plt.show()


def demo():
    """Demonstration of imbalanced learning techniques"""
    print("=" * 80)
    print("⚖️ Advanced Imbalanced Learning System - Demo")
    print("=" * 80)

    # Generate imbalanced dataset
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=2,
        weights=[0.9, 0.1],  # 90%-10% imbalance
        flip_y=0.01,
        random_state=42
    )

    print("\n📊 Dataset Information:")
    print(f"   Samples: {len(X)}")
    print(f"   Features: {X.shape[1]}")
    print(f"   Class distribution: {Counter(y)}")
    print(f"   Imbalance ratio: {max(Counter(y).values()) / min(Counter(y).values()):.1f}:1")

    # Test different resampling techniques
    print("\n" + "=" * 80)
    print("1️⃣ SMOTE (Synthetic Minority Over-sampling)")
    print("-" * 80)
    smote = SMOTE(sampling_ratio=1.0, k_neighbors=5, random_state=42)
    X_smote, y_smote = smote.fit_resample(X, y)
    print(f"   Original: {Counter(y)}")
    print(f"   After SMOTE: {Counter(y_smote)}")

    print("\n" + "=" * 80)
    print("2️⃣ ADASYN (Adaptive Synthetic Sampling)")
    print("-" * 80)
    adasyn = ADASYN(sampling_ratio=1.0, k_neighbors=5, random_state=42)
    X_adasyn, y_adasyn = adasyn.fit_resample(X, y)
    print(f"   Original: {Counter(y)}")
    print(f"   After ADASYN: {Counter(y_adasyn)}")

    print("\n" + "=" * 80)
    print("3️⃣ Random Under-sampling")
    print("-" * 80)
    undersampler = RandomUnderSampler(sampling_ratio=1.0, random_state=42)
    X_under, y_under = undersampler.fit_resample(X, y)
    print(f"   Original: {Counter(y)}")
    print(f"   After Under-sampling: {Counter(y_under)}")

    print("\n" + "=" * 80)
    print("4️⃣ Comparing All Strategies")
    print("-" * 80)

    analyzer = ImbalancedLearningAnalyzer(random_state=42)
    comparison = analyzer.compare_sampling_strategies(X, y, cv=5)

    print("\n📊 Performance Comparison:")
    print(comparison.to_string(index=False))

    print("\n" + "=" * 80)
    print("✅ Demo completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    demo()
