"""
Imbalanced Learning Toolkit

A comprehensive toolkit for handling imbalanced datasets with various
resampling techniques, cost-sensitive learning, and specialized metrics.

Author: Brill Consulting
Date: 2025-11-06
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Optional, Dict, List, Union, Any
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_recall_curve, roc_curve, auc, confusion_matrix,
    classification_report, f1_score, matthews_corrcoef
)
from collections import Counter
import warnings

warnings.filterwarnings('ignore')


class ImbalancedLearningToolkit:
    """
    Comprehensive toolkit for handling imbalanced datasets.

    Provides various resampling techniques, cost-sensitive learning methods,
    and specialized evaluation metrics for imbalanced classification problems.

    Attributes:
        random_state: Random seed for reproducibility
        verbose: Whether to print progress messages

    Example:
        >>> ilt = ImbalancedLearningToolkit(random_state=42)
        >>> X_resampled, y_resampled = ilt.smote(X, y, sampling_ratio=0.5)
        >>> metrics = ilt.evaluate_imbalanced(y_true, y_pred, y_scores)
    """

    def __init__(self, random_state: Optional[int] = None, verbose: bool = True):
        """
        Initialize the ImbalancedLearningToolkit.

        Args:
            random_state: Random seed for reproducibility
            verbose: Whether to print progress messages
        """
        self.random_state = random_state
        self.verbose = verbose
        if random_state is not None:
            np.random.seed(random_state)

    def smote(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sampling_ratio: float = 1.0,
        k_neighbors: int = 5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Synthetic Minority Over-sampling Technique (SMOTE).

        Generates synthetic samples for the minority class by interpolating
        between existing minority samples and their nearest neighbors.

        Args:
            X: Feature matrix
            y: Target labels
            sampling_ratio: Desired ratio of minority to majority samples (0 to 1)
            k_neighbors: Number of nearest neighbors to use

        Returns:
            Tuple of (X_resampled, y_resampled)
        """
        X = np.array(X)
        y = np.array(y)

        # Identify minority and majority classes
        classes, counts = np.unique(y, return_counts=True)
        minority_class = classes[np.argmin(counts)]
        majority_class = classes[np.argmax(counts)]

        minority_count = counts.min()
        majority_count = counts.max()

        if self.verbose:
            print(f"Original class distribution: {dict(zip(classes, counts))}")

        # Calculate number of synthetic samples needed
        desired_minority_count = int(majority_count * sampling_ratio)
        n_synthetic = max(0, desired_minority_count - minority_count)

        if n_synthetic == 0:
            if self.verbose:
                print("No oversampling needed.")
            return X, y

        # Get minority class samples
        minority_indices = np.where(y == minority_class)[0]
        X_minority = X[minority_indices]

        # Fit nearest neighbors
        nn = NearestNeighbors(n_neighbors=k_neighbors + 1)
        nn.fit(X_minority)

        # Generate synthetic samples
        synthetic_samples = []

        for _ in range(n_synthetic):
            # Random minority sample
            idx = np.random.randint(0, len(X_minority))
            sample = X_minority[idx]

            # Find k nearest neighbors
            neighbors_indices = nn.kneighbors(
                sample.reshape(1, -1),
                return_distance=False
            )[0][1:]  # Exclude the sample itself

            # Choose random neighbor
            neighbor_idx = np.random.choice(neighbors_indices)
            neighbor = X_minority[neighbor_idx]

            # Generate synthetic sample
            alpha = np.random.random()
            synthetic = sample + alpha * (neighbor - sample)
            synthetic_samples.append(synthetic)

        # Combine with original data
        X_resampled = np.vstack([X, np.array(synthetic_samples)])
        y_resampled = np.hstack([y, np.full(n_synthetic, minority_class)])

        if self.verbose:
            new_counts = Counter(y_resampled)
            print(f"Resampled class distribution: {dict(new_counts)}")

        return X_resampled, y_resampled

    def adasyn(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sampling_ratio: float = 1.0,
        k_neighbors: int = 5,
        beta: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Adaptive Synthetic Sampling (ADASYN).

        Similar to SMOTE but generates more synthetic samples in regions
        where minority samples are harder to learn (near majority samples).

        Args:
            X: Feature matrix
            y: Target labels
            sampling_ratio: Desired ratio of minority to majority samples
            k_neighbors: Number of nearest neighbors
            beta: Parameter controlling the density distribution (0 to 1)

        Returns:
            Tuple of (X_resampled, y_resampled)
        """
        X = np.array(X)
        y = np.array(y)

        # Identify minority and majority classes
        classes, counts = np.unique(y, return_counts=True)
        minority_class = classes[np.argmin(counts)]
        majority_class = classes[np.argmax(counts)]

        minority_count = counts.min()
        majority_count = counts.max()

        # Calculate total synthetic samples needed
        desired_minority_count = int(majority_count * sampling_ratio)
        n_synthetic_total = max(0, desired_minority_count - minority_count)

        if n_synthetic_total == 0:
            if self.verbose:
                print("No oversampling needed.")
            return X, y

        # Get minority samples
        minority_indices = np.where(y == minority_class)[0]
        X_minority = X[minority_indices]

        # Calculate density distribution
        nn = NearestNeighbors(n_neighbors=k_neighbors + 1)
        nn.fit(X)

        # For each minority sample, find ratio of majority neighbors
        ratios = []
        for sample in X_minority:
            neighbors_indices = nn.kneighbors(
                sample.reshape(1, -1),
                return_distance=False
            )[0][1:]

            majority_neighbors = np.sum(y[neighbors_indices] == majority_class)
            ratio = majority_neighbors / k_neighbors
            ratios.append(ratio)

        ratios = np.array(ratios)

        # Normalize ratios
        if ratios.sum() == 0:
            ratios = np.ones(len(ratios)) / len(ratios)
        else:
            ratios = ratios / ratios.sum()

        # Calculate number of synthetic samples per minority instance
        n_synthetic_per_instance = np.round(beta * ratios * n_synthetic_total).astype(int)

        # Generate synthetic samples
        synthetic_samples = []
        nn_minority = NearestNeighbors(n_neighbors=k_neighbors + 1)
        nn_minority.fit(X_minority)

        for idx, n_synthetic in enumerate(n_synthetic_per_instance):
            if n_synthetic == 0:
                continue

            sample = X_minority[idx]

            # Find k nearest neighbors in minority class
            neighbors_indices = nn_minority.kneighbors(
                sample.reshape(1, -1),
                return_distance=False
            )[0][1:]

            for _ in range(n_synthetic):
                # Choose random neighbor
                neighbor_idx = np.random.choice(neighbors_indices)
                neighbor = X_minority[neighbor_idx]

                # Generate synthetic sample
                alpha = np.random.random()
                synthetic = sample + alpha * (neighbor - sample)
                synthetic_samples.append(synthetic)

        if len(synthetic_samples) == 0:
            return X, y

        # Combine with original data
        X_resampled = np.vstack([X, np.array(synthetic_samples)])
        y_resampled = np.hstack([y, np.full(len(synthetic_samples), minority_class)])

        if self.verbose:
            new_counts = Counter(y_resampled)
            print(f"ADASYN resampled distribution: {dict(new_counts)}")

        return X_resampled, y_resampled

    def random_oversample(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sampling_ratio: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Random over-sampling of minority class.

        Args:
            X: Feature matrix
            y: Target labels
            sampling_ratio: Desired ratio of minority to majority samples

        Returns:
            Tuple of (X_resampled, y_resampled)
        """
        X = np.array(X)
        y = np.array(y)

        classes, counts = np.unique(y, return_counts=True)
        minority_class = classes[np.argmin(counts)]
        majority_count = counts.max()
        minority_count = counts.min()

        # Calculate samples to add
        desired_minority_count = int(majority_count * sampling_ratio)
        n_samples_to_add = max(0, desired_minority_count - minority_count)

        minority_indices = np.where(y == minority_class)[0]

        # Random sampling with replacement
        oversample_indices = np.random.choice(
            minority_indices,
            size=n_samples_to_add,
            replace=True
        )

        X_resampled = np.vstack([X, X[oversample_indices]])
        y_resampled = np.hstack([y, y[oversample_indices]])

        if self.verbose:
            new_counts = Counter(y_resampled)
            print(f"Random oversample distribution: {dict(new_counts)}")

        return X_resampled, y_resampled

    def random_undersample(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sampling_ratio: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Random under-sampling of majority class.

        Args:
            X: Feature matrix
            y: Target labels
            sampling_ratio: Desired ratio of minority to majority samples

        Returns:
            Tuple of (X_resampled, y_resampled)
        """
        X = np.array(X)
        y = np.array(y)

        classes, counts = np.unique(y, return_counts=True)
        minority_class = classes[np.argmin(counts)]
        majority_class = classes[np.argmax(counts)]
        minority_count = counts.min()

        # Calculate desired majority count
        desired_majority_count = int(minority_count / sampling_ratio)

        minority_indices = np.where(y == minority_class)[0]
        majority_indices = np.where(y == majority_class)[0]

        # Random sampling without replacement
        if desired_majority_count < len(majority_indices):
            undersample_indices = np.random.choice(
                majority_indices,
                size=desired_majority_count,
                replace=False
            )
        else:
            undersample_indices = majority_indices

        # Combine indices
        keep_indices = np.concatenate([minority_indices, undersample_indices])
        np.random.shuffle(keep_indices)

        X_resampled = X[keep_indices]
        y_resampled = y[keep_indices]

        if self.verbose:
            new_counts = Counter(y_resampled)
            print(f"Random undersample distribution: {dict(new_counts)}")

        return X_resampled, y_resampled

    def tomek_links(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Remove Tomek links (pairs of samples from different classes that are
        each other's nearest neighbors).

        Args:
            X: Feature matrix
            y: Target labels

        Returns:
            Tuple of (X_cleaned, y_cleaned)
        """
        X = np.array(X)
        y = np.array(y)

        # Find nearest neighbors
        nn = NearestNeighbors(n_neighbors=2)
        nn.fit(X)

        # Find Tomek links
        tomek_indices = set()

        for i in range(len(X)):
            # Find nearest neighbor
            neighbors = nn.kneighbors(X[i].reshape(1, -1), return_distance=False)[0]
            nearest_idx = neighbors[1]  # Skip self

            # Check if they form a Tomek link
            if y[i] != y[nearest_idx]:
                # Check if i is also nearest neighbor of nearest_idx
                neighbors_of_nearest = nn.kneighbors(
                    X[nearest_idx].reshape(1, -1),
                    return_distance=False
                )[0]

                if neighbors_of_nearest[1] == i:
                    # This is a Tomek link
                    # Remove majority class sample
                    classes, counts = np.unique(y, return_counts=True)
                    majority_class = classes[np.argmax(counts)]

                    if y[i] == majority_class:
                        tomek_indices.add(i)
                    else:
                        tomek_indices.add(nearest_idx)

        # Remove Tomek links
        keep_indices = [i for i in range(len(X)) if i not in tomek_indices]
        X_cleaned = X[keep_indices]
        y_cleaned = y[keep_indices]

        if self.verbose:
            print(f"Removed {len(tomek_indices)} Tomek links")
            new_counts = Counter(y_cleaned)
            print(f"Distribution after Tomek links removal: {dict(new_counts)}")

        return X_cleaned, y_cleaned

    def calculate_class_weights(
        self,
        y: np.ndarray,
        method: str = 'balanced'
    ) -> Dict[int, float]:
        """
        Calculate class weights for cost-sensitive learning.

        Args:
            y: Target labels
            method: Method to use ('balanced', 'inverse', or 'effective')

        Returns:
            Dictionary mapping class labels to weights
        """
        classes, counts = np.unique(y, return_counts=True)
        n_samples = len(y)
        n_classes = len(classes)

        weights = {}

        if method == 'balanced':
            # Sklearn's balanced class weight formula
            for cls, count in zip(classes, counts):
                weights[cls] = n_samples / (n_classes * count)

        elif method == 'inverse':
            # Simple inverse frequency
            for cls, count in zip(classes, counts):
                weights[cls] = 1.0 / count

            # Normalize
            total = sum(weights.values())
            weights = {k: v / total * n_classes for k, v in weights.items()}

        elif method == 'effective':
            # Effective number of samples
            beta = 0.9999
            for cls, count in zip(classes, counts):
                effective_num = (1.0 - beta ** count) / (1.0 - beta)
                weights[cls] = 1.0 / effective_num

            # Normalize
            total = sum(weights.values())
            weights = {k: v / total * n_classes for k, v in weights.items()}

        else:
            raise ValueError(f"Unknown method: {method}")

        if self.verbose:
            print(f"Class weights ({method}):")
            for cls, weight in weights.items():
                print(f"  Class {cls}: {weight:.4f}")

        return weights

    def optimize_threshold(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        metric: str = 'f1'
    ) -> Tuple[float, float]:
        """
        Optimize classification threshold for a given metric.

        Args:
            y_true: True labels
            y_scores: Predicted probabilities or scores
            metric: Metric to optimize ('f1', 'f_beta', 'gmean', or 'mcc')

        Returns:
            Tuple of (optimal_threshold, best_score)
        """
        thresholds = np.linspace(0, 1, 100)
        scores = []

        for threshold in thresholds:
            y_pred = (y_scores >= threshold).astype(int)

            if metric == 'f1':
                score = f1_score(y_true, y_pred, zero_division=0)
            elif metric.startswith('f_beta'):
                beta = float(metric.split('_')[1]) if '_' in metric else 1.0
                from sklearn.metrics import fbeta_score
                score = fbeta_score(y_true, y_pred, beta=beta, zero_division=0)
            elif metric == 'gmean':
                score = self._gmean_score(y_true, y_pred)
            elif metric == 'mcc':
                score = matthews_corrcoef(y_true, y_pred)
            else:
                raise ValueError(f"Unknown metric: {metric}")

            scores.append(score)

        best_idx = np.argmax(scores)
        optimal_threshold = thresholds[best_idx]
        best_score = scores[best_idx]

        if self.verbose:
            print(f"Optimal threshold for {metric}: {optimal_threshold:.4f}")
            print(f"Best {metric} score: {best_score:.4f}")

        return optimal_threshold, best_score

    def _gmean_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate geometric mean of sensitivity and specificity."""
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape != (2, 2):
            return 0.0

        tn, fp, fn, tp = cm.ravel()

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        return np.sqrt(sensitivity * specificity)

    def evaluate_imbalanced(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_scores: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Comprehensive evaluation metrics for imbalanced classification.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_scores: Predicted probabilities (optional, for AUC metrics)

        Returns:
            Dictionary of evaluation metrics
        """
        from sklearn.metrics import (
            precision_score, recall_score, f1_score, balanced_accuracy_score,
            cohen_kappa_score, matthews_corrcoef, roc_auc_score
        )

        metrics = {
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
            'gmean': self._gmean_score(y_true, y_pred),
            'mcc': matthews_corrcoef(y_true, y_pred),
            'cohen_kappa': cohen_kappa_score(y_true, y_pred)
        }

        # Add AUC metrics if scores provided
        if y_scores is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_scores)

            precision, recall, _ = precision_recall_curve(y_true, y_scores)
            metrics['pr_auc'] = auc(recall, precision)

        # Confusion matrix components
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
            metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0

        return metrics

    def plot_precision_recall_curve(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        figsize: Tuple[int, int] = (10, 6)
    ) -> plt.Figure:
        """
        Plot Precision-Recall curve.

        Args:
            y_true: True labels
            y_scores: Predicted probabilities
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
        pr_auc = auc(recall, precision)

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(recall, precision, linewidth=2, label=f'PR curve (AUC = {pr_auc:.3f})')

        # Baseline (random classifier)
        baseline = np.sum(y_true) / len(y_true)
        ax.axhline(y=baseline, color='gray', linestyle='--', label=f'Baseline ({baseline:.3f})')

        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve')
        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        figsize: Tuple[int, int] = (10, 6)
    ) -> plt.Figure:
        """
        Plot ROC curve.

        Args:
            y_true: True labels
            y_scores: Predicted probabilities
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(fpr, tpr, linewidth=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], 'k--', label='Random classifier')

        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve')
        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_class_distribution(
        self,
        y: np.ndarray,
        title: str = 'Class Distribution',
        figsize: Tuple[int, int] = (8, 6)
    ) -> plt.Figure:
        """
        Plot class distribution.

        Args:
            y: Target labels
            title: Plot title
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        classes, counts = np.unique(y, return_counts=True)

        fig, ax = plt.subplots(figsize=figsize)
        bars = ax.bar(classes, counts, alpha=0.7, edgecolor='black')

        # Add count labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                   f'{count}\n({count / len(y) * 100:.1f}%)',
                   ha='center', va='bottom')

        ax.set_xlabel('Class')
        ax.set_ylabel('Count')
        ax.set_title(title)
        ax.grid(axis='y', alpha=0.3)

        # Calculate imbalance ratio
        imbalance_ratio = counts.max() / counts.min()
        ax.text(0.02, 0.98, f'Imbalance Ratio: {imbalance_ratio:.2f}:1',
               transform=ax.transAxes, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        return fig

    def compare_resampling_methods(
        self,
        X: np.ndarray,
        y: np.ndarray,
        methods: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (14, 8)
    ) -> plt.Figure:
        """
        Compare different resampling methods.

        Args:
            X: Feature matrix
            y: Target labels
            methods: List of methods to compare
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        if methods is None:
            methods = ['original', 'random_oversample', 'random_undersample',
                      'smote', 'adasyn', 'tomek_links']

        results = {}

        # Original
        classes, counts = np.unique(y, return_counts=True)
        results['original'] = dict(zip(classes, counts))

        # Apply methods
        verbose_backup = self.verbose
        self.verbose = False

        for method in methods:
            if method == 'original':
                continue

            try:
                if method == 'random_oversample':
                    X_res, y_res = self.random_oversample(X, y)
                elif method == 'random_undersample':
                    X_res, y_res = self.random_undersample(X, y)
                elif method == 'smote':
                    X_res, y_res = self.smote(X, y)
                elif method == 'adasyn':
                    X_res, y_res = self.adasyn(X, y)
                elif method == 'tomek_links':
                    X_res, y_res = self.tomek_links(X, y)
                else:
                    continue

                classes_res, counts_res = np.unique(y_res, return_counts=True)
                results[method] = dict(zip(classes_res, counts_res))

            except Exception as e:
                print(f"Error in {method}: {e}")

        self.verbose = verbose_backup

        # Plot comparison
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.ravel()

        for idx, (method, distribution) in enumerate(results.items()):
            if idx >= len(axes):
                break

            ax = axes[idx]
            classes = list(distribution.keys())
            counts = list(distribution.values())

            bars = ax.bar(classes, counts, alpha=0.7, edgecolor='black')

            for bar, count in zip(bars, counts):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height,
                       f'{count}', ha='center', va='bottom', fontsize=9)

            ax.set_title(method.replace('_', ' ').title())
            ax.set_xlabel('Class')
            ax.set_ylabel('Count')
            ax.grid(axis='y', alpha=0.3)

            # Imbalance ratio
            if len(counts) > 1:
                ratio = max(counts) / min(counts)
                ax.text(0.02, 0.98, f'Ratio: {ratio:.2f}:1',
                       transform=ax.transAxes, verticalalignment='top',
                       fontsize=8,
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        return fig


def demo():
    """
    Demonstrate the ImbalancedLearningToolkit capabilities.
    """
    print("=" * 80)
    print("IMBALANCED LEARNING TOOLKIT DEMO")
    print("=" * 80)

    # Generate imbalanced dataset
    print("\n1. Generating imbalanced dataset...")
    np.random.seed(42)

    # Majority class (class 0)
    n_majority = 950
    X_majority = np.random.randn(n_majority, 10) + np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    # Minority class (class 1)
    n_minority = 50
    X_minority = np.random.randn(n_minority, 10) + np.array([2, 2, 2, 2, 2, 0, 0, 0, 0, 0])

    X = np.vstack([X_majority, X_minority])
    y = np.hstack([np.zeros(n_majority), np.ones(n_minority)])

    # Shuffle
    indices = np.random.permutation(len(X))
    X, y = X[indices], y[indices]

    print(f"   Total samples: {len(X)}")
    print(f"   Class 0 (majority): {np.sum(y == 0)}")
    print(f"   Class 1 (minority): {np.sum(y == 1)}")
    print(f"   Imbalance ratio: {np.sum(y == 0) / np.sum(y == 1):.2f}:1")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Initialize toolkit
    print("\n2. Initializing ImbalancedLearningToolkit...")
    ilt = ImbalancedLearningToolkit(random_state=42, verbose=True)

    # Test different resampling methods
    print("\n3. Testing SMOTE...")
    X_smote, y_smote = ilt.smote(X_train, y_train, sampling_ratio=0.8, k_neighbors=5)
    print(f"   Resampled size: {len(X_smote)}")

    print("\n4. Testing ADASYN...")
    X_adasyn, y_adasyn = ilt.adasyn(X_train, y_train, sampling_ratio=0.8, k_neighbors=5)
    print(f"   Resampled size: {len(X_adasyn)}")

    print("\n5. Testing random oversampling...")
    X_over, y_over = ilt.random_oversample(X_train, y_train, sampling_ratio=0.5)
    print(f"   Resampled size: {len(X_over)}")

    print("\n6. Testing random undersampling...")
    X_under, y_under = ilt.random_undersample(X_train, y_train, sampling_ratio=0.5)
    print(f"   Resampled size: {len(X_under)}")

    print("\n7. Testing Tomek links removal...")
    X_tomek, y_tomek = ilt.tomek_links(X_train, y_train)
    print(f"   Cleaned size: {len(X_tomek)}")

    # Class weights
    print("\n8. Calculating class weights...")
    weights_balanced = ilt.calculate_class_weights(y_train, method='balanced')
    weights_effective = ilt.calculate_class_weights(y_train, method='effective')

    # Train models on different datasets
    print("\n9. Training models on different resampled datasets...")

    datasets = {
        'Original': (X_train, y_train),
        'SMOTE': (X_smote, y_smote),
        'ADASYN': (X_adasyn, y_adasyn),
        'Random Oversample': (X_over, y_over),
        'Random Undersample': (X_under, y_under)
    }

    results = {}

    for name, (X_res, y_res) in datasets.items():
        print(f"\n   Training on {name} data...")
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_res, y_res)

        y_pred = model.predict(X_test)
        y_scores = model.predict_proba(X_test)[:, 1]

        metrics = ilt.evaluate_imbalanced(y_test, y_pred, y_scores)
        results[name] = metrics

        print(f"      F1 Score: {metrics['f1']:.4f}")
        print(f"      G-Mean: {metrics['gmean']:.4f}")
        print(f"      MCC: {metrics['mcc']:.4f}")
        print(f"      ROC-AUC: {metrics['roc_auc']:.4f}")
        print(f"      PR-AUC: {metrics['pr_auc']:.4f}")

    # Threshold optimization
    print("\n10. Optimizing classification threshold...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_smote, y_smote)
    y_scores = model.predict_proba(X_test)[:, 1]

    optimal_threshold, best_f1 = ilt.optimize_threshold(y_test, y_scores, metric='f1')
    print(f"    Optimal threshold: {optimal_threshold:.4f}")
    print(f"    Best F1 score: {best_f1:.4f}")

    # Compare with default threshold
    y_pred_default = model.predict(X_test)
    y_pred_optimized = (y_scores >= optimal_threshold).astype(int)

    print(f"\n    Default threshold (0.5):")
    print(f"      F1: {f1_score(y_test, y_pred_default):.4f}")
    print(f"    Optimized threshold ({optimal_threshold:.4f}):")
    print(f"      F1: {f1_score(y_test, y_pred_optimized):.4f}")

    # Visualizations
    print("\n11. Creating visualizations...")

    # Class distribution
    fig1 = ilt.plot_class_distribution(y_train, title='Training Class Distribution')
    plt.savefig('/tmp/class_distribution.png', dpi=150, bbox_inches='tight')
    print("    Saved: /tmp/class_distribution.png")
    plt.close()

    # Compare resampling methods
    fig2 = ilt.compare_resampling_methods(X_train, y_train)
    plt.savefig('/tmp/resampling_comparison.png', dpi=150, bbox_inches='tight')
    print("    Saved: /tmp/resampling_comparison.png")
    plt.close()

    # Precision-Recall curve
    fig3 = ilt.plot_precision_recall_curve(y_test, y_scores)
    plt.savefig('/tmp/precision_recall_curve.png', dpi=150, bbox_inches='tight')
    print("    Saved: /tmp/precision_recall_curve.png")
    plt.close()

    # ROC curve
    fig4 = ilt.plot_roc_curve(y_test, y_scores)
    plt.savefig('/tmp/roc_curve.png', dpi=150, bbox_inches='tight')
    print("    Saved: /tmp/roc_curve.png")
    plt.close()

    # Metrics comparison
    print("\n12. Comprehensive Metrics Comparison:")
    print("    " + "-" * 76)
    print(f"    {'Method':<20} {'F1':<10} {'G-Mean':<10} {'MCC':<10} {'ROC-AUC':<10} {'PR-AUC':<10}")
    print("    " + "-" * 76)

    for name, metrics in results.items():
        print(f"    {name:<20} {metrics['f1']:<10.4f} {metrics['gmean']:<10.4f} "
              f"{metrics['mcc']:<10.4f} {metrics['roc_auc']:<10.4f} {metrics['pr_auc']:<10.4f}")

    print("    " + "-" * 76)

    print("\n" + "=" * 80)
    print("DEMO COMPLETE!")
    print("=" * 80)
    print("\nKey Insights:")
    print("1. SMOTE and ADASYN create synthetic minority samples")
    print("2. Random oversampling duplicates existing minority samples")
    print("3. Random undersampling reduces majority class size")
    print("4. Tomek links removal cleans decision boundaries")
    print("5. Threshold optimization can significantly improve performance")
    print("6. Different metrics (F1, G-Mean, MCC) capture different aspects")
    print("7. PR-AUC is often more informative than ROC-AUC for imbalanced data")
    print("\nAll visualizations saved to /tmp/")


if __name__ == "__main__":
    demo()
