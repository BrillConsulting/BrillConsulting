"""
Anomaly Detection Toolkit
==========================

Comprehensive anomaly detection with multiple algorithms:
- Isolation Forest
- Local Outlier Factor (LOF)
- One-Class SVM
- Statistical methods (Z-score, Modified Z-score, IQR)
- DBSCAN-based detection
- Ensemble methods
- Visualization and performance metrics

Author: Brill Consulting
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')


class AnomalyDetector:
    """Comprehensive anomaly detection toolkit with multiple algorithms."""

    def __init__(self, random_state: int = 42):
        """
        Initialize anomaly detector.

        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        np.random.seed(random_state)
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}

    def isolation_forest(self, X: np.ndarray, contamination: float = 0.1,
                        n_estimators: int = 100) -> Dict:
        """
        Detect anomalies using Isolation Forest.

        Args:
            X: Input data (n_samples, n_features)
            contamination: Expected proportion of outliers
            n_estimators: Number of isolation trees

        Returns:
            Dictionary with predictions and anomaly scores
        """
        model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=self.random_state,
            n_jobs=-1
        )

        predictions = model.fit_predict(X)
        scores = model.score_samples(X)

        # Convert predictions: -1 (anomaly) to 1, 1 (normal) to 0
        anomaly_labels = (predictions == -1).astype(int)

        self.models['isolation_forest'] = model

        return {
            'anomaly_labels': anomaly_labels,
            'anomaly_scores': -scores,  # Negative for consistency (higher = more anomalous)
            'n_anomalies': np.sum(anomaly_labels),
            'anomaly_indices': np.where(anomaly_labels == 1)[0],
            'method': 'Isolation Forest'
        }

    def local_outlier_factor(self, X: np.ndarray, contamination: float = 0.1,
                            n_neighbors: int = 20) -> Dict:
        """
        Detect anomalies using Local Outlier Factor.

        Args:
            X: Input data (n_samples, n_features)
            contamination: Expected proportion of outliers
            n_neighbors: Number of neighbors to consider

        Returns:
            Dictionary with predictions and anomaly scores
        """
        model = LocalOutlierFactor(
            contamination=contamination,
            n_neighbors=n_neighbors,
            novelty=False,
            n_jobs=-1
        )

        predictions = model.fit_predict(X)
        scores = model.negative_outlier_factor_

        # Convert predictions: -1 (anomaly) to 1, 1 (normal) to 0
        anomaly_labels = (predictions == -1).astype(int)

        self.models['lof'] = model

        return {
            'anomaly_labels': anomaly_labels,
            'anomaly_scores': -scores,  # Negative for consistency
            'n_anomalies': np.sum(anomaly_labels),
            'anomaly_indices': np.where(anomaly_labels == 1)[0],
            'method': 'Local Outlier Factor'
        }

    def one_class_svm(self, X: np.ndarray, nu: float = 0.1,
                     kernel: str = 'rbf', gamma: str = 'auto') -> Dict:
        """
        Detect anomalies using One-Class SVM.

        Args:
            X: Input data (n_samples, n_features)
            nu: Upper bound on fraction of outliers
            kernel: Kernel type ('rbf', 'linear', 'poly', 'sigmoid')
            gamma: Kernel coefficient

        Returns:
            Dictionary with predictions and anomaly scores
        """
        model = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)

        predictions = model.fit_predict(X)
        scores = model.score_samples(X)

        # Convert predictions: -1 (anomaly) to 1, 1 (normal) to 0
        anomaly_labels = (predictions == -1).astype(int)

        self.models['one_class_svm'] = model

        return {
            'anomaly_labels': anomaly_labels,
            'anomaly_scores': -scores,
            'n_anomalies': np.sum(anomaly_labels),
            'anomaly_indices': np.where(anomaly_labels == 1)[0],
            'method': 'One-Class SVM'
        }

    def zscore_detection(self, X: np.ndarray, threshold: float = 3.0) -> Dict:
        """
        Detect anomalies using Z-score method.

        Args:
            X: Input data (n_samples, n_features)
            threshold: Z-score threshold for anomaly detection

        Returns:
            Dictionary with predictions and anomaly scores
        """
        # Calculate z-scores
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        z_scores = np.abs((X - mean) / (std + 1e-10))

        # Maximum z-score across features
        max_z_scores = np.max(z_scores, axis=1)
        anomaly_labels = (max_z_scores > threshold).astype(int)

        return {
            'anomaly_labels': anomaly_labels,
            'anomaly_scores': max_z_scores,
            'n_anomalies': np.sum(anomaly_labels),
            'anomaly_indices': np.where(anomaly_labels == 1)[0],
            'method': 'Z-Score'
        }

    def modified_zscore_detection(self, X: np.ndarray, threshold: float = 3.5) -> Dict:
        """
        Detect anomalies using Modified Z-score method (more robust).

        Args:
            X: Input data (n_samples, n_features)
            threshold: Modified Z-score threshold

        Returns:
            Dictionary with predictions and anomaly scores
        """
        # Use median and MAD (Median Absolute Deviation)
        median = np.median(X, axis=0)
        mad = np.median(np.abs(X - median), axis=0)

        # Modified z-score
        modified_z_scores = 0.6745 * np.abs((X - median) / (mad + 1e-10))

        # Maximum modified z-score across features
        max_modified_z = np.max(modified_z_scores, axis=1)
        anomaly_labels = (max_modified_z > threshold).astype(int)

        return {
            'anomaly_labels': anomaly_labels,
            'anomaly_scores': max_modified_z,
            'n_anomalies': np.sum(anomaly_labels),
            'anomaly_indices': np.where(anomaly_labels == 1)[0],
            'method': 'Modified Z-Score'
        }

    def iqr_detection(self, X: np.ndarray, factor: float = 1.5) -> Dict:
        """
        Detect anomalies using Interquartile Range (IQR) method.

        Args:
            X: Input data (n_samples, n_features)
            factor: IQR multiplier for outlier bounds

        Returns:
            Dictionary with predictions and anomaly scores
        """
        Q1 = np.percentile(X, 25, axis=0)
        Q3 = np.percentile(X, 75, axis=0)
        IQR = Q3 - Q1

        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR

        # Check if any feature is outside bounds
        outliers = np.logical_or(X < lower_bound, X > upper_bound)
        anomaly_labels = np.any(outliers, axis=1).astype(int)

        # Anomaly score: maximum deviation from bounds
        lower_dev = np.maximum(0, lower_bound - X)
        upper_dev = np.maximum(0, X - upper_bound)
        anomaly_scores = np.max(lower_dev + upper_dev, axis=1)

        return {
            'anomaly_labels': anomaly_labels,
            'anomaly_scores': anomaly_scores,
            'n_anomalies': np.sum(anomaly_labels),
            'anomaly_indices': np.where(anomaly_labels == 1)[0],
            'method': 'IQR'
        }

    def dbscan_detection(self, X: np.ndarray, eps: float = 0.5,
                        min_samples: int = 5) -> Dict:
        """
        Detect anomalies using DBSCAN clustering.

        Args:
            X: Input data (n_samples, n_features)
            eps: Maximum distance for neighborhood
            min_samples: Minimum samples in neighborhood

        Returns:
            Dictionary with predictions and cluster labels
        """
        model = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
        cluster_labels = model.fit_predict(X)

        # Points labeled as -1 are anomalies
        anomaly_labels = (cluster_labels == -1).astype(int)

        # Anomaly score: distance to nearest core point
        anomaly_scores = np.zeros(len(X))
        for i in range(len(X)):
            if cluster_labels[i] == -1:
                # Find distance to nearest clustered point
                clustered_points = X[cluster_labels != -1]
                if len(clustered_points) > 0:
                    distances = np.linalg.norm(clustered_points - X[i], axis=1)
                    anomaly_scores[i] = np.min(distances)
                else:
                    anomaly_scores[i] = 1.0

        self.models['dbscan'] = model

        return {
            'anomaly_labels': anomaly_labels,
            'anomaly_scores': anomaly_scores,
            'cluster_labels': cluster_labels,
            'n_anomalies': np.sum(anomaly_labels),
            'anomaly_indices': np.where(anomaly_labels == 1)[0],
            'n_clusters': len(np.unique(cluster_labels[cluster_labels != -1])),
            'method': 'DBSCAN'
        }

    def ensemble_detection(self, X: np.ndarray, methods: Optional[List[str]] = None,
                          voting: str = 'majority', contamination: float = 0.1) -> Dict:
        """
        Ensemble anomaly detection combining multiple methods.

        Args:
            X: Input data (n_samples, n_features)
            methods: List of methods to use (None = use all)
            voting: 'majority' or 'unanimous'
            contamination: Expected proportion of outliers

        Returns:
            Dictionary with ensemble predictions
        """
        if methods is None:
            methods = ['isolation_forest', 'lof', 'one_class_svm', 'modified_zscore']

        predictions = []
        scores = []

        for method in methods:
            if method == 'isolation_forest':
                result = self.isolation_forest(X, contamination=contamination)
            elif method == 'lof':
                result = self.local_outlier_factor(X, contamination=contamination)
            elif method == 'one_class_svm':
                result = self.one_class_svm(X, nu=contamination)
            elif method == 'zscore':
                result = self.zscore_detection(X)
            elif method == 'modified_zscore':
                result = self.modified_zscore_detection(X)
            elif method == 'iqr':
                result = self.iqr_detection(X)
            else:
                raise ValueError(f"Unknown method: {method}")

            predictions.append(result['anomaly_labels'])
            # Normalize scores to [0, 1]
            normalized_scores = (result['anomaly_scores'] - result['anomaly_scores'].min()) / \
                              (result['anomaly_scores'].max() - result['anomaly_scores'].min() + 1e-10)
            scores.append(normalized_scores)

        predictions = np.array(predictions)
        scores = np.array(scores)

        # Ensemble voting
        if voting == 'majority':
            # Majority vote
            anomaly_labels = (np.sum(predictions, axis=0) > len(methods) / 2).astype(int)
        elif voting == 'unanimous':
            # All methods must agree
            anomaly_labels = (np.sum(predictions, axis=0) == len(methods)).astype(int)
        else:
            raise ValueError(f"Unknown voting method: {voting}")

        # Average scores
        ensemble_scores = np.mean(scores, axis=0)

        return {
            'anomaly_labels': anomaly_labels,
            'anomaly_scores': ensemble_scores,
            'individual_predictions': predictions,
            'individual_scores': scores,
            'n_anomalies': np.sum(anomaly_labels),
            'anomaly_indices': np.where(anomaly_labels == 1)[0],
            'methods_used': methods,
            'voting': voting,
            'method': f'Ensemble ({voting})'
        }

    def evaluate_performance(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """
        Evaluate anomaly detection performance with labeled data.

        Args:
            y_true: True anomaly labels (1 = anomaly, 0 = normal)
            y_pred: Predicted anomaly labels

        Returns:
            Dictionary with performance metrics
        """
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0

        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'specificity': specificity,
            'accuracy': accuracy,
            'true_positives': int(tp),
            'false_positives': int(fp),
            'true_negatives': int(tn),
            'false_negatives': int(fn),
            'confusion_matrix': cm
        }

    def visualize_anomalies_2d(self, X: np.ndarray, anomaly_labels: np.ndarray,
                               feature_names: Optional[List[str]] = None,
                               title: str = "Anomaly Detection Results") -> plt.Figure:
        """
        Visualize anomalies in 2D space.

        Args:
            X: Input data (n_samples, 2)
            anomaly_labels: Binary labels (1 = anomaly, 0 = normal)
            feature_names: Names of features
            title: Plot title

        Returns:
            Matplotlib figure
        """
        if X.shape[1] != 2:
            raise ValueError("Data must be 2-dimensional for 2D visualization")

        fig, ax = plt.subplots(figsize=(10, 8))

        if feature_names is None:
            feature_names = ['Feature 1', 'Feature 2']

        # Plot normal points
        normal_mask = anomaly_labels == 0
        ax.scatter(X[normal_mask, 0], X[normal_mask, 1],
                  c='blue', alpha=0.6, s=50, label='Normal', edgecolors='k')

        # Plot anomalies
        anomaly_mask = anomaly_labels == 1
        ax.scatter(X[anomaly_mask, 0], X[anomaly_mask, 1],
                  c='red', alpha=0.8, s=100, label='Anomaly',
                  marker='X', edgecolors='darkred', linewidths=2)

        ax.set_xlabel(feature_names[0], fontsize=12)
        ax.set_ylabel(feature_names[1], fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)

        plt.tight_layout()
        return fig

    def visualize_anomaly_scores(self, anomaly_scores: np.ndarray,
                                 anomaly_labels: np.ndarray,
                                 method_name: str = "Anomaly Detection") -> plt.Figure:
        """
        Visualize distribution of anomaly scores.

        Args:
            anomaly_scores: Anomaly scores for each sample
            anomaly_labels: Binary labels (1 = anomaly, 0 = normal)
            method_name: Name of detection method

        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Score distribution
        axes[0].hist(anomaly_scores[anomaly_labels == 0], bins=50, alpha=0.6,
                    label='Normal', color='blue', edgecolor='black')
        axes[0].hist(anomaly_scores[anomaly_labels == 1], bins=50, alpha=0.6,
                    label='Anomaly', color='red', edgecolor='black')
        axes[0].set_xlabel('Anomaly Score', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].set_title(f'{method_name}: Score Distribution', fontsize=13, fontweight='bold')
        axes[0].legend(fontsize=11)
        axes[0].grid(alpha=0.3)

        # Score ordered plot
        sorted_scores = np.sort(anomaly_scores)
        axes[1].plot(sorted_scores, linewidth=2, color='darkblue')
        axes[1].axhline(y=np.percentile(anomaly_scores, 90), color='orange',
                       linestyle='--', label='90th percentile', linewidth=2)
        axes[1].axhline(y=np.percentile(anomaly_scores, 95), color='red',
                       linestyle='--', label='95th percentile', linewidth=2)
        axes[1].set_xlabel('Sample Index (sorted)', fontsize=12)
        axes[1].set_ylabel('Anomaly Score', fontsize=12)
        axes[1].set_title('Sorted Anomaly Scores', fontsize=13, fontweight='bold')
        axes[1].legend(fontsize=11)
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        return fig

    def visualize_method_comparison(self, X: np.ndarray, results: Dict[str, Dict]) -> plt.Figure:
        """
        Compare multiple anomaly detection methods.

        Args:
            X: Input data (n_samples, 2)
            results: Dictionary of results from different methods

        Returns:
            Matplotlib figure
        """
        n_methods = len(results)
        fig, axes = plt.subplots(2, (n_methods + 1) // 2, figsize=(15, 10))
        axes = axes.flatten() if n_methods > 1 else [axes]

        for idx, (method_name, result) in enumerate(results.items()):
            ax = axes[idx]

            # Plot normal points
            normal_mask = result['anomaly_labels'] == 0
            ax.scatter(X[normal_mask, 0], X[normal_mask, 1],
                      c='blue', alpha=0.5, s=30, label='Normal')

            # Plot anomalies
            anomaly_mask = result['anomaly_labels'] == 1
            ax.scatter(X[anomaly_mask, 0], X[anomaly_mask, 1],
                      c='red', alpha=0.8, s=80, label='Anomaly', marker='X')

            ax.set_title(f"{method_name}\n({result['n_anomalies']} anomalies)",
                        fontsize=11, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(alpha=0.3)

        # Hide unused subplots
        for idx in range(n_methods, len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()
        return fig


def demo():
    """Demonstrate anomaly detection toolkit."""
    np.random.seed(42)

    print("Anomaly Detection Toolkit Demo")
    print("=" * 70)

    # Generate synthetic data with anomalies
    print("\nGenerating synthetic data with anomalies...")
    n_normal = 300
    n_anomalies = 30

    # Normal data: two Gaussian clusters
    normal_1 = np.random.randn(n_normal // 2, 2) * 0.5 + np.array([0, 0])
    normal_2 = np.random.randn(n_normal // 2, 2) * 0.5 + np.array([3, 3])
    normal_data = np.vstack([normal_1, normal_2])

    # Anomalies: scattered random points
    anomaly_data = np.random.uniform(-3, 6, (n_anomalies, 2))

    # Combine
    X = np.vstack([normal_data, anomaly_data])
    y_true = np.hstack([np.zeros(n_normal), np.ones(n_anomalies)])

    print(f"Total samples: {len(X)}")
    print(f"Normal samples: {n_normal}")
    print(f"Anomalies: {n_anomalies}")

    detector = AnomalyDetector(random_state=42)

    # 1. Isolation Forest
    print("\n1. Isolation Forest")
    print("-" * 70)
    result_if = detector.isolation_forest(X, contamination=0.1)
    print(f"Detected anomalies: {result_if['n_anomalies']}")
    metrics_if = detector.evaluate_performance(y_true, result_if['anomaly_labels'])
    print(f"Precision: {metrics_if['precision']:.3f}")
    print(f"Recall: {metrics_if['recall']:.3f}")
    print(f"F1-Score: {metrics_if['f1_score']:.3f}")

    # 2. Local Outlier Factor
    print("\n2. Local Outlier Factor")
    print("-" * 70)
    result_lof = detector.local_outlier_factor(X, contamination=0.1)
    print(f"Detected anomalies: {result_lof['n_anomalies']}")
    metrics_lof = detector.evaluate_performance(y_true, result_lof['anomaly_labels'])
    print(f"Precision: {metrics_lof['precision']:.3f}")
    print(f"Recall: {metrics_lof['recall']:.3f}")
    print(f"F1-Score: {metrics_lof['f1_score']:.3f}")

    # 3. One-Class SVM
    print("\n3. One-Class SVM")
    print("-" * 70)
    result_svm = detector.one_class_svm(X, nu=0.1)
    print(f"Detected anomalies: {result_svm['n_anomalies']}")
    metrics_svm = detector.evaluate_performance(y_true, result_svm['anomaly_labels'])
    print(f"Precision: {metrics_svm['precision']:.3f}")
    print(f"Recall: {metrics_svm['recall']:.3f}")
    print(f"F1-Score: {metrics_svm['f1_score']:.3f}")

    # 4. Statistical Methods
    print("\n4. Statistical Methods")
    print("-" * 70)

    # Z-Score
    result_z = detector.zscore_detection(X, threshold=3.0)
    print(f"Z-Score: {result_z['n_anomalies']} anomalies")
    metrics_z = detector.evaluate_performance(y_true, result_z['anomaly_labels'])
    print(f"  F1-Score: {metrics_z['f1_score']:.3f}")

    # Modified Z-Score
    result_mz = detector.modified_zscore_detection(X, threshold=3.5)
    print(f"Modified Z-Score: {result_mz['n_anomalies']} anomalies")
    metrics_mz = detector.evaluate_performance(y_true, result_mz['anomaly_labels'])
    print(f"  F1-Score: {metrics_mz['f1_score']:.3f}")

    # IQR
    result_iqr = detector.iqr_detection(X, factor=1.5)
    print(f"IQR: {result_iqr['n_anomalies']} anomalies")
    metrics_iqr = detector.evaluate_performance(y_true, result_iqr['anomaly_labels'])
    print(f"  F1-Score: {metrics_iqr['f1_score']:.3f}")

    # 5. DBSCAN
    print("\n5. DBSCAN-based Detection")
    print("-" * 70)
    result_dbscan = detector.dbscan_detection(X, eps=0.5, min_samples=5)
    print(f"Detected anomalies: {result_dbscan['n_anomalies']}")
    print(f"Number of clusters: {result_dbscan['n_clusters']}")
    metrics_dbscan = detector.evaluate_performance(y_true, result_dbscan['anomaly_labels'])
    print(f"Precision: {metrics_dbscan['precision']:.3f}")
    print(f"Recall: {metrics_dbscan['recall']:.3f}")
    print(f"F1-Score: {metrics_dbscan['f1_score']:.3f}")

    # 6. Ensemble Detection
    print("\n6. Ensemble Anomaly Detection")
    print("-" * 70)
    result_ensemble = detector.ensemble_detection(
        X,
        methods=['isolation_forest', 'lof', 'modified_zscore'],
        voting='majority',
        contamination=0.1
    )
    print(f"Detected anomalies: {result_ensemble['n_anomalies']}")
    print(f"Methods used: {', '.join(result_ensemble['methods_used'])}")
    print(f"Voting strategy: {result_ensemble['voting']}")
    metrics_ensemble = detector.evaluate_performance(y_true, result_ensemble['anomaly_labels'])
    print(f"Precision: {metrics_ensemble['precision']:.3f}")
    print(f"Recall: {metrics_ensemble['recall']:.3f}")
    print(f"F1-Score: {metrics_ensemble['f1_score']:.3f}")

    # Visualizations
    print("\n7. Generating Visualizations")
    print("-" * 70)

    # Single method visualization
    fig1 = detector.visualize_anomalies_2d(
        X, result_ensemble['anomaly_labels'],
        feature_names=['Feature 1', 'Feature 2'],
        title='Ensemble Anomaly Detection Results'
    )
    fig1.savefig('anomaly_detection_2d.png', dpi=300, bbox_inches='tight')
    print("✓ Saved anomaly_detection_2d.png")
    plt.close()

    # Score distribution
    fig2 = detector.visualize_anomaly_scores(
        result_if['anomaly_scores'],
        result_if['anomaly_labels'],
        method_name='Isolation Forest'
    )
    fig2.savefig('anomaly_scores_distribution.png', dpi=300, bbox_inches='tight')
    print("✓ Saved anomaly_scores_distribution.png")
    plt.close()

    # Method comparison
    comparison_results = {
        'Isolation Forest': result_if,
        'LOF': result_lof,
        'One-Class SVM': result_svm,
        'Modified Z-Score': result_mz,
        'DBSCAN': result_dbscan,
        'Ensemble': result_ensemble
    }

    fig3 = detector.visualize_method_comparison(X, comparison_results)
    fig3.savefig('anomaly_methods_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved anomaly_methods_comparison.png")
    plt.close()

    # Performance comparison
    print("\n8. Performance Summary")
    print("-" * 70)
    print(f"{'Method':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 70)

    all_metrics = {
        'Isolation Forest': metrics_if,
        'LOF': metrics_lof,
        'One-Class SVM': metrics_svm,
        'Z-Score': metrics_z,
        'Modified Z-Score': metrics_mz,
        'IQR': metrics_iqr,
        'DBSCAN': metrics_dbscan,
        'Ensemble': metrics_ensemble
    }

    for method, metrics in all_metrics.items():
        print(f"{method:<20} {metrics['precision']:<12.3f} {metrics['recall']:<12.3f} {metrics['f1_score']:<12.3f}")

    print("\n" + "=" * 70)
    print("✓ Anomaly Detection Demo Complete!")
    print("=" * 70)


if __name__ == '__main__':
    demo()
