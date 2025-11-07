"""
Advanced Multi-Algorithm Anomaly Detection System v2.0
Author: BrillConsulting
Description: Production-ready anomaly detection with 8+ algorithms including Isolation Forest, One-Class SVM, and statistical methods
Version: 2.0 - Enhanced with multiple algorithms, ensemble methods, and comprehensive evaluation
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             confusion_matrix, classification_report, roc_auc_score, roc_curve)
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import argparse
import joblib
import warnings
warnings.filterwarnings('ignore')


class AnomalyDetector:
    """
    Advanced anomaly detection system with 8+ algorithms

    Features:
    - Statistical Methods: Z-Score, IQR, Modified Z-Score
    - Machine Learning: Isolation Forest, One-Class SVM, LOF, Elliptic Envelope
    - Clustering-based: DBSCAN
    - Ensemble voting for robust detection
    - Automatic contamination estimation
    - Comprehensive visualization and reporting
    """

    def __init__(self, contamination: float = 0.1, random_state: int = 42, scale_features: bool = True):
        """
        Initialize anomaly detector

        Args:
            contamination: Expected proportion of outliers (0.0 to 0.5)
            random_state: Random seed for reproducibility
            scale_features: Whether to standardize features
        """
        self.contamination = contamination
        self.random_state = random_state
        self.scale_features = scale_features
        self.scaler = StandardScaler() if scale_features else None
        self.models = {}
        self.results = {}
        self.X_original = None
        self.X_scaled = None

    def prepare_data(self, X: np.ndarray, fit: bool = True) -> np.ndarray:
        """Prepare and optionally scale data"""
        if self.scale_features:
            if fit:
                X_scaled = self.scaler.fit_transform(X)
            else:
                X_scaled = self.scaler.transform(X)
            return X_scaled
        return X

    def detect_zscore(self, X: np.ndarray, threshold: float = 3.0) -> np.ndarray:
        """
        Z-Score method for anomaly detection
        Points with |z-score| > threshold are considered anomalies
        """
        z_scores = np.abs(stats.zscore(X, axis=0))
        # Anomaly if ANY feature has z-score > threshold
        anomalies = (z_scores > threshold).any(axis=1).astype(int)
        # Convert to -1 (anomaly) and 1 (normal)
        predictions = np.where(anomalies == 1, -1, 1)

        self.models['Z-Score'] = {'threshold': threshold}
        self.results['Z-Score'] = {
            'predictions': predictions,
            'scores': z_scores.max(axis=1),  # Max z-score across features
            'n_anomalies': (predictions == -1).sum()
        }

        return predictions

    def detect_iqr(self, X: np.ndarray, multiplier: float = 1.5) -> np.ndarray:
        """
        IQR (Interquartile Range) method for anomaly detection
        Points outside [Q1 - multiplier*IQR, Q3 + multiplier*IQR] are anomalies
        """
        Q1 = np.percentile(X, 25, axis=0)
        Q3 = np.percentile(X, 75, axis=0)
        IQR = Q3 - Q1

        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR

        # Anomaly if ANY feature is outside bounds
        anomalies = ((X < lower_bound) | (X > upper_bound)).any(axis=1).astype(int)
        predictions = np.where(anomalies == 1, -1, 1)

        # Calculate outlier score (distance from bounds)
        lower_dist = np.maximum(0, lower_bound - X)
        upper_dist = np.maximum(0, X - upper_bound)
        scores = (lower_dist + upper_dist).max(axis=1)

        self.models['IQR'] = {'multiplier': multiplier, 'Q1': Q1, 'Q3': Q3}
        self.results['IQR'] = {
            'predictions': predictions,
            'scores': scores,
            'n_anomalies': (predictions == -1).sum()
        }

        return predictions

    def detect_modified_zscore(self, X: np.ndarray, threshold: float = 3.5) -> np.ndarray:
        """
        Modified Z-Score using Median Absolute Deviation (MAD)
        More robust to outliers than standard Z-Score
        """
        median = np.median(X, axis=0)
        mad = np.median(np.abs(X - median), axis=0)

        # Avoid division by zero
        mad = np.where(mad == 0, 1e-10, mad)

        modified_z_scores = 0.6745 * (X - median) / mad
        anomalies = (np.abs(modified_z_scores) > threshold).any(axis=1).astype(int)
        predictions = np.where(anomalies == 1, -1, 1)

        self.models['Modified Z-Score'] = {'threshold': threshold}
        self.results['Modified Z-Score'] = {
            'predictions': predictions,
            'scores': np.abs(modified_z_scores).max(axis=1),
            'n_anomalies': (predictions == -1).sum()
        }

        return predictions

    def detect_isolation_forest(self, X: np.ndarray) -> np.ndarray:
        """
        Isolation Forest algorithm
        Efficient for high-dimensional data, isolates anomalies using random trees
        """
        model = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            n_estimators=100
        )
        predictions = model.fit_predict(X)
        scores = -model.score_samples(X)  # Negative anomaly scores (higher = more anomalous)

        self.models['Isolation Forest'] = model
        self.results['Isolation Forest'] = {
            'predictions': predictions,
            'scores': scores,
            'n_anomalies': (predictions == -1).sum()
        }

        return predictions

    def detect_one_class_svm(self, X: np.ndarray) -> np.ndarray:
        """
        One-Class SVM
        Learns a decision boundary around normal data
        """
        model = OneClassSVM(
            kernel='rbf',
            gamma='auto',
            nu=self.contamination
        )
        predictions = model.fit_predict(X)
        scores = -model.score_samples(X)  # Distance to separating hyperplane

        self.models['One-Class SVM'] = model
        self.results['One-Class SVM'] = {
            'predictions': predictions,
            'scores': scores,
            'n_anomalies': (predictions == -1).sum()
        }

        return predictions

    def detect_lof(self, X: np.ndarray, n_neighbors: int = 20) -> np.ndarray:
        """
        Local Outlier Factor (LOF)
        Measures local density deviation compared to neighbors
        """
        model = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=self.contamination,
            novelty=False
        )
        predictions = model.fit_predict(X)
        scores = -model.negative_outlier_factor_  # LOF scores (higher = more anomalous)

        self.models['LOF'] = model
        self.results['LOF'] = {
            'predictions': predictions,
            'scores': scores,
            'n_anomalies': (predictions == -1).sum()
        }

        return predictions

    def detect_elliptic_envelope(self, X: np.ndarray) -> np.ndarray:
        """
        Elliptic Envelope (Robust Covariance)
        Assumes data comes from a Gaussian distribution
        """
        model = EllipticEnvelope(
            contamination=self.contamination,
            random_state=self.random_state
        )
        predictions = model.fit_predict(X)
        scores = -model.score_samples(X)  # Mahalanobis distance

        self.models['Elliptic Envelope'] = model
        self.results['Elliptic Envelope'] = {
            'predictions': predictions,
            'scores': scores,
            'n_anomalies': (predictions == -1).sum()
        }

        return predictions

    def detect_dbscan(self, X: np.ndarray, eps: float = 0.5, min_samples: int = 5) -> np.ndarray:
        """
        DBSCAN-based anomaly detection
        Points not assigned to any cluster are considered anomalies
        """
        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(X)

        # Label -1 indicates noise/anomaly
        predictions = np.where(labels == -1, -1, 1)

        # Calculate outlier scores based on distance to nearest core point
        scores = np.zeros(len(X))
        anomaly_mask = labels == -1
        scores[anomaly_mask] = 1.0  # Simple binary score

        self.models['DBSCAN'] = model
        self.results['DBSCAN'] = {
            'predictions': predictions,
            'scores': scores,
            'n_anomalies': (predictions == -1).sum()
        }

        return predictions

    def ensemble_voting(self, threshold: float = 0.5) -> np.ndarray:
        """
        Ensemble voting: combine predictions from all models
        A point is anomalous if >= threshold * n_models flag it as anomaly
        """
        if not self.results:
            raise ValueError("No models have been trained yet")

        all_predictions = np.array([res['predictions'] for res in self.results.values()])
        # Convert to binary: -1 -> 1 (anomaly), 1 -> 0 (normal)
        anomaly_votes = (all_predictions == -1).astype(int)

        # Average votes
        vote_ratio = anomaly_votes.mean(axis=0)
        ensemble_predictions = np.where(vote_ratio >= threshold, -1, 1)

        self.results['Ensemble Voting'] = {
            'predictions': ensemble_predictions,
            'scores': vote_ratio,
            'n_anomalies': (ensemble_predictions == -1).sum()
        }

        return ensemble_predictions

    def fit_all(self, X: np.ndarray):
        """Fit all anomaly detection algorithms"""
        self.X_original = X
        X_scaled = self.prepare_data(X, fit=True)
        self.X_scaled = X_scaled

        print(f"🔧 Training {8} anomaly detection algorithms...")
        print(f"   Contamination rate: {self.contamination:.1%}")
        print("=" * 70)

        # Statistical methods (on original scale for interpretability)
        print("  [1/8] Z-Score...")
        self.detect_zscore(X)

        print("  [2/8] IQR...")
        self.detect_iqr(X)

        print("  [3/8] Modified Z-Score...")
        self.detect_modified_zscore(X)

        # Machine learning methods (on scaled data)
        print("  [4/8] Isolation Forest...")
        self.detect_isolation_forest(X_scaled)

        print("  [5/8] One-Class SVM...")
        self.detect_one_class_svm(X_scaled)

        print("  [6/8] Local Outlier Factor (LOF)...")
        self.detect_lof(X_scaled)

        print("  [7/8] Elliptic Envelope...")
        self.detect_elliptic_envelope(X_scaled)

        print("  [8/8] DBSCAN...")
        self.detect_dbscan(X_scaled)

        # Ensemble
        print("\n  [Ensemble] Combining predictions...")
        self.ensemble_voting(threshold=0.5)

        print("=" * 70)
        print(f"✅ All {len(self.models)} algorithms trained successfully!\n")

    def evaluate(self, y_true: Optional[np.ndarray] = None) -> pd.DataFrame:
        """
        Evaluate all models
        If y_true is provided, compute metrics; otherwise just show counts
        """
        if not self.results:
            raise ValueError("No models have been trained yet")

        evaluation = []

        for name, results in self.results.items():
            predictions = results['predictions']
            n_anomalies = results['n_anomalies']
            anomaly_rate = n_anomalies / len(predictions)

            row = {
                'Algorithm': name,
                'Anomalies Detected': n_anomalies,
                'Anomaly Rate': f"{anomaly_rate:.1%}"
            }

            # If ground truth is available
            if y_true is not None:
                # Convert predictions: -1 (anomaly) -> 1, 1 (normal) -> 0
                y_pred = (predictions == -1).astype(int)
                y_true_binary = (y_true == -1).astype(int) if -1 in y_true else y_true

                precision = precision_score(y_true_binary, y_pred, zero_division=0)
                recall = recall_score(y_true_binary, y_pred, zero_division=0)
                f1 = f1_score(y_true_binary, y_pred, zero_division=0)

                row.update({
                    'Precision': precision,
                    'Recall': recall,
                    'F1 Score': f1
                })

            evaluation.append(row)

        df = pd.DataFrame(evaluation)

        if y_true is not None:
            df = df.sort_values('F1 Score', ascending=False)

        return df

    def plot_anomalies_pca(self, save_path: Optional[str] = None):
        """Visualize anomalies using PCA (2D projection)"""
        if self.X_scaled is None:
            raise ValueError("No data has been fitted yet")

        # Apply PCA for 2D visualization
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(self.X_scaled)

        # Plot top 8 algorithms
        top_algorithms = list(self.results.keys())[:8]
        n_plots = len(top_algorithms)
        n_rows = (n_plots + 2) // 3

        fig, axes = plt.subplots(n_rows, 3, figsize=(18, n_rows * 5))
        axes = axes.ravel() if n_plots > 1 else [axes]

        for idx, name in enumerate(top_algorithms):
            predictions = self.results[name]['predictions']

            # Separate normal and anomalies
            normal_mask = predictions == 1
            anomaly_mask = predictions == -1

            axes[idx].scatter(X_pca[normal_mask, 0], X_pca[normal_mask, 1],
                            c='blue', alpha=0.5, s=20, label='Normal')
            axes[idx].scatter(X_pca[anomaly_mask, 0], X_pca[anomaly_mask, 1],
                            c='red', alpha=0.8, s=50, marker='x', label='Anomaly')

            n_anomalies = self.results[name]['n_anomalies']
            axes[idx].set_title(f'{name}\nAnomalies: {n_anomalies} ({n_anomalies/len(predictions):.1%})')
            axes[idx].set_xlabel('PC1')
            axes[idx].set_ylabel('PC2')
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)

        # Hide unused subplots
        for idx in range(len(top_algorithms), len(axes)):
            axes[idx].axis('off')

        plt.suptitle('Anomaly Detection - PCA Visualization', fontsize=16, y=1.00)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 PCA visualization saved to {save_path}")

        plt.show()

    def plot_anomaly_scores(self, save_path: Optional[str] = None):
        """Plot anomaly score distributions"""
        top_algorithms = [name for name in self.results.keys() if name != 'Ensemble Voting'][:6]

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.ravel()

        for idx, name in enumerate(top_algorithms):
            scores = self.results[name]['scores']
            predictions = self.results[name]['predictions']

            # Separate scores for normal and anomalies
            normal_scores = scores[predictions == 1]
            anomaly_scores = scores[predictions == -1]

            axes[idx].hist(normal_scores, bins=50, alpha=0.6, color='blue', label='Normal')
            axes[idx].hist(anomaly_scores, bins=50, alpha=0.6, color='red', label='Anomaly')
            axes[idx].set_xlabel('Anomaly Score')
            axes[idx].set_ylabel('Frequency')
            axes[idx].set_title(f'{name}')
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)

        plt.suptitle('Anomaly Score Distributions', fontsize=16)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Score distributions saved to {save_path}")

        plt.show()

    def plot_algorithm_comparison(self, save_path: Optional[str] = None):
        """Compare number of anomalies detected by each algorithm"""
        algorithms = list(self.results.keys())
        n_anomalies = [self.results[name]['n_anomalies'] for name in algorithms]

        plt.figure(figsize=(14, 6))
        bars = plt.bar(range(len(algorithms)), n_anomalies, color='steelblue', alpha=0.7)

        # Color ensemble bar differently
        if 'Ensemble Voting' in algorithms:
            ensemble_idx = algorithms.index('Ensemble Voting')
            bars[ensemble_idx].set_color('orange')

        plt.xticks(range(len(algorithms)), algorithms, rotation=45, ha='right')
        plt.ylabel('Number of Anomalies Detected')
        plt.title('Algorithm Comparison - Anomaly Detection Count')
        plt.grid(True, alpha=0.3, axis='y')

        # Add value labels
        for idx, (algo, count) in enumerate(zip(algorithms, n_anomalies)):
            plt.text(idx, count, f'{count}', ha='center', va='bottom')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Algorithm comparison saved to {save_path}")

        plt.show()

    def get_anomalies(self, algorithm: str = 'Ensemble Voting') -> Tuple[np.ndarray, np.ndarray]:
        """
        Get anomaly indices and scores for a specific algorithm

        Returns:
            anomaly_indices: Indices of detected anomalies
            anomaly_scores: Corresponding anomaly scores
        """
        if algorithm not in self.results:
            raise ValueError(f"Algorithm '{algorithm}' not found. Available: {list(self.results.keys())}")

        predictions = self.results[algorithm]['predictions']
        scores = self.results[algorithm]['scores']

        anomaly_indices = np.where(predictions == -1)[0]
        anomaly_scores = scores[anomaly_indices]

        return anomaly_indices, anomaly_scores

    def save_anomalies(self, output_path: str, algorithm: str = 'Ensemble Voting'):
        """Save detected anomalies to CSV"""
        anomaly_indices, anomaly_scores = self.get_anomalies(algorithm)

        # Create DataFrame with original data and anomaly scores
        df = pd.DataFrame(self.X_original[anomaly_indices])
        df['anomaly_score'] = anomaly_scores
        df['anomaly_index'] = anomaly_indices

        df.to_csv(output_path, index=False)
        print(f"💾 {len(df)} anomalies saved to {output_path}")

    def save_model(self, filepath: str, algorithm: str = 'Isolation Forest'):
        """Save a trained model"""
        if algorithm not in self.models:
            raise ValueError(f"Algorithm '{algorithm}' not found")

        model_data = {
            'model': self.models[algorithm],
            'scaler': self.scaler,
            'contamination': self.contamination
        }

        joblib.dump(model_data, filepath)
        print(f"💾 {algorithm} model saved to {filepath}")


def main():
    parser = argparse.ArgumentParser(description='Anomaly Detection v2.0')
    parser.add_argument('--data', type=str, required=True, help='Path to CSV data')
    parser.add_argument('--contamination', type=float, default=0.1, help='Expected contamination rate (0.0-0.5)')
    parser.add_argument('--output-anomalies', type=str, help='Save detected anomalies to CSV')
    parser.add_argument('--output-viz', type=str, help='Save PCA visualization')
    parser.add_argument('--save-model', type=str, help='Save best model')
    parser.add_argument('--labels', type=str, help='Path to true labels (for evaluation)')

    args = parser.parse_args()

    # Load data
    print(f"📂 Loading data from {args.data}...")
    df = pd.read_csv(args.data)

    # Separate features (assume all numeric columns are features)
    X = df.select_dtypes(include=[np.number]).values

    print(f"📊 Dataset: {X.shape[0]} samples, {X.shape[1]} features\n")

    # Initialize detector
    detector = AnomalyDetector(contamination=args.contamination, scale_features=True)

    # Fit all algorithms
    detector.fit_all(X)

    # Load labels if provided
    y_true = None
    if args.labels:
        labels_df = pd.read_csv(args.labels)
        y_true = labels_df.values.ravel()

    # Evaluate
    print("📊 Algorithm Evaluation:")
    print("=" * 90)
    evaluation_df = detector.evaluate(y_true)
    print(evaluation_df.to_string(index=False))
    print("=" * 90)

    # Visualizations
    print("\n📊 Generating visualizations...")
    detector.plot_algorithm_comparison()
    detector.plot_anomalies_pca(save_path=args.output_viz)
    detector.plot_anomaly_scores()

    # Save anomalies
    if args.output_anomalies:
        detector.save_anomalies(args.output_anomalies, algorithm='Ensemble Voting')

    # Save model
    if args.save_model:
        detector.save_model(args.save_model, algorithm='Isolation Forest')

    print("\n✅ Anomaly detection completed successfully!")


if __name__ == "__main__":
    main()
