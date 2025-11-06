"""
Cluster Analysis Toolkit
=========================

Comprehensive clustering algorithms and validation:
- K-Means clustering with multiple initializations
- Hierarchical clustering (agglomerative)
- DBSCAN clustering
- Gaussian Mixture Models
- Optimal cluster determination (Elbow, Silhouette, Davies-Bouldin)
- Cluster validation metrics
- Dendrogram and cluster visualizations

Author: Brill Consulting
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, silhouette_samples, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')


class ClusterAnalyzer:
    """Comprehensive clustering and cluster analysis toolkit."""

    def __init__(self, random_state: int = 42):
        """
        Initialize cluster analyzer.

        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        np.random.seed(random_state)
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}

    def kmeans_clustering(self, X: np.ndarray, n_clusters: int = 3,
                         init: str = 'k-means++', n_init: int = 10,
                         max_iter: int = 300) -> Dict:
        """
        Perform K-Means clustering.

        Args:
            X: Input data (n_samples, n_features)
            n_clusters: Number of clusters
            init: Initialization method ('k-means++', 'random')
            n_init: Number of initializations
            max_iter: Maximum iterations

        Returns:
            Dictionary with cluster labels and statistics
        """
        model = KMeans(
            n_clusters=n_clusters,
            init=init,
            n_init=n_init,
            max_iter=max_iter,
            random_state=self.random_state
        )

        labels = model.fit_predict(X)
        centers = model.cluster_centers_
        inertia = model.inertia_

        # Calculate cluster sizes
        unique, counts = np.unique(labels, return_counts=True)
        cluster_sizes = dict(zip(unique, counts))

        # Calculate within-cluster sum of squares
        wcss = np.sum([np.sum((X[labels == i] - centers[i])**2)
                       for i in range(n_clusters)])

        self.models['kmeans'] = model

        return {
            'labels': labels,
            'centers': centers,
            'inertia': inertia,
            'wcss': wcss,
            'n_clusters': n_clusters,
            'cluster_sizes': cluster_sizes,
            'n_iter': model.n_iter_,
            'method': 'K-Means'
        }

    def hierarchical_clustering(self, X: np.ndarray, n_clusters: int = 3,
                                linkage_method: str = 'ward',
                                distance_metric: str = 'euclidean') -> Dict:
        """
        Perform hierarchical agglomerative clustering.

        Args:
            X: Input data (n_samples, n_features)
            n_clusters: Number of clusters
            linkage_method: Linkage method ('ward', 'complete', 'average', 'single')
            distance_metric: Distance metric ('euclidean', 'manhattan', 'cosine')

        Returns:
            Dictionary with cluster labels and linkage matrix
        """
        model = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=linkage_method,
            metric=distance_metric if linkage_method != 'ward' else 'euclidean'
        )

        labels = model.fit_predict(X)

        # Calculate linkage matrix for dendrogram
        if linkage_method == 'ward':
            Z = linkage(X, method=linkage_method)
        else:
            Z = linkage(X, method=linkage_method, metric=distance_metric)

        # Cluster sizes
        unique, counts = np.unique(labels, return_counts=True)
        cluster_sizes = dict(zip(unique, counts))

        self.models['hierarchical'] = model
        self.results['linkage_matrix'] = Z

        return {
            'labels': labels,
            'linkage_matrix': Z,
            'n_clusters': n_clusters,
            'cluster_sizes': cluster_sizes,
            'linkage_method': linkage_method,
            'distance_metric': distance_metric,
            'method': 'Hierarchical'
        }

    def dbscan_clustering(self, X: np.ndarray, eps: float = 0.5,
                         min_samples: int = 5) -> Dict:
        """
        Perform DBSCAN clustering.

        Args:
            X: Input data (n_samples, n_features)
            eps: Maximum distance for neighborhood
            min_samples: Minimum samples in neighborhood

        Returns:
            Dictionary with cluster labels and statistics
        """
        model = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
        labels = model.fit_predict(X)

        # Number of clusters (excluding noise points labeled as -1)
        n_clusters = len(np.unique(labels[labels != -1]))
        n_noise = np.sum(labels == -1)

        # Cluster sizes (excluding noise)
        unique, counts = np.unique(labels[labels != -1], return_counts=True)
        cluster_sizes = dict(zip(unique, counts))

        # Core samples
        core_samples_mask = np.zeros_like(labels, dtype=bool)
        core_samples_mask[model.core_sample_indices_] = True

        self.models['dbscan'] = model

        return {
            'labels': labels,
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'cluster_sizes': cluster_sizes,
            'core_samples_mask': core_samples_mask,
            'eps': eps,
            'min_samples': min_samples,
            'method': 'DBSCAN'
        }

    def gaussian_mixture(self, X: np.ndarray, n_components: int = 3,
                        covariance_type: str = 'full', n_init: int = 10) -> Dict:
        """
        Perform Gaussian Mixture Model clustering.

        Args:
            X: Input data (n_samples, n_features)
            n_components: Number of mixture components
            covariance_type: Type of covariance ('full', 'tied', 'diag', 'spherical')
            n_init: Number of initializations

        Returns:
            Dictionary with cluster labels and statistics
        """
        model = GaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            n_init=n_init,
            random_state=self.random_state
        )

        model.fit(X)
        labels = model.predict(X)
        probabilities = model.predict_proba(X)

        # Cluster sizes
        unique, counts = np.unique(labels, return_counts=True)
        cluster_sizes = dict(zip(unique, counts))

        # BIC and AIC
        bic = model.bic(X)
        aic = model.aic(X)

        self.models['gmm'] = model

        return {
            'labels': labels,
            'probabilities': probabilities,
            'means': model.means_,
            'covariances': model.covariances_,
            'weights': model.weights_,
            'n_components': n_components,
            'cluster_sizes': cluster_sizes,
            'bic': bic,
            'aic': aic,
            'converged': model.converged_,
            'n_iter': model.n_iter_,
            'method': 'Gaussian Mixture'
        }

    def elbow_method(self, X: np.ndarray, max_clusters: int = 10,
                    min_clusters: int = 2) -> Dict:
        """
        Determine optimal number of clusters using Elbow method.

        Args:
            X: Input data (n_samples, n_features)
            max_clusters: Maximum number of clusters to try
            min_clusters: Minimum number of clusters to try

        Returns:
            Dictionary with inertias and suggested number of clusters
        """
        inertias = []
        K_range = range(min_clusters, max_clusters + 1)

        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            kmeans.fit(X)
            inertias.append(kmeans.inertia_)

        # Calculate rate of change (second derivative)
        if len(inertias) >= 3:
            # Simple elbow detection: find maximum rate of decrease
            deltas = np.diff(inertias)
            delta_deltas = np.diff(deltas)
            # Elbow is where the second derivative is maximum
            elbow_idx = np.argmax(delta_deltas) + min_clusters
        else:
            elbow_idx = min_clusters

        return {
            'k_range': list(K_range),
            'inertias': inertias,
            'suggested_k': elbow_idx,
            'method': 'Elbow Method'
        }

    def silhouette_analysis(self, X: np.ndarray, max_clusters: int = 10,
                           min_clusters: int = 2) -> Dict:
        """
        Determine optimal clusters using Silhouette analysis.

        Args:
            X: Input data (n_samples, n_features)
            max_clusters: Maximum number of clusters to try
            min_clusters: Minimum number of clusters to try

        Returns:
            Dictionary with silhouette scores
        """
        silhouette_scores = []
        K_range = range(min_clusters, max_clusters + 1)

        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            labels = kmeans.fit_predict(X)
            score = silhouette_score(X, labels)
            silhouette_scores.append(score)

        # Best number of clusters
        best_k = K_range[np.argmax(silhouette_scores)]

        return {
            'k_range': list(K_range),
            'silhouette_scores': silhouette_scores,
            'best_k': best_k,
            'best_score': max(silhouette_scores),
            'method': 'Silhouette Analysis'
        }

    def davies_bouldin_analysis(self, X: np.ndarray, max_clusters: int = 10,
                                min_clusters: int = 2) -> Dict:
        """
        Determine optimal clusters using Davies-Bouldin index.

        Args:
            X: Input data (n_samples, n_features)
            max_clusters: Maximum number of clusters to try
            min_clusters: Minimum number of clusters to try

        Returns:
            Dictionary with Davies-Bouldin scores (lower is better)
        """
        db_scores = []
        K_range = range(min_clusters, max_clusters + 1)

        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            labels = kmeans.fit_predict(X)
            score = davies_bouldin_score(X, labels)
            db_scores.append(score)

        # Best number of clusters (minimum DB index)
        best_k = K_range[np.argmin(db_scores)]

        return {
            'k_range': list(K_range),
            'davies_bouldin_scores': db_scores,
            'best_k': best_k,
            'best_score': min(db_scores),
            'method': 'Davies-Bouldin Index'
        }

    def calinski_harabasz_analysis(self, X: np.ndarray, max_clusters: int = 10,
                                   min_clusters: int = 2) -> Dict:
        """
        Determine optimal clusters using Calinski-Harabasz index.

        Args:
            X: Input data (n_samples, n_features)
            max_clusters: Maximum number of clusters to try
            min_clusters: Minimum number of clusters to try

        Returns:
            Dictionary with Calinski-Harabasz scores (higher is better)
        """
        ch_scores = []
        K_range = range(min_clusters, max_clusters + 1)

        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            labels = kmeans.fit_predict(X)
            score = calinski_harabasz_score(X, labels)
            ch_scores.append(score)

        # Best number of clusters (maximum CH index)
        best_k = K_range[np.argmax(ch_scores)]

        return {
            'k_range': list(K_range),
            'calinski_harabasz_scores': ch_scores,
            'best_k': best_k,
            'best_score': max(ch_scores),
            'method': 'Calinski-Harabasz Index'
        }

    def evaluate_clustering(self, X: np.ndarray, labels: np.ndarray) -> Dict:
        """
        Evaluate clustering quality with multiple metrics.

        Args:
            X: Input data (n_samples, n_features)
            labels: Cluster labels

        Returns:
            Dictionary with evaluation metrics
        """
        # Remove noise points for metrics calculation if present
        mask = labels != -1
        X_clean = X[mask]
        labels_clean = labels[mask]

        if len(np.unique(labels_clean)) < 2:
            return {
                'silhouette_score': None,
                'davies_bouldin_score': None,
                'calinski_harabasz_score': None,
                'n_clusters': len(np.unique(labels_clean)),
                'error': 'Insufficient clusters for evaluation'
            }

        silhouette = silhouette_score(X_clean, labels_clean)
        davies_bouldin = davies_bouldin_score(X_clean, labels_clean)
        calinski_harabasz = calinski_harabasz_score(X_clean, labels_clean)

        return {
            'silhouette_score': silhouette,
            'davies_bouldin_score': davies_bouldin,
            'calinski_harabasz_score': calinski_harabasz,
            'n_clusters': len(np.unique(labels_clean))
        }

    def visualize_clusters_2d(self, X: np.ndarray, labels: np.ndarray,
                             centers: Optional[np.ndarray] = None,
                             title: str = "Cluster Analysis") -> plt.Figure:
        """
        Visualize clusters in 2D space.

        Args:
            X: Input data (n_samples, 2)
            labels: Cluster labels
            centers: Cluster centers (optional)
            title: Plot title

        Returns:
            Matplotlib figure
        """
        if X.shape[1] != 2:
            raise ValueError("Data must be 2-dimensional for 2D visualization")

        fig, ax = plt.subplots(figsize=(10, 8))

        # Get unique labels
        unique_labels = np.unique(labels)
        colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

        for label, color in zip(unique_labels, colors):
            if label == -1:
                # Noise points (for DBSCAN)
                color = 'black'
                marker = 'x'
                label_text = 'Noise'
            else:
                marker = 'o'
                label_text = f'Cluster {label}'

            mask = labels == label
            ax.scatter(X[mask, 0], X[mask, 1], c=[color], marker=marker,
                      s=80, alpha=0.6, edgecolors='black', linewidths=0.5,
                      label=label_text)

        # Plot centers if provided
        if centers is not None:
            ax.scatter(centers[:, 0], centers[:, 1], c='red', marker='*',
                      s=300, edgecolors='black', linewidths=2,
                      label='Centers', zorder=10)

        ax.set_xlabel('Feature 1', fontsize=12)
        ax.set_ylabel('Feature 2', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, loc='best')
        ax.grid(alpha=0.3)

        plt.tight_layout()
        return fig

    def visualize_clusters_3d(self, X: np.ndarray, labels: np.ndarray,
                             centers: Optional[np.ndarray] = None,
                             title: str = "3D Cluster Visualization") -> plt.Figure:
        """
        Visualize clusters in 3D space.

        Args:
            X: Input data (n_samples, 3)
            labels: Cluster labels
            centers: Cluster centers (optional)
            title: Plot title

        Returns:
            Matplotlib figure
        """
        if X.shape[1] != 3:
            raise ValueError("Data must be 3-dimensional for 3D visualization")

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Get unique labels
        unique_labels = np.unique(labels)
        colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

        for label, color in zip(unique_labels, colors):
            if label == -1:
                color = 'black'
                marker = 'x'
                label_text = 'Noise'
            else:
                marker = 'o'
                label_text = f'Cluster {label}'

            mask = labels == label
            ax.scatter(X[mask, 0], X[mask, 1], X[mask, 2],
                      c=[color], marker=marker, s=50, alpha=0.6,
                      edgecolors='black', linewidths=0.5, label=label_text)

        # Plot centers if provided
        if centers is not None:
            ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2],
                      c='red', marker='*', s=300, edgecolors='black',
                      linewidths=2, label='Centers', zorder=10)

        ax.set_xlabel('Feature 1', fontsize=11)
        ax.set_ylabel('Feature 2', fontsize=11)
        ax.set_zlabel('Feature 3', fontsize=11)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.legend(fontsize=9)

        plt.tight_layout()
        return fig

    def visualize_dendrogram(self, linkage_matrix: np.ndarray,
                            title: str = "Hierarchical Clustering Dendrogram") -> plt.Figure:
        """
        Visualize dendrogram for hierarchical clustering.

        Args:
            linkage_matrix: Linkage matrix from hierarchical clustering
            title: Plot title

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        dendrogram(linkage_matrix, ax=ax, color_threshold=0.7*max(linkage_matrix[:, 2]))

        ax.set_xlabel('Sample Index', fontsize=12)
        ax.set_ylabel('Distance', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3, axis='y')

        plt.tight_layout()
        return fig

    def visualize_optimal_k(self, elbow_result: Dict, silhouette_result: Dict,
                           davies_bouldin_result: Dict) -> plt.Figure:
        """
        Visualize optimal number of clusters using multiple methods.

        Args:
            elbow_result: Result from elbow_method()
            silhouette_result: Result from silhouette_analysis()
            davies_bouldin_result: Result from davies_bouldin_analysis()

        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Elbow method
        axes[0].plot(elbow_result['k_range'], elbow_result['inertias'],
                    'bo-', linewidth=2, markersize=8)
        axes[0].axvline(x=elbow_result['suggested_k'], color='red',
                       linestyle='--', linewidth=2, label=f"Suggested k={elbow_result['suggested_k']}")
        axes[0].set_xlabel('Number of Clusters (k)', fontsize=11)
        axes[0].set_ylabel('Inertia (WCSS)', fontsize=11)
        axes[0].set_title('Elbow Method', fontsize=12, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(alpha=0.3)

        # Silhouette analysis
        axes[1].plot(silhouette_result['k_range'], silhouette_result['silhouette_scores'],
                    'go-', linewidth=2, markersize=8)
        axes[1].axvline(x=silhouette_result['best_k'], color='red',
                       linestyle='--', linewidth=2, label=f"Best k={silhouette_result['best_k']}")
        axes[1].set_xlabel('Number of Clusters (k)', fontsize=11)
        axes[1].set_ylabel('Silhouette Score', fontsize=11)
        axes[1].set_title('Silhouette Analysis', fontsize=12, fontweight='bold')
        axes[1].legend(fontsize=10)
        axes[1].grid(alpha=0.3)

        # Davies-Bouldin index
        axes[2].plot(davies_bouldin_result['k_range'], davies_bouldin_result['davies_bouldin_scores'],
                    'ro-', linewidth=2, markersize=8)
        axes[2].axvline(x=davies_bouldin_result['best_k'], color='blue',
                       linestyle='--', linewidth=2, label=f"Best k={davies_bouldin_result['best_k']}")
        axes[2].set_xlabel('Number of Clusters (k)', fontsize=11)
        axes[2].set_ylabel('Davies-Bouldin Index', fontsize=11)
        axes[2].set_title('Davies-Bouldin Index (Lower is Better)', fontsize=12, fontweight='bold')
        axes[2].legend(fontsize=10)
        axes[2].grid(alpha=0.3)

        plt.tight_layout()
        return fig

    def visualize_silhouette_samples(self, X: np.ndarray, labels: np.ndarray,
                                     n_clusters: int) -> plt.Figure:
        """
        Visualize silhouette coefficients for each sample.

        Args:
            X: Input data
            labels: Cluster labels
            n_clusters: Number of clusters

        Returns:
            Matplotlib figure
        """
        silhouette_vals = silhouette_samples(X, labels)
        silhouette_avg = np.mean(silhouette_vals)

        fig, ax = plt.subplots(figsize=(10, 7))

        y_lower = 10
        for i in range(n_clusters):
            cluster_silhouette_vals = silhouette_vals[labels == i]
            cluster_silhouette_vals.sort()

            size_cluster_i = cluster_silhouette_vals.shape[0]
            y_upper = y_lower + size_cluster_i

            color = plt.cm.Spectral(float(i) / n_clusters)
            ax.fill_betweenx(np.arange(y_lower, y_upper),
                            0, cluster_silhouette_vals,
                            facecolor=color, edgecolor=color, alpha=0.7)

            ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i), fontsize=11)
            y_lower = y_upper + 10

        ax.axvline(x=silhouette_avg, color="red", linestyle="--",
                  linewidth=2, label=f'Average: {silhouette_avg:.3f}')
        ax.set_xlabel("Silhouette Coefficient", fontsize=12)
        ax.set_ylabel("Cluster Label", fontsize=12)
        ax.set_title(f"Silhouette Analysis (k={n_clusters})", fontsize=13, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3, axis='x')

        plt.tight_layout()
        return fig


def demo():
    """Demonstrate cluster analysis toolkit."""
    np.random.seed(42)

    print("Cluster Analysis Toolkit Demo")
    print("=" * 70)

    # Generate synthetic data with clear clusters
    print("\nGenerating synthetic data with 4 clusters...")
    from sklearn.datasets import make_blobs

    X, y_true = make_blobs(n_samples=400, centers=4, n_features=2,
                          cluster_std=0.6, random_state=42)

    print(f"Data shape: {X.shape}")
    print(f"True number of clusters: 4")

    analyzer = ClusterAnalyzer(random_state=42)

    # 1. Determine optimal number of clusters
    print("\n1. Determining Optimal Number of Clusters")
    print("-" * 70)

    elbow_result = analyzer.elbow_method(X, max_clusters=10)
    print(f"Elbow Method - Suggested k: {elbow_result['suggested_k']}")

    silhouette_result = analyzer.silhouette_analysis(X, max_clusters=10)
    print(f"Silhouette Analysis - Best k: {silhouette_result['best_k']} (score: {silhouette_result['best_score']:.3f})")

    db_result = analyzer.davies_bouldin_analysis(X, max_clusters=10)
    print(f"Davies-Bouldin Index - Best k: {db_result['best_k']} (score: {db_result['best_score']:.3f})")

    ch_result = analyzer.calinski_harabasz_analysis(X, max_clusters=10)
    print(f"Calinski-Harabasz Index - Best k: {ch_result['best_k']} (score: {ch_result['best_score']:.1f})")

    # Visualize optimal k
    fig1 = analyzer.visualize_optimal_k(elbow_result, silhouette_result, db_result)
    fig1.savefig('cluster_optimal_k.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved cluster_optimal_k.png")
    plt.close()

    # 2. K-Means Clustering
    print("\n2. K-Means Clustering")
    print("-" * 70)
    kmeans_result = analyzer.kmeans_clustering(X, n_clusters=4, init='k-means++')
    print(f"Number of clusters: {kmeans_result['n_clusters']}")
    print(f"Inertia: {kmeans_result['inertia']:.2f}")
    print(f"Iterations: {kmeans_result['n_iter']}")
    print(f"Cluster sizes: {kmeans_result['cluster_sizes']}")

    metrics = analyzer.evaluate_clustering(X, kmeans_result['labels'])
    print(f"Silhouette Score: {metrics['silhouette_score']:.3f}")
    print(f"Davies-Bouldin Index: {metrics['davies_bouldin_score']:.3f}")

    # Visualize K-Means
    fig2 = analyzer.visualize_clusters_2d(X, kmeans_result['labels'],
                                         kmeans_result['centers'],
                                         title='K-Means Clustering (k=4)')
    fig2.savefig('cluster_kmeans.png', dpi=300, bbox_inches='tight')
    print("✓ Saved cluster_kmeans.png")
    plt.close()

    # 3. Hierarchical Clustering
    print("\n3. Hierarchical Clustering")
    print("-" * 70)
    hierarchical_result = analyzer.hierarchical_clustering(X, n_clusters=4, linkage_method='ward')
    print(f"Number of clusters: {hierarchical_result['n_clusters']}")
    print(f"Linkage method: {hierarchical_result['linkage_method']}")
    print(f"Cluster sizes: {hierarchical_result['cluster_sizes']}")

    metrics = analyzer.evaluate_clustering(X, hierarchical_result['labels'])
    print(f"Silhouette Score: {metrics['silhouette_score']:.3f}")

    # Visualize Dendrogram
    fig3 = analyzer.visualize_dendrogram(hierarchical_result['linkage_matrix'],
                                        title='Hierarchical Clustering Dendrogram')
    fig3.savefig('cluster_dendrogram.png', dpi=300, bbox_inches='tight')
    print("✓ Saved cluster_dendrogram.png")
    plt.close()

    # Visualize Hierarchical Clustering
    fig4 = analyzer.visualize_clusters_2d(X, hierarchical_result['labels'],
                                         title='Hierarchical Clustering (k=4)')
    fig4.savefig('cluster_hierarchical.png', dpi=300, bbox_inches='tight')
    print("✓ Saved cluster_hierarchical.png")
    plt.close()

    # 4. DBSCAN Clustering
    print("\n4. DBSCAN Clustering")
    print("-" * 70)
    dbscan_result = analyzer.dbscan_clustering(X, eps=0.5, min_samples=5)
    print(f"Number of clusters: {dbscan_result['n_clusters']}")
    print(f"Number of noise points: {dbscan_result['n_noise']}")
    print(f"Cluster sizes: {dbscan_result['cluster_sizes']}")

    if dbscan_result['n_clusters'] > 1:
        metrics = analyzer.evaluate_clustering(X, dbscan_result['labels'])
        print(f"Silhouette Score: {metrics['silhouette_score']:.3f}")

    # Visualize DBSCAN
    fig5 = analyzer.visualize_clusters_2d(X, dbscan_result['labels'],
                                         title='DBSCAN Clustering')
    fig5.savefig('cluster_dbscan.png', dpi=300, bbox_inches='tight')
    print("✓ Saved cluster_dbscan.png")
    plt.close()

    # 5. Gaussian Mixture Model
    print("\n5. Gaussian Mixture Model")
    print("-" * 70)
    gmm_result = analyzer.gaussian_mixture(X, n_components=4, covariance_type='full')
    print(f"Number of components: {gmm_result['n_components']}")
    print(f"Converged: {gmm_result['converged']}")
    print(f"Iterations: {gmm_result['n_iter']}")
    print(f"BIC: {gmm_result['bic']:.2f}")
    print(f"AIC: {gmm_result['aic']:.2f}")
    print(f"Cluster sizes: {gmm_result['cluster_sizes']}")

    metrics = analyzer.evaluate_clustering(X, gmm_result['labels'])
    print(f"Silhouette Score: {metrics['silhouette_score']:.3f}")

    # Visualize GMM
    fig6 = analyzer.visualize_clusters_2d(X, gmm_result['labels'],
                                         gmm_result['means'],
                                         title='Gaussian Mixture Model (k=4)')
    fig6.savefig('cluster_gmm.png', dpi=300, bbox_inches='tight')
    print("✓ Saved cluster_gmm.png")
    plt.close()

    # 6. Silhouette Analysis Visualization
    print("\n6. Detailed Silhouette Analysis")
    print("-" * 70)
    fig7 = analyzer.visualize_silhouette_samples(X, kmeans_result['labels'], 4)
    fig7.savefig('cluster_silhouette_detailed.png', dpi=300, bbox_inches='tight')
    print("✓ Saved cluster_silhouette_detailed.png")
    plt.close()

    # 7. 3D Clustering Example
    print("\n7. 3D Clustering Example")
    print("-" * 70)
    X_3d, _ = make_blobs(n_samples=300, centers=3, n_features=3,
                        cluster_std=0.7, random_state=42)
    kmeans_3d = analyzer.kmeans_clustering(X_3d, n_clusters=3)
    print(f"3D K-Means - Clusters: {kmeans_3d['n_clusters']}")

    fig8 = analyzer.visualize_clusters_3d(X_3d, kmeans_3d['labels'],
                                         kmeans_3d['centers'],
                                         title='3D K-Means Clustering')
    fig8.savefig('cluster_3d.png', dpi=300, bbox_inches='tight')
    print("✓ Saved cluster_3d.png")
    plt.close()

    # Performance Summary
    print("\n8. Performance Summary (k=4)")
    print("-" * 70)
    print(f"{'Method':<20} {'Silhouette':<15} {'Davies-Bouldin':<15}")
    print("-" * 70)

    methods_results = {
        'K-Means': kmeans_result,
        'Hierarchical': hierarchical_result,
        'DBSCAN': dbscan_result,
        'Gaussian Mixture': gmm_result
    }

    for method, result in methods_results.items():
        metrics = analyzer.evaluate_clustering(X, result['labels'])
        if metrics['silhouette_score'] is not None:
            print(f"{method:<20} {metrics['silhouette_score']:<15.3f} {metrics['davies_bouldin_score']:<15.3f}")

    print("\n" + "=" * 70)
    print("✓ Cluster Analysis Demo Complete!")
    print("=" * 70)


if __name__ == '__main__':
    demo()
