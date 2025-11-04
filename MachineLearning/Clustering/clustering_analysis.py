"""
Unsupervised Clustering Analysis
Author: BrillConsulting
Description: K-Means, DBSCAN, and Hierarchical Clustering with visualization
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bould in_score, calinski_harabasz_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
import argparse


class ClusteringAnalyzer:
    """Comprehensive clustering analysis system"""

    def __init__(self, scale_features=True):
        self.scale_features = scale_features
        self.scaler = StandardScaler() if scale_features else None
        self.models = {}
        self.results = {}

    def prepare_data(self, X):
        """Prepare and scale data"""
        if self.scale_features:
            X_scaled = self.scaler.fit_transform(X)
            return X_scaled
        return X

    def kmeans_clustering(self, X, n_clusters_range=range(2, 11)):
        """K-Means with elbow method"""
        print("🔧 K-Means Clustering...")

        inertias = []
        silhouette_scores = []

        for k in n_clusters_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)

            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(X, labels))

        # Best k by silhouette score
        best_k = n_clusters_range[np.argmax(silhouette_scores)]
        best_kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        best_labels = best_kmeans.fit_predict(X)

        self.models['K-Means'] = best_kmeans
        self.results['K-Means'] = {
            'model': best_kmeans,
            'labels': best_labels,
            'n_clusters': best_k,
            'inertias': inertias,
            'silhouette_scores': silhouette_scores,
            'silhouette': silhouette_score(X, best_labels),
            'davies_bouldin': davies_bouldin_score(X, best_labels),
            'calinski_harabasz': calinski_harabasz_score(X, best_labels)
        }

        print(f"  Best k: {best_k}")
        print(f"  Silhouette Score: {silhouette_score(X, best_labels):.4f}\n")

        return best_labels

    def dbscan_clustering(self, X, eps=0.5, min_samples=5):
        """DBSCAN clustering"""
        print(f"🔧 DBSCAN Clustering (eps={eps}, min_samples={min_samples})...")

        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(X)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)

        self.models['DBSCAN'] = dbscan
        self.results['DBSCAN'] = {
            'model': dbscan,
            'labels': labels,
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'silhouette': silhouette_score(X, labels) if n_clusters > 1 else 0
        }

        print(f"  Clusters found: {n_clusters}")
        print(f"  Noise points: {n_noise}\n")

        return labels

    def hierarchical_clustering(self, X, n_clusters=3):
        """Hierarchical clustering"""
        print(f"🔧 Hierarchical Clustering (n_clusters={n_clusters})...")

        hc = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
        labels = hc.fit_predict(X)

        self.models['Hierarchical'] = hc
        self.results['Hierarchical'] = {
            'model': hc,
            'labels': labels,
            'n_clusters': n_clusters,
            'silhouette': silhouette_score(X, labels),
            'davies_bouldin': davies_bouldin_score(X, labels),
            'calinski_harabasz': calinski_harabasz_score(X, labels)
        }

        print(f"  Silhouette Score: {silhouette_score(X, labels):.4f}\n")

        return labels

    def plot_clusters(self, X, save_path=None):
        """Visualize clusters using PCA"""
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        for idx, (name, results) in enumerate(self.results.items()):
            labels = results['labels']

            scatter = axes[idx].scatter(X_pca[:, 0], X_pca[:, 1],
                                       c=labels, cmap='viridis',
                                       alpha=0.6, s=50)
            axes[idx].set_title(f'{name}\nClusters: {results["n_clusters"]}, '
                               f'Silhouette: {results.get("silhouette", 0):.3f}')
            axes[idx].set_xlabel('PC1')
            axes[idx].set_ylabel('PC2')
            plt.colorbar(scatter, ax=axes[idx])

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Clusters plot saved to {save_path}")

        plt.show()

    def plot_elbow(self, save_path=None):
        """Plot elbow curve for K-Means"""
        if 'K-Means' not in self.results:
            print("❌ Run K-Means first")
            return

        results = self.results['K-Means']
        inertias = results['inertias']
        silhouette_scores = results['silhouette_scores']

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Elbow curve
        ax1.plot(range(2, len(inertias) + 2), inertias, 'bo-')
        ax1.set_xlabel('Number of Clusters')
        ax1.set_ylabel('Inertia')
        ax1.set_title('Elbow Method')
        ax1.grid(True, alpha=0.3)

        # Silhouette scores
        ax2.plot(range(2, len(silhouette_scores) + 2), silhouette_scores, 'ro-')
        ax2.set_xlabel('Number of Clusters')
        ax2.set_ylabel('Silhouette Score')
        ax2.set_title('Silhouette Score')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Clustering Analysis')
    parser.add_argument('--data', type=str, required=True, help='Path to CSV data')
    parser.add_argument('--output', type=str, help='Output plot path')

    args = parser.parse_args()

    # Load data
    print(f"📂 Loading data from {args.data}...\n")
    df = pd.read_csv(args.data)
    X = df.select_dtypes(include=[np.number]).values

    print(f"📊 Dataset: {X.shape[0]} samples, {X.shape[1]} features\n")

    # Initialize analyzer
    analyzer = ClusteringAnalyzer(scale_features=True)
    X_scaled = analyzer.prepare_data(X)

    # Run all clustering algorithms
    analyzer.kmeans_clustering(X_scaled)
    analyzer.dbscan_clustering(X_scaled)
    analyzer.hierarchical_clustering(X_scaled, n_clusters=3)

    # Visualize
    analyzer.plot_clusters(X_scaled, save_path=args.output)
    analyzer.plot_elbow()


if __name__ == "__main__":
    main()
