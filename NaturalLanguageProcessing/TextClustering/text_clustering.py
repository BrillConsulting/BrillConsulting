"""
Advanced Text Clustering System v2.0
Author: BrillConsulting
Description: Multi-algorithm document clustering for organization and discovery

Supports K-Means, DBSCAN, Hierarchical, and Spectral clustering
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Feature extraction
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import PCA, LatentDirichletAllocation

# Clustering algorithms
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# Visualization
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Advanced methods (optional)
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Warning: sentence-transformers not available. Install with: pip install sentence-transformers")


class TextClusterer:
    """
    Advanced Text Clustering System

    Supports multiple algorithms:
    - K-Means: Partition-based clustering
    - DBSCAN: Density-based clustering
    - Hierarchical: Agglomerative clustering
    - Spectral: Graph-based clustering
    """

    def __init__(self, n_clusters=5, method='kmeans', vectorizer='tfidf', embedding_model=None):
        """
        Initialize text clusterer

        Args:
            n_clusters: Number of clusters (for K-Means, Hierarchical, Spectral)
            method: 'kmeans', 'dbscan', 'hierarchical', 'spectral'
            vectorizer: 'tfidf', 'count', 'sentence-transformers'
            embedding_model: Optional SentenceTransformer model name
        """
        self.n_clusters = n_clusters
        self.method = method
        self.vectorizer_type = vectorizer

        # Vectorizer
        if vectorizer == 'tfidf':
            self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        elif vectorizer == 'count':
            self.vectorizer = CountVectorizer(max_features=5000, stop_words='english')
        elif vectorizer == 'sentence-transformers':
            if not SENTENCE_TRANSFORMERS_AVAILABLE:
                raise ImportError("sentence-transformers not available")
            self.vectorizer = SentenceTransformer(embedding_model or 'all-MiniLM-L6-v2')
        else:
            raise ValueError(f"Unknown vectorizer: {vectorizer}")

        # Clustering model
        self.model = None
        self.labels_ = None
        self.embeddings_ = None

        # Documents
        self.documents = []

        print(f"✓ TextClusterer initialized (method={method}, n_clusters={n_clusters})")

    def fit(self, documents: List[str], **kwargs):
        """
        Fit clustering model on documents

        Args:
            documents: List of text documents
            **kwargs: Additional parameters for clustering algorithm

        Returns:
            Cluster labels
        """
        self.documents = documents

        # Vectorize documents
        print(f"Vectorizing {len(documents)} documents...")
        if self.vectorizer_type == 'sentence-transformers':
            self.embeddings_ = self.vectorizer.encode(documents, show_progress_bar=False)
        else:
            self.embeddings_ = self.vectorizer.fit_transform(documents).toarray()

        print(f"Embedding shape: {self.embeddings_.shape}")

        # Cluster
        print(f"Clustering with {self.method}...")
        if self.method == 'kmeans':
            self._fit_kmeans(**kwargs)
        elif self.method == 'dbscan':
            self._fit_dbscan(**kwargs)
        elif self.method == 'hierarchical':
            self._fit_hierarchical(**kwargs)
        elif self.method == 'spectral':
            self._fit_spectral(**kwargs)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        print(f"✓ Found {len(set(self.labels_))} clusters")

        return self.labels_

    def _fit_kmeans(self, **kwargs):
        """K-Means clustering"""
        self.model = KMeans(n_clusters=self.n_clusters, random_state=42, **kwargs)
        self.labels_ = self.model.fit_predict(self.embeddings_)

    def _fit_dbscan(self, eps=0.5, min_samples=5, **kwargs):
        """DBSCAN clustering"""
        self.model = DBSCAN(eps=eps, min_samples=min_samples, **kwargs)
        self.labels_ = self.model.fit_predict(self.embeddings_)

    def _fit_hierarchical(self, linkage='ward', **kwargs):
        """Hierarchical clustering"""
        self.model = AgglomerativeClustering(
            n_clusters=self.n_clusters,
            linkage=linkage,
            **kwargs
        )
        self.labels_ = self.model.fit_predict(self.embeddings_)

    def _fit_spectral(self, **kwargs):
        """Spectral clustering"""
        self.model = SpectralClustering(
            n_clusters=self.n_clusters,
            random_state=42,
            **kwargs
        )
        self.labels_ = self.model.fit_predict(self.embeddings_)

    def get_clusters(self) -> Dict[int, List[int]]:
        """
        Get cluster assignments

        Returns:
            Dict mapping cluster ID to list of document indices
        """
        if self.labels_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        clusters = {}
        for doc_idx, cluster_id in enumerate(self.labels_):
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(doc_idx)

        return clusters

    def get_cluster_documents(self, cluster_id: int) -> List[str]:
        """Get documents in a specific cluster"""
        clusters = self.get_clusters()
        indices = clusters.get(cluster_id, [])
        return [self.documents[idx] for idx in indices]

    def get_top_terms(self, cluster_id: int, top_n=10) -> List[str]:
        """
        Get top terms for a cluster (for TF-IDF/Count vectorizers)

        Args:
            cluster_id: Cluster ID
            top_n: Number of top terms

        Returns:
            List of top terms
        """
        if self.vectorizer_type == 'sentence-transformers':
            return []  # Not applicable for sentence transformers

        clusters = self.get_clusters()
        doc_indices = clusters.get(cluster_id, [])

        if not doc_indices:
            return []

        # Get cluster center (mean of embeddings)
        cluster_embeddings = self.embeddings_[doc_indices]
        cluster_center = np.mean(cluster_embeddings, axis=0)

        # Get top terms
        feature_names = self.vectorizer.get_feature_names_out()
        top_indices = cluster_center.argsort()[-top_n:][::-1]

        return [feature_names[idx] for idx in top_indices]

    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate clustering quality

        Returns:
            Dict with evaluation metrics
        """
        if self.labels_ is None or self.embeddings_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        # Remove noise points for evaluation (label = -1 in DBSCAN)
        valid_mask = self.labels_ >= 0
        valid_embeddings = self.embeddings_[valid_mask]
        valid_labels = self.labels_[valid_mask]

        if len(set(valid_labels)) < 2:
            return {
                'silhouette_score': 0.0,
                'davies_bouldin_score': 0.0,
                'calinski_harabasz_score': 0.0,
                'n_clusters': len(set(self.labels_)),
                'n_noise': sum(self.labels_ == -1)
            }

        return {
            'silhouette_score': silhouette_score(valid_embeddings, valid_labels),
            'davies_bouldin_score': davies_bouldin_score(valid_embeddings, valid_labels),
            'calinski_harabasz_score': calinski_harabasz_score(valid_embeddings, valid_labels),
            'n_clusters': len(set(self.labels_)),
            'n_noise': sum(self.labels_ == -1)
        }

    def visualize(self, save_path='clusters.png', method='pca'):
        """
        Visualize clusters in 2D

        Args:
            save_path: Path to save plot
            method: Dimensionality reduction method ('pca')
        """
        if self.embeddings_ is None or self.labels_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        # Reduce to 2D
        if method == 'pca':
            pca = PCA(n_components=2, random_state=42)
            embeddings_2d = pca.fit_transform(self.embeddings_)
        else:
            raise ValueError(f"Unknown method: {method}")

        # Plot
        plt.figure(figsize=(12, 8))

        unique_labels = set(self.labels_)
        colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

        for label, color in zip(unique_labels, colors):
            if label == -1:
                # Noise points
                color = 'black'
                marker = 'x'
            else:
                marker = 'o'

            mask = self.labels_ == label
            plt.scatter(
                embeddings_2d[mask, 0],
                embeddings_2d[mask, 1],
                c=[color],
                label=f'Cluster {label}',
                marker=marker,
                s=50,
                alpha=0.6
            )

        plt.title(f'Text Clustering - {self.method.upper()}')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✓ Visualization saved to {save_path}")

    def get_cluster_summary(self) -> Dict[int, Dict]:
        """
        Get summary for each cluster

        Returns:
            Dict with cluster summaries
        """
        clusters = self.get_clusters()
        summaries = {}

        for cluster_id, doc_indices in clusters.items():
            summaries[cluster_id] = {
                'size': len(doc_indices),
                'documents': [self.documents[idx][:100] + '...' for idx in doc_indices[:3]],
                'top_terms': self.get_top_terms(cluster_id, top_n=5)
            }

        return summaries


def find_optimal_clusters(documents: List[str], max_clusters=10, method='kmeans'):
    """
    Find optimal number of clusters using elbow method

    Args:
        documents: List of documents
        max_clusters: Maximum number of clusters to try
        method: Clustering method

    Returns:
        Dict with scores for each number of clusters
    """
    scores = {}

    for n in range(2, max_clusters + 1):
        print(f"Trying {n} clusters...")
        clusterer = TextClusterer(n_clusters=n, method=method)
        clusterer.fit(documents)

        metrics = clusterer.evaluate()
        scores[n] = {
            'silhouette': metrics['silhouette_score'],
            'davies_bouldin': metrics['davies_bouldin_score'],
            'calinski_harabasz': metrics['calinski_harabasz_score']
        }

    return scores


def demo_text_clustering():
    """Demonstrate text clustering"""
    # Sample documents
    documents = [
        # Cluster 1: Machine Learning
        "Machine learning is a type of artificial intelligence that allows computers to learn.",
        "Deep learning uses neural networks with multiple layers to analyze data.",
        "Supervised learning requires labeled training data to make predictions.",
        "Unsupervised learning finds patterns in data without labeled examples.",

        # Cluster 2: Natural Language Processing
        "Natural language processing enables computers to understand human language.",
        "Text classification categorizes documents into predefined categories.",
        "Sentiment analysis determines the emotional tone of text.",
        "Named entity recognition identifies entities like names and locations in text.",

        # Cluster 3: Computer Vision
        "Computer vision allows machines to interpret visual information from images.",
        "Object detection identifies and locates objects within images.",
        "Image segmentation divides an image into meaningful regions.",
        "Face recognition technology identifies individuals from facial features.",

        # Cluster 4: Data Science
        "Data science combines statistics, programming, and domain expertise.",
        "Data visualization helps communicate insights from complex datasets.",
        "Exploratory data analysis reveals patterns and relationships in data.",
        "Statistical modeling uses mathematical frameworks to represent data.",
    ]

    print("=" * 80)
    print("Advanced Text Clustering System v2.0")
    print("Author: BrillConsulting")
    print("=" * 80)
    print(f"\n📊 Sample Data: {len(documents)} documents")

    # Method 1: K-Means
    print("\n" + "=" * 80)
    print("Method 1: K-Means Clustering")
    print("=" * 80)

    clusterer_kmeans = TextClusterer(n_clusters=4, method='kmeans', vectorizer='tfidf')
    labels = clusterer_kmeans.fit(documents)

    print("\n📈 Evaluation Metrics:")
    metrics = clusterer_kmeans.evaluate()
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}" if isinstance(value, float) else f"  {metric}: {value}")

    print("\n📋 Cluster Summary:")
    summary = clusterer_kmeans.get_cluster_summary()
    for cluster_id, info in sorted(summary.items()):
        print(f"\nCluster {cluster_id} ({info['size']} documents):")
        print(f"  Top terms: {', '.join(info['top_terms'])}")
        print(f"  Sample docs:")
        for doc in info['documents']:
            print(f"    - {doc}")

    # Visualize
    clusterer_kmeans.visualize(save_path='clusters_kmeans.png')

    # Method 2: DBSCAN
    print("\n" + "=" * 80)
    print("Method 2: DBSCAN (Density-Based)")
    print("=" * 80)

    clusterer_dbscan = TextClusterer(method='dbscan', vectorizer='tfidf')
    labels = clusterer_dbscan.fit(documents, eps=0.5, min_samples=2)

    print("\n📈 Evaluation Metrics:")
    metrics = clusterer_dbscan.evaluate()
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}" if isinstance(value, float) else f"  {metric}: {value}")

    # Method 3: Hierarchical
    print("\n" + "=" * 80)
    print("Method 3: Hierarchical Clustering")
    print("=" * 80)

    clusterer_hier = TextClusterer(n_clusters=4, method='hierarchical', vectorizer='tfidf')
    labels = clusterer_hier.fit(documents, linkage='ward')

    print("\n📈 Evaluation Metrics:")
    metrics = clusterer_hier.evaluate()
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}" if isinstance(value, float) else f"  {metric}: {value}")

    print("\n" + "=" * 80)
    print("✓ Demo completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    demo_text_clustering()
