# Cluster Analysis Toolkit

A comprehensive clustering system implementing multiple state-of-the-art algorithms for discovering patterns and groupings in data.

## Description

The Cluster Analysis Toolkit provides a unified interface for clustering data using various unsupervised learning methods. It combines multiple clustering algorithms, validation metrics, and comprehensive visualization tools to identify natural groupings in datasets.

## Key Features

- **Multiple Clustering Algorithms**
  - K-means for efficient partitional clustering
  - Hierarchical clustering with multiple linkage methods
  - DBSCAN for density-based clustering
  - Gaussian Mixture Models for probabilistic clustering
  - Spectral clustering for graph-based clustering

- **Cluster Validation**
  - Silhouette analysis for cluster quality assessment
  - Elbow method for optimal k determination
  - Davies-Bouldin index
  - Calinski-Harabasz score
  - Within-cluster sum of squares (WCSS)

- **Optimal Cluster Selection**
  - Automated methods for determining the best number of clusters
  - Multiple evaluation metrics comparison
  - Gap statistic implementation

- **Comprehensive Visualizations**
  - 2D/3D cluster scatter plots
  - Dendrogram visualization for hierarchical clustering
  - Silhouette plots
  - Elbow curves
  - Cluster centroids and boundaries

## Technologies Used

- **Python 3.x**
- **NumPy** - Numerical computing
- **Pandas** - Data manipulation
- **scikit-learn** - Machine learning algorithms
- **Matplotlib/Seaborn** - Visualization
- **SciPy** - Hierarchical clustering and metrics

## Installation

```bash
# Clone the repository
cd /home/user/BrillConsulting/DataScience/ClusterAnalysis

# Install required packages
pip install numpy pandas scikit-learn matplotlib seaborn scipy
```

## Usage Examples

### Basic K-Means Clustering

```python
from cluster_analysis import ClusterAnalyzer
import numpy as np

# Generate sample data
np.random.seed(42)
X = np.vstack([
    np.random.randn(100, 2) + [2, 2],
    np.random.randn(100, 2) + [-2, -2],
    np.random.randn(100, 2) + [2, -2]
])

# Initialize analyzer
analyzer = ClusterAnalyzer(random_state=42)

# Perform K-means clustering
result = analyzer.kmeans_clustering(X, n_clusters=3)
print(f"Cluster labels: {result['labels']}")
print(f"Cluster centers: {result['cluster_centers']}")
print(f"Inertia: {result['inertia']:.3f}")
print(f"Silhouette score: {result['silhouette_score']:.3f}")
```

### Hierarchical Clustering

```python
# Perform hierarchical clustering
hierarchical_result = analyzer.hierarchical_clustering(
    X,
    n_clusters=3,
    linkage_method='ward'
)

print(f"Detected {hierarchical_result['n_clusters']} clusters")
print(f"Silhouette score: {hierarchical_result['silhouette_score']:.3f}")

# Visualize dendrogram
fig = hierarchical_result['dendrogram_fig']
fig.savefig('dendrogram.png', dpi=300, bbox_inches='tight')
```

### DBSCAN Density-Based Clustering

```python
# DBSCAN for arbitrary-shaped clusters
dbscan_result = analyzer.dbscan_clustering(
    X,
    eps=0.5,
    min_samples=5
)

print(f"Number of clusters: {dbscan_result['n_clusters']}")
print(f"Number of noise points: {dbscan_result['n_noise']}")
print(f"Silhouette score: {dbscan_result['silhouette_score']:.3f}")
```

### Gaussian Mixture Models

```python
# Probabilistic clustering with GMM
gmm_result = analyzer.gaussian_mixture_clustering(
    X,
    n_components=3,
    covariance_type='full'
)

print(f"BIC: {gmm_result['bic']:.3f}")
print(f"AIC: {gmm_result['aic']:.3f}")
print(f"Log-likelihood: {gmm_result['log_likelihood']:.3f}")
print(f"Silhouette score: {gmm_result['silhouette_score']:.3f}")

# Get probability predictions
print(f"Cluster probabilities shape: {gmm_result['probabilities'].shape}")
```

### Determining Optimal Number of Clusters

```python
# Find optimal k using multiple methods
optimal_result = analyzer.determine_optimal_clusters(
    X,
    method='all',
    k_range=range(2, 10)
)

print(f"Elbow method suggests: {optimal_result['elbow_k']} clusters")
print(f"Silhouette method suggests: {optimal_result['silhouette_k']} clusters")

# Visualize elbow curve
fig = optimal_result['elbow_fig']
fig.savefig('elbow_curve.png', dpi=300, bbox_inches='tight')
```

### Cluster Validation Metrics

```python
# Comprehensive validation metrics
validation = analyzer.cluster_validation_metrics(
    X,
    result['labels']
)

print(f"Silhouette score: {validation['silhouette_score']:.3f}")
print(f"Davies-Bouldin index: {validation['davies_bouldin_index']:.3f}")
print(f"Calinski-Harabasz score: {validation['calinski_harabasz_score']:.3f}")

# Per-sample silhouette values
print(f"Sample silhouette scores: {validation['silhouette_samples'][:5]}")

# Visualize silhouette plot
fig = validation['silhouette_fig']
fig.savefig('silhouette_plot.png', dpi=300, bbox_inches='tight')
```

## Demo Instructions

Run the comprehensive demo to see all features in action:

```bash
python cluster_analysis.py
```

The demo will:
1. Generate synthetic data with known cluster structure
2. Apply all clustering methods (K-means, Hierarchical, DBSCAN, GMM, Spectral)
3. Determine optimal number of clusters
4. Calculate validation metrics for each method
5. Generate visualizations (saved as PNG files)
6. Display a comprehensive performance comparison

## Output Examples

**Console Output:**
```
Cluster Analysis Toolkit Demo
======================================================================

Generating synthetic data with clusters...
Total samples: 300
Number of true clusters: 3

1. K-Means Clustering
----------------------------------------------------------------------
Number of clusters: 3
Inertia: 245.671
Silhouette score: 0.652
Davies-Bouldin index: 0.531

2. Hierarchical Clustering (Ward linkage)
----------------------------------------------------------------------
Number of clusters: 3
Silhouette score: 0.641
Davies-Bouldin index: 0.548

3. DBSCAN Clustering
----------------------------------------------------------------------
Number of clusters: 3
Number of noise points: 5
Silhouette score: 0.598

4. Gaussian Mixture Models
----------------------------------------------------------------------
Number of components: 3
BIC: 1523.45
AIC: 1489.12
Silhouette score: 0.665

5. Spectral Clustering
----------------------------------------------------------------------
Number of clusters: 3
Silhouette score: 0.678

Cluster Validation Summary
----------------------------------------------------------------------
Method               Silhouette   Davies-Bouldin   Calinski-Harabasz
----------------------------------------------------------------------
K-Means              0.652        0.531            425.3
Hierarchical         0.641        0.548            412.8
DBSCAN               0.598        0.612            385.2
GMM                  0.665        0.519            438.1
Spectral             0.678        0.505            445.6

Optimal clusters (Elbow method): 3
Optimal clusters (Silhouette): 3
```

**Generated Visualizations:**
- `cluster_kmeans_2d.png` - K-means clustering results in 2D
- `dendrogram.png` - Hierarchical clustering dendrogram
- `cluster_comparison.png` - Side-by-side comparison of all methods
- `elbow_curve.png` - Elbow method for optimal k determination
- `silhouette_plot.png` - Silhouette analysis visualization

## Author

**Brill Consulting**

---

For more information about the algorithms and methodologies, see the inline documentation in `cluster_analysis.py`.
