# 🎨 Clustering Analysis v2.0

Production-ready unsupervised clustering with 8+ algorithms including K-Means, DBSCAN, Hierarchical, GMM, and more.

## 🌟 Algorithms

### Partition-Based
1. **K-Means** - Centroid-based clustering with elbow method
2. **Gaussian Mixture (GMM)** - Probabilistic clustering with soft assignments

### Density-Based
3. **DBSCAN** - Density-Based Spatial Clustering with noise detection
4. **OPTICS** - Ordering Points To Identify Clustering Structure
5. **Mean Shift** - Mode-seeking algorithm without k parameter

### Hierarchical
6. **Hierarchical Clustering** - Agglomerative clustering with dendrogram

### Graph-Based
7. **Spectral Clustering** - Uses eigenvalues of similarity matrix

### Incremental
8. **BIRCH** - Balanced Iterative Reducing and Clustering using Hierarchies

## ✨ Key Features

- **8+ Clustering Algorithms** with automatic comparison
- **Automatic K Selection** using silhouette score (K-Means)
- **Multiple Metrics** - Silhouette, Davies-Bouldin, Calinski-Harabasz
- **PCA Visualization** for 2D cluster plotting
- **Elbow Method** for optimal cluster number
- **Noise Detection** (DBSCAN, OPTICS)
- **Production-Ready** with comprehensive evaluation

## 🚀 Quick Start

```bash
python clustering_analysis.py --data customers.csv --output clusters.png
```

## 🏆 Status

**Version:** 2.0
**Lines of Code:** 397
**Status:** Production-Ready ✅

---

**Author**: BrillConsulting | AI Consultant & Data Scientist
