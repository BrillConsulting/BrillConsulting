# 📉 Dimensionality Reduction v2.0

Production-ready dimensionality reduction with 10+ algorithms including PCA, t-SNE, UMAP, and manifold learning methods.

## 🌟 Algorithms

### Linear Methods
1. **PCA** - Principal Component Analysis
2. **Incremental PCA** - Memory-efficient PCA for large datasets
3. **Truncated SVD** - Singular Value Decomposition / LSA
4. **Factor Analysis** - Latent variable model

### Non-Linear Methods
5. **Kernel PCA** - Non-linear PCA with RBF kernel
6. **t-SNE** - t-Distributed Stochastic Neighbor Embedding
7. **UMAP** - Uniform Manifold Approximation (optional)
8. **Isomap** - Isometric Mapping
9. **LLE** - Locally Linear Embedding
10. **MDS** - Multidimensional Scaling

### Supervised Methods
11. **LDA** - Linear Discriminant Analysis (requires labels)

## ✨ Key Features

- **10+ Reduction Algorithms** with automatic comparison
- **Linear & Non-linear Methods** for different data types
- **Variance Explained Analysis** for interpretability
- **Reconstruction Error** computation
- **2D/3D Visualization** support
- **Performance Benchmarking** (execution time)
- **Silhouette Score** evaluation (with labels)
- **Fast Mode** to skip slow algorithms
- **Model Persistence** (save/load trained models)

## 🚀 Quick Start

```bash
python dimensionalityreduction.py --data highdim.csv --n-components 2 --output-viz viz.png
```

## 🏆 Status

**Version:** 2.0
**Lines of Code:** 662
**Status:** Production-Ready ✅

---

**Author**: BrillConsulting | AI Consultant & Data Scientist
