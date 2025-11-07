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

### Basic Usage (2D Visualization)

```bash
python dimensionalityreduction.py --data highdim.csv --n-components 2 --output-viz viz.png
```

### With Labels (Supervised)

```bash
python dimensionalityreduction.py --data data.csv --labels labels.csv --n-components 2
```

### Fast Mode (Skip Slow Algorithms)

```bash
python dimensionalityreduction.py --data large_data.csv --n-components 3 --fast-mode
```

## 📊 Example Code

```python
from dimensionalityreduction import DimensionalityReducer
import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('highdim_data.csv')
X = df.select_dtypes(include=[np.number]).values
y = df['label'].values  # Optional labels

# Initialize reducer
reducer = DimensionalityReducer(n_components=2, scale_features=True)

# Fit all algorithms
reducer.fit_all(X, y=y, fast_mode=False)

# Evaluate
evaluation_df = reducer.evaluate(y)
print(evaluation_df)

# Visualize
reducer.plot_2d_comparison(y, save_path='comparison.png')
reducer.plot_variance_explained()
reducer.plot_reconstruction_error()
reducer.plot_pca_cumulative_variance(max_components=50)

# Transform new data
X_new_reduced = reducer.transform_new_data(X_new, algorithm='PCA')

# Save model
reducer.save_model('pca_model.pkl', algorithm='PCA')
```

## 🎯 Use Cases

### 📊 Data Visualization
- **High-dimensional data** → 2D/3D plots
- **Exploratory analysis** before modeling
- **Pattern discovery** in complex datasets

### 🧠 Feature Engineering
- **Noise reduction** by keeping top components
- **Feature compression** for faster models
- **Multicollinearity** removal

### 🔬 Scientific Analysis
- **Gene expression** analysis (bioinformatics)
- **Image compression** and analysis
- **Text analysis** (LSA/SVD for document embeddings)

### 🤖 Machine Learning Pipeline
- **Preprocessing step** before classification/regression
- **Curse of dimensionality** mitigation
- **Model speedup** with reduced features

## 📈 Algorithm Comparison

| Algorithm | Type | Best For | Speed | Inverse Transform | Parameters |
|-----------|------|----------|-------|-------------------|------------|
| **PCA** | Linear | Variance maximization, general use | Fast ⚡ | ✅ Yes | n_components |
| **Incremental PCA** | Linear | Large datasets (out-of-core) | Fast ⚡ | ✅ Yes | n_components, batch_size |
| **Truncated SVD** | Linear | Sparse data, text (LSA) | Fast ⚡ | ✅ Yes | n_components |
| **Factor Analysis** | Linear | Latent variables, psychology | Medium | ✅ Yes | n_components |
| **Kernel PCA** | Non-linear | Non-linear patterns | Medium | ⚠️ Approximate | kernel, n_components |
| **t-SNE** | Non-linear | Visualization (2D/3D) | Slow 🐢 | ❌ No | perplexity, n_components |
| **UMAP** | Non-linear | Visualization + clustering | Fast ⚡ | ❌ No | n_neighbors, min_dist |
| **Isomap** | Non-linear | Manifold learning | Slow 🐢 | ✅ Yes | n_neighbors |
| **LLE** | Non-linear | Locally linear manifolds | Slow 🐢 | ❌ No | n_neighbors |
| **MDS** | Non-linear | Distance preservation | Slow 🐢 | ❌ No | n_components |
| **LDA** | Supervised | Classification (requires labels) | Fast ⚡ | ❌ No | n_components (≤ n_classes-1) |

## 🎨 Visualization Examples

### 1. 2D Comparison Plot
Compare all algorithms side-by-side in 2D space:
```python
reducer.plot_2d_comparison(y, save_path='2d_comparison.png')
```

### 2. Variance Explained
See how much information each algorithm preserves:
```python
reducer.plot_variance_explained(save_path='variance.png')
```

### 3. Reconstruction Error
Compare reconstruction quality:
```python
reducer.plot_reconstruction_error(save_path='error.png')
```

### 4. PCA Cumulative Variance
Determine optimal number of components:
```python
reducer.plot_pca_cumulative_variance(max_components=50, save_path='cumulative.png')
```

## 🔧 Advanced Configuration

### Custom Number of Components

```python
# Reduce to 3D for 3D visualization
reducer = DimensionalityReducer(n_components=3)

# Reduce to 50D for model input
reducer = DimensionalityReducer(n_components=50)
```

### Algorithm-Specific Parameters

```python
# t-SNE with custom perplexity
reducer.fit_tsne(X, perplexity=50)

# UMAP with custom parameters
reducer.fit_umap(X, n_neighbors=30, min_dist=0.01)

# LLE with more neighbors
reducer.fit_lle(X, n_neighbors=10)

# Kernel PCA with polynomial kernel
reducer.fit_kernel_pca(X, kernel='poly')
```

### Fast Mode (Skip Slow Algorithms)

```python
# Skip t-SNE, UMAP, Isomap, LLE, MDS
reducer.fit_all(X, fast_mode=True)
```

## 📊 Performance Metrics

### Variance Explained
- **What**: Percentage of original variance preserved
- **Available for**: PCA, Incremental PCA, Truncated SVD, LDA
- **Higher is better**: More information retained

### Reconstruction Error
- **What**: Mean Squared Error when reconstructing original data
- **Available for**: PCA, Incremental PCA, Truncated SVD, Factor Analysis, LLE
- **Lower is better**: Better reconstruction quality

### Silhouette Score (with labels)
- **What**: How well-separated clusters are in reduced space
- **Range**: [-1, 1]
- **Higher is better**: Well-separated clusters

### Execution Time
- **What**: Time to fit the algorithm
- **Comparison**: Identify fastest algorithms for your use case

## 📚 Algorithm Details

### PCA (Principal Component Analysis)
- **Principle**: Finds orthogonal directions of maximum variance
- **Best for**: General-purpose dimensionality reduction
- **Pros**: Fast, interpretable, inverse transform
- **Cons**: Linear only, assumes Gaussian-like data

### t-SNE (t-Distributed Stochastic Neighbor Embedding)
- **Principle**: Preserves local neighborhood structure
- **Best for**: 2D/3D visualization, finding clusters
- **Pros**: Excellent visualization, reveals clusters
- **Cons**: Slow, no inverse transform, stochastic

### UMAP (Uniform Manifold Approximation and Projection)
- **Principle**: Manifold learning with topological foundations
- **Best for**: Visualization + downstream tasks (clustering, classification)
- **Pros**: Fast, preserves global + local structure
- **Cons**: Requires umap-learn package, newer (less established)

### Kernel PCA
- **Principle**: PCA in non-linear feature space (kernel trick)
- **Best for**: Non-linear data with known structure
- **Pros**: Non-linear, still relatively fast
- **Cons**: Approximate inverse transform, parameter tuning

### LDA (Linear Discriminant Analysis)
- **Principle**: Maximizes class separability (supervised)
- **Best for**: Classification preprocessing
- **Pros**: Supervised, maximizes class separation
- **Cons**: Requires labels, max n_classes-1 components

## 🛠️ Model Persistence

### Save Model

```python
# Save PCA model
reducer.save_model('pca_model.pkl', algorithm='PCA')

# Save t-SNE model
reducer.save_model('tsne_model.pkl', algorithm='t-SNE')
```

### Load and Use Model

```python
import joblib

# Load model
model_data = joblib.load('pca_model.pkl')
model = model_data['model']
scaler = model_data['scaler']

# Transform new data
X_new_scaled = scaler.transform(X_new)
X_new_reduced = model.transform(X_new_scaled)
```

## 💡 Best Practices

### 1. **Always Scale Features**
- Essential for distance-based methods (t-SNE, UMAP, LLE)
- Enabled by default in DimensionalityReducer

### 2. **Start with PCA**
- Fast baseline for any dimensionality reduction task
- Interpretable variance explained
- Use PCA cumulative variance plot to determine optimal n_components

### 3. **Use t-SNE/UMAP for Visualization Only**
- Don't use for downstream ML tasks (no inverse transform)
- Best for exploratory 2D/3D plotting

### 4. **Consider Computation Time**
- t-SNE, Isomap, LLE, MDS are slow on large datasets
- Use fast mode or subsample data for exploration
- PCA, Incremental PCA, UMAP are fast

### 5. **Validate with Silhouette Score**
- If you have labels, check silhouette scores
- Higher score = better cluster separation in reduced space

## 🐛 Troubleshooting

**Poor Visualization Quality?**
- Try different algorithms (t-SNE vs UMAP vs PCA)
- Adjust perplexity (t-SNE) or n_neighbors (UMAP, Isomap, LLE)
- Ensure features are scaled

**High Reconstruction Error?**
- Increase n_components
- Try non-linear methods (Kernel PCA, Isomap)
- Check PCA cumulative variance plot

**Slow Performance?**
- Use fast_mode to skip slow algorithms
- Try PCA, Incremental PCA, or UMAP instead of t-SNE
- Subsample data for exploration

**Memory Issues?**
- Use Incremental PCA for large datasets
- Reduce batch_size parameter
- Subsample data

**LDA Not Working?**
- Ensure labels are provided
- Check n_components ≤ n_classes - 1
- Verify sufficient samples per class

## 📄 Dependencies

```bash
pip install numpy pandas scikit-learn matplotlib seaborn joblib

# Optional for UMAP
pip install umap-learn
```

## 🏆 Status

**Version:** 2.0
**Lines of Code:** 662
**Status:** Production-Ready ✅

**Features:**
- ✅ 10+ Reduction Algorithms
- ✅ Linear & Non-Linear Methods
- ✅ Comprehensive Evaluation
- ✅ 2D/3D Visualization
- ✅ Variance & Reconstruction Analysis
- ✅ Fast Mode Option
- ✅ Model Persistence
- ✅ Production-Ready Code

## 📞 Support

**Author**: BrillConsulting | AI Consultant & Data Scientist
**Email**: clientbrill@gmail.com
**LinkedIn**: [BrillConsulting](https://www.linkedin.com/in/brillconsulting)

---

**⭐ Star this repository if you find it useful!**

*Made with ❤️ by BrillConsulting*
