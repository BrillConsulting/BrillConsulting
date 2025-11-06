# Dimensionality Reduction Toolkit

A comprehensive toolkit implementing multiple advanced algorithms for reducing the dimensionality of high-dimensional datasets while preserving essential structure and information.

## Description

The Dimensionality Reduction Toolkit provides a unified interface for transforming high-dimensional data into lower-dimensional representations. It combines linear and non-linear techniques, manifold learning methods, and neural network-based approaches to reveal hidden patterns and structure in complex datasets.

## Key Features

- **Linear Methods**
  - Principal Component Analysis (PCA)
  - Kernel PCA for non-linear patterns
  - Factor Analysis for latent variable modeling
  - Independent Component Analysis (ICA)

- **Manifold Learning**
  - t-distributed Stochastic Neighbor Embedding (t-SNE)
  - Uniform Manifold Approximation and Projection (UMAP)
  - Multi-dimensional Scaling (MDS)
  - Isomap for geodesic distance preservation

- **Neural Network-Based**
  - Autoencoder architectures
  - Variational Autoencoders (VAE)
  - Denoising autoencoders
  - Custom encoder-decoder networks

- **Analysis Tools**
  - Variance explained analysis
  - Reconstruction error metrics
  - Scree plots and cumulative variance
  - Component interpretation

- **Visualization**
  - 2D/3D embeddings visualization
  - Component loadings plots
  - Reconstruction quality assessment
  - Interactive scatter plots with labels

## Technologies Used

- **Python 3.x**
- **NumPy** - Numerical computing
- **Pandas** - Data manipulation
- **scikit-learn** - Linear dimensionality reduction methods
- **Matplotlib/Seaborn** - Visualization
- **UMAP-learn** - UMAP implementation
- **TensorFlow/PyTorch** - Neural network-based methods (optional)

## Installation

```bash
# Clone the repository
cd /home/user/BrillConsulting/DataScience/DimensionalityReduction

# Install required packages
pip install numpy pandas scikit-learn matplotlib seaborn umap-learn
```

## Usage Examples

### Principal Component Analysis (PCA)

```python
from dimensionality_reduction import DimensionalityReducer
import numpy as np
from sklearn.datasets import load_digits

# Load high-dimensional data
digits = load_digits()
X = digits.data  # 64 dimensions
y = digits.target

# Initialize reducer
reducer = DimensionalityReducer(random_state=42)

# Apply PCA
pca_result = reducer.pca(
    X,
    n_components=2,
    return_variance=True
)

print(f"Reduced shape: {pca_result['X_transformed'].shape}")
print(f"Explained variance ratio: {pca_result['explained_variance_ratio']}")
print(f"Cumulative variance: {pca_result['cumulative_variance']:.3f}")
print(f"Components shape: {pca_result['components'].shape}")
```

### Kernel PCA for Non-linear Patterns

```python
# Apply Kernel PCA with RBF kernel
kpca_result = reducer.kernel_pca(
    X,
    n_components=2,
    kernel='rbf',
    gamma=0.001
)

print(f"Kernel PCA transformed shape: {kpca_result['X_transformed'].shape}")
print(f"Kernel: {kpca_result['kernel']}")
```

### t-SNE for Visualization

```python
# Apply t-SNE for visualization
tsne_result = reducer.tsne(
    X,
    n_components=2,
    perplexity=30,
    n_iter=1000,
    learning_rate=200
)

print(f"t-SNE embedding shape: {tsne_result['X_transformed'].shape}")
print(f"KL divergence: {tsne_result['kl_divergence']:.3f}")

# Visualize t-SNE results
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))
scatter = plt.scatter(
    tsne_result['X_transformed'][:, 0],
    tsne_result['X_transformed'][:, 1],
    c=y,
    cmap='tab10',
    alpha=0.6
)
plt.colorbar(scatter)
plt.title('t-SNE Visualization of Digits Dataset')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.savefig('tsne_visualization.png', dpi=300, bbox_inches='tight')
```

### UMAP for Scalable Manifold Learning

```python
# Apply UMAP
umap_result = reducer.umap(
    X,
    n_components=2,
    n_neighbors=15,
    min_dist=0.1,
    metric='euclidean'
)

print(f"UMAP embedding shape: {umap_result['X_transformed'].shape}")
print(f"Number of neighbors: {umap_result['n_neighbors']}")
print(f"Minimum distance: {umap_result['min_dist']}")
```

### Autoencoder for Deep Learning-Based Reduction

```python
# Apply autoencoder
autoencoder_result = reducer.autoencoder(
    X,
    encoding_dim=10,
    hidden_layers=[32, 16],
    epochs=50,
    batch_size=32
)

print(f"Encoded representation shape: {autoencoder_result['X_encoded'].shape}")
print(f"Reconstruction error: {autoencoder_result['reconstruction_error']:.4f}")
print(f"Model architecture: {autoencoder_result['model_summary']}")

# Decode back to original space
X_reconstructed = autoencoder_result['decoder'].predict(
    autoencoder_result['X_encoded']
)
print(f"Reconstructed shape: {X_reconstructed.shape}")
```

### Variance Analysis and Component Selection

```python
# Analyze variance explained by components
variance_analysis = reducer.analyze_variance(
    X,
    max_components=20
)

print(f"Optimal components (95% variance): {variance_analysis['n_components_95']}")
print(f"Optimal components (99% variance): {variance_analysis['n_components_99']}")

# Plot scree plot
fig = variance_analysis['scree_plot']
fig.savefig('scree_plot.png', dpi=300, bbox_inches='tight')

# Plot cumulative variance
fig = variance_analysis['cumulative_variance_plot']
fig.savefig('cumulative_variance.png', dpi=300, bbox_inches='tight')
```

### Multi-method Comparison

```python
# Compare multiple dimensionality reduction methods
comparison = reducer.compare_methods(
    X,
    methods=['pca', 'tsne', 'umap', 'kernel_pca'],
    n_components=2,
    labels=y
)

print("Method Comparison:")
for method, results in comparison.items():
    print(f"{method}:")
    print(f"  Silhouette score: {results['silhouette_score']:.3f}")
    print(f"  Computation time: {results['computation_time']:.3f}s")

# Visualize comparison
fig = comparison['comparison_plot']
fig.savefig('method_comparison.png', dpi=300, bbox_inches='tight')
```

## Demo Instructions

Run the comprehensive demo to see all features in action:

```bash
python dimensionality_reduction.py
```

The demo will:
1. Load high-dimensional dataset (digits or synthetic data)
2. Apply all dimensionality reduction methods (PCA, Kernel PCA, t-SNE, UMAP, autoencoders)
3. Perform variance analysis and component selection
4. Generate 2D visualizations for each method
5. Compare methods based on quality metrics
6. Save visualizations as PNG files
7. Display performance comparison

## Output Examples

**Console Output:**
```
Dimensionality Reduction Toolkit Demo
======================================================================

Loading dataset...
Original shape: (1797, 64)
Number of classes: 10

1. Principal Component Analysis (PCA)
----------------------------------------------------------------------
Reduced dimensions: 64 -> 2
Explained variance ratio: [0.1498, 0.1357]
Cumulative variance: 28.55%
Components for 95% variance: 21
Components for 99% variance: 41

2. Kernel PCA (RBF kernel)
----------------------------------------------------------------------
Reduced dimensions: 64 -> 2
Kernel: rbf
Gamma: 0.001

3. t-SNE
----------------------------------------------------------------------
Reduced dimensions: 64 -> 2
Perplexity: 30
KL divergence: 1.234
Computation time: 12.3s

4. UMAP
----------------------------------------------------------------------
Reduced dimensions: 64 -> 2
Number of neighbors: 15
Minimum distance: 0.1
Computation time: 2.1s

5. Autoencoder
----------------------------------------------------------------------
Encoding dimension: 10
Hidden layers: [32, 16]
Epochs: 50
Reconstruction error: 0.0234
Training time: 8.7s

Method Comparison
----------------------------------------------------------------------
Method               Silhouette   Time (s)     Preservation
----------------------------------------------------------------------
PCA                  0.342        0.05         Linear
Kernel PCA           0.389        0.82         Non-linear
t-SNE                0.523        12.31        Local structure
UMAP                 0.567        2.14         Global + Local
Autoencoder          0.445        8.67         Non-linear
```

**Generated Visualizations:**
- `pca_2d.png` - PCA projection in 2D space
- `scree_plot.png` - Variance explained by each component
- `cumulative_variance.png` - Cumulative variance plot
- `tsne_visualization.png` - t-SNE embedding with class labels
- `umap_visualization.png` - UMAP embedding with class labels
- `method_comparison.png` - Side-by-side comparison of all methods
- `reconstruction_analysis.png` - Autoencoder reconstruction quality

## Author

**Brill Consulting**

---

For more information about the algorithms and methodologies, see the inline documentation in `dimensionality_reduction.py`.
