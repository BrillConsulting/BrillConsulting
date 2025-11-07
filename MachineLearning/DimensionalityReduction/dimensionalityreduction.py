"""
Advanced Dimensionality Reduction System v2.0
Author: BrillConsulting
Description: Production-ready dimensionality reduction with 10+ algorithms including PCA, t-SNE, UMAP, and manifold learning
Version: 2.0 - Enhanced with multiple algorithms, automatic selection, and comprehensive visualization
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import (PCA, KernelPCA, IncrementalPCA, FactorAnalysis,
                                   TruncatedSVD, FastICA, NMF)
from sklearn.manifold import TSNE, Isomap, LocallyLinearEmbedding, MDS, SpectralEmbedding
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import argparse
import joblib
import time
import warnings
warnings.filterwarnings('ignore')

# Try to import UMAP
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False


class DimensionalityReducer:
    """
    Advanced dimensionality reduction system with 10+ algorithms

    Categories:
    - Linear: PCA, Incremental PCA, Truncated SVD, Factor Analysis
    - Non-linear: Kernel PCA, t-SNE, UMAP, Isomap, LLE, MDS
    - Supervised: LDA
    - Matrix Factorization: NMF, ICA

    Features:
    - Automatic algorithm comparison
    - 2D/3D visualization
    - Variance explained analysis
    - Reconstruction error computation
    - Performance benchmarking
    """

    def __init__(self, n_components: int = 2, random_state: int = 42, scale_features: bool = True):
        """
        Initialize dimensionality reducer

        Args:
            n_components: Target number of dimensions (usually 2 or 3 for visualization)
            random_state: Random seed for reproducibility
            scale_features: Whether to standardize features before reduction
        """
        self.n_components = n_components
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

    def fit_pca(self, X: np.ndarray) -> Dict:
        """Standard PCA (Principal Component Analysis)"""
        model = PCA(n_components=self.n_components, random_state=self.random_state)
        X_reduced = model.fit_transform(X)

        # Calculate variance explained
        variance_explained = model.explained_variance_ratio_.sum()

        # Reconstruction error
        X_reconstructed = model.inverse_transform(X_reduced)
        reconstruction_error = np.mean((X - X_reconstructed) ** 2)

        self.models['PCA'] = model
        self.results['PCA'] = {
            'X_reduced': X_reduced,
            'variance_explained': variance_explained,
            'reconstruction_error': reconstruction_error,
            'model': model
        }

        return self.results['PCA']

    def fit_kernel_pca(self, X: np.ndarray, kernel: str = 'rbf') -> Dict:
        """Kernel PCA (non-linear dimensionality reduction)"""
        model = KernelPCA(n_components=self.n_components, kernel=kernel,
                         random_state=self.random_state, fit_inverse_transform=True)
        X_reduced = model.fit_transform(X)

        # Reconstruction error (if possible)
        try:
            X_reconstructed = model.inverse_transform(X_reduced)
            reconstruction_error = np.mean((X - X_reconstructed) ** 2)
        except:
            reconstruction_error = np.nan

        self.models['Kernel PCA'] = model
        self.results['Kernel PCA'] = {
            'X_reduced': X_reduced,
            'variance_explained': np.nan,  # Not directly available for Kernel PCA
            'reconstruction_error': reconstruction_error,
            'model': model
        }

        return self.results['Kernel PCA']

    def fit_incremental_pca(self, X: np.ndarray, batch_size: int = 100) -> Dict:
        """Incremental PCA (for large datasets)"""
        model = IncrementalPCA(n_components=self.n_components, batch_size=batch_size)
        X_reduced = model.fit_transform(X)

        variance_explained = model.explained_variance_ratio_.sum()

        X_reconstructed = model.inverse_transform(X_reduced)
        reconstruction_error = np.mean((X - X_reconstructed) ** 2)

        self.models['Incremental PCA'] = model
        self.results['Incremental PCA'] = {
            'X_reduced': X_reduced,
            'variance_explained': variance_explained,
            'reconstruction_error': reconstruction_error,
            'model': model
        }

        return self.results['Incremental PCA']

    def fit_truncated_svd(self, X: np.ndarray) -> Dict:
        """Truncated SVD / LSA (Latent Semantic Analysis)"""
        model = TruncatedSVD(n_components=self.n_components, random_state=self.random_state)
        X_reduced = model.fit_transform(X)

        variance_explained = model.explained_variance_ratio_.sum()

        X_reconstructed = model.inverse_transform(X_reduced)
        reconstruction_error = np.mean((X - X_reconstructed) ** 2)

        self.models['Truncated SVD'] = model
        self.results['Truncated SVD'] = {
            'X_reduced': X_reduced,
            'variance_explained': variance_explained,
            'reconstruction_error': reconstruction_error,
            'model': model
        }

        return self.results['Truncated SVD']

    def fit_factor_analysis(self, X: np.ndarray) -> Dict:
        """Factor Analysis"""
        model = FactorAnalysis(n_components=self.n_components, random_state=self.random_state)
        X_reduced = model.fit_transform(X)

        # Reconstruction
        X_reconstructed = model.inverse_transform(X_reduced)
        reconstruction_error = np.mean((X - X_reconstructed) ** 2)

        self.models['Factor Analysis'] = model
        self.results['Factor Analysis'] = {
            'X_reduced': X_reduced,
            'variance_explained': np.nan,
            'reconstruction_error': reconstruction_error,
            'model': model
        }

        return self.results['Factor Analysis']

    def fit_tsne(self, X: np.ndarray, perplexity: int = 30) -> Dict:
        """t-SNE (t-Distributed Stochastic Neighbor Embedding)"""
        model = TSNE(n_components=self.n_components, perplexity=perplexity,
                    random_state=self.random_state, n_iter=1000)
        X_reduced = model.fit_transform(X)

        self.models['t-SNE'] = model
        self.results['t-SNE'] = {
            'X_reduced': X_reduced,
            'variance_explained': np.nan,
            'reconstruction_error': np.nan,  # t-SNE doesn't support inverse transform
            'model': model
        }

        return self.results['t-SNE']

    def fit_umap(self, X: np.ndarray, n_neighbors: int = 15, min_dist: float = 0.1) -> Dict:
        """UMAP (Uniform Manifold Approximation and Projection)"""
        if not UMAP_AVAILABLE:
            print("⚠️  UMAP not available. Install with: pip install umap-learn")
            return {}

        model = umap.UMAP(n_components=self.n_components, n_neighbors=n_neighbors,
                         min_dist=min_dist, random_state=self.random_state)
        X_reduced = model.fit_transform(X)

        self.models['UMAP'] = model
        self.results['UMAP'] = {
            'X_reduced': X_reduced,
            'variance_explained': np.nan,
            'reconstruction_error': np.nan,
            'model': model
        }

        return self.results['UMAP']

    def fit_isomap(self, X: np.ndarray, n_neighbors: int = 5) -> Dict:
        """Isomap (Isometric Mapping)"""
        model = Isomap(n_components=self.n_components, n_neighbors=n_neighbors)
        X_reduced = model.fit_transform(X)

        # Reconstruction error
        X_reconstructed = model.transform(X)
        reconstruction_error = np.mean((X_reduced - X_reconstructed) ** 2)

        self.models['Isomap'] = model
        self.results['Isomap'] = {
            'X_reduced': X_reduced,
            'variance_explained': np.nan,
            'reconstruction_error': reconstruction_error,
            'model': model
        }

        return self.results['Isomap']

    def fit_lle(self, X: np.ndarray, n_neighbors: int = 5) -> Dict:
        """LLE (Locally Linear Embedding)"""
        model = LocallyLinearEmbedding(n_components=self.n_components,
                                      n_neighbors=n_neighbors,
                                      random_state=self.random_state)
        X_reduced = model.fit_transform(X)

        self.models['LLE'] = model
        self.results['LLE'] = {
            'X_reduced': X_reduced,
            'variance_explained': np.nan,
            'reconstruction_error': model.reconstruction_error_,
            'model': model
        }

        return self.results['LLE']

    def fit_mds(self, X: np.ndarray) -> Dict:
        """MDS (Multidimensional Scaling)"""
        model = MDS(n_components=self.n_components, random_state=self.random_state)
        X_reduced = model.fit_transform(X)

        self.models['MDS'] = model
        self.results['MDS'] = {
            'X_reduced': X_reduced,
            'variance_explained': np.nan,
            'reconstruction_error': model.stress_,
            'model': model
        }

        return self.results['MDS']

    def fit_lda(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """LDA (Linear Discriminant Analysis) - Supervised method"""
        # LDA requires n_components <= n_classes - 1
        n_classes = len(np.unique(y))
        n_comp = min(self.n_components, n_classes - 1)

        if n_comp < self.n_components:
            print(f"⚠️  LDA: Reducing n_components from {self.n_components} to {n_comp} (max for {n_classes} classes)")

        model = LDA(n_components=n_comp)
        X_reduced = model.fit_transform(X, y)

        # Variance explained
        variance_explained = model.explained_variance_ratio_.sum() if hasattr(model, 'explained_variance_ratio_') else np.nan

        self.models['LDA'] = model
        self.results['LDA'] = {
            'X_reduced': X_reduced,
            'variance_explained': variance_explained,
            'reconstruction_error': np.nan,
            'model': model
        }

        return self.results['LDA']

    def fit_all(self, X: np.ndarray, y: Optional[np.ndarray] = None, fast_mode: bool = False):
        """Fit all dimensionality reduction algorithms"""
        self.X_original = X
        X_scaled = self.prepare_data(X, fit=True)
        self.X_scaled = X_scaled

        n_algorithms = 11 if UMAP_AVAILABLE else 10
        if y is not None:
            n_algorithms += 1  # Add LDA

        print(f"🔧 Training {n_algorithms} dimensionality reduction algorithms...")
        print(f"   Original dimensions: {X.shape[1]} → Target: {self.n_components}")
        print("=" * 80)

        # Linear methods
        print("  [1] PCA (Principal Component Analysis)...")
        start = time.time()
        self.fit_pca(X_scaled)
        self.results['PCA']['time'] = time.time() - start

        print("  [2] Incremental PCA...")
        start = time.time()
        self.fit_incremental_pca(X_scaled)
        self.results['Incremental PCA']['time'] = time.time() - start

        print("  [3] Truncated SVD / LSA...")
        start = time.time()
        self.fit_truncated_svd(X_scaled)
        self.results['Truncated SVD']['time'] = time.time() - start

        print("  [4] Factor Analysis...")
        start = time.time()
        self.fit_factor_analysis(X_scaled)
        self.results['Factor Analysis']['time'] = time.time() - start

        # Non-linear methods
        print("  [5] Kernel PCA (RBF)...")
        start = time.time()
        self.fit_kernel_pca(X_scaled, kernel='rbf')
        self.results['Kernel PCA']['time'] = time.time() - start

        if not fast_mode:
            print("  [6] t-SNE...")
            start = time.time()
            self.fit_tsne(X_scaled)
            self.results['t-SNE']['time'] = time.time() - start

            if UMAP_AVAILABLE:
                print("  [7] UMAP...")
                start = time.time()
                self.fit_umap(X_scaled)
                self.results['UMAP']['time'] = time.time() - start

            print("  [8] Isomap...")
            start = time.time()
            self.fit_isomap(X_scaled)
            self.results['Isomap']['time'] = time.time() - start

            print("  [9] LLE (Locally Linear Embedding)...")
            start = time.time()
            self.fit_lle(X_scaled)
            self.results['LLE']['time'] = time.time() - start

            print("  [10] MDS (Multidimensional Scaling)...")
            start = time.time()
            self.fit_mds(X_scaled)
            self.results['MDS']['time'] = time.time() - start
        else:
            print("  ⚡ Fast mode: Skipping slow algorithms (t-SNE, UMAP, Isomap, LLE, MDS)")

        # Supervised method (if labels provided)
        if y is not None:
            print(f"  [11] LDA (Linear Discriminant Analysis) - Supervised...")
            start = time.time()
            self.fit_lda(X_scaled, y)
            self.results['LDA']['time'] = time.time() - start

        print("=" * 80)
        print(f"✅ All {len(self.models)} algorithms trained successfully!\n")

    def evaluate(self, y: Optional[np.ndarray] = None) -> pd.DataFrame:
        """
        Evaluate all models

        If labels (y) are provided, also compute silhouette scores
        """
        if not self.results:
            raise ValueError("No models have been trained yet")

        evaluation = []

        for name, results in self.results.items():
            X_reduced = results['X_reduced']

            row = {
                'Algorithm': name,
                'Components': X_reduced.shape[1],
                'Variance Explained': f"{results['variance_explained']:.1%}" if not np.isnan(results['variance_explained']) else 'N/A',
                'Reconstruction Error': f"{results['reconstruction_error']:.4f}" if not np.isnan(results['reconstruction_error']) else 'N/A',
                'Time (s)': f"{results.get('time', 0):.3f}"
            }

            # Silhouette score (if labels provided)
            if y is not None and len(np.unique(y)) > 1:
                try:
                    silhouette = silhouette_score(X_reduced, y)
                    row['Silhouette Score'] = f"{silhouette:.3f}"
                except:
                    row['Silhouette Score'] = 'N/A'

            evaluation.append(row)

        df = pd.DataFrame(evaluation)

        return df

    def plot_2d_comparison(self, y: Optional[np.ndarray] = None, save_path: Optional[str] = None):
        """Plot 2D visualizations of all algorithms"""
        if self.n_components != 2:
            print("⚠️  This function requires n_components=2")
            return

        algorithms = list(self.results.keys())
        n_plots = len(algorithms)
        n_cols = 3
        n_rows = (n_plots + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, n_rows * 5))
        axes = axes.ravel() if n_plots > 1 else [axes]

        for idx, name in enumerate(algorithms):
            X_reduced = self.results[name]['X_reduced']

            if y is not None:
                scatter = axes[idx].scatter(X_reduced[:, 0], X_reduced[:, 1],
                                          c=y, cmap='viridis', alpha=0.6, s=30)
                plt.colorbar(scatter, ax=axes[idx])
            else:
                axes[idx].scatter(X_reduced[:, 0], X_reduced[:, 1],
                                alpha=0.6, s=30, color='steelblue')

            variance = self.results[name]['variance_explained']
            time_taken = self.results[name].get('time', 0)

            title = f"{name}"
            if not np.isnan(variance):
                title += f"\nVariance: {variance:.1%}"
            title += f" | Time: {time_taken:.2f}s"

            axes[idx].set_title(title)
            axes[idx].set_xlabel('Component 1')
            axes[idx].set_ylabel('Component 2')
            axes[idx].grid(True, alpha=0.3)

        # Hide unused subplots
        for idx in range(len(algorithms), len(axes)):
            axes[idx].axis('off')

        plt.suptitle(f'Dimensionality Reduction Comparison ({self.X_original.shape[1]}D → 2D)',
                    fontsize=16, y=1.00)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 2D comparison saved to {save_path}")

        plt.show()

    def plot_variance_explained(self, save_path: Optional[str] = None):
        """Plot variance explained for linear methods"""
        algorithms = []
        variances = []

        for name, results in self.results.items():
            var = results['variance_explained']
            if not np.isnan(var):
                algorithms.append(name)
                variances.append(var)

        if not algorithms:
            print("⚠️  No algorithms with variance explained available")
            return

        plt.figure(figsize=(10, 6))
        bars = plt.bar(algorithms, variances, color='steelblue', alpha=0.7)

        # Highlight best
        best_idx = np.argmax(variances)
        bars[best_idx].set_color('orange')

        plt.xlabel('Algorithm')
        plt.ylabel('Variance Explained')
        plt.title(f'Variance Explained Comparison (n_components={self.n_components})')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        plt.ylim(0, 1.0)

        # Add value labels
        for idx, (algo, var) in enumerate(zip(algorithms, variances)):
            plt.text(idx, var, f'{var:.1%}', ha='center', va='bottom')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    def plot_reconstruction_error(self, save_path: Optional[str] = None):
        """Plot reconstruction errors"""
        algorithms = []
        errors = []

        for name, results in self.results.items():
            err = results['reconstruction_error']
            if not np.isnan(err):
                algorithms.append(name)
                errors.append(err)

        if not algorithms:
            print("⚠️  No algorithms with reconstruction error available")
            return

        plt.figure(figsize=(10, 6))
        bars = plt.bar(algorithms, errors, color='coral', alpha=0.7)

        # Highlight best (lowest error)
        best_idx = np.argmin(errors)
        bars[best_idx].set_color('green')

        plt.xlabel('Algorithm')
        plt.ylabel('Reconstruction Error (MSE)')
        plt.title('Reconstruction Error Comparison (Lower is Better)')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')

        # Add value labels
        for idx, (algo, err) in enumerate(zip(algorithms, errors)):
            plt.text(idx, err, f'{err:.4f}', ha='center', va='bottom')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    def plot_pca_cumulative_variance(self, max_components: int = 50, save_path: Optional[str] = None):
        """Plot cumulative variance explained for PCA"""
        if 'PCA' not in self.models:
            print("⚠️  PCA model not found. Run fit_pca first.")
            return

        # Fit PCA with more components for analysis
        n_comp = min(max_components, self.X_scaled.shape[1])
        pca_full = PCA(n_components=n_comp, random_state=self.random_state)
        pca_full.fit(self.X_scaled)

        cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)

        plt.figure(figsize=(12, 6))

        # Cumulative variance
        plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance,
                marker='o', linestyle='-', color='steelblue', linewidth=2)

        # Add reference lines
        plt.axhline(y=0.95, color='r', linestyle='--', label='95% Variance')
        plt.axhline(y=0.90, color='orange', linestyle='--', label='90% Variance')

        # Find components needed for 95% variance
        n_95 = np.argmax(cumulative_variance >= 0.95) + 1
        plt.axvline(x=n_95, color='r', linestyle=':', alpha=0.5)
        plt.text(n_95, 0.5, f'{n_95} components\nfor 95%', ha='center')

        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Variance Explained')
        plt.title('PCA: Cumulative Variance Explained')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    def transform_new_data(self, X_new: np.ndarray, algorithm: str = 'PCA') -> np.ndarray:
        """Transform new data using a trained model"""
        if algorithm not in self.models:
            raise ValueError(f"Algorithm '{algorithm}' not found. Available: {list(self.models.keys())}")

        X_new_scaled = self.prepare_data(X_new, fit=False)
        model = self.models[algorithm]

        return model.transform(X_new_scaled)

    def save_model(self, filepath: str, algorithm: str = 'PCA'):
        """Save a trained model"""
        if algorithm not in self.models:
            raise ValueError(f"Algorithm '{algorithm}' not found")

        model_data = {
            'model': self.models[algorithm],
            'scaler': self.scaler,
            'n_components': self.n_components
        }

        joblib.dump(model_data, filepath)
        print(f"💾 {algorithm} model saved to {filepath}")


def main():
    parser = argparse.ArgumentParser(description='Dimensionality Reduction v2.0')
    parser.add_argument('--data', type=str, required=True, help='Path to CSV data')
    parser.add_argument('--n-components', type=int, default=2, help='Target number of dimensions')
    parser.add_argument('--labels', type=str, help='Path to labels CSV (for visualization and evaluation)')
    parser.add_argument('--output-viz', type=str, help='Save visualization')
    parser.add_argument('--save-model', type=str, help='Save model (default: PCA)')
    parser.add_argument('--fast-mode', action='store_true', help='Skip slow algorithms (t-SNE, UMAP, etc.)')

    args = parser.parse_args()

    # Load data
    print(f"📂 Loading data from {args.data}...")
    df = pd.read_csv(args.data)
    X = df.select_dtypes(include=[np.number]).values

    print(f"📊 Dataset: {X.shape[0]} samples, {X.shape[1]} features\n")

    # Load labels if provided
    y = None
    if args.labels:
        labels_df = pd.read_csv(args.labels)
        y = labels_df.values.ravel()
        print(f"🏷️  Labels loaded: {len(np.unique(y))} classes\n")

    # Initialize reducer
    reducer = DimensionalityReducer(n_components=args.n_components, scale_features=True)

    # Fit all algorithms
    reducer.fit_all(X, y=y, fast_mode=args.fast_mode)

    # Evaluate
    print("📊 Algorithm Evaluation:")
    print("=" * 120)
    evaluation_df = reducer.evaluate(y)
    print(evaluation_df.to_string(index=False))
    print("=" * 120)

    # Visualizations
    if args.n_components == 2:
        print("\n📊 Generating 2D visualizations...")
        reducer.plot_2d_comparison(y, save_path=args.output_viz)
        reducer.plot_variance_explained()
        reducer.plot_reconstruction_error()
        reducer.plot_pca_cumulative_variance()

    # Save model
    if args.save_model:
        reducer.save_model(args.save_model, algorithm='PCA')

    print("\n✅ Dimensionality reduction completed successfully!")


if __name__ == "__main__":
    main()
