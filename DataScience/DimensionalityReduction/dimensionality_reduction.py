"""
Dimensionality Reduction Toolkit
=================================

Comprehensive dimensionality reduction techniques:
- PCA (Principal Component Analysis)
- t-SNE (t-Distributed Stochastic Neighbor Embedding)
- Factor Analysis
- Truncated SVD
- LDA (Linear Discriminant Analysis)
- Scree plots and explained variance
- Reconstruction error analysis
- 2D/3D visualization

Author: Brill Consulting
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA, FactorAnalysis, TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')


class DimensionalityReducer:
    """Comprehensive dimensionality reduction toolkit."""

    def __init__(self, random_state: int = 42):
        """
        Initialize dimensionality reducer.

        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        np.random.seed(random_state)
        self.scaler = StandardScaler()
        self.models = {}
        self.transformed_data = {}

    def pca_reduction(self, X: np.ndarray, n_components: Optional[int] = None,
                     variance_ratio: float = 0.95) -> Dict:
        """
        Principal Component Analysis dimensionality reduction.

        Args:
            X: Input data (n_samples, n_features)
            n_components: Number of components (if None, determined by variance_ratio)
            variance_ratio: Cumulative variance ratio to retain

        Returns:
            Dictionary with transformed data and PCA statistics
        """
        # Standardize data
        X_scaled = self.scaler.fit_transform(X)

        # Determine n_components if not specified
        if n_components is None:
            pca_temp = PCA(random_state=self.random_state)
            pca_temp.fit(X_scaled)
            cumsum = np.cumsum(pca_temp.explained_variance_ratio_)
            n_components = np.argmax(cumsum >= variance_ratio) + 1

        # Fit PCA
        pca = PCA(n_components=n_components, random_state=self.random_state)
        X_transformed = pca.fit_transform(X_scaled)

        # Store model
        self.models['pca'] = pca
        self.transformed_data['pca'] = X_transformed

        return {
            'transformed_data': X_transformed,
            'explained_variance': pca.explained_variance_,
            'explained_variance_ratio': pca.explained_variance_ratio_,
            'cumulative_variance_ratio': np.cumsum(pca.explained_variance_ratio_),
            'components': pca.components_,
            'n_components': n_components,
            'total_variance_explained': np.sum(pca.explained_variance_ratio_),
            'singular_values': pca.singular_values_,
            'method': 'PCA'
        }

    def tsne_reduction(self, X: np.ndarray, n_components: int = 2,
                      perplexity: float = 30.0, n_iter: int = 1000,
                      learning_rate: float = 200.0) -> Dict:
        """
        t-SNE dimensionality reduction.

        Args:
            X: Input data (n_samples, n_features)
            n_components: Number of dimensions (typically 2 or 3)
            perplexity: Perplexity parameter
            n_iter: Number of iterations
            learning_rate: Learning rate

        Returns:
            Dictionary with transformed data
        """
        # Standardize data
        X_scaled = self.scaler.fit_transform(X)

        # Fit t-SNE
        tsne = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            n_iter=n_iter,
            learning_rate=learning_rate,
            random_state=self.random_state,
            verbose=0
        )
        X_transformed = tsne.fit_transform(X_scaled)

        # Store model and results
        self.models['tsne'] = tsne
        self.transformed_data['tsne'] = X_transformed

        return {
            'transformed_data': X_transformed,
            'n_components': n_components,
            'perplexity': perplexity,
            'n_iter': n_iter,
            'kl_divergence': tsne.kl_divergence_,
            'method': 't-SNE'
        }

    def factor_analysis(self, X: np.ndarray, n_components: int = 2,
                       max_iter: int = 1000) -> Dict:
        """
        Factor Analysis dimensionality reduction.

        Args:
            X: Input data (n_samples, n_features)
            n_components: Number of factors
            max_iter: Maximum iterations

        Returns:
            Dictionary with transformed data and factor loadings
        """
        # Standardize data
        X_scaled = self.scaler.fit_transform(X)

        # Fit Factor Analysis
        fa = FactorAnalysis(
            n_components=n_components,
            max_iter=max_iter,
            random_state=self.random_state
        )
        X_transformed = fa.fit_transform(X_scaled)

        # Store model
        self.models['factor_analysis'] = fa
        self.transformed_data['factor_analysis'] = X_transformed

        return {
            'transformed_data': X_transformed,
            'components': fa.components_,
            'noise_variance': fa.noise_variance_,
            'n_components': n_components,
            'log_likelihood': fa.score(X_scaled) * len(X_scaled),
            'n_iter': fa.n_iter_,
            'method': 'Factor Analysis'
        }

    def truncated_svd(self, X: np.ndarray, n_components: int = 2) -> Dict:
        """
        Truncated SVD (LSA) dimensionality reduction.

        Args:
            X: Input data (n_samples, n_features)
            n_components: Number of components

        Returns:
            Dictionary with transformed data and SVD statistics
        """
        # Truncated SVD doesn't require centering
        svd = TruncatedSVD(n_components=n_components, random_state=self.random_state)
        X_transformed = svd.fit_transform(X)

        # Store model
        self.models['truncated_svd'] = svd
        self.transformed_data['truncated_svd'] = X_transformed

        return {
            'transformed_data': X_transformed,
            'explained_variance': svd.explained_variance_,
            'explained_variance_ratio': svd.explained_variance_ratio_,
            'cumulative_variance_ratio': np.cumsum(svd.explained_variance_ratio_),
            'components': svd.components_,
            'n_components': n_components,
            'total_variance_explained': np.sum(svd.explained_variance_ratio_),
            'singular_values': svd.singular_values_,
            'method': 'Truncated SVD'
        }

    def lda_reduction(self, X: np.ndarray, y: np.ndarray,
                     n_components: Optional[int] = None) -> Dict:
        """
        Linear Discriminant Analysis for supervised dimensionality reduction.

        Args:
            X: Input data (n_samples, n_features)
            y: Target labels
            n_components: Number of components (max = n_classes - 1)

        Returns:
            Dictionary with transformed data and LDA statistics
        """
        # Standardize data
        X_scaled = self.scaler.fit_transform(X)

        # Determine max components
        n_classes = len(np.unique(y))
        max_components = min(n_classes - 1, X.shape[1])

        if n_components is None:
            n_components = max_components
        else:
            n_components = min(n_components, max_components)

        # Fit LDA
        lda = LinearDiscriminantAnalysis(n_components=n_components)
        X_transformed = lda.fit_transform(X_scaled, y)

        # Store model
        self.models['lda'] = lda
        self.transformed_data['lda'] = X_transformed

        return {
            'transformed_data': X_transformed,
            'explained_variance_ratio': lda.explained_variance_ratio_,
            'cumulative_variance_ratio': np.cumsum(lda.explained_variance_ratio_),
            'n_components': n_components,
            'total_variance_explained': np.sum(lda.explained_variance_ratio_),
            'scalings': lda.scalings_,
            'method': 'LDA'
        }

    def reconstruction_error(self, X: np.ndarray, model_name: str) -> float:
        """
        Calculate reconstruction error for a dimensionality reduction method.

        Args:
            X: Original data
            model_name: Name of the model ('pca', 'factor_analysis', 'truncated_svd')

        Returns:
            Reconstruction error (MSE)
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found. Run the reduction method first.")

        model = self.models[model_name]

        if model_name == 'pca':
            X_scaled = self.scaler.transform(X)
            X_transformed = model.transform(X_scaled)
            X_reconstructed = model.inverse_transform(X_transformed)
            X_reconstructed = self.scaler.inverse_transform(X_reconstructed)
        elif model_name == 'factor_analysis':
            X_scaled = self.scaler.transform(X)
            X_transformed = model.transform(X_scaled)
            X_reconstructed = model.inverse_transform(X_transformed)
            X_reconstructed = self.scaler.inverse_transform(X_reconstructed)
        elif model_name == 'truncated_svd':
            X_transformed = model.transform(X)
            X_reconstructed = model.inverse_transform(X_transformed)
        else:
            raise ValueError(f"Reconstruction not supported for '{model_name}'")

        mse = mean_squared_error(X, X_reconstructed)
        return mse

    def optimal_n_components_analysis(self, X: np.ndarray,
                                     max_components: int = 20) -> Dict:
        """
        Analyze optimal number of components using multiple criteria.

        Args:
            X: Input data (n_samples, n_features)
            max_components: Maximum number of components to test

        Returns:
            Dictionary with analysis results
        """
        X_scaled = self.scaler.fit_transform(X)

        max_comp = min(max_components, min(X.shape) - 1)
        components_range = range(1, max_comp + 1)

        explained_variances = []
        reconstruction_errors = []

        for n_comp in components_range:
            # PCA
            pca = PCA(n_components=n_comp, random_state=self.random_state)
            X_transformed = pca.fit_transform(X_scaled)

            # Explained variance
            explained_variances.append(np.sum(pca.explained_variance_ratio_))

            # Reconstruction error
            X_reconstructed = pca.inverse_transform(X_transformed)
            mse = mean_squared_error(X_scaled, X_reconstructed)
            reconstruction_errors.append(mse)

        return {
            'n_components_range': list(components_range),
            'explained_variances': explained_variances,
            'reconstruction_errors': reconstruction_errors
        }

    def visualize_scree_plot(self, explained_variance_ratio: np.ndarray,
                            title: str = "Scree Plot") -> plt.Figure:
        """
        Visualize scree plot for explained variance.

        Args:
            explained_variance_ratio: Explained variance ratio for each component
            title: Plot title

        Returns:
            Matplotlib figure
        """
        n_components = len(explained_variance_ratio)
        cumulative_variance = np.cumsum(explained_variance_ratio)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Individual variance
        axes[0].bar(range(1, n_components + 1), explained_variance_ratio,
                   alpha=0.7, edgecolor='black', color='steelblue')
        axes[0].set_xlabel('Principal Component', fontsize=11)
        axes[0].set_ylabel('Explained Variance Ratio', fontsize=11)
        axes[0].set_title('Explained Variance by Component', fontsize=12, fontweight='bold')
        axes[0].grid(alpha=0.3, axis='y')

        # Cumulative variance
        axes[1].plot(range(1, n_components + 1), cumulative_variance,
                    'o-', linewidth=2, markersize=8, color='darkred')
        axes[1].axhline(y=0.95, color='green', linestyle='--',
                       linewidth=2, label='95% variance')
        axes[1].axhline(y=0.90, color='orange', linestyle='--',
                       linewidth=2, label='90% variance')
        axes[1].set_xlabel('Number of Components', fontsize=11)
        axes[1].set_ylabel('Cumulative Explained Variance', fontsize=11)
        axes[1].set_title('Cumulative Explained Variance', fontsize=12, fontweight='bold')
        axes[1].legend(fontsize=10)
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        return fig

    def visualize_2d_projection(self, X_transformed: np.ndarray,
                                y: Optional[np.ndarray] = None,
                                title: str = "2D Projection") -> plt.Figure:
        """
        Visualize 2D projection of data.

        Args:
            X_transformed: Transformed data (n_samples, 2)
            y: Optional labels for coloring
            title: Plot title

        Returns:
            Matplotlib figure
        """
        if X_transformed.shape[1] != 2:
            raise ValueError("Data must be 2-dimensional")

        fig, ax = plt.subplots(figsize=(10, 8))

        if y is not None:
            # Color by labels
            unique_labels = np.unique(y)
            colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

            for label, color in zip(unique_labels, colors):
                mask = y == label
                ax.scatter(X_transformed[mask, 0], X_transformed[mask, 1],
                          c=[color], label=f'Class {label}', alpha=0.6,
                          s=50, edgecolors='black', linewidths=0.5)
            ax.legend(fontsize=10)
        else:
            ax.scatter(X_transformed[:, 0], X_transformed[:, 1],
                      alpha=0.6, s=50, edgecolors='black', linewidths=0.5)

        ax.set_xlabel('Component 1', fontsize=12)
        ax.set_ylabel('Component 2', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)

        plt.tight_layout()
        return fig

    def visualize_3d_projection(self, X_transformed: np.ndarray,
                                y: Optional[np.ndarray] = None,
                                title: str = "3D Projection") -> plt.Figure:
        """
        Visualize 3D projection of data.

        Args:
            X_transformed: Transformed data (n_samples, 3)
            y: Optional labels for coloring
            title: Plot title

        Returns:
            Matplotlib figure
        """
        if X_transformed.shape[1] != 3:
            raise ValueError("Data must be 3-dimensional")

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        if y is not None:
            # Color by labels
            unique_labels = np.unique(y)
            colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

            for label, color in zip(unique_labels, colors):
                mask = y == label
                ax.scatter(X_transformed[mask, 0], X_transformed[mask, 1],
                          X_transformed[mask, 2], c=[color],
                          label=f'Class {label}', alpha=0.6, s=50,
                          edgecolors='black', linewidths=0.5)
            ax.legend(fontsize=9)
        else:
            ax.scatter(X_transformed[:, 0], X_transformed[:, 1],
                      X_transformed[:, 2], alpha=0.6, s=50,
                      edgecolors='black', linewidths=0.5)

        ax.set_xlabel('Component 1', fontsize=11)
        ax.set_ylabel('Component 2', fontsize=11)
        ax.set_zlabel('Component 3', fontsize=11)
        ax.set_title(title, fontsize=13, fontweight='bold')

        plt.tight_layout()
        return fig

    def visualize_component_loadings(self, components: np.ndarray,
                                     feature_names: Optional[List[str]] = None,
                                     n_components: int = 2) -> plt.Figure:
        """
        Visualize component loadings (feature contributions).

        Args:
            components: Component matrix (n_components, n_features)
            feature_names: Names of features
            n_components: Number of components to visualize

        Returns:
            Matplotlib figure
        """
        n_features = components.shape[1]
        n_comp = min(n_components, components.shape[0])

        if feature_names is None:
            feature_names = [f'Feature {i}' for i in range(n_features)]

        fig, ax = plt.subplots(figsize=(12, 6))

        # Create heatmap
        sns.heatmap(components[:n_comp], cmap='RdBu_r', center=0,
                   xticklabels=feature_names, yticklabels=[f'PC{i+1}' for i in range(n_comp)],
                   cbar_kws={'label': 'Loading'}, ax=ax)

        ax.set_title('Component Loadings (Feature Contributions)', fontsize=13, fontweight='bold')
        plt.tight_layout()
        return fig

    def visualize_reconstruction_analysis(self, analysis_result: Dict) -> plt.Figure:
        """
        Visualize reconstruction error and explained variance analysis.

        Args:
            analysis_result: Result from optimal_n_components_analysis()

        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Explained variance
        axes[0].plot(analysis_result['n_components_range'],
                    analysis_result['explained_variances'],
                    'o-', linewidth=2, markersize=8, color='blue')
        axes[0].axhline(y=0.95, color='green', linestyle='--',
                       linewidth=2, label='95% variance')
        axes[0].set_xlabel('Number of Components', fontsize=11)
        axes[0].set_ylabel('Cumulative Explained Variance', fontsize=11)
        axes[0].set_title('Explained Variance vs Components', fontsize=12, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(alpha=0.3)

        # Reconstruction error
        axes[1].plot(analysis_result['n_components_range'],
                    analysis_result['reconstruction_errors'],
                    'o-', linewidth=2, markersize=8, color='red')
        axes[1].set_xlabel('Number of Components', fontsize=11)
        axes[1].set_ylabel('Reconstruction Error (MSE)', fontsize=11)
        axes[1].set_title('Reconstruction Error vs Components', fontsize=12, fontweight='bold')
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        return fig


def demo():
    """Demonstrate dimensionality reduction toolkit."""
    np.random.seed(42)

    print("Dimensionality Reduction Toolkit Demo")
    print("=" * 70)

    # Generate synthetic high-dimensional data
    print("\nGenerating synthetic high-dimensional data...")
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=500,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=42
    )

    print(f"Data shape: {X.shape}")
    print(f"Number of classes: {len(np.unique(y))}")

    reducer = DimensionalityReducer(random_state=42)

    # 1. Optimal Components Analysis
    print("\n1. Analyzing Optimal Number of Components")
    print("-" * 70)
    analysis = reducer.optimal_n_components_analysis(X, max_components=15)

    # Find components for 95% variance
    idx_95 = np.argmax(np.array(analysis['explained_variances']) >= 0.95)
    print(f"Components needed for 95% variance: {analysis['n_components_range'][idx_95]}")

    # Visualize
    fig1 = reducer.visualize_reconstruction_analysis(analysis)
    fig1.savefig('dimreduction_optimal_components.png', dpi=300, bbox_inches='tight')
    print("✓ Saved dimreduction_optimal_components.png")
    plt.close()

    # 2. PCA
    print("\n2. Principal Component Analysis (PCA)")
    print("-" * 70)
    pca_result = reducer.pca_reduction(X, n_components=2)
    print(f"Number of components: {pca_result['n_components']}")
    print(f"Total variance explained: {pca_result['total_variance_explained']:.4f}")
    print(f"Explained variance ratio: {pca_result['explained_variance_ratio']}")

    # Reconstruction error
    pca_full = reducer.pca_reduction(X, n_components=None, variance_ratio=0.95)
    recon_error = reducer.reconstruction_error(X, 'pca')
    print(f"Reconstruction error (95% variance): {recon_error:.6f}")

    # Scree plot
    pca_all = reducer.pca_reduction(X, n_components=10)
    fig2 = reducer.visualize_scree_plot(pca_all['explained_variance_ratio'],
                                       title='PCA Scree Plot')
    fig2.savefig('dimreduction_scree_plot.png', dpi=300, bbox_inches='tight')
    print("✓ Saved dimreduction_scree_plot.png")
    plt.close()

    # 2D visualization
    fig3 = reducer.visualize_2d_projection(pca_result['transformed_data'], y,
                                          title='PCA 2D Projection')
    fig3.savefig('dimreduction_pca_2d.png', dpi=300, bbox_inches='tight')
    print("✓ Saved dimreduction_pca_2d.png")
    plt.close()

    # 3D PCA
    pca_3d = reducer.pca_reduction(X, n_components=3)
    fig4 = reducer.visualize_3d_projection(pca_3d['transformed_data'], y,
                                          title='PCA 3D Projection')
    fig4.savefig('dimreduction_pca_3d.png', dpi=300, bbox_inches='tight')
    print("✓ Saved dimreduction_pca_3d.png")
    plt.close()

    # Component loadings
    fig5 = reducer.visualize_component_loadings(pca_result['components'],
                                               feature_names=[f'F{i}' for i in range(X.shape[1])],
                                               n_components=2)
    fig5.savefig('dimreduction_loadings.png', dpi=300, bbox_inches='tight')
    print("✓ Saved dimreduction_loadings.png")
    plt.close()

    # 3. t-SNE
    print("\n3. t-SNE Dimensionality Reduction")
    print("-" * 70)
    tsne_result = reducer.tsne_reduction(X, n_components=2, perplexity=30, n_iter=1000)
    print(f"Number of components: {tsne_result['n_components']}")
    print(f"Perplexity: {tsne_result['perplexity']}")
    print(f"KL divergence: {tsne_result['kl_divergence']:.4f}")

    fig6 = reducer.visualize_2d_projection(tsne_result['transformed_data'], y,
                                          title='t-SNE 2D Projection')
    fig6.savefig('dimreduction_tsne_2d.png', dpi=300, bbox_inches='tight')
    print("✓ Saved dimreduction_tsne_2d.png")
    plt.close()

    # 4. Factor Analysis
    print("\n4. Factor Analysis")
    print("-" * 70)
    fa_result = reducer.factor_analysis(X, n_components=2)
    print(f"Number of factors: {fa_result['n_components']}")
    print(f"Iterations: {fa_result['n_iter']}")
    print(f"Log-likelihood: {fa_result['log_likelihood']:.2f}")

    recon_error_fa = reducer.reconstruction_error(X, 'factor_analysis')
    print(f"Reconstruction error: {recon_error_fa:.6f}")

    fig7 = reducer.visualize_2d_projection(fa_result['transformed_data'], y,
                                          title='Factor Analysis 2D Projection')
    fig7.savefig('dimreduction_fa_2d.png', dpi=300, bbox_inches='tight')
    print("✓ Saved dimreduction_fa_2d.png")
    plt.close()

    # 5. Truncated SVD
    print("\n5. Truncated SVD (LSA)")
    print("-" * 70)
    svd_result = reducer.truncated_svd(X, n_components=2)
    print(f"Number of components: {svd_result['n_components']}")
    print(f"Total variance explained: {svd_result['total_variance_explained']:.4f}")
    print(f"Explained variance ratio: {svd_result['explained_variance_ratio']}")

    recon_error_svd = reducer.reconstruction_error(X, 'truncated_svd')
    print(f"Reconstruction error: {recon_error_svd:.6f}")

    fig8 = reducer.visualize_2d_projection(svd_result['transformed_data'], y,
                                          title='Truncated SVD 2D Projection')
    fig8.savefig('dimreduction_svd_2d.png', dpi=300, bbox_inches='tight')
    print("✓ Saved dimreduction_svd_2d.png")
    plt.close()

    # 6. Linear Discriminant Analysis (LDA)
    print("\n6. Linear Discriminant Analysis (Supervised)")
    print("-" * 70)
    lda_result = reducer.lda_reduction(X, y, n_components=2)
    print(f"Number of components: {lda_result['n_components']}")
    print(f"Total variance explained: {lda_result['total_variance_explained']:.4f}")
    print(f"Explained variance ratio: {lda_result['explained_variance_ratio']}")

    fig9 = reducer.visualize_2d_projection(lda_result['transformed_data'], y,
                                          title='LDA 2D Projection (Supervised)')
    fig9.savefig('dimreduction_lda_2d.png', dpi=300, bbox_inches='tight')
    print("✓ Saved dimreduction_lda_2d.png")
    plt.close()

    # 7. Method Comparison
    print("\n7. Method Comparison")
    print("-" * 70)
    print(f"{'Method':<20} {'Components':<12} {'Variance Explained':<20} {'Reconstruction Error':<25}")
    print("-" * 70)
    print(f"{'PCA':<20} {pca_result['n_components']:<12} {pca_result['total_variance_explained']:<20.4f} {recon_error:<25.6f}")
    print(f"{'t-SNE':<20} {tsne_result['n_components']:<12} {'N/A':<20} {'N/A':<25}")
    print(f"{'Factor Analysis':<20} {fa_result['n_components']:<12} {'N/A':<20} {recon_error_fa:<25.6f}")
    print(f"{'Truncated SVD':<20} {svd_result['n_components']:<12} {svd_result['total_variance_explained']:<20.4f} {recon_error_svd:<25.6f}")
    print(f"{'LDA (Supervised)':<20} {lda_result['n_components']:<12} {lda_result['total_variance_explained']:<20.4f} {'N/A':<25}")

    # 8. Visual Comparison
    print("\n8. Visual Comparison of Methods")
    print("-" * 70)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    methods_data = [
        (pca_result['transformed_data'], 'PCA'),
        (tsne_result['transformed_data'], 't-SNE'),
        (fa_result['transformed_data'], 'Factor Analysis'),
        (svd_result['transformed_data'], 'Truncated SVD'),
        (lda_result['transformed_data'], 'LDA'),
    ]

    for idx, (data, method) in enumerate(methods_data):
        ax = axes[idx]
        unique_labels = np.unique(y)
        colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

        for label, color in zip(unique_labels, colors):
            mask = y == label
            ax.scatter(data[mask, 0], data[mask, 1], c=[color],
                      label=f'Class {label}', alpha=0.6, s=30,
                      edgecolors='black', linewidths=0.5)

        ax.set_xlabel('Component 1', fontsize=10)
        ax.set_ylabel('Component 2', fontsize=10)
        ax.set_title(method, fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    # Hide unused subplot
    axes[5].axis('off')

    plt.tight_layout()
    fig.savefig('dimreduction_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved dimreduction_comparison.png")
    plt.close()

    print("\n" + "=" * 70)
    print("✓ Dimensionality Reduction Demo Complete!")
    print("=" * 70)


if __name__ == '__main__':
    demo()
