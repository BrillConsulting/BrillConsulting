"""
Synthetic Data Generation Toolkit
==================================

Comprehensive synthetic data generation with multiple methods:
- Gaussian Copula for preserving correlations
- SMOTE-style synthetic generation
- VAE-style generation (simple encoder-decoder)
- Statistical distribution fitting
- Privacy-preserving synthesis
- Time series synthetic generation
- Correlation preservation
- Statistical fidelity metrics
- Data augmentation techniques

Author: Brill Consulting
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm, multivariate_normal
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')


class SyntheticDataGenerator:
    """Comprehensive synthetic data generation toolkit with multiple methods."""

    def __init__(self, random_state: int = 42):
        """
        Initialize synthetic data generator.

        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        np.random.seed(random_state)
        self.scaler = StandardScaler()
        self.fitted_distributions = {}
        self.correlation_matrix = None

    def gaussian_copula(self, X: np.ndarray, n_samples: int) -> np.ndarray:
        """
        Generate synthetic data using Gaussian Copula to preserve correlations.

        The Gaussian copula method:
        1. Transform each feature to normal distribution
        2. Model correlations in normal space
        3. Generate new samples with same correlation structure
        4. Transform back to original marginal distributions

        Args:
            X: Original data (n_samples, n_features)
            n_samples: Number of synthetic samples to generate

        Returns:
            Synthetic data with preserved correlation structure
        """
        n_features = X.shape[1]

        # Step 1: Transform to uniform [0, 1] using empirical CDF
        uniform_data = np.zeros_like(X)
        for i in range(n_features):
            # Rank-based transformation
            ranks = stats.rankdata(X[:, i])
            uniform_data[:, i] = ranks / (len(ranks) + 1)

        # Step 2: Transform to standard normal
        normal_data = norm.ppf(uniform_data)

        # Step 3: Calculate correlation matrix
        self.correlation_matrix = np.corrcoef(normal_data.T)

        # Handle numerical issues
        self.correlation_matrix = np.nan_to_num(self.correlation_matrix)
        # Ensure positive semi-definite
        eigenvalues, eigenvectors = np.linalg.eigh(self.correlation_matrix)
        eigenvalues = np.maximum(eigenvalues, 1e-10)
        self.correlation_matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

        # Step 4: Generate new samples in normal space
        mean = np.zeros(n_features)
        synthetic_normal = np.random.multivariate_normal(mean, self.correlation_matrix, n_samples)

        # Step 5: Transform to uniform
        synthetic_uniform = norm.cdf(synthetic_normal)

        # Step 6: Transform to original marginal distributions
        synthetic_data = np.zeros_like(synthetic_uniform)
        for i in range(n_features):
            # Use empirical quantile function
            sorted_values = np.sort(X[:, i])
            # Interpolate to get values at uniform quantiles
            synthetic_data[:, i] = np.interp(
                synthetic_uniform[:, i],
                np.linspace(0, 1, len(sorted_values)),
                sorted_values
            )

        return synthetic_data

    def smote_generation(self, X: np.ndarray, n_samples: int,
                        k_neighbors: int = 5) -> np.ndarray:
        """
        Generate synthetic samples using SMOTE-style interpolation.

        Args:
            X: Original data (n_samples, n_features)
            n_samples: Number of synthetic samples to generate
            k_neighbors: Number of nearest neighbors to use

        Returns:
            Synthetic data generated via interpolation
        """
        # Fit nearest neighbors
        nn = NearestNeighbors(n_neighbors=k_neighbors + 1)
        nn.fit(X)

        synthetic_samples = []

        for _ in range(n_samples):
            # Randomly select a sample
            idx = np.random.randint(0, len(X))
            sample = X[idx]

            # Find k nearest neighbors
            distances, indices = nn.kneighbors([sample])
            # Exclude the sample itself
            neighbor_indices = indices[0][1:]

            # Randomly select one neighbor
            neighbor_idx = np.random.choice(neighbor_indices)
            neighbor = X[neighbor_idx]

            # Generate synthetic sample via interpolation
            alpha = np.random.random()
            synthetic_sample = sample + alpha * (neighbor - sample)
            synthetic_samples.append(synthetic_sample)

        return np.array(synthetic_samples)

    def vae_generation(self, X: np.ndarray, n_samples: int,
                      latent_dim: int = 10, n_epochs: int = 100) -> np.ndarray:
        """
        Generate synthetic data using VAE-style encoder-decoder approach.

        Simplified VAE using PCA for dimensionality reduction and
        Gaussian modeling in latent space.

        Args:
            X: Original data (n_samples, n_features)
            n_samples: Number of synthetic samples to generate
            latent_dim: Dimensionality of latent space
            n_epochs: Number of training epochs (for consistency, not used in PCA)

        Returns:
            Synthetic data from latent space
        """
        # Standardize data
        X_scaled = self.scaler.fit_transform(X)

        # Encode: Use PCA to compress to latent space
        latent_dim = min(latent_dim, X.shape[1], X.shape[0] - 1)
        pca = PCA(n_components=latent_dim, random_state=self.random_state)
        latent_codes = pca.fit_transform(X_scaled)

        # Model latent space as multivariate Gaussian
        latent_mean = np.mean(latent_codes, axis=0)
        latent_cov = np.cov(latent_codes.T)

        # Add small regularization for numerical stability
        latent_cov += np.eye(latent_dim) * 1e-6

        # Sample from latent space
        synthetic_latent = np.random.multivariate_normal(
            latent_mean, latent_cov, n_samples
        )

        # Decode: Transform back to original space
        synthetic_scaled = pca.inverse_transform(synthetic_latent)
        synthetic_data = self.scaler.inverse_transform(synthetic_scaled)

        return synthetic_data

    def distribution_fitting(self, X: np.ndarray, n_samples: int,
                            distributions: Optional[List[str]] = None) -> np.ndarray:
        """
        Generate synthetic data by fitting statistical distributions to each feature.

        Args:
            X: Original data (n_samples, n_features)
            n_samples: Number of synthetic samples to generate
            distributions: List of distribution names to try ('norm', 'lognorm', 'gamma', etc.)

        Returns:
            Synthetic data from fitted distributions
        """
        if distributions is None:
            distributions = ['norm', 'lognorm', 'gamma', 'beta']

        n_features = X.shape[1]
        synthetic_data = np.zeros((n_samples, n_features))

        for i in range(n_features):
            feature_data = X[:, i]

            # Try to fit multiple distributions and choose the best
            best_dist = None
            best_params = None
            best_ks_stat = np.inf

            for dist_name in distributions:
                try:
                    dist = getattr(stats, dist_name)

                    # Fit distribution
                    params = dist.fit(feature_data)

                    # Kolmogorov-Smirnov test for goodness of fit
                    ks_stat, _ = stats.kstest(feature_data, dist_name, args=params)

                    if ks_stat < best_ks_stat:
                        best_ks_stat = ks_stat
                        best_dist = dist
                        best_params = params

                except Exception:
                    continue

            # Generate samples from best distribution
            if best_dist is not None:
                synthetic_data[:, i] = best_dist.rvs(*best_params, size=n_samples)
                self.fitted_distributions[i] = {
                    'distribution': best_dist.name,
                    'params': best_params,
                    'ks_statistic': best_ks_stat
                }
            else:
                # Fallback to empirical distribution (resampling)
                synthetic_data[:, i] = np.random.choice(feature_data, size=n_samples)

        return synthetic_data

    def privacy_preserving_synthesis(self, X: np.ndarray, n_samples: int,
                                    epsilon: float = 1.0) -> np.ndarray:
        """
        Generate privacy-preserving synthetic data using differential privacy.

        Adds calibrated noise to statistics before generating synthetic data.

        Args:
            X: Original data (n_samples, n_features)
            n_samples: Number of synthetic samples to generate
            epsilon: Privacy budget (smaller = more privacy, more noise)

        Returns:
            Privacy-preserving synthetic data
        """
        n_features = X.shape[1]
        n_original = len(X)

        # Calculate noisy statistics
        sensitivity = 2.0 / n_original  # Sensitivity for mean
        noise_scale = sensitivity / epsilon

        # Noisy mean
        true_mean = np.mean(X, axis=0)
        noisy_mean = true_mean + np.random.laplace(0, noise_scale, n_features)

        # Noisy covariance
        true_cov = np.cov(X.T)
        # Add noise to covariance matrix
        noise_cov = np.random.laplace(0, noise_scale * 2, true_cov.shape)
        noisy_cov = true_cov + noise_cov

        # Ensure covariance is positive semi-definite
        eigenvalues, eigenvectors = np.linalg.eigh(noisy_cov)
        eigenvalues = np.maximum(eigenvalues, 1e-6)
        noisy_cov = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

        # Generate synthetic data from noisy statistics
        synthetic_data = np.random.multivariate_normal(noisy_mean, noisy_cov, n_samples)

        return synthetic_data

    def time_series_generation(self, X: np.ndarray, n_samples: int,
                              method: str = 'ar') -> np.ndarray:
        """
        Generate synthetic time series data.

        Args:
            X: Original time series data (n_timepoints, n_features)
            n_samples: Number of synthetic time series samples to generate
            method: Generation method ('ar' for autoregressive, 'bootstrap' for block bootstrap)

        Returns:
            Synthetic time series data
        """
        n_timepoints, n_features = X.shape

        if method == 'ar':
            # Autoregressive model
            synthetic_data = np.zeros((n_samples, n_timepoints, n_features))

            for feat in range(n_features):
                feature_data = X[:, feat]

                # Fit AR model (simple AR(1))
                if len(feature_data) > 1:
                    # Calculate AR(1) coefficient
                    phi = np.corrcoef(feature_data[:-1], feature_data[1:])[0, 1]
                    mu = np.mean(feature_data)
                    sigma = np.std(feature_data[1:] - phi * feature_data[:-1])

                    # Generate synthetic series
                    for s in range(n_samples):
                        series = np.zeros(n_timepoints)
                        series[0] = np.random.normal(mu, np.std(feature_data))

                        for t in range(1, n_timepoints):
                            series[t] = mu + phi * (series[t-1] - mu) + np.random.normal(0, sigma)

                        synthetic_data[s, :, feat] = series

        elif method == 'bootstrap':
            # Block bootstrap
            block_size = min(10, n_timepoints // 5)
            synthetic_data = np.zeros((n_samples, n_timepoints, n_features))

            for s in range(n_samples):
                current_pos = 0
                while current_pos < n_timepoints:
                    # Randomly select a block
                    start_idx = np.random.randint(0, max(1, n_timepoints - block_size))
                    end_idx = min(start_idx + block_size, n_timepoints)
                    block_length = end_idx - start_idx

                    # Copy block
                    copy_length = min(block_length, n_timepoints - current_pos)
                    synthetic_data[s, current_pos:current_pos + copy_length, :] = \
                        X[start_idx:start_idx + copy_length, :]

                    current_pos += copy_length

        else:
            raise ValueError(f"Unknown method: {method}")

        return synthetic_data

    def data_augmentation(self, X: np.ndarray, n_samples: int,
                         noise_level: float = 0.1,
                         augmentation_types: Optional[List[str]] = None) -> np.ndarray:
        """
        Augment data with various transformations.

        Args:
            X: Original data (n_samples, n_features)
            n_samples: Number of augmented samples to generate
            noise_level: Level of noise/perturbation to add
            augmentation_types: Types of augmentation ('noise', 'scale', 'rotate')

        Returns:
            Augmented data
        """
        if augmentation_types is None:
            augmentation_types = ['noise', 'scale']

        augmented_samples = []

        for _ in range(n_samples):
            # Randomly select original sample
            idx = np.random.randint(0, len(X))
            sample = X[idx].copy()

            # Apply random augmentations
            aug_type = np.random.choice(augmentation_types)

            if aug_type == 'noise':
                # Add Gaussian noise
                noise = np.random.randn(*sample.shape) * noise_level * np.std(X, axis=0)
                sample = sample + noise

            elif aug_type == 'scale':
                # Random scaling
                scale = 1.0 + np.random.uniform(-noise_level, noise_level)
                sample = sample * scale

            elif aug_type == 'rotate':
                # Random rotation (for 2D+ data)
                if len(sample) >= 2:
                    angle = np.random.uniform(-noise_level * np.pi, noise_level * np.pi)
                    rotation_matrix = np.array([
                        [np.cos(angle), -np.sin(angle)],
                        [np.sin(angle), np.cos(angle)]
                    ])
                    sample[:2] = rotation_matrix @ sample[:2]

            augmented_samples.append(sample)

        return np.array(augmented_samples)

    def evaluate_fidelity(self, original: np.ndarray,
                         synthetic: np.ndarray) -> Dict:
        """
        Evaluate statistical fidelity of synthetic data.

        Args:
            original: Original data
            synthetic: Synthetic data

        Returns:
            Dictionary with fidelity metrics
        """
        metrics = {}

        # 1. Kolmogorov-Smirnov test for each feature
        ks_statistics = []
        ks_pvalues = []

        for i in range(original.shape[1]):
            ks_stat, p_value = stats.ks_2samp(original[:, i], synthetic[:, i])
            ks_statistics.append(ks_stat)
            ks_pvalues.append(p_value)

        metrics['ks_statistics'] = ks_statistics
        metrics['ks_pvalues'] = ks_pvalues
        metrics['avg_ks_statistic'] = np.mean(ks_statistics)
        metrics['features_passed_ks'] = np.sum(np.array(ks_pvalues) > 0.05)

        # 2. Correlation comparison
        original_corr = np.corrcoef(original.T)
        synthetic_corr = np.corrcoef(synthetic.T)

        corr_difference = np.abs(original_corr - synthetic_corr)
        metrics['correlation_difference'] = np.mean(corr_difference[np.triu_indices_from(corr_difference, k=1)])
        metrics['max_correlation_difference'] = np.max(corr_difference[np.triu_indices_from(corr_difference, k=1)])

        # 3. Mean and standard deviation comparison
        mean_diff = np.abs(np.mean(original, axis=0) - np.mean(synthetic, axis=0))
        std_diff = np.abs(np.std(original, axis=0) - np.std(synthetic, axis=0))

        metrics['mean_difference'] = np.mean(mean_diff)
        metrics['std_difference'] = np.mean(std_diff)

        # 4. Distribution distance (Wasserstein)
        wasserstein_distances = []
        for i in range(original.shape[1]):
            w_dist = stats.wasserstein_distance(original[:, i], synthetic[:, i])
            wasserstein_distances.append(w_dist)

        metrics['wasserstein_distances'] = wasserstein_distances
        metrics['avg_wasserstein_distance'] = np.mean(wasserstein_distances)

        # 5. Overall fidelity score (0-1, higher is better)
        # Combine multiple metrics
        ks_score = 1.0 - metrics['avg_ks_statistic']
        corr_score = 1.0 - min(metrics['correlation_difference'], 1.0)
        mean_score = 1.0 - min(metrics['mean_difference'] / (np.mean(np.abs(np.mean(original, axis=0))) + 1e-10), 1.0)

        metrics['fidelity_score'] = (ks_score + corr_score + mean_score) / 3.0

        return metrics

    def visualize_comparison(self, original: np.ndarray, synthetic: np.ndarray,
                           feature_names: Optional[List[str]] = None,
                           max_features: int = 4) -> plt.Figure:
        """
        Visualize comparison between original and synthetic data.

        Args:
            original: Original data
            synthetic: Synthetic data
            feature_names: Names of features
            max_features: Maximum number of features to plot

        Returns:
            Matplotlib figure
        """
        n_features = min(original.shape[1], max_features)

        if feature_names is None:
            feature_names = [f'Feature {i}' for i in range(n_features)]

        fig, axes = plt.subplots(2, n_features, figsize=(5 * n_features, 8))
        if n_features == 1:
            axes = axes.reshape(-1, 1)

        for i in range(n_features):
            # Histogram comparison
            axes[0, i].hist(original[:, i], bins=50, alpha=0.6, label='Original',
                          color='blue', edgecolor='black', density=True)
            axes[0, i].hist(synthetic[:, i], bins=50, alpha=0.6, label='Synthetic',
                          color='red', edgecolor='black', density=True)
            axes[0, i].set_xlabel(feature_names[i], fontsize=11)
            axes[0, i].set_ylabel('Density', fontsize=11)
            axes[0, i].set_title(f'{feature_names[i]} Distribution', fontsize=12, fontweight='bold')
            axes[0, i].legend(fontsize=10)
            axes[0, i].grid(alpha=0.3)

            # Q-Q plot
            original_sorted = np.sort(original[:, i])
            synthetic_sorted = np.sort(synthetic[:, i])

            # Match lengths for Q-Q plot
            n_quantiles = min(len(original_sorted), len(synthetic_sorted))
            original_quantiles = np.interp(
                np.linspace(0, 1, n_quantiles),
                np.linspace(0, 1, len(original_sorted)),
                original_sorted
            )
            synthetic_quantiles = np.interp(
                np.linspace(0, 1, n_quantiles),
                np.linspace(0, 1, len(synthetic_sorted)),
                synthetic_sorted
            )

            axes[1, i].scatter(original_quantiles, synthetic_quantiles, alpha=0.5, s=10)
            min_val = min(original_quantiles.min(), synthetic_quantiles.min())
            max_val = max(original_quantiles.max(), synthetic_quantiles.max())
            axes[1, i].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Match')
            axes[1, i].set_xlabel('Original Quantiles', fontsize=11)
            axes[1, i].set_ylabel('Synthetic Quantiles', fontsize=11)
            axes[1, i].set_title(f'{feature_names[i]} Q-Q Plot', fontsize=12, fontweight='bold')
            axes[1, i].legend(fontsize=9)
            axes[1, i].grid(alpha=0.3)

        plt.tight_layout()
        return fig

    def visualize_correlation_comparison(self, original: np.ndarray,
                                        synthetic: np.ndarray) -> plt.Figure:
        """
        Visualize correlation matrix comparison.

        Args:
            original: Original data
            synthetic: Synthetic data

        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Original correlation
        original_corr = np.corrcoef(original.T)
        sns.heatmap(original_corr, annot=True, fmt='.2f', cmap='coolwarm',
                   center=0, ax=axes[0], cbar_kws={'label': 'Correlation'})
        axes[0].set_title('Original Data Correlation', fontsize=13, fontweight='bold')

        # Synthetic correlation
        synthetic_corr = np.corrcoef(synthetic.T)
        sns.heatmap(synthetic_corr, annot=True, fmt='.2f', cmap='coolwarm',
                   center=0, ax=axes[1], cbar_kws={'label': 'Correlation'})
        axes[1].set_title('Synthetic Data Correlation', fontsize=13, fontweight='bold')

        # Difference
        corr_diff = np.abs(original_corr - synthetic_corr)
        sns.heatmap(corr_diff, annot=True, fmt='.2f', cmap='Reds',
                   ax=axes[2], cbar_kws={'label': 'Absolute Difference'})
        axes[2].set_title('Correlation Difference', fontsize=13, fontweight='bold')

        plt.tight_layout()
        return fig


def demo():
    """Demonstrate synthetic data generation toolkit."""
    np.random.seed(42)

    print("Synthetic Data Generation Toolkit Demo")
    print("=" * 80)

    # 1. Generate original data
    print("\n1. Generating Original Data")
    print("-" * 80)

    n_samples = 500
    n_features = 4

    # Create correlated data
    mean = [0, 5, 10, -3]
    cov = [[1.0, 0.7, 0.3, 0.1],
           [0.7, 2.0, 0.5, 0.2],
           [0.3, 0.5, 1.5, 0.4],
           [0.1, 0.2, 0.4, 1.0]]

    original_data = np.random.multivariate_normal(mean, cov, n_samples)

    print(f"Original data shape: {original_data.shape}")
    print(f"Original mean: {np.mean(original_data, axis=0)}")
    print(f"Original std: {np.std(original_data, axis=0)}")

    generator = SyntheticDataGenerator(random_state=42)

    # 2. Gaussian Copula
    print("\n2. Gaussian Copula Generation")
    print("-" * 80)
    synthetic_copula = generator.gaussian_copula(original_data, n_samples=300)
    print(f"Synthetic data shape: {synthetic_copula.shape}")
    print(f"Synthetic mean: {np.mean(synthetic_copula, axis=0)}")
    print(f"Synthetic std: {np.std(synthetic_copula, axis=0)}")

    # 3. SMOTE-style Generation
    print("\n3. SMOTE-style Generation")
    print("-" * 80)
    synthetic_smote = generator.smote_generation(original_data, n_samples=300, k_neighbors=5)
    print(f"Synthetic data shape: {synthetic_smote.shape}")
    print(f"Generated via interpolation between nearest neighbors")

    # 4. VAE-style Generation
    print("\n4. VAE-style Generation")
    print("-" * 80)
    synthetic_vae = generator.vae_generation(original_data, n_samples=300, latent_dim=3)
    print(f"Synthetic data shape: {synthetic_vae.shape}")
    print(f"Generated from {3}-dimensional latent space")

    # 5. Distribution Fitting
    print("\n5. Distribution Fitting Generation")
    print("-" * 80)
    synthetic_dist = generator.distribution_fitting(original_data, n_samples=300)
    print(f"Synthetic data shape: {synthetic_dist.shape}")
    print("\nFitted distributions:")
    for feat_idx, dist_info in generator.fitted_distributions.items():
        print(f"  Feature {feat_idx}: {dist_info['distribution']} (KS stat: {dist_info['ks_statistic']:.4f})")

    # 6. Privacy-Preserving Synthesis
    print("\n6. Privacy-Preserving Synthesis")
    print("-" * 80)
    synthetic_private = generator.privacy_preserving_synthesis(
        original_data, n_samples=300, epsilon=1.0
    )
    print(f"Synthetic data shape: {synthetic_private.shape}")
    print(f"Privacy budget (epsilon): 1.0")
    print("Note: Added differential privacy noise to protect individual records")

    # 7. Time Series Generation
    print("\n7. Time Series Generation")
    print("-" * 80)
    # Create simple time series
    time_series = np.random.randn(100, 2).cumsum(axis=0)
    synthetic_ts = generator.time_series_generation(time_series, n_samples=5, method='ar')
    print(f"Original time series shape: {time_series.shape}")
    print(f"Synthetic time series shape: {synthetic_ts.shape}")
    print(f"Generated {synthetic_ts.shape[0]} time series using AR model")

    # 8. Data Augmentation
    print("\n8. Data Augmentation")
    print("-" * 80)
    augmented_data = generator.data_augmentation(
        original_data, n_samples=300, noise_level=0.1,
        augmentation_types=['noise', 'scale']
    )
    print(f"Augmented data shape: {augmented_data.shape}")
    print("Applied random noise and scaling augmentations")

    # 9. Evaluate Fidelity
    print("\n9. Statistical Fidelity Evaluation")
    print("-" * 80)

    methods = {
        'Gaussian Copula': synthetic_copula,
        'SMOTE': synthetic_smote,
        'VAE': synthetic_vae,
        'Distribution Fitting': synthetic_dist,
        'Privacy-Preserving': synthetic_private,
        'Data Augmentation': augmented_data
    }

    print(f"{'Method':<25} {'Fidelity':<12} {'Corr Diff':<12} {'KS Stat':<12}")
    print("-" * 80)

    fidelity_results = {}
    for method_name, synthetic_data in methods.items():
        metrics = generator.evaluate_fidelity(original_data, synthetic_data)
        fidelity_results[method_name] = metrics

        print(f"{method_name:<25} {metrics['fidelity_score']:<12.4f} "
              f"{metrics['correlation_difference']:<12.4f} "
              f"{metrics['avg_ks_statistic']:<12.4f}")

    # 10. Visualizations
    print("\n10. Generating Visualizations")
    print("-" * 80)

    # Distribution comparison
    fig1 = generator.visualize_comparison(
        original_data, synthetic_copula,
        feature_names=['Feature 0', 'Feature 1', 'Feature 2', 'Feature 3']
    )
    fig1.savefig('synthetic_distribution_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved synthetic_distribution_comparison.png")
    plt.close()

    # Correlation comparison
    fig2 = generator.visualize_correlation_comparison(original_data, synthetic_copula)
    fig2.savefig('synthetic_correlation_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved synthetic_correlation_comparison.png")
    plt.close()

    # Method comparison
    fig3, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for idx, (method_name, synthetic_data) in enumerate(methods.items()):
        ax = axes[idx]

        # Plot first two features
        ax.scatter(original_data[:, 0], original_data[:, 1],
                  alpha=0.5, s=20, label='Original', color='blue')
        ax.scatter(synthetic_data[:, 0], synthetic_data[:, 1],
                  alpha=0.5, s=20, label='Synthetic', color='red')
        ax.set_xlabel('Feature 0', fontsize=10)
        ax.set_ylabel('Feature 1', fontsize=10)
        ax.set_title(f'{method_name}\nFidelity: {fidelity_results[method_name]["fidelity_score"]:.3f}',
                    fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    fig3.savefig('synthetic_methods_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved synthetic_methods_comparison.png")
    plt.close()

    # Time series visualization
    fig4, axes = plt.subplots(2, 1, figsize=(14, 8))

    # Original time series
    axes[0].plot(time_series[:, 0], label='Feature 0', linewidth=2)
    axes[0].plot(time_series[:, 1], label='Feature 1', linewidth=2)
    axes[0].set_xlabel('Time', fontsize=11)
    axes[0].set_ylabel('Value', fontsize=11)
    axes[0].set_title('Original Time Series', fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(alpha=0.3)

    # Synthetic time series
    for i in range(min(3, synthetic_ts.shape[0])):
        axes[1].plot(synthetic_ts[i, :, 0], alpha=0.6, linewidth=1.5)
    axes[1].set_xlabel('Time', fontsize=11)
    axes[1].set_ylabel('Value', fontsize=11)
    axes[1].set_title('Synthetic Time Series (Feature 0)', fontsize=12, fontweight='bold')
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    fig4.savefig('synthetic_time_series.png', dpi=300, bbox_inches='tight')
    print("✓ Saved synthetic_time_series.png")
    plt.close()

    print("\n" + "=" * 80)
    print("✓ Synthetic Data Generation Demo Complete!")
    print("=" * 80)


if __name__ == '__main__':
    demo()
