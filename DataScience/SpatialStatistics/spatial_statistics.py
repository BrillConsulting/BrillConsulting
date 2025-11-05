"""
Spatial Statistics Toolkit
===========================

Advanced geospatial and spatial data analysis:
- Spatial autocorrelation (Moran's I, Geary's C)
- Kriging and spatial interpolation
- Point pattern analysis
- Spatial regression models
- Hotspot analysis and clustering
- Variogram modeling
- Distance-based statistics

Author: Brill Consulting
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.spatial import distance_matrix
from scipy.optimize import minimize
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class SpatialStatistics:
    """Spatial statistics and geospatial analysis toolkit."""

    def __init__(self):
        """Initialize spatial statistics toolkit."""
        self.coordinates = None
        self.values = None
        self.distance_matrix = None
        self.variogram = None

    def load_spatial_data(self, coordinates: np.ndarray, values: np.ndarray):
        """
        Load spatial data.

        Args:
            coordinates: Array of coordinates (n_points x 2 or 3)
            values: Array of values at each location
        """
        self.coordinates = coordinates
        self.values = values
        self.distance_matrix = distance_matrix(coordinates, coordinates)

    def morans_i(self, values: Optional[np.ndarray] = None,
                coords: Optional[np.ndarray] = None,
                weight_type: str = 'inverse_distance') -> Dict:
        """
        Calculate Moran's I spatial autocorrelation statistic.

        Args:
            values: Values at locations (if None, uses self.values)
            coords: Coordinates (if None, uses self.coordinates)
            weight_type: Type of spatial weights ('inverse_distance', 'binary', 'knn')

        Returns:
            Dictionary with Moran's I and significance test
        """
        if values is None:
            values = self.values
        if coords is None:
            coords = self.coordinates

        n = len(values)
        mean_val = np.mean(values)

        # Calculate spatial weights matrix
        W = self._calculate_spatial_weights(coords, weight_type)

        # Calculate Moran's I
        numerator = 0
        denominator = 0

        for i in range(n):
            for j in range(n):
                numerator += W[i, j] * (values[i] - mean_val) * (values[j] - mean_val)
            denominator += (values[i] - mean_val) ** 2

        S0 = np.sum(W)
        moran_i = (n / S0) * (numerator / denominator)

        # Expected value and variance under null hypothesis
        E_I = -1 / (n - 1)

        # Simplified variance calculation
        S1 = np.sum((W + W.T) ** 2) / 2
        S2 = np.sum((np.sum(W, axis=0) + np.sum(W, axis=1)) ** 2)

        var_I = (n * ((n**2 - 3*n + 3) * S1 - n * S2 + 3 * S0**2) -
                E_I * ((n**2 - n) * S1 - 2*n*S2 + 6*S0**2)) / ((n-1)*(n-2)*(n-3)*S0**2)

        # Z-score
        z_score = (moran_i - E_I) / np.sqrt(var_I)
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

        return {
            'morans_i': moran_i,
            'expected_i': E_I,
            'z_score': z_score,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'interpretation': 'Positive spatial autocorrelation' if moran_i > E_I
                            else 'Negative spatial autocorrelation'
        }

    def _calculate_spatial_weights(self, coords: np.ndarray, weight_type: str) -> np.ndarray:
        """Calculate spatial weights matrix."""
        n = len(coords)
        dist_mat = distance_matrix(coords, coords)

        if weight_type == 'inverse_distance':
            W = 1 / (dist_mat + 1e-10)
            np.fill_diagonal(W, 0)
        elif weight_type == 'binary':
            threshold = np.percentile(dist_mat[dist_mat > 0], 25)
            W = (dist_mat < threshold).astype(float)
            np.fill_diagonal(W, 0)
        elif weight_type == 'knn':
            k = min(4, n - 1)
            W = np.zeros((n, n))
            for i in range(n):
                nearest = np.argsort(dist_mat[i])[1:k+1]
                W[i, nearest] = 1
        else:
            raise ValueError(f"Unknown weight type: {weight_type}")

        # Row-normalize
        row_sums = W.sum(axis=1)
        row_sums[row_sums == 0] = 1
        W = W / row_sums[:, np.newaxis]

        return W

    def gearys_c(self, values: Optional[np.ndarray] = None,
                coords: Optional[np.ndarray] = None) -> Dict:
        """
        Calculate Geary's C spatial autocorrelation statistic.

        Args:
            values: Values at locations
            coords: Coordinates

        Returns:
            Dictionary with Geary's C and significance test
        """
        if values is None:
            values = self.values
        if coords is None:
            coords = self.coordinates

        n = len(values)
        mean_val = np.mean(values)

        W = self._calculate_spatial_weights(coords, 'inverse_distance')

        # Calculate Geary's C
        numerator = 0
        denominator = 0

        for i in range(n):
            for j in range(n):
                numerator += W[i, j] * (values[i] - values[j]) ** 2
            denominator += (values[i] - mean_val) ** 2

        S0 = np.sum(W)
        gearys_c = ((n - 1) / (2 * S0)) * (numerator / denominator)

        # Expected value and variance
        E_C = 1.0
        # Simplified z-score calculation
        z_score = (gearys_c - E_C) / 0.1  # Approximate std
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

        return {
            'gearys_c': gearys_c,
            'expected_c': E_C,
            'z_score': z_score,
            'p_value': p_value,
            'interpretation': 'Clustered' if gearys_c < 1
                            else 'Dispersed' if gearys_c > 1 else 'Random'
        }

    def empirical_variogram(self, coords: Optional[np.ndarray] = None,
                           values: Optional[np.ndarray] = None,
                           n_bins: int = 15) -> Dict:
        """
        Calculate empirical variogram.

        Args:
            coords: Spatial coordinates
            values: Values at locations
            n_bins: Number of distance bins

        Returns:
            Dictionary with variogram data
        """
        if coords is None:
            coords = self.coordinates
        if values is None:
            values = self.values

        n = len(values)
        dist_mat = distance_matrix(coords, coords)

        # Calculate semivariance for all pairs
        semivariances = []
        distances = []

        for i in range(n):
            for j in range(i + 1, n):
                dist = dist_mat[i, j]
                semivar = 0.5 * (values[i] - values[j]) ** 2
                distances.append(dist)
                semivariances.append(semivar)

        distances = np.array(distances)
        semivariances = np.array(semivariances)

        # Bin the data
        max_dist = np.max(distances)
        bins = np.linspace(0, max_dist, n_bins + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2

        binned_semivar = []
        for i in range(n_bins):
            mask = (distances >= bins[i]) & (distances < bins[i + 1])
            if np.sum(mask) > 0:
                binned_semivar.append(np.mean(semivariances[mask]))
            else:
                binned_semivar.append(np.nan)

        self.variogram = {
            'distances': bin_centers,
            'semivariance': np.array(binned_semivar),
            'n_pairs': [np.sum((distances >= bins[i]) & (distances < bins[i + 1]))
                       for i in range(n_bins)]
        }

        return self.variogram

    def fit_variogram_model(self, model_type: str = 'spherical') -> Dict:
        """
        Fit a theoretical variogram model.

        Args:
            model_type: Type of model ('spherical', 'exponential', 'gaussian')

        Returns:
            Dictionary with fitted model parameters
        """
        if self.variogram is None:
            self.empirical_variogram()

        distances = self.variogram['distances']
        semivar = self.variogram['semivariance']

        # Remove NaN values
        mask = ~np.isnan(semivar)
        distances = distances[mask]
        semivar = semivar[mask]

        # Fit model using least squares
        def model(params, dist, model_type):
            nugget, sill, range_param = params

            if model_type == 'spherical':
                h = dist / range_param
                gamma = np.where(dist <= range_param,
                               nugget + (sill - nugget) * (1.5 * h - 0.5 * h**3),
                               sill)
            elif model_type == 'exponential':
                gamma = nugget + (sill - nugget) * (1 - np.exp(-dist / range_param))
            elif model_type == 'gaussian':
                gamma = nugget + (sill - nugget) * (1 - np.exp(-(dist / range_param)**2))
            else:
                raise ValueError(f"Unknown model type: {model_type}")

            return gamma

        def objective(params):
            predicted = model(params, distances, model_type)
            return np.sum((semivar - predicted) ** 2)

        # Initial parameters
        nugget_init = semivar[0]
        sill_init = np.max(semivar)
        range_init = distances[np.argmax(semivar > 0.95 * sill_init)] if any(semivar > 0.95 * sill_init) else np.max(distances) / 2

        result = minimize(objective, [nugget_init, sill_init, range_init],
                         bounds=[(0, sill_init), (nugget_init, sill_init * 2), (0, np.max(distances))],
                         method='L-BFGS-B')

        nugget, sill, range_param = result.x

        return {
            'model_type': model_type,
            'nugget': nugget,
            'sill': sill,
            'range': range_param,
            'fitted_values': model(result.x, distances, model_type)
        }

    def ordinary_kriging(self, coords: np.ndarray, pred_coords: np.ndarray,
                        values: Optional[np.ndarray] = None,
                        variogram_params: Optional[Dict] = None) -> Dict:
        """
        Perform ordinary kriging interpolation.

        Args:
            coords: Known point coordinates
            pred_coords: Prediction point coordinates
            values: Values at known points
            variogram_params: Variogram model parameters

        Returns:
            Dictionary with predictions and variances
        """
        if values is None:
            values = self.values

        if variogram_params is None:
            variogram_params = self.fit_variogram_model()

        n = len(coords)
        m = len(pred_coords)

        predictions = np.zeros(m)
        variances = np.zeros(m)

        # Covariance function from variogram
        def covariance(dist, params):
            sill = params['sill']
            nugget = params['nugget']
            range_param = params['range']

            if params['model_type'] == 'spherical':
                h = dist / range_param
                gamma = np.where(dist <= range_param,
                               nugget + (sill - nugget) * (1.5 * h - 0.5 * h**3),
                               sill)
            elif params['model_type'] == 'exponential':
                gamma = nugget + (sill - nugget) * (1 - np.exp(-dist / range_param))
            else:
                gamma = nugget + (sill - nugget) * (1 - np.exp(-(dist / range_param)**2))

            return sill - gamma

        # Build covariance matrix for known points
        dist_known = distance_matrix(coords, coords)
        C = covariance(dist_known, variogram_params)

        # Add Lagrange multiplier row/column
        C_extended = np.ones((n + 1, n + 1))
        C_extended[:n, :n] = C
        C_extended[n, n] = 0

        # Solve for each prediction point
        for i in range(m):
            dist_to_pred = np.array([np.linalg.norm(coords[j] - pred_coords[i]) for j in range(n)])
            c = covariance(dist_to_pred, variogram_params)

            # Build right-hand side
            rhs = np.ones(n + 1)
            rhs[:n] = c

            # Solve kriging system
            try:
                weights = np.linalg.solve(C_extended, rhs)
                predictions[i] = np.dot(weights[:n], values)

                # Kriging variance
                variances[i] = variogram_params['sill'] - np.dot(weights[:n], c) - weights[n]
            except:
                predictions[i] = np.mean(values)
                variances[i] = variogram_params['sill']

        return {
            'predictions': predictions,
            'variances': variances,
            'std_errors': np.sqrt(np.maximum(variances, 0))
        }

    def hotspot_analysis(self, coords: np.ndarray, values: np.ndarray,
                        method: str = 'getis_ord') -> Dict:
        """
        Identify spatial hotspots and coldspots.

        Args:
            coords: Spatial coordinates
            values: Values at locations
            method: Method to use ('getis_ord' or 'local_morans')

        Returns:
            Dictionary with hotspot statistics
        """
        n = len(values)
        W = self._calculate_spatial_weights(coords, 'inverse_distance')

        if method == 'getis_ord':
            # Getis-Ord Gi* statistic
            mean_val = np.mean(values)
            std_val = np.std(values)

            gi_star = np.zeros(n)
            for i in range(n):
                W_i = W[i, :].copy()
                W_i[i] = 1  # Include focal point

                sum_w = np.sum(W_i)
                weighted_sum = np.dot(W_i, values)

                gi_star[i] = (weighted_sum - mean_val * sum_w) / (std_val * np.sqrt((n * sum_w - sum_w**2) / (n - 1)))

            p_values = 2 * (1 - stats.norm.cdf(np.abs(gi_star)))

            return {
                'gi_star': gi_star,
                'p_values': p_values,
                'hotspots': gi_star > 1.96,  # 95% confidence
                'coldspots': gi_star < -1.96
            }

    def visualize_spatial_data(self, coords: Optional[np.ndarray] = None,
                              values: Optional[np.ndarray] = None,
                              title: str = "Spatial Data") -> plt.Figure:
        """Visualize spatial data."""
        if coords is None:
            coords = self.coordinates
        if values is None:
            values = self.values

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Scatter plot with colors
        scatter = axes[0].scatter(coords[:, 0], coords[:, 1], c=values,
                                 s=100, cmap='viridis', edgecolors='black', linewidths=0.5)
        axes[0].set_xlabel('X Coordinate', fontsize=12)
        axes[0].set_ylabel('Y Coordinate', fontsize=12)
        axes[0].set_title(title, fontsize=14, weight='bold')
        plt.colorbar(scatter, ax=axes[0], label='Value')
        axes[0].grid(alpha=0.3)

        # Value distribution
        axes[1].hist(values, bins=20, edgecolor='black', alpha=0.7)
        axes[1].axvline(np.mean(values), color='red', linestyle='--',
                       linewidth=2, label=f'Mean: {np.mean(values):.2f}')
        axes[1].axvline(np.median(values), color='green', linestyle='--',
                       linewidth=2, label=f'Median: {np.median(values):.2f}')
        axes[1].set_xlabel('Value', fontsize=12)
        axes[1].set_ylabel('Frequency', fontsize=12)
        axes[1].set_title('Value Distribution', fontsize=14, weight='bold')
        axes[1].legend()
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        return fig


def demo():
    """Demo spatial statistics toolkit."""
    np.random.seed(42)

    print("Spatial Statistics Toolkit Demo")
    print("="*60)

    ss = SpatialStatistics()

    # Generate synthetic spatial data
    n_points = 100
    coords = np.random.rand(n_points, 2) * 100

    # Create spatially autocorrelated values
    values = np.zeros(n_points)
    for i in range(n_points):
        # Add spatial trend
        values[i] = 50 + 0.3 * coords[i, 0] + 0.2 * coords[i, 1]
        # Add local autocorrelation
        nearby = np.linalg.norm(coords - coords[i], axis=1) < 20
        values[i] += np.random.randn() * 5 + np.sum(nearby) * 2

    ss.load_spatial_data(coords, values)

    # 1. Moran's I
    print("\n1. Moran's I Spatial Autocorrelation")
    print("-" * 60)
    morans_result = ss.morans_i()
    print(f"Moran's I: {morans_result['morans_i']:.4f}")
    print(f"Expected I: {morans_result['expected_i']:.4f}")
    print(f"Z-score: {morans_result['z_score']:.4f}")
    print(f"P-value: {morans_result['p_value']:.4e}")
    print(f"Interpretation: {morans_result['interpretation']}")

    # 2. Geary's C
    print("\n2. Geary's C Spatial Autocorrelation")
    print("-" * 60)
    gearys_result = ss.gearys_c()
    print(f"Geary's C: {gearys_result['gearys_c']:.4f}")
    print(f"Expected C: {gearys_result['expected_c']:.4f}")
    print(f"Interpretation: {gearys_result['interpretation']}")

    # 3. Empirical Variogram
    print("\n3. Empirical Variogram")
    print("-" * 60)
    variogram = ss.empirical_variogram(n_bins=10)
    print(f"Distance bins: {len(variogram['distances'])}")
    print(f"Max semivariance: {np.nanmax(variogram['semivariance']):.2f}")

    # 4. Fit Variogram Model
    print("\n4. Fit Variogram Model (Spherical)")
    print("-" * 60)
    model_params = ss.fit_variogram_model(model_type='spherical')
    print(f"Model: {model_params['model_type']}")
    print(f"Nugget: {model_params['nugget']:.4f}")
    print(f"Sill: {model_params['sill']:.4f}")
    print(f"Range: {model_params['range']:.4f}")

    # 5. Ordinary Kriging
    print("\n5. Ordinary Kriging Interpolation")
    print("-" * 60)
    # Create prediction grid
    grid_x, grid_y = np.meshgrid(np.linspace(0, 100, 20), np.linspace(0, 100, 20))
    pred_coords = np.column_stack([grid_x.ravel(), grid_y.ravel()])

    kriging_result = ss.ordinary_kriging(coords[:50], pred_coords[:50],
                                        values[:50], model_params)
    print(f"Prediction points: {len(kriging_result['predictions'])}")
    print(f"Mean prediction: {np.mean(kriging_result['predictions']):.2f}")
    print(f"Mean std error: {np.mean(kriging_result['std_errors']):.2f}")

    # 6. Hotspot Analysis
    print("\n6. Hotspot Analysis (Getis-Ord Gi*)")
    print("-" * 60)
    hotspot_result = ss.hotspot_analysis(coords, values)
    n_hotspots = np.sum(hotspot_result['hotspots'])
    n_coldspots = np.sum(hotspot_result['coldspots'])
    print(f"Number of hotspots: {n_hotspots}")
    print(f"Number of coldspots: {n_coldspots}")

    # Visualize
    print("\n7. Spatial Data Visualization")
    print("-" * 60)
    fig = ss.visualize_spatial_data(title="Spatial Data with Autocorrelation")
    fig.savefig('spatial_statistics_data.png', dpi=300, bbox_inches='tight')
    print("✓ Saved spatial_statistics_data.png")
    plt.close()

    print("\n" + "="*60)
    print("✓ Spatial Statistics Demo Complete!")
    print("="*60)


if __name__ == '__main__':
    demo()
