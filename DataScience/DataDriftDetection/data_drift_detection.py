"""
Data Drift Detection Toolkit
=============================

Comprehensive data drift detection with multiple statistical tests:
- Kolmogorov-Smirnov (KS) test
- Population Stability Index (PSI)
- Chi-square test for categorical variables
- Kullback-Leibler divergence
- Wasserstein distance
- Jensen-Shannon divergence
- Concept drift detection
- Feature drift monitoring with alerts
- Comprehensive drift reports and visualizations

Author: Brill Consulting
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance, chi2_contingency, ks_2samp
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')


class DataDriftDetector:
    """Comprehensive data drift detection toolkit with multiple statistical methods."""

    def __init__(self, significance_level: float = 0.05):
        """
        Initialize data drift detector.

        Args:
            significance_level: Significance level for statistical tests (default: 0.05)
        """
        self.significance_level = significance_level
        self.drift_history = []
        self.reference_distributions = {}
        self.alert_thresholds = {
            'psi': 0.1,      # PSI > 0.1 indicates drift
            'ks': 0.05,      # KS p-value < 0.05 indicates drift
            'kl': 0.1,       # KL divergence > 0.1 indicates drift
            'js': 0.1,       # JS divergence > 0.1 indicates drift
            'wasserstein': 0.5  # Wasserstein distance > 0.5 indicates drift
        }

    def kolmogorov_smirnov_test(self, reference: np.ndarray,
                                current: np.ndarray) -> Dict:
        """
        Perform Kolmogorov-Smirnov test for distribution drift.

        Args:
            reference: Reference distribution data
            current: Current distribution data

        Returns:
            Dictionary with KS test results
        """
        statistic, p_value = ks_2samp(reference, current)

        drift_detected = p_value < self.significance_level

        return {
            'test': 'Kolmogorov-Smirnov',
            'statistic': float(statistic),
            'p_value': float(p_value),
            'drift_detected': drift_detected,
            'drift_score': float(statistic),  # Higher = more drift
            'threshold': self.significance_level
        }

    def population_stability_index(self, reference: np.ndarray,
                                  current: np.ndarray,
                                  n_bins: int = 10) -> Dict:
        """
        Calculate Population Stability Index (PSI).

        PSI measures the shift in population distribution:
        - PSI < 0.1: No significant change
        - PSI 0.1-0.2: Moderate change
        - PSI > 0.2: Significant change

        Args:
            reference: Reference distribution data
            current: Current distribution data
            n_bins: Number of bins for discretization

        Returns:
            Dictionary with PSI results
        """
        # Create bins based on reference distribution
        _, bin_edges = np.histogram(reference, bins=n_bins)

        # Ensure bins cover full range of both distributions
        min_val = min(reference.min(), current.min())
        max_val = max(reference.max(), current.max())
        bin_edges[0] = min_val - 1e-10
        bin_edges[-1] = max_val + 1e-10

        # Calculate distributions
        ref_counts, _ = np.histogram(reference, bins=bin_edges)
        curr_counts, _ = np.histogram(current, bins=bin_edges)

        # Convert to percentages
        ref_percents = ref_counts / len(reference)
        curr_percents = curr_counts / len(current)

        # Avoid division by zero
        ref_percents = np.where(ref_percents == 0, 0.0001, ref_percents)
        curr_percents = np.where(curr_percents == 0, 0.0001, curr_percents)

        # Calculate PSI
        psi_values = (curr_percents - ref_percents) * np.log(curr_percents / ref_percents)
        psi = np.sum(psi_values)

        # Determine drift level
        if psi < 0.1:
            drift_level = 'No significant drift'
        elif psi < 0.2:
            drift_level = 'Moderate drift'
        else:
            drift_level = 'Significant drift'

        drift_detected = psi > self.alert_thresholds['psi']

        return {
            'test': 'Population Stability Index',
            'psi': float(psi),
            'drift_level': drift_level,
            'drift_detected': drift_detected,
            'drift_score': float(psi),
            'threshold': self.alert_thresholds['psi'],
            'bin_contributions': psi_values.tolist()
        }

    def chi_square_test(self, reference: np.ndarray, current: np.ndarray) -> Dict:
        """
        Perform Chi-square test for categorical drift.

        Args:
            reference: Reference categorical data
            current: Current categorical data

        Returns:
            Dictionary with chi-square test results
        """
        # Get unique categories
        all_categories = np.unique(np.concatenate([reference, current]))

        # Create contingency table
        ref_counts = pd.Series(reference).value_counts()
        curr_counts = pd.Series(current).value_counts()

        # Ensure all categories are present
        contingency_table = []
        for cat in all_categories:
            contingency_table.append([
                ref_counts.get(cat, 0),
                curr_counts.get(cat, 0)
            ])

        contingency_table = np.array(contingency_table).T

        # Perform chi-square test
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)

        drift_detected = p_value < self.significance_level

        return {
            'test': 'Chi-Square',
            'statistic': float(chi2),
            'p_value': float(p_value),
            'degrees_of_freedom': int(dof),
            'drift_detected': drift_detected,
            'drift_score': float(chi2),
            'threshold': self.significance_level,
            'n_categories': len(all_categories)
        }

    def kullback_leibler_divergence(self, reference: np.ndarray,
                                   current: np.ndarray,
                                   n_bins: int = 50) -> Dict:
        """
        Calculate Kullback-Leibler divergence.

        KL divergence measures how one distribution differs from another.

        Args:
            reference: Reference distribution data
            current: Current distribution data
            n_bins: Number of bins for discretization

        Returns:
            Dictionary with KL divergence results
        """
        # Create common bins
        min_val = min(reference.min(), current.min())
        max_val = max(reference.max(), current.max())
        bins = np.linspace(min_val, max_val, n_bins + 1)

        # Calculate distributions
        ref_hist, _ = np.histogram(reference, bins=bins)
        curr_hist, _ = np.histogram(current, bins=bins)

        # Convert to probabilities
        ref_prob = (ref_hist + 1e-10) / (ref_hist.sum() + n_bins * 1e-10)
        curr_prob = (curr_hist + 1e-10) / (curr_hist.sum() + n_bins * 1e-10)

        # Calculate KL divergence
        kl_div = np.sum(curr_prob * np.log(curr_prob / ref_prob))

        drift_detected = kl_div > self.alert_thresholds['kl']

        return {
            'test': 'Kullback-Leibler Divergence',
            'kl_divergence': float(kl_div),
            'drift_detected': drift_detected,
            'drift_score': float(kl_div),
            'threshold': self.alert_thresholds['kl']
        }

    def jensen_shannon_divergence(self, reference: np.ndarray,
                                  current: np.ndarray,
                                  n_bins: int = 50) -> Dict:
        """
        Calculate Jensen-Shannon divergence.

        JS divergence is a symmetric version of KL divergence.

        Args:
            reference: Reference distribution data
            current: Current distribution data
            n_bins: Number of bins for discretization

        Returns:
            Dictionary with JS divergence results
        """
        # Create common bins
        min_val = min(reference.min(), current.min())
        max_val = max(reference.max(), current.max())
        bins = np.linspace(min_val, max_val, n_bins + 1)

        # Calculate distributions
        ref_hist, _ = np.histogram(reference, bins=bins)
        curr_hist, _ = np.histogram(current, bins=bins)

        # Convert to probabilities
        ref_prob = (ref_hist + 1e-10) / (ref_hist.sum() + n_bins * 1e-10)
        curr_prob = (curr_hist + 1e-10) / (curr_hist.sum() + n_bins * 1e-10)

        # Calculate JS divergence
        js_div = jensenshannon(ref_prob, curr_prob)

        drift_detected = js_div > self.alert_thresholds['js']

        return {
            'test': 'Jensen-Shannon Divergence',
            'js_divergence': float(js_div),
            'drift_detected': drift_detected,
            'drift_score': float(js_div),
            'threshold': self.alert_thresholds['js']
        }

    def wasserstein_distance_test(self, reference: np.ndarray,
                                  current: np.ndarray) -> Dict:
        """
        Calculate Wasserstein distance (Earth Mover's Distance).

        Measures the minimum cost to transform one distribution into another.

        Args:
            reference: Reference distribution data
            current: Current distribution data

        Returns:
            Dictionary with Wasserstein distance results
        """
        w_distance = wasserstein_distance(reference, current)

        drift_detected = w_distance > self.alert_thresholds['wasserstein']

        return {
            'test': 'Wasserstein Distance',
            'distance': float(w_distance),
            'drift_detected': drift_detected,
            'drift_score': float(w_distance),
            'threshold': self.alert_thresholds['wasserstein']
        }

    def detect_concept_drift(self, reference_X: np.ndarray, reference_y: np.ndarray,
                           current_X: np.ndarray, current_y: np.ndarray,
                           model=None) -> Dict:
        """
        Detect concept drift by comparing model performance.

        Args:
            reference_X: Reference features
            reference_y: Reference labels
            current_X: Current features
            current_y: Current labels
            model: Trained model (if None, trains a simple logistic regression)

        Returns:
            Dictionary with concept drift results
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, f1_score

        if model is None:
            model = LogisticRegression(random_state=42, max_iter=1000)
            model.fit(reference_X, reference_y)

        # Performance on reference data
        ref_pred = model.predict(reference_X)
        ref_accuracy = accuracy_score(reference_y, ref_pred)
        ref_f1 = f1_score(reference_y, ref_pred, average='weighted')

        # Performance on current data
        curr_pred = model.predict(current_X)
        curr_accuracy = accuracy_score(current_y, curr_pred)
        curr_f1 = f1_score(current_y, curr_pred, average='weighted')

        # Calculate performance degradation
        accuracy_drop = ref_accuracy - curr_accuracy
        f1_drop = ref_f1 - curr_f1

        # Detect concept drift if performance drops significantly
        drift_detected = accuracy_drop > 0.1 or f1_drop > 0.1

        return {
            'test': 'Concept Drift',
            'reference_accuracy': float(ref_accuracy),
            'current_accuracy': float(curr_accuracy),
            'accuracy_drop': float(accuracy_drop),
            'reference_f1': float(ref_f1),
            'current_f1': float(curr_f1),
            'f1_drop': float(f1_drop),
            'drift_detected': drift_detected,
            'drift_score': float(max(accuracy_drop, f1_drop))
        }

    def monitor_feature_drift(self, reference_df: pd.DataFrame,
                            current_df: pd.DataFrame,
                            categorical_features: Optional[List[str]] = None) -> Dict:
        """
        Monitor drift across all features in a dataset.

        Args:
            reference_df: Reference dataset
            current_df: Current dataset
            categorical_features: List of categorical feature names

        Returns:
            Dictionary with drift results for all features
        """
        if categorical_features is None:
            categorical_features = []

        drift_results = {}

        for col in reference_df.columns:
            if col in categorical_features:
                # Use chi-square test for categorical features
                try:
                    result = self.chi_square_test(
                        reference_df[col].values,
                        current_df[col].values
                    )
                    drift_results[col] = result
                except Exception as e:
                    drift_results[col] = {'error': str(e)}
            else:
                # Use multiple tests for numerical features
                try:
                    ks_result = self.kolmogorov_smirnov_test(
                        reference_df[col].values,
                        current_df[col].values
                    )
                    psi_result = self.population_stability_index(
                        reference_df[col].values,
                        current_df[col].values
                    )

                    drift_results[col] = {
                        'ks_test': ks_result,
                        'psi': psi_result,
                        'drift_detected': ks_result['drift_detected'] or psi_result['drift_detected']
                    }
                except Exception as e:
                    drift_results[col] = {'error': str(e)}

        return drift_results

    def generate_drift_report(self, drift_results: Dict) -> str:
        """
        Generate a comprehensive drift report.

        Args:
            drift_results: Dictionary of drift detection results

        Returns:
            Formatted drift report string
        """
        report = []
        report.append("=" * 80)
        report.append("DATA DRIFT DETECTION REPORT")
        report.append("=" * 80)
        report.append("")

        # Count features with drift
        features_with_drift = []
        total_features = len(drift_results)

        for feature, result in drift_results.items():
            if 'error' in result:
                continue

            if 'drift_detected' in result:
                if result['drift_detected']:
                    features_with_drift.append(feature)
            elif 'ks_test' in result:
                if result['ks_test']['drift_detected'] or result['psi']['drift_detected']:
                    features_with_drift.append(feature)

        report.append(f"Total features analyzed: {total_features}")
        report.append(f"Features with detected drift: {len(features_with_drift)}")
        report.append(f"Drift percentage: {100 * len(features_with_drift) / total_features:.1f}%")
        report.append("")

        if features_with_drift:
            report.append("FEATURES WITH DETECTED DRIFT:")
            report.append("-" * 80)
            for feature in features_with_drift:
                result = drift_results[feature]
                report.append(f"\n{feature}:")

                if 'ks_test' in result:
                    ks = result['ks_test']
                    psi = result['psi']
                    report.append(f"  KS Test: statistic={ks['statistic']:.4f}, p-value={ks['p_value']:.4f}")
                    report.append(f"  PSI: {psi['psi']:.4f} ({psi['drift_level']})")
                elif 'test' in result:
                    report.append(f"  Test: {result['test']}")
                    report.append(f"  Drift Score: {result.get('drift_score', 'N/A'):.4f}")

        report.append("")
        report.append("=" * 80)

        return "\n".join(report)

    def create_drift_alert(self, drift_results: Dict,
                          alert_threshold: float = 0.2) -> Dict:
        """
        Create drift alerts based on detection results.

        Args:
            drift_results: Dictionary of drift detection results
            alert_threshold: Threshold for triggering alerts (fraction of features)

        Returns:
            Dictionary with alert information
        """
        features_with_drift = []
        critical_features = []

        for feature, result in drift_results.items():
            if 'error' in result:
                continue

            drift_detected = False
            drift_score = 0.0

            if 'ks_test' in result:
                drift_detected = result['ks_test']['drift_detected'] or result['psi']['drift_detected']
                drift_score = max(result['ks_test']['drift_score'], result['psi']['drift_score'])
            elif 'drift_detected' in result:
                drift_detected = result['drift_detected']
                drift_score = result.get('drift_score', 0.0)

            if drift_detected:
                features_with_drift.append(feature)
                if drift_score > 0.5:  # Critical drift
                    critical_features.append((feature, drift_score))

        total_features = len([k for k in drift_results.keys() if 'error' not in drift_results[k]])
        drift_ratio = len(features_with_drift) / total_features if total_features > 0 else 0

        alert_level = 'NONE'
        if drift_ratio > alert_threshold * 2:
            alert_level = 'CRITICAL'
        elif drift_ratio > alert_threshold:
            alert_level = 'WARNING'
        elif len(features_with_drift) > 0:
            alert_level = 'INFO'

        return {
            'alert_level': alert_level,
            'drift_ratio': float(drift_ratio),
            'features_with_drift': features_with_drift,
            'critical_features': critical_features,
            'total_features': total_features,
            'alert_threshold': alert_threshold,
            'message': self._generate_alert_message(alert_level, drift_ratio, features_with_drift)
        }

    def _generate_alert_message(self, alert_level: str, drift_ratio: float,
                               features: List[str]) -> str:
        """Generate alert message based on drift detection."""
        if alert_level == 'CRITICAL':
            return f"CRITICAL: Significant drift detected in {len(features)} features ({drift_ratio:.1%} of total)"
        elif alert_level == 'WARNING':
            return f"WARNING: Moderate drift detected in {len(features)} features ({drift_ratio:.1%} of total)"
        elif alert_level == 'INFO':
            return f"INFO: Minor drift detected in {len(features)} features"
        else:
            return "No significant drift detected"

    def visualize_feature_drift(self, reference: np.ndarray, current: np.ndarray,
                               feature_name: str = "Feature") -> plt.Figure:
        """
        Visualize drift for a single feature.

        Args:
            reference: Reference distribution
            current: Current distribution
            feature_name: Name of the feature

        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Histogram comparison
        axes[0, 0].hist(reference, bins=50, alpha=0.6, label='Reference',
                       color='blue', edgecolor='black', density=True)
        axes[0, 0].hist(current, bins=50, alpha=0.6, label='Current',
                       color='red', edgecolor='black', density=True)
        axes[0, 0].set_xlabel(feature_name, fontsize=11)
        axes[0, 0].set_ylabel('Density', fontsize=11)
        axes[0, 0].set_title('Distribution Comparison', fontsize=12, fontweight='bold')
        axes[0, 0].legend(fontsize=10)
        axes[0, 0].grid(alpha=0.3)

        # Box plot comparison
        axes[0, 1].boxplot([reference, current], labels=['Reference', 'Current'])
        axes[0, 1].set_ylabel(feature_name, fontsize=11)
        axes[0, 1].set_title('Box Plot Comparison', fontsize=12, fontweight='bold')
        axes[0, 1].grid(alpha=0.3)

        # Q-Q plot
        from scipy.stats import probplot
        probplot(current, dist=stats.norm, plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot (Current vs Normal)', fontsize=12, fontweight='bold')
        axes[1, 0].grid(alpha=0.3)

        # Cumulative distribution
        ref_sorted = np.sort(reference)
        curr_sorted = np.sort(current)
        ref_cdf = np.arange(1, len(ref_sorted) + 1) / len(ref_sorted)
        curr_cdf = np.arange(1, len(curr_sorted) + 1) / len(curr_sorted)

        axes[1, 1].plot(ref_sorted, ref_cdf, label='Reference', linewidth=2, color='blue')
        axes[1, 1].plot(curr_sorted, curr_cdf, label='Current', linewidth=2, color='red')
        axes[1, 1].set_xlabel(feature_name, fontsize=11)
        axes[1, 1].set_ylabel('Cumulative Probability', fontsize=11)
        axes[1, 1].set_title('Cumulative Distribution', fontsize=12, fontweight='bold')
        axes[1, 1].legend(fontsize=10)
        axes[1, 1].grid(alpha=0.3)

        plt.suptitle(f'Drift Analysis: {feature_name}', fontsize=14, fontweight='bold', y=1.00)
        plt.tight_layout()

        return fig

    def visualize_drift_summary(self, drift_results: Dict) -> plt.Figure:
        """
        Create a summary visualization of drift across all features.

        Args:
            drift_results: Dictionary of drift detection results

        Returns:
            Matplotlib figure
        """
        # Extract drift scores
        feature_names = []
        drift_scores = []
        drift_detected = []

        for feature, result in drift_results.items():
            if 'error' in result:
                continue

            feature_names.append(feature)

            if 'ks_test' in result:
                # Take maximum of KS and PSI scores
                score = max(result['ks_test']['drift_score'],
                           result['psi']['drift_score'])
                detected = result['ks_test']['drift_detected'] or result['psi']['drift_detected']
            else:
                score = result.get('drift_score', 0.0)
                detected = result.get('drift_detected', False)

            drift_scores.append(score)
            drift_detected.append(detected)

        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Bar chart of drift scores
        colors = ['red' if d else 'blue' for d in drift_detected]
        axes[0].barh(feature_names, drift_scores, color=colors, alpha=0.7, edgecolor='black')
        axes[0].set_xlabel('Drift Score', fontsize=12)
        axes[0].set_title('Feature Drift Scores', fontsize=13, fontweight='bold')
        axes[0].grid(alpha=0.3, axis='x')

        # Add drift threshold line
        if drift_scores:
            axes[0].axvline(x=self.alert_thresholds['psi'], color='orange',
                          linestyle='--', linewidth=2, label='Threshold')
            axes[0].legend(fontsize=10)

        # Pie chart of drift detection
        drift_counts = [sum(drift_detected), len(drift_detected) - sum(drift_detected)]
        labels = ['Drift Detected', 'No Drift']
        colors_pie = ['red', 'blue']

        axes[1].pie(drift_counts, labels=labels, colors=colors_pie, autopct='%1.1f%%',
                   startangle=90, textprops={'fontsize': 11})
        axes[1].set_title('Drift Detection Summary', fontsize=13, fontweight='bold')

        plt.tight_layout()
        return fig


def demo():
    """Demonstrate data drift detection toolkit."""
    np.random.seed(42)

    print("Data Drift Detection Toolkit Demo")
    print("=" * 80)

    # 1. Generate reference and drifted data
    print("\n1. Generating Reference and Current Data")
    print("-" * 80)

    n_samples = 1000
    n_features = 5

    # Reference data (normal distribution)
    reference_data = np.random.randn(n_samples, n_features)
    reference_df = pd.DataFrame(
        reference_data,
        columns=[f'feature_{i}' for i in range(n_features)]
    )

    # Current data with drift
    # Feature 0: shift in mean
    # Feature 1: change in variance
    # Feature 2: no drift
    # Feature 3: significant shift
    # Feature 4: change in distribution shape
    current_data = np.random.randn(n_samples, n_features)
    current_data[:, 0] += 0.5  # Mean shift
    current_data[:, 1] *= 2.0  # Variance change
    current_data[:, 3] += 1.5  # Significant shift
    current_data[:, 4] = np.abs(current_data[:, 4])  # Distribution shape change

    current_df = pd.DataFrame(
        current_data,
        columns=[f'feature_{i}' for i in range(n_features)]
    )

    print(f"Reference data shape: {reference_df.shape}")
    print(f"Current data shape: {current_df.shape}")

    detector = DataDriftDetector(significance_level=0.05)

    # 2. Kolmogorov-Smirnov Test
    print("\n2. Kolmogorov-Smirnov Test (Feature 0)")
    print("-" * 80)
    ks_result = detector.kolmogorov_smirnov_test(
        reference_df['feature_0'].values,
        current_df['feature_0'].values
    )
    print(f"KS Statistic: {ks_result['statistic']:.4f}")
    print(f"P-value: {ks_result['p_value']:.4f}")
    print(f"Drift Detected: {ks_result['drift_detected']}")

    # 3. Population Stability Index
    print("\n3. Population Stability Index (Feature 0)")
    print("-" * 80)
    psi_result = detector.population_stability_index(
        reference_df['feature_0'].values,
        current_df['feature_0'].values
    )
    print(f"PSI: {psi_result['psi']:.4f}")
    print(f"Drift Level: {psi_result['drift_level']}")
    print(f"Drift Detected: {psi_result['drift_detected']}")

    # 4. Kullback-Leibler Divergence
    print("\n4. Kullback-Leibler Divergence (Feature 3)")
    print("-" * 80)
    kl_result = detector.kullback_leibler_divergence(
        reference_df['feature_3'].values,
        current_df['feature_3'].values
    )
    print(f"KL Divergence: {kl_result['kl_divergence']:.4f}")
    print(f"Drift Detected: {kl_result['drift_detected']}")

    # 5. Jensen-Shannon Divergence
    print("\n5. Jensen-Shannon Divergence (Feature 3)")
    print("-" * 80)
    js_result = detector.jensen_shannon_divergence(
        reference_df['feature_3'].values,
        current_df['feature_3'].values
    )
    print(f"JS Divergence: {js_result['js_divergence']:.4f}")
    print(f"Drift Detected: {js_result['drift_detected']}")

    # 6. Wasserstein Distance
    print("\n6. Wasserstein Distance (Feature 1)")
    print("-" * 80)
    w_result = detector.wasserstein_distance_test(
        reference_df['feature_1'].values,
        current_df['feature_1'].values
    )
    print(f"Wasserstein Distance: {w_result['distance']:.4f}")
    print(f"Drift Detected: {w_result['drift_detected']}")

    # 7. Chi-Square Test (categorical)
    print("\n7. Chi-Square Test for Categorical Data")
    print("-" * 80)
    # Create categorical data
    ref_categorical = np.random.choice(['A', 'B', 'C'], size=1000, p=[0.5, 0.3, 0.2])
    curr_categorical = np.random.choice(['A', 'B', 'C'], size=1000, p=[0.3, 0.4, 0.3])

    chi2_result = detector.chi_square_test(ref_categorical, curr_categorical)
    print(f"Chi-Square Statistic: {chi2_result['statistic']:.4f}")
    print(f"P-value: {chi2_result['p_value']:.4f}")
    print(f"Drift Detected: {chi2_result['drift_detected']}")

    # 8. Concept Drift Detection
    print("\n8. Concept Drift Detection")
    print("-" * 80)
    # Generate labeled data for concept drift
    ref_X = np.random.randn(500, 5)
    ref_y = (ref_X[:, 0] + ref_X[:, 1] > 0).astype(int)

    curr_X = np.random.randn(500, 5)
    # Concept has changed: different decision boundary
    curr_y = (curr_X[:, 0] - curr_X[:, 1] > 0).astype(int)

    concept_result = detector.detect_concept_drift(ref_X, ref_y, curr_X, curr_y)
    print(f"Reference Accuracy: {concept_result['reference_accuracy']:.4f}")
    print(f"Current Accuracy: {concept_result['current_accuracy']:.4f}")
    print(f"Accuracy Drop: {concept_result['accuracy_drop']:.4f}")
    print(f"Concept Drift Detected: {concept_result['drift_detected']}")

    # 9. Feature Drift Monitoring
    print("\n9. Feature Drift Monitoring (All Features)")
    print("-" * 80)
    drift_results = detector.monitor_feature_drift(reference_df, current_df)

    for feature, result in drift_results.items():
        if 'error' in result:
            continue
        if 'ks_test' in result:
            ks_drift = result['ks_test']['drift_detected']
            psi_drift = result['psi']['drift_detected']
            print(f"{feature}: KS Drift={ks_drift}, PSI Drift={psi_drift}")

    # 10. Generate Drift Report
    print("\n10. Comprehensive Drift Report")
    print("-" * 80)
    report = detector.generate_drift_report(drift_results)
    print(report)

    # 11. Create Drift Alert
    print("\n11. Drift Alert System")
    print("-" * 80)
    alert = detector.create_drift_alert(drift_results, alert_threshold=0.2)
    print(f"Alert Level: {alert['alert_level']}")
    print(f"Drift Ratio: {alert['drift_ratio']:.2%}")
    print(f"Message: {alert['message']}")
    print(f"Features with Drift: {', '.join(alert['features_with_drift'])}")

    # 12. Visualizations
    print("\n12. Generating Visualizations")
    print("-" * 80)

    # Feature drift visualization
    fig1 = detector.visualize_feature_drift(
        reference_df['feature_3'].values,
        current_df['feature_3'].values,
        feature_name='Feature 3 (Significant Drift)'
    )
    fig1.savefig('drift_feature_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Saved drift_feature_analysis.png")
    plt.close()

    # Drift summary visualization
    fig2 = detector.visualize_drift_summary(drift_results)
    fig2.savefig('drift_summary.png', dpi=300, bbox_inches='tight')
    print("✓ Saved drift_summary.png")
    plt.close()

    # No-drift feature for comparison
    fig3 = detector.visualize_feature_drift(
        reference_df['feature_2'].values,
        current_df['feature_2'].values,
        feature_name='Feature 2 (No Drift)'
    )
    fig3.savefig('drift_no_drift_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved drift_no_drift_comparison.png")
    plt.close()

    print("\n" + "=" * 80)
    print("✓ Data Drift Detection Demo Complete!")
    print("=" * 80)


if __name__ == '__main__':
    demo()
