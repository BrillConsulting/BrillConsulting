# Data Drift Detection Toolkit

A comprehensive toolkit for monitoring and detecting data drift, concept drift, and distribution shifts in production machine learning systems.

## Description

The Data Drift Detection Toolkit provides advanced statistical methods for identifying changes in data distributions over time. It helps maintain model reliability by detecting when input features or target variables shift from their training distributions, enabling proactive model retraining and monitoring.

## Key Features

- **Statistical Drift Tests**
  - Kolmogorov-Smirnov (KS) test for continuous features
  - Chi-square test for categorical features
  - Population Stability Index (PSI)
  - Kullback-Leibler (KL) divergence
  - Jensen-Shannon (JS) divergence
  - Wasserstein distance (Earth Mover's Distance)

- **Feature Drift Monitoring**
  - Per-feature drift detection
  - Multivariate drift detection
  - Feature importance-weighted drift scores
  - Drift severity classification

- **Concept Drift Detection**
  - Target distribution shift detection
  - Prediction drift monitoring
  - Model performance degradation tracking
  - ADWIN (Adaptive Windowing) algorithm

- **Alert System**
  - Configurable drift thresholds
  - Multi-level alert severity (warning, critical)
  - Feature-specific alerts
  - Drift trend analysis

- **Visualization**
  - Distribution comparison plots
  - Drift score heatmaps
  - Time series drift tracking
  - Feature drift dashboards

## Technologies Used

- **Python 3.x**
- **NumPy** - Numerical computing
- **Pandas** - Data manipulation
- **SciPy** - Statistical tests
- **scikit-learn** - Machine learning utilities
- **Matplotlib/Seaborn** - Visualization

## Installation

```bash
# Clone the repository
cd /home/user/BrillConsulting/DataScience/DataDriftDetection

# Install required packages
pip install numpy pandas scipy scikit-learn matplotlib seaborn
```

## Usage Examples

### Kolmogorov-Smirnov Test for Distribution Shift

```python
from data_drift_detection import DataDriftDetector
import numpy as np
import pandas as pd

# Generate reference and current datasets
np.random.seed(42)
reference_data = np.random.randn(1000)
current_data = np.random.randn(800) + 0.5  # Shifted distribution

# Initialize detector
detector = DataDriftDetector(reference_data=reference_data)

# Perform KS test
ks_result = detector.kolmogorov_smirnov_test(
    reference=reference_data,
    current=current_data
)

print(f"KS Test Results:")
print(f"  Statistic: {ks_result['statistic']:.4f}")
print(f"  P-value: {ks_result['p_value']:.4f}")
print(f"  Drift detected: {ks_result['drift_detected']}")
print(f"  Drift score: {ks_result['drift_score']:.4f}")
```

### Population Stability Index (PSI)

```python
# Calculate PSI for monitoring feature stability
psi_result = detector.population_stability_index(
    reference=reference_data,
    current=current_data,
    n_bins=10
)

print(f"PSI Results:")
print(f"  PSI value: {psi_result['psi']:.4f}")
print(f"  Interpretation: {psi_result['interpretation']}")
print(f"  Drift detected: {psi_result['drift_detected']}")
print(f"  Bin contributions: {psi_result['bin_contributions']}")

# Visualize PSI
fig = psi_result['plot']
fig.savefig('psi_analysis.png', dpi=300, bbox_inches='tight')
```

### Chi-Square Test for Categorical Features

```python
# Test categorical feature drift
reference_categorical = np.random.choice(['A', 'B', 'C'], size=1000, p=[0.5, 0.3, 0.2])
current_categorical = np.random.choice(['A', 'B', 'C'], size=800, p=[0.3, 0.4, 0.3])

chi_square_result = detector.chi_square_test(
    reference=reference_categorical,
    current=current_categorical
)

print(f"Chi-Square Test Results:")
print(f"  Statistic: {chi_square_result['statistic']:.4f}")
print(f"  P-value: {chi_square_result['p_value']:.4f}")
print(f"  Drift detected: {chi_square_result['drift_detected']}")
print(f"  Expected vs. Observed:")
for category in chi_square_result['expected_freq'].keys():
    print(f"    {category}: expected={chi_square_result['expected_freq'][category]:.1f}, "
          f"observed={chi_square_result['observed_freq'][category]:.1f}")
```

### Kullback-Leibler Divergence

```python
# Calculate KL divergence
kl_result = detector.kullback_leibler_divergence(
    reference=reference_data,
    current=current_data,
    n_bins=20
)

print(f"KL Divergence Results:")
print(f"  KL divergence: {kl_result['kl_divergence']:.4f}")
print(f"  JS divergence: {kl_result['js_divergence']:.4f}")
print(f"  Drift detected: {kl_result['drift_detected']}")
print(f"  Severity: {kl_result['severity']}")
```

### Wasserstein Distance

```python
# Calculate Wasserstein distance (Earth Mover's Distance)
wasserstein_result = detector.wasserstein_distance(
    reference=reference_data,
    current=current_data
)

print(f"Wasserstein Distance Results:")
print(f"  Distance: {wasserstein_result['distance']:.4f}")
print(f"  Normalized distance: {wasserstein_result['normalized_distance']:.4f}")
print(f"  Drift detected: {wasserstein_result['drift_detected']}")
```

### Multi-Feature Drift Detection

```python
# Create multi-feature datasets
reference_df = pd.DataFrame({
    'feature1': np.random.randn(1000),
    'feature2': np.random.randn(1000) * 2,
    'feature3': np.random.choice(['A', 'B', 'C'], 1000),
    'feature4': np.random.exponential(2, 1000)
})

current_df = pd.DataFrame({
    'feature1': np.random.randn(800) + 0.3,  # Slight shift
    'feature2': np.random.randn(800) * 2.5,  # Scale change
    'feature3': np.random.choice(['A', 'B', 'C'], 800, p=[0.2, 0.5, 0.3]),  # Distribution change
    'feature4': np.random.exponential(2.2, 800)  # Parameter change
})

# Detect drift across all features
multi_drift = detector.detect_dataset_drift(
    reference=reference_df,
    current=current_df,
    feature_types={
        'feature1': 'continuous',
        'feature2': 'continuous',
        'feature3': 'categorical',
        'feature4': 'continuous'
    }
)

print(f"Multi-Feature Drift Detection:")
print(f"  Overall drift detected: {multi_drift['overall_drift_detected']}")
print(f"  Number of features with drift: {multi_drift['n_features_with_drift']}")
print(f"\nPer-feature results:")
for feature, result in multi_drift['feature_results'].items():
    print(f"  {feature}:")
    print(f"    Drift detected: {result['drift_detected']}")
    print(f"    Test statistic: {result['test_statistic']:.4f}")
    print(f"    P-value: {result['p_value']:.4f}")
```

### Concept Drift Detection

```python
# Detect concept drift (target distribution shift)
reference_target = np.random.binomial(1, 0.3, 1000)
current_target = np.random.binomial(1, 0.5, 800)  # Target proportion changed

concept_drift = detector.detect_concept_drift(
    reference_target=reference_target,
    current_target=current_target
)

print(f"Concept Drift Detection:")
print(f"  Drift detected: {concept_drift['drift_detected']}")
print(f"  Reference positive rate: {concept_drift['reference_positive_rate']:.3f}")
print(f"  Current positive rate: {concept_drift['current_positive_rate']:.3f}")
print(f"  Relative change: {concept_drift['relative_change']:.1f}%")
```

### Drift Monitoring Dashboard

```python
# Generate comprehensive drift monitoring report
monitoring_result = detector.monitor_drift_over_time(
    reference=reference_df,
    current_batches=[current_df, current_df, current_df],
    timestamps=['2024-01-01', '2024-01-02', '2024-01-03']
)

print(f"Drift Monitoring Over Time:")
print(f"  Total monitoring periods: {len(monitoring_result['timestamps'])}")
print(f"  Periods with drift: {monitoring_result['n_periods_with_drift']}")
print(f"  Most affected feature: {monitoring_result['most_affected_feature']}")

# Visualize drift trends
fig = monitoring_result['trend_plot']
fig.savefig('drift_trends.png', dpi=300, bbox_inches='tight')

# Generate drift heatmap
fig = monitoring_result['heatmap']
fig.savefig('drift_heatmap.png', dpi=300, bbox_inches='tight')
```

## Demo Instructions

Run the comprehensive demo to see all features in action:

```bash
python data_drift_detection.py
```

The demo will:
1. Generate reference and current datasets with known distribution shifts
2. Apply all statistical drift tests (KS, Chi-square, PSI, KL, Wasserstein)
3. Detect drift in multiple features simultaneously
4. Monitor concept drift in target variables
5. Generate drift monitoring reports
6. Create visualizations (distribution comparisons, drift scores, trends)
7. Display comprehensive drift analysis results

## Output Examples

**Console Output:**
```
Data Drift Detection Toolkit Demo
======================================================================

Generating reference and current datasets...
Reference dataset: 1000 samples, 5 features
Current dataset: 800 samples, 5 features

Feature Distribution Changes:
  feature1: Mean shift from 0.02 to 0.52 (mild drift)
  feature2: Scale change from 2.0 to 2.5 (moderate drift)
  feature3: Category proportions changed (significant drift)
  feature4: No significant change
  feature5: Extreme shift (severe drift)

1. Kolmogorov-Smirnov Test (feature1)
----------------------------------------------------------------------
KS Statistic: 0.2345
P-value: < 0.001
Drift detected: Yes
Drift score: 0.2345
Severity: Moderate

2. Population Stability Index (feature1)
----------------------------------------------------------------------
PSI value: 0.1234
Interpretation: Moderate shift (0.1 < PSI < 0.25)
Drift detected: Yes
Recommendation: Monitor closely, consider retraining

3. Chi-Square Test (feature3 - categorical)
----------------------------------------------------------------------
Chi-square statistic: 45.67
P-value: < 0.001
Drift detected: Yes
Category changes:
  A: 50% → 30% (decrease)
  B: 30% → 50% (increase)
  C: 20% → 20% (stable)

4. Kullback-Leibler Divergence (feature1)
----------------------------------------------------------------------
KL divergence: 0.0876
JS divergence: 0.0438
Drift detected: Yes
Severity: Moderate
Distribution overlap: 78.5%

5. Wasserstein Distance (feature1)
----------------------------------------------------------------------
Wasserstein distance: 0.4982
Normalized distance: 0.3456
Drift detected: Yes
Interpretation: Distributions have shifted moderately

Multi-Feature Analysis
----------------------------------------------------------------------
Features analyzed: 5
Features with drift: 3
Overall drift detected: Yes

Feature drift scores:
  feature5: 0.892 (CRITICAL)
  feature3: 0.567 (WARNING)
  feature1: 0.234 (WARNING)
  feature2: 0.123 (STABLE)
  feature4: 0.045 (STABLE)

Concept Drift Detection
----------------------------------------------------------------------
Target distribution shift detected: Yes
Reference positive rate: 30.2%
Current positive rate: 48.7%
Relative change: +61.3%
Recommendation: Model retraining required

Drift Monitoring (3-day window)
----------------------------------------------------------------------
Day 1: 2 features drifted
Day 2: 3 features drifted
Day 3: 3 features drifted

Most affected feature: feature5 (consistent drift)
Drift trend: Increasing

Alerts Generated:
  [CRITICAL] feature5 - Severe drift detected (score: 0.892)
  [WARNING] feature3 - Moderate drift detected (score: 0.567)
  [WARNING] feature1 - Moderate drift detected (score: 0.234)
  [INFO] Concept drift detected in target variable
```

**Generated Visualizations:**
- `distribution_comparison.png` - Side-by-side reference vs. current distributions
- `psi_analysis.png` - PSI breakdown by bins
- `drift_scores.png` - Bar chart of drift scores per feature
- `drift_heatmap.png` - Heatmap showing drift over time
- `drift_trends.png` - Time series of drift metrics
- `feature_importance_drift.png` - Drift weighted by feature importance

## Author

**Brill Consulting**

---

For more information about the algorithms and methodologies, see the inline documentation in `data_drift_detection.py`.
