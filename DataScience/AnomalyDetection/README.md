# Anomaly Detection Toolkit

A comprehensive anomaly detection system implementing multiple state-of-the-art algorithms for identifying outliers and anomalies in data.

## Description

The Anomaly Detection Toolkit provides a unified interface for detecting anomalies using various statistical and machine learning methods. It combines multiple detection algorithms, ensemble methods, and comprehensive visualization tools to identify outliers in univariate and multivariate datasets.

## Key Features

- **Multiple Detection Algorithms**
  - Isolation Forest for efficient anomaly detection
  - Local Outlier Factor (LOF) for density-based detection
  - One-Class SVM for boundary-based detection
  - Statistical methods (Z-score, Modified Z-score, IQR)
  - DBSCAN-based clustering detection

- **Ensemble Methods**
  - Majority voting across multiple detectors
  - Unanimous voting for high-confidence detection
  - Weighted score aggregation

- **Performance Evaluation**
  - Precision, recall, and F1-score metrics
  - Confusion matrix analysis
  - ROC and precision-recall curves

- **Comprehensive Visualizations**
  - 2D scatter plots with anomaly highlighting
  - Anomaly score distributions
  - Method comparison visualizations
  - Interactive dashboards

## Technologies Used

- **Python 3.x**
- **NumPy** - Numerical computing
- **Pandas** - Data manipulation
- **scikit-learn** - Machine learning algorithms
- **Matplotlib/Seaborn** - Visualization
- **SciPy** - Statistical functions

## Installation

```bash
# Clone the repository
cd /home/user/BrillConsulting/DataScience/AnomalyDetection

# Install required packages
pip install numpy pandas scikit-learn matplotlib seaborn scipy
```

## Usage Examples

### Basic Anomaly Detection

```python
from anomaly_detection import AnomalyDetector
import numpy as np

# Generate sample data with anomalies
normal_data = np.random.randn(300, 2) * 0.5
anomaly_data = np.random.uniform(-3, 3, (30, 2))
X = np.vstack([normal_data, anomaly_data])

# Initialize detector
detector = AnomalyDetector(random_state=42)

# Detect anomalies using Isolation Forest
result = detector.isolation_forest(X, contamination=0.1)
print(f"Detected {result['n_anomalies']} anomalies")
print(f"Anomaly indices: {result['anomaly_indices']}")
```

### Multiple Detection Methods

```python
# Local Outlier Factor
lof_result = detector.local_outlier_factor(X, contamination=0.1, n_neighbors=20)
print(f"LOF detected: {lof_result['n_anomalies']} anomalies")

# One-Class SVM
svm_result = detector.one_class_svm(X, nu=0.1)
print(f"SVM detected: {svm_result['n_anomalies']} anomalies")

# Statistical methods
zscore_result = detector.zscore_detection(X, threshold=3.0)
print(f"Z-score detected: {zscore_result['n_anomalies']} anomalies")
```

### Ensemble Detection

```python
# Combine multiple methods with voting
ensemble_result = detector.ensemble_detection(
    X,
    methods=['isolation_forest', 'lof', 'one_class_svm'],
    voting='majority',
    contamination=0.1
)

print(f"Ensemble detected: {ensemble_result['n_anomalies']} anomalies")
print(f"Methods used: {ensemble_result['methods_used']}")
```

### Performance Evaluation

```python
# Evaluate with ground truth labels
y_true = np.hstack([np.zeros(300), np.ones(30)])

metrics = detector.evaluate_performance(y_true, result['anomaly_labels'])
print(f"Precision: {metrics['precision']:.3f}")
print(f"Recall: {metrics['recall']:.3f}")
print(f"F1-Score: {metrics['f1_score']:.3f}")
```

### Visualization

```python
# Visualize anomalies in 2D
fig = detector.visualize_anomalies_2d(
    X,
    result['anomaly_labels'],
    feature_names=['Feature 1', 'Feature 2'],
    title='Anomaly Detection Results'
)
fig.savefig('anomaly_results.png', dpi=300, bbox_inches='tight')

# Visualize anomaly scores
fig = detector.visualize_anomaly_scores(
    result['anomaly_scores'],
    result['anomaly_labels'],
    method_name='Isolation Forest'
)
fig.savefig('anomaly_scores.png', dpi=300, bbox_inches='tight')
```

## Demo Instructions

Run the comprehensive demo to see all features in action:

```bash
python anomaly_detection.py
```

The demo will:
1. Generate synthetic data with known anomalies
2. Apply all detection methods (Isolation Forest, LOF, SVM, statistical tests, DBSCAN)
3. Perform ensemble detection
4. Evaluate performance metrics
5. Generate visualizations (saved as PNG files)
6. Display a comprehensive performance comparison

## Output Examples

**Console Output:**
```
Anomaly Detection Toolkit Demo
======================================================================

Generating synthetic data with anomalies...
Total samples: 330
Normal samples: 300
Anomalies: 30

1. Isolation Forest
----------------------------------------------------------------------
Detected anomalies: 33
Precision: 0.879
Recall: 0.967
F1-Score: 0.921

2. Local Outlier Factor
----------------------------------------------------------------------
Detected anomalies: 33
Precision: 0.848
Recall: 0.933
F1-Score: 0.889

Performance Summary
----------------------------------------------------------------------
Method               Precision    Recall       F1-Score
----------------------------------------------------------------------
Isolation Forest     0.879        0.967        0.921
LOF                  0.848        0.933        0.889
One-Class SVM        0.806        0.833        0.819
Modified Z-Score     0.750        0.900        0.818
Ensemble             0.906        0.967        0.935
```

**Generated Visualizations:**
- `anomaly_detection_2d.png` - 2D scatter plot with anomalies highlighted
- `anomaly_scores_distribution.png` - Distribution of anomaly scores
- `anomaly_methods_comparison.png` - Side-by-side comparison of all methods

## Author

**Brill Consulting**

---

For more information about the algorithms and methodologies, see the inline documentation in `anomaly_detection.py`.
