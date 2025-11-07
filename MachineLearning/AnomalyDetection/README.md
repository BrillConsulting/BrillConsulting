# 🚨 Anomaly Detection v2.0

Production-ready anomaly detection system with 8+ algorithms including Isolation Forest, One-Class SVM, and statistical methods.

## 🌟 Algorithms

### Statistical Methods
1. **Z-Score** - Standard deviation-based outlier detection
2. **IQR (Interquartile Range)** - Quartile-based outlier detection
3. **Modified Z-Score** - Robust outlier detection using MAD (Median Absolute Deviation)

### Machine Learning Methods
4. **Isolation Forest** - Tree-based anomaly detection (efficient for high dimensions)
5. **One-Class SVM** - Support Vector Machine for outlier detection
6. **Local Outlier Factor (LOF)** - Density-based local anomaly detection
7. **Elliptic Envelope** - Gaussian distribution-based (Robust Covariance)

### Clustering-Based
8. **DBSCAN** - Density-based clustering (noise points = anomalies)

### Ensemble
9. **Ensemble Voting** - Combines all algorithms for robust detection

## ✨ Key Features

- **8+ Detection Algorithms** with automatic comparison
- **Ensemble Voting** for robust anomaly detection
- **Statistical & ML Methods** for comprehensive coverage
- **Automatic Contamination** estimation and tuning
- **PCA Visualization** for 2D anomaly plotting
- **Anomaly Scoring** with score distributions
- **Model Persistence** (save/load trained models)
- **Comprehensive Evaluation** with Precision, Recall, F1-Score
- **Production-Ready** with scaling, error handling, and logging

## 🚀 Quick Start

### Basic Usage

```bash
python anomalydetection.py --data transactions.csv --contamination 0.1 --output-anomalies anomalies.csv --output-viz anomalies.png
```

### With Ground Truth Labels

```bash
python anomalydetection.py --data data.csv --contamination 0.05 --labels true_labels.csv --save-model model.pkl
```

## 📊 Example Code

```python
from anomalydetection import AnomalyDetector
import pandas as pd

# Load data
df = pd.read_csv('transactions.csv')
X = df.select_dtypes(include=[np.number]).values

# Initialize detector
detector = AnomalyDetector(contamination=0.1, scale_features=True)

# Fit all algorithms
detector.fit_all(X)

# Evaluate
evaluation_df = detector.evaluate()
print(evaluation_df)

# Get anomalies from best algorithm
anomaly_indices, scores = detector.get_anomalies(algorithm='Ensemble Voting')
print(f"Detected {len(anomaly_indices)} anomalies")

# Visualize
detector.plot_anomalies_pca(save_path='anomalies.png')
detector.plot_algorithm_comparison()
detector.plot_anomaly_scores()

# Save results
detector.save_anomalies('anomalies.csv', algorithm='Ensemble Voting')
detector.save_model('detector.pkl', algorithm='Isolation Forest')
```

## 🎯 Use Cases

### 🏦 Finance
- **Fraud Detection**: Credit card fraud, insurance fraud
- **Trading**: Unusual market behavior, manipulation detection
- **Risk Assessment**: Abnormal transaction patterns

### 🏭 Manufacturing
- **Quality Control**: Defective product detection
- **Predictive Maintenance**: Equipment failure prediction
- **Process Monitoring**: Production anomalies

### 🔐 Cybersecurity
- **Intrusion Detection**: Network anomaly detection
- **Log Analysis**: Unusual system behavior
- **User Behavior**: Insider threat detection

### 🏥 Healthcare
- **Patient Monitoring**: Vital sign anomalies
- **Disease Detection**: Unusual medical patterns
- **Claims Analysis**: Fraudulent insurance claims

### 🌐 IoT & Sensor Data
- **Sensor Monitoring**: Faulty sensor detection
- **Environmental**: Unusual weather patterns
- **Smart Cities**: Traffic/energy anomalies

## 📈 Algorithm Comparison

| Algorithm | Type | Complexity | High Dim | Pros | Cons |
|-----------|------|------------|----------|------|------|
| **Z-Score** | Statistical | O(n) | ❌ | Fast, interpretable | Assumes Gaussian |
| **IQR** | Statistical | O(n) | ❌ | Simple, robust | Univariate |
| **Modified Z-Score** | Statistical | O(n) | ❌ | Robust to outliers | Assumes symmetry |
| **Isolation Forest** | Tree-based | O(n log n) | ✅ | Fast, scalable | Black box |
| **One-Class SVM** | SVM | O(n²) to O(n³) | ✅ | Flexible kernel | Slow on large data |
| **LOF** | Density | O(n²) | ⚠️ | Local anomalies | Computationally expensive |
| **Elliptic Envelope** | Covariance | O(n²) | ⚠️ | Statistical foundation | Assumes Gaussian |
| **DBSCAN** | Clustering | O(n log n) | ⚠️ | No assumptions | Parameter sensitive |
| **Ensemble Voting** | Meta | Combined | ✅ | Most robust | Slower |

## 🎨 Visualization Examples

### 1. PCA Visualization
Shows anomalies in 2D space using Principal Component Analysis:
```python
detector.plot_anomalies_pca(save_path='pca_viz.png')
```

### 2. Algorithm Comparison
Compares number of anomalies detected by each algorithm:
```python
detector.plot_algorithm_comparison(save_path='comparison.png')
```

### 3. Anomaly Score Distributions
Shows distribution of anomaly scores for normal vs anomalous points:
```python
detector.plot_anomaly_scores(save_path='scores.png')
```

## 🔧 Advanced Configuration

### Custom Contamination Rate

```python
# Expected 5% of data to be anomalies
detector = AnomalyDetector(contamination=0.05)
```

### Algorithm-Specific Parameters

```python
# Z-Score with custom threshold
detector.detect_zscore(X, threshold=2.5)

# IQR with custom multiplier
detector.detect_iqr(X, multiplier=2.0)

# LOF with custom neighbors
detector.detect_lof(X, n_neighbors=30)

# DBSCAN with custom parameters
detector.detect_dbscan(X, eps=0.3, min_samples=10)
```

### Ensemble Voting Threshold

```python
# Require 70% of algorithms to agree
detector.ensemble_voting(threshold=0.7)
```

## 📊 Performance Metrics

When ground truth labels are available:

- **Precision**: Percentage of detected anomalies that are true anomalies
- **Recall**: Percentage of true anomalies that were detected
- **F1 Score**: Harmonic mean of Precision and Recall

```python
y_true = np.array([...])  # Ground truth labels (-1 = anomaly, 1 = normal)
evaluation_df = detector.evaluate(y_true)
print(evaluation_df)
```

## 📚 Algorithm Details

### Isolation Forest
- **Best for**: High-dimensional data, large datasets
- **Principle**: Isolates anomalies using random decision trees
- **Complexity**: O(n log n)
- **Parameters**: `contamination`, `n_estimators`

### One-Class SVM
- **Best for**: Complex decision boundaries
- **Principle**: Learns decision boundary around normal data
- **Complexity**: O(n²) to O(n³)
- **Parameters**: `kernel`, `nu` (contamination)

### Local Outlier Factor (LOF)
- **Best for**: Local anomalies with varying densities
- **Principle**: Compares local density with neighbors
- **Complexity**: O(n²)
- **Parameters**: `n_neighbors`, `contamination`

### Elliptic Envelope
- **Best for**: Gaussian-distributed data
- **Principle**: Fits robust covariance to data
- **Complexity**: O(n²)
- **Parameters**: `contamination`

### Statistical Methods
- **Z-Score**: Measures standard deviations from mean
- **IQR**: Uses interquartile range (Q1, Q3)
- **Modified Z-Score**: Uses median and MAD (more robust)

## 🛠️ Model Persistence

### Save Model

```python
detector.save_model('isolation_forest.pkl', algorithm='Isolation Forest')
```

### Load Model

```python
import joblib

model_data = joblib.load('isolation_forest.pkl')
model = model_data['model']
scaler = model_data['scaler']

# Predict on new data
X_new_scaled = scaler.transform(X_new)
predictions = model.predict(X_new_scaled)
```

## 💡 Best Practices

### 1. **Choose Contamination Rate Carefully**
- Start with domain knowledge (e.g., 1-5% for fraud)
- Use `contamination=0.1` as a reasonable default
- Adjust based on evaluation metrics

### 2. **Use Ensemble Voting**
- Most robust method for production
- Combines strengths of all algorithms
- Reduces false positives

### 3. **Scale Features**
- Essential for distance-based algorithms (LOF, One-Class SVM)
- Always enabled by default

### 4. **Interpret Results**
- Statistical methods (Z-Score, IQR) are interpretable
- Use them to understand WHY a point is anomalous
- ML methods are more accurate but less interpretable

### 5. **Validate with Ground Truth**
- Use labeled data if available
- Evaluate Precision/Recall tradeoff
- Adjust contamination based on metrics

## 🐛 Troubleshooting

**Too Many False Positives?**
- Increase contamination rate
- Use ensemble voting with higher threshold (0.6-0.7)
- Try more conservative algorithms (LOF, Elliptic Envelope)

**Too Many False Negatives?**
- Decrease contamination rate
- Use ensemble voting with lower threshold (0.3-0.4)
- Try Isolation Forest (good at finding subtle anomalies)

**Slow Performance?**
- Use Isolation Forest (fastest for large data)
- Avoid LOF and One-Class SVM for large datasets
- Sample data if necessary

**Poor Results on High-Dimensional Data?**
- Use Isolation Forest (handles high dimensions well)
- Apply dimensionality reduction first (PCA, t-SNE)
- Remove irrelevant features

## 📄 Dependencies

```bash
pip install numpy pandas scikit-learn scipy matplotlib seaborn joblib
```

## 🏆 Status

**Version:** 2.0
**Lines of Code:** 584
**Status:** Production-Ready ✅

**Features:**
- ✅ 8+ Detection Algorithms
- ✅ Ensemble Methods
- ✅ Statistical & ML Methods
- ✅ Comprehensive Evaluation
- ✅ Advanced Visualizations
- ✅ Model Persistence
- ✅ Production-Ready Code

## 📞 Support

**Author**: BrillConsulting | AI Consultant & Data Scientist
**Email**: clientbrill@gmail.com
**LinkedIn**: [BrillConsulting](https://www.linkedin.com/in/brillconsulting)

---

**⭐ Star this repository if you find it useful!**

*Made with ❤️ by BrillConsulting*
