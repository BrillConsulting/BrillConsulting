# 🌊 Online Learning

Incremental and online learning algorithms for streaming data and evolving environments.

## 🌟 Algorithms

### Incremental Classifiers
1. **Passive-Aggressive** - Margin-based online learning
2. **Perceptron** - Classic online algorithm
3. **SGDClassifier** - Stochastic gradient descent
4. **Naive Bayes (Incremental)** - Streaming Bayesian classifier

### Incremental Regressors
5. **SGDRegressor** - Online gradient descent regression
6. **Passive-Aggressive Regressor** - Robust regression
7. **Incremental Ridge** - Online regularized regression

### Adaptive Methods
8. **Hoeffding Tree** - Incremental decision tree
9. **Adaptive Random Forest** - Streaming ensemble
10. **Streaming K-Means** - Online clustering

### Drift Detection
11. **ADWIN** - Adaptive Windowing drift detector
12. **DDM** - Drift Detection Method
13. **EDDM** - Early Drift Detection Method

## ✨ Key Features

- **Incremental Learning** - Update models with new data batches
- **Streaming Support** - Process unbounded data streams
- **Memory Efficient** - Constant memory usage
- **Drift Detection** - Detect concept drift automatically
- **Adaptive Models** - Adjust to changing distributions
- **Real-Time Predictions** - Low latency inference
- **Model Evolution** - Continuous model updates

## 🚀 Quick Start

### Basic Online Learning

```bash
python online_learning.py --data stream_data.csv --algorithm sgd --batch-size 100
```

### With Drift Detection

```bash
python online_learning.py --data stream_data.csv --algorithm adaptive-rf --drift-detector adwin
```

## 📊 Example Code

```python
from online_learning import OnlineLearner
import pandas as pd

# Initialize online learner
learner = OnlineLearner(
    algorithm='sgd',
    task='classification',
    drift_detection=True,
    drift_detector='adwin'
)

# Simulate data stream
for batch in data_stream:
    X_batch, y_batch = batch

    # Make predictions
    predictions = learner.predict(X_batch)

    # Update model with new labels
    learner.partial_fit(X_batch, y_batch)

    # Check for drift
    if learner.drift_detected:
        print(f"Concept drift detected at sample {learner.n_samples_seen}")
        learner.reset_model()

# Get performance metrics
metrics = learner.get_metrics()
print(f"Accuracy: {metrics['accuracy']:.3f}")
```

## 🎯 Use Cases

### 📊 Financial Markets
- **Stock Price Prediction**: Adapt to market changes
- **Fraud Detection**: Real-time transaction monitoring
- **Trading Strategies**: Evolving market conditions
- **Credit Scoring**: Update models with new applications

### 🌐 Social Media
- **Trending Topics**: Real-time content classification
- **Sentiment Analysis**: Evolving language patterns
- **Recommendation**: User preference changes
- **Spam Detection**: Adaptive spam filtering

### 🏭 IoT & Sensors
- **Anomaly Detection**: Equipment monitoring
- **Predictive Maintenance**: Continuous sensor data
- **Energy Forecasting**: Changing consumption patterns
- **Quality Control**: Manufacturing processes

### 📱 User Behavior
- **Click Prediction**: Ad targeting
- **Churn Prediction**: Customer retention
- **Personalization**: Evolving preferences
- **A/B Testing**: Dynamic optimization

### 🚗 Autonomous Systems
- **Adaptive Control**: Changing environments
- **Traffic Prediction**: Real-time patterns
- **Route Optimization**: Dynamic conditions

## 📚 Algorithm Details

### SGD Classifier/Regressor
**Type**: Linear, online
**Best for**: Large-scale data streams
**Pros**: Fast, memory efficient, well-established
**Cons**: Requires feature scaling, sensitive to learning rate

```python
learner = OnlineLearner(
    algorithm='sgd',
    learning_rate='optimal',
    loss='log',  # or 'hinge', 'squared_loss'
    penalty='l2'
)
```

### Passive-Aggressive
**Type**: Margin-based, online
**Best for**: Robust updates, noisy data
**Pros**: Aggressive updates, handles noise
**Cons**: More parameters to tune

```python
learner = OnlineLearner(
    algorithm='passive-aggressive',
    C=1.0,  # Aggressiveness parameter
    loss='hinge'
)
```

### Hoeffding Tree
**Type**: Decision tree, streaming
**Best for**: Non-linear patterns, interpretability
**Pros**: Handles concept drift, no feature scaling needed
**Cons**: Slower than linear methods

```python
learner = OnlineLearner(
    algorithm='hoeffding-tree',
    grace_period=200,
    split_confidence=0.0001
)
```

### Adaptive Random Forest
**Type**: Ensemble, streaming
**Best for**: High accuracy, drift adaptation
**Pros**: Ensemble benefits, drift detection
**Cons**: Higher memory, slower updates

```python
learner = OnlineLearner(
    algorithm='adaptive-rf',
    n_estimators=10,
    drift_detection=True
)
```

## 🔧 Configuration

### Learning Parameters

```python
learner = OnlineLearner(
    algorithm='sgd',
    learning_rate='optimal',  # or constant, invscaling
    learning_rate_init=0.01,  # Initial rate if constant
    batch_size=100,           # Samples per update
    shuffle=True              # Shuffle each batch
)
```

### Drift Detection

```python
drift_config = {
    'detector': 'adwin',      # ADWIN, DDM, or EDDM
    'delta': 0.002,           # Confidence level (ADWIN)
    'warning_level': 2.0,     # Warning threshold (DDM)
    'drift_level': 3.0        # Drift threshold (DDM)
}

learner = OnlineLearner(
    algorithm='sgd',
    drift_detection=True,
    drift_config=drift_config
)
```

### Memory Management

```python
memory_config = {
    'max_samples': 1000,      # Maximum samples in memory
    'buffer_size': 500,       # Sliding window size
    'forget_factor': 0.95     # Exponential forgetting
}

learner = OnlineLearner(
    algorithm='sgd',
    memory_config=memory_config
)
```

## 📊 Streaming Workflow

### 1. Initialize Learner

```python
learner = OnlineLearner(
    algorithm='sgd',
    task='classification',
    drift_detection=True
)
```

### 2. Process Stream

```python
for batch in data_stream:
    X_batch, y_batch = batch

    # Predict on new data
    y_pred = learner.predict(X_batch)

    # Evaluate predictions
    accuracy = (y_pred == y_batch).mean()

    # Update model
    learner.partial_fit(X_batch, y_batch)

    # Check drift
    if learner.drift_detected:
        print(f"Drift at sample {learner.n_samples_seen}")
```

### 3. Monitor Performance

```python
# Get running metrics
metrics = learner.get_metrics()
print(f"Running accuracy: {metrics['accuracy']:.3f}")
print(f"Samples processed: {metrics['n_samples']}")
print(f"Drifts detected: {metrics['n_drifts']}")

# Plot learning curve
learner.plot_learning_curve()
```

### 4. Handle Drift

```python
if learner.drift_detected:
    # Option 1: Reset model
    learner.reset_model()

    # Option 2: Blend old and new models
    learner.adapt_to_drift(blend_factor=0.5)

    # Option 3: Keep model but retrain
    learner.retrain_on_buffer()
```

## 💡 Best Practices

### 1. **Choose Right Algorithm**
- Linear data: SGD, Passive-Aggressive
- Non-linear: Hoeffding Tree, Adaptive RF
- High drift: Adaptive methods with drift detection

### 2. **Batch Size Selection**
- Small batches (10-100): Faster adaptation, noisy updates
- Large batches (1000+): Stable updates, slower adaptation
- Rule of thumb: 100-500 samples per batch

### 3. **Learning Rate Tuning**
- Start with 'optimal' (adaptive)
- Use constant for stable streams
- Decrease for noisy data

### 4. **Drift Detection Strategy**
- Always use for non-stationary data
- ADWIN: Good all-around choice
- DDM: Faster, less sensitive
- EDDM: Detects gradual drift better

### 5. **Memory Management**
- Keep sliding window for retraining
- Use forgetting factors for old data
- Balance memory vs adaptation speed

### 6. **Evaluation Strategy**
- Use prequential evaluation (test-then-train)
- Track metrics over sliding windows
- Monitor drift frequency

## 🐛 Troubleshooting

**Model not adapting?**
- Increase learning rate
- Reduce batch size
- Check drift detection settings
- Verify data is actually changing

**Too many false drift detections?**
- Increase drift threshold (less sensitive)
- Use larger sliding window
- Try different drift detector
- Check for label noise

**Poor accuracy?**
- Increase batch size (more stable updates)
- Add regularization
- Try ensemble methods (Adaptive RF)
- Feature scaling for linear methods

**Memory issues?**
- Reduce buffer size
- Use forgetting factors
- Increase batch processing frequency
- Consider true streaming algorithms

**Slow updates?**
- Use linear methods (SGD, PA)
- Reduce batch size
- Simplify drift detection
- Parallel processing if available

## 📈 Algorithm Comparison

| Algorithm | Speed | Memory | Drift Handling | Non-Linear | Interpretability |
|-----------|-------|--------|----------------|------------|------------------|
| **SGD** | Very Fast ⚡ | Low | Manual | ❌ | Medium |
| **Passive-Aggressive** | Fast ⚡ | Low | Manual | ❌ | Medium |
| **Hoeffding Tree** | Medium | Medium | Good | ✅ | High |
| **Adaptive RF** | Slow | High | Excellent | ✅ | Low |
| **Streaming K-Means** | Fast ⚡ | Low | Manual | ✅ | Medium |
| **ADWIN** | Fast ⚡ | Medium | Excellent | N/A | N/A (Detector) |

## 🎓 Key Concepts

### Prequential Evaluation
Test-then-train: Evaluate on new data before using it for training
```python
for X_batch, y_batch in stream:
    # 1. Test
    y_pred = model.predict(X_batch)
    accuracy = (y_pred == y_batch).mean()

    # 2. Train
    model.partial_fit(X_batch, y_batch)
```

### Concept Drift Types
- **Sudden**: Abrupt distribution change
- **Gradual**: Slow transition between concepts
- **Incremental**: Small continuous changes
- **Recurring**: Patterns repeat over time

### Forgetting Strategies
- **Sliding Window**: Keep recent N samples
- **Fading Factor**: Exponentially weight recent data
- **Landmark Window**: Fixed historical point
- **Adaptive Window**: Size changes with drift

## 📊 Drift Detection Methods

### ADWIN (Adaptive Windowing)
- **Type**: Adaptive window
- **Best for**: General purpose
- **Pros**: No parameters, automatic
- **Cons**: Higher memory

```python
detector = 'adwin'
config = {'delta': 0.002}  # Confidence level
```

### DDM (Drift Detection Method)
- **Type**: Statistical test
- **Best for**: Sudden drift
- **Pros**: Fast, low memory
- **Cons**: Misses gradual drift

```python
detector = 'ddm'
config = {
    'warning_level': 2.0,
    'drift_level': 3.0
}
```

### EDDM (Early DDM)
- **Type**: Statistical test
- **Best for**: Gradual drift
- **Pros**: Early detection
- **Cons**: More false positives

## 📄 Dependencies

```bash
pip install numpy pandas scikit-learn matplotlib

# For advanced streaming algorithms
pip install river  # Modern streaming ML library

# For traditional incremental learning
pip install scikit-multiflow  # (deprecated but useful)
```

## 🏆 Status

**Version:** 1.0
**Status:** Research/Educational

**Features:**
- ✅ SGD Classifier/Regressor
- ✅ Passive-Aggressive Methods
- ✅ Incremental Naive Bayes
- ✅ Drift Detection (ADWIN, DDM, EDDM)
- ✅ Prequential Evaluation
- ✅ Visualization Tools
- ⚠️ Hoeffding Trees - Planned
- ⚠️ Adaptive Random Forest - Planned

## 📞 Support

**Author**: BrillConsulting | AI Consultant & Data Scientist
**Email**: clientbrill@gmail.com
**LinkedIn**: [BrillConsulting](https://www.linkedin.com/in/brillconsulting)

---

**⭐ Star this repository if you find it useful!**

*Made with ❤️ by BrillConsulting*
