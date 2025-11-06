# LightGBM Gradient Boosting Framework

## 🎯 Overview

High-performance LightGBM implementations with native categorical feature support, advanced hyperparameter tuning, feature importance analysis, and GPU acceleration for lightning-fast gradient boosting.

## ✨ Features

### Core Capabilities
- **Fast Training**: Histogram-based algorithm for speed
- **Low Memory**: Efficient memory usage with binning
- **Native Categorical**: Direct categorical feature handling (no encoding needed!)
- **GPU Acceleration**: CUDA support for massive speedup
- **GOSS & EFB**: Gradient-based one-side sampling and exclusive feature bundling

### Model Types
- **Classification**: Binary and multi-class classification
- **Regression**: Gradient boosted regression
- **Ranking**: LambdaRank for learning to rank
- **Custom Objectives**: User-defined loss functions

### Optimization
- **Hyperparameter Tuning**: Grid search with early stopping
- **Cross-Validation**: K-fold CV with categorical awareness
- **Feature Importance**: Split and gain-based importance
- **Automatic Tuning**: Dart mode for dropouts

### Performance
- **Speed**: 10-20x faster than traditional GBDT
- **Accuracy**: Better accuracy with leaf-wise growth
- **Scalability**: Handle millions of samples efficiently
- **Parallel Learning**: Multi-threading and distributed training

## 📋 Requirements

```bash
pip install lightgbm>=3.3.0 scikit-learn numpy pandas matplotlib
# For GPU support
pip install lightgbm --install-option=--gpu
```

## 🚀 Quick Start

```python
from lightgbm_models import LightGBMManager

# Initialize manager
mgr = LightGBMManager()

# Train classifier
clf_result = mgr.train_classifier({
    'num_class': 10,
    'num_leaves': 31,
    'learning_rate': 0.05,
    'num_boost_round': 100
})

# Train regressor
reg_result = mgr.train_regressor({
    'num_leaves': 31,
    'learning_rate': 0.05,
    'num_boost_round': 100
})

# Train with categorical features (LightGBM specialty!)
cat_result = mgr.train_with_categorical({
    'categorical_features': ['category', 'region', 'type']
})

# Hyperparameter tuning
tuning_result = mgr.hyperparameter_tuning({
    'num_leaves': [15, 31, 63],
    'learning_rate': [0.01, 0.05, 0.1],
    'feature_fraction': [0.7, 0.8, 0.9]
})

# Feature importance
importance = mgr.feature_importance()
```

## 🏗️ Training Workflow

### Basic Training
```
1. Create lgb.Dataset with labels
2. Set parameters (objective, metric, etc.)
3. Train with lgb.train()
4. Monitor metrics with callbacks
5. Early stopping based on validation
6. Save model
```

### Categorical Feature Handling
```
1. Mark categorical features by name or index
2. LightGBM handles them natively (no encoding!)
3. Optimal splits found for categorical values
4. Better performance than one-hot encoding
```

## 💡 Use Cases

- **Tabular Data**: Structured data with categorical features
- **Click Prediction**: Ad click-through rate prediction
- **Ranking**: Search engines, recommendation systems
- **Time Series**: Fast training for forecasting
- **Large Datasets**: Millions of samples, thousands of features

## 📊 Parameter Guide

### Tree Structure
```python
params = {
    'num_leaves': 31,          # Max leaves in one tree
    'max_depth': -1,           # No limit (leaf-wise growth)
    'min_data_in_leaf': 20,    # Min samples per leaf
    'min_child_weight': 0.001, # Min sum of instance weight
}
```

### Learning Parameters
```python
params = {
    'learning_rate': 0.05,     # Shrinkage rate
    'num_boost_round': 100,    # Number of trees
    'objective': 'multiclass', # Loss function
    'metric': 'multi_logloss', # Evaluation metric
}
```

### Speed & Accuracy
```python
params = {
    'feature_fraction': 0.9,   # Column sampling
    'bagging_fraction': 0.8,   # Row sampling
    'bagging_freq': 5,         # Bagging frequency
    'num_threads': 8,          # Parallel threads
}
```

### Categorical Features
```python
params = {
    'categorical_feature': ['cat_col1', 'cat_col2'],
    # Or use column indices: [0, 3, 5]
}
```

## 🔬 Advanced Features

### Native Categorical Support
```python
import lightgbm as lgb

# Categorical features handled automatically
train_data = lgb.Dataset(
    X_train,
    label=y_train,
    categorical_feature=['category', 'region', 'product_type']
)

params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'categorical_feature': 'auto'  # Auto-detect
}

model = lgb.train(params, train_data)
```

### GPU Training
```python
params = {
    'device': 'gpu',
    'gpu_platform_id': 0,
    'gpu_device_id': 0,
    'num_gpu': 1
}

model = lgb.train(params, train_data)
```

### Early Stopping
```python
model = lgb.train(
    params,
    train_data,
    num_boost_round=1000,
    valid_sets=[train_data, valid_data],
    valid_names=['train', 'valid'],
    callbacks=[
        lgb.early_stopping(stopping_rounds=50),
        lgb.log_evaluation(period=10)
    ]
)
```

### Feature Importance
```python
# Get importance (split-based)
importance = model.feature_importance(importance_type='split')

# Get importance (gain-based)
importance = model.feature_importance(importance_type='gain')

# Plot importance
lgb.plot_importance(model, max_num_features=20)
```

## 📊 Performance Comparison

### Speed (vs XGBoost)
- CPU training: 10-20x faster
- Memory usage: 50-70% less
- GPU training: 5-10x faster

### Accuracy
- Leaf-wise growth: Better accuracy
- Handles categorical: No information loss from encoding
- Large datasets: Maintains accuracy at scale

### Categorical Features
- **LightGBM**: Native support, optimal splits
- **XGBoost**: Requires one-hot encoding (high-dimensional)
- **Traditional**: Label encoding loses order information

## 🎯 Best Practices

1. **Use Categorical Features**: Mark categorical columns for best performance
2. **Leaf-wise Growth**: More accurate than depth-wise
3. **Early Stopping**: Prevent overfitting with validation monitoring
4. **Feature Fraction**: Use < 1.0 for regularization
5. **Bagging**: Combine with feature fraction for robustness
6. **GPU for Large Data**: 10x+ speedup on millions of samples
7. **Cross-Validation**: Always validate hyperparameters

## 🚀 Why LightGBM?

### Advantages
- ⚡ **Fastest GBDT**: Histogram-based, leaf-wise growth
- 💾 **Memory Efficient**: Binning reduces memory usage
- 🏷️ **Native Categorical**: No preprocessing needed
- 📈 **Scalable**: Handles millions of rows efficiently
- 🎯 **Accurate**: State-of-the-art results on tabular data

### When to Use
- ✅ Tabular data with many categorical features
- ✅ Large datasets (>100K samples)
- ✅ Need fast training iterations
- ✅ Limited memory resources
- ✅ Competition-level accuracy required

## 📚 References

- LightGBM Documentation: https://lightgbm.readthedocs.io
- Original Paper: "LightGBM: A Highly Efficient Gradient Boosting Decision Tree" (Ke et al., 2017)
- Parameter Tuning: https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html
- Categorical Features: https://lightgbm.readthedocs.io/en/latest/Advanced-Topics.html#categorical-feature-support

## 📧 Contact

For questions or collaboration: [clientbrill@gmail.com](mailto:clientbrill@gmail.com)

---

**Author:** Brill Consulting
**LinkedIn:** [brillconsulting](https://www.linkedin.com/in/brillconsulting)
