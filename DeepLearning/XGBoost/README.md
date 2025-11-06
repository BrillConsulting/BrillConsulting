# XGBoost Gradient Boosting

## 🎯 Overview

Advanced XGBoost implementations featuring classification, regression, hyperparameter tuning, cross-validation, and feature importance analysis with GPU acceleration support.

## ✨ Features

### Model Training
- **Classification**: Multi-class and binary classification with softmax/logistic
- **Regression**: Gradient boosted regression trees
- **Custom Objectives**: User-defined loss functions
- **Early Stopping**: Prevent overfitting with validation-based stopping

### Optimization
- **Hyperparameter Tuning**: Grid search across parameter space
- **Cross-Validation**: K-fold CV with stratification
- **Feature Selection**: Importance-based feature ranking
- **Regularization**: L1/L2 penalties for tree complexity

### Performance
- **GPU Acceleration**: CUDA-based tree construction
- **Parallel Processing**: Multi-threaded CPU training
- **Column Subsampling**: Speed up training with feature sampling
- **Histogram Binning**: Efficient gradient statistics

### Analysis
- **Feature Importance**: Gain, cover, and frequency metrics
- **SHAP Values**: Model interpretation with SHAP
- **Learning Curves**: Track training/validation metrics
- **Tree Visualization**: Inspect individual trees

## 📋 Requirements

```bash
pip install xgboost>=1.7.0 scikit-learn numpy pandas matplotlib
# For GPU support
pip install xgboost[gpu]
```

## 🚀 Quick Start

```python
from xgboost_models import XGBoostManager

# Initialize manager
mgr = XGBoostManager()

# Train classifier
clf_result = mgr.train_classifier({
    'num_class': 10,
    'max_depth': 6,
    'learning_rate': 0.1,
    'n_estimators': 100
})

# Train regressor
reg_result = mgr.train_regressor({
    'max_depth': 5,
    'learning_rate': 0.05,
    'n_estimators': 150
})

# Hyperparameter tuning
tuning_result = mgr.hyperparameter_tuning({
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.3],
    'n_estimators': [50, 100, 200]
})

# Cross-validation
cv_result = mgr.cross_validation({
    'n_folds': 5,
    'params': {'max_depth': 6, 'learning_rate': 0.1}
})

# Feature importance
importance = mgr.feature_importance()
```

## 🏗️ Training Workflow

### Classification Pipeline
```
1. Create DMatrix from data
2. Set parameters (objective, num_class, etc.)
3. Train with xgb.train()
4. Monitor metrics on validation set
5. Early stopping if no improvement
6. Save best model
```

### Hyperparameter Tuning
```
1. Define parameter grid
2. For each combination:
   - Train model with params
   - Evaluate on validation set
   - Track best score
3. Return best parameters and model
```

## 💡 Use Cases

- **Classification**: Fraud detection, customer churn, image classification
- **Regression**: House price prediction, demand forecasting, scoring
- **Ranking**: Search result ranking, recommendation systems
- **Anomaly Detection**: Outlier detection with one-class classification

## 📊 Parameter Guide

### Tree Parameters
```python
params = {
    'max_depth': 6,           # Maximum tree depth
    'min_child_weight': 1,    # Minimum sum of instance weight
    'gamma': 0,               # Minimum loss reduction for split
    'subsample': 0.8,         # Row sampling ratio
    'colsample_bytree': 0.8,  # Column sampling ratio
}
```

### Regularization
```python
params = {
    'lambda': 1,              # L2 regularization (ridge)
    'alpha': 0,               # L1 regularization (lasso)
    'max_delta_step': 0,      # Maximum delta step for weight estimation
}
```

### Learning Control
```python
params = {
    'learning_rate': 0.1,     # Step size shrinkage (eta)
    'n_estimators': 100,      # Number of boosting rounds
    'objective': 'multi:softmax',  # Loss function
}
```

## 🔬 Advanced Features

### Feature Importance
```python
import xgboost as xgb

# Train model
model = xgb.train(params, dtrain, num_boost_round=100)

# Get importance scores
importance = model.get_score(importance_type='gain')

# Importance types:
# - 'gain': Average gain across splits using the feature
# - 'weight': Number of times feature appears in trees
# - 'cover': Average coverage of splits using the feature
```

### Cross-Validation
```python
cv_results = xgb.cv(
    params,
    dtrain,
    num_boost_round=100,
    nfold=5,
    metrics=['auc', 'logloss'],
    early_stopping_rounds=10,
    stratified=True
)
```

### GPU Acceleration
```python
params = {
    'tree_method': 'gpu_hist',  # GPU histogram algorithm
    'gpu_id': 0,                # GPU device ID
    'predictor': 'gpu_predictor'  # GPU prediction
}

model = xgb.train(params, dtrain)
```

### Custom Objective
```python
def custom_objective(y_pred, dtrain):
    y_true = dtrain.get_label()
    grad = y_pred - y_true
    hess = np.ones_like(y_true)
    return grad, hess

model = xgb.train(
    params,
    dtrain,
    obj=custom_objective
)
```

## 📊 Performance Metrics

### Training Speed
- CPU (8 cores): ~10K samples/sec
- GPU (CUDA): ~100K samples/sec (10x speedup)
- Histogram method: 2-3x faster than exact

### Model Quality
- Multi-class classification: 90-95% accuracy (typical)
- Regression: R² > 0.85 (typical)
- Feature importance: Top 20% features often capture 80% of signal

### Hyperparameter Impact
- `max_depth` ↑ → More complex, risk overfitting
- `learning_rate` ↓ → Better generalization, slower training
- `subsample` < 1 → Prevent overfitting, add randomness

## 🎯 Best Practices

1. **Start Simple**: Begin with default parameters
2. **Feature Engineering**: Create meaningful features
3. **Cross-Validate**: Always validate on held-out data
4. **Tune Systematically**: Grid search → Random search → Bayesian
5. **Monitor Overfitting**: Watch train/validation gap
6. **Scale Features**: Normalize for better performance
7. **Handle Imbalance**: Use `scale_pos_weight` for imbalanced data

## 📚 References

- XGBoost Documentation: https://xgboost.readthedocs.io
- Original Paper: "XGBoost: A Scalable Tree Boosting System" (Chen & Guestrin, 2016)
- Parameter Tuning: https://xgboost.readthedocs.io/en/latest/parameter.html
- GPU Training: https://xgboost.readthedocs.io/en/latest/gpu/index.html

## 📧 Contact

For questions or collaboration: [clientbrill@gmail.com](mailto:clientbrill@gmail.com)

---

**Author:** Brill Consulting
**LinkedIn:** [brillconsulting](https://www.linkedin.com/in/brillconsulting)
