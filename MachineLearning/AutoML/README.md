# 🤖 Advanced AutoML System

Automated Machine Learning with hyperparameter optimization and intelligent model selection.

## 📋 Overview

Complete AutoML system that automatically selects the best model and hyperparameters for your dataset.

## ✨ Key Features

- **Auto-Task Detection**: Automatically detects classification vs regression
- **Model Selection**: Tests 6+ algorithms per task type
- **Hyperparameter Optimization**: RandomizedSearchCV with extensive parameter grids
- **Cross-Validation**: Robust performance estimation
- **Time-Limited**: Configurable optimization time budget
- **Model Ranking**: Compare all tested models

## 🚀 Quick Start

```python
from automl import AutoML
from sklearn.datasets import load_breast_cancer

# Load data
X, y = load_breast_cancer(return_X_y=True)

# Run AutoML
automl = AutoML(task_type='auto', time_limit=300, n_iter=20, cv=5)
automl.fit(X, y)

# Make predictions
predictions = automl.predict(X_test)

# Get model ranking
print(automl.get_model_ranking())
```

## 📊 Supported Algorithms

**Classification:**
- Random Forest, Gradient Boosting
- Logistic Regression, SVM
- K-Nearest Neighbors, Decision Tree

**Regression:**
- Random Forest, Gradient Boosting  
- Ridge, Lasso, SVR
- K-Nearest Neighbors

## 📧 Contact

**Author**: BrillConsulting | **Email**: clientbrill@gmail.com
