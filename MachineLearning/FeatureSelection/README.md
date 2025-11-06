# 🔍 Advanced Feature Selection System

Production-ready feature selection framework with multiple algorithms and automatic ensemble voting.

## 📋 Overview

Comprehensive feature selection system supporting **Filter**, **Wrapper**, **Embedded**, and **Hybrid** methods for both classification and regression tasks.

## ✨ Key Features

### 1. Filter Methods
- **Univariate Statistical Tests**: F-test, Chi-square, ANOVA
- **Mutual Information**: Non-linear dependency detection
- **Variance Threshold**: Remove low-variance features

### 2. Wrapper Methods
- **Recursive Feature Elimination (RFE)**: Iterative feature removal
- **RFE with Cross-Validation (RFECV)**: Automatic optimal feature count
- **Sequential Feature Selection**: Forward/backward selection

### 3. Embedded Methods
- **L1 Regularization (Lasso)**: Sparse feature selection
- **Tree-based Importance**: Random Forest, Gradient Boosting
- **Permutation Importance**: Model-agnostic importance

### 4. Ensemble Methods
- **Voting**: Combine multiple selection methods
- **Stable Features**: Features selected by majority of methods
- **Agreement Analysis**: Visualize method consensus

## 🚀 Quick Start

### Basic Usage

```python
from feature_selection import FeatureSelector
import pandas as pd

# Load data
X = pd.read_csv('features.csv')
y = pd.read_csv('target.csv').values.ravel()

# Initialize selector
selector = FeatureSelector(task_type='classification', random_state=42)
selector.fit(X, y)

# Run feature selection
result = selector.univariate_selection(X, y, k=10)
print(f"Selected features: {result.selected_features}")
```

### Multiple Methods Comparison

```python
# Run multiple methods
selector.univariate_selection(X, y, k=15)
selector.mutual_information_selection(X, y, k=15)
selector.tree_importance_selection(X, y, threshold='mean')
selector.rfe_selection(X, y, n_features_to_select=15, use_cv=True)

# Compare results
comparison = selector.compare_methods()
print(comparison)

# Get stable features (selected by ≥70% of methods)
stable_features = selector.get_stable_features(min_agreement=0.7)
print(f"Stable features: {stable_features}")
```

## 📊 Method Selection Guide

| Method | Best For | Pros | Cons |
|--------|----------|------|------|
| **Univariate** | Quick screening | Fast, simple | Ignores interactions |
| **Mutual Information** | Non-linear relationships | Detects dependencies | Computationally expensive |
| **RFE** | Optimal subset | Considers interactions | Slow for large datasets |
| **Tree Importance** | General purpose | Fast, interpretable | Can be biased |
| **Lasso** | High-dimensional | Automatic, fast | Assumes linearity |
| **Permutation** | Model-agnostic | Unbiased | Very slow |
| **Ensemble** | Production use | Robust, stable | Requires multiple runs |

## 📧 Contact

**Author**: BrillConsulting | **Email**: clientbrill@gmail.com
