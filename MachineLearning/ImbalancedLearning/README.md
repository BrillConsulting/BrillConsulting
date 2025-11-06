# ⚖️ Advanced Imbalanced Learning System

Production-ready techniques for handling imbalanced datasets with multiple resampling strategies.

## 📋 Overview

Comprehensive system for dealing with class imbalance using **SMOTE**, **ADASYN**, **Under-sampling**, and **Hybrid** approaches.

## ✨ Key Features

### 1. Over-sampling Techniques
- **SMOTE** (Synthetic Minority Over-sampling Technique)
- **ADASYN** (Adaptive Synthetic Sampling)

### 2. Under-sampling Techniques  
- **Random Under-sampling**
- **Tomek Links** removal

### 3. Hybrid Methods
- **SMOTE + Tomek Links**: Combined over-sampling and cleaning

### 4. Integrated Classifier
- **ImbalancedClassifier**: Wrapper combining resampling + classification

## 🚀 Quick Start

```python
from imbalanced_learning import SMOTE, ADASYN, ImbalancedClassifier
import numpy as np

# Load imbalanced data
X, y = load_data()  # e.g., 90% class 0, 10% class 1

# Option 1: Direct resampling with SMOTE
smote = SMOTE(sampling_ratio=1.0, k_neighbors=5)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Option 2: Integrated classifier
clf = ImbalancedClassifier(
    sampling_strategy='smote',
    sampling_ratio=1.0
)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
```

## 📊 Methods Comparison

| Method | Type | Pros | Cons |
|--------|------|------|------|
| **SMOTE** | Over-sampling | Creates synthetic samples | May create noise |
| **ADASYN** | Over-sampling | Focuses on hard regions | Computationally expensive |
| **Under-sampling** | Under-sampling | Fast, reduces data size | Loses information |
| **Tomek Links** | Cleaning | Removes boundary noise | Minimal balancing |
| **SMOTE+Tomek** | Hybrid | Balanced + clean boundaries | Slower |

## 💡 Usage Examples

### Example: SMOTE with Classification

```python
from sklearn.ensemble import RandomForestClassifier

# Create imbalanced classifier with SMOTE
clf = ImbalancedClassifier(
    base_estimator=RandomForestClassifier(n_estimators=100),
    sampling_strategy='smote',
    sampling_ratio=1.0,
    k_neighbors=5
)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
```

### Example: Compare All Strategies

```python
from imbalanced_learning import ImbalancedLearningAnalyzer

analyzer = ImbalancedLearningAnalyzer()
comparison = analyzer.compare_sampling_strategies(X, y, cv=5)
print(comparison)
```

## 📧 Contact

**Author**: BrillConsulting | **Email**: clientbrill@gmail.com
