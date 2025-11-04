# Feature Engineering Toolkit

Comprehensive toolkit for feature creation, transformation, and selection to improve model performance.

## Features

- **Numerical Transformations**: Scaling (Standard, MinMax, Robust), polynomial features, binning
- **Interaction Features**: Create multiplicative and division features
- **Categorical Encoding**: One-hot, label, frequency encoding
- **Date/Time Features**: Extract temporal patterns
- **Aggregation Features**: Group-based statistics
- **Feature Selection**: Univariate tests, RFE, feature importance
- **Automated Pipelines**: End-to-end feature engineering

## Technologies

- Scikit-learn: Preprocessing and feature selection
- Pandas, NumPy: Data manipulation

## Usage

```python
from feature_engineering import FeatureEngineer

fe = FeatureEngineer()

# Scale features
scaled = fe.scale_features(data, ['col1', 'col2'], method='standard')

# Create polynomial features
poly = fe.create_polynomial_features(data, ['col1'], degree=2)

# Encode categorical
encoded = fe.encode_categorical(data, ['cat_col'], method='onehot')

# Feature selection
X_selected, features = fe.select_features_univariate(X, y, k=10)
```

## Demo

```bash
python feature_engineering.py
```
