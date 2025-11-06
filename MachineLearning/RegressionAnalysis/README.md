# 📈 Advanced Regression Analysis System v2.0

**Production-ready regression with 10+ algorithms including XGBoost, LightGBM, and ensemble methods**

Comprehensive regression analysis system featuring multiple algorithms with automatic model selection, hyperparameter tuning, and advanced diagnostics.

## 🌟 Key Features

- **10+ Algorithms**: Linear, Ridge, Lasso, Polynomial, Random Forest, Gradient Boosting, SVR, XGBoost, LightGBM
- **Automatic Hyperparameter Tuning**: Grid search for optimal parameters across all models
- **Model Comparison**: Side-by-side performance metrics with automatic best model selection
- **Advanced Diagnostics**: Residual analysis, feature importance, normality tests
- **Feature Scaling**: Automatic standardization
- **Visualization**: Prediction vs actual plots for all models
- **Model Persistence**: Save/load trained models

## 📦 Installation

```bash
pip install -r requirements.txt
```

## 🚀 Quick Start

### Basic Usage

```bash
python regression_models.py \
    --data housing_data.csv \
    --target price \
    --output results.png
```

### With Model Saving

```bash
python regression_models.py \
    --data data.csv \
    --target target_column \
    --test-size 0.2 \
    --save-model best_model.pkl \
    --output comparison.png
```

## 🎛️ Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--data` | Required | Path to CSV file with data |
| `--target` | Required | Name of target column |
| `--test-size` | `0.2` | Proportion of test set (0-1) |
| `--output` | - | Path to save comparison plot |
| `--save-model` | - | Path to save best model |

## 📊 Algorithms

### Linear Models

#### 1. Linear Regression
- **Use**: Simple linear relationships
- **Pros**: Fast, interpretable, baseline model
- **Cons**: Assumes linearity, sensitive to outliers

#### 2. Ridge Regression (L2 Regularization)
- **Use**: Multicollinearity, prevent overfitting
- **Pros**: Handles correlated features well, stable coefficients
- **Hyperparameter**: alpha (regularization strength: 0.001-100)

#### 3. Lasso Regression (L1 Regularization)
- **Use**: Feature selection, sparse models
- **Pros**: Automatic feature selection, interpretable
- **Hyperparameter**: alpha (regularization strength: 0.001-10)

#### 4. Polynomial Regression
- **Use**: Non-linear relationships
- **Pros**: Captures complex patterns without feature engineering
- **Hyperparameter**: degree (polynomial degree: 2-4)

### Tree-Based Models

#### 5. Random Forest ⭐
- **Use**: Non-linear relationships, robust to outliers
- **Pros**: High accuracy, feature importance, handles missing values
- **Hyperparameters**: n_estimators (50-200), max_depth (10-30)
- **Performance**: Often best for structured data

#### 6. Gradient Boosting ⭐
- **Use**: High accuracy regression tasks
- **Pros**: State-of-the-art performance, adaptive
- **Hyperparameters**: n_estimators (50-200), learning_rate (0.01-0.1), max_depth (3-7)
- **Performance**: Excellent for competition-grade models

#### 7. XGBoost ⭐ (Optional)
- **Use**: Maximum performance, large datasets
- **Pros**: Fastest gradient boosting, regularization, GPU support
- **Hyperparameters**: n_estimators, learning_rate, max_depth, subsample, colsample_bytree
- **Performance**: Industry standard for structured data
- **Note**: Requires `pip install xgboost`

#### 8. LightGBM ⭐ (Optional)
- **Use**: Very large datasets, fast training
- **Pros**: 10-20x faster than traditional GBDT, low memory
- **Hyperparameters**: n_estimators, learning_rate, num_leaves, max_depth
- **Performance**: Best for datasets >10K samples
- **Note**: Requires `pip install lightgbm`

### Other Methods

#### 9. Support Vector Regression (SVR)
- **Use**: Non-linear patterns, robust predictions
- **Pros**: Effective in high dimensions, kernel trick
- **Hyperparameters**: C (0.1-100), kernel (rbf/linear), gamma
- **Performance**: Good for small-medium datasets

## 📝 Example Code

### Python API

```python
from regression_models import RegressionAnalyzer
import pandas as pd

# Load data
df = pd.read_csv('data.csv')
X = df.drop(columns=['target'])
y = df['target']

# Initialize analyzer
analyzer = RegressionAnalyzer(scale_features=True)

# Prepare data
X_train, X_test, y_train, y_test = analyzer.prepare_data(X, y)

# Train all models
analyzer.train_all_models(X_train, y_train, X_test, y_test)

# Compare models
comparison = analyzer.compare_models()
print(comparison)

# Best model
print(f"Best: {analyzer.best_model_name}")
print(f"R²: {analyzer.results[analyzer.best_model_name]['test_r2']:.4f}")

# Plot results
analyzer.plot_results(X_test, y_test, save_path='results.png')

# Save best model
analyzer.save_model(analyzer.best_model_name, 'model.pkl')
```

### Train Individual Models

```python
# Linear Regression
results = analyzer.train_linear_regression(X_train, y_train, X_test, y_test)

# Ridge with custom alphas
results = analyzer.train_ridge_regression(
    X_train, y_train, X_test, y_test,
    alpha_range=[0.1, 1, 10, 100]
)

# Lasso
results = analyzer.train_lasso_regression(X_train, y_train, X_test, y_test)

# Polynomial (degree 3)
results = analyzer.train_polynomial_regression(
    X_train, y_train, X_test, y_test,
    degree=3
)
```

## 📊 Evaluation Metrics

All models are evaluated using:

- **R² Score**: Proportion of variance explained (higher is better)
- **RMSE**: Root Mean Squared Error (lower is better)
- **MAE**: Mean Absolute Error (lower is better)

Metrics calculated for both training and test sets to detect overfitting.

## 🎨 Use Cases

### Real Estate
- House price prediction
- Rent estimation
- Property valuation

### Finance
- Stock price prediction
- Credit scoring
- Risk assessment

### Healthcare
- Medical cost prediction
- Treatment outcome prediction
- Disease progression modeling

### Business
- Sales forecasting
- Customer lifetime value
- Demand prediction

## 📈 Sample Output

```
📊 Model Comparison:
================================================================================
                             Model  Train R²  Test R²  Train RMSE  Test RMSE
                  Ridge Regression    0.9234   0.9156      12.45      13.21
                 Linear Regression    0.9187   0.9143      12.89      13.34
                 Lasso Regression     0.9156   0.9098      13.12      13.67
Polynomial Regression (degree=2)     0.9456   0.8987      10.54      14.52
================================================================================

🏆 Best Model: Ridge Regression
   Test R²: 0.9156
   Test RMSE: 13.21
```

## 🔧 Advanced Features

### Custom Feature Engineering

```python
# Add polynomial features manually
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=3, include_bias=False)
X_poly = poly.fit_transform(X)

# Train on engineered features
X_train, X_test, y_train, y_test = analyzer.prepare_data(
    pd.DataFrame(X_poly), y
)
```

### Cross-Validation

```python
from sklearn.model_selection import cross_val_score

# Get cross-validation scores
cv_scores = cross_val_score(
    analyzer.models['Linear Regression'],
    X_train, y_train,
    cv=5,
    scoring='r2'
)

print(f"CV R²: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
```

### Feature Importance (Lasso)

```python
# Get feature importance from Lasso
lasso_model = analyzer.models['Lasso Regression']
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'coefficient': lasso_model.coef_
})

# Filter non-zero features
selected = feature_importance[feature_importance['coefficient'] != 0]
print(f"Selected {len(selected)} features out of {len(X.columns)}")
```

## 🐛 Troubleshooting

**Poor R² score**:
- Try polynomial regression for non-linear data
- Check for outliers
- Add more features
- Collect more data

**Overfitting (high train R², low test R²)**:
- Use Ridge or Lasso regularization
- Reduce polynomial degree
- Collect more training data
- Remove irrelevant features

**High RMSE**:
- Check target variable distribution
- Try log transformation for skewed targets
- Remove outliers
- Normalize features

## 📚 Theory

### Ridge Regression
Minimizes: `RSS + α * Σ(β²)`
- Shrinks coefficients but never to zero
- Good when all features are relevant

### Lasso Regression
Minimizes: `RSS + α * Σ|β|`
- Can shrink coefficients to exactly zero
- Performs feature selection
- Good for high-dimensional data

### Polynomial Regression
Transforms features: `[x₁, x₂] → [x₁, x₂, x₁², x₁x₂, x₂²]`
- Captures non-linear relationships
- Risk of overfitting with high degrees

## 📄 License

MIT License - Free for commercial and research use

---

**Author**: BrillConsulting | AI Consultant & Data Scientist
**Contact**: clientbrill@gmail.com
