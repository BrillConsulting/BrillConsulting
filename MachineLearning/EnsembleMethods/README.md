# 🎭 Advanced Ensemble Methods v2.0

**Production-ready ensemble learning with 10+ methods for classification and regression**

Comprehensive ensemble learning system featuring bagging, boosting, voting, and stacking methods with automatic model selection, hyperparameter tuning, and advanced evaluation.

## 🌟 Key Features

- **10+ Ensemble Methods**: Bagging, Random Forest, Extra Trees, AdaBoost, Gradient Boosting, XGBoost, LightGBM, Voting, Stacking
- **Dual Task Support**: Both classification and regression
- **Automatic Model Selection**: Compares all methods and selects best performer
- **Hyperparameter Tuning**: GridSearchCV for optimal parameters
- **Feature Importance**: Analysis for tree-based methods
- **Out-of-Bag (OOB) Error**: Validation without separate test set
- **Model Persistence**: Save/load trained ensembles
- **Visualization**: Performance comparison and feature importance plots

## 📦 Installation

```bash
pip install -r requirements.txt

# Optional dependencies for advanced boosting
pip install xgboost lightgbm
```

## 🚀 Quick Start

### Basic Classification

```bash
python ensemble_models.py \
    --data data.csv \
    --target label \
    --task classification
```

### Regression with Tuning

```bash
python ensemble_models.py \
    --data housing.csv \
    --target price \
    --task regression \
    --tune \
    --output comparison.png \
    --save-model best_ensemble.pkl
```

### Run Demo

```bash
python ensemble_models.py --demo
```

## 🎛️ Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--data` | Required | Path to CSV data file |
| `--target` | Required | Target column name |
| `--task` | `classification` | Task type: 'classification' or 'regression' |
| `--test-size` | `0.2` | Proportion of test set (0-1) |
| `--tune` | - | Enable hyperparameter tuning |
| `--output` | - | Path to save comparison plot |
| `--save-model` | - | Path to save best model |
| `--demo` | - | Run demonstration with synthetic data |

## 📊 Ensemble Methods

### Bagging Methods

#### 1. Bagging (Bootstrap Aggregating) ⭐
- **Use**: Reduce variance, prevent overfitting
- **Method**: Train multiple models on bootstrap samples, average predictions
- **Pros**: Simple, parallel training, OOB error estimation
- **Parameters**: n_estimators (100), max_samples (1.0)
- **Best For**: High-variance base learners (deep trees)

#### 2. Random Forest ⭐⭐
- **Use**: General-purpose ensemble, feature importance
- **Method**: Bagging + random feature selection at each split
- **Pros**: Robust, handles non-linear relationships, built-in feature importance
- **Tuning**: n_estimators, max_depth, min_samples_split, min_samples_leaf
- **Best For**: Most structured data problems

#### 3. Extra Trees (Extremely Randomized Trees)
- **Use**: Faster Random Forest alternative
- **Method**: Random splits instead of best splits
- **Pros**: Faster training, more randomization, less overfitting
- **Parameters**: n_estimators (100)
- **Best For**: Large datasets where training speed matters

### Boosting Methods

#### 4. AdaBoost (Adaptive Boosting) ⭐
- **Use**: Sequential ensemble, focus on hard examples
- **Method**: Weight misclassified samples higher in next iteration
- **Pros**: Simple, effective, works well with weak learners
- **Parameters**: n_estimators (100), learning_rate (1.0)
- **Best For**: Binary classification, combining weak learners

#### 5. Gradient Boosting ⭐⭐
- **Use**: High-accuracy predictions
- **Method**: Build trees sequentially, each correcting previous errors
- **Pros**: State-of-the-art performance, handles mixed data types
- **Tuning**: n_estimators, learning_rate, max_depth
- **Best For**: Kaggle competitions, production systems

#### 6. XGBoost ⭐⭐⭐ (Optional)
- **Use**: Maximum performance, scalability
- **Method**: Optimized gradient boosting with regularization
- **Pros**: Fastest GBDT, GPU support, handles missing values, regularization
- **Parameters**: n_estimators (100), learning_rate (0.1), max_depth (5)
- **Note**: Requires `pip install xgboost`
- **Best For**: Large datasets, competition-grade performance

#### 7. LightGBM ⭐⭐⭐ (Optional)
- **Use**: Very large datasets, fast training
- **Method**: Leaf-wise tree growth (vs level-wise)
- **Pros**: 10-20x faster than traditional GBDT, low memory, accuracy
- **Parameters**: n_estimators (100), learning_rate (0.1), num_leaves (31)
- **Note**: Requires `pip install lightgbm`
- **Best For**: Datasets >10K samples, production at scale

### Meta-Learning Methods

#### 8. Voting Ensemble ⭐
- **Use**: Combine diverse models
- **Method**: Average predictions from multiple base models
- **Types**:
  - Hard voting: Majority vote for classification
  - Soft voting: Average probabilities for classification
  - Average: Mean for regression
- **Pros**: Simple, reduces variance, leverages diversity
- **Base Models**: Random Forest, Gradient Boosting, Logistic Regression
- **Best For**: Combining models with different strengths

#### 9. Stacking Ensemble ⭐⭐
- **Use**: Learn optimal model combination
- **Method**: Train meta-learner on base model predictions
- **Pros**: Learns best combination, often beats individual models
- **Base Models**: Random Forest, Gradient Boosting, Extra Trees
- **Meta-Learner**: Logistic Regression (classification), Ridge (regression)
- **Best For**: Squeezing extra performance from diverse models

## 📝 Example Code

### Python API

```python
from ensemble_models import EnsembleAnalyzer
import pandas as pd
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv('data.csv')
X = df.drop(columns=['target'])
y = df['target']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize analyzer
analyzer = EnsembleAnalyzer(task_type='classification', random_state=42)

# Train all ensemble methods
analyzer.train_all_models(X_train, y_train, X_test, y_test, tune_params=False)

# Compare results
comparison = analyzer.compare_models()
print(comparison)

# Best model
print(f"Best: {analyzer.best_model_name}")
print(f"Accuracy: {analyzer.results[analyzer.best_model_name]['accuracy']:.4f}")

# Visualize
analyzer.plot_model_comparison(save_path='comparison.png')
analyzer.plot_feature_importance(top_k=20, save_path='importance.png')

# Save best model
analyzer.save_model(analyzer.best_model_name, 'best_ensemble.pkl')
```

### Train Individual Models

```python
# Random Forest
rf_metrics = analyzer.random_forest_ensemble(X_train, y_train, X_test, y_test, tune_params=True)

# Gradient Boosting
gb_metrics = analyzer.gradient_boosting_ensemble(X_train, y_train, X_test, y_test, tune_params=True)

# XGBoost (if available)
xgb_metrics = analyzer.xgboost_ensemble(X_train, y_train, X_test, y_test)

# Voting Ensemble
voting_metrics = analyzer.voting_ensemble(X_train, y_train, X_test, y_test, voting='soft')

# Stacking Ensemble
stacking_metrics = analyzer.stacking_ensemble(X_train, y_train, X_test, y_test)
```

### Regression Example

```python
# Initialize for regression
analyzer = EnsembleAnalyzer(task_type='regression', random_state=42)

# Train all models
analyzer.train_all_models(X_train, y_train, X_test, y_test)

# Compare
comparison = analyzer.compare_models()

# Best model metrics
best = analyzer.best_model_name
print(f"Best Model: {best}")
print(f"R²: {analyzer.results[best]['r2']:.4f}")
print(f"RMSE: {analyzer.results[best]['rmse']:.4f}")
print(f"MAE: {analyzer.results[best]['mae']:.4f}")
```

## 📊 Evaluation Metrics

### Classification
- **Accuracy**: Overall correct predictions (higher is better)
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1 Score**: Harmonic mean of precision and recall

### Regression
- **R² Score**: Proportion of variance explained (1.0 is perfect)
- **RMSE**: Root Mean Squared Error (lower is better)
- **MAE**: Mean Absolute Error (lower is better, robust to outliers)

### Additional Metrics
- **OOB Score**: Out-of-bag validation (for bagging methods)
- **Feature Importance**: Relative importance of features (tree-based methods)

## 🎨 Use Cases

### Business & Marketing
- Customer churn prediction
- Lead scoring and conversion
- Product recommendation
- Market segmentation
- Campaign response modeling

### Finance
- Credit risk assessment
- Fraud detection
- Stock price prediction
- Loan default prediction
- Algorithmic trading

### Healthcare
- Disease diagnosis
- Patient readmission prediction
- Treatment outcome prediction
- Medical image classification
- Drug response prediction

### Operations
- Equipment failure prediction
- Quality control
- Demand forecasting
- Inventory optimization
- Process optimization

## 📈 Sample Output

```
================================================================================
🚀 Training All Ensemble Models
================================================================================

🔧 Bagging Ensemble (n_estimators=100)
  OOB Score: 0.9456
  Accuracy: 0.9350 | Precision: 0.9378 | Recall: 0.9322 | F1: 0.9350

🔧 Random Forest Ensemble
  OOB Score: 0.9589
  Accuracy: 0.9500 | Precision: 0.9523 | Recall: 0.9478 | F1: 0.9500

🔧 Extra Trees Ensemble (n_estimators=100)
  Accuracy: 0.9450 | Precision: 0.9467 | Recall: 0.9433 | F1: 0.9450

🔧 AdaBoost Ensemble (n_estimators=100, lr=1.0)
  Accuracy: 0.9250 | Precision: 0.9289 | Recall: 0.9211 | F1: 0.9250

🔧 Gradient Boosting Ensemble
  Accuracy: 0.9550 | Precision: 0.9567 | Recall: 0.9533 | F1: 0.9550

🔧 XGBoost Ensemble
  Accuracy: 0.9600 | Precision: 0.9611 | Recall: 0.9589 | F1: 0.9600

🔧 LightGBM Ensemble
  Accuracy: 0.9575 | Precision: 0.9589 | Recall: 0.9561 | F1: 0.9575

🔧 Voting Ensemble (voting=soft)
  Accuracy: 0.9525 | Precision: 0.9544 | Recall: 0.9506 | F1: 0.9525

🔧 Stacking Ensemble
  Accuracy: 0.9625 | Precision: 0.9633 | Recall: 0.9617 | F1: 0.9625

================================================================================
🏆 Best Model: Stacking (Accuracy: 0.9625)
================================================================================

📊 Model Comparison:
================================================================================
                     accuracy  precision    recall  f1_score
          Stacking    0.9625     0.9633    0.9617    0.9625
           XGBoost    0.9600     0.9611    0.9589    0.9600
          LightGBM    0.9575     0.9589    0.9561    0.9575
Gradient Boosting    0.9550     0.9567    0.9533    0.9550
            Voting    0.9525     0.9544    0.9506    0.9525
     Random Forest    0.9500     0.9523    0.9478    0.9500
       Extra Trees    0.9450     0.9467    0.9433    0.9450
           Bagging    0.9350     0.9378    0.9322    0.9350
          AdaBoost    0.9250     0.9289    0.9211    0.9250
================================================================================
```

## 🔧 Advanced Features

### Hyperparameter Tuning

```python
# Enable tuning for Random Forest and Gradient Boosting
analyzer.train_all_models(X_train, y_train, X_test, y_test, tune_params=True)
```

### Feature Importance Analysis

```python
# Get feature importance from Random Forest
rf_model = analyzer.models['Random Forest']
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print(feature_importance.head(10))
```

### Custom Ensemble

```python
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC

# Create custom voting ensemble
custom_ensemble = VotingClassifier(
    estimators=[
        ('rf', analyzer.models['Random Forest']),
        ('gb', analyzer.models['Gradient Boosting']),
        ('svm', SVC(probability=True))
    ],
    voting='soft'
)

custom_ensemble.fit(X_train, y_train)
```

## 🐛 Troubleshooting

**Low accuracy on test set**:
- Try stacking or voting to combine diverse models
- Enable hyperparameter tuning with --tune flag
- Check for data leakage or imbalanced classes
- Ensure sufficient training data

**Slow training**:
- Use LightGBM for faster gradient boosting
- Reduce n_estimators for boosting methods
- Use Extra Trees instead of Random Forest
- Enable parallel processing (already enabled with n_jobs=-1)

**Overfitting (high train, low test accuracy)**:
- Reduce max_depth in tree-based methods
- Increase min_samples_split
- Use bagging methods (Random Forest, Bagging)
- Reduce learning_rate in boosting methods

**XGBoost/LightGBM not available**:
```bash
pip install xgboost lightgbm
```

## 📚 Theory

### Why Ensembles Work

**Bias-Variance Tradeoff**:
- Bagging: Reduces variance (Random Forest, Bagging)
- Boosting: Reduces bias (AdaBoost, Gradient Boosting)
- Stacking: Reduces both by learning optimal combination

**Diversity is Key**:
- Different algorithms (Random Forest vs SVM)
- Different features (random subsets)
- Different samples (bootstrap sampling)

### Ensemble Mathematics

**Voting (Average)**:
```
ŷ = (1/M) * Σ(ŷᵢ)  where M = number of models
```

**Weighted Voting**:
```
ŷ = Σ(wᵢ * ŷᵢ)  where Σ(wᵢ) = 1
```

**Stacking**:
```
ŷ = meta_learner(ŷ₁, ŷ₂, ..., ŷₘ)
```

## 📄 License

MIT License - Free for commercial and research use

---

**Author**: BrillConsulting | AI Consultant & Data Scientist
**Contact**: clientbrill@gmail.com
