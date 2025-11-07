# 🔍 Model Interpretability v2.0

Production-ready model interpretability with SHAP, LIME, feature importance, and permutation importance for explaining ML model predictions.

## 🌟 Methods

### Global Explanations
1. **Feature Importance** - Built-in model importance (tree-based models)
2. **Permutation Importance** - Model-agnostic importance by shuffling features
3. **SHAP Summary Plots** - Global feature impact (SHapley Additive exPlanations)
4. **Partial Dependence Plots** - Feature effect on predictions

### Local Explanations
5. **SHAP Waterfall** - Single prediction explanation
6. **SHAP Force Plot** - Interactive single instance explanation
7. **LIME** - Local Interpretable Model-agnostic Explanations

## ✨ Key Features

- **SHAP Integration** - Industry-standard explanations (TreeExplainer, LinearExplainer, KernelExplainer)
- **LIME Integration** - Local instance explanations
- **Model-Agnostic** - Works with any sklearn model
- **Multiple Visualizations** - Summary plots, waterfall plots, importance charts
- **Global & Local** - Understand both overall patterns and individual predictions
- **Production-Ready** - Comprehensive error handling and logging

## 🚀 Quick Start

### Basic Usage

```bash
python modelinterpretability.py --model trained_model.pkl --data-train train.csv --data-test test.csv --target label --task classification
```

### With Top K Features

```bash
python modelinterpretability.py --model model.pkl --data-train train.csv --data-test test.csv --target price --task regression --top-k 15
```

## 📊 Example Code

```python
from modelinterpretability import ModelExplainer
import pandas as pd
import joblib

# Load model and data
model = joblib.load('trained_model.pkl')
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

X_train = df_train.drop(columns=['target']).values
X_test = df_test.drop(columns=['target']).values
y_test = df_test['target'].values
feature_names = df_train.drop(columns=['target']).columns.tolist()

# Initialize explainer
explainer = ModelExplainer(
    model=model,
    X_train=X_train,
    feature_names=feature_names,
    task_type='classification'
)

# 1. Feature Importance (tree-based models)
explainer.get_feature_importance(top_k=20)
explainer.plot_feature_importance(save_path='feature_importance.png')

# 2. Permutation Importance (model-agnostic)
explainer.get_permutation_importance(X_test, y_test, n_repeats=10, top_k=20)
explainer.plot_permutation_importance(save_path='perm_importance.png')

# 3. SHAP Explanations
explainer.explain_with_shap(X_test[:100], max_display=20)
explainer.plot_shap_summary(save_path='shap_summary.png')
explainer.plot_shap_bar(save_path='shap_bar.png')
explainer.plot_shap_waterfall(instance_idx=0, save_path='shap_waterfall.png')

# 4. LIME Explanation (single instance)
lime_result = explainer.explain_instance_with_lime(X_test[0], instance_idx=0, num_features=10)
explainer.plot_lime_explanation(lime_result, save_path='lime.png')

# 5. Partial Dependence Plots
top_features = explainer.permutation_importance_results['indices'][:3].tolist()
explainer.plot_partial_dependence(top_features, X_test[:500], save_path='pdp.png')

# 6. Compare Methods
explainer.compare_importance_methods(save_path='comparison.png')
```

## 🎯 Use Cases

### 🏦 Finance & Banking
- **Credit Scoring**: Explain why a loan was approved/rejected
- **Fraud Detection**: Identify which features triggered fraud alert
- **Risk Assessment**: Understand key risk factors

### 🏥 Healthcare
- **Disease Prediction**: Explain diagnosis predictions
- **Treatment Recommendations**: Justify treatment choices
- **Patient Risk**: Identify high-risk patient characteristics

### 🏢 Business & Marketing
- **Customer Churn**: Understand why customers leave
- **Pricing Models**: Explain price predictions
- **Demand Forecasting**: Identify demand drivers

### ⚖️ Regulatory Compliance
- **GDPR Right to Explanation**: Provide explanations for automated decisions
- **Fair Lending**: Ensure no discriminatory factors
- **Model Validation**: Demonstrate model behavior to regulators

## 📚 Method Details

### SHAP (SHapley Additive exPlanations)

**What it is:**
- Game-theory based approach to explain predictions
- Assigns each feature an importance value (SHAP value)
- Additive: sum of SHAP values = prediction - baseline

**When to use:**
- Need theoretically sound explanations
- Want global AND local explanations
- Have tree-based or linear models (fast)

**Explainers:**
- **TreeExplainer**: Fast for tree-based models (RF, XGBoost, etc.)
- **LinearExplainer**: For linear models
- **KernelExplainer**: Model-agnostic (slow, for any model)

**Visualizations:**
- **Summary Plot**: Shows feature importance and impact direction
- **Bar Plot**: Mean absolute SHAP values
- **Waterfall Plot**: Single prediction breakdown
- **Force Plot**: Interactive single instance explanation

```python
# SHAP example
explainer.explain_with_shap(X_test[:100])
explainer.plot_shap_summary()  # Global importance
explainer.plot_shap_waterfall(instance_idx=0)  # Single prediction
```

### LIME (Local Interpretable Model-agnostic Explanations)

**What it is:**
- Explains individual predictions by fitting local linear model
- Creates perturbations around instance
- Trains interpretable model on neighborhood

**When to use:**
- Need local (per-instance) explanations
- Model-agnostic approach (works with any model)
- Want human-interpretable rules

**Best for:**
- Explaining specific predictions
- Debugging misclassifications
- Building trust with stakeholders

```python
# LIME example
lime_result = explainer.explain_instance_with_lime(
    instance=X_test[0],
    instance_idx=0,
    num_features=10
)
explainer.plot_lime_explanation(lime_result)
```

### Permutation Importance

**What it is:**
- Measures feature importance by randomly shuffling each feature
- Observes impact on model performance
- Model-agnostic, works for any model

**When to use:**
- Need model-agnostic importance
- Want to understand which features matter most
- Tree-based feature importance not available

**Advantages:**
- Works with any model
- Based on actual model performance
- Includes feature interactions

```python
# Permutation importance
explainer.get_permutation_importance(
    X_test, y_test,
    n_repeats=10,
    top_k=20
)
explainer.plot_permutation_importance()
```

### Partial Dependence Plots (PDP)

**What it is:**
- Shows marginal effect of a feature on predictions
- Averages predictions across feature values
- Reveals relationship between feature and target

**When to use:**
- Understand feature-target relationships
- Detect non-linear effects
- Identify interaction effects

```python
# Partial dependence
top_features = [0, 1, 2]  # Feature indices
explainer.plot_partial_dependence(top_features, X_test)
```

## 🔧 Advanced Configuration

### SHAP Explainer Selection

SHAP automatically selects the best explainer based on model type:

```python
# Tree-based models → TreeExplainer (fast)
model = RandomForestClassifier()
explainer = ModelExplainer(model, X_train)

# Linear models → LinearExplainer (fast)
model = LogisticRegression()
explainer = ModelExplainer(model, X_train)

# Any model → KernelExplainer (slow)
model = SVC()
explainer = ModelExplainer(model, X_train)
```

### Custom SHAP Parameters

```python
# Explain with custom max_display
explainer.explain_with_shap(X_test[:50], max_display=15)

# Plot waterfall for specific instance
explainer.plot_shap_waterfall(instance_idx=5)
```

### LIME Parameters

```python
# LIME with more features
lime_result = explainer.explain_instance_with_lime(
    instance=X_test[0],
    instance_idx=0,
    num_features=15  # Show top 15 features
)
```

### Permutation Importance Tuning

```python
# More repeats for stable estimates
explainer.get_permutation_importance(
    X_test, y_test,
    n_repeats=30,  # More repeats = more stable
    top_k=20
)
```

## 📊 Visualization Guide

### 1. SHAP Summary Plot
**Shows**: Global feature importance with impact direction
**Best for**: Understanding overall feature contributions
```python
explainer.plot_shap_summary(save_path='shap_summary.png')
```

### 2. SHAP Bar Plot
**Shows**: Mean absolute SHAP values
**Best for**: Quick feature importance ranking
```python
explainer.plot_shap_bar(save_path='shap_bar.png')
```

### 3. SHAP Waterfall Plot
**Shows**: Single prediction breakdown
**Best for**: Explaining individual predictions
```python
explainer.plot_shap_waterfall(instance_idx=0, save_path='waterfall.png')
```

### 4. Feature Importance
**Shows**: Built-in model importance (tree-based)
**Best for**: Quick importance check
```python
explainer.plot_feature_importance(save_path='importance.png')
```

### 5. Permutation Importance
**Shows**: Model-agnostic importance with error bars
**Best for**: Any model type
```python
explainer.plot_permutation_importance(save_path='perm_imp.png')
```

### 6. Partial Dependence
**Shows**: Feature effect on predictions
**Best for**: Understanding relationships
```python
explainer.plot_partial_dependence([0, 1, 2], X_test, save_path='pdp.png')
```

### 7. Method Comparison
**Shows**: Side-by-side comparison of importance methods
**Best for**: Validating feature importance
```python
explainer.compare_importance_methods(save_path='comparison.png')
```

## 💡 Best Practices

### 1. **Use Multiple Methods**
- Combine SHAP, LIME, and permutation importance
- Cross-validate findings across methods
- More robust conclusions

### 2. **Start with Global, Then Local**
- First understand overall model behavior (SHAP summary)
- Then dive into specific predictions (SHAP waterfall, LIME)

### 3. **Validate with Domain Knowledge**
- Check if important features make sense
- Discuss with domain experts
- Question surprising findings

### 4. **Consider Computation Time**
- TreeExplainer (SHAP) is very fast
- KernelExplainer can be slow on large datasets
- Sample data if needed

### 5. **Interpret Carefully**
- SHAP values show contribution, not causation
- Correlation ≠ causation
- Be aware of feature interactions

## 🐛 Troubleshooting

**SHAP taking too long?**
- Reduce number of samples: `explainer.explain_with_shap(X_test[:100])`
- Use TreeExplainer if possible (much faster)
- Sample background data for KernelExplainer

**LIME not working?**
- Ensure model has `predict` or `predict_proba` method
- Check task_type ('classification' vs 'regression')
- Try increasing num_samples in LIME configuration

**Feature importance not available?**
- Only tree-based models have `feature_importances_`
- Use permutation importance instead (model-agnostic)

**Negative SHAP values confusing?**
- Negative = decreases prediction
- Positive = increases prediction
- Relative to baseline (mean prediction)

**Visualizations look strange?**
- Check feature scaling (SHAP works on scaled data)
- Ensure feature names are correct
- Try different max_display values

## 📊 Method Comparison

| Method | Type | Speed | Model Type | Interpretability | Use Case |
|--------|------|-------|------------|------------------|----------|
| **SHAP** | Global + Local | Fast (Tree) / Slow (Kernel) | Any | High | Best overall |
| **LIME** | Local | Medium | Any | High | Single predictions |
| **Feature Importance** | Global | Instant | Tree-based | Medium | Quick check |
| **Permutation Importance** | Global | Slow | Any | High | Any model |
| **Partial Dependence** | Global | Medium | Any | Medium | Feature effects |

## 📄 Dependencies

```bash
pip install numpy pandas scikit-learn matplotlib seaborn joblib

# For SHAP (recommended)
pip install shap

# For LIME (recommended)
pip install lime
```

## 🏆 Status

**Version:** 2.0
**Lines of Code:** 570
**Status:** Production-Ready ✅

**Features:**
- ✅ SHAP Integration (3 explainers)
- ✅ LIME Integration
- ✅ Permutation Importance
- ✅ Feature Importance
- ✅ Partial Dependence Plots
- ✅ 7 Visualization Types
- ✅ Model-Agnostic Support
- ✅ Production-Ready Code

## 📞 Support

**Author**: BrillConsulting | AI Consultant & Data Scientist
**Email**: clientbrill@gmail.com
**LinkedIn**: [BrillConsulting](https://www.linkedin.com/in/brillconsulting)

---

**⭐ Star this repository if you find it useful!**

*Made with ❤️ by BrillConsulting*
