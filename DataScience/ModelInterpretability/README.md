# Model Interpretability Toolkit

A comprehensive toolkit for explaining and interpreting machine learning models using state-of-the-art interpretability methods.

## Description

The Model Interpretability Toolkit provides unified interfaces for understanding and explaining machine learning model predictions. It implements multiple interpretability techniques including SHAP values, LIME, feature importance analysis, and partial dependence plots to make black-box models transparent and trustworthy.

## Key Features

- **SHAP (SHapley Additive exPlanations)**
  - TreeExplainer for tree-based models
  - KernelExplainer for any model type
  - Global and local feature importance
  - Summary plots and force plots
  - Interaction effects analysis

- **LIME (Local Interpretable Model-agnostic Explanations)**
  - Local surrogate models
  - Tabular data explanations
  - Feature contribution analysis
  - Instance-level interpretability

- **Feature Importance**
  - Permutation importance
  - Built-in feature importance (tree models)
  - Drop-column importance
  - Importance ranking and visualization

- **Partial Dependence Analysis**
  - Partial Dependence Plots (PDP)
  - Individual Conditional Expectation (ICE) plots
  - 2D partial dependence for interactions
  - Accumulated Local Effects (ALE)

- **Visualization Tools**
  - Summary plots for global explanations
  - Force plots for individual predictions
  - Dependence plots showing feature relationships
  - Interactive HTML reports

## Technologies Used

- **Python 3.x**
- **NumPy** - Numerical computing
- **Pandas** - Data manipulation
- **scikit-learn** - Machine learning models and metrics
- **SHAP** - SHapley Additive exPlanations
- **LIME** - Local Interpretable Model-agnostic Explanations
- **Matplotlib/Seaborn** - Visualization
- **PDPbox** - Partial dependence plots (optional)

## Installation

```bash
# Clone the repository
cd /home/user/BrillConsulting/DataScience/ModelInterpretability

# Install required packages
pip install numpy pandas scikit-learn shap lime matplotlib seaborn
```

## Usage Examples

### SHAP Values for Tree-Based Models

```python
from model_interpretability import ModelInterpreter
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load and prepare data
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

# Train a model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Initialize interpreter
interpreter = ModelInterpreter(model, X_train, feature_names=iris.feature_names)

# Calculate SHAP values
shap_result = interpreter.shap_values(
    X_test,
    method='tree'
)

print(f"SHAP values shape: {shap_result['shap_values'].shape}")
print(f"Base value: {shap_result['base_value']}")
print(f"Expected value: {shap_result['expected_value']:.3f}")

# Visualize SHAP summary
fig = interpreter.plot_shap_summary(shap_result)
fig.savefig('shap_summary.png', dpi=300, bbox_inches='tight')
```

### SHAP Force Plot for Individual Predictions

```python
# Explain a single prediction
instance_idx = 0
force_plot = interpreter.shap_force_plot(
    shap_result,
    instance_idx=instance_idx
)

print(f"Prediction for instance {instance_idx}:")
print(f"  Actual class: {y_test[instance_idx]}")
print(f"  Predicted class: {model.predict(X_test[instance_idx:instance_idx+1])[0]}")
print(f"  Top contributing features:")
for feat, val in force_plot['top_features'].items():
    print(f"    {feat}: {val:.3f}")

# Save force plot
force_plot['plot'].savefig('shap_force_plot.html')
```

### LIME Explanations

```python
# Get LIME explanation for an instance
lime_result = interpreter.lime_explanation(
    X_test[0],
    num_features=5,
    num_samples=5000
)

print(f"LIME explanation for instance 0:")
print(f"Prediction probability: {lime_result['prediction_proba']}")
print(f"Top features:")
for feature, weight in lime_result['feature_weights'][:5]:
    print(f"  {feature}: {weight:.3f}")

# Visualize LIME explanation
fig = lime_result['plot']
fig.savefig('lime_explanation.png', dpi=300, bbox_inches='tight')
```

### Permutation Feature Importance

```python
# Calculate permutation importance
perm_importance = interpreter.permutation_importance(
    X_test,
    y_test,
    n_repeats=10,
    random_state=42
)

print(f"Feature importance ranking:")
for i, (feat, importance, std) in enumerate(zip(
    perm_importance['features'],
    perm_importance['importances_mean'],
    perm_importance['importances_std']
), 1):
    print(f"{i}. {feat}: {importance:.4f} (+/- {std:.4f})")

# Visualize importance
fig = interpreter.plot_feature_importance(perm_importance)
fig.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
```

### Partial Dependence Plots

```python
# Create partial dependence plot for a feature
pdp_result = interpreter.partial_dependence(
    feature_idx=0,
    feature_name=iris.feature_names[0],
    grid_resolution=50
)

print(f"Partial dependence computed for: {pdp_result['feature_name']}")
print(f"Feature range: [{pdp_result['feature_values'].min():.2f}, "
      f"{pdp_result['feature_values'].max():.2f}]")

# Visualize PDP
fig = pdp_result['plot']
fig.savefig('partial_dependence.png', dpi=300, bbox_inches='tight')
```

### Individual Conditional Expectation (ICE) Plots

```python
# Create ICE plot showing individual predictions
ice_result = interpreter.ice_plot(
    feature_idx=0,
    feature_name=iris.feature_names[0],
    sample_indices=range(20)  # Plot first 20 instances
)

print(f"ICE plot for {ice_result['feature_name']}")
print(f"Number of instances plotted: {len(ice_result['sample_indices'])}")

# Visualize ICE with PDP overlay
fig = ice_result['plot']
fig.savefig('ice_plot.png', dpi=300, bbox_inches='tight')
```

### 2D Partial Dependence for Feature Interactions

```python
# Analyze interaction between two features
interaction_result = interpreter.partial_dependence_2d(
    feature_idx1=0,
    feature_idx2=1,
    feature_name1=iris.feature_names[0],
    feature_name2=iris.feature_names[1],
    grid_resolution=30
)

print(f"Interaction between {interaction_result['feature_name1']} "
      f"and {interaction_result['feature_name2']}")

# Visualize interaction as heatmap
fig = interaction_result['plot']
fig.savefig('feature_interaction.png', dpi=300, bbox_inches='tight')
```

## Demo Instructions

Run the comprehensive demo to see all features in action:

```bash
python model_interpretability.py
```

The demo will:
1. Load a dataset and train a machine learning model
2. Calculate SHAP values (global and local explanations)
3. Generate LIME explanations for sample instances
4. Compute permutation feature importance
5. Create partial dependence plots
6. Generate ICE plots
7. Analyze feature interactions
8. Save all visualizations as PNG/HTML files
9. Display interpretation summary

## Output Examples

**Console Output:**
```
Model Interpretability Toolkit Demo
======================================================================

Training Random Forest Classifier...
Model accuracy: 96.67%

1. SHAP Analysis
----------------------------------------------------------------------
Computing SHAP values for 30 test instances...
Expected value: 0.333
Base value: [0.33, 0.33, 0.34]

Top 3 Most Important Features (Global):
1. petal width (cm): 0.452
2. petal length (cm): 0.389
3. sepal length (cm): 0.098

2. LIME Explanations
----------------------------------------------------------------------
Explaining instance 0:
Predicted class: 0 (probability: 0.98)
True class: 0

Top contributing features:
  petal width (cm) <= 0.8: +0.234
  petal length (cm) <= 2.5: +0.189
  sepal width (cm) > 3.0: +0.078

3. Permutation Feature Importance
----------------------------------------------------------------------
Feature importance ranking:
1. petal width (cm): 0.4267 (+/- 0.0234)
2. petal length (cm): 0.3891 (+/- 0.0198)
3. sepal length (cm): 0.0956 (+/- 0.0145)
4. sepal width (cm): 0.0421 (+/- 0.0089)

4. Partial Dependence Analysis
----------------------------------------------------------------------
Computing PDP for 'petal width (cm)'...
Feature range: [0.10, 2.50]
Prediction range: [0.05, 0.95]

5. ICE Plots
----------------------------------------------------------------------
Generated ICE plot for 'petal width (cm)'
Number of instances: 20
Mean prediction variance: 0.234

6. Feature Interactions
----------------------------------------------------------------------
Analyzing interaction: petal width (cm) x petal length (cm)
Interaction strength: 0.156
```

**Generated Visualizations:**
- `shap_summary.png` - Global feature importance from SHAP
- `shap_force_plot.html` - Interactive force plot for individual prediction
- `shap_dependence.png` - SHAP dependence plots showing interactions
- `lime_explanation.png` - LIME explanation for sample instance
- `feature_importance.png` - Permutation importance with error bars
- `partial_dependence.png` - PDP showing marginal effect
- `ice_plot.png` - ICE plots with PDP overlay
- `feature_interaction.png` - 2D interaction heatmap

## Author

**Brill Consulting**

---

For more information about the algorithms and methodologies, see the inline documentation in `model_interpretability.py`.
