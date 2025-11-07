# Explainable AI Dashboards

Interactive dashboards for model interpretability using SHAP, Captum, and Streamlit.

## Features

- **SHAP Analysis** - SHapley Additive exPlanations
- **Feature Importance** - Global and local feature attribution
- **Counterfactual Explanations** - What-if scenarios
- **Captum Integration** - PyTorch model interpretability
- **Interactive Dashboards** - Streamlit-based UI
- **Model-Agnostic** - Works with any ML model
- **Visualization Suite** - Force plots, waterfall charts, decision plots
- **Export Reports** - Generate PDF/HTML explanation reports

## Explanation Types

| Type | Scope | Use Case |
|------|-------|----------|
| **SHAP Values** | Local/Global | Feature importance |
| **Attention Weights** | Local | Transformer interpretability |
| **Counterfactuals** | Local | What-if analysis |
| **Feature Attribution** | Local | Input importance |

## Usage

### SHAP Explanations

```python
from explainable_ai import SHAPExplainer

# Initialize explainer
explainer = SHAPExplainer(model=trained_model)

# Explain prediction
explanation = explainer.explain(
    instance=X_test[0],
    feature_names=feature_names
)

# Visualize
explainer.plot_force_plot(explanation)
explainer.plot_waterfall(explanation)

# Generate report
explainer.generate_report(
    explanations=[explanation],
    output_path="report.html"
)
```

### Interactive Dashboard

```bash
# Launch dashboard
streamlit run dashboard.py

# Navigate to http://localhost:8501
```

### Captum for Deep Learning

```python
from explainable_ai import CaptumExplainer

# For PyTorch models
explainer = CaptumExplainer(
    model=neural_network,
    method="integrated_gradients"
)

# Get attributions
attributions = explainer.attribute(input_tensor)

# Visualize
explainer.visualize_attributions(attributions, input_tensor)
```

## Technologies

- SHAP 0.43+
- Captum 0.7+
- Streamlit 1.28+
- Plotly
- scikit-learn
- PyTorch
