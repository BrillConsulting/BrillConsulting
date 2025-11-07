# Bias & Fairness Analysis Toolkit

Comprehensive bias detection and fairness analysis for ML models using AIF360 and What-If Tool.

## Features

- **Bias Detection** - Identify unfair bias in datasets and models
- **Fairness Metrics** - 20+ fairness metrics (demographic parity, equal opportunity, etc.)
- **Bias Mitigation** - Pre-processing, in-processing, post-processing techniques
- **Disparate Impact Analysis** - Measure impact across protected groups
- **Counterfactual Analysis** - What-If Tool integration
- **Fairness Reports** - Automated fairness audit reports
- **Interactive Dashboards** - Visualize bias and fairness
- **Remediation Recommendations** - Actionable bias reduction strategies

## Fairness Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **Demographic Parity** | Equal positive rate across groups | ~1.0 |
| **Equal Opportunity** | Equal TPR across groups | ~1.0 |
| **Equalized Odds** | Equal TPR and FPR across groups | ~1.0 |
| **Disparate Impact** | Ratio of positive rates | 0.8-1.25 |

## Usage

```python
from bias_analysis import FairnessAnalyzer

# Initialize analyzer
analyzer = FairnessAnalyzer()

# Load data with protected attributes
analyzer.load_data(
    X=features,
    y=labels,
    protected_attributes=["gender", "race", "age"]
)

# Analyze bias in dataset
dataset_metrics = analyzer.analyze_dataset()

# Train model
model = analyzer.train_fair_model(
    algorithm="logistic_regression",
    fairness_constraint="demographic_parity"
)

# Evaluate fairness
fairness_report = analyzer.evaluate_fairness(
    model=model,
    X_test=X_test,
    y_test=y_test
)

# Generate recommendations
recommendations = analyzer.get_mitigation_recommendations()
```

## Technologies

- IBM AIF360
- Google What-If Tool
- Fairlearn
- scikit-learn
- Plotly/Streamlit
