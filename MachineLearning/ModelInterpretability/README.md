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

```bash
python modelinterpretability.py --model trained_model.pkl --data-train train.csv --data-test test.csv --target label --task classification
```

## 🏆 Status

**Version:** 2.0
**Lines of Code:** 570
**Status:** Production-Ready ✅

---

**Author**: BrillConsulting | AI Consultant & Data Scientist
