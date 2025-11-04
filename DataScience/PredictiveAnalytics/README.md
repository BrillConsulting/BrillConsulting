# Predictive Analytics Toolkit

End-to-end machine learning pipeline for predictive modeling and model deployment.

## Features

- **Data Preparation**: Train-test splitting with stratification
- **Multiple Models**: Train and compare several algorithms
- **Evaluation Metrics**: Comprehensive performance metrics
- **Cross-Validation**: K-fold validation for robust estimates
- **Hyperparameter Tuning**: GridSearchCV optimization
- **Feature Importance**: Tree-based feature ranking
- **Model Persistence**: Save and load trained models
- **Prediction Generation**: Deploy models for inference

## Technologies

- Scikit-learn: ML algorithms and evaluation
- Joblib: Model persistence
- Matplotlib, Seaborn: Visualization

## Usage

```python
from predictive_analytics import PredictiveAnalytics

# Initialize
pa = PredictiveAnalytics(task='classification')

# Prepare and train
X_train, X_test, y_train, y_test = pa.prepare_data(X, y)
pa.train_multiple_models(X_train, y_train)

# Evaluate
results = pa.evaluate_models(X_test, y_test)

# Tune and predict
tuning = pa.tune_hyperparameters(X_train, y_train)
predictions = pa.generate_predictions(X_new)
```

## Demo

```bash
python predictive_analytics.py
```
