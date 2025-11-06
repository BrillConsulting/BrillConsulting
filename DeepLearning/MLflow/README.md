# MLflow ML Lifecycle Management

## 🎯 Overview

Comprehensive MLflow implementation for end-to-end ML lifecycle management, featuring experiment tracking, model registry, hyperparameter optimization, autologging, and production deployment workflows.

## ✨ Features

### Experiment Tracking
- **Run Management**: Log parameters, metrics, and artifacts
- **Metric History**: Track metrics over training epochs
- **Hyperparameter Sweeps**: Automated grid search with logging
- **Artifact Storage**: Save models, plots, and datasets
- **Run Comparison**: Compare multiple experiments

### Model Registry
- **Version Control**: Automatic model versioning
- **Stage Management**: None → Staging → Production → Archived
- **Model Metadata**: Tags, descriptions, and annotations
- **Lineage Tracking**: Link models to training runs
- **Collaboration**: Team-based model management

### Autologging
- **Scikit-learn**: Automatic parameter and metric logging
- **TensorFlow/Keras**: Automatic epoch metrics and model saving
- **PyTorch**: Integration with training loops
- **XGBoost/LightGBM**: Native gradient boosting support

### Deployment
- **Model Loading**: Load models from registry by stage or version
- **Batch Inference**: Efficient batch predictions
- **REST API**: Model serving via MLflow Models
- **Docker Deployment**: Containerized model serving

## 📋 Requirements

```bash
pip install mlflow>=2.0.0
pip install scikit-learn tensorflow torch xgboost
```

## 🚀 Quick Start

```python
from mlflow_manager import MLflowManager

# Initialize manager
mgr = MLflowManager(tracking_uri='http://localhost:5000')

# Create experiment
exp = mgr.tracker.create_experiment({
    'name': 'fraud_detection_v2',
    'tags': {'team': 'ml-ops', 'project': 'fraud'}
})

# Log training run
run = mgr.tracker.log_training_run({
    'run_name': 'baseline_model',
    'experiment_name': 'fraud_detection_v2',
    'params': {
        'learning_rate': 0.001,
        'batch_size': 64,
        'epochs': 50
    }
})

# Hyperparameter sweep
sweep = mgr.tracker.log_hyperparameter_sweep({
    'param_grid': {
        'learning_rate': [0.001, 0.01, 0.1],
        'batch_size': [32, 64, 128],
        'optimizer': ['adam', 'sgd']
    }
})

# Register model
model = mgr.registry.register_model({
    'name': 'fraud_detector',
    'run_id': run['run_id']
})

# Transition to production
mgr.registry.transition_model_stage({
    'name': 'fraud_detector',
    'version': 1,
    'stage': 'Production'
})
```

## 🏗️ Workflow

### Training Pipeline
```
1. Create Experiment
2. Start Run
3. Log Parameters
4. Train Model
5. Log Metrics (per epoch)
6. Log Artifacts (plots, configs)
7. Log Model
8. End Run
```

### Model Registry Flow
```
1. Register Model → Version 1 (None stage)
2. Validate → Transition to Staging
3. A/B Test → Validate performance
4. Promote → Transition to Production
5. Monitor → Archive old versions
```

## 💡 Use Cases

- **Experiment Management**: Track thousands of training runs
- **Team Collaboration**: Share experiments and models across teams
- **Model Governance**: Audit trail for model changes
- **A/B Testing**: Compare model versions in production
- **Reproducibility**: Recreate any experiment from logged parameters

## 📊 Tracking Features

### Parameter Logging
```python
mlflow.log_param("learning_rate", 0.001)
mlflow.log_params({
    "batch_size": 32,
    "epochs": 100,
    "optimizer": "adam"
})
```

### Metric Logging
```python
# Log single metric
mlflow.log_metric("accuracy", 0.95)

# Log metric history
for epoch in range(100):
    mlflow.log_metric("train_loss", loss, step=epoch)
    mlflow.log_metric("val_accuracy", acc, step=epoch)
```

### Artifact Logging
```python
# Log files
mlflow.log_artifact("model_config.json")
mlflow.log_artifact("training_plot.png")

# Log directory
mlflow.log_artifacts("output_dir")

# Log model
mlflow.sklearn.log_model(model, "model")
```

## 🔬 Advanced Features

### Hyperparameter Sweep
```python
from itertools import product

param_grid = {
    'learning_rate': [0.001, 0.01, 0.1],
    'batch_size': [16, 32, 64],
    'optimizer': ['adam', 'sgd', 'rmsprop']
}

# Run all combinations
keys, values = zip(*param_grid.items())
experiments = [dict(zip(keys, v)) for v in product(*values)]

for params in experiments:
    with mlflow.start_run():
        mlflow.log_params(params)
        model = train_model(**params)
        score = evaluate_model(model)
        mlflow.log_metric("accuracy", score)
```

### Autologging Example
```python
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier

# Enable autologging
mlflow.sklearn.autolog()

with mlflow.start_run():
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    # Everything logged automatically!
```

### Model Registry Management
```python
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Register model
result = mlflow.register_model(
    model_uri=f"runs:/{run_id}/model",
    name="fraud_detector"
)

# Transition stages
client.transition_model_version_stage(
    name="fraud_detector",
    version=1,
    stage="Production",
    archive_existing_versions=True
)

# Load production model
loaded_model = mlflow.pyfunc.load_model(
    "models:/fraud_detector/Production"
)
```

## 📊 Model Registry Stages

| Stage | Description | Use Case |
|-------|-------------|----------|
| None | Initial registration | Just created |
| Staging | Testing phase | Validation/QA |
| Production | Live deployment | Serving predictions |
| Archived | Deprecated | Historical reference |

## 🎯 Best Practices

1. **Consistent Naming**: Use structured experiment and run names
2. **Tag Everything**: Add tags for filtering and organization
3. **Log Early, Log Often**: Track all important metrics
4. **Version Control**: Always register models with meaningful versions
5. **Stage Transitions**: Use proper staging workflow
6. **Artifact Organization**: Structure artifacts logically
7. **Run Comparison**: Compare runs to identify best models

## 📚 References

- MLflow Documentation: https://mlflow.org/docs/latest/index.html
- MLflow Tracking: https://mlflow.org/docs/latest/tracking.html
- Model Registry: https://mlflow.org/docs/latest/model-registry.html
- MLflow Models: https://mlflow.org/docs/latest/models.html
- Autologging: https://mlflow.org/docs/latest/tracking.html#automatic-logging

## 📧 Contact

For questions or collaboration: [clientbrill@gmail.com](mailto:clientbrill@gmail.com)

---

**Author:** Brill Consulting
**LinkedIn:** [brillconsulting](https://www.linkedin.com/in/brillconsulting)
