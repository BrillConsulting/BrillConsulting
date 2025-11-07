# Model Registry & Versioning

MLflow-based model registry with versioning, A/B testing, and deployment tracking.

## Features

- **Model Versioning** - Track model versions and lineage
- **MLflow Integration** - Industry-standard registry
- **A/B Testing** - Traffic splitting and experiments
- **Metadata Tracking** - Metrics, parameters, artifacts
- **Model Promotion** - Staging → Production workflow
- **Rollback Support** - Quick model rollbacks
- **Performance Comparison** - Compare model versions
- **Audit Trail** - Complete deployment history

## Usage

```python
from model_registry import ModelRegistry

# Initialize registry
registry = ModelRegistry(tracking_uri="mlflow-server:5000")

# Register model
model_version = registry.register_model(
    name="llama2-7b-chat",
    model_path="s3://models/llama2-7b",
    metrics={"perplexity": 5.47, "accuracy": 0.856},
    tags={"framework": "pytorch", "task": "chat"}
)

# Promote to production
registry.promote_to_production(
    model_name="llama2-7b-chat",
    version=model_version
)

# A/B test
registry.create_ab_test(
    model_a="llama2-7b-v1",
    model_b="llama2-7b-v2",
    traffic_split=0.1  # 10% to B
)
```

## Technologies

- MLflow 2.8+
- Model registry backends
- A/B testing frameworks
