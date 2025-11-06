# Feature Store

Production-ready centralized feature management system for machine learning. Store, serve, and manage features for both training (offline) and inference (online) with point-in-time correctness.

## Features

### Core Capabilities
- **Feature Registration**: Define and version feature schemas
- **Online Serving**: Low-latency feature retrieval for real-time inference
- **Offline Storage**: Batch feature storage for training
- **Point-in-Time Correctness**: Prevent data leakage with temporal joins
- **Feature Validation**: Schema and data quality checks
- **Feature Statistics**: Monitoring and profiling
- **Feature Views**: Reusable feature combinations

### Architecture
- **Dual Storage**: Separate online (fast) and offline (bulk) stores
- **Feature Groups**: Organize features by entity
- **Metadata Management**: Track feature definitions and lineage
- **Transformation Pipeline**: Apply transformations consistently

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from featurestore import FeatureStore, FeatureGroup, Feature
import pandas as pd

# Initialize store
store = FeatureStore("./feature_store")

# Define feature group
user_features = FeatureGroup(
    name="user_features",
    entity="user_id",
    features=[
        Feature("age", "int64", "User age"),
        Feature("total_purchases", "int64", "Total purchases"),
        Feature("avg_order_value", "float64", "Average order value")
    ]
)

# Register features
store.register_feature_group(user_features)

# Materialize features
data = pd.DataFrame({
    "user_id": ["user_1", "user_2", "user_3"],
    "age": [25, 35, 45],
    "total_purchases": [10, 25, 50],
    "avg_order_value": [49.99, 75.50, 120.00]
})

store.materialize_features("user_features", data, to_online=True, to_offline=True)

# Online serving (inference)
features = store.get_online_features(
    entity_ids=["user_1", "user_2"],
    feature_names=["user_features.age", "user_features.total_purchases"]
)

# Offline retrieval (training)
entity_df = pd.DataFrame({"user_id": ["user_1", "user_2"]})
historical = store.get_historical_features(entity_df, feature_names)
```

## Best Practices

1. **Feature Organization**: Group related features together
2. **Naming Convention**: Use `{group}.{feature}` format
3. **Versioning**: Track feature versions and changes
4. **Validation**: Always validate data against schemas
5. **Monitoring**: Track feature statistics regularly

## Requirements

- Python 3.7+
- pandas
- numpy

## Author

**Brill Consulting**
- Email: clientbrill@gmail.com
- LinkedIn: [brillconsulting](https://www.linkedin.com/in/brillconsulting)
