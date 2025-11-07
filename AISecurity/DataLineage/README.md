# Data Lineage & Access Control

Track data flow, transformations, and access control in ML pipelines with complete audit trails.

## Features

- **Data Lineage Tracking** - End-to-end data flow visualization
- **Access Control** - RBAC and ABAC for ML assets
- **Audit Logging** - Complete history of data access
- **Compliance Reporting** - GDPR, CCPA compliance
- **Version Control** - Dataset and model versioning
- **PII Detection** - Automatic sensitive data identification
- **Data Provenance** - Track data origins and transformations
- **Impact Analysis** - Downstream dependency tracking

## Usage

```python
from data_lineage import LineageTracker, AccessControl

# Initialize tracker
tracker = LineageTracker()

# Track dataset
dataset_id = tracker.register_dataset(
    name="customer_data",
    source="s3://data/customers.csv",
    schema={"name": "string", "email": "string", "age": "int"},
    contains_pii=True
)

# Track transformation
transformed_id = tracker.track_transformation(
    input_id=dataset_id,
    output_name="preprocessed_customers",
    transformation="remove_pii",
    metadata={"columns_dropped": ["email", "name"]}
)

# Visualize lineage
tracker.visualize_lineage(transformed_id)

# Access control
acl = AccessControl()
acl.grant_access(
    user="data_scientist_1",
    resource=dataset_id,
    permissions=["read"]
)
```

## Technologies

- Apache Atlas
- MLflow
- DVC (Data Version Control)
- NetworkX (graph visualization)
