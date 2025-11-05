# Databricks Workspace Management

Complete Databricks integration for data processing, ML, and analytics.

## Features

- **Cluster Management**: Create and configure Databricks clusters with autoscaling
- **Delta Lake**: Create and optimize Delta tables with Z-ordering
- **Spark SQL**: Execute distributed SQL queries
- **MLflow Integration**: Track experiments and model runs
- **Auto Loader**: Real-time streaming data ingestion
- **Job Orchestration**: Create and schedule multi-task workflows
- **Table Optimization**: OPTIMIZE and VACUUM Delta tables

## Technologies

- Databricks SDK
- PySpark
- Delta Lake
- MLflow

## Usage

```python
from databricks_workspace import DatabricksWorkspace

# Initialize workspace
workspace = DatabricksWorkspace(
    workspace_url='https://my-workspace.cloud.databricks.com',
    token='dapi_token_123'
)

# Create cluster
cluster = workspace.create_cluster({
    'cluster_name': 'production-cluster',
    'spark_version': '12.2.x-scala2.12',
    'autoscale': {'min_workers': 2, 'max_workers': 8}
})

# Create Delta table
table = workspace.create_delta_table({
    'table_name': 'sales_data',
    'database': 'production',
    'partition_by': ['date']
})
```

## Demo

```bash
python databricks_workspace.py
```
