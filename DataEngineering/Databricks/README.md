# ⚡ Databricks Workspace Management

**Unified analytics platform for data and ML**

## Overview
Complete Databricks SDK implementation for cluster management, job orchestration, Delta Lake operations, and MLflow integration.

## Key Features

### Cluster Management
- Auto-scaling clusters
- Spot instance optimization
- Multiple node types
- Custom configurations

### Data Processing
- Delta Lake tables
- Table optimization (Z-ordering, compaction)
- Vacuum old versions
- Spark SQL execution

### ML Operations
- MLflow experiment tracking
- Model registry
- Run management
- Auto Loader streaming

### Job Orchestration
- Multi-task workflows
- Job scheduling
- Dependency management
- Notebook execution

## Quick Start

```python
from databricks_workspace import DatabricksWorkspace

workspace = DatabricksWorkspace(
    workspace_url='https://your-workspace.cloud.databricks.com',
    token='your-token'
)

# Create cluster
cluster = workspace.create_cluster({
    'cluster_name': 'analytics-cluster',
    'spark_version': '12.2.x-scala2.12',
    'node_type_id': 'i3.xlarge',
    'num_workers': 2,
    'autoscale': {'min_workers': 2, 'max_workers': 8}
})

# Create job
job = workspace.create_job({
    'name': 'ETL Pipeline',
    'tasks': [{'task_key': 'extract'}, {'task_key': 'transform'}]
})

# Run SQL
result = workspace.run_spark_sql(
    'SELECT * FROM delta.`/mnt/data/table`',
    cluster['cluster_id']
)
```

## Use Cases
- **Data Lakehouse** - Unified analytics platform
- **ML Workflows** - End-to-end ML pipelines
- **Collaborative Analytics** - Team notebooks
- **ETL at Scale** - Process massive datasets

## Technologies
- Databricks Runtime
- Delta Lake
- MLflow
- Apache Spark

## Installation
```bash
pip install -r requirements.txt
python databricks_workspace.py
```

---

**Author:** Brill Consulting | clientbrill@gmail.com
