# Azure Synapse Analytics Integration

Advanced implementation of Azure Synapse Analytics with SQL pools, Apache Spark integration, data pipelines, and comprehensive data warehousing capabilities.

**Author:** BrillConsulting
**Contact:** clientbrill@gmail.com
**LinkedIn:** [brillconsulting](https://www.linkedin.com/in/brillconsulting)

## Overview

This project provides a comprehensive Python implementation for Azure Synapse Analytics, featuring serverless and dedicated SQL pools, Apache Spark pools, data integration pipelines, data lake connectivity, and advanced analytics capabilities. Built for enterprise-scale data warehousing and big data processing with Azure's security, performance, and scalability features.

## Features

### Core Capabilities
- **SQL Pool Management**: Serverless and dedicated SQL pools for flexible data warehousing
- **Apache Spark Integration**: Distributed big data processing with PySpark and Scala
- **Data Integration Pipelines**: ETL/ELT workflows with comprehensive activity support
- **Data Lake Connectivity**: Seamless integration with Azure Data Lake Storage Gen2
- **External Tables**: Query data directly from data lake without loading
- **Query Optimization**: Performance analysis and automatic optimization recommendations
- **Workspace Management**: Centralized control for all Synapse resources

### Advanced Features
- **Pool Scaling**: Dynamic scaling of dedicated SQL pools for cost optimization
- **Spark Job Submission**: Submit and monitor PySpark/Scala jobs
- **Notebook Support**: Interactive Spark notebooks for data exploration
- **Pipeline Orchestration**: Complex workflow management with triggers and scheduling
- **Performance Monitoring**: Real-time metrics and query statistics
- **Security & Access Control**: RBAC, firewall rules, and managed identities
- **Cost Management**: Pause/resume capabilities for dedicated pools
- **Query Plan Analysis**: Detailed execution plan insights and optimization

## Architecture

```
AzureSynapse/
├── azure_synapse.py           # Main implementation
├── requirements.txt           # Dependencies
└── README.md                  # Documentation
```

### Key Components

1. **AzureSynapseManager**: Main service interface
   - Workspace configuration
   - Resource provisioning
   - Unified management console

2. **SQL Pool Operations**:
   - Create serverless/dedicated pools
   - Execute queries and stored procedures
   - Create external tables for data lake
   - Pause/resume/scale operations

3. **Spark Pool Operations**:
   - Configure Spark pools with auto-scaling
   - Submit batch jobs
   - Execute Spark SQL queries
   - Manage notebooks

4. **Data Integration**:
   - Create and manage pipelines
   - Define datasets and linked services
   - Configure copy activities
   - Set up triggers and schedules

5. **Data Classes**:
   - `SQLPool`: SQL pool configuration
   - `SparkPool`: Spark pool settings
   - `Pipeline`: Pipeline definition
   - `Dataset`: Dataset schema and properties

## Installation

```bash
# Clone the repository
git clone https://github.com/BrillConsulting/BrillConsulting.git
cd BrillConsulting/Azure/AzureSynapse

# Install dependencies
pip install -r requirements.txt
```

## Configuration

Set up your Azure Synapse workspace credentials:

```python
from azure_synapse import AzureSynapseManager

manager = AzureSynapseManager(
    workspace_name="my-synapse-workspace",
    resource_group="my-resource-group",
    subscription_id="your-subscription-id"
)
```

### Environment Variables (Recommended)

```bash
export AZURE_SYNAPSE_WORKSPACE="my-synapse-workspace"
export AZURE_RESOURCE_GROUP="my-resource-group"
export AZURE_SUBSCRIPTION_ID="your-subscription-id"
```

## Usage Examples

### 1. Create and Manage SQL Pools

```python
from azure_synapse import AzureSynapseManager, PoolType

manager = AzureSynapseManager(
    workspace_name="my-synapse-workspace",
    resource_group="my-resource-group",
    subscription_id="subscription-id"
)

# Create a dedicated SQL pool
dedicated_pool = manager.create_sql_pool(
    pool_name="DedicatedPool01",
    pool_type=PoolType.DEDICATED,
    sku="DW100c",
    max_size_gb=240
)

print(f"Created pool: {dedicated_pool.name}")
print(f"SKU: {dedicated_pool.sku}")

# Create a serverless SQL pool for ad-hoc queries
serverless_pool = manager.create_sql_pool(
    pool_name="ServerlessPool01",
    pool_type=PoolType.SERVERLESS
)
```

### 2. Execute SQL Queries and External Tables

```python
# Execute query on dedicated pool
result = manager.execute_sql_query(
    pool_name="DedicatedPool01",
    query="SELECT * FROM sales_data WHERE year = 2024",
    parameters={"year": 2024}
)

print(f"Rows affected: {result['rows_affected']}")
print(f"Execution time: {result['execution_time_ms']}ms")

# Create external table for data lake
schema = [
    {"name": "customer_id", "type": "INT"},
    {"name": "customer_name", "type": "VARCHAR(100)"},
    {"name": "total_amount", "type": "DECIMAL(10,2)"},
    {"name": "order_date", "type": "DATE"}
]

external_table = manager.create_external_table(
    pool_name="ServerlessPool01",
    table_name="external_customers",
    data_source="my_data_lake",
    file_format="PARQUET",
    location="/data/customers/*.parquet",
    schema=schema
)

print(f"External table created: {external_table['table_name']}")
```

### 3. Pool Lifecycle Management

```python
# Pause dedicated pool to save costs
pause_result = manager.pause_sql_pool("DedicatedPool01")
print(f"Pool paused at: {pause_result['timestamp']}")

# Resume when needed
resume_result = manager.resume_sql_pool("DedicatedPool01")
print(f"Pool resumed at: {resume_result['timestamp']}")

# Scale pool for performance
scale_result = manager.scale_sql_pool("DedicatedPool01", "DW200c")
print(f"Scaled from {scale_result['old_sku']} to {scale_result['new_sku']}")
```

### 4. Apache Spark Operations

```python
from azure_synapse import SparkPoolSize

# Create Spark pool with auto-scaling
spark_pool = manager.create_spark_pool(
    pool_name="SparkPool01",
    node_size=SparkPoolSize.MEDIUM,
    node_count=3,
    spark_version="3.3",
    auto_scale=True,
    auto_pause=True
)

# Submit Spark job
job = manager.submit_spark_job(
    pool_name="SparkPool01",
    job_name="DataProcessingJob",
    main_file="scripts/process_data.py",
    arguments=["--input", "/data/raw", "--output", "/data/processed"],
    executor_count=4,
    executor_size="Medium"
)

print(f"Job ID: {job['job_id']}")
print(f"Status: {job['status']}")

# Execute Spark SQL
sql_result = manager.execute_spark_sql(
    pool_name="SparkPool01",
    sql_query="""
        SELECT category, COUNT(*) as count, SUM(amount) as total
        FROM sales
        GROUP BY category
        ORDER BY total DESC
    """
)

print(f"Rows returned: {sql_result['rows_returned']}")
```

### 5. Data Integration Pipelines

```python
# Create dataset definitions
source_schema = [
    {"name": "id", "type": "INT"},
    {"name": "data", "type": "VARCHAR(MAX)"}
]

source_dataset = manager.create_dataset(
    dataset_name="SourceDataset",
    dataset_type="AzureBlob",
    linked_service="AzureBlobLinkedService",
    schema=source_schema,
    folder_path="input"
)

# Create copy activity
copy_activity = manager.create_copy_activity(
    activity_name="CopyFromBlobToSynapse",
    source_dataset="SourceDataset",
    sink_dataset="SinkDataset",
    copy_behavior="PreserveHierarchy"
)

# Create pipeline
pipeline = manager.create_pipeline(
    pipeline_name="DataIngestionPipeline",
    description="Ingest data from blob storage to Synapse",
    activities=[copy_activity],
    parameters={"batchSize": 1000}
)

# Run pipeline
run = manager.run_pipeline("DataIngestionPipeline", {"date": "2024-01-15"})
print(f"Run ID: {run['run_id']}, Status: {run['status']}")
```

### 6. Monitoring and Optimization

```python
from datetime import datetime, timedelta

# Get query performance statistics
stats = manager.get_query_statistics(
    pool_name="DedicatedPool01",
    start_time=datetime.now() - timedelta(hours=24),
    end_time=datetime.now()
)

print(f"Total queries (24h): {stats['total_queries']}")
print(f"Average execution: {stats['avg_execution_time_ms']}ms")

# Analyze query execution plan
query = """
    SELECT c.customer_name, SUM(o.amount) as total
    FROM customers c JOIN orders o ON c.customer_id = o.customer_id
    GROUP BY c.customer_name
"""

plan = manager.analyze_query_plan(query)
print(f"Estimated cost: {plan['estimated_cost']}")
for rec in plan['recommendations']:
    print(f"  - {rec}")

# Optimize table
optimization = manager.optimize_table("sales_data")
print(f"Storage saved: {optimization['storage_saved_mb']}MB")
print(f"Performance improvement: {optimization['performance_improvement_pct']}%")
```

### 7. Pipeline Triggers and Scheduling

```python
# Create scheduled trigger
trigger = manager.create_trigger(
    trigger_name="DailyIngestion",
    trigger_type="ScheduleTrigger",
    pipeline_name="DataIngestionPipeline",
    schedule={
        "frequency": "Day",
        "interval": 1,
        "startTime": "2024-01-01T00:00:00Z"
    }
)

print(f"Trigger created: {trigger['name']}")
print(f"Schedule: {trigger['schedule']}")
```

## Running Demos

```bash
# Run all demo functions
python azure_synapse.py
```

Demo output includes:
- SQL pool creation and management
- Query execution and external tables
- Spark pool operations and job submission
- Data pipeline creation and execution
- Monitoring and optimization examples
- Workspace management

## API Reference

### AzureSynapseManager

#### SQL Pool Methods

**`create_sql_pool(pool_name, pool_type, sku, max_size_gb)`**
- Creates a SQL pool (serverless or dedicated)
- **Parameters**: pool_name (str), pool_type (PoolType), sku (str), max_size_gb (int)
- **Returns**: `SQLPool`

**`execute_sql_query(pool_name, query, parameters)`**
- Executes SQL query on a pool
- **Parameters**: pool_name (str), query (str), parameters (Dict)
- **Returns**: `Dict[str, Any]`

**`create_external_table(pool_name, table_name, data_source, file_format, location, schema)`**
- Creates external table for data lake integration
- **Parameters**: pool_name (str), table_name (str), data_source (str), file_format (str), location (str), schema (List)
- **Returns**: `Dict[str, Any]`

**`pause_sql_pool(pool_name)`**
- Pauses a dedicated SQL pool
- **Returns**: `Dict[str, Any]`

**`resume_sql_pool(pool_name)`**
- Resumes a paused dedicated SQL pool
- **Returns**: `Dict[str, Any]`

**`scale_sql_pool(pool_name, new_sku)`**
- Scales a dedicated SQL pool
- **Returns**: `Dict[str, Any]`

#### Spark Pool Methods

**`create_spark_pool(pool_name, node_size, node_count, spark_version, auto_scale, auto_pause)`**
- Creates Apache Spark pool
- **Returns**: `SparkPool`

**`submit_spark_job(pool_name, job_name, main_file, arguments, executor_count, executor_size)`**
- Submits Spark job for execution
- **Returns**: `Dict[str, Any]`

**`execute_spark_sql(pool_name, sql_query)`**
- Executes SQL query using Spark SQL
- **Returns**: `Dict[str, Any]`

**`create_spark_notebook(notebook_name, language, cells)`**
- Creates Spark notebook
- **Returns**: `Dict[str, Any]`

#### Pipeline Methods

**`create_pipeline(pipeline_name, description, activities, parameters)`**
- Creates data integration pipeline
- **Returns**: `Pipeline`

**`run_pipeline(pipeline_name, parameters)`**
- Triggers pipeline run
- **Returns**: `Dict[str, Any]`

**`get_pipeline_run_status(run_id)`**
- Gets pipeline run status
- **Returns**: `Dict[str, Any]`

**`create_trigger(trigger_name, trigger_type, pipeline_name, schedule)`**
- Creates pipeline trigger
- **Returns**: `Dict[str, Any]`

#### Monitoring Methods

**`get_query_statistics(pool_name, start_time, end_time)`**
- Gets query performance statistics
- **Returns**: `Dict[str, Any]`

**`get_spark_pool_metrics(pool_name)`**
- Gets Spark pool metrics
- **Returns**: `Dict[str, Any]`

**`analyze_query_plan(query)`**
- Analyzes query execution plan
- **Returns**: `Dict[str, Any]`

**`optimize_table(table_name)`**
- Optimizes table storage and performance
- **Returns**: `Dict[str, Any]`

## Best Practices

### 1. Choose the Right Pool Type
```python
# Use serverless for ad-hoc queries and exploration
serverless = manager.create_sql_pool("ExplorationPool", PoolType.SERVERLESS)

# Use dedicated for production workloads
dedicated = manager.create_sql_pool("ProductionPool", PoolType.DEDICATED, sku="DW100c")
```

### 2. Implement Cost Optimization
```python
# Pause during non-business hours
manager.pause_sql_pool("ProductionPool")

# Resume before business hours
manager.resume_sql_pool("ProductionPool")
```

### 3. Leverage External Tables
```python
# Query data lake directly without data duplication
external_table = manager.create_external_table(
    pool_name="ServerlessPool01",
    table_name="sales_external",
    data_source="datalake_source",
    file_format="PARQUET",
    location="/sales/*.parquet",
    schema=schema
)
```

### 4. Optimize Queries
```python
# Always analyze query plans before running expensive queries
plan = manager.analyze_query_plan(complex_query)
for recommendation in plan['recommendations']:
    implement_recommendation(recommendation)
```

### 5. Design Modular Pipelines
```python
# Create reusable activities with parameters
pipeline = manager.create_pipeline(
    pipeline_name="FlexiblePipeline",
    activities=[copy_activity],
    parameters={"date": "@pipeline().parameters.date", "source": "@pipeline().parameters.source"}
)
```

## Use Cases

### 1. Enterprise Data Warehouse
Build a centralized data warehouse with dedicated SQL pools for consolidated reporting and analytics across multiple data sources.

### 2. Big Data Processing
Process large-scale datasets using Spark pools for ETL, data transformations, and machine learning workloads.

### 3. Real-time Analytics
Query streaming data directly from data lake using serverless SQL pools for low-latency analytical queries.

### 4. Data Lake Exploration
Explore and analyze data stored in Azure Data Lake without moving or copying data using external tables.

### 5. Hybrid Analytics
Combine SQL and Spark for comprehensive analytics - use SQL for structured queries and Spark for complex transformations.

## Troubleshooting

### Common Issues

**Issue**: Query performance is slow
**Solution**: Analyze query plan, add indexes, update statistics, consider materialized views

**Issue**: Pipeline fails intermittently
**Solution**: Implement retry policies, verify dataset schemas, check linked service connectivity

**Issue**: Spark job runs out of memory
**Solution**: Increase executor size, optimize partitioning, tune Spark configuration

**Issue**: High costs
**Solution**: Pause dedicated pools when not in use, use serverless for ad-hoc queries

**Issue**: Connection timeout
**Solution**: Check firewall rules, verify network connectivity, validate credentials

## Deployment

### Azure CLI Deployment
```bash
# Create Synapse workspace
az synapse workspace create \
    --name my-synapse-workspace \
    --resource-group my-resource-group \
    --storage-account my-storage-account \
    --file-system synapse-fs \
    --sql-admin-login-user sqladmin \
    --sql-admin-login-password <password> \
    --location eastus

# Create SQL pool
az synapse sql pool create \
    --name DedicatedPool01 \
    --performance-level DW100c \
    --workspace-name my-synapse-workspace \
    --resource-group my-resource-group

# Create Spark pool
az synapse spark pool create \
    --name SparkPool01 \
    --workspace-name my-synapse-workspace \
    --resource-group my-resource-group \
    --spark-version 3.3 \
    --node-count 3 \
    --node-size Medium
```

### Container Deployment
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY azure_synapse.py .
CMD ["python", "azure_synapse.py"]
```

## Monitoring

### Key Metrics
- Query execution time and throughput
- Pool utilization (CPU, memory, storage)
- Pipeline success/failure rates
- Data movement throughput
- Active sessions and connections
- Resource consumption and costs

### Azure Monitor Integration
```bash
# Enable diagnostic logs
az synapse workspace update \
    --name my-synapse-workspace \
    --resource-group my-resource-group \
    --enable-diagnostic-settings
```

### Log Analytics Queries
```kusto
// Query workspace logs
SynapseSqlPoolExecRequests
| where TimeGenerated > ago(1h)
| summarize QueryCount = count(), AvgDuration = avg(DurationMs) by Command
| order by AvgDuration desc
```

## Dependencies

```
Python >= 3.8
azure-core >= 1.26.0
dataclasses
typing
datetime
json
enum
```

See `requirements.txt` for complete list.

## Version History

- **v1.0.0**: Initial release with SQL and Spark pools
- **v1.1.0**: Added pipeline orchestration
- **v1.2.0**: Enhanced monitoring and optimization features
- **v2.0.0**: Added workspace management and security features

## Contributing

Contributions are welcome! Please submit pull requests or open issues on GitHub.

## License

This project is part of the Brill Consulting portfolio.

## Support

For questions or support:
- **Email**: clientbrill@gmail.com
- **LinkedIn**: [brillconsulting](https://www.linkedin.com/in/brillconsulting)

## Related Projects

- [Azure Data Factory](../DataFactory/)
- [Azure Data Lake Storage](../DataLake/)
- [Azure Databricks](../Databricks/)

---

**Built with Azure Synapse Analytics** | **Brill Consulting © 2024**
