# BigQuery - Cloud Data Warehouse

Comprehensive BigQuery implementation with advanced features for data warehousing, analytics, and machine learning.

## Features

### Dataset Management
- **Dataset Creation**: Create datasets with location and retention settings
- **Access Control**: Configure dataset-level IAM permissions
- **Data Organization**: Structured data organization with schemas

### Table Management
- **Partitioned Tables**: Time-based partitioning for query optimization
- **Clustered Tables**: Column-based clustering for improved performance
- **Schema Definition**: Flexible schema management with nested fields
- **Table Metadata**: Comprehensive table information and statistics

### BigQuery ML
- **Classification Models**: Logistic regression with auto hyperparameter tuning
- **Model Training**: SQL-based machine learning model creation
- **Predictions**: Real-time predictions using trained models
- **Model Evaluation**: Performance metrics and evaluation

### Analytics & Queries
- **Window Functions**: Moving averages, cumulative sums, ranking
- **Advanced SQL**: Complex queries with CTEs and subqueries
- **Materialized Views**: Pre-computed results for faster queries
- **Query Optimization**: Cost estimation and performance recommendations

### Query Optimization
- **Performance Analysis**: Query cost estimation
- **Optimization Recommendations**: Automated suggestions for query improvement
- **Execution Plans**: Detailed query execution analysis
- **Cost Management**: Query cost tracking and budgeting

## Usage Example

```python
from bigquery import BigQueryManager

# Initialize manager
mgr = BigQueryManager(project_id='my-gcp-project')

# Create dataset
dataset = mgr.dataset.create_dataset({
    'dataset_id': 'analytics_data',
    'location': 'US',
    'description': 'Analytics dataset'
})

# Create partitioned table
table = mgr.table.create_partitioned_table({
    'dataset_id': 'analytics_data',
    'table_id': 'user_events',
    'partition_field': 'timestamp',
    'clustering_fields': ['user_id', 'event_type']
})

# Train ML model
model = mgr.ml.create_classification_model({
    'dataset_id': 'analytics_data',
    'model_name': 'user_churn_model',
    'training_table': 'user_features',
    'label_column': 'churned'
})

# Execute analytics query
results = mgr.analytics.execute_window_functions()

# Optimize query
optimization = mgr.optimizer.analyze_query_performance(
    'SELECT * FROM `project.dataset.large_table`'
)
```

## Key Components

### BigQueryDataset
- Dataset lifecycle management
- Access control configuration
- Retention policy settings

### BigQueryTable
- Table creation and schema management
- Partitioning and clustering configuration
- Table metadata operations

### BigQueryML
- Model training and evaluation
- Prediction execution
- Hyperparameter tuning

### BigQueryAnalytics
- Advanced SQL queries
- Window functions
- Aggregation operations

### BigQueryOptimizer
- Query performance analysis
- Cost estimation
- Optimization recommendations

## Best Practices

1. **Use partitioning** for large tables with time-based queries
2. **Apply clustering** on frequently filtered columns
3. **Leverage BigQuery ML** for in-database machine learning
4. **Monitor query costs** using the optimizer
5. **Use materialized views** for frequently accessed aggregations
6. **Enable table expiration** for temporary data
7. **Apply column-level security** for sensitive data

## Requirements

```
google-cloud-bigquery
google-cloud-bigquery-storage
```

## Configuration

Set up authentication:
```bash
export GOOGLE_APPLICATION_CREDENTIALS="path/to/service-account-key.json"
```

## Advanced Features

- **Streaming Inserts**: Real-time data ingestion
- **Data Transfer Service**: Automated data imports
- **BigQuery BI Engine**: In-memory analysis service
- **Federated Queries**: Query external data sources
- **Data Encryption**: Automatic encryption at rest and in transit

## Performance Tips

- Partition large tables by date
- Cluster tables by commonly filtered columns
- Avoid SELECT * queries
- Use approximate aggregation functions when possible
- Cache query results for repeated queries
- Use clustering with partitioning for maximum performance

## Author

BrillConsulting - Enterprise Cloud Solutions
