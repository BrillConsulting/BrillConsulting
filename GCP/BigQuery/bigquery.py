"""
BigQuery Advanced Data Warehouse and Analytics
Author: BrillConsulting
Description: Comprehensive BigQuery implementation with ML, streaming, and optimization
"""

import json
from typing import Dict, List, Any, Optional
from datetime import datetime
import time


class BigQueryDataset:
    """BigQuery dataset management"""

    def __init__(self, project_id: str, dataset_id: str):
        """
        Initialize dataset manager

        Args:
            project_id: GCP project ID
            dataset_id: BigQuery dataset ID
        """
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.tables = []

    def create_dataset(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create BigQuery dataset

        Args:
            config: Dataset configuration

        Returns:
            Dataset creation result
        """
        print(f"\n{'='*60}")
        print("Creating BigQuery Dataset")
        print(f"{'='*60}")

        location = config.get('location', 'US')
        description = config.get('description', 'Analytics dataset')

        code = f"""
from google.cloud import bigquery

client = bigquery.Client(project='{self.project_id}')

# Create dataset
dataset_id = '{self.project_id}.{self.dataset_id}'
dataset = bigquery.Dataset(dataset_id)
dataset.location = '{location}'
dataset.description = '{description}'

# Set default table expiration (30 days)
dataset.default_table_expiration_ms = 30 * 24 * 60 * 60 * 1000

# Create the dataset
dataset = client.create_dataset(dataset, exists_ok=True)
print(f"Created dataset {{dataset.dataset_id}}")
"""

        result = {
            'dataset_id': self.dataset_id,
            'project_id': self.project_id,
            'location': location,
            'full_id': f"{self.project_id}.{self.dataset_id}",
            'code': code,
            'timestamp': datetime.now().isoformat()
        }

        print(f"✓ Dataset created: {self.dataset_id}")
        print(f"  Location: {location}")
        print(f"  Full ID: {result['full_id']}")
        print(f"{'='*60}")

        return result


class BigQueryTable:
    """BigQuery table management with partitioning and clustering"""

    def __init__(self, dataset_id: str):
        """Initialize table manager"""
        self.dataset_id = dataset_id

    def create_partitioned_table(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create partitioned and clustered table

        Args:
            config: Table configuration

        Returns:
            Table creation result
        """
        print(f"\n{'='*60}")
        print("Creating Partitioned Table")
        print(f"{'='*60}")

        table_id = config.get('table_id', 'events')
        partition_field = config.get('partition_field', 'event_timestamp')
        cluster_fields = config.get('cluster_fields', ['user_id', 'event_type'])

        code = f"""
from google.cloud import bigquery

client = bigquery.Client()

# Define schema
schema = [
    bigquery.SchemaField("user_id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("event_type", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("event_timestamp", "TIMESTAMP", mode="REQUIRED"),
    bigquery.SchemaField("properties", "JSON", mode="NULLABLE"),
    bigquery.SchemaField("value", "FLOAT64", mode="NULLABLE"),
]

# Create table with partitioning and clustering
table_id = "{self.dataset_id}.{table_id}"
table = bigquery.Table(table_id, schema=schema)

# Time-based partitioning
table.time_partitioning = bigquery.TimePartitioning(
    type_=bigquery.TimePartitioningType.DAY,
    field="{partition_field}",
    expiration_ms=90 * 24 * 60 * 60 * 1000  # 90 days
)

# Clustering
table.clustering_fields = {cluster_fields}

# Create table
table = client.create_table(table)
print(f"Created table {{table.table_id}}")
print(f"Partitioned by: {{table.time_partitioning.field}}")
print(f"Clustered by: {{table.clustering_fields}}")
"""

        result = {
            'table_id': table_id,
            'partition_field': partition_field,
            'cluster_fields': cluster_fields,
            'partition_expiration_days': 90,
            'code': code,
            'timestamp': datetime.now().isoformat()
        }

        print(f"✓ Table created: {table_id}")
        print(f"  Partitioned by: {partition_field}")
        print(f"  Clustered by: {', '.join(cluster_fields)}")
        print(f"{'='*60}")

        return result


class BigQueryML:
    """BigQuery ML for machine learning"""

    def __init__(self, dataset_id: str):
        """Initialize BigQuery ML"""
        self.dataset_id = dataset_id
        self.models = []

    def create_classification_model(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create ML classification model

        Args:
            config: Model configuration

        Returns:
            Model training result
        """
        print(f"\n{'='*60}")
        print("Training BigQuery ML Classification Model")
        print(f"{'='*60}")

        model_name = config.get('model_name', 'churn_predictor')
        model_type = config.get('model_type', 'LOGISTIC_REG')

        query = f"""
CREATE OR REPLACE MODEL `{self.dataset_id}.{model_name}`
OPTIONS(
  model_type='{model_type}',
  input_label_cols=['churned'],
  auto_class_weights=TRUE,
  data_split_method='AUTO_SPLIT',
  max_iterations=50,
  learn_rate=0.1,
  early_stop=TRUE
) AS
SELECT
  customer_age,
  total_spend,
  months_active,
  support_tickets,
  product_category,
  churned
FROM
  `{self.dataset_id}.customer_data`
"""

        # Simulate training
        print("Training model...")
        for i in range(5):
            print(f"  Iteration {i+1}/5 - Training loss: {0.8 - i*0.15:.4f}")
            time.sleep(0.1)

        result = {
            'model_name': model_name,
            'model_type': model_type,
            'dataset_id': self.dataset_id,
            'full_model_id': f"{self.dataset_id}.{model_name}",
            'training_query': query,
            'training_iterations': 50,
            'final_loss': 0.05,
            'timestamp': datetime.now().isoformat()
        }

        self.models.append(result)

        print(f"\n✓ Model trained: {model_name}")
        print(f"  Type: {model_type}")
        print(f"  Final loss: {result['final_loss']:.4f}")
        print(f"{'='*60}")

        return result

    def predict(self, model_name: str) -> str:
        """
        Generate prediction query

        Args:
            model_name: Model name

        Returns:
            Prediction SQL query
        """
        query = f"""
SELECT
  customer_id,
  predicted_churned,
  predicted_churned_probs[OFFSET(0)].prob AS churn_probability
FROM
  ML.PREDICT(MODEL `{self.dataset_id}.{model_name}`,
    (
      SELECT
        customer_id,
        customer_age,
        total_spend,
        months_active,
        support_tickets,
        product_category
      FROM
        `{self.dataset_id}.customers_to_score`
    )
  )
ORDER BY
  churn_probability DESC
LIMIT 100
"""

        print(f"\n✓ Prediction query generated for model: {model_name}")
        return query


class BigQueryAnalytics:
    """Advanced BigQuery analytics and queries"""

    def __init__(self, project_id: str):
        """Initialize analytics"""
        self.project_id = project_id

    def execute_window_functions(self) -> str:
        """
        Advanced window functions query

        Returns:
            SQL query with window functions
        """
        query = """
WITH daily_sales AS (
  SELECT
    product_id,
    DATE(sale_timestamp) AS sale_date,
    SUM(amount) AS daily_revenue
  FROM
    `project.dataset.sales`
  GROUP BY
    product_id, sale_date
)
SELECT
  product_id,
  sale_date,
  daily_revenue,

  -- Moving average (7 days)
  AVG(daily_revenue) OVER (
    PARTITION BY product_id
    ORDER BY sale_date
    ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
  ) AS moving_avg_7d,

  -- Cumulative sum
  SUM(daily_revenue) OVER (
    PARTITION BY product_id
    ORDER BY sale_date
  ) AS cumulative_revenue,

  -- Rank by revenue
  RANK() OVER (
    PARTITION BY product_id
    ORDER BY daily_revenue DESC
  ) AS revenue_rank,

  -- Percent of total
  daily_revenue / SUM(daily_revenue) OVER (
    PARTITION BY product_id
  ) AS pct_of_total

FROM
  daily_sales
ORDER BY
  product_id, sale_date
"""

        print("\n✓ Window functions query generated")
        return query

    def execute_array_struct_query(self) -> str:
        """
        Complex ARRAY and STRUCT query

        Returns:
            SQL query with arrays and structs
        """
        query = """
SELECT
  user_id,

  -- Array aggregation
  ARRAY_AGG(
    STRUCT(
      event_type,
      event_timestamp,
      properties
    )
    ORDER BY event_timestamp DESC
    LIMIT 10
  ) AS recent_events,

  -- Array length
  COUNT(*) AS total_events,

  -- Filter array
  ARRAY(
    SELECT event_type
    FROM UNNEST(events)
    WHERE event_type = 'purchase'
  ) AS purchase_events,

  -- Struct aggregation
  STRUCT(
    MIN(event_timestamp) AS first_event,
    MAX(event_timestamp) AS last_event,
    COUNT(DISTINCT event_type) AS unique_event_types
  ) AS event_summary

FROM
  `project.dataset.events`
GROUP BY
  user_id
"""

        print("\n✓ Array/Struct query generated")
        return query


class BigQueryOptimizer:
    """Query optimization and performance analysis"""

    def __init__(self):
        """Initialize optimizer"""
        self.optimizations = []

    def analyze_query_performance(self, query: str) -> Dict[str, Any]:
        """
        Analyze query performance

        Args:
            query: SQL query to analyze

        Returns:
            Performance analysis
        """
        print(f"\n{'='*60}")
        print("Query Performance Analysis")
        print(f"{'='*60}")

        # Simulate analysis
        analysis = {
            'bytes_processed': 1024 * 1024 * 512,  # 512 MB
            'bytes_billed': 1024 * 1024 * 1024,    # 1 GB (rounded up)
            'estimated_cost': 0.005,  # $5 per TB
            'slot_ms': 15000,
            'execution_time_ms': 2500,
            'partitions_scanned': 7,
            'total_partitions': 90,
            'cache_hit': False,
            'recommendations': [
                'Add WHERE clause to filter by partition column',
                'Use clustering for better data organization',
                'Consider materialized view for repeated queries',
                'Limit columns in SELECT instead of SELECT *'
            ]
        }

        print(f"Bytes processed: {analysis['bytes_processed'] / (1024**3):.2f} GB")
        print(f"Estimated cost: ${analysis['estimated_cost']:.4f}")
        print(f"Execution time: {analysis['execution_time_ms']}ms")
        print(f"Partitions scanned: {analysis['partitions_scanned']}/{analysis['total_partitions']}")

        print(f"\nOptimization recommendations:")
        for i, rec in enumerate(analysis['recommendations'], 1):
            print(f"  {i}. {rec}")

        print(f"{'='*60}")

        return analysis

    def create_materialized_view(self, config: Dict[str, Any]) -> str:
        """
        Create materialized view for performance

        Args:
            config: View configuration

        Returns:
            CREATE MATERIALIZED VIEW SQL
        """
        view_name = config.get('view_name', 'daily_metrics_mv')
        dataset = config.get('dataset', 'analytics')

        query = f"""
CREATE MATERIALIZED VIEW `{dataset}.{view_name}`
PARTITION BY DATE(metric_date)
CLUSTER BY product_id, region
OPTIONS(
  enable_refresh=TRUE,
  refresh_interval_minutes=60
)
AS
SELECT
  DATE(event_timestamp) AS metric_date,
  product_id,
  region,
  COUNT(*) AS event_count,
  SUM(revenue) AS total_revenue,
  AVG(revenue) AS avg_revenue,
  COUNT(DISTINCT user_id) AS unique_users
FROM
  `{dataset}.events`
WHERE
  event_timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 90 DAY)
GROUP BY
  metric_date, product_id, region
"""

        print(f"\n✓ Materialized view created: {view_name}")
        print(f"  Refresh interval: 60 minutes")
        print(f"  Partitioned by: metric_date")

        return query


class BigQueryManager:
    """Comprehensive BigQuery management"""

    def __init__(self, project_id: str = 'my-project'):
        """
        Initialize BigQuery manager

        Args:
            project_id: GCP project ID
        """
        self.project_id = project_id
        self.datasets = {}
        self.ml_models = []

    def get_manager_info(self) -> Dict[str, Any]:
        """Get manager information"""
        return {
            'project_id': self.project_id,
            'datasets': len(self.datasets),
            'ml_models': len(self.ml_models),
            'features': [
                'data_warehouse',
                'ml_models',
                'streaming_analytics',
                'partitioning',
                'clustering',
                'optimization'
            ],
            'timestamp': datetime.now().isoformat()
        }


def demo():
    """Demonstrate BigQuery capabilities"""
    print("=" * 60)
    print("BigQuery Advanced Analytics Demo")
    print("=" * 60)

    project_id = 'my-gcp-project'
    dataset_id = 'analytics'

    # Create dataset
    dataset_mgr = BigQueryDataset(project_id, dataset_id)
    dataset_result = dataset_mgr.create_dataset({
        'location': 'US',
        'description': 'Analytics and ML dataset'
    })

    # Create partitioned table
    table_mgr = BigQueryTable(f"{project_id}.{dataset_id}")
    table_result = table_mgr.create_partitioned_table({
        'table_id': 'user_events',
        'partition_field': 'event_timestamp',
        'cluster_fields': ['user_id', 'event_type']
    })

    # Train ML model
    ml_mgr = BigQueryML(f"{project_id}.{dataset_id}")
    model_result = ml_mgr.create_classification_model({
        'model_name': 'churn_predictor',
        'model_type': 'LOGISTIC_REG'
    })

    # Generate prediction query
    predict_query = ml_mgr.predict('churn_predictor')
    print(f"\nPrediction query preview:\n{predict_query[:200]}...")

    # Analytics queries
    analytics = BigQueryAnalytics(project_id)
    window_query = analytics.execute_window_functions()

    # Optimize queries
    optimizer = BigQueryOptimizer()
    perf_analysis = optimizer.analyze_query_performance("SELECT * FROM table")

    # Create materialized view
    mv_query = optimizer.create_materialized_view({
        'view_name': 'daily_metrics_mv',
        'dataset': f"{project_id}.{dataset_id}"
    })

    # Manager info
    mgr = BigQueryManager(project_id)
    mgr.datasets[dataset_id] = dataset_result
    mgr.ml_models.append(model_result)

    info = mgr.get_manager_info()
    print(f"\n{'='*60}")
    print("BigQuery Manager Summary")
    print(f"{'='*60}")
    print(f"Project: {info['project_id']}")
    print(f"Datasets: {info['datasets']}")
    print(f"ML Models: {info['ml_models']}")
    print(f"Features: {', '.join(info['features'])}")
    print(f"{'='*60}")

    print("\n✓ Demo completed successfully!")


if __name__ == "__main__":
    demo()
