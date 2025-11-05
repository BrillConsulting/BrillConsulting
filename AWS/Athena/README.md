# AWS Athena

Serverless interactive query service to analyze data in Amazon S3 using standard SQL.

## Features

- **Serverless**: No infrastructure to manage
- **Standard SQL**: ANSI SQL support with Presto engine
- **Pay Per Query**: Pay only for data scanned
- **Federated Queries**: Query across multiple data sources
- **ACID Transactions**: Support for Apache Iceberg tables
- **Named Queries**: Save and reuse queries
- **Workgroups**: Organize users and control costs
- **Result Caching**: Automatic query result reuse

## Quick Start

```python
from aws_athena import AthenaManager

# Initialize
athena = AthenaManager(region='us-east-1')
output_location = 's3://my-athena-results/'

# Create database
athena.create_database(
    database_name='analytics_db',
    output_location=output_location
)

# Execute query and wait for results
result = athena.execute_query_and_wait(
    query_string='''
        SELECT customer_id, SUM(amount) as total
        FROM sales
        WHERE date >= DATE '2024-01-01'
        GROUP BY customer_id
        ORDER BY total DESC
        LIMIT 10
    ''',
    output_location=output_location,
    database='analytics_db'
)

# Process results
for row in result['results']['rows']:
    print(f"Customer {row['customer_id']}: ${row['total']}")
```

## Use Cases

- **Data Lake Analytics**: Query S3 data without ETL
- **Log Analysis**: Analyze CloudTrail, VPC Flow Logs, ALB logs
- **Business Intelligence**: Ad-hoc reporting and dashboards
- **Data Transformation**: Prepare data for ML or analytics
- **Cost Analysis**: Query AWS Cost and Usage Reports
- **Security Analysis**: Investigate security findings

## Cost Optimization

### Partition Data
```sql
CREATE EXTERNAL TABLE sales (
    customer_id INT,
    amount DECIMAL(10,2)
)
PARTITIONED BY (year INT, month INT)
STORED AS PARQUET
LOCATION 's3://my-bucket/sales/'
```

### Use Columnar Formats
- **Parquet**: 30-90% cost reduction
- **ORC**: Similar savings to Parquet
- **Compression**: GZIP, Snappy, ZSTD

### Set Workgroup Limits
```python
athena.create_workgroup(
    workgroup_name='dev-team',
    output_location='s3://results/',
    bytes_scanned_cutoff_per_query=10 * 1024**3  # 10 GB limit
)
```

## Advanced Features

### Federated Queries
Query across S3, DynamoDB, RDS, Redshift, and more using data source connectors.

### CTAS (Create Table As Select)
```sql
CREATE TABLE top_customers AS
SELECT customer_id, SUM(amount) as total
FROM sales
GROUP BY customer_id
ORDER BY total DESC
LIMIT 100
```

## Author

Brill Consulting
