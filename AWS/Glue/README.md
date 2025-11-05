# AWS Glue

Fully managed ETL (Extract, Transform, Load) service and data catalog for data lakes and analytics.

## Features

- **Data Catalog**: Centralized metadata repository
- **Crawlers**: Automatic schema discovery and table creation
- **ETL Jobs**: Serverless Apache Spark and Python Shell jobs
- **Job Bookmarks**: Track processed data to avoid reprocessing
- **Development Endpoints**: Interactive development environment
- **Triggers**: Schedule and event-driven job execution
- **Workflows**: Orchestrate complex ETL pipelines
- **Schema Registry**: Manage and validate schemas

## Quick Start

```python
from aws_glue import GlueManager

# Initialize
glue = GlueManager(region='us-east-1')

# Create database
glue.create_database(
    database_name='analytics_db',
    description='Analytics data lake'
)

# Create crawler to discover schema
glue.create_crawler(
    crawler_name='s3-crawler',
    role_arn='arn:aws:iam::123456789012:role/GlueServiceRole',
    database_name='analytics_db',
    s3_targets=['s3://my-data-lake/raw-data/'],
    schedule='cron(0 1 * * ? *)'  # Daily at 1 AM
)

# Start crawler
glue.start_crawler('s3-crawler')

# Create ETL job
glue.create_job(
    job_name='transform-data',
    role_arn='arn:aws:iam::123456789012:role/GlueServiceRole',
    script_location='s3://my-scripts/transform.py',
    glue_version='4.0',
    worker_type='G.1X',
    number_of_workers=10
)

# Run ETL job
job_run_id = glue.start_job_run('transform-data')
```

## Use Cases

- **Data Lake ETL**: Transform and prepare data in S3
- **Schema Discovery**: Automatically catalog data sources
- **Data Quality**: Clean and validate data
- **Change Data Capture**: Process incremental updates
- **Data Integration**: Combine data from multiple sources
- **Analytics Preparation**: Prepare data for Athena, Redshift, EMR

## ETL Script Example

```python
import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job

args = getResolvedOptions(sys.argv, ['JOB_NAME'])
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

# Read from Data Catalog
datasource = glueContext.create_dynamic_frame.from_catalog(
    database='analytics_db',
    table_name='raw_events'
)

# Transform
transformed = ApplyMapping.apply(
    frame=datasource,
    mappings=[
        ("event_id", "string", "id", "string"),
        ("timestamp", "long", "event_time", "timestamp"),
        ("user_id", "string", "user_id", "string")
    ]
)

# Write to S3 (Parquet format)
glueContext.write_dynamic_frame.from_options(
    frame=transformed,
    connection_type="s3",
    connection_options={"path": "s3://output/processed/"},
    format="parquet"
)

job.commit()
```

## Worker Types

- **Standard**: 4 vCPU, 16 GB memory (50 DPU)
- **G.1X**: 4 vCPU, 16 GB memory, 64 GB disk (1 DPU)
- **G.2X**: 8 vCPU, 32 GB memory, 128 GB disk (2 DPU)
- **G.025X**: 2 vCPU, 4 GB memory, 64 GB disk (0.25 DPU)

## Data Catalog Integration

Works seamlessly with:
- **Athena**: Query cataloged tables with SQL
- **Redshift Spectrum**: Query S3 data from Redshift
- **EMR**: Access catalog from Spark, Hive, Presto
- **Lake Formation**: Centralized permissions management

## Author

Brill Consulting
