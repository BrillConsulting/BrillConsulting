# Apache Spark Distributed Data Processing

Complete PySpark implementation for big data analytics and processing.

## Features

- **Spark Session**: Configure and create Spark sessions with optimizations
- **Data Reading**: Read from multiple formats (Parquet, JSON, CSV, Avro, Delta)
- **Transformations**: Filter, map, withColumn, dropDuplicates operations
- **Aggregations**: GroupBy, window functions, complex aggregations
- **Joins**: Inner, outer, left, right, cross joins
- **Spark SQL**: Execute distributed SQL queries
- **Streaming**: Structured Streaming with Kafka, Delta Lake
- **Optimization**: Caching, repartitioning, coalescing, adaptive query execution
- **Data Writing**: Write with partitioning and compression

## Technologies

- Apache Spark 3.5+
- PySpark
- Delta Lake
- Structured Streaming

## Usage

```python
from spark_processing import SparkProcessor

# Initialize processor
spark = SparkProcessor(
    app_name='DataProcessingApp',
    master='spark://master:7077'
)

# Create session
session = spark.create_spark_session({
    'spark.sql.adaptive.enabled': 'true',
    'spark.executor.memory': '4g'
})

# Read data
df = spark.read_data({
    'name': 'sales_data',
    'format': 'parquet',
    'path': 's3://bucket/sales/'
})

# Transform and aggregate
transformed = spark.transform_data({
    'source_df': 'sales_data',
    'transformations': ['filter', 'withColumn']
})
```

## Demo

```bash
python spark_processing.py
```
