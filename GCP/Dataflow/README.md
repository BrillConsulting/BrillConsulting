# Dataflow - Stream and Batch Processing

Comprehensive Dataflow implementation with Apache Beam for building data processing pipelines for both streaming and batch workloads.

## Features

### Batch Processing
- **Batch Pipelines**: Process large datasets efficiently
- **File I/O**: Read from and write to Cloud Storage
- **Transformations**: Map, Filter, GroupBy operations
- **BigQuery Integration**: Direct BigQuery read/write

### Streaming Processing
- **Real-Time Pipelines**: Process streaming data from Pub/Sub
- **Windowing**: Fixed, sliding, and session windows
- **Late Data Handling**: Configure watermarks and triggers
- **State Management**: Stateful processing with timers

### ETL Pipelines
- **Extract**: Read from BigQuery, Cloud Storage, Pub/Sub
- **Transform**: Data cleaning, enrichment, aggregation
- **Load**: Write to BigQuery, Cloud Storage, Firestore
- **Custom DoFn**: Advanced transformation logic

### Pipeline Templates
- **Reusable Templates**: Parameterized pipeline templates
- **Runtime Parameters**: Dynamic pipeline configuration
- **Template Deployment**: Deploy to Cloud Storage
- **Template Execution**: Launch templates with parameters

### Monitoring
- **Job Listing**: View all Dataflow jobs
- **Job Cancellation**: Stop running jobs
- **Metrics**: Monitor throughput and latency
- **Error Handling**: Track pipeline errors

## Usage Example

```python
from dataflow import DataflowManager

# Initialize manager
mgr = DataflowManager(
    project_id='my-gcp-project',
    region='us-central1'
)

# Create batch pipeline
batch = mgr.batch.create_batch_pipeline({
    'job_name': 'daily-batch-processing',
    'input_path': 'gs://my-bucket/input/data-*.json',
    'output_path': 'gs://my-bucket/output/',
    'temp_location': 'gs://my-bucket/temp/'
})

# Create ETL pipeline
etl = mgr.batch.create_etl_pipeline({
    'job_name': 'user-analytics-etl',
    'source_table': 'my-project.raw_data.user_events',
    'dest_table': 'my-project.analytics.user_metrics'
})

# Create streaming pipeline
streaming = mgr.streaming.create_streaming_pipeline({
    'job_name': 'realtime-analytics',
    'pubsub_topic': 'projects/my-project/topics/events',
    'output_table': 'my-project.realtime.aggregated_metrics',
    'window_duration_seconds': 60
})

# Create template
template = mgr.templates.create_template({
    'template_name': 'data-processor',
    'template_path': 'gs://my-bucket/templates/data-processor'
})

# Run template
mgr.templates.run_template({
    'template_path': 'gs://my-bucket/templates/data-processor',
    'job_name': 'process-daily-data',
    'parameters': {
        'input_path': 'gs://my-bucket/daily/*.json',
        'output_table': 'my-project.processed.daily_metrics'
    }
})
```

## Best Practices

1. **Use streaming pipelines** for real-time processing
2. **Apply windowing** for time-based aggregations
3. **Use templates** for reusable pipelines
4. **Configure appropriate autoscaling**
5. **Monitor pipeline metrics** and errors
6. **Use side inputs** for enrichment data

## Requirements

```
apache-beam[gcp]
google-cloud-dataflow
```

## Author

BrillConsulting - Enterprise Cloud Solutions
