# Dataproc - Managed Spark and Hadoop

Comprehensive Dataproc implementation for managing Apache Spark and Hadoop clusters with job submission, autoscaling, and workflow orchestration.

## Features

### Cluster Management
- **Cluster Creation**: Create fully managed Spark/Hadoop clusters
- **Machine Configuration**: Configure master and worker node types
- **Initialization Actions**: Run setup scripts on cluster startup
- **Lifecycle Policies**: Auto-delete idle clusters to save costs

### Autoscaling
- **Worker Autoscaling**: Scale workers from 2 to 10 instances
- **YARN Configuration**: Graceful decommissioning and scaling factors
- **Autoscaling Policies**: Reusable policies across clusters
- **Cost Optimization**: Automatic scale-down when idle

### Job Submission
- **Spark Jobs**: Submit Spark JAR jobs with properties
- **PySpark Jobs**: Execute Python Spark scripts
- **Hive Jobs**: Run Hive SQL queries
- **Job Parameters**: Pass arguments and configure resources

### Workflow Templates
- **Multi-Job Workflows**: Chain jobs with dependencies
- **Managed Clusters**: Auto-create/delete clusters per workflow
- **Job Orchestration**: Sequential job execution
- **Template Reuse**: Parameterized workflow templates

### Monitoring
- **Cluster Listing**: View all active clusters
- **Job Tracking**: Monitor job status and progress
- **Cluster Deletion**: Clean up unused clusters
- **Resource Monitoring**: Track cluster utilization

## Usage Example

```python
from dataproc import DataprocManager

# Initialize manager
mgr = DataprocManager(
    project_id='my-gcp-project',
    region='us-central1'
)

# Create cluster
cluster = mgr.cluster_manager.create_cluster({
    'cluster_name': 'analytics-cluster',
    'master_machine_type': 'n1-standard-4',
    'worker_machine_type': 'n1-standard-4',
    'num_workers': 2,
    'image_version': '2.1-debian11'
})

# Create autoscaling cluster
autoscaling = mgr.cluster_manager.create_autoscaling_cluster({
    'cluster_name': 'autoscaling-cluster',
    'min_workers': 2,
    'max_workers': 10
})

# Submit Spark job
spark = mgr.job_manager.submit_spark_job({
    'cluster_name': 'analytics-cluster',
    'main_class': 'com.example.ETLJob',
    'jar_file': 'gs://my-bucket/jars/etl-job.jar',
    'args': ['--input', 'gs://my-bucket/input']
})

# Submit PySpark job
pyspark = mgr.job_manager.submit_pyspark_job({
    'cluster_name': 'analytics-cluster',
    'main_python_file': 'gs://my-bucket/scripts/analysis.py'
})

# Submit Hive job
hive = mgr.job_manager.submit_hive_job({
    'cluster_name': 'analytics-cluster',
    'query_file': 'gs://my-bucket/hive/aggregation.sql'
})

# Create workflow template
workflow = mgr.workflow_manager.create_workflow_template({
    'template_id': 'daily-etl-workflow',
    'cluster_name': 'workflow-cluster'
})

# Run workflow
mgr.workflow_manager.instantiate_workflow('daily-etl-workflow')
```

## Best Practices

1. **Use autoscaling** for variable workloads
2. **Enable preemptible workers** for cost savings
3. **Use workflow templates** for complex pipelines
4. **Set idle deletion TTL** to reduce costs
5. **Monitor job metrics** and cluster utilization
6. **Use initialization actions** for custom setup

## Requirements

```
google-cloud-dataproc
```

## Author

BrillConsulting - Enterprise Cloud Solutions
