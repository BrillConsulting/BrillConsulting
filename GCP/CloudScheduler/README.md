# Cloud Scheduler - Managed Cron Job Service

Comprehensive Cloud Scheduler implementation for creating and managing scheduled jobs with HTTP, Pub/Sub, and App Engine targets.

## Features

### Job Types
- **HTTP Jobs**: Schedule HTTP/HTTPS endpoint calls
- **Pub/Sub Jobs**: Trigger Pub/Sub topic messages
- **App Engine Jobs**: Execute App Engine service handlers
- **Retry Configuration**: Automatic retry with exponential backoff

### Scheduling
- **Cron Expressions**: Standard cron format (5 fields)
- **Time Zones**: Support for all IANA time zones
- **Flexible Schedules**: Minute, hourly, daily, weekly, monthly
- **Common Patterns**: Pre-defined schedule templates

### Job Management
- **Pause/Resume**: Temporarily disable jobs
- **Manual Execution**: Force run jobs immediately
- **Job Listing**: View all scheduled jobs
- **Job Deletion**: Remove unused jobs

### Retry Policies
- **Max Attempts**: Configure retry count (up to 5)
- **Backoff Duration**: Min and max backoff times
- **Max Doublings**: Exponential backoff multiplier
- **Retry Duration**: Total retry time limit

## Usage Example

```python
from cloud_scheduler import CloudSchedulerManager

# Initialize manager
mgr = CloudSchedulerManager(
    project_id='my-gcp-project',
    location='us-central1'
)

# Create HTTP job
http_job = mgr.job_manager.create_http_job({
    'job_name': 'api-health-check',
    'schedule': '*/5 * * * *',  # Every 5 minutes
    'uri': 'https://api.example.com/health',
    'http_method': 'GET',
    'headers': {'Authorization': 'Bearer token'}
})

# Create Pub/Sub job
pubsub_job = mgr.job_manager.create_pubsub_job({
    'job_name': 'data-processing',
    'schedule': '0 */2 * * *',  # Every 2 hours
    'topic_name': 'data-process-trigger',
    'message': {'action': 'process', 'batch_size': 1000}
})

# Create App Engine job
appengine_job = mgr.job_manager.create_app_engine_job({
    'job_name': 'daily-cleanup',
    'schedule': '0 3 * * *',  # Daily at 3 AM
    'relative_uri': '/tasks/cleanup',
    'http_method': 'POST'
})

# Pause job
mgr.monitoring.pause_job('data-processing')

# Resume job
mgr.monitoring.resume_job('data-processing')

# Run job now
mgr.monitoring.run_job_now('api-health-check')
```

## Common Schedules

```
Every minute:      * * * * *
Every 5 minutes:   */5 * * * *
Every hour:        0 * * * *
Daily at midnight: 0 0 * * *
Weekly on Monday:  0 0 * * 1
Monthly (1st):     0 0 1 * *
```

## Best Practices

1. **Use appropriate retry configurations**
2. **Set time zones explicitly**
3. **Monitor job execution status**
4. **Use Pub/Sub for reliable job processing**
5. **Implement idempotent job handlers**

## Requirements

```
google-cloud-scheduler
```

## Author

BrillConsulting - Enterprise Cloud Solutions
