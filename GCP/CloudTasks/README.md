# Cloud Tasks - Asynchronous Task Queue Service

Comprehensive Cloud Tasks implementation for managing distributed task queues with HTTP and App Engine targets, rate limiting, and reliable task execution.

## Features

### Queue Management
- **Queue Creation**: Create queues with rate limiting and retry policies
- **Rate Limiting**: Control max concurrent and per-second dispatches
- **Retry Configuration**: Configure max attempts and backoff durations
- **Queue Pause/Resume**: Temporarily stop task processing

### HTTP Tasks
- **Task Creation**: Create HTTP target tasks with custom payloads
- **Task Scheduling**: Delay task execution with schedule_delay
- **Authentication**: OIDC token support for Cloud Run
- **Batch Creation**: Create multiple tasks efficiently

### App Engine Tasks
- **Service Routing**: Route tasks to specific App Engine services
- **Version Targeting**: Target specific service versions
- **Instance Selection**: Route to specific instances

### Task Monitoring
- **Task Listing**: View all tasks in a queue
- **Task Deletion**: Remove tasks from queue
- **Queue Purging**: Remove all tasks from queue
- **Execution Tracking**: Monitor dispatch and response counts

### Reliability Features
- **Max Attempts**: Configure retry attempts (1-5)
- **Backoff Strategy**: Exponential backoff with configurable parameters
- **Task TTL**: Automatic task expiration (24 hours default)
- **Deduplication**: Named tasks for preventing duplicates

## Usage Example

```python
from cloud_tasks import CloudTasksManager

# Initialize manager
mgr = CloudTasksManager(
    project_id='my-gcp-project',
    location='us-central1'
)

# Create queue
queue = mgr.queue_manager.create_queue({
    'queue_name': 'api-tasks',
    'max_concurrent_dispatches': 500,
    'max_dispatches_per_second': 100.0,
    'max_attempts': 5
})

# Create HTTP task
task = mgr.http_tasks.create_http_task({
    'queue_name': 'api-tasks',
    'task_name': 'process-user-123',
    'url': 'https://api.example.com/process',
    'http_method': 'POST',
    'payload': {'user_id': 123, 'action': 'verify'},
    'schedule_delay_seconds': 60  # Delay 1 minute
})

# Create batch tasks
batch = mgr.http_tasks.create_batch_tasks({
    'queue_name': 'api-tasks',
    'url': 'https://api.example.com/batch',
    'items': [
        {'id': 1, 'value': 'a'},
        {'id': 2, 'value': 'b'},
        {'id': 3, 'value': 'c'}
    ]
})

# List tasks
mgr.monitoring.list_tasks('api-tasks')

# Delete task
mgr.monitoring.delete_task('api-tasks', 'process-user-123')

# Purge queue
mgr.monitoring.purge_queue('api-tasks')
```

## Best Practices

1. **Configure appropriate rate limits** per queue
2. **Use task scheduling** for delayed execution
3. **Implement idempotent task handlers**
4. **Set appropriate retry configurations**
5. **Monitor task execution** and failure rates
6. **Use named tasks** for deduplication

## Requirements

```
google-cloud-tasks
```

## Author

BrillConsulting - Enterprise Cloud Solutions
