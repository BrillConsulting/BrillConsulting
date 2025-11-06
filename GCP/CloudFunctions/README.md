# Cloud Functions - Serverless Event-Driven Computing

Comprehensive Google Cloud Functions implementation for building scalable, event-driven serverless applications with HTTP, Pub/Sub, Cloud Storage, and Firestore triggers.

## Features

### Function Management
- **Function Deployment**: Deploy functions with advanced configuration
- **Runtime Support**: Python 3.9-3.11, Node.js 16-20, Go 1.19-1.21, Java 11-17
- **Memory Configuration**: 128MB-8GB (128, 256, 512, 1024, 2048, 4096, 8192)
- **Timeout Control**: 1-540 seconds (9 minutes max)
- **Autoscaling**: Configure min (0-100) and max (0-1000) instances
- **Concurrency**: 1-1000 concurrent requests per instance
- **Environment Variables**: Pass configuration securely
- **Secrets Integration**: Access Secret Manager secrets

### Trigger Types
- **HTTP Triggers**: RESTful APIs with CORS support
- **Pub/Sub Triggers**: Event-driven message processing
- **Cloud Storage Triggers**: File upload/delete/update events
- **Firestore Triggers**: Database change events (create/update/delete/write)
- **Retry Policies**: Automatic retries with exponential backoff

### IAM and Access Control
- **Invoker Permissions**: Fine-grained access control
- **Public Access**: Make functions publicly accessible (allUsers)
- **Authenticated Access**: Require authentication (allAuthenticatedUsers)
- **Service Accounts**: Custom service account configuration
- **IAM Bindings**: User and service account permissions

### Versioning and Traffic Management
- **Function Versions**: Create and manage multiple versions
- **Traffic Splitting**: Gradual rollouts and canary deployments
- **Rollback**: Quick rollback to previous versions
- **Version History**: Track all function versions

### Monitoring and Logging
- **Execution Logs**: Track all function invocations
- **Performance Metrics**: Duration, memory usage, success rate
- **Error Tracking**: Monitor errors and timeouts
- **Alerts**: Create monitoring alerts for key metrics
- **Cloud Monitoring Integration**: Integration with GCP monitoring

## Usage Example

```python
from gcp_functions import CloudFunctionsManager

# Initialize manager
mgr = CloudFunctionsManager(
    project_id='my-gcp-project',
    region='us-central1'
)

# 1. Create HTTP API function
http_func = mgr.functions.create_function({
    'name': 'api-gateway',
    'runtime': 'python311',
    'entry_point': 'handle_request',
    'memory_mb': 512,
    'timeout_seconds': 120,
    'max_instances': 50,
    'min_instances': 2,
    'concurrency': 100,
    'environment_variables': {
        'ENV': 'production',
        'API_VERSION': 'v2'
    }
})

# Create HTTP trigger with CORS
mgr.triggers.create_http_trigger({
    'function_name': 'api-gateway',
    'allow_unauthenticated': True,
    'cors': {
        'origins': ['https://example.com'],
        'methods': ['GET', 'POST', 'PUT']
    }
})

# Make it publicly accessible
mgr.iam.make_public('api-gateway')

# 2. Create Pub/Sub event processor
pubsub_func = mgr.functions.create_function({
    'name': 'process-events',
    'runtime': 'python311',
    'entry_point': 'handle_pubsub',
    'memory_mb': 1024,
    'timeout_seconds': 300,
    'max_instances': 100
})

mgr.triggers.create_pubsub_trigger({
    'function_name': 'process-events',
    'topic': 'event-stream',
    'retry_policy': {
        'retry_attempts': 5,
        'min_backoff': '10s',
        'max_backoff': '600s'
    }
})

# 3. Create Storage trigger for image processing
storage_func = mgr.functions.create_function({
    'name': 'process-images',
    'runtime': 'python311',
    'entry_point': 'process_image',
    'memory_mb': 2048,
    'timeout_seconds': 540,
    'max_instances': 20
})

mgr.triggers.create_storage_trigger({
    'function_name': 'process-images',
    'bucket': 'uploaded-images',
    'event_type': 'finalize'
})

# 4. Create Firestore trigger
firestore_func = mgr.functions.create_function({
    'name': 'sync-user-data',
    'runtime': 'python311',
    'entry_point': 'sync_data',
    'memory_mb': 256,
    'timeout_seconds': 60
})

mgr.triggers.create_firestore_trigger({
    'function_name': 'sync-user-data',
    'document_path': 'users/{userId}',
    'event_type': 'write'
})

# 5. Version management and traffic splitting
# Create versions
mgr.versions.create_version({
    'function_name': 'api-gateway',
    'version_id': 'v1.0.0',
    'description': 'Initial production release'
})

mgr.versions.create_version({
    'function_name': 'api-gateway',
    'version_id': 'v1.1.0',
    'description': 'Performance improvements'
})

# Split traffic (90% v1.0.0, 10% v1.1.0 for canary)
mgr.versions.split_traffic({
    'function_name': 'api-gateway',
    'traffic_split': {
        'v1.0.0': 90,
        'v1.1.0': 10
    }
})

# Rollback if needed
mgr.versions.rollback('api-gateway', 'v1.0.0')

# 6. Monitoring and alerts
# Log execution
mgr.monitoring.log_execution({
    'function_name': 'api-gateway',
    'execution_id': 'exec-12345',
    'status': 'success',
    'duration_ms': 150,
    'memory_used_mb': 128
})

# Get metrics
metrics = mgr.monitoring.get_metrics('api-gateway')
# Returns: total_executions, success_rate, avg_duration_ms, avg_memory_mb

# Create alert
mgr.monitoring.create_alert({
    'function_name': 'api-gateway',
    'metric': 'error_rate',
    'threshold': 5,
    'notification_channels': ['email:admin@example.com']
})
```

## Supported Runtimes

### Python
- **python39**: Python 3.9
- **python310**: Python 3.10
- **python311**: Python 3.11 (recommended)

### Node.js
- **nodejs16**: Node.js 16 LTS
- **nodejs18**: Node.js 18 LTS
- **nodejs20**: Node.js 20 LTS (recommended)

### Go
- **go119**: Go 1.19
- **go120**: Go 1.20
- **go121**: Go 1.21 (recommended)

### Java
- **java11**: Java 11
- **java17**: Java 17 (recommended)

## Memory and Pricing

### Memory Options
- **128MB**: $0.000000231/100ms ($0.000231/GB-second)
- **256MB**: $0.000000463/100ms
- **512MB**: $0.000000925/100ms
- **1024MB**: $0.000001650/100ms
- **2048MB**: $0.000003300/100ms
- **4096MB**: $0.000006600/100ms
- **8192MB**: $0.000013200/100ms

### Free Tier (per month)
- **Invocations**: 2 million free
- **Compute Time**: 400,000 GB-seconds free
- **Outbound Data**: 5GB free

## Trigger Event Types

### Cloud Storage Events
- **finalize**: Object creation complete
- **delete**: Object deleted
- **archive**: Object archived
- **metadataUpdate**: Object metadata updated

### Firestore Events
- **create**: New document created
- **update**: Document updated
- **delete**: Document deleted
- **write**: Any write (create/update)

## Best Practices

1. **Use min_instances** for latency-sensitive applications to avoid cold starts
2. **Set appropriate timeouts** to prevent hanging functions (default: 60s)
3. **Configure retry policies** for Pub/Sub triggers to handle transient failures
4. **Use environment variables** for configuration instead of hardcoding
5. **Implement idempotency** for functions triggered by Pub/Sub
6. **Monitor metrics** regularly to optimize memory and timeout settings
7. **Use traffic splitting** for safe canary deployments
8. **Grant least privilege** IAM permissions (avoid allUsers unless necessary)
9. **Enable VPC connector** for private resource access
10. **Use Secret Manager** for sensitive data instead of environment variables

## Example Function Code

### Python HTTP Function

```python
# main.py
def handle_request(request):
    """HTTP Cloud Function."""
    request_json = request.get_json(silent=True)

    if request_json and 'name' in request_json:
        name = request_json['name']
    else:
        name = 'World'

    return f'Hello {name}!'
```

### Python Pub/Sub Function

```python
# main.py
import base64

def handle_pubsub(event, context):
    """Pub/Sub Cloud Function."""
    pubsub_message = base64.b64decode(event['data']).decode('utf-8')
    print(f'Received message: {pubsub_message}')

    # Process message
    return 'OK'
```

### Python Storage Function

```python
# main.py
def process_image(event, context):
    """Cloud Storage trigger function."""
    file_name = event['name']
    bucket_name = event['bucket']

    print(f'Processing file: {file_name} from bucket: {bucket_name}')

    # Process the image
    return 'OK'
```

## Deployment Commands

### Deploy HTTP Function
```bash
gcloud functions deploy api-gateway \
    --runtime=python311 \
    --entry-point=handle_request \
    --trigger-http \
    --allow-unauthenticated \
    --memory=512MB \
    --timeout=120s \
    --max-instances=50 \
    --min-instances=2 \
    --region=us-central1
```

### Deploy Pub/Sub Function
```bash
gcloud functions deploy process-events \
    --runtime=python311 \
    --entry-point=handle_pubsub \
    --trigger-topic=event-stream \
    --memory=1024MB \
    --timeout=300s \
    --region=us-central1
```

### Deploy Storage Function
```bash
gcloud functions deploy process-images \
    --runtime=python311 \
    --entry-point=process_image \
    --trigger-bucket=uploaded-images \
    --memory=2048MB \
    --timeout=540s \
    --region=us-central1
```

## Requirements

```
google-cloud-functions
google-cloud-logging
google-cloud-monitoring
```

## Configuration

Set up authentication:
```bash
export GOOGLE_APPLICATION_CREDENTIALS="path/to/service-account-key.json"
```

## Cold Start Optimization

1. **Use min_instances**: Keep instances warm (costs apply)
2. **Optimize dependencies**: Minimize package imports
3. **Use global variables**: Reuse connections across invocations
4. **Lazy load**: Import modules only when needed
5. **Use faster runtimes**: Python 3.11, Node.js 20, Go 1.21 have better cold start times

## Monitoring Metrics

- **Invocations**: Total function executions
- **Execution Time**: Duration in milliseconds
- **Memory Usage**: MB consumed per execution
- **Error Rate**: Percentage of failed executions
- **Active Instances**: Current running instances
- **Billable Time**: GB-seconds billed

## Common Use Cases

### API Gateway
- RESTful API endpoints
- Webhook handlers
- GraphQL resolvers

### Event Processing
- Pub/Sub message processing
- Event-driven microservices
- Stream processing

### File Processing
- Image resizing/optimization
- Video transcoding
- Document conversion

### Database Triggers
- Firestore change notifications
- Data synchronization
- Audit logging

### Scheduled Tasks
- Cron jobs via Cloud Scheduler
- Periodic data cleanup
- Report generation

## Author

BrillConsulting - Enterprise Cloud Solutions
