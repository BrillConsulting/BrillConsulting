# Cloud Run - Serverless Container Platform

Comprehensive Cloud Run implementation for deploying and managing serverless containerized applications with automatic scaling.

## Features

### Service Deployment
- **Container Deployment**: Deploy from Container Registry or Artifact Registry
- **Resource Configuration**: Configure CPU and memory limits
- **Port Configuration**: Expose custom container ports
- **Environment Variables**: Set runtime environment configuration

### Autoscaling
- **Instance Scaling**: Configure min (0) and max (100) instances
- **Concurrency Control**: Set requests per container (default: 80)
- **Scale to Zero**: Automatic scale down to zero instances
- **Cold Start Optimization**: Minimize cold start latency

### Traffic Management
- **Traffic Splitting**: Gradual rollout between revisions (90% old, 10% new)
- **Canary Deployments**: Test new versions with limited traffic
- **Revision Management**: Deploy and manage multiple revisions
- **Rollback Support**: Quick rollback to previous revisions

### IAM and Security
- **Public Access**: Allow unauthenticated access with allUsers
- **Service Account Access**: Restrict to specific service accounts
- **IAM Bindings**: Fine-grained access control
- **HTTPS Only**: Automatic HTTPS with managed certificates

### Secrets Integration
- **Secret Manager**: Mount secrets as volumes or environment variables
- **Secret Versioning**: Use specific secret versions
- **Secret Rotation**: Automatic secret updates
- **Secure Access**: Encrypted secret storage

## Usage Example

```python
from cloudrun import CloudRunManager

# Initialize manager
mgr = CloudRunManager(
    project_id='my-gcp-project',
    region='us-central1'
)

# Deploy service
service = mgr.service.deploy_service({
    'service_name': 'my-api',
    'image': 'gcr.io/my-project/my-api:v1.0',
    'port': 8080,
    'memory': '512Mi',
    'cpu': '1',
    'env_vars': {
        'DATABASE_URL': 'postgres://...',
        'API_KEY': 'secret-key'
    }
})

# Configure autoscaling
mgr.scaling.configure_autoscaling({
    'service_name': 'my-api',
    'min_instances': 0,
    'max_instances': 100,
    'max_concurrency': 80
})

# Traffic splitting
mgr.traffic.split_traffic({
    'service_name': 'my-api',
    'revisions': [
        {'revision': 'my-api-v1', 'percent': 90},
        {'revision': 'my-api-v2', 'percent': 10}
    ]
})

# Canary deployment
mgr.traffic.canary_deployment('my-api', 'my-api-v2', canary_percent=10)

# Make service public
mgr.iam.make_service_public('my-api', 'us-central1')

# Mount secret
mgr.secrets.mount_secret('my-api', 'database-password', '/secrets/db')
```

## Best Practices

1. **Use autoscaling** for variable workloads
2. **Implement health checks** for reliability
3. **Use canary deployments** for safe rollouts
4. **Configure appropriate concurrency** based on application
5. **Mount secrets** instead of environment variables for sensitive data
6. **Set resource limits** to control costs

## Requirements

```
google-cloud-run
google-cloud-iam
google-cloud-secretmanager
```

## Author

BrillConsulting - Enterprise Cloud Solutions
