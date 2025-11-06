# Cloud Build - Serverless CI/CD Platform

Comprehensive Google Cloud Build implementation for building, testing, and deploying applications with automated CI/CD pipelines, build triggers, and artifact management.

## Features

### Build Management
- **Multi-Step Builds**: Sequential and parallel build steps
- **Build Configuration**: YAML-based build definition
- **Machine Types**: E2_MEDIUM, E2_HIGHCPU_8, E2_HIGHCPU_32, E2_HIGHMEM_8
- **Timeout Control**: Configure build timeout (default: 600s, max: 24h)
- **Build Substitutions**: Dynamic variables (_VAR or $VAR)
- **Service Account**: Custom service account for builds
- **Build Cancellation**: Cancel running builds
- **Build History**: Query and filter build history

### Build Triggers
- **GitHub Triggers**: Branch, tag, and pull request triggers
- **Cloud Source Repositories**: Trigger on Cloud Source commits
- **Webhook Triggers**: Manual trigger via HTTP webhook
- **Branch Patterns**: Regex patterns (^main$, ^release-.*$)
- **Tag Patterns**: Semantic version patterns (^v[0-9]+\.[0-9]+\.[0-9]+$)
- **File Filters**: Include/ignore specific files (docs/**, *.md)
- **Trigger Substitutions**: Trigger-level variables
- **Enable/Disable**: Control trigger activation

### Artifact Management
- **Cloud Storage Artifacts**: Upload build artifacts to GCS
- **Maven Artifacts**: Publish to Artifact Registry (Maven)
- **npm Artifacts**: Publish to Artifact Registry (npm)
- **Docker Images**: Push to Container Registry/Artifact Registry
- **Artifact Paths**: Configure paths to upload (dist/**, build/**)

### Build Notifications
- **Pub/Sub Notifications**: Build status to Pub/Sub topics
- **Slack Notifications**: Send build status to Slack channels
- **Status Filters**: SUCCESS, FAILURE, TIMEOUT, CANCELLED
- **Webhook Notifications**: POST build status to webhooks

### Build Analytics
- **Build Statistics**: Success rate, average duration, total builds
- **Build Queries**: Filter by status, duration, date
- **Performance Tracking**: Monitor build performance over time
- **Build Logs**: Access build logs in Cloud Console

## Usage Example

```python
from gcp_build import CloudBuildManager

# Initialize manager
mgr = CloudBuildManager(project_id='my-gcp-project')

# 1. Create multi-step build
build_config = {
    'source': 'gs://my-bucket/source.tar.gz',
    'steps': [
        {
            'id': 'test',
            'name': 'gcr.io/cloud-builders/docker',
            'args': ['run', '--rm', 'python:3.11', 'pytest', 'tests/'],
            'env': ['ENV=test']
        },
        {
            'id': 'build',
            'name': 'gcr.io/cloud-builders/docker',
            'args': ['build', '-t', 'gcr.io/my-gcp-project/myapp:$COMMIT_SHA', '.'],
            'waitFor': ['test']
        },
        {
            'id': 'push',
            'name': 'gcr.io/cloud-builders/docker',
            'args': ['push', 'gcr.io/my-gcp-project/myapp:$COMMIT_SHA'],
            'waitFor': ['build']
        },
        {
            'id': 'deploy',
            'name': 'gcr.io/cloud-builders/gcloud',
            'args': ['run', 'deploy', 'myapp', '--image', 'gcr.io/my-gcp-project/myapp:$COMMIT_SHA'],
            'waitFor': ['push']
        }
    ],
    'images': ['gcr.io/my-gcp-project/myapp:$COMMIT_SHA'],
    'substitutions': {
        '_ENV': 'production',
        '_REGION': 'us-central1'
    },
    'machine_type': 'E2_HIGHCPU_8',
    'timeout': '1200s'
}

build = mgr.builds.create_build(build_config)
build_result = mgr.builds.start_build(build['id'])

# 2. Create GitHub triggers
# Production trigger
prod_trigger = mgr.triggers.create_github_trigger({
    'name': 'deploy-production',
    'description': 'Deploy to production on main branch',
    'repo_owner': 'myorg',
    'repo_name': 'myapp',
    'branch_pattern': '^main$',
    'filename': 'cloudbuild.yaml',
    'substitutions': {'_ENV': 'production'},
    'ignored_files': ['README.md', 'docs/**']
})

# Staging trigger
staging_trigger = mgr.triggers.create_github_trigger({
    'name': 'deploy-staging',
    'repo_owner': 'myorg',
    'repo_name': 'myapp',
    'branch_pattern': '^staging$',
    'substitutions': {'_ENV': 'staging'}
})

# Release trigger (tags)
release_trigger = mgr.triggers.create_github_trigger({
    'name': 'build-release',
    'repo_owner': 'myorg',
    'repo_name': 'myapp',
    'tag_pattern': '^v[0-9]+\\.[0-9]+\\.[0-9]+$',
    'substitutions': {'_RELEASE': 'true'}
})

# 3. Webhook trigger
webhook_trigger = mgr.triggers.create_webhook_trigger({
    'name': 'manual-deploy',
    'secret': 'my-webhook-secret',
    'substitutions': {'_TRIGGER_TYPE': 'webhook'}
})

# 4. Configure artifacts
storage_artifacts = mgr.artifacts.configure_storage_artifacts({
    'location': 'gs://my-build-artifacts/',
    'paths': ['dist/**', 'build/**', '*.zip']
})

maven_artifacts = mgr.artifacts.configure_maven_artifacts({
    'repository': 'us-central1-maven.pkg.dev/my-project/maven-repo',
    'group_id': 'com.example',
    'artifact_id': 'myapp',
    'version': '1.0.0'
})

# 5. Build notifications
pubsub_notification = mgr.notifications.create_pubsub_notification({
    'topic': f'projects/{mgr.project_id}/topics/build-notifications',
    'filter': 'FAILURE'
})

slack_notification = mgr.notifications.create_slack_notification({
    'webhook_url': 'https://hooks.slack.com/services/XXX/YYY/ZZZ',
    'channel': '#deployments',
    'filter': 'ALL'
})

# 6. Build analytics
all_builds = mgr.builds.list_builds(limit=100)
stats = mgr.history.get_build_stats(all_builds)
# Returns: total_builds, success_rate, avg_duration_seconds

successful_builds = mgr.history.query_builds(all_builds, {'status': 'SUCCESS'})
```

## Machine Types

### E2 (Cost-Optimized)
- **E2_MEDIUM**: 1 vCPU, 4GB RAM ($0.033/hour)
- **E2_HIGHCPU_8**: 8 vCPUs, 8GB RAM ($0.192/hour)
- **E2_HIGHCPU_32**: 32 vCPUs, 32GB RAM ($0.769/hour)
- **E2_HIGHMEM_8**: 8 vCPUs, 64GB RAM ($0.384/hour)

### N1 (Balanced)
- **N1_STANDARD_1**: 1 vCPU, 3.75GB RAM ($0.047/hour)
- **N1_STANDARD_4**: 4 vCPUs, 15GB RAM ($0.190/hour)
- **N1_HIGHCPU_8**: 8 vCPUs, 7.2GB RAM ($0.285/hour)
- **N1_HIGHMEM_8**: 8 vCPUs, 52GB RAM ($0.474/hour)

## Build Substitutions

### Built-in Variables
- **$PROJECT_ID**: GCP project ID
- **$BUILD_ID**: Unique build ID
- **$COMMIT_SHA**: Git commit SHA (GitHub triggers)
- **$BRANCH_NAME**: Git branch name
- **$TAG_NAME**: Git tag name
- **$REPO_NAME**: Repository name
- **$SHORT_SHA**: Short commit SHA (first 7 chars)

### Custom Variables
- **_VAR**: Custom substitution (e.g., _ENV, _REGION)
- Use in build steps: `$_ENV`, `${_ENV}`

## CloudBuild YAML Example

```yaml
# cloudbuild.yaml
steps:
  # Run tests
  - name: 'gcr.io/cloud-builders/docker'
    id: 'test'
    args: ['run', '--rm', 'python:3.11', 'pytest', 'tests/']
    env:
      - 'ENV=test'

  # Build Docker image
  - name: 'gcr.io/cloud-builders/docker'
    id: 'build'
    args: ['build', '-t', 'gcr.io/$PROJECT_ID/myapp:$COMMIT_SHA', '.']
    waitFor: ['test']

  # Push to Container Registry
  - name: 'gcr.io/cloud-builders/docker'
    id: 'push'
    args: ['push', 'gcr.io/$PROJECT_ID/myapp:$COMMIT_SHA']
    waitFor: ['build']

  # Deploy to Cloud Run
  - name: 'gcr.io/cloud-builders/gcloud'
    id: 'deploy'
    args:
      - 'run'
      - 'deploy'
      - 'myapp'
      - '--image=gcr.io/$PROJECT_ID/myapp:$COMMIT_SHA'
      - '--region=us-central1'
      - '--platform=managed'
    waitFor: ['push']

# Images to push
images:
  - 'gcr.io/$PROJECT_ID/myapp:$COMMIT_SHA'

# Build timeout
timeout: '1200s'

# Machine type
options:
  machineType: E2_HIGHCPU_8

# Substitutions
substitutions:
  _ENV: 'production'
  _REGION: 'us-central1'

# Artifacts
artifacts:
  objects:
    location: 'gs://my-build-artifacts/'
    paths: ['dist/**', 'build/**']
```

## Common Build Patterns

### Docker Build and Push
```yaml
steps:
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-t', 'gcr.io/$PROJECT_ID/image:$SHORT_SHA', '.']
  - name: 'gcr.io/cloud-builders/docker'
    args: ['push', 'gcr.io/$PROJECT_ID/image:$SHORT_SHA']

images: ['gcr.io/$PROJECT_ID/image:$SHORT_SHA']
```

### Multi-Stage Build
```yaml
steps:
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '--target=test', '-t', 'test-image', '.']
  - name: 'gcr.io/cloud-builders/docker'
    args: ['run', 'test-image', 'pytest']
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '--target=prod', '-t', 'gcr.io/$PROJECT_ID/app', '.']
```

### Parallel Steps
```yaml
steps:
  # These run in parallel
  - name: 'gcr.io/cloud-builders/npm'
    id: 'lint'
    args: ['run', 'lint']
    waitFor: ['-']  # Run immediately

  - name: 'gcr.io/cloud-builders/npm'
    id: 'test'
    args: ['test']
    waitFor: ['-']  # Run immediately

  # This waits for both
  - name: 'gcr.io/cloud-builders/npm'
    id: 'build'
    args: ['run', 'build']
    waitFor: ['lint', 'test']
```

## Branch Patterns

### Exact Match
- `^main$` - Only main branch
- `^master$` - Only master branch

### Prefix Match
- `^release-.*$` - All release branches (release-1.0, release-2.0)
- `^feature/.*$` - All feature branches

### Multiple Branches
- `^(main|staging|dev)$` - main, staging, or dev

### Tag Patterns
- `^v[0-9]+\.[0-9]+\.[0-9]+$` - Semantic versions (v1.0.0, v2.1.3)
- `^v.*$` - All tags starting with v

## Pricing

### Build Time
- **First 120 build-minutes/day**: Free
- **After 120 minutes**: $0.003/build-minute

### Machine Time
- **E2_MEDIUM**: $0.033/hour
- **E2_HIGHCPU_8**: $0.192/hour
- **E2_HIGHCPU_32**: $0.769/hour

### Storage
- **Build logs**: Free (90 days)
- **Build artifacts**: Standard Cloud Storage pricing

## Best Practices

1. **Use parallel steps** when possible with `waitFor: ['-']`
2. **Set appropriate timeouts** to avoid hanging builds (default: 600s)
3. **Use build caching** to speed up builds (Docker layer caching)
4. **Limit trigger scope** with included/ignored files
5. **Use substitutions** for environment-specific values
6. **Configure notifications** for build failures
7. **Use smaller machine types** for simple builds (E2_MEDIUM)
8. **Implement proper testing** before deployment steps
9. **Version Docker images** with commit SHA or tags
10. **Monitor build statistics** to optimize performance

## Common Use Cases

### Continuous Deployment
- Auto-deploy to Cloud Run on main branch push
- Deploy to staging on feature branch
- Production deployment on tag creation

### Container Registry
- Build and push Docker images
- Multi-arch builds (amd64, arm64)
- Image vulnerability scanning

### Testing Pipeline
- Run unit tests
- Run integration tests
- Generate code coverage reports

### Artifact Publishing
- Publish npm packages
- Publish Maven artifacts
- Upload build artifacts to Cloud Storage

## Requirements

```
google-cloud-build
google-cloud-pubsub
```

## Configuration

Set up authentication:
```bash
export GOOGLE_APPLICATION_CREDENTIALS="path/to/service-account-key.json"
```

## Security

- Use **Secret Manager** for sensitive values (API keys, passwords)
- Limit **service account** permissions to minimum required
- Use **private** GitHub repositories for sensitive code
- Enable **approval** for production deployments
- Configure **VPC** for private resource access

## Monitoring

- **Cloud Console**: View build history and logs
- **Cloud Logging**: Query build logs
- **Cloud Monitoring**: Create alerts on build metrics
- **Pub/Sub**: Consume build events for custom monitoring

## Author

BrillConsulting - Enterprise Cloud Solutions
