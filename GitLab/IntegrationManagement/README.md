# Integration Management - Third-Party Service Integrations

Comprehensive GitLab integration management system for Slack, Jira, Jenkins, webhooks, monitoring tools, and OAuth applications with health monitoring and template support.

## Features

### Slack Integration
- **Team Notifications**: Automated Slack channel notifications
- **Event Filtering**: Push, merge request, pipeline, issue events
- **Custom Bot**: Configure username and channel
- **Broken Pipeline Alerts**: Only notify on failures
- **Message Attachments**: Rich message formatting

### Jira Integration
- **Issue Synchronization**: Sync GitLab issues to Jira
- **Commit Linking**: Link commits to Jira issues
- **Merge Request Sync**: Sync MR status to Jira
- **Project Mapping**: Map GitLab projects to Jira projects
- **Custom Issue Types**: Bug, Story, Task, Epic support

### Jenkins CI/CD
- **Build Triggers**: Automatic build on push/MR
- **Parameter Passing**: Custom build parameters
- **Status Updates**: Build status back to GitLab
- **Multi-Job Support**: Trigger multiple Jenkins jobs
- **Pipeline Integration**: Jenkins pipeline webhooks

### Generic Webhooks
- **Custom URLs**: POST to any endpoint
- **Event Selection**: Choose specific GitLab events
- **Secret Tokens**: Webhook verification
- **SSL Verification**: Optional SSL certificate validation
- **Testing**: Built-in webhook testing
- **Statistics**: Track trigger count and last triggered time

### Monitoring Integrations
- **Datadog**: Logs and metrics forwarding
- **Sentry**: Error tracking and performance monitoring
- **Prometheus**: Metrics export
- **New Relic**: APM integration
- **Custom Metrics**: Push custom metrics

### OAuth Applications
- **OAuth 2.0**: Standard OAuth 2.0 implementation
- **Scope Management**: api, read_user, write_repository scopes
- **Client Credentials**: Client ID and secret generation
- **Confidential Apps**: Public and confidential app support
- **Token Revocation**: Revoke app access

### Health Monitoring
- **Status Checks**: Monitor integration health
- **Response Time**: Track integration performance
- **History**: Health check history
- **Alerts**: Identify unhealthy integrations

### Integration Templates
- **Quick Setup**: Pre-configured integration templates
- **Default Values**: Common configuration defaults
- **Required Fields**: Validate configuration
- **Type Filtering**: Filter templates by integration type

## Usage Example

```python
from integration_management import IntegrationManagementManager, IntegrationType

# Initialize manager
mgr = IntegrationManagementManager(project_id='myorg/myproject')

# 1. Slack Integration
slack = mgr.slack.create_slack_integration({
    'name': 'Team Notifications',
    'webhook_url': 'https://hooks.slack.com/services/XXX/YYY/ZZZ',
    'channel': '#deployments',
    'username': 'GitLab Bot',
    'events': ['push', 'merge_request', 'pipeline'],
    'notify_only_broken_pipelines': True
})

# Send notification
mgr.slack.send_slack_notification(slack['integration_id'], {
    'text': 'Deployment successful! 🚀'
})

# 2. Jira Integration
jira = mgr.jira.create_jira_integration({
    'name': 'Jira Sync',
    'url': 'https://company.atlassian.net',
    'username': 'gitlab-bot@company.com',
    'api_token': 'your-jira-token',
    'project_key': 'PROJ',
    'issue_type': 'Task',
    'enable_commit_sync': True
})

# Sync issue
mgr.jira.sync_issue(jira['integration_id'], {
    'iid': 123,
    'title': 'Implement feature',
    'description': 'Feature description'
})

# 3. Jenkins CI/CD
jenkins = mgr.jenkins.create_jenkins_integration({
    'name': 'Jenkins CI',
    'jenkins_url': 'https://jenkins.company.com',
    'username': 'gitlab',
    'api_token': 'jenkins-token',
    'project_name': 'myapp-build',
    'trigger_on_push': True
})

# Trigger build
mgr.jenkins.trigger_jenkins_build(jenkins['integration_id'], {
    'branch': 'main',
    'commit_sha': 'abc123'
})

# 4. Generic Webhooks
webhook = mgr.webhooks.create_webhook({
    'name': 'Deployment Hook',
    'url': 'https://api.myapp.com/deploy',
    'events': ['pipeline'],
    'secret_token': 'my-secret',
    'enable_ssl_verification': True
})

# Test webhook
mgr.webhooks.test_webhook(webhook['webhook_id'])

# Trigger webhook
mgr.webhooks.trigger_webhook(webhook['webhook_id'], {
    'event_type': 'pipeline',
    'status': 'success'
})

# 5. Monitoring Integration
datadog = mgr.monitoring.create_datadog_integration({
    'name': 'Datadog Monitoring',
    'api_key': 'datadog-api-key',
    'env': 'production',
    'service': 'myapp',
    'enable_logs': True,
    'enable_metrics': True
})

sentry = mgr.monitoring.create_sentry_integration({
    'name': 'Sentry Errors',
    'dsn': 'https://xxx@sentry.io/yyy',
    'environment': 'production',
    'release': 'v1.2.3'
})

# 6. OAuth Application
oauth_app = mgr.oauth.create_oauth_application({
    'name': 'Mobile App',
    'redirect_uri': 'https://myapp.com/callback',
    'scopes': ['api', 'read_user', 'write_repository'],
    'confidential': True
})

# 7. Health Monitoring
health = mgr.health.check_integration_health(slack['integration_id'], 'slack')
# Returns: status, response_time_ms, last_check

unhealthy = mgr.health.get_unhealthy_integrations()

# 8. Templates
template = mgr.templates.create_template({
    'name': 'Standard Slack Setup',
    'integration_type': IntegrationType.SLACK.value,
    'description': 'Standard team notifications',
    'default_config': {
        'username': 'GitLab Bot',
        'events': ['push', 'merge_request']
    },
    'required_fields': ['webhook_url', 'channel']
})
```

## Supported Integrations

### Communication
- **Slack**: Team collaboration
- **Mattermost**: Open-source messaging
- **Microsoft Teams**: Microsoft collaboration
- **Discord**: Community chat

### Issue Tracking
- **Jira**: Atlassian issue tracking
- **Asana**: Project management
- **Redmine**: Open-source project management

### CI/CD
- **Jenkins**: Automation server
- **Bamboo**: Atlassian CI/CD
- **TeamCity**: JetBrains CI/CD

### Monitoring
- **Datadog**: Cloud monitoring
- **Prometheus**: Metrics collection
- **Sentry**: Error tracking
- **New Relic**: APM platform

### Security
- **SonarQube**: Code quality
- **Snyk**: Vulnerability scanning
- **WhiteSource**: Open source security

### Cloud Platforms
- **AWS CodeCommit**: AWS source control
- **Azure DevOps**: Microsoft DevOps
- **Google Cloud Build**: GCP CI/CD

## Slack Configuration

```python
slack = mgr.slack.create_slack_integration({
    'name': 'Deployment Notifications',
    'webhook_url': 'https://hooks.slack.com/services/T00/B00/XXX',
    'channel': '#deployments',
    'username': 'Deploy Bot',
    'events': [
        'push',                  # Code pushed
        'merge_request',         # MR created/updated
        'pipeline',              # Pipeline status
        'issue',                 # Issue created/updated
        'deployment'             # Deployment events
    ],
    'notify_only_broken_pipelines': True  # Only notify failures
})
```

## Jira Configuration

```python
jira = mgr.jira.create_jira_integration({
    'name': 'Jira Integration',
    'url': 'https://yourcompany.atlassian.net',
    'username': 'gitlab-integration@company.com',
    'api_token': 'your-atlassian-api-token',
    'project_key': 'PROJ',
    'issue_type': 'Task',  # Bug, Story, Task, Epic
    'transition_id': '21',  # Optional transition for commits
    'enable_commit_sync': True,
    'enable_merge_request_sync': True
})
```

## Webhook Events

### Available Events
- **push**: Code pushed to repository
- **merge_request**: MR created, updated, merged, closed
- **pipeline**: Pipeline started, succeeded, failed
- **issue**: Issue created, updated, closed
- **deployment**: Deployment succeeded, failed
- **tag_push**: Tag created or deleted
- **wiki_page**: Wiki page created or updated
- **release**: Release created

### Webhook Configuration
```python
webhook = mgr.webhooks.create_webhook({
    'name': 'Custom Integration',
    'url': 'https://api.example.com/webhooks/gitlab',
    'events': ['push', 'merge_request', 'pipeline'],
    'secret_token': 'webhook-secret-key',
    'enable_ssl_verification': True
})
```

## OAuth Scopes

### Available Scopes
- **api**: Full API access (read and write)
- **read_api**: Read-only API access
- **read_user**: Read user profile
- **read_repository**: Read repository code
- **write_repository**: Push to repository
- **read_registry**: Read container registry
- **write_registry**: Push to container registry
- **sudo**: Impersonate other users (admin only)

### OAuth Flow
```python
# Create OAuth app
app = mgr.oauth.create_oauth_application({
    'name': 'Mobile Application',
    'redirect_uri': 'myapp://oauth/callback',
    'scopes': ['api', 'read_user'],
    'confidential': True
})

# User authorizes app and receives code
# Exchange code for access token
# Use token to access GitLab API

# Revoke app if compromised
mgr.oauth.revoke_oauth_application(app['app_id'])
```

## Health Monitoring

```python
# Check integration health
health = mgr.health.check_integration_health(
    integration_id='slack-1',
    integration_type='slack'
)

# Health check returns:
{
    'status': 'healthy',        # healthy, degraded, unhealthy
    'response_time_ms': 150,
    'last_check': '2025-11-06T10:30:00Z'
}

# Get health history
history = mgr.health.get_health_history('slack-1')

# Find unhealthy integrations
unhealthy = mgr.health.get_unhealthy_integrations()
# Returns: ['jenkins-2', 'webhook-5']
```

## Integration Templates

```python
# Create template
template = mgr.templates.create_template({
    'name': 'Production Slack Notifications',
    'integration_type': IntegrationType.SLACK.value,
    'description': 'Standard setup for production deployments',
    'default_config': {
        'username': 'Production Bot',
        'events': ['pipeline', 'deployment'],
        'notify_only_broken_pipelines': True
    },
    'required_fields': ['webhook_url', 'channel']
})

# List all templates
all_templates = mgr.templates.list_templates()

# Filter by type
slack_templates = mgr.templates.list_templates(
    integration_type=IntegrationType.SLACK
)

# Use template
template = mgr.templates.get_template('template-1')
config = template['default_config']
config['webhook_url'] = 'https://hooks.slack.com/...'
config['channel'] = '#prod-deployments'

slack = mgr.slack.create_slack_integration(config)
```

## Best Practices

### Integration Setup
1. **Test First**: Always test webhooks before enabling
2. **Use Templates**: Leverage templates for consistent setup
3. **Monitor Health**: Regular health checks
4. **Secure Credentials**: Store tokens in secret managers
5. **Least Privilege**: Minimum required permissions

### Slack Integration
1. **Dedicated Channels**: Separate channels for different event types
2. **Filter Events**: Only subscribe to relevant events
3. **Broken Pipeline Alerts**: Enable for production
4. **Custom Bots**: Use descriptive bot names

### Jira Integration
1. **API Tokens**: Use API tokens, not passwords
2. **Project Mapping**: Clear GitLab-to-Jira project mapping
3. **Issue Types**: Consistent issue type usage
4. **Sync Schedule**: Avoid rate limits

### Webhook Security
1. **Secret Tokens**: Always use secret tokens
2. **SSL Verification**: Enable for production
3. **IP Whitelist**: Restrict webhook source IPs
4. **Payload Validation**: Validate webhook signatures

### OAuth Applications
1. **Minimal Scopes**: Request only needed scopes
2. **Confidential Apps**: Use for server-side apps
3. **Token Expiration**: Implement token refresh
4. **Revoke Compromised**: Immediate revocation

## Common Use Cases

### CI/CD Pipeline Notifications
```python
# Slack notification on pipeline failure
slack = mgr.slack.create_slack_integration({
    'channel': '#ci-alerts',
    'events': ['pipeline'],
    'notify_only_broken_pipelines': True
})
```

### Issue Tracking Integration
```python
# Sync all issues to Jira
jira = mgr.jira.create_jira_integration({
    'project_key': 'DEV',
    'enable_commit_sync': True,
    'enable_merge_request_sync': True
})
```

### External Build System
```python
# Trigger Jenkins on every push
jenkins = mgr.jenkins.create_jenkins_integration({
    'project_name': 'app-build',
    'trigger_on_push': True,
    'trigger_on_merge_request': False
})
```

### Monitoring Setup
```python
# Forward errors to Sentry
sentry = mgr.monitoring.create_sentry_integration({
    'dsn': 'https://xxx@sentry.io/yyy',
    'environment': 'production'
})

# Send metrics to Datadog
datadog = mgr.monitoring.create_datadog_integration({
    'api_key': 'datadog-key',
    'env': 'production',
    'service': 'web-api'
})
```

## Requirements

```
requests (for webhook HTTP calls)
```

## Configuration

### Environment Variables
```bash
export GITLAB_URL="https://gitlab.com"
export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/XXX"
export JIRA_URL="https://company.atlassian.net"
export JIRA_API_TOKEN="your-token"
```

### Python Configuration
```python
from integration_management import IntegrationManagementManager

mgr = IntegrationManagementManager(
    project_id='myorg/myproject',
    gitlab_url='https://gitlab.com'
)
```

## Troubleshooting

**Issue**: Slack notifications not sending
- Verify webhook URL is correct
- Check channel permissions
- Test webhook with test_webhook()

**Issue**: Jira sync failing
- Verify API token is valid
- Check project key exists
- Ensure user has write permissions

**Issue**: Jenkins build not triggering
- Verify Jenkins URL and credentials
- Check project name is correct
- Ensure Jenkins has GitLab plugin

**Issue**: Webhook timing out
- Check target URL is accessible
- Verify SSL certificate if using HTTPS
- Increase timeout settings

**Issue**: OAuth app authorization failing
- Verify redirect URI matches exactly
- Check scopes are valid
- Ensure app is not revoked

## Author

BrillConsulting - Enterprise Cloud Solutions
