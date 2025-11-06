"""
IntegrationManagement - Third-Party Service Integrations
Author: BrillConsulting
Description: Comprehensive GitLab integration management for Slack, Jira, Jenkins, webhooks, and custom integrations
"""
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from enum import Enum
import json


class IntegrationType(Enum):
    """Types of available integrations."""
    # Communication
    SLACK = "slack"
    MATTERMOST = "mattermost"
    MICROSOFT_TEAMS = "microsoft_teams"
    DISCORD = "discord"

    # Issue Tracking
    JIRA = "jira"
    ASANA = "asana"
    REDMINE = "redmine"

    # CI/CD
    JENKINS = "jenkins"
    BAMBOO = "bamboo"
    TEAMCITY = "teamcity"

    # Monitoring
    DATADOG = "datadog"
    PROMETHEUS = "prometheus"
    SENTRY = "sentry"
    NEW_RELIC = "new_relic"

    # Security
    SONARQUBE = "sonarqube"
    SNYK = "snyk"
    WHITESOURCE = "whitesource"

    # Cloud
    AWS_CODECOMMIT = "aws_codecommit"
    AZURE_DEVOPS = "azure_devops"
    GOOGLE_CLOUD_BUILD = "google_cloud_build"

    # Generic
    WEBHOOK = "webhook"
    OAUTH = "oauth"
    CUSTOM = "custom"


class IntegrationStatus(Enum):
    """Integration status."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    PENDING = "pending"
    TESTING = "testing"


class SlackIntegrationManager:
    """Manage Slack integrations."""

    def __init__(self, project_id: str):
        self.project_id = project_id
        self.integrations: Dict[str, Dict[str, Any]] = {}
        self.integration_counter = 1

    def create_slack_integration(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create Slack integration.

        Config:
        - name: Integration name
        - webhook_url: Slack webhook URL
        - channel: Default channel (e.g., #deployments)
        - username: Bot username (default: GitLab Bot)
        - events: List of events to notify (push, merge_request, pipeline, etc.)
        - notify_only_broken_pipelines: Only notify on failed pipelines
        """
        integration_id = f"slack-{self.integration_counter}"
        self.integration_counter += 1

        integration = {
            "integration_id": integration_id,
            "type": IntegrationType.SLACK.value,
            "name": config.get('name', 'Slack Notifications'),
            "webhook_url": config.get('webhook_url'),
            "channel": config.get('channel', '#general'),
            "username": config.get('username', 'GitLab Bot'),
            "events": config.get('events', ['push', 'merge_request', 'pipeline']),
            "notify_only_broken_pipelines": config.get('notify_only_broken_pipelines', False),
            "status": IntegrationStatus.ACTIVE.value,
            "created_at": datetime.now().isoformat()
        }

        self.integrations[integration_id] = integration
        return integration

    def send_slack_notification(self, integration_id: str, message: Dict[str, Any]) -> Dict[str, Any]:
        """Send notification to Slack."""
        integration = self.integrations.get(integration_id)
        if not integration:
            return {"status": "error", "message": "Integration not found"}

        notification = {
            "channel": integration['channel'],
            "username": integration['username'],
            "text": message.get('text'),
            "attachments": message.get('attachments', []),
            "timestamp": datetime.now().isoformat()
        }

        return {
            "status": "success",
            "integration_id": integration_id,
            "notification": notification
        }


class JiraIntegrationManager:
    """Manage Jira integrations."""

    def __init__(self, project_id: str):
        self.project_id = project_id
        self.integrations: Dict[str, Dict[str, Any]] = {}
        self.integration_counter = 1

    def create_jira_integration(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create Jira integration.

        Config:
        - name: Integration name
        - url: Jira instance URL (https://yourcompany.atlassian.net)
        - username: Jira username
        - api_token: Jira API token
        - project_key: Jira project key
        - issue_type: Default issue type (Bug, Story, Task)
        - transition_id: Transition ID for commit messages
        """
        integration_id = f"jira-{self.integration_counter}"
        self.integration_counter += 1

        integration = {
            "integration_id": integration_id,
            "type": IntegrationType.JIRA.value,
            "name": config.get('name', 'Jira Integration'),
            "url": config.get('url'),
            "username": config.get('username'),
            "api_token": config.get('api_token', '***'),
            "project_key": config.get('project_key'),
            "issue_type": config.get('issue_type', 'Task'),
            "transition_id": config.get('transition_id'),
            "enable_commit_sync": config.get('enable_commit_sync', True),
            "enable_merge_request_sync": config.get('enable_merge_request_sync', True),
            "status": IntegrationStatus.ACTIVE.value,
            "created_at": datetime.now().isoformat()
        }

        self.integrations[integration_id] = integration
        return integration

    def sync_issue(self, integration_id: str, gitlab_issue: Dict[str, Any]) -> Dict[str, Any]:
        """Sync GitLab issue to Jira."""
        integration = self.integrations.get(integration_id)
        if not integration:
            return {"status": "error", "message": "Integration not found"}

        jira_issue = {
            "project_key": integration['project_key'],
            "summary": gitlab_issue.get('title'),
            "description": gitlab_issue.get('description'),
            "issue_type": integration['issue_type'],
            "gitlab_issue_id": gitlab_issue.get('iid'),
            "created_at": datetime.now().isoformat()
        }

        return {
            "status": "success",
            "integration_id": integration_id,
            "jira_issue": jira_issue
        }


class JenkinsIntegrationManager:
    """Manage Jenkins CI/CD integrations."""

    def __init__(self, project_id: str):
        self.project_id = project_id
        self.integrations: Dict[str, Dict[str, Any]] = {}
        self.integration_counter = 1

    def create_jenkins_integration(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create Jenkins integration.

        Config:
        - name: Integration name
        - jenkins_url: Jenkins server URL
        - username: Jenkins username
        - api_token: Jenkins API token
        - project_name: Jenkins project/job name
        - trigger_on_push: Trigger build on push
        - trigger_on_merge_request: Trigger build on MR
        """
        integration_id = f"jenkins-{self.integration_counter}"
        self.integration_counter += 1

        integration = {
            "integration_id": integration_id,
            "type": IntegrationType.JENKINS.value,
            "name": config.get('name', 'Jenkins CI'),
            "jenkins_url": config.get('jenkins_url'),
            "username": config.get('username'),
            "api_token": config.get('api_token', '***'),
            "project_name": config.get('project_name'),
            "trigger_on_push": config.get('trigger_on_push', True),
            "trigger_on_merge_request": config.get('trigger_on_merge_request', True),
            "status": IntegrationStatus.ACTIVE.value,
            "created_at": datetime.now().isoformat()
        }

        self.integrations[integration_id] = integration
        return integration

    def trigger_jenkins_build(self, integration_id: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Trigger Jenkins build."""
        integration = self.integrations.get(integration_id)
        if not integration:
            return {"status": "error", "message": "Integration not found"}

        build = {
            "integration_id": integration_id,
            "project_name": integration['project_name'],
            "parameters": params or {},
            "triggered_at": datetime.now().isoformat()
        }

        return {
            "status": "success",
            "build": build
        }


class WebhookManager:
    """Manage generic webhooks."""

    def __init__(self, project_id: str):
        self.project_id = project_id
        self.webhooks: Dict[str, Dict[str, Any]] = {}
        self.webhook_counter = 1

    def create_webhook(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create webhook integration.

        Config:
        - name: Webhook name
        - url: Webhook URL
        - events: List of events to trigger (push, merge_request, pipeline, etc.)
        - secret_token: Secret token for verification
        - enable_ssl_verification: Enable SSL certificate verification
        """
        webhook_id = f"webhook-{self.webhook_counter}"
        self.webhook_counter += 1

        webhook = {
            "webhook_id": webhook_id,
            "type": IntegrationType.WEBHOOK.value,
            "name": config.get('name', 'Generic Webhook'),
            "url": config.get('url'),
            "events": config.get('events', ['push']),
            "secret_token": config.get('secret_token', '***'),
            "enable_ssl_verification": config.get('enable_ssl_verification', True),
            "status": IntegrationStatus.ACTIVE.value,
            "last_triggered": None,
            "trigger_count": 0,
            "created_at": datetime.now().isoformat()
        }

        self.webhooks[webhook_id] = webhook
        return webhook

    def trigger_webhook(self, webhook_id: str, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """Trigger webhook with event data."""
        webhook = self.webhooks.get(webhook_id)
        if not webhook:
            return {"status": "error", "message": "Webhook not found"}

        if webhook['status'] != IntegrationStatus.ACTIVE.value:
            return {"status": "error", "message": "Webhook is not active"}

        # Update webhook statistics
        webhook['last_triggered'] = datetime.now().isoformat()
        webhook['trigger_count'] += 1

        return {
            "status": "success",
            "webhook_id": webhook_id,
            "url": webhook['url'],
            "event_type": event_data.get('event_type'),
            "triggered_at": webhook['last_triggered']
        }

    def test_webhook(self, webhook_id: str) -> Dict[str, Any]:
        """Test webhook connection."""
        webhook = self.webhooks.get(webhook_id)
        if not webhook:
            return {"status": "error", "message": "Webhook not found"}

        test_payload = {
            "event_type": "test",
            "message": "This is a test webhook",
            "timestamp": datetime.now().isoformat()
        }

        return {
            "status": "success",
            "webhook_id": webhook_id,
            "test_payload": test_payload
        }


class MonitoringIntegrationManager:
    """Manage monitoring tool integrations (Datadog, Prometheus, Sentry)."""

    def __init__(self, project_id: str):
        self.project_id = project_id
        self.integrations: Dict[str, Dict[str, Any]] = {}
        self.integration_counter = 1

    def create_datadog_integration(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create Datadog integration.

        Config:
        - name: Integration name
        - api_key: Datadog API key
        - api_host: Datadog API host (default: api.datadoghq.com)
        - env: Environment tag (production, staging, development)
        - service: Service name
        """
        integration_id = f"datadog-{self.integration_counter}"
        self.integration_counter += 1

        integration = {
            "integration_id": integration_id,
            "type": IntegrationType.DATADOG.value,
            "name": config.get('name', 'Datadog Monitoring'),
            "api_key": config.get('api_key', '***'),
            "api_host": config.get('api_host', 'api.datadoghq.com'),
            "env": config.get('env', 'production'),
            "service": config.get('service'),
            "enable_logs": config.get('enable_logs', True),
            "enable_metrics": config.get('enable_metrics', True),
            "status": IntegrationStatus.ACTIVE.value,
            "created_at": datetime.now().isoformat()
        }

        self.integrations[integration_id] = integration
        return integration

    def create_sentry_integration(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create Sentry error tracking integration.

        Config:
        - name: Integration name
        - dsn: Sentry DSN
        - environment: Environment (production, staging)
        - release: Release version
        """
        integration_id = f"sentry-{self.integration_counter}"
        self.integration_counter += 1

        integration = {
            "integration_id": integration_id,
            "type": IntegrationType.SENTRY.value,
            "name": config.get('name', 'Sentry Error Tracking'),
            "dsn": config.get('dsn'),
            "environment": config.get('environment', 'production'),
            "release": config.get('release'),
            "enable_performance": config.get('enable_performance', True),
            "status": IntegrationStatus.ACTIVE.value,
            "created_at": datetime.now().isoformat()
        }

        self.integrations[integration_id] = integration
        return integration


class OAuthIntegrationManager:
    """Manage OAuth-based integrations."""

    def __init__(self, project_id: str):
        self.project_id = project_id
        self.oauth_apps: Dict[str, Dict[str, Any]] = {}
        self.app_counter = 1

    def create_oauth_application(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create OAuth application.

        Config:
        - name: Application name
        - redirect_uri: OAuth redirect URI
        - scopes: List of OAuth scopes (api, read_user, write_repository)
        - confidential: Confidential application (requires client secret)
        """
        app_id = f"oauth-app-{self.app_counter}"
        self.app_counter += 1

        # Generate client credentials
        client_id = f"client-{app_id}"
        client_secret = f"secret-{app_id}" if config.get('confidential', True) else None

        oauth_app = {
            "app_id": app_id,
            "type": IntegrationType.OAUTH.value,
            "name": config.get('name'),
            "client_id": client_id,
            "client_secret": client_secret,
            "redirect_uri": config.get('redirect_uri'),
            "scopes": config.get('scopes', ['api']),
            "confidential": config.get('confidential', True),
            "status": IntegrationStatus.ACTIVE.value,
            "created_at": datetime.now().isoformat()
        }

        self.oauth_apps[app_id] = oauth_app
        return oauth_app

    def revoke_oauth_application(self, app_id: str) -> Dict[str, Any]:
        """Revoke OAuth application access."""
        if app_id in self.oauth_apps:
            self.oauth_apps[app_id]['status'] = IntegrationStatus.INACTIVE.value
            self.oauth_apps[app_id]['revoked_at'] = datetime.now().isoformat()
            return {"status": "revoked", "app_id": app_id}
        return {"status": "not_found", "app_id": app_id}


class IntegrationHealthManager:
    """Monitor integration health and status."""

    def __init__(self):
        self.health_checks: Dict[str, List[Dict[str, Any]]] = {}

    def check_integration_health(self, integration_id: str, integration_type: str) -> Dict[str, Any]:
        """Check integration health."""
        health_check = {
            "integration_id": integration_id,
            "integration_type": integration_type,
            "status": "healthy",
            "response_time_ms": 150,
            "last_check": datetime.now().isoformat()
        }

        if integration_id not in self.health_checks:
            self.health_checks[integration_id] = []

        self.health_checks[integration_id].append(health_check)

        return health_check

    def get_health_history(self, integration_id: str) -> List[Dict[str, Any]]:
        """Get health check history for integration."""
        return self.health_checks.get(integration_id, [])

    def get_unhealthy_integrations(self) -> List[str]:
        """Get list of unhealthy integrations."""
        unhealthy = []

        for integration_id, checks in self.health_checks.items():
            if checks:
                latest_check = checks[-1]
                if latest_check['status'] != 'healthy':
                    unhealthy.append(integration_id)

        return unhealthy


class IntegrationTemplateManager:
    """Manage integration templates for quick setup."""

    def __init__(self):
        self.templates: Dict[str, Dict[str, Any]] = {}
        self.template_counter = 1

    def create_template(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create integration template.

        Config:
        - name: Template name
        - integration_type: IntegrationType
        - description: Template description
        - default_config: Default configuration values
        - required_fields: List of required configuration fields
        """
        template_id = f"template-{self.template_counter}"
        self.template_counter += 1

        template = {
            "template_id": template_id,
            "name": config.get('name'),
            "integration_type": config.get('integration_type'),
            "description": config.get('description'),
            "default_config": config.get('default_config', {}),
            "required_fields": config.get('required_fields', []),
            "created_at": datetime.now().isoformat()
        }

        self.templates[template_id] = template
        return template

    def get_template(self, template_id: str) -> Optional[Dict[str, Any]]:
        """Get integration template."""
        return self.templates.get(template_id)

    def list_templates(self, integration_type: Optional[IntegrationType] = None) -> List[Dict[str, Any]]:
        """List integration templates, optionally filtered by type."""
        if integration_type:
            integration_type_value = integration_type.value if isinstance(integration_type, IntegrationType) else integration_type
            return [t for t in self.templates.values() if t['integration_type'] == integration_type_value]
        return list(self.templates.values())


class IntegrationManagementManager:
    """Main integration management manager."""

    def __init__(self, project_id: str = 'default-project', gitlab_url: str = 'https://gitlab.com'):
        self.project_id = project_id
        self.gitlab_url = gitlab_url

        # Initialize all managers
        self.slack = SlackIntegrationManager(project_id)
        self.jira = JiraIntegrationManager(project_id)
        self.jenkins = JenkinsIntegrationManager(project_id)
        self.webhooks = WebhookManager(project_id)
        self.monitoring = MonitoringIntegrationManager(project_id)
        self.oauth = OAuthIntegrationManager(project_id)
        self.health = IntegrationHealthManager()
        self.templates = IntegrationTemplateManager()

    def get_all_integrations(self) -> Dict[str, Any]:
        """Get all configured integrations."""
        return {
            "slack": list(self.slack.integrations.values()),
            "jira": list(self.jira.integrations.values()),
            "jenkins": list(self.jenkins.integrations.values()),
            "webhooks": list(self.webhooks.webhooks.values()),
            "monitoring": list(self.monitoring.integrations.values()),
            "oauth_apps": list(self.oauth.oauth_apps.values())
        }

    def get_integration_count(self) -> Dict[str, int]:
        """Get count of integrations by type."""
        return {
            "slack": len(self.slack.integrations),
            "jira": len(self.jira.integrations),
            "jenkins": len(self.jenkins.integrations),
            "webhooks": len(self.webhooks.webhooks),
            "monitoring": len(self.monitoring.integrations),
            "oauth_apps": len(self.oauth.oauth_apps),
            "total": (
                len(self.slack.integrations) +
                len(self.jira.integrations) +
                len(self.jenkins.integrations) +
                len(self.webhooks.webhooks) +
                len(self.monitoring.integrations) +
                len(self.oauth.oauth_apps)
            )
        }

    def info(self) -> Dict[str, Any]:
        """Get integration management information."""
        return {
            "project_id": self.project_id,
            "gitlab_url": self.gitlab_url,
            "integration_count": self.get_integration_count(),
            "supported_integrations": [t.value for t in IntegrationType],
            "health_monitored": len(self.health.health_checks),
            "templates_available": len(self.templates.templates)
        }


def demo():
    """Demonstrate integration management capabilities."""
    print("=" * 80)
    print("GitLab Integration Management - Comprehensive Demo")
    print("=" * 80)

    # Initialize manager
    mgr = IntegrationManagementManager(project_id='myorg/myproject')

    print("\n💬 1. Slack Integration")
    print("-" * 80)

    # Create Slack integration
    slack_int = mgr.slack.create_slack_integration({
        'name': 'Team Notifications',
        'webhook_url': 'https://hooks.slack.com/services/XXX/YYY/ZZZ',
        'channel': '#deployments',
        'username': 'GitLab Deploy Bot',
        'events': ['push', 'merge_request', 'pipeline'],
        'notify_only_broken_pipelines': True
    })
    print(f"✓ Created Slack integration: {slack_int['name']}")
    print(f"  Channel: {slack_int['channel']}")
    print(f"  Events: {', '.join(slack_int['events'])}")

    # Send notification
    notification = mgr.slack.send_slack_notification(slack_int['integration_id'], {
        'text': 'Deployment to production successful! 🚀'
    })
    print(f"✓ Sent Slack notification to {notification['notification']['channel']}")

    print("\n🎯 2. Jira Integration")
    print("-" * 80)

    # Create Jira integration
    jira_int = mgr.jira.create_jira_integration({
        'name': 'Jira Issue Sync',
        'url': 'https://mycompany.atlassian.net',
        'username': 'gitlab-bot@mycompany.com',
        'api_token': 'jira-api-token-xxx',
        'project_key': 'PROJ',
        'issue_type': 'Task',
        'enable_commit_sync': True
    })
    print(f"✓ Created Jira integration: {jira_int['name']}")
    print(f"  Project: {jira_int['project_key']}")
    print(f"  Issue Type: {jira_int['issue_type']}")

    # Sync issue
    sync_result = mgr.jira.sync_issue(jira_int['integration_id'], {
        'iid': 123,
        'title': 'Implement new feature',
        'description': 'Add user authentication'
    })
    print(f"✓ Synced GitLab issue #123 to Jira: {sync_result['jira_issue']['summary']}")

    print("\n🔧 3. Jenkins CI/CD Integration")
    print("-" * 80)

    # Create Jenkins integration
    jenkins_int = mgr.jenkins.create_jenkins_integration({
        'name': 'Jenkins CI',
        'jenkins_url': 'https://jenkins.mycompany.com',
        'username': 'gitlab-jenkins',
        'api_token': 'jenkins-api-token-xxx',
        'project_name': 'myapp-build',
        'trigger_on_push': True,
        'trigger_on_merge_request': True
    })
    print(f"✓ Created Jenkins integration: {jenkins_int['name']}")
    print(f"  Project: {jenkins_int['project_name']}")

    # Trigger build
    build = mgr.jenkins.trigger_jenkins_build(jenkins_int['integration_id'], {
        'branch': 'main',
        'commit_sha': 'abc123'
    })
    print(f"✓ Triggered Jenkins build for project: {build['build']['project_name']}")

    print("\n🔗 4. Generic Webhooks")
    print("-" * 80)

    # Create webhooks
    webhook1 = mgr.webhooks.create_webhook({
        'name': 'Deployment Webhook',
        'url': 'https://api.myapp.com/hooks/deploy',
        'events': ['pipeline'],
        'secret_token': 'webhook-secret-xxx',
        'enable_ssl_verification': True
    })
    print(f"✓ Created webhook: {webhook1['name']}")
    print(f"  URL: {webhook1['url']}")

    webhook2 = mgr.webhooks.create_webhook({
        'name': 'Issue Tracker Webhook',
        'url': 'https://tracker.myapp.com/webhooks',
        'events': ['issue', 'merge_request']
    })
    print(f"✓ Created webhook: {webhook2['name']}")

    # Test webhook
    test_result = mgr.webhooks.test_webhook(webhook1['webhook_id'])
    print(f"✓ Webhook test: {test_result['status']}")

    # Trigger webhook
    trigger_result = mgr.webhooks.trigger_webhook(webhook1['webhook_id'], {
        'event_type': 'pipeline',
        'pipeline_status': 'success'
    })
    print(f"✓ Triggered webhook {trigger_result['webhook_id']}")

    print("\n📊 5. Monitoring Integrations")
    print("-" * 80)

    # Create Datadog integration
    datadog_int = mgr.monitoring.create_datadog_integration({
        'name': 'Datadog Monitoring',
        'api_key': 'datadog-api-key-xxx',
        'env': 'production',
        'service': 'myapp',
        'enable_logs': True,
        'enable_metrics': True
    })
    print(f"✓ Created Datadog integration: {datadog_int['name']}")
    print(f"  Environment: {datadog_int['env']}")
    print(f"  Service: {datadog_int['service']}")

    # Create Sentry integration
    sentry_int = mgr.monitoring.create_sentry_integration({
        'name': 'Sentry Error Tracking',
        'dsn': 'https://xxx@sentry.io/yyy',
        'environment': 'production',
        'release': 'v1.2.3'
    })
    print(f"✓ Created Sentry integration: {sentry_int['name']}")
    print(f"  Environment: {sentry_int['environment']}")

    print("\n🔐 6. OAuth Applications")
    print("-" * 80)

    # Create OAuth app
    oauth_app = mgr.oauth.create_oauth_application({
        'name': 'Mobile App Integration',
        'redirect_uri': 'https://myapp.com/oauth/callback',
        'scopes': ['api', 'read_user', 'write_repository'],
        'confidential': True
    })
    print(f"✓ Created OAuth application: {oauth_app['name']}")
    print(f"  Client ID: {oauth_app['client_id']}")
    print(f"  Scopes: {', '.join(oauth_app['scopes'])}")

    print("\n💚 7. Integration Health Monitoring")
    print("-" * 80)

    # Check health of integrations
    slack_health = mgr.health.check_integration_health(slack_int['integration_id'], 'slack')
    print(f"✓ Slack integration health: {slack_health['status']} ({slack_health['response_time_ms']}ms)")

    jira_health = mgr.health.check_integration_health(jira_int['integration_id'], 'jira')
    print(f"✓ Jira integration health: {jira_health['status']} ({jira_health['response_time_ms']}ms)")

    jenkins_health = mgr.health.check_integration_health(jenkins_int['integration_id'], 'jenkins')
    print(f"✓ Jenkins integration health: {jenkins_health['status']} ({jenkins_health['response_time_ms']}ms)")

    unhealthy = mgr.health.get_unhealthy_integrations()
    print(f"✓ Unhealthy integrations: {len(unhealthy)}")

    print("\n📝 8. Integration Templates")
    print("-" * 80)

    # Create templates
    slack_template = mgr.templates.create_template({
        'name': 'Standard Slack Notifications',
        'integration_type': IntegrationType.SLACK.value,
        'description': 'Standard Slack notification setup for teams',
        'default_config': {
            'username': 'GitLab Bot',
            'events': ['push', 'merge_request', 'pipeline']
        },
        'required_fields': ['webhook_url', 'channel']
    })
    print(f"✓ Created template: {slack_template['name']}")

    webhook_template = mgr.templates.create_template({
        'name': 'Deployment Webhook',
        'integration_type': IntegrationType.WEBHOOK.value,
        'description': 'Webhook for deployment notifications',
        'default_config': {
            'events': ['pipeline'],
            'enable_ssl_verification': True
        },
        'required_fields': ['url']
    })
    print(f"✓ Created template: {webhook_template['name']}")

    # List templates
    all_templates = mgr.templates.list_templates()
    print(f"✓ Total templates available: {len(all_templates)}")

    print("\n📈 9. Summary Statistics")
    print("-" * 80)

    info = mgr.info()
    counts = info['integration_count']

    print(f"Total integrations: {counts['total']}")
    print(f"  - Slack: {counts['slack']}")
    print(f"  - Jira: {counts['jira']}")
    print(f"  - Jenkins: {counts['jenkins']}")
    print(f"  - Webhooks: {counts['webhooks']}")
    print(f"  - Monitoring: {counts['monitoring']}")
    print(f"  - OAuth Apps: {counts['oauth_apps']}")
    print(f"Health monitored: {info['health_monitored']} integrations")
    print(f"Templates available: {info['templates_available']}")

    print("\n✅ Demo completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    demo()
