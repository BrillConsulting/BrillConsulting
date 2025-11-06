"""
Google Cloud Build - Advanced CI/CD Platform
==============================================

Comprehensive Cloud Build implementation with:
- Advanced build configuration (multi-step, parallel steps)
- Build triggers (GitHub, Cloud Source Repositories, webhooks)
- Artifact management
- Secret management integration
- Build notifications (Pub/Sub, Slack)
- Build history and analytics
- Build machine types and timeout control
- Substitution variables

Author: Brill Consulting
"""

from typing import List, Dict, Optional, Any
from datetime import datetime
import json


class BuildManager:
    """Manages Cloud Build builds."""

    def __init__(self, project_id: str):
        self.project_id = project_id
        self.builds = {}

    def create_build(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create and submit build.

        Config:
        - steps: List of build steps
        - source: Source location (gs://bucket/object or Cloud Source Repository)
        - images: List of images to push to Container Registry
        - artifacts: Artifacts configuration
        - substitutions: Substitution variables (_VAR or $VAR)
        - timeout: Build timeout (default: 600s, max: 86400s/24h)
        - machine_type: E2_MEDIUM, E2_HIGHCPU_8, E2_HIGHCPU_32, E2_HIGHMEM_8
        - service_account: Service account for build
        - log_streaming_option: STREAM_DEFAULT, STREAM_ON, STREAM_OFF
        """
        build_id = f"build-{int(datetime.now().timestamp())}"
        steps = config.get('steps', [])
        images = config.get('images', [])
        machine_type = config.get('machine_type', 'E2_MEDIUM')
        timeout = config.get('timeout', '600s')

        print(f"\n🔨 Creating build: {build_id}")
        print(f"   Project: {self.project_id}")
        print(f"   Steps: {len(steps)}")
        print(f"   Machine Type: {machine_type}")
        print(f"   Timeout: {timeout}")
        if images:
            print(f"   Images: {len(images)}")

        build = {
            "id": build_id,
            "project_id": self.project_id,
            "status": "QUEUED",
            "source": config.get('source'),
            "steps": steps,
            "images": images,
            "artifacts": config.get('artifacts', {}),
            "substitutions": config.get('substitutions', {}),
            "machine_type": machine_type,
            "timeout": timeout,
            "service_account": config.get('service_account'),
            "log_streaming_option": config.get('log_streaming_option', 'STREAM_DEFAULT'),
            "create_time": datetime.now().isoformat(),
            "start_time": None,
            "finish_time": None,
            "duration_seconds": 0,
            "logs_url": f"https://console.cloud.google.com/cloud-build/builds/{build_id}"
        }

        self.builds[build_id] = build

        print(f"✓ Build created")

        return build

    def start_build(self, build_id: str) -> Dict[str, Any]:
        """Start build execution."""
        if build_id not in self.builds:
            return {"error": f"Build {build_id} not found"}

        build = self.builds[build_id]

        print(f"\n▶️  Starting build: {build_id}")

        build['status'] = 'WORKING'
        build['start_time'] = datetime.now().isoformat()

        # Simulate step execution
        for i, step in enumerate(build['steps'], 1):
            step_name = step.get('name', 'unknown')
            print(f"   Step {i}/{len(build['steps'])}: {step_name}")
            step['status'] = 'SUCCESS'

        # Complete build
        build['finish_time'] = datetime.now().isoformat()
        build['status'] = 'SUCCESS'
        build['duration_seconds'] = len(build['steps']) * 5  # 5s per step

        print(f"✓ Build completed: {build['status']}")
        print(f"   Duration: {build['duration_seconds']}s")

        return build

    def cancel_build(self, build_id: str) -> Dict[str, str]:
        """Cancel running build."""
        if build_id not in self.builds:
            return {"error": f"Build {build_id} not found"}

        build = self.builds[build_id]

        print(f"\n🛑 Cancelling build: {build_id}")

        build['status'] = 'CANCELLED'
        build['finish_time'] = datetime.now().isoformat()

        print(f"✓ Build cancelled")

        return {"status": "cancelled", "build_id": build_id}

    def get_build(self, build_id: str) -> Optional[Dict[str, Any]]:
        """Get build details."""
        return self.builds.get(build_id)

    def list_builds(self, filter_status: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """List builds with optional status filter."""
        builds = list(self.builds.values())

        if filter_status:
            builds = [b for b in builds if b['status'] == filter_status]

        return builds[-limit:]

    def generate_build_config(self, config: Dict[str, Any]) -> str:
        """Generate cloudbuild.yaml from configuration."""
        yaml = "# Cloud Build Configuration\n\n"
        yaml += "steps:\n"

        for step in config.get('steps', []):
            yaml += f"- name: '{step['name']}'\n"
            if 'id' in step:
                yaml += f"  id: '{step['id']}'\n"
            if 'args' in step:
                yaml += "  args:\n"
                for arg in step['args']:
                    yaml += f"    - '{arg}'\n"
            if 'env' in step:
                yaml += "  env:\n"
                for env in step['env']:
                    yaml += f"    - '{env}'\n"
            if 'waitFor' in step:
                yaml += "  waitFor:\n"
                for wait in step['waitFor']:
                    yaml += f"    - '{wait}'\n"
            yaml += "\n"

        if config.get('images'):
            yaml += "images:\n"
            for image in config['images']:
                yaml += f"  - '{image}'\n"
            yaml += "\n"

        if config.get('timeout'):
            yaml += f"timeout: '{config['timeout']}'\n"

        if config.get('machine_type'):
            yaml += f"\noptions:\n  machineType: {config['machine_type']}\n"

        return yaml


class TriggerManager:
    """Manages Cloud Build triggers."""

    def __init__(self, project_id: str):
        self.project_id = project_id
        self.triggers = {}

    def create_github_trigger(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create GitHub trigger.

        Config:
        - name: Trigger name
        - description: Trigger description
        - repo_owner: GitHub repository owner
        - repo_name: GitHub repository name
        - branch_pattern: Branch regex (e.g., ^main$, ^release-.*$)
        - tag_pattern: Tag regex (e.g., ^v[0-9]+\.[0-9]+\.[0-9]+$)
        - pull_request: Pull request configuration
        - filename: cloudbuild.yaml path (default: cloudbuild.yaml)
        - substitutions: Trigger-level substitutions
        - included_files: List of file patterns to include
        - ignored_files: List of file patterns to ignore
        """
        name = config.get('name')
        repo_owner = config.get('repo_owner')
        repo_name = config.get('repo_name')

        print(f"\n⚡ Creating GitHub trigger: {name}")
        print(f"   Repository: {repo_owner}/{repo_name}")

        if config.get('branch_pattern'):
            print(f"   Branch Pattern: {config['branch_pattern']}")
        if config.get('tag_pattern'):
            print(f"   Tag Pattern: {config['tag_pattern']}")

        trigger_id = f"trigger-{int(datetime.now().timestamp())}"

        trigger = {
            "id": trigger_id,
            "name": name,
            "description": config.get('description', ''),
            "type": "GITHUB",
            "repo_owner": repo_owner,
            "repo_name": repo_name,
            "branch_pattern": config.get('branch_pattern'),
            "tag_pattern": config.get('tag_pattern'),
            "pull_request": config.get('pull_request'),
            "filename": config.get('filename', 'cloudbuild.yaml'),
            "substitutions": config.get('substitutions', {}),
            "included_files": config.get('included_files', []),
            "ignored_files": config.get('ignored_files', []),
            "disabled": False,
            "created_at": datetime.now().isoformat()
        }

        self.triggers[trigger_id] = trigger

        print(f"✓ Trigger created: {trigger_id}")

        return trigger

    def create_webhook_trigger(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create webhook trigger.

        Config:
        - name: Trigger name
        - secret: Webhook secret
        - substitutions: Trigger-level substitutions
        """
        name = config.get('name')

        print(f"\n🪝 Creating webhook trigger: {name}")

        trigger_id = f"trigger-{int(datetime.now().timestamp())}"
        webhook_url = f"https://cloudbuild.googleapis.com/v1/projects/{self.project_id}/triggers/{trigger_id}:webhook"

        trigger = {
            "id": trigger_id,
            "name": name,
            "type": "WEBHOOK",
            "webhook_url": webhook_url,
            "secret": config.get('secret'),
            "substitutions": config.get('substitutions', {}),
            "disabled": False,
            "created_at": datetime.now().isoformat()
        }

        self.triggers[trigger_id] = trigger

        print(f"✓ Webhook trigger created")
        print(f"   URL: {webhook_url}")

        return trigger

    def create_cloud_source_trigger(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create Cloud Source Repositories trigger.

        Config:
        - name: Trigger name
        - repo_name: Cloud Source Repository name
        - branch_pattern: Branch regex pattern
        """
        name = config.get('name')
        repo_name = config.get('repo_name')

        print(f"\n☁️  Creating Cloud Source trigger: {name}")
        print(f"   Repository: {repo_name}")

        trigger_id = f"trigger-{int(datetime.now().timestamp())}"

        trigger = {
            "id": trigger_id,
            "name": name,
            "type": "CLOUD_SOURCE",
            "repo_name": repo_name,
            "branch_pattern": config.get('branch_pattern', '^main$'),
            "disabled": False,
            "created_at": datetime.now().isoformat()
        }

        self.triggers[trigger_id] = trigger

        print(f"✓ Trigger created: {trigger_id}")

        return trigger

    def enable_trigger(self, trigger_id: str) -> Dict[str, str]:
        """Enable trigger."""
        if trigger_id not in self.triggers:
            return {"error": f"Trigger {trigger_id} not found"}

        self.triggers[trigger_id]['disabled'] = False

        print(f"✓ Trigger enabled: {trigger_id}")

        return {"status": "enabled", "trigger_id": trigger_id}

    def disable_trigger(self, trigger_id: str) -> Dict[str, str]:
        """Disable trigger."""
        if trigger_id not in self.triggers:
            return {"error": f"Trigger {trigger_id} not found"}

        self.triggers[trigger_id]['disabled'] = True

        print(f"✓ Trigger disabled: {trigger_id}")

        return {"status": "disabled", "trigger_id": trigger_id}

    def list_triggers(self) -> List[Dict[str, Any]]:
        """List all triggers."""
        return list(self.triggers.values())


class ArtifactManager:
    """Manages build artifacts."""

    def __init__(self, project_id: str):
        self.project_id = project_id
        self.artifacts = {}

    def configure_storage_artifacts(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Configure Cloud Storage artifacts.

        Config:
        - location: GCS bucket location (gs://bucket/path)
        - paths: List of paths to upload
        """
        location = config.get('location')

        print(f"\n📦 Configuring Storage artifacts")
        print(f"   Location: {location}")
        print(f"   Paths: {len(config.get('paths', []))} files")

        artifact_config = {
            "type": "STORAGE",
            "location": location,
            "paths": config.get('paths', []),
            "created_at": datetime.now().isoformat()
        }

        print(f"✓ Artifact configuration created")

        return artifact_config

    def configure_maven_artifacts(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Configure Maven artifacts.

        Config:
        - repository: Artifact Registry repository
        - group_id: Maven group ID
        - artifact_id: Maven artifact ID
        - version: Artifact version
        """
        repository = config.get('repository')

        print(f"\n📦 Configuring Maven artifacts")
        print(f"   Repository: {repository}")
        print(f"   Artifact: {config.get('group_id')}:{config.get('artifact_id')}:{config.get('version')}")

        artifact_config = {
            "type": "MAVEN",
            "repository": repository,
            "group_id": config.get('group_id'),
            "artifact_id": config.get('artifact_id'),
            "version": config.get('version'),
            "created_at": datetime.now().isoformat()
        }

        print(f"✓ Maven artifact configuration created")

        return artifact_config

    def configure_npm_artifacts(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Configure npm artifacts.

        Config:
        - repository: Artifact Registry repository
        - package_name: npm package name
        """
        repository = config.get('repository')
        package_name = config.get('package_name')

        print(f"\n📦 Configuring npm artifacts")
        print(f"   Repository: {repository}")
        print(f"   Package: {package_name}")

        artifact_config = {
            "type": "NPM",
            "repository": repository,
            "package_name": package_name,
            "created_at": datetime.now().isoformat()
        }

        print(f"✓ npm artifact configuration created")

        return artifact_config


class NotificationManager:
    """Manages build notifications."""

    def __init__(self, project_id: str):
        self.project_id = project_id
        self.notifications = {}

    def create_pubsub_notification(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create Pub/Sub notification.

        Config:
        - topic: Pub/Sub topic (projects/PROJECT/topics/TOPIC)
        - filter: Build status filter (SUCCESS, FAILURE, TIMEOUT, CANCELLED)
        """
        topic = config.get('topic')
        filter_status = config.get('filter', 'ALL')

        print(f"\n📨 Creating Pub/Sub notification")
        print(f"   Topic: {topic}")
        print(f"   Filter: {filter_status}")

        notification_id = f"notification-{int(datetime.now().timestamp())}"

        notification = {
            "id": notification_id,
            "type": "PUBSUB",
            "topic": topic,
            "filter": filter_status,
            "created_at": datetime.now().isoformat()
        }

        self.notifications[notification_id] = notification

        print(f"✓ Notification created: {notification_id}")

        return notification

    def create_slack_notification(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create Slack notification.

        Config:
        - webhook_url: Slack webhook URL
        - channel: Slack channel
        - filter: Build status filter
        """
        channel = config.get('channel')

        print(f"\n💬 Creating Slack notification")
        print(f"   Channel: {channel}")

        notification_id = f"notification-{int(datetime.now().timestamp())}"

        notification = {
            "id": notification_id,
            "type": "SLACK",
            "webhook_url": config.get('webhook_url'),
            "channel": channel,
            "filter": config.get('filter', 'ALL'),
            "created_at": datetime.now().isoformat()
        }

        self.notifications[notification_id] = notification

        print(f"✓ Slack notification created")

        return notification


class BuildHistoryManager:
    """Manages build history and analytics."""

    def __init__(self, project_id: str):
        self.project_id = project_id

    def get_build_stats(self, builds: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get build statistics."""
        if not builds:
            return {"error": "No builds found"}

        total_builds = len(builds)
        successful = sum(1 for b in builds if b.get('status') == 'SUCCESS')
        failed = sum(1 for b in builds if b.get('status') == 'FAILURE')
        cancelled = sum(1 for b in builds if b.get('status') == 'CANCELLED')
        timeout = sum(1 for b in builds if b.get('status') == 'TIMEOUT')

        durations = [b['duration_seconds'] for b in builds if 'duration_seconds' in b]
        avg_duration = sum(durations) / len(durations) if durations else 0

        stats = {
            "total_builds": total_builds,
            "successful": successful,
            "failed": failed,
            "cancelled": cancelled,
            "timeout": timeout,
            "success_rate": (successful / total_builds * 100) if total_builds > 0 else 0,
            "avg_duration_seconds": round(avg_duration, 2)
        }

        print(f"\n📊 Build Statistics")
        print(f"   Total Builds: {stats['total_builds']}")
        print(f"   Success Rate: {stats['success_rate']:.1f}%")
        print(f"   Avg Duration: {stats['avg_duration_seconds']}s")

        return stats

    def query_builds(self, builds: List[Dict[str, Any]], filter_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Query builds with filters."""
        filtered = builds

        if 'status' in filter_config:
            filtered = [b for b in filtered if b.get('status') == filter_config['status']]

        if 'min_duration' in filter_config:
            filtered = [b for b in filtered if b.get('duration_seconds', 0) >= filter_config['min_duration']]

        if 'max_duration' in filter_config:
            filtered = [b for b in filtered if b.get('duration_seconds', 0) <= filter_config['max_duration']]

        print(f"\n🔍 Query Results: {len(filtered)} builds found")

        return filtered


class CloudBuildManager:
    """Main Cloud Build manager integrating all components."""

    def __init__(self, project_id: str):
        self.project_id = project_id
        self.builds = BuildManager(project_id)
        self.triggers = TriggerManager(project_id)
        self.artifacts = ArtifactManager(project_id)
        self.notifications = NotificationManager(project_id)
        self.history = BuildHistoryManager(project_id)

    def info(self) -> Dict[str, Any]:
        """Get Cloud Build information."""
        return {
            "project_id": self.project_id,
            "total_builds": len(self.builds.builds),
            "total_triggers": len(self.triggers.triggers),
            "total_notifications": len(self.notifications.notifications)
        }


def demo():
    """Demo Cloud Build with advanced features."""
    print("=" * 70)
    print("Google Cloud Build - Advanced Demo")
    print("=" * 70)

    mgr = CloudBuildManager("my-gcp-project")

    # 1. Create advanced build
    print("\n1. Create Advanced Multi-Step Build")
    print("-" * 70)

    build_config = {
        "source": "gs://my-bucket/source.tar.gz",
        "steps": [
            {
                "id": "test",
                "name": "gcr.io/cloud-builders/docker",
                "args": ["run", "--rm", "python:3.11", "pytest", "tests/"],
                "env": ["ENV=test"]
            },
            {
                "id": "build",
                "name": "gcr.io/cloud-builders/docker",
                "args": ["build", "-t", "gcr.io/my-gcp-project/myapp:$COMMIT_SHA", "."],
                "waitFor": ["test"]
            },
            {
                "id": "push",
                "name": "gcr.io/cloud-builders/docker",
                "args": ["push", "gcr.io/my-gcp-project/myapp:$COMMIT_SHA"],
                "waitFor": ["build"]
            },
            {
                "id": "deploy",
                "name": "gcr.io/cloud-builders/gcloud",
                "args": ["run", "deploy", "myapp", "--image", "gcr.io/my-gcp-project/myapp:$COMMIT_SHA", "--region", "us-central1"],
                "waitFor": ["push"]
            }
        ],
        "images": ["gcr.io/my-gcp-project/myapp:$COMMIT_SHA"],
        "substitutions": {
            "_ENV": "production",
            "_REGION": "us-central1"
        },
        "machine_type": "E2_HIGHCPU_8",
        "timeout": "1200s"
    }

    build = mgr.builds.create_build(build_config)
    build_result = mgr.builds.start_build(build['id'])

    # Generate YAML
    yaml_config = mgr.builds.generate_build_config(build_config)
    print(f"\nGenerated cloudbuild.yaml:\n{yaml_config[:400]}...")

    # 2. GitHub triggers
    print("\n2. Create GitHub Triggers")
    print("-" * 70)

    # Production trigger
    prod_trigger = mgr.triggers.create_github_trigger({
        "name": "deploy-production",
        "description": "Deploy to production on main branch",
        "repo_owner": "myorg",
        "repo_name": "myapp",
        "branch_pattern": "^main$",
        "filename": "cloudbuild.yaml",
        "substitutions": {"_ENV": "production"},
        "ignored_files": ["README.md", "docs/**"]
    })

    # Staging trigger
    staging_trigger = mgr.triggers.create_github_trigger({
        "name": "deploy-staging",
        "repo_owner": "myorg",
        "repo_name": "myapp",
        "branch_pattern": "^staging$",
        "substitutions": {"_ENV": "staging"}
    })

    # Tag trigger for releases
    release_trigger = mgr.triggers.create_github_trigger({
        "name": "build-release",
        "repo_owner": "myorg",
        "repo_name": "myapp",
        "tag_pattern": "^v[0-9]+\\.[0-9]+\\.[0-9]+$",
        "substitutions": {"_RELEASE": "true"}
    })

    # 3. Webhook trigger
    print("\n3. Create Webhook Trigger")
    print("-" * 70)

    webhook_trigger = mgr.triggers.create_webhook_trigger({
        "name": "manual-deploy",
        "secret": "my-webhook-secret",
        "substitutions": {"_TRIGGER_TYPE": "webhook"}
    })

    # 4. Artifact configuration
    print("\n4. Configure Artifacts")
    print("-" * 70)

    storage_artifacts = mgr.artifacts.configure_storage_artifacts({
        "location": "gs://my-build-artifacts/",
        "paths": ["dist/**", "build/**", "*.zip"]
    })

    maven_artifacts = mgr.artifacts.configure_maven_artifacts({
        "repository": "us-central1-maven.pkg.dev/my-project/maven-repo",
        "group_id": "com.example",
        "artifact_id": "myapp",
        "version": "1.0.0"
    })

    # 5. Build notifications
    print("\n5. Configure Notifications")
    print("-" * 70)

    pubsub_notification = mgr.notifications.create_pubsub_notification({
        "topic": f"projects/{mgr.project_id}/topics/build-notifications",
        "filter": "FAILURE"
    })

    slack_notification = mgr.notifications.create_slack_notification({
        "webhook_url": "https://hooks.slack.com/services/XXX/YYY/ZZZ",
        "channel": "#deployments",
        "filter": "ALL"
    })

    # 6. Create multiple builds for analytics
    print("\n6. Build History and Analytics")
    print("-" * 70)

    # Simulate multiple builds
    for i in range(10):
        test_build = mgr.builds.create_build({
            "steps": [{"name": "gcr.io/cloud-builders/docker", "args": ["build", "."]}],
            "machine_type": "E2_MEDIUM",
            "timeout": "600s"
        })
        mgr.builds.start_build(test_build['id'])

    # Get statistics
    all_builds = mgr.builds.list_builds(limit=100)
    stats = mgr.history.get_build_stats(all_builds)

    # Query builds
    successful_builds = mgr.history.query_builds(all_builds, {"status": "SUCCESS"})

    # 7. List all triggers
    print("\n7. Build Triggers Summary")
    print("-" * 70)

    triggers = mgr.triggers.list_triggers()
    for trigger in triggers:
        print(f"\n  📍 {trigger['name']}")
        print(f"     Type: {trigger['type']}")
        print(f"     Disabled: {trigger['disabled']}")

    # 8. Manager info
    print("\n8. Cloud Build Summary")
    print("-" * 70)

    info = mgr.info()
    print(f"\n  Project: {info['project_id']}")
    print(f"  Total Builds: {info['total_builds']}")
    print(f"  Total Triggers: {info['total_triggers']}")
    print(f"  Total Notifications: {info['total_notifications']}")

    print("\n" + "=" * 70)
    print("✓ Cloud Build Advanced Demo Complete!")
    print("=" * 70)


if __name__ == '__main__':
    demo()
