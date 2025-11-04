"""
Google Cloud Build
==================

CI/CD with Cloud Build:
- Build configuration
- Build triggers
- Container builds
- Multi-step builds
- Build history

Author: Brill Consulting
"""

from typing import List, Dict, Optional
from datetime import datetime
import json


class BuildStep:
    """Cloud Build step."""

    def __init__(self, name: str, args: List[str], env: Optional[List[str]] = None):
        self.name = name
        self.args = args
        self.env = env or []

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "args": self.args,
            "env": self.env
        }


class CloudBuildConfig:
    """Cloud Build configuration."""

    def __init__(self):
        self.steps = []
        self.images = []
        self.substitutions = {}
        self.timeout = "600s"

    def add_step(self, step: BuildStep):
        """Add build step."""
        self.steps.append(step)

    def add_image(self, image: str):
        """Add image to push."""
        self.images.append(image)

    def set_substitutions(self, substitutions: Dict[str, str]):
        """Set substitution variables."""
        self.substitutions = substitutions

    def to_yaml(self) -> str:
        """Generate cloudbuild.yaml."""
        yaml = "# Cloud Build configuration\n\n"
        yaml += "steps:\n"

        for step in self.steps:
            yaml += f"- name: '{step.name}'\n"
            yaml += f"  args:\n"
            for arg in step.args:
                yaml += f"    - '{arg}'\n"
            if step.env:
                yaml += f"  env:\n"
                for env in step.env:
                    yaml += f"    - '{env}'\n"
            yaml += "\n"

        if self.images:
            yaml += "images:\n"
            for image in self.images:
                yaml += f"  - '{image}'\n"
            yaml += "\n"

        yaml += f"timeout: '{self.timeout}'\n"

        return yaml


class CloudBuildTrigger:
    """Cloud Build trigger."""

    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.trigger_type = "github"
        self.branch_pattern = "^main$"
        self.filename = "cloudbuild.yaml"
        self.substitutions = {}

    def configure_github(self, repo_name: str, branch_pattern: str = "^main$"):
        """Configure GitHub trigger."""
        self.trigger_type = "github"
        self.repo_name = repo_name
        self.branch_pattern = branch_pattern

        print(f"✓ GitHub trigger configured: {repo_name} ({branch_pattern})")

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "trigger_type": self.trigger_type,
            "branch_pattern": self.branch_pattern,
            "filename": self.filename
        }


class CloudBuild:
    """Google Cloud Build service."""

    def __init__(self, project_id: str):
        self.project_id = project_id
        self.builds = []
        self.triggers = []

    def create_build_config(self) -> CloudBuildConfig:
        """Create build configuration."""
        return CloudBuildConfig()

    def submit_build(self, config: CloudBuildConfig, source_location: str) -> Dict:
        """Submit build."""
        print(f"\n🔨 Submitting build")
        print(f"   Project: {self.project_id}")
        print(f"   Source: {source_location}")
        print(f"   Steps: {len(config.steps)}")

        build_id = f"build_{datetime.now().timestamp()}"
        start_time = datetime.now()

        # Execute steps
        step_results = []
        for i, step in enumerate(config.steps, 1):
            print(f"\n   Step {i}/{len(config.steps)}: {step.name}")
            print(f"     Args: {' '.join(step.args)}")

            step_results.append({
                "step": i,
                "name": step.name,
                "status": "SUCCESS",
                "duration": 2.5
            })

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        build = {
            "id": build_id,
            "project_id": self.project_id,
            "status": "SUCCESS",
            "source": source_location,
            "steps": step_results,
            "images": config.images,
            "create_time": start_time.isoformat(),
            "finish_time": end_time.isoformat(),
            "duration": duration
        }

        self.builds.append(build)

        print(f"\n✓ Build completed successfully")
        print(f"   Build ID: {build_id}")
        print(f"   Duration: {duration:.2f}s")

        if config.images:
            print(f"   Images pushed: {len(config.images)}")

        return build

    def create_trigger(self, name: str, description: str = "") -> CloudBuildTrigger:
        """Create build trigger."""
        print(f"\n⚡ Creating trigger: {name}")

        trigger = CloudBuildTrigger(name, description)
        self.triggers.append(trigger)

        print(f"✓ Trigger created")

        return trigger

    def list_builds(self, limit: int = 10) -> List[Dict]:
        """List recent builds."""
        return self.builds[-limit:]

    def list_triggers(self) -> List[Dict]:
        """List all triggers."""
        return [trigger.to_dict() for trigger in self.triggers]

    def get_build_stats(self) -> Dict:
        """Get build statistics."""
        total_builds = len(self.builds)
        successful = sum(1 for b in self.builds if b["status"] == "SUCCESS")
        failed = total_builds - successful

        avg_duration = sum(b["duration"] for b in self.builds) / total_builds if total_builds > 0 else 0

        return {
            "total_builds": total_builds,
            "successful": successful,
            "failed": failed,
            "success_rate": successful / total_builds * 100 if total_builds > 0 else 0,
            "avg_duration": avg_duration,
            "total_triggers": len(self.triggers)
        }


def demo():
    """Demo Google Cloud Build."""
    print("Google Cloud Build Demo")
    print("=" * 60)

    cloud_build = CloudBuild("my-gcp-project")

    # 1. Create build configuration
    print("\n1. Create Build Configuration")
    print("-" * 60)

    config = cloud_build.create_build_config()

    # Add build steps
    config.add_step(BuildStep(
        "gcr.io/cloud-builders/docker",
        ["build", "-t", "gcr.io/my-gcp-project/myapp:$COMMIT_SHA", "."]
    ))

    config.add_step(BuildStep(
        "gcr.io/cloud-builders/docker",
        ["push", "gcr.io/my-gcp-project/myapp:$COMMIT_SHA"]
    ))

    config.add_step(BuildStep(
        "gcr.io/cloud-builders/gcloud",
        ["run", "deploy", "myapp", "--image", "gcr.io/my-gcp-project/myapp:$COMMIT_SHA", "--region", "us-central1"]
    ))

    config.add_image("gcr.io/my-gcp-project/myapp:$COMMIT_SHA")
    config.set_substitutions({"_ENV": "production"})

    # Generate YAML
    yaml = config.to_yaml()
    print("\nGenerated cloudbuild.yaml:")
    print("-" * 60)
    print(yaml[:300] + "...")

    # 2. Submit build
    print("\n2. Submit Build")
    print("-" * 60)

    build_result = cloud_build.submit_build(config, "gs://my-bucket/source.tar.gz")

    # 3. Create triggers
    print("\n3. Create Build Triggers")
    print("-" * 60)

    # Main branch trigger
    main_trigger = cloud_build.create_trigger("deploy-production", "Deploy to production on main")
    main_trigger.configure_github("myorg/myrepo", "^main$")

    # Dev branch trigger
    dev_trigger = cloud_build.create_trigger("deploy-dev", "Deploy to dev environment")
    dev_trigger.configure_github("myorg/myrepo", "^dev$")

    # 4. Multi-step build
    print("\n4. Multi-Step Build (Test + Build + Deploy)")
    print("-" * 60)

    multi_config = cloud_build.create_build_config()

    multi_config.add_step(BuildStep(
        "gcr.io/cloud-builders/docker",
        ["run", "--rm", "python:3.11", "pytest", "tests/"]
    ))

    multi_config.add_step(BuildStep(
        "gcr.io/cloud-builders/docker",
        ["build", "-t", "gcr.io/my-gcp-project/myapp:latest", "."]
    ))

    multi_config.add_step(BuildStep(
        "gcr.io/cloud-builders/kubectl",
        ["apply", "-f", "k8s/deployment.yaml"]
    ))

    multi_build = cloud_build.submit_build(multi_config, "gs://my-bucket/source.tar.gz")

    # 5. List builds
    print("\n5. Recent Builds")
    print("-" * 60)

    recent_builds = cloud_build.list_builds(limit=5)
    for build in recent_builds:
        print(f"\n  Build ID: {build['id']}")
        print(f"    Status: {build['status']}")
        print(f"    Duration: {build['duration']:.2f}s")
        print(f"    Steps: {len(build['steps'])}")

    # 6. List triggers
    print("\n6. Build Triggers")
    print("-" * 60)

    triggers = cloud_build.list_triggers()
    for trigger in triggers:
        print(f"\n  {trigger['name']}")
        print(f"    Type: {trigger['trigger_type']}")
        print(f"    Branch: {trigger['branch_pattern']}")

    # 7. Build statistics
    print("\n7. Build Statistics")
    print("-" * 60)

    stats = cloud_build.get_build_stats()
    print(f"  Total builds: {stats['total_builds']}")
    print(f"  Successful: {stats['successful']}")
    print(f"  Failed: {stats['failed']}")
    print(f"  Success rate: {stats['success_rate']:.1f}%")
    print(f"  Avg duration: {stats['avg_duration']:.2f}s")
    print(f"  Total triggers: {stats['total_triggers']}")

    print("\n✓ Google Cloud Build Demo Complete!")


if __name__ == '__main__':
    demo()
