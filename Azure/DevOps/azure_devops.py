"""
Azure DevOps & CI/CD
====================

Azure DevOps pipelines and automation:
- Pipeline definition (YAML)
- Build pipelines
- Release pipelines
- Artifacts management
- Infrastructure deployment

Author: Brill Consulting
"""

from typing import List, Dict, Optional
from datetime import datetime
import json


class AzurePipeline:
    """Azure DevOps Pipeline."""

    def __init__(self, name: str, pipeline_type: str = "build"):
        self.name = name
        self.type = pipeline_type
        self.stages = []
        self.variables = {}
        self.triggers = []
        self.runs = []

    def add_stage(self, stage_name: str, jobs: List[Dict]):
        """Add stage to pipeline."""
        self.stages.append({
            "stage": stage_name,
            "jobs": jobs
        })
        print(f"✓ Stage added: {stage_name} ({len(jobs)} jobs)")

    def add_variable(self, name: str, value: str):
        """Add pipeline variable."""
        self.variables[name] = value

    def add_trigger(self, trigger_type: str, branches: List[str]):
        """Add pipeline trigger."""
        self.triggers.append({
            "type": trigger_type,
            "branches": branches
        })

    def run(self, trigger_reason: str = "manual") -> Dict:
        """Execute pipeline run."""
        print(f"\n🚀 Starting pipeline: {self.name}")
        print(f"   Type: {self.type}")
        print(f"   Trigger: {trigger_reason}")

        run_id = f"run_{len(self.runs) + 1}"
        start_time = datetime.now()

        results = []
        for stage in self.stages:
            print(f"\n📦 Stage: {stage['stage']}")

            for job in stage["jobs"]:
                job_name = job.get("job")
                steps = job.get("steps", [])

                print(f"   Job: {job_name}")

                for step in steps:
                    step_type = list(step.keys())[0]
                    step_name = step[step_type]
                    print(f"     • {step_type}: {step_name}")

                results.append({
                    "stage": stage["stage"],
                    "job": job_name,
                    "status": "succeeded",
                    "duration": 5.2
                })

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        run_result = {
            "run_id": run_id,
            "pipeline": self.name,
            "status": "succeeded",
            "trigger": trigger_reason,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration": duration,
            "results": results
        }

        self.runs.append(run_result)

        print(f"\n✓ Pipeline completed in {duration:.2f}s")
        return run_result

    def generate_yaml(self) -> str:
        """Generate YAML pipeline definition."""
        yaml = f"# Azure DevOps Pipeline: {self.name}\n"
        yaml += f"name: {self.name}\n\n"

        # Triggers
        if self.triggers:
            yaml += "trigger:\n"
            for trigger in self.triggers:
                yaml += f"  branches:\n"
                yaml += f"    include:\n"
                for branch in trigger["branches"]:
                    yaml += f"      - {branch}\n"
            yaml += "\n"

        # Variables
        if self.variables:
            yaml += "variables:\n"
            for key, value in self.variables.items():
                yaml += f"  {key}: {value}\n"
            yaml += "\n"

        # Stages
        yaml += "stages:\n"
        for stage in self.stages:
            yaml += f"- stage: {stage['stage']}\n"
            yaml += f"  jobs:\n"
            for job in stage["jobs"]:
                yaml += f"  - job: {job.get('job')}\n"
                yaml += f"    pool:\n"
                yaml += f"      vmImage: 'ubuntu-latest'\n"
                yaml += f"    steps:\n"
                for step in job.get("steps", []):
                    step_type = list(step.keys())[0]
                    step_value = step[step_type]
                    yaml += f"    - {step_type}: {step_value}\n"
            yaml += "\n"

        return yaml


class BuildPipeline(AzurePipeline):
    """Build pipeline."""

    def __init__(self, name: str):
        super().__init__(name, "build")

    def add_build_stage(self, language: str = "python"):
        """Add standard build stage."""
        if language == "python":
            jobs = [{
                "job": "Build",
                "steps": [
                    {"script": "pip install -r requirements.txt"},
                    {"script": "pytest tests/"},
                    {"script": "python -m build"},
                    {"task": "PublishBuildArtifacts@1"}
                ]
            }]
        elif language == "node":
            jobs = [{
                "job": "Build",
                "steps": [
                    {"script": "npm install"},
                    {"script": "npm test"},
                    {"script": "npm run build"},
                    {"task": "PublishBuildArtifacts@1"}
                ]
            }]
        else:
            jobs = []

        self.add_stage("Build", jobs)


class ReleasePipeline(AzurePipeline):
    """Release pipeline."""

    def __init__(self, name: str):
        super().__init__(name, "release")
        self.environments = []

    def add_environment(self, env_name: str, approval_required: bool = False):
        """Add deployment environment."""
        self.environments.append({
            "name": env_name,
            "approval_required": approval_required
        })
        print(f"✓ Environment added: {env_name}")

    def add_deployment_stage(self, env_name: str, deployment_type: str = "azure_app_service"):
        """Add deployment stage."""
        jobs = [{
            "job": f"Deploy_{env_name}",
            "steps": [
                {"task": "DownloadBuildArtifacts@0"},
                {"task": "AzureWebApp@1" if deployment_type == "azure_app_service" else "AzureCLI@2"},
                {"script": "echo 'Deployment complete'"}
            ]
        }]

        self.add_stage(f"Deploy_{env_name}", jobs)


class AzureArtifacts:
    """Azure Artifacts management."""

    def __init__(self, organization: str, project: str):
        self.organization = organization
        self.project = project
        self.feeds = {}

    def create_feed(self, feed_name: str, feed_type: str = "PyPI") -> Dict:
        """Create artifact feed."""
        print(f"\n📦 Creating feed: {feed_name}")

        self.feeds[feed_name] = {
            "name": feed_name,
            "type": feed_type,
            "packages": [],
            "created_at": datetime.now().isoformat()
        }

        print(f"✓ Feed created: {feed_name} ({feed_type})")
        return self.feeds[feed_name]

    def publish_package(self, feed_name: str, package_name: str, version: str) -> Dict:
        """Publish package to feed."""
        if feed_name not in self.feeds:
            return {"error": f"Feed {feed_name} not found"}

        package = {
            "name": package_name,
            "version": version,
            "published_at": datetime.now().isoformat()
        }

        self.feeds[feed_name]["packages"].append(package)
        print(f"✓ Published: {package_name} v{version} to {feed_name}")

        return package


class AzureDevOpsProject:
    """Azure DevOps project."""

    def __init__(self, organization: str, project_name: str):
        self.organization = organization
        self.project_name = project_name
        self.pipelines = {}
        self.artifacts = AzureArtifacts(organization, project_name)

    def create_build_pipeline(self, name: str, language: str = "python") -> BuildPipeline:
        """Create build pipeline."""
        pipeline = BuildPipeline(name)
        pipeline.add_build_stage(language)
        self.pipelines[name] = pipeline

        print(f"\n✓ Build pipeline created: {name}")
        return pipeline

    def create_release_pipeline(self, name: str, environments: List[str]) -> ReleasePipeline:
        """Create release pipeline."""
        pipeline = ReleasePipeline(name)

        for env in environments:
            pipeline.add_environment(env, approval_required=(env == "production"))
            pipeline.add_deployment_stage(env)

        self.pipelines[name] = pipeline

        print(f"\n✓ Release pipeline created: {name}")
        return pipeline

    def get_project_summary(self) -> Dict:
        """Get project summary."""
        return {
            "organization": self.organization,
            "project": self.project_name,
            "pipelines": len(self.pipelines),
            "artifact_feeds": len(self.artifacts.feeds)
        }


def demo():
    """Demo Azure DevOps."""
    print("Azure DevOps & CI/CD Demo")
    print("=" * 60)

    # 1. Create DevOps Project
    print("\n1. Create DevOps Project")
    print("-" * 60)

    project = AzureDevOpsProject("MyOrganization", "MyProject")
    print(f"✓ Project created: {project.project_name}")

    # 2. Create Build Pipeline
    print("\n2. Build Pipeline")
    print("-" * 60)

    build_pipeline = project.create_build_pipeline("CI-Build", language="python")

    build_pipeline.add_variable("pythonVersion", "3.11")
    build_pipeline.add_trigger("push", ["main", "develop"])

    # Add additional stages
    build_pipeline.add_stage("Test", [{
        "job": "UnitTests",
        "steps": [
            {"script": "pytest tests/unit"},
            {"task": "PublishTestResults@2"}
        ]
    }])

    build_pipeline.add_stage("CodeQuality", [{
        "job": "Lint",
        "steps": [
            {"script": "flake8 ."},
            {"script": "black --check ."},
            {"script": "mypy ."}
        ]
    }])

    # Generate YAML
    yaml_content = build_pipeline.generate_yaml()
    print("\nGenerated Pipeline YAML:")
    print("-" * 60)
    print(yaml_content[:300] + "...")

    # Run pipeline
    build_result = build_pipeline.run(trigger_reason="push:main")

    # 3. Create Release Pipeline
    print("\n3. Release Pipeline")
    print("-" * 60)

    release_pipeline = project.create_release_pipeline(
        "CD-Release",
        environments=["dev", "staging", "production"]
    )

    release_pipeline.add_variable("appServiceName", "my-web-app")

    # Run release
    release_result = release_pipeline.run(trigger_reason="build_completed")

    # 4. Azure Artifacts
    print("\n4. Azure Artifacts")
    print("-" * 60)

    project.artifacts.create_feed("my-python-packages", "PyPI")
    project.artifacts.publish_package("my-python-packages", "my-lib", "1.0.0")
    project.artifacts.publish_package("my-python-packages", "my-lib", "1.0.1")

    # 5. Infrastructure as Code Pipeline
    print("\n5. Infrastructure Pipeline")
    print("-" * 60)

    infra_pipeline = AzurePipeline("Infrastructure-Deploy", "release")

    infra_pipeline.add_stage("ValidateInfra", [{
        "job": "Validate",
        "steps": [
            {"task": "AzureCLI@2"},
            {"script": "terraform init"},
            {"script": "terraform validate"},
            {"script": "terraform plan"}
        ]
    }])

    infra_pipeline.add_stage("DeployInfra", [{
        "job": "Deploy",
        "steps": [
            {"script": "terraform apply -auto-approve"},
            {"task": "PublishPipelineArtifact@1"}
        ]
    }])

    project.pipelines["infra-deploy"] = infra_pipeline
    infra_result = infra_pipeline.run()

    # 6. Project Summary
    print("\n6. Project Summary")
    print("-" * 60)

    summary = project.get_project_summary()
    print(f"  Organization: {summary['organization']}")
    print(f"  Project: {summary['project']}")
    print(f"  Pipelines: {summary['pipelines']}")
    print(f"  Artifact Feeds: {summary['artifact_feeds']}")

    print(f"\n  Pipelines:")
    for pipeline_name, pipeline in project.pipelines.items():
        runs = len(pipeline.runs)
        print(f"    • {pipeline_name} ({pipeline.type}) - {runs} runs")

    print(f"\n  Artifact Feeds:")
    for feed_name, feed in project.artifacts.feeds.items():
        packages = len(feed["packages"])
        print(f"    • {feed_name} ({feed['type']}) - {packages} packages")

    print("\n✓ Azure DevOps Demo Complete!")


if __name__ == '__main__':
    demo()
