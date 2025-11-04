# Azure DevOps & CI/CD

Azure DevOps pipelines and automation.

## Features

- Build pipeline definition and execution
- Release pipeline with multiple environments
- YAML pipeline generation
- Azure Artifacts management
- Infrastructure deployment pipelines
- Multi-stage pipelines
- Environment approvals

## Usage

```python
from azure_devops import AzureDevOpsProject, BuildPipeline

# Create DevOps project
project = AzureDevOpsProject("MyOrganization", "MyProject")

# Create build pipeline
build_pipeline = project.create_build_pipeline("CI-Build", language="python")
build_pipeline.add_trigger("push", ["main", "develop"])

# Generate YAML
yaml = build_pipeline.generate_yaml()

# Run pipeline
result = build_pipeline.run(trigger_reason="push:main")

# Create release pipeline
release = project.create_release_pipeline("CD-Release", ["dev", "staging", "production"])
release.run()
```

## Demo

```bash
python azure_devops.py
```
