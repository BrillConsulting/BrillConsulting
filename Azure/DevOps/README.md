# Azure DevOps & CI/CD Integration

Comprehensive implementation of Azure DevOps pipelines, automation, and continuous integration/deployment workflows.

**Author:** BrillConsulting
**Contact:** clientbrill@gmail.com
**LinkedIn:** [brillconsulting](https://www.linkedin.com/in/brillconsulting)

## Overview

This project provides a complete Python implementation for Azure DevOps operations, featuring build pipeline creation, release pipeline management, YAML generation, artifact handling, and infrastructure deployment automation. Built for enterprise CI/CD workflows with Azure's DevOps platform.

## Features

### CI/CD Capabilities
- **Build Pipeline Definition**: Create and configure build pipelines
- **Release Pipeline Management**: Multi-stage deployment pipelines
- **YAML Pipeline Generation**: Generate pipeline-as-code templates
- **Multi-Stage Pipelines**: Complex workflows with dependencies
- **Environment Approvals**: Manual and automated approvals
- **Variable Groups**: Centralized configuration management
- **Service Connections**: External service integration

### Build Features
- **Multi-Language Support**: Python, Node.js, .NET, Java, Go
- **Automated Testing**: Unit, integration, and E2E tests
- **Code Quality**: Static analysis and linting
- **Build Artifacts**: Package and publish build outputs
- **Docker Image Building**: Container image creation
- **Parallel Jobs**: Faster builds with parallelization

### Release Features
- **Environment Management**: Dev, staging, production environments
- **Deployment Strategies**: Blue-green, canary, rolling updates
- **Rollback Capabilities**: Automated rollback on failure
- **Infrastructure Deployment**: ARM, Terraform, Bicep
- **Application Deployment**: Web apps, functions, containers

### Advanced Features
- **Pipeline Templates**: Reusable pipeline components
- **Conditional Execution**: Smart pipeline logic
- **Matrix Builds**: Test across multiple configurations
- **Scheduled Triggers**: Automated pipeline runs
- **Pull Request Validation**: Pre-merge testing

## Architecture

```
DevOps/
├── azure_devops.py            # Main implementation
├── requirements.txt           # Dependencies
└── README.md                  # Documentation
```

## Installation

```bash
# Clone the repository
git clone https://github.com/BrillConsulting/BrillConsulting.git
cd BrillConsulting/Azure/DevOps

# Install dependencies
pip install -r requirements.txt
```

## Configuration

```python
from azure_devops import AzureDevOpsProject, BuildPipeline

project = AzureDevOpsProject(
    organization="your-org",
    project="your-project",
    pat_token="your-personal-access-token"
)
```

## Usage Examples

### Create Build Pipeline

```python
# Create CI build pipeline
build_pipeline = project.create_build_pipeline(
    name="CI-Build",
    language="python"
)

# Add triggers
build_pipeline.add_trigger("push", ["main", "develop"])
build_pipeline.add_trigger("pull_request", ["main"])

# Add build steps
build_pipeline.add_step("install", "pip install -r requirements.txt")
build_pipeline.add_step("test", "pytest tests/")
build_pipeline.add_step("lint", "flake8 .")

# Generate YAML
yaml_content = build_pipeline.generate_yaml()
print(yaml_content)
```

### Create Release Pipeline

```python
# Create multi-stage release pipeline
release = project.create_release_pipeline(
    name="CD-Release",
    environments=["dev", "staging", "production"]
)

# Configure deployment
release.add_deployment_step("dev", "deploy_webapp", {
    "webapp_name": "app-dev",
    "resource_group": "rg-dev"
})

# Add approval gates
release.add_approval("production", approvers=["admin@company.com"])

# Run release
result = release.run(artifact_version="1.0.0")
```

### Docker Build Pipeline

```python
docker_pipeline = project.create_build_pipeline(
    name="Docker-Build",
    language="docker"
)

docker_pipeline.add_step("build", "docker build -t myapp:latest .")
docker_pipeline.add_step("push", "docker push myregistry.azurecr.io/myapp:latest")
docker_pipeline.add_step("scan", "trivy image myapp:latest")
```

### Infrastructure Deployment

```python
infra_pipeline = project.create_build_pipeline(
    name="Infrastructure-Deploy",
    language="terraform"
)

infra_pipeline.add_step("init", "terraform init")
infra_pipeline.add_step("plan", "terraform plan -out=tfplan")
infra_pipeline.add_step("apply", "terraform apply tfplan")
```

### Multi-Stage YAML Pipeline

```python
yaml_pipeline = f"""
trigger:
  branches:
    include:
      - main
      - develop

stages:
  - stage: Build
    jobs:
      - job: BuildJob
        steps:
          - task: UsePythonVersion@0
            inputs:
              versionSpec: '3.11'
          - script: pip install -r requirements.txt
          - script: pytest tests/
          - script: python -m build

  - stage: Deploy_Dev
    dependsOn: Build
    jobs:
      - deployment: DeployDev
        environment: dev
        strategy:
          runOnce:
            deploy:
              steps:
                - task: AzureWebApp@1
                  inputs:
                    appName: 'app-dev'

  - stage: Deploy_Prod
    dependsOn: Deploy_Dev
    jobs:
      - deployment: DeployProd
        environment: production
        strategy:
          runOnce:
            deploy:
              steps:
                - task: AzureWebApp@1
                  inputs:
                    appName: 'app-prod'
"""
```

## Running Demos

```bash
# Run all demo functions
python azure_devops.py
```

## API Reference

### AzureDevOpsProject

**`create_build_pipeline(name, language)`** - Create new build pipeline

**`create_release_pipeline(name, environments)`** - Create release pipeline

**`get_pipeline(pipeline_id)`** - Get pipeline by ID

**`list_pipelines()`** - List all pipelines

### BuildPipeline

**`add_trigger(trigger_type, branches)`** - Add pipeline trigger

**`add_step(name, command)`** - Add build step

**`generate_yaml()`** - Generate YAML definition

**`run(trigger_reason)`** - Execute pipeline

### ReleasePipeline

**`add_deployment_step(environment, action, config)`** - Add deployment

**`add_approval(environment, approvers)`** - Add approval gate

**`run(artifact_version)`** - Run release

## Best Practices

### 1. Version Control Pipelines
Store pipelines as YAML in repository:
```python
# Save YAML to repository
with open('.azure-pipelines/ci.yml', 'w') as f:
    f.write(build_pipeline.generate_yaml())
```

### 2. Use Variable Groups
Centralize configuration:
```python
project.create_variable_group("app-config", {
    "DATABASE_URL": "...",
    "API_KEY": "..."  # Use Azure Key Vault for secrets
})
```

### 3. Implement Quality Gates
```python
build_pipeline.add_quality_gate({
    "code_coverage": 80,
    "test_pass_rate": 100,
    "security_score": 90
})
```

### 4. Use Pipeline Templates
```python
# Create reusable template
template = project.create_pipeline_template(
    name="python-build-template",
    steps=[
        "install_dependencies",
        "run_tests",
        "build_package"
    ]
)
```

### 5. Enable Caching
```python
# Cache dependencies for faster builds
build_pipeline.enable_cache(
    key="pip | requirements.txt",
    path="~/.cache/pip"
)
```

## Use Cases

### 1. Continuous Integration
```python
# Automated testing on every commit
ci_pipeline = project.create_build_pipeline("CI")
ci_pipeline.add_trigger("push", ["*"])
ci_pipeline.add_step("test", "pytest --cov")
```

### 2. Continuous Deployment
```python
# Automated deployment to environments
cd_pipeline = project.create_release_pipeline("CD", ["dev", "prod"])
cd_pipeline.enable_auto_deploy("dev")
```

### 3. Infrastructure as Code
```python
# Deploy infrastructure changes
iac_pipeline = project.create_build_pipeline("IaC")
iac_pipeline.add_step("validate", "terraform validate")
iac_pipeline.add_step("deploy", "terraform apply -auto-approve")
```

### 4. Multi-Platform Builds
```python
# Build for multiple platforms
matrix_pipeline = project.create_build_pipeline("Multi-Platform")
matrix_pipeline.set_matrix({
    "os": ["ubuntu", "windows", "macos"],
    "python": ["3.8", "3.9", "3.10", "3.11"]
})
```

## Performance Optimization

### 1. Parallel Jobs
```python
# Run tests in parallel
build_pipeline.enable_parallel_jobs(max_parallel=5)
```

### 2. Incremental Builds
```python
# Only build changed components
build_pipeline.enable_incremental_builds()
```

### 3. Artifact Caching
```python
# Cache build artifacts
build_pipeline.cache_artifacts(["dist/", "build/"])
```

## Security Considerations

1. **Secret Management**: Use Azure Key Vault for secrets
2. **Pipeline Permissions**: Restrict who can run pipelines
3. **Code Scanning**: Integrate security scanning
4. **Audit Logging**: Track all pipeline executions
5. **Branch Protection**: Require PR reviews

## Troubleshooting

**Issue**: Pipeline fails with permission error
**Solution**: Check service principal permissions

**Issue**: Slow build times
**Solution**: Enable caching and parallel jobs

**Issue**: Deployment fails
**Solution**: Verify environment configuration

## Deployment

### Azure Setup
```bash
# Create Azure DevOps project
az devops project create \
    --name MyProject \
    --org https://dev.azure.com/myorg

# Create service connection
az devops service-endpoint azurerm create \
    --azure-rm-service-principal-id xxx \
    --azure-rm-tenant-id xxx
```

## Monitoring

### Key Metrics
- Build success rate
- Build duration
- Deployment frequency
- Mean time to recovery (MTTR)
- Change failure rate

### Integration
```python
from opencensus.ext.azure.log_exporter import AzureLogHandler

logger.addHandler(AzureLogHandler(
    connection_string='InstrumentationKey=...'
))
```

## Dependencies

```
Python >= 3.8
azure-devops >= 6.0.0
pyyaml >= 6.0
```

## Support

For questions or support:
- Email: clientbrill@gmail.com
- LinkedIn: [brillconsulting](https://www.linkedin.com/in/brillconsulting)

## Related Projects

- [Infrastructure as Code](../InfrastructureAsCode/)
- [Azure Monitor](../AzureMonitor/)
- [Container Apps](../ContainerApps/)

---

**Built with Azure DevOps** | **Brill Consulting © 2024**
