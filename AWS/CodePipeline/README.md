# AWS CodePipeline Management

**Production-ready CI/CD pipeline orchestration with multi-stage workflows.**

## 🎯 Overview

Comprehensive CodePipeline management for automated software delivery:
- Multi-stage pipeline creation (Source → Build → Deploy)
- Source integrations (CodeCommit, GitHub, S3)
- Build with CodeBuild
- Deploy to ECS, Lambda, CloudFormation
- Manual approval actions
- Execution monitoring and control

## ✨ Features

- **Pipeline Management**: Create, update, delete CI/CD pipelines
- **Execution Control**: Start, stop, monitor pipeline runs
- **Manual Approvals**: Implement approval gates
- **State Monitoring**: Track stage-by-stage progress
- **Multi-source Support**: Git repositories, S3, ECR

## 📋 Prerequisites

1. **AWS Account** with CodePipeline permissions
2. **IAM Role** with pipeline execution permissions
3. **S3 Bucket** for artifacts
4. **Source Repository** (CodeCommit/GitHub)
5. **Python 3.8+** and **boto3**

## 🚀 Quick Start

```python
from aws_codepipeline import CodePipelineManager

cp = CodePipelineManager(region='us-east-1')

# Create pipeline
pipeline = cp.create_pipeline(
    pipeline_name='my-app-pipeline',
    role_arn='arn:aws:iam::123456789012:role/CodePipelineRole',
    artifact_store={'type': 'S3', 'location': 'my-artifacts'},
    stages=[...]  # See examples below
)

# Start execution
execution = cp.start_pipeline_execution('my-app-pipeline')

# Monitor
state = cp.get_pipeline_state('my-app-pipeline')
```

## 💻 Usage Examples

### Complete Pipeline (Source → Build → Deploy)

```python
stages = [
    {
        'name': 'Source',
        'actions': [{
            'name': 'SourceAction',
            'actionTypeId': {
                'category': 'Source',
                'owner': 'AWS',
                'provider': 'CodeCommit',
                'version': '1'
            },
            'configuration': {
                'RepositoryName': 'my-repo',
                'BranchName': 'main'
            },
            'outputArtifacts': [{'name': 'SourceOutput'}]
        }]
    },
    {
        'name': 'Build',
        'actions': [{
            'name': 'BuildAction',
            'actionTypeId': {
                'category': 'Build',
                'owner': 'AWS',
                'provider': 'CodeBuild',
                'version': '1'
            },
            'configuration': {'ProjectName': 'my-build'},
            'inputArtifacts': [{'name': 'SourceOutput'}],
            'outputArtifacts': [{'name': 'BuildOutput'}]
        }]
    },
    {
        'name': 'Deploy',
        'actions': [{
            'name': 'DeployAction',
            'actionTypeId': {
                'category': 'Deploy',
                'owner': 'AWS',
                'provider': 'ECS',
                'version': '1'
            },
            'configuration': {
                'ClusterName': 'prod-cluster',
                'ServiceName': 'web-service'
            },
            'inputArtifacts': [{'name': 'BuildOutput'}]
        }]
    }
]

pipeline = cp.create_pipeline(
    pipeline_name='prod-pipeline',
    role_arn='arn:aws:iam::123456789012:role/PipelineRole',
    artifact_store={'type': 'S3', 'location': 'pipeline-artifacts'},
    stages=stages
)
```

### Pipeline with Manual Approval

```python
approval_stage = {
    'name': 'Approval',
    'actions': [{
        'name': 'ManualApproval',
        'actionTypeId': {
            'category': 'Approval',
            'owner': 'AWS',
            'provider': 'Manual',
            'version': '1'
        },
        'configuration': {
            'CustomData': 'Please review before production deploy'
        }
    }]
}

# Add approval stage before deploy
stages.insert(2, approval_stage)
```

### Monitor and Control

```python
# List all pipelines
pipelines = cp.list_pipelines()

# Start execution
exec_result = cp.start_pipeline_execution('my-pipeline')
exec_id = exec_result['execution_id']

# Check status
status = cp.get_pipeline_execution('my-pipeline', exec_id)
print(f"Status: {status['status']}")

# Get detailed state
state = cp.get_pipeline_state('my-pipeline')
for stage in state['stages']:
    print(f"{stage['name']}: {stage['status']}")

# Stop if needed
cp.stop_pipeline_execution(
    'my-pipeline',
    exec_id,
    reason='Deployment blocked due to critical bug'
)
```

### Approve/Reject

```python
# Approve
cp.put_approval_result(
    pipeline_name='my-pipeline',
    stage_name='Approval',
    action_name='ManualApproval',
    token='token-from-sns',
    result={
        'summary': 'Approved by DevOps team - tests passed',
        'status': 'Approved'
    }
)

# Reject
cp.put_approval_result(
    pipeline_name='my-pipeline',
    stage_name='Approval',
    action_name='ManualApproval',
    token='token-from-sns',
    result={
        'summary': 'Rejected - security scan failed',
        'status': 'Rejected'
    }
)
```

## 🏗️ Architecture

```
CodePipelineManager
├── Pipeline Management
│   ├── create_pipeline()
│   ├── get_pipeline()
│   ├── update_pipeline()
│   ├── delete_pipeline()
│   └── list_pipelines()
│
├── Execution Control
│   ├── start_pipeline_execution()
│   ├── get_pipeline_execution()
│   ├── list_pipeline_executions()
│   └── stop_pipeline_execution()
│
├── State & Monitoring
│   └── get_pipeline_state()
│
├── Approval Actions
│   └── put_approval_result()
│
└── Summary
    └── get_summary()
```

## 🔒 Best Practices

1. **IAM Roles**: Separate roles for pipeline and execution
2. **Artifact Encryption**: Enable S3 encryption for artifacts
3. **Approval Gates**: Add manual approvals before production
4. **Notifications**: Configure SNS for pipeline events
5. **Rollback**: Implement automated rollback on failure
6. **Versioning**: Tag pipeline versions

## 📊 Common Pipeline Patterns

### Microservices Deployment
- Source (CodeCommit) → Build (Docker) → Deploy (ECS)

### Serverless Application
- Source (GitHub) → Build (SAM) → Deploy (Lambda)

### Infrastructure as Code
- Source (Git) → Validate (CloudFormation) → Deploy (Stacks)

### Blue/Green Deployment
- Build → Deploy Blue → Approval → Swap Traffic → Deploy Green

## 🐛 Troubleshooting

**Issue: Pipeline fails at Source stage**
- **Solution**: Verify repository permissions and branch name

**Issue: Artifact not found in Build stage**
- **Solution**: Check outputArtifacts match inputArtifacts names

**Issue: Manual approval token expired**
- **Solution**: Approvals expire after 7 days; restart pipeline

## 🔗 Related Services

- **CodeCommit**: Git repositories
- **CodeBuild**: Build and test
- **CodeDeploy**: Application deployment
- **S3**: Artifact storage
- **SNS**: Pipeline notifications
- **CloudWatch**: Logs and metrics

## 📞 Support

- **Email**: clientbrill@gmail.com
- **LinkedIn**: [brillconsulting](https://www.linkedin.com/in/brillconsulting)

**Author**: Brill Consulting | **Last Updated**: November 2025
