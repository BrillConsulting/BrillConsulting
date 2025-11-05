# AWS ECS (Elastic Container Service)

Container orchestration platform supporting Docker containers with Fargate and EC2 launch types.

## Features

- **Cluster Management**: Create and manage ECS clusters
- **Task Definitions**: Define containerized applications
- **Services**: Deploy and auto-scale long-running services
- **Standalone Tasks**: Run one-off or batch jobs
- **Fargate Launch Type**: Serverless container execution
- **EC2 Launch Type**: Run containers on managed EC2 instances
- **Load Balancer Integration**: ALB/NLB support
- **Auto Scaling**: Service and task-level scaling

## Quick Start

```python
from aws_ecs import ECSManager

# Initialize
ecs = ECSManager(region='us-east-1')

# Create cluster
cluster = ecs.create_cluster(
    cluster_name='prod-cluster',
    capacity_providers=['FARGATE', 'FARGATE_SPOT']
)

# Register task definition
task_def = ecs.register_task_definition(
    family='web-app',
    container_definitions=[{
        'name': 'nginx',
        'image': 'nginx:latest',
        'memory': 512,
        'portMappings': [{'containerPort': 80, 'protocol': 'tcp'}]
    }],
    requires_compatibilities=['FARGATE'],
    network_mode='awsvpc',
    cpu='256',
    memory='512'
)

# Create service
service = ecs.create_service(
    cluster='prod-cluster',
    service_name='web-service',
    task_definition='web-app:1',
    desired_count=3,
    launch_type='FARGATE',
    network_configuration={
        'awsvpcConfiguration': {
            'subnets': ['subnet-xxx'],
            'securityGroups': ['sg-xxx'],
            'assignPublicIp': 'ENABLED'
        }
    }
)
```

## Use Cases

- **Microservices**: Deploy containerized microservices
- **Batch Processing**: Run scheduled batch jobs
- **Web Applications**: Host scalable web apps
- **CI/CD**: Container-based build and deployment pipelines
- **Machine Learning**: Deploy ML inference containers

## Launch Types

### Fargate (Serverless)
- No infrastructure management
- Pay per vCPU and memory
- Ideal for variable workloads

### EC2
- Full control over instances
- Cost-effective for steady workloads
- GPU support available

## Author

Brill Consulting
