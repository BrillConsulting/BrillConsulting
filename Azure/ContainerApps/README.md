# Azure Container Apps Service Integration

Advanced implementation of Azure Container Apps with deployment, scaling, revisions, traffic management, and Dapr integration.

**Author:** BrillConsulting
**Contact:** clientbrill@gmail.com
**LinkedIn:** [brillconsulting](https://www.linkedin.com/in/brillconsulting)

## Overview

Comprehensive Python implementation for Azure Container Apps, featuring container deployment, revision management, auto-scaling, traffic splitting, Dapr integration, and container registry management.

## Features

### Core Capabilities
- **Container Deployment**: Deploy containerized applications
- **Revision Management**: Multiple revision support with traffic control
- **Auto-Scaling**: HTTP, CPU, memory, and custom metric scaling
- **Traffic Splitting**: A/B testing, canary, and blue-green deployments
- **Dapr Integration**: Service invocation, state management, pub/sub

### Advanced Features
- **Multiple Revisions**: Run multiple versions simultaneously
- **Traffic Management**: Precise traffic control between revisions
- **Custom Domains**: SSL/TLS certificate management
- **Container Registry**: ACR and private registry integration
- **Environment Management**: Virtual network and Log Analytics integration

## Installation

```bash
pip install -r requirements.txt
```

## Usage Examples

### Container App Deployment

```python
from containerapps import (
    ContainerAppManager, Container, IngressConfig, 
    RevisionMode
)

manager = ContainerAppManager("sub-id", "my-rg", "env-name")

# Create container
container = Container(
    name="api",
    image="myregistry.azurecr.io/myapp:latest",
    cpu=0.5,
    memory="1Gi",
    env=[
        {"name": "DATABASE_URL", "value": "postgresql://..."},
        {"name": "API_KEY", "secretRef": "api-key"}
    ]
)

# Create ingress
ingress = IngressConfig(
    external=True,
    target_port=8080
)

# Deploy app
app = manager.create_container_app(
    "my-api",
    containers=[container],
    ingress=ingress,
    revision_mode=RevisionMode.MULTIPLE
)
```

### Traffic Management

```python
from containerapps import TrafficManager

traffic = TrafficManager(manager)

# Canary deployment (10% to new version)
traffic.canary_deployment(
    "my-api",
    current_revision="v1",
    canary_revision="v2",
    canary_percent=10
)

# Blue-green deployment
traffic.blue_green_deployment(
    "my-api",
    blue_revision="v1",
    green_revision="v2",
    switch_to_green=True
)
```

### Auto-Scaling

```python
from containerapps import ScalingManager

scaling = ScalingManager()

# HTTP-based scaling
http_rule = scaling.create_http_scale_rule(
    "http-scale",
    concurrent_requests=50,
    min_replicas=1,
    max_replicas=10
)

# CPU-based scaling
cpu_rule = scaling.create_cpu_scale_rule(
    "cpu-scale",
    cpu_percentage=70,
    min_replicas=2,
    max_replicas=15
)
```

### Dapr Integration

```python
from containerapps import DaprManager, DaprConfig

dapr = DaprManager()

# State store component
state_store = dapr.create_state_store_component(
    "statestore",
    "state.azure.cosmosdb",
    {
        "url": "https://mycosmosdb.documents.azure.com:443/",
        "database": "mydb"
    }
)

# Pub/sub component
pubsub = dapr.create_pubsub_component(
    "pubsub",
    "pubsub.azure.servicebus",
    {
        "connectionString": "Endpoint=sb://..."
    }
)

# Dapr config for app
dapr_config = DaprConfig(
    enabled=True,
    app_id="my-app",
    app_port=8080
)
```

## Running Demos

```bash
python containerapps.py
```

## Best Practices

1. **Revisions**: Use multiple revisions for zero-downtime deployments
2. **Scaling**: Configure appropriate min/max replicas
3. **Traffic**: Test with canary before full rollout
4. **Dapr**: Use for microservices communication
5. **Secrets**: Store sensitive data in Key Vault

## API Reference

See implementation for comprehensive API documentation.

## Support

- Email: clientbrill@gmail.com
- LinkedIn: [brillconsulting](https://www.linkedin.com/in/brillconsulting)

---

**Built with Azure Container Apps** | **Brill Consulting © 2024**
