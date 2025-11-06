# Container Management System v2.0.0

**Author:** BrillConsulting
**Category:** Linux System Administration
**License:** MIT

Production-ready container orchestration system providing comprehensive Docker and Podman management with advanced features for lifecycle management, networking, security scanning, monitoring, and compose orchestration.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Core Components](#core-components)
- [Usage Examples](#usage-examples)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

---

## Overview

The Container Management System is a comprehensive Python-based solution for managing containerized applications at scale. It provides a unified interface for Docker and Podman runtimes, offering enterprise-grade features including lifecycle management, resource constraints, health monitoring, security scanning, and multi-container orchestration.

### Key Capabilities

- **Multi-Runtime Support**: Seamlessly switch between Docker and Podman
- **Complete Lifecycle Management**: Create, start, stop, pause, restart, and remove containers
- **Advanced Networking**: Custom networks, DNS configuration, port mapping, and host management
- **Volume Management**: Persistent storage with bind mounts, volumes, and tmpfs
- **Resource Controls**: CPU, memory, and PID limits with cgroup integration
- **Health Monitoring**: Built-in health checks and real-time statistics
- **Security Scanning**: Integrated vulnerability scanning with Trivy, Grype, or Snyk
- **Compose Orchestration**: Multi-container application management
- **Metrics Export**: JSON-based metrics collection and export

---

## Features

### Container Lifecycle Management

- **Create & Run**: Full container configuration support
- **State Management**: Start, stop, pause, unpause, restart, kill
- **Interactive Execution**: Execute commands in running containers
- **Log Management**: Stream and retrieve container logs
- **Inspection**: Detailed container metadata and state information

### Image Management

- **Registry Operations**: Pull and push images from/to registries
- **Build System**: Build images from Dockerfiles with build arguments
- **Tagging**: Flexible image tagging and aliasing
- **Cleanup**: Prune unused images and reclaim disk space
- **Inspection**: Detailed image metadata and layer information

### Network Management

- **Network Creation**: Custom networks with configurable drivers
- **Container Networking**: Connect/disconnect containers dynamically
- **IP Management**: Static IP assignment and subnet configuration
- **DNS Configuration**: Custom DNS servers and host entries
- **Port Mapping**: Flexible port forwarding and exposure

### Volume Management

- **Volume Operations**: Create, remove, list, and inspect volumes
- **Mount Types**: Bind mounts, named volumes, and tmpfs
- **Driver Support**: Multiple volume drivers and options
- **Cleanup**: Prune unused volumes

### Resource Management

- **CPU Controls**: CPU shares, quotas, and pinning
- **Memory Limits**: Memory and swap constraints
- **PID Limits**: Process count restrictions
- **Kernel Memory**: Kernel memory limits

### Health & Monitoring

- **Health Checks**: Configurable health check commands
- **Real-time Stats**: CPU, memory, network, and disk I/O metrics
- **Status Tracking**: Container health status monitoring
- **Metrics Collection**: Time-series metrics for analysis

### Security Features

- **Vulnerability Scanning**: Automated image security scanning
- **Scanner Integration**: Support for Trivy, Grype, and Snyk
- **Severity Filtering**: Focus on critical vulnerabilities
- **Summary Reports**: Vulnerability counts by severity

### Compose Orchestration

- **Multi-Container Apps**: Deploy complex applications
- **Service Management**: Start, stop, restart services
- **Scaling**: Dynamic service scaling
- **Build Integration**: Build images as part of deployment
- **Log Aggregation**: Centralized logging for all services

---

## Architecture

### System Components

```
ContainerManagementSystem
├── ContainerLifecycleManager    # Container operations
├── ImageManager                 # Image operations
├── NetworkManager               # Network operations
├── VolumeManager                # Volume operations
├── MonitoringManager            # Metrics & health checks
├── SecurityScanner              # Vulnerability scanning
└── ComposeOrchestrator          # Multi-container orchestration
```

### Data Models

- **ContainerConfig**: Complete container specification
- **ResourceLimits**: CPU, memory, and process constraints
- **NetworkConfig**: Network and connectivity settings
- **VolumeMount**: Storage mount configuration
- **HealthCheck**: Health check parameters
- **ContainerStats**: Resource usage statistics

---

## Installation

### Prerequisites

- Python 3.8 or higher
- Docker or Podman installed and running
- Optional: Docker Compose for orchestration
- Optional: Trivy, Grype, or Snyk for security scanning

### Install Dependencies

```bash
cd /home/user/BrillConsulting/Linux/ContainerManagement
pip install -r requirements.txt
```

### Verify Installation

```bash
python container_management.py
```

---

## Quick Start

### Basic Usage

```python
from container_management import (
    ContainerManagementSystem,
    ContainerRuntime,
    ContainerConfig,
    NetworkConfig,
    ResourceLimits
)

# Initialize system with Docker
cms = ContainerManagementSystem(ContainerRuntime.DOCKER)

# Get system information
info = cms.get_system_info()
print(f"Runtime: {info['runtime']}")

# List containers
containers = cms.lifecycle.list_containers(all_containers=True)
print(f"Total containers: {containers['count']}")

# List images
images = cms.images.list_images()
print(f"Total images: {images['count']}")
```

### Deploy a Container

```python
# Create container configuration
config = ContainerConfig(
    name="web-app",
    image="nginx:latest",
    network=NetworkConfig(
        network="bridge",
        ports={"80": "8080"}  # container:host
    ),
    resources=ResourceLimits(
        memory="512m",
        cpu_shares=512
    ),
    environment={
        "ENV": "production"
    },
    restart_policy="unless-stopped"
)

# Deploy application
result = cms.deploy_application(config)
if result['success']:
    print(f"Container deployed: {result['container_id']}")
```

---

## Core Components

### ContainerLifecycleManager

Manages complete container lifecycle operations.

**Key Methods:**
- `create(config)` - Create container from configuration
- `run(config)` - Create and start container
- `start(container)` - Start stopped container
- `stop(container, timeout)` - Stop running container
- `restart(container)` - Restart container
- `pause(container)` - Pause running container
- `unpause(container)` - Resume paused container
- `remove(container, force, volumes)` - Remove container
- `list_containers(all)` - List containers
- `inspect(container)` - Get container details
- `logs(container, tail, follow)` - Get container logs
- `exec(container, command)` - Execute command in container

### ImageManager

Handles container image operations.

**Key Methods:**
- `pull(image, tag)` - Pull image from registry
- `push(image, tag)` - Push image to registry
- `build(path, tag, dockerfile, build_args)` - Build image
- `tag(source, target)` - Tag image
- `remove(image, force)` - Remove image
- `list_images()` - List all images
- `inspect(image)` - Get image details
- `prune(all)` - Remove unused images

### NetworkManager

Manages container networking.

**Key Methods:**
- `create(name, driver, subnet, gateway)` - Create network
- `connect(network, container, ip)` - Connect container
- `disconnect(network, container)` - Disconnect container
- `remove(network)` - Remove network
- `list_networks()` - List networks
- `inspect(network)` - Get network details
- `prune()` - Remove unused networks

### VolumeManager

Handles persistent storage.

**Key Methods:**
- `create(name, driver, options, labels)` - Create volume
- `remove(volume, force)` - Remove volume
- `list_volumes()` - List volumes
- `inspect(volume)` - Get volume details
- `prune()` - Remove unused volumes

### MonitoringManager

Provides monitoring and metrics collection.

**Key Methods:**
- `get_container_stats(container)` - Get real-time statistics
- `get_health_status(container)` - Get health check status
- `monitor_containers(interval, duration)` - Continuous monitoring

### SecurityScanner

Scans images for vulnerabilities.

**Key Methods:**
- `scan_image(image, severity)` - Scan for vulnerabilities
- `get_vulnerability_summary(results)` - Parse scan results

### ComposeOrchestrator

Manages multi-container applications.

**Key Methods:**
- `up(detach, build, scale)` - Start services
- `down(volumes, remove_orphans)` - Stop services
- `start()` - Start existing services
- `stop(timeout)` - Stop running services
- `restart()` - Restart services
- `ps()` - List services
- `logs(service, tail, follow)` - Get service logs
- `build(no_cache)` - Build services
- `pull()` - Pull service images

---

## Usage Examples

### Example 1: Container with Resource Limits

```python
from container_management import *

cms = ContainerManagementSystem(ContainerRuntime.DOCKER)

config = ContainerConfig(
    name="resource-limited",
    image="ubuntu:latest",
    command=["sleep", "infinity"],
    resources=ResourceLimits(
        memory="256m",
        memory_swap="512m",
        cpu_shares=512,
        cpuset_cpus="0,1",
        pids_limit=100
    )
)

result = cms.lifecycle.run(config)
print(f"Container ID: {result['container_id']}")
```

### Example 2: Container with Health Check

```python
config = ContainerConfig(
    name="web-service",
    image="nginx:latest",
    health_check=HealthCheck(
        test=["CMD", "curl", "-f", "http://localhost/"],
        interval="30s",
        timeout="10s",
        retries=3,
        start_period="5s"
    ),
    network=NetworkConfig(
        ports={"80": "8080"}
    )
)

result = cms.deploy_application(config)

# Wait for health check
time.sleep(10)
health = cms.monitoring.get_health_status(result['container_id'])
print(f"Health: {health.value}")
```

### Example 3: Custom Network Setup

```python
# Create custom network
network_result = cms.networks.create(
    name="app-network",
    driver="bridge",
    subnet="172.20.0.0/16",
    gateway="172.20.0.1"
)

# Deploy containers on custom network
config1 = ContainerConfig(
    name="database",
    image="postgres:latest",
    network=NetworkConfig(
        network="app-network",
        ip_address="172.20.0.10",
        hostname="db"
    ),
    environment={
        "POSTGRES_PASSWORD": "secret"
    }
)

config2 = ContainerConfig(
    name="app",
    image="myapp:latest",
    network=NetworkConfig(
        network="app-network",
        ip_address="172.20.0.20",
        hostname="app",
        extra_hosts={"db": "172.20.0.10"}
    )
)

cms.lifecycle.run(config1)
cms.lifecycle.run(config2)
```

### Example 4: Volume Management

```python
# Create named volume
cms.volumes.create(
    name="app-data",
    driver="local",
    labels={"app": "myapp", "env": "production"}
)

# Use volume in container
config = ContainerConfig(
    name="data-app",
    image="myapp:latest",
    volumes=[
        VolumeMount(
            source="app-data",
            target="/data",
            volume_type="volume"
        ),
        VolumeMount(
            source="/tmp/cache",
            target="/cache",
            volume_type="tmpfs"
        )
    ]
)

cms.lifecycle.run(config)
```

### Example 5: Security Scanning

```python
# Scan image for vulnerabilities
result = cms.security.scan_image(
    image="nginx:latest",
    severity="HIGH,CRITICAL"
)

if result['success']:
    summary = cms.security.get_vulnerability_summary(result['results'])
    print(f"Vulnerabilities found:")
    for severity, count in summary.items():
        print(f"  {severity}: {count}")
```

### Example 6: Monitoring and Metrics

```python
# Get real-time statistics
stats = cms.monitoring.get_container_stats("web-app")
if stats:
    print(f"CPU: {stats.cpu_percent:.2f}%")
    print(f"Memory: {stats.memory_usage / 1024**2:.2f}MB")
    print(f"Network RX: {stats.network_rx_bytes / 1024:.2f}KB")
    print(f"Network TX: {stats.network_tx_bytes / 1024:.2f}KB")

# Export all metrics
cms.export_metrics("metrics.json")
```

### Example 7: Docker Compose Orchestration

```python
# Using docker-compose.yml in current directory
compose = ComposeOrchestrator("docker-compose.yml")

# Start all services
compose.up(detach=True, build=False)

# Scale a service
compose.up(scale={"web": 3})

# View service status
services = compose.ps()
print(f"Running services: {services['count']}")

# Get logs
logs = compose.logs(service="web", tail=50)
print(logs['logs'])

# Stop all services
compose.down(volumes=False)
```

### Example 8: Cleanup Resources

```python
# Clean up all unused resources
result = cms.cleanup_resources(all_resources=True)
print("Cleanup completed:")
print(f"  Images: {result['results']['images']}")
print(f"  Volumes: {result['results']['volumes']}")
print(f"  Networks: {result['results']['networks']}")
```

---

## Configuration

### Environment Variables

```bash
# Docker/Podman socket (if non-standard)
export DOCKER_HOST=unix:///var/run/docker.sock

# Logging level
export LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR
```

### Resource Limits Reference

| Parameter | Description | Example |
|-----------|-------------|---------|
| `cpu_shares` | CPU share weight (relative) | `512` |
| `cpu_quota` | CPU quota in microseconds | `50000` |
| `cpuset_cpus` | CPUs to use | `"0-3"` or `"0,2"` |
| `memory` | Memory limit | `"512m"`, `"2g"` |
| `memory_swap` | Memory + swap limit | `"1g"` |
| `pids_limit` | Max processes | `100` |

### Health Check Configuration

```python
HealthCheck(
    test=["CMD", "curl", "-f", "http://localhost/health"],
    interval="30s",    # Check every 30 seconds
    timeout="10s",     # Timeout after 10 seconds
    retries=3,         # Mark unhealthy after 3 failures
    start_period="0s"  # Start checking immediately
)
```

---

## API Reference

### ContainerManagementSystem

```python
class ContainerManagementSystem:
    def __init__(self, runtime: ContainerRuntime = ContainerRuntime.DOCKER)
    def get_system_info(self) -> Dict[str, Any]
    def deploy_application(self, config: ContainerConfig) -> Dict[str, Any]
    def cleanup_resources(self, all_resources: bool = False) -> Dict[str, Any]
    def export_metrics(self, output_file: str = "container_metrics.json") -> Dict[str, Any]
```

### ContainerConfig

```python
@dataclass
class ContainerConfig:
    name: str
    image: str
    command: Optional[List[str]] = None
    environment: Dict[str, str] = field(default_factory=dict)
    volumes: List[VolumeMount] = field(default_factory=list)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    resources: ResourceLimits = field(default_factory=ResourceLimits)
    health_check: Optional[HealthCheck] = None
    restart_policy: str = "no"
    detach: bool = True
    remove: bool = False
    privileged: bool = False
    labels: Dict[str, str] = field(default_factory=dict)
```

### Return Value Format

All methods return dictionaries with consistent structure:

```python
{
    'success': bool,        # Operation success status
    'data': Any,           # Result data (if successful)
    'error': str           # Error message (if failed)
}
```

---

## Best Practices

### Security

1. **Never run privileged containers** unless absolutely necessary
2. **Scan images regularly** for vulnerabilities
3. **Use resource limits** to prevent resource exhaustion
4. **Implement health checks** for critical services
5. **Use read-only volumes** when write access isn't needed
6. **Minimize image layers** and use official base images

### Performance

1. **Set appropriate resource limits** based on application needs
2. **Use volume mounts** instead of copying large datasets
3. **Leverage image caching** during builds
4. **Use multi-stage builds** to reduce image size
5. **Monitor container metrics** regularly

### Operations

1. **Use restart policies** for automatic recovery
2. **Implement proper logging** strategies
3. **Tag images consistently** (semantic versioning)
4. **Use Docker Compose** for multi-container applications
5. **Regular cleanup** of unused resources
6. **Backup volumes** before destructive operations

### Networking

1. **Use custom networks** for container isolation
2. **Avoid host networking** unless required
3. **Document port mappings** clearly
4. **Use DNS names** instead of IP addresses

---

## Troubleshooting

### Common Issues

**Issue: Container fails to start**
```python
# Check logs
logs = cms.lifecycle.logs("container-name", tail=100)
print(logs['logs'])

# Inspect container
info = cms.lifecycle.inspect("container-name")
print(info['info']['State'])
```

**Issue: Permission denied errors**
```bash
# Add user to docker group
sudo usermod -aG docker $USER
newgrp docker
```

**Issue: Port already in use**
```python
# List containers using the port
containers = cms.lifecycle.list_containers(all_containers=True)
for c in containers['containers']:
    info = cms.lifecycle.inspect(c['ID'])
    ports = info['info']['NetworkSettings']['Ports']
    print(f"{c['Names']}: {ports}")
```

**Issue: Out of disk space**
```python
# Clean up resources
cms.cleanup_resources(all_resources=True)

# Prune specific resources
cms.images.prune(all_images=True)
cms.volumes.prune()
```

**Issue: Container health check failing**
```python
# Check health status
health = cms.monitoring.get_health_status("container-name")
print(f"Health: {health.value}")

# View health check logs
info = cms.lifecycle.inspect("container-name")
health_log = info['info']['State']['Health']['Log']
for entry in health_log:
    print(f"Exit: {entry['ExitCode']}, Output: {entry['Output']}")
```

---

## Contributing

Contributions are welcome! Please follow these guidelines:

1. **Code Style**: Follow PEP 8 conventions
2. **Documentation**: Update README for new features
3. **Testing**: Test with both Docker and Podman
4. **Error Handling**: Implement comprehensive error handling
5. **Logging**: Use appropriate logging levels

### Development Setup

```bash
# Clone repository
git clone https://github.com/BrillConsulting/container-management
cd container-management

# Install development dependencies
pip install -r requirements.txt

# Run tests
python -m pytest tests/

# Run demo
python container_management.py
```

---

## License

MIT License - see LICENSE file for details

---

## Support

For issues, questions, or contributions:
- **Email**: support@brillconsulting.com
- **Issues**: GitHub Issues
- **Documentation**: See inline code documentation

---

## Changelog

### v2.0.0 (2025-11-06)
- Complete production-ready implementation
- Docker and Podman runtime support
- Container lifecycle management
- Image management operations
- Network management
- Volume management
- Resource limits and controls
- Health check monitoring
- Real-time statistics and metrics
- Security vulnerability scanning
- Docker Compose orchestration
- Comprehensive error handling
- Detailed logging system
- Production-grade documentation

### v1.0.0 (Initial)
- Basic container management skeleton

---

**Built with precision by BrillConsulting**
