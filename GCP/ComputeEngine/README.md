# Compute Engine - Advanced VM Management

Comprehensive Google Cloud Compute Engine implementation for managing virtual machines, persistent disks, snapshots, instance groups, autoscaling, and load balancing.

## Features

### Instance Management
- **VM Creation**: Create instances with custom machine types and images
- **Preemptible Instances**: Cost-effective batch processing with spot instances
- **GPU Instances**: ML and GPU-accelerated workloads with NVIDIA Tesla GPUs
- **Startup Scripts**: Automate instance configuration with custom scripts
- **Instance Lifecycle**: Start, stop, restart, and delete instances

### Persistent Disks
- **Disk Creation**: Create persistent disks (standard, SSD, balanced)
- **Disk Attachment**: Attach/detach disks to running instances
- **Disk Types**: pd-standard, pd-ssd, pd-balanced, pd-extreme
- **Disk Sizing**: Configure disk size from 10GB to 64TB
- **Disk Snapshots**: Create point-in-time backups

### Snapshots
- **Snapshot Creation**: Create snapshots of persistent disks
- **Snapshot Restoration**: Create new disks from snapshots
- **Incremental Snapshots**: Automatic incremental backup
- **Snapshot Scheduling**: Automated snapshot policies
- **Cross-Region Snapshots**: Disaster recovery across regions

### Instance Templates
- **Template Creation**: Define reusable VM configurations
- **Machine Types**: Configure CPU, memory, and disk
- **Network Configuration**: VPC, subnets, and firewall tags
- **Metadata**: Custom metadata and startup scripts
- **Template Versioning**: Manage multiple template versions

### Managed Instance Groups
- **MIG Creation**: Create groups of identical VMs
- **Auto-Healing**: Automatic unhealthy instance replacement
- **Rolling Updates**: Zero-downtime deployments
- **Regional MIGs**: High availability across zones
- **Stateful MIGs**: Preserve instance state

### Autoscaling
- **CPU-Based Autoscaling**: Scale based on CPU utilization (default: 60%)
- **Custom Metrics**: Scale based on Cloud Monitoring metrics
- **Scale Range**: Configure min (1) and max (10+) instances
- **Cooldown Period**: Prevent flapping with configurable cooldown
- **Load Balancer Integration**: Scale based on load balancer metrics

### Load Balancing
- **HTTP(S) Load Balancing**: Global load balancing for HTTP/HTTPS
- **Backend Services**: Configure backends with health checks
- **URL Maps**: Route traffic based on URL patterns
- **SSL Certificates**: Managed SSL certificates for HTTPS
- **Health Checks**: HTTP, HTTPS, TCP health monitoring

## Usage Example

```python
from gcp_compute import ComputeEngineManager

# Initialize manager
mgr = ComputeEngineManager(
    project_id='my-gcp-project',
    zone='us-central1-a'
)

# Create regular instance
instance = mgr.instances.create_instance({
    'name': 'web-server-1',
    'machine_type': 'e2-medium',
    'disk_size_gb': 20
})

# Create preemptible instance for batch jobs
preemptible = mgr.instances.create_instance({
    'name': 'batch-worker-1',
    'machine_type': 'n1-standard-4',
    'preemptible': True
})

# Create GPU instance for ML
gpu_instance = mgr.instances.create_instance({
    'name': 'ml-training-1',
    'machine_type': 'n1-standard-4',
    'gpu_config': {'count': 1, 'type': 'nvidia-tesla-t4'}
})

# Create persistent disk
disk = mgr.disks.create_persistent_disk({
    'name': 'data-disk-1',
    'size_gb': 100,
    'disk_type': 'pd-ssd'
})

# Create snapshot
snapshot = mgr.snapshots.create_snapshot({
    'snapshot_name': 'data-backup-1',
    'source_disk': 'data-disk-1'
})

# Create instance template
template = mgr.templates.create_instance_template({
    'template_name': 'web-server-template',
    'machine_type': 'e2-medium',
    'tags': ['http-server', 'https-server']
})

# Create managed instance group
mig = mgr.mig.create_managed_instance_group({
    'group_name': 'web-servers',
    'template_name': 'web-server-template',
    'target_size': 3
})

# Configure autoscaling
autoscaling = mgr.mig.configure_autoscaling({
    'group_name': 'web-servers',
    'min_replicas': 2,
    'max_replicas': 10,
    'target_cpu_utilization': 0.7
})

# Create HTTP load balancer
lb = mgr.load_balancers.create_http_load_balancer({
    'lb_name': 'web-lb',
    'backend_service_name': 'web-backend',
    'instance_group': 'web-servers'
})
```

## Machine Types

### General Purpose
- **E2**: Cost-optimized (e2-micro, e2-small, e2-medium, e2-standard)
- **N1**: Balanced price/performance (n1-standard, n1-highmem, n1-highcpu)
- **N2**: Newer generation with better performance
- **N2D**: AMD EPYC processors

### Compute Optimized
- **C2**: Ultra-high performance (c2-standard-4, c2-standard-8)
- **C2D**: AMD EPYC 3rd gen

### Memory Optimized
- **M1**: Up to 4TB RAM (m1-ultramem, m1-megamem)
- **M2**: 2nd gen memory-optimized

### GPU Accelerated
- **NVIDIA Tesla T4**: ML inference and training
- **NVIDIA V100**: High-performance computing
- **NVIDIA A100**: Latest GPU for AI

## Disk Types

- **pd-standard**: HDD, 0.04 $/GB/month, lower performance
- **pd-balanced**: SSD, 0.10 $/GB/month, balanced price/performance
- **pd-ssd**: SSD, 0.17 $/GB/month, high performance
- **pd-extreme**: NVMe SSD, highest performance

## Best Practices

1. **Use preemptible instances** for fault-tolerant batch workloads (up to 80% savings)
2. **Enable autoscaling** for variable traffic patterns
3. **Create regular snapshots** for disaster recovery
4. **Use instance templates** for consistent deployments
5. **Implement health checks** for auto-healing
6. **Use load balancing** for high availability
7. **Right-size instances** to optimize costs
8. **Use regional MIGs** for multi-zone redundancy

## Requirements

```
google-cloud-compute
```

## Configuration

Set up authentication:
```bash
export GOOGLE_APPLICATION_CREDENTIALS="path/to/service-account-key.json"
```

## Cost Optimization

- **Sustained Use Discounts**: Automatic discounts for long-running instances
- **Committed Use Discounts**: Save up to 57% with 1-year or 3-year commitments
- **Preemptible VMs**: Up to 80% cheaper for interruptible workloads
- **Custom Machine Types**: Pay only for resources you need
- **Autoscaling**: Scale down during low traffic periods

## Performance Tips

- Use SSD persistent disks for I/O-intensive workloads
- Place instances in the same zone as data for lower latency
- Use local SSDs for highest IOPS (up to 2.4M IOPS)
- Enable live migration for zero-downtime maintenance
- Use network tags for firewall rules
- Configure appropriate health check intervals

## Author

BrillConsulting - Enterprise Cloud Solutions
