# GPU Scheduling & Scaling

Kubernetes-based GPU scheduling with NVIDIA GPU Operator for dynamic scaling and resource management.

## Features

- **Kubernetes Integration** - Native K8s GPU scheduling
- **NVIDIA GPU Operator** - GPU node management
- **Auto-scaling** - HPA based on GPU metrics
- **Multi-tenancy** - Share GPUs across workloads
- **MIG Support** - Multi-Instance GPU partitioning
- **Resource Quotas** - Fair GPU allocation
- **Job Prioritization** - Queue management
- **Cost Optimization** - Spot instance integration

## Architecture

```yaml
Kubernetes Cluster
├── GPU Node Pool 1 (A100)
│   ├── Inference Pods
│   └── Training Pods
├── GPU Node Pool 2 (V100)
│   └── Batch Jobs
└── NVIDIA GPU Operator
    ├── Device Plugin
    ├── DCGM Exporter
    └── GPU Feature Discovery
```

## Technologies

- Kubernetes 1.28+
- NVIDIA GPU Operator 23.9+
- Kueue (job queuing)
- Prometheus operator
- Helm charts
