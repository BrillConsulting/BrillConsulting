# GCP Compute Engine

VM instance and compute resources management.

## Features

- VM instance creation and management
- Instance templates
- Managed instance groups
- Load balancing
- Autoscaling configuration

## Usage

```python
from gcp_compute import GCPComputeEngine

compute = GCPComputeEngine("my-project", "us-central1-a")
compute.create_instance("web-server-1", "e2-medium")
compute.create_managed_instance_group("web-group", "template", 3)
```

## Demo

```bash
python gcp_compute.py
```
