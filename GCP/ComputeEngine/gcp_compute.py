"""
Google Cloud Compute Engine - Advanced VM Management
Author: Brill Consulting
Description: Comprehensive compute infrastructure with instances, disks, snapshots, networking, and load balancing
"""

import json
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta


class InstanceManager:
    """Manage VM instances"""

    def __init__(self, project_id: str, zone: str = 'us-central1-a'):
        """Initialize instance manager"""
        self.project_id = project_id
        self.zone = zone
        self.instances = {}

    def create_instance(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create VM instance

        Args:
            config: Instance configuration

        Returns:
            Instance creation result
        """
        print(f"\n{'='*60}")
        print("Creating VM Instance")
        print(f"{'='*60}")

        name = config.get('name', 'instance-1')
        machine_type = config.get('machine_type', 'e2-medium')
        image = config.get('image', 'projects/debian-cloud/global/images/family/debian-11')
        preemptible = config.get('preemptible', False)
        gpu_config = config.get('gpu_config', None)

        code = f"""
from google.cloud import compute_v1

compute_client = compute_v1.InstancesClient()

# Define instance configuration
instance = compute_v1.Instance()
instance.name = "{name}"
instance.machine_type = f"zones/{self.zone}/machineTypes/{machine_type}"

# Boot disk
disk = compute_v1.AttachedDisk()
disk.boot = True
disk.auto_delete = True
initialize_params = compute_v1.AttachedDiskInitializeParams()
initialize_params.source_image = "{image}"
initialize_params.disk_size_gb = {config.get('disk_size_gb', 10)}
disk.initialize_params = initialize_params
instance.disks = [disk]

# Network interface
network_interface = compute_v1.NetworkInterface()
network_interface.name = "global/networks/default"
access_config = compute_v1.AccessConfig()
access_config.name = "External NAT"
access_config.type_ = "ONE_TO_ONE_NAT"
network_interface.access_configs = [access_config]
instance.network_interfaces = [network_interface]

# Preemptible configuration
{'if ' + str(preemptible) + ':' if preemptible else '# Not preemptible'}
    scheduling = compute_v1.Scheduling()
    scheduling.preemptible = True
    instance.scheduling = scheduling

# GPU configuration
{'if gpu_config:' if gpu_config else '# No GPU'}
    accelerator = compute_v1.AcceleratorConfig()
    accelerator.accelerator_count = {gpu_config.get('count', 1) if gpu_config else 0}
    accelerator.accelerator_type = f"zones/{self.zone}/acceleratorTypes/{gpu_config.get('type', 'nvidia-tesla-t4') if gpu_config else 'nvidia-tesla-t4'}"
    instance.guest_accelerators = [accelerator]

# Create instance
operation = compute_client.insert(
    project="{self.project_id}",
    zone="{self.zone}",
    instance_resource=instance
)

# Wait for operation
operation.result()
print(f"Instance created: {name}")
"""

        result = {
            'name': name,
            'zone': self.zone,
            'machine_type': machine_type,
            'image': image,
            'preemptible': preemptible,
            'gpu_config': gpu_config,
            'status': 'RUNNING',
            'code': code,
            'timestamp': datetime.now().isoformat()
        }

        self.instances[name] = result

        print(f"✓ Instance created: {name}")
        print(f"  Machine type: {machine_type}")
        print(f"  Preemptible: {preemptible}")
        if gpu_config:
            print(f"  GPU: {gpu_config.get('count', 1)}x {gpu_config.get('type', 'nvidia-tesla-t4')}")
        print(f"{'='*60}")

        return result

    def create_instance_with_startup_script(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create instance with startup script"""
        print(f"\n{'='*60}")
        print("Creating Instance with Startup Script")
        print(f"{'='*60}")

        name = config.get('name', 'instance-1')
        startup_script = config.get('startup_script', '#!/bin/bash\napt-get update')

        code = f"""
from google.cloud import compute_v1

compute_client = compute_v1.InstancesClient()

# Instance with startup script
instance = compute_v1.Instance()
instance.name = "{name}"
instance.machine_type = f"zones/{self.zone}/machineTypes/e2-medium"

# Metadata with startup script
metadata = compute_v1.Metadata()
metadata.items = [
    compute_v1.Items(
        key="startup-script",
        value='''{startup_script}'''
    )
]
instance.metadata = metadata

# ... rest of instance configuration ...

operation = compute_client.insert(
    project="{self.project_id}",
    zone="{self.zone}",
    instance_resource=instance
)

operation.result()
print(f"Instance with startup script created: {name}")
"""

        result = {
            'name': name,
            'startup_script': startup_script,
            'code': code,
            'timestamp': datetime.now().isoformat()
        }

        print(f"✓ Instance with startup script: {name}")
        print(f"{'='*60}")

        return result


class DiskManager:
    """Manage persistent disks"""

    def __init__(self, project_id: str, zone: str = 'us-central1-a'):
        """Initialize disk manager"""
        self.project_id = project_id
        self.zone = zone
        self.disks = {}

    def create_persistent_disk(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create persistent disk"""
        print(f"\n{'='*60}")
        print("Creating Persistent Disk")
        print(f"{'='*60}")

        name = config.get('name', 'data-disk')
        size_gb = config.get('size_gb', 100)
        disk_type = config.get('disk_type', 'pd-standard')

        code = f"""
from google.cloud import compute_v1

disks_client = compute_v1.DisksClient()

# Define disk
disk = compute_v1.Disk()
disk.name = "{name}"
disk.size_gb = {size_gb}
disk.type_ = f"zones/{self.zone}/diskTypes/{disk_type}"

# Create disk
operation = disks_client.insert(
    project="{self.project_id}",
    zone="{self.zone}",
    disk_resource=disk
)

operation.result()
print(f"Persistent disk created: {name}")
"""

        result = {
            'name': name,
            'size_gb': size_gb,
            'disk_type': disk_type,
            'zone': self.zone,
            'code': code,
            'timestamp': datetime.now().isoformat()
        }

        self.disks[name] = result

        print(f"✓ Persistent disk created: {name}")
        print(f"  Size: {size_gb} GB")
        print(f"  Type: {disk_type}")
        print(f"{'='*60}")

        return result

    def attach_disk(self, instance_name: str, disk_name: str) -> str:
        """Attach disk to instance"""
        code = f"""
from google.cloud import compute_v1

compute_client = compute_v1.InstancesClient()

# Attach disk
attached_disk = compute_v1.AttachedDisk()
attached_disk.source = f"zones/{self.zone}/disks/{disk_name}"
attached_disk.device_name = "{disk_name}"
attached_disk.mode = "READ_WRITE"

operation = compute_client.attach_disk(
    project="{self.project_id}",
    zone="{self.zone}",
    instance="{instance_name}",
    attached_disk_resource=attached_disk
)

operation.result()
print(f"Disk {disk_name} attached to {instance_name}")
"""

        print(f"\n✓ Disk attach code generated: {disk_name} → {instance_name}")
        return code


class SnapshotManager:
    """Manage disk snapshots"""

    def __init__(self, project_id: str):
        """Initialize snapshot manager"""
        self.project_id = project_id
        self.snapshots = {}

    def create_snapshot(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create disk snapshot"""
        print(f"\n{'='*60}")
        print("Creating Disk Snapshot")
        print(f"{'='*60}")

        snapshot_name = config.get('snapshot_name', 'snapshot-1')
        source_disk = config.get('source_disk', 'data-disk')
        zone = config.get('zone', 'us-central1-a')

        code = f"""
from google.cloud import compute_v1

disks_client = compute_v1.DisksClient()
snapshots_client = compute_v1.SnapshotsClient()

# Create snapshot
snapshot = compute_v1.Snapshot()
snapshot.name = "{snapshot_name}"
snapshot.source_disk = f"zones/{zone}/disks/{source_disk}"

operation = snapshots_client.insert(
    project="{self.project_id}",
    snapshot_resource=snapshot
)

operation.result()
print(f"Snapshot created: {snapshot_name}")
"""

        result = {
            'snapshot_name': snapshot_name,
            'source_disk': source_disk,
            'zone': zone,
            'code': code,
            'timestamp': datetime.now().isoformat()
        }

        self.snapshots[snapshot_name] = result

        print(f"✓ Snapshot created: {snapshot_name}")
        print(f"  Source disk: {source_disk}")
        print(f"{'='*60}")

        return result

    def create_disk_from_snapshot(self, config: Dict[str, Any]) -> str:
        """Create disk from snapshot"""
        disk_name = config.get('disk_name', 'restored-disk')
        snapshot_name = config.get('snapshot_name', 'snapshot-1')
        zone = config.get('zone', 'us-central1-a')

        code = f"""
from google.cloud import compute_v1

disks_client = compute_v1.DisksClient()

# Create disk from snapshot
disk = compute_v1.Disk()
disk.name = "{disk_name}"
disk.source_snapshot = f"global/snapshots/{snapshot_name}"

operation = disks_client.insert(
    project="{self.project_id}",
    zone="{zone}",
    disk_resource=disk
)

operation.result()
print(f"Disk created from snapshot: {disk_name}")
"""

        print(f"\n✓ Disk from snapshot code: {snapshot_name} → {disk_name}")
        return code


class InstanceTemplateManager:
    """Manage instance templates"""

    def __init__(self, project_id: str):
        """Initialize template manager"""
        self.project_id = project_id
        self.templates = {}

    def create_instance_template(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create instance template"""
        print(f"\n{'='*60}")
        print("Creating Instance Template")
        print(f"{'='*60}")

        template_name = config.get('template_name', 'template-1')
        machine_type = config.get('machine_type', 'e2-medium')
        image = config.get('image', 'projects/debian-cloud/global/images/family/debian-11')
        tags = config.get('tags', ['http-server', 'https-server'])

        code = f"""
from google.cloud import compute_v1

templates_client = compute_v1.InstanceTemplatesClient()

# Define template
template = compute_v1.InstanceTemplate()
template.name = "{template_name}"

# Instance properties
instance_properties = compute_v1.InstanceProperties()
instance_properties.machine_type = "{machine_type}"

# Boot disk
disk = compute_v1.AttachedDisk()
disk.boot = True
disk.auto_delete = True
initialize_params = compute_v1.AttachedDiskInitializeParams()
initialize_params.source_image = "{image}"
disk.initialize_params = initialize_params
instance_properties.disks = [disk]

# Network
network_interface = compute_v1.NetworkInterface()
network_interface.network = "global/networks/default"
instance_properties.network_interfaces = [network_interface]

# Tags
instance_properties.tags = compute_v1.Tags(items={tags})

template.properties = instance_properties

# Create template
operation = templates_client.insert(
    project="{self.project_id}",
    instance_template_resource=template
)

operation.result()
print(f"Instance template created: {template_name}")
"""

        result = {
            'template_name': template_name,
            'machine_type': machine_type,
            'image': image,
            'tags': tags,
            'code': code,
            'timestamp': datetime.now().isoformat()
        }

        self.templates[template_name] = result

        print(f"✓ Instance template: {template_name}")
        print(f"  Machine type: {machine_type}")
        print(f"  Tags: {', '.join(tags)}")
        print(f"{'='*60}")

        return result


class ManagedInstanceGroupManager:
    """Manage instance groups with autoscaling"""

    def __init__(self, project_id: str, zone: str = 'us-central1-a'):
        """Initialize MIG manager"""
        self.project_id = project_id
        self.zone = zone
        self.instance_groups = {}

    def create_managed_instance_group(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create managed instance group"""
        print(f"\n{'='*60}")
        print("Creating Managed Instance Group")
        print(f"{'='*60}")

        group_name = config.get('group_name', 'instance-group-1')
        template_name = config.get('template_name', 'template-1')
        target_size = config.get('target_size', 3)
        base_instance_name = config.get('base_instance_name', 'instance')

        code = f"""
from google.cloud import compute_v1

mig_client = compute_v1.InstanceGroupManagersClient()

# Define MIG
mig = compute_v1.InstanceGroupManager()
mig.name = "{group_name}"
mig.base_instance_name = "{base_instance_name}"
mig.instance_template = f"global/instanceTemplates/{template_name}"
mig.target_size = {target_size}

# Create MIG
operation = mig_client.insert(
    project="{self.project_id}",
    zone="{self.zone}",
    instance_group_manager_resource=mig
)

operation.result()
print(f"Managed instance group created: {group_name}")
"""

        result = {
            'group_name': group_name,
            'template_name': template_name,
            'target_size': target_size,
            'zone': self.zone,
            'code': code,
            'timestamp': datetime.now().isoformat()
        }

        self.instance_groups[group_name] = result

        print(f"✓ MIG created: {group_name}")
        print(f"  Template: {template_name}")
        print(f"  Target size: {target_size}")
        print(f"{'='*60}")

        return result

    def configure_autoscaling(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Configure autoscaling for MIG"""
        print(f"\n{'='*60}")
        print("Configuring Autoscaling")
        print(f"{'='*60}")

        group_name = config.get('group_name', 'instance-group-1')
        min_replicas = config.get('min_replicas', 1)
        max_replicas = config.get('max_replicas', 10)
        target_cpu = config.get('target_cpu_utilization', 0.6)

        code = f"""
from google.cloud import compute_v1

autoscalers_client = compute_v1.AutoscalersClient()

# Define autoscaler
autoscaler = compute_v1.Autoscaler()
autoscaler.name = f"{group_name}-autoscaler"
autoscaler.target = f"zones/{self.zone}/instanceGroupManagers/{group_name}"

# Autoscaling policy
policy = compute_v1.AutoscalingPolicy()
policy.min_num_replicas = {min_replicas}
policy.max_num_replicas = {max_replicas}
policy.cool_down_period_sec = 60

# CPU utilization
cpu_utilization = compute_v1.AutoscalingPolicyCpuUtilization()
cpu_utilization.utilization_target = {target_cpu}
policy.cpu_utilization = cpu_utilization

autoscaler.autoscaling_policy = policy

# Create autoscaler
operation = autoscalers_client.insert(
    project="{self.project_id}",
    zone="{self.zone}",
    autoscaler_resource=autoscaler
)

operation.result()
print(f"Autoscaling configured for {group_name}")
"""

        result = {
            'group_name': group_name,
            'min_replicas': min_replicas,
            'max_replicas': max_replicas,
            'target_cpu_utilization': target_cpu,
            'code': code,
            'timestamp': datetime.now().isoformat()
        }

        print(f"✓ Autoscaling configured: {group_name}")
        print(f"  Range: {min_replicas}-{max_replicas} instances")
        print(f"  Target CPU: {target_cpu * 100}%")
        print(f"{'='*60}")

        return result


class LoadBalancerManager:
    """Manage HTTP(S) load balancers"""

    def __init__(self, project_id: str):
        """Initialize load balancer manager"""
        self.project_id = project_id
        self.load_balancers = {}

    def create_http_load_balancer(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create HTTP load balancer"""
        print(f"\n{'='*60}")
        print("Creating HTTP Load Balancer")
        print(f"{'='*60}")

        lb_name = config.get('lb_name', 'http-lb')
        backend_service_name = config.get('backend_service_name', 'backend-service')
        instance_group = config.get('instance_group', 'instance-group-1')

        code = f"""
from google.cloud import compute_v1

# 1. Create backend service
backend_services_client = compute_v1.BackendServicesClient()

backend_service = compute_v1.BackendService()
backend_service.name = "{backend_service_name}"
backend_service.protocol = "HTTP"
backend_service.load_balancing_scheme = "EXTERNAL"

# Add backend (instance group)
backend = compute_v1.Backend()
backend.group = f"zones/us-central1-a/instanceGroups/{instance_group}"
backend.balancing_mode = "UTILIZATION"
backend.max_utilization = 0.8
backend_service.backends = [backend]

# Health check
health_check = compute_v1.HealthCheck()
health_check.name = "http-health-check"
health_check.type_ = "HTTP"
health_check.http_health_check = compute_v1.HTTPHealthCheck(
    port=80,
    request_path="/"
)

# 2. Create URL map
url_maps_client = compute_v1.UrlMapsClient()

url_map = compute_v1.UrlMap()
url_map.name = "{lb_name}-url-map"
url_map.default_service = f"global/backendServices/{backend_service_name}"

# 3. Create target HTTP proxy
target_proxies_client = compute_v1.TargetHttpProxiesClient()

proxy = compute_v1.TargetHttpProxy()
proxy.name = "{lb_name}-proxy"
proxy.url_map = f"global/urlMaps/{lb_name}-url-map"

# 4. Create forwarding rule
forwarding_rules_client = compute_v1.GlobalForwardingRulesClient()

forwarding_rule = compute_v1.ForwardingRule()
forwarding_rule.name = "{lb_name}-forwarding-rule"
forwarding_rule.ip_protocol = "TCP"
forwarding_rule.port_range = "80"
forwarding_rule.target = f"global/targetHttpProxies/{lb_name}-proxy"

print(f"HTTP Load Balancer created: {lb_name}")
"""

        result = {
            'lb_name': lb_name,
            'backend_service': backend_service_name,
            'instance_group': instance_group,
            'protocol': 'HTTP',
            'code': code,
            'timestamp': datetime.now().isoformat()
        }

        self.load_balancers[lb_name] = result

        print(f"✓ HTTP Load Balancer: {lb_name}")
        print(f"  Backend: {backend_service_name}")
        print(f"  Instance group: {instance_group}")
        print(f"{'='*60}")

        return result


class ComputeEngineManager:
    """Comprehensive Compute Engine management"""

    def __init__(self, project_id: str = 'my-project', zone: str = 'us-central1-a'):
        """
        Initialize Compute Engine manager

        Args:
            project_id: GCP project ID
            zone: Default zone
        """
        self.project_id = project_id
        self.zone = zone
        self.instances = InstanceManager(project_id, zone)
        self.disks = DiskManager(project_id, zone)
        self.snapshots = SnapshotManager(project_id)
        self.templates = InstanceTemplateManager(project_id)
        self.mig = ManagedInstanceGroupManager(project_id, zone)
        self.load_balancers = LoadBalancerManager(project_id)

    def get_manager_info(self) -> Dict[str, Any]:
        """Get manager information"""
        return {
            'project_id': self.project_id,
            'zone': self.zone,
            'instances': len(self.instances.instances),
            'disks': len(self.disks.disks),
            'snapshots': len(self.snapshots.snapshots),
            'templates': len(self.templates.templates),
            'instance_groups': len(self.mig.instance_groups),
            'load_balancers': len(self.load_balancers.load_balancers),
            'features': [
                'vm_instances',
                'persistent_disks',
                'snapshots',
                'instance_templates',
                'managed_instance_groups',
                'autoscaling',
                'load_balancing',
                'gpu_support',
                'preemptible_instances'
            ],
            'timestamp': datetime.now().isoformat()
        }


def demo():
    """Demonstrate Compute Engine capabilities"""
    print("=" * 60)
    print("Compute Engine Comprehensive Demo")
    print("=" * 60)

    project_id = 'my-gcp-project'
    zone = 'us-central1-a'

    # Initialize manager
    mgr = ComputeEngineManager(project_id, zone)

    # Create regular instance
    instance = mgr.instances.create_instance({
        'name': 'web-server-1',
        'machine_type': 'e2-medium',
        'disk_size_gb': 20
    })

    # Create preemptible instance
    preemptible = mgr.instances.create_instance({
        'name': 'batch-worker-1',
        'machine_type': 'n1-standard-4',
        'preemptible': True
    })

    # Create GPU instance
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
        'source_disk': 'data-disk-1',
        'zone': zone
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

    # Create load balancer
    lb = mgr.load_balancers.create_http_load_balancer({
        'lb_name': 'web-lb',
        'backend_service_name': 'web-backend',
        'instance_group': 'web-servers'
    })

    # Manager info
    info = mgr.get_manager_info()
    print(f"\n{'='*60}")
    print("Compute Engine Manager Summary")
    print(f"{'='*60}")
    print(f"Project: {info['project_id']}")
    print(f"Zone: {info['zone']}")
    print(f"Instances: {info['instances']}")
    print(f"Disks: {info['disks']}")
    print(f"Snapshots: {info['snapshots']}")
    print(f"Templates: {info['templates']}")
    print(f"Instance groups: {info['instance_groups']}")
    print(f"Load balancers: {info['load_balancers']}")
    print(f"Features: {', '.join(info['features'])}")
    print(f"{'='*60}")

    print("\n✓ Demo completed successfully!")
    print("\nCompute Engine Best Practices:")
    print("  1. Use preemptible instances for batch workloads")
    print("  2. Enable autoscaling for variable traffic")
    print("  3. Create regular snapshots for disaster recovery")
    print("  4. Use instance templates for consistency")
    print("  5. Implement health checks for reliability")
    print("  6. Use load balancing for high availability")


if __name__ == "__main__":
    demo()
