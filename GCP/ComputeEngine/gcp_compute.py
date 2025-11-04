"""
Google Cloud Compute Engine
============================

GCP compute resources management:
- VM instance management
- Instance templates
- Managed instance groups
- Load balancing
- Autoscaling

Author: Brill Consulting
"""

from typing import List, Dict, Optional
from datetime import datetime


class GCPComputeEngine:
    """GCP Compute Engine management."""

    def __init__(self, project_id: str, zone: str = "us-central1-a"):
        self.project_id = project_id
        self.zone = zone
        self.instances = {}
        self.templates = {}
        self.instance_groups = {}

    def create_instance(self, name: str, machine_type: str = "e2-medium",
                       image_family: str = "ubuntu-2004-lts") -> Dict:
        """Create VM instance."""
        print(f"\n🖥️  Creating instance: {name}")
        print(f"   Machine type: {machine_type}")
        print(f"   Image: {image_family}")

        instance = {
            "name": name,
            "zone": self.zone,
            "machine_type": f"zones/{self.zone}/machineTypes/{machine_type}",
            "disks": [{
                "boot": True,
                "autoDelete": True,
                "initializeParams": {
                    "sourceImage": f"projects/ubuntu-os-cloud/global/images/family/{image_family}"
                }
            }],
            "networkInterfaces": [{
                "network": "global/networks/default",
                "accessConfigs": [{"name": "External NAT", "type": "ONE_TO_ONE_NAT"}]
            }],
            "status": "RUNNING",
            "created_at": datetime.now().isoformat()
        }

        self.instances[name] = instance
        print(f"✓ Instance created: {name}")

        return instance

    def stop_instance(self, name: str) -> Dict:
        """Stop VM instance."""
        if name not in self.instances:
            return {"error": f"Instance {name} not found"}

        self.instances[name]["status"] = "TERMINATED"
        print(f"⏸️  Instance stopped: {name}")

        return {"status": "stopped", "instance": name}

    def start_instance(self, name: str) -> Dict:
        """Start VM instance."""
        if name not in self.instances:
            return {"error": f"Instance {name} not found"}

        self.instances[name]["status"] = "RUNNING"
        print(f"▶️  Instance started: {name}")

        return {"status": "running", "instance": name}

    def create_instance_template(self, name: str, machine_type: str = "e2-medium") -> Dict:
        """Create instance template."""
        print(f"\n📋 Creating instance template: {name}")

        template = {
            "name": name,
            "properties": {
                "machineType": machine_type,
                "disks": [{
                    "boot": True,
                    "initializeParams": {
                        "sourceImage": "projects/ubuntu-os-cloud/global/images/family/ubuntu-2004-lts"
                    }
                }],
                "networkInterfaces": [{
                    "network": "global/networks/default"
                }]
            },
            "created_at": datetime.now().isoformat()
        }

        self.templates[name] = template
        print(f"✓ Template created: {name}")

        return template

    def create_managed_instance_group(self, name: str, template_name: str,
                                     target_size: int = 3) -> Dict:
        """Create managed instance group."""
        print(f"\n👥 Creating managed instance group: {name}")
        print(f"   Template: {template_name}")
        print(f"   Target size: {target_size}")

        group = {
            "name": name,
            "zone": self.zone,
            "instanceTemplate": template_name,
            "targetSize": target_size,
            "instances": [],
            "status": "RUNNING",
            "created_at": datetime.now().isoformat()
        }

        # Simulate creating instances
        for i in range(target_size):
            instance_name = f"{name}-{i}"
            group["instances"].append(instance_name)

        self.instance_groups[name] = group
        print(f"✓ Managed instance group created with {target_size} instances")

        return group

    def configure_autoscaling(self, group_name: str, min_size: int = 1,
                            max_size: int = 10, target_cpu: float = 0.6) -> Dict:
        """Configure autoscaling for instance group."""
        print(f"\n📊 Configuring autoscaling for: {group_name}")

        autoscaler = {
            "name": f"{group_name}-autoscaler",
            "target": group_name,
            "autoscalingPolicy": {
                "minNumReplicas": min_size,
                "maxNumReplicas": max_size,
                "cpuUtilization": {
                    "utilizationTarget": target_cpu
                },
                "coolDownPeriodSec": 60
            }
        }

        print(f"✓ Autoscaling configured: {min_size}-{max_size} instances")
        return autoscaler

    def get_summary(self) -> Dict:
        """Get compute resources summary."""
        return {
            "project_id": self.project_id,
            "zone": self.zone,
            "instances": len(self.instances),
            "templates": len(self.templates),
            "instance_groups": len(self.instance_groups)
        }


def demo():
    """Demo GCP Compute Engine."""
    print("Google Cloud Compute Engine Demo")
    print("=" * 60)

    compute = GCPComputeEngine("my-gcp-project", "us-central1-a")

    # 1. Create VM instances
    print("\n1. Create VM Instances")
    print("-" * 60)

    compute.create_instance("web-server-1", "e2-medium", "ubuntu-2004-lts")
    compute.create_instance("web-server-2", "e2-standard-2", "ubuntu-2004-lts")

    # 2. Instance operations
    print("\n2. Instance Operations")
    print("-" * 60)

    compute.stop_instance("web-server-2")
    compute.start_instance("web-server-2")

    # 3. Instance template
    print("\n3. Instance Template")
    print("-" * 60)

    compute.create_instance_template("web-server-template", "e2-medium")

    # 4. Managed instance group
    print("\n4. Managed Instance Group")
    print("-" * 60)

    compute.create_managed_instance_group("web-server-group", "web-server-template", 3)

    # 5. Autoscaling
    print("\n5. Autoscaling Configuration")
    print("-" * 60)

    compute.configure_autoscaling("web-server-group", min_size=2, max_size=10, target_cpu=0.7)

    # 6. Summary
    print("\n6. Compute Resources Summary")
    print("-" * 60)

    summary = compute.get_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")

    print("\n✓ GCP Compute Engine Demo Complete!")


if __name__ == '__main__':
    demo()
