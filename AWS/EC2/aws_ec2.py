"""
AWS EC2
=======

EC2 instance and auto-scaling management.

Author: Brill Consulting
"""

from typing import List, Dict
from datetime import datetime


class AWSEC2:
    """AWS EC2 management."""

    def __init__(self, region: str = "us-east-1"):
        self.region = region
        self.instances = {}
        self.security_groups = {}
        self.load_balancers = {}

    def launch_instance(self, name: str, instance_type: str = "t2.micro",
                       ami_id: str = "ami-12345678") -> Dict:
        """Launch EC2 instance."""
        print(f"\n🚀 Launching instance: {name}")
        print(f"   Type: {instance_type}")
        print(f"   AMI: {ami_id}")

        instance = {
            "instance_id": f"i-{datetime.now().timestamp()}",
            "name": name,
            "instance_type": instance_type,
            "ami_id": ami_id,
            "state": "running",
            "public_ip": "54.123.45.67",
            "launched_at": datetime.now().isoformat()
        }

        self.instances[instance["instance_id"]] = instance
        print(f"✓ Instance launched: {instance['instance_id']}")

        return instance

    def create_security_group(self, name: str, description: str) -> Dict:
        """Create security group."""
        print(f"\n🔒 Creating security group: {name}")

        sg = {
            "group_id": f"sg-{datetime.now().timestamp()}",
            "name": name,
            "description": description,
            "ingress_rules": [],
            "egress_rules": []
        }

        self.security_groups[sg["group_id"]] = sg
        print(f"✓ Security group created: {sg['group_id']}")

        return sg

    def create_auto_scaling_group(self, name: str, min_size: int, max_size: int) -> Dict:
        """Create auto-scaling group."""
        print(f"\n📊 Creating auto-scaling group: {name}")
        print(f"   Min: {min_size}, Max: {max_size}")

        asg = {
            "name": name,
            "min_size": min_size,
            "max_size": max_size,
            "desired_capacity": min_size,
            "instances": []
        }

        print(f"✓ Auto-scaling group created")
        return asg

    def get_summary(self) -> Dict:
        """Get EC2 summary."""
        return {
            "region": self.region,
            "instances": len(self.instances),
            "security_groups": len(self.security_groups)
        }


def demo():
    """Demo AWS EC2."""
    print("AWS EC2 Demo")
    print("=" * 60)

    ec2 = AWSEC2("us-east-1")

    ec2.launch_instance("web-server-1", "t2.medium")
    ec2.launch_instance("web-server-2", "t2.medium")

    ec2.create_security_group("web-sg", "Security group for web servers")
    ec2.create_auto_scaling_group("web-asg", 2, 10)

    print("\n📊 Summary:")
    summary = ec2.get_summary()
    for k, v in summary.items():
        print(f"  {k}: {v}")

    print("\n✓ AWS EC2 Demo Complete!")


if __name__ == '__main__':
    demo()
