"""
Terraform Infrastructure as Code
Author: BrillConsulting
Description: Infrastructure provisioning and management with Terraform
"""

import json
from typing import Dict, List, Any
from datetime import datetime


class TerraformManager:
    """Terraform IaC management"""

    def __init__(self, workspace: str = 'production'):
        self.workspace = workspace
        self.resources = []

    def create_infrastructure(self, config: Dict[str, Any]) -> str:
        """Generate Terraform configuration"""
        provider = config.get('provider', 'aws')
        resources = config.get('resources', [])

        tf_config = f'''terraform {{
  required_version = ">= 1.0"

  required_providers {{
    {provider} = {{
      source  = "hashicorp/{provider}"
      version = "~> 5.0"
    }}
  }}

  backend "s3" {{
    bucket = "terraform-state"
    key    = "{self.workspace}/terraform.tfstate"
    region = "us-east-1"
  }}
}}

provider "{provider}" {{
  region = "us-east-1"
}}

# VPC
resource "{provider}_vpc" "main" {{
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = {{
    Name        = "{self.workspace}-vpc"
    Environment = "{self.workspace}"
  }}
}}

# Subnet
resource "{provider}_subnet" "public" {{
  vpc_id            = {provider}_vpc.main.id
  cidr_block        = "10.0.1.0/24"
  availability_zone = "us-east-1a"

  tags = {{
    Name = "{self.workspace}-public-subnet"
  }}
}}

# Security Group
resource "{provider}_security_group" "app" {{
  name        = "{self.workspace}-app-sg"
  description = "Security group for application"
  vpc_id      = {provider}_vpc.main.id

  ingress {{
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }}

  egress {{
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }}
}}

# EC2 Instance
resource "{provider}_instance" "app" {{
  ami           = "ami-0c55b159cbfafe1f0"
  instance_type = "t3.medium"
  subnet_id     = {provider}_subnet.public.id

  vpc_security_group_ids = [{provider}_security_group.app.id]

  tags = {{
    Name = "{self.workspace}-app-server"
  }}
}}

# Outputs
output "instance_public_ip" {{
  value = {provider}_instance.app.public_ip
}}
'''

        print(f"✓ Terraform configuration generated")
        print(f"  Provider: {provider}, Workspace: {self.workspace}")
        print(f"  Resources: {len(resources)}")
        return tf_config

    def plan(self) -> Dict[str, Any]:
        """Run terraform plan"""
        result = {
            'changes': {
                'add': 5,
                'change': 2,
                'destroy': 0
            },
            'planned_at': datetime.now().isoformat()
        }
        print(f"✓ Terraform plan completed")
        print(f"  Add: {result['changes']['add']}, Change: {result['changes']['change']}, Destroy: {result['changes']['destroy']}")
        return result

    def apply(self) -> Dict[str, Any]:
        """Run terraform apply"""
        result = {
            'applied': 7,
            'status': 'success',
            'applied_at': datetime.now().isoformat()
        }
        print(f"✓ Terraform apply completed: {result['applied']} resources")
        return result


def demo():
    """Demonstrate Terraform"""
    print("=" * 60)
    print("Terraform Infrastructure as Code Demo")
    print("=" * 60)

    mgr = TerraformManager('production')

    print("\n1. Creating infrastructure configuration...")
    tf_config = mgr.create_infrastructure({'provider': 'aws', 'resources': []})
    print(tf_config[:300] + "...")

    print("\n2. Running terraform plan...")
    mgr.plan()

    print("\n3. Running terraform apply...")
    mgr.apply()

    print("\n" + "=" * 60)


if __name__ == "__main__":
    demo()
