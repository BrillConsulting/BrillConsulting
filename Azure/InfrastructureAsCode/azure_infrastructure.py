"""
Azure Infrastructure as Code
=============================

Infrastructure provisioning and management using Azure SDK and Terraform:
- Resource group management
- Virtual networks and subnets
- Virtual machines and scale sets
- Storage accounts
- Infrastructure templates

Author: Brill Consulting
"""

from typing import List, Dict, Optional
from datetime import datetime
import json


class AzureInfrastructure:
    """Azure infrastructure management."""

    def __init__(self, subscription_id: str, resource_group: str, location: str = "eastus"):
        """Initialize Azure infrastructure manager."""
        self.subscription_id = subscription_id
        self.resource_group = resource_group
        self.location = location
        self.resources = []

    def create_resource_group(self, tags: Optional[Dict] = None) -> Dict:
        """Create Azure resource group."""
        print(f"\n📦 Creating resource group: {self.resource_group}")

        resource_group = {
            "name": self.resource_group,
            "location": self.location,
            "tags": tags or {"environment": "dev", "project": "demo"},
            "properties": {
                "provisioningState": "Succeeded"
            }
        }

        print(f"✓ Resource group created in {self.location}")
        return resource_group

    def create_virtual_network(self, vnet_name: str, address_space: str = "10.0.0.0/16") -> Dict:
        """Create virtual network."""
        print(f"\n🌐 Creating virtual network: {vnet_name}")

        vnet = {
            "name": vnet_name,
            "type": "Microsoft.Network/virtualNetworks",
            "location": self.location,
            "properties": {
                "addressSpace": {
                    "addressPrefixes": [address_space]
                },
                "subnets": []
            }
        }

        self.resources.append(vnet)
        print(f"✓ Virtual network created with address space {address_space}")
        return vnet

    def create_subnet(self, vnet_name: str, subnet_name: str, address_prefix: str = "10.0.1.0/24") -> Dict:
        """Create subnet in virtual network."""
        print(f"\n📡 Creating subnet: {subnet_name}")

        subnet = {
            "name": subnet_name,
            "properties": {
                "addressPrefix": address_prefix
            }
        }

        # Find vnet and add subnet
        for resource in self.resources:
            if resource.get("name") == vnet_name:
                resource["properties"]["subnets"].append(subnet)
                break

        print(f"✓ Subnet created with address prefix {address_prefix}")
        return subnet

    def create_storage_account(self, storage_name: str, sku: str = "Standard_LRS") -> Dict:
        """Create storage account."""
        print(f"\n💾 Creating storage account: {storage_name}")

        storage = {
            "name": storage_name,
            "type": "Microsoft.Storage/storageAccounts",
            "location": self.location,
            "sku": {
                "name": sku
            },
            "kind": "StorageV2",
            "properties": {
                "supportsHttpsTrafficOnly": True,
                "encryption": {
                    "services": {
                        "blob": {"enabled": True},
                        "file": {"enabled": True}
                    }
                }
            }
        }

        self.resources.append(storage)
        print(f"✓ Storage account created with SKU {sku}")
        return storage

    def create_vm(self, vm_name: str, vm_size: str = "Standard_B2s", os_type: str = "Linux") -> Dict:
        """Create virtual machine."""
        print(f"\n🖥️  Creating virtual machine: {vm_name}")

        vm = {
            "name": vm_name,
            "type": "Microsoft.Compute/virtualMachines",
            "location": self.location,
            "properties": {
                "hardwareProfile": {
                    "vmSize": vm_size
                },
                "osProfile": {
                    "computerName": vm_name,
                    "adminUsername": "azureuser"
                },
                "storageProfile": {
                    "imageReference": {
                        "publisher": "Canonical" if os_type == "Linux" else "MicrosoftWindowsServer",
                        "offer": "UbuntuServer" if os_type == "Linux" else "WindowsServer",
                        "sku": "18.04-LTS" if os_type == "Linux" else "2019-Datacenter",
                        "version": "latest"
                    }
                }
            }
        }

        self.resources.append(vm)
        print(f"✓ Virtual machine created with size {vm_size}")
        return vm

    def create_app_service_plan(self, plan_name: str, tier: str = "Basic", size: str = "B1") -> Dict:
        """Create App Service plan."""
        print(f"\n📱 Creating App Service plan: {plan_name}")

        plan = {
            "name": plan_name,
            "type": "Microsoft.Web/serverfarms",
            "location": self.location,
            "sku": {
                "name": size,
                "tier": tier,
                "capacity": 1
            },
            "kind": "linux"
        }

        self.resources.append(plan)
        print(f"✓ App Service plan created with tier {tier}")
        return plan

    def create_web_app(self, app_name: str, plan_name: str, runtime: str = "PYTHON|3.9") -> Dict:
        """Create Web App."""
        print(f"\n🌐 Creating Web App: {app_name}")

        web_app = {
            "name": app_name,
            "type": "Microsoft.Web/sites",
            "location": self.location,
            "properties": {
                "serverFarmId": plan_name,
                "siteConfig": {
                    "linuxFxVersion": runtime,
                    "alwaysOn": True
                },
                "httpsOnly": True
            }
        }

        self.resources.append(web_app)
        print(f"✓ Web App created with runtime {runtime}")
        return web_app

    def generate_terraform_template(self) -> str:
        """Generate Terraform template for infrastructure."""
        print(f"\n📝 Generating Terraform template...")

        terraform = f"""
terraform {{
  required_providers {{
    azurerm = {{
      source  = "hashicorp/azurerm"
      version = "~> 3.0"
    }}
  }}
}}

provider "azurerm" {{
  features {{}}
  subscription_id = "{self.subscription_id}"
}}

resource "azurerm_resource_group" "main" {{
  name     = "{self.resource_group}"
  location = "{self.location}"

  tags = {{
    environment = "dev"
    project     = "demo"
  }}
}}
"""

        # Add resources
        for i, resource in enumerate(self.resources):
            resource_type = resource.get("type", "")

            if "virtualNetworks" in resource_type:
                terraform += f"""
resource "azurerm_virtual_network" "vnet_{i}" {{
  name                = "{resource['name']}"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  address_space       = {json.dumps(resource['properties']['addressSpace']['addressPrefixes'])}
}}
"""

            elif "storageAccounts" in resource_type:
                terraform += f"""
resource "azurerm_storage_account" "storage_{i}" {{
  name                     = "{resource['name']}"
  resource_group_name      = azurerm_resource_group.main.name
  location                 = azurerm_resource_group.main.location
  account_tier             = "Standard"
  account_replication_type = "LRS"
}}
"""

        print(f"✓ Terraform template generated with {len(self.resources)} resources")
        return terraform

    def export_arm_template(self) -> Dict:
        """Export ARM template."""
        print(f"\n📄 Exporting ARM template...")

        arm_template = {
            "$schema": "https://schema.management.azure.com/schemas/2019-04-01/deploymentTemplate.json#",
            "contentVersion": "1.0.0.0",
            "parameters": {},
            "resources": self.resources,
            "outputs": {}
        }

        print(f"✓ ARM template exported with {len(self.resources)} resources")
        return arm_template

    def get_deployment_summary(self) -> Dict:
        """Get deployment summary."""
        resource_types = {}
        for resource in self.resources:
            rtype = resource.get("type", "Unknown")
            resource_types[rtype] = resource_types.get(rtype, 0) + 1

        return {
            "subscription_id": self.subscription_id,
            "resource_group": self.resource_group,
            "location": self.location,
            "total_resources": len(self.resources),
            "resource_types": resource_types,
            "timestamp": datetime.now().isoformat()
        }


def demo():
    """Demo Azure infrastructure as code."""
    print("Azure Infrastructure as Code Demo")
    print("=" * 60)

    # Initialize infrastructure manager
    infra = AzureInfrastructure(
        subscription_id="12345678-1234-1234-1234-123456789abc",
        resource_group="rg-demo-app",
        location="eastus"
    )

    # Create resource group
    print("\n1. Resource Group")
    print("-" * 60)
    infra.create_resource_group(tags={"environment": "production", "cost-center": "IT"})

    # Create networking
    print("\n2. Virtual Network Infrastructure")
    print("-" * 60)
    infra.create_virtual_network("vnet-demo", "10.0.0.0/16")
    infra.create_subnet("vnet-demo", "subnet-web", "10.0.1.0/24")
    infra.create_subnet("vnet-demo", "subnet-db", "10.0.2.0/24")

    # Create storage
    print("\n3. Storage Account")
    print("-" * 60)
    infra.create_storage_account("stdemodemo123", "Standard_GRS")

    # Create VM
    print("\n4. Virtual Machine")
    print("-" * 60)
    infra.create_vm("vm-web-01", "Standard_D2s_v3", "Linux")

    # Create App Service
    print("\n5. App Service")
    print("-" * 60)
    infra.create_app_service_plan("plan-demo", "Standard", "S1")
    infra.create_web_app("app-demo-web", "plan-demo", "PYTHON|3.11")

    # Generate Terraform
    print("\n6. Generate Terraform Template")
    print("-" * 60)
    terraform = infra.generate_terraform_template()
    print(f"Generated {len(terraform.split('resource'))-1} Terraform resources")

    # Export ARM template
    print("\n7. Export ARM Template")
    print("-" * 60)
    arm = infra.export_arm_template()
    print(f"Exported {len(arm['resources'])} ARM resources")

    # Deployment summary
    print("\n8. Deployment Summary")
    print("-" * 60)
    summary = infra.get_deployment_summary()
    print(f"  Subscription: {summary['subscription_id']}")
    print(f"  Resource Group: {summary['resource_group']}")
    print(f"  Location: {summary['location']}")
    print(f"  Total Resources: {summary['total_resources']}")
    print(f"\n  Resource Types:")
    for rtype, count in summary['resource_types'].items():
        print(f"    • {rtype}: {count}")

    print("\n✓ Azure Infrastructure Demo Complete!")


if __name__ == '__main__':
    demo()
