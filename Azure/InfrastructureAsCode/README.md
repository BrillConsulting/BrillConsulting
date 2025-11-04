# Azure Infrastructure as Code

Infrastructure provisioning and management using Azure SDK and Terraform.

## Features

- Resource group management
- Virtual networks and subnets
- Virtual machines and scale sets
- Storage accounts
- App Service and Web Apps
- Terraform template generation
- ARM template export

## Usage

```python
from azure_infrastructure import AzureInfrastructure

infra = AzureInfrastructure(
    subscription_id="your-subscription-id",
    resource_group="rg-demo",
    location="eastus"
)

# Create resources
infra.create_resource_group()
infra.create_virtual_network("vnet-demo", "10.0.0.0/16")
infra.create_storage_account("stdemodemo123")

# Generate Terraform
terraform = infra.generate_terraform_template()
```

## Demo

```bash
python azure_infrastructure.py
```
