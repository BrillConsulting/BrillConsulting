# Azure Infrastructure as Code

Comprehensive implementation of Azure infrastructure provisioning using Azure SDK, Terraform, and ARM templates.

**Author:** BrillConsulting
**Contact:** clientbrill@gmail.com
**LinkedIn:** [brillconsulting](https://www.linkedin.com/in/brillconsulting)

## Overview

This project provides a complete Python implementation for Azure Infrastructure as Code, featuring resource provisioning, Terraform template generation, ARM template management, and infrastructure automation. Built for enterprise infrastructure management with code-based deployment and version control.

## Features

### Resource Management
- **Resource Groups**: Create and manage resource groups
- **Virtual Networks**: Configure VNets and subnets
- **Virtual Machines**: Provision and configure VMs
- **Storage Accounts**: Create and manage storage
- **App Services**: Deploy web applications
- **Container Infrastructure**: AKS clusters and ACR

### Network Configuration
- **Virtual Networks**: VNet creation and configuration
- **Subnets**: Subnet design and implementation
- **Network Security Groups**: Firewall rules
- **Load Balancers**: Traffic distribution
- **Application Gateway**: Web application firewall
- **VPN Gateway**: Site-to-site connectivity

### Infrastructure as Code
- **Terraform Templates**: Generate Terraform HCL
- **ARM Templates**: Export ARM JSON templates
- **Bicep Support**: Modern IaC syntax
- **State Management**: Track infrastructure state
- **Version Control**: Git-based infrastructure

### Automation Features
- **Resource Tagging**: Organize and track resources
- **Cost Management**: Budget and cost tracking
- **RBAC**: Role-based access control
- **Policy Management**: Compliance and governance
- **Resource Locking**: Prevent accidental deletion

### Advanced Capabilities
- **Multi-Region Deployment**: Global infrastructure
- **High Availability**: Availability zones and sets
- **Disaster Recovery**: Backup and replication
- **Scaling**: Auto-scaling configuration
- **Monitoring**: Azure Monitor integration

## Architecture

```
InfrastructureAsCode/
├── azure_infrastructure.py    # Main implementation
├── templates/                 # IaC templates
│   ├── terraform/
│   ├── arm/
│   └── bicep/
├── requirements.txt           # Dependencies
└── README.md                  # Documentation
```

## Installation

```bash
# Clone the repository
git clone https://github.com/BrillConsulting/BrillConsulting.git
cd BrillConsulting/Azure/InfrastructureAsCode

# Install dependencies
pip install -r requirements.txt
```

## Configuration

```python
from azure_infrastructure import AzureInfrastructure

infra = AzureInfrastructure(
    subscription_id="your-subscription-id",
    resource_group="rg-demo",
    location="eastus"
)
```

## Usage Examples

### Resource Group Management

```python
# Create resource group
infra.create_resource_group(
    name="rg-production",
    location="eastus",
    tags={
        "environment": "production",
        "project": "web-app",
        "cost-center": "engineering"
    }
)

# List resource groups
groups = infra.list_resource_groups()
for group in groups:
    print(f"{group.name}: {group.location}")
```

### Virtual Network Setup

```python
# Create virtual network
vnet = infra.create_virtual_network(
    name="vnet-prod",
    address_space="10.0.0.0/16",
    location="eastus"
)

# Add subnets
infra.create_subnet(
    vnet_name="vnet-prod",
    subnet_name="subnet-web",
    address_prefix="10.0.1.0/24"
)

infra.create_subnet(
    vnet_name="vnet-prod",
    subnet_name="subnet-data",
    address_prefix="10.0.2.0/24"
)
```

### Virtual Machine Deployment

```python
# Create VM
vm = infra.create_virtual_machine(
    name="vm-web-01",
    vm_size="Standard_D2s_v3",
    image="UbuntuLTS",
    admin_username="azureuser",
    ssh_public_key="ssh-rsa AAAA...",
    subnet_id=subnet.id
)

# Create VM scale set
vmss = infra.create_vm_scale_set(
    name="vmss-web",
    vm_size="Standard_B2s",
    instance_count=3,
    min_instances=2,
    max_instances=10
)
```

### Storage Account Creation

```python
# Create storage account
storage = infra.create_storage_account(
    name="stproddemo123",
    sku="Standard_LRS",
    kind="StorageV2",
    tier="Hot"
)

# Create blob container
container = storage.create_container("data")
```

### App Service Deployment

```python
# Create App Service Plan
plan = infra.create_app_service_plan(
    name="asp-production",
    sku="P1v2",
    capacity=2
)

# Create Web App
webapp = infra.create_web_app(
    name="webapp-prod-unique",
    app_service_plan_id=plan.id,
    runtime="python|3.11"
)
```

### Terraform Template Generation

```python
# Generate Terraform configuration
terraform = infra.generate_terraform_template()

print(terraform)
# Output:
# terraform {
#   required_providers {
#     azurerm = {
#       source  = "hashicorp/azurerm"
#       version = "~> 3.0"
#     }
#   }
# }
#
# resource "azurerm_resource_group" "main" {
#   name     = "rg-demo"
#   location = "eastus"
# }

# Save to file
with open("main.tf", "w") as f:
    f.write(terraform)
```

### ARM Template Export

```python
# Export ARM template
arm_template = infra.export_arm_template()

# Save template
with open("template.json", "w") as f:
    json.dump(arm_template, f, indent=2)
```

### Container Infrastructure

```python
# Create Container Registry
acr = infra.create_container_registry(
    name="acrproddemo",
    sku="Premium",
    admin_enabled=True
)

# Create AKS cluster
aks = infra.create_aks_cluster(
    name="aks-production",
    node_count=3,
    node_vm_size="Standard_D4s_v3",
    kubernetes_version="1.27"
)
```

### Network Security

```python
# Create Network Security Group
nsg = infra.create_network_security_group(
    name="nsg-web"
)

# Add security rules
nsg.add_rule(
    name="allow-https",
    priority=100,
    direction="Inbound",
    access="Allow",
    protocol="Tcp",
    source_port_range="*",
    destination_port_range="443",
    source_address_prefix="*",
    destination_address_prefix="*"
)
```

## Running Demos

```bash
# Run all demo functions
python azure_infrastructure.py
```

## API Reference

### AzureInfrastructure

**`create_resource_group(name, location, tags)`** - Create resource group

**`create_virtual_network(name, address_space, location)`** - Create VNet

**`create_virtual_machine(name, vm_size, image, admin_username)`** - Create VM

**`create_storage_account(name, sku, kind, tier)`** - Create storage

**`generate_terraform_template()`** - Generate Terraform

**`export_arm_template()`** - Export ARM template

## Best Practices

### 1. Use Tags for Organization
```python
infra.create_resource_group(
    name="rg-prod",
    tags={
        "environment": "production",
        "cost-center": "engineering",
        "owner": "platform-team"
    }
)
```

### 2. Implement Resource Locks
```python
# Prevent deletion
infra.create_resource_lock(
    resource_group="rg-prod",
    lock_name="prevent-delete",
    lock_level="CanNotDelete"
)
```

### 3. Use Naming Conventions
```python
# Follow Azure naming conventions
naming_config = {
    "resource_group": "rg-{env}-{app}",
    "virtual_network": "vnet-{env}-{app}",
    "storage_account": "st{env}{app}{random}"
}
```

### 4. Enable Diagnostics
```python
# Enable diagnostic settings
vm.enable_diagnostics(
    storage_account=diag_storage,
    log_analytics_workspace=workspace
)
```

### 5. Implement RBAC
```python
# Assign roles
infra.assign_role(
    principal_id="user-object-id",
    role="Contributor",
    scope="/subscriptions/{sub-id}/resourceGroups/rg-prod"
)
```

## Use Cases

### 1. Multi-Tier Web Application
```python
# Create infrastructure
infra.create_virtual_network("vnet-app", "10.0.0.0/16")
infra.create_subnet("vnet-app", "subnet-web", "10.0.1.0/24")
infra.create_subnet("vnet-app", "subnet-app", "10.0.2.0/24")
infra.create_subnet("vnet-app", "subnet-data", "10.0.3.0/24")

# Deploy tiers
web_tier = infra.create_vm_scale_set("vmss-web", instances=3)
app_tier = infra.create_vm_scale_set("vmss-app", instances=5)
data_tier = infra.create_sql_database("sql-db")
```

### 2. Microservices Platform
```python
# AKS cluster
aks = infra.create_aks_cluster("aks-prod", node_count=5)

# Container registry
acr = infra.create_container_registry("acrprod")

# Service mesh
infra.enable_service_mesh(aks)
```

### 3. Data Lake Architecture
```python
# Data Lake Storage
datalake = infra.create_datalake_storage("datalakeprod")

# Databricks workspace
databricks = infra.create_databricks_workspace("dbw-analytics")

# Synapse workspace
synapse = infra.create_synapse_workspace("syn-analytics")
```

## Performance Optimization

### 1. Use Proximity Placement Groups
```python
ppg = infra.create_proximity_placement_group("ppg-app")
vm = infra.create_vm(proximity_placement_group=ppg)
```

### 2. Enable Accelerated Networking
```python
vm = infra.create_vm(
    name="vm-app",
    enable_accelerated_networking=True
)
```

### 3. Use Premium Storage
```python
storage = infra.create_storage_account(
    name="stpremium",
    sku="Premium_LRS"
)
```

## Security Considerations

1. **Least Privilege**: Grant minimum required permissions
2. **Private Endpoints**: Use for services when possible
3. **Network Isolation**: Implement network segmentation
4. **Encryption**: Enable at-rest and in-transit encryption
5. **Key Management**: Use Azure Key Vault

## Troubleshooting

**Issue**: Resource deployment fails
**Solution**: Check subscription quotas and permissions

**Issue**: Network connectivity issues
**Solution**: Verify NSG rules and routing

**Issue**: High costs
**Solution**: Review resource sizing and implement auto-shutdown

## Deployment

### Terraform Workflow

```bash
# Initialize Terraform
terraform init

# Plan changes
terraform plan -out=tfplan

# Apply changes
terraform apply tfplan

# Destroy resources
terraform destroy
```

### ARM Template Deployment

```bash
# Deploy ARM template
az deployment group create \
    --resource-group rg-demo \
    --template-file template.json \
    --parameters parameters.json
```

## Monitoring

### Key Metrics
- Resource health status
- Cost trends
- Compliance score
- Security posture
- Resource utilization

### Azure Monitor Integration

```python
# Enable monitoring
infra.enable_monitoring(
    log_analytics_workspace=workspace,
    diagnostic_settings={
        "logs": ["AuditLogs", "SignInLogs"],
        "metrics": ["AllMetrics"]
    }
)
```

## Dependencies

```
Python >= 3.8
azure-mgmt-resource >= 21.0.0
azure-mgmt-network >= 20.0.0
azure-mgmt-compute >= 28.0.0
azure-mgmt-storage >= 20.0.0
```

## Support

For questions or support:
- Email: clientbrill@gmail.com
- LinkedIn: [brillconsulting](https://www.linkedin.com/in/brillconsulting)

## Related Projects

- [Azure DevOps](../DevOps/)
- [Azure Monitor](../AzureMonitor/)
- [Container Apps](../ContainerApps/)

---

**Built with Azure Infrastructure as Code** | **Brill Consulting © 2024**
