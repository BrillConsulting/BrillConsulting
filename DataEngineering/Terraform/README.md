# 🏗️ Terraform Infrastructure as Code

**Multi-cloud infrastructure provisioning**

## Overview
Terraform implementation for automated infrastructure provisioning across AWS, Azure, and GCP.

## Key Features
- Multi-cloud support
- Resource provisioning
- State management
- Module creation
- Plan and apply workflows
- Variable management

## Quick Start
```python
from terraform_manager import TerraformManager

mgr = TerraformManager('production')
config = mgr.create_infrastructure({'provider': 'aws'})
mgr.plan()
mgr.apply()
```

## Technologies
- Terraform HCL
- AWS/Azure/GCP APIs
- Infrastructure as Code

**Author:** Brill Consulting | clientbrill@gmail.com
