# Data Mesh
Domain-driven data architecture with distributed ownership and governance

## Overview

A comprehensive Data Mesh implementation enabling domain-oriented decentralized data ownership with federated computational governance. Supports data products with clear ownership, self-serve data infrastructure, and federated governance across domains while maintaining discoverability and interoperability.

## Features

### Core Capabilities
- **Domain Management**: Register and manage domains with teams and ownership
- **Data Products**: Create discoverable data products with SLAs and metadata
- **Product Publishing**: Version and publish data products with schemas and access policies
- **Consumer Subscription**: Subscribe consumers to data products with tracking
- **Product Discovery**: Search and discover data products across domains
- **Quality Metrics**: Track and update data product quality metrics

### Advanced Features
- **Federated Governance**: Create and enforce governance policies across domains
- **Policy Compliance**: Check data products against governance policies
- **Self-Serve Infrastructure**: Provision infrastructure using reusable templates
- **Infrastructure Templates**: Define standard components for data pipelines
- **Access Logging**: Track data product access and usage patterns
- **Product Lineage**: Track consumers and usage of data products
- **Mesh Reporting**: Generate comprehensive mesh health reports

## Installation

```bash
# Clone the repository
git clone https://github.com/BrillConsulting/DataArchitecture.git
cd DataArchitecture/DataMesh

# Install dependencies
pip install pandas

# Run the implementation
python datamesh.py
```

## Usage Examples

### Register Domains

```python
from datamesh import DataMesh

# Initialize Data Mesh
mesh = DataMesh()

# Register domain with ownership
mesh.register_domain("sales", {
    "name": "Sales Domain",
    "owner": "sales_team",
    "team": ["alice", "bob"],
    "description": "Sales and revenue data"
})

mesh.register_domain("marketing", {
    "name": "Marketing Domain",
    "owner": "marketing_team",
    "team": ["carol", "dave"],
    "description": "Marketing campaigns and analytics"
})
```

### Create Data Products

```python
# Create data product with SLA
sales_product = mesh.create_data_product(
    product_id="sales_transactions",
    domain="sales",
    owner="alice",
    description="Daily sales transactions",
    sla={
        "availability": 99.9,
        "freshness": "24h",
        "quality": 95
    },
    metadata={
        "format": "parquet",
        "partition": "date"
    }
)
```

### Publish Data Products

```python
# Publish a version with schema and access policy
mesh.publish_data_product(
    product_id="sales_transactions",
    version="1.0",
    schema={
        "fields": [
            {"name": "transaction_id", "type": "string"},
            {"name": "amount", "type": "decimal"},
            {"name": "date", "type": "date"}
        ]
    },
    access_policy={
        "type": "role-based",
        "roles": ["analyst", "manager"]
    }
)
```

### Subscribe to Products

```python
# Subscribe consumer to data product
subscription = mesh.subscribe_to_product(
    product_id="sales_transactions",
    consumer_id="analytics_pipeline",
    consumer_domain="analytics"
)

print(f"Subscription created for {subscription['consumer_id']}")
```

### Federated Governance

```python
# Create governance policy
mesh.create_governance_policy("data_quality", {
    "type": "quality",
    "scope": "global",
    "rules": [
        "completeness >= 95%",
        "accuracy >= 90%",
        "timeliness <= 24h"
    ],
    "enforcement": "mandatory"
})

# Check product compliance
compliance = mesh.check_policy_compliance(
    "sales_transactions",
    "data_quality"
)

print(f"Compliance: {'PASS' if compliance['compliant'] else 'FAIL'}")
```

### Self-Serve Infrastructure

```python
# Create infrastructure template
mesh.create_infrastructure_template("standard_pipeline", {
    "name": "Standard Data Pipeline",
    "type": "batch",
    "components": [
        "ingestion",
        "processing",
        "storage",
        "serving"
    ],
    "configuration": {
        "storage": "s3",
        "compute": "spark",
        "orchestration": "airflow"
    }
})

# Provision infrastructure for product
mesh.provision_infrastructure(
    product_id="sales_transactions",
    template_id="standard_pipeline"
)
```

### Update Quality Metrics

```python
# Update data product quality metrics
mesh.update_quality_metrics("sales_transactions", {
    "completeness": 98.5,
    "accuracy": 95.2,
    "timeliness_hours": 2.5,
    "row_count": 1500000
})
```

### Discover Products

```python
# Discover products by domain
products = mesh.discover_products({"domain": "sales"})

for product in products:
    print(f"{product['product_id']}: {product['description']}")
    print(f"  Owner: {product['owner']}")
    print(f"  Consumers: {len(product['consumers'])}")
```

## Demo Instructions

Run the included demonstration:

```bash
python datamesh.py
```

The demo showcases:
1. Domain registration with ownership
2. Data product creation with SLAs
3. Product publishing with schemas
4. Consumer subscriptions
5. Governance policy creation
6. Policy compliance checking
7. Infrastructure template creation
8. Self-serve infrastructure provisioning
9. Quality metrics tracking
10. Product discovery
11. Comprehensive mesh reporting

## Key Concepts

### Data Products

The fundamental unit in Data Mesh:
- **Clear Ownership**: Each product has an owner and domain
- **SLA Guarantees**: Availability, freshness, quality commitments
- **Discoverability**: Searchable with rich metadata
- **Versioning**: Track product evolution over time
- **Access Control**: Define who can consume the product

### Domain-Oriented Decentralization

- **Domain Ownership**: Each domain owns its data products
- **Autonomous Teams**: Domains can independently develop and deploy
- **Clear Boundaries**: Well-defined domain responsibilities
- **Cross-Domain Collaboration**: Products can be shared across domains

### Federated Computational Governance

- **Global Policies**: Apply standards across all domains
- **Domain Flexibility**: Domains can implement policies their way
- **Automated Compliance**: Check adherence to policies
- **Policy Types**: Quality, security, privacy, metadata standards

### Self-Serve Data Infrastructure

- **Reusable Templates**: Standard infrastructure patterns
- **Quick Provisioning**: Spin up infrastructure rapidly
- **Best Practices**: Embedded in templates
- **Reduced Overhead**: Teams focus on data, not infrastructure

## Architecture

```
┌─────────────────────────────────────────────────┐
│              Data Mesh                          │
│                                                 │
│  ┌───────────────┐        ┌───────────────┐    │
│  │ Sales Domain  │        │Marketing Domain│   │
│  │               │        │                │   │
│  │ Data Products:│        │ Data Products: │   │
│  │ - Transactions│        │ - Campaigns    │   │
│  │ - Customers   │        │ - Metrics      │   │
│  └───────┬───────┘        └───────┬────────┘   │
│          │                        │            │
│          └────────┬───────────────┘            │
│                   │                            │
│         ┌─────────▼──────────┐                 │
│         │ Federated Governance│                │
│         │ - Quality Policies  │                │
│         │ - Security Policies │                │
│         │ - Metadata Standards│                │
│         └─────────────────────┘                │
│                                                 │
│         ┌──────────────────────┐               │
│         │Self-Serve Infrastructure│            │
│         │ - Templates          │               │
│         │ - Automation         │               │
│         └──────────────────────┘               │
└─────────────────────────────────────────────────┘
```

## Use Cases

- **Decentralized Data Ownership**: Distribute data responsibility to domain experts
- **Cross-Functional Analytics**: Enable teams to share data products
- **Data Democratization**: Make data accessible to all who need it
- **Scalable Data Architecture**: Support organizational growth
- **Regulatory Compliance**: Apply consistent governance policies
- **Reduced Bottlenecks**: Eliminate central data team dependencies
- **Innovation Enablement**: Allow domains to innovate independently

## Best Practices

- Define clear domain boundaries
- Establish SLA standards across the organization
- Create comprehensive infrastructure templates
- Implement automated policy compliance checking
- Document data products thoroughly
- Monitor product usage and quality metrics
- Foster cross-domain collaboration
- Maintain a product catalog for discovery

## Migration Strategy

1. **Identify Domains**: Map organizational data to business domains
2. **Designate Owners**: Assign product owners within each domain
3. **Define Products**: Convert datasets to data products with SLAs
4. **Implement Governance**: Establish federated governance framework
5. **Build Infrastructure**: Create self-serve infrastructure platform
6. **Migrate Gradually**: Move domains incrementally to mesh pattern

## Author

**Brill Consulting**
- Email: clientbrill@gmail.com
- LinkedIn: [linkedin.com/in/brillconsulting](https://linkedin.com/in/brillconsulting)
- Specialization: Data Architecture & Engineering Solutions
