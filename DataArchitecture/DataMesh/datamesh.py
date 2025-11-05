"""
Data Mesh Framework
===================

Domain-driven data architecture with distributed data ownership:
- Data products with ownership and SLAs
- Self-serve data infrastructure
- Federated computational governance
- Domain-oriented decentralization
- Data product discovery and catalog

Author: Brill Consulting
"""

from datetime import datetime
from typing import Dict, List, Optional
import json


class DataProduct:
    """Represents a data product in the mesh."""

    def __init__(self, product_id: str, domain: str, owner: str,
                 description: str, sla: Dict):
        """Initialize data product."""
        self.product_id = product_id
        self.domain = domain
        self.owner = owner
        self.description = description
        self.sla = sla
        self.created_at = datetime.now().isoformat()
        self.consumers = []
        self.metadata = {}
        self.quality_metrics = {}

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "product_id": self.product_id,
            "domain": self.domain,
            "owner": self.owner,
            "description": self.description,
            "sla": self.sla,
            "created_at": self.created_at,
            "consumers": self.consumers,
            "metadata": self.metadata,
            "quality_metrics": self.quality_metrics
        }


class DataMesh:
    """Data Mesh implementation with domain-driven architecture."""

    def __init__(self):
        """Initialize Data Mesh."""
        self.data_products = {}
        self.domains = {}
        self.governance_policies = {}
        self.infrastructure_templates = {}
        self.access_logs = []

    def register_domain(self, domain_id: str, domain_info: Dict) -> Dict:
        """Register a domain in the mesh."""
        print(f"Registering domain: {domain_id}")

        domain = {
            "domain_id": domain_id,
            "name": domain_info.get("name", domain_id),
            "owner": domain_info.get("owner"),
            "team": domain_info.get("team", []),
            "description": domain_info.get("description"),
            "data_products": [],
            "registered_at": datetime.now().isoformat()
        }

        self.domains[domain_id] = domain
        print(f"✓ Registered domain: {domain['name']}")
        return domain

    def create_data_product(self, product_id: str, domain: str,
                           owner: str, description: str, sla: Dict,
                           metadata: Optional[Dict] = None) -> DataProduct:
        """Create a data product."""
        print(f"Creating data product: {product_id}")

        if domain not in self.domains:
            raise ValueError(f"Domain {domain} not registered")

        product = DataProduct(product_id, domain, owner, description, sla)

        if metadata:
            product.metadata = metadata

        self.data_products[product_id] = product
        self.domains[domain]["data_products"].append(product_id)

        print(f"✓ Created data product: {product_id}")
        print(f"  Domain: {domain}")
        print(f"  Owner: {owner}")
        print(f"  SLA: {sla.get('availability', 'N/A')}% availability")

        return product

    def publish_data_product(self, product_id: str, version: str,
                            schema: Dict, access_policy: Dict) -> Dict:
        """Publish a data product version."""
        print(f"Publishing data product: {product_id} v{version}")

        if product_id not in self.data_products:
            raise ValueError(f"Data product {product_id} not found")

        product = self.data_products[product_id]

        publication = {
            "product_id": product_id,
            "version": version,
            "schema": schema,
            "access_policy": access_policy,
            "published_at": datetime.now().isoformat(),
            "status": "active"
        }

        if "versions" not in product.metadata:
            product.metadata["versions"] = []

        product.metadata["versions"].append(publication)

        print(f"✓ Published {product_id} v{version}")
        return publication

    def subscribe_to_product(self, product_id: str, consumer_id: str,
                            consumer_domain: str) -> Dict:
        """Subscribe a consumer to a data product."""
        print(f"Subscribing {consumer_id} to {product_id}")

        if product_id not in self.data_products:
            raise ValueError(f"Data product {product_id} not found")

        product = self.data_products[product_id]

        subscription = {
            "consumer_id": consumer_id,
            "consumer_domain": consumer_domain,
            "product_id": product_id,
            "subscribed_at": datetime.now().isoformat(),
            "status": "active"
        }

        product.consumers.append(subscription)

        print(f"✓ Subscription created")
        print(f"  Consumer: {consumer_id} ({consumer_domain})")
        print(f"  Product: {product_id}")

        return subscription

    def create_governance_policy(self, policy_id: str, policy: Dict) -> Dict:
        """Create a federated governance policy."""
        print(f"Creating governance policy: {policy_id}")

        policy_record = {
            "policy_id": policy_id,
            "type": policy.get("type"),
            "scope": policy.get("scope", "global"),
            "rules": policy.get("rules", []),
            "enforcement": policy.get("enforcement", "advisory"),
            "created_at": datetime.now().isoformat(),
            "status": "active"
        }

        self.governance_policies[policy_id] = policy_record

        print(f"✓ Created policy: {policy_id}")
        print(f"  Type: {policy_record['type']}")
        print(f"  Scope: {policy_record['scope']}")
        print(f"  Enforcement: {policy_record['enforcement']}")

        return policy_record

    def check_policy_compliance(self, product_id: str, policy_id: str) -> Dict:
        """Check data product compliance with policy."""
        print(f"Checking compliance: {product_id} against {policy_id}")

        if product_id not in self.data_products:
            return {"compliant": False, "reason": "Product not found"}

        if policy_id not in self.governance_policies:
            return {"compliant": False, "reason": "Policy not found"}

        product = self.data_products[product_id]
        policy = self.governance_policies[policy_id]

        # Simulate compliance checks
        compliance_checks = []
        for rule in policy["rules"]:
            check = {
                "rule": rule,
                "passed": True,  # Simplified check
                "details": f"Checked {rule}"
            }
            compliance_checks.append(check)

        result = {
            "product_id": product_id,
            "policy_id": policy_id,
            "compliant": all(c["passed"] for c in compliance_checks),
            "checks": compliance_checks,
            "checked_at": datetime.now().isoformat()
        }

        print(f"✓ Compliance: {'PASS' if result['compliant'] else 'FAIL'}")
        return result

    def create_infrastructure_template(self, template_id: str,
                                      template: Dict) -> Dict:
        """Create self-serve infrastructure template."""
        print(f"Creating infrastructure template: {template_id}")

        template_record = {
            "template_id": template_id,
            "name": template.get("name"),
            "type": template.get("type"),
            "components": template.get("components", []),
            "configuration": template.get("configuration", {}),
            "created_at": datetime.now().isoformat()
        }

        self.infrastructure_templates[template_id] = template_record

        print(f"✓ Created template: {template_id}")
        print(f"  Type: {template_record['type']}")
        print(f"  Components: {len(template_record['components'])}")

        return template_record

    def provision_infrastructure(self, product_id: str, template_id: str,
                                config: Optional[Dict] = None) -> Dict:
        """Provision infrastructure for data product."""
        print(f"Provisioning infrastructure for {product_id}")

        if product_id not in self.data_products:
            raise ValueError(f"Data product {product_id} not found")

        if template_id not in self.infrastructure_templates:
            raise ValueError(f"Template {template_id} not found")

        template = self.infrastructure_templates[template_id]
        product = self.data_products[product_id]

        infrastructure = {
            "product_id": product_id,
            "template_id": template_id,
            "configuration": config or template["configuration"],
            "components": template["components"],
            "provisioned_at": datetime.now().isoformat(),
            "status": "active"
        }

        if "infrastructure" not in product.metadata:
            product.metadata["infrastructure"] = []

        product.metadata["infrastructure"].append(infrastructure)

        print(f"✓ Infrastructure provisioned")
        print(f"  Template: {template_id}")
        print(f"  Components: {len(infrastructure['components'])}")

        return infrastructure

    def log_access(self, product_id: str, consumer_id: str,
                   operation: str) -> Dict:
        """Log data product access."""
        access_log = {
            "product_id": product_id,
            "consumer_id": consumer_id,
            "operation": operation,
            "timestamp": datetime.now().isoformat()
        }

        self.access_logs.append(access_log)
        return access_log

    def update_quality_metrics(self, product_id: str, metrics: Dict) -> Dict:
        """Update data product quality metrics."""
        print(f"Updating quality metrics for {product_id}")

        if product_id not in self.data_products:
            raise ValueError(f"Data product {product_id} not found")

        product = self.data_products[product_id]
        product.quality_metrics = {
            **metrics,
            "updated_at": datetime.now().isoformat()
        }

        print(f"✓ Quality metrics updated")
        for metric, value in metrics.items():
            print(f"  {metric}: {value}")

        return product.quality_metrics

    def discover_products(self, filters: Optional[Dict] = None) -> List[Dict]:
        """Discover data products based on filters."""
        print("Discovering data products...")

        products = []
        for product_id, product in self.data_products.items():
            product_dict = product.to_dict()

            # Apply filters
            if filters:
                if "domain" in filters and product.domain != filters["domain"]:
                    continue
                if "owner" in filters and product.owner != filters["owner"]:
                    continue

            products.append(product_dict)

        print(f"✓ Found {len(products)} data products")
        return products

    def get_product_lineage(self, product_id: str) -> Dict:
        """Get lineage for a data product."""
        if product_id not in self.data_products:
            return {}

        product = self.data_products[product_id]

        lineage = {
            "product_id": product_id,
            "domain": product.domain,
            "consumers": product.consumers,
            "access_count": len([log for log in self.access_logs
                               if log["product_id"] == product_id])
        }

        return lineage

    def generate_mesh_report(self) -> Dict:
        """Generate comprehensive mesh report."""
        print("\nGenerating Data Mesh Report...")
        print("="*50)

        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_domains": len(self.domains),
                "total_products": len(self.data_products),
                "total_policies": len(self.governance_policies),
                "total_templates": len(self.infrastructure_templates)
            },
            "domains": {},
            "top_products": []
        }

        # Domain statistics
        for domain_id, domain in self.domains.items():
            report["domains"][domain_id] = {
                "name": domain["name"],
                "products_count": len(domain["data_products"]),
                "owner": domain["owner"]
            }

        # Top products by consumer count
        product_consumers = [(pid, len(p.consumers))
                           for pid, p in self.data_products.items()]
        product_consumers.sort(key=lambda x: x[1], reverse=True)
        report["top_products"] = product_consumers[:5]

        print(f"Total Domains: {report['summary']['total_domains']}")
        print(f"Total Products: {report['summary']['total_products']}")
        print(f"Total Policies: {report['summary']['total_policies']}")

        return report


def demo():
    """Demo Data Mesh."""
    print("Data Mesh Demo")
    print("="*50)

    mesh = DataMesh()

    # 1. Register domains
    print("\n1. Registering Domains")
    print("-"*50)

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

    # 2. Create data products
    print("\n2. Creating Data Products")
    print("-"*50)

    sales_product = mesh.create_data_product(
        "sales_transactions",
        domain="sales",
        owner="alice",
        description="Daily sales transactions",
        sla={"availability": 99.9, "freshness": "24h", "quality": 95},
        metadata={"format": "parquet", "partition": "date"}
    )

    marketing_product = mesh.create_data_product(
        "campaign_metrics",
        domain="marketing",
        owner="carol",
        description="Marketing campaign performance",
        sla={"availability": 99.5, "freshness": "1h", "quality": 90},
        metadata={"format": "json", "update_frequency": "hourly"}
    )

    # 3. Publish data products
    print("\n3. Publishing Data Products")
    print("-"*50)

    mesh.publish_data_product(
        "sales_transactions",
        version="1.0",
        schema={
            "fields": [
                {"name": "transaction_id", "type": "string"},
                {"name": "amount", "type": "decimal"},
                {"name": "date", "type": "date"}
            ]
        },
        access_policy={"type": "role-based", "roles": ["analyst", "manager"]}
    )

    # 4. Subscribe to products
    print("\n4. Subscribing to Data Products")
    print("-"*50)

    mesh.subscribe_to_product(
        "sales_transactions",
        consumer_id="analytics_pipeline",
        consumer_domain="analytics"
    )

    mesh.subscribe_to_product(
        "sales_transactions",
        consumer_id="ml_model",
        consumer_domain="data_science"
    )

    # 5. Create governance policies
    print("\n5. Creating Governance Policies")
    print("-"*50)

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

    mesh.create_governance_policy("privacy", {
        "type": "privacy",
        "scope": "pii_data",
        "rules": [
            "encryption_at_rest",
            "encryption_in_transit",
            "access_logging"
        ],
        "enforcement": "mandatory"
    })

    # 6. Check compliance
    print("\n6. Checking Policy Compliance")
    print("-"*50)

    compliance = mesh.check_policy_compliance("sales_transactions", "data_quality")
    print(f"Checks passed: {sum(1 for c in compliance['checks'] if c['passed'])}/{len(compliance['checks'])}")

    # 7. Create infrastructure templates
    print("\n7. Creating Infrastructure Templates")
    print("-"*50)

    mesh.create_infrastructure_template("standard_pipeline", {
        "name": "Standard Data Pipeline",
        "type": "batch",
        "components": ["ingestion", "processing", "storage", "serving"],
        "configuration": {
            "storage": "s3",
            "compute": "spark",
            "orchestration": "airflow"
        }
    })

    # 8. Provision infrastructure
    print("\n8. Provisioning Infrastructure")
    print("-"*50)

    mesh.provision_infrastructure(
        "sales_transactions",
        "standard_pipeline"
    )

    # 9. Update quality metrics
    print("\n9. Updating Quality Metrics")
    print("-"*50)

    mesh.update_quality_metrics("sales_transactions", {
        "completeness": 98.5,
        "accuracy": 95.2,
        "timeliness_hours": 2.5,
        "row_count": 1500000
    })

    # 10. Discover products
    print("\n10. Discovering Data Products")
    print("-"*50)

    products = mesh.discover_products({"domain": "sales"})
    print(f"Found {len(products)} products in sales domain")

    # 11. Generate report
    print("\n11. Mesh Report")
    print("-"*50)

    report = mesh.generate_mesh_report()
    print(f"\nDomains:")
    for domain_id, info in report["domains"].items():
        print(f"  {info['name']}: {info['products_count']} products")

    print("\n✓ Data Mesh Demo Complete!")


if __name__ == '__main__':
    demo()
