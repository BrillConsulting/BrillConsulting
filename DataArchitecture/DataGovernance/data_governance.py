"""
Data Governance Framework
==========================

Data governance, metadata, and lineage tracking:
- Metadata management
- Data lineage tracking
- Access policies
- Data quality rules
- Compliance tracking

Author: Brill Consulting
"""

from datetime import datetime
from typing import Dict, List
import json


class DataGovernance:
    """Data governance system."""

    def __init__(self):
        """Initialize governance system."""
        self.metadata_registry = {}
        self.lineage_graph = {}
        self.policies = {}
        self.compliance_records = []

    def register_dataset(self, dataset_id: str, metadata: Dict) -> Dict:
        """Register dataset with metadata."""
        print(f"Registering dataset: {dataset_id}")

        registration = {
            "dataset_id": dataset_id,
            "metadata": metadata,
            "registered_at": datetime.now().isoformat(),
            "owner": metadata.get("owner", "unknown"),
            "classification": metadata.get("classification", "internal"),
            "tags": metadata.get("tags", [])
        }

        self.metadata_registry[dataset_id] = registration
        print(f"✓ Registered {dataset_id}")
        return registration

    def track_lineage(self, dataset_id: str, source_datasets: List[str],
                     transformation: str) -> Dict:
        """Track data lineage."""
        print(f"Tracking lineage for {dataset_id}")

        lineage = {
            "dataset_id": dataset_id,
            "sources": source_datasets,
            "transformation": transformation,
            "timestamp": datetime.now().isoformat()
        }

        if dataset_id not in self.lineage_graph:
            self.lineage_graph[dataset_id] = []

        self.lineage_graph[dataset_id].append(lineage)
        print(f"✓ Tracked lineage: {len(source_datasets)} sources")
        return lineage

    def get_lineage(self, dataset_id: str, depth: int = 1) -> Dict:
        """Get data lineage for dataset."""
        if dataset_id not in self.lineage_graph:
            return {}

        lineage = {
            "dataset_id": dataset_id,
            "lineage": self.lineage_graph[dataset_id]
        }

        # Get upstream lineage
        if depth > 0:
            for record in self.lineage_graph[dataset_id]:
                for source in record["sources"]:
                    upstream = self.get_lineage(source, depth - 1)
                    if upstream:
                        lineage[f"upstream_{source}"] = upstream

        return lineage

    def create_policy(self, policy_id: str, policy: Dict) -> Dict:
        """Create data policy."""
        print(f"Creating policy: {policy_id}")

        policy_record = {
            "policy_id": policy_id,
            "type": policy.get("type", "access"),
            "rules": policy.get("rules", []),
            "created_at": datetime.now().isoformat(),
            "status": "active"
        }

        self.policies[policy_id] = policy_record
        print(f"✓ Created policy {policy_id}")
        return policy_record

    def check_compliance(self, dataset_id: str, policy_id: str) -> Dict:
        """Check dataset compliance with policy."""
        print(f"Checking compliance: {dataset_id} against {policy_id}")

        if dataset_id not in self.metadata_registry:
            return {"compliant": False, "reason": "Dataset not registered"}

        if policy_id not in self.policies:
            return {"compliant": False, "reason": "Policy not found"}

        # Simulate compliance check
        dataset = self.metadata_registry[dataset_id]
        policy = self.policies[policy_id]

        compliance = {
            "dataset_id": dataset_id,
            "policy_id": policy_id,
            "compliant": True,
            "checks": [],
            "timestamp": datetime.now().isoformat()
        }

        # Check rules
        for rule in policy.get("rules", []):
            check = {
                "rule": rule,
                "passed": True  # Simulate check
            }
            compliance["checks"].append(check)

        self.compliance_records.append(compliance)
        print(f"✓ Compliance check: {'PASSED' if compliance['compliant'] else 'FAILED'}")
        return compliance

    def get_dataset_info(self, dataset_id: str) -> Dict:
        """Get comprehensive dataset information."""
        info = {}

        if dataset_id in self.metadata_registry:
            info["metadata"] = self.metadata_registry[dataset_id]

        if dataset_id in self.lineage_graph:
            info["lineage"] = self.lineage_graph[dataset_id]

        # Find applicable policies
        info["policies"] = []
        for policy_id, policy in self.policies.items():
            if policy["status"] == "active":
                info["policies"].append(policy_id)

        return info


def demo():
    """Demo data governance."""
    print("Data Governance Demo")
    print("="*50)

    gov = DataGovernance()

    # 1. Register datasets
    print("\n1. Registering Datasets")
    print("-"*50)

    gov.register_dataset("sales_raw", {
        "owner": "data_team",
        "classification": "confidential",
        "tags": ["sales", "raw"],
        "description": "Raw sales data"
    })

    gov.register_dataset("sales_processed", {
        "owner": "analytics_team",
        "classification": "internal",
        "tags": ["sales", "processed"],
        "description": "Processed sales data"
    })

    # 2. Track lineage
    print("\n2. Tracking Data Lineage")
    print("-"*50)

    gov.track_lineage("sales_processed",
                     source_datasets=["sales_raw"],
                     transformation="clean_and_aggregate")

    gov.track_lineage("sales_report",
                     source_datasets=["sales_processed"],
                     transformation="create_report")

    # Get lineage
    lineage = gov.get_lineage("sales_report", depth=2)
    print(f"Lineage for sales_report: {len(lineage.get('lineage', []))} transformations")

    # 3. Create policies
    print("\n3. Creating Data Policies")
    print("-"*50)

    gov.create_policy("pii_policy", {
        "type": "privacy",
        "rules": [
            "must_encrypt_at_rest",
            "must_mask_in_dev",
            "require_audit_log"
        ]
    })

    gov.create_policy("retention_policy", {
        "type": "retention",
        "rules": [
            "retain_for_7_years",
            "archive_after_1_year"
        ]
    })

    # 4. Check compliance
    print("\n4. Checking Compliance")
    print("-"*50)

    compliance = gov.check_compliance("sales_processed", "pii_policy")
    print(f"Compliance status: {'COMPLIANT' if compliance['compliant'] else 'NON-COMPLIANT'}")
    print(f"Checks performed: {len(compliance['checks'])}")

    # 5. Get dataset info
    print("\n5. Dataset Information")
    print("-"*50)

    info = gov.get_dataset_info("sales_processed")
    print(f"Owner: {info['metadata']['owner']}")
    print(f"Classification: {info['metadata']['classification']}")
    print(f"Applicable policies: {len(info['policies'])}")

    print("\n✓ Data Governance Demo Complete!")


if __name__ == '__main__':
    demo()
