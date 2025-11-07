"""
Data Lineage & Access Control
==============================

Track data flow and manage access in ML pipelines

Author: Brill Consulting
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json


class ResourceType(Enum):
    """Types of ML resources."""
    DATASET = "dataset"
    MODEL = "model"
    FEATURE = "feature"
    PIPELINE = "pipeline"


class Permission(Enum):
    """Access permissions."""
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    DELETE = "delete"


@dataclass
class DataAsset:
    """ML data asset."""
    id: str
    name: str
    type: ResourceType
    source: str
    schema: Dict[str, str]
    contains_pii: bool
    created_at: str
    created_by: str
    parents: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AccessRecord:
    """Access log record."""
    user: str
    resource_id: str
    action: str
    timestamp: str
    approved: bool
    reason: Optional[str] = None


class LineageTracker:
    """Track data lineage in ML pipelines."""

    def __init__(self):
        """Initialize lineage tracker."""
        self.assets: Dict[str, DataAsset] = {}
        self.transformations: List[Dict] = []

        print(f"📊 Data Lineage Tracker initialized")

    def register_dataset(
        self,
        name: str,
        source: str,
        schema: Dict[str, str],
        contains_pii: bool = False,
        created_by: str = "system"
    ) -> str:
        """Register new dataset."""
        asset_id = f"dataset_{len(self.assets) + 1}"

        asset = DataAsset(
            id=asset_id,
            name=name,
            type=ResourceType.DATASET,
            source=source,
            schema=schema,
            contains_pii=contains_pii,
            created_at=datetime.now().isoformat(),
            created_by=created_by
        )

        self.assets[asset_id] = asset

        print(f"   ✓ Registered dataset: {name}")
        if contains_pii:
            print(f"   ⚠️  Contains PII - access restricted")

        return asset_id

    def track_transformation(
        self,
        input_id: str,
        output_name: str,
        transformation: str,
        metadata: Optional[Dict] = None
    ) -> str:
        """Track data transformation."""
        if input_id not in self.assets:
            raise ValueError(f"Input asset {input_id} not found")

        # Create output asset
        output_id = f"transformed_{len(self.assets) + 1}"

        input_asset = self.assets[input_id]

        output_asset = DataAsset(
            id=output_id,
            name=output_name,
            type=ResourceType.DATASET,
            source=f"transformed_from_{input_id}",
            schema=input_asset.schema.copy(),
            contains_pii=input_asset.contains_pii,
            created_at=datetime.now().isoformat(),
            created_by="pipeline",
            parents=[input_id],
            metadata=metadata or {}
        )

        self.assets[output_id] = output_asset

        # Record transformation
        self.transformations.append({
            "input": input_id,
            "output": output_id,
            "transformation": transformation,
            "timestamp": datetime.now().isoformat()
        })

        print(f"   ✓ Tracked transformation: {transformation}")
        print(f"     {input_id} → {output_id}")

        return output_id

    def get_lineage(self, asset_id: str) -> Dict[str, Any]:
        """Get complete lineage for asset."""
        if asset_id not in self.assets:
            return {}

        asset = self.assets[asset_id]

        lineage = {
            "asset": asset,
            "parents": [],
            "children": []
        }

        # Find parents
        for parent_id in asset.parents:
            if parent_id in self.assets:
                lineage["parents"].append(self.assets[parent_id])

        # Find children
        for other_asset in self.assets.values():
            if asset_id in other_asset.parents:
                lineage["children"].append(other_asset)

        return lineage

    def visualize_lineage(self, asset_id: str) -> None:
        """Visualize data lineage."""
        print(f"\n📈 Lineage for: {asset_id}")

        lineage = self.get_lineage(asset_id)

        if lineage["parents"]:
            print(f"\n   Parents:")
            for parent in lineage["parents"]:
                print(f"      ← {parent.id} ({parent.name})")

        print(f"\n   Current: {asset_id}")

        if lineage["children"]:
            print(f"\n   Children:")
            for child in lineage["children"]:
                print(f"      → {child.id} ({child.name})")


class AccessControl:
    """Manage access control for ML assets."""

    def __init__(self):
        """Initialize access control."""
        self.permissions: Dict[str, Dict[str, List[Permission]]] = {}
        self.access_logs: List[AccessRecord] = []

        print(f"🔐 Access Control initialized")

    def grant_access(
        self,
        user: str,
        resource: str,
        permissions: List[str]
    ) -> None:
        """Grant user access to resource."""
        if user not in self.permissions:
            self.permissions[user] = {}

        self.permissions[user][resource] = [
            Permission(p) for p in permissions
        ]

        print(f"   ✓ Granted {user} access to {resource}")
        print(f"     Permissions: {', '.join(permissions)}")

    def check_access(
        self,
        user: str,
        resource: str,
        action: str
    ) -> bool:
        """Check if user has access."""
        if user not in self.permissions:
            return False

        if resource not in self.permissions[user]:
            return False

        required_perm = Permission(action)
        has_access = required_perm in self.permissions[user][resource]

        # Log access attempt
        self.access_logs.append(AccessRecord(
            user=user,
            resource_id=resource,
            action=action,
            timestamp=datetime.now().isoformat(),
            approved=has_access
        ))

        return has_access

    def get_audit_log(self, user: Optional[str] = None) -> List[AccessRecord]:
        """Get audit log."""
        if user:
            return [log for log in self.access_logs if log.user == user]
        return self.access_logs


def demo():
    """Demonstrate data lineage and access control."""
    print("=" * 60)
    print("Data Lineage & Access Control Demo")
    print("=" * 60)

    # Lineage tracking
    tracker = LineageTracker()

    # Register dataset
    dataset_id = tracker.register_dataset(
        name="customer_data",
        source="s3://data/customers.csv",
        schema={"name": "str", "email": "str", "age": "int"},
        contains_pii=True,
        created_by="data_engineer"
    )

    # Track transformations
    cleaned_id = tracker.track_transformation(
        input_id=dataset_id,
        output_name="cleaned_customers",
        transformation="remove_duplicates"
    )

    anonymized_id = tracker.track_transformation(
        input_id=cleaned_id,
        output_name="anonymized_customers",
        transformation="remove_pii",
        metadata={"columns_dropped": ["name", "email"]}
    )

    # Visualize
    tracker.visualize_lineage(anonymized_id)

    # Access control
    print(f"\n{'='*60}")
    print("Access Control")
    print(f"{'='*60}")

    acl = AccessControl()

    # Grant permissions
    acl.grant_access(
        user="data_scientist_1",
        resource=anonymized_id,
        permissions=["read"]
    )

    acl.grant_access(
        user="admin",
        resource=dataset_id,
        permissions=["read", "write", "delete"]
    )

    # Check access
    print(f"\n   Access checks:")
    print(f"   data_scientist_1 → {anonymized_id} (read): {acl.check_access('data_scientist_1', anonymized_id, 'read')}")
    print(f"   data_scientist_1 → {dataset_id} (read): {acl.check_access('data_scientist_1', dataset_id, 'read')}")

    # Audit log
    print(f"\n   Audit log: {len(acl.access_logs)} entries")


if __name__ == "__main__":
    demo()
