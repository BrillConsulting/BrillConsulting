"""
Data Versioning Framework
=========================

Comprehensive dataset and schema versioning with time travel:
- Dataset versioning and snapshots
- Schema evolution and versioning
- Time travel queries
- Rollback capabilities
- Version comparison and diffs
- Branch and merge support

Author: Brill Consulting
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
import json
import hashlib


class DataVersion:
    """Represents a version of a dataset."""

    def __init__(self, version_id: str, dataset_id: str, version_number: str,
                 schema: Dict, metadata: Dict):
        """Initialize data version."""
        self.version_id = version_id
        self.dataset_id = dataset_id
        self.version_number = version_number
        self.schema = schema
        self.metadata = metadata
        self.created_at = datetime.now().isoformat()
        self.parent_version = None
        self.tags = []

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "version_id": self.version_id,
            "dataset_id": self.dataset_id,
            "version_number": self.version_number,
            "schema": self.schema,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "parent_version": self.parent_version,
            "tags": self.tags
        }


class DataVersioning:
    """Data versioning system with time travel and rollback."""

    def __init__(self):
        """Initialize versioning system."""
        self.datasets = {}
        self.versions = {}
        self.branches = {}
        self.snapshots = {}
        self.schema_history = {}

    def register_dataset(self, dataset_id: str, name: str, owner: str,
                        initial_schema: Dict) -> Dict:
        """Register a dataset for versioning."""
        print(f"Registering dataset: {name}")

        dataset = {
            "dataset_id": dataset_id,
            "name": name,
            "owner": owner,
            "current_version": None,
            "default_branch": "main",
            "registered_at": datetime.now().isoformat()
        }

        self.datasets[dataset_id] = dataset
        self.branches[dataset_id] = {"main": []}
        self.schema_history[dataset_id] = []

        # Create initial version
        version = self.create_version(
            dataset_id,
            version_number="1.0.0",
            schema=initial_schema,
            metadata={"message": "Initial version"},
            branch="main"
        )

        dataset["current_version"] = version.version_id

        print(f"✓ Registered dataset: {dataset_id}")
        print(f"  Initial version: {version.version_number}")

        return dataset

    def create_version(self, dataset_id: str, version_number: str,
                      schema: Dict, metadata: Dict,
                      branch: str = "main",
                      parent_version: Optional[str] = None) -> DataVersion:
        """Create a new version of a dataset."""
        print(f"Creating version {version_number} for {dataset_id}")

        if dataset_id not in self.datasets:
            raise ValueError(f"Dataset {dataset_id} not registered")

        # Generate version ID
        version_id = self._generate_version_id(dataset_id, version_number)

        version = DataVersion(
            version_id=version_id,
            dataset_id=dataset_id,
            version_number=version_number,
            schema=schema,
            metadata=metadata
        )

        if parent_version:
            version.parent_version = parent_version

        self.versions[version_id] = version

        # Add to branch
        if dataset_id not in self.branches:
            self.branches[dataset_id] = {}
        if branch not in self.branches[dataset_id]:
            self.branches[dataset_id][branch] = []

        self.branches[dataset_id][branch].append(version_id)

        # Track schema changes
        self.schema_history[dataset_id].append({
            "version_id": version_id,
            "version_number": version_number,
            "schema": schema,
            "timestamp": version.created_at
        })

        print(f"✓ Created version: {version_number}")
        print(f"  Branch: {branch}")
        print(f"  Schema fields: {len(schema.get('fields', []))}")

        return version

    def _generate_version_id(self, dataset_id: str, version_number: str) -> str:
        """Generate unique version ID."""
        content = f"{dataset_id}:{version_number}:{datetime.now().isoformat()}"
        return hashlib.md5(content.encode()).hexdigest()[:16]

    def create_snapshot(self, dataset_id: str, snapshot_name: str,
                       version_id: Optional[str] = None,
                       description: Optional[str] = None) -> Dict:
        """Create a snapshot of a dataset version."""
        print(f"Creating snapshot: {snapshot_name}")

        if dataset_id not in self.datasets:
            raise ValueError(f"Dataset {dataset_id} not registered")

        # Use current version if not specified
        if not version_id:
            version_id = self.datasets[dataset_id]["current_version"]

        if version_id not in self.versions:
            raise ValueError(f"Version {version_id} not found")

        snapshot_id = f"snap_{len(self.snapshots) + 1:06d}"

        snapshot = {
            "snapshot_id": snapshot_id,
            "name": snapshot_name,
            "dataset_id": dataset_id,
            "version_id": version_id,
            "description": description,
            "created_at": datetime.now().isoformat()
        }

        self.snapshots[snapshot_id] = snapshot

        print(f"✓ Snapshot created: {snapshot_id}")
        print(f"  Version: {self.versions[version_id].version_number}")

        return snapshot

    def rollback(self, dataset_id: str, target_version: str) -> Dict:
        """Rollback dataset to a specific version."""
        print(f"Rolling back {dataset_id} to {target_version}")

        if dataset_id not in self.datasets:
            raise ValueError(f"Dataset {dataset_id} not registered")

        # Find target version
        target_version_id = None
        for vid, version in self.versions.items():
            if (version.dataset_id == dataset_id and
                version.version_number == target_version):
                target_version_id = vid
                break

        if not target_version_id:
            raise ValueError(f"Version {target_version} not found")

        dataset = self.datasets[dataset_id]
        current_version = dataset["current_version"]

        # Update current version
        dataset["current_version"] = target_version_id

        rollback_record = {
            "dataset_id": dataset_id,
            "from_version": self.versions[current_version].version_number,
            "to_version": target_version,
            "timestamp": datetime.now().isoformat()
        }

        print(f"✓ Rolled back to version {target_version}")

        return rollback_record

    def time_travel_query(self, dataset_id: str, timestamp: str) -> Optional[Dict]:
        """Query dataset state at a specific point in time."""
        print(f"Time travel query for {dataset_id} at {timestamp}")

        if dataset_id not in self.datasets:
            return None

        query_time = datetime.fromisoformat(timestamp)

        # Find the version that was active at the given time
        active_version = None
        for version_id in self.branches[dataset_id].get("main", []):
            version = self.versions[version_id]
            version_time = datetime.fromisoformat(version.created_at)

            if version_time <= query_time:
                active_version = version
            else:
                break

        if not active_version:
            print("✗ No version found for the specified time")
            return None

        result = {
            "dataset_id": dataset_id,
            "query_timestamp": timestamp,
            "version_number": active_version.version_number,
            "version_id": active_version.version_id,
            "schema": active_version.schema,
            "metadata": active_version.metadata
        }

        print(f"✓ Found version: {active_version.version_number}")

        return result

    def compare_versions(self, dataset_id: str, version1: str,
                        version2: str) -> Dict:
        """Compare two versions of a dataset."""
        print(f"Comparing versions: {version1} vs {version2}")

        if dataset_id not in self.datasets:
            raise ValueError(f"Dataset {dataset_id} not registered")

        # Find versions
        v1 = None
        v2 = None
        for vid, version in self.versions.items():
            if version.dataset_id == dataset_id:
                if version.version_number == version1:
                    v1 = version
                if version.version_number == version2:
                    v2 = version

        if not v1 or not v2:
            raise ValueError("One or both versions not found")

        # Compare schemas
        schema_diff = self._compare_schemas(v1.schema, v2.schema)

        # Compare metadata
        metadata_diff = {
            "version1": v1.metadata,
            "version2": v2.metadata,
            "changed_keys": []
        }

        for key in set(list(v1.metadata.keys()) + list(v2.metadata.keys())):
            if v1.metadata.get(key) != v2.metadata.get(key):
                metadata_diff["changed_keys"].append(key)

        result = {
            "dataset_id": dataset_id,
            "version1": version1,
            "version2": version2,
            "schema_diff": schema_diff,
            "metadata_diff": metadata_diff
        }

        print(f"✓ Comparison complete")
        print(f"  Schema changes: {len(schema_diff['added_fields']) + len(schema_diff['removed_fields'])} fields")
        print(f"  Metadata changes: {len(metadata_diff['changed_keys'])} keys")

        return result

    def _compare_schemas(self, schema1: Dict, schema2: Dict) -> Dict:
        """Compare two schemas."""
        fields1 = {f["name"]: f for f in schema1.get("fields", [])}
        fields2 = {f["name"]: f for f in schema2.get("fields", [])}

        added_fields = [name for name in fields2 if name not in fields1]
        removed_fields = [name for name in fields1 if name not in fields2]
        modified_fields = []

        for name in fields1:
            if name in fields2 and fields1[name] != fields2[name]:
                modified_fields.append({
                    "field": name,
                    "old": fields1[name],
                    "new": fields2[name]
                })

        return {
            "added_fields": added_fields,
            "removed_fields": removed_fields,
            "modified_fields": modified_fields
        }

    def create_branch(self, dataset_id: str, branch_name: str,
                     source_branch: str = "main") -> Dict:
        """Create a new branch from an existing branch."""
        print(f"Creating branch: {branch_name}")

        if dataset_id not in self.datasets:
            raise ValueError(f"Dataset {dataset_id} not registered")

        if branch_name in self.branches[dataset_id]:
            raise ValueError(f"Branch {branch_name} already exists")

        # Copy versions from source branch
        source_versions = self.branches[dataset_id].get(source_branch, [])
        self.branches[dataset_id][branch_name] = source_versions.copy()

        branch = {
            "dataset_id": dataset_id,
            "branch_name": branch_name,
            "source_branch": source_branch,
            "version_count": len(source_versions),
            "created_at": datetime.now().isoformat()
        }

        print(f"✓ Branch created: {branch_name}")
        print(f"  Source: {source_branch}")
        print(f"  Versions: {len(source_versions)}")

        return branch

    def merge_branch(self, dataset_id: str, source_branch: str,
                    target_branch: str) -> Dict:
        """Merge one branch into another."""
        print(f"Merging {source_branch} into {target_branch}")

        if dataset_id not in self.datasets:
            raise ValueError(f"Dataset {dataset_id} not registered")

        if source_branch not in self.branches[dataset_id]:
            raise ValueError(f"Source branch {source_branch} not found")

        if target_branch not in self.branches[dataset_id]:
            raise ValueError(f"Target branch {target_branch} not found")

        source_versions = self.branches[dataset_id][source_branch]
        target_versions = self.branches[dataset_id][target_branch]

        # Find versions to merge (in source but not in target)
        new_versions = [v for v in source_versions if v not in target_versions]

        # Add new versions to target
        self.branches[dataset_id][target_branch].extend(new_versions)

        merge_record = {
            "dataset_id": dataset_id,
            "source_branch": source_branch,
            "target_branch": target_branch,
            "merged_versions": len(new_versions),
            "timestamp": datetime.now().isoformat()
        }

        print(f"✓ Merge complete")
        print(f"  Merged versions: {len(new_versions)}")

        return merge_record

    def tag_version(self, dataset_id: str, version_number: str,
                   tag: str) -> Dict:
        """Tag a version for easy reference."""
        print(f"Tagging version {version_number} as '{tag}'")

        # Find version
        for vid, version in self.versions.items():
            if (version.dataset_id == dataset_id and
                version.version_number == version_number):
                version.tags.append({
                    "tag": tag,
                    "created_at": datetime.now().isoformat()
                })

                print(f"✓ Version tagged: {tag}")

                return {
                    "dataset_id": dataset_id,
                    "version_number": version_number,
                    "tag": tag
                }

        raise ValueError(f"Version {version_number} not found")

    def get_version_history(self, dataset_id: str,
                           branch: str = "main") -> List[Dict]:
        """Get version history for a dataset."""
        if dataset_id not in self.datasets:
            return []

        version_ids = self.branches[dataset_id].get(branch, [])
        history = []

        for vid in version_ids:
            version = self.versions[vid]
            history.append({
                "version_number": version.version_number,
                "version_id": version.version_id,
                "created_at": version.created_at,
                "message": version.metadata.get("message", ""),
                "tags": version.tags
            })

        return history

    def get_schema_evolution(self, dataset_id: str) -> List[Dict]:
        """Get schema evolution history."""
        if dataset_id not in self.schema_history:
            return []

        return self.schema_history[dataset_id]

    def generate_version_report(self, dataset_id: str) -> Dict:
        """Generate comprehensive versioning report."""
        print(f"\nGenerating Version Report for {dataset_id}")
        print("="*50)

        if dataset_id not in self.datasets:
            return {"error": "Dataset not found"}

        dataset = self.datasets[dataset_id]

        # Count versions
        version_count = sum(
            1 for v in self.versions.values()
            if v.dataset_id == dataset_id
        )

        # Count snapshots
        snapshot_count = sum(
            1 for s in self.snapshots.values()
            if s["dataset_id"] == dataset_id
        )

        report = {
            "dataset_id": dataset_id,
            "dataset_name": dataset["name"],
            "owner": dataset["owner"],
            "current_version": dataset["current_version"],
            "total_versions": version_count,
            "total_snapshots": snapshot_count,
            "branches": list(self.branches[dataset_id].keys()),
            "schema_evolution_steps": len(self.schema_history[dataset_id]),
            "generated_at": datetime.now().isoformat()
        }

        print(f"Total Versions: {version_count}")
        print(f"Total Snapshots: {snapshot_count}")
        print(f"Branches: {len(report['branches'])}")

        return report


def demo():
    """Demo Data Versioning."""
    print("Data Versioning Demo")
    print("="*50)

    dv = DataVersioning()

    # 1. Register dataset
    print("\n1. Registering Dataset")
    print("-"*50)

    dv.register_dataset(
        "user_data",
        "User Dataset",
        "data_team",
        initial_schema={
            "fields": [
                {"name": "user_id", "type": "int"},
                {"name": "name", "type": "string"},
                {"name": "email", "type": "string"}
            ]
        }
    )

    # 2. Create new version with schema evolution
    print("\n2. Creating Version 2.0.0 with Schema Changes")
    print("-"*50)

    dv.create_version(
        "user_data",
        "2.0.0",
        schema={
            "fields": [
                {"name": "user_id", "type": "int"},
                {"name": "name", "type": "string"},
                {"name": "email", "type": "string"},
                {"name": "phone", "type": "string"},  # Added field
                {"name": "created_at", "type": "timestamp"}  # Added field
            ]
        },
        metadata={"message": "Added phone and created_at fields"},
        branch="main"
    )

    # 3. Create snapshot
    print("\n3. Creating Snapshot")
    print("-"*50)

    snapshot = dv.create_snapshot(
        "user_data",
        "before_major_update",
        description="Snapshot before major schema refactoring"
    )

    # 4. Create development branch
    print("\n4. Creating Development Branch")
    print("-"*50)

    dv.create_branch("user_data", "development", source_branch="main")

    # 5. Create version in development branch
    print("\n5. Creating Version in Development Branch")
    print("-"*50)

    dv.create_version(
        "user_data",
        "2.1.0-dev",
        schema={
            "fields": [
                {"name": "user_id", "type": "int"},
                {"name": "name", "type": "string"},
                {"name": "email", "type": "string"},
                {"name": "phone", "type": "string"},
                {"name": "created_at", "type": "timestamp"},
                {"name": "status", "type": "string"}  # Experimental field
            ]
        },
        metadata={"message": "Experimental status field"},
        branch="development"
    )

    # 6. Compare versions
    print("\n6. Comparing Versions")
    print("-"*50)

    comparison = dv.compare_versions("user_data", "1.0.0", "2.0.0")
    print(f"\nAdded fields: {comparison['schema_diff']['added_fields']}")
    print(f"Removed fields: {comparison['schema_diff']['removed_fields']}")

    # 7. Tag version
    print("\n7. Tagging Version")
    print("-"*50)

    dv.tag_version("user_data", "2.0.0", "production")

    # 8. Time travel query
    print("\n8. Time Travel Query")
    print("-"*50)

    # Query for a time between version 1 and 2
    query_time = datetime.now().isoformat()
    result = dv.time_travel_query("user_data", query_time)

    if result:
        print(f"Active version at {query_time[:19]}: {result['version_number']}")

    # 9. Get version history
    print("\n9. Version History")
    print("-"*50)

    history = dv.get_version_history("user_data", branch="main")
    print(f"Main branch has {len(history)} versions:")
    for h in history:
        tags_str = f" [{', '.join(t['tag'] for t in h['tags'])}]" if h['tags'] else ""
        print(f"  v{h['version_number']}{tags_str}: {h['message']}")

    # 10. Get schema evolution
    print("\n10. Schema Evolution")
    print("-"*50)

    evolution = dv.get_schema_evolution("user_data")
    print(f"Schema has evolved {len(evolution)} times:")
    for i, e in enumerate(evolution, 1):
        field_count = len(e['schema'].get('fields', []))
        print(f"  {i}. v{e['version_number']}: {field_count} fields")

    # 11. Merge branch
    print("\n11. Merging Development Branch")
    print("-"*50)

    merge = dv.merge_branch("user_data", "development", "main")

    # 12. Generate report
    print("\n12. Version Report")
    print("-"*50)

    report = dv.generate_version_report("user_data")

    print("\n✓ Data Versioning Demo Complete!")


if __name__ == '__main__':
    demo()
