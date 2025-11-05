"""
Change Data Capture (CDC)
==========================

Real-time change data capture for data synchronization:
- Log-based CDC
- Trigger-based CDC
- Timestamp-based CDC
- Change event streaming
- Schema evolution handling
- Multiple source support
- Incremental replication

Author: Brill Consulting
"""

import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum
import hashlib
import json


class CDCMethod(Enum):
    """CDC capture methods."""
    LOG_BASED = "log_based"
    TRIGGER = "trigger"
    TIMESTAMP = "timestamp"
    SNAPSHOT = "snapshot"


class ChangeType(Enum):
    """Types of changes."""
    INSERT = "INSERT"
    UPDATE = "UPDATE"
    DELETE = "DELETE"


class ChangeDataCapture:
    """CDC system for real-time data synchronization."""

    def __init__(self, method: CDCMethod = CDCMethod.TIMESTAMP):
        """Initialize CDC system."""
        self.method = method
        self.tracked_tables = {}
        self.change_log = []
        self.snapshots = {}
        self.checkpoints = {}

    def register_table(self, table_name: str, key_columns: List[str],
                      tracking_column: Optional[str] = None) -> Dict:
        """Register table for CDC tracking."""
        print(f"Registering table for CDC: {table_name}")

        table_config = {
            "table_name": table_name,
            "key_columns": key_columns,
            "tracking_column": tracking_column or "updated_at",
            "registered_at": datetime.now().isoformat(),
            "last_checkpoint": None,
            "total_changes_captured": 0
        }

        self.tracked_tables[table_name] = table_config
        print(f"✓ Registered {table_name} with keys: {key_columns}")
        return table_config

    def create_snapshot(self, table_name: str, data: pd.DataFrame) -> Dict:
        """Create initial snapshot of table."""
        if table_name not in self.tracked_tables:
            print(f"Error: Table {table_name} not registered")
            return {}

        print(f"Creating snapshot for {table_name}")

        # Create row hashes for change detection
        config = self.tracked_tables[table_name]
        data_with_hash = data.copy()
        data_with_hash['_row_hash'] = data.apply(
            lambda row: self._hash_row(row.to_dict()), axis=1
        )

        snapshot = {
            "table_name": table_name,
            "data": data_with_hash,
            "snapshot_time": datetime.now().isoformat(),
            "row_count": len(data),
            "checksum": self._calculate_checksum(data)
        }

        self.snapshots[table_name] = snapshot
        self.checkpoints[table_name] = datetime.now()

        print(f"✓ Snapshot created: {len(data)} rows")
        return snapshot

    def capture_changes(self, table_name: str, current_data: pd.DataFrame) -> List[Dict]:
        """Capture changes since last checkpoint."""
        if table_name not in self.tracked_tables:
            print(f"Error: Table {table_name} not registered")
            return []

        if table_name not in self.snapshots:
            print(f"No baseline snapshot. Creating initial snapshot...")
            self.create_snapshot(table_name, current_data)
            return []

        print(f"Capturing changes for {table_name}")

        config = self.tracked_tables[table_name]
        key_cols = config["key_columns"]
        snapshot_data = self.snapshots[table_name]["data"]

        changes = []

        if self.method == CDCMethod.TIMESTAMP:
            changes = self._timestamp_based_cdc(
                table_name, snapshot_data, current_data, config
            )
        elif self.method == CDCMethod.SNAPSHOT:
            changes = self._snapshot_based_cdc(
                table_name, snapshot_data, current_data, key_cols
            )
        elif self.method == CDCMethod.LOG_BASED:
            # Simulated log-based CDC
            changes = self._log_based_cdc(
                table_name, snapshot_data, current_data, key_cols
            )

        # Update snapshot
        current_with_hash = current_data.copy()
        current_with_hash['_row_hash'] = current_data.apply(
            lambda row: self._hash_row(row.to_dict()), axis=1
        )
        self.snapshots[table_name]["data"] = current_with_hash
        self.snapshots[table_name]["snapshot_time"] = datetime.now().isoformat()

        # Update checkpoint
        self.checkpoints[table_name] = datetime.now()
        config["last_checkpoint"] = datetime.now().isoformat()
        config["total_changes_captured"] += len(changes)

        # Log changes
        self.change_log.extend(changes)

        print(f"✓ Captured {len(changes)} changes")
        return changes

    def _timestamp_based_cdc(self, table_name: str, old_data: pd.DataFrame,
                             new_data: pd.DataFrame, config: Dict) -> List[Dict]:
        """Capture changes using timestamp column."""
        tracking_col = config["tracking_column"]
        last_checkpoint = self.checkpoints.get(table_name)

        if tracking_col not in new_data.columns:
            print(f"Warning: Tracking column {tracking_col} not found")
            return []

        # Filter rows updated since last checkpoint
        if last_checkpoint:
            new_data[tracking_col] = pd.to_datetime(new_data[tracking_col])
            changed_rows = new_data[new_data[tracking_col] > last_checkpoint]
        else:
            changed_rows = new_data

        changes = []
        key_cols = config["key_columns"]

        for _, row in changed_rows.iterrows():
            # Check if INSERT or UPDATE
            key_values = {k: row[k] for k in key_cols}

            old_row = old_data
            for k, v in key_values.items():
                old_row = old_row[old_row[k] == v]

            if len(old_row) == 0:
                change_type = ChangeType.INSERT
            else:
                change_type = ChangeType.UPDATE

            change = {
                "table": table_name,
                "change_type": change_type.value,
                "key": key_values,
                "new_values": row.to_dict(),
                "old_values": old_row.iloc[0].to_dict() if len(old_row) > 0 else {},
                "captured_at": datetime.now().isoformat()
            }

            changes.append(change)

        return changes

    def _snapshot_based_cdc(self, table_name: str, old_data: pd.DataFrame,
                           new_data: pd.DataFrame, key_cols: List[str]) -> List[Dict]:
        """Capture changes by comparing snapshots."""
        changes = []

        # Add row hashes
        old_data = old_data.copy()
        new_data = new_data.copy()

        old_data['_row_hash'] = old_data.apply(
            lambda row: self._hash_row({k: v for k, v in row.to_dict().items() if k != '_row_hash'}),
            axis=1
        )
        new_data['_row_hash'] = new_data.apply(
            lambda row: self._hash_row(row.to_dict()), axis=1
        )

        # Create key-based indices
        old_keys = set()
        new_keys = set()

        old_index = {}
        for _, row in old_data.iterrows():
            key = tuple(row[k] for k in key_cols)
            old_keys.add(key)
            old_index[key] = row

        new_index = {}
        for _, row in new_data.iterrows():
            key = tuple(row[k] for k in key_cols)
            new_keys.add(key)
            new_index[key] = row

        # Find INSERTS
        inserted_keys = new_keys - old_keys
        for key in inserted_keys:
            row = new_index[key]
            changes.append({
                "table": table_name,
                "change_type": ChangeType.INSERT.value,
                "key": {k: row[k] for k in key_cols},
                "new_values": {k: v for k, v in row.to_dict().items() if k != '_row_hash'},
                "old_values": {},
                "captured_at": datetime.now().isoformat()
            })

        # Find DELETES
        deleted_keys = old_keys - new_keys
        for key in deleted_keys:
            row = old_index[key]
            changes.append({
                "table": table_name,
                "change_type": ChangeType.DELETE.value,
                "key": {k: row[k] for k in key_cols},
                "new_values": {},
                "old_values": {k: v for k, v in row.to_dict().items() if k != '_row_hash'},
                "captured_at": datetime.now().isoformat()
            })

        # Find UPDATES
        common_keys = old_keys & new_keys
        for key in common_keys:
            old_row = old_index[key]
            new_row = new_index[key]

            if old_row['_row_hash'] != new_row['_row_hash']:
                changes.append({
                    "table": table_name,
                    "change_type": ChangeType.UPDATE.value,
                    "key": {k: new_row[k] for k in key_cols},
                    "new_values": {k: v for k, v in new_row.to_dict().items() if k != '_row_hash'},
                    "old_values": {k: v for k, v in old_row.to_dict().items() if k != '_row_hash'},
                    "captured_at": datetime.now().isoformat()
                })

        return changes

    def _log_based_cdc(self, table_name: str, old_data: pd.DataFrame,
                       new_data: pd.DataFrame, key_cols: List[str]) -> List[Dict]:
        """Simulate log-based CDC (in production would read from transaction log)."""
        # In real implementation, this would read from database transaction logs
        # For demo, falling back to snapshot-based comparison
        return self._snapshot_based_cdc(table_name, old_data, new_data, key_cols)

    def apply_changes(self, target_data: pd.DataFrame, changes: List[Dict],
                     table_name: str) -> pd.DataFrame:
        """Apply captured changes to target dataset."""
        print(f"Applying {len(changes)} changes to target")

        result = target_data.copy()
        config = self.tracked_tables[table_name]
        key_cols = config["key_columns"]

        for change in changes:
            change_type = change["change_type"]
            key = change["key"]

            if change_type == ChangeType.INSERT.value:
                # Add new row
                new_row = pd.DataFrame([change["new_values"]])
                result = pd.concat([result, new_row], ignore_index=True)

            elif change_type == ChangeType.UPDATE.value:
                # Update existing row
                mask = pd.Series([True] * len(result))
                for k, v in key.items():
                    mask &= (result[k] == v)

                for col, val in change["new_values"].items():
                    if col in result.columns:
                        result.loc[mask, col] = val

            elif change_type == ChangeType.DELETE.value:
                # Delete row
                mask = pd.Series([True] * len(result))
                for k, v in key.items():
                    mask &= (result[k] == v)

                result = result[~mask]

        print(f"✓ Applied changes. Result: {len(result)} rows")
        return result

    def get_change_summary(self, table_name: Optional[str] = None) -> Dict:
        """Get summary of captured changes."""
        if table_name:
            changes = [c for c in self.change_log if c["table"] == table_name]
        else:
            changes = self.change_log

        summary = {
            "total_changes": len(changes),
            "inserts": len([c for c in changes if c["change_type"] == ChangeType.INSERT.value]),
            "updates": len([c for c in changes if c["change_type"] == ChangeType.UPDATE.value]),
            "deletes": len([c for c in changes if c["change_type"] == ChangeType.DELETE.value]),
            "tables": {}
        }

        # Per-table summary
        for change in changes:
            table = change["table"]
            if table not in summary["tables"]:
                summary["tables"][table] = {
                    "inserts": 0,
                    "updates": 0,
                    "deletes": 0
                }

            change_type = change["change_type"]
            if change_type == ChangeType.INSERT.value:
                summary["tables"][table]["inserts"] += 1
            elif change_type == ChangeType.UPDATE.value:
                summary["tables"][table]["updates"] += 1
            elif change_type == ChangeType.DELETE.value:
                summary["tables"][table]["deletes"] += 1

        return summary

    def _hash_row(self, row_dict: Dict) -> str:
        """Create hash of row data."""
        # Remove metadata columns
        filtered = {k: v for k, v in row_dict.items()
                   if not k.startswith('_') and k != 'updated_at'}

        row_str = json.dumps(filtered, sort_keys=True, default=str)
        return hashlib.md5(row_str.encode()).hexdigest()

    def _calculate_checksum(self, data: pd.DataFrame) -> str:
        """Calculate checksum for dataset."""
        data_str = data.to_json(orient='records', date_format='iso')
        return hashlib.md5(data_str.encode()).hexdigest()


def demo():
    """Demo change data capture."""
    print("Change Data Capture Demo")
    print("="*60)

    # Initialize CDC system
    cdc = ChangeDataCapture(method=CDCMethod.SNAPSHOT)

    # 1. Register table
    print("\n1. Registering Table for CDC")
    print("-"*60)

    cdc.register_table("customers", key_columns=["customer_id"],
                      tracking_column="updated_at")

    # 2. Create initial snapshot
    print("\n2. Creating Initial Snapshot")
    print("-"*60)

    initial_data = pd.DataFrame({
        "customer_id": [1, 2, 3],
        "name": ["Alice", "Bob", "Charlie"],
        "email": ["alice@example.com", "bob@example.com", "charlie@example.com"],
        "status": ["active", "active", "active"],
        "updated_at": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-01"])
    })

    cdc.create_snapshot("customers", initial_data)

    # 3. Simulate changes
    print("\n3. Simulating Data Changes")
    print("-"*60)

    # Updated data: Bob's email changed, Charlie deleted, Dave added
    updated_data = pd.DataFrame({
        "customer_id": [1, 2, 4],
        "name": ["Alice", "Bob", "Dave"],
        "email": ["alice@example.com", "bob.new@example.com", "dave@example.com"],
        "status": ["active", "active", "active"],
        "updated_at": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-02"])
    })

    # 4. Capture changes
    print("\n4. Capturing Changes")
    print("-"*60)

    changes = cdc.capture_changes("customers", updated_data)

    print(f"\nDetected changes:")
    for change in changes:
        print(f"  {change['change_type']}: {change['key']}")
        if change['change_type'] == 'UPDATE':
            print(f"    Old: {change['old_values'].get('email')}")
            print(f"    New: {change['new_values'].get('email')}")

    # 5. Apply changes to target
    print("\n5. Applying Changes to Target")
    print("-"*60)

    target_data = initial_data.copy()
    synchronized_data = cdc.apply_changes(target_data, changes, "customers")

    print(f"\nSynchronized data:")
    print(synchronized_data[["customer_id", "name", "email"]])

    # 6. Change summary
    print("\n6. Change Summary")
    print("-"*60)

    summary = cdc.get_change_summary("customers")
    print(f"Total changes: {summary['total_changes']}")
    print(f"  Inserts: {summary['inserts']}")
    print(f"  Updates: {summary['updates']}")
    print(f"  Deletes: {summary['deletes']}")

    # 7. Multiple capture cycles
    print("\n7. Second Capture Cycle")
    print("-"*60)

    # More changes: Alice status changed
    updated_data2 = pd.DataFrame({
        "customer_id": [1, 2, 4],
        "name": ["Alice", "Bob", "Dave"],
        "email": ["alice@example.com", "bob.new@example.com", "dave@example.com"],
        "status": ["inactive", "active", "active"],
        "updated_at": pd.to_datetime(["2024-01-03", "2024-01-02", "2024-01-02"])
    })

    changes2 = cdc.capture_changes("customers", updated_data2)
    print(f"Detected {len(changes2)} changes in second cycle")

    # Overall summary
    overall_summary = cdc.get_change_summary()
    print(f"\nOverall changes across all cycles: {overall_summary['total_changes']}")

    print("\n✓ Change Data Capture Demo Complete!")


if __name__ == '__main__':
    demo()
