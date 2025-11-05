# Change Data Capture (CDC)
Real-time data synchronization with comprehensive change tracking

## Overview

A production-ready Change Data Capture system that tracks and captures data changes in real-time for synchronizing data across systems. Supports multiple CDC methods including log-based, timestamp-based, and snapshot-based approaches with full INSERT/UPDATE/DELETE tracking and change application capabilities.

## Features

### Core Capabilities
- **Multiple CDC Methods**: Log-based, trigger-based, timestamp-based, and snapshot CDC
- **Change Type Tracking**: Full support for INSERT, UPDATE, and DELETE operations
- **Table Registration**: Register tables with custom key columns and tracking configuration
- **Snapshot Management**: Create and maintain baseline snapshots for comparison
- **Change Detection**: Automatic detection of data modifications since last checkpoint
- **Change Application**: Apply captured changes to target datasets
- **Row Hashing**: Content-based change detection using row fingerprints
- **Checkpointing**: Track processing progress with checkpoint management

### Advanced Features
- **Schema Evolution**: Handle schema changes gracefully
- **Multiple Source Support**: Capture changes from various data sources
- **Incremental Replication**: Efficient change-only data transfer
- **Change Log**: Complete audit trail of all captured changes
- **Statistics & Reporting**: Comprehensive change summaries and metrics

## Installation

```bash
# Clone the repository
git clone https://github.com/BrillConsulting/DataArchitecture.git
cd DataArchitecture/ChangeDataCapture

# Install dependencies
pip install pandas

# Run the implementation
python change_data_capture.py
```

## Usage Examples

### Basic CDC Setup

```python
from change_data_capture import ChangeDataCapture, CDCMethod
import pandas as pd

# Initialize CDC system
cdc = ChangeDataCapture(method=CDCMethod.SNAPSHOT)

# Register table for tracking
cdc.register_table(
    table_name="customers",
    key_columns=["customer_id"],
    tracking_column="updated_at"
)

# Create initial snapshot
initial_data = pd.DataFrame({
    "customer_id": [1, 2, 3],
    "name": ["Alice", "Bob", "Charlie"],
    "email": ["alice@example.com", "bob@example.com", "charlie@example.com"],
    "status": ["active", "active", "active"]
})

cdc.create_snapshot("customers", initial_data)
```

### Capturing Changes

```python
# Simulate data changes
updated_data = pd.DataFrame({
    "customer_id": [1, 2, 4],  # Charlie deleted, Dave added
    "name": ["Alice", "Bob", "Dave"],
    "email": ["alice@example.com", "bob.new@example.com", "dave@example.com"],
    "status": ["active", "active", "active"]
})

# Capture changes
changes = cdc.capture_changes("customers", updated_data)

# Review detected changes
for change in changes:
    print(f"{change['change_type']}: {change['key']}")
    if change['change_type'] == 'UPDATE':
        print(f"  Old: {change['old_values']}")
        print(f"  New: {change['new_values']}")
```

### Applying Changes

```python
# Apply changes to target dataset
target_data = initial_data.copy()
synchronized_data = cdc.apply_changes(target_data, changes, "customers")

print(f"Synchronized data: {len(synchronized_data)} rows")
```

### Change Summary

```python
# Get comprehensive change summary
summary = cdc.get_change_summary("customers")

print(f"Total changes: {summary['total_changes']}")
print(f"  Inserts: {summary['inserts']}")
print(f"  Updates: {summary['updates']}")
print(f"  Deletes: {summary['deletes']}")
```

### Timestamp-Based CDC

```python
# Use timestamp-based CDC
cdc_timestamp = ChangeDataCapture(method=CDCMethod.TIMESTAMP)

cdc_timestamp.register_table(
    "orders",
    key_columns=["order_id"],
    tracking_column="modified_at"
)

# Capture only recent changes
changes = cdc_timestamp.capture_changes("orders", current_data)
```

## Demo Instructions

Run the included demonstration:

```bash
python change_data_capture.py
```

The demo showcases:
1. Table registration with key columns
2. Initial snapshot creation
3. Simulated data changes (inserts, updates, deletes)
4. Change capture and detection
5. Change application to target dataset
6. Multiple capture cycles
7. Comprehensive change summary

## Key Concepts

### CDC Methods

- **Log-Based CDC**: Reads database transaction logs (most efficient, requires database support)
- **Timestamp-Based CDC**: Uses timestamp columns to identify changed rows
- **Snapshot CDC**: Compares full dataset snapshots to detect changes
- **Trigger-Based CDC**: Uses database triggers to capture changes

### Change Types

- **INSERT**: New rows added to the dataset
- **UPDATE**: Existing rows modified
- **DELETE**: Rows removed from the dataset

### Snapshots

Baseline copies of datasets used for comparison. The system maintains:
- Row-level hashes for efficient change detection
- Checksums for data integrity verification
- Checkpoint timestamps for incremental processing

### Change Application

The system can apply captured changes to target datasets:
- Insert new rows
- Update existing rows by key
- Delete removed rows
- Maintain referential integrity

## Architecture

```
┌─────────────────┐
│  Source Tables  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ CDC Capture     │
│ - Snapshots     │
│ - Checkpoints   │
│ - Row Hashing   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Change Log     │
│ - INSERT        │
│ - UPDATE        │
│ - DELETE        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Target Systems  │
│ - Apply Changes │
│ - Synchronize   │
└─────────────────┘
```

## Use Cases

- **Data Replication**: Real-time data synchronization between systems
- **Data Warehousing**: Incremental ETL for data warehouses
- **Event Streaming**: Publish change events to message brokers
- **Audit Trails**: Track all data modifications for compliance
- **Cache Invalidation**: Update caches when source data changes
- **Microservices Sync**: Keep distributed databases synchronized

## Performance Considerations

- Use log-based CDC for high-volume databases
- Implement appropriate checkpoint intervals
- Consider batch processing for large change sets
- Monitor snapshot storage requirements
- Use timestamp-based CDC when log access is unavailable

## Author

**Brill Consulting**
- Email: clientbrill@gmail.com
- LinkedIn: [linkedin.com/in/brillconsulting](https://linkedin.com/in/brillconsulting)
- Specialization: Data Architecture & Engineering Solutions
