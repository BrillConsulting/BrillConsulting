# Data Versioning
Dataset and schema versioning with time travel and rollback capabilities

## Overview

A comprehensive data versioning framework that provides git-like version control for datasets and schemas. Features complete version history tracking, branch management, time travel queries, rollback capabilities, and schema evolution tracking for managing data assets through their lifecycle.

## Features

### Core Capabilities
- **Dataset Registration**: Register datasets with initial schemas and ownership
- **Version Creation**: Create new versions with schema and metadata tracking
- **Snapshot Management**: Capture point-in-time dataset snapshots
- **Rollback Support**: Revert to any previous version
- **Time Travel Queries**: Query dataset state at any point in time
- **Version Comparison**: Compare schemas and metadata between versions
- **Version Tagging**: Tag important versions for easy reference

### Advanced Features
- **Branch Management**: Create and manage multiple development branches
- **Branch Merging**: Merge changes between branches
- **Schema Evolution**: Track complete schema change history
- **Version Lineage**: Parent-child version relationships
- **Metadata Versioning**: Track metadata changes alongside data
- **Fingerprinting**: Unique fingerprints for version identification
- **Comprehensive Reporting**: Generate version history reports

## Installation

```bash
# Clone the repository
git clone https://github.com/BrillConsulting/DataArchitecture.git
cd DataArchitecture/DataVersioning

# Install dependencies
pip install pandas

# Run the implementation
python data_versioning.py
```

## Usage Examples

### Register Dataset

```python
from data_versioning import DataVersioning

# Initialize versioning system
dv = DataVersioning()

# Register dataset with initial schema
dv.register_dataset(
    dataset_id="user_data",
    name="User Dataset",
    owner="data_team",
    initial_schema={
        "fields": [
            {"name": "user_id", "type": "int"},
            {"name": "name", "type": "string"},
            {"name": "email", "type": "string"}
        ]
    }
)
```

### Create New Version

```python
# Create version with schema evolution
dv.create_version(
    dataset_id="user_data",
    version_number="2.0.0",
    schema={
        "fields": [
            {"name": "user_id", "type": "int"},
            {"name": "name", "type": "string"},
            {"name": "email", "type": "string"},
            {"name": "phone", "type": "string"},  # New field
            {"name": "created_at", "type": "timestamp"}  # New field
        ]
    },
    metadata={"message": "Added phone and created_at fields"},
    branch="main"
)
```

### Create Snapshot

```python
# Create named snapshot
snapshot = dv.create_snapshot(
    dataset_id="user_data",
    snapshot_name="before_major_update",
    description="Snapshot before major schema refactoring"
)

print(f"Snapshot: {snapshot['snapshot_id']}")
print(f"Version: {dv.versions[snapshot['version_id']].version_number}")
```

### Rollback to Previous Version

```python
# Rollback to version 1.0.0
rollback = dv.rollback(
    dataset_id="user_data",
    target_version="1.0.0"
)

print(f"Rolled back from {rollback['from_version']} to {rollback['to_version']}")
```

### Time Travel Query

```python
from datetime import datetime

# Query dataset state at specific time
query_time = "2024-01-15T10:30:00"
result = dv.time_travel_query("user_data", query_time)

if result:
    print(f"Version at {query_time}: {result['version_number']}")
    print(f"Fields: {len(result['schema']['fields'])}")
```

### Compare Versions

```python
# Compare two versions
comparison = dv.compare_versions("user_data", "1.0.0", "2.0.0")

print("Schema Differences:")
print(f"  Added fields: {comparison['schema_diff']['added_fields']}")
print(f"  Removed fields: {comparison['schema_diff']['removed_fields']}")
print(f"  Modified fields: {len(comparison['schema_diff']['modified_fields'])}")

print("\nMetadata changes:")
print(f"  Changed keys: {comparison['metadata_diff']['changed_keys']}")
```

### Branch Management

```python
# Create development branch
dv.create_branch(
    dataset_id="user_data",
    branch_name="development",
    source_branch="main"
)

# Create version in dev branch
dv.create_version(
    dataset_id="user_data",
    version_number="2.1.0-dev",
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

# Merge back to main
merge = dv.merge_branch("user_data", "development", "main")
print(f"Merged {merge['merged_versions']} version(s)")
```

### Tag Versions

```python
# Tag production version
dv.tag_version("user_data", "2.0.0", "production")

# Tag stable release
dv.tag_version("user_data", "2.0.0", "stable")
```

### Get Version History

```python
# Get all versions on main branch
history = dv.get_version_history("user_data", branch="main")

print("Version History:")
for h in history:
    tags = f" [{', '.join(t['tag'] for t in h['tags'])}]" if h['tags'] else ""
    print(f"  v{h['version_number']}{tags}: {h['message']}")
```

### Schema Evolution

```python
# Track schema changes over time
evolution = dv.get_schema_evolution("user_data")

print("Schema Evolution:")
for i, e in enumerate(evolution, 1):
    field_count = len(e['schema'].get('fields', []))
    print(f"  {i}. v{e['version_number']}: {field_count} fields")
```

## Demo Instructions

Run the included demonstration:

```bash
python data_versioning.py
```

The demo showcases:
1. Dataset registration with initial schema
2. Creating new versions with schema changes
3. Snapshot creation for rollback points
4. Branch creation for parallel development
5. Version comparison and diff analysis
6. Version tagging for releases
7. Time travel queries
8. Version history tracking
9. Schema evolution analysis
10. Branch merging
11. Comprehensive version reporting

## Key Concepts

### Versions

Each version represents a state of the dataset:
- **Version Number**: Semantic versioning (e.g., 1.0.0, 2.1.0)
- **Schema**: Structure definition at this version
- **Metadata**: Descriptive information and change messages
- **Parent Version**: Link to previous version
- **Timestamps**: When version was created

### Branches

Parallel version streams:
- **Main Branch**: Primary development line
- **Feature Branches**: Isolated development
- **Version Isolation**: Changes don't affect other branches
- **Merging**: Combine branch versions

### Time Travel

Query historical states:
- Access any point in history
- Reproduce analyses with historical data
- Debug issues with historical context
- Compliance and audit requirements

### Schema Evolution

Track structural changes:
- Field additions and removals
- Type modifications
- Constraint changes
- Migration planning

## Architecture

```
┌─────────────────────────────────────────┐
│       Data Versioning System            │
│                                         │
│  Dataset: user_data                     │
│                                         │
│  Main Branch:                           │
│  v1.0.0 ──▶ v2.0.0 ──▶ v3.0.0          │
│             (prod)      (latest)        │
│                                         │
│  Development Branch:                    │
│              ┌─ v2.1.0-dev              │
│  v1.0.0 ──▶ v2.0.0                      │
│              └─ merge ──▶ v3.0.0        │
│                                         │
│  Snapshots:                             │
│  - before_major_update (v2.0.0)         │
│  - stable_baseline (v1.0.0)             │
│                                         │
│  Schema Evolution:                      │
│  v1.0.0: 3 fields                       │
│  v2.0.0: 5 fields (+phone, +created_at) │
│  v3.0.0: 6 fields (+status)             │
└─────────────────────────────────────────┘
```

## Use Cases

- **Schema Management**: Control schema changes across environments
- **Data Quality**: Maintain data quality through version control
- **Reproducibility**: Recreate analyses with exact historical data
- **Compliance**: Maintain audit trails of data changes
- **Development Workflow**: Parallel feature development with branches
- **Rollback Capability**: Quickly recover from problematic changes
- **Migration Planning**: Analyze impact of schema changes
- **Documentation**: Auto-generated change documentation

## Best Practices

- Use semantic versioning for clarity (major.minor.patch)
- Tag production versions for easy identification
- Create snapshots before major changes
- Write descriptive commit messages in metadata
- Use branches for experimental changes
- Maintain linear history on main branch
- Document breaking changes clearly
- Regular cleanup of old versions

## Integration

Integrate with:
- Data catalogs for metadata management
- CI/CD pipelines for automated versioning
- Data quality tools for validation
- Monitoring systems for change tracking
- Data lineage tools for provenance

## Author

**Brill Consulting**
- Email: clientbrill@gmail.com
- LinkedIn: [linkedin.com/in/brillconsulting](https://linkedin.com/in/brillconsulting)
- Specialization: Data Architecture & Engineering Solutions
