# Schema Registry
Centralized schema management with versioning and compatibility checking

## Overview

A comprehensive schema registry providing centralized schema management, versioning, and compatibility checking for distributed data systems. Features automated schema validation, evolution tracking, multi-format support (Avro, JSON Schema, Protobuf), and backward/forward/full compatibility modes for ensuring safe schema changes across your data ecosystem.

## Features

### Core Capabilities
- **Schema Registration**: Register and version schemas with automated tracking
- **Multiple Formats**: Support for Avro, JSON Schema, and Protobuf
- **Schema Versioning**: Automatic version management with fingerprinting
- **Compatibility Checking**: Backward, forward, full, and none modes
- **Schema Validation**: Validate data against registered schemas
- **Subject Management**: Organize schemas by subject/topic
- **Schema Retrieval**: Get schemas by ID, subject, or version

### Advanced Features
- **Schema Evolution**: Track complete schema change history
- **Compatibility Enforcement**: Prevent breaking changes
- **Version Comparison**: Compare schemas to identify differences
- **Schema Deletion**: Remove subjects and all associated schemas
- **Fingerprinting**: Unique content-based schema identification
- **Registry Reporting**: Comprehensive schema statistics and analytics

## Installation

```bash
# Clone the repository
git clone https://github.com/BrillConsulting/DataArchitecture.git
cd DataArchitecture/SchemaRegistry

# Install dependencies
pip install pandas

# Run the implementation
python schema_registry.py
```

## Usage Examples

### Register Schema

```python
from schema_registry import SchemaRegistry

# Initialize schema registry
registry = SchemaRegistry()

# Register initial schema
user_schema_v1 = registry.register_schema(
    subject="user-events",
    schema_definition={
        "type": "record",
        "name": "UserEvent",
        "fields": [
            {"name": "user_id", "type": "int"},
            {"name": "event_type", "type": "string"},
            {"name": "timestamp", "type": "string"}
        ]
    },
    schema_format="avro"
)

print(f"Registered schema: {user_schema_v1.schema_id}")
print(f"Version: {user_schema_v1.version}")
```

### Set Compatibility Mode

```python
# Set compatibility mode for subject
registry.set_compatibility_mode("user-events", "backward")

print("Compatibility mode: backward")
print("New schemas must be backward compatible")
```

### Register Compatible Schema

```python
# Register v2 with backward-compatible change (add field with default)
user_schema_v2 = registry.register_schema(
    subject="user-events",
    schema_definition={
        "type": "record",
        "name": "UserEvent",
        "fields": [
            {"name": "user_id", "type": "int"},
            {"name": "event_type", "type": "string"},
            {"name": "timestamp", "type": "string"},
            {"name": "metadata", "type": "string", "default": ""}  # New field with default
        ]
    }
)

print(f"Registered v{user_schema_v2.version}: backward compatible")
```

### Check Compatibility

```python
# Test schema compatibility before registration
is_compatible, reason = registry.check_compatibility(
    subject="user-events",
    new_schema={
        "type": "record",
        "name": "UserEvent",
        "fields": [
            {"name": "user_id", "type": "int"},
            # Removed event_type without default - not backward compatible!
            {"name": "timestamp", "type": "string"}
        ]
    },
    compatibility_mode=CompatibilityMode.BACKWARD
)

print(f"Compatible: {is_compatible}")
if not is_compatible:
    print(f"Reason: {reason}")
```

### Validate Data

```python
# Validate data against schema
valid_data = {
    "user_id": 123,
    "event_type": "login",
    "timestamp": "2024-01-15T10:30:00Z"
}

validation = registry.validate_data("user-events", valid_data, version=1)

if validation["valid"]:
    print("Data is valid!")
else:
    print("Validation errors:")
    for error in validation["errors"]:
        print(f"  - {error}")
```

### Get Schema Versions

```python
# Get all versions for a subject
versions = registry.get_all_versions("user-events")

print(f"Total versions: {len(versions)}")
for schema in versions:
    field_count = len(schema.schema_definition.get('fields', []))
    print(f"  v{schema.version}: {field_count} fields")
```

### Get Schema Evolution

```python
# Track how schema has evolved
evolution = registry.get_schema_evolution("user-events")

print("Schema Evolution History:")
for step in evolution:
    print(f"  v{step['version']}: {step['field_count']} fields")

    if "changes" in step:
        changes = step["changes"]
        if changes["added_fields"]:
            print(f"    Added: {', '.join(changes['added_fields'])}")
        if changes["removed_fields"]:
            print(f"    Removed: {', '.join(changes['removed_fields'])}")
        if changes["modified_fields"]:
            print(f"    Modified: {len(changes['modified_fields'])} fields")
```

### Get Latest Schema

```python
# Get latest version of schema
latest = registry.get_schema_by_subject("user-events")

print(f"Latest version: v{latest.version}")
print(f"Format: {latest.schema_format}")
print(f"Fingerprint: {latest.fingerprint}")
```

### Multiple Subjects

```python
# Register schema for different subject
registry.register_schema(
    subject="product-events",
    schema_definition={
        "type": "record",
        "name": "ProductEvent",
        "fields": [
            {"name": "product_id", "type": "string"},
            {"name": "action", "type": "string"},
            {"name": "price", "type": "float", "default": 0.0}
        ]
    }
)

# List all subjects
subjects = registry.get_subjects()
print(f"Registered subjects: {subjects}")
```

### Test Full Compatibility

```python
# Set full compatibility (both backward and forward)
registry.set_compatibility_mode("product-events", "full")

# Register fully compatible schema (has default)
registry.register_schema(
    subject="product-events",
    schema_definition={
        "type": "record",
        "name": "ProductEvent",
        "fields": [
            {"name": "product_id", "type": "string"},
            {"name": "action", "type": "string"},
            {"name": "price", "type": "float", "default": 0.0},
            {"name": "category", "type": "string", "default": "general"}  # New field with default
        ]
    }
)

print("Schema is fully compatible (backward + forward)")
```

### Delete Subject

```python
# Delete subject and all its schemas
result = registry.delete_subject("test-subject")
print(f"Deleted {result['deleted_versions']} schema versions")
```

### Generate Registry Report

```python
# Get comprehensive registry statistics
report = registry.generate_registry_report()

print("Schema Registry Report:")
print(f"  Total Subjects: {report['summary']['total_subjects']}")
print(f"  Total Schemas: {report['summary']['total_schemas']}")

print("\nSubject Details:")
for subj in report["subjects"]:
    print(f"  {subj['name']}:")
    print(f"    Latest: v{subj['latest_version']}")
    print(f"    Total versions: {subj['versions']}")
    print(f"    Compatibility: {subj['compatibility']}")
```

## Demo Instructions

Run the included demonstration:

```bash
python schema_registry.py
```

The demo showcases:
1. Registering initial schema
2. Setting compatibility modes
3. Registering backward-compatible schema
4. Testing incompatible schema (rejection)
5. Getting all schema versions
6. Validating data against schemas
7. Registering multiple subjects
8. Listing all subjects
9. Schema evolution tracking
10. Testing different compatibility modes
11. Generating registry report

## Key Concepts

### Schema Compatibility

Ensure safe schema evolution:
- **Backward Compatible**: New readers can read old data
  - Can add fields with defaults
  - Cannot remove fields without defaults
  - Cannot change field types

- **Forward Compatible**: Old readers can read new data
  - Can remove fields
  - Can add fields with defaults
  - Cannot change field types

- **Full Compatible**: Both backward and forward
  - Most restrictive mode
  - Safest for bidirectional compatibility

- **None**: No compatibility checking
  - Maximum flexibility
  - Riskiest for production

### Schema Versioning

Track schema evolution:
- **Automatic Versioning**: Incremental version numbers
- **Fingerprinting**: Content-based identification
- **Version History**: Complete evolution tracking
- **Version Retrieval**: Access any historical version

### Subjects

Organize schemas by topic/entity:
- **Subject Naming**: Typically topic or entity name
- **Version Series**: Each subject has version sequence
- **Independent Evolution**: Subjects evolve independently
- **Compatibility Per Subject**: Each can have different mode

### Schema Validation

Ensure data quality:
- **Structure Validation**: Check required fields
- **Type Checking**: Validate field types
- **Default Values**: Handle missing optional fields
- **Error Reporting**: Detailed validation messages

## Architecture

```
┌──────────────────────────────────────────┐
│        Schema Registry                   │
│                                          │
│  ┌────────────────────────────────┐     │
│  │  Subject: user-events          │     │
│  │  ├─ v1: 3 fields               │     │
│  │  ├─ v2: 4 fields               │     │
│  │  └─ v3: 5 fields               │     │
│  │  Compatibility: BACKWARD       │     │
│  └────────────────────────────────┘     │
│                                          │
│  ┌────────────────────────────────┐     │
│  │  Subject: product-events       │     │
│  │  ├─ v1: 3 fields               │     │
│  │  └─ v2: 4 fields               │     │
│  │  Compatibility: FULL           │     │
│  └────────────────────────────────┘     │
│                                          │
│  ┌────────────────────────────────┐     │
│  │  Compatibility Checker         │     │
│  │  - Backward                    │     │
│  │  - Forward                     │     │
│  │  - Full                        │     │
│  └────────────────────────────────┘     │
│                                          │
│  ┌────────────────────────────────┐     │
│  │  Schema Validator              │     │
│  │  - Structure                   │     │
│  │  - Types                       │     │
│  │  - Required Fields             │     │
│  └────────────────────────────────┘     │
└──────────────────────────────────────────┘
```

## Use Cases

- **Event Streaming**: Schema management for Kafka/Pulsar topics
- **API Contracts**: Maintain API request/response schemas
- **Data Serialization**: Consistent data format across services
- **Microservices**: Schema versioning for service contracts
- **ETL Pipelines**: Track data structure evolution
- **Data Lake**: Enforce schema standards
- **Compliance**: Document data structure for audits

## Best Practices

- Use backward compatibility for most use cases
- Register schemas before producing data
- Validate data against schemas before sending
- Use meaningful subject names
- Document schema changes in commit messages
- Test compatibility before deploying changes
- Monitor schema evolution over time
- Clean up unused subjects periodically

## Compatibility Mode Guide

### When to use BACKWARD
- Consumers may be older than producers
- Adding optional fields
- Removing fields not used by consumers
- Most common mode

### When to use FORWARD
- Producers may be older than consumers
- Planning to remove fields
- Consumers can handle missing fields

### When to use FULL
- Both producers and consumers may be mixed versions
- Critical systems requiring strict compatibility
- Safest but most restrictive

### When to use NONE
- Development/testing environments
- Complete schema redesigns
- When you control all consumers and producers

## Integration

Integrate with:
- Apache Kafka for topic schemas
- Apache Pulsar for message schemas
- REST APIs via service integration
- CI/CD pipelines for validation
- Data catalogs for documentation
- Code generators (Avro, Protobuf)

## Author

**Brill Consulting**
- Email: clientbrill@gmail.com
- LinkedIn: [linkedin.com/in/brillconsulting](https://linkedin.com/in/brillconsulting)
- Specialization: Data Architecture & Engineering Solutions
