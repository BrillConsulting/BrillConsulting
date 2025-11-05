# Master Data Management (MDM)
Entity resolution and golden record creation with data quality management

## Overview

A comprehensive Master Data Management solution providing entity resolution, deduplication, and golden record creation capabilities. Features fuzzy matching algorithms, survivorship rules, data quality scoring, and relationship management for maintaining high-quality master data across the enterprise.

## Features

### Core Capabilities
- **Entity Registration**: Register entities from multiple source systems
- **Match Rule Definition**: Configure matching rules with custom thresholds
- **Entity Matching**: Find duplicate entities using fuzzy matching algorithms
- **Entity Merging**: Merge duplicates into consolidated golden records
- **Survivorship Rules**: Define strategies for selecting best attribute values
- **Quality Scoring**: Automatic data quality assessment
- **Relationship Management**: Track entity relationships and hierarchies

### Advanced Features
- **Fuzzy Matching**: Intelligent similarity detection for near-duplicates
- **Golden Record Management**: Maintain authoritative master records
- **Multi-Source Consolidation**: Merge data from heterogeneous sources
- **Source Prioritization**: Configure source system trust levels
- **Automatic Deduplication**: Batch deduplication with auto-merge option
- **Entity Lineage**: Track source entities contributing to golden records
- **MDM Reporting**: Comprehensive reporting and analytics

## Installation

```bash
# Clone the repository
git clone https://github.com/BrillConsulting/DataArchitecture.git
cd DataArchitecture/MasterDataManagement

# Install dependencies
pip install pandas

# Run the implementation
python masterdatamanagement.py
```

## Usage Examples

### Register Entities

```python
from masterdatamanagement import MasterDataManagement

# Initialize MDM system
mdm = MasterDataManagement()

# Register entity from CRM
mdm.register_entity(
    entity_id="customer_001_crm",
    entity_type="customer",
    attributes={
        "name": "John Smith",
        "email": "john.smith@example.com",
        "phone": "555-0100",
        "address": "123 Main St"
    },
    source="CRM"
)

# Register same customer from ERP (slight variation)
mdm.register_entity(
    entity_id="customer_001_erp",
    entity_type="customer",
    attributes={
        "name": "Jon Smith",  # Slight name variation
        "email": "john.smith@example.com",
        "phone": "555-0100",
        "company": "Acme Corp"  # Additional field
    },
    source="ERP"
)
```

### Define Match Rules

```python
# Create matching rule for customers
mdm.define_match_rule(
    rule_id="customer_email_match",
    entity_type="customer",
    rule={
        "fields": ["email", "phone"],
        "threshold": 0.85,
        "method": "fuzzy"
    }
)

print("Match rule defined: customer_email_match")
```

### Find Matching Entities

```python
# Find duplicates for specific entity
matches = mdm.find_matches("customer_001_crm", "customer_email_match")

print(f"Found {len(matches)} potential duplicates:")
for match_id, score in matches:
    print(f"  {match_id}: {score:.2%} match")
```

### Define Survivorship Rules

```python
# Define how to merge duplicate attributes
mdm.define_survivorship_rule(
    rule_id="customer_merge_rule",
    rule={
        "strategy": "most_complete",  # Use record with most fields
        "source_priority": ["CRM", "ERP", "Website"],  # Prefer CRM data
        "field_rules": {
            "email": "most_recent",
            "phone": "source_priority"
        }
    }
)
```

### Merge Entities into Golden Record

```python
# Merge duplicate entities
golden_record = mdm.merge_entities(
    entity_ids=["customer_001_crm", "customer_001_erp"],
    survivorship_rule="customer_merge_rule"
)

print(f"Golden Record: {golden_record.record_id}")
print(f"Confidence: {golden_record.confidence_score:.1f}")

print("\nConsolidated Attributes:")
for key, value in golden_record.attributes.items():
    print(f"  {key}: {value}")
```

### Update Golden Record

```python
# Update golden record with new information
mdm.update_golden_record(
    record_id=golden_record.record_id,
    updates={
        "verified": True,
        "last_contact": "2024-01-15",
        "segment": "premium"
    }
)

print("Golden record updated successfully")
```

### Create Entity Relationships

```python
# Register account entity
mdm.register_entity(
    entity_id="account_001",
    entity_type="account",
    attributes={
        "account_number": "ACC-12345",
        "type": "premium",
        "balance": 50000
    },
    source="ERP"
)

# Create relationship between customer and account
relationship = mdm.create_relationship(
    entity1_id=golden_record.record_id,
    entity2_id="account_001",
    relationship_type="owns",
    metadata={"since": "2020-01-01"}
)

print(f"Relationship created: {relationship['type']}")
```

### Get Entity Lineage

```python
# Track which golden records contain an entity
lineage = mdm.get_entity_lineage("customer_001_crm")

print(f"Entity Lineage for customer_001_crm:")
print(f"  Golden Records: {lineage['golden_records']}")
print(f"  Relationships: {len(lineage['relationships'])}")
```

### Batch Deduplication

```python
# Deduplicate all customers
dedup_result = mdm.deduplicate(
    entity_type="customer",
    auto_merge=False  # Set True to automatically merge
)

print(f"Deduplication Results:")
print(f"  Duplicate groups: {dedup_result['duplicate_groups']}")
print(f"  Total duplicates: {dedup_result['total_duplicates']}")

# Review duplicate groups
for i, group in enumerate(dedup_result['groups'][:5], 1):
    print(f"\n  Group {i}:")
    print(f"    Entities: {group['entities']}")
    print(f"    Match scores: {[f'{s:.2f}' for s in group['match_scores']]}")
```

### Generate MDM Report

```python
# Get comprehensive MDM statistics
report = mdm.generate_mdm_report()

print("MDM System Report:")
print(f"  Total Entities: {report['summary']['total_entities']}")
print(f"  Golden Records: {report['summary']['total_golden_records']}")
print(f"  Relationships: {report['summary']['total_relationships']}")

print("\nEntity Types:")
for entity_type, count in report['entity_types'].items():
    print(f"  {entity_type}: {count}")

print("\nQuality Statistics:")
print(f"  Avg Quality Score: {report['quality_stats']['avg_quality']:.1f}")
print(f"  Min Quality: {report['quality_stats']['min_quality']:.1f}")
print(f"  Max Quality: {report['quality_stats']['max_quality']:.1f}")
```

## Demo Instructions

Run the included demonstration:

```bash
python masterdatamanagement.py
```

The demo showcases:
1. Registering entities from multiple sources
2. Defining match rules for entity resolution
3. Finding duplicate entities
4. Defining survivorship rules for merging
5. Merging duplicates into golden records
6. Updating golden records
7. Creating entity relationships
8. Getting entity lineage
9. Batch deduplication across entity types
10. Generating comprehensive MDM reports

## Key Concepts

### Golden Records

The authoritative master record:
- **Single Source of Truth**: One consolidated record per entity
- **Multi-Source**: Combines data from multiple systems
- **Confidence Score**: Quality measure for the golden record
- **Source Tracking**: Know which sources contributed

### Entity Resolution

Identifying duplicate entities:
- **Exact Matching**: Identical key values
- **Fuzzy Matching**: Similarity-based matching
- **Match Scores**: Confidence in match quality
- **Configurable Thresholds**: Control precision vs. recall

### Survivorship Rules

Determining best attribute values:
- **Most Complete**: Choose record with most filled fields
- **Most Recent**: Use most recently updated value
- **Source Priority**: Trust specific sources more
- **Custom Rules**: Field-specific strategies

### Data Quality Scoring

Automatic quality assessment:
- **Completeness**: Percentage of filled fields
- **Accuracy**: Validation against rules
- **Consistency**: Cross-field validation
- **Timeliness**: Data freshness

## Architecture

```
┌────────────────────────────────────────────┐
│     Master Data Management System          │
│                                            │
│  ┌──────────┐  ┌──────────┐  ┌─────────┐  │
│  │   CRM    │  │   ERP    │  │ Website │  │
│  │ Entities │  │ Entities │  │Entities │  │
│  └────┬─────┘  └────┬─────┘  └────┬────┘  │
│       │             │             │        │
│       └─────────────┼─────────────┘        │
│                     │                      │
│                     ▼                      │
│       ┌──────────────────────────┐         │
│       │   Entity Resolution      │         │
│       │  - Match Rules           │         │
│       │  - Fuzzy Matching        │         │
│       │  - Quality Scoring       │         │
│       └──────────┬───────────────┘         │
│                  │                         │
│                  ▼                         │
│       ┌──────────────────────────┐         │
│       │  Survivorship Engine     │         │
│       │  - Merge Logic           │         │
│       │  - Source Priority       │         │
│       └──────────┬───────────────┘         │
│                  │                         │
│                  ▼                         │
│       ┌──────────────────────────┐         │
│       │    Golden Records        │         │
│       │  - Master Data           │         │
│       │  - Relationships         │         │
│       │  - Quality Scores        │         │
│       └──────────────────────────┘         │
└────────────────────────────────────────────┘
```

## Use Cases

- **Customer 360**: Unified customer view across systems
- **Product MDM**: Consistent product information
- **Vendor Management**: Consolidated supplier data
- **Data Migration**: Merge data during system consolidation
- **Data Quality**: Improve overall data quality
- **Regulatory Compliance**: Maintain accurate entity records
- **Analytics**: Reliable master data for reporting

## Best Practices

- Define clear entity types and attributes
- Configure appropriate match thresholds
- Implement comprehensive survivorship rules
- Regular quality monitoring and scoring
- Periodic deduplication runs
- Maintain audit trail of merges
- Document matching logic clearly
- Test match rules thoroughly

## Performance Considerations

- Index key matching fields
- Batch process large entity sets
- Use appropriate match thresholds
- Consider fuzzy matching performance impact
- Cache frequently accessed golden records
- Optimize survivorship rule complexity

## Integration

Integrate with:
- CRM systems (Salesforce, HubSpot)
- ERP systems (SAP, Oracle)
- Data catalogs for metadata
- Data quality tools
- ETL pipelines for data loading
- API gateways for real-time access

## Author

**Brill Consulting**
- Email: clientbrill@gmail.com
- LinkedIn: [linkedin.com/in/brillconsulting](https://linkedin.com/in/brillconsulting)
- Specialization: Data Architecture & Engineering Solutions
