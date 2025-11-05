# Data Lineage Tracking
End-to-end data lineage with column-level tracking and impact analysis

## Overview

A comprehensive data lineage tracking system that provides complete visibility into data flows, transformations, and dependencies across your data ecosystem. Features column-level lineage tracking, automated impact analysis, and visualization-ready lineage graphs for understanding data provenance and managing data pipelines effectively.

## Features

### Core Capabilities
- **Dataset Registration**: Register datasets with schemas and metadata
- **Column-Level Lineage**: Track individual column dependencies and transformations
- **Transformation Tracking**: Document data transformations with logic and mappings
- **Upstream Lineage**: Trace data origins and dependencies
- **Downstream Lineage**: Identify consumers and downstream impacts
- **Dependency Mapping**: Build complete dependency graphs
- **Impact Analysis**: Assess downstream effects of schema or data changes

### Advanced Features
- **Lineage Graph**: Build and traverse directed acyclic graphs (DAGs)
- **Path Discovery**: Find all paths between data sources and targets
- **Lineage Versioning**: Create snapshots of lineage for historical tracking
- **Multi-Format Export**: Export lineage in JSON and GraphViz DOT formats
- **Automated Extraction**: Extract lineage from transformation code
- **Relationship Types**: Support for feeds, derives, and transforms relationships

## Installation

```bash
# Clone the repository
git clone https://github.com/BrillConsulting/DataArchitecture.git
cd DataArchitecture/DataLineage

# Install dependencies
pip install pandas

# Run the implementation
python data_lineage.py
```

## Usage Examples

### Register Datasets

```python
from data_lineage import DataLineage

# Initialize lineage system
lineage = DataLineage()

# Register source datasets
lineage.register_dataset("raw_orders", {
    "order_id": "int",
    "customer_id": "int",
    "amount": "decimal",
    "order_date": "timestamp"
})

lineage.register_dataset("raw_customers", {
    "customer_id": "int",
    "name": "string",
    "email": "string"
})

# Register target dataset
lineage.register_dataset("enriched_orders", {
    "order_id": "int",
    "customer_id": "int",
    "customer_name": "string",
    "amount": "decimal",
    "order_date": "timestamp"
})
```

### Add Transformations

```python
# Document transformation with column mappings
lineage.add_transformation(
    transformation_id="enrich_orders_transform",
    source_datasets=["raw_orders", "raw_customers"],
    target_dataset="enriched_orders",
    logic="JOIN raw_orders ON raw_customers.customer_id",
    column_mappings=[
        {
            "source_column": "raw_orders.order_id",
            "target_column": "enriched_orders.order_id",
            "logic": "direct"
        },
        {
            "source_column": "raw_customers.name",
            "target_column": "enriched_orders.customer_name",
            "logic": "direct"
        }
    ]
)
```

### Track Column-Level Lineage

```python
# Track specific column transformations
lineage.track_column_lineage(
    source_column="raw_orders.amount",
    target_column="enriched_orders.amount",
    transformation="direct_copy"
)

# Get column lineage
col_lineage = lineage.get_column_lineage("enriched_orders.customer_name")
print(f"Upstream columns: {col_lineage['upstream_columns']}")
print(f"Downstream columns: {col_lineage['downstream_columns']}")
```

### Upstream and Downstream Analysis

```python
# Get upstream dependencies
upstream = lineage.get_upstream_lineage("enriched_orders")
print(f"Upstream dependencies:")
for dep in upstream["upstream"]:
    print(f"  - {dep}")

# Get downstream consumers
downstream = lineage.get_downstream_lineage("raw_customers")
print(f"Downstream consumers:")
for consumer in downstream["downstream"]:
    print(f"  - {consumer}")
```

### Impact Analysis

```python
# Analyze impact of changes to a dataset
impact = lineage.impact_analysis("raw_customers")

print(f"Impact Analysis for raw_customers:")
print(f"  Impacted datasets: {len(impact['impacted_datasets'])}")
print(f"  Impacted columns: {len(impact['impacted_columns'])}")
print(f"  Impacted transformations: {len(impact['impacted_transformations'])}")

# View impact paths
for path in impact['impact_paths']:
    print(f"  Path: {' -> '.join([p['to'] for p in path])}")
```

### Export Lineage

```python
# Export as JSON
json_export = lineage.export_lineage("json")
with open("lineage.json", "w") as f:
    f.write(json_export)

# Export as GraphViz DOT format
dot_export = lineage.export_lineage("dot")
with open("lineage.dot", "w") as f:
    f.write(dot_export)
```

### Create Lineage Snapshot

```python
# Create versioned snapshot
snapshot = lineage.create_lineage_snapshot("v1.0")

print(f"Snapshot: {snapshot['snapshot_id']}")
print(f"  Nodes: {snapshot['node_count']}")
print(f"  Edges: {snapshot['edge_count']}")
print(f"  Transformations: {snapshot['transformation_count']}")
```

## Demo Instructions

Run the included demonstration:

```bash
python data_lineage.py
```

The demo showcases:
1. Dataset registration with schemas
2. Transformation tracking with column mappings
3. Upstream lineage discovery
4. Column-level lineage tracking
5. Impact analysis for schema changes
6. Lineage snapshot creation
7. Export to multiple formats

## Key Concepts

### Lineage Nodes

Represent entities in the lineage graph:
- **Datasets**: Tables, files, or data collections
- **Columns**: Individual fields within datasets
- **Transformations**: Data processing operations

### Lineage Edges

Represent relationships between nodes:
- **Feeds**: Source provides data to transformation
- **Produces**: Transformation creates target dataset
- **Transforms**: Column-level transformation
- **Derives**: Column derived from source column

### Column-Level Lineage

Track individual column dependencies:
- Source columns that feed each target column
- Transformation logic applied
- Data type changes
- Aggregations and calculations

### Impact Analysis

Understand the ripple effects of changes:
- Which datasets will be affected
- Which columns need updating
- Which transformations must be modified
- Which downstream systems are impacted

## Architecture

```
┌──────────────────────────────────────┐
│         Lineage Graph                │
│                                      │
│  ┌────────┐                          │
│  │Dataset │──feeds──┐                │
│  └────────┘         │                │
│                     ▼                │
│  ┌────────┐   ┌──────────────┐      │
│  │Dataset │──▶│Transformation│      │
│  └────────┘   └──────┬───────┘      │
│                      │               │
│                      │ produces      │
│                      ▼               │
│                 ┌────────┐           │
│                 │Dataset │           │
│                 └────────┘           │
│                                      │
│  Column-Level Tracking:              │
│  source.col ─transforms─▶ target.col │
└──────────────────────────────────────┘
```

## Use Cases

- **Data Governance**: Track data provenance and ownership
- **Impact Analysis**: Assess effects of schema changes
- **Compliance**: Demonstrate data handling for regulations
- **Debugging**: Trace data quality issues to source
- **Documentation**: Auto-generate data flow documentation
- **Migration Planning**: Understand dependencies before changes
- **Root Cause Analysis**: Identify sources of data issues

## Performance Considerations

- Use max_depth parameter to limit traversal in large graphs
- Create periodic snapshots for faster historical queries
- Index frequently queried nodes
- Consider caching for repeated lineage queries
- Use column-level lineage selectively for critical fields

## Integration

Integrate with:
- SQL parsers for automatic lineage extraction
- Data catalogs for metadata enrichment
- BI tools for visualization
- CI/CD pipelines for validation
- Data quality tools for root cause analysis

## Author

**Brill Consulting**
- Email: clientbrill@gmail.com
- LinkedIn: [linkedin.com/in/brillconsulting](https://linkedin.com/in/brillconsulting)
- Specialization: Data Architecture & Engineering Solutions
