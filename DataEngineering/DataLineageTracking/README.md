# 🔍 Data Lineage Tracking System

**Advanced automated lineage tracking with graph visualization and impact analysis**

---

## 📋 Overview

A comprehensive data lineage tracking system that provides end-to-end visibility into data flows, transformations, and dependencies. Built on graph-based architecture, it enables impact analysis, compliance tracking, and data governance at scale.

## ✨ Key Features

### Core Capabilities
- **Graph-Based Architecture** - Nodes and edges representing data entities and relationships
- **Forward Lineage Tracing** - Track downstream impacts of data changes (DFS algorithm)
- **Backward Lineage Tracing** - Identify upstream data sources and dependencies
- **Impact Analysis** - Comprehensive upstream and downstream dependency analysis
- **Multi-Level Tracking** - Support for source, target, transformation, and intermediate nodes
- **Schema Tracking** - Track schema definitions and changes across the pipeline
- **Metadata Management** - Rich metadata support for enhanced context

### Transformation Types Supported
- **Filter** - Data filtering and selection operations
- **Join** - Multiple table join operations
- **Aggregate** - Grouping and aggregation transformations
- **Union** - Data union operations
- **Pivot** - Pivot table transformations
- **Window** - Window function operations
- **Custom** - Custom transformation logic

### Export & Visualization
- **JSON Export** - Complete lineage graph in JSON format
- **GraphViz DOT Export** - Visual graph representation for tools like Graphviz
- **Statistics Dashboard** - Comprehensive lineage metrics and analytics

## 🏗️ Architecture

### Data Structures

```python
# Node Types
- SOURCE: Original data sources (databases, files, APIs)
- TARGET: Final data destinations (warehouses, reports)
- TRANSFORMATION: Data transformation operations
- INTERMEDIATE: Temporary datasets in the pipeline

# Edge Types
- Transformation connections with detailed metadata
- Column-level lineage tracking
- Execution timestamps and versioning
```

### Graph Model

```
[Source Node] --[Join]-->  [Intermediate Node]
     |                            |
     |                      [Aggregate]
     |                            |
     v                            v
[Transformation] ----------> [Target Node]
```

## 📦 Installation

### Requirements

```bash
pip install -r requirements.txt
```

**Dependencies:**
- Python 3.8+
- No external graph libraries required (pure Python implementation)

## 🚀 Quick Start

### Basic Usage

```python
from data_lineage_tracking import DataLineageTracker, NodeType, TransformationType

# Initialize tracker
tracker = DataLineageTracker("My Data Pipeline")

# Add source node
source_id = tracker.add_node(
    name="raw_customers",
    node_type=NodeType.SOURCE,
    schema={"customer_id": "int", "name": "string", "email": "string"},
    metadata={"source": "postgresql", "table": "customers"}
)

# Add target node
target_id = tracker.add_node(
    name="customer_report",
    node_type=NodeType.TARGET,
    metadata={"destination": "snowflake", "table": "analytics.customers"}
)

# Add transformation edge
tracker.add_edge(
    source_node_id=source_id,
    target_node_id=target_id,
    transformation_type=TransformationType.FILTER,
    transformation_logic="WHERE created_at >= '2024-01-01'",
    columns_affected=["created_at"]
)
```

### Impact Analysis

```python
# Analyze impact of changes to a source
impact = tracker.get_impact_analysis(source_id)

print(f"Upstream dependencies: {impact['upstream_dependencies']}")
print(f"Downstream impacts: {impact['downstream_impacts']}")

# Show affected nodes
for node in impact['downstream_nodes']:
    print(f"  - {node['name']} ({node['type']})")
```

### Forward Lineage Tracing

```python
# Trace forward from a source to see all downstream impacts
forward_paths = tracker.trace_forward(source_id, max_depth=5)

print(f"Found {len(forward_paths)} downstream paths")
for path in forward_paths:
    path_names = [tracker.nodes[node_id].name for node_id in path]
    print(" → ".join(path_names))
```

### Backward Lineage Tracing

```python
# Trace backward from a target to find all sources
backward_paths = tracker.trace_backward(target_id)

print(f"Found {len(backward_paths)} upstream paths")
for path in backward_paths:
    path_names = [tracker.nodes[node_id].name for node_id in path]
    print(" ← ".join(path_names))
```

### Export Lineage

```python
# Export to JSON
tracker.export_to_json("lineage.json")

# Export to GraphViz DOT format
tracker.export_to_dot("lineage.dot")

# Generate PNG (requires GraphViz installed)
# dot -Tpng lineage.dot -o lineage.png
```

### Statistics

```python
# Get comprehensive statistics
stats = tracker.get_statistics()

print(f"Total Nodes: {stats['total_nodes']}")
print(f"Total Edges: {stats['total_edges']}")
print(f"Average Connections per Node: {stats['average_connections_per_node']:.2f}")

# View most connected nodes
for node in stats['most_connected_nodes']:
    print(f"{node['name']}: {node['connections']} connections")
```

## 📊 Complete Example

```python
from data_lineage_tracking import (
    DataLineageTracker, NodeType, TransformationType
)

# Create tracker
tracker = DataLineageTracker("Customer Analytics Pipeline")

# Add sources
customers_src = tracker.add_node(
    "raw_customers", NodeType.SOURCE,
    schema={"id": "int", "name": "string", "email": "string"},
    metadata={"source": "postgres", "table": "customers"}
)

orders_src = tracker.add_node(
    "raw_orders", NodeType.SOURCE,
    schema={"order_id": "int", "customer_id": "int", "amount": "decimal"},
    metadata={"source": "postgres", "table": "orders"}
)

# Add transformation
cleaned = tracker.add_node(
    "cleaned_customers", NodeType.TRANSFORMATION,
    metadata={"operation": "data_cleaning"}
)

# Add intermediate
joined = tracker.add_node(
    "customer_orders", NodeType.INTERMEDIATE,
    metadata={"operation": "join"}
)

# Add target
final = tracker.add_node(
    "customer_ltv", NodeType.TARGET,
    metadata={"destination": "snowflake"}
)

# Add relationships
tracker.add_edge(
    customers_src, cleaned,
    TransformationType.FILTER,
    "WHERE email IS NOT NULL",
    ["email"]
)

tracker.add_edge(
    cleaned, joined,
    TransformationType.JOIN,
    "LEFT JOIN ON customer_id",
    ["customer_id"]
)

tracker.add_edge(
    orders_src, joined,
    TransformationType.JOIN,
    "LEFT JOIN ON customer_id",
    ["customer_id", "amount"]
)

tracker.add_edge(
    joined, final,
    TransformationType.AGGREGATE,
    "GROUP BY customer_id, SUM(amount)",
    ["amount"]
)

# Analyze
impact = tracker.get_impact_analysis(customers_src)
tracker.visualize_summary()

# Export
tracker.export_to_json("customer_lineage.json")
tracker.export_to_dot("customer_lineage.dot")
```

## 🎯 Use Cases

### 1. **Data Governance & Compliance**
- Track data origins for regulatory compliance (GDPR, CCPA)
- Maintain audit trails of data transformations
- Document data flows for compliance reporting

### 2. **Impact Analysis**
- Assess impact before making schema changes
- Identify affected downstream reports and dashboards
- Plan data migration strategies

### 3. **Debugging & Troubleshooting**
- Trace data quality issues to their source
- Understand complex data transformations
- Identify bottlenecks in data pipelines

### 4. **Documentation**
- Auto-generate data flow documentation
- Create visual data architecture diagrams
- Maintain up-to-date pipeline documentation

### 5. **Change Management**
- Evaluate risk of pipeline changes
- Notify stakeholders of affected datasets
- Plan rollout strategies for data changes

## 🔧 Advanced Features

### Custom Metadata

```python
tracker.add_node(
    "transformed_data",
    NodeType.TRANSFORMATION,
    metadata={
        "owner": "data-team@company.com",
        "sla": "24h",
        "criticality": "high",
        "cost_center": "analytics",
        "tags": ["pii", "customer-data"]
    }
)
```

### Column-Level Lineage

```python
tracker.add_edge(
    source_id, target_id,
    TransformationType.CUSTOM,
    transformation_logic="CONCAT(first_name, ' ', last_name) AS full_name",
    columns_affected=["first_name", "last_name", "full_name"],
    metadata={"column_mapping": {
        "full_name": ["first_name", "last_name"]
    }}
)
```

### Cycle Detection

The system automatically prevents cycles during tracing with visited node tracking.

## 📈 Performance

- **Graph Traversal:** O(V + E) using DFS algorithm
- **Node Lookup:** O(1) using hash maps
- **Memory Efficient:** Adjacency list representation
- **Scalable:** Handles thousands of nodes and edges efficiently

## 🔍 Output Examples

### JSON Export Structure

```json
{
  "project_name": "Customer Analytics Pipeline",
  "exported_at": "2025-11-05T10:30:00",
  "nodes": [
    {
      "node_id": "source_a1b2c3d4",
      "name": "raw_customers",
      "node_type": "source",
      "schema": {"id": "int", "name": "string"},
      "metadata": {"source": "postgresql"}
    }
  ],
  "edges": [
    {
      "edge_id": "edge_e5f6g7h8",
      "source_node_id": "source_a1b2c3d4",
      "target_node_id": "target_i9j0k1l2",
      "transformation_type": "filter",
      "columns_affected": ["created_at"]
    }
  ]
}
```

### Statistics Output

```
======================================================================
Data Lineage Summary: Customer Analytics Pipeline
======================================================================
Total Nodes: 15
Total Edges: 20
Avg Connections/Node: 2.67

Node Types:
  source: 5
  target: 3
  transformation: 4
  intermediate: 3

Transformation Types:
  filter: 6
  join: 8
  aggregate: 4
  custom: 2

Most Connected Nodes:
  customer_orders (intermediate): 6 connections
  raw_customers (source): 4 connections
======================================================================
```

## 🧪 Testing

Run the demo:

```bash
python data_lineage_tracking.py
```

Expected output:
- Creates sample lineage graph
- Performs impact analysis
- Traces forward and backward paths
- Exports to JSON and DOT formats

## 📚 API Reference

### DataLineageTracker Class

**Methods:**
- `add_node(name, node_type, schema, metadata)` - Add a node to the graph
- `add_edge(source_id, target_id, transformation_type, logic, columns, metadata)` - Add an edge
- `trace_forward(node_id, max_depth)` - Trace downstream paths
- `trace_backward(node_id, max_depth)` - Trace upstream paths
- `get_impact_analysis(node_id)` - Analyze impact of node
- `export_to_json(filepath)` - Export lineage to JSON
- `export_to_dot(filepath)` - Export to GraphViz DOT
- `get_statistics()` - Get lineage statistics
- `visualize_summary()` - Print visual summary

### Enums

**NodeType:**
- `SOURCE` - Data source
- `TARGET` - Data destination
- `TRANSFORMATION` - Transformation operation
- `INTERMEDIATE` - Intermediate dataset

**TransformationType:**
- `FILTER` - Filtering operation
- `JOIN` - Join operation
- `AGGREGATE` - Aggregation
- `UNION` - Union operation
- `PIVOT` - Pivot operation
- `WINDOW` - Window function
- `CUSTOM` - Custom transformation

## 🤝 Contributing

This is a portfolio project by Brill Consulting. For questions or suggestions, contact [clientbrill@gmail.com](mailto:clientbrill@gmail.com).

## 📄 License

Created by Brill Consulting for portfolio demonstration purposes.

---

**Author:** Brill Consulting
**LinkedIn:** [brillconsulting](https://www.linkedin.com/in/brillconsulting)
**Email:** clientbrill@gmail.com
