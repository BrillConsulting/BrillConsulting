"""
Data Lineage Tracking
=====================

End-to-end data lineage tracking and visualization:
- Column-level lineage
- Transformation tracking
- Impact analysis
- Dependency mapping
- Lineage visualization
- Automated extraction
- Lineage versioning

Author: Brill Consulting
"""

import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict
import json


class LineageNode:
    """Represents a node in the lineage graph."""

    def __init__(self, node_id: str, node_type: str, metadata: Dict = None):
        self.node_id = node_id
        self.node_type = node_type  # table, column, transformation, dataset
        self.metadata = metadata or {}
        self.created_at = datetime.now().isoformat()


class LineageEdge:
    """Represents an edge in the lineage graph."""

    def __init__(self, source: str, target: str, edge_type: str, metadata: Dict = None):
        self.source = source
        self.target = target
        self.edge_type = edge_type  # feeds, derives, transforms
        self.metadata = metadata or {}
        self.created_at = datetime.now().isoformat()


class DataLineage:
    """Data lineage tracking system."""

    def __init__(self):
        """Initialize lineage system."""
        self.nodes = {}
        self.edges = []
        self.transformations = {}
        self.versions = {}

    def register_dataset(self, dataset_id: str, schema: Dict,
                        metadata: Dict = None) -> LineageNode:
        """Register dataset in lineage graph."""
        print(f"Registering dataset: {dataset_id}")

        node = LineageNode(dataset_id, "dataset", {
            "schema": schema,
            "metadata": metadata or {}
        })

        self.nodes[dataset_id] = node

        # Register columns as separate nodes
        for column_name, column_type in schema.items():
            column_id = f"{dataset_id}.{column_name}"
            column_node = LineageNode(column_id, "column", {
                "parent_dataset": dataset_id,
                "data_type": column_type
            })
            self.nodes[column_id] = column_node

        print(f"✓ Registered {dataset_id} with {len(schema)} columns")
        return node

    def add_transformation(self, transformation_id: str, source_datasets: List[str],
                          target_dataset: str, logic: str,
                          column_mappings: List[Dict] = None) -> Dict:
        """Add transformation to lineage."""
        print(f"Adding transformation: {transformation_id}")

        transformation = {
            "transformation_id": transformation_id,
            "sources": source_datasets,
            "target": target_dataset,
            "logic": logic,
            "column_mappings": column_mappings or [],
            "created_at": datetime.now().isoformat()
        }

        self.transformations[transformation_id] = transformation

        # Create transformation node
        trans_node = LineageNode(transformation_id, "transformation", {
            "logic": logic
        })
        self.nodes[transformation_id] = trans_node

        # Add edges from sources to transformation
        for source in source_datasets:
            edge = LineageEdge(source, transformation_id, "feeds", {
                "transformation_logic": logic
            })
            self.edges.append(edge)

        # Add edge from transformation to target
        edge = LineageEdge(transformation_id, target_dataset, "produces")
        self.edges.append(edge)

        # Add column-level lineage
        if column_mappings:
            for mapping in column_mappings:
                source_col = mapping.get("source_column")
                target_col = mapping.get("target_column")
                transformation_logic = mapping.get("logic", "direct_copy")

                if source_col and target_col:
                    col_edge = LineageEdge(
                        source_col, target_col, "transforms",
                        {"logic": transformation_logic}
                    )
                    self.edges.append(col_edge)

        print(f"✓ Added transformation with {len(column_mappings or [])} column mappings")
        return transformation

    def track_column_lineage(self, source_column: str, target_column: str,
                            transformation: str, metadata: Dict = None):
        """Track column-level lineage."""
        edge = LineageEdge(source_column, target_column, "derives", {
            "transformation": transformation,
            "metadata": metadata or {}
        })
        self.edges.append(edge)

        print(f"✓ Tracked lineage: {source_column} -> {target_column}")

    def get_upstream_lineage(self, node_id: str, max_depth: int = 10) -> Dict:
        """Get upstream lineage for a node."""
        print(f"Getting upstream lineage for: {node_id}")

        lineage = {
            "node_id": node_id,
            "upstream": [],
            "paths": []
        }

        visited = set()
        self._traverse_upstream(node_id, lineage, visited, depth=0, max_depth=max_depth, path=[])

        print(f"✓ Found {len(lineage['upstream'])} upstream dependencies")
        return lineage

    def get_downstream_lineage(self, node_id: str, max_depth: int = 10) -> Dict:
        """Get downstream lineage for a node."""
        print(f"Getting downstream lineage for: {node_id}")

        lineage = {
            "node_id": node_id,
            "downstream": [],
            "paths": []
        }

        visited = set()
        self._traverse_downstream(node_id, lineage, visited, depth=0, max_depth=max_depth, path=[])

        print(f"✓ Found {len(lineage['downstream'])} downstream dependencies")
        return lineage

    def get_column_lineage(self, column_id: str, direction: str = "both") -> Dict:
        """Get column-level lineage."""
        lineage = {
            "column_id": column_id,
            "upstream_columns": [],
            "downstream_columns": []
        }

        if direction in ["upstream", "both"]:
            upstream = self.get_upstream_lineage(column_id, max_depth=5)
            lineage["upstream_columns"] = [
                node for node in upstream["upstream"]
                if self.nodes.get(node, {}).node_type == "column"
            ]

        if direction in ["downstream", "both"]:
            downstream = self.get_downstream_lineage(column_id, max_depth=5)
            lineage["downstream_columns"] = [
                node for node in downstream["downstream"]
                if self.nodes.get(node, {}).node_type == "column"
            ]

        return lineage

    def impact_analysis(self, node_id: str) -> Dict:
        """Analyze impact of changes to a node."""
        print(f"Analyzing impact for: {node_id}")

        downstream = self.get_downstream_lineage(node_id)

        # Categorize impacted nodes
        impacted = {
            "source_node": node_id,
            "total_impacted": len(downstream["downstream"]),
            "impacted_datasets": [],
            "impacted_columns": [],
            "impacted_transformations": [],
            "impact_paths": downstream["paths"]
        }

        for node_id in downstream["downstream"]:
            if node_id not in self.nodes:
                continue

            node = self.nodes[node_id]
            if node.node_type == "dataset":
                impacted["impacted_datasets"].append(node_id)
            elif node.node_type == "column":
                impacted["impacted_columns"].append(node_id)
            elif node.node_type == "transformation":
                impacted["impacted_transformations"].append(node_id)

        print(f"✓ Impact analysis: {impacted['total_impacted']} nodes impacted")
        return impacted

    def get_dependency_graph(self, node_id: str) -> Dict:
        """Get full dependency graph for a node."""
        upstream = self.get_upstream_lineage(node_id)
        downstream = self.get_downstream_lineage(node_id)

        graph = {
            "center_node": node_id,
            "upstream_nodes": upstream["upstream"],
            "downstream_nodes": downstream["downstream"],
            "total_dependencies": len(upstream["upstream"]) + len(downstream["downstream"])
        }

        return graph

    def create_lineage_snapshot(self, snapshot_id: str) -> Dict:
        """Create versioned snapshot of lineage."""
        snapshot = {
            "snapshot_id": snapshot_id,
            "timestamp": datetime.now().isoformat(),
            "node_count": len(self.nodes),
            "edge_count": len(self.edges),
            "transformation_count": len(self.transformations),
            "nodes": {k: {
                "node_id": v.node_id,
                "node_type": v.node_type,
                "metadata": v.metadata
            } for k, v in self.nodes.items()},
            "edges": [{
                "source": e.source,
                "target": e.target,
                "edge_type": e.edge_type,
                "metadata": e.metadata
            } for e in self.edges]
        }

        self.versions[snapshot_id] = snapshot
        print(f"✓ Created lineage snapshot: {snapshot_id}")
        return snapshot

    def export_lineage(self, format: str = "json") -> str:
        """Export lineage in various formats."""
        if format == "json":
            lineage_data = {
                "nodes": [{
                    "id": node.node_id,
                    "type": node.node_type,
                    "metadata": node.metadata
                } for node in self.nodes.values()],
                "edges": [{
                    "source": edge.source,
                    "target": edge.target,
                    "type": edge.edge_type,
                    "metadata": edge.metadata
                } for edge in self.edges]
            }
            return json.dumps(lineage_data, indent=2)

        elif format == "dot":
            # GraphViz DOT format
            dot = "digraph lineage {\n"
            for node in self.nodes.values():
                dot += f'  "{node.node_id}" [label="{node.node_id}" shape=box];\n'
            for edge in self.edges:
                dot += f'  "{edge.source}" -> "{edge.target}" [label="{edge.edge_type}"];\n'
            dot += "}\n"
            return dot

        return ""

    def _traverse_upstream(self, node_id: str, lineage: Dict, visited: Set,
                          depth: int, max_depth: int, path: List):
        """Traverse upstream lineage recursively."""
        if depth >= max_depth or node_id in visited:
            return

        visited.add(node_id)

        for edge in self.edges:
            if edge.target == node_id:
                source = edge.source
                if source not in lineage["upstream"]:
                    lineage["upstream"].append(source)

                new_path = path + [{"from": source, "to": node_id, "type": edge.edge_type}]

                if depth < max_depth - 1:
                    self._traverse_upstream(source, lineage, visited, depth + 1, max_depth, new_path)
                else:
                    lineage["paths"].append(new_path)

    def _traverse_downstream(self, node_id: str, lineage: Dict, visited: Set,
                            depth: int, max_depth: int, path: List):
        """Traverse downstream lineage recursively."""
        if depth >= max_depth or node_id in visited:
            return

        visited.add(node_id)

        for edge in self.edges:
            if edge.source == node_id:
                target = edge.target
                if target not in lineage["downstream"]:
                    lineage["downstream"].append(target)

                new_path = path + [{"from": node_id, "to": target, "type": edge.edge_type}]

                if depth < max_depth - 1:
                    self._traverse_downstream(target, lineage, visited, depth + 1, max_depth, new_path)
                else:
                    lineage["paths"].append(new_path)


def demo():
    """Demo data lineage tracking."""
    print("Data Lineage Tracking Demo")
    print("="*60)

    lineage = DataLineage()

    # 1. Register datasets
    print("\n1. Registering Datasets")
    print("-"*60)

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

    lineage.register_dataset("enriched_orders", {
        "order_id": "int",
        "customer_id": "int",
        "customer_name": "string",
        "amount": "decimal",
        "order_date": "timestamp"
    })

    # 2. Add transformations
    print("\n2. Adding Transformations")
    print("-"*60)

    lineage.add_transformation(
        "enrich_orders_transform",
        source_datasets=["raw_orders", "raw_customers"],
        target_dataset="enriched_orders",
        logic="JOIN raw_orders ON raw_customers.customer_id",
        column_mappings=[
            {"source_column": "raw_orders.order_id", "target_column": "enriched_orders.order_id", "logic": "direct"},
            {"source_column": "raw_orders.customer_id", "target_column": "enriched_orders.customer_id", "logic": "direct"},
            {"source_column": "raw_customers.name", "target_column": "enriched_orders.customer_name", "logic": "direct"},
            {"source_column": "raw_orders.amount", "target_column": "enriched_orders.amount", "logic": "direct"},
        ]
    )

    # 3. Get upstream lineage
    print("\n3. Upstream Lineage")
    print("-"*60)

    upstream = lineage.get_upstream_lineage("enriched_orders")
    print(f"Upstream dependencies for enriched_orders:")
    for dep in upstream["upstream"]:
        print(f"  - {dep}")

    # 4. Column-level lineage
    print("\n4. Column-Level Lineage")
    print("-"*60)

    col_lineage = lineage.get_column_lineage("enriched_orders.customer_name")
    print(f"Column lineage for enriched_orders.customer_name:")
    print(f"  Upstream columns: {col_lineage['upstream_columns']}")

    # 5. Impact analysis
    print("\n5. Impact Analysis")
    print("-"*60)

    impact = lineage.impact_analysis("raw_customers")
    print(f"Impact if raw_customers changes:")
    print(f"  Impacted datasets: {impact['impacted_datasets']}")
    print(f"  Impacted columns: {len(impact['impacted_columns'])}")

    # 6. Create snapshot
    print("\n6. Creating Lineage Snapshot")
    print("-"*60)

    snapshot = lineage.create_lineage_snapshot("v1.0")
    print(f"Snapshot created with {snapshot['node_count']} nodes and {snapshot['edge_count']} edges")

    # 7. Export lineage
    print("\n7. Exporting Lineage")
    print("-"*60)

    json_export = lineage.export_lineage("json")
    print(f"Exported lineage (first 200 chars):")
    print(json_export[:200] + "...")

    print("\n✓ Data Lineage Demo Complete!")


if __name__ == '__main__':
    demo()
