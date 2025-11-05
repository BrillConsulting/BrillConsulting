"""
Data Lineage Tracking System
Author: BrillConsulting
Description: Advanced automated lineage tracking with graph visualization and impact analysis
"""

import json
import uuid
from typing import Dict, List, Any, Optional, Set, Tuple
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, asdict, field
from collections import defaultdict


class NodeType(Enum):
    """Types of lineage nodes"""
    SOURCE = "source"
    TARGET = "target"
    TRANSFORMATION = "transformation"
    INTERMEDIATE = "intermediate"


class TransformationType(Enum):
    """Types of data transformations"""
    FILTER = "filter"
    JOIN = "join"
    AGGREGATE = "aggregate"
    UNION = "union"
    PIVOT = "pivot"
    WINDOW = "window"
    CUSTOM = "custom"


@dataclass
class LineageNode:
    """Represents a node in the lineage graph"""
    node_id: str
    name: str
    node_type: NodeType
    schema: Optional[Dict[str, str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['node_type'] = self.node_type.value
        return result


@dataclass
class LineageEdge:
    """Represents an edge (relationship) in the lineage graph"""
    edge_id: str
    source_node_id: str
    target_node_id: str
    transformation_type: TransformationType
    transformation_logic: Optional[str] = None
    columns_affected: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['transformation_type'] = self.transformation_type.value
        return result


class DataLineageTracker:
    """
    Advanced Data Lineage Tracking System

    Features:
    - Multi-source lineage tracking
    - Graph-based relationship mapping
    - Impact analysis
    - Backward and forward tracing
    - Export to multiple formats
    """

    def __init__(self, project_name: str = "default"):
        """
        Initialize lineage tracker

        Args:
            project_name: Name of the tracking project
        """
        self.project_name = project_name
        self.nodes: Dict[str, LineageNode] = {}
        self.edges: Dict[str, LineageEdge] = {}
        self.adjacency_list: Dict[str, List[str]] = defaultdict(list)
        self.reverse_adjacency_list: Dict[str, List[str]] = defaultdict(list)

    def add_node(self, name: str, node_type: NodeType,
                 schema: Optional[Dict[str, str]] = None,
                 metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Add a node to the lineage graph

        Args:
            name: Node name
            node_type: Type of node
            schema: Optional schema definition
            metadata: Optional metadata

        Returns:
            Node ID
        """
        node_id = f"{node_type.value}_{uuid.uuid4().hex[:8]}"
        node = LineageNode(
            node_id=node_id,
            name=name,
            node_type=node_type,
            schema=schema,
            metadata=metadata or {}
        )
        self.nodes[node_id] = node
        print(f"✓ Node added: {name} (ID: {node_id}, Type: {node_type.value})")
        return node_id

    def add_edge(self, source_node_id: str, target_node_id: str,
                 transformation_type: TransformationType,
                 transformation_logic: Optional[str] = None,
                 columns_affected: Optional[List[str]] = None,
                 metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Add an edge (relationship) to the lineage graph

        Args:
            source_node_id: Source node ID
            target_node_id: Target node ID
            transformation_type: Type of transformation
            transformation_logic: Optional SQL or code
            columns_affected: List of affected columns
            metadata: Optional metadata

        Returns:
            Edge ID
        """
        if source_node_id not in self.nodes:
            raise ValueError(f"Source node {source_node_id} not found")
        if target_node_id not in self.nodes:
            raise ValueError(f"Target node {target_node_id} not found")

        edge_id = f"edge_{uuid.uuid4().hex[:8]}"
        edge = LineageEdge(
            edge_id=edge_id,
            source_node_id=source_node_id,
            target_node_id=target_node_id,
            transformation_type=transformation_type,
            transformation_logic=transformation_logic,
            columns_affected=columns_affected or [],
            metadata=metadata or {}
        )
        self.edges[edge_id] = edge
        self.adjacency_list[source_node_id].append(target_node_id)
        self.reverse_adjacency_list[target_node_id].append(source_node_id)

        source_name = self.nodes[source_node_id].name
        target_name = self.nodes[target_node_id].name
        print(f"✓ Edge added: {source_name} → {target_name} ({transformation_type.value})")
        return edge_id

    def trace_forward(self, node_id: str, max_depth: Optional[int] = None) -> List[List[str]]:
        """
        Trace lineage forward from a node (downstream impact)

        Args:
            node_id: Starting node ID
            max_depth: Maximum depth to traverse

        Returns:
            List of paths (each path is a list of node IDs)
        """
        if node_id not in self.nodes:
            raise ValueError(f"Node {node_id} not found")

        paths = []

        def dfs(current_id: str, path: List[str], depth: int):
            if max_depth is not None and depth >= max_depth:
                paths.append(path.copy())
                return

            if current_id not in self.adjacency_list or not self.adjacency_list[current_id]:
                paths.append(path.copy())
                return

            for next_id in self.adjacency_list[current_id]:
                if next_id not in path:  # Avoid cycles
                    path.append(next_id)
                    dfs(next_id, path, depth + 1)
                    path.pop()

        dfs(node_id, [node_id], 0)

        print(f"✓ Forward trace from {self.nodes[node_id].name}: {len(paths)} paths found")
        return paths

    def trace_backward(self, node_id: str, max_depth: Optional[int] = None) -> List[List[str]]:
        """
        Trace lineage backward from a node (upstream sources)

        Args:
            node_id: Starting node ID
            max_depth: Maximum depth to traverse

        Returns:
            List of paths (each path is a list of node IDs)
        """
        if node_id not in self.nodes:
            raise ValueError(f"Node {node_id} not found")

        paths = []

        def dfs(current_id: str, path: List[str], depth: int):
            if max_depth is not None and depth >= max_depth:
                paths.append(path.copy())
                return

            if current_id not in self.reverse_adjacency_list or not self.reverse_adjacency_list[current_id]:
                paths.append(path.copy())
                return

            for prev_id in self.reverse_adjacency_list[current_id]:
                if prev_id not in path:  # Avoid cycles
                    path.append(prev_id)
                    dfs(prev_id, path, depth + 1)
                    path.pop()

        dfs(node_id, [node_id], 0)

        print(f"✓ Backward trace from {self.nodes[node_id].name}: {len(paths)} paths found")
        return paths

    def get_impact_analysis(self, node_id: str) -> Dict[str, Any]:
        """
        Perform impact analysis for a node

        Args:
            node_id: Node ID to analyze

        Returns:
            Impact analysis report
        """
        if node_id not in self.nodes:
            raise ValueError(f"Node {node_id} not found")

        downstream_paths = self.trace_forward(node_id)
        upstream_paths = self.trace_backward(node_id)

        # Collect all affected nodes
        downstream_nodes = set()
        for path in downstream_paths:
            downstream_nodes.update(path[1:])  # Exclude the starting node

        upstream_nodes = set()
        for path in upstream_paths:
            upstream_nodes.update(path[1:])  # Exclude the starting node

        report = {
            'node_id': node_id,
            'node_name': self.nodes[node_id].name,
            'upstream_dependencies': len(upstream_nodes),
            'downstream_impacts': len(downstream_nodes),
            'upstream_nodes': [
                {'id': nid, 'name': self.nodes[nid].name, 'type': self.nodes[nid].node_type.value}
                for nid in upstream_nodes
            ],
            'downstream_nodes': [
                {'id': nid, 'name': self.nodes[nid].name, 'type': self.nodes[nid].node_type.value}
                for nid in downstream_nodes
            ],
            'analysis_timestamp': datetime.now().isoformat()
        }

        print(f"✓ Impact analysis for {self.nodes[node_id].name}:")
        print(f"  Upstream dependencies: {report['upstream_dependencies']}")
        print(f"  Downstream impacts: {report['downstream_impacts']}")

        return report

    def export_to_json(self, filepath: str) -> None:
        """
        Export lineage graph to JSON

        Args:
            filepath: Output file path
        """
        export_data = {
            'project_name': self.project_name,
            'exported_at': datetime.now().isoformat(),
            'nodes': [node.to_dict() for node in self.nodes.values()],
            'edges': [edge.to_dict() for edge in self.edges.values()],
            'statistics': {
                'total_nodes': len(self.nodes),
                'total_edges': len(self.edges),
                'node_types': self._get_node_type_counts()
            }
        }

        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)

        print(f"✓ Lineage exported to: {filepath}")

    def export_to_dot(self, filepath: str) -> None:
        """
        Export lineage graph to GraphViz DOT format

        Args:
            filepath: Output file path
        """
        lines = ['digraph DataLineage {', '  rankdir=LR;', '  node [shape=box];', '']

        # Add nodes
        for node in self.nodes.values():
            color = {
                NodeType.SOURCE: 'lightblue',
                NodeType.TARGET: 'lightgreen',
                NodeType.TRANSFORMATION: 'lightyellow',
                NodeType.INTERMEDIATE: 'lightgray'
            }.get(node.node_type, 'white')

            lines.append(f'  "{node.node_id}" [label="{node.name}\\n({node.node_type.value})" fillcolor="{color}" style=filled];')

        lines.append('')

        # Add edges
        for edge in self.edges.values():
            label = edge.transformation_type.value
            if edge.columns_affected:
                label += f"\\n{', '.join(edge.columns_affected[:3])}"
            lines.append(f'  "{edge.source_node_id}" -> "{edge.target_node_id}" [label="{label}"];')

        lines.append('}')

        with open(filepath, 'w') as f:
            f.write('\n'.join(lines))

        print(f"✓ Lineage exported to DOT format: {filepath}")

    def _get_node_type_counts(self) -> Dict[str, int]:
        """Get counts of each node type"""
        counts = defaultdict(int)
        for node in self.nodes.values():
            counts[node.node_type.value] += 1
        return dict(counts)

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get lineage graph statistics

        Returns:
            Statistics dictionary
        """
        stats = {
            'project_name': self.project_name,
            'total_nodes': len(self.nodes),
            'total_edges': len(self.edges),
            'node_type_counts': self._get_node_type_counts(),
            'transformation_type_counts': self._get_transformation_type_counts(),
            'average_connections_per_node': len(self.edges) / len(self.nodes) if self.nodes else 0,
            'most_connected_nodes': self._get_most_connected_nodes(5)
        }

        return stats

    def _get_transformation_type_counts(self) -> Dict[str, int]:
        """Get counts of each transformation type"""
        counts = defaultdict(int)
        for edge in self.edges.values():
            counts[edge.transformation_type.value] += 1
        return dict(counts)

    def _get_most_connected_nodes(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get most connected nodes"""
        node_connections = {}
        for node_id in self.nodes:
            incoming = len(self.reverse_adjacency_list.get(node_id, []))
            outgoing = len(self.adjacency_list.get(node_id, []))
            node_connections[node_id] = incoming + outgoing

        sorted_nodes = sorted(node_connections.items(), key=lambda x: x[1], reverse=True)[:limit]

        return [
            {
                'node_id': node_id,
                'name': self.nodes[node_id].name,
                'connections': count,
                'type': self.nodes[node_id].node_type.value
            }
            for node_id, count in sorted_nodes
        ]

    def visualize_summary(self) -> None:
        """Print a visual summary of the lineage graph"""
        stats = self.get_statistics()

        print("\n" + "=" * 70)
        print(f"Data Lineage Summary: {self.project_name}")
        print("=" * 70)
        print(f"Total Nodes: {stats['total_nodes']}")
        print(f"Total Edges: {stats['total_edges']}")
        print(f"Avg Connections/Node: {stats['average_connections_per_node']:.2f}")
        print("\nNode Types:")
        for node_type, count in stats['node_type_counts'].items():
            print(f"  {node_type}: {count}")
        print("\nTransformation Types:")
        for trans_type, count in stats['transformation_type_counts'].items():
            print(f"  {trans_type}: {count}")
        print("\nMost Connected Nodes:")
        for node in stats['most_connected_nodes']:
            print(f"  {node['name']} ({node['type']}): {node['connections']} connections")
        print("=" * 70 + "\n")


def demo():
    """Demonstrate advanced data lineage tracking"""
    print("=" * 70)
    print("Advanced Data Lineage Tracking Demo")
    print("=" * 70)

    # Initialize tracker
    tracker = DataLineageTracker("Customer Analytics Pipeline")

    # Add source nodes
    print("\n1. Adding source data nodes...")
    raw_customers = tracker.add_node(
        "raw_customers",
        NodeType.SOURCE,
        schema={"customer_id": "int", "name": "string", "email": "string", "created_at": "timestamp"},
        metadata={"source": "postgresql", "table": "customers"}
    )

    raw_orders = tracker.add_node(
        "raw_orders",
        NodeType.SOURCE,
        schema={"order_id": "int", "customer_id": "int", "amount": "decimal", "order_date": "date"},
        metadata={"source": "postgresql", "table": "orders"}
    )

    raw_products = tracker.add_node(
        "raw_products",
        NodeType.SOURCE,
        schema={"product_id": "int", "product_name": "string", "category": "string"},
        metadata={"source": "postgresql", "table": "products"}
    )

    # Add transformation nodes
    print("\n2. Adding transformation nodes...")
    cleaned_customers = tracker.add_node(
        "cleaned_customers",
        NodeType.TRANSFORMATION,
        metadata={"operation": "data_cleaning"}
    )

    customer_orders = tracker.add_node(
        "customer_orders",
        NodeType.INTERMEDIATE,
        metadata={"operation": "join"}
    )

    customer_metrics = tracker.add_node(
        "customer_metrics",
        NodeType.INTERMEDIATE,
        metadata={"operation": "aggregation"}
    )

    # Add target node
    final_report = tracker.add_node(
        "customer_lifetime_value_report",
        NodeType.TARGET,
        metadata={"destination": "snowflake", "table": "analytics.customer_ltv"}
    )

    # Add edges (transformations)
    print("\n3. Adding transformation relationships...")
    tracker.add_edge(
        raw_customers, cleaned_customers,
        TransformationType.FILTER,
        transformation_logic="WHERE email IS NOT NULL AND created_at >= '2020-01-01'",
        columns_affected=["email", "created_at"]
    )

    tracker.add_edge(
        cleaned_customers, customer_orders,
        TransformationType.JOIN,
        transformation_logic="LEFT JOIN ON customers.customer_id = orders.customer_id",
        columns_affected=["customer_id"]
    )

    tracker.add_edge(
        raw_orders, customer_orders,
        TransformationType.JOIN,
        transformation_logic="LEFT JOIN ON customers.customer_id = orders.customer_id",
        columns_affected=["customer_id", "amount", "order_date"]
    )

    tracker.add_edge(
        customer_orders, customer_metrics,
        TransformationType.AGGREGATE,
        transformation_logic="GROUP BY customer_id, AGGREGATE(SUM(amount), COUNT(order_id))",
        columns_affected=["amount", "order_id"]
    )

    tracker.add_edge(
        customer_metrics, final_report,
        TransformationType.CUSTOM,
        transformation_logic="Calculate LTV = total_spent * retention_factor",
        columns_affected=["total_spent", "order_count"]
    )

    # Perform impact analysis
    print("\n4. Performing impact analysis...")
    impact = tracker.get_impact_analysis(raw_customers)

    # Trace lineage
    print("\n5. Tracing lineage...")
    forward_paths = tracker.trace_forward(raw_customers)
    print(f"Forward paths from raw_customers: {len(forward_paths)}")

    backward_paths = tracker.trace_backward(final_report)
    print(f"Backward paths to final_report: {len(backward_paths)}")

    # Show statistics
    print("\n6. Lineage statistics...")
    tracker.visualize_summary()

    # Export lineage
    print("7. Exporting lineage...")
    tracker.export_to_json("/tmp/lineage.json")
    tracker.export_to_dot("/tmp/lineage.dot")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    demo()
