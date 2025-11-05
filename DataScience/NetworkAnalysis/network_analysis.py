"""
Network Analysis Toolkit
=========================

Advanced graph and network analysis methods:
- Graph metrics (centrality, clustering, density)
- Community detection algorithms
- Network visualization
- Path finding and connectivity
- Small-world and scale-free properties
- Network motifs and patterns
- Influence and diffusion modeling

Author: Brill Consulting
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import sparse
from scipy.sparse import linalg as sp_linalg
from scipy.cluster import hierarchy
from collections import deque, defaultdict
from typing import Dict, List, Tuple, Optional, Set
import warnings
warnings.filterwarnings('ignore')


class NetworkAnalysis:
    """Network and graph analysis toolkit."""

    def __init__(self):
        """Initialize network analysis toolkit."""
        self.adjacency_matrix = None
        self.edge_list = None
        self.nodes = None
        self.communities = None

    def create_adjacency_matrix(self, edges: List[Tuple[int, int]],
                               n_nodes: Optional[int] = None,
                               weighted: bool = False,
                               weights: Optional[List[float]] = None) -> np.ndarray:
        """
        Create adjacency matrix from edge list.

        Args:
            edges: List of (source, target) tuples
            n_nodes: Number of nodes (if None, inferred from edges)
            weighted: Whether the graph is weighted
            weights: Edge weights (if weighted)

        Returns:
            Adjacency matrix
        """
        if n_nodes is None:
            n_nodes = max(max(e) for e in edges) + 1

        adj = np.zeros((n_nodes, n_nodes))

        if weighted and weights is not None:
            for (i, j), w in zip(edges, weights):
                adj[i, j] = w
                adj[j, i] = w  # Undirected
        else:
            for i, j in edges:
                adj[i, j] = 1
                adj[j, i] = 1  # Undirected

        self.adjacency_matrix = adj
        self.edge_list = edges
        self.nodes = list(range(n_nodes))

        return adj

    def degree_centrality(self, adj_matrix: Optional[np.ndarray] = None) -> Dict:
        """
        Calculate degree centrality for all nodes.

        Args:
            adj_matrix: Adjacency matrix (if None, uses self.adjacency_matrix)

        Returns:
            Dictionary with centrality metrics
        """
        if adj_matrix is None:
            adj_matrix = self.adjacency_matrix

        n = len(adj_matrix)
        degrees = np.sum(adj_matrix > 0, axis=1)

        # Normalize by (n-1)
        degree_centrality = degrees / (n - 1)

        return {
            'degree': degrees,
            'degree_centrality': degree_centrality,
            'mean_degree': np.mean(degrees),
            'max_degree': np.max(degrees),
            'degree_distribution': np.bincount(degrees.astype(int))
        }

    def betweenness_centrality(self, adj_matrix: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Calculate betweenness centrality using shortest paths.

        Args:
            adj_matrix: Adjacency matrix

        Returns:
            Array of betweenness centrality values
        """
        if adj_matrix is None:
            adj_matrix = self.adjacency_matrix

        n = len(adj_matrix)
        betweenness = np.zeros(n)

        for s in range(n):
            # Single-source shortest paths (BFS)
            stack = []
            paths = {s: [s]}
            sigma = {i: 0 for i in range(n)}
            sigma[s] = 1
            dist = {i: -1 for i in range(n)}
            dist[s] = 0
            queue = deque([s])

            while queue:
                v = queue.popleft()
                stack.append(v)

                for w in range(n):
                    if adj_matrix[v, w] > 0:
                        # Path discovery
                        if dist[w] < 0:
                            queue.append(w)
                            dist[w] = dist[v] + 1

                        # Path counting
                        if dist[w] == dist[v] + 1:
                            sigma[w] += sigma[v]
                            if w not in paths:
                                paths[w] = []
                            paths[w].extend(paths[v])

            # Accumulation
            delta = {i: 0 for i in range(n)}
            while stack:
                w = stack.pop()
                for v in paths.get(w, []):
                    if v != w and v in sigma and sigma[w] > 0:
                        delta[v] += (sigma[v] / sigma[w]) * (1 + delta[w])
                if w != s:
                    betweenness[w] += delta[w]

        # Normalize
        if n > 2:
            betweenness = betweenness / ((n - 1) * (n - 2))

        return betweenness

    def closeness_centrality(self, adj_matrix: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Calculate closeness centrality (inverse average shortest path length).

        Args:
            adj_matrix: Adjacency matrix

        Returns:
            Array of closeness centrality values
        """
        if adj_matrix is None:
            adj_matrix = self.adjacency_matrix

        n = len(adj_matrix)
        closeness = np.zeros(n)

        for i in range(n):
            # BFS to find shortest paths from node i
            dist = {j: float('inf') for j in range(n)}
            dist[i] = 0
            queue = deque([i])

            while queue:
                v = queue.popleft()
                for w in range(n):
                    if adj_matrix[v, w] > 0 and dist[w] == float('inf'):
                        dist[w] = dist[v] + 1
                        queue.append(w)

            # Calculate closeness
            total_dist = sum(d for d in dist.values() if d != float('inf') and d > 0)
            reachable = sum(1 for d in dist.values() if d != float('inf') and d > 0)

            if reachable > 0:
                closeness[i] = reachable / total_dist

        return closeness

    def pagerank(self, adj_matrix: Optional[np.ndarray] = None,
                damping: float = 0.85, max_iter: int = 100, tol: float = 1e-6) -> np.ndarray:
        """
        Calculate PageRank centrality.

        Args:
            adj_matrix: Adjacency matrix
            damping: Damping factor (typically 0.85)
            max_iter: Maximum iterations
            tol: Convergence tolerance

        Returns:
            Array of PageRank values
        """
        if adj_matrix is None:
            adj_matrix = self.adjacency_matrix

        n = len(adj_matrix)

        # Normalize adjacency matrix (column-stochastic)
        out_degree = np.sum(adj_matrix, axis=1)
        out_degree[out_degree == 0] = 1  # Avoid division by zero

        M = adj_matrix / out_degree[:, np.newaxis]

        # Initialize PageRank
        pr = np.ones(n) / n

        for _ in range(max_iter):
            pr_new = (1 - damping) / n + damping * M.T @ pr

            if np.linalg.norm(pr_new - pr, 1) < tol:
                break

            pr = pr_new

        return pr

    def clustering_coefficient(self, adj_matrix: Optional[np.ndarray] = None) -> Dict:
        """
        Calculate clustering coefficient (local and global).

        Args:
            adj_matrix: Adjacency matrix

        Returns:
            Dictionary with clustering metrics
        """
        if adj_matrix is None:
            adj_matrix = self.adjacency_matrix

        n = len(adj_matrix)
        local_clustering = np.zeros(n)

        for i in range(n):
            neighbors = np.where(adj_matrix[i] > 0)[0]
            k = len(neighbors)

            if k < 2:
                local_clustering[i] = 0
            else:
                # Count triangles
                triangles = 0
                for j in range(len(neighbors)):
                    for l in range(j + 1, len(neighbors)):
                        if adj_matrix[neighbors[j], neighbors[l]] > 0:
                            triangles += 1

                local_clustering[i] = 2 * triangles / (k * (k - 1))

        global_clustering = np.mean(local_clustering)

        return {
            'local_clustering': local_clustering,
            'global_clustering': global_clustering,
            'mean_clustering': global_clustering
        }

    def community_detection_louvain(self, adj_matrix: Optional[np.ndarray] = None,
                                   resolution: float = 1.0) -> Dict:
        """
        Detect communities using Louvain method (simplified version).

        Args:
            adj_matrix: Adjacency matrix
            resolution: Resolution parameter

        Returns:
            Dictionary with community assignments
        """
        if adj_matrix is None:
            adj_matrix = self.adjacency_matrix

        n = len(adj_matrix)
        m = np.sum(adj_matrix) / 2  # Number of edges

        # Initialize each node in its own community
        communities = np.arange(n)

        # Calculate initial modularity
        degrees = np.sum(adj_matrix, axis=1)

        improved = True
        iteration = 0

        while improved and iteration < 10:
            improved = False
            iteration += 1

            for i in range(n):
                best_community = communities[i]
                best_delta = 0

                # Try moving node i to neighboring communities
                neighbors = np.where(adj_matrix[i] > 0)[0]
                neighbor_communities = set(communities[neighbors])

                for comm in neighbor_communities:
                    if comm == communities[i]:
                        continue

                    # Calculate modularity change
                    # Simplified calculation
                    nodes_in_comm = np.where(communities == comm)[0]
                    edges_to_comm = np.sum(adj_matrix[i, nodes_in_comm])

                    delta = edges_to_comm / m - degrees[i] * np.sum(degrees[nodes_in_comm]) / (2 * m**2)

                    if delta > best_delta:
                        best_delta = delta
                        best_community = comm

                if best_community != communities[i]:
                    communities[i] = best_community
                    improved = True

        # Relabel communities sequentially
        unique_communities = np.unique(communities)
        community_map = {old: new for new, old in enumerate(unique_communities)}
        communities = np.array([community_map[c] for c in communities])

        self.communities = communities

        # Calculate modularity
        modularity = self._calculate_modularity(adj_matrix, communities)

        return {
            'communities': communities,
            'n_communities': len(unique_communities),
            'modularity': modularity,
            'community_sizes': np.bincount(communities)
        }

    def _calculate_modularity(self, adj_matrix: np.ndarray, communities: np.ndarray) -> float:
        """Calculate modularity score."""
        m = np.sum(adj_matrix) / 2
        degrees = np.sum(adj_matrix, axis=1)
        n = len(adj_matrix)

        modularity = 0
        for c in np.unique(communities):
            nodes_in_c = np.where(communities == c)[0]
            for i in nodes_in_c:
                for j in nodes_in_c:
                    modularity += adj_matrix[i, j] - degrees[i] * degrees[j] / (2 * m)

        modularity /= (2 * m)
        return modularity

    def connected_components(self, adj_matrix: Optional[np.ndarray] = None) -> Dict:
        """
        Find connected components using BFS.

        Args:
            adj_matrix: Adjacency matrix

        Returns:
            Dictionary with component information
        """
        if adj_matrix is None:
            adj_matrix = self.adjacency_matrix

        n = len(adj_matrix)
        visited = np.zeros(n, dtype=bool)
        components = []

        for start in range(n):
            if not visited[start]:
                component = []
                queue = deque([start])
                visited[start] = True

                while queue:
                    node = queue.popleft()
                    component.append(node)

                    for neighbor in range(n):
                        if adj_matrix[node, neighbor] > 0 and not visited[neighbor]:
                            queue.append(neighbor)
                            visited[neighbor] = True

                components.append(component)

        return {
            'n_components': len(components),
            'components': components,
            'largest_component_size': len(max(components, key=len)),
            'component_sizes': [len(c) for c in components]
        }

    def shortest_path(self, adj_matrix: np.ndarray, source: int, target: int) -> Dict:
        """
        Find shortest path between two nodes using Dijkstra's algorithm.

        Args:
            adj_matrix: Adjacency matrix
            source: Source node
            target: Target node

        Returns:
            Dictionary with path information
        """
        n = len(adj_matrix)
        dist = {i: float('inf') for i in range(n)}
        dist[source] = 0
        prev = {i: None for i in range(n)}
        unvisited = set(range(n))

        while unvisited:
            # Find node with minimum distance
            u = min(unvisited, key=lambda x: dist[x])

            if dist[u] == float('inf') or u == target:
                break

            unvisited.remove(u)

            for v in range(n):
                if adj_matrix[u, v] > 0 and v in unvisited:
                    alt = dist[u] + adj_matrix[u, v]
                    if alt < dist[v]:
                        dist[v] = alt
                        prev[v] = u

        # Reconstruct path
        path = []
        if dist[target] != float('inf'):
            current = target
            while current is not None:
                path.append(current)
                current = prev[current]
            path.reverse()

        return {
            'path': path if path else None,
            'distance': dist[target],
            'exists': dist[target] != float('inf')
        }

    def visualize_network(self, adj_matrix: Optional[np.ndarray] = None,
                         communities: Optional[np.ndarray] = None,
                         layout: str = 'spring') -> plt.Figure:
        """Visualize network graph."""
        if adj_matrix is None:
            adj_matrix = self.adjacency_matrix

        n = len(adj_matrix)

        # Generate layout
        if layout == 'spring':
            pos = self._spring_layout(adj_matrix)
        elif layout == 'circular':
            angles = np.linspace(0, 2*np.pi, n, endpoint=False)
            pos = np.column_stack([np.cos(angles), np.sin(angles)])
        else:
            pos = np.random.rand(n, 2)

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        # Plot network
        for i in range(n):
            for j in range(i+1, n):
                if adj_matrix[i, j] > 0:
                    axes[0].plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]],
                               'gray', alpha=0.3, linewidth=0.5)

        # Node colors based on communities
        if communities is not None:
            colors = communities
        else:
            colors = np.zeros(n)

        scatter = axes[0].scatter(pos[:, 0], pos[:, 1], c=colors, s=100,
                                 cmap='tab10', edgecolors='black', linewidths=1)
        axes[0].set_title('Network Graph', fontsize=14, weight='bold')
        axes[0].axis('off')

        # Degree distribution
        degrees = np.sum(adj_matrix > 0, axis=1)
        axes[1].hist(degrees, bins=20, edgecolor='black', alpha=0.7)
        axes[1].set_xlabel('Degree', fontsize=12)
        axes[1].set_ylabel('Frequency', fontsize=12)
        axes[1].set_title('Degree Distribution', fontsize=14, weight='bold')
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        return fig

    def _spring_layout(self, adj_matrix: np.ndarray, iterations: int = 50) -> np.ndarray:
        """Simple spring layout for graph visualization."""
        n = len(adj_matrix)
        pos = np.random.rand(n, 2)

        k = 1.0 / np.sqrt(n)  # Optimal distance

        for _ in range(iterations):
            # Repulsive forces
            delta = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
            distance = np.linalg.norm(delta, axis=2)
            distance[distance == 0] = 0.01

            repulsion = k**2 / distance[:, :, np.newaxis] * delta / distance[:, :, np.newaxis]
            repulsion[np.isnan(repulsion)] = 0

            # Attractive forces
            attraction = np.zeros((n, 2))
            for i in range(n):
                for j in range(n):
                    if adj_matrix[i, j] > 0:
                        d = pos[j] - pos[i]
                        attraction[i] += d * np.linalg.norm(d) / k

            # Update positions
            displacement = np.sum(repulsion, axis=1) + attraction * 0.1
            pos += displacement * 0.1

        return pos


def demo():
    """Demo network analysis toolkit."""
    np.random.seed(42)

    print("Network Analysis Toolkit Demo")
    print("="*60)

    na = NetworkAnalysis()

    # Create a sample network (Karate Club inspired)
    edges = [
        (0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3),
        (0, 4), (4, 5), (4, 6), (5, 6), (6, 7), (7, 8),
        (8, 9), (8, 10), (9, 10), (10, 11), (11, 12),
        (3, 13), (13, 14), (13, 15), (14, 15), (15, 16),
        (16, 17), (16, 18), (17, 18), (18, 19), (19, 20)
    ]

    adj = na.create_adjacency_matrix(edges, n_nodes=21)

    # 1. Degree Centrality
    print("\n1. Degree Centrality")
    print("-" * 60)
    degree_result = na.degree_centrality()
    print(f"Mean degree: {degree_result['mean_degree']:.2f}")
    print(f"Max degree: {degree_result['max_degree']}")
    print(f"Top 5 nodes by degree:")
    top_nodes = np.argsort(degree_result['degree'])[-5:][::-1]
    for node in top_nodes:
        print(f"  Node {node}: degree = {degree_result['degree'][node]:.0f}")

    # 2. PageRank
    print("\n2. PageRank Centrality")
    print("-" * 60)
    pagerank = na.pagerank()
    print(f"Top 5 nodes by PageRank:")
    top_pr = np.argsort(pagerank)[-5:][::-1]
    for node in top_pr:
        print(f"  Node {node}: PageRank = {pagerank[node]:.4f}")

    # 3. Betweenness Centrality
    print("\n3. Betweenness Centrality")
    print("-" * 60)
    betweenness = na.betweenness_centrality()
    print(f"Top 5 nodes by betweenness:")
    top_bet = np.argsort(betweenness)[-5:][::-1]
    for node in top_bet:
        print(f"  Node {node}: betweenness = {betweenness[node]:.4f}")

    # 4. Closeness Centrality
    print("\n4. Closeness Centrality")
    print("-" * 60)
    closeness = na.closeness_centrality()
    print(f"Mean closeness: {np.mean(closeness):.4f}")
    print(f"Top 5 nodes by closeness:")
    top_close = np.argsort(closeness)[-5:][::-1]
    for node in top_close:
        print(f"  Node {node}: closeness = {closeness[node]:.4f}")

    # 5. Clustering Coefficient
    print("\n5. Clustering Coefficient")
    print("-" * 60)
    clustering_result = na.clustering_coefficient()
    print(f"Global clustering coefficient: {clustering_result['global_clustering']:.4f}")
    print(f"Mean local clustering: {clustering_result['mean_clustering']:.4f}")

    # 6. Community Detection
    print("\n6. Community Detection (Louvain)")
    print("-" * 60)
    community_result = na.community_detection_louvain()
    print(f"Number of communities: {community_result['n_communities']}")
    print(f"Modularity: {community_result['modularity']:.4f}")
    print(f"Community sizes: {community_result['community_sizes']}")

    # 7. Connected Components
    print("\n7. Connected Components")
    print("-" * 60)
    components_result = na.connected_components()
    print(f"Number of components: {components_result['n_components']}")
    print(f"Largest component size: {components_result['largest_component_size']}")
    print(f"Component sizes: {components_result['component_sizes']}")

    # 8. Shortest Path
    print("\n8. Shortest Path")
    print("-" * 60)
    path_result = na.shortest_path(adj, source=0, target=20)
    if path_result['exists']:
        print(f"Shortest path from 0 to 20: {path_result['path']}")
        print(f"Path length: {len(path_result['path']) - 1}")
    else:
        print("No path exists")

    # Visualize network
    print("\n9. Network Visualization")
    print("-" * 60)
    fig = na.visualize_network(communities=community_result['communities'])
    fig.savefig('network_analysis_graph.png', dpi=300, bbox_inches='tight')
    print("✓ Saved network_analysis_graph.png")
    plt.close()

    print("\n" + "="*60)
    print("✓ Network Analysis Demo Complete!")
    print("="*60)


if __name__ == '__main__':
    demo()
