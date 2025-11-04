"""
Network Visualizations with NetworkX
====================================

Comprehensive network and graph visualization toolkit:
- Graph creation and manipulation
- Multiple layout algorithms
- Community detection visualization
- Centrality and importance metrics
- Interactive and static visualizations

Features:
- Social network analysis
- Organizational charts
- Knowledge graphs
- Dependency networks
- Custom node/edge styling

Technologies: NetworkX, Matplotlib, Plotly
Author: Brill Consulting
"""

import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import plotly.graph_objects as go
from collections import defaultdict


class NetworkVisualizer:
    """Network and graph visualization toolkit."""

    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize network visualizer.

        Args:
            figsize: Default figure size
        """
        self.figsize = figsize

    def create_graph_from_edges(self, edges: List[Tuple], directed: bool = False) -> nx.Graph:
        """
        Create graph from edge list.

        Args:
            edges: List of (source, target) or (source, target, weight) tuples
            directed: Whether graph is directed

        Returns:
            NetworkX graph object
        """
        G = nx.DiGraph() if directed else nx.Graph()

        for edge in edges:
            if len(edge) == 2:
                G.add_edge(edge[0], edge[1])
            elif len(edge) == 3:
                G.add_edge(edge[0], edge[1], weight=edge[2])

        return G

    def visualize_basic_network(self, G: nx.Graph, title: str = "Network Graph",
                                layout: str = 'spring', node_size: int = 500) -> plt.Figure:
        """
        Create basic network visualization.

        Args:
            G: NetworkX graph
            title: Plot title
            layout: Layout algorithm ('spring', 'circular', 'kamada_kawai', 'spectral')
            node_size: Size of nodes

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        # Select layout
        if layout == 'spring':
            pos = nx.spring_layout(G, seed=42)
        elif layout == 'circular':
            pos = nx.circular_layout(G)
        elif layout == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(G)
        elif layout == 'spectral':
            pos = nx.spectral_layout(G)
        else:
            pos = nx.spring_layout(G)

        # Draw network
        nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color='skyblue',
                              edgecolors='black', ax=ax)
        nx.draw_networkx_edges(G, pos, alpha=0.5, ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=10, ax=ax)

        ax.set_title(title, fontsize=16, pad=20)
        ax.axis('off')

        plt.tight_layout()
        return fig

    def visualize_weighted_network(self, G: nx.Graph, title: str = "Weighted Network") -> plt.Figure:
        """
        Visualize network with edge weights.

        Args:
            G: NetworkX graph with weights
            title: Plot title

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        pos = nx.spring_layout(G, seed=42)

        # Get edge weights
        edges = G.edges()
        weights = [G[u][v].get('weight', 1) for u, v in edges]

        # Normalize weights for visualization
        max_weight = max(weights) if weights else 1
        widths = [5 * w / max_weight for w in weights]

        # Draw network
        nx.draw_networkx_nodes(G, pos, node_size=700, node_color='lightcoral',
                              edgecolors='black', ax=ax)
        nx.draw_networkx_edges(G, pos, width=widths, alpha=0.6, ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=10, ax=ax)

        # Draw edge labels
        edge_labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8, ax=ax)

        ax.set_title(title, fontsize=16, pad=20)
        ax.axis('off')

        plt.tight_layout()
        return fig

    def visualize_communities(self, G: nx.Graph, title: str = "Community Detection") -> plt.Figure:
        """
        Visualize network with community detection.

        Args:
            G: NetworkX graph
            title: Plot title

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        # Detect communities using greedy modularity
        communities = nx.community.greedy_modularity_communities(G)

        # Create color map
        color_map = {}
        colors = plt.cm.Set3(np.linspace(0, 1, len(communities)))

        for idx, community in enumerate(communities):
            for node in community:
                color_map[node] = colors[idx]

        node_colors = [color_map[node] for node in G.nodes()]

        # Layout
        pos = nx.spring_layout(G, seed=42)

        # Draw
        nx.draw_networkx_nodes(G, pos, node_size=600, node_color=node_colors,
                              edgecolors='black', ax=ax)
        nx.draw_networkx_edges(G, pos, alpha=0.3, ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=9, ax=ax)

        ax.set_title(f"{title}\nFound {len(communities)} communities", fontsize=16, pad=20)
        ax.axis('off')

        plt.tight_layout()
        return fig

    def visualize_centrality(self, G: nx.Graph, centrality_type: str = 'degree') -> plt.Figure:
        """
        Visualize network with node centrality.

        Args:
            G: NetworkX graph
            centrality_type: Type of centrality ('degree', 'betweenness', 'closeness', 'eigenvector')

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        # Calculate centrality
        if centrality_type == 'degree':
            centrality = nx.degree_centrality(G)
        elif centrality_type == 'betweenness':
            centrality = nx.betweenness_centrality(G)
        elif centrality_type == 'closeness':
            centrality = nx.closeness_centrality(G)
        elif centrality_type == 'eigenvector':
            try:
                centrality = nx.eigenvector_centrality(G, max_iter=1000)
            except:
                centrality = nx.degree_centrality(G)
        else:
            centrality = nx.degree_centrality(G)

        # Node sizes based on centrality
        node_sizes = [3000 * centrality[node] for node in G.nodes()]

        # Node colors based on centrality
        node_colors = [centrality[node] for node in G.nodes()]

        # Layout
        pos = nx.spring_layout(G, seed=42)

        # Draw
        nodes = nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors,
                                      cmap='YlOrRd', edgecolors='black', ax=ax)
        nx.draw_networkx_edges(G, pos, alpha=0.3, ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=9, ax=ax)

        # Colorbar
        plt.colorbar(nodes, ax=ax, label=f'{centrality_type.capitalize()} Centrality')

        ax.set_title(f'{centrality_type.capitalize()} Centrality Analysis', fontsize=16, pad=20)
        ax.axis('off')

        plt.tight_layout()
        return fig

    def create_interactive_network(self, G: nx.Graph, title: str = "Interactive Network") -> go.Figure:
        """
        Create interactive network visualization with Plotly.

        Args:
            G: NetworkX graph
            title: Plot title

        Returns:
            Plotly figure
        """
        # Layout
        pos = nx.spring_layout(G, seed=42)

        # Create edges
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines'
        )

        # Create nodes
        node_x = []
        node_y = []
        node_text = []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(str(node))

        # Node colors by degree
        node_adjacencies = []
        for node in G.nodes():
            node_adjacencies.append(len(list(G.neighbors(node))))

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=node_text,
            textposition="top center",
            marker=dict(
                showscale=True,
                colorscale='YlGnBu',
                size=20,
                color=node_adjacencies,
                colorbar=dict(
                    thickness=15,
                    title='Node Connections',
                    xanchor='left',
                    titleside='right'
                ),
                line=dict(width=2, color='black')
            )
        )

        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           title=title,
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=0, l=0, r=0, t=40),
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                       ))

        return fig

    def analyze_network(self, G: nx.Graph) -> Dict:
        """
        Analyze network properties.

        Args:
            G: NetworkX graph

        Returns:
            Dictionary with network metrics
        """
        metrics = {
            'num_nodes': G.number_of_nodes(),
            'num_edges': G.number_of_edges(),
            'density': nx.density(G),
            'is_connected': nx.is_connected(G) if not nx.is_directed(G) else nx.is_weakly_connected(G),
            'avg_clustering': nx.average_clustering(G),
            'num_triangles': sum(nx.triangles(G).values()) // 3
        }

        # Diameter (only if connected)
        try:
            if metrics['is_connected']:
                metrics['diameter'] = nx.diameter(G)
                metrics['avg_shortest_path'] = nx.average_shortest_path_length(G)
        except:
            pass

        # Degree statistics
        degrees = [d for n, d in G.degree()]
        metrics['avg_degree'] = np.mean(degrees)
        metrics['max_degree'] = max(degrees)
        metrics['min_degree'] = min(degrees)

        return metrics

    def save_plot(self, fig, filename: str, dpi: int = 300):
        """Save plot to file."""
        if isinstance(fig, plt.Figure):
            fig.savefig(filename, dpi=dpi, bbox_inches='tight')
        else:
            fig.write_html(filename)
        print(f"Plot saved to {filename}")


def demo():
    """Demonstrate network visualizations."""
    np.random.seed(42)

    # Create sample social network
    edges = [
        ('Alice', 'Bob'), ('Alice', 'Charlie'), ('Bob', 'David'),
        ('Charlie', 'David'), ('David', 'Eve'), ('Eve', 'Frank'),
        ('Frank', 'Alice'), ('Bob', 'Charlie'), ('Eve', 'Alice'),
        ('Frank', 'David'), ('Charlie', 'Eve'), ('Bob', 'Frank')
    ]

    # Create weighted network
    weighted_edges = [
        ('A', 'B', 5), ('A', 'C', 3), ('B', 'D', 2),
        ('C', 'D', 4), ('D', 'E', 6), ('E', 'F', 3),
        ('F', 'A', 2), ('B', 'C', 7)
    ]

    viz = NetworkVisualizer()

    print("Creating network visualizations...")

    # 1. Basic network
    print("\n1. Basic network...")
    G1 = viz.create_graph_from_edges(edges)
    fig1 = viz.visualize_basic_network(G1, layout='spring')
    viz.save_plot(fig1, 'basic_network.png')
    plt.close()

    # 2. Weighted network
    print("\n2. Weighted network...")
    G2 = viz.create_graph_from_edges(weighted_edges)
    fig2 = viz.visualize_weighted_network(G2)
    viz.save_plot(fig2, 'weighted_network.png')
    plt.close()

    # 3. Community detection
    print("\n3. Community detection...")
    # Create larger network
    G3 = nx.karate_club_graph()
    fig3 = viz.visualize_communities(G3)
    viz.save_plot(fig3, 'community_network.png')
    plt.close()

    # 4. Centrality analysis
    print("\n4. Centrality analysis...")
    fig4 = viz.visualize_centrality(G1, centrality_type='betweenness')
    viz.save_plot(fig4, 'centrality_network.png')
    plt.close()

    # 5. Interactive network
    print("\n5. Interactive network...")
    fig5 = viz.create_interactive_network(G1)
    viz.save_plot(fig5, 'interactive_network.html')

    # 6. Network analysis
    print("\n6. Network analysis...")
    metrics = viz.analyze_network(G1)
    print("\nNetwork Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")

    print("\n✓ All network visualizations created successfully!")
    print("\nGenerated files:")
    print("  - basic_network.png")
    print("  - weighted_network.png")
    print("  - community_network.png")
    print("  - centrality_network.png")
    print("  - interactive_network.html")


if __name__ == '__main__':
    demo()
