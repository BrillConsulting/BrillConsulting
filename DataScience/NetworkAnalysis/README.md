# Network Analysis Toolkit

Advanced graph and network analysis methods for centrality metrics, community detection, and network visualization.

## Overview

The Network Analysis Toolkit provides comprehensive methods for analyzing complex networks and graphs. It implements various centrality measures, community detection algorithms, pathfinding, and network visualization.

## Key Features

- **Centrality Metrics**: Degree, betweenness, closeness, and PageRank centrality
- **Community Detection**: Louvain method for finding network communities
- **Clustering Coefficient**: Local and global clustering metrics
- **Connected Components**: Identify disconnected subgraphs
- **Shortest Path**: Dijkstra's algorithm for path finding
- **Network Visualization**: Spring layout and community-colored graphs
- **Degree Distribution**: Analyze network structure

## Technologies Used

- **NumPy**: Numerical computing
- **Pandas**: Data manipulation
- **SciPy**: Sparse matrix operations and clustering
- **Matplotlib & Seaborn**: Visualization

## Installation

```bash
cd NetworkAnalysis/
pip install numpy pandas scipy matplotlib seaborn
```

## Usage Examples

### Create and Analyze Network

```python
from network_analysis import NetworkAnalysis

na = NetworkAnalysis()

# Create network from edge list
edges = [(0, 1), (0, 2), (1, 2), (2, 3), (3, 4)]
adj = na.create_adjacency_matrix(edges, n_nodes=5)
```

### Centrality Analysis

```python
# Degree centrality
degree_result = na.degree_centrality()
print(f"Mean degree: {degree_result['mean_degree']:.2f}")
print(f"Max degree node: {degree_result['degree'].argmax()}")

# PageRank
pagerank = na.pagerank()
top_nodes = np.argsort(pagerank)[-5:][::-1]
print(f"Top 5 nodes by PageRank: {top_nodes}")

# Betweenness centrality
betweenness = na.betweenness_centrality()
print(f"Most central node: {betweenness.argmax()}")
```

### Community Detection

```python
# Detect communities using Louvain
community_result = na.community_detection_louvain()
print(f"Number of communities: {community_result['n_communities']}")
print(f"Modularity: {community_result['modularity']:.4f}")
print(f"Community sizes: {community_result['community_sizes']}")
```

### Network Visualization

```python
# Visualize network with communities
fig = na.visualize_network(communities=community_result['communities'])
fig.savefig('network_graph.png')
```

## Demo

```bash
python network_analysis.py
```

The demo includes:
- Degree centrality analysis
- PageRank computation
- Betweenness centrality
- Closeness centrality
- Clustering coefficient
- Community detection
- Connected components
- Shortest path finding
- Network visualization

## Output Examples

- `network_analysis_graph.png`: Network graph with communities and degree distribution
- Console output with centrality metrics and community statistics

## Author

**Brill Consulting**
- Email: clientbrill@gmail.com
- LinkedIn: [brillconsulting](https://www.linkedin.com/in/brillconsulting)
