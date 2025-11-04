# Network Visualizations

Comprehensive network and graph visualization toolkit using **NetworkX** for social networks, organizational charts, and relationship analysis.

## Features

- **Multiple Layouts**: Spring, circular, Kamada-Kawai, spectral algorithms
- **Community Detection**: Automatic grouping of related nodes
- **Centrality Analysis**: Degree, betweenness, closeness, eigenvector centrality
- **Weighted Networks**: Edge weight visualization
- **Interactive Plots**: Plotly-based interactive networks
- **Network Metrics**: Comprehensive graph analysis

## Technologies

- **NetworkX**: Graph creation and analysis
- **Matplotlib**: Static visualizations
- **Plotly**: Interactive visualizations
- **NumPy/Pandas**: Data processing

## Visualization Types

1. **Basic Networks**: Simple node-edge graphs with various layouts
2. **Weighted Networks**: Edge thickness representing weights
3. **Community Detection**: Color-coded communities
4. **Centrality Maps**: Node importance visualization
5. **Interactive Networks**: Web-based exploration

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
from network_visualizer import NetworkVisualizer

# Initialize visualizer
viz = NetworkVisualizer()

# Create graph from edges
edges = [('A', 'B'), ('B', 'C'), ('C', 'A')]
G = viz.create_graph_from_edges(edges)

# Visualize
fig = viz.visualize_basic_network(G, layout='spring')
viz.save_plot(fig, 'network.png')

# Analyze
metrics = viz.analyze_network(G)
print(metrics)
```

## Demo

Run demo to generate example visualizations:

```bash
python network_visualizer.py
```

Creates 5 visualizations and prints network metrics.
