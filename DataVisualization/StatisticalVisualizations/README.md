# Statistical Visualizations

Comprehensive statistical visualization toolkit using **Matplotlib** and **Seaborn** for publication-ready plots and in-depth data analysis.

## Features

- **Distribution Analysis**: Histograms, KDE plots, box plots, violin plots
- **Correlation Analysis**: Heatmaps, correlation matrices
- **Regression Plots**: Linear regression with confidence intervals, residual plots
- **Categorical Comparisons**: Bar plots, count plots, swarm plots
- **Pairwise Relationships**: Pair plots for multivariate analysis
- **Statistical Testing**: Visual comparison with t-tests, Mann-Whitney U tests
- **Professional Styling**: Publication-ready plots with customizable themes

## Technologies

- **Matplotlib**: Core plotting library
- **Seaborn**: Statistical visualization
- **SciPy**: Statistical tests
- **Pandas**: Data manipulation
- **NumPy**: Numerical operations

## Visualization Types

1. **Distribution Plots**: Analyze single variable distributions with statistics
2. **Comparison Plots**: Compare groups across categories
3. **Correlation Matrix**: Visualize relationships between variables
4. **Regression Analysis**: Linear relationships with confidence intervals
5. **Pairwise Plots**: Comprehensive multivariate exploration
6. **Categorical Analysis**: In-depth categorical data examination
7. **Statistical Tests**: Visual hypothesis testing

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
from statistical_plots import StatisticalVisualizer
import pandas as pd

# Initialize visualizer
viz = StatisticalVisualizer(style='whitegrid', palette='husl')

# Create distribution plot
fig = viz.plot_distribution(data['column_name'])

# Create correlation matrix
fig = viz.plot_correlation_matrix(data)

# Save plot
viz.save_plot(fig, 'output.png', dpi=300)
```

## Demo

Run the demo to see all visualization types:

```bash
python statistical_plots.py
```

Generates 6 example plots demonstrating all features.
