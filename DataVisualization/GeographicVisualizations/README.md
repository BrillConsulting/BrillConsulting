# Geographic Visualizations

Interactive map visualizations using **Folium** and **GeoPandas** for geographic data analysis and presentation.

## Features

- **Interactive Maps**: Markers, popups, tooltips with multiple base map styles
- **Heatmaps**: Density visualization for point data
- **Choropleth Maps**: Regional data visualization with color coding
- **Marker Clustering**: Efficient display of large datasets
- **Route Visualization**: Path and route mapping
- **Circle Markers**: Size-based data representation
- **HTML Export**: Web-ready interactive maps

## Technologies

- **Folium**: Interactive map visualization
- **GeoPandas**: Geographic data manipulation
- **Pandas**: Data processing
- **Shapely**: Geometric operations

## Map Types

1. **Marker Maps**: Point locations with custom icons and popups
2. **Heatmaps**: Density-based visualization
3. **Choropleth**: Regional value mapping
4. **Cluster Maps**: Automatic grouping of nearby markers
5. **Route Maps**: Path visualization with waypoints

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
from geo_visualizer import GeoVisualizer
import pandas as pd

# Initialize visualizer
viz = GeoVisualizer(default_location=(40.7128, -74.0060))

# Create marker map
m = viz.create_base_map()
m = viz.add_markers(m, locations_df)
viz.save_map(m, 'output_map.html')

# Create heatmap
m = viz.create_heatmap(locations_df, weight_col='intensity')
viz.save_map(m, 'heatmap.html')
```

## Demo

Run demo to generate 5 example maps:

```bash
python geo_visualizer.py
```

Opens in web browser for interactive exploration.
