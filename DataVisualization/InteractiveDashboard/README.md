# Interactive Dashboard

A comprehensive interactive dashboard application built with **Dash** and **Plotly** featuring real-time data visualization, interactive filters, and multi-tab interface.

## Features

- **Multi-Tab Interface**: Overview, Sales Analytics, Customer Analysis, Geographic Distribution
- **Interactive Filters**: Region selection, date range pickers
- **KPI Cards**: Real-time metrics display
- **Multiple Chart Types**: Line charts, bar charts, pie charts, scatter plots, geographic maps
- **Responsive Design**: Clean, professional UI with custom styling
- **Real-Time Updates**: Automatic chart updates based on filter selection

## Technologies

- **Dash**: Web application framework
- **Plotly**: Interactive visualization library
- **Pandas**: Data manipulation
- **NumPy**: Numerical computations

## Visualizations Included

1. **Sales Trend Analysis**: Time series line charts with cumulative sales
2. **Category Distribution**: Interactive pie charts
3. **Regional Performance**: Bar charts and geographic maps
4. **Customer Insights**: Age distribution histograms, scatter plots
5. **Geographic Maps**: US map with sales by location

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
from dashboard_app import InteractiveDashboard

# Create and run dashboard
dashboard = InteractiveDashboard()
dashboard.run(port=8050)
```

Visit `http://localhost:8050` in your browser.

## Dashboard Tabs

1. **Overview**: KPIs and summary visualizations
2. **Sales Analytics**: Detailed sales analysis with filters
3. **Customer Analysis**: Customer demographics and behavior
4. **Geographic Distribution**: Sales by location with interactive maps

## Customization

Modify `generate_sample_data()` to use your own data sources or connect to databases.
