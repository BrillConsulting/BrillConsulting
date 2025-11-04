# Time Series Visualizations

Comprehensive time series visualization toolkit for financial and temporal data analysis using **Matplotlib**, **Plotly**, and **Statsmodels**.

## Features

- **Line Plots**: Time series with trends and moving averages
- **Candlestick Charts**: OHLC financial data visualization
- **Volume Analysis**: Price-volume relationship charts
- **Technical Indicators**: SMA, EMA, Bollinger Bands
- **Seasonal Decomposition**: Trend, seasonal, and residual components
- **Autocorrelation**: ACF and PACF plots
- **Returns Analysis**: Distribution and cumulative returns

## Technologies

- **Matplotlib**: Static time series plots
- **Plotly**: Interactive financial charts
- **Statsmodels**: Time series decomposition
- **Pandas**: Time series manipulation
- **SciPy**: Statistical analysis

## Visualization Types

1. **Basic Time Series**: Line plots with moving averages and trends
2. **Candlestick Charts**: Interactive OHLC visualizations
3. **Volume Charts**: Price and volume combined
4. **Technical Indicators**: Multiple indicators on price charts
5. **Seasonal Decomposition**: Component analysis
6. **Returns Analysis**: Statistical returns visualization

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
from timeseries_visualizer import TimeSeriesVisualizer
import pandas as pd

# Initialize visualizer
viz = TimeSeriesVisualizer()

# Create candlestick chart
fig = viz.plot_candlestick(ohlc_data)
viz.save_plot(fig, 'candlestick.html')

# Add technical indicators
fig = viz.plot_with_indicators(data, sma_windows=[20, 50])
viz.save_plot(fig, 'indicators.html')

# Seasonal decomposition
fig = viz.plot_seasonal_decomposition(series, period=365)
viz.save_plot(fig, 'decomposition.png')
```

## Demo

Run demo to generate example charts:

```bash
python timeseries_visualizer.py
```

Creates 6 visualizations with simulated stock data.
