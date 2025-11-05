# Time Series Analysis Toolkit

Advanced time series analysis and forecasting with ARIMA, exponential smoothing, and decomposition methods.

## Overview

The Time Series Analysis Toolkit provides comprehensive methods for analyzing temporal data and generating forecasts. It implements ARIMA models, exponential smoothing, trend/seasonality decomposition, and stationarity tests.

## Key Features

- **ARIMA Forecasting**: AutoRegressive Integrated Moving Average models
- **Exponential Smoothing**: Simple exponential smoothing and Holt-Winters
- **Time Series Decomposition**: Separate trend, seasonality, and residuals
- **Stationarity Tests**: Augmented Dickey-Fuller (ADF) test
- **ACF/PACF Analysis**: Autocorrelation and partial autocorrelation functions
- **Ljung-Box Test**: Test for autocorrelation in residuals
- **Seasonal Naive Forecasting**: Baseline forecasts using seasonal patterns
- **Prediction Intervals**: Uncertainty quantification for forecasts
- **Visualization**: Time series plots, ACF/PACF, and decomposition charts

## Technologies Used

- **NumPy**: Numerical computing
- **Pandas**: Time series data manipulation
- **SciPy**: Statistical analysis and signal processing
- **Matplotlib & Seaborn**: Visualization

## Installation

```bash
cd TimeSeriesAnalysis/
pip install numpy pandas scipy matplotlib seaborn
```

## Usage Examples

### Stationarity Testing

```python
from time_series_analysis import TimeSeriesAnalysis

tsa = TimeSeriesAnalysis()

# Test for stationarity
adf_result = tsa.adf_test(series)

print(f"ADF Statistic: {adf_result['adf_statistic']:.4f}")
print(f"P-value: {adf_result['p_value']:.4f}")
print(f"Stationary: {adf_result['stationary']}")
```

### Time Series Decomposition

```python
# Decompose time series
decomp = tsa.decompose(series, period=12, model='additive')

print(f"Trend range: [{np.nanmin(decomp['trend']):.2f}, {np.nanmax(decomp['trend']):.2f}]")
print(f"Seasonal range: [{np.nanmin(decomp['seasonal']):.2f}, {np.nanmax(decomp['seasonal']):.2f}]")
```

### ARIMA Forecasting

```python
# Fit ARIMA and generate forecasts
arima_result = tsa.arima_forecast(
    series,
    p=2,  # AR order
    d=1,  # Differencing order
    q=1,  # MA order
    n_forecast=12
)

print(f"Forecasts: {arima_result['forecasts']}")
print(f"95% CI: [{arima_result['lower_bound']}, {arima_result['upper_bound']}]")
```

### Holt-Winters Smoothing

```python
# Fit Holt-Winters model
hw_result = tsa.holt_winters(
    series,
    period=12,
    alpha=0.3,
    beta=0.1,
    gamma=0.1,
    n_forecast=12
)

print(f"Next 12 forecasts: {hw_result['forecasts']}")
```

### ACF/PACF Analysis

```python
# Analyze autocorrelation
acf_values = tsa.acf(series, nlags=40)
pacf_values = tsa.pacf(series, nlags=40)

print(f"ACF at lag 1: {acf_values[1]:.4f}")
print(f"PACF at lag 1: {pacf_values[1]:.4f}")
```

## Demo

```bash
python time_series_analysis.py
```

The demo includes:
- Augmented Dickey-Fuller stationarity test
- Time series decomposition (trend, seasonal, residual)
- ACF and PACF analysis
- ARIMA forecasting with confidence intervals
- Simple exponential smoothing
- Holt-Winters exponential smoothing
- Ljung-Box test for residuals
- Seasonal naive forecasting
- Comprehensive visualization

## Output Examples

- `time_series_analysis.png`: Time series with forecasts, ACF, PACF, and distribution
- `time_series_decomposition.png`: Decomposed components (original, trend, seasonal, residual)
- Console output with test statistics and forecast values

## Key Concepts

**Stationarity**: Statistical properties constant over time (constant mean, variance)

**Autocorrelation**: Correlation of a series with lagged versions of itself

**Trend**: Long-term increase or decrease in the data

**Seasonality**: Regular, periodic fluctuations

**ARIMA**: Combines autoregression, differencing, and moving averages

## Applications

- Sales and demand forecasting
- Stock price prediction
- Economic indicator forecasting
- Weather prediction
- Energy consumption forecasting
- Web traffic prediction
- Inventory management

## Author

**Brill Consulting**
- Email: clientbrill@gmail.com
- LinkedIn: [brillconsulting](https://www.linkedin.com/in/brillconsulting)
