# Sequential Analysis Toolkit

A comprehensive toolkit for analyzing sequential and time series data with advanced statistical methods and forecasting algorithms.

## Description

The Sequential Analysis Toolkit provides specialized methods for analyzing temporal patterns, forecasting future values, and detecting anomalies in sequential data. It implements classical time series models, state space methods, and changepoint detection algorithms.

## Key Features

- **Time Series Decomposition**
  - Trend extraction
  - Seasonal decomposition (additive/multiplicative)
  - Cyclical pattern detection
  - STL decomposition

- **Classical Forecasting Models**
  - ARIMA (AutoRegressive Integrated Moving Average)
  - SARIMA (Seasonal ARIMA)
  - Exponential Smoothing (Simple, Double, Triple)
  - Holt-Winters method

- **State Space Models**
  - Kalman Filter
  - Hidden Markov Models (HMM)
  - Dynamic Linear Models
  - State estimation and prediction

- **Changepoint Detection**
  - CUSUM (Cumulative Sum) method
  - Binary segmentation
  - PELT (Pruned Exact Linear Time)
  - Bayesian changepoint detection

- **Autocorrelation Analysis**
  - ACF (Autocorrelation Function)
  - PACF (Partial Autocorrelation Function)
  - Ljung-Box test
  - Durbin-Watson statistic

- **Anomaly Detection**
  - Statistical control charts
  - Seasonal anomaly detection
  - Trend-based anomaly detection
  - Rolling statistics methods

## Technologies Used

- **Python 3.x**
- **NumPy** - Numerical computing
- **Pandas** - Time series manipulation
- **statsmodels** - Statistical models and time series analysis
- **Matplotlib/Seaborn** - Visualization
- **SciPy** - Statistical functions

## Installation

```bash
# Clone the repository
cd /home/user/BrillConsulting/DataScience/SequentialAnalysis

# Install required packages
pip install numpy pandas statsmodels matplotlib seaborn scipy
```

## Usage Examples

### Time Series Decomposition

```python
from sequential_analysis import SequentialAnalyzer
import numpy as np
import pandas as pd

# Generate sample time series with trend, seasonality, and noise
np.random.seed(42)
time = np.arange(365)
trend = 0.05 * time
seasonal = 10 * np.sin(2 * np.pi * time / 365)
noise = np.random.randn(365) * 2
data = trend + seasonal + 50 + noise

# Create time series
ts = pd.Series(data, index=pd.date_range('2023-01-01', periods=365, freq='D'))

# Initialize analyzer
analyzer = SequentialAnalyzer()

# Decompose time series
decomposition = analyzer.decompose_timeseries(
    ts,
    model='additive',
    period=365
)

print(f"Trend component shape: {decomposition['trend'].shape}")
print(f"Seasonal component shape: {decomposition['seasonal'].shape}")
print(f"Residual component shape: {decomposition['residual'].shape}")

# Visualize decomposition
fig = decomposition['plot']
fig.savefig('decomposition.png', dpi=300, bbox_inches='tight')
```

### ARIMA Forecasting

```python
# Fit ARIMA model
arima_result = analyzer.fit_arima(
    ts,
    order=(2, 1, 2),
    seasonal_order=None
)

print(f"ARIMA parameters:")
print(f"  AR coefficients: {arima_result['ar_params']}")
print(f"  MA coefficients: {arima_result['ma_params']}")
print(f"  AIC: {arima_result['aic']:.2f}")
print(f"  BIC: {arima_result['bic']:.2f}")

# Forecast future values
forecast = analyzer.forecast_arima(
    arima_result['model'],
    steps=30,
    confidence_level=0.95
)

print(f"30-day forecast:")
print(f"  Mean predictions: {forecast['predictions'][:5]}")
print(f"  95% confidence intervals: {forecast['conf_int'][:5]}")

# Visualize forecast
fig = analyzer.plot_forecast(ts, forecast)
fig.savefig('arima_forecast.png', dpi=300, bbox_inches='tight')
```

### Exponential Smoothing

```python
# Apply Holt-Winters exponential smoothing
holt_winters = analyzer.holt_winters(
    ts,
    seasonal_periods=365,
    trend='add',
    seasonal='add'
)

print(f"Holt-Winters model:")
print(f"  Alpha (level): {holt_winters['alpha']:.3f}")
print(f"  Beta (trend): {holt_winters['beta']:.3f}")
print(f"  Gamma (seasonal): {holt_winters['gamma']:.3f}")
print(f"  MSE: {holt_winters['mse']:.3f}")

# Generate forecast
hw_forecast = analyzer.forecast_holt_winters(
    holt_winters['model'],
    steps=30
)

print(f"Holt-Winters 30-day forecast: {hw_forecast['predictions'][:5]}")
```

### Changepoint Detection

```python
# Detect changepoints in time series
changepoints = analyzer.detect_changepoints(
    ts,
    method='pelt',
    penalty='BIC',
    min_size=5
)

print(f"Number of changepoints detected: {len(changepoints['changepoints'])}")
print(f"Changepoint locations: {changepoints['changepoints']}")
print(f"Segment means: {changepoints['segment_means']}")

# Visualize changepoints
fig = analyzer.plot_changepoints(ts, changepoints)
fig.savefig('changepoints.png', dpi=300, bbox_inches='tight')
```

### Autocorrelation Analysis

```python
# Calculate ACF and PACF
acf_result = analyzer.autocorrelation_analysis(
    ts,
    nlags=40
)

print(f"ACF values: {acf_result['acf'][:10]}")
print(f"PACF values: {acf_result['pacf'][:10]}")
print(f"Ljung-Box test p-value: {acf_result['ljung_box_pvalue']:.4f}")
print(f"Durbin-Watson statistic: {acf_result['durbin_watson']:.3f}")

# Visualize ACF and PACF
fig = acf_result['plot']
fig.savefig('acf_pacf.png', dpi=300, bbox_inches='tight')
```

### Kalman Filter State Estimation

```python
# Apply Kalman filter for state estimation
kalman_result = analyzer.kalman_filter(
    ts,
    process_variance=0.01,
    measurement_variance=1.0
)

print(f"Filtered states shape: {kalman_result['filtered_states'].shape}")
print(f"Predicted states shape: {kalman_result['predicted_states'].shape}")
print(f"Kalman gain: {kalman_result['kalman_gain'][:5]}")

# Visualize filtered vs. observed
fig = analyzer.plot_kalman_filter(ts, kalman_result)
fig.savefig('kalman_filter.png', dpi=300, bbox_inches='tight')
```

### Anomaly Detection in Time Series

```python
# Detect anomalies using statistical methods
anomalies = analyzer.detect_anomalies(
    ts,
    method='rolling_stats',
    window=30,
    n_std=3.0
)

print(f"Number of anomalies detected: {len(anomalies['anomaly_indices'])}")
print(f"Anomaly indices: {anomalies['anomaly_indices'][:10]}")
print(f"Anomaly scores: {anomalies['anomaly_scores'][:10]}")

# Visualize anomalies
fig = analyzer.plot_anomalies(ts, anomalies)
fig.savefig('anomalies.png', dpi=300, bbox_inches='tight')
```

## Demo Instructions

Run the comprehensive demo to see all features in action:

```bash
python sequential_analysis.py
```

The demo will:
1. Generate synthetic time series with trend, seasonality, and anomalies
2. Perform time series decomposition
3. Fit ARIMA and exponential smoothing models
4. Generate forecasts with confidence intervals
5. Detect changepoints
6. Perform autocorrelation analysis
7. Apply Kalman filtering
8. Detect anomalies
9. Save all visualizations as PNG files
10. Display comprehensive analysis results

## Output Examples

**Console Output:**
```
Sequential Analysis Toolkit Demo
======================================================================

Generating synthetic time series...
Length: 365 days
Components: Trend + Seasonal + Noise

1. Time Series Decomposition
----------------------------------------------------------------------
Model: Additive
Seasonal period: 365
Trend strength: 0.856
Seasonal strength: 0.723
Residual variance: 4.12

2. ARIMA Model
----------------------------------------------------------------------
Order: (2, 1, 2)
AIC: 2145.67
BIC: 2168.34
Log-likelihood: -1066.84

AR coefficients: [0.245, -0.123]
MA coefficients: [0.456, 0.234]

In-sample fit:
  MSE: 3.456
  RMSE: 1.859
  MAE: 1.423

3. ARIMA Forecast (30 days)
----------------------------------------------------------------------
Mean forecast: [52.3, 52.8, 53.1, 53.4, ...]
95% Confidence interval width: ±3.84

4. Holt-Winters Exponential Smoothing
----------------------------------------------------------------------
Trend: Additive
Seasonal: Additive
Seasonal periods: 365

Parameters:
  Alpha (level): 0.234
  Beta (trend): 0.045
  Gamma (seasonal): 0.156

In-sample MSE: 3.234
Forecast MSE: 4.567

5. Changepoint Detection (PELT)
----------------------------------------------------------------------
Method: PELT (Pruned Exact Linear Time)
Penalty: BIC
Number of changepoints: 3
Locations: [98, 187, 276]

Segment statistics:
  Segment 1 [0-98]: mean=48.5, std=5.2
  Segment 2 [98-187]: mean=52.3, std=4.8
  Segment 3 [187-276]: mean=55.1, std=5.5
  Segment 4 [276-365]: mean=58.7, std=6.1

6. Autocorrelation Analysis
----------------------------------------------------------------------
Lag-1 ACF: 0.867
Lag-1 PACF: 0.867

Ljung-Box test (lag 20):
  Test statistic: 456.78
  P-value: < 0.001
  Result: Significant autocorrelation detected

Durbin-Watson: 0.234 (positive autocorrelation)

7. Kalman Filter
----------------------------------------------------------------------
Process variance: 0.01
Measurement variance: 1.0
Filtered vs. Observed RMSE: 0.856
Smoothing effect: 42.3%

8. Anomaly Detection
----------------------------------------------------------------------
Method: Rolling statistics (3-sigma rule)
Window size: 30 days
Threshold: 3.0 standard deviations

Anomalies detected: 12 (3.3%)
Anomaly dates: [2023-02-15, 2023-04-22, 2023-07-08, ...]
Severity scores: [3.4, 4.1, 3.8, ...]
```

**Generated Visualizations:**
- `decomposition.png` - Trend, seasonal, and residual components
- `arima_forecast.png` - ARIMA predictions with confidence intervals
- `hw_forecast.png` - Holt-Winters forecast
- `changepoints.png` - Time series with detected changepoints
- `acf_pacf.png` - Autocorrelation and partial autocorrelation plots
- `kalman_filter.png` - Filtered vs. observed values
- `anomalies.png` - Time series with anomalies highlighted

## Author

**Brill Consulting**

---

For more information about the algorithms and methodologies, see the inline documentation in `sequential_analysis.py`.
