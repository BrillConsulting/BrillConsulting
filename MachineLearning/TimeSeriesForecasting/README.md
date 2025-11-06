# 📈 Advanced Time Series Forecasting v2.0

**Production-ready time series forecasting with 10+ models and comprehensive diagnostics**

Comprehensive time series forecasting system featuring baseline, statistical, and advanced models with automatic model selection, stationarity testing, decomposition, and anomaly detection.

## 🌟 Key Features

- **10+ Forecasting Models**: Baseline (Naive, Seasonal Naive), Statistical (ARIMA, SARIMA, Auto-ARIMA), Exponential Smoothing, Prophet
- **Automatic Model Selection**: Compares all models and selects the best performer
- **Stationarity Testing**: Augmented Dickey-Fuller (ADF) test
- **ACF/PACF Analysis**: Visual tools for ARIMA parameter selection
- **Decomposition**: Separate trend, seasonal, and residual components
- **Comprehensive Metrics**: RMSE, MAE, MAPE, SMAPE for robust evaluation
- **Residual Diagnostics**: Ljung-Box test for model validation
- **Anomaly Detection**: IQR and Z-score methods
- **Visualization**: Comparative forecast plots with confidence indicators

## 📦 Installation

```bash
pip install -r requirements.txt

# Optional dependencies for advanced models
pip install prophet pmdarima
```

## 🚀 Quick Start

### Basic Usage

```bash
python time_series.py --data sales.csv --steps 12 --output forecast.png
```

### With Train/Test Split

```bash
python time_series.py \
    --data data.csv \
    --steps 24 \
    --test-size 0.2 \
    --seasonal-period 12 \
    --output results.png
```

### Run Demo

```bash
python time_series.py --demo
```

## 🎛️ Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--data` | Required | Path to CSV file with date index |
| `--steps` | `12` | Number of steps to forecast ahead |
| `--test-size` | `0.2` | Proportion of test set (0-1) |
| `--seasonal-period` | `12` | Seasonal period (12 for monthly, 4 for quarterly) |
| `--output` | - | Path to save comparison plot |
| `--demo` | - | Run demonstration with synthetic data |

## 📊 Forecasting Models

### Baseline Models

#### 1. Naive Forecast
- **Use**: Simple baseline for comparison
- **Method**: Last value repeated for all future steps
- **Pros**: Fast, simple, interpretable
- **Cons**: No trend or seasonality

#### 2. Seasonal Naive
- **Use**: Seasonal patterns with no trend
- **Method**: Last seasonal value for each period
- **Pros**: Captures seasonality, simple
- **Cons**: No trend adaptation

#### 3. Moving Average
- **Use**: Smoothed baseline forecast
- **Method**: Average of last N observations
- **Pros**: Reduces noise, simple
- **Cons**: Lags behind trends

### Statistical Models

#### 4. ARIMA (AutoRegressive Integrated Moving Average) ⭐
- **Use**: General time series forecasting
- **Parameters**: (p, d, q) - AR order, differencing, MA order
- **Pros**: Flexible, handles non-stationary data, well-understood
- **Diagnostics**: AIC, BIC, Ljung-Box test for residuals
- **Best For**: Univariate time series without strong seasonality

#### 5. Auto-ARIMA ⭐ (Optional)
- **Use**: Automatic ARIMA parameter selection
- **Method**: Grid search with AIC/BIC optimization
- **Pros**: Automatic, optimal parameters, saves time
- **Note**: Requires `pip install pmdarima`
- **Best For**: Quick deployment without manual tuning

#### 6. SARIMA (Seasonal ARIMA) ⭐
- **Use**: Time series with seasonality
- **Parameters**: (p,d,q)(P,D,Q,m) - ARIMA + seasonal components
- **Pros**: Handles trend and seasonality, robust
- **Best For**: Monthly/quarterly data with clear seasons

#### 7. Exponential Smoothing (Holt-Winters) ⭐
- **Use**: Trend and seasonal patterns
- **Method**: Weighted exponential smoothing
- **Pros**: Fast, intuitive, handles multiple components
- **Types**: Additive or multiplicative trend/seasonal
- **Best For**: Business forecasting with clear patterns

### Advanced Models

#### 8. Prophet (Facebook) ⭐ (Optional)
- **Use**: Business time series with holidays/events
- **Features**: Automatic changepoint detection, holiday effects, multiple seasonality
- **Pros**: Robust to missing data, handles outliers, interpretable components
- **Note**: Requires `pip install prophet`
- **Best For**: Daily data with special events, holidays

## 📝 Example Code

### Python API

```python
from time_series import TimeSeriesAnalyzer
import pandas as pd

# Load data
df = pd.read_csv('sales_data.csv', parse_dates=['date'], index_col='date')
data = df['sales']

# Train/test split
train_size = int(0.8 * len(data))
train = data.iloc[:train_size]
test = data.iloc[train_size:]
forecast_steps = len(test)

# Initialize analyzer
analyzer = TimeSeriesAnalyzer(random_state=42)

# Check stationarity
stationarity = analyzer.check_stationarity(data)
print(f"Is stationary: {stationarity['is_stationary']}")

# Decompose series
decomposition = analyzer.decompose_series(data, model='additive', period=12)

# Train all models
analyzer.train_all_models(train, test, forecast_steps, seasonal_period=12)

# Compare models
comparison = analyzer.compare_models()
print(comparison)

# Best model
print(f"Best: {analyzer.best_model_name}")
print(f"RMSE: {analyzer.results[analyzer.best_model_name]['rmse']:.4f}")

# Plot results
analyzer.plot_forecasts(train, test, forecast_steps, save_path='forecast.png')
```

### Train Individual Models

```python
# ARIMA
results = analyzer.arima_forecast(train, test, forecast_steps, order=(2, 1, 2))

# Auto-ARIMA (if available)
results = analyzer.auto_arima_forecast(train, test, forecast_steps,
                                      seasonal=True, m=12)

# SARIMA
results = analyzer.sarima_forecast(train, test, forecast_steps,
                                   order=(1, 1, 1),
                                   seasonal_order=(1, 1, 1, 12))

# Exponential Smoothing
results = analyzer.exp_smoothing_forecast(train, test, forecast_steps,
                                         seasonal_periods=12)

# Prophet (if available)
results = analyzer.prophet_forecast(train, test, forecast_steps)
```

### Diagnostics

```python
# ACF/PACF for ARIMA parameter selection
analyzer.plot_acf_pacf(data, lags=40, save_path='acf_pacf.png')

# Anomaly detection
anomalies = analyzer.detect_anomalies(data, method='iqr', threshold=2.0)
print(f"Found {anomalies['n_anomalies']} anomalies")
```

## 📊 Evaluation Metrics

All models are evaluated using:

- **RMSE** (Root Mean Squared Error): Lower is better, sensitive to large errors
- **MAE** (Mean Absolute Error): Lower is better, robust to outliers
- **MAPE** (Mean Absolute Percentage Error): Percentage error, scale-independent
- **SMAPE** (Symmetric MAPE): Better for values near zero

Additional diagnostics:
- **AIC/BIC**: Information criteria for model comparison (lower is better)
- **Ljung-Box Test**: Residual autocorrelation check (p-value > 0.05 desired)

## 🎨 Use Cases

### Business & Sales
- Sales forecasting
- Demand prediction
- Inventory optimization
- Revenue projections

### Finance
- Stock price prediction
- Portfolio risk analysis
- Currency exchange forecasting
- Trading volume prediction

### Operations
- Resource planning
- Capacity forecasting
- Energy consumption prediction
- Supply chain optimization

### Other Domains
- Website traffic forecasting
- Customer behavior prediction
- Weather forecasting
- Healthcare demand planning

## 📈 Sample Output

```
================================================================================
🚀 Training All Time Series Models
================================================================================

🔧 Naive Forecast (Baseline)
  RMSE: 15.2341 | MAE: 12.4567 | MAPE: 8.23% | SMAPE: 7.98%

🔧 Seasonal Naive Forecast
  RMSE: 12.5678 | MAE: 10.1234 | MAPE: 6.45% | SMAPE: 6.21%

🔧 ARIMA(1,1,1) Forecast
  AIC: 1245.67 | BIC: 1256.89
  RMSE: 8.4321 | MAE: 6.7890 | MAPE: 4.12% | SMAPE: 3.98%

🔧 Auto-ARIMA Forecast (automatic parameter selection)
  Selected order: (2, 1, 1)
  Seasonal order: (1, 1, 1, 12)
  AIC: 1238.45
  RMSE: 7.8901 | MAE: 6.3456 | MAPE: 3.89% | SMAPE: 3.76%

🔧 SARIMA(1,1,1)x(1,1,1,12) Forecast
  AIC: 1242.34 | BIC: 1258.90
  RMSE: 8.1234 | MAE: 6.5678 | MAPE: 3.95% | SMAPE: 3.82%

================================================================================
🏆 Best Model: Auto-ARIMA (RMSE: 7.8901)
================================================================================

📊 Model Comparison:
================================================================================
                 model      rmse       mae      mape     smape
         Auto-ARIMA  7.8901   6.3456   3.89%    3.76%
              SARIMA  8.1234   6.5678   3.95%    3.82%
               ARIMA  8.4321   6.7890   4.12%    3.98%
     Seasonal Naive 12.5678  10.1234   6.45%    6.21%
               Naive 15.2341  12.4567   8.23%    7.98%
================================================================================
```

## 🔧 Advanced Features

### Stationarity Testing

```python
# Test for stationarity
result = analyzer.check_stationarity(data, significance_level=0.05)

if not result['is_stationary']:
    # Apply differencing
    data_diff = data.diff().dropna()
    analyzer.check_stationarity(data_diff)
```

### Time Series Decomposition

```python
# Decompose into components
decomposition = analyzer.decompose_series(data, model='additive', period=12)

# Access components
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid
```

### Walk-Forward Validation

```python
# For production deployment, use walk-forward validation
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
results = []

for train_idx, test_idx in tscv.split(data):
    train = data.iloc[train_idx]
    test = data.iloc[test_idx]

    analyzer.arima_forecast(train, test, len(test))
    results.append(analyzer.results['ARIMA']['rmse'])

print(f"Average RMSE: {np.mean(results):.4f} (+/- {np.std(results):.4f})")
```

## 🐛 Troubleshooting

**Non-stationary series warning**:
- Apply differencing: `data.diff().dropna()`
- Use log transformation for exponential trends: `np.log(data)`
- ARIMA will handle with d parameter

**Poor forecast accuracy**:
- Check ACF/PACF plots for better ARIMA parameters
- Try Auto-ARIMA for automatic parameter selection
- Consider Prophet for complex patterns
- Increase seasonal_period if seasonality is longer

**Seasonality not captured**:
- Use SARIMA instead of ARIMA
- Adjust seasonal_period parameter
- Try Exponential Smoothing with seasonal components

**Model won't converge**:
- Simplify ARIMA parameters (lower p, q values)
- Check for outliers and missing values
- Ensure sufficient data (at least 2 seasonal cycles)

## 📚 Theory

### ARIMA Components

**AR (AutoRegressive)**: Uses past values
- `y_t = c + φ₁y_{t-1} + φ₂y_{t-2} + ... + φₚy_{t-p} + ε_t`

**I (Integrated)**: Differencing for stationarity
- First difference: `Δy_t = y_t - y_{t-1}`

**MA (Moving Average)**: Uses past errors
- `y_t = c + ε_t + θ₁ε_{t-1} + θ₂ε_{t-2} + ... + θ_qε_{t-q}`

### Selecting ARIMA Parameters

**ACF Plot**: Shows autocorrelation at different lags
- Helps determine q (MA order)
- Exponential decay suggests AR process

**PACF Plot**: Shows partial autocorrelation
- Helps determine p (AR order)
- Sharp cut-off indicates MA process

**ADF Test**: Tests for stationarity
- p-value < 0.05: Series is stationary (d=0)
- p-value ≥ 0.05: Apply differencing (increase d)

## 📄 License

MIT License - Free for commercial and research use

---

**Author**: BrillConsulting | AI Consultant & Data Scientist
**Contact**: clientbrill@gmail.com
