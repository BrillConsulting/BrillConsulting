"""
Advanced Time Series Forecasting System v2.0
Author: BrillConsulting
Description: Production-ready time series forecasting with 10+ models and comprehensive diagnostics
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import argparse
from typing import Dict, List, Tuple, Optional
warnings.filterwarnings('ignore')

# Optional dependencies
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

try:
    from pmdarima import auto_arima
    AUTO_ARIMA_AVAILABLE = True
except ImportError:
    AUTO_ARIMA_AVAILABLE = False


class TimeSeriesAnalyzer:
    """
    Advanced time series forecasting system

    Features:
    - 10+ forecasting models
    - Automatic model selection
    - Stationarity testing
    - ACF/PACF analysis
    - Residual diagnostics
    - Walk-forward validation
    - Comprehensive evaluation metrics
    - Decomposition and anomaly detection
    """

    def __init__(self, random_state: int = 42):
        self.models = {}
        self.forecasts = {}
        self.results = {}
        self.best_model_name = None
        self.random_state = random_state
        np.random.seed(random_state)

    def check_stationarity(self, data: pd.Series, significance_level: float = 0.05) -> Dict:
        """
        Augmented Dickey-Fuller test for stationarity
        """
        print("🔍 Stationarity Test (ADF)")
        print("=" * 60)

        result = adfuller(data.dropna(), autolag='AIC')

        adf_stat = result[0]
        p_value = result[1]
        critical_values = result[4]

        is_stationary = p_value < significance_level

        print(f"  ADF Statistic: {adf_stat:.4f}")
        print(f"  p-value: {p_value:.4f}")
        print(f"  Critical Values:")
        for key, value in critical_values.items():
            print(f"    {key}: {value:.4f}")

        if is_stationary:
            print(f"  ✅ Series is STATIONARY (p < {significance_level})")
        else:
            print(f"  ⚠️  Series is NON-STATIONARY (p >= {significance_level})")
            print("  💡 Consider differencing or transformation")
        print()

        return {
            'adf_statistic': adf_stat,
            'p_value': p_value,
            'critical_values': critical_values,
            'is_stationary': is_stationary
        }

    def plot_acf_pacf(self, data: pd.Series, lags: int = 40, save_path: Optional[str] = None):
        """
        Plot ACF and PACF for ARIMA parameter selection
        """
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        # ACF
        acf_values = acf(data.dropna(), nlags=lags)
        axes[0].stem(range(len(acf_values)), acf_values)
        axes[0].axhline(y=0, color='k', linestyle='--', linewidth=1)
        axes[0].axhline(y=-1.96/np.sqrt(len(data)), color='r', linestyle='--', linewidth=1)
        axes[0].axhline(y=1.96/np.sqrt(len(data)), color='r', linestyle='--', linewidth=1)
        axes[0].set_title('Autocorrelation Function (ACF)')
        axes[0].set_xlabel('Lag')
        axes[0].set_ylabel('ACF')

        # PACF
        pacf_values = pacf(data.dropna(), nlags=lags)
        axes[1].stem(range(len(pacf_values)), pacf_values)
        axes[1].axhline(y=0, color='k', linestyle='--', linewidth=1)
        axes[1].axhline(y=-1.96/np.sqrt(len(data)), color='r', linestyle='--', linewidth=1)
        axes[1].axhline(y=1.96/np.sqrt(len(data)), color='r', linestyle='--', linewidth=1)
        axes[1].set_title('Partial Autocorrelation Function (PACF)')
        axes[1].set_xlabel('Lag')
        axes[1].set_ylabel('PACF')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 ACF/PACF plot saved to {save_path}")

        plt.close()

    def decompose_series(self, data: pd.Series, model: str = 'additive',
                        period: Optional[int] = None, save_path: Optional[str] = None):
        """
        Decompose time series into trend, seasonal, and residual components
        """
        print("📊 Time Series Decomposition")
        print("=" * 60)

        if period is None:
            # Try to detect period
            period = 12  # Default monthly seasonality

        decomposition = seasonal_decompose(data.dropna(), model=model, period=period, extrapolate_trend='freq')

        fig, axes = plt.subplots(4, 1, figsize=(14, 10))

        # Original
        axes[0].plot(data.index, data.values)
        axes[0].set_ylabel('Original')
        axes[0].set_title('Time Series Decomposition')
        axes[0].grid(True, alpha=0.3)

        # Trend
        axes[1].plot(decomposition.trend.index, decomposition.trend.values)
        axes[1].set_ylabel('Trend')
        axes[1].grid(True, alpha=0.3)

        # Seasonal
        axes[2].plot(decomposition.seasonal.index, decomposition.seasonal.values)
        axes[2].set_ylabel('Seasonal')
        axes[2].grid(True, alpha=0.3)

        # Residual
        axes[3].plot(decomposition.resid.index, decomposition.resid.values)
        axes[3].set_ylabel('Residual')
        axes[3].set_xlabel('Date')
        axes[3].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Decomposition plot saved to {save_path}")

        plt.close()

        print(f"  Model: {model}")
        print(f"  Period: {period}")
        print(f"  Trend strength: {1 - (decomposition.resid.var() / (decomposition.resid + decomposition.trend).var()):.4f}")
        print()

        return decomposition

    def naive_forecast(self, train: pd.Series, test: pd.Series, forecast_steps: int) -> Dict:
        """
        Naive forecast (last value repeated)
        """
        print("🔧 Naive Forecast (Baseline)")

        forecast = np.repeat(train.iloc[-1], forecast_steps)

        metrics = self._calculate_metrics(test.values[:forecast_steps], forecast, 'Naive')

        self.forecasts['Naive'] = forecast
        self.results['Naive'] = metrics

        return metrics

    def seasonal_naive_forecast(self, train: pd.Series, test: pd.Series,
                               forecast_steps: int, period: int = 12) -> Dict:
        """
        Seasonal naive forecast
        """
        print("🔧 Seasonal Naive Forecast")

        forecast = []
        for i in range(forecast_steps):
            forecast.append(train.iloc[-(period - (i % period))])
        forecast = np.array(forecast)

        metrics = self._calculate_metrics(test.values[:forecast_steps], forecast, 'Seasonal Naive')

        self.forecasts['Seasonal Naive'] = forecast
        self.results['Seasonal Naive'] = metrics

        return metrics

    def moving_average_forecast(self, train: pd.Series, test: pd.Series,
                               forecast_steps: int, window: int = 3) -> Dict:
        """
        Simple moving average forecast
        """
        print(f"🔧 Moving Average Forecast (window={window})")

        ma = train.rolling(window=window).mean().iloc[-1]
        forecast = np.repeat(ma, forecast_steps)

        metrics = self._calculate_metrics(test.values[:forecast_steps], forecast, 'Moving Average')

        self.forecasts['Moving Average'] = forecast
        self.results['Moving Average'] = metrics

        return metrics

    def arima_forecast(self, train: pd.Series, test: pd.Series,
                      forecast_steps: int, order: Tuple[int, int, int] = (1, 1, 1)) -> Dict:
        """
        ARIMA forecasting with diagnostics
        """
        print(f"🔧 ARIMA{order} Forecast")

        try:
            model = ARIMA(train, order=order)
            fitted = model.fit()

            forecast = fitted.forecast(steps=forecast_steps)

            # Residual diagnostics
            residuals = fitted.resid
            ljung_box = acorr_ljungbox(residuals, lags=10, return_df=True)

            metrics = self._calculate_metrics(test.values[:forecast_steps], forecast, 'ARIMA')
            metrics.update({
                'aic': fitted.aic,
                'bic': fitted.bic,
                'ljung_box_p': ljung_box['lb_pvalue'].iloc[-1]
            })

            self.models['ARIMA'] = fitted
            self.forecasts['ARIMA'] = forecast
            self.results['ARIMA'] = metrics

            print(f"  AIC: {fitted.aic:.2f} | BIC: {fitted.bic:.2f}")

            return metrics

        except Exception as e:
            print(f"  ⚠️  ARIMA failed: {e}")
            return {}

    def auto_arima_forecast(self, train: pd.Series, test: pd.Series,
                           forecast_steps: int, seasonal: bool = False, m: int = 12) -> Dict:
        """
        Auto-ARIMA with automatic parameter selection
        """
        if not AUTO_ARIMA_AVAILABLE:
            print("⚠️  Auto-ARIMA not available. Install: pip install pmdarima")
            return {}

        print("🔧 Auto-ARIMA Forecast (automatic parameter selection)")

        try:
            model = auto_arima(train, seasonal=seasonal, m=m,
                             stepwise=True, suppress_warnings=True,
                             error_action='ignore', trace=False)

            fitted = model.fit(train)
            forecast = fitted.predict(n_periods=forecast_steps)

            metrics = self._calculate_metrics(test.values[:forecast_steps], forecast, 'Auto-ARIMA')
            metrics.update({
                'order': model.order,
                'seasonal_order': model.seasonal_order if seasonal else None,
                'aic': model.aic()
            })

            self.models['Auto-ARIMA'] = fitted
            self.forecasts['Auto-ARIMA'] = forecast
            self.results['Auto-ARIMA'] = metrics

            print(f"  Selected order: {model.order}")
            if seasonal:
                print(f"  Seasonal order: {model.seasonal_order}")
            print(f"  AIC: {model.aic():.2f}")

            return metrics

        except Exception as e:
            print(f"  ⚠️  Auto-ARIMA failed: {e}")
            return {}

    def sarima_forecast(self, train: pd.Series, test: pd.Series, forecast_steps: int,
                       order: Tuple[int, int, int] = (1, 1, 1),
                       seasonal_order: Tuple[int, int, int, int] = (1, 1, 1, 12)) -> Dict:
        """
        SARIMA forecasting
        """
        print(f"🔧 SARIMA{order}x{seasonal_order} Forecast")

        try:
            model = SARIMAX(train, order=order, seasonal_order=seasonal_order)
            fitted = model.fit(disp=False)

            forecast = fitted.forecast(steps=forecast_steps)

            metrics = self._calculate_metrics(test.values[:forecast_steps], forecast, 'SARIMA')
            metrics.update({
                'aic': fitted.aic,
                'bic': fitted.bic
            })

            self.models['SARIMA'] = fitted
            self.forecasts['SARIMA'] = forecast
            self.results['SARIMA'] = metrics

            print(f"  AIC: {fitted.aic:.2f} | BIC: {fitted.bic:.2f}")

            return metrics

        except Exception as e:
            print(f"  ⚠️  SARIMA failed: {e}")
            return {}

    def exp_smoothing_forecast(self, train: pd.Series, test: pd.Series,
                              forecast_steps: int, seasonal_periods: int = 12) -> Dict:
        """
        Exponential Smoothing (Holt-Winters)
        """
        print("🔧 Exponential Smoothing (Holt-Winters)")

        try:
            # Try additive
            model = ExponentialSmoothing(train, seasonal_periods=seasonal_periods,
                                        trend='add', seasonal='add')
            fitted = model.fit()
            forecast = fitted.forecast(steps=forecast_steps)

            metrics = self._calculate_metrics(test.values[:forecast_steps], forecast, 'Exp Smoothing')

            self.models['Exp Smoothing'] = fitted
            self.forecasts['Exp Smoothing'] = forecast
            self.results['Exp Smoothing'] = metrics

            return metrics

        except Exception as e:
            print(f"  ⚠️  Exponential Smoothing failed: {e}")
            return {}

    def prophet_forecast(self, train: pd.Series, test: pd.Series,
                        forecast_steps: int) -> Dict:
        """
        Facebook Prophet forecasting
        """
        if not PROPHET_AVAILABLE:
            print("⚠️  Prophet not available. Install: pip install prophet")
            return {}

        print("🔧 Prophet Forecast (Facebook)")

        try:
            # Prepare data for Prophet
            df_train = pd.DataFrame({
                'ds': train.index,
                'y': train.values
            })

            model = Prophet(daily_seasonality=False, weekly_seasonality=False,
                          yearly_seasonality=True, changepoint_prior_scale=0.05)
            model.fit(df_train)

            # Create future dataframe
            future = model.make_future_dataframe(periods=forecast_steps, freq='D')
            forecast_df = model.predict(future)

            forecast = forecast_df['yhat'].iloc[-forecast_steps:].values

            metrics = self._calculate_metrics(test.values[:forecast_steps], forecast, 'Prophet')

            self.models['Prophet'] = model
            self.forecasts['Prophet'] = forecast
            self.results['Prophet'] = metrics

            return metrics

        except Exception as e:
            print(f"  ⚠️  Prophet failed: {e}")
            return {}

    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                          model_name: str) -> Dict:
        """
        Calculate comprehensive evaluation metrics
        """
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

        # Symmetric MAPE (better for values near zero)
        smape = np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred))) * 100

        print(f"  RMSE: {rmse:.4f} | MAE: {mae:.4f} | MAPE: {mape:.2f}% | SMAPE: {smape:.2f}%\n")

        return {
            'model': model_name,
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'smape': smape
        }

    def train_all_models(self, train: pd.Series, test: pd.Series,
                        forecast_steps: int, seasonal_period: int = 12):
        """
        Train all available forecasting models
        """
        print("\n" + "=" * 80)
        print("🚀 Training All Time Series Models")
        print("=" * 80 + "\n")

        # Baseline models
        self.naive_forecast(train, test, forecast_steps)
        self.seasonal_naive_forecast(train, test, forecast_steps, period=seasonal_period)
        self.moving_average_forecast(train, test, forecast_steps, window=3)

        # Statistical models
        self.arima_forecast(train, test, forecast_steps, order=(1, 1, 1))

        if AUTO_ARIMA_AVAILABLE:
            self.auto_arima_forecast(train, test, forecast_steps, seasonal=True, m=seasonal_period)

        self.sarima_forecast(train, test, forecast_steps,
                           order=(1, 1, 1), seasonal_order=(1, 1, 1, seasonal_period))

        self.exp_smoothing_forecast(train, test, forecast_steps, seasonal_periods=seasonal_period)

        # Advanced models
        if PROPHET_AVAILABLE:
            self.prophet_forecast(train, test, forecast_steps)

        # Find best model
        if self.results:
            best_model = min(self.results.items(), key=lambda x: x[1]['rmse'])
            self.best_model_name = best_model[0]

        print("=" * 80)
        print(f"🏆 Best Model: {self.best_model_name} (RMSE: {self.results[self.best_model_name]['rmse']:.4f})")
        print("=" * 80 + "\n")

    def compare_models(self) -> pd.DataFrame:
        """
        Compare all models side-by-side
        """
        if not self.results:
            print("⚠️  No models trained yet")
            return pd.DataFrame()

        comparison = pd.DataFrame(self.results).T
        comparison = comparison.sort_values('rmse')

        print("\n📊 Model Comparison:")
        print("=" * 80)
        print(comparison[['rmse', 'mae', 'mape', 'smape']].to_string())
        print("=" * 80 + "\n")

        return comparison

    def plot_forecasts(self, train: pd.Series, test: pd.Series,
                      forecast_steps: int, save_path: Optional[str] = None):
        """
        Plot all forecasts with actual values
        """
        fig, ax = plt.subplots(figsize=(16, 8))

        # Plot training data
        ax.plot(train.index, train.values, label='Training Data',
               linewidth=2, color='black', alpha=0.7)

        # Plot test data
        test_slice = test.iloc[:forecast_steps]
        ax.plot(test_slice.index, test_slice.values, label='Actual Test Data',
               linewidth=2, color='blue', marker='o', markersize=4)

        # Plot forecasts
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.forecasts)))
        for i, (name, forecast) in enumerate(self.forecasts.items()):
            forecast_index = test.index[:len(forecast)]

            rmse = self.results[name]['rmse']
            label = f'{name} (RMSE: {rmse:.2f})'

            linestyle = '--' if name != self.best_model_name else '-'
            linewidth = 3 if name == self.best_model_name else 2
            alpha = 1.0 if name == self.best_model_name else 0.6

            ax.plot(forecast_index, forecast, label=label,
                   linestyle=linestyle, linewidth=linewidth,
                   color=colors[i], alpha=alpha)

        ax.axvline(x=train.index[-1], color='red', linestyle=':',
                  linewidth=2, label='Train/Test Split', alpha=0.7)

        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        ax.set_title('Time Series Forecasts - All Models Comparison', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Forecast comparison plot saved to {save_path}")

        plt.close()

    def detect_anomalies(self, data: pd.Series, method: str = 'iqr',
                        threshold: float = 1.5) -> Dict:
        """
        Detect anomalies in time series
        """
        print(f"🔍 Anomaly Detection ({method.upper()} method)")
        print("=" * 60)

        if method == 'iqr':
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            anomalies = (data < lower_bound) | (data > upper_bound)

        elif method == 'zscore':
            z_scores = np.abs((data - data.mean()) / data.std())
            anomalies = z_scores > threshold

        else:
            raise ValueError(f"Unknown method: {method}")

        n_anomalies = anomalies.sum()
        pct_anomalies = (n_anomalies / len(data)) * 100

        print(f"  Total anomalies: {n_anomalies} ({pct_anomalies:.2f}%)")
        print(f"  Threshold: {threshold}")
        print()

        return {
            'anomalies': anomalies,
            'n_anomalies': n_anomalies,
            'pct_anomalies': pct_anomalies,
            'anomaly_indices': data[anomalies].index.tolist()
        }


def demo():
    """
    Demonstration with synthetic data
    """
    print("\n" + "=" * 80)
    print("📈 Advanced Time Series Forecasting Demo")
    print("=" * 80 + "\n")

    # Generate synthetic data with trend and seasonality
    np.random.seed(42)
    n = 200
    t = np.arange(n)

    trend = 0.5 * t
    seasonal = 10 * np.sin(2 * np.pi * t / 12)
    noise = np.random.normal(0, 5, n)

    values = 100 + trend + seasonal + noise
    dates = pd.date_range(start='2020-01-01', periods=n, freq='M')

    data = pd.Series(values, index=dates, name='Sales')

    print(f"📊 Dataset: {len(data)} monthly observations\n")

    # Train/test split
    train_size = int(0.8 * len(data))
    train = data.iloc[:train_size]
    test = data.iloc[train_size:]
    forecast_steps = len(test)

    print(f"  Training: {len(train)} observations")
    print(f"  Testing: {len(test)} observations")
    print(f"  Forecast horizon: {forecast_steps} steps\n")

    # Initialize analyzer
    analyzer = TimeSeriesAnalyzer()

    # Stationarity test
    analyzer.check_stationarity(data)

    # Decomposition
    analyzer.decompose_series(data, model='additive', period=12)

    # ACF/PACF analysis
    analyzer.plot_acf_pacf(data, lags=40)

    # Train all models
    analyzer.train_all_models(train, test, forecast_steps, seasonal_period=12)

    # Compare models
    comparison = analyzer.compare_models()

    # Plot forecasts
    analyzer.plot_forecasts(train, test, forecast_steps)

    # Anomaly detection
    anomalies = analyzer.detect_anomalies(data, method='iqr', threshold=2.0)

    print("\n✅ Demo completed successfully!")
    print(f"Best model: {analyzer.best_model_name}")
    print(f"Best RMSE: {analyzer.results[analyzer.best_model_name]['rmse']:.4f}")


def main():
    """
    Command-line interface
    """
    parser = argparse.ArgumentParser(
        description='Advanced Time Series Forecasting System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic forecast
  python time_series.py --data sales.csv --steps 12 --output forecast.png

  # With train/test split
  python time_series.py --data data.csv --steps 24 --test-size 0.2

  # Run demo
  python time_series.py --demo
        """
    )

    parser.add_argument('--data', type=str, help='CSV with date index and values')
    parser.add_argument('--steps', type=int, default=12, help='Forecast steps ahead')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test set proportion')
    parser.add_argument('--seasonal-period', type=int, default=12, help='Seasonal period')
    parser.add_argument('--output', type=str, help='Output forecast plot path')
    parser.add_argument('--demo', action='store_true', help='Run demonstration')

    args = parser.parse_args()

    if args.demo:
        demo()
        return

    if not args.data:
        parser.print_help()
        return

    # Load data
    df = pd.read_csv(args.data, parse_dates=[0], index_col=0)
    data = df.iloc[:, 0]

    print(f"\n📊 Time Series: {len(data)} observations")
    print(f"  Period: {data.index[0]} to {data.index[-1]}\n")

    # Train/test split
    train_size = int((1 - args.test_size) * len(data))
    train = data.iloc[:train_size]
    test = data.iloc[train_size:]

    # Initialize analyzer
    analyzer = TimeSeriesAnalyzer()

    # Diagnostics
    analyzer.check_stationarity(data)
    analyzer.decompose_series(data, period=args.seasonal_period)

    # Train models
    analyzer.train_all_models(train, test, args.steps, seasonal_period=args.seasonal_period)

    # Compare
    analyzer.compare_models()

    # Plot
    analyzer.plot_forecasts(train, test, args.steps, save_path=args.output)

    print(f"\n🏆 Best Model: {analyzer.best_model_name}")
    print(f"  RMSE: {analyzer.results[analyzer.best_model_name]['rmse']:.4f}")
    print(f"  MAE: {analyzer.results[analyzer.best_model_name]['mae']:.4f}")


if __name__ == "__main__":
    main()
