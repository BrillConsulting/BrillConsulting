"""
Time Series Forecasting
Author: BrillConsulting
Description: ARIMA, SARIMA, and Exponential Smoothing for time series prediction
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import argparse


class TimeSeriesAnalyzer:
    """Time series forecasting system"""

    def __init__(self):
        self.models = {}
        self.forecasts = {}

    def arima_forecast(self, data, order=(1, 1, 1), forecast_steps=10):
        """ARIMA forecasting"""
        print(f"🔧 ARIMA{order} Forecasting...")

        model = ARIMA(data, order=order)
        fitted = model.fit()

        forecast = fitted.forecast(steps=forecast_steps)

        self.models['ARIMA'] = fitted
        self.forecasts['ARIMA'] = forecast

        print(f"  AIC: {fitted.aic:.2f}")
        print(f"  BIC: {fitted.bic:.2f}\n")

        return forecast

    def sarima_forecast(self, data, order=(1, 1, 1),
                       seasonal_order=(1, 1, 1, 12), forecast_steps=10):
        """SARIMA forecasting"""
        print(f"🔧 SARIMA{order}x{seasonal_order} Forecasting...")

        model = SARIMAX(data, order=order, seasonal_order=seasonal_order)
        fitted = model.fit(disp=False)

        forecast = fitted.forecast(steps=forecast_steps)

        self.models['SARIMA'] = fitted
        self.forecasts['SARIMA'] = forecast

        print(f"  AIC: {fitted.aic:.2f}\n")

        return forecast

    def exp_smoothing_forecast(self, data, forecast_steps=10):
        """Exponential Smoothing"""
        print("🔧 Exponential Smoothing...")

        model = ExponentialSmoothing(data, seasonal_periods=12, trend='add', seasonal='add')
        fitted = model.fit()

        forecast = fitted.forecast(steps=forecast_steps)

        self.models['Exp Smoothing'] = fitted
        self.forecasts['Exp Smoothing'] = forecast

        return forecast

    def plot_forecasts(self, data, save_path=None):
        """Plot all forecasts"""
        plt.figure(figsize=(14, 6))

        plt.plot(data.index, data.values, label='Actual', linewidth=2)

        for name, forecast in self.forecasts.items():
            forecast_index = pd.date_range(start=data.index[-1],
                                          periods=len(forecast) + 1, freq='D')[1:]
            plt.plot(forecast_index, forecast, label=f'{name} Forecast',
                    linestyle='--', linewidth=2)

        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.title('Time Series Forecasts')
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Forecast plot saved to {save_path}")

        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Time Series Forecasting')
    parser.add_argument('--data', type=str, required=True, help='CSV with date and value')
    parser.add_argument('--steps', type=int, default=10, help='Forecast steps')
    parser.add_argument('--output', type=str, help='Output plot')

    args = parser.parse_args()

    # Load data
    df = pd.read_csv(args.data, parse_dates=[0], index_col=0)
    data = df.iloc[:, 0]

    print(f"📊 Time series: {len(data)} observations\n")

    # Forecast
    analyzer = TimeSeriesAnalyzer()
    analyzer.arima_forecast(data, forecast_steps=args.steps)
    analyzer.exp_smoothing_forecast(data, forecast_steps=args.steps)

    # Plot
    analyzer.plot_forecasts(data, save_path=args.output)


if __name__ == "__main__":
    main()
