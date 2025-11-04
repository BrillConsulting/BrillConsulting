"""
Time Series Visualizations
==========================

Comprehensive time series visualization toolkit for financial and temporal data:
- Line plots with trends and seasonality
- Candlestick and OHLC charts
- Moving averages and indicators
- Seasonal decomposition
- Autocorrelation plots
- Interactive financial dashboards

Features:
- Stock market visualizations
- Financial indicators (SMA, EMA, Bollinger Bands)
- Seasonal pattern analysis
- Forecasting visualization
- Multiple time scales

Technologies: Matplotlib, Plotly, Pandas, Statsmodels
Author: Brill Consulting
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Tuple
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


class TimeSeriesVisualizer:
    """Time series visualization toolkit."""

    def __init__(self, figsize: Tuple[int, int] = (14, 7)):
        """
        Initialize time series visualizer.

        Args:
            figsize: Default figure size
        """
        self.figsize = figsize

    def plot_time_series(self, data: pd.Series, title: str = "Time Series",
                        show_trend: bool = False, ma_windows: Optional[list] = None) -> plt.Figure:
        """
        Create time series line plot with optional moving averages.

        Args:
            data: Time series data with datetime index
            title: Plot title
            show_trend: Whether to show trend line
            ma_windows: List of moving average window sizes

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        # Plot main series
        ax.plot(data.index, data.values, label='Original', linewidth=2, alpha=0.8)

        # Add moving averages
        if ma_windows:
            for window in ma_windows:
                ma = data.rolling(window=window).mean()
                ax.plot(ma.index, ma.values, label=f'{window}-day MA', linewidth=2, alpha=0.7)

        # Add trend line
        if show_trend:
            z = np.polyfit(range(len(data)), data.values, 1)
            p = np.poly1d(z)
            ax.plot(data.index, p(range(len(data))), '--', label='Trend', linewidth=2)

        ax.set_title(title, fontsize=16, pad=20)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        # Format x-axis dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)

        plt.tight_layout()
        return fig

    def plot_candlestick(self, data: pd.DataFrame, title: str = "Candlestick Chart") -> go.Figure:
        """
        Create interactive candlestick chart.

        Args:
            data: DataFrame with columns: Open, High, Low, Close
            title: Chart title

        Returns:
            Plotly figure
        """
        fig = go.Figure(data=[go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='OHLC'
        )])

        fig.update_layout(
            title=title,
            yaxis_title='Price',
            xaxis_title='Date',
            template='plotly_white',
            xaxis_rangeslider_visible=False,
            height=600
        )

        return fig

    def plot_with_volume(self, data: pd.DataFrame, title: str = "Price and Volume") -> go.Figure:
        """
        Create price chart with volume subplot.

        Args:
            data: DataFrame with columns: Open, High, Low, Close, Volume
            title: Chart title

        Returns:
            Plotly figure
        """
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.7, 0.3]
        )

        # Candlestick
        fig.add_trace(go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='OHLC'
        ), row=1, col=1)

        # Volume
        colors = ['red' if close < open else 'green'
                 for close, open in zip(data['Close'], data['Open'])]

        fig.add_trace(go.Bar(
            x=data.index,
            y=data['Volume'],
            name='Volume',
            marker_color=colors
        ), row=2, col=1)

        fig.update_layout(
            title=title,
            yaxis_title='Price',
            yaxis2_title='Volume',
            template='plotly_white',
            xaxis_rangeslider_visible=False,
            height=700
        )

        return fig

    def plot_with_indicators(self, data: pd.DataFrame,
                            sma_windows: list = [20, 50],
                            show_bollinger: bool = True) -> go.Figure:
        """
        Create price chart with technical indicators.

        Args:
            data: DataFrame with Close prices
            sma_windows: Windows for simple moving averages
            show_bollinger: Whether to show Bollinger Bands

        Returns:
            Plotly figure
        """
        fig = go.Figure()

        # Price line
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['Close'],
            mode='lines',
            name='Close Price',
            line=dict(color='blue', width=2)
        ))

        # SMAs
        colors = ['orange', 'red', 'purple']
        for i, window in enumerate(sma_windows):
            sma = data['Close'].rolling(window=window).mean()
            fig.add_trace(go.Scatter(
                x=data.index,
                y=sma,
                mode='lines',
                name=f'SMA-{window}',
                line=dict(color=colors[i % len(colors)], width=2)
            ))

        # Bollinger Bands
        if show_bollinger:
            window = 20
            rolling_mean = data['Close'].rolling(window=window).mean()
            rolling_std = data['Close'].rolling(window=window).std()

            upper_band = rolling_mean + (rolling_std * 2)
            lower_band = rolling_mean - (rolling_std * 2)

            fig.add_trace(go.Scatter(
                x=data.index,
                y=upper_band,
                mode='lines',
                name='Upper BB',
                line=dict(color='gray', width=1, dash='dash')
            ))

            fig.add_trace(go.Scatter(
                x=data.index,
                y=lower_band,
                mode='lines',
                name='Lower BB',
                line=dict(color='gray', width=1, dash='dash'),
                fill='tonexty',
                fillcolor='rgba(128, 128, 128, 0.1)'
            ))

        fig.update_layout(
            title='Price with Technical Indicators',
            yaxis_title='Price',
            xaxis_title='Date',
            template='plotly_white',
            height=600,
            hovermode='x unified'
        )

        return fig

    def plot_seasonal_decomposition(self, data: pd.Series,
                                    period: int = 365,
                                    model: str = 'additive') -> plt.Figure:
        """
        Create seasonal decomposition plot.

        Args:
            data: Time series data
            period: Seasonal period
            model: 'additive' or 'multiplicative'

        Returns:
            Matplotlib figure
        """
        # Perform decomposition
        decomposition = seasonal_decompose(data, model=model, period=period)

        # Create plot
        fig, axes = plt.subplots(4, 1, figsize=(14, 10))

        # Original
        axes[0].plot(data.index, data.values, label='Original')
        axes[0].set_ylabel('Original')
        axes[0].legend(loc='best')
        axes[0].grid(True, alpha=0.3)

        # Trend
        axes[1].plot(decomposition.trend.index, decomposition.trend.values, label='Trend', color='orange')
        axes[1].set_ylabel('Trend')
        axes[1].legend(loc='best')
        axes[1].grid(True, alpha=0.3)

        # Seasonal
        axes[2].plot(decomposition.seasonal.index, decomposition.seasonal.values, label='Seasonal', color='green')
        axes[2].set_ylabel('Seasonal')
        axes[2].legend(loc='best')
        axes[2].grid(True, alpha=0.3)

        # Residual
        axes[3].plot(decomposition.resid.index, decomposition.resid.values, label='Residual', color='red')
        axes[3].set_ylabel('Residual')
        axes[3].set_xlabel('Date')
        axes[3].legend(loc='best')
        axes[3].grid(True, alpha=0.3)

        plt.suptitle(f'Seasonal Decomposition ({model.capitalize()})', fontsize=16, y=0.995)
        plt.tight_layout()

        return fig

    def plot_autocorrelation(self, data: pd.Series, lags: int = 40) -> plt.Figure:
        """
        Create ACF and PACF plots.

        Args:
            data: Time series data
            lags: Number of lags to plot

        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 2, figsize=self.figsize)

        # ACF
        plot_acf(data, lags=lags, ax=axes[0])
        axes[0].set_title('Autocorrelation Function (ACF)')

        # PACF
        plot_pacf(data, lags=lags, ax=axes[1])
        axes[1].set_title('Partial Autocorrelation Function (PACF)')

        plt.tight_layout()
        return fig

    def plot_comparison(self, data_dict: dict, title: str = "Time Series Comparison") -> plt.Figure:
        """
        Compare multiple time series.

        Args:
            data_dict: Dictionary of {label: series}
            title: Plot title

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        for label, series in data_dict.items():
            ax.plot(series.index, series.values, label=label, linewidth=2, alpha=0.7)

        ax.set_title(title, fontsize=16, pad=20)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        plt.xticks(rotation=45)
        plt.tight_layout()

        return fig

    def plot_returns(self, data: pd.Series, title: str = "Returns Distribution") -> plt.Figure:
        """
        Plot returns distribution and statistics.

        Args:
            data: Price series
            title: Plot title

        Returns:
            Matplotlib figure
        """
        # Calculate returns
        returns = data.pct_change().dropna()

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Returns over time
        axes[0, 0].plot(returns.index, returns.values, alpha=0.7)
        axes[0, 0].axhline(0, color='red', linestyle='--', linewidth=1)
        axes[0, 0].set_title('Returns Over Time')
        axes[0, 0].set_ylabel('Returns')
        axes[0, 0].grid(True, alpha=0.3)

        # Returns histogram
        axes[0, 1].hist(returns, bins=50, edgecolor='black', alpha=0.7)
        axes[0, 1].axvline(returns.mean(), color='red', linestyle='--', linewidth=2, label='Mean')
        axes[0, 1].set_title('Returns Distribution')
        axes[0, 1].set_xlabel('Returns')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].legend()

        # Cumulative returns
        cumulative_returns = (1 + returns).cumprod()
        axes[1, 0].plot(cumulative_returns.index, cumulative_returns.values, linewidth=2)
        axes[1, 0].set_title('Cumulative Returns')
        axes[1, 0].set_ylabel('Cumulative Return')
        axes[1, 0].grid(True, alpha=0.3)

        # Q-Q plot
        from scipy import stats
        stats.probplot(returns, dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title('Q-Q Plot')

        # Statistics text
        stats_text = f'Mean: {returns.mean():.4f}\nStd: {returns.std():.4f}\n'
        stats_text += f'Skew: {returns.skew():.4f}\nKurt: {returns.kurtosis():.4f}'
        axes[0, 1].text(0.7, 0.95, stats_text, transform=axes[0, 1].transAxes,
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.suptitle(title, fontsize=16, y=0.995)
        plt.tight_layout()

        return fig

    def save_plot(self, fig, filename: str, dpi: int = 300):
        """Save plot to file."""
        if isinstance(fig, plt.Figure):
            fig.savefig(filename, dpi=dpi, bbox_inches='tight')
        else:
            fig.write_html(filename)
        print(f"Plot saved to {filename}")


def demo():
    """Demonstrate time series visualizations."""
    np.random.seed(42)

    # Generate sample stock data
    dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
    n = len(dates)

    # Simulate price data
    trend = np.linspace(100, 150, n)
    seasonal = 10 * np.sin(np.linspace(0, 4 * np.pi, n))
    noise = np.random.normal(0, 5, n)
    close = trend + seasonal + noise

    # Generate OHLC data
    data = pd.DataFrame({
        'Close': close,
        'Open': close + np.random.normal(0, 2, n),
        'High': close + np.abs(np.random.normal(2, 1, n)),
        'Low': close - np.abs(np.random.normal(2, 1, n)),
        'Volume': np.random.randint(1000000, 10000000, n)
    }, index=dates)

    viz = TimeSeriesVisualizer()

    print("Creating time series visualizations...")

    # 1. Basic time series with MAs
    print("\n1. Time series with moving averages...")
    fig1 = viz.plot_time_series(data['Close'], show_trend=True, ma_windows=[7, 30])
    viz.save_plot(fig1, 'timeseries_ma.png')
    plt.close()

    # 2. Candlestick chart
    print("\n2. Candlestick chart...")
    fig2 = viz.plot_candlestick(data)
    viz.save_plot(fig2, 'candlestick.html')

    # 3. Price with volume
    print("\n3. Price with volume...")
    fig3 = viz.plot_with_volume(data)
    viz.save_plot(fig3, 'price_volume.html')

    # 4. Technical indicators
    print("\n4. Technical indicators...")
    fig4 = viz.plot_with_indicators(data, sma_windows=[20, 50], show_bollinger=True)
    viz.save_plot(fig4, 'technical_indicators.html')

    # 5. Seasonal decomposition
    print("\n5. Seasonal decomposition...")
    fig5 = viz.plot_seasonal_decomposition(data['Close'], period=30)
    viz.save_plot(fig5, 'seasonal_decomposition.png')
    plt.close()

    # 6. Returns analysis
    print("\n6. Returns analysis...")
    fig6 = viz.plot_returns(data['Close'])
    viz.save_plot(fig6, 'returns_analysis.png')
    plt.close()

    print("\n✓ All time series visualizations created successfully!")
    print("\nGenerated files:")
    print("  - timeseries_ma.png")
    print("  - candlestick.html")
    print("  - price_volume.html")
    print("  - technical_indicators.html")
    print("  - seasonal_decomposition.png")
    print("  - returns_analysis.png")


if __name__ == '__main__':
    demo()
