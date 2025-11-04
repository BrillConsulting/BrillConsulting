"""
Interactive Dashboard with Plotly and Dash
==========================================

A comprehensive interactive dashboard application featuring:
- Multi-tab interface with different chart types
- Real-time data updates
- Interactive filters and controls
- Custom visualizations with Plotly
- Responsive layout design

Features:
- Sales Analytics Dashboard
- Customer Behavior Analysis
- Geographic Distribution Maps
- Time Series Trends
- KPI Cards and Metrics

Technologies: Dash, Plotly, Pandas
Author: Brill Consulting
"""

import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class InteractiveDashboard:
    """Main dashboard application class."""

    def __init__(self):
        """Initialize the dashboard application."""
        self.app = dash.Dash(__name__, suppress_callback_exceptions=True)
        self.setup_layout()
        self.setup_callbacks()
        self.sample_data = self.generate_sample_data()

    def generate_sample_data(self):
        """Generate sample data for demonstration."""
        np.random.seed(42)

        # Generate date range
        dates = pd.date_range(
            start=datetime.now() - timedelta(days=365),
            end=datetime.now(),
            freq='D'
        )

        # Sales data
        sales_data = pd.DataFrame({
            'date': dates,
            'sales': np.random.normal(10000, 2000, len(dates)).cumsum(),
            'customers': np.random.randint(50, 200, len(dates)),
            'region': np.random.choice(['North', 'South', 'East', 'West'], len(dates)),
            'product_category': np.random.choice(['Electronics', 'Clothing', 'Food', 'Books'], len(dates))
        })

        # Customer data
        customer_data = pd.DataFrame({
            'customer_id': range(1, 1001),
            'age': np.random.randint(18, 70, 1000),
            'gender': np.random.choice(['M', 'F'], 1000),
            'lifetime_value': np.random.exponential(500, 1000),
            'purchase_frequency': np.random.poisson(5, 1000)
        })

        # Geographic data
        geo_data = pd.DataFrame({
            'city': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix',
                    'Philadelphia', 'San Antonio', 'San Diego', 'Dallas', 'San Jose'],
            'lat': [40.7128, 34.0522, 41.8781, 29.7604, 33.4484,
                   39.9526, 29.4241, 32.7157, 32.7767, 37.3382],
            'lon': [-74.0060, -118.2437, -87.6298, -95.3698, -112.0740,
                   -75.1652, -98.4936, -117.1611, -96.7970, -121.8863],
            'sales': np.random.randint(100000, 1000000, 10)
        })

        return {
            'sales': sales_data,
            'customers': customer_data,
            'geography': geo_data
        }

    def setup_layout(self):
        """Setup the dashboard layout."""
        self.app.layout = html.Div([
            # Header
            html.Div([
                html.H1('Interactive Business Analytics Dashboard',
                       style={'textAlign': 'center', 'color': '#2c3e50', 'padding': '20px'}),
                html.Hr()
            ]),

            # Tabs
            dcc.Tabs(id='dashboard-tabs', value='overview', children=[
                dcc.Tab(label='Overview', value='overview'),
                dcc.Tab(label='Sales Analytics', value='sales'),
                dcc.Tab(label='Customer Analysis', value='customers'),
                dcc.Tab(label='Geographic Distribution', value='geography')
            ]),

            # Content area
            html.Div(id='tab-content', style={'padding': '20px'})
        ], style={'fontFamily': 'Arial, sans-serif', 'backgroundColor': '#ecf0f1'})

    def create_overview_tab(self):
        """Create overview tab with KPIs and summary charts."""
        data = self.sample_data['sales']

        # Calculate KPIs
        total_sales = data['sales'].iloc[-1]
        total_customers = data['customers'].sum()
        avg_daily_sales = data['sales'].diff().mean()

        return html.Div([
            # KPI Cards
            html.Div([
                self.create_kpi_card('Total Sales', f'${total_sales:,.0f}', 'green'),
                self.create_kpi_card('Total Customers', f'{total_customers:,}', 'blue'),
                self.create_kpi_card('Avg Daily Sales', f'${avg_daily_sales:,.0f}', 'orange')
            ], style={'display': 'flex', 'justifyContent': 'space-around', 'marginBottom': '20px'}),

            # Charts
            html.Div([
                dcc.Graph(id='overview-sales-trend', figure=self.create_sales_trend_chart()),
                dcc.Graph(id='overview-category-pie', figure=self.create_category_pie_chart())
            ])
        ])

    def create_kpi_card(self, title, value, color):
        """Create a KPI card component."""
        return html.Div([
            html.H4(title, style={'color': '#34495e', 'marginBottom': '10px'}),
            html.H2(value, style={'color': color, 'fontWeight': 'bold'})
        ], style={
            'backgroundColor': 'white',
            'padding': '20px',
            'borderRadius': '10px',
            'boxShadow': '0 4px 6px rgba(0,0,0,0.1)',
            'textAlign': 'center',
            'minWidth': '200px'
        })

    def create_sales_trend_chart(self):
        """Create sales trend line chart."""
        data = self.sample_data['sales']

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=data['date'],
            y=data['sales'],
            mode='lines',
            name='Cumulative Sales',
            line=dict(color='#3498db', width=2),
            fill='tozeroy',
            fillcolor='rgba(52, 152, 219, 0.2)'
        ))

        fig.update_layout(
            title='Sales Trend Over Time',
            xaxis_title='Date',
            yaxis_title='Cumulative Sales ($)',
            hovermode='x unified',
            template='plotly_white'
        )

        return fig

    def create_category_pie_chart(self):
        """Create product category distribution pie chart."""
        data = self.sample_data['sales']
        category_sales = data.groupby('product_category')['sales'].count()

        fig = go.Figure(data=[go.Pie(
            labels=category_sales.index,
            values=category_sales.values,
            hole=0.4,
            marker=dict(colors=['#e74c3c', '#3498db', '#2ecc71', '#f39c12'])
        )])

        fig.update_layout(
            title='Sales Distribution by Category',
            template='plotly_white'
        )

        return fig

    def create_sales_tab(self):
        """Create sales analytics tab."""
        return html.Div([
            # Filters
            html.Div([
                html.Label('Select Region:'),
                dcc.Dropdown(
                    id='region-filter',
                    options=[{'label': r, 'value': r} for r in ['All', 'North', 'South', 'East', 'West']],
                    value='All',
                    style={'width': '200px'}
                ),
                html.Label('Date Range:', style={'marginLeft': '20px'}),
                dcc.DatePickerRange(
                    id='date-range',
                    start_date=datetime.now() - timedelta(days=90),
                    end_date=datetime.now()
                )
            ], style={'marginBottom': '20px'}),

            # Charts
            html.Div([
                dcc.Graph(id='sales-by-region'),
                dcc.Graph(id='daily-sales-bar')
            ])
        ])

    def create_customer_tab(self):
        """Create customer analysis tab."""
        data = self.sample_data['customers']

        # Age distribution
        age_hist = px.histogram(
            data, x='age', nbins=20,
            title='Customer Age Distribution',
            labels={'age': 'Age', 'count': 'Number of Customers'},
            color_discrete_sequence=['#9b59b6']
        )

        # Lifetime value vs purchase frequency
        scatter = px.scatter(
            data, x='purchase_frequency', y='lifetime_value',
            color='gender',
            title='Customer Lifetime Value vs Purchase Frequency',
            labels={'purchase_frequency': 'Purchase Frequency',
                   'lifetime_value': 'Lifetime Value ($)'},
            color_discrete_map={'M': '#3498db', 'F': '#e74c3c'}
        )

        return html.Div([
            dcc.Graph(figure=age_hist),
            dcc.Graph(figure=scatter)
        ])

    def create_geography_tab(self):
        """Create geographic distribution tab."""
        data = self.sample_data['geography']

        # Map visualization
        fig = go.Figure(data=go.Scattergeo(
            lon=data['lon'],
            lat=data['lat'],
            text=data['city'],
            mode='markers',
            marker=dict(
                size=data['sales'] / 10000,
                color=data['sales'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Sales ($)")
            )
        ))

        fig.update_layout(
            title='Sales by Geographic Location',
            geo=dict(
                scope='usa',
                showland=True,
                landcolor='rgb(243, 243, 243)',
                coastlinecolor='rgb(204, 204, 204)'
            ),
            template='plotly_white'
        )

        # Bar chart
        bar_fig = px.bar(
            data.sort_values('sales', ascending=False),
            x='city', y='sales',
            title='Top Cities by Sales',
            labels={'city': 'City', 'sales': 'Sales ($)'},
            color='sales',
            color_continuous_scale='Blues'
        )

        return html.Div([
            dcc.Graph(figure=fig),
            dcc.Graph(figure=bar_fig)
        ])

    def setup_callbacks(self):
        """Setup dashboard callbacks for interactivity."""

        @self.app.callback(
            Output('tab-content', 'children'),
            Input('dashboard-tabs', 'value')
        )
        def render_tab_content(tab):
            """Render content based on selected tab."""
            if tab == 'overview':
                return self.create_overview_tab()
            elif tab == 'sales':
                return self.create_sales_tab()
            elif tab == 'customers':
                return self.create_customer_tab()
            elif tab == 'geography':
                return self.create_geography_tab()

        @self.app.callback(
            [Output('sales-by-region', 'figure'),
             Output('daily-sales-bar', 'figure')],
            [Input('region-filter', 'value'),
             Input('date-range', 'start_date'),
             Input('date-range', 'end_date')]
        )
        def update_sales_charts(region, start_date, end_date):
            """Update sales charts based on filters."""
            data = self.sample_data['sales'].copy()

            # Apply filters
            if region != 'All':
                data = data[data['region'] == region]

            if start_date and end_date:
                data = data[(data['date'] >= start_date) & (data['date'] <= end_date)]

            # Region sales chart
            region_sales = data.groupby('region')['sales'].sum().reset_index()
            fig1 = px.bar(
                region_sales, x='region', y='sales',
                title='Sales by Region',
                labels={'region': 'Region', 'sales': 'Total Sales ($)'},
                color='sales',
                color_continuous_scale='Blues'
            )

            # Daily sales chart
            daily_sales = data.groupby(data['date'].dt.date)['customers'].sum().reset_index()
            daily_sales.columns = ['date', 'customers']
            fig2 = px.line(
                daily_sales, x='date', y='customers',
                title='Daily Customer Count',
                labels={'date': 'Date', 'customers': 'Number of Customers'}
            )

            return fig1, fig2

    def run(self, debug=True, port=8050):
        """Run the dashboard application."""
        print(f"Starting Interactive Dashboard on http://localhost:{port}")
        print("Press CTRL+C to stop the server")
        self.app.run_server(debug=debug, port=port)


def main():
    """Main function to run the dashboard."""
    dashboard = InteractiveDashboard()
    dashboard.run()


if __name__ == '__main__':
    main()
