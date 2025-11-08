"""
Bokeh Interactive Visualizations
Server-side interactive plots with Python
"""

from bokeh.plotting import figure, output_file, save, show
from bokeh.layouts import column, row, gridplot
from bokeh.models import HoverTool, ColumnDataSource, Select, CustomJS
from bokeh.palettes import Category20, Viridis256
import numpy as np
import pandas as pd

# ========== 1. INTERACTIVE LINE PLOT ==========
def create_line_plot():
    """Create interactive line plot with hover tools"""
    x = np.linspace(0, 4*np.pi, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)

    p = figure(
        title="Interactive Line Plot with Hover",
        x_axis_label='X',
        y_axis_label='Y',
        width=800,
        height=400,
        tools="pan,wheel_zoom,box_zoom,reset,save"
    )

    # Add lines
    line1 = p.line(x, y1, legend_label="sin(x)", line_width=2, color="#667eea")
    line2 = p.line(x, y2, legend_label="cos(x)", line_width=2, color="#764ba2")

    # Add hover tool
    hover = HoverTool(
        tooltips=[("X", "$x{0.00}"), ("Y", "$y{0.00}")],
        renderers=[line1, line2]
    )
    p.add_tools(hover)

    p.legend.location = "top_right"
    p.legend.click_policy = "hide"

    return p

# ========== 2. SCATTER PLOT WITH SELECTION ==========
def create_scatter_plot():
    """Create scatter plot with selection tools"""
    np.random.seed(42)
    N = 500

    source = ColumnDataSource(data=dict(
        x=np.random.randn(N),
        y=np.random.randn(N),
        colors=np.random.choice(['#667eea', '#764ba2', '#f093fb'], N),
        sizes=np.random.randint(5, 20, N)
    ))

    p = figure(
        title="Interactive Scatter Plot",
        tools="pan,wheel_zoom,box_select,lasso_select,reset",
        width=800,
        height=400
    )

    p.circle(
        'x', 'y',
        source=source,
        size='sizes',
        color='colors',
        alpha=0.6,
        selection_color="red",
        nonselection_alpha=0.2
    )

    hover = HoverTool(tooltips=[("X", "@x{0.00}"), ("Y", "@y{0.00}")])
    p.add_tools(hover)

    return p

# ========== 3. BAR CHART ==========
def create_bar_chart():
    """Create interactive bar chart"""
    categories = ['Q1', 'Q2', 'Q3', 'Q4']
    values = [85, 92, 78, 95]

    p = figure(
        x_range=categories,
        title="Quarterly Performance",
        toolbar_location="above",
        tools="save",
        width=800,
        height=400
    )

    p.vbar(
        x=categories,
        top=values,
        width=0.5,
        color="#667eea",
        alpha=0.8
    )

    p.y_range.start = 0
    p.xgrid.grid_line_color = None
    p.axis.minor_tick_line_color = None

    # Add value labels
    from bokeh.models import Label
    for i, (cat, val) in enumerate(zip(categories, values)):
        label = Label(x=i, y=val, text=str(val), text_align='center', y_offset=5)
        p.add_layout(label)

    return p

# ========== 4. HEATMAP ==========
def create_heatmap():
    """Create interactive heatmap"""
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    hours = [str(i) for i in range(24)]

    # Create data
    x, y = np.meshgrid(range(len(days)), range(len(hours)))
    x = x.flatten()
    y = y.flatten()
    colors = np.random.rand(len(x))

    source = ColumnDataSource(data=dict(
        x=[days[i] for i in x],
        y=[hours[j] for j in y],
        colors=colors,
        values=[f"{c:.2f}" for c in colors]
    ))

    p = figure(
        title="Activity Heatmap",
        x_range=days,
        y_range=list(reversed(hours)),
        toolbar_location="above",
        tools="hover,save",
        tooltips=[('Day', '@x'), ('Hour', '@y'), ('Value', '@values')],
        width=800,
        height=600
    )

    p.rect(
        x='x', y='y',
        width=1, height=1,
        source=source,
        fill_color={'field': 'colors', 'transform': linear_cmap('colors', Viridis256, 0, 1)},
        line_color=None
    )

    return p

# ========== 5. AREA PLOT ==========
def create_area_plot():
    """Create stacked area plot"""
    x = np.linspace(0, 4*np.pi, 100)
    y1 = np.sin(x) + 2
    y2 = np.cos(x) + 2
    y3 = (np.sin(x) + np.cos(x)) / 2 + 2

    p = figure(
        title="Stacked Area Plot",
        x_axis_label='Time',
        y_axis_label='Value',
        width=800,
        height=400,
        tools="pan,wheel_zoom,reset,save"
    )

    p.varea(x=x, y1=0, y2=y1, alpha=0.6, color="#667eea", legend_label="Series 1")
    p.varea(x=x, y1=0, y2=y2, alpha=0.6, color="#764ba2", legend_label="Series 2")
    p.varea(x=x, y1=0, y2=y3, alpha=0.6, color="#f093fb", legend_label="Series 3")

    p.legend.location = "top_left"

    return p

# ========== 6. MULTI-LINE PLOT ==========
def create_multi_line():
    """Create multi-line plot with legend"""
    x = np.linspace(0, 10, 100)

    p = figure(
        title="Multi-Line Time Series",
        x_axis_label='Time',
        y_axis_label='Value',
        width=800,
        height=400,
        tools="pan,wheel_zoom,box_zoom,reset,save"
    )

    for i, color in enumerate(['#667eea', '#764ba2', '#f093fb', '#4facfe', '#43e97b']):
        y = np.sin(x + i) + i * 0.5
        p.line(x, y, legend_label=f"Series {i+1}", line_width=2, color=color, alpha=0.8)

    p.legend.location = "top_left"
    p.legend.click_policy = "hide"

    hover = HoverTool(tooltips=[("X", "$x{0.0}"), ("Y", "$y{0.00}")])
    p.add_tools(hover)

    return p

# ========== 7. BOX PLOT ==========
def create_box_plot():
    """Create box plot"""
    from bokeh.models import Whisker

    categories = ['Group A', 'Group B', 'Group C', 'Group D']
    data = {
        'categories': categories,
        'q1': [20, 25, 30, 22],
        'q2': [30, 35, 40, 32],  # median
        'q3': [40, 45, 50, 42],
        'lower': [10, 15, 20, 12],
        'upper': [50, 55, 60, 52]
    }

    p = figure(
        x_range=categories,
        title="Distribution Box Plot",
        toolbar_location="above",
        tools="save",
        width=800,
        height=400
    )

    # Boxes
    p.vbar(
        x='categories',
        top='q3',
        bottom='q2',
        width=0.4,
        source=data,
        color="#667eea",
        alpha=0.6
    )

    p.vbar(
        x='categories',
        top='q2',
        bottom='q1',
        width=0.4,
        source=data,
        color="#764ba2",
        alpha=0.6
    )

    # Whiskers
    p.segment(x0='categories', y0='upper', x1='categories', y1='q3', source=data, color="black")
    p.segment(x0='categories', y0='lower', x1='categories', y1='q1', source=data, color="black")

    return p

# ========== MAIN FUNCTION ==========
def generate_all_visualizations():
    """Generate all Bokeh visualizations"""
    print("Generating Bokeh interactive visualizations...\n")

    # Create output file
    output_file("bokeh_dashboard.html")

    # Create all plots
    plots = [
        [create_line_plot(), create_scatter_plot()],
        [create_bar_chart(), create_area_plot()],
        [create_multi_line(), create_box_plot()]
    ]

    # Create grid layout
    grid = gridplot(plots, sizing_mode='scale_width')

    # Save to HTML
    save(grid)

    print("✓ All visualizations generated!")
    print("  Output: bokeh_dashboard.html")
    print("  Open in browser to interact with plots")

if __name__ == "__main__":
    from bokeh.transform import linear_cmap
    generate_all_visualizations()
