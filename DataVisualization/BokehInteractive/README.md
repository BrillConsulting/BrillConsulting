# Bokeh Interactive Visualizations

Server-side interactive plots with Python Bokeh for data exploration and dashboards.

## 📊 Visualizations

### 1. **Interactive Line Plot**
- Sine and cosine waves
- Hover tooltips
- Pan and zoom tools
- Legend toggle (click to hide/show)

### 2. **Scatter Plot with Selection**
- 500 random points
- Box and lasso selection
- Color and size encoding
- Selection highlighting

### 3. **Bar Chart**
- Quarterly performance
- Value labels
- Clean minimal design
- Save tool included

### 4. **Heatmap**
- 24x7 activity grid
- Color-coded values
- Hover details
- Viridis color scheme

### 5. **Area Plot**
- Stacked areas
- Multiple series
- Transparency
- Interactive legend

### 6. **Multi-Line Plot**
- 5 time series
- Click legend to hide
- Hover tooltips
- Zoom and pan

### 7. **Box Plot**
- Distribution comparison
- Quartiles visualization
- Whiskers for range
- Multiple groups

## 🚀 Quick Start

### Installation
```bash
pip install bokeh numpy pandas
```

### Run
```python
python bokeh_visualizations.py
```

### Output
Opens `bokeh_dashboard.html` in your browser with interactive plots.

## ✨ Features

- **Interactive**: Pan, zoom, select, hover
- **Server-side**: Python-generated, browser-rendered
- **Responsive**: Adapts to screen size
- **Export**: Save as PNG
- **Customizable**: Full control over styling
- **Fast**: Efficient rendering

## 🎯 Use Cases

- Interactive dashboards
- Data exploration tools
- Web applications
- Jupyter notebooks
- Real-time monitoring
- Scientific visualization

## 📝 Customization

### Add Hover Tool
```python
hover = HoverTool(tooltips=[("X", "@x"), ("Y", "@y")])
p.add_tools(hover)
```

### Change Colors
```python
p.circle(x, y, size=10, color="#667eea", alpha=0.6)
```

### Add Tools
```python
tools = "pan,wheel_zoom,box_select,reset,save"
p = figure(tools=tools)
```

## 📚 Resources

- [Bokeh Documentation](https://docs.bokeh.org/)
- [Bokeh Gallery](https://docs.bokeh.org/en/latest/docs/gallery.html)
- [Bokeh Tutorial](https://mybinder.org/v2/gh/bokeh/bokeh-notebooks/master?filepath=tutorial%2F00%20-%20Introduction%20and%20Setup.ipynb)

## 🔧 Requirements

```
bokeh>=3.0.0
numpy>=1.20.0
pandas>=1.3.0
```

## 👤 Author

**Brill Consulting**
- Email: clientbrill@gmail.com
- LinkedIn: [brillconsulting](https://www.linkedin.com/in/brillconsulting)

---

**Made with Bokeh | Python Interactive Visualization**
