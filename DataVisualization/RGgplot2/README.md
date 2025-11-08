# R Data Visualization with ggplot2

Professional statistical graphics and publication-ready plots using R and ggplot2.

## 📊 Visualizations

### 1. **Scatter Plot with Regression**
- Multiple categories
- Linear regression lines
- Confidence intervals
- Color-coded groups

### 2. **Box Plot**
- Distribution comparison
- Outlier detection
- Mean indicators
- Multiple categories

### 3. **Time Series Line Plot**
- Daily data for full year
- Multiple metrics
- Smooth trends
- Date formatting

### 4. **Bar Chart with Error Bars**
- Quarterly performance
- Standard deviation
- Value labels
- Professional styling

### 5. **Heatmap**
- Correlation matrix
- Gradient coloring
- 10x8 grid
- Clean layout

### 6. **Violin Plot**
- Distribution shapes
- Combined with box plots
- Group comparison
- Density visualization

### 7. **Faceted Plot**
- Small multiples
- Individual trends
- LOESS smoothing
- Free y-axis scales

## 🚀 Quick Start

### Prerequisites
```r
install.packages(c("ggplot2", "dplyr", "tidyr", "scales", "gridExtra"))
```

### Run Visualizations
```r
source("visualizations.R")

# Or run individual plots
scatter_plot()
box_plot()
time_series_plot()
```

## 📁 Output

All plots are saved as high-resolution PNG files (300 DPI):
- `scatter_plot.png`
- `box_plot.png`
- `time_series.png`
- `bar_chart.png`
- `heatmap.png`
- `violin_plot.png`
- `faceted_plot.png`

## 🎨 Features

- **Publication quality**: 300 DPI, perfect for papers
- **Consistent theming**: Professional styling
- **Color schemes**: Modern gradient colors
- **Annotations**: Informative labels and titles
- **Statistical elements**: Regression, confidence intervals, error bars
- **Responsive**: Scales well to different sizes

## 📝 Customization

### Change Colors
```r
scale_color_manual(values = c("#your_color1", "#your_color2"))
```

### Modify Theme
```r
theme_set(theme_bw())  # Or theme_classic(), theme_light()
```

### Adjust Size
```r
ggsave("plot.png", width = 12, height = 8, dpi = 300)
```

## 🎯 Use Cases

- Academic publications
- Research reports
- Business presentations
- Data journalism
- Statistical analysis
- Exploratory data analysis

## 📚 Resources

- [ggplot2 Documentation](https://ggplot2.tidyverse.org/)
- [R Graph Gallery](https://r-graph-gallery.com/)
- [ggplot2 Book](https://ggplot2-book.org/)

## 👤 Author

**Brill Consulting**
- Email: clientbrill@gmail.com
- LinkedIn: [brillconsulting](https://www.linkedin.com/in/brillconsulting)

---

**Made with R & ggplot2 | Grammar of Graphics**
