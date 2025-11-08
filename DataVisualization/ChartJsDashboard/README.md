# Chart.js Dashboard

Professional responsive dashboard with multiple chart types, real-time updates, and smooth animations using Chart.js v4.

## 📊 Features

### Chart Types
1. **Line Chart** - Sales trends with multiple datasets and filled areas
2. **Doughnut Chart** - Category distribution with color-coded segments
3. **Bar Chart** - Quarterly revenue vs costs comparison
4. **Radar Chart** - Multi-dimensional performance metrics
5. **Polar Area Chart** - Regional growth visualization
6. **Pie Chart** - Geographic sales distribution

### Dashboard Features
- 📈 KPI stat cards with trend indicators
- 🔄 Real-time data refresh for each chart
- 🎨 Professional color scheme
- 📱 Fully responsive design
- ✨ Smooth animations and transitions
- 💫 Hover effects and interactivity

## 🚀 Quick Start

Simply open `index.html` in a web browser. No build process required!

```bash
# Optional: Use a local server
python -m http.server 8000
# Visit: http://localhost:8000
```

## 🎯 Use Cases

- Business intelligence dashboards
- Sales analytics
- Performance monitoring
- Financial reporting
- E-commerce analytics
- Project management metrics

## 🛠️ Technologies

- **Chart.js v4**: Professional JavaScript charting library
- **HTML5**: Semantic structure
- **CSS3**: Modern styling with gradients and shadows
- **Vanilla JavaScript**: No framework dependencies

## 📝 Customization

### Update Chart Data
```javascript
lineChart.data.datasets[0].data = [10, 20, 30, 40];
lineChart.update();
```

### Change Colors
```javascript
const colors = {
    primary: '#667eea',
    secondary: '#764ba2',
    // Add your colors
};
```

### Add New Chart
```javascript
const newChart = new Chart(ctx, {
    type: 'line', // or 'bar', 'pie', etc.
    data: { /* your data */ },
    options: { /* your options */ }
});
```

## 📚 Chart.js Documentation

- [Chart.js Official Docs](https://www.chartjs.org/docs/latest/)
- [Chart.js Samples](https://www.chartjs.org/samples/)

## 👤 Author

**Brill Consulting**
- Email: clientbrill@gmail.com
- LinkedIn: [brillconsulting](https://www.linkedin.com/in/brillconsulting)

---

**Made with Chart.js v4 | Professional Data Visualization**
