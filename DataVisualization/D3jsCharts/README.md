# D3.js Interactive Charts

Professional data visualization using D3.js v7 with interactive features, smooth transitions, and responsive design.

## 🎯 Overview

This project demonstrates advanced D3.js visualization techniques including force-directed graphs, animated bar charts, treemaps, sunburst charts, and chord diagrams. All visualizations are fully interactive with custom controls and smooth transitions.

## ✨ Features

### 1. **Force-Directed Network Graph**
- Interactive node dragging
- Physics-based simulation
- Customizable forces (charge, links, collision)
- Toggle labels
- Reset simulation
- Color-coded node groups

### 2. **Animated Bar Chart**
- Smooth transitions
- Sort functionality
- Add/remove bars dynamically
- Randomize data
- Value labels
- Hover effects

### 3. **Hierarchical Treemap**
- Nested rectangles for hierarchical data
- Color-coded categories
- Responsive text sizing
- Hover interactions
- Business metrics visualization

### 4. **Interactive Sunburst Chart**
- Radial hierarchical layout
- Click to zoom functionality
- Smooth arc transitions
- Portfolio/category visualization
- Breadcrumb navigation

### 5. **Chord Diagram**
- Relationship visualization
- Matrix-based data
- Hover to see connections
- Color-coded entities
- Programming language relationships example

## 🚀 Quick Start

### Option 1: Open Directly
Simply open `index.html` in a modern web browser (Chrome, Firefox, Safari, Edge).

### Option 2: Local Server (Recommended)
```bash
# Using Python
python -m http.server 8000

# Using Node.js
npx http-server

# Then visit: http://localhost:8000
```

## 📊 Use Cases

- **Network Analysis**: Social networks, knowledge graphs, dependency visualization
- **Data Comparison**: Sales data, performance metrics, categorical comparisons
- **Hierarchical Data**: Organization structures, file systems, market segments
- **Portfolio Visualization**: Skills, projects, resource allocation
- **Relationship Mapping**: Technology stacks, team connections, data flows

## 🛠️ Technologies

- **D3.js v7**: Core visualization library
- **HTML5**: Structure and semantics
- **CSS3**: Styling and animations
- **Vanilla JavaScript**: Interactivity and controls

## 📝 Customization

### Update Force Graph Data
```javascript
const nodes = [
    {id: 'node1', group: 1},
    {id: 'node2', group: 2},
    // Add more nodes
];

const links = [
    {source: 'node1', target: 'node2', value: 5},
    // Add more links
];
```

### Modify Bar Chart Data
```javascript
let data = [
    {name: 'Category A', value: 75},
    {name: 'Category B', value: 50},
    // Add more items
];
```

### Customize Colors
Update the color schemes in `script.js`:
```javascript
const color = d3.scaleOrdinal(d3.schemeCategory10);
// Or use custom colors:
const color = d3.scaleOrdinal(['#667eea', '#764ba2', '#f093fb']);
```

## 🎨 Styling

All styles are in `styles.css`. Key customization points:

- **Color scheme**: Update gradient colors in header and buttons
- **Dimensions**: Modify SVG width/height in JavaScript
- **Spacing**: Adjust margins and padding
- **Fonts**: Change font-family in body CSS

## 📱 Responsive Design

The visualizations automatically adapt to different screen sizes:
- Flexible SVG containers
- Responsive tab navigation
- Mobile-friendly controls
- Adaptive text sizing

## 🔧 Advanced Features

### Custom Force Simulation
```javascript
const simulation = d3.forceSimulation(nodes)
    .force('link', d3.forceLink(links).distance(100))
    .force('charge', d3.forceManyBody().strength(-300))
    .force('center', d3.forceCenter(width / 2, height / 2));
```

### Animated Transitions
```javascript
bars.transition()
    .duration(750)
    .attr('y', d => y(d.value))
    .attr('height', d => height - y(d.value));
```

### Event Handling
```javascript
cell.on('click', function(event, d) {
    // Custom click handler
});
```

## 📚 Learning Resources

- [D3.js Official Documentation](https://d3js.org/)
- [Observable D3 Gallery](https://observablehq.com/@d3/gallery)
- [D3 Graph Gallery](https://d3-graph-gallery.com/)

## 🎯 Performance Tips

1. **Limit data points**: Keep nodes under 200 for smooth interactions
2. **Debounce resize events**: Prevent excessive redraws
3. **Use CSS transitions**: For simple animations
4. **Optimize force simulation**: Adjust alpha decay for faster settling

## 📄 License

MIT License - Feel free to use in your projects

## 👤 Author

**Brill Consulting**
- LinkedIn: [brillconsulting](https://www.linkedin.com/in/brillconsulting)
- Email: clientbrill@gmail.com

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

---

**Created with D3.js v7 | Professional Data Visualization**
