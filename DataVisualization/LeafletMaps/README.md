# Leaflet Interactive Maps

Professional mapping solutions with Leaflet.js featuring markers, clustering, GeoJSON, heat visualizations, and route mapping.

## 🗺️ Features

### 1. **Marker Map**
- Custom markers with popups
- Interactive click events
- Location details on demand
- Famous NYC landmarks example

### 2. **Cluster Map**
- Marker clustering for large datasets
- Automatic cluster aggregation
- Zoom to expand clusters
- 100+ markers example in San Francisco

### 3. **GeoJSON Map**
- Polygon visualization
- Choropleth coloring
- Interactive feature properties
- US states example with custom styling

### 4. **Heat Map (Circle Markers)**
- Variable-sized circles
- Color-coded by value
- Data intensity visualization
- London area example

### 5. **Route Map**
- Polyline paths
- Numbered waypoints
- Custom route styling
- Paris tourist route example

## 🚀 Quick Start

Open `index.html` in a web browser. No installation required!

```bash
# Optional: Use local server
python -m http.server 8000
```

## 📊 Use Cases

- Store locator maps
- Real estate property maps
- Delivery route planning
- Geographic data visualization
- Tourism and travel guides
- Fleet tracking
- Event location maps
- Census data visualization

## 🛠️ Technologies

- **Leaflet.js v1.9.4**: Leading open-source mapping library
- **Leaflet.markercluster**: Marker clustering plugin
- **OpenStreetMap**: Free tile provider
- **HTML5/CSS3**: Modern web standards
- **Vanilla JavaScript**: No framework dependencies

## 📝 Customization

### Change Base Map
```javascript
// Dark mode tiles
L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png').addTo(map);

// Satellite imagery
L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}').addTo(map);
```

### Add Custom Marker
```javascript
const customIcon = L.icon({
    iconUrl: 'path/to/icon.png',
    iconSize: [32, 32],
    iconAnchor: [16, 32],
    popupAnchor: [0, -32]
});

L.marker([lat, lng], {icon: customIcon}).addTo(map);
```

### GeoJSON Styling
```javascript
L.geoJSON(data, {
    style: feature => ({
        fillColor: getColor(feature.properties.value),
        weight: 2,
        color: 'white',
        fillOpacity: 0.7
    })
}).addTo(map);
```

## 🎨 Custom Popups

```javascript
const popupContent = `
    <div class="custom-popup">
        <h4>${title}</h4>
        <p>${description}</p>
        <img src="${imageUrl}" style="width:100%"/>
        <a href="${linkUrl}">Learn More</a>
    </div>
`;

marker.bindPopup(popupContent);
```

## 🌍 Free Tile Providers

- OpenStreetMap (default)
- CartoDB (light/dark themes)
- Stamen (terrain, toner, watercolor)
- Esri World Imagery (satellite)
- OpenTopoMap (topographic)

## 📚 Leaflet Documentation

- [Leaflet Official Docs](https://leafletjs.com/)
- [Leaflet Plugins](https://leafletjs.com/plugins.html)
- [Leaflet Tutorials](https://leafletjs.com/examples.html)

## 🔌 Popular Plugins

- **Leaflet.markercluster**: Marker clustering (included)
- **Leaflet.heat**: Heatmap visualization
- **Leaflet.draw**: Drawing and editing tools
- **Leaflet.fullscreen**: Fullscreen control
- **Leaflet.control.geocoder**: Address search

## 👤 Author

**Brill Consulting**
- Email: clientbrill@gmail.com
- LinkedIn: [brillconsulting](https://www.linkedin.com/in/brillconsulting)

---

**Made with Leaflet.js | Open-Source Mapping**
