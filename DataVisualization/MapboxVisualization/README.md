# Mapbox GL JS Visualization

Professional vector tile maps with 3D buildings, satellite imagery, and smooth animations using Mapbox GL JS.

## ✨ Features

- **3D Buildings**: Realistic building extrusions with height data
- **Multiple Map Styles**: Streets, satellite, dark theme
- **Smooth Animations**: Fly-to transitions between locations
- **Custom Markers**: Interactive popups with city information
- **Vector Tiles**: Fast, scalable map rendering
- **Responsive Design**: Works on all devices

## 🚀 Quick Start

### Get Mapbox Token
1. Sign up at [mapbox.com](https://www.mapbox.com/)
2. Go to [Access Tokens](https://account.mapbox.com/access-tokens/)
3. Copy your token
4. Replace `YOUR_MAPBOX_TOKEN` in `index.html`

### Run Locally
```bash
python -m http.server 8000
# Visit: http://localhost:8000
```

## 🗺️ Map Styles

- **streets-v12**: Default street map
- **satellite-v9**: Satellite imagery
- **dark-v11**: Dark theme for night mode
- **light-v11**: Light minimalist theme
- **outdoors-v12**: Topographic outdoor map

## 📝 Customization

### Change Initial View
```javascript
const map = new mapboxgl.Map({
    center: [lng, lat],  // [longitude, latitude]
    zoom: 12,
    pitch: 45,  // 3D tilt angle
    bearing: 0  // Map rotation
});
```

### Add Custom Marker
```javascript
new mapboxgl.Marker({color: '#FF0000'})
    .setLngLat([lng, lat])
    .setPopup(new mapboxgl.Popup().setHTML('<h3>Title</h3>'))
    .addTo(map);
```

### Add Data Layer
```javascript
map.addLayer({
    'id': 'data-layer',
    'type': 'circle',
    'source': {
        'type': 'geojson',
        'data': yourGeoJSON
    },
    'paint': {
        'circle-radius': 6,
        'circle-color': '#B42222'
    }
});
```

## 🎨 Advanced Features

### Custom 3D Extrusions
```javascript
'fill-extrusion-color': ['get', 'color'],
'fill-extrusion-height': ['*', ['get', 'floors'], 3],
'fill-extrusion-opacity': 0.8
```

### Smooth Animations
```javascript
map.flyTo({
    center: [lng, lat],
    zoom: 15,
    pitch: 60,
    bearing: 30,
    duration: 3000,
    essential: true
});
```

## 📚 Resources

- [Mapbox GL JS Docs](https://docs.mapbox.com/mapbox-gl-js/)
- [Mapbox Examples](https://docs.mapbox.com/mapbox-gl-js/examples/)
- [Mapbox Studio](https://www.mapbox.com/mapbox-studio/)

## 💡 Use Cases

- Real estate visualization
- Logistics and routing
- Geographic data storytelling
- Travel and tourism apps
- Fleet tracking dashboards
- Urban planning visualization

## 👤 Author

**Brill Consulting**
- Email: clientbrill@gmail.com
- LinkedIn: [brillconsulting](https://www.linkedin.com/in/brillconsulting)

---

**Made with Mapbox GL JS | Vector Tile Technology**
