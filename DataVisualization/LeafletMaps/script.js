// Tab switching
document.querySelectorAll('.tab-btn').forEach(button => {
    button.addEventListener('click', () => {
        const tabId = button.getAttribute('data-tab');

        document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
        button.classList.add('active');

        document.querySelectorAll('.map-container').forEach(container => container.classList.remove('active'));
        document.getElementById(tabId + '-container').classList.add('active');

        // Invalidate size to fix map rendering after tab switch
        setTimeout(() => {
            window[tabId.replace(/-/g, '_')].invalidateSize();
        }, 100);
    });
});

// ========== MARKER MAP ==========
const marker_map = L.map('marker-map').setView([40.7128, -74.0060], 11);

L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '© OpenStreetMap contributors',
    maxZoom: 18
}).addTo(marker_map);

// Sample markers
const locations = [
    { coords: [40.7589, -73.9851], title: 'Times Square', desc: 'Major commercial intersection' },
    { coords: [40.7484, -73.9857], title: 'Empire State Building', desc: 'Iconic skyscraper' },
    { coords: [40.7061, -74.0087], title: 'One World Trade', desc: 'Tallest building in Western Hemisphere' },
    { coords: [40.7614, -73.9776], title: 'Central Park', desc: 'Urban park' },
    { coords: [40.7580, -73.9855], title: 'Rockefeller Center', desc: 'Commercial complex' }
];

locations.forEach(loc => {
    const marker = L.marker(loc.coords).addTo(marker_map);
    marker.bindPopup(`<h4>${loc.title}</h4><p>${loc.desc}</p>`);
});

// ========== CLUSTER MAP ==========
const cluster_map = L.map('cluster-map').setView([37.7749, -122.4194], 11);

L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '© OpenStreetMap contributors'
}).addTo(cluster_map);

// Create marker cluster group
const markers = L.markerClusterGroup();

// Generate 100 random markers in San Francisco area
for (let i = 0; i < 100; i++) {
    const lat = 37.7749 + (Math.random() - 0.5) * 0.2;
    const lng = -122.4194 + (Math.random() - 0.5) * 0.2;
    const marker = L.marker([lat, lng]);
    marker.bindPopup(`<h4>Location ${i+1}</h4><p>Lat: ${lat.toFixed(4)}<br>Lng: ${lng.toFixed(4)}</p>`);
    markers.addLayer(marker);
}

cluster_map.addLayer(markers);

// ========== GEOJSON MAP ==========
const geojson_map = L.map('geojson-map').setView([39.8283, -98.5795], 4);

L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '© OpenStreetMap contributors'
}).addTo(geojson_map);

// Sample GeoJSON data (simplified US states)
const geojsonData = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "properties": { "name": "California", "value": 95 },
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[-124, 42], [-120, 42], [-120, 32], [-114, 32], [-114, 35], [-124, 35], [-124, 42]]]
            }
        },
        {
            "type": "Feature",
            "properties": { "name": "Texas", "value": 87 },
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[-106, 32], [-94, 32], [-94, 26], [-106, 26], [-106, 32]]]
            }
        },
        {
            "type": "Feature",
            "properties": { "name": "Florida", "value": 72 },
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[-87, 31], [-80, 31], [-80, 24], [-82, 24], [-87, 29], [-87, 31]]]
            }
        }
    ]
};

// Style function
function style(feature) {
    return {
        fillColor: getColor(feature.properties.value),
        weight: 2,
        opacity: 1,
        color: 'white',
        dashArray: '3',
        fillOpacity: 0.7
    };
}

function getColor(value) {
    return value > 90 ? '#800026' :
           value > 80 ? '#BD0026' :
           value > 70 ? '#E31A1C' :
           value > 60 ? '#FC4E2A' :
                        '#FD8D3C';
}

// Add GeoJSON layer
L.geoJSON(geojsonData, {
    style: style,
    onEachFeature: (feature, layer) => {
        layer.bindPopup(`<h4>${feature.properties.name}</h4><p>Value: ${feature.properties.value}</p>`);
    }
}).addTo(geojson_map);

// ========== HEAT MAP (Circle Markers) ==========
const heatmap = L.map('heatmap').setView([51.505, -0.09], 10);

L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '© OpenStreetMap contributors'
}).addTo(heatmap);

// Generate circle markers with varying sizes and colors
for (let i = 0; i < 50; i++) {
    const lat = 51.505 + (Math.random() - 0.5) * 0.4;
    const lng = -0.09 + (Math.random() - 0.5) * 0.4;
    const value = Math.random() * 100;

    const circle = L.circleMarker([lat, lng], {
        radius: value / 5,
        fillColor: value > 70 ? '#ff0000' : value > 40 ? '#ffaa00' : '#00ff00',
        color: '#fff',
        weight: 1,
        opacity: 1,
        fillOpacity: 0.6
    }).addTo(heatmap);

    circle.bindPopup(`<h4>Data Point</h4><p>Value: ${value.toFixed(2)}</p>`);
}

// ========== ROUTE MAP ==========
const route_map = L.map('route-map').setView([48.8566, 2.3522], 12);

L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '© OpenStreetMap contributors'
}).addTo(route_map);

// Define route coordinates
const routeCoords = [
    [48.8566, 2.3522],  // Paris
    [48.8606, 2.3376],  // Eiffel Tower
    [48.8738, 2.2950],  // Arc de Triomphe
    [48.8529, 2.3499],  // Notre-Dame
    [48.8584, 2.2945]   // Louvre
];

// Draw route
const polyline = L.polyline(routeCoords, {
    color: '#667eea',
    weight: 5,
    opacity: 0.7,
    smoothFactor: 1
}).addTo(route_map);

// Add markers at waypoints
const waypoints = [
    { pos: routeCoords[0], label: 'Start: Paris Center' },
    { pos: routeCoords[1], label: 'Eiffel Tower' },
    { pos: routeCoords[2], label: 'Arc de Triomphe' },
    { pos: routeCoords[3], label: 'Notre-Dame' },
    { pos: routeCoords[4], label: 'End: Louvre Museum' }
];

waypoints.forEach((wp, idx) => {
    const marker = L.marker(wp.pos, {
        icon: L.divIcon({
            className: 'custom-div-icon',
            html: `<div style="background-color: #667eea; width: 30px; height: 30px; border-radius: 50%; border: 3px solid white; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold;">${idx + 1}</div>`,
            iconSize: [30, 30],
            iconAnchor: [15, 15]
        })
    }).addTo(route_map);

    marker.bindPopup(`<h4>${wp.label}</h4><p>Waypoint ${idx + 1}</p>`);
});

// Fit map to route
route_map.fitBounds(polyline.getBounds());
