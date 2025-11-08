# Three.js 3D WebGL Visualization

Advanced 3D data visualization using Three.js with particle systems, animated geometries, and interactive 3D scenes.

## 🎨 Features

### 1. **Particle System**
- 10,000 colored particles
- Smooth rotation animation
- GPU-accelerated rendering
- Vertex colors

### 2. **3D Geometries**
- 5 different geometric shapes
- Phong shading with lighting
- Synchronized rotation
- Material shininess

### 3. **Wave Animation**
- Dynamic sine wave surface
- Wireframe visualization
- Real-time vertex manipulation
- Procedural animation

### 4. **Animated Sphere**
- Morphing icosahedron
- Wireframe overlay
- Vertex displacement
- Pulsating effect

### 5. **Torus Knot**
- Complex mathematical geometry
- Smooth rotation
- Directional lighting
- Professional shading

### 6. **3D Data Visualization**
- 3D bar chart (50 data points)
- Color-coded by value (HSL)
- Orbital camera movement
- Spatial data representation

## 🚀 Quick Start

Simply open `index.html` in a modern web browser. No build process required!

```bash
# Optional: Use local server
python -m http.server 8000
```

## 🎮 Controls

- **Mouse Movement**: Rotate camera view
- **Buttons**: Switch between different visualizations
- **Responsive**: Auto-adjusts to window size

## 🛠️ Technologies

- **Three.js r128**: 3D graphics library
- **WebGL**: GPU-accelerated rendering
- **HTML5 Canvas**: Rendering context
- **Vanilla JavaScript**: No dependencies

## 📊 Use Cases

- Scientific data visualization
- 3D charts and graphs
- Product demonstrations
- Interactive presentations
- Game prototypes
- Particle effects
- Architectural visualization
- Data exploration tools

## 💡 Customization

### Change Particle Count
```javascript
for (let i = 0; i < 5000; i++) {  // Change 10000 to 5000
    vertices.push(
        Math.random() * 200 - 100,
        Math.random() * 200 - 100,
        Math.random() * 200 - 100
    );
}
```

### Modify Colors
```javascript
const material = new THREE.MeshPhongMaterial({
    color: 0xff0000,  // Red color
    shininess: 100
});
```

### Add Your Own Geometry
```javascript
const geometry = new THREE.CylinderGeometry(5, 5, 20, 32);
const material = new THREE.MeshPhongMaterial({ color: 0x00ff00 });
const cylinder = new THREE.Mesh(geometry, material);
scene.add(cylinder);
```

## 🎯 Advanced Features

### Custom Animation Loop
```javascript
currentAnimation = () => {
    mesh.rotation.x += 0.01;
    mesh.rotation.y += 0.02;
    mesh.position.y = Math.sin(Date.now() * 0.001) * 5;
};
```

### Lighting Setup
```javascript
const light = new THREE.DirectionalLight(0xffffff, 1);
light.position.set(10, 10, 10);
scene.add(light);

const ambientLight = new THREE.AmbientLight(0x404040);
scene.add(ambientLight);
```

### Camera Controls
```javascript
camera.position.set(x, y, z);
camera.lookAt(scene.position);
```

## 📚 Resources

- [Three.js Documentation](https://threejs.org/docs/)
- [Three.js Examples](https://threejs.org/examples/)
- [Three.js Fundamentals](https://threejsfundamentals.org/)

## 🔧 Performance Tips

1. **Use BufferGeometry**: More efficient than Geometry
2. **Limit particle count**: Keep under 50,000 for smooth performance
3. **Dispose resources**: Clean up geometries and materials
4. **Use frustum culling**: Don't render off-screen objects
5. **Optimize materials**: Fewer materials = better performance

## 📱 Browser Support

- Chrome ✅
- Firefox ✅
- Safari ✅
- Edge ✅
- Mobile browsers with WebGL support ✅

## 👤 Author

**Brill Consulting**
- Email: clientbrill@gmail.com
- LinkedIn: [brillconsulting](https://www.linkedin.com/in/brillconsulting)

---

**Made with Three.js | WebGL 3D Graphics**
