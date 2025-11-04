"""
Geographic Visualizations with Folium and GeoPandas
===================================================

Interactive map visualizations for geographic data analysis:
- Interactive maps with markers, popups, and tooltips
- Choropleth maps for regional data
- Heat maps for density visualization
- Route and path visualization
- Marker clustering for large datasets

Features:
- Multiple base map styles (OpenStreetMap, Stamen, CartoDB)
- Custom marker icons and colors
- Interactive popups with rich content
- GeoJSON and Shapefile support
- Export to HTML for web integration

Technologies: Folium, GeoPandas, Pandas
Author: Brill Consulting
"""

import folium
from folium import plugins
import pandas as pd
import numpy as np
import json
from typing import List, Tuple, Optional, Dict
import geopandas as gpd
from shapely.geometry import Point


class GeoVisualizer:
    """Geographic data visualization toolkit."""

    def __init__(self, default_location: Tuple[float, float] = (40.7128, -74.0060),
                 default_zoom: int = 10):
        """
        Initialize geo visualizer.

        Args:
            default_location: Default center coordinates (lat, lon)
            default_zoom: Default zoom level
        """
        self.default_location = default_location
        self.default_zoom = default_zoom

    def create_base_map(self, location: Optional[Tuple[float, float]] = None,
                       zoom: Optional[int] = None, tiles: str = 'OpenStreetMap') -> folium.Map:
        """
        Create base map.

        Args:
            location: Center coordinates (lat, lon)
            zoom: Zoom level
            tiles: Map tile style

        Returns:
            Folium map object
        """
        loc = location or self.default_location
        z = zoom or self.default_zoom

        m = folium.Map(location=loc, zoom_start=z, tiles=tiles)
        return m

    def add_markers(self, map_obj: folium.Map, locations: pd.DataFrame,
                   lat_col: str = 'latitude', lon_col: str = 'longitude',
                   popup_col: Optional[str] = None, color_col: Optional[str] = None) -> folium.Map:
        """
        Add markers to map.

        Args:
            map_obj: Folium map object
            locations: DataFrame with location data
            lat_col: Latitude column name
            lon_col: Longitude column name
            popup_col: Column for popup text
            color_col: Column for marker color

        Returns:
            Updated map object
        """
        colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred',
                 'lightred', 'beige', 'darkblue', 'darkgreen']

        for idx, row in locations.iterrows():
            popup_text = row[popup_col] if popup_col else f"Location {idx}"
            color = row[color_col] if color_col and color_col in row else 'blue'

            folium.Marker(
                location=[row[lat_col], row[lon_col]],
                popup=folium.Popup(str(popup_text), max_width=300),
                icon=folium.Icon(color=color if color in colors else 'blue')
            ).add_to(map_obj)

        return map_obj

    def create_heatmap(self, locations: pd.DataFrame,
                      lat_col: str = 'latitude', lon_col: str = 'longitude',
                      weight_col: Optional[str] = None) -> folium.Map:
        """
        Create heat map visualization.

        Args:
            locations: DataFrame with location data
            lat_col: Latitude column name
            lon_col: Longitude column name
            weight_col: Column for heat intensity weights

        Returns:
            Map with heatmap layer
        """
        center_lat = locations[lat_col].mean()
        center_lon = locations[lon_col].mean()

        m = self.create_base_map(location=(center_lat, center_lon))

        # Prepare heat data
        if weight_col:
            heat_data = [[row[lat_col], row[lon_col], row[weight_col]]
                        for idx, row in locations.iterrows()]
        else:
            heat_data = [[row[lat_col], row[lon_col]]
                        for idx, row in locations.iterrows()]

        # Add heatmap
        plugins.HeatMap(heat_data, radius=15, blur=25, max_zoom=13).add_to(m)

        return m

    def create_choropleth(self, geo_data: gpd.GeoDataFrame, value_col: str,
                         key_col: str = 'name') -> folium.Map:
        """
        Create choropleth map.

        Args:
            geo_data: GeoDataFrame with geometries and values
            value_col: Column with values to visualize
            key_col: Column with region identifiers

        Returns:
            Map with choropleth layer
        """
        # Calculate center
        bounds = geo_data.total_bounds
        center_lat = (bounds[1] + bounds[3]) / 2
        center_lon = (bounds[0] + bounds[2]) / 2

        m = self.create_base_map(location=(center_lat, center_lon), zoom=6)

        # Add choropleth
        folium.Choropleth(
            geo_data=geo_data,
            data=geo_data,
            columns=[key_col, value_col],
            key_on=f'feature.properties.{key_col}',
            fill_color='YlOrRd',
            fill_opacity=0.7,
            line_opacity=0.2,
            legend_name=value_col
        ).add_to(m)

        return m

    def create_cluster_map(self, locations: pd.DataFrame,
                          lat_col: str = 'latitude', lon_col: str = 'longitude',
                          popup_col: Optional[str] = None) -> folium.Map:
        """
        Create map with marker clustering.

        Args:
            locations: DataFrame with location data
            lat_col: Latitude column name
            lon_col: Longitude column name
            popup_col: Column for popup text

        Returns:
            Map with clustered markers
        """
        center_lat = locations[lat_col].mean()
        center_lon = locations[lon_col].mean()

        m = self.create_base_map(location=(center_lat, center_lon))

        # Create marker cluster
        marker_cluster = plugins.MarkerCluster().add_to(m)

        # Add markers to cluster
        for idx, row in locations.iterrows():
            popup_text = row[popup_col] if popup_col else f"Location {idx}"

            folium.Marker(
                location=[row[lat_col], row[lon_col]],
                popup=str(popup_text)
            ).add_to(marker_cluster)

        return m

    def create_route_map(self, waypoints: List[Tuple[float, float]],
                        route_name: str = "Route") -> folium.Map:
        """
        Create map with route visualization.

        Args:
            waypoints: List of (lat, lon) coordinates
            route_name: Name of the route

        Returns:
            Map with route
        """
        center_lat = np.mean([wp[0] for wp in waypoints])
        center_lon = np.mean([wp[1] for wp in waypoints])

        m = self.create_base_map(location=(center_lat, center_lon))

        # Add route line
        folium.PolyLine(
            waypoints,
            color='blue',
            weight=5,
            opacity=0.7,
            popup=route_name
        ).add_to(m)

        # Add start and end markers
        folium.Marker(
            waypoints[0],
            popup="Start",
            icon=folium.Icon(color='green', icon='play')
        ).add_to(m)

        folium.Marker(
            waypoints[-1],
            popup="End",
            icon=folium.Icon(color='red', icon='stop')
        ).add_to(m)

        return m

    def add_circle_markers(self, map_obj: folium.Map, locations: pd.DataFrame,
                          lat_col: str = 'latitude', lon_col: str = 'longitude',
                          radius_col: Optional[str] = None, color: str = 'blue') -> folium.Map:
        """
        Add circle markers to map.

        Args:
            map_obj: Folium map object
            locations: DataFrame with location data
            lat_col: Latitude column name
            lon_col: Longitude column name
            radius_col: Column for circle radius
            color: Circle color

        Returns:
            Updated map object
        """
        for idx, row in locations.iterrows():
            radius = row[radius_col] if radius_col else 10

            folium.CircleMarker(
                location=[row[lat_col], row[lon_col]],
                radius=radius,
                color=color,
                fill=True,
                fillColor=color,
                fillOpacity=0.6
            ).add_to(map_obj)

        return map_obj

    def save_map(self, map_obj: folium.Map, filename: str):
        """
        Save map to HTML file.

        Args:
            map_obj: Folium map object
            filename: Output filename
        """
        map_obj.save(filename)
        print(f"Map saved to {filename}")


def demo():
    """Demonstrate geographic visualizations."""
    np.random.seed(42)

    # Create sample data - US major cities
    cities_data = pd.DataFrame({
        'city': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix',
                'Philadelphia', 'San Antonio', 'San Diego', 'Dallas', 'San Jose'],
        'latitude': [40.7128, 34.0522, 41.8781, 29.7604, 33.4484,
                    39.9526, 29.4241, 32.7157, 32.7767, 37.3382],
        'longitude': [-74.0060, -118.2437, -87.6298, -95.3698, -112.0740,
                     -75.1652, -98.4936, -117.1611, -96.7970, -121.8863],
        'population': [8336817, 3979576, 2693976, 2320268, 1680992,
                      1584064, 1547253, 1423851, 1343573, 1021795],
        'sales': np.random.randint(500000, 2000000, 10)
    })

    # Initialize visualizer
    viz = GeoVisualizer(default_location=(39.8283, -98.5795), default_zoom=4)

    print("Creating geographic visualizations...")

    # 1. Basic marker map
    print("\n1. Creating marker map...")
    m1 = viz.create_base_map(tiles='OpenStreetMap')
    m1 = viz.add_markers(m1, cities_data, popup_col='city', color_col=None)
    viz.save_map(m1, 'city_markers_map.html')

    # 2. Heatmap
    print("\n2. Creating heatmap...")
    # Generate more points for heatmap
    heat_data = pd.DataFrame({
        'latitude': np.random.normal(39.8283, 5, 1000),
        'longitude': np.random.normal(-98.5795, 10, 1000),
        'intensity': np.random.exponential(10, 1000)
    })
    m2 = viz.create_heatmap(heat_data, weight_col='intensity')
    viz.save_map(m2, 'heatmap.html')

    # 3. Cluster map
    print("\n3. Creating cluster map...")
    m3 = viz.create_cluster_map(cities_data, popup_col='city')
    viz.save_map(m3, 'cluster_map.html')

    # 4. Route map
    print("\n4. Creating route map...")
    route = [
        (40.7128, -74.0060),  # New York
        (39.9526, -75.1652),  # Philadelphia
        (38.9072, -77.0369),  # Washington DC
        (33.7490, -84.3880),  # Atlanta
        (29.7604, -95.3698)   # Houston
    ]
    m4 = viz.create_route_map(route, "East Coast Route")
    viz.save_map(m4, 'route_map.html')

    # 5. Circle markers with size
    print("\n5. Creating circle marker map...")
    m5 = viz.create_base_map()
    cities_data['radius'] = cities_data['population'] / 100000
    m5 = viz.add_circle_markers(m5, cities_data, radius_col='radius', color='red')
    viz.save_map(m5, 'circle_map.html')

    print("\n✓ All geographic visualizations created successfully!")
    print("\nGenerated files:")
    print("  - city_markers_map.html")
    print("  - heatmap.html")
    print("  - cluster_map.html")
    print("  - route_map.html")
    print("  - circle_map.html")
    print("\nOpen these HTML files in a web browser to view interactive maps.")


if __name__ == '__main__':
    demo()
