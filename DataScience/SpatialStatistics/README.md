# Spatial Statistics Toolkit

Advanced geospatial and spatial data analysis with autocorrelation, kriging, and hotspot detection.

## Overview

The Spatial Statistics Toolkit provides comprehensive methods for analyzing spatial data and identifying geographic patterns. It implements spatial autocorrelation tests, kriging interpolation, variogram modeling, and hotspot analysis.

## Key Features

- **Moran's I**: Test for spatial autocorrelation
- **Geary's C**: Alternative spatial autocorrelation measure
- **Empirical Variogram**: Model spatial correlation structure
- **Variogram Fitting**: Spherical, exponential, and Gaussian models
- **Ordinary Kriging**: Spatial interpolation with uncertainty
- **Hotspot Analysis**: Getis-Ord Gi* statistic for identifying clusters
- **Spatial Weights**: Inverse distance, binary, and k-nearest neighbors
- **Visualization**: Spatial data maps and variogram plots

## Technologies Used

- **NumPy**: Numerical computing
- **Pandas**: Data manipulation
- **SciPy**: Statistical analysis and optimization
- **Matplotlib & Seaborn**: Visualization

## Installation

```bash
cd SpatialStatistics/
pip install numpy pandas scipy matplotlib seaborn
```

## Usage Examples

### Spatial Autocorrelation

```python
from spatial_statistics import SpatialStatistics

ss = SpatialStatistics()
ss.load_spatial_data(coordinates, values)

# Moran's I test
morans_result = ss.morans_i()
print(f"Moran's I: {morans_result['morans_i']:.4f}")
print(f"P-value: {morans_result['p_value']:.4e}")
print(f"Interpretation: {morans_result['interpretation']}")

# Geary's C test
gearys_result = ss.gearys_c()
print(f"Geary's C: {gearys_result['gearys_c']:.4f}")
print(f"Interpretation: {gearys_result['interpretation']}")
```

### Variogram Analysis and Kriging

```python
# Compute empirical variogram
variogram = ss.empirical_variogram(n_bins=10)

# Fit theoretical model
model_params = ss.fit_variogram_model(model_type='spherical')
print(f"Nugget: {model_params['nugget']:.4f}")
print(f"Sill: {model_params['sill']:.4f}")
print(f"Range: {model_params['range']:.4f}")

# Perform kriging interpolation
kriging_result = ss.ordinary_kriging(
    coords=known_points,
    pred_coords=prediction_grid,
    values=known_values,
    variogram_params=model_params
)
print(f"Mean prediction: {np.mean(kriging_result['predictions']):.2f}")
print(f"Mean std error: {np.mean(kriging_result['std_errors']):.2f}")
```

### Hotspot Detection

```python
# Identify spatial hotspots
hotspot_result = ss.hotspot_analysis(coords, values, method='getis_ord')
print(f"Number of hotspots: {np.sum(hotspot_result['hotspots'])}")
print(f"Number of coldspots: {np.sum(hotspot_result['coldspots'])}")
```

## Demo

```bash
python spatial_statistics.py
```

The demo includes:
- Moran's I spatial autocorrelation test
- Geary's C test
- Empirical variogram calculation
- Variogram model fitting
- Ordinary kriging interpolation
- Hotspot analysis (Getis-Ord Gi*)
- Spatial data visualization

## Output Examples

- `spatial_statistics_data.png`: Spatial data map with value distribution
- Console output with autocorrelation tests and kriging results

## Author

**Brill Consulting**
- Email: clientbrill@gmail.com
- LinkedIn: [brillconsulting](https://www.linkedin.com/in/brillconsulting)
