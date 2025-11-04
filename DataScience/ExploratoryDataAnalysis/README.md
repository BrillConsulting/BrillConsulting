# Exploratory Data Analysis (EDA) Toolkit

Comprehensive toolkit for automated exploratory data analysis, data profiling, and insights discovery.

## Features

- **Automated Data Profiling**: Shape, memory, dtypes, missing values, duplicates
- **Distribution Analysis**: Histograms with skewness and kurtosis
- **Outlier Detection**: IQR and Z-score methods
- **Correlation Analysis**: Heatmaps and top correlations
- **Missing Data Patterns**: Visualization and analysis
- **Categorical Analysis**: Value counts and distributions
- **Summary Statistics**: Extended stats with outliers
- **Full Report Generation**: Automated comprehensive reports

## Technologies

- Pandas, NumPy: Data manipulation
- Matplotlib, Seaborn: Visualization
- SciPy: Statistical analysis

## Usage

```python
from eda_toolkit import EDAToolkit
import pandas as pd

# Load data
data = pd.read_csv('your_data.csv')

# Initialize toolkit
eda = EDAToolkit(data)

# Generate full report
eda.create_full_report(output_dir='./reports')

# Individual analyses
profile = eda.generate_profile()
stats = eda.generate_summary_stats()
outliers = eda.detect_outliers(method='iqr')
```

## Demo

```bash
python eda_toolkit.py
```

Generates complete EDA report with visualizations.
