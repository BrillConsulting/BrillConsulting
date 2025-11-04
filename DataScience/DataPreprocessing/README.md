# Data Preprocessing Toolkit

Comprehensive toolkit for data cleaning, validation, and quality assurance.

## Features

- **Missing Value Handling**: Mean, median, KNN imputation, forward/backward fill
- **Outlier Detection & Treatment**: IQR, Z-score methods with capping or removal
- **Duplicate Removal**: Smart duplicate detection
- **Type Conversion**: Automatic dtype optimization
- **Text Cleaning**: Normalization, special character removal
- **Name Standardization**: Consistent column naming
- **Data Validation**: Range checks and business rules
- **Quality Reporting**: Comprehensive data quality metrics

## Technologies

- Pandas, NumPy: Data manipulation
- Scikit-learn: Advanced imputation
- SciPy: Statistical methods

## Usage

```python
from data_preprocessing import DataPreprocessor

preprocessor = DataPreprocessor(data)

# Handle missing values
clean = preprocessor.handle_missing_values(strategy='mean')

# Handle outliers
clean = preprocessor.handle_outliers(['col1'], method='iqr', action='cap')

# Remove duplicates
clean = preprocessor.remove_duplicates()

# Quality report
report = preprocessor.generate_quality_report()
```

## Demo

```bash
python data_preprocessing.py
```
