# 📊 Advanced Data Profiling System

**Comprehensive automated data profiling with statistical analysis, quality assessment, and anomaly detection**

---

## 📋 Overview

An enterprise-grade data profiling engine that automatically analyzes datasets to provide deep insights into data quality, structure, patterns, and statistical properties. Perfect for data discovery, quality assessment, and schema design.

## ✨ Key Features

### Automatic Data Type Detection
Intelligently detects 9 different data types:
- **Numeric** - Integers and floats
- **String** - Text data
- **Boolean** - True/false values
- **Date** - Date values (YYYY-MM-DD)
- **Timestamp** - Date and time values
- **Email** - Email addresses with validation
- **URL** - Web URLs (http/https)
- **JSON** - JSON objects and arrays
- **Unknown** - Unidentified types

### Statistical Analysis
- **Numeric Columns**: Mean, median, std dev, min/max, percentiles (25th, 75th), IQR
- **String Columns**: Min/max/average length, pattern analysis
- **All Columns**: Distinct count, null count, duplicates, completeness

### Quality Metrics
- **Completeness** - Percentage of non-null values
- **Uniqueness** - Ratio of distinct to total values
- **Validity** - Format compliance rate
- **Top Values** - Most frequent values with frequencies

### Advanced Features
- **Anomaly Detection** - IQR-based outlier detection for numeric data
- **Pattern Recognition** - Identifies alphanumeric, uppercase, lowercase, special characters patterns
- **Correlation Analysis** - Pearson correlation between numeric columns
- **Schema Inference** - Automatic SQL schema generation with types and constraints

### Profiling Levels
- **BASIC** - Essential statistics only
- **STANDARD** - Includes pattern analysis (default)
- **COMPREHENSIVE** - Full analysis with correlations and anomalies

## 📦 Installation

```bash
pip install -r requirements.txt
```

**Requirements:**
- Python 3.8+
- No external dependencies (pure Python)

## 🚀 Quick Start

### Basic Profiling

```python
from data_profiling import DataProfiler, ProfileLevel

# Initialize profiler
profiler = DataProfiler(profile_level=ProfileLevel.STANDARD)

# Sample data
data = [
    {'customer_id': 1001, 'name': 'John Doe', 'email': 'john@example.com', 'age': 32},
    {'customer_id': 1002, 'name': 'Jane Smith', 'email': 'jane@example.com', 'age': 28},
    {'customer_id': 1003, 'name': 'Bob Johnson', 'email': 'bob@example.com', 'age': 45}
]

# Profile the dataset
profile = profiler.profile_dataset('customers', data)

# View results
print(f"Overall Completeness: {profile.overall_completeness:.2%}")
print(f"Total Rows: {profile.total_rows}")
print(f"Total Columns: {profile.total_columns}")
```

### Generate Report

```python
# Generate human-readable report
report = profiler.generate_report('customers')
print(report)

# Export to JSON
profiler.export_profile('customers', 'customer_profile.json')
```

### Access Column Profiles

```python
# Get profile for specific column
age_profile = profile.column_profiles['age']

print(f"Data Type: {age_profile.data_type.value}")
print(f"Mean: {age_profile.mean:.2f}")
print(f"Median: {age_profile.median:.2f}")
print(f"Std Dev: {age_profile.std_dev:.2f}")
print(f"Range: [{age_profile.min_value}, {age_profile.max_value}]")
print(f"Completeness: {age_profile.completeness:.2%}")
```

### Comprehensive Profiling with Correlations

```python
# Use comprehensive profiling
profiler = DataProfiler(profile_level=ProfileLevel.COMPREHENSIVE)
profile = profiler.profile_dataset('sales_data', large_dataset)

# View correlations
for col, correlations in profile.correlations.items():
    print(f"\n{col} correlations:")
    for corr in correlations:
        print(f"  {corr['column']}: {corr['correlation']:.4f} ({corr['strength']})")
```

## 📊 Example Output

### Text Report

```
================================================================================
DATA PROFILING REPORT: customers
================================================================================
Generated: 2025-11-05T10:30:00
Profiling Time: 0.25s

DATASET OVERVIEW
--------------------------------------------------------------------------------
Total Rows: 1,000
Total Columns: 7
Overall Completeness: 97.50%
Duplicate Rows: 5

COLUMN PROFILES
--------------------------------------------------------------------------------

customer_id (numeric)
  Completeness: 100.00% (0 nulls)
  Uniqueness: 100.00% (1,000 distinct)
  Validity: 100.00%
  Range: [1001, 2000]
  Mean: 1500.50, Median: 1500.00
  Std Dev: 288.82

email (email)
  Completeness: 98.00% (20 nulls)
  Uniqueness: 98.50% (985 distinct)
  Validity: 99.80%
  Length: min=15, max=45, avg=28.5
  Anomalies: 0 types detected

age (numeric)
  Completeness: 95.00% (50 nulls)
  Uniqueness: 45.00% (45 distinct)
  Validity: 100.00%
  Range: [18, 85]
  Mean: 42.30, Median: 41.00
  Std Dev: 15.20
  Anomalies: 1 types detected

INFERRED SCHEMA
--------------------------------------------------------------------------------
  customer_id: NUMERIC NOT NULL
  name: VARCHAR(50) NOT NULL
  email: VARCHAR(255) NULL
  age: NUMERIC NULL
  city: VARCHAR(30) NOT NULL
  total_purchases: NUMERIC NULL
  is_active: BOOLEAN NOT NULL
================================================================================
```

### JSON Export Structure

```json
{
  "dataset_name": "customers",
  "total_rows": 1000,
  "total_columns": 7,
  "overall_completeness": 0.975,
  "duplicate_rows": 5,
  "profiling_duration_seconds": 0.25,
  "column_profiles": {
    "customer_id": {
      "column_name": "customer_id",
      "data_type": "numeric",
      "total_count": 1000,
      "null_count": 0,
      "distinct_count": 1000,
      "completeness": 1.0,
      "uniqueness": 1.0,
      "validity": 1.0,
      "min_value": 1001,
      "max_value": 2000,
      "mean": 1500.5,
      "median": 1500.0,
      "std_dev": 288.82
    }
  },
  "inferred_schema": {
    "customer_id": "NUMERIC NOT NULL",
    "name": "VARCHAR(50) NOT NULL"
  }
}
```

## 🎯 Use Cases

### 1. **Data Discovery**
- Explore unknown datasets quickly
- Understand data distributions and patterns
- Identify data types automatically

### 2. **Data Quality Assessment**
- Measure completeness and validity
- Detect anomalies and outliers
- Identify data quality issues early

### 3. **Schema Design**
- Generate SQL schemas automatically
- Determine appropriate column types
- Set constraints (NULL/NOT NULL)

### 4. **Data Migration**
- Profile source and target systems
- Verify data consistency
- Identify transformation requirements

### 5. **Compliance & Governance**
- Document data characteristics
- Track PII and sensitive data (emails, etc.)
- Maintain data catalogs

## 🔧 Advanced Features

### Anomaly Detection

```python
# Anomalies are automatically detected in COMPREHENSIVE mode
for col_name, col_profile in profile.column_profiles.items():
    if col_profile.anomalies:
        print(f"\n{col_name} anomalies:")
        for anomaly in col_profile.anomalies:
            print(f"  {anomaly['type']}: {anomaly['count']} instances")
            print(f"  {anomaly['description']}")
```

### Pattern Analysis

```python
# View patterns in string columns
email_profile = profile.column_profiles['email']
patterns = email_profile.pattern_analysis

print(f"Contains special chars: {patterns['contains_special_chars']}")
print(f"Mixed case: {patterns['mixed_case']}")
```

### Top Values

```python
# View most frequent values
for value_info in col_profile.top_values[:5]:
    print(f"{value_info['value']}: {value_info['count']} ({value_info['percentage']:.1f}%)")
```

### Sampling Large Datasets

```python
# Profile only first 10,000 rows for performance
profile = profiler.profile_dataset(
    'large_dataset',
    huge_data,
    sample_size=10000
)
```

## 📈 Performance

- **Speed**: Profiles 1M rows in ~3-5 seconds (STANDARD mode)
- **Memory**: Efficient memory usage with sampling support
- **Scalability**: Handles datasets with thousands of columns
- **Algorithms**: O(n) complexity for most operations

## 🧪 Testing

Run the demo:

```bash
python data_profiling.py
```

Expected output:
- Profiles sample customer dataset
- Generates comprehensive report
- Detects data types and anomalies
- Exports profile to JSON

## 📚 API Reference

### DataProfiler Class

**Constructor:**
```python
DataProfiler(profile_level: ProfileLevel = ProfileLevel.STANDARD)
```

**Methods:**
- `profile_dataset(dataset_name, data, sample_size)` - Profile a dataset
- `generate_report(dataset_name)` - Generate text report
- `export_profile(dataset_name, filepath)` - Export to JSON

### DatasetProfile Class

**Attributes:**
- `dataset_name: str` - Name of dataset
- `total_rows: int` - Number of rows
- `total_columns: int` - Number of columns
- `column_profiles: Dict[str, ColumnProfile]` - Column profiles
- `overall_completeness: float` - Overall completeness score
- `duplicate_rows: int` - Number of duplicate rows
- `correlations: Dict` - Correlation analysis (COMPREHENSIVE mode)
- `inferred_schema: Dict[str, str]` - Inferred SQL schema

### ColumnProfile Class

**Attributes:**
- `column_name: str`
- `data_type: DataType`
- `total_count: int`
- `null_count: int`
- `distinct_count: int`
- `completeness: float` - Percentage non-null
- `uniqueness: float` - Distinct ratio
- `validity: float` - Format compliance
- `min_value, max_value, mean, median, std_dev` - Numeric stats
- `min_length, max_length, avg_length` - String stats
- `top_values: List` - Most frequent values
- `pattern_analysis: Dict` - Pattern counts
- `anomalies: List` - Detected anomalies

### ProfileLevel Enum

- `BASIC` - Essential stats only
- `STANDARD` - Includes patterns (default)
- `COMPREHENSIVE` - Full analysis with correlations

## 🤝 Contributing

This is a portfolio project by Brill Consulting. For questions or suggestions, contact [clientbrill@gmail.com](mailto:clientbrill@gmail.com).

## 📄 License

Created by Brill Consulting for portfolio demonstration purposes.

---

**Author:** Brill Consulting
**LinkedIn:** [brillconsulting](https://www.linkedin.com/in/brillconsulting)
**Email:** clientbrill@gmail.com
