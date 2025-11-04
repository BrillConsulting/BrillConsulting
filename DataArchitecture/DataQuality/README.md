# Data Quality Framework

Comprehensive data quality validation, profiling, and monitoring.

## Features

- Validation rules (not null, unique, range, format)
- Data profiling and statistics
- Anomaly detection (Z-score, IQR)
- Quality reporting
- Validation history tracking

## Usage

```python
from data_quality import DataQuality

dq = DataQuality()

# Define validation rules
rules = [
    {"type": "not_null", "column": "id"},
    {"type": "range", "column": "age", "min": 0, "max": 120}
]

# Run validation
results = dq.run_validation(data, rules)

# Profile data
profile = dq.profile_data(data)

# Detect anomalies
anomalies = dq.detect_anomalies(data, "amount", method="zscore")
```

## Demo

```bash
python data_quality.py
```
