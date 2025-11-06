# Data Validation System

Production-ready data validation framework for ML pipelines with comprehensive schema checking, distribution drift detection, and automated reporting.

## Features

### Validation Rules
- **Type Checking**: Validate data types (int, float, str, bool, datetime)
- **Range Validation**: Min/max value constraints for numeric columns
- **Null Checking**: Control null value policies per column
- **Uniqueness**: Ensure unique values in key columns
- **Pattern Matching**: Regex validation for string formats (emails, IDs)
- **Enum Validation**: Categorical value validation
- **Custom Rules**: Extensible validation with custom functions

### Distribution Drift Detection
- **KS Test**: Kolmogorov-Smirnov test for numeric distributions
- **Chi-Square Test**: Categorical distribution comparison
- **Baseline Computation**: Store reference statistics for comparison
- **Automated Alerts**: Flag significant distribution shifts

### Reporting & Monitoring
- **Detailed Reports**: JSON reports with error examples
- **Severity Levels**: Errors vs warnings
- **Success Metrics**: Validation pass rates and statistics
- **Examples**: Failed data examples for debugging

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from datavalidation import DataValidator
import pandas as pd

# Create validator
validator = DataValidator(name="my_validator")

# Add validation rules
validator.add_type_check("age", "int")
validator.add_range_check("age", min_value=0, max_value=120)
validator.add_null_check("email", allow_null=False)
validator.add_pattern_check("email", r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")
validator.add_enum_check("status", ["active", "inactive", "pending"])

# Validate data
data = pd.read_csv("data.csv")
results = validator.validate(data)

# Generate report
report = validator.generate_report(results, save_path="validation_report.json")
print(f"Success rate: {report['summary']['success_rate']:.1%}")
```

## Usage Examples

### 1. Schema Validation

```python
# Define validation rules
validator = DataValidator("user_data_validator")

# Type checks
validator.add_type_check("user_id", "int")
validator.add_type_check("signup_date", "datetime")
validator.add_type_check("is_premium", "bool")

# Range checks
validator.add_range_check("age", min_value=18, max_value=100)
validator.add_range_check("credit_score", min_value=300, max_value=850)

# Pattern validation
validator.add_pattern_check("phone", r"^\+?1?\d{9,15}$")
validator.add_pattern_check("postal_code", r"^\d{5}(-\d{4})?$")

# Run validation
results = validator.validate(data)

# Check results
for result in results:
    if not result.passed:
        print(f"✗ {result.message}")
        print(f"  Failed: {result.failed_count}/{result.total_count}")
        print(f"  Examples: {result.examples[:3]}")
```

### 2. Distribution Drift Detection

```python
# Compute baseline from training data
baseline_data = pd.read_csv("training_data.csv")
validator.compute_baseline(baseline_data)

# Check new data for drift
production_data = pd.read_csv("production_data.csv")
drift_results = validator.detect_drift(production_data, threshold=0.05)

# Analyze drift
for column, drift_info in drift_results.items():
    if drift_info['drift_detected']:
        print(f"⚠ DRIFT DETECTED in {column}")
        print(f"  Test: {drift_info['test']}")
        print(f"  P-value: {drift_info['p_value']:.4f}")

        if 'mean_shift' in drift_info:
            print(f"  Mean shift: {drift_info['mean_shift']:.2f}")
```

### 3. Custom Validation Rules

```python
# Define custom validation function
def is_valid_customer_id(value):
    """Customer ID must start with 'C' and have 8 digits."""
    return isinstance(value, str) and value.startswith('C') and len(value) == 9

# Add custom rule
validator.add_custom_check(
    name="customer_id_format",
    column="customer_id",
    check_func=is_valid_customer_id,
    severity="error",
    description="Validate customer ID format (C + 8 digits)"
)

# Complex custom validation
def is_reasonable_transaction(row):
    """Transaction amount should match product price * quantity."""
    return abs(row['amount'] - (row['price'] * row['quantity'])) < 0.01

# Apply to DataFrame
validator.add_custom_check(
    name="transaction_validation",
    column="amount",
    check_func=lambda x: True,  # Placeholder for row-level validation
    description="Validate transaction calculations"
)
```

### 4. Validation with Severity Levels

```python
# Critical errors (will fail pipeline)
validator.add_type_check("user_id", "int", severity="error")
validator.add_null_check("user_id", allow_null=False, severity="error")

# Warnings (log but don't fail)
validator.add_range_check("age", min_value=0, max_value=150, severity="warning")
validator.add_pattern_check("phone", r"^\d{10}$", severity="warning")

# Run validation
results = validator.validate(data)

# Generate report with error/warning breakdown
report = validator.generate_report(results)
print(f"Errors: {report['summary']['errors']}")
print(f"Warnings: {report['summary']['warnings']}")

# Fail pipeline only on errors
if report['summary']['errors'] > 0:
    raise ValueError(f"Validation failed with {report['summary']['errors']} errors")
```

### 5. Comprehensive Data Quality Report

```python
# Add multiple validation rules
validator.add_type_check("user_id", "int")
validator.add_unique_check("user_id")
validator.add_null_check("email", allow_null=False)
validator.add_pattern_check("email", r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")
validator.add_enum_check("country", ["US", "UK", "CA", "AU", "DE"])
validator.add_range_check("purchase_amount", min_value=0, max_value=10000)

# Validate
results = validator.validate(data)

# Generate detailed report
report = validator.generate_report(results, save_path="./reports/validation.json")

# Report structure:
# {
#   "timestamp": "2024-01-15T10:30:00",
#   "validator_name": "my_validator",
#   "summary": {
#     "total_rules": 6,
#     "passed": 4,
#     "failed": 2,
#     "errors": 2,
#     "warnings": 0,
#     "success_rate": 0.67
#   },
#   "errors": [
#     {
#       "rule": "pattern_check_email",
#       "column": "email",
#       "message": "...",
#       "failed_count": 15,
#       "examples": ["invalid1", "invalid2"]
#     }
#   ]
# }
```

### 6. Monitoring Pipeline Data Quality

```python
# Production pipeline validation
class DataQualityMonitor:
    def __init__(self):
        self.validator = DataValidator("pipeline_monitor")
        self._setup_rules()

    def _setup_rules(self):
        """Setup validation rules for pipeline."""
        # Critical checks
        self.validator.add_type_check("timestamp", "datetime", severity="error")
        self.validator.add_null_check("user_id", allow_null=False, severity="error")

        # Quality checks
        self.validator.add_range_check("value", min_value=0, severity="warning")

    def validate_batch(self, data: pd.DataFrame) -> bool:
        """Validate a data batch."""
        results = self.validator.validate(data)
        report = self.validator.generate_report(results)

        # Log metrics
        self.log_metrics(report['summary'])

        # Fail on errors
        return report['summary']['errors'] == 0

    def log_metrics(self, summary: dict):
        """Log validation metrics to monitoring system."""
        print(f"Validation success rate: {summary['success_rate']:.1%}")
        # Send to Prometheus, CloudWatch, etc.

# Use in pipeline
monitor = DataQualityMonitor()

for batch in data_stream:
    if not monitor.validate_batch(batch):
        alert("Data quality check failed!")
        break
```

## Validation Rule Types

### Type Check
Validates column data types.
```python
validator.add_type_check("column_name", "int")  # int, float, str, bool, datetime
```

### Range Check
Validates numeric ranges.
```python
validator.add_range_check("score", min_value=0, max_value=100)
```

### Null Check
Controls null value policies.
```python
validator.add_null_check("required_field", allow_null=False)
```

### Unique Check
Ensures uniqueness.
```python
validator.add_unique_check("primary_key")
```

### Pattern Check
Regex validation.
```python
validator.add_pattern_check("email", r"^[a-zA-Z0-9._%+-]+@...")
```

### Enum Check
Categorical validation.
```python
validator.add_enum_check("status", ["active", "inactive", "pending"])
```

### Custom Check
Custom validation function.
```python
validator.add_custom_check("custom_rule", "column", my_function, description="...")
```

## Distribution Drift Detection

### KS Test (Numeric Columns)
Kolmogorov-Smirnov test for continuous distributions.

```python
# Detects changes in:
# - Mean shifts
# - Variance changes
# - Distribution shape changes
```

### Chi-Square Test (Categorical Columns)
Tests for changes in categorical distributions.

```python
# Detects changes in:
# - Category frequencies
# - New/removed categories
# - Distribution shifts
```

### Baseline Management
```python
# Compute baseline from training data
validator.compute_baseline(training_data)

# Check production data
drift_results = validator.detect_drift(production_data, threshold=0.05)

# Interpret results
for column, info in drift_results.items():
    if info['drift_detected']:
        print(f"Drift in {column}: p-value = {info['p_value']:.4f}")
```

## Best Practices

### 1. Define Validation Early
Define validation rules during data exploration:
```python
# During EDA, identify constraints
print(data['age'].describe())  # Range: 0-120
print(data['status'].unique())  # Values: ['active', 'inactive']

# Convert to rules
validator.add_range_check("age", min_value=0, max_value=120)
validator.add_enum_check("status", ["active", "inactive"])
```

### 2. Use Appropriate Severity
```python
# Critical for pipeline - use "error"
validator.add_null_check("user_id", allow_null=False, severity="error")

# Quality checks - use "warning"
validator.add_range_check("age", min_value=0, max_value=150, severity="warning")
```

### 3. Monitor Drift Regularly
```python
# Re-compute baseline periodically
if days_since_baseline > 30:
    validator.compute_baseline(recent_training_data)

# Check drift on each production batch
drift_results = validator.detect_drift(production_batch)
```

### 4. Save Validation Reports
```python
# Save reports for audit trail
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
report_path = f"./reports/validation_{timestamp}.json"
validator.generate_report(results, save_path=report_path)
```

### 5. Integrate with CI/CD
```python
# In CI pipeline
def test_data_quality():
    validator = DataValidator("ci_validator")
    validator.add_type_check("id", "int")
    validator.add_null_check("id", allow_null=False)

    test_data = pd.read_csv("test_data.csv")
    results = validator.validate(test_data)

    report = validator.generate_report(results)
    assert report['summary']['errors'] == 0, "Data quality check failed"
```

## Integration with ML Pipelines

### Training Pipeline
```python
# Validate training data
train_validator = DataValidator("training")
train_validator.add_range_check("feature_1", min_value=-3, max_value=3)
results = train_validator.validate(X_train)

if not all(r.passed for r in results):
    raise ValueError("Training data validation failed")

# Compute baseline for production monitoring
train_validator.compute_baseline(X_train)
```

### Inference Pipeline
```python
# Validate inference input
inference_validator = DataValidator("inference")
inference_validator.add_type_check("feature_1", "float")
inference_validator.add_null_check("feature_1", allow_null=False)

# Check for drift
drift_results = inference_validator.detect_drift(X_inference)
if any(d['drift_detected'] for d in drift_results.values()):
    alert("Data drift detected in production!")
```

## Example Output

```
Data Validation Demo
======================================================================

2. Adding Validation Rules
----------------------------------------------------------------------
✓ Added 6 validation rules

4. Running Validation
----------------------------------------------------------------------
✓ ✓ Check that user_id has type int
✗ ✗ Check that user_id values are unique (1/6 failed)
  Examples: [5]
✓ ✓ Check that age is in range [0, 120]
✗ ✗ Check null values in email (1/6 failed)
  Examples: [None]
✗ ✗ Check that email matches pattern ^[a-zA-Z0-9._%+-]+@... (1/6 failed)
  Examples: ['invalid']
✗ ✗ Check that status is in ['active', 'inactive', 'pending'] (1/6 failed)
  Examples: ['banned']

5. Validation Report
----------------------------------------------------------------------
Total rules: 6
Passed: 2
Failed: 4
Errors: 4
Warnings: 0
Success rate: 33.3%

6. Distribution Drift Detection
----------------------------------------------------------------------
✓ Computed baseline statistics for 2 columns

⚠ DRIFT age
  Test: ks_test
  P-value: 0.0000
  Mean shift: 10.02

✓ OK score
  Test: ks_test
  P-value: 0.8234
  Mean shift: 0.12

✓ Data Validation Demo Complete!
```

## Requirements

- Python 3.7+
- pandas
- numpy
- scipy

## Production Considerations

- **Performance**: Validation adds overhead - sample large datasets
- **Baseline Updates**: Refresh baselines periodically (e.g., monthly)
- **Alert Thresholds**: Tune p-value thresholds based on use case
- **Report Storage**: Archive validation reports for compliance
- **Monitoring Integration**: Send metrics to Prometheus, DataDog, etc.

## Author

**Brill Consulting**
- Email: clientbrill@gmail.com
- LinkedIn: [brillconsulting](https://www.linkedin.com/in/brillconsulting)

## License

Professional implementation for portfolio demonstration.
