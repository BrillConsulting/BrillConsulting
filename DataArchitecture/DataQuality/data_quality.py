"""
Data Quality Framework
======================

Data quality validation, profiling, and monitoring:
- Data validation rules
- Quality metrics calculation
- Profiling and statistics
- Anomaly detection
- Quality reporting

Author: Brill Consulting
"""

import pandas as pd
import numpy as np
from typing import Dict, List
from datetime import datetime


class DataQuality:
    """Data quality framework."""

    def __init__(self):
        """Initialize data quality system."""
        self.rules = {}
        self.validation_history = []

    def add_rule(self, rule_id: str, rule: Dict):
        """Add validation rule."""
        self.rules[rule_id] = {
            **rule,
            "created_at": datetime.now().isoformat()
        }
        print(f"✓ Added rule: {rule_id}")

    def validate_not_null(self, data: pd.DataFrame, column: str) -> Dict:
        """Validate no null values."""
        null_count = data[column].isnull().sum()
        total = len(data)

        return {
            "rule": "not_null",
            "column": column,
            "passed": null_count == 0,
            "null_count": int(null_count),
            "null_percentage": float(null_count / total * 100)
        }

    def validate_unique(self, data: pd.DataFrame, column: str) -> Dict:
        """Validate unique values."""
        total = len(data)
        unique = data[column].nunique()
        duplicates = total - unique

        return {
            "rule": "unique",
            "column": column,
            "passed": duplicates == 0,
            "duplicate_count": int(duplicates),
            "unique_percentage": float(unique / total * 100)
        }

    def validate_range(self, data: pd.DataFrame, column: str,
                      min_val: float, max_val: float) -> Dict:
        """Validate value range."""
        out_of_range = ((data[column] < min_val) | (data[column] > max_val)).sum()
        total = len(data)

        return {
            "rule": "range",
            "column": column,
            "passed": out_of_range == 0,
            "out_of_range_count": int(out_of_range),
            "in_range_percentage": float((total - out_of_range) / total * 100),
            "min": min_val,
            "max": max_val
        }

    def validate_format(self, data: pd.DataFrame, column: str, pattern: str) -> Dict:
        """Validate data format (regex pattern)."""
        # Simplified format check
        invalid = data[column].astype(str).str.contains(pattern, regex=True, na=False) == False
        invalid_count = invalid.sum()
        total = len(data)

        return {
            "rule": "format",
            "column": column,
            "passed": invalid_count == 0,
            "invalid_count": int(invalid_count),
            "valid_percentage": float((total - invalid_count) / total * 100),
            "pattern": pattern
        }

    def run_validation(self, data: pd.DataFrame, rules: List[Dict]) -> Dict:
        """Run all validation rules."""
        print(f"Running {len(rules)} validation rules...")

        results = {
            "timestamp": datetime.now().isoformat(),
            "total_rules": len(rules),
            "passed": 0,
            "failed": 0,
            "results": []
        }

        for rule in rules:
            rule_type = rule["type"]
            column = rule["column"]

            if rule_type == "not_null":
                result = self.validate_not_null(data, column)
            elif rule_type == "unique":
                result = self.validate_unique(data, column)
            elif rule_type == "range":
                result = self.validate_range(data, column, rule["min"], rule["max"])
            elif rule_type == "format":
                result = self.validate_format(data, column, rule["pattern"])
            else:
                result = {"rule": rule_type, "column": column, "passed": False, "error": "Unknown rule"}

            if result["passed"]:
                results["passed"] += 1
            else:
                results["failed"] += 1

            results["results"].append(result)

        self.validation_history.append(results)

        print(f"✓ Validation complete: {results['passed']}/{results['total_rules']} passed")
        return results

    def profile_data(self, data: pd.DataFrame) -> Dict:
        """Generate data quality profile."""
        print("Profiling data...")

        profile = {
            "row_count": len(data),
            "column_count": len(data.columns),
            "memory_usage_mb": data.memory_usage(deep=True).sum() / 1024**2,
            "columns": {}
        }

        for col in data.columns:
            col_profile = {
                "dtype": str(data[col].dtype),
                "null_count": int(data[col].isnull().sum()),
                "null_percentage": float(data[col].isnull().sum() / len(data) * 100),
                "unique_count": int(data[col].nunique()),
                "unique_percentage": float(data[col].nunique() / len(data) * 100)
            }

            # Numeric stats
            if np.issubdtype(data[col].dtype, np.number):
                col_profile["mean"] = float(data[col].mean())
                col_profile["std"] = float(data[col].std())
                col_profile["min"] = float(data[col].min())
                col_profile["max"] = float(data[col].max())
                col_profile["median"] = float(data[col].median())

            profile["columns"][col] = col_profile

        print(f"✓ Profiled {profile['column_count']} columns")
        return profile

    def detect_anomalies(self, data: pd.DataFrame, column: str, method: str = "zscore") -> Dict:
        """Detect anomalies in data."""
        print(f"Detecting anomalies in {column}...")

        if not np.issubdtype(data[column].dtype, np.number):
            return {"error": "Column must be numeric"}

        if method == "zscore":
            z_scores = np.abs((data[column] - data[column].mean()) / data[column].std())
            anomalies = z_scores > 3
        elif method == "iqr":
            Q1 = data[column].quantile(0.25)
            Q3 = data[column].quantile(0.75)
            IQR = Q3 - Q1
            anomalies = (data[column] < Q1 - 1.5 * IQR) | (data[column] > Q3 + 1.5 * IQR)
        else:
            return {"error": "Unknown method"}

        anomaly_count = anomalies.sum()

        result = {
            "column": column,
            "method": method,
            "anomaly_count": int(anomaly_count),
            "anomaly_percentage": float(anomaly_count / len(data) * 100),
            "anomaly_indices": anomalies[anomalies].index.tolist()[:10]  # First 10
        }

        print(f"✓ Found {anomaly_count} anomalies ({result['anomaly_percentage']:.2f}%)")
        return result

    def generate_report(self, data: pd.DataFrame) -> Dict:
        """Generate comprehensive quality report."""
        print("\nGenerating Quality Report...")
        print("="*50)

        report = {
            "timestamp": datetime.now().isoformat(),
            "profile": self.profile_data(data),
            "validation_history": len(self.validation_history)
        }

        if self.validation_history:
            latest = self.validation_history[-1]
            report["latest_validation"] = {
                "passed": latest["passed"],
                "failed": latest["failed"],
                "pass_rate": latest["passed"] / latest["total_rules"] * 100
            }

        print("✓ Report generated")
        return report


def demo():
    """Demo data quality."""
    print("Data Quality Demo")
    print("="*50)

    # Sample data
    data = pd.DataFrame({
        "user_id": list(range(1, 101)) + [50],  # Has duplicate
        "age": list(range(18, 118)) + [200],  # Has outlier
        "email": [f"user{i}@example.com" for i in range(1, 101)] + ["invalid"],
        "amount": [100 + np.random.normal(0, 10) for _ in range(101)]
    })

    # Add some nulls
    data.loc[0:5, "email"] = None

    dq = DataQuality()

    # 1. Define rules
    print("\n1. Defining Validation Rules")
    print("-"*50)

    rules = [
        {"type": "not_null", "column": "user_id"},
        {"type": "unique", "column": "user_id"},
        {"type": "range", "column": "age", "min": 18, "max": 100},
        {"type": "not_null", "column": "email"}
    ]

    # 2. Run validation
    print("\n2. Running Validation")
    print("-"*50)

    validation = dq.run_validation(data, rules)
    print(f"\nResults: {validation['passed']} passed, {validation['failed']} failed")

    for result in validation["results"]:
        status = "✓" if result["passed"] else "✗"
        print(f"  {status} {result['rule']} on {result['column']}")

    # 3. Data profiling
    print("\n3. Data Profiling")
    print("-"*50)

    profile = dq.profile_data(data)
    print(f"Rows: {profile['row_count']}, Columns: {profile['column_count']}")
    print(f"Memory: {profile['memory_usage_mb']:.2f} MB")

    print("\nColumn profiles:")
    for col, col_profile in profile["columns"].items():
        print(f"  {col}: {col_profile['null_percentage']:.1f}% null, "
              f"{col_profile['unique_percentage']:.1f}% unique")

    # 4. Anomaly detection
    print("\n4. Anomaly Detection")
    print("-"*50)

    anomalies = dq.detect_anomalies(data, "age", method="iqr")
    print(f"Anomalies in age: {anomalies['anomaly_count']} "
          f"({anomalies['anomaly_percentage']:.2f}%)")

    # 5. Generate report
    print("\n5. Quality Report")
    print("-"*50)

    report = dq.generate_report(data)
    if "latest_validation" in report:
        print(f"Validation pass rate: {report['latest_validation']['pass_rate']:.1f}%")

    print("\n✓ Data Quality Demo Complete!")


if __name__ == '__main__':
    demo()
