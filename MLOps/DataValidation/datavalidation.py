"""
Data Validation for ML Pipelines
==================================

Production-ready data validation system:
- Schema validation
- Distribution shift detection
- Constraint checking (ranges, formats)
- Anomaly detection
- Data quality metrics
- Automated reporting and alerts

Author: Brill Consulting
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime
from scipy import stats
import json
from pathlib import Path


@dataclass
class ValidationRule:
    """Single validation rule."""
    name: str
    column: str
    rule_type: str  # "type", "range", "pattern", "custom"
    params: Dict[str, Any] = field(default_factory=dict)
    severity: str = "error"  # "error", "warning"
    description: str = ""


@dataclass
class ValidationResult:
    """Result of validation."""
    rule_name: str
    column: str
    passed: bool
    severity: str
    message: str
    failed_count: int = 0
    total_count: int = 0
    examples: List[Any] = field(default_factory=list)


class DataValidator:
    """
    Comprehensive data validation system.

    Validates:
    - Data types and schemas
    - Value ranges and constraints
    - Statistical distributions
    - Data quality metrics
    """

    def __init__(self, name: str = "data_validator"):
        """Initialize validator."""
        self.name = name
        self.rules: List[ValidationRule] = []
        self.baseline_stats: Optional[Dict] = None

    def add_rule(self, rule: ValidationRule):
        """Add validation rule."""
        self.rules.append(rule)

    def add_type_check(self, column: str, expected_type: str,
                       severity: str = "error"):
        """Add data type validation rule."""
        rule = ValidationRule(
            name=f"type_check_{column}",
            column=column,
            rule_type="type",
            params={"expected_type": expected_type},
            severity=severity,
            description=f"Check that {column} has type {expected_type}"
        )
        self.add_rule(rule)

    def add_range_check(self, column: str, min_value: Optional[float] = None,
                       max_value: Optional[float] = None,
                       severity: str = "error"):
        """Add range validation rule."""
        rule = ValidationRule(
            name=f"range_check_{column}",
            column=column,
            rule_type="range",
            params={"min": min_value, "max": max_value},
            severity=severity,
            description=f"Check that {column} is in range [{min_value}, {max_value}]"
        )
        self.add_rule(rule)

    def add_null_check(self, column: str, allow_null: bool = False,
                      severity: str = "error"):
        """Add null value check."""
        rule = ValidationRule(
            name=f"null_check_{column}",
            column=column,
            rule_type="null",
            params={"allow_null": allow_null},
            severity=severity,
            description=f"Check null values in {column}"
        )
        self.add_rule(rule)

    def add_unique_check(self, column: str, severity: str = "error"):
        """Add uniqueness check."""
        rule = ValidationRule(
            name=f"unique_check_{column}",
            column=column,
            rule_type="unique",
            params={},
            severity=severity,
            description=f"Check that {column} values are unique"
        )
        self.add_rule(rule)

    def add_pattern_check(self, column: str, pattern: str,
                         severity: str = "error"):
        """Add regex pattern check."""
        rule = ValidationRule(
            name=f"pattern_check_{column}",
            column=column,
            rule_type="pattern",
            params={"pattern": pattern},
            severity=severity,
            description=f"Check that {column} matches pattern {pattern}"
        )
        self.add_rule(rule)

    def add_enum_check(self, column: str, valid_values: List[Any],
                      severity: str = "error"):
        """Add enum/categorical check."""
        rule = ValidationRule(
            name=f"enum_check_{column}",
            column=column,
            rule_type="enum",
            params={"valid_values": valid_values},
            severity=severity,
            description=f"Check that {column} is in {valid_values}"
        )
        self.add_rule(rule)

    def add_custom_check(self, name: str, column: str,
                        check_func: Callable,
                        severity: str = "error",
                        description: str = ""):
        """Add custom validation function."""
        rule = ValidationRule(
            name=name,
            column=column,
            rule_type="custom",
            params={"func": check_func},
            severity=severity,
            description=description
        )
        self.add_rule(rule)

    def validate(self, data: pd.DataFrame) -> List[ValidationResult]:
        """
        Run all validation rules.

        Args:
            data: DataFrame to validate

        Returns:
            List of validation results
        """
        results = []

        for rule in self.rules:
            result = self._apply_rule(data, rule)
            results.append(result)

        return results

    def _apply_rule(self, data: pd.DataFrame,
                   rule: ValidationRule) -> ValidationResult:
        """Apply single validation rule."""

        if rule.column not in data.columns:
            return ValidationResult(
                rule_name=rule.name,
                column=rule.column,
                passed=False,
                severity=rule.severity,
                message=f"Column {rule.column} not found in data",
                failed_count=len(data),
                total_count=len(data)
            )

        column_data = data[rule.column]

        # Apply rule based on type
        if rule.rule_type == "type":
            passed, failed_mask = self._check_type(column_data, rule.params)

        elif rule.rule_type == "range":
            passed, failed_mask = self._check_range(column_data, rule.params)

        elif rule.rule_type == "null":
            passed, failed_mask = self._check_null(column_data, rule.params)

        elif rule.rule_type == "unique":
            passed, failed_mask = self._check_unique(column_data, rule.params)

        elif rule.rule_type == "pattern":
            passed, failed_mask = self._check_pattern(column_data, rule.params)

        elif rule.rule_type == "enum":
            passed, failed_mask = self._check_enum(column_data, rule.params)

        elif rule.rule_type == "custom":
            passed, failed_mask = self._check_custom(column_data, rule.params)

        else:
            return ValidationResult(
                rule_name=rule.name,
                column=rule.column,
                passed=False,
                severity=rule.severity,
                message=f"Unknown rule type: {rule.rule_type}",
                failed_count=0,
                total_count=len(data)
            )

        # Get examples of failures
        failed_count = failed_mask.sum() if isinstance(failed_mask, pd.Series) else 0
        examples = column_data[failed_mask].head(5).tolist() if isinstance(failed_mask, pd.Series) and failed_count > 0 else []

        message = f"{'✓' if passed else '✗'} {rule.description}"
        if not passed:
            message += f" ({failed_count}/{len(data)} failed)"

        return ValidationResult(
            rule_name=rule.name,
            column=rule.column,
            passed=passed,
            severity=rule.severity,
            message=message,
            failed_count=failed_count,
            total_count=len(data),
            examples=examples
        )

    def _check_type(self, data: pd.Series, params: Dict) -> tuple:
        """Check data type."""
        expected_type = params["expected_type"]

        # Map common type names
        type_map = {
            "int": np.integer,
            "float": np.floating,
            "str": object,
            "bool": bool,
            "datetime": "datetime64"
        }

        expected = type_map.get(expected_type, expected_type)

        if expected == "datetime64":
            is_valid = pd.api.types.is_datetime64_any_dtype(data)
            failed_mask = pd.Series([not is_valid] * len(data), index=data.index)
        else:
            is_valid = pd.api.types.is_dtype_equal(data.dtype, expected) or \
                      isinstance(data.dtype, type(np.dtype(expected)))
            failed_mask = pd.Series([not is_valid] * len(data), index=data.index)

        return is_valid, failed_mask

    def _check_range(self, data: pd.Series, params: Dict) -> tuple:
        """Check value range."""
        min_val = params.get("min")
        max_val = params.get("max")

        failed_mask = pd.Series([False] * len(data), index=data.index)

        if min_val is not None:
            failed_mask |= (data < min_val)

        if max_val is not None:
            failed_mask |= (data > max_val)

        passed = not failed_mask.any()
        return passed, failed_mask

    def _check_null(self, data: pd.Series, params: Dict) -> tuple:
        """Check null values."""
        allow_null = params.get("allow_null", False)

        failed_mask = data.isnull()

        if allow_null:
            passed = True
            failed_mask = pd.Series([False] * len(data), index=data.index)
        else:
            passed = not failed_mask.any()

        return passed, failed_mask

    def _check_unique(self, data: pd.Series, params: Dict) -> tuple:
        """Check uniqueness."""
        duplicates = data.duplicated()
        passed = not duplicates.any()
        return passed, duplicates

    def _check_pattern(self, data: pd.Series, params: Dict) -> tuple:
        """Check regex pattern."""
        pattern = params["pattern"]

        # Only check non-null string values
        string_data = data.astype(str)
        failed_mask = ~string_data.str.match(pattern, na=False)

        # Don't fail on null values
        failed_mask = failed_mask & data.notnull()

        passed = not failed_mask.any()
        return passed, failed_mask

    def _check_enum(self, data: pd.Series, params: Dict) -> tuple:
        """Check categorical values."""
        valid_values = params["valid_values"]

        failed_mask = ~data.isin(valid_values)

        # Don't fail on null values
        failed_mask = failed_mask & data.notnull()

        passed = not failed_mask.any()
        return passed, failed_mask

    def _check_custom(self, data: pd.Series, params: Dict) -> tuple:
        """Check custom function."""
        check_func = params["func"]

        try:
            failed_mask = ~data.apply(check_func)
            passed = not failed_mask.any()
        except Exception as e:
            # Function failed
            passed = False
            failed_mask = pd.Series([True] * len(data), index=data.index)

        return passed, failed_mask

    def compute_baseline(self, data: pd.DataFrame):
        """Compute baseline statistics for distribution drift detection."""
        stats_dict = {}

        for column in data.columns:
            col_data = data[column]

            if pd.api.types.is_numeric_dtype(col_data):
                stats_dict[column] = {
                    "mean": float(col_data.mean()),
                    "std": float(col_data.std()),
                    "min": float(col_data.min()),
                    "max": float(col_data.max()),
                    "quantiles": col_data.quantile([0.25, 0.5, 0.75]).tolist()
                }
            else:
                stats_dict[column] = {
                    "unique_values": int(col_data.nunique()),
                    "most_common": col_data.mode().tolist() if len(col_data.mode()) > 0 else [],
                    "value_counts": col_data.value_counts().head(10).to_dict()
                }

        self.baseline_stats = stats_dict
        print(f"✓ Computed baseline statistics for {len(stats_dict)} columns")

    def detect_drift(self, data: pd.DataFrame,
                    threshold: float = 0.05) -> Dict[str, Dict]:
        """
        Detect distribution drift from baseline.

        Args:
            data: New data to compare
            threshold: P-value threshold for drift detection

        Returns:
            Dictionary with drift results per column
        """
        if self.baseline_stats is None:
            raise ValueError("No baseline computed. Call compute_baseline() first.")

        drift_results = {}

        for column in data.columns:
            if column not in self.baseline_stats:
                continue

            col_data = data[column].dropna()

            if pd.api.types.is_numeric_dtype(col_data):
                # KS test for numeric columns
                baseline_mean = self.baseline_stats[column]["mean"]
                baseline_std = self.baseline_stats[column]["std"]

                # Generate baseline samples from stored stats
                baseline_samples = np.random.normal(
                    baseline_mean,
                    baseline_std,
                    size=1000
                )

                statistic, p_value = stats.ks_2samp(baseline_samples, col_data)

                drift_results[column] = {
                    "test": "ks_test",
                    "p_value": p_value,
                    "drift_detected": p_value < threshold,
                    "statistic": statistic,
                    "current_mean": float(col_data.mean()),
                    "baseline_mean": baseline_mean,
                    "mean_shift": float(col_data.mean() - baseline_mean)
                }
            else:
                # Chi-square test for categorical columns
                baseline_counts = self.baseline_stats[column].get("value_counts", {})
                current_counts = col_data.value_counts().to_dict()

                # Get common categories
                all_categories = set(baseline_counts.keys()) | set(current_counts.keys())

                baseline_freq = [baseline_counts.get(cat, 0) for cat in all_categories]
                current_freq = [current_counts.get(cat, 0) for cat in all_categories]

                # Normalize
                baseline_freq = np.array(baseline_freq) / (sum(baseline_freq) + 1e-10)
                current_freq = np.array(current_freq) / (sum(current_freq) + 1e-10)

                # Chi-square test
                chi2_stat = np.sum((current_freq - baseline_freq) ** 2 / (baseline_freq + 1e-10))
                p_value = 1 - stats.chi2.cdf(chi2_stat, len(all_categories) - 1)

                drift_results[column] = {
                    "test": "chi_square",
                    "p_value": p_value,
                    "drift_detected": p_value < threshold,
                    "statistic": chi2_stat
                }

        return drift_results

    def generate_report(self, results: List[ValidationResult],
                       save_path: Optional[str] = None) -> Dict:
        """Generate validation report."""
        total_rules = len(results)
        passed_rules = sum(1 for r in results if r.passed)
        failed_rules = total_rules - passed_rules

        errors = [r for r in results if not r.passed and r.severity == "error"]
        warnings = [r for r in results if not r.passed and r.severity == "warning"]

        report = {
            "timestamp": datetime.now().isoformat(),
            "validator_name": self.name,
            "summary": {
                "total_rules": total_rules,
                "passed": passed_rules,
                "failed": failed_rules,
                "errors": len(errors),
                "warnings": len(warnings),
                "success_rate": passed_rules / total_rules if total_rules > 0 else 0
            },
            "errors": [
                {
                    "rule": r.rule_name,
                    "column": r.column,
                    "message": r.message,
                    "failed_count": r.failed_count,
                    "examples": r.examples[:3]
                }
                for r in errors
            ],
            "warnings": [
                {
                    "rule": r.rule_name,
                    "column": r.column,
                    "message": r.message,
                    "failed_count": r.failed_count
                }
                for r in warnings
            ]
        }

        if save_path:
            path = Path(save_path)
            path.parent.mkdir(parents=True, exist_ok=True)

            with open(path, 'w') as f:
                json.dump(report, f, indent=2)

            print(f"✓ Saved validation report to {save_path}")

        return report


def demo():
    """Demo data validation."""
    print("Data Validation Demo")
    print("="*70 + "\n")

    # 1. Create validator
    print("1. Creating Validator")
    print("-"*70)

    validator = DataValidator(name="user_data_validator")
    print("✓ Validator created\n")

    # 2. Add validation rules
    print("2. Adding Validation Rules")
    print("-"*70)

    validator.add_type_check("age", "int")
    validator.add_range_check("age", min_value=0, max_value=120)
    validator.add_null_check("email", allow_null=False)
    validator.add_pattern_check("email", r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")
    validator.add_enum_check("status", ["active", "inactive", "pending"])
    validator.add_unique_check("user_id")

    print(f"✓ Added {len(validator.rules)} validation rules\n")

    # 3. Create test data (with some violations)
    print("3. Creating Test Data")
    print("-"*70)

    data = pd.DataFrame({
        "user_id": [1, 2, 3, 4, 5, 5],  # Duplicate!
        "age": [25, 35, -5, 150, 45, 30],  # Negative and out of range!
        "email": ["user1@example.com", "user2@example.com", "invalid", None, "user5@example.com", "user6@example.com"],  # Invalid pattern and null!
        "status": ["active", "inactive", "banned", "pending", "active", "inactive"]  # Invalid enum!
    })

    print(f"✓ Created data with {len(data)} rows\n")

    # 4. Run validation
    print("4. Running Validation")
    print("-"*70)

    results = validator.validate(data)

    for result in results:
        status = "✓" if result.passed else "✗"
        print(f"{status} {result.message}")
        if not result.passed and result.examples:
            print(f"  Examples: {result.examples[:3]}")

    print()

    # 5. Generate report
    print("5. Validation Report")
    print("-"*70)

    report = validator.generate_report(results, save_path="./validation_report.json")

    print(f"Total rules: {report['summary']['total_rules']}")
    print(f"Passed: {report['summary']['passed']}")
    print(f"Failed: {report['summary']['failed']}")
    print(f"Errors: {report['summary']['errors']}")
    print(f"Warnings: {report['summary']['warnings']}")
    print(f"Success rate: {report['summary']['success_rate']:.1%}")
    print()

    # 6. Baseline and drift detection
    print("6. Distribution Drift Detection")
    print("-"*70)

    # Compute baseline from good data
    baseline_data = pd.DataFrame({
        "age": np.random.normal(35, 10, 1000),
        "score": np.random.normal(75, 15, 1000)
    })

    validator.compute_baseline(baseline_data)

    # New data with drift
    new_data = pd.DataFrame({
        "age": np.random.normal(45, 10, 500),  # Mean shifted!
        "score": np.random.normal(75, 15, 500)  # No drift
    })

    drift_results = validator.detect_drift(new_data, threshold=0.05)

    for column, drift_info in drift_results.items():
        drift_status = "⚠ DRIFT" if drift_info["drift_detected"] else "✓ OK"
        print(f"{drift_status} {column}")
        print(f"  Test: {drift_info['test']}")
        print(f"  P-value: {drift_info['p_value']:.4f}")
        if "mean_shift" in drift_info:
            print(f"  Mean shift: {drift_info['mean_shift']:.2f}")
        print()

    print("="*70)
    print("✓ Data Validation Demo Complete!")


if __name__ == '__main__':
    demo()
