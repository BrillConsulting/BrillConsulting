"""
Advanced Data Quality Framework
Author: BrillConsulting
Description: Comprehensive data quality automation with rules engine,
validation, monitoring, and alerting
"""

import json
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
from dataclasses import dataclass, asdict, field
from enum import Enum
import re


class QualityDimension(Enum):
    """Data quality dimensions"""
    COMPLETENESS = "completeness"
    ACCURACY = "accuracy"
    CONSISTENCY = "consistency"
    VALIDITY = "validity"
    TIMELINESS = "timeliness"
    UNIQUENESS = "uniqueness"


class RuleSeverity(Enum):
    """Severity levels for quality rules"""
    CRITICAL = "critical"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class RuleStatus(Enum):
    """Status of rule execution"""
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


@dataclass
class QualityRule:
    """Represents a data quality rule"""
    rule_id: str
    name: str
    dimension: QualityDimension
    severity: RuleSeverity
    description: str
    rule_logic: str
    threshold: Optional[float] = None
    enabled: bool = True
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['dimension'] = self.dimension.value
        result['severity'] = self.severity.value
        return result


@dataclass
class QualityCheckResult:
    """Result of a quality rule execution"""
    rule_id: str
    rule_name: str
    status: RuleStatus
    dimension: QualityDimension
    severity: RuleSeverity
    score: float  # 0.0 to 1.0
    passed_count: int
    failed_count: int
    total_count: int
    failure_details: List[Dict[str, Any]] = field(default_factory=list)
    execution_time_ms: float = 0.0
    checked_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['status'] = self.status.value
        result['dimension'] = self.dimension.value
        result['severity'] = self.severity.value
        return result


@dataclass
class QualityReport:
    """Comprehensive quality assessment report"""
    dataset_name: str
    total_rules: int
    passed_rules: int
    failed_rules: int
    overall_score: float
    dimension_scores: Dict[str, float] = field(default_factory=dict)
    check_results: List[QualityCheckResult] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['check_results'] = [cr.to_dict() for cr in self.check_results]
        return result


class DataQualityFramework:
    """
    Advanced Data Quality Framework

    Features:
    - Rule-based quality validation
    - Multi-dimensional quality assessment
    - Automated quality checks
    - Threshold-based alerting
    - Quality scoring and reporting
    - Custom rule definitions
    """

    def __init__(self, framework_name: str = "default"):
        """
        Initialize quality framework

        Args:
            framework_name: Name of the framework instance
        """
        self.framework_name = framework_name
        self.rules: Dict[str, QualityRule] = {}
        self.reports: List[QualityReport] = []

    def add_rule(self, rule: QualityRule) -> None:
        """
        Add a quality rule to the framework

        Args:
            rule: Quality rule to add
        """
        self.rules[rule.rule_id] = rule
        print(f"✓ Rule added: {rule.name} ({rule.dimension.value}, {rule.severity.value})")

    def create_standard_rules(self) -> None:
        """Create a set of standard quality rules"""

        # Completeness rules
        self.add_rule(QualityRule(
            rule_id="COMP_001",
            name="No Null Values",
            dimension=QualityDimension.COMPLETENESS,
            severity=RuleSeverity.ERROR,
            description="Check for null or empty values in required fields",
            rule_logic="value IS NOT NULL AND value != ''",
            threshold=0.95,
            tags=["completeness", "required"]
        ))

        # Accuracy rules
        self.add_rule(QualityRule(
            rule_id="ACC_001",
            name="Email Format Validation",
            dimension=QualityDimension.ACCURACY,
            severity=RuleSeverity.ERROR,
            description="Validate email addresses format",
            rule_logic="regex_match(value, '^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$')",
            threshold=0.99,
            tags=["accuracy", "email"]
        ))

        # Consistency rules
        self.add_rule(QualityRule(
            rule_id="CONS_001",
            name="Date Consistency",
            dimension=QualityDimension.CONSISTENCY,
            severity=RuleSeverity.WARNING,
            description="Ensure dates are in consistent format",
            rule_logic="date_format_match(value, 'YYYY-MM-DD')",
            threshold=1.0,
            tags=["consistency", "date"]
        ))

        # Validity rules
        self.add_rule(QualityRule(
            rule_id="VAL_001",
            name="Numeric Range Validation",
            dimension=QualityDimension.VALIDITY,
            severity=RuleSeverity.ERROR,
            description="Check if numeric values are within valid range",
            rule_logic="value >= min_value AND value <= max_value",
            threshold=0.98,
            tags=["validity", "range"]
        ))

        # Uniqueness rules
        self.add_rule(QualityRule(
            rule_id="UNIQ_001",
            name="Primary Key Uniqueness",
            dimension=QualityDimension.UNIQUENESS,
            severity=RuleSeverity.CRITICAL,
            description="Ensure primary key values are unique",
            rule_logic="COUNT(DISTINCT value) = COUNT(value)",
            threshold=1.0,
            tags=["uniqueness", "primary_key"]
        ))

        print(f"✓ {len(self.rules)} standard rules created")

    def run_quality_checks(self, dataset_name: str, data: List[Dict[str, Any]],
                          column_rules: Optional[Dict[str, List[str]]] = None) -> QualityReport:
        """
        Run quality checks on dataset

        Args:
            dataset_name: Name of dataset
            data: Dataset to check
            column_rules: Optional mapping of columns to rule IDs

        Returns:
            Quality report
        """
        start_time = datetime.now()

        print(f"Running quality checks on: {dataset_name}")
        print(f"  Total records: {len(data)}")
        print(f"  Active rules: {len([r for r in self.rules.values() if r.enabled])}")

        check_results = []

        # If no column rules specified, apply all rules to all columns
        if not column_rules and data:
            columns = data[0].keys()
            column_rules = {col: list(self.rules.keys()) for col in columns}

        # Execute rules for each column
        for column, rule_ids in (column_rules or {}).items():
            for rule_id in rule_ids:
                if rule_id not in self.rules or not self.rules[rule_id].enabled:
                    continue

                rule = self.rules[rule_id]
                result = self._execute_rule(rule, data, column)
                check_results.append(result)

        # Calculate overall metrics
        total_rules = len(check_results)
        passed_rules = sum(1 for r in check_results if r.status == RuleStatus.PASSED)
        failed_rules = sum(1 for r in check_results if r.status == RuleStatus.FAILED)

        overall_score = passed_rules / total_rules if total_rules > 0 else 0.0

        # Calculate dimension scores
        dimension_scores = {}
        for dimension in QualityDimension:
            dim_results = [r for r in check_results if r.dimension == dimension]
            if dim_results:
                dimension_scores[dimension.value] = sum(r.score for r in dim_results) / len(dim_results)

        # Create report
        report = QualityReport(
            dataset_name=dataset_name,
            total_rules=total_rules,
            passed_rules=passed_rules,
            failed_rules=failed_rules,
            overall_score=overall_score,
            dimension_scores=dimension_scores,
            check_results=check_results,
            summary={
                'total_records': len(data),
                'check_duration_ms': (datetime.now() - start_time).total_seconds() * 1000,
                'critical_failures': sum(1 for r in check_results
                                       if r.status == RuleStatus.FAILED
                                       and r.severity == RuleSeverity.CRITICAL)
            }
        )

        self.reports.append(report)

        print(f"✓ Quality checks completed")
        print(f"  Overall score: {overall_score:.2%}")
        print(f"  Passed: {passed_rules}/{total_rules}")
        print(f"  Failed: {failed_rules}/{total_rules}")

        return report

    def _execute_rule(self, rule: QualityRule, data: List[Dict[str, Any]],
                     column: str) -> QualityCheckResult:
        """Execute a single quality rule"""
        start_time = datetime.now()

        passed_count = 0
        failed_count = 0
        failure_details = []

        # Extract column values
        values = [row.get(column) for row in data]
        total_count = len(values)

        # Apply rule logic based on dimension
        if rule.dimension == QualityDimension.COMPLETENESS:
            for idx, value in enumerate(values):
                if value is not None and value != "":
                    passed_count += 1
                else:
                    failed_count += 1
                    if len(failure_details) < 10:  # Limit failure details
                        failure_details.append({
                            'row_index': idx,
                            'column': column,
                            'value': value,
                            'reason': 'Null or empty value'
                        })

        elif rule.dimension == QualityDimension.ACCURACY:
            if 'email' in rule.tags:
                email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
                for idx, value in enumerate(values):
                    if value and re.match(email_pattern, str(value)):
                        passed_count += 1
                    else:
                        failed_count += 1
                        if len(failure_details) < 10:
                            failure_details.append({
                                'row_index': idx,
                                'column': column,
                                'value': value,
                                'reason': 'Invalid email format'
                            })

        elif rule.dimension == QualityDimension.UNIQUENESS:
            unique_values = set(str(v) for v in values if v is not None)
            non_null_count = sum(1 for v in values if v is not None)

            if len(unique_values) == non_null_count:
                passed_count = total_count
            else:
                failed_count = non_null_count - len(unique_values)
                passed_count = total_count - failed_count

                # Find duplicates
                value_counts = {}
                for idx, value in enumerate(values):
                    if value is not None:
                        val_str = str(value)
                        if val_str in value_counts:
                            value_counts[val_str].append(idx)
                        else:
                            value_counts[val_str] = [idx]

                for value, indices in value_counts.items():
                    if len(indices) > 1 and len(failure_details) < 10:
                        failure_details.append({
                            'row_indices': indices,
                            'column': column,
                            'value': value,
                            'reason': f'Duplicate value (appears {len(indices)} times)'
                        })

        elif rule.dimension == QualityDimension.VALIDITY:
            # Example: numeric range validation
            for idx, value in enumerate(values):
                try:
                    if value is not None:
                        float(value)  # Check if numeric
                        passed_count += 1
                    else:
                        failed_count += 1
                except (ValueError, TypeError):
                    failed_count += 1
                    if len(failure_details) < 10:
                        failure_details.append({
                            'row_index': idx,
                            'column': column,
                            'value': value,
                            'reason': 'Invalid numeric value'
                        })

        else:
            # Default behavior for other dimensions
            passed_count = total_count

        # Calculate score
        score = passed_count / total_count if total_count > 0 else 0.0

        # Determine status
        if rule.threshold is not None:
            status = RuleStatus.PASSED if score >= rule.threshold else RuleStatus.FAILED
        else:
            status = RuleStatus.PASSED if failed_count == 0 else RuleStatus.FAILED

        execution_time = (datetime.now() - start_time).total_seconds() * 1000

        return QualityCheckResult(
            rule_id=rule.rule_id,
            rule_name=rule.name,
            status=status,
            dimension=rule.dimension,
            severity=rule.severity,
            score=score,
            passed_count=passed_count,
            failed_count=failed_count,
            total_count=total_count,
            failure_details=failure_details,
            execution_time_ms=execution_time
        )

    def export_report(self, report: QualityReport, filepath: str) -> None:
        """Export quality report to JSON"""
        with open(filepath, 'w') as f:
            json.dump(report.to_dict(), f, indent=2)
        print(f"✓ Report exported to: {filepath}")

    def generate_html_report(self, report: QualityReport) -> str:
        """Generate HTML quality report"""
        html = f"""
        <html>
        <head>
            <title>Data Quality Report - {report.dataset_name}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background: #2c3e50; color: white; padding: 20px; }}
                .score {{ font-size: 48px; font-weight: bold; }}
                .passed {{ color: #27ae60; }}
                .failed {{ color: #e74c3c; }}
                table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #34495e; color: white; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Data Quality Report</h1>
                <p>Dataset: {report.dataset_name}</p>
                <p>Generated: {report.generated_at}</p>
            </div>
            <div style="margin-top: 20px;">
                <h2>Overall Score: <span class="score {'passed' if report.overall_score >= 0.9 else 'failed'}">{report.overall_score:.1%}</span></h2>
                <p>Passed Rules: {report.passed_rules}/{report.total_rules}</p>
                <p>Failed Rules: {report.failed_rules}/{report.total_rules}</p>
            </div>
            <h3>Dimension Scores</h3>
            <table>
                <tr><th>Dimension</th><th>Score</th></tr>
                {''.join(f'<tr><td>{dim}</td><td>{score:.2%}</td></tr>' for dim, score in report.dimension_scores.items())}
            </table>
            <h3>Check Results</h3>
            <table>
                <tr><th>Rule</th><th>Dimension</th><th>Status</th><th>Score</th><th>Passed/Total</th></tr>
                {''.join(f'<tr><td>{cr.rule_name}</td><td>{cr.dimension.value}</td><td class="{'passed' if cr.status == RuleStatus.PASSED else 'failed'}">{cr.status.value}</td><td>{cr.score:.2%}</td><td>{cr.passed_count}/{cr.total_count}</td></tr>' for cr in report.check_results)}
            </table>
        </body>
        </html>
        """
        return html

    def get_alerts(self, report: QualityReport) -> List[Dict[str, Any]]:
        """Get alerts for critical quality issues"""
        alerts = []

        for result in report.check_results:
            if result.status == RuleStatus.FAILED:
                if result.severity == RuleSeverity.CRITICAL:
                    alerts.append({
                        'level': 'CRITICAL',
                        'rule': result.rule_name,
                        'message': f"Critical quality issue: {result.rule_name} failed",
                        'score': result.score,
                        'failed_count': result.failed_count,
                        'dimension': result.dimension.value
                    })
                elif result.severity == RuleSeverity.ERROR and result.score < 0.9:
                    alerts.append({
                        'level': 'ERROR',
                        'rule': result.rule_name,
                        'message': f"Quality error: {result.rule_name} below threshold",
                        'score': result.score,
                        'failed_count': result.failed_count,
                        'dimension': result.dimension.value
                    })

        return alerts


def demo():
    """Demonstrate data quality framework"""
    print("=" * 80)
    print("Advanced Data Quality Framework Demo")
    print("=" * 80)

    # Sample data with quality issues
    sample_data = [
        {'customer_id': 1001, 'name': 'John Doe', 'email': 'john@example.com', 'age': 32},
        {'customer_id': 1002, 'name': 'Jane Smith', 'email': 'jane@example.com', 'age': 28},
        {'customer_id': 1003, 'name': '', 'email': 'invalid-email', 'age': -5},  # Issues
        {'customer_id': 1004, 'name': 'Alice Brown', 'email': 'alice@example.com', 'age': 35},
        {'customer_id': 1001, 'name': 'Duplicate', 'email': 'dup@example.com', 'age': 40},  # Duplicate ID
    ]

    # Initialize framework
    print("\n1. Initializing framework...")
    framework = DataQualityFramework("Customer Data Quality")

    # Create standard rules
    print("\n2. Creating quality rules...")
    framework.create_standard_rules()

    # Define column-specific rules
    column_rules = {
        'customer_id': ['UNIQ_001'],
        'name': ['COMP_001'],
        'email': ['ACC_001', 'COMP_001'],
        'age': ['VAL_001', 'COMP_001']
    }

    # Run quality checks
    print("\n3. Running quality checks...")
    report = framework.run_quality_checks('customer_data', sample_data, column_rules)

    # Get alerts
    print("\n4. Checking for alerts...")
    alerts = framework.get_alerts(report)
    if alerts:
        print(f"  ⚠ {len(alerts)} alerts generated:")
        for alert in alerts:
            print(f"    [{alert['level']}] {alert['message']}")

    # Export report
    print("\n5. Exporting reports...")
    framework.export_report(report, '/tmp/quality_report.json')

    print("\n" + "=" * 80)


if __name__ == "__main__":
    demo()
