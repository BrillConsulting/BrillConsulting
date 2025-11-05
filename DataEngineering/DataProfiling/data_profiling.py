"""
Advanced Data Profiling System
Author: BrillConsulting
Description: Comprehensive automated data profiling with statistical analysis,
quality assessment, and anomaly detection
"""

import json
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from dataclasses import dataclass, asdict, field
from enum import Enum
from collections import Counter
import re


class DataType(Enum):
    """Detected data types"""
    NUMERIC = "numeric"
    STRING = "string"
    BOOLEAN = "boolean"
    DATE = "date"
    TIMESTAMP = "timestamp"
    EMAIL = "email"
    URL = "url"
    JSON = "json"
    UNKNOWN = "unknown"


class ProfileLevel(Enum):
    """Profiling depth levels"""
    BASIC = "basic"
    STANDARD = "standard"
    COMPREHENSIVE = "comprehensive"


@dataclass
class ColumnProfile:
    """Profile statistics for a single column"""
    column_name: str
    data_type: DataType
    total_count: int
    null_count: int = 0
    distinct_count: int = 0
    duplicate_count: int = 0

    # Numeric statistics
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    mean: Optional[float] = None
    median: Optional[float] = None
    std_dev: Optional[float] = None
    percentile_25: Optional[float] = None
    percentile_75: Optional[float] = None

    # String statistics
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    avg_length: Optional[float] = None

    # Patterns and formats
    pattern_analysis: Dict[str, int] = field(default_factory=dict)
    top_values: List[Dict[str, Any]] = field(default_factory=list)

    # Quality metrics
    completeness: float = 0.0
    uniqueness: float = 0.0
    validity: float = 0.0

    # Anomalies
    anomalies: List[Dict[str, Any]] = field(default_factory=list)

    profiled_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['data_type'] = self.data_type.value
        return result


@dataclass
class DatasetProfile:
    """Profile for entire dataset"""
    dataset_name: str
    total_rows: int
    total_columns: int
    column_profiles: Dict[str, ColumnProfile] = field(default_factory=dict)

    # Dataset-level metrics
    overall_completeness: float = 0.0
    memory_usage_mb: float = 0.0
    duplicate_rows: int = 0

    # Relationships
    correlations: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)

    # Schema
    inferred_schema: Dict[str, str] = field(default_factory=dict)

    # Metadata
    profiling_duration_seconds: float = 0.0
    profiled_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['column_profiles'] = {
            name: profile.to_dict()
            for name, profile in self.column_profiles.items()
        }
        return result


class DataProfiler:
    """
    Advanced Data Profiling Engine

    Features:
    - Automatic data type detection
    - Statistical analysis (numeric and categorical)
    - Pattern recognition
    - Data quality metrics
    - Anomaly detection
    - Correlation analysis
    - Schema inference
    """

    def __init__(self, profile_level: ProfileLevel = ProfileLevel.STANDARD):
        """
        Initialize data profiler

        Args:
            profile_level: Depth of profiling to perform
        """
        self.profile_level = profile_level
        self.profiles: Dict[str, DatasetProfile] = {}

    def profile_dataset(self, dataset_name: str, data: List[Dict[str, Any]],
                       sample_size: Optional[int] = None) -> DatasetProfile:
        """
        Profile an entire dataset

        Args:
            dataset_name: Name of the dataset
            data: List of dictionaries representing rows
            sample_size: Optional sample size for large datasets

        Returns:
            Complete dataset profile
        """
        start_time = datetime.now()

        if not data:
            raise ValueError("Dataset is empty")

        # Sample data if needed
        working_data = data[:sample_size] if sample_size else data

        print(f"Profiling dataset: {dataset_name}")
        print(f"  Total rows: {len(working_data)}")
        print(f"  Total columns: {len(data[0].keys())}")

        # Create dataset profile
        profile = DatasetProfile(
            dataset_name=dataset_name,
            total_rows=len(working_data),
            total_columns=len(data[0].keys())
        )

        # Profile each column
        columns = data[0].keys()
        for col_name in columns:
            col_values = [row.get(col_name) for row in working_data]
            col_profile = self._profile_column(col_name, col_values)
            profile.column_profiles[col_name] = col_profile

        # Calculate dataset-level metrics
        profile.overall_completeness = self._calculate_overall_completeness(profile)
        profile.duplicate_rows = self._count_duplicate_rows(working_data)
        profile.inferred_schema = self._infer_schema(profile)

        # Correlation analysis (if comprehensive profiling)
        if self.profile_level == ProfileLevel.COMPREHENSIVE:
            profile.correlations = self._analyze_correlations(working_data, profile)

        # Calculate profiling duration
        end_time = datetime.now()
        profile.profiling_duration_seconds = (end_time - start_time).total_seconds()

        self.profiles[dataset_name] = profile

        print(f"✓ Profiling completed in {profile.profiling_duration_seconds:.2f}s")
        print(f"  Overall completeness: {profile.overall_completeness:.2%}")
        print(f"  Duplicate rows: {profile.duplicate_rows}")

        return profile

    def _profile_column(self, column_name: str, values: List[Any]) -> ColumnProfile:
        """Profile a single column"""
        total_count = len(values)
        null_count = sum(1 for v in values if v is None or v == "")
        non_null_values = [v for v in values if v is not None and v != ""]

        # Detect data type
        data_type = self._detect_data_type(non_null_values)

        # Basic counts
        distinct_values = set(str(v) for v in non_null_values)
        distinct_count = len(distinct_values)
        duplicate_count = total_count - distinct_count

        # Quality metrics
        completeness = (total_count - null_count) / total_count if total_count > 0 else 0
        uniqueness = distinct_count / total_count if total_count > 0 else 0

        profile = ColumnProfile(
            column_name=column_name,
            data_type=data_type,
            total_count=total_count,
            null_count=null_count,
            distinct_count=distinct_count,
            duplicate_count=duplicate_count,
            completeness=completeness,
            uniqueness=uniqueness
        )

        # Type-specific profiling
        if data_type == DataType.NUMERIC:
            self._profile_numeric(profile, non_null_values)

        if data_type in [DataType.STRING, DataType.EMAIL, DataType.URL]:
            self._profile_string(profile, non_null_values)

        # Top values (for all types)
        profile.top_values = self._get_top_values(non_null_values, limit=10)

        # Pattern analysis
        if self.profile_level in [ProfileLevel.STANDARD, ProfileLevel.COMPREHENSIVE]:
            profile.pattern_analysis = self._analyze_patterns(non_null_values)

        # Anomaly detection
        if self.profile_level == ProfileLevel.COMPREHENSIVE:
            profile.anomalies = self._detect_anomalies(profile, non_null_values)

        # Validity check
        profile.validity = self._calculate_validity(profile, non_null_values)

        return profile

    def _detect_data_type(self, values: List[Any]) -> DataType:
        """Detect the data type of a column"""
        if not values:
            return DataType.UNKNOWN

        # Sample a few values for type detection
        sample = values[:min(100, len(values))]

        # Count type matches
        type_scores = {
            DataType.NUMERIC: 0,
            DataType.BOOLEAN: 0,
            DataType.EMAIL: 0,
            DataType.URL: 0,
            DataType.DATE: 0,
            DataType.TIMESTAMP: 0,
            DataType.JSON: 0,
            DataType.STRING: 0
        }

        for value in sample:
            str_value = str(value).strip()

            # Check numeric
            try:
                float(str_value)
                type_scores[DataType.NUMERIC] += 1
                continue
            except (ValueError, TypeError):
                pass

            # Check boolean
            if str_value.lower() in ['true', 'false', 't', 'f', 'yes', 'no', 'y', 'n', '0', '1']:
                type_scores[DataType.BOOLEAN] += 1
                continue

            # Check email
            if re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', str_value):
                type_scores[DataType.EMAIL] += 1
                continue

            # Check URL
            if re.match(r'^https?://', str_value):
                type_scores[DataType.URL] += 1
                continue

            # Check JSON
            if str_value.startswith('{') or str_value.startswith('['):
                try:
                    json.loads(str_value)
                    type_scores[DataType.JSON] += 1
                    continue
                except json.JSONDecodeError:
                    pass

            # Check date patterns
            if re.match(r'^\d{4}-\d{2}-\d{2}', str_value):
                if 'T' in str_value or ' ' in str_value:
                    type_scores[DataType.TIMESTAMP] += 1
                else:
                    type_scores[DataType.DATE] += 1
                continue

            # Default to string
            type_scores[DataType.STRING] += 1

        # Return type with highest score
        detected_type = max(type_scores, key=type_scores.get)
        return detected_type if type_scores[detected_type] > 0 else DataType.STRING

    def _profile_numeric(self, profile: ColumnProfile, values: List[Any]) -> None:
        """Add numeric statistics to profile"""
        try:
            numeric_values = [float(v) for v in values]
            numeric_values.sort()

            n = len(numeric_values)
            profile.min_value = numeric_values[0]
            profile.max_value = numeric_values[-1]
            profile.mean = sum(numeric_values) / n

            # Median
            if n % 2 == 0:
                profile.median = (numeric_values[n//2 - 1] + numeric_values[n//2]) / 2
            else:
                profile.median = numeric_values[n//2]

            # Standard deviation
            variance = sum((x - profile.mean) ** 2 for x in numeric_values) / n
            profile.std_dev = variance ** 0.5

            # Percentiles
            profile.percentile_25 = numeric_values[int(n * 0.25)]
            profile.percentile_75 = numeric_values[int(n * 0.75)]

        except (ValueError, TypeError) as e:
            print(f"Warning: Error profiling numeric column {profile.column_name}: {e}")

    def _profile_string(self, profile: ColumnProfile, values: List[Any]) -> None:
        """Add string statistics to profile"""
        str_values = [str(v) for v in values]
        lengths = [len(s) for s in str_values]

        profile.min_length = min(lengths) if lengths else 0
        profile.max_length = max(lengths) if lengths else 0
        profile.avg_length = sum(lengths) / len(lengths) if lengths else 0

    def _get_top_values(self, values: List[Any], limit: int = 10) -> List[Dict[str, Any]]:
        """Get most frequent values"""
        counter = Counter(str(v) for v in values)
        most_common = counter.most_common(limit)

        total = len(values)
        return [
            {
                'value': value,
                'count': count,
                'percentage': (count / total * 100) if total > 0 else 0
            }
            for value, count in most_common
        ]

    def _analyze_patterns(self, values: List[Any]) -> Dict[str, int]:
        """Analyze common patterns in data"""
        patterns = {
            'alphanumeric': 0,
            'alphabetic': 0,
            'numeric': 0,
            'uppercase': 0,
            'lowercase': 0,
            'mixed_case': 0,
            'contains_special_chars': 0,
            'contains_whitespace': 0
        }

        for value in values[:min(1000, len(values))]:  # Sample for performance
            str_value = str(value)

            if str_value.isalnum():
                patterns['alphanumeric'] += 1
            if str_value.isalpha():
                patterns['alphabetic'] += 1
            if str_value.isnumeric():
                patterns['numeric'] += 1
            if str_value.isupper():
                patterns['uppercase'] += 1
            if str_value.islower():
                patterns['lowercase'] += 1
            if any(c.isupper() for c in str_value) and any(c.islower() for c in str_value):
                patterns['mixed_case'] += 1
            if re.search(r'[^a-zA-Z0-9\s]', str_value):
                patterns['contains_special_chars'] += 1
            if ' ' in str_value:
                patterns['contains_whitespace'] += 1

        return patterns

    def _detect_anomalies(self, profile: ColumnProfile, values: List[Any]) -> List[Dict[str, Any]]:
        """Detect anomalies in data"""
        anomalies = []

        # Numeric outliers (using IQR method)
        if profile.data_type == DataType.NUMERIC and profile.percentile_25 and profile.percentile_75:
            try:
                numeric_values = [float(v) for v in values]
                iqr = profile.percentile_75 - profile.percentile_25
                lower_bound = profile.percentile_25 - (1.5 * iqr)
                upper_bound = profile.percentile_75 + (1.5 * iqr)

                outliers = [v for v in numeric_values if v < lower_bound or v > upper_bound]
                if outliers:
                    anomalies.append({
                        'type': 'numeric_outliers',
                        'count': len(outliers),
                        'percentage': (len(outliers) / len(values) * 100),
                        'description': f'Values outside [{lower_bound:.2f}, {upper_bound:.2f}]'
                    })
            except (ValueError, TypeError):
                pass

        # Length anomalies for strings
        if profile.avg_length and profile.avg_length > 0:
            long_strings = [v for v in values if len(str(v)) > profile.avg_length * 3]
            if long_strings:
                anomalies.append({
                    'type': 'unusually_long_strings',
                    'count': len(long_strings),
                    'percentage': (len(long_strings) / len(values) * 100),
                    'description': f'Strings longer than 3x average length'
                })

        return anomalies

    def _calculate_validity(self, profile: ColumnProfile, values: List[Any]) -> float:
        """Calculate validity score based on data type"""
        if not values:
            return 0.0

        valid_count = 0

        for value in values:
            str_value = str(value).strip()

            if profile.data_type == DataType.EMAIL:
                if re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', str_value):
                    valid_count += 1
            elif profile.data_type == DataType.URL:
                if re.match(r'^https?://', str_value):
                    valid_count += 1
            elif profile.data_type == DataType.NUMERIC:
                try:
                    float(str_value)
                    valid_count += 1
                except ValueError:
                    pass
            else:
                # For other types, assume valid if not empty
                if str_value:
                    valid_count += 1

        return valid_count / len(values) if values else 0.0

    def _calculate_overall_completeness(self, profile: DatasetProfile) -> float:
        """Calculate overall dataset completeness"""
        if not profile.column_profiles:
            return 0.0

        completeness_values = [col.completeness for col in profile.column_profiles.values()]
        return sum(completeness_values) / len(completeness_values)

    def _count_duplicate_rows(self, data: List[Dict[str, Any]]) -> int:
        """Count duplicate rows in dataset"""
        row_strings = [json.dumps(row, sort_keys=True) for row in data]
        return len(row_strings) - len(set(row_strings))

    def _infer_schema(self, profile: DatasetProfile) -> Dict[str, str]:
        """Infer schema from profiled data"""
        schema = {}
        for col_name, col_profile in profile.column_profiles.items():
            # Map to SQL-like types
            type_mapping = {
                DataType.NUMERIC: 'NUMERIC',
                DataType.STRING: 'VARCHAR',
                DataType.BOOLEAN: 'BOOLEAN',
                DataType.DATE: 'DATE',
                DataType.TIMESTAMP: 'TIMESTAMP',
                DataType.EMAIL: 'VARCHAR(255)',
                DataType.URL: 'VARCHAR(2048)',
                DataType.JSON: 'JSON',
                DataType.UNKNOWN: 'VARCHAR'
            }

            base_type = type_mapping.get(col_profile.data_type, 'VARCHAR')

            # Add length for strings if available
            if col_profile.data_type == DataType.STRING and col_profile.max_length:
                base_type = f'VARCHAR({col_profile.max_length})'

            # Add NULL constraint
            nullable = 'NULL' if col_profile.null_count > 0 else 'NOT NULL'

            schema[col_name] = f'{base_type} {nullable}'

        return schema

    def _analyze_correlations(self, data: List[Dict[str, Any]],
                             profile: DatasetProfile) -> Dict[str, List[Dict[str, Any]]]:
        """Analyze correlations between numeric columns"""
        numeric_columns = [
            name for name, prof in profile.column_profiles.items()
            if prof.data_type == DataType.NUMERIC
        ]

        correlations = {}

        for col1 in numeric_columns:
            col1_values = [float(row.get(col1, 0)) for row in data]
            corr_list = []

            for col2 in numeric_columns:
                if col1 != col2:
                    col2_values = [float(row.get(col2, 0)) for row in data]
                    corr = self._pearson_correlation(col1_values, col2_values)

                    if abs(corr) > 0.5:  # Only include significant correlations
                        corr_list.append({
                            'column': col2,
                            'correlation': round(corr, 4),
                            'strength': 'strong' if abs(corr) > 0.7 else 'moderate'
                        })

            if corr_list:
                correlations[col1] = sorted(corr_list, key=lambda x: abs(x['correlation']), reverse=True)

        return correlations

    def _pearson_correlation(self, x: List[float], y: List[float]) -> float:
        """Calculate Pearson correlation coefficient"""
        n = len(x)
        if n == 0:
            return 0.0

        mean_x = sum(x) / n
        mean_y = sum(y) / n

        numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
        denominator_x = sum((x[i] - mean_x) ** 2 for i in range(n)) ** 0.5
        denominator_y = sum((y[i] - mean_y) ** 2 for i in range(n)) ** 0.5

        if denominator_x == 0 or denominator_y == 0:
            return 0.0

        return numerator / (denominator_x * denominator_y)

    def export_profile(self, dataset_name: str, filepath: str) -> None:
        """Export profile to JSON file"""
        if dataset_name not in self.profiles:
            raise ValueError(f"Profile for {dataset_name} not found")

        profile = self.profiles[dataset_name]

        with open(filepath, 'w') as f:
            json.dump(profile.to_dict(), f, indent=2)

        print(f"✓ Profile exported to: {filepath}")

    def generate_report(self, dataset_name: str) -> str:
        """Generate a human-readable profiling report"""
        if dataset_name not in self.profiles:
            raise ValueError(f"Profile for {dataset_name} not found")

        profile = self.profiles[dataset_name]

        report_lines = [
            "=" * 80,
            f"DATA PROFILING REPORT: {profile.dataset_name}",
            "=" * 80,
            f"Generated: {profile.profiled_at}",
            f"Profiling Time: {profile.profiling_duration_seconds:.2f}s",
            "",
            "DATASET OVERVIEW",
            "-" * 80,
            f"Total Rows: {profile.total_rows:,}",
            f"Total Columns: {profile.total_columns}",
            f"Overall Completeness: {profile.overall_completeness:.2%}",
            f"Duplicate Rows: {profile.duplicate_rows:,}",
            "",
            "COLUMN PROFILES",
            "-" * 80
        ]

        for col_name, col_prof in profile.column_profiles.items():
            report_lines.extend([
                f"\n{col_name} ({col_prof.data_type.value})",
                f"  Completeness: {col_prof.completeness:.2%} ({col_prof.null_count:,} nulls)",
                f"  Uniqueness: {col_prof.uniqueness:.2%} ({col_prof.distinct_count:,} distinct)",
                f"  Validity: {col_prof.validity:.2%}"
            ])

            if col_prof.data_type == DataType.NUMERIC:
                report_lines.extend([
                    f"  Range: [{col_prof.min_value}, {col_prof.max_value}]",
                    f"  Mean: {col_prof.mean:.2f}, Median: {col_prof.median:.2f}",
                    f"  Std Dev: {col_prof.std_dev:.2f}"
                ])

            if col_prof.avg_length:
                report_lines.append(
                    f"  Length: min={col_prof.min_length}, max={col_prof.max_length}, avg={col_prof.avg_length:.1f}"
                )

            if col_prof.anomalies:
                report_lines.append(f"  Anomalies: {len(col_prof.anomalies)} types detected")

        if profile.inferred_schema:
            report_lines.extend([
                "",
                "INFERRED SCHEMA",
                "-" * 80
            ])
            for col_name, col_type in profile.inferred_schema.items():
                report_lines.append(f"  {col_name}: {col_type}")

        report_lines.append("=" * 80)

        return "\n".join(report_lines)


def demo():
    """Demonstrate advanced data profiling"""
    print("=" * 80)
    print("Advanced Data Profiling Demo")
    print("=" * 80)

    # Sample data
    sample_data = [
        {
            'customer_id': 1001,
            'name': 'John Doe',
            'email': 'john@example.com',
            'age': 32,
            'city': 'New York',
            'total_purchases': 2500.50,
            'is_active': True
        },
        {
            'customer_id': 1002,
            'name': 'Jane Smith',
            'email': 'jane@example.com',
            'age': 28,
            'city': 'Los Angeles',
            'total_purchases': 1800.75,
            'is_active': True
        },
        {
            'customer_id': 1003,
            'name': 'Bob Johnson',
            'email': 'bob@example.com',
            'age': 45,
            'city': 'Chicago',
            'total_purchases': 3200.00,
            'is_active': False
        },
        {
            'customer_id': 1004,
            'name': 'Alice Brown',
            'email': 'alice@example.com',
            'age': 35,
            'city': 'New York',
            'total_purchases': 4100.25,
            'is_active': True
        },
        {
            'customer_id': 1005,
            'name': 'Charlie Wilson',
            'email': 'charlie@example.com',
            'age': None,  # Missing value
            'city': 'Boston',
            'total_purchases': 950.00,
            'is_active': True
        }
    ]

    # Initialize profiler
    print("\n1. Initializing profiler...")
    profiler = DataProfiler(profile_level=ProfileLevel.COMPREHENSIVE)

    # Profile dataset
    print("\n2. Profiling dataset...")
    profile = profiler.profile_dataset('customer_data', sample_data)

    # Generate and print report
    print("\n3. Generating report...")
    report = profiler.generate_report('customer_data')
    print("\n" + report)

    # Export profile
    print("\n4. Exporting profile...")
    profiler.export_profile('customer_data', '/tmp/customer_profile.json')

    print("\n" + "=" * 80)


if __name__ == "__main__":
    demo()
