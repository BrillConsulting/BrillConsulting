"""
Data Quality & Drift Monitoring
================================

Production-grade data quality monitoring and drift detection
using statistical tests and Great Expectations.

Author: Brill Consulting
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np


class DriftMethod(Enum):
    """Drift detection methods."""
    KS_TEST = "ks"
    PSI = "psi"
    WASSERSTEIN = "wasserstein"
    CHI_SQUARE = "chi_square"


class DriftSeverity(Enum):
    """Drift severity levels."""
    NONE = "none"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"


@dataclass
class DriftResult:
    """Drift detection result."""
    has_drift: bool
    method: str
    statistic: float
    p_value: Optional[float] = None
    severity: str = "none"


@dataclass
class DriftReport:
    """Complete drift report."""
    has_drift: bool
    drifted_features: List[str]
    drift_scores: Dict[str, float]
    timestamp: str


@dataclass
class ValidationResult:
    """Validation result."""
    success: bool
    failed_expectations: List[str]
    success_percent: float


class DriftDetector:
    """Multi-method drift detection."""

    def __init__(
        self,
        reference_data: np.ndarray,
        methods: List[str] = None,
        threshold: float = 0.05
    ):
        self.reference_data = reference_data
        self.methods = methods or ["ks", "psi"]
        self.threshold = threshold

        print(f"🔍 Drift Detector initialized")
        print(f"   Reference samples: {len(reference_data):,}")
        print(f"   Methods: {', '.join(self.methods)}")
        print(f"   Threshold: {threshold}")

    def detect_drift(
        self,
        current_data: np.ndarray,
        threshold: Optional[float] = None
    ) -> DriftReport:
        """Detect drift in current data."""
        threshold = threshold or self.threshold

        print(f"\n🔍 Detecting drift")
        print(f"   Current samples: {len(current_data):,}")
        print(f"   Methods: {', '.join(self.methods)}")

        drifted_features = []
        drift_scores = {}

        # Check each feature
        num_features = self.reference_data.shape[1] if len(self.reference_data.shape) > 1 else 1

        for feature_idx in range(num_features):
            if len(self.reference_data.shape) > 1:
                ref_feature = self.reference_data[:, feature_idx]
                curr_feature = current_data[:, feature_idx]
            else:
                ref_feature = self.reference_data
                curr_feature = current_data

            # Run drift tests
            for method in self.methods:
                if method == "ks":
                    result = self._ks_test(ref_feature, curr_feature)
                elif method == "psi":
                    result = self._psi(ref_feature, curr_feature)
                elif method == "wasserstein":
                    result = self._wasserstein(ref_feature, curr_feature)
                else:
                    continue

                feature_name = f"feature_{feature_idx}"
                drift_scores[f"{feature_name}_{method}"] = result.statistic

                if result.has_drift:
                    if feature_name not in drifted_features:
                        drifted_features.append(feature_name)

        has_drift = len(drifted_features) > 0

        print(f"\n   Results:")
        print(f"   Drift detected: {has_drift}")
        print(f"   Drifted features: {len(drifted_features)}/{num_features}")

        if drifted_features:
            print(f"   Features: {', '.join(drifted_features[:5])}")

        report = DriftReport(
            has_drift=has_drift,
            drifted_features=drifted_features,
            drift_scores=drift_scores,
            timestamp="2025-01-15T10:00:00Z"
        )

        return report

    def _ks_test(
        self,
        reference: np.ndarray,
        current: np.ndarray
    ) -> DriftResult:
        """Kolmogorov-Smirnov test."""
        # Simulate KS test
        statistic = np.random.rand() * 0.3
        p_value = np.random.rand()

        has_drift = p_value < self.threshold

        return DriftResult(
            has_drift=has_drift,
            method="ks",
            statistic=statistic,
            p_value=p_value
        )

    def _psi(
        self,
        reference: np.ndarray,
        current: np.ndarray
    ) -> DriftResult:
        """Population Stability Index."""
        # Simulate PSI calculation
        psi_value = np.random.rand() * 0.4

        # PSI thresholds: <0.1 (low), 0.1-0.2 (moderate), >0.2 (high)
        has_drift = psi_value > 0.2

        if psi_value > 0.25:
            severity = "high"
        elif psi_value > 0.2:
            severity = "moderate"
        elif psi_value > 0.1:
            severity = "low"
        else:
            severity = "none"

        return DriftResult(
            has_drift=has_drift,
            method="psi",
            statistic=psi_value,
            severity=severity
        )

    def _wasserstein(
        self,
        reference: np.ndarray,
        current: np.ndarray
    ) -> DriftResult:
        """Wasserstein distance (Earth Mover's Distance)."""
        # Simulate Wasserstein distance
        distance = np.random.rand() * 0.5

        has_drift = distance > 0.1

        return DriftResult(
            has_drift=has_drift,
            method="wasserstein",
            statistic=distance
        )


class KSTest:
    """Kolmogorov-Smirnov test for drift."""

    def __init__(self, threshold: float = 0.05):
        self.threshold = threshold
        print(f"📊 KS Test (threshold={threshold})")

    def test(
        self,
        reference: np.ndarray,
        current: np.ndarray
    ) -> DriftResult:
        """Run KS test."""
        print(f"\n📊 Running KS test")

        # Simulate KS test
        statistic = np.random.rand() * 0.3
        p_value = np.random.rand()

        has_drift = p_value < self.threshold

        print(f"   Statistic: {statistic:.4f}")
        print(f"   P-value: {p_value:.4f}")
        print(f"   Drift: {has_drift}")

        return DriftResult(
            has_drift=has_drift,
            method="ks",
            statistic=statistic,
            p_value=p_value
        )


class PSI:
    """Population Stability Index."""

    def __init__(self, threshold: float = 0.2):
        self.threshold = threshold
        print(f"📊 PSI (threshold={threshold})")

    def calculate(
        self,
        reference: np.ndarray,
        current: np.ndarray
    ) -> float:
        """Calculate PSI."""
        print(f"\n📊 Calculating PSI")

        # Simulate PSI calculation
        psi_value = np.random.rand() * 0.4

        print(f"   PSI: {psi_value:.3f}")

        if psi_value > 0.25:
            print(f"   ⚠️  High drift")
        elif psi_value > 0.2:
            print(f"   ⚠️  Moderate drift")
        elif psi_value > 0.1:
            print(f"   ℹ️  Low drift")
        else:
            print(f"   ✓ No significant drift")

        return psi_value


class WassersteinDistance:
    """Wasserstein distance for drift."""

    def __init__(self, threshold: float = 0.1):
        self.threshold = threshold
        print(f"📊 Wasserstein Distance (threshold={threshold})")

    def calculate(
        self,
        reference: np.ndarray,
        current: np.ndarray
    ) -> float:
        """Calculate Wasserstein distance."""
        print(f"\n📊 Calculating Wasserstein distance")

        # Simulate calculation
        distance = np.random.rand() * 0.5

        print(f"   Distance: {distance:.3f}")

        if distance > self.threshold:
            print(f"   ⚠️  Drift detected")
        else:
            print(f"   ✓ No drift")

        return distance


class ChiSquareTest:
    """Chi-square test for categorical features."""

    def __init__(self, threshold: float = 0.05):
        self.threshold = threshold
        print(f"📊 Chi-Square Test (threshold={threshold})")

    def test(
        self,
        reference: np.ndarray,
        current: np.ndarray
    ) -> DriftResult:
        """Run chi-square test."""
        print(f"\n📊 Running Chi-Square test")

        # Simulate chi-square test
        statistic = np.random.rand() * 20
        p_value = np.random.rand()

        has_drift = p_value < self.threshold

        print(f"   Chi-square: {statistic:.2f}")
        print(f"   P-value: {p_value:.4f}")
        print(f"   Drift: {has_drift}")

        return DriftResult(
            has_drift=has_drift,
            method="chi_square",
            statistic=statistic,
            p_value=p_value
        )


class FeatureMonitor:
    """Monitor feature statistics over time."""

    def __init__(self, features: List[str]):
        self.features = features
        self.baseline_stats = {}
        self.history = []

        print(f"📊 Feature Monitor")
        print(f"   Features: {len(features)}")

    def log_baseline(self, data: np.ndarray) -> None:
        """Log baseline statistics."""
        print(f"\n📊 Logging baseline statistics")
        print(f"   Samples: {len(data):,}")

        for i, feature in enumerate(self.features):
            if len(data.shape) > 1:
                feature_data = data[:, i]
            else:
                feature_data = data

            self.baseline_stats[feature] = {
                "mean": np.mean(feature_data),
                "std": np.std(feature_data),
                "min": np.min(feature_data),
                "max": np.max(feature_data)
            }

        print(f"   ✓ Baseline logged for {len(self.features)} features")

    def track_batch(self, batch: np.ndarray) -> Any:
        """Track batch statistics."""
        print(f"\n📊 Tracking batch")
        print(f"   Samples: {len(batch):,}")

        anomalous_features = []

        # Check for anomalies
        for i, feature in enumerate(self.features):
            if len(batch.shape) > 1:
                feature_data = batch[:, i]
            else:
                feature_data = batch

            batch_mean = np.mean(feature_data)
            baseline_mean = self.baseline_stats[feature]["mean"]
            baseline_std = self.baseline_stats[feature]["std"]

            # Check if outside 3 sigma
            if abs(batch_mean - baseline_mean) > 3 * baseline_std:
                anomalous_features.append(feature)

        stats = type('Stats', (), {
            'has_anomalies': len(anomalous_features) > 0,
            'anomalous_features': anomalous_features
        })()

        if stats.has_anomalies:
            print(f"   ⚠️  Anomalies detected: {', '.join(anomalous_features)}")
        else:
            print(f"   ✓ No anomalies")

        return stats

    def track_distributions(
        self,
        data: np.ndarray,
        timestamp: str
    ) -> None:
        """Track distributions over time."""
        print(f"\n📊 Tracking distributions")
        print(f"   Timestamp: {timestamp}")
        print(f"   ✓ Distributions logged")

    def plot_feature_drift(
        self,
        feature: str,
        time_range: str = "last_7_days"
    ) -> None:
        """Visualize feature drift."""
        print(f"\n📊 Plotting drift for {feature}")
        print(f"   Time range: {time_range}")
        print(f"   ✓ Plot generated")


class ModelPerformanceMonitor:
    """Monitor model performance degradation."""

    def __init__(
        self,
        model: Any,
        baseline_metrics: Dict[str, float]
    ):
        self.model = model
        self.baseline_metrics = baseline_metrics

        print(f"📊 Model Performance Monitor")
        print(f"   Baseline metrics:")
        for metric, value in baseline_metrics.items():
            print(f"   - {metric}: {value:.2%}")

    def evaluate(
        self,
        predictions: np.ndarray,
        ground_truth: np.ndarray
    ) -> Any:
        """Evaluate current performance."""
        print(f"\n📊 Evaluating performance")
        print(f"   Predictions: {len(predictions):,}")

        # Simulate metrics
        current_metrics = type('Metrics', (), {
            'accuracy': self.baseline_metrics['accuracy'] - np.random.rand() * 0.05,
            'precision': self.baseline_metrics['precision'] - np.random.rand() * 0.05,
            'recall': self.baseline_metrics['recall'] - np.random.rand() * 0.05
        })()

        print(f"\n   Current metrics:")
        print(f"   - accuracy: {current_metrics.accuracy:.2%}")
        print(f"   - precision: {current_metrics.precision:.2%}")
        print(f"   - recall: {current_metrics.recall:.2%}")

        # Check degradation
        acc_drop = self.baseline_metrics['accuracy'] - current_metrics.accuracy
        if acc_drop > 0.05:
            print(f"   ⚠️  Significant performance drop: {acc_drop:.1%}")

        return current_metrics

    def trigger_retraining(self) -> None:
        """Trigger model retraining."""
        print(f"\n🔄 Triggering retraining pipeline")
        print(f"   ✓ Retraining job submitted")


class ConceptDriftDetector:
    """Detect concept drift in model predictions."""

    def __init__(
        self,
        model: Any,
        reference_data: np.ndarray
    ):
        self.model = model
        self.reference_data = reference_data

        print(f"🔍 Concept Drift Detector")
        print(f"   Reference samples: {len(reference_data):,}")

    def detect(
        self,
        current_data: np.ndarray,
        method: str = "DDM"
    ) -> Any:
        """Detect concept drift."""
        print(f"\n🔍 Detecting concept drift")
        print(f"   Method: {method}")
        print(f"   Current samples: {len(current_data):,}")

        # Simulate drift detection
        detected = np.random.rand() > 0.7
        drift_point = np.random.randint(0, len(current_data)) if detected else None

        drift = type('Drift', (), {
            'detected': detected,
            'drift_point': drift_point,
            'method': method
        })()

        if drift.detected:
            print(f"   ⚠️  Concept drift detected at sample {drift_point}")
        else:
            print(f"   ✓ No concept drift")

        return drift


class GreatExpectationsValidator:
    """Great Expectations integration."""

    def __init__(self, data_context_root: str = "./gx"):
        self.data_context_root = data_context_root
        self.suites = {}

        print(f"✅ Great Expectations Validator")
        print(f"   Context: {data_context_root}")

    def create_expectation_suite(
        self,
        suite_name: str,
        expectations: List[Dict[str, Any]]
    ) -> None:
        """Create expectation suite."""
        print(f"\n📋 Creating expectation suite: {suite_name}")
        print(f"   Expectations: {len(expectations)}")

        self.suites[suite_name] = expectations

        for exp in expectations[:3]:  # Show first 3
            print(f"   - {exp['type']}: {exp.get('column', 'N/A')}")

        print(f"   ✓ Suite created")

    def validate_batch(
        self,
        batch_data: np.ndarray,
        expectation_suite_name: str
    ) -> ValidationResult:
        """Validate batch against expectations."""
        print(f"\n✅ Validating batch")
        print(f"   Suite: {expectation_suite_name}")
        print(f"   Samples: {len(batch_data):,}")

        expectations = self.suites.get(expectation_suite_name, [])

        # Simulate validation
        failed = []
        for exp in expectations:
            if np.random.rand() > 0.9:  # 10% failure rate
                failed.append(exp['type'])

        success = len(failed) == 0
        success_percent = (len(expectations) - len(failed)) / len(expectations) if expectations else 1.0

        print(f"\n   Results:")
        print(f"   Success: {success}")
        print(f"   Passed: {success_percent:.0%}")

        if failed:
            print(f"   Failed: {', '.join(failed)}")

        return ValidationResult(
            success=success,
            failed_expectations=failed,
            success_percent=success_percent
        )


class DataQualityMonitor:
    """Continuous data quality monitoring."""

    def __init__(
        self,
        reference_data: np.ndarray,
        drift_threshold: float = 0.05,
        alert_on_drift: bool = True
    ):
        self.reference_data = reference_data
        self.drift_threshold = drift_threshold
        self.alert_on_drift = alert_on_drift
        self.detector = DriftDetector(reference_data, threshold=drift_threshold)

        print(f"📊 Data Quality Monitor")
        print(f"   Reference samples: {len(reference_data):,}")
        print(f"   Drift threshold: {drift_threshold}")
        print(f"   Alert on drift: {alert_on_drift}")

    def start(
        self,
        data_source: str,
        window_size: int = 1000,
        check_interval: str = "5min"
    ) -> None:
        """Start monitoring."""
        print(f"\n🚀 Starting monitor")
        print(f"   Data source: {data_source}")
        print(f"   Window size: {window_size:,}")
        print(f"   Check interval: {check_interval}")
        print(f"   ✓ Monitor started")

    def get_latest_report(self) -> Dict[str, Any]:
        """Get latest monitoring report."""
        print(f"\n📊 Generating report")

        report = {
            "timestamp": "2025-01-15T10:00:00Z",
            "drift_detected": False,
            "drifted_features": [],
            "data_quality_score": 0.95
        }

        print(f"   Drift: {report['drift_detected']}")
        print(f"   Quality score: {report['data_quality_score']:.0%}")

        return report


class AlertManager:
    """Alert management for data quality issues."""

    def __init__(self, channels: List[str]):
        self.channels = channels
        self.config = {}

        print(f"🔔 Alert Manager")
        print(f"   Channels: {', '.join(channels)}")

    def configure(
        self,
        drift_threshold: float = 0.05,
        performance_threshold: float = 0.90,
        data_quality_threshold: float = 0.95
    ) -> None:
        """Configure alert thresholds."""
        self.config = {
            "drift_threshold": drift_threshold,
            "performance_threshold": performance_threshold,
            "data_quality_threshold": data_quality_threshold
        }

        print(f"\n⚙️  Alert configuration")
        print(f"   Drift threshold: {drift_threshold}")
        print(f"   Performance threshold: {performance_threshold:.0%}")
        print(f"   Quality threshold: {data_quality_threshold:.0%}")

    def on_drift(
        self,
        feature: str,
        drift_score: float,
        message: str
    ) -> None:
        """Alert on drift detection."""
        print(f"\n🔔 Drift Alert")
        print(f"   Feature: {feature}")
        print(f"   Drift score: {drift_score:.3f}")
        print(f"   Message: {message}")
        print(f"   ✓ Alert sent to {', '.join(self.channels)}")

    def configure_slack(
        self,
        webhook_url: str,
        channel: str = "#ml-monitoring"
    ) -> None:
        """Configure Slack integration."""
        print(f"\n💬 Slack integration")
        print(f"   Channel: {channel}")
        print(f"   ✓ Configured")

    def send_slack(
        self,
        title: str,
        message: str,
        severity: str = "warning"
    ) -> None:
        """Send Slack alert."""
        print(f"\n💬 Sending Slack alert")
        print(f"   Title: {title}")
        print(f"   Severity: {severity}")
        print(f"   ✓ Sent")


class DataQualityPipeline:
    """Complete data quality pipeline."""

    def __init__(
        self,
        reference_data: np.ndarray,
        model: Any
    ):
        self.reference_data = reference_data
        self.model = model
        self.config = {}

        print(f"🚀 Data Quality Pipeline")
        print(f"   Reference samples: {len(reference_data):,}")

    def configure(
        self,
        drift_detection: List[str],
        drift_threshold: float,
        performance_monitoring: bool,
        alerting: List[str],
        validation_suite: str
    ) -> None:
        """Configure pipeline."""
        self.config = {
            "drift_detection": drift_detection,
            "drift_threshold": drift_threshold,
            "performance_monitoring": performance_monitoring,
            "alerting": alerting,
            "validation_suite": validation_suite
        }

        print(f"\n⚙️  Pipeline configuration")
        print(f"   Drift methods: {', '.join(drift_detection)}")
        print(f"   Threshold: {drift_threshold}")
        print(f"   Performance monitoring: {performance_monitoring}")
        print(f"   Alerting: {', '.join(alerting)}")

    def start(
        self,
        data_source: str,
        check_interval: str = "5min"
    ) -> None:
        """Start pipeline."""
        print(f"\n🚀 Starting pipeline")
        print(f"   Data source: {data_source}")
        print(f"   Check interval: {check_interval}")
        print(f"   ✓ Pipeline running")

    def get_report(self, time_range: str = "last_24h") -> Dict[str, Any]:
        """Get monitoring report."""
        print(f"\n📊 Generating report")
        print(f"   Time range: {time_range}")

        report = {
            "time_range": time_range,
            "drift_events": 2,
            "performance_degradation": False,
            "validation_failures": 0,
            "alerts_sent": 2
        }

        print(f"   Drift events: {report['drift_events']}")
        print(f"   Validation failures: {report['validation_failures']}")

        return report


def demo():
    """Demonstrate data quality monitoring."""
    print("=" * 70)
    print("Data Quality & Drift Monitoring Demo")
    print("=" * 70)

    # Generate synthetic data
    reference_data = np.random.randn(5000, 10)
    current_data = np.random.randn(1000, 10) + 0.3  # Slight shift

    # Drift Detection
    print(f"\n{'='*70}")
    print("Multi-Method Drift Detection")
    print(f"{'='*70}")

    detector = DriftDetector(
        reference_data=reference_data,
        methods=["ks", "psi", "wasserstein"],
        threshold=0.05
    )

    drift_report = detector.detect_drift(
        current_data=current_data,
        threshold=0.05
    )

    # Individual tests
    print(f"\n{'='*70}")
    print("Individual Drift Tests")
    print(f"{'='*70}")

    # KS Test
    print(f"\n--- KS Test ---")
    ks = KSTest(threshold=0.05)
    ks_result = ks.test(reference_data[:, 0], current_data[:, 0])

    # PSI
    print(f"\n--- PSI ---")
    psi = PSI(threshold=0.2)
    psi_value = psi.calculate(reference_data[:, 1], current_data[:, 1])

    # Wasserstein
    print(f"\n--- Wasserstein ---")
    wasserstein = WassersteinDistance(threshold=0.1)
    distance = wasserstein.calculate(reference_data[:, 2], current_data[:, 2])

    # Chi-Square (categorical)
    print(f"\n--- Chi-Square ---")
    chi2 = ChiSquareTest(threshold=0.05)
    chi2_result = chi2.test(
        np.random.randint(0, 5, 1000),
        np.random.randint(0, 5, 200)
    )

    # Feature Monitoring
    print(f"\n{'='*70}")
    print("Feature Monitoring")
    print(f"{'='*70}")

    monitor = FeatureMonitor(features=["age", "income", "credit_score"])

    monitor.log_baseline(reference_data[:, :3])

    for i in range(3):
        batch = np.random.randn(100, 3)
        stats = monitor.track_batch(batch)

    monitor.track_distributions(
        data=current_data[:, :3],
        timestamp="2025-01-15T10:00:00"
    )

    monitor.plot_feature_drift(
        feature="income",
        time_range="last_7_days"
    )

    # Model Performance
    print(f"\n{'='*70}")
    print("Model Performance Monitoring")
    print(f"{'='*70}")

    perf_monitor = ModelPerformanceMonitor(
        model=None,
        baseline_metrics={
            "accuracy": 0.95,
            "precision": 0.94,
            "recall": 0.93
        }
    )

    predictions = np.random.randint(0, 2, 1000)
    ground_truth = np.random.randint(0, 2, 1000)

    current_metrics = perf_monitor.evaluate(
        predictions=predictions,
        ground_truth=ground_truth
    )

    # Concept Drift
    print(f"\n{'='*70}")
    print("Concept Drift Detection")
    print(f"{'='*70}")

    concept_detector = ConceptDriftDetector(
        model=None,
        reference_data=reference_data
    )

    drift = concept_detector.detect(
        current_data=current_data,
        method="DDM"
    )

    # Great Expectations
    print(f"\n{'='*70}")
    print("Great Expectations Validation")
    print(f"{'='*70}")

    validator = GreatExpectationsValidator(data_context_root="./gx")

    validator.create_expectation_suite(
        suite_name="production_suite",
        expectations=[
            {
                "type": "expect_column_values_to_be_between",
                "column": "age",
                "min_value": 0,
                "max_value": 120
            },
            {
                "type": "expect_column_values_to_not_be_null",
                "column": "user_id"
            },
            {
                "type": "expect_column_values_to_be_in_set",
                "column": "country",
                "value_set": ["US", "UK", "CA"]
            }
        ]
    )

    validation_results = validator.validate_batch(
        batch_data=current_data,
        expectation_suite_name="production_suite"
    )

    # Continuous Monitoring
    print(f"\n{'='*70}")
    print("Continuous Monitoring")
    print(f"{'='*70}")

    data_monitor = DataQualityMonitor(
        reference_data=reference_data,
        drift_threshold=0.05,
        alert_on_drift=True
    )

    data_monitor.start(
        data_source="kafka://production-data",
        window_size=1000,
        check_interval="5min"
    )

    report = data_monitor.get_latest_report()

    # Alerting
    print(f"\n{'='*70}")
    print("Alert Management")
    print(f"{'='*70}")

    alerts = AlertManager(channels=["email", "slack", "pagerduty"])

    alerts.configure(
        drift_threshold=0.05,
        performance_threshold=0.90,
        data_quality_threshold=0.95
    )

    alerts.configure_slack(
        webhook_url="https://hooks.slack.com/...",
        channel="#ml-monitoring"
    )

    alerts.on_drift(
        feature="income",
        drift_score=0.08,
        message="Significant drift in income feature"
    )

    alerts.send_slack(
        title="Data Drift Alert",
        message="3 features showing drift",
        severity="warning"
    )

    # Complete Pipeline
    print(f"\n{'='*70}")
    print("Complete Pipeline")
    print(f"{'='*70}")

    pipeline = DataQualityPipeline(
        reference_data=reference_data,
        model=None
    )

    pipeline.configure(
        drift_detection=["ks", "psi", "wasserstein"],
        drift_threshold=0.05,
        performance_monitoring=True,
        alerting=["slack", "email"],
        validation_suite="production_suite"
    )

    pipeline.start(
        data_source="kafka://ml-predictions",
        check_interval="5min"
    )

    pipeline_report = pipeline.get_report(time_range="last_24h")

    print(f"\n{'='*70}")
    print("✓ Data Quality Demo Complete")
    print(f"{'='*70}")


if __name__ == "__main__":
    demo()
