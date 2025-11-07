# Data Quality & Drift Monitoring

Production-grade data quality monitoring, drift detection, and validation using Great Expectations, statistical tests, and ML-based drift detection.

## Features

- **Data Drift Detection** - Covariate, concept, and prediction drift
- **Statistical Tests** - KS test, Chi-square, PSI, Wasserstein distance
- **Great Expectations** - Automated data validation and profiling
- **Real-time Monitoring** - Stream-based drift detection
- **Feature Monitoring** - Track feature distributions over time
- **Model Performance** - Accuracy, precision, recall degradation
- **Alerting** - Threshold-based alerts and notifications
- **Visualization** - Drift dashboards with Evidently AI

## Architecture

```
[Production Data] → [Drift Detector] → [Statistical Tests] → [Alerts]
                          ↓
                    [Reference Data]
                          ↓
                    [Drift Metrics] → [Dashboard]
```

## Usage

### Data Drift Detection

```python
from data_quality import DriftDetector

detector = DriftDetector(
    reference_data=train_data,
    methods=["ks", "psi", "wasserstein"]
)

# Detect drift in production data
drift_report = detector.detect_drift(
    current_data=production_data,
    threshold=0.05
)

print(f"Drift detected: {drift_report.has_drift}")
print(f"Drifted features: {drift_report.drifted_features}")
```

### Great Expectations Integration

```python
from data_quality import GreatExpectationsValidator

validator = GreatExpectationsValidator(
    data_context_root="./gx"
)

# Create expectations
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
        }
    ]
)

# Validate batch
results = validator.validate_batch(
    batch_data=new_data,
    expectation_suite_name="production_suite"
)

print(f"Validation passed: {results.success}")
```

### Continuous Monitoring

```python
from data_quality import DataQualityMonitor

monitor = DataQualityMonitor(
    reference_data=baseline_data,
    drift_threshold=0.05,
    alert_on_drift=True
)

# Start monitoring
monitor.start(
    data_source="kafka://production-data",
    window_size=1000,
    check_interval="5min"
)

# Get monitoring report
report = monitor.get_latest_report()
```

## Drift Detection Methods

### Kolmogorov-Smirnov Test

Statistical test for distribution shift:

```python
from data_quality import KSTest

ks_test = KSTest(threshold=0.05)

# Test for drift
result = ks_test.test(
    reference=reference_feature,
    current=current_feature
)

print(f"P-value: {result.p_value:.4f}")
print(f"Drift: {result.has_drift}")
```

### Population Stability Index (PSI)

Measure distribution shift:

```python
from data_quality import PSI

psi = PSI(threshold=0.2)  # 0.2 = moderate drift

psi_value = psi.calculate(
    reference=reference_dist,
    current=current_dist
)

print(f"PSI: {psi_value:.3f}")
if psi_value > 0.2:
    print("⚠️  Significant drift detected")
```

### Wasserstein Distance

Earth Mover's Distance for drift:

```python
from data_quality import WassersteinDistance

wasserstein = WassersteinDistance(threshold=0.1)

distance = wasserstein.calculate(
    reference=ref_data,
    current=current_data
)

print(f"Wasserstein distance: {distance:.3f}")
```

### Chi-Square Test

For categorical features:

```python
from data_quality import ChiSquareTest

chi2 = ChiSquareTest(threshold=0.05)

result = chi2.test(
    reference=ref_categories,
    current=current_categories
)

print(f"Chi-square statistic: {result.statistic:.2f}")
print(f"Drift: {result.has_drift}")
```

## Feature Monitoring

### Track Feature Statistics

```python
from data_quality import FeatureMonitor

monitor = FeatureMonitor(features=["age", "income", "credit_score"])

# Log baseline statistics
monitor.log_baseline(train_data)

# Monitor production data
for batch in production_batches:
    stats = monitor.track_batch(batch)

    # Check for anomalies
    if stats.has_anomalies:
        print(f"⚠️  Anomalies in: {stats.anomalous_features}")
```

### Feature Distribution Tracking

```python
# Track distributions over time
monitor.track_distributions(
    data=current_batch,
    timestamp="2025-01-15T10:00:00"
)

# Visualize drift
monitor.plot_feature_drift(
    feature="income",
    time_range="last_7_days"
)
```

## Model Performance Monitoring

### Performance Degradation

```python
from data_quality import ModelPerformanceMonitor

perf_monitor = ModelPerformanceMonitor(
    model=production_model,
    baseline_metrics={
        "accuracy": 0.95,
        "precision": 0.94,
        "recall": 0.93
    }
)

# Monitor performance
current_metrics = perf_monitor.evaluate(
    predictions=model_predictions,
    ground_truth=true_labels
)

# Check degradation
if current_metrics.accuracy < 0.90:
    print("⚠️  Model performance degraded")
    perf_monitor.trigger_retraining()
```

### Concept Drift

```python
from data_quality import ConceptDriftDetector

concept_detector = ConceptDriftDetector(
    model=model,
    reference_data=train_data
)

# Detect concept drift
drift = concept_detector.detect(
    current_data=production_data,
    method="DDM"  # Drift Detection Method
)

if drift.detected:
    print(f"⚠️  Concept drift at sample {drift.drift_point}")
```

## Great Expectations Integration

### Creating Expectations

```python
import great_expectations as gx

context = gx.get_context()

# Create expectation suite
suite = context.add_expectation_suite("data_quality_suite")

# Add expectations
validator = context.get_validator(
    batch_request=batch_request,
    expectation_suite_name="data_quality_suite"
)

# Expect column to exist
validator.expect_column_to_exist("user_id")

# Expect values in range
validator.expect_column_values_to_be_between(
    column="age",
    min_value=18,
    max_value=100
)

# Expect no nulls
validator.expect_column_values_to_not_be_null("email")

# Expect values in set
validator.expect_column_values_to_be_in_set(
    column="country",
    value_set=["US", "UK", "CA", "AU"]
)

# Save suite
validator.save_expectation_suite()
```

### Data Profiling

```python
# Auto-generate expectations from data
validator.expect_column_mean_to_be_between(
    column="price",
    auto=True  # Infer from data
)

# Profile entire dataset
profiler = UserConfigurableProfiler(
    profile_dataset=training_data
)

suite = profiler.build_suite()
```

### Validation

```python
# Validate production batch
results = validator.validate(
    batch_request=production_batch
)

# Check results
if not results.success:
    print("❌ Validation failed")
    for result in results.results:
        if not result.success:
            print(f"  - {result.expectation_config.kwargs}")
```

## Alerting

### Configure Alerts

```python
from data_quality import AlertManager

alerts = AlertManager(
    channels=["email", "slack", "pagerduty"]
)

# Configure thresholds
alerts.configure(
    drift_threshold=0.05,
    performance_threshold=0.90,
    data_quality_threshold=0.95
)

# Alert on drift
alerts.on_drift(
    feature="income",
    drift_score=0.08,
    message="Significant drift in income feature"
)
```

### Slack Integration

```python
alerts.configure_slack(
    webhook_url="https://hooks.slack.com/...",
    channel="#ml-monitoring"
)

# Send alert
alerts.send_slack(
    title="Data Drift Alert",
    message="3 features showing drift",
    severity="warning"
)
```

## Visualization

### Evidently AI Integration

```python
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

report = Report(metrics=[
    DataDriftPreset()
])

report.run(
    reference_data=reference_df,
    current_data=current_df
)

# Save HTML report
report.save_html("drift_report.html")

# Get metrics
metrics = report.as_dict()
```

### Custom Dashboards

```python
from data_quality import DriftDashboard

dashboard = DriftDashboard(
    reference_data=baseline,
    port=8050
)

# Start dashboard
dashboard.serve()

# Access at http://localhost:8050
```

## Complete Pipeline

### Production Monitoring Pipeline

```python
from data_quality import DataQualityPipeline

pipeline = DataQualityPipeline(
    reference_data=train_data,
    model=production_model
)

# Configure
pipeline.configure(
    drift_detection=["ks", "psi", "wasserstein"],
    drift_threshold=0.05,
    performance_monitoring=True,
    alerting=["slack", "email"],
    validation_suite="production_suite"
)

# Run monitoring
pipeline.start(
    data_source="kafka://ml-predictions",
    check_interval="5min"
)

# Get report
report = pipeline.get_report(time_range="last_24h")
```

## Use Cases

### ML Model Monitoring

```python
# Monitor model in production
monitor = ModelPerformanceMonitor(model=model)

for batch in production_stream:
    # Detect data drift
    drift = monitor.detect_data_drift(batch)

    # Monitor performance
    perf = monitor.track_performance(batch)

    # Alert if needed
    if drift.has_drift or perf.degraded:
        monitor.send_alert()
```

### Data Pipeline Validation

```python
# Validate data pipeline outputs
validator = GreatExpectationsValidator()

# Run validation on each stage
for stage in ["raw", "cleaned", "features"]:
    results = validator.validate(
        data=pipeline_output[stage],
        suite=f"{stage}_expectations"
    )

    if not results.success:
        raise ValueError(f"Validation failed at {stage}")
```

### Real-time Stream Monitoring

```python
# Monitor streaming data
stream_monitor = StreamDriftDetector(
    reference_window=baseline_window,
    detection_window=1000
)

for event in kafka_stream:
    stream_monitor.add_event(event)

    # Check drift periodically
    if stream_monitor.window_full():
        drift = stream_monitor.check_drift()

        if drift.detected:
            handle_drift(drift)
```

## Technologies

- **Drift Detection**: Evidently AI, Alibi Detect, scikit-multiflow
- **Validation**: Great Expectations, Pandera, Pydantic
- **Monitoring**: Prometheus, Grafana, InfluxDB
- **Streaming**: Kafka, Flink
- **Statistical**: scipy, statsmodels
- **Visualization**: Plotly, Dash, Evidently

## Performance

### Drift Detection Latency

| Method | Computation Time | Memory |
|--------|------------------|--------|
| KS Test | <1ms per feature | O(n) |
| PSI | <1ms per feature | O(bins) |
| Wasserstein | 2-5ms per feature | O(n) |
| Chi-Square | <1ms per feature | O(categories) |

### Monitoring Overhead

- Batch validation: 10-50ms per 1000 rows
- Stream monitoring: <1ms per event
- Dashboard updates: 100ms per refresh

## Best Practices

✅ Establish baseline on representative training data
✅ Use multiple drift detection methods
✅ Set appropriate thresholds (0.05 for statistical tests)
✅ Monitor both features and model performance
✅ Implement gradual alerting (warning → critical)
✅ Version expectation suites with DVC
✅ Automate retraining on sustained drift
✅ Log all drift events for analysis

## Alerting Thresholds

### Drift Severity

- **PSI**: 0.1 (low), 0.2 (moderate), 0.25+ (high)
- **KS test**: p-value < 0.05 (drift detected)
- **Wasserstein**: Domain-specific thresholds

### Performance Degradation

- **Warning**: 2-5% drop from baseline
- **Critical**: >5% drop from baseline

## References

- Great Expectations: https://greatexpectations.io/
- Evidently AI: https://www.evidentlyai.com/
- Alibi Detect: https://docs.seldon.io/projects/alibi-detect/
- Data Drift: https://arxiv.org/abs/2004.03045
- Concept Drift: https://arxiv.org/abs/1010.4784
