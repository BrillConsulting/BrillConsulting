# Data Observability
Comprehensive data pipeline monitoring and quality tracking

## Overview

A production-grade data observability framework that provides complete visibility into data pipeline health, quality, and performance. Features automated monitoring, anomaly detection, SLA tracking, and alerting to ensure data reliability and quick incident response.

## Features

### Core Capabilities
- **Pipeline Registration**: Register pipelines with custom SLAs and configurations
- **Run Tracking**: Record detailed execution metrics for each pipeline run
- **Data Freshness Monitoring**: Track data age against SLA requirements
- **Quality Metrics**: Calculate success rates, error rates, and performance stats
- **Anomaly Detection**: Identify unusual patterns using statistical methods
- **SLA Compliance**: Automated checking against defined service levels
- **Alert Management**: Create and track alerts for pipeline issues
- **Incident Tracking**: Manage incidents with status and resolution tracking

### Advanced Features
- **Multi-Window Analysis**: Support for various time windows (24h, 7d, 30d)
- **Statistical Anomaly Detection**: Z-score based detection with configurable thresholds
- **Comprehensive Monitoring**: All-in-one pipeline health checks
- **Alert Severity Levels**: Critical, warning, and info classifications
- **State Management**: Track pipeline status (healthy, degraded, failed)
- **Health Reporting**: Generate system-wide health reports
- **Performance Metrics**: Duration tracking and optimization insights

## Installation

```bash
# Clone the repository
git clone https://github.com/BrillConsulting/DataArchitecture.git
cd DataArchitecture/DataObservability

# Install dependencies
pip install pandas

# Run the implementation
python data_observability.py
```

## Usage Examples

### Register Pipeline

```python
from data_observability import DataObservability

# Initialize observability system
obs = DataObservability()

# Register pipeline with SLA
obs.register_pipeline(
    pipeline_id="sales_etl",
    name="Sales ETL Pipeline",
    owner="data_team",
    sla={
        "freshness_minutes": 60,
        "success_rate_percent": 95,
        "max_error_rate_percent": 1.0
    }
)
```

### Record Pipeline Runs

```python
# Record successful run
obs.record_pipeline_run(
    pipeline_id="sales_etl",
    status="success",
    duration_seconds=45.2,
    rows_processed=100000,
    errors=12
)

# Record failed run
obs.record_pipeline_run(
    pipeline_id="sales_etl",
    status="failed",
    duration_seconds=120.5,
    rows_processed=50000,
    errors=500
)
```

### Check Data Freshness

```python
# Check if data meets freshness SLA
freshness = obs.check_data_freshness("sales_etl")

print(f"Data age: {freshness['age_minutes']:.1f} minutes")
print(f"SLA: {freshness['sla_minutes']} minutes")
print(f"Status: {freshness['status']}")  # fresh or stale

if not freshness['is_fresh']:
    print(f"⚠ Data is stale!")
```

### Calculate Quality Metrics

```python
# Get metrics for last 24 hours
metrics = obs.calculate_quality_metrics("sales_etl", window_hours=24)

print(f"Success Rate: {metrics['success_rate']:.1f}%")
print(f"Average Duration: {metrics['avg_duration_seconds']:.2f}s")
print(f"Error Rate: {metrics['error_rate']:.4f}%")
print(f"Total Runs: {metrics['total_runs']}")
print(f"Total Rows: {metrics['total_rows_processed']:,}")
```

### Detect Anomalies

```python
# Detect anomalies in duration
anomalies = obs.detect_anomalies(
    pipeline_id="sales_etl",
    metric="duration_seconds",
    threshold=3.0  # 3 standard deviations
)

if anomalies['anomalies_count'] > 0:
    print(f"⚠ Found {anomalies['anomalies_count']} anomalies")
    for anomaly in anomalies['anomalies'][:5]:
        print(f"  Value: {anomaly['value']:.2f}, Z-score: {anomaly['z_score']:.2f}")
```

### SLA Compliance Check

```python
# Check all SLA requirements
compliance = obs.check_sla_compliance("sales_etl")

print(f"SLA Compliance: {compliance['compliant']}")

for check in compliance['checks']:
    status = "✓" if check['passed'] else "✗"
    print(f"  {status} {check['check']}: {check.get('actual', 'N/A')}")
```

### Create Alerts

```python
# Create alert for high error rate
alert = obs.create_alert(
    pipeline_id="sales_etl",
    severity="critical",
    message="Error rate exceeds threshold",
    details={
        "error_rate": 2.5,
        "threshold": 1.0
    }
)

print(f"Alert created: {alert['alert_id']}")
```

### Comprehensive Monitoring

```python
# Run all monitoring checks at once
monitoring = obs.monitor_pipeline("sales_etl")

print("Monitoring Results:")
print(f"  Freshness: {monitoring['checks']['freshness']['status']}")
print(f"  Quality: {monitoring['checks']['quality_metrics']['success_rate']:.1f}%")
print(f"  Anomalies: {monitoring['checks']['anomalies']['anomalies_count']}")
print(f"  SLA: {'PASS' if monitoring['checks']['sla_compliance']['compliant'] else 'FAIL'}")
```

### Incident Management

```python
# Create incident for critical issue
incident = obs.create_incident(
    pipeline_id="sales_etl",
    title="High error rate detected",
    severity="high",
    description="Last run had 500 errors, significantly above normal baseline"
)

print(f"Incident: {incident['incident_id']}")
```

### Health Reporting

```python
# Generate system-wide health report
report = obs.generate_health_report()

print(f"Total Pipelines: {report['summary']['total_pipelines']}")
print(f"Healthy: {report['summary']['healthy_pipelines']}")
print(f"Degraded: {report['summary']['degraded_pipelines']}")
print(f"Open Alerts: {report['summary']['open_alerts']}")
print(f"Open Incidents: {report['summary']['open_incidents']}")

print("\nAlerts by Severity:")
for severity, count in report['alerts_by_severity'].items():
    if count > 0:
        print(f"  {severity}: {count}")
```

## Demo Instructions

Run the included demonstration:

```bash
python data_observability.py
```

The demo showcases:
1. Pipeline registration with SLAs
2. Recording multiple pipeline runs
3. Data freshness monitoring
4. Quality metrics calculation
5. Anomaly detection in performance
6. SLA compliance checking
7. Alert creation for violations
8. Comprehensive pipeline monitoring
9. Incident management
10. System-wide health reporting

## Key Concepts

### Data Freshness

Measures how recent your data is:
- Compare last update time against SLA
- Alert when data becomes stale
- Track maximum acceptable age
- Monitor real-time pipelines closely

### Quality Metrics

Track pipeline reliability:
- **Success Rate**: Percentage of successful runs
- **Error Rate**: Errors per row processed
- **Duration**: Processing time statistics
- **Throughput**: Rows processed per run

### Anomaly Detection

Identify unusual behavior:
- Statistical analysis (Z-scores, standard deviation)
- Historical baseline comparison
- Configurable sensitivity thresholds
- Early warning for degradation

### SLA Compliance

Ensure service level agreements:
- **Freshness SLA**: Data must be updated within X minutes
- **Success Rate SLA**: Pipeline must succeed Y% of the time
- **Error Rate SLA**: Maximum acceptable error rate
- **Duration SLA**: Processing time limits (optional)

## Architecture

```
┌──────────────────────────────────────────┐
│      Data Observability Platform         │
│                                          │
│  ┌─────────────────────────────────┐    │
│  │   Pipeline Monitoring           │    │
│  │   - Run Tracking                │    │
│  │   - Freshness Checks            │    │
│  │   - Quality Metrics             │    │
│  └──────────────┬──────────────────┘    │
│                 │                        │
│                 ▼                        │
│  ┌─────────────────────────────────┐    │
│  │   Anomaly Detection Engine      │    │
│  │   - Statistical Analysis        │    │
│  │   - Pattern Recognition         │    │
│  │   - Threshold Alerts            │    │
│  └──────────────┬──────────────────┘    │
│                 │                        │
│                 ▼                        │
│  ┌─────────────────────────────────┐    │
│  │   Alert & Incident Management   │    │
│  │   - Alert Generation            │    │
│  │   - Severity Classification     │    │
│  │   - Incident Tracking           │    │
│  └──────────────┬──────────────────┘    │
│                 │                        │
│                 ▼                        │
│  ┌─────────────────────────────────┐    │
│  │   Health Reporting              │    │
│  │   - System Overview             │    │
│  │   - Trend Analysis              │    │
│  │   - SLA Compliance              │    │
│  └─────────────────────────────────┘    │
└──────────────────────────────────────────┘
```

## Use Cases

- **Production Monitoring**: Track data pipeline health 24/7
- **SLA Management**: Ensure commitments to data consumers
- **Incident Response**: Quickly identify and resolve issues
- **Performance Optimization**: Find and fix bottlenecks
- **Data Quality Assurance**: Maintain high data standards
- **Capacity Planning**: Analyze trends for resource planning
- **Compliance Reporting**: Document system reliability

## Best Practices

- Define realistic SLAs based on business needs
- Set up alerts for critical violations only
- Review anomaly detection thresholds periodically
- Maintain runbook for common incidents
- Analyze trends weekly for proactive improvements
- Document pipeline dependencies
- Implement gradual rollout for changes
- Monitor during and after deployments

## Integration

Integrate with:
- Data orchestration tools (Airflow, Prefect)
- Alert platforms (PagerDuty, Opsgenie)
- Monitoring dashboards (Grafana, Datadog)
- Incident management (Jira, ServiceNow)
- Data catalogs for metadata enrichment

## Author

**Brill Consulting**
- Email: clientbrill@gmail.com
- LinkedIn: [linkedin.com/in/brillconsulting](https://linkedin.com/in/brillconsulting)
- Specialization: Data Architecture & Engineering Solutions
