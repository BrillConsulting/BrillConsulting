# Azure Monitor Service Integration

Advanced implementation of Azure Monitor with metrics, logs, alerts, diagnostics, and Application Insights integration.

**Author:** BrillConsulting
**Contact:** clientbrill@gmail.com
**LinkedIn:** [brillconsulting](https://www.linkedin.com/in/brillconsulting)

## Overview

Comprehensive Python implementation for Azure Monitor, featuring metrics collection, log analytics, alerting, diagnostic settings, and Application Insights integration. Built for production monitoring and observability.

## Features

### Core Capabilities
- **Metrics Collection**: Custom metrics, aggregations, time-series queries
- **Log Analytics**: KQL queries, custom tables, log ingestion
- **Alert Management**: Metric alerts, log query alerts, action groups
- **Diagnostic Settings**: Log and metric routing to multiple destinations
- **Application Insights**: Request tracking, dependencies, exceptions, custom events

### Advanced Features
- **Real-time Monitoring**: Live metrics and log streaming
- **Custom Dashboards**: Workbook integration
- **Alert Automation**: Automated alert creation and management
- **Multi-destination Routing**: Send data to Log Analytics, Storage, Event Hub
- **KQL Query Execution**: Full Kusto Query Language support

## Installation

```bash
pip install -r requirements.txt
```

## Usage Examples

### Metrics Collection

```python
from azuremonitor import MetricsManager, MetricAggregation

metrics = MetricsManager()

# Define metric
metrics.define_metric(
    "RequestCount",
    "MyApp",
    "Request Count",
    "Count",
    [MetricAggregation.COUNT, MetricAggregation.TOTAL]
)

# Emit metric
metrics.emit_metric("MyApp", "RequestCount", 1)

# Query metrics
stats = metrics.get_metric_statistics("MyApp", "RequestCount", start_time, end_time)
print(f"Average: {stats['average']}")
```

### Log Analytics

```python
from azuremonitor import LogAnalyticsManager, LogLevel

workspace = LogAnalyticsManager("ws-id", "workspace-name")

# Ingest logs
workspace.ingest_log(
    LogLevel.ERROR,
    "Database connection failed",
    "/subscriptions/.../resourceGroups/.../mydb",
    properties={"error_code": "Timeout"}
)

# Query logs
error_logs = workspace.query_logs("* | where Level == 'Error'")
```

### Alert Management

```python
from azuremonitor import AlertManager, AlertSeverity

alerts = AlertManager()

# Create action group
alerts.create_action_group(
    "ops-team",
    "OpsTeam",
    email_receivers=["ops@example.com"]
)

# Create metric alert
alerts.create_metric_alert(
    "high-cpu",
    "CPU exceeds 80%",
    resource_id,
    "Microsoft.Compute/virtualMachines",
    "Percentage CPU",
    ">",
    80.0,
    AlertSeverity.WARNING,
    action_groups=["ops-team"]
)
```

### Application Insights

```python
from azuremonitor import ApplicationInsightsManager

app_insights = ApplicationInsightsManager("instrumentation-key")

# Track request
app_insights.track_request(
    "GET /api/users",
    "https://api.example.com/api/users",
    duration_ms=125.5,
    response_code=200,
    success=True
)

# Track dependency
app_insights.track_dependency(
    "SQL Query",
    "SQL",
    "sqlserver.database.windows.net",
    "SELECT * FROM Users",
    duration_ms=45.3,
    success=True
)

# Track exception
app_insights.track_exception(
    "NullReferenceException",
    "Object reference not set",
    stack_trace
)
```

## Running Demos

```bash
python azuremonitor.py
```

## Best Practices

1. **Metrics**: Use appropriate aggregation types and time granularity
2. **Logs**: Structure logs with consistent properties for better querying
3. **Alerts**: Set appropriate thresholds and action groups
4. **Application Insights**: Track all critical paths and dependencies

## API Reference

See implementation for comprehensive API documentation.

## Support

- Email: clientbrill@gmail.com
- LinkedIn: [brillconsulting](https://www.linkedin.com/in/brillconsulting)

---

**Built with Azure Monitor** | **Brill Consulting © 2024**
