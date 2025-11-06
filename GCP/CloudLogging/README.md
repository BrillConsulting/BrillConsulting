# Cloud Logging - Centralized Logging and Monitoring

Comprehensive Cloud Logging implementation for structured logging, log analytics, and monitoring integration.

## Features

### Log Writing
- **Structured Logging**: JSON-formatted logs with severity levels
- **Batch Logging**: Efficient bulk log writing for high throughput
- **Custom Severity**: Support for DEBUG, INFO, WARNING, ERROR, CRITICAL
- **Log Metadata**: Resource labels and log entry labels

### Log Querying
- **Time-Based Queries**: Query logs from last N hours/days
- **Severity Filtering**: Filter by log severity level
- **Resource Filtering**: Query logs by resource type
- **Advanced Filters**: Complex filter expressions

### Log-Based Metrics
- **Counter Metrics**: Count specific log patterns
- **Distribution Metrics**: Track value distributions
- **Custom Metrics**: Create metrics from log data
- **Metric Alerts**: Alert on metric thresholds

### Log Analytics
- **Aggregation Queries**: GROUP BY and COUNT operations
- **Time-Series Analysis**: Analyze logs over time
- **Pattern Detection**: Identify common log patterns
- **Error Analysis**: Automated error trend detection

### Log Sinks
- **BigQuery Sink**: Export logs to BigQuery for long-term analysis
- **Cloud Storage Sink**: Archive logs to Cloud Storage
- **Pub/Sub Sink**: Stream logs to Pub/Sub topics
- **Custom Sinks**: Route logs to external systems

### Alerting
- **Alert Policies**: Create alerts based on log patterns
- **Notification Channels**: Email, SMS, Slack, PagerDuty
- **Threshold-Based Alerts**: Alert when metrics exceed thresholds
- **Multi-Condition Alerts**: Complex alerting logic

## Usage Example

```python
from cloudlogging import CloudLoggingManager

# Initialize manager
mgr = CloudLoggingManager(project_id='my-gcp-project')

# Write structured log
mgr.writer.write_structured_log({
    'message': 'User login successful',
    'user_id': '123',
    'ip_address': '192.168.1.1'
}, severity='INFO')

# Batch write logs
mgr.writer.write_batch_logs([
    {'event': 'login', 'user': 'user1'},
    {'event': 'logout', 'user': 'user2'}
])

# Query error logs
errors = mgr.reader.query_error_logs(hours=24)

# Create log-based metric
metric = mgr.metrics.create_log_based_metric({
    'metric_name': 'error_count',
    'filter_expression': 'severity >= ERROR',
    'metric_kind': 'DELTA'
})

# Create alert policy
alert = mgr.metrics.create_alert_policy({
    'policy_name': 'high_error_rate',
    'metric_name': 'error_count',
    'threshold': 100
})

# Create BigQuery sink
sink = mgr.sink.create_bigquery_sink({
    'sink_name': 'logs_to_bigquery',
    'dataset_id': 'logs_dataset',
    'filter_expression': 'severity >= WARNING'
})

# Analyze log patterns
patterns = mgr.analytics.analyze_log_patterns('application')
```

## Best Practices

1. **Use structured logging** with JSON for better queryability
2. **Set appropriate severity levels** for consistent filtering
3. **Create log-based metrics** for monitoring key events
4. **Configure sinks** for long-term log retention
5. **Use exclusion filters** to reduce logging costs
6. **Enable audit logs** for compliance requirements

## Requirements

```
google-cloud-logging
google-cloud-monitoring
```

## Author

BrillConsulting - Enterprise Cloud Solutions
