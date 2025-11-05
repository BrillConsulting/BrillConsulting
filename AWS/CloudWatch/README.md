# AWS CloudWatch

Comprehensive monitoring and observability service for AWS resources and applications.

## Features

- **Metrics**: Publish and retrieve custom and standard metrics
- **Alarms**: Create threshold-based and anomaly detection alarms
- **Logs**: Centralized log collection, storage, and analysis
- **Dashboards**: Visual monitoring with customizable widgets
- **Log Insights**: Query and analyze log data with SQL-like syntax
- **Metrics Math**: Perform calculations on metrics
- **Composite Alarms**: Combine multiple alarms with logic
- **Events**: Event-driven automation (EventBridge integration)

## Quick Start

```python
from aws_cloudwatch import CloudWatchManager

# Initialize
cw = CloudWatchManager(region='us-east-1')

# Publish custom metric
cw.put_metric_data(
    namespace='MyApp',
    metric_name='RequestCount',
    value=125.0,
    unit='Count',
    dimensions=[{'Name': 'Environment', 'Value': 'Production'}]
)

# Create alarm
cw.put_metric_alarm(
    alarm_name='HighCPU',
    comparison_operator='GreaterThanThreshold',
    evaluation_periods=2,
    metric_name='CPUUtilization',
    namespace='AWS/EC2',
    period=300,
    statistic='Average',
    threshold=80.0,
    actions_enabled=True,
    alarm_actions=['arn:aws:sns:us-east-1:123456789012:alerts']
)

# Send logs
cw.put_log_events(
    log_group_name='/aws/lambda/my-function',
    log_stream_name='2024/11/05',
    messages=['Application started', 'Processing request', 'Request completed']
)

# Create dashboard
cw.put_dashboard(
    dashboard_name='AppMetrics',
    dashboard_body={
        'widgets': [{
            'type': 'metric',
            'properties': {
                'metrics': [['AWS/EC2', 'CPUUtilization']],
                'period': 300,
                'stat': 'Average',
                'region': 'us-east-1',
                'title': 'EC2 CPU Usage'
            }
        }]
    }
)
```

## Use Cases

- **Application Monitoring**: Track application performance metrics
- **Infrastructure Monitoring**: Monitor EC2, RDS, Lambda resources
- **Log Aggregation**: Centralize logs from multiple sources
- **Alerting**: Automated notifications for threshold breaches
- **Performance Analysis**: Analyze system bottlenecks
- **Cost Optimization**: Track resource utilization
- **Debugging**: Search and analyze logs for troubleshooting

## Metrics Types

### Standard Metrics (AWS Services)
- **EC2**: CPUUtilization, DiskReadBytes, NetworkIn
- **RDS**: DatabaseConnections, FreeStorageSpace
- **Lambda**: Invocations, Duration, Errors
- **DynamoDB**: ConsumedReadCapacityUnits, ThrottledRequests

### Custom Metrics
```python
cw.put_metric_data(
    namespace='MyApplication',
    metric_name='OrdersProcessed',
    value=42,
    timestamp=datetime.now(),
    dimensions=[
        {'Name': 'Region', 'Value': 'US'},
        {'Name': 'Service', 'Value': 'OrderProcessor'}
    ]
)
```

## Alarm States

- **OK**: Metric is within threshold
- **ALARM**: Metric breached threshold
- **INSUFFICIENT_DATA**: Not enough data to evaluate

## Log Insights Queries

```python
# Query logs
results = cw.start_query(
    log_group_name='/aws/lambda/my-function',
    query_string='''
        fields @timestamp, @message
        | filter @message like /ERROR/
        | sort @timestamp desc
        | limit 100
    ''',
    start_time=datetime.now() - timedelta(hours=1),
    end_time=datetime.now()
)
```

## Best Practices

- Use namespace prefixes to organize metrics
- Set appropriate alarm thresholds to avoid noise
- Use composite alarms for complex conditions
- Enable detailed monitoring for critical resources
- Implement log retention policies
- Use metric filters to extract metrics from logs
- Create dashboards for team visibility

## Author

Brill Consulting
