# Linux Log Management System

> Production-ready centralized log management with aggregation, parsing, rotation, real-time monitoring, alerting, archival, and analytics

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

A comprehensive, enterprise-grade log management system for Linux environments that provides complete log lifecycle management from collection to archival. Built for production environments requiring robust logging, monitoring, and compliance capabilities.

## Key Features

### Log Aggregation & Parsing
- **Multi-format parsing** - Syslog, Apache, Nginx, JSON, and custom formats
- **Regex-based extraction** - Flexible pattern matching with named groups
- **Structured data extraction** - Convert unstructured logs to searchable fields
- **Real-time parsing** - Stream processing with minimal latency
- **Metrics extraction** - Automatic metric generation from parsed logs

### Advanced Log Rotation
- **Size-based rotation** - Rotate logs when size threshold is reached
- **Time-based rotation** - Daily, weekly, monthly rotation schedules
- **Compression** - gzip compression with configurable delay
- **Retention policies** - Automatic cleanup of old logs
- **Date extensions** - Timestamped log archives
- **Pre/post-rotation hooks** - Execute commands before/after rotation
- **Multiple configurations** - Manage rotation for different log types

### Real-Time Monitoring
- **Log tailing** - Follow log files in real-time
- **Pattern watching** - Alert on specific log patterns
- **journalctl integration** - Monitor systemd journal logs
- **Stream filtering** - Real-time grep with pattern matching
- **Multi-file monitoring** - Watch multiple log files simultaneously

### Multi-Channel Alerting
- **Email notifications** - SMTP-based alert delivery
- **Slack integration** - Real-time alerts to Slack channels
- **Webhook support** - POST alerts to custom endpoints
- **Alert rules** - Condition-based triggering with thresholds
- **Cooldown periods** - Prevent alert fatigue
- **Severity levels** - Critical, high, medium, low priorities
- **ElastAlert integration** - Advanced alerting for Elasticsearch

### Archival & Compression
- **gzip compression** - 70-80% space reduction
- **tar.gz archives** - Bundle multiple log files
- **Retention policies** - Automatic deletion of expired logs
- **Archive metadata** - Track archive size and contents
- **Compression ratios** - Monitor storage savings
- **Decompression utilities** - Restore archived logs

### Log Analytics
- **Pattern analysis** - Identify recurring patterns
- **Error detection** - Automatic error classification
- **IP analysis** - Track unique IPs and top sources
- **Status code analysis** - HTTP response code statistics
- **Time-series analysis** - Temporal pattern detection
- **Custom queries** - Complex search with multiple filters
- **Statistical reports** - Generate comprehensive analytics

### rsyslog/journalctl Integration
- **rsyslog configuration** - Advanced centralized logging setup
- **TLS/SSL support** - Encrypted log transmission
- **Disk-assisted queues** - Reliable log forwarding
- **Rate limiting** - Prevent log flooding
- **Facility filtering** - Selective log collection
- **journalctl queries** - Systemd journal integration
- **Priority filtering** - Focus on important events

### Elasticsearch Support
- **Index management** - Create and manage log indices
- **Query DSL** - Full Elasticsearch query capabilities
- **Aggregations** - Group and analyze log data
- **Time-based indices** - Daily/monthly index rotation
- **ILM policies** - Index lifecycle management
- **Template management** - Index mapping templates
- **Bulk operations** - Efficient batch processing

### Log Forwarding
- **Filebeat** - Lightweight log shipper
- **Logstash** - Advanced log processing pipeline
- **Fluentd** - Unified logging layer
- **Multiple outputs** - Elasticsearch, file, syslog
- **Load balancing** - Distribute logs across nodes
- **Failover** - Automatic retry with queuing
- **TLS encryption** - Secure log transmission

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Log Sources                                  │
│  Applications │ System Logs │ Web Servers │ Databases            │
└────────────────────┬────────────────────────────────────────────┘
                     │
        ┌────────────┴─────────────┬──────────────┐
        │                          │              │
┌───────▼────────┐      ┌─────────▼──────┐  ┌───▼──────────┐
│    rsyslog     │      │   Filebeat      │  │   Fluentd    │
│  (Collection)  │      │   (Shipper)     │  │  (Forwarder) │
└───────┬────────┘      └─────────┬──────┘  └───┬──────────┘
        │                         │              │
        └────────────┬────────────┴──────────────┘
                     │
             ┌───────▼──────────┐
             │    Logstash      │
             │   (Processing)   │
             │  Parsing/Filter  │
             └───────┬──────────┘
                     │
        ┌────────────┴─────────────┬──────────────┐
        │                          │              │
┌───────▼────────┐      ┌─────────▼──────┐  ┌───▼──────────┐
│ Elasticsearch  │      │   Log Files    │  │   Archive    │
│   (Storage)    │      │   (Local)      │  │  (Long-term) │
└───────┬────────┘      └────────────────┘  └──────────────┘
        │
┌───────▼────────┐
│    Kibana      │
│ (Visualization)│
└────────────────┘
        │
┌───────▼────────┐
│  Alert Manager │
│ Email/Slack/   │
│    Webhook     │
└────────────────┘
```

## Installation

```bash
# Clone repository
cd /home/user/BrillConsulting/Linux/LogManagement

# Install dependencies (if any)
pip install -r requirements.txt

# Run demo
python log_management.py
```

## Quick Start

### Basic Setup

```python
from log_management import LogManagement

# Initialize log management
log_mgr = LogManagement(hostname='prod-server-01')

# Configure rsyslog
log_mgr.configure_rsyslog({
    'remote_host': 'logserver.example.com',
    'remote_port': 514,
    'protocol': 'tcp',
    'tls_enabled': True,
    'queue_size': 50000
})

# Setup log rotation
log_mgr.configure_logrotate('app-logs', {
    'log_paths': ['/var/log/app/*.log'],
    'frequency': 'daily',
    'rotate': 30,
    'size': '500M',
    'compress': True
})

# Configure Elasticsearch
log_mgr.setup_elasticsearch({
    'hosts': ['es1.example.com:9200'],
    'index_pattern': 'logs-*'
})
```

### Log Shipping

```python
# Setup Filebeat
log_mgr.setup_filebeat({
    'log_paths': ['/var/log/app/*.log', '/var/log/nginx/*.log'],
    'elasticsearch_hosts': ['es1.example.com:9200'],
    'fields': {
        'environment': 'production',
        'application': 'web-app'
    }
})

# Setup Logstash
log_mgr.setup_logstash({
    'input_port': 5044,
    'elasticsearch_hosts': ['es1.example.com:9200'],
    'workers': 4
})

# Setup Fluentd
log_mgr.setup_fluentd({
    'elasticsearch_host': 'es1.example.com',
    'elasticsearch_port': 9200
})
```

### Real-Time Monitoring

```python
# Monitor log file
monitor_cmd = log_mgr.monitor.follow_log('/var/log/app/app.log', pattern='ERROR')

# Monitor journalctl
journal_cmd = log_mgr.monitor.monitor_journalctl(
    unit='nginx.service',
    priority='err',
    since='today'
)

# Create watch patterns
log_mgr.monitor.create_watch_pattern(r'ERROR.*database', 'db_alert')
```

### Alerting

```python
# Create multi-channel alert
log_mgr.create_log_alert({
    'name': 'high_error_rate',
    'condition': 'error_count',
    'threshold': 100,
    'severity': 'critical',
    'channels': ['email', 'slack', 'webhook'],
    'email': 'ops@example.com',
    'slack_webhook': 'https://hooks.slack.com/...',
    'cooldown_minutes': 10
})
```

### Log Analytics

```python
# Parse and analyze logs
analysis = log_mgr.analyze_logs('/var/log/app/app.log', format_type='json')

# Search Elasticsearch
results = log_mgr.search_elasticsearch({
    'time_range': {'from': 'now-1h', 'to': 'now'},
    'match': {'level': 'ERROR'},
    'size': 100
})
```

### Archival & Compression

```python
# Compress log file
result = log_mgr.archive_manager.compress_log('/var/log/app/old.log')

# Archive multiple logs
archive = log_mgr.archive_manager.archive_logs(
    ['/var/log/app/*.log'],
    'monthly-backup'
)

# Apply retention policy
cleanup = log_mgr.archive_manager.apply_retention_policy(
    '/var/log/archive',
    retention_days=90
)
```

## Configuration Examples

### rsyslog with TLS

```python
log_mgr.configure_rsyslog({
    'remote_host': 'logserver.example.com',
    'remote_port': 514,
    'protocol': 'tcp',
    'tls_enabled': True,
    'disk_queue': True,
    'queue_size': 50000,
    'rate_limit': {'interval': 5, 'burst': 200}
})
```

### Advanced Log Rotation

```python
log_mgr.configure_logrotate('nginx-logs', {
    'log_paths': ['/var/log/nginx/*.log'],
    'frequency': 'daily',
    'rotate': 30,
    'size': '500M',
    'maxsize': '1G',
    'compress': True,
    'delaycompress': True,
    'dateext': True,
    'maxage': 90,
    'postrotate': [
        'nginx -s reload',
        'systemctl reload rsyslog'
    ]
})
```

### journalctl Query

```python
log_mgr.query_journalctl({
    'unit': 'nginx.service',
    'since': 'today',
    'until': '2 hours ago',
    'priority': 'err',
    'lines': 100,
    'json': True
})
```

### Elasticsearch Query

```python
log_mgr.search_elasticsearch({
    'time_range': {
        'from': 'now-24h',
        'to': 'now'
    },
    'match': {
        'level': 'ERROR',
        'application': 'web-app'
    },
    'terms': {
        'environment': 'production'
    },
    'aggregations': {
        'errors_by_host': {
            'terms': {'field': 'hostname'}
        },
        'errors_over_time': {
            'date_histogram': {
                'field': '@timestamp',
                'interval': '1h'
            }
        }
    },
    'size': 100
})
```

## Components

### LogParser
Parses log files in multiple formats (syslog, Apache, Nginx, JSON)
- Regex-based pattern matching
- Named group extraction
- Metrics generation

### LogRotationManager
Manages log rotation with advanced configuration
- Size and time-based rotation
- Compression and archival
- Pre/post rotation scripts

### RealTimeMonitor
Real-time log monitoring and streaming
- Tail log files
- Pattern watching
- journalctl integration

### LogArchiveManager
Archival, compression, and retention
- gzip compression
- tar.gz archives
- Retention policies

### AlertManager
Multi-channel alerting system
- Email, Slack, webhook
- Conditional triggers
- Cooldown management

### LogAnalytics
Advanced log analytics and statistics
- Pattern analysis
- Error detection
- Statistical reports

### LogManagement
Main orchestration class integrating all components

## System Requirements

- **OS**: Linux (Ubuntu 18.04+, CentOS 7+, Debian 9+)
- **Python**: 3.8 or higher
- **Disk Space**: Varies by log volume (recommend 100GB+)
- **Memory**: 4GB+ recommended
- **Network**: Bandwidth for log forwarding

### Optional Dependencies

- **rsyslog**: System logging daemon
- **logrotate**: Log rotation utility
- **Elasticsearch**: 7.x or 8.x
- **Logstash**: 7.x or 8.x
- **Kibana**: 7.x or 8.x
- **Filebeat**: 7.x or 8.x
- **Fluentd**: td-agent 4.x

## Use Cases

### Centralized Log Aggregation
Collect logs from multiple servers into a central repository for unified analysis and monitoring.

### Security Monitoring (SIEM)
Track authentication failures, suspicious activities, and security events across infrastructure.

### Application Debugging
Correlate application logs across services to troubleshoot issues and identify root causes.

### Compliance & Auditing
Maintain comprehensive log archives with retention policies for regulatory compliance.

### Real-Time Alerting
Detect and respond to critical events as they occur with multi-channel notifications.

### Performance Monitoring
Analyze response times, error rates, and system metrics to optimize performance.

### Capacity Planning
Track log volume trends and storage requirements for infrastructure planning.

## Best Practices

1. **Centralize Logs Early** - Set up centralized logging before you need it
2. **Use Structured Logging** - JSON logs are easier to parse and search
3. **Implement Retention Policies** - Don't keep logs forever, define clear policies
4. **Compress Old Logs** - Save storage space with compression
5. **Monitor Log Volume** - Track log rates to detect anomalies
6. **Secure Log Transmission** - Use TLS for remote log forwarding
7. **Index Strategically** - Balance query performance with storage costs
8. **Alert Wisely** - Too many alerts lead to alert fatigue
9. **Regular Testing** - Test log rotation and archival processes
10. **Document Everything** - Keep configuration documentation up-to-date

## Performance Optimization

- **Batching**: Send logs in batches rather than one-by-one
- **Compression**: Use compression for network transmission
- **Filtering**: Filter unnecessary logs at source
- **Indexing**: Create indices on frequently searched fields
- **Sharding**: Distribute logs across multiple Elasticsearch shards
- **Buffering**: Use disk-assisted queues for reliability
- **Workers**: Scale Logstash workers based on load

## Troubleshooting

### Logs not forwarding
- Check network connectivity
- Verify rsyslog/Filebeat configuration
- Check disk queue status
- Review firewall rules

### High disk usage
- Implement log rotation
- Apply retention policies
- Enable compression
- Archive old logs

### Elasticsearch performance
- Optimize index settings
- Reduce shard count
- Implement ILM policies
- Scale cluster horizontally

### Missing logs
- Check parser configurations
- Verify log formats
- Review filter rules
- Check permissions

## Security Considerations

- **TLS Encryption**: Always use TLS for remote log transmission
- **Access Control**: Restrict access to log files and Elasticsearch
- **Sensitive Data**: Filter sensitive information (passwords, tokens)
- **Audit Logs**: Maintain audit trail of log access
- **Network Segmentation**: Isolate log infrastructure
- **Regular Updates**: Keep all components updated

## License

MIT License - See LICENSE file for details

## Author

**BrillConsulting**

## Version

**v2.0.0** - Production-Ready Release

## Support

For issues, questions, or contributions, please contact BrillConsulting.

---

**Production-ready log management for enterprise Linux environments**
