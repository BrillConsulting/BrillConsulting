"""
Cloud Logging Service
Author: BrillConsulting
Description: Advanced logging, monitoring, and log analytics with Cloud Logging
"""

import json
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import time


class LogWriter:
    """Cloud Logging log writer"""

    def __init__(self, project_id: str, log_name: str):
        """
        Initialize log writer

        Args:
            project_id: GCP project ID
            log_name: Log name
        """
        self.project_id = project_id
        self.log_name = log_name
        self.entries_written = 0

    def write_log_entry(self, severity: str, message: str, labels: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Write single log entry

        Args:
            severity: Log severity (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            message: Log message
            labels: Optional labels

        Returns:
            Log entry details
        """
        print(f"\n{'='*60}")
        print("Writing Log Entry")
        print(f"{'='*60}")

        labels = labels or {}

        code = f"""
from google.cloud import logging

client = logging.Client(project='{self.project_id}')
logger = client.logger('{self.log_name}')

# Write log entry
logger.log_text(
    "{message}",
    severity="{severity}",
    labels={labels}
)

print("Log entry written")
"""

        result = {
            'log_name': self.log_name,
            'severity': severity,
            'message': message,
            'labels': labels,
            'timestamp': datetime.now().isoformat(),
            'code': code
        }

        self.entries_written += 1

        print(f"✓ Log entry written")
        print(f"  Severity: {severity}")
        print(f"  Message: {message[:50]}...")
        print(f"{'='*60}")

        return result

    def write_structured_log(self, data: Dict[str, Any], severity: str = 'INFO') -> Dict[str, Any]:
        """
        Write structured log entry

        Args:
            data: Structured log data
            severity: Log severity

        Returns:
            Log entry details
        """
        code = f"""
from google.cloud import logging

client = logging.Client(project='{self.project_id}')
logger = client.logger('{self.log_name}')

# Write structured log
logger.log_struct(
    {data},
    severity="{severity}",
    resource=logging.Resource(
        type="global",
        labels={{}}
    )
)

print("Structured log written")
"""

        result = {
            'log_name': self.log_name,
            'data': data,
            'severity': severity,
            'structured': True,
            'code': code
        }

        print(f"\n✓ Structured log written: {list(data.keys())}")
        return result

    def write_batch_logs(self, entries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Write batch of log entries

        Args:
            entries: List of log entries

        Returns:
            Batch write result
        """
        print(f"\n{'='*60}")
        print("Writing Batch Logs")
        print(f"{'='*60}")

        code = f"""
from google.cloud import logging

client = logging.Client(project='{self.project_id}')
logger = client.logger('{self.log_name}')

# Write batch entries
entries = {entries}

for entry in entries:
    logger.log_struct(
        entry['data'],
        severity=entry['severity']
    )

print(f"Batch of {{len(entries)}} logs written")
"""

        result = {
            'entries_count': len(entries),
            'log_name': self.log_name,
            'code': code,
            'timestamp': datetime.now().isoformat()
        }

        self.entries_written += len(entries)

        print(f"✓ Batch written: {len(entries)} entries")
        print(f"  Total entries: {self.entries_written}")
        print(f"{'='*60}")

        return result


class LogReader:
    """Cloud Logging log reader and query"""

    def __init__(self, project_id: str):
        """Initialize log reader"""
        self.project_id = project_id

    def read_logs(self, filter_str: str, limit: int = 10) -> str:
        """
        Read logs with filter

        Args:
            filter_str: Log filter query
            limit: Maximum entries to read

        Returns:
            Read logs code
        """
        print(f"\n{'='*60}")
        print("Reading Logs")
        print(f"{'='*60}")

        code = f"""
from google.cloud import logging

client = logging.Client(project='{self.project_id}')

# Read logs with filter
filter_str = '''{filter_str}'''

entries = client.list_entries(
    filter_=filter_str,
    order_by=logging.DESCENDING,
    max_results={limit}
)

# Process entries
for entry in entries:
    print(f"{{entry.timestamp}}: {{entry.payload}}")
    print(f"  Severity: {{entry.severity}}")
    print(f"  Resource: {{entry.resource}}")
"""

        print(f"✓ Log query configured")
        print(f"  Filter: {filter_str[:50]}...")
        print(f"  Limit: {limit}")
        print(f"{'='*60}")

        return code

    def query_error_logs(self, hours: int = 24) -> str:
        """
        Query error logs from last N hours

        Args:
            hours: Hours to look back

        Returns:
            Error log query
        """
        filter_str = f'''
severity >= ERROR
timestamp >= "{datetime.now() - timedelta(hours=hours)}"
'''

        code = f"""
from google.cloud import logging
from datetime import datetime, timedelta

client = logging.Client(project='{self.project_id}')

# Query errors from last {hours} hours
start_time = datetime.now() - timedelta(hours={hours})
filter_str = f'''
severity >= ERROR
timestamp >= "{{start_time}}"
'''

entries = client.list_entries(filter_=filter_str)

# Analyze errors
error_counts = {{}}
for entry in entries:
    error_type = entry.payload.get('error_type', 'unknown')
    error_counts[error_type] = error_counts.get(error_type, 0) + 1

print("Error summary:")
for error_type, count in sorted(error_counts.items(), key=lambda x: x[1], reverse=True):
    print(f"  {{error_type}}: {{count}}")
"""

        print(f"\n✓ Error log query: last {hours} hours")
        return code

    def search_logs_by_user(self, user_id: str) -> str:
        """
        Search logs by user ID

        Args:
            user_id: User ID to search

        Returns:
            User log search query
        """
        code = f"""
from google.cloud import logging

client = logging.Client(project='{self.project_id}')

# Search by user ID
filter_str = f'''
jsonPayload.user_id="{user_id}"
'''

entries = client.list_entries(filter_=filter_str)

# Analyze user activity
for entry in entries:
    print(f"{{entry.timestamp}}: {{entry.payload.get('action')}}")
"""

        print(f"\n✓ User log search configured: {user_id}")
        return code


class LogMetrics:
    """Cloud Logging metrics and monitoring"""

    def __init__(self, project_id: str):
        """Initialize log metrics"""
        self.project_id = project_id

    def create_log_based_metric(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create log-based metric

        Args:
            config: Metric configuration

        Returns:
            Metric details
        """
        print(f"\n{'='*60}")
        print("Creating Log-Based Metric")
        print(f"{'='*60}")

        metric_name = config.get('metric_name', 'error_count')
        filter_str = config.get('filter', 'severity >= ERROR')
        description = config.get('description', 'Count of error logs')

        code = f"""
from google.cloud import logging_v2

client = logging_v2.MetricsServiceV2Client()

# Create log-based metric
parent = f"projects/{self.project_id}"
metric = {{
    "name": f"{{parent}}/metrics/{metric_name}",
    "description": "{description}",
    "filter": "{filter_str}",
    "metric_descriptor": {{
        "metric_kind": "DELTA",
        "value_type": "INT64",
        "labels": [
            {{"key": "severity", "value_type": "STRING"}},
            {{"key": "resource_type", "value_type": "STRING"}}
        ]
    }}
}}

response = client.create_log_metric(parent=parent, metric=metric)
print(f"Created metric: {{response.name}}")
"""

        result = {
            'metric_name': metric_name,
            'filter': filter_str,
            'description': description,
            'code': code,
            'timestamp': datetime.now().isoformat()
        }

        print(f"✓ Log-based metric created: {metric_name}")
        print(f"  Filter: {filter_str}")
        print(f"{'='*60}")

        return result

    def create_alert_policy(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create alert policy for log metrics

        Args:
            config: Alert configuration

        Returns:
            Alert policy details
        """
        policy_name = config.get('policy_name', 'high_error_rate')
        metric_name = config.get('metric_name', 'error_count')
        threshold = config.get('threshold', 100)

        code = f"""
from google.cloud import monitoring_v3

client = monitoring_v3.AlertPolicyServiceClient()

# Create alert policy
project_name = f"projects/{self.project_id}"

policy = monitoring_v3.AlertPolicy(
    display_name="{policy_name}",
    conditions=[
        monitoring_v3.AlertPolicy.Condition(
            display_name="Error rate too high",
            condition_threshold=monitoring_v3.AlertPolicy.Condition.MetricThreshold(
                filter=f'metric.type="logging.googleapis.com/user/{metric_name}"',
                comparison=monitoring_v3.ComparisonType.COMPARISON_GT,
                threshold_value={threshold},
                duration={{"seconds": 300}}
            )
        )
    ],
    notification_channels=[
        # Add notification channel IDs
    ],
    alert_strategy=monitoring_v3.AlertPolicy.AlertStrategy(
        auto_close={{"seconds": 86400}}  # 24 hours
    )
)

response = client.create_alert_policy(name=project_name, alert_policy=policy)
print(f"Created alert policy: {{response.name}}")
"""

        result = {
            'policy_name': policy_name,
            'metric_name': metric_name,
            'threshold': threshold,
            'code': code
        }

        print(f"\n✓ Alert policy created: {policy_name}")
        print(f"  Threshold: {threshold}")
        return result


class LogSink:
    """Cloud Logging sink for export"""

    def __init__(self, project_id: str):
        """Initialize log sink"""
        self.project_id = project_id

    def create_bigquery_sink(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create sink to BigQuery

        Args:
            config: Sink configuration

        Returns:
            Sink details
        """
        print(f"\n{'='*60}")
        print("Creating BigQuery Log Sink")
        print(f"{'='*60}")

        sink_name = config.get('sink_name', 'logs_to_bigquery')
        dataset_id = config.get('dataset_id', 'logs')
        filter_str = config.get('filter', 'severity >= INFO')

        code = f"""
from google.cloud import logging_v2

client = logging_v2.ConfigServiceV2Client()

# Create BigQuery sink
parent = f"projects/{self.project_id}"
sink = {{
    "name": "{sink_name}",
    "destination": f"bigquery.googleapis.com/projects/{self.project_id}/datasets/{dataset_id}",
    "filter": "{filter_str}",
    "output_version_format": "V2"
}}

response = client.create_sink(parent=parent, sink=sink)
print(f"Created sink: {{response.name}}")
print(f"Writer identity: {{response.writer_identity}}")
"""

        result = {
            'sink_name': sink_name,
            'destination': f"bigquery.googleapis.com/projects/{self.project_id}/datasets/{dataset_id}",
            'filter': filter_str,
            'code': code,
            'timestamp': datetime.now().isoformat()
        }

        print(f"✓ BigQuery sink created: {sink_name}")
        print(f"  Dataset: {dataset_id}")
        print(f"  Filter: {filter_str}")
        print(f"{'='*60}")

        return result

    def create_storage_sink(self, bucket_name: str, filter_str: str = '') -> str:
        """
        Create sink to Cloud Storage

        Args:
            bucket_name: GCS bucket name
            filter_str: Log filter

        Returns:
            Storage sink code
        """
        code = f"""
from google.cloud import logging_v2

client = logging_v2.ConfigServiceV2Client()

# Create Cloud Storage sink
parent = f"projects/{self.project_id}"
sink = {{
    "name": "logs_to_storage",
    "destination": f"storage.googleapis.com/{bucket_name}",
    "filter": "{filter_str}"
}}

response = client.create_sink(parent=parent, sink=sink)
print(f"Created storage sink: {{response.name}}")
"""

        print(f"\n✓ Storage sink configured: {bucket_name}")
        return code


class LogAnalyzer:
    """Advanced log analysis"""

    def __init__(self, project_id: str):
        """Initialize log analyzer"""
        self.project_id = project_id

    def analyze_request_latency(self) -> str:
        """
        Analyze request latency from logs

        Returns:
            Analysis code
        """
        code = f"""
from google.cloud import logging
import statistics

client = logging.Client(project='{self.project_id}')

# Query latency logs
filter_str = '''
jsonPayload.latency_ms:*
timestamp >= "2024-01-01"
'''

entries = client.list_entries(filter_=filter_str, max_results=1000)

# Analyze latencies
latencies = []
for entry in entries:
    latency = entry.payload.get('latency_ms', 0)
    latencies.append(latency)

if latencies:
    print(f"Latency statistics:")
    print(f"  Count: {{len(latencies)}}")
    print(f"  Mean: {{statistics.mean(latencies):.2f}}ms")
    print(f"  Median: {{statistics.median(latencies):.2f}}ms")
    print(f"  P95: {{sorted(latencies)[int(len(latencies)*0.95)]:.2f}}ms")
    print(f"  P99: {{sorted(latencies)[int(len(latencies)*0.99)]:.2f}}ms")
"""

        print("\n✓ Latency analysis configured")
        return code

    def detect_error_patterns(self) -> str:
        """
        Detect error patterns in logs

        Returns:
            Pattern detection code
        """
        code = f"""
from google.cloud import logging
from collections import Counter

client = logging.Client(project='{self.project_id}')

# Query error logs
filter_str = 'severity >= ERROR'
entries = client.list_entries(filter_=filter_str, max_results=1000)

# Analyze error patterns
error_messages = []
error_sources = []

for entry in entries:
    msg = str(entry.payload)[:100]
    error_messages.append(msg)
    error_sources.append(entry.resource.type)

# Top error messages
print("Top error messages:")
for msg, count in Counter(error_messages).most_common(5):
    print(f"  {{count}}x: {{msg}}")

# Error distribution by source
print("\\nErrors by source:")
for source, count in Counter(error_sources).most_common(5):
    print(f"  {{source}}: {{count}}")
"""

        print("\n✓ Error pattern detection configured")
        return code


class CloudLoggingManager:
    """Comprehensive Cloud Logging management"""

    def __init__(self, project_id: str = 'my-project'):
        """Initialize Cloud Logging manager"""
        self.project_id = project_id
        self.logs_written = 0
        self.metrics = []
        self.sinks = []

    def get_manager_info(self) -> Dict[str, Any]:
        """Get manager information"""
        return {
            'project_id': self.project_id,
            'logs_written': self.logs_written,
            'metrics': len(self.metrics),
            'sinks': len(self.sinks),
            'features': [
                'structured_logging',
                'log_queries',
                'metrics',
                'alerts',
                'sinks',
                'analytics'
            ],
            'timestamp': datetime.now().isoformat()
        }


def demo():
    """Demonstrate Cloud Logging capabilities"""
    print("=" * 60)
    print("Cloud Logging Service Demo")
    print("=" * 60)

    project_id = 'my-gcp-project'

    # Write logs
    writer = LogWriter(project_id, 'application-logs')

    log_entry = writer.write_log_entry(
        severity='INFO',
        message='User login successful',
        labels={'user_id': '123', 'ip': '192.168.1.1'}
    )

    struct_log = writer.write_structured_log(
        data={
            'user_id': '123',
            'action': 'purchase',
            'amount': 99.99,
            'items': ['item1', 'item2']
        },
        severity='INFO'
    )

    batch_entries = [
        {'data': {'event': 'page_view', 'page': f'/page{i}'}, 'severity': 'INFO'}
        for i in range(10)
    ]
    batch_result = writer.write_batch_logs(batch_entries)

    # Read logs
    reader = LogReader(project_id)
    read_code = reader.read_logs('severity >= WARNING', limit=20)
    error_query = reader.query_error_logs(hours=24)
    user_search = reader.search_logs_by_user('user_123')

    # Create metrics
    metrics_mgr = LogMetrics(project_id)
    metric = metrics_mgr.create_log_based_metric({
        'metric_name': 'error_count',
        'filter': 'severity >= ERROR',
        'description': 'Count of error logs'
    })

    alert = metrics_mgr.create_alert_policy({
        'policy_name': 'high_error_rate',
        'metric_name': 'error_count',
        'threshold': 100
    })

    # Create sinks
    sink_mgr = LogSink(project_id)
    bq_sink = sink_mgr.create_bigquery_sink({
        'sink_name': 'logs_to_bigquery',
        'dataset_id': 'application_logs',
        'filter': 'severity >= INFO'
    })

    storage_sink = sink_mgr.create_storage_sink(
        bucket_name='my-logs-bucket',
        filter_str='severity >= WARNING'
    )

    # Analytics
    analyzer = LogAnalyzer(project_id)
    latency_analysis = analyzer.analyze_request_latency()
    error_patterns = analyzer.detect_error_patterns()

    # Manager info
    mgr = CloudLoggingManager(project_id)
    mgr.logs_written = writer.entries_written
    mgr.metrics.append(metric)
    mgr.sinks.append(bq_sink)

    info = mgr.get_manager_info()
    print(f"\n{'='*60}")
    print("Cloud Logging Manager Summary")
    print(f"{'='*60}")
    print(f"Project: {info['project_id']}")
    print(f"Logs written: {info['logs_written']}")
    print(f"Metrics: {info['metrics']}")
    print(f"Sinks: {info['sinks']}")
    print(f"Features: {', '.join(info['features'])}")
    print(f"{'='*60}")

    print("\n✓ Demo completed successfully!")


if __name__ == "__main__":
    demo()
