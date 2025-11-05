"""
Azure Monitor Service Integration
Author: BrillConsulting
Contact: clientbrill@gmail.com
LinkedIn: brillconsulting
Description: Advanced Azure Monitor implementation with metrics, logs, alerts, and diagnostics
"""

from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict, field
from enum import Enum
import json


class MetricAggregation(Enum):
    """Metric aggregation types"""
    AVERAGE = "Average"
    COUNT = "Count"
    MINIMUM = "Minimum"
    MAXIMUM = "Maximum"
    TOTAL = "Total"


class AlertSeverity(Enum):
    """Alert severity levels"""
    CRITICAL = 0
    ERROR = 1
    WARNING = 2
    INFORMATIONAL = 3
    VERBOSE = 4


class AlertState(Enum):
    """Alert states"""
    NEW = "New"
    ACKNOWLEDGED = "Acknowledged"
    CLOSED = "Closed"


class LogLevel(Enum):
    """Log severity levels"""
    TRACE = "Trace"
    DEBUG = "Debug"
    INFORMATION = "Information"
    WARNING = "Warning"
    ERROR = "Error"
    CRITICAL = "Critical"


@dataclass
class MetricValue:
    """Metric data point"""
    timestamp: str
    value: float
    count: Optional[int] = None
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    total: Optional[float] = None


@dataclass
class MetricDefinition:
    """Metric definition"""
    name: str
    namespace: str
    display_name: str
    unit: str
    aggregations: List[str]
    dimensions: Optional[List[str]] = None


@dataclass
class AlertRule:
    """Alert rule configuration"""
    name: str
    description: str
    severity: AlertSeverity
    enabled: bool
    condition: Dict[str, Any]
    actions: List[str]
    frequency: int  # minutes
    window_size: int  # minutes
    created_at: str
    updated_at: str


@dataclass
class LogEntry:
    """Log entry"""
    timestamp: str
    level: LogLevel
    message: str
    resource_id: str
    properties: Optional[Dict[str, Any]] = None
    correlation_id: Optional[str] = None


@dataclass
class DiagnosticSetting:
    """Diagnostic setting configuration"""
    name: str
    resource_id: str
    workspace_id: Optional[str] = None
    storage_account_id: Optional[str] = None
    event_hub_id: Optional[str] = None
    logs: List[Dict[str, Any]] = field(default_factory=list)
    metrics: List[Dict[str, Any]] = field(default_factory=list)


class MetricsManager:
    """
    Manage Azure Monitor metrics
    
    Features:
    - Metrics collection and storage
    - Time-series queries
    - Aggregation operations
    - Metric definitions
    """
    
    def __init__(self):
        self.metrics: Dict[str, List[MetricValue]] = {}
        self.metric_definitions: Dict[str, MetricDefinition] = {}
    
    def define_metric(
        self,
        name: str,
        namespace: str,
        display_name: str,
        unit: str,
        aggregations: List[MetricAggregation],
        dimensions: Optional[List[str]] = None
    ) -> MetricDefinition:
        """
        Define a custom metric
        
        Args:
            name: Metric name
            namespace: Metric namespace
            display_name: Display name
            unit: Unit of measurement
            aggregations: Supported aggregations
            dimensions: Metric dimensions
            
        Returns:
            MetricDefinition object
        """
        metric_def = MetricDefinition(
            name=name,
            namespace=namespace,
            display_name=display_name,
            unit=unit,
            aggregations=[agg.value for agg in aggregations],
            dimensions=dimensions or []
        )
        
        metric_key = f"{namespace}/{name}"
        self.metric_definitions[metric_key] = metric_def
        
        return metric_def
    
    def emit_metric(
        self,
        namespace: str,
        name: str,
        value: float,
        timestamp: Optional[datetime] = None,
        dimensions: Optional[Dict[str, str]] = None
    ) -> MetricValue:
        """
        Emit a metric value
        
        Args:
            namespace: Metric namespace
            name: Metric name
            value: Metric value
            timestamp: Timestamp (now if not specified)
            dimensions: Dimension values
            
        Returns:
            MetricValue object
        """
        metric_key = f"{namespace}/{name}"
        
        metric_value = MetricValue(
            timestamp=(timestamp or datetime.now()).isoformat(),
            value=value
        )
        
        if metric_key not in self.metrics:
            self.metrics[metric_key] = []
        
        self.metrics[metric_key].append(metric_value)
        
        return metric_value
    
    def query_metrics(
        self,
        namespace: str,
        name: str,
        start_time: datetime,
        end_time: datetime,
        aggregation: MetricAggregation = MetricAggregation.AVERAGE,
        interval: Optional[int] = None
    ) -> List[MetricValue]:
        """
        Query metrics within time range
        
        Args:
            namespace: Metric namespace
            name: Metric name
            start_time: Start time
            end_time: End time
            aggregation: Aggregation type
            interval: Time interval in minutes
            
        Returns:
            List of MetricValue objects
        """
        metric_key = f"{namespace}/{name}"
        
        if metric_key not in self.metrics:
            return []
        
        # Filter by time range
        results = [
            m for m in self.metrics[metric_key]
            if start_time.isoformat() <= m.timestamp <= end_time.isoformat()
        ]
        
        return results
    
    def get_metric_statistics(
        self,
        namespace: str,
        name: str,
        start_time: datetime,
        end_time: datetime
    ) -> Dict[str, Any]:
        """
        Get statistical summary of metrics
        
        Args:
            namespace: Metric namespace
            name: Metric name
            start_time: Start time
            end_time: End time
            
        Returns:
            Statistics dictionary
        """
        values = self.query_metrics(namespace, name, start_time, end_time)
        
        if not values:
            return {
                "count": 0,
                "average": 0,
                "minimum": 0,
                "maximum": 0,
                "sum": 0
            }
        
        value_list = [v.value for v in values]
        
        return {
            "count": len(value_list),
            "average": sum(value_list) / len(value_list),
            "minimum": min(value_list),
            "maximum": max(value_list),
            "sum": sum(value_list)
        }


class LogAnalyticsManager:
    """
    Manage Log Analytics workspace
    
    Features:
    - Log ingestion
    - KQL query execution
    - Log retention
    - Custom tables
    """
    
    def __init__(self, workspace_id: str, workspace_name: str):
        self.workspace_id = workspace_id
        self.workspace_name = workspace_name
        self.logs: List[LogEntry] = []
        self.custom_tables: Dict[str, List[Dict[str, Any]]] = {}
    
    def ingest_log(
        self,
        level: LogLevel,
        message: str,
        resource_id: str,
        properties: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None
    ) -> LogEntry:
        """
        Ingest a log entry
        
        Args:
            level: Log level
            message: Log message
            resource_id: Resource identifier
            properties: Additional properties
            correlation_id: Correlation ID
            
        Returns:
            LogEntry object
        """
        log_entry = LogEntry(
            timestamp=datetime.now().isoformat(),
            level=level,
            message=message,
            resource_id=resource_id,
            properties=properties or {},
            correlation_id=correlation_id
        )
        
        self.logs.append(log_entry)
        
        return log_entry
    
    def query_logs(
        self,
        query: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[LogEntry]:
        """
        Execute KQL query on logs
        
        Args:
            query: KQL query string
            start_time: Start time filter
            end_time: End time filter
            
        Returns:
            List of matching LogEntry objects
        """
        # Simple filtering (real implementation would use KQL parser)
        results = self.logs.copy()
        
        if start_time:
            results = [
                log for log in results
                if datetime.fromisoformat(log.timestamp) >= start_time
            ]
        
        if end_time:
            results = [
                log for log in results
                if datetime.fromisoformat(log.timestamp) <= end_time
            ]
        
        # Simulate query filtering
        if "where" in query.lower():
            if "error" in query.lower():
                results = [log for log in results if log.level == LogLevel.ERROR]
            elif "warning" in query.lower():
                results = [log for log in results if log.level == LogLevel.WARNING]
        
        return results
    
    def create_custom_table(
        self,
        table_name: str,
        schema: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """
        Create custom log table
        
        Args:
            table_name: Table name
            schema: Table schema definition
            
        Returns:
            Table creation result
        """
        self.custom_tables[table_name] = []
        
        return {
            "table_name": table_name,
            "schema": schema,
            "created_at": datetime.now().isoformat()
        }
    
    def insert_custom_log(
        self,
        table_name: str,
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Insert data into custom table
        
        Args:
            table_name: Table name
            data: Data to insert
            
        Returns:
            Insert result
        """
        if table_name not in self.custom_tables:
            raise ValueError(f"Table '{table_name}' not found")
        
        record = {
            **data,
            "TimeGenerated": datetime.now().isoformat()
        }
        
        self.custom_tables[table_name].append(record)
        
        return {
            "table": table_name,
            "inserted_at": record["TimeGenerated"]
        }


class AlertManager:
    """
    Manage Azure Monitor alerts
    
    Features:
    - Metric alerts
    - Log query alerts
    - Activity log alerts
    - Alert rules and action groups
    """
    
    def __init__(self):
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: List[Dict[str, Any]] = []
        self.action_groups: Dict[str, Dict[str, Any]] = {}
    
    def create_metric_alert(
        self,
        name: str,
        description: str,
        resource_id: str,
        metric_namespace: str,
        metric_name: str,
        operator: str,
        threshold: float,
        severity: AlertSeverity,
        frequency: int = 5,
        window_size: int = 5,
        action_groups: Optional[List[str]] = None
    ) -> AlertRule:
        """
        Create metric alert rule
        
        Args:
            name: Alert rule name
            description: Alert description
            resource_id: Target resource
            metric_namespace: Metric namespace
            metric_name: Metric name
            operator: Comparison operator (>, <, >=, <=, ==)
            threshold: Threshold value
            severity: Alert severity
            frequency: Evaluation frequency in minutes
            window_size: Time window in minutes
            action_groups: Action group IDs
            
        Returns:
            AlertRule object
        """
        condition = {
            "type": "metric",
            "resource_id": resource_id,
            "metric_namespace": metric_namespace,
            "metric_name": metric_name,
            "operator": operator,
            "threshold": threshold
        }
        
        alert_rule = AlertRule(
            name=name,
            description=description,
            severity=severity,
            enabled=True,
            condition=condition,
            actions=action_groups or [],
            frequency=frequency,
            window_size=window_size,
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat()
        )
        
        self.alert_rules[name] = alert_rule
        
        return alert_rule
    
    def create_log_query_alert(
        self,
        name: str,
        description: str,
        workspace_id: str,
        query: str,
        severity: AlertSeverity,
        threshold: int,
        frequency: int = 5,
        window_size: int = 5,
        action_groups: Optional[List[str]] = None
    ) -> AlertRule:
        """
        Create log query alert rule
        
        Args:
            name: Alert rule name
            description: Alert description
            workspace_id: Log Analytics workspace ID
            query: KQL query
            severity: Alert severity
            threshold: Result count threshold
            frequency: Evaluation frequency in minutes
            window_size: Time window in minutes
            action_groups: Action group IDs
            
        Returns:
            AlertRule object
        """
        condition = {
            "type": "log_query",
            "workspace_id": workspace_id,
            "query": query,
            "threshold": threshold
        }
        
        alert_rule = AlertRule(
            name=name,
            description=description,
            severity=severity,
            enabled=True,
            condition=condition,
            actions=action_groups or [],
            frequency=frequency,
            window_size=window_size,
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat()
        )
        
        self.alert_rules[name] = alert_rule
        
        return alert_rule
    
    def trigger_alert(
        self,
        alert_rule_name: str,
        fired_at: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Trigger an alert
        
        Args:
            alert_rule_name: Alert rule name
            fired_at: Timestamp when fired
            
        Returns:
            Alert instance
        """
        if alert_rule_name not in self.alert_rules:
            raise ValueError(f"Alert rule '{alert_rule_name}' not found")
        
        rule = self.alert_rules[alert_rule_name]
        
        alert = {
            "alert_id": f"alert-{datetime.now().timestamp()}",
            "alert_rule": alert_rule_name,
            "severity": rule.severity.value,
            "state": AlertState.NEW.value,
            "fired_at": (fired_at or datetime.now()).isoformat(),
            "description": rule.description,
            "condition": rule.condition
        }
        
        self.active_alerts.append(alert)
        
        return alert
    
    def create_action_group(
        self,
        name: str,
        short_name: str,
        email_receivers: Optional[List[str]] = None,
        sms_receivers: Optional[List[Dict[str, str]]] = None,
        webhook_receivers: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Create action group for alert notifications
        
        Args:
            name: Action group name
            short_name: Short name (12 chars max)
            email_receivers: Email addresses
            sms_receivers: SMS receivers
            webhook_receivers: Webhook URLs
            
        Returns:
            Action group configuration
        """
        action_group = {
            "name": name,
            "short_name": short_name[:12],
            "email_receivers": email_receivers or [],
            "sms_receivers": sms_receivers or [],
            "webhook_receivers": webhook_receivers or [],
            "created_at": datetime.now().isoformat()
        }
        
        self.action_groups[name] = action_group
        
        return action_group
    
    def acknowledge_alert(self, alert_id: str) -> Dict[str, Any]:
        """Acknowledge an alert"""
        for alert in self.active_alerts:
            if alert["alert_id"] == alert_id:
                alert["state"] = AlertState.ACKNOWLEDGED.value
                alert["acknowledged_at"] = datetime.now().isoformat()
                return alert
        
        raise ValueError(f"Alert '{alert_id}' not found")
    
    def close_alert(self, alert_id: str) -> Dict[str, Any]:
        """Close an alert"""
        for alert in self.active_alerts:
            if alert["alert_id"] == alert_id:
                alert["state"] = AlertState.CLOSED.value
                alert["closed_at"] = datetime.now().isoformat()
                return alert
        
        raise ValueError(f"Alert '{alert_id}' not found")


class DiagnosticsManager:
    """
    Manage diagnostic settings
    
    Features:
    - Configure diagnostic logs and metrics
    - Route to Log Analytics, Storage, Event Hub
    - Retention policies
    """
    
    def __init__(self):
        self.diagnostic_settings: Dict[str, DiagnosticSetting] = {}
    
    def create_diagnostic_setting(
        self,
        name: str,
        resource_id: str,
        workspace_id: Optional[str] = None,
        storage_account_id: Optional[str] = None,
        event_hub_id: Optional[str] = None,
        logs: Optional[List[Dict[str, Any]]] = None,
        metrics: Optional[List[Dict[str, Any]]] = None
    ) -> DiagnosticSetting:
        """
        Create diagnostic setting
        
        Args:
            name: Setting name
            resource_id: Resource to monitor
            workspace_id: Log Analytics workspace
            storage_account_id: Storage account for archiving
            event_hub_id: Event Hub for streaming
            logs: Log categories to collect
            metrics: Metrics to collect
            
        Returns:
            DiagnosticSetting object
        """
        setting = DiagnosticSetting(
            name=name,
            resource_id=resource_id,
            workspace_id=workspace_id,
            storage_account_id=storage_account_id,
            event_hub_id=event_hub_id,
            logs=logs or [],
            metrics=metrics or []
        )
        
        key = f"{resource_id}/{name}"
        self.diagnostic_settings[key] = setting
        
        return setting
    
    def get_diagnostic_setting(
        self,
        resource_id: str,
        name: str
    ) -> Optional[DiagnosticSetting]:
        """Get diagnostic setting"""
        key = f"{resource_id}/{name}"
        return self.diagnostic_settings.get(key)
    
    def list_diagnostic_settings(
        self,
        resource_id: str
    ) -> List[DiagnosticSetting]:
        """List all diagnostic settings for a resource"""
        return [
            setting for key, setting in self.diagnostic_settings.items()
            if setting.resource_id == resource_id
        ]


class ApplicationInsightsManager:
    """
    Manage Application Insights
    
    Features:
    - Request tracking
    - Dependency tracking
    - Exception tracking
    - Custom events and metrics
    """
    
    def __init__(self, instrumentation_key: str):
        self.instrumentation_key = instrumentation_key
        self.requests: List[Dict[str, Any]] = []
        self.dependencies: List[Dict[str, Any]] = []
        self.exceptions: List[Dict[str, Any]] = []
        self.custom_events: List[Dict[str, Any]] = []
    
    def track_request(
        self,
        name: str,
        url: str,
        duration_ms: float,
        response_code: int,
        success: bool,
        timestamp: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Track HTTP request
        
        Args:
            name: Request name
            url: Request URL
            duration_ms: Duration in milliseconds
            response_code: HTTP response code
            success: Whether request succeeded
            timestamp: Timestamp
            
        Returns:
            Request telemetry
        """
        request = {
            "name": name,
            "url": url,
            "duration": duration_ms,
            "response_code": response_code,
            "success": success,
            "timestamp": (timestamp or datetime.now()).isoformat()
        }
        
        self.requests.append(request)
        
        return request
    
    def track_dependency(
        self,
        name: str,
        dependency_type: str,
        target: str,
        data: str,
        duration_ms: float,
        success: bool,
        timestamp: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Track dependency call
        
        Args:
            name: Dependency name
            dependency_type: Type (SQL, HTTP, Azure, etc.)
            target: Target resource
            data: Command or query
            duration_ms: Duration in milliseconds
            success: Whether call succeeded
            timestamp: Timestamp
            
        Returns:
            Dependency telemetry
        """
        dependency = {
            "name": name,
            "type": dependency_type,
            "target": target,
            "data": data,
            "duration": duration_ms,
            "success": success,
            "timestamp": (timestamp or datetime.now()).isoformat()
        }
        
        self.dependencies.append(dependency)
        
        return dependency
    
    def track_exception(
        self,
        exception_type: str,
        message: str,
        stack_trace: str,
        properties: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Track exception
        
        Args:
            exception_type: Exception type
            message: Error message
            stack_trace: Stack trace
            properties: Additional properties
            timestamp: Timestamp
            
        Returns:
            Exception telemetry
        """
        exception = {
            "type": exception_type,
            "message": message,
            "stack_trace": stack_trace,
            "properties": properties or {},
            "timestamp": (timestamp or datetime.now()).isoformat()
        }
        
        self.exceptions.append(exception)
        
        return exception
    
    def track_event(
        self,
        name: str,
        properties: Optional[Dict[str, Any]] = None,
        measurements: Optional[Dict[str, float]] = None,
        timestamp: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Track custom event
        
        Args:
            name: Event name
            properties: Event properties
            measurements: Event measurements
            timestamp: Timestamp
            
        Returns:
            Event telemetry
        """
        event = {
            "name": name,
            "properties": properties or {},
            "measurements": measurements or {},
            "timestamp": (timestamp or datetime.now()).isoformat()
        }
        
        self.custom_events.append(event)
        
        return event


class AzureMonitorManager:
    """
    Comprehensive Azure Monitor manager
    
    Features:
    - Unified monitoring interface
    - Metrics and logs
    - Alerts and diagnostics
    - Application Insights integration
    """
    
    def __init__(
        self,
        subscription_id: str,
        resource_group: str
    ):
        self.subscription_id = subscription_id
        self.resource_group = resource_group
        
        self.metrics = MetricsManager()
        self.alerts = AlertManager()
        self.diagnostics = DiagnosticsManager()
        
        self.workspaces: Dict[str, LogAnalyticsManager] = {}
        self.app_insights: Dict[str, ApplicationInsightsManager] = {}
    
    def create_log_analytics_workspace(
        self,
        workspace_id: str,
        workspace_name: str
    ) -> LogAnalyticsManager:
        """Create Log Analytics workspace"""
        workspace = LogAnalyticsManager(workspace_id, workspace_name)
        self.workspaces[workspace_id] = workspace
        return workspace
    
    def get_workspace(self, workspace_id: str) -> Optional[LogAnalyticsManager]:
        """Get Log Analytics workspace"""
        return self.workspaces.get(workspace_id)
    
    def create_application_insights(
        self,
        name: str,
        instrumentation_key: str
    ) -> ApplicationInsightsManager:
        """Create Application Insights instance"""
        app_insights = ApplicationInsightsManager(instrumentation_key)
        self.app_insights[name] = app_insights
        return app_insights
    
    def get_application_insights(self, name: str) -> Optional[ApplicationInsightsManager]:
        """Get Application Insights instance"""
        return self.app_insights.get(name)


# Demo functions continue below...


def demo_metrics():
    """Demonstrate metrics collection"""
    print("=== Metrics Demo ===\n")
    
    metrics = MetricsManager()
    
    # Define custom metrics
    metrics.define_metric(
        "RequestCount",
        "MyApp",
        "Request Count",
        "Count",
        [MetricAggregation.COUNT, MetricAggregation.TOTAL]
    )
    
    metrics.define_metric(
        "ResponseTime",
        "MyApp",
        "Response Time",
        "Milliseconds",
        [MetricAggregation.AVERAGE, MetricAggregation.MINIMUM, MetricAggregation.MAXIMUM]
    )
    
    # Emit metrics
    for i in range(10):
        metrics.emit_metric("MyApp", "RequestCount", 1)
        metrics.emit_metric("MyApp", "ResponseTime", 100 + i * 10)
    
    print("Emitted 10 metric data points\n")
    
    # Query metrics
    start = datetime.now() - timedelta(minutes=10)
    end = datetime.now()
    
    results = metrics.query_metrics("MyApp", "ResponseTime", start, end)
    print(f"Query returned {len(results)} data points")
    
    # Get statistics
    stats = metrics.get_metric_statistics("MyApp", "ResponseTime", start, end)
    print(f"\nResponse Time Statistics:")
    print(f"  Count: {stats['count']}")
    print(f"  Average: {stats['average']:.2f}ms")
    print(f"  Min: {stats['minimum']:.2f}ms")
    print(f"  Max: {stats['maximum']:.2f}ms\n")


def demo_log_analytics():
    """Demonstrate log analytics"""
    print("=== Log Analytics Demo ===\n")
    
    workspace = LogAnalyticsManager("ws-12345", "my-workspace")
    
    # Ingest logs
    workspace.ingest_log(
        LogLevel.INFORMATION,
        "Application started successfully",
        "/subscriptions/.../resourceGroups/.../providers/Microsoft.Web/sites/myapp"
    )
    
    workspace.ingest_log(
        LogLevel.WARNING,
        "High memory usage detected",
        "/subscriptions/.../resourceGroups/.../providers/Microsoft.Web/sites/myapp",
        properties={"memory_percent": 85}
    )
    
    workspace.ingest_log(
        LogLevel.ERROR,
        "Database connection failed",
        "/subscriptions/.../resourceGroups/.../providers/Microsoft.Sql/servers/mydb",
        properties={"error_code": "ConnectionTimeout"}
    )
    
    print("Ingested 3 log entries\n")
    
    # Query logs
    all_logs = workspace.query_logs("* | where TimeGenerated > ago(1h)")
    print(f"Total logs: {len(all_logs)}")
    
    error_logs = workspace.query_logs("* | where Level == 'Error'")
    print(f"Error logs: {len(error_logs)}")
    
    for log in error_logs:
        print(f"  [{log.timestamp}] {log.level.value}: {log.message}\n")
    
    # Create custom table
    schema = [
        {"name": "EventId", "type": "string"},
        {"name": "EventName", "type": "string"},
        {"name": "Count", "type": "int"}
    ]
    
    table = workspace.create_custom_table("CustomEvents_CL", schema)
    print(f"Created custom table: {table['table_name']}\n")
    
    # Insert custom log
    workspace.insert_custom_log("CustomEvents_CL", {
        "EventId": "evt-001",
        "EventName": "UserLogin",
        "Count": 1
    })
    print("Inserted custom log entry\n")


def demo_alerts():
    """Demonstrate alert management"""
    print("=== Alerts Demo ===\n")
    
    alerts = AlertManager()
    
    # Create action group
    action_group = alerts.create_action_group(
        "critical-alerts",
        "CritAlerts",
        email_receivers=["admin@example.com", "ops@example.com"],
        sms_receivers=[{"name": "OnCall", "phone": "+1234567890"}]
    )
    print(f"Created action group: {action_group['name']}")
    print(f"  Email receivers: {len(action_group['email_receivers'])}")
    print(f"  SMS receivers: {len(action_group['sms_receivers'])}\n")
    
    # Create metric alert
    metric_alert = alerts.create_metric_alert(
        "high-cpu-alert",
        "Alert when CPU exceeds 80%",
        "/subscriptions/.../resourceGroups/.../providers/Microsoft.Compute/virtualMachines/vm1",
        "Microsoft.Compute/virtualMachines",
        "Percentage CPU",
        ">",
        80.0,
        AlertSeverity.WARNING,
        frequency=5,
        window_size=5,
        action_groups=["critical-alerts"]
    )
    print(f"Created metric alert: {metric_alert.name}")
    print(f"  Severity: {metric_alert.severity.name}")
    print(f"  Condition: {metric_alert.condition['metric_name']} {metric_alert.condition['operator']} {metric_alert.condition['threshold']}\n")
    
    # Create log query alert
    log_alert = alerts.create_log_query_alert(
        "error-spike-alert",
        "Alert on error log spike",
        "ws-12345",
        "Logs | where Level == 'Error' | summarize count() by bin(TimeGenerated, 5m)",
        AlertSeverity.ERROR,
        threshold=10,
        frequency=5,
        action_groups=["critical-alerts"]
    )
    print(f"Created log query alert: {log_alert.name}")
    print(f"  Severity: {log_alert.severity.name}")
    print(f"  Threshold: {log_alert.condition['threshold']} errors\n")
    
    # Trigger alert
    alert_instance = alerts.trigger_alert("high-cpu-alert")
    print(f"Triggered alert: {alert_instance['alert_id']}")
    print(f"  State: {alert_instance['state']}")
    print(f"  Fired at: {alert_instance['fired_at']}\n")
    
    # Acknowledge alert
    alerts.acknowledge_alert(alert_instance['alert_id'])
    print(f"Alert acknowledged\n")


def demo_diagnostics():
    """Demonstrate diagnostic settings"""
    print("=== Diagnostics Demo ===\n")
    
    diagnostics = DiagnosticsManager()
    
    # Create diagnostic setting
    setting = diagnostics.create_diagnostic_setting(
        "send-to-workspace",
        "/subscriptions/.../resourceGroups/.../providers/Microsoft.Web/sites/webapp1",
        workspace_id="/subscriptions/.../resourcegroups/.../providers/microsoft.operationalinsights/workspaces/ws1",
        logs=[
            {"category": "AppServiceHTTPLogs", "enabled": True},
            {"category": "AppServiceConsoleLogs", "enabled": True},
            {"category": "AppServiceAppLogs", "enabled": True}
        ],
        metrics=[
            {"category": "AllMetrics", "enabled": True}
        ]
    )
    
    print(f"Created diagnostic setting: {setting.name}")
    print(f"  Resource: {setting.resource_id.split('/')[-1]}")
    print(f"  Log categories: {len(setting.logs)}")
    print(f"  Metric categories: {len(setting.metrics)}")
    
    for log in setting.logs:
        print(f"    - {log['category']}: {log['enabled']}")
    
    print()


def demo_application_insights():
    """Demonstrate Application Insights"""
    print("=== Application Insights Demo ===\n")
    
    app_insights = ApplicationInsightsManager("ikey-12345678-1234-1234-1234-123456789012")
    
    # Track requests
    app_insights.track_request(
        "GET /api/users",
        "https://api.example.com/api/users",
        duration_ms=125.5,
        response_code=200,
        success=True
    )
    
    app_insights.track_request(
        "POST /api/orders",
        "https://api.example.com/api/orders",
        duration_ms=450.2,
        response_code=201,
        success=True
    )
    
    print(f"Tracked {len(app_insights.requests)} requests")
    for req in app_insights.requests:
        print(f"  {req['name']}: {req['response_code']} ({req['duration']:.1f}ms)")
    print()
    
    # Track dependencies
    app_insights.track_dependency(
        "SQL Query",
        "SQL",
        "sqlserver.database.windows.net/mydb",
        "SELECT * FROM Users WHERE Id = @id",
        duration_ms=45.3,
        success=True
    )
    
    app_insights.track_dependency(
        "Redis Cache",
        "Redis",
        "mycache.redis.cache.windows.net",
        "GET user:12345",
        duration_ms=5.2,
        success=True
    )
    
    print(f"Tracked {len(app_insights.dependencies)} dependencies")
    for dep in app_insights.dependencies:
        print(f"  {dep['name']} ({dep['type']}): {dep['duration']:.1f}ms")
    print()
    
    # Track exception
    app_insights.track_exception(
        "NullReferenceException",
        "Object reference not set to an instance of an object",
        "at MyApp.Controllers.UserController.Get(Int32 id)\n  at System.Web.Mvc.ActionMethodDispatcher.Execute(...)",
        properties={"user_id": "12345", "endpoint": "/api/users/12345"}
    )
    
    print(f"Tracked {len(app_insights.exceptions)} exceptions")
    for exc in app_insights.exceptions:
        print(f"  {exc['type']}: {exc['message'][:50]}...")
    print()
    
    # Track custom events
    app_insights.track_event(
        "UserLogin",
        properties={"user_id": "user123", "login_method": "OAuth"},
        measurements={"login_duration": 1.25}
    )
    
    app_insights.track_event(
        "OrderPlaced",
        properties={"order_id": "ord-456", "amount": 99.99},
        measurements={"items": 3}
    )
    
    print(f"Tracked {len(app_insights.custom_events)} custom events")
    for event in app_insights.custom_events:
        print(f"  {event['name']}: {event['properties']}")
    print()


def demo_integrated_monitoring():
    """Demonstrate integrated monitoring"""
    print("=== Integrated Monitoring Demo ===\n")
    
    monitor = AzureMonitorManager("sub-12345", "my-rg")
    
    # Create workspace
    workspace = monitor.create_log_analytics_workspace("ws-001", "production-workspace")
    print(f"Created workspace: {workspace.workspace_name}\n")
    
    # Create Application Insights
    app_insights = monitor.create_application_insights("web-app-insights", "ikey-123")
    print(f"Created Application Insights: {app_insights.instrumentation_key}\n")
    
    # Configure monitoring
    monitor.metrics.define_metric(
        "ActiveConnections",
        "MyService",
        "Active Connections",
        "Count",
        [MetricAggregation.AVERAGE, MetricAggregation.MAXIMUM]
    )
    
    # Emit sample data
    for i in range(5):
        monitor.metrics.emit_metric("MyService", "ActiveConnections", 100 + i * 5)
        workspace.ingest_log(
            LogLevel.INFORMATION,
            f"Processing batch {i+1}",
            "/subscriptions/.../providers/MyService"
        )
    
    print("Emitted metrics and logs")
    
    # Create alert
    alert = monitor.alerts.create_metric_alert(
        "connection-limit-alert",
        "Alert when connections exceed limit",
        "/subscriptions/.../providers/MyService",
        "MyService",
        "ActiveConnections",
        ">",
        150.0,
        AlertSeverity.WARNING
    )
    
    print(f"\nCreated alert: {alert.name}")
    print(f"Total alert rules: {len(monitor.alerts.alert_rules)}\n")


if __name__ == "__main__":
    print("Azure Monitor - Advanced Implementation")
    print("=" * 60)
    print()
    
    # Run all demos
    demo_metrics()
    demo_log_analytics()
    demo_alerts()
    demo_diagnostics()
    demo_application_insights()
    demo_integrated_monitoring()
    
    print("=" * 60)
    print("All demos completed successfully!")
