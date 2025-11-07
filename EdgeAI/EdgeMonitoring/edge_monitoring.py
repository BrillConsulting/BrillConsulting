"""
Edge Monitoring
===============

Device health and performance monitoring

Author: Brill Consulting
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import time
import numpy as np


class Severity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class DeviceMetrics:
    """Device metrics snapshot."""
    device_id: str
    cpu_percent: float
    memory_mb: float
    temperature_c: float
    disk_percent: float
    timestamp: str


@dataclass
class PerformanceStats:
    """Performance statistics."""
    avg_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    fps: float
    throughput: float


class DeviceMonitor:
    """Monitor device health."""

    def __init__(
        self,
        device_id: str,
        metrics_endpoint: str,
        alert_endpoint: Optional[str] = None
    ):
        self.device_id = device_id
        self.metrics_endpoint = metrics_endpoint
        self.alert_endpoint = alert_endpoint
        self.is_running = False

        print(f"📊 Device Monitor initialized")
        print(f"   Device: {device_id}")
        print(f"   Metrics endpoint: {metrics_endpoint}")

    def start(self, interval_sec: int = 10) -> None:
        """Start monitoring."""
        print(f"\n🚀 Starting monitoring")
        print(f"   Interval: {interval_sec}s")

        self.is_running = True

        # Simulate monitoring loop
        for i in range(3):
            metrics = self.get_metrics()
            print(f"\n   Metrics collected:")
            print(f"   CPU: {metrics.cpu_percent}%")
            print(f"   Memory: {metrics.memory_mb:.0f}MB")
            print(f"   Temperature: {metrics.temperature_c}°C")

            time.sleep(1)  # Simulated interval

        print(f"\n   ✓ Monitoring started")

    def get_metrics(self) -> DeviceMetrics:
        """Get current device metrics."""
        # Simulate metric collection
        return DeviceMetrics(
            device_id=self.device_id,
            cpu_percent=np.random.uniform(30, 80),
            memory_mb=np.random.uniform(1000, 3000),
            temperature_c=np.random.uniform(45, 70),
            disk_percent=np.random.uniform(40, 60),
            timestamp=datetime.now().isoformat()
        )

    def stop(self) -> None:
        """Stop monitoring."""
        self.is_running = False
        print(f"⏹️  Monitoring stopped")


class PerformanceTracker:
    """Track inference performance."""

    def __init__(self):
        self.latencies: List[float] = []
        self.start_time = None

        print(f"⏱️  Performance Tracker initialized")

    def track_inference(self):
        """Context manager for tracking inference."""
        class InferenceTracker:
            def __init__(self, tracker):
                self.tracker = tracker

            def __enter__(self):
                self.start = time.time()
                return self

            def __exit__(self, *args):
                latency_ms = (time.time() - self.start) * 1000
                self.tracker.latencies.append(latency_ms)

        return InferenceTracker(self)

    def get_statistics(self) -> PerformanceStats:
        """Get performance statistics."""
        if not self.latencies:
            return PerformanceStats(0, 0, 0, 0, 0)

        latencies = np.array(self.latencies)

        return PerformanceStats(
            avg_latency_ms=float(np.mean(latencies)),
            p95_latency_ms=float(np.percentile(latencies, 95)),
            p99_latency_ms=float(np.percentile(latencies, 99)),
            fps=1000.0 / np.mean(latencies),
            throughput=len(latencies)
        )


class AlertManager:
    """Manage alerts."""

    def __init__(self):
        self.rules: List[Dict] = []
        print(f"🚨 Alert Manager initialized")

    def add_rule(
        self,
        name: str,
        metric: str,
        threshold: float,
        operator: str = "greater_than",
        severity: str = "warning"
    ) -> None:
        """Add alert rule."""
        rule = {
            "name": name,
            "metric": metric,
            "threshold": threshold,
            "operator": operator,
            "severity": Severity(severity)
        }

        self.rules.append(rule)
        print(f"\n🔔 Alert rule added: {name}")
        print(f"   Metric: {metric}")
        print(f"   Threshold: {threshold}")
        print(f"   Severity: {severity}")

    def check_and_notify(self, metrics: Optional[DeviceMetrics] = None) -> None:
        """Check rules and send alerts."""
        print(f"\n🔍 Checking alert rules")

        # Simulate metric values
        test_metrics = {
            "temperature_c": 72,
            "cpu_percent": 85,
            "fps": 12
        }

        triggered = []

        for rule in self.rules:
            metric_value = test_metrics.get(rule["metric"], 0)
            threshold = rule["threshold"]

            if rule["operator"] == "greater_than":
                if metric_value > threshold:
                    triggered.append(rule)
            elif rule["operator"] == "less_than":
                if metric_value < threshold:
                    triggered.append(rule)

        if triggered:
            print(f"   ⚠️  {len(triggered)} alert(s) triggered")
            for alert in triggered:
                print(f"   - {alert['name']} ({alert['severity'].value})")
        else:
            print(f"   ✓ No alerts")


class GrafanaDashboard:
    """Grafana dashboard integration."""

    def __init__(self, grafana_url: str, api_key: str):
        self.grafana_url = grafana_url
        self.api_key = api_key

        print(f"📈 Grafana Dashboard initialized")
        print(f"   URL: {grafana_url}")

    def create(
        self,
        name: str,
        panels: List[Dict[str, str]]
    ) -> None:
        """Create dashboard."""
        print(f"\n📊 Creating dashboard: {name}")
        print(f"   Panels: {len(panels)}")

        for panel in panels:
            print(f"   - {panel['title']}: {panel['metric']}")

        print(f"   ✓ Dashboard created")


def demo():
    """Demonstrate edge monitoring."""
    print("=" * 60)
    print("Edge Monitoring Demo")
    print("=" * 60)

    # Device monitoring
    print(f"\n{'='*60}")
    print("Device Monitoring")
    print(f"{'='*60}")

    monitor = DeviceMonitor(
        device_id="jetson_001",
        metrics_endpoint="http://prometheus:9090",
        alert_endpoint="http://alertmanager:9093"
    )

    monitor.start(interval_sec=10)

    # Performance tracking
    print(f"\n{'='*60}")
    print("Performance Tracking")
    print(f"{'='*60}")

    tracker = PerformanceTracker()

    # Simulate inferences
    for i in range(100):
        with tracker.track_inference():
            time.sleep(0.02)  # Simulate inference

    stats = tracker.get_statistics()
    print(f"\n   Performance Statistics:")
    print(f"   Avg latency: {stats.avg_latency_ms:.1f}ms")
    print(f"   P95 latency: {stats.p95_latency_ms:.1f}ms")
    print(f"   FPS: {stats.fps:.1f}")

    # Alerting
    print(f"\n{'='*60}")
    print("Alert Management")
    print(f"{'='*60}")

    alerts = AlertManager()

    alerts.add_rule(
        name="high_temperature",
        metric="temperature_c",
        threshold=75,
        operator="greater_than",
        severity="critical"
    )

    alerts.add_rule(
        name="low_fps",
        metric="fps",
        threshold=15,
        operator="less_than",
        severity="warning"
    )

    alerts.check_and_notify()

    # Grafana dashboard
    print(f"\n{'='*60}")
    print("Grafana Dashboard")
    print(f"{'='*60}")

    dashboard = GrafanaDashboard(
        grafana_url="http://grafana:3000",
        api_key="secret"
    )

    dashboard.create(
        name="Edge Fleet Overview",
        panels=[
            {"title": "CPU Usage", "metric": "cpu_percent"},
            {"title": "Temperature", "metric": "temperature_c"},
            {"title": "Inference Latency", "metric": "latency_ms"},
            {"title": "FPS", "metric": "fps"}
        ]
    )


if __name__ == "__main__":
    demo()
