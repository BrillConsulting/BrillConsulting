"""
Data Observability Framework
============================

Comprehensive data pipeline monitoring and quality tracking:
- Data pipeline monitoring and health checks
- Data quality metrics and SLAs
- Anomaly detection and alerting
- Data freshness tracking
- Schema drift detection
- Performance monitoring

Author: Brill Consulting
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional
import random
import statistics


class DataPipeline:
    """Represents a data pipeline."""

    def __init__(self, pipeline_id: str, name: str, owner: str):
        """Initialize pipeline."""
        self.pipeline_id = pipeline_id
        self.name = name
        self.owner = owner
        self.status = "healthy"
        self.last_run = None
        self.metrics = []
        self.sla = {}

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "pipeline_id": self.pipeline_id,
            "name": self.name,
            "owner": self.owner,
            "status": self.status,
            "last_run": self.last_run,
            "metrics_count": len(self.metrics)
        }


class DataObservability:
    """Data observability and monitoring system."""

    def __init__(self):
        """Initialize observability system."""
        self.pipelines = {}
        self.metrics_history = []
        self.alerts = []
        self.sla_definitions = {}
        self.incidents = []

    def register_pipeline(self, pipeline_id: str, name: str, owner: str,
                         sla: Optional[Dict] = None) -> DataPipeline:
        """Register a data pipeline for monitoring."""
        print(f"Registering pipeline: {name}")

        pipeline = DataPipeline(pipeline_id, name, owner)

        if sla:
            pipeline.sla = sla

        self.pipelines[pipeline_id] = pipeline

        print(f"✓ Registered pipeline: {pipeline_id}")
        print(f"  Owner: {owner}")
        if sla:
            print(f"  SLA: {sla.get('freshness_minutes', 'N/A')} min freshness")

        return pipeline

    def record_pipeline_run(self, pipeline_id: str, status: str,
                           duration_seconds: float, rows_processed: int,
                           errors: int = 0) -> Dict:
        """Record pipeline execution metrics."""
        if pipeline_id not in self.pipelines:
            raise ValueError(f"Pipeline {pipeline_id} not registered")

        pipeline = self.pipelines[pipeline_id]

        run_record = {
            "pipeline_id": pipeline_id,
            "status": status,
            "duration_seconds": duration_seconds,
            "rows_processed": rows_processed,
            "errors": errors,
            "timestamp": datetime.now().isoformat(),
            "success": status == "success"
        }

        pipeline.metrics.append(run_record)
        pipeline.last_run = run_record["timestamp"]
        pipeline.status = "healthy" if status == "success" else "degraded"

        self.metrics_history.append(run_record)

        return run_record

    def check_data_freshness(self, pipeline_id: str) -> Dict:
        """Check data freshness against SLA."""
        print(f"Checking data freshness: {pipeline_id}")

        if pipeline_id not in self.pipelines:
            return {"error": "Pipeline not found"}

        pipeline = self.pipelines[pipeline_id]

        if not pipeline.last_run:
            return {
                "pipeline_id": pipeline_id,
                "status": "no_data",
                "message": "No pipeline runs recorded"
            }

        last_run_time = datetime.fromisoformat(pipeline.last_run)
        age_minutes = (datetime.now() - last_run_time).total_seconds() / 60

        sla_freshness = pipeline.sla.get("freshness_minutes", 60)
        is_fresh = age_minutes <= sla_freshness

        result = {
            "pipeline_id": pipeline_id,
            "last_run": pipeline.last_run,
            "age_minutes": age_minutes,
            "sla_minutes": sla_freshness,
            "is_fresh": is_fresh,
            "status": "fresh" if is_fresh else "stale"
        }

        if not is_fresh:
            print(f"⚠ Data is stale: {age_minutes:.1f} minutes old (SLA: {sla_freshness})")
        else:
            print(f"✓ Data is fresh: {age_minutes:.1f} minutes old")

        return result

    def calculate_quality_metrics(self, pipeline_id: str,
                                 window_hours: int = 24) -> Dict:
        """Calculate data quality metrics for a pipeline."""
        print(f"Calculating quality metrics for {pipeline_id}")

        if pipeline_id not in self.pipelines:
            return {"error": "Pipeline not found"}

        pipeline = self.pipelines[pipeline_id]

        # Get recent metrics
        cutoff_time = datetime.now() - timedelta(hours=window_hours)
        recent_metrics = [
            m for m in pipeline.metrics
            if datetime.fromisoformat(m["timestamp"]) > cutoff_time
        ]

        if not recent_metrics:
            return {
                "pipeline_id": pipeline_id,
                "message": "No recent metrics"
            }

        total_runs = len(recent_metrics)
        successful_runs = sum(1 for m in recent_metrics if m["success"])
        failed_runs = total_runs - successful_runs

        durations = [m["duration_seconds"] for m in recent_metrics]
        total_rows = sum(m["rows_processed"] for m in recent_metrics)
        total_errors = sum(m["errors"] for m in recent_metrics)

        metrics = {
            "pipeline_id": pipeline_id,
            "window_hours": window_hours,
            "total_runs": total_runs,
            "successful_runs": successful_runs,
            "failed_runs": failed_runs,
            "success_rate": (successful_runs / total_runs * 100) if total_runs > 0 else 0,
            "avg_duration_seconds": statistics.mean(durations) if durations else 0,
            "min_duration_seconds": min(durations) if durations else 0,
            "max_duration_seconds": max(durations) if durations else 0,
            "total_rows_processed": total_rows,
            "total_errors": total_errors,
            "error_rate": (total_errors / total_rows * 100) if total_rows > 0 else 0,
            "calculated_at": datetime.now().isoformat()
        }

        print(f"✓ Quality metrics calculated")
        print(f"  Success rate: {metrics['success_rate']:.1f}%")
        print(f"  Avg duration: {metrics['avg_duration_seconds']:.2f}s")
        print(f"  Error rate: {metrics['error_rate']:.4f}%")

        return metrics

    def detect_anomalies(self, pipeline_id: str, metric: str = "duration_seconds",
                        threshold: float = 3.0) -> Dict:
        """Detect anomalies in pipeline metrics."""
        print(f"Detecting anomalies in {pipeline_id} ({metric})")

        if pipeline_id not in self.pipelines:
            return {"error": "Pipeline not found"}

        pipeline = self.pipelines[pipeline_id]

        if len(pipeline.metrics) < 10:
            return {
                "pipeline_id": pipeline_id,
                "message": "Insufficient data for anomaly detection"
            }

        values = [m[metric] for m in pipeline.metrics if metric in m]

        if not values:
            return {"error": f"Metric {metric} not found"}

        mean = statistics.mean(values)
        stdev = statistics.stdev(values) if len(values) > 1 else 0

        # Z-score based anomaly detection
        anomalies = []
        for i, value in enumerate(values):
            if stdev > 0:
                z_score = abs((value - mean) / stdev)
                if z_score > threshold:
                    anomalies.append({
                        "index": i,
                        "value": value,
                        "z_score": z_score,
                        "deviation": value - mean
                    })

        result = {
            "pipeline_id": pipeline_id,
            "metric": metric,
            "mean": mean,
            "stdev": stdev,
            "threshold": threshold,
            "total_points": len(values),
            "anomalies_count": len(anomalies),
            "anomalies": anomalies[:5]  # Top 5
        }

        if anomalies:
            print(f"⚠ Found {len(anomalies)} anomalies")
        else:
            print(f"✓ No anomalies detected")

        return result

    def check_sla_compliance(self, pipeline_id: str) -> Dict:
        """Check SLA compliance for a pipeline."""
        print(f"Checking SLA compliance: {pipeline_id}")

        if pipeline_id not in self.pipelines:
            return {"error": "Pipeline not found"}

        pipeline = self.pipelines[pipeline_id]

        if not pipeline.sla:
            return {
                "pipeline_id": pipeline_id,
                "message": "No SLA defined"
            }

        compliance_checks = []

        # Check freshness
        if "freshness_minutes" in pipeline.sla:
            freshness = self.check_data_freshness(pipeline_id)
            compliance_checks.append({
                "check": "freshness",
                "passed": freshness.get("is_fresh", False),
                "details": freshness
            })

        # Check success rate
        if "success_rate_percent" in pipeline.sla:
            metrics = self.calculate_quality_metrics(pipeline_id, window_hours=24)
            required_rate = pipeline.sla["success_rate_percent"]
            actual_rate = metrics.get("success_rate", 0)
            compliance_checks.append({
                "check": "success_rate",
                "passed": actual_rate >= required_rate,
                "required": required_rate,
                "actual": actual_rate
            })

        # Check error rate
        if "max_error_rate_percent" in pipeline.sla:
            metrics = self.calculate_quality_metrics(pipeline_id, window_hours=24)
            max_rate = pipeline.sla["max_error_rate_percent"]
            actual_rate = metrics.get("error_rate", 0)
            compliance_checks.append({
                "check": "error_rate",
                "passed": actual_rate <= max_rate,
                "max_allowed": max_rate,
                "actual": actual_rate
            })

        all_passed = all(check["passed"] for check in compliance_checks)

        result = {
            "pipeline_id": pipeline_id,
            "compliant": all_passed,
            "checks": compliance_checks,
            "checked_at": datetime.now().isoformat()
        }

        status = "COMPLIANT" if all_passed else "NON-COMPLIANT"
        print(f"✓ SLA Status: {status}")
        print(f"  Checks: {sum(1 for c in compliance_checks if c['passed'])}/{len(compliance_checks)} passed")

        return result

    def create_alert(self, pipeline_id: str, severity: str,
                    message: str, details: Optional[Dict] = None) -> Dict:
        """Create an alert for a pipeline issue."""
        alert = {
            "alert_id": f"alert_{len(self.alerts) + 1}",
            "pipeline_id": pipeline_id,
            "severity": severity,
            "message": message,
            "details": details or {},
            "created_at": datetime.now().isoformat(),
            "status": "open"
        }

        self.alerts.append(alert)

        print(f"⚠ Alert created: {severity.upper()}")
        print(f"  Pipeline: {pipeline_id}")
        print(f"  Message: {message}")

        return alert

    def monitor_pipeline(self, pipeline_id: str) -> Dict:
        """Comprehensive pipeline monitoring check."""
        print(f"\nMonitoring pipeline: {pipeline_id}")
        print("-"*50)

        if pipeline_id not in self.pipelines:
            return {"error": "Pipeline not found"}

        results = {
            "pipeline_id": pipeline_id,
            "monitored_at": datetime.now().isoformat(),
            "checks": {}
        }

        # Freshness check
        freshness = self.check_data_freshness(pipeline_id)
        results["checks"]["freshness"] = freshness

        if freshness.get("status") == "stale":
            self.create_alert(
                pipeline_id,
                "warning",
                "Data freshness SLA violated",
                freshness
            )

        # Quality metrics
        metrics = self.calculate_quality_metrics(pipeline_id)
        results["checks"]["quality_metrics"] = metrics

        # Anomaly detection
        anomalies = self.detect_anomalies(pipeline_id)
        results["checks"]["anomalies"] = anomalies

        if anomalies.get("anomalies_count", 0) > 0:
            self.create_alert(
                pipeline_id,
                "info",
                f"Detected {anomalies['anomalies_count']} anomalies",
                anomalies
            )

        # SLA compliance
        sla = self.check_sla_compliance(pipeline_id)
        results["checks"]["sla_compliance"] = sla

        if not sla.get("compliant", True):
            self.create_alert(
                pipeline_id,
                "critical",
                "SLA compliance violated",
                sla
            )

        return results

    def get_open_alerts(self, severity: Optional[str] = None) -> List[Dict]:
        """Get open alerts, optionally filtered by severity."""
        alerts = [a for a in self.alerts if a["status"] == "open"]

        if severity:
            alerts = [a for a in alerts if a["severity"] == severity]

        return alerts

    def resolve_alert(self, alert_id: str, resolution: str) -> Dict:
        """Resolve an alert."""
        for alert in self.alerts:
            if alert["alert_id"] == alert_id:
                alert["status"] = "resolved"
                alert["resolution"] = resolution
                alert["resolved_at"] = datetime.now().isoformat()
                print(f"✓ Alert {alert_id} resolved")
                return alert

        return {"error": "Alert not found"}

    def create_incident(self, pipeline_id: str, title: str,
                       severity: str, description: str) -> Dict:
        """Create an incident for a pipeline issue."""
        incident = {
            "incident_id": f"INC-{len(self.incidents) + 1:04d}",
            "pipeline_id": pipeline_id,
            "title": title,
            "severity": severity,
            "description": description,
            "status": "open",
            "created_at": datetime.now().isoformat(),
            "updates": []
        }

        self.incidents.append(incident)

        print(f"⚠ Incident created: {incident['incident_id']}")
        print(f"  Title: {title}")
        print(f"  Severity: {severity}")

        return incident

    def generate_health_report(self) -> Dict:
        """Generate comprehensive health report."""
        print("\nGenerating Health Report...")
        print("="*50)

        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_pipelines": len(self.pipelines),
                "healthy_pipelines": 0,
                "degraded_pipelines": 0,
                "failed_pipelines": 0,
                "open_alerts": len([a for a in self.alerts if a["status"] == "open"]),
                "open_incidents": len([i for i in self.incidents if i["status"] == "open"])
            },
            "pipelines": []
        }

        # Count pipeline statuses
        for pipeline in self.pipelines.values():
            if pipeline.status == "healthy":
                report["summary"]["healthy_pipelines"] += 1
            elif pipeline.status == "degraded":
                report["summary"]["degraded_pipelines"] += 1
            else:
                report["summary"]["failed_pipelines"] += 1

            report["pipelines"].append(pipeline.to_dict())

        # Alert breakdown
        report["alerts_by_severity"] = {
            "critical": len([a for a in self.alerts if a["severity"] == "critical" and a["status"] == "open"]),
            "warning": len([a for a in self.alerts if a["severity"] == "warning" and a["status"] == "open"]),
            "info": len([a for a in self.alerts if a["severity"] == "info" and a["status"] == "open"])
        }

        print(f"Total Pipelines: {report['summary']['total_pipelines']}")
        print(f"Healthy: {report['summary']['healthy_pipelines']}")
        print(f"Degraded: {report['summary']['degraded_pipelines']}")
        print(f"Open Alerts: {report['summary']['open_alerts']}")

        return report


def demo():
    """Demo Data Observability."""
    print("Data Observability Demo")
    print("="*50)

    obs = DataObservability()

    # 1. Register pipelines
    print("\n1. Registering Pipelines")
    print("-"*50)

    obs.register_pipeline(
        "sales_etl",
        "Sales ETL Pipeline",
        "data_team",
        sla={
            "freshness_minutes": 60,
            "success_rate_percent": 95,
            "max_error_rate_percent": 1.0
        }
    )

    obs.register_pipeline(
        "user_analytics",
        "User Analytics Pipeline",
        "analytics_team",
        sla={
            "freshness_minutes": 30,
            "success_rate_percent": 98
        }
    )

    # 2. Record pipeline runs
    print("\n2. Recording Pipeline Runs")
    print("-"*50)

    # Simulate multiple runs
    for i in range(10):
        duration = 45 + random.uniform(-10, 15)
        rows = 100000 + random.randint(-5000, 10000)
        errors = random.randint(0, 50) if i < 9 else 500  # Last run has more errors

        status = "success" if i < 9 else "failed"

        obs.record_pipeline_run(
            "sales_etl",
            status=status,
            duration_seconds=duration,
            rows_processed=rows,
            errors=errors
        )

    print(f"✓ Recorded 10 pipeline runs for sales_etl")

    # 3. Check data freshness
    print("\n3. Checking Data Freshness")
    print("-"*50)

    freshness = obs.check_data_freshness("sales_etl")

    # 4. Calculate quality metrics
    print("\n4. Calculating Quality Metrics")
    print("-"*50)

    metrics = obs.calculate_quality_metrics("sales_etl", window_hours=24)

    # 5. Detect anomalies
    print("\n5. Detecting Anomalies")
    print("-"*50)

    anomalies = obs.detect_anomalies("sales_etl", metric="duration_seconds")

    # 6. Check SLA compliance
    print("\n6. Checking SLA Compliance")
    print("-"*50)

    compliance = obs.check_sla_compliance("sales_etl")

    # 7. Create alerts
    print("\n7. Creating Alerts")
    print("-"*50)

    if not compliance["compliant"]:
        for check in compliance["checks"]:
            if not check["passed"]:
                obs.create_alert(
                    "sales_etl",
                    "warning",
                    f"SLA check failed: {check['check']}",
                    check
                )

    # 8. Monitor pipeline
    print("\n8. Comprehensive Pipeline Monitoring")
    print("-"*50)

    monitoring = obs.monitor_pipeline("sales_etl")

    # 9. Review alerts
    print("\n9. Reviewing Open Alerts")
    print("-"*50)

    open_alerts = obs.get_open_alerts()
    print(f"Open alerts: {len(open_alerts)}")

    for alert in open_alerts[:3]:
        print(f"  [{alert['severity'].upper()}] {alert['message']}")

    # 10. Create incident
    print("\n10. Creating Incident")
    print("-"*50)

    incident = obs.create_incident(
        "sales_etl",
        "High error rate detected",
        "high",
        "Last pipeline run had 500 errors, significantly higher than normal"
    )

    # 11. Generate health report
    print("\n11. Health Report")
    print("-"*50)

    report = obs.generate_health_report()

    print(f"\nAlert Breakdown:")
    for severity, count in report["alerts_by_severity"].items():
        if count > 0:
            print(f"  {severity.capitalize()}: {count}")

    print("\n✓ Data Observability Demo Complete!")


if __name__ == '__main__':
    demo()
