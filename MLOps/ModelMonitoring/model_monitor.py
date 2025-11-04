"""
Model Monitoring System
=======================

Monitor ML models in production:
- Performance metrics tracking
- Data drift detection
- Prediction distribution monitoring
- Alert system
- Dashboard integration

Author: Brill Consulting
"""

import numpy as np
from typing import Dict, List
from datetime import datetime
from scipy import stats


class ModelMonitor:
    """Model monitoring system."""

    def __init__(self, baseline_data: np.ndarray):
        """Initialize with baseline data."""
        self.baseline_mean = baseline_data.mean(axis=0)
        self.baseline_std = baseline_data.std(axis=0)
        self.baseline_dist = baseline_data
        self.metrics_history = []

    def detect_data_drift(self, new_data: np.ndarray, threshold: float = 0.05) -> Dict:
        """Detect data drift using KS test."""
        drift_detected = {}

        for i in range(new_data.shape[1]):
            statistic, p_value = stats.ks_2samp(
                self.baseline_dist[:, i],
                new_data[:, i]
            )

            drift_detected[f"feature_{i}"] = {
                "p_value": p_value,
                "drift": p_value < threshold,
                "statistic": statistic
            }

        return drift_detected

    def monitor_predictions(self, predictions: np.ndarray) -> Dict:
        """Monitor prediction distribution."""
        metrics = {
            "mean": float(predictions.mean()),
            "std": float(predictions.std()),
            "min": float(predictions.min()),
            "max": float(predictions.max()),
            "timestamp": datetime.now().isoformat()
        }

        self.metrics_history.append(metrics)
        return metrics

    def check_performance_degradation(self, current_accuracy: float,
                                     baseline_accuracy: float,
                                     threshold: float = 0.05) -> Dict:
        """Check for performance degradation."""
        degradation = baseline_accuracy - current_accuracy
        alert = degradation > threshold

        return {
            "current_accuracy": current_accuracy,
            "baseline_accuracy": baseline_accuracy,
            "degradation": degradation,
            "alert": alert
        }

    def generate_report(self) -> Dict:
        """Generate monitoring report."""
        if not self.metrics_history:
            return {"error": "No metrics available"}

        recent_metrics = self.metrics_history[-100:]

        report = {
            "total_predictions": len(self.metrics_history),
            "recent_stats": {
                "avg_prediction": np.mean([m["mean"] for m in recent_metrics]),
                "prediction_stability": np.std([m["std"] for m in recent_metrics])
            },
            "timestamp": datetime.now().isoformat()
        }

        return report


def demo():
    """Demo monitoring."""
    print("Model Monitoring Demo")
    print("="*50)

    # Baseline data
    baseline = np.random.randn(1000, 5)

    # Initialize monitor
    monitor = ModelMonitor(baseline)

    # 1. Data drift detection
    print("\n1. Data Drift Detection")
    print("-"*50)

    # Normal data (no drift)
    new_data_normal = np.random.randn(500, 5)
    drift_normal = monitor.detect_data_drift(new_data_normal)

    print("Normal data:")
    drifted_features = [k for k, v in drift_normal.items() if v["drift"]]
    print(f"  Drifted features: {len(drifted_features)}/{len(drift_normal)}")

    # Shifted data (drift)
    new_data_shifted = np.random.randn(500, 5) + 2.0
    drift_shifted = monitor.detect_data_drift(new_data_shifted)

    print("Shifted data:")
    drifted_features = [k for k, v in drift_shifted.items() if v["drift"]]
    print(f"  Drifted features: {len(drifted_features)}/{len(drift_shifted)}")

    # 2. Prediction monitoring
    print("\n2. Prediction Monitoring")
    print("-"*50)

    for i in range(5):
        predictions = np.random.rand(100)
        metrics = monitor.monitor_predictions(predictions)
        print(f"Batch {i+1}: mean={metrics['mean']:.4f}, std={metrics['std']:.4f}")

    # 3. Performance degradation
    print("\n3. Performance Degradation Check")
    print("-"*50)

    baseline_acc = 0.90
    current_acc = 0.85

    degradation = monitor.check_performance_degradation(current_acc, baseline_acc)
    print(f"Current accuracy: {degradation['current_accuracy']:.4f}")
    print(f"Baseline accuracy: {degradation['baseline_accuracy']:.4f}")
    print(f"Degradation: {degradation['degradation']:.4f}")
    print(f"Alert: {degradation['alert']}")

    # 4. Generate report
    print("\n4. Monitoring Report")
    print("-"*50)

    report = monitor.generate_report()
    print(f"Total predictions monitored: {report['total_predictions']}")
    print(f"Avg prediction: {report['recent_stats']['avg_prediction']:.4f}")
    print(f"Prediction stability: {report['recent_stats']['prediction_stability']:.4f}")

    print("\n✓ Monitoring Demo Complete!")


if __name__ == '__main__':
    demo()
