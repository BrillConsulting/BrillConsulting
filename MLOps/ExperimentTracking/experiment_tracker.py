"""
Experiment Tracking System
===========================

Track ML experiments, parameters, and results:
- Parameter logging
- Metrics tracking
- Artifact management
- Experiment comparison
- Visualization
- MLflow-like interface

Author: Brill Consulting
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import pickle


class ExperimentTracker:
    """Experiment tracking system."""

    def __init__(self, experiment_name: str, tracking_dir: str = "./experiments"):
        """Initialize experiment tracker."""
        self.experiment_name = experiment_name
        self.tracking_dir = Path(tracking_dir)
        self.tracking_dir.mkdir(exist_ok=True)

        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.tracking_dir / experiment_name / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.params = {}
        self.metrics = {}
        self.artifacts = {}

    def log_param(self, key: str, value: Any):
        """Log a parameter."""
        self.params[key] = value
        print(f"Logged param: {key}={value}")

    def log_params(self, params: Dict[str, Any]):
        """Log multiple parameters."""
        self.params.update(params)
        print(f"Logged {len(params)} params")

    def log_metric(self, key: str, value: float, step: Optional[int] = None):
        """Log a metric."""
        if key not in self.metrics:
            self.metrics[key] = []

        self.metrics[key].append({
            "value": value,
            "step": step if step is not None else len(self.metrics[key]),
            "timestamp": datetime.now().isoformat()
        })

        print(f"Logged metric: {key}={value}")

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log multiple metrics."""
        for key, value in metrics.items():
            self.log_metric(key, value, step)

    def log_artifact(self, name: str, artifact: Any):
        """Log an artifact."""
        artifact_path = self.run_dir / f"{name}.pkl"

        with open(artifact_path, 'wb') as f:
            pickle.dump(artifact, f)

        self.artifacts[name] = str(artifact_path)
        print(f"Logged artifact: {name} -> {artifact_path}")

    def save_experiment(self):
        """Save experiment data."""
        experiment_data = {
            "experiment_name": self.experiment_name,
            "run_id": self.run_id,
            "params": self.params,
            "metrics": self.metrics,
            "artifacts": self.artifacts,
            "timestamp": datetime.now().isoformat()
        }

        metadata_path = self.run_dir / "experiment.json"
        with open(metadata_path, 'w') as f:
            json.dump(experiment_data, f, indent=2)

        print(f"✓ Saved experiment to {metadata_path}")

    def get_metric_history(self, metric_name: str) -> list:
        """Get history of a metric."""
        return self.metrics.get(metric_name, [])

    def get_best_metric(self, metric_name: str, mode: str = "max") -> Dict:
        """Get best value of a metric."""
        history = self.get_metric_history(metric_name)

        if not history:
            return {}

        if mode == "max":
            best = max(history, key=lambda x: x["value"])
        else:
            best = min(history, key=lambda x: x["value"])

        return best

    @staticmethod
    def load_experiment(experiment_path: str) -> Dict:
        """Load experiment data."""
        with open(experiment_path, 'r') as f:
            data = json.load(f)

        return data

    @staticmethod
    def compare_experiments(experiment_paths: list) -> Dict:
        """Compare multiple experiments."""
        experiments = []

        for path in experiment_paths:
            exp_data = ExperimentTracker.load_experiment(path)
            experiments.append(exp_data)

        comparison = {
            "experiments": [exp["run_id"] for exp in experiments],
            "params": {},
            "metrics": {}
        }

        # Compare params
        all_param_keys = set()
        for exp in experiments:
            all_param_keys.update(exp["params"].keys())

        for key in all_param_keys:
            comparison["params"][key] = [
                exp["params"].get(key, None) for exp in experiments
            ]

        # Compare metrics
        all_metric_keys = set()
        for exp in experiments:
            all_metric_keys.update(exp["metrics"].keys())

        for key in all_metric_keys:
            values = []
            for exp in experiments:
                metric_history = exp["metrics"].get(key, [])
                if metric_history:
                    values.append(metric_history[-1]["value"])
                else:
                    values.append(None)

            comparison["metrics"][key] = values

        return comparison


def demo():
    """Demo experiment tracking."""
    print("Experiment Tracking Demo")
    print("="*50 + "\n")

    # Create experiment
    tracker = ExperimentTracker("sentiment_classification")

    # Log parameters
    print("1. Logging Parameters")
    print("-"*50)
    tracker.log_params({
        "model": "RandomForest",
        "n_estimators": 100,
        "max_depth": 10,
        "learning_rate": 0.01
    })

    # Log metrics over epochs
    print("\n2. Logging Metrics")
    print("-"*50)
    for epoch in range(5):
        metrics = {
            "train_loss": 2.0 - epoch * 0.3,
            "val_loss": 2.2 - epoch * 0.25,
            "train_accuracy": 0.5 + epoch * 0.08,
            "val_accuracy": 0.45 + epoch * 0.07
        }
        tracker.log_metrics(metrics, step=epoch)

    # Log artifacts
    print("\n3. Logging Artifacts")
    print("-"*50)
    model = {"type": "classifier", "trained": True}
    tracker.log_artifact("model", model)

    # Get metric history
    print("\n4. Metric History")
    print("-"*50)
    val_acc_history = tracker.get_metric_history("val_accuracy")
    print(f"Validation accuracy history: {len(val_acc_history)} values")
    print(f"Final accuracy: {val_acc_history[-1]['value']:.4f}")

    # Get best metric
    print("\n5. Best Metric")
    print("-"*50)
    best = tracker.get_best_metric("val_accuracy", mode="max")
    print(f"Best validation accuracy: {best['value']:.4f} at step {best['step']}")

    # Save experiment
    print("\n6. Saving Experiment")
    print("-"*50)
    tracker.save_experiment()

    print("\n✓ Experiment Tracking Demo Complete!")


if __name__ == '__main__':
    demo()
