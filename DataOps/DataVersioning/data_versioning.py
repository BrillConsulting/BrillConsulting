"""
Data Versioning & Pipeline Management
======================================

Production data versioning and ML pipeline management using DVC.

Author: Brill Consulting
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import os


class Stage(Enum):
    """Model stages."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"


@dataclass
class ModelMetadata:
    """Model metadata."""
    name: str
    version: str
    stage: str
    metrics: Dict[str, float]
    params: Dict[str, Any]
    tags: List[str]


@dataclass
class PipelineResult:
    """Pipeline execution result."""
    success: bool
    stages_executed: List[str]
    execution_time: float
    metrics: Dict[str, float]


@dataclass
class LineageInfo:
    """Data lineage information."""
    training_data: str
    code_commit: str
    params: Dict[str, Any]
    created_at: str


class DVCTracker:
    """DVC-based data and model tracking."""

    def __init__(
        self,
        project_dir: str = ".",
        remote: str = "storage"
    ):
        self.project_dir = project_dir
        self.remote = remote

        print(f"📦 DVC Tracker initialized")
        print(f"   Project: {project_dir}")
        print(f"   Remote: {remote}")

    def track_data(
        self,
        path: str,
        message: str = "Track data"
    ) -> None:
        """Track data file with DVC."""
        print(f"\n📦 Tracking data: {path}")
        print(f"   Message: {message}")

        # Simulate DVC add
        print(f"   $ dvc add {path}")
        print(f"   ✓ Created {path}.dvc")

        # Git commit
        print(f"   $ git add {path}.dvc")
        print(f"   $ git commit -m \"{message}\"")
        print(f"   ✓ Committed to Git")

    def track_model(
        self,
        path: str,
        metrics: Dict[str, float],
        message: str = "Track model"
    ) -> None:
        """Track model with metrics."""
        print(f"\n🤖 Tracking model: {path}")
        print(f"   Message: {message}")
        print(f"   Metrics:")
        for metric, value in metrics.items():
            print(f"   - {metric}: {value:.4f}")

        # Track with DVC
        print(f"   $ dvc add {path}")
        print(f"   ✓ Model tracked")

    def add(self, path: str) -> None:
        """Add file to DVC tracking."""
        print(f"\n📦 Adding to DVC: {path}")
        print(f"   $ dvc add {path}")
        print(f"   ✓ Created {path}.dvc")

    def commit(self, message: str) -> None:
        """Commit DVC changes to Git."""
        print(f"\n💾 Committing to Git")
        print(f"   $ git commit -m \"{message}\"")
        print(f"   ✓ Committed")

    def push(self) -> None:
        """Push data to remote storage."""
        print(f"\n☁️  Pushing to remote: {self.remote}")
        print(f"   $ dvc push")
        print(f"   ✓ Data pushed to remote")

    def pull(self) -> None:
        """Pull data from remote storage."""
        print(f"\n☁️  Pulling from remote: {self.remote}")
        print(f"   $ dvc pull")
        print(f"   ✓ Data pulled from remote")

    def checkout_version(self, tag: str) -> None:
        """Checkout specific version."""
        print(f"\n🔄 Checking out version: {tag}")
        print(f"   $ git checkout {tag}")
        print(f"   $ dvc checkout")
        print(f"   ✓ Checked out {tag}")


class PipelineManager:
    """DVC pipeline management."""

    def __init__(self, pipeline_file: str = "dvc.yaml"):
        self.pipeline_file = pipeline_file
        self.stages = []

        print(f"⚙️  Pipeline Manager")
        print(f"   Pipeline file: {pipeline_file}")

    def run(self) -> PipelineResult:
        """Run entire pipeline."""
        print(f"\n▶️  Running pipeline")
        print(f"   $ dvc repro")

        # Simulate stages
        stages = ["preprocess", "train", "evaluate"]

        for stage in stages:
            print(f"\n   Stage: {stage}")
            print(f"   ✓ Completed")

        result = PipelineResult(
            success=True,
            stages_executed=stages,
            execution_time=125.3,
            metrics={"accuracy": 0.95, "f1": 0.93}
        )

        print(f"\n   Pipeline completed in {result.execution_time:.1f}s")
        return result

    def run_stage(self, stage_name: str) -> None:
        """Run specific pipeline stage."""
        print(f"\n▶️  Running stage: {stage_name}")
        print(f"   $ dvc repro {stage_name}")
        print(f"   ✓ Stage completed")

    def reproduce(self) -> None:
        """Reproduce pipeline."""
        print(f"\n🔄 Reproducing pipeline")
        print(f"   $ dvc repro")
        print(f"   ✓ Pipeline reproduced")

    def show_dag(self) -> None:
        """Show pipeline DAG."""
        print(f"\n📊 Pipeline DAG")
        print(f"   $ dvc dag")
        print(f"""
        +-------------+
        | preprocess  |
        +-------------+
              |
              v
        +-------------+
        |    train    |
        +-------------+
              |
              v
        +-------------+
        |  evaluate   |
        +-------------+
        """)


class ExperimentTracker:
    """Track ML experiments."""

    def __init__(self):
        self.current_run = None
        self.runs = []

        print(f"🧪 Experiment Tracker initialized")

    def start_run(self, name: str) -> 'ExperimentTracker':
        """Start experiment run."""
        self.current_run = {
            "name": name,
            "params": {},
            "metrics": {},
            "artifacts": []
        }

        print(f"\n🧪 Starting run: {name}")
        return self

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self.current_run:
            self.runs.append(self.current_run)
            print(f"   ✓ Run completed: {self.current_run['name']}")
            self.current_run = None

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log experiment parameters."""
        print(f"\n📝 Logging parameters:")
        for key, value in params.items():
            print(f"   - {key}: {value}")

        if self.current_run:
            self.current_run["params"].update(params)

    def log_metrics(self, metrics: Dict[str, float]) -> None:
        """Log experiment metrics."""
        print(f"\n📊 Logging metrics:")
        for key, value in metrics.items():
            print(f"   - {key}: {value:.4f}")

        if self.current_run:
            self.current_run["metrics"].update(metrics)

    def log_model(
        self,
        model: Any,
        path: str
    ) -> None:
        """Log trained model."""
        print(f"\n🤖 Logging model: {path}")
        print(f"   ✓ Model logged")

        if self.current_run:
            self.current_run["artifacts"].append(path)

    def compare_experiments(
        self,
        experiment_ids: List[str]
    ) -> Dict[str, Any]:
        """Compare multiple experiments."""
        print(f"\n📊 Comparing {len(experiment_ids)} experiments")

        comparison = {
            "experiments": experiment_ids,
            "metrics": {}
        }

        print(f"   ✓ Comparison generated")
        return comparison


class ModelRegistry:
    """Model registry for versioning."""

    def __init__(self, remote: str = "s3://models"):
        self.remote = remote
        self.models = {}

        print(f"📚 Model Registry")
        print(f"   Remote: {remote}")

    def register(
        self,
        name: str,
        version: str,
        model_path: str,
        metrics: Dict[str, float],
        stage: str = "development",
        tags: Optional[List[str]] = None
    ) -> None:
        """Register model version."""
        print(f"\n📚 Registering model")
        print(f"   Name: {name}")
        print(f"   Version: {version}")
        print(f"   Stage: {stage}")
        print(f"   Metrics:")
        for metric, value in metrics.items():
            print(f"   - {metric}: {value:.4f}")

        model_key = f"{name}/{version}"
        self.models[model_key] = {
            "path": model_path,
            "metrics": metrics,
            "stage": stage,
            "tags": tags or []
        }

        print(f"   ✓ Model registered: {model_key}")

    def load_model(
        self,
        name: str,
        version: str
    ) -> Any:
        """Load model from registry."""
        model_key = f"{name}/{version}"
        print(f"\n📚 Loading model: {model_key}")

        if model_key in self.models:
            model_info = self.models[model_key]
            print(f"   Path: {model_info['path']}")
            print(f"   Stage: {model_info['stage']}")
            print(f"   ✓ Model loaded")
            return model_info
        else:
            print(f"   ❌ Model not found")
            return None

    def transition(
        self,
        name: str,
        version: str,
        stage: str
    ) -> None:
        """Transition model to different stage."""
        model_key = f"{name}/{version}"
        print(f"\n🔄 Transitioning model: {model_key}")
        print(f"   New stage: {stage}")

        if model_key in self.models:
            self.models[model_key]["stage"] = stage
            print(f"   ✓ Model transitioned to {stage}")
        else:
            print(f"   ❌ Model not found")

    def list_models(
        self,
        name: Optional[str] = None,
        stage: Optional[str] = None
    ) -> List[str]:
        """List registered models."""
        print(f"\n📚 Listing models")

        if name:
            print(f"   Filter by name: {name}")
        if stage:
            print(f"   Filter by stage: {stage}")

        models = []
        for key, info in self.models.items():
            if name and not key.startswith(name):
                continue
            if stage and info["stage"] != stage:
                continue
            models.append(key)

        for model in models:
            print(f"   - {model} ({self.models[model]['stage']})")

        return models


def load_params(params_file: str = "params.yaml") -> Dict[str, Any]:
    """Load parameters from YAML file."""
    print(f"\n📋 Loading parameters: {params_file}")

    # Simulate loading params
    params = {
        "train": {
            "epochs": 100,
            "batch_size": 32,
            "learning_rate": 0.001
        },
        "model": {
            "architecture": "yolov8n",
            "pretrained": True
        }
    }

    print(f"   ✓ Parameters loaded")
    return params


class MetricsComparator:
    """Compare metrics across runs."""

    def __init__(self):
        print(f"📊 Metrics Comparator")

    def diff(
        self,
        baseline: str,
        comparison: str
    ) -> Dict[str, float]:
        """Compare metrics between versions."""
        print(f"\n📊 Comparing metrics")
        print(f"   Baseline: {baseline}")
        print(f"   Comparison: {comparison}")

        # Simulate metrics diff
        diff = {
            "accuracy": 0.02,
            "f1": 0.015,
            "precision": 0.018,
            "recall": 0.012
        }

        print(f"\n   Differences:")
        for metric, value in diff.items():
            sign = "+" if value >= 0 else ""
            print(f"   - {metric}: {sign}{value:.3f}")

        return diff


class LineageTracker:
    """Track data and model lineage."""

    def __init__(self):
        print(f"🔍 Lineage Tracker")

    def get_lineage(self, artifact: str) -> LineageInfo:
        """Get lineage for artifact."""
        print(f"\n🔍 Getting lineage: {artifact}")

        # Simulate lineage info
        lineage = LineageInfo(
            training_data="data/train_v2.csv",
            code_commit="abc123def",
            params={
                "epochs": 100,
                "batch_size": 32
            },
            created_at="2025-01-15T10:00:00Z"
        )

        print(f"   Training data: {lineage.training_data}")
        print(f"   Code commit: {lineage.code_commit}")
        print(f"   Created: {lineage.created_at}")

        return lineage

    def get_audit_trail(
        self,
        path: str
    ) -> List[Dict[str, str]]:
        """Get audit trail for file."""
        print(f"\n🔍 Getting audit trail: {path}")

        # Simulate audit trail
        trail = [
            {
                "timestamp": "2025-01-10T10:00:00Z",
                "message": "Initial version",
                "author": "user@example.com"
            },
            {
                "timestamp": "2025-01-12T14:30:00Z",
                "message": "Add 10k samples",
                "author": "user@example.com"
            },
            {
                "timestamp": "2025-01-15T09:00:00Z",
                "message": "Fix label errors",
                "author": "user@example.com"
            }
        ]

        for entry in trail:
            print(f"   [{entry['timestamp']}] {entry['message']} by {entry['author']}")

        return trail


class RemoteStorage:
    """Remote storage configuration."""

    def __init__(self, backend: str = "s3"):
        self.backend = backend
        print(f"☁️  Remote Storage: {backend}")

    def configure_s3(
        self,
        bucket: str,
        region: str = "us-west-2"
    ) -> None:
        """Configure S3 remote."""
        print(f"\n☁️  Configuring S3")
        print(f"   Bucket: {bucket}")
        print(f"   Region: {region}")
        print(f"   $ dvc remote add -d s3remote s3://{bucket}/dvc-storage")
        print(f"   $ dvc remote modify s3remote region {region}")
        print(f"   ✓ S3 configured")

    def configure_gcs(
        self,
        bucket: str,
        project: str
    ) -> None:
        """Configure GCS remote."""
        print(f"\n☁️  Configuring GCS")
        print(f"   Bucket: {bucket}")
        print(f"   Project: {project}")
        print(f"   $ dvc remote add -d gcsremote gs://{bucket}/dvc-storage")
        print(f"   $ dvc remote modify gcsremote projectname {project}")
        print(f"   ✓ GCS configured")

    def configure_azure(
        self,
        container: str,
        account: str
    ) -> None:
        """Configure Azure Blob Storage."""
        print(f"\n☁️  Configuring Azure Blob Storage")
        print(f"   Container: {container}")
        print(f"   Account: {account}")
        print(f"   $ dvc remote add -d azureremote azure://{container}/path")
        print(f"   $ dvc remote modify azureremote account_name {account}")
        print(f"   ✓ Azure configured")


def demo():
    """Demonstrate data versioning."""
    print("=" * 70)
    print("Data Versioning & Pipeline Management Demo")
    print("=" * 70)

    # DVC Tracker
    print(f"\n{'='*70}")
    print("DVC Tracking")
    print(f"{'='*70}")

    tracker = DVCTracker(
        project_dir=".",
        remote="storage"
    )

    # Track data
    tracker.track_data(
        path="data/training.csv",
        message="Add training dataset v1.0"
    )

    # Track model
    tracker.track_model(
        path="models/yolov8n.pt",
        metrics={"accuracy": 0.95, "f1": 0.93},
        message="Train YOLOv8n model"
    )

    # Push to remote
    tracker.push()

    # Pipeline Management
    print(f"\n{'='*70}")
    print("Pipeline Management")
    print(f"{'='*70}")

    pipeline = PipelineManager(pipeline_file="dvc.yaml")

    # Show DAG
    pipeline.show_dag()

    # Run pipeline
    results = pipeline.run()

    # Run specific stage
    pipeline.run_stage("train")

    # Reproduce
    pipeline.reproduce()

    # Experiment Tracking
    print(f"\n{'='*70}")
    print("Experiment Tracking")
    print(f"{'='*70}")

    exp_tracker = ExperimentTracker()

    # Run experiment
    with exp_tracker.start_run(name="yolov8_training"):
        exp_tracker.log_params({
            "model": "yolov8n",
            "epochs": 100,
            "batch_size": 32,
            "learning_rate": 0.001
        })

        exp_tracker.log_metrics({
            "accuracy": 0.95,
            "precision": 0.94,
            "recall": 0.93,
            "f1": 0.935
        })

        exp_tracker.log_model(
            model=None,
            path="models/yolov8n.pt"
        )

    # Compare experiments
    comparison = exp_tracker.compare_experiments(
        experiment_ids=["exp1", "exp2", "exp3"]
    )

    # Model Registry
    print(f"\n{'='*70}")
    print("Model Registry")
    print(f"{'='*70}")

    registry = ModelRegistry(remote="s3://models")

    # Register models
    registry.register(
        name="yolov8_production",
        version="1.0.0",
        model_path="models/yolov8n_v1.pt",
        metrics={"accuracy": 0.95, "f1": 0.93},
        stage="production",
        tags=["production", "v1.0"]
    )

    registry.register(
        name="yolov8_production",
        version="1.1.0",
        model_path="models/yolov8n_v1.1.pt",
        metrics={"accuracy": 0.96, "f1": 0.94},
        stage="staging",
        tags=["staging", "v1.1"]
    )

    # Load model
    model = registry.load_model(
        name="yolov8_production",
        version="1.0.0"
    )

    # Transition model
    registry.transition(
        name="yolov8_production",
        version="1.1.0",
        stage="production"
    )

    # List models
    models = registry.list_models(
        name="yolov8_production",
        stage="production"
    )

    # Parameters
    print(f"\n{'='*70}")
    print("Parameters Management")
    print(f"{'='*70}")

    params = load_params("params.yaml")

    print(f"\n   Train params:")
    print(f"   - epochs: {params['train']['epochs']}")
    print(f"   - batch_size: {params['train']['batch_size']}")
    print(f"   - learning_rate: {params['train']['learning_rate']}")

    # Metrics Comparison
    print(f"\n{'='*70}")
    print("Metrics Comparison")
    print(f"{'='*70}")

    comparator = MetricsComparator()

    diff = comparator.diff(
        baseline="HEAD~1",
        comparison="HEAD"
    )

    # Lineage Tracking
    print(f"\n{'='*70}")
    print("Lineage Tracking")
    print(f"{'='*70}")

    lineage_tracker = LineageTracker()

    lineage = lineage_tracker.get_lineage(
        artifact="models/yolov8n.pt"
    )

    audit_trail = lineage_tracker.get_audit_trail(
        path="data/training.csv"
    )

    # Remote Storage
    print(f"\n{'='*70}")
    print("Remote Storage Configuration")
    print(f"{'='*70}")

    storage = RemoteStorage(backend="s3")

    storage.configure_s3(
        bucket="my-ml-bucket",
        region="us-west-2"
    )

    storage.configure_gcs(
        bucket="my-ml-bucket",
        project="my-project"
    )

    storage.configure_azure(
        container="mldata",
        account="myaccount"
    )

    # Version Control Workflow
    print(f"\n{'='*70}")
    print("Version Control Workflow")
    print(f"{'='*70}")

    # Version 1
    tracker.track_data("data/v1/train.csv", message="Initial dataset")

    # Version 2
    tracker.track_data("data/v2/train.csv", message="Add 10k samples")

    # Version 3
    tracker.track_data("data/v3/train.csv", message="Fix label errors")

    # Checkout version
    tracker.checkout_version(tag="v1.0")

    print(f"\n{'='*70}")
    print("✓ Data Versioning Demo Complete")
    print(f"{'='*70}")


if __name__ == "__main__":
    demo()
